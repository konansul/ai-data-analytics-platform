# backend/api/forecasting.py
from __future__ import annotations

import io
import base64
import json
import pandas as pd

from typing import Any, Dict, List
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.api.auth import get_current_user
from backend.database.db import get_db

from backend.database.storage import put_bytes, to_jsonable, new_id, forecast_prefix
from backend.database.models import User, ForecastRun, Artifact

from backend.app.forecasting.execution import run_forecast
from backend.api.models import ForecastRunRequest, ForecastRunResponse, ForecastPlanRequest, ForecastPlanResponse, ForecastSignalsRequest, ForecastSignalsResponse, ForecastRunOneResult

from backend.app.forecasting.helpers import numeric_target_candidates, try_llm_planning, grouping_candidates, infer_freq_label, letters_ratio, datetime_parse_success_ratio, fallback_plan, to_agent_signals
from backend.api.helpers.ownership import get_owned_dataset_or_404, get_owned_clean_run_or_404, get_owned_forecast_run_or_404, get_owned_visualization_run_or_404
from backend.api.helpers.datasets import load_dataset_df
from backend.api.helpers.artifacts import add_artifact
from backend.api.helpers.json_utils import json_safe_records

router = APIRouter()


@router.post("/forecast/signals", response_model=ForecastSignalsResponse)
def forecast_signals(
    req: ForecastSignalsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ForecastSignalsResponse:

    df = load_dataset_df(
        db=db,
        dataset_id=req.dataset_id,
        user_id=current_user.user_id,
        version=req.version,
    )

    candidates: List[Dict[str, Any]] = []
    for col in df.columns:
        s = df[col]

        if pd.api.types.is_datetime64_any_dtype(s):
            candidates.append({"column": col, "success_ratio": 1.0, "letters_ratio": 0.0})
            continue

        if pd.api.types.is_numeric_dtype(s) and ("date" not in col.lower() and "time" not in col.lower()):
            continue

        success = datetime_parse_success_ratio(s)
        if success >= 0.8:
            candidates.append(
                {
                    "column": col,
                    "success_ratio": round(success, 4),
                    "letters_ratio": round(letters_ratio(s), 4),
                }
            )

    inferred_frequency = "irregular"
    if candidates:
        best = sorted(candidates, key=lambda x: x["success_ratio"], reverse=True)[0]
        dt = pd.to_datetime(df[best["column"]], errors="coerce", utc=False).dropna()
        if len(dt) >= 10:
            inferred_frequency = infer_freq_label(pd.DatetimeIndex(dt).sort_values())

    numeric_targets = numeric_target_candidates(df)
    groupings = grouping_candidates(df, max_cardinality=50)
    feasible = bool(candidates) and bool(numeric_targets)

    return ForecastSignalsResponse(
        dataset_id=req.dataset_id,
        datetime_candidates=candidates,
        inferred_frequency=inferred_frequency,
        numeric_target_candidates=numeric_targets,
        grouping_candidates=groupings,
        feasible=feasible,
    )


@router.post("/forecast/plan", response_model=ForecastPlanResponse)
def forecast_plan(req: ForecastPlanRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
) -> ForecastPlanResponse:

    df = load_dataset_df(
        db=db,
        dataset_id=req.dataset_id,
        user_id=current_user.user_id,
        version=req.version,
    )

    head_n = max(1, min(int(req.head_rows), 50))
    df_head = df.head(head_n)

    max_targets = max(1, min(int(req.max_targets), 10))


    return try_llm_planning(
        dataset_id=req.dataset_id,
        df=df,
        df_head=df_head,
        signals=req.signals,
        profile=req.profile,
        user_intent=req.user_intent,
        max_targets=max_targets,
        head_rows=head_n,
        horizon=int(req.horizon),  # NEW
    )


@router.post("/forecast/run", response_model=ForecastRunResponse)
def forecast_run(
    req: ForecastRunRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ForecastRunResponse:
    if req.plan.mode == "skipped" or not req.plan.suitable:
        return ForecastRunResponse(
            dataset_id=req.dataset_id,
            run_id=req.run_id,
            forecast_run_id="",
            results=[],
        )

    if not req.plan.datetime_column:
        raise HTTPException(status_code=400, detail="plan.datetime_column is required")
    if not req.plan.targets:
        raise HTTPException(status_code=400, detail="plan.targets must be non-empty")

    ds = get_owned_dataset_or_404(db, req.dataset_id, current_user.user_id)
    get_owned_clean_run_or_404(db, req.run_id, current_user.user_id)

    df = load_dataset_df(
        db=db,
        dataset_id=req.dataset_id,
        user_id=current_user.user_id,
        version=req.version,
    )

    forecast_run_id = new_id("frun")
    plan_json = req.plan.model_dump() if hasattr(req.plan, "model_dump") else req.plan.dict()

    row = ForecastRun(
        forecast_run_id=forecast_run_id,
        user_id=current_user.user_id,
        dataset_id=ds.dataset_id,
        run_id=req.run_id,
        status="running",
        error=None,
        model=req.model or "auto",
        horizon=int(req.horizon),
        datetime_column=req.plan.datetime_column,
        targets=[t.column for t in (req.plan.targets or [])],
        plan_json=to_jsonable(plan_json),
        result_json={},
        forecast_parquet_key=None,
        forecast_json_key=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(row)

    ds.forecast_status = "running"
    ds.last_forecast_run_id = forecast_run_id
    db.commit()

    override_h = int(req.horizon)

    class _ExecTarget:
        def __init__(self, column: str, horizon: int):
            self.column = column
            self.horizon = horizon

    class _ExecPlan:
        def __init__(self, p: ForecastPlanResponse, horizon_override: int):
            self.suitable = p.suitable
            self.mode = p.mode
            self.datetime_column = p.datetime_column
            self.group_by = p.group_by
            self.inferred_frequency = p.inferred_frequency
            self.targets = [_ExecTarget(t.column, horizon_override) for t in p.targets]

    exec_plan = _ExecPlan(req.plan, override_h)

    try:
        results = run_forecast(
            dataset_id=req.dataset_id,
            df=df,
            plan=exec_plan,
            model=req.model,
        )
    except Exception as e:
        row.status = "failed"
        row.error = f"{type(e).__name__}: {e}"
        row.updated_at = datetime.now(timezone.utc)
        ds.forecast_status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Forecast execution failed: {e}")

    prefix = forecast_prefix(current_user.user_id, req.run_id, forecast_run_id)
    result_json_key = f"{prefix}/result.json"
    wide_parquet_key = f"{prefix}/forecast_wide.parquet"
    wide_csv_key = f"{prefix}/forecast_wide.csv"

    wide = None
    safe_results_meta: List[Dict[str, Any]] = []

    try:
        for r in results:
            safe_results_meta.append(
                to_jsonable({
                    "target": r.target,
                    "mode": r.mode,
                    "model_used": r.model_used,
                    "datetime_column": r.datetime_column,
                    "group_by": r.group_by,
                    "horizon": r.horizon,
                    "frequency": r.frequency,
                    "meta": r.meta,
                })
            )

            fdf = r.forecast_df.copy()

            dt_col = r.datetime_column
            if dt_col not in fdf.columns:
                if "dt" in fdf.columns:
                    dt_col = "dt"
                elif "ds" in fdf.columns:
                    dt_col = "ds"
                else:
                    continue

            if "yhat" not in fdf.columns:
                continue

            part = fdf[[dt_col, "yhat"]].rename(columns={dt_col: "dt", "yhat": r.target})
            part["dt"] = pd.to_datetime(part["dt"], errors="coerce")
            part = part.dropna(subset=["dt"]).sort_values("dt")

            if wide is None:
                wide = part
            else:
                wide = wide.merge(part, on="dt", how="outer")

        if wide is None:
            wide = pd.DataFrame()

        payload = to_jsonable({
            "forecast_run_id": forecast_run_id,
            "run_id": req.run_id,
            "dataset_id": req.dataset_id,
            "user_id": current_user.user_id,
            "version": req.version,
            "model": req.model,
            "horizon": int(req.horizon),
            "plan": plan_json,
            "results_meta": safe_results_meta,
        })

        put_bytes(result_json_key, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))

        bufp = io.BytesIO()
        wide.to_parquet(bufp, index=False)
        put_bytes(wide_parquet_key, bufp.getvalue())

        put_bytes(wide_csv_key, wide.to_csv(index=False).encode("utf-8"))

    except Exception as e:
        row.status = "failed"
        row.error = f"Persist failed: {type(e).__name__}: {e}"
        row.updated_at = datetime.now(timezone.utc)
        ds.forecast_status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to persist forecast artifacts: {e}")

    try:
        add_artifact(
            db,
            user_id=current_user.user_id,
            dataset_id=ds.dataset_id,
            run_type="forecast",
            run_id=forecast_run_id,
            kind="forecast_result_json",
            mime_type="application/json",
            storage_key=result_json_key,
        )
        add_artifact(
            db,
            user_id=current_user.user_id,
            dataset_id=ds.dataset_id,
            run_type="forecast",
            run_id=forecast_run_id,
            kind="forecast_wide_parquet",
            mime_type="application/octet-stream",
            storage_key=wide_parquet_key,
        )
        add_artifact(
            db,
            user_id=current_user.user_id,
            dataset_id=ds.dataset_id,
            run_type="forecast",
            run_id=forecast_run_id,
            kind="forecast_wide_csv",
            mime_type="text/csv",
            storage_key=wide_csv_key,
        )

        row.status = "done"
        row.updated_at = datetime.now(timezone.utc)
        row.forecast_json_key = result_json_key
        row.forecast_parquet_key = wide_parquet_key
        row.result_json = to_jsonable({
            "result_json_key": result_json_key,
            "wide_parquet_key": wide_parquet_key,
            "wide_csv_key": wide_csv_key,
        })

        ds.forecast_status = "done"
        ds.last_forecast_run_id = forecast_run_id
        ds.forecasted_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as e:
        row.status = "failed"
        row.error = f"Artifact registry failed: {type(e).__name__}: {e}"
        row.updated_at = datetime.now(timezone.utc)
        ds.forecast_status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to register forecast artifacts: {e}")

    preview_n = max(1, min(int(req.preview_rows or 50), 500))

    out: List[ForecastRunOneResult] = []
    for r in results:
        out.append(
            ForecastRunOneResult(
                target=r.target,
                mode=r.mode,
                model_used=r.model_used,
                datetime_column=r.datetime_column,
                group_by=r.group_by,
                horizon=r.horizon,
                frequency=r.frequency,
                preview= json_safe_records(r.forecast_df, limit=preview_n),
                meta=r.meta,
            )
        )

    return ForecastRunResponse(
        dataset_id=req.dataset_id,
        run_id=req.run_id,
        forecast_run_id=forecast_run_id,
        results=out,
    )


@router.post("/forecast/plots")
def save_forecast_plot(
    payload: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    forecast_run_id = payload["forecast_run_id"]
    dataset_id = payload["dataset_id"]
    target = payload.get("target", "unknown")
    png_b64 = payload["png_base64"]

    fr = get_owned_forecast_run_or_404(db, forecast_run_id, current_user.user_id)

    if not fr.run_id:
        raise HTTPException(status_code=400, detail="ForecastRun.run_id is missing (cannot place plot under runs/<run_id>)")

    try:
        png = base64.b64decode(png_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid png_base64")

    prefix = forecast_prefix(current_user.user_id, fr.run_id, forecast_run_id)
    key = f"{prefix}/plots/{target}.png"

    put_bytes(key, png, content_type="image/png")

    artifact = Artifact(
        artifact_id=new_id("art"),
        user_id=current_user.user_id,
        dataset_id=dataset_id,
        run_type="forecast",
        run_id=forecast_run_id,
        kind="forecast_plot_png",
        mime_type="image/png",
        bucket="local",
        storage_key=key,
        meta={"target": target, "run_id": fr.run_id},
        created_at=datetime.now(timezone.utc)
    )

    db.add(artifact)
    db.commit()

    return {"ok": True, "storage_key": key}