# backend/api/forecasting.py
from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.api.auth import get_current_user
from backend.database.db import get_db
from backend.database.models import Dataset, User
from backend.database.storage import get_bytes

from backend.app.forecasting.execution import run_forecast
from backend.api.models import ForecastRunRequest, ForecastRunResponse, ForecastPlanRequest, ForecastPlanResponse, ForecastSignalsRequest, ForecastSignalsResponse, ForecastTargetIn, ForecastRunOneResult

router = APIRouter()


def _get_owned_dataset_or_404(db: Session, dataset_id: str, user_id: str) -> Dataset:
    row = (
        db.query(Dataset)
        .filter(Dataset.dataset_id == dataset_id, Dataset.user_id == user_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return row


def load_dataset_as_dataframe(db: Session, dataset_id: str, user_id: str, *, version: Literal["raw", "current"] = "current",
) -> pd.DataFrame:
    ds = _get_owned_dataset_or_404(db, dataset_id, user_id)

    if version == "raw":
        key = getattr(ds, "raw_parquet_key", None)
        if not key:
            raise HTTPException(status_code=500, detail="raw_parquet_key is missing (migration needed)")
    else:
        key = ds.current_parquet_key

    try:
        parquet_bytes = get_bytes(key)
        return pd.read_parquet(io.BytesIO(parquet_bytes))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read parquet: {e}")


def _json_safe_records(df: pd.DataFrame, limit: int = 200) -> List[Dict[str, Any]]:
    safe = df.where(pd.notnull(df), None).copy()

    for c in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[c]):
            safe[c] = safe[c].apply(lambda x: x.isoformat() if x is not None else None)

    out: List[Dict[str, Any]] = []
    for row in safe.head(limit).to_dict(orient="records"):
        cleaned: Dict[str, Any] = {}
        for k, v in row.items():
            # numpy scalar -> python scalar
            if hasattr(v, "item") and callable(getattr(v, "item")):
                try:
                    cleaned[k] = v.item()
                except Exception:
                    cleaned[k] = v
            else:
                cleaned[k] = v
        out.append(cleaned)
    return out


def _datetime_parse_success_ratio(s: pd.Series) -> float:
    if s.empty:
        return 0.0
    parsed = pd.to_datetime(s, errors="coerce", utc=False)
    return float(parsed.notna().mean())


def _letters_ratio(s: pd.Series) -> float:
    if s.empty:
        return 0.0
    ss = s.astype(str)
    letters = ss.str.contains(r"[A-Za-zА-Яа-яƏəÖöÜüĞğÇçŞşıİ]", regex=True, na=False)
    return float(letters.mean())


def _infer_freq_label(dt_index: pd.DatetimeIndex) -> str:
    try:
        f = pd.infer_freq(dt_index)
    except Exception:
        f = None
    if not f:
        return "irregular"

    f = f.upper()
    if f.startswith("D"):
        return "daily"
    if f.startswith("W"):
        return "weekly"
    if f.startswith("M"):
        return "monthly"
    if f.startswith("Q"):
        return "quarterly"
    if f.startswith("A") or f.startswith("Y"):
        return "yearly"
    return "irregular"


_ID_RE = re.compile(r"(^|[^a-z0-9])id([^a-z0-9]|$)", re.IGNORECASE)

def _is_id_like(col: str) -> bool:
    c = col.strip().lower()

    if c in {"index", "row"}:
        return True
    if c.startswith("unnamed"):
        return True

    if _ID_RE.search(c):
        return True

    if "uuid" in c or "guid" in c:
        return True

    return False


def _numeric_target_candidates(df: pd.DataFrame) -> List[str]:
    numeric = df.select_dtypes(include=[np.number]).copy()

    cands: List[str] = []
    for col in numeric.columns:
        s = numeric[col]
        nunique = int(s.nunique(dropna=True))
        notna = float(s.notna().mean())
        is_id = _is_id_like(col)

        if is_id:
            print(" -> skip id_like")
            continue
        if nunique <= 1:
            print(" -> skip constant")
            continue
        if notna < 0.2:
            print(" -> skip sparse")
            continue

        cands.append(col)

    return cands


def _grouping_candidates(df: pd.DataFrame, max_cardinality: int = 50) -> List[str]:
    cands: List[str] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        nunique = int(df[col].nunique(dropna=True))
        if 2 <= nunique <= max_cardinality:
            cands.append(col)
    return cands


def _to_agent_signals(*, df: pd.DataFrame, signals: ForecastSignalsResponse, ) -> Dict[str, Any]:

    dt_candidates = []
    for c in signals.datetime_candidates or []:
        dt_candidates.append(
            {
                "column": c.get("column"),
                "success_ratio": float(c.get("success_ratio", 0.0) or 0.0),
                "letters_ratio": float(c.get("letters_ratio", 0.0) or 0.0),
            }
        )

    tgt_candidates = [{"column": col, "score": 1.0} for col in (signals.numeric_target_candidates or [])]

    grp_candidates = []
    for col in signals.grouping_candidates or []:
        try:
            card = int(df[col].nunique(dropna=True))
        except Exception:
            card = None
        grp_candidates.append({"column": col, "cardinality": card, "score": 1.0})

    reason = ""
    feasible = bool(signals.feasible)
    if not feasible:
        if not dt_candidates:
            reason = "No datetime candidates."
        elif not tgt_candidates:
            reason = "No numeric target candidates."
        else:
            reason = "Forecast not feasible by heuristic."

    return {
        "forecast_feasible": feasible,
        "reason_if_not_feasible": reason,
        "datetime_candidates": dt_candidates,
        "target_candidates": tgt_candidates,
        "grouping_candidates": grp_candidates,
        "inferred_frequency": signals.inferred_frequency or "unknown",
    }


def _try_llm_planning(
    *,
    dataset_id: str,
    df: pd.DataFrame,
    df_head: pd.DataFrame,
    signals: ForecastSignalsResponse,
    profile: Optional[Dict[str, Any]],
    user_intent: Optional[str],
    max_targets: int,
    head_rows: int,
    horizon: int,
) -> ForecastPlanResponse:

    try:
        from backend.app.forecasting.planning_agent import ForecastPlanningAgent
    except Exception as e:
        # if import fails, do fallback
        fb = _fallback_plan(dataset_id=dataset_id, df=df, signals=signals, max_targets=max_targets)
        fb.reasoning = f"Fallback: could not import ForecastPlanningAgent: {e}"
        fb.reasons = [fb.reasoning]
        fb.planner_source = "fallback"
        fb.llm_model = None
        return fb

    agent_signals = _to_agent_signals(df=df, signals=signals)

    report: Dict[str, Any] = {}
    if isinstance(profile, dict) and profile:
        report = {"post_profile": profile}

    head_rows = max(1, min(int(head_rows), 50))

    try:
        agent = ForecastPlanningAgent()
        plan_obj = agent.create_plan(
            dataset_id=dataset_id,
            df=df,
            report=report,
            forecast_signals=agent_signals,
            user_intent=user_intent,
            max_targets=max_targets,
            head_rows=head_rows,
            horizon=int(horizon)
        )

        plan_dict = plan_obj.model_dump() if hasattr(plan_obj, "model_dump") else dict(plan_obj)  # type: ignore

        targets_in = []
        for t in plan_dict.get("targets") or []:
            targets_in.append(ForecastTargetIn(column=t.get("column"), horizon=int(t.get("horizon", 30) or 30)))

        resp = ForecastPlanResponse(
            dataset_id=dataset_id,
            suitable=bool(plan_dict.get("suitable")),
            mode=plan_dict.get("mode") or "skipped",
            datetime_column=plan_dict.get("datetime_column"),
            inferred_frequency=plan_dict.get("inferred_frequency") or signals.inferred_frequency,
            group_by=plan_dict.get("group_by"),
            targets=targets_in,
            planner_source="llm",
            llm_model=getattr(agent.llm, "model", None) or getattr(agent.llm, "model_name", None),
            reasoning=plan_dict.get("reasoning") or "",
            reasons=[],
        )

        if not resp.suitable:
            resp.mode = "skipped"
            resp.datetime_column = None
            resp.targets = []
            resp.group_by = None
        if resp.mode == "grouped" and not resp.group_by:
            resp.mode = "overall"

        return resp

    except Exception as e:
        fb = _fallback_plan(dataset_id=dataset_id, df=df, signals=signals, max_targets=max_targets)
        fb.planner_source = "fallback"
        fb.llm_model = None
        fb.reasoning = f"Fallback: LLM planning failed: {e}"
        fb.reasons = [fb.reasoning]
        return fb


def _fallback_plan(
    *,
    dataset_id: str,
    df: pd.DataFrame,
    signals: ForecastSignalsResponse,
    max_targets: int,
    horizon: int,
) -> ForecastPlanResponse:

    reasons: List[str] = []

    if not signals.datetime_candidates:
        reasons.append("No datetime column candidates found. Forecasting skipped.")
        return ForecastPlanResponse(
            dataset_id=dataset_id,
            suitable=False,
            mode="skipped",
            planner_source="fallback",
            llm_model=None,
            reasoning="; ".join(reasons),
            reasons=reasons,
        )

    best = sorted(
        signals.datetime_candidates,
        key=lambda x: (x.get("success_ratio", 0.0), -x.get("letters_ratio", 1.0)),
        reverse=True,
    )[0]
    dt_col = best.get("column")

    if float(best.get("success_ratio", 0.0) or 0.0) < 0.8:
        reasons.append(f"Best datetime candidate '{dt_col}' has low parse success ratio. Forecasting skipped.")
        return ForecastPlanResponse(
            dataset_id=dataset_id,
            suitable=False,
            mode="skipped",
            datetime_column=dt_col,
            inferred_frequency=signals.inferred_frequency,
            planner_source="fallback",
            llm_model=None,
            reasoning="; ".join(reasons),
            reasons=reasons,
        )

    if not signals.numeric_target_candidates:
        reasons.append("No numeric target candidates found. Forecasting skipped.")
        return ForecastPlanResponse(
            dataset_id=dataset_id,
            suitable=False,
            mode="skipped",
            datetime_column=dt_col,
            inferred_frequency=signals.inferred_frequency,
            planner_source="fallback",
            llm_model=None,
            reasoning="; ".join(reasons),
            reasons=reasons,
        )

    targets = [
        ForecastTargetIn(column=c, horizon=int(horizon))
        for c in signals.numeric_target_candidates[:max_targets]
    ]

    mode: Literal["overall", "grouped"] = "overall"
    group_by: Optional[str] = None

    if signals.grouping_candidates:
        g = signals.grouping_candidates[0]
        try:
            card = int(df[g].nunique(dropna=True))
        except Exception:
            card = 999999
        if 2 <= card <= 30:
            group_by = g
            mode = "grouped"

    return ForecastPlanResponse(
        dataset_id=dataset_id,
        suitable=True,
        mode=mode,
        datetime_column=dt_col,
        inferred_frequency=signals.inferred_frequency,
        group_by=group_by,
        targets=targets,
        planner_source="fallback",
        llm_model=None,
        reasoning="Fallback plan (deterministic).",
        reasons=[],
    )


@router.post("/forecast/signals", response_model=ForecastSignalsResponse)
def forecast_signals(
    req: ForecastSignalsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ForecastSignalsResponse:
    df = load_dataset_as_dataframe(db, req.dataset_id, current_user.user_id, version=req.version)

    candidates: List[Dict[str, Any]] = []
    for col in df.columns:
        s = df[col]

        if pd.api.types.is_datetime64_any_dtype(s):
            candidates.append({"column": col, "success_ratio": 1.0, "letters_ratio": 0.0})
            continue

        if pd.api.types.is_numeric_dtype(s) and ("date" not in col.lower() and "time" not in col.lower()):
            continue

        success = _datetime_parse_success_ratio(s)
        if success >= 0.8:
            candidates.append(
                {
                    "column": col,
                    "success_ratio": round(success, 4),
                    "letters_ratio": round(_letters_ratio(s), 4),
                }
            )

    inferred_frequency = "irregular"
    if candidates:
        best = sorted(candidates, key=lambda x: x["success_ratio"], reverse=True)[0]
        dt = pd.to_datetime(df[best["column"]], errors="coerce", utc=False).dropna()
        if len(dt) >= 10:
            inferred_frequency = _infer_freq_label(pd.DatetimeIndex(dt).sort_values())

    numeric_targets = _numeric_target_candidates(df)
    groupings = _grouping_candidates(df, max_cardinality=50)
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
    df = load_dataset_as_dataframe(db, req.dataset_id, current_user.user_id, version=req.version)

    head_n = max(1, min(int(req.head_rows), 50))
    df_head = df.head(head_n)

    max_targets = max(1, min(int(req.max_targets), 10))


    return _try_llm_planning(
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
def forecast_run( req: ForecastRunRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
) -> ForecastRunResponse:
    if req.plan.mode == "skipped" or not req.plan.suitable:
        return ForecastRunResponse(dataset_id=req.dataset_id, results=[])

    if not req.plan.datetime_column:
        raise HTTPException(status_code=400, detail="plan.datetime_column is required")

    if not req.plan.targets:
        raise HTTPException(status_code=400, detail="plan.targets must be non-empty")
    df = load_dataset_as_dataframe(db, req.dataset_id, current_user.user_id, version=req.version)

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
        raise HTTPException(status_code=500, detail=f"Forecast execution failed: {e}")

    preview_n = max(1, min(int(req.preview_rows), 500))

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
                preview=_json_safe_records(r.forecast_df, limit=preview_n),
                meta=r.meta,
            )
        )

    return ForecastRunResponse(dataset_id=req.dataset_id, results=out)