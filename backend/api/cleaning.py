# backend/api/cleaning.py
from __future__ import annotations

import io
import json
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.api.auth import get_current_user
from backend.api.models import CleaningRunRequest, CleaningRunResponse

from backend.app.cleaning.cleaning_steps.main_pipeline import run_cleaning_pipeline

from backend.database.storage import get_bytes, put_bytes, delete_key, to_jsonable, new_id, run_prefix
from backend.database.db import get_db
from backend.database.models import Dataset, CleaningRun, User

from backend.api.helpers.ownership import get_owned_dataset_or_404

from fastapi.responses import StreamingResponse
router = APIRouter()

LOCAL_BUCKET_NAME = "local"

@router.get("/cleaning/runs")
def list_my_runs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    rows = (
        db.query(CleaningRun, Dataset)
        .join(Dataset, Dataset.dataset_id == CleaningRun.dataset_id)
        .filter(CleaningRun.user_id == current_user.user_id)
        .order_by(CleaningRun.created_at.desc())
        .limit(200)
        .all()
    )

    runs: List[Dict[str, Any]] = []
    for r, ds in rows:
        runs.append({
            "run_id": r.run_id,
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "dataset_id": r.dataset_id,
            "file_name": ds.file_name,
            "sheet_name": ds.sheet_name,
            "original_dataset_id": ds.original_dataset_id,

            "has_report": bool(r.report_key),
            "has_cleaned_xlsx": bool(r.cleaned_xlsx_key),
        })

    return {"runs": runs}


@router.post("/cleaning/runs", response_model=CleaningRunResponse)
def run_cleaning(
    req: CleaningRunRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    ds = get_owned_dataset_or_404(db, req.dataset_id, current_user.user_id)

    try:
        parquet_bytes = get_bytes(ds.current_parquet_key)
        df = pd.read_parquet(io.BytesIO(parquet_bytes))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset parquet: {e}")

    kwargs: Dict[str, Any] = {
        "use_llm": bool(req.use_llm),
        "llm_model": req.llm_model,
    }
    if req.missing_threshold is not None:
        kwargs["missing_threshold"] = float(req.missing_threshold)
    if req.impute is not None:
        kwargs["impute"] = bool(req.impute)
    if req.numeric_strategy is not None:
        kwargs["numeric_strategy"] = req.numeric_strategy
    if req.categorical_strategy is not None:
        kwargs["categorical_strategy"] = req.categorical_strategy
    if req.datetime_strategy is not None:
        kwargs["datetime_strategy"] = req.datetime_strategy
    if req.fill_value is not None:
        kwargs["fill_value"] = req.fill_value
    if req.datetime_success_ratio is not None:
        kwargs["datetime_success_ratio"] = float(req.datetime_success_ratio)
    if req.categorical_numeric_max_unique is not None:
        kwargs["categorical_numeric_max_unique"] = int(req.categorical_numeric_max_unique)

    run_id = new_id("run")
    run_row = CleaningRun(
        run_id=run_id,
        user_id=current_user.user_id,
        dataset_id=ds.dataset_id,
        status="running",
        bucket=LOCAL_BUCKET_NAME,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(run_row)
    db.commit()

    try:
        clean_df, report = run_cleaning_pipeline(df, **kwargs)
    except Exception as e:
        run_row.status = "failed"
        run_row.error = str(e)
        run_row.updated_at = datetime.now(timezone.utc)
        db.commit()

        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Cleaning pipeline failed: {e}",
                "type": type(e).__name__,
                "traceback": traceback.format_exc(),
            },
        )

    prefix = run_prefix(current_user.user_id, run_id)
    report_key = f"{prefix}/report.json"
    cleaned_parquet_key = f"{prefix}/cleaned.parquet"
    cleaned_xlsx_key = f"{prefix}/cleaned.xlsx"

    try:
        buf = io.BytesIO()
        clean_df.to_parquet(buf, index=False)
        put_bytes(cleaned_parquet_key, buf.getvalue())

        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            clean_df.to_excel(writer, index=False, sheet_name="data")
        out.seek(0)
        put_bytes(cleaned_xlsx_key, out.getvalue())

        safe_report = to_jsonable(report)
        put_bytes(report_key, json.dumps(safe_report, ensure_ascii=False, indent=2).encode("utf-8"))

    except Exception as e:
        run_row.status = "failed"
        run_row.error = f"Persist failed: {e}"
        run_row.updated_at = datetime.now(timezone.utc)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to persist artifacts: {e}")

    # 7) обновляем dataset current.parquet (как у тебя было)
    try:
        buf2 = io.BytesIO()
        clean_df.to_parquet(buf2, index=False)
        put_bytes(ds.current_parquet_key, buf2.getvalue())
    except Exception:
        pass

    run_row.status = "done"
    run_row.report_key = report_key
    run_row.cleaned_parquet_key = cleaned_parquet_key
    run_row.cleaned_xlsx_key = cleaned_xlsx_key
    run_row.updated_at = datetime.now(timezone.utc)
    db.commit()

    return {"run_id": run_id}


@router.get("/cleaning/runs/{run_id}")
def get_run_status(
    run_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    row: Optional[CleaningRun] = (
        db.query(CleaningRun)
        .filter(CleaningRun.run_id == run_id, CleaningRun.user_id == current_user.user_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")

    return {
        "run_id": row.run_id,
        "status": row.status,
        "has_report": bool(row.report_key),
        "has_cleaned_xlsx": bool(row.cleaned_xlsx_key),
        "error": row.error,
    }


@router.get("/cleaning/runs/{run_id}/report")
def get_run_report(
    run_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    row: Optional[CleaningRun] = (
        db.query(CleaningRun)
        .filter(CleaningRun.run_id == run_id, CleaningRun.user_id == current_user.user_id)
        .first()
    )
    if not row or not row.report_key:
        raise HTTPException(status_code=404, detail="Report not found")

    try:
        raw = get_bytes(row.report_key)
        import json
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read report: {e}")

@router.get("/cleaning/runs/{run_id}/artifacts/{name}")
def download_run_artifact(
    run_id: str,
    name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row: Optional[CleaningRun] = (
        db.query(CleaningRun)
        .filter(CleaningRun.run_id == run_id, CleaningRun.user_id == current_user.user_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")

    if name == "cleaned.xlsx":
        if not row.cleaned_xlsx_key:
            raise HTTPException(status_code=404, detail="cleaned.xlsx not found")
        try:
            data = get_bytes(row.cleaned_xlsx_key)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read cleaned.xlsx: {e}")

        return StreamingResponse(
            io.BytesIO(data),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{run_id}_cleaned.xlsx"'},
        )

    if name == "report.json":
        if not row.report_key:
            raise HTTPException(status_code=404, detail="report.json not found")
        try:
            data = get_bytes(row.report_key)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read report.json: {e}")

        return StreamingResponse(
            io.BytesIO(data),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{run_id}_report.json"'},
        )

    raise HTTPException(status_code=400, detail="Unknown artifact name. Use cleaned.xlsx or report.json")

@router.delete("/cleaning/runs/{run_id}")
def delete_run(
    run_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = (
        db.query(CleaningRun)
        .filter(CleaningRun.run_id == run_id, CleaningRun.user_id == current_user.user_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")

    keys = [row.report_key, row.cleaned_parquet_key, row.cleaned_xlsx_key]
    for k in keys:
        if k:
            try:
                delete_key(k)
            except Exception:
                pass

    db.delete(row)
    db.commit()

    return {"ok": True, "deleted_run_id": run_id}