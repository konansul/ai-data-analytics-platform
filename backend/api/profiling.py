# backend/api/profiling.py
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.api.auth import get_current_user
from backend.api.models import ProfilingRequest, ProfilingResponse
from backend.app.profiling.profiling import profile_dataframe

from backend.database.db import get_db
from backend.database.models import Dataset, Profile, User
from backend.database.storage import get_bytes, put_bytes, to_jsonable, new_id, profile_prefix

from backend.api.helpers.ownership import get_owned_dataset_or_404

router = APIRouter()

@router.post("/profiling", response_model=ProfilingResponse)
def run_profiling(
    req: ProfilingRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ds = get_owned_dataset_or_404(db, req.dataset_id, current_user.user_id)

    try:
        parquet_bytes = get_bytes(ds.current_parquet_key)
        df = pd.read_parquet(io.BytesIO(parquet_bytes))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset parquet: {e}")

    options = req.options or {}
    try:
        report = profile_dataframe(df, **options)
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"Bad profiling options: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling failed: {e}")

    profile_id = new_id("prof")
    prefix = profile_prefix(current_user.user_id, profile_id)
    report_key = f"{prefix}/report.json"

    try:
        safe_report = to_jsonable(report)
        put_bytes(
            report_key,
            json.dumps(safe_report, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist profile report: {e}")

    try:
        row = Profile(
            profile_id=profile_id,
            dataset_id=ds.dataset_id,
            bucket="local",
            report_key=report_key,
            created_at=datetime.now(timezone.utc),
        )
        db.add(row)
        db.commit()
    except Exception:
        pass

    return {"profile_id": profile_id}


@router.get("/profiling/{profile_id}")
def get_profiling_report(profile_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    row: Optional[Profile] = (
        db.query(Profile)
        .join(Dataset, Dataset.dataset_id == Profile.dataset_id)
        .filter(Profile.profile_id == profile_id, Dataset.user_id == current_user.user_id)
        .first()
    )
    if not row or not row.report_key:
        raise HTTPException(status_code=404, detail="Profile report not found")

    try:
        raw = get_bytes(row.report_key)
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read profile report: {e}")