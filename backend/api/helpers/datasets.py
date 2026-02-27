# backend/services/datasets.py

from __future__ import annotations

import io
from typing import Any, Dict, Literal

import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from backend.database.models import Dataset
from backend.database.storage import get_bytes
from backend.api.helpers.ownership import get_owned_dataset_or_404


def dataset_to_meta(d: Dataset) -> Dict[str, Any]:
    return {
        "dataset_id": d.dataset_id,
        "original_dataset_id": d.original_dataset_id,
        "file_name": d.file_name,
        "sheet_name": d.sheet_name,
        "shape": [int(d.n_rows), int(d.n_cols)],
        "dtypes": dict(d.dtypes or {}),
        "storage": {
            "type": "local",
            "bucket": d.bucket,
            "raw_key": d.raw_key,
            "raw_parquet_key": getattr(d, "raw_parquet_key", None),
            "current_parquet_key": d.current_parquet_key,
        },
    }


def load_dataset_df(
    db: Session,
    *,
    dataset_id: str,
    user_id: str,
    version: Literal["raw", "current"] = "current",
) -> pd.DataFrame:
    ds = get_owned_dataset_or_404(db, dataset_id, user_id)

    if version == "raw":
        key = getattr(ds, "raw_parquet_key", None)
        if not key:
            raise HTTPException(
                status_code=500,
                detail="raw_parquet_key is missing in DB (migration needed)",
            )
    else:
        key = ds.current_parquet_key

    try:
        return pd.read_parquet(io.BytesIO(get_bytes(key)))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read parquet: {e}",
        )