# backend/api/datasets.py
from __future__ import annotations

import io
from typing import Any, Dict, List, Literal

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.api.auth import get_current_user
from backend.api.models import PreviewResponse, UploadResponse
from backend.app.ingestion.dataset_loader import load_from_upload
from backend.database.storage import put_bytes, get_bytes, new_id, dataset_prefix
from backend.database.db import get_db
from backend.database.models import Dataset, User

from backend.api.helpers.ownership import get_owned_dataset_or_404
from backend.api.helpers.json_utils import json_safe_records
from backend.api.helpers.datasets import dataset_to_meta
router = APIRouter()

LOCAL_BUCKET_NAME = "local"

@router.get("/datasets")
def list_datasets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    rows = (
        db.query(Dataset)
        .filter(Dataset.user_id == current_user.user_id)
        .order_by(Dataset.created_at.desc())
        .limit(200)
        .all()
    )
    return {"datasets": [dataset_to_meta(r) for r in rows]}


@router.post("/datasets", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    filename = file.filename or "upload.bin"
    content = await file.read()

    try:
        sheet_contexts = load_from_upload(content, filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload parse error: {e}")

    created: List[Dict[str, Any]] = []

    for sc in sheet_contexts:
        ds_id = new_id("ds")

        prefix = dataset_prefix(current_user.user_id, ds_id)
        raw_key = f"{prefix}/raw.bin"
        raw_parquet_key = f"{prefix}/raw.parquet"
        current_parquet_key = f"{prefix}/current.parquet"

        put_bytes(raw_key, content)

        try:
            buf = io.BytesIO()
            sc.df.to_parquet(buf, index=False)
            parquet_bytes = buf.getvalue()

            put_bytes(raw_parquet_key, parquet_bytes)

            put_bytes(current_parquet_key, parquet_bytes)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to persist parquet locally: {e}")

        row = Dataset(
            dataset_id=ds_id,
            user_id=current_user.user_id,
            original_dataset_id=sc.dataset_id,
            file_name=sc.file_name,
            sheet_name=sc.sheet_name,
            n_rows=int(sc.shape[0]),
            n_cols=int(sc.shape[1]),
            dtypes=dict(sc.dtypes),
            bucket=LOCAL_BUCKET_NAME,
            raw_key=raw_key,
            raw_parquet_key=raw_parquet_key,
            current_parquet_key=current_parquet_key,
        )
        db.add(row)
        created.append(dataset_to_meta(row))

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB commit failed: {e}")

    return {"datasets": created}


@router.get("/datasets/{dataset_id}")
def get_dataset_meta(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    row = get_owned_dataset_or_404(db, dataset_id, current_user.user_id)
    return dataset_to_meta(row)


@router.get("/datasets/{dataset_id}/preview", response_model=PreviewResponse)
def preview_dataset(
    dataset_id: str,
    rows: int = 50,
    version: Literal["raw", "current"] = "current",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = get_owned_dataset_or_404(db, dataset_id, current_user.user_id)

    if version == "raw":
        key = getattr(row, "raw_parquet_key", None)
        if not key:
            raise HTTPException(status_code=500, detail="raw_parquet_key is missing in DB (migration needed)")
    else:
        key = row.current_parquet_key

    try:
        parquet_bytes = get_bytes(key)
        df = pd.read_parquet(io.BytesIO(parquet_bytes))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read parquet from local storage: {e}")

    head = df.head(max(0, min(rows, 500)))
    return {
        "dataset_id": dataset_id,
        "columns": [str(c) for c in df.columns.tolist()],
        "rows": json_safe_records(head),
    }


@router.get("/datasets/{dataset_id}/download")
def download_dataset(
    dataset_id: str,
    version: Literal["raw", "current"] = "current",
    fmt: str = "xlsx",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = get_owned_dataset_or_404(db, dataset_id, current_user.user_id)

    if version == "raw":
        key = getattr(row, "raw_parquet_key", None)
        if not key:
            raise HTTPException(status_code=500, detail="raw_parquet_key is missing in DB (migration needed)")

        try:
            parquet_bytes = get_bytes(key)
            df = pd.read_parquet(io.BytesIO(parquet_bytes))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read raw parquet: {e}")

    else:
        try:
            parquet_bytes = get_bytes(row.current_parquet_key)
            df = pd.read_parquet(io.BytesIO(parquet_bytes))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read current parquet: {e}")

    fmt = fmt.lower()
    if fmt == "csv":
        out = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
        return StreamingResponse(
            out,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{dataset_id}_{version}.csv"'},
        )

    if fmt == "xlsx":
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="data")
        out.seek(0)
        return StreamingResponse(
            out,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{dataset_id}_{version}.xlsx"'},
        )

    raise HTTPException(status_code=400, detail="fmt must be xlsx or csv")