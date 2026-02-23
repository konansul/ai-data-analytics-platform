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
from backend.api.storage import new_id
from backend.app.ingestion.dataset_loader import load_from_upload
from backend.database.storage import put_bytes, get_bytes  # local FS blob store
from backend.database.db import get_db
from backend.database.models import Dataset, User

router = APIRouter()

LOCAL_BUCKET_NAME = "local"


def _json_safe_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    safe = df.where(pd.notnull(df), None)

    for c in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[c]):
            safe[c] = safe[c].apply(lambda x: x.isoformat() if x is not None else None)

    records: List[Dict[str, Any]] = []
    for row in safe.to_dict(orient="records"):
        cleaned: Dict[str, Any] = {}
        for k, v in row.items():
            if hasattr(v, "item") and callable(getattr(v, "item")):
                try:
                    cleaned[k] = v.item()
                except Exception:
                    cleaned[k] = v
            else:
                cleaned[k] = v
        records.append(cleaned)

    return records


def _user_dataset_prefix(user_id: str, dataset_id: str) -> str:
    # ключи (пути) в local storage
    return f"users/{user_id}/datasets/{dataset_id}"


def _dataset_to_meta(d: Dataset) -> Dict[str, Any]:
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
            "raw_key": d.raw_key,  # raw bytes (bin)
            "raw_parquet_key": getattr(d, "raw_parquet_key", None),  # ✅ new
            "current_parquet_key": d.current_parquet_key,
        },
    }


def _get_owned_dataset_or_404(db: Session, dataset_id: str, user_id: str) -> Dataset:
    row = (
        db.query(Dataset)
        .filter(Dataset.dataset_id == dataset_id, Dataset.user_id == user_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return row

def load_dataset_as_dataframe(
    dataset_id: str,
    user_id: str,
    *,
    version: Literal["raw", "current"] = "current",
    db: Session,
) -> pd.DataFrame:

    row = _get_owned_dataset_or_404(db, dataset_id, user_id)

    if version == "raw":
        key = getattr(row, "raw_parquet_key", None)
        if not key:
            raise HTTPException(status_code=500, detail="raw_parquet_key is missing in DB (migration needed)")
    else:
        key = row.current_parquet_key

    try:
        parquet_bytes = get_bytes(key)
        return pd.read_parquet(io.BytesIO(parquet_bytes))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read parquet from local storage: {e}")


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
    return {"datasets": [_dataset_to_meta(r) for r in rows]}


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

        prefix = _user_dataset_prefix(current_user.user_id, ds_id)

        raw_key = f"{prefix}/raw.bin"
        raw_parquet_key = f"{prefix}/raw.parquet"
        current_parquet_key = f"{prefix}/current.parquet"

        # 1) raw bytes -> local storage
        put_bytes(raw_key, content)

        # 2) parquet bytes -> local storage (raw + current)
        try:
            buf = io.BytesIO()

            df_to_store = sc.df_parquet_safe if sc.parquet_safe_needed else sc.df

            df_to_store.to_parquet(buf, index=False)
            parquet_bytes = buf.getvalue()

            put_bytes(raw_parquet_key, parquet_bytes)
            put_bytes(current_parquet_key, parquet_bytes)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to persist parquet locally: {e}")

        # 3) meta -> Postgres (привязка к пользователю)
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
        created.append(_dataset_to_meta(row))

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
    row = _get_owned_dataset_or_404(db, dataset_id, current_user.user_id)
    return _dataset_to_meta(row)


@router.get("/datasets/{dataset_id}/preview", response_model=PreviewResponse)
def preview_dataset(
    dataset_id: str,
    rows: int = 50,
    version: Literal["raw", "current"] = "current",  # ✅ NEW
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = _get_owned_dataset_or_404(db, dataset_id, current_user.user_id)

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
        "rows": _json_safe_records(head),
    }


@router.get("/datasets/{dataset_id}/download")
def download_dataset(
    dataset_id: str,
    version: Literal["raw", "current"] = "current",  # ✅ raw = raw.parquet or raw.bin? (см. ниже)
    fmt: str = "xlsx",                               # xlsx | csv
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = _get_owned_dataset_or_404(db, dataset_id, current_user.user_id)

    # Если хочешь "raw" именно как исходные байты файла (raw.bin) — оставляем как было.
    # Если хочешь "raw" как parquet (чтобы скачивать raw в csv/xlsx) — ниже сделаем это.
    if version == "raw":
        # ✅ Download RAW as parquet -> allow csv/xlsx too
        key = getattr(row, "raw_parquet_key", None)
        if not key:
            raise HTTPException(status_code=500, detail="raw_parquet_key is missing in DB (migration needed)")

        try:
            parquet_bytes = get_bytes(key)
            df = pd.read_parquet(io.BytesIO(parquet_bytes))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read raw parquet: {e}")

    else:
        # current
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