# backend/database/storage.py

from __future__ import annotations

import os
import io
import json
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

BLOB_DIR = Path(os.getenv("LOCAL_BLOB_DIR", "storage")).resolve()


def ensure_blob_dir() -> None:
    BLOB_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_key(key: str) -> str:
    key = (key or "").strip().lstrip("/").replace("\\", "/")
    parts = [p for p in key.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ValueError("Invalid blob key: contains '..'")
    return "/".join(parts)


def _full_path(key: str) -> Path:
    key = _sanitize_key(key)
    path = (BLOB_DIR / key).resolve()

    try:
        path.relative_to(BLOB_DIR)
    except ValueError:
        raise ValueError("Invalid blob key: resolves outside blob dir")

    return path

def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def put_bytes(key: str, data: bytes, content_type: Optional[str] = None) -> None:
    ensure_blob_dir()
    path = _full_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def get_bytes(key: str) -> bytes:
    path = _full_path(key)
    if not path.exists():
        raise FileNotFoundError(f"Blob not found: {key}")
    return path.read_bytes()


def exists(key: str) -> bool:
    try:
        return _full_path(key).exists()
    except Exception:
        return False


def delete_key(key: str) -> None:
    path = _full_path(key)
    if path.exists():
        path.unlink()

def to_jsonable(x: Any) -> Any:
    if x is None:
        return None

    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, (pd.Timestamp, datetime, date)):
        return x.isoformat()

    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]

    if isinstance(x, pd.Series):
        return [to_jsonable(v) for v in x.tolist()]
    if isinstance(x, pd.Index):
        return [to_jsonable(v) for v in x.tolist()]
    if isinstance(x, pd.DataFrame):
        return to_jsonable(x.to_dict(orient="records"))
    if isinstance(x, np.ndarray):
        return [to_jsonable(v) for v in x.tolist()]

    return x


def put_text(key: str, text: str, encoding: str = "utf-8") -> None:
    put_bytes(key, text.encode(encoding))


def get_text(key: str, encoding: str = "utf-8") -> str:
    return get_bytes(key).decode(encoding)


def put_json(key: str, obj: Any, *, indent: int = 2) -> None:
    payload = json.dumps(to_jsonable(obj), ensure_ascii=False, indent=indent).encode("utf-8")
    put_bytes(key, payload, content_type="application/json")


def get_json(key: str) -> Any:
    return json.loads(get_text(key))


def put_df_parquet(key: str, df: pd.DataFrame) -> None:
    bio = io.BytesIO()
    df.to_parquet(bio, index=False)
    put_bytes(key, bio.getvalue(), content_type="application/octet-stream")


def get_df_parquet(key: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(get_bytes(key)))


def put_df_csv(key: str, df: pd.DataFrame) -> None:
    bio = io.StringIO()
    df.to_csv(bio, index=False)
    put_text(key, bio.getvalue())


def user_root(user_id: str) -> str:
    return f"users/{user_id}"


def dataset_root(user_id: str, dataset_id: str) -> str:
    return f"{user_root(user_id)}/datasets/{dataset_id}"


def dataset_key(user_id: str, dataset_id: str, filename: str) -> str:
    return f"{dataset_root(user_id, dataset_id)}/{filename}"


def runs_root(user_id: str) -> str:
    return f"{user_root(user_id)}/runs"


def run_root(user_id: str, run_id: str) -> str:
    return f"{runs_root(user_id)}/{run_id}"


def run_key(user_id: str, run_id: str, filename: str) -> str:
    return f"{run_root(user_id, run_id)}/{filename}"


def stage_root(user_id: str, run_id: str, stage: str) -> str:
    stage = (stage or "").strip().lower()
    if stage not in {"cleaning", "viz", "forecast", "report"}:
        raise ValueError(f"Invalid stage: {stage}")
    return f"{run_root(user_id, run_id)}/{stage}"


def stage_key(user_id: str, run_id: str, stage: str, filename: str) -> str:
    return f"{stage_root(user_id, run_id, stage)}/{filename}"


def cleaning_root(user_id: str, run_id: str) -> str:
    return stage_root(user_id, run_id, "cleaning")


def cleaning_key(user_id: str, run_id: str, filename: str) -> str:
    return stage_key(user_id, run_id, "cleaning", filename)


def viz_root(user_id: str, run_id: str) -> str:
    return stage_root(user_id, run_id, "viz")


def viz_key(user_id: str, run_id: str, filename: str) -> str:
    return stage_key(user_id, run_id, "viz", filename)


def viz_plot_key(user_id: str, run_id: str, filename: str) -> str:
    return f"{viz_root(user_id, run_id)}/plots/{filename}"


def forecast_root(user_id: str, run_id: str) -> str:
    return stage_root(user_id, run_id, "forecast")


def forecast_key(user_id: str, run_id: str, filename: str) -> str:
    return stage_key(user_id, run_id, "forecast", filename)


def forecast_plot_key(user_id: str, run_id: str, filename: str) -> str:
    return f"{forecast_root(user_id, run_id)}/plots/{filename}"


def report_root(user_id: str, run_id: str) -> str:
    return stage_root(user_id, run_id, "report")


def report_key(user_id: str, run_id: str, filename: str) -> str:
    return stage_key(user_id, run_id, "report", filename)


def dataset_prefix(user_id: str, dataset_id: str) -> str:
    return f"users/{user_id}/datasets/{dataset_id}"


def run_prefix(user_id: str, run_id: str) -> str:
    return f"users/{user_id}/runs/{run_id}"


def cleaning_prefix(user_id: str, run_id: str) -> str:
    return f"{run_prefix(user_id, run_id)}/cleaning"


def profile_prefix(user_id: str, profile_id: str) -> str:
    return f"users/{user_id}/profiles/{profile_id}"


def viz_prefix(user_id: str, run_id: str) -> str:
    return f"{run_prefix(user_id, run_id)}/viz"


def forecast_prefix(user_id: str, run_id: str, forecast_run_id: str) -> str:
    return f"{run_prefix(user_id, run_id)}/forecast/{forecast_run_id}"

def list_keys(prefix: str) -> List[str]:
    prefix = _sanitize_key(prefix).rstrip("/")
    base = _full_path(prefix)

    if not base.exists():
        return []

    out: List[str] = []
    for p in base.rglob("*"):
        if p.is_file():
            rel = p.relative_to(BLOB_DIR).as_posix()
            out.append(rel)
    return sorted(out)