# backend/api/helpers/artifacts.py

from datetime import datetime, timezone
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from backend.database.storage import new_id
from backend.database.models import Artifact
from backend.database.storage import to_jsonable, get_json, exists


def add_artifact(
    db: Session,
    *,
    user_id: str,
    dataset_id: str,
    run_type: str,
    run_id: str,
    kind: str,
    mime_type: str,
    storage_key: str,
    meta: Optional[Dict[str, Any]] = None,
    parent_run_id: Optional[str] = None,
):
    art = Artifact(
        artifact_id=new_id("art"),
        user_id=user_id,
        dataset_id=dataset_id,
        run_type=run_type,
        run_id=run_id,
        parent_run_id=parent_run_id,
        kind=kind,
        mime_type=mime_type,
        bucket="local",
        storage_key=storage_key,
        meta=to_jsonable(meta or {}),
        created_at=datetime.now(timezone.utc),
    )
    db.add(art)
    return art

def _artifact_id() -> str:
    return new_id("art")

def read_json_from_storage_safe(storage_key: str) -> Optional[Dict[str, Any]]:

    try:
        if not exists(storage_key):
            return None
        obj = get_json(storage_key)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None