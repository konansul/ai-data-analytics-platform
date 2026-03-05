# backend/api/helpers/json_utils.py

from __future__ import annotations

import pandas as pd
from typing import Any, Dict, List, Optional

def json_safe_records(df: pd.DataFrame, limit: Optional[int] = 200) -> List[Dict[str, Any]]:

    if df is None or df.empty:
        return []

    safe = df.where(pd.notnull(df), None).copy()

    for c in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[c]):
            safe[c] = safe[c].apply(lambda x: x.isoformat() if x is not None else None)

    if limit is not None:
        safe = safe.head(int(limit))

    out: List[Dict[str, Any]] = []
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
        out.append(cleaned)

    return out