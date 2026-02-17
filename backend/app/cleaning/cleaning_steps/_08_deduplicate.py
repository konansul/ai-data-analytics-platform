# backend/app/cleaning/_08_deduplicate.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


def deduplicate(
    df: pd.DataFrame,
    *,
    enabled: bool = True,
    mode: str = "full_row",
    subset: Optional[List[str]] = None,
    keep: str = "first",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Drop duplicate rows. Deterministic.
    """
    report: Dict[str, Any] = {
        "enabled": bool(enabled),
        "mode": mode,
        "subset": subset,
        "keep": keep,
        "rows_before": int(df.shape[0]),
        "rows_after": int(df.shape[0]),
        "dropped_duplicates": 0,
    }

    if not enabled:
        return df, report

    clean_df = df.copy()

    if mode == "subset":
        use_subset = [c for c in (subset or []) if c in clean_df.columns]
        if not use_subset:
            use_subset = None
    else:
        use_subset = None

    before = int(clean_df.shape[0])
    clean_df = clean_df.drop_duplicates(subset=use_subset, keep=keep)
    after = int(clean_df.shape[0])

    report["rows_before"] = before
    report["rows_after"] = after
    report["dropped_duplicates"] = int(before - after)
    return clean_df, report