# backend/app/cleaning/_02_trim_strings.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import pandas as pd

def trim_strings(
    df: pd.DataFrame,
    *,
    enabled: bool = True,
    collapse_whitespace: bool = True,
    strip: bool = True,
    normalize_nbsp: bool = True,
    apply_to: str = "object_and_category",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans textual columns:
      - strip() spaces
      - normalize NBSP to regular space
      - collapse multiple whitespace to single space
    """
    report: Dict[str, Any] = {
        "enabled": bool(enabled),
        "columns_touched": [],
        "collapse_whitespace": bool(collapse_whitespace),
        "strip": bool(strip),
        "normalize_nbsp": bool(normalize_nbsp),
    }
    if not enabled:
        return df, report

    clean_df = df.copy()

    def _is_text_col(s: pd.Series) -> bool:
        if pd.api.types.is_object_dtype(s):
            return True
        if apply_to == "object_and_category" and str(s.dtype) == "category":
            return True
        return False

    touched: List[str] = []

    for col in clean_df.columns:
        s = clean_df[col]
        if not _is_text_col(s):
            continue

        x = s.astype("string")

        if normalize_nbsp:
            x = x.str.replace("\u00A0", " ", regex=False)

        if strip:
            x = x.str.strip()

        if collapse_whitespace:
            x = x.str.replace(r"\s+", " ", regex=True)

        clean_df[col] = x.where(~x.isna(), other=pd.NA)
        touched.append(col)

    report["columns_touched"] = touched
    return clean_df, report