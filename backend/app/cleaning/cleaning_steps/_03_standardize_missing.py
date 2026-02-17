# backend/app/cleaning/_03_standardize_missing.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import pandas as pd


DEFAULT_MISSING_TOKENS = [
    "", " ", "  ", "-", "--", "---",
    "na", "n/a", "none", "null", "nan",
    "missing", "unknown", "undefined",
    "#n/a", "#na", "#null!",
]


def standardize_missing(
    df: pd.DataFrame,
    *,
    enabled: bool = True,
    tokens: List[str] | None = None,
    case_insensitive: bool = True,
    treat_whitespace_as_missing: bool = True,
    apply_to: str = "object_and_category",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Replace common "missing" string tokens with NA (pd.NA) in text columns.
    """
    report: Dict[str, Any] = {
        "enabled": bool(enabled),
        "columns_touched": [],
        "case_insensitive": bool(case_insensitive),
        "treat_whitespace_as_missing": bool(treat_whitespace_as_missing),
        "tokens_used": tokens or DEFAULT_MISSING_TOKENS,
    }
    if not enabled:
        return df, report

    clean_df = df.copy()
    toks = tokens or DEFAULT_MISSING_TOKENS
    toks_set = set([t for t in toks if isinstance(t, str)])

    touched: List[str] = []

    def _is_text_col(s: pd.Series) -> bool:
        if pd.api.types.is_object_dtype(s):
            return True
        if apply_to == "object_and_category" and str(s.dtype) == "category":
            return True
        return False

    for col in clean_df.columns:
        s = clean_df[col]
        if not _is_text_col(s):
            continue

        x = s.astype("string")

        if treat_whitespace_as_missing:
            x = x.str.strip()

        if case_insensitive:
            x_cmp = x.str.lower()
            mask = x_cmp.isin([t.lower() for t in toks_set])
        else:
            mask = x.isin(list(toks_set))

        if bool(mask.fillna(False).any()):
            clean_df[col] = x.mask(mask, other=pd.NA)
            touched.append(col)

    report["columns_touched"] = touched
    return clean_df, report