# backend/app/cleaning/_05_encode_booleans.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import pandas as pd


TRUE_TOKENS = {"true", "t", "yes", "y", "1", "on"}
FALSE_TOKENS = {"false", "f", "no", "n", "0", "off"}


def encode_booleans(
    df: pd.DataFrame,
    *,
    enabled: bool = True,
    output: str = "bool",  # "bool" | "int"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convert boolean-like columns (object/category) into bool or 0/1.
    """
    report: Dict[str, Any] = {
        "enabled": bool(enabled),
        "output": output,
        "columns_converted": [],
    }
    if not enabled:
        return df, report

    clean_df = df.copy()
    converted: List[str] = []

    for col in clean_df.columns:
        s = clean_df[col]

        if pd.api.types.is_bool_dtype(s):
            continue

        if not (pd.api.types.is_object_dtype(s) or str(s.dtype) == "category"):
            continue

        x = s.astype("string").str.strip().str.lower()
        non_null = x.dropna()

        if non_null.empty:
            continue

        uniq = set(non_null.unique().tolist())
        if not uniq:
            continue

        if not (uniq.issubset(TRUE_TOKENS | FALSE_TOKENS)):
            continue

        mapped = x.map(lambda v: True if v in TRUE_TOKENS else (False if v in FALSE_TOKENS else pd.NA))

        if output == "int":
            clean_df[col] = mapped.astype("boolean").astype("Int64")
        else:
            clean_df[col] = mapped.astype("boolean")

        converted.append(col)

    report["columns_converted"] = converted
    return clean_df, report