# backend/app/cleaning/_07_datetime_inference.py
from __future__ import annotations

from typing import List, Tuple, Optional
import pandas as pd


def infer_datetime_columns(
    df: pd.DataFrame,
    *,
    enabled: bool = True,
    datetime_success_ratio: float = 0.8,
    sample_size: int = 200,
    max_letters_ratio: float = 0.30,
    dayfirst: Optional[bool] = None,
    utc: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Try converting object/category columns to datetime if conversion succeeds
    for at least datetime_success_ratio of non-null values.

    Protection:
      - skip text-like columns (letters ratio too high)
    """
    if not enabled:
        return df, []

    clean_df = df.copy()
    inferred: List[str] = []

    for col in clean_df.columns:
        s = clean_df[col]

        if pd.api.types.is_datetime64_any_dtype(s):
            continue

        if not (pd.api.types.is_object_dtype(s) or str(s.dtype) == "category"):
            continue

        non_null = s.dropna()
        if non_null.empty:
            continue

        sample = non_null.astype(str).head(int(sample_size))
        if sample.empty:
            continue

        letters_ratio = sample.str.contains(r"[A-Za-zА-Яа-я]", regex=True).mean()
        if float(letters_ratio) > float(max_letters_ratio):
            continue

        converted = pd.to_datetime(
            s,
            errors="coerce",
            dayfirst=dayfirst,
            utc=utc,
        )

        total = int(s.notna().sum())
        ok = int(converted.notna().sum())
        if total > 0 and (ok / total) >= float(datetime_success_ratio):
            clean_df[col] = converted
            inferred.append(col)

    return clean_df, inferred