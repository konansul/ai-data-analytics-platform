# backend/app/cleaning/_06_drop_rules.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


def drop_empty_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    cols = [c for c in df.columns if df[c].isna().all()]
    if not cols:
        return df, []
    return df.drop(columns=cols), cols


def drop_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if not cols:
        return df, []
    return df.drop(columns=cols), cols


def drop_high_missing_columns(df: pd.DataFrame, missing_threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    missing_frac = df.isna().mean(numeric_only=False)
    cols = [c for c, frac in missing_frac.items() if float(frac) > float(missing_threshold)]
    if not cols:
        return df, []
    return df.drop(columns=cols), cols


def drop_high_missing_rows(
    df: pd.DataFrame,
    row_missing_threshold: float = 0.80,
    *,
    ignore_columns: Optional[List[str]] = None,
    min_cols_to_apply: int = 1,
) -> Tuple[pd.DataFrame, int]:
    """
    Drop rows where missing fraction across considered columns >= row_missing_threshold.
    Returns: (df, dropped_rows_count)

    Notes:
    - Should be called AFTER standardize_missing (so "", "NA", etc become NaN).
    - ignore_columns is useful for ID-like columns you don't want to count.
    """
    if df.shape[1] < int(min_cols_to_apply):
        return df, 0

    cols = list(df.columns)
    if ignore_columns:
        ignore = set(ignore_columns)
        cols = [c for c in cols if c not in ignore]

    if not cols:
        return df, 0

    miss_frac_per_row = df[cols].isna().mean(axis=1)
    to_drop_mask = miss_frac_per_row >= float(row_missing_threshold)

    dropped = int(to_drop_mask.sum())
    if dropped == 0:
        return df, 0

    return df.loc[~to_drop_mask].copy(), dropped


def apply_drop_rules(
    df: pd.DataFrame,
    *,
    missing_threshold: float = 0.5,
    row_missing_threshold: float = 0.80,
    drop_rows: bool = True,
    ignore_columns_for_row_drop: Optional[List[str]] = None,
    min_cols_to_apply_for_row_drop: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Unified drop-rules step: columns + (optional) rows.
    Returns: (clean_df, report)
    """
    clean_df = df.copy()

    report: Dict[str, Any] = {
        "enabled": True,
        "missing_threshold": float(missing_threshold),
        "row_missing_threshold": float(row_missing_threshold),
        "drop_rows": bool(drop_rows),
        "ignore_columns_for_row_drop": list(ignore_columns_for_row_drop or []),
        "min_cols_to_apply_for_row_drop": int(min_cols_to_apply_for_row_drop),
        "dropped_empty_columns": [],
        "dropped_constant_columns": [],
        "dropped_high_missing_columns": [],
        "dropped_rows_high_missing": 0,
        "rows_before": int(clean_df.shape[0]),
        "cols_before": int(clean_df.shape[1]),
        "rows_after": None,
        "cols_after": None,
        "dropped_total_columns": 0,
    }

    # 1. Drop column rules
    clean_df, dropped_empty = drop_empty_columns(clean_df)
    clean_df, dropped_constant = drop_constant_columns(clean_df)
    clean_df, dropped_high = drop_high_missing_columns(clean_df, missing_threshold)

    report["dropped_empty_columns"] = dropped_empty
    report["dropped_constant_columns"] = dropped_constant
    report["dropped_high_missing_columns"] = dropped_high

    # 2. Drop high-missing rows
    if drop_rows:
        clean_df, dropped_rows = drop_high_missing_rows(
            clean_df,
            row_missing_threshold=float(row_missing_threshold),
            ignore_columns=ignore_columns_for_row_drop,
            min_cols_to_apply=int(min_cols_to_apply_for_row_drop),
        )
        report["dropped_rows_high_missing"] = int(dropped_rows)

    # 3. Final counters
    report["rows_after"] = int(clean_df.shape[0])
    report["cols_after"] = int(clean_df.shape[1])
    report["dropped_total_columns"] = int(len(dropped_empty) + len(dropped_constant) + len(dropped_high))

    return clean_df, report