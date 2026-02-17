# backend/app/cleaning/_10_impute_missing.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd


def _is_int_series(s: pd.Series) -> bool:
    try:
        return pd.api.types.is_integer_dtype(s.dtype)
    except Exception:
        return False


def _is_integer_value(x: Any) -> bool:
    try:
        xf = float(x)
    except Exception:
        return False
    return xf.is_integer()


def _safe_fill_numeric(series: pd.Series, value: Any) -> pd.Series:
    if _is_int_series(series):
        if value is None:
            return series
        if isinstance(value, bool):
            return series.fillna(int(value))
        if isinstance(value, (int,)):
            return series.fillna(int(value))
        if isinstance(value, (float,)):
            if _is_integer_value(value):
                return series.fillna(int(value))
            return series.astype("Float64").fillna(float(value))
        try:
            fv = float(value)
            if _is_integer_value(fv):
                return series.fillna(int(fv))
            return series.astype("Float64").fillna(float(fv))
        except Exception:
            return series
    return series.fillna(value)


def impute_missing_values(
    df: pd.DataFrame,
    *,
    enabled: bool = True,
    numeric_strategy: Optional[str] = "median",
    categorical_strategy: Optional[str] = "mode",
    datetime_strategy: Optional[str] = None,
    fill_value: Union[int, float, str, None] = 0,
    categorical_numeric_max_unique: int = 20,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    clean_df = df.copy()

    report: Dict[str, Any] = {
        "enabled": bool(enabled),
        "numeric_strategy": numeric_strategy,
        "categorical_strategy": categorical_strategy,
        "datetime_strategy": datetime_strategy,
        "categorical_numeric_max_unique": int(categorical_numeric_max_unique),
        "filled_counts": {},
        "fill_values_used": {},
        "categorical_numeric_columns": [],
        "dtype_upcasts": {},
        "total_filled": 0,
    }

    if not enabled:
        return clean_df, report

    for col in clean_df.columns:
        missing_before = int(clean_df[col].isna().sum())
        if missing_before == 0:
            continue

        s = clean_df[col]
        before_dtype = str(s.dtype)

        if pd.api.types.is_datetime64_any_dtype(s):
            if datetime_strategy == "ffill":
                clean_df[col] = s.ffill()
            elif datetime_strategy == "bfill":
                clean_df[col] = s.bfill()
            else:
                continue

        elif pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean":
            if categorical_strategy == "mode":
                modes = s.mode(dropna=True)
                if modes.empty:
                    continue
                value = bool(modes.iloc[0])
                clean_df[col] = s.fillna(value)
                report["fill_values_used"][col] = value
            elif categorical_strategy == "constant":
                clean_df[col] = s.fillna(bool(fill_value))
                report["fill_values_used"][col] = bool(fill_value)
            else:
                continue

        elif pd.api.types.is_numeric_dtype(s):
            nunique = int(s.nunique(dropna=True))

            if nunique <= int(categorical_numeric_max_unique):
                if categorical_strategy == "mode":
                    modes = s.mode(dropna=True)
                    if modes.empty:
                        continue
                    value = modes.iloc[0]
                    clean_df[col] = _safe_fill_numeric(s, value) if pd.api.types.is_numeric_dtype(s) else s.fillna(value)
                    report["fill_values_used"][col] = value
                    report["categorical_numeric_columns"].append(col)
                elif categorical_strategy == "constant":
                    clean_df[col] = _safe_fill_numeric(s, fill_value)
                    report["fill_values_used"][col] = fill_value
                    report["categorical_numeric_columns"].append(col)
                else:
                    continue
            else:
                if numeric_strategy == "mean":
                    value = s.mean()
                    if pd.isna(value):
                        continue
                    value = float(value)
                    clean_df[col] = _safe_fill_numeric(s, value)
                    report["fill_values_used"][col] = value

                elif numeric_strategy == "median":
                    value = s.median()
                    if pd.isna(value):
                        continue
                    value = float(value)
                    clean_df[col] = _safe_fill_numeric(s, value)
                    report["fill_values_used"][col] = value

                elif numeric_strategy == "constant":
                    clean_df[col] = _safe_fill_numeric(s, fill_value)
                    report["fill_values_used"][col] = fill_value

                else:
                    continue

        else:
            if categorical_strategy == "mode":
                modes = s.mode(dropna=True)
                if modes.empty:
                    continue
                value = modes.iloc[0]
                clean_df[col] = s.fillna(value)
                report["fill_values_used"][col] = value
            elif categorical_strategy == "constant":
                clean_df[col] = s.fillna(fill_value)
                report["fill_values_used"][col] = fill_value
            else:
                continue

        after_dtype = str(clean_df[col].dtype)
        if after_dtype != before_dtype:
            report["dtype_upcasts"][col] = {"from": before_dtype, "to": after_dtype}

        missing_after = int(clean_df[col].isna().sum())
        filled = missing_before - missing_after
        if filled > 0:
            report["filled_counts"][col] = int(filled)

    report["total_filled"] = int(sum(report["filled_counts"].values()))
    return clean_df, report