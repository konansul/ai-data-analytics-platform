# backend/app/cleaning/_09_outliers.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def handle_outliers(
    df: pd.DataFrame,
    *,
    params: Optional[Dict[str, Any]] = None,
    enabled: bool = True,
    method: str = "quantile",             # "quantile" | "iqr"
    quantiles: Tuple[float, float] = (0.01, 0.99),
    iqr_k: float = 1.5,
    min_rows: int = 30,
    skip_low_unique_ratio: float = 0.02,  # skip id-like cols (almost all values unique)
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    params = params or {}

    enabled = bool(params.get("outliers", enabled))
    method = str(params.get("outliers_method", method))
    outliers_action = str(params.get("outliers_action", "clip"))
    iqr_k = float(params.get("iqr_k", iqr_k))
    zscore_threshold = float(params.get("zscore_threshold", 3.0))  # reserved
    quantiles = tuple(params.get("quantiles", quantiles))  # type: ignore[assignment]

    report: Dict[str, Any] = {
        "enabled": bool(enabled),
        "method": method,
        "action": outliers_action,
        "columns_clipped": [],
        "params": {
            "quantiles": quantiles,
            "iqr_k": float(iqr_k),
            "min_rows": int(min_rows),
            "skip_low_unique_ratio": float(skip_low_unique_ratio),
            "zscore_threshold": float(zscore_threshold),
        },
    }

    if (not enabled) or outliers_action in {"none", "off", "disable"}:
        return df, report

    if df.shape[0] < int(min_rows):
        return df, report

    clean_df = df.copy()
    cols_clipped: List[str] = []

    for col in clean_df.columns:
        s = clean_df[col]

        if pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean":
            continue

        if not pd.api.types.is_numeric_dtype(s):
            continue

        non_null = s.dropna()
        if non_null.empty:
            continue

        try:
            unique_ratio = float(non_null.nunique() / len(non_null))
        except Exception:
            unique_ratio = 0.0
        if unique_ratio >= (1.0 - float(skip_low_unique_ratio)):
            continue

        nn = pd.to_numeric(non_null, errors="coerce")
        try:
            nn = nn.astype(float)
        except Exception:
            nn = pd.Series(nn, dtype="float64")
        nn = nn.dropna()
        if nn.empty:
            continue

        if method == "iqr":
            q1 = float(nn.quantile(0.25))
            q3 = float(nn.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0.0:
                continue
            lo = q1 - float(iqr_k) * iqr
            hi = q3 + float(iqr_k) * iqr
        else:
            q_lo, q_hi = quantiles
            lo = float(nn.quantile(float(q_lo)))
            hi = float(nn.quantile(float(q_hi)))

        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            continue

        is_integer = pd.api.types.is_integer_dtype(s) or str(s.dtype) in {"Int64", "UInt64", "Int32", "UInt32"}
        if is_integer:
            lo_i = int(np.floor(lo))
            hi_i = int(np.ceil(hi))
            clean_df[col] = s.clip(lower=lo_i, upper=hi_i)
        else:
            clean_df[col] = pd.to_numeric(s, errors="coerce").clip(lower=lo, upper=hi)

        cols_clipped.append(col)

    report["columns_clipped"] = cols_clipped
    return clean_df, report