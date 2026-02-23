# backend/app/profiling/profile.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

def _make_table_snippet(
    df: pd.DataFrame,
    *,
    max_rows: int = 12,
    max_cols: int = 14,
    max_cell_chars: int = 80,
) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"rows": [], "shape": [0, 0]}

    r = max(1, min(int(max_rows), 50))
    c = max(1, min(int(max_cols), 50))
    max_cell_chars = max(20, min(int(max_cell_chars), 500))

    head = df.iloc[:r, :c].copy()

    # Convert to string-ish table, safe for LLM
    head = head.where(pd.notnull(head), "")

    # Convert datetimes to isoformat
    for col in head.columns:
        if pd.api.types.is_datetime64_any_dtype(head[col]):
            head[col] = head[col].apply(lambda x: x.isoformat() if x else "")

    # Everything -> string, clamp cell size
    def _cell(v: Any) -> str:
        s = str(v)
        s = s.replace("\n", " ").replace("\r", " ").strip()
        if len(s) > max_cell_chars:
            s = s[:max_cell_chars] + "…"
        return s

    header = [_cell(x) for x in list(head.columns)]
    body = [[_cell(v) for v in row] for row in head.to_numpy().tolist()]

    return {
        "rows": [header] + body,   # first row is header
        "shape": [int(head.shape[0]), int(head.shape[1])],
    }


def profile_dataframe(
    df: pd.DataFrame,
    *,
    max_categories: int = 30,
    top_k: int = 10,
    max_corr_numeric_cols: int = 30,
    corr_sample_rows: int = 50_000,
    skew_min_rows: int = 20,
    sample_size: int = 200,
    datetime_success_ratio: float = 0.8,
    iqr_k: float = 1.5,
) -> Dict[str, Any]:
    profile: Dict[str, Any] = {}

    n_rows, n_cols = df.shape
    profile["n_rows"] = int(n_rows)
    profile["n_cols"] = int(n_cols)

    profile["table_snippet"] = _make_table_snippet(
        df,
        max_rows=12,
        max_cols=14,
        max_cell_chars=80,
    )

    mem_bytes = int(df.memory_usage(deep=True).sum()) if n_cols > 0 else 0
    profile["memory_bytes"] = mem_bytes
    profile["memory_mb"] = float(np.round(mem_bytes / (1024 * 1024), 3))

    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    bool_cols = [c for c in df.columns if pd.api.types.is_bool_dtype(df[c])]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in bool_cols]
    categorical_cols = [
        c
        for c in df.columns
        if (pd.api.types.is_object_dtype(df[c]) or str(df[c].dtype) == "category") and c not in datetime_cols
    ]

    profile["columns"] = {
        "datetime": datetime_cols,
        "numeric": numeric_cols,
        "boolean": bool_cols,
        "categorical": categorical_cols,
    }
    profile["counts"] = {
        "datetime": int(len(datetime_cols)),
        "numeric": int(len(numeric_cols)),
        "boolean": int(len(bool_cols)),
        "categorical": int(len(categorical_cols)),
    }

    has_time_index = len(datetime_cols) > 0
    time_col = datetime_cols[0] if datetime_cols else None
    profile["has_time_index"] = bool(has_time_index)
    profile["time_column"] = time_col

    if has_time_index and len(numeric_cols) > 0:
        dataset_type = "time_series"
    elif not has_time_index:
        dataset_type = "tabular"
    else:
        dataset_type = "mixed"
    profile["dataset_type"] = dataset_type

    if n_cols == 0:
        overall_missing_pct = 0.0
        top_missing = {}
        missing_frac = pd.Series(dtype=float)
    else:
        missing_frac = df.isna().mean(numeric_only=False)
        overall_missing_pct = float(missing_frac.mean() * 100.0)
        top_missing = (
            missing_frac.sort_values(ascending=False)
            .head(top_k)
            .mul(100.0)
            .round(3)
            .to_dict()
        )

    profile["missingness"] = {
        "overall_missing_%": float(np.round(overall_missing_pct, 3)),
        "top_missing_columns": top_missing,
    }
    profile["overall_missing_%"] = profile["missingness"]["overall_missing_%"]
    profile["top_missing_columns"] = profile["missingness"]["top_missing_columns"]

    cat_info: List[Dict[str, Any]] = []
    for c in categorical_cols:
        try:
            nunique = int(df[c].nunique(dropna=True))
        except Exception:
            nunique = -1
        cat_info.append({"column": c, "unique_values": nunique})

    cat_info.sort(key=lambda x: x["unique_values"], reverse=True)
    profile["categorical_cardinality"] = cat_info

    warnings: List[str] = []
    for item in cat_info:
        if item["unique_values"] > max_categories:
            warnings.append(f"Column '{item['column']}' has high cardinality ({item['unique_values']} categories).")
    profile["warnings"] = warnings

    dup_info: Dict[str, Any] = {"duplicate_rows": 0, "duplicate_rows_%": 0.0}
    if n_rows > 0 and n_cols > 0:
        try:
            dup_count = int(df.duplicated().sum())
            dup_info["duplicate_rows"] = dup_count
            dup_info["duplicate_rows_%"] = float(np.round((dup_count / max(n_rows, 1)) * 100.0, 3))
        except Exception:
            pass
    profile["duplicates"] = dup_info

    string_quality: Dict[str, Any] = {"columns": {}}
    missing_markers = {"", "na", "n/a", "null", "none", "nan", "-", "--", "—", "?", "unknown", "undefined"}
    for c in categorical_cols:
        try:
            s = df[c]
            non_null = s.dropna()
            if non_null.empty:
                continue
            sample = non_null.astype(str).head(sample_size)
            stripped = sample.str.strip()
            leading_trailing_pct = float(np.round((sample != stripped).mean() * 100.0, 3))
            empty_after_strip_pct = float(np.round((stripped == "").mean() * 100.0, 3))
            marker_pct = float(np.round(stripped.str.lower().isin(missing_markers).mean() * 100.0, 3))
            string_quality["columns"][c] = {
                "leading_trailing_spaces_%": leading_trailing_pct,
                "empty_after_strip_%": empty_after_strip_pct,
                "missing_marker_like_%": marker_pct,
            }
        except Exception:
            continue
    profile["string_quality"] = string_quality

    boolean_candidates: List[Dict[str, Any]] = []
    bool_sets = [
        {"true", "false"},
        {"yes", "no"},
        {"y", "n"},
        {"1", "0"},
        {"t", "f"},
        {"on", "off"},
    ]
    for c in categorical_cols:
        try:
            s = df[c].dropna()
            if s.empty:
                continue
            sample = s.astype(str).head(sample_size).str.strip().str.lower()
            vals = set(sample.unique().tolist())
            vals = {v for v in vals if v != ""}
            if 0 < len(vals) <= 5:
                for bs in bool_sets:
                    if vals.issubset(bs):
                        boolean_candidates.append({"column": c, "values_sample": sorted(list(vals))})
                        break
        except Exception:
            continue
    profile["boolean_candidates"] = boolean_candidates

    datetime_candidates: List[Dict[str, Any]] = []
    for c in categorical_cols:
        try:
            s = df[c].dropna()
            if s.empty:
                continue
            sample = s.astype(str).head(sample_size)
            sample = sample[sample.astype(str).str.strip() != ""]
            if sample.empty:
                continue
            letters_ratio = float(sample.str.contains(r"[A-Za-zА-Яа-я]", regex=True).mean())
            if letters_ratio > 0.3:
                continue
            converted = pd.to_datetime(sample, errors="coerce")
            ok_ratio = float(converted.notna().mean())
            if ok_ratio >= float(datetime_success_ratio):
                datetime_candidates.append(
                    {"column": c, "success_ratio": float(np.round(ok_ratio, 4)), "letters_ratio": float(np.round(letters_ratio, 4))}
                )
        except Exception:
            continue
    profile["datetime_candidates"] = datetime_candidates

    skewness: Dict[str, float] = {}
    for c in numeric_cols:
        series = df[c].dropna()
        if len(series) < skew_min_rows:
            continue
        try:
            skewness[c] = float(np.round(series.skew(), 4))
        except Exception:
            continue

    top_abs_skewed = dict(sorted(skewness.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k])
    profile["skewness"] = {"per_numeric_column": skewness, "top_abs_skewed": top_abs_skewed}
    profile["skewness_top_abs"] = top_abs_skewed

    outlier_signals: Dict[str, Any] = {"per_numeric_column": {}, "top_outlier_cols": []}
    for c in numeric_cols:
        try:
            s = df[c].dropna()
            if len(s) < 20:
                continue
            q1 = float(np.nanpercentile(s.values, 25))
            q3 = float(np.nanpercentile(s.values, 75))
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr == 0:
                continue
            lo = q1 - float(iqr_k) * iqr
            hi = q3 + float(iqr_k) * iqr
            out_frac = float(np.mean((s.values < lo) | (s.values > hi)))
            outlier_signals["per_numeric_column"][c] = {
                "outliers_%": float(np.round(out_frac * 100.0, 3)),
                "iqr_k": float(iqr_k),
            }
        except Exception:
            continue

    if outlier_signals["per_numeric_column"]:
        top_out = sorted(
            outlier_signals["per_numeric_column"].items(),
            key=lambda kv: kv[1].get("outliers_%", 0.0),
            reverse=True,
        )[:top_k]
        outlier_signals["top_outlier_cols"] = [{"column": k, **v} for k, v in top_out]
    profile["outliers"] = outlier_signals

    corr_signals: Dict[str, Any] = {"top_abs_pairs": [], "max_abs_corr": None}
    corr_cols = numeric_cols[:max_corr_numeric_cols]
    if len(corr_cols) >= 2:
        work_df = df[corr_cols]
        if len(work_df) > corr_sample_rows:
            work_df = work_df.sample(n=corr_sample_rows, random_state=42)

        try:
            corr = work_df.corr(numeric_only=True)
            cols_list = list(corr.columns)
            pairs: List[Tuple[str, str, float]] = []
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    val = corr.iloc[i, j]
                    if pd.isna(val):
                        continue
                    pairs.append((cols_list[i], cols_list[j], float(val)))

            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            top_pairs = pairs[:top_k]
            corr_signals["top_abs_pairs"] = [{"col_x": a, "col_y": b, "corr": float(np.round(v, 4))} for a, b, v in top_pairs]
            if top_pairs:
                corr_signals["max_abs_corr"] = float(np.round(abs(top_pairs[0][2]), 4))
        except Exception:
            pass

    profile["correlation"] = corr_signals
    return profile