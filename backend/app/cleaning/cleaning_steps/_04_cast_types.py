# backend/app/cleaning/_04_cast_types.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


_TRUE_TOKENS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_TOKENS = {"0", "false", "f", "no", "n", "off"}


def _looks_like_money(text: str) -> bool:
    t = text.lower()
    return ("$" in t) or ("usd" in t) or ("eur" in t) or ("gbp" in t) or ("₼" in t) or ("₽" in t) or ("€" in t) or ("£" in t)


def _to_money_numeric(s: pd.Series) -> pd.Series:
    x = s.astype("string")
    x = x.str.replace(r"[\s\u00A0]", "", regex=True)
    x = x.str.replace(r"(?i)\b(usd|eur|gbp|azn|rub)\b", "", regex=True)
    x = x.str.replace(r"[$€£₼₽]", "", regex=True)
    has_dot = x.str.contains(r"\.", na=False)
    has_comma = x.str.contains(r",", na=False)
    x = x.where(~(has_dot & has_comma), x.str.replace(",", "", regex=False))
    x = x.where(~(~has_dot & has_comma), x.str.replace(",", ".", regex=False))
    return pd.to_numeric(x, errors="coerce")


def _to_percent_numeric(s: pd.Series) -> pd.Series:
    x = s.astype("string")
    x = x.str.replace(r"[\s\u00A0]", "", regex=True)
    x = x.str.replace("%", "", regex=False)
    has_dot = x.str.contains(r"\.", na=False)
    has_comma = x.str.contains(r",", na=False)
    x = x.where(~(has_dot & has_comma), x.str.replace(",", "", regex=False))
    x = x.where(~(~has_dot & has_comma), x.str.replace(",", ".", regex=False))
    return pd.to_numeric(x, errors="coerce")


def _to_bool_nullable(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip().str.lower()
    mapped = x.map(lambda v: True if v in _TRUE_TOKENS else (False if v in _FALSE_TOKENS else pd.NA))
    return mapped.astype("boolean")


def _normalize_gender(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip().str.lower()
    x = x.replace(
        {
            "m": "male",
            "male": "male",
            "man": "male",
            "f": "female",
            "female": "female",
            "woman": "female",
        }
    )
    x = x.replace({"": pd.NA})
    return x


def cast_types(
    df: pd.DataFrame,
    *,
    enabled: bool = True,
    numeric_success_ratio: float = 0.90,
    integer_if_possible_ratio: float = 0.99,
    max_unique_for_category: int = 50,
    convert_categories: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    report: Dict[str, Any] = {
        "enabled": bool(enabled),
        "numeric_success_ratio": float(numeric_success_ratio),
        "integer_if_possible_ratio": float(integer_if_possible_ratio),
        "max_unique_for_category": int(max_unique_for_category),
        "converted_to_numeric": [],
        "converted_to_int": [],
        "converted_to_category": [],
        "parsed_money_columns": [],
        "parsed_percent_columns": [],
        "parsed_boolean_columns": [],
        "normalized_enum_columns": [],
    }

    if not enabled:
        return df, report

    clean_df = df.copy()

    for col in clean_df.columns:
        s = clean_df[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue

        col_l = str(col).strip().lower()

        non_null = s.dropna()
        if non_null.empty:
            continue

        sample = non_null.astype("string").head(200)

        if ("satisfaction" in col_l) or ("percent" in col_l) or ("pct" in col_l) or sample.str.contains(r"%\s*$", na=False).any():
            parsed = _to_percent_numeric(s)
            ok = int(parsed.notna().sum())
            total = int(s.notna().sum()) or 1
            if (ok / total) >= 0.60:
                clean_df[col] = parsed
                report["parsed_percent_columns"].append(col)
                report["converted_to_numeric"].append(col)
                continue

        if ("rate" in col_l) or ("price" in col_l) or ("salary" in col_l) or ("cost" in col_l) or ("usd" in col_l) or sample.map(lambda v: _looks_like_money(str(v))).any():
            parsed = _to_money_numeric(s)
            ok = int(parsed.notna().sum())
            total = int(s.notna().sum()) or 1
            if (ok / total) >= 0.60:
                clean_df[col] = parsed
                report["parsed_money_columns"].append(col)
                report["converted_to_numeric"].append(col)
                continue

        if ("is_" in col_l) or col_l.startswith("has_") or col_l.endswith("_flag") or col_l.endswith("_bool") or ("active" in col_l):
            parsed_bool = _to_bool_nullable(s)
            ok = int(parsed_bool.notna().sum())
            total = int(s.notna().sum()) or 1
            if (ok / total) >= 0.60:
                clean_df[col] = parsed_bool
                report["parsed_boolean_columns"].append(col)
                continue

        if col_l in {"gender", "sex"}:
            clean_df[col] = _normalize_gender(s)
            report["normalized_enum_columns"].append(col)

    for col in clean_df.columns:
        s = clean_df[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue

        non_null = s.dropna()
        if non_null.empty:
            continue

        numeric = pd.to_numeric(s, errors="coerce")
        ok = int(numeric.notna().sum())
        total = int(s.notna().sum())
        if total == 0:
            continue

        if (ok / total) >= float(numeric_success_ratio):
            clean_df[col] = numeric
            if col not in report["converted_to_numeric"]:
                report["converted_to_numeric"].append(col)

    for col in clean_df.columns:
        s = clean_df[col]

        if pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean":
            continue

        if not pd.api.types.is_numeric_dtype(s):
            continue

        non_null = s.dropna()
        if non_null.empty:
            continue

        arr = pd.to_numeric(non_null, errors="coerce").to_numpy(dtype=float)
        if arr.size == 0 or np.isnan(arr).all():
            continue

        rounded = np.rint(arr)
        is_int_like = np.isclose(arr, rounded, atol=1e-9, rtol=0.0)
        ratio = float(np.nanmean(is_int_like))

        if ratio >= float(integer_if_possible_ratio):
            try:
                clean_df[col] = pd.to_numeric(s, errors="coerce").round().astype("Int64")
                report["converted_to_int"].append(col)
            except Exception:
                pass

    if convert_categories:
        for col in clean_df.columns:
            s = clean_df[col]
            if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
                continue
            try:
                nunique = int(s.nunique(dropna=True))
            except Exception:
                continue
            if 1 < nunique <= int(max_unique_for_category):
                try:
                    clean_df[col] = s.astype("category")
                    report["converted_to_category"].append(col)
                except Exception:
                    pass

    report["converted_to_numeric"] = sorted(set(report["converted_to_numeric"]))
    report["converted_to_int"] = sorted(set(report["converted_to_int"]))
    report["converted_to_category"] = sorted(set(report["converted_to_category"]))
    report["parsed_money_columns"] = sorted(set(report["parsed_money_columns"]))
    report["parsed_percent_columns"] = sorted(set(report["parsed_percent_columns"]))
    report["parsed_boolean_columns"] = sorted(set(report["parsed_boolean_columns"]))
    report["normalized_enum_columns"] = sorted(set(report["normalized_enum_columns"]))

    return clean_df, report