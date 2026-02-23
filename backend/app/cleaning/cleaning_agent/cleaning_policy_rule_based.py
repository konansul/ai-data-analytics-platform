# backend/app/cleaning_agent/cleaning_policy_rule_based.py
from __future__ import annotations

from typing import Any, Dict, List
import re

from .schemas import CleaningPlan
from .cleaning_policy_utils import (
    _get_int,
    _get_float,
    _safe_int,
    _estimate_overall_missing_pct,
)


def build_cleaning_plan_rule_based(pre_profile: Dict[str, Any]) -> CleaningPlan:
    rows = _get_int(pre_profile, ["rows", "n_rows", "rows_before"], default=0)
    cols = _get_int(pre_profile, ["cols", "n_cols", "cols_before"], default=0)

    missing_overall = _get_float(
        pre_profile,
        keys=[
            "missing_overall",
            "overall_missing_pct",
            "overall_missing_%",
            "overall_missing_percent",
        ],
        default=None,
    )

    missing_fraction = (
        pre_profile.get("missing_fraction")
        or pre_profile.get("missing_fraction_before")
        or pre_profile.get("missing_fraction_mid")
        or {}
    )

    counts = (
        pre_profile.get("counts")
        or pre_profile.get("column_groups")
        or pre_profile.get("column_group_counts")
        or {}
    )
    n_numeric = _safe_int(counts.get("numeric", 0))
    n_datetime = _safe_int(counts.get("datetime", 0))
    n_boolean = _safe_int(counts.get("boolean", 0))
    n_categorical = _safe_int(counts.get("categorical", 0))

    dataset_type = str(pre_profile.get("dataset_type", "")).lower()
    has_time_index = bool(pre_profile.get("has_time_index", False)) or (dataset_type == "time_series")
    time_column = pre_profile.get("time_column")

    if missing_overall is None:
        missing_overall = _estimate_overall_missing_pct(missing_fraction)

    default_plan = CleaningPlan.default()
    enabled_steps = dict(default_plan.enabled_steps)
    params = dict(default_plan.params)
    notes: List[str] = []

    enabled_steps["normalize"] = True
    enabled_steps["trim_strings"] = True
    enabled_steps["standardize_missing"] = True
    enabled_steps["cast_types"] = True
    enabled_steps["encode_booleans"] = True
    enabled_steps["drop_rules"] = True

    if (not has_time_index) and (n_datetime == 0):
        enabled_steps["datetime_inference"] = False
        notes.append("No time index / datetime columns detected → disable datetime inference.")
    else:
        enabled_steps["datetime_inference"] = True
        if time_column:
            notes.append(f"Time column detected: '{time_column}'.")

    enabled_steps["deduplicate"] = True
    if rows <= 5:
        notes.append("Very few rows → deduplicate enabled but impact likely minimal.")

    if n_numeric >= 1 and rows >= 30:
        enabled_steps["outliers"] = True
        notes.append("Numeric columns + enough rows → enable outliers handling.")
    else:
        enabled_steps["outliers"] = False
        notes.append("Not enough numeric signal/rows → disable outliers handling.")

    IMPUTE_MIN_MISSING_PCT = 0.0
    if float(missing_overall) < IMPUTE_MIN_MISSING_PCT:
        enabled_steps["impute_missing"] = False
        notes.append(
            f"Overall missingness is low ({missing_overall:.2f}%) < {IMPUTE_MIN_MISSING_PCT:.1f}% → disable imputation."
        )
    else:
        enabled_steps["impute_missing"] = True
        notes.append(
            f"Overall missingness is {missing_overall:.2f}% ≥ {IMPUTE_MIN_MISSING_PCT:.1f}% → enable imputation."
        )

    if rows < 10:
        notes.append("Too few rows (<10). Cleaning works, but statistics may be unreliable.")

    params["missing_threshold"] = 0.4 if cols >= 40 else 0.5
    if cols >= 40:
        notes.append("Many columns detected → using missing_threshold=0.4.")

    params["row_missing_threshold"] = 0.80
    params["drop_rows"] = True if cols >= 5 else False
    params["ignore_columns_for_row_drop"] = []

    if cols < 5:
        notes.append("Too few columns → disable dropping rows by missingness (drop_rows=false).")

    params["datetime_success_ratio"] = 0.7 if has_time_index else 0.8

    skew_top = pre_profile.get("skewness_top_abs") or pre_profile.get("skewness_top")
    if skew_top:
        params["numeric_strategy"] = "median"
        notes.append("Skewness detected → numeric_strategy='median'.")
    else:
        params["numeric_strategy"] = "mean"

    params["categorical_strategy"] = "mode"
    params["datetime_strategy"] = None
    params["fill_value"] = 0

    params["categorical_numeric_max_unique"] = 30 if n_categorical > 20 else 20
    if n_categorical > 20:
        notes.append("Many categorical columns → categorical_numeric_max_unique=30.")

    if enabled_steps["outliers"]:
        params["outliers_method"] = "iqr"
        params["outliers_action"] = "clip"
        params["iqr_k"] = 1.5
        params["zscore_threshold"] = 3.0
    else:
        params["outliers_method"] = "none"
        params["outliers_action"] = "none"
        params["iqr_k"] = 1.5
        params["zscore_threshold"] = 3.0

    if n_boolean > 0:
        notes.append(f"Boolean columns detected: {n_boolean}.")

    snippet = pre_profile.get("table_snippet") or {}
    snippet_rows = snippet.get("rows") or []

    def _looks_like_year(x: Any) -> bool:
        try:
            s = str(x).strip().replace("*", "")
            if len(s) != 4:
                return False
            y = int(s)
            return 1900 <= y <= 2100
        except Exception:
            return False

    def _letters_ratio(vals: List[Any]) -> float:
        if not vals:
            return 0.0
        rx = re.compile(r"[A-Za-zА-Яа-яƏəÖöÜüĞğÇçŞşıİ]")
        hits = 0
        for v in vals:
            if rx.search(str(v)):
                hits += 1
        return hits / max(len(vals), 1)

    is_static_table = False
    static_reasoning = ""
    static_rebuild_recommended = False

    if (
        rows > 0
        and cols > 0
        and rows <= 40
        and cols >= 8
        and isinstance(snippet_rows, list)
        and len(snippet_rows) >= 2
        and isinstance(snippet_rows[0], list)
    ):
        header = snippet_rows[0]
        header_tail = header[1:] if len(header) > 1 else []
        year_cnt = sum(1 for v in header_tail if _looks_like_year(v))
        year_ratio = year_cnt / max(len(header_tail), 1)

        first_col_vals = []
        for r in snippet_rows[1:]:
            if isinstance(r, list) and len(r) > 0:
                first_col_vals.append(r[0])

        first_col_letters_ratio = _letters_ratio(first_col_vals)

        if year_cnt >= 3 and year_ratio >= 0.35 and first_col_letters_ratio >= 0.50:
            is_static_table = True
            static_rebuild_recommended = True
            static_reasoning = (
                f"Header looks like years (years={year_cnt}, ratio={year_ratio:.2f}); "
                f"first column looks like labels (letters_ratio={first_col_letters_ratio:.2f}); "
                f"shape={rows}x{cols}."
            )
            notes.append("Static table layout detected → recommend LLM rebuild to tidy CSV before cleaning.")

    return CleaningPlan(
        enabled_steps=enabled_steps,
        params=params,
        notes=notes,
        source="rule_based",
        version=2,
        is_static_table=is_static_table,
        static_reasoning=static_reasoning,
        static_rebuild_recommended=static_rebuild_recommended,
    )