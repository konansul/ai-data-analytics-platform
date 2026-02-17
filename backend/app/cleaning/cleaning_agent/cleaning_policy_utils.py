# backend/app/cleaning_agent/cleaning_policy_utils.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .schemas import CleaningPlan


def _estimate_overall_missing_pct(missing_fraction: Dict[str, Any]) -> float:
    if not isinstance(missing_fraction, dict) or not missing_fraction:
        return 0.0

    vals: List[float] = []
    for v in missing_fraction.values():
        try:
            fv = float(v)
            if 0.0 <= fv <= 1.0:
                fv *= 100.0
            vals.append(fv)
        except Exception:
            continue

    return (sum(vals) / len(vals)) if vals else 0.0


def _get_int(d: Dict[str, Any], keys: List[str], default: int = 0) -> int:
    for k in keys:
        if k in d:
            return _safe_int(d.get(k), default=default)
    return default


def _get_float(d: Dict[str, Any], keys: List[str], default: Optional[float] = None) -> Optional[float]:
    for k in keys:
        if k in d:
            try:
                return float(d.get(k))
            except Exception:
                return default
    return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _clamp_float(x: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if v < lo or v > hi:
        return float(default)
    return float(v)


def _clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
    except Exception:
        return int(default)
    if v < lo or v > hi:
        return int(default)
    return int(v)


def _as_bool(x: Any, default: bool) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "0", "no", "n", "off"}:
            return False
    return default


def _as_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    return [str(x)]


def _sanitize_plan(plan: CleaningPlan) -> CleaningPlan:
    """
    Final safety layer after validate_plan_dict:
    - clamp numeric thresholds
    - enforce allowed enums
    - coerce booleans/lists
    - keep params consistent with enabled_steps (especially impute)
    """
    defaults = CleaningPlan.default().params

    plan.params["missing_threshold"] = _clamp_float(
        plan.params.get("missing_threshold"),
        lo=0.10,
        hi=0.90,
        default=float(defaults["missing_threshold"]),
    )

    plan.params["row_missing_threshold"] = _clamp_float(
        plan.params.get("row_missing_threshold"),
        lo=0.50,
        hi=0.99,
        default=float(defaults.get("row_missing_threshold", 0.80)),
    )

    plan.params["drop_rows"] = _as_bool(
        plan.params.get("drop_rows"),
        default=bool(defaults.get("drop_rows", True)),
    )

    plan.params["ignore_columns_for_row_drop"] = _as_str_list(
        plan.params.get("ignore_columns_for_row_drop", defaults.get("ignore_columns_for_row_drop", []))
    )

    plan.params["datetime_success_ratio"] = _clamp_float(
        plan.params.get("datetime_success_ratio"),
        lo=0.50,
        hi=0.99,
        default=float(defaults["datetime_success_ratio"]),
    )

    allowed_numeric = {"mean", "median", "constant", None}
    if plan.params.get("numeric_strategy") not in allowed_numeric:
        plan.params["numeric_strategy"] = defaults["numeric_strategy"]

    allowed_cat = {"mode", "constant", None}
    if plan.params.get("categorical_strategy") not in allowed_cat:
        plan.params["categorical_strategy"] = defaults["categorical_strategy"]

    allowed_dt = {"ffill", "bfill", None}
    if plan.params.get("datetime_strategy") not in allowed_dt:
        plan.params["datetime_strategy"] = defaults["datetime_strategy"]

    plan.params["categorical_numeric_max_unique"] = _clamp_int(
        plan.params.get("categorical_numeric_max_unique", defaults["categorical_numeric_max_unique"]),
        lo=2,
        hi=10_000,
        default=int(defaults["categorical_numeric_max_unique"]),
    )

    plan.params["impute"] = bool(plan.enabled_steps.get("impute_missing", True))

    allowed_outliers_method = {"iqr", "zscore", "none", None}
    if plan.params.get("outliers_method") not in allowed_outliers_method:
        plan.params["outliers_method"] = defaults.get("outliers_method", "iqr")

    allowed_outliers_action = {"clip", "remove", "none", None}
    if plan.params.get("outliers_action") not in allowed_outliers_action:
        plan.params["outliers_action"] = defaults.get("outliers_action", "clip")

    plan.params["iqr_k"] = _clamp_float(
        plan.params.get("iqr_k"),
        lo=0.5,
        hi=10.0,
        default=float(defaults.get("iqr_k", 1.5)),
    )

    plan.params["zscore_threshold"] = _clamp_float(
        plan.params.get("zscore_threshold"),
        lo=2.0,
        hi=10.0,
        default=float(defaults.get("zscore_threshold", 3.0)),
    )

    if not bool(plan.enabled_steps.get("outliers", True)):
        plan.params["outliers_method"] = "none"
        plan.params["outliers_action"] = "none"

    return plan