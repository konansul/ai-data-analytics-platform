# backend/app/forecasting/signals.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import re

from backend.app.forecasting.schemas import DatetimeCandidate, Frequency, TargetCandidate, GroupingCandidate, ForecastSignals

_ID_LIKE_RE = re.compile(
    r"(?:^|_|\b)(id|idx|index|uuid|guid|code|key|номер|nomer|kod)(?:$|_|\b)",
    flags=re.IGNORECASE,
)


def _get_profile(report: Dict[str, Any], prefer_post: bool = True) -> Tuple[str, Dict[str, Any]]:
    if prefer_post and isinstance(report.get("post_profile"), dict):
        return "post_profile", report["post_profile"]
    if isinstance(report.get("pre_profile"), dict):
        return "pre_profile", report["pre_profile"]
    return "post_profile", {}


def _safe_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return []


def _column_exists(df: Optional[pd.DataFrame], col: str) -> bool:
    return df is not None and col in df.columns


def _looks_like_year_column_name(col: str) -> bool:
    s = str(col).strip()
    if re.fullmatch(r"\d{4}(\.0)?", s):
        year = int(float(s))
        return 1900 <= year <= 2100
    return False


def _is_id_like(col: str, df: Optional[pd.DataFrame]) -> bool:
    if _ID_LIKE_RE.search(col or ""):
        return True
    if str(col).lower().startswith("unnamed"):
        return True

    if df is not None and col in df.columns:
        s = df[col]
        try:
            nunique = s.nunique(dropna=True)
            n = len(s)
            if n >= 50 and nunique / max(n, 1) > 0.98:
                return True
        except Exception:
            pass
    return False


def _is_constant(col: str, df: Optional[pd.DataFrame]) -> bool:
    if df is None or col not in df.columns:
        return False
    try:
        nunique = df[col].nunique(dropna=True)
        return nunique <= 1
    except Exception:
        return False


def _infer_frequency_from_series(dt: pd.Series) -> Frequency:

    dt = pd.to_datetime(dt, errors="coerce")
    dt = dt.dropna().sort_values()
    if len(dt) < 10:
        return "unknown"

    deltas = dt.diff().dropna()
    if deltas.empty:
        return "unknown"

    days = deltas.dt.total_seconds() / 86400.0
    med = float(np.median(days))
    if np.std(days) > 0.5 * max(med, 1e-9):
        return "irregular"

    if 0.8 <= med <= 1.2:
        return "daily"
    if 6.0 <= med <= 8.0:
        return "weekly"
    if 25.0 <= med <= 35.0:
        return "monthly"
    if 80.0 <= med <= 100.0:
        return "quarterly"
    if 340.0 <= med <= 390.0:
        return "yearly"
    return "irregular"


def _rank_datetime_candidates(profile: Dict[str, Any], df: Optional[pd.DataFrame], ) -> List[DatetimeCandidate]:

    candidates: List[DatetimeCandidate] = []

    for item in _safe_list(profile.get("datetime_candidates")):
        col = str(item.get("column", "")).strip()
        if not col:
            continue
        success_ratio = float(item.get("success_ratio", 0.0) or 0.0)
        notes = []
        score = 0.0

        score += 2.0 * success_ratio
        if success_ratio >= 0.95:
            score += 1.0
            notes.append("high parse success ratio")
        elif success_ratio >= 0.8:
            notes.append("ok parse success ratio")
        else:
            notes.append("low parse success ratio")

        if _column_exists(df, col):
            parsed = pd.to_datetime(df[col], errors="coerce")
            non_null = parsed.notna().mean() if len(parsed) else 0.0
            score += 2.0 * float(non_null)
            if non_null >= 0.95:
                notes.append("df confirms datetime parse")
            else:
                notes.append(f"df parse coverage {non_null:.2f}")

            try:
                srt = parsed.dropna().sort_values()
                if len(srt) >= 10:
                    is_mono = srt.is_monotonic_increasing or srt.is_monotonic_decreasing
                    if is_mono:
                        score += 0.5
                        notes.append("monotonic datetime")
            except Exception:
                pass

        candidates.append(DatetimeCandidate(column=col, score=score, success_ratio=success_ratio, notes=notes))

    for col in _safe_list((profile.get("columns") or {}).get("datetime")):
        col = str(col).strip()
        if not col:
            continue
        if any(c.column == col for c in candidates):
            continue
        score = 1.0
        notes = ["profile classified as datetime"]
        if _column_exists(df, col):
            parsed = pd.to_datetime(df[col], errors="coerce")
            non_null = parsed.notna().mean() if len(parsed) else 0.0
            score += 2.0 * float(non_null)
            if non_null >= 0.95:
                notes.append("df confirms datetime parse")
        candidates.append(DatetimeCandidate(column=col, score=score, success_ratio=0.0, notes=notes))

    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates


def _rank_targets(profile: Dict[str, Any], df: Optional[pd.DataFrame]) -> List[TargetCandidate]:
    numeric_cols = _safe_list((profile.get("columns") or {}).get("numeric"))

    if df is not None and not numeric_cols:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    out: List[TargetCandidate] = []
    for col in numeric_cols:
        col = str(col).strip()
        if not col:
            continue
        notes: List[str] = []
        score = 0.0

        score += 1.0

        if _is_id_like(col, df):
            notes.append("id-like -> deprioritized")
            score -= 2.0

        if _is_constant(col, df):
            notes.append("constant -> excluded")
            continue

        if _column_exists(df, col):
            s = pd.to_numeric(df[col], errors="coerce")
            miss = float(s.isna().mean()) if len(s) else 1.0
            score += 1.0 * (1.0 - miss)
            if miss > 0.3:
                notes.append(f"high missingness {miss:.2f}")

            try:
                std = float(np.nanstd(s.values))
                if std > 0:
                    score += 0.5
                    notes.append("has variability")
            except Exception:
                pass

        out.append(TargetCandidate(column=col, score=score, notes=notes))

    out.sort(key=lambda x: x.score, reverse=True)
    return out


def _rank_groupings(profile: Dict[str, Any], df: Optional[pd.DataFrame]) -> List[GroupingCandidate]:
    cat_cols = _safe_list((profile.get("columns") or {}).get("categorical"))
    cardinality_list = _safe_list(profile.get("categorical_cardinality"))

    cardinality_map: Dict[str, int] = {}
    for item in cardinality_list:
        try:
            c = str(item.get("column", "")).strip()
            u = int(item.get("unique_values"))
            if c:
                cardinality_map[c] = u
        except Exception:
            continue

    out: List[GroupingCandidate] = []
    for col in cat_cols:
        col = str(col).strip()
        if not col:
            continue

        if _looks_like_year_column_name(col):
            continue

        notes: List[str] = []
        score = 0.0

        card = cardinality_map.get(col)
        if card is None and _column_exists(df, col):
            try:
                card = int(df[col].nunique(dropna=True))
            except Exception:
                card = None

        if card is not None:
            if card < 2:
                continue
            if card > 50:
                notes.append(f"high cardinality ({card}) -> excluded")
                continue
            if card <= 15:
                score += 1.5
                notes.append("low cardinality")
            elif card <= 30:
                score += 1.0
                notes.append("medium cardinality")
            else:
                score += 0.4
                notes.append("borderline cardinality")

        if _is_id_like(col, df):
            notes.append("id-like -> excluded")
            continue

        out.append(GroupingCandidate(column=col, cardinality=card, score=score, notes=notes))

    out.sort(key=lambda x: x.score, reverse=True)
    return out


def build_forecast_signals(
    report: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
    *,
    prefer_post_profile: bool = True,
    max_datetime_candidates: int = 5,
    max_target_candidates: int = 8,
    max_grouping_candidates: int = 5,
) -> ForecastSignals:

    profile_name, profile = _get_profile(report, prefer_post=prefer_post_profile)

    dt_candidates = _rank_datetime_candidates(profile, df)
    dt_candidates = dt_candidates[: max(1, max_datetime_candidates)]

    feasible = False
    reason = None
    if not dt_candidates:
        feasible = False
        reason = "No datetime candidates found in profile (forecasting not applicable)."
    else:
        best = dt_candidates[0]
        if best.success_ratio >= 0.8 or best.score >= 2.0:
            feasible = True
        else:
            feasible = False
            reason = "Datetime candidate(s) exist but parse confidence is too low."

    freq: Frequency = "unknown"
    if feasible and df is not None:
        best_col = dt_candidates[0].column
        if best_col in df.columns:
            freq = _infer_frequency_from_series(df[best_col])

    targets = _rank_targets(profile, df)[: max(1, max_target_candidates)]
    groups = _rank_groupings(profile, df)[: max(0, max_grouping_candidates)]

    if feasible and not targets:
        feasible = False
        reason = "No numeric target candidates found (nothing to forecast)."

    return ForecastSignals(
        forecast_feasible=feasible,
        reason_if_not_feasible=reason,
        datetime_candidates=dt_candidates,
        inferred_frequency=freq,
        target_candidates=targets,
        grouping_candidates=groups,
        source_profile=profile_name,
    )