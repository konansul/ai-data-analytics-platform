# backend/app/forecasting/helpers.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from backend.api.models import (
    ForecastSignalsResponse,
    ForecastTargetIn,
    ForecastPlanResponse,
)

def to_agent_signals(*, df: pd.DataFrame, signals: ForecastSignalsResponse) -> Dict[str, Any]:
    dt_candidates: List[Dict[str, Any]] = []
    for c in (signals.datetime_candidates or []):
        dt_candidates.append(
            {
                "column": c.get("column"),
                "success_ratio": float(c.get("success_ratio", 0.0) or 0.0),
                "letters_ratio": float(c.get("letters_ratio", 0.0) or 0.0),
            }
        )

    tgt_candidates = [{"column": col, "score": 1.0} for col in (signals.numeric_target_candidates or [])]

    grp_candidates: List[Dict[str, Any]] = []
    for col in (signals.grouping_candidates or []):
        card: Optional[int]
        try:
            card = int(df[col].nunique(dropna=True))
        except Exception:
            card = None
        grp_candidates.append({"column": col, "cardinality": card, "score": 1.0})

    reason = ""
    feasible = bool(signals.feasible)
    if not feasible:
        if not dt_candidates:
            reason = "No datetime candidates."
        elif not tgt_candidates:
            reason = "No numeric target candidates."
        else:
            reason = "Forecast not feasible by heuristic."

    return {
        "forecast_feasible": feasible,
        "reason_if_not_feasible": reason,
        "datetime_candidates": dt_candidates,
        "target_candidates": tgt_candidates,
        "grouping_candidates": grp_candidates,
        "inferred_frequency": signals.inferred_frequency or "unknown",
    }

def try_llm_planning(
    *,
    dataset_id: str,
    df: pd.DataFrame,
    df_head: pd.DataFrame,
    signals: ForecastSignalsResponse,
    profile: Optional[Dict[str, Any]],
    user_intent: Optional[str],
    max_targets: int,
    head_rows: int,
    horizon: int,
) -> ForecastPlanResponse:
    try:
        from backend.app.forecasting.planning_agent import ForecastPlanningAgent
    except Exception as e:
        fb = fallback_plan(
            dataset_id=dataset_id,
            df=df,
            signals=signals,
            max_targets=max_targets,
            horizon=horizon,
        )
        fb.reasoning = f"Fallback: could not import ForecastPlanningAgent: {e}"
        fb.reasons = [fb.reasoning]
        fb.planner_source = "fallback"
        fb.llm_model = None
        return fb

    agent_signals = to_agent_signals(df=df, signals=signals)

    report: Dict[str, Any] = {}
    if isinstance(profile, dict) and profile:
        report = {"post_profile": profile}

    head_rows = max(1, min(int(head_rows), 50))

    try:
        agent = ForecastPlanningAgent()
        plan_obj = agent.create_plan(
            dataset_id=dataset_id,
            df=df,
            report=report,
            forecast_signals=agent_signals,
            user_intent=user_intent,
            max_targets=int(max_targets),
            head_rows=int(head_rows),
            horizon=int(horizon),
        )

        plan_dict = plan_obj.model_dump() if hasattr(plan_obj, "model_dump") else dict(plan_obj)  # type: ignore

        targets_in: List[ForecastTargetIn] = []
        for t in plan_dict.get("targets") or []:
            targets_in.append(
                ForecastTargetIn(
                    column=t.get("column"),
                    horizon=int(t.get("horizon", horizon) or horizon),
                )
            )

        resp = ForecastPlanResponse(
            dataset_id=dataset_id,
            suitable=bool(plan_dict.get("suitable")),
            mode=plan_dict.get("mode") or "skipped",
            datetime_column=plan_dict.get("datetime_column"),
            inferred_frequency=plan_dict.get("inferred_frequency") or signals.inferred_frequency,
            group_by=plan_dict.get("group_by"),
            targets=targets_in,
            planner_source="llm",
            llm_model=getattr(getattr(agent, "llm", None), "model", None)
            or getattr(getattr(agent, "llm", None), "model_name", None),
            reasoning=plan_dict.get("reasoning") or "",
            reasons=[],
        )

        if not resp.suitable:
            resp.mode = "skipped"
            resp.datetime_column = None
            resp.targets = []
            resp.group_by = None

        if resp.mode == "grouped" and not resp.group_by:
            resp.mode = "overall"

        return resp

    except Exception as e:
        fb = fallback_plan(
            dataset_id=dataset_id,
            df=df,
            signals=signals,
            max_targets=max_targets,
            horizon=horizon,
        )
        fb.planner_source = "fallback"
        fb.llm_model = None
        fb.reasoning = f"Fallback: LLM planning failed: {e}"
        fb.reasons = [fb.reasoning]
        return fb


def fallback_plan(
    *,
    dataset_id: str,
    df: pd.DataFrame,
    signals: ForecastSignalsResponse,
    max_targets: int,
    horizon: int,
) -> ForecastPlanResponse:
    reasons: List[str] = []

    if not signals.datetime_candidates:
        reasons.append("No datetime column candidates found. Forecasting skipped.")
        return ForecastPlanResponse(
            dataset_id=dataset_id,
            suitable=False,
            mode="skipped",
            planner_source="fallback",
            llm_model=None,
            reasoning="; ".join(reasons),
            reasons=reasons,
        )

    best = sorted(
        signals.datetime_candidates,
        key=lambda x: (float(x.get("success_ratio", 0.0) or 0.0), -float(x.get("letters_ratio", 1.0) or 1.0)),
        reverse=True,
    )[0]
    dt_col = best.get("column")

    if float(best.get("success_ratio", 0.0) or 0.0) < 0.8:
        reasons.append(f"Best datetime candidate '{dt_col}' has low parse success ratio. Forecasting skipped.")
        return ForecastPlanResponse(
            dataset_id=dataset_id,
            suitable=False,
            mode="skipped",
            datetime_column=dt_col,
            inferred_frequency=signals.inferred_frequency,
            planner_source="fallback",
            llm_model=None,
            reasoning="; ".join(reasons),
            reasons=reasons,
        )

    if not signals.numeric_target_candidates:
        reasons.append("No numeric target candidates found. Forecasting skipped.")
        return ForecastPlanResponse(
            dataset_id=dataset_id,
            suitable=False,
            mode="skipped",
            datetime_column=dt_col,
            inferred_frequency=signals.inferred_frequency,
            planner_source="fallback",
            llm_model=None,
            reasoning="; ".join(reasons),
            reasons=reasons,
        )

    targets = [
        ForecastTargetIn(column=c, horizon=int(horizon))
        for c in (signals.numeric_target_candidates or [])[: int(max_targets)]
    ]

    mode: Literal["overall", "grouped", "skipped"] = "overall"
    group_by: Optional[str] = None

    if signals.grouping_candidates:
        g = (signals.grouping_candidates or [None])[0]
        if g and g in df.columns:
            try:
                card = int(df[g].nunique(dropna=True))
            except Exception:
                card = 999999
            if 2 <= card <= 30:
                group_by = g
                mode = "grouped"

    return ForecastPlanResponse(
        dataset_id=dataset_id,
        suitable=True,
        mode=mode,
        datetime_column=dt_col,
        inferred_frequency=signals.inferred_frequency,
        group_by=group_by,
        targets=targets,
        planner_source="fallback",
        llm_model=None,
        reasoning="Fallback plan (deterministic).",
        reasons=[],
    )


def datetime_parse_success_ratio(s: pd.Series) -> float:
    if s.empty:
        return 0.0
    parsed = pd.to_datetime(s, errors="coerce", utc=False)
    return float(parsed.notna().mean())


def letters_ratio(s: pd.Series) -> float:
    if s.empty:
        return 0.0
    ss = s.astype(str)
    letters = ss.str.contains(r"[A-Za-zА-Яа-яƏəÖöÜüĞğÇçŞşıİ]", regex=True, na=False)
    return float(letters.mean())


def infer_freq_label(dt_index: pd.DatetimeIndex) -> str:
    try:
        f = pd.infer_freq(dt_index)
    except Exception:
        f = None
    if not f:
        return "irregular"

    f = f.upper()
    if f.startswith("D"):
        return "daily"
    if f.startswith("W"):
        return "weekly"
    if f.startswith("M"):
        return "monthly"
    if f.startswith("Q"):
        return "quarterly"
    if f.startswith("A") or f.startswith("Y"):
        return "yearly"
    return "irregular"


_ID_RE = re.compile(r"(^|[^a-z0-9])id([^a-z0-9]|$)", re.IGNORECASE)


def _is_id_like(col: str) -> bool:
    c = col.strip().lower()
    if c in {"index", "row"}:
        return True
    if c.startswith("unnamed"):
        return True
    if _ID_RE.search(c):
        return True
    if "uuid" in c or "guid" in c:
        return True
    return False


def numeric_target_candidates(df: pd.DataFrame) -> List[str]:
    numeric = df.select_dtypes(include=[np.number]).copy()

    cands: List[str] = []
    for col in numeric.columns:
        s = numeric[col]
        nunique = int(s.nunique(dropna=True))
        notna = float(s.notna().mean())
        if _is_id_like(col):
            continue
        if nunique <= 1:
            continue
        if notna < 0.2:
            continue
        cands.append(col)

    return cands


def grouping_candidates(df: pd.DataFrame, max_cardinality: int = 50) -> List[str]:
    cands: List[str] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        nunique = int(df[col].nunique(dropna=True))
        if 2 <= nunique <= max_cardinality:
            cands.append(col)
    return cands

