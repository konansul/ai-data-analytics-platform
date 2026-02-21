# backend/app/forecasting/planning_agent.py
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from pydantic import ValidationError

from backend.app.cleaning.cleaning_agent.llm_client import LLMClient
from backend.app.forecasting.schemas import ForecastPlan, ForecastTarget, ForecastMode

logger = logging.getLogger("forecasting.planning_agent")

def _json_safe_rows(df_head: pd.DataFrame) -> List[Dict[str, Any]]:
    safe = df_head.where(pd.notnull(df_head), None)
    for c in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[c]):
            safe[c] = safe[c].apply(lambda x: x.isoformat() if x is not None else None)
    return safe.to_dict(orient="records")


def _pick_profile(report: Dict[str, Any], prefer_post: bool = True) -> Dict[str, Any]:
    if prefer_post and isinstance(report.get("post_profile"), dict):
        return report["post_profile"]
    if isinstance(report.get("pre_profile"), dict):
        return report["pre_profile"]
    return {}


def _compact_profile_for_llm(profile: Dict[str, Any]) -> Dict[str, Any]:

    cols = profile.get("columns") or {}
    out = {
        "n_rows": profile.get("n_rows"),
        "n_cols": profile.get("n_cols"),
        "dataset_type": profile.get("dataset_type"),
        "has_time_index": profile.get("has_time_index"),
        "time_column": profile.get("time_column"),
        "columns": {
            "datetime": cols.get("datetime", []),
            "numeric": cols.get("numeric", []),
            "categorical": cols.get("categorical", []),
            "boolean": cols.get("boolean", []),
        },
        "datetime_candidates": profile.get("datetime_candidates", []),
        "categorical_cardinality": (profile.get("categorical_cardinality") or [])[:25],
        "missingness": (profile.get("missingness") or {}),
        "warnings": (profile.get("warnings") or [])[:15],
    }
    return out


def _get_target_candidate_columns(signals: Dict[str, Any]) -> List[str]:
    out: List[str] = []

    ntc = signals.get("numeric_target_candidates")
    if isinstance(ntc, list) and ntc and all(isinstance(x, (str,)) for x in ntc):
        out = [x.strip() for x in ntc if isinstance(x, str) and x.strip()]
        return out

    tcs = signals.get("target_candidates") or []
    if isinstance(tcs, list):
        for x in tcs:
            if isinstance(x, dict):
                col = x.get("column")
                if isinstance(col, str) and col.strip():
                    out.append(col.strip())
            elif isinstance(x, str) and x.strip():
                out.append(x.strip())

    return out


def _get_datetime_candidate_columns(signals: Dict[str, Any]) -> List[str]:
    dts = signals.get("datetime_candidates") or []
    out: List[str] = []
    if isinstance(dts, list):
        for x in dts:
            if isinstance(x, dict):
                col = x.get("column")
                if isinstance(col, str) and col.strip():
                    out.append(col.strip())
            elif isinstance(x, str) and x.strip():
                out.append(x.strip())
    return out


def _signals_has_any_datetime(signals: Dict[str, Any]) -> bool:
    return len(_get_datetime_candidate_columns(signals)) > 0


def _signals_has_any_targets(signals: Dict[str, Any]) -> bool:
    return len(_get_target_candidate_columns(signals)) > 0


def _force_skip(plan: ForecastPlan, reason: str) -> ForecastPlan:
    plan.suitable = False
    plan.mode = "skipped"
    plan.datetime_column = None
    plan.targets = []
    plan.group_by = None
    if reason:
        plan.reasoning = (plan.reasoning or "").strip()
        if plan.reasoning:
            plan.reasoning += f" | Forced skip: {reason}"
        else:
            plan.reasoning = f"Forced skip: {reason}"
    return plan


def _coerce_plan_horizon(plan: ForecastPlan, horizon: int) -> None:
    h = int(horizon)
    for t in plan.targets:
        t.horizon = h


def _fill_targets_to_max(
    *,
    plan: ForecastPlan,
    signals: Dict[str, Any],
    want: int,
    horizon: int,
) -> None:
    want = max(1, min(int(want), 10))
    cand_cols = _get_target_candidate_columns(signals)

    existing: Set[str] = set()
    cleaned_targets: List[ForecastTarget] = []
    for t in plan.targets:
        col = (t.column or "").strip()
        if not col or col in existing:
            continue
        cleaned_targets.append(t)
        existing.add(col)

    plan.targets = cleaned_targets

    for col in cand_cols:
        if col in existing:
            continue
        plan.targets.append(ForecastTarget(column=col, horizon=int(horizon), notes="auto-filled to max_targets"))
        existing.add(col)
        if len(plan.targets) >= want:
            break

    plan.targets = plan.targets[:want]
    _coerce_plan_horizon(plan, horizon=int(horizon))


def _fallback_plan(
    signals: Dict[str, Any],
    *,
    reason: str,
    max_targets: int,
    horizon: int,
) -> ForecastPlan:
    dt_candidates = signals.get("datetime_candidates") or []
    groups = signals.get("grouping_candidates") or []

    dt_cols = _get_datetime_candidate_columns(signals)
    target_cols = _get_target_candidate_columns(signals)

    if not dt_cols:
        return ForecastPlan(
            mode="skipped",
            suitable=False,
            inferred_frequency=signals.get("inferred_frequency") or "unknown",
            reasoning=signals.get("reason_if_not_feasible") or "No datetime candidate found.",
            planner_source="fallback",
            llm_model=None,
            fallback_reason=reason,
        )

    if not target_cols:
        return ForecastPlan(
            mode="skipped",
            suitable=False,
            datetime_column=None,
            inferred_frequency=signals.get("inferred_frequency") or "unknown",
            reasoning=signals.get("reason_if_not_feasible") or "No numeric target candidates found.",
            planner_source="fallback",
            llm_model=None,
            fallback_reason=reason,
        )

    dt = dt_cols[0]

    picked_targets: List[ForecastTarget] = []
    want = max(1, min(int(max_targets), 10))
    for col in target_cols[:want]:
        picked_targets.append(ForecastTarget(column=col, horizon=int(horizon), notes="fallback pick"))

    mode: ForecastMode = "overall"
    group_by = None
    if isinstance(groups, list) and groups:
        g0 = groups[0]

        if isinstance(g0, dict):
            card = g0.get("cardinality")
            score = float(g0.get("score", 0.0) or 0.0)
            col = g0.get("column")
            try:
                card_i = int(card) if card is not None else None
            except Exception:
                card_i = None
            if col and card_i is not None and 2 <= card_i <= 15 and score >= 1.0:
                mode = "grouped"
                group_by = str(col)
        elif isinstance(g0, str) and g0.strip():
            pass

    return ForecastPlan(
        mode=mode,
        suitable=True,
        datetime_column=dt,
        targets=picked_targets,
        group_by=group_by,
        inferred_frequency=signals.get("inferred_frequency") or "unknown",
        reasoning="Fallback plan (LLM failed or invalid).",
        planner_source="fallback",
        llm_model=None,
        fallback_reason=reason,
    )


class ForecastPlanningAgent:
    """
    Decides WHAT to forecast and HOW (overall vs grouped vs skipped). Does not train models.

    - LLM is ALWAYS called.
    - Deterministic logic is used for:
      - enforcing hard constraints
      - filling targets to max_targets
      - forcing horizon to the planning horizon
      - fallback if LLM fails / invalid JSON
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        self.llm = LLMClient.from_env(model=model)

    def create_plan(
        self,
        *,
        dataset_id: str,
        df: pd.DataFrame,
        report: Dict[str, Any],
        forecast_signals: Dict[str, Any],
        user_intent: Optional[str] = None,
        max_targets: int = 10,   # NEW: up to 10
        head_rows: int = 10,
        horizon: int = 30,
    ) -> ForecastPlan:
        max_targets = max(1, min(int(max_targets), 10))
        horizon = max(1, min(int(horizon), 3650))

        profile = _pick_profile(report, prefer_post=True)
        compact_profile = _compact_profile_for_llm(profile)

        head_rows = max(1, min(int(head_rows), 50))
        sample = _json_safe_rows(df.head(head_rows))

        signals_have_dt = _signals_has_any_datetime(forecast_signals)
        signals_have_targets = _signals_has_any_targets(forecast_signals)

        prompt = f"""
You are the Forecast Planning Agent in a data system.

Your job:
- Decide if the dataset is suitable for time-series forecasting.
- If suitable, choose:
  1) ONE datetime column
  2) 1..{max_targets} numeric target columns to forecast
  3) Optional grouping dimension (ONLY if it makes sense and has low-to-medium cardinality)
  4) Forecasting mode: "overall" OR "grouped" OR "skipped"

REQUESTED CONTROLS (must respect):
- planning_horizon_steps = {horizon}  (use this horizon for ALL targets you return)
- max_targets = {max_targets}

HARD CONSTRAINTS (must follow):
- Never forecast if there is no valid datetime column.
- If there are no numeric targets to forecast, skip.
- If suitable==false then mode must be "skipped".

Guidance:
- If multiple datetime columns exist, choose the best.
- If multiple numeric targets exist, choose the most forecastable (stable, meaningful, not id-like).
- Grouped forecasting is optional; only choose it if group_by is reasonable (2..30 groups).
- If forecasting is not meaningful (no real signal, too messy, wrong semantics), return "skipped".
- You may use the sample rows ONLY to understand column semantics. Do NOT output sample rows back.

INPUTS:

DATASET_ID:
{dataset_id}

FORECAST SIGNALS:
{json.dumps(forecast_signals, ensure_ascii=False, indent=2)}

DATASET PROFILE (compact):
{json.dumps(compact_profile, ensure_ascii=False, indent=2)}

SAMPLE ROWS (first {head_rows}):
{json.dumps(sample, ensure_ascii=False, indent=2)}

USER INTENT:
{user_intent or "None"}

OUTPUT:
Return ONLY valid JSON in this exact shape:

{{
  "suitable": true/false,
  "mode": "overall" | "grouped" | "skipped",
  "datetime_column": "col" | null,
  "targets": [
    {{"column": "target_col", "horizon": {horizon}, "notes": "short reason"}}
  ],
  "group_by": "col" | null,
  "inferred_frequency": "daily|weekly|monthly|quarterly|yearly|irregular|unknown",
  "reasoning": "short explanation (1-4 sentences)",
  "requires_user_approval_for_tabular_ml": true,
  "allow_tabular_ml": false
}}

Rules:
- targets must be 1..{max_targets} if suitable==true and mode!="skipped"
- every target.horizon MUST equal {horizon}
- if mode=="grouped" then group_by must be non-null
- if suitable==false then mode must be "skipped"
""".strip()

        try:
            logger.info(
                "LLM planning start dataset=%s model=%s max_targets=%s horizon=%s",
                dataset_id,
                self.model,
                max_targets,
                horizon,
            )
            text = self.llm.complete(prompt)
            data = self.llm.extract_json(text)

            plan = ForecastPlan(**data)
            plan.planner_source = "llm"
            plan.llm_model = self.model
            plan.fallback_reason = None

            if not signals_have_dt:
                return _force_skip(plan, "No datetime candidates in signals.")
            if not signals_have_targets:
                return _force_skip(plan, "No numeric target candidates in signals.")
            if not plan.suitable:
                return _force_skip(plan, "LLM marked dataset as not suitable.")
            if plan.mode == "skipped":
                return _force_skip(plan, "LLM chose skipped mode.")

            if not plan.datetime_column:
                return _fallback_plan(
                    forecast_signals,
                    reason="LLM plan missing datetime_column.",
                    max_targets=max_targets,
                    horizon=horizon,
                )

            _fill_targets_to_max(plan=plan, signals=forecast_signals, want=max_targets, horizon=horizon)

            if len(plan.targets) == 0:
                return _fallback_plan(
                    forecast_signals,
                    reason="LLM plan missing targets (after fill).",
                    max_targets=max_targets,
                    horizon=horizon,
                )

            if plan.mode == "grouped" and not plan.group_by:
                plan.mode = "overall"
                plan.reasoning = (plan.reasoning or "").strip()
                if plan.reasoning:
                    plan.reasoning += " | Grouped downgraded to overall (group_by missing)."
                else:
                    plan.reasoning = "Grouped downgraded to overall (group_by missing)."

            if not plan.inferred_frequency:
                plan.inferred_frequency = forecast_signals.get("inferred_frequency") or "unknown"

            logger.info(
                "LLM planning ok dataset=%s mode=%s suitable=%s dt=%s targets=%s horizon=%s",
                dataset_id,
                plan.mode,
                plan.suitable,
                plan.datetime_column,
                [t.column for t in plan.targets],
                horizon,
            )
            return plan

        except (ValidationError, Exception) as e:
            logger.exception("LLM planning failed dataset=%s -> fallback. err=%s", dataset_id, str(e))
            return _fallback_plan(
                forecast_signals,
                reason=f"LLM planning failed: {str(e)}",
                max_targets=max_targets,
                horizon=horizon,
            )