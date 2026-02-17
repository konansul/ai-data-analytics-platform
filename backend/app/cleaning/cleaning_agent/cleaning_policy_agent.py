# backend/app/cleaning_agent/cleaning_policy_agent.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .schemas import CleaningPlan
from .llm_client import LLMClient, LLMUnavailableError
from .cleaning_policy_rule_based import build_cleaning_plan_rule_based
from .cleaning_policy_llm import build_cleaning_plan_llm


def build_cleaning_plan(
    pre_profile: Dict[str, Any],
    *,
    use_llm: bool = False,
    llm_client: Optional[LLMClient] = None,
    model: str = "gemini-2.5-flash",
) -> CleaningPlan:
    """
    Public API. Always returns a CleaningPlan.
    If use_llm=True: try LLM first, else fallback to rule-based.
    """
    rule_based_plan = build_cleaning_plan_rule_based(pre_profile)

    if not use_llm:
        return rule_based_plan

    try:
        client = llm_client or LLMClient.from_env(model=model)
        return build_cleaning_plan_llm(pre_profile, client)

    except (
        LLMUnavailableError,
        json.JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
    ) as e:
        notes = list(rule_based_plan.notes)
        notes.append(f"LLM fallback → {type(e).__name__}: {e}")

        return CleaningPlan(
            enabled_steps=dict(rule_based_plan.enabled_steps),
            params=dict(rule_based_plan.params),
            notes=notes,
            source="rule_based",
            version=rule_based_plan.version,
        )