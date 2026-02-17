# backend/app/cleaning_agent/cleaning_policy_llm.py
from __future__ import annotations

from typing import Any, Dict

from .schemas import CleaningPlan, validate_plan_dict
from .llm_client import LLMClient
from .cleaning_policy_utils import _sanitize_plan


def build_cleaning_plan_llm(pre_profile: Dict[str, Any], client: LLMClient) -> CleaningPlan:
    prompt = make_llm_prompt(pre_profile)

    raw_text = client.complete(prompt)
    plan_dict = client.extract_json(raw_text)

    plan = validate_plan_dict(plan_dict)

    defaults = CleaningPlan.default().params
    for k, v in defaults.items():
        plan.params.setdefault(k, v)

    return _sanitize_plan(plan)


def make_llm_prompt(pre_profile: Dict[str, Any]) -> str:
    return f"""
You are a data-cleaning policy agent.

IMPORTANT:
- Return only valid JSON (no markdown, no extra text).
- You do NOT receive raw dataset rows, only summary signals.

Goal:
Create a safe, deterministic cleaning plan for a pandas pipeline.

Allowed steps (only these keys are allowed):
normalize,
trim_strings,
standardize_missing,
cast_types,
encode_booleans,
drop_rules,
datetime_inference,
deduplicate,
outliers,
impute_missing

Return JSON exactly in this schema:
{{
  "version": 2,
  "source": "llm",
  "enabled_steps": {{
    "normalize": true,
    "trim_strings": true,
    "standardize_missing": true,
    "cast_types": true,
    "encode_booleans": true,
    "drop_rules": true,
    "datetime_inference": true,
    "deduplicate": true,
    "outliers": true,
    "impute_missing": true
  }},
  "params": {{
    "missing_threshold": 0.5,
    "row_missing_threshold": 0.8,
    "drop_rows": true,
    "ignore_columns_for_row_drop": [],

    "datetime_success_ratio": 0.8,

    "numeric_strategy": "mean",
    "categorical_strategy": "mode",
    "datetime_strategy": null,
    "fill_value": 0,
    "categorical_numeric_max_unique": 20,

    "outliers_method": "iqr",
    "outliers_action": "clip",
    "iqr_k": 1.5,
    "zscore_threshold": 3.0
  }},
  "notes": ["short reason 1", "short reason 2"]
}}

Rules / constraints:
- Keep enabled_steps keys EXACTLY as listed. Do not add other steps.
- missing_threshold must be in [0.10, 0.90].
- row_missing_threshold must be in [0.50, 0.99].
- datetime_success_ratio must be in [0.50, 0.99].
- numeric_strategy: one of ["mean","median","constant"].
- categorical_strategy: one of ["mode","constant"].
- datetime_strategy: one of ["ffill","bfill", null].
- outliers_method: one of ["iqr","zscore","none"].
- outliers_action: one of ["clip","remove","none"].
- iqr_k must be in [0.5, 10.0].
- zscore_threshold must be in [2.0, 10.0].
- If dataset has no datetime/time signals, set datetime_inference=false.
- If overall missingness is very low, set impute_missing=false.
- If dataset looks like mostly strings/categories, set outliers_method="none" and/or outliers=false.
- Prefer numeric_strategy="median" when skewness is present.
- DO NOT invent columns that do not exist (you do not know column names anyway).

Important behavior guidelines:
- standardize_missing should almost always be enabled (it unlocks all other steps).
- drop_rules should run after standardize_missing.
- deduplicate should be enabled for most datasets unless rows are extremely small.
- If outliers is disabled, set outliers_method="none" and outliers_action="none".

Signals (pre_profile dict):
{pre_profile}
""".strip()