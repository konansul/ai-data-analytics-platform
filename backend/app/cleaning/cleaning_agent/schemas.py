# backend/app/cleaning_agent/schema.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

StepName = Literal[
    "normalize",
    "trim_strings",
    "standardize_missing",
    "cast_types",
    "encode_booleans",
    "drop_rules",
    "datetime_inference",
    "deduplicate",
    "outliers",
    "impute_missing",
]


@dataclass(frozen=True)
class CleaningPlan:
    """
    A structured, testable plan for running the cleaning pipeline.

    - enabled_steps: which pipeline steps are enabled/disabled
    - params: thresholds/strategies for deterministic Pandas logic
    - notes: human-readable reasoning/explanations (from rule-based or LLM)
    - source: where the plan came from (rule_based / llm)
    - version: plan schema version

    """
    enabled_steps: Dict[StepName, bool]
    params: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    source: Literal["rule_based", "llm"] = "rule_based"
    version: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "enabled_steps": dict(self.enabled_steps),
            "params": dict(self.params),
            "notes": list(self.notes),
        }

    @staticmethod
    def default(
        *,
        # drop rules
        missing_threshold: float = 0.5,
        row_missing_threshold: float = 0.80,
        drop_rows: bool = True,
        ignore_columns_for_row_drop: Optional[List[str]] = None,
        # datetime inference
        datetime_success_ratio: float = 0.8,
        # imputation
        impute: bool = True,
        numeric_strategy: Optional[str] = "mean",
        categorical_strategy: Optional[str] = "mode",
        datetime_strategy: Optional[str] = None,
        fill_value: Any = 0,
        categorical_numeric_max_unique: int = 20,
        # outliers
        outliers_method: str = "iqr",      # "iqr" | "zscore" | "none"
        outliers_action: str = "clip",     # "clip" | "remove" | "none"
        iqr_k: float = 1.5,
        zscore_threshold: float = 3.0,
    ) -> "CleaningPlan":
        """
        Default plan: enable everything (safe defaults).
        LLM/rule-based agent can disable steps when unnecessary.
        """
        enabled: Dict[StepName, bool] = {
            "normalize": True,
            "trim_strings": True,
            "standardize_missing": True,
            "cast_types": True,
            "encode_booleans": True,
            "drop_rules": True,
            "datetime_inference": True,
            "deduplicate": True,
            "outliers": True,
            "impute_missing": bool(impute),
        }

        params: Dict[str, Any] = {
            "missing_threshold": float(missing_threshold),
            "row_missing_threshold": float(row_missing_threshold),
            "drop_rows": bool(drop_rows),
            "ignore_columns_for_row_drop": list(ignore_columns_for_row_drop or []),

            "datetime_success_ratio": float(datetime_success_ratio),

            "impute": bool(impute),
            "numeric_strategy": numeric_strategy,
            "categorical_strategy": categorical_strategy,
            "datetime_strategy": datetime_strategy,
            "fill_value": fill_value,
            "categorical_numeric_max_unique": int(categorical_numeric_max_unique),

            "outliers_method": outliers_method,
            "outliers_action": outliers_action,
            "iqr_k": float(iqr_k),
            "zscore_threshold": float(zscore_threshold),
        }

        return CleaningPlan(
            enabled_steps=enabled,
            params=params,
            notes=[],
            source="rule_based",
            version=2,
        )


def validate_plan_dict(plan: Dict[str, Any]) -> CleaningPlan:

    # Convert a plain dict (e.g., from LLM JSON) into a validated CleaningPlan

    if not isinstance(plan, dict):
        raise TypeError("Cleaning plan must be a dict")

    enabled_steps_raw = plan.get("enabled_steps", {})
    params = plan.get("params", {})
    notes = plan.get("notes", [])
    source = plan.get("source", "llm")

    if not isinstance(enabled_steps_raw, dict):
        raise TypeError("enabled_steps must be a dict")
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")
    if not isinstance(notes, list):
        raise TypeError("notes must be a list")

    allowed_steps = [
        "normalize",
        "trim_strings",
        "standardize_missing",
        "cast_types",
        "encode_booleans",
        "drop_rules",
        "datetime_inference",
        "deduplicate",
        "outliers",
        "impute_missing",
    ]

    enabled_steps: Dict[StepName, bool] = {}

    for step in allowed_steps:
        value = enabled_steps_raw.get(step, True)
        enabled_steps[step] = bool(value)  # type: ignore[assignment]

    if "missing_threshold" in params:
        params["missing_threshold"] = float(params["missing_threshold"])
    if "row_missing_threshold" in params:
        params["row_missing_threshold"] = float(params["row_missing_threshold"])
    if "drop_rows" in params:
        params["drop_rows"] = bool(params["drop_rows"])
    if "ignore_columns_for_row_drop" in params:
        if params["ignore_columns_for_row_drop"] is None:
            params["ignore_columns_for_row_drop"] = []
        if not isinstance(params["ignore_columns_for_row_drop"], list):
            params["ignore_columns_for_row_drop"] = [str(params["ignore_columns_for_row_drop"])]
        params["ignore_columns_for_row_drop"] = [str(x) for x in params["ignore_columns_for_row_drop"]]

    if "datetime_success_ratio" in params:
        params["datetime_success_ratio"] = float(params["datetime_success_ratio"])

    if "categorical_numeric_max_unique" in params:
        params["categorical_numeric_max_unique"] = int(params["categorical_numeric_max_unique"])
    if "impute" in params:
        params["impute"] = bool(params["impute"])

    if "iqr_k" in params:
        params["iqr_k"] = float(params["iqr_k"])
    if "zscore_threshold" in params:
        params["zscore_threshold"] = float(params["zscore_threshold"])

    notes = [str(x) for x in notes]

    src: Literal["rule_based", "llm"] = "llm" if source != "rule_based" else "rule_based"

    return CleaningPlan(
        enabled_steps=enabled_steps,
        params=params,
        notes=notes,
        source=src,
        version=int(plan.get("version", 2)),
    )