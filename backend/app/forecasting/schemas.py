import pandas as pd

from pydantic import BaseModel, Field, ValidationError
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

ModelType = Literal["auto", "arima", "prophet", "random_forest"]
Frequency = Literal["daily", "weekly", "monthly", "quarterly", "yearly", "irregular", "unknown"]
PlannerSource = Literal["llm", "fallback"]
ForecastMode = Literal["overall", "grouped", "skipped"]

@dataclass
class ForecastResult:
    dataset_id: str
    mode: Literal["overall", "grouped"]
    model_used: str
    datetime_column: str
    target: str
    group_by: Optional[str]
    horizon: int
    frequency: str
    forecast_df: pd.DataFrame
    meta: Dict[str, Any]

class ForecastTarget(BaseModel):
    column: str
    horizon: int = Field(default=30, ge=1, le=3650)
    notes: Optional[str] = None

class ForecastPlan(BaseModel):
    mode: ForecastMode
    suitable: bool
    datetime_column: Optional[str] = None
    targets: List[ForecastTarget] = Field(default_factory=list)
    group_by: Optional[str] = None
    inferred_frequency: Optional[str] = None  # daily/weekly/...
    reasoning: str = ""
    requires_user_approval_for_tabular_ml: bool = True
    allow_tabular_ml: bool = False
    planner_source: PlannerSource = "fallback"
    llm_model: Optional[str] = None
    fallback_reason: Optional[str] = None

class DatetimeCandidate(BaseModel):
    column: str
    score: float = 0.0
    success_ratio: float = 0.0
    notes: List[str] = Field(default_factory=list)


class TargetCandidate(BaseModel):
    column: str
    score: float = 0.0
    notes: List[str] = Field(default_factory=list)


class GroupingCandidate(BaseModel):
    column: str
    cardinality: Optional[int] = None
    score: float = 0.0
    notes: List[str] = Field(default_factory=list)


class ForecastSignals(BaseModel):
    forecast_feasible: bool
    reason_if_not_feasible: Optional[str] = None
    datetime_candidates: List[DatetimeCandidate] = Field(default_factory=list)
    inferred_frequency: Frequency = "unknown"
    target_candidates: List[TargetCandidate] = Field(default_factory=list)
    grouping_candidates: List[GroupingCandidate] = Field(default_factory=list)
    source_profile: Literal["pre_profile", "post_profile"] = "post_profile"