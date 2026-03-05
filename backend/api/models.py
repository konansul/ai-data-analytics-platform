# backend/api/models.py
from __future__ import annotations

from typing import Any, Dict, Literal, Optional, List

from pydantic import BaseModel, Field


class DatasetRef(BaseModel):
    dataset_id: str


class UploadResponse(BaseModel):
    datasets: list[Dict[str, Any]]


class PreviewResponse(BaseModel):
    dataset_id: str
    columns: list[str]
    rows: list[Dict[str, Any]]


class ProfilingRequest(BaseModel):
    dataset_id: str
    options: Optional[Dict[str, Any]] = None


class ProfilingResponse(BaseModel):
    profile_id: str


class PolicySuggestRequest(BaseModel):
    dataset_id: str
    mode: Literal["rule_based", "llm"] = "rule_based"
    llm_model: str = "gemini-2.5-flash"


class PolicySuggestResponse(BaseModel):
    policy: Dict[str, Any]
    source: str
    notes: list[str] = Field(default_factory=list)


class CleaningRunRequest(BaseModel):
    dataset_id: str
    use_llm: bool = False
    llm_model: str = "gemini-2.5-flash"

    missing_threshold: Optional[float] = None
    impute: Optional[bool] = None
    numeric_strategy: Optional[str] = None
    categorical_strategy: Optional[str] = None
    datetime_strategy: Optional[str] = None
    fill_value: Optional[Any] = None
    datetime_success_ratio: Optional[float] = None
    categorical_numeric_max_unique: Optional[int] = None


class CleaningRunResponse(BaseModel):
    run_id: str

class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserMeResponse(BaseModel):
    user_id: str
    email: str


class ForecastSignalsRequest(BaseModel):
    dataset_id: str
    version: Literal["raw", "current"] = "current"


class ForecastSignalsResponse(BaseModel):
    dataset_id: str
    datetime_candidates: List[Dict[str, Any]] = Field(default_factory=list)  # [{column, success_ratio, letters_ratio}]
    inferred_frequency: str = "unknown"
    numeric_target_candidates: List[str] = Field(default_factory=list)
    grouping_candidates: List[str] = Field(default_factory=list)
    feasible: bool


class ForecastPlanRequest(BaseModel):
    dataset_id: str
    version: Literal["raw", "current"] = "current"
    signals: ForecastSignalsResponse
    profile: Optional[Dict[str, Any]] = None
    user_intent: Optional[str] = None
    max_targets: int = 4
    head_rows: int = 10
    horizon: int = Field(default=30, ge=1, le=3650)


class ForecastTargetIn(BaseModel):
    column: str
    horizon: int = 30


class ForecastPlanResponse(BaseModel):
    dataset_id: str
    suitable: bool
    mode: Literal["overall", "grouped", "skipped"]
    datetime_column: Optional[str] = None
    inferred_frequency: Optional[str] = None
    group_by: Optional[str] = None
    targets: List[ForecastTargetIn] = Field(default_factory=list)
    planner_source: Literal["llm", "fallback"] = "fallback"
    llm_model: Optional[str] = None
    reasoning: str = ""
    reasons: List[str] = Field(default_factory=list)


class ForecastRunRequest(BaseModel):
    dataset_id: str
    run_id: str
    plan: ForecastPlanResponse
    model: Literal["auto", "arima", "prophet", "random_forest"] = "auto"
    version: Literal["raw", "current"] = "current"
    horizon: int = Field(default=30, ge=1, le=3650)
    preview_rows: int = 50


class ForecastRunOneResult(BaseModel):
    target: str
    mode: Literal["overall", "grouped"]
    model_used: str
    datetime_column: str
    group_by: Optional[str]
    horizon: int
    frequency: str
    preview: List[Dict[str, Any]]
    meta: Dict[str, Any]


class ForecastRunResponse(BaseModel):
    dataset_id: str
    run_id: str
    forecast_run_id: str
    results: List[ForecastRunOneResult]

class VizRequest(BaseModel):
    dataset_id: str
    run_id: str
    profile_data: Dict[str, Any]

class GenerateReportRequest(BaseModel):
    dataset_id: str = Field(..., description="Dataset id")
    run_id: str = Field(..., description="Cleaning run id (run_xxx)")
    title: Optional[str] = Field(None, description="Optional report title")
    max_viz_plots: int = 3
    max_forecast_plots: int = 3
    return_pdf: bool = True


class GenerateReportResponse(BaseModel):
    report_run_id: str
    artifact_id: str
    storage_key: str