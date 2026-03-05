from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class PlotArtifactOut(BaseModel):
    artifact_id: str
    kind: str
    mime_type: str
    storage_key: str
    meta: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class ReportBuilderOutput(BaseModel):
    title: str = "AI Data Analysis Report"
    user_id: str
    dataset_id: str
    run_id: str
    cleaning_report: Dict[str, Any] = Field(default_factory=dict)
    signals: Dict[str, Any] = Field(default_factory=dict)
    viz_summary: Dict[str, Any] = Field(default_factory=dict)
    forecast_summary: Dict[str, Any] = Field(default_factory=dict)
    viz_plots: List[PlotArtifactOut] = Field(default_factory=list)
    forecast_plots: List[PlotArtifactOut] = Field(default_factory=list)



class LLMReportOutput(BaseModel):
    executive_summary: Optional[str] = None
    cleaning_notes: Optional[str] = None
    signals_notes: Optional[str] = None
    visualization_notes: Optional[str] = None
    forecasting_notes: Optional[str] = None
    conclusion: Optional[str] = None


class ReportGenerateRequest(BaseModel):
    dataset_id: str
    run_id: str
    title: Optional[str] = None
    max_viz_plots: int = Field(default=3, ge=0, le=20)
    max_forecast_plots: int = Field(default=3, ge=0, le=20)
    use_llm: bool = True
    llm_model: Optional[str] = None


class ReportGenerateResponse(BaseModel):
    ok: bool = True
    report_run_id: str
    artifact_id: str
    storage_key: str
    mime_type: str = "application/pdf"
    title: Optional[str] = None
    dataset_id: str
    run_id: str

class ReportServiceResult(BaseModel):
    artifact_id: str
    storage_key: str
    mime_type: str = "application/pdf"

    builder_output: Optional[ReportBuilderOutput] = None
    llm_output: Optional[LLMReportOutput] = None


ReportStage = Literal["cleaning", "viz", "forecast", "report"]