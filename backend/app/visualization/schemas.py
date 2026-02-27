# backend/app/visualization/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class PlotConfig(BaseModel):
    title: str
    plot_type: Literal["line", "bar", "scatter", "histogram", "box", "heatmap", "pie"]
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    description: Optional[str] = None
    # Constraints like {"top_k": 10} or {"aggregate": "mean"}
    constraints: Dict[str, Any] = Field(default_factory=dict)

class VisualizationPlan(BaseModel):
    dataset_id: str
    plots: List[PlotConfig]


#Request Schema for Explanation
class ExplainRequest(BaseModel):
    plot_title: str
    axis_info: str  # e.g. "X: Year, Y: Sales"


#  Response Schema for /explain ---
class ExplainResponse(BaseModel):
    explanation: str

class SaveVizPlotRequest(BaseModel):
    run_id: str
    viz_run_id: str
    dataset_id: str
    title: str
    plot_type: str
    png_base64: str
    meta: Dict[str, Any] = Field(default_factory=dict)

class VizRequest(BaseModel):
    dataset_id: str
    run_id: str
    profile_data: Dict[str, Any]