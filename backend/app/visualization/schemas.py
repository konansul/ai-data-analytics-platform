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


class ExplainRequest(BaseModel):
    plot_title: str
    axis_info: str  # e.g. "X: Year, Y: Sales"


class ExplainResponse(BaseModel):
    explanation: str