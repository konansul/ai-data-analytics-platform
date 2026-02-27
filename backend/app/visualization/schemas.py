from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

PlotType = Literal["line", "bar", "scatter", "histogram", "box", "heatmap", "pie"]


class ColumnPairing(BaseModel):
    columns: List[str]
    rationale: str
    rank: Optional[int] = None
    score: Optional[float] = None
    template: Optional[Literal[
        "date_numeric", "cat_numeric", "num_num", "num_univariate"
    ]] = None


class ColumnPairingPlan(BaseModel):
    """Full output of Stage 1."""
    dataset_id: str
    pairings: List[ColumnPairing]


class PlotConfig(BaseModel):
    title: str
    plot_type: PlotType
    alt_plot_type: Optional[PlotType] = None
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    description: Optional[str] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    source_pairing: Optional[List[str]] = None


class PlotsRequest(BaseModel):
    """
    Request body for POST /visualization/plots (Stage 2).
    User sends only the pairings they selected from Stage 1.
    """
    dataset_id: str
    profile_data: Dict[str, Any]
    selected_pairings: List[ColumnPairing]


class VisualizationPlan(BaseModel):
    """Backward-compatible combined plan."""
    dataset_id: str
    pairings: List[ColumnPairing] = Field(default_factory=list)
    plots: List[PlotConfig]


class ExplainRequest(BaseModel):
    plot_title: str
    axis_info: str


class ExplainResponse(BaseModel):
    explanation: str
