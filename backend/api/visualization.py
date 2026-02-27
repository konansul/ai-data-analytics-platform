# backend/api/visualization.py
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
from pydantic import BaseModel

import numpy as np
from backend.app.visualization.agent import VisualizationAgent
from backend.app.visualization.schemas import (
    ColumnPairingPlan,
    ColumnPairing,
    PlotConfig,
    PlotsRequest,
    VisualizationPlan,
    ExplainRequest,
    ExplainResponse,
)
from backend.api.auth import get_current_user
from backend.database.models import User
from backend.app.visualization.service import get_rich_metrics
from backend.api.datasets import load_dataset_as_dataframe

router = APIRouter()


class VizRequest(BaseModel):
    dataset_id: str
    profile_data: Dict[str, Any]


def _enrich_profile(profile_data: Dict[str, Any], df) -> Dict[str, Any]:
    """
    Guarantees typed column lists are present regardless of profile structure,
    by deriving them directly from the loaded DataFrame.
    """
    enriched = dict(profile_data)
    if not enriched.get("numeric_columns"):
        enriched["numeric_columns"] = df.select_dtypes(include=[np.number]).columns.tolist()
    if not enriched.get("categorical_columns"):
        enriched["categorical_columns"] = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
    if not enriched.get("date_columns"):
        enriched["date_columns"] = df.select_dtypes(
            include=["datetime64", "datetimetz"]
        ).columns.tolist()
    if not any(enriched.get(k) for k in ["columns", "column_stats", "fields", "schema", "variables", "features"]):
        enriched["columns"] = {col: {"dtype": str(df[col].dtype)} for col in df.columns}
    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Generate column pairings (user picks which ones to visualize)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/visualization/pairings", response_model=ColumnPairingPlan)
def get_visualization_pairings(
        req: VizRequest,
        current_user: User = Depends(get_current_user),
):
    """
    Stage 1: Runs the Visualization Pairing Agent.
    Returns a ranked list of column pairings — the user selects which to visualize.
    No plot types are chosen at this stage.
    """
    if not req.profile_data:
        raise HTTPException(status_code=400, detail="Profile signals are required.")
    try:
        df = load_dataset_as_dataframe(req.dataset_id)
        metrics = get_rich_metrics(df)
        enriched = _enrich_profile(req.profile_data, df)

        agent = VisualizationAgent()
        pairing_plan = agent.get_pairings(req.dataset_id, enriched, metrics=metrics)
        return pairing_plan

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pairing Agent failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Generate plots for user-selected pairings
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/visualization/plots", response_model=List[PlotConfig])
def get_visualization_plots(
        req: PlotsRequest,
        current_user: User = Depends(get_current_user),
):
    """
    Stage 2: Runs the Visualization Plot Selection Agent on the pairings
    the user selected from Stage 1. Returns fully-specified PlotConfig objects.
    """
    if not req.selected_pairings:
        raise HTTPException(status_code=400, detail="No pairings selected.")
    try:
        df = load_dataset_as_dataframe(req.dataset_id)
        metrics = get_rich_metrics(df)
        enriched = _enrich_profile(req.profile_data, df)

        agent = VisualizationAgent()
        plots = agent.get_plots(
            req.dataset_id,
            enriched,
            selected_pairings=req.selected_pairings,
            metrics=metrics,
        )
        return plots

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plot Selection Agent failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Explain — plain-language insight for a rendered chart
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/visualization/explain", response_model=ExplainResponse)
def explain_chart_endpoint(
        req: ExplainRequest,
        current_user: User = Depends(get_current_user),
):
    """Generates a plain-language explanation for a specific chart."""
    try:
        agent = VisualizationAgent()
        text_result = agent.explain_visualization(req.plot_title, req.axis_info)
        return ExplainResponse(explanation=text_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Legacy combined endpoint — kept for backward compatibility
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/visualization/suggest", response_model=VisualizationPlan)
def suggest_visualizations(
        req: VizRequest,
        current_user: User = Depends(get_current_user),
):
    """Legacy: runs both stages in one call."""
    if not req.profile_data:
        raise HTTPException(status_code=400, detail="Profile signals are required.")
    try:
        df = load_dataset_as_dataframe(req.dataset_id)
        metrics = get_rich_metrics(df)
        enriched = _enrich_profile(req.profile_data, df)

        agent = VisualizationAgent()
        plan = agent.create_plan(req.dataset_id, enriched, metrics=metrics)
        return plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization Agent failed: {str(e)}")
