# backend/api/visualization.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List
from sqlalchemy.orm import Session

from backend.api.helpers.ownership import get_owned_dataset_or_404
from backend.app.visualization.agent import VisualizationAgent
from backend.app.visualization.schemas import ColumnPairingPlan, PlotConfig, PlotsRequest, VisualizationPlan, ExplainRequest, ExplainResponse, VizRequest

from backend.api.auth import get_current_user
from backend.database.db import get_db
from backend.database.models import User
from backend.app.visualization.service import get_rich_metrics

from backend.api.helpers.datasets import load_dataset_df
from backend.api.helpers.visualization import _enrich_profile

router = APIRouter()


@router.post("/visualization/pairings", response_model=ColumnPairingPlan)
def get_visualization_pairings(
    req: VizRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not req.profile_data:
        raise HTTPException(status_code=400, detail="Profile signals are required.")
    try:
        get_owned_dataset_or_404(db, req.dataset_id, current_user.user_id)

        df = load_dataset_df(
            db=db,
            dataset_id=req.dataset_id,
            user_id=current_user.user_id,
            version="current",
        )
        metrics = get_rich_metrics(df)
        enriched = _enrich_profile(req.profile_data, df)

        agent = VisualizationAgent()
        return agent.get_pairings(req.dataset_id, enriched, metrics=metrics)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pairing Agent failed: {str(e)}")


@router.post("/visualization/plots", response_model=List[PlotConfig])
def get_visualization_plots(
    req: PlotsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not req.selected_pairings:
        raise HTTPException(status_code=400, detail="No pairings selected.")
    try:
        get_owned_dataset_or_404(db, req.dataset_id, current_user.user_id)

        df = load_dataset_df(
            db=db,
            dataset_id=req.dataset_id,
            user_id=current_user.user_id,
            version="current",
        )
        metrics = get_rich_metrics(df)
        enriched = _enrich_profile(req.profile_data, df)

        agent = VisualizationAgent()
        return agent.get_plots(
            req.dataset_id,
            enriched,
            selected_pairings=req.selected_pairings,
            metrics=metrics,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plot Selection Agent failed: {str(e)}")


@router.post("/visualization/explain", response_model=ExplainResponse)
def explain_chart_endpoint(
        req: ExplainRequest,
):
    try:
        agent = VisualizationAgent()
        text_result = agent.explain_visualization(req.plot_title, req.axis_info)
        return ExplainResponse(explanation=text_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.post("/visualization/suggest", response_model=VisualizationPlan)
def suggest_visualizations(
    req: VizRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not req.profile_data:
        raise HTTPException(status_code=400, detail="Profile signals are required.")
    try:
        get_owned_dataset_or_404(db, req.dataset_id, current_user.user_id)

        df = load_dataset_df(
            db=db,
            dataset_id=req.dataset_id,
            user_id=current_user.user_id,
            version="current",
        )
        metrics = get_rich_metrics(df)
        enriched = _enrich_profile(req.profile_data, df)

        agent = VisualizationAgent()
        return agent.create_plan(req.dataset_id, enriched, metrics=metrics)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization Agent failed: {str(e)}")
