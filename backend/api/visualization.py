# backend/api/visualization.py
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.visualization.agent import VisualizationAgent
from backend.app.visualization.schemas import VisualizationPlan, ExplainResponse, ExplainRequest
from backend.api.auth import get_current_user
from backend.database.db import get_db
from backend.database.models import User
from backend.app.visualization.service import get_rich_metrics
from backend.api.datasets import load_dataset_as_dataframe

router = APIRouter()


class VizRequest(BaseModel):
    dataset_id: str
    profile_data: Dict[str, Any]


@router.post("/visualization/suggest", response_model=VisualizationPlan)
def suggest_visualizations(
    req: VizRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not req.profile_data:
        raise HTTPException(status_code=400, detail="Profile signals are required.")

    try:
        df = load_dataset_as_dataframe(
            req.dataset_id,
            current_user.user_id,
            version="current",
            db=db,
        )
        metrics = get_rich_metrics(df)  # keep if you need it (or remove if unused)

        agent = VisualizationAgent()
        plan = agent.create_plan(req.dataset_id, req.profile_data)
        return plan

    except HTTPException:
        # keep original HTTP errors (404, 500 raw_parquet missing, etc.)
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization Agent failed: {str(e)}")


@router.post("/visualization/explain", response_model=ExplainResponse)
def explain_chart_endpoint(
    req: ExplainRequest,
    current_user: User = Depends(get_current_user),
):
    try:
        agent = VisualizationAgent()
        text_result = agent.explain_visualization(req.plot_title, req.axis_info)
        return ExplainResponse(explanation=text_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")