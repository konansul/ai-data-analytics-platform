# backend/api/visualization.py

import base64

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from sqlalchemy.orm import Session

from backend.api.helpers.ownership import get_owned_dataset_or_404
from backend.app.visualization.agent import VisualizationAgent
from backend.app.visualization.schemas import ColumnPairingPlan, PlotConfig, PlotsRequest, ExplainRequest, ExplainResponse, VizRequest

from backend.api.auth import get_current_user
from backend.database.db import get_db
from backend.database.models import User
from backend.database.storage import viz_plot_key, put_bytes
from backend.app.visualization.service import get_rich_metrics

from backend.api.helpers.datasets import load_dataset_df
from backend.api.helpers.visualization import enrich_profile, safe_filename
from backend.api.helpers.artifacts import add_artifact

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
        enriched = enrich_profile(req.profile_data, df)

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
        enriched = enrich_profile(req.profile_data, df)

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
def explain_chart_endpoint(req: ExplainRequest, ):
    try:
        agent = VisualizationAgent()
        text_result = agent.explain_visualization(req.plot_title, req.axis_info)
        return ExplainResponse(explanation=text_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@router.post("/visualization/plots/save")
def save_visualization_plot(
    payload: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dataset_id = payload.get("dataset_id")
    run_id = payload.get("run_id")
    title = payload.get("title") or "plot"
    png_base64 = payload.get("png_base64")

    if not dataset_id or not run_id or not png_base64:
        raise HTTPException(status_code=400, detail="dataset_id, run_id, png_base64 are required")

    get_owned_dataset_or_404(db, dataset_id, current_user.user_id)

    try:
        png_bytes = base64.b64decode(png_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid png_base64")

    filename = safe_filename(title)
    storage_key = viz_plot_key(current_user.user_id, run_id, filename)

    try:
        put_bytes(storage_key, png_bytes)

        add_artifact(
            db,
            user_id=current_user.user_id,
            dataset_id=dataset_id,
            run_type="viz",
            run_id=run_id,
            kind="viz_plot_png",
            mime_type="image/png",
            storage_key=storage_key,
            meta={"title": title},
            parent_run_id=run_id,
        )
        db.commit()

        return {"ok": True, "storage_key": storage_key, "filename": filename}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save plot: {e}")
