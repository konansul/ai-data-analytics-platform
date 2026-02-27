from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.api.auth import get_current_user
from backend.database.db import get_db
from backend.database.models import User, VisualizationRun
from backend.database.storage import put_bytes, to_jsonable, new_id, viz_prefix

from backend.app.visualization.agent import VisualizationAgent
from backend.app.visualization.schemas import ExplainResponse, ExplainRequest, SaveVizPlotRequest, VizRequest
from backend.app.visualization.service import get_rich_metrics

from backend.api.helpers.ownership import get_owned_dataset_or_404, get_owned_clean_run_or_404, get_owned_visualization_run_or_404
from backend.api.helpers.datasets import load_dataset_df
from backend.api.helpers.artifacts import add_artifact

router = APIRouter()


@router.post("/visualization/suggest")
def suggest_visualizations(
    req: VizRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    if not req.profile_data:
        raise HTTPException(status_code=400, detail="Profile signals are required.")
    if not req.run_id:
        raise HTTPException(status_code=400, detail="run_id is required (cleaning run id).")

    ds = get_owned_dataset_or_404(db, req.dataset_id, current_user.user_id)
    clean_run = get_owned_clean_run_or_404(db, req.run_id, current_user.user_id)

    if clean_run.dataset_id != ds.dataset_id:
        raise HTTPException(status_code=400, detail="run_id does not belong to this dataset.")

    df = load_dataset_df(
        db=db,
        dataset_id=ds.dataset_id,
        user_id=current_user.user_id,
        version="current",
    )

    viz_run_id = new_id("viz")
    row = VisualizationRun(
        viz_run_id=viz_run_id,
        user_id=current_user.user_id,
        dataset_id=ds.dataset_id,
        run_id=req.run_id,
        status="running",
        error=None,
        mode="auto",
        meta_json={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(row)

    ds.viz_status = "running"
    ds.last_viz_run_id = viz_run_id
    db.commit()

    try:
        metrics = get_rich_metrics(df)
        agent = VisualizationAgent()
        plan = agent.create_plan(req.dataset_id, req.profile_data)
        plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan.dict()  # type: ignore
    except Exception as e:
        row.status = "failed"
        row.error = f"{type(e).__name__}: {e}"
        row.updated_at = datetime.now(timezone.utc)
        ds.viz_status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Visualization Agent failed: {str(e)}")

    prefix = viz_prefix(current_user.user_id, req.run_id)
    plan_key = f"{prefix}/viz_plan.json"
    metrics_key = f"{prefix}/viz_metrics.json"
    profile_key = f"{prefix}/viz_profile_data.json"

    try:
        put_bytes(plan_key, json.dumps(to_jsonable(plan_dict), ensure_ascii=False, indent=2).encode("utf-8"))
        put_bytes(metrics_key, json.dumps(to_jsonable(metrics), ensure_ascii=False, indent=2).encode("utf-8"))
        put_bytes(profile_key, json.dumps(to_jsonable(req.profile_data), ensure_ascii=False, indent=2).encode("utf-8"))
    except Exception as e:
        row.status = "failed"
        row.error = f"Persist failed: {type(e).__name__}: {e}"
        row.updated_at = datetime.now(timezone.utc)
        ds.viz_status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to persist visualization artifacts: {e}")

    try:
        add_artifact(
            db,
            user_id=current_user.user_id,
            dataset_id=ds.dataset_id,
            run_type="viz",
            run_id=viz_run_id,
            parent_run_id=req.run_id,
            kind="viz_plan_json",
            mime_type="application/json",
            storage_key=plan_key,
        )
        add_artifact(
            db,
            user_id=current_user.user_id,
            dataset_id=ds.dataset_id,
            run_type="viz",
            run_id=viz_run_id,
            parent_run_id=req.run_id,
            kind="viz_metrics_json",
            mime_type="application/json",
            storage_key=metrics_key,
        )
        add_artifact(
            db,
            user_id=current_user.user_id,
            dataset_id=ds.dataset_id,
            run_type="viz",
            run_id=viz_run_id,
            parent_run_id=req.run_id,
            kind="viz_profile_json",
            mime_type="application/json",
            storage_key=profile_key,
        )
    except Exception as e:
        row.status = "failed"
        row.error = f"Artifact registry failed: {type(e).__name__}: {e}"
        row.updated_at = datetime.now(timezone.utc)
        ds.viz_status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to register visualization artifacts: {e}")

    row.status = "done"
    row.error = None
    row.updated_at = datetime.now(timezone.utc)
    row.meta_json = to_jsonable(
        {
            "run_id": req.run_id,
            "plan_key": plan_key,
            "metrics_key": metrics_key,
            "profile_key": profile_key,
        }
    )

    ds.viz_status = "done"
    ds.last_viz_run_id = viz_run_id
    ds.visualized_at = datetime.now(timezone.utc)
    db.commit()

    return {"viz_run_id": viz_run_id, "run_id": req.run_id, "plan": plan_dict}


@router.post("/visualization/explain", response_model=ExplainResponse)
def explain_chart_endpoint(req: ExplainRequest, ):
    try:
        agent = VisualizationAgent()
        text_result = agent.explain_visualization(req.plot_title, req.axis_info)
        return ExplainResponse(explanation=text_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.post("/visualization/plots")
def save_visualization_plot(
    req: SaveVizPlotRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    get_owned_dataset_or_404(db, req.dataset_id, current_user.user_id)
    viz_row = get_owned_visualization_run_or_404(db, req.viz_run_id, current_user.user_id)

    if not viz_row.run_id:
        raise HTTPException(status_code=500, detail="VisualizationRun.run_id is missing (migration/data issue)")

    try:
        png_bytes = base64.b64decode(req.png_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid png_base64")

    safe_title = "".join(ch if ch.isalnum() else "_" for ch in (req.title or "plot"))[:80].strip("_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{ts}_{safe_title}.png"

    prefix = viz_prefix(current_user.user_id, viz_row.run_id)
    key = f"{prefix}/plots/{filename}"

    try:
        put_bytes(key, png_bytes, content_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist png: {e}")

    try:
        add_artifact(
            db,
            user_id=current_user.user_id,
            dataset_id=req.dataset_id,
            run_type="viz",
            run_id=req.viz_run_id,
            parent_run_id=viz_row.run_id,
            kind="viz_plot_png",
            mime_type="image/png",
            storage_key=key,
            meta=to_jsonable(
                {
                    "title": req.title,
                    "plot_type": req.plot_type,
                    **(req.meta or {}),
                }
            ),
        )
        db.commit()
    except Exception:
        db.rollback()

    return {"ok": True, "storage_key": key, "file": filename, "run_id": viz_row.run_id}