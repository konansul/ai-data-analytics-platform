# backend/api/reporting.py

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response
from sqlalchemy.orm import Session

from backend.api.auth import get_current_user
from backend.api.helpers.ownership import get_owned_dataset_or_404
from backend.database.db import get_db
from backend.database.models import User

from backend.database import storage as blob
from backend.api.helpers.artifacts import add_artifact
from backend.api.models import GenerateReportRequest, GenerateReportResponse

from backend.app.reporting.report_builder import ReportBuilder, ReportBuilderConfig
from backend.app.reporting.report_agent import ReportAgent, ReportAgentConfig
from backend.app.reporting.pdf_renderer import PDFRenderer
from backend.app.reporting.llm_report_agent import LLMReportConfig

router = APIRouter()


@router.post("/reporting/generate", response_model=GenerateReportResponse)
def generate_report(
    req: GenerateReportRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        # 1) Ownership
        get_owned_dataset_or_404(db, req.dataset_id, current_user.user_id)

        user_id = current_user.user_id
        dataset_id = req.dataset_id
        run_id = req.run_id

        # 2) Build inputs
        builder = ReportBuilder(
            db=db,
            config=ReportBuilderConfig(
                max_viz_plots=int(req.max_viz_plots),
                max_forecast_plots=int(req.max_forecast_plots),
            ),
        )
        builder_output = builder.build(
            user_id=user_id,
            dataset_id=dataset_id,
            run_id=run_id,
            title=req.title,
        )

        agent = ReportAgent(
            config=ReportAgentConfig(
                llm=LLMReportConfig(provider="gemini"),
                max_viz_plots=int(req.max_viz_plots),
                max_forecast_plots=int(req.max_forecast_plots),
            )
        )
        renderer = PDFRenderer()

        result = agent.run(builder_output=builder_output, pdf_renderer=renderer)
        pdf_bytes = result.get("pdf_bytes")
        payload = result.get("payload") or {}

        if not pdf_bytes:
            raise RuntimeError("ReportAgent produced empty pdf_bytes")

        # 4) Save to storage
        report_run_id = blob.new_id("rpt")
        pdf_name = f"{report_run_id}.pdf"
        storage_key = blob.report_key(user_id, run_id, pdf_name)

        blob.put_bytes(storage_key, pdf_bytes, content_type="application/pdf")

        # Optional: save payload json next to pdf
        try:
            blob.put_json(blob.report_key(user_id, run_id, f"{report_run_id}.json"), payload)
        except Exception:
            pass

        # 5) DB artifact
        art = add_artifact(
            db,
            user_id=user_id,
            dataset_id=dataset_id,
            run_type="report",
            run_id=run_id,
            parent_run_id=run_id,
            kind="report_pdf",
            mime_type="application/pdf",
            storage_key=storage_key,
            meta={
                "report_run_id": report_run_id,
                "filename": pdf_name,
                "title": payload.get("title") or req.title or "AI Data Analysis Report",
            },
        )
        db.commit()

        return GenerateReportResponse(
            report_run_id=report_run_id,
            artifact_id=art.artifact_id,
            storage_key=storage_key,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generate failed: {str(e)}")


@router.get("/reporting/download")
def download_report_pdf(
    storage_key: str,
    current_user: User = Depends(get_current_user),
):
    try:
        pdf_bytes = blob.get_bytes(storage_key)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": 'inline; filename="report.pdf"'},
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not load PDF: {str(e)}")