from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from backend.app.reporting.llm_report_agent import LLMReportConfig, generate_llm_output


@dataclass(frozen=True)
class ReportAgentConfig:
    llm: LLMReportConfig = LLMReportConfig(provider="gemini")
    max_viz_plots: int = 3
    max_forecast_plots: int = 3


class ReportAgent:
    def __init__(self, *, config: Optional[ReportAgentConfig] = None):
        self.config = config or ReportAgentConfig()

    def run(self, *, builder_output: Dict[str, Any], pdf_renderer: Any) -> Dict[str, Any]:
        viz_plots = list(builder_output.get("viz_plots") or [])[: int(self.config.max_viz_plots)]
        forecast_plots = list(builder_output.get("forecast_plots") or [])[: int(self.config.max_forecast_plots)]

        builder_output = dict(builder_output)
        builder_output["viz_plots"] = viz_plots
        builder_output["forecast_plots"] = forecast_plots

        llm_output = generate_llm_output(builder_output, self.config.llm)
        pdf_bytes = pdf_renderer.render_pdf(builder_output=builder_output, llm_output=llm_output)

        return {
            "pdf_bytes": pdf_bytes,
            "builder_output": builder_output,
            "llm_output": llm_output,
        }