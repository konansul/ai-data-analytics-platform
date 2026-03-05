from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class LLMReportConfig:
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    api_key_env: str = "GEMINI_API_KEY"
    max_chars: int = 120_000
    timeout_sec: int = 60


class LLMReportOutput(BaseModel):
    executive_summary: str = Field("")
    section_overview: str = Field("")
    plot_captions: Dict[str, str] = Field(default_factory=dict)
    conclusion: str = Field("")
    visualization_notes: str = Field("")
    forecasting_notes: str = Field("")
    cleaning_notes: str = Field("")
    signals_notes: str = Field("")


def _compact_builder(builder_output: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    def take(obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, list):
            out = []
            for x in obj:
                if isinstance(x, dict):
                    d = {}
                    if "plot_id" in x:
                        d["plot_id"] = x.get("plot_id")
                    if "storage_key" in x:
                        d["storage_key"] = x.get("storage_key")
                    if "kind" in x:
                        d["kind"] = x.get("kind")
                    if "mime_type" in x:
                        d["mime_type"] = x.get("mime_type")
                    if "meta" in x:
                        d["meta"] = take(x.get("meta") or {})
                    if "data" in x:
                        d["data"] = take(x.get("data"))
                    out.append(d)
                else:
                    out.append(take(x))
            return out
        if isinstance(obj, dict):
            return {k: take(v) for k, v in obj.items()}
        return str(obj)

    compact = {
        "title": builder_output.get("title"),
        "dataset_id": builder_output.get("dataset_id"),
        "run_id": builder_output.get("run_id"),
        "cleaning_report": take(builder_output.get("cleaning_report") or {}),
        "signals": take(builder_output.get("signals") or {}),
        "viz_summary": take(builder_output.get("viz_summary") or {}),
        "forecast_summary": take(builder_output.get("forecast_summary") or {}),
        "signals_plots": take(builder_output.get("signals_plots") or []),
        "viz_plots": take(builder_output.get("viz_plots") or []),
        "forecast_plots": take(builder_output.get("forecast_plots") or []),
    }

    s = json.dumps(compact, ensure_ascii=False)
    if len(s) <= max_chars:
        return compact

    compact["cleaning_report"] = {}
    compact["signals"] = {}
    compact["viz_summary"] = {}
    compact["forecast_summary"] = {}
    return compact


def _gemini_generate_json(prompt: str, config: LLMReportConfig) -> Dict[str, Any]:
    api_key = os.getenv(config.api_key_env) or ""
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")

    from google import genai

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model=config.model, contents=prompt)

    text = getattr(resp, "text", None) or ""
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def generate_llm_output(builder_output: Dict[str, Any], config: LLMReportConfig) -> Dict[str, Any]:
    if (config.provider or "none").lower() == "none":
        out = LLMReportOutput(
            executive_summary="This report was generated in template mode (no LLM). Enable Gemini to get richer narrative insights.",
            plot_captions={},
        )
        return out.model_dump()

    compact = _compact_builder(builder_output, config.max_chars)

    schema_hint = {
        "executive_summary": "string",
        "section_overview": "string",
        "cleaning_notes": "string",
        "signals_notes": "string",
        "visualization_notes": "string",
        "forecasting_notes": "string",
        "plot_captions": {
            "<storage_key>": "string caption for viz/forecast plots",
            "<plot_id>": "string caption for in-memory signals plots",
        },
        "conclusion": "string",
    }

    prompt = (
        "You are an analytics report writer. Return ONLY valid JSON.\n"
        "Language: English.\n"
        "Task:\n"
        "1) Write a coherent executive_summary.\n"
        "2) Provide short section notes.\n"
        "3) For every plot in viz_plots + forecast_plots, produce a caption keyed by storage_key.\n"
        "4) For every plot in signals_plots, produce a caption keyed by plot_id.\n"
        "Caption must explain what the plot shows, what trend/pattern is visible, and 1 practical takeaway.\n"
        "Use provided data only; do not invent numbers.\n"
        "If data is insufficient, describe qualitatively and say what additional data would help.\n"
        "JSON schema example:\n"
        f"{json.dumps(schema_hint, ensure_ascii=False)}\n"
        "Data:\n"
        f"{json.dumps(compact, ensure_ascii=False)}"
    )

    raw = _gemini_generate_json(prompt, config)
    parsed = LLMReportOutput.model_validate(raw)
    return parsed.model_dump()