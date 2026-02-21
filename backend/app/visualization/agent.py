# backend/app/visualization/agent.py
import json
import os
import requests
from typing import Any, Dict
from backend.app.cleaning.cleaning_agent.llm_client import LLMClient
from backend.app.visualization.schemas import PlotConfig, VisualizationPlan


class VisualizationAgent:
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.gemini_client = LLMClient.from_env(model=model)
        # Placeholder for the primary test model URL
        self.atllama_url = os.getenv("ATLLAMA_API_URL")

    def create_plan(self, dataset_id: str, profile: Dict[str, Any], metrics: Dict[str, Any] = None) -> VisualizationPlan:
        # 1. Summarize Profile (Reduce context, ensure privacy)
        summary = self._summarize_profile(profile)

        prompt = f"""
        You are a Data Visualization Expert in a multi-agent system.
        Your task is to recommend a set of insightful visualizations for a dataset based STRICTLY on its profile signals.
        
        CRITICAL: You DO NOT have access to raw rows. 
        
        DATASET PROFILE:
        {json.dumps(summary, indent=2)}
        
        STATISTICAL METRICS (SD, Mean, R^2):
        {json.dumps(metrics, indent=2) if metrics else "No extra metrics provided."}
        
       STRICT DIVERSITY RULES:
        1. **R-SQUARED (R²)**: If a pair has R² > 0.6, suggest ONE scatter plot. Mention the R² in the description.
        2. **STANDARD DEVIATION (SD)**: If a numeric column has a high SD relative to its Mean, you MUST suggest a 'box' or 'histogram' to show distribution/outliers. Mention the SD and Mean in the description.
        3. **MEAN/CATEGORICAL**: Use 'bar' charts to compare Means across categorical groups.
        4. **NO REPETITION**: Do not suggest 3 plots of the same type. Provide a mix (e.g., 1 scatter, 1 histogram, 1 bar).
        
        REQUIREMENTS:
        1. Suggest EXACTLY 3 plots that reveal patterns. (distributions, correlations, trends).
        2. Assign a 'plot_type' from: ["line", "bar", "scatter", "histogram", "box", "heatmap", "pie"].
        3. 'x_column' and 'y_column' must match the columns in the profile.
        4. ONLY suggest a Time Series plot if a valid date column is listed in the profile.
        5. For high cardinality categorical columns, use 'bar' with constraints (e.g., {{"top_k": 10}}).
        
        OUTPUT FORMAT:
        Return ONLY a valid JSON object with a key "plots" containing the configuration.
        Example:
        {{
            "plots": [
                {{
                    "title": "Sales by Region",
                    "plot_type": "bar",
                    "x_column": "Region",
                    "y_column": "Sales",
                    "description": "Top performing regions.",
                    "constraints": {{"top_k": 10}}
                }}
            ]
        }}
        """

        # 2. Test-First Policy: Try AtLLama (Primary), Fallback to Gemini (Secondary)
        try:
            if self.atllama_url:
                # Primary Path: Local/Private Model
                response = requests.post(
                    self.atllama_url,
                    json={"prompt": prompt},
                    timeout=10
                )
                response.raise_for_status()
                response_text = response.json().get("text", "")
                data = self._extract_json(response_text)
            else:
                # Force fallback if not configured
                raise RuntimeError("AtLLama not configured")

        except Exception:
            # Secondary Path: Gemini (Fallback)
            response_text = self.gemini_client.complete(prompt)
            data = self.gemini_client.extract_json(response_text)

        # 3. Parse and Validate
        plots = []
        for p in data.get("plots", []):
            try:
                plots.append(PlotConfig(**p))
            except Exception:
                continue

        return VisualizationPlan(dataset_id=dataset_id, plots=plots)


    def explain_visualization(self, plot_title: str, axis_info: str) -> str:
        """
        Asks the AI to explain the business insight of a specific chart.
        """
        prompt = f"""
        You are a generic Data Analyst.
        Interpret the following visualization context in 2 concise sentences.
        
        Chart Title: {plot_title}
        Data Context: {axis_info}
        
        Goal: Explain the potential insight or pattern this chart might reveal.
        Do NOT mention specific numbers unless provided.
        """

        try:
            # Try .complete() first (based on your old files)
            return self.gemini_client.complete(prompt)
        except AttributeError:
            # Fallback to .generate() if you updated the client
            return self.gemini_client.generate(prompt)

    def _extract_json(self, text: str) -> Dict:
        # Re-use the robust extraction from the existing client for consistency
        return self.gemini_client.extract_json(text)

    def _summarize_profile(self, profile: Dict) -> Dict:
        """Extract only signals relevant to visualization to prevent token waste."""
        return {
            "dataset_type": profile.get("dataset_type"),
            "n_rows": profile.get("n_rows"),
            "columns": list(profile.get("columns", {}).keys()),
            "numeric_columns": profile.get("numeric_columns", []),
            "categorical_columns": profile.get("categorical_columns", []),
            "date_columns": profile.get("date_columns", []),
            "correlations": profile.get("correlation", {}).get("top_abs_pairs", []),
            "missing_stats": profile.get("missingness", {}).get("top_missing_columns", {})
        }