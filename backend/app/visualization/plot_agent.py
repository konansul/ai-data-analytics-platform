import json
from typing import Any, Dict, List, Optional, Tuple

from backend.app.visualization.schemas import PlotConfig, ColumnPairing

PLOT_TYPES = ["line", "bar", "scatter", "histogram", "box", "heatmap", "pie"]


class PlotAgent:
    def __init__(self, llm_utils):
        self.llm = llm_utils

    def get_plots(
        self,
        pairings: List[ColumnPairing],
        summary: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> List[PlotConfig]:
        metrics = metrics or {}

        pairing_data = [
            p.model_dump() if hasattr(p, "model_dump") else p.dict()
            for p in pairings
        ]

        template_guidance = """
        TEMPLATE → RECOMMENDED PLOT TYPES (use as primary guidance):
          date_numeric    → line (primary), bar (secondary) — add rolling_mean constraint where useful
          cat_numeric     → bar (primary), box (secondary)  — apply top_k if cardinality > 20
          num_num         → scatter (primary), heatmap (secondary) — add trend=true constraint
          num_univariate  → histogram (primary), box (secondary) — add bins constraint
        """

        prompt = f"""
        You are the Visualization Plot Selection Agent.

        GOAL
        For EACH approved pairing below, select the most appropriate plot type(s) and constraints.
        Focus on interpretability, readability, and analytical usefulness.

        CRITICAL RULES
        1. Return exactly ONE entry per approved pairing — DO NOT drop any pairing.
        2. DO NOT change columns or invent new column combinations.
        3. Use the pairing "template" field to anchor your plot-type selection.
        4. Add interpretability warnings where the plot could be misleading.
        {template_guidance}
        APPROVED PAIRINGS (from Stage 1 — these are final):
        {json.dumps(pairing_data, indent=2)}

        DATASET CONTEXT:
        {json.dumps({k: v for k, v in summary.items() if k != "column_meta"}, indent=2)}

        STATISTICAL METRICS:
        {json.dumps(metrics, indent=2)}

        ALLOWED PLOT TYPES: {json.dumps(PLOT_TYPES)}

        Return ONLY strict JSON — no markdown fences, no commentary:
        {{
          "plots": [
            {{
              "title": "descriptive chart title",
              "plot_type": "bar",
              "alt_plot_type": "box",
              "x_column": "region",
              "y_column": "sales",
              "color_column": null,
              "description": "one sentence explaining why this plot is useful",
              "constraints": {{"top_k": 10}},
              "warnings": ["optional interpretability warning"]
            }}
          ]
        }}
        """

        data = self.llm.safe_llm_call(prompt)
        raw_list = self.llm.ensure_list(data, "plots")

        plots: List[PlotConfig] = []
        for i, pairing in enumerate(pairings):
            item = raw_list[i] if i < len(raw_list) and isinstance(raw_list[i], dict) else {}
            plots.append(self._validate_or_fallback_plot(item, pairing, summary, metrics))
        return plots

    def _validate_or_fallback_plot(
            self,
            item: Dict[str, Any],
            pairing: ColumnPairing,
            summary: Dict[str, Any],
            metrics: Dict[str, Any],
    ) -> PlotConfig:

        cols = pairing.columns
        template = pairing.template or self._infer_template(cols, summary) or "num_univariate"

        num = set(summary.get("numeric_columns", []))
        cat = set(summary.get("categorical_columns", []))
        dt = set(summary.get("date_columns", []))

        x, y = None, None
        if len(cols) == 1:
            x = cols[0]
        else:
            a, b = cols[0], cols[1]
            if template == "date_numeric":
                x = a if a in dt else b
                y = b if x == a else a
            elif template == "cat_numeric":
                x = a if a in cat else b
                y = b if x == a else a
            elif template == "num_num":
                x, y = a, b

        plot_type = item.get("plot_type")
        alt_plot = item.get("alt_plot_type")

        if plot_type not in PLOT_TYPES:
            plot_type, alt_plot = self._default_plot_types_for_template(template)

        if alt_plot is not None and alt_plot not in PLOT_TYPES:
            alt_plot = None

        constraints = item.get("constraints") if isinstance(item.get("constraints"), dict) else {}
        warnings = item.get("warnings") if isinstance(item.get("warnings"), list) else []

        constraints, warnings = self._apply_default_constraints_and_warnings(
            template=template,
            x=x, y=y,
            summary=summary,
            metrics=metrics,
            constraints=constraints,
            warnings=warnings,
        )

        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            title = self._default_title(template, x, y)

        desc = item.get("description")
        if not isinstance(desc, str) or not desc.strip():
            desc = pairing.rationale

        return PlotConfig(
            title=title.strip(),
            plot_type=plot_type,
            alt_plot_type=alt_plot,
            x_column=x,
            y_column=y,
            description=desc,
            constraints=constraints,
            warnings=warnings,
        )

    def _default_plot_types_for_template(
            self, template: str
    ) -> Tuple[str, Optional[str]]:
        return {
            "date_numeric": ("line", None),
            "cat_numeric": ("bar", "box"),
            "num_num": ("scatter", "heatmap"),
            "num_univariate": ("histogram", "box"),
        }.get(template, ("histogram", "box"))

    def _default_title(
            self, template: str, x: Optional[str], y: Optional[str]
    ) -> str:
        if template == "num_univariate":
            return f"Distribution of {x}"
        if template == "date_numeric":
            return f"{y} over Time"
        if template == "cat_numeric":
            return f"{y} by {x}"
        return f"{y} vs {x}"

    def _apply_default_constraints_and_warnings(
            self,
            template: str,
            x: Optional[str],
            y: Optional[str],
            summary: Dict[str, Any],
            metrics: Dict[str, Any],
            constraints: Dict[str, Any],
            warnings: List[str],
    ) -> Tuple[Dict[str, Any], List[str]]:

        card = summary.get("cardinality") or {}
        stats = (metrics.get("stats") or {})

        if template == "cat_numeric" and x:
            x_card = card.get(x)
            if isinstance(x_card, int) and x_card > 20:
                constraints.setdefault("top_k", 10)
                warnings.append(
                    f"High-cardinality axis '{x}' ({x_card} categories); top-10 applied for readability."
                )

        if template == "date_numeric":
            constraints.setdefault("resample", "W")
            constraints.setdefault("rolling_window", 4)

        if template == "num_univariate" and x:
            constraints.setdefault("bins", 20)
            st = stats.get(x)
            if st and isinstance(st.get("cv"), (int, float)) and st["cv"] > 1.0:
                warnings.append(
                    f"High coefficient of variation (cv={st['cv']}) detected in '{x}'; "
                    "consider log scale or robust summaries."
                )

        if template == "num_num":
            constraints.setdefault("trend", True)

        warnings = list(dict.fromkeys(str(w).strip() for w in warnings if str(w).strip()))
        return constraints, warnings