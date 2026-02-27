"""
VisualizationAgent — two-stage LLM planning pipeline
=====================================================

Stage 1 · Visualization Pairing Agent
    Decides WHAT to visualize: produces a ranked, scored list of column pairings.
    Rules enforced here:
      - Only allowed templates: date_numeric | cat_numeric | num_num | num_univariate
      - Coverage guarantee: every non-ID numeric column appears in ≥1 pairing
      - Never chooses plot types

Stage 2 · Visualization Plot Selection Agent
    Decides HOW to visualize each approved pairing.
    Rules enforced here:
      - Never drops approved pairings
      - Never invents new column combinations
      - Anchors plot-type choices on the pairing template
"""

import json
import logging
import os
import re
import requests
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

from backend.app.cleaning_agent.llm_client import LLMClient
from backend.app.visualization.schemas import (
    ColumnPairing,
    ColumnPairingPlan,
    PlotConfig,
    VisualizationPlan,
)

PLOT_TYPES = ["line", "bar", "scatter", "histogram", "box", "heatmap", "pie"]


class VisualizationAgent:
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.gemini_client = LLMClient.from_env(model=model)
        self.atllama_url = os.getenv("ATLLAMA_API_URL")

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def create_plan(
            self,
            dataset_id: str,
            profile: Dict[str, Any],
            metrics: Dict[str, Any] | None = None,
    ) -> VisualizationPlan:
        """
        Runs both stages and returns a VisualizationPlan that includes both
        the approved pairings (Stage 1) and the resolved plots (Stage 2).
        """
        logger.info("[VizAgent] create_plan called. dataset_id=%s, profile top-level keys=%s",
                    dataset_id, list(profile.keys()))

        summary = self._summarize_profile(profile)

        logger.info("[VizAgent] _summarize_profile result — numeric=%s  cat=%s  date=%s  total_cols=%s",
                    summary["numeric_columns"],
                    summary["categorical_columns"],
                    summary["date_columns"],
                    len(summary["columns"]))

        if not summary["columns"]:
            logger.error(
                "[VizAgent] EMPTY column list after profile parsing. "
                "Profile keys were: %s. Cannot generate visualizations.",
                list(profile.keys())
            )
            return VisualizationPlan(dataset_id=dataset_id, pairings=[], plots=[])

        # Prefer explicitly passed metrics; fall back to profile-embedded ones
        if not metrics:
            metrics = profile.get("_metrics") or {}

        # Stage 1: Pairing Agent - WHAT to visualize
        approved_pairings = self._get_pairings(summary, metrics)
        logger.info("[VizAgent] Stage 1 complete — %d pairings approved", len(approved_pairings))

        # Stage 2: Plot Selection Agent - HOW to visualize
        plots = self._get_plot_configs(approved_pairings, summary, metrics)
        logger.info("[VizAgent] Stage 2 complete — %d plots generated", len(plots))

        return VisualizationPlan(
            dataset_id=dataset_id,
            pairings=approved_pairings,
            plots=plots,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public single-stage entry points (used by the split API endpoints)
    # ─────────────────────────────────────────────────────────────────────────

    def get_pairings(
            self,
            dataset_id: str,
            profile: Dict[str, Any],
            metrics: Dict[str, Any] | None = None,
    ) -> "ColumnPairingPlan":
        """
        Stage 1 only — returns ranked column pairings without selecting plot types.
        Called by POST /visualization/pairings.
        """
        summary = self._summarize_profile(profile)
        if not metrics:
            metrics = profile.get("_metrics") or {}
        pairings = self._get_pairings(summary, metrics)
        return ColumnPairingPlan(dataset_id=dataset_id, pairings=pairings)

    def get_plots(
            self,
            dataset_id: str,
            profile: Dict[str, Any],
            selected_pairings: list,
            metrics: Dict[str, Any] | None = None,
    ) -> List[PlotConfig]:
        """
        Stage 2 only — takes user-selected pairings and returns PlotConfigs.
        Called by POST /visualization/plots.
        """
        summary = self._summarize_profile(profile)
        if not metrics:
            metrics = profile.get("_metrics") or {}
        plots = self._get_plot_configs(selected_pairings, summary, metrics)
        return plots

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1: Pairing Agent
    # ─────────────────────────────────────────────────────────────────────────

    def _get_pairings(
            self, summary: Dict[str, Any], metrics: Dict[str, Any]
    ) -> List[ColumnPairing]:
        """
        Stage 1: Decide WHAT to visualize.
        Enforces:
          - Only allowed templates
          - Ranking + scoring
          - Numeric coverage guarantee (non-ID numeric columns)
        """
        # Build a compact column-metadata block to help the LLM reason about types
        col_meta_block = self._build_column_meta_block(summary)

        prompt = f"""
You are the Visualization Pairing Agent.

GOAL
Return a RANKED, SCORED list of the most informative column pairings to visualize.
You DO NOT choose plot types — that is handled by a separate agent.

INPUTS (no raw data values are included):

COLUMN METADATA:
{col_meta_block}

STATISTICAL METRICS:
{json.dumps(metrics, indent=2)}

DATASET SUMMARY (column lists + cardinality):
{json.dumps({k: v for k, v in summary.items() if k != "column_meta"}, indent=2)}

ALLOWED TEMPLATES — use ONLY these four:
  1. (date, numeric)      — time-series relationship
  2. (categorical, numeric) — group comparison
  3. (numeric, numeric)   — correlation / scatter
  4. (numeric)            — univariate distribution

RULES
1. COVERAGE GUARANTEE — every non-ID numeric column MUST appear in ≥1 pairing unless it is
   explicitly ID-like (column name ends with _id, starts with id_, or is named exactly "id")
   or has is_unique=true with cardinality > 50.
2. DO NOT mention chart or plot type names. This agent only selects columns.
3. DO NOT invent pairing patterns outside the four templates above.
4. Assign a rank (1 = highest priority) and a score (0.0–1.0 relevance/confidence).
5. Return ONLY strict JSON — no markdown fences, no commentary:

{{
  "pairings": [
    {{
      "columns": ["colA", "colB"],
      "rationale": "short, specific reason",
      "rank": 1,
      "score": 0.95
    }},
    {{
      "columns": ["numericCol"],
      "rationale": "univariate distribution check",
      "rank": 4,
      "score": 0.60
    }}
  ]
}}
"""

        data = self._safe_llm_call(prompt)
        raw_list = self._ensure_list(data, "pairings")

        # ── Parse + validate ───────────────────────────────────────────────
        pairings: List[ColumnPairing] = []
        for p in raw_list:
            if not isinstance(p, dict):
                continue
            cols = p.get("columns")
            if not isinstance(cols, list) or not cols or any(
                    not isinstance(c, str) for c in cols
            ):
                continue

            rationale = str(p.get("rationale", "")).strip() or "Selected for signal and interpretability."
            rank = p.get("rank")
            score = p.get("score")

            template = self._infer_template(cols, summary)
            if template is None:
                continue  # reject pairings that violate allowed templates

            pairings.append(
                ColumnPairing(
                    columns=cols,
                    rationale=rationale,
                    rank=int(rank) if isinstance(rank, (int, float)) else None,
                    score=float(score) if isinstance(score, (int, float)) else None,
                    template=template,
                )
            )

        pairings = self._dedupe_pairings(pairings)
        pairings = self._assign_ranks_if_missing(pairings)

        # Coverage guarantee — add fallback pairings for any uncovered numeric cols
        pairings = self._enforce_numeric_coverage(pairings, summary, metrics)

        # Final sort by rank (stable)
        pairings.sort(key=lambda x: x.rank if x.rank is not None else 10**9)
        return pairings

    def _build_column_meta_block(self, summary: Dict[str, Any]) -> str:
        """
        Formats a human-readable column-metadata table for the Stage 1 prompt.
        Includes: column name, inferred role, cardinality, uniqueness, and
        basic stats (mean, sd, cv) when available.
        """
        meta = summary.get("column_meta") or {}
        num_set = set(summary.get("numeric_columns", []))
        cat_set = set(summary.get("categorical_columns", []))
        dt_set  = set(summary.get("date_columns", []))
        card    = summary.get("cardinality") or {}

        lines = []
        for col in summary.get("columns", []):
            role = (
                "numeric"     if col in num_set else
                "datetime"    if col in dt_set  else
                "categorical" if col in cat_set else
                "unknown"
            )
            col_card = card.get(col)
            col_m    = meta.get(col) or {}
            is_uniq  = col_m.get("is_unique")
            is_id    = col_m.get("is_id")

            parts = [f"  {col} [{role}]"]
            if col_card is not None:
                parts.append(f"cardinality={col_card}")
            if is_uniq is not None:
                parts.append(f"unique={is_uniq}")
            if is_id:
                parts.append("is_id=true")

            lines.append(", ".join(parts))

        return "\n".join(lines) if lines else "(no column metadata available)"

    def _infer_template(
            self, cols: List[str], summary: Dict[str, Any]
    ) -> Optional[str]:
        num = set(summary.get("numeric_columns", []))
        cat = set(summary.get("categorical_columns", []))
        dt  = set(summary.get("date_columns", []))

        if len(cols) == 1:
            return "num_univariate" if cols[0] in num else None

        if len(cols) != 2:
            return None

        a, b = cols[0], cols[1]
        if (a in dt and b in num) or (b in dt and a in num):
            return "date_numeric"
        if (a in cat and b in num) or (b in cat and a in num):
            return "cat_numeric"
        if a in num and b in num and a != b:
            return "num_num"
        return None

    def _dedupe_pairings(
            self, pairings: List[ColumnPairing]
    ) -> List[ColumnPairing]:
        seen: set = set()
        out: List[ColumnPairing] = []
        for p in pairings:
            key = tuple(sorted(p.columns))  # order-agnostic deduplication
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out

    def _assign_ranks_if_missing(
            self, pairings: List[ColumnPairing]
    ) -> List[ColumnPairing]:
        any_rank = any(p.rank is not None for p in pairings)
        if not any_rank:
            for i, p in enumerate(pairings, start=1):
                p.rank = i
            return pairings
        used = sorted({p.rank for p in pairings if p.rank is not None})
        next_rank = (used[-1] + 1) if used else 1
        for p in pairings:
            if p.rank is None:
                p.rank = next_rank
                next_rank += 1
        return pairings

    def _is_id_like_numeric(
            self, col: str, summary: Dict[str, Any]
    ) -> bool:
        name = col.lower().strip()
        if re.fullmatch(r"id", name) or name.endswith("_id") or name.startswith("id_"):
            return True
        meta = (summary.get("column_meta") or {}).get(col, {})
        if meta.get("is_id") is True:
            return True
        card = meta.get("cardinality")
        uniq = meta.get("is_unique")
        if uniq is True and isinstance(card, int) and card > 50:
            return True
        return False

    def _enforce_numeric_coverage(
            self,
            pairings: List[ColumnPairing],
            summary: Dict[str, Any],
            metrics: Dict[str, Any],
    ) -> List[ColumnPairing]:
        """
        Coverage guarantee: every non-ID numeric column must appear in ≥1 pairing.
        Appends fallback pairings at the end of the ranked list for any gaps.
        Priority order for fallback pairing partner:
          date column → best categorical (lowest cardinality) → correlated numeric → univariate
        """
        numeric_cols = [
            c for c in summary.get("numeric_columns", [])
            if not self._is_id_like_numeric(c, summary)
        ]
        covered = {c for p in pairings for c in p.columns if c in numeric_cols}
        missing = [c for c in numeric_cols if c not in covered]

        if not missing:
            return pairings

        date_cols = summary.get("date_columns") or []
        cat_cols  = summary.get("categorical_columns") or []

        # Pick lowest-cardinality categorical column for readable bars
        best_cat: Optional[str] = None
        best_card: Optional[int] = None
        card = summary.get("cardinality") or {}
        for c in cat_cols:
            c_card = card.get(c)
            if isinstance(c_card, int) and (best_card is None or c_card < best_card):
                best_cat, best_card = c, c_card

        # Best correlated numeric partner per column
        corr_pairs = metrics.get("correlations") or []

        def best_numeric_partner(target: str) -> Optional[str]:
            for entry in corr_pairs:
                entry_cols = entry.get("columns") or []
                if len(entry_cols) == 2 and target in entry_cols:
                    other = entry_cols[0] if entry_cols[1] == target else entry_cols[1]
                    if other in numeric_cols:
                        return other
            # Fallback: any other numeric column
            for c in numeric_cols:
                if c != target:
                    return c
            return None

        max_rank = max((p.rank for p in pairings if p.rank is not None), default=0)
        for m in missing:
            max_rank += 1
            if date_cols:
                pairings.append(ColumnPairing(
                    columns=[date_cols[0], m],
                    rationale="Coverage fallback: paired with date column.",
                    rank=max_rank,
                    score=0.5,
                    template="date_numeric",
                ))
            elif best_cat:
                pairings.append(ColumnPairing(
                    columns=[best_cat, m],
                    rationale="Coverage fallback: paired with lowest-cardinality categorical column.",
                    rank=max_rank,
                    score=0.5,
                    template="cat_numeric",
                ))
            else:
                partner = best_numeric_partner(m)
                if partner:
                    pairings.append(ColumnPairing(
                        columns=[partner, m],
                        rationale="Coverage fallback: numeric-numeric relationship.",
                        rank=max_rank,
                        score=0.5,
                        template="num_num",
                    ))
                else:
                    pairings.append(ColumnPairing(
                        columns=[m],
                        rationale="Coverage fallback: univariate distribution view.",
                        rank=max_rank,
                        score=0.4,
                        template="num_univariate",
                    ))

        return pairings

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2: Plot Selection Agent
    # ─────────────────────────────────────────────────────────────────────────

    def _get_plot_configs(
            self,
            pairings: List[ColumnPairing],
            summary: Dict[str, Any],
            metrics: Dict[str, Any],
    ) -> List[PlotConfig]:
        """
        Stage 2: Decide HOW to visualize each approved pairing.
        Strict rules:
          - Must produce exactly one entry per approved pairing (no drops).
          - Must not invent new column combinations.
          - Plot type must be compatible with the pairing's template.
        """
        pairing_data = [
            p.model_dump() if hasattr(p, "model_dump") else p.dict()
            for p in pairings
        ]

        # Build a compact template → recommended plot guidance block
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

        data = self._safe_llm_call(prompt)
        raw_list = self._ensure_list(data, "plots")

        plots: List[PlotConfig] = []
        for i, pairing in enumerate(pairings):
            item = (
                raw_list[i]
                if i < len(raw_list) and isinstance(raw_list[i], dict)
                else {}
            )
            plot = self._validate_or_fallback_plot(item, pairing, summary, metrics)
            plot.source_pairing = pairing.columns   # back-reference for traceability
            plots.append(plot)

        return plots

    def _validate_or_fallback_plot(
            self,
            item: Dict[str, Any],
            pairing: ColumnPairing,
            summary: Dict[str, Any],
            metrics: Dict[str, Any],
    ) -> PlotConfig:
        """
        Validates the LLM-produced plot config against the approved pairing.
        Falls back to deterministic defaults when the LLM output is invalid or missing.
        """
        cols     = pairing.columns
        template = pairing.template or self._infer_template(cols, summary) or "num_univariate"

        num = set(summary.get("numeric_columns", []))
        cat = set(summary.get("categorical_columns", []))
        dt  = set(summary.get("date_columns", []))

        # ── Resolve x / y from pairing (not from LLM — prevents column invention) ──
        x, y = None, None
        if len(cols) == 1:
            x = cols[0]
        else:
            a, b = cols[0], cols[1]
            if template == "date_numeric":
                x = a if a in dt  else b
                y = b if x == a   else a
            elif template == "cat_numeric":
                x = a if a in cat else b
                y = b if x == a   else a
            elif template == "num_num":
                x, y = a, b

        # ── Plot types — validate LLM choice, fall back to template default ──
        plot_type = item.get("plot_type")
        alt_plot  = item.get("alt_plot_type")

        if plot_type not in PLOT_TYPES:
            plot_type, alt_plot = self._default_plot_types_for_template(template)

        if alt_plot is not None and alt_plot not in PLOT_TYPES:
            alt_plot = None

        # ── Constraints and warnings ──────────────────────────────────────────
        constraints = item.get("constraints") if isinstance(item.get("constraints"), dict) else {}
        warnings    = item.get("warnings")    if isinstance(item.get("warnings"),    list)  else []

        constraints, warnings = self._apply_default_constraints_and_warnings(
            template=template,
            x=x, y=y,
            summary=summary,
            metrics=metrics,
            constraints=constraints,
            warnings=warnings,
        )

        # ── Title and description ─────────────────────────────────────────────
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
            "date_numeric":   ("line",      None),
            "cat_numeric":    ("bar",       "box"),
            "num_num":        ("scatter",   "heatmap"),
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
        """
        Applies deterministic constraint and warning rules on top of
        whatever the LLM produced — setdefault so LLM values are preserved.
        """
        card  = summary.get("cardinality") or {}
        stats = (metrics.get("stats") or {})

        if template == "cat_numeric" and x:
            x_card = card.get(x)
            if isinstance(x_card, int) and x_card > 20:
                constraints.setdefault("top_k", 10)
                warnings.append(
                    f"High-cardinality axis '{x}' ({x_card} categories); top-10 applied for readability."
                )

        if template == "date_numeric":
            constraints.setdefault("resample", "W")        # weekly aggregation
            constraints.setdefault("rolling_window", 4)    # 4-period rolling mean

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

        # Deduplicate warnings (preserve order)
        warnings = list(dict.fromkeys(str(w).strip() for w in warnings if str(w).strip()))
        return constraints, warnings

    # ─────────────────────────────────────────────────────────────────────────
    # Explain endpoint
    # ─────────────────────────────────────────────────────────────────────────

    def explain_visualization(self, plot_title: str, axis_info: str) -> str:
        """
        Generates a plain-language insight for a rendered chart.
        Called by the /visualization/explain API endpoint.

        Args:
            plot_title: The chart title (e.g. "Sales over Time").
            axis_info:  Axis description string (e.g. "X-Axis: date, Y-Axis: sales").

        Returns:
            A concise analytical explanation of what the chart likely shows.
        """
        prompt = f"""
You are a data analyst explaining a chart to a business user.

CHART TITLE: {plot_title}
AXIS INFO:   {axis_info}

Write a concise, plain-language explanation (2–4 sentences) of:
1. What this chart is showing.
2. What patterns or insights a viewer should look for.

Do not make up specific numbers. Focus on what the chart type and axes reveal analytically.
Return only the explanation text — no bullet points, no JSON, no preamble.
"""
        try:
            return self.gemini_client.complete(prompt).strip()
        except Exception as e:
            return f"Could not generate explanation: {e}"

    # ─────────────────────────────────────────────────────────────────────────
    # LLM helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _safe_llm_call(self, prompt: str) -> Union[Dict, List]:
        try:
            if self.atllama_url:
                resp = requests.post(
                    self.atllama_url,
                    json={"prompt": prompt},
                    timeout=20,
                )
                resp.raise_for_status()
                return self.gemini_client.extract_json(resp.json().get("text", ""))
            raise RuntimeError("No ATLLaMa URL configured, falling back to Gemini.")
        except Exception:
            return self.gemini_client.extract_json(
                self.gemini_client.complete(prompt)
            )

    def _ensure_list(self, data: Any, key: str) -> List:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get(key, [])
        return []

    # ─────────────────────────────────────────────────────────────────────────
    # Profile summarizer
    # ─────────────────────────────────────────────────────────────────────────

    def _summarize_profile(self, profile: Dict) -> Dict:
        """
        Robustly extracts column signals from the cleaning pipeline profile.

        Handles all known profile structures:
          - profile["columns"]       as dict  {colName: {dtype, cardinality, ...}}
          - profile["columns"]       as list  [{"name": ..., "dtype": ...}, ...]
          - profile["column_stats"]  as dict  (alternate key used by some profilers)
          - profile["fields"]        as dict  (alternate key)
          - profile["schema"]        as dict  (alternate key)
          - profile["variables"]     as dict  (ydata-profiling style)

        Typed column lists (numeric_columns, categorical_columns, date_columns) are
        accepted from the profile directly, or inferred from dtype if absent.
        """
        # ── Step 1: Find the column metadata object, trying all known key variants ──
        COLUMN_KEYS = ["columns", "column_stats", "fields", "schema", "variables", "features"]
        cols_obj = None
        used_key = None
        for key in COLUMN_KEYS:
            val = profile.get(key)
            if val:
                cols_obj = val
                used_key = key
                break

        logger.info("[VizAgent] _summarize_profile: found column data under key=%r, type=%s",
                    used_key, type(cols_obj).__name__)

        # ── Step 2: Normalize to {name: meta_dict} ─────────────────────────────
        columns_dict: Dict[str, Any] = {}
        if isinstance(cols_obj, dict):
            columns_dict = cols_obj
        elif isinstance(cols_obj, list):
            for item in cols_obj:
                if isinstance(item, dict):
                    name = (
                            item.get("name")
                            or item.get("column")
                            or item.get("col")
                            or item.get("column_name")
                            or item.get("field")
                    )
                    if isinstance(name, str) and name:
                        columns_dict[name] = item

        logger.info("[VizAgent] _summarize_profile: normalized to %d columns: %s",
                    len(columns_dict), list(columns_dict.keys())[:20])

        # ── Step 3: Typed column lists — prefer explicit, infer from dtype if missing ──
        numeric_cols: List[str] = list(profile.get("numeric_columns") or [])
        cat_cols:     List[str] = list(profile.get("categorical_columns") or [])
        date_cols:    List[str] = list(profile.get("date_columns") or [])

        if not (numeric_cols or cat_cols or date_cols):
            logger.info("[VizAgent] No pre-classified column lists found — inferring from dtype fields")
            for name, meta in columns_dict.items():
                if not isinstance(meta, dict):
                    continue
                dtype = (
                        meta.get("dtype")
                        or meta.get("type")
                        or meta.get("data_type")
                        or meta.get("col_type")
                        or ""
                ).lower()
                if any(t in dtype for t in ["int", "float", "double", "number", "numeric", "decimal"]):
                    numeric_cols.append(name)
                elif any(t in dtype for t in ["date", "datetime", "time", "timestamp"]):
                    date_cols.append(name)
                else:
                    cat_cols.append(name)
        else:
            logger.info("[VizAgent] Using pre-classified lists from profile")

        logger.info("[VizAgent] Final column classification — numeric=%s  cat=%s  date=%s",
                    numeric_cols, cat_cols, date_cols)

        # ── Step 4: Cardinality map ────────────────────────────────────────────
        cardinality: Dict[str, Any] = {}
        for k, v in columns_dict.items():
            if not isinstance(v, dict):
                continue
            card = (
                    v.get("cardinality")
                    or v.get("n_unique")
                    or v.get("unique_count")
                    or v.get("distinct_count")
            )
            cardinality[k] = card

        # ── Step 5: Correlations and missingness (optional signals) ────────────
        corr_section = profile.get("correlation") or profile.get("correlations") or {}
        correlations = (
                corr_section.get("top_abs_pairs")
                or corr_section.get("pairs")
                or []
        )

        miss_section = profile.get("missingness") or profile.get("missing") or {}
        missingness = (
                miss_section.get("top_missing_columns")
                or miss_section.get("columns")
                or {}
        )

        return {
            "columns":             list(columns_dict.keys()),
            "numeric_columns":     numeric_cols,
            "categorical_columns": cat_cols,
            "date_columns":        date_cols,
            "cardinality":         cardinality,
            "correlations":        correlations,
            "missingness":         missingness,
            "column_meta": {
                k: {
                    "dtype": (
                            v.get("dtype") or v.get("type") or v.get("data_type")
                    ) if isinstance(v, dict) else None,
                    "cardinality": cardinality.get(k),
                    "is_unique":   v.get("is_unique") if isinstance(v, dict) else None,
                    "is_id":       v.get("is_id")     if isinstance(v, dict) else None,
                }
                for k, v in columns_dict.items()
                if isinstance(v, dict)
            },
        }


