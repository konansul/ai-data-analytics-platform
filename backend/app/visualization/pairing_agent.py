import json
import re
from typing import Any, Dict, List, Optional

from backend.app.visualization.schemas import ColumnPairing, ColumnPairingPlan
from backend.app.visualization.llm_utils import LLMUtils


class PairingAgent:
    def __init__(self, llm_utils):
        self.llm = llm_utils

    def get_pairings(
            self,
            dataset_id: str,
            summary: Dict[str, Any],
            profile: Dict[str, Any],
            metrics: Optional[Dict[str, Any]] = None,
    ) -> ColumnPairingPlan:

        if not metrics:
            metrics = profile.get("_metrics") or {}
        pairings = self._get_pairings(summary, metrics)
        return ColumnPairingPlan(dataset_id=dataset_id, pairings=pairings)

    def _get_pairings(
            self,
            summary: Dict[str, Any],
            metrics: Dict[str, Any],
    ) -> List[ColumnPairing]:

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

        data = self.llm.safe_llm_call(prompt)
        raw_list = self.llm.ensure_list(data, "pairings")

        pairings: List[ColumnPairing] = []
        for p in raw_list:
            if not isinstance(p, dict):
                continue
            cols = p.get("columns")
            if not isinstance(cols, list) or not cols:
                continue

            template = self._infer_template(cols, summary)
            if template is None:
                continue

            pairings.append(
                ColumnPairing(
                    columns=cols,
                    rationale=str(p.get("rationale", "")).strip(),
                    rank=p.get("rank"),
                    score=p.get("score"),
                    template=template,
                )
            )

        pairings = self._dedupe_pairings(pairings)
        pairings = self._assign_ranks_if_missing(pairings)
        pairings = self._enforce_numeric_coverage(pairings, summary, metrics)
        pairings.sort(key=lambda x: x.rank if x.rank is not None else 10 ** 9)
        return pairings

    def _build_column_meta_block(self, summary: Dict[str, Any]) -> str:
        meta = summary.get("column_meta") or {}
        num_set = set(summary.get("numeric_columns", []))
        cat_set = set(summary.get("categorical_columns", []))
        dt_set = set(summary.get("date_columns", []))
        card = summary.get("cardinality") or {}

        lines = []
        for col in summary.get("columns", []):
            role = (
                "numeric" if col in num_set
                else "datetime" if col in dt_set
                else "categorical" if col in cat_set
                else "unknown"
            )
            col_card = card.get(col)
            col_m = meta.get(col) or {}
            parts = [f"{col} [{role}]"]
            if col_card is not None:
                parts.append(f"cardinality={col_card}")
            if col_m.get("is_unique") is not None:
                parts.append(f"unique={col_m.get('is_unique')}")
            if col_m.get("is_id"):
                parts.append("is_id=true")
            lines.append(", ".join(parts))

        return "\n".join(lines)

    def _infer_template(self, cols: List[str], summary: Dict[str, Any]) -> Optional[str]:
        num = set(summary.get("numeric_columns", []))
        cat = set(summary.get("categorical_columns", []))
        dt = set(summary.get("date_columns", []))

        if len(cols) == 1:
            return "num_univariate" if cols[0] in num else None
        if len(cols) != 2:
            return None

        a, b = cols
        if (a in dt and b in num) or (b in dt and a in num):
            return "date_numeric"
        if (a in cat and b in num) or (b in cat and a in num):
            return "cat_numeric"
        if a in num and b in num and a != b:
            return "num_num"
        return None

    def _dedupe_pairings(self, pairings: List[ColumnPairing]) -> List[ColumnPairing]:
        seen = set()
        out = []
        for p in pairings:
            key = tuple(sorted(p.columns))
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out

    def _assign_ranks_if_missing(self, pairings: List[ColumnPairing]) -> List[ColumnPairing]:
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

    def _is_id_like_numeric(self, col: str, summary: Dict[str, Any]) -> bool:
        name = col.lower().strip()
        if re.fullmatch(r"id", name) or name.endswith("_id") or name.startswith("id_"):
            return True
        meta = (summary.get("column_meta") or {}).get(col, {})
        if meta.get("is_id"):
            return True
        card = meta.get("cardinality")
        uniq = meta.get("is_unique")
        if uniq and isinstance(card, int) and card > 50:
            return True
        return False

    def _enforce_numeric_coverage(
            self,
            pairings: List[ColumnPairing],
            summary: Dict[str, Any],
            metrics: Dict[str, Any],
    ) -> List[ColumnPairing]:

        numeric_cols = [
            c for c in summary.get("numeric_columns", [])
            if not self._is_id_like_numeric(c, summary)
        ]
        covered = {c for p in pairings for c in p.columns if c in numeric_cols}
        missing = [c for c in numeric_cols if c not in covered]

        if not missing:
            return pairings

        date_cols = summary.get("date_columns") or []
        cat_cols = summary.get("categorical_columns") or []

        best_cat: Optional[str] = None
        best_card: Optional[int] = None
        card = summary.get("cardinality") or {}
        for c in cat_cols:
            c_card = card.get(c)
            if isinstance(c_card, int) and (best_card is None or c_card < best_card):
                best_cat, best_card = c, c_card

        corr_pairs = metrics.get("correlations") or []

        def best_numeric_partner(target: str) -> Optional[str]:
            for entry in corr_pairs:
                entry_cols = entry.get("columns") or []
                if len(entry_cols) == 2 and target in entry_cols:
                    other = entry_cols[0] if entry_cols[1] == target else entry_cols[1]
                    if other in numeric_cols:
                        return other
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