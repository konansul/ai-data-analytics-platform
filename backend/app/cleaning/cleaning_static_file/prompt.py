from __future__ import annotations

from typing import Dict, Any


def make_static_rebuild_prompt(payload: Dict[str, Any]) -> str:
    return f"""
You are a STRICT data reconstruction agent.

You receive a statistical Excel-like table (messy headers, merged cells, "Unnamed" columns).
You MUST convert it into a NORMALIZED LONG-FORM CSV with EXACTLY 3 columns:

indicator,year,value

ABSOLUTE RULES (must follow):
1) Output ONLY valid JSON (no markdown, no commentary).
2) The "csv" field MUST be a valid CSV text.
3) The CSV header MUST be exactly:
   indicator,year,value
4) Every data row MUST have exactly 3 comma-separated fields.
5) "year" MUST be a 4-digit integer (e.g., 1990). Remove *, .0, spaces, etc.
6) "indicator" is the row label (string). It MUST NOT be empty.
7) "value" is numeric float OR empty (blank). No thousands separators. Use dot decimal.
8) Convert tokens "-", "—", "…", "...", "" (empty), "na", "n/a" to empty value.
9) DO NOT hallucinate values. Use only values present in the table.
10) DO NOT output wide format. Years MUST NOT be columns.

RECONSTRUCTION STEPS:
- Remove title/junk rows above the real header.
- Identify the header row that contains "Göstəricilər" (or similar indicator label) and many years.
- Use those years as the "year" values.
- For each indicator row, produce one output row per year.

QUALITY CHECK (before responding):
- If you cannot confidently find years and indicators, set confidence < 0.5 and still output best-effort CSV.
- Ensure at least 10 rows are produced if possible.
- Ensure years are within 1900..2100.

JSON SCHEMA:
{{
  "version": 1,
  "csv": "indicator,year,value\\n...",
  "notes": ["short bullet notes about what you detected and how many rows you output"],
  "confidence": 0.0
}}

INPUT:
- table_tsv: a TSV preview of the table (rows and columns)
- pre_profile: metadata signals (may be noisy)

payload:
{payload}
""".strip()