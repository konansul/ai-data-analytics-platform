from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from backend.app.cleaning.cleaning_agent.llm_client import LLMClient, LLMUnavailableError

from .prompt import make_static_rebuild_prompt
from .schemas import validate_static_result
from .utils import df_to_tsv_for_llm, csv_from_llm_to_df


def rebuild_static_table(
    df: pd.DataFrame,
    *,
    pre_profile: Optional[Dict[str, Any]] = None,
    use_llm: bool = False,
    llm_model: str = "gemini-2.5-flash",
    llm_client: Optional[LLMClient] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Rebuilds static / report-like Excel sheets (wide-by-years, pivot-ish) into long format.

    Returns:
      (rebuilt_df, report)

    On failure: returns (original_df, report with error).
    """

    report: Dict[str, Any] = {
        "enabled": bool(use_llm),
        "model": llm_model,
        "status": "skipped" if not use_llm else "started",
        "error": None,
    }

    if df is None or df.empty:
        report["status"] = "empty_input"
        return df, report

    if not use_llm:
        return df, report

    payload: Dict[str, Any] = dict(pre_profile or {})
    payload["table_tsv"] = df_to_tsv_for_llm(df)

    prompt = make_static_rebuild_prompt(payload)

    try:
        client = llm_client or LLMClient.from_env(model=llm_model)

        raw = client.complete(prompt)
        parsed = client.extract_json(raw)

        result = validate_static_result(parsed)

        rebuilt = csv_from_llm_to_df(result.csv)
        if rebuilt is None or rebuilt.empty:
            raise ValueError("LLM returned empty dataframe")

        report["status"] = "ok"
        report["rows_before"] = int(df.shape[0])
        report["cols_before"] = int(df.shape[1])
        report["rows_after"] = int(rebuilt.shape[0])
        report["cols_after"] = int(rebuilt.shape[1])
        report["meta"] = getattr(result, "meta", None)

        return rebuilt, report

    except (LLMUnavailableError, ValueError, KeyError, TypeError) as e:
        report["status"] = "failed"
        report["error"] = f"{type(e).__name__}: {e}"
        return df, report