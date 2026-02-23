# backend/app/cleaning/cleaning_steps/main_pipeline.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from backend.app.profiling.profiling import profile_dataframe
from backend.app.cleaning.cleaning_agent.cleaning_policy_agent import build_cleaning_plan
from backend.app.cleaning.cleaning_agent.schemas import CleaningPlan
from backend.app.cleaning.cleaning_agent.llm_client import LLMClient

from backend.app.cleaning.cleaning_static_file.agent import rebuild_static_table

from ._01_normalize import normalize_columns
from ._02_trim_strings import trim_strings
from ._03_standardize_missing import standardize_missing
from ._04_cast_types import cast_types
from ._05_encode_booleans import encode_booleans
from ._06_drop_rules import apply_drop_rules
from ._07_datetime_inference import infer_datetime_columns
from ._08_deduplicate import deduplicate
from ._09_outliers import handle_outliers
from ._10_impute_missing import impute_missing_values


def run_cleaning_pipeline(
    df: pd.DataFrame,
    missing_threshold: float = 0.5,
    row_missing_threshold: float = 0.80,
    datetime_success_ratio: float = 0.8,
    impute: bool = True,
    numeric_strategy: Optional[str] = "mean",
    categorical_strategy: Optional[str] = "mode",
    datetime_strategy: Optional[str] = None,
    fill_value: Union[int, float, str, None] = 0,
    categorical_numeric_max_unique: int = 20,
    use_llm: bool = False,
    llm_model: str = "gemini-2.5-flash",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    report: Dict[str, Any] = {}

    # Create one shared LLM client for the whole pipeline (policy + static rebuild),
    # but only if LLM is enabled.
    llm_client: Optional[LLMClient] = None
    if use_llm:
        try:
            llm_client = LLMClient.from_env(model=llm_model)
        except Exception:
            llm_client = None

    # 1) Pre-profile
    pre_profile = profile_dataframe(df)
    report["pre_profile"] = pre_profile

    # 2) Build cleaning plan (LLM optional)
    plan: CleaningPlan = build_cleaning_plan(
        pre_profile,
        use_llm=use_llm,
        model=llm_model,
        llm_client=llm_client,
    )
    report["cleaning_plan"] = plan.to_dict()

    # 3) If static table detected -> rebuild to normalized CSV-like table FIRST
    #    Then re-profile + rebuild plan again, because schema & types change.
    report["static_rebuild"] = {
        "triggered": False,
        "attempted": False,
        "success": False,
    }

    if bool(getattr(plan, "is_static_table", False)) and bool(getattr(plan, "static_rebuild_recommended", False)):
        report["static_rebuild"]["triggered"] = True
        report["static_rebuild"]["attempted"] = True

        try:
            rebuilt_df, static_report = rebuild_static_table(
                df,
                pre_profile=pre_profile,
                use_llm=use_llm,
                llm_model=llm_model,
                llm_client=llm_client,
            )
            report["static_rebuild"]["success"] = True
            report["static_rebuild"].update(static_report)

            # Replace working DF with rebuilt version
            df = rebuilt_df

            # Re-profile on rebuilt DF
            pre_profile2 = profile_dataframe(df)
            report["pre_profile_after_static_rebuild"] = pre_profile2

            # Rebuild plan on rebuilt DF
            plan = build_cleaning_plan(
                pre_profile2,
                use_llm=use_llm,
                model=llm_model,
                llm_client=llm_client,
            )
            report["cleaning_plan_after_static_rebuild"] = plan.to_dict()


        except Exception as e:
            report["static_rebuild"]["success"] = False
            report["static_rebuild"]["error"] = f"{type(e).__name__}: {e}"
            print("STATIC REBUILD FAILED:", e)

    # 4) Apply the plan as usual
    enabled: Dict[str, bool] = dict(plan.enabled_steps or {})
    params: Dict[str, Any] = dict(plan.params or {})

    missing_threshold = float(params.get("missing_threshold", missing_threshold))
    row_missing_threshold = float(params.get("row_missing_threshold", row_missing_threshold))
    datetime_success_ratio = float(params.get("datetime_success_ratio", datetime_success_ratio))
    numeric_strategy = params.get("numeric_strategy", numeric_strategy)
    categorical_strategy = params.get("categorical_strategy", categorical_strategy)
    datetime_strategy = params.get("datetime_strategy", datetime_strategy)
    fill_value = params.get("fill_value", fill_value)
    categorical_numeric_max_unique = int(params.get("categorical_numeric_max_unique", categorical_numeric_max_unique))

    impute_enabled = bool(enabled.get("impute_missing", True))
    if not impute:
        impute_enabled = False

    clean_df = df.copy()

    if enabled.get("normalize", True):
        clean_df, normalized_map = normalize_columns(clean_df)
        report["normalize"] = {"enabled": True, "renamed_columns": normalized_map}
    else:
        report["normalize"] = {"enabled": False}

    if enabled.get("trim_strings", True):
        clean_df, trim_report = trim_strings(clean_df)
        report["trim_strings"] = trim_report
    else:
        report["trim_strings"] = {"enabled": False}

    if enabled.get("standardize_missing", True):
        clean_df, missing_report = standardize_missing(clean_df)
        report["standardize_missing"] = missing_report
    else:
        report["standardize_missing"] = {"enabled": False}

    if enabled.get("cast_types", True):
        clean_df, cast_report = cast_types(clean_df)
        report["cast_types"] = cast_report
    else:
        report["cast_types"] = {"enabled": False}

    if enabled.get("encode_booleans", True):
        clean_df, bool_report = encode_booleans(clean_df)
        report["encode_booleans"] = bool_report
    else:
        report["encode_booleans"] = {"enabled": False}

    if enabled.get("drop_rules", True):
        drop_rows_flag = bool(params.get("drop_rows", True))
        ignore_cols = params.get("ignore_columns_for_row_drop")
        clean_df, drop_report = apply_drop_rules(
            clean_df,
            missing_threshold=missing_threshold,
            row_missing_threshold=row_missing_threshold,
            drop_rows=drop_rows_flag,
            ignore_columns_for_row_drop=ignore_cols,
        )
        report["drop_rules"] = {"enabled": True, **drop_report}
    else:
        report["drop_rules"] = {
            "enabled": False,
            "missing_threshold": float(missing_threshold),
            "row_missing_threshold": float(row_missing_threshold),
        }

    if enabled.get("datetime_inference", True):
        clean_df, inferred_dt_cols = infer_datetime_columns(
            clean_df,
            datetime_success_ratio=datetime_success_ratio,
        )
        report["datetime_inference"] = {
            "enabled": True,
            "inferred_datetime_columns": inferred_dt_cols,
            "datetime_success_ratio": float(datetime_success_ratio),
        }
    else:
        report["datetime_inference"] = {
            "enabled": False,
            "datetime_success_ratio": float(datetime_success_ratio),
        }

    if enabled.get("deduplicate", True):
        clean_df, dedup_report = deduplicate(clean_df)
        report["deduplicate"] = dedup_report
    else:
        report["deduplicate"] = {"enabled": False}

    if enabled.get("outliers", True):
        out_method = str(params.get("outliers_method", params.get("method", "quantile"))).lower()
        if out_method not in {"quantile", "iqr"}:
            out_method = "quantile"

        q = params.get("quantiles", (0.01, 0.99))
        try:
            q_lo = float(q[0])
            q_hi = float(q[1])
        except Exception:
            q_lo, q_hi = 0.01, 0.99

        if not (0.0 < q_lo < q_hi < 1.0):
            q_lo, q_hi = 0.01, 0.99

        iqr_k = float(params.get("iqr_k", 1.5))
        min_rows = int(params.get("outliers_min_rows", params.get("min_rows", 30)))
        skip_low_unique_ratio = float(params.get("skip_low_unique_ratio", 0.02))

        clean_df, out_report = handle_outliers(
            clean_df,
            enabled=True,
            method=out_method,
            quantiles=(q_lo, q_hi),
            iqr_k=iqr_k,
            min_rows=min_rows,
            skip_low_unique_ratio=skip_low_unique_ratio,
        )
        report["outliers"] = out_report
    else:
        report["outliers"] = {"enabled": False}

    if impute_enabled:
        clean_df, imputation_report = impute_missing_values(
            clean_df,
            enabled=True,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
            datetime_strategy=datetime_strategy,
            fill_value=fill_value,
            categorical_numeric_max_unique=categorical_numeric_max_unique,
        )
        report["imputation"] = imputation_report
    else:
        report["imputation"] = {"enabled": False, "skipped": True}

    post_profile = profile_dataframe(clean_df)
    report["post_profile"] = post_profile

    report["rows_before"] = int(df.shape[0])
    report["cols_before"] = int(df.shape[1])
    report["rows_after"] = int(clean_df.shape[0])
    report["cols_after"] = int(clean_df.shape[1])

    return clean_df, report