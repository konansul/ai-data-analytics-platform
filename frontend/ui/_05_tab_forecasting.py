# frontend/ui/_05_tab_forecasting.py
from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
import streamlit as st

from ui import data_access


def _get_post_profile_from_runs_store(dataset_id: str) -> Optional[Dict[str, Any]]:
    item = st.session_state.get("runs_store", {}).get(dataset_id)
    if not item:
        return None
    report = item.get("report")
    if isinstance(report, dict):
        return report.get("post_profile") or report.get("pre_profile") or report
    return None


def _pretty_bool(v: Any) -> str:
    return "Yes ✅" if v else "No ❌"


def _pick_dt_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None

    for c in ["dt", "ds", "date", "datetime", "timestamp"]:
        if c in df.columns:
            return c

    for c in df.columns:
        lc = c.lower()
        if "date" in lc or "time" in lc:
            return c
    return None


def _to_datetime_sorted(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    out = df.copy()
    out[dt_col] = pd.to_datetime(out[dt_col], errors="coerce")
    out = out.dropna(subset=[dt_col]).sort_values(dt_col)
    return out


def _build_wide_tables(results_list: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Returns:
      wide_yhat_df: dt + one column per target (yhat)
      intervals_by_target: target -> df(dt, yhat, yhat_lower, yhat_upper) if available
    """
    wide = None
    intervals: Dict[str, pd.DataFrame] = {}

    for r in results_list:
        target = r.get("target", "unknown_target")
        preview = r.get("forecast_preview") or r.get("preview") or []
        if not isinstance(preview, list) or not preview:
            continue

        df = pd.DataFrame(preview)
        dt_col = _pick_dt_col(df) or "dt"
        if dt_col not in df.columns:
            continue

        cols = [dt_col]
        if "yhat" in df.columns:
            cols.append("yhat")
        if "yhat_lower" in df.columns:
            cols.append("yhat_lower")
        if "yhat_upper" in df.columns:
            cols.append("yhat_upper")

        df = df[cols].copy()
        df = _to_datetime_sorted(df, dt_col)

        if "yhat" in df.columns:
            part = df[[dt_col, "yhat"]].rename(columns={"yhat": target})
            if wide is None:
                wide = part
            else:
                wide = wide.merge(part, on=dt_col, how="outer")

        if {"yhat", "yhat_lower", "yhat_upper"}.issubset(df.columns):
            intervals[target] = df.rename(columns={dt_col: "dt"})[["dt", "yhat", "yhat_lower", "yhat_upper"]]

    if wide is None:
        wide = pd.DataFrame()

    dt_col = _pick_dt_col(wide)
    if dt_col and dt_col != "dt":
        wide = wide.rename(columns={dt_col: "dt"})
    if "dt" in wide.columns:
        wide = _to_datetime_sorted(wide, "dt")

    return wide, intervals


def render_tab_forecasting(dataset_id: str) -> None:
    st.subheader("Forecasting")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        version = st.selectbox("Dataset version", ["current", "raw"], index=0)
    with col2:
        horizon = st.number_input("Steps", min_value=1, max_value=3650, value=30, step=1)
    with col3:
        max_targets = st.selectbox("Max targets", [1, 2, 3, 4, 5], index=2)
    with col4:
        model = st.selectbox("Model", ["auto", "arima", "prophet", "random_forest"], index=0)

    user_intent = st.text_input(
        "Optional user intent",
        value="",
        placeholder="e.g. forecast meantemp for next 30 days",
    )

    st.divider()

    st.markdown("### 1. Forecast signals")

    if st.button("Generate forecast signals", type="primary"):
        try:
            signals = data_access.forecast_signals(dataset_id=dataset_id, version=version)
            st.session_state["forecast_signals"] = signals
            st.success("Signals generated.")
        except Exception as e:
            st.error(f"Signals error: {e}")
            return

    signals = st.session_state.get("forecast_signals")
    if signals:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Feasible", _pretty_bool(signals.get("feasible")))
        with c2:
            st.metric("Frequency", str(signals.get("inferred_frequency", "unknown")))
        with c3:
            st.metric("Datetime candidates", len(signals.get("datetime_candidates") or []))
        with c4:
            st.metric("Targets", len(signals.get("numeric_target_candidates") or []))

        with st.expander("Signals JSON", expanded=False):
            st.json(signals)

        if not signals.get("feasible"):
            st.warning(
                "This dataset does not look suitable for time-series forecasting: "
                "no valid datetime column or no numeric target columns."
            )
            st.info("In this case forecasting can be skipped — the system will proceed to analysis/report without forecast.")
            return
    else:
        st.info("Please click **Generate forecast signals** first.")
        return

    st.divider()

    st.markdown("### 2. Forecast planning agent")

    if st.button("Create forecast plan"):
        try:
            profile = _get_post_profile_from_runs_store(dataset_id)
            plan = data_access.forecast_plan(
                dataset_id=dataset_id,
                version=version,
                signals=signals,
                profile=profile,
                user_intent=user_intent or None,
                head_rows=int(10),
                max_targets=int(max_targets),
            )
            st.session_state["forecast_plan"] = plan
            st.success("Plan created.")
        except Exception as e:
            st.error(f"Plan error: {e}")
            return

    plan = st.session_state.get("forecast_plan")
    if plan:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Mode", str(plan.get("mode", "unknown")))
        with c2:
            st.metric("Datetime column", str(plan.get("datetime_column", "—")))
        with c3:
            st.metric("Targets", str(len(plan.get("targets") or [])))

        with st.expander("Plan JSON", expanded=False):
            st.json(plan)

        if plan.get("mode") in {"skip", "forecasting_skipped", "skipped"}:
            st.warning("Planner decided to skip forecasting for this dataset.")
            return
    else:
        st.info("Please click **Create forecast plan** first.")
        return

    st.divider()

    st.markdown("### 3. Forecast run")

    if st.button("Run forecast"):
        try:
            result = data_access.forecast_run(
                dataset_id=dataset_id,
                version=version,
                plan=plan,
                horizon=int(horizon),
                model=model,
            )
            st.session_state["forecast_result"] = result
            st.success("Forecast executed.")
        except Exception as e:
            st.error(f"Run error: {e}")
            return

    result = st.session_state.get("forecast_result")
    if not result:
        st.info("Click **Run forecast** to get results.")
        return

    with st.expander("Execution JSON", expanded=False):
        st.json(result)

    results_list = result.get("results") or []
    if not isinstance(results_list, list) or len(results_list) == 0:
        st.info("Execution returned empty results.")
        return

    wide_df, intervals_by_target = _build_wide_tables(results_list)

    st.markdown("### Forecast preview (yhat)")
    if wide_df.empty:
        st.info("No preview data available to build combined table.")
        return

    st.dataframe(wide_df, width='stretch')

    if intervals_by_target:
        with st.expander("Prediction intervals (yhat_lower / yhat_upper) — per target", expanded=False):
            for target, idf in intervals_by_target.items():
                st.markdown(f"**{target}**")
                st.dataframe(idf, width='stretch')
    else:
        st.caption("Prediction intervals (yhat_lower/yhat_upper) were not returned by backend or model.")

    st.divider()

    st.markdown("### Plots")
    if "dt" not in wide_df.columns:
        st.info("Could not find time column in combined table — cannot build plots.")
        return

    plot_base = wide_df.copy()
    plot_base["dt"] = pd.to_datetime(plot_base["dt"], errors="coerce")
    plot_base = plot_base.dropna(subset=["dt"]).sort_values("dt")

    target_cols = [c for c in plot_base.columns if c != "dt"]
    if not target_cols:
        st.info("No target columns available for plotting.")
        return

    for tcol in target_cols:
        st.subheader(tcol)
        try:
            st.line_chart(plot_base.set_index("dt")[tcol])
        except Exception:
            st.dataframe(plot_base[["dt", tcol]],  width='stretch')