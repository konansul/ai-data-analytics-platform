# frontend/ui/tab_05_forecasting.py
from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple

import io
import base64
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import copy

from frontend.helpers.data_access import api_forecast_signals, api_forecast_plan, api_forecast_run, dataset_download_bytes, api_save_forecast_plot

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

    if not dataset_id:
        st.info("Please upload/select a dataset first.")
        return

    if "last_run_id" not in st.session_state:
        st.warning("No cleaning run found. Please run Data Cleaning first (Tab 2).")
        return

    run_id = st.session_state["last_run_id"]

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

    c_run, c_reset = st.columns([1, 1])
    with c_run:
        run_btn = st.button("Generate forecast", type="primary")
    with c_reset:
        reset_btn = st.button("Reset forecast state")

    if reset_btn:
        for k in ["forecast_signals", "forecast_plan", "forecast_result", "last_forecast_run_id"]:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Forecast state cleared.")
        st.stop()

    if run_btn:
        for k in ["forecast_signals", "forecast_plan", "forecast_result"]:
            if k in st.session_state:
                del st.session_state[k]

        profile = _get_post_profile_from_runs_store(dataset_id)

        with st.spinner("Running forecasting pipeline (signals → plan → run)..."):
            try:
                signals = api_forecast_signals(dataset_id=dataset_id, version=version)
                st.session_state["forecast_signals"] = signals

                feasible = bool(signals.get("feasible"))
                if not feasible:
                    st.warning(
                        "This dataset is not suitable for forecasting "
                        "(no valid datetime column or no numeric target columns)."
                    )
                    st.stop()

                plan = api_forecast_plan(
                    dataset_id=dataset_id,
                    version=version,
                    signals=signals,
                    profile=profile,
                    user_intent=(user_intent.strip() or None),
                    max_targets=int(max_targets),
                    head_rows=10,
                    horizon=int(horizon),
                )
                st.session_state["forecast_plan"] = plan

                if plan.get("mode") in {"skipped", "skip", "forecasting_skipped"} or not plan.get("suitable", True):
                    st.warning("Planner decided to skip forecasting for this dataset.")
                    st.stop()

                plan_for_run = copy.deepcopy(plan)

                targets = plan_for_run.get("targets") or []
                if isinstance(targets, list):
                    plan_for_run["targets"] = targets[: int(max_targets)]
                    for t in plan_for_run["targets"]:
                        if isinstance(t, dict):
                            t["horizon"] = int(horizon)

                result = api_forecast_run(
                    dataset_id=dataset_id,
                    version=version,
                    run_id=run_id,
                    plan=plan_for_run,
                    horizon=int(horizon),
                    model=model,
                    preview_rows=int(horizon),
                )
                st.session_state["forecast_result"] = result

                frun_id = result.get("forecast_run_id") or ""
                if frun_id:
                    st.session_state["last_forecast_run_id"] = frun_id

                st.success("Forecast completed.")

            except Exception as e:
                st.error(f"Forecast failed: {e}")
                st.stop()

    result = st.session_state.get("forecast_result")
    if not result:
        st.info("Click **Generate forecast** to run everything automatically.")
        return

    results_list = result.get("results") or []
    if not isinstance(results_list, list) or len(results_list) == 0:
        st.info("Forecast returned empty results.")
        return

    signals = st.session_state.get("forecast_signals") or {}
    plan = st.session_state.get("forecast_plan") or {}
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Feasible", _pretty_bool(signals.get("feasible")))
    with c2:
        st.metric("Mode", str(plan.get("mode", "unknown")))
    with c3:
        st.metric("Datetime", str(plan.get("datetime_column", "—")))
    with c4:
        st.metric("Targets", str(len(plan.get("targets") or [])))

    wide_df, intervals_by_target = _build_wide_tables(results_list)

    st.markdown("### Forecast preview (yhat)")
    if wide_df.empty:
        st.info("No preview data available.")
        return

    st.dataframe(wide_df, width="stretch")

    if intervals_by_target:
        with st.expander("Prediction intervals (optional)", expanded=False):
            for target, idf in intervals_by_target.items():
                st.markdown(f"**{target}**")
                st.dataframe(idf, width="stretch")

    st.divider()
    st.markdown("### Plots")

    try:
        hist_csv_bytes = dataset_download_bytes(
            dataset_id=dataset_id,
            version=version,
            fmt="csv",
        )
        hist_df = pd.read_csv(io.BytesIO(hist_csv_bytes))
    except Exception as e:
        st.error(f"Could not load historical dataset for plotting: {e}")
        return

    dt_col_hist = plan.get("datetime_column") or "dt"
    hist_dt_col = _pick_dt_col(hist_df) or dt_col_hist
    if hist_dt_col not in hist_df.columns:
        st.error(f"Could not find datetime column in historical dataset. Tried: {hist_dt_col}")
        return

    hist_df = _to_datetime_sorted(hist_df, hist_dt_col).rename(columns={hist_dt_col: "dt"})
    if hist_df.empty:
        st.info("Historical dataset is empty after datetime parsing — cannot plot.")
        return

    train_end_dt = hist_df["dt"].max()

    plot_forecast = wide_df.copy()
    plot_forecast["dt"] = pd.to_datetime(plot_forecast["dt"], errors="coerce")
    plot_forecast = plot_forecast.dropna(subset=["dt"]).sort_values("dt")

    target_cols = [c for c in plot_forecast.columns if c != "dt"]
    if not target_cols:
        st.info("No target columns to plot.")
        return

    st.caption(f"Train ends at: {train_end_dt.date()}")

    frun_id = st.session_state.get("last_forecast_run_id")
    if "saved_forecast_plots" not in st.session_state:
        st.session_state["saved_forecast_plots"] = set()

    for tcol in target_cols:
        st.subheader(tcol)

        train_part = None
        if tcol in hist_df.columns:
            train_part = hist_df[["dt", tcol]].rename(columns={tcol: "train"}).copy()
            train_part = train_part[train_part["dt"] <= train_end_dt].dropna(subset=["dt", "train"])
        else:
            st.info(f"'{tcol}' not found in historical dataset — plotting forecast only.")

        fc_part = plot_forecast[["dt", tcol]].rename(columns={tcol: "forecast"}).copy()
        fc_part = fc_part[fc_part["dt"] > train_end_dt].dropna(subset=["dt", "forecast"])

        if (train_part is None or train_part.empty) and fc_part.empty:
            st.info("No data to plot for this target.")
            continue

        if train_part is None:
            merged = fc_part.rename(columns={"forecast": "value"}).set_index("dt")
            st.line_chart(merged)
        else:
            merged = pd.merge(train_part, fc_part, on="dt", how="outer").sort_values("dt").set_index("dt")
            merged = merged[["train", "forecast"]]
            st.line_chart(merged)

        if not frun_id:
            continue

        plot_key = f"{frun_id}::{tcol}"
        if plot_key in st.session_state["saved_forecast_plots"]:
            continue

        try:
            fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)

            if train_part is not None and not train_part.empty:
                ax.plot(train_part["dt"], train_part["train"], label="train", linewidth=2)

            if not fc_part.empty:
                ax.plot(fc_part["dt"], fc_part["forecast"], label="forecast", linewidth=2)

            ax.axvline(train_end_dt, linestyle="--", linewidth=1)
            ax.set_title(tcol)
            ax.set_xlabel("Date")
            ax.set_ylabel(tcol)
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
            plt.close(fig)

            api_save_forecast_plot(
                forecast_run_id=frun_id,
                dataset_id=dataset_id,
                target=tcol,
                png_bytes=buf.getvalue(),
            )

            st.session_state["saved_forecast_plots"].add(plot_key)

        except Exception as e:
            st.warning(f"Plot save failed for {tcol}: {e}")