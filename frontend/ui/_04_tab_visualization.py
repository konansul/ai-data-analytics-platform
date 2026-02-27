# frontend/ui/_04_tab_visualization.py

import base64
from io import BytesIO

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from ui.data_access import API_BASE, _auth_headers, _raise, download_dataset


# -----------------------------
# API HELPERS
# -----------------------------

def save_viz_plot_api(
    viz_run_id: str,
    dataset_id: str,
    plot_index: int,
    title: str,
    plot_type: str,
    fig,
) -> dict:

    if "last_run_id" not in st.session_state:
        raise RuntimeError("Cleaning run_id missing in session_state")

    png_bytes = fig.to_image(format="png", scale=2)

    payload = {
        "run_id": st.session_state["last_run_id"],  # cleaning run id
        "viz_run_id": viz_run_id,
        "dataset_id": dataset_id,
        "title": title,
        "plot_type": plot_type,
        "png_base64": base64.b64encode(png_bytes).decode("utf-8"),
        "meta": {
            "plot_index": plot_index
        },
    }

    resp = requests.post(
        f"{API_BASE}/visualization/plots",
        json=payload,
        headers=_auth_headers(),
        timeout=60,
    )

    _raise(resp)
    return resp.json()


def suggest_visualizations_api(dataset_id: str, profile_data: dict) -> dict:

    if "last_run_id" not in st.session_state:
        raise RuntimeError("Cleaning run_id missing in session_state")

    resp = requests.post(
        f"{API_BASE}/visualization/suggest",
        json={
            "dataset_id": dataset_id,
            "run_id": st.session_state["last_run_id"],  # 🔥 ОБЯЗАТЕЛЬНО
            "profile_data": profile_data,
        },
        headers=_auth_headers(),
        timeout=120,
    )

    _raise(resp)
    return resp.json()


def explain_chart_api(title: str, x_col: str, y_col: str) -> str:
    axis_info = f"X-Axis: {x_col}, Y-Axis: {y_col}"

    resp = requests.post(
        f"{API_BASE}/visualization/explain",
        json={"plot_title": title, "axis_info": axis_info},
        headers=_auth_headers(),
        timeout=30,
    )

    if resp.status_code == 200:
        return resp.json().get("explanation", "No explanation available.")
    return f"Error: {resp.text}"


# -----------------------------
# STATE
# -----------------------------

def _ensure_state():
    if "viz_plan" not in st.session_state:
        st.session_state.viz_plan = {}
    if "viz_run_id" not in st.session_state:
        st.session_state.viz_run_id = {}
    if "saved_viz_plots" not in st.session_state:
        st.session_state.saved_viz_plots = set()
    if "insights" not in st.session_state:
        st.session_state.insights = {}


# -----------------------------
# MAIN TAB
# -----------------------------

def render_tab_visualization():
    _ensure_state()

    st.header("AI-Driven Visualization Agent")

    dataset_id = st.session_state.get("active_dataset_id")
    if not dataset_id:
        st.info("Please upload and select a dataset first.")
        return

    if "last_run_id" not in st.session_state:
        st.warning("No cleaning run found. Please run Data Cleaning first.")
        return

    run_store = st.session_state.get("runs_store", {}).get(dataset_id)
    if not run_store or "report" not in run_store:
        st.warning("No cleaning report found. Please run the Data Cleaning pipeline (Tab 2) first.")
        return

    cleaning_report = run_store["report"]
    profile_data = cleaning_report.get("post_profile")
    if not profile_data:
        st.error("The cleaning report does not contain a profile.")
        return

    st.markdown("This agent analyzes dataset signals to suggest optimal plots.")

    # -----------------------------
    # GENERATE PLAN
    # -----------------------------

    if st.button("Generate Plot Plan", type="primary"):
        with st.spinner("Consulting Visualization Agent..."):
            try:
                payload = suggest_visualizations_api(dataset_id, profile_data)
                plan = payload.get("plan") or payload

                st.session_state.viz_plan[dataset_id] = plan

                if payload.get("viz_run_id"):
                    st.session_state.viz_run_id[dataset_id] = payload["viz_run_id"]

            except Exception as e:
                st.error(f"Agent failed: {e}")
                return

    plan = st.session_state.viz_plan.get(dataset_id)
    viz_run_id = st.session_state.viz_run_id.get(dataset_id)

    if not plan:
        st.info("Generate a plot plan to see visualizations.")
        return

    plots = plan.get("plots", []) if isinstance(plan, dict) else []
    st.success(f"Agent generated {len(plots)} visualizations.")

    # -----------------------------
    # LOAD DATASET
    # -----------------------------

    try:
        with st.spinner("Fetching dataset for rendering..."):
            data_bytes = download_dataset(dataset_id, version="current", fmt="xlsx")
            df = pd.read_excel(BytesIO(data_bytes))
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return

    # -----------------------------
    # RENDER PLOTS
    # -----------------------------

    for i, plot_cfg in enumerate(plots):
        title = plot_cfg.get("title") or f"Plot {i + 1}"
        desc = plot_cfg.get("description") or ""
        ptype = plot_cfg.get("plot_type")
        x = plot_cfg.get("x_column")
        y = plot_cfg.get("y_column")
        color = plot_cfg.get("color_column")
        constraints = plot_cfg.get("constraints", {}) or {}

        st.subheader(f"{i + 1}. {title}")
        if desc:
            st.caption(desc)

        try:
            plot_df = df.copy()

            if constraints.get("top_k") and x:
                top_k = int(constraints["top_k"])
                if y:
                    top_cats = plot_df.groupby(x)[y].sum().nlargest(top_k).index
                else:
                    top_cats = plot_df[x].value_counts().nlargest(top_k).index
                plot_df = plot_df[plot_df[x].isin(top_cats)]

            fig = None

            if ptype == "bar":
                fig = px.bar(plot_df, x=x, y=y, color=color)
            elif ptype == "line":
                fig = px.line(plot_df, x=x, y=y, color=color)
            elif ptype == "scatter":
                fig = px.scatter(plot_df, x=x, y=y, color=color)
            elif ptype == "histogram":
                fig = px.histogram(plot_df, x=x, y=y, color=color)
            elif ptype == "box":
                fig = px.box(plot_df, x=x, y=y, color=color)
            elif ptype == "heatmap":
                if not x and not y:
                    corr = plot_df.select_dtypes(include="number").corr()
                    fig = px.imshow(corr, text_auto=True)
                else:
                    fig = px.density_heatmap(plot_df, x=x, y=y)
            elif ptype == "pie":
                fig = px.pie(plot_df, names=x, values=y)

            if fig is None:
                st.warning(f"Unsupported plot type: {ptype}")
                st.divider()
                continue

            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # -----------------------------
            # AUTO SAVE
            # -----------------------------

            if viz_run_id:
                save_key = f"{viz_run_id}::{i}"
                if save_key not in st.session_state.saved_viz_plots:
                    try:
                        save_viz_plot_api(
                            viz_run_id=viz_run_id,
                            dataset_id=dataset_id,
                            plot_index=i,
                            title=title,
                            plot_type=ptype or "unknown",
                            fig=fig,
                        )
                        st.session_state.saved_viz_plots.add(save_key)
                    except Exception as e:
                        st.warning(f"Auto-save failed: {e}")

            # -----------------------------
            # INSIGHTS
            # -----------------------------

            c1, c2 = st.columns([1, 4])

            with c1:
                if st.button("✨ Explain Insight", key=f"explain_{dataset_id}_{i}"):
                    with st.spinner("Analyzing..."):
                        insight = explain_chart_api(title, x or "Index", y or "Value")
                        st.session_state.insights[f"{dataset_id}::{i}"] = insight

            with c2:
                insight_key = f"{dataset_id}::{i}"
                if st.session_state.insights.get(insight_key):
                    st.info(st.session_state.insights[insight_key])

        except Exception as e:
            st.error(f"Could not render plot: {e}")

        st.divider()