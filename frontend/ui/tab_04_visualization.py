# frontend/ui/tab_04_visualization.py

from __future__ import annotations

from io import BytesIO
from typing import Dict, Any, Optional, List

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.helpers.data_access import download_dataset, get_visualization_pairings, get_visualization_plots, explain_chart, api_save_viz_plot


def _render_pairing_section(*, dataset_id: str, profile: Dict[str, Any]) -> None:
    st.subheader("📊 Visualization Planning — Stage 1")
    st.caption(
        "The Pairing Agent analyzes your cleaned dataset signals and proposes the most "
        "informative column combinations to visualize. Select the ones you want and proceed "
        "to Stage 2 to generate plots."
    )

    pairing_key = f"viz_pairings_{dataset_id}"

    if st.button("Generate Column Pairings", type="primary", key=f"btn_pairings_{dataset_id}"):
        with st.spinner("Running Pairing Agent (Stage 1)…"):
            try:
                result = get_visualization_pairings(dataset_id, profile)
                st.session_state[pairing_key] = result.get("pairings", [])
                st.session_state.pop(f"viz_selected_{dataset_id}", None)
                # reset plots cache too
                for k in list(st.session_state.keys()):
                    if k.startswith(f"viz_plot_{dataset_id}_"):
                        st.session_state.pop(k, None)
            except Exception as e:
                st.error(f"Pairing Agent failed: {e}")

    pairings = st.session_state.get(pairing_key)
    if not pairings:
        st.info("No pairings yet — click **Generate Column Pairings**.")
        return

    st.success(f"Found **{len(pairings)} column pairings**. Select which ones you want to visualize:")

    pairing_rows: List[Dict[str, Any]] = []
    for p in sorted(pairings, key=lambda x: x.get("rank") or 999):
        pairing_rows.append(
            {
                "Rank": p.get("rank", "—"),
                "Score": f"{p['score']:.2f}" if p.get("score") is not None else "—",
                "Columns": " + ".join(p.get("columns", [])),
                "Template": p.get("template", "—"),
                "Rationale": p.get("rationale", ""),
            }
        )

    df_pairings = pd.DataFrame(pairing_rows)

    selected_key = f"viz_selected_{dataset_id}"
    if selected_key not in st.session_state:
        st.session_state[selected_key] = list(range(len(pairings)))  # all selected by default

    st.write("**Tick the pairings you want to visualize:**")
    newly_selected: List[int] = []

    for i, row in df_pairings.iterrows():
        checked = i in st.session_state[selected_key]
        label = f"**{row['Columns']}** — {row['Template']}  ·  score {row['Score']}  ·  {row['Rationale']}"
        if st.checkbox(label, value=checked, key=f"pair_chk_{dataset_id}_{i}"):
            newly_selected.append(int(i))

    st.session_state[selected_key] = newly_selected

    n_selected = len(newly_selected)
    if n_selected == 0:
        st.warning("No pairings selected — select at least one before Stage 2.")
    else:
        st.info(
            f"✅ **{n_selected} pairing(s) selected.** "
            "Go to **Stage 2 (Plots)** tab and click **Generate** on any pairing."
        )


def _render_plot(
    plot_cfg: dict,
    df: pd.DataFrame,
    slot_key: str,
    *,
    dataset_id: str,
    run_id: str,
) -> None:
    title = plot_cfg.get("title", "Plot")
    description = plot_cfg.get("description", "")
    ptype = plot_cfg.get("plot_type")
    alt_ptype = plot_cfg.get("alt_plot_type")
    x = plot_cfg.get("x_column")
    y = plot_cfg.get("y_column")
    color = plot_cfg.get("color_column")
    constraints = plot_cfg.get("constraints") or {}
    warnings = plot_cfg.get("warnings") or []
    source = plot_cfg.get("source_pairing")

    h1, h2 = st.columns([6, 2])
    with h1:
        st.subheader(title)
        if description:
            st.caption(description)
    with h2:
        if source:
            st.caption(f"🔗 `{' + '.join(source)}`")
        if alt_ptype:
            st.caption(f"Alt: `{alt_ptype}`")

    for w in warnings:
        st.warning(f"⚠️ {w}")

    try:
        plot_df = df.copy()

        if constraints.get("top_k") and x:
            top_k = int(constraints["top_k"])
            top_cats = (
                plot_df.groupby(x)[y].sum().nlargest(top_k).index
                if y else
                plot_df[x].value_counts().nlargest(top_k).index
            )
            plot_df = plot_df[plot_df[x].isin(top_cats)]

        fig = None
        if ptype == "bar":
            fig = px.bar(plot_df, x=x, y=y, color=color, title=title)
        elif ptype == "line":
            fig = px.line(plot_df, x=x, y=y, color=color, title=title)
        elif ptype == "scatter":
            if constraints.get("trend"):
                fig = px.scatter(
                    plot_df,
                    x=x,
                    y=y,
                    color=color,
                    title=title,
                    trendline="ols",
                    trendline_color_override="#FF4B4B",
                )
                fig.update_traces(selector=dict(mode="lines"), line=dict(width=4, dash="solid"))
                fig.update_traces(selector=dict(mode="markers"), marker=dict(opacity=0.55, size=7))
            else:
                fig = px.scatter(plot_df, x=x, y=y, color=color, title=title)
        elif ptype == "histogram":
            bins = constraints.get("bins", 20)
            fig = px.histogram(plot_df, x=x, nbins=bins, color=color, title=title)
        elif ptype == "box":
            fig = px.box(plot_df, x=x, y=y, color=color, title=title)
        elif ptype == "heatmap":
            if not x and not y:
                corr = plot_df.select_dtypes(include="number").corr()
                fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
            else:
                fig = px.density_heatmap(plot_df, x=x, y=y, title=title)
        elif ptype == "pie":
            fig = px.pie(plot_df, names=x, values=y, title=title)

        if not fig:
            st.warning(f"Could not render plot type: `{ptype}`")
            return

        st.plotly_chart(fig, use_container_width=True, key=f"chart_{slot_key}")

        # auto-save once per plot slot
        save_key = f"saved_viz_plot::{slot_key}"
        if save_key not in st.session_state:
            st.session_state[save_key] = False

        if not st.session_state[save_key]:
            try:
                png_bytes = fig.to_image(format="png", scale=2)
                api_save_viz_plot(
                    dataset_id=dataset_id,
                    run_id=run_id,
                    title=title,
                    png_bytes=png_bytes,
                )
                st.session_state[save_key] = True
                st.caption("✅ Saved to storage")
            except Exception as e:
                st.warning(f"Plot save failed: {e}")

        btn_col, txt_col = st.columns([1, 4])
        with btn_col:
            if st.button("✨ Explain Insight", key=f"explain_btn_{slot_key}"):
                with st.spinner("Analyzing…"):
                    insight = explain_chart(title, x or "Index", y or "Value")
                    st.session_state[f"insight_{slot_key}"] = insight

        if st.session_state.get(f"insight_{slot_key}"):
            with txt_col:
                st.info(st.session_state[f"insight_{slot_key}"])

    except Exception as e:
        st.error(f"Could not render plot: {e}")


def _render_tab_plots(*, dataset_id: str, run_id: str, profile_data: Dict[str, Any]) -> None:
    st.subheader("📈 Visualization — Stage 2")
    st.caption("Generate plots for the pairings you selected in Stage 1.")

    pairing_key = f"viz_pairings_{dataset_id}"
    selected_key = f"viz_selected_{dataset_id}"

    all_pairings = st.session_state.get(pairing_key)
    selected_idx = st.session_state.get(selected_key)

    if not all_pairings:
        st.info("No pairings yet — go to **Stage 1** and generate/select pairings first.")
        return

    selected_pairings = [all_pairings[i] for i in (selected_idx or []) if i < len(all_pairings)]
    if not selected_pairings:
        st.warning("No pairings selected — go back to **Stage 1** and tick at least one.")
        return

    df_key = f"viz_df_{dataset_id}"
    if df_key not in st.session_state:
        try:
            with st.spinner("Loading dataset…"):
                data_bytes = download_dataset(dataset_id, version="current", fmt="xlsx")
                st.session_state[df_key] = pd.read_excel(BytesIO(data_bytes))
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            return

    df = st.session_state[df_key]

    st.markdown(
        f"**{len(selected_pairings)} pairing(s)** ready. "
        "Click **Generate** on each one to produce its visualization."
    )
    st.markdown("---")

    for i, pairing in enumerate(selected_pairings):
        cols_label = " + ".join(pairing.get("columns", []))
        template = pairing.get("template", "")
        score = pairing.get("score")
        score_str = f"{score:.2f}" if score is not None else "—"
        plot_key = f"viz_plot_{dataset_id}_{i}"

        with st.container(border=True):
            top_left, top_right = st.columns([5, 2])
            with top_left:
                st.markdown(f"#### {i + 1}. `{cols_label}`")
                st.caption(f"Template: **{template}** · Score: **{score_str}** · {pairing.get('rationale', '')}")

            with top_right:
                if st.button(
                    "Generate ▶",
                    key=f"gen_btn_{dataset_id}_{i}",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner(f"Generating plot for {cols_label}…"):
                        try:
                            plots = get_visualization_plots(dataset_id, profile_data, [pairing])
                            if plots:
                                st.session_state[plot_key] = plots[0]
                        except Exception as e:
                            st.error(f"Plot generation failed: {e}")

            plot_cfg = st.session_state.get(plot_key)
            if plot_cfg:
                st.divider()
                _render_plot(
                    plot_cfg,
                    df,
                    slot_key=f"{dataset_id}_{i}",
                    dataset_id=dataset_id,
                    run_id=run_id,
                )


def render_tab_visualization() -> None:

    dataset_id = st.session_state.get("active_dataset_id")
    if not dataset_id:
        st.info("Please upload and select a dataset first.")
        return

    run_store = st.session_state.get("runs_store", {}).get(dataset_id)
    if not run_store or "report" not in run_store:
        st.warning("No cleaning report found. Please run the Data Cleaning pipeline (Tab 2) first.")
        return

    run_id = st.session_state.get("last_run_id")
    if not run_id:
        st.warning("No cleaning run found. Please run Data Cleaning first (Tab 2).")
        return

    profile_data = run_store["report"].get("post_profile")
    if not isinstance(profile_data, dict) or not profile_data:
        st.error("The cleaning report does not contain a valid post_profile.")
        return

    tab_stage1, tab_stage2 = st.tabs(["Stage 1 — Pairings", "Stage 2 — Plots"])

    with tab_stage1:
        _render_pairing_section(dataset_id=dataset_id, profile=profile_data)

    with tab_stage2:
        _render_tab_plots(dataset_id=dataset_id, run_id=run_id, profile_data=profile_data)