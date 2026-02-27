# frontend/ui/_04_tab_visualization.py
from __future__ import annotations

from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from ui.data_access import (
    download_dataset,
    get_visualization_plots,
    explain_chart,
)


def _render_plot(plot_cfg: dict, df: pd.DataFrame, slot_key: str):
    """Renders a single PlotConfig dict onto the page."""
    title       = plot_cfg.get("title", "Plot")
    description = plot_cfg.get("description", "")
    ptype       = plot_cfg.get("plot_type")
    alt_ptype   = plot_cfg.get("alt_plot_type")
    x           = plot_cfg.get("x_column")
    y           = plot_cfg.get("y_column")
    color       = plot_cfg.get("color_column")
    constraints = plot_cfg.get("constraints") or {}
    warnings    = plot_cfg.get("warnings") or []
    source      = plot_cfg.get("source_pairing")

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

        # Apply top-K for high-cardinality bar charts
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
                    plot_df, x=x, y=y, color=color, title=title,
                    trendline="ols",
                    trendline_color_override="#FF4B4B",  # bright red trendline
                )
                # Make trendline thick and prominent
                fig.update_traces(
                    selector=dict(mode="lines"),
                    line=dict(width=4, dash="solid"),
                )
                # Make scatter points semi-transparent so trendline pops
                fig.update_traces(
                    selector=dict(mode="markers"),
                    marker=dict(opacity=0.55, size=7),
                )
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

        if fig:
            # Unique key prevents Streamlit from hiding/caching stale charts
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{slot_key}")

            btn_col, txt_col = st.columns([1, 4])
            with btn_col:
                if st.button("✨ Explain Insight", key=f"explain_btn_{slot_key}"):
                    with st.spinner("Analyzing…"):
                        insight = explain_chart(title, x or "Index", y or "Value")
                        st.session_state[f"insight_{slot_key}"] = insight

            if st.session_state.get(f"insight_{slot_key}"):
                with txt_col:
                    st.info(st.session_state[f"insight_{slot_key}"])
        else:
            st.warning(f"Could not render plot type: `{ptype}`")

    except Exception as e:
        st.error(f"Could not render plot: {e}")


def render_tab_visualization():
    st.header("AI-Driven Visualization Agent")

    dataset_id = st.session_state.get("active_dataset_id")
    if not dataset_id:
        st.info("Please upload and select a dataset first.")
        return

    # ── Require cleaning report ───────────────────────────────────────────────
    run_store = st.session_state.get("runs_store", {}).get(dataset_id)
    if not run_store or "report" not in run_store:
        st.warning("No cleaning report found. Please run the Data Cleaning pipeline (Tab 2) first.")
        return

    profile_data = run_store["report"].get("post_profile")
    if not profile_data:
        st.error("The cleaning report does not contain a profile.")
        return

    # ── Check pairings were selected in the Signals tab ──────────────────────
    pairing_key  = f"viz_pairings_{dataset_id}"
    selected_key = f"viz_selected_{dataset_id}"

    all_pairings = st.session_state.get(pairing_key)
    selected_idx = st.session_state.get(selected_key)

    if not all_pairings:
        st.info(
            "No column pairings yet. Go to the **Signals** tab → **Generate Column Pairings** "
            "→ select combinations → come back here."
        )
        return

    selected_pairings = [
        all_pairings[i] for i in (selected_idx or []) if i < len(all_pairings)
    ]

    if not selected_pairings:
        st.warning("No pairings selected. Go back to the **Signals** tab and tick at least one.")
        return

    # ── Load dataset once (shared across all plots) ───────────────────────────
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

    # ── Per-pairing cards ─────────────────────────────────────────────────────
    st.markdown(
        f"**{len(selected_pairings)} pairing(s)** ready. "
        "Click **Generate** on each one to produce its visualization."
    )
    st.markdown("---")

    for i, pairing in enumerate(selected_pairings):
        cols_label = " + ".join(pairing.get("columns", []))
        template   = pairing.get("template", "")
        score      = pairing.get("score")
        score_str  = f"{score:.2f}" if score is not None else "—"
        plot_key   = f"viz_plot_{dataset_id}_{i}"

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
                            # Call Stage 2 with just this one pairing
                            plots = get_visualization_plots(
                                dataset_id,
                                profile_data,
                                [pairing],          # single pairing → single plot back
                            )
                            if plots:
                                st.session_state[plot_key] = plots[0]
                        except Exception as e:
                            st.error(f"Plot generation failed: {e}")

            # Render the plot if it has been generated
            plot_cfg = st.session_state.get(plot_key)
            if plot_cfg:
                st.divider()
                _render_plot(plot_cfg, df, slot_key=f"{dataset_id}_{i}")

        st.markdown("")   # breathing room between cards
