# frontend/ui/_04_tab_visualization.py
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from io import BytesIO
from ui.data_access import API_BASE, _auth_headers, _raise, download_dataset

def suggest_visualizations_api(dataset_id: str, profile_data: dict) -> dict:
    """Wrapper to call the new backend endpoint."""
    resp = requests.post(
        f"{API_BASE}/visualization/suggest",
        json={"dataset_id": dataset_id, "profile_data": profile_data},
        headers=_auth_headers(),
        timeout=120,
    )
    _raise(resp)
    return resp.json()

# Helper Function for the new API
def explain_chart_api(title: str, x_col: str, y_col: str) -> str:
    """Calls the backend to get an explanation."""
    axis_info = f"X-Axis: {x_col}, Y-Axis: {y_col}"
    resp = requests.post(
        f"{API_BASE}/visualization/explain",
        json={"plot_title": title, "axis_info": axis_info},
        headers=_auth_headers(),
        timeout=30
    )
    if resp.status_code == 200:
        return resp.json().get("explanation", "No explanation available.")
    return f"Error: {resp.text}"

def render_tab_visualization():
    st.header("AI-Driven Visualization Agent")

    dataset_id = st.session_state.get("active_dataset_id")
    if not dataset_id:
        st.info("Please upload and select a dataset first.")
        return

    # 1. Retrieve Signals (Profile) from the cleaning run
    run_store = st.session_state.get("runs_store", {}).get(dataset_id)
    if not run_store or "report" not in run_store:
        st.warning("No cleaning report found. Please run the Data Cleaning pipeline (Tab 2) first.")
        return

    cleaning_report = run_store["report"]
    profile_data = cleaning_report.get("post_profile")

    if not profile_data:
        st.error("The cleaning report does not contain a profile. Ensure profiling is enabled in the pipeline.")
        return

    # 2. Generate Plan
    st.markdown("This agent analyzes the **dataset signals** (columns, correlations, types) to suggest optimal plots.")

    if "viz_plan" not in st.session_state:
        st.session_state.viz_plan = {}

    col_btn, col_status = st.columns([1, 4])
    with col_btn:
        if st.button("Generate Plot Plan", type="primary"):
            with st.spinner("Consulting Visualization Agent..."):
                try:
                    plan = suggest_visualizations_api(dataset_id, profile_data)
                    st.session_state.viz_plan[dataset_id] = plan
                except Exception as e:
                    st.error(f"Agent failed: {e}")

    # 3. Render Plots
    plan = st.session_state.viz_plan.get(dataset_id)
    if plan:
        st.success(f"Agent generated {len(plan.get('plots', []))} visualizations.")

        # Download actual data for plotting
        try:
            with st.spinner("Fetching dataset for rendering..."):
                # We use the 'current' version which is the clean parquet/xlsx
                data_bytes = download_dataset(dataset_id, version="current", fmt="xlsx")
                df = pd.read_excel(BytesIO(data_bytes))
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            return

        for i, plot_cfg in enumerate(plan.get("plots", [])):
            with st.container():
                st.subheader(f"{i+1}. {plot_cfg.get('title')}")
                st.caption(plot_cfg.get('description'))

                try:
                    ptype = plot_cfg.get("plot_type")
                    x = plot_cfg.get("x_column")
                    y = plot_cfg.get("y_column")
                    color = plot_cfg.get("color_column")
                    constraints = plot_cfg.get("constraints", {})

                    # Apply Top-K constraint for readable bars
                    plot_df = df.copy()
                    if constraints.get("top_k") and x:
                        top_k = int(constraints["top_k"])
                        # Keep top K categories by frequency or sum of Y
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
                        # Special case: Correlation heatmap
                        if not x and not y:
                            corr = plot_df.select_dtypes(include='number').corr()
                            fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
                        else:
                            fig = px.density_heatmap(plot_df, x=x, y=y)
                    elif ptype == "pie":
                        fig = px.pie(plot_df, names=x, values=y)

                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        # --- NEW CODE: Add "Analyze" Button ---
                        btn_col, text_col = st.columns([1, 4])
                        with btn_col:
                            if st.button(f"✨ Explain Insight", key=f"explain_{i}"):
                                with st.spinner("Analyzing..."):
                                    insight = explain_chart_api(
                                        plot_cfg.get("title"),
                                        plot_cfg.get("x_column", "Index"),
                                        plot_cfg.get("y_column", "Value")
                                    )
                                    # Store in session state to keep it visible
                                    st.session_state[f"insight_{i}"] = insight

                        # Display the insight if it exists
                        if st.session_state.get(f"insight_{i}"):
                            with text_col:
                                st.info(st.session_state[f"insight_{i}"])
                    else:
                        st.warning(f"Could not render plot type: {ptype}")

                except Exception as e:
                    st.error(f"Could not render plot: {e}")

                st.divider()