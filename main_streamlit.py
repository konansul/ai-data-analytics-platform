# main_streamlit.py

import streamlit as st

from frontend.ui.tab_00_authentication import render_tab_auth
from frontend.ui.tab_01_excel_upload import render_tab_ingestion
from frontend.ui.tab_02_cleaning import render_tab_cleaning
from frontend.ui.tab_03_signals import render_tab_signals
from frontend.ui.tab_04_visualization import render_tab_visualization
from frontend.ui.tab_05_forecasting import render_tab_forecasting
from frontend.ui.tab_06_generate_pdf_report import render_tab_generate_pdf_report
from frontend.ui.tab_07_save_all_files import render_tab_saved_datasets


st.set_page_config(page_title="4CAST", layout="wide")
st.title("4CAST — AI Data Cleaning, Visualization, and Forecasting Platform")
st.caption(
    "4CAST is an end-to-end data analytics platform for transforming raw Excel and CSV files into clean, analysis-ready datasets with automated visualization and time-series forecasting. The system provides user-scoped ingestion of multi-sheet spreadsheets, dataset profiling, rule-based and LLM-assisted cleaning pipelines, persistent execution history, and reproducible exports. On top of cleaned data, 4CAST generates intelligent visualization suggestions and executes forecasting workflows using a signal, planning, execution architecture. The backend is built with FastAPI and PostgreSQL for durable storage of datasets and runs, while the Streamlit frontend delivers an interactive workflow for ingestion, cleaning, exploration, visualization, and forecasting — all fully decoupled through REST APIs."
)

if "auth_token" not in st.session_state:
    st.session_state.auth_token = None

if "runs_store" not in st.session_state:
    st.session_state.runs_store = {}

if "last_run_id" not in st.session_state:
    st.session_state.last_run_id = None


def is_authed() -> bool:
    return bool(st.session_state.get("auth_token"))


def get_active_dataset():
    dataset_id = st.session_state.get("active_dataset_id")
    sheet_meta = st.session_state.get("active_sheet_meta")
    selected_file = st.session_state.get("active_file_name")
    if not dataset_id or not sheet_meta or not selected_file:
        return None, None, None
    return selected_file, sheet_meta, dataset_id


tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "0) Authentication",
    "1) Upload Files",
    "2) Data Cleaning",
    "3) Signal Generation",
    "4) Visualization",
    "5) Forecasting",
    "6) Generate PDF Report",
    "7) Download All Cleaned Files",
])

with tab0:
    render_tab_auth()

with tab1:
    if not is_authed():
        st.info("Please login first.")
    else:
        render_tab_ingestion()

selected_file, sheet_meta, dataset_id = get_active_dataset()

with tab2:
    if not is_authed():
        st.info("Please login first.")
    elif not dataset_id:
        st.info("Please upload the dataset first.")
    else:
        run_id, report = render_tab_cleaning(selected_file, sheet_meta, dataset_id=dataset_id)
        if run_id and report:
            st.session_state.runs_store[dataset_id] = {
                "file_name": selected_file,
                "sheet_name": sheet_meta.get("sheet_name"),
                "run_id": run_id,
                "report": report,
            }
            st.session_state.last_run_id = run_id

with tab3:
    if not is_authed():
        st.info("Please login first.")
    elif not dataset_id:
        st.info("Please upload the dataset first.")
    else:
        item = st.session_state.runs_store.get(dataset_id)
        if not item:
            st.info("Please clean the dataset first in third tab.")
        else:
            render_tab_signals(item["report"])

with tab4:
    if not is_authed():
        st.info("Please login first.")
    elif not dataset_id:
        st.info("Please upload the dataset first.")
    else:
        item = st.session_state.runs_store.get(dataset_id)
        if not item:
            st.info("Please clean the dataset first in third tab.")
        else:
            render_tab_visualization()

with tab5:
    if not is_authed():
        st.info("Please login first.")
    elif not dataset_id:
        st.info("Please upload the dataset first.")
    else:
        item = st.session_state.runs_store.get(dataset_id)
        if not item:
            st.info("Please clean the dataset first in third tab.")
        else:
            render_tab_forecasting(dataset_id=dataset_id)

with tab6:
    if not is_authed():
        st.info("Please login first.")
    elif not dataset_id:
        st.info("Please upload and clean a dataset first.")
    else:
        item = st.session_state.runs_store.get(dataset_id)
        if item and not st.session_state.get("last_run_id"):
            st.session_state.last_run_id = item.get("run_id")

        render_tab_generate_pdf_report()

with tab7:
    if not is_authed():
        st.info("Please login first.")
    else:
        render_tab_saved_datasets()