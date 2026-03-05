from __future__ import annotations

import streamlit as st

from frontend.helpers.data_access import api_generate_pdf_report, api_download_report_pdf


def _get_dataset_filename(dataset_id: str) -> str:
    runs_store = st.session_state.get("runs_store")
    if isinstance(runs_store, dict):
        rec = runs_store.get(dataset_id) or {}
        fn = rec.get("file_name")
        if isinstance(fn, str) and fn.strip():
            return fn.strip()

def render_tab_generate_pdf_report():
    st.header("🧾 Generate PDF Report")

    dataset_id = st.session_state.get("active_dataset_id")
    if not dataset_id:
        st.info("Please upload and select a dataset first.")
        return

    run_id = st.session_state.get("last_run_id")
    if not run_id:
        st.warning("No cleaning run found. Please run Data Cleaning first (Tab 2).")
        return

    filename = _get_dataset_filename(dataset_id)

    st.caption(
        f"Dataset: `{dataset_id}` • Run: `{run_id}`\n\n"
    )

    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input("Report title", value=f"AI Data Analysis Report for {filename}")

    with col2:
        st.write("")
        st.write("")
        generate_btn = st.button(" Generate PDF Report", type="primary", use_container_width=True)

    st.divider()

    state_key = f"reporting::last_pdf::{dataset_id}::{run_id}"

    if generate_btn:
        with st.spinner("Generating report…"):
            try:
                resp = api_generate_pdf_report(
                    dataset_id=dataset_id,
                    run_id=run_id,
                    title=title or None
                )
                st.session_state[state_key] = resp
                st.success("✅ Report generated and saved.")
            except Exception as e:
                st.error(f"Report generation failed: {e}")
                return

    info = st.session_state.get(state_key)

    if not info:
        st.info("Click Generate PDF Report to create your first report.")
        return

    storage_key = info.get("storage_key")
    report_run_id = info.get("report_run_id")

    if not storage_key:
        st.warning("No storage_key returned from backend.")
        return


    if st.button("⬇️ Download PDF", use_container_width=True):
        with st.spinner("Downloading PDF…"):
            try:
                pdf_bytes = api_download_report_pdf(storage_key=storage_key)
                filename = f"report_{report_run_id or 'latest'}.pdf"
                st.download_button(
                    label="✅ Click to save PDF",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Download failed: {e}")