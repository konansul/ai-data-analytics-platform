from __future__ import annotations

import io
import re
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from frontend.helpers.components import sheets_summary_table
from frontend.helpers.data_access import cached_upload, get_preview


def _is_excel_file(file_name: str) -> bool:
    f = (file_name or "").lower()
    return f.endswith(".xlsx") or f.endswith(".xls")


def _safe_sheet_name(name: str, fallback: str = "Sheet") -> str:
    n = (name or "").strip() or fallback
    n = re.sub(r"[:\\/?*\[\]]", "_", n)
    n = n[:31].strip()
    return n or fallback


def render_tab_ingestion() -> None:
    st.subheader("Upload files (.xlsx, .xls, .csv)")

    token = st.session_state.get("auth_token")
    if not token:
        st.info("Login first in tab 0) Auth")
        return

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        key="uploader_tab1",
    )

    if not uploaded_files:
        st.info("Upload one or more files to start")
        return

    if "files_registry" not in st.session_state:
        st.session_state.files_registry = {}

    for up in uploaded_files:
        if up.name not in st.session_state.files_registry:
            st.session_state.files_registry[up.name] = {
                "bytes": up.getvalue(),
                "datasets": None,
            }

    for fname, meta in list(st.session_state.files_registry.items()):
        if meta.get("datasets") is not None:
            continue

        try:
            datasets_meta_i = cached_upload(meta["bytes"], fname, token_cache_key=token)
            st.session_state.files_registry[fname]["datasets"] = datasets_meta_i
        except Exception as e:
            st.error(f"Failed to upload/ingest {fname} via API: {e}")
            return

    file_names = list(st.session_state.files_registry.keys())
    if not file_names:
        st.error("files_registry is empty after upload. Please try again.")
        return

    selected_file = st.selectbox(
        "Select file to work with",
        file_names,
        key="selected_file_tab1",
    )

    datasets_meta = st.session_state.files_registry[selected_file].get("datasets") or []
    if not isinstance(datasets_meta, list) or not datasets_meta:
        st.error("No sheets/datasets returned from API for selected file.")
        return

    sheet_names = [d.get("sheet_name") for d in datasets_meta if isinstance(d, dict)]
    sheet_names = [s for s in sheet_names if s]
    if not sheet_names:
        st.error("No sheet names found in API response.")
        return

    selected_sheet_name = st.selectbox("Select sheet", sheet_names, key="selected_sheet_tab1")

    try:
        sheet_meta = next(d for d in datasets_meta if d.get("sheet_name") == selected_sheet_name)
    except StopIteration:
        st.error("Selected sheet not found in datasets meta.")
        return

    dataset_id = sheet_meta.get("dataset_id")
    if not dataset_id:
        st.error("Missing dataset_id for selected sheet.")
        return

    st.session_state["active_file_name"] = selected_file
    st.session_state["active_datasets_meta"] = datasets_meta
    st.session_state["active_sheet_meta"] = sheet_meta
    st.session_state["active_dataset_id"] = dataset_id
    st.session_state["datasets_meta"] = datasets_meta

    st.divider()

    st.subheader("Sheets summary")
    st.dataframe(sheets_summary_table(datasets_meta), width="stretch", hide_index=True)

    st.subheader(f"Raw preview — {sheet_meta.get('sheet_name')}")
    try:
        preview = get_preview(dataset_id, rows=30)
        st.dataframe(pd.DataFrame(preview.get("rows") or []), width="stretch", hide_index=True)
    except Exception as e:
        st.error(f"Failed to fetch preview: {e}")
        return

    st.subheader("Raw dtypes")
    dtypes = sheet_meta.get("dtypes", {}) or {}
    if isinstance(dtypes, dict) and dtypes:
        st.dataframe(
            pd.DataFrame([{"column": k, "dtype": v} for k, v in dtypes.items()]),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No dtype info returned for this sheet.")