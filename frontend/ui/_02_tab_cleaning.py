# frontend/ui/_02_tab_cleaning.py
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from ui.data_access import (
    suggest_policy,
    run_cleaning,
    get_run_report,
    download_artifact,
    get_preview,
)


def _is_excel_file(file_name: str) -> bool:
    name = (file_name or "").lower()
    return name.endswith(".xlsx") or name.endswith(".xls")


def _safe_sheet_name(name: str, fallback: str = "Sheet") -> str:
    bad = set("[]:*?/\\")
    cleaned = "".join("_" if c in bad else c for c in (name or "").strip())
    cleaned = cleaned[:31] if cleaned else fallback
    return cleaned


def _render_execution_summary(report: Dict[str, Any]) -> None:
    st.subheader("Execution summary")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Rows (before)", report.get("rows_before"))
    with s2:
        st.metric("Cols (before)", report.get("cols_before"))
    with s3:
        st.metric("Rows (after)", report.get("rows_after"))
    with s4:
        st.metric("Cols (after)", report.get("cols_after"))


def _render_policy_block(pol: Dict[str, Any], *, dataset_id: str) -> None:
    plan = pol.get("policy", {}) or {}
    st.subheader("Cleaning Plan")
    st.metric("Source", str(pol.get("source", plan.get("source", "unknown"))))

    enabled_steps = plan.get("enabled_steps", {}) or {}
    params = plan.get("params", {}) or {}
    notes = pol.get("notes") or plan.get("notes") or []

    if enabled_steps:
        steps_df = pd.DataFrame([{"step": k, "enabled": bool(v)} for k, v in enabled_steps.items()]).sort_values("step")
        st.dataframe(steps_df, width="stretch", hide_index=True)

    if params:
        params_df = pd.DataFrame([{"param": str(k), "value": str(params[k])} for k in sorted(params.keys())])
        st.dataframe(params_df, width="stretch", hide_index=True)

    if notes:
        st.write("Notes")
        for n in notes:
            st.write(f"- {n}")

    st.divider()


def _render_after_preview_and_downloads(
    *,
    dataset_id: str,
    run_id: Optional[str],
    file_name: str,
    sheet_name: str,
) -> None:
    st.subheader("Cleaned preview")
    try:
        preview_after = get_preview(dataset_id, rows=30)
        st.dataframe(pd.DataFrame(preview_after.get("rows") or []), width="stretch", hide_index=True)
    except Exception as e:
        st.warning(f"Preview after cleaning failed: {e}")

    if not run_id:
        st.info("No run_id found for this dataset yet.")
        return


def _render_multi_file_cleaning_block(files_registry: Dict[str, Any]) -> None:

    uploaded_files_count = len(files_registry)

    st.subheader(f"Clean all uploaded files: {uploaded_files_count}")

    st.info(f"Run cleaning for all uploaded files.")

    cA, cB, cC = st.columns([1, 1, 2])
    with cA:
        use_llm_all_files = st.checkbox("Use LLM", value=False, key="use_llm_all_files")
    with cB:
        llm_model_all_files = st.selectbox(
            "LLM model",
            ["gemini-2.5-flash"],
            index=0,
            disabled=not use_llm_all_files,
            key="llm_model_all_files",
        )
    with cC:
        st.caption("")

    if st.button(
        "Clean all uploaded files",
        type="primary",
        width="stretch",
        key="btn_clean_all_uploaded_files",
    ):
        tasks: List[Dict[str, Any]] = []
        for fname, meta in files_registry.items():
            datasets_meta_i = meta.get("datasets") or []
            if not isinstance(datasets_meta_i, list) or not datasets_meta_i:
                tasks.append({"file_name": fname, "error": "No datasets meta (ingestion not completed)."})
                continue

            for j, ds_meta in enumerate(datasets_meta_i, start=1):
                ds_id = ds_meta.get("dataset_id")
                sh_name = ds_meta.get("sheet_name") or f"Sheet{j}"
                if not ds_id:
                    tasks.append({"file_name": fname, "sheet_name": sh_name, "error": "Missing dataset_id"})
                    continue
                tasks.append({"file_name": fname, "sheet_name": sh_name, "dataset_id": ds_id})

        real_tasks = [t for t in tasks if t.get("dataset_id")]
        total = len(real_tasks)

        if total == 0:
            st.error("No datasets found to clean. Make sure files were ingested in tab 1.")
            return

        progress = st.progress(0.0)
        status = st.empty()

        results: List[Dict[str, Any]] = []

        for i, t in enumerate(real_tasks, start=1):
            fname = t["file_name"]
            sh_name = t.get("sheet_name") or "Sheet"
            ds_id = t["dataset_id"]

            status.info(f"Cleaning {i}/{total}: {fname} — {sh_name}")

            try:
                mode = "llm" if use_llm_all_files else "rule_based"
                pol_i = suggest_policy(ds_id, mode=mode, llm_model=llm_model_all_files)
                st.session_state[f"policy_{ds_id}"] = pol_i
            except Exception as e:
                results.append({"file_name": fname, "sheet_name": sh_name, "dataset_id": ds_id, "policy_error": str(e)})

            try:
                run_id_i = run_cleaning(ds_id, use_llm=use_llm_all_files, llm_model=llm_model_all_files)
                report_i = get_run_report(run_id_i)

                st.session_state["last_run_id"] = run_id_i

                if "runs_store" not in st.session_state or not isinstance(st.session_state.runs_store, dict):
                    st.session_state.runs_store = {}

                st.session_state.runs_store[ds_id] = {
                    "file_name": fname,
                    "sheet_name": sh_name,
                    "run_id": run_id_i,
                    "report": report_i,
                }

                st.session_state[f"run_{ds_id}"] = run_id_i
                st.session_state[f"report_{ds_id}"] = report_i

                results.append({"file_name": fname, "sheet_name": sh_name, "dataset_id": ds_id, "run_id": run_id_i})
            except Exception as e:
                results.append({"file_name": fname, "sheet_name": sh_name, "dataset_id": ds_id, "error": str(e)})

            progress.progress(i / total)

        st.success("Cleaning finished for all uploaded files ✅")
        st.session_state["clean_all_uploaded_files_results"] = results
        st.divider()

    batch = st.session_state.get("clean_all_uploaded_files_results")
    if isinstance(batch, list) and batch:
        st.subheader("Cleaning results")
        st.dataframe(pd.DataFrame(batch), width="stretch", hide_index=True)


def render_tab_cleaning(file_name: str, sheet_meta: dict, dataset_id: str):
    sheet_name = sheet_meta.get("sheet_name") or "Sheet"
    st.subheader(f"Active file – {file_name}: {sheet_name}")

    files_registry = st.session_state.get("files_registry") or {}
    multi_file_mode = isinstance(files_registry, dict) and len(files_registry) >= 2

    if multi_file_mode:

        _render_multi_file_cleaning_block(files_registry)

        active_run_id = st.session_state.get(f"run_{dataset_id}")
        active_report = st.session_state.get(f"report_{dataset_id}")
        active_pol = st.session_state.get(f"policy_{dataset_id}")

        if active_pol:
            _render_policy_block(active_pol, dataset_id=dataset_id)

        if active_report:
            _render_execution_summary(active_report)
            _render_after_preview_and_downloads(
                dataset_id=dataset_id,
                run_id=active_run_id,
                file_name=file_name,
                sheet_name=sheet_name,
            )
            return active_run_id, active_report

        st.info("Clean all files first, then select a file and sheet in Upload Files to view results here.")
        return None, None

    datasets_meta = (
        st.session_state.get("active_datasets_meta")
        or st.session_state.get("datasets_meta")
        or sheet_meta.get("datasets_meta")
        or []
    )

    is_excel = _is_excel_file(file_name)
    has_many_sheets = isinstance(datasets_meta, list) and len(datasets_meta) > 1
    multi_sheet_excel_mode = bool(is_excel and has_many_sheets)

    if multi_sheet_excel_mode:
        st.info(f"Detected excel file with {len(datasets_meta)} sheets, cleaning will run for all sheets.")

        st.subheader("Clean all Excel sheets")

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            use_llm_all = st.checkbox("Use LLM", value=False, key="use_llm_all")
        with c2:
            llm_model_all = st.selectbox(
                "LLM model",
                ["gemini-2.5-flash"],
                index=0,
                disabled=not use_llm_all,
                key="llm_model_all",
            )
        with c3:
            st.caption("")

        if st.button("Clean all Excel sheets", type="primary", width="stretch", key="btn_clean_all"):
            progress = st.progress(0.0)
            status = st.empty()

            cleaned_frames: List[Tuple[str, pd.DataFrame]] = []
            total = len(datasets_meta)

            try:
                for i, meta in enumerate(datasets_meta, start=1):
                    ds_id = meta.get("dataset_id")
                    sh_name = meta.get("sheet_name") or f"Sheet{i}"
                    safe_name = _safe_sheet_name(sh_name, fallback=f"Sheet{i}")

                    if not ds_id:
                        progress.progress(i / total)
                        continue

                    status.info(f"Cleaning {i}/{total}: {sh_name}")

                    try:
                        mode = "llm" if use_llm_all else "rule_based"
                        pol_i = suggest_policy(ds_id, mode=mode, llm_model=llm_model_all)
                        st.session_state[f"policy_{ds_id}"] = pol_i
                    except Exception:
                        pass

                    run_id_i = run_cleaning(ds_id, use_llm=use_llm_all, llm_model=llm_model_all)
                    report_i = get_run_report(run_id_i)

                    st.session_state["last_run_id"] = run_id_i

                    if "runs_store" not in st.session_state or not isinstance(st.session_state.runs_store, dict):
                        st.session_state.runs_store = {}

                    st.session_state.runs_store[ds_id] = {
                        "file_name": file_name,
                        "sheet_name": sh_name,
                        "run_id": run_id_i,
                        "report": report_i,
                    }

                    st.session_state[f"run_{ds_id}"] = run_id_i
                    st.session_state[f"report_{ds_id}"] = report_i

                    xlsx_bytes = download_artifact(run_id_i, "cleaned.xlsx")
                    df_clean = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name=0)
                    cleaned_frames.append((safe_name, df_clean))

                    progress.progress(i / total)

                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
                    used = set()
                    for name, df in cleaned_frames:
                        final_name = name
                        if final_name in used:
                            base = final_name[:28]
                            k = 1
                            while f"{base}_{k}" in used:
                                k += 1
                            final_name = f"{base}_{k}"
                        used.add(final_name)
                        df.to_excel(writer, index=False, sheet_name=final_name)

                out.seek(0)
                st.success("All sheets cleaned ✅")
                st.divider()

            except Exception as e:
                st.error(f"Clean ALL failed: {e}")

        active_run_id = st.session_state.get(f"run_{dataset_id}")
        active_report = st.session_state.get(f"report_{dataset_id}")
        active_pol = st.session_state.get(f"policy_{dataset_id}")

        if active_pol:
            _render_policy_block(active_pol, dataset_id=dataset_id)

        if active_report:
            _render_execution_summary(active_report)
            _render_after_preview_and_downloads(
                dataset_id=dataset_id,
                run_id=active_run_id,
                file_name=file_name,
                sheet_name=sheet_name,
            )
            return active_run_id, active_report

        st.info("Run 'Clean all Excel sheets' to generate cleaned data + reports for all sheets.")
        return None, None

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        use_llm = st.checkbox("Use LLM", value=False, key=f"use_llm_{dataset_id}")
    with c2:
        llm_model = st.selectbox(
            "LLM model",
            ["gemini-2.5-flash"],
            index=0,
            disabled=not use_llm,
            key=f"llm_model_{dataset_id}",
        )
    with c3:
        st.caption("")

    if st.button("Run cleaning", type="primary", width="stretch", key=f"btn_run_clean_{dataset_id}"):
        try:
            with st.spinner("Running on backend…"):
                mode = "llm" if use_llm else "rule_based"
                pol = suggest_policy(dataset_id, mode=mode, llm_model=llm_model)
                st.session_state[f"policy_{dataset_id}"] = pol

                run_id = run_cleaning(dataset_id, use_llm=use_llm, llm_model=llm_model)
                report = get_run_report(run_id)

                st.session_state["last_run_id"] = run_id

                if "runs_store" not in st.session_state or not isinstance(st.session_state.runs_store, dict):
                    st.session_state.runs_store = {}

                st.session_state.runs_store[dataset_id] = {
                    "file_name": file_name,
                    "sheet_name": sheet_name,
                    "run_id": run_id,
                    "report": report,
                }

                st.session_state[f"run_{dataset_id}"] = run_id
                st.session_state[f"report_{dataset_id}"] = report

            st.success("Cleaning finished ✅")
        except Exception as e:
            st.error(f"Cleaning failed: {e}")

    pol = st.session_state.get(f"policy_{dataset_id}")
    if pol:
        _render_policy_block(pol, dataset_id=dataset_id)

    run_id = st.session_state.get(f"run_{dataset_id}")
    report = st.session_state.get(f"report_{dataset_id}")

    if report:
        _render_execution_summary(report)
        _render_after_preview_and_downloads(
            dataset_id=dataset_id,
            run_id=run_id,
            file_name=file_name,
            sheet_name=sheet_name,
        )
        return run_id, report

    return None, None