# frontend/ui/tab_07_save_all_files.py
from __future__ import annotations

import io
import zipfile
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import streamlit as st

from frontend.helpers.data_access import list_my_runs, download_artifact, delete_run


def _safe(s: str) -> str:
    return (s or "").replace("/", "_").replace("\\", "_").replace(" ", "_")


def _safe_sheet_name(name: str) -> str:
    bad = r'[]:*?/\\'
    n = (name or "").strip() or "Sheet"
    for ch in bad:
        n = n.replace(ch, "_")
    n = n[:31].strip()
    return n or "Sheet"


def _parse_dt(s: Optional[str]) -> datetime:
    if not s:
        return datetime.min
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return datetime.min


def _is_excel_name(file_name: str) -> bool:
    f = (file_name or "").lower()
    return f.endswith(".xlsx") or f.endswith(".xls")


def _is_csv_name(file_name: str) -> bool:
    f = (file_name or "").lower()
    return f.endswith(".csv")


def _pick_latest_per_sheet(entries: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Return [(sheet_name, run_dict)] picking latest run per sheet_name."""
    latest_per_sheet: Dict[str, Dict[str, Any]] = {}
    for r in entries:
        sheet = r.get("sheet_name") or "Sheet"
        cur = latest_per_sheet.get(sheet)
        if cur is None or _parse_dt(r.get("created_at")) > _parse_dt(cur.get("created_at")):
            latest_per_sheet[sheet] = r
    return sorted(latest_per_sheet.items(), key=lambda kv: kv[0].lower())


def render_tab_saved_datasets():
    token = st.session_state.get("auth_token")
    st.subheader("Save cleaned datasets")

    if not token:
        st.info("Please login first.")
        return

    try:
        payload = list_my_runs(token)
        runs: List[Dict[str, Any]] = payload.get("runs", []) or []
    except Exception as e:
        st.error(f"Failed to load runs: {e}")
        return

    if not runs:
        st.info("No cleaned datasets yet.")
        return

    by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        file_name = r.get("file_name") or "(unknown file)"
        by_file[file_name].append(r)

    for file_name, entries in sorted(by_file.items(), key=lambda x: x[0].lower()):
        sheets_sorted = _pick_latest_per_sheet(entries)
        sheet_names = [s for s, _ in sheets_sorted]

        is_excel = _is_excel_name(file_name)
        is_csv = _is_csv_name(file_name)

        if not is_excel and not is_csv:
            if len(sheet_names) > 1:
                is_excel = True
            else:
                is_csv = True

        with st.container(border=True):
            st.write(f"📄 **{file_name}**")

            if is_excel and len(sheet_names) > 1:
                st.caption(f"Sheets: {', '.join(sheet_names)}")
            elif is_excel and len(sheet_names) == 1:
                st.caption(f"Sheet: {sheet_names[0]}")
            else:
                st.caption("CSV dataset")

            b1, b2, b3 = st.columns([2.4, 2.0, 1.4])

            with b1:
                if is_csv:
                    latest_run = max(entries, key=lambda r: _parse_dt(r.get("created_at")))
                    run_id = latest_run["run_id"]

                    if st.button("⬇️ Download cleaned CSV", key=f"dl_csv_{file_name}", width="stretch"):
                        try:
                            cleaned_bytes = download_artifact(run_id, "cleaned.xlsx")
                            df = pd.read_excel(io.BytesIO(cleaned_bytes), sheet_name=0)

                            csv_bytes = df.to_csv(index=False).encode("utf-8")
                            out_name = _safe(file_name.rsplit(".", 1)[0]) + "_cleaned.csv"

                            st.download_button(
                                "✅ Click to download",
                                data=csv_bytes,
                                file_name=out_name,
                                mime="text/csv",
                                key=f"dl_csv_btn_{file_name}",
                                width="stretch",
                            )
                        except Exception as e:
                            st.error(f"Download cleaned CSV failed: {e}")

                else:
                    if st.button("⬇️ Download combined Excel", key=f"dl_xlsx_{file_name}", width="stretch"):
                        try:
                            out = io.BytesIO()
                            used = set()
                            with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
                                for sheet_name, r in sheets_sorted:
                                    run_id = r["run_id"]
                                    cleaned_bytes = download_artifact(run_id, "cleaned.xlsx")
                                    df = pd.read_excel(io.BytesIO(cleaned_bytes), sheet_name=0)

                                    safe_sheet = _safe_sheet_name(sheet_name)
                                    final_sheet = safe_sheet
                                    if final_sheet in used:
                                        base = final_sheet[:28]
                                        k = 1
                                        while f"{base}_{k}" in used:
                                            k += 1
                                        final_sheet = f"{base}_{k}"
                                    used.add(final_sheet)

                                    df.to_excel(writer, index=False, sheet_name=final_sheet)

                            out.seek(0)
                            out_name = _safe(file_name.rsplit(".", 1)[0]) + "_cleaned.xlsx"

                            st.download_button(
                                "✅ Click to download",
                                data=out.getvalue(),
                                file_name=out_name,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"dl_xlsx_btn_{file_name}",
                                width="stretch",
                            )
                        except Exception as e:
                            st.error(f"Build combined Excel failed: {e}")

            with b2:
                if st.button("⬇️ Download reports.zip", key=f"dl_reports_{file_name}", width="stretch"):
                    try:
                        zbuf = io.BytesIO()
                        with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                            if is_csv:
                                latest_run = max(entries, key=lambda r: _parse_dt(r.get("created_at")))
                                run_id = latest_run["run_id"]
                                rep = download_artifact(run_id, "report.json")
                                zf.writestr(f"{_safe(file_name.rsplit('.',1)[0])}_{run_id}_report.json", rep)
                            else:
                                for sheet_name, r in sheets_sorted:
                                    run_id = r["run_id"]
                                    rep = download_artifact(run_id, "report.json")
                                    zf.writestr(f"{_safe_sheet_name(sheet_name)}_{run_id}_report.json", rep)

                        zbuf.seek(0)
                        out_name = _safe(file_name.rsplit(".", 1)[0]) + "_reports.zip"

                        st.download_button(
                            "✅ Click to download",
                            data=zbuf.getvalue(),
                            file_name=out_name,
                            mime="application/zip",
                            key=f"dl_zip_btn_{file_name}",
                            width="stretch",
                        )
                    except Exception as e:
                        st.error(f"Build reports.zip failed: {e}")

            with b3:
                if st.button("❌ Delete file", key=f"del_file_{file_name}", width="stretch"):
                    try:
                        for r in entries:
                            delete_run(r["run_id"])
                        st.success("Deleted ✅")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")