import pandas as pd
import streamlit as st


def sheets_summary_table(sheets_or_meta):

    rows = []
    for i, s in enumerate(sheets_or_meta or [], start=1):
        if isinstance(s, dict):
            sheet_name = s.get("sheet_name")
            shape = s.get("shape") or [None, None]
            dataset_id = s.get("dataset_id")
        else:
            sheet_name = getattr(s, "sheet_name", None)
            shape = getattr(s, "shape", (None, None)) or (None, None)
            dataset_id = getattr(s, "dataset_id", None)

        if isinstance(shape, (tuple, list)) and len(shape) >= 2:
            n_rows, n_cols = shape[0], shape[1]
        else:
            n_rows, n_cols = None, None

        rows.append({
            "#": i,
            "Sheet": sheet_name,
            "Rows": n_rows,
            "Cols": n_cols,
            "Dataset ID": dataset_id,
        })

    return pd.DataFrame(rows)


def render_cleaning_report(report: dict):
    report = report or {}

    dropped_total = int(report.get("dropped_total", 0))
    total_filled = int((report.get("imputation") or {}).get("total_filled", 0))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{report.get('rows_before', 0)} → {report.get('rows_after', 0)}")
    c2.metric("Cols", f"{report.get('cols_before', 0)} → {report.get('cols_after', 0)}")
    c3.metric("Dropped total", dropped_total)
    c4.metric("Imputed (filled)", total_filled)
    c5.metric("Datetime inferred", len(report.get("inferred_datetime_columns", []) or []))

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Dropped columns")
        dropped = (
            (report.get("dropped_empty_columns") or [])
            + (report.get("dropped_constant_columns") or [])
            + (report.get("dropped_high_missing_columns") or [])
        )
        st.write(", ".join(dropped)) if dropped else st.info("No columns removed.")

        st.subheader("Renamed columns")
        renamed = report.get("normalized_columns") or {}
        if renamed:
            st.dataframe(
                pd.DataFrame([{"old": k, "new": v} for k, v in renamed.items()]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No renames.")

        st.subheader("Dtypes changed (only diffs)")
        dtypes_changed = report.get("dtypes_changed") or {}
        if dtypes_changed:
            df_dt = pd.DataFrame([
                {"column": c, "before": v.get("before"), "after": v.get("after")}
                for c, v in dtypes_changed.items()
            ]).sort_values("column")
            st.dataframe(df_dt, width="stretch", hide_index=True)
        else:
            st.info("No dtype changes detected.")

        with st.expander("Show all dtypes (before/after)", expanded=False):
            dtypes_before = report.get("dtypes_before") or {}
            dtypes_after = report.get("dtypes_after") or {}
            cols = sorted(set(dtypes_before.keys()) | set(dtypes_after.keys()))
            df_all_dt = pd.DataFrame([
                {"column": c, "before": dtypes_before.get(c), "after": dtypes_after.get(c)}
                for c in cols
            ])
            st.dataframe(df_all_dt, width="stretch", hide_index=True)

    with right:
        st.subheader("Missing values — before vs after (only diffs)")
        miss_before = report.get("missing_fraction_before") or {}
        miss_after = report.get("missing_fraction_after") or {}
        miss_changed = report.get("missing_changed") or {}

        if miss_before and miss_after:
            rows = []
            if miss_changed:
                for c, v in miss_changed.items():
                    b = float(v.get("before", 0.0)) * 100
                    a = float(v.get("after", 0.0)) * 100
                    rows.append({"column": c, "missing_%_before": b, "missing_%_after": a, "delta_pp": a - b})
            else:
                for c in sorted(set(miss_before) & set(miss_after)):
                    b = float(miss_before.get(c, 0.0)) * 100
                    a = float(miss_after.get(c, 0.0)) * 100
                    if a != b:
                        rows.append({"column": c, "missing_%_before": b, "missing_%_after": a, "delta_pp": a - b})

            if rows:
                st.dataframe(pd.DataFrame(rows).sort_values("delta_pp"), width="stretch", hide_index=True)
            else:
                st.info("No missingness changes detected.")
        else:
            st.info("Missing before/after info is not available.")

        with st.expander("Show all missingness (before/after)", expanded=False):
            if miss_before and miss_after:
                cols = sorted(set(miss_before) | set(miss_after))
                df_all_m = pd.DataFrame([{
                    "column": c,
                    "missing_%_before": float(miss_before.get(c, 0.0)) * 100,
                    "missing_%_after": float(miss_after.get(c, 0.0)) * 100,
                    "delta_pp": (float(miss_after.get(c, 0.0)) - float(miss_before.get(c, 0.0))) * 100
                } for c in cols]).sort_values("delta_pp")
                st.dataframe(df_all_m, width="stretch", hide_index=True)
            else:
                st.info("Missingness before/after not available.")

        st.subheader("Imputation details")
        imp = report.get("imputation") or {}
        filled_counts = imp.get("filled_counts") or {}
        fill_values_used = imp.get("fill_values_used") or {}

        if filled_counts:
            df_fill = pd.DataFrame([
                {"column": c, "filled": int(n), "value_used": fill_values_used.get(c)}
                for c, n in filled_counts.items()
                if int(n) > 0
            ]).sort_values("filled", ascending=False)
            st.dataframe(df_fill, width="stretch", hide_index=True)
        else:
            st.info("No values were imputed (or imputation is disabled).")

        inferred_dt = report.get("inferred_datetime_columns") or []
        if inferred_dt:
            st.caption("Datetime inferred columns: " + ", ".join(inferred_dt))

    with st.expander("Raw JSON (debug)", expanded=False):
        st.json(report)