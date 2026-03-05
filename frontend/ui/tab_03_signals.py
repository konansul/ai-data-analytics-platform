# frontend/ui/tab_03_signals.py
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st


def _to_jsonish(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (dict, list, tuple, set)):
        try:
            return str(v)
        except Exception:
            return "<unserializable>"
    return v


def _kv_table(d: dict, key_name: str, value_name: str) -> pd.DataFrame:
    d = d or {}
    rows = []
    for k, v in d.items():
        rows.append({key_name: str(k), value_name: _to_jsonish(v)})
    return pd.DataFrame(rows)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, (np.integer,)):
            return int(x.item())
        return int(float(x))
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (np.floating, np.integer)):
            return float(x.item())
        return float(x)
    except Exception:
        return default


def _normalize_counts(profile: Dict[str, Any]) -> Dict[str, int]:
    counts = profile.get("counts") or profile.get("column_groups")
    if isinstance(counts, dict) and counts:
        return {str(k): _safe_int(v, 0) for k, v in counts.items()}
    cols = profile.get("columns") or {}
    if isinstance(cols, dict) and cols:
        return {str(k): len(v) if isinstance(v, list) else 0 for k, v in cols.items()}
    return {}


def _get_top_missing(profile: Dict[str, Any]) -> Dict[str, float]:
    miss = profile.get("missingness") or {}
    top = miss.get("top_missing_columns") or profile.get("top_missing_columns") or {}
    if not isinstance(top, dict):
        return {}
    return {str(k): _safe_float(v, 0.0) for k, v in top.items()}


def _get_corr_pairs(profile: Dict[str, Any]) -> list:
    corr = profile.get("correlation") or {}
    pairs = corr.get("top_abs_pairs") or []
    return pairs if isinstance(pairs, list) else []


def _get_skew_top(profile: Dict[str, Any]) -> Dict[str, float]:
    skew = profile.get("skewness") or {}
    top = skew.get("top_abs_skewed") or profile.get("skewness_top_abs") or {}
    if not isinstance(top, dict):
        return {}
    return {str(k): _safe_float(v, 0.0) for k, v in top.items()}


def _profile_overall_missing(profile: Dict[str, Any]) -> float:
    miss = profile.get("missingness") or {}
    return _safe_float(miss.get("overall_missing_%"), 0.0)


def _render_profile_block(title: str, profile: Optional[Dict[str, Any]]):
    profile = profile or {}

    st.subheader(title)

    dataset_type = str(profile.get("dataset_type"))
    n_rows = _safe_int(profile.get("n_rows"))
    n_cols = _safe_int(profile.get("n_cols"))
    overall_missing = _profile_overall_missing(profile)
    has_time_index = bool(profile.get("has_time_index"))
    time_column = profile.get("time_column")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Dataset type", dataset_type)
    with c2:
        st.metric("Rows", n_rows)
    with c3:
        st.metric("Cols", n_cols)
    with c4:
        st.metric("Overall missing %", overall_missing)

    c5, c6 = st.columns(2)
    with c5:
        st.metric("Has time index", has_time_index)
    with c6:
        st.metric("Time column", "None" if time_column is None else str(time_column))

    st.write("**Column groups (counts)**")
    counts = _normalize_counts(profile)
    if counts:
        st.dataframe(_kv_table(counts, "group", "count"), width="stretch", hide_index=True)
    else:
        st.info("No column group info found in profile.")

    st.write("**Top missing columns (%)**")
    miss = _get_top_missing(profile)
    if miss:
        miss_df = _kv_table(miss, "column", "missing_%")
        if "missing_%" in miss_df.columns:
            miss_df["missing_%"] = pd.to_numeric(miss_df["missing_%"], errors="coerce")
        miss_df = miss_df.sort_values("missing_%", ascending=False, na_position="last")
        st.dataframe(miss_df, width="stretch", hide_index=True)
    else:
        st.info("No missingness info found in profile.")

    st.write("**Top correlations (abs)**")
    corr = _get_corr_pairs(profile)
    if corr:
        corr_df = pd.DataFrame([{k: _to_jsonish(v) for k, v in row.items()} for row in corr])
        st.dataframe(corr_df, width="stretch", hide_index=True)
    else:
        st.info("No correlation pairs (need >= 2 numeric columns).")

    st.write("**Skewness (top abs)**")
    skew = _get_skew_top(profile)
    if skew:
        skew_df = _kv_table(skew, "column", "skew")
        if "skew" in skew_df.columns:
            skew_df["skew"] = pd.to_numeric(skew_df["skew"], errors="coerce")
            skew_df["abs_skew"] = skew_df["skew"].abs()
            skew_df = skew_df.sort_values("abs_skew", ascending=False, na_position="last").drop(columns=["abs_skew"])
        st.dataframe(skew_df, width="stretch", hide_index=True)
    else:
        st.info("No skewness computed (need enough numeric data).")

    warnings = profile.get("warnings") or []
    if isinstance(warnings, list) and warnings:
        st.warning("\n".join([str(x) for x in warnings]))



def render_tab_signals(cleaning_report: dict):
    pre  = (cleaning_report or {}).get("pre_profile")
    post = (cleaning_report or {}).get("post_profile")

    if not isinstance(pre, dict) and not isinstance(post, dict):
        st.error("No profiling data found in report. Make sure pipeline saves pre_profile/post_profile.")
        return

    dataset_id = st.session_state.get("active_dataset_id")

    tab_pre, tab_post = st.tabs(["Before cleaning", "After cleaning"])

    with tab_pre:
        _render_profile_block("Before cleaning", pre if isinstance(pre, dict) else {})

    with tab_post:
        _render_profile_block("After cleaning", post if isinstance(post, dict) else {})