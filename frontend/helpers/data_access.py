from __future__ import annotations

import requests
import base64
import streamlit as st
from typing import Any, Dict, List, Optional

API_BASE = "http://127.0.0.1:8000/v1"
TIMEOUT = 120

def dataset_download_bytes(dataset_id: str, version: str, fmt: str = "csv") -> bytes:
    return download_dataset(dataset_id=dataset_id, version=version, fmt=fmt)

def _raise(resp: requests.Response):
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"API error {resp.status_code}: {detail}")

def _auth_headers(token: Optional[str] = None) -> Dict[str, str]:

    tok = token or st.session_state.get("auth_token")
    if not tok or not isinstance(tok, str):
        return {}
    return {"Authorization": f"Bearer {tok}"}

def register_user(email: str, password: str) -> Dict[str, Any]:
    payload = {"email": email, "password": password}
    resp = requests.post(f"{API_BASE}/auth/register", json=payload, timeout=TIMEOUT)
    _raise(resp)
    return resp.json()


def login_user(email: str, password: str) -> Dict[str, Any]:
    payload = {"email": email, "password": password}
    resp = requests.post(f"{API_BASE}/auth/login", json=payload, timeout=TIMEOUT)
    _raise(resp)
    data = resp.json()

    token = data.get("access_token")
    if not token:
        raise RuntimeError("Login succeeded but no access_token returned")

    return data


def auth_me() -> Dict[str, Any]:
    resp = requests.get(f"{API_BASE}/auth/me", headers=_auth_headers(), timeout=TIMEOUT)
    _raise(resp)
    return resp.json()


def logout_user() -> None:
    st.session_state.pop("auth_token", None)
    try:
        st.cache_data.clear()
    except Exception:
        pass

@st.cache_data(show_spinner=False)
def cached_upload(file_bytes: bytes, filename: str, token_cache_key: str) -> List[Dict[str, Any]]:

    files = {"file": (filename, file_bytes)}

    resp = requests.post(
        f"{API_BASE}/datasets",
        files=files,
        headers=_auth_headers(),
        timeout=TIMEOUT,
    )
    _raise(resp)
    return resp.json()["datasets"]


def get_preview(dataset_id: str, rows: int = 50) -> Dict[str, Any]:
    resp = requests.get(
        f"{API_BASE}/datasets/{dataset_id}/preview",
        params={"rows": rows},
        headers=_auth_headers(),
        timeout=TIMEOUT,
    )
    _raise(resp)
    return resp.json()


def get_meta(dataset_id: str) -> Dict[str, Any]:
    resp = requests.get(
        f"{API_BASE}/datasets/{dataset_id}",
        headers=_auth_headers(),
        timeout=TIMEOUT,
    )
    _raise(resp)
    return resp.json()


def download_dataset(dataset_id: str, version: str = "current", fmt: str = "xlsx") -> bytes:
    resp = requests.get(
        f"{API_BASE}/datasets/{dataset_id}/download",
        params={"version": version, "fmt": fmt},
        headers=_auth_headers(),
        timeout=TIMEOUT,
    )
    _raise(resp)
    return resp.content

def run_profiling(dataset_id: str, options: Optional[Dict[str, Any]] = None) -> str:
    resp = requests.post(
        f"{API_BASE}/profiling",
        json={"dataset_id": dataset_id, "options": options},
        headers=_auth_headers(),
        timeout=TIMEOUT,
    )
    _raise(resp)
    return resp.json()["profile_id"]


def get_profile(profile_id: str) -> Dict[str, Any]:
    resp = requests.get(
        f"{API_BASE}/profiling/{profile_id}",
        headers=_auth_headers(),
        timeout=TIMEOUT,
    )
    _raise(resp)
    return resp.json()

def suggest_policy(dataset_id: str, mode: str = "rule_based", llm_model: str = "gemini-2.5-flash") -> Dict[str, Any]:
    resp = requests.post(
        f"{API_BASE}/policy/suggest",
        json={"dataset_id": dataset_id, "mode": mode, "llm_model": llm_model},
        headers=_auth_headers(),
        timeout=TIMEOUT,
    )
    _raise(resp)
    return resp.json()

def run_cleaning(
        dataset_id: str,
        use_llm: bool = False,
        llm_model: str = "gemini-2.5-flash",
        overrides: Optional[Dict[str, Any]] = None,
) -> str:
    payload: Dict[str, Any] = {"dataset_id": dataset_id, "use_llm": use_llm, "llm_model": llm_model}
    if overrides:
        payload.update(overrides)

    resp = requests.post(
        f"{API_BASE}/cleaning/runs",
        json=payload,
        headers=_auth_headers(),
        timeout=TIMEOUT,
    )
    _raise(resp)
    return resp.json()["run_id"]


def get_run_status(run_id: str) -> Dict[str, Any]:
    resp = requests.get(
        f"{API_BASE}/cleaning/runs/{run_id}",
        headers=_auth_headers(),
        timeout=TIMEOUT,
    )
    _raise(resp)
    return resp.json()


def get_run_report(run_id: str) -> Dict[str, Any]:
    resp = requests.get(
        f"{API_BASE}/cleaning/runs/{run_id}/report",
        headers=_auth_headers(),
        timeout=TIMEOUT,
    )
    _raise(resp)
    return resp.json()

def download_artifact(run_id: str, name: str) -> bytes:
    resp = requests.get(
        f"{API_BASE}/cleaning/runs/{run_id}/artifacts/{name}",
        headers=_auth_headers(),
        timeout=TIMEOUT,
    )
    _raise(resp)
    return resp.content

def list_my_runs(token: str) -> Dict[str, Any]:
    resp = requests.get(f"{API_BASE}/cleaning/runs", headers=_auth_headers(token), timeout=TIMEOUT)
    _raise(resp)
    return resp.json()

def delete_run(run_id: str) -> Dict[str, Any]:
    resp = requests.delete(
        f"{API_BASE}/cleaning/runs/{run_id}",
        headers=_auth_headers(),
        timeout=TIMEOUT,
    )
    _raise(resp)
    return resp.json()

def get_visualization_pairings(dataset_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{API_BASE}/visualization/pairings",
        json={"dataset_id": dataset_id, "profile_data": profile_data},
        headers=_auth_headers(),
        timeout=120,
    )
    _raise(resp)
    return resp.json()


def get_visualization_plots(dataset_id: str, profile_data: Dict[str, Any], selected_pairings: List[Dict[str, Any]],) -> List[Dict[str, Any]]:
    resp = requests.post(
        f"{API_BASE}/visualization/plots",
        json={
            "dataset_id": dataset_id,
            "profile_data": profile_data,
            "selected_pairings": selected_pairings,
        },
        headers=_auth_headers(),
        timeout=120,
    )
    _raise(resp)
    return resp.json()


def explain_chart(plot_title: str, x_col: str, y_col: str) -> str:
    resp = requests.post(
        f"{API_BASE}/visualization/explain",
        json={"plot_title": plot_title, "axis_info": f"X-Axis: {x_col}, Y-Axis: {y_col}"},
        headers=_auth_headers(),
        timeout=30,
    )
    if resp.status_code == 200:
        return resp.json().get("explanation", "No explanation available.")
    return f"Error: {resp.text}"


def api_forecast_signals(*, dataset_id: str, version: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{API_BASE}/forecast/signals",
        json={"dataset_id": dataset_id, "version": version},
        headers=_auth_headers(),
        timeout=120,
    )
    _raise(resp)
    return resp.json()


def api_forecast_plan(
    *,
    dataset_id: str,
    version: str,
    signals: Dict[str, Any],
    profile: Optional[Dict[str, Any]],
    user_intent: Optional[str],
    max_targets: int,
    head_rows: int,
    horizon: int,
) -> Dict[str, Any]:
    payload = {
        "dataset_id": dataset_id,
        "version": version,
        "signals": signals,
        "profile": profile,
        "user_intent": user_intent,
        "max_targets": int(max_targets),
        "head_rows": int(head_rows),
        "horizon": int(horizon),
    }
    resp = requests.post(
        f"{API_BASE}/forecast/plan",
        json=payload,
        headers=_auth_headers(),
        timeout=180,
    )
    _raise(resp)
    return resp.json()


def api_forecast_run(
    *,
    dataset_id: str,
    version: str,
    run_id: str,
    plan: Dict[str, Any],
    horizon: int,
    model: str,
    preview_rows: int = 50,
) -> Dict[str, Any]:
    payload = {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "plan": plan,
        "model": model,
        "version": version,
        "horizon": int(horizon),
        "preview_rows": int(preview_rows),
    }
    resp = requests.post(
        f"{API_BASE}/forecast/run",
        json=payload,
        headers=_auth_headers(),
        timeout=600,
    )
    _raise(resp)
    return resp.json()

def api_save_viz_plot(*, dataset_id: str, run_id: str, title: str, png_bytes: bytes) -> None:
    payload = {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "title": title,
        "png_base64": base64.b64encode(png_bytes).decode("utf-8"),
    }
    resp = requests.post(
        f"{API_BASE}/visualization/plots/save",
        json=payload,
        headers=_auth_headers(),
        timeout=60,
    )
    _raise(resp)

def api_save_forecast_plot(*, forecast_run_id: str, dataset_id: str, target: str, png_bytes: bytes) -> None:
    payload = {
        "forecast_run_id": forecast_run_id,
        "dataset_id": dataset_id,
        "target": target,
        "png_base64": base64.b64encode(png_bytes).decode("utf-8"),
    }
    resp = requests.post(
        f"{API_BASE}/forecast/plots",
        json=payload,
        headers=_auth_headers(),
        timeout=60,
    )
    _raise(resp)


def api_generate_pdf_report(
    *,
    dataset_id: str,
    run_id: str,
    title: str | None = None,
    max_viz_plots: int = 3,
    max_forecast_plots: int = 3,
) -> dict:
    url = f"{API_BASE}/reporting/generate"
    payload = {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "title": title,
        "max_viz_plots": max_viz_plots,
        "max_forecast_plots": max_forecast_plots,
        "return_pdf": False,
    }

    r = requests.post(url, json=payload, headers=_auth_headers(), timeout=180)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()


def api_download_report_pdf(*, storage_key: str) -> bytes:
    url = f"{API_BASE}/reporting/download"
    r = requests.get(url, params={"storage_key": storage_key}, headers=_auth_headers(), timeout=180)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.content