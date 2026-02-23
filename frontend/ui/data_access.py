from __future__ import annotations

import requests
import streamlit as st
from typing import Any, Dict, List, Optional

API_BASE = "http://127.0.0.1:8000/v1"
TIMEOUT = 240


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

def forecast_signals(dataset_id: str, version: str = "current") -> Dict[str, Any]:
    resp = requests.post(
        f"{API_BASE}/forecast/signals",
        json={"dataset_id": dataset_id, "version": version},
        headers=_auth_headers(),
        timeout=TIMEOUT,
    )
    _raise(resp)
    return resp.json()


def forecast_plan(
    dataset_id: str,
    signals: Dict[str, Any],
    profile: Optional[Dict[str, Any]] = None,
    user_intent: Optional[str] = None,
    head_rows: int = 10,
    max_targets: int = 3,
    version: str = "current",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "version": version,
        "signals": signals,
        "profile": profile,
        "user_intent": user_intent,
        "head_rows": int(head_rows),
        "max_targets": int(max_targets),
    }
    resp = requests.post(
        f"{API_BASE}/forecast/plan",
        json=payload,
        headers=_auth_headers(),
        timeout=TIMEOUT,
    )
    _raise(resp)
    return resp.json()


def forecast_run(
    dataset_id: str,
    plan: Dict[str, Any],
    horizon: int = 30,
    model: str = "auto",
    version: str = "current",
    preview_rows: Optional[int] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "version": version,
        "plan": plan,
        "horizon": int(horizon),
        "model": model,
        "preview_rows": int(preview_rows if preview_rows is not None else min(int(horizon), 500)),
    }
    resp = requests.post(
        f"{API_BASE}/forecast/run",
        json=payload,
        headers=_auth_headers(),
        timeout=TIMEOUT,
    )
    _raise(resp)
    return resp.json()