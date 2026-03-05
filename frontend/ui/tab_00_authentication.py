# frontend/ui/tab_00_authentication.py
from __future__ import annotations

from typing import Optional
from frontend.helpers.data_access import register_user, login_user, auth_me, logout_user

import streamlit as st
import extra_streamlit_components as stx

cookie_manager = stx.CookieManager()
COOKIE_KEY = "auth_token"

def _set_token(token: Optional[str]) -> None:
    if token and isinstance(token, str):
        st.session_state["auth_token"] = token
        cookie_manager.set(COOKIE_KEY, token, key="set_auth_cookie")
    else:
        st.session_state.pop("auth_token", None)
        cookie_manager.delete(COOKIE_KEY, key="del_auth_cookie")


def render_tab_auth() -> None:
    st.subheader("Authentication")

    if not st.session_state.get("auth_token"):
        tok = cookie_manager.get(COOKIE_KEY)
        if isinstance(tok, str) and tok:
            st.session_state["auth_token"] = tok

    token = st.session_state.get("auth_token")
    if token:
        try:
            me = auth_me()
            st.success(f"Logged in as {me.get('email')} ✅")
        except Exception:
            st.warning("Token exists but is invalid/expired. Please login again.")
            _set_token(None)
            st.rerun()

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Logout"):
                logout_user()
                _set_token(None)
                st.toast("Logged out")
                st.rerun()

        st.divider()

    mode = st.radio("Choose action", ["Login", "Register"], horizontal=True)

    email = st.text_input("Email", value="", placeholder="you@example.com")
    password = st.text_input("Password", value="", type="password", placeholder="min 8 chars")

    colA, _ = st.columns([1, 3])

    if mode == "Register":
        with colA:
            if st.button("Create account"):
                try:
                    res = register_user(email=email, password=password)
                    st.success(f"User created: {res.get('email')}")
                    st.info("Now make log in.")
                except Exception as e:
                    st.error(str(e))

    if mode == "Login":
        with colA:
            if st.button("Login"):
                try:
                    data = login_user(email=email, password=password)  #
                    token_str = data["access_token"]
                    _set_token(token_str)
                    st.success("Logged in successfully ✅")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))