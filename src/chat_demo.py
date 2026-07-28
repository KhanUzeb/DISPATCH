"""Demo-only Streamlit chat UI for Dispatch.

Does not contain routing logic — it only calls the local FastAPI service.
Run API first, then:

  uv run streamlit run src/chat_demo.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Dispatch chat demo", layout="centered")

st.markdown(
    """
<style>
  .stApp { background-color: #0d1117; color: #c9d1d9; }
  h1, h2, h3 { color: #58a6ff; font-family: monospace; }
  [data-testid="stMetricValue"] { color: #3fb950; font-family: monospace; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Dispatch // chat demo")
st.caption("Demo UI only. Routing happens in the FastAPI service.")

with st.sidebar:
    st.header("Connection")
    api_base = st.text_input(
        "API base URL",
        value=os.environ.get("ROUTER_API_BASE", "http://localhost:8000"),
    ).rstrip("/")
    mode = st.radio("Endpoint", ["complete", "openai /v1"], index=0)
    max_tokens = st.slider("Max output tokens", min_value=64, max_value=1024, value=192, step=32)
    structured = st.checkbox("Expects structured output", value=False)
    api_key = st.text_input(
        "Bearer token (optional)",
        value=os.environ.get("ROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "sk-demo",
        type="password",
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


def call_complete(prompt: str) -> dict:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(
        f"{api_base}/complete",
        headers=headers,
        json={
            "prompt": prompt,
            "max_output_tokens": max_tokens,
            "expects_structured_output": structured,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def call_openai(prompt: str) -> dict:
    resp = requests.post(
        f"{api_base}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "dispatch",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
            "expects_structured_output": structured,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def normalize_result(raw: dict, used_mode: str) -> dict:
    if used_mode == "complete":
        return {
            "text": raw.get("response") or "",
            "model": raw.get("provider_model_id") or raw.get("model_key"),
            "provider": raw.get("provider"),
            "tier": raw.get("tier"),
            "latency_ms": raw.get("latency_ms"),
            "cost": (raw.get("cost") or {}).get("actual"),
            "request_id": raw.get("request_id"),
            "raw": raw,
        }
    choice = ((raw.get("choices") or [{}])[0].get("message") or {})
    meta = raw.get("dispatch") or {}
    return {
        "text": choice.get("content") or "",
        "model": raw.get("model") or meta.get("model_key"),
        "provider": meta.get("provider"),
        "tier": meta.get("tier"),
        "latency_ms": None,
        "cost": meta.get("cost_usd"),
        "request_id": meta.get("request_id"),
        "raw": raw,
    }


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            m = msg["meta"]
            st.caption(
                f"tier={m.get('tier')} · provider={m.get('provider')} · model={m.get('model')} · "
                f"latency={m.get('latency_ms')} ms · cost=${m.get('cost')}"
            )

prompt = st.chat_input("Ask Dispatch…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Routing + generating…"):
            try:
                raw = call_complete(prompt) if mode == "complete" else call_openai(prompt)
                result = normalize_result(raw, "complete" if mode == "complete" else "openai")
                st.markdown(result["text"] or "_(empty response)_")
                cols = st.columns(4)
                cols[0].metric("Tier", result.get("tier") or "-")
                cols[1].metric("Provider", result.get("provider") or "-")
                cols[2].metric("Latency (ms)", f"{(result.get('latency_ms') or 0):.0f}")
                cols[3].metric("Cost ($)", f"{(result.get('cost') or 0):.5f}")
                with st.expander("Raw response"):
                    st.json(result["raw"])
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["text"] or "_(empty response)_",
                        "meta": result,
                    }
                )
            except requests.HTTPError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"API error: {detail}")
            except requests.RequestException as exc:
                st.error(
                    f"Could not reach API at {api_base}. "
                    f"Start it with: uv run uvicorn src.api:app --reload\n\n{exc}"
                )
