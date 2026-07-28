"""Optional Streamlit dashboard for Langfuse-backed Dispatch metrics.

Prefer the local HTML dashboard at http://localhost:8000/dashboard for demos.
This page reads recent observations from Langfuse when keys are configured.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Dispatch dashboard", layout="wide")

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

st.title("Dispatch // dashboard")
st.caption("Langfuse-backed view. For local demos use http://localhost:8000/dashboard")


@st.cache_data(ttl=60)
def load_events(limit: int = 100) -> pd.DataFrame:
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return pd.DataFrame()

    from langfuse import Langfuse

    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

    rows: list[dict] = []
    # Langfuse Python SDK v4 exposes REST under client.api
    try:
        response = client.api.observations.get_many(name="dispatch-routing", limit=limit)
        observations = getattr(response, "data", None) or []
        for obs in observations:
            meta = getattr(obs, "metadata", None) or {}
            if isinstance(meta, dict):
                rows.append(meta)
    except Exception:
        # Fall back quietly; local HTML dashboard still works.
        return pd.DataFrame()
    return pd.DataFrame(rows)


df = load_events()

if df.empty:
    st.info(
        "No Langfuse routing events yet (or API unavailable). "
        "Send a few /complete requests, then refresh. "
        "Local live metrics: http://localhost:8000/dashboard"
    )
else:
    col1, col2, col3, col4 = st.columns(4)
    if "actual_cost_usd" in df.columns:
        col1.metric("Total actual cost ($)", f"{df['actual_cost_usd'].fillna(0).sum():.4f}")
    else:
        col1.metric("Total actual cost ($)", "n/a")
    if "total_latency_ms" in df.columns:
        col2.metric("Avg latency (ms)", f"{df['total_latency_ms'].fillna(0).mean():.1f}")
    else:
        col2.metric("Avg latency (ms)", "n/a")
    if "cache_hit" in df.columns:
        col3.metric("Cache hit rate", f"{df['cache_hit'].fillna(False).mean() * 100:.1f}%")
    else:
        col3.metric("Cache hit rate", "n/a")
    col4.metric("Events", len(df))

    if "tier" in df.columns:
        st.subheader("Routing tier distribution")
        st.bar_chart(df["tier"].value_counts())

    st.subheader("Raw events")
    st.dataframe(df, use_container_width=True)
