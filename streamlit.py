"""
streamlit.py
Interactive Streamlit front-end for the Email Insight Engine.
Launch locally:
    streamlit run streamlit.py
Or deploy to Streamlit 
"""

import streamlit as st
import pandas as pd
import numpy as np
import semantic_search as ss

st.set_page_config(page_title="Email Insight Explorer", layout="wide")

# Cached data loaders (avoid re-encoding each refresh)
@st.cache_resource(show_spinner="Loading embeddings …")
def get_embeddings():
    return ss.load_embeddings()

@st.cache_resource(show_spinner="Loading email dataframe …")
def get_emails():
    return ss.load_emails()

embeddings = get_embeddings()
df_emails = get_emails()

# Sidebar controls
st.sidebar.header("Search Controls")
query = st.sidebar.text_input("Search phrase", value="invoice overdue")

top_k = st.sidebar.slider("Top results", min_value=5, max_value=50, value=20, step=5)

max_cluster_id = int(df_emails["cluster"].max()) if "cluster" in df_emails.columns else None
cluster_input = st.sidebar.number_input(
    "Cluster ID (optional)",
    min_value=0 if max_cluster_id is not None else 0,
    max_value=max_cluster_id if max_cluster_id is not None else 0,
    value=None,
    step=1,
    format="%d",
)
cluster_id = int(cluster_input) if (max_cluster_id is not None and cluster_input is not None) else None

run_search = st.sidebar.button("Run search")


# 3  Main content
st.title("Email Insight Explorer")
st.markdown(
    "This demo searches **≈10,000 Enron‑corpus emails** already embedded and clustered offline."
    "Type a phrase, choose the number of results, optionally filter by semantic cluster, and press **Run search** to surface the most relevant emails."
)

if run_search:
    with st.spinner("Computing similarities …"):
        # replicate search_api logic (no need to modify semantic_search.py)
        emb = embeddings
        df = df_emails
        if cluster_id is not None:
            mask = df["cluster"] == cluster_id
            emb = emb[mask]
            df = df[mask].reset_index(drop=True)
        q_vec = ss.encode_query(query)
        idx, sims = ss.search(q_vec, emb, top_k)
        results = df.iloc[idx].copy()
        results.insert(0, "similarity", sims.round(3))
    if results.empty:
        st.info("No results found. Try a different query or cluster.")
    else:
        st.subheader(f"Top {len(results)} results for '{query}'" + (f" in cluster {cluster_id}" if cluster_id is not None else ""))
        # Robust column selection: accept 'from_' or fallback to 'from'
        preferred_cols = [
            "similarity",
            "from_",   # default column name from helpers
            "from",    # fallback if CSV uses 'from'
            "to",
            "body",
        ]
        display_cols = [c for c in preferred_cols if c in results.columns]

        st.dataframe(
            results[display_cols],
            use_container_width=True,
            height=600,
        )
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            csv,
            file_name="email_search_results.csv",
            mime="text/csv",
        )
else:
    st.info("Enter a query in the sidebar and click **Run search** to begin.")
