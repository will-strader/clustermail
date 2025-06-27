"""
semantic_search.py
Quick semantic search over clustered email corpus using Sentence-BERT embeddings.
Run:
    python semantic_search.py --query "invoice overdue" --top 20
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# Config – change paths here if needed
# ----------------------------------------------------------------------------
DATA_DIR = Path("data")
EMB_PATH = DATA_DIR / "email_embeddings.npy"
CSV_PATH = DATA_DIR / "emails_with_clusters.csv"
MODEL_NAME = "all-MiniLM-L6-v2"  # must match embedding file

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def load_embeddings() -> np.ndarray:
    if not EMB_PATH.exists():
        raise FileNotFoundError(f"Embeddings not found at {EMB_PATH}. Run bert_hdbscan.py first.")
    return np.load(EMB_PATH)


def load_emails() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Processed CSV not found at {CSV_PATH}. Run bert_hdbscan.py first.")
    return pd.read_csv(CSV_PATH)


def encode_query(query: str, model_name: str = MODEL_NAME) -> np.ndarray:
    model = SentenceTransformer(model_name)
    return model.encode([query])


def search(query_vec: np.ndarray, embeddings: np.ndarray, top_k: int = 20):
    sims = cosine_similarity(query_vec, embeddings).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    return top_idx, sims[top_idx]

# ----------------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Semantic email search via SBERT embeddings")
    parser.add_argument("--query", required=True, help="Search phrase or sentence")
    parser.add_argument("--top", type=int, default=20, help="Number of results to return")
    args = parser.parse_args()

    print("🔍 Loading data ...")
    embeddings = load_embeddings()
    df_emails = load_emails()

    print("🚀 Encoding query ...")
    q_vec = encode_query(args.query)

    print("📈 Computing cosine similarities ...")
    idx, scores = search(q_vec, embeddings, top_k=args.top)

    results = df_emails.iloc[idx].copy()
    results["similarity"] = scores

    # Display nicely in console
    for i, row in results.iterrows():
        print("\n" + "=" * 80)
        print(f"[Sim {row.similarity:.3f}]  From: {row.from_}  →  To: {row.to}")
        print("-" * 80)
        print(row.body[:500].replace("\n", " ") + (" ..." if len(row.body) > 500 else ""))

    # Save optional CSV for downstream use
    out_path = DATA_DIR / "search_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\n✅ Saved top {args.top} results to {out_path}")


if __name__ == "__main__":
    main()
