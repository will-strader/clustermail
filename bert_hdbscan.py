"""
bert_hdbscan.py
Semantic email clustering using Sentence Transformers + UMAP + HDBSCAN
Part 2 of the email-insights pipeline.
"""

import numpy as np
import pandas as pd
import umap
import hdbscan
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import (
    extract_email_fields,
    top_features_by_cluster,
    plot_cluster_features,
)

# Data loading & parsing
def load_and_parse(path: str = "data/split_emails.csv") -> pd.DataFrame:
    """Read raw CSV and return a cleaned DataFrame with FROM / TO / BODY fields."""
    raw = pd.read_csv(path)
    emails = pd.DataFrame(extract_email_fields(raw.message))
    emails = emails.query("body != ''").reset_index(drop=True)
    return emails

# Sentence-BERT embeddings
def embed_texts(texts, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
    """Encode text into dense vectors using a pretrained SentenceTransformer."""
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)

# Clustering with UMAP to HDBSCAN
def cluster_embeddings(
    embeddings,
    *,
    min_cluster_size: int = 30,
    n_neighbors: int = 15,
    umap_dim: int = 50,
):
    """Return cluster labels plus fitted reducers / clusterer."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=umap_dim,
        metric="cosine",
        random_state=42,
    )
    emb_umap = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",  # UMAP puts us in euclidean space.
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(emb_umap)
    return labels, reducer, clusterer

# Viz helpers
def visualise_clusters(reducer2d, labels, palette=None):
    if palette is None:
        palette = np.array([
            "#CCCCCC",  # noise (label −1)
            "#2AB0E9",
            "#2BAF74",
            "#D7665E",
            "#D2CA0D",
            "#522A64",
            "#A3DB05",
            "#FC6514",
            "#FF9A54",
        ])
    colours = palette[(labels + 1) % len(palette)]  # −1 → 0 (gray)
    plt.figure(figsize=(8, 6))
    plt.scatter(reducer2d[:, 0], reducer2d[:, 1], s=5, c=colours, alpha=0.7)
    plt.title("SBERT + HDBSCAN Email Clusters")
    plt.axis("off")
    plt.show()


# Main: orchestrates pipeline
def main():
    emails = load_and_parse()
    print(f"Loaded {len(emails)} cleaned emails")

    embeddings = embed_texts(emails.body.tolist())

    labels, reducer50, clusterer = cluster_embeddings(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Identified {n_clusters} clusters (and {(labels == -1).sum()} noise points)")

    # 2-D projection for plotting
    reducer2d = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        metric="cosine",
        random_state=42,
    ).fit_transform(embeddings)

    visualise_clusters(reducer2d, labels)


    # Interpret clusters via TF-IDF keywords
    tfidf = TfidfVectorizer(stop_words="english", max_df=0.3, min_df=2)
    X_tfidf = tfidf.fit_transform(emails.body)
    feature_names = tfidf.get_feature_names_out()

    cluster_dfs = top_features_by_cluster(
        X_tfidf, labels, feature_names, min_score=0.1, top_n=20
    )
    plot_cluster_features(cluster_dfs)

    # Save artefacts
    emails["cluster"] = labels
    emails.to_csv("data/emails_with_clusters.csv", index=False)
    np.save("data/email_embeddings.npy", embeddings)
    print("✅ Saved clustered dataset and embeddings to /data\n")


if __name__ == "__main__":
    main()
