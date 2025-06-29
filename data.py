import pandas as pd
import numpy as np

df = pd.read_csv("data/emails_with_clusters.csv")

labels = df["cluster"].to_numpy()

n_noise     = np.sum(labels == -1)
n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)

print(f"Total emails  : {len(labels):,}")
print(f"Noise points  : {n_noise:,}")
print(f"Cluster count : {n_clusters}")

# Show each cluster label and how many emails it contains
counts = df["cluster"].value_counts().sort_index()
print(counts.head(20))          # first 20 labels