import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD 
from sklearn.preprocessing import normalize 

# email_clustering_pipeline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from classifier import extract_email_fields, top_features_by_cluster, plot_cluster_features

# Load raw email data
raw_emails = pd.read_csv('emails.csv')

# Parse relevant fields: from, to, body
email_data = pd.DataFrame(extract_email_fields(raw_emails.message))

# Dropping rows with missing or empty critical fields
email_data.drop(email_data.query("body == '' or to == '' or from_ == ''").index, inplace=True)

# Defining stop words
custom_stopwords = ENGLISH_STOP_WORDS.union({'ect', 'hou', 'com', 'recipient'})

# Vectorizing emails using TF-IDF
vectorizer = TfidfVectorizer(
    analyzer='word',
    stop_words=custom_stopwords,
    max_df=0.3,
    min_df=2
)
X_tfidf = vectorizer.fit_transform(email_data.body)
feature_names = vectorizer.get_feature_names_out()

# Cluster using KMeans
n_clusters = 3
kmeans = KMeans(
    n_clusters=n_clusters,
    init='k-means++',
    n_init=1,
    max_iter=100,
    random_state=42
)
labels = kmeans.fit_predict(X_tfidf)

# Reduce dimensionality for visualization using PCA
X_dense = X_tfidf.todense()
pca = PCA(n_components=2, random_state=42)
reduced_coords = pca.fit_transform(X_dense)

# Define color palette for clusters
color_palette = ["#2AB0E9", "#2BAF74", "#D7665E", "#CCCCCC", "#D2CA0D", "#522A64", "#A3DB05", "#FC6514"]
colors = [color_palette[label] for label in labels]

# Optional: visualize email clusters
# plt.scatter(reduced_coords[:, 0], reduced_coords[:, 1], c=colors, alpha=0.6)
# centroid_coords = pca.transform(kmeans.cluster_centers_)
# plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker='X', s=200, c='#444d60', edgecolors='k')
# plt.title('Email Clusters via PCA')
# plt.show()

# Display top TF-IDF features per cluster
cluster_top_terms = top_features_by_cluster(X_tfidf, labels, feature_names, min_tfidf=0.1, top_n=25)
plot_cluster_features(cluster_top_terms)