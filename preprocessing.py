import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_email(raw_email):
    """Extracts metadata and body content from a raw email string."""
    email_data = {}
    body_lines = []
    for line in raw_email.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            if key in ['from', 'to']:
                email_data[key] = value
        else:
            body_lines.append(line.strip())
    email_data['body'] = ' '.join(body_lines)
    return email_data

def extract_email_fields(email_list):
    """Parses a list of raw email messages into structured fields."""
    parsed_emails = [parse_email(email) for email in email_list]
    return {
        'from': [email.get('from', '') for email in parsed_emails],
        'to': [email.get('to', '') for email in parsed_emails],
        'body': [email.get('body', '') for email in parsed_emails]
    }

def get_top_tfidf_terms(tfidf_row, feature_names, top_n=20):
    """Returns top N TF-IDF features from a row vector."""
    top_indices = np.argsort(tfidf_row)[::-1][:top_n]
    top_features = [(feature_names[i], tfidf_row[i]) for i in top_indices]
    return pd.DataFrame(top_features, columns=['feature', 'score'])

def top_features_in_document(tfidf_matrix, feature_names, doc_index, top_n=25):
    """Returns top TF-IDF features in a given document."""
    doc_vector = np.squeeze(tfidf_matrix[doc_index].toarray())
    return get_top_tfidf_terms(doc_vector, feature_names, top_n)

def average_top_features(tfidf_matrix, feature_names, indices=None, min_score=0.1, top_n=25):
    """Computes top mean TF-IDF features for a group of documents."""
    matrix = tfidf_matrix[indices].toarray() if indices is not None else tfidf_matrix.toarray()
    matrix[matrix < min_score] = 0
    mean_scores = np.mean(matrix, axis=0)
    return get_top_tfidf_terms(mean_scores, feature_names, top_n)

def top_features_by_cluster(tfidf_matrix, labels, feature_names, min_score=0.1, top_n=25):
    """Gets top TF-IDF features for each cluster."""
    results = []
    for label in np.unique(labels):
        doc_indices = np.where(labels == label)[0]
        cluster_features = average_top_features(tfidf_matrix, feature_names, indices=doc_indices, min_score=min_score, top_n=top_n)
        cluster_features['cluster'] = label
        results.append(cluster_features)
    return results

def plot_cluster_features(cluster_feature_dfs):
    """Plots top TF-IDF features for each cluster as horizontal bar charts."""
    num_clusters = len(cluster_feature_dfs)
    fig, axes = plt.subplots(1, num_clusters, figsize=(13, 6), constrained_layout=True)
    if num_clusters == 1:
        axes = [axes]
    
    for i, df in enumerate(cluster_feature_dfs):
        ax = axes[i]
        y_positions = np.arange(len(df))
        ax.barh(y_positions, df['score'], color='#7530FF')
        ax.set_yticks(y_positions)
        ax.set_yticklabels(df['feature'])
        ax.invert_yaxis()
        ax.set_title(f"Cluster {df['cluster'].iloc[0]}")
        ax.set_xlabel("TF-IDF Score")
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
    plt.show()
    