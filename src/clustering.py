"""
KMeans fitting and rule-based cluster naming.

Naming logic: each business label is assigned by picking the cluster with
the most extreme mean on the feature that defines it (highest MonetaryValue
= VIP, highest Recency = Churned, etc.), working through segments in order
of least ambiguity first so earlier picks don't get reused.
"""

import pandas as pd
from sklearn.cluster import KMeans


def fit_kmeans(pca_data, n_clusters=5, random_state=42, max_iter=50, n_init=50):
    """Fit KMeans and return (fitted_model, cluster_labels_array)."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter, n_init=n_init)
    labels = km.fit_predict(pca_data)
    return km, labels


def assign_cluster_names(non_outliers_df: pd.DataFrame) -> dict:
    """
    Derive a {cluster_id: business_label} mapping from cluster-mean feature
    values. Requires a 'Cluster' column already assigned on the DataFrame.
    """
    cluster_means = non_outliers_df.groupby("Cluster")[
        ["MonetaryValue", "Frequency", "Recency", "AOV"]
    ].mean()

    vip_cluster = cluster_means["MonetaryValue"].idxmax()
    churned_cluster = cluster_means["Recency"].idxmax()

    remaining = cluster_means.drop([vip_cluster, churned_cluster])
    atrisk_hv_cluster = remaining["AOV"].idxmax()

    remaining2 = remaining.drop([atrisk_hv_cluster])
    atrisk_freq_cluster = remaining2["Recency"].idxmax()

    assigned = [vip_cluster, churned_cluster, atrisk_hv_cluster, atrisk_freq_cluster]
    promising_cluster = [c for c in cluster_means.index if c not in assigned][0]

    return {
        vip_cluster: "VIP",
        churned_cluster: "Churned",
        atrisk_hv_cluster: "At-Risk High-Value",
        atrisk_freq_cluster: "At-Risk Frequent",
        promising_cluster: "Promising",
    }
