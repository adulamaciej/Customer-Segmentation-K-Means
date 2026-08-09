"""
Tests for KMeans fitting and the rule-based cluster naming logic.

The naming logic assigns business labels via sequential idxmax() calls on
cluster-mean features. These tests cover the expected/clean case and the
collision case where two labels would land on the same cluster -- the
specific failure mode that made this logic fragile.
"""

import numpy as np
import pandas as pd
import pytest

from src.clustering import fit_kmeans, assign_cluster_names


def _clean_cluster_means():
    return pd.DataFrame({
        "Cluster":       [0, 1, 2, 3, 4],
        "MonetaryValue": [5000, 200, 1200, 900, 600],
        "Frequency":     [12, 1, 2, 4, 3],
        "Recency":       [10, 400, 300, 250, 30],
        "AOV":           [400, 150, 700, 220, 200],
    })


def test_fit_kmeans_returns_requested_cluster_count():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(100, 3))
    model, labels = fit_kmeans(data, n_clusters=5, random_state=42)
    assert model.n_clusters == 5
    assert len(labels) == 100


def test_assign_cluster_names_unambiguous_case():
    means_df = _clean_cluster_means()
    names = assign_cluster_names(means_df)

    assert names[0] == "VIP"
    assert names[1] == "Churned"
    assert names[2] == "At-Risk High-Value"
    assert names[3] == "At-Risk Frequent"
    assert names[4] == "Promising"
    assert len(names) == 5
    assert len(set(names.values())) == 5


def test_assign_cluster_names_raises_on_vip_churned_collision():
    """Same cluster wins both highest MonetaryValue and highest Recency."""
    means_df = pd.DataFrame({
        "Cluster":       [0, 1, 2, 3, 4],
        "MonetaryValue": [5000, 200, 1200, 900, 600],
        "Frequency":     [1, 5, 2, 4, 3],
        "Recency":       [500, 50, 300, 250, 30],
        "AOV":           [5000, 150, 700, 220, 200],
    })

    with pytest.raises(ValueError, match="collision"):
        assign_cluster_names(means_df)