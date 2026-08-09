"""
Tests for the Yeo-Johnson power transform and PCA fitting step, on a
small synthetic dataset with known skew and correlation structure.
"""

import numpy as np
import pandas as pd

from src.transform import fit_power_transformer, fit_pca


def _skewed_non_outliers_df(n=500, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "MonetaryValue": rng.exponential(scale=200, size=n),
        "Frequency":     rng.exponential(scale=3, size=n) + 1,
        "Recency":       rng.uniform(0, 400, size=n),
        "AOV":           rng.exponential(scale=50, size=n) + 1,
    })


def test_fit_power_transformer_reduces_skew():
    df = _skewed_non_outliers_df()
    skewed_columns = ["MonetaryValue", "Frequency", "AOV"]  # Recency is uniform, not skewed
    all_columns = ["MonetaryValue", "Frequency", "Recency", "AOV"]

    original_skew = df[skewed_columns].skew().abs()
    pt, transformed_df = fit_power_transformer(df, all_columns)
    transformed_skew = transformed_df[skewed_columns].skew().abs()

    for col in skewed_columns:
        assert transformed_skew[col] < original_skew[col]


def test_fit_power_transformer_preserves_shape_and_columns():
    df = _skewed_non_outliers_df(n=200)
    columns = ["MonetaryValue", "Frequency", "Recency", "AOV"]

    pt, transformed_df = fit_power_transformer(df, columns)

    assert transformed_df.shape == (200, len(columns))
    assert list(transformed_df.columns) == columns


def test_fit_pca_output_shape_matches_n_components():
    df = _skewed_non_outliers_df(n=300)
    columns = ["MonetaryValue", "Frequency", "Recency", "AOV"]
    _, transformed_df = fit_power_transformer(df, columns)

    pca, pca_data = fit_pca(transformed_df, n_components=3)

    assert pca_data.shape == (300, 3)
    assert pca.n_components_ == 3


def test_fit_pca_explained_variance_is_valid_and_ordered():
    df = _skewed_non_outliers_df(n=300)
    columns = ["MonetaryValue", "Frequency", "Recency", "AOV"]
    _, transformed_df = fit_power_transformer(df, columns)

    pca, _ = fit_pca(transformed_df, n_components=3)
    ratios = pca.explained_variance_ratio_

    assert (ratios >= 0).all()
    assert ratios.sum() <= 1.0 + 1e-9
    assert ratios[0] == max(ratios)