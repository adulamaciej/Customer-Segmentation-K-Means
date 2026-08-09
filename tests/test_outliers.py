"""
Tests for IQR-based outlier detection on a small, hand-built customer
table with one obvious extreme value planted in it.
"""

import pandas as pd
from src.outliers import compute_iqr_bounds, split_outliers


def test_extreme_value_flagged_as_outlier():
    df = pd.DataFrame({
        "MonetaryValue": [100, 110, 105, 95, 90, 100000],  # last one is an obvious outlier
        "Frequency": [2, 3, 2, 3, 2, 2],
        "AOV": [50, 55, 52, 48, 45, 50],
    })

    bounds = compute_iqr_bounds(df)
    non_outliers, outliers = split_outliers(df, bounds)

    assert len(outliers) == 1
    assert outliers.iloc[0]["MonetaryValue"] == 100000
    assert len(non_outliers) == 5


def test_no_outliers_when_data_is_uniform():
    df = pd.DataFrame({
        "MonetaryValue": [90, 95, 100, 105, 110, 115, 120],
        "Frequency": [2, 2, 3, 2, 3, 2, 3],
        "AOV": [45, 48, 50, 47, 52, 46, 51],
    })

    bounds = compute_iqr_bounds(df)
    non_outliers, outliers = split_outliers(df, bounds)

    assert len(outliers) == 0
    assert len(non_outliers) == 7