"""
Yeo-Johnson power transformation and PCA dimensionality reduction,
fit on non-outlier customer data.
"""

import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA


def fit_power_transformer(non_outliers_df: pd.DataFrame, columns) -> tuple:
    """
    Fit Yeo-Johnson on the given columns of non-outlier data.
    Returns (fitted_transformer, transformed_df).
    """
    pt = PowerTransformer(method="yeo-johnson")
    transformed = non_outliers_df[columns].copy()
    transformed[columns] = pt.fit_transform(non_outliers_df[columns])
    return pt, transformed


def fit_pca(transformed_df: pd.DataFrame, n_components: int = 3) -> tuple:
    """
    Fit PCA on transformed features.
    Returns (fitted_pca, pca_data_array).
    """
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(transformed_df)
    return pca, pca_data
