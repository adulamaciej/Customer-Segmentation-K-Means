"""
End-to-end inference: given raw customer inputs (dates + spend/frequency
figures), predict the business segment using the artifacts fitted at
training time. No retraining happens here.
"""

import joblib
import pandas as pd

from .config import ARTIFACT_PATHS, RAW_RFM_COLUMNS, CLUSTERING_COLUMNS


def load_artifacts() -> dict:
    """Load all fitted pipeline artifacts from disk."""
    return {name: joblib.load(path) for name, path in ARTIFACT_PATHS.items()}


def is_outlier(row: pd.Series, iqr_bounds: dict) -> bool:
    return (
        row["MonetaryValue"] > iqr_bounds["MonetaryValue_Q3"] + 1.5 * iqr_bounds["MonetaryValue_IQR"]
        or row["Frequency"] > iqr_bounds["Frequency_Q3"] + 1.5 * iqr_bounds["Frequency_IQR"]
        or row["AOV"] > iqr_bounds["AOV_Q3"] + 1.5 * iqr_bounds["AOV_IQR"]
    )


def predict_segment(customers_raw: pd.DataFrame, artifacts: dict = None) -> pd.DataFrame:
    """
    customers_raw: DataFrame with columns
        last_purchase, first_purchase, MonetaryValue, Frequency, AOV
    (dates as datetime or parseable strings)

    Returns customers_raw with added columns: Cluster, Segment, IsOutlier, IsWhale.
    """
    if artifacts is None:
        artifacts = load_artifacts()

    pt = artifacts["power_transformer"]
    pca = artifacts["pca"]
    kmeans = artifacts["kmeans"]
    iqr_bounds = artifacts["iqr_bounds"]
    reference_date = artifacts["reference_date"]
    cluster_labels_names = artifacts["cluster_labels_names"]

    df = customers_raw.copy()
    df["last_purchase"] = pd.to_datetime(df["last_purchase"])
    df["first_purchase"] = pd.to_datetime(df["first_purchase"])
    df["Recency"] = (reference_date - df["last_purchase"]).dt.days
    df["Tenure"] = (df["last_purchase"] - df["first_purchase"]).dt.days

    features = df[RAW_RFM_COLUMNS]

    df["IsOutlier"] = df.apply(lambda r: is_outlier(r, iqr_bounds), axis=1)

    transformed = pt.transform(features)
    transformed_df = pd.DataFrame(transformed, columns=RAW_RFM_COLUMNS)[CLUSTERING_COLUMNS]

    pca_input = pca.transform(transformed_df)
    clusters = kmeans.predict(pca_input)

    df["Cluster"] = clusters
    df["Segment"] = df["Cluster"].map(cluster_labels_names)
    df["IsWhale"] = df["IsOutlier"] & (df["Segment"] == "VIP")

    return df
