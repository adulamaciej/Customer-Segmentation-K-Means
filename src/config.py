"""
Central configuration: file paths, constants, and display settings shared
across the pipeline.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "online_retail_II.xlsx"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

ARTIFACT_PATHS = {
    "power_transformer": ARTIFACTS_DIR / "power_transformer.pkl",
    "pca": ARTIFACTS_DIR / "pca.pkl",
    "kmeans": ARTIFACTS_DIR / "kmeans_k5.pkl",
    "reference_date": ARTIFACTS_DIR / "reference_date.pkl",
    "cluster_labels_names": ARTIFACTS_DIR / "cluster_labels_names.pkl",
    "iqr_bounds": ARTIFACTS_DIR / "iqr_bounds.pkl",
}

RANDOM_STATE = 42
N_CLUSTERS = 5
KMEANS_MAX_ITER = 50
KMEANS_N_INIT = 50
PCA_N_COMPONENTS = 3

RAW_RFM_COLUMNS = ["MonetaryValue", "Frequency", "Recency", "Tenure", "AOV"]
CLUSTERING_COLUMNS = ["MonetaryValue", "Frequency", "Recency", "AOV"]

STABILITY_SPLIT_DATE = "2010-12-01"

CLUSTER_COLORS = {
    "VIP": "#ff7f0e",
    "Churned": "#1f77b4",
    "At-Risk High-Value": "#2ca02c",
    "At-Risk Frequent": "#9467bd",
    "Promising": "#d62728",
    "VIP (Whale)": "#ff0000",
}
