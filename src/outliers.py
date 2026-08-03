"""
IQR-based outlier detection on MonetaryValue, Frequency, and AOV.
"""

import pandas as pd


def compute_iqr_bounds(df: pd.DataFrame, columns=("MonetaryValue", "Frequency", "AOV")) -> dict:
    bounds = {}
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        bounds[f"{col}_Q1"] = q1
        bounds[f"{col}_Q3"] = q3
        bounds[f"{col}_IQR"] = q3 - q1
    return bounds


def is_outlier_row(row: pd.Series, bounds: dict, columns=("MonetaryValue", "Frequency", "AOV")) -> bool:
    for col in columns:
        upper = bounds[f"{col}_Q3"] + 1.5 * bounds[f"{col}_IQR"]
        if row[col] > upper:
            return True
    return False


def split_outliers(df: pd.DataFrame, bounds: dict, columns=("MonetaryValue", "Frequency", "AOV")):
    is_outlier = df.apply(lambda row: is_outlier_row(row, bounds, columns), axis=1)
    return df[~is_outlier].copy(), df[is_outlier].copy()
