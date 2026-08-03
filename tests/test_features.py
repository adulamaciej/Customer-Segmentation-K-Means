"""
Tests for feature engineering on a small, hand-built transaction table
where the correct Recency/Frequency/MonetaryValue/AOV/Tenure values are
known in advance.
"""

import pandas as pd
from src.features import aggregate_customer_features


def test_aggregate_customer_features_basic():
    df = pd.DataFrame({
        "Customer ID": ["A", "A", "B"],
        "Invoice": ["1001", "1002", "1003"],
        "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-01-10", "2011-01-05"]),
        "Quantity": [2, 1, 3],
        "Price": [10.0, 5.0, 20.0],
    })

    result = aggregate_customer_features(df).set_index("Customer ID")

    # Customer A: 2 invoices, spend = 2*10 + 1*5 = 25, AOV = 25/2 = 12.5
    assert result.loc["A", "Frequency"] == 2
    assert result.loc["A", "MonetaryValue"] == 25.0
    assert result.loc["A", "AOV"] == 12.5
    assert result.loc["A", "Tenure"] == 9  # Jan 10 - Jan 1

    # Customer B: 1 invoice, spend = 3*20 = 60
    assert result.loc["B", "Frequency"] == 1
    assert result.loc["B", "MonetaryValue"] == 60.0
    assert result.loc["B", "Tenure"] == 0  # single purchase, first == last


def test_recency_uses_max_date_by_default():
    df = pd.DataFrame({
        "Customer ID": ["A", "B"],
        "Invoice": ["1001", "1002"],
        "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-01-10"]),
        "Quantity": [1, 1],
        "Price": [10.0, 10.0],
    })

    result = aggregate_customer_features(df).set_index("Customer ID")

    # Max date in the data is Jan 10 -> customer A is 9 days before it, B is 0
    assert result.loc["A", "Recency"] == 9
    assert result.loc["B", "Recency"] == 0


def test_recency_uses_explicit_reference_date():
    df = pd.DataFrame({
        "Customer ID": ["A"],
        "Invoice": ["1001"],
        "InvoiceDate": pd.to_datetime(["2011-01-01"]),
        "Quantity": [1],
        "Price": [10.0],
    })

    result = aggregate_customer_features(df, reference_date=pd.Timestamp("2011-01-11"))

    assert result.loc[0, "Recency"] == 10
