"""
Tests for the cleaning pipeline on a small, hand-built transaction table
covering each filtering rule individually.
"""

import pandas as pd
from src.cleaning import clean_pipeline


def _base_row(**overrides):
    row = {
        "Invoice": "536365",
        "StockCode": "85123A",
        "Quantity": 1,
        "Price": 10.0,
        "Customer ID": "1001",
    }
    row.update(overrides)
    return row


def test_drops_cancelled_invoices():
    df = pd.DataFrame([
        _base_row(),
        _base_row(Invoice="C536365"),  # cancellation prefix
    ])
    result = clean_pipeline(df)
    assert len(result) == 1
    assert result.iloc[0]["Invoice"] == "536365"


def test_drops_invalid_stockcode():
    df = pd.DataFrame([
        _base_row(),
        _base_row(StockCode="POST"),  # not a valid 5-digit code
    ])
    result = clean_pipeline(df)
    assert len(result) == 1
    assert result.iloc[0]["StockCode"] == "85123A"


def test_drops_non_positive_price_and_quantity():
    df = pd.DataFrame([
        _base_row(),
        _base_row(Price=0.0),
        _base_row(Quantity=-1),
    ])
    result = clean_pipeline(df)
    assert len(result) == 1


def test_drops_missing_customer_id():
    df = pd.DataFrame([
        _base_row(),
        _base_row(**{"Customer ID": None}),
    ])
    result = clean_pipeline(df)
    assert len(result) == 1
    assert result.iloc[0]["Customer ID"] == "1001"


def test_drops_exact_duplicates():
    df = pd.DataFrame([_base_row(), _base_row()])
    result = clean_pipeline(df)
    assert len(result) == 1
