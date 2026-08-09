"""
Aggregates cleaned transaction-level data into one row per customer with
the five features used downstream: MonetaryValue, Frequency, Recency,
Tenure, AOV.
"""

import pandas as pd


def aggregate_customer_features(cleaned_df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    """
    Aggregate cleaned transaction rows to customer level.

    reference_date: optional fixed date for Recency. If None, uses the max
    InvoiceDate in the data (training-time behavior). Pass an explicit date
    at inference time.
    """
    df = cleaned_df.copy()
    df["SalesLineTotal"] = df["Quantity"] * df["Price"]

    agg = df.groupby("Customer ID", as_index=False).agg(
        MonetaryValue=("SalesLineTotal", "sum"),
        Frequency=("Invoice", "nunique"),
        LastInvoiceDate=("InvoiceDate", "max"),
        FirstInvoiceDate=("InvoiceDate", "min"),
    )

    ref_date = reference_date if reference_date is not None else agg["LastInvoiceDate"].max()

    agg["Recency"] = (ref_date - agg["LastInvoiceDate"]).dt.days
    agg["Tenure"] = (agg["LastInvoiceDate"] - agg["FirstInvoiceDate"]).dt.days
    agg["AOV"] = agg["MonetaryValue"] / agg["Frequency"]

    agg = agg.drop(columns=["LastInvoiceDate", "FirstInvoiceDate"])

    return agg
