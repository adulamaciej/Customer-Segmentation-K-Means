"""
Data loading and cleaning for the Online Retail II dataset.
"""

import pandas as pd

VALID_INVOICE_RE = r"^\d{6}$"
VALID_STOCKCODE_RE = r"^\d{5}$|^\d{5}[a-zA-Z]+$"


def load_raw(data_path) -> pd.DataFrame:
    """Load all sheets of the Online Retail II workbook into one DataFrame."""
    sheets = pd.read_excel(data_path, sheet_name=None, dtype={"Customer ID": str})
    return pd.concat(sheets.values(), ignore_index=True)


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning sequence and return a new, cleaned DataFrame."""
    cleaned = df.copy()

    cleaned["Invoice"] = cleaned["Invoice"].astype(str)
    cleaned = cleaned[cleaned["Invoice"].str.match(VALID_INVOICE_RE)]

    cleaned["StockCode"] = cleaned["StockCode"].astype(str)
    cleaned = cleaned[cleaned["StockCode"].str.match(VALID_STOCKCODE_RE)]

    cleaned = cleaned[cleaned["Price"] > 0]
    cleaned = cleaned[cleaned["Quantity"] > 0]

    cleaned = cleaned.dropna(subset=["Customer ID"])
    cleaned = cleaned.drop_duplicates()

    return cleaned.reset_index(drop=True)
