"""
End-to-end inference test using the same 7 synthetic customers from the
notebook's inference demo. Requires trained artifacts on disk (run the
notebook first) -- skipped automatically if they're not present.
"""

import pandas as pd
import pytest

from src.config import ARTIFACT_PATHS
from src.inference import predict_segment

ARTIFACTS_EXIST = all(p.exists() for p in ARTIFACT_PATHS.values())

pytestmark = pytest.mark.skipif(
    not ARTIFACTS_EXIST,
    reason="Trained artifacts not found -- run the notebook first to generate artifacts/*.pkl",
)


@pytest.fixture
def demo_customers():
    return pd.DataFrame([
        {"last_purchase": "2011-11-20", "first_purchase": "2011-01-01", "MonetaryValue": 3000.0, "Frequency": 10, "AOV": 400.0},   # expect VIP
        {"last_purchase": "2011-11-08", "first_purchase": "2011-06-01", "MonetaryValue": 521.0,  "Frequency": 3,  "AOV": 210.0},   # expect Promising
        {"last_purchase": "2011-07-01", "first_purchase": "2011-04-01", "MonetaryValue": 700.0,  "Frequency": 1,  "AOV": 600.0},   # expect At-Risk High-Value
        {"last_purchase": "2011-06-01", "first_purchase": "2011-05-01", "MonetaryValue": 150.0,  "Frequency": 1,  "AOV": 80.0},    # expect Churned
        {"last_purchase": "2011-04-01", "first_purchase": "2010-12-01", "MonetaryValue": 900.0,  "Frequency": 4,  "AOV": 225.0},   # expect At-Risk Frequent
    ])


def test_unambiguous_customers_land_in_expected_segments(demo_customers):
    expected = ["VIP", "Promising", "At-Risk High-Value", "Churned", "At-Risk Frequent"]
    result = predict_segment(demo_customers)
    assert list(result["Segment"]) == expected


def test_extreme_outlier_flagged_and_classified_as_vip():
    outlier = pd.DataFrame([{
        "last_purchase": "2011-11-20", "first_purchase": "2010-12-01",
        "MonetaryValue": 10000.0, "Frequency": 100, "AOV": 1000.0,
    }])

    result = predict_segment(outlier)

    assert result.iloc[0]["IsOutlier"] == True
    assert result.iloc[0]["Segment"] == "VIP"
    assert result.iloc[0]["IsWhale"] == True
