"""
Illustrative business impact estimates: win-back revenue potential
(At-Risk, Churned segments) and upsell revenue potential (Promising -> VIP).

All rates are industry benchmarks, not fitted to this dataset -- see
README for sourcing and caveats.
"""


def estimate_recovery_scenarios(recovery_scenarios: dict) -> dict:
    """
    recovery_scenarios: {segment_name: {"customers": int, "aov": float, "rates": [low, mid, high]}}
    Returns {segment_name: {"low": ..., "mid": ..., "high": ...}}
    """
    results = {}
    for segment, d in recovery_scenarios.items():
        low, mid, high = [d["customers"] * r * d["aov"] for r in d["rates"]]
        results[segment] = {"low": low, "mid": mid, "high": high}
    return results


def estimate_upsell_scenarios(upsell_scenarios: dict, spend_delta: float) -> dict:
    """
    upsell_scenarios: {segment_name: {"customers": int, "rates": [low, mid, high]}}
    spend_delta: target avg spend minus current avg spend for the migrating segment.
    Returns {segment_name: {"low": ..., "mid": ..., "high": ...}}
    """
    results = {}
    for segment, d in upsell_scenarios.items():
        low, mid, high = [d["customers"] * r * spend_delta for r in d["rates"]]
        results[segment] = {"low": low, "mid": mid, "high": high}
    return results
