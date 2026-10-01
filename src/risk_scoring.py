# ============================================================
# RISK SEGMENTATION
# ============================================================

def assign_risk_segment(probability):
    """
    Assign a risk segment based on fraud probability.

    Risk thresholds are the operational thresholds
    defined for this portfolio project.
    """

    if probability < 0.01:
        return "Very Low"

    elif probability < 0.05:
        return "Low"

    elif probability < 0.10:
        return "Medium"

    elif probability < 0.50:
        return "High"

    else:
        return "Critical"


# ============================================================
# INVESTIGATION PRIORITY
# ============================================================

def assign_priority(risk_segment):
    """
    Convert risk segment into investigation priority.

    Priority 1 = highest investigation priority.
    Priority 5 = lowest investigation priority.
    """

    priority_map = {
        "Very Low": 5,
        "Low": 4,
        "Medium": 3,
        "High": 2,
        "Critical": 1
    }

    if risk_segment not in priority_map:
        raise ValueError(
            f"Unknown risk segment: {risk_segment}"
        )

    return priority_map[risk_segment]