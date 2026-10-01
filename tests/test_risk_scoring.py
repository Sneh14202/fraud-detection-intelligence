from src.risk_scoring import assign_risk_segment
from src.risk_scoring import assign_priority


def test_risk_segments():
    assert assign_risk_segment(0.005) == "Very Low"
    assert assign_risk_segment(0.03) == "Low"
    assert assign_risk_segment(0.07) == "Medium"
    assert assign_risk_segment(0.25) == "High"
    assert assign_risk_segment(0.75) == "Critical"


def test_priority_mapping():
    assert assign_priority("Very Low") == 5
    assert assign_priority("Low") == 4
    assert assign_priority("Medium") == 3
    assert assign_priority("High") == 2
    assert assign_priority("Critical") == 1


def test_boundary_values():
    assert assign_risk_segment(0.01) == "Low"
    assert assign_risk_segment(0.05) == "Medium"
    assert assign_risk_segment(0.10) == "High"
    assert assign_risk_segment(0.50) == "Critical"