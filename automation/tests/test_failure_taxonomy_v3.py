from __future__ import annotations

from automation.analyze_failure_taxonomy_v3 import FEATURE_INDEX, classify_failure


def observation(**values):
    vector = [0.0] * 42
    for name, value in values.items():
        vector[FEATURE_INDEX[name]] = value
    return vector


def test_default_optimal_has_priority_over_geometry_label():
    value = observation(any_surface_saturation=1.0, safety_margin_norm=-1.0)
    assert classify_failure(value, "crossing_left", 0.0) == "I_PURE_BT_ALREADY_OPTIMAL"


def test_server_safe_safety_and_surface_labels_are_deterministic():
    assert classify_failure(observation(safety_margin_norm=-0.5), "lateral_left", 0.01) == "J_ENERGY_ALTITUDE_SAFETY"
    assert classify_failure(observation(any_surface_saturation=1.0), "lateral_left", 0.01) == "G_SURFACE_AUTHORITY_LIMIT"


def test_crossing_fallback_label_uses_no_health_feature():
    value = observation(range_norm=-0.5, safety_margin_norm=1.0)
    assert classify_failure(value, "crossing_right", 0.01) == "H_CROSSING_LEAD_SHORTFALL"
