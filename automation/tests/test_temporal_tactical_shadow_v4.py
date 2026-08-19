from __future__ import annotations

from automation.evaluate_temporal_tactical_shadow_v4 import summarize_shadow


def _row(geometry: str, nondefault: int) -> dict:
    return {
        "geometry": geometry,
        "frames": 100,
        "nondefault_predictions": nondefault,
        "abstention_reasons": {"OOD": 5},
        "exact_bt_mismatches": 0,
        "invalid_frames": 0,
        "throttle_violations": 0,
        "ownship_crash": False,
        "latency_over_166_7ms": 0,
    }


def test_shadow_gate_requires_multiple_geometry_actionability_and_exact_bt() -> None:
    summary = summarize_shadow([_row("left", 2), _row("right", 1)])
    assert summary["decision"] == "SHADOW_GATE_PASSED"
    assert summary["nondefault_geometry_count"] == 2


def test_shadow_nondefault_zero_is_gate_failure_not_promotion() -> None:
    summary = summarize_shadow([_row("left", 0), _row("right", 0)])
    assert summary["decision"] == "SHADOW_GATE_FAILED"
    assert summary["gate"]["nondefault_prediction_exists"] is False
