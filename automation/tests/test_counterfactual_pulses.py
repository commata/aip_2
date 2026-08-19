from __future__ import annotations

from automation.evaluate_counterfactual_pulses import (
    analyze_baseline,
    canonical_numbers,
)


def _record(controller: str, repeat: int, trajectory: str) -> dict:
    return {
        "geometry": "case",
        "controller": controller,
        "repeat": repeat,
        "trajectory_sha256": trajectory,
        "damage_dealt": 1.0,
        "damage_received": 0.0,
        "outcome": "win",
        "end_condition": "target destroyed",
        "episode_seconds": 10.0,
        "mean_los_deg": 1.0,
        "los_rate_rms_deg_s": 2.0,
        "damage_cone_time_s": 3.0,
        "time_to_first_damage_s": 4.0,
        "min_altitude_m": 1000.0,
    }


def test_canonical_numbers_treats_signed_zero_as_exact() -> None:
    assert canonical_numbers({"action": [-0.0, 0.0]}) == {
        "action": [0.0, 0.0]
    }


def test_baseline_freezes_zero_variance_thresholds() -> None:
    records = []
    for repeat in (1, 2, 3):
        records.append(_record("pure_0815", repeat, "same"))
        records.append(_record("zero", repeat, "same"))
    analysis = analyze_baseline(records)
    assert analysis["pure_exact_determinism"] is True
    assert analysis["zero_residual_exact_equality"] is True
    assert analysis["p95_abs_damage_deviation"] == 0.0
    assert analysis["minimum_meaningful_damage_delta"] == 0.001
    assert analysis["maximum_geometry_regression"] == -0.003
