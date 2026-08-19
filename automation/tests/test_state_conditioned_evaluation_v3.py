from __future__ import annotations

from automation.evaluate_state_conditioned_hybrid_v3 import evaluation_cases, summarize


def test_development_and_heldout_case_contracts_are_disjoint_and_balanced():
    development = evaluation_cases("development")
    heldout = evaluation_cases("heldout")
    assert len(development) == 60
    assert len(heldout) == 36
    assert {row["case_id"] for row in development}.isdisjoint(
        {row["case_id"] for row in heldout}
    )
    assert {row["geometry"] for row in development} == {
        "lateral_left", "lateral_right", "vertical_high", "vertical_low", "crossing_left", "crossing_right"
    }
    assert {row["opponent"] for row in development} == {"autopilot", "bt_0815"}


def _record(case_id: str, controller: str, margin: float, *, shadow=False):
    return {
        "case_id": case_id,
        "controller": controller,
        "opponent": "autopilot",
        "geometry": "lateral_left",
        "health_margin": margin,
        "target_crash": False,
        "ownship_crash": False,
        "invalid_actions": 0,
        "throttle_violations": 0,
        "predicted_nondefault_frames": 1 if shadow else 0,
        "nonzero_intervention_frames": 0 if shadow else 1,
        "e2e_latency_ms_p50": 1.0,
        "e2e_latency_ms_p95": 2.0,
        "e2e_latency_ms_p99": 3.0,
        "e2e_latency_ms_max": 4.0,
        "e2e_over_166_7ms": 0,
    }


def test_shadow_gate_requires_exact_bt_and_nondefault_predictions():
    records = [_record("a", "pure", 0.2), _record("a", "shadow", 0.2, shadow=True)]
    result = summarize(records, "shadow")
    assert result["gate_passed"]
    assert result["nonzero_intervention_frames"] == 0


def test_live_gate_uses_damage_median_positive_ratio_and_tail():
    records = []
    for index, delta in enumerate((0.01, 0.02, 0.03, 0.0, -0.001)):
        case_id = str(index)
        records.extend([_record(case_id, "pure", 0.1), _record(case_id, "hybrid", 0.1 + delta)])
    result = summarize(records, "micro")
    assert result["clean_damage_delta_mean"] > 0.0
    assert result["clean_damage_delta_median"] > 0.0
    assert result["positive_ratio"] == 0.6
    assert result["gate_passed"]
