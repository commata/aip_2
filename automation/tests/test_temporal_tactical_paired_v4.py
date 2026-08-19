from __future__ import annotations

from automation.evaluate_temporal_tactical_paired_v4 import expand_cases, summarize_pairs


def _row(delta: float, *, intervention: int = 1) -> dict:
    return {
        "clean": True,
        "damage_delta": delta,
        "nondefault_predictions": intervention,
        "hybrid_ownship_crash": False,
        "pure_ownship_crash": False,
        "invalid_frames": 0,
        "throttle_violations": 0,
        "latency_over_166_7ms": 0,
        "opponent": "autopilot",
        "geometry": "g",
        "phase1_frames": 1,
        "phase2_frames": 0,
        "phase3_frames": 0,
    }


def test_micro_rejects_negative_tail_even_when_mean_is_positive() -> None:
    rows = [_row(0.01) for _ in range(11)] + [_row(-0.001)]
    summary = summarize_pairs(rows, stage="MICRO", minimum_clean_pairs=12)
    assert summary["decision"] == "MICRO_GATE_FAILED"
    assert summary["gate"]["no_meaningful_negative_tail"] is False


def test_development_requires_positive_median_and_pair_ratio() -> None:
    rows = [_row(0.01) for _ in range(7)] + [_row(-0.001) for _ in range(3)]
    summary = summarize_pairs(rows, stage="OFFICIAL_DEVELOPMENT", minimum_clean_pairs=10)
    assert summary["gate"]["damage_median_positive"] is True
    assert summary["gate"]["positive_pair_ratio_at_least_60pct"] is True
    assert summary["gate"]["opponent_coverage_at_least_3"] is False
    assert summary["gate"]["phase1_2_3_flight_coverage"] is False


def test_opponent_expansion_creates_distinct_paired_cases() -> None:
    cases = [
        {"case_id": "c1", "geometry": "left"},
        {"case_id": "c2", "geometry": "right"},
    ]
    opponents = [
        {"id": "autopilot", "backend": "autopilot"},
        {"id": "pastel", "backend": "bt", "dll": "pastel.dll"},
    ]
    expanded = expand_cases(cases, opponents)
    assert len(expanded) == 4
    assert {(case["case_id"], opponent["id"]) for case, opponent in expanded} == {
        ("c1", "autopilot"),
        ("c2", "autopilot"),
        ("c1", "pastel"),
        ("c2", "pastel"),
    }


def test_heldout_uses_final_paired_damage_gates() -> None:
    rows = [_row(0.01) for _ in range(7)] + [_row(-0.001) for _ in range(3)]
    summary = summarize_pairs(rows, stage="HELD_OUT", minimum_clean_pairs=10)
    assert summary["gate"]["damage_mean_positive"] is True
    assert summary["gate"]["damage_median_positive"] is True
    assert summary["gate"]["positive_pair_ratio_at_least_60pct"] is True
    assert summary["decision"] == "HELD_OUT_GATE_PASSED"
