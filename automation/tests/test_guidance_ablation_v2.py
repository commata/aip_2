from __future__ import annotations

from automation.evaluate_guidance_ablation_v2 import (
    build_near_event_cases,
    build_vertical_high_focus_cases,
    candidate_grid,
    summarize,
)


def record(case, candidate, margin, *, family="lateral_left", frames=6):
    return {
        "case_id": case,
        "seed": 1,
        "family": family,
        "distance_band": "near",
        "closing_band": "positive",
        "candidate_id": candidate,
        "action": "VP_AZ_POS_SMALL" if candidate not in {"PURE_BT", "BT_DEFAULT"} else "BT_DEFAULT",
        "magnitude_deg": 0.1,
        "duration_frames": frames,
        "damage_dealt": margin,
        "damage_received": 0.0,
        "health_margin": margin,
        "ownship_crash": False,
        "target_crash": False,
        "outcome": "timeout",
        "end_condition": "max time out",
        "episode_seconds": 2.0,
        "cone_entries": 1,
        "cone_time_s": 0.2,
        "mean_los_deg": 1.0,
        "los_rate_rms_deg_s": 0.5,
        "min_altitude_m": 4000.0,
        "min_speed_m_s": 200.0,
        "intervention_frames": 0 if candidate in {"PURE_BT", "BT_DEFAULT"} else frames,
        "throttle_violations": 0,
        "latency_ms_max": 0.1,
        "process_returncode": 0,
    }


def test_grid_covers_action_magnitude_duration_axes():
    grid = candidate_grid()
    assert len(grid) == 4 * 3 * 5
    assert len({row["candidate_id"] for row in grid}) == len(grid)
    assert {row["magnitude_deg"] for row in grid} == {0.10, 0.25, 0.50}
    assert {row["duration_frames"] for row in grid} == {6, 12, 18, 24, 36}


def test_near_event_suite_balances_families_and_observable_ranges():
    cases = build_near_event_cases(18)
    assert len(cases) == 18
    assert len({case["case_id"] for case in cases}) == 18
    families = {case["family"] for case in cases}
    assert len(families) == 6
    assert all(sum(case["family"] == family for case in cases) == 3 for family in families)
    distances = [case["scenario"]["env_config"]["target"][0] for case in cases]
    assert min(distances) == 680.0
    assert max(distances) == 820.0


def test_vertical_high_focus_suite_is_unique_and_bounded():
    cases = build_vertical_high_focus_cases(30)
    assert len({case["case_id"] for case in cases}) == 30
    assert {case["family"] for case in cases} == {"vertical_high"}
    distances = [case["scenario"]["env_config"]["target"][0] for case in cases]
    altitude_deltas = [
        -case["scenario"]["env_config"]["target"][2]
        + case["scenario"]["env_config"]["ownship"][2]
        for case in cases
    ]
    assert min(distances) == 650.0 and max(distances) == 850.0
    assert min(altitude_deltas) == 20.0 and max(altitude_deltas) == 95.0


def test_summary_requires_multi_family_positive_signal_and_exact_default():
    candidate = candidate_grid()[0]
    rows = []
    for index, family in enumerate(("lateral_left", "lateral_right", "vertical_high")):
        case = f"case_{index}"
        rows.extend(
            [
                record(case, "PURE_BT", 0.1, family=family, frames=0),
                record(case, "BT_DEFAULT", 0.1, family=family, frames=0),
                record(case, candidate["candidate_id"], 0.102, family=family),
            ]
        )
        rows[-1].update(candidate)
    aggregate, pairs = summarize(rows)
    assert aggregate["default_parity_passed"]
    assert aggregate["signal_candidates"] == [candidate["candidate_id"]]
    assert aggregate["candidates"][0]["dominant_positive_contribution"] <= 0.5
    assert len(pairs) == 3


def test_single_family_dominance_blocks_signal():
    candidate = candidate_grid()[0]
    rows = []
    for index, (family, delta) in enumerate((
        ("lateral_left", 0.010),
        ("lateral_right", 0.0001),
        ("vertical_high", 0.0001),
    )):
        case = f"case_{index}"
        rows.extend([
            record(case, "PURE_BT", 0.1, family=family, frames=0),
            record(case, "BT_DEFAULT", 0.1, family=family, frames=0),
            record(case, candidate["candidate_id"], 0.1 + delta, family=family),
        ])
        rows[-1].update(candidate)
    aggregate, _ = summarize(rows)
    assert aggregate["signal_candidates"] == []
    assert aggregate["candidates"][0]["dominant_positive_contribution"] > 0.5
