from __future__ import annotations

import pytest

from automation.collect_counterfactual_v3 import (
    build_adaptive_cases,
    build_evaluation_boundary_cases,
    coarse_candidates,
    verify_pure_baseline,
)


def test_adaptive_cases_are_unique_balanced_and_deterministic():
    first = build_adaptive_cases(120, start_index=200)
    second = build_adaptive_cases(120, start_index=200)
    assert first == second
    assert len({case["case_id"] for case in first}) == 120
    assert len({case["seed"] for case in first}) == 120
    families = {case["family"] for case in first}
    assert len(families) == 6
    assert all(sum(case["family"] == family for case in first) == 20 for family in families)
    assert all(not case["scenario"]["env_config"]["ownship_randomization"]["enabled"] for case in first)


def test_coarse_stage_covers_every_primary_axis_and_sign_once():
    candidates = coarse_candidates()
    assert len(candidates) == 4
    assert {candidate["action"] for candidate in candidates} == {
        "VP_AZ_POS_SMALL",
        "VP_AZ_NEG_SMALL",
        "VP_EL_POS_SMALL",
        "VP_EL_NEG_SMALL",
    }
    assert {candidate["magnitude_deg"] for candidate in candidates} == {0.25}
    assert {candidate["duration_frames"] for candidate in candidates} == {36}


def test_evaluation_boundary_cases_cover_clean_geometry_without_exact_state_reuse():
    cases = build_evaluation_boundary_cases(120, start_index=500)
    assert len(cases) == 120
    assert len({case["case_id"] for case in cases}) == 120
    assert {case["family"] for case in cases} == {
        "lateral_left",
        "lateral_right",
        "vertical_high",
        "vertical_low",
        "crossing_left",
        "crossing_right",
    }
    distances = [case["scenario"]["env_config"]["target"][0] for case in cases]
    assert min(distances) == 1005.0
    assert max(distances) == 1395.0
    assert all(case["distance_band"] == "evaluation_boundary" for case in cases)


def test_baseline_preflight_fails_before_rollout_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pure BT DLL"):
        verify_pure_baseline(tmp_path / "missing.dll", tmp_path / "missing.xml")
