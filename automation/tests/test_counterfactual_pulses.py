from __future__ import annotations

from automation.evaluate_counterfactual_pulses import (
    analyze_baseline,
    analyze_pulses,
    canonical_numbers,
    prepare_output,
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


GEOMETRIES = (
    "lateral_left", "lateral_right", "crossing_left", "crossing_right",
    "vertical_high", "vertical_low",
)
BEST = {
    "lateral_left": "yaw_pos", "lateral_right": "yaw_neg",
    "crossing_left": "roll_pos", "crossing_right": "roll_neg",
    "vertical_high": "pitch_pos", "vertical_low": "pitch_neg",
}
PULSES = ("roll_pos", "roll_neg", "pitch_pos", "pitch_neg", "yaw_pos", "yaw_neg")
THRESHOLDS = {
    "minimum_meaningful_damage_delta": 0.001,
    "maximum_geometry_regression": -0.003,
}


def _pulse_record(geometry: str, controller: str, delta: float = 0.0) -> dict:
    return {
        "state_id": geometry,
        "geometry": geometry,
        "canonical_geometry": geometry.split("_")[0],
        "state_neighborhood": {"shot_window_elapsed_frames": 0},
        "shot_window_elapsed_frames": 0,
        "controller": controller,
        "seed": 7101,
        "trajectory_sha256": "pure" if controller in {"pure_0815", "zero"} else controller,
        "damage_dealt": 1.0 + delta,
        "damage_received": 0.0,
        "time_to_first_damage_s": 1.0,
        "mean_los_deg": 1.0,
        "damage_cone_time_s": 1.0,
        "rl_correction_steps": 6,
        "rl_inference_calls": 0,
        "ownship_crash": False,
        "target_crash": False,
        "returncode": 0,
        "outcome": "win",
        "invalid_or_nonfinite_actions": 0,
        "throttle_violations": 0,
    }


def _signal_records(*, positives_per_state: int = 5) -> list[dict]:
    rows = []
    for geometry in GEOMETRIES:
        rows.extend((_pulse_record(geometry, "pure_0815"), _pulse_record(geometry, "zero")))
        ordered = (BEST[geometry],) + tuple(p for p in PULSES if p != BEST[geometry])
        for index, pulse in enumerate(ordered):
            delta = 0.002 if index == 0 else (0.0002 if index < positives_per_state else -0.0002)
            rows.append(_pulse_record(geometry, pulse, delta))
    return rows


def test_dataset_gate_passes_only_with_full_clean_signal() -> None:
    analysis = analyze_pulses(_signal_records(), THRESHOLDS)
    assert analysis["status"] == "COUNTERFACTUAL_SIGNAL_SUFFICIENT"
    assert analysis["pooled_clean_positive_ratio"] >= 2 / 3
    assert analysis["canonical_mirror_consistent"] is True
    assert analysis["raw_aggregate_recompute_match"] is True


def test_pooled_positive_ratio_below_two_thirds_fails() -> None:
    analysis = analyze_pulses(_signal_records(positives_per_state=2), THRESHOLDS)
    assert analysis["status"] == "COUNTERFACTUAL_SIGNAL_INSUFFICIENT"
    assert analysis["pooled_clean_positive_ratio"] < 2 / 3


def test_any_large_clean_regression_fails() -> None:
    rows = _signal_records()
    candidate = next(row for row in rows if row["controller"] == "roll_neg")
    candidate["damage_dealt"] = 0.996
    analysis = analyze_pulses(rows, THRESHOLDS)
    assert analysis["status"] == "COUNTERFACTUAL_SIGNAL_INSUFFICIENT"
    assert analysis["geometry_regressions"]


def test_contamination_process_error_and_none_are_never_success() -> None:
    rows = _signal_records()
    rows[2]["target_crash"] = True
    rows[3]["returncode"] = 1
    rows[3]["outcome"] = "process_error"
    rows[4]["damage_dealt"] = None
    analysis = analyze_pulses(rows, THRESHOLDS)
    assert analysis["status"] == "COUNTERFACTUAL_SIGNAL_INSUFFICIENT"
    assert analysis["process_errors"] == 1
    assert analysis["raw_recomputed"]["target_crash_contaminated"] == 1


def test_resume_requires_exact_fingerprint(tmp_path) -> None:
    output = tmp_path / "run"
    fingerprint = {
        "git_sha": "abc",
        "ownship_bt_dll": {"sha256": "dll"},
        "bt_rule_xml": {"sha256": "xml"},
        "suite": {"sha256": "scenario"},
        "residual_scale": 0.125,
        "pulse_frames": 6,
    }
    prepare_output(output, fingerprint, resume=False)
    prepare_output(output, fingerprint, resume=True)
    mismatches = (
        {**fingerprint, "git_sha": "def"},
        {**fingerprint, "ownship_bt_dll": {"sha256": "other"}},
        {**fingerprint, "bt_rule_xml": {"sha256": "other"}},
        {**fingerprint, "suite": {"sha256": "other"}},
        {**fingerprint, "residual_scale": 0.15},
        {**fingerprint, "pulse_frames": 12},
    )
    for mismatch in mismatches:
        try:
            prepare_output(output, mismatch, resume=True)
        except ValueError as error:
            assert "fingerprint mismatch" in str(error)
        else:
            raise AssertionError("mismatched resume fingerprint was accepted")
