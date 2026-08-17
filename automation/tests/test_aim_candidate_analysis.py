import pytest

from automation.analyze_aim_candidates import analyze


def row(seed, controller, damage, los, *, variant="left", crash=False):
    return {
        "seed": seed,
        "controller": controller,
        "variant_name": variant,
        "run_id": f"{seed}_{controller}",
        "outcome": "timeout",
        "ownship_crash": crash,
        "damage_dealt": damage,
        "mean_los_deg": los,
        "los_rate_rms_deg_s": los,
        "damage_cone_time_s": damage,
        "time_to_first_damage_s": 2.0,
        "min_altitude_m": 1000.0,
        "action_saturated_ratio": 0.1,
        "gate_active_ratio": 0.5,
        "roll_applied_to_requested_ratio": 0.4,
        "final_roll_saturation_ratio": 0.6,
    }


def test_analysis_preserves_paired_deltas_variants_and_representatives():
    payload = {
        "records": [
            row(1, "pure_0815", 0.1, 2.0),
            row(1, "hybrid_0.125", 0.3, 1.5),
            row(2, "pure_0815", 0.4, 1.0, variant="right"),
            row(2, "hybrid_0.125", 0.3, 1.2, variant="right", crash=True),
        ]
    }
    result = analyze(payload, "hybrid_0.125", bootstrap_samples=100, bootstrap_seed=7)

    assert result["pairs"] == 2
    assert result["crash_regressions"] == 1
    assert result["metrics"]["damage_dealt"]["mean"] == pytest.approx(0.05)
    assert result["per_variant"]["left"]["metrics"]["mean_los_deg"]["mean"] == -0.5
    assert result["authority"]["roll_applied_to_requested_ratio"] == 0.4
    assert result["representatives"]["worst_damage"]["seed"] == 2
    assert result["representatives"]["best_damage"]["seed"] == 1
