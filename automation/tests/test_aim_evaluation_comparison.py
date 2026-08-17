from automation.compare_aim_evaluations import compare


def row(seed, controller, damage, *, pure_los=1.0):
    return {
        "seed": seed,
        "controller": controller,
        "run_id": f"seed{seed}_{controller}",
        "wall_seconds": 1.0,
        "variant_name": "left",
        "outcome": "timeout",
        "ownship_crash": False,
        "damage_dealt": damage,
        "mean_los_deg": pure_los,
        "los_rate_rms_deg_s": 1.0,
        "damage_cone_time_s": 1.0,
        "time_to_first_damage_s": 2.0,
        "min_altitude_m": 1000.0,
        "action_saturated_ratio": 0.1,
    }


def payload(pure_los=1.0, hybrid_damage=0.2):
    return {
        "records": [
            row(1, "pure_0815", 0.1, pure_los=pure_los),
            row(1, "hybrid_0.125", hybrid_damage, pure_los=0.8),
        ]
    }


def test_comparison_accepts_runtime_noise_but_detects_baseline_metric_drift():
    same = compare(
        [("a", payload(), None), ("b", payload(hybrid_damage=0.3), None)],
        controller="hybrid_0.125",
        bootstrap_samples=10,
        bootstrap_seed=1,
    )
    assert same["baseline_consistency"]["all_record_values_equal"] is True

    drift = compare(
        [("a", payload(), None), ("b", payload(pure_los=1.1), None)],
        controller="hybrid_0.125",
        bootstrap_samples=10,
        bootstrap_seed=1,
    )
    baseline = drift["baseline_consistency"]
    assert baseline["all_record_values_equal"] is False
    assert baseline["candidates"]["b"]["mismatches"] == [
        {"seed": 1, "fields": ["mean_los_deg"]}
    ]
