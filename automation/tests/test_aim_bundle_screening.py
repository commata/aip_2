from pathlib import Path

import pytest

from automation.screen_aim_bundles import discover_bundles, summarize_bundle


def _payload(
    damage: float,
    los: float,
    crash: int = 0,
    *,
    pure_target_crash: int = 0,
    hybrid_target_crash: int = 0,
):
    return {
        "preflight": {"bundle_weights_sha256": "ABC"},
        "summary": {
            "controllers": {
                "pure_0815": {
                    "ownship_crashes": 0,
                    "target_crashes": pure_target_crash,
                },
                "hybrid_0.125": {
                    "ownship_crashes": crash,
                    "target_crashes": hybrid_target_crash,
                    "gate_active_ratio": 0.5,
                    "final_roll_saturation_ratio": 0.1,
                    "final_pitch_saturation_ratio": 0.2,
                    "final_yaw_saturation_ratio": 0.3,
                },
            },
            "paired": {
                "hybrid_0.125": {
                    "delta_hybrid_minus_pure": {
                        "damage_dealt": damage,
                        "mean_los_deg": los,
                        "los_rate_rms_deg_s": 0.01,
                        "damage_cone_time_s": 0.02,
                        "time_to_first_damage_s": -0.03,
                        "min_altitude_m": 1.0,
                        "action_saturated_ratio": 0.04,
                    }
                }
            },
        },
    }


def test_discover_bundles_orders_periodic_and_appends_final():
    tmp_path = Path("artifacts/test_tmp/aim_bundle_screen_fixture/discovery")
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in ("bundle_000600", "bundle_000300"):
        bundle = tmp_path / name
        bundle.mkdir(exist_ok=True)
        (bundle / "metadata.json").write_text("{}", encoding="utf-8")
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    assert [tag for tag, _ in discover_bundles(tmp_path, True)] == ["000300", "000600", "final"]


def test_summarize_bundle_preserves_directional_deltas_and_crash_regression():
    tmp_path = Path("artifacts/test_tmp/aim_bundle_screen_fixture/summary")
    tmp_path.mkdir(parents=True, exist_ok=True)
    summary = summarize_bundle(
        "000300",
        tmp_path,
        {
            "left": _payload(0.1, -0.2),
            "right": _payload(
                -0.3,
                0.4,
                crash=1,
                pure_target_crash=1,
            ),
        },
        "hybrid_0.125",
    )
    aggregate = summary["aggregate"]
    assert aggregate["damage_delta_mean"] == pytest.approx(-0.1)
    assert aggregate["damage_delta_worst"] == -0.3
    assert aggregate["left_right_damage_gap"] == pytest.approx(0.4)
    assert aggregate["los_delta_worst_deg"] == 0.4
    assert aggregate["ownship_crash_regressions"] == 1
    assert aggregate["target_crash_pairs"] == 1
    assert aggregate["target_crash_asymmetries"] == 1
