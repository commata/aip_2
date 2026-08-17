from automation.plot_aim_trajectory import active_spans, series


def frame(time_s: float, *, active: bool, damage: float):
    return {
        "sim_time_s": time_s,
        "ata_deg": 2.0,
        "distance_m": 800.0,
        "los_azimuth_rate_deg_s": 3.0,
        "los_elevation_rate_deg_s": 4.0,
        "target_damage_cumulative": damage,
        "ownship": {"speed_kcas": 220.0, "altitude_m": 4000.0},
        "ownship_action": [0.1, 0.2, 0.3, 1.0],
        "hybrid": {
            "gate": {"active": active},
            "surface_authority": {
                "requested_surface_correction": [0.02, 0.03, 0.04],
                "applied_surface_correction": [0.01, 0.02, 0.0],
            },
        },
    }


def test_series_and_gate_spans_preserve_simulator_time_contract():
    frames = [
        frame(0.0, active=False, damage=0.0),
        frame(1.0, active=True, damage=0.1),
        frame(2.0, active=True, damage=0.2),
        frame(3.0, active=False, damage=0.3),
    ]
    values = series(frames)
    assert values["los_rate"] == [5.0] * 4
    assert values["damage"][-1] == 0.3
    assert values["requested_yaw"] == [0.04] * 4
    assert values["applied_yaw"] == [0.0] * 4
    assert active_spans(frames) == [(1.0, 2.0)]
