from __future__ import annotations

import unittest

from automation.analyze_aim_gate_windows import analyze


def frame(time_s: float, los_deg: float, *, entry: bool = False, active: bool = False):
    return {
        "record_type": "frame",
        "sim_time_s": time_s,
        "ata_deg": los_deg,
        "target_ata_deg": 150.0,
        "distance_m": 800.0,
        "los_azimuth_rate_deg_s": 1.0,
        "los_elevation_rate_deg_s": 0.0,
        "target_damage_cumulative": time_s * 0.1,
        "ownship_damage_cumulative": 0.0,
        "ownship": {
            "speed_kcas": 200.0,
            "altitude_m": 5000.0,
        },
        "ownship_action": [0.1, 0.2, 0.3, 1.0],
        "hybrid": {
            "gate": {"entry": entry, "active": active},
            "applied_rl_correction": [0.01, 0.0, -0.01, 0.0],
            "surface_authority": {
                "requested_surface_correction": [0.02, 0.01, -0.02],
                "applied_surface_correction": [0.01, 0.0, -0.01],
                "positive_headroom": [0.9, 0.8, 0.7],
                "negative_headroom": [1.1, 1.2, 1.3],
                "directional_headroom": [0.9, 0.8, 1.3],
                "request_nonzero": [True, True, True],
                "bt_surface_saturated": [False, True, False],
                "final_surface_saturated": [False, True, True],
            },
        },
    }


class AimGateWindowTests(unittest.TestCase):
    def test_entry_window_uses_same_time_paired_delta(self) -> None:
        pure = [frame(0.0, 3.0), frame(1.0, 2.0), frame(2.0, 1.0)]
        hybrid = [
            frame(0.0, 3.0),
            frame(1.0, 1.5, entry=True, active=True),
            frame(2.0, 0.8, active=True),
        ]

        result = analyze(pure, hybrid, offsets_s=(0.0, 1.0))

        self.assertEqual(result["gate_entries"], 1)
        self.assertAlmostEqual(result["gate_active_ratio"], 2.0 / 3.0)
        points = result["entry_windows"][0]["points"]
        self.assertAlmostEqual(
            points[0]["delta_hybrid_minus_pure"]["los_deg"], -0.5
        )
        self.assertAlmostEqual(
            points[1]["delta_hybrid_minus_pure"]["los_deg"], -0.2
        )
        authority = result["surface_authority_summary"]["axes"]
        self.assertAlmostEqual(authority["roll"]["applied_to_requested_mean"], 0.5)
        self.assertAlmostEqual(authority["pitch"]["authority_blocked_ratio"], 1.0)
        self.assertAlmostEqual(authority["pitch"]["bt_saturation_ratio"], 1.0)
        self.assertAlmostEqual(authority["yaw"]["final_saturation_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
