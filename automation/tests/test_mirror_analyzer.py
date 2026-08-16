from __future__ import annotations

import unittest

from automation.analyze_mirror_symmetry import compare_frames


def frame(azimuth: float, action, *, residual=None):
    return {
        "sim_time_s": 0.0,
        "aim_azimuth_deg": azimuth,
        "aim_elevation_deg": 2.0,
        "los_azimuth_rate_deg_s": 3.0,
        "los_elevation_rate_deg_s": 4.0,
        "distance_m": 1000.0,
        "closing_rate_m_s": 10.0,
        "ata_deg": 5.0,
        "target_ata_deg": 170.0,
        "ownship_action": action,
        "hybrid": {
            "bt_action": action,
            "raw_residual_action": residual,
            "applied_rl_correction": residual,
        },
    }


class MirrorAnalyzerTests(unittest.TestCase):
    def test_exact_lateral_pair_has_zero_geometry_and_action_error(self) -> None:
        left = frame(-6.0, [0.2, -0.3, 0.1, 0.8], residual=[0.4, 0.2, -0.1, 0.0])
        right = frame(6.0, [-0.2, -0.3, -0.1, 0.8], residual=[-0.4, 0.2, 0.1, 0.0])
        right["los_azimuth_rate_deg_s"] = -3.0

        result = compare_frames(
            [left],
            [right],
            axis="lateral",
            action_signs=(-1.0, 1.0, -1.0, 1.0),
        )

        self.assertEqual(result["geometry"]["aim_azimuth_deg"]["rmse"], 0.0)
        self.assertEqual(result["actions"]["bt_action"]["roll"], 0.0)
        self.assertEqual(result["actions"]["raw_residual_action"]["yaw"], 0.0)


if __name__ == "__main__":
    unittest.main()
