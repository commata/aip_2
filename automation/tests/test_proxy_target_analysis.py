from __future__ import annotations

import unittest

from automation.evaluate_proxy_targets import compare_frames, summarize_frames


def _frame(index: int, *, east_offset: float = 0.0, action: float = 0.0):
    return {
        "record_type": "frame",
        "sim_time_s": index / 60.0,
        "distance_m": 1000.0 - index,
        "ata_deg": 5.0,
        "target_ata_deg": 170.0,
        "target_damage": 0.1 if index >= 2 else 0.0,
        "target": {
            "position_ned_m": [float(index), east_offset, -5000.0],
            "altitude_m": 5000.0,
            "attitude_deg": [10.0, 2.0, 359.0 + index],
            "speed_kcas": 200.0,
            "health": 1.0,
        },
        "target_action": [action, 0.0, 0.0, 1.0],
    }


class ProxyTargetAnalysisTests(unittest.TestCase):
    def test_summary_unwraps_heading_and_finds_first_damage(self) -> None:
        summary = summarize_frames([_frame(index) for index in range(4)])

        self.assertEqual(summary["heading_delta_deg"], 3.0)
        self.assertEqual(summary["turn_direction"], "right")
        self.assertAlmostEqual(summary["first_target_damage_s"], 2.0 / 60.0)

    def test_pairwise_rmse_separates_position_and_action(self) -> None:
        left = [_frame(index) for index in range(4)]
        right = [_frame(index, east_offset=3.0, action=0.5) for index in range(4)]

        comparison = compare_frames(left, right)

        self.assertAlmostEqual(comparison["position_rmse_m"], 3.0)
        self.assertAlmostEqual(comparison["action_rmse"], 0.5)
        self.assertAlmostEqual(comparison["attitude_rmse_deg"], 0.0)


if __name__ == "__main__":
    unittest.main()
