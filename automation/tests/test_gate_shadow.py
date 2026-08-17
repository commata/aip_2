from __future__ import annotations

import unittest

from automation.analyze_gate_shadow import aggregate, analyze_frames


def _frame(index: int, *, lateral_m: float, damage: float = 0.0):
    return {
        "record_type": "frame",
        "sim_time_s": index / 60.0,
        "distance_m": (900.0**2 + lateral_m**2) ** 0.5,
        "ata_deg": 0.0,
        "target_ata_deg": 180.0,
        "closing_rate_m_s": 20.0,
        "target_damage": damage,
        "ownship": {
            "position_ned_m": [0.0, 0.0, -5000.0],
            "attitude_deg": [0.0, 0.0, 0.0],
            "speed_kcas": 200.0,
        },
        "target": {
            "position_ned_m": [900.0, lateral_m, -5000.0],
            "attitude_deg": [0.0, 0.0, 0.0],
            "speed_kcas": 200.0,
        },
        "ownship_action": [1.0, 0.0, -1.0, 1.0],
    }


class GateShadowTests(unittest.TestCase):
    def test_combined_gate_is_intersection_and_records_damage_window(self) -> None:
        frames = [
            _frame(0, lateral_m=500.0),
            _frame(1, lateral_m=0.0),
            _frame(2, lateral_m=0.0, damage=0.1),
        ]

        result = analyze_frames(frames)

        aim = result["gates"]["aim"]
        offensive = result["gates"]["offensive"]
        combined = result["gates"]["combined"]
        self.assertLessEqual(combined["active_steps"], aim["active_steps"])
        self.assertLessEqual(combined["active_steps"], offensive["active_steps"])
        self.assertEqual(combined["damage_events"], 1)
        self.assertEqual(combined["damage_events_with_activation_previous_3s"], 1)
        self.assertEqual(combined["bt_surface_saturation_ratio_axis"], [1.0, 0.0, 1.0])

    def test_aggregate_uses_step_weighted_active_ratio(self) -> None:
        first = analyze_frames([_frame(0, lateral_m=0.0)])
        second = analyze_frames([_frame(i, lateral_m=5000.0) for i in range(3)])

        summary = aggregate([first, second])

        self.assertAlmostEqual(summary["gates"]["aim"]["active_ratio"], 0.25)
        self.assertEqual(summary["episodes"], 2)


if __name__ == "__main__":
    unittest.main()
