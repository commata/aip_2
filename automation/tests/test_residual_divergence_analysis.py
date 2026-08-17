from __future__ import annotations

import unittest

from automation.analyze_residual_divergence import build_comparison


def frame(index: int, *, hybrid: bool) -> dict:
    state_offset = 0.01 if hybrid and index >= 2 else 0.0
    los_offset = 0.02 if hybrid and index >= 2 else 0.0
    damage = 0.1 if hybrid and index >= 3 else 0.0
    action = [0.2, -0.1, 0.3, 0.8]
    record = {
        "record_type": "frame",
        "frame": index,
        "sim_time_s": index / 60.0,
        "distance_m": 800.0,
        "ata_deg": 0.5 + los_offset,
        "target_ata_deg": 150.0,
        "aim_azimuth_deg": 0.5 + los_offset,
        "aim_elevation_deg": 0.0,
        "los_azimuth_rate_deg_s": 0.1,
        "los_elevation_rate_deg_s": 0.0,
        "closing_rate_m_s": 20.0,
        "target_damage": damage,
        "in_wez": index >= 3 if hybrid else index >= 4,
        "ownship_action": action,
        "ownship": {
            "position_ned_m": [float(index) + state_offset, 0.0, -5000.0],
            "altitude_m": 5000.0,
            "attitude_deg": [0.0, state_offset, 0.0],
            "speed_kcas": 300.0,
        },
        "target": {
            "position_ned_m": [1000.0, 0.0, -5000.0],
            "altitude_m": 5000.0,
            "attitude_deg": [0.0, 0.0, 180.0],
            "speed_kcas": 300.0,
            "health": 1.0 - damage,
        },
    }
    if hybrid:
        correction = [0.0, 0.002, 0.0, 0.0] if index >= 1 else [0.0] * 4
        bt_action = list(action)
        if index >= 2:
            bt_action[2] += 0.001
        final = [bt_action[i] + correction[i] for i in range(4)]
        record["ownship_action"] = final
        record["hybrid"] = {
            "gate": {
                "active": index >= 1,
                "entry": index == 1,
                "exit": False,
            },
            "bt_action": bt_action,
            "raw_residual_action": [0.0, 0.016, 0.0, 0.0],
            "applied_rl_correction": correction,
            "final_action": final,
            "action_saturation": False,
        }
    return record


class ResidualDivergenceAnalysisTest(unittest.TestCase):
    def test_detects_expected_causal_order(self) -> None:
        rows, summary = build_comparison(
            [frame(index, hybrid=False) for index in range(1, 5)],
            [frame(index, hybrid=True) for index in range(1, 5)],
        )

        self.assertEqual(summary["first_residual_frame"], 1)
        self.assertEqual(summary["first_command_divergence_frame"], 1)
        self.assertEqual(summary["first_state_divergence_frame"], 2)
        self.assertEqual(summary["first_bt_command_divergence_frame"], 2)
        self.assertEqual(summary["first_LOS_divergence_frame"], 2)
        self.assertEqual(summary["first_cone_divergence_frame"], 3)
        self.assertEqual(summary["first_damage_divergence_frame"], 3)
        self.assertTrue(summary["causal_chain_monotonic_for_observed_events"])
        self.assertEqual(summary["gate"]["entry_count"], 1)
        self.assertEqual(summary["gate"]["max_active_duration_frames"], 4)
        self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
