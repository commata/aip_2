from __future__ import annotations

import json
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from automation.analyze_maneuvers import analyze_frames
from dogfight.ai.maneuver_telemetry_logger import ManeuverTelemetryLogger
from dogfight.sim.state_schema import StateIndex


def state(n: float, e: float, yaw: float, speed: float = 250.0) -> np.ndarray:
    value = np.zeros(51, dtype=np.float64)
    value[StateIndex.N] = n
    value[StateIndex.E] = e
    value[StateIndex.D] = -5000.0
    value[StateIndex.YAW] = yaw
    value[StateIndex.KCAS] = speed
    value[StateIndex.ALT] = 5000.0
    value[StateIndex.HEALTH] = 1.0
    return value


class ManeuverTelemetryTests(unittest.TestCase):
    def test_jsonl_logger_records_geometry_actions_and_gate(self):
        buffer = io.StringIO()
        with patch.object(Path, "mkdir"), patch.object(Path, "open", return_value=buffer):
            path = Path("ignored.jsonl")
            logger = ManeuverTelemetryLogger(path, sim_hz=60, flush_every=1)
            logger.start_episode(seed=7)
            logger.record(
                state(0.0, 0.0, 0.0),
                state(1000.0, 0.0, 180.0),
                [0.1, 0.2, 0.3, 0.7],
                [0.0, 0.0, 0.0, 0.8],
                {
                    "offensive_gate": {"active": True},
                    "native": np.array([1.0, 2.0]),
                    "bt_action": [1.0, -1.0, 0.25, 0.7],
                },
                target_damage=0.01,
                in_wez=True,
            )
            records = [json.loads(line) for line in buffer.getvalue().splitlines()]
            frames = [record for record in records if record.get("record_type") == "frame"]
            self.assertEqual(len(frames), 1)
            self.assertAlmostEqual(frames[0]["distance_m"], 1000.0)
            self.assertAlmostEqual(frames[0]["ata_deg"], 0.0)
            self.assertTrue(frames[0]["hybrid"]["offensive_gate"]["active"])
            self.assertEqual(frames[0]["hybrid"]["native"], [1.0, 2.0])
            self.assertEqual(records[0]["seed"], 7)
            summary = logger.summary()
            self.assertEqual(summary["damage_cone_entries"], 1)
            self.assertAlmostEqual(summary["damage_cone_time_s"], 1.0 / 60.0)
            self.assertEqual(summary["time_to_first_damage_s"], 0.0)
            self.assertEqual(summary["bt_roll_saturation_ratio"], 1.0)
            self.assertEqual(summary["bt_pitch_saturation_ratio"], 1.0)
            self.assertEqual(summary["final_roll_saturation_ratio"], 0.0)
            self.assertAlmostEqual(summary["bt_roll_positive_headroom_mean"], 0.0)
            self.assertAlmostEqual(summary["bt_pitch_negative_headroom_mean"], 0.0)
            self.assertAlmostEqual(summary["final_roll_positive_headroom_mean"], 0.9)

    def test_analyzer_detects_gate_chatter_saturation_and_missed_window(self):
        frames = []
        for index in range(20):
            active = index % 2 == 0
            frames.append(
                {
                    "sim_time_s": index / 60.0,
                    "distance_m": 1000.0,
                    "ata_deg": 20.0,
                    "target_ata_deg": 140.0,
                    "ownship": {
                        "attitude_deg": [50.0, 0.0, 0.0],
                        "speed_kcas": 250.0 - index,
                        "altitude_m": 5000.0,
                    },
                    "ownship_action": [1.0, 0.0, 0.0, 1.0],
                    "hybrid": {
                        "offensive_gate": {
                            "active": active,
                            "entry": active,
                            "exit": not active,
                        },
                        "applied_rl_correction": [0.1, 0.0, 0.0, -0.05],
                        "action_saturation": True,
                    },
                }
            )
        report = analyze_frames(frames)
        self.assertEqual(report["gate_entries"], 10)
        self.assertEqual(report["gate_exits"], 10)
        self.assertEqual(report["action_saturation_ratio"], 1.0)
        self.assertGreater(report["gate_transitions_per_min"], 8.0)
        self.assertTrue(any("hysteresis" in note for note in report["recommendations"]))


if __name__ == "__main__":
    unittest.main()
