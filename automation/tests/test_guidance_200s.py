from __future__ import annotations

import unittest

from automation.evaluate_guidance_200s import aggregate, full_fight_cases


def record(case_id, controller, margin, *, nonzero=0):
    return {
        "case_id": case_id,
        "controller": controller,
        "split": "held_out",
        "opponent": "autopilot",
        "side": "left",
        "health_margin": margin,
        "target_crash": False,
        "ownship_crash": False,
        "min_altitude_m": 500.0,
        "min_speed_m_s": 200.0,
        "invalid_or_nonfinite_actions": 0,
        "throttle_violation_steps": 0,
        "throttle_difference_max": 0.0,
        "latency_ms_p50": 0.1,
        "latency_ms_p95": 0.2,
        "latency_ms_p99": 0.3,
        "latency_ms_max": 0.4,
        "latency_over_166_7ms": 0,
        "nonzero_intervention_frames": nonzero,
        "gate_active_ratio": 0.1,
        "low_altitude_duration_s": 0.0,
        "low_speed_duration_s": 0.0,
        "process_returncode": 0,
        "episode_seconds": 200.0,
    }


class Guidance200sTests(unittest.TestCase):
    def test_matrix_has_12_independent_split_opponent_side_cases(self):
        cases = full_fight_cases()
        self.assertEqual(len(cases), 12)
        self.assertEqual({case["split"] for case in cases}, {"development", "held_out"})
        self.assertEqual({case["opponent"] for case in cases}, {"autopilot", "bt_0815", "bt_aip2"})
        self.assertEqual({case["side"] for case in cases}, {"left", "right"})

    def test_operational_gate_requires_nonzero_but_not_performance_gain(self):
        records = []
        for index in range(12):
            case_id = f"case_{index}"
            records.extend(
                [
                    record(case_id, "pure", 0.0),
                    record(case_id, "bt_default", 0.0),
                    record(case_id, "bc", -0.001, nonzero=1 if index == 0 else 0),
                ]
            )
        summary = aggregate(records)
        self.assertTrue(summary["operational_ready"])
        self.assertEqual(summary["promotion_status"], "NOT_PROMOTED")
        self.assertEqual(summary["status"], "SUBMISSION_READY_HYBRID_CANDIDATE")


if __name__ == "__main__":
    unittest.main()
