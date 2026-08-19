from __future__ import annotations

import unittest

from automation.evaluate_guidance_counterfactual import build_cases, summarize
from dogfight.ai.guidance_selector import GUIDANCE_ACTIONS


class GuidanceCounterfactualTests(unittest.TestCase):
    def test_frozen_suite_has_100_actually_distinct_states(self):
        cases = build_cases(100)
        self.assertEqual(len(cases), 100)
        states = {
            (
                tuple(case["scenario"]["env_config"]["ownship"]),
                tuple(case["scenario"]["env_config"]["target"]),
            )
            for case in cases
        }
        self.assertEqual(len(states), 100)
        self.assertEqual({case["seed"] for case in cases}, set(range(8601, 8701)))
        self.assertEqual(
            {case["family"] for case in cases},
            {
                "lateral_left",
                "lateral_right",
                "vertical_high",
                "vertical_low",
                "crossing_left",
                "crossing_right",
            },
        )

    def test_summary_labels_only_meaningful_clean_actions(self):
        case = build_cases(1)[0]
        records = []
        for action in GUIDANCE_ACTIONS:
            records.append(
                {
                    "case_id": case["case_id"],
                    "seed": case["seed"],
                    "family": case["family"],
                    "action": action,
                    "health_margin": 0.002 if action == "VP_EL_POS_SMALL" else 0.0,
                    "ownship_crash": False,
                    "target_crash": False,
                    "nonzero_intervention_frames": 1 if action != "BT_DEFAULT" else 0,
                    "first_selector_snapshot": {"observation": [0.0] * 45},
                    "process_returncode": 0,
                    "throttle_violation_steps": 0,
                    "latency_ms_max": 0.1,
                }
            )
        aggregate, dataset = summarize([case], records)
        self.assertEqual(aggregate["rollouts"], 9)
        self.assertEqual(dataset[0]["label"], "VP_EL_POS_SMALL")


if __name__ == "__main__":
    unittest.main()

