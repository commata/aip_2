from __future__ import annotations

import unittest

from automation.evaluate_guidance_development import summarize_candidates


def row(controller, case_id, margin, *, target_crash=False):
    return {
        "controller": controller,
        "case_id": case_id,
        "health_margin": margin,
        "target_crash": target_crash,
        "ownship_crash": False,
        "throttle_violation_steps": 0,
        "nonzero_intervention_frames": 1,
        "latency_ms_max": 0.1,
    }


class GuidanceDevelopmentTests(unittest.TestCase):
    def test_ppo_gate_requires_four_of_six_and_no_large_regression(self):
        records = []
        for index in range(6):
            case_id = f"case_{index}"
            records.append(row("pure", case_id, 0.0))
            records.append(row("seed_8701_c0.65", case_id, 0.001 if index < 4 else 0.0))
        summary = summarize_candidates(records)
        candidate = summary["candidate_summaries"]["seed_8701_c0.65"]
        self.assertTrue(candidate["ppo_gate_passed"])
        self.assertEqual(candidate["positive_pairs"], 4)

    def test_target_crash_is_excluded_not_promoted(self):
        records = []
        for index in range(6):
            case_id = f"case_{index}"
            records.append(row("pure", case_id, 0.0))
            records.append(
                row("seed_8701_c0.65", case_id, 1.0, target_crash=index == 0)
            )
        summary = summarize_candidates(records)
        candidate = summary["candidate_summaries"]["seed_8701_c0.65"]
        self.assertFalse(candidate["ppo_gate_passed"])
        self.assertEqual(candidate["clean_pairs"], 5)


if __name__ == "__main__":
    unittest.main()
