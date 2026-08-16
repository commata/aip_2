from __future__ import annotations

import unittest

from automation.analyze_longrun_training import analyze_rows


class LongrunTrainingAnalysisTests(unittest.TestCase):
    def test_episode_deltas_reconstruct_profile_and_variant_counts(self) -> None:
        rows = [
            {
                "iter": "0", "sampled_steps": "128", "learner_steps": "n/a",
                "episodes": "0", "effective_learner_time_s": "0",
                "reward_mean": "nan", "crash_rate": "nan",
                "target_profile_fraction_auto": "", "target_profile_fraction_bt": "",
                "aim_variant_fraction_left": "", "aim_variant_fraction_right": "",
            },
            {
                "iter": "1", "sampled_steps": "256", "learner_steps": "256",
                "episodes": "2", "effective_learner_time_s": "2.5",
                "reward_mean": "1", "crash_rate": "0",
                "target_profile_fraction_auto": "0.5", "target_profile_fraction_bt": "0.5",
                "aim_variant_fraction_left": "1", "aim_variant_fraction_right": "0",
            },
            {
                "iter": "2", "sampled_steps": "384", "learner_steps": "512",
                "episodes": "3", "effective_learner_time_s": "3.7",
                "reward_mean": "-1", "crash_rate": "1",
                "target_profile_fraction_auto": "0", "target_profile_fraction_bt": "1",
                "aim_variant_fraction_left": "0", "aim_variant_fraction_right": "1",
            },
        ]

        result = analyze_rows(rows)

        self.assertEqual(result["sampled_steps"], 384)
        self.assertEqual(result["learner_steps"], 512)
        self.assertEqual(result["estimated_crash_episodes"], 1)
        self.assertEqual(result["critical_nan_rows"], [])
        self.assertEqual(
            result["curriculum"]["target_profile_"]["counts"],
            {"auto": 1.0, "bt": 2.0},
        )
        self.assertEqual(
            result["curriculum"]["aim_variant_"]["counts"],
            {"left": 2.0, "right": 1.0},
        )


if __name__ == "__main__":
    unittest.main()
