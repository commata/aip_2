from __future__ import annotations

import unittest

from automation.optimize_offensive_hybrid import build_trials, is_better


class OffensiveOptimizerTests(unittest.TestCase):
    def test_trial_order_ends_with_heldout(self):
        config = {
            "scales": [0.1, 0.15],
            "development_seeds": [1],
            "heldout_seeds": [2],
            "development_max_engage_time": 30,
            "heldout_max_engage_time": 200,
            "gate_candidates": [{"name": "default"}, {"name": "tight"}],
        }
        trials = build_trials(config)
        self.assertEqual(trials[0]["phase"], "development_scale")
        self.assertEqual(trials[-1]["phase"], "heldout")
        self.assertEqual(len(trials), 4)

    def test_only_higher_safe_score_replaces_best(self):
        self.assertTrue(is_better({"score": 2.0}, {"score": 1.0}))
        self.assertFalse(is_better({"score": 0.5}, {"score": 1.0}))
        self.assertFalse(is_better(None, {"score": 1.0}))


if __name__ == "__main__":
    unittest.main()
