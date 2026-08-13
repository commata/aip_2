from __future__ import annotations

from argparse import Namespace
import unittest

from automation.evaluate_offensive_hybrid import ALLOWED_SCALES, aggregate


def record(controller, outcome, margin, saturation, altitude, ata=20.0):
    return {
        "controller": controller,
        "outcome": outcome,
        "health_margin": margin,
        "total_reward": margin,
        "mean_ata_deg": ata,
        "action_saturation_ratio": saturation,
        "gate_active_ratio": 0.1 if controller != "bt" else 0.0,
        "rl_inference_calls": 10 if controller != "bt" else 0,
        "min_altitude_m": altitude,
    }


class OffensiveEvaluatorTests(unittest.TestCase):
    def test_scale_contract_is_fixed(self):
        self.assertEqual(ALLOWED_SCALES, (0.10, 0.125, 0.15, 0.175, 0.20))

    def test_unsafe_candidate_cannot_be_selected(self):
        args = Namespace(
            max_crash_rate_regression=0.05,
            max_win_rate_regression=0.0,
            max_health_margin_regression=0.05,
            max_saturation_ratio=0.20,
            max_saturation_rate_regression=0.02,
            minimum_safe_altitude_m=300.0,
        )
        records = [
            record("bt", "draw", 0.0, 0.1, 1000.0, ata=40.0),
            record("hybrid_0.15", "win", 1.0, 0.5, 1000.0, ata=10.0),
            record("hybrid_0.125", "draw", 0.1, 0.1, 1000.0, ata=30.0),
        ]
        summary = aggregate(records, args)
        self.assertFalse(summary["candidates"][1]["valid"])
        self.assertEqual(summary["best_valid_candidate"]["controller"], "hybrid_0.125")

    def test_win_rate_regression_is_not_a_valid_candidate(self):
        args = Namespace(
            max_crash_rate_regression=0.05,
            max_win_rate_regression=0.0,
            max_health_margin_regression=0.05,
            max_saturation_ratio=1.0,
            max_saturation_rate_regression=0.02,
            minimum_safe_altitude_m=300.0,
        )
        summary = aggregate(
            [
                record("bt", "win", 1.0, 0.5, 1000.0),
                record("hybrid_0.2", "timeout", 0.0, 0.4, 1000.0),
            ],
            args,
        )
        self.assertIsNone(summary["best_valid_candidate"])


if __name__ == "__main__":
    unittest.main()
