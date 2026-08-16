from __future__ import annotations

import unittest

from automation.evaluate_aim_residual import (
    aggregate,
    controller_observation_mode,
    merge_unique_records,
)


def record(controller: str, seed: int, variant: str, los: float = 2.0):
    return {
        "controller": controller,
        "seed": seed,
        "variant_name": variant,
        "outcome": "win",
        "end_condition": "target destroyed",
        "ownship_crash": False,
        "target_crash": False,
        "mean_los_deg": los,
        "episode_seconds": 10.0,
        "damage_dealt": 1.0,
        "los_rate_rms_deg_s": 3.0,
        "damage_cone_time_s": 4.0,
        "time_to_first_damage_s": 5.0,
        "min_altitude_m": 4000.0,
    }


class AimEvaluatorQualityTests(unittest.TestCase):
    def test_btaware_bundle_does_not_pre_tick_pure_bt_baseline(self) -> None:
        self.assertEqual(
            controller_observation_mode(
                "aim_residual13_btaware",
                "pure_0815",
            ),
            "aim_residual10_v2",
        )
        self.assertEqual(
            controller_observation_mode(
                "aim_residual13_btaware",
                "hybrid_0.125",
            ),
            "aim_residual13_btaware",
        )

    def test_duplicate_seed_results_are_not_counted_as_unique(self) -> None:
        rows = [
            record("pure_0815", 1, "fixed"),
            record("pure_0815", 2, "fixed"),
            record("hybrid_0.125", 1, "fixed", 1.9),
            record("hybrid_0.125", 2, "fixed", 1.9),
        ]

        summary = aggregate(rows)

        self.assertEqual(
            summary["controllers"]["pure_0815"]["unique_result_signatures"],
            1,
        )
        self.assertTrue(summary["data_quality_warnings"])

    def test_variant_mismatch_is_reported(self) -> None:
        rows = [
            record("pure_0815", 1, "left"),
            record("hybrid_0.125", 1, "right", 1.9),
        ]

        summary = aggregate(rows)

        self.assertTrue(
            any("variant 불일치" in warning for warning in summary["data_quality_warnings"])
        )

    def test_resume_merge_preserves_prior_seeds_without_duplicates(self) -> None:
        existing = [
            record("pure_0815", 1, "left"),
            record("hybrid_0.125", 1, "left", 1.9),
        ]
        additions = [
            record("pure_0815", 1, "left"),
            record("pure_0815", 2, "right"),
            record("hybrid_0.125", 2, "right", 1.8),
        ]

        merged = merge_unique_records(existing, additions)

        self.assertEqual(len(merged), 4)
        self.assertEqual(
            {(row["seed"], row["controller"]) for row in merged},
            {
                (1, "pure_0815"),
                (1, "hybrid_0.125"),
                (2, "pure_0815"),
                (2, "hybrid_0.125"),
            },
        )


if __name__ == "__main__":
    unittest.main()
