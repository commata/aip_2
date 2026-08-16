from __future__ import annotations

import unittest

from automation.evaluate_aim_residual import aggregate


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


if __name__ == "__main__":
    unittest.main()
