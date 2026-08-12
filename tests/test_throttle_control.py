from __future__ import annotations

import unittest

import numpy as np

from dogfight.ai.action_provider import ActionProvider, ActionResult
from dogfight.ai.bt_action_provider import TurnThrottleConfig, optimize_full_turn_throttle
from dogfight.ai.hybrid_action_provider import HybridActionProvider


CONFIG = TurnThrottleConfig()


def _optimized(raw, roll, pitch, speed, altitude=0.0, *, tas=False):
    return optimize_full_turn_throttle(
        raw,
        roll,
        pitch,
        speed,
        altitude,
        speed_is_true_airspeed=tas,
        config=CONFIG,
    )


class _FixedProvider(ActionProvider):
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def compute_action(self, context) -> ActionResult:
        return ActionResult(action=self.action.copy(), source="fixed")


class ThrottleControlTests(unittest.TestCase):
    def test_tactical_throttle_is_preserved(self) -> None:
        throttle, info = _optimized(0.72, 1.0, -1.0, 300.0)
        self.assertEqual(throttle, 0.72)
        self.assertEqual(info["reason"], "tactical_throttle")

    def test_full_throttle_is_preserved_when_not_turning_or_low_energy(self) -> None:
        straight, straight_info = _optimized(1.0, 0.2, -0.2, 300.0)
        slow, slow_info = _optimized(1.0, 1.0, -1.0, 160.0)
        self.assertEqual(straight, 1.0)
        self.assertEqual(straight_info["reason"], "not_aggressive_turn")
        self.assertEqual(slow, 1.0)
        self.assertEqual(slow_info["reason"], "low_energy")

    def test_aggressive_turn_reduces_full_throttle_above_corner_speed(self) -> None:
        throttle, info = _optimized(1.0, 1.0, -1.0, 280.0)
        self.assertGreaterEqual(throttle, CONFIG.minimum_turn_throttle)
        self.assertLess(throttle, 0.65)
        self.assertIs(info["active"], True)
        self.assertEqual(info["reason"], "corner_speed_control")

    def test_remote_tas_target_increases_with_altitude(self) -> None:
        sea_level, sea_info = _optimized(1.0, 1.0, -1.0, 300.0, 0.0, tas=True)
        altitude, altitude_info = _optimized(
            1.0, 1.0, -1.0, 300.0, 7000.0, tas=True
        )
        self.assertGreater(
            altitude_info["target_speed_mps"], sea_info["target_speed_mps"]
        )
        self.assertGreater(altitude, sea_level)

    def test_residual_hybrid_does_not_add_throttles_to_saturation(self) -> None:
        provider = HybridActionProvider(
            primary_provider=_FixedProvider([0.4, -0.2, 0.2, 0.8]),
            secondary_provider=_FixedProvider([0.3, -0.4, 0.1, 0.9]),
            mode="residual",
            residual_scale=0.25,
        )

        result = provider.compute_action(None)

        np.testing.assert_allclose(result.action[:3], [0.4, -0.45, 0.15], atol=1e-6)
        self.assertTrue(np.isclose(result.action[3], 0.875))
        self.assertLess(result.action[3], 1.0)


if __name__ == "__main__":
    unittest.main()
