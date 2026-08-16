from __future__ import annotations

import unittest

import numpy as np

from dogfight.envs.observation import aim_residual_geometry
from dogfight.envs.reward import compute_aim_residual_reward
from dogfight.sim.state_schema import StateIndex


def state(n, e, yaw, speed=220.0, altitude=5000.0) -> np.ndarray:
    result = np.zeros(51, dtype=np.float32)
    result[:3] = [n, e, -altitude]
    result[StateIndex.YAW] = yaw
    result[6] = speed
    result[StateIndex.KCAS] = speed
    result[StateIndex.ALT] = altitude
    result[StateIndex.HEALTH] = 1.0
    result[StateIndex.SIM_TIME] = 50.0
    return result


class AimResidualRewardTests(unittest.TestCase):
    def test_damage_dominates_small_cone_dwell_bonus(self) -> None:
        own = state(0.0, 0.0, 0.0)
        target = state(800.0, 0.0, 180.0)
        reward, components, _, _ = compute_aim_residual_reward(
            own,
            target,
            0.0,
            0.01,
            {"mode": "aim_residual"},
            False,
            False,
            "",
            previous_geometry=aim_residual_geometry(own, target),
            action_info={"gate": {"active": True}},
            previous_correction=np.zeros(4),
        )
        self.assertGreater(components["damage"], components["cone_dwell"] * 10.0)
        self.assertGreater(reward, 0.0)

    def test_gate_off_has_no_residual_or_aim_credit(self) -> None:
        own = state(0.0, 0.0, 0.0)
        target = state(800.0, 0.0, 180.0)
        _, components, _, correction = compute_aim_residual_reward(
            own,
            target,
            0.0,
            0.0,
            {"mode": "aim_residual"},
            False,
            False,
            "",
            previous_geometry=None,
            action_info={
                "gate": {"active": False},
                "applied_rl_correction": [0.1, 0.1, 0.1, 0.0],
                "action_clipped": True,
            },
            previous_correction=np.zeros(4),
        )
        for key in (
            "aim_progress",
            "aim_quality",
            "los_rate",
            "cone_dwell",
            "residual_l2",
            "residual_smooth",
            "clipping",
        ):
            self.assertEqual(components[key], 0.0)
        np.testing.assert_allclose(correction, [0.1, 0.1, 0.1, 0.0])


if __name__ == "__main__":
    unittest.main()
