from __future__ import annotations

import unittest

import numpy as np

from automation.validate_udp_loopback import assert_loopback_contract, run_udp_loopback
from dogfight.ai.action_provider import ActionProvider, ActionResult
from dogfight.ai.hybrid_action_provider import ResidualInferenceActionProvider
from dogfight.envs.observation import TACTICAL16_HEALTH_CONSTANT_ONE
from dogfight.unreal.policies import ProviderCommandPolicy
from dogfight.unreal.protocol import PlaneInfo, Rotation3D, Vector3D


class _Provider(ActionProvider):
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.calls = 0

    def compute_action(self, context) -> ActionResult:
        self.calls += 1
        return ActionResult(self.action.copy(), "udp-loopback")


def _pair(frame: int, *, target_yaw: float = 0.0):
    return (
        PlaneInfo(
            index=frame,
            plane_id=1,
            position=Vector3D(0.0, 0.0, 5000.0),
            rotation=Rotation3D(0.0, 0.0, 0.0),
            velocity=Vector3D(230.0, 0.0, 0.0),
        ),
        PlaneInfo(
            index=frame,
            plane_id=2,
            position=Vector3D(800.0, 0.0, 5000.0),
            rotation=Rotation3D(0.0, 0.0, target_yaw),
            velocity=Vector3D(225.0, 0.0, 0.0),
        ),
    )


class SubmissionUdpLoopbackTests(unittest.TestCase):
    def _policy(self, target_yaw: float = 0.0):
        bt = _Provider([0.1, -0.2, 0.05, 0.7])
        rl = _Provider([0.8, -0.4, 0.2, 1.0])
        hybrid = ResidualInferenceActionProvider(
            bt,
            rl,
            residual_scale=0.125,
            gate_kind="rear120",
            rl_action_repeat=6,
            composition_mode="saturation_aware",
        )
        policy = ProviderCommandPolicy(
            hybrid,
            observation_mode="tactical16",
            action_repeat=1,
            wez_config={"min_range_m": 152.4, "max_range_m": 914.4, "angle_deg": 2.0},
            health_source=TACTICAL16_HEALTH_CONSTANT_ONE,
            expected_sim_hz=60,
        )
        return policy, bt, rl

    def test_60hz_udp_loopback_preserves_frames_and_action_repeat(self) -> None:
        policy, bt, rl = self._policy()
        result = run_udp_loopback(
            policy,
            [_pair(frame) for frame in range(60)],
            expected_hz=60.0,
            real_time=True,
        )

        assert_loopback_contract(result)
        self.assertEqual(bt.calls, 60)
        self.assertEqual(rl.calls, 10)
        self.assertEqual(result.latency["over_threshold_ratio"], 0.0)
        self.assertTrue(
            all(abs(command["throttle_cmd"] - 0.7) < 1e-6 for command in result.commands)
        )

    def test_negative_geometry_uses_exact_bt_and_skips_rl(self) -> None:
        policy, bt, rl = self._policy(target_yaw=90.0)
        result = run_udp_loopback(
            policy,
            [_pair(frame, target_yaw=90.0) for frame in range(30)],
            expected_hz=60.0,
            real_time=False,
        )

        assert_loopback_contract(result)
        self.assertEqual(bt.calls, 30)
        self.assertEqual(rl.calls, 0)
        for command in result.commands:
            np.testing.assert_array_equal(
                np.asarray(
                    [
                        command["roll_cmd"],
                        command["pitch_cmd"],
                        command["yaw_cmd"],
                        command["throttle_cmd"],
                    ],
                    dtype=np.float32,
                ),
                bt.action,
            )


if __name__ == "__main__":
    unittest.main()
