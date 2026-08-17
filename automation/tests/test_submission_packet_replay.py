from __future__ import annotations

import unittest

import numpy as np

from automation.replay_submission_packets import latency_summary, replay_packet_pairs
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
        return ActionResult(self.action.copy(), "test")


def _pair(frame: int, *, target_yaw: float = 0.0):
    own = PlaneInfo(
        index=frame,
        plane_id=1,
        position=Vector3D(0.0, 0.0, 5000.0),
        rotation=Rotation3D(0.0, 0.0, 0.0),
        velocity=Vector3D(230.0, 0.0, 0.0),
    )
    target = PlaneInfo(
        index=frame,
        plane_id=2,
        position=Vector3D(800.0, 0.0, 5000.0),
        rotation=Rotation3D(0.0, 0.0, target_yaw),
        velocity=Vector3D(225.0, 0.0, 0.0),
    )
    return own, target


class SubmissionPacketReplayTests(unittest.TestCase):
    def _policy(self, *, target_yaw: float = 0.0):
        bt = _Provider([0.1, -0.2, 0.05, 0.7])
        rl = _Provider([0.8, -0.4, 0.2, 0.0])
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

    def test_replay_runs_bt_each_frame_and_rl_only_at_repeat(self) -> None:
        policy, bt, rl = self._policy()
        records = replay_packet_pairs(policy, [_pair(frame) for frame in range(6)])

        self.assertEqual(bt.calls, 6)
        self.assertEqual(rl.calls, 1)
        self.assertEqual([item["command"]["index"] for item in records], list(range(6)))
        self.assertTrue(all(len(item["observation"]) == 16 for item in records))
        self.assertTrue(all(item["provider_last_frame"]["gate"]["active"] for item in records))
        for item in records:
            command = item["command"]
            self.assertAlmostEqual(command["throttle_cmd"], 0.7, places=6)
            np.testing.assert_allclose(
                [command["roll_cmd"], command["pitch_cmd"], command["yaw_cmd"]],
                [0.19, -0.24, 0.07375],
                atol=1e-6,
            )

    def test_negative_beam_replay_skips_rl_and_is_exact_bt(self) -> None:
        policy, bt, rl = self._policy(target_yaw=90.0)
        records = replay_packet_pairs(
            policy,
            [_pair(frame, target_yaw=90.0) for frame in range(60)],
        )

        self.assertEqual(bt.calls, 60)
        self.assertEqual(rl.calls, 0)
        for item in records:
            command = item["command"]
            np.testing.assert_array_equal(
                np.asarray(
                    [command["roll_cmd"], command["pitch_cmd"], command["yaw_cmd"], command["throttle_cmd"]],
                    dtype=np.float32,
                ),
                bt.action,
            )
            self.assertFalse(item["provider_last_frame"]["gate"]["active"])

    def test_latency_summary_reports_required_percentiles(self) -> None:
        summary = latency_summary(
            [{"latency_ms": value} for value in (1.0, 2.0, 3.0, 200.0)]
        )
        self.assertEqual(summary["count"], 4)
        self.assertGreater(summary["p95_ms"], 3.0)
        self.assertEqual(summary["max_ms"], 200.0)
        self.assertEqual(summary["over_threshold_ratio"], 0.25)


if __name__ == "__main__":
    unittest.main()
