from __future__ import annotations

import unittest

import numpy as np

from GeoMathUtil import GeometryInfo
from dogfight.ai.action_provider import ActionProvider, ActionResult
from dogfight.ai.guidance_advantage import build_server_guidance_observation
from dogfight.ai.guidance_selector import GuidanceSetpoint
from dogfight.envs.observation import (
    TACTICAL16_HEALTH_CONSTANT_ONE,
    build_observation,
)
from dogfight.sim.state_schema import StateIndex
from dogfight.unreal.policies import ProviderCommandPolicy, plane_info_to_state
from dogfight.unreal.protocol import (
    PlaneInfo,
    Rotation3D,
    Vector3D,
    pack_plane_info,
    unpack_plane_info,
)


class _UnusedProvider(ActionProvider):
    def compute_action(self, context) -> ActionResult:
        return ActionResult(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), "test")


def _plane(
    *,
    index: int,
    plane_id: int,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float],
    velocity: tuple[float, float, float],
) -> PlaneInfo:
    return PlaneInfo(
        index=index,
        plane_id=plane_id,
        position=Vector3D(*position),
        rotation=Rotation3D(*rotation),
        velocity=Vector3D(*velocity),
    )


class SubmissionObservationParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wez = {
            "min_range_m": 152.4,
            "max_range_m": 914.4,
            "angle_deg": 2.0,
        }
        self.policy = ProviderCommandPolicy(
            _UnusedProvider(),
            observation_mode="tactical16",
            wez_config=self.wez,
            health_source=TACTICAL16_HEALTH_CONSTANT_ONE,
            expected_sim_hz=60,
        )

    def test_training_and_submission_tactical16_vectors_are_byte_equal(self) -> None:
        own_packet = _plane(
            index=600,
            plane_id=1,
            position=(0.0, 0.0, 5000.0),
            rotation=(5.0, -2.0, 355.0),
            velocity=(230.0, 0.0, 0.0),
        )
        target_packet = _plane(
            index=600,
            plane_id=2,
            position=(800.0, -20.0, 5050.0),
            rotation=(-3.0, 1.0, 5.0),
            velocity=(220.0, 5.0, 0.0),
        )
        own_state = plane_info_to_state(own_packet)
        target_state = plane_info_to_state(target_packet)

        submission = self.policy._build_observation(own_state, target_state)
        training = build_observation(
            "tactical16",
            own_state,
            target_state,
            GeometryInfo(),
            self.wez,
            health_source=TACTICAL16_HEALTH_CONSTANT_ONE,
        )

        self.assertEqual(submission.dtype, np.float32)
        self.assertEqual(submission.tobytes(), training.tobytes())
        self.assertEqual(float(submission[5]), 1.0)
        self.assertEqual(float(submission[13]), 1.0)

    def test_plane_info_state_uses_ned_down_and_body_velocity(self) -> None:
        packet = _plane(
            index=1,
            plane_id=1,
            position=(10.0, 20.0, 3000.0),
            rotation=(0.0, 0.0, 90.0),
            velocity=(0.0, 200.0, 0.0),
        )
        state = plane_info_to_state(packet)

        self.assertEqual(float(state[StateIndex.D]), -3000.0)
        self.assertEqual(float(state[StateIndex.ALT]), 3000.0)
        np.testing.assert_allclose(state[6:9], [200.0, 0.0, 0.0], atol=1e-5)
        self.assertEqual(float(state[StateIndex.HEALTH]), 1.0)

    def test_submission_phase_clock_is_frame_based_at_60hz(self) -> None:
        self.policy._match_start_frame_index = 9000
        self.assertEqual(self.policy._sim_time_s(9000), 0.0)
        self.assertEqual(self.policy._sim_time_s(15000), 100.0)
        self.assertEqual(self.policy._sim_time_s(18000), 150.0)
        self.assertEqual(self.policy._sim_time_s(21000), 200.0)

    def test_server_guidance_observation_is_packet_replay_identical(self) -> None:
        own_packet = _plane(
            index=120,
            plane_id=1,
            position=(0.0, 0.0, 5000.0),
            rotation=(2.0, -1.0, 359.0),
            velocity=(230.0, 2.0, 0.0),
        )
        target_packet = _plane(
            index=120,
            plane_id=2,
            position=(750.0, 30.0, 5040.0),
            rotation=(-1.0, 2.0, 4.0),
            velocity=(215.0, -3.0, 0.0),
        )
        replay_own = unpack_plane_info(pack_plane_info(own_packet))
        replay_target = unpack_plane_info(pack_plane_info(target_packet))
        bt_action = np.asarray([0.1, -0.2, 0.05, 0.9], dtype=np.float32)
        base = GuidanceSetpoint(1.0, -0.5, 800.0, 230.0)
        kwargs = dict(
            sim_time_s=2.0,
            previous_action_id=0,
            action_hold_frames=0,
            gate_elapsed_frames=1,
            gate_active=True,
            minimum_action_hold_frames=6,
            maximum_active_frames=36,
        )
        live = build_server_guidance_observation(
            plane_info_to_state(own_packet),
            plane_info_to_state(target_packet),
            bt_action,
            base,
            **kwargs,
        )
        replay = build_server_guidance_observation(
            plane_info_to_state(replay_own),
            plane_info_to_state(replay_target),
            bt_action,
            base,
            **kwargs,
        )
        self.assertEqual(live.tobytes(), replay.tobytes())

    def test_tactical16_wez_switches_on_official_phase_boundaries(self) -> None:
        own = plane_info_to_state(
            _plane(
                index=0,
                plane_id=1,
                position=(0.0, 0.0, 5000.0),
                rotation=(0.0, 0.0, 0.0),
                velocity=(230.0, 0.0, 0.0),
            )
        )
        target = plane_info_to_state(
            _plane(
                index=0,
                plane_id=2,
                position=(1000.0, 0.0, 5000.0),
                rotation=(0.0, 0.0, 0.0),
                velocity=(225.0, 0.0, 0.0),
            )
        )

        phase1 = self.policy._build_observation(own, target, sim_time_s=100.0)
        phase2 = self.policy._build_observation(
            own,
            target,
            sim_time_s=100.0 + 1.0 / 60.0,
        )

        self.assertEqual(float(phase1[14]), -1.0)
        self.assertEqual(float(phase2[14]), 1.0)


if __name__ == "__main__":
    unittest.main()
