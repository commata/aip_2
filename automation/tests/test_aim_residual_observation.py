from __future__ import annotations

import unittest

import numpy as np

from dogfight.envs.observation import (
    TACTICAL16_CONTRACT_VERSION,
    TACTICAL16_FEATURES,
    TACTICAL16_HEALTH_CONSTANT_ONE,
    aim_residual_geometry,
    body_to_ned_rotation,
    build_observation,
    observation_size,
)
from GeoMathUtil import GeometryInfo
from dogfight.sim.state_schema import StateIndex


def state(n, e, d, yaw, body_velocity=(0.0, 0.0, 0.0)) -> np.ndarray:
    result = np.zeros(51, dtype=np.float32)
    result[:3] = [n, e, d]
    result[StateIndex.YAW] = yaw
    result[6:9] = body_velocity
    result[StateIndex.KCAS] = float(np.linalg.norm(body_velocity))
    result[StateIndex.ALT] = -d
    return result


class AimResidualObservationTests(unittest.TestCase):
    def test_body_to_ned_rotation_respects_heading(self) -> None:
        east = body_to_ned_rotation([0.0, 0.0, 90.0]) @ [1.0, 0.0, 0.0]
        np.testing.assert_allclose(east, [0.0, 1.0, 0.0], atol=1e-7)

    def test_geometry_has_signed_aim_error_and_closing_rate(self) -> None:
        own = state(0.0, 0.0, -5000.0, 0.0, (200.0, 0.0, 0.0))
        target = state(1000.0, 100.0, -5100.0, 180.0, (100.0, 0.0, 0.0))
        geometry = aim_residual_geometry(own, target)
        self.assertGreater(geometry["aim_azimuth_deg"], 0.0)
        self.assertGreater(geometry["aim_elevation_deg"], 0.0)
        self.assertGreater(geometry["closing_rate_m_s"], 0.0)
        self.assertLess(geometry["ata_deg"], 10.0)
        self.assertLess(geometry["target_ata_deg"], 10.0)

    def test_observation_is_ten_dimensional_and_bounded(self) -> None:
        own = state(0.0, 0.0, -5000.0, 0.0, (250.0, 0.0, 0.0))
        target = state(1200.0, 30.0, -5000.0, 180.0, (200.0, 0.0, 0.0))
        observation = build_observation("aim_residual10", own, target, None)
        self.assertEqual(observation_size("aim_residual10"), 10)
        self.assertEqual(observation.shape, (10,))
        self.assertTrue(np.all(np.isfinite(observation)))
        self.assertTrue(np.all(np.abs(observation) <= 1.0))

    def test_v2_preserves_features_but_amplifies_local_aim_errors(self) -> None:
        own = state(0.0, 0.0, -5000.0, 0.0, (230.0, 0.0, 0.0))
        target = state(1000.0, 100.0, -5000.0, 0.0, (225.0, 0.0, 0.0))
        v1 = build_observation("aim_residual10", own, target, None)
        v2 = build_observation("aim_residual10_v2", own, target, None)

        self.assertEqual(observation_size("aim_residual10_v2"), 10)
        self.assertEqual(v2.shape, (10,))
        self.assertTrue(np.all(np.isfinite(v2)))
        self.assertTrue(np.all(np.abs(v2) <= 1.0))
        self.assertGreater(abs(float(v2[0])), abs(float(v1[0])) * 5.0)

    def test_btaware_observation_appends_exact_surface_commands(self) -> None:
        own = state(0.0, 0.0, -5000.0, 0.0, (230.0, 0.0, 0.0))
        target = state(1000.0, 100.0, -5000.0, 0.0, (225.0, 0.0, 0.0))
        bt_action = np.array([0.25, -0.5, 0.75, 0.8], dtype=np.float32)

        observation = build_observation(
            "aim_residual13_btaware",
            own,
            target,
            None,
            bt_action=bt_action,
        )

        self.assertEqual(observation_size("aim_residual13_btaware"), 13)
        self.assertEqual(observation.shape, (13,))
        np.testing.assert_allclose(
            observation[:10],
            build_observation("aim_residual10_v2", own, target, None),
        )
        np.testing.assert_array_equal(observation[10:], bt_action[:3])

    def test_btaware_observation_requires_same_frame_bt_action(self) -> None:
        own = state(0.0, 0.0, -5000.0, 0.0, (230.0, 0.0, 0.0))
        target = state(1000.0, 100.0, -5000.0, 0.0, (225.0, 0.0, 0.0))

        with self.assertRaisesRegex(ValueError, "bt_action"):
            build_observation("aim_residual13_btaware", own, target, None)

    def test_tactical16_constant_health_contract_ignores_simulator_health(self) -> None:
        own = state(0.0, 0.0, -5000.0, 0.0, (230.0, 0.0, 0.0))
        target = state(800.0, 0.0, -5000.0, 0.0, (225.0, 0.0, 0.0))
        own[StateIndex.HEALTH] = 0.2
        target[StateIndex.HEALTH] = 0.4
        wez = {"min_range_m": 152.4, "max_range_m": 914.4, "angle_deg": 2.0}

        observation = build_observation(
            "tactical16",
            own,
            target,
            GeometryInfo(),
            wez,
            health_source=TACTICAL16_HEALTH_CONSTANT_ONE,
        )

        self.assertEqual(observation.shape, (len(TACTICAL16_FEATURES),))
        self.assertEqual(TACTICAL16_CONTRACT_VERSION, "tactical16.v1")
        self.assertEqual(float(observation[5]), 1.0)
        self.assertEqual(float(observation[13]), 1.0)
        self.assertEqual(float(observation[14]), 1.0)


if __name__ == "__main__":
    unittest.main()
