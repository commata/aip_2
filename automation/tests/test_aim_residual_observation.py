from __future__ import annotations

import unittest

import numpy as np

from dogfight.envs.observation import (
    aim_residual_geometry,
    body_to_ned_rotation,
    build_observation,
    observation_size,
)
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


if __name__ == "__main__":
    unittest.main()
