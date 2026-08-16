from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from dogfight.envs.observation import aim_residual_geometry, build_observation
from dogfight.research.mirror_symmetry import (
    LATERAL_CONTRACT,
    VERTICAL_CONTRACT,
    mirror_action,
    mirror_pose_lateral,
    mirror_pose_vertical,
    mirror_state_lateral,
    mirror_state_vertical,
)
from dogfight.sim.state_schema import StateIndex


def state(position, attitude, body_velocity) -> np.ndarray:
    value = np.zeros(51, dtype=np.float64)
    value[:3] = position
    value[3:6] = attitude
    value[6:9] = body_velocity
    value[StateIndex.KCAS] = np.linalg.norm(body_velocity)
    value[StateIndex.ALT] = -value[StateIndex.D]
    value[StateIndex.HEALTH] = 1.0
    return value


class MirrorSymmetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.own = state(
            (0.0, 40.0, -5000.0),
            (10.0, 3.0, 20.0),
            (230.0, 4.0, -2.0),
        )
        self.target = state(
            (1100.0, -350.0, -5200.0),
            (-5.0, -2.0, 5.0),
            (220.0, -3.0, 1.0),
        )

    def assert_geometry_contract(self, mirrored_own, mirrored_target, contract) -> None:
        original = aim_residual_geometry(self.own, self.target)
        mirrored = aim_residual_geometry(mirrored_own, mirrored_target)
        for key, sign in contract.geometry_signs.items():
            self.assertAlmostEqual(
                mirrored[key],
                sign * original[key],
                places=9,
                msg=key,
            )

    def test_lateral_geometry_has_exact_declared_signs(self) -> None:
        self.assert_geometry_contract(
            mirror_state_lateral(self.own),
            mirror_state_lateral(self.target),
            LATERAL_CONTRACT,
        )

    def test_vertical_geometry_has_exact_declared_signs(self) -> None:
        self.assert_geometry_contract(
            mirror_state_vertical(self.own, down_origin_m=-5000.0),
            mirror_state_vertical(self.target, down_origin_m=-5000.0),
            VERTICAL_CONTRACT,
        )

    def test_10d_lateral_observation_sign_contract(self) -> None:
        original = build_observation("aim_residual10_v2", self.own, self.target, None)
        mirrored = build_observation(
            "aim_residual10_v2",
            mirror_state_lateral(self.own),
            mirror_state_lateral(self.target),
            None,
        )
        np.testing.assert_allclose(mirrored, original * [-1, 1, -1, 1, 1, 1, 1, 1, 1, 1])

    def test_10d_vertical_observation_sign_contract(self) -> None:
        original = build_observation("aim_residual10_v2", self.own, self.target, None)
        mirrored = build_observation(
            "aim_residual10_v2",
            mirror_state_vertical(self.own, down_origin_m=-5000.0),
            mirror_state_vertical(self.target, down_origin_m=-5000.0),
            None,
        )
        np.testing.assert_allclose(mirrored, original * [1, -1, 1, -1, 1, 1, 1, 1, 1, 1])

    def test_requested_applied_and_final_action_contracts_are_explicit(self) -> None:
        action = np.asarray([0.2, -0.3, 0.1, 0.77])
        np.testing.assert_allclose(
            mirror_action(action, LATERAL_CONTRACT),
            [-0.2, -0.3, -0.1, 0.77],
        )
        np.testing.assert_allclose(
            mirror_action(action, VERTICAL_CONTRACT),
            [-0.2, 0.3, 0.1, 0.77],
        )

    def test_scenario_pose_pairs_are_mathematical_mirrors(self) -> None:
        lateral_left = [1100.0, -350.0, -5000.0, 0.0, 0.0, 5.0, 225.0]
        self.assertEqual(
            mirror_pose_lateral(lateral_left),
            [1100.0, 350.0, -5000.0, -0.0, 0.0, 355.0, 225.0],
        )
        vertical_high = [1100.0, 0.0, -5300.0, 0.0, 0.0, 0.0, 225.0]
        self.assertEqual(
            mirror_pose_vertical(vertical_high, down_origin_m=-5000.0),
            [1100.0, 0.0, -4700.0, -0.0, -0.0, 0.0, 225.0],
        )

    def test_checked_in_scenario_files_match_their_declared_mirrors(self) -> None:
        root = Path(__file__).resolve().parents[1] / "scenarios" / "0815_aim_mirror"
        for left_name, right_name in (
            ("lateral_left", "lateral_right"),
            ("crossing_left", "crossing_right"),
        ):
            left = json.loads((root / f"{left_name}.json").read_text(encoding="utf-8"))
            right = json.loads((root / f"{right_name}.json").read_text(encoding="utf-8"))
            self.assertEqual(left["mirror_pair"], right_name)
            self.assertEqual(right["mirror_pair"], left_name)
            self.assertEqual(
                mirror_pose_lateral(left["env_config"]["ownship"]),
                right["env_config"]["ownship"],
            )
            self.assertEqual(
                mirror_pose_lateral(left["env_config"]["target"]),
                right["env_config"]["target"],
            )

        high = json.loads((root / "vertical_high.json").read_text(encoding="utf-8"))
        low = json.loads((root / "vertical_low.json").read_text(encoding="utf-8"))
        origin = float(high["mirror_down_origin_m"])
        self.assertEqual(origin, float(low["mirror_down_origin_m"]))
        self.assertEqual(
            mirror_pose_vertical(high["env_config"]["ownship"], down_origin_m=origin),
            low["env_config"]["ownship"],
        )
        self.assertEqual(
            mirror_pose_vertical(high["env_config"]["target"], down_origin_m=origin),
            low["env_config"]["target"],
        )


if __name__ == "__main__":
    unittest.main()
