from __future__ import annotations

import unittest

from dogfight.envs.single_agent_env import DogFightEnv


class OutcomeClassificationTests(unittest.TestCase):
    def test_ownship_altitude_termination_is_crash(self) -> None:
        outcome = DogFightEnv._classify_outcome(
            True,
            False,
            "ownship altitude below min",
            0.8,
            1.0,
        )
        self.assertEqual(outcome, "crash")

    def test_target_altitude_termination_is_win(self) -> None:
        outcome = DogFightEnv._classify_outcome(
            True,
            False,
            "target altitude below min",
            1.0,
            0.9,
        )
        self.assertEqual(outcome, "win")

    def test_damage_destruction_takes_priority(self) -> None:
        outcome = DogFightEnv._classify_outcome(
            True,
            False,
            "target destroyed",
            0.5,
            0.0,
        )
        self.assertEqual(outcome, "win")


if __name__ == "__main__":
    unittest.main()
