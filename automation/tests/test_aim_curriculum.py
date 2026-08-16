from __future__ import annotations

import unittest

import numpy as np

from dogfight.envs.single_agent_env import DogFightEnv


class AimCurriculumTests(unittest.TestCase):
    def test_variant_sets_both_aircraft_and_target_autopilot(self) -> None:
        env = DogFightEnv.__new__(DogFightEnv)
        env.np_random = np.random.default_rng(7)
        env.config = {
            "target_autopilot": {
                "heading_cmd": 0.0,
                "altitude_cmd": 5000.0,
                "speed_cmd": 220.0,
            }
        }
        calls = []
        env.change_init_position = lambda flight, **values: calls.append(
            (flight, values)
        )
        scenario = {
            "variants": [
                {
                    "name": "left",
                    "ownship": [0, 0, -5000, 0, 0, 0, 230],
                    "target": [1100, -350, -5000, 0, 0, 5, 225],
                    "target_autopilot": {
                        "heading_cmd": 5.0,
                        "altitude_cmd": 5000.0,
                        "speed_cmd": 225.0,
                    },
                }
            ]
        }

        env._apply_aim_residual_initial_scenario(scenario)

        self.assertEqual([call[0] for call in calls], ["ownship", "target"])
        self.assertEqual(calls[1][1]["init_e"], -350)
        self.assertEqual(env.config["target_autopilot"]["heading_cmd"], 5.0)
        self.assertEqual(env._initial_scenario_metrics["aim_curriculum_variant_name"], "left")

    def test_empty_variant_list_is_rejected(self) -> None:
        env = DogFightEnv.__new__(DogFightEnv)
        env.np_random = np.random.default_rng(1)
        with self.assertRaises(ValueError):
            env._apply_aim_residual_initial_scenario({"variants": []})


if __name__ == "__main__":
    unittest.main()
