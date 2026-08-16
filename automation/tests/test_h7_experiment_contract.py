from __future__ import annotations

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load(name: str) -> dict:
    return yaml.safe_load((ROOT / "experiments" / name).read_text(encoding="utf-8"))


class H7ExperimentContractTests(unittest.TestCase):
    def test_independent_seed_configs_differ_only_in_identity_and_seed(self) -> None:
        configs = [
            load(f"0815_aim_residual_mirror_balanced_s{seed}.yaml")
            for seed in (1701, 1702, 1703)
        ]
        for value in configs:
            value.pop("name")
            value.pop("notes")
            value["output"].pop("tag")
            value["runtime"].pop("seed")
        self.assertEqual(configs[0], configs[1])
        self.assertEqual(configs[0], configs[2])

    def test_h7_changes_h6_scenario_distribution_but_freezes_core_settings(self) -> None:
        h6 = load("0815_aim_residual_balanced_reward_short.yaml")
        h7 = load("0815_aim_residual_mirror_balanced_s1701.yaml")
        for path in (
            ("env", "observation_mode"),
            ("env", "target_mode"),
            ("env", "max_engage_time"),
            ("env_config", "step_ratio"),
            ("env_config", "residual_training"),
            ("env_config", "reward"),
            ("algo",),
        ):
            left = h6
            right = h7
            for key in path:
                left = left[key]
                right = right[key]
            self.assertEqual(left, right, msg=".".join(path))

        variants = h7["env_config"]["initial_scenario"]["variants"]
        self.assertEqual(
            [variant["name"] for variant in variants],
            [
                "lateral_left",
                "lateral_right",
                "crossing_left",
                "crossing_right",
                "vertical_high",
                "vertical_low",
            ],
        )
        self.assertEqual(h7["runtime"]["seed"], 1701)
        self.assertEqual(h7["runtime"]["iterations"], h6["runtime"]["iterations"])


if __name__ == "__main__":
    unittest.main()
