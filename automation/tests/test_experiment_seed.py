from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.run_experiment import build_argv
import train_rllib


class ExperimentSeedTests(unittest.TestCase):
    def test_runtime_seed_is_forwarded_to_training_cli(self) -> None:
        experiment = {
            "script": "train_rllib",
            "output": {"name": "seed_test", "tag": "s1701"},
            "algo": {"name": "sac"},
            "runtime": {"iterations": 1, "seed": 1701},
            "dashboard": {"enabled": False},
        }

        _, argv = build_argv(experiment, Path("seed_test.yaml"))

        index = argv.index("--seed")
        self.assertEqual(argv[index + 1], "1701")

    def test_runtime_diagnostics_and_ray_cpu_cap_are_forwarded(self) -> None:
        experiment = {
            "script": "train_rllib",
            "output": {"name": "runtime_test", "tag": "diagnostic"},
            "algo": {"name": "sac"},
            "runtime": {
                "iterations": 1,
                "runtime_diagnostics": True,
                "ray_num_cpus": 2,
            },
            "dashboard": {"enabled": False},
        }

        _, argv = build_argv(experiment, Path("runtime_test.yaml"))

        self.assertIn("--runtime-diagnostics", argv)
        index = argv.index("--ray-num-cpus")
        self.assertEqual(argv[index + 1], "2")

    def test_algorithm_log_root_is_inside_workspace_artifacts(self) -> None:
        args = type(
            "Args",
            (),
            {"output_name": "runtime_test", "output_tag": "diagnostic"},
        )()

        root = train_rllib._algorithm_log_root(args)

        self.assertEqual(
            root,
            train_rllib.ROOT
            / "artifacts"
            / "ray_results"
            / "runtime_test"
            / "diagnostic",
        )

    def test_training_seed_initializes_numpy_before_algorithm_build(self) -> None:
        with patch.object(train_rllib.random, "seed") as python_seed, patch.object(
            train_rllib.np.random, "seed"
        ) as numpy_seed, patch("torch.manual_seed") as torch_seed:
            train_rllib._seed_training_runtime(1701)

        python_seed.assert_called_once_with(1701)
        numpy_seed.assert_called_once_with(1701)
        torch_seed.assert_called_once_with(1701)

    def test_variant_fraction_metrics_are_preserved_in_training_rows(self) -> None:
        result = {
            "env_runners": {
                "custom_metrics": {
                    "aim_variant_fraction_lateral_left_mean": 0.4,
                    "aim_variant_fraction_lateral_right_mean": 0.6,
                }
            }
        }

        metrics = train_rllib._extract_custom_metrics(result)

        self.assertEqual(metrics["aim_variant_fraction_lateral_left"], 0.4)
        self.assertEqual(metrics["aim_variant_fraction_lateral_right"], 0.6)


if __name__ == "__main__":
    unittest.main()
