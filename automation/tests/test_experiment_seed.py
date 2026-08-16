from __future__ import annotations

from pathlib import Path
import unittest

from scripts.run_experiment import build_argv


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


if __name__ == "__main__":
    unittest.main()
