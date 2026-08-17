from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dogfight.submission.config import load_submission_config


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


class SubmissionConfigTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        bundle = root / "policy"
        bundle.mkdir()
        weights = bundle / "policy_weights.pkl.gz"
        weights.write_bytes(b"weights")
        wez = {"min_range_m": 152.4, "max_range_m": 914.4, "angle_deg": 2.0}
        (bundle / "metadata.json").write_text(
            json.dumps(
                {
                    "algorithm_config": {
                        "env_config": {
                            "observation_mode": "tactical16",
                            "observation_contract": {
                                "version": "tactical16.v1",
                                "normalization_version": "tactical16.norm.v1",
                                "health_source": "unavailable_constant_one",
                            },
                            "wez": wez,
                            "phase_config": [
                                {"phase": 1, "end_s": 100.0, "half_angle_deg": 1.0, "max_range_m": 914.4},
                                {"phase": 2, "end_s": 150.0, "half_angle_deg": 2.0, "max_range_m": 1066.8},
                                {"phase": 3, "end_s": 200.0, "half_angle_deg": 3.0, "max_range_m": 1219.2}
                            ],
                        }
                    },
                    "metadata": {
                        "observation_mode": "tactical16",
                        "observation_size": 16,
                        "observation_contract_version": "tactical16.v1",
                        "normalization_version": "tactical16.norm.v1",
                        "health_source": "unavailable_constant_one",
                        "wez_contract": wez,
                        "phase_config": [
                            {"phase": 1, "end_s": 100.0, "half_angle_deg": 1.0, "max_range_m": 914.4},
                            {"phase": 2, "end_s": 150.0, "half_angle_deg": 2.0, "max_range_m": 1066.8},
                            {"phase": 3, "end_s": 200.0, "half_angle_deg": 3.0, "max_range_m": 1219.2}
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )
        dll = root / "bt.dll"
        xml = root / "rule.xml"
        dll.write_bytes(b"dll")
        xml.write_bytes(b"xml")
        config = {
            "mode": "bt_residual",
            "observation_mode": "auto",
            "observation_size": 16,
            "observation_contract_version": "tactical16.v1",
            "normalization_version": "tactical16.norm.v1",
            "health_source": "unavailable_constant_one",
            "bundle_path": "policy",
            "bundle_sha256": _sha(weights),
            "policy_id": "default_policy",
            "bt": {
                "dll_path": "bt.dll",
                "dll_sha256": _sha(dll),
                "xml_path": "rule.xml",
                "xml_sha256": _sha(xml),
            },
            "residual_scale": 0.125,
            "composition_mode": "saturation_aware",
            "rl_action_repeat": 6,
            "hard_eligibility_gate": {"kind": "rear120"},
            "activation_gate": {"kind": "rear120_and_offensive_or_pre_aim"},
            "wez": wez,
            "phase_config": [
                {"phase": 1, "end_s": 100.0, "half_angle_deg": 1.0, "max_range_m": 914.4},
                {"phase": 2, "end_s": 150.0, "half_angle_deg": 2.0, "max_range_m": 1066.8},
                {"phase": 3, "end_s": 200.0, "half_angle_deg": 3.0, "max_range_m": 1219.2}
            ],
            "throttle_policy": "bt_only",
            "expected_sim_hz": 60,
            "latency_threshold_s": 0.1667,
        }
        config_path = root / "submission.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path

    def test_auto_mode_uses_and_verifies_bundle_contract(self) -> None:
        with TemporaryDirectory() as temp:
            loaded = load_submission_config(self._fixture(Path(temp)))

        self.assertEqual(loaded.observation_mode, "tactical16")
        self.assertEqual(loaded.observation_size, 16)
        self.assertEqual(loaded.health_source, "unavailable_constant_one")
        self.assertEqual(loaded.residual_scale, 0.125)
        self.assertEqual(loaded.expected_sim_hz, 60)

    def test_observation_mode_mismatch_fails_before_runtime(self) -> None:
        with TemporaryDirectory() as temp:
            path = self._fixture(Path(temp))
            config = json.loads(path.read_text(encoding="utf-8"))
            config["observation_mode"] = "aim_residual10_v2"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "observation mode mismatch"):
                load_submission_config(path)

    def test_health_or_wez_mismatch_fails_before_runtime(self) -> None:
        with TemporaryDirectory() as temp:
            path = self._fixture(Path(temp))
            config = json.loads(path.read_text(encoding="utf-8"))
            config["health_source"] = "simulator"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "health_source mismatch"):
                load_submission_config(path)

        with TemporaryDirectory() as temp:
            path = self._fixture(Path(temp))
            config = json.loads(path.read_text(encoding="utf-8"))
            config["wez"]["max_range_m"] = 1000.0
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "WEZ contract mismatch"):
                load_submission_config(path)

    def test_invalid_scale_or_hash_fails_fast(self) -> None:
        with TemporaryDirectory() as temp:
            path = self._fixture(Path(temp))
            config = json.loads(path.read_text(encoding="utf-8"))
            config["residual_scale"] = 0.2
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "residual_scale"):
                load_submission_config(path)

        with TemporaryDirectory() as temp:
            path = self._fixture(Path(temp))
            config = json.loads(path.read_text(encoding="utf-8"))
            config["bundle_sha256"] = "0" * 64
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bundle SHA256 mismatch"):
                load_submission_config(path)


if __name__ == "__main__":
    unittest.main()
