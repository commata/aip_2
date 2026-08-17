from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from automation.target_profiles import (
    TargetProfileError,
    apply_target_profile,
    load_target_profile,
)


class TargetProfileTests(unittest.TestCase):
    def _write_profile(self, root: Path, dll: Path, xml: Path) -> Path:
        profile = {
            "profile_id": "test_bt",
            "backend_type": "behavior_tree",
            "dll": {
                "default_path": str(dll),
                "path_env": "TEST_TARGET_DLL",
                "sha256": hashlib.sha256(dll.read_bytes()).hexdigest(),
            },
            "xml": {
                "default_path": str(xml),
                "path_env": "TEST_TARGET_XML",
                "sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
            },
            "rule_aliases": ["Rule_test.xml"],
            "source": "unit test",
            "smoke_status": "PENDING",
            "behavior_cluster": "test_cluster",
            "use": {"training": True, "validation": False, "held_out": False},
        }
        path = root / "test_bt.json"
        path.write_text(json.dumps(profile), encoding="utf-8")
        return path

    def test_profile_hashes_and_environment_override_are_resolved(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            dll = root / "default.dll"
            override = root / "override.dll"
            xml = root / "rule.xml"
            dll.write_bytes(b"default")
            override.write_bytes(b"override")
            xml.write_text("<root/>", encoding="utf-8")
            profile_path = self._write_profile(root, override, xml)

            profile = load_target_profile(
                profile_path,
                environ={"TEST_TARGET_DLL": str(override)},
            )

            self.assertEqual(profile["dll"]["resolved_path"], str(override.resolve()))
            self.assertEqual(profile["dll"]["path_source"], "env:TEST_TARGET_DLL")

    def test_hash_mismatch_is_rejected(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            dll = root / "target.dll"
            xml = root / "rule.xml"
            dll.write_bytes(b"original")
            xml.write_text("<root/>", encoding="utf-8")
            profile_path = self._write_profile(root, dll, xml)
            dll.write_bytes(b"changed")

            with self.assertRaisesRegex(TargetProfileError, "sha256 mismatch"):
                load_target_profile(profile_path, environ={})

    def test_apply_profile_injects_backend_without_observation_identity(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            dll = root / "target.dll"
            xml = root / "rule.xml"
            dll.write_bytes(b"dll")
            xml.write_text("<root/>", encoding="utf-8")
            self._write_profile(root, dll, xml)
            experiment = {
                "env": {
                    "target_profile": "test_bt",
                    "observation_mode": "aim_residual10_v2",
                }
            }

            merged, profile = apply_target_profile(
                experiment,
                environ={},
                profile_dir=root,
            )

            self.assertEqual(merged["env"]["target_mode"], "behavior_tree")
            self.assertEqual(merged["env"]["target_behavior_dll"], str(dll.resolve()))
            self.assertEqual(merged["env"]["target_rule_aliases"], ["Rule_test.xml"])
            self.assertEqual(merged["env"]["observation_mode"], "aim_residual10_v2")
            self.assertNotIn("target_profile_id", merged["env"])
            self.assertEqual(profile["behavior_cluster"], "test_cluster")

    def test_repository_profiles_resolve_current_external_inventory(self) -> None:
        for profile_id in ("bt_0815", "bt_aip2", "bt_aip3"):
            with self.subTest(profile_id=profile_id):
                profile = load_target_profile(profile_id)
                self.assertEqual(
                    profile["dll"]["actual_sha256"],
                    profile["dll"]["sha256"],
                )

    def test_profile_pool_resolves_weighted_backends_without_observation_identity(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            dll = root / "target.dll"
            xml = root / "rule.xml"
            dll.write_bytes(b"dll")
            xml.write_text("<root/>", encoding="utf-8")
            self._write_profile(root, dll, xml)
            autopilot = {
                "profile_id": "test_autopilot",
                "backend_type": "autopilot",
                "source": "unit test",
                "smoke_status": "PASSED",
                "behavior_cluster": "scripted",
                "use": {"training": True, "validation": True, "held_out": False},
            }
            (root / "test_autopilot.json").write_text(
                json.dumps(autopilot), encoding="utf-8"
            )
            experiment = {
                "env": {
                    "observation_mode": "aim_residual10_v2",
                    "target_profile_pool": [
                        {"profile": "test_autopilot", "weight": 2},
                        {"profile": "test_bt", "weight": 1},
                    ],
                }
            }

            merged, result = apply_target_profile(
                experiment,
                environ={},
                profile_dir=root,
            )

            env = merged["env"]
            self.assertEqual(env["target_mode"], "profile_curriculum")
            self.assertEqual(
                [item["profile_id"] for item in env["target_profile_curriculum"]],
                ["test_autopilot", "test_bt"],
            )
            self.assertEqual(env["target_rule_aliases"], ["Rule_test.xml"])
            self.assertNotIn("target_profile_id", env)
            self.assertIn("profile_pool", result)


if __name__ == "__main__":
    unittest.main()
