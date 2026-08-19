from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from dogfight.ai.guidance_selector import (
    GUIDANCE_ACTIONS,
    GUIDANCE_SELECTOR_CONTRACT_VERSION,
    GUIDANCE_SELECTOR_FEATURES,
    GUIDANCE_SELECTOR_NORMALIZATION_VERSION,
    GUIDANCE_SELECTOR_OBSERVATION_SIZE,
)
from dogfight.ai.guidance_advantage import (
    GUIDANCE_ADVANTAGE_ACTIONS,
    GUIDANCE_SERVER_CONTRACT_VERSION,
    GUIDANCE_SERVER_FEATURES,
    GUIDANCE_SERVER_NORMALIZATION_VERSION,
    GUIDANCE_SERVER_OBSERVATION_SIZE,
)
from dogfight.envs.observation import OFFICIAL_DAMAGE_PHASES
from dogfight.submission.guidance_config import load_guidance_submission_config
from run_unreal_inference import resolve_runtime_contract


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


class GuidanceSubmissionConfigTests(unittest.TestCase):
    def fixture(self, root: Path) -> Path:
        bundle = root / "bundle"
        bundle.mkdir()
        np.savez(
            bundle / "model.npz",
            w1=np.zeros((45, 4), dtype=np.float32),
            b1=np.zeros(4, dtype=np.float32),
            w2=np.zeros((4, 4), dtype=np.float32),
            b2=np.zeros(4, dtype=np.float32),
            w3=np.zeros((4, 9), dtype=np.float32),
            b3=np.zeros(9, dtype=np.float32),
        )
        model_hash = digest(bundle / "model.npz")
        (bundle / "metadata.json").write_text(
            json.dumps({"model_sha256": model_hash}), encoding="utf-8"
        )
        dll = root / "bt.dll"
        xml = root / "bt.xml"
        dll.write_bytes(b"dll")
        xml.write_bytes(b"xml")
        payload = {
            "status": "SUBMISSION_READY_HYBRID_CANDIDATE",
            "mode": "guidance_selector",
            "runtime_observation_mode": "tactical16",
            "selector_observation_contract": GUIDANCE_SELECTOR_CONTRACT_VERSION,
            "selector_observation_size": GUIDANCE_SELECTOR_OBSERVATION_SIZE,
            "normalization_version": GUIDANCE_SELECTOR_NORMALIZATION_VERSION,
            "observation_features": list(GUIDANCE_SELECTOR_FEATURES),
            "bundle_path": "bundle",
            "bundle_sha256": model_hash,
            "policy_id": "guidance_selector_bc",
            "bt": {
                "dll_path": "bt.dll",
                "dll_sha256": digest(dll),
                "xml_path": "bt.xml",
                "xml_sha256": digest(xml),
                "rule_aliases": ["Rule_DCS_GDCC_0815.xml"],
            },
            "action_library": list(GUIDANCE_ACTIONS),
            "action_magnitude": {
                "angular_offset_deg": 0.5,
                "range_offset_m": 50.0,
                "target_speed_offset_m_s": 10.0,
            },
            "controller": {
                "roll_per_angular_action": 0.04,
                "pitch_per_angular_action": 0.04,
                "yaw_per_angular_action": 0.02,
                "pitch_per_range_action": 0.01,
                "pitch_per_speed_action": 0.02,
                "maximum_surface_correction": 0.08,
            },
            "runtime": {
                "selector_action_repeat_frames": 6,
                "minimum_action_hold_frames": 18,
                "maximum_active_frames": 90,
                "cooldown_frames": 30,
                "confidence_threshold": 0.65,
                "inference_timeout_s": 0.1667,
                "sim_hz": 60,
            },
            "hard_eligibility_gate": {
                "kind": "rear120",
                "enter_target_ata_deg": 120.0,
                "exit_target_ata_deg": 110.0,
            },
            "activation_gate": {
                "kind": "rear120_and_offensive_or_pre_aim",
                "offensive": {},
                "phase_pre_aim": {},
                "safety_veto": {},
            },
            "wez": {"min_range_m": 152.4, "max_range_m": 914.4, "angle_deg": 2.0},
            "phase_config": [dict(item) for item in OFFICIAL_DAMAGE_PHASES],
            "health_source": "unavailable_constant_one",
            "throttle_policy": "bt_only",
            "fallback_mode": "exact_pure_bt",
            "expected_sim_hz": 60,
            "latency_threshold_s": 0.1667,
        }
        path = root / "submission.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loads_and_verifies_complete_contract(self):
        with TemporaryDirectory() as temp:
            loaded = load_guidance_submission_config(self.fixture(Path(temp)))
            self.assertEqual(loaded.runtime_config.minimum_action_hold_frames, 18)
            self.assertEqual(loaded.bundle_path.name, "bundle")

    def test_loads_server_safe_state_action_contract(self):
        with TemporaryDirectory() as temp:
            path = self.fixture(Path(temp))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload.update(
                {
                    "selector_observation_contract": GUIDANCE_SERVER_CONTRACT_VERSION,
                    "selector_observation_size": GUIDANCE_SERVER_OBSERVATION_SIZE,
                    "normalization_version": GUIDANCE_SERVER_NORMALIZATION_VERSION,
                    "observation_features": list(GUIDANCE_SERVER_FEATURES),
                    "action_library": list(GUIDANCE_ADVANTAGE_ACTIONS),
                }
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_guidance_submission_config(path)
            self.assertEqual(
                loaded.raw["selector_observation_contract"],
                GUIDANCE_SERVER_CONTRACT_VERSION,
            )

    def test_hash_mismatch_fails_fast(self):
        with TemporaryDirectory() as temp:
            path = self.fixture(Path(temp))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["bundle_sha256"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_guidance_submission_config(path)

    def test_action_order_mismatch_fails_fast(self):
        with TemporaryDirectory() as temp:
            path = self.fixture(Path(temp))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["action_library"] = list(reversed(payload["action_library"]))
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "action library mismatch"):
                load_guidance_submission_config(path)

    def test_unreal_runtime_resolves_guidance_mode_from_config(self):
        with TemporaryDirectory() as temp:
            path = self.fixture(Path(temp))
            args = SimpleNamespace(
                submission_config=str(path),
                mode=None,
                bundle_dir=None,
                policy_id="",
                observation_mode="auto",
                bt_dll="",
                bt_rule_xml="",
                bt_rule_alias=[],
                action_repeat=0,
                residual_scale=0.1,
                ownship_force_side=1,
                target_force_side=2,
            )
            submission = resolve_runtime_contract(args)
            self.assertEqual(args.mode, "guidance")
            self.assertEqual(args.observation_mode, "tactical16")
            self.assertEqual(args.action_repeat, 6)
            self.assertEqual(submission.bundle_path.name, "bundle")


if __name__ == "__main__":
    unittest.main()
