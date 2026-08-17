from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import run_unreal_inference as runtime
from dogfight.ai.action_provider import ActionProvider, ActionResult
from dogfight.ai.hybrid_action_provider import ResidualInferenceActionProvider
from dogfight.submission.config import SubmissionConfig


class _Provider(ActionProvider):
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def compute_action(self, context) -> ActionResult:
        self.calls += 1
        return ActionResult(np.array([0.0, 0.0, 0.0, 0.8], dtype=np.float32), "test")


def _submission() -> SubmissionConfig:
    raw = {
        "hard_eligibility_gate": {
            "kind": "rear120",
            "enter_target_ata_deg": 120.0,
            "exit_target_ata_deg": 110.0,
        },
        "activation_gate": {
            "kind": "rear120_and_offensive_or_pre_aim",
            "offensive": {
                "enter_min_target_ata_deg": 120.0,
                "exit_min_target_ata_deg": 110.0,
            },
            "phase_pre_aim": {},
            "safety_veto": {},
        },
    }
    return SubmissionConfig(
        source_path=Path("submission.json"),
        raw=raw,
        observation_mode="tactical16",
        observation_size=16,
        observation_contract_version="tactical16.v1",
        normalization_version="tactical16.norm.v1",
        health_source="unavailable_constant_one",
        bundle_path=Path("policy"),
        policy_id="default_policy",
        bt_dll_path=Path("bt.dll"),
        bt_xml_path=Path("rule.xml"),
        residual_scale=0.125,
        composition_mode="saturation_aware",
        rl_action_repeat=6,
        expected_sim_hz=60,
        latency_threshold_s=0.1667,
        wez_config={"min_range_m": 152.4, "max_range_m": 914.4, "angle_deg": 2.0},
        phase_config=[
            {"phase": 1, "end_s": 100.0, "half_angle_deg": 1.0, "max_range_m": 914.4},
            {"phase": 2, "end_s": 150.0, "half_angle_deg": 2.0, "max_range_m": 1066.8},
            {"phase": 3, "end_s": 200.0, "half_angle_deg": 3.0, "max_range_m": 1219.2},
        ],
    )


class SubmissionRuntimeTests(unittest.TestCase):
    def test_hybrid_requires_single_source_submission_config(self) -> None:
        args = Namespace(mode="hybrid", bundle_dir="policy", _submission_config=None)
        with self.assertRaisesRegex(ValueError, "submission-config"):
            runtime.build_action_provider(args)

    def test_submission_builds_residual_inference_not_legacy_hybrid(self) -> None:
        args = Namespace(
            mode="hybrid",
            bundle_dir="policy",
            policy_id="default_policy",
            explore=False,
            bt_dll="bt.dll",
            _submission_config=_submission(),
        )
        with patch.object(runtime, "RLActionProvider", _Provider), patch.object(
            runtime, "BTActionProvider", _Provider
        ):
            provider = runtime.build_action_provider(args)

        self.assertIsInstance(provider, ResidualInferenceActionProvider)
        self.assertEqual(provider.gate_kind, "rear120")
        self.assertEqual(provider.residual_scale, 0.125)
        self.assertEqual(provider.rl_action_repeat, 6)
        self.assertEqual(provider.composition_mode, "saturation_aware")


if __name__ == "__main__":
    unittest.main()
