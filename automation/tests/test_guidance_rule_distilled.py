from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "build_guidance_rule_distilled", ROOT / "automation/build_guidance_rule_distilled.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frozen_rule_is_narrow_and_defaults_outside_preaim_window():
    observation = np.zeros(45, dtype=np.float32)
    observation[43] = 1.0
    observation[42] = -1.0
    assert MODULE.rule_action(observation) == MODULE.RULE_ACTION_ID

    observation[42] = 0.0
    assert MODULE.rule_action(observation) == 0
    observation[43] = -1.0
    observation[42] = -1.0
    assert MODULE.rule_action(observation) == 0


def test_distilled_bundle_matches_rule_grid(tmp_path):
    result = MODULE.build(tmp_path / "bundle")
    assert result["verification"]["matched"] == result["verification"]["total"] == 14
