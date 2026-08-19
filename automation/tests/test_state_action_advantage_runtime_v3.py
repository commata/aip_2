from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from dogfight.ai.guidance_advantage import (
    GUIDANCE_ADVANTAGE_ACTIONS,
    GUIDANCE_SERVER_CONTRACT_VERSION,
    GUIDANCE_SERVER_FEATURES,
)
from dogfight.ai.state_action_advantage import (
    MODEL_KIND,
    NumpyStateActionAdvantageSelector,
    factorized_action,
)


def _bundle(path, *, gate_passed: bool = True):
    hidden = 4
    arrays = {
        "model_0_input_mean": np.zeros(48, dtype=np.float32),
        "model_0_input_scale": np.ones(48, dtype=np.float32),
        "model_0_shared.0.weight": np.zeros((hidden, 48), dtype=np.float32),
        "model_0_shared.0.bias": np.zeros(hidden, dtype=np.float32),
        "model_0_shared.2.weight": np.zeros((hidden, hidden), dtype=np.float32),
        "model_0_shared.2.bias": np.zeros(hidden, dtype=np.float32),
        "model_0_mean_head.weight": np.zeros((1, hidden), dtype=np.float32),
        "model_0_mean_head.bias": np.asarray([0.5], dtype=np.float32),
        "model_0_positive_head.weight": np.zeros((1, hidden), dtype=np.float32),
        "model_0_positive_head.bias": np.asarray([8.0], dtype=np.float32),
        "model_0_regression_head.weight": np.zeros((1, hidden), dtype=np.float32),
        "model_0_regression_head.bias": np.asarray([-8.0], dtype=np.float32),
        "support_examples": np.zeros((2, 42), dtype=np.float32),
        "support_mean": np.zeros(42, dtype=np.float32),
        "support_scale": np.ones(42, dtype=np.float32),
    }
    model_path = path / "model.npz"
    np.savez_compressed(model_path, **arrays)
    metadata = {
        "model_kind": MODEL_KIND,
        "observation_contract": GUIDANCE_SERVER_CONTRACT_VERSION,
        "features": list(GUIDANCE_SERVER_FEATURES),
        "actions": list(GUIDANCE_ADVANTAGE_ACTIONS),
        "offline_gate_passed": gate_passed,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest().upper(),
        "ensemble_size": 1,
        "runtime_threshold": {
            "score": 0.001,
            "positive": 0.8,
            "regression": 0.05,
            "lambda": 1.0,
        },
        "runtime_candidates": [
            {"action": action, "magnitude_deg": 0.25, "duration_frames": 36}
            for action in GUIDANCE_ADVANTAGE_ACTIONS[1:]
        ],
        "runtime_ood_support": {
            "kind": "nearest_training_state_rms_z_v1",
            "threshold": 0.5,
            "training_states": 2,
            "fallback": "BT_DEFAULT",
        },
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_factorized_action_contract() -> None:
    assert factorized_action("VP_AZ_NEG_SMALL", 0.25, 36).tolist() == [
        0.0,
        1.0,
        0.0,
        -1.0,
        0.5,
        1.0,
    ]


def test_numpy_runtime_enumerates_full_action_grid(tmp_path) -> None:
    _bundle(tmp_path)
    selector = NumpyStateActionAdvantageSelector(tmp_path)
    scored = selector.score_actions(np.zeros(42, dtype=np.float32))
    action_id, confidence, probabilities = selector.predict(np.zeros(42, dtype=np.float32))
    assert len(scored) == 4
    assert action_id == 1
    assert confidence > 0.99
    assert probabilities.shape == (9,)
    assert all(row["conservative_score"] > 0.009 for row in scored)


def test_runtime_refuses_offline_gate_failure(tmp_path) -> None:
    _bundle(tmp_path, gate_passed=False)
    with pytest.raises(ValueError, match="offline policy gate"):
        NumpyStateActionAdvantageSelector(tmp_path)


def test_runtime_abstains_to_exact_default_outside_training_support(tmp_path) -> None:
    _bundle(tmp_path)
    selector = NumpyStateActionAdvantageSelector(tmp_path)
    action_id, confidence, probabilities = selector.predict(
        np.full(42, 10.0, dtype=np.float32)
    )
    assert action_id == 0
    assert confidence == 1.0
    assert probabilities.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert selector.last_prediction["fallback_reason"] == "OOD_STATE"
