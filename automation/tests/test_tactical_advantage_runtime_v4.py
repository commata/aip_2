from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from dogfight.ai.tactical_advantage import (
    TACTICAL_MODEL_INPUT_SIZE,
    TACTICAL_MODEL_KIND,
    NumpyTemporalTacticalAdvantageSelector,
    encode_tactical_option,
)
from dogfight.ai.tactical_modes import TACTICAL_HOLD_FRAMES, TACTICAL_MODES_T1
from dogfight.ai.temporal_observation import TEMPORAL_FEATURES


def _bundle(tmp_path, *, ood_threshold: float = 100.0):
    hidden = 4
    arrays = {
        "model_0_input_mean": np.zeros(TACTICAL_MODEL_INPUT_SIZE, dtype=np.float32),
        "model_0_input_scale": np.ones(TACTICAL_MODEL_INPUT_SIZE, dtype=np.float32),
        "model_0_shared.0.weight": np.zeros((hidden, TACTICAL_MODEL_INPUT_SIZE), dtype=np.float32),
        "model_0_shared.0.bias": np.zeros(hidden, dtype=np.float32),
        "model_0_shared.2.weight": np.zeros((hidden, hidden), dtype=np.float32),
        "model_0_shared.2.bias": np.zeros(hidden, dtype=np.float32),
        "model_0_mean_head.weight": np.zeros((1, hidden), dtype=np.float32),
        "model_0_mean_head.bias": np.ones(1, dtype=np.float32),
        "model_0_positive_head.weight": np.zeros((1, hidden), dtype=np.float32),
        "model_0_positive_head.bias": np.full(1, 10.0, dtype=np.float32),
        "model_0_regression_head.weight": np.zeros((1, hidden), dtype=np.float32),
        "model_0_regression_head.bias": np.full(1, -10.0, dtype=np.float32),
        "support_examples": np.zeros((1, 93), dtype=np.float32),
        "support_scale": np.ones(93, dtype=np.float32),
    }
    model_path = tmp_path / "model.npz"
    np.savez_compressed(model_path, **arrays)
    metadata = {
        "model_kind": TACTICAL_MODEL_KIND,
        "observation_contract": "guidance_selector_server_temporal_v4",
        "features": list(TEMPORAL_FEATURES),
        "offline_gate_passed": True,
        "runtime_threshold": {"score": 0.001, "positive": 0.6, "regression": 0.1, "lambda": 1.0},
        "runtime_candidates": [
            {"mode": mode, "hold_frames": hold}
            for mode in TACTICAL_MODES_T1[1:]
            for hold in TACTICAL_HOLD_FRAMES
        ],
        "target_scale": 0.01,
        "ensemble_size": 1,
        "runtime_ood_support": {"threshold": ood_threshold},
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest().upper(),
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return tmp_path


def test_factorized_tactical_option_is_five_dimensional_and_frozen() -> None:
    vector = encode_tactical_option("LEAD_PURSUIT_T060", 60)
    assert vector.shape == (5,)
    np.testing.assert_array_equal(vector[:4], [0, 0, 1, 0])
    assert vector[4] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        encode_tactical_option("BT_DEFAULT", 30)


def test_runtime_enumerates_options_and_selects_nondefault(tmp_path) -> None:
    selector = NumpyTemporalTacticalAdvantageSelector(_bundle(tmp_path))
    result = selector.select(np.zeros(93, dtype=np.float32))
    assert result["mode"] != "BT_DEFAULT"
    assert result["hold_frames"] in TACTICAL_HOLD_FRAMES
    assert len(result["options"]) == 9
    assert result["q10_advantage"] == pytest.approx(result["q50_advantage"])


def test_runtime_ood_abstains_with_explicit_reason(tmp_path) -> None:
    selector = NumpyTemporalTacticalAdvantageSelector(
        _bundle(tmp_path, ood_threshold=0.1)
    )
    result = selector.select(np.ones(93, dtype=np.float32))
    assert result["mode"] == "BT_DEFAULT"
    assert result["abstention_reason"] == "OOD"
