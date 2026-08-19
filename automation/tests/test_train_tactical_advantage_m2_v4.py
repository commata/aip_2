from __future__ import annotations

import numpy as np
import torch

from automation.train_tactical_advantage_m2_v4 import (
    M2_INPUT_SIZE,
    TacticalGRUAdvantageNetwork,
    encode_m2_input,
)


def test_m2_reconstructs_four_step_server_safe_sequence() -> None:
    observation = np.zeros(93, dtype=np.float32)
    row = {
        "server_observation": observation.tolist(),
        "mode": "PURE_PURSUIT",
        "hold_frames": 30,
    }
    vector = encode_m2_input(row)
    assert vector.shape == (M2_INPUT_SIZE,)
    assert np.all(np.isfinite(vector))


def test_m2_network_outputs_three_distributional_heads() -> None:
    model = TacticalGRUAdvantageNetwork()
    outputs = model(torch.zeros((3, M2_INPUT_SIZE), dtype=torch.float32))
    assert len(outputs) == 3
    assert all(value.shape == (3,) for value in outputs)
