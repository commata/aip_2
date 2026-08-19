from __future__ import annotations

import numpy as np

from automation.train_tactical_advantage_long_v4 import (
    LONG_MODEL_INPUT_SIZE,
    encode_long_input,
)
from dogfight.ai.temporal_observation import (
    LONG_TEMPORAL_OBSERVATION_SIZE,
    LongTemporalServerObservationBuilder,
)


def test_long_temporal_padding_and_size() -> None:
    builder = LongTemporalServerObservationBuilder()
    vector = builder.append_observation(np.zeros(42, dtype=np.float32))
    assert vector.shape == (LONG_TEMPORAL_OBSERVATION_SIZE,)
    np.testing.assert_array_equal(vector[42:], np.zeros(85, dtype=np.float32))


def test_long_model_input_is_server_observation_plus_option() -> None:
    row = {
        "server_observation": np.zeros(LONG_TEMPORAL_OBSERVATION_SIZE).tolist(),
        "mode": "LEAD_PURSUIT_T060",
        "hold_frames": 60,
    }
    assert encode_long_input(row).shape == (LONG_MODEL_INPUT_SIZE,)
