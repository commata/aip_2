from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from dogfight.ai.guidance_advantage import (
    GUIDANCE_SERVER_FEATURE_SPECS,
    GUIDANCE_SERVER_FEATURES,
    GUIDANCE_SERVER_OBSERVATION_SIZE,
    build_server_guidance_observation,
    validate_server_guidance_observation,
)


TEMPORAL_SERVER_CONTRACT_VERSION = "guidance_selector_server_temporal_v4"
TEMPORAL_SERVER_NORMALIZATION_VERSION = "guidance_selector_server_temporal.norm.v1"
TEMPORAL_HISTORY_LAGS = (6, 12, 30)
LONG_TEMPORAL_HISTORY_LAGS = (6, 12, 30, 60, 120)
TEMPORAL_PADDING = "repeat_first_zero_delta"
LONG_TEMPORAL_SERVER_CONTRACT_VERSION = "guidance_selector_server_temporal_long120_v4"

# These features are the compact temporal signal selected before dataset discovery.
# Values are already normalized by guidance_selector_server_v2. Deltas are divided
# by two so the difference of two bounded values remains in [-1, 1].
TEMPORAL_DELTA_SOURCE_FEATURES = (
    "signed_aim_azimuth_norm",
    "signed_aim_elevation_norm",
    "los_azimuth_rate_norm",
    "los_elevation_rate_norm",
    "range_norm",
    "closing_rate_norm",
    "ownship_roll_norm",
    "ownship_pitch_norm",
    "ownship_yaw_norm",
    "bt_roll",
    "bt_pitch",
    "bt_yaw",
    "bt_vp_local_azimuth_norm",
    "bt_vp_local_elevation_norm",
    "bt_vp_distance_norm",
    "any_surface_saturation",
    "recent_applied_requested_authority_ratio",
)
_FEATURE_INDEX = {name: index for index, name in enumerate(GUIDANCE_SERVER_FEATURES)}
TEMPORAL_DELTA_SOURCE_INDICES = tuple(
    _FEATURE_INDEX[name] for name in TEMPORAL_DELTA_SOURCE_FEATURES
)
TEMPORAL_FEATURES = (
    *GUIDANCE_SERVER_FEATURES,
    *tuple(
        f"delta_t{lag}_{name}"
        for lag in TEMPORAL_HISTORY_LAGS
        for name in TEMPORAL_DELTA_SOURCE_FEATURES
    ),
)
TEMPORAL_OBSERVATION_SIZE = len(TEMPORAL_FEATURES)
LONG_TEMPORAL_FEATURES = (
    *GUIDANCE_SERVER_FEATURES,
    *tuple(
        f"delta_t{lag}_{name}"
        for lag in LONG_TEMPORAL_HISTORY_LAGS
        for name in TEMPORAL_DELTA_SOURCE_FEATURES
    ),
)
LONG_TEMPORAL_OBSERVATION_SIZE = len(LONG_TEMPORAL_FEATURES)


def validate_temporal_observation(observation: Any) -> np.ndarray:
    vector = np.asarray(observation, dtype=np.float32)
    if vector.shape != (TEMPORAL_OBSERVATION_SIZE,):
        raise ValueError(
            f"temporal observation must have shape ({TEMPORAL_OBSERVATION_SIZE},), "
            f"got {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("temporal observation contains nonfinite values")
    return np.clip(vector, -1.0, 1.0)


class TemporalServerObservationBuilder:
    """Build the v4 current-plus-delta observation from local frame history."""

    def __init__(self) -> None:
        self._history: deque[np.ndarray] = deque(maxlen=max(TEMPORAL_HISTORY_LAGS) + 1)

    @property
    def frames_seen(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        self._history.clear()

    def append_observation(self, current_observation: Any) -> np.ndarray:
        current = validate_server_guidance_observation(current_observation).copy()
        self._history.append(current)
        deltas: list[np.ndarray] = []
        for lag in TEMPORAL_HISTORY_LAGS:
            lagged = self._history[-1 - lag] if len(self._history) > lag else self._history[0]
            delta = 0.5 * (
                current[np.asarray(TEMPORAL_DELTA_SOURCE_INDICES)]
                - lagged[np.asarray(TEMPORAL_DELTA_SOURCE_INDICES)]
            )
            deltas.append(np.asarray(delta, dtype=np.float32))
        return validate_temporal_observation(np.concatenate((current, *deltas)))

    def build(
        self,
        ownship_state,
        target_state,
        bt_action,
        base_guidance,
        **kwargs: Any,
    ) -> np.ndarray:
        current = build_server_guidance_observation(
            ownship_state,
            target_state,
            bt_action,
            base_guidance,
            **kwargs,
        )
        return self.append_observation(current)


def validate_long_temporal_observation(observation: Any) -> np.ndarray:
    vector = np.asarray(observation, dtype=np.float32)
    if vector.shape != (LONG_TEMPORAL_OBSERVATION_SIZE,):
        raise ValueError(
            "long temporal observation must have shape "
            f"({LONG_TEMPORAL_OBSERVATION_SIZE},), got {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("long temporal observation contains nonfinite values")
    return np.clip(vector, -1.0, 1.0)


class LongTemporalServerObservationBuilder:
    """One-variable v4 extension: retain exact server-safe history through t-120."""

    def __init__(self) -> None:
        self._history: deque[np.ndarray] = deque(
            maxlen=max(LONG_TEMPORAL_HISTORY_LAGS) + 1
        )

    @property
    def frames_seen(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        self._history.clear()

    def append_observation(self, current_observation: Any) -> np.ndarray:
        current = validate_server_guidance_observation(current_observation).copy()
        self._history.append(current)
        deltas = []
        for lag in LONG_TEMPORAL_HISTORY_LAGS:
            lagged = self._history[-1 - lag] if len(self._history) > lag else self._history[0]
            deltas.append(
                0.5
                * (
                    current[np.asarray(TEMPORAL_DELTA_SOURCE_INDICES)]
                    - lagged[np.asarray(TEMPORAL_DELTA_SOURCE_INDICES)]
                )
            )
        return validate_long_temporal_observation(np.concatenate((current, *deltas)))

    def build(self, ownship_state, target_state, bt_action, base_guidance, **kwargs: Any):
        current = build_server_guidance_observation(
            ownship_state, target_state, bt_action, base_guidance, **kwargs
        )
        return self.append_observation(current)


def temporal_server_observation_contract() -> dict[str, Any]:
    delta_specs = [
        {
            "name": f"delta_t{lag}_{name}",
            "dtype": "float32",
            "source": f"local deterministic history current minus t-{lag}",
            "unit": "normalized delta",
            "normalization": "(current_normalized-lagged_normalized)/2 clipped [-1,1]",
        }
        for lag in TEMPORAL_HISTORY_LAGS
        for name in TEMPORAL_DELTA_SOURCE_FEATURES
    ]
    return {
        "contract_version": TEMPORAL_SERVER_CONTRACT_VERSION,
        "normalization_version": TEMPORAL_SERVER_NORMALIZATION_VERSION,
        "dtype": "float32",
        "size": TEMPORAL_OBSERVATION_SIZE,
        "current_state_size": GUIDANCE_SERVER_OBSERVATION_SIZE,
        "history_frames": [0, *TEMPORAL_HISTORY_LAGS],
        "startup_padding": TEMPORAL_PADDING,
        "features": [*map(dict, GUIDANCE_SERVER_FEATURE_SPECS), *delta_specs],
        "health_features": [],
        "runtime_sources": [
            "ownship and target server packet location/rotation/velocity",
            "same-frame Pure BT action and VP",
            "local deterministic frame history",
        ],
        "offline_label_only": ["ownship health", "target health", "Damage"],
    }


def long_temporal_server_observation_contract() -> dict[str, Any]:
    contract = temporal_server_observation_contract()
    contract.update(
        {
            "contract_version": LONG_TEMPORAL_SERVER_CONTRACT_VERSION,
            "size": LONG_TEMPORAL_OBSERVATION_SIZE,
            "history_frames": [0, *LONG_TEMPORAL_HISTORY_LAGS],
            "features": [
                *map(dict, GUIDANCE_SERVER_FEATURE_SPECS),
                *(
                    {
                        "name": f"delta_t{lag}_{name}",
                        "dtype": "float32",
                        "source": f"local deterministic history current minus t-{lag}",
                        "unit": "normalized delta",
                        "normalization": "(current_normalized-lagged_normalized)/2 clipped [-1,1]",
                    }
                    for lag in LONG_TEMPORAL_HISTORY_LAGS
                    for name in TEMPORAL_DELTA_SOURCE_FEATURES
                ),
            ],
        }
    )
    return contract
