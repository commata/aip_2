from __future__ import annotations

import json

import numpy as np

from dogfight.ai.guidance_advantage import build_server_guidance_observation
from dogfight.ai.guidance_selector import GuidanceSetpoint
from dogfight.ai.temporal_observation import (
    TEMPORAL_DELTA_SOURCE_FEATURES,
    TEMPORAL_FEATURES,
    TEMPORAL_OBSERVATION_SIZE,
    TemporalServerObservationBuilder,
    temporal_server_observation_contract,
)
from dogfight.sim.state_schema import StateIndex
from dogfight.unreal.policies import plane_info_to_state
from dogfight.unreal.protocol import (
    PlaneInfo,
    Rotation3D,
    Vector3D,
    pack_plane_info,
    unpack_plane_info,
)


def _plane(index: int, *, target: bool = False) -> PlaneInfo:
    offset = float(index)
    return PlaneInfo(
        index=index,
        plane_id=2 if target else 1,
        position=Vector3D(700.0 + offset if target else offset, 30.0, 5000.0),
        rotation=Rotation3D(1.0, -2.0, 4.0 if target else 359.0),
        velocity=Vector3D(215.0 if target else 230.0, -3.0 if target else 2.0, 0.0),
    )


def _kwargs(frame: int) -> dict:
    return {
        "sim_time_s": frame / 60.0,
        "previous_action_id": 0,
        "action_hold_frames": 0,
        "gate_elapsed_frames": frame,
        "gate_active": True,
        "minimum_action_hold_frames": 30,
        "maximum_active_frames": 120,
    }


def _current(frame: int, *, packet_roundtrip: bool = False) -> np.ndarray:
    own = _plane(frame)
    target = _plane(frame, target=True)
    if packet_roundtrip:
        own = unpack_plane_info(pack_plane_info(own))
        target = unpack_plane_info(pack_plane_info(target))
    return build_server_guidance_observation(
        plane_info_to_state(own),
        plane_info_to_state(target),
        np.asarray([0.1, -0.2, 0.05, 0.9], dtype=np.float32),
        GuidanceSetpoint(1.0, -0.5, 800.0, 230.0),
        **_kwargs(frame),
    )


def test_temporal_contract_is_93d_server_safe_and_frozen() -> None:
    contract = temporal_server_observation_contract()
    assert TEMPORAL_OBSERVATION_SIZE == 93
    assert len(TEMPORAL_FEATURES) == 93
    assert contract["history_frames"] == [0, 6, 12, 30]
    assert contract["startup_padding"] == "repeat_first_zero_delta"
    assert contract["health_features"] == []
    assert all("health" not in feature.lower() for feature in TEMPORAL_FEATURES)


def test_startup_padding_repeats_first_and_produces_zero_delta() -> None:
    vector = TemporalServerObservationBuilder().append_observation(_current(0))
    assert vector.dtype == np.float32
    np.testing.assert_array_equal(vector[42:], np.zeros(51, dtype=np.float32))


def test_history_lags_use_exact_frames() -> None:
    builder = TemporalServerObservationBuilder()
    result = None
    currents = []
    for frame in range(31):
        current = _current(frame)
        currents.append(current)
        result = builder.append_observation(current)
    assert result is not None
    indices = [TEMPORAL_FEATURES.index(name) for name in TEMPORAL_DELTA_SOURCE_FEATURES]
    expected = np.concatenate(
        [0.5 * (currents[30][indices] - currents[30 - lag][indices]) for lag in (6, 12, 30)]
    )
    np.testing.assert_array_equal(result[42:], expected.astype(np.float32))


def test_live_packet_pack_replay_and_recorded_replay_are_byte_identical() -> None:
    live_builder = TemporalServerObservationBuilder()
    packet_builder = TemporalServerObservationBuilder()
    recorded_builder = TemporalServerObservationBuilder()
    recorded = []
    live_vectors = []
    packet_vectors = []
    for frame in range(31):
        live_current = _current(frame)
        packet_current = _current(frame, packet_roundtrip=True)
        recorded.append(live_current.tolist())
        live_vectors.append(live_builder.append_observation(live_current))
        packet_vectors.append(packet_builder.append_observation(packet_current))
    serialized = json.loads(json.dumps(recorded))
    replay_vectors = [recorded_builder.append_observation(item) for item in serialized]
    assert all(a.tobytes() == b.tobytes() for a, b in zip(live_vectors, packet_vectors))
    assert all(a.tobytes() == b.tobytes() for a, b in zip(live_vectors, replay_vectors))


def test_health_is_not_read() -> None:
    own = plane_info_to_state(_plane(0))
    target = plane_info_to_state(_plane(0, target=True))
    own_changed = own.copy()
    target_changed = target.copy()
    own_changed[StateIndex.HEALTH] = -999.0
    target_changed[StateIndex.HEALTH] = 123.0
    bt = np.asarray([0.1, -0.2, 0.05, 0.9], dtype=np.float32)
    base = GuidanceSetpoint(1.0, -0.5, 800.0, 230.0)
    first = TemporalServerObservationBuilder().build(own, target, bt, base, **_kwargs(0))
    second = TemporalServerObservationBuilder().build(
        own_changed, target_changed, bt, base, **_kwargs(0)
    )
    assert first.tobytes() == second.tobytes()
