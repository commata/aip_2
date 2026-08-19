from __future__ import annotations

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.ai.prefix_replay import (
    PrefixReplayIntervention,
    PrefixReplayTacticalActionProvider,
    build_prefix_snapshot,
    compare_prefix_snapshots,
)
from dogfight.sim.state_schema import StateIndex


class FakeBT(ActionProvider):
    def __init__(self) -> None:
        self.calls = 0

    def compute_action(self, context: ActionContext) -> ActionResult:
        self.calls += 1
        action = np.array([0.1, -0.2, 0.05, 0.71], dtype=np.float32)
        return ActionResult(action, "bt", info={"vp": [1000.0, 0.0, -4500.0]})


def state(position=(0.0, 0.0, -4500.0)) -> np.ndarray:
    value = np.zeros(51, dtype=np.float64)
    value[:3] = position
    value[6] = 220.0
    value[StateIndex.KCAS] = 220.0
    value[StateIndex.ALT] = -position[2]
    value[StateIndex.HEALTH] = 0.25
    return value


def context(frame: int = 0) -> ActionContext:
    return ActionContext(
        sim=None,
        opponent_sim=None,
        ownship_state=state(),
        target_state=state((900.0, 120.0, -4550.0)),
        info={"sim_time_s": frame / 60.0},
    )


def test_prefix_hash_excludes_health_and_damage() -> None:
    one = context()
    two = context()
    two.ownship_state[StateIndex.HEALTH] = 0.99
    two.target_state[StateIndex.HEALTH] = 0.01
    action = np.array([0.1, -0.2, 0.05, 0.71], dtype=np.float32)
    vp = np.array([1000.0, 0.0, -4500.0])
    left = build_prefix_snapshot(10, one, action, vp)
    right = build_prefix_snapshot(10, two, action, vp)
    assert left["observable_telemetry_hash"] == right["observable_telemetry_hash"]
    assert compare_prefix_snapshots(left, right)["match"] is True


def test_bt_default_override_is_exact_and_bt_runs_every_frame() -> None:
    bt = FakeBT()
    provider = PrefixReplayTacticalActionProvider(
        bt,
        PrefixReplayIntervention(2, 60, "BT_DEFAULT"),
    )
    expected = np.array([0.1, -0.2, 0.05, 0.71], dtype=np.float32)
    for frame in range(8):
        result = provider.compute_action(context(frame))
        assert np.array_equal(result.action, expected)
        assert result.info["exact_bt_command"] is True
    assert bt.calls == 8
    assert provider.telemetry()["intervention_frames"] == 0
    assert provider.telemetry()["prefix_snapshot"]["frame"] == 2


def test_nondefault_runs_only_frozen_window_and_preserves_throttle() -> None:
    bt = FakeBT()
    provider = PrefixReplayTacticalActionProvider(
        bt,
        PrefixReplayIntervention(2, 3, "PURE_PURSUIT"),
    )
    active = []
    for frame in range(8):
        result = provider.compute_action(context(frame))
        active.append(result.info["intervention_active"])
        assert result.action[3] == np.float32(0.71)
    assert active == [False, False, True, True, True, False, False, False]
    telemetry = provider.telemetry()
    assert telemetry["intervention_frames"] == 3
    assert telemetry["throttle_violations"] == 0
    assert bt.calls == 8


def test_prefix_comparison_rejects_one_observable_change() -> None:
    action = np.array([0.1, -0.2, 0.05, 0.71], dtype=np.float32)
    vp = np.array([1000.0, 0.0, -4500.0])
    left_context = context()
    right_context = context()
    right_context.target_state[1] += 1e-3
    left = build_prefix_snapshot(4, left_context, action, vp)
    right = build_prefix_snapshot(4, right_context, action, vp)
    comparison = compare_prefix_snapshots(left, right)
    assert comparison["match"] is False
    assert comparison["reason"] == "target_server_observable_value_mismatch"
