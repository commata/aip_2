from __future__ import annotations

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.ai.temporal_tactical_provider import TemporalTacticalActionProvider
from dogfight.sim.state_schema import StateIndex


class _BT(ActionProvider):
    def compute_action(self, context) -> ActionResult:
        return ActionResult(
            np.asarray([0.1, -0.2, 0.05, 0.73], dtype=np.float32),
            "bt",
            info={"vp": np.asarray([1000.0, 0.0, 4500.0])},
        )


class _Selector:
    def select(self, observation):
        return {
            "mode": "PURE_PURSUIT",
            "hold_frames": 30,
            "abstention_reason": "",
        }


def _state(*, speed=220.0, target=False):
    value = np.zeros(51, dtype=np.float64)
    value[:3] = [900.0 if target else 0.0, 120.0 if target else 0.0, -4500.0]
    value[6] = speed
    value[StateIndex.KCAS] = speed
    value[StateIndex.ALT] = 4500.0
    value[StateIndex.HEALTH] = 1.0
    return value


def _context(speed=220.0):
    return ActionContext(
        sim=None,
        opponent_sim=None,
        ownship_state=_state(speed=speed),
        target_state=_state(speed=210.0, target=True),
        info={"sim_time_s": 1.0},
    )


def test_shadow_returns_exact_bt_while_recording_nondefault_prediction() -> None:
    provider = TemporalTacticalActionProvider(_BT(), _Selector(), shadow_mode=True)
    result = provider.compute_action(_context())
    np.testing.assert_array_equal(
        result.action, np.asarray([0.1, -0.2, 0.05, 0.73], dtype=np.float32)
    )
    assert result.info["selected_tactical_mode"] == "PURE_PURSUIT"
    assert result.info["exact_bt_command"] is True
    assert provider.telemetry()["nondefault_predictions"] == 1
    assert provider.telemetry()["applied_frames"] == 0


def test_live_tactical_mode_preserves_exact_bt_throttle() -> None:
    provider = TemporalTacticalActionProvider(_BT(), _Selector(), shadow_mode=False)
    result = provider.compute_action(_context())
    assert result.action[3] == np.float32(0.73)
    assert provider.telemetry()["throttle_violations"] == 0


def test_safety_veto_returns_exact_bt_and_records_reason() -> None:
    provider = TemporalTacticalActionProvider(_BT(), _Selector(), shadow_mode=False)
    result = provider.compute_action(_context(speed=100.0))
    np.testing.assert_array_equal(
        result.action, np.asarray([0.1, -0.2, 0.05, 0.73], dtype=np.float32)
    )
    assert result.info["prediction"]["abstention_reason"] == "SAFETY_VETO"
    assert "low_speed" in result.info["safety_veto_reasons"]
