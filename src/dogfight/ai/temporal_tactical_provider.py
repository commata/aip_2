from __future__ import annotations

from collections import Counter
from time import perf_counter
from typing import Any

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action
from dogfight.ai.hybrid_action_provider import SafetyVetoConfig, _body_velocity_to_ned
from dogfight.ai.tactical_advantage import NumpyTemporalTacticalAdvantageSelector
from dogfight.ai.tactical_modes import (
    apply_tactical_mode,
    champion_vp_to_local_setpoint,
    tactical_mode_id,
)
from dogfight.ai.temporal_observation import TemporalServerObservationBuilder
from dogfight.sim.state_schema import StateIndex


class TemporalTacticalActionProvider(ActionProvider):
    """Conservative Tactical selector with exact BT throttle/fallback and fixed holds."""

    def __init__(
        self,
        bt_provider: ActionProvider,
        selector: NumpyTemporalTacticalAdvantageSelector,
        *,
        shadow_mode: bool = False,
        cooldown_frames: int = 30,
        safety_config: SafetyVetoConfig | None = None,
    ) -> None:
        self.bt_provider = bt_provider
        self.selector = selector
        self.shadow_mode = bool(shadow_mode)
        self.cooldown_frames = int(cooldown_frames)
        if self.cooldown_frames < 0:
            raise ValueError("Tactical cooldown must be non-negative")
        self.safety_config = safety_config or SafetyVetoConfig()
        self.safety_config.validate()
        self.history = TemporalServerObservationBuilder()
        self.reset(None)

    def reset(self, context: ActionContext | None = None) -> None:
        self.bt_provider.reset(context)
        self.history.reset()
        self._frame = 0
        self._active_mode = "BT_DEFAULT"
        self._hold_remaining = 0
        self._hold_total = 0
        self._cooldown_remaining = 0
        self._recent_authority_ratio = 1.0
        self._last_prediction: dict[str, Any] = {}
        self._abstentions: Counter[str] = Counter()
        self._latency_ms: list[float] = []
        self._nondefault_predictions = 0
        self._applied_frames = 0
        self._safety_veto_frames = 0
        self._invalid_frames = 0
        self._throttle_violations = 0

    def _safety_veto(
        self, ownship_state: Any, target_state: Any, bt_action: np.ndarray
    ) -> tuple[bool, list[str]]:
        own = np.asarray(ownship_state, dtype=np.float64)
        target = np.asarray(target_state, dtype=np.float64)
        reasons = []
        cfg = self.safety_config
        if own.size <= StateIndex.ALT or target.size <= StateIndex.ALT:
            return True, ["missing_state"]
        if float(own[StateIndex.ALT]) <= cfg.minimum_altitude_m:
            reasons.append("low_altitude")
        if float(own[StateIndex.KCAS]) <= cfg.minimum_speed_m_s:
            reasons.append("low_speed")
        relative = target[:3] - own[:3]
        distance = float(np.linalg.norm(relative))
        if distance > 1e-9:
            closing = -float(
                np.dot(
                    _body_velocity_to_ned(target) - _body_velocity_to_ned(own),
                    relative / distance,
                )
            )
            if closing > cfg.maximum_closing_rate_m_s:
                reasons.append("high_closure")
        if cfg.veto_if_all_surfaces_saturated and np.all(
            np.isclose(np.abs(bt_action[:3]), 1.0, atol=1e-6)
        ):
            reasons.append("no_surface_authority")
        return bool(reasons), reasons

    def _build_observation(
        self,
        context: ActionContext,
        bt_action: np.ndarray,
        bt_vp: np.ndarray,
    ) -> np.ndarray:
        base = champion_vp_to_local_setpoint(bt_vp, context.ownship_state)
        return self.history.build(
            context.ownship_state,
            context.target_state,
            bt_action,
            base,
            sim_time_s=float(context.info.get("sim_time_s", self._frame / 60.0)),
            previous_action_id=tactical_mode_id(self._active_mode),
            action_hold_frames=max(0, self._hold_total - self._hold_remaining),
            gate_elapsed_frames=max(0, self._hold_total - self._hold_remaining),
            gate_active=self._active_mode != "BT_DEFAULT",
            minimum_action_hold_frames=30,
            maximum_active_frames=120,
            recent_authority_ratio=self._recent_authority_ratio,
            safety_config=self.safety_config,
        )

    def compute_action(self, context: ActionContext) -> ActionResult:
        started = perf_counter()
        self._frame += 1
        bt_result = self.bt_provider.compute_action(context)
        bt_action = clip_action(bt_result.action)
        bt_vp = np.asarray(bt_result.info.get("vp"), dtype=np.float64)
        if bt_vp.shape != (3,) or not np.all(np.isfinite(bt_vp)):
            own = np.asarray(context.ownship_state, dtype=np.float64)
            bt_vp = np.asarray([own[0], own[1], own[StateIndex.ALT]], dtype=np.float64)
        veto, veto_reasons = self._safety_veto(
            context.ownship_state, context.target_state, bt_action
        )
        try:
            observation = self._build_observation(context, bt_action, bt_vp)
            if veto:
                self._safety_veto_frames += 1
                self._active_mode = "BT_DEFAULT"
                self._hold_remaining = 0
                prediction = {
                    "mode": "BT_DEFAULT",
                    "hold_frames": 0,
                    "abstention_reason": "SAFETY_VETO",
                    "safety_veto_reasons": veto_reasons,
                }
            elif self._hold_remaining > 0:
                prediction = {
                    "mode": self._active_mode,
                    "hold_frames": self._hold_remaining,
                    "abstention_reason": "",
                    "held_selection": True,
                }
            elif self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
                self._active_mode = "BT_DEFAULT"
                prediction = {
                    "mode": "BT_DEFAULT",
                    "hold_frames": 0,
                    "abstention_reason": "COOLDOWN",
                }
            else:
                prediction = self.selector.select(observation)
                if prediction["mode"] != "BT_DEFAULT":
                    self._active_mode = str(prediction["mode"])
                    self._hold_total = int(prediction["hold_frames"])
                    self._hold_remaining = self._hold_total
                    self._nondefault_predictions += 1
                else:
                    self._active_mode = "BT_DEFAULT"
            self._last_prediction = dict(prediction)
        except Exception as exc:
            self._invalid_frames += 1
            self._active_mode = "BT_DEFAULT"
            self._hold_remaining = 0
            prediction = {
                "mode": "BT_DEFAULT",
                "hold_frames": 0,
                "abstention_reason": "INVALID",
                "error": f"{type(exc).__name__}:{exc}",
            }
            self._last_prediction = dict(prediction)
        reason = str(prediction.get("abstention_reason", ""))
        if reason:
            self._abstentions[reason] += 1
        final = bt_action.copy()
        tactical_info: dict[str, Any] = {}
        if self._active_mode != "BT_DEFAULT" and self._hold_remaining > 0:
            candidate, tactical_info = apply_tactical_mode(
                self._active_mode,
                bt_action,
                bt_vp,
                context.ownship_state,
                context.target_state,
            )
            ratio = tactical_info.get("controller", {}).get(
                "applied_to_requested_ratio", [1.0, 1.0, 1.0]
            )
            self._recent_authority_ratio = float(np.mean(ratio))
            if not self.shadow_mode:
                final = candidate
                self._applied_frames += int(not tactical_info.get("fallback", False))
            self._hold_remaining -= 1
            if self._hold_remaining == 0:
                self._active_mode = "BT_DEFAULT"
                self._cooldown_remaining = self.cooldown_frames
        if final[3] != bt_action[3]:
            self._throttle_violations += 1
            final = bt_action.copy()
        latency_ms = (perf_counter() - started) * 1000.0
        self._latency_ms.append(latency_ms)
        exact_bt = bool(np.array_equal(final, bt_action))
        return ActionResult(
            final,
            "temporal_tactical_shadow" if self.shadow_mode else "temporal_tactical",
            1.0,
            {
                "mode": "temporal_tactical_v4",
                "shadow_mode": self.shadow_mode,
                "prediction": prediction,
                "selected_tactical_mode": prediction["mode"],
                "bt_action": bt_action.tolist(),
                "final_action": final.tolist(),
                "bt_vp": bt_vp.tolist(),
                "tactical": tactical_info,
                "safety_veto": veto,
                "safety_veto_reasons": veto_reasons,
                "throttle_bt_only": True,
                "exact_bt_command": exact_bt,
                "latency_ms": latency_ms,
            },
        )

    def telemetry(self) -> dict[str, Any]:
        latency = np.asarray(self._latency_ms, dtype=np.float64)
        return {
            "contract": "temporal_tactical_runtime_v4",
            "shadow_mode": self.shadow_mode,
            "frames": self._frame,
            "nondefault_predictions": self._nondefault_predictions,
            "applied_frames": self._applied_frames,
            "safety_veto_frames": self._safety_veto_frames,
            "invalid_frames": self._invalid_frames,
            "throttle_violations": self._throttle_violations,
            "abstention_reasons": dict(self._abstentions),
            "last_prediction": dict(self._last_prediction),
            "latency_ms": {
                "p50": float(np.quantile(latency, 0.50)) if latency.size else 0.0,
                "p95": float(np.quantile(latency, 0.95)) if latency.size else 0.0,
                "p99": float(np.quantile(latency, 0.99)) if latency.size else 0.0,
                "max": float(np.max(latency)) if latency.size else 0.0,
                "over_166_7ms": int(np.sum(latency > 166.7)),
            },
        }

    def close(self) -> None:
        self.bt_provider.close()
