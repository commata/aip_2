from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action


def _unsigned_ata_deg(observer_state, target_state) -> float:
    """Return unsigned 3-D antenna train angle in NED coordinates."""
    observer = np.asarray(observer_state, dtype=np.float64)
    target = np.asarray(target_state, dtype=np.float64)
    line = target[:3] - observer[:3]
    distance = float(np.linalg.norm(line))
    if distance <= 1e-9:
        return 0.0
    roll, pitch, yaw = np.radians(observer[3:6])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    forward = rotation @ np.array([1.0, 0.0, 0.0])
    cosine = float(np.dot(forward, line / distance))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


@dataclass(frozen=True)
class OffensiveGateConfig:
    min_range_m: float = 152.4
    enter_max_range_m: float = 2400.0
    exit_max_range_m: float = 3000.0
    enter_ata_deg: float = 30.0
    exit_ata_deg: float = 45.0
    enter_min_target_ata_deg: float = 105.0
    exit_min_target_ata_deg: float = 80.0

    def validate(self) -> None:
        if self.min_range_m < 0.0:
            raise ValueError("offensive min range must be non-negative")
        if self.enter_max_range_m <= self.min_range_m:
            raise ValueError("offensive enter range must exceed min range")
        if self.exit_max_range_m < self.enter_max_range_m:
            raise ValueError("offensive exit range must cover enter range")
        if not 0.0 <= self.enter_ata_deg <= self.exit_ata_deg <= 180.0:
            raise ValueError("offensive ATA must satisfy 0 <= enter <= exit <= 180")
        if not 0.0 <= self.exit_min_target_ata_deg <= self.enter_min_target_ata_deg <= 180.0:
            raise ValueError(
                "target ATA must satisfy 0 <= exit <= enter <= 180"
            )


class OffensiveResidualGate:
    """Hysteretic gate that exposes RL only during an offensive setup."""

    def __init__(self, config: OffensiveGateConfig | dict | None = None):
        if config is None:
            config = OffensiveGateConfig()
        elif isinstance(config, dict):
            config = OffensiveGateConfig(**config)
        config.validate()
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.steps = 0
        self.active_steps = 0
        self.entries = 0
        self.exits = 0
        self.last_geometry: dict[str, float | bool] = {
            "distance_m": float("nan"),
            "ata_deg": float("nan"),
            "target_ata_deg": float("nan"),
            "active": False,
            "entry": False,
            "exit": False,
        }

    def update(self, ownship_state, target_state) -> dict[str, float | bool]:
        previous = self.active
        if ownship_state is None or target_state is None:
            next_active = False
            distance = ata = target_ata = float("nan")
        else:
            own = np.asarray(ownship_state, dtype=np.float64)
            target = np.asarray(target_state, dtype=np.float64)
            distance = float(np.linalg.norm(target[:3] - own[:3]))
            ata = _unsigned_ata_deg(own, target)
            target_ata = _unsigned_ata_deg(target, own)
            cfg = self.config
            if previous:
                next_active = (
                    cfg.min_range_m <= distance <= cfg.exit_max_range_m
                    and ata <= cfg.exit_ata_deg
                    and target_ata >= cfg.exit_min_target_ata_deg
                )
            else:
                next_active = (
                    cfg.min_range_m <= distance <= cfg.enter_max_range_m
                    and ata <= cfg.enter_ata_deg
                    and target_ata >= cfg.enter_min_target_ata_deg
                )

        entry = bool(next_active and not previous)
        exit_event = bool(previous and not next_active)
        self.entries += int(entry)
        self.exits += int(exit_event)
        self.active = bool(next_active)
        self.steps += 1
        self.active_steps += int(self.active)
        self.last_geometry = {
            "distance_m": distance,
            "ata_deg": ata,
            "target_ata_deg": target_ata,
            "active": self.active,
            "entry": entry,
            "exit": exit_event,
        }
        return dict(self.last_geometry)

    def telemetry(self) -> dict[str, int | float | bool | dict]:
        return {
            "offensive_gate_config": asdict(self.config),
            "offensive_gate_steps": self.steps,
            "offensive_gate_active_steps": self.active_steps,
            "offensive_gate_active_ratio": self.active_steps / max(1, self.steps),
            "offensive_gate_entries": self.entries,
            "offensive_gate_exits": self.exits,
            "offensive_gate_active_final": self.active,
        }


def _compose_residual(bt_action, rl_action, scale: float) -> tuple[np.ndarray, dict]:
    """Add signed surface corrections and convex-blend simulator throttle."""
    bt = clip_action(bt_action)
    rl = clip_action(rl_action)
    unclipped = bt.copy()
    unclipped[:3] = bt[:3] + scale * rl[:3]
    unclipped[3] = (1.0 - scale) * bt[3] + scale * rl[3]
    final = clip_action(unclipped)
    correction = final - bt
    return final, {
        "bt_action": bt.tolist(),
        "raw_rl_action": rl.tolist(),
        "applied_rl_correction": correction.tolist(),
        "final_action": final.tolist(),
        "action_clipped": bool(np.any(np.abs(final - unclipped) > 1e-7)),
        "action_saturation": bool(np.any(np.isclose(np.abs(final[:3]), 1.0))),
        "throttle_at_boundary": bool(np.isclose(final[3], 0.0) or np.isclose(final[3], 1.0)),
    }


class HybridActionProvider(ActionProvider):
    def __init__(
        self,
        primary_provider: ActionProvider,
        secondary_provider: ActionProvider,
        mode: str = "residual",
        alpha: float = 0.5,
        residual_scale: float = 0.15,
        offensive_gate: OffensiveGateConfig | dict | None = None,
        primary_action_repeat: int = 1,
        selector: Callable[[ActionContext, ActionResult, ActionResult], str | bool] | None = None,
        confidence: float = 0.95,
    ):
        if mode == "offensive_residual" and not 0.10 <= residual_scale <= 0.20:
            raise ValueError("offensive residual scale must be within [0.10, 0.20]")
        self.primary_provider = primary_provider
        self.secondary_provider = secondary_provider
        self.mode = mode
        self.alpha = float(alpha)
        self.residual_scale = float(residual_scale)
        self.selector = selector
        self.confidence = confidence
        self.offensive_gate = OffensiveResidualGate(offensive_gate)
        self.primary_action_repeat = max(1, int(primary_action_repeat))
        self.reset(None)

    def reset(self, context: ActionContext | None = None) -> None:
        self.primary_provider.reset(context)
        self.secondary_provider.reset(context)
        self.offensive_gate.reset()
        self._active_frames = 0
        self._cached_primary_action: np.ndarray | None = None
        self._rl_inference_calls = 0
        self._rl_correction_steps = 0
        self._rl_correction_abs_sum = np.zeros(4, dtype=np.float64)
        self._rl_correction_abs_max = np.zeros(4, dtype=np.float64)
        self._clipped_steps = 0
        self._saturated_steps = 0
        self._last_frame_info: dict = {}

    def compute_action(self, context: ActionContext) -> ActionResult:
        secondary = self.secondary_provider.compute_action(context)
        if self.mode == "offensive_residual":
            return self._compute_offensive(context, secondary)

        primary = self.primary_provider.compute_action(context)
        if self.mode == "switch":
            decision = self.selector(context, primary, secondary) if self.selector else "primary"
            use_primary = decision if isinstance(decision, bool) else decision != "secondary"
            chosen = primary if use_primary else secondary
            return ActionResult(
                action=clip_action(chosen.action), source="hybrid", confidence=self.confidence,
                info={"mode": self.mode, "selected": chosen.source},
            )
        if self.mode == "blend":
            action = self.alpha * primary.action + (1.0 - self.alpha) * secondary.action
        else:
            action, _ = _compose_residual(secondary.action, primary.action, self.residual_scale)
        return ActionResult(
            action=clip_action(action), source="hybrid", confidence=self.confidence,
            info={"mode": self.mode, "alpha": self.alpha, "residual_scale": self.residual_scale},
        )

    def _compute_offensive(self, context: ActionContext, secondary: ActionResult) -> ActionResult:
        gate = self.offensive_gate.update(context.ownship_state, context.target_state)
        if not gate["active"]:
            self._cached_primary_action = None
            self._active_frames = 0
            final = clip_action(secondary.action)
            frame = {
                "mode": self.mode,
                "selected": secondary.source,
                "offensive_gate": gate,
                "effective_residual_scale": 0.0,
                "primary_action_refreshed": False,
                "bt_action": final.tolist(),
                "raw_rl_action": None,
                "applied_rl_correction": [0.0] * 4,
                "final_action": final.tolist(),
                "action_clipped": False,
                "action_saturation": bool(np.any(np.isclose(np.abs(final[:3]), 1.0))),
                "throttle_at_boundary": bool(np.isclose(final[3], 0.0) or np.isclose(final[3], 1.0)),
            }
            self._last_frame_info = frame
            return ActionResult(final, "hybrid", self.confidence, frame)

        if gate["entry"]:
            self.primary_provider.reset(context)
        refresh = (
            self._cached_primary_action is None
            or self._active_frames % self.primary_action_repeat == 0
        )
        if refresh:
            primary = self.primary_provider.compute_action(context)
            self._cached_primary_action = clip_action(primary.action)
            primary_source = primary.source
            self._rl_inference_calls += 1
        else:
            primary_source = "cached_rl"
        final, composition = _compose_residual(
            secondary.action, self._cached_primary_action, self.residual_scale
        )
        correction = np.asarray(composition["applied_rl_correction"])
        self._rl_correction_steps += 1
        self._rl_correction_abs_sum += np.abs(correction)
        self._rl_correction_abs_max = np.maximum(self._rl_correction_abs_max, np.abs(correction))
        self._clipped_steps += int(composition["action_clipped"])
        self._saturated_steps += int(composition["action_saturation"])
        self._active_frames += 1
        frame = {
            "mode": self.mode,
            "selected": "bt+rl",
            "offensive_gate": gate,
            "effective_residual_scale": self.residual_scale,
            "primary_source": primary_source,
            "primary_action_refreshed": refresh,
            "primary_action_repeat": self.primary_action_repeat,
            **composition,
        }
        self._last_frame_info = frame
        return ActionResult(final, "hybrid", self.confidence, frame)

    def telemetry(self) -> dict:
        result = self.offensive_gate.telemetry()
        result.update(
            {
                "rl_correction_steps": self._rl_correction_steps,
                "rl_inference_calls": self._rl_inference_calls,
                "rl_action_repeat": self.primary_action_repeat,
                "rl_correction_abs_mean": (
                    self._rl_correction_abs_sum / max(1, self._rl_correction_steps)
                ).tolist(),
                "rl_correction_abs_max": self._rl_correction_abs_max.tolist(),
                "action_clipped_steps": self._clipped_steps,
                "action_saturated_steps": self._saturated_steps,
                "last_frame": dict(self._last_frame_info),
            }
        )
        return result

    def close(self) -> None:
        self.primary_provider.close()
        self.secondary_provider.close()
