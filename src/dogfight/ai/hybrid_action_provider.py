from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action
from dogfight.sim.state_schema import StateIndex


ALLOWED_AIM_RESIDUAL_SCALES = (0.10, 0.125, 0.15)
RESIDUAL_COMPOSITION_MODES = ("additive", "saturation_aware")
SURFACE_AXES = ("roll", "pitch", "yaw")
RESIDUAL_AXIS_MASKS = {
    "roll": (1.0, 0.0, 0.0, 0.0),
    "pitch": (0.0, 1.0, 0.0, 0.0),
    "yaw": (0.0, 0.0, 1.0, 0.0),
    "pitch_yaw": (0.0, 1.0, 1.0, 0.0),
    "roll_pitch_yaw": (1.0, 1.0, 1.0, 0.0),
}


def _compose_aim_surface_residual(
    bt_action: np.ndarray,
    residual: np.ndarray,
    residual_scale: float,
    composition_mode: str,
) -> np.ndarray:
    """Return an unclipped surface command while bounding correction by scale."""
    result = np.asarray(bt_action, dtype=np.float32).copy()
    if composition_mode == "additive":
        result[:3] = result[:3] + residual_scale * residual[:3]
        return result
    if composition_mode != "saturation_aware":
        raise ValueError(f"unsupported residual composition: {composition_mode!r}")
    surfaces = result[:3]
    requested = np.asarray(residual[:3], dtype=np.float32)
    available = np.where(requested >= 0.0, 1.0 - surfaces, surfaces + 1.0)
    authority = np.clip(available, 0.0, 1.0)
    result[:3] = surfaces + residual_scale * requested * authority
    return result


def _surface_authority_diagnostics(
    bt_action: np.ndarray,
    residual: np.ndarray | None,
    final_action: np.ndarray,
    residual_scale: float,
    *,
    active: bool,
) -> dict:
    """Expose clipping-independent surface headroom and realised authority."""
    bt = np.asarray(bt_action, dtype=np.float64)[:3]
    final = np.asarray(final_action, dtype=np.float64)[:3]
    raw = (
        np.asarray(residual, dtype=np.float64)[:3]
        if residual is not None and active
        else np.zeros(3, dtype=np.float64)
    )
    requested = float(residual_scale) * raw
    applied = final - bt
    positive = np.clip(1.0 - bt, 0.0, 2.0)
    negative = np.clip(bt + 1.0, 0.0, 2.0)
    directional = np.where(requested >= 0.0, positive, negative)
    nonzero = np.abs(requested) > 1e-12
    applied_to_requested = np.zeros(3, dtype=np.float64)
    applied_to_requested[nonzero] = (
        np.abs(applied[nonzero]) / np.abs(requested[nonzero])
    )
    requested_to_applied = np.zeros(3, dtype=np.float64)
    realised = np.abs(applied) > 1e-12
    requested_to_applied[realised] = (
        np.abs(requested[realised]) / np.abs(applied[realised])
    )
    requested_to_applied_values = [
        None if nonzero[index] and not realised[index] else float(requested_to_applied[index])
        for index in range(3)
    ]
    return {
        "positive_headroom": positive.tolist(),
        "negative_headroom": negative.tolist(),
        "directional_headroom": directional.tolist(),
        "requested_surface_correction": requested.tolist(),
        "applied_surface_correction": applied.tolist(),
        "applied_to_requested_ratio": applied_to_requested.tolist(),
        "requested_to_applied_ratio": requested_to_applied_values,
        "request_nonzero": nonzero.tolist(),
        "bt_surface_saturated": np.isclose(np.abs(bt), 1.0, atol=1e-6).tolist(),
        "final_surface_saturated": np.isclose(
            np.abs(final), 1.0, atol=1e-6
        ).tolist(),
    }


def _reset_authority_counters(provider) -> None:
    provider._bt_saturated_steps_axis = np.zeros(3, dtype=np.int64)
    provider._final_saturated_steps_axis = np.zeros(3, dtype=np.int64)
    provider._positive_headroom_sum = np.zeros(3, dtype=np.float64)
    provider._negative_headroom_sum = np.zeros(3, dtype=np.float64)
    provider._directional_headroom_sum = np.zeros(3, dtype=np.float64)
    provider._requested_correction_abs_sum = np.zeros(3, dtype=np.float64)
    provider._applied_correction_abs_sum = np.zeros(3, dtype=np.float64)
    provider._authority_ratio_sum = np.zeros(3, dtype=np.float64)
    provider._authority_ratio_count = np.zeros(3, dtype=np.int64)


def _update_authority_counters(provider, diagnostics: dict) -> None:
    provider._bt_saturated_steps_axis += np.asarray(
        diagnostics["bt_surface_saturated"], dtype=np.int64
    )
    provider._final_saturated_steps_axis += np.asarray(
        diagnostics["final_surface_saturated"], dtype=np.int64
    )
    provider._positive_headroom_sum += diagnostics["positive_headroom"]
    provider._negative_headroom_sum += diagnostics["negative_headroom"]
    provider._directional_headroom_sum += diagnostics["directional_headroom"]
    provider._requested_correction_abs_sum += np.abs(
        diagnostics["requested_surface_correction"]
    )
    provider._applied_correction_abs_sum += np.abs(
        diagnostics["applied_surface_correction"]
    )
    nonzero = np.asarray(diagnostics["request_nonzero"], dtype=bool)
    provider._authority_ratio_sum[nonzero] += np.asarray(
        diagnostics["applied_to_requested_ratio"], dtype=np.float64
    )[nonzero]
    provider._authority_ratio_count += nonzero.astype(np.int64)


def _authority_telemetry(provider, active_steps: int) -> dict:
    denominator = max(1, int(active_steps))
    ratio_denominator = np.maximum(1, provider._authority_ratio_count)
    return {
        "surface_axis_names": list(SURFACE_AXES),
        "bt_surface_saturation_ratio_axis": (
            provider._bt_saturated_steps_axis / denominator
        ).tolist(),
        "final_surface_saturation_ratio_axis": (
            provider._final_saturated_steps_axis / denominator
        ).tolist(),
        "positive_headroom_mean_axis": (
            provider._positive_headroom_sum / denominator
        ).tolist(),
        "negative_headroom_mean_axis": (
            provider._negative_headroom_sum / denominator
        ).tolist(),
        "directional_headroom_mean_axis": (
            provider._directional_headroom_sum / denominator
        ).tolist(),
        "requested_surface_correction_abs_mean_axis": (
            provider._requested_correction_abs_sum / denominator
        ).tolist(),
        "applied_surface_correction_abs_mean_axis": (
            provider._applied_correction_abs_sum / denominator
        ).tolist(),
        "applied_to_requested_ratio_mean_axis": (
            provider._authority_ratio_sum / ratio_denominator
        ).tolist(),
        "authority_ratio_samples_axis": provider._authority_ratio_count.tolist(),
    }


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


def target_ata_deg(ownship_state, target_state) -> float:
    """Return the unsigned angle from target nose to the target→ownship line."""
    return _unsigned_ata_deg(target_state, ownship_state)


@dataclass(frozen=True)
class Rear120GateConfig:
    enter_target_ata_deg: float = 120.0
    exit_target_ata_deg: float = 110.0
    sim_hz: int = 60

    def validate(self) -> None:
        if not 0.0 <= self.exit_target_ata_deg <= self.enter_target_ata_deg <= 180.0:
            raise ValueError("rear gate must satisfy 0 <= exit <= enter <= 180")
        if self.sim_hz <= 0:
            raise ValueError("rear gate sim_hz must be positive")


class Rear120EligibilityGate:
    """Hard hysteretic target rear-sector envelope based on unsigned target ATA."""

    def __init__(self, config: Rear120GateConfig | dict | None = None):
        if config is None:
            config = Rear120GateConfig()
        elif isinstance(config, dict):
            config = Rear120GateConfig(**config)
        config.validate()
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.steps = 0
        self.active_steps = 0
        self.entries = 0
        self.exits = 0
        self._current_active_steps = 0
        self._completed_active_steps: list[int] = []
        self.last_geometry = {
            "target_ata_deg": float("nan"),
            "active": False,
            "entry": False,
            "exit": False,
        }

    def update(self, ownship_state, target_state) -> dict:
        previous = self.active
        if ownship_state is None or target_state is None:
            angle = float("nan")
            next_active = False
        else:
            angle = target_ata_deg(ownship_state, target_state)
            threshold = (
                self.config.exit_target_ata_deg
                if previous
                else self.config.enter_target_ata_deg
            )
            next_active = bool(angle >= threshold)
        entry = bool(next_active and not previous)
        exit_event = bool(previous and not next_active)
        self.active = bool(next_active)
        self.steps += 1
        self.active_steps += int(self.active)
        self.entries += int(entry)
        self.exits += int(exit_event)
        if self.active:
            self._current_active_steps += 1
        elif exit_event:
            self._completed_active_steps.append(self._current_active_steps)
            self._current_active_steps = 0
        self.last_geometry = {
            "target_ata_deg": angle,
            "active": self.active,
            "entry": entry,
            "exit": exit_event,
        }
        return dict(self.last_geometry)

    def telemetry(self) -> dict:
        durations = list(self._completed_active_steps)
        if self.active and self._current_active_steps:
            durations.append(self._current_active_steps)
        elapsed_min = self.steps / float(self.config.sim_hz) / 60.0
        return {
            "rear120_gate_config": asdict(self.config),
            "rear120_gate_steps": self.steps,
            "rear120_gate_active_steps": self.active_steps,
            "rear120_gate_active_ratio": self.active_steps / max(1, self.steps),
            "rear120_gate_entries": self.entries,
            "rear120_gate_exits": self.exits,
            "rear120_gate_boundary_reentries": max(0, self.entries - 1),
            "rear120_gate_transitions_per_min": (
                (self.entries + self.exits) / elapsed_min if elapsed_min > 0.0 else 0.0
            ),
            "rear120_gate_mean_active_steps": (
                float(np.mean(durations)) if durations else 0.0
            ),
            "rear120_gate_min_active_steps": min(durations) if durations else 0,
            "rear120_gate_active_final": self.active,
        }


@dataclass(frozen=True)
class SafetyVetoConfig:
    minimum_altitude_m: float = 350.0
    minimum_speed_m_s: float = 170.0
    maximum_closing_rate_m_s: float = 250.0
    veto_if_all_surfaces_saturated: bool = True

    def validate(self) -> None:
        if self.minimum_altitude_m < 0.0 or self.minimum_speed_m_s < 0.0:
            raise ValueError("safety altitude and speed thresholds must be non-negative")
        if self.maximum_closing_rate_m_s <= 0.0:
            raise ValueError("maximum closing rate must be positive")


class Rear120ActivationGate:
    """rear120 AND (offensive OR phase pre-aim) AND NOT safety veto."""

    def __init__(
        self,
        rear120_config: Rear120GateConfig | dict | None = None,
        aim_config: AimGateConfig | dict | None = None,
        offensive_config: OffensiveGateConfig | dict | None = None,
        safety_config: SafetyVetoConfig | dict | None = None,
    ):
        self.rear = Rear120EligibilityGate(rear120_config)
        self.aim = AimResidualGate(aim_config)
        self.offensive = OffensiveResidualGate(offensive_config)
        if safety_config is None:
            safety_config = SafetyVetoConfig()
        elif isinstance(safety_config, dict):
            safety_config = SafetyVetoConfig(**safety_config)
        safety_config.validate()
        self.safety_config = safety_config
        self.reset()

    def reset(self) -> None:
        self.rear.reset()
        self.aim.reset()
        self.offensive.reset()
        self.active = False
        self.steps = 0
        self.active_steps = 0
        self.entries = 0
        self.exits = 0
        self.safety_veto_steps = 0
        self._current_active_steps = 0
        self._active_durations: list[int] = []
        self.last_geometry = {"active": False, "entry": False, "exit": False}

    def update(
        self,
        ownship_state,
        target_state,
        *,
        sim_time_s: float | None = None,
        bt_action=None,
    ) -> dict:
        previous = self.active
        rear = self.rear.update(ownship_state, target_state)
        aim = self.aim.update(ownship_state, target_state, sim_time_s=sim_time_s)
        offensive = self.offensive.update(ownship_state, target_state)
        veto, veto_reasons = self._safety_veto(
            ownship_state,
            target_state,
            bt_action,
        )
        self.active = bool(
            rear["active"]
            and (offensive["active"] or aim["active"])
            and not veto
        )
        entry = bool(self.active and not previous)
        exit_event = bool(previous and not self.active)
        self.steps += 1
        self.active_steps += int(self.active)
        self.entries += int(entry)
        self.exits += int(exit_event)
        self.safety_veto_steps += int(veto)
        if self.active:
            self._current_active_steps += 1
        elif exit_event:
            self._active_durations.append(self._current_active_steps)
            self._current_active_steps = 0
        self.last_geometry = {
            "distance_m": aim["distance_m"],
            "aim_error_deg": aim["aim_error_deg"],
            "ata_deg": offensive["ata_deg"],
            "target_ata_deg": rear["target_ata_deg"],
            "phase": aim["phase"],
            "rear120_eligible": rear["active"],
            "offensive_eligible": offensive["active"],
            "pre_aim_eligible": aim["active"],
            "safety_veto": veto,
            "safety_veto_reasons": veto_reasons,
            "active": self.active,
            "entry": entry,
            "exit": exit_event,
            "rear120_gate": rear,
            "aim_gate": aim,
            "offensive_gate": offensive,
        }
        return dict(self.last_geometry)

    def _safety_veto(self, ownship_state, target_state, bt_action) -> tuple[bool, list[str]]:
        if ownship_state is None or target_state is None:
            return True, ["missing_state"]
        own = np.asarray(ownship_state, dtype=np.float64)
        target = np.asarray(target_state, dtype=np.float64)
        cfg = self.safety_config
        reasons: list[str] = []
        if float(own[StateIndex.ALT]) <= cfg.minimum_altitude_m:
            reasons.append("low_altitude")
        if float(own[StateIndex.KCAS]) <= cfg.minimum_speed_m_s:
            reasons.append("low_speed")
        line = target[:3] - own[:3]
        distance = float(np.linalg.norm(line))
        if distance > 1e-9:
            own_velocity = _body_velocity_to_ned(own)
            target_velocity = _body_velocity_to_ned(target)
            closing = -float(np.dot(target_velocity - own_velocity, line / distance))
            if closing > cfg.maximum_closing_rate_m_s:
                reasons.append("high_closure")
        if cfg.veto_if_all_surfaces_saturated and bt_action is not None:
            surfaces = np.asarray(bt_action, dtype=np.float64)[:3]
            if bool(np.all(np.isclose(np.abs(surfaces), 1.0, atol=1e-6))):
                reasons.append("no_surface_authority")
        return bool(reasons), reasons

    def telemetry(self) -> dict:
        result = self.rear.telemetry()
        durations = list(self._active_durations)
        if self.active and self._current_active_steps:
            durations.append(self._current_active_steps)
        result.update(
            {
                "rear120_activation_steps": self.steps,
                "rear120_activation_active_steps": self.active_steps,
                "rear120_activation_active_ratio": self.active_steps / max(1, self.steps),
                "rear120_activation_entries": self.entries,
                "rear120_activation_exits": self.exits,
                "rear120_activation_mean_active_steps": (
                    sum(durations) / len(durations) if durations else 0.0
                ),
                "rear120_activation_min_active_steps": min(durations) if durations else 0,
                "rear120_activation_max_active_steps": max(durations) if durations else 0,
                "rear120_activation_safety_veto_steps": self.safety_veto_steps,
                "rear120_activation_safety_veto_ratio": self.safety_veto_steps / max(1, self.steps),
                "rear120_activation_active_final": self.active,
                "rear120_activation_aim_gate": self.aim.telemetry(),
                "rear120_activation_offensive_gate": self.offensive.telemetry(),
            }
        )
        return result


def _body_velocity_to_ned(state: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.radians(state[3:6])
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
    return rotation @ np.asarray(state[6:9], dtype=np.float64)


@dataclass(frozen=True)
class OffensiveGateConfig:
    min_range_m: float = 152.4
    enter_max_range_m: float = 1500.0
    exit_max_range_m: float = 2000.0
    enter_ata_deg: float = 15.0
    exit_ata_deg: float = 25.0
    enter_min_target_ata_deg: float = 135.0
    exit_min_target_ata_deg: float = 110.0

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
        self._current_active_steps = 0
        self._completed_active_steps: list[int] = []
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
        if self.active:
            self._current_active_steps += 1
        elif exit_event:
            self._completed_active_steps.append(self._current_active_steps)
            self._current_active_steps = 0
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
        durations = list(self._completed_active_steps)
        if self.active and self._current_active_steps:
            durations.append(self._current_active_steps)
        return {
            "offensive_gate_config": asdict(self.config),
            "offensive_gate_steps": self.steps,
            "offensive_gate_active_steps": self.active_steps,
            "offensive_gate_active_ratio": self.active_steps / max(1, self.steps),
            "offensive_gate_entries": self.entries,
            "offensive_gate_exits": self.exits,
            "offensive_gate_active_final": self.active,
            "offensive_gate_mean_active_steps": (
                float(np.mean(durations)) if durations else 0.0
            ),
            "offensive_gate_min_active_steps": min(durations) if durations else 0,
        }


@dataclass(frozen=True)
class AimGateConfig:
    """Phase-aware pre-aim gate expressed in official half-angle semantics."""

    min_range_m: float = 152.4
    enter_angle_margin_deg: float = 7.0
    exit_angle_margin_deg: float = 10.0
    enter_range_margin_m: float = 300.0
    exit_range_margin_m: float = 550.0
    min_hold_steps: int = 12

    def validate(self) -> None:
        if self.min_range_m < 0.0:
            raise ValueError("aim min range must be non-negative")
        if not 0.0 <= self.enter_angle_margin_deg <= self.exit_angle_margin_deg:
            raise ValueError("aim angle margins must satisfy 0 <= enter <= exit")
        if not 0.0 <= self.enter_range_margin_m <= self.exit_range_margin_m:
            raise ValueError("aim range margins must satisfy 0 <= enter <= exit")
        if self.min_hold_steps < 0:
            raise ValueError("aim minimum hold steps must be non-negative")


class AimResidualGate:
    """Hysteretic pre-aim gate aligned with the three official damage phases."""

    _PHASES = (
        (1, 100.0, 1.0, 3000.0 * 0.3048),
        (2, 150.0, 2.0, 3500.0 * 0.3048),
        (3, float("inf"), 3.0, 4000.0 * 0.3048),
    )

    def __init__(self, config: AimGateConfig | dict | None = None):
        if config is None:
            config = AimGateConfig()
        elif isinstance(config, dict):
            config = AimGateConfig(**config)
        config.validate()
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.steps = 0
        self.active_steps = 0
        self.entries = 0
        self.exits = 0
        self._current_active_steps = 0
        self._completed_active_steps: list[int] = []
        self.last_geometry: dict[str, float | int | bool] = {
            "distance_m": float("nan"),
            "aim_error_deg": float("nan"),
            "target_ata_deg": float("nan"),
            "phase": 1,
            "active": False,
            "entry": False,
            "exit": False,
        }

    @classmethod
    def phase_limits(cls, sim_time_s: float) -> tuple[int, float, float]:
        for phase, end_s, half_angle_deg, max_range_m in cls._PHASES:
            if sim_time_s <= end_s:
                return phase, half_angle_deg, max_range_m
        raise AssertionError("phase table must have an infinite final interval")

    def update(
        self,
        ownship_state,
        target_state,
        *,
        sim_time_s: float | None = None,
    ) -> dict[str, float | int | bool]:
        previous = self.active
        if ownship_state is None or target_state is None:
            next_active = False
            distance = aim_error = target_ata = float("nan")
            phase, half_angle, phase_range = self.phase_limits(sim_time_s or 0.0)
        else:
            own = np.asarray(ownship_state, dtype=np.float64)
            target = np.asarray(target_state, dtype=np.float64)
            distance = float(np.linalg.norm(target[:3] - own[:3]))
            aim_error = _unsigned_ata_deg(own, target)
            target_ata = _unsigned_ata_deg(target, own)
            if sim_time_s is None:
                sim_time_s = (
                    float(own[StateIndex.SIM_TIME])
                    if len(own) > StateIndex.SIM_TIME
                    else 0.0
                )
            phase, half_angle, phase_range = self.phase_limits(float(sim_time_s))
            cfg = self.config
            enter = (
                cfg.min_range_m <= distance <= phase_range + cfg.enter_range_margin_m
                and aim_error <= half_angle + cfg.enter_angle_margin_deg
            )
            remain = (
                cfg.min_range_m <= distance <= phase_range + cfg.exit_range_margin_m
                and aim_error <= half_angle + cfg.exit_angle_margin_deg
            )
            minimum_hold = (
                previous
                and self._current_active_steps < cfg.min_hold_steps
                and np.isfinite(distance)
                and distance >= cfg.min_range_m
            )
            next_active = remain or minimum_hold if previous else enter

        entry = bool(next_active and not previous)
        exit_event = bool(previous and not next_active)
        self.entries += int(entry)
        self.exits += int(exit_event)
        self.active = bool(next_active)
        self.steps += 1
        self.active_steps += int(self.active)
        if self.active:
            self._current_active_steps += 1
        elif exit_event:
            self._completed_active_steps.append(self._current_active_steps)
            self._current_active_steps = 0
        self.last_geometry = {
            "distance_m": distance,
            "aim_error_deg": aim_error,
            "target_ata_deg": target_ata,
            "phase": phase,
            "phase_half_angle_deg": half_angle,
            "phase_max_range_m": phase_range,
            "active": self.active,
            "entry": entry,
            "exit": exit_event,
        }
        return dict(self.last_geometry)

    def telemetry(self) -> dict[str, int | float | bool | dict]:
        durations = list(self._completed_active_steps)
        if self.active and self._current_active_steps:
            durations.append(self._current_active_steps)
        return {
            "aim_gate_config": asdict(self.config),
            "aim_gate_steps": self.steps,
            "aim_gate_active_steps": self.active_steps,
            "aim_gate_active_ratio": self.active_steps / max(1, self.steps),
            "aim_gate_entries": self.entries,
            "aim_gate_exits": self.exits,
            "aim_gate_active_final": self.active,
            "aim_gate_mean_active_steps": (
                float(np.mean(durations)) if durations else 0.0
            ),
            "aim_gate_min_active_steps": min(durations) if durations else 0,
        }


class CombinedResidualGate:
    """Activate only when phase-aware aim and offensive gates overlap."""

    def __init__(
        self,
        aim_config: AimGateConfig | dict | None = None,
        offensive_config: OffensiveGateConfig | dict | None = None,
    ):
        self.aim_gate = AimResidualGate(aim_config)
        self.offensive_gate = OffensiveResidualGate(offensive_config)
        self.reset()

    def reset(self) -> None:
        self.aim_gate.reset()
        self.offensive_gate.reset()
        self.active = False
        self.steps = 0
        self.active_steps = 0
        self.entries = 0
        self.exits = 0
        self._current_active_steps = 0
        self._completed_active_steps: list[int] = []
        self.last_geometry: dict = {
            "active": False,
            "entry": False,
            "exit": False,
            "aim_gate": dict(self.aim_gate.last_geometry),
            "offensive_gate": dict(self.offensive_gate.last_geometry),
        }

    def update(
        self,
        ownship_state,
        target_state,
        *,
        sim_time_s: float | None = None,
    ) -> dict:
        previous = self.active
        aim = self.aim_gate.update(
            ownship_state,
            target_state,
            sim_time_s=sim_time_s,
        )
        offensive = self.offensive_gate.update(ownship_state, target_state)
        self.active = bool(aim["active"] and offensive["active"])
        entry = bool(self.active and not previous)
        exit_event = bool(previous and not self.active)
        self.entries += int(entry)
        self.exits += int(exit_event)
        self.steps += 1
        self.active_steps += int(self.active)
        if self.active:
            self._current_active_steps += 1
        elif exit_event:
            self._completed_active_steps.append(self._current_active_steps)
            self._current_active_steps = 0
        self.last_geometry = {
            "distance_m": aim["distance_m"],
            "aim_error_deg": aim["aim_error_deg"],
            "ata_deg": offensive["ata_deg"],
            "target_ata_deg": offensive["target_ata_deg"],
            "phase": aim["phase"],
            "active": self.active,
            "entry": entry,
            "exit": exit_event,
            "aim_gate": aim,
            "offensive_gate": offensive,
        }
        return dict(self.last_geometry)

    def telemetry(self) -> dict:
        durations = list(self._completed_active_steps)
        if self.active and self._current_active_steps:
            durations.append(self._current_active_steps)
        return {
            "combined_gate_steps": self.steps,
            "combined_gate_active_steps": self.active_steps,
            "combined_gate_active_ratio": self.active_steps / max(1, self.steps),
            "combined_gate_entries": self.entries,
            "combined_gate_exits": self.exits,
            "combined_gate_active_final": self.active,
            "combined_gate_mean_active_steps": (
                float(np.mean(durations)) if durations else 0.0
            ),
            "combined_gate_min_active_steps": min(durations) if durations else 0,
            "combined_aim_gate": self.aim_gate.telemetry(),
            "combined_offensive_gate": self.offensive_gate.telemetry(),
        }


def _compose_residual(
    bt_action, rl_action, scale: float, *, throttle_scale: float | None = None
) -> tuple[np.ndarray, dict]:
    """Add signed surface corrections and convex-blend simulator throttle."""
    bt = clip_action(bt_action)
    rl = clip_action(rl_action)
    unclipped = bt.copy()
    unclipped[:3] = bt[:3] + scale * rl[:3]
    effective_throttle_scale = scale if throttle_scale is None else float(throttle_scale)
    unclipped[3] = (
        (1.0 - effective_throttle_scale) * bt[3]
        + effective_throttle_scale * rl[3]
    )
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
        "effective_throttle_scale": effective_throttle_scale,
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
        min_throttle_blend_speed: float = 210.0,
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
        self.min_throttle_blend_speed = float(min_throttle_blend_speed)
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
        self._throttle_guard_steps = 0
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
        bt_throttle = float(clip_action(secondary.action)[3])
        rl_throttle = float(self._cached_primary_action[3])
        speed = (
            float(context.ownship_state[StateIndex.KCAS])
            if context.ownship_state is not None
            and len(context.ownship_state) > StateIndex.KCAS
            else float("inf")
        )
        throttle_guard_active = (
            speed < self.min_throttle_blend_speed and rl_throttle < bt_throttle
        )
        throttle_scale = 0.0 if throttle_guard_active else self.residual_scale
        self._throttle_guard_steps += int(throttle_guard_active)
        final, composition = _compose_residual(
            secondary.action,
            self._cached_primary_action,
            self.residual_scale,
            throttle_scale=throttle_scale,
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
            "speed_for_throttle_guard": speed,
            "min_throttle_blend_speed": self.min_throttle_blend_speed,
            "throttle_guard_active": throttle_guard_active,
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
                "throttle_guard_steps": self._throttle_guard_steps,
                "last_frame": dict(self._last_frame_info),
            }
        )
        return result

    def close(self) -> None:
        self.primary_provider.close()
        self.secondary_provider.close()


class ResidualInferenceActionProvider(ActionProvider):
    """Run raw BT every frame and apply a held policy residual only inside a gate."""

    def __init__(
        self,
        bt_provider: ActionProvider,
        residual_provider: ActionProvider,
        *,
        residual_scale: float,
        gate_kind: str = "aim",
        aim_gate: AimGateConfig | dict | None = None,
        offensive_gate: OffensiveGateConfig | dict | None = None,
        rear120_gate: Rear120GateConfig | dict | None = None,
        safety_veto: SafetyVetoConfig | dict | None = None,
        rl_action_repeat: int = 6,
        composition_mode: str = "additive",
        inference_timeout_s: float = 0.1667,
        confidence: float = 0.95,
        residual_axis_mask: str = "roll_pitch_yaw",
    ):
        if residual_scale not in ALLOWED_AIM_RESIDUAL_SCALES:
            raise ValueError(
                "aim residual scale must be one of "
                f"{ALLOWED_AIM_RESIDUAL_SCALES}, got {residual_scale}"
            )
        if gate_kind == "aim":
            gate = AimResidualGate(aim_gate)
        elif gate_kind == "offensive":
            gate = OffensiveResidualGate(offensive_gate)
        elif gate_kind == "combined":
            gate = CombinedResidualGate(aim_gate, offensive_gate)
        elif gate_kind == "rear120":
            gate = Rear120ActivationGate(
                rear120_gate,
                aim_gate,
                offensive_gate,
                safety_veto,
            )
        else:
            raise ValueError(f"unsupported residual inference gate: {gate_kind!r}")
        self.bt_provider = bt_provider
        self.residual_provider = residual_provider
        self.residual_scale = float(residual_scale)
        if composition_mode not in RESIDUAL_COMPOSITION_MODES:
            raise ValueError(f"unsupported residual composition: {composition_mode!r}")
        self.composition_mode = composition_mode
        if residual_axis_mask not in RESIDUAL_AXIS_MASKS:
            raise ValueError(
                f"unsupported residual axis mask: {residual_axis_mask!r}"
            )
        self.residual_axis_mask = residual_axis_mask
        self._residual_axis_vector = np.asarray(
            RESIDUAL_AXIS_MASKS[residual_axis_mask], dtype=np.float32
        )
        self.gate_kind = gate_kind
        self.gate = gate
        self.rl_action_repeat = max(1, int(rl_action_repeat))
        self.inference_timeout_s = float(inference_timeout_s)
        if self.inference_timeout_s <= 0.0:
            raise ValueError("inference_timeout_s must be positive")
        self.confidence = float(confidence)
        self.reset(None)

    def reset(self, context: ActionContext | None = None) -> None:
        self.bt_provider.reset(context)
        self.residual_provider.reset(context)
        self.gate.reset()
        self._prepared_bt_result: ActionResult | None = None
        self._active_frames = 0
        self._cached_residual: np.ndarray | None = None
        self._rl_inference_calls = 0
        self._rl_inference_latency_ms: list[float] = []
        self._rl_fallback_steps = 0
        self._rl_exception_fallback_steps = 0
        self._rl_nonfinite_fallback_steps = 0
        self._rl_timeout_fallback_steps = 0
        self._correction_steps = 0
        self._correction_abs_sum = np.zeros(4, dtype=np.float64)
        self._correction_abs_max = np.zeros(4, dtype=np.float64)
        self._clipped_steps = 0
        self._saturated_steps = 0
        _reset_authority_counters(self)
        self._last_frame: dict = {}

    @property
    def prepared_bt_action(self) -> np.ndarray | None:
        if self._prepared_bt_result is None:
            return None
        return clip_action(self._prepared_bt_result.action).copy()

    def prepare_bt_action(self, context: ActionContext) -> np.ndarray:
        if self._prepared_bt_result is None:
            result = self.bt_provider.compute_action(context)
            self._prepared_bt_result = ActionResult(
                clip_action(result.action),
                result.source,
                result.confidence,
                dict(result.info),
            )
        return clip_action(self._prepared_bt_result.action).copy()

    def _consume_bt_result(self, context: ActionContext) -> ActionResult:
        if self._prepared_bt_result is None:
            return self.bt_provider.compute_action(context)
        result = self._prepared_bt_result
        self._prepared_bt_result = None
        return result

    def compute_action(self, context: ActionContext) -> ActionResult:
        bt_result = self._consume_bt_result(context)
        bt_action = clip_action(bt_result.action)
        if self.gate_kind == "rear120":
            gate_info = self.gate.update(
                context.ownship_state,
                context.target_state,
                sim_time_s=context.info.get("sim_time_s"),
                bt_action=bt_action,
            )
        elif self.gate_kind in ("aim", "combined"):
            gate_info = self.gate.update(
                context.ownship_state,
                context.target_state,
                sim_time_s=context.info.get("sim_time_s"),
            )
        else:
            gate_info = self.gate.update(
                context.ownship_state,
                context.target_state,
            )

        if not gate_info["active"]:
            self._cached_residual = None
            self._active_frames = 0
            frame = self._frame_info(
                gate_info=gate_info,
                bt_action=bt_action,
                residual=None,
                final=bt_action.copy(),
                refreshed=False,
                clipped=False,
                saturated=False,
            )
            self._last_frame = frame
            return ActionResult(bt_action.copy(), "bt_residual_inference", self.confidence, frame)

        if gate_info["entry"]:
            self.residual_provider.reset(context)
        refreshed = (
            self._cached_residual is None
            or self._active_frames % self.rl_action_repeat == 0
        )
        if refreshed:
            start = perf_counter()
            try:
                residual_result = self.residual_provider.compute_action(context)
            except Exception as exc:
                latency_ms = (perf_counter() - start) * 1000.0
                self._rl_inference_latency_ms.append(latency_ms)
                return self._fallback_to_bt(
                    gate_info,
                    bt_action,
                    reason="inference_exception",
                    detail=type(exc).__name__,
                )
            latency_ms = (perf_counter() - start) * 1000.0
            self._rl_inference_latency_ms.append(latency_ms)
            residual = np.asarray(residual_result.action, dtype=np.float32)
            if residual.shape != (4,):
                return self._fallback_to_bt(
                    gate_info,
                    bt_action,
                    reason="invalid_shape",
                    detail=str(residual.shape),
                )
            raw_policy = residual_result.info.get("raw_policy_action")
            raw_policy = residual if raw_policy is None else np.asarray(raw_policy)
            if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(raw_policy)):
                return self._fallback_to_bt(
                    gate_info,
                    bt_action,
                    reason="nonfinite_output",
                )
            if latency_ms > self.inference_timeout_s * 1000.0:
                return self._fallback_to_bt(
                    gate_info,
                    bt_action,
                    reason="inference_timeout",
                    detail=f"{latency_ms:.6f}ms",
                )
            self._cached_residual = np.clip(
                residual,
                -1.0,
                1.0,
            )
            self._rl_inference_calls += 1

        residual = np.asarray(self._cached_residual, dtype=np.float32)
        masked_residual = residual * self._residual_axis_vector
        unclipped = _compose_aim_surface_residual(
            bt_action,
            masked_residual,
            self.residual_scale,
            self.composition_mode,
        )
        final = clip_action(unclipped)
        final[3] = bt_action[3]
        correction = final - bt_action
        clipped = bool(np.any(np.abs(final[:3] - unclipped[:3]) > 1e-7))
        saturated = bool(np.any(np.isclose(np.abs(final[:3]), 1.0)))
        self._correction_steps += 1
        self._correction_abs_sum += np.abs(correction)
        self._correction_abs_max = np.maximum(
            self._correction_abs_max,
            np.abs(correction),
        )
        self._clipped_steps += int(clipped)
        self._saturated_steps += int(saturated)
        diagnostics = _surface_authority_diagnostics(
            bt_action,
            masked_residual,
            final,
            self.residual_scale,
            active=True,
        )
        _update_authority_counters(self, diagnostics)
        self._active_frames += 1
        frame = self._frame_info(
            gate_info=gate_info,
            bt_action=bt_action,
            residual=residual,
            masked_residual=masked_residual,
            final=final,
            refreshed=refreshed,
            clipped=clipped,
            saturated=saturated,
        )
        self._last_frame = frame
        return ActionResult(final, "bt_residual_inference", self.confidence, frame)

    def _fallback_to_bt(
        self,
        gate_info: dict,
        bt_action: np.ndarray,
        *,
        reason: str,
        detail: str = "",
    ) -> ActionResult:
        self._cached_residual = None
        self._rl_fallback_steps += 1
        if reason in {"inference_exception", "invalid_shape"}:
            self._rl_exception_fallback_steps += 1
        elif reason == "nonfinite_output":
            self._rl_nonfinite_fallback_steps += 1
        elif reason == "inference_timeout":
            self._rl_timeout_fallback_steps += 1
        frame = self._frame_info(
            gate_info=gate_info,
            bt_action=bt_action,
            residual=None,
            final=bt_action.copy(),
            refreshed=True,
            clipped=False,
            saturated=False,
        )
        frame.update(
            {
                "rl_fallback": True,
                "rl_fallback_reason": reason,
                "rl_fallback_detail": detail,
            }
        )
        self._last_frame = frame
        return ActionResult(
            bt_action.copy(),
            "bt_residual_inference_fallback",
            self.confidence,
            frame,
        )

    def _frame_info(
        self,
        *,
        gate_info: dict,
        bt_action: np.ndarray,
        residual: np.ndarray | None,
        final: np.ndarray,
        refreshed: bool,
        clipped: bool,
        saturated: bool,
        masked_residual: np.ndarray | None = None,
    ) -> dict:
        correction = final - bt_action
        effective_residual = residual if masked_residual is None else masked_residual
        authority = _surface_authority_diagnostics(
            bt_action,
            effective_residual,
            final,
            self.residual_scale,
            active=bool(gate_info["active"]),
        )
        return {
            "mode": "bt_residual_inference",
            "gate_kind": self.gate_kind,
            "gate": gate_info,
            f"{self.gate_kind}_gate": gate_info,
            "effective_residual_scale": (
                self.residual_scale if gate_info["active"] else 0.0
            ),
            "residual_composition_mode": self.composition_mode,
            "rl_action_repeat": self.rl_action_repeat,
            "rl_action_refreshed": refreshed,
            "bt_action": bt_action.tolist(),
            "raw_residual_action": residual.tolist() if residual is not None else None,
            "residual_axis_mask": self.residual_axis_mask,
            "masked_residual_action": (
                effective_residual.tolist() if effective_residual is not None else None
            ),
            "applied_rl_correction": correction.tolist(),
            "surface_authority": authority,
            "final_action": final.tolist(),
            "throttle_residual_forced_zero": True,
            "action_clipped": clipped,
            "action_saturation": saturated,
        }

    def telemetry(self) -> dict:
        result = self.gate.telemetry()
        latency = np.asarray(self._rl_inference_latency_ms, dtype=np.float64)
        result.update(
            {
                "residual_inference_gate_kind": self.gate_kind,
                "residual_scale": self.residual_scale,
                "residual_composition_mode": self.composition_mode,
                "residual_axis_mask": self.residual_axis_mask,
                "rl_inference_calls": self._rl_inference_calls,
                "rl_action_repeat": self.rl_action_repeat,
                "rl_inference_timeout_s": self.inference_timeout_s,
                "rl_fallback_steps": self._rl_fallback_steps,
                "rl_exception_fallback_steps": self._rl_exception_fallback_steps,
                "rl_nonfinite_fallback_steps": self._rl_nonfinite_fallback_steps,
                "rl_timeout_fallback_steps": self._rl_timeout_fallback_steps,
                "rl_correction_steps": self._correction_steps,
                "rl_correction_abs_mean": (
                    self._correction_abs_sum / max(1, self._correction_steps)
                ).tolist(),
                "rl_correction_abs_max": self._correction_abs_max.tolist(),
                "action_clipped_steps": self._clipped_steps,
                "action_saturated_steps": self._saturated_steps,
                **_authority_telemetry(self, self._correction_steps),
                "rl_inference_latency_ms_p50": (
                    float(np.percentile(latency, 50)) if latency.size else 0.0
                ),
                "rl_inference_latency_ms_p95": (
                    float(np.percentile(latency, 95)) if latency.size else 0.0
                ),
                "rl_inference_latency_ms_p99": (
                    float(np.percentile(latency, 99)) if latency.size else 0.0
                ),
                "rl_inference_latency_ms_max": (
                    float(np.max(latency)) if latency.size else 0.0
                ),
                "rl_inference_over_166_7ms_ratio": (
                    float(np.mean(latency > 166.7)) if latency.size else 0.0
                ),
                "last_frame": dict(self._last_frame),
            }
        )
        return result

    def close(self) -> None:
        self.bt_provider.close()
        self.residual_provider.close()


class ResidualTrainingActionProvider(ActionProvider):
    """Compose policy residuals with raw BT actions inside the training loop."""

    def __init__(
        self,
        bt_provider: ActionProvider,
        *,
        residual_scale: float,
        gate_kind: str = "aim",
        aim_gate: AimGateConfig | dict | None = None,
        offensive_gate: OffensiveGateConfig | dict | None = None,
        rear120_gate: Rear120GateConfig | dict | None = None,
        safety_veto: SafetyVetoConfig | dict | None = None,
        composition_mode: str = "additive",
        confidence: float = 0.95,
        residual_axis_mask: str = "roll_pitch_yaw",
    ):
        if residual_scale not in ALLOWED_AIM_RESIDUAL_SCALES:
            raise ValueError(
                "aim residual scale must be one of "
                f"{ALLOWED_AIM_RESIDUAL_SCALES}, got {residual_scale}"
            )
        if gate_kind == "aim":
            gate = AimResidualGate(aim_gate)
        elif gate_kind == "offensive":
            gate = OffensiveResidualGate(offensive_gate)
        elif gate_kind == "combined":
            gate = CombinedResidualGate(aim_gate, offensive_gate)
        elif gate_kind == "rear120":
            gate = Rear120ActivationGate(
                rear120_gate,
                aim_gate,
                offensive_gate,
                safety_veto,
            )
        else:
            raise ValueError(f"unsupported residual training gate: {gate_kind!r}")
        self.bt_provider = bt_provider
        self.residual_scale = float(residual_scale)
        if composition_mode not in RESIDUAL_COMPOSITION_MODES:
            raise ValueError(f"unsupported residual composition: {composition_mode!r}")
        self.composition_mode = composition_mode
        if residual_axis_mask not in RESIDUAL_AXIS_MASKS:
            raise ValueError(
                f"unsupported residual axis mask: {residual_axis_mask!r}"
            )
        self.residual_axis_mask = residual_axis_mask
        self._residual_axis_vector = np.asarray(
            RESIDUAL_AXIS_MASKS[residual_axis_mask], dtype=np.float32
        )
        self.gate_kind = gate_kind
        self.gate = gate
        self.confidence = float(confidence)
        self.reset(None)

    def reset(self, context: ActionContext | None = None) -> None:
        self.bt_provider.reset(context)
        self.gate.reset()
        self._prepared_bt_result: ActionResult | None = None
        self._correction_steps = 0
        self._correction_abs_sum = np.zeros(4, dtype=np.float64)
        self._correction_abs_max = np.zeros(4, dtype=np.float64)
        self._clipped_steps = 0
        self._saturated_steps = 0
        self._requested_throttle_abs_sum = 0.0
        _reset_authority_counters(self)
        self._last_frame: dict = {}

    @property
    def prepared_bt_action(self) -> np.ndarray | None:
        if self._prepared_bt_result is None:
            return None
        return clip_action(self._prepared_bt_result.action).copy()

    def prepare_bt_action(self, context: ActionContext) -> np.ndarray:
        """Tick BT once and cache the command for observation and composition."""
        if self._prepared_bt_result is None:
            result = self.bt_provider.compute_action(context)
            self._prepared_bt_result = ActionResult(
                clip_action(result.action),
                result.source,
                result.confidence,
                dict(result.info),
            )
        return clip_action(self._prepared_bt_result.action).copy()

    def _consume_bt_result(self, context: ActionContext) -> ActionResult:
        if self._prepared_bt_result is None:
            return self.bt_provider.compute_action(context)
        result = self._prepared_bt_result
        self._prepared_bt_result = None
        return result

    def compute_action(self, context: ActionContext) -> ActionResult:
        bt_result = self._consume_bt_result(context)
        bt_action = clip_action(bt_result.action)
        residual = np.asarray(
            context.info.get("residual_action", np.zeros(4)),
            dtype=np.float32,
        )
        if residual.shape != (4,):
            raise ValueError(f"expected four residual axes, got shape {residual.shape}")
        residual = np.clip(
            np.nan_to_num(residual, nan=0.0, posinf=1.0, neginf=-1.0),
            -1.0,
            1.0,
        )
        self._requested_throttle_abs_sum += abs(float(residual[3]))
        masked_residual = residual * self._residual_axis_vector

        if self.gate_kind == "rear120":
            gate_info = self.gate.update(
                context.ownship_state,
                context.target_state,
                sim_time_s=context.info.get("sim_time_s"),
                bt_action=bt_action,
            )
        elif self.gate_kind in ("aim", "combined"):
            gate_info = self.gate.update(
                context.ownship_state,
                context.target_state,
                sim_time_s=context.info.get("sim_time_s"),
            )
        else:
            gate_info = self.gate.update(
                context.ownship_state,
                context.target_state,
            )

        unclipped = bt_action.copy()
        if gate_info["active"]:
            unclipped = _compose_aim_surface_residual(
                bt_action,
                masked_residual,
                self.residual_scale,
                self.composition_mode,
            )
        final = clip_action(unclipped)
        final[3] = bt_action[3]
        correction = final - bt_action
        clipped = bool(np.any(np.abs(final[:3] - unclipped[:3]) > 1e-7))
        saturated = bool(np.any(np.isclose(np.abs(final[:3]), 1.0)))
        if gate_info["active"]:
            self._correction_steps += 1
            self._correction_abs_sum += np.abs(correction)
            self._correction_abs_max = np.maximum(
                self._correction_abs_max,
                np.abs(correction),
            )
            self._clipped_steps += int(clipped)
            self._saturated_steps += int(saturated)

        authority = _surface_authority_diagnostics(
            bt_action,
            masked_residual,
            final,
            self.residual_scale,
            active=bool(gate_info["active"]),
        )
        if gate_info["active"]:
            _update_authority_counters(self, authority)

        frame = {
            "mode": "bt_residual_training",
            "gate_kind": self.gate_kind,
            "gate": gate_info,
            f"{self.gate_kind}_gate": gate_info,
            "effective_residual_scale": (
                self.residual_scale if gate_info["active"] else 0.0
            ),
            "residual_composition_mode": self.composition_mode,
            "residual_axis_mask": self.residual_axis_mask,
            "bt_action": bt_action.tolist(),
            "raw_residual_action": residual.tolist(),
            "masked_residual_action": masked_residual.tolist(),
            "applied_rl_correction": correction.tolist(),
            "surface_authority": authority,
            "final_action": final.tolist(),
            "throttle_residual_forced_zero": True,
            "action_clipped": clipped,
            "action_saturation": saturated,
        }
        self._last_frame = frame
        return ActionResult(final, "bt_residual_training", self.confidence, frame)

    def telemetry(self) -> dict:
        result = self.gate.telemetry()
        gate_steps = int(result.get(f"{self.gate_kind}_gate_steps", 0))
        result.update(
            {
                "residual_training_gate_kind": self.gate_kind,
                "residual_scale": self.residual_scale,
                "residual_composition_mode": self.composition_mode,
                "residual_axis_mask": self.residual_axis_mask,
                "rl_correction_steps": self._correction_steps,
                "rl_correction_abs_mean": (
                    self._correction_abs_sum / max(1, self._correction_steps)
                ).tolist(),
                "rl_correction_abs_max": self._correction_abs_max.tolist(),
                "requested_throttle_residual_abs_mean": (
                    self._requested_throttle_abs_sum / max(1, gate_steps)
                ),
                "action_clipped_steps": self._clipped_steps,
                "action_saturated_steps": self._saturated_steps,
                **_authority_telemetry(self, self._correction_steps),
                "last_frame": dict(self._last_frame),
            }
        )
        return result

    def close(self) -> None:
        self.bt_provider.close()
