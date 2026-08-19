from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np

from dogfight.ai.action_provider import clip_action
from dogfight.ai.guidance_selector import (
    GuidanceActionConfig,
    GuidanceControllerConfig,
    GuidanceSetpoint,
    guidance_to_surface_action,
)
from dogfight.envs.observation import body_to_ned_rotation
from dogfight.sim.state_schema import StateIndex


TACTICAL_MODES_T1 = (
    "BT_DEFAULT",
    "PURE_PURSUIT",
    "LEAD_PURSUIT_T060",
    "LAG_PURSUIT_D250",
)
TACTICAL_MODES_T2 = (
    "LOS_RATE_DAMPED_PURSUIT",
    "CROSSING_LEAD_T100",
    "CONE_CAPTURE",
)
TACTICAL_MODES = (*TACTICAL_MODES_T1, *TACTICAL_MODES_T2)
TACTICAL_MODE_TO_ID = {name: index for index, name in enumerate(TACTICAL_MODES)}
TACTICAL_HOLD_FRAMES = (30, 60, 120)


@dataclass(frozen=True)
class TacticalModeConfig:
    lead_time_s: float = 0.60
    lag_distance_m: float = 250.0
    los_rate_damping_time_s: float = 0.50
    crossing_lead_time_s: float = 1.00
    cone_capture_min_time_s: float = 0.25
    cone_capture_max_time_s: float = 1.50
    minimum_vp_altitude_m: float = 1000.0
    maximum_vp_distance_m: float = 5000.0

    def validate(self) -> None:
        finite_positive = (
            self.lead_time_s,
            self.lag_distance_m,
            self.los_rate_damping_time_s,
            self.crossing_lead_time_s,
            self.cone_capture_min_time_s,
            self.cone_capture_max_time_s,
            self.minimum_vp_altitude_m,
            self.maximum_vp_distance_m,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in finite_positive):
            raise ValueError("Tactical mode parameters must be finite and positive")
        if self.cone_capture_max_time_s < self.cone_capture_min_time_s:
            raise ValueError("cone capture maximum time must cover minimum time")


@dataclass(frozen=True)
class TacticalControllerConfig:
    guidance_action: GuidanceActionConfig = GuidanceActionConfig()
    guidance_controller: GuidanceControllerConfig = GuidanceControllerConfig(
        kind="vp_error_pd_v2",
        maximum_surface_correction=0.15,
    )

    def validate(self) -> None:
        self.guidance_action.validate()
        self.guidance_controller.validate()


def tactical_mode_id(mode: str | int) -> int:
    if isinstance(mode, str):
        if mode not in TACTICAL_MODE_TO_ID:
            raise ValueError(f"unsupported Tactical mode: {mode!r}")
        return TACTICAL_MODE_TO_ID[mode]
    mode_id = int(mode)
    if not 0 <= mode_id < len(TACTICAL_MODES):
        raise ValueError(f"Tactical mode id out of range: {mode_id}")
    return mode_id


def tactical_mode_name(mode: str | int) -> str:
    return TACTICAL_MODES[tactical_mode_id(mode)]


def _server_visible_state(state: Any) -> np.ndarray:
    vector = np.asarray(state, dtype=np.float64)
    if vector.ndim != 1 or vector.size <= StateIndex.ALT:
        raise ValueError("Tactical mode requires the server state schema through altitude")
    required = np.concatenate((vector[:9], vector[[StateIndex.KCAS, StateIndex.ALT]]))
    if not np.all(np.isfinite(required)):
        raise ValueError("Tactical mode state contains nonfinite server-visible values")
    return vector


def velocity_ned_from_server_state(state: Any) -> np.ndarray:
    """Return NED velocity using only packet-derived attitude/body velocity."""
    vector = _server_visible_state(state)
    return body_to_ned_rotation(vector[3:6]) @ vector[6:9]


def _safe_vp(vp: np.ndarray, own: np.ndarray, config: TacticalModeConfig) -> np.ndarray:
    desired = np.asarray(vp, dtype=np.float64).copy()
    if desired.shape != (3,) or not np.all(np.isfinite(desired)):
        raise ValueError("Tactical VP must be a finite NED vector")
    relative = desired - own[:3]
    distance = float(np.linalg.norm(relative))
    if distance > config.maximum_vp_distance_m:
        relative *= config.maximum_vp_distance_m / distance
        desired = own[:3] + relative
    # NED down is positive. Altitude must remain at or above the safety floor.
    desired[2] = min(desired[2], -config.minimum_vp_altitude_m)
    return desired


def generate_tactical_vp(
    mode: str | int,
    ownship_state: Any,
    target_state: Any,
    *,
    config: TacticalModeConfig | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Generate an auditable VP from server-visible state only.

    BT_DEFAULT deliberately returns no VP so callers can return the exact BT
    command without running a controller.
    """
    name = tactical_mode_name(mode)
    cfg = config or TacticalModeConfig()
    cfg.validate()
    if name == "BT_DEFAULT":
        return None, {"mode": name, "meaning": "exact Pure BT command and VP"}

    own = _server_visible_state(ownship_state)
    target = _server_visible_state(target_state)
    target_position = target[:3]
    own_velocity = velocity_ned_from_server_state(own)
    target_velocity = velocity_ned_from_server_state(target)
    relative = target_position - own[:3]
    distance = float(np.linalg.norm(relative))
    los = relative / max(distance, 1e-9)
    relative_velocity = target_velocity - own_velocity
    radial_velocity = float(np.dot(relative_velocity, los))
    transverse_velocity = relative_velocity - radial_velocity * los

    if name == "PURE_PURSUIT":
        vp = target_position
        parameters = {}
        meaning = "current target position"
    elif name == "LEAD_PURSUIT_T060":
        vp = target_position + cfg.lead_time_s * target_velocity
        parameters = {"lead_time_s": cfg.lead_time_s}
        meaning = "target velocity extrapolation"
    elif name == "LAG_PURSUIT_D250":
        target_speed = float(np.linalg.norm(target_velocity))
        flight_path = target_velocity / max(target_speed, 1e-9)
        vp = target_position - cfg.lag_distance_m * flight_path
        parameters = {"lag_distance_m": cfg.lag_distance_m}
        meaning = "point behind target along target flight path"
    elif name == "LOS_RATE_DAMPED_PURSUIT":
        vp = target_position + cfg.los_rate_damping_time_s * transverse_velocity
        parameters = {"damping_time_s": cfg.los_rate_damping_time_s}
        meaning = "advance along transverse relative motion to reduce LOS-rate lag"
    elif name == "CROSSING_LEAD_T100":
        vp = target_position + cfg.crossing_lead_time_s * target_velocity
        parameters = {"lead_time_s": cfg.crossing_lead_time_s}
        meaning = "longer target flight-path lead for crossing geometry"
    else:
        closing_speed = max(1.0, -radial_velocity)
        capture_time = float(
            np.clip(
                distance / closing_speed,
                cfg.cone_capture_min_time_s,
                cfg.cone_capture_max_time_s,
            )
        )
        vp = target_position + capture_time * transverse_velocity
        parameters = {"capture_time_s": capture_time}
        meaning = "bounded transverse intercept for cone capture"

    safe_vp = _safe_vp(vp, own, cfg)
    return safe_vp, {
        "mode": name,
        "meaning": meaning,
        "parameters": parameters,
        "distance_m": distance,
        "radial_relative_velocity_m_s": radial_velocity,
        "transverse_relative_speed_m_s": float(np.linalg.norm(transverse_velocity)),
        "throttle_effect": "none",
        "runtime_sources": [
            "ownship packet location/rotation/velocity",
            "target packet location/rotation/velocity",
        ],
    }


def vp_to_local_setpoint(vp: Any, ownship_state: Any) -> GuidanceSetpoint:
    own = _server_visible_state(ownship_state)
    desired = np.asarray(vp, dtype=np.float64)
    if desired.shape != (3,) or not np.all(np.isfinite(desired)):
        raise ValueError("VP must be a finite NED vector")
    relative_ned = desired - own[:3]
    relative_body = body_to_ned_rotation(own[3:6]).T @ relative_ned
    distance = float(np.linalg.norm(relative_body))
    if distance <= 1e-9:
        azimuth = 0.0
        elevation = 0.0
    else:
        azimuth = float(np.degrees(np.arctan2(relative_body[1], relative_body[0])))
        horizontal = float(np.hypot(relative_body[0], relative_body[1]))
        elevation = float(
            np.degrees(np.arctan2(-relative_body[2], max(horizontal, 1e-9)))
        )
    return GuidanceSetpoint(
        local_azimuth_deg=azimuth,
        local_elevation_deg=elevation,
        distance_m=max(1.0, distance),
        target_speed_m_s=max(1.0, float(own[StateIndex.KCAS])),
    )


def apply_tactical_mode(
    mode: str | int,
    bt_action: Any,
    bt_vp: Any,
    ownship_state: Any,
    target_state: Any,
    *,
    mode_config: TacticalModeConfig | None = None,
    controller_config: TacticalControllerConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one Tactical mode while preserving exact Pure BT throttle."""
    started = perf_counter()
    name = tactical_mode_name(mode)
    bt = clip_action(bt_action)
    if name == "BT_DEFAULT":
        return bt.copy(), {
            "mode": name,
            "fallback": True,
            "fallback_reason": "bt_default",
            "bt_action": bt.tolist(),
            "final_action": bt.tolist(),
            "throttle_bt_only": True,
            "latency_ms": (perf_counter() - started) * 1000.0,
        }

    try:
        cfg = controller_config or TacticalControllerConfig()
        cfg.validate()
        desired_vp, mode_info = generate_tactical_vp(
            name, ownship_state, target_state, config=mode_config
        )
        assert desired_vp is not None
        base = vp_to_local_setpoint(bt_vp, ownship_state)
        desired = vp_to_local_setpoint(desired_vp, ownship_state)
        desired = GuidanceSetpoint(
            desired.local_azimuth_deg,
            desired.local_elevation_deg,
            desired.distance_m,
            base.target_speed_m_s,
        )
        final, diagnostics = guidance_to_surface_action(
            bt,
            base,
            desired,
            cfg.guidance_action,
            cfg.guidance_controller,
            ownship_state=ownship_state,
            target_state=target_state,
        )
        if final.shape != (4,) or not np.all(np.isfinite(final)):
            raise ValueError("Tactical controller returned an invalid command")
        final[3] = bt[3]
        if not np.array_equal(final[3:], bt[3:]):
            raise ValueError("Tactical controller changed Pure BT throttle")
        return final, {
            "mode": name,
            "fallback": False,
            "mode_contract": mode_info,
            "mode_config": asdict(mode_config or TacticalModeConfig()),
            "controller_config": {
                "guidance_action": asdict(cfg.guidance_action),
                "guidance_controller": asdict(cfg.guidance_controller),
            },
            "bt_vp": np.asarray(bt_vp, dtype=np.float64).tolist(),
            "desired_vp": desired_vp.tolist(),
            "base_guidance": asdict(base),
            "desired_guidance": asdict(desired),
            "controller": diagnostics,
            "bt_action": bt.tolist(),
            "final_action": final.tolist(),
            "throttle_bt_only": True,
            "latency_ms": (perf_counter() - started) * 1000.0,
        }
    except Exception as exc:
        return bt.copy(), {
            "mode": name,
            "fallback": True,
            "fallback_reason": f"{type(exc).__name__}:{exc}",
            "bt_action": bt.tolist(),
            "final_action": bt.tolist(),
            "throttle_bt_only": True,
            "latency_ms": (perf_counter() - started) * 1000.0,
        }


def tactical_action_contract() -> dict[str, Any]:
    return {
        "contract_version": "temporal_tactical_modes_v4",
        "actions": list(TACTICAL_MODES),
        "level_t1": list(TACTICAL_MODES_T1),
        "level_t2": list(TACTICAL_MODES_T2),
        "hold_frames": list(TACTICAL_HOLD_FRAMES),
        "default_action": "BT_DEFAULT",
        "controller": "vp_error_pd_v2",
        "throttle": "exact same-frame Pure BT",
        "runtime_forbidden": ["health", "Damage", "hidden FDM truth"],
        "fallback": "exception/nonfinite/invalid => exact BT action",
    }
