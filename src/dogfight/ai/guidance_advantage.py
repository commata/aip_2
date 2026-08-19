from __future__ import annotations

from typing import Any

import numpy as np

from dogfight.ai.guidance_selector import GuidanceSetpoint
from dogfight.ai.hybrid_action_provider import SafetyVetoConfig
from dogfight.envs.observation import aim_residual_geometry, normalize, official_phase_wez_config
from dogfight.sim.state_schema import StateIndex


GUIDANCE_ADVANTAGE_ACTIONS = (
    "BT_DEFAULT",
    "VP_AZ_POS_SMALL",
    "VP_AZ_NEG_SMALL",
    "VP_EL_POS_SMALL",
    "VP_EL_NEG_SMALL",
)
GUIDANCE_SERVER_CONTRACT_VERSION = "guidance_selector_server_v2"
GUIDANCE_SERVER_NORMALIZATION_VERSION = "guidance_selector_server.norm.v2"


def _spec(name: str, source: str, unit: str, normalization: str) -> dict[str, str]:
    return {
        "name": name,
        "dtype": "float32",
        "source": source,
        "unit": unit,
        "normalization": normalization,
    }


GUIDANCE_SERVER_FEATURE_SPECS = (
    _spec("ownship_roll_norm", "ownship.rotation.roll", "deg", "[-180,180]->[-1,1]"),
    _spec("ownship_pitch_norm", "ownship.rotation.pitch", "deg", "[-90,90]->[-1,1]"),
    _spec("ownship_yaw_norm", "ownship.rotation.yaw", "deg", "[0,360]->[-1,1]"),
    _spec("ownship_speed_norm", "norm(ownship.velocity)", "m/s", "[100,400]->[-1,1]"),
    _spec("ownship_altitude_norm", "-ownship.location.down", "m", "[0,10000]->[-1,1]"),
    _spec("target_roll_norm", "target.rotation.roll", "deg", "[-180,180]->[-1,1]"),
    _spec("target_pitch_norm", "target.rotation.pitch", "deg", "[-90,90]->[-1,1]"),
    _spec("target_yaw_norm", "target.rotation.yaw", "deg", "[0,360]->[-1,1]"),
    _spec("target_speed_norm", "norm(target.velocity)", "m/s", "[100,400]->[-1,1]"),
    _spec("target_altitude_norm", "-target.location.down", "m", "[0,10000]->[-1,1]"),
    _spec("delta_north_norm", "target.location-ownship.location", "m", "[-3000,3000]->[-1,1]"),
    _spec("delta_east_norm", "target.location-ownship.location", "m", "[-3000,3000]->[-1,1]"),
    _spec("delta_down_norm", "target.location-ownship.location", "m", "[-3000,3000]->[-1,1]"),
    _spec("signed_aim_azimuth_norm", "derived current/local history", "deg", "[-15,15]->[-1,1]"),
    _spec("signed_aim_elevation_norm", "derived current/local history", "deg", "[-15,15]->[-1,1]"),
    _spec("los_azimuth_rate_norm", "derived current/local history", "deg/s", "[-15,15]->[-1,1]"),
    _spec("los_elevation_rate_norm", "derived current/local history", "deg/s", "[-15,15]->[-1,1]"),
    _spec("range_norm", "derived relative position", "m", "[0,3000]->[-1,1]"),
    _spec("closing_rate_norm", "derived position/velocity", "m/s", "[-250,250]->[-1,1]"),
    _spec("target_ata_norm", "derived target frame", "deg", "[90,180]->[-1,1]"),
    _spec("phase_norm", "local frame counter", "phase", "[1,3]->[-1,1]"),
    _spec("bt_roll", "same-frame BT action", "command", "identity [-1,1]"),
    _spec("bt_pitch", "same-frame BT action", "command", "identity [-1,1]"),
    _spec("bt_yaw", "same-frame BT action", "command", "identity [-1,1]"),
    _spec("bt_throttle_bipolar", "same-frame BT action", "command", "[0,1]->[-1,1]"),
    _spec("bt_vp_local_azimuth_norm", "same-frame BT VP", "deg", "[-45,45]->[-1,1]"),
    _spec("bt_vp_local_elevation_norm", "same-frame BT VP", "deg", "[-45,45]->[-1,1]"),
    _spec("bt_vp_distance_norm", "same-frame BT VP", "m", "[0,5000]->[-1,1]"),
    _spec("bt_target_speed_norm", "same-frame BT context", "m/s", "[100,400]->[-1,1]"),
    *tuple(
        _spec(f"{axis}_{direction}_headroom_norm", "same-frame BT action", "command", "[0,2]->[-1,1]")
        for direction in ("positive", "negative")
        for axis in ("roll", "pitch", "yaw")
    ),
    _spec("any_surface_saturation", "same-frame BT action", "boolean", "false=-1,true=1"),
    _spec("recent_applied_requested_authority_ratio", "local history", "ratio", "[0,1]->[-1,1]"),
    _spec("previous_selected_action_norm", "local history", "action id", "[0,4]->[-1,1]"),
    _spec("current_action_hold_norm", "local history", "frame", "[0,min_hold]->[-1,1]"),
    _spec("gate_elapsed_norm", "local history", "frame", "[0,max_active]->[-1,1]"),
    _spec("gate_active", "local gate", "boolean", "false=-1,true=1"),
    _spec("safety_margin_norm", "derived server state", "ratio", "clipped [-1,1]"),
)
GUIDANCE_SERVER_FEATURES = tuple(spec["name"] for spec in GUIDANCE_SERVER_FEATURE_SPECS)
GUIDANCE_SERVER_OBSERVATION_SIZE = len(GUIDANCE_SERVER_FEATURES)


def validate_server_guidance_observation(observation: Any) -> np.ndarray:
    vector = np.asarray(observation, dtype=np.float32)
    if vector.shape != (GUIDANCE_SERVER_OBSERVATION_SIZE,):
        raise ValueError(
            f"server Guidance observation must have shape "
            f"({GUIDANCE_SERVER_OBSERVATION_SIZE},), got {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("server Guidance observation contains nonfinite values")
    return np.clip(vector, -1.0, 1.0)


def build_server_guidance_observation(
    ownship_state,
    target_state,
    bt_action,
    base_guidance: GuidanceSetpoint,
    *,
    sim_time_s: float,
    previous_action_id: int,
    action_hold_frames: int,
    gate_elapsed_frames: int,
    gate_active: bool,
    minimum_action_hold_frames: int,
    maximum_active_frames: int,
    recent_authority_ratio: float = 1.0,
    safety_config: SafetyVetoConfig | None = None,
) -> np.ndarray:
    own = np.asarray(ownship_state, dtype=np.float64)
    target = np.asarray(target_state, dtype=np.float64)
    bt = np.asarray(bt_action, dtype=np.float32)
    if own.shape[0] <= StateIndex.HEALTH or target.shape[0] <= StateIndex.HEALTH:
        raise ValueError("server Guidance state vectors are incomplete")
    if bt.shape != (4,) or not np.all(np.isfinite(bt)):
        raise ValueError("server Guidance requires a finite BT action")
    geometry = aim_residual_geometry(own, target)
    phase = official_phase_wez_config(sim_time_s)["phase"]
    positive = np.clip(1.0 - bt[:3], 0.0, 2.0) - 1.0
    negative = np.clip(bt[:3] + 1.0, 0.0, 2.0) - 1.0
    safety = safety_config or SafetyVetoConfig()
    altitude_margin = (float(own[StateIndex.ALT]) - safety.minimum_altitude_m) / 1000.0
    speed_margin = (float(own[StateIndex.KCAS]) - safety.minimum_speed_m_s) / 100.0
    closing_margin = (safety.maximum_closing_rate_m_s - geometry["closing_rate_m_s"]) / 250.0
    safety_margin = float(np.clip(min(altitude_margin, speed_margin, closing_margin), -1, 1))
    delta = target[:3] - own[:3]
    vector = np.asarray(
        [
            normalize(own[StateIndex.ROLL], -180.0, 180.0),
            normalize(own[StateIndex.PITCH], -90.0, 90.0),
            normalize(own[StateIndex.YAW], 0.0, 360.0),
            normalize(own[StateIndex.KCAS], 100.0, 400.0),
            normalize(own[StateIndex.ALT], 0.0, 10000.0),
            normalize(target[StateIndex.ROLL], -180.0, 180.0),
            normalize(target[StateIndex.PITCH], -90.0, 90.0),
            normalize(target[StateIndex.YAW], 0.0, 360.0),
            normalize(target[StateIndex.KCAS], 100.0, 400.0),
            normalize(target[StateIndex.ALT], 0.0, 10000.0),
            normalize(delta[0], -3000.0, 3000.0),
            normalize(delta[1], -3000.0, 3000.0),
            normalize(delta[2], -3000.0, 3000.0),
            normalize(geometry["aim_azimuth_deg"], -15.0, 15.0),
            normalize(geometry["aim_elevation_deg"], -15.0, 15.0),
            normalize(geometry["los_azimuth_rate_deg_s"], -15.0, 15.0),
            normalize(geometry["los_elevation_rate_deg_s"], -15.0, 15.0),
            normalize(geometry["distance_m"], 0.0, 3000.0),
            normalize(geometry["closing_rate_m_s"], -250.0, 250.0),
            normalize(geometry["target_ata_deg"], 90.0, 180.0),
            normalize(float(phase), 1.0, 3.0),
            bt[0], bt[1], bt[2], 2.0 * bt[3] - 1.0,
            normalize(base_guidance.local_azimuth_deg, -45.0, 45.0),
            normalize(base_guidance.local_elevation_deg, -45.0, 45.0),
            normalize(base_guidance.distance_m, 0.0, 5000.0),
            normalize(base_guidance.target_speed_m_s, 100.0, 400.0),
            *positive, *negative,
            1.0 if np.any(np.isclose(np.abs(bt[:3]), 1.0, atol=1e-6)) else -1.0,
            np.clip(2.0 * recent_authority_ratio - 1.0, -1.0, 1.0),
            normalize(float(previous_action_id), 0.0, len(GUIDANCE_ADVANTAGE_ACTIONS) - 1),
            normalize(float(action_hold_frames), 0.0, max(1, minimum_action_hold_frames)),
            normalize(float(gate_elapsed_frames), 0.0, max(1, maximum_active_frames)),
            1.0 if gate_active else -1.0,
            safety_margin,
        ],
        dtype=np.float32,
    )
    return validate_server_guidance_observation(vector)


def server_observation_contract() -> dict[str, Any]:
    return {
        "contract_version": GUIDANCE_SERVER_CONTRACT_VERSION,
        "normalization_version": GUIDANCE_SERVER_NORMALIZATION_VERSION,
        "dtype": "float32",
        "size": GUIDANCE_SERVER_OBSERVATION_SIZE,
        "actions": list(GUIDANCE_ADVANTAGE_ACTIONS),
        "features": [dict(spec) for spec in GUIDANCE_SERVER_FEATURE_SPECS],
        "health_features": [],
        "runtime_sources": [
            "ownship location/rotation/velocity",
            "target location/rotation/velocity",
            "same-frame BT action/VP/controller context",
            "local frame counter and bounded local history",
        ],
        "offline_label_only": ["ownship health", "target health", "Damage"],
    }
