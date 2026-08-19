from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action
from dogfight.ai.hybrid_action_provider import (
    AimGateConfig,
    OffensiveGateConfig,
    Rear120ActivationGate,
    Rear120GateConfig,
    SafetyVetoConfig,
)
from dogfight.envs.observation import (
    TACTICAL16_FEATURES,
    aim_residual_geometry,
    body_to_ned_rotation,
    normalize,
    official_phase_wez_config,
)
from dogfight.sim.state_schema import StateIndex


GUIDANCE_ACTIONS = (
    "BT_DEFAULT",
    "VP_AZ_POS_SMALL",
    "VP_AZ_NEG_SMALL",
    "VP_EL_POS_SMALL",
    "VP_EL_NEG_SMALL",
    "VP_RANGE_FORWARD_SMALL",
    "VP_RANGE_BACKWARD_SMALL",
    "TARGET_SPEED_UP_SMALL",
    "TARGET_SPEED_DOWN_SMALL",
)
GUIDANCE_ACTION_TO_ID = {name: index for index, name in enumerate(GUIDANCE_ACTIONS)}

GUIDANCE_SELECTOR_FEATURES = (
    *TACTICAL16_FEATURES,
    "signed_aim_azimuth_norm",
    "signed_aim_elevation_norm",
    "los_azimuth_rate_norm",
    "los_elevation_rate_norm",
    "range_norm",
    "closing_rate_norm",
    "target_ata_norm",
    "phase_norm",
    "bt_roll",
    "bt_pitch",
    "bt_yaw",
    "bt_throttle_bipolar",
    "bt_vp_local_azimuth_norm",
    "bt_vp_local_elevation_norm",
    "bt_vp_distance_norm",
    "bt_target_speed_norm",
    "roll_positive_headroom_norm",
    "pitch_positive_headroom_norm",
    "yaw_positive_headroom_norm",
    "roll_negative_headroom_norm",
    "pitch_negative_headroom_norm",
    "yaw_negative_headroom_norm",
    "any_surface_saturation",
    "recent_applied_requested_authority_ratio",
    "previous_selected_action_norm",
    "current_action_hold_norm",
    "gate_elapsed_norm",
    "gate_active",
    "safety_margin_norm",
)
GUIDANCE_SELECTOR_OBSERVATION_SIZE = len(GUIDANCE_SELECTOR_FEATURES)
GUIDANCE_SELECTOR_CONTRACT_VERSION = "guidance_selector_v1"
GUIDANCE_SELECTOR_NORMALIZATION_VERSION = "guidance_selector.norm.v1"


@dataclass(frozen=True)
class GuidanceActionConfig:
    angular_offset_deg: float = 0.5
    range_offset_m: float = 50.0
    target_speed_offset_m_s: float = 10.0

    def validate(self) -> None:
        if not 0.0 < self.angular_offset_deg <= 1.0:
            raise ValueError("angular Guidance offset must be in (0, 1] deg")
        if not 0.0 < self.range_offset_m <= 150.0:
            raise ValueError("range Guidance offset must be in (0, 150] m")
        if not 0.0 < self.target_speed_offset_m_s <= 20.0:
            raise ValueError("target-speed Guidance offset must be in (0, 20] m/s")


@dataclass(frozen=True)
class GuidanceControllerConfig:
    roll_per_angular_action: float = 0.04
    pitch_per_angular_action: float = 0.04
    yaw_per_angular_action: float = 0.02
    pitch_per_range_action: float = 0.01
    pitch_per_speed_action: float = 0.02
    maximum_surface_correction: float = 0.08

    def validate(self) -> None:
        values = tuple(asdict(self).values())
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Guidance controller gains must be finite and non-negative")
        if not 0.0 < self.maximum_surface_correction <= 0.15:
            raise ValueError("maximum surface correction must be in (0, 0.15]")


@dataclass(frozen=True)
class GuidanceRuntimeConfig:
    selector_action_repeat_frames: int = 6
    minimum_action_hold_frames: int = 18
    maximum_active_frames: int = 90
    cooldown_frames: int = 30
    confidence_threshold: float = 0.65
    inference_timeout_s: float = 0.1667
    sim_hz: int = 60

    def validate(self) -> None:
        if self.selector_action_repeat_frames <= 0:
            raise ValueError("selector action repeat must be positive")
        if self.minimum_action_hold_frames < self.selector_action_repeat_frames:
            raise ValueError("minimum action hold must cover at least one selector repeat")
        if self.maximum_active_frames < self.minimum_action_hold_frames:
            raise ValueError("maximum active window must cover minimum hold")
        if self.cooldown_frames < 0 or self.sim_hz != 60:
            raise ValueError("Guidance runtime requires 60Hz and non-negative cooldown")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence threshold must be in [0, 1]")
        if self.inference_timeout_s <= 0.0:
            raise ValueError("inference timeout must be positive")


@dataclass(frozen=True)
class GuidanceSetpoint:
    local_azimuth_deg: float
    local_elevation_deg: float
    distance_m: float
    target_speed_m_s: float


class GuidanceSelector(Protocol):
    def predict(self, observation: np.ndarray) -> tuple[int, float, np.ndarray]: ...


class FixedGuidanceSelector:
    """Deterministic selector used for counterfactuals and contract smoke tests."""

    def __init__(self, action: str | int, confidence: float = 1.0):
        self.action_id = guidance_action_id(action)
        self.confidence = float(confidence)

    def predict(self, observation: np.ndarray) -> tuple[int, float, np.ndarray]:
        observation = validate_guidance_observation(observation)
        probabilities = np.zeros(len(GUIDANCE_ACTIONS), dtype=np.float32)
        probabilities[self.action_id] = 1.0
        return self.action_id, self.confidence, probabilities


class NumpyMLPGuidanceSelector:
    """Small categorical model with an auditable, dependency-light bundle."""

    def __init__(self, bundle_path: str | Path):
        bundle = Path(bundle_path).resolve()
        metadata_path = bundle / "metadata.json"
        weights_path = bundle / "model.npz"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model_kind") != "numpy_mlp_categorical":
            raise ValueError("unsupported Guidance selector model kind")
        if metadata.get("observation_contract") != GUIDANCE_SELECTOR_CONTRACT_VERSION:
            raise ValueError("Guidance observation contract mismatch")
        if int(metadata.get("observation_size", -1)) != GUIDANCE_SELECTOR_OBSERVATION_SIZE:
            raise ValueError("Guidance observation size mismatch")
        if tuple(metadata.get("features", ())) != GUIDANCE_SELECTOR_FEATURES:
            raise ValueError("Guidance feature order mismatch")
        if tuple(metadata.get("actions", ())) != GUIDANCE_ACTIONS:
            raise ValueError("Guidance action contract mismatch")
        expected = str(metadata.get("model_sha256", "")).upper()
        actual = hashlib.sha256(weights_path.read_bytes()).hexdigest().upper()
        if expected != actual:
            raise ValueError(f"Guidance model SHA256 mismatch: expected={expected}, actual={actual}")
        arrays = np.load(weights_path, allow_pickle=False)
        self.weights = [
            np.asarray(arrays["w1"], dtype=np.float32),
            np.asarray(arrays["b1"], dtype=np.float32),
            np.asarray(arrays["w2"], dtype=np.float32),
            np.asarray(arrays["b2"], dtype=np.float32),
            np.asarray(arrays["w3"], dtype=np.float32),
            np.asarray(arrays["b3"], dtype=np.float32),
        ]
        self._validate_shapes()
        self.bundle_path = bundle
        self.metadata = metadata

    def _validate_shapes(self) -> None:
        w1, b1, w2, b2, w3, b3 = self.weights
        if w1.shape[0] != GUIDANCE_SELECTOR_OBSERVATION_SIZE or b1.shape != (w1.shape[1],):
            raise ValueError("invalid Guidance MLP first layer")
        if w2.shape != (w1.shape[1], b2.shape[0]):
            raise ValueError("invalid Guidance MLP second layer")
        if w3.shape != (b2.shape[0], len(GUIDANCE_ACTIONS)) or b3.shape != (
            len(GUIDANCE_ACTIONS),
        ):
            raise ValueError("invalid Guidance MLP output layer")

    def predict(self, observation: np.ndarray) -> tuple[int, float, np.ndarray]:
        vector = validate_guidance_observation(observation)
        w1, b1, w2, b2, w3, b3 = self.weights
        hidden1 = np.tanh(vector @ w1 + b1)
        hidden2 = np.tanh(hidden1 @ w2 + b2)
        logits = hidden2 @ w3 + b3
        logits = logits - float(np.max(logits))
        probabilities = np.exp(logits, dtype=np.float32)
        probabilities /= float(np.sum(probabilities))
        action_id = int(np.argmax(probabilities))
        return action_id, float(probabilities[action_id]), probabilities


def guidance_action_id(action: str | int) -> int:
    if isinstance(action, str):
        if action not in GUIDANCE_ACTION_TO_ID:
            raise ValueError(f"unsupported Guidance action: {action!r}")
        return GUIDANCE_ACTION_TO_ID[action]
    action_id = int(action)
    if not 0 <= action_id < len(GUIDANCE_ACTIONS):
        raise ValueError(f"Guidance action id out of range: {action_id}")
    return action_id


def mirror_guidance_action(action: str | int, axis: str) -> int:
    action_id = guidance_action_id(action)
    name = GUIDANCE_ACTIONS[action_id]
    if axis == "lateral":
        pairs = {
            "VP_AZ_POS_SMALL": "VP_AZ_NEG_SMALL",
            "VP_AZ_NEG_SMALL": "VP_AZ_POS_SMALL",
        }
    elif axis == "vertical":
        pairs = {
            "VP_EL_POS_SMALL": "VP_EL_NEG_SMALL",
            "VP_EL_NEG_SMALL": "VP_EL_POS_SMALL",
        }
    else:
        raise ValueError("mirror axis must be 'lateral' or 'vertical'")
    return GUIDANCE_ACTION_TO_ID[pairs.get(name, name)]


def mirror_guidance_observation(observation, axis: str) -> np.ndarray:
    """Mirror a normalized Guidance observation without changing its contract."""
    mirrored = validate_guidance_observation(observation).copy()
    if axis == "lateral":
        signed_indexes = (0, 2, 7, 11, 16, 18, 24, 26, 28)
    elif axis == "vertical":
        signed_indexes = (1, 8, 12, 17, 19, 25, 29)
    else:
        raise ValueError("mirror axis must be 'lateral' or 'vertical'")
    mirrored[list(signed_indexes)] *= -1.0
    previous_id = int(np.clip(np.rint((mirrored[40] + 1.0) * 4.0), 0, 8))
    mirrored[40] = normalize(float(mirror_guidance_action(previous_id, axis)), 0.0, 8.0)
    return mirrored


def canonicalize_guidance_observation(
    observation,
    *,
    lateral_sign: int = 1,
    vertical_sign: int = 1,
) -> np.ndarray:
    """Map left/down mirror states to the positive canonical quadrant."""
    if lateral_sign not in (-1, 1) or vertical_sign not in (-1, 1):
        raise ValueError("canonical mirror signs must be -1 or +1")
    canonical = validate_guidance_observation(observation).copy()
    if lateral_sign < 0:
        canonical = mirror_guidance_observation(canonical, "lateral")
    if vertical_sign < 0:
        canonical = mirror_guidance_observation(canonical, "vertical")
    return canonical


def guidance_action_delta(
    action: str | int,
    config: GuidanceActionConfig | None = None,
) -> dict[str, float]:
    cfg = config or GuidanceActionConfig()
    cfg.validate()
    name = GUIDANCE_ACTIONS[guidance_action_id(action)]
    delta = {
        "azimuth_deg": 0.0,
        "elevation_deg": 0.0,
        "range_m": 0.0,
        "target_speed_m_s": 0.0,
    }
    if name == "VP_AZ_POS_SMALL":
        delta["azimuth_deg"] = cfg.angular_offset_deg
    elif name == "VP_AZ_NEG_SMALL":
        delta["azimuth_deg"] = -cfg.angular_offset_deg
    elif name == "VP_EL_POS_SMALL":
        delta["elevation_deg"] = cfg.angular_offset_deg
    elif name == "VP_EL_NEG_SMALL":
        delta["elevation_deg"] = -cfg.angular_offset_deg
    elif name == "VP_RANGE_FORWARD_SMALL":
        delta["range_m"] = cfg.range_offset_m
    elif name == "VP_RANGE_BACKWARD_SMALL":
        delta["range_m"] = -cfg.range_offset_m
    elif name == "TARGET_SPEED_UP_SMALL":
        delta["target_speed_m_s"] = cfg.target_speed_offset_m_s
    elif name == "TARGET_SPEED_DOWN_SMALL":
        delta["target_speed_m_s"] = -cfg.target_speed_offset_m_s
    return delta


def compose_guidance_setpoint(
    base: GuidanceSetpoint,
    action: str | int,
    config: GuidanceActionConfig | None = None,
) -> GuidanceSetpoint:
    delta = guidance_action_delta(action, config)
    return GuidanceSetpoint(
        local_azimuth_deg=float(base.local_azimuth_deg + delta["azimuth_deg"]),
        local_elevation_deg=float(base.local_elevation_deg + delta["elevation_deg"]),
        distance_m=float(max(1.0, base.distance_m + delta["range_m"])),
        target_speed_m_s=float(max(1.0, base.target_speed_m_s + delta["target_speed_m_s"])),
    )


def guidance_to_surface_action(
    bt_action,
    base: GuidanceSetpoint,
    corrected: GuidanceSetpoint,
    action_config: GuidanceActionConfig | None = None,
    controller_config: GuidanceControllerConfig | None = None,
) -> tuple[np.ndarray, dict]:
    bt = clip_action(bt_action)
    action_cfg = action_config or GuidanceActionConfig()
    control_cfg = controller_config or GuidanceControllerConfig()
    action_cfg.validate()
    control_cfg.validate()
    az_units = (corrected.local_azimuth_deg - base.local_azimuth_deg) / action_cfg.angular_offset_deg
    el_units = (corrected.local_elevation_deg - base.local_elevation_deg) / action_cfg.angular_offset_deg
    range_units = (corrected.distance_m - base.distance_m) / action_cfg.range_offset_m
    speed_units = (corrected.target_speed_m_s - base.target_speed_m_s) / action_cfg.target_speed_offset_m_s
    requested = np.array(
        [
            control_cfg.roll_per_angular_action * az_units,
            control_cfg.pitch_per_angular_action * el_units
            - control_cfg.pitch_per_range_action * range_units
            - control_cfg.pitch_per_speed_action * speed_units,
            control_cfg.yaw_per_angular_action * az_units,
        ],
        dtype=np.float32,
    )
    requested = np.clip(
        requested,
        -control_cfg.maximum_surface_correction,
        control_cfg.maximum_surface_correction,
    )
    positive = np.clip(1.0 - bt[:3], 0.0, 2.0)
    negative = np.clip(bt[:3] + 1.0, 0.0, 2.0)
    directional = np.where(requested >= 0.0, positive, negative)
    applied = requested * np.clip(directional, 0.0, 1.0)
    final = bt.copy()
    final[:3] = np.clip(bt[:3] + applied, -1.0, 1.0)
    final[3] = bt[3]
    ratio = np.zeros(3, dtype=np.float32)
    nonzero = np.abs(requested) > 1e-12
    ratio[nonzero] = np.abs(applied[nonzero]) / np.abs(requested[nonzero])
    return final, {
        "requested_surface_correction": requested.tolist(),
        "applied_surface_correction": (final[:3] - bt[:3]).tolist(),
        "applied_to_requested_ratio": ratio.tolist(),
        "positive_headroom": positive.tolist(),
        "negative_headroom": negative.tolist(),
    }


def _vp_local_setpoint(bt_result: ActionResult, ownship_state) -> GuidanceSetpoint:
    own = np.asarray(ownship_state, dtype=np.float64)
    vp = np.asarray(bt_result.info.get("vp", own[:3]), dtype=np.float64)
    if vp.shape != (3,) or not np.all(np.isfinite(vp)):
        vp = own[:3].copy()
    relative_ned = np.array(
        [vp[0] - own[StateIndex.N], vp[1] - own[StateIndex.E], own[StateIndex.ALT] - vp[2]],
        dtype=np.float64,
    )
    relative_body = body_to_ned_rotation(own[3:6]).T @ relative_ned
    distance = float(np.linalg.norm(relative_body))
    if distance <= 1e-6:
        azimuth = elevation = 0.0
    else:
        azimuth = float(np.degrees(np.arctan2(relative_body[1], relative_body[0])))
        horizontal = float(np.hypot(relative_body[0], relative_body[1]))
        elevation = float(np.degrees(np.arctan2(-relative_body[2], max(horizontal, 1e-9))))
    throttle_info = bt_result.info.get("throttle_control", {})
    target_speed = float(throttle_info.get("target_speed_mps", own[StateIndex.KCAS]))
    return GuidanceSetpoint(azimuth, elevation, max(distance, 1.0), max(target_speed, 1.0))


def validate_guidance_observation(observation) -> np.ndarray:
    vector = np.asarray(observation, dtype=np.float32)
    if vector.shape != (GUIDANCE_SELECTOR_OBSERVATION_SIZE,):
        raise ValueError(
            f"Guidance observation must have shape ({GUIDANCE_SELECTOR_OBSERVATION_SIZE},), "
            f"got {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("Guidance observation contains nonfinite values")
    return np.clip(vector, -1.0, 1.0)


def build_guidance_observation(
    tactical16,
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
    tactical = np.asarray(tactical16, dtype=np.float32)
    if tactical.shape != (16,) or not np.all(np.isfinite(tactical)):
        raise ValueError("Guidance selector requires a finite Tactical16 observation")
    own = np.asarray(ownship_state, dtype=np.float64)
    target = np.asarray(target_state, dtype=np.float64)
    bt = np.asarray(bt_action, dtype=np.float32)
    if bt.shape != (4,) or not np.all(np.isfinite(bt)):
        raise ValueError("Guidance selector requires a finite BT action")
    geometry = aim_residual_geometry(own, target)
    phase = official_phase_wez_config(sim_time_s)["phase"]
    positive = np.clip(1.0 - bt[:3], 0.0, 2.0) - 1.0
    negative = np.clip(bt[:3] + 1.0, 0.0, 2.0) - 1.0
    any_saturated = 1.0 if np.any(np.isclose(np.abs(bt[:3]), 1.0, atol=1e-6)) else -1.0
    safety = safety_config or SafetyVetoConfig()
    altitude_margin = (float(own[StateIndex.ALT]) - safety.minimum_altitude_m) / 1000.0
    speed_margin = (float(own[StateIndex.KCAS]) - safety.minimum_speed_m_s) / 100.0
    closing_margin = (safety.maximum_closing_rate_m_s - geometry["closing_rate_m_s"]) / 250.0
    safety_margin = float(np.clip(min(altitude_margin, speed_margin, closing_margin), -1.0, 1.0))
    observation = np.concatenate(
        (
            np.clip(tactical, -1.0, 1.0),
            np.array(
                [
                    normalize(geometry["aim_azimuth_deg"], -15.0, 15.0),
                    normalize(geometry["aim_elevation_deg"], -15.0, 15.0),
                    normalize(geometry["los_azimuth_rate_deg_s"], -15.0, 15.0),
                    normalize(geometry["los_elevation_rate_deg_s"], -15.0, 15.0),
                    normalize(geometry["distance_m"], 0.0, 3000.0),
                    normalize(geometry["closing_rate_m_s"], -250.0, 250.0),
                    normalize(geometry["target_ata_deg"], 90.0, 180.0),
                    normalize(float(phase), 1.0, 3.0),
                    bt[0],
                    bt[1],
                    bt[2],
                    2.0 * bt[3] - 1.0,
                    normalize(base_guidance.local_azimuth_deg, -45.0, 45.0),
                    normalize(base_guidance.local_elevation_deg, -45.0, 45.0),
                    normalize(base_guidance.distance_m, 0.0, 5000.0),
                    normalize(base_guidance.target_speed_m_s, 100.0, 400.0),
                    *positive,
                    *negative,
                    any_saturated,
                    np.clip(2.0 * recent_authority_ratio - 1.0, -1.0, 1.0),
                    normalize(float(previous_action_id), 0.0, len(GUIDANCE_ACTIONS) - 1.0),
                    normalize(float(action_hold_frames), 0.0, float(max(1, minimum_action_hold_frames))),
                    normalize(float(gate_elapsed_frames), 0.0, float(max(1, maximum_active_frames))),
                    1.0 if gate_active else -1.0,
                    safety_margin,
                ],
                dtype=np.float32,
            ),
        )
    ).astype(np.float32)
    return validate_guidance_observation(observation)


class GuidanceSelectorActionProvider(ActionProvider):
    """Run BT at 60Hz and select bounded Guidance primitives at 10Hz."""

    def __init__(
        self,
        bt_provider: ActionProvider,
        selector: GuidanceSelector,
        *,
        action_config: GuidanceActionConfig | dict | None = None,
        controller_config: GuidanceControllerConfig | dict | None = None,
        runtime_config: GuidanceRuntimeConfig | dict | None = None,
        rear120_config: Rear120GateConfig | dict | None = None,
        aim_config: AimGateConfig | dict | None = None,
        offensive_config: OffensiveGateConfig | dict | None = None,
        safety_config: SafetyVetoConfig | dict | None = None,
    ):
        self.bt_provider = bt_provider
        self.selector = selector
        self.action_config = _coerce_config(action_config, GuidanceActionConfig)
        self.controller_config = _coerce_config(controller_config, GuidanceControllerConfig)
        self.runtime_config = _coerce_config(runtime_config, GuidanceRuntimeConfig)
        self.safety_config = _coerce_config(safety_config, SafetyVetoConfig)
        self.action_config.validate()
        self.controller_config.validate()
        self.runtime_config.validate()
        self.safety_config.validate()
        self.gate = Rear120ActivationGate(
            rear120_config,
            aim_config,
            offensive_config,
            self.safety_config,
        )
        self.reset(None)

    def reset(self, context: ActionContext | None = None) -> None:
        self.bt_provider.reset(context)
        self.gate.reset()
        self._current_action_id = 0
        self._action_hold_frames = 0
        self._gate_elapsed_frames = 0
        self._cooldown_remaining = 0
        self._recent_authority_ratio = 1.0
        self._selector_calls = 0
        self._selector_latency_ms: list[float] = []
        self._fallback_counts: dict[str, int] = {}
        self._action_counts = np.zeros(len(GUIDANCE_ACTIONS), dtype=np.int64)
        self._gate_steps = 0
        self._nonzero_intervention_frames = 0
        self._requested_abs_sum = np.zeros(3, dtype=np.float64)
        self._applied_abs_sum = np.zeros(3, dtype=np.float64)
        self._throttle_violation_steps = 0
        self._last_frame: dict = {}

    def _fallback(
        self,
        reason: str,
        bt_action: np.ndarray,
        gate_info: dict,
        *,
        reset_action: bool = True,
        count_action: bool = True,
    ) -> ActionResult:
        self._fallback_counts[reason] = self._fallback_counts.get(reason, 0) + 1
        if reset_action:
            self._current_action_id = 0
            self._action_hold_frames = 0
        frame = {
            "mode": "guidance_selector",
            "gate": gate_info,
            "selected_action_id": 0,
            "selected_action": GUIDANCE_ACTIONS[0],
            "fallback_reason": reason,
            "bt_action": bt_action.tolist(),
            "final_action": bt_action.tolist(),
            "throttle_bt_only": True,
        }
        if count_action:
            self._action_counts[0] += 1
        self._last_frame = frame
        return ActionResult(bt_action.copy(), "bt_guidance_selector", 1.0, frame)

    def compute_action(self, context: ActionContext) -> ActionResult:
        try:
            bt_result = self.bt_provider.compute_action(context)
            raw_bt = np.asarray(bt_result.action, dtype=np.float32)
            if raw_bt.shape != (4,) or not np.all(np.isfinite(raw_bt)):
                raise ValueError("invalid BT action")
            bt_action = clip_action(raw_bt)
        except Exception:
            raise
        gate_info = self.gate.update(
            context.ownship_state,
            context.target_state,
            sim_time_s=context.info.get("sim_time_s"),
            bt_action=bt_action,
        )
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
        if not gate_info["active"] or self._cooldown_remaining > 0:
            self._gate_elapsed_frames = 0
            self._current_action_id = 0
            self._action_hold_frames = 0
            return self._fallback(
                "gate_off" if not gate_info["active"] else "cooldown",
                bt_action,
                gate_info,
            )
        self._gate_steps += 1
        self._gate_elapsed_frames += 1
        if self._gate_elapsed_frames > self.runtime_config.maximum_active_frames:
            self._cooldown_remaining = self.runtime_config.cooldown_frames
            return self._fallback("maximum_active_window", bt_action, gate_info)

        try:
            base = _vp_local_setpoint(bt_result, context.ownship_state)
            observation = build_guidance_observation(
                context.observation,
                context.ownship_state,
                context.target_state,
                bt_action,
                base,
                sim_time_s=float(context.info.get("sim_time_s", 0.0)),
                previous_action_id=self._current_action_id,
                action_hold_frames=self._action_hold_frames,
                gate_elapsed_frames=self._gate_elapsed_frames,
                gate_active=True,
                minimum_action_hold_frames=self.runtime_config.minimum_action_hold_frames,
                maximum_active_frames=self.runtime_config.maximum_active_frames,
                recent_authority_ratio=self._recent_authority_ratio,
                safety_config=self.safety_config,
            )
        except Exception as exc:
            return self._fallback(f"observation_{type(exc).__name__}", bt_action, gate_info)

        may_refresh = self._action_hold_frames >= self.runtime_config.minimum_action_hold_frames
        refresh = self._action_hold_frames == 0 or (
            may_refresh
            and self._gate_elapsed_frames % self.runtime_config.selector_action_repeat_frames == 0
        )
        confidence = 1.0
        probabilities = None
        if refresh:
            started = perf_counter()
            try:
                action_id, confidence, probabilities = self.selector.predict(observation)
            except Exception as exc:
                elapsed = perf_counter() - started
                self._selector_latency_ms.append(elapsed * 1000.0)
                return self._fallback(f"inference_{type(exc).__name__}", bt_action, gate_info)
            elapsed = perf_counter() - started
            self._selector_latency_ms.append(elapsed * 1000.0)
            self._selector_calls += 1
            try:
                action_id = guidance_action_id(action_id)
            except Exception:
                return self._fallback("invalid_action_id", bt_action, gate_info)
            if not np.isfinite(confidence) or elapsed > self.runtime_config.inference_timeout_s:
                reason = "inference_nonfinite" if not np.isfinite(confidence) else "inference_timeout"
                return self._fallback(reason, bt_action, gate_info)
            if probabilities is not None:
                probabilities = np.asarray(probabilities, dtype=np.float32)
                if probabilities.shape != (len(GUIDANCE_ACTIONS),) or not np.all(
                    np.isfinite(probabilities)
                ):
                    return self._fallback("invalid_probabilities", bt_action, gate_info)
            if confidence < self.runtime_config.confidence_threshold:
                action_id = 0
            self._current_action_id = action_id
            self._action_hold_frames = 0

        action_id = self._current_action_id
        self._action_hold_frames += 1
        self._action_counts[action_id] += 1
        if action_id == 0:
            return self._fallback(
                "bt_default",
                bt_action,
                gate_info,
                reset_action=False,
                count_action=False,
            )

        corrected = compose_guidance_setpoint(base, action_id, self.action_config)
        final, controller = guidance_to_surface_action(
            bt_action,
            base,
            corrected,
            self.action_config,
            self.controller_config,
        )
        if final.shape != (4,) or not np.all(np.isfinite(final)):
            return self._fallback("controller_invalid", bt_action, gate_info)
        final[3] = bt_action[3]
        if not np.array_equal(final[3:], bt_action[3:]):
            self._throttle_violation_steps += 1
            return self._fallback("throttle_violation", bt_action, gate_info)
        requested = np.asarray(controller["requested_surface_correction"], dtype=np.float64)
        applied = np.asarray(controller["applied_surface_correction"], dtype=np.float64)
        self._requested_abs_sum += np.abs(requested)
        self._applied_abs_sum += np.abs(applied)
        nonzero = np.abs(requested) > 1e-12
        self._recent_authority_ratio = float(
            np.mean(np.abs(applied[nonzero]) / np.abs(requested[nonzero])) if np.any(nonzero) else 1.0
        )
        self._nonzero_intervention_frames += int(np.any(np.abs(applied) > 1e-12))
        frame = {
            "mode": "guidance_selector",
            "gate": gate_info,
            "observation_contract": GUIDANCE_SELECTOR_CONTRACT_VERSION,
            "observation": observation.tolist(),
            "selected_action_id": action_id,
            "selected_action": GUIDANCE_ACTIONS[action_id],
            "selector_refreshed": refresh,
            "selector_confidence": float(confidence),
            "selector_probabilities": probabilities.tolist() if probabilities is not None else None,
            "base_guidance": asdict(base),
            "corrected_guidance": asdict(corrected),
            "controller": controller,
            "bt_action": bt_action.tolist(),
            "final_action": final.tolist(),
            "throttle_bt_only": True,
        }
        self._last_frame = frame
        return ActionResult(final, "bt_guidance_selector", float(confidence), frame)

    def telemetry(self) -> dict:
        latency = np.asarray(self._selector_latency_ms, dtype=np.float64)
        total = int(np.sum(self._action_counts))
        return {
            **self.gate.telemetry(),
            "mode": "guidance_selector",
            "observation_contract": GUIDANCE_SELECTOR_CONTRACT_VERSION,
            "observation_size": GUIDANCE_SELECTOR_OBSERVATION_SIZE,
            "action_config": asdict(self.action_config),
            "controller_config": asdict(self.controller_config),
            "runtime_config": asdict(self.runtime_config),
            "selector_inference_calls": self._selector_calls,
            "selector_inference_latency_ms_p50": float(np.percentile(latency, 50)) if latency.size else 0.0,
            "selector_inference_latency_ms_p95": float(np.percentile(latency, 95)) if latency.size else 0.0,
            "selector_inference_latency_ms_p99": float(np.percentile(latency, 99)) if latency.size else 0.0,
            "selector_inference_latency_ms_max": float(np.max(latency)) if latency.size else 0.0,
            "selector_inference_over_166_7ms": int(np.sum(latency > 166.7)),
            "action_counts": {
                name: int(self._action_counts[index]) for index, name in enumerate(GUIDANCE_ACTIONS)
            },
            "action_distribution": {
                name: float(self._action_counts[index] / max(1, total))
                for index, name in enumerate(GUIDANCE_ACTIONS)
            },
            "gate_active_frames": self._gate_steps,
            "nonzero_intervention_frames": self._nonzero_intervention_frames,
            "requested_guidance_surface_abs_sum": self._requested_abs_sum.tolist(),
            "applied_guidance_surface_abs_sum": self._applied_abs_sum.tolist(),
            "throttle_violation_steps": self._throttle_violation_steps,
            "fallback_counts": dict(self._fallback_counts),
            "last_frame": dict(self._last_frame),
        }

    def close(self) -> None:
        self.bt_provider.close()


def _coerce_config(value, kind):
    if value is None:
        return kind()
    if isinstance(value, kind):
        return value
    if isinstance(value, dict):
        return kind(**value)
    raise TypeError(f"expected {kind.__name__}, dict, or None")
