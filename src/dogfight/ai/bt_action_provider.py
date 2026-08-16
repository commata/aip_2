from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action
from dogfight.ai.native_bt import AIPilot
from dogfight.sim.state_schema import StateIndex


REMOTE_BT_FIGHTER_ID = 0
SAFE_VP = np.zeros(3, dtype=np.float32)


@dataclass(frozen=True)
class TurnThrottleConfig:
    """Energy-management settings for native BTs that hard-code full throttle."""

    corner_speed_eas_mps: float = 210.0
    max_corner_speed_tas_mps: float = 340.0
    turn_activation: float = 0.55
    full_throttle_below_ratio: float = 0.82
    trim_throttle: float = 0.90
    speed_error_gain: float = 2.0
    minimum_turn_throttle: float = 0.35
    full_throttle_threshold: float = 0.98


def _isa_density_ratio(altitude_m: float) -> float:
    """Return ISA troposphere density divided by sea-level density."""
    altitude_m = float(np.clip(altitude_m, 0.0, 11000.0))
    temperature_ratio = 1.0 - 2.25577e-5 * altitude_m
    return max(0.20, temperature_ratio**4.25588)


def optimize_full_turn_throttle(
    raw_throttle: float,
    roll_cmd: float,
    pitch_cmd: float,
    speed_mps: float,
    altitude_m: float,
    *,
    speed_is_true_airspeed: bool,
    config: TurnThrottleConfig,
) -> tuple[float, dict[str, float | bool | str]]:
    """Replace only a native full-throttle command during an aggressive turn.

    The local FDM exposes calibrated airspeed while Unreal exposes velocity
    magnitude (approximately TAS), so the latter is converted to an
    altitude-adjusted corner-speed target. Commands below full throttle are
    intentional tactical decisions from XML tasks and are never overwritten.
    """
    throttle = float(np.clip(raw_throttle, 0.0, 1.0))
    turn_demand = float(np.clip(max(abs(roll_cmd), abs(pitch_cmd)), 0.0, 1.0))
    info: dict[str, float | bool | str] = {
        "active": False,
        "raw_throttle": throttle,
        "turn_demand": turn_demand,
        "speed_mps": float(speed_mps),
    }

    if throttle < config.full_throttle_threshold:
        info["reason"] = "tactical_throttle"
        return throttle, info
    if turn_demand < config.turn_activation:
        info["reason"] = "not_aggressive_turn"
        return throttle, info
    if not np.isfinite(speed_mps) or speed_mps <= 0.0:
        info["reason"] = "invalid_speed"
        return throttle, info

    target_speed = config.corner_speed_eas_mps
    if speed_is_true_airspeed:
        target_speed /= np.sqrt(_isa_density_ratio(altitude_m))
        target_speed = min(target_speed, config.max_corner_speed_tas_mps)

    info["target_speed_mps"] = float(target_speed)
    if speed_mps <= target_speed * config.full_throttle_below_ratio:
        info["reason"] = "low_energy"
        return throttle, info

    speed_error_ratio = (target_speed - speed_mps) / target_speed
    corner_throttle = float(
        np.clip(
            config.trim_throttle + config.speed_error_gain * speed_error_ratio,
            config.minimum_turn_throttle,
            1.0,
        )
    )
    blend = (turn_demand - config.turn_activation) / (1.0 - config.turn_activation)
    optimized = float(np.clip(throttle + blend * (corner_throttle - throttle), 0.0, 1.0))
    info.update(
        {
            "active": True,
            "reason": "corner_speed_control",
            "corner_throttle": corner_throttle,
            "output_throttle": optimized,
        }
    )
    return optimized, info


class BTActionProvider(ActionProvider):
    def __init__(
        self,
        dll_name: str = "AIP_DCS_base.dll",
        ai_pilot: AIPilot | None = None,
        confidence: float = 0.85,
        turn_throttle_config: TurnThrottleConfig | None = None,
        enable_turn_throttle_optimization: bool = True,
    ):
        self.ai_pilot = ai_pilot if ai_pilot is not None else AIPilot(dll_name)
        self.confidence = confidence
        self.turn_throttle_config = turn_throttle_config or TurnThrottleConfig()
        self.enable_turn_throttle_optimization = bool(
            enable_turn_throttle_optimization
        )
        self._registered_fighter_ids: dict[int, int] = {}

    def reset(self, context: ActionContext | None = None) -> None:
        # 2026-05-26: Keep native BT alive across episode resets for multienv.
        return None

    def _remove_behavior_tree(self, fighter_id: int) -> None:
        try:
            self.ai_pilot.RemoveBT(fighter_id)
        except Exception:
            pass
        self._registered_fighter_ids.pop(fighter_id, None)

    def _ensure_behavior_tree(self, context: ActionContext) -> None:
        model = context.sim.get_model()
        fighter_id = model.fighterID
        force_side = int(model._forceSide)
        registered_force = self._registered_fighter_ids.get(fighter_id)
        if registered_force == force_side:
            return
        if registered_force is not None:
            raise RuntimeError(
                "BT fighter id reused with a different force side: "
                f"fighter_id={fighter_id}, previous={registered_force}, current={force_side}"
            )
        # 2026-05-26: Create native BT once and reuse it until provider close().
        self.ai_pilot.CreateBehaviorTree(fighter_id, force_side)
        self._registered_fighter_ids[fighter_id] = force_side

    def _ensure_remote_behavior_tree(self, fighter_id: int, force_side: int) -> None:
        registered_force = self._registered_fighter_ids.get(fighter_id)
        if registered_force == force_side:
            return
        if registered_force is not None:
            raise RuntimeError(
                "Remote BT fighter id reused with a different force side: "
                f"fighter_id={fighter_id}, previous={registered_force}, current={force_side}"
            )
        # 2026-05-26: Remote BT is created once and reused until provider close().
        self.ai_pilot.CreateBehaviorTree(fighter_id, force_side)
        self._registered_fighter_ids[fighter_id] = force_side

    @staticmethod
    def _vp_to_array(vp) -> tuple[np.ndarray, bool]:
        vp_array = np.array([vp.X, vp.Y, vp.Z], dtype=np.float32)
        if np.all(np.isfinite(vp_array)):
            return vp_array, True
        return SAFE_VP.copy(), False

    def compute_action(self, context: ActionContext) -> ActionResult:
        if context.sim is None or context.opponent_sim is None:
            return self._compute_remote_action(context)

        self._ensure_behavior_tree(context)
        model = context.sim.get_model()
        opponent_model = context.opponent_sim.get_model()

        control_action = self.ai_pilot.Step(
            model.fighterID,
            model._forceSide,
            opponent_model.fighterID,
            opponent_model._forceSide,
            model.get_fdm_data(),
            opponent_model.get_fdm_data(),
        )
        vp = self.ai_pilot.GetVP(model.fighterID, model._forceSide, model.get_fdm_data())
        vp_array, vp_valid = self._vp_to_array(vp)

        action = clip_action(
            [
                control_action.RollCMD,
                control_action.PitchCMD,
                control_action.RudderCMD,
                control_action.Throttle,
            ]
        )
        action[3], throttle_info = self._optimize_turn_throttle(context, action)

        if hasattr(context.sim, "action"):
            context.sim.action[:] = action
        if hasattr(context.sim, "VP"):
            context.sim.VP[:] = vp_array

        return ActionResult(
            action=action,
            source="bt",
            confidence=self.confidence,
            info={
                "vp": vp_array,
                "vp_valid": vp_valid,
                "fighter_id": model.fighterID,
                "force_side": model._forceSide,
                "target_fighter_id": opponent_model.fighterID,
                "target_force_side": opponent_model._forceSide,
                "throttle_control": throttle_info,
            },
        )

    def _compute_remote_action(self, context: ActionContext) -> ActionResult:
        my_plane = context.info["my_plane_data"]
        target_plane = context.info["target_plane_data"]
        fighter_id = int(context.info.get("my_plane_id", 1))
        bt_fighter_id = REMOTE_BT_FIGHTER_ID
        force_side = int(context.info.get("my_force_side", 1))

        self._ensure_remote_behavior_tree(bt_fighter_id, force_side)
        control_action = self.ai_pilot.StepWithPlaneData(my_plane, target_plane)
        vp = self.ai_pilot.GetVPWithPlaneData(my_plane)
        vp_array, vp_valid = self._vp_to_array(vp)
        action = clip_action(
            [
                control_action.RollCMD,
                control_action.PitchCMD,
                control_action.RudderCMD,
                control_action.Throttle,
            ]
        )
        action[3], throttle_info = self._optimize_turn_throttle(context, action)
        return ActionResult(
            action=action,
            source="bt",
            confidence=self.confidence,
            info={
                "vp": vp_array,
                "vp_valid": vp_valid,
                "fighter_id": fighter_id,
                "bt_fighter_id": bt_fighter_id,
                "force_side": force_side,
                "throttle_control": throttle_info,
            },
        )

    def _optimize_turn_throttle(
        self,
        context: ActionContext,
        action: np.ndarray,
    ) -> tuple[float, dict[str, float | bool | str]]:
        if not self.enable_turn_throttle_optimization:
            return float(action[3]), {
                "active": False,
                "reason": "raw_bt",
                "raw_throttle": float(action[3]),
            }
        state = context.ownship_state
        if state is None or len(state) <= StateIndex.ALT:
            return float(action[3]), {
                "active": False,
                "reason": "missing_state",
                "raw_throttle": float(action[3]),
            }

        return optimize_full_turn_throttle(
            raw_throttle=float(action[3]),
            roll_cmd=float(action[0]),
            pitch_cmd=float(action[1]),
            speed_mps=float(state[StateIndex.KCAS]),
            altitude_m=float(state[StateIndex.ALT]),
            speed_is_true_airspeed=context.sim is None,
            config=self.turn_throttle_config,
        )

    def close(self) -> None:
        for fighter_id in list(self._registered_fighter_ids):
            self._remove_behavior_tree(fighter_id)
