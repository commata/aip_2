from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Any

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action
from dogfight.ai.tactical_modes import apply_tactical_mode, tactical_mode_name
from dogfight.sim.state_schema import StateIndex


PREFIX_OBSERVABLE_CONTRACT = "prefix_replay_observable_v4"


@dataclass(frozen=True)
class PrefixReplayIntervention:
    start_frame: int
    hold_frames: int
    tactical_mode: str = "BT_DEFAULT"

    def validate(self) -> None:
        if self.start_frame < 0:
            raise ValueError("prefix replay start frame must be non-negative")
        if self.hold_frames < 0:
            raise ValueError("prefix replay hold frames must be non-negative")
        tactical_mode_name(self.tactical_mode)
        if self.tactical_mode != "BT_DEFAULT" and self.hold_frames <= 0:
            raise ValueError("nondefault Tactical intervention requires positive hold")


def _observable_state(state: Any) -> np.ndarray:
    vector = np.asarray(state, dtype=np.float64)
    if vector.ndim != 1 or vector.size <= StateIndex.ALT:
        raise ValueError("prefix replay requires state through altitude")
    # These fields all originate from actual server packets or deterministic
    # packet conversion. Health/Damage and hidden engine/surface truth are absent.
    indexes = (*range(9), StateIndex.KCAS, StateIndex.ALT)
    visible = vector[list(indexes)]
    if not np.all(np.isfinite(visible)):
        raise ValueError("prefix replay observable state contains nonfinite values")
    return visible


def observable_telemetry_hash(
    frame: int,
    ownship_state: Any,
    target_state: Any,
    bt_action: Any,
    bt_vp: Any,
) -> str:
    own = _observable_state(ownship_state).astype("<f8", copy=False)
    target = _observable_state(target_state).astype("<f8", copy=False)
    action = np.asarray(bt_action, dtype="<f4")
    vp = np.asarray(bt_vp, dtype="<f8")
    if action.shape != (4,) or vp.shape != (3,):
        raise ValueError("prefix replay hash requires action[4] and VP[3]")
    if not np.all(np.isfinite(action)) or not np.all(np.isfinite(vp)):
        raise ValueError("prefix replay hash inputs must be finite")
    digest = hashlib.sha256()
    digest.update(PREFIX_OBSERVABLE_CONTRACT.encode("ascii"))
    digest.update(struct.pack("<q", int(frame)))
    digest.update(own.tobytes())
    digest.update(target.tobytes())
    digest.update(action.tobytes())
    digest.update(vp.tobytes())
    return digest.hexdigest().upper()


def build_prefix_snapshot(
    frame: int,
    context: ActionContext,
    bt_action: Any,
    bt_vp: Any,
) -> dict[str, Any]:
    own = _observable_state(context.ownship_state)
    target = _observable_state(context.target_state)
    action = np.asarray(bt_action, dtype=np.float32)
    vp = np.asarray(bt_vp, dtype=np.float64)
    return {
        "contract": PREFIX_OBSERVABLE_CONTRACT,
        "frame": int(frame),
        "sim_time_s": float(context.info.get("sim_time_s", frame / 60.0)),
        "ownship_server_observable": own.tolist(),
        "target_server_observable": target.tolist(),
        "bt_action": action.tolist(),
        "bt_vp": vp.tolist(),
        "observable_telemetry_hash": observable_telemetry_hash(
            frame, context.ownship_state, context.target_state, action, vp
        ),
    }


def compare_prefix_snapshots(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    atol: float = 0.0,
) -> dict[str, Any]:
    if int(left.get("frame", -1)) != int(right.get("frame", -2)):
        return {"match": False, "reason": "frame_mismatch"}
    fields = (
        "ownship_server_observable",
        "target_server_observable",
        "bt_action",
        "bt_vp",
    )
    maximum_error = 0.0
    for field in fields:
        lhs = np.asarray(left.get(field), dtype=np.float64)
        rhs = np.asarray(right.get(field), dtype=np.float64)
        if lhs.shape != rhs.shape:
            return {"match": False, "reason": f"{field}_shape_mismatch"}
        error = float(np.max(np.abs(lhs - rhs))) if lhs.size else 0.0
        maximum_error = max(maximum_error, error)
        if not np.allclose(lhs, rhs, rtol=0.0, atol=atol):
            return {
                "match": False,
                "reason": f"{field}_value_mismatch",
                "maximum_absolute_error": maximum_error,
            }
    hash_match = (
        left.get("observable_telemetry_hash")
        == right.get("observable_telemetry_hash")
    )
    return {
        "match": bool(hash_match if atol == 0.0 else True),
        "reason": "exact_hash_match" if hash_match else "within_tolerance",
        "maximum_absolute_error": maximum_error,
        "hash_match": hash_match,
    }


class PrefixReplayTacticalActionProvider(ActionProvider):
    """Run exact BT at 60Hz and intervene only inside one frozen frame window."""

    def __init__(
        self,
        bt_provider: ActionProvider,
        intervention: PrefixReplayIntervention,
    ) -> None:
        intervention.validate()
        self.bt_provider = bt_provider
        self.intervention = intervention
        self.reset(None)

    def reset(self, context: ActionContext | None = None) -> None:
        self.bt_provider.reset(context)
        self._frame = 0
        self._prefix_snapshot: dict[str, Any] = {}
        self._initial_snapshot: dict[str, Any] = {}
        self._last_frame: dict[str, Any] = {}
        self._intervention_frames = 0
        self._fallback_frames = 0
        self._throttle_violations = 0

    def compute_action(self, context: ActionContext) -> ActionResult:
        frame = self._frame
        self._frame += 1
        bt_result = self.bt_provider.compute_action(context)
        bt_action = clip_action(bt_result.action)
        bt_vp = np.asarray(bt_result.info.get("vp"), dtype=np.float64)
        if bt_vp.shape != (3,) or not np.all(np.isfinite(bt_vp)):
            bt_vp = np.asarray(context.ownship_state, dtype=np.float64)[:3].copy()
        snapshot = build_prefix_snapshot(frame, context, bt_action, bt_vp)
        if frame == 0:
            self._initial_snapshot = snapshot
        if frame == self.intervention.start_frame:
            self._prefix_snapshot = snapshot

        active = (
            self.intervention.tactical_mode != "BT_DEFAULT"
            and self.intervention.start_frame
            <= frame
            < self.intervention.start_frame + self.intervention.hold_frames
        )
        if not active:
            frame_info = {
                "mode": "prefix_replay",
                "frame": frame,
                "intervention_active": False,
                "selected_tactical_mode": "BT_DEFAULT",
                "prefix_snapshot": snapshot
                if frame == self.intervention.start_frame
                else None,
                "bt_action": bt_action.tolist(),
                "final_action": bt_action.tolist(),
                "bt_vp": bt_vp.tolist(),
                "throttle_bt_only": True,
                "exact_bt_command": True,
            }
            self._last_frame = frame_info
            return ActionResult(bt_action.copy(), "bt_prefix_replay", 1.0, frame_info)

        final, tactical = apply_tactical_mode(
            self.intervention.tactical_mode,
            bt_action,
            bt_vp,
            context.ownship_state,
            context.target_state,
        )
        self._intervention_frames += 1
        self._fallback_frames += int(bool(tactical.get("fallback")))
        if final[3] != bt_action[3]:
            self._throttle_violations += 1
            final = bt_action.copy()
            tactical = {
                **tactical,
                "fallback": True,
                "fallback_reason": "provider_throttle_violation",
                "final_action": final.tolist(),
            }
        frame_info = {
            "mode": "prefix_replay",
            "frame": frame,
            "intervention_active": True,
            "selected_tactical_mode": self.intervention.tactical_mode,
            "prefix_snapshot": snapshot
            if frame == self.intervention.start_frame
            else None,
            "bt_action": bt_action.tolist(),
            "final_action": final.tolist(),
            "bt_vp": bt_vp.tolist(),
            "tactical": tactical,
            "throttle_bt_only": True,
            "exact_bt_command": bool(np.array_equal(final, bt_action)),
        }
        self._last_frame = frame_info
        return ActionResult(final, "tactical_prefix_replay", 1.0, frame_info)

    def telemetry(self) -> dict[str, Any]:
        return {
            "contract": PREFIX_OBSERVABLE_CONTRACT,
            "frames": self._frame,
            "intervention": {
                "start_frame": self.intervention.start_frame,
                "hold_frames": self.intervention.hold_frames,
                "tactical_mode": self.intervention.tactical_mode,
            },
            "prefix_snapshot": dict(self._prefix_snapshot),
            "initial_snapshot": dict(self._initial_snapshot),
            "intervention_frames": self._intervention_frames,
            "fallback_frames": self._fallback_frames,
            "throttle_violations": self._throttle_violations,
            "last_frame": dict(self._last_frame),
        }

    def close(self) -> None:
        self.bt_provider.close()
