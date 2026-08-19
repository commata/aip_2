from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dogfight.ai.hybrid_action_provider import _unsigned_ata_deg
from dogfight.envs.observation import aim_residual_geometry
from dogfight.sim.state_schema import StateIndex


class ManeuverTelemetryLogger:
    """Stream simulator-frame geometry, controls, and hybrid decisions to JSONL."""

    def __init__(self, path: str | Path | None, *, sim_hz: int, flush_every: int = 60):
        self.path = Path(path) if path else None
        self.sim_hz = int(sim_hz)
        self.flush_every = max(1, int(flush_every))
        self._file = None
        self._episode = -1
        self._frame = 0
        self._records = 0
        self._reset_summary()

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def start_episode(self, *, seed: int | None = None) -> None:
        self._episode += 1
        self._frame = 0
        self._reset_summary()
        if not self.enabled:
            return
        if self._file is None:
            assert self.path is not None
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w", encoding="utf-8")
        self._write({"record_type": "episode_start", "episode": self._episode, "seed": seed})

    def record(
        self,
        ownship_state,
        target_state,
        ownship_action,
        target_action,
        ownship_action_info: dict[str, Any] | None = None,
        *,
        ownship_damage: float = 0.0,
        target_damage: float = 0.0,
        in_wez: bool = False,
    ) -> None:
        own = np.asarray(ownship_state, dtype=np.float64)
        target = np.asarray(target_state, dtype=np.float64)
        distance = float(np.linalg.norm(target[:3] - own[:3]))
        ata = _unsigned_ata_deg(own, target)
        target_ata = _unsigned_ata_deg(target, own)
        aim = aim_residual_geometry(own, target)
        self._update_summary(
            own,
            target,
            aim,
            in_wez=bool(in_wez),
            target_damage=float(target_damage),
        )
        action_info = dict(ownship_action_info or {})
        self._update_surface_summary(ownship_action, action_info)
        if not self.enabled:
            self._frame += 1
            return
        payload = {
            "record_type": "frame",
            "episode": self._episode,
            "frame": self._frame,
            "sim_time_s": self._frame / max(1, self.sim_hz),
            "distance_m": distance,
            "ata_deg": ata,
            "target_ata_deg": target_ata,
            "aa_deg": abs(180.0 - target_ata),
            "aim_azimuth_deg": aim["aim_azimuth_deg"],
            "aim_elevation_deg": aim["aim_elevation_deg"],
            "los_azimuth_rate_deg_s": aim["los_azimuth_rate_deg_s"],
            "los_elevation_rate_deg_s": aim["los_elevation_rate_deg_s"],
            "closing_rate_m_s": aim["closing_rate_m_s"],
            "ownship_damage": float(ownship_damage),
            "target_damage": float(target_damage),
            "in_wez": bool(in_wez),
            "ownship": self._state_payload(own),
            "target": self._state_payload(target),
            "ownship_action": np.asarray(ownship_action, dtype=np.float32).tolist(),
            "target_action": np.asarray(target_action, dtype=np.float32).tolist(),
            "hybrid": _json_safe(action_info),
        }
        self._write(payload)
        self._frame += 1

    def _reset_summary(self) -> None:
        self._ata_values: list[float] = []
        self._target_ata_values: list[float] = []
        self._los_rate_values: list[float] = []
        self._distance_values: list[float] = []
        self._speed_values: list[float] = []
        self._altitude_values: list[float] = []
        self._cone_entries = 0
        self._cone_steps = 0
        self._phase_cone_steps = {1: 0, 2: 0, 3: 0}
        self._previous_in_wez = False
        self._time_to_first_wez_s: float | None = None
        self._time_to_first_damage_s: float | None = None
        self._final_surface_saturated_steps = np.zeros(3, dtype=np.int64)
        self._bt_surface_saturated_steps = np.zeros(3, dtype=np.int64)
        self._final_positive_headroom_sum = np.zeros(3, dtype=np.float64)
        self._final_negative_headroom_sum = np.zeros(3, dtype=np.float64)
        self._bt_positive_headroom_sum = np.zeros(3, dtype=np.float64)
        self._bt_negative_headroom_sum = np.zeros(3, dtype=np.float64)

    def _update_summary(
        self,
        own: np.ndarray,
        target: np.ndarray,
        aim: dict[str, float],
        *,
        in_wez: bool,
        target_damage: float,
    ) -> None:
        sim_time_s = self._frame / max(1, self.sim_hz)
        ata = float(aim["ata_deg"])
        self._ata_values.append(ata)
        self._target_ata_values.append(float(aim["target_ata_deg"]))
        self._los_rate_values.append(
            float(
                np.hypot(
                    aim["los_azimuth_rate_deg_s"],
                    aim["los_elevation_rate_deg_s"],
                )
            )
        )
        self._distance_values.append(float(aim["distance_m"]))
        self._speed_values.append(float(own[StateIndex.KCAS]))
        self._altitude_values.append(float(own[StateIndex.ALT]))
        if in_wez:
            self._cone_steps += 1
            phase = 1 if sim_time_s <= 100.0 else 2 if sim_time_s <= 150.0 else 3
            self._phase_cone_steps[phase] += 1
            if not self._previous_in_wez:
                self._cone_entries += 1
            if self._time_to_first_wez_s is None:
                self._time_to_first_wez_s = sim_time_s
        if target_damage > 0.0 and self._time_to_first_damage_s is None:
            self._time_to_first_damage_s = sim_time_s
        self._previous_in_wez = in_wez

    def _update_surface_summary(self, final_action, action_info: dict[str, Any]) -> None:
        final = np.asarray(final_action, dtype=np.float64)[:3]
        bt_value = action_info.get("bt_action")
        bt = (
            np.asarray(bt_value, dtype=np.float64)[:3]
            if isinstance(bt_value, (list, tuple, np.ndarray)) and len(bt_value) >= 3
            else final
        )
        self._final_surface_saturated_steps += np.isclose(
            np.abs(final), 1.0, atol=1e-6
        )
        self._bt_surface_saturated_steps += np.isclose(
            np.abs(bt), 1.0, atol=1e-6
        )
        self._final_positive_headroom_sum += np.clip(1.0 - final, 0.0, 2.0)
        self._final_negative_headroom_sum += np.clip(final + 1.0, 0.0, 2.0)
        self._bt_positive_headroom_sum += np.clip(1.0 - bt, 0.0, 2.0)
        self._bt_negative_headroom_sum += np.clip(bt + 1.0, 0.0, 2.0)

    @staticmethod
    def _state_payload(state: np.ndarray) -> dict[str, Any]:
        return {
            "position_ned_m": state[StateIndex.N : StateIndex.D + 1].tolist(),
            "body_velocity_m_s": state[6:9].tolist(),
            "altitude_m": float(state[StateIndex.ALT]),
            "attitude_deg": state[StateIndex.ROLL : StateIndex.YAW + 1].tolist(),
            "speed_kcas": float(state[StateIndex.KCAS]),
            "health": float(state[StateIndex.HEALTH]),
        }

    def _write(self, payload: dict[str, Any]) -> None:
        assert self._file is not None
        self._file.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
        self._records += 1
        if self._records % self.flush_every == 0:
            self._file.flush()

    def summary(self) -> dict[str, Any]:
        ata = np.asarray(self._ata_values, dtype=np.float64)
        los_rate = np.asarray(self._los_rate_values, dtype=np.float64)
        speed = np.asarray(self._speed_values, dtype=np.float64)
        altitude = np.asarray(self._altitude_values, dtype=np.float64)
        target_ata = np.asarray(self._target_ata_values, dtype=np.float64)
        result = {
            "enabled": self.enabled,
            "path": str(self.path) if self.path else "",
            "episode": self._episode,
            "frames": self._frame,
            "records": self._records,
            "sim_hz": self.sim_hz,
            "mean_los_deg": float(np.mean(ata)) if ata.size else 0.0,
            "median_los_deg": float(np.median(ata)) if ata.size else 0.0,
            "p95_los_deg": float(np.percentile(ata, 95)) if ata.size else 0.0,
            "min_los_deg": float(np.min(ata)) if ata.size else 0.0,
            "los_rate_rms_deg_s": (
                float(np.sqrt(np.mean(np.square(los_rate)))) if los_rate.size else 0.0
            ),
            "mean_ata_deg": float(np.mean(ata)) if ata.size else 0.0,
            "min_ata_deg": float(np.min(ata)) if ata.size else 0.0,
            "mean_target_ata_deg": (
                float(np.mean(target_ata)) if target_ata.size else 0.0
            ),
            "damage_cone_entries": self._cone_entries,
            "damage_cone_time_s": self._cone_steps / max(1, self.sim_hz),
            "phase1_cone_time_s": self._phase_cone_steps[1] / max(1, self.sim_hz),
            "phase2_cone_time_s": self._phase_cone_steps[2] / max(1, self.sim_hz),
            "phase3_cone_time_s": self._phase_cone_steps[3] / max(1, self.sim_hz),
            "time_to_first_wez_s": self._time_to_first_wez_s,
            "time_to_first_damage_s": self._time_to_first_damage_s,
            "mean_speed_m_s": float(np.mean(speed)) if speed.size else 0.0,
            "min_speed_m_s": float(np.min(speed)) if speed.size else 0.0,
            "min_altitude_m": float(np.min(altitude)) if altitude.size else 0.0,
        }
        frames = max(1, self._frame)
        for index, axis in enumerate(("roll", "pitch", "yaw")):
            result[f"final_{axis}_saturation_ratio"] = float(
                self._final_surface_saturated_steps[index] / frames
            )
            result[f"bt_{axis}_saturation_ratio"] = float(
                self._bt_surface_saturated_steps[index] / frames
            )
            result[f"final_{axis}_positive_headroom_mean"] = float(
                self._final_positive_headroom_sum[index] / frames
            )
            result[f"final_{axis}_negative_headroom_mean"] = float(
                self._final_negative_headroom_sum[index] / frames
            )
            result[f"bt_{axis}_positive_headroom_mean"] = float(
                self._bt_positive_headroom_sum[index] / frames
            )
            result[f"bt_{axis}_negative_headroom_mean"] = float(
                self._bt_negative_headroom_sum[index] / frames
            )
        return result

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
