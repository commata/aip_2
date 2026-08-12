from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dogfight.ai.hybrid_action_provider import _unsigned_ata_deg
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

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def start_episode(self, *, seed: int | None = None) -> None:
        self._episode += 1
        self._frame = 0
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
    ) -> None:
        if not self.enabled:
            return
        own = np.asarray(ownship_state, dtype=np.float64)
        target = np.asarray(target_state, dtype=np.float64)
        distance = float(np.linalg.norm(target[:3] - own[:3]))
        ata = _unsigned_ata_deg(own, target)
        target_ata = _unsigned_ata_deg(target, own)
        payload = {
            "record_type": "frame",
            "episode": self._episode,
            "frame": self._frame,
            "sim_time_s": self._frame / max(1, self.sim_hz),
            "distance_m": distance,
            "ata_deg": ata,
            "target_ata_deg": target_ata,
            "aa_deg": abs(180.0 - target_ata),
            "ownship": self._state_payload(own),
            "target": self._state_payload(target),
            "ownship_action": np.asarray(ownship_action, dtype=np.float32).tolist(),
            "target_action": np.asarray(target_action, dtype=np.float32).tolist(),
            "hybrid": _json_safe(dict(ownship_action_info or {})),
        }
        self._write(payload)
        self._frame += 1

    @staticmethod
    def _state_payload(state: np.ndarray) -> dict[str, Any]:
        return {
            "position_ned_m": state[StateIndex.N : StateIndex.D + 1].tolist(),
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
        return {
            "enabled": self.enabled,
            "path": str(self.path) if self.path else "",
            "episode": self._episode,
            "frames": self._frame,
            "records": self._records,
            "sim_hz": self.sim_hz,
        }

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
