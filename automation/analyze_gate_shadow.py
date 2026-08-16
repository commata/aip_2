from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dogfight.ai.hybrid_action_provider import (
    AimGateConfig,
    AimResidualGate,
    OffensiveGateConfig,
    OffensiveResidualGate,
)


GATE_NAMES = ("aim", "offensive", "combined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure BT telemetry에서 Gate A/B/C를 action 적용 없이 shadow 재생"
    )
    parser.add_argument("--telemetry", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_frames(path: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("record_type") == "frame":
                frames.append(record)
    return frames


def analyze_frames(
    frames: list[dict[str, Any]],
    *,
    aim_config: AimGateConfig | None = None,
    offensive_config: OffensiveGateConfig | None = None,
) -> dict[str, Any]:
    if not frames:
        raise ValueError("shadow gate analysis requires frame telemetry")
    aim_gate = AimResidualGate(aim_config)
    offensive_gate = OffensiveResidualGate(offensive_config)
    active = {name: [] for name in GATE_NAMES}
    geometry = {name: [] for name in GATE_NAMES}
    previous = {name: False for name in GATE_NAMES}
    entries = {name: 0 for name in GATE_NAMES}
    exits = {name: 0 for name in GATE_NAMES}
    durations = {name: [] for name in GATE_NAMES}
    current_duration = {name: 0 for name in GATE_NAMES}

    for frame in frames:
        ownship = _state_from_frame(frame["ownship"], float(frame["sim_time_s"]))
        target = _state_from_frame(frame["target"], float(frame["sim_time_s"]))
        aim = aim_gate.update(
            ownship,
            target,
            sim_time_s=float(frame["sim_time_s"]),
        )
        offensive = offensive_gate.update(ownship, target)
        states = {
            "aim": bool(aim["active"]),
            "offensive": bool(offensive["active"]),
            "combined": bool(aim["active"] and offensive["active"]),
        }
        for name, is_active in states.items():
            active[name].append(is_active)
            geometry[name].append(
                {
                    "distance_m": float(frame["distance_m"]),
                    "ata_deg": float(frame["ata_deg"]),
                    "target_ata_deg": float(frame["target_ata_deg"]),
                    "closing_rate_m_s": float(frame["closing_rate_m_s"]),
                }
            )
            if is_active and not previous[name]:
                entries[name] += 1
            if previous[name] and not is_active:
                exits[name] += 1
                durations[name].append(current_duration[name])
                current_duration[name] = 0
            if is_active:
                current_duration[name] += 1
            previous[name] = is_active
    for name in GATE_NAMES:
        if current_duration[name]:
            durations[name].append(current_duration[name])

    damage_events = _event_indices(
        [float(frame.get("target_damage", 0.0)) > 0.0 for frame in frames]
    )
    sim_step = _median_step_seconds(frames)
    rows: dict[str, Any] = {}
    for name in GATE_NAMES:
        mask = np.asarray(active[name], dtype=bool)
        active_indices = np.flatnonzero(mask)
        actions = np.asarray(
            [frame.get("ownship_action", [0.0] * 4)[:3] for frame in frames],
            dtype=np.float64,
        )
        exposed_actions = actions[mask] if np.any(mask) else np.empty((0, 3))
        active_geometry = [geometry[name][index] for index in active_indices]
        damage_while_active = sum(bool(mask[index]) for index in damage_events)
        pre_damage_hits = sum(
            bool(np.any(mask[max(0, index - round(3.0 / sim_step)) : index + 1]))
            for index in damage_events
        )
        rows[name] = {
            "steps": len(frames),
            "active_steps": int(np.sum(mask)),
            "active_ratio": float(np.mean(mask)),
            "entries": entries[name],
            "exits": exits[name],
            "mean_active_duration_s": (
                statistics.fmean(durations[name]) * sim_step
                if durations[name]
                else 0.0
            ),
            "min_active_duration_s": (
                min(durations[name]) * sim_step if durations[name] else 0.0
            ),
            "damage_events": len(damage_events),
            "damage_events_while_active": damage_while_active,
            "damage_events_with_activation_previous_3s": pre_damage_hits,
            "defensive_active_ratio": (
                statistics.fmean(
                    float(item["target_ata_deg"] < 90.0) for item in active_geometry
                )
                if active_geometry
                else 0.0
            ),
            "headon_active_ratio": (
                statistics.fmean(
                    float(
                        item["ata_deg"] > 150.0
                        and item["target_ata_deg"] > 150.0
                    )
                    for item in active_geometry
                )
                if active_geometry
                else 0.0
            ),
            "active_distance_mean_m": _mean_geometry(active_geometry, "distance_m"),
            "active_ata_mean_deg": _mean_geometry(active_geometry, "ata_deg"),
            "active_target_ata_mean_deg": _mean_geometry(
                active_geometry, "target_ata_deg"
            ),
            "active_closing_rate_mean_m_s": _mean_geometry(
                active_geometry, "closing_rate_m_s"
            ),
            "bt_surface_saturation_ratio_axis": (
                np.mean(np.isclose(np.abs(exposed_actions), 1.0, atol=1e-6), axis=0).tolist()
                if exposed_actions.size
                else [0.0, 0.0, 0.0]
            ),
        }
    return {
        "duration_s": float(frames[-1]["sim_time_s"]),
        "sim_step_s": sim_step,
        "gates": rows,
    }


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("no shadow gate episodes to aggregate")
    result: dict[str, Any] = {"episodes": len(episodes), "gates": {}}
    for name in GATE_NAMES:
        rows = [episode["gates"][name] for episode in episodes]
        total_steps = sum(int(row["steps"]) for row in rows)
        active_steps = sum(int(row["active_steps"]) for row in rows)
        result["gates"][name] = {
            "active_ratio": active_steps / max(1, total_steps),
            "episode_active_ratio_median": statistics.median(
                float(row["active_ratio"]) for row in rows
            ),
            "episode_active_ratio_min": min(float(row["active_ratio"]) for row in rows),
            "episode_active_ratio_max": max(float(row["active_ratio"]) for row in rows),
            "entries": sum(int(row["entries"]) for row in rows),
            "exits": sum(int(row["exits"]) for row in rows),
            "damage_events": sum(int(row["damage_events"]) for row in rows),
            "damage_events_while_active": sum(
                int(row["damage_events_while_active"]) for row in rows
            ),
            "damage_events_with_activation_previous_3s": sum(
                int(row["damage_events_with_activation_previous_3s"]) for row in rows
            ),
            "defensive_active_ratio_mean": statistics.fmean(
                float(row["defensive_active_ratio"]) for row in rows
            ),
            "headon_active_ratio_mean": statistics.fmean(
                float(row["headon_active_ratio"]) for row in rows
            ),
            "bt_surface_saturation_ratio_axis_mean": np.mean(
                [row["bt_surface_saturation_ratio_axis"] for row in rows], axis=0
            ).tolist(),
        }
    return result


def _state_from_frame(side: dict[str, Any], sim_time_s: float) -> np.ndarray:
    roll, pitch, heading = side["attitude_deg"]
    return np.asarray(
        [*side["position_ned_m"], roll, pitch, heading, side["speed_kcas"], sim_time_s],
        dtype=np.float64,
    )


def _event_indices(values: list[bool]) -> list[int]:
    return [
        index
        for index, value in enumerate(values)
        if value and (index == 0 or not values[index - 1])
    ]


def _median_step_seconds(frames: list[dict[str, Any]]) -> float:
    if len(frames) < 2:
        return 1.0 / 60.0
    deltas = [
        float(right["sim_time_s"]) - float(left["sim_time_s"])
        for left, right in zip(frames, frames[1:])
    ]
    return statistics.median(delta for delta in deltas if delta > 0.0)


def _mean_geometry(rows: list[dict[str, float]], key: str) -> float | None:
    return statistics.fmean(row[key] for row in rows) if rows else None


def main() -> int:
    args = parse_args()
    paths = [Path(value).resolve() for value in args.telemetry]
    episodes = []
    for path in paths:
        analysis = analyze_frames(load_frames(path))
        analysis["telemetry"] = str(path)
        episodes.append(analysis)
    payload = {
        "source": "Pure BT simulator-rate telemetry; shadow only, no residual applied",
        "episodes": episodes,
        "aggregate": aggregate(episodes),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

