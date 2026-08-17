"""Compare paired Pure BT and Hybrid telemetry at simulator-frame resolution.

This is a diagnostic tool.  It never rewrites source telemetry and it does not
promote a checkpoint.  Outputs are written to a caller-selected, unique path.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


AXES = ("roll", "pitch", "yaw", "throttle")


@dataclass(frozen=True)
class DivergenceThresholds:
    command: float = 1.0e-7
    state_position_m: float = 1.0e-6
    state_attitude_deg: float = 1.0e-6
    state_speed_kcas: float = 1.0e-6
    los_deg: float = 1.0e-6
    damage: float = 1.0e-10
    correction: float = 1.0e-9


def load_frames(path: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") != "frame":
                continue
            if "frame" not in record:
                raise ValueError(f"{path}:{line_number}: frame record has no frame index")
            frames.append(record)
    if not frames:
        raise ValueError(f"no frame records in {path}")
    return frames


def pair_frames(
    pure_frames: Iterable[dict[str, Any]],
    hybrid_frames: Iterable[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pure_by_frame = {int(row["frame"]): row for row in pure_frames}
    hybrid_by_frame = {int(row["frame"]): row for row in hybrid_frames}
    common = sorted(pure_by_frame.keys() & hybrid_by_frame.keys())
    if not common:
        raise ValueError("Pure and Hybrid telemetry have no common frame indices")
    return [(pure_by_frame[index], hybrid_by_frame[index]) for index in common]


def _array(value: Any, *, length: int, default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(length, default, dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"expected length-{length} vector, got {array.shape}")
    return array


def _hybrid_info(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("hybrid")
    return value if isinstance(value, dict) else {}


def _los_deg(row: dict[str, Any]) -> float:
    azimuth = float(row.get("aim_azimuth_deg", 0.0))
    elevation = float(row.get("aim_elevation_deg", 0.0))
    return float(np.hypot(azimuth, elevation))


def _los_rate_deg_s(row: dict[str, Any]) -> float:
    azimuth = float(row.get("los_azimuth_rate_deg_s", 0.0))
    elevation = float(row.get("los_elevation_rate_deg_s", 0.0))
    return float(np.hypot(azimuth, elevation))


def _first_frame(rows: list[dict[str, Any]], key: str) -> int | None:
    for row in rows:
        if bool(row[key]):
            return int(row["frame"])
    return None


def _state_vector(row: dict[str, Any], side: str, key: str, length: int) -> np.ndarray:
    state = row.get(side, {}) or {}
    return _array(state.get(key), length=length)


def build_comparison(
    pure_frames: Iterable[dict[str, Any]],
    hybrid_frames: Iterable[dict[str, Any]],
    *,
    thresholds: DivergenceThresholds | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    thresholds = thresholds or DivergenceThresholds()
    pure_frames = list(pure_frames)
    hybrid_frames = list(hybrid_frames)
    paired = pair_frames(pure_frames, hybrid_frames)
    rows: list[dict[str, Any]] = []
    active_durations: list[int] = []
    active_start: int | None = None

    for pure, hybrid in paired:
        info = _hybrid_info(hybrid)
        gate = info.get("gate", {}) or {}
        pure_action = _array(pure.get("ownship_action"), length=4)
        hybrid_final = _array(
            info.get("final_action", hybrid.get("ownship_action")), length=4
        )
        hybrid_bt = _array(info.get("bt_action", hybrid_final), length=4)
        raw_residual = _array(info.get("raw_residual_action"), length=4)
        correction = _array(info.get("applied_rl_correction"), length=4)
        final_delta = hybrid_final - pure_action
        bt_delta = hybrid_bt - pure_action

        own_position_delta = _state_vector(
            hybrid, "ownship", "position_ned_m", 3
        ) - _state_vector(pure, "ownship", "position_ned_m", 3)
        own_attitude_delta = _state_vector(
            hybrid, "ownship", "attitude_deg", 3
        ) - _state_vector(pure, "ownship", "attitude_deg", 3)
        own_speed_delta = float(
            (hybrid.get("ownship", {}) or {}).get("speed_kcas", 0.0)
        ) - float((pure.get("ownship", {}) or {}).get("speed_kcas", 0.0))
        target_position_delta = _state_vector(
            hybrid, "target", "position_ned_m", 3
        ) - _state_vector(pure, "target", "position_ned_m", 3)
        target_attitude_delta = _state_vector(
            hybrid, "target", "attitude_deg", 3
        ) - _state_vector(pure, "target", "attitude_deg", 3)

        pure_los = _los_deg(pure)
        hybrid_los = _los_deg(hybrid)
        pure_damage_step = float(pure.get("target_damage", 0.0))
        hybrid_damage_step = float(hybrid.get("target_damage", 0.0))
        # Episode summaries define dealt Damage from target health.  The
        # target_damage field is a per-step value and must not be accumulated
        # again because simulator-rate telemetry can observe action-repeat
        # bookkeeping at a different grain.
        pure_damage_cumulative = 1.0 - float(
            (pure.get("target", {}) or {}).get("health", 1.0)
        )
        hybrid_damage_cumulative = 1.0 - float(
            (hybrid.get("target", {}) or {}).get("health", 1.0)
        )
        gate_active = bool(gate.get("active", False))
        frame = int(hybrid["frame"])
        if gate_active and active_start is None:
            active_start = frame
        elif not gate_active and active_start is not None:
            active_durations.append(frame - active_start)
            active_start = None

        row: dict[str, Any] = {
            "frame": frame,
            "sim_time_s": float(hybrid.get("sim_time_s", 0.0)),
            "gate_active": gate_active,
            "gate_entry": bool(gate.get("entry", False)),
            "gate_exit": bool(gate.get("exit", False)),
            "command_diverged": bool(
                np.max(np.abs(final_delta[:3])) > thresholds.command
            ),
            "bt_command_diverged": bool(
                np.max(np.abs(bt_delta[:3])) > thresholds.command
            ),
            "residual_nonzero": bool(
                np.max(np.abs(correction[:3])) > thresholds.correction
            ),
            "state_diverged": bool(
                np.linalg.norm(own_position_delta) > thresholds.state_position_m
                or np.max(np.abs(own_attitude_delta)) > thresholds.state_attitude_deg
                or abs(own_speed_delta) > thresholds.state_speed_kcas
            ),
            "los_diverged": abs(hybrid_los - pure_los) > thresholds.los_deg,
            "cone_diverged": bool(hybrid.get("in_wez", False))
            != bool(pure.get("in_wez", False)),
            "damage_diverged": abs(
                hybrid_damage_cumulative - pure_damage_cumulative
            ) > thresholds.damage,
            "pure_los_deg": pure_los,
            "hybrid_los_deg": hybrid_los,
            "los_delta_deg": hybrid_los - pure_los,
            "pure_los_rate_deg_s": _los_rate_deg_s(pure),
            "hybrid_los_rate_deg_s": _los_rate_deg_s(hybrid),
            "pure_in_cone": bool(pure.get("in_wez", False)),
            "hybrid_in_cone": bool(hybrid.get("in_wez", False)),
            "pure_target_damage_step": pure_damage_step,
            "hybrid_target_damage_step": hybrid_damage_step,
            "pure_target_damage_cumulative": pure_damage_cumulative,
            "hybrid_target_damage_cumulative": hybrid_damage_cumulative,
            "damage_delta": hybrid_damage_cumulative - pure_damage_cumulative,
            "command_delta_norm": float(np.linalg.norm(final_delta[:3])),
            "bt_command_delta_norm": float(np.linalg.norm(bt_delta[:3])),
            "own_position_delta_norm_m": float(np.linalg.norm(own_position_delta)),
            "own_attitude_delta_norm_deg": float(np.linalg.norm(own_attitude_delta)),
            "own_speed_delta_kcas": own_speed_delta,
            "target_position_delta_norm_m": float(np.linalg.norm(target_position_delta)),
            "target_attitude_delta_norm_deg": float(np.linalg.norm(target_attitude_delta)),
            "distance_m_pure": float(pure.get("distance_m", 0.0)),
            "distance_m_hybrid": float(hybrid.get("distance_m", 0.0)),
            "closing_rate_m_s_pure": float(pure.get("closing_rate_m_s", 0.0)),
            "closing_rate_m_s_hybrid": float(hybrid.get("closing_rate_m_s", 0.0)),
            "ata_deg_pure": float(pure.get("ata_deg", 0.0)),
            "ata_deg_hybrid": float(hybrid.get("ata_deg", 0.0)),
            "target_ata_deg_pure": float(pure.get("target_ata_deg", 0.0)),
            "target_ata_deg_hybrid": float(hybrid.get("target_ata_deg", 0.0)),
            "action_saturation": bool(info.get("action_saturation", False)),
        }
        for index, axis in enumerate(AXES):
            row[f"pure_action_{axis}"] = float(pure_action[index])
            row[f"hybrid_bt_{axis}"] = float(hybrid_bt[index])
            row[f"raw_residual_{axis}"] = float(raw_residual[index])
            row[f"correction_{axis}"] = float(correction[index])
            row[f"hybrid_final_{axis}"] = float(hybrid_final[index])
            row[f"final_delta_{axis}"] = float(final_delta[index])
            row[f"bt_delta_{axis}"] = float(bt_delta[index])
        for side in ("ownship", "target"):
            for trajectory, source in (("pure", pure), ("hybrid", hybrid)):
                state = source.get(side, {}) or {}
                for index, axis in enumerate(("n", "e", "d")):
                    row[f"{trajectory}_{side}_position_{axis}_m"] = float(
                        _array(state.get("position_ned_m"), length=3)[index]
                    )
                for index, axis in enumerate(("roll", "pitch", "yaw")):
                    row[f"{trajectory}_{side}_attitude_{axis}_deg"] = float(
                        _array(state.get("attitude_deg"), length=3)[index]
                    )
                row[f"{trajectory}_{side}_altitude_m"] = float(
                    state.get("altitude_m", 0.0)
                )
                row[f"{trajectory}_{side}_speed_kcas"] = float(
                    state.get("speed_kcas", 0.0)
                )
        rows.append(row)

    if active_start is not None:
        active_durations.append(int(rows[-1]["frame"]) - active_start + 1)

    active_rows = [row for row in rows if row["gate_active"]]
    correction_stats = {}
    for axis in AXES:
        values = np.asarray(
            [abs(float(row[f"correction_{axis}"])) for row in active_rows],
            dtype=np.float64,
        )
        correction_stats[axis] = {
            "abs_mean": float(np.mean(values)) if values.size else 0.0,
            "abs_max": float(np.max(values)) if values.size else 0.0,
            "nonzero_ratio": float(np.mean(values > thresholds.correction))
            if values.size
            else 0.0,
        }

    first_residual = _first_frame(rows, "residual_nonzero")
    first_command = _first_frame(rows, "command_diverged")
    first_bt_command = _first_frame(rows, "bt_command_diverged")
    first_state = _first_frame(rows, "state_diverged")
    first_los = _first_frame(rows, "los_diverged")
    first_cone = _first_frame(rows, "cone_diverged")
    first_damage = _first_frame(rows, "damage_diverged")
    causal_order = [
        first_residual,
        first_command,
        first_state,
        first_bt_command,
        first_los,
        first_cone,
        first_damage,
    ]
    finite_order = [value for value in causal_order if value is not None]
    pure_final_damage = 1.0 - float(
        (pure_frames[-1].get("target", {}) or {}).get("health", 1.0)
    )
    hybrid_final_damage = 1.0 - float(
        (hybrid_frames[-1].get("target", {}) or {}).get("health", 1.0)
    )
    summary = {
        "paired_frames": len(rows),
        "pure_episode_frames": len(pure_frames),
        "hybrid_episode_frames": len(hybrid_frames),
        "first_residual_frame": first_residual,
        "first_command_divergence_frame": first_command,
        "first_state_divergence_frame": first_state,
        "first_bt_command_divergence_frame": first_bt_command,
        "first_LOS_divergence_frame": first_los,
        "first_cone_divergence_frame": first_cone,
        "first_damage_divergence_frame": first_damage,
        "causal_chain_monotonic_for_observed_events": finite_order == sorted(finite_order),
        "gate": {
            "active_ratio": len(active_rows) / max(1, len(rows)),
            "entry_count": sum(int(row["gate_entry"]) for row in rows),
            "exit_count": sum(int(row["gate_exit"]) for row in rows),
            "mean_active_duration_frames": float(np.mean(active_durations))
            if active_durations
            else 0.0,
            "max_active_duration_frames": max(active_durations, default=0),
        },
        "correction": correction_stats,
        "saturation_ratio": float(
            np.mean([bool(row["action_saturation"]) for row in rows])
        ),
        "final": {
            "pure_damage": pure_final_damage,
            "hybrid_damage": hybrid_final_damage,
            "damage_delta": hybrid_final_damage - pure_final_damage,
            "los_delta_deg": float(rows[-1]["los_delta_deg"]),
            "position_delta_norm_m": float(rows[-1]["own_position_delta_norm_m"]),
            "attitude_delta_norm_deg": float(
                rows[-1]["own_attitude_delta_norm_deg"]
            ),
        },
        "thresholds": thresholds.__dict__,
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, Any]], label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_s = np.asarray([row["sim_time_s"] for row in rows])
    figure, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    for axis in AXES[:3]:
        axes[0].plot(time_s, [row[f"correction_{axis}"] for row in rows], label=axis)
    axes[0].set_ylabel("RL correction")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(time_s, [row["command_delta_norm"] for row in rows], label="final cmd")
    axes[1].plot(time_s, [row["bt_command_delta_norm"] for row in rows], label="BT cmd")
    axes[1].set_yscale("symlog", linthresh=1e-8)
    axes[1].set_ylabel("command delta norm")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.25)

    axes[2].plot(time_s, [row["own_position_delta_norm_m"] for row in rows], label="position m")
    axes[2].plot(time_s, [row["own_attitude_delta_norm_deg"] for row in rows], label="attitude deg")
    axes[2].set_yscale("symlog", linthresh=1e-8)
    axes[2].set_ylabel("state divergence")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.25)

    axes[3].plot(time_s, [row["los_delta_deg"] for row in rows], label="LOS delta deg")
    axes[3].plot(time_s, [row["damage_delta"] for row in rows], label="Damage delta")
    axes[3].step(time_s, [int(row["cone_diverged"]) for row in rows], where="post", label="cone differs")
    axes[3].set_ylabel("effect")
    axes[3].set_xlabel("sim time (s)")
    axes[3].legend(loc="upper left")
    axes[3].grid(alpha=0.25)
    figure.suptitle(label)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="Pure BT vs Hybrid residual divergence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = build_comparison(load_frames(args.pure), load_frames(args.hybrid))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "frame_comparison.csv", rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_plot(output / "divergence.png", rows, args.label)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
