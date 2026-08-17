"""Pure BT/Hybrid 60 Hz trajectory를 시간 정렬된 정적 small-multiple로 그린다.

Chart contract
- Question: Gate 진입 뒤 조준·Damage·에너지·control authority가 어떻게 갈라지는가?
- Family: time-trend small multiples, one simulator-frame row (약 1,800 points/30 s).
- Encoding: Pure grey dashed, Hybrid blue solid, Gate active gold span; color 외 line style 병행.
- Surface: 재현 가능한 PNG research artifact; 저장 후 실제 image QA가 필요하다.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from automation.analyze_aim_gate_windows import gate_info, load_frames


PURE = "#525866"
HYBRID = "#1769AA"
GATE = "#D8A72D"
AXIS_COLORS = ("#1769AA", "#C46A22", "#7A6A9D")


def series(frames: list[dict[str, Any]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {
        "time": [], "los": [], "los_rate": [], "distance": [], "damage": [],
        "speed": [], "altitude": [], "roll": [], "pitch": [], "yaw": [],
        "requested_roll": [], "requested_pitch": [], "requested_yaw": [],
        "applied_roll": [], "applied_pitch": [], "applied_yaw": [],
    }
    for frame in frames:
        hybrid = frame.get("hybrid", {}) or {}
        authority = hybrid.get("surface_authority", {}) or {}
        requested = authority.get("requested_surface_correction", [0.0] * 3)
        applied = authority.get("applied_surface_correction", [0.0] * 3)
        action = frame["ownship_action"]
        values["time"].append(float(frame["sim_time_s"]))
        values["los"].append(float(frame["ata_deg"]))
        values["los_rate"].append(math.hypot(
            float(frame.get("los_azimuth_rate_deg_s", 0.0)),
            float(frame.get("los_elevation_rate_deg_s", 0.0)),
        ))
        values["distance"].append(float(frame["distance_m"]))
        values["damage"].append(float(frame.get("target_damage_cumulative", 0.0)))
        values["speed"].append(float(frame["ownship"]["speed_kcas"]))
        values["altitude"].append(float(frame["ownship"]["altitude_m"]))
        for index, axis in enumerate(("roll", "pitch", "yaw")):
            values[axis].append(float(action[index]))
            values[f"requested_{axis}"].append(float(requested[index]))
            values[f"applied_{axis}"].append(float(applied[index]))
    return values


def active_spans(frames: list[dict[str, Any]]) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    start: float | None = None
    previous = float(frames[0]["sim_time_s"])
    for frame in frames:
        time_s = float(frame["sim_time_s"])
        active = bool(gate_info(frame).get("active", False))
        if active and start is None:
            start = time_s
        if not active and start is not None:
            spans.append((start, previous))
            start = None
        previous = time_s
    if start is not None:
        spans.append((start, previous))
    return spans


def plot(pure_frames, hybrid_frames, output: Path, title: str) -> None:
    pure, hybrid = series(pure_frames), series(hybrid_frames)
    fig, axes = plt.subplots(4, 2, figsize=(14, 13), sharex=True, constrained_layout=True)
    fig.suptitle(title, fontsize=15, color="#232832")

    panels = (
        ("los", "LOS / ATA (deg)"),
        ("los_rate", "LOS rate magnitude (deg/s)"),
        ("distance", "Distance (m)"),
        ("damage", "Cumulative damage dealt"),
        ("speed", "Ownship speed (KCAS)"),
        ("altitude", "Ownship altitude (m)"),
    )
    for axis, (key, label) in zip(axes.flat[:6], panels):
        axis.plot(pure["time"], pure[key], color=PURE, linestyle="--", linewidth=1.1, label="Pure 0815")
        axis.plot(hybrid["time"], hybrid[key], color=HYBRID, linestyle="-", linewidth=1.2, label="Hybrid")
        axis.set_ylabel(label)
        axis.grid(True, color="#D7DAE0", linewidth=0.6, alpha=0.7)

    control = axes.flat[6]
    for index, name in enumerate(("roll", "pitch", "yaw")):
        color = AXIS_COLORS[index]
        control.plot(pure["time"], pure[name], color=color, linestyle="--", linewidth=0.8, alpha=0.55)
        control.plot(hybrid["time"], hybrid[name], color=color, linestyle="-", linewidth=1.0, label=f"Hybrid {name}")
    control.set_ylabel("Surface command")
    control.set_ylim(-1.08, 1.08)
    control.grid(True, color="#D7DAE0", linewidth=0.6, alpha=0.7)

    residual = axes.flat[7]
    for index, name in enumerate(("roll", "pitch", "yaw")):
        color = AXIS_COLORS[index]
        residual.plot(hybrid["time"], hybrid[f"requested_{name}"], color=color, linestyle=":", linewidth=0.8)
        residual.plot(hybrid["time"], hybrid[f"applied_{name}"], color=color, linestyle="-", linewidth=1.1, label=f"Applied {name}")
    residual.axhline(0.0, color="#808792", linewidth=0.7)
    residual.set_ylabel("Residual correction\n(requested dotted)")
    residual.grid(True, color="#D7DAE0", linewidth=0.6, alpha=0.7)

    spans = active_spans(hybrid_frames)
    for axis in axes.flat:
        for start, end in spans:
            axis.axvspan(start, end, color=GATE, alpha=0.09, linewidth=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors="#424956", labelsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(Patch(facecolor=GATE, alpha=0.22, edgecolor="none"))
    labels.append("Gate active")
    axes.flat[0].legend(handles, labels, loc="upper right", frameon=False, fontsize=8)
    control.legend(loc="upper right", frameon=False, fontsize=7, ncol=3)
    residual.legend(loc="upper right", frameon=False, fontsize=7, ncol=3)
    axes.flat[-2].set_xlabel("Simulation time (s)")
    axes.flat[-1].set_xlabel("Simulation time (s)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure", required=True)
    parser.add_argument("--hybrid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Pure 0815 vs Hybrid trajectory")
    args = parser.parse_args()
    plot(load_frames(args.pure), load_frames(args.hybrid), args.output, args.title)
    print(args.output)


if __name__ == "__main__":
    main()
