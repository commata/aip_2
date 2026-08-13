"""Segment simulator-rate telemetry and compare a hybrid run with paired BT."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


def load_frames(path: str | Path) -> list[dict[str, Any]]:
    frames = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("record_type") == "frame":
                frames.append(record)
    if not frames:
        raise ValueError(f"no frame telemetry found: {path}")
    return frames


def classify_phase(frame: dict[str, Any], previous: dict[str, Any] | None) -> str:
    gate = frame.get("hybrid", {}).get("offensive_gate", {}).get("active", False)
    roll = abs(float(frame["ownship"]["attitude_deg"][0]))
    ata = float(frame["ata_deg"])
    distance = float(frame["distance_m"])
    action_roll = float(frame["ownship_action"][0])
    previous_roll = float(previous["ownship_action"][0]) if previous else action_roll
    if action_roll * previous_roll < -0.15:
        return "reversal"
    if gate:
        return "offensive_correction"
    if distance <= 3000.0 and ata <= 45.0:
        return "close_pursuit"
    if roll >= 45.0:
        return "hard_turn"
    if ata >= 90.0:
        return "extension_or_defensive"
    return "transition"


def analyze_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    phases: dict[str, list[dict[str, Any]]] = {}
    segments: list[dict[str, Any]] = []
    current_name = ""
    current: list[dict[str, Any]] = []
    overshoots = 0
    overshoots_gate_active = 0
    gate_entries = 0
    gate_exits = 0
    previous = None
    for frame in frames:
        name = classify_phase(frame, previous)
        phases.setdefault(name, []).append(frame)
        if name != current_name:
            if current:
                segments.append(_segment(current_name, current))
            current_name, current = name, []
        current.append(frame)
        gate = frame.get("hybrid", {}).get("offensive_gate", {})
        gate_entries += int(bool(gate.get("entry", False)))
        gate_exits += int(bool(gate.get("exit", False)))
        if previous is not None:
            is_overshoot = (
                float(previous["ata_deg"]) <= 30.0
                and float(frame["ata_deg"]) >= 60.0
                and float(frame["distance_m"]) <= 2000.0
            )
            overshoots += int(is_overshoot)
            overshoots_gate_active += int(is_overshoot and bool(gate.get("active", False)))
        previous = frame
    if current:
        segments.append(_segment(current_name, current))

    duration = max(float(frames[-1]["sim_time_s"]), 1e-6)
    active = [f for f in frames if f.get("hybrid", {}).get("offensive_gate", {}).get("active", False)]
    relaxed_opportunities = [
        f for f in frames
        if float(f["distance_m"]) <= 3000.0
        and float(f["ata_deg"]) <= 45.0
        and float(f["target_ata_deg"]) >= 80.0
    ]
    missed = [
        f for f in relaxed_opportunities
        if not f.get("hybrid", {}).get("offensive_gate", {}).get("active", False)
    ]
    result = {
        "frames": len(frames),
        "duration_s": duration,
        "gate_active_ratio": len(active) / len(frames),
        "gate_entries": gate_entries,
        "gate_exits": gate_exits,
        "gate_transitions_per_min": (gate_entries + gate_exits) * 60.0 / duration,
        "relaxed_offensive_opportunity_ratio": len(relaxed_opportunities) / len(frames),
        "missed_offensive_opportunity_ratio": len(missed) / max(1, len(relaxed_opportunities)),
        "overshoot_events": overshoots,
        "overshoot_events_while_gate_active": overshoots_gate_active,
        "action_saturation_ratio": _ratio(frames, _saturated),
        "mean_speed_kcas": _mean(frames, lambda f: f["ownship"]["speed_kcas"]),
        "min_speed_kcas": min(float(f["ownship"]["speed_kcas"]) for f in frames),
        "min_altitude_m": min(float(f["ownship"]["altitude_m"]) for f in frames),
        "mean_ata_deg": _mean(frames, lambda f: f["ata_deg"]),
        "min_ata_deg": min(float(f["ata_deg"]) for f in frames),
        "mean_distance_m": _mean(frames, lambda f: f["distance_m"]),
        "min_distance_m": min(float(f["distance_m"]) for f in frames),
        "phase_summary": {name: _phase_summary(rows, len(frames)) for name, rows in phases.items()},
        "segments": segments,
    }
    result["recommendations"] = recommendations(result)
    return result


def _segment(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    first, last = rows[0], rows[-1]
    return {
        "phase": name,
        "start_s": float(first["sim_time_s"]),
        "end_s": float(last["sim_time_s"]),
        "frames": len(rows),
        "ata_change_deg": float(last["ata_deg"]) - float(first["ata_deg"]),
        "speed_change_kcas": float(last["ownship"]["speed_kcas"]) - float(first["ownship"]["speed_kcas"]),
        "altitude_change_m": float(last["ownship"]["altitude_m"]) - float(first["ownship"]["altitude_m"]),
        "min_distance_m": min(float(row["distance_m"]) for row in rows),
    }


def _phase_summary(rows: list[dict[str, Any]], total: int) -> dict[str, Any]:
    corrections = [
        f.get("hybrid", {}).get("applied_rl_correction", [0.0] * 4) for f in rows
    ]
    return {
        "frames": len(rows),
        "ratio": len(rows) / total,
        "mean_ata_deg": _mean(rows, lambda f: f["ata_deg"]),
        "mean_speed_kcas": _mean(rows, lambda f: f["ownship"]["speed_kcas"]),
        "mean_throttle": _mean(rows, lambda f: f["ownship_action"][3]),
        "mean_abs_rl_correction": [
            fmean(abs(float(c[index])) for c in corrections) for index in range(4)
        ],
    }


def _saturated(frame: dict[str, Any]) -> bool:
    hybrid_value = frame.get("hybrid", {}).get("action_saturation")
    if hybrid_value is not None:
        return bool(hybrid_value)
    action = [float(value) for value in frame["ownship_action"]]
    return any(math.isclose(abs(value), 1.0, abs_tol=1e-6) for value in action[:3])


def _mean(rows, getter) -> float:
    return fmean(float(getter(row)) for row in rows)


def _ratio(rows, predicate) -> float:
    return sum(bool(predicate(row)) for row in rows) / max(1, len(rows))


def recommendations(report: dict[str, Any]) -> list[str]:
    notes = []
    occupancy = report["gate_active_ratio"]
    if occupancy < 0.01 and report["relaxed_offensive_opportunity_ratio"] > 0.02:
        notes.append("공격 기회 대비 gate 점유율이 너무 낮습니다. 진입 거리 또는 ATA 조건을 한 번에 하나씩 완화하세요.")
    elif occupancy > 0.35:
        notes.append("Gate 점유 범위가 너무 넓습니다. residual scale을 높이기 전에 target aspect 또는 ATA 진입 조건을 강화하세요.")
    if report["gate_transitions_per_min"] > 8.0:
        notes.append("Gate 전환 빈도가 높습니다. 진입/이탈 hysteresis 간격을 넓히세요.")
    if report["overshoot_events_while_gate_active"]:
        notes.append("Correction 활성 상태에서 overshoot가 발생했습니다. Scale을 낮추거나 gate 이탈 거리를 줄이세요.")
    if report["action_saturation_ratio"] > 0.20:
        notes.append("조종면 포화율이 20%를 초과했습니다. Throttle 동작을 바꾸기 전에 조종면 scale을 낮추세요.")
    if report["missed_offensive_opportunity_ratio"] > 0.75:
        notes.append("완화된 공격 기회의 대부분을 놓쳤습니다. 어떤 gate 진입 조건이 병목인지 확인하세요.")
    return notes or ["이 궤적에서는 뚜렷한 gate 또는 residual 이상이 발견되지 않았습니다."]


def compare(baseline: dict[str, Any], hybrid: dict[str, Any]) -> dict[str, Any]:
    keys = ["mean_ata_deg", "min_ata_deg", "min_distance_m", "mean_speed_kcas", "action_saturation_ratio", "overshoot_events"]
    return {
        "baseline": baseline,
        "hybrid": hybrid,
        "delta_hybrid_minus_bt": {key: hybrid[key] - baseline[key] for key in keys},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry")
    parser.add_argument("--baseline")
    parser.add_argument("--hybrid")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.baseline or args.hybrid:
        if not (args.baseline and args.hybrid):
            parser.error("--baseline and --hybrid must be provided together")
        payload = compare(analyze_frames(load_frames(args.baseline)), analyze_frames(load_frames(args.hybrid)))
    elif args.telemetry:
        payload = analyze_frames(load_frames(args.telemetry))
    else:
        parser.error("provide --telemetry or both --baseline and --hybrid")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
