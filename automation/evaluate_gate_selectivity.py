"""Evaluate rear120 activation selectivity on deterministic synthetic traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from dogfight.ai.hybrid_action_provider import Rear120ActivationGate
from dogfight.sim.state_schema import StateIndex


def state(
    n: float,
    e: float,
    d: float,
    *,
    yaw_deg: float,
    speed_m_s: float,
) -> np.ndarray:
    value = np.zeros(51, dtype=np.float32)
    value[:6] = [n, e, d, 0.0, 0.0, yaw_deg % 360.0]
    value[6] = speed_m_s
    value[StateIndex.KCAS] = speed_m_s
    value[StateIndex.ALT] = -d
    value[StateIndex.HEALTH] = 1.0
    return value


def geometry(
    *,
    target_ata_deg: float,
    ownship_ata_deg: float,
    distance_m: float,
    altitude_m: float,
    speed_m_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    target_yaw = 0.0
    target = state(
        1000.0,
        0.0,
        -altitude_m,
        yaw_deg=target_yaw,
        speed_m_s=speed_m_s,
    )
    bearing_target_to_own = math.radians(target_yaw + target_ata_deg)
    own_n = float(target[0]) + distance_m * math.cos(bearing_target_to_own)
    own_e = float(target[1]) + distance_m * math.sin(bearing_target_to_own)
    bearing_own_to_target = target_yaw + target_ata_deg + 180.0
    ownship = state(
        own_n,
        own_e,
        -altitude_m,
        yaw_deg=bearing_own_to_target + ownship_ata_deg,
        speed_m_s=speed_m_s,
    )
    return ownship, target


def official_cone(sim_time_s: float) -> tuple[float, float]:
    if sim_time_s <= 100.0:
        return 1.0, 3000.0 * 0.3048
    if sim_time_s <= 150.0:
        return 2.0, 3500.0 * 0.3048
    return 3.0, 4000.0 * 0.3048


def _sequence(case: dict[str, Any], key: str, length: int) -> list[float]:
    sequence_key = f"{key}_sequence"
    if sequence_key in case:
        values = [float(value) for value in case[sequence_key]]
        if len(values) != length:
            raise ValueError(f"{case['name']}: {sequence_key} length mismatch")
        return values
    return [float(case[key])] * length


def evaluate_case(case: dict[str, Any], sim_hz: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_sequence = [float(value) for value in case["target_ata_deg_sequence"]]
    length = len(target_sequence)
    if not length:
        raise ValueError(f"{case['name']}: empty target ATA sequence")
    own_ata_sequence = _sequence(case, "ownship_ata_deg", length)
    distance_sequence = _sequence(case, "distance_m", length)
    altitude_sequence = _sequence(case, "altitude_m", length)
    speed_sequence = _sequence(case, "speed_m_s", length)
    sim_time_sequence = _sequence(case, "sim_time_s", length)
    bt_action = np.asarray(case.get("bt_action", [0.2, -0.1, 0.0, 0.8]), dtype=np.float32)
    gate = Rear120ActivationGate()
    rows: list[dict[str, Any]] = []
    cone_entries = 0
    previous_cone = False

    for index in range(length):
        ownship, target = geometry(
            target_ata_deg=target_sequence[index],
            ownship_ata_deg=own_ata_sequence[index],
            distance_m=distance_sequence[index],
            altitude_m=altitude_sequence[index],
            speed_m_s=speed_sequence[index],
        )
        result = gate.update(
            ownship,
            target,
            sim_time_s=sim_time_sequence[index],
            bt_action=bt_action,
        )
        cone_angle, cone_range = official_cone(sim_time_sequence[index])
        in_cone = bool(
            152.4 <= distance_sequence[index] <= cone_range
            and abs(own_ata_sequence[index]) <= cone_angle
        )
        cone_entries += int(in_cone and not previous_cone)
        previous_cone = in_cone
        rows.append(
            {
                "case": case["name"],
                "category": case["category"],
                "step": index,
                "sim_time_s": sim_time_sequence[index],
                "target_ata_deg": result["target_ata_deg"],
                "ownship_ata_deg": result["ata_deg"],
                "distance_m": result["distance_m"],
                "rear120_eligible": result["rear120_eligible"],
                "offensive_eligible": result["offensive_eligible"],
                "pre_aim_eligible": result["pre_aim_eligible"],
                "safety_veto": result["safety_veto"],
                "gate_active": result["active"],
                "gate_entry": result["entry"],
                "gate_exit": result["exit"],
                "in_damage_cone": in_cone,
            }
        )

    telemetry = gate.telemetry()
    expected = case.get("expected_gate_active")
    actual = [bool(row["gate_active"]) for row in rows]
    if expected is not None and [bool(value) for value in expected] != actual:
        raise AssertionError(
            f"{case['name']}: expected gate {expected}, observed {actual}"
        )
    summary = {
        "case": case["name"],
        "category": case["category"],
        "steps": length,
        "gate_active_ratio": telemetry["rear120_activation_active_ratio"],
        "gate_entry_count": telemetry["rear120_activation_entries"],
        "gate_exit_count": telemetry["rear120_activation_exits"],
        "mean_active_duration_steps": telemetry[
            "rear120_activation_mean_active_steps"
        ],
        "max_active_duration_steps": telemetry[
            "rear120_activation_max_active_steps"
        ],
        "mean_active_duration_s": telemetry[
            "rear120_activation_mean_active_steps"
        ] / sim_hz,
        "max_active_duration_s": telemetry[
            "rear120_activation_max_active_steps"
        ] / sim_hz,
        "cone_entry_count": cone_entries,
        "cone_duration_s": sum(int(row["in_damage_cone"]) for row in rows) / sim_hz,
        "time_to_first_damage_s": None,
        "time_to_first_damage_note": "static gate trace has no damage integration",
        "safety_veto_ratio": telemetry["rear120_activation_safety_veto_ratio"],
    }
    return rows, summary


def evaluate_suite(payload: dict[str, Any]) -> dict[str, Any]:
    sim_hz = int(payload.get("sim_hz", 60))
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for case in payload["cases"]:
        rows, summary = evaluate_case(case, sim_hz)
        all_rows.extend(rows)
        summaries.append(summary)
    return {
        "suite": payload.get("name"),
        "purpose": payload.get("purpose"),
        "sim_hz": sim_hz,
        "cases": summaries,
        "aggregate": {
            "cases": len(summaries),
            "steps": len(all_rows),
            "gate_active_ratio": sum(int(row["gate_active"]) for row in all_rows)
            / max(1, len(all_rows)),
            "gate_entry_count": sum(summary["gate_entry_count"] for summary in summaries),
            "gate_exit_count": sum(summary["gate_exit_count"] for summary in summaries),
            "safety_veto_steps": sum(int(row["safety_veto"]) for row in all_rows),
            "cone_steps": sum(int(row["in_damage_cone"]) for row in all_rows),
        },
        "rows": all_rows,
    }


def write_outputs(output: Path, result: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rows = result["rows"]
    with (output / "steps.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.suite.read_text(encoding="utf-8"))
    result = evaluate_suite(payload)
    write_outputs(args.output, result)
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
