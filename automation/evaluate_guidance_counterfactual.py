from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dogfight.ai.guidance_selector import GUIDANCE_ACTIONS


PURE_DLL = Path(
    "C:/Users/shy66/Downloads/aip_final_0815/aip_final_0815/AIP_DCS_GDCC_0815.dll"
)
PURE_XML = Path(
    "C:/Users/shy66/Downloads/aip_final_0815/aip_final_0815/Rule_DCS_GDCC_0815.xml"
)
F16_INIT = ROOT / "aircraft" / "f16" / "f16_init.xml"
MEANINGFUL_DAMAGE_DELTA = 0.001
MAXIMUM_GEOMETRY_REGRESSION = -0.003


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build_cases(count: int = 100) -> list[dict[str, Any]]:
    """Return actually distinct, deterministic initial states with no randomization."""
    families = (
        "lateral_left",
        "lateral_right",
        "vertical_high",
        "vertical_low",
        "crossing_left",
        "crossing_right",
    )
    cases = []
    for index in range(count):
        family = families[index % len(families)]
        band = index // len(families)
        distance = 680.0 + 37.0 * (index % 17)
        lateral = 0.0
        altitude_delta = 0.0
        target_heading = 0.0
        if family == "lateral_left":
            lateral = -(35.0 + 11.0 * (band % 13))
        elif family == "lateral_right":
            lateral = 35.0 + 11.0 * (band % 13)
        elif family == "vertical_high":
            altitude_delta = 25.0 + 9.0 * (band % 13)
        elif family == "vertical_low":
            altitude_delta = -(25.0 + 9.0 * (band % 13))
        elif family == "crossing_left":
            lateral = -(55.0 + 8.0 * (band % 13))
            target_heading = -(25.0 + float(band % 4))
        elif family == "crossing_right":
            lateral = 55.0 + 8.0 * (band % 13)
            target_heading = 25.0 + float(band % 4)
        own_altitude = 4700.0 + 35.0 * (index % 9)
        own_speed = 220.0 + 4.0 * (index % 8)
        target_speed = 214.0 + 5.0 * ((index * 3) % 9)
        own_heading = float((index % 5) - 2)
        target_altitude = own_altitude + altitude_delta
        seed = 8601 + index
        cases.append(
            {
                "case_id": f"state_{index + 1:03d}_{family}",
                "seed": seed,
                "family": family,
                "distance_band": "near" if distance < 950.0 else "far",
                "closing_band": "positive" if own_speed > target_speed else "negative",
                "scenario": {
                    "name": f"guidance_counterfactual_{index + 1:03d}_{family}",
                    "env_config": {
                        "ownship": [
                            0.0,
                            0.0,
                            -own_altitude,
                            0.0,
                            0.0,
                            own_heading,
                            own_speed,
                        ],
                        "target": [
                            distance,
                            lateral,
                            -target_altitude,
                            0.0,
                            0.0,
                            target_heading,
                            target_speed,
                        ],
                        "initial_scenario": {
                            "mode": "default",
                            "legacy_use_random_scenario": False,
                        },
                        "ownship_randomization": {"enabled": False},
                        "target_randomization": {"enabled": False},
                        "target_autopilot": {
                            "heading_cmd": target_heading,
                            "altitude_cmd": target_altitude,
                            "speed_cmd": target_speed,
                        },
                    },
                },
            }
        )
    return cases


def run_command(case: dict, action: str, output_root: Path, timeout_s: float) -> dict:
    case_root = output_root / "runs" / case["case_id"]
    case_root.mkdir(parents=True, exist_ok=True)
    scenario_path = case_root / "scenario.json"
    if not scenario_path.exists():
        scenario_path.write_text(
            json.dumps(case["scenario"], indent=2, sort_keys=True), encoding="utf-8"
        )
    result_path = case_root / f"{action}.json"
    stdout_path = case_root / f"{action}.stdout.txt"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["resumed"] = True
        return result
    command = [
        sys.executable,
        str(ROOT / "run_local_dogfight.py"),
        "--ownship-backend",
        "guidance_selector",
        "--target-backend",
        "autopilot",
        "--guidance-fixed-action",
        action,
        "--ownship-bt-dll",
        str(PURE_DLL),
        "--bt-rule-xml",
        str(PURE_XML),
        "--bt-rule-alias",
        "Rule_DCS_GDCC_0815.xml",
        "--bt-rule-alias-only",
        "--bt-turn-throttle-mode",
        "raw",
        "--observation-mode",
        "tactical16",
        "--scenario-file",
        str(scenario_path),
        "--seed",
        str(case["seed"]),
        "--max-engage-time",
        "2",
        "--episode-step-limit",
        "120",
        "--result-json",
        str(result_path),
    ]
    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
        check=False,
    )
    wall_seconds = perf_counter() - started
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0 or not result_path.exists():
        raise RuntimeError(
            f"counterfactual failed case={case['case_id']} action={action} "
            f"returncode={completed.returncode}; see {stdout_path}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["process_returncode"] = completed.returncode
    result["process_wall_seconds"] = wall_seconds
    result["resumed"] = False
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def compact_record(case: dict, action: str, result: dict) -> dict[str, Any]:
    provider = result.get("ownship_provider_telemetry", {}) or {}
    maneuver = result.get("maneuver_telemetry", {}) or {}
    target_health = float(result.get("target_health", 1.0))
    ownship_health = float(result.get("ownship_health", 1.0))
    damage_dealt = 1.0 - target_health
    damage_received = 1.0 - ownship_health
    return {
        "case_id": case["case_id"],
        "seed": case["seed"],
        "family": case["family"],
        "distance_band": case["distance_band"],
        "closing_band": case["closing_band"],
        "action": action,
        "action_id": GUIDANCE_ACTIONS.index(action),
        "outcome": result.get("outcome"),
        "end_condition": result.get("end_condition"),
        "episode_seconds": result.get("episode_seconds"),
        "damage_dealt": damage_dealt,
        "damage_received": damage_received,
        "health_margin": damage_dealt - damage_received,
        "ownship_crash": bool(result.get("ownship_crash")),
        "target_crash": bool(result.get("target_crash")),
        "min_altitude_m": maneuver.get("min_altitude_m"),
        "min_speed_m_s": maneuver.get("min_speed_m_s"),
        "gate_active_frames": provider.get("gate_active_frames", 0),
        "selector_inference_calls": provider.get("selector_inference_calls", 0),
        "nonzero_intervention_frames": provider.get("nonzero_intervention_frames", 0),
        "throttle_violation_steps": provider.get("throttle_violation_steps", 0),
        "latency_ms_max": provider.get("selector_inference_latency_ms_max", 0.0),
        "first_selector_snapshot": provider.get("first_selector_snapshot", {}),
        "process_returncode": result.get("process_returncode", 0),
        "process_wall_seconds": result.get("process_wall_seconds"),
        "resumed": bool(result.get("resumed")),
    }


def summarize(cases: list[dict], records: list[dict]) -> tuple[dict, list[dict]]:
    by_key = {(row["case_id"], row["action"]): row for row in records}
    paired = []
    dataset = []
    for case in cases:
        baseline = by_key[(case["case_id"], "BT_DEFAULT")]
        candidates = []
        for action in GUIDANCE_ACTIONS[1:]:
            row = by_key[(case["case_id"], action)]
            contaminated = bool(row["ownship_crash"] or row["target_crash"])
            delta = float(row["health_margin"] - baseline["health_margin"])
            pair = {
                "case_id": case["case_id"],
                "seed": case["seed"],
                "family": case["family"],
                "action": action,
                "damage_delta": delta,
                "contaminated": contaminated,
                "ownship_crash": row["ownship_crash"],
                "target_crash": row["target_crash"],
                "nonzero_intervention_frames": row["nonzero_intervention_frames"],
            }
            paired.append(pair)
            if not contaminated:
                candidates.append(pair)
        best = max(candidates, key=lambda item: item["damage_delta"], default=None)
        label = (
            best["action"]
            if best is not None and best["damage_delta"] >= MEANINGFUL_DAMAGE_DELTA
            else "BT_DEFAULT"
        )
        snapshot = baseline.get("first_selector_snapshot") or {}
        dataset.append(
            {
                "case_id": case["case_id"],
                "seed": case["seed"],
                "family": case["family"],
                "observation": snapshot.get("observation"),
                "label": label,
                "label_id": GUIDANCE_ACTIONS.index(label),
                "best_clean_damage_delta": best["damage_delta"] if best else None,
                "clean_candidate_count": len(candidates),
            }
        )
    clean = [row for row in paired if not row["contaminated"]]
    deltas = np.asarray([row["damage_delta"] for row in clean], dtype=np.float64)
    family_best = {}
    for family in sorted({case["family"] for case in cases}):
        rows = [row for row in clean if row["family"] == family]
        family_best[family] = max((row["damage_delta"] for row in rows), default=None)
    return {
        "status": "COMPLETED" if len(records) == len(cases) * len(GUIDANCE_ACTIONS) else "INCOMPLETE",
        "unique_states": len(cases),
        "actions_per_state": len(GUIDANCE_ACTIONS),
        "rollouts": len(records),
        "clean_nondefault_pairs": len(clean),
        "contaminated_pairs": len(paired) - len(clean),
        "clean_damage_delta_mean": float(np.mean(deltas)) if deltas.size else None,
        "clean_damage_delta_median": float(np.median(deltas)) if deltas.size else None,
        "clean_damage_delta_min": float(np.min(deltas)) if deltas.size else None,
        "clean_damage_delta_max": float(np.max(deltas)) if deltas.size else None,
        "positive_pairs": int(np.sum(deltas > 0.0)),
        "positive_ratio": float(np.mean(deltas > 0.0)) if deltas.size else None,
        "meaningful_positive_pairs": int(np.sum(deltas >= MEANINGFUL_DAMAGE_DELTA)),
        "large_regressions": int(np.sum(deltas < MAXIMUM_GEOMETRY_REGRESSION)),
        "nondefault_labels": sum(row["label"] != "BT_DEFAULT" for row in dataset),
        "family_best_damage_delta": family_best,
        "ownship_crashes": sum(row["ownship_crash"] for row in records),
        "target_crashes": sum(row["target_crash"] for row in records),
        "process_errors": sum(int(row["process_returncode"] != 0) for row in records),
        "throttle_violations": sum(row["throttle_violation_steps"] for row in records),
        "maximum_latency_ms": max((row["latency_ms_max"] for row in records), default=0.0),
        "horizon_s": 2.0,
        "phase_coverage": {
            "counterfactual": [1],
            "note": "2초 초기상태 counterfactual은 Phase 1만 포함하며 Phase 2/3은 200초 operational matrix에서 검증",
        },
    }, dataset


def write_csv(path: Path, records: list[dict]) -> None:
    fields = [key for key in records[0] if key != "first_selector_snapshot"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fields})


def parse_args():
    parser = argparse.ArgumentParser(description="Run Guidance Selector 100x9 counterfactual pilot")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/evaluations/guidance_selector/counterfactual_v1_20260819",
    )
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.states < 1 or args.states > 100:
        raise ValueError("--states must be between 1 and frozen pilot size 100")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = build_cases(args.states)
    (output_root / "suite.json").write_text(
        json.dumps({"cases": cases}, indent=2, sort_keys=True), encoding="utf-8"
    )
    protected_before = F16_INIT.read_bytes()
    records = []
    started = perf_counter()
    try:
        for case_index, case in enumerate(cases, start=1):
            for action in GUIDANCE_ACTIONS:
                result = run_command(case, action, output_root, args.timeout_s)
                records.append(compact_record(case, action, result))
            progress = {
                "completed_states": case_index,
                "total_states": len(cases),
                "completed_rollouts": len(records),
                "wall_seconds": perf_counter() - started,
            }
            (output_root / "progress.json").write_text(
                json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(json.dumps(progress, sort_keys=True), flush=True)
    finally:
        if F16_INIT.read_bytes() != protected_before:
            F16_INIT.write_bytes(protected_before)
    aggregate, dataset = summarize(cases, records)
    aggregate["wall_seconds"] = perf_counter() - started
    aggregate["pure_dll_sha256"] = sha256(PURE_DLL)
    aggregate["pure_xml_sha256"] = sha256(PURE_XML)
    aggregate["protected_f16_init_restored_exactly"] = F16_INIT.read_bytes() == protected_before
    (output_root / "records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(output_root / "episode_records.csv", records)
    (output_root / "dataset.json").write_text(
        json.dumps({"samples": dataset}, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_root / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
