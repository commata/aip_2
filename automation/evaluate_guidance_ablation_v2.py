from __future__ import annotations

import argparse
from collections import defaultdict
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

from automation.evaluate_guidance_counterfactual import build_cases
from dogfight.ai.guidance_advantage import GUIDANCE_ADVANTAGE_ACTIONS


PURE_DLL = Path("C:/Users/shy66/Downloads/aip_final_0815/aip_final_0815/AIP_DCS_GDCC_0815.dll")
PURE_XML = Path("C:/Users/shy66/Downloads/aip_final_0815/aip_final_0815/Rule_DCS_GDCC_0815.xml")
MAGNITUDES_DEG = (0.10, 0.25, 0.50)
DURATIONS_FRAMES = (6, 12, 18, 24, 36)
NONDEFAULT_ACTIONS = GUIDANCE_ADVANTAGE_ACTIONS[1:]
MEANINGFUL_DAMAGE_DELTA = 0.001
LARGE_REGRESSION_DELTA = -0.003


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def candidate_id(action: str, magnitude_deg: float, duration_frames: int) -> str:
    return f"{action}__m{magnitude_deg:.2f}__d{duration_frames:02d}"


def candidate_grid() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": candidate_id(action, magnitude, duration),
            "action": action,
            "magnitude_deg": magnitude,
            "duration_frames": duration,
        }
        for action in NONDEFAULT_ACTIONS
        for magnitude in MAGNITUDES_DEG
        for duration in DURATIONS_FRAMES
    ]


def build_near_event_cases(count: int) -> list[dict[str, Any]]:
    """Build unique near-shot states so a two-second Damage horizon is observable."""
    if count % 6:
        raise ValueError("near_event_v2 state count must be a multiple of six")
    families = (
        "lateral_left", "lateral_right", "vertical_high",
        "vertical_low", "crossing_left", "crossing_right",
    )
    cases = []
    for index in range(count):
        family_index = index % len(families)
        family = families[family_index]
        replicate = index // len(families)
        distance = 680.0 + 70.0 * replicate
        lateral = 0.0
        altitude_delta = 0.0
        target_heading = 0.0
        if family == "lateral_left":
            lateral = -(35.0 + 24.0 * replicate)
        elif family == "lateral_right":
            lateral = 35.0 + 24.0 * replicate
        elif family == "vertical_high":
            altitude_delta = 25.0 + 18.0 * replicate
        elif family == "vertical_low":
            altitude_delta = -(25.0 + 18.0 * replicate)
        elif family == "crossing_left":
            lateral = -(55.0 + 20.0 * replicate)
            target_heading = -(25.0 + 4.0 * replicate)
        elif family == "crossing_right":
            lateral = 55.0 + 20.0 * replicate
            target_heading = 25.0 + 4.0 * replicate
        own_altitude = 4700.0 + 90.0 * replicate + 15.0 * family_index
        own_speed = 218.0 + 4.0 * ((family_index + replicate) % 5)
        target_speed = 214.0 + 5.0 * ((2 * family_index + replicate) % 5)
        target_altitude = own_altitude + altitude_delta
        seed = 9701 + index
        cases.append(
            {
                "case_id": f"near_state_{index + 1:03d}_{family}",
                "seed": seed,
                "family": family,
                "distance_band": "near",
                "closing_band": "positive" if own_speed > target_speed else "negative",
                "scenario": {
                    "name": f"guidance_ablation_near_{index + 1:03d}_{family}",
                    "env_config": {
                        "ownship": [0.0, 0.0, -own_altitude, 0.0, 0.0, 0.0, own_speed],
                        "target": [
                            distance, lateral, -target_altitude,
                            0.0, 0.0, target_heading, target_speed,
                        ],
                        "initial_scenario": {"mode": "default", "legacy_use_random_scenario": False},
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


def build_vertical_high_focus_cases(count: int) -> list[dict[str, Any]]:
    """Independent target-high states for the smallest state/action hypothesis test."""
    cases = []
    for index in range(count):
        distance = 650.0 + 50.0 * (index % 5)
        altitude_delta = 20.0 + 15.0 * (index % 6)
        lateral = float(((-1) ** index) * 18.0 * (index % 4))
        own_altitude = 4550.0 + 55.0 * (index % 7)
        own_speed = 216.0 + 4.0 * (index % 6)
        target_speed = 212.0 + 5.0 * ((index * 2) % 6)
        target_heading = float(((-1) ** index) * 3.0 * (index % 4))
        target_altitude = own_altitude + altitude_delta
        seed = 9901 + index
        cases.append(
            {
                "case_id": f"vertical_high_focus_{index + 1:03d}",
                "seed": seed,
                "family": "vertical_high",
                "distance_band": "near",
                "closing_band": "positive" if own_speed > target_speed else "negative",
                "scenario": {
                    "name": f"guidance_vertical_high_focus_{index + 1:03d}",
                    "env_config": {
                        "ownship": [0.0, 0.0, -own_altitude, 0.0, 0.0, 0.0, own_speed],
                        "target": [
                            distance, lateral, -target_altitude,
                            0.0, 0.0, target_heading, target_speed,
                        ],
                        "initial_scenario": {"mode": "default", "legacy_use_random_scenario": False},
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


def _scenario_path(case: dict[str, Any], case_root: Path) -> Path:
    path = case_root / "scenario.json"
    if not path.exists():
        path.write_text(json.dumps(case["scenario"], indent=2, sort_keys=True), encoding="utf-8")
    return path


def _common_command(case: dict[str, Any], scenario_path: Path, result_path: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "run_local_dogfight.py"),
        "--target-backend", "autopilot",
        "--ownship-bt-dll", str(PURE_DLL),
        "--bt-rule-xml", str(PURE_XML),
        "--bt-rule-alias", "Rule_DCS_GDCC_0815.xml",
        "--bt-rule-alias-only",
        "--bt-turn-throttle-mode", "raw",
        "--observation-mode", "tactical16",
        "--scenario-file", str(scenario_path),
        "--seed", str(case["seed"]),
        "--max-engage-time", "2",
        "--episode-step-limit", "120",
        "--result-json", str(result_path),
    ]


def run_one(
    case: dict[str, Any],
    candidate: dict[str, Any],
    output_root: Path,
    timeout_s: float,
) -> dict[str, Any]:
    case_root = output_root / "runs" / case["case_id"]
    case_root.mkdir(parents=True, exist_ok=True)
    scenario_path = _scenario_path(case, case_root)
    name = candidate["candidate_id"]
    result_path = case_root / f"{name}.json"
    stdout_path = case_root / f"{name}.stdout.txt"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["resumed"] = True
        return result
    command = _common_command(case, scenario_path, result_path)
    if name == "PURE_BT":
        command[2:2] = ["--ownship-backend", "bt"]
    else:
        command[2:2] = [
            "--ownship-backend", "guidance_selector",
            "--guidance-fixed-action", candidate["action"],
            "--guidance-controller-kind", "vp_error_pd_v2",
            "--guidance-angular-offset-deg", str(candidate["magnitude_deg"]),
            "--guidance-minimum-hold-frames", str(candidate["duration_frames"]),
            "--guidance-maximum-active-frames", str(candidate["duration_frames"]),
            "--guidance-cooldown-frames", "120",
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
            f"ablation failed case={case['case_id']} candidate={name} "
            f"returncode={completed.returncode}; see {stdout_path}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["process_returncode"] = completed.returncode
    result["process_wall_seconds"] = wall_seconds
    result["resumed"] = False
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def compact(case: dict[str, Any], candidate: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    provider = result.get("ownship_provider_telemetry", {}) or {}
    maneuver = result.get("maneuver_telemetry", {}) or {}
    damage_dealt = 1.0 - float(result.get("target_health", 1.0))
    damage_received = 1.0 - float(result.get("ownship_health", 1.0))
    snapshot = provider.get("first_selector_snapshot", {}) or {}
    observation = snapshot.get("observation") or []
    gate = snapshot.get("gate") or {}
    bt_action = np.asarray(snapshot.get("bt_action", [0.0, 0.0, 0.0, 1.0]), dtype=float)
    action = candidate.get("action", "BT_DEFAULT")
    action_sign = 1.0 if "POS" in action else -1.0 if "NEG" in action else 0.0
    axis_index = 0 if "AZ" in action else 1 if "EL" in action else None
    if axis_index is None:
        directional_headroom = None
    elif action_sign > 0:
        directional_headroom = float(np.clip(1.0 - bt_action[axis_index], 0.0, 2.0))
    else:
        directional_headroom = float(np.clip(bt_action[axis_index] + 1.0, 0.0, 2.0))
    return {
        "case_id": case["case_id"],
        "seed": case["seed"],
        "family": case["family"],
        "distance_band": case["distance_band"],
        "closing_band": case["closing_band"],
        **candidate,
        "damage_dealt": damage_dealt,
        "damage_received": damage_received,
        "health_margin": damage_dealt - damage_received,
        "ownship_crash": bool(result.get("ownship_crash")),
        "target_crash": bool(result.get("target_crash")),
        "outcome": result.get("outcome"),
        "end_condition": result.get("end_condition"),
        "episode_seconds": result.get("episode_seconds"),
        "cone_entries": maneuver.get("damage_cone_entries", 0),
        "cone_time_s": maneuver.get("damage_cone_time_s", 0.0),
        "mean_los_deg": maneuver.get("mean_los_deg"),
        "los_rate_rms_deg_s": maneuver.get("los_rate_rms_deg_s"),
        "min_altitude_m": maneuver.get("min_altitude_m"),
        "min_speed_m_s": maneuver.get("min_speed_m_s"),
        "intervention_frames": provider.get("nonzero_intervention_frames", 0),
        "throttle_violations": provider.get("throttle_violation_steps", 0),
        "latency_ms_max": provider.get("selector_inference_latency_ms_max", 0.0),
        "initial_signed_azimuth": (
            float(observation[16]) if len(observation) == 45 else None
        ),
        "initial_signed_elevation": (
            float(observation[17]) if len(observation) == 45 else None
        ),
        "initial_los_azimuth_rate": (
            float(observation[18]) if len(observation) == 45 else None
        ),
        "initial_los_elevation_rate": (
            float(observation[19]) if len(observation) == 45 else None
        ),
        "initial_range_m": gate.get("distance_m"),
        "initial_directional_headroom": directional_headroom,
        "surface_authority_band": (
            "low" if directional_headroom is not None and directional_headroom < 0.25
            else "medium" if directional_headroom is not None and directional_headroom < 0.75
            else "high" if directional_headroom is not None else "default"
        ),
        "process_returncode": result.get("process_returncode", 0),
        "process_wall_seconds": result.get("process_wall_seconds"),
        "resumed": bool(result.get("resumed")),
    }


def summarize(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baselines = {row["case_id"]: row for row in records if row["candidate_id"] == "PURE_BT"}
    defaults = {row["case_id"]: row for row in records if row["candidate_id"] == "BT_DEFAULT"}
    pairs = []
    for row in records:
        if row["candidate_id"] in {"PURE_BT", "BT_DEFAULT"}:
            continue
        baseline = baselines[row["case_id"]]
        contaminated = bool(
            baseline["ownship_crash"] or baseline["target_crash"]
            or row["ownship_crash"] or row["target_crash"]
        )
        pairs.append(
            {
                **row,
                "damage_delta": row["health_margin"] - baseline["health_margin"],
                "cone_entry_delta": row["cone_entries"] - baseline["cone_entries"],
                "cone_time_delta_s": row["cone_time_s"] - baseline["cone_time_s"],
                "los_improvement_deg": baseline["mean_los_deg"] - row["mean_los_deg"],
                "los_rate_improvement_deg_s": (
                    baseline["los_rate_rms_deg_s"] - row["los_rate_rms_deg_s"]
                ),
                "contaminated": contaminated,
            }
        )
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        by_candidate[row["candidate_id"]].append(row)
    candidates = []
    for name, rows in sorted(by_candidate.items()):
        clean = [row for row in rows if not row["contaminated"]]
        damage = np.asarray([row["damage_delta"] for row in clean], dtype=float)
        cone = np.asarray([row["cone_time_delta_s"] for row in clean], dtype=float)
        los = np.asarray([row["los_improvement_deg"] for row in clean], dtype=float)
        families = sorted({row["family"] for row in clean})
        family_positive = int(sum(
            np.mean([row["damage_delta"] for row in clean if row["family"] == family]) > 0.0
            for family in families
        ))
        positive_by_family = {
            family: max(
                0.0,
                float(np.sum([row["damage_delta"] for row in clean if row["family"] == family])),
            )
            for family in families
        }
        positive_total = float(sum(positive_by_family.values()))
        dominant_positive_contribution = (
            max(positive_by_family.values(), default=0.0) / positive_total
            if positive_total > 0.0
            else 1.0
        )
        candidate = {
            "candidate_id": name,
            "action": rows[0]["action"],
            "magnitude_deg": rows[0]["magnitude_deg"],
            "duration_frames": rows[0]["duration_frames"],
            "pairs": len(rows),
            "clean_pairs": len(clean),
            "contaminated_pairs": len(rows) - len(clean),
            "damage_delta_mean": float(np.mean(damage)) if damage.size else None,
            "damage_delta_median": float(np.median(damage)) if damage.size else None,
            "damage_delta_min": float(np.min(damage)) if damage.size else None,
            "damage_delta_p10": float(np.percentile(damage, 10)) if damage.size else None,
            "positive_ratio": float(np.mean(damage > 0.0)) if damage.size else None,
            "meaningful_positive_ratio": (
                float(np.mean(damage >= MEANINGFUL_DAMAGE_DELTA)) if damage.size else None
            ),
            "large_regressions": int(np.sum(damage < LARGE_REGRESSION_DELTA)),
            "cone_time_delta_mean_s": float(np.mean(cone)) if cone.size else None,
            "los_improvement_mean_deg": float(np.mean(los)) if los.size else None,
            "family_positive_count": family_positive,
            "family_count": len(families),
            "positive_damage_by_family": positive_by_family,
            "dominant_positive_contribution": dominant_positive_contribution,
            "ownship_crashes": sum(row["ownship_crash"] for row in rows),
            "target_crashes": sum(row["target_crash"] for row in rows),
            "intervention_frames": sum(row["intervention_frames"] for row in rows),
            "throttle_violations": sum(row["throttle_violations"] for row in rows),
        }
        candidate["causal_signal_candidate"] = bool(
            damage.size
            and candidate["damage_delta_mean"] > 0.0
            and candidate["damage_delta_median"] > 0.0
            and candidate["positive_ratio"] >= 0.60
            and family_positive >= max(2, int(np.ceil(0.60 * max(1, len(families)))))
            and dominant_positive_contribution <= 0.50
            and candidate["large_regressions"] == 0
            and candidate["ownship_crashes"] == 0
            and candidate["throttle_violations"] == 0
        )
        candidates.append(candidate)
    default_parity = []
    for case_id, pure in baselines.items():
        default = defaults[case_id]
        default_parity.append(
            {
                "case_id": case_id,
                "health_margin_delta": default["health_margin"] - pure["health_margin"],
                "damage_dealt_delta": default["damage_dealt"] - pure["damage_dealt"],
                "damage_received_delta": default["damage_received"] - pure["damage_received"],
                "intervention_frames": default["intervention_frames"],
                "throttle_violations": default["throttle_violations"],
            }
        )
    clean_pairs = [row for row in pairs if not row["contaminated"]]
    return {
        "schema_version": "guidance_ablation_v2.v1",
        "status": "COMPLETED",
        "cases": len(baselines),
        "families": sorted({row["family"] for row in records}),
        "candidate_configs": len(candidates),
        "rollouts": len(records),
        "paired_nondefault": len(pairs),
        "clean_pairs": len(clean_pairs),
        "contaminated_pairs": len(pairs) - len(clean_pairs),
        "default_parity": default_parity,
        "default_parity_passed": all(
            row["health_margin_delta"] == 0.0
            and row["damage_dealt_delta"] == 0.0
            and row["damage_received_delta"] == 0.0
            and row["intervention_frames"] == 0
            and row["throttle_violations"] == 0
            for row in default_parity
        ),
        "signal_candidates": [
            row["candidate_id"] for row in candidates if row["causal_signal_candidate"]
        ],
        "candidates": candidates,
        "safety": {
            "ownship_crashes": sum(row["ownship_crash"] for row in records),
            "target_crashes": sum(row["target_crash"] for row in records),
            "throttle_violations": sum(row["throttle_violations"] for row in records),
            "process_errors": sum(row["process_returncode"] != 0 for row in records),
            "latency_ms_max": max((row["latency_ms_max"] for row in records), default=0.0),
        },
    }, pairs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guidance Controller v2 causal action ablation")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/evaluations/guidance_advantage_v2/ablation_pilot_20260819",
    )
    parser.add_argument("--states", type=int, default=6)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--suite-mode",
        choices=("original_v1", "near_event_v2", "vertical_high_focus_v2"),
        default="original_v1",
    )
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help="Run only the named frozen candidate; repeat for a small revalidation set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.states < 6 or args.states > 100:
        raise ValueError("--states must be between 6 and 100")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.suite_mode == "near_event_v2":
        cases = build_near_event_cases(args.states)
    elif args.suite_mode == "vertical_high_focus_v2":
        cases = build_vertical_high_focus_cases(args.states)
    else:
        cases = build_cases(args.states)
    grid = candidate_grid()
    if args.candidate_id:
        requested = set(args.candidate_id)
        known = {row["candidate_id"] for row in grid}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown --candidate-id values: {unknown}")
        grid = [row for row in grid if row["candidate_id"] in requested]
    (output_root / "suite.json").write_text(
        json.dumps(
            {"suite_mode": args.suite_mode, "cases": cases, "candidates": grid},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    records = []
    started = perf_counter()
    baseline_candidates = [
        {"candidate_id": "PURE_BT", "action": "BT_DEFAULT", "magnitude_deg": 0.0, "duration_frames": 0},
        {"candidate_id": "BT_DEFAULT", "action": "BT_DEFAULT", "magnitude_deg": 0.10, "duration_frames": 6},
    ]
    for case_index, case in enumerate(cases, start=1):
        for candidate in (*baseline_candidates, *grid):
            result = run_one(case, candidate, output_root, args.timeout_s)
            records.append(compact(case, candidate, result))
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
    aggregate, pairs = summarize(records)
    aggregate["wall_seconds"] = perf_counter() - started
    aggregate["pure_dll_sha256"] = sha256(PURE_DLL)
    aggregate["pure_xml_sha256"] = sha256(PURE_XML)
    (output_root / "records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_root / "pairs.json").write_text(
        json.dumps(pairs, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(output_root / "paired_results.csv", pairs)
    (output_root / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
