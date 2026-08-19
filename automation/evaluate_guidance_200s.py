from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PURE_DLL = Path("C:/Users/shy66/Downloads/aip_final_0815/aip_final_0815/AIP_DCS_GDCC_0815.dll")
PURE_XML = Path("C:/Users/shy66/Downloads/aip_final_0815/aip_final_0815/Rule_DCS_GDCC_0815.xml")
AIP2_DLL = Path("C:/Users/shy66/Downloads/aip2/aip2/AIP_DCS_new.dll")


def full_fight_cases() -> list[dict[str, Any]]:
    cases = []
    for split, seed_base, scenario_root in (
        ("development", 8801, "automation/scenarios/0815_aim_mirror"),
        ("held_out", 8901, "automation/scenarios/0815_aim_mirror_holdout"),
    ):
        for index, (opponent, side) in enumerate(
            (
                ("autopilot", "left"),
                ("autopilot", "right"),
                ("bt_0815", "left"),
                ("bt_0815", "right"),
                ("bt_aip2", "left"),
                ("bt_aip2", "right"),
            )
        ):
            seed = seed_base + index
            cases.append(
                {
                    "case_id": f"{split}_{opponent}_{side}_s{seed}",
                    "split": split,
                    "opponent": opponent,
                    "side": side,
                    "seed": seed,
                    "scenario": ROOT / f"{scenario_root}/lateral_{side}.json",
                }
            )
    return cases


def command_for(
    case: dict,
    controller: str,
    result_path: Path,
    telemetry_path: Path,
    *,
    bundle: Path,
    confidence_threshold: float,
) -> list[str]:
    target_backend = "autopilot" if case["opponent"] == "autopilot" else "bt"
    ownship_backend = "bt" if controller == "pure" else "guidance_selector"
    command = [
        sys.executable,
        str(ROOT / "run_local_dogfight.py"),
        "--ownship-backend",
        ownship_backend,
        "--target-backend",
        target_backend,
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
        str(case["scenario"]),
        "--seed",
        str(case["seed"]),
        "--max-engage-time",
        "200",
        "--episode-step-limit",
        "12000",
        "--result-json",
        str(result_path),
        "--telemetry-jsonl",
        str(telemetry_path),
    ]
    if controller == "bt_default":
        command.extend(["--guidance-fixed-action", "BT_DEFAULT"])
    elif controller == "bc":
        command.extend(
            [
                "--ownship-bundle-dir",
                str(bundle),
                "--guidance-confidence-threshold",
                str(confidence_threshold),
            ]
        )
    if case["opponent"] == "bt_0815":
        command.extend(["--target-bt-dll", str(PURE_DLL)])
    elif case["opponent"] == "bt_aip2":
        command.extend(
            [
                "--target-bt-dll",
                str(AIP2_DLL),
                "--bt-rule-alias",
                "Rule_sei_AIP2_default.xml",
            ]
        )
    return command


def run_fight(
    case: dict,
    controller: str,
    output_root: Path,
    *,
    bundle: Path,
    confidence_threshold: float,
) -> tuple[dict, Path]:
    case_root = output_root / case["case_id"]
    case_root.mkdir(parents=True, exist_ok=True)
    result_path = case_root / f"{controller}.json"
    telemetry_path = case_root / f"{controller}.telemetry.jsonl"
    stdout_path = case_root / f"{controller}.stdout.txt"
    if result_path.exists() and telemetry_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8")), telemetry_path
    command = command_for(
        case,
        controller,
        result_path,
        telemetry_path,
        bundle=bundle,
        confidence_threshold=confidence_threshold,
    )
    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600.0,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0 or not result_path.exists() or not telemetry_path.exists():
        raise RuntimeError(f"200s run failed: {case['case_id']} {controller}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["process_returncode"] = completed.returncode
    result["process_wall_seconds"] = perf_counter() - started
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result, telemetry_path


def telemetry_summary(path: Path) -> dict[str, Any]:
    frames = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if value.get("record_type") == "frame":
                frames.append(value)
    if not frames:
        raise ValueError(f"no simulator frames in {path}")
    altitudes = np.asarray([row["ownship"]["altitude_m"] for row in frames], dtype=np.float64)
    speeds = np.asarray([row["ownship"]["speed_kcas"] for row in frames], dtype=np.float64)
    ranges = np.asarray([row["distance_m"] for row in frames], dtype=np.float64)
    closing = np.asarray([row["closing_rate_m_s"] for row in frames], dtype=np.float64)
    throttle_difference = []
    invalid = 0
    for row in frames:
        hybrid = row.get("hybrid", {}) or {}
        bt_action = hybrid.get("bt_action")
        final_action = hybrid.get("final_action")
        if bt_action is not None and final_action is not None:
            throttle_difference.append(abs(float(final_action[3]) - float(bt_action[3])))
        action = np.asarray(row.get("ownship_action", []), dtype=np.float64)
        invalid += int(action.shape != (4,) or not np.all(np.isfinite(action)))
    return {
        "telemetry_frames": len(frames),
        "range_mean_m": float(np.mean(ranges)),
        "range_median_m": float(np.median(ranges)),
        "range_p95_m": float(np.percentile(ranges, 95)),
        "closing_rate_mean_m_s": float(np.mean(closing)),
        "closing_rate_p95_m_s": float(np.percentile(closing, 95)),
        "min_altitude_m": float(np.min(altitudes)),
        "mean_altitude_m": float(np.mean(altitudes)),
        "low_altitude_duration_s": float(np.sum(altitudes <= 350.0) / 60.0),
        "min_speed_m_s": float(np.min(speeds)),
        "mean_speed_m_s": float(np.mean(speeds)),
        "low_speed_duration_s": float(np.sum(speeds <= 170.0) / 60.0),
        "throttle_difference_max": max(throttle_difference, default=0.0),
        "invalid_or_nonfinite_actions": invalid,
        "telemetry_sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
    }


def compact(case: dict, controller: str, result: dict, telemetry_path: Path) -> dict:
    provider = result.get("ownship_provider_telemetry", {}) or {}
    maneuver = result.get("maneuver_telemetry", {}) or {}
    raw = telemetry_summary(telemetry_path)
    damage_dealt = 1.0 - float(result.get("target_health", 1.0))
    damage_received = 1.0 - float(result.get("ownship_health", 1.0))
    return {
        "case_id": case["case_id"],
        "split": case["split"],
        "opponent": case["opponent"],
        "side": case["side"],
        "seed": case["seed"],
        "controller": controller,
        "outcome": result.get("outcome"),
        "end_condition": result.get("end_condition"),
        "episode_seconds": result.get("episode_seconds"),
        "damage_dealt": damage_dealt,
        "damage_received": damage_received,
        "health_margin": damage_dealt - damage_received,
        "ownship_crash": bool(result.get("ownship_crash")),
        "target_crash": bool(result.get("target_crash")),
        "first_damage_s": maneuver.get("time_to_first_damage_s"),
        "phase1_cone_time_s": maneuver.get("phase1_cone_time_s"),
        "phase2_cone_time_s": maneuver.get("phase2_cone_time_s"),
        "phase3_cone_time_s": maneuver.get("phase3_cone_time_s"),
        "los_mean_deg": maneuver.get("mean_los_deg"),
        "los_median_deg": maneuver.get("median_los_deg"),
        "los_p95_deg": maneuver.get("p95_los_deg"),
        "los_rate_rms_deg_s": maneuver.get("los_rate_rms_deg_s"),
        "gate_active_ratio": provider.get("rear120_activation_active_ratio", 0.0),
        "gate_entries": provider.get("rear120_activation_entries", 0),
        "gate_exits": provider.get("rear120_activation_exits", 0),
        "action_distribution": provider.get("action_distribution", {}),
        "nonzero_intervention_frames": provider.get("nonzero_intervention_frames", 0),
        "requested_guidance_surface_abs_sum": provider.get(
            "requested_guidance_surface_abs_sum", [0.0, 0.0, 0.0]
        ),
        "applied_guidance_surface_abs_sum": provider.get(
            "applied_guidance_surface_abs_sum", [0.0, 0.0, 0.0]
        ),
        "throttle_violation_steps": provider.get("throttle_violation_steps", 0),
        "latency_ms_p50": provider.get("selector_inference_latency_ms_p50", 0.0),
        "latency_ms_p95": provider.get("selector_inference_latency_ms_p95", 0.0),
        "latency_ms_p99": provider.get("selector_inference_latency_ms_p99", 0.0),
        "latency_ms_max": provider.get("selector_inference_latency_ms_max", 0.0),
        "latency_over_166_7ms": provider.get("selector_inference_over_166_7ms", 0),
        "process_returncode": result.get("process_returncode", 0),
        "process_wall_seconds": result.get("process_wall_seconds"),
        **raw,
    }


def aggregate(records: list[dict]) -> dict[str, Any]:
    pure = {row["case_id"]: row for row in records if row["controller"] == "pure"}
    paired = {}
    for controller in ("bt_default", "bc"):
        pairs = []
        for row in records:
            if row["controller"] != controller:
                continue
            baseline = pure[row["case_id"]]
            contaminated = bool(row["target_crash"] or baseline["target_crash"])
            pairs.append(
                {
                    "case_id": row["case_id"],
                    "split": row["split"],
                    "opponent": row["opponent"],
                    "side": row["side"],
                    "damage_delta": float(row["health_margin"] - baseline["health_margin"]),
                    "contaminated": contaminated,
                }
            )
        clean = [pair for pair in pairs if not pair["contaminated"]]
        deltas = np.asarray([pair["damage_delta"] for pair in clean], dtype=np.float64)
        paired[controller] = {
            "pairs": len(pairs),
            "clean_pairs": len(clean),
            "contaminated_pairs": len(pairs) - len(clean),
            "clean_damage_delta_mean": float(np.mean(deltas)) if deltas.size else None,
            "clean_damage_delta_median": float(np.median(deltas)) if deltas.size else None,
            "clean_damage_delta_min": float(np.min(deltas)) if deltas.size else None,
            "clean_damage_delta_max": float(np.max(deltas)) if deltas.size else None,
            "positive_pairs": int(np.sum(deltas > 0.0)),
            "positive_ratio": float(np.mean(deltas > 0.0)) if deltas.size else 0.0,
            "pair_records": pairs,
        }
    bc = [row for row in records if row["controller"] == "bc"]
    operational_ready = bool(
        len(bc) == 12
        and all(row["process_returncode"] == 0 for row in bc)
        and all(not row["ownship_crash"] for row in bc)
        and all(row["min_altitude_m"] > 304.8 for row in bc)
        and all(row["invalid_or_nonfinite_actions"] == 0 for row in bc)
        and all(row["throttle_violation_steps"] == 0 for row in bc)
        and all(row["throttle_difference_max"] == 0.0 for row in bc)
        and all(row["latency_ms_max"] < 166.7 for row in bc)
        and any(row["nonzero_intervention_frames"] > 0 for row in bc)
    )
    performance = paired.get("bc", {})
    promoted = bool(
        operational_ready
        and performance.get("clean_damage_delta_mean", 0.0) > 0.0
        and performance.get("clean_damage_delta_median", 0.0) > 0.0
        and performance.get("positive_ratio", 0.0) >= 2.0 / 3.0
        and performance.get("contaminated_pairs", 1) == 0
    )
    return {
        "status": "PROMOTED_LOCAL" if promoted else (
            "SUBMISSION_READY_HYBRID_CANDIDATE" if operational_ready else "OPERATIONAL_GATE_FAILED"
        ),
        "promotion_status": "PROMOTED_LOCAL" if promoted else "NOT_PROMOTED",
        "records": len(records),
        "configured_200s_runs": len(records),
        "completed_200s_timeout_runs": sum(
            float(row.get("episode_seconds") or 0.0) >= 199.0 for row in records
        ),
        "early_terminal_runs": sum(
            float(row.get("episode_seconds") or 0.0) < 199.0 for row in records
        ),
        "paired": paired,
        "operational_ready": operational_ready,
        "nonzero_intervention_frames": sum(row["nonzero_intervention_frames"] for row in bc),
        "gate_active_ratio_mean": float(np.mean([row["gate_active_ratio"] for row in bc])) if bc else 0.0,
        "ownship_crashes": sum(row["ownship_crash"] for row in bc),
        "target_crashes": sum(row["target_crash"] for row in bc),
        "minimum_altitude_m": min((row["min_altitude_m"] for row in bc), default=None),
        "minimum_speed_m_s": min((row["min_speed_m_s"] for row in bc), default=None),
        "low_altitude_duration_s": sum(row["low_altitude_duration_s"] for row in bc),
        "low_speed_duration_s": sum(row["low_speed_duration_s"] for row in bc),
        "latency_ms_p50_max": max((row["latency_ms_p50"] for row in bc), default=0.0),
        "latency_ms_p95_max": max((row["latency_ms_p95"] for row in bc), default=0.0),
        "latency_ms_p99_max": max((row["latency_ms_p99"] for row in bc), default=0.0),
        "latency_ms_max": max((row["latency_ms_max"] for row in bc), default=0.0),
        "latency_over_166_7ms": sum(row["latency_over_166_7ms"] for row in bc),
        "throttle_violations": sum(row["throttle_violation_steps"] for row in bc),
        "throttle_difference_max": max((row["throttle_difference_max"] for row in bc), default=0.0),
        "invalid_or_nonfinite_actions": sum(row["invalid_or_nonfinite_actions"] for row in bc),
        "process_errors": sum(row["process_returncode"] != 0 for row in bc),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run frozen Pure/BT_DEFAULT/BC 200s matrix")
    parser.add_argument(
        "--development-aggregate",
        type=Path,
        default=ROOT / "artifacts/evaluations/guidance_selector/development_v1_20260819/aggregate.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/evaluations/guidance_selector/full_200s_v1_20260819",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selection = json.loads(args.development_aggregate.read_text(encoding="utf-8"))
    bundle = Path(selection["selected_bundle"]).resolve()
    threshold = float(selection["selected_confidence_threshold"])
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for case in full_fight_cases():
        for controller in ("pure", "bt_default", "bc"):
            result, telemetry_path = run_fight(
                case,
                controller,
                output_root,
                bundle=bundle,
                confidence_threshold=threshold,
            )
            records.append(compact(case, controller, result, telemetry_path))
            print(
                json.dumps(
                    {"case": case["case_id"], "controller": controller, "records": len(records)},
                    sort_keys=True,
                ),
                flush=True,
            )
    summary = aggregate(records)
    summary["selected_bundle"] = str(bundle)
    summary["selected_confidence_threshold"] = threshold
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    summary["records_sha256"] = hashlib.sha256(canonical).hexdigest().upper()
    (output_root / "records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_root / "aggregate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
