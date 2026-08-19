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
CONFIDENCE_THRESHOLDS = (0.55, 0.60, 0.65, 0.70)
BC_SEEDS = (8701, 8702, 8703)
RULE_DISTILLED_SEED = 8799


def development_cases() -> list[dict[str, Any]]:
    cases = []
    combinations = (
        ("autopilot", "left", 8801),
        ("autopilot", "right", 8802),
        ("bt_0815", "left", 8803),
        ("bt_0815", "right", 8804),
        ("bt_aip2", "left", 8805),
        ("bt_aip2", "right", 8806),
    )
    for opponent, side, seed in combinations:
        cases.append(
            {
                "case_id": f"{opponent}_{side}_s{seed}",
                "opponent": opponent,
                "side": side,
                "seed": seed,
                "scenario": ROOT / f"automation/scenarios/0815_aim_mirror/lateral_{side}.json",
            }
        )
    return cases


def _command(case: dict, result_path: Path, *, bundle: Path | None, threshold: float | None):
    target_backend = "autopilot" if case["opponent"] == "autopilot" else "bt"
    command = [
        sys.executable,
        str(ROOT / "run_local_dogfight.py"),
        "--ownship-backend",
        "bt" if bundle is None else "guidance_selector",
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
        "30",
        "--episode-step-limit",
        "1800",
        "--result-json",
        str(result_path),
    ]
    if bundle is not None:
        command.extend(
            [
                "--ownship-bundle-dir",
                str(bundle),
                "--guidance-confidence-threshold",
                str(threshold),
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


def run_case(
    case: dict,
    output_root: Path,
    *,
    bundle: Path | None = None,
    threshold: float | None = None,
) -> dict:
    controller = (
        "pure"
        if bundle is None
        else f"seed_{bundle.name.removeprefix('seed_')}_c{threshold:.2f}"
    )
    root = output_root / case["case_id"]
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / f"{controller}.json"
    stdout_path = root / f"{controller}.stdout.txt"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    command = _command(case, result_path, bundle=bundle, threshold=threshold)
    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180.0,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0 or not result_path.exists():
        raise RuntimeError(f"development run failed: {case['case_id']} {controller}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["process_returncode"] = completed.returncode
    result["process_wall_seconds"] = perf_counter() - started
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def compact(case: dict, controller: str, result: dict) -> dict:
    maneuver = result.get("maneuver_telemetry", {}) or {}
    provider = result.get("ownship_provider_telemetry", {}) or {}
    damage_dealt = 1.0 - float(result.get("target_health", 1.0))
    damage_received = 1.0 - float(result.get("ownship_health", 1.0))
    return {
        "case_id": case["case_id"],
        "opponent": case["opponent"],
        "side": case["side"],
        "seed": case["seed"],
        "controller": controller,
        "outcome": result.get("outcome"),
        "damage_dealt": damage_dealt,
        "damage_received": damage_received,
        "health_margin": damage_dealt - damage_received,
        "ownship_crash": bool(result.get("ownship_crash")),
        "target_crash": bool(result.get("target_crash")),
        "min_altitude_m": maneuver.get("min_altitude_m"),
        "min_speed_m_s": maneuver.get("min_speed_m_s"),
        "gate_active_ratio": provider.get("rear120_activation_active_ratio", 0.0),
        "nonzero_intervention_frames": provider.get("nonzero_intervention_frames", 0),
        "throttle_violation_steps": provider.get("throttle_violation_steps", 0),
        "latency_ms_max": provider.get("selector_inference_latency_ms_max", 0.0),
        "action_distribution": provider.get("action_distribution", {}),
    }


def summarize_candidates(records: list[dict]) -> dict[str, Any]:
    pure = {row["case_id"]: row for row in records if row["controller"] == "pure"}
    controllers = sorted({row["controller"] for row in records if row["controller"] != "pure"})
    summaries = {}
    for controller in controllers:
        pairs = []
        for candidate in records:
            if candidate["controller"] != controller:
                continue
            baseline = pure[candidate["case_id"]]
            contaminated = bool(candidate["target_crash"] or baseline["target_crash"])
            delta = float(candidate["health_margin"] - baseline["health_margin"])
            pairs.append({**candidate, "damage_delta": delta, "contaminated": contaminated})
        clean = [row for row in pairs if not row["contaminated"]]
        deltas = np.asarray([row["damage_delta"] for row in clean], dtype=np.float64)
        summary = {
            "pairs": len(pairs),
            "clean_pairs": len(clean),
            "contaminated_pairs": len(pairs) - len(clean),
            "clean_damage_delta_mean": float(np.mean(deltas)) if deltas.size else None,
            "clean_damage_delta_median": float(np.median(deltas)) if deltas.size else None,
            "clean_damage_delta_min": float(np.min(deltas)) if deltas.size else None,
            "clean_damage_delta_max": float(np.max(deltas)) if deltas.size else None,
            "positive_pairs": int(np.sum(deltas > 0.0)),
            "positive_ratio": float(np.mean(deltas > 0.0)) if deltas.size else 0.0,
            "large_regressions": int(np.sum(deltas < -0.003)),
            "ownship_crashes": sum(row["ownship_crash"] for row in pairs),
            "target_crashes": sum(row["target_crash"] for row in pairs),
            "throttle_violations": sum(row["throttle_violation_steps"] for row in pairs),
            "nonzero_intervention_frames": sum(row["nonzero_intervention_frames"] for row in pairs),
            "latency_ms_max": max((row["latency_ms_max"] for row in pairs), default=0.0),
            "pair_records": pairs,
        }
        is_bc_candidate = not controller.startswith(f"seed_{RULE_DISTILLED_SEED}_")
        summary["candidate_kind"] = (
            "RULE_DISTILLED_SAFE" if not is_bc_candidate else "COUNTERFACTUAL_BC"
        )
        summary["ppo_gate_passed"] = bool(
            is_bc_candidate
            and len(clean) == 6
            and summary["nonzero_intervention_frames"] > 0
            and summary["clean_damage_delta_mean"] > 0.0
            and summary["clean_damage_delta_median"] > 0.0
            and summary["positive_ratio"] >= 2.0 / 3.0
            and summary["large_regressions"] == 0
            and summary["ownship_crashes"] == 0
            and summary["target_crashes"] == 0
            and summary["throttle_violations"] == 0
        )
        summaries[controller] = summary
    eligible = [name for name, value in summaries.items() if value["ppo_gate_passed"]]
    ranked = sorted(
        summaries,
        key=lambda name: (
            summaries[name]["ppo_gate_passed"],
            summaries[name]["nonzero_intervention_frames"] > 0,
            -summaries[name]["large_regressions"],
            summaries[name]["positive_ratio"],
            summaries[name]["clean_damage_delta_mean"] or -999.0,
        ),
        reverse=True,
    )
    return {
        "status": "DEVELOPMENT_COMPLETED",
        "candidate_summaries": summaries,
        "ppo_eligible_candidates": eligible,
        "selected_controller": ranked[0] if ranked else None,
        "selection_kind": "PPO_ELIGIBLE_BC" if eligible else "CONSERVATIVE_BC_ONLY",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BC Guidance candidates on frozen development seeds")
    parser.add_argument(
        "--models-root",
        type=Path,
        default=ROOT / "artifacts/models/guidance_selector_bc_v1",
    )
    parser.add_argument(
        "--include-rule-distilled",
        action="store_true",
        help="also evaluate the frozen seed_8799 rule-distilled fallback bundle",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/evaluations/guidance_selector/development_v1_20260819",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = development_cases()
    records = []
    for case in cases:
        result = run_case(case, output_root)
        records.append(compact(case, "pure", result))
    seeds = BC_SEEDS + ((RULE_DISTILLED_SEED,) if args.include_rule_distilled else ())
    for seed in seeds:
        bundle = args.models_root.resolve() / f"seed_{seed}"
        for threshold in CONFIDENCE_THRESHOLDS:
            controller = f"seed_{seed}_c{threshold:.2f}"
            for case in cases:
                result = run_case(case, output_root, bundle=bundle, threshold=threshold)
                records.append(compact(case, controller, result))
                print(
                    json.dumps(
                        {"controller": controller, "case": case["case_id"], "records": len(records)},
                        sort_keys=True,
                    ),
                    flush=True,
                )
    summary = summarize_candidates(records)
    selected = summary["selected_controller"]
    if selected:
        seed_text, threshold_text = selected.removeprefix("seed_").split("_c")
        summary["selected_seed"] = int(seed_text)
        summary["selected_confidence_threshold"] = float(threshold_text)
        summary["selected_bundle"] = str(
            args.models_root.resolve() / f"seed_{summary['selected_seed']}"
        )
        if summary["selected_seed"] == RULE_DISTILLED_SEED:
            summary["selection_kind"] = "EXPERIMENTAL_SAFE_HYBRID"
    summary["records_sha256"] = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()
    (output_root / "records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_root / "aggregate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
