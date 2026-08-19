from __future__ import annotations

import argparse
import copy
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
GEOMETRIES = (
    "lateral_left",
    "lateral_right",
    "vertical_high",
    "vertical_low",
    "crossing_left",
    "crossing_right",
)
STAGE_VARIANTS = {"shadow": 1, "micro": 2, "development": 5, "heldout": 3}
STAGE_OPPONENTS = {
    "shadow": ("autopilot",),
    "micro": ("autopilot",),
    "development": ("autopilot", "bt_0815"),
    "heldout": ("autopilot", "bt_0815"),
}


def evaluation_cases(stage: str) -> list[dict[str, Any]]:
    if stage not in STAGE_VARIANTS:
        raise ValueError(f"unsupported evaluation stage: {stage}")
    scenario_group = "0815_aim_mirror_holdout" if stage == "heldout" else "0815_aim_mirror"
    cases = []
    for opponent in STAGE_OPPONENTS[stage]:
        for geometry in GEOMETRIES:
            for variant in range(STAGE_VARIANTS[stage]):
                seed = 34000 + 1000 * list(STAGE_VARIANTS).index(stage) + 100 * variant + len(cases)
                cases.append(
                    {
                        "case_id": f"{stage}_{opponent}_{geometry}_v{variant + 1:02d}",
                        "stage": stage,
                        "opponent": opponent,
                        "geometry": geometry,
                        "variant": variant,
                        "seed": seed,
                        "base_scenario": ROOT / "automation/scenarios" / scenario_group / f"{geometry}.json",
                    }
                )
    return cases


def materialize_scenario(case: dict[str, Any], path: Path) -> None:
    payload = json.loads(case["base_scenario"].read_text(encoding="utf-8"))
    payload = copy.deepcopy(payload)
    env = payload["env_config"]
    centered = case["variant"] - 0.5 * (STAGE_VARIANTS[case["stage"]] - 1)
    env["target"][0] += 20.0 * centered
    env["target"][1] += 4.0 * centered
    env["target"][2] -= 8.0 * centered
    env["ownship"][5] += centered
    env["target"][5] -= centered
    env["ownship"][6] += 2.0 * centered
    env["target"][6] -= 1.0 * centered
    autopilot = env["target_autopilot"]
    autopilot["heading_cmd"] = env["target"][5]
    autopilot["altitude_cmd"] = -env["target"][2]
    autopilot["speed_cmd"] = env["target"][6]
    payload["name"] = case["case_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _command(
    case: dict[str, Any], scenario: Path, result: Path, *, bundle: Path | None
) -> list[str]:
    stage = case["stage"]
    max_seconds = 8 if stage in {"shadow", "micro"} else 30
    command = [
        sys.executable,
        str(ROOT / "run_local_dogfight.py"),
        "--ownship-backend", "bt" if bundle is None else "guidance_selector",
        "--target-backend", "autopilot" if case["opponent"] == "autopilot" else "bt",
        "--ownship-bt-dll", str(PURE_DLL),
        "--bt-rule-xml", str(PURE_XML),
        "--bt-rule-alias", "Rule_DCS_GDCC_0815.xml",
        "--bt-rule-alias-only",
        "--bt-turn-throttle-mode", "raw",
        "--observation-mode", "tactical16",
        "--scenario-file", str(scenario),
        "--seed", str(case["seed"]),
        "--max-engage-time", str(max_seconds),
        "--episode-step-limit", str(max_seconds * 60),
        "--result-json", str(result),
    ]
    if case["opponent"] == "bt_0815":
        command.extend(["--target-bt-dll", str(PURE_DLL)])
    if bundle is not None:
        command.extend(
            [
                "--ownship-bundle-dir", str(bundle),
                "--guidance-confidence-threshold", "0.0",
                "--guidance-angular-offset-deg", "0.25",
                "--guidance-controller-kind", "vp_error_pd_v2",
                "--guidance-minimum-hold-frames", "36",
                "--guidance-maximum-active-frames", "36",
                "--guidance-cooldown-frames", "30",
            ]
        )
        if stage == "shadow":
            command.append("--guidance-shadow-mode")
    return command


def run_case(
    case: dict[str, Any], output: Path, *, bundle: Path | None, timeout_s: float = 120.0
) -> dict[str, Any]:
    controller = "pure" if bundle is None else ("shadow" if case["stage"] == "shadow" else "hybrid")
    case_root = output / "runs" / case["case_id"]
    scenario = case_root / "scenario.json"
    materialize_scenario(case, scenario)
    result_path = case_root / f"{controller}.json"
    stdout_path = case_root / f"{controller}.stdout.txt"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    started = perf_counter()
    completed = subprocess.run(
        _command(case, scenario, result_path, bundle=bundle),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0 or not result_path.exists():
        raise RuntimeError(f"v3 evaluation failed: {case['case_id']} {controller}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["process_wall_seconds"] = perf_counter() - started
    result["process_returncode"] = completed.returncode
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def compact(case: dict[str, Any], controller: str, result: dict[str, Any]) -> dict[str, Any]:
    provider = result.get("ownship_provider_telemetry", {}) or {}
    maneuver = result.get("maneuver_telemetry", {}) or {}
    damage_dealt = 1.0 - float(result.get("target_health", 1.0))
    damage_received = 1.0 - float(result.get("ownship_health", 1.0))
    action_counts = provider.get("action_counts", {})
    predicted_nondefault = sum(
        int(count) for action, count in action_counts.items() if action != "BT_DEFAULT"
    )
    return {
        **{key: case[key] for key in ("case_id", "stage", "opponent", "geometry", "variant", "seed")},
        "controller": controller,
        "outcome": result.get("outcome"),
        "end_condition": result.get("end_condition"),
        "episode_seconds": result.get("episode_seconds"),
        "damage_dealt": damage_dealt,
        "damage_received": damage_received,
        "health_margin": damage_dealt - damage_received,
        "ownship_crash": bool(result.get("ownship_crash")),
        "target_crash": bool(result.get("target_crash")),
        "invalid_actions": int(result.get("invalid_or_nonfinite_actions", 0)),
        "throttle_violations": int(provider.get("throttle_violation_steps", 0)),
        "nonzero_intervention_frames": int(provider.get("nonzero_intervention_frames", 0)),
        "predicted_nondefault_frames": predicted_nondefault,
        "e2e_latency_ms_p50": float(provider.get("e2e_ai_latency_ms_p50", 0.0)),
        "e2e_latency_ms_p95": float(provider.get("e2e_ai_latency_ms_p95", 0.0)),
        "e2e_latency_ms_p99": float(provider.get("e2e_ai_latency_ms_p99", 0.0)),
        "e2e_latency_ms_max": float(provider.get("e2e_ai_latency_ms_max", 0.0)),
        "e2e_over_166_7ms": int(provider.get("e2e_ai_latency_over_166_7ms", 0)),
        "phase1_frames": int(maneuver.get("phase1_steps", 0)),
        "phase2_frames": int(maneuver.get("phase2_steps", 0)),
        "phase3_frames": int(maneuver.get("phase3_steps", 0)),
    }


def bootstrap_ci(values: np.ndarray, *, seed: int = 35001, samples: int = 10000) -> list[float]:
    if not values.size:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(samples, values.size), replace=True), axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(records: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    pure = {row["case_id"]: row for row in records if row["controller"] == "pure"}
    candidate_name = "shadow" if stage == "shadow" else "hybrid"
    pairs = []
    for candidate in records:
        if candidate["controller"] != candidate_name:
            continue
        baseline = pure[candidate["case_id"]]
        contaminated = bool(candidate["target_crash"] or baseline["target_crash"])
        pairs.append(
            {
                "case_id": candidate["case_id"],
                "opponent": candidate["opponent"],
                "geometry": candidate["geometry"],
                "damage_delta": float(candidate["health_margin"] - baseline["health_margin"]),
                "contaminated": contaminated,
                "candidate": candidate,
                "pure": baseline,
            }
        )
    clean = [row for row in pairs if not row["contaminated"]]
    values = np.asarray([row["damage_delta"] for row in clean], dtype=np.float64)
    candidate_rows = [row["candidate"] for row in pairs]
    summary = {
        "schema_version": "state_conditioned_hybrid_v3.evaluation.v1",
        "stage": stage,
        "pairs": len(pairs),
        "clean_pairs": len(clean),
        "contaminated_pairs": len(pairs) - len(clean),
        "clean_damage_delta_mean": float(np.mean(values)) if values.size else 0.0,
        "clean_damage_delta_median": float(np.median(values)) if values.size else 0.0,
        "positive_ratio": float(np.mean(values > 0.0)) if values.size else 0.0,
        "large_regression_ratio": float(np.mean(values <= -0.003)) if values.size else 0.0,
        "bootstrap_mean_95ci": bootstrap_ci(values),
        "predicted_nondefault_frames": sum(row["predicted_nondefault_frames"] for row in candidate_rows),
        "nonzero_intervention_frames": sum(row["nonzero_intervention_frames"] for row in candidate_rows),
        "ownship_crashes": sum(row["ownship_crash"] for row in candidate_rows),
        "target_crashes": sum(row["target_crash"] for row in candidate_rows),
        "invalid_actions": sum(row["invalid_actions"] for row in candidate_rows),
        "throttle_violations": sum(row["throttle_violations"] for row in candidate_rows),
        "e2e_latency_ms_p50_max": max((row["e2e_latency_ms_p50"] for row in candidate_rows), default=0.0),
        "e2e_latency_ms_p95_max": max((row["e2e_latency_ms_p95"] for row in candidate_rows), default=0.0),
        "e2e_latency_ms_p99_max": max((row["e2e_latency_ms_p99"] for row in candidate_rows), default=0.0),
        "e2e_latency_ms_max": max((row["e2e_latency_ms_max"] for row in candidate_rows), default=0.0),
        "e2e_over_166_7ms": sum(row["e2e_over_166_7ms"] for row in candidate_rows),
        "opponents": sorted({row["opponent"] for row in clean}),
        "geometries": sorted({row["geometry"] for row in clean}),
        "pair_records": pairs,
    }
    if stage == "shadow":
        summary["gate_passed"] = bool(
            len(clean) == len(pairs)
            and np.all(values == 0.0)
            and summary["predicted_nondefault_frames"] > 0
            and summary["nonzero_intervention_frames"] == 0
            and summary["throttle_violations"] == 0
        )
    else:
        summary["gate_passed"] = bool(
            values.size > 0
            and summary["clean_damage_delta_mean"] > 0.0
            and summary["clean_damage_delta_median"] > 0.0
            and summary["positive_ratio"] >= 0.60
            and summary["large_regression_ratio"] <= 0.05
            and summary["nonzero_intervention_frames"] > 0
            and summary["ownship_crashes"] == 0
            and summary["invalid_actions"] == 0
            and summary["throttle_violations"] == 0
            and summary["e2e_over_166_7ms"] == 0
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate State-Conditioned Hybrid v3")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_VARIANTS), required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    output = (args.output_root or ROOT / "artifacts/evaluations/state_conditioned_hybrid_v3" / args.stage).resolve()
    bundle = args.bundle.resolve()
    records = []
    cases = evaluation_cases(args.stage)
    for index, case in enumerate(cases, start=1):
        for controller_bundle, controller in ((None, "pure"), (bundle, "shadow" if args.stage == "shadow" else "hybrid")):
            records.append(compact(case, controller, run_case(case, output, bundle=controller_bundle)))
        progress = {"completed_pairs": index, "total_pairs": len(cases)}
        (output / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
        print(json.dumps(progress), flush=True)
    summary = summarize(records, args.stage)
    (output / "records.json").write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    (output / "aggregate.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
