"""Run deterministic paired BT/offensive-hybrid evaluations and rank safe scales."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from automation.analyze_maneuvers import analyze_frames, load_frames


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SCALES = (0.10, 0.125, 0.15, 0.175, 0.20)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    bundle = Path(args.bundle).resolve()
    metadata_path = bundle / "metadata.json"
    weights_path = bundle / "policy_weights.pkl.gz"
    required = [metadata_path, weights_path, Path(args.ownship_bt_dll), Path(args.target_bt_dll), Path(args.bt_rule_xml)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"preflight missing files: {missing}")
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = metadata_payload.get("metadata", metadata_payload)
    obs_mode = metadata.get("obs_mode") or metadata_payload.get("algorithm_config", {}).get("env_config", {}).get("observation_mode")
    if obs_mode != args.observation_mode:
        raise ValueError(f"bundle observation mismatch: {obs_mode!r} != {args.observation_mode!r}")
    invalid_scales = [scale for scale in args.scales if scale not in ALLOWED_SCALES]
    if invalid_scales:
        raise ValueError(f"scales must be selected from {ALLOWED_SCALES}: {invalid_scales}")
    return {
        "bundle": str(bundle),
        "bundle_obs_mode": obs_mode,
        "bundle_metadata_sha256": sha256(metadata_path),
        "bundle_weights_sha256": sha256(weights_path),
        "ownship_bt_dll": str(Path(args.ownship_bt_dll).resolve()),
        "target_bt_dll": str(Path(args.target_bt_dll).resolve()),
        "bt_rule_xml": str(Path(args.bt_rule_xml).resolve()),
        "bt_rule_sha256": sha256(Path(args.bt_rule_xml)),
    }


def _scenario_name(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("name", path.stem))


def _controller_specs(scales: list[float]) -> list[tuple[str, float | None]]:
    return [("bt", None)] + [(f"hybrid_{scale:g}", scale) for scale in scales]


def run_match(
    args: argparse.Namespace,
    output: Path,
    *,
    scenario: Path,
    seed: int,
    controller: str,
    scale: float | None,
) -> dict[str, Any]:
    run_id = f"{_scenario_name(scenario)}_seed{seed}_{controller}"
    result_path = output / "summaries" / f"{run_id}.json"
    telemetry_path = output / "telemetry" / f"{run_id}.jsonl"
    analysis_path = output / "analysis" / f"{run_id}.json"
    stdout_path = output / "raw" / f"{run_id}.stdout.txt"
    stderr_path = output / "raw" / f"{run_id}.stderr.txt"
    if args.resume and result_path.is_file() and analysis_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        return build_record(run_id, scenario, seed, controller, scale, result, analysis, 0, False, True)

    cmd = [
        sys.executable,
        str(ROOT / "run_local_dogfight.py"),
        "--ownship-backend", "bt" if controller == "bt" else "hybrid",
        "--target-backend", "bt",
        "--ownship-bt-dll", str(Path(args.ownship_bt_dll).resolve()),
        "--target-bt-dll", str(Path(args.target_bt_dll).resolve()),
        "--bt-rule-xml", str(Path(args.bt_rule_xml).resolve()),
        "--observation-mode", args.observation_mode,
        "--scenario-file", str(scenario),
        "--seed", str(seed),
        "--max-engage-time", str(args.max_engage_time),
        "--episode-step-limit", str(args.episode_step_limit),
        "--result-json", str(result_path),
        "--telemetry-jsonl", str(telemetry_path),
    ]
    if controller != "bt":
        cmd += [
            "--ownship-bundle-dir", str(Path(args.bundle).resolve()),
            "--hybrid-mode", "offensive_residual",
            "--residual-scale", str(scale),
            "--rl-action-repeat", str(args.rl_action_repeat),
            "--offensive-min-range-m", str(args.offensive_min_range_m),
            "--offensive-enter-range-m", str(args.offensive_enter_range_m),
            "--offensive-exit-range-m", str(args.offensive_exit_range_m),
            "--offensive-enter-ata-deg", str(args.offensive_enter_ata_deg),
            "--offensive-exit-ata-deg", str(args.offensive_exit_ata_deg),
            "--offensive-enter-target-ata-deg", str(args.offensive_enter_target_ata_deg),
            "--offensive-exit-target-ata-deg", str(args.offensive_exit_target_ata_deg),
        ]
    started = time.monotonic()
    timed_out = False
    try:
        process = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=args.timeout_seconds, check=False)
        returncode = process.returncode
        stdout, stderr = process.stdout, process.stderr
    except subprocess.TimeoutExpired as error:
        returncode, timed_out = 124, True
        stdout, stderr = error.stdout or "", error.stderr or ""
    stdout_path.write_text(str(stdout), encoding="utf-8")
    stderr_path.write_text(str(stderr), encoding="utf-8")
    duration = time.monotonic() - started
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    analysis = {}
    if telemetry_path.is_file():
        try:
            analysis = analyze_frames(load_frames(telemetry_path))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            analysis = {"analysis_error": str(error)}
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    return build_record(run_id, scenario, seed, controller, scale, result, analysis, returncode, timed_out, False, duration)


def build_record(run_id, scenario, seed, controller, scale, result, analysis, returncode, timed_out, resumed, duration=0.0):
    outcome = "timeout" if timed_out else ("crash" if returncode else str(result.get("outcome", "unknown")))
    own_health = _finite(result.get("ownship_health"))
    target_health = _finite(result.get("target_health"))
    provider = result.get("ownship_provider_telemetry", {})
    return {
        "run_id": run_id,
        "scenario": _scenario_name(Path(scenario)),
        "scenario_file": str(scenario),
        "seed": seed,
        "controller": controller,
        "scale": scale,
        "outcome": outcome,
        "returncode": returncode,
        "timed_out": timed_out,
        "resumed": resumed,
        "wall_seconds": round(duration, 3),
        "total_reward": _finite(result.get("total_reward")),
        "ownship_health": own_health,
        "target_health": target_health,
        "health_margin": own_health - target_health if own_health is not None and target_health is not None else None,
        "episode_seconds": _finite(result.get("episode_seconds")),
        "gate_active_ratio": _finite(provider.get("offensive_gate_active_ratio")) or 0.0,
        "rl_inference_calls": _finite(provider.get("rl_inference_calls")) or 0.0,
        "action_saturation_ratio": _finite(analysis.get("action_saturation_ratio")),
        "mean_ata_deg": _finite(analysis.get("mean_ata_deg")),
        "min_ata_deg": _finite(analysis.get("min_ata_deg")),
        "min_distance_m": _finite(analysis.get("min_distance_m")),
        "min_altitude_m": _finite(analysis.get("min_altitude_m")),
        "overshoot_events": _finite(analysis.get("overshoot_events")),
    }


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def aggregate(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_controller: dict[str, dict[str, Any]] = {}
    for controller in sorted({record["controller"] for record in records}):
        rows = [record for record in records if record["controller"] == controller]
        by_controller[controller] = {
            "matches": len(rows),
            "win_rate": _rate(rows, lambda row: row["outcome"] == "win"),
            "crash_rate": _rate(rows, lambda row: row["outcome"] == "crash"),
            "timeout_rate": _rate(rows, lambda row: row["outcome"] == "timeout"),
            "mean_health_margin": _mean_key(rows, "health_margin"),
            "mean_reward": _mean_key(rows, "total_reward"),
            "mean_ata_deg": _mean_key(rows, "mean_ata_deg"),
            "mean_saturation_ratio": _mean_key(rows, "action_saturation_ratio"),
            "mean_gate_active_ratio": _mean_key(rows, "gate_active_ratio"),
            "mean_rl_inference_calls": _mean_key(rows, "rl_inference_calls"),
            "minimum_altitude_m": min((row["min_altitude_m"] for row in rows if row["min_altitude_m"] is not None), default=None),
        }
    bt = by_controller.get("bt", {})
    candidates = []
    for controller, metrics in by_controller.items():
        if controller == "bt":
            continue
        valid = (
            metrics["crash_rate"] <= bt.get("crash_rate", 0.0) + args.max_crash_rate_regression
            and (metrics["mean_saturation_ratio"] or 0.0) <= (bt.get("mean_saturation_ratio") or 0.0) + args.max_saturation_rate_regression
            and (metrics["mean_saturation_ratio"] or 0.0) <= args.max_saturation_ratio
            and (metrics["minimum_altitude_m"] is None or metrics["minimum_altitude_m"] >= args.minimum_safe_altitude_m)
        )
        score = (
            100.0 * metrics["win_rate"]
            + 10.0 * (metrics["mean_health_margin"] or 0.0)
            + 0.05 * ((bt.get("mean_ata_deg") or 0.0) - (metrics["mean_ata_deg"] or 0.0))
            - 100.0 * metrics["crash_rate"]
            - 20.0 * (metrics["mean_saturation_ratio"] or 0.0)
        )
        delta_vs_bt = {
            "win_rate": metrics["win_rate"] - bt.get("win_rate", 0.0),
            "crash_rate": metrics["crash_rate"] - bt.get("crash_rate", 0.0),
            "health_margin": (metrics["mean_health_margin"] or 0.0) - (bt.get("mean_health_margin") or 0.0),
            "reward": (metrics["mean_reward"] or 0.0) - (bt.get("mean_reward") or 0.0),
            "mean_ata_deg": (metrics["mean_ata_deg"] or 0.0) - (bt.get("mean_ata_deg") or 0.0),
            "saturation_ratio": (metrics["mean_saturation_ratio"] or 0.0) - (bt.get("mean_saturation_ratio") or 0.0),
        }
        candidates.append({"controller": controller, "valid": valid, "score": score, "delta_vs_bt": delta_vs_bt, **metrics})
    valid_candidates = [candidate for candidate in candidates if candidate["valid"]]
    best = max(valid_candidates, key=lambda item: item["score"], default=None)
    return {"controllers": by_controller, "candidates": candidates, "best_valid_candidate": best}


def _rate(rows, predicate) -> float:
    return sum(bool(predicate(row)) for row in rows) / max(1, len(rows))


def _mean_key(rows, key) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def write_outputs(output: Path, args: argparse.Namespace, preflight_result: dict, records: list[dict]) -> dict:
    summary = aggregate(records, args)
    payload = {"preflight": preflight_result, "settings": dict(vars(args)), "summary": summary, "records": records}
    payload["settings"]["scenarios"] = [str(path) for path in args.scenarios]
    (output / "evaluation.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with (output / "matches.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else ["run_id"])
        writer.writeheader()
        writer.writerows(records)
    best = summary["best_valid_candidate"]
    lines = ["# Offensive Hybrid Paired Evaluation", "", f"- Paired runs: `{len(records)}`", f"- BT baseline: `{summary['controllers'].get('bt')}`", f"- Best safe candidate: `{best['controller'] if best else 'none'}`", "", "## Candidates", ""]
    for candidate in summary["candidates"]:
        lines.append(f"- `{candidate['controller']}`: valid={candidate['valid']}, score={candidate['score']:.3f}, win={candidate['win_rate']:.1%}, crash={candidate['crash_rate']:.1%}, margin={candidate['mean_health_margin']}, delta_vs_bt={candidate['delta_vs_bt']}")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ownship-bt-dll", default=str(ROOT / "AIP_Pastel_AggressiveFinish_v2.dll"))
    parser.add_argument("--target-bt-dll", default=str(ROOT / "AIP_Pastel_AggressiveFinish_v2.dll"))
    parser.add_argument("--bt-rule-xml", default=str(ROOT / "Rule_Pastel_AggressiveFinish_v2.xml"))
    parser.add_argument("--observation-mode", default="tactical16")
    parser.add_argument("--scenarios", nargs="+", type=Path, default=[ROOT / "automation/scenarios/offensive_tail.json", ROOT / "automation/scenarios/crossing_left.json", ROOT / "automation/scenarios/crossing_right.json"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 211])
    parser.add_argument("--scales", nargs="+", type=float, default=list(ALLOWED_SCALES))
    parser.add_argument("--max-pairs", type=int, default=0, help="Limit scenario/seed pairs; zero runs all pairs.")
    parser.add_argument("--max-engage-time", type=float, default=60.0)
    parser.add_argument("--episode-step-limit", type=int, default=3600)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--rl-action-repeat", type=int, default=6)
    parser.add_argument("--offensive-min-range-m", type=float, default=152.4)
    parser.add_argument("--offensive-enter-range-m", type=float, default=2400.0)
    parser.add_argument("--offensive-exit-range-m", type=float, default=3000.0)
    parser.add_argument("--offensive-enter-ata-deg", type=float, default=30.0)
    parser.add_argument("--offensive-exit-ata-deg", type=float, default=45.0)
    parser.add_argument("--offensive-enter-target-ata-deg", type=float, default=105.0)
    parser.add_argument("--offensive-exit-target-ata-deg", type=float, default=80.0)
    parser.add_argument("--max-crash-rate-regression", type=float, default=0.05)
    parser.add_argument("--max-saturation-ratio", type=float, default=1.0)
    parser.add_argument("--max-saturation-rate-regression", type=float, default=0.02)
    parser.add_argument("--minimum-safe-altitude-m", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.scenarios = [path.resolve() for path in args.scenarios]
    output = Path(args.output).resolve()
    for name in ("summaries", "telemetry", "analysis", "raw"):
        (output / name).mkdir(parents=True, exist_ok=True)
    preflight_result = preflight(args)
    pairs = [(scenario, seed) for scenario in args.scenarios for seed in args.seeds]
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    records = []
    for scenario, seed in pairs:
        for controller, scale in _controller_specs(args.scales):
            print(f"[evaluate] scenario={_scenario_name(scenario)} seed={seed} controller={controller}", flush=True)
            records.append(run_match(args, output, scenario=scenario, seed=seed, controller=controller, scale=scale))
            write_outputs(output, args, preflight_result, records)
    payload = write_outputs(output, args, preflight_result, records)
    print(json.dumps(payload["summary"]["best_valid_candidate"], indent=2))


if __name__ == "__main__":
    main()
