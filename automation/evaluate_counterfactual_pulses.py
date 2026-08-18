"""Repeat Pure/zero baselines and evaluate symmetric first-window residual pulses."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from automation.evaluate_aim_residual import build_record


PULSES = (
    "roll_pos", "roll_neg", "pitch_pos", "pitch_neg", "yaw_pos", "yaw_neg"
)
SIGNATURE_KEYS = (
    "outcome", "end_condition", "episode_seconds", "damage_dealt",
    "damage_received", "mean_los_deg", "los_rate_rms_deg_s",
    "damage_cone_time_s", "time_to_first_damage_s", "min_altitude_m",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_suite(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) < 6:
        raise ValueError("counterfactual suite requires at least six cases")
    names = [str(case["name"]) for case in cases]
    seeds = [int(case["seed"]) for case in cases]
    if len(set(names)) != len(names) or len(set(seeds)) != len(seeds):
        raise ValueError("counterfactual case names and seeds must be unique")
    return payload


def historical_pulse_magnitude(patterns: list[str]) -> dict[str, Any]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(ROOT.glob(pattern))
    files = sorted(set(path.resolve() for path in files))
    actions: list[list[float]] = []
    entry_rows = 0
    for path in files:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                hybrid = row.get("hybrid", {}) or {}
                gate = hybrid.get("shot_window_gate", {}) or {}
                entry_rows += int(bool(gate.get("entry")))
                action = hybrid.get("raw_residual_action")
                if hybrid.get("rl_action_refreshed") and isinstance(action, list):
                    actions.append([float(value) for value in action[:3]])
    if actions:
        absolute = np.abs(np.asarray(actions, dtype=np.float64))
        magnitude = float(np.median(absolute))
        source = "historical_ppo_i15_median_abs_surface_action"
        axis_medians = np.median(absolute, axis=0).tolist()
        p75 = float(np.percentile(absolute, 75))
    else:
        magnitude = 0.5
        source = "engineering_fallback_no_historical_raw_action"
        axis_medians = None
        p75 = None
    return {
        "source": source,
        "raw_magnitude": magnitude,
        "files": [str(path.relative_to(ROOT)) for path in files],
        "file_count": len(files),
        "refreshed_action_count": len(actions),
        "entry_row_count": entry_rows,
        "axis_median_abs": axis_medians,
        "p75_abs_all_surfaces": p75,
    }


def trajectory_signature(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("record_type") != "frame":
                continue
            stable = {
                "frame": row.get("frame"),
                "distance_m": row.get("distance_m"),
                "ata_deg": row.get("ata_deg"),
                "target_ata_deg": row.get("target_ata_deg"),
                "ownship_damage": row.get("ownship_damage"),
                "target_damage": row.get("target_damage"),
                "ownship": row.get("ownship"),
                "target": row.get("target"),
                "ownship_action": row.get("ownship_action"),
                "target_action": row.get("target_action"),
            }
            digest.update(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode())
            digest.update(b"\n")
    return digest.hexdigest().upper()


def common_command(args: argparse.Namespace, case: dict[str, Any], result: Path, telemetry: Path) -> list[str]:
    command = [
        sys.executable, str(ROOT / "run_local_dogfight.py"),
        "--target-backend", args.target_backend,
        "--ownship-bt-dll", str(Path(args.ownship_bt_dll).resolve()),
        "--target-bt-dll", str(Path(args.target_bt_dll).resolve()),
        "--bt-rule-xml", str(Path(args.bt_rule_xml).resolve()),
        "--bt-rule-alias-only", "--bt-turn-throttle-mode", "raw",
        "--observation-mode", "tactical16",
        "--scenario-file", str((ROOT / case["scenario"]).resolve()),
        "--seed", str(case["seed"]),
        "--max-engage-time", str(args.max_engage_time),
        "--episode-step-limit", str(args.episode_step_limit),
        "--result-json", str(result), "--telemetry-jsonl", str(telemetry),
    ]
    for alias in args.bt_rule_alias:
        command += ["--bt-rule-alias", alias]
    return command


def run_episode(
    args: argparse.Namespace,
    case: dict[str, Any],
    output: Path,
    controller: str,
    repeat: int,
    magnitude: float,
) -> dict[str, Any]:
    run_id = f"{case['name']}_s{case['seed']}_{controller}_r{repeat:02d}"
    result_path = output / "summaries" / f"{run_id}.json"
    telemetry_path = output / "telemetry" / f"{run_id}.jsonl"
    stdout_path = output / "raw" / f"{run_id}.stdout.txt"
    stderr_path = output / "raw" / f"{run_id}.stderr.txt"
    if args.resume and result_path.is_file() and telemetry_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        returncode, wall_seconds, resumed = 0, 0.0, True
    else:
        command = common_command(args, case, result_path, telemetry_path)
        if controller == "pure_0815":
            command += ["--ownship-backend", "bt"]
        else:
            command += [
                "--ownship-backend", "counterfactual_pulse",
                "--counterfactual-pulse", controller,
                "--counterfactual-pulse-magnitude", str(magnitude),
                "--counterfactual-pulse-frames", str(args.pulse_frames),
                "--residual-scale", str(args.residual_scale),
                "--residual-composition", "saturation_aware",
                "--shot-window-max-active-steps", "60",
                "--shot-window-cooldown-steps", "30",
                "--shot-window-rearm-mode", "condition_exit",
            ]
        protected = ROOT / "aircraft" / "f16" / "f16_init.xml"
        protected_bytes = protected.read_bytes()
        started = time.monotonic()
        try:
            process = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=args.timeout_seconds,
                check=False,
            )
            returncode = process.returncode
            stdout, stderr = process.stdout, process.stderr
        except subprocess.TimeoutExpired as error:
            returncode = 124
            stdout, stderr = error.stdout or "", error.stderr or ""
        finally:
            if protected.read_bytes() != protected_bytes:
                protected.write_bytes(protected_bytes)
        wall_seconds = time.monotonic() - started
        resumed = False
        stdout_path.write_text(str(stdout), encoding="utf-8")
        stderr_path.write_text(str(stderr), encoding="utf-8")
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    record = build_record(
        run_id, int(case["seed"]), controller, None if controller == "pure_0815" else args.residual_scale,
        result, returncode=returncode, wall_seconds=wall_seconds, resumed=resumed,
    )
    record.update(
        {
            "geometry": case["name"],
            "scenario": case["scenario"],
            "repeat": repeat,
            "pulse_raw_magnitude": 0.0 if controller in ("pure_0815", "zero") else magnitude,
            "trajectory_sha256": trajectory_signature(telemetry_path),
            "result_sha256": sha256(result_path) if result_path.is_file() else None,
            "telemetry_sha256": sha256(telemetry_path) if telemetry_path.is_file() else None,
        }
    )
    return record


def signature(row: dict[str, Any]) -> tuple[Any, ...]:
    values = []
    for key in SIGNATURE_KEYS:
        value = row.get(key)
        values.append(round(float(value), 12) if isinstance(value, (int, float)) else value)
    return tuple(values)


def spread(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": math.nan, "max": math.nan, "range": math.nan, "mad": math.nan}
    median = statistics.median(values)
    return {
        "min": min(values), "max": max(values), "range": max(values) - min(values),
        "mad": statistics.median(abs(value - median) for value in values),
    }


def analyze_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    all_abs_deviations: list[float] = []
    exact = True
    zero_exact = True
    for geometry in sorted({str(row["geometry"]) for row in records}):
        pure = [row for row in records if row["geometry"] == geometry and row["controller"] == "pure_0815"]
        zero = [row for row in records if row["geometry"] == geometry and row["controller"] == "zero"]
        damages = [float(row["damage_dealt"]) for row in pure]
        median_damage = statistics.median(damages)
        all_abs_deviations.extend(abs(value - median_damage) for value in damages)
        pure_exact = len({row["trajectory_sha256"] for row in pure}) == 1
        zero_match = {row["trajectory_sha256"] for row in zero} == {row["trajectory_sha256"] for row in pure}
        exact = exact and pure_exact
        zero_exact = zero_exact and zero_match
        cases[geometry] = {
            "pure_repeats": len(pure), "zero_repeats": len(zero),
            "pure_unique_trajectories": len({row["trajectory_sha256"] for row in pure}),
            "zero_unique_trajectories": len({row["trajectory_sha256"] for row in zero}),
            "pure_damage": spread(damages),
            "pure_zero_trajectory_exact": zero_match,
            "pure_zero_metric_exact": {signature(row) for row in pure} == {signature(row) for row in zero},
        }
    p95_variation = float(np.percentile(all_abs_deviations, 95)) if all_abs_deviations else 0.0
    meaningful = max(0.001, 2.0 * p95_variation)
    return {
        "status": "BASELINE_FROZEN",
        "pure_exact_determinism": exact,
        "zero_residual_exact_equality": zero_exact,
        "p95_abs_damage_deviation": p95_variation,
        "minimum_meaningful_damage_delta": meaningful,
        "maximum_geometry_regression": -max(0.003, 3.0 * meaningful),
        "cases": cases,
    }


def analyze_pulses(records: list[dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    meaningful = float(thresholds["minimum_meaningful_damage_delta"])
    regression = float(thresholds["maximum_geometry_regression"])
    rows: list[dict[str, Any]] = []
    best_by_geometry: list[dict[str, Any]] = []
    zero_exact = True
    direction_wins: Counter[str] = Counter()
    for geometry in sorted({str(row["geometry"]) for row in records}):
        pure = next(row for row in records if row["geometry"] == geometry and row["controller"] == "pure_0815")
        zero = next(row for row in records if row["geometry"] == geometry and row["controller"] == "zero")
        zero_exact = zero_exact and pure["trajectory_sha256"] == zero["trajectory_sha256"]
        clean = []
        for row in records:
            if row["geometry"] != geometry or row["controller"] not in PULSES:
                continue
            delta = float(row["damage_dealt"]) - float(pure["damage_dealt"])
            item = {
                "geometry": geometry, "seed": row["seed"], "pulse": row["controller"],
                "damage_pure": pure["damage_dealt"], "damage_pulse": row["damage_dealt"],
                "damage_delta": delta,
                "first_damage_delta_s": None if row.get("time_to_first_damage_s") is None or pure.get("time_to_first_damage_s") is None else float(row["time_to_first_damage_s"]) - float(pure["time_to_first_damage_s"]),
                "los_delta_deg": None if row.get("mean_los_deg") is None or pure.get("mean_los_deg") is None else float(row["mean_los_deg"]) - float(pure["mean_los_deg"]),
                "cone_delta_s": None if row.get("damage_cone_time_s") is None or pure.get("damage_cone_time_s") is None else float(row["damage_cone_time_s"]) - float(pure["damage_cone_time_s"]),
                "pulse_steps": row.get("rl_correction_steps"),
                "inference_calls": row.get("rl_inference_calls"),
                "ownship_crash": row.get("ownship_crash"), "target_crash": row.get("target_crash"),
                "contamination": "TARGET_CRASH_CONTAMINATED" if row.get("target_crash") else "CLEAN",
            }
            rows.append(item)
            if not row.get("target_crash") and not row.get("ownship_crash"):
                clean.append(item)
        if clean:
            best = max(clean, key=lambda item: item["damage_delta"])
            best_by_geometry.append(best)
            if best["damage_delta"] >= meaningful:
                direction_wins[best["pulse"]] += 1
    clean_rows = [row for row in rows if row["contamination"] == "CLEAN" and not row["ownship_crash"]]
    best_deltas = [float(row["damage_delta"]) for row in best_by_geometry]
    significant_best = [row for row in best_by_geometry if row["damage_delta"] >= meaningful]
    geometry_regressions = [row for row in best_by_geometry if row["damage_delta"] < regression]
    consistent_direction, consistent_count = direction_wins.most_common(1)[0] if direction_wins else (None, 0)
    sufficient = bool(
        zero_exact
        and len(significant_best) >= 4
        and consistent_count >= 2
        and statistics.median(best_deltas) >= meaningful
        and not geometry_regressions
        and all(float(row.get("inference_calls") or 0.0) == 0.0 for row in clean_rows)
    )
    return {
        "status": "COUNTERFACTUAL_SIGNAL_SUFFICIENT" if sufficient else "COUNTERFACTUAL_SIGNAL_INSUFFICIENT",
        "promotion_status": "DATASET_ALLOWED" if sufficient else "NOT_PROMOTED",
        "thresholds": thresholds,
        "zero_residual_exact_equality": zero_exact,
        "clean_pulse_pairs": len(clean_rows),
        "contaminated_pulse_pairs": len(rows) - len(clean_rows),
        "significant_positive_best_geometries": len(significant_best),
        "best_geometry_count": len(best_by_geometry),
        "pooled_clean_positive_ratio": sum(row["damage_delta"] > 0 for row in clean_rows) / max(1, len(clean_rows)),
        "median_best_geometry_damage_delta": statistics.median(best_deltas) if best_deltas else None,
        "consistent_winning_direction": consistent_direction,
        "consistent_winning_direction_count": consistent_count,
        "direction_win_counts": dict(direction_wins),
        "geometry_regressions": geometry_regressions,
        "best_by_geometry": best_by_geometry,
        "pulse_pairs": rows,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mode", choices=("baseline", "pulses"), required=True)
    value.add_argument("--suite", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--threshold-manifest", type=Path)
    value.add_argument("--ownship-bt-dll", required=True)
    value.add_argument("--target-backend", choices=("autopilot", "bt"), default="autopilot")
    value.add_argument("--target-bt-dll", required=True)
    value.add_argument("--bt-rule-xml", required=True)
    value.add_argument("--bt-rule-alias", action="append", default=[])
    value.add_argument("--baseline-repeats", type=int, default=3)
    value.add_argument("--residual-scale", type=float, default=0.125)
    value.add_argument("--pulse-frames", type=int, default=6)
    value.add_argument("--pulse-magnitude", type=float)
    value.add_argument("--historical-glob", action="append", default=["artifacts/evaluations/shot_window_research/stage1_ppo*i000015*screening_20260818/case_*/telemetry/*hybrid*.jsonl"])
    value.add_argument("--max-engage-time", type=float, default=30.0)
    value.add_argument("--episode-step-limit", type=int, default=1800)
    value.add_argument("--timeout-seconds", type=float, default=120.0)
    value.add_argument("--resume", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    suite_path = args.suite.resolve()
    suite = load_suite(suite_path)
    output = args.output.resolve()
    for name in ("summaries", "telemetry", "raw"):
        (output / name).mkdir(parents=True, exist_ok=True)
    historical = historical_pulse_magnitude(args.historical_glob)
    magnitude = float(args.pulse_magnitude) if args.pulse_magnitude is not None else float(historical["raw_magnitude"])
    controllers = ("pure_0815", "zero") if args.mode == "baseline" else ("pure_0815", "zero", *PULSES)
    repeats = args.baseline_repeats if args.mode == "baseline" else 1
    records = []
    total = len(suite["cases"]) * len(controllers) * repeats
    index = 0
    for case in suite["cases"]:
        for repeat in range(1, repeats + 1):
            for controller in controllers:
                index += 1
                print(f"[{index}/{total}] {case['name']} seed={case['seed']} {controller} repeat={repeat}", flush=True)
                record = run_episode(args, case, output, controller, repeat, magnitude)
                records.append(record)
                payload = {"settings": vars_json(args), "suite": suite, "historical_pulse": historical, "records": records}
                (output / "evaluation.partial.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.mode == "baseline":
        analysis = analyze_baseline(records)
    else:
        if args.threshold_manifest is None:
            raise ValueError("pulses mode requires --threshold-manifest frozen before pulse runs")
        threshold_payload = json.loads(args.threshold_manifest.read_text(encoding="utf-8"))
        thresholds = threshold_payload.get("analysis", threshold_payload)
        analysis = analyze_pulses(records, thresholds)
    payload = {
        "settings": vars_json(args), "suite": suite,
        "suite_sha256": sha256(suite_path),
        "historical_pulse": historical,
        "frozen_pulse": {"raw_magnitude": magnitude, "scale": args.residual_scale, "frames": args.pulse_frames},
        "analysis": analysis, "records": records,
    }
    (output / "evaluation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    return 0


def vars_json(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


if __name__ == "__main__":
    raise SystemExit(main())
