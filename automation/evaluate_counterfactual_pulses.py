"""Repeat Pure/zero baselines and evaluate symmetric first-window residual pulses."""
from __future__ import annotations

import argparse
import csv
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
from dogfight.research.mirror_symmetry import (
    action_class_to_canonical,
    canonical_geometry,
)


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


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256(resolved)}


def build_run_fingerprint(
    args: argparse.Namespace,
    suite_path: Path,
    magnitude: float,
) -> dict[str, Any]:
    """Freeze every input that can change a resumed counterfactual result."""
    return {
        "schema_version": "counterfactual.resume.v2",
        "git_sha": git_sha(),
        "mode": args.mode,
        "suite": file_identity(suite_path),
        "ownship_bt_dll": file_identity(Path(args.ownship_bt_dll)),
        "target_bt_dll": file_identity(Path(args.target_bt_dll)),
        "bt_rule_xml": file_identity(Path(args.bt_rule_xml)),
        "bt_rule_alias": list(args.bt_rule_alias),
        "target_backend": args.target_backend,
        "pure_reference": (
            file_identity(args.pure_reference)
            if args.pure_reference is not None else None
        ),
        "residual_scale": float(args.residual_scale),
        "pulse_frames": int(args.pulse_frames),
        "pulse_start_offset_frames": int(args.pulse_start_offset_frames),
        "pulse_magnitude": float(magnitude),
        "max_engage_time": float(args.max_engage_time),
        "episode_step_limit": int(args.episode_step_limit),
        "shot_window": {
            "max_active_steps": 60,
            "cooldown_steps": 30,
            "rearm_mode": "condition_exit",
        },
    }


def prepare_output(
    output: Path,
    fingerprint: dict[str, Any],
    *,
    resume: bool,
) -> None:
    manifest_path = output / "run_manifest.json"
    existing_files = list(output.iterdir()) if output.is_dir() else []
    if resume:
        if existing_files and not manifest_path.is_file():
            raise ValueError("resume refused: existing output has no run_manifest.json")
        if manifest_path.is_file():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous != fingerprint:
                raise ValueError("resume refused: run fingerprint mismatch")
    elif existing_files:
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(fingerprint, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def telemetry_quality(path: Path) -> dict[str, int]:
    invalid = 0
    throttle_violations = 0
    if not path.is_file():
        return {"invalid_or_nonfinite_actions": 1, "throttle_violations": 1}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("record_type") != "frame":
                continue
            hybrid = row.get("hybrid", {}) or {}
            action = hybrid.get("final_action", row.get("ownship_action"))
            if (
                not isinstance(action, list)
                or len(action) != 4
                or not all(math.isfinite(float(value)) for value in action)
            ):
                invalid += 1
            bt_action = hybrid.get("bt_action")
            if isinstance(action, list) and isinstance(bt_action, list):
                if len(action) != 4 or len(bt_action) != 4:
                    throttle_violations += 1
                elif not math.isclose(
                    float(action[3]), float(bt_action[3]), rel_tol=0.0, abs_tol=1e-7
                ):
                    throttle_violations += 1
    return {
        "invalid_or_nonfinite_actions": invalid,
        "throttle_violations": throttle_violations,
    }


def load_suite(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) < 6:
        raise ValueError("counterfactual suite requires at least six cases")
    names = [str(case["name"]) for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("counterfactual case names must be unique")
    for case in cases:
        int(case["seed"])
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
            digest.update(
                json.dumps(
                    canonical_numbers(stable), sort_keys=True, separators=(",", ":")
                ).encode()
            )
            digest.update(b"\n")
    return digest.hexdigest().upper()


def canonical_numbers(value: Any) -> Any:
    """Normalize signed zero while preserving exact simulator float values."""
    if isinstance(value, dict):
        return {key: canonical_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonical_numbers(item) for item in value]
    if isinstance(value, float) and value == 0.0:
        return 0.0
    return value


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
    offset = int(
        case.get("shot_window_elapsed_frames", args.pulse_start_offset_frames)
    )
    run_id = (
        f"{case['name']}_f{offset:02d}_s{case['seed']}_{controller}_r{repeat:02d}"
    )
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
                "--counterfactual-pulse-start-offset-frames", str(offset),
                "--residual-scale", str(args.residual_scale),
                "--residual-composition", "saturation_aware",
                "--shot-window-max-active-steps", "60",
                "--shot-window-cooldown-steps", "30",
                "--shot-window-rearm-mode", "condition_exit",
            ]
        with (output / "command_history.txt").open("a", encoding="utf-8") as stream:
            stream.write(subprocess.list2cmdline(command) + "\n")
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
    quality = telemetry_quality(telemetry_path)
    provider = result.get("ownship_provider_telemetry", {}) or {}
    snapshot = provider.get("counterfactual_pulse_snapshot", {}) or {}
    world_geometry = str(case.get("geometry", case["name"]))
    record.update(
        {
            "state_id": case["name"],
            "geometry": world_geometry,
            "canonical_geometry": str(
                case.get("canonical_geometry", canonical_geometry(world_geometry))
            ),
            "mirror_pair": case.get("mirror_pair"),
            "state_neighborhood": case.get("state_neighborhood", {}),
            "shot_window_elapsed_frames": offset,
            "scenario": case["scenario"],
            "repeat": repeat,
            "pulse_raw_magnitude": 0.0 if controller in ("pure_0815", "zero") else magnitude,
            "trajectory_sha256": trajectory_signature(telemetry_path),
            "result_sha256": sha256(result_path) if result_path.is_file() else None,
            "telemetry_sha256": sha256(telemetry_path) if telemetry_path.is_file() else None,
            "snapshot": snapshot,
            **quality,
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
    best_by_state: list[dict[str, Any]] = []
    zero_exact = True
    direction_wins: Counter[str] = Counter()
    state_ids = sorted({str(row.get("state_id", row["geometry"])) for row in records})
    for state_id in state_ids:
        state_records = [
            row for row in records if str(row.get("state_id", row["geometry"])) == state_id
        ]
        pure_rows = [row for row in state_records if row["controller"] == "pure_0815"]
        zero_rows = [row for row in state_records if row["controller"] == "zero"]
        if len(pure_rows) != 1 or len(zero_rows) != 1:
            raise ValueError(f"state {state_id!r} requires exactly one Pure and ZERO record")
        pure, zero = pure_rows[0], zero_rows[0]
        geometry = str(pure["geometry"])
        zero_exact = zero_exact and pure["trajectory_sha256"] == zero["trajectory_sha256"]
        clean = []
        for row in state_records:
            if row["controller"] not in PULSES:
                continue
            process_error = bool(row.get("returncode")) or row.get("outcome") == "process_error"
            invalid = int(row.get("invalid_or_nonfinite_actions") or 0)
            throttle_violations = int(row.get("throttle_violations") or 0)
            finite_damage = row.get("damage_dealt") is not None and pure.get("damage_dealt") is not None
            delta = (
                float(row["damage_dealt"]) - float(pure["damage_dealt"])
                if finite_damage
                else None
            )
            item = {
                "state_id": state_id,
                "geometry": geometry,
                "canonical_geometry": row.get("canonical_geometry", canonical_geometry(geometry)),
                "state_neighborhood": row.get("state_neighborhood", {}),
                "shot_window_elapsed_frames": row.get("shot_window_elapsed_frames", 0),
                "seed": row["seed"],
                "pulse": row["controller"],
                "canonical_pulse": action_class_to_canonical(row["controller"], geometry),
                "damage_pure": pure["damage_dealt"], "damage_pulse": row["damage_dealt"],
                "damage_delta": delta,
                "first_damage_delta_s": None if row.get("time_to_first_damage_s") is None or pure.get("time_to_first_damage_s") is None else float(row["time_to_first_damage_s"]) - float(pure["time_to_first_damage_s"]),
                "los_delta_deg": None if row.get("mean_los_deg") is None or pure.get("mean_los_deg") is None else float(row["mean_los_deg"]) - float(pure["mean_los_deg"]),
                "cone_delta_s": None if row.get("damage_cone_time_s") is None or pure.get("damage_cone_time_s") is None else float(row["damage_cone_time_s"]) - float(pure["damage_cone_time_s"]),
                "pulse_steps": row.get("rl_correction_steps"),
                "inference_calls": row.get("rl_inference_calls"),
                "ownship_crash": row.get("ownship_crash"), "target_crash": row.get("target_crash"),
                "process_error": process_error,
                "invalid_or_nonfinite_actions": invalid,
                "throttle_violations": throttle_violations,
                "snapshot": row.get("snapshot", {}),
                "contamination": "TARGET_CRASH_CONTAMINATED" if row.get("target_crash") else "CLEAN",
            }
            rows.append(item)
            if (
                not row.get("target_crash")
                and not row.get("ownship_crash")
                and not process_error
                and not invalid
                and not throttle_violations
                and finite_damage
            ):
                clean.append(item)
        if clean:
            best = max(clean, key=lambda item: item["damage_delta"])
            best_by_state.append(best)
            if best["damage_delta"] >= meaningful:
                direction_wins[best["canonical_pulse"]] += 1
    clean_rows = [
        row for row in rows
        if row["contamination"] == "CLEAN"
        and not row["ownship_crash"]
        and not row["process_error"]
        and not row["invalid_or_nonfinite_actions"]
        and not row["throttle_violations"]
        and row["damage_delta"] is not None
    ]
    best_deltas = [float(row["damage_delta"]) for row in best_by_state]
    significant_best = [row for row in best_by_state if row["damage_delta"] >= meaningful]
    geometry_regressions = [row for row in clean_rows if row["damage_delta"] < regression]
    consistent_direction, consistent_count = direction_wins.most_common(1)[0] if direction_wins else (None, 0)
    mirror_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in significant_best:
        key = (
            str(row["canonical_geometry"]),
            int(row.get("shot_window_elapsed_frames") or 0),
        )
        mirror_groups.setdefault(key, []).append(row)
    mirror_pairs = [group for group in mirror_groups.values() if len(group) >= 2]
    inconsistent_mirror_pairs = [
        group for group in mirror_pairs
        if len({row["canonical_pulse"] for row in group}) != 1
    ]
    process_errors = sum(bool(row.get("returncode")) for row in records)
    invalid_actions = sum(int(row.get("invalid_or_nonfinite_actions") or 0) for row in records)
    throttle_violations = sum(int(row.get("throttle_violations") or 0) for row in records)
    ownship_crashes = sum(bool(row.get("ownship_crash")) for row in records)
    positive_ratio = (
        sum(float(row["damage_delta"]) > 0.0 for row in clean_rows) / len(clean_rows)
        if clean_rows else 0.0
    )
    meaningful_geometries = {str(row["geometry"]) for row in significant_best}
    raw_recomputed = recompute_pulse_summary(rows, meaningful, regression)
    recompute_matches = bool(
        raw_recomputed["clean_pairs"] == len(clean_rows)
        and math.isclose(raw_recomputed["positive_ratio"], positive_ratio, abs_tol=1e-15)
        and raw_recomputed["large_regressions"] == len(geometry_regressions)
        and raw_recomputed["process_errors"] == sum(row["process_error"] for row in rows)
    )
    sufficient = bool(
        zero_exact
        and len(meaningful_geometries) >= 4
        and consistent_count >= 2
        and best_deltas
        and statistics.median(best_deltas) > 0.0
        and positive_ratio >= 2.0 / 3.0
        and not geometry_regressions
        and mirror_pairs
        and not inconsistent_mirror_pairs
        and process_errors == 0
        and invalid_actions == 0
        and throttle_violations == 0
        and ownship_crashes == 0
        and all(float(row.get("inference_calls") or 0.0) == 0.0 for row in clean_rows)
        and recompute_matches
    )
    return {
        "status": "COUNTERFACTUAL_SIGNAL_SUFFICIENT" if sufficient else "COUNTERFACTUAL_SIGNAL_INSUFFICIENT",
        "promotion_status": "DATASET_ALLOWED" if sufficient else "NOT_PROMOTED",
        "thresholds": thresholds,
        "zero_residual_exact_equality": zero_exact,
        "clean_pulse_pairs": len(clean_rows),
        "contaminated_pulse_pairs": len(rows) - len(clean_rows),
        "significant_positive_best_geometries": len(significant_best),
        "meaningful_world_geometry_count": len(meaningful_geometries),
        "best_geometry_count": len(best_by_state),
        "pooled_clean_positive_ratio": positive_ratio,
        "median_best_geometry_damage_delta": statistics.median(best_deltas) if best_deltas else None,
        "consistent_winning_direction": consistent_direction,
        "consistent_winning_direction_count": consistent_count,
        "direction_win_counts": dict(direction_wins),
        "canonical_mirror_pair_count": len(mirror_pairs),
        "canonical_mirror_consistent": bool(mirror_pairs and not inconsistent_mirror_pairs),
        "canonical_mirror_inconsistencies": inconsistent_mirror_pairs,
        "geometry_regressions": geometry_regressions,
        "process_errors": process_errors,
        "invalid_or_nonfinite_actions": invalid_actions,
        "throttle_violations": throttle_violations,
        "ownship_crashes": ownship_crashes,
        "raw_recomputed": raw_recomputed,
        "raw_aggregate_recompute_match": recompute_matches,
        "best_by_geometry": best_by_state,
        "pulse_pairs": rows,
    }


def recompute_pulse_summary(
    rows: list[dict[str, Any]],
    meaningful: float,
    regression: float,
) -> dict[str, Any]:
    """Independently recompute gate-critical totals from raw paired rows."""
    clean = []
    for row in rows:
        if (
            row.get("contamination") == "CLEAN"
            and not row.get("ownship_crash")
            and not row.get("process_error")
            and not row.get("invalid_or_nonfinite_actions")
            and not row.get("throttle_violations")
            and row.get("damage_delta") is not None
        ):
            clean.append(row)
    return {
        "clean_pairs": len(clean),
        "positive_pairs": sum(float(row["damage_delta"]) > 0.0 for row in clean),
        "positive_ratio": (
            sum(float(row["damage_delta"]) > 0.0 for row in clean) / len(clean)
            if clean else 0.0
        ),
        "meaningful_positive_pairs": sum(
            float(row["damage_delta"]) >= meaningful for row in clean
        ),
        "large_regressions": sum(
            float(row["damage_delta"]) < regression for row in clean
        ),
        "process_errors": sum(bool(row.get("process_error")) for row in rows),
        "target_crash_contaminated": sum(
            row.get("contamination") == "TARGET_CRASH_CONTAMINATED" for row in rows
        ),
    }


def write_episode_records(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "run_id", "state_id", "seed", "geometry", "canonical_geometry",
        "shot_window_elapsed_frames", "controller", "damage_dealt", "damage_received",
        "ownship_crash", "target_crash", "returncode",
        "invalid_or_nonfinite_actions", "throttle_violations", "trajectory_sha256",
        "result_sha256", "telemetry_sha256",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def reuse_pure_records(reference_path: Path, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clone a frozen deterministic Pure trajectory for case neighborhoods only."""
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    analysis = payload.get("analysis", {}) or {}
    if not analysis.get("pure_exact_determinism"):
        raise ValueError("Pure reference is not marked exact deterministic")
    source_records = [
        row for row in payload.get("records", []) if row.get("controller") == "pure_0815"
    ]
    reused = []
    for case in cases:
        geometry = str(case.get("geometry", case["name"]))
        matches = [row for row in source_records if row.get("geometry") == geometry]
        if not matches:
            raise ValueError(f"Pure reference has no geometry {geometry!r}")
        row = dict(matches[0])
        row.update(
            {
                "run_id": f"{case['name']}_pure_reference",
                "state_id": case["name"],
                "seed": int(case["seed"]),
                "geometry": geometry,
                "canonical_geometry": str(
                    case.get("canonical_geometry", canonical_geometry(geometry))
                ),
                "mirror_pair": case.get("mirror_pair"),
                "state_neighborhood": case.get("state_neighborhood", {}),
                "shot_window_elapsed_frames": int(
                    case.get("shot_window_elapsed_frames", 0)
                ),
                "baseline_reused": True,
                "baseline_reference_path": str(reference_path.resolve()),
                "baseline_reference_sha256": sha256(reference_path),
                "invalid_or_nonfinite_actions": 0,
                "throttle_violations": 0,
            }
        )
        reused.append(row)
    return reused


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mode", choices=("baseline", "pulses"), required=True)
    value.add_argument("--suite", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--threshold-manifest", type=Path)
    value.add_argument("--pure-reference", type=Path)
    value.add_argument("--ownship-bt-dll", required=True)
    value.add_argument("--target-backend", choices=("autopilot", "bt"), default="autopilot")
    value.add_argument("--target-bt-dll", required=True)
    value.add_argument("--bt-rule-xml", required=True)
    value.add_argument("--bt-rule-alias", action="append", default=[])
    value.add_argument("--baseline-repeats", type=int, default=3)
    value.add_argument("--residual-scale", type=float, default=0.125)
    value.add_argument("--pulse-frames", type=int, default=6)
    value.add_argument("--pulse-start-offset-frames", type=int, default=0)
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
    historical = historical_pulse_magnitude(args.historical_glob)
    magnitude = float(args.pulse_magnitude) if args.pulse_magnitude is not None else float(historical["raw_magnitude"])
    output = args.output.resolve()
    fingerprint = build_run_fingerprint(args, suite_path, magnitude)
    prepare_output(output, fingerprint, resume=args.resume)
    for name in ("summaries", "telemetry", "raw"):
        (output / name).mkdir(parents=True, exist_ok=True)
    controllers = (
        ("pure_0815", "zero")
        if args.mode == "baseline"
        else (("zero", *PULSES) if args.pure_reference else ("pure_0815", "zero", *PULSES))
    )
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
    if args.mode == "pulses" and args.pure_reference is not None:
        records.extend(reuse_pure_records(args.pure_reference, suite["cases"]))
    if args.mode == "baseline":
        analysis = analyze_baseline(records)
    else:
        if args.threshold_manifest is None:
            raise ValueError("pulses mode requires --threshold-manifest frozen before pulse runs")
        threshold_payload = json.loads(args.threshold_manifest.read_text(encoding="utf-8"))
        thresholds = threshold_payload.get("analysis", threshold_payload)
        analysis = analyze_pulses(records, thresholds)
    payload = {
        "run_fingerprint": fingerprint,
        "settings": vars_json(args), "suite": suite,
        "suite_sha256": sha256(suite_path),
        "historical_pulse": historical,
        "frozen_pulse": {"raw_magnitude": magnitude, "scale": args.residual_scale, "frames": args.pulse_frames},
        "analysis": analysis, "records": records,
    }
    (output / "evaluation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_episode_records(output / "episode_records.csv", records)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    return 0


def vars_json(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


if __name__ == "__main__":
    raise SystemExit(main())
