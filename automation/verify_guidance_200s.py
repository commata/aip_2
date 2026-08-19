from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTROLLERS = ("pure", "bt_default", "bc")


def read_telemetry(path: Path) -> dict:
    frames = 0
    min_altitude = float("inf")
    min_speed = float("inf")
    invalid = 0
    throttle_difference_max = 0.0
    digest = hashlib.sha256()
    with path.open("rb") as binary:
        for raw_line in binary:
            digest.update(raw_line)
            row = json.loads(raw_line.decode("utf-8"))
            if row.get("record_type") != "frame":
                continue
            frames += 1
            min_altitude = min(min_altitude, float(row["ownship"]["altitude_m"]))
            min_speed = min(min_speed, float(row["ownship"]["speed_kcas"]))
            action = np.asarray(row.get("ownship_action", []), dtype=np.float64)
            invalid += int(action.shape != (4,) or not np.all(np.isfinite(action)))
            hybrid = row.get("hybrid", {}) or {}
            bt_action = hybrid.get("bt_action")
            final_action = hybrid.get("final_action")
            if bt_action is not None and final_action is not None:
                throttle_difference_max = max(
                    throttle_difference_max,
                    abs(float(final_action[3]) - float(bt_action[3])),
                )
    if frames == 0:
        raise ValueError(f"no frame records: {path}")
    return {
        "frames": frames,
        "min_altitude_m": min_altitude,
        "min_speed_m_s": min_speed,
        "invalid_actions": invalid,
        "throttle_difference_max": throttle_difference_max,
        "sha256": digest.hexdigest().upper(),
    }


def verify(evaluation_root: Path, aggregate_path: Path) -> dict:
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    rows = []
    for case_root in sorted(path for path in evaluation_root.iterdir() if path.is_dir()):
        for controller in CONTROLLERS:
            result_path = case_root / f"{controller}.json"
            telemetry_path = case_root / f"{controller}.telemetry.jsonl"
            if not result_path.is_file() or not telemetry_path.is_file():
                raise FileNotFoundError(f"incomplete raw run: {case_root.name}/{controller}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            telemetry = read_telemetry(telemetry_path)
            provider = result.get("ownship_provider_telemetry", {}) or {}
            dealt = 1.0 - float(result["target_health"])
            received = 1.0 - float(result["ownship_health"])
            rows.append(
                {
                    "case_id": case_root.name,
                    "controller": controller,
                    "margin": dealt - received,
                    "target_crash": bool(result.get("target_crash")),
                    "ownship_crash": bool(result.get("ownship_crash")),
                    "episode_seconds": float(result.get("episode_seconds", 0.0)),
                    "end_condition": result.get("end_condition"),
                    "nonzero": int(provider.get("nonzero_intervention_frames", 0)),
                    "latency_ms_max": float(
                        provider.get("selector_inference_latency_ms_max", 0.0)
                    ),
                    "throttle_violations": int(provider.get("throttle_violation_steps", 0)),
                    **telemetry,
                }
            )
    if len(rows) != 36:
        raise ValueError(f"expected 36 raw runs, got {len(rows)}")

    pure = {row["case_id"]: row for row in rows if row["controller"] == "pure"}
    hybrid = [row for row in rows if row["controller"] == "bc"]
    deltas = []
    contaminated = 0
    for row in hybrid:
        baseline = pure[row["case_id"]]
        if row["target_crash"] or baseline["target_crash"]:
            contaminated += 1
        else:
            deltas.append(row["margin"] - baseline["margin"])
    values = np.asarray(deltas, dtype=np.float64)
    recomputed = {
        "raw_runs": len(rows),
        "configured_200s_runs": len(rows),
        "completed_200s_timeout_runs": sum(row["episode_seconds"] >= 199.0 for row in rows),
        "natural_terminal_runs": sum(row["episode_seconds"] < 199.0 for row in rows),
        "episode_seconds_min": min(row["episode_seconds"] for row in rows),
        "episode_seconds_max": max(row["episode_seconds"] for row in rows),
        "clean_pairs": len(deltas),
        "contaminated_pairs": contaminated,
        "clean_damage_delta_mean": float(np.mean(values)),
        "clean_damage_delta_median": float(np.median(values)),
        "clean_damage_delta_min": float(np.min(values)),
        "clean_damage_delta_max": float(np.max(values)),
        "positive_pairs": int(np.sum(values > 0.0)),
        "hybrid_nonzero_intervention_frames": sum(row["nonzero"] for row in hybrid),
        "hybrid_ownship_crashes": sum(row["ownship_crash"] for row in hybrid),
        "hybrid_min_altitude_m": min(row["min_altitude_m"] for row in hybrid),
        "hybrid_min_speed_m_s": min(row["min_speed_m_s"] for row in hybrid),
        "hybrid_invalid_actions": sum(row["invalid_actions"] for row in hybrid),
        "hybrid_throttle_violations": sum(row["throttle_violations"] for row in hybrid),
        "hybrid_throttle_difference_max": max(
            row["throttle_difference_max"] for row in hybrid
        ),
        "hybrid_latency_ms_max": max(row["latency_ms_max"] for row in hybrid),
        "telemetry_checksums_verified": len(rows),
        "end_condition_counts": {
            name: sum(row["end_condition"] == name for row in rows)
            for name in sorted({row["end_condition"] for row in rows})
        },
    }
    expected = aggregate["paired"]["bc"]
    comparisons = {
        "configured_runs": recomputed["configured_200s_runs"] == aggregate["configured_200s_runs"],
        "clean_pairs": recomputed["clean_pairs"] == expected["clean_pairs"],
        "contaminated_pairs": recomputed["contaminated_pairs"] == expected["contaminated_pairs"],
        "mean": math.isclose(
            recomputed["clean_damage_delta_mean"], expected["clean_damage_delta_mean"], abs_tol=1e-12
        ),
        "median": math.isclose(
            recomputed["clean_damage_delta_median"], expected["clean_damage_delta_median"], abs_tol=1e-12
        ),
        "min": math.isclose(
            recomputed["clean_damage_delta_min"], expected["clean_damage_delta_min"], abs_tol=1e-12
        ),
        "max": math.isclose(
            recomputed["clean_damage_delta_max"], expected["clean_damage_delta_max"], abs_tol=1e-12
        ),
        "nonzero": recomputed["hybrid_nonzero_intervention_frames"]
        == aggregate["nonzero_intervention_frames"],
        "altitude": math.isclose(
            recomputed["hybrid_min_altitude_m"], aggregate["minimum_altitude_m"], abs_tol=1e-9
        ),
        "latency": math.isclose(
            recomputed["hybrid_latency_ms_max"], aggregate["latency_ms_max"], abs_tol=1e-9
        ),
    }
    if not all(comparisons.values()):
        raise RuntimeError(f"independent recomputation mismatch: {comparisons}")
    return {
        "status": "INDEPENDENT_RECOMPUTATION_PASS",
        "source": "raw result JSON plus frame telemetry JSONL",
        "aggregate": str(aggregate_path.resolve()),
        "recomputed": recomputed,
        "comparisons": comparisons,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Independently recompute 200s Guidance evidence")
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=ROOT / "artifacts/evaluations/guidance_selector/full_200s_v1_20260819",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "automation/evidence/guidance_selector_v1/independent_verification.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = args.evaluation_root.resolve()
    result = verify(root, root / "aggregate.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
