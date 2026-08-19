"""Export compact, Git-trackable evidence from a state-conditioned evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AXES = {"roll": 0, "pitch": 1, "yaw": 2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def residual_vector(action_class: str, magnitude: float) -> list[float]:
    result = [0.0, 0.0, 0.0, 0.0]
    axis, direction = action_class.rsplit("_", 1)
    result[AXES[axis]] = magnitude if direction == "pos" else -magnitude
    return result


def export(evaluation: Path, output: Path) -> dict[str, Any]:
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    analysis = payload["analysis"]
    suite_cases = {case["name"]: case for case in payload["suite"]["cases"]}
    records = {
        (row.get("state_id"), row["controller"]): row
        for row in payload["records"]
        if row["controller"] not in {"pure_0815", "zero"}
    }
    magnitude = float(payload["frozen_pulse"]["raw_magnitude"])
    scale = float(payload["frozen_pulse"]["scale"])
    fields = (
        "sample_id", "source_git_sha", "scenario_hash", "state_seed", "geometry",
        "canonical_geometry", "snapshot_frame", "shot_window_elapsed",
        "observation_tactical16", "bt_action", "surface_authority",
        "action_class_world", "action_class_canonical", "raw_residual",
        "requested_correction", "applied_correction", "damage_pure",
        "damage_candidate", "damage_delta", "cone_delta", "first_damage_delta",
        "contamination", "ownship_crash", "target_crash", "label_confidence",
        "label_reason",
    )
    rows = []
    for pair in analysis["pulse_pairs"]:
        record = records[(pair["state_id"], pair["pulse"])]
        case = suite_cases[pair["state_id"]]
        scenario = ROOT / case["scenario"]
        snapshot = pair.get("snapshot", {}) or {}
        raw = residual_vector(pair["pulse"], magnitude)
        requested = [scale * value for value in raw]
        applied_abs = [
            float(record.get("correction_roll_mean") or 0.0),
            float(record.get("correction_pitch_mean") or 0.0),
            float(record.get("correction_yaw_mean") or 0.0),
            0.0,
        ]
        applied = [
            (-value if requested[index] < 0.0 else value)
            for index, value in enumerate(applied_abs)
        ]
        meaningful = float(pair["damage_delta"] or 0.0) >= float(
            analysis["thresholds"]["minimum_meaningful_damage_delta"]
        )
        rows.append(
            {
                "sample_id": f"{pair['state_id']}__{pair['pulse']}",
                "source_git_sha": payload["run_fingerprint"]["git_sha"],
                "scenario_hash": sha256(scenario),
                "state_seed": pair["seed"],
                "geometry": pair["geometry"],
                "canonical_geometry": pair["canonical_geometry"],
                "snapshot_frame": (
                    snapshot.get("frame")
                    if snapshot.get("frame") is not None
                    else round(float(snapshot.get("sim_time_s") or 0.0) * 60.0)
                ),
                "shot_window_elapsed": pair["shot_window_elapsed_frames"],
                "observation_tactical16": compact(snapshot.get("observation_tactical16")),
                "bt_action": compact(snapshot.get("bt_action")),
                "surface_authority": compact(
                    {
                        "applied_to_requested_ratio_mean_axis": [
                            record.get("roll_applied_to_requested_ratio"),
                            record.get("pitch_applied_to_requested_ratio"),
                            record.get("yaw_applied_to_requested_ratio"),
                        ]
                    }
                ),
                "action_class_world": pair["pulse"],
                "action_class_canonical": pair["canonical_pulse"],
                "raw_residual": compact(raw),
                "requested_correction": compact(requested),
                "applied_correction": compact(applied),
                "damage_pure": pair["damage_pure"],
                "damage_candidate": pair["damage_pulse"],
                "damage_delta": pair["damage_delta"],
                "cone_delta": pair["cone_delta_s"],
                "first_damage_delta": pair["first_damage_delta_s"],
                "contamination": pair["contamination"],
                "ownship_crash": pair["ownship_crash"],
                "target_crash": pair["target_crash"],
                "label_confidence": 0.0,
                "label_reason": (
                    "MEANINGFUL_ACTION_BUT_DATASET_GATE_FAILED"
                    if meaningful else "EXCLUDED_SIGNAL_GATE_FAILED"
                ),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "csv_sha256": sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.evaluation, args.output), indent=2))


if __name__ == "__main__":
    main()
