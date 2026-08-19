from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REASON_PRIORITY = {
    "OOD": 5,
    "HIGH_UNCERTAINTY": 4,
    "HIGH_REGRESSION_RISK": 3,
    "LOW_PPOSITIVE": 2,
    "NO_ACTION_ADVANTAGE": 1,
}


def load_frames(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if '"record_type":"frame"' in line
    ]


def candidate_from_frame(
    frame: dict[str, Any], *, case_id: str, geometry: str, seed: int
) -> dict[str, Any] | None:
    hybrid = frame.get("hybrid", {})
    prediction = hybrid.get("prediction", {})
    reason = str(prediction.get("abstention_reason", ""))
    priority = REASON_PRIORITY.get(reason, 0)
    if priority <= 0:
        return None
    best = prediction.get("best_rejected_option", {})
    uncertainty = float(best.get("ensemble_std", 0.0))
    conservative = float(best.get("conservative_score", 0.0))
    token = f"{case_id}|{seed}|{frame['frame']}|{reason}"
    event_id = "evt_" + hashlib.sha256(token.encode()).hexdigest()[:16]
    return {
        "event_id": event_id,
        "fight_id": f"fight_{case_id}_s{seed}",
        "trajectory_id": f"fight_{case_id}_s{seed}",
        "scenario_id": geometry,
        "opponent_id": "autopilot",
        "seed": seed,
        "frame": int(frame["frame"]),
        "sim_time_s": float(frame["sim_time_s"]),
        "event_type": "shadow_active_acquisition",
        "diagnostic_failure_family": f"MODEL_{reason}",
        "diagnostic_is_training_label": False,
        "primary_label": "PENDING_PREFIX_REPLAY_PAIRED_DAMAGE",
        "bt_action": hybrid.get("bt_action", frame["ownship_action"]),
        "bt_vp": hybrid.get("bt_vp", [0.0, 0.0, 0.0]),
        "geometry": {
            "distance_m": float(frame["distance_m"]),
            "aim_azimuth_deg": float(frame["aim_azimuth_deg"]),
            "aim_elevation_deg": float(frame["aim_elevation_deg"]),
            "los_azimuth_rate_deg_s": float(frame["los_azimuth_rate_deg_s"]),
            "los_elevation_rate_deg_s": float(frame["los_elevation_rate_deg_s"]),
            "closing_rate_m_s": float(frame["closing_rate_m_s"]),
            "in_wez": bool(frame["in_wez"]),
        },
        "acquisition": {
            "reason": reason,
            "priority": priority,
            "ensemble_std": uncertainty,
            "absolute_conservative_score": abs(conservative),
            "best_rejected_option": best,
        },
    }


def select_active_events(
    candidates: list[dict[str, Any]], count: int, *, minimum_frame_separation: int = 30
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda row: (
            -row["acquisition"]["priority"],
            -row["acquisition"]["ensemble_std"],
            row["acquisition"]["absolute_conservative_score"],
            row["event_id"],
        ),
    )
    selected = []
    by_fight: dict[str, list[int]] = {}
    for row in ordered:
        frames = by_fight.setdefault(row["fight_id"], [])
        if any(abs(row["frame"] - previous) < minimum_frame_separation for previous in frames):
            continue
        selected.append(row)
        frames.append(row["frame"])
        if len(selected) >= count:
            break
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire Shadow decision events for v4")
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--event-count", type=int, default=60)
    parser.add_argument("--minimum-frame-separation", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shadow = args.shadow_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite active acquisition: {output}")
    summary = json.loads((shadow / "summary.json").read_text(encoding="utf-8"))
    candidates = []
    cases = []
    for row in summary["rows"]:
        original_case_id = row["case_id"]
        case_id = f"shadow_{original_case_id}"
        run = shadow / "runs" / original_case_id
        scenario = json.loads((run / "scenario.json").read_text(encoding="utf-8"))
        cases.append(
            {
                "case_id": case_id,
                "geometry": row["geometry"],
                "opponent": row["opponent"],
                "seed": int(row["seed"]),
                "scenario": scenario,
            }
        )
        for frame in load_frames(run / "telemetry.jsonl"):
            candidate = candidate_from_frame(
                frame,
                case_id=case_id,
                geometry=row["geometry"],
                seed=int(row["seed"]),
            )
            if candidate is not None:
                candidates.append(candidate)
    selected = select_active_events(
        candidates,
        args.event_count,
        minimum_frame_separation=args.minimum_frame_separation,
    )
    output.mkdir(parents=True)
    (output / "suite.json").write_text(
        json.dumps({"cases": cases}, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "events.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8"
    )
    acquisition = {
        "schema_version": "shadow_active_acquisition_v4.v1",
        "shadow_decision": summary["decision"],
        "candidate_frames": len(candidates),
        "selected_events": len(selected),
        "minimum_frame_separation": args.minimum_frame_separation,
        "reason_counts": {
            reason: sum(row["acquisition"]["reason"] == reason for row in selected)
            for reason in REASON_PRIORITY
        },
        "primary_label": "PENDING_PREFIX_REPLAY_PAIRED_DAMAGE",
        "diagnostic_taxonomy_is_label": False,
    }
    (output / "summary.json").write_text(
        json.dumps(acquisition, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(acquisition, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
