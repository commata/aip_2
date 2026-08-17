from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from dogfight.ai.hybrid_action_provider import AimResidualGate


FEATURES = (
    "distance_m",
    "ata_deg",
    "target_ata_deg",
    "aa_deg",
    "aim_azimuth_deg",
    "aim_elevation_deg",
    "los_azimuth_rate_deg_s",
    "los_elevation_rate_deg_s",
    "closing_rate_m_s",
)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _quantiles(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    values = np.asarray(
        [value for row in rows if (value := _finite(row.get(key))) is not None],
        dtype=np.float64,
    )
    if not values.size:
        return None
    return {
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _load_frames(path: Path) -> list[dict[str, Any]]:
    return [
        row
        for line in path.read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line)).get("record_type") == "frame"
    ]


def _pure_record(case_dir: Path) -> dict[str, Any]:
    evaluation = json.loads((case_dir / "evaluation.json").read_text(encoding="utf-8"))
    records = evaluation.get("records", [])
    return next(
        (dict(row) for row in records if row.get("controller") == "pure_0815"),
        {},
    )


def _shot_condition(row: dict[str, Any], config: dict[str, Any]) -> bool:
    phase, half_angle, phase_range = AimResidualGate.phase_limits(
        float(row.get("sim_time_s", 0.0))
    )
    del phase
    distance = _finite(row.get("distance_m"))
    error = _finite(row.get("ata_deg"))
    target_ata = _finite(row.get("target_ata_deg"))
    return bool(
        distance is not None
        and error is not None
        and target_ata is not None
        and config["min_range_m"] <= distance
        <= phase_range + config["enter_range_margin_m"]
        and error <= half_angle + config["enter_angle_margin_deg"]
        and target_ata >= config["enter_min_target_ata_deg"]
    )


def analyze(input_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    telemetry_files = sorted(input_root.glob("case_*/telemetry/*pure_0815*.jsonl"))
    if not telemetry_files:
        raise FileNotFoundError(f"Pure BT telemetry not found under {input_root}")
    segments: dict[str, list[dict[str, Any]]] = {
        "negative_control_minus6_to_minus3s": [],
        "pre_damage_minus3_to_0s": [],
        "pre_damage_minus1_to_0s": [],
        "damage_onset_0_to_plus0_25s": [],
        "cone_inside": [],
        "near_shot_without_damage_next0_25s": [],
    }
    episodes = []
    for path in telemetry_files:
        case_dir = path.parent.parent
        record = _pure_record(case_dir)
        frames = _load_frames(path)
        contaminated = bool(
            record.get("target_crash")
            or record.get("ownship_crash")
            or record.get("returncode")
        )
        first_damage = next(
            (
                float(row["sim_time_s"])
                for row in frames
                if float(row.get("target_damage", 0.0)) > 1e-12
            ),
            None,
        )
        first_cone = next(
            (float(row["sim_time_s"]) for row in frames if row.get("in_wez")),
            None,
        )
        first_candidate = next(
            (
                float(row["sim_time_s"])
                for row in frames
                if _shot_condition(row, config)
            ),
            None,
        )
        episodes.append(
            {
                "case": case_dir.name,
                "seed": record.get("seed"),
                "frames": len(frames),
                "target_crash_contaminated": contaminated,
                "first_candidate_s": first_candidate,
                "first_cone_s": first_cone,
                "first_damage_s": first_damage,
                "candidate_lead_to_damage_s": (
                    first_damage - first_candidate
                    if first_damage is not None and first_candidate is not None
                    else None
                ),
                "final_damage": (
                    float(frames[-1].get("target_damage", 0.0)) if frames else None
                ),
            }
        )
        if contaminated or first_damage is None:
            continue
        for index, row in enumerate(frames):
            relative = float(row["sim_time_s"]) - first_damage
            if -6.0 <= relative < -3.0:
                segments["negative_control_minus6_to_minus3s"].append(row)
            if -3.0 <= relative < 0.0:
                segments["pre_damage_minus3_to_0s"].append(row)
            if -1.0 <= relative < 0.0:
                segments["pre_damage_minus1_to_0s"].append(row)
            if 0.0 <= relative <= 0.25:
                segments["damage_onset_0_to_plus0_25s"].append(row)
            if row.get("in_wez"):
                segments["cone_inside"].append(row)
            if _shot_condition(row, config):
                future = frames[index : index + 16]
                initial_damage = float(row.get("target_damage", 0.0))
                future_damage = max(
                    (float(item.get("target_damage", 0.0)) for item in future),
                    default=initial_damage,
                )
                if future_damage <= initial_damage + 1e-12:
                    segments["near_shot_without_damage_next0_25s"].append(row)

    statistics = {}
    for name, rows in segments.items():
        statistics[name] = {
            "frames": len(rows),
            "in_wez_ratio": (
                sum(bool(row.get("in_wez")) for row in rows) / max(1, len(rows))
            ),
            "candidate_condition_ratio": (
                sum(_shot_condition(row, config) for row in rows) / max(1, len(rows))
            ),
            "features": {key: _quantiles(rows, key) for key in FEATURES},
        }
    clean = [row for row in episodes if not row["target_crash_contaminated"]]
    leads = [
        float(row["candidate_lead_to_damage_s"])
        for row in clean
        if row["candidate_lead_to_damage_s"] is not None
    ]
    return {
        "schema_version": "shot_window_feature_analysis.v1",
        "source_root": str(input_root.resolve()),
        "feature_contract": list(FEATURES),
        "shot_window_candidate": dict(config),
        "episodes": episodes,
        "clean_episode_count": len(clean),
        "contaminated_episode_count": len(episodes) - len(clean),
        "candidate_lead_to_damage_s": {
            "mean": float(np.mean(leads)) if leads else None,
            "min": min(leads) if leads else None,
            "max": max(leads) if leads else None,
        },
        "segments": statistics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enter-angle-margin-deg", type=float, default=1.5)
    parser.add_argument("--enter-range-margin-m", type=float, default=25.0)
    parser.add_argument("--enter-min-target-ata-deg", type=float, default=150.0)
    parser.add_argument("--min-range-m", type=float, default=152.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = analyze(
        args.input_root,
        {
            "min_range_m": args.min_range_m,
            "enter_angle_margin_deg": args.enter_angle_margin_deg,
            "enter_range_margin_m": args.enter_range_margin_m,
            "enter_min_target_ata_deg": args.enter_min_target_ata_deg,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
