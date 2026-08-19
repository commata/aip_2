"""Extract clean PPO i15 Damage pairs and first Shot-Window pulse telemetry."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def first_window(path: Path, frames: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    simulator_rows = [row for row in rows if row.get("record_type") == "frame"]
    entry_index = next(
        (
            index
            for index, row in enumerate(simulator_rows)
            if (row.get("hybrid", {}).get("shot_window_gate", {}) or {}).get("entry")
        ),
        None,
    )
    if entry_index is None:
        return {"entry_observed": False, "frames": []}
    selected = simulator_rows[entry_index : entry_index + frames]
    samples = []
    for row in selected:
        hybrid = row.get("hybrid", {}) or {}
        gate = hybrid.get("shot_window_gate", {}) or {}
        samples.append(
            {
                "frame": row.get("frame"),
                "sim_time_s": row.get("sim_time_s"),
                "distance_m": row.get("distance_m"),
                "ata_deg": row.get("ata_deg"),
                "target_ata_deg": row.get("target_ata_deg"),
                "los_azimuth_rate_deg_s": row.get("los_azimuth_rate_deg_s"),
                "los_elevation_rate_deg_s": row.get("los_elevation_rate_deg_s"),
                "aim_error_deg": gate.get("aim_error_deg"),
                "aim_error_rate_deg_s": gate.get("aim_error_rate_deg_s"),
                "raw_residual_action": hybrid.get("raw_residual_action"),
                "applied_rl_correction": hybrid.get("applied_rl_correction"),
                "bt_action": hybrid.get("bt_action"),
                "final_action": hybrid.get("final_action"),
                "rl_action_refreshed": hybrid.get("rl_action_refreshed"),
            }
        )
    return {
        "entry_observed": True,
        "entry_frame": selected[0].get("frame") if selected else None,
        "entry_time_s": selected[0].get("sim_time_s") if selected else None,
        "frames": samples,
    }


def controller_records(payload: dict[str, Any]) -> tuple[dict[str, dict], dict[str, dict]]:
    pure: dict[str, dict] = {}
    hybrid: dict[str, dict] = {}
    for row in payload.get("records", []):
        geometry = str(row.get("suite_case") or row.get("variant_name"))
        if row.get("controller") == "pure_0815":
            pure[geometry] = row
        else:
            hybrid[geometry] = row
    return pure, hybrid


def analyze(inputs: list[Path], frames: int) -> dict[str, Any]:
    pairs = []
    first_windows = []
    raw_actions = []
    for root in inputs:
        match = re.search(r"s(\d+)_i000015", root.name)
        training_seed = int(match.group(1)) if match else None
        payload = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
        pure, hybrid = controller_records(payload)
        for geometry in sorted(set(pure) & set(hybrid)):
            base, candidate = pure[geometry], hybrid[geometry]
            damage_delta = finite(candidate.get("damage_dealt")) - finite(base.get("damage_dealt"))
            contaminated = bool(base.get("target_crash") or candidate.get("target_crash"))
            pairs.append(
                {
                    "training_seed": training_seed,
                    "geometry": geometry,
                    "evaluation_seed": candidate.get("seed"),
                    "damage_delta": damage_delta,
                    "first_damage_delta_s": None
                    if finite(candidate.get("time_to_first_damage_s")) is None
                    or finite(base.get("time_to_first_damage_s")) is None
                    else finite(candidate.get("time_to_first_damage_s"))
                    - finite(base.get("time_to_first_damage_s")),
                    "los_delta_deg": finite(candidate.get("mean_los_deg"))
                    - finite(base.get("mean_los_deg")),
                    "cone_delta_s": finite(candidate.get("damage_cone_time_s"))
                    - finite(base.get("damage_cone_time_s")),
                    "ownship_crash": bool(candidate.get("ownship_crash")),
                    "target_crash": bool(candidate.get("target_crash")),
                    "contamination": "TARGET_CRASH_CONTAMINATED" if contaminated else "CLEAN",
                }
            )
        for telemetry in sorted(root.glob("case_*/telemetry/*hybrid*.jsonl")):
            window = first_window(telemetry, frames)
            window.update({"training_seed": training_seed, "telemetry": str(telemetry)})
            first_windows.append(window)
            for row in window["frames"]:
                action = row.get("raw_residual_action")
                if isinstance(action, list):
                    raw_actions.append([float(value) for value in action[:3]])
    clean = [row for row in pairs if row["contamination"] == "CLEAN" and not row["ownship_crash"]]
    actions = np.abs(np.asarray(raw_actions, dtype=np.float64)) if raw_actions else np.empty((0, 3))
    per_seed = {}
    for seed in sorted({row["training_seed"] for row in clean}):
        rows = [row for row in clean if row["training_seed"] == seed]
        per_seed[str(seed)] = {
            "clean_pairs": len(rows),
            "mean_damage_delta": float(np.mean([row["damage_delta"] for row in rows])),
            "positive_pairs": sum(row["damage_delta"] > 0.0 for row in rows),
        }
    return {
        "status": "DIAGNOSTIC_COMPLETE",
        "inputs": [str(path) for path in inputs],
        "clean_pairs": len(clean),
        "contaminated_pairs": len(pairs) - len(clean),
        "clean_positive_pairs": sum(row["damage_delta"] > 0.0 for row in clean),
        "clean_positive_ratio": sum(row["damage_delta"] > 0.0 for row in clean) / max(1, len(clean)),
        "clean_mean_damage_delta": float(np.mean([row["damage_delta"] for row in clean])),
        "per_training_seed": per_seed,
        "historical_raw_action": {
            "samples": len(raw_actions),
            "median_abs_all_surfaces": float(np.median(actions)) if actions.size else None,
            "median_abs_axis": np.median(actions, axis=0).tolist() if actions.size else None,
            "p75_abs_all_surfaces": float(np.percentile(actions, 75)) if actions.size else None,
        },
        "pairs": pairs,
        "first_windows": first_windows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=6)
    args = parser.parse_args()
    payload = analyze([path.resolve() for path in args.input], args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in ("pairs", "first_windows")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
