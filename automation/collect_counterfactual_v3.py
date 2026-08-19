from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from automation.evaluate_guidance_ablation_v2 import compact, run_one, summarize, write_csv
from dogfight.ai.guidance_advantage import GUIDANCE_ADVANTAGE_ACTIONS


DEFAULT_OUTPUT = ROOT / "artifacts/evaluations/state_conditioned_hybrid_v3/adaptive_stage1_20260819"


def build_adaptive_cases(count: int, *, start_index: int = 0) -> list[dict[str, Any]]:
    if count <= 0 or count % 6:
        raise ValueError("adaptive state count must be a positive multiple of six")
    families = (
        "lateral_left",
        "lateral_right",
        "vertical_high",
        "vertical_low",
        "crossing_left",
        "crossing_right",
    )
    cases = []
    for local_index in range(count):
        index = start_index + local_index
        family = families[index % len(families)]
        band = index // len(families)
        distance = 650.0 + 25.0 * (index % 9)
        lateral = float(((index * 11) % 31) - 15)
        altitude_delta = float(((index * 7) % 31) - 15)
        target_heading = float(((index * 5) % 5) - 2)
        if family == "lateral_left":
            lateral = -(15.0 + 4.0 * (band % 15))
        elif family == "lateral_right":
            lateral = 15.0 + 4.0 * (band % 15)
        elif family == "vertical_high":
            altitude_delta = 15.0 + 5.0 * (band % 15)
        elif family == "vertical_low":
            altitude_delta = -(15.0 + 5.0 * (band % 15))
        elif family == "crossing_left":
            lateral = -(20.0 + 4.0 * (band % 15))
            target_heading = -(3.0 + float(band % 8))
        elif family == "crossing_right":
            lateral = 20.0 + 4.0 * (band % 15)
            target_heading = 3.0 + float(band % 8)
        own_altitude = 4500.0 + 35.0 * (index % 13)
        own_speed = 212.0 + 3.0 * (index % 8)
        target_speed = 210.0 + 3.0 * ((index * 3) % 8)
        own_heading = float((index % 5) - 2)
        target_altitude = own_altitude + altitude_delta
        seed = 12001 + index
        cases.append(
            {
                "case_id": f"v3_state_{index + 1:04d}_{family}",
                "seed": seed,
                "family": family,
                "distance_band": "near" if distance < 900.0 else "mid",
                "closing_band": "positive" if own_speed > target_speed else "negative",
                "scenario": {
                    "name": f"state_conditioned_v3_{index + 1:04d}_{family}",
                    "env_config": {
                        "ownship": [
                            0.0,
                            0.0,
                            -own_altitude,
                            0.0,
                            0.0,
                            own_heading,
                            own_speed,
                        ],
                        "target": [
                            distance,
                            lateral,
                            -target_altitude,
                            0.0,
                            0.0,
                            target_heading,
                            target_speed,
                        ],
                        "initial_scenario": {
                            "mode": "default",
                            "legacy_use_random_scenario": False,
                        },
                        "ownship_randomization": {"enabled": False},
                        "target_randomization": {"enabled": False},
                        "target_autopilot": {
                            "heading_cmd": target_heading,
                            "altitude_cmd": target_altitude,
                            "speed_cmd": target_speed,
                        },
                    },
                },
            }
        )
    return cases


def coarse_candidates() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": f"{action}__m0.25__d36",
            "action": action,
            "magnitude_deg": 0.25,
            "duration_frames": 36,
        }
        for action in GUIDANCE_ADVANTAGE_ACTIONS[1:]
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect adaptive state-action counterfactuals v3")
    parser.add_argument("--pure-bt-dll", type=Path, required=True)
    parser.add_argument("--pure-bt-xml", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--states", type=int, default=120)
    parser.add_argument("--start-index", type=int, default=200)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pure_dll = args.pure_bt_dll.resolve()
    pure_xml = args.pure_bt_xml.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = build_adaptive_cases(args.states, start_index=args.start_index)
    candidates = [
        {
            "candidate_id": "PURE_BT",
            "action": "BT_DEFAULT",
            "magnitude_deg": 0.0,
            "duration_frames": 0,
        },
        {
            "candidate_id": "BT_DEFAULT",
            "action": "BT_DEFAULT",
            "magnitude_deg": 0.25,
            "duration_frames": 36,
        },
        *coarse_candidates(),
    ]
    (output / "suite.json").write_text(
        json.dumps(
            {
                "schema_version": "state_conditioned_acquisition_v3.v1",
                "strategy": "coarse_all_axis_sign_m0.25_d36",
                "cases": cases,
                "candidates": candidates,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    records = []
    started = perf_counter()
    for case_index, case in enumerate(cases, start=1):
        for candidate in candidates:
            result = run_one(
                case,
                candidate,
                output,
                args.timeout_s,
                pure_dll,
                pure_xml,
            )
            records.append(compact(case, candidate, result))
        progress = {
            "completed_states": case_index,
            "total_states": len(cases),
            "completed_rollouts": len(records),
            "total_rollouts": len(cases) * len(candidates),
            "wall_seconds": perf_counter() - started,
        }
        (output / "progress.json").write_text(
            json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(progress, sort_keys=True), flush=True)
    aggregate, pairs = summarize(records)
    aggregate["schema_version"] = "state_conditioned_acquisition_v3.v1"
    aggregate["wall_seconds"] = perf_counter() - started
    (output / "records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "pairs.json").write_text(
        json.dumps(pairs, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(output / "paired_results.csv", pairs)
    (output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
