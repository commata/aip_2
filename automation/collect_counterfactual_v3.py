from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from automation.evaluate_guidance_ablation_v2 import (
    EXPECTED_PURE_DLL_SHA256,
    EXPECTED_PURE_XML_SHA256,
    compact,
    run_one,
    summarize,
    write_csv,
)
from dogfight.ai.guidance_advantage import GUIDANCE_ADVANTAGE_ACTIONS
from dogfight.ai.guidance_selector import GUIDANCE_COMPOSITE_ACTIONS, GUIDANCE_RATE_AWARE_ACTIONS


DEFAULT_OUTPUT = ROOT / "artifacts/evaluations/state_conditioned_hybrid_v3/adaptive_stage1_20260819"
EVALUATION_SCENARIO_ROOT = ROOT / "automation/scenarios/0815_aim_mirror"


def verify_pure_baseline(dll: Path, xml: Path) -> None:
    for path, expected, label in (
        (dll, EXPECTED_PURE_DLL_SHA256, "Pure BT DLL"),
        (xml, EXPECTED_PURE_XML_SHA256, "Pure BT XML"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        if actual != expected:
            raise ValueError(f"{label} SHA256 mismatch: expected={expected}, actual={actual}")


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


def build_evaluation_boundary_cases(
    count: int, *, start_index: int = 0
) -> list[dict[str, Any]]:
    """Sample near the clean-suite geometry without copying its exact states."""
    if count <= 0 or count % 6:
        raise ValueError("evaluation-boundary state count must be a positive multiple of six")
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
        family = families[local_index % len(families)]
        band = local_index // len(families)
        centered = band - 0.5 * (count // len(families) - 1)
        payload = json.loads(
            (EVALUATION_SCENARIO_ROOT / f"{family}.json").read_text(encoding="utf-8")
        )
        env = copy.deepcopy(payload["env_config"])
        env["target"][0] += 10.0 * centered
        lateral_sign = -1.0 if "left" in family else 1.0
        if family.startswith(("lateral", "crossing")):
            env["target"][1] += lateral_sign * 2.0 * centered
        vertical_sign = -1.0 if family == "vertical_high" else 1.0
        if family.startswith("vertical"):
            env["target"][2] += vertical_sign * 3.0 * centered
        heading_sign = -1.0 if "right" in family else 1.0
        env["ownship"][5] += heading_sign * 0.25 * centered
        env["target"][5] -= heading_sign * 0.25 * centered
        env["ownship"][6] += 0.5 * centered
        env["target"][6] -= 0.25 * centered
        env["target_autopilot"]["heading_cmd"] = env["target"][5]
        env["target_autopilot"]["altitude_cmd"] = -env["target"][2]
        env["target_autopilot"]["speed_cmd"] = env["target"][6]
        index = start_index + local_index
        cases.append(
            {
                "case_id": f"v3_boundary_{index + 1:04d}_{family}",
                "seed": 22001 + index,
                "family": family,
                "distance_band": "evaluation_boundary",
                "closing_band": "positive" if env["ownship"][6] > env["target"][6] else "negative",
                "scenario": {
                    "name": f"state_conditioned_v3_boundary_{index + 1:04d}_{family}",
                    "env_config": env,
                },
            }
        )
    return cases


def build_shadow_trace_cases(
    trace_root: Path, *, start_index: int = 0, limit: int | None = None
) -> list[dict[str, Any]]:
    """Turn server-safe Pure-BT shadow decision states into restartable scenarios."""
    cases = []
    seen = set()
    for result_path in sorted((trace_root / "runs").glob("*/shadow.json")):
        family = result_path.parent.name
        for prefix in ("shadow_autopilot_", "micro_autopilot_"):
            if family.startswith(prefix):
                family = family[len(prefix) :].rsplit("_v", 1)[0]
                break
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        trace = payload["ownship_provider_telemetry"].get("selector_decision_trace", [])
        for decision in trace:
            own = [float(value) for value in decision["ownship_server_state"]]
            target = [float(value) for value in decision["target_server_state"]]
            key = tuple(round(value, 6) for value in (*own, *target))
            if key in seen:
                continue
            seen.add(key)
            index = start_index + len(cases)
            cases.append(
                {
                    "case_id": f"v3_trace_{index + 1:04d}_{family}",
                    "seed": 32001 + index,
                    "family": family,
                    "distance_band": "shadow_dynamic_trace",
                    "closing_band": "trace_derived",
                    "scenario": {
                        "name": f"state_conditioned_v3_trace_{index + 1:04d}_{family}",
                        "env_config": {
                            "ownship": own,
                            "target": target,
                            "initial_scenario": {
                                "mode": "default",
                                "legacy_use_random_scenario": False,
                            },
                            "ownship_randomization": {"enabled": False},
                            "target_randomization": {"enabled": False},
                            "target_autopilot": {
                                "heading_cmd": target[5],
                                "altitude_cmd": -target[2],
                                "speed_cmd": target[6],
                            },
                        },
                    },
                    "trace_sim_time_s": float(decision["sim_time_s"]),
                }
            )
            if limit is not None and len(cases) >= limit:
                return cases
    if not cases:
        raise ValueError(f"no selector decision trace states found under {trace_root}")
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


def two_axis_candidates() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": f"{action}__m0.25__d36",
            "action": action,
            "magnitude_deg": 0.25,
            "duration_frames": 36,
        }
        for action in GUIDANCE_COMPOSITE_ACTIONS
    ]


def rate_aware_candidates() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": f"{action}__m0.25__d36",
            "action": action,
            "magnitude_deg": 0.25,
            "duration_frames": 36,
        }
        for action in GUIDANCE_RATE_AWARE_ACTIONS
    ]


def continuous_grid_candidates() -> list[dict[str, Any]]:
    directions = (*GUIDANCE_ADVANTAGE_ACTIONS[1:], *GUIDANCE_COMPOSITE_ACTIONS)
    return [
        {
            "candidate_id": f"{action}__m{magnitude:.2f}__d36",
            "action": action,
            "magnitude_deg": magnitude,
            "duration_frames": 36,
        }
        for magnitude in (0.10, 0.50)
        for action in directions
    ]


def limit_cases_per_family(cases: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("trace per-family limit must be positive")
    counts: dict[str, int] = {}
    selected = []
    for case in cases:
        family = case["family"]
        if counts.get(family, 0) >= limit:
            continue
        counts[family] = counts.get(family, 0) + 1
        selected.append(case)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect adaptive state-action counterfactuals v3")
    parser.add_argument("--pure-bt-dll", type=Path, required=True)
    parser.add_argument("--pure-bt-xml", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--states", type=int, default=120)
    parser.add_argument("--start-index", type=int, default=200)
    parser.add_argument(
        "--profile",
        choices=("adaptive", "evaluation-boundary", "shadow-trace"),
        default="adaptive",
    )
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--trace-per-family", type=int)
    parser.add_argument(
        "--action-level",
        choices=("single-axis", "two-axis", "rate-aware", "continuous-grid"),
        default="single-axis",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pure_dll = args.pure_bt_dll.resolve()
    pure_xml = args.pure_bt_xml.resolve()
    verify_pure_baseline(pure_dll, pure_xml)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.profile == "adaptive":
        cases = build_adaptive_cases(args.states, start_index=args.start_index)
    elif args.profile == "evaluation-boundary":
        cases = build_evaluation_boundary_cases(args.states, start_index=args.start_index)
    else:
        if args.trace_root is None:
            raise ValueError("--trace-root is required for the shadow-trace profile")
        cases = build_shadow_trace_cases(
            args.trace_root.resolve(), start_index=args.start_index, limit=args.states
        )
        if args.trace_per_family is not None:
            cases = limit_cases_per_family(cases, args.trace_per_family)
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
        *(
            coarse_candidates()
            if args.action_level == "single-axis"
            else two_axis_candidates()
            if args.action_level == "two-axis"
            else rate_aware_candidates()
            if args.action_level == "rate-aware"
            else continuous_grid_candidates()
        ),
    ]
    (output / "suite.json").write_text(
        json.dumps(
            {
                "schema_version": "state_conditioned_acquisition_v3.v1",
                "strategy": f"coarse_{args.action_level}_m0.25_d36_{args.profile}",
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
