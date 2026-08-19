from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dogfight.ai.prefix_replay import compare_prefix_snapshots
from dogfight.ai.tactical_modes import TACTICAL_HOLD_FRAMES, TACTICAL_MODES_T1, TACTICAL_MODES_T2


EXPECTED_DLL = "4C93B4C6719CB0423388D5FC721D356020A3A36CD5AD2C56B5C3CA795BFE18C9"
EXPECTED_XML = "D84C27B0B8BA22E1649AF2375BE0B83C762BC62EB5047BB590539B374F8271EE"
HORIZONS_SECONDS = (1, 2, 4, 8, 12)
EVENT_PRIORITY = (
    "cone_exit",
    "phase_cone_approach",
    "aim_error_growth_turn",
    "target_crossing",
    "los_rate_sign_reversal",
    "closing_extreme",
    "first_damage_pre",
    "damage_window_post",
    "aim_error_local_minimum",
    "range_boundary_crossing",
    "bt_vp_jump",
    "surface_saturation",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def verify_champion(dll: Path, xml: Path) -> None:
    if sha256(dll) != EXPECTED_DLL or sha256(xml) != EXPECTED_XML:
        raise ValueError("Pure BT Champion hash mismatch")


def select_balanced_events(
    events: list[dict[str, Any]],
    count: int,
    *,
    offset: int = 0,
) -> list[dict[str, Any]]:
    if count <= 0 or offset < 0:
        raise ValueError("event count must be positive and offset non-negative")
    priority = {name: index for index, name in enumerate(EVENT_PRIORITY)}
    ordered = sorted(
        events,
        key=lambda row: (
            priority.get(row["event_type"], len(priority)),
            row["frame"],
            row["event_id"],
        ),
    )
    by_geometry: dict[str, list[dict[str, Any]]] = {}
    for row in ordered:
        if row["event_type"] == "miss_recovery":
            continue
        by_geometry.setdefault(row["scenario_id"], []).append(row)
    selected = []
    consumed = {geometry: 0 for geometry in by_geometry}
    geometries = sorted(by_geometry)
    target = count + offset
    while len(selected) < target:
        progressed = False
        for geometry in geometries:
            index = consumed[geometry]
            rows = by_geometry[geometry]
            if index >= len(rows):
                continue
            selected.append(rows[index])
            consumed[geometry] += 1
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break
    result = selected[offset : offset + count]
    if len(result) < count:
        raise ValueError(f"only {len(result)} balanced events available, requested {count}")
    return result


def candidate_options(level: str) -> list[dict[str, Any]]:
    modes = TACTICAL_MODES_T1[1:] if level == "T1" else TACTICAL_MODES_T2
    return [
        {
            "option_id": f"{mode}__d{duration}",
            "mode": mode,
            "hold_frames": duration,
        }
        for mode in modes
        for duration in TACTICAL_HOLD_FRAMES
    ]


def load_frames(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if '"record_type":"frame"' in line
    ]


def run_rollout(
    *,
    label: str,
    output: Path,
    scenario: Path,
    seed: int,
    dll: Path,
    xml: Path,
    decision_frame: int,
    mode: str,
    hold_frames: int,
    episode_frames: int,
) -> dict[str, Any]:
    run_root = output / "runs" / label
    run_root.mkdir(parents=True, exist_ok=False)
    result_path = run_root / "result.json"
    telemetry_path = run_root / "telemetry.jsonl"
    command = [
        sys.executable,
        str(ROOT / "run_local_dogfight.py"),
        "--ownship-backend",
        "prefix_tactical",
        "--target-backend",
        "autopilot",
        "--ownship-bt-dll",
        str(dll),
        "--bt-rule-xml",
        str(xml),
        "--bt-rule-alias",
        "Rule_DCS_GDCC_0815.xml",
        "--bt-rule-alias-only",
        "--bt-turn-throttle-mode",
        "raw",
        "--prefix-tactical-mode",
        mode,
        "--prefix-start-frame",
        str(decision_frame),
        "--prefix-hold-frames",
        str(hold_frames),
        "--scenario-file",
        str(scenario),
        "--seed",
        str(seed),
        "--max-engage-time",
        str(episode_frames / 60.0),
        "--episode-step-limit",
        str(episode_frames),
        "--result-json",
        str(result_path),
        "--telemetry-jsonl",
        str(telemetry_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=max(120.0, episode_frames / 10.0),
        check=False,
    )
    (run_root / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (run_root / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(f"Tactical rollout failed: {label} rc={completed.returncode}")
    return {
        "result": json.loads(result_path.read_text(encoding="utf-8")),
        "frames": load_frames(telemetry_path),
    }


def cumulative_outcome(
    frames: list[dict[str, Any]], decision_frame: int, horizon_frames: int | None
) -> dict[str, float]:
    end = len(frames) if horizon_frames is None else min(
        len(frames), decision_frame + horizon_frames
    )
    segment = frames[:end]
    damage_dealt = float(sum(float(row["target_damage"]) for row in segment))
    damage_received = float(sum(float(row["ownship_damage"]) for row in segment))
    if not segment:
        return {
            "damage_dealt": 0.0,
            "damage_received": 0.0,
            "net_health_margin": 0.0,
            "cone_dwell_s": 0.0,
        }
    return {
        "damage_dealt": damage_dealt,
        "damage_received": damage_received,
        "net_health_margin": damage_dealt - damage_received,
        "cone_dwell_s": sum(bool(row["in_wez"]) for row in segment) / 60.0,
        "range_m": float(segment[-1]["distance_m"]),
        "los_deg": float(segment[-1]["ata_deg"]),
        "closing_m_s": float(segment[-1]["closing_rate_m_s"]),
        "altitude_m": float(segment[-1]["ownship"]["altitude_m"]),
        "speed_m_s": float(segment[-1]["ownship"]["speed_kcas"]),
    }


def paired_record(
    event: dict[str, Any],
    option: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    decision_frame = int(event["frame"]) + 1
    baseline_snapshot = baseline["result"]["ownship_provider_telemetry"][
        "prefix_snapshot"
    ]
    candidate_snapshot = candidate["result"]["ownship_provider_telemetry"][
        "prefix_snapshot"
    ]
    parity = compare_prefix_snapshots(baseline_snapshot, candidate_snapshot)
    horizons: dict[str, Any] = {}
    for seconds in HORIZONS_SECONDS:
        base = cumulative_outcome(baseline["frames"], decision_frame, seconds * 60)
        trial = cumulative_outcome(candidate["frames"], decision_frame, seconds * 60)
        horizons[f"plus_{seconds}s"] = {
            "baseline": base,
            "candidate": trial,
            "damage_dealt_delta": trial["damage_dealt"] - base["damage_dealt"],
            "damage_received_delta": trial["damage_received"] - base["damage_received"],
            "net_health_margin_delta": trial["net_health_margin"]
            - base["net_health_margin"],
        }
    base_terminal = cumulative_outcome(baseline["frames"], decision_frame, None)
    trial_terminal = cumulative_outcome(candidate["frames"], decision_frame, None)
    base_result = baseline["result"]
    trial_result = candidate["result"]
    contaminated = bool(base_result.get("target_crash") or trial_result.get("target_crash"))
    throttle_violations = int(
        candidate["result"]["ownship_provider_telemetry"].get(
            "throttle_violations", 0
        )
    )
    return {
        "event_id": event["event_id"],
        "fight_id": event["fight_id"],
        "scenario_id": event["scenario_id"],
        "opponent_id": event["opponent_id"],
        "seed": event["seed"],
        "decision_frame": decision_frame,
        "event_type": event["event_type"],
        "diagnostic_failure_family": event["diagnostic_failure_family"],
        "option_id": option["option_id"],
        "mode": option["mode"],
        "hold_frames": option["hold_frames"],
        "prefix_parity": parity,
        "clean": bool(parity["match"] and not contaminated and throttle_violations == 0),
        "target_crash_contaminated": contaminated,
        "ownship_crash": bool(trial_result.get("ownship_crash", False)),
        "throttle_violations": throttle_violations,
        "horizons": horizons,
        "terminal": {
            "baseline": base_terminal,
            "candidate": trial_terminal,
            "damage_dealt_delta": trial_terminal["damage_dealt"]
            - base_terminal["damage_dealt"],
            "damage_received_delta": trial_terminal["damage_received"]
            - base_terminal["damage_received"],
            "net_health_margin_delta": trial_terminal["net_health_margin"]
            - base_terminal["net_health_margin"],
            "baseline_end_condition": base_result.get("end_condition", ""),
            "candidate_end_condition": trial_result.get("end_condition", ""),
        },
    }


def summarize_oracle(
    records: list[dict[str, Any]],
    *,
    epsilon: float = 1e-9,
    large_regression_threshold: float = 1e-6,
) -> dict[str, Any]:
    clean = [row for row in records if row["clean"]]
    by_event: dict[str, list[dict[str, Any]]] = {}
    for row in clean:
        by_event.setdefault(row["event_id"], []).append(row)
    oracle = []
    for event_id, rows in by_event.items():
        best = max(
            rows,
            key=lambda row: (
                row["terminal"]["damage_dealt_delta"],
                row["terminal"]["net_health_margin_delta"],
                -row["hold_frames"],
                row["option_id"],
            ),
        )
        uplift = float(best["terminal"]["damage_dealt_delta"])
        if uplift <= epsilon:
            oracle.append(
                {
                    "event_id": event_id,
                    "scenario_id": best["scenario_id"],
                    "diagnostic_failure_family": best["diagnostic_failure_family"],
                    "selected_option": "BT_DEFAULT",
                    "hold_frames": 0,
                    "damage_dealt_delta": 0.0,
                    "net_health_margin_delta": 0.0,
                }
            )
        else:
            oracle.append(
                {
                    "event_id": event_id,
                    "scenario_id": best["scenario_id"],
                    "diagnostic_failure_family": best["diagnostic_failure_family"],
                    "selected_option": best["option_id"],
                    "hold_frames": best["hold_frames"],
                    "damage_dealt_delta": uplift,
                    "net_health_margin_delta": float(
                        best["terminal"]["net_health_margin_delta"]
                    ),
                }
            )
    uplifts = np.asarray([row["damage_dealt_delta"] for row in oracle], dtype=np.float64)
    interventions = [row for row in oracle if row["selected_option"] != "BT_DEFAULT"]
    option_ids = sorted({row["option_id"] for row in clean})
    static_values = {}
    for option_id in option_ids:
        rows = [row for row in clean if row["option_id"] == option_id]
        static_values[option_id] = float(
            np.mean([row["terminal"]["damage_dealt_delta"] for row in rows])
        )
    best_static = max(static_values, key=static_values.get) if static_values else ""
    geometry_positive = sorted(
        {row["scenario_id"] for row in interventions if row["damage_dealt_delta"] > epsilon}
    )
    large_regressions = sum(
        row["terminal"]["damage_dealt_delta"] < -large_regression_threshold
        for row in clean
    )
    coverage = len(interventions) / max(1, len(oracle))
    gate = {
        "positive_opportunities": len(interventions),
        "meaningful_nondefault_coverage": coverage,
        "positive_geometry_count": len(geometry_positive),
        "coverage_or_absolute_count": coverage >= 0.05 or len(interventions) >= 6,
        "multiple_geometries": len(geometry_positive) >= 2,
        "direction_positive": bool(uplifts.size and float(np.mean(uplifts)) > epsilon),
        "requires_independent_revalidation": True,
    }
    return {
        "events_requested": len({row["event_id"] for row in records}),
        "events_with_clean_options": len(oracle),
        "clean_pairs": len(clean),
        "contaminated_or_invalid_pairs": len(records) - len(clean),
        "oracle_nondefault_coverage": coverage,
        "oracle_intervention_mean": float(
            np.mean([row["damage_dealt_delta"] for row in interventions])
        )
        if interventions
        else 0.0,
        "oracle_intervention_median": float(
            np.median([row["damage_dealt_delta"] for row in interventions])
        )
        if interventions
        else 0.0,
        "oracle_intervention_positive_ratio": float(
            np.mean([row["damage_dealt_delta"] > epsilon for row in interventions])
        )
        if interventions
        else 0.0,
        "overall_policy_value": float(np.mean(uplifts)) if uplifts.size else 0.0,
        "best_static_tactical_mode": best_static,
        "best_static_value": static_values.get(best_static, 0.0),
        "oracle_static_gap": (
            float(np.mean(uplifts)) - static_values.get(best_static, 0.0)
            if uplifts.size
            else 0.0
        ),
        "positive_geometries": geometry_positive,
        "large_regression_pairs": int(large_regressions),
        "duration_distribution": {
            str(duration): sum(row["hold_frames"] == duration for row in interventions)
            for duration in TACTICAL_HOLD_FRAMES
        },
        "gate": gate,
        "oracle": oracle,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tactical Oracle v4")
    parser.add_argument("--pure-bt-dll", type=Path, required=True)
    parser.add_argument("--pure-bt-xml", type=Path, required=True)
    parser.add_argument("--events-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--level", choices=("T1", "T2"), default="T1")
    parser.add_argument("--event-count", type=int, default=12)
    parser.add_argument("--event-offset", type=int, default=0)
    parser.add_argument("--post-event-frames", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dll = args.pure_bt_dll.resolve()
    xml = args.pure_bt_xml.resolve()
    events_root = args.events_root.resolve()
    output = args.output_root.resolve()
    verify_champion(dll, xml)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Tactical Oracle evidence: {output}")
    output.mkdir(parents=True)
    events = json.loads((events_root / "events.json").read_text(encoding="utf-8"))
    suite = json.loads((events_root / "suite.json").read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in suite["cases"]}
    selected = select_balanced_events(events, args.event_count, offset=args.event_offset)
    options = candidate_options(args.level)
    (output / "selection.json").write_text(
        json.dumps(
            {"level": args.level, "events": selected, "options": options},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    records = []
    started = perf_counter()
    for event_index, event in enumerate(selected, start=1):
        case_id = event["fight_id"][len("fight_") :].rsplit("_s", 1)[0]
        case = cases[case_id]
        event_root = output / "scenarios" / event["event_id"]
        event_root.mkdir(parents=True)
        scenario_path = event_root / "scenario.json"
        scenario_path.write_text(
            json.dumps(case["scenario"], indent=2, sort_keys=True), encoding="utf-8"
        )
        decision_frame = int(event["frame"]) + 1
        episode_frames = decision_frame + args.post_event_frames
        baseline = run_rollout(
            label=f"{event['event_id']}__BT_DEFAULT",
            output=output,
            scenario=scenario_path,
            seed=int(event["seed"]),
            dll=dll,
            xml=xml,
            decision_frame=decision_frame,
            mode="BT_DEFAULT",
            hold_frames=0,
            episode_frames=episode_frames,
        )
        for option in options:
            candidate = run_rollout(
                label=f"{event['event_id']}__{option['option_id']}",
                output=output,
                scenario=scenario_path,
                seed=int(event["seed"]),
                dll=dll,
                xml=xml,
                decision_frame=decision_frame,
                mode=option["mode"],
                hold_frames=option["hold_frames"],
                episode_frames=episode_frames,
            )
            records.append(paired_record(event, option, baseline, candidate))
        progress = {
            "completed_events": event_index,
            "total_events": len(selected),
            "completed_pairs": len(records),
            "total_pairs": len(selected) * len(options),
            "wall_seconds": perf_counter() - started,
        }
        (output / "progress.json").write_text(
            json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(progress, sort_keys=True), flush=True)
    summary = summarize_oracle(records)
    summary.update(
        {
            "schema_version": "tactical_oracle_v4.v1",
            "stage": "DISCOVERY" if args.event_offset == 0 else "INDEPENDENT_REVALIDATION",
            "action_space_level": args.level,
            "epsilon": 1e-9,
            "large_regression_threshold": 1e-6,
            "wall_seconds": perf_counter() - started,
        }
    )
    (output / "pairs.json").write_text(
        json.dumps(records, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "oracle.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "oracle"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
