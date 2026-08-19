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


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dogfight.research.tactical_events import extract_decision_events, summarize_events


EXPECTED_DLL = "4C93B4C6719CB0423388D5FC721D356020A3A36CD5AD2C56B5C3CA795BFE18C9"
EXPECTED_XML = "D84C27B0B8BA22E1649AF2375BE0B83C762BC62EB5047BB590539B374F8271EE"
BASE_GEOMETRIES = (
    "lateral_left",
    "lateral_right",
    "vertical_high",
    "vertical_low",
    "crossing_left",
    "crossing_right",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def verify_champion(dll: Path, xml: Path) -> None:
    if sha256(dll) != EXPECTED_DLL or sha256(xml) != EXPECTED_XML:
        raise ValueError("Pure BT Champion hash mismatch")


def build_discovery_cases(variants_per_geometry: int = 3) -> list[dict[str, Any]]:
    if variants_per_geometry <= 0 or variants_per_geometry % 2 == 0:
        raise ValueError("variants per geometry must be a positive odd number")
    centered = range(-(variants_per_geometry // 2), variants_per_geometry // 2 + 1)
    cases = []
    for geometry in BASE_GEOMETRIES:
        source = ROOT / "automation/scenarios/0815_aim_mirror" / f"{geometry}.json"
        base = json.loads(source.read_text(encoding="utf-8"))
        for variant in centered:
            payload = json.loads(json.dumps(base))
            env = payload["env_config"]
            env["target"][0] += 60.0 * variant
            if geometry.startswith(("lateral", "crossing")):
                sign = -1.0 if "left" in geometry else 1.0
                env["target"][1] += sign * 20.0 * variant
            if geometry.startswith("vertical"):
                sign = -1.0 if geometry == "vertical_high" else 1.0
                env["target"][2] += sign * 40.0 * variant
            heading_sign = -1.0 if "right" in geometry else 1.0
            env["target"][5] += heading_sign * variant
            env["target"][6] += 3.0 * variant
            env["target_autopilot"]["heading_cmd"] = env["target"][5]
            env["target_autopilot"]["altitude_cmd"] = -env["target"][2]
            env["target_autopilot"]["speed_cmd"] = env["target"][6]
            case_id = f"{geometry}_v{variant:+d}".replace("+", "p").replace("-", "m")
            payload["name"] = f"temporal_tactical_v4_{case_id}"
            cases.append(
                {
                    "case_id": case_id,
                    "geometry": geometry,
                    "opponent": "autopilot",
                    "seed": 72001 + len(cases),
                    "scenario": payload,
                }
            )
    extras = (
        ("neutral", [0.0, 0.0, -5000.0, 0.0, 0.0, 0.0, 240.0], [1000.0, 0.0, -5000.0, 0.0, 0.0, 0.0, 225.0]),
        ("head_on", [0.0, 0.0, -5000.0, 0.0, 0.0, 0.0, 240.0], [1600.0, 0.0, -5000.0, 0.0, 0.0, 180.0, 230.0]),
        ("tail_chase", [0.0, 0.0, -5000.0, 0.0, 0.0, 0.0, 235.0], [1200.0, 0.0, -5000.0, 0.0, 0.0, 0.0, 205.0]),
        ("high_closing", [0.0, 0.0, -5000.0, 0.0, 0.0, 0.0, 300.0], [1400.0, 100.0, -5000.0, 0.0, 0.0, 5.0, 180.0]),
        ("long_range", [0.0, 0.0, -5000.0, 0.0, 0.0, 0.0, 250.0], [2600.0, 0.0, -5100.0, 0.0, 0.0, 0.0, 215.0]),
    )
    for name, ownship, target in extras:
        cases.append(
            {
                "case_id": name,
                "geometry": name,
                "opponent": "autopilot",
                "seed": 72001 + len(cases),
                "scenario": {
                    "name": f"temporal_tactical_v4_{name}",
                    "env_config": {
                        "ownship": ownship,
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
            }
        )
    return cases


def run_case(
    case: dict[str, Any],
    *,
    output: Path,
    dll: Path,
    xml: Path,
    episode_frames: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_root = output / "runs" / case["case_id"]
    run_root.mkdir(parents=True, exist_ok=False)
    scenario_path = run_root / "scenario.json"
    result_path = run_root / "result.json"
    telemetry_path = run_root / "telemetry.jsonl"
    scenario_path.write_text(
        json.dumps(case["scenario"], indent=2, sort_keys=True), encoding="utf-8"
    )
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
        "BT_DEFAULT",
        "--prefix-start-frame",
        "0",
        "--prefix-hold-frames",
        "0",
        "--scenario-file",
        str(scenario_path),
        "--seed",
        str(case["seed"]),
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
    if completed.returncode != 0:
        raise RuntimeError(
            f"Pure BT trajectory failed: {case['case_id']} rc={completed.returncode}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    frames = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if '"record_type":"frame"' in line
    ]
    return result, frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Pure BT decision events for v4")
    parser.add_argument("--pure-bt-dll", type=Path, required=True)
    parser.add_argument("--pure-bt-xml", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants-per-geometry", type=int, default=3)
    parser.add_argument("--episode-frames", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dll = args.pure_bt_dll.resolve()
    xml = args.pure_bt_xml.resolve()
    output = args.output_root.resolve()
    verify_champion(dll, xml)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite event evidence: {output}")
    output.mkdir(parents=True)
    cases = build_discovery_cases(args.variants_per_geometry)
    (output / "suite.json").write_text(
        json.dumps({"cases": cases}, indent=2, sort_keys=True), encoding="utf-8"
    )
    started = perf_counter()
    all_events = []
    fights = []
    for index, case in enumerate(cases, start=1):
        result, frames = run_case(
            case,
            output=output,
            dll=dll,
            xml=xml,
            episode_frames=args.episode_frames,
        )
        fight_id = f"fight_{case['case_id']}_s{case['seed']}"
        events = extract_decision_events(
            frames,
            fight_id=fight_id,
            scenario_id=case["geometry"],
            opponent_id=case["opponent"],
            seed=case["seed"],
        )
        all_events.extend(events)
        fights.append(
            {
                "fight_id": fight_id,
                "case_id": case["case_id"],
                "geometry": case["geometry"],
                "opponent": case["opponent"],
                "seed": case["seed"],
                "frames": len(frames),
                "events": len(events),
                "target_crash": bool(result.get("target_crash", False)),
                "ownship_crash": bool(result.get("ownship_crash", False)),
            }
        )
        progress = {
            "completed_fights": index,
            "total_fights": len(cases),
            "events": len(all_events),
            "wall_seconds": perf_counter() - started,
        }
        (output / "progress.json").write_text(
            json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(progress, sort_keys=True), flush=True)
    summary = summarize_events(all_events)
    summary.update(
        {
            "schema_version": "pure_bt_decision_events_v4.v1",
            "target_event_goal": 300,
            "target_event_goal_met": summary["unique_events"] >= 300,
            "raw_frames": sum(row["frames"] for row in fights),
            "target_crashes": sum(row["target_crash"] for row in fights),
            "ownship_crashes": sum(row["ownship_crash"] for row in fights),
            "pure_bt_dll_sha256": EXPECTED_DLL,
            "pure_bt_xml_sha256": EXPECTED_XML,
            "wall_seconds": perf_counter() - started,
        }
    )
    (output / "events.json").write_text(
        json.dumps(all_events, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "fights.json").write_text(
        json.dumps(fights, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
