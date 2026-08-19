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
EXPECTED_DLL = "4C93B4C6719CB0423388D5FC721D356020A3A36CD5AD2C56B5C3CA795BFE18C9"
EXPECTED_XML = "D84C27B0B8BA22E1649AF2375BE0B83C762BC62EB5047BB590539B374F8271EE"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def select_cases(suite: dict[str, Any], count: int) -> list[dict[str, Any]]:
    cases = list(suite["cases"])
    if count <= 0 or count > len(cases):
        raise ValueError(f"case count must be in [1,{len(cases)}]")
    ordered = sorted(
        cases,
        key=lambda row: hashlib.sha256(
            f"{row['geometry']}|{row['case_id']}|{row['seed']}".encode()
        ).hexdigest(),
    )
    return ordered[:count]


def load_frames(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if '"record_type":"frame"' in line
    ]


def summarize_shadow(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frames = sum(int(row["frames"]) for row in rows)
    nondefault = sum(int(row["nondefault_predictions"]) for row in rows)
    ood = sum(int(row["abstention_reasons"].get("OOD", 0)) for row in rows)
    nondefault_geometries = sorted(
        {row["geometry"] for row in rows if row["nondefault_predictions"] > 0}
    )
    exact_mismatches = sum(int(row["exact_bt_mismatches"]) for row in rows)
    invalid = sum(int(row["invalid_frames"]) for row in rows)
    throttle = sum(int(row["throttle_violations"]) for row in rows)
    crashes = sum(bool(row["ownship_crash"]) for row in rows)
    over = sum(int(row["latency_over_166_7ms"]) for row in rows)
    gate = {
        "nondefault_prediction_exists": nondefault > 0,
        "multiple_geometry_actionability": len(nondefault_geometries) >= 2,
        "ood_collapse_absent": ood / max(1, frames) < 0.95,
        "exact_shadow_command": exact_mismatches == 0,
        "invalid_zero": invalid == 0,
        "throttle_violation_zero": throttle == 0,
        "ownship_crash_zero": crashes == 0,
        "latency_over_limit_zero": over == 0,
    }
    return {
        "schema_version": "temporal_tactical_shadow_v4.v1",
        "fights": len(rows),
        "frames": frames,
        "nondefault_predictions": nondefault,
        "nondefault_geometry_count": len(nondefault_geometries),
        "nondefault_geometries": nondefault_geometries,
        "ood_abstention_ratio": ood / max(1, frames),
        "exact_bt_mismatches": exact_mismatches,
        "invalid_frames": invalid,
        "throttle_violations": throttle,
        "ownship_crashes": crashes,
        "latency_over_166_7ms": over,
        "gate": gate,
        "decision": "SHADOW_GATE_PASSED" if all(gate.values()) else "SHADOW_GATE_FAILED",
        "rows": rows,
    }


def run_case(
    case: dict[str, Any],
    *,
    output: Path,
    bundle: Path,
    dll: Path,
    xml: Path,
    episode_frames: int,
) -> dict[str, Any]:
    run = output / "runs" / case["case_id"]
    run.mkdir(parents=True)
    scenario = run / "scenario.json"
    result_path = run / "result.json"
    telemetry_path = run / "telemetry.jsonl"
    scenario.write_text(
        json.dumps(case["scenario"], indent=2, sort_keys=True), encoding="utf-8"
    )
    command = [
        sys.executable,
        str(ROOT / "run_local_dogfight.py"),
        "--ownship-backend",
        "temporal_tactical",
        "--target-backend",
        "autopilot",
        "--ownship-bundle-dir",
        str(bundle),
        "--ownship-bt-dll",
        str(dll),
        "--bt-rule-xml",
        str(xml),
        "--bt-rule-alias",
        "Rule_DCS_GDCC_0815.xml",
        "--bt-rule-alias-only",
        "--bt-turn-throttle-mode",
        "raw",
        "--tactical-shadow-mode",
        "--scenario-file",
        str(scenario),
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
    (run / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (run / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(f"Shadow run failed: {case['case_id']} rc={completed.returncode}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    provider = result["ownship_provider_telemetry"]
    frames = load_frames(telemetry_path)
    exact_mismatches = sum(
        not bool(row.get("hybrid", {}).get("exact_bt_command", False)) for row in frames
    )
    return {
        "case_id": case["case_id"],
        "geometry": case["geometry"],
        "opponent": case["opponent"],
        "seed": case["seed"],
        "frames": provider["frames"],
        "nondefault_predictions": provider["nondefault_predictions"],
        "abstention_reasons": provider["abstention_reasons"],
        "exact_bt_mismatches": exact_mismatches,
        "invalid_frames": provider["invalid_frames"],
        "throttle_violations": provider["throttle_violations"],
        "ownship_crash": bool(result.get("ownship_crash", False)),
        "target_crash": bool(result.get("target_crash", False)),
        "latency_p99_ms": provider["latency_ms"]["p99"],
        "latency_max_ms": provider["latency_ms"]["max"],
        "latency_over_166_7ms": provider["latency_ms"]["over_166_7ms"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Temporal Tactical v4 Shadow")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--pure-bt-dll", type=Path, required=True)
    parser.add_argument("--pure-bt-xml", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-count", type=int, default=11)
    parser.add_argument("--episode-frames", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = args.bundle.resolve()
    dll = args.pure_bt_dll.resolve()
    xml = args.pure_bt_xml.resolve()
    output = args.output_root.resolve()
    if sha256(dll) != EXPECTED_DLL or sha256(xml) != EXPECTED_XML:
        raise ValueError("Pure BT Champion hash mismatch")
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("offline_gate_passed", False):
        raise ValueError("refusing Shadow for offline-gate-failed model")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Shadow evidence: {output}")
    output.mkdir(parents=True)
    suite = json.loads(args.suite.resolve().read_text(encoding="utf-8"))
    cases = select_cases(suite, args.case_count)
    started = perf_counter()
    rows = []
    for index, case in enumerate(cases, start=1):
        rows.append(
            run_case(
                case,
                output=output,
                bundle=bundle,
                dll=dll,
                xml=xml,
                episode_frames=args.episode_frames,
            )
        )
        print(json.dumps({"completed": index, "total": len(cases)}), flush=True)
    summary = summarize_shadow(rows)
    summary["wall_seconds"] = perf_counter() - started
    summary["model_sha256"] = metadata["model_sha256"]
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
