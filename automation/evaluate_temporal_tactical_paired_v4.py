from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DLL = "4C93B4C6719CB0423388D5FC721D356020A3A36CD5AD2C56B5C3CA795BFE18C9"
EXPECTED_XML = "D84C27B0B8BA22E1649AF2375BE0B83C762BC62EB5047BB590539B374F8271EE"
EPSILON = 1e-9
LARGE_REGRESSION = -1e-6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_frames(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if '"record_type":"frame"' in line
    ]


def outcome(result: dict[str, Any], frames: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "damage_dealt": float(sum(float(row["target_damage"]) for row in frames)),
        "damage_received": float(sum(float(row["ownship_damage"]) for row in frames)),
        "ownship_crash": bool(result.get("ownship_crash", False)),
        "target_crash": bool(result.get("target_crash", False)),
        "end_condition": result.get("end_condition", ""),
        "phase1_frames": sum(float(row["sim_time_s"]) <= 100.0 for row in frames),
        "phase2_frames": sum(100.0 < float(row["sim_time_s"]) <= 150.0 for row in frames),
        "phase3_frames": sum(float(row["sim_time_s"]) > 150.0 for row in frames),
    }


def bootstrap_ci(values: list[float], *, samples: int = 10000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return [0.0, 0.0]
    rng = np.random.default_rng(44001)
    means = np.mean(rng.choice(array, size=(samples, array.size), replace=True), axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize_pairs(
    rows: list[dict[str, Any]], *, stage: str, minimum_clean_pairs: int
) -> dict[str, Any]:
    clean = [row for row in rows if row["clean"]]
    deltas = np.asarray([row["damage_delta"] for row in clean], dtype=np.float64)
    intervened = [row for row in clean if row["nondefault_predictions"] > 0]
    intervention_precision = (
        float(np.mean([row["damage_delta"] > EPSILON for row in intervened]))
        if intervened
        else 0.0
    )
    ci = bootstrap_ci(deltas.tolist())
    common_gate = {
        "minimum_clean_pairs": len(clean) >= minimum_clean_pairs,
        "damage_mean_positive": bool(deltas.size and float(np.mean(deltas)) > EPSILON),
        "crash_not_increased": not any(
            row["hybrid_ownship_crash"] and not row["pure_ownship_crash"] for row in clean
        ),
        "invalid_zero": sum(row["invalid_frames"] for row in clean) == 0,
        "throttle_violation_zero": sum(row["throttle_violations"] for row in clean) == 0,
        "latency_over_limit_zero": sum(row["latency_over_166_7ms"] for row in clean) == 0,
    }
    if stage == "MICRO":
        stage_gate = {
            "no_meaningful_negative_tail": bool(
                deltas.size and not np.any(deltas < LARGE_REGRESSION)
            ),
            "intervention_precision_at_least_60pct": intervention_precision >= 0.60,
            "bootstrap_ci_lower_positive": ci[0] > 0.0,
        }
    else:
        stage_gate = {
            "damage_median_positive": bool(
                deltas.size and float(np.median(deltas)) > EPSILON
            ),
            "positive_pair_ratio_at_least_60pct": bool(
                deltas.size and float(np.mean(deltas > EPSILON)) >= 0.60
            ),
            "bootstrap_ci_lower_positive": ci[0] > 0.0,
        }
        if stage == "OFFICIAL_DEVELOPMENT":
            stage_gate.update(
                {
                    "opponent_coverage_at_least_3": len(
                        {row["opponent"] for row in clean}
                    )
                    >= 3,
                    "geometry_coverage_at_least_6": len(
                        {row["geometry"] for row in clean}
                    )
                    >= 6,
                    "phase1_2_3_flight_coverage": all(
                        sum(row[f"phase{phase}_frames"] for row in clean) > 0
                        for phase in (1, 2, 3)
                    ),
                }
            )
    gate = {**common_gate, **stage_gate}
    return {
        "schema_version": "temporal_tactical_paired_v4.v1",
        "stage": stage,
        "pairs": len(rows),
        "clean_pairs": len(clean),
        "target_crash_contaminated_pairs": len(rows) - len(clean),
        "damage_delta_mean": float(np.mean(deltas)) if deltas.size else 0.0,
        "damage_delta_median": float(np.median(deltas)) if deltas.size else 0.0,
        "positive_pair_ratio": float(np.mean(deltas > EPSILON)) if deltas.size else 0.0,
        "large_regression_pairs": int(np.sum(deltas < LARGE_REGRESSION)),
        "intervention_pairs": len(intervened),
        "intervention_precision": intervention_precision,
        "paired_bootstrap_95ci": ci,
        "opponents": sorted({row["opponent"] for row in clean}),
        "geometries": sorted({row["geometry"] for row in clean}),
        "phase_coverage": {
            "phase1_frames": sum(row["phase1_frames"] for row in clean),
            "phase2_frames": sum(row["phase2_frames"] for row in clean),
            "phase3_frames": sum(row["phase3_frames"] for row in clean),
        },
        "gate": gate,
        "decision": f"{stage}_GATE_PASSED" if all(gate.values()) else f"{stage}_GATE_FAILED",
        "rows": rows,
    }


def run_one(
    *,
    label: str,
    backend: str,
    run: Path,
    scenario: dict[str, Any],
    seed: int,
    bundle: Path,
    dll: Path,
    xml: Path,
    episode_frames: int,
    opponent: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run.mkdir(parents=True)
    scenario_path = run / "scenario.json"
    result_path = run / "result.json"
    telemetry_path = run / "telemetry.jsonl"
    scenario_path.write_text(json.dumps(scenario, indent=2, sort_keys=True), encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "run_local_dogfight.py"),
        "--ownship-backend",
        backend,
        "--target-backend",
        str(opponent["backend"]),
        "--ownship-bt-dll",
        str(dll),
        "--bt-rule-xml",
        str(xml),
        "--bt-rule-alias",
        "Rule_DCS_GDCC_0815.xml",
        "--bt-rule-alias-only",
        "--bt-turn-throttle-mode",
        "raw",
        "--scenario-file",
        str(scenario_path),
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
    if backend == "prefix_tactical":
        command.extend(
            [
                "--prefix-tactical-mode",
                "BT_DEFAULT",
                "--prefix-start-frame",
                "0",
                "--prefix-hold-frames",
                "0",
            ]
        )
    else:
        command.extend(["--ownship-bundle-dir", str(bundle)])
    if opponent["backend"] == "bt":
        command.extend(["--target-bt-dll", str(opponent["dll"])])
        if opponent.get("xml"):
            command.extend(["--target-bt-rule-xml", str(opponent["xml"])])
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=max(180.0, episode_frames / 5.0),
        check=False,
    )
    (run / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (run / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(f"paired {label} run failed rc={completed.returncode}")
    return json.loads(result_path.read_text(encoding="utf-8")), load_frames(telemetry_path)


def paired_case(
    case: dict[str, Any], *, output: Path, bundle: Path, dll: Path, xml: Path,
    episode_frames: int, opponent: dict[str, Any]
) -> dict[str, Any]:
    evaluation_id = f"{case['case_id']}__vs_{opponent['id']}"
    root = output / "runs" / evaluation_id
    pure_result, pure_frames = run_one(
        label="pure",
        backend="prefix_tactical",
        run=root / "pure",
        scenario=case["scenario"],
        seed=int(case["seed"]),
        bundle=bundle,
        dll=dll,
        xml=xml,
        episode_frames=episode_frames,
        opponent=opponent,
    )
    hybrid_result, hybrid_frames = run_one(
        label="hybrid",
        backend="temporal_tactical",
        run=root / "hybrid",
        scenario=case["scenario"],
        seed=int(case["seed"]),
        bundle=bundle,
        dll=dll,
        xml=xml,
        episode_frames=episode_frames,
        opponent=opponent,
    )
    pure = outcome(pure_result, pure_frames)
    hybrid = outcome(hybrid_result, hybrid_frames)
    provider = hybrid_result["ownship_provider_telemetry"]
    contaminated = bool(pure["target_crash"] or hybrid["target_crash"])
    return {
        "case_id": case["case_id"],
        "evaluation_id": evaluation_id,
        "geometry": case["geometry"],
        "opponent": opponent["id"],
        "seed": case["seed"],
        "clean": not contaminated,
        "target_crash_contaminated": contaminated,
        "damage_delta": hybrid["damage_dealt"] - pure["damage_dealt"],
        "net_health_margin_delta": (
            hybrid["damage_dealt"] - hybrid["damage_received"]
            - pure["damage_dealt"]
            + pure["damage_received"]
        ),
        "pure_ownship_crash": pure["ownship_crash"],
        "hybrid_ownship_crash": hybrid["ownship_crash"],
        "nondefault_predictions": provider["nondefault_predictions"],
        "applied_frames": provider["applied_frames"],
        "invalid_frames": provider["invalid_frames"],
        "throttle_violations": provider["throttle_violations"],
        "latency_over_166_7ms": provider["latency_ms"]["over_166_7ms"],
        "phase1_frames": hybrid["phase1_frames"],
        "phase2_frames": hybrid["phase2_frames"],
        "phase3_frames": hybrid["phase3_frames"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired evaluation for Temporal Tactical v4")
    parser.add_argument("--stage", choices=("MICRO", "SHORT_DEVELOPMENT", "OFFICIAL_DEVELOPMENT"), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--pure-bt-dll", type=Path, required=True)
    parser.add_argument("--pure-bt-xml", type=Path, required=True)
    parser.add_argument("--suite", type=Path, action="append", required=True)
    parser.add_argument("--prior-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, required=True)
    parser.add_argument("--episode-frames", type=int, required=True)
    parser.add_argument(
        "--opponent-config",
        type=Path,
        help="Optional JSON list of opponent specs: id/backend and BT dll/xml paths.",
    )
    return parser.parse_args()


def load_opponents(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return [{"id": "autopilot", "backend": "autopilot"}]
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    values = payload.get("opponents", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("opponent config must contain a non-empty list")
    opponents = []
    for raw in values:
        row = dict(raw)
        if row.get("backend") not in ("autopilot", "bt") or not row.get("id"):
            raise ValueError("each opponent requires id and backend=autopilot|bt")
        if row["backend"] == "bt":
            if not row.get("dll"):
                raise ValueError("BT opponent requires dll")
            row["dll"] = str(Path(row["dll"]).resolve())
            if row.get("xml"):
                row["xml"] = str(Path(row["xml"]).resolve())
        opponents.append(row)
    ids = [row["id"] for row in opponents]
    if len(ids) != len(set(ids)):
        raise ValueError("opponent ids must be unique")
    return opponents


def expand_cases(
    cases: list[dict[str, Any]], opponents: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    expanded = [(case, opponent) for opponent in opponents for case in cases]
    return sorted(
        expanded,
        key=lambda item: hashlib.sha256(
            f"{item[0]['geometry']}|{item[0]['case_id']}|{item[1]['id']}".encode()
        ).hexdigest(),
    )


def main() -> None:
    args = parse_args()
    expected_prior = {
        "MICRO": "SHADOW_GATE_PASSED",
        "SHORT_DEVELOPMENT": "MICRO_GATE_PASSED",
        "OFFICIAL_DEVELOPMENT": "SHORT_DEVELOPMENT_GATE_PASSED",
    }[args.stage]
    prior = json.loads(args.prior_summary.resolve().read_text(encoding="utf-8"))
    if prior.get("decision") != expected_prior:
        raise ValueError(f"refusing {args.stage}: prior gate is not {expected_prior}")
    bundle = args.bundle.resolve()
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("offline_gate_passed", False):
        raise ValueError("refusing paired evaluation for offline-gate-failed model")
    dll, xml = args.pure_bt_dll.resolve(), args.pure_bt_xml.resolve()
    if sha256(dll) != EXPECTED_DLL or sha256(xml) != EXPECTED_XML:
        raise ValueError("Pure BT Champion hash mismatch")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite paired evidence: {output}")
    cases = []
    for suite_path in args.suite:
        cases.extend(json.loads(suite_path.resolve().read_text(encoding="utf-8"))["cases"])
    unique = {case["case_id"]: case for case in cases}
    opponents = load_opponents(args.opponent_config)
    ordered = expand_cases(list(unique.values()), opponents)
    if len(ordered) < args.pair_count:
        raise ValueError("insufficient unique scenarios for requested paired evaluation")
    output.mkdir(parents=True)
    rows = []
    for index, (case, opponent) in enumerate(ordered[: args.pair_count], start=1):
        rows.append(
            paired_case(
                case,
                output=output,
                bundle=bundle,
                dll=dll,
                xml=xml,
                episode_frames=args.episode_frames,
                opponent=opponent,
            )
        )
        print(json.dumps({"completed": index, "total": args.pair_count}), flush=True)
    summary = summarize_pairs(rows, stage=args.stage, minimum_clean_pairs=args.pair_count)
    summary["model_sha256"] = metadata["model_sha256"]
    if args.stage == "OFFICIAL_DEVELOPMENT":
        seed_count = int(
            metadata.get("selected_oof_policy", {}).get("consistent_seed_count", 0)
        )
        summary["consistent_model_seed_count"] = seed_count
        summary["gate"]["minimum_2_model_seed_direction"] = seed_count >= 2
        summary["decision"] = (
            "OFFICIAL_DEVELOPMENT_GATE_PASSED"
            if all(summary["gate"].values())
            else "OFFICIAL_DEVELOPMENT_GATE_FAILED"
        )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
