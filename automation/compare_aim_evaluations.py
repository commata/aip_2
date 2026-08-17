"""여러 aim residual 평가의 Pure baseline 동일성과 후보 paired 효과를 비교한다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from automation.analyze_aim_candidates import analyze


VOLATILE_FIELDS = {"wall_seconds", "returncode", "resumed", "run_id"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def stable_record(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in VOLATILE_FIELDS}


def pure_rows(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = {
        int(row["seed"]): row
        for row in payload["records"]
        if row.get("controller") == "pure_0815"
    }
    if not rows:
        raise ValueError("Pure 0815 record가 없습니다")
    return rows


def telemetry_hash(evaluation: Path | None, row: dict[str, Any]) -> str | None:
    if evaluation is None or not row.get("run_id"):
        return None
    path = evaluation.parent / "telemetry" / f"{row['run_id']}.jsonl"
    return sha256(path) if path.is_file() else None


def compare(
    inputs: list[tuple[str, dict[str, Any], Path | None]],
    *,
    controller: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if len(inputs) < 2:
        raise ValueError("비교에는 두 개 이상의 평가가 필요합니다")
    reference_name, reference_payload, reference_path = inputs[0]
    reference = pure_rows(reference_payload)
    baseline: dict[str, Any] = {
        "reference": reference_name,
        "seed_set": sorted(reference),
        "candidates": {},
    }
    all_records_equal = True
    all_available_telemetry_equal = True
    for name, payload, path in inputs:
        candidate = pure_rows(payload)
        seed_set_equal = set(candidate) == set(reference)
        mismatches = []
        telemetry = []
        for seed in sorted(set(reference) & set(candidate)):
            left, right = stable_record(reference[seed]), stable_record(candidate[seed])
            fields = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
            if fields:
                mismatches.append({"seed": seed, "fields": fields})
            left_hash = telemetry_hash(reference_path, reference[seed])
            right_hash = telemetry_hash(path, candidate[seed])
            equal = left_hash == right_hash if left_hash is not None and right_hash is not None else None
            telemetry.append(
                {
                    "seed": seed,
                    "reference_sha256": left_hash,
                    "candidate_sha256": right_hash,
                    "equal": equal,
                }
            )
            if equal is False:
                all_available_telemetry_equal = False
        records_equal = seed_set_equal and not mismatches
        all_records_equal = all_records_equal and records_equal
        baseline["candidates"][name] = {
            "seed_set_equal": seed_set_equal,
            "record_values_equal": records_equal,
            "mismatches": mismatches,
            "telemetry": telemetry,
        }
    baseline["all_record_values_equal"] = all_records_equal
    baseline["all_available_telemetry_equal"] = all_available_telemetry_equal

    candidates = {}
    for index, (name, payload, path) in enumerate(inputs):
        result = analyze(
            payload,
            controller,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index * 100,
        )
        candidates[name] = {
            "evaluation": str(path.resolve()) if path is not None else None,
            "pairs": result["pairs"],
            "crash_regressions": result["crash_regressions"],
            "metrics": result["metrics"],
            "authority": result["authority"],
            "per_variant": result["per_variant"],
            "reward_chain_correlations": result["reward_chain_correlations"],
        }
    return {
        "controller": controller,
        "baseline_consistency": baseline,
        "candidates": candidates,
        "bootstrap": {"samples": bootstrap_samples, "seed": bootstrap_seed},
    }


def parse_input(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"evaluation은 NAME=PATH 형식이어야 합니다: {value}")
    return name, Path(path)


def fmt(value: Any) -> str:
    return "자료 없음" if value is None else f"{float(value):+.6f}"


def report(payload: dict[str, Any]) -> str:
    baseline = payload["baseline_consistency"]
    lines = [
        "# Aim residual 후보 교차 비교",
        "",
        f"- Pure record 전체 동일: `{baseline['all_record_values_equal']}`",
        f"- 사용 가능한 Pure telemetry SHA 전체 동일: `{baseline['all_available_telemetry_equal']}`",
        "",
        "| 후보 | pairs | crash 회귀 | Damage Δ | LOS Δ | LOS-rate Δ | Cone Δ | First Damage Δ | saturation | roll 적용/요청 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in payload["candidates"].items():
        metrics = result["metrics"]
        authority = result["authority"]
        lines.append(
            f"| {name} | {result['pairs']} | {result['crash_regressions']} | "
            f"{fmt(metrics['damage_dealt']['mean'])} | {fmt(metrics['mean_los_deg']['mean'])} | "
            f"{fmt(metrics['los_rate_rms_deg_s']['mean'])} | "
            f"{fmt(metrics['damage_cone_time_s']['mean'])} | "
            f"{fmt(metrics['time_to_first_damage_s']['mean'])} | "
            f"{fmt(metrics['action_saturated_ratio']['mean'])} | "
            f"{fmt(authority['roll_applied_to_requested_ratio'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--controller", default="hybrid_0.125")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=817)
    args = parser.parse_args()
    inputs = []
    for value in args.evaluation:
        name, path = parse_input(value)
        inputs.append((name, json.loads(path.read_text(encoding="utf-8")), path))
    result = compare(
        inputs,
        controller=args.controller,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = args.report or args.output.with_suffix(".md")
    report_path.write_text(report(result), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
