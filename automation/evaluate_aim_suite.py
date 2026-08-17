"""고정된 scenario별 단일 seed 계약으로 aim residual 후보를 paired 평가한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from automation.evaluate_aim_residual import aggregate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_suite(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("suite cases는 비어 있지 않은 list여야 합니다")
    names = [str(case.get("name", "")) for case in cases]
    seeds = [int(case["seed"]) for case in cases]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("suite case name은 비어 있지 않고 고유해야 합니다")
    if len(set(seeds)) != len(seeds):
        raise ValueError("suite case seed label은 고유해야 합니다")
    return payload


def combine_results(
    suite: dict[str, Any], case_payloads: list[tuple[dict[str, Any], dict[str, Any]]]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    artifacts = []
    expected_controllers: set[str] | None = None
    bundle_hashes = set()
    for case, payload in case_payloads:
        case_records = payload["records"]
        controllers = {str(row["controller"]) for row in case_records}
        expected_controllers = controllers if expected_controllers is None else expected_controllers
        if controllers != expected_controllers:
            raise ValueError(f"case {case['name']}: controller 집합 불일치")
        for row in case_records:
            copied = dict(row)
            copied["source_variant_name"] = copied.get("variant_name")
            copied["variant_name"] = str(case["name"])
            copied["suite_case"] = str(case["name"])
            copied["suite_seed_label"] = int(case["seed"])
            records.append(copied)
        bundle_hashes.add(payload["preflight"]["bundle_weights_sha256"])
        artifacts.append(
            {
                "case": case["name"],
                "seed": int(case["seed"]),
                "scenario": case["scenario"],
                "evaluation_sha256": case.get("evaluation_sha256"),
            }
        )
    if len(bundle_hashes) != 1:
        raise ValueError(f"case별 bundle hash 불일치: {sorted(bundle_hashes)}")
    return {
        "suite": suite,
        "contract": {
            "cases": len(case_payloads),
            "deterministic_geometry_samples": len(case_payloads),
            "stochastic_independent_samples": 0,
            "bundle_weights_sha256": next(iter(bundle_hashes)),
        },
        "case_artifacts": artifacts,
        "records": records,
        "summary": aggregate(records),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--suite", type=Path, required=True)
    value.add_argument("--bundle", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--ownship-bt-dll", required=True)
    value.add_argument("--target-backend", choices=("autopilot", "bt"), default="autopilot")
    value.add_argument("--target-bt-dll", required=True)
    value.add_argument("--bt-rule-xml", required=True)
    value.add_argument("--bt-rule-alias", action="append", default=[])
    value.add_argument("--scale", type=float, default=0.125)
    value.add_argument(
        "--residual-axis-mask",
        choices=("roll", "pitch", "yaw", "pitch_yaw", "roll_pitch_yaw"),
        default="roll_pitch_yaw",
    )
    value.add_argument(
        "--gate-kind",
        choices=("aim", "offensive", "combined", "rear120"),
        default="aim",
    )
    value.add_argument("--max-engage-time", type=float, default=30.0)
    value.add_argument("--episode-step-limit", type=int, default=1800)
    value.add_argument("--timeout-seconds", type=float, default=120.0)
    value.add_argument("--resume", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    suite = load_suite(args.suite)
    args.output.mkdir(parents=True, exist_ok=True)
    evaluator = Path(__file__).with_name("evaluate_aim_residual.py")
    payloads = []
    for index, case in enumerate(suite["cases"], start=1):
        case_output = args.output / f"case_{index:02d}_{case['name']}"
        command = [
            sys.executable, str(evaluator),
            "--bundle", str(args.bundle),
            "--output", str(case_output),
            "--scenario", str(case["scenario"]),
            "--seeds", str(case["seed"]),
            "--scales", str(args.scale),
            "--gate-kind", args.gate_kind,
            "--composition-mode", "saturation_aware",
            "--residual-axis-mask", args.residual_axis_mask,
            "--ownship-bt-dll", args.ownship_bt_dll,
            "--target-backend", args.target_backend,
            "--target-bt-dll", args.target_bt_dll,
            "--bt-rule-xml", args.bt_rule_xml,
            "--max-engage-time", str(args.max_engage_time),
            "--episode-step-limit", str(args.episode_step_limit),
            "--timeout-seconds", str(args.timeout_seconds),
            "--rl-action-repeat", "6",
            "--quiet",
        ]
        for alias in args.bt_rule_alias:
            command.extend(("--bt-rule-alias", alias))
        if args.resume:
            command.append("--resume")
        print(f"[{index}/{len(suite['cases'])}] {case['name']} seed={case['seed']}", flush=True)
        subprocess.run(command, check=True)
        evaluation = case_output / "evaluation.json"
        enriched_case = {**case, "evaluation_sha256": sha256(evaluation)}
        payloads.append((enriched_case, json.loads(evaluation.read_text(encoding="utf-8"))))
    combined = combine_results(suite, payloads)
    output = args.output / "evaluation.json"
    output.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
