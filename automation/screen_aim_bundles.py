#!/usr/bin/env python
"""Aim residual bundle들을 동일 exact scenario로 순차 screening한다."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


METRICS = (
    "damage_dealt",
    "mean_los_deg",
    "los_rate_rms_deg_s",
    "damage_cone_time_s",
    "time_to_first_damage_s",
    "min_altitude_m",
    "action_saturated_ratio",
    "roll_applied_to_requested_ratio",
    "pitch_applied_to_requested_ratio",
    "yaw_applied_to_requested_ratio",
)


def discover_bundles(bundle_root: Path, include_final: bool) -> list[tuple[str, Path]]:
    bundles = [
        (path.name.removeprefix("bundle_"), path)
        for path in bundle_root.glob("bundle_*")
        if path.is_dir() and (path / "metadata.json").is_file()
    ]
    bundles.sort(key=lambda item: int(item[0]))
    if include_final and (bundle_root / "metadata.json").is_file():
        bundles.append(("final", bundle_root))
    return bundles


def paired_delta(payload: dict[str, Any], controller: str) -> dict[str, Any]:
    return payload["summary"]["paired"][controller]["delta_hybrid_minus_pure"]


def controller_summary(payload: dict[str, Any], controller: str) -> dict[str, Any]:
    return payload["summary"]["controllers"][controller]


def finite(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result.append(float(value))
    return result


def summarize_bundle(
    tag: str,
    bundle: Path,
    scenario_payloads: dict[str, dict[str, Any]],
    controller: str,
) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    for name, payload in scenario_payloads.items():
        delta = paired_delta(payload, controller)
        hybrid = controller_summary(payload, controller)
        pure = controller_summary(payload, "pure_0815")
        scenarios[name] = {
            "delta": {metric: delta.get(metric) for metric in METRICS},
            "hybrid_ownship_crashes": hybrid.get("ownship_crashes", 0),
            "pure_ownship_crashes": pure.get("ownship_crashes", 0),
            "hybrid_gate_active_ratio": hybrid.get("gate_active_ratio"),
            "hybrid_final_roll_saturation_ratio": hybrid.get("final_roll_saturation_ratio"),
            "hybrid_final_pitch_saturation_ratio": hybrid.get("final_pitch_saturation_ratio"),
            "hybrid_final_yaw_saturation_ratio": hybrid.get("final_yaw_saturation_ratio"),
        }

    def metric_values(metric: str) -> list[float]:
        return finite(item["delta"].get(metric) for item in scenarios.values())

    damage = metric_values("damage_dealt")
    los = metric_values("mean_los_deg")
    los_rate = metric_values("los_rate_rms_deg_s")
    cone = metric_values("damage_cone_time_s")
    first_damage = metric_values("time_to_first_damage_s")
    saturation = metric_values("action_saturated_ratio")
    crash_regressions = sum(
        max(0, int(item["hybrid_ownship_crashes"]) - int(item["pure_ownship_crashes"]))
        for item in scenarios.values()
    )
    aggregate = {
        "damage_delta_mean": sum(damage) / len(damage) if damage else None,
        "damage_delta_worst": min(damage) if damage else None,
        "los_delta_mean_deg": sum(los) / len(los) if los else None,
        "los_delta_worst_deg": max(los) if los else None,
        "los_rate_delta_mean_deg_s": sum(los_rate) / len(los_rate) if los_rate else None,
        "cone_delta_mean_s": sum(cone) / len(cone) if cone else None,
        "first_damage_delta_mean_s": (
            sum(first_damage) / len(first_damage) if first_damage else None
        ),
        "action_saturation_delta_max": max(saturation) if saturation else None,
        "ownship_crash_regressions": crash_regressions,
    }
    if len(damage) == 2:
        aggregate["left_right_damage_gap"] = abs(damage[0] - damage[1])
    return {
        "tag": tag,
        "bundle": str(bundle.resolve()),
        "bundle_weights_sha256": next(iter(scenario_payloads.values()))["preflight"][
            "bundle_weights_sha256"
        ],
        "scenarios": scenarios,
        "aggregate": aggregate,
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Aim residual periodic bundle screening",
        "",
        "이 표는 후보 제거용 exact mirror screening이다. 단일 seed 결과이므로 성능 승격 근거가 아니다.",
        "",
        "| bundle | Damage 평균 | Damage 최악 | 좌우 Damage gap | LOS 평균 Δ | LOS-rate 평균 Δ | Cone 평균 Δ | First Damage 평균 Δ | saturation Δ 최대 | crash 회귀 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(value: Any, digits: int = 5) -> str:
        return "-" if value is None else f"{float(value):.{digits}f}"

    for item in payload["bundles"]:
        a = item["aggregate"]
        lines.append(
            "| {tag} | {damage} | {worst} | {gap} | {los} | {rate} | {cone} | {first} | {sat} | {crash} |".format(
                tag=item["tag"],
                damage=fmt(a["damage_delta_mean"]),
                worst=fmt(a["damage_delta_worst"]),
                gap=fmt(a.get("left_right_damage_gap")),
                los=fmt(a["los_delta_mean_deg"]),
                rate=fmt(a["los_rate_delta_mean_deg_s"]),
                cone=fmt(a["cone_delta_mean_s"]),
                first=fmt(a["first_damage_delta_mean_s"]),
                sat=fmt(a["action_saturation_delta_max"]),
                crash=a["ownship_crash_regressions"],
            )
        )
    return "\n".join(lines) + "\n"


def parse_scenarios(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"scenario는 NAME=PATH 형식이어야 합니다: {value}")
        result.append((name, Path(path)))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--scenario", action="append", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--ownship-bt-dll", required=True)
    parser.add_argument("--bt-rule-xml", required=True)
    parser.add_argument("--bt-rule-alias", action="append", default=[])
    parser.add_argument("--target-bt-dll", required=True)
    parser.add_argument("--include-final", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-engage-time", type=float, default=30.0)
    parser.add_argument("--episode-step-limit", type=int, default=1800)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scenarios = parse_scenarios(args.scenario)
    bundles = discover_bundles(args.bundle_root, args.include_final)
    if not bundles:
        raise FileNotFoundError(f"bundle을 찾지 못했습니다: {args.bundle_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    evaluator = Path(__file__).with_name("evaluate_aim_residual.py")
    controller = "hybrid_0.125"
    for index, (tag, bundle) in enumerate(bundles, start=1):
        payloads: dict[str, dict[str, Any]] = {}
        for scenario_name, scenario_path in scenarios:
            output = args.output_root / f"bundle_{tag}_{scenario_name}"
            command = [
                sys.executable,
                str(evaluator),
                "--bundle",
                str(bundle),
                "--output",
                str(output),
                "--scenario",
                str(scenario_path),
                "--seeds",
                *(str(seed) for seed in args.seeds),
                "--scales",
                "0.125",
                "--gate-kind",
                "aim",
                "--composition-mode",
                "saturation_aware",
                "--ownship-bt-dll",
                args.ownship_bt_dll,
                "--target-backend",
                "autopilot",
                "--target-bt-dll",
                args.target_bt_dll,
                "--bt-rule-xml",
                args.bt_rule_xml,
                "--max-engage-time",
                str(args.max_engage_time),
                "--episode-step-limit",
                str(args.episode_step_limit),
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--rl-action-repeat",
                "6",
                "--quiet",
            ]
            for alias in args.bt_rule_alias:
                command.extend(("--bt-rule-alias", alias))
            if args.resume:
                command.append("--resume")
            print(f"[{index}/{len(bundles)}] {tag} {scenario_name}", flush=True)
            subprocess.run(command, check=True)
            payloads[scenario_name] = json.loads(
                (output / "evaluation.json").read_text(encoding="utf-8")
            )
        results.append(summarize_bundle(tag, bundle, payloads, controller))
        current = {
            "contract": {
                "seeds": args.seeds,
                "scenarios": [name for name, _ in scenarios],
                "scale": 0.125,
                "gate": "aim",
                "composition": "saturation_aware",
                "purpose": "development screening only",
            },
            "bundles": results,
        }
        args.summary.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        args.report.write_text(render_report(current), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
