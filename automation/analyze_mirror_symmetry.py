"""Exact mirror 평가의 episode 지표와 시간 정렬 제어 대칭 오차를 분석한다."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


PAIR_SPECS = (
    ("lateral", "lateral_left", "lateral_right", (-1.0, 1.0, -1.0, 1.0)),
    ("lateral", "crossing_left", "crossing_right", (-1.0, 1.0, -1.0, 1.0)),
    ("vertical", "vertical_high", "vertical_low", (-1.0, -1.0, 1.0, 1.0)),
)

GEOMETRY_SIGNS = {
    "lateral": {
        "aim_azimuth_deg": -1.0,
        "aim_elevation_deg": 1.0,
        "los_azimuth_rate_deg_s": -1.0,
        "los_elevation_rate_deg_s": 1.0,
        "distance_m": 1.0,
        "closing_rate_m_s": 1.0,
        "ata_deg": 1.0,
        "target_ata_deg": 1.0,
    },
    "vertical": {
        "aim_azimuth_deg": 1.0,
        "aim_elevation_deg": -1.0,
        "los_azimuth_rate_deg_s": 1.0,
        "los_elevation_rate_deg_s": -1.0,
        "distance_m": 1.0,
        "closing_rate_m_s": 1.0,
        "ata_deg": 1.0,
        "target_ata_deg": 1.0,
    },
}

EPISODE_METRICS = (
    "time_to_first_damage_s",
    "mean_los_deg",
    "los_rate_rms_deg_s",
    "damage_cone_time_s",
    "damage_dealt",
    "min_altitude_m",
    "bt_roll_saturation_ratio",
    "bt_pitch_saturation_ratio",
    "bt_yaw_saturation_ratio",
    "roll_applied_to_requested_ratio",
    "pitch_applied_to_requested_ratio",
    "yaw_applied_to_requested_ratio",
)


def load_frames(path: Path) -> list[dict[str, Any]]:
    return [
        row
        for line in path.read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line)).get("record_type") == "frame"
    ]


def _rmse(values: list[float]) -> float | None:
    return math.sqrt(fmean(value * value for value in values)) if values else None


def _action(frame: dict[str, Any], kind: str) -> list[float] | None:
    hybrid = frame.get("hybrid", {}) or {}
    if kind == "final_action":
        return frame.get("ownship_action")
    if kind == "bt_action":
        return hybrid.get("bt_action") or frame.get("ownship_action")
    if kind == "raw_residual_action":
        return hybrid.get("raw_residual_action")
    if kind == "applied_rl_correction":
        return hybrid.get("applied_rl_correction")
    raise ValueError(kind)


def compare_frames(
    first: list[dict[str, Any]],
    mirrored: list[dict[str, Any]],
    *,
    axis: str,
    action_signs: tuple[float, ...],
    horizon_s: float | None = None,
) -> dict[str, Any]:
    count = min(len(first), len(mirrored))
    pairs = []
    for index in range(count):
        left = first[index]
        right = mirrored[index]
        if horizon_s is not None and float(left["sim_time_s"]) > horizon_s:
            break
        pairs.append((left, right))

    geometry = {}
    for key, sign in GEOMETRY_SIGNS[axis].items():
        errors = [float(right[key]) - sign * float(left[key]) for left, right in pairs]
        geometry[key] = {"rmse": _rmse(errors), "max_abs": max(map(abs, errors), default=None)}

    actions = {}
    for kind in ("bt_action", "raw_residual_action", "applied_rl_correction", "final_action"):
        axis_errors = [[] for _ in range(4)]
        for left, right in pairs:
            left_action = _action(left, kind)
            right_action = _action(right, kind)
            if left_action is None or right_action is None:
                continue
            for index in range(4):
                axis_errors[index].append(
                    float(right_action[index]) - action_signs[index] * float(left_action[index])
                )
        actions[kind] = {
            name: _rmse(axis_errors[index])
            for index, name in enumerate(("roll", "pitch", "yaw", "throttle"))
        }
    return {"matched_frames": len(pairs), "geometry": geometry, "actions": actions}


def analyze(root: Path, prefix: str, seed: int) -> dict[str, Any]:
    evaluations = {}
    telemetry = {}
    for _, first, second, _ in PAIR_SPECS:
        for scenario in (first, second):
            directory = root / f"{prefix}{scenario}"
            payload = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
            evaluations[scenario] = {
                row["controller"]: row for row in payload["records"] if row["seed"] == seed
            }
            telemetry[scenario] = {
                controller: load_frames(
                    directory / "telemetry" / f"seed{seed}_{controller}_aim.jsonl"
                )
                for controller in evaluations[scenario]
            }

    pairs = []
    for axis, first, second, action_signs in PAIR_SPECS:
        controller_results = {}
        controllers = sorted(set(evaluations[first]) & set(evaluations[second]))
        for controller in controllers:
            first_record = evaluations[first][controller]
            second_record = evaluations[second][controller]
            controller_results[controller] = {
                "episode_gap_second_minus_first": {
                    metric: (
                        None
                        if first_record.get(metric) is None or second_record.get(metric) is None
                        else float(second_record[metric]) - float(first_record[metric])
                    )
                    for metric in EPISODE_METRICS
                },
                "first_1s": compare_frames(
                    telemetry[first][controller],
                    telemetry[second][controller],
                    axis=axis,
                    action_signs=action_signs,
                    horizon_s=1.0,
                ),
                "full_overlap": compare_frames(
                    telemetry[first][controller],
                    telemetry[second][controller],
                    axis=axis,
                    action_signs=action_signs,
                ),
            }
        pairs.append(
            {
                "axis": axis,
                "first": first,
                "second": second,
                "controllers": controller_results,
                "interpretation_limit": (
                    "상하 pair는 중력이 반전되지 않으므로 순간 운동학 sign 진단이며 "
                    "전체 trajectory 동역학 대칭을 요구하지 않는다."
                    if axis == "vertical"
                    else "좌우 pair는 정확한 초기 운동학 mirror이며 이후 오차는 BT/동역학/정책 차이를 포함한다."
                ),
            }
        )
    return {"seed": seed, "prefix": prefix, "pairs": pairs}


def _fmt(value: Any) -> str:
    return "자료 없음" if value is None else f"{float(value):+.6f}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Mirror 대칭 진단",
        "",
        f"- seed: `{payload['seed']}`",
        "- episode gap 정의: 두 번째 scenario - 첫 번째 scenario",
        "- 시간 정렬 오차: mirrored - sign × original의 RMSE",
        "",
    ]
    for pair in payload["pairs"]:
        lines += [
            f"## {pair['first']} ↔ {pair['second']}",
            "",
            f"- {pair['interpretation_limit']}",
            "",
            "| Controller | First Damage gap(s) | LOS gap(°) | LOS-rate gap(°/s) | Cone gap(s) | Damage gap | BT R/P/Y 포화 gap |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for controller, result in pair["controllers"].items():
            gap = result["episode_gap_second_minus_first"]
            lines.append(
                f"| {controller} | {_fmt(gap['time_to_first_damage_s'])} | "
                f"{_fmt(gap['mean_los_deg'])} | {_fmt(gap['los_rate_rms_deg_s'])} | "
                f"{_fmt(gap['damage_cone_time_s'])} | {_fmt(gap['damage_dealt'])} | "
                f"{_fmt(gap['bt_roll_saturation_ratio'])} / "
                f"{_fmt(gap['bt_pitch_saturation_ratio'])} / "
                f"{_fmt(gap['bt_yaw_saturation_ratio'])} |"
            )
        lines += ["", "### 최초 1초 제어 mirror RMSE", "", "| Controller | Source | Roll | Pitch | Yaw | Throttle |", "|---|---|---:|---:|---:|---:|"]
        for controller, result in pair["controllers"].items():
            for source, values in result["first_1s"]["actions"].items():
                if all(value is None for value in values.values()):
                    continue
                lines.append(
                    f"| {controller} | {source} | {_fmt(values['roll'])} | "
                    f"{_fmt(values['pitch'])} | {_fmt(values['yaw'])} | {_fmt(values['throttle'])} |"
                )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.root, args.prefix, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.output.with_suffix(".md"), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

