from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from automation.target_profiles import load_target_profile


OWNSHIP_PROFILE_ID = "bt_0815"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="동일 seed에서 Proxy Target BT의 simulator-rate 행동을 비교"
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["bt_0815", "bt_aip2", "bt_aip3"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2201, 2206)))
    parser.add_argument("--max-engage-time", type=float, default=30.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_frames(path: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("record_type") == "frame":
                frames.append(record)
    return frames


def summarize_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    if not frames:
        raise ValueError("target telemetry contains no frame records")
    target = [frame["target"] for frame in frames]
    actions = [frame.get("target_action", [0.0] * 4) for frame in frames]
    headings = [float(item["attitude_deg"][2]) for item in target]
    unwrapped = _unwrap_degrees(headings)
    damages = [float(frame.get("target_damage", 0.0)) for frame in frames]
    first_damage = next(
        (float(frame["sim_time_s"]) for frame, damage in zip(frames, damages) if damage > 0),
        None,
    )
    heading_delta = unwrapped[-1] - unwrapped[0]
    return {
        "frames": len(frames),
        "duration_s": float(frames[-1]["sim_time_s"]),
        "heading_delta_deg": heading_delta,
        "turn_direction": "right" if heading_delta > 1.0 else "left" if heading_delta < -1.0 else "neutral",
        "roll_abs_mean_deg": statistics.fmean(
            abs(float(item["attitude_deg"][0])) for item in target
        ),
        "pitch_abs_mean_deg": statistics.fmean(
            abs(float(item["attitude_deg"][1])) for item in target
        ),
        "speed_mean_kcas": statistics.fmean(float(item["speed_kcas"]) for item in target),
        "speed_min_kcas": min(float(item["speed_kcas"]) for item in target),
        "altitude_mean_m": statistics.fmean(float(item["altitude_m"]) for item in target),
        "altitude_min_m": min(float(item["altitude_m"]) for item in target),
        "ata_mean_deg": statistics.fmean(float(frame["ata_deg"]) for frame in frames),
        "target_ata_mean_deg": statistics.fmean(
            float(frame["target_ata_deg"]) for frame in frames
        ),
        "distance_mean_m": statistics.fmean(float(frame["distance_m"]) for frame in frames),
        "target_action_abs_mean": [
            statistics.fmean(abs(float(action[axis])) for action in actions)
            for axis in range(4)
        ],
        "first_target_damage_s": first_damage,
        "final_target_damage": damages[-1],
    }


def compare_frames(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, float | int]:
    count = min(len(left), len(right))
    if count == 0:
        raise ValueError("cannot compare empty target trajectories")
    position_sq: list[float] = []
    attitude_sq: list[float] = []
    action_sq: list[float] = []
    speed_sq: list[float] = []
    altitude_sq: list[float] = []
    for lframe, rframe in zip(left[:count], right[:count]):
        ltarget, rtarget = lframe["target"], rframe["target"]
        position_sq.append(
            sum(
                (float(lvalue) - float(rvalue)) ** 2
                for lvalue, rvalue in zip(
                    ltarget["position_ned_m"], rtarget["position_ned_m"]
                )
            )
        )
        lroll, lpitch, lheading = map(float, ltarget["attitude_deg"])
        rroll, rpitch, rheading = map(float, rtarget["attitude_deg"])
        attitude_sq.append(
            (lroll - rroll) ** 2
            + (lpitch - rpitch) ** 2
            + _angle_delta_deg(lheading, rheading) ** 2
        )
        action_sq.append(
            sum(
                (float(lvalue) - float(rvalue)) ** 2
                for lvalue, rvalue in zip(
                    lframe.get("target_action", [0.0] * 4),
                    rframe.get("target_action", [0.0] * 4),
                )
            )
        )
        speed_sq.append(
            (float(ltarget["speed_kcas"]) - float(rtarget["speed_kcas"])) ** 2
        )
        altitude_sq.append(
            (float(ltarget["altitude_m"]) - float(rtarget["altitude_m"])) ** 2
        )
    return {
        "aligned_frames": count,
        "position_rmse_m": math.sqrt(statistics.fmean(position_sq)),
        "attitude_rmse_deg": math.sqrt(statistics.fmean(attitude_sq)),
        "action_rmse": math.sqrt(statistics.fmean(action_sq)),
        "speed_rmse_kcas": math.sqrt(statistics.fmean(speed_sq)),
        "altitude_rmse_m": math.sqrt(statistics.fmean(altitude_sq)),
    }


def _unwrap_degrees(values: list[float]) -> list[float]:
    result = [values[0]]
    for value in values[1:]:
        result.append(result[-1] + _angle_delta_deg(value, result[-1]))
    return result


def _angle_delta_deg(value: float, reference: float) -> float:
    return (value - reference + 180.0) % 360.0 - 180.0


def run_profile(
    profile: dict[str, Any],
    ownship: dict[str, Any],
    seed: int,
    output: Path,
    max_engage_time: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_id = f"{profile['profile_id']}_s{seed}"
    summary_path = output / "summaries" / f"{run_id}.json"
    telemetry_path = output / "telemetry" / f"{run_id}.jsonl"
    stdout_path = output / "raw" / f"{run_id}.stdout.txt"
    stderr_path = output / "raw" / f"{run_id}.stderr.txt"
    aliases = list(dict.fromkeys(ownship.get("rule_aliases", []) + profile.get("rule_aliases", [])))
    command = [
        sys.executable,
        str(ROOT / "run_local_dogfight.py"),
        "--ownship-backend",
        "bt",
        "--target-backend",
        "bt",
        "--ownship-bt-dll",
        ownship["dll"]["resolved_path"],
        "--target-bt-dll",
        profile["dll"]["resolved_path"],
        "--bt-rule-xml",
        ownship["xml"]["resolved_path"],
        "--bt-rule-alias-only",
        "--seed",
        str(seed),
        "--max-engage-time",
        str(max_engage_time),
        "--episode-step-limit",
        str(math.ceil(max_engage_time * 60)),
        "--result-json",
        str(summary_path),
        "--telemetry-jsonl",
        str(telemetry_path),
    ]
    for alias in aliases:
        command.extend(("--bt-rule-alias", alias))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=max(120.0, max_engage_time * 8.0),
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"proxy target run failed ({run_id}, rc={completed.returncode}): "
            f"{completed.stderr[-1000:]}"
        )
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    frames = load_frames(telemetry_path)
    row = {
        "profile_id": profile["profile_id"],
        "behavior_cluster_declared": profile["behavior_cluster"],
        "seed": seed,
        "end_condition": result.get("end_condition"),
        "outcome": result.get("outcome"),
        "ownship_health": result.get("ownship_health"),
        "target_health": result.get("target_health"),
        **summarize_frames(frames),
    }
    return row, frames


def main() -> int:
    args = parse_args()
    output = Path(args.output).resolve()
    for name in ("summaries", "telemetry", "raw"):
        (output / name).mkdir(parents=True, exist_ok=True)
    ownship = load_target_profile(OWNSHIP_PROFILE_ID)
    profiles = [load_target_profile(profile_id) for profile_id in args.profiles]
    if any(profile["backend_type"] != "behavior_tree" for profile in profiles):
        raise ValueError("evaluate_proxy_targets currently accepts behavior_tree profiles")

    rows: list[dict[str, Any]] = []
    trajectories: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for seed in args.seeds:
        for profile in profiles:
            print(f"[proxy-target] profile={profile['profile_id']} seed={seed}")
            row, frames = run_profile(
                profile, ownship, seed, output, args.max_engage_time
            )
            rows.append(row)
            trajectories[(profile["profile_id"], seed)] = frames

    pairwise: list[dict[str, Any]] = []
    for index, left in enumerate(profiles):
        for right in profiles[index + 1 :]:
            comparisons = [
                compare_frames(
                    trajectories[(left["profile_id"], seed)],
                    trajectories[(right["profile_id"], seed)],
                )
                for seed in args.seeds
            ]
            pairwise.append(
                {
                    "left": left["profile_id"],
                    "right": right["profile_id"],
                    "seeds": args.seeds,
                    **{
                        key: statistics.fmean(float(item[key]) for item in comparisons)
                        for key in (
                            "aligned_frames",
                            "position_rmse_m",
                            "attitude_rmse_deg",
                            "action_rmse",
                            "speed_rmse_kcas",
                            "altitude_rmse_m",
                        )
                    },
                }
            )
    payload = {
        "profiles": [profile["profile_id"] for profile in profiles],
        "seeds": args.seeds,
        "max_engage_time": args.max_engage_time,
        "runs": rows,
        "pairwise": pairwise,
    }
    (output / "evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(pairwise, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

