"""장시간 SAC progress CSV의 유효 시간과 curriculum 실행 빈도를 검증한다."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def analyze_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("training log has no rows")
    fraction_prefixes = (
        "target_profile_fraction_",
        "aim_variant_fraction_",
    )
    fraction_columns = {
        prefix: sorted(
            key for key in rows[0] if key.startswith(prefix)
        )
        for prefix in fraction_prefixes
    }
    weighted_counts = {
        prefix: {column.removeprefix(prefix): 0.0 for column in columns}
        for prefix, columns in fraction_columns.items()
    }
    # RLlib restores lifetime counters from native checkpoints.  The first row
    # is therefore the run-local baseline, not newly completed evidence.
    episode_lifetime_offset = finite(rows[0].get("episodes")) or 0.0
    previous_episodes = episode_lifetime_offset
    counted_episodes = {prefix: 0.0 for prefix in fraction_prefixes}
    crash_episodes = 0.0
    completed_episode_delta = 0.0
    finite_rewards: list[float] = []
    critical_nan_rows: list[int] = []

    for index, row in enumerate(rows):
        sampled = finite(row.get("sampled_steps"))
        learner = finite(row.get("learner_steps"))
        effective = finite(row.get("effective_learner_time_s"))
        if sampled is None or (index > 0 and learner is None) or effective is None:
            critical_nan_rows.append(index)
        reward = finite(row.get("reward_mean"))
        if reward is not None:
            finite_rewards.append(reward)
        episodes = finite(row.get("episodes"))
        if episodes is None:
            continue
        episode_delta = max(0.0, episodes - previous_episodes)
        previous_episodes = max(previous_episodes, episodes)
        if episode_delta <= 0.0:
            continue
        completed_episode_delta += episode_delta
        crash_rate = finite(row.get("crash_rate"))
        if crash_rate is not None:
            crash_episodes += episode_delta * crash_rate
        for prefix, columns in fraction_columns.items():
            values = [finite(row.get(column)) for column in columns]
            if values and all(value is not None for value in values):
                for column, value in zip(columns, values):
                    weighted_counts[prefix][column.removeprefix(prefix)] += (
                        episode_delta * float(value)
                    )
                counted_episodes[prefix] += episode_delta

    last = rows[-1]
    result: dict[str, Any] = {
        "rows": len(rows),
        "last_iteration": finite(last.get("iter")),
        "sampled_steps": finite(last.get("sampled_steps")),
        "learner_steps": finite(last.get("learner_steps")),
        "episodes": finite(last.get("episodes")),
        "episode_lifetime_offset": episode_lifetime_offset,
        "effective_learner_time_s": finite(last.get("effective_learner_time_s")),
        "completed_episode_delta": completed_episode_delta,
        "estimated_crash_episodes": crash_episodes,
        "critical_nan_rows": critical_nan_rows,
        "finite_reward_rows": len(finite_rewards),
        "reward_mean_finite": (
            sum(finite_rewards) / len(finite_rewards) if finite_rewards else None
        ),
        "curriculum": {},
    }
    for prefix in fraction_prefixes:
        total = counted_episodes[prefix]
        counts = weighted_counts[prefix]
        result["curriculum"][prefix.removesuffix("fraction_")] = {
            "counted_episodes": total,
            "counts": counts,
            "ratios": {
                key: value / total if total > 0.0 else None
                for key, value in counts.items()
            },
        }
    return result


def render_report(payload: dict[str, Any], source: Path) -> str:
    target = payload["curriculum"]["target_profile_"]
    variants = payload["curriculum"]["aim_variant_"]
    lines = [
        "# 장시간 학습 Progress 분석",
        "",
        f"- source: `{source}`",
        f"- SHA256: `{payload['training_log_sha256']}`",
        f"- iteration: {payload['last_iteration']}",
        f"- sampled steps: {payload['sampled_steps']}",
        f"- learner steps: {payload['learner_steps']}",
        f"- effective learner time: {payload['effective_learner_time_s']}s",
        f"- episode lifetime offset: {payload['episode_lifetime_offset']}",
        f"- completed episodes in this log: {payload['completed_episode_delta']}",
        f"- estimated crash episodes: {payload['estimated_crash_episodes']}",
        f"- critical NaN rows: {payload['critical_nan_rows']}",
        "",
        "## Target profile 실행 빈도",
        "",
        "| profile | episodes | ratio |",
        "|---|---:|---:|",
    ]
    for name, count in target["counts"].items():
        lines.append(f"| {name} | {count:.3f} | {target['ratios'][name]:.4f} |")
    lines += [
        "",
        "## Aim geometry 실행 빈도",
        "",
        "| geometry | episodes | ratio |",
        "|---|---:|---:|",
    ]
    for name, count in variants["counts"].items():
        lines.append(f"| {name} | {count:.3f} | {variants['ratios'][name]:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    source = Path(args.training_log)
    with source.open("r", encoding="utf-8", newline="") as stream:
        payload = analyze_rows(list(csv.DictReader(stream)))
    payload["training_log"] = str(source.resolve())
    payload["training_log_sha256"] = sha256(source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render_report(payload, source), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
