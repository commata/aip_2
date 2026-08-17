"""Pure 0815 paired aim 평가의 기하별 효과와 불확실성을 분석한다."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any


METRICS = (
    "damage_dealt",
    "mean_los_deg",
    "los_rate_rms_deg_s",
    "damage_cone_time_s",
    "time_to_first_damage_s",
    "min_altitude_m",
    "action_saturated_ratio",
)
AUTHORITY_METRICS = (
    "gate_active_ratio",
    "roll_applied_to_requested_ratio",
    "pitch_applied_to_requested_ratio",
    "yaw_applied_to_requested_ratio",
    "bt_roll_saturation_ratio",
    "bt_pitch_saturation_ratio",
    "bt_yaw_saturation_ratio",
    "final_roll_saturation_ratio",
    "final_pitch_saturation_ratio",
    "final_yaw_saturation_ratio",
    "bt_roll_positive_headroom_mean",
    "bt_roll_negative_headroom_mean",
    "bt_pitch_positive_headroom_mean",
    "bt_pitch_negative_headroom_mean",
    "bt_yaw_positive_headroom_mean",
    "bt_yaw_negative_headroom_mean",
)


def finite(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def bootstrap_mean_ci(
    values: list[float], *, samples: int, seed: int
) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = [
        fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(samples)
    ]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def summarize_values(values: list[float], *, samples: int, seed: int) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": fmean(values) if values else None,
        "median": median(values) if values else None,
        "std": pstdev(values) if len(values) > 1 else (0.0 if values else None),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "bootstrap_mean_95ci": bootstrap_mean_ci(values, samples=samples, seed=seed),
    }


def pair_records(records: list[dict[str, Any]], controller: str) -> list[dict[str, Any]]:
    pure = {int(row["seed"]): row for row in records if row.get("controller") == "pure_0815"}
    hybrid = {int(row["seed"]): row for row in records if row.get("controller") == controller}
    if set(pure) != set(hybrid):
        raise ValueError(
            f"Pure/Hybrid seed 집합 불일치: pure={sorted(pure)}, hybrid={sorted(hybrid)}"
        )
    pairs = []
    for seed in sorted(pure):
        baseline, candidate = pure[seed], hybrid[seed]
        if baseline.get("variant_name") != candidate.get("variant_name"):
            raise ValueError(f"seed {seed}: Pure/Hybrid variant 불일치")
        delta = {}
        for metric in METRICS:
            left, right = finite(baseline.get(metric)), finite(candidate.get(metric))
            delta[metric] = right - left if left is not None and right is not None else None
        pairs.append(
            {
                "seed": seed,
                "variant_name": candidate.get("variant_name"),
                "pure_run_id": baseline.get("run_id"),
                "hybrid_run_id": candidate.get("run_id"),
                "pure_outcome": baseline.get("outcome"),
                "hybrid_outcome": candidate.get("outcome"),
                "pure_ownship_crash": bool(baseline.get("ownship_crash", False)),
                "hybrid_ownship_crash": bool(candidate.get("ownship_crash", False)),
                "delta": delta,
                "hybrid_authority": {
                    metric: finite(candidate.get(metric)) for metric in AUTHORITY_METRICS
                },
            }
        )
    return pairs


def metric_summary(
    pairs: list[dict[str, Any]], *, samples: int, seed: int
) -> dict[str, Any]:
    return {
        metric: summarize_values(
            [value for pair in pairs if (value := pair["delta"][metric]) is not None],
            samples=samples,
            seed=seed + index,
        )
        for index, metric in enumerate(METRICS)
    }


def authority_summary(pairs: list[dict[str, Any]]) -> dict[str, float | None]:
    result = {}
    for metric in AUTHORITY_METRICS:
        values = [
            value
            for pair in pairs
            if (value := pair["hybrid_authority"][metric]) is not None
        ]
        result[metric] = fmean(values) if values else None
    return result


def representatives(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [pair for pair in pairs if pair["delta"]["damage_dealt"] is not None]
    if not valid:
        return {}
    mean_damage = fmean(pair["delta"]["damage_dealt"] for pair in valid)
    return {
        "worst_damage": min(valid, key=lambda pair: pair["delta"]["damage_dealt"]),
        "nearest_mean_damage": min(
            valid, key=lambda pair: abs(pair["delta"]["damage_dealt"] - mean_damage)
        ),
        "best_damage": max(valid, key=lambda pair: pair["delta"]["damage_dealt"]),
    }


def pearson(pairs: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    values = [
        (x, y)
        for pair in pairs
        if (x := pair["delta"].get(left)) is not None
        and (y := pair["delta"].get(right)) is not None
    ]
    if len(values) < 3:
        return {"n": len(values), "pearson_r": None}
    xs, ys = zip(*values)
    mean_x, mean_y = fmean(xs), fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in values)
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return {"n": len(values), "pearson_r": numerator / denominator if denominator else None}


def reward_chain_correlations(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "los_delta_vs_cone_delta": pearson(
            pairs, "mean_los_deg", "damage_cone_time_s"
        ),
        "los_rate_delta_vs_cone_delta": pearson(
            pairs, "los_rate_rms_deg_s", "damage_cone_time_s"
        ),
        "cone_delta_vs_damage_delta": pearson(
            pairs, "damage_cone_time_s", "damage_dealt"
        ),
        "first_damage_delta_vs_damage_delta": pearson(
            pairs, "time_to_first_damage_s", "damage_dealt"
        ),
        "saturation_delta_vs_damage_delta": pearson(
            pairs, "action_saturated_ratio", "damage_dealt"
        ),
    }


def analyze(
    payload: dict[str, Any], controller: str, *, bootstrap_samples: int, bootstrap_seed: int
) -> dict[str, Any]:
    pairs = pair_records(payload["records"], controller)
    variants = sorted({str(pair["variant_name"]) for pair in pairs})
    return {
        "controller": controller,
        "pairs": len(pairs),
        "crash_regressions": sum(
            pair["hybrid_ownship_crash"] and not pair["pure_ownship_crash"]
            for pair in pairs
        ),
        "metrics": metric_summary(pairs, samples=bootstrap_samples, seed=bootstrap_seed),
        "authority": authority_summary(pairs),
        "per_variant": {
            variant: {
                "pairs": len(rows := [pair for pair in pairs if pair["variant_name"] == variant]),
                "metrics": metric_summary(
                    rows, samples=bootstrap_samples, seed=bootstrap_seed + 1000 + index * 20
                ),
                "authority": authority_summary(rows),
            }
            for index, variant in enumerate(variants)
        },
        "representatives": representatives(pairs),
        "reward_chain_correlations": reward_chain_correlations(pairs),
        "paired_records": pairs,
        "bootstrap": {"samples": bootstrap_samples, "seed": bootstrap_seed},
    }


def fmt(value: Any, digits: int = 6) -> str:
    return "자료 없음" if value is None else f"{float(value):+.{digits}f}"


def render_markdown(result: dict[str, Any], source: Path) -> str:
    lines = [
        "# Aim residual paired 후보 분석",
        "",
        f"- source: `{source}`",
        f"- controller: `{result['controller']}`",
        f"- paired episodes: `{result['pairs']}`",
        f"- crash regressions: `{result['crash_regressions']}`",
        "",
        "## 전체 paired delta",
        "",
        "| 지표 | 평균 | 중앙값 | 최악(min) | 최대 | Bootstrap 평균 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, values in result["metrics"].items():
        ci = values["bootstrap_mean_95ci"]
        ci_text = "자료 없음" if ci is None else f"[{fmt(ci[0])}, {fmt(ci[1])}]"
        lines.append(
            f"| {metric} | {fmt(values['mean'])} | {fmt(values['median'])} | "
            f"{fmt(values['min'])} | {fmt(values['max'])} | {ci_text} |"
        )
    lines += ["", "## 기하별", ""]
    for variant, summary in result["per_variant"].items():
        damage = summary["metrics"]["damage_dealt"]
        los = summary["metrics"]["mean_los_deg"]
        saturation = summary["authority"]["final_roll_saturation_ratio"]
        lines.append(
            f"- `{variant}` n={summary['pairs']}: Damage {fmt(damage['mean'])}, "
            f"LOS {fmt(los['mean'])}°, final roll saturation {fmt(saturation)}"
        )
    lines += ["", "## 대표 episode", ""]
    for label, pair in result["representatives"].items():
        lines.append(
            f"- {label}: seed `{pair['seed']}`, `{pair['variant_name']}`, "
            f"Damage Δ {fmt(pair['delta']['damage_dealt'])}, "
            f"LOS Δ {fmt(pair['delta']['mean_los_deg'])}°"
        )
    lines += [
        "",
        "## Reward-chain 설명 상관",
        "",
        "표본 내 paired delta의 Pearson 상관이며 인과 또는 독립 검증 근거가 아니다.",
        "",
    ]
    for label, values in result["reward_chain_correlations"].items():
        lines.append(
            f"- `{label}`: r={fmt(values['pearson_r'])}, n={values['n']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--controller", default="hybrid_0.125")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=815)
    args = parser.parse_args()
    payload = json.loads(args.evaluation.read_text(encoding="utf-8"))
    result = analyze(
        payload,
        args.controller,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = args.report or args.output.with_suffix(".md")
    report.write_text(render_markdown(result, args.evaluation), encoding="utf-8")
    print(json.dumps({"pairs": result["pairs"], "metrics": result["metrics"]}))


if __name__ == "__main__":
    main()
