from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = ROOT / "artifacts/evaluations/guidance_advantage_v2"
DEFAULT_OUTPUT = ROOT / "automation/evidence/state_conditioned_hybrid_v3"
DEFAULT_REPORT = ROOT / "automation/reports/StateConditionedHybrid_v3_Oracle_분석.md"
MEANINGFUL_DELTA = 0.001
LARGE_REGRESSION_DELTA = -0.003
EXPECTED_V2_EXPERIMENTS = (
    "ablation_pilot_20260819",
    "ablation_elpos_revalidation_20260819",
    "ablation_elpos_near_revalidation_20260819",
    "ablation_signed_near_revalidation_20260819",
    "ablation_vertical_high_focus_20260819",
)


def _denormalize(value: float, minimum: float, maximum: float) -> float:
    return minimum + (float(value) + 1.0) * 0.5 * (maximum - minimum)


def _normalize(value: float, minimum: float, maximum: float) -> float:
    midpoint = 0.5 * (minimum + maximum)
    half = 0.5 * (maximum - minimum)
    return float(np.clip((float(value) - midpoint) / half, -1.0, 1.0))


def reconstruct_server_v2_observation(
    legacy_observation: Iterable[float], scenario: dict[str, Any]
) -> list[float]:
    """Recover the 42D server-v2 vector without using either health feature.

    The v2 raw result stores the 45D selector-v1 vector. Its final 29 features
    already have the server-v2 normalization. The first 13 server-v2 fields
    are reconstructed from health-free Tactical16 values and the deterministic
    target scenario.
    """

    legacy = np.asarray(tuple(legacy_observation), dtype=np.float64)
    if legacy.shape != (45,) or not np.all(np.isfinite(legacy)):
        raise ValueError("expected finite 45D legacy Guidance observation")
    env = scenario["env_config"]
    target = tuple(float(value) for value in env["target"])
    if len(target) != 7:
        raise ValueError("scenario target must contain N/E/D/roll/pitch/yaw/speed")

    own_roll = _denormalize(legacy[0], -180.0, 180.0)
    own_pitch = _denormalize(legacy[1], -90.0, 90.0)
    own_yaw = _denormalize(legacy[2], 0.0, 360.0)
    own_speed = _denormalize(legacy[3], 0.0, 600.0)
    own_altitude = _denormalize(legacy[4], 0.0, 15000.0)
    delta_n = _denormalize(legacy[6], -15000.0, 15000.0)
    delta_e = _denormalize(legacy[7], -15000.0, 15000.0)
    delta_d = _denormalize(legacy[8], -8000.0, 8000.0)
    target_roll, target_pitch, target_yaw = target[3], target[4], target[5]
    target_speed = target[6]
    target_altitude = -target[2]

    prefix = [
        _normalize(own_roll, -180.0, 180.0),
        _normalize(own_pitch, -90.0, 90.0),
        _normalize(own_yaw, 0.0, 360.0),
        _normalize(own_speed, 100.0, 400.0),
        _normalize(own_altitude, 0.0, 10000.0),
        _normalize(target_roll, -180.0, 180.0),
        _normalize(target_pitch, -90.0, 90.0),
        _normalize(target_yaw, 0.0, 360.0),
        _normalize(target_speed, 100.0, 400.0),
        _normalize(target_altitude, 0.0, 10000.0),
        _normalize(delta_n, -3000.0, 3000.0),
        _normalize(delta_e, -3000.0, 3000.0),
        _normalize(delta_d, -3000.0, 3000.0),
    ]
    vector = np.asarray([*prefix, *legacy[16:].tolist()], dtype=np.float32)
    # v2 ablation reused maximum_active_frames as the tested duration. That made
    # gate_elapsed_norm action-dependent before the action had been selected.
    # Canonicalize all first-selection history fields to the frozen v3 runtime
    # context so the state hash cannot leak magnitude or duration.
    vector[36] = 1.0  # recent authority ratio = 1.0
    vector[37] = -1.0  # previous action = BT_DEFAULT
    vector[38] = -1.0  # current hold = 0
    vector[39] = _normalize(1.0, 0.0, 36.0)  # first active gate frame
    vector[40] = 1.0  # gate active
    if vector.shape != (42,) or not np.all(np.isfinite(vector)):
        raise ValueError("reconstructed server-v2 observation is invalid")
    return np.clip(vector, -1.0, 1.0).astype(float).tolist()


def canonical_state_hash(
    scenario: dict[str, Any], server_observation: Iterable[float], sim_time_s: float
) -> tuple[str, dict[str, Any]]:
    payload = {
        # Evaluator display names are provenance, not physical state. Excluding
        # them prevents renamed replicas from inflating the independent-state count.
        "scenario_env_config": scenario["env_config"],
        "sim_time_s": round(float(sim_time_s), 9),
        "server_observation": [round(float(value), 8) for value in server_observation],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest().upper(), payload


def encode_action(action: str, magnitude_deg: float, duration_frames: int) -> dict[str, Any]:
    if action == "BT_DEFAULT":
        axis, sign = "default", 0
    elif "_AZ_" in action:
        axis, sign = "azimuth", 1 if "_POS_" in action else -1
    elif "_EL_" in action:
        axis, sign = "elevation", 1 if "_POS_" in action else -1
    else:
        raise ValueError(f"unsupported v3 action: {action}")
    return {
        "axis": axis,
        "axis_one_hot": [
            1.0 if axis == "default" else 0.0,
            1.0 if axis == "azimuth" else 0.0,
            1.0 if axis == "elevation" else 0.0,
        ],
        "sign": sign,
        "magnitude_deg": float(magnitude_deg),
        "magnitude_norm": float(magnitude_deg) / 0.5 if magnitude_deg else 0.0,
        "duration_frames": int(duration_frames),
        "duration_norm": float(duration_frames) / 36.0 if duration_frames else 0.0,
    }


def _load_experiment(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    pairs = json.loads((root / "pairs.json").read_text(encoding="utf-8"))
    scenarios: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "runs").glob("*/scenario.json")):
        scenarios[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    return pairs, scenarios


def rebuild_rows(
    v2_root: Path = V2_ROOT,
    additional_experiments: Iterable[Path] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observed_rows: list[dict[str, Any]] = []
    state_payloads: dict[str, dict[str, Any]] = {}
    state_sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duplicate_values: dict[tuple[str, str], list[float]] = defaultdict(list)

    experiment_roots = [
        (experiment, v2_root / experiment) for experiment in EXPECTED_V2_EXPERIMENTS
    ]
    experiment_roots.extend(
        (Path(root).resolve().name, Path(root).resolve()) for root in additional_experiments
    )
    seen_names: set[str] = set()
    for experiment, root in experiment_roots:
        if experiment in seen_names:
            raise ValueError(f"duplicate experiment name: {experiment}")
        seen_names.add(experiment)
        pairs, scenarios = _load_experiment(root)
        for pair in pairs:
            case_id = str(pair["case_id"])
            scenario = scenarios[case_id]
            result_path = root / "runs" / case_id / f"{pair['candidate_id']}.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            snapshot = result["ownship_provider_telemetry"]["first_selector_snapshot"]
            server_observation = reconstruct_server_v2_observation(
                snapshot["observation"], scenario
            )
            state_hash, canonical = canonical_state_hash(
                scenario, server_observation, snapshot["sim_time_s"]
            )
            if state_hash in state_payloads and state_payloads[state_hash] != canonical:
                raise RuntimeError(f"state hash collision: {state_hash}")
            state_payloads[state_hash] = canonical
            source = {"experiment": experiment, "case_id": case_id, "seed": int(pair["seed"])}
            if source not in state_sources[state_hash]:
                state_sources[state_hash].append(source)
            action_parameters = encode_action(
                str(pair["action"]), float(pair["magnitude_deg"]), int(pair["duration_frames"])
            )
            row = {
                "state_hash": state_hash,
                "source_experiment": experiment,
                "source_case_id": case_id,
                "seed": int(pair["seed"]),
                "family": str(pair["family"]),
                "candidate_id": str(pair["candidate_id"]),
                "action": str(pair["action"]),
                "action_parameters": action_parameters,
                "server_observation": server_observation,
                "damage_delta": float(pair["damage_delta"]),
                "cone_time_delta_s": float(pair["cone_time_delta_s"]),
                "los_improvement_deg": float(pair["los_improvement_deg"]),
                "los_rate_improvement_deg_s": float(pair["los_rate_improvement_deg_s"]),
                "ownship_crash": bool(pair["ownship_crash"]),
                "target_crash": bool(pair["target_crash"]),
                "contaminated": bool(pair["contaminated"]),
                "throttle_violations": int(pair["throttle_violations"]),
                "intervention_frames": int(pair["intervention_frames"]),
                "initial_signed_azimuth": pair.get("initial_signed_azimuth"),
                "initial_signed_elevation": pair.get("initial_signed_elevation"),
                "initial_los_azimuth_rate": pair.get("initial_los_azimuth_rate"),
                "initial_los_elevation_rate": pair.get("initial_los_elevation_rate"),
                "initial_range_m": pair.get("initial_range_m"),
                "initial_directional_headroom": pair.get("initial_directional_headroom"),
            }
            observed_rows.append(row)
            duplicate_values[(state_hash, row["candidate_id"])].append(row["damage_delta"])

    duplicate_spread = {
        f"{state_hash}:{candidate}": max(values) - min(values)
        for (state_hash, candidate), values in duplicate_values.items()
        if len(values) > 1
    }
    if duplicate_spread and max(duplicate_spread.values()) > 1e-9:
        raise RuntimeError(f"non-deterministic duplicate state/action results: {duplicate_spread}")

    states = [
        {
            "state_id": f"v3_state_{index:04d}",
            "state_hash": state_hash,
            "family": next(row["family"] for row in observed_rows if row["state_hash"] == state_hash),
            "sources": state_sources[state_hash],
            "canonical": state_payloads[state_hash],
        }
        for index, state_hash in enumerate(sorted(state_payloads), start=1)
    ]
    id_by_hash = {state["state_hash"]: state["state_id"] for state in states}
    rows = []
    for replicate_id, row in enumerate(
        sorted(
            observed_rows,
            key=lambda value: (
                value["state_hash"],
                value["candidate_id"],
                value["source_experiment"],
            ),
        ),
        start=1,
    ):
        rows.append(
            {
                "state_id": id_by_hash[row["state_hash"]],
                "replicate_id": f"v3_pair_{replicate_id:04d}",
                **row,
            }
        )
    return states, rows


def _metrics(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "positive_ratio": float(np.mean(array > 0.0)),
        "positive_ratio_epsilon_1e_6": float(np.mean(array > 1e-6)),
        "meaningful_positive_ratio": float(np.mean(array >= MEANINGFUL_DELTA)),
        "large_regression_ratio": float(np.mean(array <= LARGE_REGRESSION_DELTA)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def bootstrap_ci(values: Iterable[float], *, seed: int = 260819, samples: int = 10000) -> list[float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(array, size=(samples, array.size), replace=True), axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def analyze_oracle(states: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Preserve all raw replicates in the dataset, but count each physical
    # state/action once for oracle and static-policy value.
    replicated: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        replicated[(row["state_hash"], row["candidate_id"])].append(row)
    unique_rows = []
    for replicate_rows in replicated.values():
        representative = dict(replicate_rows[0])
        representative["damage_delta"] = float(
            np.mean([float(row["damage_delta"]) for row in replicate_rows])
        )
        representative["replicate_count"] = len(replicate_rows)
        unique_rows.append(representative)
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique_rows:
        by_state[row["state_hash"]].append(row)
        by_candidate[row["candidate_id"]].append(row)

    oracle_values: list[float] = []
    oracle_rows: list[dict[str, Any]] = []
    default_optimal = 0
    family_positive: dict[str, int] = defaultdict(int)
    candidate_count_histogram: dict[int, int] = defaultdict(int)
    for state in states:
        options = by_state[state["state_hash"]]
        candidate_count_histogram[len(options)] += 1
        best = max(options, key=lambda row: row["damage_delta"])
        value = max(0.0, float(best["damage_delta"]))
        selected = "BT_DEFAULT" if value <= 0.0 else best["candidate_id"]
        default_optimal += int(selected == "BT_DEFAULT")
        family_positive[state["family"]] += int(value > 0.0)
        oracle_values.append(value)
        oracle_rows.append(
            {
                "state_id": state["state_id"],
                "state_hash": state["state_hash"],
                "family": state["family"],
                "observed_candidates": len(options),
                "oracle_candidate_id": selected,
                "oracle_damage_delta": value,
                "runner_up_damage_delta": float(
                    sorted([0.0, *(float(row["damage_delta"]) for row in options)], reverse=True)[1]
                ),
            }
        )

    static = []
    for candidate, candidate_rows in by_candidate.items():
        values = [float(row["damage_delta"]) for row in candidate_rows]
        static.append(
            {
                "candidate_id": candidate,
                "coverage_states": len(candidate_rows),
                "coverage_ratio": len(candidate_rows) / len(states),
                **_metrics(values),
            }
        )
    eligible_static = [row for row in static if row["coverage_states"] >= 30]
    best_static = max(eligible_static, key=lambda row: (row["mean"], row["coverage_states"]))
    best_static_states = {row["state_hash"] for row in by_candidate[best_static["candidate_id"]]}
    oracle_shared = [
        row["oracle_damage_delta"] for row in oracle_rows if row["state_hash"] in best_static_states
    ]
    oracle_shared_metrics = _metrics(oracle_shared)

    family_rule = {}
    family_rule_values = []
    for family in sorted({state["family"] for state in states}):
        family_rows = [row for row in unique_rows if row["family"] == family]
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in family_rows:
            grouped[row["candidate_id"]].append(float(row["damage_delta"]))
        candidates = [
            (candidate, values) for candidate, values in grouped.items() if len(values) >= 3
        ]
        candidate, values = max(candidates, key=lambda item: np.mean(item[1]))
        family_rule[family] = {"candidate_id": candidate, **_metrics(values)}
        family_rule_values.extend(values)

    oracle_metrics = _metrics(oracle_values)
    oracle_metrics["bootstrap_mean_95ci"] = bootstrap_ci(oracle_values)
    oracle_metrics["coverage_states"] = len(oracle_values)
    oracle_metrics["coverage_ratio"] = len(oracle_values) / len(states)
    oracle_metrics["default_optimal_ratio"] = default_optimal / len(states)
    family_count = sum(count > 0 for count in family_positive.values())
    shared_gap = float(oracle_shared_metrics["mean"] - best_static["mean"])
    feasible = (
        oracle_metrics["mean"] > 0.0
        and oracle_metrics["median"] > 0.0
        and oracle_metrics["positive_ratio"] >= 0.60
        and family_count >= 2
        and shared_gap >= MEANINGFUL_DELTA
    )
    sparse_state_ratio = float(
        sum(states_count for count, states_count in candidate_count_histogram.items() if count < 10)
        / len(states)
    )
    if feasible:
        feasibility = "ORACLE_FEASIBLE"
    elif oracle_metrics["mean"] > 0.0 and shared_gap >= MEANINGFUL_DELTA and sparse_state_ratio >= 0.5:
        feasibility = "ORACLE_UNDERSAMPLED"
    else:
        feasibility = "ORACLE_INSUFFICIENT"
    return {
        "schema_version": "state_conditioned_oracle_v3.v1",
        "unique_states": len(states),
        "observed_nondefault_state_action_pairs": len(rows),
        "unique_nondefault_state_action_pairs": len(unique_rows),
        "candidate_count_by_state": {
            str(key): value for key, value in sorted(candidate_count_histogram.items())
        },
        "pure_bt": _metrics([0.0] * len(states)),
        "oracle": oracle_metrics,
        "oracle_rows": oracle_rows,
        "best_static_min_30_states": best_static,
        "oracle_on_best_static_states": oracle_shared_metrics,
        "oracle_mean_gap_over_static_on_shared_states": shared_gap,
        "geometry_rule_in_sample_diagnostic": {
            "families": family_rule,
            "aggregate": _metrics(family_rule_values),
        },
        "positive_oracle_state_families": family_positive,
        "state_families_with_positive_oracle": family_count,
        "static_candidates": sorted(static, key=lambda row: row["mean"], reverse=True),
        "sparse_state_ratio_lt_10_actions": sparse_state_ratio,
        "feasibility": feasibility,
        "feasibility_gate": {
            "oracle_mean_gt_zero": oracle_metrics["mean"] > 0.0,
            "oracle_median_gt_zero": oracle_metrics["median"] > 0.0,
            "oracle_positive_ratio_gte_0_60": oracle_metrics["positive_ratio"] >= 0.60,
            "positive_state_families_gte_2": family_count >= 2,
            "oracle_shared_mean_gap_gte_0_001": shared_gap >= MEANINGFUL_DELTA,
        },
        "caveat": (
            "in-sample sparse oracle is an upper bound and is optimistic when a state has many "
            "observed actions; it proves action-space potential, not learnable policy value"
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_outputs(
    states: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    analysis: dict[str, Any],
    output: Path,
    report_path: Path,
    source_experiments: Iterable[str] = EXPECTED_V2_EXPERIMENTS,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    states_path = output / "state_matrix_v3.json"
    dataset_path = output / "counterfactual_bootstrap_v3.dataset.json"
    analysis_path = output / "oracle_analysis_v3.json"
    states_payload = {
        "schema_version": "counterfactual_state_matrix_v3.v1",
        "missing_action_policy": "preserve_missing_never_impute_zero",
        "states": states,
        "rows": rows,
    }
    states_path.write_text(json.dumps(states_payload, indent=2, sort_keys=True), encoding="utf-8")
    dataset_rows = list(rows)
    by_state = {row["state_hash"]: row for row in rows}
    for state in states:
        example = by_state[state["state_hash"]]
        dataset_rows.append(
            {
                "state_id": state["state_id"],
                "state_hash": state["state_hash"],
                "source_experiment": "derived_exact_default",
                "source_case_id": example["source_case_id"],
                "seed": example["seed"],
                "family": state["family"],
                "candidate_id": "BT_DEFAULT",
                "action": "BT_DEFAULT",
                "action_parameters": encode_action("BT_DEFAULT", 0.0, 0),
                "server_observation": example["server_observation"],
                "damage_delta": 0.0,
                "label_source": "definition_vs_exact_pure_bt",
            }
        )
    dataset_path.write_text(
        json.dumps(
            sorted(dataset_rows, key=lambda row: (row["state_hash"], row["candidate_id"])),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": "counterfactual_bootstrap_v3.manifest.v1",
        "source_experiments": list(source_experiments),
        "unique_states": len(states),
        "observed_nondefault_pairs": len(rows),
        "dataset_rows_including_default": len(dataset_rows),
        "observation_contract": "guidance_selector_server_v2_reconstructed_from_v2_raw",
        "health_runtime_features": [],
        "state_matrix_sha256": _sha256(states_path),
        "dataset_sha256": _sha256(dataset_path),
        "oracle_analysis_sha256": _sha256(analysis_path),
        "feasibility": analysis["feasibility"],
    }
    (output / "manifest_v3.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    oracle = analysis["oracle"]
    static = analysis["best_static_min_30_states"]
    lines = [
        "# State-Conditioned Counterfactual Hybrid v3 Oracle 분석",
        "",
        "## 결론",
        "",
        f"- 판정: `{analysis['feasibility']}`",
        f"- canonical unique state: {len(states)}",
        f"- observed nondefault state-action pair: {len(rows)}",
        f"- State Oracle ΔDamage mean/median: {oracle['mean']:.9f} / {oracle['median']:.9f}",
        f"- Oracle positive/meaningful-positive ratio: {oracle['positive_ratio']:.2%} / {oracle['meaningful_positive_ratio']:.2%}",
        f"- Oracle positive ratio (epsilon 1e-6): {oracle['positive_ratio_epsilon_1e_6']:.2%}",
        f"- Oracle bootstrap mean 95% CI: [{oracle['bootstrap_mean_95ci'][0]:.9f}, {oracle['bootstrap_mean_95ci'][1]:.9f}]",
        f"- default-optimal state ratio: {oracle['default_optimal_ratio']:.2%}",
        "",
        "## Best Static 비교",
        "",
        f"- candidate: `{static['candidate_id']}`",
        f"- coverage: {static['coverage_states']}/{len(states)} ({static['coverage_ratio']:.2%})",
        f"- ΔDamage mean/median/positive: {static['mean']:.9f} / {static['median']:.9f} / {static['positive_ratio']:.2%}",
        f"- 같은 state에서 Oracle mean gap: {analysis['oracle_mean_gap_over_static_on_shared_states']:.9f}",
        "",
        "## 해석 주의",
        "",
        "이 Oracle은 state별로 관측된 action 중 최댓값을 고른 in-sample upper bound다. 60개 action을 관측한 state와 3개만 관측한 state가 섞여 있으므로 action 수가 많은 state에서 selection optimism이 커질 수 있다. 따라서 `ORACLE_FEASIBLE`은 학습 가능성을 검증할 다음 단계 진입 조건이지 Promotion 근거가 아니다.",
        "",
        "결측 action은 0으로 채우지 않았고 sparse matrix에서 결측으로 유지했다.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild v2 pairs and evaluate state oracle v3")
    parser.add_argument("--v2-root", type=Path, default=V2_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--additional-experiment",
        action="append",
        type=Path,
        default=[],
        help="Additional evaluator root containing pairs.json and runs/*/scenario.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    additional = [path.resolve() for path in args.additional_experiment]
    states, rows = rebuild_rows(args.v2_root.resolve(), additional)
    analysis = analyze_oracle(states, rows)
    source_experiments = [*EXPECTED_V2_EXPERIMENTS, *(path.name for path in additional)]
    write_outputs(
        states,
        rows,
        analysis,
        args.output.resolve(),
        args.report.resolve(),
        source_experiments,
    )
    print(json.dumps({key: analysis[key] for key in ("unique_states", "observed_nondefault_state_action_pairs", "oracle", "best_static_min_30_states", "oracle_mean_gap_over_static_on_shared_states", "feasibility")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
