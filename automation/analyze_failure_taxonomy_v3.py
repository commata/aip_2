from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from dogfight.ai.guidance_advantage import GUIDANCE_SERVER_FEATURES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "automation/evidence/state_conditioned_hybrid_v3/state_matrix_v3.json"
DEFAULT_ORACLE = ROOT / "automation/evidence/state_conditioned_hybrid_v3/oracle_analysis_v3.json"
DEFAULT_OUTPUT = ROOT / "automation/evidence/state_conditioned_hybrid_v3/failure_taxonomy_v3.json"
DEFAULT_REPORT = ROOT / "automation/reports/StateConditionedHybrid_v3_BT실패분류.md"
FEATURE_INDEX = {name: index for index, name in enumerate(GUIDANCE_SERVER_FEATURES)}
FAILURE_TYPES = (
    "A_AZIMUTH_OVERSHOOT",
    "B_ELEVATION_OVERSHOOT",
    "C_LOS_ANGULAR_RATE_HIGH",
    "D_RANGE_TOO_CLOSE",
    "E_RANGE_TOO_FAR",
    "F_CLOSING_TOO_HIGH",
    "G_SURFACE_AUTHORITY_LIMIT",
    "H_CROSSING_LEAD_SHORTFALL",
    "I_PURE_BT_ALREADY_OPTIMAL",
    "J_ENERGY_ALTITUDE_SAFETY",
)


def _denormalize(value: float, minimum: float, maximum: float) -> float:
    return minimum + (float(value) + 1.0) * 0.5 * (maximum - minimum)


def classify_failure(observation: list[float], family: str, oracle_value: float) -> str:
    value = np.asarray(observation, dtype=np.float64)
    if oracle_value <= 1e-6:
        return "I_PURE_BT_ALREADY_OPTIMAL"
    safety = value[FEATURE_INDEX["safety_margin_norm"]]
    saturated = value[FEATURE_INDEX["any_surface_saturation"]] > 0.0
    azimuth = 15.0 * value[FEATURE_INDEX["signed_aim_azimuth_norm"]]
    elevation = 15.0 * value[FEATURE_INDEX["signed_aim_elevation_norm"]]
    azimuth_rate = 15.0 * value[FEATURE_INDEX["los_azimuth_rate_norm"]]
    elevation_rate = 15.0 * value[FEATURE_INDEX["los_elevation_rate_norm"]]
    distance = _denormalize(value[FEATURE_INDEX["range_norm"]], 0.0, 3000.0)
    closing = _denormalize(value[FEATURE_INDEX["closing_rate_norm"]], -250.0, 250.0)
    if safety < -0.2:
        return "J_ENERGY_ALTITUDE_SAFETY"
    if saturated:
        return "G_SURFACE_AUTHORITY_LIMIT"
    if distance < 500.0:
        return "D_RANGE_TOO_CLOSE"
    if distance > 1200.0:
        return "E_RANGE_TOO_FAR"
    if closing > 120.0:
        return "F_CLOSING_TOO_HIGH"
    if abs(azimuth) >= abs(elevation) and abs(azimuth) > 0.5 and azimuth * azimuth_rate > 0.0:
        return "A_AZIMUTH_OVERSHOOT"
    if abs(elevation) > abs(azimuth) and abs(elevation) > 0.5 and elevation * elevation_rate > 0.0:
        return "B_ELEVATION_OVERSHOOT"
    if max(abs(azimuth_rate), abs(elevation_rate)) > 3.0:
        return "C_LOS_ANGULAR_RATE_HIGH"
    if family.startswith("crossing"):
        return "H_CROSSING_LEAD_SHORTFALL"
    return "A_AZIMUTH_OVERSHOOT" if abs(azimuth) >= abs(elevation) else "B_ELEVATION_OVERSHOOT"


def analyze(matrix: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    state_by_hash = {state["state_hash"]: state for state in matrix["states"]}
    row_by_state_candidate = {
        (row["state_hash"], row["candidate_id"]): row for row in matrix["rows"]
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    state_rows = []
    for oracle_row in oracle["oracle_rows"]:
        state = state_by_hash[oracle_row["state_hash"]]
        candidate = oracle_row["oracle_candidate_id"]
        example = next(row for row in matrix["rows"] if row["state_hash"] == state["state_hash"])
        failure = classify_failure(
            example["server_observation"],
            state["family"],
            float(oracle_row["oracle_damage_delta"]),
        )
        action_parameters = None
        if candidate != "BT_DEFAULT":
            selected = row_by_state_candidate.get((state["state_hash"], candidate))
            action_parameters = selected["action_parameters"] if selected else None
        record = {
            "state_hash": state["state_hash"],
            "family": state["family"],
            "failure_type": failure,
            "oracle_candidate_id": candidate,
            "oracle_damage_delta": float(oracle_row["oracle_damage_delta"]),
            "action_parameters": action_parameters,
        }
        groups[failure].append(record)
        state_rows.append(record)
    summaries = {}
    for failure in FAILURE_TYPES:
        records = groups[failure]
        values = np.asarray([row["oracle_damage_delta"] for row in records], dtype=np.float64)
        actions = Counter(row["oracle_candidate_id"] for row in records)
        parameters = [row["action_parameters"] for row in records if row["action_parameters"]]
        summaries[failure] = {
            "states": len(records),
            "state_ratio": len(records) / len(state_rows),
            "oracle_mean": float(np.mean(values)) if values.size else None,
            "oracle_median": float(np.median(values)) if values.size else None,
            "oracle_positive_ratio_epsilon_1e_6": float(np.mean(values > 1e-6)) if values.size else None,
            "oracle_action_distribution": dict(actions),
            "magnitude_distribution": dict(Counter(str(row["magnitude_deg"]) for row in parameters)),
            "duration_distribution": dict(Counter(str(row["duration_frames"]) for row in parameters)),
            "families": dict(Counter(row["family"] for row in records)),
        }
    return {
        "schema_version": "bt_failure_taxonomy_v3.v1",
        "classification_kind": "deterministic_server_safe_diagnostic_rule",
        "states": len(state_rows),
        "summaries": summaries,
        "state_rows": state_rows,
        "caveat": "Failure taxonomy is diagnostic; model labels and promotion remain actual paired Damage.",
    }


def write_report(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# State-Conditioned Hybrid v3 Pure BT 실패 분류",
        "",
        "이 분류는 server-safe geometry와 same-frame BT context만 사용한 진단 규칙이다. 학습 label과 Promotion primary는 실제 paired Damage이며, 분류명 자체를 성능 근거로 사용하지 않는다.",
        "",
        "| failure type | states | oracle mean | oracle positive | dominant oracle action |",
        "|---|---:|---:|---:|---|",
    ]
    for name, summary in result["summaries"].items():
        dominant = (
            max(summary["oracle_action_distribution"], key=summary["oracle_action_distribution"].get)
            if summary["oracle_action_distribution"]
            else "N/A"
        )
        oracle_mean = summary["oracle_mean"] or 0.0
        oracle_positive = summary["oracle_positive_ratio_epsilon_1e_6"] or 0.0
        lines.append(
            f"| {name} | {summary['states']} | {oracle_mean:+.9f} | "
            f"{oracle_positive:.2%} | {dominant} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Pure BT failure taxonomy v3")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = analyze(
        json.loads(args.matrix.read_text(encoding="utf-8")),
        json.loads(args.oracle.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(result, args.report)
    print(json.dumps(result["summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
