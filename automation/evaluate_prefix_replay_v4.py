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
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dogfight.ai.prefix_replay import compare_prefix_snapshots


EXPECTED_PURE_DLL_SHA256 = (
    "4C93B4C6719CB0423388D5FC721D356020A3A36CD5AD2C56B5C3CA795BFE18C9"
)
EXPECTED_PURE_XML_SHA256 = (
    "D84C27B0B8BA22E1649AF2375BE0B83C762BC62EB5047BB590539B374F8271EE"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def verify_champion(dll: Path, xml: Path) -> None:
    for path, expected, label in (
        (dll, EXPECTED_PURE_DLL_SHA256, "Pure BT DLL"),
        (xml, EXPECTED_PURE_XML_SHA256, "Pure BT XML"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{label} SHA256 mismatch: expected={expected}, actual={actual}")


def load_jsonl_frames(path: Path) -> list[dict[str, Any]]:
    frames = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("record_type") == "frame":
            frames.append(record)
    return frames


def _wrapped_angle_error(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs((left - right + 180.0) % 360.0 - 180.0)


def compare_trajectory_frames(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    left_start: int = 0,
    right_start: int = 0,
    count: int | None = None,
) -> dict[str, Any]:
    available = min(len(left) - left_start, len(right) - right_start)
    sample_count = available if count is None else min(available, int(count))
    if sample_count <= 0:
        return {"frames": 0, "exact": False, "reason": "no_aligned_frames"}
    metrics = {
        "ownship_position_error_m_max": 0.0,
        "target_position_error_m_max": 0.0,
        "ownship_attitude_error_deg_max": 0.0,
        "target_attitude_error_deg_max": 0.0,
        "ownship_speed_error_m_s_max": 0.0,
        "target_speed_error_m_s_max": 0.0,
        "distance_error_m_max": 0.0,
        "los_error_deg_max": 0.0,
        "bt_action_error_max": 0.0,
        "bt_vp_error_m_max": 0.0,
        "target_damage_error_max": 0.0,
        "ownship_damage_error_max": 0.0,
    }
    exact = True
    for offset in range(sample_count):
        lhs = left[left_start + offset]
        rhs = right[right_start + offset]
        for side in ("ownship", "target"):
            lstate = lhs[side]
            rstate = rhs[side]
            position_error = float(
                np.linalg.norm(
                    np.asarray(lstate["position_ned_m"], dtype=np.float64)
                    - np.asarray(rstate["position_ned_m"], dtype=np.float64)
                )
            )
            attitude_error = float(
                np.max(
                    _wrapped_angle_error(
                        np.asarray(lstate["attitude_deg"], dtype=np.float64),
                        np.asarray(rstate["attitude_deg"], dtype=np.float64),
                    )
                )
            )
            speed_error = abs(
                float(lstate["speed_kcas"]) - float(rstate["speed_kcas"])
            )
            metrics[f"{side}_position_error_m_max"] = max(
                metrics[f"{side}_position_error_m_max"], position_error
            )
            metrics[f"{side}_attitude_error_deg_max"] = max(
                metrics[f"{side}_attitude_error_deg_max"], attitude_error
            )
            metrics[f"{side}_speed_error_m_s_max"] = max(
                metrics[f"{side}_speed_error_m_s_max"], speed_error
            )
        metrics["distance_error_m_max"] = max(
            metrics["distance_error_m_max"],
            abs(float(lhs["distance_m"]) - float(rhs["distance_m"])),
        )
        metrics["los_error_deg_max"] = max(
            metrics["los_error_deg_max"],
            abs(float(lhs["ata_deg"]) - float(rhs["ata_deg"])),
        )
        lhs_bt = np.asarray(lhs["hybrid"].get("bt_action", lhs["ownship_action"]))
        rhs_bt = np.asarray(rhs["hybrid"].get("bt_action", rhs["ownship_action"]))
        action_error = float(np.max(np.abs(lhs_bt - rhs_bt)))
        metrics["bt_action_error_max"] = max(
            metrics["bt_action_error_max"], action_error
        )
        lhs_vp = np.asarray(lhs["hybrid"].get("bt_vp", [0.0, 0.0, 0.0]))
        rhs_vp = np.asarray(rhs["hybrid"].get("bt_vp", [0.0, 0.0, 0.0]))
        metrics["bt_vp_error_m_max"] = max(
            metrics["bt_vp_error_m_max"], float(np.linalg.norm(lhs_vp - rhs_vp))
        )
        for damage in ("target_damage", "ownship_damage"):
            metrics[f"{damage}_error_max"] = max(
                metrics[f"{damage}_error_max"],
                abs(float(lhs[damage]) - float(rhs[damage])),
            )
        exact = exact and all(
            np.array_equal(
                np.asarray(lhs[key] if key in lhs else lhs["hybrid"].get(key)),
                np.asarray(rhs[key] if key in rhs else rhs["hybrid"].get(key)),
            )
            for key in ("ownship_action", "target_action")
        )
        exact = exact and all(value == 0.0 for value in metrics.values())
    return {"frames": sample_count, "exact": bool(exact), **metrics}


def restart_fidelity_status(comparison: dict[str, Any]) -> str:
    invalid = (
        comparison.get("frames", 0) <= 0
        or comparison["ownship_position_error_m_max"] > 1.0
        or comparison["target_position_error_m_max"] > 1.0
        or comparison["ownship_attitude_error_deg_max"] > 1.0
        or comparison["target_attitude_error_deg_max"] > 1.0
        or comparison["ownship_speed_error_m_s_max"] > 1.0
        or comparison["target_speed_error_m_s_max"] > 1.0
        or comparison["bt_action_error_max"] > 0.05
        or comparison["bt_vp_error_m_max"] > 10.0
    )
    return "RESTART_STATE_CAUSAL_INVALID" if invalid else "RESTART_STATE_PARITY_PASSED"


def materialize_restart_scenario(
    source_scenario: dict[str, Any],
    decision_record: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(json.dumps(source_scenario))
    env = payload.setdefault("env_config", {})
    own = decision_record["ownship"]
    target = decision_record["target"]
    env["ownship"] = [
        *own["position_ned_m"],
        *own["attitude_deg"],
        own["speed_kcas"],
    ]
    env["target"] = [
        *target["position_ned_m"],
        *target["attitude_deg"],
        target["speed_kcas"],
    ]
    env["initial_scenario"] = {
        "mode": "default",
        "legacy_use_random_scenario": False,
    }
    env["ownship_randomization"] = {"enabled": False}
    env["target_randomization"] = {"enabled": False}
    env["target_autopilot"] = {
        "heading_cmd": target["attitude_deg"][2],
        "altitude_cmd": target["altitude_m"],
        "speed_cmd": target["speed_kcas"],
    }
    return payload


def run_local(
    *,
    label: str,
    output: Path,
    scenario: Path,
    seed: int,
    dll: Path,
    xml: Path,
    start_frame: int,
    hold_frames: int,
    tactical_mode: str,
    episode_frames: int,
) -> dict[str, Any]:
    run_root = output / "runs" / label
    run_root.mkdir(parents=True, exist_ok=False)
    result_path = run_root / "result.json"
    telemetry_path = run_root / "telemetry.jsonl"
    command = [
        sys.executable,
        str(ROOT / "run_local_dogfight.py"),
        "--ownship-backend",
        "prefix_tactical",
        "--target-backend",
        "autopilot",
        "--ownship-bt-dll",
        str(dll),
        "--bt-rule-xml",
        str(xml),
        "--bt-rule-alias",
        "Rule_DCS_GDCC_0815.xml",
        "--bt-rule-alias-only",
        "--bt-turn-throttle-mode",
        "raw",
        "--prefix-tactical-mode",
        tactical_mode,
        "--prefix-start-frame",
        str(start_frame),
        "--prefix-hold-frames",
        str(hold_frames),
        "--scenario-file",
        str(scenario),
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
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=max(120.0, episode_frames / 10.0),
        check=False,
    )
    (run_root / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (run_root / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(
            f"prefix replay run failed label={label} returncode={completed.returncode}"
        )
    return {
        "result": json.loads(result_path.read_text(encoding="utf-8")),
        "frames": load_jsonl_frames(telemetry_path),
        "returncode": completed.returncode,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit v4 prefix-replay causal fidelity")
    parser.add_argument("--pure-bt-dll", type=Path, required=True)
    parser.add_argument("--pure-bt-xml", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=71001)
    parser.add_argument("--decision-frame", type=int, default=60)
    parser.add_argument("--episode-frames", type=int, default=180)
    parser.add_argument("--restart-compare-frames", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dll = args.pure_bt_dll.resolve()
    xml = args.pure_bt_xml.resolve()
    scenario = args.scenario.resolve()
    output = args.output_root.resolve()
    verify_champion(dll, xml)
    if not scenario.is_file():
        raise FileNotFoundError(f"scenario not found: {scenario}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prefix evidence: {output}")
    output.mkdir(parents=True)

    common = dict(
        output=output,
        scenario=scenario,
        seed=args.seed,
        dll=dll,
        xml=xml,
        start_frame=args.decision_frame,
        hold_frames=max(1, args.episode_frames - args.decision_frame),
        episode_frames=args.episode_frames,
    )
    baseline_a = run_local(label="pure_a", tactical_mode="BT_DEFAULT", **common)
    baseline_b = run_local(label="pure_b", tactical_mode="BT_DEFAULT", **common)
    default_override = run_local(
        label="bt_default_override", tactical_mode="BT_DEFAULT", **common
    )
    snapshots = [
        run["result"]["ownship_provider_telemetry"]["prefix_snapshot"]
        for run in (baseline_a, baseline_b, default_override)
    ]
    prefix_ab = compare_prefix_snapshots(snapshots[0], snapshots[1])
    prefix_default = compare_prefix_snapshots(snapshots[0], snapshots[2])
    full_ab = compare_trajectory_frames(baseline_a["frames"], baseline_b["frames"])
    full_default = compare_trajectory_frames(
        baseline_a["frames"], default_override["frames"]
    )

    decision_record_index = max(0, args.decision_frame - 1)
    decision_record = baseline_a["frames"][decision_record_index]
    restart_payload = materialize_restart_scenario(
        json.loads(scenario.read_text(encoding="utf-8")), decision_record
    )
    restart_scenario_path = output / "restart_scenario.json"
    restart_scenario_path.write_text(
        json.dumps(restart_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    restart = run_local(
        label="reconstructed_restart",
        output=output,
        scenario=restart_scenario_path,
        seed=args.seed,
        dll=dll,
        xml=xml,
        start_frame=0,
        hold_frames=max(1, args.restart_compare_frames),
        tactical_mode="BT_DEFAULT",
        episode_frames=args.restart_compare_frames,
    )
    restart_comparison = compare_trajectory_frames(
        baseline_a["frames"],
        restart["frames"],
        left_start=args.decision_frame,
        right_start=0,
        count=args.restart_compare_frames,
    )
    restart_status = restart_fidelity_status(restart_comparison)
    target_damage = [
        float(run["result"].get("target_damage", 0.0))
        for run in (baseline_a, baseline_b, default_override)
    ]
    ownship_damage = [
        float(run["result"].get("ownship_damage", 0.0))
        for run in (baseline_a, baseline_b, default_override)
    ]
    repeat_noise = max(
        np.ptp(np.asarray(target_damage, dtype=np.float64)),
        np.ptp(np.asarray(ownship_damage, dtype=np.float64)),
    )
    deterministic = bool(full_ab["exact"] and full_default["exact"] and repeat_noise == 0.0)
    epsilon = 1e-9 if deterministic else max(1e-9, float(repeat_noise) * 3.0)
    aggregate = {
        "schema_version": "prefix_replay_fidelity_v4.v1",
        "status": "PREFIX_REPLAY_PARITY_PASSED"
        if prefix_ab["match"] and prefix_default["match"] and full_default["exact"]
        else "PREFIX_REPLAY_PARITY_FAILED",
        "restart_status": restart_status,
        "causal_truth": "PREFIX_REPLAY",
        "reconstructed_restart_allowed_for_labels": restart_status
        == "RESTART_STATE_PARITY_PASSED",
        "seed": args.seed,
        "decision_frame": args.decision_frame,
        "prefix_pure_a_vs_b": prefix_ab,
        "prefix_pure_vs_bt_default_override": prefix_default,
        "trajectory_pure_a_vs_b": full_ab,
        "trajectory_pure_vs_bt_default_override": full_default,
        "restart_vs_original_continuation": restart_comparison,
        "noise_floor": {
            "deterministic": deterministic,
            "observed_repeat_damage_range": float(repeat_noise),
            "epsilon": epsilon,
            "large_regression_threshold": max(1e-6, 10.0 * epsilon),
        },
        "pure_bt": {
            "dll_sha256": EXPECTED_PURE_DLL_SHA256,
            "xml_sha256": EXPECTED_PURE_XML_SHA256,
        },
    }
    (output / "evaluation.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    if aggregate["status"] != "PREFIX_REPLAY_PARITY_PASSED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
