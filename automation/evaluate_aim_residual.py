"""0815 Pure BT와 조준 잔차 Hybrid를 동일 조건으로 paired 평가한다."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SCALES = (0.10, 0.125, 0.15)
MANEUVER_METRICS = (
    "mean_los_deg",
    "median_los_deg",
    "p95_los_deg",
    "min_los_deg",
    "los_rate_rms_deg_s",
    "mean_ata_deg",
    "min_ata_deg",
    "damage_cone_entries",
    "damage_cone_time_s",
    "phase1_cone_time_s",
    "phase2_cone_time_s",
    "phase3_cone_time_s",
    "time_to_first_wez_s",
    "time_to_first_damage_s",
    "mean_speed_m_s",
    "min_speed_m_s",
    "min_altitude_m",
    *(
        f"{source}_{axis}_{metric}"
        for source in ("bt", "final")
        for axis in ("roll", "pitch", "yaw")
        for metric in (
            "saturation_ratio",
            "positive_headroom_mean",
            "negative_headroom_mean",
        )
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    bundle = Path(args.bundle).resolve()
    metadata = bundle / "metadata.json"
    weights = bundle / "policy_weights.pkl.gz"
    required = [metadata, weights, Path(args.ownship_bt_dll), Path(args.bt_rule_xml)]
    if args.target_backend == "bt":
        required.append(Path(args.target_bt_dll))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"필수 파일 누락: {missing}")
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    obs_mode = (
        payload.get("metadata", {}).get("obs_mode")
        or payload.get("algorithm_config", {}).get("env_config", {}).get("observation_mode")
    )
    if obs_mode not in (
        "aim_residual10",
        "aim_residual10_v2",
        "aim_residual13_btaware",
    ):
        raise ValueError(f"bundle observation 불일치: {obs_mode!r}")
    args.observation_mode = obs_mode
    invalid = [scale for scale in args.scales if scale not in ALLOWED_SCALES]
    if invalid:
        raise ValueError(f"잔차 강도는 {ALLOWED_SCALES} 중 하나여야 함: {invalid}")
    result = {
        "bundle": str(bundle),
        "bundle_metadata_sha256": sha256(metadata),
        "bundle_weights_sha256": sha256(weights),
        "ownship_bt_dll": str(Path(args.ownship_bt_dll).resolve()),
        "ownship_bt_dll_sha256": sha256(Path(args.ownship_bt_dll)),
        "bt_rule_xml": str(Path(args.bt_rule_xml).resolve()),
        "bt_rule_xml_sha256": sha256(Path(args.bt_rule_xml)),
        "observation_mode": obs_mode,
    }
    if args.target_backend == "bt":
        result.update(
            {
                "target_bt_dll": str(Path(args.target_bt_dll).resolve()),
                "target_bt_dll_sha256": sha256(Path(args.target_bt_dll)),
            }
        )
    return result


def controller_specs(scales: list[float]) -> list[tuple[str, float | None]]:
    return [("pure_0815", None)] + [
        (f"hybrid_{scale:g}", scale) for scale in scales
    ]


def controller_observation_mode(bundle_mode: str, controller: str) -> str:
    """Keep Pure BT observation-free when evaluating a BT-aware policy."""
    if controller == "pure_0815" and bundle_mode == "aim_residual13_btaware":
        return "aim_residual10_v2"
    return bundle_mode


def run_match(
    args: argparse.Namespace,
    output: Path,
    *,
    seed: int,
    controller: str,
    scale: float | None,
) -> dict[str, Any]:
    run_id = f"seed{seed}_{controller}_{args.gate_kind}"
    result_path = output / "summaries" / f"{run_id}.json"
    telemetry_path = output / "telemetry" / f"{run_id}.jsonl"
    stdout_path = output / "raw" / f"{run_id}.stdout.txt"
    stderr_path = output / "raw" / f"{run_id}.stderr.txt"
    if args.resume and result_path.is_file():
        return build_record(
            run_id,
            seed,
            controller,
            scale,
            json.loads(result_path.read_text(encoding="utf-8")),
            returncode=0,
            wall_seconds=0.0,
            resumed=True,
        )

    cmd = [
        sys.executable,
        str(ROOT / "run_local_dogfight.py"),
        "--ownship-backend", "bt" if controller == "pure_0815" else "residual_hybrid",
        "--target-backend", args.target_backend,
        "--ownship-bt-dll", str(Path(args.ownship_bt_dll).resolve()),
        "--target-bt-dll", str(Path(args.target_bt_dll).resolve()),
        "--bt-rule-xml", str(Path(args.bt_rule_xml).resolve()),
        "--bt-rule-alias-only",
        "--bt-turn-throttle-mode", "raw",
        "--observation-mode",
        controller_observation_mode(args.observation_mode, controller),
        "--scenario-file", str(Path(args.scenario).resolve()),
        "--seed", str(seed),
        "--max-engage-time", str(args.max_engage_time),
        "--episode-step-limit", str(args.episode_step_limit),
        "--result-json", str(result_path),
        "--telemetry-jsonl", str(telemetry_path),
    ]
    for alias in args.bt_rule_alias:
        cmd += ["--bt-rule-alias", alias]
    if controller != "pure_0815":
        cmd += [
            "--ownship-bundle-dir", str(Path(args.bundle).resolve()),
            "--residual-gate", args.gate_kind,
            "--residual-composition", args.composition_mode,
            "--residual-scale", str(scale),
            "--rl-action-repeat", str(args.rl_action_repeat),
            "--aim-min-range-m", str(args.aim_min_range_m),
            "--aim-enter-angle-margin-deg", str(args.aim_enter_angle_margin_deg),
            "--aim-exit-angle-margin-deg", str(args.aim_exit_angle_margin_deg),
            "--aim-enter-range-margin-m", str(args.aim_enter_range_margin_m),
            "--aim-exit-range-margin-m", str(args.aim_exit_range_margin_m),
            "--aim-min-hold-steps", str(args.aim_min_hold_steps),
            "--offensive-enter-ata-deg", str(args.offensive_enter_ata_deg),
            "--offensive-exit-ata-deg", str(args.offensive_exit_ata_deg),
            "--offensive-enter-target-ata-deg", str(args.offensive_enter_target_ata_deg),
            "--offensive-exit-target-ata-deg", str(args.offensive_exit_target_ata_deg),
            "--rear120-enter-target-ata-deg", str(args.rear120_enter_target_ata_deg),
            "--rear120-exit-target-ata-deg", str(args.rear120_exit_target_ata_deg),
            "--safety-minimum-altitude-m", str(args.safety_minimum_altitude_m),
            "--safety-minimum-speed-m-s", str(args.safety_minimum_speed_m_s),
            "--safety-maximum-closing-rate-m-s", str(args.safety_maximum_closing_rate_m_s),
        ]

    protected = ROOT / "aircraft" / "f16" / "f16_init.xml"
    protected_bytes = protected.read_bytes()
    started = time.monotonic()
    try:
        process = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
        returncode = process.returncode
        stdout, stderr = process.stdout, process.stderr
    except subprocess.TimeoutExpired as error:
        returncode = 124
        stdout, stderr = error.stdout or "", error.stderr or ""
    finally:
        if protected.read_bytes() != protected_bytes:
            protected.write_bytes(protected_bytes)
    wall_seconds = time.monotonic() - started
    stdout_path.write_text(str(stdout), encoding="utf-8")
    stderr_path.write_text(str(stderr), encoding="utf-8")
    result = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.is_file()
        else {}
    )
    return build_record(
        run_id,
        seed,
        controller,
        scale,
        result,
        returncode=returncode,
        wall_seconds=wall_seconds,
        resumed=False,
    )


def build_record(
    run_id: str,
    seed: int,
    controller: str,
    scale: float | None,
    result: dict[str, Any],
    *,
    returncode: int,
    wall_seconds: float,
    resumed: bool,
) -> dict[str, Any]:
    maneuver = result.get("maneuver_telemetry", {}) or {}
    provider = result.get("ownship_provider_telemetry", {}) or {}
    own_health = finite(result.get("ownship_health"))
    target_health = finite(result.get("target_health"))
    record = {
        "run_id": run_id,
        "seed": seed,
        "variant_index": finite(result.get("aim_curriculum_variant_index")),
        "variant_name": result.get("aim_curriculum_variant_name"),
        "controller": controller,
        "scale": scale,
        "gate_kind": provider.get("residual_inference_gate_kind"),
        "outcome": "process_error" if returncode else result.get("outcome", "unknown"),
        "end_condition": result.get("end_condition", ""),
        "ownship_crash": bool(result.get("ownship_crash", False)),
        "target_crash": bool(result.get("target_crash", False)),
        "returncode": returncode,
        "resumed": resumed,
        "wall_seconds": round(wall_seconds, 3),
        "episode_seconds": finite(result.get("episode_seconds")),
        "ownship_health": own_health,
        "target_health": target_health,
        "damage_dealt": None if target_health is None else 1.0 - target_health,
        "damage_received": None if own_health is None else 1.0 - own_health,
        "health_margin": (
            own_health - target_health
            if own_health is not None and target_health is not None
            else None
        ),
        "gate_active_ratio": finite(provider.get(f"{provider.get('residual_inference_gate_kind')}_gate_active_ratio")) or 0.0,
        "gate_entries": finite(provider.get(f"{provider.get('residual_inference_gate_kind')}_gate_entries")) or 0.0,
        "gate_exits": finite(provider.get(f"{provider.get('residual_inference_gate_kind')}_gate_exits")) or 0.0,
        "gate_mean_active_steps": finite(provider.get(f"{provider.get('residual_inference_gate_kind')}_gate_mean_active_steps")) or 0.0,
        "rl_inference_calls": finite(provider.get("rl_inference_calls")) or 0.0,
        "rl_correction_steps": finite(provider.get("rl_correction_steps")) or 0.0,
        "correction_roll_mean": _axis(provider, "rl_correction_abs_mean", 0),
        "correction_pitch_mean": _axis(provider, "rl_correction_abs_mean", 1),
        "correction_yaw_mean": _axis(provider, "rl_correction_abs_mean", 2),
        "requested_roll_correction_mean": _axis(
            provider, "requested_surface_correction_abs_mean_axis", 0
        ),
        "requested_pitch_correction_mean": _axis(
            provider, "requested_surface_correction_abs_mean_axis", 1
        ),
        "requested_yaw_correction_mean": _axis(
            provider, "requested_surface_correction_abs_mean_axis", 2
        ),
        "roll_applied_to_requested_ratio": _axis(
            provider, "applied_to_requested_ratio_mean_axis", 0
        ),
        "pitch_applied_to_requested_ratio": _axis(
            provider, "applied_to_requested_ratio_mean_axis", 1
        ),
        "yaw_applied_to_requested_ratio": _axis(
            provider, "applied_to_requested_ratio_mean_axis", 2
        ),
        "action_clipped_steps": finite(provider.get("action_clipped_steps")) or 0.0,
        "action_saturated_steps": finite(provider.get("action_saturated_steps")) or 0.0,
        "action_clipped_ratio": (
            (finite(provider.get("action_clipped_steps")) or 0.0)
            / max(1.0, finite(provider.get("rl_correction_steps")) or 0.0)
        ),
        "action_saturated_ratio": (
            (finite(provider.get("action_saturated_steps")) or 0.0)
            / max(1.0, finite(provider.get("rl_correction_steps")) or 0.0)
        ),
        "inference_latency_ms_p50": finite(provider.get("rl_inference_latency_ms_p50")),
        "inference_latency_ms_p95": finite(provider.get("rl_inference_latency_ms_p95")),
        "inference_latency_ms_p99": finite(provider.get("rl_inference_latency_ms_p99")),
        "inference_latency_ms_max": finite(provider.get("rl_inference_latency_ms_max")),
    }
    record.update({metric: finite(maneuver.get(metric)) for metric in MANEUVER_METRICS})
    return record


def _axis(provider: dict[str, Any], key: str, index: int) -> float | None:
    values = provider.get(key)
    return finite(values[index]) if isinstance(values, list) and len(values) > index else None


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    controllers = sorted({row["controller"] for row in records})
    metrics = (
        "episode_seconds", "damage_dealt", "damage_received", "health_margin",
        "mean_los_deg", "median_los_deg", "p95_los_deg", "los_rate_rms_deg_s",
        "damage_cone_time_s", "time_to_first_wez_s", "time_to_first_damage_s",
        "mean_speed_m_s", "min_speed_m_s", "min_altitude_m", "gate_active_ratio",
        "rl_inference_calls", "correction_roll_mean", "correction_pitch_mean",
        "correction_yaw_mean", "action_clipped_steps", "action_saturated_steps",
        "requested_roll_correction_mean", "requested_pitch_correction_mean",
        "requested_yaw_correction_mean", "roll_applied_to_requested_ratio",
        "pitch_applied_to_requested_ratio", "yaw_applied_to_requested_ratio",
        "action_clipped_ratio", "action_saturated_ratio",
        "inference_latency_ms_p50", "inference_latency_ms_p95",
        "inference_latency_ms_p99", "inference_latency_ms_max",
        *MANEUVER_METRICS[-18:],
    )
    by_controller: dict[str, Any] = {}
    for controller in controllers:
        rows = [row for row in records if row["controller"] == controller]
        by_controller[controller] = {
            "episodes": len(rows),
            "unique_result_signatures": len({_result_signature(row) for row in rows}),
            "wins": sum(row["outcome"] == "win" for row in rows),
            "ownship_crashes": sum(row["ownship_crash"] for row in rows),
            "target_crashes": sum(row["target_crash"] for row in rows),
            "timeouts": sum(row["outcome"] == "timeout" for row in rows),
            **{metric: mean(rows, metric) for metric in metrics},
        }

    paired = {}
    pure_by_seed = {
        row["seed"]: row for row in records if row["controller"] == "pure_0815"
    }
    for controller in controllers:
        if controller == "pure_0815":
            continue
        deltas: dict[str, list[float]] = {metric: [] for metric in metrics}
        for row in records:
            if row["controller"] != controller or row["seed"] not in pure_by_seed:
                continue
            pure = pure_by_seed[row["seed"]]
            for metric in metrics:
                if row.get(metric) is not None and pure.get(metric) is not None:
                    deltas[metric].append(float(row[metric]) - float(pure[metric]))
        paired[controller] = {
            "pairs": len([row for row in records if row["controller"] == controller]),
            "delta_hybrid_minus_pure": {
                metric: (sum(values) / len(values) if values else None)
                for metric, values in deltas.items()
            },
            "per_seed": deltas,
            "unique_delta_signatures": len(
                {
                    tuple(
                        None if values[index] is None else round(values[index], 12)
                        for values in deltas.values()
                        if index < len(values)
                    )
                    for index in range(max((len(values) for values in deltas.values()), default=0))
                }
            ),
        }
    warnings = []
    for controller, values in by_controller.items():
        if values["unique_result_signatures"] < values["episodes"]:
            warnings.append(
                f"{controller}: {values['episodes']}회 중 고유 결과는 "
                f"{values['unique_result_signatures']}개이므로 반복 seed를 독립 근거로 세지 않음"
            )
    pure_variants = {
        row["seed"]: row.get("variant_name")
        for row in records
        if row["controller"] == "pure_0815"
    }
    for row in records:
        if row["controller"] == "pure_0815":
            continue
        if pure_variants.get(row["seed"]) != row.get("variant_name"):
            warnings.append(
                f"seed {row['seed']}: Pure/Hybrid variant 불일치 "
                f"({pure_variants.get(row['seed'])!r} != {row.get('variant_name')!r})"
            )
    return {
        "controllers": by_controller,
        "paired": paired,
        "data_quality_warnings": warnings,
    }


def _result_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    keys = (
        "outcome",
        "end_condition",
        "variant_name",
        "episode_seconds",
        "damage_dealt",
        "mean_los_deg",
        "los_rate_rms_deg_s",
        "damage_cone_time_s",
        "time_to_first_damage_s",
        "min_altitude_m",
    )
    return tuple(
        round(float(row[key]), 12) if isinstance(row.get(key), (int, float)) else row.get(key)
        for key in keys
    )


def write_outputs(
    output: Path,
    args: argparse.Namespace,
    preflight_result: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = aggregate(records)
    settings = dict(vars(args))
    settings["scenario"] = str(settings["scenario"])
    payload = {
        "preflight": preflight_result,
        "settings": settings,
        "metric_contract": {
            "source": "run_local_dogfight simulator-rate maneuver telemetry",
            "grain": "한 행은 controller/seed 한 episode",
            "paired_window": "동일 scenario와 seed의 episode 전체",
            "delta_semantics": "Hybrid - Pure 0815",
        },
        "summary": summary,
        "records": records,
    }
    (output / "evaluation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    if records:
        with (output / "matches.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    write_report(output / "report.md", payload)
    return payload


def _record_identity(record: dict[str, Any]) -> tuple[int, str]:
    """Return the episode identity used by a resumed paired evaluation."""
    return int(record["seed"]), str(record["controller"])


def merge_unique_records(
    existing: list[dict[str, Any]],
    additions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append only previously unseen seed/controller episodes.

    A resumed evaluation must preserve prior evidence rather than replacing it,
    while repeated invocations for the same seed must not inflate the sample.
    """
    merged = list(existing)
    identities = {_record_identity(record) for record in merged}
    for record in additions:
        identity = _record_identity(record)
        if identity in identities:
            continue
        merged.append(record)
        identities.add(identity)
    return merged


def load_resume_records(
    output: Path,
    args: argparse.Namespace,
    preflight_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Load compatible prior evidence for ``--resume`` aggregation."""
    evaluation_path = output / "evaluation.json"
    if not args.resume or not evaluation_path.exists():
        return []
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if payload.get("preflight") != preflight_result:
        raise RuntimeError(
            "resume evaluation preflight differs from the existing output; "
            "use a new output directory"
        )
    existing_settings = dict(payload.get("settings", {}))
    current_settings = dict(vars(args))
    current_settings["scenario"] = str(current_settings["scenario"])
    ignored = {"seeds", "resume", "quiet", "scales"}
    for key, current in current_settings.items():
        if key in ignored:
            continue
        if existing_settings.get(key) != current:
            raise RuntimeError(
                f"resume evaluation setting differs for {key!r}; "
                "use a new output directory"
            )
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise RuntimeError("resume evaluation records must be a list")
    return [dict(record) for record in records]


def fmt(value: Any, digits: int = 4) -> str:
    return "자료 없음" if value is None else f"{float(value):.{digits}f}"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    pure = summary["controllers"].get("pure_0815", {})
    lines = [
        "# 0815 조준 잔차 Paired 평가",
        "",
        "## 평가 계약",
        "",
        "- 분석 단위: 동일 scenario와 seed의 episode 한 쌍",
        "- 차이 정의: Hybrid - Pure 0815",
        "- 학습 episode는 평가 근거에 포함하지 않음",
        f"- Pure 0815 episode 수: `{pure.get('episodes', 0)}`",
        f"- Pure 0815 고유 결과 수: `{pure.get('unique_result_signatures', 0)}`",
        "",
        "## 기준선",
        "",
        f"- 평균 LOS: `{fmt(pure.get('mean_los_deg'))}°`",
        f"- LOS rate RMS: `{fmt(pure.get('los_rate_rms_deg_s'))}°/s`",
        f"- Damage Cone 체류: `{fmt(pure.get('damage_cone_time_s'))}초`",
        f"- 최초 Damage: `{fmt(pure.get('time_to_first_damage_s'))}초`",
        "",
        "## Paired 차이",
        "",
    ]
    for controller, comparison in summary["paired"].items():
        delta = comparison["delta_hybrid_minus_pure"]
        controller_metrics = summary["controllers"][controller]
        lines += [
            f"### {controller}",
            "",
            f"- Pair 수: `{comparison['pairs']}`",
            f"- 평균 LOS 차이: `{fmt(delta.get('mean_los_deg'))}°`",
            f"- LOS rate RMS 차이: `{fmt(delta.get('los_rate_rms_deg_s'))}°/s`",
            f"- Damage Cone 체류 차이: `{fmt(delta.get('damage_cone_time_s'))}초`",
            f"- 최초 Damage 시간 차이: `{fmt(delta.get('time_to_first_damage_s'))}초`",
            f"- Damage dealt 차이: `{fmt(delta.get('damage_dealt'))}`",
            f"- 최소 속도 차이: `{fmt(delta.get('min_speed_m_s'))}m/s`",
            f"- 최소 고도 차이: `{fmt(delta.get('min_altitude_m'))}m`",
            f"- Gate 활성 비율: `{fmt(controller_metrics.get('gate_active_ratio'))}`",
            f"- Roll/Pitch/Yaw 평균 보정: `"
            f"{fmt(controller_metrics.get('correction_roll_mean'))} / "
            f"{fmt(controller_metrics.get('correction_pitch_mean'))} / "
            f"{fmt(controller_metrics.get('correction_yaw_mean'))}`",
            f"- 활성 step 중 clipping 비율: `"
            f"{fmt(controller_metrics.get('action_clipped_ratio'))}`",
            f"- Roll/Pitch/Yaw 요청 대비 적용 비율: `"
            f"{fmt(controller_metrics.get('roll_applied_to_requested_ratio'))} / "
            f"{fmt(controller_metrics.get('pitch_applied_to_requested_ratio'))} / "
            f"{fmt(controller_metrics.get('yaw_applied_to_requested_ratio'))}`",
            f"- BT Roll/Pitch/Yaw 포화 비율: `"
            f"{fmt(controller_metrics.get('bt_roll_saturation_ratio'))} / "
            f"{fmt(controller_metrics.get('bt_pitch_saturation_ratio'))} / "
            f"{fmt(controller_metrics.get('bt_yaw_saturation_ratio'))}`",
            f"- RL 추론 P95: `{fmt(controller_metrics.get('inference_latency_ms_p95'))}ms`",
            "",
        ]
    lines += [
        "## 데이터 품질 경고",
        "",
        *(
            [f"- {warning}" for warning in summary["data_quality_warnings"]]
            or ["- 탐지된 중복 결과 없음"]
        ),
        "",
        "## 판단 제한",
        "",
        "10 pair 미만 결과는 smoke 근거이며 PROMOTE에 사용하지 않는다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario", type=Path, default=ROOT / "automation/scenarios/0815_aim_stage1.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1201, 1202, 1203])
    parser.add_argument("--scales", nargs="+", type=float, default=[0.125])
    parser.add_argument(
        "--gate-kind",
        choices=["aim", "offensive", "combined", "rear120"],
        default="aim",
    )
    parser.add_argument(
        "--composition-mode",
        choices=["additive", "saturation_aware"],
        default="additive",
    )
    parser.add_argument("--ownship-bt-dll", required=True)
    parser.add_argument("--target-backend", choices=["autopilot", "bt"], default="autopilot")
    parser.add_argument("--target-bt-dll", default=str(ROOT / "AIP_BASE_target.dll"))
    parser.add_argument("--bt-rule-xml", required=True)
    parser.add_argument("--bt-rule-alias", action="append", default=["Rule_DCS_GDCC_0815.xml"])
    parser.add_argument("--max-engage-time", type=float, default=30.0)
    parser.add_argument("--episode-step-limit", type=int, default=1800)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--rl-action-repeat", type=int, default=6)
    parser.add_argument("--aim-min-range-m", type=float, default=152.4)
    parser.add_argument("--aim-enter-angle-margin-deg", type=float, default=7.0)
    parser.add_argument("--aim-exit-angle-margin-deg", type=float, default=10.0)
    parser.add_argument("--aim-enter-range-margin-m", type=float, default=300.0)
    parser.add_argument("--aim-exit-range-margin-m", type=float, default=550.0)
    parser.add_argument("--aim-min-hold-steps", type=int, default=12)
    parser.add_argument("--offensive-enter-ata-deg", type=float, default=30.0)
    parser.add_argument("--offensive-exit-ata-deg", type=float, default=45.0)
    parser.add_argument("--offensive-enter-target-ata-deg", type=float, default=120.0)
    parser.add_argument("--offensive-exit-target-ata-deg", type=float, default=110.0)
    parser.add_argument("--rear120-enter-target-ata-deg", type=float, default=120.0)
    parser.add_argument("--rear120-exit-target-ata-deg", type=float, default=110.0)
    parser.add_argument("--safety-minimum-altitude-m", type=float, default=350.0)
    parser.add_argument("--safety-minimum-speed-m-s", type=float, default=170.0)
    parser.add_argument("--safety-maximum-closing-rate-m-s", type=float, default=250.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="요약 JSON stdout 출력을 생략한다.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    for name in ("summaries", "telemetry", "raw"):
        (output / name).mkdir(parents=True, exist_ok=True)
    preflight_result = preflight(args)
    records = load_resume_records(output, args, preflight_result)
    completed = {_record_identity(record) for record in records}
    for seed in args.seeds:
        for controller, scale in controller_specs(args.scales):
            if (int(seed), controller) in completed:
                print(
                    f"[paired-eval] skip existing seed={seed} "
                    f"controller={controller}",
                    flush=True,
                )
                continue
            print(f"[paired-eval] seed={seed} controller={controller}", flush=True)
            new_record = run_match(
                args,
                output,
                seed=seed,
                controller=controller,
                scale=scale,
            )
            records = merge_unique_records(records, [new_record])
            completed.add(_record_identity(new_record))
            write_outputs(output, args, preflight_result, records)
    payload = write_outputs(output, args, preflight_result, records)
    if not args.quiet:
        print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
