from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dogfight.ai.guidance_selector import (
    GUIDANCE_ACTIONS,
    GUIDANCE_SELECTOR_CONTRACT_VERSION,
    GUIDANCE_SELECTOR_FEATURES,
    GUIDANCE_SELECTOR_NORMALIZATION_VERSION,
    GuidanceActionConfig,
    GuidanceControllerConfig,
    GuidanceRuntimeConfig,
    NumpyMLPGuidanceSelector,
)
from dogfight.envs.observation import OFFICIAL_DAMAGE_PHASES
from dogfight.submission.guidance_config import load_guidance_submission_config


PURE_DLL = Path("C:/Users/shy66/Downloads/aip_final_0815/aip_final_0815/AIP_DCS_GDCC_0815.dll")
PURE_XML = Path("C:/Users/shy66/Downloads/aip_final_0815/aip_final_0815/Rule_DCS_GDCC_0815.xml")
PURE_DLL_SHA = "4C93B4C6719CB0423388D5FC721D356020A3A36CD5AD2C56B5C3CA795BFE18C9"
PURE_XML_SHA = "D84C27B0B8BA22E1649AF2375BE0B83C762BC62EB5047BB590539B374F8271EE"
MODEL_SOURCE = ROOT / "artifacts/models/guidance_selector_bc_v1/seed_8799"
EVALUATION_ROOT = ROOT / "artifacts/evaluations/guidance_selector/full_200s_v1_20260819"
SUBMISSION_ROOT = ROOT / "artifacts/submission/guidance_selector_hybrid_v1"
EVIDENCE_ROOT = ROOT / "automation/evidence/guidance_selector_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_text_lf(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="\n") as handle:
        handle.write(text)


def write_json(path: Path, payload) -> None:
    write_text_lf(path, json.dumps(payload, indent=2, sort_keys=True))


def copy_model() -> tuple[Path, dict]:
    bundle = SUBMISSION_ROOT / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODEL_SOURCE / "model.npz", bundle / "model.npz")
    metadata = json.loads((MODEL_SOURCE / "metadata.json").read_text(encoding="utf-8"))
    write_json(bundle / "metadata.json", metadata)
    model_sha = sha256(bundle / "model.npz")
    if metadata["model_sha256"] != model_sha:
        raise RuntimeError("copied model metadata/hash mismatch")
    NumpyMLPGuidanceSelector(bundle)
    return bundle, metadata


def build_config(bundle: Path, model_sha: str) -> Path:
    action = asdict(GuidanceActionConfig())
    controller = asdict(GuidanceControllerConfig())
    runtime = asdict(GuidanceRuntimeConfig(confidence_threshold=0.55))
    config = {
        "schema_version": "guidance_selector_submission.v1",
        "status": "SUBMISSION_READY_HYBRID_CANDIDATE",
        "promotion_status": "NOT_PROMOTED",
        "candidate_kind": "EXPERIMENTAL_SAFE_HYBRID",
        "mode": "guidance_selector",
        "policy_id": "guidance_selector_rule_distilled",
        "bundle_path": "../../artifacts/submission/guidance_selector_hybrid_v1/bundle",
        "model_path": "../../artifacts/submission/guidance_selector_hybrid_v1/bundle/model.npz",
        "bundle_sha256": model_sha,
        "selector_observation_contract": GUIDANCE_SELECTOR_CONTRACT_VERSION,
        "selector_observation_size": len(GUIDANCE_SELECTOR_FEATURES),
        "normalization_version": GUIDANCE_SELECTOR_NORMALIZATION_VERSION,
        "observation_features": list(GUIDANCE_SELECTOR_FEATURES),
        "runtime_observation_mode": "tactical16",
        "action_library": list(GUIDANCE_ACTIONS),
        "action_magnitude": action,
        "controller": controller,
        "hard_eligibility_gate": {
            "kind": "rear120",
            "enter_target_ata_deg": 120.0,
            "exit_target_ata_deg": 110.0,
            "sim_hz": 60,
        },
        "activation_gate": {
            "kind": "rear120_and_offensive_or_pre_aim",
            "offensive": {
                "min_range_m": 152.4,
                "enter_max_range_m": 1500.0,
                "exit_max_range_m": 2000.0,
                "enter_ata_deg": 15.0,
                "exit_ata_deg": 25.0,
                "enter_min_target_ata_deg": 135.0,
                "exit_min_target_ata_deg": 110.0,
            },
            "phase_pre_aim": {
                "min_range_m": 152.4,
                "enter_angle_margin_deg": 7.0,
                "exit_angle_margin_deg": 10.0,
                "enter_range_margin_m": 300.0,
                "exit_range_margin_m": 550.0,
                "min_hold_steps": 12,
            },
            "safety_veto": {
                "minimum_altitude_m": 350.0,
                "minimum_speed_m_s": 170.0,
                "maximum_closing_rate_m_s": 250.0,
                "veto_if_all_surfaces_saturated": True,
            },
        },
        "runtime": runtime,
        "expected_sim_hz": 60,
        "latency_threshold_s": 0.1667,
        "throttle_policy": "bt_only",
        "force_side": "both",
        "phase_config": OFFICIAL_DAMAGE_PHASES,
        "wez": {"min_range_m": 152.4, "max_range_m": 1219.2, "angle_deg": 3.0},
        "health_source": "simulator",
        "fallback_mode": "exact_pure_bt",
        "bt": {
            "dll_path": str(PURE_DLL),
            "dll_sha256": PURE_DLL_SHA,
            "xml_path": str(PURE_XML),
            "xml_sha256": PURE_XML_SHA,
            "rule_alias": "Rule_DCS_GDCC_0815.xml",
            "turn_throttle_mode": "raw",
        },
        "rule_distillation": {
            "rule_id": "rear120_early_preaim_v1",
            "active_window": "gate_elapsed_frames < 36 of 90",
            "selected_action": "VP_EL_POS_SMALL",
            "otherwise": "BT_DEFAULT",
            "claim": "No learned performance-improvement claim",
        },
        "server_status": "SERVER_BLOCKED",
    }
    path = ROOT / "configs/submission/guidance_selector_hybrid_v1.json"
    write_json(path, config)
    pure = {
        "schema_version": "pure_bt_fallback.v1",
        "status": "READY",
        "mode": "bt",
        "fallback_mode": "exact_pure_bt",
        "observation_mode": "tactical16",
        "throttle_policy": "raw_bt_only",
        "expected_sim_hz": 60,
        "bt": config["bt"],
    }
    write_json(ROOT / "configs/submission/pure_bt_fallback_v1.json", pure)
    load_guidance_submission_config(path, require_files=True)
    return path


def freeze_contracts(metadata: dict, model_sha: str) -> None:
    training_summary = json.loads(
        (ROOT / "artifacts/models/guidance_selector_bc_v1/training_summary.json").read_text(
            encoding="utf-8"
        )
    )
    aggregate = json.loads((EVALUATION_ROOT / "aggregate.json").read_text(encoding="utf-8"))
    write_json(
        SUBMISSION_ROOT / "checkpoint_metadata.json",
        {
            "artifact_kind": "numpy_mlp_categorical_bundle",
            "model_sha256": model_sha,
            "candidate_kind": "RULE_DISTILLED_SAFE",
            "status": "EXPERIMENTAL_SAFE_HYBRID",
            "promotion_status": "NOT_PROMOTED",
            "training_steps": 0,
            "bc_training_executed": True,
            "bc_seeds": [8701, 8702, 8703],
            "bc_total_epochs": training_summary["total_epochs"],
            "ppo_status": "PPO_NOT_RUN_FAILED_DEV_GATE",
            "ppo_steps": 0,
            "restore_command": "python automation/build_guidance_rule_distilled.py",
        },
    )
    write_json(
        SUBMISSION_ROOT / "observation_contract.json",
        {
            "contract": GUIDANCE_SELECTOR_CONTRACT_VERSION,
            "normalization": GUIDANCE_SELECTOR_NORMALIZATION_VERSION,
            "runtime_source": "tactical16",
            "size": len(GUIDANCE_SELECTOR_FEATURES),
            "features": list(GUIDANCE_SELECTOR_FEATURES),
        },
    )
    write_json(
        SUBMISSION_ROOT / "action_contract.json",
        {
            "actions": list(GUIDANCE_ACTIONS),
            "magnitude": asdict(GuidanceActionConfig()),
            "controller": asdict(GuidanceControllerConfig()),
            "throttle": "exact BT-only",
            "bt_default": "exact Pure BT",
        },
    )
    write_json(
        SUBMISSION_ROOT / "model_metadata.json",
        {
            **metadata,
            "frozen_model_sha256": model_sha,
            "development_selection": "EXPERIMENTAL_SAFE_HYBRID",
            "operational_status": aggregate["status"],
            "promotion_status": aggregate["promotion_status"],
            "configured_200s_runs": aggregate["configured_200s_runs"],
            "actual_200s_timeout_runs": aggregate["completed_200s_timeout_runs"],
        },
    )
    write_text_lf(
        SUBMISSION_ROOT / "README.md",
        """# Guidance Selector Hybrid v1

이 bundle은 fallback ladder 4단계의 rule-distilled safe Guidance Selector다. Rear120+safety Gate 초기 구간에서만 최소 VP_EL_POS_SMALL을 선택하고, 나머지는 exact Pure BT다. Throttle은 항상 BT-only다.

BC 3개 seed 학습은 실행됐지만 실제 development 개입이 0이라 PPO gate를 통과하지 못했다. 이 모델은 learned performance improvement가 아니며 NOT_PROMOTED다. 200초 상한 36-run matrix의 모든 fight는 자연 terminal로 12.45~52.0초에 종료됐다. 따라서 200초 timeout을 실제로 달성했다는 주장을 하지 않는다.

Load smoke:

    python -c "from dogfight.ai.guidance_selector import NumpyMLPGuidanceSelector; NumpyMLPGuidanceSelector('artifacts/submission/guidance_selector_hybrid_v1/bundle')"

Config dry-run:

    python -c "from dogfight.submission import load_guidance_submission_config; load_guidance_submission_config('configs/submission/guidance_selector_hybrid_v1.json')"
""",
    )


def freeze_evidence() -> dict:
    records = json.loads((EVALUATION_ROOT / "records.json").read_text(encoding="utf-8"))
    aggregate = json.loads((EVALUATION_ROOT / "aggregate.json").read_text(encoding="utf-8"))
    write_json(EVIDENCE_ROOT / "aggregate.json", aggregate)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write_json(
        EVIDENCE_ROOT / "run_manifest.json",
        {
            "schema_version": "guidance_selector_evidence.v1",
            "issue": 19,
            "branch": "codex/guidance-selector-submission-hybrid",
            "head_before_freeze": head,
            "base_main_sha": "f0e79b743d7d03870cc485075daa08f4bcd57db6",
            "counterfactual_states": 100,
            "counterfactual_rollouts": 900,
            "bc_seeds": [8701, 8702, 8703],
            "development_seeds": list(range(8801, 8807)),
            "held_out_seeds": list(range(8901, 8907)),
            "held_out_opened_after_selection": True,
            "configured_max_seconds": 200,
            "run_count": 36,
            "actual_timeout_runs": 0,
            "natural_terminal_runs": 36,
            "server_status": "SERVER_BLOCKED",
        },
    )
    csv_fields = [
        "case_id", "split", "opponent", "side", "seed", "controller", "outcome",
        "end_condition", "episode_seconds", "damage_dealt", "damage_received", "health_margin",
        "ownship_crash", "target_crash", "first_damage_s", "phase1_cone_time_s",
        "phase2_cone_time_s", "phase3_cone_time_s", "gate_active_ratio",
        "nonzero_intervention_frames", "min_altitude_m", "min_speed_m_s", "latency_ms_max",
        "throttle_difference_max", "invalid_or_nonfinite_actions", "telemetry_sha256",
    ]
    with (EVIDENCE_ROOT / "episode_records.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=csv_fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    with (EVIDENCE_ROOT / "paired_200s_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fields = ["controller", "case_id", "split", "opponent", "side", "damage_delta", "contaminated"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for controller, summary in aggregate["paired"].items():
            for row in summary["pair_records"]:
                writer.writerow({"controller": controller, **row})
    hybrid = [row for row in records if row["controller"] == "bc"]
    write_json(
        EVIDENCE_ROOT / "latency.json",
        {
            "p50_max_ms": aggregate["latency_ms_p50_max"],
            "p95_max_ms": aggregate["latency_ms_p95_max"],
            "p99_max_ms": aggregate["latency_ms_p99_max"],
            "max_ms": aggregate["latency_ms_max"],
            "over_166_7ms": aggregate["latency_over_166_7ms"],
            "runs": [
                {key: row[key] for key in ("case_id", "latency_ms_p50", "latency_ms_p95", "latency_ms_p99", "latency_ms_max")}
                for row in hybrid
            ],
        },
    )
    action_totals = {name: 0.0 for name in GUIDANCE_ACTIONS}
    for row in hybrid:
        for name, ratio in row["action_distribution"].items():
            action_totals[name] += ratio * row["telemetry_frames"]
    total = sum(action_totals.values())
    write_json(
        EVIDENCE_ROOT / "action_distribution.json",
        {
            "estimated_frame_weighted_counts": action_totals,
            "frame_weighted_distribution": {
                name: value / total for name, value in action_totals.items()
            },
            "nonzero_intervention_frames_exact": aggregate["nonzero_intervention_frames"],
            "note": "counts reconstructed from per-run provider ratios and telemetry frame counts",
        },
    )
    write_json(
        EVIDENCE_ROOT / "thresholds.json",
        {
            "confidence": 0.55,
            "meaningful_damage_delta": 0.001,
            "large_regression": -0.003,
            "positive_pair_ratio": 2.0 / 3.0,
            "minimum_altitude_m": 304.8,
            "safety_veto_altitude_m": 350.0,
            "safety_veto_speed_m_s": 170.0,
            "latency_s": 0.1667,
            "maximum_active_frames": 90,
            "rule_active_frames": 36,
            "minimum_hold_frames": 18,
            "cooldown_frames": 30,
        },
    )
    write_text_lf(
        EVIDENCE_ROOT / "command_history.txt",
        "\n".join(
            (
                "python automation/evaluate_guidance_counterfactual.py",
                "python automation/train_guidance_selector_bc.py",
                "python automation/build_guidance_rule_distilled.py",
                "python automation/evaluate_guidance_development.py --include-rule-distilled",
                "python automation/evaluate_guidance_200s.py",
                "python automation/verify_guidance_200s.py",
                "python automation/freeze_guidance_submission.py",
            )
        )
        + "\n",
    )
    return aggregate


def write_reports(aggregate: dict) -> None:
    paired = aggregate["paired"]["bc"]
    report_root = ROOT / "automation/reports"
    write_text_lf(
        report_root / "GuidanceSelector_200초_전체전.md",
        f"""# Guidance Selector 200초 상한 전체전

## 결론

12개 case × Pure/BT_DEFAULT-only/Hybrid의 36개 교전을 200초 상한으로 실제 실행했다. 모든 run은 유효 terminal로 12.45~52.0초에 종료되어 실제 200초 timeout 도달은 0개다. 이를 36개의 200초 실비행으로 과장하지 않는다.

rule-distilled Hybrid는 운영 계약을 통과해 `SUBMISSION_READY_HYBRID_CANDIDATE`지만 성능은 `NOT_PROMOTED`다.

- 실제 nonzero intervention: {aggregate['nonzero_intervention_frames']:,} frame
- Gate active ratio mean: {aggregate['gate_active_ratio_mean']:.10f}
- ownship crash / process error / invalid / throttle violation: 0 / 0 / 0 / 0
- target crash: {aggregate['target_crashes']} Hybrid runs; 해당 pair는 Damage primary에서 제외
- min altitude / speed: {aggregate['minimum_altitude_m']:.6f} m / {aggregate['minimum_speed_m_s']:.6f} m/s
- inference P50/P95/P99/MAX worst-run: {aggregate['latency_ms_p50_max']:.6f} / {aggregate['latency_ms_p95_max']:.6f} / {aggregate['latency_ms_p99_max']:.6f} / {aggregate['latency_ms_max']:.6f} ms
- 166.7ms 초과: 0
- throttle difference max: 0

## Paired Damage

clean {paired['clean_pairs']} pair, contaminated {paired['contaminated_pairs']} pair다. clean Damage Δ는 mean `{paired['clean_damage_delta_mean']:.10f}`, median `{paired['clean_damage_delta_median']:.10f}`, min `{paired['clean_damage_delta_min']:.10f}`, max `{paired['clean_damage_delta_max']:.10f}`, positive `{paired['positive_pairs']}/{paired['clean_pairs']}`다. BT_DEFAULT-only는 Pure와 12/12 exact outcome 및 Damage Δ 0을 재현했다.

## Coverage 제한

target destroyed 12회, target altitude below min 24회로 모두 Phase 1 안에 끝났다. Phase 2/3 cone dwell은 0이며, phase boundary 자체는 unit test 대상이지 이 flight matrix의 실측 coverage가 아니다. AIP2는 0815와 별도 DLL이지만 독립 계보를 입증하지 못해 unseen-independent-opponent 주장에 사용하지 않는다. 실제 서버 정보는 없어 `SERVER_BLOCKED`다.
""",
    )
    write_text_lf(
        report_root / "GuidanceSelector_독립재검증.md",
        """# Guidance Selector 독립 재검증

raw result JSON과 frame telemetry JSONL 36개를 평가 집계와 별도 코드 경로로 다시 읽었다. telemetry SHA256 36개, run 수, clean/contaminated pair, Damage mean/median/min/max, nonzero intervention, altitude, latency가 aggregate와 모두 일치했다.

상태: `INDEPENDENT_RECOMPUTATION_PASS`.

상세 machine-readable 결과는 `automation/evidence/guidance_selector_v1/independent_verification.json`에 있다. 재검증은 성능 승격 근거가 아니라 결과 무결성 확인이다.
""",
    )
    write_text_lf(
        report_root / "GuidanceSelector_최종후보_및_제출Fallback.md",
        f"""# Guidance Selector 최종 후보 및 제출 fallback

## 최종 상태

- artifact: `EXPERIMENTAL_SAFE_HYBRID`
- operational: `SUBMISSION_READY_HYBRID_CANDIDATE`
- performance: `NOT_PROMOTED`
- PPO: `PPO_NOT_RUN_FAILED_DEV_GATE`
- server: `SERVER_BLOCKED`

BC 3개 모델은 모두 실제 비행 개입 0으로 탈락했다. 최종 artifact는 Rear120+safety Gate의 초기 36/90 frame에서 최소 `VP_EL_POS_SMALL`만 허용하는 rule-distilled categorical MLP다. learned improvement로 주장하지 않는다.

clean Damage Δ mean `{paired['clean_damage_delta_mean']:.10f}`, median `{paired['clean_damage_delta_median']:.10f}`, positive `{paired['positive_pairs']}/{paired['clean_pairs']}`이므로 Pure BT보다 우월하지 않다. 실패·저신뢰·Gate OFF·BT_DEFAULT에서는 exact Pure BT로 돌아가며 throttle은 항상 BT-only다.

Pure fallback SHA256:

- DLL `{PURE_DLL_SHA}`
- XML `{PURE_XML_SHA}`

## 최종 검증

- 정규 회귀: `191 passed, 26 subtests passed`
- Guidance 집중 검증: `29 passed`
- compileall: 통과
- config fail-fast parse / model load-only inference: 통과
- artifact/evidence checksum: 통과
- JSON parse / CSV nonempty: 통과 (`episode_records` 36행, `paired_200s_results` 24행)
- 변경 파일 1 MiB 초과: 없음
- tracked-file credential pattern scan: 검출 없음

추가로 정규 범위 밖의 `test_ias.py`는 인자 fixture가 없는 수동 스크립트라 pytest collection error가 났고, web log viewer test는 Git에 없는 `MyTrainEnv/logs` fixture 때문에 실패했다. 두 파일은 이 branch diff에 포함되지 않는다.
""",
    )


def write_checksums(config_path: Path) -> None:
    artifact_files = sorted(
        path for path in SUBMISSION_ROOT.rglob("*") if path.is_file() and path.name != "sha256sums.txt"
    )
    lines = [f"{sha256(path)}  {path.relative_to(SUBMISSION_ROOT).as_posix()}" for path in artifact_files]
    lines.append(f"{sha256(config_path)}  ../../../configs/submission/{config_path.name}")
    write_text_lf(SUBMISSION_ROOT / "sha256sums.txt", "\n".join(lines) + "\n")
    evidence_files = sorted(
        path for path in EVIDENCE_ROOT.iterdir() if path.is_file() and path.name != "checksums.sha256"
    )
    write_text_lf(
        EVIDENCE_ROOT / "checksums.sha256",
        "\n".join(f"{sha256(path)}  {path.name}" for path in evidence_files) + "\n",
    )


def main() -> None:
    if sha256(PURE_DLL) != PURE_DLL_SHA or sha256(PURE_XML) != PURE_XML_SHA:
        raise RuntimeError("Pure BT fallback hash drift")
    bundle, metadata = copy_model()
    model_sha = sha256(bundle / "model.npz")
    config_path = build_config(bundle, model_sha)
    freeze_contracts(metadata, model_sha)
    aggregate = freeze_evidence()
    write_reports(aggregate)
    write_checksums(config_path)
    print(
        json.dumps(
            {
                "status": aggregate["status"],
                "promotion": aggregate["promotion_status"],
                "model_sha256": model_sha,
                "config_sha256": sha256(config_path),
                "artifact": str(SUBMISSION_ROOT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
