from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dogfight.ai.guidance_advantage import (
    GUIDANCE_ADVANTAGE_ACTIONS,
    GUIDANCE_SERVER_CONTRACT_VERSION,
    GUIDANCE_SERVER_FEATURES,
    GUIDANCE_SERVER_NORMALIZATION_VERSION,
    server_observation_contract,
)
from dogfight.ai.guidance_selector import (
    GuidanceActionConfig,
    GuidanceControllerConfig,
    GuidanceRuntimeConfig,
)
from dogfight.ai.state_action_advantage import load_guidance_selector_bundle
from dogfight.envs.observation import OFFICIAL_DAMAGE_PHASES
from dogfight.submission.guidance_config import load_guidance_submission_config


PURE_DLL = Path("C:/Users/shy66/Downloads/aip_final_0815/aip_final_0815/AIP_DCS_GDCC_0815.dll")
PURE_XML = Path("C:/Users/shy66/Downloads/aip_final_0815/aip_final_0815/Rule_DCS_GDCC_0815.xml")
PURE_DLL_SHA = "4C93B4C6719CB0423388D5FC721D356020A3A36CD5AD2C56B5C3CA795BFE18C9"
PURE_XML_SHA = "D84C27B0B8BA22E1649AF2375BE0B83C762BC62EB5047BB590539B374F8271EE"
SUBMISSION_NAME = "state_conditioned_hybrid_v3"
SUBMISSION_ROOT = ROOT / "artifacts/submission" / SUBMISSION_NAME
CONFIG_PATH = ROOT / "configs/submission/state_conditioned_hybrid_v3.json"
FALLBACK_PATH = ROOT / "configs/submission/pure_bt_fallback_v3.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True))
        handle.write("\n")


def promotion_prerequisites(
    model_metadata: dict[str, Any], evaluations: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    seed_policies = model_metadata.get("seed_oof_policies_at_selected_threshold", {})
    positive_seed_count = sum(
        value.get("mean", 0.0) > 0.0
        and value.get("intervention_precision", 0.0) >= 0.60
        and value.get("large_regression_ratio", 1.0) <= 0.05
        for value in seed_policies.values()
    )
    gates = {
        "offline_gate": bool(model_metadata.get("offline_gate_passed")),
        "runtime_ood_abstention": (
            model_metadata.get("runtime_ood_support", {}).get("fallback") == "BT_DEFAULT"
        ),
        "independent_model_seeds_gte_2": positive_seed_count >= 2,
        "shadow_gate": bool(evaluations["shadow"].get("gate_passed")),
        "micro_gate": bool(evaluations["micro"].get("gate_passed")),
        "development_gate": bool(evaluations["development"].get("gate_passed")),
        "development_clean_pairs_gte_60": evaluations["development"].get("clean_pairs", 0) >= 60,
        "heldout_gate": bool(evaluations["heldout"].get("gate_passed")),
        "heldout_clean_pairs_gte_30": evaluations["heldout"].get("clean_pairs", 0) >= 30,
        "development_heldout_direction_match": (
            evaluations["development"].get("clean_damage_delta_mean", 0.0) > 0.0
            and evaluations["heldout"].get("clean_damage_delta_mean", 0.0) > 0.0
        ),
    }
    gates["all_passed"] = all(gates.values())
    gates["positive_model_seed_count"] = positive_seed_count
    return gates


def build_config(model_sha: str) -> dict[str, Any]:
    action = asdict(GuidanceActionConfig(angular_offset_deg=0.25))
    controller = asdict(GuidanceControllerConfig(kind="vp_error_pd_v2"))
    runtime = asdict(
        GuidanceRuntimeConfig(
            selector_action_repeat_frames=6,
            minimum_action_hold_frames=36,
            maximum_active_frames=36,
            cooldown_frames=30,
            confidence_threshold=0.0,
            shadow_mode=False,
        )
    )
    return {
        "schema_version": "state_conditioned_hybrid_submission.v3",
        "status": "PROMOTED_LOCAL_STATE_CONDITIONED_HYBRID",
        "promotion_status": "PROMOTED",
        "candidate_kind": "STATE_ACTION_DISTRIBUTIONAL_ADVANTAGE",
        "mode": "guidance_selector",
        "policy_id": "state_conditioned_advantage_v3",
        "bundle_path": f"../../artifacts/submission/{SUBMISSION_NAME}/bundle",
        "model_path": f"../../artifacts/submission/{SUBMISSION_NAME}/bundle/model.npz",
        "bundle_sha256": model_sha,
        "selector_observation_contract": GUIDANCE_SERVER_CONTRACT_VERSION,
        "selector_observation_size": len(GUIDANCE_SERVER_FEATURES),
        "normalization_version": GUIDANCE_SERVER_NORMALIZATION_VERSION,
        "observation_features": list(GUIDANCE_SERVER_FEATURES),
        "runtime_observation_mode": "tactical16",
        "action_library": list(GUIDANCE_ADVANTAGE_ACTIONS),
        "action_magnitude": action,
        "controller": controller,
        "hard_eligibility_gate": {
            "kind": "rear120",
            "enter_target_ata_deg": 130.0,
            "exit_target_ata_deg": 120.0,
            "sim_hz": 60,
        },
        "activation_gate": {
            "kind": "rear120_and_offensive_or_pre_aim",
            "offensive": {
                "min_range_m": 152.4,
                "enter_max_range_m": 1200.0,
                "exit_max_range_m": 1500.0,
                "enter_ata_deg": 10.0,
                "exit_ata_deg": 15.0,
                "enter_min_target_ata_deg": 145.0,
                "exit_min_target_ata_deg": 130.0,
            },
            "phase_pre_aim": {
                "min_range_m": 152.4,
                "enter_angle_margin_deg": 5.0,
                "exit_angle_margin_deg": 7.0,
                "enter_range_margin_m": 250.0,
                "exit_range_margin_m": 400.0,
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
        "force_side": {"ownship": 1, "target": 2},
        "phase_config": [dict(row) for row in OFFICIAL_DAMAGE_PHASES],
        "wez": {"min_range_m": 152.4, "max_range_m": 1219.2, "angle_deg": 3.0},
        "health_source": "unavailable_constant_one",
        "fallback_mode": "exact_pure_bt",
        "bt": {
            "dll_path": f"../../artifacts/submission/{SUBMISSION_NAME}/bt/AIP_DCS_GDCC_0815.dll",
            "dll_sha256": PURE_DLL_SHA,
            "xml_path": f"../../artifacts/submission/{SUBMISSION_NAME}/bt/Rule_DCS_GDCC_0815.xml",
            "xml_sha256": PURE_XML_SHA,
            "rule_aliases": ["Rule_DCS_GDCC_0815.xml"],
            "turn_throttle_mode": "raw",
        },
        "server_status": "SERVER_BLOCKED",
    }


def portable_load_test() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        copied_artifact = root / "artifacts/submission" / SUBMISSION_NAME
        copied_config = root / "configs/submission" / CONFIG_PATH.name
        shutil.copytree(SUBMISSION_ROOT, copied_artifact)
        copied_config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CONFIG_PATH, copied_config)
        loaded = load_guidance_submission_config(copied_config, require_files=True)
        load_guidance_selector_bundle(loaded.bundle_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze promoted State-Conditioned Hybrid v3")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    for stage in ("shadow", "micro", "development", "heldout"):
        parser.add_argument(f"--{stage}-aggregate", type=Path, required=True)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    model_metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    evaluations = {
        stage: json.loads(getattr(args, f"{stage}_aggregate").read_text(encoding="utf-8"))
        for stage in ("shadow", "micro", "development", "heldout")
    }
    gates = promotion_prerequisites(model_metadata, evaluations)
    if not gates["all_passed"]:
        raise RuntimeError(f"refusing to freeze unpromoted v3 candidate: {gates}")
    if sha256(PURE_DLL) != PURE_DLL_SHA or sha256(PURE_XML) != PURE_XML_SHA:
        raise RuntimeError("Pure BT Champion hash mismatch")

    bundle = SUBMISSION_ROOT / "bundle"
    bt = SUBMISSION_ROOT / "bt"
    bundle.mkdir(parents=True, exist_ok=True)
    bt.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_dir / "model.npz", bundle / "model.npz")
    shutil.copy2(model_dir / "metadata.json", bundle / "metadata.json")
    shutil.copy2(PURE_DLL, bt / "AIP_DCS_GDCC_0815.dll")
    shutil.copy2(PURE_XML, bt / "Rule_DCS_GDCC_0815.xml")
    model_sha = sha256(bundle / "model.npz")
    if model_sha != model_metadata["model_sha256"]:
        raise RuntimeError("copied model hash mismatch")
    write_json(SUBMISSION_ROOT / "observation_contract.json", server_observation_contract())
    write_json(
        SUBMISSION_ROOT / "action_contract.json",
        {
            "actions": list(GUIDANCE_ADVANTAGE_ACTIONS),
            "parameterization": ["axis one-hot", "sign", "magnitude_norm", "duration_norm"],
            "runtime_candidates": model_metadata["runtime_candidates"],
            "runtime_threshold": model_metadata["runtime_threshold"],
            "runtime_ood_support": model_metadata["runtime_ood_support"],
            "bt_default": "byte-exact Pure BT action",
            "throttle": "exact BT-only",
        },
    )
    write_json(
        SUBMISSION_ROOT / "controller_contract.json",
        {
            "controller": asdict(GuidanceControllerConfig(kind="vp_error_pd_v2")),
            "angular_offset_deg": 0.25,
            "minimum_hold_frames": 36,
            "maximum_active_frames": 36,
            "early_abort": "safety/gate/nonfinite -> exact BT_DEFAULT",
        },
    )
    config = build_config(model_sha)
    write_json(CONFIG_PATH, config)
    write_json(
        FALLBACK_PATH,
        {
            "schema_version": "pure_bt_fallback.v3",
            "status": "READY",
            "mode": "bt",
            "fallback_mode": "exact_pure_bt",
            "expected_sim_hz": 60,
            "throttle_policy": "raw_bt_only",
            "bt": config["bt"],
        },
    )
    dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    base_sha = subprocess.check_output(["git", "merge-base", "HEAD", "origin/main"], cwd=ROOT, text=True).strip()
    write_json(
        SUBMISSION_ROOT / "training_metadata.json",
        {
            "base_main_sha": base_sha,
            "pure_bt_dll_sha256": PURE_DLL_SHA,
            "pure_bt_xml_sha256": PURE_XML_SHA,
            "model_sha256": model_sha,
            "config_sha256": sha256(CONFIG_PATH),
            "dataset_sha256": dataset_manifest["dataset_sha256"],
            "dataset_unique_states": model_metadata["unique_states"],
            "dataset_state_action_pairs": model_metadata["unique_state_action_pairs"],
            "model_seeds": model_metadata["seeds"],
            "epochs_per_model": model_metadata["epochs_per_model"],
            "optimizer_updates": model_metadata["optimizer_updates"],
            "ppo_iterations": 0,
            "ppo_steps": 0,
            "ppo_episodes": 0,
            "promotion_gates": gates,
            "evaluations": evaluations,
            "actual_server_status": "SERVER_BLOCKED",
        },
    )
    write_json(
        SUBMISSION_ROOT / "model_metadata.json",
        {**model_metadata, "frozen_model_sha256": model_sha, "promotion_status": "PROMOTED"},
    )
    (SUBMISSION_ROOT / "README.md").write_text(
        "# State-Conditioned Hybrid v3\n\n"
        "Pure BT Champion을 기본으로 사용하고, 보수적 distributional advantage gate가 통과한 상태에서만 bounded Guidance correction을 적용한다. 모든 path는 bundle/config 기준 상대 경로이며 throttle은 항상 exact BT-only다.\n\n"
        "Load: `python -c \"from dogfight.ai.state_action_advantage import load_guidance_selector_bundle; load_guidance_selector_bundle('artifacts/submission/state_conditioned_hybrid_v3/bundle')\"`\n",
        encoding="utf-8",
    )
    checksum_paths = sorted(
        path for path in SUBMISSION_ROOT.rglob("*") if path.is_file() and path.name != "sha256sums.txt"
    )
    (SUBMISSION_ROOT / "sha256sums.txt").write_text(
        "\n".join(f"{sha256(path)}  {path.relative_to(SUBMISSION_ROOT).as_posix()}" for path in checksum_paths) + "\n",
        encoding="utf-8",
    )
    load_guidance_submission_config(CONFIG_PATH, require_files=True)
    load_guidance_selector_bundle(bundle)
    portable_load_test()
    print(json.dumps({"status": "SUBMISSION_READY_HYBRID", "model_sha256": model_sha, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
