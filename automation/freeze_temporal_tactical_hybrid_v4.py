from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dogfight.ai.tactical_advantage import NumpyTemporalTacticalAdvantageSelector
from dogfight.ai.tactical_modes import tactical_action_contract
from dogfight.ai.temporal_observation import temporal_server_observation_contract


PURE_DLL_SHA = "4C93B4C6719CB0423388D5FC721D356020A3A36CD5AD2C56B5C3CA795BFE18C9"
PURE_XML_SHA = "D84C27B0B8BA22E1649AF2375BE0B83C762BC62EB5047BB590539B374F8271EE"
SUBMISSION_NAME = "temporal_tactical_hybrid_v4"
SUBMISSION_ROOT = ROOT / "artifacts" / "submission" / SUBMISSION_NAME
CONFIG_PATH = ROOT / "configs" / "submission" / f"{SUBMISSION_NAME}.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def promotion_prerequisites(
    metadata: dict[str, Any], evaluations: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    shadow = evaluations["shadow"]
    micro = evaluations["micro"]
    development = evaluations["development"]
    heldout = evaluations["heldout"]
    gates = {
        "offline_gate": bool(metadata.get("offline_gate_passed")),
        "minimum_2_model_seed_direction": int(
            metadata.get("selected_oof_policy", {}).get("consistent_seed_count", 0)
        )
        >= 2,
        "ood_exact_default_fallback": (
            metadata.get("runtime_ood_support", {}).get("fallback") == "BT_DEFAULT"
        ),
        "shadow_gate": shadow.get("decision") == "SHADOW_GATE_PASSED",
        "shadow_exact_command": bool(
            shadow.get("gate", {}).get("exact_shadow_command", False)
        ),
        "shadow_latency": bool(
            shadow.get("gate", {}).get("latency_over_limit_zero", False)
        ),
        "micro_gate": micro.get("decision") == "MICRO_GATE_PASSED",
        "development_gate": (
            development.get("decision") == "OFFICIAL_DEVELOPMENT_GATE_PASSED"
        ),
        "development_clean_pairs_gte_60": int(development.get("clean_pairs", 0)) >= 60,
        "heldout_gate": heldout.get("decision") == "HELD_OUT_GATE_PASSED",
        "pure_fallback_declared": metadata.get("candidate_modes", [None])[0]
        == "BT_DEFAULT",
    }
    gates["all_passed"] = all(gates.values())
    return gates


def build_config(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "temporal_tactical_hybrid_submission.v4",
        "status": "PROMOTED_LOCAL_TEMPORAL_TACTICAL_HYBRID",
        "promotion_status": "PROMOTED",
        "mode": "temporal_tactical",
        "policy_id": "temporal_tactical_advantage_v4",
        "bundle_path": f"../../artifacts/submission/{SUBMISSION_NAME}/bundle",
        "model_path": f"../../artifacts/submission/{SUBMISSION_NAME}/bundle/model.npz",
        "observation_contract_path": (
            f"../../artifacts/submission/{SUBMISSION_NAME}/observation_contract.json"
        ),
        "temporal_contract_path": (
            f"../../artifacts/submission/{SUBMISSION_NAME}/temporal_contract.json"
        ),
        "action_contract_path": (
            f"../../artifacts/submission/{SUBMISSION_NAME}/action_contract.json"
        ),
        "runtime_candidates": metadata["runtime_candidates"],
        "runtime_threshold": metadata["runtime_threshold"],
        "runtime_ood_support": metadata["runtime_ood_support"],
        "expected_sim_hz": 60,
        "latency_threshold_s": 0.1667,
        "throttle_policy": "exact_same_frame_pure_bt_only",
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


def portable_load_test(config: dict[str, Any]) -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        copied = root / "artifacts" / "submission" / SUBMISSION_NAME
        shutil.copytree(SUBMISSION_ROOT, copied)
        NumpyTemporalTacticalAdvantageSelector(copied / "bundle")
        serialized = json.dumps(config)
        if "C:\\" in serialized or "C:/" in serialized:
            raise RuntimeError("portable submission config contains an absolute Windows path")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze promoted Temporal Tactical Hybrid v4")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-metadata", type=Path, required=True)
    parser.add_argument("--pure-bt-dll", type=Path, required=True)
    parser.add_argument("--pure-bt-xml", type=Path, required=True)
    for stage in ("shadow", "micro", "development", "heldout"):
        parser.add_argument(f"--{stage}-summary", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    evaluations = {
        stage: json.loads(
            getattr(args, f"{stage}_summary").resolve().read_text(encoding="utf-8")
        )
        for stage in ("shadow", "micro", "development", "heldout")
    }
    gates = promotion_prerequisites(metadata, evaluations)
    pure_dll, pure_xml = args.pure_bt_dll.resolve(), args.pure_bt_xml.resolve()
    gates["pure_dll_hash"] = sha256(pure_dll) == PURE_DLL_SHA
    gates["pure_xml_hash"] = sha256(pure_xml) == PURE_XML_SHA
    gates["all_passed"] = all(value for key, value in gates.items() if key != "all_passed")
    if not gates["all_passed"]:
        raise RuntimeError(f"refusing to freeze unpromoted candidate: {gates}")
    if SUBMISSION_ROOT.exists() or CONFIG_PATH.exists():
        raise FileExistsError("refusing to overwrite an existing frozen v4 submission")

    bundle = SUBMISSION_ROOT / "bundle"
    bt = SUBMISSION_ROOT / "bt"
    bundle.mkdir(parents=True)
    bt.mkdir(parents=True)
    for name in ("model.npz", "metadata.json"):
        shutil.copy2(model_dir / name, bundle / name)
    shutil.copy2(pure_dll, bt / "AIP_DCS_GDCC_0815.dll")
    shutil.copy2(pure_xml, bt / "Rule_DCS_GDCC_0815.xml")
    if sha256(bundle / "model.npz") != metadata["model_sha256"]:
        raise RuntimeError("copied model hash mismatch")

    observation = temporal_server_observation_contract()
    write_json(SUBMISSION_ROOT / "observation_contract.json", observation)
    write_json(SUBMISSION_ROOT / "temporal_contract.json", observation)
    write_json(SUBMISSION_ROOT / "action_contract.json", tactical_action_contract())
    write_json(
        SUBMISSION_ROOT / "controller_contract.json",
        {
            "controller": "vp_error_pd_v2",
            "mode_hold_frames": [30, 60, 120],
            "cooldown_frames": 30,
            "throttle": "exact same-frame Pure BT",
            "failure_fallback": "exact Pure BT command",
        },
    )
    dataset = json.loads(args.dataset_metadata.resolve().read_text(encoding="utf-8"))
    write_json(
        SUBMISSION_ROOT / "training_metadata.json",
        {
            "model_sha256": metadata["model_sha256"],
            "dataset_sha256": dataset["dataset_sha256"],
            "dataset_unique_events": dataset["unique_events"],
            "dataset_state_action_pairs": dataset["state_action_pairs"],
            "model_seeds": metadata["seeds"],
            "epochs_per_model": metadata["epochs_per_model"],
            "optimizer_updates": metadata["optimizer_updates"],
            "ppo_iterations": 0,
            "ppo_steps": 0,
            "ppo_episodes": 0,
        },
    )
    write_json(
        SUBMISSION_ROOT / "evaluation_manifest.json",
        {"promotion_gates": gates, "evaluations": evaluations, "actual_server": "SERVER_BLOCKED"},
    )
    config = build_config(metadata)
    write_json(CONFIG_PATH, config)
    (SUBMISSION_ROOT / "README.md").write_text(
        "# Temporal Tactical Hybrid v4\n\n"
        "Pure BT Champion을 기본으로 사용하며 검증된 Tactical opportunity에서만 VP mode를 전환한다. "
        "Throttle과 모든 오류/OOD fallback은 exact Pure BT다. 모든 경로는 config 기준 상대 경로다.\n",
        encoding="utf-8",
    )
    checksum_paths = sorted(
        path for path in SUBMISSION_ROOT.rglob("*") if path.is_file() and path.name != "sha256sums.txt"
    )
    (SUBMISSION_ROOT / "sha256sums.txt").write_text(
        "\n".join(
            f"{sha256(path)}  {path.relative_to(SUBMISSION_ROOT).as_posix()}"
            for path in checksum_paths
        )
        + "\n",
        encoding="utf-8",
    )
    NumpyTemporalTacticalAdvantageSelector(bundle)
    portable_load_test(config)
    print(json.dumps({"status": "SUBMISSION_READY_HYBRID", "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
