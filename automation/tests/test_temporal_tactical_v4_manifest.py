from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_v4_manifest_freezes_phase0_contract() -> None:
    manifest = json.loads(
        (ROOT / "automation/manifests/temporal_tactical_hybrid_v4.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["base_main_sha"] == "ae4b2e0c1ea43d9f1b74e783a65293b5490ffcc4"
    assert manifest["phase0_inventory"]["champion_runtime_mode_switch"] is False
    assert (
        manifest["phase0_inventory"]["implementation_path"]
        == "B_SERVER_VISIBLE_DETERMINISTIC_VP_GENERATORS"
    )
    assert manifest["tactical_action_space"]["default_action"] == "BT_DEFAULT"
    assert manifest["tactical_action_space"]["discovery_hold_frames_frozen"] == [
        30,
        60,
        120,
    ]
    assert manifest["tactical_action_space"]["throttle"] == "EXACT_PURE_BT_ONLY"
    assert manifest["counterfactual"]["causal_truth"] == "PREFIX_REPLAY"
    assert manifest["promotion"]["submission_freeze_allowed"] is False
    assert manifest["promotion"]["held_out_opened"] is False
    assert manifest["promotion"]["ppo_allowed"] is False


def test_v4_hashes_match_frozen_contract() -> None:
    manifest = json.loads(
        (ROOT / "automation/manifests/temporal_tactical_hybrid_v4.json").read_text(
            encoding="utf-8"
        )
    )
    hashes = manifest["frozen_hashes"]
    assert hashes["pure_bt_dll_sha256"].startswith("4C93B4C6")
    assert hashes["pure_bt_xml_sha256"].startswith("D84C27B0")
    assert hashes["hybrid_v3_model_sha256"].startswith("7B97E8DF")
    assert hashes["hybrid_v3_dataset_sha256"].startswith("4DFBF311")
