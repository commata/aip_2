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
    assert manifest["temporal_observation"]["observation_size"] == 93
    assert manifest["temporal_observation"]["history_frames"] == [0, 6, 12, 30]
    assert manifest["temporal_observation"]["startup_padding"] == "repeat_first_zero_delta"
    assert manifest["temporal_observation"]["packet_replay_parity"] == "BYTE_IDENTICAL"
    assert manifest["counterfactual"]["causal_truth"] == "PREFIX_REPLAY"
    assert "RESTART_STATE_CAUSAL_INVALID" in manifest["counterfactual"]["restart_state_use"]
    assert manifest["noise_floor"] == {
        "status": "DETERMINISTIC_REPEAT_NOISE_ZERO",
        "epsilon": 1e-9,
        "large_regression_threshold": 1e-6,
    }
    assert manifest["promotion"]["submission_freeze_allowed"] is False
    assert manifest["promotion"]["held_out_opened"] is False
    assert manifest["promotion"]["ppo_allowed"] is False
    assert manifest["pure_bt_decision_events"]["unique_events"] >= 300
    assert manifest["pure_bt_decision_events"]["diagnostic_taxonomy_is_label"] is False
    assert manifest["tactical_oracle"]["status"] == "TACTICAL_ORACLE_FEASIBLE"
    assert manifest["tactical_oracle"]["independent_revalidation_target_events"] >= 36
    assert manifest["tactical_oracle"]["risk_head_required"] is True
    assert manifest["tactical_oracle"]["action_space_gate_passed"] is True
    assert manifest["tactical_oracle"]["independent_revalidation_events"] >= 36


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
