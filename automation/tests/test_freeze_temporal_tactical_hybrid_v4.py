from __future__ import annotations

from automation.freeze_temporal_tactical_hybrid_v4 import promotion_prerequisites


def _evaluations() -> dict:
    return {
        "shadow": {
            "decision": "SHADOW_GATE_PASSED",
            "gate": {"exact_shadow_command": True, "latency_over_limit_zero": True},
        },
        "micro": {"decision": "MICRO_GATE_PASSED"},
        "development": {
            "decision": "OFFICIAL_DEVELOPMENT_GATE_PASSED",
            "clean_pairs": 60,
        },
        "heldout": {"decision": "HELD_OUT_GATE_PASSED"},
    }


def _metadata() -> dict:
    return {
        "offline_gate_passed": True,
        "selected_oof_policy": {"consistent_seed_count": 2},
        "runtime_ood_support": {"fallback": "BT_DEFAULT"},
        "candidate_modes": ["BT_DEFAULT", "PURE_PURSUIT"],
    }


def test_freezer_requires_every_promotion_stage() -> None:
    result = promotion_prerequisites(_metadata(), _evaluations())
    assert result["all_passed"] is True
    failed = _evaluations()
    failed["heldout"]["decision"] = "HELD_OUT_GATE_FAILED"
    assert promotion_prerequisites(_metadata(), failed)["all_passed"] is False


def test_freezer_requires_two_model_seeds_and_exact_default() -> None:
    metadata = _metadata()
    metadata["selected_oof_policy"]["consistent_seed_count"] = 1
    metadata["candidate_modes"] = ["PURE_PURSUIT"]
    result = promotion_prerequisites(metadata, _evaluations())
    assert result["minimum_2_model_seed_direction"] is False
    assert result["pure_fallback_declared"] is False
    assert result["all_passed"] is False
