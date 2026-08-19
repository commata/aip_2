from __future__ import annotations

from automation.freeze_state_conditioned_hybrid_v3 import promotion_prerequisites


def evaluation(stage: str, *, passed: bool = True, clean_pairs: int = 60):
    return {
        "stage": stage,
        "gate_passed": passed,
        "clean_pairs": clean_pairs,
        "clean_damage_delta_mean": 0.01 if passed else -0.01,
    }


def test_freezer_requires_every_performance_and_seed_gate():
    metadata = {
        "offline_gate_passed": True,
        "runtime_ood_support": {"fallback": "BT_DEFAULT"},
        "seed_oof_policies_at_selected_threshold": {
            "1": {"mean": 0.01, "intervention_precision": 0.8, "large_regression_ratio": 0.0},
            "2": {"mean": 0.02, "intervention_precision": 0.7, "large_regression_ratio": 0.0},
            "3": {"mean": -0.01, "intervention_precision": 0.9, "large_regression_ratio": 0.0},
        },
    }
    evaluations = {
        "shadow": evaluation("shadow"),
        "micro": evaluation("micro"),
        "development": evaluation("development", clean_pairs=60),
        "heldout": evaluation("heldout", clean_pairs=36),
    }
    assert promotion_prerequisites(metadata, evaluations)["all_passed"]
    evaluations["heldout"]["gate_passed"] = False
    assert not promotion_prerequisites(metadata, evaluations)["all_passed"]


def test_freezer_rejects_single_positive_model_seed():
    metadata = {
        "offline_gate_passed": True,
        "runtime_ood_support": {"fallback": "BT_DEFAULT"},
        "seed_oof_policies_at_selected_threshold": {
            "1": {"mean": 0.01, "intervention_precision": 0.8, "large_regression_ratio": 0.0},
            "2": {"mean": -0.01, "intervention_precision": 0.9, "large_regression_ratio": 0.0},
            "3": {"mean": 0.0, "intervention_precision": 0.9, "large_regression_ratio": 0.0},
        },
    }
    evaluations = {
        "shadow": evaluation("shadow"),
        "micro": evaluation("micro"),
        "development": evaluation("development", clean_pairs=60),
        "heldout": evaluation("heldout", clean_pairs=36),
    }
    result = promotion_prerequisites(metadata, evaluations)
    assert not result["all_passed"]
    assert result["positive_model_seed_count"] == 1


def test_freezer_requires_ood_exact_default_abstention():
    metadata = {
        "offline_gate_passed": True,
        "runtime_ood_support": {},
        "seed_oof_policies_at_selected_threshold": {
            "1": {"mean": 0.01, "intervention_precision": 0.8, "large_regression_ratio": 0.0},
            "2": {"mean": 0.02, "intervention_precision": 0.8, "large_regression_ratio": 0.0},
        },
    }
    evaluations = {
        "shadow": evaluation("shadow"),
        "micro": evaluation("micro"),
        "development": evaluation("development", clean_pairs=60),
        "heldout": evaluation("heldout", clean_pairs=36),
    }
    result = promotion_prerequisites(metadata, evaluations)
    assert not result["runtime_ood_abstention"]
    assert not result["all_passed"]
