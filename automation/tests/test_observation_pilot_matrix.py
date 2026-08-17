from __future__ import annotations

import pytest

from automation.run_observation_pilot_matrix import _deep_merge, expand_matrix


def _payload():
    return {
        "name": "template",
        "output": {"name": "pilot", "tag": "template"},
        "env": {},
        "env_config": {"residual_training": {"scale": 0.125}},
        "runtime": {"iterations": 20},
        "pilot_matrix": {
            "run_suffix": "r1",
            "seeds": [3101, 3102],
            "observations": [
                {"label": "r10", "mode": "aim_residual10_v2", "contract": {"version": "legacy"}},
                {"label": "t16", "mode": "tactical16", "contract": {"version": "tactical16.v1"}},
            ],
        },
    }


def test_expands_two_observations_and_two_independent_seeds() -> None:
    expanded = expand_matrix(_payload())
    assert [tag for tag, _ in expanded] == [
        "rear120_r10_s3101_r1",
        "rear120_r10_s3102_r1",
        "rear120_t16_s3101_r1",
        "rear120_t16_s3102_r1",
    ]
    assert [item["runtime"]["seed"] for _, item in expanded] == [3101, 3102, 3101, 3102]
    assert expanded[2][1]["env_config"]["observation_contract"]["version"] == "tactical16.v1"
    assert "pilot_matrix" not in expanded[0][1]


def test_requires_two_seeds() -> None:
    payload = _payload()
    payload["pilot_matrix"]["seeds"] = [3101]
    with pytest.raises(ValueError, match="at least two"):
        expand_matrix(payload)


def test_seed_specific_overrides_keep_independent_initial_bundles() -> None:
    payload = _payload()
    payload["pilot_matrix"]["seed_overrides"] = {
        "3101": {"runtime": {"init_bundle": "bundle_a"}},
        "3102": {"runtime": {"init_bundle": "bundle_b"}},
    }
    expanded = expand_matrix(payload)

    assert expanded[0][1]["runtime"]["init_bundle"] == "bundle_a"
    assert expanded[1][1]["runtime"]["init_bundle"] == "bundle_b"
    assert expanded[2][1]["runtime"]["init_bundle"] == "bundle_a"
    assert expanded[3][1]["runtime"]["init_bundle"] == "bundle_b"


def test_overlay_deep_merges_nested_experiment_config() -> None:
    payload = _deep_merge(
        {
            "env_config": {
                "residual_training": {"scale": 0.125, "gate_kind": "rear120"}
            }
        },
        {"env_config": {"residual_training": {"residual_axis_mask": "roll"}}},
    )

    assert payload["env_config"]["residual_training"] == {
        "scale": 0.125,
        "gate_kind": "rear120",
        "residual_axis_mask": "roll",
    }
