from __future__ import annotations

import pytest

from automation.run_observation_pilot_matrix import expand_matrix


def _payload():
    return {
        "name": "template",
        "output": {"name": "pilot", "tag": "template"},
        "env": {},
        "env_config": {"residual_training": {"scale": 0.125}},
        "runtime": {"iterations": 20},
        "pilot_matrix": {
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
        "rear120_r10_s3101",
        "rear120_r10_s3102",
        "rear120_t16_s3101",
        "rear120_t16_s3102",
    ]
    assert [item["runtime"]["seed"] for _, item in expanded] == [3101, 3102, 3101, 3102]
    assert expanded[2][1]["env_config"]["observation_contract"]["version"] == "tactical16.v1"
    assert "pilot_matrix" not in expanded[0][1]


def test_requires_two_seeds() -> None:
    payload = _payload()
    payload["pilot_matrix"]["seeds"] = [3101]
    with pytest.raises(ValueError, match="at least two"):
        expand_matrix(payload)
