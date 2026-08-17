from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load(name: str) -> dict:
    return yaml.safe_load((ROOT / "experiments" / name).read_text(encoding="utf-8"))


def frozen_core(payload: dict) -> dict:
    result = deepcopy(payload)
    result.pop("name")
    result.pop("notes")
    result.pop("output")
    runtime = result["runtime"]
    for key in ("seed", "max_effective_learner_time_s", "restore_checkpoint"):
        runtime.pop(key, None)
    return result


def test_s2101_resume_and_s2102_fresh_share_the_same_frozen_core() -> None:
    s2101 = load("0815_aim_residual_proxy_mixed_longrun_s2101.yaml")
    s2102 = load("0815_aim_residual_proxy_mixed_longrun_s2102.yaml")

    assert frozen_core(s2101) == frozen_core(s2102)
    assert s2101["runtime"]["seed"] == 2101
    assert s2102["runtime"]["seed"] == 2102
    assert s2101["runtime"]["max_effective_learner_time_s"] == 16200
    assert s2102["runtime"]["max_effective_learner_time_s"] == 18000
    assert "restore_checkpoint" in s2101["runtime"]
    assert "restore_checkpoint" not in s2102["runtime"]


def test_longrun_policy_authority_and_curriculum_are_frozen() -> None:
    payload = load("0815_aim_residual_proxy_mixed_longrun_s2102.yaml")
    env = payload["env"]
    env_config = payload["env_config"]
    residual = env_config["residual_training"]

    assert env["observation_mode"] == "aim_residual10_v2"
    assert env_config["ownship_control_mode"] == "bt_residual"
    assert residual["scale"] == 0.125
    assert residual["gate_kind"] == "aim"
    assert residual["composition_mode"] == "saturation_aware"
    assert env_config["step_ratio"] == 6
    assert payload["algo"]["name"] == "sac"
    assert [item["profile"] for item in env["target_profile_pool"]] == [
        "autopilot_crossing",
        "bt_0815",
        "bt_aip2",
    ]
    assert [item["weight"] for item in env["target_profile_pool"]] == [2.0, 1.0, 1.0]
    assert [item["name"] for item in env_config["initial_scenario"]["variants"]] == [
        "lateral_left",
        "lateral_right",
        "crossing_left",
        "crossing_right",
        "vertical_high",
        "vertical_low",
    ]
