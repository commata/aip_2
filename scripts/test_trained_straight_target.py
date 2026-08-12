# -*- coding: utf-8 -*-
"""Run a verification dogfight using the trained straight-target SAC bundle and test_straight_target.yaml config."""
from __future__ import annotations

import sys
from pathlib import Path
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from DogFightEnvWrapper import DogFightWrapper
from dogfight.ai.rl_action_provider import RLActionProvider
from dogfight.ai.rllib_utils import build_algorithm_from_bundle


def main():
    yaml_path = ROOT / "experiments" / "test_straight_target.yaml"
    bundle_dir = ROOT / "artifacts" / "models" / "student_test" / "straight_target_sac_v5"

    print(f"[test] Loading config from: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    env_config = data.get("env_config", {})
    env_sec = data.get("env", {})
    env_config.update({
        "observation_mode": env_sec.get("observation_mode", "tactical16"),
        "target_mode": env_sec.get("target_mode", "autopilot"),
        "reward_module": env_sec.get("reward_module", "student.my_reward"),
        "max_engage_time": env_sec.get("max_engage_time", 60.0),
        "episode_step_limit": env_sec.get("episode_step_limit", 3600),
    })

    print(f"[test] Loading trained RL bundle from: {bundle_dir}")
    ownship_provider = RLActionProvider(
        bundle_dir=str(bundle_dir),
        algorithm_factory=build_algorithm_from_bundle,
        policy_id="default_policy"
    )

    env = DogFightWrapper(
        env_config=env_config,
        ownship_action_provider=ownship_provider,
        target_action_provider=None,
    )

    print("[test] Starting verification episode against straight-flying autopilot target...")
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0
    steps = 0

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        total_reward += reward
        steps += 1

    print("=" * 60)
    print("                VERIFICATION COMBAT RESULT")
    print("=" * 60)
    print(f"Total Steps       : {steps} (approx {steps * env._delta_t:.1f} sec)")
    print(f"End Condition     : {info.get('end_condition', 'n/a')}")
    print(f"Outcome           : {info.get('outcome', 'n/a').upper()}")
    print(f"Total Reward      : {total_reward:.4f}")
    print(f"Ownship Health    : {info.get('ownship_health', 'n/a'):.4f}")
    print(f"Target Health     : {info.get('target_health', 'n/a'):.4f}")
    print(f"Ownship Damage    : {info.get('ownship_damage', 'n/a'):.4f}")
    print(f"Target Damage     : {info.get('target_damage', 'n/a'):.4f}")
    print("=" * 60)

    env.make_tacviewLog()
    print("[test] Tacview 3D flight log saved successfully!")


if __name__ == "__main__":
    main()
