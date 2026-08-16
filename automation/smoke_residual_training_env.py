from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.ai.bt_rule_manager import activate_rule_xml
from dogfight.envs.single_agent_env import DogFightEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="0815 BT-in-the-loop 잔차 학습 환경의 1초 통합 스모크"
    )
    parser.add_argument(
        "--bt-dll",
        default=r"C:\Users\shy66\Downloads\aip_final_0815\aip_final_0815\AIP_DCS_GDCC_0815.dll",
    )
    parser.add_argument(
        "--bt-xml",
        default=r"C:\Users\shy66\Downloads\aip_final_0815\aip_final_0815\Rule_DCS_GDCC_0815.xml",
    )
    parser.add_argument(
        "--target-dll",
        default=r"C:\Users\shy66\Downloads\aip2\aip2\AIP_DCS_new.dll",
    )
    parser.add_argument(
        "--target-xml",
        default=r"C:\Users\shy66\Downloads\aip2\aip2\Rule_sei_AIP2_default.xml",
    )
    parser.add_argument("--scale", type=float, default=0.125)
    parser.add_argument("--gate-kind", choices=("aim", "offensive"), default="aim")
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--result-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "sim_hz": 60,
        "step_ratio": 1,
        "max_engage_time": 1.0,
        "episode_step_limit": 60,
        "min_altitude": 300.0,
        "observation_mode": "aim_residual10",
        "ownship_control_mode": "bt_residual",
        "ownship_behavior_dll": args.bt_dll,
        "target_mode": "behavior_tree",
        "target_behavior_dll": args.target_dll,
        "ownship": [0.0, 0.0, -5000.0, 0.0, 0.0, 0.0, 240.0],
        "target": [800.0, 0.0, -5000.0, 0.0, 0.0, 180.0, 220.0],
        "initial_scenario": {
            "mode": "default",
            "legacy_use_random_scenario": False,
        },
        "residual_training": {
            "scale": args.scale,
            "gate_kind": args.gate_kind,
        },
        "reward": {"mode": "aim_residual"},
    }
    with ExitStack() as stack:
        stack.enter_context(
            activate_rule_xml(
                args.bt_xml,
                ROOT,
                aliases=["Rule_DCS_GDCC_0815.xml"],
            )
        )
        stack.enter_context(
            activate_rule_xml(
                args.target_xml,
                ROOT,
                aliases=["Rule_sei_AIP2_default.xml"],
            )
        )
        env = DogFightEnv(config)
        try:
            observation, _ = env.reset(seed=args.seed)
            initial_observation = observation.tolist()
            terminated = truncated = False
            info = {}
            steps = 0
            action = np.array([0.8, -0.4, 0.2, -1.0], dtype=np.float32)
            while not (terminated or truncated):
                observation, _, terminated, truncated, info = env.step(action)
                steps += 1
            telemetry = info["ownship_provider_telemetry"]
            last = telemetry["last_frame"]
            result = {
                "seed": args.seed,
                "steps": steps,
                "observation_size": len(initial_observation),
                "observation_finite": bool(np.all(np.isfinite(initial_observation))),
                "gate_kind": args.gate_kind,
                "residual_scale": args.scale,
                "gate_active_ratio": telemetry[f"{args.gate_kind}_gate_active_ratio"],
                "gate_entries": telemetry[f"{args.gate_kind}_gate_entries"],
                "rl_correction_steps": telemetry["rl_correction_steps"],
                "bt_throttle": last["bt_action"][3],
                "final_throttle": last["final_action"][3],
                "gate_off_action_equal": (
                    last["gate"]["active"]
                    or last["final_action"] == last["bt_action"]
                ),
                "last_applied_correction": last["applied_rl_correction"],
                "episode_reward_components": info.get("ep_reward_components", {}),
                "throttle_residual_forced_zero": last[
                    "throttle_residual_forced_zero"
                ],
                "outcome": info.get("outcome"),
                "end_condition": info.get("end_condition"),
            }
        finally:
            env.close()
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    print(encoded)
    if args.result_json:
        output = Path(args.result_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
