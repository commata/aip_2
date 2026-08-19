from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent   # Release/ 루트
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from DogFightEnvWrapper import DogFightWrapper
from dogfight.ai.bt_action_provider import BTActionProvider
from dogfight.ai.bt_rule_manager import activate_rule_xml
from dogfight.ai.hybrid_action_provider import (
    AimGateConfig,
    CounterfactualPulseActionProvider,
    HybridActionProvider,
    OffensiveGateConfig,
    Rear120GateConfig,
    RESIDUAL_AXIS_MASKS,
    ResidualInferenceActionProvider,
    SafetyVetoConfig,
    ShotWindowGateConfig,
)
from dogfight.ai.guidance_selector import (
    GUIDANCE_ACTIONS,
    FixedGuidanceSelector,
    GuidanceActionConfig,
    GuidanceControllerConfig,
    GuidanceRuntimeConfig,
    GuidanceSelectorActionProvider,
    NumpyMLPGuidanceSelector,
)
from dogfight.ai.state_action_advantage import load_guidance_selector_bundle
from dogfight.ai.rllib_utils import build_inference_module_from_bundle
from dogfight.ai.rl_action_provider import RLActionProvider
from dogfight.ai.student_hooks import load_observation_hook


@contextmanager
def preserve_runtime_file(path: Path):
    """Restore simulator-mutated runtime input exactly after every local run."""
    original = path.read_bytes() if path.exists() else None
    try:
        yield
    finally:
        if original is not None and path.read_bytes() != original:
            path.write_bytes(original)


def parse_args():
    parser = argparse.ArgumentParser(description="Run local dogfight simulation between two inference backends.")
    backend_choices = ["rl", "bt", "hybrid", "residual_hybrid", "counterfactual_pulse", "guidance_selector", "fixed", "autopilot"]
    parser.add_argument("--ownship-backend", choices=backend_choices, required=True)
    parser.add_argument("--target-backend", choices=backend_choices, required=True)
    parser.add_argument("--ownship-bundle-dir")
    parser.add_argument("--target-bundle-dir")
    parser.add_argument("--ownship-bt-dll", default="AIP_DCS_ownship.dll")
    parser.add_argument("--target-bt-dll", default="AIP_BASE_target.dll")
    parser.add_argument("--bt-rule-xml", help="Optional Rule.xml source to activate while the simulation runs.")
    parser.add_argument(
        "--bt-rule-alias",
        action="append",
        default=[],
        help="Additional hard-coded Rule XML filename required by a native BT DLL; repeat as needed.",
    )
    parser.add_argument(
        "--bt-rule-alias-only",
        action="store_true",
        help="Activate only hard-coded aliases and leave Rule_forTraining.xml untouched.",
    )
    parser.add_argument("--ownship-policy-id", default="default_policy")
    parser.add_argument("--target-policy-id", default="default_policy")
    parser.add_argument(
        "--observation-mode",
        default="tactical16",
        choices=[
            "classic12",
            "relative14",
            "tactical16",
            "aim_residual10",
            "aim_residual10_v2",
            "aim_residual13_btaware",
            "custom",
        ],
    )
    parser.add_argument("--observation-module", default="", help="Optional custom observation module.")
    parser.add_argument("--hybrid-mode", choices=["offensive_residual", "residual", "blend", "switch"], default="offensive_residual")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--residual-scale", type=float, default=0.10)
    parser.add_argument(
        "--residual-gate",
        choices=["aim", "offensive", "combined", "rear120", "shot_window"],
        default="aim",
    )
    parser.add_argument(
        "--residual-composition",
        choices=["additive", "saturation_aware"],
        default="additive",
    )
    parser.add_argument(
        "--residual-axis-mask",
        choices=sorted(RESIDUAL_AXIS_MASKS),
        default="roll_pitch_yaw",
        help="Diagnostic inference-time surface mask; submission default keeps all surfaces.",
    )
    parser.add_argument("--rl-action-repeat", type=int, default=6, help="RL inference cadence while the offensive gate is active; BT still runs every simulator frame.")
    parser.add_argument(
        "--counterfactual-pulse",
        choices=("zero", "roll_pos", "roll_neg", "pitch_pos", "pitch_neg", "yaw_pos", "yaw_neg"),
        default="zero",
        help="Fixed first-window pulse used only by counterfactual_pulse backend.",
    )
    parser.add_argument("--counterfactual-pulse-magnitude", type=float, default=0.5)
    parser.add_argument("--counterfactual-pulse-frames", type=int, default=6)
    parser.add_argument("--counterfactual-pulse-start-offset-frames", type=int, default=0)
    parser.add_argument(
        "--guidance-fixed-action",
        choices=GUIDANCE_ACTIONS,
        help="Use one deterministic Guidance action instead of loading a selector bundle.",
    )
    parser.add_argument("--guidance-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--guidance-angular-offset-deg", type=float, default=0.5)
    parser.add_argument(
        "--guidance-controller-kind",
        choices=("fixed_action_v1", "vp_error_pd_v2"),
        default="fixed_action_v1",
    )
    parser.add_argument("--guidance-maximum-surface-correction", type=float, default=0.08)
    parser.add_argument("--guidance-roll-per-azimuth-degree", type=float, default=0.08)
    parser.add_argument("--guidance-pitch-per-elevation-degree", type=float, default=0.08)
    parser.add_argument("--guidance-yaw-per-azimuth-degree", type=float, default=0.04)
    parser.add_argument("--guidance-los-rate-damping", type=float, default=0.001)
    parser.add_argument("--guidance-minimum-hold-frames", type=int, default=18)
    parser.add_argument("--guidance-maximum-active-frames", type=int, default=90)
    parser.add_argument("--guidance-cooldown-frames", type=int, default=30)
    parser.add_argument(
        "--guidance-shadow-mode",
        action="store_true",
        help="Run selector and telemetry while returning exact Pure BT commands.",
    )
    parser.add_argument("--min-throttle-blend-speed", type=float, default=210.0, help="Preserve BT throttle below this speed when RL requests less power.")
    parser.add_argument(
        "--bt-turn-throttle-mode",
        choices=["raw", "optimized"],
        default="optimized",
        help="Use raw native-BT throttle for an immutable baseline or the legacy turn optimization.",
    )
    parser.add_argument("--offensive-min-range-m", type=float, default=152.4)
    parser.add_argument("--offensive-enter-range-m", type=float, default=1500.0)
    parser.add_argument("--offensive-exit-range-m", type=float, default=2000.0)
    parser.add_argument("--offensive-enter-ata-deg", type=float, default=15.0)
    parser.add_argument("--offensive-exit-ata-deg", type=float, default=25.0)
    parser.add_argument("--offensive-enter-target-ata-deg", type=float, default=135.0)
    parser.add_argument("--offensive-exit-target-ata-deg", type=float, default=110.0)
    parser.add_argument("--aim-min-range-m", type=float, default=152.4)
    parser.add_argument("--aim-enter-angle-margin-deg", type=float, default=7.0)
    parser.add_argument("--aim-exit-angle-margin-deg", type=float, default=10.0)
    parser.add_argument("--aim-enter-range-margin-m", type=float, default=300.0)
    parser.add_argument("--aim-exit-range-margin-m", type=float, default=550.0)
    parser.add_argument("--aim-min-hold-steps", type=int, default=12)
    parser.add_argument("--rear120-enter-target-ata-deg", type=float, default=120.0)
    parser.add_argument("--rear120-exit-target-ata-deg", type=float, default=110.0)
    parser.add_argument("--shot-window-enter-angle-margin-deg", type=float, default=1.5)
    parser.add_argument("--shot-window-exit-angle-margin-deg", type=float, default=2.5)
    parser.add_argument("--shot-window-enter-range-margin-m", type=float, default=25.0)
    parser.add_argument("--shot-window-exit-range-margin-m", type=float, default=75.0)
    parser.add_argument("--shot-window-enter-target-ata-deg", type=float, default=150.0)
    parser.add_argument("--shot-window-exit-target-ata-deg", type=float, default=140.0)
    parser.add_argument("--shot-window-max-active-steps", type=int, default=30)
    parser.add_argument("--shot-window-cooldown-steps", type=int, default=30)
    parser.add_argument(
        "--shot-window-residual-decay", choices=("none", "linear"), default="none"
    )
    parser.add_argument("--shot-window-residual-decay-floor", type=float, default=0.0)
    parser.add_argument(
        "--shot-window-rearm-mode",
        choices=["condition_exit", "time_only"],
        default="condition_exit",
    )
    parser.add_argument("--safety-minimum-altitude-m", type=float, default=350.0)
    parser.add_argument("--safety-minimum-speed-m-s", type=float, default=170.0)
    parser.add_argument("--safety-maximum-closing-rate-m-s", type=float, default=250.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario-file", help="JSON file containing an initial_scenario object or a full env_config object.")
    parser.add_argument("--result-json", help="Write deterministic episode result and provider telemetry to this path.")
    parser.add_argument("--telemetry-jsonl", help="Write simulator-rate maneuver and hybrid telemetry as JSON Lines.")
    parser.add_argument("--max-engage-time", type=float, default=300.0)
    parser.add_argument("--episode-step-limit", type=int, default=18000)
    parser.add_argument("--min-altitude", type=float, default=300.0)
    parser.add_argument("--save-log", action="store_true", help="Save tacview CSV log after the episode.")
    return parser.parse_args()


def build_provider(side: str, backend: str, bundle_dir: str | None, bt_dll: str, policy_id: str, hybrid_mode: str, alpha: float, residual_scale: float, residual_gate: str, residual_composition: str, aim_gate: AimGateConfig, offensive_gate: OffensiveGateConfig, rear120_gate: Rear120GateConfig, shot_window_gate: ShotWindowGateConfig, safety_veto: SafetyVetoConfig, rl_action_repeat: int, min_throttle_blend_speed: float, bt_turn_throttle_mode: str, residual_axis_mask: str = "roll_pitch_yaw", counterfactual_pulse: str = "zero", counterfactual_pulse_magnitude: float = 0.5, counterfactual_pulse_frames: int = 6, counterfactual_pulse_start_offset_frames: int = 0, guidance_fixed_action: str | None = None, guidance_action_config: GuidanceActionConfig | None = None, guidance_controller_config: GuidanceControllerConfig | None = None, guidance_confidence_threshold: float = 0.65, guidance_minimum_hold_frames: int = 18, guidance_maximum_active_frames: int = 90, guidance_cooldown_frames: int = 30, guidance_shadow_mode: bool = False):
    if backend in ("fixed", "autopilot"):
        return None
    if backend == "bt":
        return BTActionProvider(
            dll_name=bt_dll,
            enable_turn_throttle_optimization=bt_turn_throttle_mode == "optimized",
        )
    if backend == "rl":
        if not bundle_dir:
            raise ValueError(f"--{side}-bundle-dir is required when {side}-backend=rl")
        return RLActionProvider(bundle_dir=bundle_dir, algorithm_factory=build_inference_module_from_bundle, policy_id=policy_id)
    if backend == "hybrid":
        if not bundle_dir:
            raise ValueError(f"--{side}-bundle-dir is required when {side}-backend=hybrid")
        rl_provider = RLActionProvider(bundle_dir=bundle_dir, algorithm_factory=build_inference_module_from_bundle, policy_id=policy_id)
        bt_provider = BTActionProvider(
            dll_name=bt_dll,
            enable_turn_throttle_optimization=bt_turn_throttle_mode == "optimized",
        )
        return HybridActionProvider(
            primary_provider=rl_provider,
            secondary_provider=bt_provider,
            mode=hybrid_mode,
            alpha=alpha,
            residual_scale=residual_scale,
            offensive_gate=offensive_gate,
            primary_action_repeat=rl_action_repeat,
            min_throttle_blend_speed=min_throttle_blend_speed,
        )
    if backend == "residual_hybrid":
        if not bundle_dir:
            raise ValueError(
                f"--{side}-bundle-dir is required when {side}-backend=residual_hybrid"
            )
        residual_provider = RLActionProvider(
            bundle_dir=bundle_dir,
            algorithm_factory=build_inference_module_from_bundle,
            policy_id=policy_id,
            explore=False,
        )
        bt_provider = BTActionProvider(
            dll_name=bt_dll,
            enable_turn_throttle_optimization=bt_turn_throttle_mode == "optimized",
        )
        return ResidualInferenceActionProvider(
            bt_provider,
            residual_provider,
            residual_scale=residual_scale,
            gate_kind=residual_gate,
            aim_gate=aim_gate,
            offensive_gate=offensive_gate,
            rear120_gate=rear120_gate,
            shot_window_gate=shot_window_gate,
            safety_veto=safety_veto,
            rl_action_repeat=rl_action_repeat,
            composition_mode=residual_composition,
            residual_axis_mask=residual_axis_mask,
        )
    if backend == "counterfactual_pulse":
        pulse_axes = {
            "zero": (0.0, 0.0, 0.0, 0.0),
            "roll_pos": (1.0, 0.0, 0.0, 0.0),
            "roll_neg": (-1.0, 0.0, 0.0, 0.0),
            "pitch_pos": (0.0, 1.0, 0.0, 0.0),
            "pitch_neg": (0.0, -1.0, 0.0, 0.0),
            "yaw_pos": (0.0, 0.0, 1.0, 0.0),
            "yaw_neg": (0.0, 0.0, -1.0, 0.0),
        }
        pulse = np.asarray(pulse_axes[counterfactual_pulse], dtype=np.float32)
        pulse[:3] *= float(counterfactual_pulse_magnitude)
        return CounterfactualPulseActionProvider(
            BTActionProvider(
                dll_name=bt_dll,
                enable_turn_throttle_optimization=bt_turn_throttle_mode == "optimized",
            ),
            pulse,
            residual_scale=residual_scale,
            pulse_frames=counterfactual_pulse_frames,
            pulse_start_offset_frames=counterfactual_pulse_start_offset_frames,
            aim_gate=aim_gate,
            offensive_gate=offensive_gate,
            rear120_gate=rear120_gate,
            shot_window_gate=shot_window_gate,
            safety_veto=safety_veto,
            composition_mode=residual_composition,
        )
    if backend == "guidance_selector":
        if guidance_fixed_action:
            selector = FixedGuidanceSelector(guidance_fixed_action)
        elif bundle_dir:
            selector = load_guidance_selector_bundle(bundle_dir)
        else:
            raise ValueError(
                f"--{side}-bundle-dir or --guidance-fixed-action is required "
                "when backend=guidance_selector"
            )
        return GuidanceSelectorActionProvider(
            BTActionProvider(
                dll_name=bt_dll,
                enable_turn_throttle_optimization=bt_turn_throttle_mode == "optimized",
            ),
            selector,
            action_config=guidance_action_config,
            controller_config=guidance_controller_config,
            runtime_config=GuidanceRuntimeConfig(
                selector_action_repeat_frames=rl_action_repeat,
                minimum_action_hold_frames=guidance_minimum_hold_frames,
                maximum_active_frames=guidance_maximum_active_frames,
                cooldown_frames=guidance_cooldown_frames,
                confidence_threshold=guidance_confidence_threshold,
                shadow_mode=guidance_shadow_mode,
            ),
            rear120_config=rear120_gate,
            aim_config=aim_gate,
            offensive_config=offensive_gate,
            safety_config=safety_veto,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def backend_to_env_mode(backend: str) -> str:
    if backend in ("fixed", "autopilot"):
        return backend
    return "rl"


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main():
    args = parse_args()
    if "guidance_selector" in (args.ownship_backend, args.target_backend):
        if args.observation_mode != "tactical16" or args.observation_module:
            raise ValueError("guidance_selector local runtime requires builtin tactical16")
    observation_hook = load_observation_hook(args.observation_module) if args.observation_module else None
    offensive_gate = OffensiveGateConfig(
        min_range_m=args.offensive_min_range_m,
        enter_max_range_m=args.offensive_enter_range_m,
        exit_max_range_m=args.offensive_exit_range_m,
        enter_ata_deg=args.offensive_enter_ata_deg,
        exit_ata_deg=args.offensive_exit_ata_deg,
        enter_min_target_ata_deg=args.offensive_enter_target_ata_deg,
        exit_min_target_ata_deg=args.offensive_exit_target_ata_deg,
    )
    aim_gate = AimGateConfig(
        min_range_m=args.aim_min_range_m,
        enter_angle_margin_deg=args.aim_enter_angle_margin_deg,
        exit_angle_margin_deg=args.aim_exit_angle_margin_deg,
        enter_range_margin_m=args.aim_enter_range_margin_m,
        exit_range_margin_m=args.aim_exit_range_margin_m,
        min_hold_steps=args.aim_min_hold_steps,
    )
    rear120_gate = Rear120GateConfig(
        enter_target_ata_deg=args.rear120_enter_target_ata_deg,
        exit_target_ata_deg=args.rear120_exit_target_ata_deg,
    )
    shot_window_gate = ShotWindowGateConfig(
        enter_angle_margin_deg=args.shot_window_enter_angle_margin_deg,
        exit_angle_margin_deg=args.shot_window_exit_angle_margin_deg,
        enter_range_margin_m=args.shot_window_enter_range_margin_m,
        exit_range_margin_m=args.shot_window_exit_range_margin_m,
        enter_min_target_ata_deg=args.shot_window_enter_target_ata_deg,
        exit_min_target_ata_deg=args.shot_window_exit_target_ata_deg,
        max_active_steps=args.shot_window_max_active_steps,
        cooldown_steps=args.shot_window_cooldown_steps,
        residual_decay_mode=args.shot_window_residual_decay,
        residual_decay_floor=args.shot_window_residual_decay_floor,
        require_condition_exit_for_rearm=(
            args.shot_window_rearm_mode == "condition_exit"
        ),
    )
    safety_veto = SafetyVetoConfig(
        minimum_altitude_m=args.safety_minimum_altitude_m,
        minimum_speed_m_s=args.safety_minimum_speed_m_s,
        maximum_closing_rate_m_s=args.safety_maximum_closing_rate_m_s,
    )
    guidance_action_config = GuidanceActionConfig(
        angular_offset_deg=args.guidance_angular_offset_deg,
    )
    guidance_controller_config = GuidanceControllerConfig(
        kind=args.guidance_controller_kind,
        maximum_surface_correction=args.guidance_maximum_surface_correction,
        roll_per_azimuth_degree=args.guidance_roll_per_azimuth_degree,
        pitch_per_elevation_degree=args.guidance_pitch_per_elevation_degree,
        yaw_per_azimuth_degree=args.guidance_yaw_per_azimuth_degree,
        los_rate_damping_per_deg_s=args.guidance_los_rate_damping,
    )

    ownship_provider = build_provider(
        side="ownship",
        backend=args.ownship_backend,
        bundle_dir=args.ownship_bundle_dir,
        bt_dll=args.ownship_bt_dll,
        policy_id=args.ownship_policy_id,
        hybrid_mode=args.hybrid_mode,
        alpha=args.alpha,
        residual_scale=args.residual_scale,
        residual_gate=args.residual_gate,
        residual_composition=args.residual_composition,
        aim_gate=aim_gate,
        offensive_gate=offensive_gate,
        rear120_gate=rear120_gate,
        shot_window_gate=shot_window_gate,
        safety_veto=safety_veto,
        rl_action_repeat=args.rl_action_repeat,
        min_throttle_blend_speed=args.min_throttle_blend_speed,
        bt_turn_throttle_mode=args.bt_turn_throttle_mode,
        residual_axis_mask=args.residual_axis_mask,
        counterfactual_pulse=args.counterfactual_pulse,
        counterfactual_pulse_magnitude=args.counterfactual_pulse_magnitude,
        counterfactual_pulse_frames=args.counterfactual_pulse_frames,
        counterfactual_pulse_start_offset_frames=args.counterfactual_pulse_start_offset_frames,
        guidance_fixed_action=args.guidance_fixed_action,
        guidance_action_config=guidance_action_config,
        guidance_controller_config=guidance_controller_config,
        guidance_confidence_threshold=args.guidance_confidence_threshold,
        guidance_minimum_hold_frames=args.guidance_minimum_hold_frames,
        guidance_maximum_active_frames=args.guidance_maximum_active_frames,
        guidance_cooldown_frames=args.guidance_cooldown_frames,
        guidance_shadow_mode=args.guidance_shadow_mode,
    )
    target_provider = build_provider(
        side="target",
        backend=args.target_backend,
        bundle_dir=args.target_bundle_dir,
        bt_dll=args.target_bt_dll,
        policy_id=args.target_policy_id,
        hybrid_mode=args.hybrid_mode,
        alpha=args.alpha,
        residual_scale=args.residual_scale,
        residual_gate=args.residual_gate,
        residual_composition=args.residual_composition,
        aim_gate=aim_gate,
        offensive_gate=offensive_gate,
        rear120_gate=rear120_gate,
        shot_window_gate=shot_window_gate,
        safety_veto=safety_veto,
        rl_action_repeat=args.rl_action_repeat,
        min_throttle_blend_speed=args.min_throttle_blend_speed,
        bt_turn_throttle_mode=args.bt_turn_throttle_mode,
        residual_axis_mask=args.residual_axis_mask,
        counterfactual_pulse=args.counterfactual_pulse,
        counterfactual_pulse_magnitude=args.counterfactual_pulse_magnitude,
        counterfactual_pulse_frames=args.counterfactual_pulse_frames,
        counterfactual_pulse_start_offset_frames=args.counterfactual_pulse_start_offset_frames,
        guidance_fixed_action=args.guidance_fixed_action,
        guidance_action_config=guidance_action_config,
        guidance_controller_config=guidance_controller_config,
        guidance_confidence_threshold=args.guidance_confidence_threshold,
        guidance_minimum_hold_frames=args.guidance_minimum_hold_frames,
        guidance_maximum_active_frames=args.guidance_maximum_active_frames,
        guidance_cooldown_frames=args.guidance_cooldown_frames,
        guidance_shadow_mode=args.guidance_shadow_mode,
    )

    with preserve_runtime_file(ROOT / "aircraft" / "f16" / "f16_init.xml"), activate_rule_xml(
        args.bt_rule_xml,
        ROOT,
        aliases=args.bt_rule_alias,
        include_default=not args.bt_rule_alias_only,
    ):
        env_config = {
                "observation_mode": observation_hook["mode"] if observation_hook else args.observation_mode,
                "observation_module": args.observation_module,
                "ownship_control_mode": backend_to_env_mode(args.ownship_backend),
                "target_mode": backend_to_env_mode(args.target_backend),
                "max_engage_time": args.max_engage_time,
                "episode_step_limit": args.episode_step_limit,
                "min_altitude": args.min_altitude,
                "maneuver_telemetry_path": args.telemetry_jsonl,
            }
        if args.scenario_file:
            scenario_payload = json.loads(Path(args.scenario_file).read_text(encoding="utf-8"))
            if "env_config" in scenario_payload:
                env_config.update(scenario_payload["env_config"])
            elif "initial_scenario" in scenario_payload:
                env_config.update(scenario_payload)
            else:
                env_config["initial_scenario"] = scenario_payload
        env = DogFightWrapper(
            env_config=env_config,
            observation_fn=observation_hook["build_observation"] if observation_hook else None,
            observation_size=observation_hook["size"] if observation_hook else None,
            observation_low=observation_hook["low"] if observation_hook else None,
            observation_high=observation_hook["high"] if observation_hook else None,
            ownship_action_provider=ownship_provider,
            target_action_provider=target_provider,
        )

        try:
            observation, info = env.reset(seed=args.seed)
            terminated = False
            truncated = False
            total_reward = 0.0
            while not (terminated or truncated):
                observation, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
                total_reward += reward

            print("simulation finished")
            print(f"end_condition: {info.get('end_condition', '')}")
            print(f"terminated: {terminated} truncated: {truncated}")
            print(f"total_reward: {total_reward:.4f}")
            print(f"ownship_health: {info.get('ownship_health', 'n/a')}")
            print(f"target_health: {info.get('target_health', 'n/a')}")

            if args.result_json:
                result = {
                    **info,
                    "seed": args.seed,
                    "terminated": terminated,
                    "truncated": truncated,
                    "total_reward": float(total_reward),
                    "episode_seconds": float(info.get("ep_step_count", 0)) / 60.0,
                    "ownship_backend": args.ownship_backend,
                    "target_backend": args.target_backend,
                    "hybrid_mode": args.hybrid_mode,
                    "residual_scale": args.residual_scale,
                }
                result_path = Path(args.result_json)
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(
                    json.dumps(result, indent=2, sort_keys=True, default=_json_default),
                    encoding="utf-8",
                )
                print(f"result_json: {result_path}")

            if args.save_log:
                env.make_tacviewLog()
                print("tacview log saved")
        finally:
            env.close()


if __name__ == "__main__":
    main()
