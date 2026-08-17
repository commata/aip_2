from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent   # Release/ 루트
SRC = ROOT / "src"
RELEASE_ROOT = ROOT
DEFAULT_BT_DLL = RELEASE_ROOT / "AIP_BASE.dll"
DEFAULT_BT_RULE_XML = RELEASE_ROOT / "Rule_forTraining.xml"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dogfight.ai.bt_action_provider import BTActionProvider
from dogfight.ai.bt_rule_manager import activate_rule_xml
from dogfight.ai.hybrid_action_provider import ResidualInferenceActionProvider
from dogfight.ai.rllib_utils import build_inference_module_from_bundle
from dogfight.ai.rl_action_provider import RLActionProvider
from dogfight.ai.student_hooks import load_observation_hook
from dogfight.submission import (
    load_bundle_observation_contract,
    load_submission_config,
)
from dogfight.unreal import AIType, ProviderCommandPolicy, UnrealAIPilotUDPClient

# python run_unreal_inference.py --mode rl --bundle-dir artifacts\models\team01\v1 --team-name team01
# python run_unreal_inference.py --mode bt --team-name team01

def parse_args():
    parser = argparse.ArgumentParser(description="Run RL/BT/Hybrid inference and communicate with the Unreal AI server over UDP.")
    parser.add_argument("--mode", choices=["rl", "bt", "hybrid"], help="Inference backend to use.")
    parser.add_argument(
        "--submission-config",
        help="Single-source submission JSON. When set, mode is forced to hybrid residual.",
    )
    parser.add_argument("--server-ip", default="192.168.10.115", help="Unreal server IP address.")
    parser.add_argument("--server-port", type=int, default=9999, help="Unreal server UDP port.")
    parser.add_argument("--team-name", default="FDSA", help="Client team name sent to the Unreal server.")
    parser.add_argument("--simulation-state", type=int, default=1, help="Heartbeat simulation state value.")
    parser.add_argument("--heartbeat-sec", type=float, default=1.0, help="Heartbeat interval in seconds.")
    parser.add_argument("--command-delay-sec", type=float, default=0.0, help="Delay before replying with CMD after both PlaneInfo packets are ready.")
    parser.add_argument("--recv-timeout-sec", type=float, default=0.2, help="UDP socket receive timeout.")
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=6,
        help=(
            "Number of completed own/enemy PlaneInfo pairs to hold each action. "
            "Use 6 to match Release training step_ratio=6; use 1 for per-packet policy calls."
        ),
    )
    parser.add_argument(
        "--debug-action-repeat",
        action="store_true",
        help="Print action-repeat counter, frame indices, update/hold state, and action values.",
    )
    parser.add_argument("--packet-monitor", action="store_true", help="Render live RX/TX packet values in the terminal.")
    parser.add_argument("--packet-monitor-interval-sec", type=float, default=0.2, help="Refresh interval for the live packet monitor.")
    parser.add_argument(
        "--observation-mode",
        default="auto",
        choices=[
            "auto",
            "classic12",
            "relative14",
            "tactical16",
            "aim_residual10",
            "aim_residual10_v2",
            "aim_residual13_btaware",
            "custom",
        ],
        help="Observation mode. auto requires bundle metadata and fails on mismatch.",
    )
    parser.add_argument("--observation-module", default="", help="Optional custom observation module.")
    parser.add_argument("--ownship-force-side", type=int, default=1, help="Force side to use for the ownship in BT inference.")
    parser.add_argument("--target-force-side", type=int, default=2, help="Force side to use for the enemy in BT inference.")
    parser.add_argument(
        "--bt-dll",
        default=str(DEFAULT_BT_DLL),
        help="Behavior tree DLL path for BT inference.",
    )
    parser.add_argument(
        "--bt-rule-xml",
        default=str(DEFAULT_BT_RULE_XML),
        help=(
            "Rule XML source to activate while the client runs. "
            "Use Rule_forTraining.xml by default, or pass a team file such as "
            "Rule_team01.xml."
        ),
    )
    parser.add_argument(
        "--bt-rule-alias",
        action="append",
        default=[],
        help="Additional hard-coded Rule XML filename required by the BT DLL.",
    )
    parser.add_argument("--bundle-dir", help="Lightweight RL bundle directory created by train_rllib.py.")
    parser.add_argument("--policy-id", default="default_policy", help="RLlib policy id to load from the lightweight bundle.")
    parser.add_argument("--explore", action="store_true", help="Enable stochastic action sampling for RL inference.")
    parser.add_argument("--hybrid-mode", choices=["offensive_residual", "residual", "blend", "switch"], default="offensive_residual", help="Hybrid action composition strategy.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend weight for hybrid blend mode.")
    parser.add_argument("--residual-scale", type=float, default=0.10, help="Residual scaling factor; offensive mode accepts only 0.10 through 0.20.")
    parser.add_argument(
        "--bt-turn-throttle-mode",
        choices=["raw", "optimized"],
        default="optimized",
        help="Use raw native-BT throttle or the legacy aggressive-turn optimization.",
    )
    parser.add_argument("--offensive-min-range-m", type=float, default=152.4)
    parser.add_argument("--offensive-enter-range-m", type=float, default=1500.0)
    parser.add_argument("--offensive-exit-range-m", type=float, default=2000.0)
    parser.add_argument("--offensive-enter-ata-deg", type=float, default=15.0)
    parser.add_argument("--offensive-exit-ata-deg", type=float, default=25.0)
    parser.add_argument("--offensive-enter-target-ata-deg", type=float, default=135.0)
    parser.add_argument("--offensive-exit-target-ata-deg", type=float, default=110.0)
    parser.add_argument("--min-throttle-blend-speed", type=float, default=210.0, help="Preserve BT throttle below this speed when RL requests less power.")
    parser.add_argument(
        "--ai-type",
        choices=["rule", "rl", "sl", "fusion", "etc"],
        default="rl",
        help="AI type announced to the Unreal server in ClientJoinInfo.",
    )
    return parser.parse_args()


def build_action_provider(args):
    if args.mode == "bt":
        return BTActionProvider(
            dll_name=args.bt_dll,
            enable_turn_throttle_optimization=args.bt_turn_throttle_mode == "optimized",
        )

    if args.bundle_dir is None:
        raise ValueError("--bundle-dir is required for rl and hybrid modes")

    submission = getattr(args, "_submission_config", None)
    if args.mode == "hybrid" and submission is None:
        raise ValueError(
            "--submission-config is required for hybrid mode; legacy throttle/blend "
            "hybrid is not a submission-safe residual path"
        )

    rl_provider = RLActionProvider(
        bundle_dir=args.bundle_dir,
        algorithm_factory=build_inference_module_from_bundle,
        policy_id=args.policy_id,
        explore=args.explore,
    )

    if args.mode == "rl":
        return rl_provider

    bt_provider = BTActionProvider(
        dll_name=args.bt_dll,
        enable_turn_throttle_optimization=False,
    )
    hard_gate = dict(submission.raw["hard_eligibility_gate"])
    hard_gate.pop("kind", None)
    hard_gate.setdefault("sim_hz", submission.expected_sim_hz)
    activation = submission.raw["activation_gate"]
    activation_kind = str(activation.get("kind", ""))
    gate_kind = (
        "shot_window"
        if activation_kind.startswith("shot_window")
        else "rear120"
    )
    return ResidualInferenceActionProvider(
        bt_provider=bt_provider,
        residual_provider=rl_provider,
        residual_scale=submission.residual_scale,
        gate_kind=gate_kind,
        rear120_gate=hard_gate,
        shot_window_gate=activation.get("shot_window"),
        aim_gate=activation.get("phase_pre_aim"),
        offensive_gate=activation.get("offensive"),
        safety_veto=activation.get("safety_veto"),
        rl_action_repeat=submission.rl_action_repeat,
        composition_mode=submission.composition_mode,
        inference_timeout_s=submission.latency_threshold_s,
    )


def resolve_runtime_contract(args):
    if args.submission_config:
        submission = load_submission_config(args.submission_config)
        args._submission_config = submission
        args.mode = "hybrid"
        args.bundle_dir = str(submission.bundle_path)
        args.policy_id = submission.policy_id
        args.observation_mode = submission.observation_mode
        args.bt_dll = str(submission.bt_dll_path)
        args.bt_rule_xml = str(submission.bt_xml_path)
        args.bt_rule_alias = list(
            submission.raw.get("bt", {}).get("rule_aliases", [])
        )
        args.residual_scale = submission.residual_scale
        args.action_repeat = submission.rl_action_repeat
        force_side = submission.raw.get("force_side", {})
        args.ownship_force_side = int(force_side.get("ownship", 1))
        args.target_force_side = int(force_side.get("target", 2))
        return submission

    args._submission_config = None
    if args.mode is None:
        raise ValueError("--mode or --submission-config is required")
    if args.mode == "hybrid":
        raise ValueError("hybrid mode requires --submission-config")
    if args.mode == "rl":
        if args.bundle_dir is None:
            raise ValueError("--bundle-dir is required for rl mode")
        bundle_contract = load_bundle_observation_contract(args.bundle_dir)
        bundle_mode = bundle_contract["mode"]
        if args.observation_mode == "auto":
            args.observation_mode = bundle_mode
        elif args.observation_mode != bundle_mode:
            raise ValueError(
                "observation mode mismatch: "
                f"cli={args.observation_mode!r}, bundle={bundle_mode!r}"
            )
    elif args.observation_mode == "auto":
        args.observation_mode = "classic12"
    return None


def parse_ai_type(value: str) -> AIType:
    mapping = {
        "rule": AIType.RuleBased,
        "rl": AIType.ReinforcementLearning,
        "sl": AIType.SupervisedLearning,
        "fusion": AIType.Fusion,
        "etc": AIType.etc,
    }
    return mapping[value]


def main():
    args = parse_args()
    submission = resolve_runtime_contract(args)
    observation_hook = load_observation_hook(args.observation_module) if args.observation_module else None
    with activate_rule_xml(args.bt_rule_xml, ROOT, aliases=args.bt_rule_alias):
        action_provider = build_action_provider(args)
        command_policy = ProviderCommandPolicy(
            action_provider=action_provider,
            observation_mode=observation_hook["mode"] if observation_hook else args.observation_mode,
            observation_fn=observation_hook["build_observation"] if observation_hook else None,
            ownship_force_side=args.ownship_force_side,
            target_force_side=args.target_force_side,
            action_repeat=(
                1 if args.mode == "hybrid" else args.action_repeat
            ),
            debug_action_repeat=args.debug_action_repeat,
            wez_config=submission.wez_config if submission else None,
            health_source=(submission.health_source if submission else "simulator"),
            expected_sim_hz=(submission.expected_sim_hz if submission else 60),
            phase_config=(submission.phase_config if submission else None),
        )
        client = UnrealAIPilotUDPClient(
            command_policy=command_policy,
            server_ip=args.server_ip,
            server_port=args.server_port,
            team_name=args.team_name,
            ai_type=parse_ai_type(args.ai_type),
            simulation_state=args.simulation_state,
            heartbeat_interval_sec=args.heartbeat_sec,
            command_delay_sec=args.command_delay_sec,
            recv_timeout_sec=args.recv_timeout_sec,
            enable_terminal_monitor=args.packet_monitor,
            terminal_monitor_interval_sec=args.packet_monitor_interval_sec,
        )

        try:
            client.run()
        finally:
            action_provider.close()


if __name__ == "__main__":
    main()
