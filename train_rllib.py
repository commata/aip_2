from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import faulthandler
import json
import math
import random
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np
from ray.tune.registry import register_env

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
_pythonpath_entries = [str(ROOT), str(SRC)]
_existing_pythonpath = os.environ.get("PYTHONPATH", "")
if _existing_pythonpath:
    _pythonpath_entries.append(_existing_pythonpath)
os.environ["PYTHONPATH"] = os.pathsep.join(_pythonpath_entries)
_RAY_RAYLET_START_WAIT_TIME_S = "60"
_RUNTIME_DIAGNOSTICS_ENABLED = False

from DogFightEnvWrapper import DogFightWrapper
from dogfight.ai.checkpoint_io import (
    apply_lightweight_policy_bundle,
    save_lightweight_policy_bundle,
)
from dogfight.ai.bt_rule_manager import activate_rule_xml
from dogfight.ai.callbacks import aim_variant_metric_name
from dogfight.ai.dashboard_logger import (
    DashboardJsonlLogger,
    copy_experiment_yaml,
    load_experiment_metadata,
)
from dogfight.ai.engagement_replay_logger import EngagementReplayLogger
from dogfight.ai.policy_probe_logger import PolicyProbeLogger
from dogfight.ai.rllib_utils import build_algorithm_config, normalize_algorithm_name
from dogfight.ai.student_hooks import load_observation_hook, load_reward_hook
from dogfight.ai.training.config_io import deep_update, load_experiment_env_config
from dogfight.ai.training_record import save_training_record
from dogfight.envs.observation import (
    observation_size as builtin_observation_size,
)


def _runtime_snapshot() -> dict[str, Any]:
    """Return a compact process/memory snapshot without requiring psutil."""

    snapshot: dict[str, Any] = {
        "pid": os.getpid(),
        "threads": threading.active_count(),
    }
    try:
        import psutil

        process = psutil.Process()
        memory = process.memory_info()
        virtual = psutil.virtual_memory()
        snapshot.update(
            {
                "rss_mb": round(memory.rss / (1024 * 1024), 3),
                "system_available_mb": round(virtual.available / (1024 * 1024), 3),
                "children": [
                    {"pid": child.pid, "name": child.name()}
                    for child in process.children(recursive=True)
                ],
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostics must never block training.
        snapshot["snapshot_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def _runtime_stage(stage: str, **details: Any) -> None:
    """Emit a flushed monotonic stage marker for training hang diagnosis."""

    if not _RUNTIME_DIAGNOSTICS_ENABLED:
        return
    payload = {
        "event": "training_runtime_stage",
        "stage": stage,
        "monotonic_s": round(time.monotonic(), 6),
        **_runtime_snapshot(),
        **details,
    }
    print(f"[RUNTIME_DIAGNOSTIC] {json.dumps(payload, ensure_ascii=False)}", flush=True)


def _set_runtime_diagnostics(enabled: bool) -> None:
    """Enable stage markers and periodic Python stack dumps for a run."""

    global _RUNTIME_DIAGNOSTICS_ENABLED
    _RUNTIME_DIAGNOSTICS_ENABLED = bool(enabled)
    if not enabled:
        return
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    _runtime_stage("diagnostics_enabled")


def _disable_runtime_diagnostics() -> None:
    global _RUNTIME_DIAGNOSTICS_ENABLED
    if _RUNTIME_DIAGNOSTICS_ENABLED:
        faulthandler.cancel_dump_traceback_later()
    _RUNTIME_DIAGNOSTICS_ENABLED = False


def _algorithm_log_root(args) -> Path:
    """Keep RLlib's per-run logger state inside the workspace artifacts tree."""

    return ROOT / "artifacts" / "ray_results" / args.output_name / args.output_tag


def _build_algorithm_logger_creator(args):
    """Build the normal UnifiedLogger without writing to ~/ray_results."""

    from ray.tune.logger import UnifiedLogger

    root = _algorithm_log_root(args)
    root.mkdir(parents=True, exist_ok=True)

    def logger_creator(config):
        logdir = tempfile.mkdtemp(prefix="run_", dir=root)
        _runtime_stage("algorithm_logger_created", logdir=logdir)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def _ensure_ray_runtime_env(*, num_cpus: int | None = None) -> None:
    """Restart Ray with local project paths available to worker actors."""
    import ray

    # 2026-05-29: Give slower PCs more time for raylet/GCS startup after shutdown.
    os.environ.setdefault("RAY_raylet_start_wait_time_s", _RAY_RAYLET_START_WAIT_TIME_S)
    _runtime_stage("ray_shutdown_start")
    ray.shutdown()
    _runtime_stage("ray_shutdown_done")
    init_kwargs: dict[str, Any] = {}
    if num_cpus is not None:
        init_kwargs["num_cpus"] = int(num_cpus)
    _runtime_stage("ray_init_start", ray_num_cpus=num_cpus)
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env={
            "env_vars": {
                "PYTHONPATH": os.environ["PYTHONPATH"],
                "RAY_raylet_start_wait_time_s": os.environ[
                    "RAY_raylet_start_wait_time_s"
                ],
            }
        },
        **init_kwargs,
    )
    _runtime_stage("ray_init_done", ray_num_cpus=num_cpus)


def env_creator(env_config):
    cfg = dict(env_config)
    cfg["_runner_index"] = getattr(
        env_config,
        "worker_index",
        cfg.get("_runner_index", "local"),
    )
    cfg["_env_index"] = getattr(
        env_config,
        "vector_index",
        cfg.get("_env_index", 0),
    )
    reward_fn = None
    observation_hook = None
    reward_module = str(cfg.get("reward_module", "")).strip()
    if reward_module:
        reward_fn, reward_config = load_reward_hook(reward_module)
        cfg.setdefault("reward", reward_config)
    observation_module = str(cfg.get("observation_module", "")).strip()
    if observation_module:
        observation_hook = load_observation_hook(observation_module)
        cfg["observation_mode"] = observation_hook["mode"]
        cfg["observation_module"] = observation_module
        cfg["observation_summary"] = observation_hook["description"]
    env = DogFightWrapper(
        cfg,
        reward_fn=reward_fn,
        observation_fn=observation_hook["build_observation"] if observation_hook else None,
        observation_size=observation_hook["size"] if observation_hook else None,
        observation_low=observation_hook["low"] if observation_hook else None,
        observation_high=observation_hook["high"] if observation_hook else None,
    )
    if reward_module and "reward" in cfg:
        env.config["reward"] = dict(cfg["reward"])
    return env


# ── Metric helpers ────────────────────────────────────────────────────────────

def _extract_learner_stats(result: dict) -> dict:
    """Extract algorithm-level stats from RLlib result (new & old API)."""
    keys = (
        "policy_loss", "vf_loss", "entropy", "kl", "clip_frac", "explained_var",
        "actor_loss", "critic_loss", "alpha_loss", "alpha", "target_entropy",
        "replay_buffer_size", "replay_buffer_memory_mb", "env_steps_per_sec",
        "learner_steps_per_sec", "iteration_time_s",
    )
    stats = {k: "n/a" for k in keys}

    # New API (RLlib 2.x): result["learners"][module_id][...]
    learners = result.get("learners", {})
    if learners:
        ps = next(iter(learners.values()), {})
        stats["policy_loss"]   = ps.get("policy_loss", "n/a")
        stats["vf_loss"]       = ps.get("vf_loss", "n/a")
        stats["entropy"]       = ps.get("entropy", "n/a")
        stats["kl"]            = ps.get("mean_kl_loss", ps.get("kl", "n/a"))
        stats["clip_frac"]     = ps.get("clip_frac", "n/a")
        stats["explained_var"] = ps.get("vf_explained_var", "n/a")
        _fill_optional_learner_stats(stats, ps, result)
        return stats

    # Old API: result["info"]["learner"][policy_id][...]
    ps = next(iter(result.get("info", {}).get("learner", {}).values()), {})
    if ps:
        stats["policy_loss"]   = ps.get("policy_loss", "n/a")
        stats["vf_loss"]       = ps.get("vf_loss", "n/a")
        stats["entropy"]       = ps.get("entropy", "n/a")
        stats["kl"]            = ps.get("kl", "n/a")
        stats["clip_frac"]     = ps.get("clip_frac", "n/a")
        stats["explained_var"] = ps.get("vf_explained_var", "n/a")
        _fill_optional_learner_stats(stats, ps, result)
    else:
        _fill_optional_learner_stats(stats, {}, result)
    return stats


def _iter_nested_items(value: Any, prefix: str = "", depth: int = 0):
    """Yield flattened result items without expanding large arrays."""
    if depth > 8:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}/{key_text}" if prefix else key_text
            yield from _iter_nested_items(item, path, depth + 1)
    elif isinstance(value, (list, tuple)) and len(value) <= 16:
        for index, item in enumerate(value):
            path = f"{prefix}/{index}" if prefix else str(index)
            yield from _iter_nested_items(item, path, depth + 1)
    else:
        yield prefix, value


def _find_nested_metric(source: Any, *names: str):
    """Find the first scalar metric whose final path segment matches names."""
    wanted = set(names)
    for path, value in _iter_nested_items(source):
        if not path:
            continue
        path_parts = {part.lower() for part in path.split("/")}
        if "config" in path_parts or "replay_buffer_config" in path_parts:
            continue
        key = path.rsplit("/", 1)[-1]
        if key not in wanted or value is None:
            continue
        if isinstance(value, (int, float)):
            return value
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if hasattr(value, "shape") and getattr(value, "shape", None) == ():
            try:
                return float(value)
            except Exception:
                pass
    return "n/a"


def _fill_optional_learner_stats(stats: dict, policy_stats: dict, result: dict) -> None:
    """Populate SAC/performance stats when RLlib exposes them."""

    def first_present(*names: str):
        for source in (policy_stats, result):
            for name in names:
                if name in source and source[name] is not None:
                    return source[name]
        nested = _find_nested_metric(result, *names)
        if nested != "n/a":
            return nested
        return "n/a"

    stats["actor_loss"] = first_present("actor_loss", "policy_loss")
    stats["critic_loss"] = first_present("critic_loss", "qf_loss", "q_loss")
    stats["alpha_loss"] = first_present("alpha_loss")
    stats["alpha"] = first_present("alpha", "alpha_value")
    if stats["alpha"] == "n/a":
        log_alpha = first_present("log_alpha_value", "curr_log_alpha")
        if isinstance(log_alpha, (int, float)):
            stats["alpha"] = math.exp(float(log_alpha))
    stats["target_entropy"] = first_present("target_entropy")
    stats["replay_buffer_size"] = first_present(
        "replay_buffer_size", "num_steps_trained_this_iter"
    )
    stats["env_steps_per_sec"] = first_present(
        "env_steps_per_sec", "num_env_steps_sampled_throughput_per_sec"
    )
    stats["learner_steps_per_sec"] = first_present(
        "learner_steps_per_sec", "num_env_steps_trained_throughput_per_sec"
    )
    stats["iteration_time_s"] = first_present("time_this_iter_s")


def _print_learner_result_debug(result: dict, iteration: int) -> None:
    """Print compact learner/result key hints when SAC loss metrics are missing."""
    interesting = (
        "learner",
        "learners",
        "loss",
        "alpha",
        "td_error",
        "num_env_steps_trained",
        "num_module_steps_trained",
    )
    matches = []
    for path, value in _iter_nested_items(result):
        lower_path = path.lower()
        if any(token in lower_path for token in interesting):
            shape = getattr(value, "shape", None)
            if shape is not None:
                summary = f"shape={tuple(shape)}"
            elif isinstance(value, (str, int, float, bool)) or value is None:
                summary = repr(value)
            else:
                summary = type(value).__name__
            matches.append(f"{path}={summary}")
        if len(matches) >= 80:
            break
    print(
        "[DogFightEnv][RLlibResult][LEARNER_KEYS] "
        f"iteration={iteration} count={len(matches)}"
    )
    for item in matches:
        print(f"[DogFightEnv][RLlibResult][LEARNER_KEYS] {item}")


def _fill_algorithm_runtime_stats(stats: dict, algorithm: Any) -> None:
    """Fill direct-loop stats that RLlib does not always place in result."""
    replay_buffer = getattr(algorithm, "local_replay_buffer", None)
    if replay_buffer is None:
        return

    if stats.get("replay_buffer_memory_mb") == "n/a":
        stats["replay_buffer_memory_mb"] = _estimate_object_memory_mb(replay_buffer)

    if stats.get("replay_buffer_size") != "n/a":
        return

    for getter_name in ("get_num_timesteps", "get_num_episodes"):
        getter = getattr(replay_buffer, getter_name, None)
        if getter is None:
            continue
        try:
            stats["replay_buffer_size"] = getter()
            return
        except Exception:
            pass

    try:
        stats["replay_buffer_size"] = len(replay_buffer)
    except Exception:
        pass


def _estimate_object_memory_mb(obj: Any) -> Any:
    """Estimate Python object memory recursively, returned in MiB."""
    if obj is None:
        return "n/a"

    seen: set[int] = set()

    def sizeof(value: Any, depth: int = 0) -> int:
        obj_id = id(value)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        size = sys.getsizeof(value, 0)
        nbytes = getattr(value, "nbytes", None)
        if isinstance(nbytes, int):
            size += nbytes
            return size
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            try:
                size += int(value.numel()) * int(value.element_size())
                return size
            except Exception:
                return size
        if depth >= 8:
            return size

        if isinstance(value, dict):
            for key, item in value.items():
                size += sizeof(key, depth + 1)
                size += sizeof(item, depth + 1)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                size += sizeof(item, depth + 1)
        elif hasattr(value, "__dict__"):
            size += sizeof(vars(value), depth + 1)
        return size

    try:
        return sizeof(obj) / (1024.0 * 1024.0)
    except Exception:
        return "n/a"


def _extract_custom_metrics(result: dict) -> dict:
    """Extract DogFightCallbacks custom metrics from result (mean values)."""
    env_metrics = result.get("env_runners", {})
    cm = env_metrics.get("custom_metrics", {})

    def metric(name: str, default: Any = "n/a"):
        for metrics in (cm, result.get("custom_metrics", {})):
            for key in (f"{name}_mean", name):
                if key in metrics and metrics[key] is not None:
                    return metrics[key]
        return default

    extracted = {
        "win_rate":             metric("win"),
        "loss_rate":            metric("loss"),
        "timeout_rate":         metric("timeout"),
        "crash_rate":           metric("crash"),
        "ep_wez_steps":         metric("ep_wez_steps"),
        "ep_mean_distance":     metric("ep_mean_distance"),
        "ep_min_distance":      metric("ep_min_distance"),
        "ep_reward_pursuit":    metric("ep_reward_pursuit"),
        "ep_reward_damage":     metric("ep_reward_damage"),
        "ep_reward_safety":     metric("ep_reward_safety"),
        "ep_reward_survival":   metric("ep_reward_survival"),
        "ep_reward_aim_progress": metric("ep_reward_aim_progress"),
        "ep_reward_aim_quality": metric("ep_reward_aim_quality"),
        "ep_reward_los_rate": metric("ep_reward_los_rate"),
        "ep_reward_cone_dwell": metric("ep_reward_cone_dwell"),
        "ep_reward_residual_l2": metric("ep_reward_residual_l2"),
        "ep_reward_residual_smooth": metric("ep_reward_residual_smooth"),
        "ep_reward_low_speed": metric("ep_reward_low_speed"),
        "ep_altitude_penalty_steps": metric("ep_altitude_penalty_steps"),
        "initial_alpha_deg":    metric("initial_alpha_deg"),
        "initial_ata_deg":      metric("initial_ata_deg"),
        "initial_aa_deg":       metric("initial_aa_deg"),
        "initial_distance_m":   metric("initial_distance_m"),
        "final_ata_deg":        metric("final_ata_deg"),
        "final_aa_deg":         metric("final_aa_deg"),
        "headon_guard_fail":    metric("headon_guard_fail"),
        "action_roll_mean":     metric("action_roll_mean"),
        "action_pitch_mean":    metric("action_pitch_mean"),
        "action_rudder_mean":   metric("action_rudder_mean"),
        "action_throttle_mean": metric("action_throttle_mean"),
        "action_roll_std":      metric("action_roll_std"),
        "action_pitch_std":     metric("action_pitch_std"),
        "action_rudder_std":    metric("action_rudder_std"),
        "action_throttle_std":  metric("action_throttle_std"),
        "action_sat_rate":      metric("action_saturation_rate"),
        **{
            key: metric(key)
            for key in (
                "mean_los_deg",
                "median_los_deg",
                "p95_los_deg",
                "min_los_deg",
                "los_rate_rms_deg_s",
                "mean_ata_deg",
                "min_ata_deg",
                "mean_target_ata_deg",
                "damage_cone_entries",
                "damage_cone_time_s",
                "phase1_cone_time_s",
                "phase2_cone_time_s",
                "phase3_cone_time_s",
                "time_to_first_wez_s",
                "time_to_first_damage_s",
                "mean_speed_m_s",
                "min_speed_m_s",
                "min_altitude_m",
                "gate_active_ratio",
                "gate_entries",
                "gate_exits",
                "gate_mean_active_s",
                "gate_min_active_s",
                "rl_correction_steps",
                "rl_correction_ratio",
                "roll_residual_abs_mean",
                "pitch_residual_abs_mean",
                "yaw_residual_abs_mean",
                "roll_residual_abs_max",
                "pitch_residual_abs_max",
                "yaw_residual_abs_max",
                "action_clipping_ratio",
                "action_saturation_ratio",
                "requested_throttle_residual_abs_mean",
            )
        },
    }
    for metrics in (cm, result.get("custom_metrics", {})):
        for raw_key, value in metrics.items():
            key = raw_key[:-5] if raw_key.endswith("_mean") else raw_key
            if key.startswith("aim_variant_fraction_") and value is not None:
                extracted.setdefault(key, value)
    return extracted


def _extract_progress_metrics(result: dict) -> dict:
    """Extract rollout progress metrics that exist before episodes complete."""
    env_metrics = result.get("env_runners", {})

    def first_present(mapping: dict, keys: tuple[str, ...], default="n/a"):
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
        return default

    return {
        "sampled_steps": first_present(
            env_metrics,
            ("num_env_steps_sampled_lifetime", "num_agent_steps_sampled_lifetime"),
        ),
        "episodes": first_present(
            env_metrics,
            ("num_episodes_lifetime", "num_episodes"),
        ),
    }


def _fmt(val, fmt=".4f"):
    return f"{val:{fmt}}" if isinstance(val, (int, float)) else str(val)


def _build_tune_progress_reporter(algorithm_name: str):
    """Build a Ray Tune table reporter with stable RLlib 2.x metric paths."""
    from ray.tune import CLIReporter

    metric_columns = {
        "training_iteration": "iter",
        "time_total_s": "time_s",
        "env_runners/num_env_steps_sampled_lifetime": "steps",
        "env_runners/num_episodes_lifetime": "eps",
        "env_runners/episode_return_mean": "reward",
        "env_runners/episode_len_mean": "len",
        "env_runners/custom_metrics/win_mean": "win",
        "env_runners/custom_metrics/crash_mean": "crash",
        "env_runners/custom_metrics/ep_wez_steps_mean": "wez",
    }
    if algorithm_name == "sac":
        metric_columns.update({
            "learners/default_policy/actor_loss": "actor",
            "learners/default_policy/critic_loss": "critic",
            "learners/default_policy/alpha_value": "alpha",
        })
    else:
        metric_columns.update({
            "learners/default_policy/entropy": "entropy",
            "learners/default_policy/vf_loss": "vf_loss",
            "learners/default_policy/mean_kl_loss": "kl",
        })
    return CLIReporter(
        metric_columns=metric_columns,
        max_report_frequency=2,
        print_intermediate_tables=True,
    )


def _console_header(algorithm_name: str) -> str:
    base = (
        f"{'Iter':>6} | {'Steps':>10} | {'Eps':>5} | "
        f"{'Reward':>10} | {'WinRate':>8} | {'WEZ_ep':>7} | "
    )
    if algorithm_name == "sac":
        return (
            base
            + f"{'Actor':>9} | {'Critic':>9} | {'Alpha':>8} | {'ReplayMB':>9} |"
        )
    return base + f"{'Entropy':>8} | {'VF_loss':>8} | {'KL':>7} |"


def _console_row(
    algorithm_name: str,
    iteration: int,
    progress: dict,
    reward_mean: Any,
    custom: dict,
    learner_stats: dict,
) -> str:
    base = (
        f"iter=[{iteration}] | "
        f"Steps=[{_fmt(progress['sampled_steps'], '.0f')}] | "
        f"Eps=[{_fmt(progress['episodes'], '.0f')}] | "
        f"Reward=[{_fmt(reward_mean, '.4f')}] | "
        f"WinRate=[{_fmt(custom['win_rate'], '.3f')}] | "
        f"WEZ_ep=[{_fmt(custom['ep_wez_steps'], '.1f')}] | "
    )
    if algorithm_name == "sac":
        return (
            base
            + f"Actor=[{_fmt(learner_stats['actor_loss'], '.4f')}] | "
            f"Critic=[{_fmt(learner_stats['critic_loss'], '.4f')}] | "
            f"Alpha=[{_fmt(learner_stats['alpha'], '.4f')}] | "
            f"ReplayMem=[{_fmt(learner_stats['replay_buffer_memory_mb'], '.1f')}MB]"
        )
    return (
        base
        + f"Entropy=[{_fmt(learner_stats['entropy'], '.4f')}] | "
        f"VF_loss=[{_fmt(learner_stats['vf_loss'], '.4f')}] | "
        f"KL=[{_fmt(learner_stats['kl'], '.4f')}]"
    )


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _resolve_dashboard_root(path: str) -> Path:
    root = Path(path)
    return root if root.is_absolute() else ROOT / root


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a single-agent dogfight policy with RLlib."
    )
    parser.add_argument(
        "--algorithm",
        choices=["ppo", "sac"],
        default="ppo",
        help="RLlib algorithm to use.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of training iterations.",
    )
    parser.add_argument(
        "--framework",
        default="torch",
        choices=["torch"],
        help="Deep learning framework.",
    )
    parser.add_argument(
        "--num-env-runners",
        type=int,
        default=1,
        help="Number of RLlib env runners.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Independent training seed for Python, NumPy, Torch, RLlib, and env runners.",
    )
    parser.add_argument(
        "--ray-num-cpus",
        type=int,
        default=None,
        help="Optional Ray CPU resource cap for local runtime diagnosis/training.",
    )
    parser.add_argument(
        "--runtime-diagnostics",
        action="store_true",
        help="Emit flushed stage/process snapshots and 60-second Python stack dumps.",
    )
    parser.add_argument(
        "--num-envs-per-env-runner",
        type=int,
        default=1,
        help="Number of vectorized envs per env runner.",
    )
    parser.add_argument(
        "--rollout-fragment-length",
        default="auto",
        help="RLlib rollout fragment length, or 'auto'.",
    )
    parser.add_argument(
        "--batch-mode",
        default="truncate_episodes",
        choices=["truncate_episodes", "complete_episodes"],
    )
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
    parser.add_argument(
        "--observation-module",
        default="",
        help="Optional module with custom observation size and build_observation(...).",
    )
    parser.add_argument(
        "--target-mode",
        default="behavior_tree",
        choices=["behavior_tree", "fixed", "loiter", "autopilot"],
    )
    parser.add_argument("--target-behavior-dll", default="AIP_BASE_target.dll")
    parser.add_argument("--bt-rule-xml", default="")
    parser.add_argument("--bt-rule-alias", action="append", default=[])
    parser.add_argument("--target-rule-xml", default="")
    parser.add_argument("--target-rule-alias", action="append", default=[])
    parser.add_argument(
        "--reward-module",
        default="",
        help="Optional module with MY_REWARD_CONFIG and compute_reward(...).",
    )
    parser.add_argument("--max-engage-time", type=float, default=300.0)
    parser.add_argument("--episode-step-limit", type=int, default=18000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target-entropy", default="auto")
    parser.add_argument(
        "--replay-buffer-capacity",
        type=int,
        default=None,
        help="SAC replay buffer capacity. Ignored by PPO.",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=None,
        help="SAC environment steps collected before learner updates begin.",
    )
    parser.add_argument(
        "--model-fcnet-hiddens",
        default=None,
        help="Comma-separated RLlib model hidden sizes, e.g. 512,256,128.",
    )
    parser.add_argument(
        "--model-fcnet-activation",
        default=None,
        help="RLlib model encoder activation, e.g. relu or tanh.",
    )
    parser.add_argument(
        "--model-head-fcnet-hiddens",
        default=None,
        help="Comma-separated RLlib model head hidden sizes, or empty for none.",
    )
    parser.add_argument(
        "--model-head-fcnet-activation",
        default=None,
        help="RLlib model head activation, e.g. relu or tanh.",
    )
    parser.add_argument(
        "--model-vf-share-layers",
        default=None,
        help="Whether PPO value function shares layers: true or false.",
    )
    parser.add_argument(
        "--network-spec-json",
        default="",
        help=(
            "JSON object for DogFight sequence_v1 network layout. "
            "Usually supplied by scripts/run_experiment.py from algo.network."
        ),
    )
    parser.add_argument(
        "--use-lstm",
        action="store_true",
        help="Enable RLlib DefaultModelConfig LSTM for non-SAC algorithms such as PPO.",
    )
    parser.add_argument(
        "--use-lstm-sac",
        action="store_true",
        help="Enable the patched Ray 2.54 SAC actor-LSTM path.",
    )
    parser.add_argument(
        "--lstm-scope",
        choices=["actor_only", "actor_critic"],
        default="actor_only",
        help="SAC LSTM scope: actor_only or actor_critic recurrent Q.",
    )
    parser.add_argument(
        "--lstm-cell-size",
        type=int,
        default=64,
        help="LSTM hidden state size for --use-lstm-sac.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=8,
        help="Replay/train sequence length for --use-lstm-sac.",
    )
    parser.add_argument(
        "--debug-io",
        dest="debug_io",
        action="store_true",
        help="Print recurrent SAC/RLlib debug I/O shape checks.",
    )
    parser.add_argument(
        "--debug-lstm-io",
        dest="debug_io",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--use-lstm-prioritized-replay",
        action="store_true",
        help=(
            "Use patched PrioritizedEpisodeReplayBuffer sequence sampling for "
            "--use-lstm-sac. Requires the RLLibLstm replay patch."
        ),
    )
    parser.add_argument("--output-name", default="f16_single_agent")
    parser.add_argument("--output-tag", default="latest")
    parser.add_argument(
        "--notes",
        default="",
        help="Optional free-text notes for this training run.",
    )
    parser.add_argument(
        "--save-lightweight-bundle",
        dest="save_lightweight_bundle",
        action="store_true",
        default=True,
        help="Save lightweight policy bundles for inference.",
    )
    parser.add_argument(
        "--no-save-lightweight-bundle",
        dest="save_lightweight_bundle",
        action="store_false",
        help="Disable lightweight policy bundle saves.",
    )
    parser.add_argument(
        "--lightweight-bundle-frequency",
        type=int,
        default=0,
        help=(
            "Save a lightweight bundle every N direct-loop iterations. "
            "0 means final bundle only."
        ),
    )
    parser.add_argument(
        "--save-native-checkpoint",
        action="store_true",
        help="Also save the full RLlib checkpoint.",
    )
    parser.add_argument(
        "--restore-checkpoint",
        default="",
        help="Restore a full RLlib native checkpoint before training.",
    )
    parser.add_argument(
        "--init-bundle",
        "--restart-from-bundle",
        dest="init_bundle",
        default="",
        help="Load lightweight policy bundle weights before fresh training.",
    )
    parser.add_argument(
        "--use-tune",
        action="store_true",
        help="Run training through Ray Tune/AIR.",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=0,
        help=(
            "Legacy native/Tune checkpoint frequency in training iterations. "
            "Prefer --native-checkpoint-frequency for direct training."
        ),
    )
    parser.add_argument(
        "--native-checkpoint-frequency",
        type=int,
        default=None,
        help=(
            "Save an RLlib native checkpoint every N direct-loop iterations. "
            "0 means final native checkpoint only when enabled."
        ),
    )
    parser.add_argument(
        "--dashboard-logdir",
        default="artifacts/dashboard",
        help="Dashboard JSONL root directory.",
    )
    parser.add_argument(
        "--disable-dashboard-log",
        action="store_true",
        help="Disable dashboard metrics.jsonl output.",
    )
    parser.add_argument(
        "--policy-probe-interval",
        type=int,
        default=0,
        help=(
            "Log fixed policy probe actions every N iterations. "
            "0 disables policy_probe.csv/jsonl."
        ),
    )
    parser.add_argument(
        "--policy-probe-steps",
        type=int,
        default=4,
        help="Number of recurrent inference steps per policy probe.",
    )
    parser.add_argument(
        "--no-policy-probe-print",
        action="store_true",
        help="Write policy probe files without console summaries.",
    )
    parser.add_argument(
        "--engagement-log-interval",
        type=int,
        default=0,
        help=(
            "Run a short policy-vs-target replay every N iterations and save "
            "Tacview CSV logs. 0 disables engagement_replays/."
        ),
    )
    parser.add_argument(
        "--engagement-log-steps",
        type=int,
        default=600,
        help="Maximum environment steps per engagement replay episode.",
    )
    parser.add_argument(
        "--engagement-log-episodes",
        type=int,
        default=1,
        help="Number of replay episodes to save at each engagement-log interval.",
    )
    parser.add_argument(
        "--no-engagement-log-print",
        action="store_true",
        help="Write engagement replay files without console summaries.",
    )
    parser.add_argument(
        "--experiment-yaml",
        default="",
        help="Optional YAML experiment definition; env_config is deep-merged.",
    )
    args = parser.parse_args()
    if args.restore_checkpoint and args.init_bundle:
        parser.error("--restore-checkpoint and --init-bundle are mutually exclusive.")
    return args


def _build_model_config_args(args) -> dict[str, Any]:
    model_config = {
        "fcnet_hiddens": args.model_fcnet_hiddens,
        "fcnet_activation": args.model_fcnet_activation,
        "head_fcnet_hiddens": args.model_head_fcnet_hiddens,
        "head_fcnet_activation": args.model_head_fcnet_activation,
        "vf_share_layers": args.model_vf_share_layers,
    }
    if args.network_spec_json:
        model_config["network_spec"] = json.loads(args.network_spec_json)
    model_config["enabled"] = any(value is not None for value in model_config.values())
    return model_config


def _build_algorithm_args(args) -> dict:
    return {
        "framework": args.framework,
        "num_env_runners": args.num_env_runners,
        "num_envs_per_env_runner": args.num_envs_per_env_runner,
        "rollout_fragment_length": args.rollout_fragment_length,
        "batch_mode": args.batch_mode,
        "lr": args.lr,
        "gamma": args.gamma,
        "train_batch_size": args.train_batch_size,
        "minibatch_size": args.minibatch_size,
        "gae_lambda": args.gae_lambda,
        "clip_param": args.clip_param,
        "tau": args.tau,
        "target_entropy": args.target_entropy,
        "replay_buffer_capacity": args.replay_buffer_capacity,
        "learning_starts": args.learning_starts,
        "model_config": _build_model_config_args(args),
        "network_spec": args.network_spec_json,
        "use_lstm": args.use_lstm,
        "use_lstm_sac": args.use_lstm_sac,
        "use_lstm_prioritized_replay": args.use_lstm_prioritized_replay,
        "lstm_scope": args.lstm_scope,
        "lstm_cell_size": args.lstm_cell_size,
        "max_seq_len": args.max_seq_len,
        "debug_io": args.debug_io,
        "seed": args.seed,
    }


def _sync_lstm_args_from_init_bundle(args) -> None:
    """Align SAC LSTM architecture args with a lightweight bundle before build."""

    if not args.init_bundle:
        return

    bundle_path = Path(args.init_bundle)
    metadata_path = bundle_path / "metadata.json"
    if not metadata_path.exists():
        return

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    bundle_meta = payload.get("metadata", {})
    saved_model_config = bundle_meta.get("model_config") or {}
    bundle_uses_lstm = bool(
        bundle_meta.get("use_lstm_sac")
        or saved_model_config.get("use_lstm")
    )
    if not bundle_uses_lstm:
        return

    algorithm_name = normalize_algorithm_name(args.algorithm)
    lstm_scope = (
        saved_model_config.get("dogfight_lstm_scope")
        or bundle_meta.get("lstm_scope")
        or "actor_only"
    )
    lstm_cell_size = int(
        saved_model_config.get("lstm_cell_size")
        or bundle_meta.get("lstm_cell_size")
        or args.lstm_cell_size
    )
    max_seq_len = int(
        saved_model_config.get("max_seq_len")
        or bundle_meta.get("max_seq_len")
        or args.max_seq_len
    )
    network_spec = (
        saved_model_config.get("dogfight_network_spec")
        or bundle_meta.get("network_spec")
    )

    if algorithm_name != "sac":
        changed = (
            not args.use_lstm
            or args.lstm_cell_size != lstm_cell_size
            or args.max_seq_len != max_seq_len
            or (
                network_spec is not None
                and args.network_spec_json
                != json.dumps(network_spec, ensure_ascii=False, separators=(",", ":"))
            )
        )
        args.use_lstm = True
        args.lstm_cell_size = lstm_cell_size
        args.max_seq_len = max_seq_len
        if network_spec is not None:
            args.network_spec_json = json.dumps(
                network_spec, ensure_ascii=False, separators=(",", ":")
            )
        if changed:
            print(
                "[DogFightEnv][LSTM_RESUME] "
                f"init_bundle={bundle_path} use_lstm=True "
                f"lstm_cell_size={lstm_cell_size} max_seq_len={max_seq_len} "
                f"network_type={(network_spec or {}).get('type')}"
            )
        return

    changed = (
        not args.use_lstm_sac
        or args.lstm_scope != lstm_scope
        or args.lstm_cell_size != lstm_cell_size
        or args.max_seq_len != max_seq_len
        or (
            network_spec is not None
            and args.network_spec_json
            != json.dumps(network_spec, ensure_ascii=False, separators=(",", ":"))
        )
    )
    args.use_lstm_sac = True
    args.lstm_scope = lstm_scope
    args.lstm_cell_size = lstm_cell_size
    args.max_seq_len = max_seq_len
    if network_spec is not None:
        args.network_spec_json = json.dumps(
            network_spec, ensure_ascii=False, separators=(",", ":")
        )
    if changed:
        print(
            "[DogFightEnv][LSTM_RESUME] "
            f"init_bundle={bundle_path} use_lstm_sac=True "
            f"lstm_scope={lstm_scope} lstm_cell_size={lstm_cell_size} "
            f"max_seq_len={max_seq_len} "
            f"network_type={(network_spec or {}).get('type')}"
        )


def _native_checkpoint_frequency(args) -> int:
    """Return the native checkpoint interval, including the legacy alias."""
    if args.native_checkpoint_frequency is not None:
        return max(0, int(args.native_checkpoint_frequency))
    return max(0, int(args.checkpoint_frequency))


def _build_observation_bundle_metadata(
    env_config: dict,
    fallback_mode: str,
) -> dict[str, Any]:
    """Return serializable observation metadata for lightweight bundles."""
    mode = str(env_config.get("observation_mode", fallback_mode))
    summary = env_config.get("observation_summary")
    if isinstance(summary, dict):
        observation_summary = dict(summary)
    else:
        observation_summary = None

    observation_size = None
    if observation_summary is not None:
        observation_size = _to_positive_int(observation_summary.get("size"))
    if observation_size is None:
        observation_size = _to_positive_int(env_config.get("observation_size"))
    if observation_size is None:
        observation_size = _to_positive_int(builtin_observation_size(mode))

    return {
        "obs_mode": mode,
        "observation_mode": mode,
        "observation_module": env_config.get("observation_module", ""),
        "observation_size": observation_size,
        "observation_summary": observation_summary,
    }


def _to_positive_int(value: Any) -> int | None:
    """Return a positive integer value or None if unavailable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _build_bundle_metadata(
    args,
    algorithm_name: str,
    env_config: dict,
    record_dir: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata shared by final and periodic lightweight bundles."""
    observation_metadata = _build_observation_bundle_metadata(
        env_config,
        args.observation_mode,
    )
    metadata = {
        "model_name": args.output_name,
        "algorithm": algorithm_name,
        **observation_metadata,
        "action_dim": 4,
        "env_class": "DogFightWrapper",
        "target_mode": args.target_mode,
        "record_dir": str(record_dir),
        "use_lstm": args.use_lstm,
        "use_lstm_sac": args.use_lstm_sac,
        "use_lstm_prioritized_replay": (
            args.use_lstm_prioritized_replay if args.use_lstm_sac else None
        ),
        "lstm_cell_size": (
            args.lstm_cell_size if args.use_lstm or args.use_lstm_sac else None
        ),
        "max_seq_len": (
            args.max_seq_len if args.use_lstm or args.use_lstm_sac else None
        ),
        "lstm_scope": args.lstm_scope if args.use_lstm_sac else None,
        "network_spec": (
            json.loads(args.network_spec_json) if args.network_spec_json else None
        ),
        "training_seed": args.seed,
    }
    if extra:
        metadata.update(extra)
    return metadata


def _save_lightweight_bundle(
    algorithm,
    bundle_dir: Path,
    args,
    algorithm_name: str,
    env_config: dict,
    record_dir: Path,
    *,
    label: str,
    iteration: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a lightweight policy bundle with common metadata."""
    metadata_extra = dict(extra or {})
    if iteration is not None:
        metadata_extra["iteration"] = iteration
    save_lightweight_policy_bundle(
        algorithm,
        bundle_dir,
        metadata=_build_bundle_metadata(
            args,
            algorithm_name,
            env_config,
            record_dir,
            extra=metadata_extra,
        ),
    )
    print(f"{label} lightweight bundle saved to {bundle_dir}")


def _save_native_checkpoint(algorithm, checkpoint_dir: Path, *, label: str) -> None:
    """Save an RLlib native checkpoint to the requested directory."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = algorithm.save(str(checkpoint_dir))
    print(f"{label} rllib checkpoint saved to {checkpoint_path}")


def _save_tune_outputs(args, algorithm_name: str, config, env_config: dict, result_grid) -> None:
    from ray.rllib.algorithms.algorithm import Algorithm

    result = result_grid[0]
    checkpoint = getattr(result, "checkpoint", None)
    if checkpoint is None:
        try:
            checkpoint = result.get_best_checkpoint(
                metric="env_runners/episode_return_mean",
                mode="max",
            )
        except Exception:
            checkpoint = None
    if checkpoint is None:
        raise RuntimeError("Tune run finished without a checkpoint to export.")

    algorithm = Algorithm.from_checkpoint(checkpoint)
    try:
        bundle_dir = ROOT / "artifacts" / "models" / args.output_name / args.output_tag
        record_dir = ROOT / "artifacts" / "records" / args.output_name / args.output_tag
        if args.save_lightweight_bundle:
            _save_lightweight_bundle(
                algorithm,
                bundle_dir,
                args,
                algorithm_name,
                env_config,
                record_dir,
                label="final",
                extra={"tune_checkpoint": str(checkpoint)},
            )
        else:
            print("lightweight bundle save skipped by --no-save-lightweight-bundle")

        metrics = _json_safe(getattr(result, "metrics", {}) or {})
        if not args.disable_dashboard_log:
            experiment_config = load_experiment_metadata(
                args.experiment_yaml,
                script_name="train_rllib",
                cli_argv=sys.argv[1:],
            )
            logger = DashboardJsonlLogger(
                _resolve_dashboard_root(args.dashboard_logdir),
                f"{args.output_name}_{args.output_tag}",
                config={
                    **experiment_config,
                    "algorithm": algorithm_name,
                    "cli_args": vars(args),
                    "env_config": env_config,
                },
            )
            env_metrics = metrics.get("env_runners", {})
            row = {
                "iter": metrics.get("training_iteration", args.iterations),
                "sampled_steps": env_metrics.get("num_env_steps_sampled_lifetime"),
                "episodes": env_metrics.get("num_episodes_lifetime"),
                "reward_mean": env_metrics.get("episode_return_mean"),
                "ep_len_mean": env_metrics.get("episode_len_mean"),
            }
            logger.write_row(row)
            print(f"dashboard log saved to {logger.metrics_path}")
        save_training_record(
            output_dir=record_dir,
            algorithm_name=algorithm_name,
            cli_args=vars(args),
            env_config=env_config,
            algorithm_config=_json_safe(config.to_dict() if hasattr(config, "to_dict") else {}),
            result_history=[{"iteration": "tune_final", **metrics}],
            workspace_root=ROOT,
        )
        copy_experiment_yaml(args.experiment_yaml, record_dir)
        print(f"training record saved to {record_dir}")
    finally:
        algorithm.stop()


def _run_with_tune(args, algorithm_name: str, config, env_config: dict) -> None:
    from ray import air, tune
    from ray.air import CheckpointConfig

    tune_dir = ROOT / "artifacts" / "tune" / args.output_name
    trainable = algorithm_name.upper()
    checkpoint_config = CheckpointConfig(
        checkpoint_frequency=_native_checkpoint_frequency(args),
        checkpoint_at_end=True,
        num_to_keep=2,
    )
    run_config = air.RunConfig(
        name=args.output_tag,
        storage_path=str(tune_dir),
        stop={"training_iteration": args.iterations},
        checkpoint_config=checkpoint_config,
        progress_reporter=_build_tune_progress_reporter(algorithm_name),
        verbose=1,
    )
    tuner = tune.Tuner(
        trainable,
        param_space=config.to_dict() if hasattr(config, "to_dict") else dict(config),
        run_config=run_config,
    )
    print(
        f"starting Tune run: {trainable}, "
        f"env_runners={args.num_env_runners}, "
        f"envs_per_runner={args.num_envs_per_env_runner}"
    )
    result_grid = tuner.fit()
    _save_tune_outputs(args, algorithm_name, config, env_config, result_grid)


def _seed_training_runtime(seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:  # pragma: no cover - Torch is present for RLlib runs.
            pass


def _run_training(args):
    _runtime_stage("training_seed_start", training_seed=args.seed)
    _seed_training_runtime(args.seed)
    _runtime_stage("training_seed_done", training_seed=args.seed)
    algorithm_name = normalize_algorithm_name(args.algorithm)
    if args.use_tune and (args.restore_checkpoint or args.init_bundle):
        raise RuntimeError(
            "Checkpoint/bundle restart is supported by the direct training loop, "
            "not --use-tune."
        )
    _sync_lstm_args_from_init_bundle(args)

    env_config = {
        "observation_mode": args.observation_mode,
        "target_mode": args.target_mode,
        "target_behavior_dll": args.target_behavior_dll,
        "ownship_control_mode": "rl",
        "max_engage_time": args.max_engage_time,
        "episode_step_limit": args.episode_step_limit,
    }
    deep_update(env_config, load_experiment_env_config(args.experiment_yaml, ROOT))
    if args.reward_module:
        env_config["reward_module"] = args.reward_module
    if args.observation_module:
        env_config["observation_module"] = args.observation_module
    _runtime_stage("env_preview_start")
    env_preview = env_creator(env_config)
    env_config["reward"] = dict(env_preview.config["reward"])
    env_config["wez"] = dict(env_preview.config["wez"])
    if args.observation_module:
        env_config["observation_mode"] = env_preview.config["observation_mode"]
        env_config["observation_module"] = args.observation_module
        env_config["observation_summary"] = dict(env_preview.config["observation_summary"])
    obs_shape = getattr(env_preview.observation_space, "shape", (0,))
    action_shape = getattr(env_preview.action_space, "shape", (4,))
    probe_obs_dim = int(obs_shape[0]) if obs_shape else 0
    probe_action_dim = int(action_shape[0]) if action_shape else 4
    env_preview.close()
    _runtime_stage(
        "env_preview_done",
        observation_dim=probe_obs_dim,
        action_dim=probe_action_dim,
    )

    env_name = "dogfight-single-agent-v0"
    register_env(env_name, env_creator)

    _runtime_stage("algorithm_config_start", algorithm=algorithm_name)
    config = build_algorithm_config(
        algorithm_name=algorithm_name,
        env_name=env_name,
        env_config=env_config,
        args=_build_algorithm_args(args),
    )
    _runtime_stage("algorithm_config_done", algorithm=algorithm_name)

    _ensure_ray_runtime_env(num_cpus=args.ray_num_cpus)

    if args.use_tune:
        _run_with_tune(args, algorithm_name, config, env_config)
        return

    _runtime_stage("algorithm_build_start", algorithm=algorithm_name)
    algorithm = config.build_algo(
        logger_creator=_build_algorithm_logger_creator(args),
    )
    _runtime_stage("algorithm_build_done", algorithm=algorithm_name)
    if args.restore_checkpoint:
        checkpoint_path = Path(args.restore_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"restore checkpoint not found: {checkpoint_path}")
        print(f"restoring native RLlib checkpoint from {checkpoint_path}")
        algorithm.restore(str(checkpoint_path))
    elif args.init_bundle:
        bundle_path = Path(args.init_bundle)
        if not bundle_path.exists():
            raise FileNotFoundError(f"lightweight bundle not found: {bundle_path}")
        print(f"loading lightweight bundle weights from {bundle_path}")
        apply_lightweight_policy_bundle(algorithm, bundle_path)

    native_checkpoint_dir = None
    result_history = []

    # CSV log setup
    log_dir = ROOT / "artifacts" / "logs" / args.output_name / args.output_tag
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"
    _CSV_FIELDS = [
        "iter", "sampled_steps", "episodes", "reward_mean", "ep_len_mean",
        "win_rate", "loss_rate", "timeout_rate", "crash_rate",
        "ep_wez_steps", "ep_mean_distance", "ep_min_distance",
        "ep_reward_pursuit", "ep_reward_damage", "ep_reward_safety",
        "ep_reward_survival", "ep_reward_aim_progress", "ep_reward_aim_quality",
        "ep_reward_los_rate", "ep_reward_cone_dwell", "ep_reward_residual_l2",
        "ep_reward_residual_smooth", "ep_reward_low_speed",
        "ep_altitude_penalty_steps",
        "initial_alpha_deg", "initial_ata_deg", "initial_aa_deg",
        "initial_distance_m", "final_ata_deg", "final_aa_deg",
        "headon_guard_fail",
        "action_roll_mean", "action_pitch_mean", "action_rudder_mean",
        "action_throttle_mean", "action_roll_std", "action_pitch_std",
        "action_rudder_std", "action_throttle_std",
        "action_sat_rate",
        "mean_los_deg", "median_los_deg", "p95_los_deg", "min_los_deg",
        "los_rate_rms_deg_s", "mean_ata_deg", "min_ata_deg",
        "mean_target_ata_deg", "damage_cone_entries", "damage_cone_time_s",
        "phase1_cone_time_s", "phase2_cone_time_s", "phase3_cone_time_s",
        "time_to_first_wez_s", "time_to_first_damage_s", "mean_speed_m_s",
        "min_speed_m_s", "min_altitude_m", "gate_active_ratio",
        "gate_entries", "gate_exits", "gate_mean_active_s", "gate_min_active_s",
        "rl_correction_steps", "rl_correction_ratio", "roll_residual_abs_mean",
        "pitch_residual_abs_mean", "yaw_residual_abs_mean", "roll_residual_abs_max",
        "pitch_residual_abs_max", "yaw_residual_abs_max", "action_clipping_ratio",
        "action_saturation_ratio", "requested_throttle_residual_abs_mean",
        "policy_loss", "vf_loss", "entropy", "kl", "clip_frac", "explained_var",
        "actor_loss", "critic_loss", "alpha_loss", "alpha", "target_entropy",
        "replay_buffer_size", "replay_buffer_memory_mb", "env_steps_per_sec",
        "learner_steps_per_sec", "iteration_time_s",
    ]
    scenario_variants = (
        env_config.get("initial_scenario", {}).get("variants", [])
    )
    _CSV_FIELDS.extend(
        aim_variant_metric_name(variant.get("name", index))
        for index, variant in enumerate(scenario_variants)
    )
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDS)
    csv_writer.writeheader()
    policy_probe_logger = PolicyProbeLogger(
        log_dir,
        obs_dim=probe_obs_dim,
        action_dim=probe_action_dim,
        interval=args.policy_probe_interval,
        sequence_steps=args.policy_probe_steps,
        print_to_console=not args.no_policy_probe_print,
    )
    engagement_replay_logger = EngagementReplayLogger(
        log_dir,
        env_factory=env_creator,
        env_config=env_config,
        interval=args.engagement_log_interval,
        max_steps=args.engagement_log_steps,
        episodes=args.engagement_log_episodes,
        print_to_console=not args.no_engagement_log_print,
    )
    dashboard_logger = None
    if not args.disable_dashboard_log:
        experiment_config = load_experiment_metadata(
            args.experiment_yaml,
            script_name="train_rllib",
            cli_argv=sys.argv[1:],
        )
        dashboard_logger = DashboardJsonlLogger(
            _resolve_dashboard_root(args.dashboard_logdir),
            f"{args.output_name}_{args.output_tag}",
            config={
                **experiment_config,
                "algorithm": algorithm_name,
                "cli_args": vars(args),
                "env_config": env_config,
                "csv_path": str(csv_path),
            },
        )

    try:
        policy_probe_logger.__enter__()
        engagement_replay_logger.__enter__()
        bundle_root = (
            ROOT / "artifacts" / "models" / args.output_name / args.output_tag
        )
        checkpoint_root = (
            ROOT / "artifacts" / "checkpoints" / args.output_name / args.output_tag
        )
        record_dir = (
            ROOT / "artifacts" / "records" / args.output_name / args.output_tag
        )
        bundle_frequency = max(0, int(args.lightweight_bundle_frequency))
        native_frequency = _native_checkpoint_frequency(args)

        for iteration in range(args.iterations):
            _runtime_stage("train_iteration_start", iteration=iteration)
            result = algorithm.train()
            _runtime_stage("train_iteration_done", iteration=iteration)
            env_metrics   = result.get("env_runners", {})
            reward_mean      = env_metrics.get("episode_return_mean", "n/a")
            episode_len_mean = env_metrics.get("episode_len_mean", "n/a")
            progress = _extract_progress_metrics(result)
            learner_stats = _extract_learner_stats(result)
            _fill_algorithm_runtime_stats(learner_stats, algorithm)
            if (
                args.debug_io
                and args.use_lstm_sac
                and learner_stats.get("actor_loss") == "n/a"
                and iteration >= 4
                and not getattr(algorithm, "_dogfight_printed_learner_keys", False)
            ):
                _print_learner_result_debug(result, iteration)
                setattr(algorithm, "_dogfight_printed_learner_keys", True)
            custom        = _extract_custom_metrics(result)

            row = {
                "iter":              iteration,
                "sampled_steps":     progress["sampled_steps"],
                "episodes":          progress["episodes"],
                "reward_mean":       reward_mean,
                "ep_len_mean":       episode_len_mean,
                **custom,
                **learner_stats,
            }
            csv_writer.writerow(row)
            csv_file.flush()
            if dashboard_logger is not None:
                dashboard_logger.write_row(row)
            policy_probe_logger.maybe_log(
                algorithm,
                iteration=iteration,
                sampled_steps=progress["sampled_steps"],
            )
            engagement_replay_logger.maybe_log(
                algorithm,
                iteration=iteration,
                sampled_steps=progress["sampled_steps"],
            )

            result_history.append({
                "iteration":       iteration,
                "reward_mean":     reward_mean,
                "episode_len_mean": episode_len_mean,
                **custom,
                **learner_stats,
            })

            # Console row
            print(_console_row(
                algorithm_name,
                iteration,
                progress,
                reward_mean,
                custom,
                learner_stats,
            ))
            iteration_number = iteration + 1
            if args.save_lightweight_bundle and bundle_frequency > 0:
                if iteration_number % bundle_frequency == 0:
                    periodic_bundle_dir = bundle_root / f"bundle_{iteration_number:06d}"
                    _save_lightweight_bundle(
                        algorithm,
                        periodic_bundle_dir,
                        args,
                        algorithm_name,
                        env_config,
                        record_dir,
                        label=f"periodic iter {iteration_number}",
                        iteration=iteration_number,
                    )
            if args.save_native_checkpoint and native_frequency > 0:
                if iteration_number % native_frequency == 0:
                    checkpoint_dir = (
                        checkpoint_root / f"checkpoint_{iteration_number:06d}"
                    )
                    _save_native_checkpoint(
                        algorithm,
                        checkpoint_dir,
                        label=f"periodic iter {iteration_number}",
                    )
        csv_file.close()
        print(f"training log saved to {csv_path}")
        if dashboard_logger is not None:
            print(f"dashboard log saved to {dashboard_logger.metrics_path}")
        if policy_probe_logger.enabled:
            print(f"policy probe CSV saved to {policy_probe_logger.csv_path}")
            print(f"policy probe JSONL saved to {policy_probe_logger.jsonl_path}")

        if args.save_lightweight_bundle:
            _save_lightweight_bundle(
                algorithm,
                bundle_root,
                args,
                algorithm_name,
                env_config,
                record_dir,
                label="final",
            )
        else:
            print("lightweight bundle save skipped by --no-save-lightweight-bundle")

        save_training_record(
            output_dir=record_dir,
            algorithm_name=algorithm_name,
            cli_args=vars(args),
            env_config=env_config,
            algorithm_config=_json_safe(config.to_dict() if hasattr(config, "to_dict") else {}),
            result_history=result_history,
            workspace_root=ROOT,
        )
        copy_experiment_yaml(args.experiment_yaml, record_dir)
        print(f"training record saved to {record_dir}")

        if args.save_native_checkpoint:
            _save_native_checkpoint(
                algorithm,
                checkpoint_root / "checkpoint_final",
                label="final",
            )
    finally:
        policy_probe_logger.__exit__(None, None, None)
        engagement_replay_logger.__exit__(None, None, None)
        if policy_probe_logger.enabled:
            print(f"policy probe CSV saved to {policy_probe_logger.csv_path}")
            print(f"policy probe JSONL saved to {policy_probe_logger.jsonl_path}")
        if engagement_replay_logger.enabled:
            print(
                "engagement replay index saved to "
                f"{engagement_replay_logger.csv_path}"
            )
            print(
                "engagement replay JSONL saved to "
                f"{engagement_replay_logger.jsonl_path}"
            )
        algorithm.stop()


def main():
    args = parse_args()
    _set_runtime_diagnostics(args.runtime_diagnostics)
    try:
        with ExitStack() as stack:
            if args.bt_rule_xml:
                stack.enter_context(
                    activate_rule_xml(
                        args.bt_rule_xml,
                        ROOT,
                        aliases=args.bt_rule_alias,
                        include_default=False,
                    )
                )
            if args.target_rule_xml:
                stack.enter_context(
                    activate_rule_xml(
                        args.target_rule_xml,
                        ROOT,
                        aliases=args.target_rule_alias,
                        include_default=False,
                    )
                )
            _run_training(args)
    finally:
        _disable_runtime_diagnostics()


if __name__ == "__main__":
    main()
