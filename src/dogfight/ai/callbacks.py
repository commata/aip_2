from __future__ import annotations

import re

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks


def aim_variant_metric_name(name: object) -> str:
    """curriculum variant 이름을 안정적인 평균 빈도 metric key로 변환한다."""
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", str(name).strip()).strip("_")
    return f"aim_variant_fraction_{normalized.lower() or 'unnamed'}"


def target_profile_metric_name(name: object) -> str:
    """target profile 이름을 안정적인 episode fraction metric으로 변환한다."""
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", str(name).strip()).strip("_")
    return f"target_profile_fraction_{normalized.lower() or 'unnamed'}"


class DogFightCallbacks(DefaultCallbacks):
    """RLlib callbacks that collect per-episode dogfight metrics.

    Metrics recorded in episode.custom_metrics (auto-aggregated by RLlib):
      Outcome    : win, loss, draw, timeout, crash
      Reward     : ep_reward_{step,pursuit,damage,safety,terminal}
      Tactical   : ep_wez_steps, ep_mean_distance, ep_min_distance,
                   ep_altitude_penalty_steps, initial/final ATA/AA,
                   headon_guard_fail
      Action     : action_{roll,pitch,rudder,throttle}_{mean,std},
                   action_saturation_rate
    """

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def on_episode_start(self, *, episode, **kwargs):
        self._episode_data(episode)["actions"] = []

    def on_episode_step(self, *, episode, **kwargs):
        action = self._last_action(episode)
        if action is not None:
            try:
                self._episode_data(episode)["actions"].append(
                    np.asarray(action, dtype=np.float32)
                )
            except Exception:
                pass

    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs):
        info = self._last_info(episode)
        if not info:
            return

        # ── Outcome ───────────────────────────────────────────────────────
        outcome = info.get("outcome", "other")
        for key in ("win", "loss", "draw", "timeout", "crash"):
            self._record_metric(episode, metrics_logger, key, float(outcome == key))

        # ── Reward components (cumulative episode totals) ─────────────────
        for key, val in info.get("ep_reward_components", {}).items():
            self._record_metric(
                episode, metrics_logger, f"ep_reward_{key}", float(val)
            )

        # ── Tactical metrics ──────────────────────────────────────────────
        self._record_metric(
            episode, metrics_logger, "ep_wez_steps", float(info.get("ep_wez_steps", 0))
        )
        self._record_metric(
            episode,
            metrics_logger,
            "ep_mean_distance",
            float(info.get("ep_mean_distance", 0.0)),
        )
        self._record_metric(
            episode,
            metrics_logger,
            "ep_min_distance",
            float(info.get("ep_min_distance", 0.0)),
        )
        self._record_metric(
            episode,
            metrics_logger,
            "ep_altitude_penalty_steps",
            float(info.get("ep_altitude_penalty_steps", 0)),
        )
        for key in (
            "initial_alpha_deg",
            "initial_ata_deg",
            "initial_aa_deg",
            "initial_distance_m",
            "final_ata_deg",
            "final_aa_deg",
        ):
            if key in info:
                self._record_metric(
                    episode, metrics_logger, key, float(info.get(key, 0.0))
                )
        self._record_metric(
            episode,
            metrics_logger,
            "headon_guard_fail",
            float(bool(info.get("headon_guard_fail", False))),
        )
        selected_variant = info.get("aim_curriculum_variant_name")
        variant_names = info.get("aim_curriculum_variant_names", [])
        if selected_variant is not None and isinstance(variant_names, (list, tuple)):
            for variant_name in variant_names:
                self._record_metric(
                    episode,
                    metrics_logger,
                    aim_variant_metric_name(variant_name),
                    float(str(variant_name) == str(selected_variant)),
                )
        selected_profile = info.get("target_profile_id")
        profile_names = info.get("target_profile_ids", [])
        if selected_profile is not None and isinstance(profile_names, (list, tuple)):
            for profile_name in profile_names:
                self._record_metric(
                    episode,
                    metrics_logger,
                    target_profile_metric_name(profile_name),
                    float(str(profile_name) == str(selected_profile)),
                )

        # ── Aim and hybrid telemetry ─────────────────────────────────────
        maneuver = info.get("maneuver_telemetry", {}) or {}
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
        ):
            value = maneuver.get(key)
            if value is not None:
                self._record_metric(episode, metrics_logger, key, float(value))

        provider = info.get("ownship_provider_telemetry", {}) or {}
        gate_kind = provider.get("residual_training_gate_kind")
        if gate_kind in ("aim", "offensive", "rear120"):
            prefix = (
                "rear120_activation"
                if gate_kind == "rear120"
                else f"{gate_kind}_gate"
            )
            sim_hz = max(1.0, float(maneuver.get("sim_hz", 60)))
            gate_steps = float(provider.get(f"{prefix}_steps", 0))
            correction_steps = float(provider.get("rl_correction_steps", 0))
            hybrid_metrics = {
                "gate_active_ratio": provider.get(f"{prefix}_active_ratio", 0.0),
                "gate_entries": provider.get(f"{prefix}_entries", 0),
                "gate_exits": provider.get(f"{prefix}_exits", 0),
                "gate_mean_active_s": provider.get(
                    f"{prefix}_mean_active_steps", 0
                )
                / sim_hz,
                "gate_min_active_s": provider.get(
                    f"{prefix}_min_active_steps", 0
                )
                / sim_hz,
                "rl_correction_steps": correction_steps,
                "rl_correction_ratio": correction_steps / max(1.0, gate_steps),
                "action_clipping_ratio": float(
                    provider.get("action_clipped_steps", 0)
                )
                / max(1.0, correction_steps),
                "action_saturation_ratio": float(
                    provider.get("action_saturated_steps", 0)
                )
                / max(1.0, correction_steps),
                "requested_throttle_residual_abs_mean": provider.get(
                    "requested_throttle_residual_abs_mean", 0.0
                ),
            }
            for key, value in hybrid_metrics.items():
                self._record_metric(episode, metrics_logger, key, float(value))
            for metric_name, values in (
                ("residual_abs_mean", provider.get("rl_correction_abs_mean", [])),
                ("residual_abs_max", provider.get("rl_correction_abs_max", [])),
            ):
                for axis, value in zip(("roll", "pitch", "yaw"), values):
                    self._record_metric(
                        episode,
                        metrics_logger,
                        f"{axis}_{metric_name}",
                        float(value),
                    )

        batch_contract = info.get("training_batch_contract", {}) or {}
        for key in (
            "eligible_sample_fraction",
            "rear120_sample_fraction",
            "offensive_sample_fraction",
            "pre_aim_sample_fraction",
            "ineligible_sample_count",
            "boundary_exit_transition_count",
        ):
            if key in batch_contract:
                self._record_metric(
                    episode, metrics_logger, key, float(batch_contract[key])
                )
        for histogram_name, counts in batch_contract.get("histograms", {}).items():
            for bin_name, count in counts.items():
                self._record_metric(
                    episode,
                    metrics_logger,
                    f"{histogram_name}_hist_{bin_name}",
                    float(count),
                )

        # ── Action distribution ───────────────────────────────────────────
        actions = self._episode_data(episode).get("actions", [])
        if actions:
            arr = np.stack(actions)  # (steps, 4)
            means = arr.mean(axis=0)
            stds = arr.std(axis=0)
            for i, name in enumerate(("roll", "pitch", "rudder", "throttle")):
                self._record_metric(
                    episode, metrics_logger, f"action_{name}_mean", float(means[i])
                )
                self._record_metric(
                    episode, metrics_logger, f"action_{name}_std", float(stds[i])
                )
            # Saturation: fraction of steps where any axis hits ±1
            self._record_metric(
                episode,
                metrics_logger,
                "action_saturation_rate",
                float(np.mean(np.abs(arr) >= 0.99)),
            )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _episode_data(episode) -> dict:
        """Return mutable per-episode storage for old and new RLlib APIs."""
        if hasattr(episode, "user_data"):
            return episode.user_data
        return episode.custom_data

    @staticmethod
    def _record_metric(episode, metrics_logger, key: str, value: float) -> None:
        """Record a metric through the callback API available in this RLlib version."""
        if hasattr(episode, "custom_metrics"):
            episode.custom_metrics[key] = value
        elif metrics_logger is not None:
            metrics_logger.log_value(("custom_metrics", key), value, reduce="mean", window=100)

    @staticmethod
    def _last_action(episode):
        """Retrieve last action, handling both old and new RLlib API."""
        try:
            return episode.last_action_for()
        except Exception:
            pass

        try:
            return episode.get_actions(-1)
        except Exception:
            return None

    @staticmethod
    def _last_info(episode) -> dict:
        """Retrieve last info dict, handling both old and new RLlib API."""
        try:
            return episode.last_info_for() or {}
        except TypeError:
            # New API: requires agent_id kwarg
            try:
                return episode.last_info_for(agent_id=None) or {}
            except Exception:
                pass
        except Exception:
            pass
        try:
            return episode.get_infos(-1) or {}
        except Exception:
            pass
        return {}
