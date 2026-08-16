from __future__ import annotations

import numpy as np

from dogfight.envs.observation import aim_residual_geometry
from dogfight.sim.state_schema import StateIndex


AIM_RESIDUAL_REWARD_DEFAULTS = {
    "step_penalty": -0.001,
    "aim_progress_scale": 0.40,
    "aim_quality_scale": 0.01,
    "los_rate_penalty_scale": 0.02,
    "cone_dwell_scale": 0.02,
    "damage_scale": 80.0,
    "residual_l2_penalty_scale": 0.02,
    "residual_smooth_penalty_scale": 0.03,
    "clipping_penalty": 0.05,
    "saturation_penalty": 0.02,
    "minimum_safe_speed_m_s": 170.0,
    "low_speed_penalty_scale": 0.05,
    "minimum_safe_altitude_m": 600.0,
    "low_altitude_penalty_scale": 0.10,
    "win_reward": 15.0,
    "loss_reward": -15.0,
    "crash_reward": -20.0,
}


def _official_damage_cone(sim_time_s: float) -> tuple[float, float]:
    if sim_time_s <= 100.0:
        return 1.0, 3000.0 * 0.3048
    if sim_time_s <= 150.0:
        return 2.0, 3500.0 * 0.3048
    return 3.0, 4000.0 * 0.3048


def compute_aim_residual_reward(
    ownship_state,
    target_state,
    ownship_damage: float,
    target_damage: float,
    reward_config: dict,
    terminated: bool,
    truncated: bool,
    end_condition: str,
    *,
    previous_geometry: dict[str, float] | None,
    action_info: dict | None,
    previous_correction,
) -> tuple[float, dict, dict[str, float], np.ndarray]:
    """Reward small gated corrections for stable aim and actual damage."""
    cfg = {
        **AIM_RESIDUAL_REWARD_DEFAULTS,
        **dict((reward_config or {}).get("aim_residual", {})),
    }
    geometry = aim_residual_geometry(ownship_state, target_state)
    action_info = dict(action_info or {})
    gate_active = bool(action_info.get("gate", {}).get("active", False))
    correction = np.asarray(
        action_info.get("applied_rl_correction", np.zeros(4)), dtype=np.float64
    )
    if correction.shape != (4,):
        correction = np.zeros(4, dtype=np.float64)
    prior_correction = np.asarray(previous_correction, dtype=np.float64)
    if prior_correction.shape != (4,):
        prior_correction = np.zeros(4, dtype=np.float64)

    components: dict[str, float] = {
        "step": float(cfg["step_penalty"]),
        "aim_progress": 0.0,
        "aim_quality": 0.0,
        "los_rate": 0.0,
        "cone_dwell": 0.0,
        "damage": float(cfg["damage_scale"])
        * (float(target_damage) - float(ownship_damage)),
        "residual_l2": 0.0,
        "residual_smooth": 0.0,
        "clipping": 0.0,
        "saturation": 0.0,
        "low_speed": 0.0,
        "safety": 0.0,
        "terminal": 0.0,
    }

    if gate_active:
        ata = geometry["ata_deg"]
        if previous_geometry is not None:
            progress = float(previous_geometry.get("ata_deg", ata)) - ata
            components["aim_progress"] = float(cfg["aim_progress_scale"]) * float(
                np.clip(progress, -5.0, 5.0)
            )
        components["aim_quality"] = float(cfg["aim_quality_scale"]) * max(
            0.0, 1.0 - ata / 15.0
        )
        los_rate_rms = float(
            np.hypot(
                geometry["los_azimuth_rate_deg_s"],
                geometry["los_elevation_rate_deg_s"],
            )
        )
        components["los_rate"] = -float(cfg["los_rate_penalty_scale"]) * min(
            los_rate_rms / 60.0, 1.0
        )
        phase_angle, phase_range = _official_damage_cone(
            float(ownship_state[StateIndex.SIM_TIME])
        )
        in_cone = (
            152.4 <= geometry["distance_m"] <= phase_range
            and ata <= phase_angle
        )
        components["cone_dwell"] = (
            float(cfg["cone_dwell_scale"]) if in_cone else 0.0
        )
        components["residual_l2"] = -float(
            cfg["residual_l2_penalty_scale"]
        ) * float(np.mean(np.square(correction[:3])))
        components["residual_smooth"] = -float(
            cfg["residual_smooth_penalty_scale"]
        ) * float(np.mean(np.square(correction[:3] - prior_correction[:3])))
        if action_info.get("action_clipped", False):
            components["clipping"] = -float(cfg["clipping_penalty"])
        if action_info.get("action_saturation", False):
            components["saturation"] = -float(cfg["saturation_penalty"])

    speed = float(ownship_state[StateIndex.KCAS])
    minimum_speed = float(cfg["minimum_safe_speed_m_s"])
    if speed < minimum_speed:
        components["low_speed"] = -float(cfg["low_speed_penalty_scale"]) * (
            minimum_speed - speed
        ) / max(minimum_speed, 1.0)
    altitude = float(ownship_state[StateIndex.ALT])
    minimum_altitude = float(cfg["minimum_safe_altitude_m"])
    if altitude < minimum_altitude:
        components["safety"] = -float(cfg["low_altitude_penalty_scale"]) * (
            minimum_altitude - altitude
        ) / max(minimum_altitude, 1.0)

    if terminated:
        ownship_health = float(ownship_state[StateIndex.HEALTH])
        target_health = float(target_state[StateIndex.HEALTH])
        if end_condition in ("ownship altitude below min", "FDM Update Fail"):
            components["terminal"] = float(cfg["crash_reward"])
        elif target_health <= 0.0 < ownship_health:
            components["terminal"] = float(cfg["win_reward"])
        elif ownship_health <= 0.0 < target_health:
            components["terminal"] = float(cfg["loss_reward"])

    total = float(sum(components.values()))
    return total, components, geometry, correction


def compute_reward(
    ownship_state,
    target_state,
    ownship_damage: float,
    target_damage: float,
    geo_info,
    wez_config: dict,
    reward_config: dict,
    terminated: bool,
    truncated: bool,
    end_condition: str,
) -> tuple[float, dict]:
    """Compute step reward and return (total, components) tuple.

    components keys: step, pursuit, damage, safety, terminal
    """
    reward_mode = reward_config.get("mode", "default")
    if reward_mode not in (None, "default"):
        raise ValueError(
            f"Release reward mode {reward_mode!r} is not supported. "
            "Research-only reward modes such as ref_old_1vs1 live in MyTrainEnv."
        )

    components: dict[str, float] = {}

    # ── 0. Survival bonus (curriculum Stage 0 only, defaults to 0) ────────
    r_survival = float(reward_config.get("survival_bonus", 0.0))
    components["survival"] = r_survival

    # ── 1. Step penalty (time efficiency) ─────────────────────────────────
    r_step = float(reward_config["step_penalty"])
    components["step"] = r_step

    # ── 2. Pursuit shaping: smooth ATA × range gradient ───────────────────
    #   Replaces the old binary wez_bonus with a continuous gradient that
    #   provides learning signal even before entering the narrow WEZ cone.
    distance = geo_info._get_distance(ownship_state, target_state)
    ata = abs(geo_info._get_antenna_train_angle(ownship_state, target_state, False))
    half_angle = float(reward_config["pursuit_half_angle_deg"])
    pursuit_range = float(reward_config["pursuit_range_m"])
    ata_factor = max(0.0, 1.0 - ata / half_angle)
    range_factor = max(0.0, 1.0 - distance / pursuit_range)
    r_pursuit = float(reward_config["pursuit_scale"]) * ata_factor * range_factor
    components["pursuit"] = r_pursuit

    # ── 3. Damage differential ─────────────────────────────────────────────
    #   Peaks inside the WEZ naturally — no separate wez_bonus needed.
    #   Scale reduced (200 → 20) so terminal rewards retain directional pull.
    r_damage = float(reward_config["damage_scale"]) * (target_damage - ownship_damage)
    components["damage"] = r_damage

    # ── 4. Safety: low altitude penalty ───────────────────────────────────
    r_safety = 0.0
    if float(ownship_state[StateIndex.ALT]) < 600.0:
        r_safety = -float(reward_config["low_altitude_penalty"])
    components["safety"] = r_safety

    # ── 5. Terminal reward ─────────────────────────────────────────────────
    #   timeout_health_scale removed — damage_scale already integrates health
    #   differential throughout the episode.
    r_terminal = 0.0
    if terminated:
        ownship_health = float(ownship_state[StateIndex.HEALTH])
        target_health = float(target_state[StateIndex.HEALTH])
        if end_condition == "two circle headon guard fail":
            r_terminal = float(reward_config.get("guard_fail_penalty", -50.0))
        elif target_health <= 0.0 < ownship_health:
            r_terminal = float(reward_config["win_reward"])
        elif ownship_health <= 0.0 < target_health:
            r_terminal = float(reward_config["loss_reward"])
        else:
            r_terminal = float(reward_config["draw_reward"])
    components["terminal"] = r_terminal

    total = r_survival + r_step + r_pursuit + r_damage + r_safety + r_terminal
    return float(total), components


def describe_reward(reward_config: dict, wez_config: dict) -> dict:
    return {
        "description": (
            "Survival bonus (curriculum) + step penalty + pursuit shaping (smooth ATA×range gradient) "
            "+ damage differential + low altitude penalty + terminal rewards."
        ),
        "survival_bonus": reward_config.get("survival_bonus", 0.0),
        "step_penalty": reward_config["step_penalty"],
        "damage_scale": reward_config["damage_scale"],
        "pursuit_scale": reward_config.get("pursuit_scale", 0.0),
        "pursuit_half_angle_deg": reward_config.get("pursuit_half_angle_deg", 30.0),
        "pursuit_range_m": reward_config.get("pursuit_range_m", 3000.0),
        "low_altitude_penalty": reward_config["low_altitude_penalty"],
        "win_reward": reward_config["win_reward"],
        "loss_reward": reward_config["loss_reward"],
        "draw_reward": reward_config["draw_reward"],
        "guard_fail_penalty": reward_config.get("guard_fail_penalty", -50.0),
        "wez": {
            "angle_deg": wez_config["angle_deg"],
            "min_range_m": wez_config["min_range_m"],
            "max_range_m": wez_config["max_range_m"],
        },
    }
