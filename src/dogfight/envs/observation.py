from __future__ import annotations

import numpy as np

from dogfight.sim.state_schema import StateIndex


def normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    clipped = float(np.clip(value, minimum, maximum))
    midpoint = (maximum + minimum) / 2.0
    half_range = (maximum - minimum) / 2.0
    return (clipped - midpoint) / half_range


def observation_size(mode: str) -> int:
    if mode in ("aim_residual10", "aim_residual10_v2"):
        return 10
    if mode == "aim_residual13_btaware":
        return 13
    if mode == "tactical16":
        return 16
    if mode == "relative14":
        return 14
    return 12  # classic12


def build_observation(
    mode: str,
    ownship_state,
    target_state,
    geo_info,
    wez_config=None,
    *,
    bt_action=None,
) -> np.ndarray:
    if mode == "aim_residual10":
        return _build_aim_residual10(ownship_state, target_state)
    if mode == "aim_residual10_v2":
        return _build_aim_residual10_v2(ownship_state, target_state)
    if mode == "aim_residual13_btaware":
        return _build_aim_residual13_btaware(
            ownship_state,
            target_state,
            bt_action,
        )
    if mode == "tactical16":
        return _build_tactical16(ownship_state, target_state, geo_info, wez_config)
    if mode == "relative14":
        return _build_relative14(ownship_state, target_state, geo_info)
    return _build_classic12(ownship_state, target_state)


def describe_observation(mode: str) -> dict:
    if mode == "aim_residual10":
        return {
            "mode": "aim_residual10",
            "size": 10,
            "features": [
                "aim_azimuth_error_norm",
                "aim_elevation_error_norm",
                "los_azimuth_rate_norm",
                "los_elevation_rate_norm",
                "ata_norm",
                "target_ata_norm",
                "distance_norm",
                "closing_rate_norm",
                "ownship_speed_norm",
                "ownship_altitude_norm",
            ],
            "description": (
                "조준 잔차 학습용 10차원 관측: 기체축 조준 오차, 관성 LOS rate, "
                "공격 기하, 거리/접근률, 속도와 고도. 모두 [-1, 1]로 정규화한다."
            ),
        }
    if mode == "aim_residual10_v2":
        return {
            "mode": "aim_residual10_v2",
            "size": 10,
            "features": describe_observation("aim_residual10")["features"],
            "description": (
                "조준 잔차 학습용 10차원 관측 v2: feature는 v1과 같고 "
                "Gate 활성 로그의 근접 조준 범위에 맞춰 정규화한다."
            ),
            "normalization": {
                "aim_azimuth_deg": [-15.0, 15.0],
                "aim_elevation_deg": [-15.0, 15.0],
                "los_azimuth_rate_deg_s": [-15.0, 15.0],
                "los_elevation_rate_deg_s": [-15.0, 15.0],
                "ata_deg": [0.0, 15.0],
                "target_ata_deg": [90.0, 180.0],
                "distance_m": [0.0, 2000.0],
                "closing_rate_m_s": [-150.0, 150.0],
                "ownship_speed_m_s": [100.0, 400.0],
                "ownship_altitude_m": [0.0, 10000.0],
            },
        }
    if mode == "aim_residual13_btaware":
        return {
            "mode": "aim_residual13_btaware",
            "size": 13,
            "features": [
                *describe_observation("aim_residual10_v2")["features"],
                "bt_roll_command",
                "bt_pitch_command",
                "bt_yaw_command",
            ],
            "description": (
                "조준 잔차 학습용 13차원 BT-aware 관측: 10D v2를 그대로 "
                "보존하고 같은 simulator frame에 실제 residual과 결합할 "
                "BT roll/pitch/yaw 명령을 추가한다."
            ),
            "normalization": {
                **describe_observation("aim_residual10_v2")["normalization"],
                "bt_surface_commands": [-1.0, 1.0],
            },
        }
    if mode == "tactical16":
        return {
            "mode": "tactical16",
            "size": 16,
            "features": [
                "ownship_roll_norm",
                "ownship_pitch_norm",
                "ownship_yaw_norm",
                "ownship_speed_norm",
                "ownship_alt_norm",
                "ownship_health_norm",
                "delta_n_norm",
                "delta_e_norm",
                "delta_d_norm",
                "ata_norm",
                "aa_norm",
                "az_norm",
                "el_norm",
                "target_health_norm",
                "in_wez",
                "pursuit_score_norm",
            ],
            "description": (
                "Full tactical observation: ownship attitude + speed + altitude + health, "
                "relative geometry (ATA, AA, LOS), target health, WEZ flag, pursuit score. "
                "All features normalized to [-1, 1]. Observation space bounds: [-1, 1]."
            ),
        }
    if mode == "relative14":
        return {
            "mode": "relative14",
            "size": 14,
            "features": [
                "delta_n",
                "delta_e",
                "delta_d",
                "ownship_roll_norm",
                "ownship_pitch_norm",
                "ownship_yaw_norm",
                "target_roll_norm",
                "target_pitch_norm",
                "target_yaw_norm",
                "distance_norm",
                "ata_norm",
                "aa_norm",
                "az_norm",
                "el_norm",
            ],
            "description": "Relative geometry observation with normalized attitude and LOS terms.",
        }
    return {
        "mode": "classic12",
        "size": 12,
        "features": [
            "ownship_n",
            "ownship_e",
            "ownship_d",
            "target_n",
            "target_e",
            "target_d",
            "ownship_roll_norm",
            "ownship_pitch_norm",
            "ownship_yaw_norm",
            "target_roll_norm",
            "target_pitch_norm",
            "target_yaw_norm",
        ],
        "description": "Basic position and normalized attitude observation.",
    }


def body_to_ned_rotation(attitude_deg) -> np.ndarray:
    """Return the body-to-NED direction-cosine matrix for roll/pitch/yaw."""
    roll, pitch, yaw = np.radians(np.asarray(attitude_deg, dtype=np.float64))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def aim_residual_geometry(ownship_state, target_state) -> dict[str, float]:
    """Compute stateless aim geometry with velocities interpreted in body axes."""
    own = np.asarray(ownship_state, dtype=np.float64)
    target = np.asarray(target_state, dtype=np.float64)
    relative_ned = target[:3] - own[:3]
    distance = float(np.linalg.norm(relative_ned))
    if distance <= 1e-6:
        return {
            "aim_azimuth_deg": 0.0,
            "aim_elevation_deg": 0.0,
            "los_azimuth_rate_deg_s": 0.0,
            "los_elevation_rate_deg_s": 0.0,
            "ata_deg": 0.0,
            "target_ata_deg": 0.0,
            "distance_m": 0.0,
            "closing_rate_m_s": 0.0,
        }

    own_rotation = body_to_ned_rotation(own[3:6])
    target_rotation = body_to_ned_rotation(target[3:6])
    relative_body = own_rotation.T @ relative_ned
    horizontal_body = float(np.hypot(relative_body[0], relative_body[1]))
    aim_azimuth = float(np.degrees(np.arctan2(relative_body[1], relative_body[0])))
    aim_elevation = float(
        np.degrees(np.arctan2(-relative_body[2], max(horizontal_body, 1e-9)))
    )
    ata = float(
        np.degrees(
            np.arccos(np.clip(relative_body[0] / distance, -1.0, 1.0))
        )
    )
    target_relative_body = target_rotation.T @ -relative_ned
    target_ata = float(
        np.degrees(
            np.arccos(np.clip(target_relative_body[0] / distance, -1.0, 1.0))
        )
    )

    own_velocity_ned = own_rotation @ own[6:9]
    target_velocity_ned = target_rotation @ target[6:9]
    relative_velocity = target_velocity_ned - own_velocity_ned
    unit_line = relative_ned / distance
    closing_rate = -float(np.dot(unit_line, relative_velocity))

    north, east, down = relative_ned
    vn, ve, vd = relative_velocity
    horizontal_ned = float(np.hypot(north, east))
    horizontal_sq = max(horizontal_ned * horizontal_ned, 1e-9)
    los_az_rate = float(
        np.degrees((north * ve - east * vn) / horizontal_sq)
    )
    horizontal_rate = float((north * vn + east * ve) / max(horizontal_ned, 1e-9))
    los_el_rate = float(
        np.degrees((-vd * horizontal_ned + down * horizontal_rate) / (distance * distance))
    )
    return {
        "aim_azimuth_deg": aim_azimuth,
        "aim_elevation_deg": aim_elevation,
        "los_azimuth_rate_deg_s": los_az_rate,
        "los_elevation_rate_deg_s": los_el_rate,
        "ata_deg": ata,
        "target_ata_deg": target_ata,
        "distance_m": distance,
        "closing_rate_m_s": closing_rate,
    }


def _build_aim_residual10(ownship_state, target_state) -> np.ndarray:
    geometry = aim_residual_geometry(ownship_state, target_state)
    observation = np.array(
        [
            normalize(geometry["aim_azimuth_deg"], -180.0, 180.0),
            normalize(geometry["aim_elevation_deg"], -90.0, 90.0),
            normalize(geometry["los_azimuth_rate_deg_s"], -60.0, 60.0),
            normalize(geometry["los_elevation_rate_deg_s"], -60.0, 60.0),
            normalize(geometry["ata_deg"], 0.0, 180.0),
            normalize(geometry["target_ata_deg"], 0.0, 180.0),
            normalize(geometry["distance_m"], 0.0, 5000.0),
            normalize(geometry["closing_rate_m_s"], -500.0, 500.0),
            normalize(float(ownship_state[StateIndex.KCAS]), 0.0, 400.0),
            normalize(float(ownship_state[StateIndex.ALT]), 0.0, 15000.0),
        ],
        dtype=np.float32,
    )
    return np.clip(observation, -1.0, 1.0)


def _build_aim_residual10_v2(ownship_state, target_state) -> np.ndarray:
    """Scale the same ten features for the local pre-aim operating region."""
    geometry = aim_residual_geometry(ownship_state, target_state)
    observation = np.array(
        [
            normalize(geometry["aim_azimuth_deg"], -15.0, 15.0),
            normalize(geometry["aim_elevation_deg"], -15.0, 15.0),
            normalize(geometry["los_azimuth_rate_deg_s"], -15.0, 15.0),
            normalize(geometry["los_elevation_rate_deg_s"], -15.0, 15.0),
            normalize(geometry["ata_deg"], 0.0, 15.0),
            normalize(geometry["target_ata_deg"], 90.0, 180.0),
            normalize(geometry["distance_m"], 0.0, 2000.0),
            normalize(geometry["closing_rate_m_s"], -150.0, 150.0),
            normalize(float(ownship_state[StateIndex.KCAS]), 100.0, 400.0),
            normalize(float(ownship_state[StateIndex.ALT]), 0.0, 10000.0),
        ],
        dtype=np.float32,
    )
    return np.clip(observation, -1.0, 1.0)


def _build_aim_residual13_btaware(
    ownship_state,
    target_state,
    bt_action,
) -> np.ndarray:
    if bt_action is None:
        raise ValueError(
            "aim_residual13_btaware requires the same-frame bt_action"
        )
    bt = np.asarray(bt_action, dtype=np.float32)
    if bt.shape != (4,):
        raise ValueError(f"bt_action must have shape (4,), got {bt.shape}")
    observation = np.concatenate(
        (
            _build_aim_residual10_v2(ownship_state, target_state),
            np.clip(bt[:3], -1.0, 1.0),
        )
    ).astype(np.float32)
    return np.clip(observation, -1.0, 1.0)


def _build_classic12(ownship_state, target_state) -> np.ndarray:
    observation = np.zeros(12, dtype=np.float32)
    observation[0] = ownship_state[StateIndex.N]
    observation[1] = ownship_state[StateIndex.E]
    observation[2] = ownship_state[StateIndex.D]
    observation[3] = target_state[StateIndex.N]
    observation[4] = target_state[StateIndex.E]
    observation[5] = target_state[StateIndex.D]
    observation[6] = normalize(ownship_state[StateIndex.ROLL], -180.0, 180.0)
    observation[7] = normalize(ownship_state[StateIndex.PITCH], -90.0, 90.0)
    observation[8] = normalize(ownship_state[StateIndex.YAW], 0.0, 360.0)
    observation[9] = normalize(target_state[StateIndex.ROLL], -180.0, 180.0)
    observation[10] = normalize(target_state[StateIndex.PITCH], -90.0, 90.0)
    observation[11] = normalize(target_state[StateIndex.YAW], 0.0, 360.0)
    return observation


def _build_relative14(ownship_state, target_state, geo_info) -> np.ndarray:
    observation = np.zeros(14, dtype=np.float32)
    delta = target_state[:3] - ownship_state[:3]
    distance = geo_info._get_distance(ownship_state, target_state)
    ata = geo_info._get_antenna_train_angle(ownship_state, target_state, False)
    aa = geo_info._get_aspect_angle(ownship_state, target_state, False)
    az, el = geo_info._get_los_angle(ownship_state, target_state)

    observation[0] = normalize(delta[0], -10000.0, 10000.0)
    observation[1] = normalize(delta[1], -10000.0, 10000.0)
    observation[2] = normalize(delta[2], -5000.0, 5000.0)
    observation[3] = normalize(ownship_state[StateIndex.ROLL], -180.0, 180.0)
    observation[4] = normalize(ownship_state[StateIndex.PITCH], -90.0, 90.0)
    observation[5] = normalize(ownship_state[StateIndex.YAW], 0.0, 360.0)
    observation[6] = normalize(target_state[StateIndex.ROLL], -180.0, 180.0)
    observation[7] = normalize(target_state[StateIndex.PITCH], -90.0, 90.0)
    observation[8] = normalize(target_state[StateIndex.YAW], 0.0, 360.0)
    observation[9] = normalize(distance, 0.0, 20000.0)
    observation[10] = normalize(ata, -180.0, 180.0)
    observation[11] = normalize(aa, -180.0, 180.0)
    observation[12] = normalize(az, -180.0, 180.0)
    observation[13] = normalize(el, -90.0, 90.0)
    return observation


def _build_tactical16(ownship_state, target_state, geo_info, wez_config=None) -> np.ndarray:
    """16-feature tactical observation.

    Index map:
      0-5   ownship: roll, pitch, yaw, speed(KCAS), altitude, health
      6-8   relative position: delta_n, delta_e, delta_d
      9-12  geometry: ATA, AA, LOS_az, LOS_el
      13    target health
      14    in_wez flag  (-1 / +1)
      15    pursuit score (smooth ATA×range gradient, normalized to [-1,1])
    """
    obs = np.zeros(16, dtype=np.float32)

    delta = target_state[:3] - ownship_state[:3]
    distance = geo_info._get_distance(ownship_state, target_state)
    ata = geo_info._get_antenna_train_angle(ownship_state, target_state, False)
    aa = geo_info._get_aspect_angle(ownship_state, target_state, False)
    az, el = geo_info._get_los_angle(ownship_state, target_state)

    # Ownship state
    obs[0] = normalize(float(ownship_state[StateIndex.ROLL]),   -180.0, 180.0)
    obs[1] = normalize(float(ownship_state[StateIndex.PITCH]),   -90.0,  90.0)
    obs[2] = normalize(float(ownship_state[StateIndex.YAW]),       0.0, 360.0)
    obs[3] = normalize(float(ownship_state[StateIndex.KCAS]),      0.0, 600.0)
    obs[4] = normalize(float(ownship_state[StateIndex.ALT]),       0.0, 15000.0)
    obs[5] = normalize(float(ownship_state[StateIndex.HEALTH]),    0.0,  1.0)

    # Relative position
    obs[6] = normalize(float(delta[0]), -15000.0, 15000.0)
    obs[7] = normalize(float(delta[1]), -15000.0, 15000.0)
    obs[8] = normalize(float(delta[2]),  -8000.0,  8000.0)

    # Geometry
    obs[9]  = normalize(float(ata),  -180.0, 180.0)
    obs[10] = normalize(float(aa),   -180.0, 180.0)
    obs[11] = normalize(float(az),   -180.0, 180.0)
    obs[12] = normalize(float(el),    -90.0,  90.0)

    # Target health
    obs[13] = normalize(float(target_state[StateIndex.HEALTH]), 0.0, 1.0)

    # WEZ flag: +1 if ownship is inside weapon engagement zone, -1 otherwise
    if wez_config is not None:
        ata_abs = abs(float(ata))
        in_wez = (
            wez_config["min_range_m"] <= distance <= wez_config["max_range_m"]
            and ata_abs <= wez_config["angle_deg"] / 2.0
        )
        obs[14] = 1.0 if in_wez else -1.0
    else:
        obs[14] = -1.0

    # Pursuit score: smooth ATA×range gradient in [-1, 1]
    # Thresholds are observation-level constants (not tied to reward config)
    ata_factor   = max(0.0, 1.0 - abs(float(ata)) / 30.0)   # full score at ATA=0°
    range_factor = max(0.0, 1.0 - distance / 3000.0)          # full score at distance=0
    pursuit_raw  = ata_factor * range_factor                   # [0, 1]
    obs[15] = 2.0 * pursuit_raw - 1.0                         # → [-1, 1]

    return obs
