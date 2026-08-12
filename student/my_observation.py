# -*- coding: utf-8 -*-
"""Minimal custom observation example.

This module is loaded with:

  python train_rllib.py --observation-mode custom --observation-module student.my_observation

Keep OBSERVATION_SIZE synchronized with the vector returned by build_observation().
"""
from __future__ import annotations

import math
import numpy as np

from dogfight.envs.observation import normalize
from dogfight.sim.state_schema import StateIndex


OBSERVATION_MODE = "student19"
OBSERVATION_SIZE = 19
OBSERVATION_LOW = -1.0
OBSERVATION_HIGH = 1.0


def _get_v_ned(attitude_deg, v_body):
    """
    Body 좌표계 속도(u, v, w)를 NED 좌표계 속도(VN, VE, VD)로 변환
    attitude_deg: [Roll, Pitch, Yaw] (단위: degree)
    v_body: [u, v, w] (단위: m/s)
    """
    D2R = math.pi / 180.0
    phi, theta, psi = attitude_deg * D2R
    
    # Body to NED rotation matrices components (T_nb 의 전치행렬 T_bn)
    # tx, ty, tz: Roll, Pitch, Yaw
    tx = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    ty = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    tz = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    
    # R_bn = Rz * Ry * Rx
    R_bn = np.matmul(tz, np.matmul(ty, tx))
    return np.matmul(R_bn, v_body)


def build_observation(ownship_state, target_state, geo_info, wez_config=None):
    """Return a custom 19-D observation vector as float32."""
    
    # 1. 원본 16차원 관련 값 계산
    delta = target_state[:3] - ownship_state[:3]
    distance = geo_info._get_distance(ownship_state, target_state)
    ata = geo_info._get_antenna_train_angle(ownship_state, target_state, False)
    aa = geo_info._get_aspect_angle(ownship_state, target_state, False)
    az, el = geo_info._get_los_angle(ownship_state, target_state)

    obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)

    # Ownship state (0-5)
    obs[0] = normalize(float(ownship_state[StateIndex.ROLL]),   -180.0, 180.0)
    obs[1] = normalize(float(ownship_state[StateIndex.PITCH]),   -90.0,  90.0)
    obs[2] = normalize(float(ownship_state[StateIndex.YAW]),       0.0, 360.0)
    obs[3] = normalize(float(ownship_state[StateIndex.KCAS]),      0.0, 600.0)
    obs[4] = normalize(float(ownship_state[StateIndex.ALT]),       0.0, 15000.0)
    obs[5] = normalize(float(ownship_state[StateIndex.HEALTH]),    0.0,  1.0)

    # Relative position (6-8)
    obs[6] = normalize(float(delta[0]), -15000.0, 15000.0)
    obs[7] = normalize(float(delta[1]), -15000.0, 15000.0)
    obs[8] = normalize(float(delta[2]),  -8000.0,  8000.0)

    # Geometry (9-12)
    obs[9]  = normalize(float(ata),  -180.0, 180.0)
    obs[10] = normalize(float(aa),   -180.0, 180.0)
    obs[11] = normalize(float(az),   -180.0, 180.0)
    obs[12] = normalize(float(el),    -90.0,  90.0)

    # Target health (13)
    obs[13] = normalize(float(target_state[StateIndex.HEALTH]), 0.0, 1.0)

    # WEZ flag (14)
    if wez_config is not None:
        ata_abs = abs(float(ata))
        in_wez = (
            wez_config["min_range_m"] <= distance <= wez_config["max_range_m"]
            and ata_abs <= wez_config["angle_deg"] / 2.0
        )
        obs[14] = 1.0 if in_wez else -1.0
    else:
        obs[14] = -1.0

    # Pursuit score (15)
    ata_factor   = max(0.0, 1.0 - abs(float(ata)) / 30.0)
    range_factor = max(0.0, 1.0 - distance / 3000.0)
    pursuit_raw  = ata_factor * range_factor
    obs[15] = 2.0 * pursuit_raw - 1.0

    # 2. 추가 3차원 (Rates) 계산
    own_att = ownship_state[StateIndex.ROLL : StateIndex.YAW + 1]
    own_v_body = ownship_state[6:9] # u, v, w (m/s)
    v_own_ned = _get_v_ned(own_att, own_v_body)

    tgt_att = target_state[StateIndex.ROLL : StateIndex.YAW + 1]
    tgt_v_body = target_state[6:9] # u, v, w (m/s)
    v_tgt_ned = _get_v_ned(tgt_att, tgt_v_body)

    # 상대 위치 벡터 P(NED) 및 상대 속도 벡터 V(NED)
    p_rel_ned = target_state[0:3] - ownship_state[0:3]
    v_rel_ned = v_tgt_ned - v_own_ned

    N, E, D = p_rel_ned
    N_dot, E_dot, D_dot = v_rel_ned

    # Range Rate (접근율, m/s) -> 음수면 접근중, 양수면 멀어짐
    if distance > 0.0:
        range_rate = np.dot(p_rel_ned, v_rel_ned) / distance
    else:
        range_rate = 0.0

    # LOS Azimuth Rate (deg/s)
    n_e_sq = N**2 + E**2
    if n_e_sq > 0.0:
        az_rate_rad = (N * E_dot - E * N_dot) / n_e_sq
    else:
        az_rate_rad = 0.0
    az_rate = math.degrees(az_rate_rad)

    # LOS Elevation Rate (deg/s)
    r_xy = math.sqrt(n_e_sq)
    if r_xy > 0.0:
        r_xy_dot = (N * N_dot + E * E_dot) / r_xy
        el_rate_rad = (r_xy * (-D_dot) - (-D) * r_xy_dot) / (r_xy**2 + D**2)
    else:
        el_rate_rad = 0.0
    el_rate = math.degrees(el_rate_rad)

    # Normalize new features
    obs[16] = normalize(float(range_rate), -1000.0, 1000.0)
    obs[17] = normalize(float(az_rate), -90.0, 90.0)
    obs[18] = normalize(float(el_rate), -90.0, 90.0)

    return obs


def describe_observation():
    return {
        "mode": OBSERVATION_MODE,
        "size": OBSERVATION_SIZE,
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
            "range_rate_norm",
            "los_az_rate_norm",
            "los_el_rate_norm",
        ],
        "description": (
            "Custom 19-D observation: tactical16 + 3 rate features "
            "(range rate, LOS az rate, LOS el rate) for mitigating vibration."
        ),
    }
