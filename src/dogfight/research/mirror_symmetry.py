from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dogfight.sim.state_schema import StateIndex


@dataclass(frozen=True)
class MirrorContract:
    """한 mirror 변환에서 동일/부호 반전되어야 하는 관측과 제어 축."""

    geometry_signs: dict[str, float]
    action_signs: tuple[float, float, float, float]


LATERAL_CONTRACT = MirrorContract(
    geometry_signs={
        "aim_azimuth_deg": -1.0,
        "aim_elevation_deg": 1.0,
        "los_azimuth_rate_deg_s": -1.0,
        "los_elevation_rate_deg_s": 1.0,
        "ata_deg": 1.0,
        "target_ata_deg": 1.0,
        "distance_m": 1.0,
        "closing_rate_m_s": 1.0,
    },
    action_signs=(-1.0, 1.0, -1.0, 1.0),
)

VERTICAL_CONTRACT = MirrorContract(
    geometry_signs={
        "aim_azimuth_deg": 1.0,
        "aim_elevation_deg": -1.0,
        "los_azimuth_rate_deg_s": 1.0,
        "los_elevation_rate_deg_s": -1.0,
        "ata_deg": 1.0,
        "target_ata_deg": 1.0,
        "distance_m": 1.0,
        "closing_rate_m_s": 1.0,
    },
    # 반사된 right-handed body frame에서 axial control 축은 roll/pitch가 반전된다.
    action_signs=(-1.0, -1.0, 1.0, 1.0),
)


def _copy_state(state) -> np.ndarray:
    value = np.asarray(state, dtype=np.float64).copy()
    if value.ndim != 1 or value.size < 9:
        raise ValueError("mirror state는 body velocity를 포함한 1차원 state여야 함")
    return value


def _normalize_heading(value: float) -> float:
    return float(value % 360.0)


def mirror_state_lateral(state, *, east_origin_m: float = 0.0) -> np.ndarray:
    """N-D 평면을 기준으로 world/body 좌표를 함께 반사한다."""
    mirrored = _copy_state(state)
    mirrored[StateIndex.E] = 2.0 * east_origin_m - mirrored[StateIndex.E]
    mirrored[StateIndex.ROLL] *= -1.0
    mirrored[StateIndex.YAW] = _normalize_heading(-mirrored[StateIndex.YAW])
    mirrored[7] *= -1.0  # body-axis lateral velocity
    return mirrored


def mirror_state_vertical(state, *, down_origin_m: float) -> np.ndarray:
    """지정 NED down 평면에서 기하를 반사한다.

    이는 중력까지 반전하는 동역학 대칭 주장이 아니라, 순간 조준/LOS 계산의
    상하 sign convention을 검증하기 위한 정확한 운동학 변환이다.
    """
    mirrored = _copy_state(state)
    mirrored[StateIndex.D] = 2.0 * down_origin_m - mirrored[StateIndex.D]
    mirrored[StateIndex.ROLL] *= -1.0
    mirrored[StateIndex.PITCH] *= -1.0
    mirrored[8] *= -1.0  # body-axis vertical velocity
    if mirrored.size > StateIndex.ALT:
        mirrored[StateIndex.ALT] = -mirrored[StateIndex.D]
    return mirrored


def mirror_action(action, contract: MirrorContract) -> np.ndarray:
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (4,):
        raise ValueError(f"mirror action은 4축이어야 함: {value.shape}")
    return value * np.asarray(contract.action_signs, dtype=np.float64)


def mirror_pose_lateral(pose) -> list[float]:
    """[N,E,D,roll,pitch,heading,speed] 초기 자세를 좌우 반사한다."""
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (7,):
        raise ValueError("초기 pose는 7개 값이어야 함")
    result = value.copy()
    result[1] *= -1.0
    result[3] *= -1.0
    result[5] = _normalize_heading(-result[5])
    return result.tolist()


def mirror_pose_vertical(pose, *, down_origin_m: float) -> list[float]:
    """[N,E,D,roll,pitch,heading,speed] 초기 자세를 상하 반사한다."""
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (7,):
        raise ValueError("초기 pose는 7개 값이어야 함")
    result = value.copy()
    result[2] = 2.0 * down_origin_m - result[2]
    result[3] *= -1.0
    result[4] *= -1.0
    return result.tolist()
