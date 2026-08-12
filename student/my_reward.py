# -*- coding: utf-8 -*-
"""학습 수렴을 위한 보상 함수 — 강한 경사(gradient) 버전.

설계 원칙:
  1. 에이전트가 랜덤 탐색만 해도 "표적 방향 = 좋다"를 즉시 느낄 수 있을 만큼
     강한 보상 경사를 만든다.
  2. 보상 레이어를 넓은 깔때기 → 좁은 깔때기 → WEZ 순으로 쌓아,
     어느 위치에서든 "다음에 뭘 해야 보상이 올라가는지" 명확하게 한다.

     [출발] ─────────────── 3000m ─── 1000m ──|WEZ ±1°|
     ▓▓▓▓▓ approach (거리) ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
     ▓▓▓▓▓ heading  (cos ATA, 전방향 경사) ▓▓▓▓▓▓▓▓▓▓▓
                    ▓▓ precision (15° 이내) ▓▓▓▓▓▓▓▓▓▓▓
                                      ████ wez_entry ██

  3. step_penalty를 최소화해서 약한 양의 신호를 묻지 않게 한다.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.sim.state_schema import StateIndex

FEET_TO_METER = 0.3048


MY_REWARD_CONFIG = {
    # ── 0. 시간 패널티 (최소화) ──
    "step_penalty": -0.002,

    # ── 1. 접근 보상 (거리 기반 포텐셜) ──
    "approach_scale": 0.6,         # 거리가 가까울수록 매 스텝 양의 보상
    "approach_max_dist": 5000.0,   # 이 거리 이상이면 접근 보상 0

    # ── 2. 기수 정렬 보상 (cos ATA 기반, 전방향 경사) ──
    "heading_scale": 0.5,          # ATA=0°: 0.5, ATA=90°: 0.25, ATA=180°: 0.0

    # ── 3. 후방 위치 보상 ──
    "tail_scale": 0.3,             # 적기 후방에 있을수록 보너스
    "tail_start_deg": 30.0,        # 이 각도 이상부터 보상 시작

    # ── 4. 정밀 조준 보상 (WEZ 진입 전 중간 깔때기) ──
    "precision_scale": 1.5,        # 가까이 + 정확히 조준할수록 강한 보상
    "precision_dist": 2500.0,      # 이 거리 이내에서만 활성화
    "precision_half_angle": 15.0,  # ATA가 이 각도 이내일 때만 활성화

    # ── 5. WEZ 진입 보너스 ──
    "wez_entry_bonus": 5.0,        # WEZ 안에 있을 때 매 스텝 고정 보상

    # ── 6. 데미지 ──
    "damage_scale": 50.0,

    # ── 7. 안전 규칙 ──
    "min_safe_alt_ft": 2000.0,
    "low_alt_penalty_scale": 0.5,
    "min_safe_speed_kcas": 180.0,
    "max_safe_speed_kcas": 550.0,
    "speed_penalty_scale": 0.3,
    "collision_dist_m": 100.0,
    "collision_penalty_scale": 1.0,

    # ── 8. 터미널 ──
    "win_reward": 200.0,
    "loss_reward": -100.0,
    "draw_reward": -50.0,
}


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
    """강한 경사(gradient)를 가진 보상 함수.

    모든 보상 컴포넌트는 매 스텝 독립적으로 계산되며,
    에피소드 내 상태(streak 등)에 의존하지 않는다.
    """
    cfg = {**MY_REWARD_CONFIG, **(reward_config or {})}
    components: dict[str, float] = {}

    # ── 기본 측정값 ──
    distance = geo_info._get_distance(ownship_state, target_state)
    ata = geo_info._get_antenna_train_angle(ownship_state, target_state, False)
    ata_abs = abs(float(ata))
    target_ata = abs(float(
        geo_info._get_antenna_train_angle(target_state, ownship_state, False)
    ))
    alt_m = float(ownship_state[StateIndex.ALT])
    speed_kcas = float(ownship_state[StateIndex.KCAS])

    wez_angle = float(wez_config.get("angle_deg", 2.0)) if wez_config else 2.0
    wez_max = float(wez_config.get("max_range_m", 914.4)) if wez_config else 914.4
    wez_min = float(wez_config.get("min_range_m", 152.4)) if wez_config else 152.4
    wez_half = wez_angle / 2.0

    # ── 0. Step penalty (최소) ──
    components["step"] = float(cfg.get("step_penalty", -0.002))

    # ── 1. 접근 보상: 거리가 가까울수록 양의 보상 ──
    #   distance=5000m → 0.0,  distance=2000m → +0.36,
    #   distance=500m  → +0.54
    max_approach = float(cfg.get("approach_max_dist", 5000.0))
    approach_scale = float(cfg.get("approach_scale", 0.6))
    components["approach"] = approach_scale * max(0.0, 1.0 - distance / max_approach)

    # ── 2. 기수 정렬: cos(ATA) 기반 — 모든 각도에서 경사 존재 ──
    #   ATA=0° → +0.50,  ATA=45° → +0.43,
    #   ATA=90° → +0.25, ATA=180° → 0.0
    heading_scale = float(cfg.get("heading_scale", 0.5))
    cos_ata = math.cos(math.radians(min(ata_abs, 180.0)))
    components["heading"] = heading_scale * (cos_ata + 1.0) / 2.0

    # ── 3. 후방 위치: 적기 뒤에 있으면 보너스 ──
    #   target_ata=180°(완전 후방) → +0.30
    #   target_ata=30° 미만(전방) → 0.0
    tail_scale = float(cfg.get("tail_scale", 0.3))
    tail_start = float(cfg.get("tail_start_deg", 30.0))
    if target_ata > tail_start:
        components["tail_position"] = tail_scale * (target_ata - tail_start) / (180.0 - tail_start)
    else:
        components["tail_position"] = 0.0

    # ── 4. 정밀 조준: 가까이 + ATA 작을 때 강한 보상 ──
    #   WEZ 직전 영역에서 정확한 조준으로 유도하는 중간 깔때기.
    #   distance=1000m, ATA=0° → +1.5 ×(1.0)×(0.8) = +1.20
    #   distance=2000m, ATA=5° → +1.5 ×(0.67)×(0.6) = +0.60
    #   ATA>15° 또는 distance>2500m → 0.0
    prec_dist = float(cfg.get("precision_dist", 2500.0))
    prec_half = float(cfg.get("precision_half_angle", 15.0))
    prec_scale = float(cfg.get("precision_scale", 1.5))
    if distance < prec_dist and ata_abs < prec_half:
        ata_f = 1.0 - ata_abs / prec_half
        dist_f = 1.0 - distance / prec_dist
        components["precision"] = prec_scale * ata_f * (0.5 + 0.5 * dist_f)
    else:
        components["precision"] = 0.0

    # ── 5. WEZ 진입 보너스 ──
    in_wez = (ata_abs <= wez_half) and (wez_min <= distance <= wez_max)
    components["wez_entry"] = float(cfg.get("wez_entry_bonus", 5.0)) if in_wez else 0.0

    # ── 6. 데미지 차이 보상 ──
    components["damage"] = float(cfg.get("damage_scale", 50.0)) * (
        float(target_damage) - float(ownship_damage)
    )

    # ── 7. 안전 ──
    # 고도
    min_safe_alt = float(cfg.get("min_safe_alt_ft", 2000.0)) * FEET_TO_METER
    if alt_m < min_safe_alt:
        deficit = (min_safe_alt - alt_m) / max(1.0, min_safe_alt)
        components["safety"] = -float(cfg.get("low_alt_penalty_scale", 0.5)) * deficit
    else:
        components["safety"] = 0.0

    # 속도
    min_kcas = float(cfg.get("min_safe_speed_kcas", 180.0))
    max_kcas = float(cfg.get("max_safe_speed_kcas", 550.0))
    if speed_kcas < min_kcas:
        deficit = (min_kcas - speed_kcas) / max(1.0, min_kcas)
        components["safety_speed"] = -float(cfg.get("speed_penalty_scale", 0.3)) * deficit
    elif speed_kcas > max_kcas:
        excess = (speed_kcas - max_kcas) / max(1.0, max_kcas)
        components["safety_speed"] = -float(cfg.get("speed_penalty_scale", 0.3)) * excess
    else:
        components["safety_speed"] = 0.0

    # 충돌 위험
    coll_dist = float(cfg.get("collision_dist_m", 100.0))
    if distance < coll_dist:
        deficit = (coll_dist - distance) / max(1.0, coll_dist)
        components["collision"] = -float(cfg.get("collision_penalty_scale", 1.0)) * deficit
    else:
        components["collision"] = 0.0

    # ── 8. 터미널 (승패) ──
    r_terminal = 0.0
    if terminated or truncated:
        own_h = float(ownship_state[StateIndex.HEALTH])
        tgt_h = float(target_state[StateIndex.HEALTH])
        if tgt_h <= 0.0 < own_h:
            r_terminal = float(cfg.get("win_reward", 200.0))
        elif own_h <= 0.0 < tgt_h:
            r_terminal = float(cfg.get("loss_reward", -100.0))
        else:
            r_terminal = float(cfg.get("draw_reward", -50.0))
    components["terminal"] = r_terminal

    return float(sum(components.values())), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
