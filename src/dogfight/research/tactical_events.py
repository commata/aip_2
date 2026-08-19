from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

import numpy as np


FAILURE_TAXONOMY = {
    "A_AZIMUTH_OVERSHOOT": "signed azimuth error crossed zero and began growing",
    "B_ELEVATION_OVERSHOOT": "signed elevation error crossed zero and began growing",
    "C_LOS_RATE_EXCESSIVE": "combined LOS rate exceeded the frozen threshold",
    "D_CONE_APPROACH_MISS": "near official cone/range but outside damage cone",
    "E_CONE_EXIT": "left the official damage cone",
    "F_TARGET_CROSSING_LEAD": "target crossing with material transverse LOS rate",
    "G_TOO_FAST_CLOSING": "closing rate exceeded the frozen threshold",
    "H_RANGE_MAINTENANCE": "crossed an official range boundary outward",
    "I_BT_VP_SWITCH_LATE": "BT VP changed sharply after aim error was already growing",
    "J_PURE_BT_ALREADY_OPTIMAL": "sustained official cone occupancy",
    "K_LOW_AUTHORITY_SATURATION": "one or more BT surfaces saturated",
    "L_ENERGY_ALTITUDE_SAFETY": "speed or altitude approached the safety floor",
}


EVENT_TYPES = (
    "phase_cone_approach",
    "cone_entry",
    "cone_exit",
    "aim_error_local_minimum",
    "aim_error_growth_turn",
    "los_rate_sign_reversal",
    "closing_extreme",
    "range_boundary_crossing",
    "bt_vp_jump",
    "surface_saturation",
    "target_crossing",
    "first_damage_pre",
    "damage_window_post",
    "miss_recovery",
)


@dataclass(frozen=True)
class EventExtractionConfig:
    sim_hz: int = 60
    minimum_event_separation_frames: int = 12
    los_rate_excessive_deg_s: float = 8.0
    closing_extreme_m_s: float = 180.0
    bt_vp_jump_m: float = 100.0
    safety_speed_m_s: float = 180.0
    safety_altitude_m: float = 1200.0

    def validate(self) -> None:
        if self.sim_hz != 60 or self.minimum_event_separation_frames <= 0:
            raise ValueError("event extraction requires 60Hz and positive separation")


def _sign_reversal(left: float, right: float, *, floor: float = 1e-6) -> bool:
    return abs(left) > floor and abs(right) > floor and np.sign(left) != np.sign(right)


def _phase_limits(sim_time_s: float) -> tuple[float, float]:
    if sim_time_s <= 100.0:
        return 1.0, 914.4
    if sim_time_s <= 150.0:
        return 2.0, 1066.8
    return 3.0, 1219.2


def _bt_vp(frame: dict[str, Any]) -> np.ndarray:
    value = frame.get("hybrid", {}).get("bt_vp", [0.0, 0.0, 0.0])
    vector = np.asarray(value, dtype=np.float64)
    return vector if vector.shape == (3,) and np.all(np.isfinite(vector)) else np.zeros(3)


def _event_candidates(
    frames: list[dict[str, Any]], config: EventExtractionConfig
) -> Iterable[tuple[int, str, str]]:
    first_damage_frame = next(
        (index for index, row in enumerate(frames) if float(row["target_damage"]) > 0.0),
        None,
    )
    for index in range(1, len(frames) - 1):
        previous, current, following = frames[index - 1 : index + 2]
        sim_time = float(current["sim_time_s"])
        cone_angle, cone_range = _phase_limits(sim_time)
        aim = float(
            np.hypot(current["aim_azimuth_deg"], current["aim_elevation_deg"])
        )
        previous_aim = float(
            np.hypot(previous["aim_azimuth_deg"], previous["aim_elevation_deg"])
        )
        following_aim = float(
            np.hypot(following["aim_azimuth_deg"], following["aim_elevation_deg"])
        )
        los_rate = float(
            np.hypot(
                current["los_azimuth_rate_deg_s"],
                current["los_elevation_rate_deg_s"],
            )
        )
        if (
            float(current["distance_m"]) <= cone_range + 150.0
            and aim <= cone_angle + 2.0
            and not bool(current["in_wez"])
        ):
            yield index, "phase_cone_approach", "D_CONE_APPROACH_MISS"
        if not bool(previous["in_wez"]) and bool(current["in_wez"]):
            yield index, "cone_entry", "J_PURE_BT_ALREADY_OPTIMAL"
        if bool(previous["in_wez"]) and not bool(current["in_wez"]):
            yield index, "cone_exit", "E_CONE_EXIT"
        if aim <= previous_aim and aim < following_aim:
            family = (
                "J_PURE_BT_ALREADY_OPTIMAL" if bool(current["in_wez"])
                else "D_CONE_APPROACH_MISS"
            )
            yield index, "aim_error_local_minimum", family
        if previous_aim > aim and following_aim > aim + 0.02:
            yield index, "aim_error_growth_turn", "D_CONE_APPROACH_MISS"
        if _sign_reversal(
            float(previous["los_azimuth_rate_deg_s"]),
            float(current["los_azimuth_rate_deg_s"]),
        ) or _sign_reversal(
            float(previous["los_elevation_rate_deg_s"]),
            float(current["los_elevation_rate_deg_s"]),
        ):
            yield index, "los_rate_sign_reversal", "C_LOS_RATE_EXCESSIVE"
        if los_rate >= config.los_rate_excessive_deg_s:
            yield index, "target_crossing", "F_TARGET_CROSSING_LEAD"
        closing = float(current["closing_rate_m_s"])
        if abs(closing) >= config.closing_extreme_m_s:
            yield index, "closing_extreme", "G_TOO_FAST_CLOSING"
        for boundary in (914.4, 1066.8, 1219.2):
            if float(previous["distance_m"]) <= boundary < float(current["distance_m"]):
                yield index, "range_boundary_crossing", "H_RANGE_MAINTENANCE"
                break
        if float(np.linalg.norm(_bt_vp(current) - _bt_vp(previous))) >= config.bt_vp_jump_m:
            yield index, "bt_vp_jump", "I_BT_VP_SWITCH_LATE"
        action = np.asarray(
            current.get("hybrid", {}).get("bt_action", current["ownship_action"]),
            dtype=np.float64,
        )
        if np.any(np.isclose(np.abs(action[:3]), 1.0, atol=1e-6)):
            yield index, "surface_saturation", "K_LOW_AUTHORITY_SATURATION"
        if _sign_reversal(
            float(previous["aim_azimuth_deg"]), float(current["aim_azimuth_deg"])
        ):
            yield index, "target_crossing", "A_AZIMUTH_OVERSHOOT"
        if _sign_reversal(
            float(previous["aim_elevation_deg"]),
            float(current["aim_elevation_deg"]),
        ):
            yield index, "target_crossing", "B_ELEVATION_OVERSHOOT"
        if (
            float(current["ownship"]["speed_kcas"]) <= config.safety_speed_m_s
            or float(current["ownship"]["altitude_m"]) <= config.safety_altitude_m
        ):
            yield index, "miss_recovery", "L_ENERGY_ALTITUDE_SAFETY"
        if first_damage_frame is not None and index == max(1, first_damage_frame - 1):
            yield index, "first_damage_pre", "J_PURE_BT_ALREADY_OPTIMAL"
        if float(previous["target_damage"]) > 0.0 and float(current["target_damage"]) == 0.0:
            yield index, "damage_window_post", "E_CONE_EXIT"


def extract_decision_events(
    frames: list[dict[str, Any]],
    *,
    fight_id: str,
    scenario_id: str,
    opponent_id: str,
    seed: int,
    config: EventExtractionConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = config or EventExtractionConfig()
    cfg.validate()
    if not frames:
        return []
    last_by_type: dict[str, int] = {}
    events = []
    seen_frames: set[tuple[int, str]] = set()
    for frame_index, event_type, family in _event_candidates(frames, cfg):
        if event_type not in EVENT_TYPES or family not in FAILURE_TAXONOMY:
            raise ValueError("unknown event type or diagnostic family")
        key = (frame_index, event_type)
        if key in seen_frames:
            continue
        seen_frames.add(key)
        previous_frame = last_by_type.get(event_type, -10**9)
        if frame_index - previous_frame < cfg.minimum_event_separation_frames:
            continue
        last_by_type[event_type] = frame_index
        row = frames[frame_index]
        stable_key = f"{fight_id}|{frame_index}|{event_type}|{family}"
        event_id = "evt_" + hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:16]
        events.append(
            {
                "event_id": event_id,
                "fight_id": fight_id,
                "trajectory_id": fight_id,
                "scenario_id": scenario_id,
                "opponent_id": opponent_id,
                "seed": int(seed),
                "frame": int(frame_index),
                "sim_time_s": float(row["sim_time_s"]),
                "event_type": event_type,
                "diagnostic_failure_family": family,
                "diagnostic_is_training_label": False,
                "primary_label": "PENDING_PREFIX_REPLAY_PAIRED_DAMAGE",
                "geometry": {
                    "distance_m": float(row["distance_m"]),
                    "aim_azimuth_deg": float(row["aim_azimuth_deg"]),
                    "aim_elevation_deg": float(row["aim_elevation_deg"]),
                    "los_azimuth_rate_deg_s": float(row["los_azimuth_rate_deg_s"]),
                    "los_elevation_rate_deg_s": float(row["los_elevation_rate_deg_s"]),
                    "closing_rate_m_s": float(row["closing_rate_m_s"]),
                    "in_wez": bool(row["in_wez"]),
                },
                "bt_action": list(
                    row.get("hybrid", {}).get("bt_action", row["ownship_action"])
                ),
                "bt_vp": _bt_vp(row).tolist(),
            }
        )
    return events


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    fights = {row["fight_id"] for row in events}
    scenarios = {row["scenario_id"] for row in events}
    opponents = {row["opponent_id"] for row in events}
    by_type = {event_type: 0 for event_type in EVENT_TYPES}
    by_family = {family: 0 for family in FAILURE_TAXONOMY}
    for row in events:
        by_type[row["event_type"]] += 1
        by_family[row["diagnostic_failure_family"]] += 1
    return {
        "unique_events": len({row["event_id"] for row in events}),
        "fights": len(fights),
        "scenarios": len(scenarios),
        "opponents": len(opponents),
        "by_event_type": by_type,
        "by_diagnostic_family": by_family,
        "diagnostic_taxonomy_is_label": False,
        "primary_label": "paired Damage / official outcome from prefix replay",
    }
