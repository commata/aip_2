from __future__ import annotations

from dogfight.research.tactical_events import (
    EventExtractionConfig,
    extract_decision_events,
    summarize_events,
)


def frame(index: int, *, azimuth: float, in_wez: bool = False, action: float = 0.0):
    return {
        "frame": index,
        "sim_time_s": index / 60.0,
        "distance_m": 900.0,
        "ata_deg": abs(azimuth),
        "aim_azimuth_deg": azimuth,
        "aim_elevation_deg": 0.0,
        "los_azimuth_rate_deg_s": azimuth,
        "los_elevation_rate_deg_s": 0.0,
        "closing_rate_m_s": 20.0,
        "target_damage": 0.001 if in_wez else 0.0,
        "ownship_damage": 0.0,
        "in_wez": in_wez,
        "ownship": {"speed_kcas": 220.0, "altitude_m": 4500.0},
        "ownship_action": [action, 0.0, 0.0, 1.0],
        "hybrid": {
            "bt_action": [action, 0.0, 0.0, 1.0],
            "bt_vp": [900.0 + index, 0.0, -4500.0],
        },
    }


def test_event_ids_are_stable_and_taxonomy_is_not_label() -> None:
    frames = [
        frame(0, azimuth=2.0),
        frame(1, azimuth=0.2),
        frame(2, azimuth=-1.0),
        frame(3, azimuth=-2.0),
    ]
    config = EventExtractionConfig(minimum_event_separation_frames=1)
    first = extract_decision_events(
        frames,
        fight_id="fight_a",
        scenario_id="crossing_left",
        opponent_id="autopilot",
        seed=1,
        config=config,
    )
    second = extract_decision_events(
        frames,
        fight_id="fight_a",
        scenario_id="crossing_left",
        opponent_id="autopilot",
        seed=1,
        config=config,
    )
    assert [row["event_id"] for row in first] == [row["event_id"] for row in second]
    assert any(row["diagnostic_failure_family"] == "A_AZIMUTH_OVERSHOOT" for row in first)
    assert all(row["diagnostic_is_training_label"] is False for row in first)
    assert all(row["primary_label"].startswith("PENDING_PREFIX") for row in first)


def test_adjacent_events_are_clustered_by_type() -> None:
    frames = [frame(index, azimuth=10.0) for index in range(30)]
    events = extract_decision_events(
        frames,
        fight_id="fight_b",
        scenario_id="lateral_left",
        opponent_id="autopilot",
        seed=2,
        config=EventExtractionConfig(minimum_event_separation_frames=12),
    )
    crossings = [row for row in events if row["event_type"] == "target_crossing"]
    assert len(crossings) <= 3


def test_summary_counts_unique_group_units() -> None:
    frames = [
        frame(0, azimuth=2.0),
        frame(1, azimuth=0.2, in_wez=True),
        frame(2, azimuth=-1.0),
        frame(3, azimuth=-2.0),
    ]
    events = extract_decision_events(
        frames,
        fight_id="fight_c",
        scenario_id="neutral",
        opponent_id="autopilot",
        seed=3,
        config=EventExtractionConfig(minimum_event_separation_frames=1),
    )
    summary = summarize_events(events)
    assert summary["unique_events"] == len(events)
    assert summary["fights"] == 1
    assert summary["diagnostic_taxonomy_is_label"] is False
