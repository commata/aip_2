from __future__ import annotations

from automation.evaluate_tactical_oracle_v4 import (
    candidate_options,
    select_balanced_events,
    summarize_oracle,
)


def event(index: int, geometry: str) -> dict:
    return {
        "event_id": f"e{index}",
        "fight_id": f"fight_{geometry}_s{index}",
        "scenario_id": geometry,
        "opponent_id": "autopilot",
        "seed": index,
        "frame": index,
        "event_type": "cone_exit" if index % 2 else "target_crossing",
        "diagnostic_failure_family": "E_CONE_EXIT",
    }


def record(event_id: str, option: str, delta: float, geometry: str) -> dict:
    mode, duration = option.rsplit("__d", 1)
    return {
        "event_id": event_id,
        "scenario_id": geometry,
        "diagnostic_failure_family": "E_CONE_EXIT",
        "option_id": option,
        "mode": mode,
        "hold_frames": int(duration),
        "clean": True,
        "terminal": {
            "damage_dealt_delta": delta,
            "net_health_margin_delta": delta,
        },
    }


def test_candidate_grid_freezes_modes_and_durations() -> None:
    options = candidate_options("T1")
    assert len(options) == 9
    assert {row["hold_frames"] for row in options} == {30, 60, 120}
    assert all(row["mode"] != "BT_DEFAULT" for row in options)


def test_balanced_selection_uses_multiple_geometry_groups() -> None:
    events = [event(index, "left" if index % 2 else "right") for index in range(20)]
    selected = select_balanced_events(events, 8)
    assert len(selected) == 8
    assert {row["scenario_id"] for row in selected} == {"left", "right"}


def test_oracle_abstains_on_nonpositive_events() -> None:
    rows = [
        record("e1", "PURE_PURSUIT__d30", 0.002, "left"),
        record("e1", "LEAD_PURSUIT_T060__d30", -0.001, "left"),
        record("e2", "PURE_PURSUIT__d30", -0.002, "right"),
        record("e2", "LEAD_PURSUIT_T060__d30", 0.0, "right"),
    ]
    summary = summarize_oracle(rows)
    assert summary["events_with_clean_options"] == 2
    assert summary["oracle_nondefault_coverage"] == 0.5
    selected = {row["event_id"]: row["selected_option"] for row in summary["oracle"]}
    assert selected["e1"] == "PURE_PURSUIT__d30"
    assert selected["e2"] == "BT_DEFAULT"
