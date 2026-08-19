from __future__ import annotations

from automation.acquire_shadow_events_v4 import select_active_events


def _row(event: str, fight: str, frame: int, reason: str, priority: int, std: float):
    return {
        "event_id": event,
        "fight_id": fight,
        "frame": frame,
        "acquisition": {
            "reason": reason,
            "priority": priority,
            "ensemble_std": std,
            "absolute_conservative_score": 0.001,
        },
    }


def test_active_acquisition_prioritizes_ood_and_enforces_event_separation() -> None:
    candidates = [
        _row("near", "f1", 110, "HIGH_UNCERTAINTY", 4, 0.5),
        _row("ood", "f1", 100, "OOD", 5, 0.1),
        _row("far", "f1", 150, "LOW_PPOSITIVE", 2, 0.2),
        _row("other", "f2", 105, "HIGH_REGRESSION_RISK", 3, 0.3),
    ]
    selected = select_active_events(candidates, 3, minimum_frame_separation=30)
    assert selected[0]["event_id"] == "ood"
    assert "near" not in {row["event_id"] for row in selected}
    assert {row["event_id"] for row in selected} == {"ood", "far", "other"}
