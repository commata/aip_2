from __future__ import annotations

import json
from pathlib import Path

from automation.evaluate_gate_selectivity import evaluate_suite


ROOT = Path(__file__).resolve().parents[2]


def test_committed_gate_selectivity_suite_covers_required_regions() -> None:
    payload = json.loads(
        (ROOT / "automation/scenarios/residual_gate_selectivity_v1.json").read_text(
            encoding="utf-8"
        )
    )

    result = evaluate_suite(payload)

    categories = " ".join(case["category"] for case in result["cases"])
    for required in (
        "deep rear",
        "rear120",
        "beam",
        "front",
        "offensive",
        "non-offensive",
        "WEZ inside",
        "WEZ outside",
        "entry / exit / reentry",
    ):
        assert required in categories
    trace = next(
        case for case in result["cases"]
        if case["case"] == "rear120_hysteresis_exit_reentry"
    )
    assert trace["gate_entry_count"] == 2
    assert trace["gate_exit_count"] == 1
    assert trace["mean_active_duration_steps"] == 2.0
    assert trace["max_active_duration_steps"] == 3
    assert result["aggregate"]["gate_active_ratio"] < 0.8
