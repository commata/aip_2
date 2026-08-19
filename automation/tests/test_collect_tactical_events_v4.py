from __future__ import annotations

from automation.collect_tactical_events_v4 import build_discovery_cases


def test_discovery_cases_cover_required_geometry_families() -> None:
    cases = build_discovery_cases(3)
    geometries = {case["geometry"] for case in cases}
    assert {
        "lateral_left",
        "lateral_right",
        "vertical_high",
        "vertical_low",
        "crossing_left",
        "crossing_right",
        "tail_chase",
        "neutral",
        "head_on",
        "high_closing",
        "long_range",
    } <= geometries
    assert len({case["case_id"] for case in cases}) == len(cases) == 23
    assert all(
        case["scenario"]["env_config"]["initial_scenario"][
            "legacy_use_random_scenario"
        ]
        is False
        for case in cases
    )
