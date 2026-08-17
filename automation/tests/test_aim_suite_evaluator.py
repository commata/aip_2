import json
from pathlib import Path

import pytest

from automation.evaluate_aim_suite import combine_results, load_suite


ROOT = Path(__file__).resolve().parents[2]


def evaluation(seed, damage):
    records = []
    for controller, value in (("pure_0815", 0.1), ("hybrid_0.125", damage)):
        records.append(
            {
                "seed": seed,
                "controller": controller,
                "variant_name": None,
                "outcome": "timeout",
                "ownship_crash": False,
                "target_crash": False,
                "damage_dealt": value,
                "damage_received": 0.0,
                "health_margin": value,
                "episode_seconds": 30.0,
            }
        )
    return {"preflight": {"bundle_weights_sha256": "ABC"}, "records": records}


def test_combine_records_preserves_case_identity_and_zero_stochastic_claim():
    suite = {"name": "test", "cases": [{"name": "left", "seed": 1}, {"name": "right", "seed": 2}]}
    result = combine_results(
        suite,
        [
            ({"name": "left", "seed": 1, "scenario": "left.json"}, evaluation(1, 0.2)),
            ({"name": "right", "seed": 2, "scenario": "right.json"}, evaluation(2, 0.3)),
        ],
    )
    assert result["contract"]["deterministic_geometry_samples"] == 2
    assert result["contract"]["stochastic_independent_samples"] == 0
    assert {row["suite_case"] for row in result["records"]} == {"left", "right"}
    assert {row["variant_name"] for row in result["records"]} == {"left", "right"}
    assert {row["source_variant_name"] for row in result["records"]} == {None}
    assert result["summary"]["paired"]["hybrid_0.125"]["pairs"] == 2


def test_suite_rejects_duplicate_seed_labels():
    path = Path("artifacts/test_tmp/aim_suite_evaluator/duplicate.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cases": [{"name": "a", "seed": 1}, {"name": "b", "seed": 1}]}))
    with pytest.raises(ValueError, match="seed label"):
        load_suite(path)


def test_checked_in_holdout_suite_covers_each_frozen_mirror_geometry_once():
    suite = load_suite(
        ROOT / "automation/scenarios/0815_aim_proxy_longrun_holdout_suite.json"
    )
    assert [case["name"] for case in suite["cases"]] == [
        "lateral_left",
        "lateral_right",
        "crossing_left",
        "crossing_right",
        "vertical_high",
        "vertical_low",
    ]
    for case in suite["cases"]:
        scenario = ROOT / case["scenario"]
        assert scenario.is_file()
        payload = json.loads(scenario.read_text(encoding="utf-8"))
        assert payload["name"].endswith(case["name"])
