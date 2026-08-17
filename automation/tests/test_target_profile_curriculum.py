from __future__ import annotations

import unittest

import numpy as np

from dogfight.ai.target_profile_curriculum import TargetProfileCurriculum


class _FakeProvider:
    def __init__(self, profile_id: str) -> None:
        self.profile_id = profile_id
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class TargetProfileCurriculumTests(unittest.TestCase):
    def test_weighted_episode_selection_is_seeded_and_resets_active_bt(self) -> None:
        built: dict[str, _FakeProvider] = {}

        def factory(profile):
            provider = _FakeProvider(profile["profile_id"])
            built[profile["profile_id"]] = provider
            return provider

        profiles = [
            {
                "profile_id": "autopilot",
                "backend_type": "autopilot",
                "behavior_cluster": "scripted",
                "weight": 1,
            },
            {
                "profile_id": "bt0815",
                "backend_type": "behavior_tree",
                "behavior_cluster": "bt_family",
                "dll_path": "unused.dll",
                "weight": 3,
            },
        ]
        first = TargetProfileCurriculum(profiles, provider_factory=factory)
        second = TargetProfileCurriculum(profiles, provider_factory=factory)
        rng_a = np.random.default_rng(1701)
        rng_b = np.random.default_rng(1701)

        sequence_a = [first.select_episode(rng_a).profile_id for _ in range(12)]
        sequence_b = [second.select_episode(rng_b).profile_id for _ in range(12)]

        self.assertEqual(sequence_a, sequence_b)
        self.assertIn("autopilot", sequence_a)
        self.assertIn("bt0815", sequence_a)
        self.assertEqual(first.telemetry()["target_profile_episode_total"], 12)
        self.assertGreater(built["bt0815"].close_calls, 0)

    def test_identity_is_telemetry_only_not_an_observation_value(self) -> None:
        curriculum = TargetProfileCurriculum(
            [
                {
                    "profile_id": "autopilot",
                    "backend_type": "autopilot",
                    "behavior_cluster": "scripted",
                    "weight": 1,
                }
            ]
        )

        selection = curriculum.select_episode(np.random.default_rng(1))

        self.assertEqual(selection.profile_id, "autopilot")
        self.assertIsNone(curriculum.active_provider)
        self.assertEqual(curriculum.telemetry()["target_profile_id"], "autopilot")


if __name__ == "__main__":
    unittest.main()
