from __future__ import annotations

import unittest

import numpy as np

from dogfight.envs.training_batch_contract import Rear120TrainingBatchTracker
from dogfight.sim.state_schema import StateIndex


def _state(n: float, e: float, yaw: float) -> np.ndarray:
    state = np.zeros(51, dtype=np.float32)
    state[StateIndex.N] = n
    state[StateIndex.E] = e
    state[StateIndex.D] = -5000.0
    state[StateIndex.YAW] = yaw
    state[6] = 230.0
    state[StateIndex.KCAS] = 230.0
    state[StateIndex.ALT] = 5000.0
    return state


class Rear120TrainingBatchContractTests(unittest.TestCase):
    def tracker(self):
        return Rear120TrainingBatchTracker(
            {
                "mode": "rear120_segment",
                "minimum_target_ata_deg": 120.0,
                "truncate_on_exit": True,
                "mask_exit_reward": True,
            }
        )

    def test_records_action_state_inside_hard_envelope(self) -> None:
        tracker = self.tracker()
        own = _state(0.0, 0.0, 0.0)
        target = _state(1000.0, 0.0, 0.0)
        tracker.validate_initial_state(own, target)

        sample = tracker.record_action_state(
            own,
            target,
            {"gate": {"offensive_eligible": True, "pre_aim_eligible": False}},
        )
        summary = tracker.summary()

        self.assertTrue(sample["rear120_eligible"])
        self.assertEqual(summary["rear120_sample_fraction"], 1.0)
        self.assertEqual(summary["offensive_sample_fraction"], 1.0)
        self.assertEqual(summary["ineligible_sample_count"], 0)
        self.assertEqual(sum(summary["histograms"]["target_ata_deg"].values()), 1)

    def test_exit_truncates_before_next_ineligible_action(self) -> None:
        tracker = self.tracker()
        own = _state(0.0, 0.0, 0.0)
        target = _state(1000.0, 0.0, 0.0)
        tracker.record_action_state(own, target, {})

        target[StateIndex.YAW] = 90.0
        self.assertTrue(tracker.should_truncate_after_step(own, target))
        self.assertEqual(tracker.summary()["boundary_exit_transition_count"], 1)
        self.assertEqual(tracker.summary()["ineligible_sample_count"], 0)

    def test_invalid_initial_state_fails_fast(self) -> None:
        tracker = self.tracker()
        own = _state(0.0, 0.0, 0.0)
        target = _state(1000.0, 0.0, 180.0)
        with self.assertRaisesRegex(ValueError, "outside the hard envelope"):
            tracker.validate_initial_state(own, target)


if __name__ == "__main__":
    unittest.main()
