from __future__ import annotations

import unittest

from automation.analyze_guidance_v1_events import intervention_events


def frame(index: int, action: str) -> dict:
    return {
        "frame": index,
        "hybrid": {"selected_action": action},
    }


class GuidanceV2EventAnalysisTests(unittest.TestCase):
    def test_groups_contiguous_nondefault_frames(self) -> None:
        frames = [
            frame(0, "BT_DEFAULT"),
            frame(1, "VP_EL_POS_SMALL"),
            frame(2, "VP_EL_POS_SMALL"),
            frame(3, "BT_DEFAULT"),
            frame(4, "VP_EL_POS_SMALL"),
        ]
        events = intervention_events(frames)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["intervention_event_id"], 1)
        self.assertEqual(events[0]["duration_frames"], 2)
        self.assertEqual(events[1]["start_frame"], 4)

    def test_action_change_starts_new_event(self) -> None:
        events = intervention_events(
            [frame(0, "VP_AZ_POS_SMALL"), frame(1, "VP_AZ_NEG_SMALL")]
        )
        self.assertEqual([event["duration_frames"] for event in events], [1, 1])
        self.assertEqual([event["action"] for event in events], [
            "VP_AZ_POS_SMALL", "VP_AZ_NEG_SMALL"
        ])


if __name__ == "__main__":
    unittest.main()
