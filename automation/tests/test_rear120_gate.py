from __future__ import annotations

import math
import time
import unittest

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.ai.hybrid_action_provider import (
    Rear120ActivationGate,
    Rear120EligibilityGate,
    ResidualInferenceActionProvider,
    target_ata_deg,
)
from dogfight.sim.state_schema import StateIndex


def _state(
    n: float,
    e: float,
    d: float,
    *,
    yaw: float,
    pitch: float = 0.0,
    roll: float = 0.0,
    speed: float = 230.0,
) -> np.ndarray:
    result = np.zeros(51, dtype=np.float32)
    result[:6] = [n, e, d, roll, pitch, yaw]
    result[6] = speed
    result[StateIndex.KCAS] = speed
    result[StateIndex.ALT] = -d
    result[StateIndex.HEALTH] = 1.0
    return result


def _horizontal_geometry(target_ata: float, *, mirror: int = 1, heading: float = 0.0):
    target = _state(1000.0, 0.0, -5000.0, yaw=heading)
    relative_bearing = mirror * target_ata
    bearing = math.radians(heading + relative_bearing)
    own_n = target[0] + 1000.0 * math.cos(bearing)
    own_e = target[1] + 1000.0 * math.sin(bearing)
    own_to_target_heading = (heading + relative_bearing + 180.0) % 360.0
    own = _state(own_n, own_e, -5000.0, yaw=own_to_target_heading)
    return own, target


class _CountingProvider(ActionProvider):
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.calls = 0

    def compute_action(self, context) -> ActionResult:
        self.calls += 1
        return ActionResult(self.action.copy(), "test")


class _FailingProvider(_CountingProvider):
    def __init__(self, mode: str):
        super().__init__([0.0, 0.0, 0.0, 0.0])
        self.mode = mode

    def compute_action(self, context) -> ActionResult:
        self.calls += 1
        if self.mode == "exception":
            raise RuntimeError("inference failed")
        if self.mode == "nonfinite":
            return ActionResult(
                np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32),
                "test",
            )
        if self.mode == "slow":
            time.sleep(0.002)
        return ActionResult(np.zeros(4, dtype=np.float32), "test")


class Rear120GateTests(unittest.TestCase):
    def test_canonical_target_ata_meaning_and_boundaries(self) -> None:
        cases = [
            (180.0, 1, 180.0, True),
            (120.0, 1, 120.0, True),
            (120.0, -1, 120.0, True),
            (119.0, 1, 119.0, False),
            (90.0, 1, 90.0, False),
            (0.0, 1, 0.0, False),
        ]
        for angle, mirror, expected, eligible in cases:
            with self.subTest(angle=angle, mirror=mirror):
                own, target = _horizontal_geometry(angle, mirror=mirror)
                measured = target_ata_deg(own, target)
                self.assertAlmostEqual(measured, expected, places=3)
                gate = Rear120EligibilityGate()
                self.assertEqual(gate.update(own, target)["active"], eligible)

    def test_vertical_attitude_roll_and_heading_wrap_preserve_tail_meaning(self) -> None:
        pitch = math.radians(20.0)
        forward = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
        target = _state(1000.0, 0.0, -5000.0, yaw=0.0, pitch=20.0, roll=75.0)
        own_position = target[:3] - 1000.0 * forward
        own = _state(*own_position, yaw=0.0, pitch=20.0, roll=-45.0)
        self.assertAlmostEqual(target_ata_deg(own, target), 180.0, places=3)

        for heading in (5.0, 355.0):
            own, target = _horizontal_geometry(180.0, heading=heading)
            self.assertAlmostEqual(target_ata_deg(own, target), 180.0, places=3)

    def test_exit_hysteresis_keeps_110_to_120_band_active(self) -> None:
        gate = Rear120EligibilityGate(
            {"enter_target_ata_deg": 120.0, "exit_target_ata_deg": 110.0}
        )
        own, target = _horizontal_geometry(120.0)
        self.assertTrue(gate.update(own, target)["entry"])
        own, target = _horizontal_geometry(115.0)
        self.assertTrue(gate.update(own, target)["active"])
        own, target = _horizontal_geometry(109.0)
        self.assertTrue(gate.update(own, target)["exit"])

    def test_rear120_activation_requires_attack_geometry_and_safety(self) -> None:
        gate = Rear120ActivationGate()
        own, target = _horizontal_geometry(180.0)
        result = gate.update(
            own,
            target,
            sim_time_s=0.0,
            bt_action=np.array([0.2, -0.1, 0.0, 0.8]),
        )
        self.assertTrue(result["rear120_eligible"])
        self.assertTrue(result["offensive_eligible"] or result["pre_aim_eligible"])
        self.assertTrue(result["active"])

        low = own.copy()
        low[StateIndex.ALT] = 300.0
        low[StateIndex.D] = -300.0
        result = gate.update(
            low,
            target,
            sim_time_s=0.0,
            bt_action=np.array([0.2, -0.1, 0.0, 0.8]),
        )
        self.assertFalse(result["active"])
        self.assertIn("low_altitude", result["safety_veto_reasons"])

    def test_negative_gate_skips_rl_and_preserves_all_bt_axes(self) -> None:
        bt = _CountingProvider([0.25, -0.4, 0.1, 0.73])
        rl = _CountingProvider([1.0, 1.0, 1.0, -1.0])
        provider = ResidualInferenceActionProvider(
            bt,
            rl,
            residual_scale=0.125,
            gate_kind="rear120",
            composition_mode="saturation_aware",
        )
        own, target = _horizontal_geometry(90.0)
        context = ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=own,
            target_state=target,
            observation=np.zeros(10, dtype=np.float32),
            info={"sim_time_s": 0.0},
        )

        result = provider.compute_action(context)

        np.testing.assert_array_equal(result.action, bt.action)
        self.assertEqual(bt.calls, 1)
        self.assertEqual(rl.calls, 0)
        self.assertEqual(float(result.action[3]), float(bt.action[3]))
        self.assertEqual(provider.telemetry()["rl_inference_calls"], 0)

    def test_inference_exception_nonfinite_and_timeout_fall_back_to_bt(self) -> None:
        own, target = _horizontal_geometry(180.0)
        context = ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=own,
            target_state=target,
            observation=np.zeros(10, dtype=np.float32),
            info={"sim_time_s": 0.0},
        )
        for mode, telemetry_key in (
            ("exception", "rl_exception_fallback_steps"),
            ("nonfinite", "rl_nonfinite_fallback_steps"),
            ("slow", "rl_timeout_fallback_steps"),
        ):
            with self.subTest(mode=mode):
                bt = _CountingProvider([0.25, -0.4, 0.1, 0.73])
                provider = ResidualInferenceActionProvider(
                    bt,
                    _FailingProvider(mode),
                    residual_scale=0.125,
                    gate_kind="rear120",
                    inference_timeout_s=0.0001 if mode == "slow" else 0.1667,
                )
                result = provider.compute_action(context)
                np.testing.assert_array_equal(result.action, bt.action)
                self.assertEqual(result.source, "bt_residual_inference_fallback")
                self.assertEqual(result.info["rl_fallback_reason"], {
                    "exception": "inference_exception",
                    "nonfinite": "nonfinite_output",
                    "slow": "inference_timeout",
                }[mode])
                self.assertEqual(provider.telemetry()[telemetry_key], 1)
                self.assertAlmostEqual(float(result.action[3]), 0.73, places=6)

    def test_gate_off_is_exact_bt_for_600_frames_and_200_seconds(self) -> None:
        own, target = _horizontal_geometry(90.0)
        context = ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=own,
            target_state=target,
            observation=np.zeros(10, dtype=np.float32),
            info={"sim_time_s": 0.0},
        )
        for frame_count in (600, 200 * 60):
            with self.subTest(frame_count=frame_count):
                bt = _CountingProvider([0.25, -0.4, 0.1, 0.73])
                rl = _CountingProvider([1.0, 1.0, 1.0, 0.0])
                provider = ResidualInferenceActionProvider(
                    bt,
                    rl,
                    residual_scale=0.125,
                    gate_kind="rear120",
                    composition_mode="saturation_aware",
                )
                for frame in range(frame_count):
                    context.info["sim_time_s"] = frame / 60.0
                    result = provider.compute_action(context)
                    np.testing.assert_array_equal(result.action, bt.action)
                self.assertEqual(bt.calls, frame_count)
                self.assertEqual(rl.calls, 0)
                self.assertEqual(provider.telemetry()["rl_inference_calls"], 0)


if __name__ == "__main__":
    unittest.main()
