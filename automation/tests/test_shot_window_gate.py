from __future__ import annotations

import math
import unittest

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.ai.hybrid_action_provider import (
    ResidualInferenceActionProvider,
    ShotWindowActivationGate,
    ShotWindowGateConfig,
)
from dogfight.sim.state_schema import StateIndex


def _state(
    n: float,
    e: float,
    d: float = -5000.0,
    *,
    yaw: float,
    speed: float = 230.0,
) -> np.ndarray:
    result = np.zeros(51, dtype=np.float32)
    result[:6] = [n, e, d, 0.0, 0.0, yaw]
    result[6] = speed
    result[StateIndex.KCAS] = speed
    result[StateIndex.ALT] = -d
    result[StateIndex.HEALTH] = 1.0
    return result


def _geometry(
    target_ata_deg: float,
    *,
    distance_m: float = 930.0,
    own_aim_error_deg: float = 0.0,
    speed: float = 230.0,
) -> tuple[np.ndarray, np.ndarray]:
    target = _state(1000.0, 0.0, yaw=0.0, speed=speed)
    bearing = math.radians(target_ata_deg)
    own_n = target[0] + distance_m * math.cos(bearing)
    own_e = target[1] + distance_m * math.sin(bearing)
    own_to_target = (target_ata_deg + 180.0 + own_aim_error_deg) % 360.0
    own = _state(own_n, own_e, yaw=own_to_target, speed=speed)
    return own, target


def _context(own, target, *, sim_time_s: float = 0.0) -> ActionContext:
    return ActionContext(
        sim=None,
        opponent_sim=None,
        ownship_state=own,
        target_state=target,
        observation=np.zeros(16, dtype=np.float32),
        info={"sim_time_s": sim_time_s},
    )


class _Provider(ActionProvider):
    def __init__(self, action, *, fail: bool = False):
        self.action = np.asarray(action, dtype=np.float32)
        self.calls = 0
        self.fail = fail

    def compute_action(self, context) -> ActionResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("diagnostic failure")
        return ActionResult(self.action.copy(), "test")


class ShotWindowGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bt_action = np.array([0.2, -0.1, 0.05, 0.82], dtype=np.float32)

    def _update(self, gate, own, target, frame=0):
        return gate.update(
            own,
            target,
            sim_time_s=frame / 60.0,
            bt_action=self.bt_action,
        )

    def test_static_geometry_and_wez_suite(self) -> None:
        cases = (
            ("deep_rear", 180.0, 930.0, 0.0, True, True),
            ("rear120_boundary", 120.0, 930.0, 0.0, True, False),
            ("beam", 90.0, 930.0, 0.0, False, False),
            ("front", 0.0, 930.0, 0.0, False, False),
            ("wez_outside", 180.0, 1050.0, 0.0, True, False),
            ("wez_approaching", 180.0, 930.0, 0.0, True, True),
            ("wez_inside", 180.0, 900.0, 0.0, True, True),
            ("los_good", 180.0, 930.0, 0.5, True, True),
            ("los_bad", 180.0, 930.0, 8.0, True, False),
        )
        for name, target_ata, distance, error, armed, shot in cases:
            with self.subTest(name=name):
                gate = ShotWindowActivationGate()
                own, target = _geometry(
                    target_ata,
                    distance_m=distance,
                    own_aim_error_deg=error,
                )
                result = self._update(gate, own, target)
                self.assertEqual(result["arming_condition"], armed)
                self.assertEqual(result["shot_enter_condition"], shot)
                self.assertFalse(result["active"])

    def test_entry_timeout_cooldown_condition_exit_and_reentry(self) -> None:
        gate = ShotWindowActivationGate(
            ShotWindowGateConfig(max_active_steps=5, cooldown_steps=3)
        )
        own, target = _geometry(180.0, distance_m=930.0)

        self.assertEqual(self._update(gate, own, target, 0)["state"], "ARMED")
        self.assertTrue(self._update(gate, own, target, 1)["entry"])
        for frame in range(2, 6):
            self.assertTrue(self._update(gate, own, target, frame)["active"])
        timeout = self._update(gate, own, target, 6)
        self.assertEqual(timeout["state"], "COOLDOWN")
        self.assertEqual(timeout["exit_reason"], "timeout")

        for frame in range(7, 12):
            self.assertEqual(
                self._update(gate, own, target, frame)["state"], "COOLDOWN"
            )

        outside, outside_target = _geometry(180.0, distance_m=1100.0)
        rearmed = self._update(gate, outside, outside_target, 12)
        self.assertEqual(rearmed["state"], "ARMED")
        reentered = self._update(gate, own, target, 13)
        self.assertTrue(reentered["entry"])

        telemetry = gate.telemetry()
        self.assertEqual(telemetry["window_entry_count"], 2)
        self.assertEqual(telemetry["window_exit_count"], 1)
        self.assertEqual(telemetry["window_timeout_count"], 1)
        self.assertEqual(telemetry["window_reentry_count"], 1)
        self.assertEqual(telemetry["active_duration_max"], 5 / 60.0)
        self.assertGreaterEqual(telemetry["cooldown_duration"], 3 / 60.0)

    def test_time_only_cooldown_is_an_explicit_alternative(self) -> None:
        gate = ShotWindowActivationGate(
            ShotWindowGateConfig(
                max_active_steps=2,
                cooldown_steps=2,
                require_condition_exit_for_rearm=False,
            )
        )
        own, target = _geometry(180.0, distance_m=930.0)
        states = [self._update(gate, own, target, frame)["state"] for frame in range(8)]
        self.assertEqual(
            states,
            ["ARMED", "ACTIVE", "ACTIVE", "COOLDOWN", "COOLDOWN", "ARMED", "ACTIVE", "ACTIVE"],
        )
        self.assertEqual(gate.telemetry()["window_reentry_count"], 1)

    def test_condition_exit_and_boundary_oscillation(self) -> None:
        gate = ShotWindowActivationGate(
            ShotWindowGateConfig(max_active_steps=100, cooldown_steps=3)
        )
        own, target = _geometry(180.0, distance_m=930.0)
        self._update(gate, own, target, 0)
        self._update(gate, own, target, 1)
        for frame, distance in enumerate((935.0, 950.0, 980.0, 935.0), start=2):
            noisy_own, noisy_target = _geometry(180.0, distance_m=distance)
            self.assertTrue(self._update(gate, noisy_own, noisy_target, frame)["active"])
        exit_own, exit_target = _geometry(180.0, distance_m=1000.0)
        exited = self._update(gate, exit_own, exit_target, 6)
        self.assertEqual(exited["exit_reason"], "condition_exit")
        self.assertEqual(gate.telemetry()["window_condition_exit_count"], 1)

    def test_high_and_low_los_rate_are_observable(self) -> None:
        gate = ShotWindowActivationGate()
        own, target = _geometry(180.0, own_aim_error_deg=0.0)
        self._update(gate, own, target, 0)
        fast_own, fast_target = _geometry(180.0, own_aim_error_deg=1.0)
        fast = self._update(gate, fast_own, fast_target, 1)
        slow_own, slow_target = _geometry(180.0, own_aim_error_deg=1.1)
        slow = self._update(gate, slow_own, slow_target, 2)
        self.assertGreater(abs(fast["aim_error_rate_deg_s"]), 50.0)
        self.assertLess(abs(slow["aim_error_rate_deg_s"]), 10.0)
        self.assertTrue(np.isfinite(fast["closing_rate_m_s"]))

    def test_low_altitude_low_speed_and_surface_authority_veto(self) -> None:
        own, target = _geometry(180.0)
        variants = []
        low_altitude = own.copy()
        low_altitude[StateIndex.ALT] = 300.0
        variants.append((low_altitude, self.bt_action, "low_altitude"))
        low_speed = own.copy()
        low_speed[StateIndex.KCAS] = 160.0
        variants.append((low_speed, self.bt_action, "low_speed"))
        variants.append((own, np.array([1.0, -1.0, 1.0, 0.8]), "no_surface_authority"))
        for state, bt_action, reason in variants:
            with self.subTest(reason=reason):
                gate = ShotWindowActivationGate()
                result = gate.update(
                    state,
                    target,
                    sim_time_s=0.0,
                    bt_action=bt_action,
                )
                self.assertFalse(result["active"])
                self.assertIn(reason, result["safety_veto_reasons"])

    def test_gate_off_exact_bt_inference_skip_and_throttle_bt_only(self) -> None:
        bt = _Provider([0.2, -0.1, 0.05, 0.82])
        rl = _Provider([1.0, 1.0, 1.0, -1.0])
        provider = ResidualInferenceActionProvider(
            bt,
            rl,
            residual_scale=0.125,
            gate_kind="shot_window",
            composition_mode="saturation_aware",
        )
        beam_own, beam_target = _geometry(90.0)
        off = provider.compute_action(_context(beam_own, beam_target))
        np.testing.assert_array_equal(off.action, bt.action)
        self.assertEqual(rl.calls, 0)

        own, target = _geometry(180.0)
        provider.compute_action(_context(own, target, sim_time_s=1 / 60.0))
        active = provider.compute_action(_context(own, target, sim_time_s=2 / 60.0))
        self.assertEqual(rl.calls, 1)
        self.assertAlmostEqual(float(active.action[3]), float(bt.action[3]), places=7)

    def test_shot_window_inference_failure_falls_back_to_exact_bt(self) -> None:
        bt = _Provider([0.2, -0.1, 0.05, 0.82])
        rl = _Provider([0.0, 0.0, 0.0, 0.0], fail=True)
        provider = ResidualInferenceActionProvider(
            bt,
            rl,
            residual_scale=0.125,
            gate_kind="shot_window",
        )
        own, target = _geometry(180.0)
        provider.compute_action(_context(own, target, sim_time_s=0.0))
        result = provider.compute_action(_context(own, target, sim_time_s=1 / 60.0))
        np.testing.assert_array_equal(result.action, bt.action)
        self.assertEqual(result.info["rl_fallback_reason"], "inference_exception")


if __name__ == "__main__":
    unittest.main()
