from __future__ import annotations

import unittest

import numpy as np

from dogfight.ai.action_provider import (
    ActionContext,
    ActionProvider,
    ActionResult,
    policy_action_to_sim_action,
)
from dogfight.ai.hybrid_action_provider import (
    AimGateConfig,
    AimResidualGate,
    HybridActionProvider,
    OffensiveGateConfig,
    OffensiveResidualGate,
    ResidualTrainingActionProvider,
)


def state(n: float, e: float, yaw: float) -> np.ndarray:
    value = np.zeros(51, dtype=np.float32)
    value[0] = n
    value[1] = e
    value[5] = yaw
    value[12] = 220.0
    value[44] = 3000.0
    value[45] = 1.0
    return value


def context(own=None, target=None, *, residual=None, sim_time_s=None) -> ActionContext:
    info = {}
    if residual is not None:
        info["residual_action"] = residual
    if sim_time_s is not None:
        info["sim_time_s"] = sim_time_s
    return ActionContext(None, None, own, target, np.zeros(16), info)


class CountingProvider(ActionProvider):
    def __init__(self, action, source):
        self.action = np.asarray(action, dtype=np.float32)
        self.source = source
        self.calls = 0
        self.resets = 0

    def reset(self, context=None) -> None:
        self.resets += 1

    def compute_action(self, context) -> ActionResult:
        self.calls += 1
        return ActionResult(self.action.copy(), self.source)


class OffensiveHybridTests(unittest.TestCase):
    def setUp(self) -> None:
        self.own = state(0.0, 0.0, 0.0)
        self.target = state(1000.0, 0.0, 0.0)

    def test_policy_throttle_mapping_occurs_once(self) -> None:
        self.assertEqual(policy_action_to_sim_action([0, 0, 0, -1])[3], 0.0)
        self.assertEqual(policy_action_to_sim_action([0, 0, 0, 0])[3], 0.5)
        self.assertEqual(policy_action_to_sim_action([0, 0, 0, 1])[3], 1.0)

    def test_gate_off_is_exact_bt_and_skips_rl(self) -> None:
        rl = CountingProvider([0.8, -0.8, 0.5, 0.0], "rl")
        bt = CountingProvider([0.2, -0.3, 0.1, 0.8], "bt")
        provider = HybridActionProvider(rl, bt, mode="offensive_residual", residual_scale=0.15)
        target = state(0.0, 1000.0, 0.0)

        result = provider.compute_action(context(self.own, target))

        np.testing.assert_array_equal(result.action, bt.action)
        self.assertEqual(rl.calls, 0)
        self.assertEqual(result.info["effective_residual_scale"], 0.0)

    def test_gate_on_scale_and_convex_throttle(self) -> None:
        rl = CountingProvider([0.4, -0.2, 0.2, 0.2], "rl")
        bt = CountingProvider([0.3, -0.4, 0.1, 0.9], "bt")
        provider = HybridActionProvider(rl, bt, mode="offensive_residual", residual_scale=0.15)

        result = provider.compute_action(context(self.own, self.target))

        np.testing.assert_allclose(result.action[:3], [0.36, -0.43, 0.13], atol=1e-6)
        self.assertAlmostEqual(float(result.action[3]), 0.795, places=6)
        self.assertTrue(0.10 <= result.info["effective_residual_scale"] <= 0.20)

    def test_offensive_scale_is_bounded(self) -> None:
        rl = CountingProvider([0, 0, 0, 0.5], "rl")
        bt = CountingProvider([0, 0, 0, 0.5], "bt")
        for value in (0.0, 0.09, 0.21, 0.35):
            with self.assertRaises(ValueError):
                HybridActionProvider(rl, bt, mode="offensive_residual", residual_scale=value)

    def test_low_energy_guard_preserves_bt_throttle_reduction_authority(self) -> None:
        rl = CountingProvider([0.4, 0.0, 0.0, 0.2], "rl")
        bt = CountingProvider([0.3, 0.0, 0.0, 0.9], "bt")
        provider = HybridActionProvider(rl, bt, mode="offensive_residual", residual_scale=0.15)
        low_energy = self.own.copy()
        low_energy[12] = 150.0

        result = provider.compute_action(context(low_energy, self.target))

        self.assertAlmostEqual(float(result.action[3]), 0.9, places=6)
        self.assertAlmostEqual(float(result.action[0]), 0.36, places=6)
        self.assertTrue(result.info["throttle_guard_active"])
        self.assertEqual(result.info["effective_throttle_scale"], 0.0)

    def test_hysteresis_prevents_boundary_chatter(self) -> None:
        gate = OffensiveResidualGate(
            OffensiveGateConfig(
                enter_ata_deg=30.0,
                exit_ata_deg=45.0,
                enter_min_target_ata_deg=100.0,
                exit_min_target_ata_deg=80.0,
            )
        )
        self.assertTrue(gate.update(self.own, self.target)["active"])
        for e in (400.0, 500.0, 600.0, 500.0):
            noisy_target = state(1000.0, e, 0.0)
            self.assertTrue(gate.update(self.own, noisy_target)["active"])
        self.assertEqual(gate.entries, 1)
        self.assertEqual(gate.exits, 0)

    def test_rl_is_cached_and_reset_on_gate_entry(self) -> None:
        rl = CountingProvider([0.1, 0.1, 0.1, 0.5], "rl")
        bt = CountingProvider([0, 0, 0, 0.8], "bt")
        provider = HybridActionProvider(
            rl, bt, mode="offensive_residual", residual_scale=0.15,
            primary_action_repeat=3,
        )
        initial_resets = rl.resets
        for _ in range(5):
            provider.compute_action(context(self.own, self.target))
        self.assertEqual(rl.calls, 2)
        self.assertEqual(rl.resets, initial_resets + 1)
        self.assertEqual(bt.calls, 5)

    def test_reset_clears_gate_cache_and_telemetry(self) -> None:
        rl = CountingProvider([1, 1, 1, 1], "rl")
        bt = CountingProvider([1, 1, 1, 1], "bt")
        provider = HybridActionProvider(rl, bt, mode="offensive_residual", residual_scale=0.2)
        provider.compute_action(context(self.own, self.target))
        self.assertEqual(provider.telemetry()["rl_inference_calls"], 1)

        provider.reset(None)

        telemetry = provider.telemetry()
        self.assertEqual(telemetry["rl_inference_calls"], 0)
        self.assertEqual(telemetry["offensive_gate_entries"], 0)
        self.assertEqual(telemetry["rl_correction_steps"], 0)

    def test_aim_gate_uses_phase_half_angles_and_hysteresis(self) -> None:
        gate = AimResidualGate(
            AimGateConfig(
                enter_angle_margin_deg=1.0,
                exit_angle_margin_deg=3.0,
                enter_range_margin_m=0.0,
                exit_range_margin_m=200.0,
                min_hold_steps=0,
            )
        )
        phase1_target = state(900.0, 20.0, 0.0)
        entered = gate.update(self.own, phase1_target, sim_time_s=50.0)
        self.assertEqual(entered["phase"], 1)
        self.assertTrue(entered["active"])

        hysteresis_target = state(1000.0, 50.0, 0.0)
        held = gate.update(self.own, hysteresis_target, sim_time_s=50.0)
        self.assertTrue(held["active"])

        exited = gate.update(
            self.own,
            state(1000.0, 200.0, 0.0),
            sim_time_s=50.0,
        )
        self.assertFalse(exited["active"])
        self.assertEqual(gate.entries, 1)
        self.assertEqual(gate.exits, 1)

    def test_training_residual_forces_bt_throttle_and_gate_off_equality(self) -> None:
        bt = CountingProvider([0.2, -0.3, 0.1, 0.77], "bt")
        provider = ResidualTrainingActionProvider(
            bt,
            residual_scale=0.125,
            gate_kind="aim",
        )
        off_target = state(0.0, 3000.0, 0.0)

        off = provider.compute_action(
            context(
                self.own,
                off_target,
                residual=[1.0, 1.0, 1.0, -1.0],
                sim_time_s=0.0,
            )
        )
        np.testing.assert_array_equal(off.action, bt.action)

        on = provider.compute_action(
            context(
                self.own,
                self.target,
                residual=[0.8, -0.4, 0.2, -1.0],
                sim_time_s=0.0,
            )
        )
        np.testing.assert_allclose(on.action[:3], [0.3, -0.35, 0.125], atol=1e-6)
        self.assertAlmostEqual(float(on.action[3]), 0.77, places=6)
        self.assertTrue(on.info["throttle_residual_forced_zero"])


if __name__ == "__main__":
    unittest.main()
