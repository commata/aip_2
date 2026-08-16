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
    CombinedResidualGate,
    HybridActionProvider,
    OffensiveGateConfig,
    OffensiveResidualGate,
    ResidualInferenceActionProvider,
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

    def test_combined_gate_requires_aim_and_offensive_overlap(self) -> None:
        gate = CombinedResidualGate()
        aim_only_target = state(1000.0, 0.0, 180.0)

        aim_only = gate.update(self.own, aim_only_target, sim_time_s=0.0)
        combined = gate.update(self.own, self.target, sim_time_s=0.0)

        self.assertFalse(aim_only["active"])
        self.assertTrue(aim_only["aim_gate"]["active"])
        self.assertFalse(aim_only["offensive_gate"]["active"])
        self.assertTrue(combined["active"])
        telemetry = gate.telemetry()
        self.assertEqual(telemetry["combined_gate_entries"], 1)
        self.assertEqual(telemetry["combined_gate_active_steps"], 1)

    def test_combined_inference_gate_off_is_exact_bt(self) -> None:
        bt = CountingProvider([0.2, -0.3, 0.1, 0.77], "bt")
        rl = CountingProvider([0.8, -0.4, 0.2, 0.0], "rl")
        provider = ResidualInferenceActionProvider(
            bt,
            rl,
            residual_scale=0.125,
            gate_kind="combined",
        )

        result = provider.compute_action(
            context(self.own, state(1000.0, 0.0, 180.0), sim_time_s=0.0)
        )

        np.testing.assert_array_equal(result.action, bt.action)
        self.assertEqual(rl.calls, 0)
        self.assertEqual(provider.telemetry()["combined_gate_active_ratio"], 0.0)

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

    def test_inference_residual_gate_off_is_exact_bt_and_skips_policy(self) -> None:
        bt = CountingProvider([0.2, -0.3, 0.1, 0.77], "bt")
        rl = CountingProvider([0.8, -0.4, 0.2, 0.0], "rl")
        provider = ResidualInferenceActionProvider(
            bt,
            rl,
            residual_scale=0.125,
            gate_kind="aim",
        )

        result = provider.compute_action(
            context(self.own, state(0.0, 3000.0, 0.0), sim_time_s=0.0)
        )

        np.testing.assert_array_equal(result.action, bt.action)
        self.assertEqual(rl.calls, 0)
        self.assertEqual(provider.telemetry()["rl_inference_calls"], 0)

    def test_inference_residual_runs_bt_each_frame_and_holds_policy(self) -> None:
        bt = CountingProvider([0.2, -0.3, 0.1, 0.77], "bt")
        rl = CountingProvider([0.8, -0.4, 0.2, 0.0], "rl")
        provider = ResidualInferenceActionProvider(
            bt,
            rl,
            residual_scale=0.125,
            gate_kind="aim",
            rl_action_repeat=3,
        )

        results = [
            provider.compute_action(
                context(self.own, self.target, sim_time_s=0.0)
            )
            for _ in range(5)
        ]

        self.assertEqual(bt.calls, 5)
        self.assertEqual(rl.calls, 2)
        for result in results:
            np.testing.assert_allclose(
                result.action,
                [0.3, -0.35, 0.125, 0.77],
                atol=1e-6,
            )
        telemetry = provider.telemetry()
        self.assertEqual(telemetry["rl_correction_steps"], 5)
        self.assertEqual(telemetry["rl_action_repeat"], 3)
        self.assertAlmostEqual(telemetry["rl_inference_over_166_7ms_ratio"], 0.0)

    def test_saturation_aware_training_residual_respects_headroom(self) -> None:
        bt = CountingProvider([1.0, -1.0, 0.9, 0.77], "bt")
        provider = ResidualTrainingActionProvider(
            bt,
            residual_scale=0.125,
            gate_kind="aim",
            composition_mode="saturation_aware",
        )

        outward = provider.compute_action(
            context(
                self.own,
                self.target,
                residual=[1.0, -1.0, 1.0, -1.0],
                sim_time_s=0.0,
            )
        )
        np.testing.assert_allclose(
            outward.action,
            [1.0, -1.0, 0.9125, 0.77],
            atol=1e-6,
        )
        self.assertFalse(outward.info["action_clipped"])

        inward = provider.compute_action(
            context(
                self.own,
                self.target,
                residual=[-1.0, 1.0, -1.0, 1.0],
                sim_time_s=0.0,
            )
        )
        np.testing.assert_allclose(
            inward.action,
            [0.875, -0.875, 0.775, 0.77],
            atol=1e-6,
        )
        self.assertLessEqual(
            max(abs(value) for value in inward.info["applied_rl_correction"][:3]),
            0.125,
        )
        self.assertEqual(
            provider.telemetry()["residual_composition_mode"],
            "saturation_aware",
        )
        telemetry = provider.telemetry()
        np.testing.assert_allclose(
            telemetry["bt_surface_saturation_ratio_axis"],
            [1.0, 1.0, 0.0],
        )
        np.testing.assert_allclose(
            telemetry["requested_surface_correction_abs_mean_axis"],
            [0.125, 0.125, 0.125],
        )
        np.testing.assert_allclose(
            telemetry["applied_to_requested_ratio_mean_axis"],
            [0.5, 0.5, 0.55],
            atol=1e-6,
        )
        authority = inward.info["surface_authority"]
        np.testing.assert_allclose(
            authority["positive_headroom"], [0.0, 2.0, 0.1], atol=1e-7
        )
        np.testing.assert_allclose(
            authority["negative_headroom"], [2.0, 0.0, 1.9], atol=1e-7
        )
        np.testing.assert_allclose(authority["applied_to_requested_ratio"], [1.0, 1.0, 1.0])

    def test_btaware_prepare_is_consumed_once_by_residual_composition(self) -> None:
        bt = CountingProvider([0.25, -0.5, 0.75, 0.8], "bt")
        provider = ResidualTrainingActionProvider(
            bt,
            residual_scale=0.125,
            gate_kind="aim",
        )
        ctx = context(
            self.own,
            self.target,
            residual=np.zeros(4, dtype=np.float32),
            sim_time_s=0.0,
        )

        prepared = provider.prepare_bt_action(ctx)
        prepared_again = provider.prepare_bt_action(ctx)
        self.assertEqual(bt.calls, 1)
        np.testing.assert_array_equal(prepared, prepared_again)
        np.testing.assert_array_equal(provider.prepared_bt_action, prepared)

        result = provider.compute_action(ctx)
        self.assertEqual(bt.calls, 1)
        self.assertIsNone(provider.prepared_bt_action)
        np.testing.assert_array_equal(result.info["bt_action"], prepared)
        np.testing.assert_array_equal(result.action, prepared)

        provider.compute_action(ctx)
        self.assertEqual(bt.calls, 2)

    def test_btaware_reset_discards_unconsumed_cache(self) -> None:
        bt = CountingProvider([0.1, 0.2, 0.3, 0.9], "bt")
        provider = ResidualTrainingActionProvider(
            bt,
            residual_scale=0.125,
            gate_kind="aim",
        )
        ctx = context(self.own, self.target, sim_time_s=0.0)

        provider.prepare_bt_action(ctx)
        self.assertIsNotNone(provider.prepared_bt_action)
        provider.reset(ctx)
        self.assertIsNone(provider.prepared_bt_action)


if __name__ == "__main__":
    unittest.main()
