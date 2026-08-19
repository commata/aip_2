from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from time import sleep

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.ai.guidance_selector import (
    GUIDANCE_ACTIONS,
    GUIDANCE_SELECTOR_CONTRACT_VERSION,
    GUIDANCE_SELECTOR_FEATURES,
    GUIDANCE_SELECTOR_OBSERVATION_SIZE,
    FixedGuidanceSelector,
    GuidanceRuntimeConfig,
    GuidanceSelectorActionProvider,
    GuidanceSetpoint,
    NumpyMLPGuidanceSelector,
    build_guidance_observation,
    canonicalize_guidance_observation,
    compose_guidance_setpoint,
    guidance_action_delta,
    guidance_to_surface_action,
    mirror_guidance_action,
    mirror_guidance_observation,
)
from dogfight.sim.state_schema import StateIndex


class FakeBT(ActionProvider):
    def __init__(self, action=(0.2, -0.1, 0.3, 0.7)):
        self.action = np.asarray(action, dtype=np.float32)
        self.calls = 0

    def compute_action(self, context: ActionContext) -> ActionResult:
        self.calls += 1
        return ActionResult(
            self.action.copy(),
            "fake_bt",
            info={
                "vp": np.array([1000.0, 0.0, 5000.0], dtype=np.float32),
                "throttle_control": {"target_speed_mps": 230.0},
            },
        )


class FakeGate:
    def __init__(self, active: bool):
        self.active = active

    def reset(self):
        return None

    def update(self, *args, **kwargs):
        return {
            "active": self.active,
            "entry": self.active,
            "exit": False,
            "safety_veto": False,
        }

    def telemetry(self):
        return {"rear120_activation_gate_active_ratio": float(self.active)}


class FailingSelector:
    def predict(self, observation):
        raise RuntimeError("expected")


class InvalidProbabilitySelector:
    def predict(self, observation):
        return 1, 0.9, np.ones(8, dtype=np.float32)


class SlowSelector:
    def predict(self, observation):
        sleep(0.002)
        probabilities = np.zeros(9, dtype=np.float32)
        probabilities[1] = 1.0
        return 1, 1.0, probabilities


def state(n=0.0, e=0.0, altitude=5000.0, yaw=0.0, speed=230.0):
    value = np.zeros(46, dtype=np.float64)
    value[StateIndex.N] = n
    value[StateIndex.E] = e
    value[StateIndex.D] = -altitude
    value[StateIndex.YAW] = yaw
    value[StateIndex.KCAS] = speed
    value[StateIndex.ALT] = altitude
    value[StateIndex.HEALTH] = 1.0
    value[6] = speed
    return value


def context():
    return ActionContext(
        sim=None,
        opponent_sim=None,
        ownship_state=state(),
        target_state=state(n=1100.0, e=100.0, yaw=180.0, speed=225.0),
        observation=np.zeros(16, dtype=np.float32),
        info={"sim_time_s": 10.0},
    )


def normalize_for_action(action_id: int) -> float:
    return (float(action_id) - 4.0) / 4.0


class GuidanceActionLibraryTests(unittest.TestCase):
    def test_action_library_has_frozen_nine_classes(self):
        self.assertEqual(len(GUIDANCE_ACTIONS), 9)
        self.assertEqual(GUIDANCE_ACTIONS[0], "BT_DEFAULT")
        self.assertEqual(guidance_action_delta(0), {
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
            "range_m": 0.0,
            "target_speed_m_s": 0.0,
        })

    def test_each_nondefault_changes_one_guidance_dimension(self):
        base = GuidanceSetpoint(2.0, -1.0, 1000.0, 230.0)
        for action_id in range(1, len(GUIDANCE_ACTIONS)):
            corrected = compose_guidance_setpoint(base, action_id)
            differences = np.asarray(
                [
                    corrected.local_azimuth_deg - base.local_azimuth_deg,
                    corrected.local_elevation_deg - base.local_elevation_deg,
                    corrected.distance_m - base.distance_m,
                    corrected.target_speed_m_s - base.target_speed_m_s,
                ]
            )
            self.assertEqual(int(np.count_nonzero(differences)), 1)

    def test_guidance_controller_preserves_throttle_and_bounds_surfaces(self):
        bt = np.array([0.99, -0.99, 0.99, 0.61], dtype=np.float32)
        base = GuidanceSetpoint(0.0, 0.0, 1000.0, 230.0)
        for action_id in range(len(GUIDANCE_ACTIONS)):
            final, diagnostics = guidance_to_surface_action(
                bt, base, compose_guidance_setpoint(base, action_id)
            )
            self.assertEqual(float(final[3]), float(bt[3]))
            self.assertTrue(np.all(final[:3] <= 1.0))
            self.assertTrue(np.all(final[:3] >= -1.0))
            self.assertTrue(
                np.all(np.abs(diagnostics["requested_surface_correction"]) <= 0.080001)
            )

    def test_double_mirror_identity(self):
        for axis in ("lateral", "vertical"):
            for action_id in range(len(GUIDANCE_ACTIONS)):
                self.assertEqual(
                    mirror_guidance_action(
                        mirror_guidance_action(action_id, axis), axis
                    ),
                    action_id,
                )

    def test_observation_double_mirror_and_canonical_round_trip(self):
        observation = np.linspace(-0.9, 0.9, 45, dtype=np.float32)
        observation[40] = normalize_for_action(3)
        for axis in ("lateral", "vertical"):
            self.assertTrue(
                np.array_equal(
                    mirror_guidance_observation(
                        mirror_guidance_observation(observation, axis), axis
                    ),
                    observation,
                )
            )
        canonical = canonicalize_guidance_observation(
            observation, lateral_sign=-1, vertical_sign=-1
        )
        world = mirror_guidance_observation(
            mirror_guidance_observation(canonical, "vertical"), "lateral"
        )
        self.assertTrue(np.array_equal(world, observation))


class GuidanceObservationTests(unittest.TestCase):
    def test_observation_contract_is_finite_and_ordered(self):
        base = GuidanceSetpoint(1.0, -2.0, 1000.0, 230.0)
        observation = build_guidance_observation(
            np.zeros(16, dtype=np.float32),
            state(),
            state(n=1100.0, e=100.0, yaw=180.0, speed=225.0),
            np.array([0.2, -0.1, 0.3, 0.7], dtype=np.float32),
            base,
            sim_time_s=10.0,
            previous_action_id=0,
            action_hold_frames=0,
            gate_elapsed_frames=1,
            gate_active=True,
            minimum_action_hold_frames=18,
            maximum_active_frames=90,
        )
        self.assertEqual(GUIDANCE_SELECTOR_OBSERVATION_SIZE, 45)
        self.assertEqual(len(GUIDANCE_SELECTOR_FEATURES), 45)
        self.assertEqual(observation.shape, (45,))
        self.assertTrue(np.all(np.isfinite(observation)))
        self.assertTrue(np.all(np.abs(observation) <= 1.0))

    def test_training_and_runtime_builder_are_exactly_identical(self):
        kwargs = dict(
            sim_time_s=12.0,
            previous_action_id=2,
            action_hold_frames=7,
            gate_elapsed_frames=11,
            gate_active=True,
            minimum_action_hold_frames=18,
            maximum_active_frames=90,
        )
        args = (
            np.linspace(-1.0, 1.0, 16, dtype=np.float32),
            state(),
            state(n=900.0, e=-150.0, yaw=170.0, speed=220.0),
            np.array([0.1, -0.2, 0.3, 0.8], dtype=np.float32),
            GuidanceSetpoint(2.0, -1.0, 950.0, 230.0),
        )
        self.assertTrue(
            np.array_equal(
                build_guidance_observation(*args, **kwargs),
                build_guidance_observation(*args, **kwargs),
            )
        )


class GuidanceProviderTests(unittest.TestCase):
    def provider(self, selector, active=True):
        provider = GuidanceSelectorActionProvider(
            FakeBT(),
            selector,
            runtime_config=GuidanceRuntimeConfig(confidence_threshold=0.65),
        )
        provider.gate = FakeGate(active)
        return provider

    def test_gate_off_returns_exact_bt_and_skips_selector(self):
        selector = FixedGuidanceSelector("VP_AZ_POS_SMALL")
        provider = self.provider(selector, active=False)
        result = provider.compute_action(context())
        self.assertTrue(np.array_equal(result.action, provider.bt_provider.action))
        self.assertEqual(provider.telemetry()["selector_inference_calls"], 0)

    def test_bt_default_returns_exact_bt(self):
        provider = self.provider(FixedGuidanceSelector("BT_DEFAULT"))
        result = provider.compute_action(context())
        self.assertTrue(np.array_equal(result.action, provider.bt_provider.action))
        self.assertEqual(result.action[3], provider.bt_provider.action[3])

    def test_e2e_latency_covers_every_action_frame(self):
        provider = self.provider(FixedGuidanceSelector("BT_DEFAULT"))
        provider.compute_action(context())
        provider.compute_action(context())
        telemetry = provider.telemetry()
        self.assertEqual(telemetry["e2e_ai_latency_samples"], 2)
        self.assertGreaterEqual(telemetry["e2e_ai_latency_ms_p50"], 0.0)
        self.assertGreaterEqual(telemetry["e2e_ai_latency_ms_p95"], 0.0)
        self.assertGreaterEqual(telemetry["e2e_ai_latency_ms_p99"], 0.0)
        self.assertGreaterEqual(telemetry["e2e_ai_latency_ms_max"], 0.0)
        self.assertEqual(telemetry["e2e_ai_latency_over_166_7ms"], 0)

    def test_nondefault_applies_guidance_with_bt_throttle(self):
        provider = self.provider(FixedGuidanceSelector("VP_AZ_POS_SMALL"))
        result = provider.compute_action(context())
        self.assertFalse(np.array_equal(result.action[:3], provider.bt_provider.action[:3]))
        self.assertEqual(result.action[3], provider.bt_provider.action[3])
        self.assertEqual(provider.telemetry()["nonzero_intervention_frames"], 1)
        snapshot = provider.telemetry()["first_nondefault_selector_snapshot"]
        self.assertEqual(snapshot["selected_action"], "VP_AZ_POS_SMALL")
        self.assertEqual(len(snapshot["observation"]), 45)
        self.assertEqual(len(snapshot["ownship_server_state"]), 7)
        self.assertEqual(len(snapshot["target_server_state"]), 7)
        self.assertEqual(
            provider.telemetry()["selector_decision_trace"][0]["selected_action"],
            "VP_AZ_POS_SMALL",
        )
        self.assertEqual(
            len(provider.telemetry()["selector_decision_trace"][0]["ownship_server_state"]),
            7,
        )

    def test_shadow_nondefault_predicts_but_returns_exact_bt(self):
        provider = GuidanceSelectorActionProvider(
            FakeBT(),
            FixedGuidanceSelector("VP_AZ_POS_SMALL"),
            runtime_config=GuidanceRuntimeConfig(
                confidence_threshold=0.65,
                shadow_mode=True,
            ),
        )
        provider.gate = FakeGate(True)
        result = provider.compute_action(context())
        self.assertTrue(np.array_equal(result.action, provider.bt_provider.action))
        self.assertEqual(result.info["selected_action"], "VP_AZ_POS_SMALL")
        self.assertTrue(result.info["shadow_command_exact_bt"])
        telemetry = provider.telemetry()
        self.assertTrue(telemetry["shadow_mode"])
        self.assertEqual(telemetry["nonzero_intervention_frames"], 0)
        self.assertEqual(
            telemetry["first_nondefault_selector_snapshot"]["selected_action"],
            "VP_AZ_POS_SMALL",
        )

    def test_exception_falls_back_to_exact_bt(self):
        provider = self.provider(FailingSelector())
        result = provider.compute_action(context())
        self.assertTrue(np.array_equal(result.action, provider.bt_provider.action))
        self.assertEqual(
            provider.telemetry()["fallback_counts"]["inference_RuntimeError"], 1
        )

    def test_confidence_fallback_is_cached_and_exact(self):
        provider = self.provider(FixedGuidanceSelector("VP_EL_POS_SMALL", confidence=0.5))
        first = provider.compute_action(context())
        second = provider.compute_action(context())
        self.assertTrue(np.array_equal(first.action, provider.bt_provider.action))
        self.assertTrue(np.array_equal(second.action, provider.bt_provider.action))
        self.assertEqual(provider.telemetry()["selector_inference_calls"], 1)

    def test_invalid_probability_shape_falls_back(self):
        provider = self.provider(InvalidProbabilitySelector())
        result = provider.compute_action(context())
        self.assertTrue(np.array_equal(result.action, provider.bt_provider.action))
        self.assertEqual(provider.telemetry()["fallback_counts"]["invalid_probabilities"], 1)

    def test_timeout_falls_back_to_exact_bt(self):
        provider = GuidanceSelectorActionProvider(
            FakeBT(),
            SlowSelector(),
            runtime_config=GuidanceRuntimeConfig(
                confidence_threshold=0.65,
                inference_timeout_s=0.0001,
            ),
        )
        provider.gate = FakeGate(True)
        result = provider.compute_action(context())
        self.assertTrue(np.array_equal(result.action, provider.bt_provider.action))
        self.assertEqual(provider.telemetry()["fallback_counts"]["inference_timeout"], 1)


class GuidanceBundleTests(unittest.TestCase):
    def test_numpy_bundle_load_and_hash_contract(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            weights = root / "model.npz"
            rng = np.random.default_rng(7)
            np.savez(
                weights,
                w1=rng.normal(size=(45, 8)).astype(np.float32),
                b1=np.zeros(8, dtype=np.float32),
                w2=rng.normal(size=(8, 8)).astype(np.float32),
                b2=np.zeros(8, dtype=np.float32),
                w3=rng.normal(size=(8, 9)).astype(np.float32),
                b3=np.zeros(9, dtype=np.float32),
            )
            metadata = {
                "model_kind": "numpy_mlp_categorical",
                "observation_contract": GUIDANCE_SELECTOR_CONTRACT_VERSION,
                "observation_size": GUIDANCE_SELECTOR_OBSERVATION_SIZE,
                "features": list(GUIDANCE_SELECTOR_FEATURES),
                "actions": list(GUIDANCE_ACTIONS),
                "model_sha256": hashlib.sha256(weights.read_bytes()).hexdigest().upper(),
            }
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            selector = NumpyMLPGuidanceSelector(root)
            action_id, confidence, probabilities = selector.predict(
                np.zeros(45, dtype=np.float32)
            )
            self.assertIn(action_id, range(9))
            self.assertGreaterEqual(confidence, 0.0)
            self.assertAlmostEqual(float(np.sum(probabilities)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
