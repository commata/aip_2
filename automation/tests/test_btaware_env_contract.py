from __future__ import annotations

import unittest

import numpy as np

from dogfight.ai.action_provider import ActionProvider, ActionResult
from dogfight.ai.hybrid_action_provider import ResidualTrainingActionProvider
from dogfight.envs.single_agent_env import DogFightEnv
from dogfight.sim.state_schema import StateIndex


class CountingBT(ActionProvider):
    def __init__(self, action) -> None:
        self.action = np.asarray(action, dtype=np.float32)
        self.calls = 0

    def compute_action(self, context) -> ActionResult:
        self.calls += 1
        return ActionResult(self.action.copy(), "counting_bt")


class DummySim:
    def __init__(self) -> None:
        self.stepped_actions = []

    def step(self, action) -> None:
        self.stepped_actions.append(np.asarray(action, dtype=np.float32).copy())


def state(n: float, e: float, yaw: float, sim_time: float) -> np.ndarray:
    value = np.zeros(51, dtype=np.float32)
    value[StateIndex.N] = n
    value[StateIndex.E] = e
    value[StateIndex.D] = -5000.0
    value[StateIndex.YAW] = yaw
    value[6] = 230.0
    value[StateIndex.KCAS] = 230.0
    value[StateIndex.ALT] = 5000.0
    value[StateIndex.SIM_TIME] = sim_time
    return value


class BTAwareEnvironmentContractTests(unittest.TestCase):
    def build_env(self):
        bt = CountingBT([0.25, -0.5, 0.75, 0.8])
        provider = ResidualTrainingActionProvider(
            bt,
            residual_scale=0.125,
            gate_kind="aim",
        )
        env = DogFightEnv.__new__(DogFightEnv)
        env._observation_mode = "aim_residual13_btaware"
        env._observation_fn = None
        env.num_observation = 13
        env._geo_info = None
        env._wez = {}
        env._sim = DummySim()
        env._target_sim = DummySim()
        env._ownship_state = state(0.0, 0.0, 0.0, 1.0)
        env._target_state = state(1000.0, 100.0, 180.0, 1.0)
        env._ownship_action_provider = provider
        env._target_action_provider = None
        env.pre_obs = np.zeros(13, dtype=np.float32)
        env.current_timestep = 60
        env._last_ownship_action_info = {}
        return env, bt, provider

    def test_observation_and_composition_share_one_bt_tick(self) -> None:
        env, bt, provider = self.build_env()

        observation = env.get_observation()
        self.assertEqual(bt.calls, 1)
        np.testing.assert_array_equal(observation[-3:], bt.action[:3])
        np.testing.assert_array_equal(provider.prepared_bt_action, bt.action)

        env._step_controlled_aircraft(np.zeros(4, dtype=np.float32))
        self.assertEqual(bt.calls, 1)
        self.assertIsNone(provider.prepared_bt_action)
        np.testing.assert_array_equal(env._sim.stepped_actions[0], bt.action)
        np.testing.assert_array_equal(
            env._last_ownship_action_info["bt_action"],
            observation[-3:].tolist() + [float(bt.action[3])],
        )

    def test_terminal_observation_does_not_tick_stateful_bt(self) -> None:
        env, bt, provider = self.build_env()

        terminal_observation = env._build_terminal_observation()

        self.assertEqual(bt.calls, 0)
        self.assertIsNone(provider.prepared_bt_action)
        np.testing.assert_array_equal(terminal_observation[-3:], np.zeros(3))


if __name__ == "__main__":
    unittest.main()
