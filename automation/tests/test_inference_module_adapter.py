from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import unittest
from unittest.mock import patch

from dogfight.ai.rl_action_provider import RLActionProvider
from dogfight.ai.rllib_utils import RLModuleInferenceAdapter


class _FakeModule:
    def __init__(self) -> None:
        self.state = None

    def set_state(self, state) -> None:
        self.state = state


class InferenceModuleAdapterTests(unittest.TestCase):
    def test_adapter_accepts_bundle_weights_without_algorithm_workers(self) -> None:
        module = _FakeModule()
        adapter = RLModuleInferenceAdapter(module, "default_policy")
        weights = OrderedDict({"actor.weight": 1.0})

        with patch(
            "dogfight.ai.rl_action_provider.load_lightweight_policy_bundle",
            return_value=({"policy_id": "default_policy"}, weights),
        ):
            provider = RLActionProvider(
                bundle_dir=str(Path("unused")),
                algorithm_factory=lambda metadata: adapter,
            )

        self.assertIs(module.state, weights)
        self.assertIs(adapter.get_module("default_policy"), module)
        self.assertIsNone(adapter.get_module("other_policy"))
        provider.close()


if __name__ == "__main__":
    unittest.main()
