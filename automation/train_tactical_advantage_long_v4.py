from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import automation.train_state_action_advantage_v3 as core
import automation.train_tactical_advantage_v4 as m1
from dogfight.ai.tactical_advantage import TACTICAL_ACTION_FEATURES, encode_tactical_option
from dogfight.ai.temporal_observation import (
    LONG_TEMPORAL_FEATURES,
    LONG_TEMPORAL_OBSERVATION_SIZE,
    LONG_TEMPORAL_SERVER_CONTRACT_VERSION,
)


LONG_MODEL_KIND = "numpy_temporal_tactical_long120_distributional_advantage"
LONG_MODEL_INPUT_SIZE = LONG_TEMPORAL_OBSERVATION_SIZE + len(TACTICAL_ACTION_FEATURES)
LONG_SEEDS = (41021, 41022, 41023)


class LongTacticalAdvantageNetwork(core.AdvantageNetwork):
    def __init__(self) -> None:
        super().__init__(input_size=LONG_MODEL_INPUT_SIZE, hidden_size=64)


def encode_long_input(row: dict[str, Any]) -> np.ndarray:
    state = np.asarray(row["server_observation"], dtype=np.float32)
    vector = np.concatenate(
        (state, encode_tactical_option(row["mode"], int(row["hold_frames"])))
    ).astype(np.float32)
    if vector.shape != (LONG_MODEL_INPUT_SIZE,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"expected finite {LONG_MODEL_INPUT_SIZE}D long temporal input")
    return vector


def configure_core(target_scale: float) -> None:
    core.AdvantageNetwork = LongTacticalAdvantageNetwork
    core.encode_factorized_input = encode_long_input
    core.assign_group_folds = m1.assign_scenario_folds
    core.policy_value = m1.tactical_policy_value
    core.POSITIVE_EPSILON = m1.POSITIVE_EPSILON
    core.LARGE_REGRESSION = m1.LARGE_REGRESSION
    core.TARGET_SCALE = target_scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Train long-history Temporal Tactical M1 v4")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=240)
    args = parser.parse_args()
    dataset_path = args.dataset.resolve()
    dataset_metadata = json.loads(
        (dataset_path.parent / "metadata.json").read_text(encoding="utf-8")
    )
    if dataset_metadata.get("observation_contract") != LONG_TEMPORAL_SERVER_CONTRACT_VERSION:
        raise ValueError("long-history trainer requires the long120 observation contract")
    rows = m1.transform_rows(m1.load_dataset(dataset_path))
    nondefault = np.asarray(
        [row["damage_delta"] for row in rows if row["action"] != "BT_DEFAULT"],
        dtype=np.float64,
    )
    target_scale = max(1e-4, float(np.quantile(np.abs(nondefault), 0.90)))
    configure_core(target_scale)
    result, models, support = core.train_and_evaluate(
        rows, folds=args.folds, epochs=args.epochs, seeds=LONG_SEEDS
    )
    result["seeds"] = list(LONG_SEEDS)
    result["epochs_per_model"] = args.epochs
    result["optimizer_updates"] = args.epochs * len(LONG_SEEDS) * (args.folds + 1)
    metadata = m1.save_bundle(
        args.output.resolve(),
        models,
        result,
        support,
        dataset_path=dataset_path,
        dataset_metadata=dataset_metadata,
        target_scale=target_scale,
    )
    metadata.update(
        {
            "schema_version": "temporal_tactical_advantage_v4.long120.v1",
            "model_kind": LONG_MODEL_KIND,
            "input_contract": "127D long120 temporal server observation + 5D Tactical option",
            "input_size": LONG_MODEL_INPUT_SIZE,
            "observation_contract": LONG_TEMPORAL_SERVER_CONTRACT_VERSION,
            "observation_size": LONG_TEMPORAL_OBSERVATION_SIZE,
            "features": list(LONG_TEMPORAL_FEATURES),
            "model_architecture": "long120 MLP 132-64-64 distributional ensemble",
        }
    )
    (args.output.resolve() / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "offline_gate_passed": metadata["offline_gate_passed"],
                "selected_oof_policy": metadata["selected_oof_policy"],
                "model_sha256": metadata["model_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
