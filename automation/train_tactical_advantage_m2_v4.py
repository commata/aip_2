from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import automation.train_state_action_advantage_v3 as core
import automation.train_tactical_advantage_v4 as m1
from dogfight.ai.tactical_advantage import TACTICAL_ACTION_FEATURES, encode_tactical_option
from dogfight.ai.tactical_modes import TACTICAL_HOLD_FRAMES, TACTICAL_MODES_T1
from dogfight.ai.temporal_observation import (
    TEMPORAL_DELTA_SOURCE_INDICES,
    TEMPORAL_DELTA_SOURCE_FEATURES,
    TEMPORAL_FEATURES,
    TEMPORAL_OBSERVATION_SIZE,
    TEMPORAL_SERVER_CONTRACT_VERSION,
)


M2_SEEDS = (41011, 41012, 41013)
SEQUENCE_STEPS = 4
SEQUENCE_FEATURES = len(TEMPORAL_DELTA_SOURCE_INDICES)
STATIC_ACTION_SIZE = 42 + len(TACTICAL_ACTION_FEATURES)
M2_INPUT_SIZE = SEQUENCE_STEPS * SEQUENCE_FEATURES + STATIC_ACTION_SIZE
M2_MODEL_KIND = "numpy_temporal_tactical_gru_distributional_advantage"


def encode_m2_input(row: dict[str, Any]) -> np.ndarray:
    observation = np.asarray(row["server_observation"], dtype=np.float32)
    if observation.shape != (TEMPORAL_OBSERVATION_SIZE,):
        raise ValueError("M2 requires a 93D Temporal Tactical observation")
    indices = np.asarray(TEMPORAL_DELTA_SOURCE_INDICES)
    current = observation[indices]
    delta6 = observation[42 : 42 + SEQUENCE_FEATURES]
    delta12 = observation[42 + SEQUENCE_FEATURES : 42 + 2 * SEQUENCE_FEATURES]
    delta30 = observation[42 + 2 * SEQUENCE_FEATURES : 42 + 3 * SEQUENCE_FEATURES]
    # Temporal contract stores (current-lagged)/2. Reconstruct the four
    # normalized snapshots without introducing any offline-only inputs.
    sequence = np.stack(
        (current - 2.0 * delta30, current - 2.0 * delta12, current - 2.0 * delta6, current)
    )
    static_action = np.concatenate(
        (observation[:42], encode_tactical_option(row["mode"], int(row["hold_frames"])))
    )
    vector = np.concatenate((sequence.reshape(-1), static_action)).astype(np.float32)
    if vector.shape != (M2_INPUT_SIZE,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"expected finite {M2_INPUT_SIZE}D GRU input")
    return vector


class TacticalGRUAdvantageNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(SEQUENCE_FEATURES, 32, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(32 + STATIC_ACTION_SIZE, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(64, 1)
        self.positive_head = nn.Linear(64, 1)
        self.regression_head = nn.Linear(64, 1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = values[:, : SEQUENCE_STEPS * SEQUENCE_FEATURES].reshape(
            -1, SEQUENCE_STEPS, SEQUENCE_FEATURES
        )
        static_action = values[:, SEQUENCE_STEPS * SEQUENCE_FEATURES :]
        _, hidden = self.gru(sequence)
        fused = self.fusion(torch.cat((hidden[-1], static_action), dim=1))
        return (
            self.mean_head(fused).squeeze(-1),
            self.positive_head(fused).squeeze(-1),
            self.regression_head(fused).squeeze(-1),
        )


def configure_core(target_scale: float) -> None:
    core.AdvantageNetwork = TacticalGRUAdvantageNetwork
    core.encode_factorized_input = encode_m2_input
    core.assign_group_folds = m1.assign_scenario_folds
    core.policy_value = m1.tactical_policy_value
    core.POSITIVE_EPSILON = m1.POSITIVE_EPSILON
    core.LARGE_REGRESSION = m1.LARGE_REGRESSION
    core.TARGET_SCALE = target_scale


def save_bundle(
    output: Path,
    models,
    result: dict[str, Any],
    support: dict[str, np.ndarray],
    *,
    dataset_path: Path,
    dataset_metadata: dict[str, Any],
    target_scale: float,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Tactical M2 bundle: {output}")
    output.mkdir(parents=True)
    arrays: dict[str, np.ndarray] = {}
    for index, (model, normalization) in enumerate(models):
        prefix = f"model_{index}_"
        arrays[prefix + "input_mean"] = normalization["mean"]
        arrays[prefix + "input_scale"] = normalization["scale"]
        for name, value in model.state_dict().items():
            arrays[prefix + name] = value.detach().numpy()
    for name, value in support.items():
        arrays[f"support_{name}"] = value
    model_path = output / "model.npz"
    np.savez_compressed(model_path, **arrays)
    selected = result["selected_oof_policy"]
    metadata = {
        **result,
        "schema_version": "temporal_tactical_advantage_v4.m2.v1",
        "model_kind": M2_MODEL_KIND,
        "input_contract": "4x17 reconstructed temporal snapshots + 42D current + 5D option",
        "input_size": M2_INPUT_SIZE,
        "observation_contract": TEMPORAL_SERVER_CONTRACT_VERSION,
        "observation_size": TEMPORAL_OBSERVATION_SIZE,
        "features": list(TEMPORAL_FEATURES),
        "sequence_features": list(TEMPORAL_DELTA_SOURCE_FEATURES),
        "action_features": list(TACTICAL_ACTION_FEATURES),
        "candidate_modes": list(TACTICAL_MODES_T1),
        "candidate_hold_frames": [0, *TACTICAL_HOLD_FRAMES],
        "model_architecture": "M2 GRU 4x17-32 + current42/action5 -> 64-64 distributional heads",
        "target_scale": target_scale,
        "offline_gate_passed": bool(selected["offline_gate_passed"]),
        "runtime_threshold": dict(selected["threshold"]),
        "runtime_candidates": [
            {"mode": mode, "hold_frames": hold}
            for mode in TACTICAL_MODES_T1[1:]
            for hold in TACTICAL_HOLD_FRAMES
        ],
        "ensemble_size": len(models),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest().upper(),
        "dataset_unique_events": int(dataset_metadata["unique_events"]),
        "dataset_state_action_pairs": int(dataset_metadata["state_action_pairs"]),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest().upper(),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Temporal Tactical Advantage M2 GRU v4")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=240)
    args = parser.parse_args()
    dataset_path = args.dataset.resolve()
    dataset_metadata = json.loads(
        (dataset_path.parent / "metadata.json").read_text(encoding="utf-8")
    )
    rows = m1.transform_rows(m1.load_dataset(dataset_path))
    nondefault = np.asarray(
        [row["damage_delta"] for row in rows if row["action"] != "BT_DEFAULT"],
        dtype=np.float64,
    )
    target_scale = max(1e-4, float(np.quantile(np.abs(nondefault), 0.90)))
    configure_core(target_scale)
    result, models, support = core.train_and_evaluate(
        rows, folds=args.folds, epochs=args.epochs, seeds=M2_SEEDS
    )
    result["seeds"] = list(M2_SEEDS)
    result["epochs_per_model"] = args.epochs
    result["optimizer_updates"] = args.epochs * len(M2_SEEDS) * (args.folds + 1)
    metadata = save_bundle(
        args.output.resolve(),
        models,
        result,
        support,
        dataset_path=dataset_path,
        dataset_metadata=dataset_metadata,
        target_scale=target_scale,
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
