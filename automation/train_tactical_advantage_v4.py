from __future__ import annotations

import argparse
import hashlib
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
from dogfight.ai.tactical_advantage import (
    TACTICAL_ACTION_FEATURES,
    TACTICAL_MODEL_INPUT_SIZE,
    TACTICAL_MODEL_KIND,
    encode_tactical_option,
)
from dogfight.ai.tactical_modes import TACTICAL_HOLD_FRAMES, TACTICAL_MODES_T1
from dogfight.ai.temporal_observation import (
    TEMPORAL_FEATURES,
    TEMPORAL_OBSERVATION_SIZE,
    TEMPORAL_SERVER_CONTRACT_VERSION,
)


ENSEMBLE_SEEDS = (41001, 41002, 41003)
POSITIVE_EPSILON = 1e-9
LARGE_REGRESSION = -1e-6


class TacticalAdvantageNetwork(core.AdvantageNetwork):
    def __init__(self, input_size: int = TACTICAL_MODEL_INPUT_SIZE, hidden_size: int = 64):
        super().__init__(input_size=input_size, hidden_size=hidden_size)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("Tactical dataset is empty")
    return rows


def transform_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transformed = []
    for row in rows:
        transformed.append(
            {
                "action": row["mode"],
                "state_hash": row["event_id"],
                "candidate_id": row["option_id"],
                "damage_delta": float(row["damage_advantage"]),
                "family": f"{row['opponent_id']}|{row['scenario_id']}",
                "server_observation": row["observation"],
                "mode": row["mode"],
                "hold_frames": int(row["hold_frames"]),
                "scenario_id": row["scenario_id"],
                "opponent_id": row["opponent_id"],
                "fight_id": row["fight_id"],
                "seed": int(row["seed"]),
            }
        )
    return transformed


def encode_input(row: dict[str, Any]) -> np.ndarray:
    state = np.asarray(row["server_observation"], dtype=np.float32)
    vector = np.concatenate(
        (state, encode_tactical_option(row["mode"], int(row["hold_frames"])))
    ).astype(np.float32)
    if vector.shape != (TACTICAL_MODEL_INPUT_SIZE,) or not np.all(np.isfinite(vector)):
        raise ValueError("expected finite 93D temporal state + 5D Tactical option")
    return vector


def assign_scenario_folds(
    rows: list[dict[str, Any]], folds: int = 5
) -> dict[str, int]:
    families = sorted(
        {row["family"] for row in rows},
        key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
    )
    if len(families) < folds:
        raise ValueError("scenario-group OOF requires at least one family per fold")
    family_fold = {family: index % folds for index, family in enumerate(families)}
    assignment = {}
    for row in rows:
        fold = family_fold[row["family"]]
        previous = assignment.setdefault(row["state_hash"], fold)
        if previous != fold:
            raise ValueError("event leaked across scenario folds")
    return assignment


def configure_core(target_scale: float) -> None:
    core.AdvantageNetwork = TacticalAdvantageNetwork
    core.encode_factorized_input = encode_input
    core.assign_group_folds = assign_scenario_folds
    core.POSITIVE_EPSILON = POSITIVE_EPSILON
    core.LARGE_REGRESSION = LARGE_REGRESSION
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
        raise FileExistsError(f"refusing to overwrite Tactical model bundle: {output}")
    output.mkdir(parents=True)
    arrays: dict[str, np.ndarray] = {}
    for index, (model, normalization) in enumerate(models):
        prefix = f"model_{index}_"
        arrays[prefix + "input_mean"] = normalization["mean"]
        arrays[prefix + "input_scale"] = normalization["scale"]
        for name, value in model.state_dict().items():
            arrays[prefix + name] = value.detach().numpy()
    arrays["support_examples"] = support["examples"]
    arrays["support_mean"] = support["mean"]
    arrays["support_scale"] = support["scale"]
    model_path = output / "model.npz"
    np.savez_compressed(model_path, **arrays)
    selected = result["selected_oof_policy"]
    metadata = {
        **result,
        "schema_version": "temporal_tactical_advantage_v4.v1",
        "model_kind": TACTICAL_MODEL_KIND,
        "input_contract": "93D temporal server observation + 5D Tactical option",
        "input_size": TACTICAL_MODEL_INPUT_SIZE,
        "observation_contract": TEMPORAL_SERVER_CONTRACT_VERSION,
        "observation_size": TEMPORAL_OBSERVATION_SIZE,
        "features": list(TEMPORAL_FEATURES),
        "action_features": list(TACTICAL_ACTION_FEATURES),
        "candidate_modes": list(TACTICAL_MODES_T1),
        "candidate_hold_frames": [0, *TACTICAL_HOLD_FRAMES],
        "model_architecture": "M1 MLP 98-64-64 ensemble; mean/Ppositive/Pregression heads",
        "distributional_outputs": [
            "mean_advantage",
            "P(advantage>epsilon)",
            "P(large_regression)",
            "ensemble_q10",
            "ensemble_q50",
        ],
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Temporal Tactical Advantage M1 v4")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    dataset_metadata = json.loads(
        (dataset_path.parent / "metadata.json").read_text(encoding="utf-8")
    )
    rows = transform_rows(load_dataset(dataset_path))
    nondefault_damage = np.asarray(
        [row["damage_delta"] for row in rows if row["action"] != "BT_DEFAULT"],
        dtype=np.float64,
    )
    target_scale = max(1e-4, float(np.quantile(np.abs(nondefault_damage), 0.90)))
    configure_core(target_scale)
    result, models, support = core.train_and_evaluate(
        rows,
        folds=args.folds,
        epochs=args.epochs,
        seeds=ENSEMBLE_SEEDS,
    )
    result["seeds"] = list(ENSEMBLE_SEEDS)
    result["epochs_per_model"] = args.epochs
    result["optimizer_updates"] = args.epochs * len(ENSEMBLE_SEEDS) * (args.folds + 1)
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
