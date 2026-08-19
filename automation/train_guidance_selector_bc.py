from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dogfight.ai.guidance_selector import (
    GUIDANCE_ACTIONS,
    GUIDANCE_SELECTOR_CONTRACT_VERSION,
    GUIDANCE_SELECTOR_FEATURES,
    GUIDANCE_SELECTOR_NORMALIZATION_VERSION,
    GUIDANCE_SELECTOR_OBSERVATION_SIZE,
    NumpyMLPGuidanceSelector,
    mirror_guidance_action,
    mirror_guidance_observation,
)


BC_SEEDS = (8701, 8702, 8703)


class SelectorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(GUIDANCE_SELECTOR_OBSERVATION_SIZE, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, len(GUIDANCE_ACTIONS)),
        )

    def forward(self, value):
        return self.network(value)


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    if len(samples) < 100:
        raise ValueError(f"frozen BC requires 100 counterfactual states, got {len(samples)}")
    observations = []
    labels = []
    for sample in samples:
        observation = np.asarray(sample.get("observation"), dtype=np.float32)
        if observation.shape != (GUIDANCE_SELECTOR_OBSERVATION_SIZE,) or not np.all(
            np.isfinite(observation)
        ):
            raise ValueError(f"invalid observation for {sample.get('case_id')}")
        label = int(sample.get("label_id", -1))
        if not 0 <= label < len(GUIDANCE_ACTIONS):
            raise ValueError(f"invalid label for {sample.get('case_id')}")
        observations.append(observation)
        labels.append(label)
    return np.stack(observations), np.asarray(labels, dtype=np.int64), samples


def augment_mirrors(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observations = [x]
    labels = [y]
    for axis in ("lateral", "vertical"):
        observations.append(np.stack([mirror_guidance_observation(row, axis) for row in x]))
        labels.append(np.asarray([mirror_guidance_action(label, axis) for label in y]))
    both = np.stack(
        [
            mirror_guidance_observation(
                mirror_guidance_observation(row, "lateral"), "vertical"
            )
            for row in x
        ]
    )
    both_labels = np.asarray(
        [
            mirror_guidance_action(
                mirror_guidance_action(label, "lateral"), "vertical"
            )
            for label in y
        ]
    )
    observations.append(both)
    labels.append(both_labels)
    return np.concatenate(observations), np.concatenate(labels)


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    prediction = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(prediction == y_true))
    f1_values = []
    for action_id in range(len(GUIDANCE_ACTIONS)):
        tp = int(np.sum((prediction == action_id) & (y_true == action_id)))
        fp = int(np.sum((prediction == action_id) & (y_true != action_id)))
        fn = int(np.sum((prediction != action_id) & (y_true == action_id)))
        denominator = 2 * tp + fp + fn
        if denominator:
            f1_values.append(2.0 * tp / denominator)
    default_mask = y_true == 0
    nondefault_prediction = prediction != 0
    return {
        "accuracy": accuracy,
        "macro_f1_present_classes": float(np.mean(f1_values)) if f1_values else 0.0,
        "bt_default_recall": float(np.mean(prediction[default_mask] == 0))
        if np.any(default_mask)
        else None,
        "nondefault_precision": float(
            np.mean(y_true[nondefault_prediction] == prediction[nondefault_prediction])
        )
        if np.any(nondefault_prediction)
        else None,
        "mean_confidence": float(np.mean(np.max(probabilities, axis=1))),
        "expected_calibration_error": expected_calibration_error(
            y_true, prediction, np.max(probabilities, axis=1)
        ),
        "prediction_counts": {
            name: int(np.sum(prediction == action_id))
            for action_id, name in enumerate(GUIDANCE_ACTIONS)
        },
    }


def expected_calibration_error(y_true, prediction, confidence, bins: int = 10) -> float:
    result = 0.0
    for lower in np.linspace(0.0, 0.9, bins):
        upper = lower + 0.1
        mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= 1.0)
        if np.any(mask):
            result += float(np.mean(mask)) * abs(
                float(np.mean(prediction[mask] == y_true[mask]))
                - float(np.mean(confidence[mask]))
            )
    return result


def train_one(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    output_dir: Path,
    *,
    max_epochs: int = 400,
    patience: int = 40,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    permutation = np.random.default_rng(8600).permutation(len(x))
    validation_indexes = permutation[::5]
    training_indexes = np.setdiff1d(permutation, validation_indexes)
    train_x, train_y = augment_mirrors(x[training_indexes], y[training_indexes])
    validation_x = x[validation_indexes]
    validation_y = y[validation_indexes]
    counts = np.bincount(train_y, minlength=len(GUIDANCE_ACTIONS)).astype(np.float64)
    weights = np.ones(len(GUIDANCE_ACTIONS), dtype=np.float32)
    present = counts > 0
    weights[present] = np.sqrt(np.sum(counts) / (len(GUIDANCE_ACTIONS) * counts[present]))
    weights = np.clip(weights, 0.5, 3.0)

    model = SelectorMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
    train_tensor = torch.from_numpy(train_x)
    train_target = torch.from_numpy(train_y)
    validation_tensor = torch.from_numpy(validation_x)
    validation_target = torch.from_numpy(validation_y)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_tensor), train_target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(validation_tensor), validation_target))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.detach()),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("BC training did not produce a state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_probabilities = torch.softmax(model(torch.from_numpy(x[training_indexes])), dim=1).numpy()
        validation_probabilities = torch.softmax(model(validation_tensor), dim=1).numpy()
    layers = [module for module in model.network if isinstance(module, nn.Linear)]
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "model.npz"
    np.savez(
        weights_path,
        w1=layers[0].weight.detach().numpy().T.astype(np.float32),
        b1=layers[0].bias.detach().numpy().astype(np.float32),
        w2=layers[1].weight.detach().numpy().T.astype(np.float32),
        b2=layers[1].bias.detach().numpy().astype(np.float32),
        w3=layers[2].weight.detach().numpy().T.astype(np.float32),
        b3=layers[2].bias.detach().numpy().astype(np.float32),
    )
    model_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest().upper()
    result = {
        "seed": seed,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "training_samples_raw": int(len(training_indexes)),
        "training_samples_augmented": int(len(train_x)),
        "validation_samples": int(len(validation_indexes)),
        "class_counts": {
            name: int(np.sum(y == action_id))
            for action_id, name in enumerate(GUIDANCE_ACTIONS)
        },
        "class_weights": {
            name: float(weights[action_id])
            for action_id, name in enumerate(GUIDANCE_ACTIONS)
        },
        "training_metrics": metrics(y[training_indexes], train_probabilities),
        "validation_metrics": metrics(validation_y, validation_probabilities),
        "history_tail": history[-10:],
        "model_sha256": model_sha,
    }
    metadata = {
        "model_kind": "numpy_mlp_categorical",
        "policy_id": "guidance_selector_bc",
        "observation_contract": GUIDANCE_SELECTOR_CONTRACT_VERSION,
        "normalization_version": GUIDANCE_SELECTOR_NORMALIZATION_VERSION,
        "observation_size": GUIDANCE_SELECTOR_OBSERVATION_SIZE,
        "features": list(GUIDANCE_SELECTOR_FEATURES),
        "actions": list(GUIDANCE_ACTIONS),
        "hidden_layers": [64, 64],
        "activation": "tanh",
        "training_seed": seed,
        "model_sha256": model_sha,
        "training_result": result,
        "status": "CONSERVATIVE_FILTERED_BC_ONLY",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    NumpyMLPGuidanceSelector(output_dir).predict(np.zeros(45, dtype=np.float32))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Train three Guidance Selector BC bundles")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "artifacts/evaluations/guidance_selector/counterfactual_v1_20260819/dataset.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/models/guidance_selector_bc_v1",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    x, y, samples = load_dataset(args.dataset.resolve())
    output_root = args.output_root.resolve()
    results = []
    for seed in BC_SEEDS:
        result = train_one(x, y, seed, output_root / f"seed_{seed}")
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    selected = max(
        results,
        key=lambda item: (
            item["validation_metrics"]["macro_f1_present_classes"],
            item["validation_metrics"]["bt_default_recall"] or 0.0,
            -item["best_validation_loss"],
        ),
    )
    selected_source = output_root / f"seed_{selected['seed']}"
    selected_target = output_root / "selected_bundle"
    if selected_target.exists():
        raise FileExistsError(f"refusing to overwrite existing selected bundle: {selected_target}")
    shutil.copytree(selected_source, selected_target)
    summary = {
        "status": "BC_COMPLETED",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest().upper(),
        "samples": len(samples),
        "seeds": list(BC_SEEDS),
        "results": results,
        "selected_seed": selected["seed"],
        "selected_bundle": str(selected_target),
        "selected_model_sha256": selected["model_sha256"],
        "total_epochs": sum(item["epochs_run"] for item in results),
        "ppo_status": "PENDING_DEVELOPMENT_GATE",
    }
    (output_root / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
