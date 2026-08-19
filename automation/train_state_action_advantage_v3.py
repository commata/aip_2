from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import rankdata

from dogfight.ai.guidance_advantage import (
    GUIDANCE_ADVANTAGE_ACTIONS,
    GUIDANCE_SERVER_CONTRACT_VERSION,
    GUIDANCE_SERVER_FEATURES,
)
from dogfight.ai.state_action_advantage import MODEL_KIND


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "automation/evidence/state_conditioned_hybrid_v3/state_matrix_v3.json"
DEFAULT_OUTPUT = ROOT / "automation/evidence/state_conditioned_hybrid_v3/model_pilot_v3"
POSITIVE_EPSILON = 1e-6
LARGE_REGRESSION = -0.003
TARGET_SCALE = 0.02
ENSEMBLE_SEEDS = (31001, 31002, 31003)


def encode_factorized_input(row: dict[str, Any]) -> np.ndarray:
    state = np.asarray(row["server_observation"], dtype=np.float32)
    action = row["action_parameters"]
    vector = np.concatenate(
        (
            state,
            np.asarray(
                [
                    *action["axis_one_hot"],
                    action["sign"],
                    action["magnitude_norm"],
                    action["duration_norm"],
                ],
                dtype=np.float32,
            ),
        )
    )
    if vector.shape != (48,) or not np.all(np.isfinite(vector)):
        raise ValueError("expected finite 42D state + 6D factorized action")
    return vector


def unique_state_actions(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["action"] != "BT_DEFAULT":
            grouped[(row["state_hash"], row["candidate_id"])].append(row)
    unique = []
    for replicate_rows in grouped.values():
        row = dict(replicate_rows[0])
        row["damage_delta"] = float(
            np.mean([float(item["damage_delta"]) for item in replicate_rows])
        )
        row["replicate_count"] = len(replicate_rows)
        unique.append(row)
    return sorted(unique, key=lambda row: (row["state_hash"], row["candidate_id"]))


def assign_group_folds(rows: list[dict[str, Any]], folds: int = 5) -> dict[str, int]:
    by_family: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["state_hash"] not in by_family[row["family"]]:
            by_family[row["family"]].append(row["state_hash"])
    assignment: dict[str, int] = {}
    for family, groups in sorted(by_family.items()):
        ordered = sorted(
            groups,
            key=lambda group: hashlib.sha256(f"{family}:{group}".encode()).hexdigest(),
        )
        for index, group in enumerate(ordered):
            assignment[group] = index % folds
    return assignment


class AdvantageNetwork(nn.Module):
    def __init__(self, input_size: int = 48, hidden_size: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_size, 1)
        self.positive_head = nn.Linear(hidden_size, 1)
        self.regression_head = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.shared(values)
        return (
            self.mean_head(hidden).squeeze(-1),
            self.positive_head(hidden).squeeze(-1),
            self.regression_head(hidden).squeeze(-1),
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def fit_model(
    x: np.ndarray,
    damage: np.ndarray,
    *,
    seed: int,
    epochs: int,
) -> tuple[AdvantageNetwork, dict[str, np.ndarray], list[float]]:
    _seed_everything(seed)
    mean = np.mean(x, axis=0).astype(np.float32)
    scale = np.std(x, axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    inputs = torch.from_numpy(((x - mean) / scale).astype(np.float32))
    target_mean = torch.from_numpy((damage / TARGET_SCALE).astype(np.float32))
    target_positive = torch.from_numpy((damage > POSITIVE_EPSILON).astype(np.float32))
    target_regression = torch.from_numpy((damage <= LARGE_REGRESSION).astype(np.float32))
    model = AdvantageNetwork()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    positive_weight = torch.tensor(
        max(1.0, float(np.sum(damage <= POSITIVE_EPSILON) / max(1, np.sum(damage > POSITIVE_EPSILON))))
    )
    regression_weight = torch.tensor(
        min(20.0, max(1.0, float(np.sum(damage > LARGE_REGRESSION) / max(1, np.sum(damage <= LARGE_REGRESSION)))))
    )
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        predicted_mean, positive_logit, regression_logit = model(inputs)
        loss = F.smooth_l1_loss(predicted_mean, target_mean, beta=0.25)
        loss = loss + 0.5 * F.binary_cross_entropy_with_logits(
            positive_logit, target_positive, pos_weight=positive_weight
        )
        loss = loss + 0.5 * F.binary_cross_entropy_with_logits(
            regression_logit, target_regression, pos_weight=regression_weight
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return model, {"mean": mean, "scale": scale}, losses


def predict(
    model: AdvantageNetwork, normalization: dict[str, np.ndarray], x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs = torch.from_numpy(
        ((x - normalization["mean"]) / normalization["scale"]).astype(np.float32)
    )
    model.eval()
    with torch.no_grad():
        mean, positive, regression = model(inputs)
    return (
        mean.numpy().astype(np.float64) * TARGET_SCALE,
        torch.sigmoid(positive).numpy().astype(np.float64),
        torch.sigmoid(regression).numpy().astype(np.float64),
    )


def ensemble_predictions(
    models: list[tuple[AdvantageNetwork, dict[str, np.ndarray]]], x: np.ndarray
) -> dict[str, np.ndarray]:
    predictions = [predict(model, normalization, x) for model, normalization in models]
    means = np.stack([item[0] for item in predictions])
    return {
        "mean": np.mean(means, axis=0),
        "std": np.std(means, axis=0),
        "positive_probability": np.mean(np.stack([item[1] for item in predictions]), axis=0),
        "regression_probability": np.mean(np.stack([item[2] for item in predictions]), axis=0),
    }


def threshold_grid() -> list[dict[str, float]]:
    return [
        {"score": score, "positive": positive, "regression": regression, "lambda": penalty}
        for score in (0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.010)
        for positive in (0.60, 0.70, 0.80, 0.90)
        for regression in (0.02, 0.05, 0.10, 0.20)
        for penalty in (1.0, 2.0)
    ]


def policy_value(
    rows: list[dict[str, Any]], predictions: dict[str, np.ndarray], threshold: dict[str, float]
) -> dict[str, Any]:
    by_state: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_state[row["state_hash"]].append(index)
    values = []
    oracle = []
    intervention_flags = []
    selected_rows = []
    for state_hash, indices in by_state.items():
        passing = []
        for index in indices:
            score = float(
                predictions["mean"][index]
                - threshold["lambda"] * predictions["std"][index]
            )
            if (
                score > threshold["score"]
                and predictions["positive_probability"][index] > threshold["positive"]
                and predictions["regression_probability"][index] < threshold["regression"]
            ):
                passing.append((score, index))
        oracle_value = max(0.0, *(float(rows[index]["damage_delta"]) for index in indices))
        oracle.append(oracle_value)
        if passing:
            _, selected = max(passing)
            value = float(rows[selected]["damage_delta"])
            selected_rows.append(
                {
                    "state_hash": state_hash,
                    "candidate_id": rows[selected]["candidate_id"],
                    "actual_damage_delta": value,
                }
            )
            intervention_flags.append(True)
        else:
            value = 0.0
            intervention_flags.append(False)
        values.append(value)
    array = np.asarray(values, dtype=np.float64)
    intervened = np.asarray(intervention_flags, dtype=bool)
    selected_values = array[intervened]
    return {
        "states": len(values),
        "interventions": int(np.sum(intervened)),
        "coverage": float(np.mean(intervened)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "positive_ratio_all_states": float(np.mean(array > POSITIVE_EPSILON)),
        "intervention_precision": float(np.mean(selected_values > POSITIVE_EPSILON)) if selected_values.size else 0.0,
        "large_regression_ratio": float(np.mean(selected_values <= LARGE_REGRESSION)) if selected_values.size else 0.0,
        "oracle_regret_mean": float(np.mean(np.asarray(oracle) - array)),
        "selected_rows": selected_rows,
    }


def select_threshold(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in diagnostics
        if row["policy"]["interventions"] >= 5
        and row["policy"]["intervention_precision"] >= 0.65
        and row["policy"]["mean"] > 0.0
        and row["policy"]["large_regression_ratio"] <= 0.05
    ]
    candidates = eligible or diagnostics
    selected = max(
        candidates,
        key=lambda row: (
            row["policy"]["mean"],
            row["policy"]["intervention_precision"],
            -row["policy"]["large_regression_ratio"],
        ),
    )
    return {
        **selected,
        "offline_gate_passed": bool(eligible),
        "selection_status": (
            "OFFLINE_POLICY_GATE_PASSED" if eligible else "OFFLINE_POLICY_GATE_FAILED"
        ),
    }


def prediction_diagnostics(
    rows: list[dict[str, Any]], damage: np.ndarray, predictions: dict[str, np.ndarray]
) -> dict[str, Any]:
    mean = predictions["mean"]
    positive_label = (damage > POSITIVE_EPSILON).astype(np.float64)
    regression_label = (damage <= LARGE_REGRESSION).astype(np.float64)
    pearson = float(np.corrcoef(mean, damage)[0, 1]) if np.std(mean) > 1e-12 else 0.0
    spearman = float(
        np.corrcoef(rankdata(mean, method="average"), rankdata(damage, method="average"))[0, 1]
    )
    by_state: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_state[row["state_hash"]].append(index)
    agreement = []
    regrets = []
    for indices in by_state.values():
        predicted = max(indices, key=lambda index: mean[index])
        actual = max(indices, key=lambda index: damage[index])
        agreement.append(rows[predicted]["candidate_id"] == rows[actual]["candidate_id"])
        regrets.append(float(damage[actual] - damage[predicted]))

    def calibration(probability: np.ndarray, label: np.ndarray) -> list[dict[str, Any]]:
        bins = []
        for lower in np.linspace(0.0, 0.8, 5):
            upper = lower + 0.2
            selected = (probability >= lower) & (
                probability <= upper if upper >= 1.0 else probability < upper
            )
            if np.any(selected):
                bins.append(
                    {
                        "lower": float(lower),
                        "upper": float(upper),
                        "count": int(np.sum(selected)),
                        "predicted_mean": float(np.mean(probability[selected])),
                        "actual_ratio": float(np.mean(label[selected])),
                    }
                )
        return bins

    return {
        "predicted_actual_advantage_pearson": pearson,
        "predicted_actual_advantage_spearman": spearman,
        "positive_brier": float(np.mean((predictions["positive_probability"] - positive_label) ** 2)),
        "large_regression_brier": float(
            np.mean((predictions["regression_probability"] - regression_label) ** 2)
        ),
        "top_action_agreement": float(np.mean(agreement)),
        "ungated_top_action_regret_mean": float(np.mean(regrets)),
        "positive_calibration": calibration(predictions["positive_probability"], positive_label),
        "large_regression_calibration": calibration(
            predictions["regression_probability"], regression_label
        ),
    }


def train_and_evaluate(
    rows: list[dict[str, Any]], *, folds: int, epochs: int, seeds: tuple[int, ...]
) -> tuple[dict[str, Any], list[tuple[AdvantageNetwork, dict[str, np.ndarray]]]]:
    unique = unique_state_actions(rows)
    x = np.asarray([encode_factorized_input(row) for row in unique], dtype=np.float32)
    damage = np.asarray([row["damage_delta"] for row in unique], dtype=np.float32)
    assignment = assign_group_folds(unique, folds)
    oof = {
        key: np.zeros(len(unique), dtype=np.float64)
        for key in ("mean", "std", "positive_probability", "regression_probability")
    }
    member_oof = [
        {
            key: np.zeros(len(unique), dtype=np.float64)
            for key in ("mean", "std", "positive_probability", "regression_probability")
        }
        for _ in seeds
    ]
    fold_summaries = []
    for fold in range(folds):
        test = np.asarray([assignment[row["state_hash"]] == fold for row in unique])
        train = ~test
        fold_models = []
        for seed in seeds:
            model, normalization, losses = fit_model(
                x[train], damage[train], seed=seed + fold * 100, epochs=epochs
            )
            fold_models.append((model, normalization))
        predicted = ensemble_predictions(fold_models, x[test])
        for key in oof:
            oof[key][test] = predicted[key]
        for member_index, (model, normalization) in enumerate(fold_models):
            member_mean, member_positive, member_regression = predict(
                model, normalization, x[test]
            )
            member_oof[member_index]["mean"][test] = member_mean
            member_oof[member_index]["std"][test] = 0.0
            member_oof[member_index]["positive_probability"][test] = member_positive
            member_oof[member_index]["regression_probability"][test] = member_regression
        fold_summaries.append(
            {
                "fold": fold,
                "train_states": len({unique[index]["state_hash"] for index in np.flatnonzero(train)}),
                "test_states": len({unique[index]["state_hash"] for index in np.flatnonzero(test)}),
                "train_rows": int(np.sum(train)),
                "test_rows": int(np.sum(test)),
            }
        )
    threshold_diagnostics = []
    for threshold in threshold_grid():
        threshold_diagnostics.append(
            {"threshold": threshold, "policy": policy_value(unique, oof, threshold)}
        )
    selected = select_threshold(threshold_diagnostics)
    seed_policies = {
        str(seed): policy_value(unique, member_oof[index], selected["threshold"])
        for index, seed in enumerate(seeds)
    }
    final_models = []
    training_losses = {}
    for seed in seeds:
        model, normalization, losses = fit_model(x, damage, seed=seed, epochs=epochs)
        final_models.append((model, normalization))
        training_losses[str(seed)] = {"initial": losses[0], "final": losses[-1]}
    result = {
        "schema_version": "state_action_distributional_advantage_v3.v1",
        "unique_states": len({row["state_hash"] for row in unique}),
        "unique_state_action_pairs": len(unique),
        "input_contract": "guidance_selector_server_v2_42d + factorized_action_6d",
        "model_kind": MODEL_KIND,
        "observation_contract": GUIDANCE_SERVER_CONTRACT_VERSION,
        "observation_size": 42,
        "features": list(GUIDANCE_SERVER_FEATURES),
        "actions": list(GUIDANCE_ADVANTAGE_ACTIONS),
        "model_architecture": "MLP 48-64-64, mean/P(positive)/P(large-regression) heads",
        "seeds": list(seeds),
        "epochs_per_model": epochs,
        "optimizer_updates": epochs * len(seeds) * (folds + 1),
        "folds": fold_summaries,
        "threshold_grid_frozen_before_oof_evaluation": threshold_grid(),
        "selected_oof_policy": selected,
        "oof_prediction_diagnostics": prediction_diagnostics(unique, damage, oof),
        "seed_oof_policies_at_selected_threshold": seed_policies,
        "all_threshold_diagnostics": threshold_diagnostics,
        "training_losses": training_losses,
    }
    return result, final_models


def save_bundle(
    output: Path,
    models: list[tuple[AdvantageNetwork, dict[str, np.ndarray]]],
    result: dict[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for model_index, (model, normalization) in enumerate(models):
        arrays[f"model_{model_index}_input_mean"] = normalization["mean"]
        arrays[f"model_{model_index}_input_scale"] = normalization["scale"]
        for name, value in model.state_dict().items():
            arrays[f"model_{model_index}_{name}"] = value.detach().numpy()
    model_path = output / "model.npz"
    np.savez_compressed(model_path, **arrays)
    metadata = dict(result)
    selected = result["selected_oof_policy"]
    metadata["offline_gate_passed"] = bool(selected["offline_gate_passed"])
    metadata["runtime_threshold"] = dict(selected["threshold"])
    metadata["runtime_candidates"] = [
        {"action": action, "magnitude_deg": 0.25, "duration_frames": 36}
        for action in GUIDANCE_ADVANTAGE_ACTIONS[1:]
    ]
    metadata["ensemble_size"] = len(models)
    metadata["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest().upper()
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train factorized distributional advantage v3")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument(
        "--include-experiment",
        action="append",
        default=[],
        help="Train only rows from one or more named source experiments",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    rows = payload["rows"]
    if args.include_experiment:
        included = set(args.include_experiment)
        rows = [row for row in rows if row["source_experiment"] in included]
        if not rows:
            raise ValueError(f"no rows matched source experiments: {sorted(included)}")
    result, models = train_and_evaluate(
        rows, folds=args.folds, epochs=args.epochs, seeds=ENSEMBLE_SEEDS
    )
    result["included_source_experiments"] = list(args.include_experiment) or ["all"]
    save_bundle(args.output.resolve(), models, result)
    print(json.dumps(result["selected_oof_policy"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
