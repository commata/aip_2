from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from dogfight.ai.tactical_modes import TACTICAL_HOLD_FRAMES, TACTICAL_MODES_T1
from dogfight.ai.temporal_observation import (
    TEMPORAL_FEATURES,
    TEMPORAL_OBSERVATION_SIZE,
    TEMPORAL_SERVER_CONTRACT_VERSION,
    validate_temporal_observation,
)


TACTICAL_MODEL_KIND = "numpy_temporal_tactical_distributional_advantage"
TACTICAL_ACTION_FEATURES = (*tuple(f"mode_{name}" for name in TACTICAL_MODES_T1), "hold_norm")
TACTICAL_MODEL_INPUT_SIZE = TEMPORAL_OBSERVATION_SIZE + len(TACTICAL_ACTION_FEATURES)


def encode_tactical_option(mode: str, hold_frames: int) -> np.ndarray:
    if mode not in TACTICAL_MODES_T1:
        raise ValueError(f"unsupported T1 Tactical mode: {mode}")
    if mode == "BT_DEFAULT":
        if hold_frames != 0:
            raise ValueError("BT_DEFAULT hold must be zero")
    elif hold_frames not in TACTICAL_HOLD_FRAMES:
        raise ValueError(f"unsupported Tactical hold: {hold_frames}")
    return np.asarray(
        [*(float(mode == candidate) for candidate in TACTICAL_MODES_T1), hold_frames / 120.0],
        dtype=np.float32,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


class NumpyTemporalTacticalAdvantageSelector:
    observation_contract = TEMPORAL_SERVER_CONTRACT_VERSION

    def __init__(self, bundle_path: str | Path, *, require_offline_gate: bool = True):
        self.bundle_path = Path(bundle_path).resolve()
        self.metadata = json.loads(
            (self.bundle_path / "metadata.json").read_text(encoding="utf-8")
        )
        if self.metadata.get("model_kind") != TACTICAL_MODEL_KIND:
            raise ValueError("unsupported Temporal Tactical model kind")
        if self.metadata.get("observation_contract") != self.observation_contract:
            raise ValueError("Temporal Tactical observation contract mismatch")
        if tuple(self.metadata.get("features", ())) != TEMPORAL_FEATURES:
            raise ValueError("Temporal Tactical feature order mismatch")
        if require_offline_gate and not self.metadata.get("offline_gate_passed", False):
            raise ValueError("Temporal Tactical bundle did not pass the offline gate")
        model_path = self.bundle_path / "model.npz"
        expected = str(self.metadata.get("model_sha256", "")).upper()
        actual = hashlib.sha256(model_path.read_bytes()).hexdigest().upper()
        if expected != actual:
            raise ValueError("Temporal Tactical model SHA256 mismatch")
        arrays = np.load(model_path, allow_pickle=False)
        self.threshold = dict(self.metadata["runtime_threshold"])
        self.target_scale = float(self.metadata["target_scale"])
        self.candidates = tuple(self.metadata["runtime_candidates"])
        self.models = []
        for index in range(int(self.metadata["ensemble_size"])):
            prefix = f"model_{index}_"
            model = {
                "input_mean": np.asarray(arrays[prefix + "input_mean"], dtype=np.float32),
                "input_scale": np.asarray(arrays[prefix + "input_scale"], dtype=np.float32),
                "w1": np.asarray(arrays[prefix + "shared.0.weight"], dtype=np.float32),
                "b1": np.asarray(arrays[prefix + "shared.0.bias"], dtype=np.float32),
                "w2": np.asarray(arrays[prefix + "shared.2.weight"], dtype=np.float32),
                "b2": np.asarray(arrays[prefix + "shared.2.bias"], dtype=np.float32),
                "mean_w": np.asarray(arrays[prefix + "mean_head.weight"], dtype=np.float32),
                "mean_b": np.asarray(arrays[prefix + "mean_head.bias"], dtype=np.float32),
                "positive_w": np.asarray(arrays[prefix + "positive_head.weight"], dtype=np.float32),
                "positive_b": np.asarray(arrays[prefix + "positive_head.bias"], dtype=np.float32),
                "regression_w": np.asarray(arrays[prefix + "regression_head.weight"], dtype=np.float32),
                "regression_b": np.asarray(arrays[prefix + "regression_head.bias"], dtype=np.float32),
            }
            self._validate_model(model)
            self.models.append(model)
        self.support_examples = np.asarray(arrays["support_examples"], dtype=np.float32)
        self.support_scale = np.asarray(arrays["support_scale"], dtype=np.float32)
        self.ood = dict(self.metadata["runtime_ood_support"])
        if self.support_examples.ndim != 2 or self.support_examples.shape[1] != 93:
            raise ValueError("Temporal Tactical support examples must be N x 93")
        if self.support_scale.shape != (93,) or np.any(self.support_scale <= 0.0):
            raise ValueError("Temporal Tactical support scale must be positive 93D")
        self.last_prediction: dict[str, Any] = {}

    @staticmethod
    def _validate_model(model: dict[str, np.ndarray]) -> None:
        if model["input_mean"].shape != (TACTICAL_MODEL_INPUT_SIZE,):
            raise ValueError("Temporal Tactical model normalization must be 98D")
        if model["input_scale"].shape != (TACTICAL_MODEL_INPUT_SIZE,) or np.any(
            model["input_scale"] <= 0.0
        ):
            raise ValueError("Temporal Tactical model scale must be positive 98D")
        hidden = model["w1"].shape[0]
        if model["w1"].shape != (hidden, TACTICAL_MODEL_INPUT_SIZE):
            raise ValueError("invalid Temporal Tactical first layer")
        if model["w2"].shape != (hidden, hidden):
            raise ValueError("invalid Temporal Tactical second layer")
        for head in ("mean", "positive", "regression"):
            if model[f"{head}_w"].shape != (1, hidden):
                raise ValueError(f"invalid Temporal Tactical {head} head")

    def _forward(
        self, model: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = (inputs - model["input_mean"]) / model["input_scale"]
        hidden = np.tanh(values @ model["w1"].T + model["b1"])
        hidden = np.tanh(hidden @ model["w2"].T + model["b2"])
        mean = (hidden @ model["mean_w"].T + model["mean_b"]).reshape(-1)
        positive = _sigmoid((hidden @ model["positive_w"].T + model["positive_b"]).reshape(-1))
        regression = _sigmoid(
            (hidden @ model["regression_w"].T + model["regression_b"]).reshape(-1)
        )
        return mean * self.target_scale, positive, regression

    def ood_distance(self, observation: Any) -> float:
        state = validate_temporal_observation(observation)
        normalized = (self.support_examples - state) / self.support_scale
        return float(np.sqrt(np.min(np.mean(normalized * normalized, axis=1))))

    def score_options(self, observation: Any) -> list[dict[str, Any]]:
        state = validate_temporal_observation(observation)
        inputs = np.stack(
            [
                np.concatenate(
                    (state, encode_tactical_option(row["mode"], int(row["hold_frames"])))
                )
                for row in self.candidates
            ]
        ).astype(np.float32)
        predicted = [self._forward(model, inputs) for model in self.models]
        means = np.stack([item[0] for item in predicted])
        mean = means.mean(axis=0)
        std = means.std(axis=0)
        positive = np.stack([item[1] for item in predicted]).mean(axis=0)
        regression = np.stack([item[2] for item in predicted]).mean(axis=0)
        conservative = mean - float(self.threshold["lambda"]) * std
        return [
            {
                **dict(candidate),
                "mean_advantage": float(mean[index]),
                "q10_advantage": float(mean[index] - 1.2815515655 * std[index]),
                "q50_advantage": float(mean[index]),
                "ensemble_std": float(std[index]),
                "positive_probability": float(positive[index]),
                "large_regression_probability": float(regression[index]),
                "conservative_score": float(conservative[index]),
            }
            for index, candidate in enumerate(self.candidates)
        ]

    def select(self, observation: Any) -> dict[str, Any]:
        distance = self.ood_distance(observation)
        if distance > float(self.ood["threshold"]):
            self.last_prediction = {
                "mode": "BT_DEFAULT",
                "hold_frames": 0,
                "abstention_reason": "OOD",
                "ood_distance": distance,
                "options": [],
            }
            return dict(self.last_prediction)
        scored = self.score_options(observation)
        eligible = [
            row
            for row in scored
            if row["conservative_score"] > float(self.threshold["score"])
            and row["positive_probability"] > float(self.threshold["positive"])
            and row["large_regression_probability"] < float(self.threshold["regression"])
        ]
        if eligible:
            selected = max(eligible, key=lambda row: row["conservative_score"])
            self.last_prediction = {
                **selected,
                "abstention_reason": "",
                "ood_distance": distance,
                "options": scored,
            }
            return dict(self.last_prediction)
        best = max(scored, key=lambda row: row["conservative_score"])
        if best["large_regression_probability"] >= float(self.threshold["regression"]):
            reason = "HIGH_REGRESSION_RISK"
        elif best["positive_probability"] <= float(self.threshold["positive"]):
            reason = "LOW_PPOSITIVE"
        elif best["mean_advantage"] > float(self.threshold["score"]):
            reason = "HIGH_UNCERTAINTY"
        else:
            reason = "NO_ACTION_ADVANTAGE"
        self.last_prediction = {
            "mode": "BT_DEFAULT",
            "hold_frames": 0,
            "abstention_reason": reason,
            "ood_distance": distance,
            "best_rejected_option": best,
            "options": scored,
        }
        return dict(self.last_prediction)
