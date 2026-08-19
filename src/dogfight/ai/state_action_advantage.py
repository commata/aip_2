from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from dogfight.ai.guidance_advantage import (
    GUIDANCE_ADVANTAGE_ACTIONS,
    GUIDANCE_SERVER_CONTRACT_VERSION,
    GUIDANCE_SERVER_FEATURES,
    validate_server_guidance_observation,
)
from dogfight.ai.guidance_selector import GUIDANCE_ACTIONS, GUIDANCE_ACTION_TO_ID


MODEL_KIND = "numpy_state_action_distributional_advantage"
ACTION_FEATURES = (
    "axis_default",
    "axis_azimuth",
    "axis_elevation",
    "sign",
    "magnitude_norm",
    "duration_norm",
)


def factorized_action(action: str, magnitude_deg: float, duration_frames: int) -> np.ndarray:
    if action == "BT_DEFAULT":
        axis, sign = "default", 0.0
    elif "_AZ_" in action:
        axis, sign = "azimuth", 1.0 if "_POS_" in action else -1.0
    elif "_EL_" in action:
        axis, sign = "elevation", 1.0 if "_POS_" in action else -1.0
    else:
        raise ValueError(f"unsupported factorized action: {action}")
    return np.asarray(
        [
            float(axis == "default"),
            float(axis == "azimuth"),
            float(axis == "elevation"),
            sign,
            float(magnitude_deg) / 0.5 if magnitude_deg else 0.0,
            float(duration_frames) / 36.0 if duration_frames else 0.0,
        ],
        dtype=np.float32,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


class NumpyStateActionAdvantageSelector:
    """Dependency-light conservative Q_adv ensemble for the server-safe 42D contract."""

    observation_contract = GUIDANCE_SERVER_CONTRACT_VERSION

    def __init__(self, bundle_path: str | Path):
        self.bundle_path = Path(bundle_path).resolve()
        metadata_path = self.bundle_path / "metadata.json"
        weights_path = self.bundle_path / "model.npz"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("model_kind") != MODEL_KIND:
            raise ValueError("unsupported state-action advantage model kind")
        if self.metadata.get("observation_contract") != GUIDANCE_SERVER_CONTRACT_VERSION:
            raise ValueError("state-action observation contract mismatch")
        if tuple(self.metadata.get("features", ())) != GUIDANCE_SERVER_FEATURES:
            raise ValueError("state-action feature order mismatch")
        if not self.metadata.get("offline_gate_passed", False):
            raise ValueError("state-action bundle did not pass the frozen offline policy gate")
        expected = str(self.metadata.get("model_sha256", "")).upper()
        actual = hashlib.sha256(weights_path.read_bytes()).hexdigest().upper()
        if expected != actual:
            raise ValueError(f"state-action model SHA256 mismatch: expected={expected}, actual={actual}")
        self.threshold = dict(self.metadata["runtime_threshold"])
        self.candidates = tuple(self.metadata["runtime_candidates"])
        if not self.candidates or any(row["action"] not in GUIDANCE_ADVANTAGE_ACTIONS[1:] for row in self.candidates):
            raise ValueError("runtime candidates must be nondefault primary Guidance actions")
        arrays = np.load(weights_path, allow_pickle=False)
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

    @staticmethod
    def _validate_model(model: dict[str, np.ndarray]) -> None:
        if model["input_mean"].shape != (48,) or model["input_scale"].shape != (48,):
            raise ValueError("state-action normalization must be 48D")
        if np.any(model["input_scale"] <= 0.0):
            raise ValueError("state-action input scale must be positive")
        if model["w1"].shape[1] != 48 or model["b1"].shape != (model["w1"].shape[0],):
            raise ValueError("invalid state-action first layer")
        hidden = model["w1"].shape[0]
        if model["w2"].shape != (hidden, hidden) or model["b2"].shape != (hidden,):
            raise ValueError("invalid state-action second layer")
        for head in ("mean", "positive", "regression"):
            if model[f"{head}_w"].shape != (1, hidden) or model[f"{head}_b"].shape != (1,):
                raise ValueError(f"invalid state-action {head} head")

    @staticmethod
    def _forward(model: dict[str, np.ndarray], inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = (inputs - model["input_mean"]) / model["input_scale"]
        hidden = np.tanh(values @ model["w1"].T + model["b1"])
        hidden = np.tanh(hidden @ model["w2"].T + model["b2"])
        mean = (hidden @ model["mean_w"].T + model["mean_b"]).reshape(-1) * float(
            model.get("target_scale", 0.02)
        )
        positive = _sigmoid((hidden @ model["positive_w"].T + model["positive_b"]).reshape(-1))
        regression = _sigmoid((hidden @ model["regression_w"].T + model["regression_b"]).reshape(-1))
        return mean, positive, regression

    def score_actions(self, observation: Any) -> list[dict[str, Any]]:
        state = validate_server_guidance_observation(observation)
        inputs = np.asarray(
            [
                np.concatenate(
                    (
                        state,
                        factorized_action(
                            candidate["action"],
                            float(candidate["magnitude_deg"]),
                            int(candidate["duration_frames"]),
                        ),
                    )
                )
                for candidate in self.candidates
            ],
            dtype=np.float32,
        )
        predicted = [self._forward(model, inputs) for model in self.models]
        means = np.stack([row[0] for row in predicted])
        positive = np.mean(np.stack([row[1] for row in predicted]), axis=0)
        regression = np.mean(np.stack([row[2] for row in predicted]), axis=0)
        mean = np.mean(means, axis=0)
        std = np.std(means, axis=0)
        conservative = mean - float(self.threshold["lambda"]) * std
        return [
            {
                **dict(candidate),
                "mean_advantage": float(mean[index]),
                "ensemble_std": float(std[index]),
                "positive_probability": float(positive[index]),
                "large_regression_probability": float(regression[index]),
                "conservative_score": float(conservative[index]),
            }
            for index, candidate in enumerate(self.candidates)
        ]

    def predict(self, observation: np.ndarray) -> tuple[int, float, np.ndarray]:
        scored = self.score_actions(observation)
        eligible = [
            row
            for row in scored
            if row["conservative_score"] > float(self.threshold["score"])
            and row["positive_probability"] > float(self.threshold["positive"])
            and row["large_regression_probability"] < float(self.threshold["regression"])
        ]
        probabilities = np.zeros(len(GUIDANCE_ACTIONS), dtype=np.float32)
        if not eligible:
            probabilities[0] = 1.0
            return 0, 1.0, probabilities
        selected = max(eligible, key=lambda row: row["conservative_score"])
        action_id = GUIDANCE_ACTION_TO_ID[selected["action"]]
        probabilities[action_id] = float(selected["positive_probability"])
        probabilities[0] = 1.0 - probabilities[action_id]
        return action_id, float(selected["positive_probability"]), probabilities


def load_guidance_selector_bundle(bundle_path: str | Path):
    bundle = Path(bundle_path).resolve()
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("model_kind") == MODEL_KIND:
        return NumpyStateActionAdvantageSelector(bundle)
    from dogfight.ai.guidance_selector import NumpyMLPGuidanceSelector

    return NumpyMLPGuidanceSelector(bundle)
