from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


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
)


RULE_ID = "rear120_early_preaim_v1"
RULE_ACTION = "VP_EL_POS_SMALL"
RULE_ACTION_ID = GUIDANCE_ACTIONS.index(RULE_ACTION)


def rule_action(observation: np.ndarray) -> int:
    """Narrow prior: only the first 36 frames of a safe Rear120 gate window."""
    vector = np.asarray(observation, dtype=np.float32)
    gate_elapsed_norm = float(vector[42])
    gate_active = float(vector[43])
    return RULE_ACTION_ID if gate_active > 0.5 and gate_elapsed_norm < -0.2 else 0


def build_arrays() -> dict[str, np.ndarray]:
    """Analytically distill the two-condition rule into a 45-64-64-9 tanh MLP."""
    w1 = np.zeros((GUIDANCE_SELECTOR_OBSERVATION_SIZE, 64), dtype=np.float32)
    b1 = np.zeros(64, dtype=np.float32)
    w2 = np.zeros((64, 64), dtype=np.float32)
    b2 = np.zeros(64, dtype=np.float32)
    w3 = np.zeros((64, len(GUIDANCE_ACTIONS)), dtype=np.float32)
    b3 = np.full(len(GUIDANCE_ACTIONS), -8.0, dtype=np.float32)

    # h1[0] is positive before the frozen 36/90-frame cutoff; h1[1] is
    # positive only while the Rear120+safety gate is active.
    w1[42, 0] = -8.0
    b1[0] = -1.6
    w1[43, 1] = 8.0

    # Sharpen both predicates independently. The output layer performs the AND.
    w2[0, 0] = 10.0
    w2[1, 1] = 10.0

    # Positive h2 chooses the smallest vertical Guidance primitive; negative
    # h2 chooses exact BT_DEFAULT. Other actions remain unreachable.
    w3[0, 0] = -6.0
    w3[1, 0] = -6.0
    w3[0, RULE_ACTION_ID] = 6.0
    w3[1, RULE_ACTION_ID] = 6.0
    b3[0] = 6.0
    b3[RULE_ACTION_ID] = -6.0
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}


def verify_rule(bundle: Path) -> dict:
    selector = NumpyMLPGuidanceSelector(bundle)
    checks = []
    for gate in (-1.0, 1.0):
        for elapsed in (-1.0, -0.8, -0.4, -0.21, -0.19, 0.0, 1.0):
            observation = np.zeros(GUIDANCE_SELECTOR_OBSERVATION_SIZE, dtype=np.float32)
            observation[42] = elapsed
            observation[43] = gate
            expected = rule_action(observation)
            actual, confidence, _ = selector.predict(observation)
            checks.append(
                {
                    "gate_active": gate,
                    "gate_elapsed_norm": elapsed,
                    "expected_action": GUIDANCE_ACTIONS[expected],
                    "actual_action": GUIDANCE_ACTIONS[actual],
                    "confidence": confidence,
                    "matched": actual == expected,
                }
            )
    if not all(row["matched"] for row in checks):
        raise RuntimeError("distilled selector does not match the frozen rule grid")
    return {"checks": checks, "matched": len(checks), "total": len(checks)}


def build(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "model.npz"
    np.savez(weights_path, **build_arrays())
    model_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest().upper()
    metadata = {
        "model_kind": "numpy_mlp_categorical",
        "policy_id": "guidance_selector_rule_distilled",
        "observation_contract": GUIDANCE_SELECTOR_CONTRACT_VERSION,
        "normalization_version": GUIDANCE_SELECTOR_NORMALIZATION_VERSION,
        "observation_size": GUIDANCE_SELECTOR_OBSERVATION_SIZE,
        "features": list(GUIDANCE_SELECTOR_FEATURES),
        "actions": list(GUIDANCE_ACTIONS),
        "hidden_layers": [64, 64],
        "activation": "tanh",
        "training_seed": 8799,
        "training_steps": 0,
        "distillation_kind": "ANALYTIC_RULE_DISTILLATION",
        "rule_id": RULE_ID,
        "rule": {
            "gate": "Rear120 plus safety veto active",
            "preaim_window": "gate_elapsed_frames < 36 of maximum_active_frames=90",
            "action": RULE_ACTION,
            "otherwise": "BT_DEFAULT",
        },
        "model_sha256": model_sha256,
        "status": "EXPERIMENTAL_SAFE_HYBRID",
        "promotion_status": "NOT_PROMOTED",
        "claim": "No learned performance-improvement claim",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    verification = verify_rule(output_dir)
    result = {
        "bundle": str(output_dir.resolve()),
        "model_sha256": model_sha256,
        "rule_id": RULE_ID,
        "verification": verification,
    }
    (output_dir / "distillation_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Build the frozen safe rule-distilled Guidance selector")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/models/guidance_selector_bc_v1/seed_8799",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args().output_dir.resolve()), indent=2, sort_keys=True))
