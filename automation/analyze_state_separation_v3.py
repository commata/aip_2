from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dogfight.ai.guidance_advantage import GUIDANCE_SERVER_FEATURES


DEFAULT_DATASET = ROOT / "automation/evidence/state_conditioned_hybrid_v3/state_matrix_v3.json"
DEFAULT_OUTPUT = ROOT / "automation/evidence/state_conditioned_hybrid_v3/state_separation_v3.json"
DEFAULT_REPORT = ROOT / "automation/reports/StateConditionedHybrid_v3_상태분리.md"


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    values = np.asarray(score, dtype=np.float64)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = rankdata(values, method="average")
    return float((np.sum(ranks[y == 1]) - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    y = np.asarray(y_true, dtype=np.int64)[order]
    positives = int(np.sum(y))
    if positives == 0:
        return 0.0
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float(np.sum(precision * y) / positives)


def assign_group_folds(groups: list[str], labels: np.ndarray, folds: int = 5) -> dict[str, int]:
    group_labels: dict[str, list[int]] = {}
    for group, label in zip(groups, labels, strict=True):
        group_labels.setdefault(group, []).append(int(label))
    positive = sorted(
        (group for group, values in group_labels.items() if np.mean(values) >= 0.5),
        key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
    )
    negative = sorted(
        (group for group, values in group_labels.items() if np.mean(values) < 0.5),
        key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
    )
    assignment = {}
    for bucket in (positive, negative):
        for index, group in enumerate(bucket):
            assignment[group] = index % folds
    return assignment


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale[scale < 1e-8] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def fit_logistic(x: np.ndarray, y: np.ndarray, *, l2: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack((np.ones(len(x)), x))

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        logits = np.clip(design @ weights, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        loss = -np.sum(y * np.log(probability + 1e-12) + (1.0 - y) * np.log(1.0 - probability + 1e-12))
        loss += 0.5 * l2 * float(np.sum(weights[1:] ** 2))
        gradient = design.T @ (probability - y)
        gradient[1:] += l2 * weights[1:]
        return float(loss), gradient

    result = minimize(
        lambda weights: objective(weights)[0],
        np.zeros(design.shape[1], dtype=np.float64),
        jac=lambda weights: objective(weights)[1],
        method="L-BFGS-B",
        options={"maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"logistic optimization failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def predict_logistic(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    logits = np.column_stack((np.ones(len(x)), x)) @ weights
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def _decision_stump(train_x: np.ndarray, train_y: np.ndarray) -> tuple[int, float, bool]:
    best = (-1.0, 0, 0.0, True)
    for feature in range(train_x.shape[1]):
        values = np.unique(train_x[:, feature])
        if len(values) < 2:
            continue
        thresholds = 0.5 * (values[:-1] + values[1:])
        if len(thresholds) > 20:
            thresholds = np.quantile(thresholds, np.linspace(0.05, 0.95, 20))
        for threshold in thresholds:
            for positive_above in (True, False):
                prediction = (train_x[:, feature] >= threshold) == positive_above
                tpr = np.mean(prediction[train_y == 1]) if np.any(train_y == 1) else 0.0
                tnr = np.mean(~prediction[train_y == 0]) if np.any(train_y == 0) else 0.0
                balanced = 0.5 * (tpr + tnr)
                if balanced > best[0]:
                    best = (float(balanced), feature, float(threshold), positive_above)
    return best[1], best[2], best[3]


def grouped_cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    *,
    folds: int = 5,
) -> dict[str, Any]:
    assignment = assign_group_folds(groups, y, folds)
    logistic_probability = np.zeros(len(y), dtype=np.float64)
    stump_probability = np.zeros(len(y), dtype=np.float64)
    knn_probability = np.zeros(len(y), dtype=np.float64)
    fold_rows = []
    group_array = np.asarray(groups)
    for fold in range(folds):
        test = np.asarray([assignment[group] == fold for group in groups], dtype=bool)
        train = ~test
        train_x, test_x = _standardize(x[train], x[test])
        weights = fit_logistic(train_x, y[train], l2=2.0)
        logistic_probability[test] = predict_logistic(test_x, weights)

        feature, threshold, positive_above = _decision_stump(train_x, y[train])
        stump_probability[test] = ((test_x[:, feature] >= threshold) == positive_above).astype(float)

        distances = np.sqrt(np.sum((test_x[:, None, :] - train_x[None, :, :]) ** 2, axis=2))
        neighbors = np.argsort(distances, axis=1)[:, : min(7, int(np.sum(train)))]
        knn_probability[test] = np.mean(y[train][neighbors], axis=1)
        fold_rows.append(
            {
                "fold": fold,
                "train_groups": int(len(set(group_array[train]))),
                "test_groups": int(len(set(group_array[test]))),
                "train_rows": int(np.sum(train)),
                "test_rows": int(np.sum(test)),
                "test_positive_ratio": float(np.mean(y[test])),
            }
        )
    return {
        "folds": fold_rows,
        "logistic_probability": logistic_probability,
        "stump_probability": stump_probability,
        "knn_probability": knn_probability,
    }


def threshold_diagnostics(
    probability: np.ndarray, labels: np.ndarray, damage: np.ndarray
) -> list[dict[str, Any]]:
    rows = []
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
        selected = probability >= threshold
        rows.append(
            {
                "threshold": threshold,
                "selected": int(np.sum(selected)),
                "coverage": float(np.mean(selected)),
                "precision": float(np.mean(labels[selected])) if np.any(selected) else 0.0,
                "actual_damage_mean": float(np.mean(damage[selected])) if np.any(selected) else 0.0,
                "actual_damage_median": float(np.median(damage[selected])) if np.any(selected) else 0.0,
                "large_regression_ratio": float(np.mean(damage[selected] <= -0.003)) if np.any(selected) else 0.0,
            }
        )
    return rows


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    focus = [
        row
        for row in rows
        if row["source_experiment"] == "ablation_vertical_high_focus_20260819"
        and row["action"] == "VP_AZ_POS_SMALL"
        and int(row["action_parameters"]["duration_frames"]) == 36
    ]
    if len(focus) != 90 or len({row["state_hash"] for row in focus}) != 30:
        raise RuntimeError("expected 30 target-high states x 3 AZ+ magnitudes")
    state = np.asarray([row["server_observation"] for row in focus], dtype=np.float64)
    action = np.asarray(
        [
            [
                row["action_parameters"]["magnitude_norm"],
                row["action_parameters"]["duration_norm"],
            ]
            for row in focus
        ],
        dtype=np.float64,
    )
    x = np.column_stack((state, action))
    feature_names = [*GUIDANCE_SERVER_FEATURES, "action_magnitude_norm", "action_duration_norm"]
    damage = np.asarray([row["damage_delta"] for row in focus], dtype=np.float64)
    labels = (damage > 1e-6).astype(np.int64)
    groups = [row["state_hash"] for row in focus]
    cv = grouped_cross_validation(x, labels, groups)
    logistic_probability = cv.pop("logistic_probability")
    stump_probability = cv.pop("stump_probability")
    knn_probability = cv.pop("knn_probability")

    univariate = []
    for index, name in enumerate(feature_names):
        positive = x[labels == 1, index]
        negative = x[labels == 0, index]
        pooled = np.sqrt(0.5 * (np.var(positive) + np.var(negative)))
        effect = float((np.mean(positive) - np.mean(negative)) / pooled) if pooled > 1e-12 else 0.0
        correlation = float(np.corrcoef(x[:, index], damage)[0, 1]) if np.std(x[:, index]) > 1e-12 else 0.0
        quantiles = np.unique(np.quantile(x[:, index], [0.0, 0.25, 0.5, 0.75, 1.0]))
        bins = []
        for lower, upper in zip(quantiles[:-1], quantiles[1:], strict=True):
            include_upper = upper == quantiles[-1]
            selected = (x[:, index] >= lower) & (
                (x[:, index] <= upper) if include_upper else (x[:, index] < upper)
            )
            if np.any(selected):
                bins.append(
                    {
                        "lower": float(lower),
                        "upper": float(upper),
                        "count": int(np.sum(selected)),
                        "positive_ratio": float(np.mean(labels[selected])),
                        "damage_mean": float(np.mean(damage[selected])),
                    }
                )
        univariate.append(
            {
                "feature": name,
                "cohens_d": effect,
                "damage_correlation": correlation,
                "bins": bins,
            }
        )
    univariate.sort(key=lambda row: abs(row["cohens_d"]), reverse=True)

    rng = np.random.default_rng(260819)
    base_auc = roc_auc(labels, logistic_probability)
    permutation = []
    for index, name in enumerate(feature_names):
        drops = []
        for _ in range(3):
            permuted_x = x.copy()
            permuted_x[:, index] = rng.permutation(permuted_x[:, index])
            permuted_cv = grouped_cross_validation(permuted_x, labels, groups)
            drops.append(
                base_auc - roc_auc(labels, permuted_cv["logistic_probability"])
            )
        permutation.append({"feature": name, "mean_auc_drop": float(np.mean(drops))})
    permutation.sort(key=lambda row: row["mean_auc_drop"], reverse=True)

    thresholds = threshold_diagnostics(logistic_probability, labels, damage)
    viable = [
        row
        for row in thresholds
        if row["selected"] >= 6
        and row["precision"] >= 0.70
        and row["actual_damage_mean"] > 0.0
        and row["large_regression_ratio"] == 0.0
    ]
    logistic_auc = roc_auc(labels, logistic_probability)
    separable = logistic_auc >= 0.65 and bool(viable)
    return {
        "schema_version": "state_separation_v3.v1",
        "focus": "target_high/VP_AZ_POS_SMALL/duration_36",
        "rows": len(focus),
        "unique_state_groups": len(set(groups)),
        "positive_definition": "damage_delta > 1e-6",
        "positive_ratio": float(np.mean(labels)),
        "damage_mean": float(np.mean(damage)),
        "damage_median": float(np.median(damage)),
        "group_cv": {
            **cv,
            "logistic_auc": logistic_auc,
            "logistic_average_precision": average_precision(labels, logistic_probability),
            "decision_stump_auc": roc_auc(labels, stump_probability),
            "knn_auc": roc_auc(labels, knn_probability),
            "thresholds": thresholds,
        },
        "univariate_separation": univariate,
        "permutation_importance_diagnostic": permutation,
        "current_observation_separable": separable,
        "temporal_recommendation": (
            "CURRENT_42D_SUFFICIENT_FOR_MODEL_PILOT"
            if separable
            else "COLLECT_TEMPORAL_TELEMETRY_BEFORE_MODEL_ESCALATION"
        ),
        "caveats": [
            "30 independent state groups are a pilot-sized sample",
            "permutation ranking and separability metrics both use group out-of-fold predictions",
            "no health or Damage value is present in runtime features",
        ],
    }


def write_outputs(analysis: dict[str, Any], output: Path, report: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    cv = analysis["group_cv"]
    lines = [
        "# State-Conditioned Hybrid v3 Positive/Negative 상태 분리",
        "",
        "## 결론",
        "",
        f"- current 42D 판정: `{analysis['temporal_recommendation']}`",
        f"- target-high/AZ+ rows/state groups: {analysis['rows']} / {analysis['unique_state_groups']}",
        f"- positive ratio (>1e-6): {analysis['positive_ratio']:.2%}",
        f"- group-CV logistic AUC/AP: {cv['logistic_auc']:.4f} / {cv['logistic_average_precision']:.4f}",
        f"- group-CV decision stump/KNN AUC: {cv['decision_stump_auc']:.4f} / {cv['knn_auc']:.4f}",
        "",
        "## 상위 단변량 분리",
        "",
    ]
    for row in analysis["univariate_separation"][:10]:
        lines.append(
            f"- `{row['feature']}`: Cohen d {row['cohens_d']:+.4f}, Damage corr {row['damage_correlation']:+.4f}"
        )
    lines.extend(["", "## 보수적 threshold", ""])
    for row in cv["thresholds"]:
        lines.append(
            f"- p>={row['threshold']:.2f}: coverage {row['coverage']:.2%}, precision {row['precision']:.2%}, Damage mean {row['actual_damage_mean']:+.9f}, large regression {row['large_regression_ratio']:.2%}"
        )
    lines.extend(
        [
            "",
            "30개 state는 pilot 크기이며 같은 state의 세 magnitude는 반드시 같은 CV fold에 묶었다. row random split 결과는 사용하지 않았다.",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze target-high AZ+ state separability")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    analysis = analyze(payload["rows"])
    write_outputs(analysis, args.output, args.report)
    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
