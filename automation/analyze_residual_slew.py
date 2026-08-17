from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze_file(path: Path) -> dict:
    first_actions: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    changes: list[np.ndarray] = []
    previous: np.ndarray | None = None
    windows = 0
    refreshed = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("record_type") != "frame":
                continue
            hybrid = record.get("hybrid") or {}
            gate = hybrid.get("gate") or {}
            if not gate.get("active"):
                previous = None
                continue
            if gate.get("entry"):
                previous = None
                windows += 1
            if not hybrid.get("rl_action_refreshed"):
                continue
            raw = hybrid.get("raw_residual_action")
            if raw is None:
                continue
            action = np.asarray(raw[:3], dtype=np.float64)
            refreshed += 1
            all_actions.append(action)
            if previous is None:
                first_actions.append(action)
            else:
                changes.append(action - previous)
            previous = action
    first = np.abs(np.asarray(first_actions, dtype=np.float64))
    signed = np.asarray(all_actions, dtype=np.float64)
    delta = np.abs(np.asarray(changes, dtype=np.float64))
    return {
        "telemetry": str(path),
        "windows": windows,
        "refreshed_actions": refreshed,
        "first_action_abs_mean": float(first.mean()) if first.size else 0.0,
        "first_action_abs_max": float(first.max()) if first.size else 0.0,
        "action_signed_mean_axis": (
            signed.mean(axis=0).tolist() if signed.size else [0.0, 0.0, 0.0]
        ),
        "action_abs_mean_axis": (
            np.abs(signed).mean(axis=0).tolist()
            if signed.size
            else [0.0, 0.0, 0.0]
        ),
        "successive_delta_abs_mean": float(delta.mean()) if delta.size else 0.0,
        "successive_delta_abs_max": float(delta.max()) if delta.size else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    files = sorted(args.evaluation.glob("case_*/telemetry/*hybrid*.jsonl"))
    result = {"evaluation": str(args.evaluation), "cases": [analyze_file(path) for path in files]}
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
