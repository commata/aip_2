"""Restartable scale/gate search built on deterministic paired evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "automation" / "offensive_optimization.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def build_trials(config: dict[str, Any]) -> list[dict[str, Any]]:
    default_gate = config["gate_candidates"][0]
    trials = [
        {
            "name": "scale_grid",
            "phase": "development_scale",
            "gate": default_gate,
            "scales": list(config["scales"]),
            "seeds": list(config["development_seeds"]),
            "max_engage_time": config["development_max_engage_time"],
        }
    ]
    for gate in config["gate_candidates"]:
        trials.append(
            {
                "name": f"gate_{gate['name']}",
                "phase": "development_gate",
                "gate": gate,
                "scales": None,
                "seeds": list(config["development_seeds"]),
                "max_engage_time": config["development_max_engage_time"],
            }
        )
    trials.append(
        {
            "name": "heldout_validation",
            "phase": "heldout",
            "gate": None,
            "scales": None,
            "seeds": list(config["heldout_seeds"]),
            "max_engage_time": config["heldout_max_engage_time"],
        }
    )
    return trials


def _best_scale(best: dict[str, Any] | None, fallback: float = 0.15) -> float:
    if not best:
        return fallback
    controller = str(best.get("controller", ""))
    try:
        return float(controller.split("_", 1)[1])
    except (IndexError, ValueError):
        return fallback


def command_for_trial(config, trial, trial_output: Path, best: dict[str, Any] | None, max_pairs: int) -> list[str]:
    gate = trial["gate"] or (best or {}).get("gate") or config["gate_candidates"][0]
    scales = trial["scales"] or [_best_scale(best)]
    cmd = [
        sys.executable,
        str(ROOT / "automation" / "evaluate_offensive_hybrid.py"),
        "--bundle", str(ROOT / config["bundle"]),
        "--output", str(trial_output),
        "--ownship-bt-dll", str(ROOT / config["ownship_bt_dll"]),
        "--target-bt-dll", str(ROOT / config["target_bt_dll"]),
        "--bt-rule-xml", str(ROOT / config["bt_rule_xml"]),
        "--scenarios", *[str(ROOT / path) for path in config["scenarios"]],
        "--seeds", *[str(seed) for seed in trial["seeds"]],
        "--scales", *[str(scale) for scale in scales],
        "--max-engage-time", str(trial["max_engage_time"]),
        "--episode-step-limit", str(max(1, int(trial["max_engage_time"] * 60))),
        "--timeout-seconds", str(config["timeout_seconds"]),
        "--offensive-enter-range-m", str(gate["enter_range_m"]),
        "--offensive-exit-range-m", str(gate["exit_range_m"]),
        "--offensive-enter-ata-deg", str(gate["enter_ata_deg"]),
        "--offensive-exit-ata-deg", str(gate["exit_ata_deg"]),
        "--offensive-enter-target-ata-deg", str(gate["enter_target_ata_deg"]),
        "--offensive-exit-target-ata-deg", str(gate["exit_target_ata_deg"]),
        "--resume",
    ]
    if max_pairs > 0:
        cmd += ["--max-pairs", str(max_pairs)]
    return cmd


def candidate_from_evaluation(evaluation: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any] | None:
    candidate = evaluation.get("summary", {}).get("best_valid_candidate")
    if candidate is None:
        return None
    return {**candidate, "gate": trial["gate"], "trial": trial["name"], "phase": trial["phase"]}


def is_better(candidate: dict[str, Any] | None, best: dict[str, Any] | None) -> bool:
    if candidate is None:
        return False
    if best is None:
        return True
    return float(candidate.get("score", -float("inf"))) > float(best.get("score", -float("inf")))


def write_report(output: Path, state: dict[str, Any]) -> None:
    best = state.get("best")
    lines = [
        "# Offensive Hybrid Optimization",
        "",
        f"- Status: `{state['status']}`",
        f"- Completed iterations: `{state['next_iteration']}`",
        f"- Best: `{best}`",
        "",
        "## History",
        "",
    ]
    for item in state.get("history", []):
        lines.append(f"- `{item['iteration']}` `{item['trial']}`: returncode={item['returncode']}, candidate={item.get('candidate')}")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state.json"
    if args.resume and state_path.is_file():
        state = load_json(state_path)
    else:
        state = {"status": "running", "next_iteration": 0, "best": None, "history": [], "started_at": datetime.now(timezone.utc).isoformat()}
    trials = build_trials(config)
    max_iterations = min(args.max_iterations or config["maximum_iterations"], len(trials))
    wall_budget = args.max_wall_seconds or config["maximum_wall_seconds"]
    started = time.monotonic()

    while state["next_iteration"] < max_iterations:
        if time.monotonic() - started >= wall_budget:
            state["status"] = "budget_exhausted"
            break
        index = int(state["next_iteration"])
        trial = trials[index]
        if trial["phase"] == "heldout" and state.get("best") is None:
            state["status"] = "no_safe_candidate"
            break
        trial_output = output / "iterations" / f"{index:02d}_{trial['name']}"
        cmd = command_for_trial(config, trial, trial_output, state.get("best"), args.max_pairs)
        print(f"[optimize] iteration={index} trial={trial['name']}", flush=True)
        process = subprocess.run(cmd, cwd=ROOT, check=False)
        evaluation_path = trial_output / "evaluation.json"
        evaluation = load_json(evaluation_path) if evaluation_path.is_file() else {}
        candidate = candidate_from_evaluation(evaluation, trial)
        if trial["phase"] == "heldout":
            if candidate is not None:
                state["heldout"] = candidate
        elif is_better(candidate, state.get("best")):
            state["best"] = candidate
            atomic_json(output / "best_config.json", candidate)
        history_item = {
            "iteration": index,
            "trial": trial["name"],
            "phase": trial["phase"],
            "returncode": process.returncode,
            "candidate": candidate,
            "evaluation": str(evaluation_path),
        }
        state["history"].append(history_item)
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(history_item, sort_keys=True) + "\n")
        state["next_iteration"] = index + 1
        atomic_json(state_path, state)
        write_report(output, state)
    if state["next_iteration"] >= len(trials):
        state["status"] = "complete"
    elif state["next_iteration"] >= max_iterations and state["status"] == "running":
        state["status"] = "iteration_budget_exhausted"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(state_path, state)
    write_report(output, state)
    print(json.dumps({"status": state["status"], "best": state.get("best"), "heldout": state.get("heldout")}, indent=2))


if __name__ == "__main__":
    main()
