from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def expand_matrix(payload: dict) -> list[tuple[str, dict]]:
    matrix = payload.get("pilot_matrix") or {}
    seeds = list(matrix.get("seeds") or [])
    observations = list(matrix.get("observations") or [])
    if len(seeds) < 2:
        raise ValueError("pilot_matrix.seeds must contain at least two independent seeds")
    if not observations:
        raise ValueError("pilot_matrix.observations must not be empty")

    expanded: list[tuple[str, dict]] = []
    for observation in observations:
        mode = str(observation["mode"])
        label = str(observation["label"])
        contract = dict(observation.get("contract") or {})
        for seed in seeds:
            item = deepcopy(payload)
            item.pop("pilot_matrix", None)
            tag = f"rear120_{label}_s{int(seed)}"
            item["name"] = f"0815_submission_{tag}"
            item.setdefault("output", {})["tag"] = tag
            item.setdefault("env", {})["observation_mode"] = mode
            item.setdefault("env_config", {})["observation_mode"] = mode
            item["env_config"]["observation_contract"] = contract
            item.setdefault("runtime", {})["seed"] = int(seed)
            item["notes"] = (
                "Rear120 R10/T16 same-budget pilot. "
                f"Only observation contract and independent seed vary: {mode}, {seed}."
            )
            expanded.append((tag, item))
    return expanded


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="동일 budget 10D/Tactical16 pilot matrix 실행")
    value.add_argument("experiment_yaml", type=Path)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--only", action="append", default=[])
    return value


def main() -> int:
    args = parser().parse_args()
    source = args.experiment_yaml.resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    generated_root = ROOT / "artifacts" / "experiment_matrix" / source.stem
    generated_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.only)

    for tag, experiment in expand_matrix(payload):
        if selected and tag not in selected:
            continue
        generated = generated_root / f"{tag}.yaml"
        generated.write_text(
            yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_experiment.py"),
            str(generated),
        ]
        print(f"[pilot-matrix] {tag}: {generated}", flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode:
            return int(completed.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
