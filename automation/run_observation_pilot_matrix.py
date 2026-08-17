from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_experiment_payload(source: Path, seen: set[Path] | None = None) -> dict:
    source = source.resolve()
    seen = set() if seen is None else set(seen)
    if source in seen:
        raise ValueError(f"cyclic base_experiment reference: {source}")
    seen.add(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    base_reference = payload.pop("base_experiment", None)
    if not base_reference:
        return payload
    base_path = Path(base_reference)
    if not base_path.is_absolute():
        sibling = source.parent / base_path
        base_path = sibling if sibling.is_file() else ROOT / base_path
    base = load_experiment_payload(base_path, seen)
    return _deep_merge(base, payload)


def expand_matrix(payload: dict) -> list[tuple[str, dict]]:
    matrix = payload.get("pilot_matrix") or {}
    seeds = list(matrix.get("seeds") or [])
    observations = list(matrix.get("observations") or [])
    seed_overrides = dict(matrix.get("seed_overrides") or {})
    run_suffix = str(matrix.get("run_suffix") or "").strip()
    matrix_notes = str(matrix.get("notes") or "").strip()
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
            override = dict(seed_overrides.get(str(int(seed))) or {})
            if override:
                item = _deep_merge(item, override)
            tag = f"rear120_{label}_s{int(seed)}"
            if run_suffix:
                tag = f"{tag}_{run_suffix}"
            item["name"] = f"0815_submission_{tag}"
            item.setdefault("output", {})["tag"] = tag
            item.setdefault("env", {})["observation_mode"] = mode
            item.setdefault("env_config", {})["observation_mode"] = mode
            item["env_config"]["observation_contract"] = contract
            item.setdefault("runtime", {})["seed"] = int(seed)
            item["notes"] = matrix_notes or (
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
    payload = load_experiment_payload(source)
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
