from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


PROFILE_DIR = Path(__file__).resolve().parent
SUPPORTED_BACKENDS = {"autopilot", "behavior_tree"}


class TargetProfileError(ValueError):
    """Raised when a proxy target profile cannot be resolved safely."""


def load_target_profile(
    profile: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    profile_dir: Path = PROFILE_DIR,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Load, resolve environment overrides, and verify one target profile."""
    path = _resolve_profile_path(profile, profile_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TargetProfileError(f"cannot load target profile {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TargetProfileError("target profile root must be an object")

    required = (
        "profile_id",
        "backend_type",
        "source",
        "smoke_status",
        "behavior_cluster",
        "use",
    )
    missing = [name for name in required if data.get(name) in (None, "")]
    if missing:
        raise TargetProfileError(f"target profile missing fields: {', '.join(missing)}")
    if data["backend_type"] not in SUPPORTED_BACKENDS:
        raise TargetProfileError(
            f"unsupported target backend_type: {data['backend_type']!r}"
        )
    use = data["use"]
    if not isinstance(use, dict) or any(
        key not in use for key in ("training", "validation", "held_out")
    ):
        raise TargetProfileError(
            "target profile use must define training, validation, and held_out"
        )

    resolved = deepcopy(data)
    resolved["profile_path"] = str(path.resolve())
    env = os.environ if environ is None else environ
    if data["backend_type"] == "behavior_tree":
        for key in ("dll", "xml"):
            artifact = data.get(key)
            if not isinstance(artifact, dict):
                raise TargetProfileError(f"behavior_tree profile requires {key} object")
            resolved[key] = _resolve_artifact(
                artifact,
                environment=env,
                verify_hash=verify_hashes,
                label=key,
            )
    else:
        resolved["dll"] = None
        resolved["xml"] = None
    return resolved


def apply_target_profile(
    experiment: dict[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    profile_dir: Path = PROFILE_DIR,
    verify_hashes: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Inject a profile into an experiment without exposing identity to observation."""
    merged = deepcopy(experiment)
    env_section = merged.setdefault("env", {})
    if not isinstance(env_section, dict):
        raise TargetProfileError("experiment env must be an object")
    profile_ref = env_section.get("target_profile")
    if profile_ref in (None, ""):
        return merged, None
    profile = load_target_profile(
        str(profile_ref),
        environ=environ,
        profile_dir=profile_dir,
        verify_hashes=verify_hashes,
    )
    env_section["target_mode"] = profile["backend_type"]
    if profile["backend_type"] == "behavior_tree":
        env_section["target_behavior_dll"] = profile["dll"]["resolved_path"]
        env_section["target_rule_xml"] = profile["xml"]["resolved_path"]
        env_section["target_rule_aliases"] = list(profile.get("rule_aliases", []))
    else:
        env_section.pop("target_behavior_dll", None)
        env_section.pop("target_rule_xml", None)
        env_section.pop("target_rule_aliases", None)
    return merged, profile


def _resolve_profile_path(profile: str | Path, profile_dir: Path) -> Path:
    requested = Path(profile)
    candidates = [requested]
    if requested.suffix.lower() != ".json":
        candidates.append(profile_dir / f"{requested.name}.json")
    candidates.append(profile_dir / requested)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise TargetProfileError(f"target profile not found: {profile}")


def _resolve_artifact(
    artifact: dict[str, Any],
    *,
    environment: Mapping[str, str],
    verify_hash: bool,
    label: str,
) -> dict[str, Any]:
    path_env = artifact.get("path_env")
    raw_path = environment.get(str(path_env)) if path_env else None
    if not raw_path:
        raw_path = artifact.get("default_path")
    if not raw_path:
        raise TargetProfileError(f"{label} requires default_path or {path_env} override")
    path = Path(str(raw_path)).expanduser().resolve()
    if not path.is_file():
        raise TargetProfileError(f"{label} file not found: {path}")
    expected = str(artifact.get("sha256", "")).upper()
    if len(expected) != 64:
        raise TargetProfileError(f"{label} sha256 must contain 64 hex characters")
    actual = _sha256(path)
    if verify_hash and actual != expected:
        raise TargetProfileError(
            f"{label} sha256 mismatch for {path}: expected {expected}, actual {actual}"
        )
    resolved = deepcopy(artifact)
    resolved["resolved_path"] = str(path)
    resolved["actual_sha256"] = actual
    resolved["path_source"] = f"env:{path_env}" if raw_path == environment.get(str(path_env)) else "default"
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()

