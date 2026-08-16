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
    pool_ref = env_section.get("target_profile_pool")
    if profile_ref not in (None, "") and pool_ref not in (None, "", []):
        raise TargetProfileError(
            "target_profile and target_profile_pool are mutually exclusive"
        )
    if pool_ref not in (None, "", []):
        pool = resolve_target_profile_pool(
            pool_ref,
            environ=environ,
            profile_dir=profile_dir,
            verify_hashes=verify_hashes,
        )
        env_section["target_mode"] = "profile_curriculum"
        env_section["target_profile_curriculum"] = pool
        bt_profiles = [item for item in pool if item["backend_type"] == "behavior_tree"]
        if bt_profiles:
            xml_hashes = {item["xml_sha256"] for item in bt_profiles}
            if len(xml_hashes) != 1:
                raise TargetProfileError(
                    "profile curriculum currently requires identical BT XML hashes"
                )
            env_section["target_rule_xml"] = bt_profiles[0]["xml_path"]
            own_aliases = set(env_section.get("bt_rule_aliases", []))
            env_section["target_rule_aliases"] = sorted(
                {
                    alias
                    for item in bt_profiles
                    for alias in item.get("rule_aliases", [])
                }
                - own_aliases
            )
        env_section.pop("target_behavior_dll", None)
        return merged, {"profile_pool": pool}
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


def resolve_target_profile_pool(
    pool: Any,
    *,
    environ: Mapping[str, str] | None = None,
    profile_dir: Path = PROFILE_DIR,
    verify_hashes: bool = True,
) -> list[dict[str, Any]]:
    """Resolve a weighted training pool to identity-free simulator settings."""
    if not isinstance(pool, list) or not pool:
        raise TargetProfileError("target_profile_pool must be a non-empty list")
    resolved_pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_item in enumerate(pool):
        if isinstance(raw_item, str):
            profile_ref = raw_item
            weight = 1.0
        elif isinstance(raw_item, dict):
            profile_ref = raw_item.get("profile") or raw_item.get("profile_id")
            weight = raw_item.get("weight", 1.0)
        else:
            raise TargetProfileError(
                f"target_profile_pool[{index}] must be a profile name or object"
            )
        if not profile_ref:
            raise TargetProfileError(
                f"target_profile_pool[{index}] requires profile"
            )
        try:
            weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise TargetProfileError(
                f"target_profile_pool[{index}] weight must be numeric"
            ) from exc
        if weight <= 0:
            raise TargetProfileError(
                f"target_profile_pool[{index}] weight must be positive"
            )
        profile = load_target_profile(
            str(profile_ref),
            environ=environ,
            profile_dir=profile_dir,
            verify_hashes=verify_hashes,
        )
        profile_id = str(profile["profile_id"])
        if profile_id in seen:
            raise TargetProfileError(f"duplicate target profile in pool: {profile_id}")
        if not bool(profile["use"].get("training", False)):
            raise TargetProfileError(
                f"target profile is not approved for training: {profile_id}"
            )
        seen.add(profile_id)
        item: dict[str, Any] = {
            "profile_id": profile_id,
            "backend_type": profile["backend_type"],
            "behavior_cluster": str(profile["behavior_cluster"]),
            "weight": weight,
            "profile_path": profile["profile_path"],
        }
        if profile["backend_type"] == "behavior_tree":
            item.update(
                {
                    "dll_path": profile["dll"]["resolved_path"],
                    "dll_sha256": profile["dll"]["actual_sha256"],
                    "xml_path": profile["xml"]["resolved_path"],
                    "xml_sha256": profile["xml"]["actual_sha256"],
                    "rule_aliases": list(profile.get("rule_aliases", [])),
                }
            )
        resolved_pool.append(item)
    return resolved_pool


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
