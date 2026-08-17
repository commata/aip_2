from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from dogfight.ai.hybrid_action_provider import ALLOWED_AIM_RESIDUAL_SCALES
from dogfight.envs.observation import observation_size
from dogfight.envs.observation import OFFICIAL_DAMAGE_PHASES


@dataclass(frozen=True)
class SubmissionConfig:
    source_path: Path
    raw: dict[str, Any]
    observation_mode: str
    observation_size: int
    observation_contract_version: str
    normalization_version: str
    health_source: str
    bundle_path: Path
    policy_id: str
    bt_dll_path: Path
    bt_xml_path: Path
    residual_scale: float
    composition_mode: str
    rl_action_repeat: int
    expected_sim_hz: int
    latency_threshold_s: float
    wez_config: dict[str, Any]
    phase_config: list[dict[str, Any]]


def load_submission_config(
    config_path: str | Path,
    *,
    require_files: bool = True,
) -> SubmissionConfig:
    """Load and fail-fast validate one submission runtime contract."""
    source = Path(config_path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("submission config root must be a JSON object")

    if payload.get("mode") != "bt_residual":
        raise ValueError("submission mode must be 'bt_residual'")
    if payload.get("throttle_policy") != "bt_only":
        raise ValueError("submission throttle_policy must be 'bt_only'")

    bundle_path = _resolve_path(source, payload.get("bundle_path"), "bundle_path")
    bundle_metadata_path = bundle_path / "metadata.json"
    weights_path = bundle_path / "policy_weights.pkl.gz"
    if require_files:
        _require_file(bundle_metadata_path, "bundle metadata")
        _require_file(weights_path, "bundle weights")
    bundle = _read_json(bundle_metadata_path, "bundle metadata")
    bundle_contract = _bundle_observation_contract(bundle)

    requested_mode = str(payload.get("observation_mode", "auto")).strip()
    bundle_mode = bundle_contract["mode"]
    resolved_mode = bundle_mode if requested_mode == "auto" else requested_mode
    if resolved_mode != bundle_mode:
        raise ValueError(
            f"observation mode mismatch: config={resolved_mode!r}, bundle={bundle_mode!r}"
        )
    resolved_size = int(payload.get("observation_size", bundle_contract["size"]))
    if resolved_size != bundle_contract["size"]:
        raise ValueError(
            "observation size mismatch: "
            f"config={resolved_size}, bundle={bundle_contract['size']}"
        )
    builtin_size = observation_size(resolved_mode)
    if resolved_size != builtin_size:
        raise ValueError(
            f"observation schema mismatch: {resolved_mode!r} requires {builtin_size}, "
            f"got {resolved_size}"
        )

    contract_version = _match_contract_text(
        payload,
        bundle_contract,
        "observation_contract_version",
    )
    normalization_version = _match_contract_text(
        payload,
        bundle_contract,
        "normalization_version",
    )
    health_source = _match_contract_text(payload, bundle_contract, "health_source")

    weights_sha = str(payload.get("bundle_sha256", "")).upper()
    if require_files:
        _verify_sha256(weights_path, weights_sha, "bundle")

    bt = payload.get("bt")
    if not isinstance(bt, dict):
        raise ValueError("submission config requires a bt object")
    bt_dll_path = _resolve_path(source, bt.get("dll_path"), "bt.dll_path")
    bt_xml_path = _resolve_path(source, bt.get("xml_path"), "bt.xml_path")
    if require_files:
        _require_file(bt_dll_path, "BT DLL")
        _require_file(bt_xml_path, "BT XML")
        _verify_sha256(bt_dll_path, str(bt.get("dll_sha256", "")), "BT DLL")
        _verify_sha256(bt_xml_path, str(bt.get("xml_sha256", "")), "BT XML")

    residual_scale = float(payload.get("residual_scale"))
    if residual_scale not in ALLOWED_AIM_RESIDUAL_SCALES:
        raise ValueError(
            f"residual_scale must be one of {ALLOWED_AIM_RESIDUAL_SCALES}"
        )
    composition_mode = str(payload.get("composition_mode", ""))
    if composition_mode not in {"additive", "saturation_aware"}:
        raise ValueError("unsupported composition_mode")
    rl_action_repeat = int(payload.get("rl_action_repeat", 0))
    if rl_action_repeat <= 0:
        raise ValueError("rl_action_repeat must be positive")
    expected_sim_hz = int(payload.get("expected_sim_hz", 0))
    if expected_sim_hz != 60:
        raise ValueError("submission expected_sim_hz must be 60")
    latency_threshold_s = float(payload.get("latency_threshold_s", 0.0))
    if latency_threshold_s <= 0.0:
        raise ValueError("latency_threshold_s must be positive")

    wez_config = payload.get("wez")
    if not isinstance(wez_config, dict):
        raise ValueError("submission config requires a wez object")
    _validate_wez(wez_config)
    bundle_wez = bundle_contract.get("wez")
    if bundle_wez and _canonical_json(bundle_wez) != _canonical_json(wez_config):
        raise ValueError("WEZ contract mismatch between config and bundle")
    phase_config = payload.get("phase_config")
    if not isinstance(phase_config, list):
        raise ValueError("submission config requires phase_config")
    _validate_phase_config(phase_config)
    bundle_phase = bundle_contract.get("phase_config")
    if bundle_phase and _canonical_json(bundle_phase) != _canonical_json(phase_config):
        raise ValueError("phase_config mismatch between config and bundle")

    for required_gate in ("hard_eligibility_gate", "activation_gate"):
        if not isinstance(payload.get(required_gate), dict):
            raise ValueError(f"submission config requires {required_gate}")

    return SubmissionConfig(
        source_path=source,
        raw=payload,
        observation_mode=resolved_mode,
        observation_size=resolved_size,
        observation_contract_version=contract_version,
        normalization_version=normalization_version,
        health_source=health_source,
        bundle_path=bundle_path,
        policy_id=str(payload.get("policy_id", "default_policy")),
        bt_dll_path=bt_dll_path,
        bt_xml_path=bt_xml_path,
        residual_scale=residual_scale,
        composition_mode=composition_mode,
        rl_action_repeat=rl_action_repeat,
        expected_sim_hz=expected_sim_hz,
        latency_threshold_s=latency_threshold_s,
        wez_config=dict(wez_config),
        phase_config=[dict(item) for item in phase_config],
    )


def load_bundle_observation_contract(bundle_path: str | Path) -> dict[str, Any]:
    """Read the observation contract without constructing an RLlib module."""
    metadata_path = Path(bundle_path).resolve() / "metadata.json"
    _require_file(metadata_path, "bundle metadata")
    return _bundle_observation_contract(_read_json(metadata_path, "bundle metadata"))


def _bundle_observation_contract(bundle: dict[str, Any]) -> dict[str, Any]:
    metadata = bundle.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    env_config = bundle.get("algorithm_config", {}).get("env_config", {})
    if not isinstance(env_config, dict):
        env_config = {}
    summary = metadata.get("observation_summary")
    if not isinstance(summary, dict):
        summary = env_config.get("observation_summary", {})
    mode = str(
        metadata.get("observation_mode")
        or metadata.get("obs_mode")
        or env_config.get("observation_mode")
        or ""
    )
    if not mode:
        raise ValueError("bundle metadata has no observation mode")
    size = metadata.get("observation_size") or summary.get("size")
    if size is None:
        size = observation_size(mode)
    contract = env_config.get("observation_contract")
    if not isinstance(contract, dict):
        contract = {}
    return {
        "mode": mode,
        "size": int(size),
        "observation_contract_version": metadata.get(
            "observation_contract_version", contract.get("version", "legacy")
        ),
        "normalization_version": metadata.get(
            "normalization_version", contract.get("normalization_version", "legacy")
        ),
        "health_source": metadata.get(
            "health_source", contract.get("health_source", "simulator")
        ),
        "wez": metadata.get("wez_contract") or env_config.get("wez"),
        "phase_config": metadata.get("phase_config") or env_config.get("phase_config"),
    }


def _match_contract_text(
    payload: dict[str, Any], bundle: dict[str, Any], key: str
) -> str:
    configured = str(payload.get(key, "")).strip()
    bundled = str(bundle.get(key, "")).strip()
    if not configured or not bundled:
        raise ValueError(f"missing {key} in config or bundle")
    if configured != bundled:
        raise ValueError(f"{key} mismatch: config={configured!r}, bundle={bundled!r}")
    return configured


def _resolve_path(source: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"submission config requires {label}")
    path = Path(value)
    if not path.is_absolute():
        path = source.parent / path
    return path.resolve()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _verify_sha256(path: Path, expected: str, label: str) -> None:
    expected = expected.strip().upper()
    if len(expected) != 64:
        raise ValueError(f"{label} SHA256 must contain 64 hex characters")
    actual = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected={expected}, actual={actual}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _validate_wez(wez: dict[str, Any]) -> None:
    minimum = float(wez.get("min_range_m", -1.0))
    maximum = float(wez.get("max_range_m", -1.0))
    angle = float(wez.get("angle_deg", -1.0))
    if not 0.0 <= minimum < maximum or angle <= 0.0:
        raise ValueError("invalid WEZ range or angle")


def _validate_phase_config(phases: list[dict[str, Any]]) -> None:
    expected = [dict(item) for item in OFFICIAL_DAMAGE_PHASES]
    if len(phases) != len(expected):
        raise ValueError("phase_config must match the official 100/150/200 second contract")
    for actual, reference in zip(phases, expected):
        if not isinstance(actual, dict):
            raise ValueError("phase_config entries must be objects")
        if int(actual.get("phase", -1)) != int(reference["phase"]):
            raise ValueError("phase_config phase identifiers do not match official rules")
        for key in ("end_s", "half_angle_deg", "max_range_m"):
            if not math.isclose(
                float(actual.get(key, float("nan"))),
                float(reference[key]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "phase_config must match the official 100/150/200 second contract"
                )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
