from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from dogfight.ai.guidance_selector import (
    GUIDANCE_ACTIONS,
    GUIDANCE_SELECTOR_CONTRACT_VERSION,
    GUIDANCE_SELECTOR_FEATURES,
    GUIDANCE_SELECTOR_NORMALIZATION_VERSION,
    GUIDANCE_SELECTOR_OBSERVATION_SIZE,
    GuidanceActionConfig,
    GuidanceControllerConfig,
    GuidanceRuntimeConfig,
)
from dogfight.envs.observation import OFFICIAL_DAMAGE_PHASES


@dataclass(frozen=True)
class GuidanceSubmissionConfig:
    source_path: Path
    raw: dict[str, Any]
    bundle_path: Path
    model_path: Path
    bt_dll_path: Path
    bt_xml_path: Path
    policy_id: str
    action_config: GuidanceActionConfig
    controller_config: GuidanceControllerConfig
    runtime_config: GuidanceRuntimeConfig
    rear120_config: dict[str, Any]
    aim_config: dict[str, Any]
    offensive_config: dict[str, Any]
    safety_config: dict[str, Any]
    expected_sim_hz: int
    latency_threshold_s: float
    wez_config: dict[str, Any]
    phase_config: list[dict[str, Any]]
    health_source: str


def load_guidance_submission_config(
    config_path: str | Path,
    *,
    require_files: bool = True,
) -> GuidanceSubmissionConfig:
    source = Path(config_path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Guidance submission config root must be an object")
    if payload.get("mode") != "guidance_selector":
        raise ValueError("Guidance submission mode must be 'guidance_selector'")
    if payload.get("status") not in {
        "SUBMISSION_READY_HYBRID_CANDIDATE",
        "PROMOTED_LOCAL_GUIDANCE_HYBRID",
    }:
        raise ValueError("Guidance submission status is not runnable")
    if payload.get("throttle_policy") != "bt_only":
        raise ValueError("Guidance submission throttle_policy must be bt_only")
    if payload.get("fallback_mode") != "exact_pure_bt":
        raise ValueError("Guidance fallback_mode must be exact_pure_bt")
    if payload.get("selector_observation_contract") != GUIDANCE_SELECTOR_CONTRACT_VERSION:
        raise ValueError("Guidance selector observation contract mismatch")
    if payload.get("normalization_version") != GUIDANCE_SELECTOR_NORMALIZATION_VERSION:
        raise ValueError("Guidance normalization version mismatch")
    if int(payload.get("selector_observation_size", -1)) != GUIDANCE_SELECTOR_OBSERVATION_SIZE:
        raise ValueError("Guidance selector observation size mismatch")
    if tuple(payload.get("observation_features", ())) != GUIDANCE_SELECTOR_FEATURES:
        raise ValueError("Guidance selector feature order mismatch")
    if tuple(payload.get("action_library", ())) != GUIDANCE_ACTIONS:
        raise ValueError("Guidance action library mismatch")
    if payload.get("runtime_observation_mode") != "tactical16":
        raise ValueError("Guidance runtime source observation must be tactical16")

    bundle_path = _resolve(source, payload.get("bundle_path"), "bundle_path")
    model_path = bundle_path / "model.npz"
    metadata_path = bundle_path / "metadata.json"
    if require_files:
        _require_file(model_path, "Guidance model")
        _require_file(metadata_path, "Guidance metadata")
        _verify_hash(model_path, payload.get("bundle_sha256"), "Guidance model")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model_sha256", "").upper() != str(
            payload.get("bundle_sha256", "")
        ).upper():
            raise ValueError("Guidance metadata/config model hash mismatch")

    bt = payload.get("bt")
    if not isinstance(bt, dict):
        raise ValueError("Guidance config requires bt object")
    bt_dll = _resolve(source, bt.get("dll_path"), "bt.dll_path")
    bt_xml = _resolve(source, bt.get("xml_path"), "bt.xml_path")
    if require_files:
        _require_file(bt_dll, "BT DLL")
        _require_file(bt_xml, "BT XML")
        _verify_hash(bt_dll, bt.get("dll_sha256"), "BT DLL")
        _verify_hash(bt_xml, bt.get("xml_sha256"), "BT XML")

    action_config = GuidanceActionConfig(**_object(payload, "action_magnitude"))
    controller_config = GuidanceControllerConfig(**_object(payload, "controller"))
    runtime = _object(payload, "runtime")
    runtime_config = GuidanceRuntimeConfig(**runtime)
    action_config.validate()
    controller_config.validate()
    runtime_config.validate()
    if int(payload.get("expected_sim_hz", 0)) != 60 or runtime_config.sim_hz != 60:
        raise ValueError("Guidance submission requires 60Hz")
    latency_threshold = float(payload.get("latency_threshold_s", 0.0))
    if not math.isclose(
        latency_threshold,
        runtime_config.inference_timeout_s,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Guidance latency threshold/runtime timeout mismatch")

    hard_gate = _object(payload, "hard_eligibility_gate")
    if hard_gate.get("kind") != "rear120":
        raise ValueError("Guidance hard eligibility gate must be rear120")
    rear120 = dict(hard_gate)
    rear120.pop("kind", None)
    activation = _object(payload, "activation_gate")
    if activation.get("kind") != "rear120_and_offensive_or_pre_aim":
        raise ValueError("unsupported Guidance activation gate")
    aim = _object(activation, "phase_pre_aim")
    offensive = _object(activation, "offensive")
    safety = _object(activation, "safety_veto")

    phase_config = payload.get("phase_config")
    if not isinstance(phase_config, list):
        raise ValueError("Guidance config requires phase_config")
    _validate_phases(phase_config)
    wez = _object(payload, "wez")
    if not 0.0 <= float(wez.get("min_range_m", -1)) < float(wez.get("max_range_m", -1)):
        raise ValueError("invalid Guidance WEZ range")
    if float(wez.get("angle_deg", -1)) <= 0.0:
        raise ValueError("invalid Guidance WEZ angle")
    health_source = str(payload.get("health_source", "")).strip()
    if health_source not in {"simulator", "unavailable_constant_one"}:
        raise ValueError("unsupported Guidance health source")
    return GuidanceSubmissionConfig(
        source_path=source,
        raw=payload,
        bundle_path=bundle_path,
        model_path=model_path,
        bt_dll_path=bt_dll,
        bt_xml_path=bt_xml,
        policy_id=str(payload.get("policy_id", "guidance_selector_bc")),
        action_config=action_config,
        controller_config=controller_config,
        runtime_config=runtime_config,
        rear120_config=rear120,
        aim_config=aim,
        offensive_config=offensive,
        safety_config=safety,
        expected_sim_hz=60,
        latency_threshold_s=latency_threshold,
        wez_config=wez,
        phase_config=[dict(item) for item in phase_config],
        health_source=health_source,
    )


def submission_config_mode(path: str | Path) -> str:
    payload = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("submission config root must be an object")
    return str(payload.get("mode", ""))


def _object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Guidance config requires {key} object")
    return dict(value)


def _resolve(source: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Guidance config requires {label}")
    path = Path(value)
    if not path.is_absolute():
        path = source.parent / path
    return path.resolve()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _verify_hash(path: Path, expected: Any, label: str) -> None:
    expected_text = str(expected or "").strip().upper()
    if len(expected_text) != 64:
        raise ValueError(f"{label} SHA256 must contain 64 hex characters")
    actual = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    if actual != expected_text:
        raise ValueError(f"{label} SHA256 mismatch: expected={expected_text}, actual={actual}")


def _validate_phases(phases: list[dict[str, Any]]) -> None:
    if len(phases) != len(OFFICIAL_DAMAGE_PHASES):
        raise ValueError("Guidance phase_config must match official phases")
    for actual, expected in zip(phases, OFFICIAL_DAMAGE_PHASES):
        if int(actual.get("phase", -1)) != int(expected["phase"]):
            raise ValueError("Guidance phase id mismatch")
        for key in ("end_s", "half_angle_deg", "max_range_m"):
            if not math.isclose(
                float(actual.get(key, float("nan"))),
                float(expected[key]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("Guidance phase_config mismatch")
