from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from dogfight.ai.hybrid_action_provider import target_ata_deg
from dogfight.unreal.client import PlaneSnapshot, RemoteClientContext
from dogfight.unreal.policies import ProviderCommandPolicy, plane_info_to_state
from dogfight.unreal.protocol import PlaneInfo, Rotation3D, Vector3D


def build_remote_context(
    own_plane: PlaneInfo,
    target_plane: PlaneInfo,
    *,
    ownship_plane_id: int | None = None,
) -> RemoteClientContext:
    own_snapshot = PlaneSnapshot()
    target_snapshot = PlaneSnapshot()
    own_snapshot.update(own_plane)
    target_snapshot.update(target_plane)
    return RemoteClientContext(
        plane_id=own_plane.plane_id if ownship_plane_id is None else ownship_plane_id,
        frame_index=max(own_plane.index, target_plane.index),
        own_plane=own_snapshot,
        enemy_plane=target_snapshot,
    )


def replay_packet_pairs(
    policy: ProviderCommandPolicy,
    pairs: Iterable[tuple[PlaneInfo, PlaneInfo]],
) -> list[dict]:
    """Replay server-shaped packet pairs through the real submission policy path."""
    records: list[dict] = []
    for own_plane, target_plane in pairs:
        context = build_remote_context(own_plane, target_plane)
        own_state = plane_info_to_state(own_plane)
        target_state = plane_info_to_state(target_plane)
        start = perf_counter()
        command = policy.compute_command(context)
        latency_ms = (perf_counter() - start) * 1000.0
        provider_telemetry = getattr(policy.action_provider, "telemetry", None)
        telemetry = provider_telemetry() if callable(provider_telemetry) else {}
        sim_time_s = policy._sim_time_s(context.frame_index)
        observation = policy._build_observation(
            own_state,
            target_state,
            sim_time_s=sim_time_s,
        )
        records.append(
            {
                "frame_index": context.frame_index,
                "own_plane": asdict(own_plane),
                "target_plane": asdict(target_plane),
                "ownship_state": own_state.tolist(),
                "target_state": target_state.tolist(),
                "target_ata_deg": target_ata_deg(own_state, target_state),
                "sim_time_s": sim_time_s,
                "observation": observation.tolist(),
                "command": asdict(command),
                "latency_ms": latency_ms,
                "provider_last_frame": telemetry.get("last_frame", {}),
            }
        )
    return records


def load_packet_pairs(path: str | Path) -> list[tuple[PlaneInfo, PlaneInfo]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = []
    for item in payload:
        pairs.append((_plane_from_dict(item["own_plane"]), _plane_from_dict(item["target_plane"])))
    return pairs


def _plane_from_dict(value: dict) -> PlaneInfo:
    return PlaneInfo(
        index=int(value["index"]),
        plane_id=int(value["plane_id"]),
        position=Vector3D(**value["position"]),
        rotation=Rotation3D(**value["rotation"]),
        velocity=Vector3D(**value["velocity"]),
    )


def latency_summary(records: list[dict], threshold_ms: float = 166.7) -> dict:
    values = np.asarray([item["latency_ms"] for item in records], dtype=np.float64)
    if not values.size:
        return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0, "over_threshold_ratio": 0.0}
    return {
        "count": int(values.size),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
        "over_threshold_ratio": float(np.mean(values > threshold_ms)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay Unreal PlaneInfo packet pairs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(
        "이 모듈은 제출 policy를 구성한 검증 스크립트에서 import해 사용합니다. "
        "단독 CLI는 policy bundle/BT config 결합 후 제공됩니다."
    )
