from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dogfight.ai.tactical_modes import (
    TACTICAL_HOLD_FRAMES,
    TACTICAL_MODES_T1,
    champion_vp_to_local_setpoint,
)
from dogfight.ai.temporal_observation import (
    TEMPORAL_FEATURES,
    TemporalServerObservationBuilder,
)
from dogfight.sim.state_schema import StateIndex


EPSILON = 1e-9
LARGE_REGRESSION_THRESHOLD = 1e-6
EXPECTED_OPTIONS = {
    f"{mode}__d{duration}"
    for mode in TACTICAL_MODES_T1[1:]
    for duration in TACTICAL_HOLD_FRAMES
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_frames(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if '"record_type":"frame"' in line
    ]


def state_from_payload(payload: dict[str, Any]) -> np.ndarray:
    if "body_velocity_m_s" not in payload:
        raise ValueError("v4 primary dataset requires exact packet-visible body velocity")
    state = np.zeros(int(StateIndex.ALT) + 1, dtype=np.float64)
    state[StateIndex.N : StateIndex.D + 1] = payload["position_ned_m"]
    state[StateIndex.ROLL : StateIndex.YAW + 1] = payload["attitude_deg"]
    state[6:9] = payload["body_velocity_m_s"]
    state[StateIndex.KCAS] = float(payload["speed_kcas"])
    state[StateIndex.ALT] = float(payload["altitude_m"])
    return state


def temporal_observation_at_decision(
    frames: list[dict[str, Any]],
    decision_frame: int,
    prefix_snapshot: dict[str, Any],
) -> np.ndarray:
    if decision_frame <= 0 or decision_frame >= len(frames):
        raise ValueError("decision frame must provide a prior state and same-frame BT")
    builder = TemporalServerObservationBuilder()
    current_context: tuple[np.ndarray, np.ndarray] | None = None
    # For events before frame 30 the builder's frozen repeat-first contract
    # pads unavailable history and therefore produces zero deltas for it.
    for frame in range(max(1, decision_frame - 30), decision_frame + 1):
        # Telemetry stores the post-step state for frame k. The action info in
        # row k+1 was computed from that state, so this reconstructs selector
        # context without a one-frame look-ahead.
        state_row = frames[frame - 1]
        action_row = frames[frame]
        own = state_from_payload(state_row["ownship"])
        target = state_from_payload(state_row["target"])
        hybrid = action_row["hybrid"]
        bt_action = np.asarray(hybrid["bt_action"], dtype=np.float32)
        bt_vp = np.asarray(hybrid["bt_vp"], dtype=np.float64)
        base = champion_vp_to_local_setpoint(bt_vp, own)
        vector = builder.build(
            own,
            target,
            bt_action,
            base,
            sim_time_s=frame / 60.0,
            previous_action_id=0,
            action_hold_frames=0,
            gate_elapsed_frames=0,
            gate_active=False,
            minimum_action_hold_frames=30,
            maximum_active_frames=120,
            recent_authority_ratio=1.0,
        )
        if frame == decision_frame:
            current_context = own, target
    assert current_context is not None
    snapshot_own = np.asarray(prefix_snapshot["ownship_server_observable"], dtype=np.float64)
    snapshot_target = np.asarray(prefix_snapshot["target_server_observable"], dtype=np.float64)
    own, target = current_context
    own_observable = np.concatenate((own[:9], own[[StateIndex.KCAS, StateIndex.ALT]]))
    target_observable = np.concatenate(
        (target[:9], target[[StateIndex.KCAS, StateIndex.ALT]])
    )
    if not np.allclose(own_observable, snapshot_own, atol=1e-9, rtol=0.0):
        raise ValueError("event identity mismatch: ownship decision state")
    if not np.allclose(target_observable, snapshot_target, atol=1e-9, rtol=0.0):
        raise ValueError("event identity mismatch: target decision state")
    return vector


def _baseline_run(root: Path, event_id: str) -> Path:
    return root / "runs" / f"{event_id}__BT_DEFAULT"


def records_from_oracle_root(
    root: Path, *, exclusions: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    pairs = json.loads((root / "pairs.json").read_text(encoding="utf-8"))
    by_event: dict[str, list[dict[str, Any]]] = {}
    for row in pairs:
        if row.get("clean"):
            by_event.setdefault(row["event_id"], []).append(row)
    records: list[dict[str, Any]] = []
    for event_id, options in sorted(by_event.items()):
        option_ids = {row["option_id"] for row in options}
        if option_ids != EXPECTED_OPTIONS:
            raise ValueError(f"incomplete Tactical options for {event_id}: {option_ids}")
        run = _baseline_run(root, event_id)
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        prefix = result["ownship_provider_telemetry"]["prefix_snapshot"]
        decision_frame = int(options[0]["decision_frame"])
        if decision_frame <= 30:
            if exclusions is not None:
                exclusions.append(
                    {
                        "event_id": event_id,
                        "reason": "INITIAL_PACKET_HISTORY_NOT_RECORDED",
                        "decision_frame": decision_frame,
                    }
                )
            continue
        observation = temporal_observation_at_decision(
            load_frames(run / "telemetry.jsonl"), decision_frame, prefix
        )
        common = {
            "event_id": event_id,
            "fight_id": options[0]["fight_id"],
            "trajectory_id": options[0]["fight_id"],
            "scenario_id": options[0]["scenario_id"],
            "opponent_id": options[0]["opponent_id"],
            "seed": int(options[0]["seed"]),
            "event_type": options[0]["event_type"],
            "diagnostic_failure_family": options[0]["diagnostic_failure_family"],
            "decision_frame": decision_frame,
            "observation": observation.tolist(),
        }
        records.append(
            {
                **common,
                "option_id": "BT_DEFAULT",
                "mode": "BT_DEFAULT",
                "hold_frames": 0,
                "damage_advantage": 0.0,
                "net_health_margin_advantage": 0.0,
                "positive": False,
                "large_regression": False,
                "horizons": {},
            }
        )
        for row in sorted(options, key=lambda item: item["option_id"]):
            advantage = float(row["terminal"]["damage_dealt_delta"])
            records.append(
                {
                    **common,
                    "option_id": row["option_id"],
                    "mode": row["mode"],
                    "hold_frames": int(row["hold_frames"]),
                    "damage_advantage": advantage,
                    "net_health_margin_advantage": float(
                        row["terminal"]["net_health_margin_delta"]
                    ),
                    "positive": advantage > EPSILON,
                    "large_regression": advantage < -LARGE_REGRESSION_THRESHOLD,
                    "horizons": row["horizons"],
                }
            )
    return records


def validate_group_assignment(records: Iterable[dict[str, Any]], assignment: dict[str, str]) -> None:
    seen: dict[tuple[str, Any], str] = {}
    group_fields = ("fight_id", "trajectory_id", "event_id", "scenario_id", "seed")
    for row in records:
        split = assignment[row["event_id"]]
        for field in group_fields:
            key = field, row[field]
            previous = seen.setdefault(key, split)
            if previous != split:
                raise ValueError(f"group leakage for {field}={row[field]}")


def grouped_split(records: list[dict[str, Any]]) -> dict[str, str]:
    events: dict[str, dict[str, Any]] = {}
    for row in records:
        events.setdefault(row["event_id"], row)
    assignment = {}
    for event_id, row in sorted(events.items()):
        # Scenario is the widest correlated unit. Hashing it assigns every
        # fight/trajectory/event/seed nested under the same geometry together.
        token = f"{row['opponent_id']}|{row['scenario_id']}".encode()
        bucket = int(hashlib.sha256(token).hexdigest()[:8], 16) % 10
        assignment[event_id] = "test" if bucket == 0 else "validation" if bucket < 3 else "train"
    validate_group_assignment(records, assignment)
    return assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build temporal Tactical dataset v4")
    parser.add_argument("--oracle-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Tactical dataset: {output}")
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    source_roots = [path.resolve() for path in args.oracle_root]
    for root in source_roots:
        records.extend(records_from_oracle_root(root, exclusions=exclusions))
    event_ids = [row["event_id"] for row in records if row["mode"] == "BT_DEFAULT"]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("duplicate event identity across Tactical dataset sources")
    assignment = grouped_split(records)
    for row in records:
        row["split"] = assignment[row["event_id"]]
    output.mkdir(parents=True)
    dataset_path = output / "dataset.jsonl"
    dataset_path.write_text(
        "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in records),
        encoding="utf-8",
    )
    split_counts = {
        split: sum(value == split for value in assignment.values())
        for split in ("train", "validation", "test")
    }
    metadata = {
        "schema_version": "temporal_tactical_dataset_v4.v1",
        "observation_contract": "guidance_selector_server_temporal_v4",
        "observation_size": len(TEMPORAL_FEATURES),
        "features": list(TEMPORAL_FEATURES),
        "candidate_modes": list(TACTICAL_MODES_T1),
        "candidate_hold_frames": [0, *TACTICAL_HOLD_FRAMES],
        "unique_events": len(event_ids),
        "state_action_pairs": len(records),
        "clean_nondefault_pairs": len(records) - len(event_ids),
        "split_unit": ["fight", "trajectory", "event", "scenario", "seed-group"],
        "split_event_counts": split_counts,
        "epsilon": EPSILON,
        "large_regression_threshold": LARGE_REGRESSION_THRESHOLD,
        "primary_label": "prefix-replay paired terminal Damage advantage",
        "diagnostic_taxonomy_is_label": False,
        "excluded_events": exclusions,
        "runtime_forbidden": ["health", "Damage", "hidden FDM truth", "offline labels"],
        "source_roots": [str(path.relative_to(ROOT)).replace("\\", "/") for path in source_roots],
        "dataset_sha256": sha256(dataset_path),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
