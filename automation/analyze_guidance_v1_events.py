from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "artifacts"
    / "evaluations"
    / "guidance_selector"
    / "full_200s_v1_20260819"
)
DEFAULT_EVIDENCE = (
    ROOT / "automation" / "evidence" / "guidance_advantage_v2" / "phase_a_analysis.json"
)
DEFAULT_REPORT = ROOT / "automation" / "reports" / "GuidanceHybrid_v2_원인분석.md"
HORIZONS_S = (0.5, 1.0, 2.0, 4.0, 8.0)


def read_frames(path: Path) -> list[dict[str, Any]]:
    frames = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("record_type") == "frame":
                frames.append(payload)
    return frames


def intervention_events(frames: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group contiguous non-default frames into independent intervention events."""
    events: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for frame in frames:
        hybrid = frame.get("hybrid") or {}
        action = str(hybrid.get("selected_action", "BT_DEFAULT"))
        active = action != "BT_DEFAULT"
        frame_index = int(frame["frame"])
        if active and (
            current is None
            or current["action"] != action
            or frame_index != current["end_frame"] + 1
        ):
            if current is not None:
                events.append(current)
            current = {
                "action": action,
                "start_frame": frame_index,
                "end_frame": frame_index,
            }
        elif active and current is not None:
            current["end_frame"] = frame_index
        elif current is not None:
            events.append(current)
            current = None
    if current is not None:
        events.append(current)
    for event_id, event in enumerate(events, start=1):
        event["intervention_event_id"] = event_id
        event["duration_frames"] = event["end_frame"] - event["start_frame"] + 1
    return events


def _snapshot(frame: dict[str, Any]) -> dict[str, Any]:
    hybrid = frame.get("hybrid") or {}
    controller = hybrid.get("controller") or {}
    bt = np.asarray(hybrid.get("bt_action", frame.get("ownship_action", [0, 0, 0, 0])), dtype=float)
    aim_az = float(frame.get("aim_azimuth_deg", 0.0))
    aim_el = float(frame.get("aim_elevation_deg", 0.0))
    los_az_rate = float(frame.get("los_azimuth_rate_deg_s", 0.0))
    los_el_rate = float(frame.get("los_elevation_rate_deg_s", 0.0))
    return {
        "frame": int(frame["frame"]),
        "sim_time_s": float(frame.get("sim_time_s", 0.0)),
        "distance_m": float(frame.get("distance_m", 0.0)),
        "signed_aim_azimuth_deg": aim_az,
        "signed_aim_elevation_deg": aim_el,
        "aim_error_deg": float(np.hypot(aim_az, aim_el)),
        "los_azimuth_rate_deg_s": los_az_rate,
        "los_elevation_rate_deg_s": los_el_rate,
        "los_rate_deg_s": float(np.hypot(los_az_rate, los_el_rate)),
        "closing_rate_m_s": float(frame.get("closing_rate_m_s", 0.0)),
        "in_wez": bool(frame.get("in_wez", False)),
        "ownship_damage": float(frame.get("ownship_damage", 0.0)),
        "target_damage": float(frame.get("target_damage", 0.0)),
        "health_margin": float(frame.get("target_damage", 0.0))
        - float(frame.get("ownship_damage", 0.0)),
        "bt_roll": float(bt[0]),
        "bt_pitch": float(bt[1]),
        "bt_yaw": float(bt[2]),
        "positive_headroom": controller.get("positive_headroom"),
        "negative_headroom": controller.get("negative_headroom"),
        "requested_surface_correction": controller.get("requested_surface_correction"),
        "applied_surface_correction": controller.get("applied_surface_correction"),
    }


def _at_or_after(frames: list[dict[str, Any]], frame_index: int) -> dict[str, Any]:
    if not frames:
        raise ValueError("telemetry has no frame records")
    bounded = min(max(frame_index, int(frames[0]["frame"])), int(frames[-1]["frame"]))
    return frames[bounded - int(frames[0]["frame"])]


def _delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "health_margin": candidate["health_margin"] - baseline["health_margin"],
        "target_damage": candidate["target_damage"] - baseline["target_damage"],
        "ownship_damage": candidate["ownship_damage"] - baseline["ownship_damage"],
        "aim_error_deg": candidate["aim_error_deg"] - baseline["aim_error_deg"],
        "los_rate_deg_s": candidate["los_rate_deg_s"] - baseline["los_rate_deg_s"],
        "cone": float(candidate["in_wez"]) - float(baseline["in_wez"]),
    }


def enrich_event(
    event: dict[str, Any],
    candidate_frames: list[dict[str, Any]],
    baseline_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    start = int(event["start_frame"])
    end = int(event["end_frame"])
    enriched = dict(event)
    enriched["start"] = _snapshot(_at_or_after(candidate_frames, start))
    enriched["end"] = _snapshot(_at_or_after(candidate_frames, end))
    enriched["paired_bt_default_start"] = _snapshot(_at_or_after(baseline_frames, start))
    enriched["paired_bt_default_end"] = _snapshot(_at_or_after(baseline_frames, end))
    enriched["paired_delta_at_end"] = _delta(
        enriched["end"], enriched["paired_bt_default_end"]
    )
    horizons = {}
    for seconds in HORIZONS_S:
        offset = int(round(seconds * 60.0))
        candidate = _snapshot(_at_or_after(candidate_frames, end + offset))
        baseline = _snapshot(_at_or_after(baseline_frames, end + offset))
        horizons[f"+{seconds:g}s"] = {
            "candidate": candidate,
            "paired_bt_default": baseline,
            "paired_delta": _delta(candidate, baseline),
        }
    candidate_terminal = _snapshot(candidate_frames[-1])
    baseline_terminal = _snapshot(baseline_frames[-1])
    horizons["terminal"] = {
        "candidate": candidate_terminal,
        "paired_bt_default": baseline_terminal,
        "paired_delta": _delta(candidate_terminal, baseline_terminal),
    }
    enriched["outcomes"] = horizons
    return enriched


def analyze(input_root: Path) -> dict[str, Any]:
    case_records = []
    all_events = []
    for case_root in sorted(path for path in input_root.iterdir() if path.is_dir()):
        candidate_path = case_root / "bc.telemetry.jsonl"
        baseline_path = case_root / "bt_default.telemetry.jsonl"
        if not candidate_path.exists() or not baseline_path.exists():
            continue
        candidate_frames = read_frames(candidate_path)
        baseline_frames = read_frames(baseline_path)
        raw_events = intervention_events(candidate_frames)
        events = [
            {
                "case_id": case_root.name,
                **enrich_event(event, candidate_frames, baseline_frames),
            }
            for event in raw_events
        ]
        for event in events:
            event["local_event_index"] = event["intervention_event_id"]
            event["intervention_event_id"] = (
                f"{case_root.name}:event_{event['local_event_index']:03d}"
            )
        all_events.extend(events)
        gate_entries = sum(
            int(bool((frame.get("hybrid") or {}).get("gate", {}).get("entry")))
            for frame in candidate_frames
        )
        gate_active_frames = sum(
            int(bool((frame.get("hybrid") or {}).get("gate", {}).get("active")))
            for frame in candidate_frames
        )
        case_records.append(
            {
                "case_id": case_root.name,
                "frames": len(candidate_frames),
                "gate_entries": gate_entries,
                "gate_active_frames": gate_active_frames,
                "gate_active_ratio": gate_active_frames / max(1, len(candidate_frames)),
                "intervention_events": len(events),
                "intervention_frames": sum(event["duration_frames"] for event in events),
            }
        )

    duration = np.asarray([event["duration_frames"] for event in all_events], dtype=float)
    terminal_damage = np.asarray(
        [event["outcomes"]["terminal"]["paired_delta"]["health_margin"] for event in all_events],
        dtype=float,
    )
    action_counts = Counter(event["action"] for event in all_events)
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in all_events:
        by_action[event["action"]].append(event)
    action_summary = {}
    for action, events in sorted(by_action.items()):
        deltas = np.asarray(
            [event["outcomes"]["terminal"]["paired_delta"]["health_margin"] for event in events],
            dtype=float,
        )
        action_summary[action] = {
            "events": len(events),
            "frames": sum(event["duration_frames"] for event in events),
            "terminal_damage_delta_mean": float(np.mean(deltas)),
            "terminal_damage_delta_median": float(np.median(deltas)),
            "terminal_positive_ratio": float(np.mean(deltas > 0.0)),
        }
    start_metric_summary = {}
    for metric in (
        "signed_aim_azimuth_deg",
        "signed_aim_elevation_deg",
        "los_azimuth_rate_deg_s",
        "los_elevation_rate_deg_s",
        "closing_rate_m_s",
        "bt_roll",
        "bt_pitch",
        "bt_yaw",
    ):
        values = np.asarray([event["start"][metric] for event in all_events], dtype=float)
        start_metric_summary[metric] = {
            "mean": float(np.mean(values)) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "min": float(np.min(values)) if values.size else None,
            "max": float(np.max(values)) if values.size else None,
            "positive_ratio": float(np.mean(values > 0.0)) if values.size else None,
        }
    horizon_summary = {}
    for horizon in (*[f"+{seconds:g}s" for seconds in HORIZONS_S], "terminal"):
        deltas = [event["outcomes"][horizon]["paired_delta"] for event in all_events]
        horizon_summary[horizon] = {}
        for metric in ("health_margin", "aim_error_deg", "los_rate_deg_s", "cone"):
            values = np.asarray([delta[metric] for delta in deltas], dtype=float)
            horizon_summary[horizon][metric] = {
                "mean": float(np.mean(values)) if values.size else None,
                "median": float(np.median(values)) if values.size else None,
                "positive_ratio": float(np.mean(values > 0.0)) if values.size else None,
            }
    return {
        "schema_version": "guidance_hybrid_v2.phase_a.v1",
        "source": str(input_root),
        "cases": len(case_records),
        "frames": sum(record["frames"] for record in case_records),
        "gate_entries": sum(record["gate_entries"] for record in case_records),
        "gate_active_frames": sum(record["gate_active_frames"] for record in case_records),
        "gate_active_ratio": (
            sum(record["gate_active_frames"] for record in case_records)
            / max(1, sum(record["frames"] for record in case_records))
        ),
        "intervention_events": len(all_events),
        "intervention_frames": int(np.sum(duration)) if duration.size else 0,
        "duration_frames_mean": float(np.mean(duration)) if duration.size else 0.0,
        "duration_frames_median": float(np.median(duration)) if duration.size else 0.0,
        "duration_frames_p95": float(np.percentile(duration, 95)) if duration.size else 0.0,
        "duration_frames_max": int(np.max(duration)) if duration.size else 0,
        "action_event_counts": dict(action_counts),
        "action_summary": action_summary,
        "start_metric_summary": start_metric_summary,
        "horizon_summary": horizon_summary,
        "event_terminal_damage_delta_mean": (
            float(np.mean(terminal_damage)) if terminal_damage.size else None
        ),
        "event_terminal_damage_delta_median": (
            float(np.median(terminal_damage)) if terminal_damage.size else None
        ),
        "case_records": case_records,
        "events": all_events,
        "interpretation_limits": [
            "3,148 intervention frames are temporally correlated and are not independent samples.",
            "Later events start after candidate and BT_DEFAULT trajectories have diverged; paired horizon deltas are diagnostic, not randomized causal estimates.",
            "The v1 controller divides angular offset by the configured angular magnitude, so 0.10/0.25/0.50 degree choices map to the same unit surface correction.",
        ],
    }


def render_report(analysis: dict[str, Any]) -> str:
    summary = analysis["action_summary"]
    rows = [
        f"| {action} | {value['events']} | {value['frames']} | "
        f"{value['terminal_damage_delta_mean']:+.7f} | "
        f"{value['terminal_damage_delta_median']:+.7f} | "
        f"{value['terminal_positive_ratio']:.1%} |"
        for action, value in summary.items()
    ]
    horizon_rows = [
        f"| {horizon} | {values['health_margin']['mean']:+.7f} | "
        f"{values['health_margin']['median']:+.7f} | "
        f"{values['aim_error_deg']['mean']:+.4f} | "
        f"{values['los_rate_deg_s']['mean']:+.4f} | "
        f"{values['cone']['mean']:+.4f} |"
        for horizon, values in analysis["horizon_summary"].items()
    ]
    return f"""# Guidance Hybrid v2 원인 분석

## 결론

Hybrid v1의 3,148개 개입 frame은 독립 표본이 아니라 **{analysis['intervention_events']}개 연속 intervention event**다. frame 수를 학습 표본 수나 causal evidence 수로 사용하면 안 된다. 전체 clean full-fight 결과도 mean/median Damage Δ가 각각 `-0.0030799766/-0.0033841133`, positive `0/4`였으므로 v1은 계속 `NOT_PROMOTED`다.

가장 직접적인 controller 원인은 v1이 angular offset을 같은 설정값으로 다시 나눠 action unit을 계산한다는 점이다. 따라서 `0.10°`, `0.25°`, `0.50°` 모두 같은 roll/pitch/yaw correction을 만든다. action magnitude 이름과 실제 controller effect가 분리돼 있었고, v2에서는 degree 비례 closed-loop 계약으로 바꿔야 한다.

## Event 재구성

- 분석 fight: {analysis['cases']}
- 전체 telemetry frame: {analysis['frames']}
- Gate entry: {analysis['gate_entries']}
- Gate active frame/ratio: {analysis['gate_active_frames']} / {analysis['gate_active_ratio']:.2%}
- intervention event/frame: {analysis['intervention_events']} / {analysis['intervention_frames']}
- event duration mean/median/P95/max: {analysis['duration_frames_mean']:.2f} / {analysis['duration_frames_median']:.2f} / {analysis['duration_frames_p95']:.2f} / {analysis['duration_frames_max']} frame

각 event에는 시작·종료 geometry와 `+0.5s/+1s/+2s/+4s/+8s/terminal` candidate 및 동일 시간대 BT_DEFAULT snapshot을 연결했다. 원본은 `automation/evidence/guidance_advantage_v2/phase_a_analysis.json`에 보존한다.

## Action별 event 진단

| action | event | frame | terminal ΔDamage mean | median | positive |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Event 이후 paired trajectory 진단

| horizon | ΔDamage mean | median | Δaim error mean | ΔLOS rate mean | Δcone mean |
|---|---:|---:|---:|---:|---:|
{chr(10).join(horizon_rows)}

event 시작 시 signed aim azimuth/elevation, LOS az/el rate, closing rate, BT roll/pitch/yaw의 mean/median/min/max 및 부호 분포를 evidence JSON에 기록했다. 각 frame의 controller payload에서 requested/applied correction과 positive/negative surface headroom도 event 시작·종료·각 horizon에 보존했다.

## 확인된 손실 원인

1. 고정 `VP_EL_POS_SMALL`만 실제 개입해 signed azimuth/elevation 방향을 사용하지 않았다.
2. 3,148 frame은 {analysis['intervention_events']} event 내부의 강한 시계열 중복이다.
3. angular magnitude 설정이 실제 surface correction 크기를 바꾸지 않는다.
4. v1은 45D observation에 ownship/target health를 포함해 서버 보장 feature 계약을 위반한다.
5. v1 primary library에는 Range/Target Speed 이름이 있으나 실제로는 pitch bias로 변환되어 의미가 일치하지 않는다.
6. Gate active ratio가 {analysis['gate_active_ratio']:.2%}로 넓고, rule은 gate 초기 36 frame마다 개입해 precision보다 개입량을 키웠다.
7. clean full-fight pair가 4개뿐이고 8개 pair는 target-crash contaminated였다.

## 해석 한계와 다음 검증

- 뒤쪽 event는 candidate와 BT_DEFAULT trajectory가 이미 갈라진 뒤 시작하므로 horizon delta는 진단값이지 randomized causal estimate가 아니다.
- 다음 단계는 primary action을 `BT_DEFAULT/AZ±/EL±`로 제한하고, degree 비례 controller에서 magnitude·duration을 동일 초기 state paired ablation해야 한다.
- clean Damage가 여러 geometry에서 반복 양수일 때만 dataset/Advantage model로 진행한다.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Guidance Hybrid v1 intervention events")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--evidence-json", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = analyze(args.input_root.resolve())
    args.evidence_json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_json.write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    args.report.write_text(render_report(analysis), encoding="utf-8")
    print(json.dumps({key: analysis[key] for key in (
        "cases", "gate_entries", "gate_active_ratio", "intervention_events",
        "intervention_frames", "duration_frames_median",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
