"""Gate 진입 시점을 기준으로 paired Pure BT/Hybrid 기동 window를 비교한다."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


DEFAULT_OFFSETS_S = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0)


def load_frames(path: str | Path) -> list[dict[str, Any]]:
    frames = []
    target_damage_cumulative = 0.0
    ownship_damage_cumulative = 0.0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("record_type") != "frame":
                continue
            target_damage_cumulative += float(row.get("target_damage", 0.0))
            ownship_damage_cumulative += float(row.get("ownship_damage", 0.0))
            row["target_damage_cumulative"] = target_damage_cumulative
            row["ownship_damage_cumulative"] = ownship_damage_cumulative
            frames.append(row)
    if not frames:
        raise ValueError(f"frame telemetry가 없음: {path}")
    return frames


def gate_info(frame: dict[str, Any]) -> dict[str, Any]:
    hybrid = frame.get("hybrid", {}) or {}
    for key in ("gate", "aim_gate", "offensive_gate"):
        value = hybrid.get(key)
        if isinstance(value, dict):
            return value
    return {}


def frame_metrics(frame: dict[str, Any]) -> dict[str, float]:
    los_rate = math.hypot(
        float(frame.get("los_azimuth_rate_deg_s", 0.0)),
        float(frame.get("los_elevation_rate_deg_s", 0.0)),
    )
    correction = (
        frame.get("hybrid", {}).get("applied_rl_correction") or [0.0] * 4
    )
    ownship = frame["ownship"]
    return {
        "los_deg": float(frame["ata_deg"]),
        "los_rate_deg_s": los_rate,
        "target_ata_deg": float(frame["target_ata_deg"]),
        "distance_m": float(frame["distance_m"]),
        "speed_m_s": float(ownship["speed_kcas"]),
        "altitude_m": float(ownship["altitude_m"]),
        "damage_dealt_cumulative": float(frame["target_damage_cumulative"]),
        "damage_received_cumulative": float(frame["ownship_damage_cumulative"]),
        "roll_cmd": float(frame["ownship_action"][0]),
        "pitch_cmd": float(frame["ownship_action"][1]),
        "yaw_cmd": float(frame["ownship_action"][2]),
        "throttle_cmd": float(frame["ownship_action"][3]),
        "roll_correction": float(correction[0]),
        "pitch_correction": float(correction[1]),
        "yaw_correction": float(correction[2]),
    }


def nearest_frame(frames: list[dict[str, Any]], time_s: float) -> dict[str, Any] | None:
    if time_s < float(frames[0]["sim_time_s"]) or time_s > float(frames[-1]["sim_time_s"]):
        return None
    return min(frames, key=lambda row: abs(float(row["sim_time_s"]) - time_s))


def subtract(hybrid: dict[str, float], pure: dict[str, float]) -> dict[str, float]:
    return {key: hybrid[key] - pure[key] for key in hybrid}


def summarize_matched_active(
    pure_frames: list[dict[str, Any]],
    hybrid_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    pairs = []
    for hybrid_frame in hybrid_frames:
        if not gate_info(hybrid_frame).get("active", False):
            continue
        pure_frame = nearest_frame(pure_frames, float(hybrid_frame["sim_time_s"]))
        if pure_frame is not None:
            pairs.append((frame_metrics(pure_frame), frame_metrics(hybrid_frame)))
    keys = tuple(frame_metrics(hybrid_frames[0]))
    return {
        "matched_frames": len(pairs),
        "mean_pure": {
            key: fmean(pure[key] for pure, _ in pairs) if pairs else None
            for key in keys
        },
        "mean_hybrid": {
            key: fmean(hybrid[key] for _, hybrid in pairs) if pairs else None
            for key in keys
        },
        "mean_delta_hybrid_minus_pure": {
            key: fmean(hybrid[key] - pure[key] for pure, hybrid in pairs)
            if pairs
            else None
            for key in keys
        },
    }


def summarize_surface_authority(hybrid_frames: list[dict[str, Any]]) -> dict[str, Any]:
    axes = ("roll", "pitch", "yaw")
    active = [frame for frame in hybrid_frames if gate_info(frame).get("active", False)]
    result: dict[str, Any] = {"active_frames": len(active), "axes": {}}
    for index, axis in enumerate(axes):
        rows = []
        for frame in active:
            hybrid = frame.get("hybrid", {}) or {}
            authority = hybrid.get("surface_authority", {}) or {}
            requested = authority.get("requested_surface_correction", [0.0] * 3)
            applied = authority.get("applied_surface_correction", [0.0] * 3)
            directional = authority.get("directional_headroom", [0.0] * 3)
            positive = authority.get("positive_headroom", [0.0] * 3)
            negative = authority.get("negative_headroom", [0.0] * 3)
            request_nonzero = authority.get("request_nonzero", [False] * 3)
            bt_saturated = authority.get("bt_surface_saturated", [False] * 3)
            final_saturated = authority.get("final_surface_saturated", [False] * 3)
            req = float(requested[index])
            app = float(applied[index])
            nonzero = bool(request_nonzero[index]) and abs(req) > 1e-12
            rows.append(
                {
                    "requested_abs": abs(req),
                    "applied_abs": abs(app),
                    "directional_headroom": float(directional[index]),
                    "positive_headroom": float(positive[index]),
                    "negative_headroom": float(negative[index]),
                    "request_nonzero": nonzero,
                    "applied_ratio": abs(app / req) if nonzero else None,
                    "authority_blocked": nonzero and abs(app) <= abs(req) * 0.01,
                    "bt_saturated": bool(bt_saturated[index]),
                    "final_saturated": bool(final_saturated[index]),
                }
            )
        requested_rows = [row for row in rows if row["request_nonzero"]]
        result["axes"][axis] = {
            "requested_abs_mean": fmean(row["requested_abs"] for row in rows) if rows else None,
            "applied_abs_mean": fmean(row["applied_abs"] for row in rows) if rows else None,
            "applied_to_requested_mean": (
                fmean(row["applied_ratio"] for row in requested_rows)
                if requested_rows
                else None
            ),
            "positive_headroom_mean": fmean(row["positive_headroom"] for row in rows) if rows else None,
            "negative_headroom_mean": fmean(row["negative_headroom"] for row in rows) if rows else None,
            "directional_headroom_mean": fmean(row["directional_headroom"] for row in rows) if rows else None,
            "request_nonzero_frames": len(requested_rows),
            "authority_blocked_ratio": (
                sum(row["authority_blocked"] for row in requested_rows) / len(requested_rows)
                if requested_rows
                else None
            ),
            "bt_saturation_ratio": sum(row["bt_saturated"] for row in rows) / len(rows) if rows else None,
            "final_saturation_ratio": sum(row["final_saturated"] for row in rows) / len(rows) if rows else None,
        }
    return result


def analyze(
    pure_frames: list[dict[str, Any]],
    hybrid_frames: list[dict[str, Any]],
    offsets_s: tuple[float, ...] = DEFAULT_OFFSETS_S,
) -> dict[str, Any]:
    entries = [
        frame for frame in hybrid_frames if gate_info(frame).get("entry", False)
    ]
    windows = []
    for entry_index, entry in enumerate(entries, start=1):
        entry_time = float(entry["sim_time_s"])
        points = []
        for offset in offsets_s:
            sample_time = entry_time + offset
            pure = nearest_frame(pure_frames, sample_time)
            hybrid = nearest_frame(hybrid_frames, sample_time)
            if pure is None or hybrid is None:
                points.append({"offset_s": offset, "available": False})
                continue
            pure_metrics = frame_metrics(pure)
            hybrid_metrics = frame_metrics(hybrid)
            points.append(
                {
                    "offset_s": offset,
                    "sample_time_s": sample_time,
                    "available": True,
                    "gate_active": bool(gate_info(hybrid).get("active", False)),
                    "pure": pure_metrics,
                    "hybrid": hybrid_metrics,
                    "delta_hybrid_minus_pure": subtract(hybrid_metrics, pure_metrics),
                }
            )
        windows.append(
            {
                "entry_index": entry_index,
                "entry_time_s": entry_time,
                "entry_geometry": gate_info(entry),
                "points": points,
            }
        )
    return {
        "pure_frames": len(pure_frames),
        "hybrid_frames": len(hybrid_frames),
        "gate_entries": len(entries),
        "gate_active_ratio": sum(
            bool(gate_info(frame).get("active", False)) for frame in hybrid_frames
        ) / len(hybrid_frames),
        "matched_gate_active_summary": summarize_matched_active(
            pure_frames, hybrid_frames
        ),
        "surface_authority_summary": summarize_surface_authority(hybrid_frames),
        "entry_windows": windows,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["matched_gate_active_summary"]
    delta = summary["mean_delta_hybrid_minus_pure"]
    lines = [
        "# Gate 진입 전후 Paired 기동 분석",
        "",
        f"- Gate 진입 횟수: `{payload['gate_entries']}`",
        f"- Gate 활성 비율: `{payload['gate_active_ratio']:.3f}`",
        f"- 시간 정렬 활성 frame 수: `{summary['matched_frames']}`",
        f"- 활성 구간 평균 LOS 차이: `{_fmt(delta['los_deg'])}°`",
        f"- 활성 구간 평균 LOS rate 차이: `{_fmt(delta['los_rate_deg_s'])}°/s`",
        f"- 활성 구간 평균 속도 차이: `{_fmt(delta['speed_m_s'])}m/s`",
        f"- 활성 구간 평균 고도 차이: `{_fmt(delta['altitude_m'])}m`",
        f"- 활성 구간 누적 Damage 차이: `{_fmt(delta['damage_dealt_cumulative'])}`",
        "",
        "## 활성 구간 조종 권한",
        "",
        "| 축 | 요청 평균 | 적용 평균 | 적용/요청 | 방향 headroom | BT 포화율 | 최종 포화율 | authority 차단율 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for axis, values in payload["surface_authority_summary"]["axes"].items():
        lines.append(
            f"| {axis} | {_fmt(values['requested_abs_mean'])} | "
            f"{_fmt(values['applied_abs_mean'])} | "
            f"{_fmt(values['applied_to_requested_mean'])} | "
            f"{_fmt(values['directional_headroom_mean'])} | "
            f"{_fmt(values['bt_saturation_ratio'])} | "
            f"{_fmt(values['final_saturation_ratio'])} | "
            f"{_fmt(values['authority_blocked_ratio'])} |"
        )
    lines += [
        "",
        "## 진입 window",
        "",
    ]
    for window in payload["entry_windows"]:
        lines += [
            f"### 진입 {window['entry_index']} — {window['entry_time_s']:.3f}초",
            "",
            "| Offset | Gate | LOS Δ(°) | LOS rate Δ(°/s) | 거리 Δ(m) | 속도 Δ(m/s) | 고도 Δ(m) | Damage Δ |",
            "|---:|:---:|---:|---:|---:|---:|---:|---:|",
        ]
        for point in window["points"]:
            if not point["available"]:
                lines.append(f"| {point['offset_s']:+.0f}s | - | 자료 없음 | 자료 없음 | 자료 없음 | 자료 없음 | 자료 없음 | 자료 없음 |")
                continue
            d = point["delta_hybrid_minus_pure"]
            lines.append(
                f"| {point['offset_s']:+.0f}s | {'ON' if point['gate_active'] else 'OFF'} | "
                f"{_fmt(d['los_deg'])} | {_fmt(d['los_rate_deg_s'])} | "
                f"{_fmt(d['distance_m'])} | {_fmt(d['speed_m_s'])} | "
                f"{_fmt(d['altitude_m'])} | {_fmt(d['damage_dealt_cumulative'])} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: float | None) -> str:
    return "자료 없음" if value is None else f"{value:+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure", required=True)
    parser.add_argument("--hybrid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown")
    args = parser.parse_args()
    payload = analyze(load_frames(args.pure), load_frames(args.hybrid))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    markdown = Path(args.markdown) if args.markdown else output.with_suffix(".md")
    write_markdown(markdown, payload)
    print(json.dumps(payload["matched_gate_active_summary"], indent=2))


if __name__ == "__main__":
    main()
