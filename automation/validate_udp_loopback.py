from __future__ import annotations

from dataclasses import asdict, dataclass
import socket
import threading
import time
from typing import Iterable

import numpy as np

from automation.replay_submission_packets import latency_summary
from dogfight.unreal.client import UnrealAIPilotUDPClient
from dogfight.unreal.protocol import (
    CMD,
    MessageType,
    PlaneInfo,
    SetPlaneID,
    pack_plane_info,
    pack_set_plane_id,
    unpack_cmd,
    unpack_message_type,
)


@dataclass(frozen=True)
class LoopbackResult:
    requested_frames: int
    command_count: int
    missing_frames: tuple[int, ...]
    duplicate_frames: tuple[int, ...]
    wrong_index_count: int
    latency: dict[str, float | int]
    commands: tuple[dict, ...]


def run_udp_loopback(
    command_policy,
    packet_pairs: Iterable[tuple[PlaneInfo, PlaneInfo]],
    *,
    expected_hz: float = 60.0,
    real_time: bool = True,
    response_timeout_s: float = 0.5,
) -> LoopbackResult:
    """Exercise the actual UDP client receive/send path on localhost.

    Each frame sends the target and ownship PlaneInfo datagrams, then waits for
    exactly one CMD bearing that frame index.  Latency is measured from the
    second PlaneInfo send to CMD receipt, which includes deserialization,
    observation/Gate/provider work, serialization, and both local UDP hops.
    """

    pairs = list(packet_pairs)
    if expected_hz <= 0:
        raise ValueError("expected_hz must be positive")

    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    server.settimeout(response_timeout_s)
    host, port = server.getsockname()

    client = UnrealAIPilotUDPClient(
        command_policy=command_policy,
        server_ip=str(host),
        server_port=int(port),
        team_name="UDP_LOOPBACK",
        heartbeat_interval_sec=0.05,
        command_delay_sec=0.0,
        recv_timeout_sec=0.02,
    )
    client_thread = threading.Thread(target=client.run, daemon=True)
    client_thread.start()

    client_addr: tuple[str, int] | None = None
    deadline = time.perf_counter() + response_timeout_s
    while time.perf_counter() < deadline and client_addr is None:
        try:
            packet, address = server.recvfrom(1024)
        except socket.timeout:
            break
        message_type = unpack_message_type(packet)
        if message_type in {MessageType.MT_ClientInfo, MessageType.MT_SimState}:
            client_addr = address

    if client_addr is None:
        client.stop()
        server.close()
        raise TimeoutError("UDP loopback client did not send join/heartbeat")

    server.sendto(pack_set_plane_id(SetPlaneID(plane_id=1)), client_addr)
    frame_period_s = 1.0 / expected_hz
    commands: list[CMD] = []
    response_records: list[dict[str, float]] = []

    try:
        for expected_frame, (own, target) in enumerate(pairs):
            frame_started = time.perf_counter()
            server.sendto(pack_plane_info(target), client_addr)
            second_sent = time.perf_counter()
            server.sendto(pack_plane_info(own), client_addr)

            command: CMD | None = None
            command_deadline = time.perf_counter() + response_timeout_s
            while time.perf_counter() < command_deadline:
                packet, _ = server.recvfrom(1024)
                if unpack_message_type(packet) == MessageType.MT_CMD:
                    command = unpack_cmd(packet)
                    break
            if command is None:
                continue
            commands.append(command)
            response_records.append(
                {"latency_ms": (time.perf_counter() - second_sent) * 1000.0}
            )

            if real_time:
                remaining = frame_period_s - (time.perf_counter() - frame_started)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        client.stop()
        client_thread.join(timeout=1.0)
        server.close()

    received = [int(command.index) for command in commands]
    requested = [int(pair[0].index) for pair in pairs]
    counts = {frame: received.count(frame) for frame in set(received)}
    missing = tuple(frame for frame in requested if frame not in counts)
    duplicates = tuple(sorted(frame for frame, count in counts.items() if count > 1))
    wrong_index_count = sum(
        int(command.index != pairs[index][0].index)
        for index, command in enumerate(commands[: len(pairs)])
    )
    return LoopbackResult(
        requested_frames=len(pairs),
        command_count=len(commands),
        missing_frames=missing,
        duplicate_frames=duplicates,
        wrong_index_count=wrong_index_count,
        latency=latency_summary(response_records),
        commands=tuple(asdict(command) for command in commands),
    )


def assert_loopback_contract(result: LoopbackResult) -> None:
    if result.command_count != result.requested_frames:
        raise AssertionError(
            f"CMD count mismatch: {result.command_count}/{result.requested_frames}"
        )
    if result.missing_frames:
        raise AssertionError(f"missing CMD frames: {result.missing_frames}")
    if result.duplicate_frames:
        raise AssertionError(f"duplicate CMD frames: {result.duplicate_frames}")
    if result.wrong_index_count:
        raise AssertionError(f"wrong CMD index count: {result.wrong_index_count}")
    if not np.isfinite(float(result.latency["max_ms"])):
        raise AssertionError("non-finite UDP loopback latency")
