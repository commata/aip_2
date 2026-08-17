from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymap3d as pm

from dogfight.ai.action_provider import ActionProvider
from dogfight.ai.native_bt import AIPilot
from GeoMathUtil import GeometryInfo
from dogfight.ai.action_provider import ActionContext
from dogfight.ai.rl_action_provider import RLActionProvider
from dogfight.envs.observation import body_to_ned_rotation, build_observation
from dogfight.envs.observation import (
    TACTICAL16_HEALTH_SIMULATOR,
    official_phase_wez_config,
)
from dogfight.unreal.client import RemoteClientContext
from dogfight.unreal.protocol import CMD


@dataclass
class ConstantCommandPolicy:
    roll_cmd: float = 0.0
    pitch_cmd: float = 0.0
    yaw_cmd: float = 0.0
    throttle_cmd: float = 1.0

    def reset(self, context: RemoteClientContext) -> None:
        return None

    def compute_command(self, context: RemoteClientContext) -> CMD:
        return CMD(
            plane_id=context.plane_id,
            index=context.frame_index,
            roll_cmd=self.roll_cmd,
            pitch_cmd=self.pitch_cmd,
            yaw_cmd=self.yaw_cmd,
            throttle_cmd=self.throttle_cmd,
        )


class RLLightweightCommandPolicy:
    def __init__(
        self,
        action_provider: RLActionProvider,
        observation_mode: str = "relative14",
        observation_fn=None,
    ):
        self.action_provider = action_provider
        self.observation_mode = observation_mode
        self.observation_fn = observation_fn
        self.geometry = GeometryInfo()

    def reset(self, context: RemoteClientContext) -> None:
        self.action_provider.reset(None)

    def compute_command(self, context: RemoteClientContext) -> CMD:
        if context.own_plane.plane_info is None or context.enemy_plane.plane_info is None:
            return CMD(
                plane_id=context.plane_id,
                index=context.frame_index,
                roll_cmd=0.0,
                pitch_cmd=0.0,
                yaw_cmd=0.0,
                throttle_cmd=1.0,
            )

        ownship_state = plane_info_to_state(context.own_plane.plane_info)
        target_state = plane_info_to_state(context.enemy_plane.plane_info)
        observation = self._build_observation(ownship_state, target_state)

        action_result = self.action_provider.compute_action(
            ActionContext(
                sim=None,
                opponent_sim=None,
                ownship_state=ownship_state,
                target_state=target_state,
                observation=observation,
                info={"frame_index": context.frame_index},
            )
        )
        action = np.asarray(action_result.action, dtype=np.float32)

        return CMD(
            plane_id=context.plane_id,
            index=context.frame_index,
            roll_cmd=float(action[0]),
            pitch_cmd=float(action[1]),
            yaw_cmd=float(action[2]),
            throttle_cmd=float(action[3]),
        )

    def _build_observation(self, ownship_state, target_state) -> np.ndarray:
        if self.observation_fn is not None:
            return np.asarray(
                self.observation_fn(
                    ownship_state,
                    target_state,
                    self.geometry,
                    None,
                ),
                dtype=np.float32,
            )
        return build_observation(
            self.observation_mode,
            ownship_state,
            target_state,
            self.geometry,
        )


class ProviderCommandPolicy:
    def __init__(
        self,
        action_provider: ActionProvider,
        observation_mode: str = "relative14",
        observation_fn=None,
        ownship_force_side: int = 1,
        target_force_side: int = 2,
        action_repeat: int = 1,
        debug_action_repeat: bool = False,
        wez_config: dict | None = None,
        health_source: str = TACTICAL16_HEALTH_SIMULATOR,
        expected_sim_hz: int = 60,
        phase_config: list[dict] | None = None,
    ):
        self.action_provider = action_provider
        self.observation_mode = observation_mode
        self.observation_fn = observation_fn
        self.ownship_force_side = ownship_force_side
        self.target_force_side = target_force_side
        self.action_repeat = max(1, int(action_repeat))
        self.debug_action_repeat = debug_action_repeat
        self.wez_config = dict(wez_config) if wez_config is not None else None
        self.health_source = str(health_source)
        self.expected_sim_hz = int(expected_sim_hz)
        self.phase_config = list(phase_config or [])
        if self.expected_sim_hz <= 0:
            raise ValueError("expected_sim_hz must be positive")
        self.geometry = GeometryInfo()
        self._state_pair_count = 0
        self._cached_action: np.ndarray | None = None
        self._last_policy_count: int | None = None
        self._last_policy_frame_index: int | None = None
        self._match_start_frame_index: int | None = None

    def reset(self, context: RemoteClientContext) -> None:
        self.action_provider.reset(None)
        self._state_pair_count = 0
        self._cached_action = None
        self._last_policy_count = None
        self._last_policy_frame_index = None
        self._match_start_frame_index = None

    def compute_command(self, context: RemoteClientContext) -> CMD:
        if context.own_plane.plane_info is None or context.enemy_plane.plane_info is None:
            return CMD(
                plane_id=context.plane_id,
                index=context.frame_index,
                roll_cmd=0.0,
                pitch_cmd=0.0,
                yaw_cmd=0.0,
                throttle_cmd=1.0,
            )

        own_plane = context.own_plane.plane_info
        enemy_plane = context.enemy_plane.plane_info
        pair_count = self._state_pair_count
        self._state_pair_count += 1
        if self._match_start_frame_index is None:
            self._match_start_frame_index = int(context.frame_index)

        policy_updated = (
            self._cached_action is None
            or pair_count % self.action_repeat == 0
        )
        if policy_updated:
            action = self._compute_provider_action(context, own_plane, enemy_plane)
            self._cached_action = action
            self._last_policy_count = pair_count
            self._last_policy_frame_index = context.frame_index
        else:
            action = np.asarray(self._cached_action, dtype=np.float32)

        if self.debug_action_repeat:
            self._print_action_repeat_debug(
                context=context,
                pair_count=pair_count,
                policy_updated=policy_updated,
                action=action,
            )

        return CMD(
            plane_id=context.plane_id,
            index=context.frame_index,
            roll_cmd=float(action[0]),
            pitch_cmd=float(action[1]),
            yaw_cmd=float(action[2]),
            throttle_cmd=float(action[3]),
        )

    def _compute_provider_action(self, context, own_plane, enemy_plane) -> np.ndarray:
        ownship_state = plane_info_to_state(own_plane)
        target_state = plane_info_to_state(enemy_plane)
        sim_time_s = self._sim_time_s(context.frame_index)
        observation = self._build_observation(
            ownship_state,
            target_state,
            sim_time_s=sim_time_s,
        )

        own_speed = float(
            np.linalg.norm([own_plane.velocity.x, own_plane.velocity.y, own_plane.velocity.z])
        )
        target_speed = float(
            np.linalg.norm([enemy_plane.velocity.x, enemy_plane.velocity.y, enemy_plane.velocity.z])
        )
        # Base origin for the environment to match LLAtoCartesian in BT DLL
        OriLAT = 37.91455691666666
        OriLON = 128.18188127777776
        OriALT = 0.0

        my_lat, my_lon, my_alt = pm.ned2geodetic(
            own_plane.position.x,
            own_plane.position.y,
            -own_plane.position.z,
            OriLAT, OriLON, OriALT,
        )
        target_lat, target_lon, target_alt = pm.ned2geodetic(
            enemy_plane.position.x,
            enemy_plane.position.y,
            -enemy_plane.position.z,
            OriLAT, OriLON, OriALT,
        )

        my_plane_data = AIPilot.BuildPlaneData(
            [my_lat, my_lon, my_alt],
            [own_plane.rotation.roll, own_plane.rotation.pitch, own_plane.rotation.yaw],
            own_speed,
            self.ownship_force_side,
        )
        target_plane_data = AIPilot.BuildPlaneData(
            [target_lat, target_lon, target_alt],
            [enemy_plane.rotation.roll, enemy_plane.rotation.pitch, enemy_plane.rotation.yaw],
            target_speed,
            self.target_force_side,
        )

        action_result = self.action_provider.compute_action(
            ActionContext(
                sim=None,
                opponent_sim=None,
                ownship_state=ownship_state,
                target_state=target_state,
                observation=observation,
                info={
                    "frame_index": context.frame_index,
                    "sim_time_s": sim_time_s,
                    "my_plane_id": context.plane_id,
                    "target_plane_id": enemy_plane.plane_id,
                    "my_force_side": self.ownship_force_side,
                    "target_force_side": self.target_force_side,
                    "my_plane_data": my_plane_data,
                    "target_plane_data": target_plane_data,
                },
            )
        )
        return np.asarray(action_result.action, dtype=np.float32)

    def _sim_time_s(self, frame_index: int) -> float:
        if self._match_start_frame_index is None:
            return 0.0
        elapsed_frames = max(0, int(frame_index) - self._match_start_frame_index)
        return elapsed_frames / float(self.expected_sim_hz)

    def _print_action_repeat_debug(
        self,
        context: RemoteClientContext,
        pair_count: int,
        policy_updated: bool,
        action: np.ndarray,
    ) -> None:
        repeat_offset = pair_count % self.action_repeat
        print(
            "[DogFightEnv][Unreal][ACTION_REPEAT] "
            f"pair_count={pair_count} repeat={self.action_repeat} "
            f"repeat_offset={repeat_offset} policy_updated={policy_updated} "
            f"cmd_frame={context.frame_index} "
            f"own_frame={context.own_plane.frame_index} "
            f"enemy_frame={context.enemy_plane.frame_index} "
            f"policy_frame={self._last_policy_frame_index} "
            f"policy_count={self._last_policy_count} "
            f"action={np.asarray(action, dtype=np.float32).tolist()}"
        )

    def _build_observation(
        self,
        ownship_state,
        target_state,
        *,
        sim_time_s: float = 0.0,
    ) -> np.ndarray:
        if self.observation_fn is not None:
            return np.asarray(
                self.observation_fn(
                    ownship_state,
                    target_state,
                    self.geometry,
                    None,
                ),
                dtype=np.float32,
            )
        wez_config = self.wez_config
        if self.observation_mode == "tactical16" and wez_config is not None:
            wez_config = official_phase_wez_config(
                sim_time_s,
                min_range_m=float(wez_config["min_range_m"]),
            )
        return build_observation(
            self.observation_mode,
            ownship_state,
            target_state,
            self.geometry,
            wez_config,
            health_source=self.health_source,
        )


def plane_info_to_state(plane_info) -> np.ndarray:
    state = np.zeros(51, dtype=np.float32)
    state[0] = plane_info.position.x
    state[1] = plane_info.position.y
    state[2] = -plane_info.position.z  # Z is UP in Unreal, D is DOWN in NED
    state[3] = plane_info.rotation.roll
    state[4] = plane_info.rotation.pitch
    state[5] = plane_info.rotation.yaw
    velocity_ned = np.array(
        [plane_info.velocity.x, plane_info.velocity.y, -plane_info.velocity.z],
        dtype=np.float64,
    )
    rotation = body_to_ned_rotation(state[3:6])
    state[6:9] = rotation.T @ velocity_ned
    
    # [HOTFIX] RL State 매핑 누락 복구
    # 인덱스 12 (KCAS, 속도): 언리얼의 속도 벡터 크기를 계산
    state[12] = float(np.linalg.norm([plane_info.velocity.x, plane_info.velocity.y, plane_info.velocity.z]))
    # 인덱스 44 (Altitude, 고도): 언리얼의 Z축을 고도로 사용
    state[44] = plane_info.position.z
    # 인덱스 45 (Health, 체력): 기본 체력을 1.0으로 고정하여 사망 상태 오인 방지
    state[45] = 1.0
    
    return state
