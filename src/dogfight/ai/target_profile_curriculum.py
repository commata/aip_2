from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from dogfight.ai.bt_action_provider import BTActionProvider
from dogfight.ai.native_bt import AIPilot


@dataclass(frozen=True)
class TargetProfileSelection:
    profile_id: str
    backend_type: str
    behavior_cluster: str
    index: int


class TargetProfileCurriculum:
    """Select one target backend per episode without exposing identity to policy."""

    def __init__(
        self,
        profiles: list[dict[str, Any]],
        *,
        provider_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        if not profiles:
            raise ValueError("target profile curriculum requires at least one profile")
        self._profiles = [dict(profile) for profile in profiles]
        self._weights = np.asarray(
            [float(profile.get("weight", 1.0)) for profile in self._profiles],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(self._weights)) or np.any(self._weights <= 0.0):
            raise ValueError("target profile weights must be finite and positive")
        self._weights /= float(np.sum(self._weights))
        self._provider_factory = provider_factory or self._build_bt_provider
        self._providers: dict[str, Any] = {}
        self._active_provider = None
        self._selection: TargetProfileSelection | None = None
        self._counts = {str(item["profile_id"]): 0 for item in self._profiles}

    @staticmethod
    def _build_bt_provider(profile: dict[str, Any]) -> BTActionProvider:
        pilot = AIPilot(str(profile["dll_path"]))
        return BTActionProvider(
            ai_pilot=pilot,
            enable_turn_throttle_optimization=False,
        )

    @property
    def active_provider(self):
        return self._active_provider

    @property
    def selection(self) -> TargetProfileSelection | None:
        return self._selection

    @property
    def profile_ids(self) -> list[str]:
        return [str(item["profile_id"]) for item in self._profiles]

    def select_episode(self, rng) -> TargetProfileSelection:
        if self._active_provider is not None:
            self._active_provider.close()
            self._active_provider = None
        index = int(rng.choice(len(self._profiles), p=self._weights))
        profile = self._profiles[index]
        backend = str(profile["backend_type"])
        if backend not in ("autopilot", "behavior_tree"):
            raise ValueError(f"unsupported target profile backend: {backend}")
        profile_id = str(profile["profile_id"])
        if backend == "behavior_tree":
            provider = self._providers.get(profile_id)
            if provider is None:
                provider = self._provider_factory(profile)
                self._providers[profile_id] = provider
            self._active_provider = provider
        self._selection = TargetProfileSelection(
            profile_id=profile_id,
            backend_type=backend,
            behavior_cluster=str(profile["behavior_cluster"]),
            index=index,
        )
        self._counts[profile_id] += 1
        return self._selection

    def telemetry(self) -> dict[str, Any]:
        total = sum(self._counts.values())
        return {
            "target_profile_episode_counts": dict(self._counts),
            "target_profile_episode_total": total,
            "target_profile_id": (
                self._selection.profile_id if self._selection is not None else None
            ),
        }

    def close(self) -> None:
        for provider in self._providers.values():
            try:
                provider.close()
            except Exception:
                pass
        self._active_provider = None

