from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from dogfight.ai.hybrid_action_provider import target_ata_deg
from dogfight.envs.observation import aim_residual_geometry


TARGET_ATA_BINS = (0.0, 60.0, 90.0, 115.0, 120.0, 135.0, 150.0, 165.0, 180.0001)
OWNSHIP_ATA_BINS = (0.0, 5.0, 15.0, 30.0, 45.0, 90.0, 180.0001)
RANGE_BINS_M = (0.0, 152.4, 500.0, 914.4, 1500.0, 2500.0, float("inf"))
CLOSING_RATE_BINS_M_S = (float("-inf"), -100.0, 0.0, 100.0, 250.0, float("inf"))


@dataclass(frozen=True)
class Rear120BatchContractConfig:
    mode: str = "disabled"
    minimum_target_ata_deg: float = 120.0
    truncate_on_exit: bool = True
    mask_exit_reward: bool = True

    def validate(self) -> None:
        if self.mode not in {"disabled", "rear120_segment"}:
            raise ValueError(f"unsupported residual batch contract mode: {self.mode!r}")
        if not 0.0 <= self.minimum_target_ata_deg <= 180.0:
            raise ValueError("minimum_target_ata_deg must be within [0, 180]")


def _histogram_bin(value: float, edges: tuple[float, ...]) -> str:
    for lower, upper in zip(edges, edges[1:]):
        if lower <= value < upper or (math.isinf(upper) and value >= lower):
            lower_text = "neg_inf" if math.isinf(lower) and lower < 0 else f"{lower:g}"
            upper_text = "inf" if math.isinf(upper) else f"{upper:g}"
            return f"{lower_text}_to_{upper_text}"
    return "out_of_range"


class Rear120TrainingBatchTracker:
    """Measure and enforce that residual actions originate inside rear120.

    Eligibility is measured on the state that produced the policy action
    (obs_t), not on obs_t+1.  If the next state leaves the hard envelope, the
    episode is truncated immediately so no next policy action is sampled from
    an ineligible state.  The exit transition is retained as an explicit
    boundary transition and its reward can be masked.
    """

    def __init__(self, config: Rear120BatchContractConfig | dict | None = None):
        if config is None:
            config = Rear120BatchContractConfig()
        elif isinstance(config, dict):
            config = Rear120BatchContractConfig(**config)
        config.validate()
        self.config = config
        self.reset()

    @property
    def enabled(self) -> bool:
        return self.config.mode == "rear120_segment"

    def reset(self) -> None:
        self.samples = 0
        self.rear120_samples = 0
        self.offensive_samples = 0
        self.pre_aim_samples = 0
        self.ineligible_samples = 0
        self.boundary_exit_transitions = 0
        self.histograms = {
            "target_ata_deg": {},
            "ownship_ata_deg": {},
            "range_m": {},
            "closing_rate_m_s": {},
        }

    def eligible(self, ownship_state, target_state) -> bool:
        return target_ata_deg(ownship_state, target_state) >= self.config.minimum_target_ata_deg

    def validate_initial_state(self, ownship_state, target_state) -> None:
        if self.enabled and not self.eligible(ownship_state, target_state):
            measured = target_ata_deg(ownship_state, target_state)
            raise ValueError(
                "rear120_segment initial state is outside the hard envelope: "
                f"target_ata_deg={measured:.6f} < {self.config.minimum_target_ata_deg:.6f}"
            )

    def record_action_state(self, ownship_state, target_state, action_info: dict | None) -> dict:
        geometry = aim_residual_geometry(ownship_state, target_state)
        measured_target_ata = target_ata_deg(ownship_state, target_state)
        rear120 = measured_target_ata >= self.config.minimum_target_ata_deg
        gate = dict((action_info or {}).get("gate", {}) or {})

        self.samples += 1
        self.rear120_samples += int(rear120)
        self.ineligible_samples += int(not rear120)
        self.offensive_samples += int(bool(gate.get("offensive_eligible", False)))
        self.pre_aim_samples += int(bool(gate.get("pre_aim_eligible", False)))
        values = {
            "target_ata_deg": measured_target_ata,
            "ownship_ata_deg": float(geometry["ata_deg"]),
            "range_m": float(geometry["distance_m"]),
            "closing_rate_m_s": float(geometry["closing_rate_m_s"]),
        }
        for name, edges in (
            ("target_ata_deg", TARGET_ATA_BINS),
            ("ownship_ata_deg", OWNSHIP_ATA_BINS),
            ("range_m", RANGE_BINS_M),
            ("closing_rate_m_s", CLOSING_RATE_BINS_M_S),
        ):
            key = _histogram_bin(values[name], edges)
            histogram = self.histograms[name]
            histogram[key] = int(histogram.get(key, 0)) + 1
        return {**values, "rear120_eligible": rear120}

    def should_truncate_after_step(self, ownship_state, target_state) -> bool:
        exit_envelope = self.enabled and not self.eligible(ownship_state, target_state)
        if exit_envelope:
            self.boundary_exit_transitions += 1
        return bool(exit_envelope and self.config.truncate_on_exit)

    def summary(self) -> dict:
        denominator = max(1, self.samples)
        return {
            "mode": self.config.mode,
            "minimum_target_ata_deg": self.config.minimum_target_ata_deg,
            "eligible_sample_fraction": self.rear120_samples / denominator,
            "rear120_sample_fraction": self.rear120_samples / denominator,
            "offensive_sample_fraction": self.offensive_samples / denominator,
            "pre_aim_sample_fraction": self.pre_aim_samples / denominator,
            "ineligible_sample_count": self.ineligible_samples,
            "sample_count": self.samples,
            "boundary_exit_transition_count": self.boundary_exit_transitions,
            "histograms": {
                name: dict(counts) for name, counts in self.histograms.items()
            },
        }
