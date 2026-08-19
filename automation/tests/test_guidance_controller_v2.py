from __future__ import annotations

import numpy as np

from dogfight.ai.guidance_selector import (
    GuidanceActionConfig,
    GuidanceControllerConfig,
    GuidanceSetpoint,
    compose_guidance_setpoint,
    guidance_to_surface_action,
)
from dogfight.sim.state_schema import StateIndex


def state(*, n=0.0, e=0.0, altitude=5000.0, yaw=0.0, speed=230.0):
    value = np.zeros(46, dtype=np.float64)
    value[StateIndex.N] = n
    value[StateIndex.E] = e
    value[StateIndex.D] = -altitude
    value[StateIndex.YAW] = yaw
    value[StateIndex.KCAS] = speed
    value[StateIndex.ALT] = altitude
    value[StateIndex.HEALTH] = 1.0
    value[6] = speed
    return value


def test_v2_controller_magnitude_and_sign_are_physical():
    own = state()
    target = state(n=1000.0, e=100.0, yaw=180.0)
    bt = np.asarray([0.0, 0.0, 0.0, 0.63], dtype=np.float32)
    base = GuidanceSetpoint(0.0, 0.0, 1000.0, 230.0)
    controller = GuidanceControllerConfig(
        kind="vp_error_pd_v2", los_rate_damping_per_deg_s=0.0
    )

    def requested(action, magnitude):
        action_config = GuidanceActionConfig(angular_offset_deg=magnitude)
        corrected = compose_guidance_setpoint(base, action, action_config)
        final, diagnostics = guidance_to_surface_action(
            bt,
            base,
            corrected,
            action_config,
            controller,
            ownship_state=own,
            target_state=target,
        )
        assert final[3] == bt[3]
        return np.asarray(diagnostics["requested_surface_correction"])

    small = requested("VP_AZ_POS_SMALL", 0.1)
    large = requested("VP_AZ_POS_SMALL", 0.5)
    negative = requested("VP_AZ_NEG_SMALL", 0.5)
    assert np.allclose(large[[0, 2]], 5.0 * small[[0, 2]], atol=1e-7)
    assert np.all(large[[0, 2]] > 0.0)
    assert np.all(negative[[0, 2]] < 0.0)


def test_v2_controller_requires_live_geometry_and_preserves_throttle():
    bt = np.asarray([0.99, -0.99, 0.99, 0.61], dtype=np.float32)
    base = GuidanceSetpoint(0.0, 0.0, 1000.0, 230.0)
    config = GuidanceActionConfig(angular_offset_deg=0.25)
    final, diagnostics = guidance_to_surface_action(
        bt,
        base,
        compose_guidance_setpoint(base, "VP_EL_POS_SMALL", config),
        config,
        GuidanceControllerConfig(kind="vp_error_pd_v2"),
        ownship_state=state(),
        target_state=state(n=1000.0, altitude=5050.0, yaw=180.0),
    )
    assert final[3] == bt[3]
    assert diagnostics["kind"] == "vp_error_pd_v2"
    assert np.all(final[:3] <= 1.0)
    assert np.all(final[:3] >= -1.0)
