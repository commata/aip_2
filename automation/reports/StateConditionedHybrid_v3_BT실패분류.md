# State-Conditioned Hybrid v3 Pure BT 실패 분류

이 분류는 server-safe geometry와 same-frame BT context만 사용한 진단 규칙이다. 학습 label과 Promotion primary는 실제 paired Damage이며, 분류명 자체를 성능 근거로 사용하지 않는다.

| failure type | states | oracle mean | oracle positive | dominant oracle action |
|---|---:|---:|---:|---|
| A_AZIMUTH_OVERSHOOT | 23 | +0.007041873 | 100.00% | VP_EL_POS_SMALL__m0.25__d36 |
| B_ELEVATION_OVERSHOOT | 46 | +0.043907583 | 100.00% | VP_EL_POS_SMALL__m0.25__d36 |
| C_LOS_ANGULAR_RATE_HIGH | 0 | +0.000000000 | 0.00% | N/A |
| D_RANGE_TOO_CLOSE | 0 | +0.000000000 | 0.00% | N/A |
| E_RANGE_TOO_FAR | 0 | +0.000000000 | 0.00% | N/A |
| F_CLOSING_TOO_HIGH | 0 | +0.000000000 | 0.00% | N/A |
| G_SURFACE_AUTHORITY_LIMIT | 219 | +0.009781913 | 100.00% | VP_EL_POS_SMALL__m0.25__d36 |
| H_CROSSING_LEAD_SHORTFALL | 1 | +0.010017725 | 100.00% | VP_EL_POS_SMALL__m0.25__d36 |
| I_PURE_BT_ALREADY_OPTIMAL | 221 | +0.000000010 | 0.00% | BT_DEFAULT |
| J_ENERGY_ALTITUDE_SAFETY | 0 | +0.000000000 | 0.00% | N/A |
