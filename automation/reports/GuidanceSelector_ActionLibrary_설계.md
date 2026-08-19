# Guidance Selector action library 설계

RL 출력은 Roll/Pitch/Yaw가 아니라 9-class categorical Guidance ID다. `BT_DEFAULT`는 bitwise-equivalent Pure BT action을 반환한다. 나머지 action은 Pure BT가 계산한 VP local azimuth/elevation/distance 또는 target-speed setpoint를 한 축씩만 보정한다.

| ID | action | frozen delta |
|---:|---|---:|
| 0 | `BT_DEFAULT` | 없음 |
| 1 | `VP_AZ_POS_SMALL` | +0.5° |
| 2 | `VP_AZ_NEG_SMALL` | -0.5° |
| 3 | `VP_EL_POS_SMALL` | +0.5° |
| 4 | `VP_EL_NEG_SMALL` | -0.5° |
| 5 | `VP_RANGE_FORWARD_SMALL` | +50m |
| 6 | `VP_RANGE_BACKWARD_SMALL` | -50m |
| 7 | `TARGET_SPEED_UP_SMALL` | +10m/s |
| 8 | `TARGET_SPEED_DOWN_SMALL` | -10m/s |

Pure BT DLL은 VP 조회는 제공하지만 외부 setpoint 주입 API를 제공하지 않는다. 따라서 Python Guidance composer가 BT VP를 body-local spherical setpoint로 변환하고 delta를 적용한 뒤, 고정 gain bounded controller가 surface command를 산출한다. selector는 surface 값을 직접 만들거나 보지 않는다. controller 최대 correction은 축별 0.08이며 BT의 방향별 남은 authority로 다시 제한한다. target-speed action도 throttle을 변경하지 않고 작은 pitch guidance bias만 만든다.

Throttle은 모든 action에서 exact Pure BT다. Gate OFF, 낮은 confidence, invalid shape/action/probability, nonfinite, exception, timeout, controller 오류에서는 해당 frame을 exact Pure BT로 fallback한다.

Observation `guidance_selector_v1`은 Tactical16 전체, signed aim/LOS-rate/range/closing/target ATA/phase, BT action/VP/target-speed, controller headroom/saturation/authority, previous action/hold/Gate/safety context를 포함하는 45D float32다.

