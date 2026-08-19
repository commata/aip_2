# Guidance Hybrid v2 Causal Action Ablation 결과

## 결론

`HARD_BLOCKED / REVALIDATION_FAILED / NOT_PROMOTED`다. 실제 simulator에서 총 810 rollout, 630 clean nondefault pair를 실행했지만 Pure BT보다 반복적으로 좋은 bounded Guidance action/state bucket을 찾지 못했다. Dataset v2와 Advantage/PPO 학습은 causal phase gate를 통과하지 못했으므로 실행하지 않았다.

## 실행 범위

- primary action: `BT_DEFAULT`, `VP_AZ_POS_SMALL`, `VP_AZ_NEG_SMALL`, `VP_EL_POS_SMALL`, `VP_EL_NEG_SMALL`
- angular magnitude: `0.10°`, `0.25°`, `0.50°`
- duration: `6/12/18/24/36 frame`
- geometry: lateral left/right, vertical high/low, crossing left/right
- 추가 분리: near-shot 18 state, target-high focus 30 state
- paired baseline: 동일 초기 state의 실제 Pure BT
- clean nondefault pair: 630
- BT_DEFAULT parity pair: 90, 모두 exact Damage parity 및 intervention 0
- ownship/target crash, contamination, process error, throttle violation: 모두 0

## 단계별 결과

| 단계 | rollout | clean pair | 결과 |
|---|---:|---:|---|
| 전체 4 action × 3 magnitude × 5 duration pilot | 372 | 360 | family 편중 gate 후 signal 0 |
| EL+ 0.25° 18/24/36f 18-state 재검증 | 90 | 54 | positive 최대 38.9%, median 0 |
| near-shot benchmark 재설계 후 재검증 | 90 | 54 | large regression 및 family 편중 |
| AZ±/EL± signed near-state 재검증 | 108 | 72 | 모든 고정 action gate 실패 |
| target-high/AZ+ 0.10/0.25/0.50° focus | 150 | 90 | 세 magnitude 모두 gate 실패 |

## 마지막 가설 반증

초기 18-state signed 비교에서 target-high/AZ+가 3/3 양수였기 때문에 이 bucket을 30개 독립 state로 확장했다. 그러나 36-frame 결과는 다음과 같다.

| magnitude | mean ΔDamage | median | positive | large regression | min |
|---:|---:|---:|---:|---:|---:|
| 0.10° | +0.0010731 | 0 | 43.3% | 1 | -0.0256755 |
| 0.25° | +0.0078592 | 0 | 43.3% | 2 | -0.0358798 |
| 0.50° | +0.0205402 | 0 | 43.3% | 1 | -0.0154476 |

평균은 소수의 큰 양수에 의해 상승했지만 median과 positive ratio가 실패했고 large regression tail이 남았다. magnitude를 키워 평균만 높이는 것은 제출 후보 근거가 아니다.

## 안전 및 controller 계약

- `vp_error_pd_v2`는 angular degree에 비례하는 실제 surface correction을 생성한다.
- LOS-rate damping, speed/altitude dynamic-pressure proxy, directional headroom/saturation을 반영한다.
- throttle은 exact Pure BT다.
- 단일 controller smoke와 90개 parity pair에서 `BT_DEFAULT`가 Pure BT와 일치했다.
- 최대 selector latency는 `0.0921 ms`, 166.7ms 초과는 0이다.

## 연구 중단 근거

Hard Stop D를 충족한다. 네 signed action, 세 magnitude, 다섯 duration을 여러 geometry와 재설계 benchmark에서 반복 검증했으나 multi-geometry positive signal이 남지 않았다. 첫 positive-looking bucket도 30-state 독립 재검증에서 실패했다.

따라서 다음을 실행하지 않았다.

- Dataset v2: `SKIPPED_CAUSAL_GATE_FAILED`, unique state 0, rollout 0
- Advantage ensemble: seed/epoch/update 0
- Shadow/micro intervention: 0
- development/held-out: candidate가 없어 미실행, held-out 미개봉
- PPO: iteration/environment step/episode 0
- Hybrid v2 submission bundle: 미생성

최종 권장은 기존 hash가 검증된 Pure BT Champion fallback이다.
