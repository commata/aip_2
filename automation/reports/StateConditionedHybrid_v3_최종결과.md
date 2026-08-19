# State-Conditioned Counterfactual Hybrid v3 최종 결과

## 결론

Hybrid v3는 최종 Promotion Gate를 통과하지 못했다. 따라서 submission bundle과 default config를 생성하지 않았고, Pure BT Champion을 최종 제출안으로 유지한다.

최종 상태는 `NOT_PROMOTED / PURE_BT_FALLBACK_RECOMMENDED / SERVER_BLOCKED`다.

## 보존 기준

- 시작 main: `5e52665ab0be12006c777aec61b44149fef1f591`
- Pure BT DLL: `4C93B4C6719CB0423388D5FC721D356020A3A36CD5AD2C56B5C3CA795BFE18C9`
- Pure BT XML: `D84C27B0B8BA22E1649AF2375BE0B83C762BC62EB5047BB590539B374F8271EE`
- Hybrid v1 model: `FAD2BCB268D4669BD65DB89CE8F36832B1E8C4B646163897E9BBC2A6372FBB6C`

기존 v1/v2 artifact와 raw telemetry는 삭제하거나 덮어쓰지 않았다.

## 데이터와 Oracle

Stage 1/2의 444-state acquisition에 평가 경계 120 state와 실제 Shadow decision trace 48 state를 추가했다. 최종 재구성 matrix는 678 unique state, 3,078 observed nondefault pair다. 학습에는 action coverage가 동일한 612 state, 2,448 state-action pair만 사용했다.

평가 경계 120 state의 0.25도/36-frame 단일 pulse는 네 action 모두 Damage delta 0이었다. 실제 decision-time 48 state에서 Level 1 oracle은 mean `+0.00009949`, median `0`, positive `20.83%`였다.

Action Space Ladder 결과:

- Level 1, single axis: mean `+0.00009949`, median `0`, positive `20.83%`
- Level 2, two axis 포함: mean `+0.00010092`, median `0`, positive `20.83%`
- Level 3, rate-aware 포함: Level 2와 동일하여 추가 oracle uplift 0
- Level 4, 8방향 x 0.10/0.50도 balanced 12-state screen: 192 nondefault pair 전부 Damage delta 0

Level 2의 mean 증가는 `+0.00000144`에 불과했고 positive coverage를 늘리지 못했다. Level 3/4도 새로운 positive family를 만들지 못했다.

## Advantage model

모델은 42D server-safe state와 6D factorized action을 입력으로 하는 48-64-64 MLP ensemble이다. 세 seed `31001/31002/31003`, 각 160 epoch, 총 2,880 optimizer update를 실행했다. 출력은 mean advantage, P(positive), P(large regression)이다.

최종 개발 모델의 group-OOF 결과:

- coverage: `1.307%`
- intervention precision: `75.0%`
- mean offline policy value: `+0.00038301`
- large regression: `0%`
- 일치 seed: `2`
- Pearson/Spearman: `0.3850 / 0.3300`
- top-action agreement: `36.27%`
- oracle regret: `0.00082765`

선택 threshold는 `score > 0.002`, `Ppositive > 0.6`, `Pregression < 0.2`, `lambda=2`다. OOD는 nearest-training-state RMS-z threshold 2.0을 넘으면 exact `BT_DEFAULT`로 abstain한다.

## Shadow와 Micro

첫 모델의 Micro 12 clean pair는 mean `-0.00002154`, median `0`, positive `8.33%`, bootstrap CI `[-0.00009571, +0.00004570]`로 실패했다. 손실은 lateral-left에 집중됐고 학습 range 650~850m 대비 평가 range 1.1~1.3km의 OOD가 확인됐다.

평가 경계와 dynamic trace를 추가한 최종 모델은 offline gate를 통과했지만 6-geometry Shadow에서 nondefault 예측이 0이었다. Shadow command parity, crash, invalid, throttle, latency는 모두 안전했지만 실개입 가능성이 없으므로 Shadow gate는 실패다.

- Shadow clean pair: 6
- predicted nondefault frame: 0
- actual intervention frame: 0
- Damage delta: 정확히 0
- E2E P99 max: `0.9839 ms`
- E2E MAX: `1.7184 ms`
- 166.7ms 초과: 0

따라서 새 Micro, Development, Held-out, PPO는 실행하지 않았다. held-out을 tuning data로 열지 않았다.

## Hard Stop 근거

현재 평가 trajectory에서 Level 1 oracle positive coverage가 20.83%에 그쳤고 median은 0이었다. Level 2와 Level 3는 의미 있는 oracle uplift를 만들지 못했으며, Level 4 bounded grid도 balanced screen에서 전부 0이었다. server-safe 관측과 conservative OOD fallback을 적용한 두 후속 ensemble 모두 Shadow에서 실개입을 생성하지 못했다.

이는 단순 fixed-action positive ratio만으로 중단한 것이 아니다. 실제 decision-time state에서 Action Space Ladder Level 1~4를 순차적으로 검증했고, positive oracle의 희소성과 out-of-sample non-actionability를 함께 확인했다.

## 안전성과 재현성

이번 추가 causal 연구는 1,800 rollout을 실행했다. crash, contamination, process error, throttle violation은 모두 0이며 exact BT_DEFAULT parity를 유지했다. 최종 전체 테스트는 `249 passed, 26 subtests passed`, compileall은 성공했다.

최종 dataset SHA256은 `4DFBF311AA680AA8F8054F9ED969A0729CAB89B1A1EEEFF0D7D8492F8614D6A3`, 마지막 개발 model SHA256은 `7B97E8DF6CEA3250AE873FAC00C77C1B4AE923B12D0AC596C0DC25F1F36550F2`다. 이 모델은 제출 후보로 동결하지 않는다.

## 권장

대회 제출에는 hash가 검증된 Pure BT Champion을 사용한다. 향후 Hybrid 재개는 현재 pulse magnitude/방향의 추가 sweep이 아니라, Pure BT 내부 pursuit mode 선택 또는 더 긴 horizon에서 Damage causal signal을 만드는 별도 action contract로 새 version에서 시작해야 한다.

`NOT_PROMOTED`

`PURE_BT_FALLBACK_RECOMMENDED`

`SERVER_BLOCKED`
