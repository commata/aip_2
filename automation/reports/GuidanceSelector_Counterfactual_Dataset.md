# Guidance Selector counterfactual dataset

## 결론

frozen 2초 horizon에서 실제로 서로 다른 초기 state 100개와 9개 Guidance action을 모두 실행해 900 rollout을 완료했다. process error, ownship/target crash, target-crash contamination, throttle violation은 모두 0이다.

clean nondefault 800 pair의 `candidate health margin - BT_DEFAULT health margin`은 평균 `+0.0003915174`, median `0`, min `-0.2727009613`, max `+0.3414904163`이다. positive pair는 `97/800`(12.125%), 사전 의미 임계값 `+0.001` 이상은 54건이다. 반면 `-0.003` 미만 large regression도 44건이므로 이 결과만으로 policy 성능을 승격하지 않는다.

## Dataset 구성

- seed: `8601~8700`
- 실제 unique initial state: 100
- action/state: 9
- rollout: 900
- horizon: 2초(120 simulator frame 제한)
- family: lateral left/right, vertical high/low, crossing left/right
- 변동: range, lateral/vertical offset, own/target speed, own/target heading, altitude
- randomization: off; seed 번호만 바꾼 동일 trajectory를 독립 state로 세지 않음
- phase: short counterfactual은 실제 Phase 1만 포함. Phase 2/3은 후속 200초 raw telemetry에서 검증한다.

## Label

각 state에서 crash/contamination 없는 action 중 Damage delta가 `+0.001` 이상인 최선 action만 nondefault label로 사용했다. 나머지는 `BT_DEFAULT`다.

| label | sample |
|---|---:|
| `BT_DEFAULT` | 82 |
| `VP_EL_POS_SMALL` | 10 |
| `VP_EL_NEG_SMALL` | 3 |
| `VP_AZ_POS_SMALL` | 2 |
| `VP_RANGE_FORWARD_SMALL` | 1 |
| `VP_RANGE_BACKWARD_SMALL` | 1 |
| `TARGET_SPEED_UP_SMALL` | 1 |

100개 observation은 모두 45D finite float로 확인했다. negative/unsafe action을 label에서 제거한 conservative filtered BC-only dataset이며, classification 성능만으로 승격하지 않는다.

## Artifact

- root: `artifacts/evaluations/guidance_selector/counterfactual_v1_20260819`
- `aggregate.json`: `DFBC301471ACA4FB1A2194FE08EB224C4B6B114F17F389C59219CF2EA7079A9A`
- `dataset.json`: `291E93BCA7D8F329C3E9A7D1B48A2BB1E289535940C6720107895B583A52D5C0`
- `records.json`: `79612EA4D45EB593CA5831A5F3F2407984EDBD21C30986739333440252FCF3C3`
- `episode_records.csv`: `51C3FF9F95029277263BFA9031E5DD6BD3A4E27912CB8332B67ACF8ACEE5944A`
- wall time: 845.652초

