# 대체 타겟 BT 행동 분석

## 결론 요약

0815, AIP2, AIP3를 세 개의 독립 상대라고 셀 수 없다. 30초 simulator-rate 대조에서 0815와 AIP3는 default 반복 5회와 6개 명시적 geometry 모두에서 위치·자세·속도·고도·조종명령이 frame 단위로 정확히 같았다. AIP2는 일부 geometry에서 분명히 달랐으므로 별도 pursuit cluster로 유지한다.

현재 proxy cluster는 다음과 같다.

- `bt_0815_family`: 0815, AIP3
- `bt_aip2_pursuit`: AIP2
- `scripted_straight`: autopilot straight family
- `scripted_crossing`: autopilot crossing family

상태는 `PROXY_TARGET_VALIDATED`이지만 최종 대회 Target BT는 여전히 `TARGET_BT_PENDING`이다.

## 정적 inventory

| Profile | DLL SHA256 | XML SHA256 | Task_Pursue SHA256 | 행동 cluster |
|---|---|---|---|---|
| bt_0815 | `4C93B4C6...BFE18C9` | `D84C27B0...F8271EE` | `22F4A6D0...D579F4F` | bt_0815_family |
| bt_aip2 | `3AD96D6A...43B007E` | `D84C27B0...F8271EE` | `DD89D991...CB0177` | bt_aip2_pursuit |
| bt_aip3 | `815B2864...E1DB4A8` | `D84C27B0...F8271EE` | `7AC35C3F...73C2B` | bt_0815_family |

세 XML은 811 byte이며 SHA256이 완전히 같다. Controller, OptimalTurn, PurePursuit, LagPursuit, LibMain source도 세 모델에서 동일하다. 기존 소스 diff에서 AIP2의 sustained speed table과 head-on 조건 차이가 확인됐고, AIP3의 차이는 logging 활성화에 집중됐다.

## Profile loader

`automation/target_profiles`의 JSON은 profile ID, backend, DLL/XML 경로와 hash, source, smoke status, cluster, training/validation/held-out 사용 여부를 기록한다. DLL/XML 전체 경로는 profile별 환경 변수로 override할 수 있으며, 실제 SHA256이 profile과 다르면 실행을 거부한다. 새 Target BT는 코드 변경 없이 profile JSON만 추가할 수 있다.

Target identity는 observation에 주입하지 않는다. 실행 시 profile은 backend/DLL/XML/rule alias만 실험 config에 적용한다.

## 실제 DLL/XML smoke

seed 2101, 1초, 0815 residual ownship의 같은 상태에서 `bt_0815`, `bt_aip2`, `bt_aip3` 모두 실제 DLL 초기화와 59 step을 완료했다.

- observation: 10D finite
- throttle residual: 강제 0
- final throttle: BT throttle과 동일
- clipping/surface saturation: 0
- 종료: 1초 timeout

1초에는 세 상대 궤적이 같았으므로 행동 다양성 증거로 사용하지 않았다.

## 30초 default 반복의 한계

seed 2201~2205를 사용했지만 다섯 결과가 profile별로 완전히 반복됐다. `run_local_dogfight`의 해당 default scenario에서는 seed가 초기 geometry 다양성을 만들지 않았다. 따라서 이를 다섯 독립 표본으로 부르지 않고 deterministic replication으로만 취급한다.

이 반복에서도 0815와 AIP3는 각 run 1,800 frame에서 모든 RMSE가 0이었다. 0815와 AIP2는 위치 RMSE 435.47m, 자세 RMSE 56.64°, target action RMSE 1.059로 달랐다.

## 6개 명시적 geometry 대조

고정 seed 2301에서 lateral/crossing 좌우와 vertical 상하 6개 geometry를 profile별 동일 조건으로 30초 실행했다.

| Pair | 위치 RMSE | 자세 RMSE | Action RMSE | 속도 RMSE | 고도 RMSE |
|---|---:|---:|---:|---:|---:|
| 0815 vs AIP3 | 0.000m | 0.000° | 0.000 | 0.000 KCAS | 0.000m |
| 0815 vs AIP2 | 42.479m | 7.721° | 0.327 | 1.143 KCAS | 19.015m |

AIP2는 lateral-left, crossing-left, crossing-right, vertical-high에서 차이가 났고 lateral-right와 vertical-low에서는 0815와 같았다. 즉 AIP2도 모든 상태에서 독립적으로 움직이는 완전히 다른 전술은 아니며, 특정 pursue/head-on 조건이 발동할 때 갈라지는 모델로 해석한다.

vertical-high에서는 세 profile 모두 target altitude termination이 발생했다. target health가 약 0.886까지 감소했지만 격추가 아니므로 공격 성공으로 계산하지 않는다. vertical-low에서는 최저 속도가 약 55.6 KCAS까지 내려가므로 조준 reward와 별개로 저에너지 위험 사례로 유지한다.

## 학습/검증 분리

독립 BT cluster가 두 개뿐이므로 한 번에 모든 BT를 학습에 넣고 같은 모델로 평가하지 않는다.

- Fold A: scripted T0/T1 + 0815 family 학습, AIP2 validation
- Fold B: scripted T0/T1 + AIP2 학습, 0815 family validation
- AIP3: 0815 family의 DLL/rule alias 호환성 확인에는 사용하되 독립 성능 표본으로 세지 않음
- 최종 held-out: 학습에 없던 maneuver family, geometry, side, seed를 함께 유지

## 잠정 판단과 자기 반박

잠정 결론은 0815/AIP3 동일 cluster, AIP2 별도 cluster다. 신뢰도는 각각 높음과 중간이다.

이 결론이 틀렸다고 가정하면 AIP3는 30초 이후 또는 다른 BFM branch에서만 0815와 갈라질 수 있고, AIP2 차이는 제공 source가 아니라 DLL build/logging 차이의 부작용일 수 있다. 따라서 장시간 curriculum 선택 전에 최소 한 번의 200초 side reversal과 다른 초기 거리/고도에서 cluster를 재확인한다.

