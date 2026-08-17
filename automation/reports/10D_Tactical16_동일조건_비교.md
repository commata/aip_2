# 10D와 Tactical16 동일 조건 비교

## 가설

제출 관측 계약을 고정한 Tactical16이 기존 `aim_residual10_v2`보다 rear120 조준 보정의 seed 재현성과 Damage chain을 개선할 수 있다.

## 변경 변수와 고정 변수

변경 변수는 observation mode와 독립 training seed뿐이다. SAC 128x128, reward, rear120 Gate, residual `0.125`, saturation-aware composition, throttle BT 고정, target/geometry 분포, 20 iteration과 sampled step `2,560`을 고정했다.

## 정책 hash

| 관측 | seed | weights SHA256 |
|---|---:|---|
| 10D | 3101 | `B5301A3FB0C708E341E9A3100993D88E9D0F6AD9FA9A29BB731B1BA3535EB3B7` |
| 10D | 3102 | `62B57782D7F3A7CBDA8B617D862C7A3F0694BEF4EDB656D4916F4348B7F2460D` |
| Tactical16 | 3101 | `2790F3303CAF98DD78F5889038601EADCE44D0B012FC0004BB6946F96C23C57F` |
| Tactical16 | 3102 | `A836531283CC1B20BC8A700161343B83267BD15CB4B701E3FBF8DB88715F0E2D` |

## 동일 six-geometry paired 결과

수치는 Hybrid minus Pure 0815의 6-pair 평균이다.

| 관측/seed | Damage | First Damage(s) | LOS(°) | LOS-rate(°/s) | Cone(s) | min altitude(m) | saturation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10D/3101 | -0.001661 | -0.0111 | -0.0187 | -0.0103 | -0.0167 | -0.38 | 0.1731 |
| 10D/3102 | -0.000514 | +0.0083 | -0.0152 | -0.0076 | -0.0167 | -0.86 | 0.0822 |
| T16/3101 | -0.002256 | +0.0028 | -0.0524 | -0.0045 | -0.0083 | -3.87 | 0.1464 |
| T16/3102 | +0.000323 | -0.0250 | -0.0213 | -0.0073 | +0.0083 | -0.06 | 0.1179 |

모든 실행에서 ownship crash는 0, clipping은 0이었다. Hybrid inference P95는 `0.53~0.57ms`였다. 이 suite는 rear/offensive 초기 상태라 최종 Gate active ratio가 `99.37~99.49%`로 지나치게 높다. 따라서 Gate 선택성 비교가 아니라 정책이 이미 적격인 구간에서 만드는 trajectory 차이만 보여준다.

## 잠정 결론과 반대 가설

T16 seed 3102만 First Damage/Cone 방향이 좋아졌지만 seed 3101은 Damage가 6개 기하 모두 감소했다. 두 seed 평균 Damage도 T16 약 `-0.000966`, 10D 약 `-0.001088`로 둘 다 Pure BT보다 낮다. 평균 LOS는 T16이 더 낮지만 이것이 안정적인 Damage 증가로 이어지지 않았다.

반대 가설은 T16의 정보가 부족한 것이 아니라 pilot budget이 너무 짧아 seed variance가 지배했다는 것이다. 그러나 사용자 기준상 두 seed가 반대면 장시간으로 덮지 않고 `REVALIDATION_FAILED`로 판정해야 한다.

## 판단

- 10D: `FROZEN_10D_REFERENCE`는 기존 S2101 000300을 유지하며, 신규 short pilot은 승격하지 않는다.
- Tactical16: `REVALIDATION_FAILED / NOT_PROMOTED`
- scale 비교·장시간 연장·200초 전체전: 후보 freeze 조건 미충족으로 실행하지 않음
- 최종 Target BT: `TARGET_BT_PENDING`

신뢰도는 `중간`이다. 두 독립 training seed와 6개 mirror geometry에서 방향 불일치가 확인됐지만, 실제 최종 Target과 200초 전체전은 아직 없다.
