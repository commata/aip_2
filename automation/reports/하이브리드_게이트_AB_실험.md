# 0815 하이브리드 Gate A/B 실험

## 결론

현재 학습 정책에서는 Aim Gate A가 Offensive Gate B보다 상대적으로 안정적이지만, 두 Gate 모두 Pure 0815 BT 대비 승격할 수준의 조준 및 Damage 개선을 만들지 못했다. Gate A는 `CHANGE_HYPOTHESIS`, Gate B는 `ROLLBACK`이다.

## 공통 조건

- 공식 0815 BT DLL/XML 고정
- SAC, `aim_residual10`, aim residual reward
- additive가 아닌 `saturation_aware` 결합
- throttle residual 0
- residual scale 0.125
- RL repeat 6 frame
- 혼합 기하 validation seed 1401, 1402, 1403, 1407, 1410
- 정상 격추 3건과 target altitude crash 2건을 분리 판정

## Gate 정의

### Gate A: 조준점 근접형

- Phase 공식 반각에 진입 +7도, 이탈 +10도
- Phase 공식 사거리에 진입 +300m, 이탈 +550m
- 최소 거리 152.4m
- 최소 유지 12 simulator step

### Gate B: Offensive 상황형

- 진입 거리 1,500m 이하
- ownship ATA 15도 이하
- target ATA 135도 이상
- 이탈 거리 1,750m, ownship ATA 22도, target ATA 120도

## 동일 조건 결과

| 지표 | Gate A | Gate B |
|---|---:|---:|
| Gate 활성률 | 83.9% | 95.0% |
| Damage delta | +0.00293 | +0.03125 |
| 정상 격추 Damage delta 평균 | +0.00051 | +0.00002 |
| 평균 LOS delta | +0.01379도 | +0.06570도 |
| LOS median delta | -0.00970도 | -0.06448도 |
| LOS P95 delta | +0.39912도 | +1.25453도 |
| LOS rate RMS delta | +0.01153도/초 | -0.00248도/초 |
| Cone 체류 delta | +0.03667초 | +0.22000초 |
| 최초 Damage delta | +0.00667초 | +0.27000초 |
| clipping | 0% | 0% |
| 조종면 포화 frame 비율 | 31.8% | 39.9% |
| ownship crash | 0 | 0 |

## 해석

Gate B의 전체 Damage 증가는 target crash 한 건에 크게 의존하며 정상 격추에서는 효과가 사실상 0이다. Gate B는 Stage 1 공격 초기조건에서 거의 상시 활성화되어 RL 역할을 최종 조준 보정으로 제한한다는 구조적 목표에도 Gate A보다 불리하다.

Gate A는 평균/P95 LOS 악화폭과 개입률이 Gate B보다 낮다. 그러나 Pure BT보다 최초 Damage가 늦고 평균/P95 LOS가 나빠 현재 정책을 승격할 수 없다. Gate A를 단순히 +2/+4도 및 100/250m로 축소한 ablation은 활성률을 55.7%로 낮췄지만 평균 LOS +0.08825도와 P95 +1.79472도로 더 악화됐다.

## Residual scale 비교

Gate A final 정책의 0.10/0.125/0.15 비교에서 0.125가 평균/P95 LOS 악화가 가장 작았다. 0.15는 정상 격추 Damage가 가장 높았지만 최초 Damage 지연과 P95 악화가 더 컸다. 따라서 연구 기준은 0.125를 유지하며 최종 선정으로 해석하지 않는다.

## 다음 가설

Gate 안에서 실제 signed aim error가 작음에도 관측 정규화가 azimuth ±180도, elevation ±90도, ATA 0~180도를 사용해 입력 신호가 지나치게 압축된다. feature와 10차원 구조는 유지하고 근접 조준 범위에 맞춘 정규화만 적용해 재학습한다.
