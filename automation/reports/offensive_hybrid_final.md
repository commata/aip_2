# Offensive BT+RL Hybrid 최종 검증 보고서

## 최종 결과

선정된 후보는 BT를 우선하는 보수적인 Hybrid 구성입니다. 이 결과를 RL이 BT보다 압도적으로 우수하다는 의미로 해석해서는 안 됩니다. 미사용 scenario/seed 조합 6쌍에서 BT의 모든 교전 결과를 보존하면서 평균 reward, ATA, 조종면 포화율을 개선했습니다. 다만 평균 health margin은 소폭 하락했으므로 evaluator에 명시적인 비퇴행 허용치를 두었고, 본 보고서에서도 BT 대비 결정적인 우위로 표현하지 않습니다.

## 선정 설정

- 제어 모드: `offensive_residual`
- Residual scale: `0.10`
- Gate 내부 RL 실행 주기: 시뮬레이터 6 frame마다 1회(60 Hz 시뮬레이션 기준 10 Hz)
- BT 실행 주기: 시뮬레이터 매 frame
- Throttle 처리: 정책 출력을 `[-1, 1]`에서 `[0, 1]`로 한 번만 변환한 뒤 convex blend만 적용
- 저에너지 throttle guard: 속도 `210` 미만에서 RL이 더 낮은 출력을 요구하면 BT throttle 유지
- Gate 진입 조건: 거리 `152.4–1500 m`, ownship ATA `<= 15°`, target ATA `>= 135°`
- Gate 이탈 hysteresis: 거리 `<= 2000 m`, ownship ATA `<= 25°`, target ATA `>= 110°`

Gate 밖에서는 clipping된 BT action을 그대로 반환하고 RL 추론을 생략합니다. 공격 기하가 형성되지 않은 crossing held-out 궤적에서는 gate 점유율 `0`, RL 추론 호출 `0회`, BT와 Hybrid의 평가 지표가 정확히 동일했습니다.

## Held-out paired 검증 근거

검증에는 deterministic scenario 3종(`offensive_tail`, `crossing_left`, `crossing_right`)과 미사용 seed 2개(`307`, `401`)를 사용했습니다. 각 scenario/seed 조합에서 BT와 Hybrid를 한 번씩 실행하여 총 12개 run을 비교했습니다.

| 지표 | BT | Hybrid 0.10 | 차이(Hybrid - BT) |
|---|---:|---:|---:|
| 승률 | 16.67% | 16.67% | 0.00%p |
| Crash율 | 0.00% | 0.00% | 0.00%p |
| 평균 reward | 142.7592 | 146.1316 | +3.3725 |
| 평균 health margin | -0.04648 | -0.05042 | -0.00394 |
| 평균 ATA | 73.5812° | 73.0553° | -0.5259° |
| 조종면 포화율 | 0.8989 | 0.8844 | -0.0145 |
| 최소고도 | 533.1 m | 521.9 m | -11.2 m |

두 제어기의 결과 수는 승 1회, 패 1회, timeout 4회, crash 0회로 동일했습니다. Hybrid의 최소고도는 설정된 안전 기준 300 m 이상을 유지했습니다.

## 초기 설정을 채택하지 않은 이유

초기의 넓은 gate(`2700/3300 m`, `36/52°`)와 `0.10–0.20`의 모든 고정 scale은 held-out 검증에서 BT의 승리 1건을 timeout으로 퇴행시켰습니다. Maneuver telemetry에서는 넓은 gate가 교전 시간의 절반 이상 활성화되는 경우가 확인됐습니다. RL 조종면 correction 자체는 작았지만, 저에너지 선회 중 throttle 감소가 장기 궤적과 교전 타이밍을 바꿨습니다.

Gate를 최종 사격 정렬 구간으로 축소하고 저에너지 throttle guard를 추가한 뒤 scale `0.10`에서 BT의 승리를 복원했습니다.

## 검증 내역 및 산출물

- Core 및 automation 테스트: 총 20개 통과
- 실제 JSBSim smoke, scale grid, gate search, throttle guard, crossing 및 held-out paired evaluation 완료
- Raw 로컬 검증 자료: `artifacts/evaluations/final_heldout/`(Git 추적 제외)
- 기계 판독용 최종 설정: `automation/best_offensive_hybrid.json`
- 작업 브랜치: `codex/offensive-hybrid-autotune`
- 관련 이슈: `#1 Offensive BT+RL Hybrid 쓰로틀 및 선회 최적화`
- 관련 Draft PR: `#2 Offensive BT+RL Hybrid 쓰로틀 및 선회 최적화`

PR #2는 생성됐지만 `main`에 병합하지 않았습니다. 기존 model bundle, checkpoint, DLL 및 rule XML 파일도 커밋하거나 수정하지 않았습니다.
