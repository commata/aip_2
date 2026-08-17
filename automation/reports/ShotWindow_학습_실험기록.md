# Shot-Window 학습 실험 기록

## 고정 조건

- algorithm: SAC, MLP 128×128
- observation: Tactical16 v1
- residual scale: 0.125
- composition: saturation-aware
- action: roll+pitch+yaw residual, throttle BT-only
- RL cadence: 6 frame
- training seeds: 5101, 5102; 충돌 시 5103
- short budget: seed당 20 iteration
- primary metric: Hybrid Damage - Pure 0815 BT Damage

## 실험 대기열

1. 기존 checkpoint + Shot-Window inference diagnostic
2. diagnostic에서 선택한 v1 1.0초/condition-exit short pilot seed 5101/5102
3. paired development screening seed 5201~5206
4. 통과 시에만 held-out seed 5301~5306

## 기존 checkpoint 진단 결론

s3101/s3102에서 1.0초 window의 pooled clean mean Damage Δ는 +0.000300, positive 8/12였다. 이는 Gate 구조 선택 근거일 뿐 새 policy의 승격 근거가 아니다. 0.25/0.50/0.75/1.50초 후보는 각각 너무 짧거나 pooled 재현성/추가 이득이 부족해 1차 학습에서 제외했다.

## Short v1 — broad rear/offensive curriculum

seed 5101/5102를 각각 20 iteration, 2,560 env step 학습했다. 5/10/15/20 iteration bundle을 모두 동일 development screening 6-case로 paired 평가했으며 target crash와 ownship crash는 없었다.

| seed | iter 5 | iter 10 | iter 15 | iter 20 | iter 20 positive |
|---:|---:|---:|---:|---:|---:|
| 5101 | -0.000446 | -0.002262 | -0.001008 | -0.000890 | 2/6 |
| 5102 | -0.000427 | -0.001015 | -0.000961 | -0.001499 | 1/6 |

모든 checkpoint의 clean mean Damage Δ가 음수라 checkpoint cherry-pick을 중단했다. iteration 20 window active ratio는 두 seed 모두 약 8.52%, ACTIVE max 1.0초, RL inference 평균 14회로 runtime Gate contract는 정상이다. 그러나 학습 완료 episode에서 gate active ratio 평균은 seed 5101 1.94%, seed 5102 6.15%에 불과했다. Damage reward contribution은 각각 평균 6.72와 15.63으로 shaping보다 컸으므로 reward가 Damage를 가리는 근거는 약하다.

판정: `REVALIDATION_FAILED / NOT_PROMOTED`. 다음 단일 변수는 reward나 axis가 아니라, policy가 실제 window transition을 충분히 보도록 damage 직전/cone 직전에서 시작하는 Stage-1 curriculum이다.

현재 상태: `SHORT_TRAINING`, 후보 없음.

## Stage-1 shot-window curriculum v1

Gate 내부 경험 부족을 분리하기 위해 880m, cone center/edge 5개 geometry에서 1.2초 episode를 시작했다. 나머지 Gate/reward/action/observation/scale/target pool은 고정했다. 두 seed 모두 20 iteration/2,560 step에서 213 episode를 완료했다.

| seed | clean mean Damage Δ | positive | LOS Δ | Cone Δ |
|---:|---:|---:|---:|---:|
| 5101 | -0.000573 | 1/6 | +0.00022° | +0.0056초 |
| 5102 | -0.000618 | 1/6 | +0.00146° | 0초 |

broad v1보다 평균 회귀는 작아졌지만 두 seed 모두 promotion 기준을 통과하지 못했다. 이어서 residual scale 0.125를 window 시작에서 유지하고 1.0초 끝에서 0까지 선형 감쇠하는 Family B를 inference-only로 비교했다. s5101 -0.001196(2/6), s5102 -0.001175(1/6)로 더 악화되어 재학습 없이 기각했다.

raw policy action의 successive max jump는 s5101 0.132, s5102 0.042였고 실제 scale 적용 후 각각 0.0165/0.0053 이하이며 seed 간 크기도 일관되지 않았다. 따라서 slew가 공통 원인이라는 evidence가 부족해 Family C rate-limit 실험은 보류한다.

반면 random-init Stage-1 policy의 signed action 방향은 두 seed에서 거의 반대였고, 기존 s3101/s3102 checkpoint는 새 1.0초 Gate inference diagnostic에서 각각 clean mean Damage Δ +0.000392/+0.000208이었다. 다음 최소 가설은 이 양수 초기점을 각각 유지한 채 동일 Stage-1 curriculum으로 20 iteration fine-tuning하는 것이다.

## Paired warm-start Stage-1

s3101→seed5101, s3102→seed5102로 bundle 계보를 고정해 각각 20 iteration fine-tuning했다. 로드 로그로 두 초기화가 정확히 대응됨을 확인했다.

| seed | iter 5 Damage Δ | iter 5 positive | iter 20 Damage Δ | iter 20 positive |
|---:|---:|---:|---:|---:|
| 5101 | -0.000368 | 4/6 | -0.001046 | 4/6 |
| 5102 | -0.002031 | 3/6 | -0.000919 | 1/6 |

iteration 5부터 두 seed mean이 모두 음수였고 seed5102 crossing-left는 Damage Δ -0.013294, LOS Δ +3.879°의 대규모 회귀를 보였다. 따라서 fine-tuning checkpoint를 cherry-pick하지 않는다. Gate timing, curriculum sample density, reward dominance, decay, warm initialization을 분리한 뒤에도 SAC update가 seed별 상반 방향으로 policy를 이동시키는 evidence가 남았다.

판정: `REVALIDATION_FAILED / NOT_PROMOTED`. repository가 실제 지원하는 PPO를 동일 Stage-1 Gate/reward/action/observation/seed/budget에서 algorithm만 바꾸는 최후 비교로 진행한다.

## PPO 단일 변수 비교

SAC와 동일한 Stage-1 Gate/reward/action/observation/scale/seed/budget을 유지하고 algorithm만 PPO로 바꿨다. seed당 20 iteration, 2,560 env step, 213 episode를 완료했다.

| seed | clean mean Damage Δ | positive | 오염 제외 | 최소 geometry Δ | crash |
|---:|---:|---:|---:|---:|---:|
| 5101 | -0.000769 | 2/5 | crossing-left target crash 1건 | -0.003994 | 0 |
| 5102 | +0.001316 | 4/6 | 없음 | -0.001118 | 0 |

두 seed의 방향이 충돌하므로 양수 seed 5102만 선택하지 않는다. 기존 manifest에 사전 고정한 conflict seed 5103을 같은 계약으로 추가해 알고리즘 효과의 재현 방향을 판별한다. held-out seed 5301~5306은 아직 열지 않았다.

seed 5103은 최초 r1에서 Ray logger sandbox 권한 오류로 sample 0에서 중단됐다. artifact를 보존하고 고유 r2 run ID로 같은 계약을 재실행해 20 iteration/2,560 step/213 episode를 완료했다. iteration 20 clean mean Damage Δ는 +0.000357, positive 3/6, 최소 geometry Δ -0.003503이었다.

## PPO checkpoint screening과 최종 판정

development set에서 periodic 5/10/15/20 checkpoint를 세 training seed 모두 같은 시점으로 비교했다.

| iter | seed 5101 Δ | seed 5102 Δ | seed 5103 Δ | pooled positive | 오염 | 최악 geometry Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | -0.000167 | -0.000697 | +0.000397 | 7/18 | 0 | -0.005690 |
| 10 | +0.000383 | +0.000820 | +0.000464 | 7/15 | 3 | -0.004521 |
| 15 | +0.000667 | +0.000930 | +0.002072 | 8/16 | 2 | -0.003960 |
| 20 | -0.000769 | +0.001316 | +0.000357 | 9/17 | 1 | -0.003994 |

iteration 15는 세 seed 평균 방향은 양수지만 pooled positive가 정확히 50%이고 target-crash 오염 2건, geometry 회귀가 남았다. iteration 20은 pooled positive가 9/17이지만 seed 5101 평균이 음수다. 따라서 어느 시점도 `SHORT_PROMOTE_CANDIDATE`의 두 독립 seed 일치, 절반을 명확히 넘는 pooled positive, 대규모 geometry 회귀 없음 조건을 동시에 만족하지 않는다.

iteration 15 clean pair를 축별로 분석했을 때 음수/양수 pair의 평균 correction은 roll 0.04047/0.03585, pitch 0.03203/0.03011, yaw 0.01793/0.01838이었다. roll 차이는 작고 seed/geometry 전반의 공통 원인으로 분리되지 않았으며 이전 roll-only 실패도 있어 axis mask 재학습 근거로 사용하지 않았다. Damage reward가 shaping보다 우세했던 기존 contribution 분석 때문에 reward 변경 근거도 없고, Tactical16의 구분력 부족 증거도 없다. 따라서 reward/observation/추가 algorithm을 근거 없이 함께 바꾸지 않는다.

총 학습량은 성공한 run 기준 180 iteration, 23,040 env step, 1,533 episode다. sample 0에서 권한 오류로 중단된 PPO seed5103 r1은 계산에서 제외했다.

최종 검증: automation `140 passed, 26 subtests passed`, core `9 passed`, compileall 통과, manifest JSON 23개 parse 성공, `git diff --check` 통과.

최종 상태: `DIAGNOSTIC_COMPLETE / REVALIDATION_FAILED / NOT_PROMOTED / TARGET_BT_PENDING / SERVER_BLOCKED`.
