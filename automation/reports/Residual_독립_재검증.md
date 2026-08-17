# Residual 독립 재검증

## 최종 판단

- branch: `codex/offensive-residual-causal-rework`
- 연구 결과 기준 HEAD: `37d43c8`
- 관련 Issue: #11
- 선행 PR: #10
- 상태: `DIAGNOSTIC_COMPLETE / TRAINING_COMPLETE / REVALIDATION_FAILED / NOT_PROMOTED`
- candidate: 없음
- 외부 상태: `TARGET_BT_PENDING / SERVER_BLOCKED`

이번 branch에서는 Pure 0815 BT가 전체 trajectory를 담당하고 RL이 공격 기회에 Damage를 늘리는 미세 조준 보정만 수행하도록 원인을 분석하고 action, reward, Gate 가설을 하나씩 검증했다. 어느 가설도 두 독립 training seed에서 contamination-free Damage 개선 방향을 재현하지 못했다. 따라서 checkpoint를 승격하지 않고 장시간 학습, scale sweep, 200초 full-fight를 실행하지 않았다.

## Causal conclusion

Tactical16 지정 10/10 trajectory와 frozen 10D 12/12 trajectory에서 다음 순서가 재현됐다.

`작은 residual → final command divergence → 3~5 frame 뒤 state divergence → 다음 frame BT raw command divergence → LOS/Cone/Damage divergence`

Gate가 약 99% 활성인 상태에서는 절대값 0.001~0.007의 correction도 한 번의 미세 보정이 아니다. 바뀐 state를 60Hz stateful BT가 매 frame 다시 읽으면서 command 차이를 증폭해 전체 공격 trajectory를 바꾼다. Tactical16 feature 수 자체가 원인은 아니었다.

## 독립 실험 결과

모든 Δ는 Hybrid - Pure 0815다. crossing-left는 Pure/Hybrid 모두 target crash가 발생해 `TARGET_CRASH_CONTAMINATED`로 표시하고 5-geometry 주 판정에서 제외했다.

| 가설 | training/reused seed | clean mean Damage Δ | positive pair | 95% bootstrap CI | 판단 |
|---|---:|---:|---:|---:|---|
| 기존 checkpoint roll-only axis diagnostic | reused 3101 | +0.000761 | 3/5 | [-0.001237,+0.002926] | 학습 전 진단만 양수 |
| 기존 checkpoint roll-only axis diagnostic | reused 3102 | +0.001876 | 4/5 | [-0.001345,+0.004623] | 학습 전 진단만 양수 |
| effective roll-only short training | 4101 | -0.001260 | 1/5 | [-0.003393,+0.000834] | 실패 |
| effective roll-only short training | 4102 | -0.000371 | 3/5 | [-0.002863,+0.002092] | 실패 |
| aim_progress 제거 | 4201 | -0.001276 | 1/5 | [-0.003995,+0.001764] | 실패 |
| aim_progress 제거 | 4202 | +0.000765 | 2/5 | [-0.002375,+0.003905] | seed 충돌 |
| aim_progress 제거 제3 seed | 4203 | -0.000524 | 3/5 | [-0.003375,+0.001904] | 실패 판별 |
| offensive ATA 10°/15° Gate diagnostic | reused 3101 | -0.001704 | 2/5 | [-0.004294,+0.000330] | 학습 전 탈락 |
| offensive ATA 10°/15° Gate diagnostic | reused 3102 | -0.000186 | 2/5 | [-0.001377,+0.001045] | 학습 전 탈락 |

### Action 가설

공동-action checkpoint의 inference-time ablation에서는 roll-only만 두 seed의 clean Damage 방향이 양수였다. 그러나 실제 roll-only로 새로 학습한 seed 4101·4102는 모두 aggregate Damage가 감소했다. 공동-action checkpoint 마스킹은 분포 밖 diagnostic이었으며 effective action space의 성능 근거가 아니었다.

### Reward 가설

기존 T16 training에서 `aim_progress` 평균 기여는 +6.38/+4.20으로 Damage 다음으로 컸고 episode 변동폭도 컸다. 이 term 하나만 제거했지만 seed 4201·4202의 방향이 충돌했고, 코드·contamination·고정 Pure 결과를 확인한 뒤 추가한 seed 4203도 음수였다. 3-seed clean mean 평균은 -0.000345였다.

### Gate 가설

static suite에서 hard semantics와 rear120 hysteresis는 맞았지만 eligible-attack local episode의 Gate active ratio는 0.9943, mean active duration 706 frame(11.77s), 최대 관측 duration 1302 frame(21.70s)이었다. offensive ATA condition만 30°/45°에서 10°/15°로 좁히자 active ratio는 약 0.933까지 낮아졌지만 두 checkpoint의 clean Damage는 모두 감소했다. pre-aim OR branch 자체가 geometry별 59.6~88.9% 활성이라 leaf condition 하나만으로 짧은 shot window를 만들 수 없었다.

## Safety와 runtime 계약

- 새 short training 5개는 모두 20 iteration, 각각 sampled step 2,560, learner step 4,864를 완료했다.
- 전체 short training budget은 100 iteration, sampled step 12,800이다.
- critical NaN 0, invalid action 0, ownship crash 0, action clipping 0이었다.
- roll-only run에서 pitch/yaw correction은 정확히 0이었다.
- throttle은 모든 residual 경로에서 BT-only로 유지했다.
- Gate OFF Pure BT equality, RL inference skip, exception/NaN/timeout Pure BT fallback 계약을 유지했다.
- evaluation inference P95는 약 0.54~0.60ms였다. 이는 localhost 수치이며 실제 서버 latency로 일반화하지 않는다.
- Tactical16 observation/metadata fail-fast, 60Hz BT, action repeat 6, packet replay/UDP 계약은 변경하지 않았다.

## 오염과 반례

- crossing-left target crash는 모든 주 평가에서 `TARGET_CRASH_CONTAMINATED`이며 승격 evidence에서 제외했다.
- LOS가 감소해도 Damage가 감소한 seed 4101/4102와 4203 결과가 있었다. LOS 개선 자체는 성능 evidence가 아니다.
- seed 4202만 양수였지만 4201/4203이 음수라 cherry-pick하지 않았다.
- First Damage 또는 Cone의 한두 frame 변화도 aggregate Damage 실패를 뒤집는 근거로 사용하지 않았다.

## 미실행 항목

- `PROMOTE_CANDIDATE`: 없음
- residual scale 0.10/0.125/0.15 비교: promotion 선행 조건 미달로 미실행
- 장시간 training/checkpoint screening: 미실행
- 200초 full-fight: 미실행
- `PROMOTED_LOCAL`: 해당 없음
- `FINAL_CONFIRMED`: 사용하지 않음

## 최종 테스트

- `python -m pytest automation/tests -q -p no:cacheprovider`: 128 passed, 14 subtests passed
- `python -m pytest tests -q -p no:cacheprovider`: 9 passed
- `python -m compileall -q src automation scripts run_local_dogfight.py run_unreal_inference.py train_rllib.py`: 통과
- residual manifest JSON 전체 parse: 통과
- `git diff --check`: 통과

## Target BT와 실제 서버

Release에 존재하는 target DLL과 XML은 기존 proxy inventory이며 최종 대회 Target pair로 식별되는 신규 파일은 발견하지 못했다. 따라서 `TARGET_BT_PENDING`을 유지한다.

학생 템플릿의 IP/port/team 값은 `TODO` placeholder이고 사용할 scenario, 서버 실행 시간, OpenServer/runtime이 제공되지 않았다. 외부 join/heartbeat를 임의 실행하지 않았으며 `SERVER_BLOCKED`를 유지한다. localhost UDP 근거는 실제 서버 검증을 대체하지 않는다.

## 재현 artifact

- causal analysis: `artifacts/analysis/offensive_residual_causal_rework/phase_a_historical_20260817_r4`
- 10D cross-check: `artifacts/analysis/offensive_residual_causal_rework/phase_a_historical_10d_20260817`
- Gate selectivity: `artifacts/evaluations/offensive_residual_causal_rework/gate_selectivity_v1_20260817`
- roll-only evaluation: `artifacts/evaluations/offensive_residual_causal_rework/roll_only_short_s410{1,2}_20260817`
- reward evaluation: `artifacts/evaluations/offensive_residual_causal_rework/no_aim_progress_short_s420{1,2,3}_20260817`
- Gate ATA diagnostic: `artifacts/evaluations/offensive_residual_causal_rework/gate_ata10_diagnostic_t16_s310{1,2}_20260817`

## 다음 연구에 필요한 사항

다음 branch에서는 여러 Gate leaf threshold를 동시에 바꾸지 말고, `offensive OR pre-aim` 중복 활성 문제를 해결할 새로운 shot-window Gate 계약을 먼저 설계해야 한다. 이 계약은 사격 직전 진입 시점, 최대 intervention duration, 재진입 조건을 명시하고 static boundary suite와 기존 checkpoint inference diagnostic에서 Damage 보존을 먼저 보여야 한다. 그 전에는 새 장시간 학습을 시작하지 않는다.
