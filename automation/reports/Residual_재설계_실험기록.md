# Residual 재설계 실험기록

## 상태

- branch: `codex/offensive-residual-causal-rework`
- 최신 실험 코드 기준 HEAD: `0d01cc9329cc6296ecf2069120c5c66d784872d2`
- Issue: #11
- 현재 상태: `DIAGNOSTIC_COMPLETE / TRAINING_COMPLETE / REVALIDATION_FAILED / NOT_PROMOTED`
- 외부 상태: `TARGET_BT_PENDING / SERVER_BLOCKED`

## Phase B — Gate 선택성 재검증

### 가설

기존 rear/offensive six-geometry에서 Gate active ratio가 약 99%였던 이유가 Gate 구현 오류인지, 평가 geometry가 거의 전부 `rear120 AND offensive`였기 때문인지 분리한다.

### 변경 변수와 고정 변수

Gate behavior는 변경하지 않았다. activation Gate에 active 구간의 mean/min/max duration telemetry만 추가했다. hard condition, entry/exit threshold, safety veto, observation, reward, action repeat, residual scale, BT/RL composition은 모두 고정했다.

### static selectivity suite

`residual_gate_selectivity_v1`은 다음 상태를 포함한다.

- deep rear, rear120 exact/below-entry, beam, front
- offensive, pre-aim, non-offensive
- WEZ inside, pre-aim margin, WEZ outside
- safety veto
- rear120 entry, exit, reentry, boundary oscillation
- phase1→phase2 cone boundary

12 case, 25 synthetic step 결과는 다음과 같다.

| 항목 | 결과 |
|---|---:|
| aggregate Gate active ratio | 0.6000 |
| Gate entry / exit | 9 / 2 |
| safety veto step | 2 |
| cone step | 5 |

beam, front, rear120 below-entry, deep-rear non-offensive, deep-rear low-altitude safety veto에서는 Gate가 모두 OFF였다. rear120 hysteresis trace `[119, 121, 115, 111, 109, 121]°`에서는 active가 `[OFF, ON, ON, ON, OFF, ON]`, entry 2, exit 1, mean active 2 frame, max active 3 frame이었다.

반면 deep-rear + offensive이면 WEZ 밖이고 pre-aim 밖이어도 Gate가 ON이었다. 이는 현재 계약 `rear120 AND (offensive OR phase pre-aim) AND NOT safety veto`에 부합한다.

### simulator local duration 재계측

T16 seed 3101 checkpoint의 roll-only six-geometry를 같은 조건으로 재실행해 duration telemetry를 추가했다.

| geometry | active ratio | entry / exit | mean active frame | max active frame | First Damage | cone entry | cone duration | Damage | contamination |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lateral-left | 0.9990 | 1 / 0 | 1023 | 1023 | 8.7000s | 3 | 5.1833s | 1.004063 | clean |
| lateral-right | 0.9987 | 1 / 0 | 748 | 748 | 7.1167s | 1 | 5.3667s | 1.006234 | clean |
| crossing-left | 0.9837 | 6 / 6 | 251.5 | 1302 | 14.7167s | 2 | 5.4333s | 0.614262 | `TARGET_CRASH_CONTAMINATED` |
| crossing-right | 0.9936 | 1 / 0 | 929 | 929 | 10.2333s | 1 | 5.3500s | 1.004069 | clean |
| vertical-high | 0.9988 | 1 / 0 | 845 | 845 | 8.0167s | 1 | 6.0833s | 1.004983 | clean |
| vertical-low | 0.9921 | 2 / 1 | 440 | 786 | 9.8167s | 1 | 4.9667s | 1.000807 | clean |

aggregate active ratio는 `0.994317`, mean active duration은 `706.08 frame = 11.77s`, episode별 max active duration 평균은 `938.83 frame = 15.65s`였다. crossing-left의 단일 active run은 최대 `1302 frame = 21.70s`였다. 현재 Gate는 선택 조건 자체는 정확하지만, rear/offensive episode 안에서는 RL을 짧은 사격 보정 구간으로 제한하지 못한다.

### 판단

Gate를 즉시 변경하지 않는다. Phase A에서 유일하게 두 checkpoint의 contamination-free Damage 방향을 재현한 변수는 roll-only action structure였고, Gate와 action을 동시에 바꾸면 원인을 분리할 수 없기 때문이다.

다음 short pilot은 Gate v1을 고정하고 roll-only action structure만 변경한다. roll-only가 독립 training seed에서 실패하면 폐기한다. roll-only가 재현되더라도 장시간 학습으로 바로 가지 않고, 별도 Gate 실험에서 offensive branch duration/range 조건 하나만 paired 변경한다.

Gate v1은 통과/실패로 단순 표현하지 않는다. 의미 계약은 검증됐지만 intervention duration은 과도하다. 상태는 `DIAGNOSTIC_COMPLETE / NOT_PROMOTED`다.

### artifact

- static: `artifacts/evaluations/offensive_residual_causal_rework/gate_selectivity_v1_20260817`
- local: `artifacts/evaluations/offensive_residual_causal_rework/gate_duration_t16_s3101_roll_20260817`
- manifest: `automation/manifests/residual_gate_selectivity_v1.json`

## Phase C1 — roll-only effective action short pilot

### 가설

기존 Tactical16 checkpoint를 inference에서 축별 마스킹했을 때 roll-only만 두 training seed의 contamination-free 평균 Damage 방향이 양수였다. 따라서 roll 축만 실제 학습 action으로 허용하면 pitch/yaw가 만든 폐루프 trajectory 교란을 제거하면서 Damage 개선 방향을 재현할 수 있다고 가정했다.

### 변경 변수와 고정 변수

변경 변수는 `residual_axis_mask=roll` 하나다. Tactical16 v1, Rear120 Gate v1, 기존 `aim_residual` reward, SAC 128x128, 20 iteration, residual scale 0.125, saturation-aware composition, action repeat 6, target/geometry distribution, throttle BT-only는 PR #10 pilot과 동일하게 고정했다.

training seed는 4101·4102이며 각각 20 iteration, sampled step 2,560, learner step 4,864를 완료했다. 두 run 모두 crash 0, action clipping 0, pitch/yaw applied correction 0이었다. 첫 `roll_v1` 실행은 Ray CSV logger의 Windows 접근 권한 오류로 iteration 1 직후 중단됐다. 이 partial artifact는 보존했고 동일 설정을 새 run ID `roll_v1r1`로 재실행했다.

### contamination-free paired 결과

평가는 Pure 0815 BT와 evaluation seed 2801~2806을 고정했다. 모든 controller에서 target crash가 발생한 crossing-left는 `TARGET_CRASH_CONTAMINATED`로 표시하고 아래 5-geometry 주 판정에서 제외했다.

| training seed | mean Damage Δ | median | min / max | positive pair | bootstrap 95% CI | First Damage Δ | LOS Δ | LOS-rate Δ | Cone Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4101 | -0.001260 | -0.001297 | -0.004857 / +0.002464 | 1/5 | [-0.003393, +0.000834] | 0.0000s | -0.002526° | +0.000769°/s | -0.0033s |
| 4102 | -0.000371 | +0.000212 | -0.004749 / +0.003890 | 3/5 | [-0.002863, +0.002092] | +0.0033s | -0.002094° | +0.000412°/s | -0.0067s |

per-geometry Damage Δ는 다음과 같다.

| geometry | seed 4101 | seed 4102 | contamination |
|---|---:|---:|---|
| lateral-left | -0.002356 | -0.001605 | clean |
| lateral-right | -0.001297 | +0.000396 | clean |
| crossing-left | +0.002476 | -0.002423 | `TARGET_CRASH_CONTAMINATED` |
| crossing-right | -0.004857 | -0.004749 | clean |
| vertical-high | +0.002464 | +0.003890 | clean |
| vertical-low | -0.000253 | +0.000212 | clean |

두 seed 모두 ownship crash 0, target crash 1, inference P95 0.547/0.568ms, Gate active ratio 0.9936/0.9942였다. applied correction 평균은 roll 0.003615/0.001458, pitch/yaw 0이었다. Gate-active 구간이 여전히 episode 대부분이며, roll-only로 축을 좁혀도 Pure BT trajectory를 장시간 perturbation하는 문제는 남았다.

### 판단

두 독립 training seed 모두 contamination-free aggregate Damage가 Pure 0815 BT보다 낮다. LOS 평균은 두 seed 모두 소폭 감소했지만 Damage와 Cone은 악화됐다. 이는 LOS 개선을 승격 근거로 쓰지 않는 원칙의 또 다른 반례다.

inference-time axis ablation의 양수 결과는 공동 action checkpoint에 대한 분포 밖 진단이었고, 실제 roll-only 재학습으로 재현되지 않았다. 이 가설은 `REVALIDATION_FAILED / NOT_PROMOTED`이며 scale sweep, 장시간 학습, 200초 전체전으로 확장하지 않는다.

### artifact

- training: `artifacts/models/0817_offensive_residual_causal_rework/rear120_t16_roll_s4101_roll_v1r1`, `rear120_t16_roll_s4102_roll_v1r1`
- evaluation: `artifacts/evaluations/offensive_residual_causal_rework/roll_only_short_s4101_20260817`, `roll_only_short_s4102_20260817`
- manifest: `automation/manifests/residual_roll_only_short_v1.json`
