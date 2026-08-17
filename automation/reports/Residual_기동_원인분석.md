# Residual 기동 원인분석

## 연구 상태

- branch: `codex/offensive-residual-causal-rework`
- 분석 코드 기준 HEAD: `5d0bd4c8188723249a4d43be8b3064d01d5045cd`
- 선행 PR: #10
- Issue: #11
- 상태: `DIAGNOSTIC_COMPLETE / TRAINING_PENDING / NOT_PROMOTED`
- 외부 상태: `TARGET_BT_PENDING / SERVER_BLOCKED`

이 문서는 기존 Tactical16·frozen 10D checkpoint와 telemetry만 재사용한 Phase A 결과다. 새로운 정책을 학습하거나 checkpoint를 승격하지 않았다.

## 질문과 결론

질문은 절대값 약 `0.001~0.007` 수준의 작은 residual correction이 왜 stateful Pure BT trajectory를 바꿔 Damage를 감소시키는가였다.

frame 단위 비교 결과 다음 순서가 Tactical16 지정 10/10 trajectory와 frozen 10D 12/12 trajectory에서 모두 관측됐다.

1. Gate 진입 뒤 작은 residual이 최종 command를 즉시 바꾼다.
2. 3~5 simulator frame 안에 ownship 위치·자세가 Pure와 달라진다.
3. 달라진 state를 입력받은 BT raw command가 state divergence 다음 frame에 달라진다.
4. LOS가 같은 시점 또는 직후 달라지고, cone 진입/유지와 누적 Damage가 후행한다.

따라서 작은 correction 자체의 크기가 문제가 아니라, correction이 폐루프 입력 state를 바꾸고 60Hz stateful BT가 다음 command를 다시 계산하면서 차이가 누적되는 구조가 원인이다. Gate가 episode의 약 99% 동안 켜져 있었기 때문에 미세 correction이 짧은 조준 보정이 아니라 사실상 전체 공격 trajectory perturbation으로 작동했다.

## 계측 정의

`automation/analyze_residual_divergence.py`는 기존 JSONL을 수정하지 않고 공통 frame을 paired 비교한다.

- `first_command_divergence_frame`: Hybrid final command와 동일 frame Pure BT command의 첫 차이
- `first_state_divergence_frame`: ownship 위치·자세·속도의 첫 차이
- `first_bt_command_divergence_frame`: Hybrid trajectory에서 재계산된 BT raw command와 Pure BT command의 첫 차이
- `first_LOS_divergence_frame`: aim azimuth/elevation norm의 첫 차이
- `first_cone_divergence_frame`: 공식 damage cone boolean의 첫 차이
- `first_damage_divergence_frame`: target health에서 계산한 누적 Damage의 첫 차이

최종 Damage는 per-step `target_damage`를 재누적하지 않고 evaluator와 동일하게 `1 - target health`로 계산한다. episode 길이가 다른 경우 최종 Damage는 각 episode의 마지막 frame을 사용하고, state/LOS 발산은 공통 frame 구간만 비교한다.

## 대표 발산 frame

표의 순서는 `residual / final command / state / BT raw command / LOS / cone / Damage`다.

| checkpoint | geometry | 최초 frame 순서 | 최종 Damage Δ |
|---|---|---|---:|
| T16 3101 | lateral-left | `1 / 1 / 4 / 5 / 4 / 522 / 523` | -0.001445 |
| T16 3101 | crossing-left | `6 / 6 / 9 / 10 / 9 / 884 / 885` | -0.004943 |
| T16 3101 | vertical-low | `1 / 1 / 4 / 5 / 4 / 587 / 588` | -0.000463 |
| T16 3102 | lateral-left | `1 / 1 / 4 / 5 / 4 / 701 / 523` | -0.005452 |
| T16 3102 | crossing-left | `6 / 6 / 10 / 11 / 10 / 878 / 879` | +0.006963, target crash 오염 |
| T16 3102 | vertical-high | `1 / 1 / 4 / 5 / 7 / 479 / 480` | +0.004387 |

T16 10개 trajectory의 Gate active ratio 평균은 `0.993234`, Gate-active correction 절대평균 roll/pitch/yaw는 `0.002881 / 0.002620 / 0.002323`이었다. 10D 12개 trajectory에서도 Gate active ratio `0.994529`, correction `0.002651 / 0.001881 / 0.001636`으로 동일한 발산 순서가 재현됐다. 즉 이 현상은 Tactical16 feature 수 자체로 설명되지 않는다.

상세 CSV·PNG·JSON은 다음 고유 artifact에 있다.

- `artifacts/analysis/offensive_residual_causal_rework/phase_a_historical_20260817_r4`
- `artifacts/analysis/offensive_residual_causal_rework/phase_a_historical_10d_20260817`

## residual axis ablation

기존 정책의 checkpoint 출력은 그대로 두고 inference에서 선택하지 않은 surface를 0으로 마스킹했다. Pure, roll-only, pitch-only, yaw-only, pitch+yaw, roll+pitch+yaw를 비교했다. scale `0.125`, Tactical16, rear120 Gate, saturation-aware composition, action repeat 6, six-geometry와 evaluation seed 2801~2806은 고정했다.

crossing-left는 모든 controller에서 target crash가 기록되어 `TARGET_CRASH_CONTAMINATED`로 표시하고 아래 주 판정에서는 제외했다. 표는 contamination-free 5 geometry 결과다.

| training seed | axis | mean Damage Δ | median | min / max | positive pair | bootstrap 95% CI |
|---:|---|---:|---:|---:|---:|---:|
| 3101 | roll | +0.000761 | +0.000248 | -0.002301 / +0.004387 | 3/5 | [-0.001237, +0.002926] |
| 3102 | roll | +0.001876 | +0.002068 | -0.003977 / +0.006086 | 4/5 | [-0.001345, +0.004623] |
| 3101 | pitch | -0.000991 | -0.000010 | -0.004698 / +0.000207 | 2/5 | [-0.002875, +0.000095] |
| 3102 | pitch | -0.000719 | +0.000281 | -0.004771 / +0.001770 | 3/5 | [-0.002996, +0.001140] |
| 3101 | yaw | -0.001640 | -0.001859 | -0.004534 / +0.002369 | 1/5 | [-0.003848, +0.000586] |
| 3102 | yaw | -0.000910 | -0.000581 | -0.002399 / +0.000423 | 1/5 | [-0.001838, -0.000034] |
| 3101 | pitch+yaw | -0.001026 | -0.001664 | -0.005197 / +0.004786 | 1/5 | [-0.003687, +0.002206] |
| 3102 | pitch+yaw | -0.001164 | +0.000526 | -0.005954 / +0.003591 | 3/5 | [-0.004254, +0.001925] |
| 3101 | all | -0.001719 | -0.001445 | -0.004190 / -0.000248 | 0/5 | [-0.003056, -0.000573] |
| 3102 | all | -0.001005 | -0.000422 | -0.005452 / +0.004387 | 2/5 | [-0.003978, +0.001925] |

표본이 5개뿐이므로 roll-only CI도 0을 포함한다. 성능 확정 근거가 아니다. 다만 contamination을 제거한 평균 Damage 방향이 두 독립 training checkpoint에서 roll-only만 양수이고 나머지 action 조합은 모두 음수라는 사실은 다음 short pilot의 단일 가설을 고르기에 충분하다.

## 해석과 한계

- 기존 checkpoint는 roll/pitch/yaw 공동 action을 전제로 학습됐다. inference-time 마스크는 분포 밖 진단이며 최종 action space 결정이 아니다.
- roll-only 결과는 LOS가 반드시 개선되지 않아도 Damage가 개선됐다. 이 연구의 우선순위가 LOS가 아니라 Damage여야 한다는 기존 결론과 일치한다.
- yaw-only는 LOS-rate가 개선되는 경우에도 Damage가 감소했다. LOS 계열 shaping을 직접 성능 evidence로 쓸 수 없다.
- all-axis는 correction 자체가 작아도 장시간 Gate 활성 때문에 BT trajectory를 지속적으로 교란했다.
- target crash crossing-left의 양수 결과는 promotion evidence에서 제외한다.

## 다음 가설

가장 좁고 causal한 다음 변경 변수는 `action structure = roll-only`다. Gate, Tactical16 observation, reward, SAC/network, scale 0.125, target/geometry 분포는 고정하고 roll-only로 새 정책을 short training한다. 두 독립 training seed에서 contamination-free paired Damage 방향이 재현되지 않으면 즉시 폐기한다.

이 진단만으로는 `PROMOTE_CANDIDATE`가 아니다. 현재 판단은 `DIAGNOSTIC_COMPLETE / TRAINING_PENDING / NOT_PROMOTED`다.
