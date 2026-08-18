# Counterfactual Pulse 선행결과 분석

## 결론

PPO iteration 15는 clean aggregate Damage 평균이 `+0.001188`이지만 clean positive pair가 `8/16 = 50%`에 불과해 후보가 아니었다. 그러나 세 training seed 모두 `vertical_high/vertical_low`에서 양수이고 lateral 계열은 대체로 음수였다. 이는 “PPO를 그대로 더 학습”하기보다 동일 Shot-Window entry에서 축과 부호를 대칭적으로 분리하는 counterfactual pulse가 필요한 근거다.

`DIAGNOSTIC_COMPLETE / NOT_PROMOTED`이며 held-out `5301~5306`은 열지 않았다.

## Clean paired 결과

| training seed | clean mean Damage Δ | positive/clean |
|---:|---:|---:|
| 5101 | +0.000667 | 2/6 |
| 5102 | +0.000930 | 3/5 |
| 5103 | +0.002072 | 3/5 |
| pooled | +0.001188 | 8/16 |

`crossing_left`의 seed 5102/5103 두 episode는 target crash를 포함해 `TARGET_CRASH_CONTAMINATED`로 분리했다. 두 차이 `-0.013224`, `-0.013738`은 promotion primary aggregate에 포함하지 않았다.

Clean geometry 패턴은 다음과 같다.

- vertical_high: `+0.002264 / +0.001999 / +0.002440`
- vertical_low: `+0.005160 / +0.003825 / +0.005075`
- lateral_left: `-0.000333 / -0.000193 / -0.001477`
- lateral_right: `-0.002601 / -0.003960 / -0.000081`
- crossing_right: `-0.000423 / +0.002980 / +0.004403`

정책 하나의 평균 action을 좋은 label로 간주할 수 없고 geometry에 따라 유효 축/부호가 달라질 가능성이 크다.

## 첫 Shot-Window 6 frame

세 PPO i15 checkpoint의 18개 simulator-rate telemetry에서 첫 ACTIVE entry와 이어지는 최대 6 frame을 추출했다. entry는 18/18 episode에 존재했다. 이 구간의 raw surface action 108개 표본에서 절대값은 다음과 같다.

- roll 중앙값: `0.295223`
- pitch 중앙값: `0.271379`
- yaw 중앙값: `0.120368`
- 모든 surface pooled 중앙값: `0.198680`
- pooled P75: `0.296422`

따라서 pulse 결과를 보기 전에 raw pulse magnitude를 pooled first-window 중앙값 `0.19867990911006927`로 고정한다. residual scale `0.125`를 적용하므로 포화 전 nominal surface correction은 약 `0.024835`이며, 실제 correction은 saturation-aware headroom으로 더 줄어들 수 있다.

## 근거와 한계

- 원본: `artifacts/evaluations/shot_window_research/stage1_ppo_v1_s510{1,2,3}_i000015_screening_20260818/`
- 재계산: `artifacts/analysis/counterfactual_pulse/ppo_i15_first_window_v1_20260819.json`
- 분석 SHA256: `D454C594B0B6AF22CEF4168F510CBF8BB364875EA6F3754C3EB31EA1F5D2BBE7`
- extraction은 관측된 raw telemetry만 사용했으며 존재하지 않는 observation/reward contribution은 추정하지 않았다.
- PPO action과 Damage의 상관은 인과 방향 label이 아니다. 이 때문에 다음 단계에서 같은 reset의 `ZERO, ±ROLL, ±PITCH, ±YAW`를 직접 비교한다.
