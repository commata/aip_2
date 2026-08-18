# Counterfactual Pulse 기준선 재현성

## 결론

`BASELINE_FROZEN`이다. 개발 geometry 6종과 사전 고정 seed `6201~6206`에서 Pure 0815와 ZERO residual을 각각 3회 실행했다. Pure BT는 geometry별 simulator trajectory와 Damage가 세 반복 모두 정확히 같았고, ZERO residual도 부호가 다른 `-0.0/0.0`을 수치적으로 정규화한 뒤 Pure와 정확히 같았다.

따라서 pulse 결과를 보기 전에 최소 의미 Damage 차이를 `+0.001`, geometry별 최대 허용 회귀를 `-0.003`으로 고정한다. 반복 변동은 Damage range `0`, MAD `0`, P95 절대 편차 `0`이다.

## 실행 계약

- scenario: `automation/scenarios/counterfactual_pulse_dev_suite_v1.json`
- geometry: lateral 좌/우, crossing 좌/우, vertical 상/하
- seed: `6201~6206`
- 각 geometry Pure 3회 + ZERO 3회, 총 36 episode
- Pure BT: 60 Hz raw 0815
- ZERO: Shot-Window v1, 최대 60 frame, cooldown 30 frame, condition-exit rearm
- residual scale: `0.125`, saturation-aware, throttle BT-only
- target: local scripted autopilot
- held-out `5301~5306`: 미개봉

## Damage 기준선

| geometry | Pure Damage | range | MAD | Pure=ZERO trajectory |
|---|---:|---:|---:|---|
| lateral_left | 1.005980 | 0 | 0 | PASS |
| lateral_right | 1.005233 | 0 | 0 | PASS |
| crossing_left | 1.002521 | 0 | 0 | PASS |
| crossing_right | 1.000998 | 0 | 0 | PASS |
| vertical_high | 1.000083 | 0 | 0 | PASS |
| vertical_low | 1.000898 | 0 | 0 | PASS |

Damage가 1을 약간 넘는 것은 simulator의 마지막 damage tick이 health 0 경계를 통과하기 때문이며, Hybrid-Pure paired 차이는 동일 정의로 계산한다.

## 재현 명령

```powershell
$env:PYTHONPATH='src'
python automation/evaluate_counterfactual_pulses.py --mode baseline --suite automation/scenarios/counterfactual_pulse_dev_suite_v1.json --output artifacts/evaluations/counterfactual_pulse/baseline_repeat_v1_20260819 --ownship-bt-dll <0815.dll> --target-backend autopilot --target-bt-dll AIP_BASE_target.dll --bt-rule-xml <0815.xml> --bt-rule-alias Rule_DCS_GDCC_0815.xml --baseline-repeats 3 --max-engage-time 30 --episode-step-limit 1800
```

원본 episode, stdout/stderr, simulator-rate telemetry와 aggregate는 `artifacts/evaluations/counterfactual_pulse/baseline_repeat_v1_20260819/`에 보존한다.
