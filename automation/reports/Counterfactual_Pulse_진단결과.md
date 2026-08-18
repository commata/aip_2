# Counterfactual Pulse 진단결과

## 판정

`COUNTERFACTUAL_SIGNAL_INSUFFICIENT / NOT_PROMOTED`다. 첫 Shot-Window에서 6 frame 동안 한 축에만 대칭 pulse를 주는 인과 실험은 runtime 계약을 지켰지만, 여러 geometry에서 반복되는 Damage 개선 방향을 찾지 못했다.

따라서 결과를 label로 바꾸지 않았고 dataset, BC, conservative PPO, scale sweep, 장시간 학습, held-out, 200초를 실행하지 않았다.

## 사전 고정 조건

- development geometry/seed: 6종, `6201~6206`
- controllers: Pure, ZERO, `±ROLL`, `±PITCH`, `±YAW`
- raw magnitude: `0.19867990911006927` (PPO i15 first-window raw action 절대값 pooled 중앙값)
- scale: `0.125`; nominal correction `0.024835`
- pulse: 첫 ACTIVE window의 첫 6 simulator frame만
- ACTIVE 최대 60 frame, cooldown 30 frame, condition-exit rearm
- saturation-aware, throttle BT-only, inference 0회
- 의미 기준: Damage Δ `≥ +0.001`
- 회귀 기준: geometry Damage Δ `< -0.003`

## 결과

- 총 48 episode: Pure 6, ZERO 6, nonzero pulse 36
- clean pulse pair: 33; `TARGET_CRASH_CONTAMINATED`: 3
- clean 평균 Damage Δ: `-0.000641`
- clean 중앙값: `0`
- clean positive: `15/33 = 45.45%`
- 의미 있는 양수 pulse: `4/33`, 모두 vertical_high 내부
- geometry별 최선이 의미 기준을 넘은 수: `1/6`
- geometry별 최선 Damage Δ 중앙값: `+0.000374`
- clean 대규모 회귀: 4개
- ownship crash/invalid/nonfinite/process error: 모두 0

| geometry | 최선 pulse | Damage Δ | First Damage Δ | LOS Δ | Cone Δ |
|---|---|---:|---:|---:|---:|
| lateral_left | yaw_pos | +0.000702 | 0.0000s | +0.000613° | 0.0000s |
| lateral_right | yaw_neg | +0.000091 | 0.0000s | -0.000247° | 0.0000s |
| crossing_left | yaw_neg | +0.000340 | 0.0000s | +0.000924° | 0.0000s |
| crossing_right | yaw_pos | +0.000407 | 0.0000s | -0.000487° | 0.0000s |
| vertical_high | pitch_pos | +0.002040 | 0.0000s | -0.004587° | +0.0167s |
| vertical_low | roll_pos | +0.000021 | 0.0000s | -0.000398° | 0.0000s |

가장 큰 clean 회귀는 `lateral_left / roll_neg`의 Damage `-0.005819`, Cone `-0.0167초`다. `lateral_left / roll_pos`, `pitch_neg`과 `lateral_right / pitch_pos`도 `-0.003` 회귀 한계를 넘었다.

## Contamination

`crossing_left`의 roll_pos, pitch_neg, yaw_pos 세 episode는 target altitude 종료를 포함해 `TARGET_CRASH_CONTAMINATED`로 표시하고 primary metric에서 제외했다. artifact는 삭제하지 않았다. 이 세 결과를 포함해도 성공 판정은 되지 않는다.

## Runtime 계약

- ZERO trajectory exact equality: PASS
- pulse 적용: 각 hybrid episode 첫 window 6 frame
- 전체 counterfactual 적용 frame(ZERO 포함): 252
- RL inference: 0회
- throttle BT-only 위반: 0 frame
- invalid/nonfinite final action: 0 frame
- ownship crash 증가: 0

## Artifact

- aggregate: `artifacts/evaluations/counterfactual_pulse/symmetric_first_window_v1_20260819/evaluation.json`
- SHA256: `EE5188A519F53D731B69C166912345613DDA521F46BA506325C1812D95D341A2`
- 같은 디렉터리에 48개 result, simulator-rate telemetry, stdout/stderr를 보존했다.

단일 `vertical_high` geometry의 좋은 결과를 policy label로 채택하면 좋은 geometry 하나를 고르는 것과 같으므로 독립 개선 근거가 아니다.
