# Shot-Window Gate 설계 및 검증

## 출발점

PR #12에서 작은 residual이 final command를 바꾸고 3~5 frame 뒤 state와 다음 BT raw command를 갈라 전체 trajectory와 Damage를 바꾸는 causal chain을 확인했다. 기존 Gate v1은 local active ratio 0.9943, 평균 706 frame(11.77초), 최대 1,302 frame(21.70초)였으므로 residual amplitude가 아니라 개입 시간 계약을 우선 수정한다.

## Pure BT 사격 직전 분포

분석 원본은 기존 Pure 0815 BT six-geometry simulator-rate telemetry다. crossing-left 1건은 target crash contamination으로 분포에서 제외했고 clean 5건만 사용했다.

| 구간 | frame | 거리 중앙값 | own ATA 중앙값 | target ATA 중앙값 | closing 중앙값 |
|---|---:|---:|---:|---:|---:|
| first Damage -6~-3초 | 900 | 1,157.82m | 0.49° | 149.47° | 49.81m/s |
| first Damage -3~0초 | 899 | 991.97m | 0.21° | 163.15° | 54.54m/s |
| first Damage -1~0초 | 299 | 939.89m | 0.26° | 167.52° | 53.62m/s |
| first Damage 0~+0.25초 | 80 | 907.87m | 0.28° | 169.41° | 53.47m/s |

Pure BT의 first Damage는 first WEZ frame과 같았다. 공식 phase range +75m 진입은 Damage보다 평균 1.453초 빨라 0.5초 window가 사격 전에 끝났다. range margin만 +0/+15/+25/+35/+50/+75m로 비교했을 때 +25m는 clean 5건에서 0.417~0.567초, 평균 0.490초 선행했다. 따라서 v1은 +25m를 사용한다.

분석 artifact:

- `artifacts/analysis/shot_window_research/pure_bt_feature_distribution_v1_20260818/analysis.json`: +75m 조기 진입 반례
- `artifacts/analysis/shot_window_research/pure_bt_feature_distribution_v1b_range25_20260818/analysis.json`: 선택한 +25m 계약

## 상태 머신

`DISARMED → ARMED → ACTIVE → COOLDOWN`

- `DISARMED`: Pure BT exact path, RL inference 없음
- `ARMED`: 기존 rear120 AND (offensive OR pre-aim) AND NOT safety를 사격 준비 envelope로만 사용, RL inference 없음
- `ACTIVE`: 공식 phase half-angle +1.5°, phase range +25m, target ATA 150° 이상에서만 residual 허용
- `COOLDOWN`: timeout 또는 condition/safety exit 직후 RL inference 금지

ACTIVE exit hysteresis는 half-angle +2.5°, phase range +75m, target ATA 140°다. ACTIVE는 최대 30 frame(0.5초)이며 cooldown 30 frame(0.5초)과 shot condition 이탈을 모두 충족해야 재진입한다. time-only cooldown은 명시적 비교 가능한 대안으로 구현했지만 v1 기본값은 condition-exit 방식이다.

## 계측 계약

- `window_entry_count`, `window_exit_count`, `window_timeout_count`
- `window_condition_exit_count`, `window_safety_exit_count`
- `window_reentry_count`
- `active_duration_mean`, `active_duration_p95`, `active_duration_max`
- `cooldown_duration`
- 상태별 frame 수, RL inference/fallback/latency, throttle BT-only

## Static 검증

`automation/tests/test_shot_window_gate.py`가 deep rear/rear120 boundary/beam/front, WEZ outside/approaching/inside, LOS good/bad, high/low LOS-rate 관측, entry/timeout/exit/cooldown/reentry, boundary oscillation, low altitude/low speed/no authority veto, Gate OFF exact BT, inference skip, fallback, throttle BT-only를 검증한다.

현재 targeted 결과는 `8 passed, 12 subtests passed`이며 milestone 전체 검증은 automation `137 passed, 26 subtests passed`, core tests `9 passed`, compileall 및 `git diff --check` 통과다.

training-environment smoke는 `artifacts/smoke/shot_window_v1_s5101_r2_20260818/result.json`에 보존했다. 59 step 동안 entry/exit/timeout 각 1회, ACTIVE 30 frame(0.5초), RL correction 30회였고 Gate OFF exact BT 및 throttle BT-only가 모두 참이었다. episode가 cooldown 종료 3 frame 전에 끝나 관측 cooldown은 0.45초였지만 정적 경계 테스트에서 30 frame 계약을 별도로 검증했다.

## 현재 판단

`DIAGNOSTIC_COMPLETE`는 기존 checkpoint inference diagnostic까지 통과한 뒤 사용한다. 현재는 상태 머신과 static contract가 구현된 `DIAGNOSTIC_PENDING`이다.
