# Counterfactual Pulse 데이터셋 설계

## 상태

`SKIPPED / DATASET_NOT_CREATED`.

설계상 동일 reset의 `ZERO, ±ROLL, ±PITCH, ±YAW` 중 사전 고정 Damage 기준을 반복적으로 넘는 action만 Tactical16 state의 label로 사용하고, 유효 action이 없으면 ZERO를 label로 두려 했다. training seed는 `6101~6103`, development는 `6201~6206`, held-out은 `5301~5306`으로 분리했다.

그러나 Phase 2에서 의미 있는 최선 pulse가 `1/6` geometry에만 나타났고, 방향 반복성은 최대 1회, clean pooled 평균은 음수였다. 이 결과로 dataset을 만들면 `vertical_high`의 특정 geometry와 simulator branch를 외우는 label selection이 된다. 따라서 sample, schema artifact, split 파일을 만들지 않았다.

held-out `5301~5306`은 미개봉이다.
