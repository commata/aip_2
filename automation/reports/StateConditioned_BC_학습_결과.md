# 상태 조건부 BC 학습 결과

## 상태

`BC_NOT_RUN`.

Counterfactual Stage A가 `COUNTERFACTUAL_SIGNAL_INSUFFICIENT / DATASET_NOT_CREATED`로 종료되어 BC 입력 sample이 0이다. geometry-specific vertical_high 결과만 label로 채택하거나 threshold를 낮추지 않았다.

- checkpoint: N/A
- model seed: N/A
- epoch/iteration: 0
- environment step: 0
- classification/calibration: N/A
- paired development Damage: N/A
- inference latency: N/A

BC failure corrective branch를 소비한 것이 아니라, BC 시작 전 필수 dataset gate에서 종료한 것이다.
