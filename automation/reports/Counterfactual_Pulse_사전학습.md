# Counterfactual Pulse 사전학습

## 상태

`SKIPPED / NO_VALID_COUNTERFACTUAL_DATASET`.

BC-only 사전학습은 dataset phase gate를 통과한 경우에만 training seed `6101~6103`으로 수행하도록 고정했다. Counterfactual signal이 부족해 dataset을 만들지 않았으므로 모델, checkpoint, reward curve, BC evaluation은 존재하지 않는다. 기존 PPO checkpoint를 좋은 geometry 하나에 맞춰 재사용하지 않았다.
