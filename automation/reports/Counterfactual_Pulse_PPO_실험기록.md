# Counterfactual Pulse PPO 실험기록

## 상태

`SKIPPED / BC_GATE_NOT_REACHED`.

Conservative PPO는 BC-only가 Pure, ZERO, 과거 PPO와의 development 비교에서 사전 기준을 통과한 뒤에만 실행하는 단계였다. BC 자체가 실행되지 않았으므로 PPO iteration `5/10/15/20`, training seed `6101/6102/(conflict 6103)`, scale sweep `0.10/0.125/0.15`, 장시간 학습을 모두 실행하지 않았다.

총 신규 학습량은 iteration `0`, environment step `0`, 생성 checkpoint `0`이다.
