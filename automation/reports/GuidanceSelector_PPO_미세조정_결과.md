# Guidance Selector PPO 미세조정 결과

## 결론

`PPO_NOT_RUN_FAILED_DEV_GATE`.

BC 3개 seed × confidence 4개 후보 모두 동결된 6개 development 비행에서 비기본 Guidance 개입이 0이었고, clean mean Damage Δ와 median도 0이었다. 사전 조건인 mean/median 양수, positive ratio 2/3 이상, 실제 비기본 개입을 만족한 후보가 없으므로 PPO를 실행하지 않았다.

## 실행량

- PPO iteration: 0
- PPO environment step: 0
- PPO episode: 0
- PPO seed: 사용하지 않음
- checkpoint: 생성하지 않음

이는 학습 실패를 숨기기 위한 생략이 아니라, 약한 BC에서 PPO를 시작해 근거 없는 탐색·회귀를 확대하지 않도록 동결한 사전 게이트의 적용 결과다.

## Fallback

fallback ladder 4단계의 `Rule-distilled safe Guidance Selector Hybrid`를 생성했다.

- rule: Rear120+safety Gate가 활성화된 초기 36/90 frame에서만 `VP_EL_POS_SMALL`, 그 외 `BT_DEFAULT`
- model: 실제 load 가능한 `45-64-64-9` NumPy MLP
- rule grid 일치: 14/14
- model SHA256: `FAD2BCB268D4669BD65DB89CE8F36832B1E8C4B646163897E9BBC2A6372FBB6C`
- development 실제 개입: 1,842 frame / 6 runs
- development clean Damage Δ: mean `-0.0009252800`, median `0`, min `-0.0052003859`
- crash/target crash/throttle violation: 모두 0
- inference max: `0.1313001 ms`

이 후보는 learned performance improvement가 아니다. 상태는 `EXPERIMENTAL_SAFE_HYBRID / NOT_PROMOTED`이며, 200초 operational validation 대상으로만 유지한다.
