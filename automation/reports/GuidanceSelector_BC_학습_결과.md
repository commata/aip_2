# Guidance Selector BC 학습 결과

## 결론

counterfactual 100개 state를 이용한 categorical BC를 실제로 3개 초기화 seed에서 학습했다. 세 모델 모두 load 가능한 `45-64-64-9` NumPy MLP bundle로 생성됐지만, 동결된 30초 development 비행에서는 모두 `BT_DEFAULT`만 예측해 비기본 개입이 0이었다. 따라서 BC 성능을 승격하지 않고 fallback ladder의 rule-distilled 후보로 이동했다.

## 동결 조건

- dataset SHA256: `291E93BCA7D8F329C3E9A7D1B48A2BB1E289535940C6720107895B583A52D5C0`
- raw sample: 100
- split: 고정 permutation seed 8600, train 80 / validation 20
- train augmentation: lateral/vertical/both mirror, 320 sample
- model seed: 8701, 8702, 8703
- optimizer: Adam, learning rate `1e-3`, weight decay `1e-4`
- loss: class-weighted cross entropy
- early stopping patience: 40
- 총 실제 epoch: 231

## 결과

| seed | epoch(best) | train accuracy | validation accuracy | validation macro F1 | nondefault precision | model SHA256 |
|---:|---:|---:|---:|---:|---:|---|
| 8701 | 78 (38) | 0.85 | 0.70 | 0.20588 | 측정 불가 | `E7A4E6D368BAB7C84F840A878CCD1BCCE0E2D05D00B86F615256F77A4856E5EC` |
| 8702 | 77 (37) | 0.85 | 0.70 | 0.20588 | 측정 불가 | `E4EC64E494CD82372CEFEB6A98F00FBDABF4C9CF0AE57795CC5A7FA771EE4B90` |
| 8703 | 76 (36) | 0.85 | 0.70 | 0.20588 | 측정 불가 | `F31FC4FB68BA692AEB072EC266E136D2E4F663869A69F1356A42FAE8C8CECED7` |

검증 loss 기준 선택 모델은 seed 8702였으나, 세 모델 모두 train/validation argmax가 전부 `BT_DEFAULT`였다. 불균형 dataset에서 accuracy 0.70은 nondefault Guidance 학습 성공을 뜻하지 않는다.

## 실제 development 결과

상대 `autopilot`, `BT0815`, `AIP2` × left/right 6개 상태에서 각 모델과 confidence `0.55/0.60/0.65/0.70`을 평가했다. 12개 후보 모두 다음과 같았다.

- non-BT_DEFAULT action: 0 frame
- Damage Δ mean/median: 0 / 0
- crash, target crash, throttle violation: 0
- PPO development gate 통과: 0개

결론: `BC_DEVELOPMENT_GATE_FAILED`, `NOT_PROMOTED`.

## Artifact

- 학습 root: `artifacts/models/guidance_selector_bc_v1`
- training summary: `artifacts/models/guidance_selector_bc_v1/training_summary.json`
- development aggregate: `artifacts/evaluations/guidance_selector/development_v1_20260819/aggregate.json`
