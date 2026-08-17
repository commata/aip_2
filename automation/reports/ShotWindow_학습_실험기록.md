# Shot-Window 학습 실험 기록

## 고정 조건

- algorithm: SAC, MLP 128×128
- observation: Tactical16 v1
- residual scale: 0.125
- composition: saturation-aware
- action: roll+pitch+yaw residual, throttle BT-only
- RL cadence: 6 frame
- training seeds: 5101, 5102; 충돌 시 5103
- short budget: seed당 20 iteration
- primary metric: Hybrid Damage - Pure 0815 BT Damage

## 실험 대기열

1. 기존 checkpoint + Shot-Window inference diagnostic
2. diagnostic에서 선택한 v1 1.0초/condition-exit short pilot seed 5101/5102
3. paired development screening seed 5201~5206
4. 통과 시에만 held-out seed 5301~5306

## 기존 checkpoint 진단 결론

s3101/s3102에서 1.0초 window의 pooled clean mean Damage Δ는 +0.000300, positive 8/12였다. 이는 Gate 구조 선택 근거일 뿐 새 policy의 승격 근거가 아니다. 0.25/0.50/0.75/1.50초 후보는 각각 너무 짧거나 pooled 재현성/추가 이득이 부족해 1차 학습에서 제외했다.

현재 상태: `DIAGNOSTIC_COMPLETE`, short pilot 준비, `NOT_PROMOTED`.
