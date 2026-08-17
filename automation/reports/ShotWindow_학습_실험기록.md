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
2. v1 0.5초/condition-exit short pilot seed 5101/5102
3. paired development screening seed 5201~5206
4. 통과 시에만 held-out seed 5301~5306

현재 상태: `SHORT_TRAINING` 전, `NOT_PROMOTED`.

