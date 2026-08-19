# State-Conditioned Hybrid v3 Positive/Negative 상태 분리

## 결론

- current 42D 판정: `CURRENT_42D_SUFFICIENT_FOR_MODEL_PILOT`
- target-high/AZ+ rows/state groups: 90 / 30
- positive ratio (>1e-6): 32.22%
- group-CV logistic AUC/AP: 0.8451 / 0.8114
- group-CV decision stump/KNN AUC: 0.6670 / 0.7725

## 상위 단변량 분리

- `any_surface_saturation`: Cohen d -1.1882, Damage corr -0.1311
- `target_ata_norm`: Cohen d +0.7155, Damage corr +0.0957
- `target_yaw_norm`: Cohen d -0.6497, Damage corr -0.1450
- `los_elevation_rate_norm`: Cohen d -0.6037, Damage corr -0.2036
- `ownship_yaw_norm`: Cohen d -0.4598, Damage corr -0.0692
- `ownship_roll_norm`: Cohen d -0.4569, Damage corr -0.0688
- `ownship_pitch_norm`: Cohen d -0.4480, Damage corr -0.0674
- `closing_rate_norm`: Cohen d -0.4080, Damage corr -0.2152
- `bt_vp_local_azimuth_norm`: Cohen d +0.2766, Damage corr +0.0800
- `signed_aim_azimuth_norm`: Cohen d +0.2755, Damage corr +0.0783

## 보수적 threshold

- p>=0.50: coverage 24.44%, precision 81.82%, Damage mean +0.016023552, large regression 9.09%
- p>=0.60: coverage 16.67%, precision 93.33%, Damage mean +0.017241206, large regression 6.67%
- p>=0.70: coverage 12.22%, precision 100.00%, Damage mean +0.024333137, large regression 0.00%
- p>=0.80: coverage 10.00%, precision 100.00%, Damage mean +0.028257916, large regression 0.00%
- p>=0.90: coverage 8.89%, precision 100.00%, Damage mean +0.024777032, large regression 0.00%

30개 state는 pilot 크기이며 같은 state의 세 magnitude는 반드시 같은 CV fold에 묶었다. row random split 결과는 사용하지 않았다.
