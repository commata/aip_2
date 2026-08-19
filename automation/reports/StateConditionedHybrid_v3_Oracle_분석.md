# State-Conditioned Counterfactual Hybrid v3 Oracle 분석

## 결론

- 판정: `ORACLE_UNDERSAMPLED`
- canonical unique state: 186
- observed nondefault state-action pair: 1110
- State Oracle ΔDamage mean/median: 0.012266727 / 0.000001464
- Oracle positive/meaningful-positive ratio: 56.45% / 35.48%
- Oracle positive ratio (epsilon 1e-6): 50.00%
- Oracle bootstrap mean 95% CI: [0.007115028, 0.018474955]
- default-optimal state ratio: 43.55%

## Best Static 비교

- candidate: `VP_AZ_POS_SMALL__m0.50__d36`
- coverage: 36/186 (19.35%)
- ΔDamage mean/median/positive: 0.018601428 / 0.000000000 / 44.44%
- 같은 state에서 Oracle mean gap: 0.005093138

## 해석 주의

이 Oracle은 state별로 관측된 action 중 최댓값을 고른 in-sample upper bound다. 60개 action을 관측한 state와 3개만 관측한 state가 섞여 있으므로 action 수가 많은 state에서 selection optimism이 커질 수 있다. 따라서 `ORACLE_FEASIBLE`은 학습 가능성을 검증할 다음 단계 진입 조건이지 Promotion 근거가 아니다.

결측 action은 0으로 채우지 않았고 sparse matrix에서 결측으로 유지했다.
