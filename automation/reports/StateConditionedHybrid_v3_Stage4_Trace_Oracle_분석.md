# State-Conditioned Counterfactual Hybrid v3 Oracle 분석

## 결론

- 판정: `ORACLE_UNDERSAMPLED`
- canonical unique state: 678
- observed nondefault state-action pair: 3078
- State Oracle ΔDamage mean/median: 0.006399330 / 0.000000000
- Oracle positive/meaningful-positive ratio: 47.05% / 32.45%
- Oracle positive ratio (epsilon 1e-6): 44.10%
- Oracle bootstrap mean 95% CI: [0.004686336, 0.008347247]
- default-optimal state ratio: 52.95%

## Best Static 비교

- 30-state minimum candidate: `VP_AZ_POS_SMALL__m0.50__d36` (coverage 5.31%)
- majority-coverage candidate: `VP_EL_POS_SMALL__m0.25__d36`
- majority coverage: 648/678 (95.58%)
- majority-static ΔDamage mean/median/positive: 0.004576243 / 0.000000000 / 35.19%
- 같은 state에서 Oracle mean gap: 0.001138705

## 해석 주의

이 Oracle은 state별로 관측된 action 중 최댓값을 고른 in-sample upper bound다. 60개 action을 관측한 state와 3개만 관측한 state가 섞여 있으므로 action 수가 많은 state에서 selection optimism이 커질 수 있다. 따라서 `ORACLE_FEASIBLE`은 학습 가능성을 검증할 다음 단계 진입 조건이지 Promotion 근거가 아니다.

결측 action은 0으로 채우지 않았고 sparse matrix에서 결측으로 유지했다.
