# State-Conditioned Counterfactual Hybrid v3 Oracle 분석

## 결론

- 판정: `ORACLE_FEASIBLE`
- canonical unique state: 510
- observed nondefault state-action pair: 2406
- State Oracle ΔDamage mean/median: 0.008497982 / 0.000023309
- Oracle positive/meaningful-positive ratio: 60.59% / 42.94%
- Oracle positive ratio (epsilon 1e-6): 56.67%
- Oracle bootstrap mean 95% CI: [0.006229225, 0.011055059]
- default-optimal state ratio: 39.41%

## Best Static 비교

- 30-state minimum candidate: `VP_AZ_POS_SMALL__m0.50__d36` (coverage 7.06%)
- majority-coverage candidate: `VP_EL_POS_SMALL__m0.25__d36`
- majority coverage: 480/510 (94.12%)
- majority-static ΔDamage mean/median/positive: 0.006168787 / 0.000000000 / 46.46%
- 같은 state에서 Oracle mean gap: 0.001536445

## 해석 주의

이 Oracle은 state별로 관측된 action 중 최댓값을 고른 in-sample upper bound다. 60개 action을 관측한 state와 3개만 관측한 state가 섞여 있으므로 action 수가 많은 state에서 selection optimism이 커질 수 있다. 따라서 `ORACLE_FEASIBLE`은 학습 가능성을 검증할 다음 단계 진입 조건이지 Promotion 근거가 아니다.

결측 action은 0으로 채우지 않았고 sparse matrix에서 결측으로 유지했다.
