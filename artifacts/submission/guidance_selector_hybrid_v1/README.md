# Guidance Selector Hybrid v1

이 bundle은 fallback ladder 4단계의 rule-distilled safe Guidance Selector다. Rear120+safety Gate 초기 구간에서만 최소 VP_EL_POS_SMALL을 선택하고, 나머지는 exact Pure BT다. Throttle은 항상 BT-only다.

BC 3개 seed 학습은 실행됐지만 실제 development 개입이 0이라 PPO gate를 통과하지 못했다. 이 모델은 learned performance improvement가 아니며 NOT_PROMOTED다. 200초 상한 36-run matrix의 모든 fight는 자연 terminal로 12.45~52.0초에 종료됐다. 따라서 200초 timeout을 실제로 달성했다는 주장을 하지 않는다.

Load smoke:

    python -c "from dogfight.ai.guidance_selector import NumpyMLPGuidanceSelector; NumpyMLPGuidanceSelector('artifacts/submission/guidance_selector_hybrid_v1/bundle')"

Config dry-run:

    python -c "from dogfight.submission import load_guidance_submission_config; load_guidance_submission_config('configs/submission/guidance_selector_hybrid_v1.json')"
