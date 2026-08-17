# Hybrid 후보 독립 재검증

held-out seed 5301~5306과 mirror-holdout geometry는 결과 확인 전에 `automation/manifests/shot_window_research_v1.json`에 고정했다. development screening에서 `SHORT_PROMOTE_CANDIDATE`가 나오기 전에는 이 set을 열지 않는다.

SAC broad, Stage-1 random, Stage-1 paired warm-start는 두 training seed에서 모두 음수였다. PPO iteration 10/15는 세 seed 평균이 양수였지만 pooled positive가 각각 7/15, 8/16에 그쳤고 target-crash 오염과 geometry 회귀가 남았다. iteration 20은 seed 5101이 음수였다. 따라서 `SHORT_PROMOTE_CANDIDATE`가 성립하지 않았다.

사전에 고정한 held-out seed 5301~5306은 후보 선택 이후의 독립 재검증용이므로 열지 않았다. 실패한 development 모델로 held-out을 반복 조회해 checkpoint를 선택하지 않는다. scale sweep, 장시간 학습, proxy target 다양화, 최종 bundle도 같은 이유로 실행하지 않았다.

현재 상태: 후보 없음, `REVALIDATION_FAILED / NOT_PROMOTED / TARGET_BT_PENDING / SERVER_BLOCKED`.
