# Guidance Selector 최종 후보 및 제출 fallback

## 최종 상태

- artifact: `EXPERIMENTAL_SAFE_HYBRID`
- operational: `SUBMISSION_READY_HYBRID_CANDIDATE`
- performance: `NOT_PROMOTED`
- PPO: `PPO_NOT_RUN_FAILED_DEV_GATE`
- server: `SERVER_BLOCKED`

BC 3개 모델은 모두 실제 비행 개입 0으로 탈락했다. 최종 artifact는 Rear120+safety Gate의 초기 36/90 frame에서 최소 `VP_EL_POS_SMALL`만 허용하는 rule-distilled categorical MLP다. learned improvement로 주장하지 않는다.

clean Damage Δ mean `-0.0030799766`, median `-0.0033841133`, positive `0/4`이므로 Pure BT보다 우월하지 않다. 실패·저신뢰·Gate OFF·BT_DEFAULT에서는 exact Pure BT로 돌아가며 throttle은 항상 BT-only다.

Pure fallback SHA256:

- DLL `4C93B4C6719CB0423388D5FC721D356020A3A36CD5AD2C56B5C3CA795BFE18C9`
- XML `D84C27B0B8BA22E1649AF2375BE0B83C762BC62EB5047BB590539B374F8271EE`

## 최종 검증

- 정규 회귀: `191 passed, 26 subtests passed`
- Guidance 집중 검증: `29 passed`
- compileall: 통과
- config fail-fast parse / model load-only inference: 통과
- artifact/evidence checksum: 통과
- JSON parse / CSV nonempty: 통과 (`episode_records` 36행, `paired_200s_results` 24행)
- 변경 파일 1 MiB 초과: 없음
- tracked-file credential pattern scan: 검출 없음

추가로 정규 범위 밖의 `test_ias.py`는 인자 fixture가 없는 수동 스크립트라 pytest collection error가 났고, web log viewer test는 Git에 없는 `MyTrainEnv/logs` fixture 때문에 실패했다. 두 파일은 이 branch diff에 포함되지 않는다.
