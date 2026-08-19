# 상태 조건부 최종 후보 및 제출 Fallback

## 결론

최종 연구 상태는 `COUNTERFACTUAL_SIGNAL_INSUFFICIENT / DATASET_NOT_CREATED / NOT_PROMOTED / TARGET_BT_PENDING / SERVER_BLOCKED`다. Hybrid checkpoint/bundle/config는 생성하지 않았다.

## 후보 계층

1. PROMOTED_LOCAL Hybrid: 없음
2. BC-only Hybrid: 없음
3. 안전 fallback: Pure 0815 BT

Pure fallback은 DLL `4C93B4C6719CB0423388D5FC721D356020A3A36CD5AD2C56B5C3CA795BFE18C9`, XML `D84C27B0B8BA22E1649AF2375BE0B83C762BC62EB5047BB590539B374F8271EE`, raw native BT, throttle BT-only다. 이번 branch에서 Pure BT DLL/XML/JSBSim controller 성능을 수정하지 않았다.

## 실행 총량

- counterfactual episode: 84
- Pure BT preflight smoke: 1
- training iteration/environment step/episode: 0/0/0
- dataset sample: 0 (compact rejected pair evidence 72행)
- BC/PPO/scale/long/held-out/proxy/200초: 모두 미실행

제출 시점까지 새 Hybrid 근거가 추가되지 않으면 검증되지 않은 RL 대신 이 Pure BT fallback을 사용한다. 최종 Target과 실제 server 근거가 없으므로 `FINAL_CONFIRMED`는 사용하지 않는다.

## 종료 검증

- automation: `153 passed, 26 subtests passed`
- core: `9 passed`
- compileall / manifest 30개 parse / `git diff --check`: PASS
- compact evidence SHA256: 5/5 일치
- secret scan / tracked 10MB 초과 파일: 0 / 0
- held-out seed 실행 record: 0 (Ray 로그 파일명의 우연한 64진수 부분 문자열은 제외)
- state-conditioned checkpoint/bundle: `N/A_NOT_CREATED`
