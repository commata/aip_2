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
