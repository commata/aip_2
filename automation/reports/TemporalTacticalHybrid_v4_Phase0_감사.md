# Temporal Tactical-Mode Hybrid v4 Phase 0 감사

## 결론

Champion BT 내부 pursuit mode를 Python에서 직접 선택할 수 있는 공개 계약은 확인되지 않았다. 따라서 v4는 Champion DLL/XML을 수정하지 않고, 실제 서버 입력으로 재현 가능한 deterministic VP generator를 Python에 구현하는 경로 B를 사용한다. 모든 nondefault mode는 same-frame Pure BT를 먼저 실행한 뒤 roll/pitch/yaw만 보정하며 throttle은 Pure BT 값을 byte-for-byte 유지한다.

## 기준선

- 시작 main: `ae4b2e0c1ea43d9f1b74e783a65293b5490ffcc4`
- Issue: <https://github.com/commata/aip_2/issues/28>
- Pure BT DLL: `4C93B4C6719CB0423388D5FC721D356020A3A36CD5AD2C56B5C3CA795BFE18C9`
- Pure BT XML: `D84C27B0B8BA22E1649AF2375BE0B83C762BC62EB5047BB590539B374F8271EE`
- v3 model: `7B97E8DF6CEA3250AE873FAC00C77C1B4AE923B12D0AC596C0DC25F1F36550F2`
- v3 dataset: `4DFBF311AA680AA8F8054F9ED969A0729CAB89B1A1EEEFF0D7D8492F8614D6A3`

실제 파일에서 SHA256을 다시 계산했고 모두 일치했다.

## Champion capability inventory

Champion XML은 target/geometry service 뒤에 단일 `Task_Pursue`를 실행한다. XML에는 `Task_LagPursuit`, `Task_LeadPursuit`, mode selector 또는 runtime parameter port가 없다.

Champion DLL에서 현재 wrapper가 사용하는 export는 다음과 같다.

- `CreateBehaviorTree`
- `ChangeData`
- `Step`
- `GetVP`
- `Reset`
- `RemoveBT`

`PurePursuitRun`, `LagPursuitRun`, `LeadPursuitRun`, `SetPursuitMode`, `SetMode`, `SetVP`는 export probe에서 확인되지 않았다. 저장소의 다른 상대 XML에 Lag/Lead/Predictive pursuit node가 존재하지만, 그것이 Champion DLL의 runtime-switch API를 의미하지는 않는다.

## 확인된 연결 경로

`BTActionProvider`는 local FDM에서 `Step`과 `GetVP`, Unreal packet에서 `StepWithPlaneData`와 `GetVPWithPlaneData`를 호출한다. `GuidanceSelectorActionProvider`는 same-frame BT action/VP를 얻은 뒤 `vp_error_pd_v2` controller로 surface correction을 합성하고 마지막에 throttle을 BT 값으로 복원한다. `run_local_dogfight.py`와 `run_unreal_inference.py`가 이 provider contract를 공유한다.

## v4 action-space contract

T1은 `BT_DEFAULT`, server-visible state 기반 Pure pursuit, velocity-extrapolated Lead pursuit, target flight-path 뒤의 Lag pursuit로 구성한다. T2는 LOS-rate damping, crossing lead, cone capture처럼 이름과 계산 효과가 직접 일치하는 deterministic VP mode다. 초기 hold duration은 결과를 보기 전에 30/60/120 frame으로 동결한다. 240 frame은 T1/T2에서 delayed effect가 관측될 때만 별도 level로 연다.

각 mode 입력은 ownship/target location, rotation, speed와 same-frame BT action/VP뿐이다. 출력은 desired VP와 roll/pitch/yaw correction이며 throttle 출력은 없다. 비정상 입력, safety veto, timeout, 예외, OOD에서는 exact `BT_DEFAULT`로 복귀한다.

## Counterfactual 판정

v3의 reconstructed 7D restart는 body velocity, angular rate, surface/engine state를 보존하지 않는다. v4에서는 이를 causal truth로 사용하지 않는다. 동일 initial scenario/seed를 frame N 직전까지 Pure BT로 재생하는 prefix replay만 causal label 생성에 사용한다. 먼저 Pure A/B와 `BT_DEFAULT` override parity를 검증하고, reconstructed restart는 원 continuation과의 fidelity audit 결과만 기록한다.

실제 `vertical_high`, seed 71001, decision frame 60 감사에서 Pure A/B와 `BT_DEFAULT` override는 180 frame 전체 position/attitude/speed/action/VP/Damage가 exact parity였다. Pure repeat Damage range는 0이므로 초기 meaningful epsilon을 `1e-9`, large regression threshold를 `1e-6`으로 동결했다.

같은 decision state를 7D position/attitude/speed로 다시 시작한 trajectory는 원 continuation과 59 aligned frame 안에 ownship position 49.4219m, attitude 11.529°, speed 40.435m/s, BT action 1.22506, BT VP 50.6757m까지 벌어졌다. 따라서 상태는 `RESTART_STATE_CAUSAL_INVALID`이며 reconstructed restart를 v4 causal label 생성에 사용하지 않는다.

## 현재 상태

- `IN_PROGRESS`
- `SERVER_BLOCKED`
- submission freeze 금지
- held-out 봉인 유지
- PPO 금지
- Pure BT Champion fallback 유지
