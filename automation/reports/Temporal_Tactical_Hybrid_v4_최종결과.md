# Temporal Tactical-Mode Hybrid v4 최종 결과

## 결론

Hybrid v4는 **승격하지 않는다**. T1 Tactical action space의 Oracle은 충분히 강했지만, 실제 server-safe temporal observation으로 positive opportunity와 large regression을 group-OOF에서 안정적으로 분리하지 못했다. 제출 기본값은 동결된 Pure BT Champion을 유지한다.

최종 상태:

- `NOT_PROMOTED`
- `PURE_BT_FALLBACK_RECOMMENDED`
- `TACTICAL_ORACLE_FEASIBLE`
- `HARD_BLOCKED` (D: server-safe temporal observation generalization 반복 실패)
- `SERVER_BLOCKED`

## 기준과 식별자

| 항목 | 값 |
|---|---|
| 시작 main SHA | `ae4b2e0c1ea43d9f1b74e783a65293b5490ffcc4` |
| Issue | `https://github.com/commata/aip_2/issues/28` |
| branch | `codex/temporal-tactical-mode-hybrid-v4` |
| Pure DLL SHA256 | `4C93B4C6719CB0423388D5FC721D356020A3A36CD5AD2C56B5C3CA795BFE18C9` |
| Pure XML SHA256 | `D84C27B0B8BA22E1649AF2375BE0B83C762BC62EB5047BB590539B374F8271EE` |
| v3 model SHA256 | `7B97E8DF6CEA3250AE873FAC00C77C1B4AE923B12D0AC596C0DC25F1F36550F2` |
| v3 dataset SHA256 | `4DFBF311AA680AA8F8054F9ED969A0729CAB89B1A1EEEFF0D7D8492F8614D6A3` |

## Counterfactual fidelity

- reconstructed-state restart는 원 trajectory continuation과 의미 있게 달라 `RESTART_STATE_CAUSAL_INVALID`로 판정했다.
- causal label은 initial scenario/seed에서 decision frame 직전까지 Pure BT를 그대로 실행하는 Prefix Replay만 사용했다.
- Pure A/B 및 `BT_DEFAULT` override는 intervention 전 180 frame exact parity를 통과했다.
- noise floor는 동일 Pure repeat에서 0이었고 epsilon은 `1e-9`, large regression threshold는 `-1e-6`으로 동결했다.

## Tactical action library와 Oracle

Primary action은 `BT_DEFAULT`이며 nondefault T1은 다음과 같다.

- `PURE_PURSUIT`
- `LEAD_PURSUIT_T060`
- `LAG_PURSUIT_D250`
- hold duration: 30/60/120 frame
- throttle: 항상 same-frame exact Pure BT

258-event acquisition aggregate 결과:

| 지표 | 결과 |
|---|---:|
| clean counterfactual pair | 2,322 |
| Oracle nondefault coverage | 72.09% |
| Oracle intervention mean | +0.063078 |
| Oracle intervention median | +0.004032 |
| overall Oracle policy value | +0.045475 |
| positive geometry | 9 |
| large-regression pair | 799 |
| best static option | `PURE_PURSUIT__d120` |
| best static value | +0.021040 |

따라서 v3의 bounded pulse와 달리 Tactical action space 자체는 feasible하다. 문제는 Oracle option을 server-safe state에서 예측하는 단계다.

## Decision event와 dataset

- Pure BT full-fight: 23 fights, 16,482 frame, 2,038 dynamic decision events, 11 geometry family.
- primary group-fixed dataset: 300 unique event / 3,000 row / 2,700 nondefault pair.
- 93D dataset SHA256: `370A53E66F7691B683955AD8F97DAFDAC5AD1215D041BF339EF06D9EA5D8EEF1`.
- long120 dataset: 독립 revalidation 42 event를 추가해 300 event / 3,000 row, 35 scenario, exclusion 0.
- 127D dataset SHA256: `D36D6FA5CED7FC3FEA8C94C22D27F8E097EB4C28BAA5704521B877F7082A291B`.
- split은 row random이 아니며 fight/trajectory/event/scenario/seed group을 보존한다.

초기 dataset의 `scenario_id`가 실제 case가 아니라 geometry family였던 오류를 발견해, 실제 case ID와 geometry를 분리한 append-only v2 dataset으로 교정했다. 이전 결과와 raw artifact는 덮어쓰지 않았다.

## Temporal observation과 model ladder

기본 contract는 42D current + t-6/t-12/t-30의 17D delta, 총 93D다. startup은 repeat-first/zero-delta이며 packet replay는 byte-identical이다. 추가 진단에서는 t-60/t-120을 더한 127D long120 contract를 사용했다. health/Damage/hidden FDM truth는 입력에 없다.

| 모델 | group-OOF value | precision | coverage | large regression | seed 일치 | 판정 |
|---|---:|---:|---:|---:|---:|---|
| M1 93D MLP 64-64 | +0.000591 | 48.76% | 40.33% | 40.50% | 0 | 실패 |
| M2 4x17 GRU-32 | -0.003349 | 39.29% | 18.67% | 46.43% | 0 | 실패 |
| long120 MLP 64-64 | +0.014467 | 42.57% | 33.67% | 38.61% | 0 | 실패 |

최종 long120 모델은 seeds 41021/41022/41023, 모델별 240 epoch, 총 4,320 optimizer update다. model SHA256은 `D57DCA26BB7648222D84F19D9E623E467EE9E03B703D2843FF0CA52333F51158`이다. 이 artifact는 진단용이며 submission bundle로 동결하지 않았다.

가장 가까운 cross-scenario 93D state끼리도 option별 label sign agreement가 33.7–46.0%였고 large-regression disagreement가 40.7–54.7%였다. 0.5초 MLP, 명시적 GRU, 2초 history 모두 risk를 분리하지 못했으므로 단순 threshold 완화나 epoch/width 증가는 정당화되지 않는다.

## Gate 결과

| 단계 | 결과 |
|---|---|
| Offline policy | 실패 |
| Active counterfactual acquisition | 1 cycle, 독립 42 event 추가 |
| Shadow | offline gate 때문에 미실행 |
| Micro | 미실행 |
| Short/Official Development | 미실행 |
| 200초/Phase 1·2·3 | 미실행 |
| Held-out | 개봉하지 않음 |
| PPO | 조건 미충족으로 미실행 (iteration/step/episode = 0/0/0) |
| Submission freeze | `refusing to freeze unpromoted candidate` |

따라서 paired bootstrap CI, final opponent coverage, final crash/latency 수치는 생성하지 않았다. 존재하지 않는 downstream 결과를 0으로 가장하지 않는다.

## 검증 인프라와 runtime

- exact Prefix Replay evaluator 및 event-boundary checkpoint
- Tactical Oracle/aggregate
- 93D 및 127D server-safe temporal contracts
- distributional MLP/GRU model ladder와 scenario group-OOF
- exact shadow command, conservative abstention, OOD reason, BT-only throttle runtime
- multi-opponent paired gate와 Held-out ordering
- promotion 실패 시 bundle/config 생성을 거부하는 freeze tool
- runnable opponents: autopilot, Champion self-play, Pastel v3.2 BT/XML pair

Tactical runtime unit/smoke에서 invalid 0, throttle violation 0, exact BT shadow/fallback을 확인했다. 다만 offline gate 실패 모델을 실제 개입시키지 않았으므로 performance runtime/200초 latency claim은 하지 않는다.

## 실제 서버와 한계

저장소와 환경에서 실제 대회 endpoint/credential을 찾지 못해 `SERVER_BLOCKED`를 유지한다. 핵심 한계는 현재 server-visible state와 최대 2초 local history가 terminal tactical advantage의 부호와 negative tail을 충분히 식별하지 못한다는 점이다. 다음 연구 line은 대회 규정상 허용되는 더 긴 controller-state surrogate 또는 action-space 자체의 risk-bounded redesign을 별도 version에서 검증해야 한다.

## 최종 제출 권고

`temporal_tactical_hybrid_v4`는 제출하지 않는다. 최종 submission default는 SHA가 동결된 Pure BT Champion이다. 이번 PR에서는 unpromoted model/config를 포함하지 않고, 재사용 가능한 독립 검증 인프라와 실패 근거만 main에 병합한다.
