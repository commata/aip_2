# 상태 조건부 Shot-Window Residual Preflight 및 가설

## 결론

PR #16은 `OPEN / NOT_MERGED / MERGEABLE`이고 head는 `a72492946938bbeeb034a88e7660826c8a511aa9`다. origin/main은 `0b5ac7eb066f1f9d410ed8cec6ee71a8993394b7`이며 시작 시 로컬 tracked worktree는 clean이었다. 따라서 PR #16 head에서 `codex/state-conditioned-shot-window-residual` branch를 분리하고 Issue #17을 생성했다.

이 단계의 최종 상태는 `PREFLIGHT_COMPLETE`다. 정확성 보강 전체 테스트와 Pure BT smoke가 통과했으므로 다음 gate는 `COUNTERFACTUAL_STAGE_A`다.

## 기존 근거 재검증

- PR #16 first-window pulse: clean mean Damage Δ `-0.0006406865`, positive `15/33 = 45.45%`
- 의미 있는 최선 action: `1/6` geometry, vertical_high 집중
- clean large regression: 4개, 최악 `-0.0058194477`
- 기존 baseline/pulse evaluation SHA256은 manifest와 일치
- final held-out seed `5301~5306` 이름의 artifact는 발견되지 않았고 기존 manifest도 `opened=false`
- `artifacts/evaluations/final_heldout` 등 과거 폴더는 seed 307/401의 별도 2026-08-13 연구이며 이번 봉인 set과 다르다.
- 신규 최종 Target DLL/XML 근거나 실제 server 계약은 없다: `TARGET_BT_PENDING / SERVER_BLOCKED`

## Pure BT 동결

| 항목 | 값 |
|---|---|
| alias | Pure 0815 BT |
| DLL | 외부 `AIP_DCS_GDCC_0815.dll` |
| DLL SHA256 | `4C93B4C6719CB0423388D5FC721D356020A3A36CD5AD2C56B5C3CA795BFE18C9` |
| XML | 외부 `Rule_DCS_GDCC_0815.xml` |
| XML SHA256 | `D84C27B0B8BA22E1649AF2375BE0B83C762BC62EB5047BB590539B374F8271EE` |
| rule alias | `Rule_DCS_GDCC_0815.xml` |
| action/throttle | raw native BT / BT-only |

두 파일은 저장소 밖 Downloads 폴더에 있어 repo-relative path는 `N/A_EXTERNAL`로 기록한다. 이 정확한 pair는 선행 84 episode에서 load·실행됐으며 Phase 0에서 짧은 smoke를 다시 수행한다.

## 단일 핵심 가설과 반증 기준

가설은 signed geometry와 LOS/BT dynamics에 따라 유효 action의 축·부호·ZERO 여부가 달라지며, mirror canonicalization과 동일 상태 counterfactual label로 작은 selector를 만들 수 있다는 것이다.

canonical state에서 action 방향이 반복되지 않거나, pooled clean positive ratio가 `2/3` 미만이거나, best-state Damage 중앙값이 0 이하이거나, 한 clean action이라도 `-0.003`보다 크게 회귀하거나, mirror consistency가 깨지거나, crash/process/nonfinite/throttle 오류가 있으면 가설 실패다.

## Stage A에서 바꾸는 변수 하나

변경 변수는 `Shot-Window elapsed frame` 하나이며 값은 `0/3/6`이다. Pure BT feature 분석에서 +25m Shot-Window entry는 first Damage보다 평균 약 `0.490초` 빨랐다. 기존 pulse는 entry 직후 6 frame(`0.100초`)만 고정 개입해 실패했으므로 같은 action을 3 frame 간격으로 늦춰 local state 변화에 조건부인지를 본다.

Gate, Tactical16 의미, scale `0.125`, raw magnitude `0.1986799091`, 6-frame duration, saturation-aware compose, throttle BT-only, target, horizon은 고정한다. lateral 좌/우와 vertical 상/하 4 geometry × elapsed 3개 × 7 action = 신규 84 episode다. Pure는 선행 exact-deterministic baseline을 hash로 재사용한다.

seed `7101~7103`은 결과 전 고정했지만 randomization이 꺼져 있으므로 seed 번호 자체를 독립 state 근거로 세지 않는다. 실제 neighborhood 독립성은 first ACTIVE window 안의 elapsed state 이동에서 나온다. development `7201~7206`, held-out `5301~5306`은 아직 사용하지 않는다.

## 정확성 인프라 변경

- pooled clean positive ratio `>= 2/3`를 실제 gate에 포함
- best action뿐 아니라 모든 clean action의 large regression을 gate에 포함
- crash/process/nonfinite/throttle violation을 성공에서 제외
- raw paired rows로 gate-critical aggregate 독립 재계산
- Git/DLL/XML/scenario/scale/duration fingerprint가 다르면 resume 거부
- world action을 canonical mirror action으로 변환하고 double-mirror 테스트
- delayed pulse snapshot에 Tactical16 observation, BT action, geometry/LOS dynamics 기록
- ZERO exact BT, throttle BT-only, Gate OFF inference skip 기존 테스트 유지

## 환경

- Python: `C:/Users/shy66/anaconda3/envs/aip/python.exe`, 3.11.15
- Ray/RLlib: 2.54.0 / present
- NumPy/PyTorch: 2.2.6 / 2.13.0+cpu
- OS: Windows 10 build 26200
- GPU: 도구 없음, CPU-only
- sandbox 기본 tempfile은 `PermissionError`가 재현되어 `.runtime_tmp/<run_id>`와 권한 가능한 동일 명령 재실행을 사용한다.

## Preflight 실행 결과

- targeted: `31 passed, 12 subtests passed`
- automation: `153 passed, 26 subtests passed`
- core: `9 passed`
- compileall: PASS
- manifest JSON: 30개 parse PASS
- `git diff --check`: PASS
- Pure 0815 raw-BT smoke: 59 step, 0.9833초, crash 0, timeout 정상 종료
- smoke 후 변경된 `f16_init.xml`은 시작 HEAD 바이트로 복원했고 Rule alias도 제거됨을 확인했다.
