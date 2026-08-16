# SAC 학습 Runtime 정지 진단

## 목적

H7 세 번째 독립 seed `1703`이 SAC `config.build_algo()` 이후 5분 이상 iteration 0에 진입하지 못한 원인을 정책·환경·Ray 자원·파일 권한으로 분리했다. 이 문서는 정책 성능 보고서가 아니라 장시간 학습 실행 기반의 검증 기록이다.

## 고정 조건

- 알고리즘: SAC
- observation: `aim_residual10_v2`
- Gate: Aim Gate A
- residual scale: `0.125`
- 결합: `saturation_aware`
- throttle: BT command 유지
- training seed: `1703`
- 학습 예산: 1 iteration, 256 sampled steps
- scenario: H7의 6개 mirror-balanced variant

변경 변수는 RLlib logger 경로와 Ray CPU 제한뿐이다. CPU 제한 대조 이후의 장시간 실행에서는 `OMP/MKL/OPENBLAS/NUMEXPR` thread 수 역시 YAML의 `runtime.math_threads`로 동결한다.

## 진단 결과

### 1. 기본 logger, 제한 없는 Ray

- Ray 초기화: `40.67초`
- 생성된 child process: `23개`
- 이후 위치: `Algorithm.__init__ -> tempfile.mkdtemp`
- 60초 간격 stack dump 10회가 모두 같은 호출 경로를 가리켰다.
- 632초 동안 sampled step은 `0`이었다.
- Windows `tempfile`이 RLlib 기본 `~/ray_results` 아래 디렉터리 생성의 `PermissionError`를 이름 충돌처럼 재시도해 CPU를 소비했다.

따라서 이 정지는 SAC 정책, replay buffer, simulator step, seed 1703의 특정 물리 상태 때문에 발생한 것이 아니다.

### 2. 작업공간 logger, 제한 없는 Ray

작업공간의 `artifacts/ray_results/<output>/<tag>`를 명시하는 logger를 적용하자 algorithm build는 약 `1.42초`에 끝났고 첫 train iteration 계산도 진행됐다. 제한된 실행 권한에서는 child가 `progress.csv`를 생성할 때 ACL 오류가 발생했지만, 정상 로컬 권한으로 같은 설정을 실행하면 다음이 완료됐다.

- Ray 초기화: `41.06초`
- algorithm build: `1.42초`
- iteration: `2.72초`
- sampled steps: `256`
- completed episodes: `1`
- lightweight bundle: 생성
- native checkpoint: 생성

### 3. 작업공간 logger, Ray CPU 2개

- Ray 초기화: `8.80초` (`41.06초` 대비 약 `78.6%` 감소)
- child process: `9개` (`23개`에서 감소)
- algorithm build: `1.41초`
- iteration: `2.68초`
- sampled steps: `256`
- completed episodes: `1`
- lightweight bundle 및 native checkpoint: 생성

환경 episode return과 주요 custom metric은 같았지만 replay sample count가 `164 -> 160`, policy loss가 `-2.5966 -> -2.5990`, Q loss가 `2.2753 -> 2.3230`으로 달랐다. Ray CPU 제한이 초기화 비용만 바꾼다고 단정하지 않는다. 장시간 비교에서는 CPU/thread 조건을 모든 후보에 동일하게 고정한다.

## 단일 episode telemetry의 해석 제한

CPU 2개 대조에서 reward `40.8046`, Gate active ratio `0.9787`, action saturation ratio `0.3396`, clipping ratio `0`, crash `0`이 기록됐다. 선택 variant는 `vertical_high`였다. 이는 runtime 통과 여부를 확인한 한 episode일 뿐이며 정책 PROMOTE 근거로 사용하지 않는다. Gate active ratio와 saturation은 오히려 후속 분석 대상이다.

## Artifact

- CPU2 bundle weights SHA256: `9377C090EBF6BE83F45D4810576AF81193980FCAB7FAF2DBC8479ADFA55028C4`
- CPU2 bundle metadata SHA256: `88C5E21DFB8B58E4162EDEB1BC3A4F4D73293D26FD4D24BA6717A3EA3B8B47D8`
- CPU2 RL module state SHA256: `08180381E9AA38E523EB6C709A337852B1F0D2C9EC704AA1D4789AD01435A16F`
- CPU2 stdout SHA256: `42FCB454186E2152EAB67190AF92B073A60BD375E771DB1B851A8A106EB6C7B1`
- CPU2 stderr SHA256: `72736F1BE488BA998CB1566C0254911C3D0FBF65839DD421759908780B8F2C6C`

대형 bundle, checkpoint, Ray result, raw process log는 Git에 추가하지 않고 `artifacts/` 아래 보존한다.

## 검증

- `PYTHONPATH=src;.` automation suite: `53 passed`
- top-level suite: `9 passed`
- `compileall` 및 `git diff --check`: 통과
- import 경로를 지정하지 않은 첫 automation 실행은 `dogfight` module 수집 오류 3건으로 실패했으며 코드 regression으로 계산하지 않았다.

## 판단

- Runtime 기반 판단: `PROMOTE_CANDIDATE`
- 정책 성능 판단: `NEED_MORE_EVIDENCE`
- 기존 상태 `LOCAL_TRAINING_BLOCKED`는 로컬 정상 권한 실행 기준으로 해소됐다.
- 이후 고정 runtime: workspace logger, `ray_num_cpus: 2`, `math_threads: 2`

## 자기 반박과 남은 불확실성

현재 결론이 틀렸다고 가정하면, 정상 권한 실행이 파일 권한 문제를 숨겼거나 CPU 제한에 따른 learner 수치 차이가 장시간 누적될 수 있다. 또한 이번 완료 episode는 하나뿐이다. 따라서 최소 다중 iteration pilot과 독립 seed에서 checkpoint/복구/variant frequency를 다시 검증하기 전에는 장시간 학습 기반을 최종 확정하지 않는다.
