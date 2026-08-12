# Custom observation bundle restore 권장 패치노트

## 1. 패치 개요

### 패치명

Release custom observation lightweight bundle 복원 차원 불일치 수정

### 대상 코드베이스

- `DogFightEnv/Release`

### 해결한 문제

custom observation 모듈로 20차원 등 기본 관측이 아닌 observation을 사용해 학습한
lightweight bundle을 추론 로드할 때 다음과 같은 차원 불일치가 발생할 수 있었다.

```text
RuntimeError: size mismatch for pi_encoder.net.mlp.0.weight
checkpoint shape = [256, 20]
current model shape = [256, 12]
```

### 원인

학습 시에는 `student.my_observation` 같은 custom observation hook이 적용되어 RLlib
model이 20차원 입력으로 생성된다. 그러나 추론 bundle 복원 시에는
`build_algorithm_from_bundle()`이 metadata에서 custom observation size를 복원하지 않아
`RLLibInferenceEnv`가 기본 12차원 observation space를 만들 수 있었다.

### 판단 근거

- 세 추론 진입점(`run_local_dogfight.py`, `run_unreal_inference.py`,
  `student/my_submission.py`)은 모두 `RLActionProvider -> build_algorithm_from_bundle()`
  경로로 lightweight bundle을 로드한다.
- weight load 실패는 `env_runner.set_state({"rl_module": weights})`에서 보이지만,
  실제 원인은 그 직전에 만들어진 RLModule 입력 차원이 12로 잘못 확정된 것이다.
- `RLLibInferenceEnv`는 이미 `env_config["observation_size"]`를 받을 수 있으므로,
  loader에서 bundle metadata를 읽어 이 값을 주입하는 방식이 가장 작은 수정이다.

## 2. 변경 파일

### 필수 hotfix 파일

아래 파일이 이번 문제의 핵심 수정이다.

```text
DogFightEnv/Release/src/dogfight/ai/rllib_utils.py
```

변경 내용:

- `_restore_bundle_observation_config()` 추가
- `_resolve_bundle_observation_size()` 추가
- `build_algorithm_from_bundle()`에서 RLlib Algorithm/RLModule build 전에
  observation config 복원 호출 추가
- 복원 우선순위:
  1. `algorithm_config.env_config.observation_summary.size`
  2. `metadata.metadata.observation_summary.size`
  3. `algorithm_config.env_config.observation_size`
  4. `metadata.metadata.observation_size`
  5. `metadata.metadata.observation_module` import 후 `OBSERVATION_SIZE`
  6. built-in `observation_size(mode)` fallback

### 권장 full patch 파일

새로 생성되는 bundle에도 관측 차원 정보가 명시적으로 남도록 아래 파일도 함께 수정했다.

```text
DogFightEnv/Release/train_rllib.py
DogFightEnv/Release/train_curriculum.py
```

변경 내용:

- 새 lightweight bundle metadata에 다음 필드를 저장한다.
  - `observation_mode`
  - `obs_mode`
  - `observation_module`
  - `observation_size`
  - `observation_summary`
- `train_curriculum.py`는 stage final bundle뿐 아니라 emergency bundle에도 같은 metadata를 저장한다.

### 문서/예시 파일

```text
DogFightEnv/Release/README.md
DogFightEnv/Release/experiments/README.md
DogFightEnv/Release/experiments/student_mixed_initial_sac_mlp.yaml
DogFightEnv/Release/experiments/student_mixed_initial_sac_lstm.yaml
```

변경 내용:

- custom observation bundle 복원 방식 설명 추가
- 기존 bundle의 `metadata.json` 응급 우회 방법 추가
- YAML custom observation 작성 예시 추가
- mixed initial scenario YAML에도 custom observation 설정 주석 추가

## 3. 붙여넣기/교체 안내

### 3.1 기존 bundle 추론 오류만 빠르게 고칠 때

참가자 PC에 빠르게 hotfix하려면 아래 파일만 교체해도 된다.

```text
DogFightEnv/Release/src/dogfight/ai/rllib_utils.py
```

이 파일만 교체하면 기존 bundle도 다음 중 하나를 통해 observation size를 복원할 수 있다.

- `metadata.json`에 `observation_summary.size`가 있는 경우
- `metadata.json`에 `observation_size`가 있는 경우
- `metadata.json`에 `observation_module`이 있고, 해당 Python 모듈이 실행 PC에 있는 경우

### 3.2 새 학습/export까지 안정화할 때

배포본을 다시 묶거나 새 학습 결과까지 안정적으로 관리하려면 아래 파일을 함께 교체한다.

```text
DogFightEnv/Release/src/dogfight/ai/rllib_utils.py
DogFightEnv/Release/train_rllib.py
DogFightEnv/Release/train_curriculum.py
DogFightEnv/Release/README.md
DogFightEnv/Release/experiments/README.md
DogFightEnv/Release/experiments/student_mixed_initial_sac_mlp.yaml
DogFightEnv/Release/experiments/student_mixed_initial_sac_lstm.yaml
```

기존 `student_sac_mlp.yaml`, `student_sac_lstm.yaml`, `student_ppo_mlp.yaml`,
`student_ppo_lstm.yaml`에는 이미 custom observation 주석이 있으므로 필수 교체 대상은
아니다.

## 4. 사용자 config 작성 방법

### 4.1 YAML로 학습할 때

`experiments/*.yaml`의 `env` 섹션을 다음처럼 작성한다.

```yaml
env:
  observation_mode: custom
  observation_module: student.my_observation
  target_mode: behavior_tree
  target_behavior_dll: AIP_BASE_target.dll
```

`observation_size`는 YAML에 직접 쓰지 않는다. 관측 차원은
`student/my_observation.py`의 `OBSERVATION_SIZE` 또는 `observation_size()`에서 읽는다.

### 4.2 CLI로 학습할 때

```powershell
cd C:\Users\USER\workspace\DogFightEnv\DogFightEnv\Release
C:\Users\USER\anaconda3\envs\aip\python.exe train_rllib.py `
  --algorithm sac `
  --observation-mode custom `
  --observation-module student.my_observation `
  --output-name team01 `
  --output-tag custom_obs_v1
```

### 4.3 `student/my_train.py`로 학습할 때

`TRAINING_CONFIG`에 custom observation module을 지정한다.

```python
TRAINING_CONFIG = {
    "team_name": "team01",
    "output_tag": "custom_obs_v1",
    "algorithm": "sac",
    "iterations": 50,
    "reward_module": "student.my_reward",
    "observation_module": "student.my_observation",
}
```

혼선을 줄이려면 `ENV_CONFIG["observation_mode"]`도 `"custom"`으로 맞춰둔다.

### 4.4 로컬 추론

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe run_local_dogfight.py `
  --ownship-backend rl `
  --ownship-bundle-dir artifacts\models\team01\custom_obs_v1 `
  --target-backend bt `
  --observation-mode custom `
  --observation-module student.my_observation
```

### 4.5 Unreal 추론

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe run_unreal_inference.py `
  --mode rl `
  --bundle-dir artifacts\models\team01\custom_obs_v1 `
  --team-name team01 `
  --observation-mode custom `
  --observation-module student.my_observation
```

### 4.6 `student/my_submission.py`

```python
BUNDLE_DIR = "artifacts/models/team01/custom_obs_v1"
OBSERVATION_MODE = "custom"
OBSERVATION_MODULE = "student.my_observation"
```

## 5. 기존 bundle 응급 우회

코드 패치를 바로 적용하기 어렵거나 기존 `metadata.json`에 관측 차원 정보가 없는 경우,
아래 값을 `metadata.json`의 `algorithm_config.env_config`에 추가할 수 있다.

```json
{
  "observation_mode": "custom",
  "observation_module": "student.my_observation",
  "observation_size": 20
}
```

주의:

- `observation_size`는 실제 학습된 weight 입력 차원과 반드시 같아야 한다.
- shape만 맞추고 feature 순서가 바뀌면 추론 성능은 무너질 수 있다.
- 가능하면 metadata 수동 수정 대신 `rllib_utils.py` hotfix를 적용한다.

## 6. 검증 결과

### 실행한 검증

문법 검증:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe -m py_compile `
  DogFightEnv\Release\src\dogfight\ai\rllib_utils.py `
  DogFightEnv\Release\train_rllib.py `
  DogFightEnv\Release\train_curriculum.py `
  DogFightEnv\Release\src\dogfight\ai\inference_env.py
```

결과: 통과

metadata 복원 smoke:

- `metadata.metadata.observation_summary.size = 20` 입력
- 결과: `env_config["observation_size"] == 20`

`RLLibInferenceEnv` smoke:

- 입력: `{"observation_mode": "custom", "observation_size": 20}`
- 결과: `observation_space.shape == (20,)`, reset observation shape `(20,)`

기존 bundle fallback smoke:

- `observation_summary` 없이 `observation_module = "student.my_observation"`만 입력
- 결과: 배포 예시 module 기준 `OBSERVATION_SIZE == 8`, mode `student8` 복원

### 미실행 검증

다음 검증은 실행 시간이 길고 local Ray node 상태에 영향을 줄 수 있어 사용자 수행 항목으로
남긴다.

- 실제 Ray/RLlib Algorithm build smoke
- 실제 custom 20차원 bundle weight load smoke
- `run_unreal_inference.py` 서버 접속 smoke

권장 실행 전 절차:

```powershell
ray stop --force
```

또는 스크립트 내부 `try/finally: ray.shutdown()` 사용을 권장한다.

## 7. 참가자 안내 문구

이번 오류는 학습 weight가 잘못 저장된 문제가 아니라, Release lightweight bundle 복원
경로에서 custom observation size를 모델 생성 전에 복원하지 못한 문제입니다.

패치 후에는 기존 bundle을 다시 학습하지 않아도 됩니다. 다만 추론 PC에 학습 때 사용한
`student/my_observation.py`가 있어야 하며, `OBSERVATION_SIZE`와 feature 순서는 학습 때와
동일해야 합니다.

새로 학습/export하는 bundle에는 `metadata.json`에 `observation_size`와
`observation_summary`가 함께 저장되므로, custom observation 모델도 추론 loader가 올바른
입력 차원으로 RLModule을 생성합니다.

## 8. 적용된 작업 로그

관련 작업 로그:

- `LogDevelop/260708_1039_custom_observation_bundle_restore_patch.md`
- `LogDevelop/260708_1026_custom_observation_patch_status_check.md`
- `LogDevelop/260708_1021_custom_observation_bundle_patch_note.md`
