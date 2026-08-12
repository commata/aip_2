# AI Pilot DogFight RL 모델 개발 및 학습 이어하기 핵심 가이드 (0708 업데이트 반영판)

본 문서는 `0708 RL 환경 업데이트` 및 최신 매뉴얼(`rev9`, `student_manual_rev1`)을 반영하여 **F-16 1v1 공중전 강화학습(RL) 모델 개발 전략**, **커스텀 관측(Custom Observation) 설정 규칙**, **학습 이어하기 전략**을 총정리한 실전 가이드입니다.

---

## 1. 0708 업데이트 핵심 요약

1. **Custom Observation 번들 로드 차원 불일치 버그 해결**
   - 기본 관측(12~16차원) 외에 학생이 정의한 커스텀 관측 벡터(예: 20차원 등)로 학습한 lightweight bundle을 로드할 때 발생하던 입력 차원 불일치 오류(`RuntimeError: size mismatch...`)가 해결되었습니다.
   - 번들 내 `metadata.json`에 `observation_size`, `observation_summary` 정보가 자동으로 기록되어 학습과 평가 시 차원이 자동 일치합니다.
2. **평가 및 제출 진입점 통합**
   - 로컬 교전(`run_local_dogfight.py`), 언리얼 서버 연동(`run_unreal_inference.py`), 제출 클라이언트(`student/my_submission.py`) 모두 동일하게 lightweight bundle(`artifacts/models/...`) 경로를 사용합니다.

---

## 2. 핵심 아키텍처 및 학생 수정 5대 파일

> **중요**: 모든 명령어는 반드시 자산(DLL, XML, aircraft 디렉터리)이 위치한 `DogFightEnv/Release/` 폴더를 작업 디렉터리(cwd)로 하여 실행합니다.

```text
DogFightEnv/Release/
├── train_rllib.py                    # [본체] 1v1 학습 루프 및 체크포인트/번들 내보내기 담당
├── scripts/run_experiment.py         # [본체] YAML 설정 기반 통합 실험 실행기
├── experiments/*.yaml                # [설정] 실험 관리 (반복횟수, 상대 모드, 이어하기 등 설정)
└── student/                          # [학생 영역] 모델 전략 및 동작 핵심 5개 파일
    ├── my_reward.py                  # ⭐ 1순위 수정: 공중전 보상함수 (compute_reward)
    ├── my_observation.py             # 선택 수정: 커스텀 상태 관측 벡터 (build_observation)
    ├── my_curriculum.py              # 선택 수정: 커리큘럼 단계 정의
    ├── my_train.py                   # 선택 수정: 파이썬 기반 학습 래퍼
    └── my_submission.py              # 대회 서버 제출용 Unreal 통신 클라이언트
```

---

## 3. RL 모델 개발 4단계 실전 워크플로

### STEP 1. 전략적 보상 함수 설계 (`student/my_reward.py`)
에이전트가 어떤 상황에서 칭찬을 받고 벌을 받을지 규정합니다.
- **기본 보상**: 승리(`win_reward`), 패배(`loss_reward`), 무승부(`draw_reward`)
- **전술적 보상 항목**:
  - 적기 후방 WEZ(무장 유효 사거리) 진입 보상
  - 적기와의 각도(ATA, AA) 및 거리 축소에 따른 Shaping Reward
  - 시간 지연 방지를 위한 스텝 페널티(`step_penalty`)

---

### STEP 2. 관측 모드 및 YAML 설정 (`experiments/*.yaml`)
실험 관리를 위해 `experiments/student_sac_mlp.yaml` 등을 복사하여 설정합니다.

#### ① 기본 관측(`tactical16`) 및 커스텀 보상(`student.my_reward`) 사용할 때
```yaml
env:
  observation_mode: tactical16
  reward_module: student.my_reward
  target_mode: behavior_tree
  target_behavior_dll: AIP_BASE_target.dll
```

#### ② 커스텀 관측(예: 19차원 `student.my_observation`) 및 커스텀 보상 사용할 때 (현재 규정)
> **주의**: YAML에 `observation_size`를 직접 적지 않습니다. 차원은 `student/my_observation.py`의 `OBSERVATION_SIZE` 상수를 통해 자동 인식합니다.

```yaml
env:
  observation_mode: custom
  observation_module: student.my_observation
  reward_module: student.my_reward
  target_mode: behavior_tree
  target_behavior_dll: AIP_BASE_target.dll
```

---

### STEP 3. 학습 검증 및 실행
```powershell
# 1. 문법 검증 (Dry-run)
python scripts\run_experiment.py experiments\student_sac_mlp.yaml --dry-run

# 2. 본 학습 실행
python scripts\run_experiment.py experiments\student_sac_mlp.yaml
```

---

### STEP 4. 모델 번들 산출 및 로컬 교전 검증
학습 완료/중단 시 최종 모델이 아래에 자동 저장됩니다:
```text
artifacts/models/<team_name>/<output_tag>/
  ├── metadata.json           # 신경망 구조 및 관측 차원 정보 (0708 패치 반영)
  └── policy_weights.pkl.gz   # 정책 신경망 가중치
```

생성된 모델 번들로 AIP_BASE_target.dll 상대 1v1 교전을 실행해 성능을 검증합니다:
```powershell
python run_local_dogfight.py --ownship-bundle-dir artifacts\models\surion\v1
```

---

## 4. 학습 중단 시 이어서 하는 방법 (재학습/전이학습 전략)

### ① 모델 번들 가중치 이어받기 (`init_bundle`) — ⭐ 가장 권장
기존 학습으로 생성된 신경망 가중치(`artifacts/models/<team>/<tag>`)를 불러와 추가로 학습(파인튜닝)하는 방법입니다. 0708 업데이트로 번들 내 메타데이터(`metadata.json`)를 읽어 관측 차원 및 LSTM 구조가 자동으로 동기화됩니다.

- **YAML 설정 시**:
  ```yaml
  output:
    name: surion
    tag: sac_mlp_v2                          # 새로 저장될 태그명
  runtime:
    iterations: 100                          # 추가로 학습할 반복 횟수
    init_bundle: artifacts/models/surion/v1  # 이전 학습 번들 경로
  ```
- **CLI 직접 실행 시**:
  ```powershell
  python train_rllib.py --algorithm sac --iterations 100 --output-name surion --output-tag sac_mlp_v2 --init-bundle artifacts\models\surion\v1
  ```

---

### ② RLlib 완벽 체크포인트 복원 (`restore_checkpoint`)
신경망 가중치뿐 아니라 옵티마이저(Optimizer) 상태와 스텝 정보까지 정확히 복원할 때 사용합니다.

> **⚠️ 주의 (Windows 환경 PyArrow 경로 인식 에러 방지)**
> Windows 환경에서 상대 경로(`artifacts/...`)를 사용하면 내부적으로 PyArrow가 `ArrowInvalid: URI has empty scheme` 에러를 발생시킬 수 있습니다.
> 따라서 `restore_checkpoint`를 사용할 때는 반드시 **절대 경로(Absolute Path)** 형식(예: `C:/Users/...`)으로 기입하는 것을 권장합니다.

- **YAML 설정 시 (절대 경로 사용)**:
  ```yaml
  runtime:
    restore_checkpoint: C:/Users/사용자명/Desktop/AIpilot_gemini/DogFightEnv/Release/artifacts/checkpoints/surion/v1/checkpoint_000050
  ```
- **CLI 직접 실행 시**:
  ```powershell
  python train_rllib.py --restore-checkpoint C:\Users\사용자명\Desktop\AIpilot_gemini\DogFightEnv\Release\artifacts\checkpoints\surion\v1\checkpoint_000050
  ```

---

### ③ 커리큘럼 학습 이어하기 (`resume`)
커리큘럼 학습(`train_curriculum.py`)이 멈췄을 때 이전 진행 스테이지에서 자동으로 재개합니다.

- **YAML 설정 시**:
  ```yaml
  script: train_curriculum
  runtime:
    resume: true                  # 이어하기 활성화
    # start_stage: 2              # (선택) 특정 단계부터 강제 시작할 때
  ```
- **CLI 직접 실행 시**:
  ```powershell
  python train_curriculum.py --output-name surion --output-tag curriculum_v1 --resume
  ```

---

## 5. 실전 명령어 치트시트

```powershell
# 1. 작업 폴더 이동 (항상 Release 루트에서 실행)
cd C:\Users\idos0\Desktop\AIpilot_gemini\DogFightEnv\Release

# 2. YAML 설정 파일 문법 검증
python scripts\run_experiment.py experiments\student_sac_mlp.yaml --dry-run

# 3. YAML 기반 실제 학습 실행
python scripts\run_experiment.py experiments\student_sac_mlp.yaml

# 4. 이전 학습 번들을 불러와 이어서 학습 (CLI)
python train_rllib.py --algorithm sac --iterations 100 --output-name surion --output-tag sac_v2 --init-bundle artifacts\models\surion\v1

# 5. 로컬 1v1 교전 테스트 및 로그 파일 산출
python run_local_dogfight.py --ownship-bundle-dir artifacts\models\surion\v1
```
