# BT+RL 조준 잔차 좌우·상하 대칭성 연구

## 연구 기준

- 부모 main: `6b95bd9bda5767e2f41b779a45150a77a6d06e56`
- 연구 Issue: `#5`
- 기준 정책: H6 `stage1_aim_balanced_s0125_r06`
- 기본 residual scale: `0.125`
- 관측: `aim_residual10_v2` 10D
- Gate/Reward/Network/학습 budget: H6와 동결
- H7의 핵심 변경 변수: training scenario distribution의 정확한 좌우·상하 mirror 균형화

## 공중전 참고 자료 접근 기록

`http://www.aircombat.pe.kr/frame1.htm` 직접 접근을 시도했으나 2026-08-17 기준 HTTPS 전환 후 timeout으로 본문을 가져오지 못했다. 검색으로 확인된 자료는 관련 없는 과거 게시판 글뿐이어서 조준·pursuit·에너지에 관한 구체적 설명을 연구 근거로 사용하지 않았다. 접근하지 못한 내용을 추측해 reward나 Gate에 반영하지 않는다.

BFM 해석 우선순위는 공식 규정 → 실제 simulator 코드 → 0815 BT 동작 → telemetry → 외부 참고 자료로 유지한다.

## H7 사전 진단 — exact mirror 계약

### 변환 정의

좌우 mirror:

- aim azimuth, LOS azimuth rate: 부호 반전
- aim elevation, LOS elevation rate: 동일
- roll/yaw command: 부호 반전
- pitch/throttle: 동일
- distance, closing rate, ATA, target ATA: 동일

상하 mirror:

- aim elevation, LOS elevation rate: 부호 반전
- aim azimuth, LOS azimuth rate: 동일
- roll/pitch command: 부호 반전
- yaw/throttle: 동일
- distance, closing rate, ATA, target ATA: 동일

상하 변환은 중력까지 반전하는 동역학 대칭 주장이 아니라 순간 운동학 sign convention 진단이다.

### 자동 검증

- 수학적 state mirror의 geometry 오차: `1e-9` 자리까지 일치
- 10D 관측의 부호 계약 검증
- 요청 residual / 적용 residual / final action 제어축 계약 검증
- 체크인된 6개 scenario가 선언한 mirror pair와 정확히 일치하는지 검증

## H7 사전 진단 — 실제 0815 BT와 H6

### 설정

- seed: `1701`
- scenario: exact mirror 6종
- target: 동일 autopilot
- episode: 최대 30초, 60Hz
- 비교: Pure 0815 / H6 Hybrid 0.125
- 해석 상태: `PROVISIONAL`

### Pure 0815 자체의 episode mirror gap

gap은 두 번째 scenario - 첫 번째 scenario다.

| Pair | First Damage gap | 평균 LOS gap | LOS-rate gap | Cone dwell gap | Roll/Pitch/Yaw 포화 gap |
|---|---:|---:|---:|---:|---:|
| lateral right - left | +3.1167s | +2.5373° | +0.8132°/s | -1.7667s | +0.4145 / +0.0287 / +0.4361 |
| crossing right - left | -2.7333s | -0.0689° | +0.7538°/s | -0.7333s | -0.1215 / +0.0000 / -0.0964 |
| vertical low - high | -2.7333s | +0.1522° | -2.1934°/s | -2.3833s | -0.0709 / -0.0336 / -0.1022 |

좌우 exact scenario의 최초 수 frame에서 roll/yaw command는 부호까지 정확히 mirror였고 throttle도 같았다. 그러나 lateral pair의 최초 1초 pitch command mirror RMSE는 `0.0852`였으며, 작은 초기 수치 차이가 다른 BT/동역학 trajectory로 확대됐다. episode 수준의 lateral-right 지연 대부분은 H6 residual을 넣기 전 Pure 0815에도 존재한다.

### H6 Hybrid - Pure 0815 paired delta

| Scenario | First Damage Δ | 평균 LOS Δ | LOS-rate Δ | Cone Δ | Damage Δ | Gate | 활성 step 포화 |
|---|---:|---:|---:|---:|---:|---:|---:|
| lateral_left | -0.0167s | +0.0120° | +0.0095°/s | -0.0333s | -0.00235 | 0.918 | 0.163 |
| lateral_right | +0.0333s | +0.0405° | +0.0053°/s | -0.0833s | -0.00208 | 0.932 | 0.664 |
| crossing_left | 0.0000s | 약 +0.01° | 약 0.00°/s | 0.0000s | 소폭 음수 | 0.797 | 0.097 |
| crossing_right | 0.0000s | 약 +0.02° | 약 +0.01°/s | -0.0167s | 소폭 양수 | 0.765 | 0.000 |
| vertical_high | +1.0500s | +1.3232° | +0.8641°/s | -3.2500s | 소폭 양수 | 0.904 | 0.499 |
| vertical_low | 0.0000s | 약 0.00° | 약 0.00°/s | 0.0000s | 약 0.00 | 0.867 | 0.144 |

### Authority 분리

- lateral_left 요청 대비 적용 비율: roll `0.973`, pitch `0.857`, yaw `0.742`
- lateral_right 요청 대비 적용 비율: roll `0.748`, pitch `0.767`, yaw `0.432`
- lateral_right는 lateral_left보다 적용 authority가 모든 축에서 작고, 특히 yaw가 작다.
- clipping은 0%지만 lateral_right 활성 step 포화는 66.4%다.

따라서 clipping 제거는 authority 문제 해결을 의미하지 않는다.

## 잠정 결론

현재 exact mirror seed 1개에서는 다음이 보인다.

1. observation geometry/sign 계산 자체는 수학적으로 대칭이다.
2. 과거 lateral-right 지연 전체를 10D partial observability로 설명할 수 없다.
3. Pure 0815 BT/동역학 자체의 episode 좌우 차이가 큰 혼입 변수다.
4. H6 residual이 exact lateral pair에 추가한 first-Damage 좌우 gap은 약 `0.05초`지만, lateral-right의 포화와 authority 손실은 훨씬 크다.
5. H6의 더 뚜렷한 residual 회귀는 exact `vertical_high`에서 나타났다.

판단: `NEED_MORE_EVIDENCE`

신뢰도: `낮음~중간`

## 반대 가설과 자기 반박

현재 결론이 틀렸다고 가정할 때 이를 지지하는 취약점:

- actual BT 진단 seed가 하나뿐이다.
- 좌우 동역학 divergence가 JSBSim 수치 민감도 또는 heading 5°/355° 표현 차이일 수 있다.
- 이전 H6의 +1.3167초 우측 회귀는 randomization이 포함된 curriculum variant에서 관측됐고, 이번 exact scenario와 조건이 다르다.
- H6 residual이 작아 exact lateral pair에서는 문제를 충분히 자극하지 않았을 수 있다.
- 상하 pair는 중력 때문에 full trajectory mirror 비교가 원래 성립하지 않는다.

## 다음 재검증

1. H7 seed 1701/1702를 동일 budget으로 독립 학습한다.
2. 각 정책을 exact mirror 6종에서 새로운 validation seed로 Pure BT와 paired 평가한다.
3. lateral-left/right의 Hybrid-Pure delta 차이와 requested/applied authority를 비교한다.
4. 개선 방향이 두 training seed에서 반복되는지 본다.
5. right/vertical pathology가 반복될 때만 H8 13D BT-aware로 전환한다.
