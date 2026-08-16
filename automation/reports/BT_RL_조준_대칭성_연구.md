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

## H7 — mirror-balanced 10D 학습

### 학습 계약

- 독립 training seed: `1701`, `1702`
- 각 30 iteration, seed 1701 기록 기준 sampled step `7,424`
- 변경 변수: 학습 scenario distribution만 exact 좌우·상하 mirror 6종으로 균형화
- 고정 변수: SAC, 10D v2, reward, Gate A, scale `0.125`, saturation-aware composition, action repeat 6, network, budget
- seed 1701 weights SHA256: `31229A7DB673A4344B417384EDA881ECAD61E599614D3730C5AC1BD40DF1559D`
- seed 1702 weights SHA256: `DCC1C67ADEE085FB096CC0F4116E7BFF3A4E79F21D1A57B363845EE1569BF846`

학습 reward는 정책 승격 근거로 사용하지 않았다.

### exact mirror paired 평가

주요 값은 Hybrid - Pure 0815다.

| Training seed / scenario | First Damage | 평균 LOS | LOS-rate | Cone | Damage | 최소 고도 |
|---|---:|---:|---:|---:|---:|---:|
| 1701 lateral_left | -0.0167s | +0.011° | +0.012°/s | -0.033s | -0.0032 | +6.3m |
| 1701 lateral_right | 0.0000s | +0.018° | +0.002°/s | -0.033s | -0.0025 | -6.5m |
| 1701 crossing_left | 0.0000s | +3.651° | +0.160°/s | 0.000s | -0.0031 | -1010.6m |
| 1701 crossing_right | +0.0167s | -0.011° | -0.018°/s | +0.017s | +0.0026 | -6.2m |
| 1701 vertical_high | +0.5667s | +0.858° | +0.790°/s | -3.983s | +0.0057 | +1066.6m |
| 1702 lateral_left | -0.0167s | +0.006° | +0.009°/s | -0.017s | -0.0043 | +2.9m |
| 1702 lateral_right | +0.0167s | -0.009° | 약 0.000°/s | +0.067s | -0.0018 | +8.3m |
| 1702 crossing_left | 0.0000s | -0.001° | -0.001°/s | +0.017s | -0.0006 | -0.4m |
| 1702 crossing_right | +0.0167s | -0.014° | -0.006°/s | 0.000s | +0.0030 | -2.9m |
| 1702 vertical_high | +0.7000s | +0.941° | +0.794°/s | -3.850s | +0.0057 | +1066.6m |

`vertical_low`는 두 seed 모두 First Damage 변화가 0이고 aim 지표 변화도 거의 0이었다.

seed 1701 `crossing_left`는 Pure가 19.88초에 target destroy로 끝났지만 Hybrid는 target destroy가 아니라 25.30초 target altitude termination으로 끝났다. 승리 표시는 같아도 Hybrid의 평균 LOS는 `1.269° → 4.920°`, 최소 고도는 `2251.8m → 1241.3m`로 악화됐다. outcome 하나만 보면 숨겨지는 실제 기동 회귀다.

두 정책의 `vertical_high`는 First Damage가 각각 `+0.5667s`, `+0.7000s` 늦고 LOS/LOS-rate/Cone이 함께 악화됐다. 다만 target destroy 종료 시점은 Pure 20.82초보다 Hybrid 약 15.7초로 빨랐다. 최초 조준 정렬은 나빠졌지만 Damage 누적 구간의 기동은 달라졌다는 뜻이며, 하나의 metric으로 성공/실패를 단정할 수 없다.

### unseen mirror hold-out 재검증

학습에 없는 거리, 횡오프셋, 속도, 고도 조합 6종을 exact mirror로 추가했다. 결정론적 scenario이므로 seed `1901` 숫자 자체를 독립 표본 근거로 사용하지 않고, 새로운 geometry와 독립 training seed 2개만 독립성 근거로 사용한다. Stage-1 비교 horizon은 Pure/Hybrid 모두 15초로 고정했다.

- lateral-right First Damage delta: seed 1701 `0.0000s`, seed 1702 `+0.0167s`
- crossing 좌우 First Damage gap은 seed 1701에서 `0.0667s`, seed 1702에서 반대 방향 `-0.0167s`로 일관되지 않았다.
- hold-out LOS delta 범위: 약 `-0.015° ~ +0.011°`
- crash: 0
- Gate active ratio: `0.587 ~ 0.876`
- inference P95: `0.456 ~ 0.619ms`, 최대 `1.499ms`
- roll/pitch/yaw applied/requested ratio는 scenario별로 대체로 `0.72 ~ 1.00`이지만, exact lateral-right와 이전 H6에서는 더 큰 authority 손실이 존재했다.

hold-out에서 과거 H6의 `+1.25 ~ +1.32s` lateral-right 지연은 재현되지 않았다. 그러나 seed 1701 crossing-right Damage는 `-0.01068`, seed 1702 crossing-left Damage는 `-0.00197`였고 Gate가 최대 87.6% 켜져 있었다. H7을 개선 정책으로 승격할 근거는 없다.

## 추론 경로 데이터 품질 문제와 수정

초기 hold-out 실행에서 Pure는 정상 완료했지만 Hybrid subprocess 12개가 모두 120초 timeout(return code 124)으로 끝났다. 300초 재시도 6개와 15초 probe도 같은 방식으로 실패했고 telemetry가 없었다. 이 결과는 정책 실패가 아니라 무효 데이터로 분류했다.

H6/H7 모두 Ray가 시작된 뒤 SAC `config.build_algo()`에서 멈췄다. 같은 metadata로 learner/replay 없는 inference-only RLModule을 직접 구성하면 약 `0.03초`에 만들어졌다. 새 adapter를 적용한 뒤:

- 구 full-Algorithm 기록과 120 frame 비교
- residual 활성 48 frame raw residual 최대 절대차: `0`
- final action 최대 절대차: `0`
- exact 12 episode의 First Damage, 평균 LOS, Damage가 기존 결과와 소수점 9자리까지 동일
- 1초 paired probe: 181초 timeout → 약 6.3초 전체 평가 완료

따라서 정책 의미를 바꾸지 않고 불필요한 Algorithm 초기화 병목만 제거한 것으로 판단한다. 서버 동작은 아직 실제 UDP 환경에서 검증하지 않았으므로 서버 검증 완료라고 기록하지 않는다.

## 대표 trajectory 재확인

### 가장 좋아 보인 사례

hold-out seed 1701 `crossing_left`는 First Damage `-0.05s`, Cone `+0.05s`, Damage `+0.00279`였다. 그러나 gate-window 평균 LOS는 `+0.0071°`, LOS-rate는 `+0.0086°/s`로 소폭 나빠졌다. 작은 시간 이득만으로 조준 개선이라고 결론내릴 수 없다.

### 가장 나쁜 사례

exact seed 1701 `crossing_left`는 앞서 설명한 것처럼 target destroy가 target altitude termination으로 바뀌고, 후반 trajectory에서 LOS와 고도가 크게 악화됐다. Gate가 켜진 공통 시간대 평균 residual correction은 roll `+0.00351`, pitch `-0.00046`, yaw `+0.00530` 수준으로 작았지만 작은 지속 편향이 종료 분기와 후반 기동을 바꿨다.

### 거의 중립인 사례

hold-out seed 1702 `lateral_left`는 LOS `-0.00047°`, First Damage `0s`, Damage `+0.00035`, 최소 고도 `-0.57m`였다. gate-window 누적 Damage 차이 `+0.00150` 외에는 실질적으로 Pure와 가깝다.

## H7 최종 판단

판단: `NEED_MORE_EVIDENCE`

신뢰도: `중간`

1. 수학적 observation/sign 오류는 발견되지 않았다.
2. mirror-balanced 학습 후 과거 1.25초급 lateral-right 병리는 exact/hold-out에서 반복되지 않았다.
3. 그러나 lateral-right가 해결됐다고 단정할 수 없다. deterministic hold-out 수가 작고 one-frame 지연과 authority 비대칭이 남아 있다.
4. seed 1701 crossing-left의 큰 trajectory 회귀와 두 seed vertical-high의 aim 회귀는 10D 정책의 seed/geometry 일반화가 불안정함을 보여준다.
5. 따라서 H7을 PROMOTE하지 않으며, 사용자 조건에 따라 지금 즉시 H8 13D로 전환하지 않는다.

### 반대 가설과 자기 반박

현재 결론이 틀렸다고 가정할 때의 근거:

- H7의 hold-out은 결정론적 geometry 1세트라 표본 독립성이 제한된다.
- exact vertical-high는 First Damage/LOS는 나쁘지만 최종 target destroy는 빨라 단순 실패로만 해석할 수 없다.
- Gate active ratio가 59~88%로 높아 observation보다 Gate 폭이 회귀를 키웠을 수 있다. 다만 이번 단계에서는 Gate를 동결했다.
- seed 1701 crossing-left만 크게 붕괴했으므로 stochastic training outlier일 수 있다.

### 다음 가설

H8 전환 전에 최소 한 개의 추가 독립 training seed와 training variant 실제 선택 빈도를 계측해 distribution 균형이 실행 중에도 성립했는지 재검증한다. 동일 pathology가 반복되면 그때 `aim_residual13_btaware`를 구현하며, 같은 frame의 단일 BT tick cache를 observation과 composition이 공유하는 자동 테스트를 먼저 작성한다.
