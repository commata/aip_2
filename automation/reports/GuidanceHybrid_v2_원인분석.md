# Guidance Hybrid v2 원인 분석

## 결론

Hybrid v1의 3,148개 개입 frame은 독립 표본이 아니라 **89개 연속 intervention event**다. frame 수를 학습 표본 수나 causal evidence 수로 사용하면 안 된다. 전체 clean full-fight 결과도 mean/median Damage Δ가 각각 `-0.0030799766/-0.0033841133`, positive `0/4`였으므로 v1은 계속 `NOT_PROMOTED`다.

가장 직접적인 controller 원인은 v1이 angular offset을 같은 설정값으로 다시 나눠 action unit을 계산한다는 점이다. 따라서 `0.10°`, `0.25°`, `0.50°` 모두 같은 roll/pitch/yaw correction을 만든다. action magnitude 이름과 실제 controller effect가 분리돼 있었고, v2에서는 degree 비례 closed-loop 계약으로 바꿔야 한다.

## Event 재구성

- 분석 fight: 12
- 전체 telemetry frame: 25004
- Gate entry: 41
- Gate active frame/ratio: 7527 / 30.10%
- intervention event/frame: 89 / 3148
- event duration mean/median/P95/max: 35.37 / 41.00 / 41.00 / 41 frame

각 event에는 시작·종료 geometry와 `+0.5s/+1s/+2s/+4s/+8s/terminal` candidate 및 동일 시간대 BT_DEFAULT snapshot을 연결했다. 원본은 `automation/evidence/guidance_advantage_v2/phase_a_analysis.json`에 보존한다.

## Action별 event 진단

| action | event | frame | terminal ΔDamage mean | median | positive |
|---|---:|---:|---:|---:|---:|
| VP_EL_POS_SMALL | 89 | 3148 | -0.0000363 | +0.0000000 | 7.9% |

## Event 이후 paired trajectory 진단

| horizon | ΔDamage mean | median | Δaim error mean | ΔLOS rate mean | Δcone mean |
|---|---:|---:|---:|---:|---:|
| +0.5s | -0.0000071 | +0.0000000 | -0.1764 | +0.5501 | +0.0000 |
| +1s | -0.0000067 | +0.0000000 | -0.0473 | +0.0507 | +0.0000 |
| +2s | -0.0000105 | +0.0000000 | -0.0122 | -1.1326 | +0.0000 |
| +4s | -0.0000145 | +0.0000000 | +0.6277 | -1.1645 | +0.0000 |
| +8s | -0.0000227 | +0.0000000 | +0.9657 | +0.7090 | +0.0000 |
| terminal | -0.0000363 | +0.0000000 | +0.9047 | -0.9328 | +0.0000 |

event 시작 시 signed aim azimuth/elevation, LOS az/el rate, closing rate, BT roll/pitch/yaw의 mean/median/min/max 및 부호 분포를 evidence JSON에 기록했다. 각 frame의 controller payload에서 requested/applied correction과 positive/negative surface headroom도 event 시작·종료·각 horizon에 보존했다.

## 확인된 손실 원인

1. 고정 `VP_EL_POS_SMALL`만 실제 개입해 signed azimuth/elevation 방향을 사용하지 않았다.
2. 3,148 frame은 89 event 내부의 강한 시계열 중복이다.
3. angular magnitude 설정이 실제 surface correction 크기를 바꾸지 않는다.
4. v1은 45D observation에 ownship/target health를 포함해 서버 보장 feature 계약을 위반한다.
5. v1 primary library에는 Range/Target Speed 이름이 있으나 실제로는 pitch bias로 변환되어 의미가 일치하지 않는다.
6. Gate active ratio가 30.10%로 넓고, rule은 gate 초기 36 frame마다 개입해 precision보다 개입량을 키웠다.
7. clean full-fight pair가 4개뿐이고 8개 pair는 target-crash contaminated였다.

## 해석 한계와 다음 검증

- 뒤쪽 event는 candidate와 BT_DEFAULT trajectory가 이미 갈라진 뒤 시작하므로 horizon delta는 진단값이지 randomized causal estimate가 아니다.
- 다음 단계는 primary action을 `BT_DEFAULT/AZ±/EL±`로 제한하고, degree 비례 controller에서 magnitude·duration을 동일 초기 state paired ablation해야 한다.
- clean Damage가 여러 geometry에서 반복 양수일 때만 dataset/Advantage model로 진행한다.
