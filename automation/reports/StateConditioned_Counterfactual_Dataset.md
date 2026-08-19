# 상태 조건부 Counterfactual Dataset 판정

## 결론

`COUNTERFACTUAL_SIGNAL_INSUFFICIENT / DATASET_NOT_CREATED`다. Shot-Window elapsed `0/3/6` frame의 상태 조건부 library는 84/84 episode를 오류 없이 완료했지만, clean Damage gate를 통과하지 못했다. label을 만들면 다시 `vertical_high` 한 geometry를 선택하는 결과가 되므로 dataset sample은 0이다.

## 실행 계약

- 변경 변수: `shot_window_elapsed_frames = 0, 3, 6`
- geometry: lateral 좌/우, vertical 상/하
- action: ZERO, ±ROLL, ±PITCH, ±YAW
- 신규 episode: `4 × 3 × 7 = 84`
- Pure reference: exact-deterministic baseline 12 record를 hash로 재사용
- seed: 7101/7102/7103, seed 숫자만 독립 state로 세지 않음
- Gate/scale/magnitude/duration: Shot-Window v1 / 0.125 / 0.1986799091 / 6 frame
- throttle: BT-only, RL inference: 0

## clean Damage 결과

| 범위 | mean | median | min | max | positive |
|---|---:|---:|---:|---:|---:|
| 전체 72 pair | -0.0005015 | 0.0000000 | -0.0058194 | +0.0022513 | 31/72 (43.06%) |
| elapsed 0 | -0.0007514 | - | -0.0058194 | +0.0020402 | 11/24 |
| elapsed 3 | -0.0003084 | - | -0.0056901 | +0.0022513 | 13/24 |
| elapsed 6 | -0.0004446 | - | -0.0038528 | +0.0011711 | 7/24 |

| geometry | mean | min | max | positive |
|---|---:|---:|---:|---:|
| lateral_left | -0.0018441 | -0.0058194 | +0.0007774 | 6/18 |
| lateral_right | -0.0010902 | -0.0042804 | +0.0000905 | 2/18 |
| vertical_high | +0.0010051 | 0.0000000 | +0.0022513 | 16/18 |
| vertical_low | -0.0000767 | -0.0002748 | +0.0000647 | 7/18 |

## gate 실패 원인

- pooled clean positive ratio: `43.06% < 66.67%`
- 의미 있는 world geometry: `1/4 < 4/4`
- 의미 있는 best state: 3/12, 전부 `vertical_high`
- best-state Damage Δ 중앙값: `+0.0003730`이지만 다른 필수 조건 실패
- clean large regression: 9개
- 최악: `lateral_left / elapsed 0 / roll_neg = -0.0058194`
- canonical mirror pair: 0, 따라서 mirror consistency 미확인/실패
- vertical_high best action도 elapsed 0에서는 `pitch_pos`, elapsed 3/6에서는 `yaw_pos`로 바뀜

의미 있는 positive 3개는 각각 `+0.0020402`, `+0.0022513`, `+0.0011711`이지만 모두 vertical_high다. vertical_low mirror에서는 최선이 최대 `+0.0000647`로 threshold보다 작아 canonical pair가 성립하지 않았다.

## 보조 지표와 authority

- First Damage Δ 평균: `-0.00139초`
- Cone Δ 평균: `+0.00231초`
- Gate active ratio 평균: `0.09725`
- ACTIVE duration mean/P95/max: `0.9444/1.0/1.0초`
- nominal requested correction(active axis): `0.024835`
- aggregate applied/requested: roll `0.758`, pitch `0.898`, yaw `0.819`

First Damage/Cone가 아주 조금 좋아졌어도 실제 clean Damage 평균은 음수이므로 label 근거로 사용하지 않는다. authority는 0이 아니어서 measurement bug/authority-zero 예외 수정도 허용되지 않는다.

## 데이터 품질

- Pure/ZERO trajectory exact equality: PASS
- target crash contamination: 0
- ownship crash: 0
- process error: 0
- invalid/nonfinite: 0
- throttle violation: 0
- raw/aggregate 독립 재계산: 일치
- minimum altitude/speed: 2388.39m / 183.40m/s

raw evaluation SHA256은 `B565B7AB86103832FC12BF6C30D3FA31159954BBD93D18E643E3CEAA21AE224E`다. compact evidence에는 72 counterfactual pair를 보존했지만 모두 label confidence 0이며 dataset으로 승격하지 않는다.
