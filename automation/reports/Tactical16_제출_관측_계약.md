# Tactical16 제출 관측 계약

## 결론

학습과 제출이 공유하는 관측 계약을 `tactical16.v1`로 고정했다. 동일 synthetic state와 공식 WEZ 설정을 입력하면 두 경로의 float32 16개 벡터가 byte 단위로 같다.

## Feature 순서

1. ownship roll
2. ownship pitch
3. ownship yaw
4. ownship speed
5. ownship altitude
6. ownship health
7. relative north
8. relative east
9. relative down
10. ATA
11. Aspect Angle
12. LOS azimuth
13. LOS elevation
14. target health
15. in WEZ
16. pursuit score

## 제출용 의미

- health: 서버 PlaneInfo에 health가 없으므로 ownship/target 모두 상수 `+1.0`이다. metadata 값은 `unavailable_constant_one`이다.
- WEZ: 공식 최소 거리 500ft와 phase별 거리·반각을 공통 함수로 계산한다.
- phase: 첫 paired frame을 0초로 두고 `(frame_index - start_frame) / 60`으로 계산한다.
- 좌표: Unreal Z-up position은 NED down으로 부호 변환한다.
- bundle: config의 mode·size·contract·normalization·health·WEZ·phase와 metadata가 다르면 서버 join 전에 중단한다.

## 검증

| 항목 | 결과 |
|---|---|
| synthetic train/submission vector | byte 동일 |
| health 상수 | 양 기체 `+1.0` |
| 100/150/200초 phase 경계 | 동일 |
| Phase 1/2 WEZ 경계 사례 | 예상 부호 확인 |
| yaw wrap 및 좌표 변환 | 단위 테스트 통과 |
| invalid metadata/config | fail-fast |

## 남은 불확실성

PlaneInfo velocity magnitude를 현재 simulator state의 `KCAS` 위치에 넣는다. 외부 서버가 제공하는 속도 단위와 학습 simulator의 해당 feature 의미가 같다는 실제 packet 근거는 아직 없다.

## 판단

`SUBMISSION_PARITY_CONFIRMED` — synthetic 관측 계약 범위. 실제 bundle과 외부 서버는 별도 검증 대상이다.

## 자기 반박

byte 동일성은 같은 변환 함수를 공유해서 얻어진 결과일 수 있다. 독립 server packet의 속도·회전 convention이 다르면 양 경로가 같이 틀릴 수 있으므로 실제 서버 telemetry와 교차 확인해야 한다.
