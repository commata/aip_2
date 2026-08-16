# AIP2/AIP3 상대 모델 분석

## 파일 정체성

AIP2와 AIP3의 DLL SHA256은 다르다. 하지만 XML 3종(0815/AIP2/AIP3)은 모두 811 byte이고 SHA256 `D84C27B0...F8271EE`로 동일하다. XML 파일명 차이는 행동 다양성의 근거가 아니다.

`Controller_CY.cpp`, `Task_OptimalTurn.cpp`, `Task_PurePursuit.cpp`, `Task_LagPursuit.cpp`, `LibMain.cpp`는 세 모델에서 동일한 SHA256을 가진다. 핵심 행동 차이는 `Task_Pursue.cpp`에 집중된다.

## AIP2와 0815 차이

AIP2는 0815 대비 다음 행동 차이가 있다.

- 고도별 optimal sustained TAS table을 전 구간에서 약 3 m/s 높게 사용한다.
- Head-on 진입 aspect threshold가 `176°` 초과에서 `175°` 초과로 넓어졌다.
- Head-on 진입 최소 거리가 `max firing range + 100 m`에서 `+1800 m`로 커졌다. 근거리 head-on 반응이 큰 폭으로 달라질 수 있다.
- 상태 로깅 활성화 차이가 있지만 이 부분은 조종 로직 차이로 계산하지 않았다.

실제 seed `511` smoke에서 0815는 AIP2 상대로 blue/red 모두 약 68–71초에 저고도 충돌했다. 소스 차이가 실제 궤적 차이로 이어짐을 확인했으나, 1 seed만으로 일반화하지 않는다.

## AIP3와 0815 차이

AIP3의 `Task_Pursue.cpp`는 0815와 비교했을 때 Head-on, GCAS, speed tracking 상태 로깅의 주석 해제 차이만 확인됐다. Turn performance table, Head-on threshold, pursuit 명령 로직은 동일하다. AIP3 DLL은 0815와 바이너리 해시가 다르지만, 제공된 소스 기준으로는 독립적인 전술 다양성이 없다.

seed `511`에서 0815/AIP3의 side를 반전한 두 교전은 종료 시간 `161.8 s`와 ownship/target health 쌍이 대칭적으로 일치했다. 이 결과는 두 모델의 기동이 사실상 동일하다는 소스 분석을 지지한다.

## 평가 사용 판단

- AIP2: 행동적으로 별도 상대로 유효하다.
- AIP3: DLL 적재·서버 호환성 평가에는 유효하지만, 전술 다양성의 독립적 증거로 세지 않는다.
- 최종 강건성 평가에서는 AIP2/AIP3 두 이름만으로 충분하다고 판단하지 않고, 초기 기하·side·seed 변화를 별도 축으로 사용한다.
