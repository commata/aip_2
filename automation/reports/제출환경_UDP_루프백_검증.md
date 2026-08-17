# 제출 환경 UDP 루프백 검증

## 조건

- localhost 실제 UDP socket
- 600 paired frame
- 60Hz, 10초
- BT 매 frame
- RL action repeat 6
- threshold 166.7ms

## 결과

| 지표 | 값 |
|---|---:|
| 요청 frame | 600 |
| CMD | 600 |
| 누락 | 0 |
| 중복 | 0 |
| 잘못된 CMD index | 0 |
| BT 호출 | 600 |
| RL 호출 | 100 |
| P50 | 1.391ms |
| P95 | 2.787ms |
| P99 | 3.550ms |
| MAX | 6.349ms |
| 166.7ms 초과 | 0.0% |

별도 negative geometry 30 frame에서도 CMD 누락 없이 RL 0회, exact BT, throttle 차이 0을 확인했다.

## 판단

localhost 제출 전송 계층은 60Hz 계약을 만족한다. 외부 네트워크·실제 DLL·실제 bundle이 포함된 서버 지연으로 일반화하지 않으며 현재 실제 서버 상태는 `SERVER_BLOCKED`다.

## 자기 반박

localhost는 packet loss와 WAN jitter가 사실상 없다. 실제 서버에서는 join/heartbeat 경쟁, 네트워크 손실, bundle 초기화 및 inference tail latency가 더 클 수 있다.
