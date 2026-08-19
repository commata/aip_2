# Guidance Selector 200초 상한 전체전

## 결론

12개 case × Pure/BT_DEFAULT-only/Hybrid의 36개 교전을 200초 상한으로 실제 실행했다. 모든 run은 유효 terminal로 12.45~52.0초에 종료되어 실제 200초 timeout 도달은 0개다. 이를 36개의 200초 실비행으로 과장하지 않는다.

rule-distilled Hybrid는 운영 계약을 통과해 `SUBMISSION_READY_HYBRID_CANDIDATE`지만 성능은 `NOT_PROMOTED`다.

- 실제 nonzero intervention: 3,148 frame
- Gate active ratio mean: 0.4607393743
- ownship crash / process error / invalid / throttle violation: 0 / 0 / 0 / 0
- target crash: 8 Hybrid runs; 해당 pair는 Damage primary에서 제외
- min altitude / speed: 441.833203 m / 181.803096 m/s
- inference P50/P95/P99/MAX worst-run: 0.039200 / 0.084095 / 0.106604 / 0.114900 ms
- 166.7ms 초과: 0
- throttle difference max: 0

## Paired Damage

clean 4 pair, contaminated 8 pair다. clean Damage Δ는 mean `-0.0030799766`, median `-0.0033841133`, min `-0.0052003859`, max `-0.0003512940`, positive `0/4`다. BT_DEFAULT-only는 Pure와 12/12 exact outcome 및 Damage Δ 0을 재현했다.

## Coverage 제한

target destroyed 12회, target altitude below min 24회로 모두 Phase 1 안에 끝났다. Phase 2/3 cone dwell은 0이며, phase boundary 자체는 unit test 대상이지 이 flight matrix의 실측 coverage가 아니다. AIP2는 0815와 별도 DLL이지만 독립 계보를 입증하지 못해 unseen-independent-opponent 주장에 사용하지 않는다. 실제 서버 정보는 없어 `SERVER_BLOCKED`다.
