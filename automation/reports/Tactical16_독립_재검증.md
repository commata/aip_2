# Tactical16 독립 재검증

## 대표 trajectory 재확인

T16 seed 3102의 최선/최악/평균 근접 episode를 60Hz telemetry로 다시 그린 뒤 PNG를 직접 열어 확인했다.

| 역할 | 기하 | Damage Δ | First Damage Δ | LOS Δ | LOS-rate Δ | Cone Δ | 해석 |
|---|---|---:|---:|---:|---:|---:|---|
| 최선 | crossing-left | +0.00696 | -0.1000s | -0.1127° | -0.0526°/s | +0.1167s | target crash 종료가 섞여 승격 근거에서 제외 |
| 최악 | lateral-left | -0.00545 | 0.0000s | +0.00747° | -0.00111°/s | -0.0500s | 작은 지속 residual이 Damage 누적을 악화 |
| 평균 근접 | vertical-low | +0.00011 | -0.0167s | -0.00463° | +0.00179°/s | 0.0000s | Pure와 거의 같은 기동 |

Gate window의 평균 residual correction은 대체로 절대값 `0.001~0.007` 범위였다. 작은 correction도 stateful BT의 이후 자세와 사격 누적을 바꿨다. throttle은 각 Hybrid frame에서 BT throttle과 정확히 같고, Pure trajectory와의 throttle 차이는 이미 달라진 state에 대한 BT 출력 차이다.

## 자기 반박

“Tactical16이 실패했다”는 결론이 틀릴 가능성을 지지하는 데이터는 seed 3102 crossing-left와 vertical-high의 개선이다. 하지만 crossing-left는 target crash 오염이 있고, seed 3101은 6/6 Damage가 음수다. 따라서 이 반례는 장시간 연장을 정당화하지 못한다.

## 최종 판단

첫 pilot의 `PROMOTE_CANDIDATE` 조건은 충족되지 않았다. 독립 seed에서 방향이 재현되지 않았으므로 `REVALIDATION_FAILED`다. 이 branch의 성과는 Tactical16 승격이 아니라 제출 관측/Gate/runtime 계약과 rear120 eligible-only 학습 기반을 검증한 것이다.

실제 서버 정보가 없어 상태는 `SERVER_BLOCKED`이며, 최종 Target BT가 없어 `FINAL_CONFIRMED`를 사용하지 않는다.
