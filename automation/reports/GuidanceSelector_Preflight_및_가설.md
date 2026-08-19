# Guidance Selector 사전점검 및 가설

## 결론

Guidance Selector v1은 Pure 0815 BT를 기준으로 동결한다. Dynamic-Limit 후보는 현재 저장소와 로컬 profile에서 DLL/XML pair 및 독립 paired Damage 근거를 함께 고정할 수 없어 기준 후보에서 제외했다. Pure 0815는 현재 파일의 SHA256을 재계산했고 60-frame raw-BT load smoke를 완료했다.

## 기준선

- 시작 `main`: `f0e79b743d7d03870cc485075daa08f4bcd57db6`
- Issue: `#19`
- branch: `codex/guidance-selector-submission-hybrid`
- DLL: `C:/Users/shy66/Downloads/aip_final_0815/aip_final_0815/AIP_DCS_GDCC_0815.dll`
- DLL SHA256: `4C93B4C6719CB0423388D5FC721D356020A3A36CD5AD2C56B5C3CA795BFE18C9`
- XML: `C:/Users/shy66/Downloads/aip_final_0815/aip_final_0815/Rule_DCS_GDCC_0815.xml`
- XML SHA256: `D84C27B0B8BA22E1649AF2375BE0B83C762BC62EB5047BB590539B374F8271EE`
- rule alias: `Rule_DCS_GDCC_0815.xml`
- throttle: raw Pure BT only

60-frame smoke는 seed 8101, lateral-left, autopilot 상대에서 59 step을 정상 완료했다. crash, invalid action, process error는 없었고 종료는 1초 제한 timeout이었다. simulator가 `aircraft/f16/f16_init.xml`을 변경할 수 있으므로 모든 후속 evaluator는 실행 전 bytes를 보존하고 `finally`에서 exact restore한다.

## 이전 residual 실패 근거

상태 조건부 surface residual 72 clean pair의 Damage delta는 평균 `-0.0005014775`, median `0`, positive ratio `31/72`였다. 의미 있는 world geometry는 `1/6`, large regression은 9건이었다. 따라서 raw surface residual을 다시 주 경로로 사용하지 않는다.

## 동결 가설과 계약

Pure BT는 전술, 안전, 기본 guidance, 매 frame control, throttle을 유지한다. selector는 frozen Gate 안에서만 9개 categorical Guidance primitive를 고른다. Guidance composer는 azimuth/elevation/range/target-speed setpoint를 보정하고, bounded controller가 이를 surface command로 변환한다. `BT_DEFAULT`, Gate OFF, 낮은 confidence, invalid/nonfinite/exception/timeout은 exact Pure BT다.

- observation: `guidance_selector_v1`, 32D float32, `[-1, 1]`
- angular magnitude: `0.5 deg`
- range magnitude: `50 m`
- target-speed magnitude: `10 m/s`
- counterfactual horizon: `2.0 s`
- Gate: `rear120 AND (offensive OR phase-aware pre-aim) AND NOT safety veto`
- selector cadence: 10Hz, minimum hold 18 frame, maximum active 90 frame, cooldown 30 frame
- confidence development candidates: `0.55, 0.60, 0.65, 0.70`
- primary metric: clean paired `candidate Damage - Pure BT Damage`

## Seed 분리

- counterfactual state: `8601~8700`
- BC initialization: `8701, 8702, 8703`
- development: `8801~8806`
- held-out: `8901~8906`

held-out 결과를 본 뒤 observation, action library, magnitude, Gate, reward, confidence threshold를 변경하지 않는다. Counterfactual 신호가 부족해도 filtered BC 또는 rule-distilled safe selector로 loadable artifact를 만들지만 성능 승격은 강제하지 않는다.
