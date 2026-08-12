import os
import sys
import numpy as np
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
import FighterSim
import JSBSimWrapper

battle_space_id = JSBSimWrapper.create_battleSpace()

def measure_turn_rate(speed_ms):
    init_state = [1, 1, 0.0, 0.0, -7000.0, 90.0, 0.0, 0.0, speed_ms]
    sim = FighterSim.JSBSim(init_state, None, 60, battle_space_id)
    
    max_turn_rate = 0.0
    
    for step in range(180):
        action = np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float32)
        sim.step(action)
        
        state = sim.get_state()
        heading_current = state[5]
        if step > 60:
            if 'heading_prev' in locals():
                diff = heading_current - heading_prev
                if diff < -180: diff += 360
                if diff > 180: diff -= 360
                turn_rate = abs(diff) * 60.0 # deg/sec
                if turn_rate > max_turn_rate:
                    max_turn_rate = turn_rate
            heading_prev = heading_current
            
    return max_turn_rate

if __name__ == "__main__":
    KTS_TO_MS = 0.514444
    MS_TO_KTS = 1.94384
    
    print("=== F-16 Turn Rate Analysis (Empirical Test in JSBSim at 7000m Altitude) ===\n")
    
    # 1. Broad Scan around Corner Speed envelope (350 to 530 Knots)
    print("## 1. 주요 고속 선회 영역 속도별 선회율 (Coarse Scan: 10노트 간격)")
    print("| 속도 (Knots TAS) | 속도 (m/s) | 최대 선회율 (deg/s) |")
    print("| :---: | :---: | :---: |")
    
    coarse_speeds_kts = range(350, 540, 10)
    best_rate_coarse = 0
    best_speed_coarse_kts = 460
    
    for kts in coarse_speeds_kts:
        ms = kts * KTS_TO_MS
        rate = measure_turn_rate(ms)
        print(f"| {kts:16} | {ms:10.1f} | {rate:19.2f} |")
        if rate > best_rate_coarse:
            best_rate_coarse = rate
            best_speed_coarse_kts = kts
            
    print("\n")
    
    # 2. Fine Scan around optimal (best_speed_coarse_kts +/- 25 knots)
    fine_start_kts = best_speed_coarse_kts - 25
    fine_end_kts = best_speed_coarse_kts + 26
    
    print(f"## 2. 최적 선회속도 부근 정밀 분석 (1노트 단위: {fine_start_kts} ~ {fine_end_kts - 1} Knots)")
    print("| 속도 (Knots TAS) | 속도 (m/s) | 최대 선회율 (deg/s) | 비고 |")
    print("| :---: | :---: | :---: | :---: |")
    
    best_rate_fine = 0
    best_speed_fine_kts = fine_start_kts
    best_speed_fine_ms = 0
    
    results = []
    for kts in range(fine_start_kts, fine_end_kts):
        ms = kts * KTS_TO_MS
        rate = measure_turn_rate(ms)
        results.append((kts, ms, rate))
        if rate > best_rate_fine:
            best_rate_fine = rate
            best_speed_fine_kts = kts
            best_speed_fine_ms = ms
            
    for kts, ms, rate in results:
        note = "**[최적 코너 스피드]**" if kts == best_speed_fine_kts else ""
        print(f"| {kts:16} | {ms:10.1f} | {rate:19.2f} | {note} |")
        
    print("\n---")
    print(f"### 최종 분석 결과")
    print(f"- **최적 코너 스피드 (Optimal Corner Speed):** **{best_speed_fine_kts} Knots TAS** ({best_speed_fine_ms:.1f} m/s)")
    print(f"- **최대 순간 선회율 (Max Instantaneous Turn Rate):** **{best_rate_fine:.2f} deg/sec**")
