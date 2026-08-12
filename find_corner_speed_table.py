import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
import FighterSim
import JSBSimWrapper

battle_space_id = JSBSimWrapper.create_battleSpace()

def measure_turn_rate_at_alt(alt_meters, speed_ms):
    init_state = [1, 1, 0.0, 0.0, -alt_meters, 90.0, 0.0, 0.0, speed_ms]
    sim = FighterSim.JSBSim(init_state, None, 60, battle_space_id)
    
    max_turn_rate = 0.0
    actual_tas_at_max = speed_ms
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
                turn_rate = abs(diff) * 60.0
                if turn_rate > max_turn_rate:
                    max_turn_rate = turn_rate
                    actual_tas_at_max = state[8]
            heading_prev = heading_current
    return max_turn_rate, actual_tas_at_max

if __name__ == "__main__":
    KTS_TO_MS = 0.514444
    alts = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
    
    print("=== F-16 Optimal Corner Speed Scan (0 ~ 12,000m / ~40,000ft) ===\n")
    print("| 고도 (m) | 고도 (ft) | 최적 TAS (Knots) | 최적 TAS (m/s) | 최대 선회율 (deg/s) |")
    print("| :---: | :---: | :---: | :---: | :---: |")
    
    table_cpp = []
    for alt in alts:
        best_rate = 0
        best_tas_kts = 0
        best_tas_ms = 0
        best_actual_tas_ms = 0
        for tas_kts in range(380, 530, 5):
            ms = tas_kts * KTS_TO_MS
            rate, actual_tas = measure_turn_rate_at_alt(alt, ms)
            if rate > best_rate:
                best_rate = rate
                best_tas_kts = tas_kts
                best_tas_ms = ms
                best_actual_tas_ms = actual_tas
        print(f"| {alt} | {int(alt*3.28084)} | 진입 {best_tas_kts} Kts ({best_tas_ms:.1f} m/s) -> 선회 정점 실제 속도 {best_actual_tas_ms:.1f} m/s | {best_rate:.2f} |")
        table_cpp.append((alt, best_actual_tas_ms))
        
    print("\n// C++ Lookup Table:")
    for alt, ms in table_cpp:
        print(f"  {{ {alt}.0f, {ms:.1f}f }},")
