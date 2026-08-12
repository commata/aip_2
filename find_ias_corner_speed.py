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
    best_kcas = 0.0
    
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
                    best_kcas = state[12]
            heading_prev = heading_current
            
    return max_turn_rate, best_kcas

if __name__ == "__main__":
    KTS_TO_MS = 0.514444
    MS_TO_KTS = 1.94384
    
    print("=== F-16 Optimal Corner Speed Scan (Aerodynamic Combat Envelope: 350 ~ 550 Knots TAS) ===\n")
    print("| 고도 (m) | 고도 (ft) | 최적 TAS (Knots) | 최적 TAS (m/s) | 최적 IAS/KCAS (Knots) | 최대 선회율 (deg/s) |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for alt in [1000, 3000, 5000, 7000, 10000, 15000, 20000]:
        best_rate = 0
        best_tas_kts = 0
        best_tas_ms = 0
        best_ias_kts = 0
        
        # Scan normal combat speeds 350 to 550 knots TAS
        for tas_kts in range(350, 560, 10):
            ms = tas_kts * KTS_TO_MS
            rate, kcas_ms = measure_turn_rate_at_alt(alt, ms)
            if rate > best_rate:
                best_rate = rate
                best_tas_kts = tas_kts
                best_tas_ms = ms
                best_ias_kts = kcas_ms * MS_TO_KTS
                
        print(f"| {alt:8.0f} | {alt*3.28084:9.0f} | {best_tas_kts:16} | {best_tas_ms:14.1f} | {best_ias_kts:14.1f} kts ({kcas_ms:4.1f} m/s) | {best_rate:19.2f} |")
