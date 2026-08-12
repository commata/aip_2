import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
import FighterSim
import JSBSimWrapper

battle_space_id = JSBSimWrapper.create_battleSpace()

def test_speed_at_alt(alt_meters, speed_ms):
    init_state = [1, 1, 0.0, 0.0, -alt_meters, 90.0, 0.0, 0.0, speed_ms]
    sim = FighterSim.JSBSim(init_state, None, 60, battle_space_id)
    
    action = np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float32)
    sim.step(action)
    state = sim.get_state()
    
    kcas_ms = state[12] # StateIndex.KCAS = 12 (in m/s!)
    kcas_kts = kcas_ms * 1.94384
    print(f"Alt: {alt_meters:5.0f}m ({alt_meters*3.28084:6.0f}ft) | TAS: {speed_ms:6.1f} m/s ({speed_ms*1.94384:5.1f} kts) | KCAS (IAS): {kcas_kts:6.1f} kts ({kcas_ms:5.1f} m/s) | Ratio TAS/IAS: {(speed_ms*1.94384)/max(1, kcas_kts):.2f}")

if __name__ == "__main__":
    print("=== Checking KCAS (IAS) vs TAS in JSBSim at different altitudes ===")
    for alt in [1000, 3000, 5000, 7000, 9000]:
        test_speed_at_alt(alt, 247.4)
