import os
import sys
import numpy as np
import math

# Add src to path to import FighterSim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
import FighterSim
import JSBSimWrapper

battle_space_id = JSBSimWrapper.create_battleSpace()

def measure_turn_rate(speed_ms):
    # Initialize JSBSim F-16 at 7000m altitude
    # [is_ownship, id, north, east, down, roll, pitch, heading, speed]
    # We start banked at 90 degrees to perform a horizontal turn
    init_state = [1, 1, 0.0, 0.0, -7000.0, 90.0, 0.0, 0.0, speed_ms]
    
    # Create simulator with no AI (we will override controls)
    sim = FighterSim.JSBSim(init_state, None, 60, battle_space_id)
    
    max_turn_rate = 0.0
    
    # Warm up for 1 second to let dynamics settle, then measure 2 seconds
    for step in range(180):
        # Action: [Roll, Pitch, Rudder, Throttle]
        # We command Max Pull (Pitch = -1.0) and Full Throttle (Throttle = 1.0)
        # Roll = 0.0 to hold the current 90 deg bank
        action = np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float32)
        sim.step(action)
        
        state = sim.get_state()
        # state is a list of floats. We need the turn rate.
        # Since we are banked 90 deg, the pitch rate (q) is approximately the turn rate.
        # Let's extract velocity vector or just use the JSBSim state properties.
        # JSBSim GetState() format usually: [N, E, D, Roll, Pitch, Heading, u, v, w, p, q, r, alpha, beta, mach]
        # Assuming index 5 is Heading (deg). Let's just track heading change per step.
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
    print("Method 2: Empirical Test in FighterSim")
    print("Altitude: 7000m. Commanding Max-G (PitchCMD = -1.0) at various initial speeds.")
    print("Speed (m/s) | Speed (knots) | Max Turn Rate (deg/sec)")
    print("-" * 55)
    
    best_speed = 0
    best_rate = 0
    
    # Test True Airspeeds from 150 m/s to 350 m/s (approx 300 to 680 knots TAS)
    for speed_ms in range(150, 360, 10):
        rate = measure_turn_rate(speed_ms)
        speed_kts = speed_ms * 1.94384
        print(f"{speed_ms:9} | {speed_kts:13.1f} | {rate:.2f}")
        
        if rate > best_rate:
            best_rate = rate
            best_speed = speed_ms
            
    print("-" * 55)
    print(f"Optimal Corner Speed (TAS) at 7000m: {best_speed} m/s ({best_speed*1.94384:.1f} knots)")
    print(f"Maximum Instantaneous Turn Rate: {best_rate:.2f} deg/sec")
