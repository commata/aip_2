"""
F-16A(JSBSim) Instantaneous & Sustained Turn Rate Measurement v2
================================================================
v1 문제점:
  - 수평 직진에서 뱅크 진입 시 FDM 과도 응답으로 비현실적 G값(23.9G 등) 측정
  - cp949 인코딩 에러 (em-dash)

v2 개선:
  1. 초기 뱅크각 90도로 시작 (기존 검증된 방식) → 안정적 FDM
  2. 순간선회: heading 변화율로 측정 (첫 1~3초의 PEAK)
  3. 지속선회: 30초 비행 후 speed 수렴 시점의 heading 변화율
  4. Nz는 검증용으로만 기록 (9G 초과 시 FDM 이상으로 표기)
  5. 모든 문자열 ASCII 호환으로 변경
"""
import os
import sys
import numpy as np
import math
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
import FighterSim
import JSBSimWrapper

# Constants
G = 9.80665
KTS_TO_MS = 0.514444
MS_TO_KTS = 1.94384
SIM_HZ = 60

battle_space_id = JSBSimWrapper.create_battleSpace()


def heading_diff(h1, h2):
    """heading 차이 계산 (-180~180 범위)"""
    d = h2 - h1
    if d > 180: d -= 360
    if d < -180: d += 360
    return d


def measure_instantaneous(alt_m, tas_ms):
    """순간선회율: 풀G 풀러에서 첫 1~3초 내 최대 heading 변화율
    
    90도 뱅크 상태로 초기화 → pitch=-1.0(풀G), throttle=1.0
    heading 변화로 turn rate 측정 (0.5초 이동평균)
    """
    # init: [type, side, N, E, D, Roll, Pitch, Heading, Speed]
    init_state = [1, 1, 0.0, 0.0, -alt_m, 89.999, 0.0, 90.0, tas_ms]
    sim = FighterSim.JSBSim(init_state, None, SIM_HZ, battle_space_id)
    
    # Warmup 30 steps (0.5s) - FDM 안정화
    for _ in range(30):
        action = np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float32)
        sim.step(action)
    
    max_turn_rate = 0.0
    nz_at_max = 0.0
    tas_at_max = tas_ms
    kcas_at_max = 0.0
    heading_history = []
    
    # 3초(180 steps) 풀G 선회 측정
    for step in range(180):
        action = np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float32)
        sim.step(action)
        state = sim.get_state()
        
        heading_history.append(state[5])
        ktas = state[27]
        nz = abs(state[31])
        kcas = state[12]
        
        # 30 steps(0.5s) 이동평균으로 turn rate 측정 (노이즈 제거)
        if len(heading_history) >= 30:
            h_old = heading_history[-30]
            h_new = heading_history[-1]
            dt = 30.0 / SIM_HZ  # 0.5 sec
            tr = abs(heading_diff(h_old, h_new)) / dt
            
            if tr > max_turn_rate:
                max_turn_rate = tr
                nz_at_max = nz
                tas_at_max = ktas
                kcas_at_max = kcas
    
    return {
        'tr': max_turn_rate,
        'nz': nz_at_max,
        'tas_ms': tas_at_max,
        'kcas_ms': kcas_at_max,
    }


def measure_sustained(alt_m, tas_ms, duration_s=30.0):
    """지속선회율: 풀G+풀스로틀에서 속도가 수렴된 시점의 turn rate
    
    수렴 기준: 최근 3초간 TAS 변화 < 1.0 m/s
    """
    init_state = [1, 1, 0.0, 0.0, -alt_m, 89.999, 0.0, 90.0, tas_ms]
    sim = FighterSim.JSBSim(init_state, None, SIM_HZ, battle_space_id)
    
    # Warmup
    for _ in range(30):
        action = np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float32)
        sim.step(action)
    
    total_steps = int(duration_s * SIM_HZ)
    heading_history = []
    speed_history = []
    
    converged_tr = 0.0
    converged_nz = 0.0
    converged_tas = 0.0
    converged_kcas = 0.0
    found = False
    
    for step in range(total_steps):
        action = np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float32)
        sim.step(action)
        state = sim.get_state()
        
        heading_history.append(state[5])
        ktas = state[27]
        speed_history.append(ktas)
        nz = abs(state[31])
        kcas = state[12]
        
        # 최소 5초 경과 후부터 수렴 체크 (300 steps)
        if len(speed_history) >= 180 and len(heading_history) >= 30:
            # 최근 3초(180 steps) 속도 변화
            dv = abs(speed_history[-1] - speed_history[-180])
            
            if dv < 1.0 and not found:
                # 수렴! 현재 turn rate 측정
                h_old = heading_history[-30]
                h_new = heading_history[-1]
                dt = 30.0 / SIM_HZ
                tr = abs(heading_diff(h_old, h_new)) / dt
                
                converged_tr = tr
                converged_nz = nz
                converged_tas = ktas
                converged_kcas = kcas
                found = True
    
    # 수렴 못찾으면 마지막 값 사용
    if not found and len(heading_history) >= 30:
        h_old = heading_history[-30]
        h_new = heading_history[-1]
        dt = 30.0 / SIM_HZ
        converged_tr = abs(heading_diff(h_old, h_new)) / dt
        state = sim.get_state()
        converged_nz = abs(state[31])
        converged_tas = state[27]
        converged_kcas = state[12]
    
    return {
        'tr': converged_tr,
        'nz': converged_nz,
        'tas_ms': converged_tas,
        'kcas_ms': converged_kcas,
        'converged': found,
        'final_tas_ms': speed_history[-1] if speed_history else 0,
    }


if __name__ == "__main__":
    test_alts = [1000, 3000, 5000, 7000, 10000]
    
    print("=" * 100)
    print("F-16A(JSBSim) Instantaneous / Sustained Turn Rate Measurement v2")
    print("=" * 100)
    print()
    print("Method:")
    print("  Instantaneous: 90deg bank + full G pull -> peak heading rate (first 3 sec)")
    print("  Sustained: 90deg bank + full G + full throttle -> converged heading rate (30 sec)")
    print("  Convergence: |dV| < 1.0 m/s over 3 seconds")
    print()
    
    # ===== Phase 1: Coarse Scan (10 kts) =====
    print("-" * 100)
    print("Phase 1: Coarse Scan (10 kts step, 300~550 kts TAS)")
    print("-" * 100)
    
    coarse_results = {}
    
    for alt in test_alts:
        print(f"\n>>> Alt {alt}m ({alt * 3.28084:.0f} ft)...")
        
        best_inst = {'tr': 0.0}
        best_inst_kts = 0
        best_sus = {'tr': 0.0}
        best_sus_kts = 0
        
        detail_rows = []
        
        for tas_kts in range(300, 551, 10):
            tas_ms = tas_kts * KTS_TO_MS
            
            inst = measure_instantaneous(alt, tas_ms)
            sus = measure_sustained(alt, tas_ms, duration_s=25.0)
            
            detail_rows.append((tas_kts, inst, sus))
            
            if inst['tr'] > best_inst['tr']:
                best_inst = inst
                best_inst_kts = tas_kts
            if sus['tr'] > best_sus['tr']:
                best_sus = sus
                best_sus_kts = tas_kts
        
        coarse_results[alt] = {
            'best_inst_kts': best_inst_kts, 'best_inst': best_inst,
            'best_sus_kts': best_sus_kts, 'best_sus': best_sus,
            'details': detail_rows,
        }
        
        # Print detail table for this altitude
        print(f"  | TAS(kts) | Inst TR(deg/s) | Inst Nz(G) | Actual TAS(m/s) | Sus TR(deg/s) | Sus Nz(G) | Sus TAS(m/s) | Converged |")
        print(f"  |" + ":---:|" * 8)
        for kts, inst, sus in detail_rows:
            mark_i = " **" if kts == best_inst_kts else ""
            mark_s = " **" if kts == best_sus_kts else ""
            print(f"  | {kts:3} | {inst['tr']:6.2f}{mark_i} | {inst['nz']:4.1f} | {inst['tas_ms']:6.1f} "
                  f"| {sus['tr']:6.2f}{mark_s} | {sus['nz']:4.1f} | {sus['tas_ms']:6.1f} | {'Y' if sus['converged'] else 'N'} |")
        
        print(f"\n  [BEST] Instantaneous: {best_inst_kts} kts -> {best_inst['tr']:.2f} deg/s @ {best_inst['nz']:.1f}G (actual TAS {best_inst['tas_ms']:.1f} m/s)")
        print(f"  [BEST] Sustained:     {best_sus_kts} kts -> {best_sus['tr']:.2f} deg/s @ {best_sus['nz']:.1f}G (actual TAS {best_sus['tas_ms']:.1f} m/s, conv={'Y' if best_sus['converged'] else 'N'})")
    
    # ===== Phase 2: Fine Scan (5 kts) =====
    print()
    print("-" * 100)
    print("Phase 2: Fine Scan (5 kts step, around coarse optima +/- 30 kts)")
    print("-" * 100)
    
    fine_results = {}
    
    for alt in test_alts:
        coarse = coarse_results[alt]
        
        inst_center = coarse['best_inst_kts']
        sus_center = coarse['best_sus_kts']
        
        # Merge scan ranges
        scan_min = max(250, min(inst_center, sus_center) - 30)
        scan_max = min(600, max(inst_center, sus_center) + 30)
        
        print(f"\n>>> Alt {alt}m - Fine scan: {scan_min}~{scan_max} kts")
        
        best_inst = {'tr': 0.0}
        best_inst_kts = 0
        best_sus = {'tr': 0.0}
        best_sus_kts = 0
        
        for tas_kts in range(scan_min, scan_max + 1, 5):
            tas_ms = tas_kts * KTS_TO_MS
            inst = measure_instantaneous(alt, tas_ms)
            sus = measure_sustained(alt, tas_ms, duration_s=25.0)
            
            if inst['tr'] > best_inst['tr']:
                best_inst = inst
                best_inst_kts = tas_kts
            if sus['tr'] > best_sus['tr']:
                best_sus = sus
                best_sus_kts = tas_kts
        
        fine_results[alt] = {
            'best_inst_kts': best_inst_kts, 'best_inst': best_inst,
            'best_sus_kts': best_sus_kts, 'best_sus': best_sus,
        }
        
        print(f"  [FINE] Instantaneous: {best_inst_kts} kts TAS -> TR = {best_inst['tr']:.2f} deg/s @ {best_inst['nz']:.1f}G")
        print(f"         actual TAS = {best_inst['tas_ms']:.1f} m/s, KCAS = {best_inst['kcas_ms'] * MS_TO_KTS:.1f} kts")
        print(f"  [FINE] Sustained:     {best_sus_kts} kts TAS -> TR = {best_sus['tr']:.2f} deg/s @ {best_sus['nz']:.1f}G")
        print(f"         actual TAS = {best_sus['tas_ms']:.1f} m/s, KCAS = {best_sus['kcas_ms'] * MS_TO_KTS:.1f} kts, conv={'Y' if best_sus['converged'] else 'N'}")
    
    # ===== Final Summary =====
    print()
    print("=" * 120)
    print("FINAL RESULTS TABLE")
    print("=" * 120)
    print()
    print("| Alt(m) | Alt(ft) | Inst Opt TAS(kts) | Inst TAS(m/s) | Inst TR(deg/s) | Inst Nz(G) |"
          " Sus Opt TAS(kts) | Sus TAS(m/s) | Sus TR(deg/s) | Sus Nz(G) |")
    print("|" + ":---:|" * 10)
    
    for alt in test_alts:
        r = fine_results[alt]
        i = r['best_inst']
        s = r['best_sus']
        print(f"| {alt:5} | {alt*3.28084:7.0f} "
              f"| {r['best_inst_kts']:4} "
              f"| {i['tas_ms']:6.1f} "
              f"| {i['tr']:6.2f} "
              f"| {i['nz']:4.1f} "
              f"| {r['best_sus_kts']:4} "
              f"| {s['tas_ms']:6.1f} "
              f"| {s['tr']:6.2f} "
              f"| {s['nz']:4.1f} |")
    
    # ===== C++ Lookup Table =====
    print()
    print("// === C++ Lookup Table for Task_Pursue.cpp ===")
    print("struct TurnPerfEntry {")
    print("  float altM;         // Altitude (m)")
    print("  float instTAS_MS;   // Instantaneous Turn Optimal TAS (m/s)")
    print("  float instTR_DegS;  // Max Instantaneous Turn Rate (deg/s)")
    print("  float susTAS_MS;    // Sustained Turn Optimal TAS (m/s)")
    print("  float susTR_DegS;   // Max Sustained Turn Rate (deg/s)")
    print("};")
    print("static const TurnPerfEntry TURN_PERF_TABLE[] = {")
    for alt in test_alts:
        r = fine_results[alt]
        i = r['best_inst']
        s = r['best_sus']
        print(f"  {{ {float(alt):.1f}f, {i['tas_ms']:.1f}f, {i['tr']:.2f}f, "
              f"{s['tas_ms']:.1f}f, {s['tr']:.2f}f }},")
    print("};")
    print(f"static const int TURN_PERF_TABLE_SIZE = {len(test_alts)};")
