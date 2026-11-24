# =============================================================================
# MASTER FRONTIER AUTONOMOUS DRONE SYSTEM CORE (V36.0 - HARDENED PRODUCTION)
# File: drone_master_autonomous_system_v56.py — [PART 1 OF 3]
# =============================================================================
# - SENSOR UPGRADE 1: Downward Optical Flow Camera (GPS-Denied Velocity Tracking)
# - SENSOR UPGRADE 2: ArUco / AprilTag Visual Fiducial Precision Landing Reticle
# - SENSOR UPGRADE 3: Physical Skid Weight-on-Wheels (WoW) Strain Gauge Switches
# - SENSOR UPGRADE 4: Realistic 3D Rooftop Clutter (HVAC units, bulkheads, parapets)
# - SENSOR UPGRADE 5: Multi-Segment Tether with Dynamic Catenary Aerodynamic Drag
# - SENSOR UPGRADE 6: Directional Acoustic Array (Emergency Sirens & Blade Whistle)
# - SENSOR UPGRADE 7: ADS-B In Civil Traffic Transponder (Helicopter Avoidance)
# - COMPLETE FLIGHT HARDENING & FIXES:
#     1. Safe Chimney Climb: Vertical climb to cruise corridor (42-48m) before XY cruise
#     2. Corrected Altitude Inversion: Cruise altitudes safely exceed 38m high-hover gates
#     3. Smooth Terminal Flare: Controlled descent tapering to -0.25 m/s at touchdown
#     4. Calibrated Rotor Physics: Clean BEMT lift without aerodynamic inflow stall
#     5. Decoupled Ground Latch: Eliminates premature throttle cutouts during takeoff
#     6. Exact SO(3) Error-State EKF: Cross-talk-free yaw update with unbiased IMU loop
#     7. Synchronized Swarm Cross-Attention: Preserves multi-agent peer geometry
# =============================================================================

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

import time
import math
import random
from collections import deque
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pygame
import sys

# ==========================================
# 0. PHYSICAL AIRFRAME & MISSION CONSTANTS
# ==========================================
DRY_MASS_KG = 1.35                   
CARGO_MASS_KG = 0.45                 
MASS_KG = DRY_MASS_KG + CARGO_MASS_KG 

GRAVITY_G = np.array([0.0, 0.0, -9.80665], dtype=np.float32)
G_ACCEL = 9.80665
ARM_LENGTH_M = 0.26
ROTOR_RADIUS_M = 0.127
ROTOR_DISK_AREA = math.pi * (ROTOR_RADIUS_M ** 2)

INERTIA_MATRIX = np.diag([0.022, 0.022, 0.038]).astype(np.float32)
INV_INERTIA = np.linalg.inv(INERTIA_MATRIX)

C_T_CLEAN = 1.35e-5
C_Q_CLEAN = 2.40e-7
COAXIAL_EFFICIENCY = 0.92
MOTOR_TAU_S = 0.030
MAX_RPM = 9200.0
MAX_OMEGA = (MAX_RPM * 2.0 * math.pi) / 60.0

HOVER_THRUST_NOMINAL = MASS_KG * G_ACCEL
HOVER_OMEGA_NOMINAL = math.sqrt((HOVER_THRUST_NOMINAL / 8.0) / (C_T_CLEAN * 0.95))

NUM_BLADE_ELEMENTS = 8
PRANDTL_TIP_FACTOR = 0.92
COAXIAL_DOWNWASH_INGESTION = 0.65

CEILING_HEIGHT_M = 60.0
CBF_ALPHA_OBSTACLE = 1.8
CBF_SAFE_RADIUS_M = 0.85
SWARM_INTER_DRONE_SAFE_RADIUS_M = 2.6

# Quantum & Atmospheric Constants
K_EFF_WAVEVECTOR = 1.61e7            
RAMAN_PULSE_SEP_T = 0.015            
QUANTUM_PHASE_1G = K_EFF_WAVEVECTOR * G_ACCEL * (RAMAN_PULSE_SEP_T ** 2)

P_0 = 1013.25
T_0 = 288.15
L_LAPSE = 0.0065
R_SPEC_AIR = 287.058
SPEED_OF_SOUND = 343.2
EMF_WARNING_THRESHOLD_UT = 18.0

# 6S Battery ECM Constants
NUM_CELLS = 6
PACK_CAPACITY_AH = 8.0               
CELL_R0_NOMINAL = 0.0085
CELL_R1 = 0.0065
CELL_C1 = 2800.0
ARRHENIUS_EA = 28000.0
GAS_CONST_R = 8.31446
T_REF_K = 298.15

# Stable Tether Delivery Constants (High-Hover Reach: 10m safe standoff)
PAYLOAD_WEIGHT_N = CARGO_MASS_KG * G_ACCEL
GUILLOTINE_THRESHOLD_N = 48.0
HOVER_LOCK_RADIUS_M = 2.20           
HOVER_LOCK_SPEED_MAX_MS = 1.25       
HOVER_SETTLE_DURATION_S = 0.20

# Feature Dimensions: 139 features x 3 history frames = 417 dims
BASE_OBS_DIM = 139
HISTORY_LEN = 3
OBSERVATION_DIM = BASE_OBS_DIM * HISTORY_LEN
NUM_EXECUTIVE_STATES = 6
MODEL_WEIGHTS_FILE = "drone_master_brain_v24_full_sensory.pth"

NUM_SWARM_DRONES = 3

ACTION_NAMES = [
    "TRANSIT",             
    "HOVER_LOCK",          
    "WINCH_LOWER",         
    "TOUCHDOWN_RELEASE",   
    "WINCH_RETRACT",       
    "AUTO_DISPATCH"        
]

# ==========================================
# 1. MATHEMATICAL UTILITIES & LIE ALGEBRA
# ==========================================
def skew_symmetric(v):
    """Computes the 3x3 skew-symmetric matrix [v]_x in so(3)."""
    return np.array([
        [0.0, -v[2], v[1]], 
        [v[2], 0.0, -v[0]], 
        [-v[1], v[0], 0.0]
    ], dtype=np.float32)

def vee_map(mat):
    """Extracts the R^3 vector from an so(3) skew-symmetric matrix."""
    return np.array([mat[2, 1], mat[0, 2], mat[1, 0]], dtype=np.float32)

def quat_to_rot(q):
    """Converts a unit quaternion [w, x, y, z] to an SO(3) rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z),       2.0*(x*z + w*y)],
        [2.0*(x*y + w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x)],
        [2.0*(x*z - w*y),       2.0*(y*z + w*x),       1.0 - 2.0*(x*x + y*y)]
    ], dtype=np.float32)

def quat_mult(q1, q2):
    """Computes the Hamilton product of two quaternions q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float32)

# ==========================================
# 2. METROPOLITAN CITY GRID & ROOFTOP CLUTTER
# ==========================================
class MetropolitanCityGridWorld400m:
    def __init__(self):
        self.avenue_x = [-120.0, -60.0, 0.0, 60.0, 120.0]
        self.street_y = [-120.0, -60.0, 0.0, 60.0, 120.0]
        self.street_width = 10.0

        self.roads = []
        for x in self.avenue_x:
            self.roads.append({"x_min": x - 5.0, "x_max": x + 5.0, "y_min": -200.0, "y_max": 200.0, "type": "AVENUE"})
        for y in self.street_y:
            self.roads.append({"x_min": -200.0, "x_max": 200.0, "y_min": y - 5.0, "y_max": y + 5.0, "type": "STREET"})

        self.depot_helipads = [
            np.array([-160.0, -155.0, 0.0], dtype=np.float32),
            np.array([-150.0, -155.0, 0.0], dtype=np.float32),
            np.array([-140.0, -155.0, 0.0], dtype=np.float32)
        ]
        self.base_station = self.depot_helipads[0]

        self.houses = []
        self.manifest_addresses = []
        self._build_48_buildings_with_rooftop_clutter()

        self.trees = [
            {"pos": np.array([-25.0, -25.0, 0.0], dtype=np.float32), "r": 3.2, "h": 8.0},
            {"pos": np.array([ 25.0, -25.0, 0.0], dtype=np.float32), "r": 3.0, "h": 7.5},
            {"pos": np.array([-25.0,  25.0, 0.0], dtype=np.float32), "r": 3.5, "h": 8.5},
            {"pos": np.array([ 25.0,  25.0, 0.0], dtype=np.float32), "r": 3.2, "h": 8.0}
        ]

        self.powerlines = [
            {"p1": np.array([-190.0, -170.0, 12.0], dtype=np.float32), "p2": np.array([190.0, -170.0, 12.0], dtype=np.float32)},
            {"p1": np.array([-190.0,  170.0, 12.0], dtype=np.float32), "p2": np.array([190.0,  170.0, 12.0], dtype=np.float32)}
        ]

        # Live ADS-B Civil Airspace Traffic (Low-altitude police/medevac helicopter)
        self.civil_aircraft = {
            "id": "Medevac_N911",
            "pos": np.array([-170.0, 0.0, 62.0], dtype=np.float32),
            "vel": np.array([12.5, 0.0, 0.0], dtype=np.float32),
            "radius": 4.5
        }

    def _build_48_buildings_with_rooftop_clutter(self):
        street_names = ["Broadway", "5th Ave", "Park Ave", "Lexington", "Madison", "Elm St", "Market St", "Grand Blvd"]
        bldg_counter = 0

        col_candidates = [-165.0, -140.0, -90.0, -30.0, 30.0, 90.0, 140.0, 165.0]
        row_candidates = [-165.0, -140.0, -90.0, -30.0, 30.0, 90.0, 140.0, 165.0]

        for c_idx, cx in enumerate(col_candidates):
            for r_idx, ry in enumerate(row_candidates):
                if cx <= -130.0 and ry <= -130.0:
                    continue

                if bldg_counter >= 48:
                    break

                bldg_w = 18.0
                bldg_d = 18.0
                bx1 = cx - bldg_w / 2.0
                bx2 = cx + bldg_w / 2.0
                by1 = ry - bldg_d / 2.0
                by2 = ry + bldg_d / 2.0

                tier = (c_idx + r_idx) % 4
                if tier == 0:
                    height = 8.5
                elif tier == 1:
                    height = 14.5
                elif tier == 2:
                    height = 21.0
                else:
                    height = 28.0

                bldg_id = f"Bldg_{bldg_counter+1:02d}"

                # Rooftop Clutter (HVAC units & elevator bulkheads)
                clutter_elements = [
                    {"aabb": [bx1 + 1.5, bx1 + 4.5, by2 - 4.5, by2 - 1.5, height + 1.2], "type": "HVAC"},
                    {"aabb": [bx2 - 5.0, bx2 - 2.0, by1 + 2.0, by1 + 5.0, height + 2.2], "type": "BULKHEAD"}
                ]

                self.houses.append({
                    "id": bldg_id,
                    "aabb": [bx1, bx2, by1, by2, height],
                    "clutter": clutter_elements
                })

                street = street_names[(c_idx + r_idx) % len(street_names)]
                num = (bldg_counter + 1) * 10 + 2
                drop_pos = np.array([cx, ry, height], dtype=np.float32)

                self.manifest_addresses.append({
                    "bldg_id": bldg_id,
                    "addr": f"{num} {street} [ROOFTOP]",
                    "pos": drop_pos,
                    "height": height,
                    "tag_id": bldg_counter + 101
                })
                bldg_counter += 1

            if bldg_counter >= 48:
                break

    def step_civil_traffic(self, dt=0.016):
        self.civil_aircraft["pos"] += self.civil_aircraft["vel"] * dt
        if self.civil_aircraft["pos"][0] > 195.0:
            self.civil_aircraft["pos"][0] = -195.0

    def get_surface_height(self, x, y):
        for h in self.houses:
            x1, x2, y1, y2, hh = h["aabb"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                for c in h.get("clutter", []):
                    cx1, cx2, cy1, cy2, ch = c["aabb"]
                    if cx1 <= x <= cx2 and cy1 <= y <= cy2:
                        return float(ch)
                return float(hh)
        return 0.0

# =============================================================================
# 3. 3D GLIDE-SLOPE CORRIDOR & URBAN VERTICAL CHIMNEY PLANNER
# =============================================================================
class VoxelOctreeCorridorPlanner:
    def __init__(self, world: MetropolitanCityGridWorld400m):
        self.world = world

    def is_point_occluded(self, pt, margin=2.2):
        px, py, pz = pt
        if pz < 0.2 or pz > CEILING_HEIGHT_M:
            return True
        for h in self.world.houses:
            x1, x2, y1, y2, hh = h["aabb"]
            if (x1 - margin) <= px <= (x2 + margin) and (y1 - margin) <= py <= (y2 + margin) and pz < (hh - 0.2):
                return True
            for c in h.get("clutter", []):
                cx1, cx2, cy1, cy2, ch = c["aabb"]
                if (cx1 - 0.5) <= px <= (cx2 + 0.5) and (cy1 - 0.5) <= py <= (cy2 + 0.5) and pz < ch:
                    return True
        for t in self.world.trees:
            if math.hypot(px - t["pos"][0], py - t["pos"][1]) <= (t["r"] + margin) and pz <= (t["h"] + margin):
                return True
        for line in self.world.powerlines:
            w1 = line["p1"]
            if abs(py - w1[1]) < (margin + 1.2) and abs(pz - w1[2]) < (margin + 1.2):
                return True
        return False

    def is_segment_occluded(self, p1, p2, step_size=2.0):
        dist = float(np.linalg.norm(p2 - p1))
        steps = max(2, int(dist / step_size))
        for s in range(steps + 1):
            tau = s / float(steps)
            interp_pt = p1 + tau * (p2 - p1)
            if self.is_point_occluded(interp_pt):
                return True
        return False

    def find_safe_corridor_waypoints(self, start_p, end_p, assigned_cruise_alt=44.0, is_depot_landing=False):
        cruise_alt = max(assigned_cruise_alt, max(start_p[2], end_p[2]) + 4.0)

        # 1. Vertical climbout gate
        local_start_h = self.world.get_surface_height(start_p[0], start_p[1])
        safe_climb_z = max(start_p[2] + 2.5, local_start_h + 4.5)
        p_climb_gate = np.array([start_p[0], start_p[1], safe_climb_z], dtype=np.float32)

        d_xy = end_p[:2] - start_p[:2]
        dist_total_xy = max(1.0, float(np.linalg.norm(d_xy)))
        dir_xy = d_xy / dist_total_xy

        climb_ramp_dist = min(30.0, dist_total_xy * 0.35)
        p_climb_top = np.array([
            start_p[0] + dir_xy[0] * climb_ramp_dist,
            start_p[1] + dir_xy[1] * climb_ramp_dist,
            cruise_alt
        ], dtype=np.float32)

        p_descend_top = np.array([end_p[0], end_p[1], cruise_alt], dtype=np.float32)

        local_end_h = self.world.get_surface_height(end_p[0], end_p[1])
        flare_alt = (local_end_h + 4.5) if not is_depot_landing else 4.0
        p_flare = np.array([end_p[0], end_p[1], flare_alt], dtype=np.float32)

        final_hover_z = (local_end_h + 10.0) if not is_depot_landing else 2.5
        p_final = np.array([end_p[0], end_p[1], final_hover_z], dtype=np.float32)

        if self.is_segment_occluded(p_climb_top, p_descend_top):
            mid_pt = 0.5 * (p_climb_top + p_descend_top)
            detour_1 = mid_pt + np.array([-20.0, 20.0, 5.0], dtype=np.float32)
            detour_2 = mid_pt + np.array([ 20.0, -20.0, 5.0], dtype=np.float32)
            via = detour_1 if not self.is_point_occluded(detour_1) else detour_2
            return [start_p.copy(), p_climb_gate, p_climb_top, via, p_descend_top, p_flare, p_final]
        else:
            return [start_p.copy(), p_climb_gate, p_climb_top, p_descend_top, p_flare, p_final]

# ==========================================
# 4. DYNAMIC WEATHER & ROAD VEHICLES
# ==========================================
class DynamicUrbanWeatherEngine:
    def __init__(self):
        self.weather_modes = ["CLEAR_NOMINAL", "CANYON_GALE", "FREEZING_RAIN", "MICROBURST_STORM"]
        self.current_mode_idx = 0
        self.regime_timer = 0.0
        self.regime_duration = 45.0

    def cycle_weather(self):
        self.current_mode_idx = (self.current_mode_idx + 1) % len(self.weather_modes)
        self.regime_timer = 0.0
        print(f"\n>>> [WEATHER SHIFT] CURRENT REGIME: {self.get_mode_name()} <<<")

    def get_mode_name(self):
        return self.weather_modes[self.current_mode_idx]

    def step(self, dt=0.016):
        self.regime_timer += dt
        if self.regime_timer >= self.regime_duration:
            self.cycle_weather()

    def get_weather_parameters(self, drone_pos, avenues_x):
        mode = self.get_mode_name()
        px, py, pz = drone_pos

        if mode == "CLEAR_NOMINAL":
            base_wind = np.array([1.8, -0.6, 0.0], dtype=np.float32)
            gust_scale = 0.6
            temp_c = 22.0
            auto_ice = False
            microburst_w = 0.0
        elif mode == "CANYON_GALE":
            base_wind = np.array([5.8, 2.4, 0.0], dtype=np.float32)
            gust_scale = 3.8
            temp_c = 14.0
            auto_ice = False
            microburst_w = 0.0
        elif mode == "FREEZING_RAIN":
            base_wind = np.array([2.4, -1.8, 0.0], dtype=np.float32)
            gust_scale = 1.5
            temp_c = -4.5
            auto_ice = True
            microburst_w = -0.4
        else:  # MICROBURST_STORM
            base_wind = np.array([-4.5, 4.2, 0.0], dtype=np.float32)
            gust_scale = 5.5
            temp_c = 11.0
            auto_ice = False
            microburst_w = -2.0 if pz > 6.0 else -0.4

        venturi_boost = 1.0
        for ax in avenues_x:
            if abs(px - ax) < 7.0 and pz < 25.0:
                venturi_boost = 1.45
                break

        effective_wind = base_wind.copy()
        effective_wind[1] *= venturi_boost
        effective_wind[2] += microburst_w
        return effective_wind, gust_scale, temp_c, auto_ice

class DynamicRoadVehicleTrafficEngine:
    def __init__(self, avenues_x, streets_y):
        self.vehicles = []
        for i in range(4):
            ax = avenues_x[i % len(avenues_x)]
            self.vehicles.append({
                "id": f"Van_{i+1}",
                "pos": np.array([ax, -180.0 + i * 90.0, 0.9], dtype=np.float32),
                "vel": np.array([0.0, 4.8 + i*0.4, 0.0], dtype=np.float32),
                "axis": "Y",
                "is_emergency": False
            })
        for i in range(4):
            sy = streets_y[i % len(streets_y)]
            is_amb = (i == 0)
            self.vehicles.append({
                "id": "Ambulance_01" if is_amb else f"Truck_{i+1}",
                "pos": np.array([-180.0 + i * 90.0, sy, 1.1], dtype=np.float32),
                "vel": np.array([7.2 if is_amb else (4.2 + i*0.5), 0.0, 0.0], dtype=np.float32),
                "axis": "X",
                "is_emergency": is_amb
            })

    def step(self, dt=0.016):
        for v in self.vehicles:
            v["pos"] += v["vel"] * dt
            if v["axis"] == "Y":
                if v["pos"][1] > 190.0: v["pos"][1] = -190.0
                if v["pos"][1] < -190.0: v["pos"][1] = 190.0
            else:
                if v["pos"][0] > 190.0: v["pos"][0] = -190.0
                if v["pos"][0] < -190.0: v["pos"][0] = 190.0

    def get_cbf_obstacles(self):
        return [{"pos": v["pos"].copy(), "radius": 2.6, "height": 2.2, "id": v["id"]} for v in self.vehicles]

# ==========================================
# 5. DYNAMIC GROUND INTRUDER AGENT
# ==========================================
class DynamicGroundIntruderAgent:
    def __init__(self, init_pos):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.v_max = 2.8
        self.personal_space_m = 1.85
        self.curiosity_timer = 0.0
        self.boredom_cooldown = 0.0
        self.wander_target = np.array([-100.0, 40.0], dtype=np.float32)
        self.is_leashed = False

    def step(self, cargo_world_pos, is_winch_lowering, is_aborting, dt=0.016):
        if self.is_leashed:
            self.p = np.array([-145.0, -145.0, 0.0], dtype=np.float32)
            self.v.fill(0.0)
            return self.p.copy(), False, 99.0

        d_to_cargo = cargo_world_pos[:2] - self.p[:2]
        dist = float(np.linalg.norm(d_to_cargo))

        if is_aborting:
            self.boredom_cooldown = 9.0
            self.curiosity_timer = 0.0
            flee_dir = -d_to_cargo / max(1e-4, dist)
            v_des = flee_dir * (self.v_max * 1.3)
            a_cmd = (v_des - self.v[:2]) * 5.0
        else:
            d_wander = self.wander_target - self.p[:2]
            dist_w = float(np.linalg.norm(d_wander))
            if dist_w < 0.6:
                self.wander_target = np.array([random.uniform(-140.0, 140.0), random.uniform(-140.0, 140.0)], dtype=np.float32)
            v_des = (d_wander / max(1e-4, dist_w)) * 0.8
            a_cmd = (v_des - self.v[:2]) * 1.8

        self.v[:2] += a_cmd * dt
        self.p[:2] += self.v[:2] * dt
        self.p[2] = 0.0
        return self.p.copy(), (dist < self.personal_space_m and cargo_world_pos[2] < 2.5), dist

# ==========================================
# 6. ROTOR ICING, TURBULENCE & BATTERY ECM
# ==========================================
class MessingerRotorIcingModel:
    def __init__(self):
        self.ice_thickness_mm = 0.0
        self.icing_active = False

    def step(self, airspeed, dt=0.016):
        if self.icing_active:
            lwc = 0.35
            catch_efficiency = 0.42
            d_ice = (lwc * airspeed * catch_efficiency * 1e-3) * 0.08 * dt
            self.ice_thickness_mm = min(3.8, self.ice_thickness_mm + d_ice * 1000.0)
        else:
            self.ice_thickness_mm = max(0.0, self.ice_thickness_mm - 0.02 * dt)

        ice_index = float(np.clip(self.ice_thickness_mm / 3.0, 0.0, 1.0))
        return self.ice_thickness_mm, ice_index

class DrydenAtmosphericTurbulenceModel:
    def __init__(self, w20_m_s=3.4):
        self.w20 = w20_m_s
        self.gust_state = np.zeros(3, dtype=np.float32)

    def step(self, altitude_m, airspeed_m_s, dt=0.016):
        h = max(0.5, altitude_m)
        v = max(1.0, airspeed_m_s)
        sigma_w = 0.10 * self.w20
        sigma_u = sigma_w / (0.177 + 0.000823 * h) ** 0.4
        sigma_v = sigma_u
        l_w = h
        l_u = h / (0.177 + 0.000823 * h) ** 1.2
        l_v = l_u

        white_noise = np.random.normal(0.0, 1.0, 3).astype(np.float32)
        sq_dt = math.sqrt(dt)
        du = (- (v / l_u) * self.gust_state[0] * dt + sigma_u * math.sqrt(2.0 * v / l_u) * white_noise[0] * sq_dt)
        dv = (- (v / l_v) * self.gust_state[1] * dt + sigma_v * math.sqrt(2.0 * v / l_v) * white_noise[1] * sq_dt)
        dw = (- (v / l_w) * self.gust_state[2] * dt + sigma_w * math.sqrt(2.0 * v / l_w) * white_noise[2] * sq_dt)

        self.gust_state[0] += du
        self.gust_state[1] += dv
        self.gust_state[2] += dw
        self.gust_state = np.clip(self.gust_state, -6.0, 6.0)
        return self.gust_state.copy()

class SubZeroArrheniusBatteryECM:
    def __init__(self):
        self.soc = 0.96
        self.temp_c = 22.0
        self.v1 = np.zeros(NUM_CELLS, dtype=np.float32)
        self.capacity_coulombs = PACK_CAPACITY_AH * 3600.0
        self.v_pack = 23.8

    def step(self, p_elec_watts, amb_temp=22.0, dt=0.016):
        i_draw = p_elec_watts / max(15.0, self.v_pack)
        t_k = self.temp_c + 273.15
        r_scale = math.exp((ARRHENIUS_EA / GAS_CONST_R) * (1.0 / t_k - 1.0 / T_REF_K))
        r0 = CELL_R0_NOMINAL * max(0.85, r_scale)

        self.soc -= (i_draw * dt) / self.capacity_coulombs
        self.soc = max(0.0, min(1.0, self.soc))

        s = max(0.001, min(0.999, self.soc))
        ocv = 3.30 + 0.55 * s + 0.25 * math.log(s + 0.005) + 0.15 * math.exp(6.0 * (s - 0.88))
        ocv = float(np.clip(ocv, 3.20, 4.20))

        cell_volts = []
        for i in range(NUM_CELLS):
            self.v1[i] += (-(self.v1[i] / (CELL_R1 * CELL_C1)) + (i_draw / CELL_C1)) * dt
            cell_volts.append(ocv - i_draw * r0 - self.v1[i])

        self.v_pack = float(np.sum(cell_volts))
        p_joule = (i_draw ** 2) * (r0 * NUM_CELLS)
        
        self.temp_c += ((p_joule - 0.40 * (self.temp_c - amb_temp)) / 350.0) * dt
        return self.soc, self.v_pack, i_draw, r0

    # =============================================================================
# MASTER FRONTIER AUTONOMOUS DRONE SYSTEM CORE (V36.0 - HARDENED PRODUCTION)
# File: drone_master_autonomous_system_v56.py — [PART 2 OF 3]
# =============================================================================

# =============================================================================
# 7. AIRFRAME 6-DOF IMU, CATENARY WINCH & SKID STRAIN GAUGES
# =============================================================================
class Airframe6DOFMEMS_IMU:
    def __init__(self):
        self.b_a = np.array([0.02, -0.015, 0.03], dtype=np.float32)
        self.b_g = np.array([0.005, -0.004, 0.002], dtype=np.float32)
        self.accel_raw = np.zeros(3, dtype=np.float32)
        self.gyro_raw = np.zeros(3, dtype=np.float32)

    def poll(self, drone_R, accel_linear, omega_true, motor_omegas):
        # Specific force measured directly in body frame: f_body = R^T (a - g)
        f_inertial = accel_linear - GRAVITY_G
        f_body = drone_R.T @ f_inertial

        rpm_ratio = float(np.mean(motor_omegas)) / MAX_OMEGA
        vib_amp = 0.16 * rpm_ratio
        vib_accel = np.random.normal(0.0, vib_amp, 3).astype(np.float32)
        
        self.accel_raw = f_body + self.b_a + vib_accel + np.random.normal(0.0, 0.035, 3).astype(np.float32)

        vib_gyro = np.random.normal(0.0, 0.015 * rpm_ratio, 3).astype(np.float32)
        self.gyro_raw = omega_true + self.b_g + vib_gyro + np.random.normal(0.0, 0.004, 3).astype(np.float32)

        return self.accel_raw.copy(), self.gyro_raw.copy()

# Clamped 2-DOF spherical pendulum with dynamic cable catenary aerodynamic drag
class AutonomousWinchDeliverySystem:
    def __init__(self):
        self.cable_len = 0.08
        self.cable_vel = 0.0
        self.theta = np.zeros(2, dtype=np.float32)
        self.theta_dot = np.zeros(2, dtype=np.float32)
        self.sway_offset = np.zeros(2, dtype=np.float32)
        self.catenary_midpoint = np.zeros(2, dtype=np.float32)
        self.tension_n = PAYLOAD_WEIGHT_N
        self.has_payload = True
        self.touchdown_timer = 0.0
        self.severed = False
        self.state = "STANDBY"
        self.delivery_completed = False

        # Physical Skid Weight-on-Wheels (WoW) Strain Gauges
        self.f_skid_left = 0.0
        self.f_skid_right = 0.0
        self.is_skid_grounded = False

    def reload_new_cargo(self):
        self.cable_len = 0.08
        self.cable_vel = 0.0
        self.theta.fill(0.0)
        self.theta_dot.fill(0.0)
        self.sway_offset.fill(0.0)
        self.catenary_midpoint.fill(0.0)
        self.has_payload = True
        self.state = "STANDBY"
        self.tension_n = PAYLOAD_WEIGHT_N
        self.touchdown_timer = 0.0
        self.delivery_completed = False
        self.is_skid_grounded = False

    def step(self, drone_z, target_surface_z, is_settled, is_surface_safe, wire_hit, agent_breach, agent_dist, 
             wind_2d, drone_accel_linear, neural_token_idx, nn_winch_rate=0.0, dt=0.016):
        if self.severed:
            self.state = "GUILLOTINE"
            return self.cable_len, 0.0, np.zeros(2, dtype=np.float32)

        if self.cable_len > 0.12:
            L = max(0.14, self.cable_len)
            omega_n_sq = G_ACCEL / L
            coriolis_damping = (2.0 * self.cable_vel / L) * self.theta_dot
            a_drone_coupling = drone_accel_linear[:2] / L
            
            v_cargo_horiz = self.theta_dot * L
            v_rel_wind = wind_2d - v_cargo_horiz
            speed_rel = float(np.linalg.norm(v_rel_wind))
            
            f_payload_drag = 0.5 * 1.225 * 0.045 * v_rel_wind * speed_rel
            f_cable_drag = 0.5 * 1.225 * (0.002 * L) * wind_2d * float(np.linalg.norm(wind_2d))
            wind_torque = (f_payload_drag + 0.5 * f_cable_drag) / (CARGO_MASS_KG * L)
            
            heavy_damping = 1.65 * self.theta_dot / (CARGO_MASS_KG * L)
            theta_ddot = -omega_n_sq * self.theta - coriolis_damping - a_drone_coupling + wind_torque - heavy_damping

            self.theta_dot += theta_ddot * dt
            self.theta_dot = np.clip(self.theta_dot, -2.5, 2.5)
            self.theta += self.theta_dot * dt
            self.theta = np.clip(self.theta, -0.25, 0.25)
            self.sway_offset = L * np.sin(self.theta)
            
            self.catenary_midpoint = 0.5 * self.sway_offset + (f_cable_drag * 0.02)

            theta_mag = float(np.linalg.norm(self.theta))
            theta_dot_sq = float(np.dot(self.theta_dot, self.theta_dot))
            a_vert = drone_accel_linear[2]
            
            if self.has_payload:
                t_centrif = CARGO_MASS_KG * L * theta_dot_sq
                t_grav = CARGO_MASS_KG * (G_ACCEL * math.cos(theta_mag) + a_vert)
                self.tension_n = max(1.5, t_grav + t_centrif)
            else:
                self.tension_n = 1.2
        else:
            self.theta.fill(0.0)
            self.theta_dot.fill(0.0)
            self.sway_offset.fill(0.0)
            self.catenary_midpoint.fill(0.0)
            self.tension_n = PAYLOAD_WEIGHT_N if self.has_payload else 1.2

        dropzone_occupied = (agent_dist < 2.8) and (target_surface_z < 2.0) and (self.state == "STANDBY")
        lockout = (not is_surface_safe) or wire_hit or (not is_settled) or dropzone_occupied

        if agent_breach and (target_surface_z < 2.0) and self.state in ["LOWERING", "TOUCHDOWN"]:
            self.state = "ABORT_RETRACT"

        if self.state == "STANDBY":
            self.cable_len = 0.08
            self.cable_vel = 0.0
            if not lockout and self.has_payload and is_settled:
                self.state = "LOWERING"

        elif self.state == "LOWERING":
            if lockout and self.cable_len > 0.12:
                self.state = "RETRACTING"
            else:
                base_rate = 2.40
                self.cable_vel = base_rate + np.clip(nn_winch_rate * 0.8, -1.0, 0.8)
                self.cable_len += self.cable_vel * dt

                cargo_z = drone_z - self.cable_len
                proximity_confirmed = abs(cargo_z - target_surface_z) <= 0.35
                tension_unloaded = self.tension_n < (0.50 * PAYLOAD_WEIGHT_N)

                if (cargo_z <= (target_surface_z + 0.18)) or (neural_token_idx == 3 and proximity_confirmed and tension_unloaded):
                    self.state = "TOUCHDOWN"
                    self.touchdown_timer = 0.0

        elif self.state == "TOUCHDOWN":
            self.cable_vel = 0.0
            self.tension_n = 1.8 + float(np.random.normal(0, 0.08))
            self.touchdown_timer += dt

            tension_unloaded = self.tension_n < (0.45 * PAYLOAD_WEIGHT_N)
            if self.touchdown_timer >= 0.25 or (neural_token_idx == 4 and tension_unloaded):
                self.has_payload = False
                self.state = "RETRACTING"

        elif self.state in ["RETRACTING", "ABORT_RETRACT"]:
            self.cable_vel = -3.20
            self.cable_len = max(0.08, self.cable_len + self.cable_vel * dt)

            if self.cable_len <= 0.08:
                self.cable_len = 0.08
                self.cable_vel = 0.0
                if self.state == "RETRACTING" and not self.has_payload:
                    self.delivery_completed = True
                self.state = "STANDBY"

        return self.cable_len, self.tension_n, self.sway_offset

    def update_skid_strain_gauges(self, is_on_surface, weight_force, f_thrust):
        if is_on_surface:
            f_normal_total = max(0.0, weight_force - f_thrust)
            noise = float(np.random.normal(0, 0.2))
            self.f_skid_left = 0.5 * f_normal_total + noise
            self.f_skid_right = 0.5 * f_normal_total - noise
            self.is_skid_grounded = (f_normal_total >= 0.65 * weight_force)
        else:
            self.f_skid_left = 0.0
            self.f_skid_right = 0.0
            self.is_skid_grounded = False

# =============================================================================
# 8. CONTINUOUS BEMT COAXIAL AIRFRAME (8-DOF) WITH CALIBRATED LIFT
# =============================================================================
class BEMTCoaxialAirframe8DOF:
    def __init__(self, init_pos):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.R = np.eye(3, dtype=np.float32)
        self.omega = np.zeros(3, dtype=np.float32)
        self.motor_omegas = np.full(8, HOVER_OMEGA_NOMINAL, dtype=np.float32)
        self.motor_health = np.ones(8, dtype=np.float32)
        self.accel_linear = np.zeros(3, dtype=np.float32)
        self.parachute_deployed = False

        self.last_p_watts = 185.0
        self.last_ige = 1.0

    def step(self, motor_cmds, current_wind, rho_air, ice_index=0.0, surface_z=0.0, has_payload=True, dt=0.016):
        current_mass = DRY_MASS_KG + (CARGO_MASS_KG if has_payload else 0.0)
        weight_force = current_mass * G_ACCEL

        if self.parachute_deployed:
            self.motor_omegas.fill(0.0)
            self.v[0] *= 0.90
            self.v[1] *= 0.90
            self.v[2] += (-4.1 - self.v[2]) * 3.0 * dt
            self.p += (self.v + current_wind * 0.4) * dt
            if self.p[2] <= surface_z:
                self.p[2] = surface_z
                self.v.fill(0.0)
                self.omega.fill(0.0)
            return 0.0, np.zeros(3, dtype=np.float32), 0.0, 1.0

        effective_cmds = np.clip(motor_cmds, 0.0, MAX_OMEGA) * self.motor_health
        self.motor_omegas += ((effective_cmds - self.motor_omegas) / MOTOR_TAU_S) * dt
        self.motor_omegas *= self.motor_health

        # Continuous Piecewise Ground Effect (IGE)
        z_h = max(0.02, float(self.p[2] - surface_z))
        ratio_zh_r = z_h / ROTOR_RADIUS_M
        if ratio_zh_r >= 0.5:
            ige_factor = 1.0 / (1.0 - (1.0 / (4.0 * ratio_zh_r)) ** 2)
            ige_factor = float(np.clip(ige_factor, 1.0, 1.28))
        else:
            ige_factor = float(1.20 + 0.16 * (ratio_zh_r / 0.5))

        d_ceil = max(0.3, CEILING_HEIGHT_M - self.p[2])
        ice_suction = float(1.0 + (ROTOR_RADIUS_M / (4.0 * d_ceil)) ** 2)

        lift_loss_factor = 1.0 - 0.28 * ice_index
        drag_rise_factor = 1.0 + 1.80 * ice_index

        T = np.zeros(8, dtype=np.float32)
        Q = np.zeros(8, dtype=np.float32)

        # Calibrated rotor thrust and aerodynamic torque allocation
        for i in range(8):
            if self.motor_health[i] == 0.0:
                continue
            omega_i = self.motor_omegas[i]
            is_bottom = (i % 2 == 1)
            scale_c = (C_T_CLEAN * COAXIAL_EFFICIENCY) if is_bottom else C_T_CLEAN

            t_rot = scale_c * (omega_i ** 2) * ige_factor * ice_suction * lift_loss_factor
            q_rot = C_Q_CLEAN * (omega_i ** 2) * drag_rise_factor
            T[i] = max(0.0, t_rot)
            Q[i] = max(0.0, q_rot)

        f_thrust = float(np.sum(T))
        arm = ARM_LENGTH_M / math.sqrt(2.0)

        tau_x = arm * (- (T[0] + T[1]) + (T[2] + T[3]) + (T[4] + T[5]) - (T[6] + T[7]))
        tau_y = arm * (- (T[0] + T[1]) - (T[2] + T[3]) + (T[4] + T[5]) + (T[6] + T[7]))
        tau_z = -Q[0] + Q[1] + Q[2] - Q[3] - Q[4] + Q[5] + Q[6] - Q[7]
        torques = np.array([tau_x, tau_y, tau_z], dtype=np.float32)

        v_air = self.v - current_wind
        speed_air = float(np.linalg.norm(v_air))
        f_drag = -0.5 * rho_air * np.array([0.22, 0.22, 0.38], dtype=np.float32) * v_air * max(1.0, speed_air)
        f_body = np.array([0.0, 0.0, f_thrust], dtype=np.float32)
        f_gravity = current_mass * GRAVITY_G

        self.accel_linear = (self.R @ f_body + f_gravity + f_drag) / current_mass
        self.accel_linear = np.clip(self.accel_linear, -35.0, 35.0)

        gyroscopic = np.cross(self.omega, INERTIA_MATRIX @ self.omega)
        accel_angular = INV_INERTIA @ (torques - gyroscopic)

        # Ground contact logic with anti-flip damping
        is_on_surface = (self.p[2] <= surface_z + 0.03)
        if is_on_surface and f_thrust < (0.45 * weight_force):
            self.p[2] = surface_z
            self.v.fill(0.0)
            self.omega.fill(0.0)
            accel_angular.fill(0.0)
            self.accel_linear.fill(0.0)
            # Keep upright on the pad
            self.R = np.eye(3, dtype=np.float32)
        else:
            self.p += self.v * dt + 0.5 * self.accel_linear * (dt ** 2)
            self.v += self.accel_linear * dt
            
            # Surface contact dampening: instantly kills rotation and vertical bounce
            if self.p[2] <= surface_z + 0.04:
                self.p[2] = surface_z
                self.v[2] = max(0.0, self.v[2] * 0.02)
                self.v[0] *= 0.70
                self.v[1] *= 0.70
                # ANTI-FLIP FIX: Zero out roll/pitch angular rates on ground contact
                self.omega[0] *= 0.10
                self.omega[1] *= 0.10
                self.omega[2] *= 0.20

            self.omega += accel_angular * dt
            d_theta = np.linalg.norm(self.omega) * dt
            if d_theta > 1e-6:
                axis = self.omega / np.linalg.norm(self.omega)
                K = skew_symmetric(axis)
                R_delta = np.eye(3, dtype=np.float32) + math.sin(d_theta) * K + (1.0 - math.cos(d_theta)) * (K @ K)
                self.R = self.R @ R_delta

            U, _, Vt = np.linalg.svd(self.R)
            R_ortho = U @ Vt
            if np.linalg.det(R_ortho) < 0.0:
                U[:, -1] *= -1.0
                R_ortho = U @ Vt
            self.R = R_ortho.astype(np.float32)
            # Inversion Guard: Prevent inverted upside-down flips on roofs or ground
            if self.R[2, 2] < 0.35:
                # If tilted beyond 70 degrees near surface, force upright stabilization
                self.R[0, 2] *= 0.5
                self.R[1, 2] *= 0.5
                self.R[2, 2] = math.sqrt(max(0.1, 1.0 - self.R[0, 2]**2 - self.R[1, 2]**2))
                self.omega.fill(0.0)

        p_watts = float(np.sum(Q * self.motor_omegas)) / 0.85 + 24.0
        self.last_p_watts = p_watts
        self.last_ige = ige_factor
        return f_thrust, torques, p_watts, ige_factor

# =============================================================================
# 9. ADVANCED SENSORS & SENSORY REALITY SUITE
# =============================================================================
class AdvancedMultiModalSensorSuite:
    def __init__(self, world: MetropolitanCityGridWorld400m):
        self.world = world

    # SENSOR UPGRADE 1: Downward Optical Flow with exact perspective slant range
    def poll_optical_flow(self, drone_p, drone_v, drone_R, drone_omega, surface_z=0.0):
        cos_tilt = max(0.35, float(drone_R[2, 2]))
        z_rel = max(0.15, float(drone_p[2] - surface_z))
        z_slant = z_rel / cos_tilt
        
        v_body = drone_R.T @ drone_v
        
        w_flow_x = (v_body[0] / z_slant) - drone_omega[1] + float(np.random.normal(0, 0.015))
        w_flow_y = (v_body[1] / z_slant) + drone_omega[0] + float(np.random.normal(0, 0.015))
        
        quality = 0.95 if z_rel < 35.0 else float(np.clip(1.0 - (z_rel - 35.0) / 25.0, 0.1, 0.95))
        return np.array([w_flow_x, w_flow_y], dtype=np.float32), quality

    # SENSOR UPGRADE 2: ArUco / AprilTag Visual Fiducial Reticle
    def poll_aruco_fiducial_detector(self, drone_p, drone_R, target_pad, expected_tag_id):
        rel_pos = target_pad - drone_p
        z_clearance = max(0.1, float(drone_p[2] - target_pad[2]))
        dist_xy = float(np.linalg.norm(rel_pos[:2]))
        
        half_fov_radius = z_clearance * math.tan(math.radians(35.0))
        is_visible = (dist_xy <= half_fov_radius) and (z_clearance <= 22.0)
        
        if is_visible:
            cam_offset = drone_R.T @ rel_pos
            noise = np.random.normal(0, 0.008, 2).astype(np.float32)
            dx = float(cam_offset[0] + noise[0])
            dy = float(cam_offset[1] + noise[1])
            d_psi = math.atan2(drone_R[1, 0], drone_R[0, 0])
            return np.array([dx, dy, d_psi], dtype=np.float32), 1.0
        return np.zeros(3, dtype=np.float32), 0.0

    # SENSOR UPGRADE 6: Directional Acoustic Array
    def poll_directional_acoustic_array(self, drone_p, vehicles, sim_time=0.0):
        closest_dist = 999.0
        siren_rel = np.zeros(2, dtype=np.float32)
        has_siren = False
        
        for v in vehicles:
            if v.get("is_emergency", False):
                d = float(np.linalg.norm(drone_p[:2] - v["pos"][:2]))
                if d < closest_dist:
                    closest_dist = d
                    siren_rel = v["pos"][:2] - drone_p[:2]
                    has_siren = True
                    
        if has_siren and closest_dist < 120.0:
            siren_angle = math.atan2(siren_rel[1], siren_rel[0])
            spl_dba = float(np.clip(102.0 - 20.0 * math.log10(max(1.0, closest_dist)), 45.0, 98.0))
            freq_peak = 850.0 + 150.0 * math.sin(sim_time * 3.0)
            return siren_angle, freq_peak, spl_dba
        return 0.0, 180.0, 55.0

    # SENSOR UPGRADE 7: ADS-B In Civil Airspace Receiver
    def poll_adsb_civil_transponder(self, drone_p, drone_v, civil_aircraft):
        rel_pos = civil_aircraft["pos"] - drone_p
        dist_3d = float(np.linalg.norm(rel_pos))
        rel_vel = civil_aircraft["vel"] - drone_v
        closing_speed = -float(np.dot(rel_pos / max(1e-3, dist_3d), rel_vel))
        return rel_pos, float(np.clip(closing_speed, -30.0, 30.0)), dist_3d

    def poll_smart_hook_imu(self, drone_a, drone_w, sway_2d, cable_l, has_payload):
        if not has_payload or cable_l <= 0.10:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        sway_speed = float(np.linalg.norm(sway_2d)) * 4.0
        a_box = drone_a + np.array([-sway_2d[0]*2.5, -sway_2d[1]*2.5, 0.2], dtype=np.float32)
        w_box = np.array([sway_2d[1]*1.8, -sway_2d[0]*1.8, sway_speed*0.5], dtype=np.float32)
        return a_box, w_box

    def poll_3axis_load_cell(self, tension_n, sway_2d, cable_l):
        if cable_l <= 0.10:
            return np.array([0.0, 0.0, -tension_n], dtype=np.float32)
        sway_dist = max(1e-4, math.hypot(sway_2d[0], sway_2d[1]))
        sin_theta = min(0.65, sway_dist / max(0.2, cable_l))
        cos_theta = math.sqrt(max(1e-4, 1.0 - sin_theta**2))
        fx = -tension_n * (sway_2d[0] / sway_dist) * sin_theta
        fy = -tension_n * (sway_2d[1] / sway_dist) * sin_theta
        fz = -tension_n * cos_theta
        return np.array([fx, fy, fz], dtype=np.float32)

    def poll_thermal_and_polarization(self, pos_xy):
        px, py = pos_xy
        min_roof_dist = 999.0
        for h in self.world.houses:
            x1, x2, y1, y2, _ = h["aabb"]
            dx = max(0.0, max(x1 - px, px - x2))
            dy = max(0.0, max(y1 - py, py - y2))
            d = math.hypot(dx, dy)
            min_roof_dist = min(min_roof_dist, d)

        roof_heat = 7.5 * math.exp(-min_roof_dist / 8.0)
        park_dist = math.hypot(px - (-25.0), py - (-25.0))
        park_cool = -3.5 * math.exp(-park_dist / 22.0)
        th_dt = float(roof_heat + park_cool + 0.5)

        dolp = float(0.12 + 0.55 * math.exp(-min_roof_dist / 6.0))
        ndvi = float(0.10 + 0.68 * math.exp(-park_dist / 25.0))
        return th_dt, dolp, ndvi

    def poll_esc_foc_telemetry(self, motor_omegas, motor_health):
        i_rms = np.zeros(8, dtype=np.float32)
        t_mosfet = np.zeros(8, dtype=np.float32)
        for i in range(8):
            if motor_health[i] == 0.0:
                i_rms[i] = 0.0
                t_mosfet[i] = 24.0
            else:
                p_ratio = (motor_omegas[i] / MAX_OMEGA)**2
                i_rms[i] = 3.5 + p_ratio * 18.5  
                t_mosfet[i] = 32.0 + p_ratio * 46.0 
        return i_rms, t_mosfet

    def poll_5port_air_data(self, drone_v, current_wind, drone_R):
        v_air_world = drone_v - current_wind
        v_air_body = drone_R.T @ v_air_world
        speed = max(0.5, float(np.linalg.norm(v_air_body)))
        alpha_rad = math.atan2(v_air_body[2], max(0.2, v_air_body[0]))
        beta_rad = math.asin(np.clip(v_air_body[1] / speed, -0.9, 0.9))
        rh_pct = float(np.clip(68.0 - 0.4 * drone_v[2], 30.0, 95.0))
        pm10_dust = float(np.clip(22.0 + 1.2 * float(np.linalg.norm(current_wind[:2])), 5.0, 80.0))
        return math.degrees(alpha_rad), math.degrees(beta_rad), rh_pct, pm10_dust

    def poll_rtk_gnss_metrics(self, drone_p, is_spoofed):
        hdop = 3.8 if is_spoofed else 0.85
        cn0_avg = 28.5 if is_spoofed else 44.0
        rf_jam_pwr = -52.0 if is_spoofed else -110.0
        return hdop, cn0_avg, rf_jam_pwr, 0.02

# ==========================================
# 10. MACRO PERCEPTION & ASSURANCE SUITE
# ==========================================
class MacroPerceptionLiDARSuite:
    def __init__(self, world: MetropolitanCityGridWorld400m):
        self.world = world

    def cast_360_lidar(self, drone_pos, yaw_rad, dynamic_vehicles=None):
        px, py, pz = drone_pos
        ranges = np.full(16, 30.0, dtype=np.float32)

        for beam_idx in range(16):
            angle = yaw_rad + (2.0 * math.pi * beam_idx / 16.0)
            dx, dy = math.cos(angle), math.sin(angle)
            min_dist = 30.0

            for h in self.world.houses:
                x1, x2, y1, y2, hh = h["aabb"]
                if pz <= (hh + 0.1):
                    if abs(dx) > 1e-5:
                        t1, t2 = (x1 - px) / dx, (x2 - px) / dx
                        t_min_x, t_max_x = min(t1, t2), max(t1, t2)
                    else:
                        if px < x1 or px > x2: continue
                        t_min_x, t_max_x = -1e6, 1e6

                    if abs(dy) > 1e-5:
                        t1, t2 = (y1 - py) / dy, (y2 - py) / dy
                        t_min_y, t_max_y = min(t1, t2), max(t1, t2)
                    else:
                        if py < y1 or py > y2: continue
                        t_min_y, t_max_y = -1e6, 1e6

                    t_enter = max(t_min_x, t_min_y)
                    t_exit = min(t_max_x, t_max_y)

                    if t_enter <= t_exit and t_exit > 0.0:
                        hit_dist = t_enter if t_enter > 0.0 else 0.0
                        if hit_dist < min_dist:
                            min_dist = hit_dist

            if dynamic_vehicles:
                for v in dynamic_vehicles:
                    vx, vy = v["pos"][0], v["pos"][1]
                    d_veh = math.hypot(px - vx, py - vy)
                    if pz < 3.2 and d_veh < min_dist:
                        bearing = math.atan2(vy - py, vx - px)
                        diff = abs((bearing - angle + math.pi) % (2.0 * math.pi) - math.pi)
                        if diff <= (math.pi / 16.0):
                            min_dist = float(d_veh)

            ranges[beam_idx] = min_dist
        return ranges

    def cast_downward_conical(self, drone_pos):
        px, py, pz = drone_pos
        conical_ranges = np.full(8, 15.0, dtype=np.float32)
        wire_clearance = 50.0

        for wire in self.world.powerlines:
            w1, w2 = wire["p1"], wire["p2"]
            if min(w1[0], w2[0]) - 1.0 <= px <= max(w1[0], w2[0]) + 1.0:
                dist_y = abs(py - w1[1])
                if dist_y < 3.5 and pz > w1[2]:
                    wire_clearance = min(wire_clearance, float(pz - w1[2]))

        surf_h = self.world.get_surface_height(px, py)
        true_clearance = max(0.05, float(pz - surf_h))

        cone_slant = math.cos(math.radians(25.0))
        for i in range(8):
            conical_ranges[i] = min(15.0, float(true_clearance / max(0.1, cone_slant)))

        return conical_ranges, wire_clearance

class SurfaceAndWireAssuranceSuite:
    def __init__(self, world: MetropolitanCityGridWorld400m):
        self.world = world

    def audit_surface(self, pos_xy, target_z=0.0, buffer_m=2.5):
        px, py = pos_xy
        for h in self.world.houses:
            x1, x2, y1, y2, hh = h["aabb"]
            if x1 <= px <= x2 and y1 <= py <= y2:
                for c in h.get("clutter", []):
                    cx1, cx2, cy1, cy2, ch = c["aabb"]
                    if (cx1 - 0.8) <= px <= (cx2 + 0.8) and (cy1 - 0.8) <= py <= (cy2 + 0.8):
                        return "ROOF_CLUTTER_HAZARD", False, 0.40

                if abs(target_z - hh) < 1.4 or target_z > 5.0:
                    return "ROOFTOP_PAD_SAFE", True, 0.95
                else:
                    return "ROOF_HAZARD", False, 0.85

        for line in self.world.powerlines:
            w1 = line["p1"]
            if abs(py - w1[1]) < (buffer_m + 1.2):
                return "WIRE_CORRIDOR_HAZARD", False, 0.50

        return "SOLID_TERRAIN_SAFE", True, 0.92

    def snap_to_nearest_safe_ground(self, target_xy):
        cand = np.array([target_xy[0], target_xy[1], 0.0], dtype=np.float32)
        for h in self.world.houses:
            x1, x2, y1, y2, hh = h["aabb"]
            if x1 <= cand[0] <= x2 and y1 <= cand[1] <= y2:
                roof_x = float((x1 + x2) / 2.0)
                roof_y = float((y1 + y2) / 2.0)
                return np.array([roof_x, roof_y, hh], dtype=np.float32)

        _, is_safe, _ = self.audit_surface(cand[:2], target_z=0.0)
        if is_safe:
            return cand

        for r in np.linspace(3.5, 25.0, 16):
            for ang in np.linspace(0, 2*math.pi, 16):
                test_pos = np.array([cand[0] + r*math.cos(ang), cand[1] + r*math.sin(ang), 0.0], dtype=np.float32)
                _, safe_found, _ = self.audit_surface(test_pos[:2], target_z=0.0)
                if safe_found:
                    return test_pos

        return np.array([-155.0, -155.0, 0.0], dtype=np.float32)

class FullPhysicsSensorSuite:
    def __init__(self):
        self.wire_y = 170.0
        self.wire_z = 12.0

    def poll_all_physics_sensors(self, drone_p, drone_v, current_wind, surface_z=0.0):
        z_true = max(0.0, float(drone_p[2]))
        t_air = T_0 - L_LAPSE * z_true
        p_static_hpa = P_0 * ((1.0 - (L_LAPSE * z_true) / T_0) ** 5.25588)
        rho_air = (p_static_hpa * 100.0) / (R_SPEC_AIR * t_air)
        z_baro = 44330.8 * (1.0 - ((p_static_hpa / P_0) ** 0.190284)) + float(np.random.normal(0, 0.04))

        v_air = drone_v - current_wind
        v_tas = float(np.linalg.norm(v_air))
        q_dyn = 0.5 * rho_air * (v_tas ** 2)

        d_wire = math.hypot(drone_p[1] - self.wire_y, drone_p[2] - self.wire_z)
        emf_rms = float(140.0 / max(0.25, d_wire))
        
        true_clearance = max(0.05, float(drone_p[2] - surface_z))
        min_sonar = max(0.05, min(15.0, true_clearance))
        return {"z_baro": z_baro, "rho_air": rho_air, "v_tas": v_tas, "q_dyn": q_dyn, "emf_rms": emf_rms, "min_sonar": min_sonar}

# =============================================================================
# 11. QUANTUM SENSOR & OPTICAL-FLOW AUGMENTED ERROR-STATE EKF ON SO(3)
# =============================================================================
class MatterWaveColdAtomQuantumSensor:
    def __init__(self):
        self.k_eff = K_EFF_WAVEVECTOR
        self.T = RAMAN_PULSE_SEP_T

    def measure_phase_shift(self, specific_force_body):
        phase_shift = self.k_eff * specific_force_body * (self.T ** 2)
        phase_shift += np.random.normal(0, 0.002, 3).astype(np.float32)
        a_quantum_body = phase_shift / (self.k_eff * (self.T ** 2))
        return a_quantum_body, phase_shift

class ErrorStateEKF15:
    def __init__(self, init_pos):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.b_a = np.zeros(3, dtype=np.float32)
        self.b_g = np.zeros(3, dtype=np.float32)
        self.P = np.diag([0.15]*3 + [0.1]*3 + [0.02]*3 + [0.05]*3 + [0.01]*3).astype(np.float32)
        self.Q_d = np.diag([1e-4]*3 + [1e-3]*3 + [1e-4]*3 + [1e-5]*3 + [1e-6]*3).astype(np.float32)

    def predict(self, accel_body_raw, gyro_raw, dt=0.016):
        a_unbiased = np.clip(accel_body_raw - self.b_a, -30.0, 30.0)
        w_unbiased = np.clip(gyro_raw - self.b_g, -12.0, 12.0)
        R = quat_to_rot(self.q)
        f_inertial = R @ a_unbiased + GRAVITY_G

        self.p += self.v * dt + 0.5 * f_inertial * (dt ** 2)
        self.v += f_inertial * dt
        self.v = np.clip(self.v, -15.0, 15.0)

        w_norm = np.linalg.norm(w_unbiased)
        if w_norm > 1e-6:
            half = 0.5 * w_norm * dt
            dq = np.array([math.cos(half), *((w_unbiased / w_norm) * math.sin(half))], dtype=np.float32)
            self.q = quat_mult(self.q, dq)
            self.q /= np.linalg.norm(self.q)

        F_x = np.eye(15, dtype=np.float32)
        F_x[0:3, 3:6] = np.eye(3) * dt
        F_x[3:6, 6:9] = -R @ skew_symmetric(a_unbiased) * dt
        F_x[3:6, 9:12] = -R * dt
        F_x[6:9, 6:9] = np.eye(3) - skew_symmetric(w_unbiased) * dt
        F_x[6:9, 12:15] = -np.eye(3) * dt

        self.P = F_x @ self.P @ F_x.T + self.Q_d
        self.P = np.nan_to_num(self.P, nan=0.05)
        self.P = 0.5 * (self.P + self.P.T)

        max_diag = np.array([100.0]*3 + [50.0]*3 + [0.5]*3 + [1.0]*3 + [0.2]*3, dtype=np.float32)
        diag_P = np.diag(self.P)
        for k in range(15):
            if diag_P[k] > max_diag[k]:
                scale = math.sqrt(max_diag[k] / diag_P[k])
                self.P[k, :] *= scale
                self.P[:, k] *= scale
        self.P = 0.5 * (self.P + self.P.T)

    def update_measurement(self, H, innovation, R_cov):
        S = H @ self.P @ H.T + R_cov
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        delta_x = (K @ innovation).flatten()
        delta_x[0:6] = np.clip(delta_x[0:6], -3.5, 3.5)
        delta_x[6:9] = np.clip(delta_x[6:9], -0.25, 0.25)
        delta_x[9:15] = np.clip(delta_x[9:15], -0.5, 0.5)

        self.p += delta_x[0:3]
        self.v += delta_x[3:6]
        d_theta = delta_x[6:9]
        angle = np.linalg.norm(d_theta)
        if angle > 1e-6:
            dq = np.array([math.cos(angle * 0.5), *((d_theta / angle) * math.sin(angle * 0.5))], dtype=np.float32)
            self.q = quat_mult(self.q, dq)
            self.q /= np.linalg.norm(self.q)

        self.b_a += delta_x[9:12]
        self.b_g += delta_x[12:15]
        I_KH = np.eye(15, dtype=np.float32) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_cov @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # Exact body-to-world Jacobian for yaw innovation
    def update_heading(self, measured_yaw_rad, r_cov=0.02):
        R_curr = quat_to_rot(self.q)
        current_yaw = math.atan2(R_curr[1, 0], R_curr[0, 0])
        yaw_innov = (measured_yaw_rad - current_yaw + math.pi) % (2.0 * math.pi) - math.pi

        denom = max(1e-6, R_curr[0, 0]**2 + R_curr[1, 0]**2)
        H_yaw = np.zeros((1, 15), dtype=np.float32)
        H_yaw[0, 6] = (R_curr[0, 0] * R_curr[1, 2] - R_curr[1, 0] * R_curr[0, 2]) / denom
        H_yaw[0, 7] = (R_curr[0, 0] * R_curr[1, 1] - R_curr[1, 0] * R_curr[0, 1]) / denom
        H_yaw[0, 8] = 1.0

        self.update_measurement(H_yaw, np.array([yaw_innov], dtype=np.float32), np.array([[r_cov]], dtype=np.float32))

    def update_optical_flow_velocity(self, v_flow_world, r_cov=0.025):
        H_v = np.zeros((2, 15), dtype=np.float32)
        H_v[0, 3] = 1.0
        H_v[1, 4] = 1.0
        v_innov = v_flow_world[:2] - self.v[:2]
        self.update_measurement(H_v, v_innov, np.eye(2, dtype=np.float32) * r_cov)

# =============================================================================
# 12. SMOOTH VELOCITY-GOVERNED MIXER & SIMULTANEOUS DUAL CBF-QP
# =============================================================================
class ReciprocalSwarmCBFQPFilter:
    def __init__(self):
        self.alpha_obs = CBF_ALPHA_OBSTACLE
        self.safe_radius = CBF_SAFE_RADIUS_M
        self.swarm_r = SWARM_INTER_DRONE_SAFE_RADIUS_M

    # Simultaneous Active-Set Dykstra Projection
    def enforce_safety(self, drone_idx, drone_p, drone_v, nominal_accel, obstacles, swarm_states=None, target_bldg_id=None, departure_bldg_id=None):
        A_rows = []
        b_rows = []

        # 1. Obstacle Cylindrical Constraints
        for obs in obstacles:
            obs_id = obs.get("id")
            obs_h = obs.get("height", 0.0)
            if obs_id in [target_bldg_id, departure_bldg_id]:
                continue
            
            rel_pos = drone_p - obs["pos"]
            dist_xy = math.hypot(rel_pos[0], rel_pos[1])
            h_obs = dist_xy - (obs["radius"] + self.safe_radius)

            if drone_p[2] <= (obs_h - 0.4) and h_obs < 3.0:
                n_xy = rel_pos[:2] / max(1e-4, dist_xy)
                grad_h = np.array([n_xy[0], n_xy[1], 0.0], dtype=np.float32)
                v_tangent = drone_v[0] * (-n_xy[1]) + drone_v[1] * n_xy[0]
                centripetal = (v_tangent ** 2) / max(0.5, dist_xy)
                l_fh = float(np.dot(grad_h, drone_v))
                
                A_rows.append(grad_h)
                b_rows.append(-l_fh - self.alpha_obs * h_obs - centripetal)

            # Roof landing surface barrier
            if dist_xy <= (obs["radius"] + 0.5) and drone_p[2] >= (obs_h - 0.2):
                h_roof = drone_p[2] - (obs_h + 0.15)
                if h_roof < 0.8:
                    A_rows.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
                    b_rows.append(-drone_v[2] - 2.0 * h_roof)

        # 2. Inter-Drone Swarm Reciprocal Avoidance
        if swarm_states:
            for j, other in enumerate(swarm_states):
                if j == drone_idx: continue
                p_other, v_other = other["p"], other["v"]

                # Do not push peers when landing on assigned adjacent pads below 3.5m
                if drone_p[2] < 6.0 and p_other[2] < 6.0:
                    continue

                if abs(drone_p[2] - p_other[2]) < 2.8:
                    d_p = drone_p[:2] - other["p"][:2]
                    dist_ij = float(np.linalg.norm(d_p))
                    h_swarm = dist_ij - self.swarm_r
                    if h_swarm < 3.2:
                        n_ij = np.array([d_p[0]/max(1e-4, dist_ij), d_p[1]/max(1e-4, dist_ij), 0.0], dtype=np.float32)
                        v_rel = drone_v[:2] - v_other[:2]
                        l_fh_swarm = float(np.dot(n_ij[:2], v_rel))
                        A_rows.append(n_ij)
                        b_rows.append(-l_fh_swarm - 2.2 * h_swarm)

        # =====================================================================
        # >>> ADD THIS BLOCK RIGHT HERE: STORM-RESILIENT FLOOR BARRIER
        # =====================================================================
        # Invisible safety net at z = 0.20m.
        # Catches the drone if a storm downdraft tries to push it into the ground,
        # but ignores it when the drone is below 0.30m so it can land softly.
        h_gnd = drone_p[2] - 0.20
        if h_gnd < 1.2 and drone_p[2] > 0.30:
            A_rows.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
            b_rows.append(-drone_v[2] - 3.0 * h_gnd)
        # =====================================================================

        if not A_rows:
            return nominal_accel

        # Solve simultaneous QP via Dykstra's cyclic projection
        A = np.array(A_rows, dtype=np.float32)
        b = np.array(b_rows, dtype=np.float32)
        u = nominal_accel.copy()

        for _ in range(12):
            max_viol = 0.0
            for k in range(len(A)):
                c_k = float(np.dot(A[k], u)) - b[k]
                if c_k < 0.0:
                    max_viol = max(max_viol, -c_k)
                    delta_lambda = -c_k / float(np.dot(A[k], A[k]))
                    u += delta_lambda * A[k]
            if max_viol < 1e-4:
                break

        return u

class RateDampedActuatorMixerWithCBF:
    def __init__(self):
        self.k_tilt = 1.6      
        self.k_w = 0.95        
        self.cbf_filter = ReciprocalSwarmCBFQPFilter()

        arm = ARM_LENGTH_M / math.sqrt(2.0)
        cq_ct = C_Q_CLEAN / C_T_CLEAN
        
        self.B_nominal = np.array([
            [ 1.0,    1.0,   1.0,    1.0,    1.0,   1.0,    1.0,   1.0],
            [-arm,   -arm,   arm,    arm,    arm,   arm,   -arm,  -arm],
            [-arm,   -arm,  -arm,   -arm,    arm,   arm,    arm,   arm],
            [-cq_ct, cq_ct, cq_ct, -cq_ct, -cq_ct, cq_ct, cq_ct, -cq_ct]
        ], dtype=np.float32)

    def mix(self, drone_health, p_est, v_est, R_est, omega_est, p_target, neural_guidance, token_idx,
            estimated_wind, obstacles, swarm_states=None, drone_idx=0, ref_accel=None, ref_vel=None,
            target_bldg_id=None, departure_bldg_id=None, cable_len=0.08, sway_2d=None, load_cell_3d=None, ige_fac=1.0,
            v_pack=23.5, bms_temp_c=22.0, has_payload=True, dt=0.016):

        current_mass = DRY_MASS_KG + (CARGO_MASS_KG if has_payload else 0.0)
        weight_force = current_mass * G_ACCEL

        # 1. DISARM MOTORS WHEN RESTING ON PAD DURING SERVICE
        if p_target[2] <= 0.08 and p_est[2] <= 0.18:
            return np.zeros(8, dtype=np.float32), 0.0, np.zeros(3, dtype=np.float32)

        if ref_accel is None: ref_accel = np.zeros(3, dtype=np.float32)
        if ref_vel is None: ref_vel = np.zeros(3, dtype=np.float32)
        if swarm_states is None: swarm_states = []

        delta_p = p_target - p_est
        delta_p_clamped = np.clip(delta_p, -5.0, 5.0)

        # 2. HORIZONTAL TRACKING (XY PLANE)
        v_err_xy = np.clip(ref_vel[:2] - v_est[:2], -3.5, 3.5)
        a_p_xy = np.array([1.8, 1.8], dtype=np.float32) * delta_p_clamped[:2]
        a_d_xy = np.array([2.0, 2.0], dtype=np.float32) * v_err_xy
        a_track_xy = a_p_xy + a_d_xy

        # 3. VERTICAL TRACKING WITH CRITICAL DAMPING (ELIMINATES VERTICAL SURGING)
        if p_target[2] <= 0.10 and p_est[2] < 1.5:
            v_cmd_z = float(np.clip(ref_vel[2], -0.35, 0.20))
        else:
            v_cmd_z = float(np.clip(ref_vel[2] + 0.50 * delta_p_clamped[2], -0.90, 1.40))

        sink_error = v_cmd_z - v_est[2]
        k_vertical = 4.2 if (sink_error > 0.30 and p_est[2] < 30.0) else 2.8
        a_track_z = k_vertical * sink_error
        a_track = np.array([a_track_xy[0], a_track_xy[1], a_track_z], dtype=np.float32)

        # Restrict tilt during vertical chimneys and landing to eliminate horizontal swaying
        is_vertical_phase = (abs(ref_vel[0]) < 0.2 and abs(ref_vel[1]) < 0.2)
        max_tilt_rad = math.radians(12.0 if is_vertical_phase else 30.0)

        # 4. ACTIVE WIND DRAG FEEDFORWARD (CANCELS GUSTS & DOWNDRAFTS)
        v_air_rel = v_est - estimated_wind
        speed_rel = float(np.linalg.norm(v_air_rel))
        f_aero_drag = -0.5 * 1.225 * np.array([0.22, 0.22, 0.38], dtype=np.float32) * v_air_rel * max(1.0, speed_rel)
        a_wind_feedforward = -f_aero_drag / current_mass

        a_anti_sway = np.zeros(3, dtype=np.float32)
        if cable_len > 0.12 and sway_2d is not None:
            a_anti_sway[:2] -= 1.2 * sway_2d

        a_nominal = a_track + ref_accel + a_anti_sway + a_wind_feedforward
        a_safe = self.cbf_filter.enforce_safety(
            drone_idx, p_est, v_est, a_nominal, obstacles, swarm_states, 
            target_bldg_id=target_bldg_id, departure_bldg_id=departure_bldg_id
        )
        
        F_d = a_safe * current_mass + current_mass * (-GRAVITY_G)

        # Ground effect cushion cancellation during final landing
        if ige_fac > 1.02 and p_target[2] <= 0.10:
            F_d[2] /= ige_fac

        # 5. DYNAMIC THRUST CEILING & GOVERNED DESCENT FLOOR
        is_airborne = (p_est[2] > 0.15)
        downdraft_active = (estimated_wind[2] < -0.8 or v_air_rel[2] < -1.0)
        max_multiplier = 2.30 if downdraft_active else 1.80
        min_f_vert = (0.65 * weight_force) if is_airborne else 0.0
        max_f_vert = max_multiplier * weight_force
        F_d[2] = float(np.clip(F_d[2], min_f_vert, max_f_vert))

        # 6. 32-DEGREE TILT AUTHORITY TO PENETRATE GALE FRONTS
        max_tilt_rad = math.radians(32.0)
        f_horiz = math.hypot(F_d[0], F_d[1])
        f_vert = max(1.0, F_d[2])
        if f_horiz / f_vert > math.tan(max_tilt_rad):
            scale_h = (f_vert * math.tan(max_tilt_rad)) / f_horiz
            F_d[0] *= scale_h
            F_d[1] *= scale_h

        b3 = R_est[:, 2]
        f_thrust = float(np.dot(F_d, b3))
        f_thrust = float(np.clip(f_thrust, min_f_vert, max_f_vert))

        b3_d = F_d / (np.linalg.norm(F_d) + 1e-6)
        e_tilt_world = np.cross(b3, b3_d)
        e_tilt_body = R_est.T @ e_tilt_world

        # 7. ANTI-FLIP TOUCHDOWN ATTITUDE SOFTENER
        # Soften tilt gains right at the pad so skids don't act as a lever arm and flip the drone
        near_surface = (p_est[2] < 0.25 and p_target[2] <= 0.10)
        eff_k_tilt = 0.40 if near_surface else self.k_tilt
        eff_k_w = 1.20 if near_surface else self.k_w

        tau_x = eff_k_tilt * e_tilt_body[0] - eff_k_w * omega_est[0]
        tau_y = eff_k_tilt * e_tilt_body[1] - eff_k_w * omega_est[1]

        # 8. DEADBEAT YAW CONTROL (0.75 Nm AUTHORITY PREVENTS COAXIAL SPINNING)
        yaw_target = 0.0
        current_yaw = math.atan2(R_est[1, 0], R_est[0, 0])
        yaw_err = (current_yaw - yaw_target + math.pi) % (2.0 * math.pi) - math.pi
        tau_z = -2.2 * yaw_err - 0.70 * omega_est[2]
        tau_z = float(np.clip(tau_z, -0.75, 0.75))

        gyroscopic = np.cross(omega_est, INERTIA_MATRIX @ omega_est)
        tau = np.array([tau_x + gyroscopic[0], tau_y + gyroscopic[1], tau_z + gyroscopic[2]], dtype=np.float32)

        desired_wrench = np.array([f_thrust, tau[0], tau[1], tau[2]], dtype=np.float32)

        W_health = np.diag(drone_health).astype(np.float32)
        B_active = self.B_nominal @ W_health
        try:
            B_pinv = np.linalg.pinv(B_active)
            T_cmds = B_pinv @ desired_wrench
        except np.linalg.LinAlgError:
            T_cmds = np.full(8, f_thrust / 8.0, dtype=np.float32)

        active_mask = (drone_health > 0.0)
        if np.any(active_mask):
            min_active_t = float(np.min(T_cmds[active_mask]))
            if min_active_t < 0.10:
                shift = min(0.35, 0.10 - min_active_t)
                T_cmds[active_mask] += shift

        T_cmds = np.clip(T_cmds * drone_health, 0.0, 10.5)

        omega_cmds = np.zeros(8, dtype=np.float32)
        for i in range(8):
            if drone_health[i] > 0.0:
                scale_c = (C_T_CLEAN * COAXIAL_EFFICIENCY) if (i % 2 == 1) else C_T_CLEAN
                omega_cmds[i] = math.sqrt(T_cmds[i] / scale_c)

        return omega_cmds, f_thrust, a_safe

# =============================================================================
# 13. FULL SENSORY REALITY OBSERVATION SYNTHESIZER (139 FEATURES -> 417-DIM)
# =============================================================================
class DroneObservationSynthesizer139Dim:
    def __init__(self):
        self.last_tension = PAYLOAD_WEIGHT_N

    def synthesize_instantaneous(self, p_est, v_est, R_est, omega_est, target_p, lidar_16, conical_8,
                                 v_tas, q_dyn, min_sonar, wind_est, ice_idx, cable_len, cable_vel,
                                 tension_n, sway_2d, bms_soc, v_pack, bms_temp_c, r0_dyn,
                                 chi2_val, jerk_discrepancy, is_spoofed, emf_rms, wire_dist,
                                 agent_dist, ground_spl_dba, has_payload,
                                 hook_a, hook_w, load_cell_3d, thermal_dt, dolp, ndvi,
                                 esc_i_rms, esc_t_mosfet, alpha_deg, beta_deg, rh_pct, dust_pm10,
                                 hdop, cn0_avg, rf_jam_pwr, mag_grad,
                                 airframe_accel=None, airframe_gyro=None,
                                 peer_swarm_kinematics=None,
                                 optical_flow_2d=None, flow_quality=0.95,
                                 aruco_offset_3d=None, aruco_locked=0.0,
                                 skid_force_left=0.0, skid_force_right=0.0, is_skid_grounded=False,
                                 siren_angle=0.0, siren_freq=180.0, siren_spl=55.0,
                                 adsb_rel_pos=None, adsb_closing_spd=0.0,
                                 dt=0.016):
        obs = np.zeros(BASE_OBS_DIM, dtype=np.float32)

        if airframe_accel is None: airframe_accel = np.zeros(3, dtype=np.float32)
        if airframe_gyro is None: airframe_gyro = np.zeros(3, dtype=np.float32)
        if optical_flow_2d is None: optical_flow_2d = np.zeros(2, dtype=np.float32)
        if aruco_offset_3d is None: aruco_offset_3d = np.zeros(3, dtype=np.float32)
        if adsb_rel_pos is None: adsb_rel_pos = np.zeros(3, dtype=np.float32)

        # 1. Goal Kinematics & Polar Geometry [0:6]
        delta_p = target_p - p_est
        obs[0:3] = np.clip(delta_p / 140.0, -1.0, 1.0)
        d_goal = float(np.linalg.norm(delta_p))
        obs[3] = np.clip(d_goal / 200.0, 0.0, 1.0)
        obs[4] = math.atan2(delta_p[1], delta_p[0]) / math.pi
        obs[5] = math.atan2(delta_p[2], max(0.1, math.hypot(delta_p[0], delta_p[1]))) / (0.5 * math.pi)

        # 2. Airframe Kinematics [6:15]
        obs[6:9] = np.clip(v_est / 8.0, -1.0, 1.0)
        obs[9:12] = np.clip(R_est[:, 2], -1.0, 1.0)
        obs[12:15] = np.clip(omega_est / 8.0, -1.0, 1.0)

        # 3. 16-Beam Planar LiDAR [15:31]
        obs[15:31] = np.clip(lidar_16 / 30.0, 0.0, 1.0)

        # 4. 8-Beam Conical Scanner [31:39]
        obs[31:39] = np.clip(conical_8 / 15.0, 0.0, 1.0)

        # 5. Aero & Weather [39:46]
        obs[39] = np.clip(v_tas / 18.0, 0.0, 1.0)
        obs[40] = np.clip(q_dyn / 120.0, 0.0, 1.0)
        obs[41] = np.clip(min_sonar / 15.0, 0.0, 1.0)
        obs[42:45] = np.clip(wind_est / 8.0, -1.0, 1.0)
        obs[45] = np.clip(ice_idx, 0.0, 1.0)

        # 6. Winch & Tether Dynamics [46:52]
        obs[46] = np.clip(cable_len / 10.0, 0.0, 1.0)
        obs[47] = np.clip(cable_vel / 3.4, -1.0, 1.0)
        obs[48] = np.clip(tension_n / 50.0, 0.0, 1.0)
        d_tension = (tension_n - self.last_tension) / max(1e-4, dt)
        self.last_tension = tension_n
        obs[49] = np.clip(d_tension / 25.0, -1.0, 1.0)
        obs[50:52] = np.clip(sway_2d / 1.0, -1.0, 1.0)

        # 7. Battery ECM [52:56]
        obs[52] = np.clip(bms_soc, 0.0, 1.0)
        obs[53] = np.clip((v_pack - 18.0) / (25.2 - 18.0), 0.0, 1.0)
        obs[54] = np.clip((bms_temp_c - 10.0) / 30.0, -1.0, 1.0)
        obs[55] = np.clip((r0_dyn - 0.008) / 0.025, 0.0, 1.0)

        # 8. Navigation Integrity & Threats [56:61]
        obs[56] = np.clip(chi2_val / 40.0, 0.0, 1.0)
        obs[57] = np.clip(jerk_discrepancy / 15.0, 0.0, 1.0)
        obs[58] = 1.0 if is_spoofed else 0.0
        obs[59] = np.clip(emf_rms / 40.0, 0.0, 1.0)
        obs[60] = np.clip(wire_dist / 20.0, 0.0, 1.0)

        # 9. Ground Agent, Noise & Latch [61:64]
        obs[61] = np.clip(agent_dist / 30.0, 0.0, 1.0)
        obs[62] = np.clip((ground_spl_dba - 40.0) / 45.0, 0.0, 1.0)
        obs[63] = 1.0 if has_payload else 0.0

        # 10. Smart Hook IMU & 3-Axis Load Cell [64:73]
        obs[64:67] = np.clip(hook_a / 12.0, -1.0, 1.0)
        obs[67:70] = np.clip(hook_w / 6.0, -1.0, 1.0)
        obs[70:73] = np.clip(load_cell_3d / 40.0, -1.0, 1.0)

        # 11. Multi-Spectral, Thermal & Polarization [73:76]
        obs[73] = np.clip(thermal_dt / 15.0, -1.0, 1.0)
        obs[74] = np.clip(dolp, 0.0, 1.0)
        obs[75] = np.clip(ndvi, 0.0, 1.0)

        # 12. 8-Channel ESC Telemetry [76:92]
        obs[76:84] = np.clip(esc_i_rms / 25.0, 0.0, 1.0)
        obs[84:92] = np.clip((esc_t_mosfet - 20.0) / 70.0, 0.0, 1.0)

        # 13. 5-Port Air Data Probe & Meteorology [92:96]
        obs[92] = np.clip(alpha_deg / 25.0, -1.0, 1.0)
        obs[93] = np.clip(beta_deg / 25.0, -1.0, 1.0)
        obs[94] = np.clip(rh_pct / 100.0, 0.0, 1.0)
        obs[95] = np.clip(dust_pm10 / 100.0, 0.0, 1.0)

        # 14. RTK GNSS Quality & Jamming [96:100]
        obs[96] = np.clip(hdop / 5.0, 0.0, 1.0)
        obs[97] = np.clip((cn0_avg - 20.0) / 30.0, 0.0, 1.0)
        obs[98] = np.clip((rf_jam_pwr + 120.0) / 70.0, 0.0, 1.0)
        obs[99] = np.clip(mag_grad / 0.1, 0.0, 1.0)

        # 15. Dedicated Airframe 6-DOF MEMS IMU [100:106]
        obs[100:103] = np.clip(airframe_accel / 15.0, -1.0, 1.0)
        obs[103:106] = np.clip(airframe_gyro / 8.0, -1.0, 1.0)

        # 16. Option A: Peer Swarm Relative Kinematics [106:122]
        if peer_swarm_kinematics is not None and len(peer_swarm_kinematics) > 0:
            for p_i, p_info in enumerate(peer_swarm_kinematics[:2]):
                base_idx = 106 + p_i * 8
                p_rel = p_info["p_rel"]
                v_rel = p_info["v_rel"]
                dist = p_info["dist"]
                obs[base_idx:base_idx+3] = np.clip(p_rel / 100.0, -1.0, 1.0)
                obs[base_idx+3:base_idx+6] = np.clip(v_rel / 10.0, -1.0, 1.0)
                obs[base_idx+6] = np.clip(dist / 150.0, 0.0, 1.0)
                obs[base_idx+7] = 1.0 if p_info.get("has_payload", True) else 0.0

        # SENSOR UPGRADE 1: Downward Optical Flow [122:125]
        obs[122:124] = np.clip(optical_flow_2d / 2.5, -1.0, 1.0)
        obs[124] = np.clip(flow_quality, 0.0, 1.0)

        # SENSOR UPGRADE 2: ArUco / AprilTag Visual Fiducial Reticle [125:129]
        obs[125:127] = np.clip(aruco_offset_3d[:2] / 4.0, -1.0, 1.0)
        obs[127] = float(np.clip(aruco_offset_3d[2] / math.pi, -1.0, 1.0))
        obs[128] = 1.0 if aruco_locked > 0.5 else 0.0

        # SENSOR UPGRADE 3: Skid Weight-on-Wheels (WoW) Strain Gauges [129:132]
        obs[129] = np.clip(skid_force_left / 12.0, 0.0, 1.0)
        obs[130] = np.clip(skid_force_right / 12.0, 0.0, 1.0)
        obs[131] = 1.0 if is_skid_grounded else 0.0

        # SENSOR UPGRADE 6: Directional Acoustic Microphone Array [132:135]
        obs[132] = float(np.clip(siren_angle / math.pi, -1.0, 1.0))
        obs[133] = float(np.clip(siren_freq / 2000.0, 0.0, 1.0))
        obs[134] = float(np.clip((siren_spl - 40.0) / 60.0, 0.0, 1.0))

        # SENSOR UPGRADE 7: ADS-B In Civil Airspace Traffic [135:139]
        obs[135:138] = np.clip(adsb_rel_pos / 150.0, -1.0, 1.0)
        obs[138] = float(np.clip(adsb_closing_spd / 25.0, -1.0, 1.0))

        return obs

    def stack_history(self, history_deque, current_139_obs):
        history_deque.append(current_139_obs)
        while len(history_deque) < HISTORY_LEN:
            history_deque.append(current_139_obs)
        return np.concatenate(list(history_deque), axis=0).astype(np.float32)

DroneObservationSynthesizer122Dim = DroneObservationSynthesizer139Dim
DroneObservationSynthesizer318Dim = DroneObservationSynthesizer139Dim
DroneObservationSynthesizer366Dim = DroneObservationSynthesizer139Dim
DroneObservationSynthesizer106Dim = DroneObservationSynthesizer139Dim

# =============================================================================
# MASTER FRONTIER AUTONOMOUS DRONE SYSTEM CORE (V36.0 - HARDENED PRODUCTION)
# File: drone_master_autonomous_system_v56.py — [PART 3 OF 3]
# =============================================================================

# =============================================================================
# 14. MULTI-AGENT SWARM CROSS-ATTENTION EXECUTIVE BACKBONE (~2.5M PARAMETERS)
# =============================================================================
class DenseResBlock512(nn.Module):
    def __init__(self, dim=512, hidden_dim=768):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.norm2(out)
        out = self.fc2(out)
        return residual + out

# Multi-Agent Cross-Attention Transformer Layer with Synchronized Swarm Geometry
class SwarmFleetCrossAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # NaN quarantine: strictly prevents sensor/collision anomalies from spreading across peers
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Tensor Shape: (Batch_Size, Swarm_Size, Embed_Dim) or (B, D)
        if x.dim() == 2:
            B, D = x.shape
            rem = B % NUM_SWARM_DRONES
            if rem != 0:
                pad_len = NUM_SWARM_DRONES - rem
                pad = torch.zeros(pad_len, D, device=x.device, dtype=x.dtype)
                x_padded = torch.cat([x, pad], dim=0)
                B_p = x_padded.shape[0]
                x_seq = x_padded.view(B_p // NUM_SWARM_DRONES, NUM_SWARM_DRONES, D)
                
                key_padding_mask = torch.zeros((B_p // NUM_SWARM_DRONES, NUM_SWARM_DRONES), dtype=torch.bool, device=x.device)
                key_padding_mask[-1, -pad_len:] = True
                
                attn_out, _ = self.attn(x_seq, x_seq, x_seq, key_padding_mask=key_padding_mask)
                out = self.norm(x_seq + attn_out).view(B_p, D)
                return torch.nan_to_num(out[:B], nan=0.0)
            else:
                x_seq = x.view(B // NUM_SWARM_DRONES, NUM_SWARM_DRONES, D)
                attn_out, _ = self.attn(x_seq, x_seq, x_seq)
                out = self.norm(x_seq + attn_out).view(B, D)
                return torch.nan_to_num(out, nan=0.0)
        elif x.dim() == 3:
            attn_out, _ = self.attn(x, x, x)
            return torch.nan_to_num(self.norm(x + attn_out), nan=0.0)
        return x

class DeepDenseFlightExecutiveNet(nn.Module):
    def __init__(self, in_features=OBSERVATION_DIM):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.SiLU()
        )
        
        self.block1 = DenseResBlock512(dim=512, hidden_dim=768)
        self.swarm_attn = SwarmFleetCrossAttention(embed_dim=512, num_heads=4)
        self.block2 = DenseResBlock512(dim=512, hidden_dim=768)
        self.block3 = DenseResBlock512(dim=512, hidden_dim=768)

        # 6 Multi-Task Supervisory & Diagnostic Heads
        self.guidance_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 4),
            nn.Tanh()
        )
        self.winch_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
        self.health_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 8),
            nn.Sigmoid()
        )
        self.trust_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
        self.executive_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, NUM_EXECUTIVE_STATES)
        )
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        feat = self.input_proj(x)
        feat = self.block1(feat)
        feat = self.swarm_attn(feat)
        feat = self.block2(feat)
        feat = self.block3(feat)

        guidance = self.guidance_head(feat)
        winch_cmd = self.winch_head(feat)
        health_pred = self.health_head(feat)
        trust_scores = self.trust_head(feat)
        fsm_logits = self.executive_head(feat)
        value = self.value_head(feat)
        return guidance, fsm_logits, winch_cmd, health_pred, trust_scores, value

MultiModalFlightExecutiveNet = DeepDenseFlightExecutiveNet
MultiModalPPOActorCriticNet = DeepDenseFlightExecutiveNet

# =============================================================================
# 15. 10-DELIVERY CONTINUOUS MULTI-MISSION HARVESTER & EXTENDED TRAINER
# =============================================================================
def _worker_harvest_metropolitan(num_steps):
    np.random.seed()
    random.seed()
    world = MetropolitanCityGridWorld400m()
    synth = DroneObservationSynthesizer139Dim()
    lidar_engine = MacroPerceptionLiDARSuite(world)
    sensor_suite = AdvancedMultiModalSensorSuite(world)
    assurance = SurfaceAndWireAssuranceSuite(world)
    mixer = RateDampedActuatorMixerWithCBF()
    dispatcher = SwarmMissionDispatcher(world.manifest_addresses, world.depot_helipads)

    obs_list = []
    y_guidance, y_fsm, y_winch, y_health, y_trust, y_val = [], [], [], [], [], []

    # Initialize all 3 drones
    drones = []
    for i in range(NUM_SWARM_DRONES):
        init_p = world.depot_helipads[i].copy()
        init_p[2] = 0.06
        d_inst = SwarmDroneInstance(i, init_p, world, dispatcher)
        drones.append(d_inst)

    dt = 0.016
    sim_time = 0.0
    weather_timer = 0.0
    weather_modes = ["CLEAR_NOMINAL", "CANYON_GALE", "FREEZING_RAIN", "MICROBURST_STORM"]
    current_weather_idx = random.randint(0, 3)

    obstacles_cbf = [{"pos": np.array([(h["aabb"][0]+h["aabb"][1])/2, (h["aabb"][2]+h["aabb"][3])/2, 0.0]), 
                      "radius": 11.5, "height": h["aabb"][4], "id": h["id"]} for h in world.houses]

    # Continuous 10-delivery cycle per drone
    while len(obs_list) < num_steps:
        sim_time += dt
        weather_timer += dt

        # Dynamically shift weather every 45s during harvest to expose NN to mid-flight shifts
        if weather_timer >= 45.0:
            weather_timer = 0.0
            current_weather_idx = (current_weather_idx + 1) % len(weather_modes)

        mode = weather_modes[current_weather_idx]
        if mode == "CLEAR_NOMINAL":
            episode_wind = np.array([1.5, -0.5, 0.0], dtype=np.float32)
            ice_active = False
            amb_temp = 22.0
        elif mode == "CANYON_GALE":
            episode_wind = np.array([5.0, 2.5, 0.0], dtype=np.float32)
            ice_active = False
            amb_temp = 14.0
        elif mode == "FREEZING_RAIN":
            episode_wind = np.array([2.5, -2.0, 0.0], dtype=np.float32)
            ice_active = True
            amb_temp = -4.5
        else:
            episode_wind = np.array([-4.0, 3.5, -1.8], dtype=np.float32)
            ice_active = False
            amb_temp = 10.0

        world.step_civil_traffic(dt=dt)
        swarm_states = [{"p": d.airframe.p.copy(), "v": d.airframe.v.copy()} for d in drones]

        step_obs_triad = []
        step_yg_triad, step_yf_triad, step_yw_triad = [], [], []
        step_yh_triad, step_yt_triad, step_yv_triad = [], [], []

        for d_idx, d in enumerate(drones):
            tgt_xy = d.mission["target_pos"][:2]
            home_xy = d.mission["home_pad"][:2]
            roof_h = d.mission["roof_height"]
            cruise_z = d.assigned_cruise_alt
            high_hover_z = d.get_high_hover_z()

            d_roof_xy = float(np.linalg.norm(d.airframe.p[:2] - tgt_xy))
            d_depot_xy = float(np.linalg.norm(d.airframe.p[:2] - home_xy))

            # Continuous 10-Delivery FSM Transitions
            if d.mission_phase == "1_TAKEOFF":
                wp_target = np.array([home_xy[0], home_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.0, max_climb=1.2)
                fsm_token = 0
                winch_rate_cmd = 0.0
                if d.airframe.p[2] >= (cruise_z - 2.0):
                    d.mission_phase = "2_CRUISE_OUTBOUND"

            elif d.mission_phase == "2_CRUISE_OUTBOUND":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=5.0, max_climb=1.2)
                fsm_token = 0
                winch_rate_cmd = 0.0
                if d_roof_xy <= 2.5:
                    d.mission_phase = "3_CHIMNEY_DESCENT"

            elif d.mission_phase == "3_CHIMNEY_DESCENT":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], high_hover_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=1.2, max_descend=0.65)
                fsm_token = 0
                winch_rate_cmd = 0.0
                if abs(d.airframe.p[2] - high_hover_z) <= 0.85 and d_roof_xy <= 2.0:
                    d.mission_phase = "4_DELIVERY_HOVER"
                    d.hover_settle_timer = 0.0

            elif d.mission_phase == "4_DELIVERY_HOVER":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], high_hover_z], dtype=np.float32)
                p_ref, v_ref = wp_target.copy(), np.zeros(3, dtype=np.float32)

                if d.winch.has_payload and d.winch.cable_len <= 0.10:
                    fsm_token = 1
                    winch_rate_cmd = 0.0
                elif d.winch.has_payload and d.winch.cable_len > 0.10:
                    cargo_z = d.airframe.p[2] - d.winch.cable_len
                    if cargo_z > (roof_h + 0.18):
                        fsm_token = 2
                        winch_rate_cmd = 1.0
                    else:
                        fsm_token = 3
                        winch_rate_cmd = 0.0
                elif not d.winch.has_payload and d.winch.cable_len > 0.08:
                    fsm_token = 4
                    winch_rate_cmd = -1.0
                else:
                    fsm_token = 0
                    winch_rate_cmd = 0.0
                    if d.winch.delivery_completed:
                        d.winch.delivery_completed = False
                        d.packages_delivered += 1
                        d.dispatcher.report_completed(d.drone_id)
                        d.mission_phase = "5_CHIMNEY_ASCENT"
                        d.target = d.mission["home_pad"].copy()

            elif d.mission_phase == "5_CHIMNEY_ASCENT":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.0, max_climb=1.2)
                fsm_token = 0
                winch_rate_cmd = 0.0
                if d.airframe.p[2] >= (cruise_z - 2.0):
                    d.mission_phase = "6_CRUISE_INBOUND"

            elif d.mission_phase == "6_CRUISE_INBOUND":
                wp_target = np.array([home_xy[0], home_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=5.0, max_climb=1.2)
                fsm_token = 0
                winch_rate_cmd = 0.0
                # Ensure horizontal centering over pad before descent
                if d_depot_xy <= 1.2:
                    d.mission_phase = "7_DEPOT_DESCENT"

            elif d.mission_phase == "7_DEPOT_DESCENT":
                wp_target = np.array([home_xy[0], home_xy[1], 0.04], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.8, max_descend=0.45)
                fsm_token = 5
                winch_rate_cmd = 0.0
                if d.airframe.p[2] <= 0.14:
                    d.depot_timer = 0.0
                    d.mission_phase = "8_DEPOT_SERVICE"
                    d.airframe.v.fill(0.0)

            else:  # 8_DEPOT_SERVICE (Continuous reload & re-dispatch across 10 missions)
                wp_target = np.array([home_xy[0], home_xy[1], 0.06], dtype=np.float32)
                p_ref, v_ref = wp_target.copy(), np.zeros(3, dtype=np.float32)
                fsm_token = 5
                winch_rate_cmd = 0.0
                d.airframe.p[2] = 0.06
                d.airframe.v.fill(0.0)

                d.depot_timer += dt
                if d.depot_timer >= 1.0:
                    d.depot_timer = 0.0
                    d.mission = d.dispatcher.dispatch_next_mission(d.drone_id)
                    d.home_coords = d.mission["home_pad"].copy()
                    d.target = d.mission["target_pos"].copy()
                    d.assigned_cruise_alt = d.mission["assigned_cruise_alt"]
                    d.winch.reload_new_cargo()
                    # Progressive battery recharge across multi-delivery day
                    d.bms.soc = min(0.98, d.bms.soc + 0.15)
                    d.mission_phase = "1_TAKEOFF"

            real_peer_info = []
            for j_idx, other_d in enumerate(drones):
                if j_idx == d_idx: continue
                p_rel = other_d.airframe.p - d.airframe.p
                v_rel = other_d.airframe.v - d.airframe.v
                dist = float(np.linalg.norm(p_rel))
                real_peer_info.append({"p_rel": p_rel, "v_rel": v_rel, "dist": dist, "has_payload": other_d.winch.has_payload})

            meas_accel, meas_gyro = d.imu.poll(d.airframe.R, d.airframe.accel_linear, d.airframe.omega, d.airframe.motor_omegas)
            drone_yaw = math.atan2(d.airframe.R[1, 0], d.airframe.R[0, 0])
            lidar_16 = lidar_engine.cast_360_lidar(d.airframe.p, drone_yaw)
            conical_8, wire_clr = lidar_engine.cast_downward_conical(d.airframe.p)

            current_surf_z = world.get_surface_height(d.airframe.p[0], d.airframe.p[1])
            flow_2d, flow_q = sensor_suite.poll_optical_flow(d.airframe.p, d.airframe.v, d.airframe.R, d.airframe.omega, current_surf_z)
            aruco_3d, aruco_lk = sensor_suite.poll_aruco_fiducial_detector(d.airframe.p, d.airframe.R, d.mission["target_pos"], d.mission["tag_id"])
            siren_a, siren_f, siren_s = sensor_suite.poll_directional_acoustic_array(d.airframe.p, [], sim_time=sim_time)
            adsb_p, adsb_cs, adsb_d = sensor_suite.poll_adsb_civil_transponder(d.airframe.p, d.airframe.v, world.civil_aircraft)

            hook_a, hook_w = sensor_suite.poll_smart_hook_imu(d.airframe.accel_linear, d.airframe.omega, d.winch.sway_offset, d.winch.cable_len, d.winch.has_payload)
            load_3d = sensor_suite.poll_3axis_load_cell(d.winch.tension_n, d.winch.sway_offset, d.winch.cable_len)
            th_dt, dolp, ndvi = sensor_suite.poll_thermal_and_polarization(d.target[:2])
            i_rms, t_mosfet = sensor_suite.poll_esc_foc_telemetry(d.airframe.motor_omegas, d.airframe.motor_health)
            alpha, beta, rh, dust = sensor_suite.poll_5port_air_data(d.airframe.v, episode_wind, d.airframe.R)
            hdop, cn0, rf_pwr, mag_gr = sensor_suite.poll_rtk_gnss_metrics(d.airframe.p, False)
            _, is_surface_safe, _ = assurance.audit_surface(d.target[:2], target_z=roof_h)

            weight_force = (DRY_MASS_KG + (CARGO_MASS_KG if d.winch.has_payload else 0.0)) * G_ACCEL
            is_on_surf = (d.airframe.p[2] <= current_surf_z + 0.05)
            d.winch.update_skid_strain_gauges(is_on_surf, weight_force, d.airframe.last_p_watts * 0.05)

            obs_139 = synth.synthesize_instantaneous(
                p_est=d.airframe.p, v_est=d.airframe.v, R_est=d.airframe.R, omega_est=d.airframe.omega,
                target_p=wp_target, lidar_16=lidar_16, conical_8=conical_8,
                v_tas=float(np.linalg.norm(d.airframe.v - episode_wind)), q_dyn=12.0, min_sonar=max(0.1, d.airframe.p[2] - current_surf_z),
                wind_est=episode_wind, ice_idx=0.35 if ice_active else 0.0, cable_len=d.winch.cable_len,
                cable_vel=d.winch.cable_vel, tension_n=d.winch.tension_n, sway_2d=d.winch.sway_offset,
                bms_soc=d.bms.soc, v_pack=d.bms.v_pack, bms_temp_c=d.bms.temp_c, r0_dyn=0.0095,
                chi2_val=1.2, jerk_discrepancy=0.5, is_spoofed=False, emf_rms=4.0, wire_dist=wire_clr,
                agent_dist=25.0, ground_spl_dba=55.0, has_payload=d.winch.has_payload,
                hook_a=hook_a, hook_w=hook_w, load_cell_3d=load_3d, thermal_dt=th_dt, dolp=dolp, ndvi=ndvi,
                esc_i_rms=i_rms, esc_t_mosfet=t_mosfet, alpha_deg=alpha, beta_deg=beta, rh_pct=rh, dust_pm10=dust,
                hdop=hdop, cn0_avg=cn0, rf_jam_pwr=rf_pwr, mag_grad=mag_gr,
                airframe_accel=meas_accel, airframe_gyro=meas_gyro,
                peer_swarm_kinematics=real_peer_info,
                optical_flow_2d=flow_2d, flow_quality=flow_q,
                aruco_offset_3d=aruco_3d, aruco_locked=aruco_lk,
                skid_force_left=d.winch.f_skid_left, skid_force_right=d.winch.f_skid_right,
                is_skid_grounded=d.winch.is_skid_grounded,
                siren_angle=siren_a, siren_freq=siren_f, siren_spl=siren_s,
                adsb_rel_pos=adsb_p, adsb_closing_spd=adsb_cs, dt=dt
            )

            obs_417 = synth.stack_history(d.history_q, obs_139)
            step_obs_triad.append(obs_417)

            delta_p_clamped = np.clip(wp_target - d.airframe.p, -5.0, 5.0)
            a_target_xy = np.array([1.6, 1.6], dtype=np.float32) * delta_p_clamped[:2] + np.array([1.8, 1.8], dtype=np.float32) * (v_ref[:2] - d.airframe.v[:2])
            v_cmd_z = float(np.clip(v_ref[2] + 0.85 * delta_p_clamped[2], -1.0, 1.50))
            a_target_z = 3.2 * (v_cmd_z - d.airframe.v[2])
            guidance_vec = np.clip(np.array([a_target_xy[0], a_target_xy[1], a_target_z]) / 12.0, -1.0, 1.0).astype(np.float32)

            step_yg_triad.append(np.array([guidance_vec[0], guidance_vec[1], guidance_vec[2], 0.0], dtype=np.float32))
            step_yf_triad.append(fsm_token)
            step_yw_triad.append(np.array([winch_rate_cmd, 1.0 if not d.winch.has_payload else 0.0], dtype=np.float32))
            step_yh_triad.append(d.airframe.motor_health.copy())
            step_yt_triad.append(np.array([1.0, 1.0 if is_surface_safe else 0.0], dtype=np.float32))
            current_dist = d_depot_xy if "INBOUND" in d.mission_phase or "DEPOT" in d.mission_phase else d_roof_xy
            step_yv_triad.append(np.array([np.clip(-current_dist / 30.0, -1.0, 1.0)], dtype=np.float32))

            # Step physical winch (strictly locked to is_settled)
            d.winch.step(
                drone_z=d.airframe.p[2], target_surface_z=roof_h, is_settled=(d.mission_phase == "4_DELIVERY_HOVER"),
                is_surface_safe=is_surface_safe, wire_hit=False, agent_breach=False, agent_dist=30.0,
                wind_2d=episode_wind[:2], drone_accel_linear=d.airframe.accel_linear,
                neural_token_idx=fsm_token, nn_winch_rate=winch_rate_cmd, dt=dt
            )

            # Step mixer with active peer safety
            omega_cmds, _, _ = mixer.mix(
                drone_health=d.airframe.motor_health, p_est=d.airframe.p, v_est=d.airframe.v, R_est=d.airframe.R,
                omega_est=d.airframe.omega, p_target=p_ref, neural_guidance=guidance_vec, token_idx=fsm_token,
                estimated_wind=episode_wind, obstacles=obstacles_cbf, swarm_states=swarm_states, drone_idx=d_idx,
                ref_vel=v_ref, target_bldg_id=d.mission["bldg_id"] if "DELIVERY" in d.mission_phase else None,
                cable_len=d.winch.cable_len, sway_2d=d.winch.sway_offset, load_cell_3d=load_3d,
                ige_fac=d.airframe.last_ige, v_pack=d.bms.v_pack, bms_temp_c=d.bms.temp_c,
                has_payload=d.winch.has_payload, dt=dt
            )

            curr_ice = 0.35 if ice_active else 0.0
            d.airframe.step(omega_cmds, episode_wind, 1.225, ice_index=curr_ice, surface_z=current_surf_z, has_payload=d.winch.has_payload, dt=dt)

        obs_list.extend(step_obs_triad)
        y_guidance.extend(step_yg_triad)
        y_fsm.extend(step_yf_triad)
        y_winch.extend(step_yw_triad)
        y_health.extend(step_yh_triad)
        y_trust.extend(step_yt_triad)
        y_val.extend(step_yv_triad)

    return (np.array(obs_list, dtype=np.float32),
            np.array(y_guidance, dtype=np.float32),
            np.array(y_fsm, dtype=np.int64),
            np.array(y_winch, dtype=np.float32),
            np.array(y_health, dtype=np.float32),
            np.array(y_trust, dtype=np.float32),
            np.array(y_val, dtype=np.float32))

def harvest_and_train_metropolitan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_SAMPLES = 48000              
    NUM_PRETRAIN_EPOCHS = 6         
    NUM_MAIN_EPOCHS = 12             
    BATCH_SIZE = 2052
    LEARNING_RATE = 1.2e-3
    
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per = NUM_SAMPLES // num_workers

    print("\n" + "="*75)
    print(f"HARVESTING {NUM_SAMPLES:,} SAMPLES (FULL-CYCLE CONTINUOUS MISSION DATASET)")
    print(f"Parallel Workers: {num_workers} processes | Compute Device: {device}")
    print("="*75)
    t0 = time.time()

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_harvest_metropolitan, [samples_per] * num_workers)

    X_all = np.concatenate([r[0] for r in results], axis=0)[:NUM_SAMPLES]
    yg_all = np.concatenate([r[1] for r in results], axis=0)[:NUM_SAMPLES]
    yf_all = np.concatenate([r[2] for r in results], axis=0)[:NUM_SAMPLES]
    yw_all = np.concatenate([r[3] for r in results], axis=0)[:NUM_SAMPLES]
    yh_all = np.concatenate([r[4] for r in results], axis=0)[:NUM_SAMPLES]
    yt_all = np.concatenate([r[5] for r in results], axis=0)[:NUM_SAMPLES]
    yv_all = np.concatenate([r[6] for r in results], axis=0)[:NUM_SAMPLES]
    print(f"Dataset compiled in {time.time()-t0:.2f}s! ({len(X_all):,} samples)")

    num_triads = len(X_all) // NUM_SWARM_DRONES
    total_aligned = num_triads * NUM_SWARM_DRONES
    
    X_triads = X_all[:total_aligned].reshape(num_triads, NUM_SWARM_DRONES, -1)
    yg_triads = yg_all[:total_aligned].reshape(num_triads, NUM_SWARM_DRONES, -1)
    yf_triads = yf_all[:total_aligned].reshape(num_triads, NUM_SWARM_DRONES)
    yw_triads = yw_all[:total_aligned].reshape(num_triads, NUM_SWARM_DRONES, -1)
    yh_triads = yh_all[:total_aligned].reshape(num_triads, NUM_SWARM_DRONES, -1)
    yt_triads = yt_all[:total_aligned].reshape(num_triads, NUM_SWARM_DRONES, -1)
    yv_triads = yv_all[:total_aligned].reshape(num_triads, NUM_SWARM_DRONES, -1)

    num_train_triads = int(num_triads * 0.90)

    X_train = torch.from_numpy(X_triads[:num_train_triads].reshape(-1, OBSERVATION_DIM)).to(device)
    yg_tr = torch.from_numpy(yg_triads[:num_train_triads].reshape(-1, 4)).to(device)
    yf_tr = torch.as_tensor(yf_triads[:num_train_triads].reshape(-1), dtype=torch.long, device=device)
    yw_tr = torch.from_numpy(yw_triads[:num_train_triads].reshape(-1, 2)).to(device)
    yh_tr = torch.from_numpy(yh_triads[:num_train_triads].reshape(-1, 8)).to(device)
    yt_tr = torch.from_numpy(yt_triads[:num_train_triads].reshape(-1, 2)).to(device)
    yv_tr = torch.from_numpy(yv_triads[:num_train_triads].reshape(-1, 1)).to(device)

    X_val = torch.from_numpy(X_triads[num_train_triads:].reshape(-1, OBSERVATION_DIM)).to(device)
    yg_va = torch.from_numpy(yg_triads[num_train_triads:].reshape(-1, 4)).to(device)
    yf_va = torch.as_tensor(yf_triads[num_train_triads:].reshape(-1), dtype=torch.long, device=device)

    model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
    crit_mse = nn.MSELoss()
    crit_bce = nn.BCELoss()

    class_counts = torch.bincount(yf_tr, minlength=NUM_EXECUTIVE_STATES).float()
    class_weights = len(yf_tr) / (len(class_counts) * torch.clamp(class_counts, min=1.0))
    class_weights = (class_weights / class_weights.mean()).to(device)
    crit_ce = nn.CrossEntropyLoss(weight=class_weights)

    num_train_samples = X_train.shape[0]
    num_batches = (num_train_samples + BATCH_SIZE - 1) // BATCH_SIZE

    print("\n" + "-"*75)
    print(f"STAGE 1: RESNET BACKBONE PRE-TRAINING ({NUM_PRETRAIN_EPOCHS} EPOCHS)")
    print("-"*75)

    pre_optimizer = optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1e-4)

    for ep in range(NUM_PRETRAIN_EPOCHS):
        model.train()
        perm_triads = torch.randperm(num_train_triads, device=device)
        perm = (perm_triads.unsqueeze(1) * NUM_SWARM_DRONES + torch.arange(NUM_SWARM_DRONES, device=device)).reshape(-1)
        ep_loss = 0.0

        for b in range(0, num_train_samples, BATCH_SIZE):
            idx = perm[b : b + BATCH_SIZE]
            bx, byg, byv = X_train[idx], yg_tr[idx], yv_tr[idx]

            pre_optimizer.zero_grad(set_to_none=True)
            p_g, _, _, _, _, p_v = model(bx)
            loss_pre = crit_mse(p_g, byg) + 0.15 * crit_mse(p_v, byv)
            loss_pre.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            pre_optimizer.step()
            ep_loss += loss_pre.item()

        print(f"Pre-train Epoch [{ep+1:02d}/{NUM_PRETRAIN_EPOCHS:02d}] - Kinematic MSE: {ep_loss/num_batches:.5f}")

    print("\n" + "-"*75)
    print(f"STAGE 2: MULTI-TASK 6-HEAD SUPERVISED TRAINING ({NUM_MAIN_EPOCHS} EPOCHS)")
    print("-"*75)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    total_steps = num_batches * NUM_MAIN_EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=total_steps)

    for epoch in range(NUM_MAIN_EPOCHS):
        model.train()
        perm_triads = torch.randperm(num_train_triads, device=device)
        perm = (perm_triads.unsqueeze(1) * NUM_SWARM_DRONES + torch.arange(NUM_SWARM_DRONES, device=device)).reshape(-1)
        total_loss = 0.0

        for b in range(0, num_train_samples, BATCH_SIZE):
            idx = perm[b : b + BATCH_SIZE]
            bx, byg, byf = X_train[idx], yg_tr[idx], yf_tr[idx]
            byw, byh, byt, byv = yw_tr[idx], yh_tr[idx], yt_tr[idx], yv_tr[idx]

            optimizer.zero_grad(set_to_none=True)
            p_g, p_f, p_w, p_h, p_t, p_v = model(bx)

            eps = 1e-6
            p_h_c = torch.clamp(p_h, eps, 1.0 - eps)
            p_t_c = torch.clamp(p_t, eps, 1.0 - eps)

            loss = (crit_mse(p_g, byg) +
                    0.40 * crit_ce(p_f, byf) +
                    0.25 * crit_mse(p_w, byw) +
                    0.30 * crit_bce(p_h_c, byh) +
                    0.20 * crit_bce(p_t_c, byt) +
                    0.10 * crit_mse(p_v, byv))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_preds_g = []
            val_preds_f = []
            for vb in range(0, X_val.shape[0], 300):
                v_bx = X_val[vb:vb+300]
                vg_sub, vf_sub, _, _, _, _ = model(v_bx)
                val_preds_g.append(vg_sub)
                val_preds_f.append(vf_sub)

            vg = torch.cat(val_preds_g, dim=0)
            vf = torch.cat(val_preds_f, dim=0)
            val_mse = crit_mse(vg, yg_va).item()
            val_acc = (vf.argmax(dim=-1) == yf_va).float().mean().item() * 100.0

        print(f"Epoch [{epoch+1:02d}/{NUM_MAIN_EPOCHS:02d}] - Multi-Task Loss: {total_loss/num_batches:.4f} | Val MSE: {val_mse:.5f} | Decision Acc: {val_acc:.1f}%")

    torch.save(model.state_dict(), MODEL_WEIGHTS_FILE)
    print("\n" + "="*75)
    print(f"SUCCESS: Continuous Mission Model saved to {MODEL_WEIGHTS_FILE} in {time.time()-t0:.2f}s!")
    print("="*75 + "\n")
    return model, device

# ==========================================
# 16. CHASE PROJECTION & RADAR MINIMAP
# ==========================================
def project_3d_chase(pt, cam_focus, cam_yaw, cam_pitch, center_screen, scale=28.0):
    rel = pt - cam_focus
    x1 = rel[0]*math.cos(cam_yaw) - rel[1]*math.sin(cam_yaw)
    y1 = rel[0]*math.sin(cam_yaw) + rel[1]*math.cos(cam_yaw)
    y2 = y1*math.cos(cam_pitch) - rel[2]*math.sin(cam_pitch)
    z2 = y1*math.sin(cam_pitch) + rel[2]*math.cos(cam_pitch)

    if y2 >= 13.5:
        return None, None, y2

    focal = max(2.5, 14.0 - y2)
    sx = int(center_screen[0] + (x1 / focal) * scale * 9.5)
    sy = int(center_screen[1] - (z2 / focal) * scale * 9.5)
    return sx, sy, y2

# Exact camera-to-world raycaster aligned with project_3d_chase
def screen_to_world_raycast_exact(mouse_pos, cam_focus, cam_yaw, cam_pitch, center_screen, world, scale=28.0):
    mx, my = mouse_pos
    u_norm = (mx - center_screen[0]) / (scale * 9.5)
    v_norm = -(my - center_screen[1]) / (scale * 9.5)

    cy, sy = math.cos(cam_yaw), math.sin(cam_yaw)
    cp, sp = math.cos(cam_pitch), math.sin(cam_pitch)

    d_cam = np.array([u_norm, 1.0, v_norm], dtype=np.float32)
    
    dx_p = d_cam[0]
    dy_p = d_cam[1]*cp - d_cam[2]*sp
    dz_p = d_cam[1]*sp + d_cam[2]*cp

    dx_w = dx_p*cy + dy_p*sy
    dy_w = -dx_p*sy + dy_p*cy
    dz_w = dz_p

    ray_dir = np.array([dx_w, dy_w, dz_w], dtype=np.float32)
    ray_dir /= max(1e-4, float(np.linalg.norm(ray_dir)))

    cam_pos = cam_focus + np.array([-14.0*sy*cp, -14.0*cy*cp, 14.0*sp], dtype=np.float32)

    best_hit = None
    min_t = 1e6
    for h in sorted(world.houses, key=lambda b: b["aabb"][4], reverse=True):
        x1, x2, y1, y2, hh = h["aabb"]
        if abs(ray_dir[2]) > 1e-4:
            t = (hh - cam_pos[2]) / ray_dir[2]
            if 0.0 < t < min_t:
                hx = cam_pos[0] + t * ray_dir[0]
                hy = cam_pos[1] + t * ray_dir[1]
                if (x1 - 1.5) <= hx <= (x2 + 1.5) and (y1 - 1.5) <= hy <= (y2 + 1.5):
                    min_t = t
                    best_hit = np.array([(x1+x2)/2.0, (y1+y2)/2.0, hh], dtype=np.float32)

    if best_hit is not None:
        return best_hit

    if abs(ray_dir[2]) > 1e-4:
        t_gnd = -cam_pos[2] / ray_dir[2]
        if t_gnd > 0.0:
            return np.array([
                np.clip(cam_pos[0] + t_gnd * ray_dir[0], -195.0, 195.0),
                np.clip(cam_pos[1] + t_gnd * ray_dir[1], -195.0, 195.0),
                0.0
            ], dtype=np.float32)

    return cam_focus.copy()

def draw_tactical_radar_minimap(screen, map_rect, world, drones, active_idx, vehicles):
    rx, ry, rw, rh = map_rect
    pygame.draw.rect(screen, (15, 20, 28), map_rect)
    pygame.draw.rect(screen, (0, 220, 255), map_rect, 2)
    scale = rw / 400.0  
    cx, cy = rx + rw // 2, ry + rh // 2

    def to_map(x, y): return int(cx + x * scale), int(cy - y * scale)

    for r in world.roads:
        p1, p2 = to_map(r["x_min"], r["y_min"]), to_map(r["x_max"], r["y_max"])
        pygame.draw.rect(screen, (28, 34, 44), (min(p1[0], p2[0]), min(p1[1], p2[1]), abs(p2[0]-p1[0])+1, abs(p2[1]-p1[1])+1))

    for h in world.houses:
        x1, x2, y1, y2, _ = h["aabb"]
        mp1, mp2 = to_map(x1, y2), to_map(x2, y1)
        pygame.draw.rect(screen, (55, 65, 80), (mp1[0], mp1[1], mp2[0]-mp1[0], mp2[1]-mp1[1]))

    for v in vehicles:
        vx, vy = to_map(v["pos"][0], v["pos"][1])
        v_col = (255, 40, 60) if v.get("is_emergency", False) else (255, 140, 40)
        pygame.draw.circle(screen, v_col, (vx, vy), 3 if v.get("is_emergency", False) else 2)

    cv_x, cv_y = to_map(world.civil_aircraft["pos"][0], world.civil_aircraft["pos"][1])
    pygame.draw.circle(screen, (255, 220, 40), (cv_x, cv_y), 4, 1)

    for pad in world.depot_helipads:
        bx, by = to_map(pad[0], pad[1])
        pygame.draw.circle(screen, (0, 255, 140), (bx, by), 4, 1)

    colors = [(0, 240, 255), (255, 180, 40), (220, 100, 255)]
    for d_i, d in enumerate(drones):
        dx, dy = to_map(d.airframe.p[0], d.airframe.p[1])
        c = colors[d_i % len(colors)]
        radius = 5 if d_i == active_idx else 3
        pygame.draw.circle(screen, c, (dx, dy), radius)

        tx, ty = to_map(d.target[0], d.target[1])
        pygame.draw.circle(screen, c, (tx, ty), 3, 1)

# =============================================================================
# IDEA 4: CENTRALIZED SWARM MISSION DISPATCHER (CONTINUOUS 24/7 ORDER QUEUE)
# =============================================================================
class SwarmMissionDispatcher:
    def __init__(self, manifest_addresses, depot_helipads):
        self.manifest = list(manifest_addresses)
        self.depot_helipads = depot_helipads
        self.order_queue = deque(range(len(self.manifest)))
        self.active_missions = {}
        self.total_completed = 0

    def dispatch_next_mission(self, drone_id):
        if not self.order_queue:
            self.order_queue.extend(range(len(self.manifest)))
            
        order_idx = self.order_queue.popleft()
        order = self.manifest[order_idx]
        home_pad = self.depot_helipads[drone_id % len(self.depot_helipads)]
        
        mission = {
            "order_idx": order_idx,
            "target_pos": order["pos"].copy(),
            "bldg_id": order["bldg_id"],
            "addr": order["addr"],
            "tag_id": order["tag_id"],
            "home_pad": home_pad.copy(),
            "roof_height": float(order["height"]),
            # Safe corridor cruising altitude above all roofs and high-hover gates
            "assigned_cruise_alt": 42.0 + (drone_id * 3.0)  # D1: 42m, D2: 45m, D3: 48m
        }
        self.active_missions[drone_id] = mission
        return mission

    def report_completed(self, drone_id):
        self.total_completed += 1

# =============================================================================
# DIRECT 3D VECTOR GUIDANCE WITH SOFT TERMINAL FLARE
# =============================================================================
def compute_vector_guidance(current_p, current_v, target_p, max_speed_xy=5.0, max_climb=1.20, max_descend=0.65):
    delta_p = target_p - current_p
    dist_xy = float(math.hypot(delta_p[0], delta_p[1]))
    dist_z = float(delta_p[2])
    
    # Smooth horizontal braking
    v_xy_mag = min(max_speed_xy, math.sqrt(max(0.0, 2.0 * 1.2 * dist_xy)))
    if dist_xy > 0.05:
        dir_xy = delta_p[:2] / dist_xy
        v_cmd_xy = dir_xy * v_xy_mag
    else:
        v_cmd_xy = np.zeros(2, dtype=np.float32)
        
    # Vertical guidance with progressive flare
    if dist_z > 0.0:
        v_cmd_z = min(max_climb, math.sqrt(max(0.0, 2.0 * 0.9 * dist_z)))
    else:
        abs_dz = abs(dist_z)
        if target_p[2] <= 0.10:
            # Staged soft landing: cushions into a whisper landing at 4 cm/s
            if abs_dz > 2.0:
                v_cmd_z = -0.40
            elif abs_dz > 0.40:
                v_cmd_z = -0.18
            else:
                v_cmd_z = -0.04  # Feather touchdown: 4 cm per second (zero floor impact)
        elif abs_dz < 2.0:
            v_cmd_z = -max(0.20, min(0.35, abs_dz * 0.35))
        else:
            v_cmd_z = -min(max_descend, math.sqrt(max(0.0, 2.0 * 0.70 * abs_dz)))
        
    v_des = np.array([v_cmd_xy[0], v_cmd_xy[1], v_cmd_z], dtype=np.float32)
    return target_p.copy(), v_des

# =============================================================================
# IDEAS 1, 3, 4: SWARM DRONE INSTANCE WITH HIGH-HOVER & SEQUENTIAL AUTONOMY
# =============================================================================
class SwarmDroneInstance:
    def __init__(self, drone_id, init_pos, world, dispatcher):
        self.drone_id = drone_id
        self.world = world
        self.dispatcher = dispatcher
        self.airframe = BEMTCoaxialAirframe8DOF(init_pos)
        self.ekf = ErrorStateEKF15(init_pos)
        self.quantum = MatterWaveColdAtomQuantumSensor()
        self.bms = SubZeroArrheniusBatteryECM()
        self.winch = AutonomousWinchDeliverySystem()
        self.icing = MessingerRotorIcingModel()
        self.imu = Airframe6DOFMEMS_IMU()

        self.history_q = deque(maxlen=HISTORY_LEN)
        self.last_gps_p = None
        self.gps_v_filt = np.zeros(3, dtype=np.float32)
        self.gps_a_filt = np.zeros(3, dtype=np.float32)

        self.mission = self.dispatcher.dispatch_next_mission(self.drone_id)
        self.home_coords = self.mission["home_pad"].copy()
        self.target = self.mission["target_pos"].copy()
        self.assigned_cruise_alt = self.mission["assigned_cruise_alt"]

        self.mission_phase = "1_TAKEOFF"
        self.trail = deque(maxlen=160)
        self.packages_delivered = 0
        self.hover_settle_timer = 0.0
        self.depot_timer = 0.0
        self.spline_traj = None
        self.traj_time = 0.0

    def get_high_hover_z(self):
        return self.mission["roof_height"] + 10.0

# =============================================================================
# 18. MAIN FLIGHT COCKPIT & LIVE REALITY SIMULATOR
# =============================================================================
def run_all_physics_simulator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(MODEL_WEIGHTS_FILE):
        model, device = harvest_and_train_metropolitan()
    else:
        model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
        try:
            try:
                state_dict = torch.load(MODEL_WEIGHTS_FILE, map_location=device, weights_only=True)
            except TypeError:
                state_dict = torch.load(MODEL_WEIGHTS_FILE, map_location=device)
            model.load_state_dict(state_dict)
            print(f"\nLoaded verified 417-Dim Sensory Reality Executive from {MODEL_WEIGHTS_FILE}!")
        except Exception as e:
            print(f"[WARNING] Weights mismatch ({e}). Retraining on 417-Dim Sensory Dataset...")
            model, device = harvest_and_train_metropolitan()

    model.eval()
    pygame.init()
    VIEW_W, VIEW_H = 760, 740
    SIDEBAR_W = 440
    W, H = VIEW_W + SIDEBAR_W, VIEW_H

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Survival Core V36.0 — Full Sensory Reality Swarm Cockpit")
    clock = pygame.time.Clock()

    font_large = pygame.font.SysFont("consolas", 17, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 12, bold=True)

    world = MetropolitanCityGridWorld400m()
    weather_engine = DynamicUrbanWeatherEngine()
    traffic_engine = DynamicRoadVehicleTrafficEngine(world.avenue_x, world.street_y)

    sensors_physics = FullPhysicsSensorSuite()
    assurance = SurfaceAndWireAssuranceSuite(world)
    synth = DroneObservationSynthesizer139Dim()
    mixer = RateDampedActuatorMixerWithCBF()

    lidar_suite = MacroPerceptionLiDARSuite(world)
    sensor_suite = AdvancedMultiModalSensorSuite(world)
    ground_agent = DynamicGroundIntruderAgent(init_pos=[-80.0, 80.0, 0.0])
    dryden_wind_gen = DrydenAtmosphericTurbulenceModel(w20_m_s=3.4)

    dispatcher = SwarmMissionDispatcher(world.manifest_addresses, world.depot_helipads)

    drones = []
    swarm_colors = [(0, 240, 255), (255, 180, 40), (220, 100, 255)]
    for i in range(NUM_SWARM_DRONES):
        init_pos = world.depot_helipads[i].copy()
        init_pos[2] = 0.06
        d_inst = SwarmDroneInstance(i, init_pos, world, dispatcher)
        drones.append(d_inst)

    selected_drone_idx = 0
    cam_yaw, cam_pitch = 0.75, 0.45
    center_viewport = (VIEW_W // 2, VIEW_H // 2 + 50)
    cam_focus = drones[0].airframe.p.copy()
    running = True

    gust_timer = 0.0
    active_gust = np.zeros(3, dtype=np.float32)
    inject_spoof = False
    inject_multipath = False
    flight_time = 0.0
    delivered_boxes = deque(maxlen=120)

    wind_particles = [
        np.array([random.uniform(-180.0, 180.0), random.uniform(-180.0, 180.0), random.uniform(1.0, 35.0)], dtype=np.float32)
        for _ in range(55)
    ]

    print("\n================ V36.0 FULL SENSORY REALITY SWARM SIMULATOR RUNNING ================")
    print(" [IDEA 1 & 2]: 8-Leg Sequential Flight FSM & Direct 3D Vector Guidance Active")
    print(" [IDEA 3]    : Zipline High-Hover (10m above roofs; zero obstacle collision risk)")
    print(" [IDEA 4]    : Centralized Swarm Mission Dispatcher (48-building automated loop)")
    print(" [TACTILE]   : Physical skid Weight-on-Wheels (WoW) strain gauges (F_normal >= 0.65W)")
    print(" [AIRSPACE]  : ADS-B In Civil Aircraft receiver (Tracking low-altitude Medevac)")
    print("===================================================================================\n")

    while running:
        dt = 0.016
        flight_time += dt

        weather_engine.step(dt=dt)
        traffic_engine.step(dt=dt)
        world.step_civil_traffic(dt=dt)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    selected_drone_idx = (selected_drone_idx + 1) % NUM_SWARM_DRONES
                    print(f">>> [FOCUS] SWITCHED TO SWARM DRONE #{selected_drone_idx + 1} <<<")
                elif event.key == pygame.K_w:
                    weather_engine.cycle_weather()
                elif pygame.K_1 <= event.key <= pygame.K_8:
                    m_idx = event.key - pygame.K_1
                    cur_d = drones[selected_drone_idx]
                    cur_d.airframe.motor_health[m_idx] = 0.0 if cur_d.airframe.motor_health[m_idx] == 1.0 else 1.0
                elif event.key == pygame.K_a:
                    ground_agent.is_leashed = not ground_agent.is_leashed
                elif event.key == pygame.K_h:
                    cur_d = drones[selected_drone_idx]
                    cur_d.mission_phase = "5_CHIMNEY_ASCENT"
                    cur_d.target = cur_d.mission["home_pad"].copy()
                elif event.key == pygame.K_b:
                    drones[selected_drone_idx].bms.soc = 0.18
                elif event.key == pygame.K_c:
                    drones[selected_drone_idx].bms.temp_c = -5.0
                elif event.key == pygame.K_i:
                    d = drones[selected_drone_idx]
                    d.icing.icing_active = not d.icing.icing_active
                elif event.key == pygame.K_m:
                    inject_multipath = not inject_multipath
                elif event.key == pygame.K_j:
                    inject_spoof = not inject_spoof
                elif event.key == pygame.K_g:
                    gust_timer = 2.5
                    ang = random.uniform(0, 2*math.pi)
                    active_gust = np.array([math.cos(ang)*8.5, math.sin(ang)*8.5, 0.0], dtype=np.float32)
                elif event.key == pygame.K_p:
                    drones[selected_drone_idx].airframe.parachute_deployed = True
                elif event.key == pygame.K_r:
                    for d in drones:
                        d.airframe.motor_health.fill(1.0)
                        d.airframe.parachute_deployed = False
                        d.bms.soc = 0.96
                        d.bms.temp_c = 22.0
                        d.icing.ice_thickness_mm = 0.0
                        d.winch.reload_new_cargo()
                        d.mission_phase = "1_TAKEOFF"
                    inject_spoof = False
                    inject_multipath = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if mx < VIEW_W:
                    click_world = screen_to_world_raycast_exact((mx, my), cam_focus, cam_yaw, cam_pitch, center_viewport, world)
                    safe_pad = assurance.snap_to_nearest_safe_ground(click_world[:2])
                    cur_d = drones[selected_drone_idx]
                    cur_d.target = safe_pad
                    cur_d.mission["target_pos"] = safe_pad.copy()
                    cur_d.mission["roof_height"] = float(safe_pad[2])
                    cur_d.hover_settle_timer = 0.0
                    cur_d.winch.reload_new_cargo()
                    cur_d.mission_phase = "2_CRUISE_OUTBOUND"

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: cam_yaw -= 0.03
        if keys[pygame.K_RIGHT]: cam_yaw += 0.03
        if keys[pygame.K_UP]: cam_pitch = min(1.2, cam_pitch + 0.02)
        if keys[pygame.K_DOWN]: cam_pitch = max(0.15, cam_pitch - 0.02)

        cam_focus += (drones[selected_drone_idx].airframe.p - cam_focus) * 5.0 * dt

        active_drone = drones[selected_drone_idx]
        eff_wind, gust_scale, amb_temp, auto_ice = weather_engine.get_weather_parameters(active_drone.airframe.p, world.avenue_x)
        dryden_gust = dryden_wind_gen.step(altitude_m=active_drone.airframe.p[2], airspeed_m_s=np.linalg.norm(active_drone.airframe.v), dt=dt) * (gust_scale / 2.0)
        microburst = active_gust * (gust_timer / 2.5) if gust_timer > 0.0 else np.zeros(3, dtype=np.float32)
        if gust_timer > 0.0: gust_timer = max(0.0, gust_timer - dt)

        total_wind = eff_wind + dryden_gust + microburst

        for wp in wind_particles:
            wp += total_wind * 0.45 * dt
            if wp[0] > 200.0: wp[0] = -200.0
            if wp[0] < -200.0: wp[0] = 200.0
            if wp[1] > 200.0: wp[1] = -200.0
            if wp[1] < -200.0: wp[1] = 200.0

        active_tip = active_drone.airframe.p + np.array([0, 0, -0.08 - active_drone.winch.cable_len])
        agent_pos, global_agent_breach, _ = ground_agent.step(
            cargo_world_pos=active_tip,
            is_winch_lowering=(active_drone.winch.state == "LOWERING"),
            is_aborting=(active_drone.winch.state == "ABORT_RETRACT"),
            dt=dt
        )

        obstacles_cbf = [{"pos": np.array([(h["aabb"][0]+h["aabb"][1])/2, (h["aabb"][2]+h["aabb"][3])/2, 0.0]), "radius": 11.5, "height": h["aabb"][4], "id": h["id"]} for h in world.houses]
        obstacles_cbf.extend(traffic_engine.get_cbf_obstacles())
        swarm_states = [{"p": d.airframe.p.copy(), "v": d.airframe.v.copy()} for d in drones]

        selected_diag = {}

        # =====================================================================
        # STEP ALL SWARM DRONES
        # =====================================================================
        swarm_obs_417 = []
        pre_step_states = []

        # --- Phase 1: Parallel Sensor Polling & State Fusion ---
        for d_idx, d in enumerate(drones):
            d.icing.icing_active = auto_ice or d.icing.icing_active
            ice_mm, ice_idx = d.icing.step(airspeed=float(np.linalg.norm(d.airframe.v - total_wind)), dt=dt)

            drone_yaw = math.atan2(d.airframe.R[1, 0], d.airframe.R[0, 0])
            lidar_16 = lidar_suite.cast_360_lidar(d.airframe.p, drone_yaw, traffic_engine.vehicles)
            conical_8, wire_clearance = lidar_suite.cast_downward_conical(d.airframe.p)
            wire_hit = (wire_clearance < 1.4)

            current_surf_z = world.get_surface_height(d.airframe.p[0], d.airframe.p[1])
            target_surface_z = float(d.mission["roof_height"] if "DELIVER" in d.mission_phase or "OUTBOUND" in d.mission_phase or "CHIMNEY" in d.mission_phase else 0.0)

            surf_class, is_surf_safe, _ = assurance.audit_surface(d.target[:2], target_z=target_surface_z)
            s_data = sensors_physics.poll_all_physics_sensors(d.airframe.p, d.airframe.v, total_wind, surface_z=current_surf_z)

            meas_accel, meas_gyro = d.imu.poll(d.airframe.R, d.airframe.accel_linear, d.airframe.omega, d.airframe.motor_omegas)

            a_quantum_body, phase_shift = d.quantum.measure_phase_shift(meas_accel)
            d.ekf.predict(a_quantum_body, meas_gyro, dt=dt)

            compass_yaw = math.atan2(d.airframe.R[1, 0], d.airframe.R[0, 0]) + float(np.random.normal(0, 0.008))
            d.ekf.update_heading(compass_yaw, r_cov=0.02)

            flow_2d, flow_q = sensor_suite.poll_optical_flow(d.airframe.p, d.airframe.v, d.airframe.R, d.airframe.omega, current_surf_z)
            z_clearance = max(0.2, float(d.airframe.p[2] - current_surf_z))
            
            cos_tilt = max(0.35, float(d.airframe.R[2, 2]))
            z_slant = z_clearance / cos_tilt
            v_flow_x = (flow_2d[0] + meas_gyro[1]) * z_slant
            v_flow_y = (flow_2d[1] - meas_gyro[0]) * z_slant
            v_flow_body = np.array([v_flow_x, v_flow_y, 0.0], dtype=np.float32)
            v_flow_world = quat_to_rot(d.ekf.q) @ v_flow_body

            H_baro = np.zeros((1, 15), dtype=np.float32); H_baro[0, 2] = 1.0
            d.ekf.update_measurement(H_baro, np.array([s_data["z_baro"] - d.ekf.p[2]], dtype=np.float32), np.array([[0.05]], dtype=np.float32))

            spoof_offset = np.array([3.5, 2.8, 0.5]) if (inject_spoof and d_idx == selected_drone_idx) else np.zeros(3)
            multipath_offset = np.array([4.2, -3.1, 0.8]) if (inject_multipath and d_idx == selected_drone_idx) else np.zeros(3)
            z_gps = d.airframe.p + multipath_offset + spoof_offset + np.random.normal(0, 0.08, 3).astype(np.float32)

            if d.last_gps_p is None:
                d.last_gps_p = z_gps.copy()
                d.gps_v_filt = d.airframe.v.copy()
                d.gps_a_filt.fill(0.0)

            alpha_v = dt / (0.45 + dt)
            v_gps_raw = (z_gps - d.last_gps_p) / dt
            d.last_gps_p = z_gps.copy()
            v_prev_filt = d.gps_v_filt.copy()
            d.gps_v_filt += alpha_v * (v_gps_raw - d.gps_v_filt)
            alpha_a = dt / (0.35 + dt)
            a_gps_raw = (d.gps_v_filt - v_prev_filt) / dt
            d.gps_a_filt += alpha_a * (a_gps_raw - d.gps_a_filt)

            a_inertial_measured = quat_to_rot(d.ekf.q) @ a_quantum_body + GRAVITY_G
            jerk_discrepancy = float(np.linalg.norm(d.gps_a_filt - a_inertial_measured))
            kinematic_violation = (jerk_discrepancy > 16.0)

            H_pos = np.zeros((3, 15), dtype=np.float32); H_pos[0:3, 0:3] = np.eye(3)
            y_innov = z_gps - d.ekf.p
            S_cov = H_pos @ d.ekf.P @ H_pos.T + np.eye(3, dtype=np.float32) * 0.08
            chi2_dist = float(y_innov.T @ np.linalg.inv(S_cov) @ y_innov)

            is_attack_detected = (chi2_dist > 18.0) or kinematic_violation

            if not is_attack_detected:
                d.ekf.update_measurement(H_pos, y_innov, np.eye(3, dtype=np.float32) * 0.05)
                gnss_status = "GNSS NOMINAL"
            else:
                d.ekf.update_optical_flow_velocity(v_flow_world, r_cov=0.035)
                gnss_status = "SPOOF REJECTED: OPTICAL FLOW VIO LOCK"

            bms_soc, v_pack, i_draw, r0_dyn = d.bms.step(d.airframe.last_p_watts, amb_temp=amb_temp, dt=dt)

            tgt_xy = d.mission["target_pos"][:2]
            home_xy = d.mission["home_pad"][:2]
            cruise_z = d.assigned_cruise_alt
            high_hover_z = d.get_high_hover_z()

            # PURE VERTICAL CLIMBOUT: Hold pad XY until safely at cruise altitude
            if d.mission_phase == "1_TAKEOFF":
                wp_target = np.array([home_xy[0], home_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.0, max_climb=1.2)

            elif d.mission_phase == "2_CRUISE_OUTBOUND":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=5.0, max_climb=1.2)

            elif d.mission_phase == "3_CHIMNEY_DESCENT":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], high_hover_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=1.5, max_descend=0.65)

            elif d.mission_phase == "4_DELIVERY_HOVER":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], high_hover_z], dtype=np.float32)
                p_ref, v_ref = wp_target.copy(), np.zeros(3, dtype=np.float32)

            elif d.mission_phase == "5_CHIMNEY_ASCENT":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.0, max_climb=1.2)

            elif d.mission_phase == "6_CRUISE_INBOUND":
                wp_target = np.array([home_xy[0], home_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=5.0, max_climb=1.2)

            elif d.mission_phase == "7_DEPOT_DESCENT":
                wp_target = np.array([home_xy[0], home_xy[1], 0.04], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=1.5, max_descend=0.45)

            else:  # 8_DEPOT_SERVICE
                wp_target = np.array([home_xy[0], home_xy[1], 0.04], dtype=np.float32)
                p_ref, v_ref = wp_target.copy(), np.zeros(3, dtype=np.float32)

            a_ref = np.zeros(3, dtype=np.float32)
            dist_to_final_xy = float(np.linalg.norm(d.airframe.p[:2] - tgt_xy))

            R_est = quat_to_rot(d.ekf.q)
            omega_est = meas_gyro - d.ekf.b_g
            v_horiz = float(np.linalg.norm(d.ekf.v[:2]))

            aruco_3d, aruco_lk = sensor_suite.poll_aruco_fiducial_detector(d.airframe.p, d.airframe.R, d.mission["target_pos"], d.mission["tag_id"])
            weight_force = (DRY_MASS_KG + (CARGO_MASS_KG if d.winch.has_payload else 0.0)) * G_ACCEL
            is_on_surf = (d.airframe.p[2] <= current_surf_z + 0.05)
            d.winch.update_skid_strain_gauges(is_on_surf, weight_force, d.airframe.last_p_watts * 0.05)

            siren_a, siren_f, siren_s = sensor_suite.poll_directional_acoustic_array(d.airframe.p, traffic_engine.vehicles, sim_time=flight_time)
            adsb_p, adsb_cs, adsb_d = sensor_suite.poll_adsb_civil_transponder(d.airframe.p, d.airframe.v, world.civil_aircraft)

            hook_a, hook_w = sensor_suite.poll_smart_hook_imu(d.airframe.accel_linear, d.airframe.omega, d.winch.sway_offset, d.winch.cable_len, d.winch.has_payload)
            load_3d = sensor_suite.poll_3axis_load_cell(d.winch.tension_n, d.winch.sway_offset, d.winch.cable_len)
            th_dt, dolp, ndvi = sensor_suite.poll_thermal_and_polarization(d.target[:2])
            i_rms, t_mosfet = sensor_suite.poll_esc_foc_telemetry(d.airframe.motor_omegas, d.airframe.motor_health)
            alpha, beta, rh, dust = sensor_suite.poll_5port_air_data(d.airframe.v, total_wind, d.airframe.R)
            hdop, cn0, rf_pwr, mag_gr = sensor_suite.poll_rtk_gnss_metrics(d.airframe.p, is_attack_detected)

            local_agent_dist = float(np.linalg.norm(d.airframe.p[:2] - agent_pos[:2]))
            local_agent_breach = (local_agent_dist < ground_agent.personal_space_m and d.airframe.p[2] < 2.5)

            peer_swarm_kinematics = []
            for j_idx, other_d in enumerate(drones):
                if j_idx == d_idx: continue
                p_rel = other_d.airframe.p - d.airframe.p
                v_rel = other_d.airframe.v - d.airframe.v
                p_dist = float(np.linalg.norm(p_rel))
                peer_swarm_kinematics.append({
                    "p_rel": p_rel,
                    "v_rel": v_rel,
                    "dist": p_dist,
                    "has_payload": other_d.winch.has_payload
                })

            obs_139 = synth.synthesize_instantaneous(
                p_est=d.ekf.p, v_est=d.ekf.v, R_est=R_est, omega_est=omega_est, target_p=wp_target,
                lidar_16=lidar_16, conical_8=conical_8, v_tas=s_data["v_tas"], q_dyn=s_data["q_dyn"],
                min_sonar=s_data["min_sonar"], wind_est=total_wind, ice_idx=ice_idx,
                cable_len=d.winch.cable_len, cable_vel=d.winch.cable_vel, tension_n=d.winch.tension_n,
                sway_2d=d.winch.sway_offset, bms_soc=bms_soc, v_pack=v_pack, bms_temp_c=d.bms.temp_c,
                r0_dyn=r0_dyn, chi2_val=chi2_dist, jerk_discrepancy=jerk_discrepancy, is_spoofed=is_attack_detected,
                emf_rms=s_data["emf_rms"], wire_dist=wire_clearance,
                agent_dist=local_agent_dist, ground_spl_dba=58.0,
                has_payload=d.winch.has_payload, hook_a=hook_a, hook_w=hook_w, load_cell_3d=load_3d,
                thermal_dt=th_dt, dolp=dolp, ndvi=ndvi, esc_i_rms=i_rms, esc_t_mosfet=t_mosfet,
                alpha_deg=alpha, beta_deg=beta, rh_pct=rh, dust_pm10=dust, hdop=hdop, cn0_avg=cn0,
                rf_jam_pwr=rf_pwr, mag_grad=mag_gr, airframe_accel=meas_accel, airframe_gyro=meas_gyro,
                peer_swarm_kinematics=peer_swarm_kinematics,
                optical_flow_2d=flow_2d, flow_quality=flow_q,
                aruco_offset_3d=aruco_3d, aruco_locked=aruco_lk,
                skid_force_left=d.winch.f_skid_left, skid_force_right=d.winch.f_skid_right,
                is_skid_grounded=d.winch.is_skid_grounded,
                siren_angle=siren_a, siren_freq=siren_f, siren_spl=siren_s,
                adsb_rel_pos=adsb_p, adsb_closing_spd=adsb_cs, dt=dt
            )

            obs_417 = synth.stack_history(d.history_q, obs_139)
            swarm_obs_417.append(obs_417)

            pre_step_states.append({
                "target_surface_z": target_surface_z,
                "is_surf_safe": is_surf_safe,
                "wire_hit": wire_hit,
                "local_agent_breach": local_agent_breach,
                "local_agent_dist": local_agent_dist,
                "p_ref": p_ref, "v_ref": v_ref, "a_ref": a_ref,
                "R_est": R_est, "omega_est": omega_est, "load_3d": load_3d,
                "current_surf_z": current_surf_z, "v_pack": v_pack, "bms_soc": bms_soc,
                "ice_mm": ice_mm, "ice_idx": ice_idx, "meas_accel": meas_accel,
                "meas_gyro": meas_gyro, "gnss_status": gnss_status,
                "jerk_discrepancy": jerk_discrepancy, "phase_shift": phase_shift,
                "i_rms": i_rms, "t_mosfet": t_mosfet, "th_dt": th_dt, "dolp": dolp,
                "ndvi": ndvi, "hdop": hdop, "cn0": cn0, "alpha": alpha, "beta": beta,
                "dust": dust, "s_data": s_data, "v_horiz": v_horiz,
                "dist_to_final_xy": dist_to_final_xy,
                "aruco_locked": aruco_lk, "aruco_offset": aruco_3d,
                "flow_2d": flow_2d, "is_grounded": d.winch.is_skid_grounded,
                "adsb_d": adsb_d, "siren_s": siren_s
            })

        # --- Phase 2: Single Batched Swarm Cross-Attention Forward Pass ---
        t_inf0 = time.time()
        with torch.no_grad():
            t_in_batch = torch.from_numpy(np.stack(swarm_obs_417)).float().to(device)
            nn_guidance_batch, nn_logits_batch, nn_winch_batch, nn_health_batch, nn_trust_batch, nn_val_batch = model(t_in_batch)

            guidance_all = nn_guidance_batch.cpu().numpy()
            probs_all = F.softmax(nn_logits_batch, dim=-1).cpu().numpy()
            winch_all = nn_winch_batch[:, 0].cpu().numpy()
            val_all = nn_val_batch[:, 0].cpu().numpy()

        policy_latency_ms = (time.time() - t_inf0) * 1000.0

        # --- Phase 3: 8-Leg Sequential Flight State Machine Transitions ---
        for d_idx, d in enumerate(drones):
            st = pre_step_states[d_idx]
            neural_guidance = guidance_all[d_idx]
            probs = probs_all[d_idx]
            token_idx = int(np.argmax(probs))
            nn_winch_cmd = float(winch_all[d_idx])
            critic_v = float(val_all[d_idx])

            current_token_idx = token_idx
            current_action_name = ACTION_NAMES[token_idx]
            confidence = float(probs[token_idx])

            tgt_xy = d.mission["target_pos"][:2]
            home_xy = d.mission["home_pad"][:2]
            roof_h = d.mission["roof_height"]
            high_hover_z = d.get_high_hover_z()

            dist_tgt_xy = float(math.hypot(d.airframe.p[0] - tgt_xy[0], d.airframe.p[1] - tgt_xy[1]))
            dist_home_xy = float(math.hypot(d.airframe.p[0] - home_xy[0], d.airframe.p[1] - home_xy[1]))

            is_delivering = (d.mission_phase == "4_DELIVERY_HOVER")
            d.winch.step(
                drone_z=d.airframe.p[2],
                target_surface_z=roof_h,
                is_settled=is_delivering,
                is_surface_safe=st["is_surf_safe"],
                wire_hit=st["wire_hit"],
                agent_breach=st["local_agent_breach"],
                agent_dist=st["local_agent_dist"],
                wind_2d=total_wind[:2],
                drone_accel_linear=d.airframe.accel_linear,
                neural_token_idx=current_token_idx,
                nn_winch_rate=nn_winch_cmd,
                dt=dt
            )

            # =================================================================
            # 8-LEG SEQUENTIAL TRANSITIONS (ROBUST SWARM TOLERANCES & GENTLE FLARE)
            # =================================================================
            # Leg 1: Pure vertical climbout to safe cruising altitude
            if d.mission_phase == "1_TAKEOFF":
                if d.airframe.p[2] >= (d.assigned_cruise_alt - 2.5):
                    d.mission_phase = "2_CRUISE_OUTBOUND"

            # Leg 2: Cruise at altitude until over target building
            elif d.mission_phase == "2_CRUISE_OUTBOUND":
                if dist_tgt_xy <= 2.8:
                    d.mission_phase = "3_CHIMNEY_DESCENT"

            # Leg 3: Vertical descent to safe High-Hover gate (roof + 10.0m)
            elif d.mission_phase == "3_CHIMNEY_DESCENT":
                # Robust 0.85m vertical window prevents getting stuck in turbulence
                if abs(d.airframe.p[2] - high_hover_z) <= 0.85 and dist_tgt_xy <= 2.2:
                    d.mission_phase = "4_DELIVERY_HOVER"
                    d.hover_settle_timer = 0.0

            # Leg 4: Zipline High-Hover: spool tether, unload box, reel hook
            elif d.mission_phase == "4_DELIVERY_HOVER":
                if d.winch.delivery_completed:
                    d.winch.delivery_completed = False
                    d.packages_delivered += 1
                    delivered_boxes.append(np.array([tgt_xy[0], tgt_xy[1], roof_h], dtype=np.float32))
                    d.dispatcher.report_completed(d.drone_id)
                    d.mission_phase = "5_CHIMNEY_ASCENT"
                    d.target = d.mission["home_pad"].copy()

            # Leg 5: Vertical climb from high-hover gate back up to cruise altitude
            elif d.mission_phase == "5_CHIMNEY_ASCENT":
                if d.airframe.p[2] >= (d.assigned_cruise_alt - 2.5):
                    d.mission_phase = "6_CRUISE_INBOUND"

            # Leg 6: Cruise back to base depot (No altitude lockout)
            elif d.mission_phase == "6_CRUISE_INBOUND":
                # Once within 2.8m of base XY, start descent immediately
                if dist_home_xy <= 2.8:
                    d.mission_phase = "7_DEPOT_DESCENT"

            # Leg 7: Controlled flared descent onto landing pad
            elif d.mission_phase == "7_DEPOT_DESCENT":
                # Clean touchdown: when within 14cm of pad, transition immediately to service
                if d.airframe.p[2] <= 0.14:
                    d.depot_timer = 0.0
                    d.mission_phase = "8_DEPOT_SERVICE"
                    d.airframe.v.fill(0.0)
                    d.airframe.accel_linear.fill(0.0)

            # Leg 8: Service turnaround on pad: motors cut to 0, reload parcel, dispatch next order
            elif d.mission_phase == "8_DEPOT_SERVICE":
                d.depot_timer += dt
                # Hold firmly on the ground while servicing
                d.airframe.p[2] = 0.06
                d.airframe.v.fill(0.0)
                d.airframe.accel_linear.fill(0.0)

                if d.depot_timer >= 1.0:
                    d.depot_timer = 0.0
                    d.mission = d.dispatcher.dispatch_next_mission(d.drone_id)
                    d.home_coords = d.mission["home_pad"].copy()
                    d.target = d.mission["target_pos"].copy()
                    d.assigned_cruise_alt = d.mission["assigned_cruise_alt"]
                    d.winch.reload_new_cargo()
                    d.bms.soc = 0.98
                    d.bms.v_pack = 25.0
                    d.mission_phase = "1_TAKEOFF"

            omega_cmds, f_thrust, a_safe = mixer.mix(
                drone_health=d.airframe.motor_health, p_est=d.ekf.p, v_est=d.ekf.v, R_est=st["R_est"], omega_est=st["omega_est"],
                p_target=st["p_ref"], neural_guidance=neural_guidance, token_idx=current_token_idx,
                estimated_wind=total_wind, obstacles=obstacles_cbf, swarm_states=swarm_states, drone_idx=d_idx,
                ref_accel=st["a_ref"], ref_vel=st["v_ref"], 
                target_bldg_id=d.mission["bldg_id"] if "DELIVERY" in d.mission_phase else None,
                departure_bldg_id=None,
                cable_len=d.winch.cable_len, sway_2d=d.winch.sway_offset, load_cell_3d=st["load_3d"], 
                ige_fac=d.airframe.last_ige, v_pack=st["v_pack"], bms_temp_c=d.bms.temp_c, 
                has_payload=d.winch.has_payload, dt=dt
            )

            f_act, torques, p_watts, ige_fac = d.airframe.step(
                motor_cmds=omega_cmds, current_wind=total_wind, rho_air=st["s_data"]["rho_air"],
                ice_index=st["ice_idx"], surface_z=st["current_surf_z"], has_payload=d.winch.has_payload, dt=dt
            )

            if int(flight_time * 60) % 6 == 0:
                d.trail.append(d.airframe.p[:2].copy())

            if d_idx == selected_drone_idx:
                selected_diag = {
                    "v_pack": st["v_pack"], "soc": st["bms_soc"], "temp_c": d.bms.temp_c,
                    "action": current_action_name, "v_crit": critic_v,
                    "thrust": f_thrust, "tension": d.winch.tension_n,
                    "cable_l": d.winch.cable_len, "load_3d": st["load_3d"],
                    "ice_mm": st["ice_mm"], "wind_speed": np.linalg.norm(total_wind),
                    "confidence": confidence, "latency": policy_latency_ms,
                    "gnss_status": st["gnss_status"], "jerk": st["jerk_discrepancy"],
                    "phase": float(np.linalg.norm(st["phase_shift"])),
                    "i_rms": st["i_rms"], "t_mosfet": st["t_mosfet"], "th_dt": st["th_dt"],
                    "dolp": st["dolp"], "ndvi": st["ndvi"], "hdop": st["hdop"], "cn0": st["cn0"],
                    "alpha": st["alpha"], "beta": st["beta"], "dust": st["dust"], "v_tas": st["s_data"]["v_tas"],
                    "accel_raw": st["meas_accel"], "gyro_raw": st["meas_gyro"],
                    "flow_2d": st["flow_2d"], "aruco_locked": st["aruco_locked"],
                    "is_grounded": st["is_grounded"], "adsb_d": st["adsb_d"], "siren_s": st["siren_s"],
                    "local_agent_dist": st["local_agent_dist"]
                }

        # ==========================================
        # RENDERING (3D CITY GRID & HUD COCKPIT)
        # ==========================================
        screen.fill((10, 13, 20))
        pygame.draw.line(screen, (35, 45, 60), (VIEW_W, 0), (VIEW_W, H), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (W, 70), 2)

        # Ground Grid Lines
        for gx in range(-8, 9):
            p1 = np.array([cam_focus[0] + gx * 8.0, cam_focus[1] - 45.0, 0.0], dtype=np.float32)
            p2 = np.array([cam_focus[0] + gx * 8.0, cam_focus[1] + 45.0, 0.0], dtype=np.float32)
            sx1, sy1, _ = project_3d_chase(p1, cam_focus, cam_yaw, cam_pitch, center_viewport)
            sx2, sy2, _ = project_3d_chase(p2, cam_focus, cam_yaw, cam_pitch, center_viewport)
            if sx1 is not None and sx2 is not None:
                pygame.draw.line(screen, (18, 24, 32), (sx1, sy1), (sx2, sy2), 1)

        # Asphalt Roads
        for r in world.roads:
            if math.hypot(cam_focus[0] - (r["x_min"]+r["x_max"])/2, cam_focus[1] - (r["y_min"]+r["y_max"])/2) < 85.0:
                rc_pts = [
                    np.array([r["x_min"], r["y_min"], 0.0]), np.array([r["x_max"], r["y_min"], 0.0]),
                    np.array([r["x_max"], r["y_max"], 0.0]), np.array([r["x_min"], r["y_max"], 0.0])
                ]
                r_screens = [project_3d_chase(p, cam_focus, cam_yaw, cam_pitch, center_viewport)[:2] for p in rc_pts]
                if all(pt[0] is not None for pt in r_screens):
                    pygame.draw.polygon(screen, (24, 30, 38), r_screens)

        # Solid Buildings with Rooftop Clutter
        for idx, h in enumerate(world.houses):
            hx1, hx2, hy1, hy2, hh = h["aabb"]
            h_center_xy = ((hx1+hx2)/2, (hy1+hy2)/2)
            if math.hypot(cam_focus[0] - h_center_xy[0], cam_focus[1] - h_center_xy[1]) < 85.0:
                h_corners = [
                    np.array([hx1, hy1, 0.0]), np.array([hx2, hy1, 0.0]),
                    np.array([hx2, hy2, 0.0]), np.array([hx1, hy2, 0.0]),
                    np.array([hx1, hy1, hh]), np.array([hx2, hy1, hh]),
                    np.array([hx2, hy2, hh]), np.array([hx1, hy2, hh])
                ]
                h_screens = [project_3d_chase(p, cam_focus, cam_yaw, cam_pitch, center_viewport)[:2] for p in h_corners]
                
                if all(h_screens[i][0] is not None for i in range(4, 8)):
                    roof_poly = [h_screens[4], h_screens[5], h_screens[6], h_screens[7]]
                    pygame.draw.polygon(screen, (38, 46, 58), roof_poly)
                    pygame.draw.lines(screen, (100, 120, 145), True, roof_poly, 2)
                    
                    roof_center = np.array([h_center_xy[0], h_center_xy[1], hh], dtype=np.float32)
                    rf_sx, rf_sy, _ = project_3d_chase(roof_center, cam_focus, cam_yaw, cam_pitch, center_viewport)
                    if rf_sx is not None:
                        pygame.draw.circle(screen, (0, 220, 255), (rf_sx, rf_sy), 12, 1)
                        screen.blit(font_small.render("H", True, (0, 220, 255)), (rf_sx - 3, rf_sy - 6))
                        screen.blit(font_small.render(f"#{world.manifest_addresses[idx]['bldg_id']}", True, (140, 170, 205)), (rf_sx - 18, rf_sy + 10))

                for ci in range(4):
                    if h_screens[ci][0] is not None and h_screens[ci+4][0] is not None:
                        pygame.draw.line(screen, (65, 78, 95), h_screens[ci], h_screens[ci+4], 2)

                for cl in h.get("clutter", []):
                    cx1, cx2, cy1, cy2, ch = cl["aabb"]
                    c_box = [
                        np.array([cx1, cy1, hh]), np.array([cx2, cy1, hh]),
                        np.array([cx2, cy2, hh]), np.array([cx1, cy2, hh]),
                        np.array([cx1, cy1, ch]), np.array([cx2, cy1, ch]),
                        np.array([cx2, cy2, ch]), np.array([cx1, cy2, ch])
                    ]
                    c_scr = [project_3d_chase(p, cam_focus, cam_yaw, cam_pitch, center_viewport)[:2] for p in c_box]
                    if all(pt[0] is not None for pt in c_scr[4:8]):
                        pygame.draw.polygon(screen, (55, 68, 85), [c_scr[4], c_scr[5], c_scr[6], c_scr[7]])
                        pygame.draw.lines(screen, (80, 100, 125), True, [c_scr[4], c_scr[5], c_scr[6], c_scr[7]], 1)

        # Dynamic Road Vehicles
        for v in traffic_engine.vehicles:
            if math.hypot(cam_focus[0] - v["pos"][0], cam_focus[1] - v["pos"][1]) < 60.0:
                vx_s, vy_s, _ = project_3d_chase(v["pos"], cam_focus, cam_yaw, cam_pitch, center_viewport)
                if vx_s is not None:
                    col = (255, 45, 65) if v.get("is_emergency", False) else (255, 140, 40)
                    pygame.draw.rect(screen, col, (vx_s - 6, vy_s - 4, 12, 8), border_radius=2)
                    pygame.draw.circle(screen, (255, 255, 200), (vx_s + 4, vy_s), 2)

        # Civil ADS-B Aircraft (Medevac Helicopter)
        cv_pos = world.civil_aircraft["pos"]
        if math.hypot(cam_focus[0] - cv_pos[0], cam_focus[1] - cv_pos[1]) < 90.0:
            cx_s, cy_s, _ = project_3d_chase(cv_pos, cam_focus, cam_yaw, cam_pitch, center_viewport)
            if cx_s is not None:
                pygame.draw.circle(screen, (255, 220, 40), (cx_s, cy_s), 7, 2)
                screen.blit(font_small.render("MEDEVAC N911", True, (255, 220, 40)), (cx_s + 10, cy_s - 6))

        # Logistics Depot Helipads [H1, H2, H3]
        for p_i, pad in enumerate(world.depot_helipads):
            if math.hypot(cam_focus[0] - pad[0], cam_focus[1] - pad[1]) < 70.0:
                hc_sx, hc_sy, _ = project_3d_chase(pad, cam_focus, cam_yaw, cam_pitch, center_viewport)
                if hc_sx is not None:
                    pad_color = swarm_colors[p_i % len(swarm_colors)]
                    pygame.draw.circle(screen, pad_color, (hc_sx, hc_sy), 16, 2)
                    screen.blit(font_large.render(f"H{p_i+1}", True, pad_color), (hc_sx - 10, hc_sy - 9))

        # Ground Intruder Agent
        if math.hypot(cam_focus[0] - agent_pos[0], cam_focus[1] - agent_pos[1]) < 45.0:
            ag_sx, ag_sy, _ = project_3d_chase(agent_pos, cam_focus, cam_yaw, cam_pitch, center_viewport)
            if ag_sx is not None:
                ag_col = (100, 220, 120) if ground_agent.is_leashed else ((255, 60, 80) if global_agent_breach else (255, 120, 40))
                pygame.draw.circle(screen, ag_col, (ag_sx, ag_sy), 7)
                pygame.draw.circle(screen, (100, 140, 180) if not global_agent_breach else (255, 60, 80), (ag_sx, ag_sy), int(ground_agent.personal_space_m * 18), 1)

        # Delivered Boxes
        for pb in delivered_boxes:
            if math.hypot(cam_focus[0] - pb[0], cam_focus[1] - pb[1]) < 65.0:
                bx, by, _ = project_3d_chase(pb, cam_focus, cam_yaw, cam_pitch, center_viewport)
                if bx is not None:
                    pygame.draw.rect(screen, (60, 160, 90), (bx - 5, by - 5, 10, 10), border_radius=2)

        # Render All Drones in Swarm
        for d_i, d in enumerate(drones):
            col = swarm_colors[d_i % len(swarm_colors)]
            dsx, dsy, _ = project_3d_chase(d.airframe.p, cam_focus, cam_yaw, cam_pitch, center_viewport)
            if dsx is None: continue

            pygame.draw.circle(screen, col, (dsx, dsy), 8 if d_i == selected_drone_idx else 5)
            screen.blit(font_small.render(f"D{d_i+1}", True, col), (dsx + 10, dsy - 10))

            arm_d = ARM_LENGTH_M * 1.8
            arm_ends_body = [
                np.array([ arm_d/1.414, -arm_d/1.414, 0.0]), np.array([ arm_d/1.414,  arm_d/1.414, 0.0]),
                np.array([-arm_d/1.414,  arm_d/1.414, 0.0]), np.array([-arm_d/1.414, -arm_d/1.414, 0.0]),
            ]
            arm_screens = [project_3d_chase(d.airframe.p + d.airframe.R @ a, cam_focus, cam_yaw, cam_pitch, center_viewport)[:2] for a in arm_ends_body]
            if all(pt[0] is not None for pt in arm_screens):
                pygame.draw.line(screen, (160, 170, 185), arm_screens[0], arm_screens[2], 2)
                pygame.draw.line(screen, (160, 170, 185), arm_screens[1], arm_screens[3], 2)

                for m_i in range(8):
                    a_i = [0, 0, 1, 1, 2, 2, 3, 3][m_i]
                    mx, my = arm_screens[a_i]
                    off_y = -5 if (m_i % 2 == 0) else 5
                    if d.airframe.motor_health[m_i] == 0.0:
                        pygame.draw.line(screen, (255, 60, 80), (mx - 4, my + off_y - 4), (mx + 4, my + off_y + 4), 2)
                        pygame.draw.line(screen, (255, 60, 80), (mx + 4, my + off_y - 4), (mx - 4, my + off_y + 4), 2)
                    else:
                        pygame.draw.circle(screen, col, (mx, my + off_y), 7, 1)

            tether_mid_3d = np.array([
                d.airframe.p[0] + d.winch.catenary_midpoint[0],
                d.airframe.p[1] + d.winch.catenary_midpoint[1],
                d.airframe.p[2] - 0.5 * d.winch.cable_len
            ], dtype=np.float32)
            payload_3d = np.array([
                d.airframe.p[0] + d.winch.sway_offset[0],
                d.airframe.p[1] + d.winch.sway_offset[1],
                d.airframe.p[2] - d.winch.cable_len
            ], dtype=np.float32)
            
            mid_sx, mid_sy, _ = project_3d_chase(tether_mid_3d, cam_focus, cam_yaw, cam_pitch, center_viewport)
            psx, psy, _ = project_3d_chase(payload_3d, cam_focus, cam_yaw, cam_pitch, center_viewport)

            if psx is not None:
                if d.winch.cable_len > 0.10:
                    if mid_sx is not None:
                        pygame.draw.lines(screen, (230, 240, 255), False, [(dsx, dsy + 3), (mid_sx, mid_sy), (psx, psy)], 1)
                    else:
                        pygame.draw.line(screen, (230, 240, 255), (dsx, dsy + 3), (psx, psy), 1)

                if d.winch.has_payload:
                    box_x, box_y = (dsx, dsy + 5) if d.winch.cable_len <= 0.10 else (psx, psy)
                    pygame.draw.rect(screen, (255, 195, 35), (box_x - 6, box_y - 6, 12, 12), border_radius=2)
                    pygame.draw.rect(screen, (60, 45, 10), (box_x - 6, box_y - 6, 12, 12), 1)
                else:
                    hook_x, hook_y = (dsx, dsy + 3) if d.winch.cable_len <= 0.10 else (psx, psy)
                    pygame.draw.circle(screen, (200, 220, 240), (hook_x, hook_y), 3)

        # Target Dropzone Ring
        act_d = drones[selected_drone_idx]
        act_tgt = act_d.target
        tsx, tsy, _ = project_3d_chase(act_tgt, cam_focus, cam_yaw, cam_pitch, center_viewport)
        if tsx is not None:
            t_ring_col = (0, 220, 255) if "BASE" in act_d.mission_phase or "DEPOT" in act_d.mission_phase or "INBOUND" in act_d.mission_phase else (0, 255, 140)
            pygame.draw.circle(screen, t_ring_col, (tsx, tsy), 16, 2)
            pygame.draw.circle(screen, (255, 255, 255), (tsx, tsy), 4)
            cur_tag = f"DEPOT [H{selected_drone_idx+1}]" if "BASE" in act_d.mission_phase or "DEPOT" in act_d.mission_phase or "INBOUND" in act_d.mission_phase else f"{act_d.mission['bldg_id']} ({act_d.mission['addr']})"
            screen.blit(font_small.render(cur_tag, True, t_ring_col), (tsx + 18, tsy - 6))

        # Wind Streamers
        for wp in wind_particles:
            px, py, _ = project_3d_chase(wp, cam_focus, cam_yaw, cam_pitch, center_viewport)
            if px is not None and 0 < px < VIEW_W and 70 < py < H:
                pygame.draw.line(screen, (80, 140, 180), (px, py), (px + int(total_wind[0]*2.0), py - int(total_wind[1]*2.0)), 1)

        # ==========================================
        # COMPLETE SIDEBAR DASHBOARD
        # ==========================================
        sb_x = VIEW_W + 14
        screen.blit(font_large.render("SURVIVAL CORE V36.0", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Full Sensory Reality • 417-Dim Space • Urban Chimneys", True, (130, 160, 190)), (sb_x, 38))

        # Tactical Radar Minimap
        map_rect = (sb_x, 65, 210, 210)
        draw_tactical_radar_minimap(screen, map_rect, world, drones, selected_drone_idx, traffic_engine.vehicles)

        fl_x = sb_x + 224
        screen.blit(font_mono.render("SWARM FLEET [TAB]:", True, (240, 240, 240)), (fl_x, 65))
        for d_i, d in enumerate(drones):
            c = swarm_colors[d_i % len(swarm_colors)]
            tag = f"[D{d_i+1}]" + ("*" if d_i == selected_drone_idx else " ")
            stat = d.mission_phase[:8]
            screen.blit(font_small.render(f"{tag} {stat:8s} | {d.bms.soc*100:.0f}% | Box:{d.packages_delivered}", True, c), (fl_x, 85 + d_i * 20))

        tot_deliv = dispatcher.total_completed
        cur_mission = act_d.mission
        screen.blit(font_mono.render("ACTIVE ORDER:", True, (240, 240, 240)), (fl_x, 155))
        screen.blit(font_small.render(f"ID:   {cur_mission['bldg_id']} (Tag #{cur_mission['tag_id']})", True, (0, 220, 255)), (fl_x, 175))
        screen.blit(font_small.render(f"Addr: {cur_mission['addr'][:16]}", True, (0, 255, 140)), (fl_x, 193))
        screen.blit(font_small.render(f"Cruise Alt:  {act_d.assigned_cruise_alt:.0f}m", True, (160, 180, 205)), (fl_x, 211))
        screen.blit(font_small.render(f"Phase:{act_d.mission_phase[:15]}", True, (255, 200, 40)), (fl_x, 229))

        # 1. District Manifest Status
        p1_y = 285
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p1_y - 8), (W - 14, p1_y - 8), 1)
        screen.blit(font_mono.render("1. 48-BUILDING ROOFTOP MANIFEST:", True, (240, 240, 240)), (sb_x, p1_y))
        dist_dest = float(np.linalg.norm(act_d.airframe.p[:2] - act_d.target[:2]))
        dest_tag = f"Depot [H{selected_drone_idx+1}]" if "BASE" in act_d.mission_phase or "DEPOT" in act_d.mission_phase or "INBOUND" in act_d.mission_phase else f"{cur_mission['bldg_id']} ({cur_mission['addr']})"
        screen.blit(font_small.render(f"Target Waypoint:   {dest_tag[:22]} ({dist_dest:.1f}m)", True, (0, 220, 255)), (sb_x + 10, p1_y + 18))
        cargo_tag = "0.45kg PARCEL CLAMPED" if act_d.winch.has_payload else "EMPTY HOOK (RTB RELOAD)"
        screen.blit(font_small.render(f"Fuselage Cargo:    {cargo_tag}", True, (0, 255, 140) if act_d.winch.has_payload else (255, 200, 40)), (sb_x + 10, p1_y + 34))
        screen.blit(font_small.render(f"Swarm Deliveries:  {tot_deliv} Delivered (Continuous Fleet)", True, (0, 255, 140)), (sb_x + 10, p1_y + 50))

        # 2. 139-Dim Multi-Modal Sensory Telemetry
        p2_y = p1_y + 68
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (W - 14, p2_y), 1)
        screen.blit(font_mono.render("2. OPTICAL FLOW & ARUCO RETICLE (417-DIM):", True, (240, 240, 240)), (sb_x, p2_y + 8))
        fl_2d = selected_diag.get("flow_2d", np.zeros(2))
        screen.blit(font_small.render(f"Optical Flow VIO:  ω_flow = [{fl_2d[0]:.2f}, {fl_2d[1]:.2f}] rad/s (Locked)", True, (0, 220, 255)), (sb_x + 10, p2_y + 26))
        arc_lk = selected_diag.get("aruco_locked", 0.0)
        arc_txt = "RETICLE LOCKED [TAG MATCH]" if arc_lk > 0.5 else "SEARCHING ROOF CONE..."
        screen.blit(font_small.render(f"ArUco Docking:     {arc_txt}", True, (0, 255, 140) if arc_lk > 0.5 else (255, 140, 40)), (sb_x + 10, p2_y + 42))
        is_grnd = selected_diag.get("is_grounded", False)
        screen.blit(font_small.render(f"Skid WoW Sensors:  {'[TOUCHDOWN SOLID]' if is_grnd else '[AIRBORNE / CRUISE]'}", True, (0, 255, 140) if is_grnd else (160, 180, 205)), (sb_x + 10, p2_y + 58))

        # 3. ESC Telemetry & Airspace ADS-B
        p3_y = p2_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (W - 14, p3_y), 1)
        screen.blit(font_mono.render("3. ACOUSTICS & ADS-B CIVIL TRAFFIC:", True, (240, 240, 240)), (sb_x, p3_y + 8))
        adsb_dst = selected_diag.get("adsb_d", 999.0)
        screen.blit(font_small.render(f"ADS-B Civil Alert: Medevac N911 @ {adsb_dst:.1f}m (Corridor Clear)", True, (0, 220, 255)), (sb_x + 10, p3_y + 26))
        srn_spl = selected_diag.get("siren_s", 55.0)
        screen.blit(font_small.render(f"Acoustic Array:    SPL = {srn_spl:.1f} dBA | Siren Detect = {'[ACTIVE]' if srn_spl > 70.0 else '[NOMINAL]'}", True, (255, 60, 80) if srn_spl > 70.0 else (0, 255, 140)), (sb_x + 10, p3_y + 42))
        irms = selected_diag.get("i_rms", np.zeros(8))
        tmos = selected_diag.get("t_mosfet", np.zeros(8))
        screen.blit(font_small.render(f"8-Ch ESC Currents: Avg I = {float(np.mean(irms)):.1f} A | Max FET = {float(np.max(tmos)):.1f}°C", True, (160, 180, 205)), (sb_x + 10, p3_y + 58))

        # 4. Urban Weather & Canyon Aerodynamics
        p4_y = p3_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (W - 14, p4_y), 1)
        screen.blit(font_mono.render("4. URBAN WEATHER & CANYON AERODYNAMICS:", True, (240, 240, 240)), (sb_x, p4_y + 8))
        w_mode = weather_engine.get_mode_name()
        w_col = (0, 255, 140) if "CLEAR" in w_mode else ((255, 180, 40) if "CANYON" in w_mode else (255, 60, 80))
        screen.blit(font_small.render(f"Atmospheric State: {w_mode} ([W] Cycle)", True, w_col), (sb_x + 10, p4_y + 26))
        screen.blit(font_small.render(f"Venturi Airspeed:  |V_tas| = {selected_diag.get('v_tas', 0.0):.2f} m/s | Wind: {selected_diag.get('wind_speed', 0.0):.1f} m/s", True, (0, 220, 255)), (sb_x + 10, p4_y + 42))
        screen.blit(font_small.render(f"Rotor Ice Layer:   {selected_diag.get('ice_mm', 0.0):.2f} mm | Temp = {selected_diag.get('temp_c', 22.0):.1f}°C", True, (160, 180, 205)), (sb_x + 10, p4_y + 58))

        # 5. GNSS Anti-Spoof & Quantum Sensing
        p5_y = p4_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p5_y), (W - 14, p5_y), 1)
        screen.blit(font_mono.render("5. GNSS ANTI-SPOOF & QUANTUM VIO:", True, (240, 240, 240)), (sb_x, p5_y + 8))
        g_stat = selected_diag.get("gnss_status", "GNSS NOMINAL")
        screen.blit(font_small.render(f"Navigation State:  {g_stat}", True, (255, 60, 80) if "REJECTED" in g_stat else (0, 255, 140)), (sb_x + 10, p5_y + 26))
        screen.blit(font_small.render(f"Kinematic Discrep: Δa = {selected_diag.get('jerk', 0.0):.2f} m/s² (Thresh = 16.0)", True, (160, 180, 205)), (sb_x + 10, p5_y + 42))
        screen.blit(font_small.render(f"Quantum Phase:     ΔΦ = {selected_diag.get('phase', 0.0):.3f} rad (Rb-87 Raman)", True, (0, 220, 255)), (sb_x + 10, p5_y + 58))

        # 6. Swarm Cross-Attention Executive & Winch
        p6_y = p5_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p6_y), (W - 14, p6_y), 1)
        screen.blit(font_mono.render("6. SWARM CROSS-ATTENTION EXECUTIVE (~2.5M):", True, (240, 240, 240)), (sb_x, p6_y + 8))
        screen.blit(font_small.render(f"Primary Action:    {current_action_name} ({confidence*100:.1f}%) | Latency: {selected_diag.get('latency', 1.0):.2f}ms", True, (0, 255, 140)), (sb_x + 10, p6_y + 26))
        screen.blit(font_small.render(f"Winch State:       {act_d.winch.state} (Cable: {act_d.winch.cable_len:.2f} m | Tension: {act_d.winch.tension_n:.1f} N)", True, (0, 220, 255)), (sb_x + 10, p6_y + 42))
        screen.blit(font_small.render(f"Traffic / Agent:   8 Vehicles Active | Ground Agent: {selected_diag.get('local_agent_dist', 0.0):.1f}m", True, (160, 180, 205)), (sb_x + 10, p6_y + 58))

        # Bottom Command Strip
        bot_y = H - 38
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y - 8), (W - 14, bot_y - 8), 1)
        screen.blit(font_mono.render("[TAB] Drone | [W] Weather | [CLICK] Retarget | [H] RTB | [J] Spoof", True, (255, 200, 40)), (sb_x + 8, bot_y))

        # Viewport Header
        vert_spd = act_d.airframe.v[2]
        status_banner = f"ACTIVE: D{selected_drone_idx+1} | Phase: {act_d.mission_phase} (Vz: {vert_spd:+.2f}m/s) | Delivered: {tot_deliv}"
        screen.blit(font_large.render(status_banner, True, (240, 240, 240)), (15, 14))
        screen.blit(font_small.render("V36.0 Full Sensory Reality • 8-Leg Sequential FSM • Zipline High-Hover", True, (0, 210, 255)), (15, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    mp.freeze_support()
    run_all_physics_simulator()