# =============================================================================
# MASTER FRONTIER AUTONOMOUS DRONE SYSTEM CORE (V17.3 - METROPOLITAN RESTORED)
# File: drone_master_autonomous_system_v17.py
# =============================================================================
# - 400m Metropolitan City Grid: 10 Avenues/Streets, 48 Multi-Story Buildings
# - 48 Individual Building Addresses with Verified Courtyard Dropzones
# - Dynamic Weather Engine: Clear, Canyon Gale (Venturi), Freezing Rain, Storm
# - Dynamic Obstacles: Moving Road Traffic Vehicles & Ground Pedestrian Intruder
# - Complete 2-Stage Multi-Processing Training Engine (Harvest -> Train -> Save/Load)
# - Neural Dominance: ResNet-Backbone with 5 Multi-Task Heads (100-Dim Perception)
# - Controls: [CLICK] Retarget, [W] Weather, [1-8] Motor Kill, [A] Leash, [H] RTB,
#             [B] Low Batt, [C] Cold-Soak, [I] Ice, [M] Multipath, [J] Spoof,
#             [G] Gust, [P] Parachute, [R] Reset, [Arrows] Camera Orbit
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

INERTIA_MATRIX = np.diag([0.022, 0.022, 0.038]).astype(np.float32)
INV_INERTIA = np.linalg.inv(INERTIA_MATRIX)

C_T_CLEAN = 1.25e-5
C_Q_CLEAN = 2.40e-7
COAXIAL_EFFICIENCY = 0.88
MOTOR_TAU_S = 0.030
MAX_RPM = 9200.0
MAX_OMEGA = (MAX_RPM * 2.0 * math.pi) / 60.0

HOVER_THRUST_NOMINAL = MASS_KG * G_ACCEL
HOVER_OMEGA_NOMINAL = math.sqrt((HOVER_THRUST_NOMINAL / 8.0) / (C_T_CLEAN * 0.94))

NUM_BLADE_ELEMENTS = 8
PRANDTL_TIP_FACTOR = 0.92
COAXIAL_DOWNWASH_INGESTION = 0.82

CEILING_HEIGHT_M = 45.0
CBF_ALPHA_OBSTACLE = 1.8
CBF_SAFE_RADIUS_M = 0.85

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

# Stable Tether Delivery Constants
PAYLOAD_WEIGHT_N = CARGO_MASS_KG * G_ACCEL
GUILLOTINE_THRESHOLD_N = 48.0
HOVER_LOCK_RADIUS_M = 0.90
HOVER_LOCK_SPEED_MAX_MS = 0.40
HOVER_SETTLE_DURATION_S = 0.30

# 100-Dimensional Multi-Modal Sensor Space
OBSERVATION_DIM = 100
NUM_EXECUTIVE_STATES = 6
MODEL_WEIGHTS_FILE = "drone_master_brain_v17_metropolitan.pth"

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
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float32)

def vee_map(mat):
    return np.array([mat[2, 1], mat[0, 2], mat[1, 0]], dtype=np.float32)

def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z),       2.0*(x*z + w*y)],
        [2.0*(x*y + w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x)],
        [2.0*(x*z - w*y),       2.0*(y*z + w*x),       1.0 - 2.0*(x*x + y*y)]
    ], dtype=np.float32)

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float32)

# ==========================================
# 2. METROPOLITAN CITY GRID & 48 BUILDINGS
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

        self.base_station = np.array([-155.0, -155.0, 0.0], dtype=np.float32)
        self.houses = []
        self.manifest_addresses = []
        self._build_48_buildings_with_addresses()

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

    def _build_48_buildings_with_addresses(self):
        street_names = ["Broadway", "5th Ave", "Park Ave", "Lexington", "Madison", "Elm St", "Market St", "Grand Blvd"]
        bldg_counter = 0

        col_ranges = [(-175.0, -135.0), (-105.0, -75.0), (-45.0, -15.0), (15.0, 45.0), (75.0, 105.0), (135.0, 175.0)]
        row_ranges = [(-175.0, -135.0), (-105.0, -75.0), (-45.0, -15.0), (15.0, 45.0), (75.0, 105.0), (135.0, 175.0)]

        for c_idx, (cx1, cx2) in enumerate(col_ranges):
            for r_idx, (ry1, ry2) in enumerate(row_ranges):
                if c_idx == 0 and r_idx == 0:
                    continue  # Helipad Base Depot
                if c_idx == 2 and r_idx == 2:
                    continue  # Central Park plaza

                if bldg_counter >= 48:
                    break

                bldg_w = min(20.0, cx2 - cx1 - 6.0)
                bldg_d = min(22.0, ry2 - ry1 - 6.0)
                bx1 = cx1 + 3.0
                bx2 = bx1 + bldg_w
                by1 = ry1 + 3.0
                by2 = by1 + bldg_d

                tier = (c_idx + r_idx) % 4
                if tier == 0:
                    height = 7.5   # 2-story retail
                elif tier == 1:
                    height = 14.0  # 4-story commercial
                elif tier == 2:
                    height = 22.0  # 7-story residential
                else:
                    height = 28.0  # 9-story tower

                bldg_id = f"Bldg_{bldg_counter+1:02d}"
                self.houses.append({"id": bldg_id, "aabb": [bx1, bx2, by1, by2, height]})

                street = street_names[(c_idx + r_idx) % len(street_names)]
                num = (bldg_counter + 1) * 10 + 2

                # Verified safe delivery courtyard right outside entrance
                drop_x = float(bx2 + 3.5) if bx2 < 170.0 else float(bx1 - 3.5)
                drop_y = float((by1 + by2) / 2.0)
                drop_pos = np.array([drop_x, drop_y, 0.0], dtype=np.float32)

                self.manifest_addresses.append({
                    "bldg_id": bldg_id,
                    "addr": f"{num} {street}",
                    "pos": drop_pos,
                    "height": height
                })
                bldg_counter += 1

# ==========================================
# 3. DYNAMIC WEATHER & ROAD VEHICLES
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
        print(f"\n>>> [WEATHER CHANGE] CURRENT REGIME: {self.get_mode_name()} <<<")

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
            microburst_w = -3.2 if pz > 3.0 else -0.8

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
                "axis": "Y"
            })
        for i in range(4):
            sy = streets_y[i % len(streets_y)]
            self.vehicles.append({
                "id": f"Truck_{i+1}",
                "pos": np.array([-180.0 + i * 90.0, sy, 1.1], dtype=np.float32),
                "vel": np.array([4.2 + i*0.5, 0.0, 0.0], dtype=np.float32),
                "axis": "X"
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
        return [{"pos": v["pos"].copy(), "radius": 2.6} for v in self.vehicles]

# ==========================================
# 4. SENSORS & 100-DIM MULTI-MODAL SUITE
# ==========================================
class AdvancedMultiModalSensorSuite:
    def __init__(self, world: MetropolitanCityGridWorld400m):
        self.world = world

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
        for h in self.world.houses:
            x1, x2, y1, y2, _ = h["aabb"]
            if x1 <= px <= x2 and y1 <= py <= y2:
                return 8.2, 0.45, 0.08  
        return 0.5, 0.12, 0.74  

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
        return math.degrees(alpha_rad), math.degrees(beta_rad), 68.0, 22.0

    def poll_rtk_gnss_metrics(self, drone_p, is_spoofed):
        hdop = 3.8 if is_spoofed else 0.85
        cn0_avg = 28.5 if is_spoofed else 44.0
        rf_jam_pwr = -52.0 if is_spoofed else -110.0
        return hdop, cn0_avg, rf_jam_pwr, 0.02

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
                if pz <= hh:
                    tx1 = (x1 - px) / dx if abs(dx) > 1e-4 else -1e6
                    tx2 = (x2 - px) / dx if abs(dx) > 1e-4 else 1e6
                    ty1 = (y1 - py) / dy if abs(dy) > 1e-4 else -1e6
                    ty2 = (y2 - py) / dy if abs(dy) > 1e-4 else 1e6

                    t_enter = max(min(tx1, tx2), min(ty1, ty2))
                    t_exit = min(max(tx1, tx2), max(ty1, ty2))
                    if t_exit >= t_enter and t_enter > 0.0:
                        min_dist = min(min_dist, t_enter)

            if dynamic_vehicles:
                for v in dynamic_vehicles:
                    d_veh = math.hypot(px - v["pos"][0], py - v["pos"][1])
                    if pz < 3.0 and d_veh < min_dist:
                        min_dist = min(min_dist, d_veh)

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

        cone_slant = math.cos(math.radians(25.0))
        for i in range(8):
            conical_ranges[i] = min(15.0, float(pz / max(0.1, cone_slant)))

        return conical_ranges, wire_clearance

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
        return self.p.copy(), (dist < self.personal_space_m), dist

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
        du = (- (v / l_u) * self.gust_state[0] + sigma_u * math.sqrt(2.0 * v / l_u) * white_noise[0]) * dt
        dv = (- (v / l_v) * self.gust_state[1] + sigma_v * math.sqrt(2.0 * v / l_v) * white_noise[1]) * dt
        dw = (- (v / l_w) * self.gust_state[2] + sigma_w * math.sqrt(2.0 * v / l_w) * white_noise[2]) * dt

        self.gust_state[0] += du
        self.gust_state[1] += dv
        self.gust_state[2] += dw
        self.gust_state = np.clip(self.gust_state, -8.0, 8.0)
        return self.gust_state.copy()

class SubZeroArrheniusBatteryECM:
    def __init__(self):
        self.soc = 0.96
        self.temp_c = 22.0
        self.v1 = np.zeros(NUM_CELLS, dtype=np.float32)
        self.capacity_coulombs = PACK_CAPACITY_AH * 3600.0
        self.v_pack = 23.8

    def step(self, p_elec_watts, dt=0.016):
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
        self.temp_c += ((p_joule - 0.35 * (self.temp_c - 15.0)) / 350.0) * dt
        return self.soc, self.v_pack, i_draw, r0

# ==========================================
# 7. WINCH DELIVERY SYSTEM WITH NN LOGIC
# ==========================================
class AutonomousWinchDeliverySystem:
    def __init__(self):
        self.cable_len = 0.08
        self.cable_vel = 0.0
        self.tension_n = PAYLOAD_WEIGHT_N
        self.has_payload = True
        self.touchdown_timer = 0.0
        self.severed = False
        self.sway_offset = np.zeros(2, dtype=np.float32)
        self.state = "STANDBY"
        self.delivery_completed = False

    def reload_new_cargo(self):
        self.cable_len = 0.08
        self.cable_vel = 0.0
        self.has_payload = True
        self.state = "STANDBY"
        self.tension_n = PAYLOAD_WEIGHT_N
        self.touchdown_timer = 0.0
        self.delivery_completed = False
        self.sway_offset.fill(0.0)

    def step(self, drone_z, is_settled, is_surface_safe, wire_hit, agent_breach, agent_dist, wind_2d, neural_token_idx, dt=0.016):
        if self.severed:
            self.state = "GUILLOTINE"
            return self.cable_len, 0.0, np.zeros(2, dtype=np.float32)

        if self.cable_len > 0.12:
            target_sway = wind_2d * (self.cable_len * 0.06)
            self.sway_offset += (target_sway - self.sway_offset) * 4.0 * dt
        else:
            self.sway_offset.fill(0.0)

        dropzone_occupied = (agent_dist < 2.8) and (self.state == "STANDBY")
        lockout = (not is_surface_safe) or wire_hit or (not is_settled) or dropzone_occupied

        if agent_breach and self.state in ["LOWERING", "TOUCHDOWN"]:
            self.state = "ABORT_RETRACT"

        if self.state == "STANDBY":
            self.cable_len = 0.08
            self.cable_vel = 0.0
            self.tension_n = PAYLOAD_WEIGHT_N if self.has_payload else 1.2
            if not lockout and not agent_breach and self.has_payload and (neural_token_idx == 2 or is_settled):
                self.state = "LOWERING"

        elif self.state == "LOWERING":
            if lockout:
                self.state = "RETRACTING"
            else:
                self.cable_vel = 2.40
                self.cable_len += self.cable_vel * dt
                self.tension_n = PAYLOAD_WEIGHT_N + float(np.random.normal(0, 0.15))

                cargo_z = drone_z - self.cable_len
                if cargo_z <= 0.06 or neural_token_idx == 3:
                    self.state = "TOUCHDOWN"
                    self.touchdown_timer = 0.0

        elif self.state == "TOUCHDOWN":
            self.cable_vel = 0.0
            self.tension_n = 1.8 + float(np.random.normal(0, 0.08))
            self.touchdown_timer += dt
            if self.touchdown_timer >= 0.40 or neural_token_idx == 4:
                self.has_payload = False
                self.state = "RETRACTING"

        elif self.state in ["RETRACTING", "ABORT_RETRACT"]:
            self.cable_vel = -3.20
            self.cable_len = max(0.08, self.cable_len + self.cable_vel * dt)
            self.tension_n = 1.2 + float(np.random.normal(0, 0.05))

            if self.cable_len <= 0.08:
                self.cable_len = 0.08
                self.cable_vel = 0.0
                if self.state == "ABORT_RETRACT":
                    self.state = "STANDBY"
                else:
                    self.state = "STANDBY"
                    self.delivery_completed = True

        return self.cable_len, self.tension_n, self.sway_offset

# ==========================================
# 8. CONTINUOUS BEMT COAXIAL AIRFRAME
# ==========================================
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

    def step(self, motor_cmds, current_wind, rho_air, ice_index=0.0, dt=0.016):
        if self.parachute_deployed:
            self.motor_omegas.fill(0.0)
            self.v[0] *= 0.90
            self.v[1] *= 0.90
            self.v[2] += (-4.1 - self.v[2]) * 3.0 * dt
            self.p += (self.v + current_wind * 0.4) * dt
            if self.p[2] <= 0.0:
                self.p[2] = 0.0
                self.v.fill(0.0)
            return 0.0, np.zeros(3), 0.0, 1.0

        effective_cmds = np.clip(motor_cmds, 0.0, MAX_OMEGA) * self.motor_health
        self.motor_omegas += ((effective_cmds - self.motor_omegas) / MOTOR_TAU_S) * dt
        self.motor_omegas *= self.motor_health

        z_h = max(0.12, float(self.p[2]))
        ige_factor = 1.0 / max(0.35, (1.0 - (ROTOR_RADIUS_M / (4.0 * z_h)) ** 2))
        ige_factor = float(np.clip(ige_factor, 1.0, 1.35))
        d_ceil = max(0.3, CEILING_HEIGHT_M - z_h)
        ice_suction = float(1.0 + (ROTOR_RADIUS_M / (4.0 * d_ceil)) ** 2)

        T = np.zeros(8, dtype=np.float32)
        Q = np.zeros(8, dtype=np.float32)
        r_steps = np.linspace(0.15 * ROTOR_RADIUS_M, ROTOR_RADIUS_M, NUM_BLADE_ELEMENTS)
        dr = r_steps[1] - r_steps[0]

        lift_loss_factor = 1.0 - 0.28 * ice_index
        drag_rise_factor = 1.0 + 1.80 * ice_index

        for i in range(8):
            if self.motor_health[i] == 0.0: continue
            omega_i = self.motor_omegas[i]
            is_bottom = (i % 2 == 1)

            top_downwash = 0.0
            if is_bottom and self.motor_health[i - 1] > 0.0:
                top_downwash = 0.045 * (self.motor_omegas[i - 1] * ROTOR_RADIUS_M) * COAXIAL_DOWNWASH_INGESTION

            t_blade, q_blade = 0.0, 0.0
            for r in r_steps:
                v_axial = -self.v[2] + top_downwash
                v_tan = omega_i * r
                v_eff_sq = v_axial**2 + v_tan**2
                phi = math.atan2(max(0.1, v_axial), max(0.1, v_tan))

                theta_r = math.radians(14.0 - 8.0 * (r / ROTOR_RADIUS_M))
                alpha = theta_r - phi
                c_l = 2.0 * math.pi * alpha * lift_loss_factor
                c_d = (0.012 + 0.04 * (alpha ** 2)) * drag_rise_factor

                exp_arg = -max(0.01, (ROTOR_RADIUS_M - r) / (2.0 * r * math.sin(max(0.01, phi))))
                acos_arg = min(1.0, max(-1.0, math.exp(exp_arg)))
                f_tip = max(0.2, min(1.0, (2.0 / math.pi) * math.acos(acos_arg)))

                dT = 0.5 * rho_air * v_eff_sq * 0.018 * (c_l * math.cos(phi) - c_d * math.sin(phi)) * f_tip * dr
                dQ = 0.5 * rho_air * v_eff_sq * 0.018 * (c_l * math.sin(phi) + c_d * math.cos(phi)) * f_tip * r * dr
                t_blade += dT
                q_blade += dQ

            T[i] = t_blade * 2.0 * ige_factor * ice_suction
            Q[i] = q_blade * 2.0

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
        f_gravity = MASS_KG * GRAVITY_G

        self.accel_linear = (self.R @ f_body + f_gravity + f_drag) / MASS_KG
        gyroscopic = np.cross(self.omega, INERTIA_MATRIX @ self.omega)
        accel_angular = INV_INERTIA @ (torques - gyroscopic)

        self.p += self.v * dt + 0.5 * self.accel_linear * (dt ** 2)
        self.v += self.accel_linear * dt
        if self.p[2] < 0.0:
            self.p[2] = 0.0
            self.v[2] = max(0.0, self.v[2])

        self.omega += accel_angular * dt
        d_theta = np.linalg.norm(self.omega) * dt
        if d_theta > 1e-6:
            axis = self.omega / np.linalg.norm(self.omega)
            K = skew_symmetric(axis)
            R_delta = np.eye(3) + math.sin(d_theta) * K + (1.0 - math.cos(d_theta)) * (K @ K)
            self.R = self.R @ R_delta

        U, _, Vt = np.linalg.svd(self.R)
        self.R = U @ Vt

        p_watts = float(np.sum(Q * self.motor_omegas)) / 0.85 + 24.0
        return f_thrust, torques, p_watts, ige_factor

# ==========================================
# 9. CBF-QP & ACTUATOR MIXER (ZERO-SPIN)
# ==========================================
class ControlBarrierFunctionQPFilter:
    def __init__(self):
        self.alpha_obs = CBF_ALPHA_OBSTACLE
        self.safe_radius = CBF_SAFE_RADIUS_M

    def enforce_safety(self, drone_p, drone_v, nominal_accel, obstacles):
        safe_accel = nominal_accel.copy()
        for obs in obstacles:
            rel_pos = drone_p - obs["pos"]
            dist_xy = math.hypot(rel_pos[0], rel_pos[1])
            h_obs = dist_xy - (obs["radius"] + self.safe_radius)
            if h_obs < 1.6:
                grad_h = np.array([rel_pos[0] / dist_xy, rel_pos[1] / dist_xy, 0.0], dtype=np.float32)
                l_fh = float(np.dot(grad_h, drone_v))
                b_cbf = -l_fh - self.alpha_obs * h_obs
                val = float(np.dot(grad_h, safe_accel))
                if val < b_cbf:
                    lambda_kkt = (b_cbf - val) / float(np.dot(grad_h, grad_h))
                    safe_accel += lambda_kkt * grad_h

        h_gnd = drone_p[2] - 0.20
        if h_gnd < 0.6:
            l_f_gnd = drone_v[2]
            b_gnd = -l_f_gnd - 2.5 * h_gnd
            if safe_accel[2] < b_gnd:
                safe_accel[2] = b_gnd

        return safe_accel

class RateDampedActuatorMixerWithCBF:
    def __init__(self):
        self.k_R = 2.0
        self.k_w = 0.40
        self.cbf_filter = ControlBarrierFunctionQPFilter()

        arm = ARM_LENGTH_M / math.sqrt(2.0)
        cq_ct = C_Q_CLEAN / C_T_CLEAN
        self.B_nominal = np.array([
            [ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
            [-arm, -arm,  arm,  arm,  arm,  arm, -arm, -arm],
            [-arm, -arm, -arm, -arm,  arm,  arm,  arm,  arm],
            [-cq_ct, cq_ct, cq_ct, -cq_ct, -cq_ct, cq_ct, cq_ct, -cq_ct]
        ], dtype=np.float32)

    def mix(self, drone_health, p_est, v_est, R_est, omega_est, p_target, neural_guidance, token_idx,
            estimated_wind, obstacles, cable_len=0.08, sway_2d=None, load_cell_3d=None, ige_fac=1.0,
            v_pack=23.5, bms_temp_c=22.0, dt=0.016):
        delta_p = p_target - p_est
        d_horiz = float(np.linalg.norm(delta_p[:2]))

        a_neural_cmd = neural_guidance[:3] * 9.5
        v_damp = -np.array([2.5, 2.5, 4.0], dtype=np.float32) * v_est

        a_anti_sway = np.zeros(3, dtype=np.float32)
        if cable_len > 0.12 and sway_2d is not None:
            a_anti_sway[:2] -= 2.8 * sway_2d
            if load_cell_3d is not None and np.linalg.norm(load_cell_3d[:2]) > 0.2:
                a_anti_sway[:2] -= 0.15 * (load_cell_3d[:2] / CARGO_MASS_KG)

        a_aero_feedforward = np.zeros(3, dtype=np.float32)
        if ige_fac > 1.05:
            a_aero_feedforward[2] -= 1.8 * (ige_fac - 1.0)
        if estimated_wind[2] < -0.3:
            a_aero_feedforward[2] -= 0.6 * estimated_wind[2]

        a_nominal = a_neural_cmd + v_damp + a_anti_sway + a_aero_feedforward
        a_safe = self.cbf_filter.enforce_safety(p_est, v_est, a_nominal, obstacles)
        F_d = a_safe * MASS_KG + MASS_KG * (-GRAVITY_G)

        max_tilt_rad = math.radians(26.0)
        f_horiz = math.hypot(F_d[0], F_d[1])
        f_vert = max(1.0, F_d[2])
        if f_horiz / f_vert > math.tan(max_tilt_rad):
            scale_h = (f_vert * math.tan(max_tilt_rad)) / f_horiz
            F_d[0] *= scale_h
            F_d[1] *= scale_h

        b3 = R_est[:, 2]
        f_thrust = float(np.dot(F_d, b3))
        f_thrust = max(0.1, min(42.0, f_thrust))

        b3_d = F_d / (np.linalg.norm(F_d) + 1e-6)

        # Yaw alignment: direction of travel during transit, locked to 0° in hover
        if d_horiz > 1.2 and float(np.linalg.norm(v_est[:2])) > 0.5:
            yaw_target = math.atan2(v_est[1], v_est[0])
        else:
            yaw_target = 0.0
        b1_c = np.array([math.cos(yaw_target), math.sin(yaw_target), 0.0], dtype=np.float32)

        b2_d = np.cross(b3_d, b1_c); b2_d /= (np.linalg.norm(b2_d) + 1e-6)
        b1_d = np.cross(b2_d, b3_d)
        R_d = np.column_stack([b1_d, b2_d, b3_d]).astype(np.float32)

        R_err_mat = 0.5 * (R_d.T @ R_est - R_est.T @ R_d)
        e_R = vee_map(R_err_mat)
        tau = -self.k_R * e_R - self.k_w * omega_est + np.cross(omega_est, INERTIA_MATRIX @ omega_est)
        tau[2] = np.clip(tau[2], -0.20, 0.20) 

        desired_wrench = np.array([f_thrust, tau[0], tau[1], tau[2]], dtype=np.float32)

        W_health = np.diag(drone_health).astype(np.float32)
        B_active = self.B_nominal @ W_health
        try:
            B_pinv = np.linalg.pinv(B_active)
            T_cmds = B_pinv @ desired_wrench
        except np.linalg.LinAlgError:
            T_cmds = np.full(8, f_thrust / 8.0, dtype=np.float32)

        T_cmds = np.clip(T_cmds * drone_health, 0.0, 10.5)

        omega_cmds = np.zeros(8, dtype=np.float32)
        for i in range(8):
            if drone_health[i] > 0.0:
                scale_c = C_T_CLEAN if (i % 2 == 0) else (C_T_CLEAN * COAXIAL_EFFICIENCY)
                omega_cmds[i] = math.sqrt(T_cmds[i] / scale_c)

        return omega_cmds, f_thrust, a_safe

# ==========================================
# 10. SURFACE ASSURANCE & PHYSICS SENSORS
# ==========================================
class SurfaceAndWireAssuranceSuite:
    def __init__(self, world: MetropolitanCityGridWorld400m):
        self.world = world

    def audit_surface(self, pos_xy, buffer_m=2.5):
        px, py = pos_xy
        for h in self.world.houses:
            x1, x2, y1, y2, _ = h["aabb"]
            if (x1 - buffer_m) <= px <= (x2 + buffer_m) and (y1 - buffer_m) <= py <= (y2 + buffer_m):
                return "ROOF_HAZARD", False, 0.85
        for line in self.world.powerlines:
            w1 = line["p1"]
            if abs(py - w1[1]) < (buffer_m + 1.2):
                return "WIRE_CORRIDOR_HAZARD", False, 0.50
        return "SOLID_TERRAIN_SAFE", True, 0.92

    def snap_to_nearest_safe_ground(self, target_xy):
        cand = np.array([target_xy[0], target_xy[1], 0.0], dtype=np.float32)
        _, is_safe, _ = self.audit_surface(cand[:2])
        if is_safe:
            return cand

        for r in np.linspace(3.5, 25.0, 16):
            for ang in np.linspace(0, 2*math.pi, 16):
                test_pos = np.array([cand[0] + r*math.cos(ang), cand[1] + r*math.sin(ang), 0.0], dtype=np.float32)
                _, safe_found, _ = self.audit_surface(test_pos[:2])
                if safe_found:
                    return test_pos
        return np.array([-140.0, -140.0, 0.0], dtype=np.float32)

class FullPhysicsSensorSuite:
    def __init__(self):
        self.wire_y = 170.0
        self.wire_z = 12.0

    def poll_all_physics_sensors(self, drone_p, drone_v, current_wind):
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
        min_sonar = max(0.05, min(15.0, z_true))
        return {"z_baro": z_baro, "rho_air": rho_air, "v_tas": v_tas, "q_dyn": q_dyn, "emf_rms": emf_rms, "min_sonar": min_sonar}

# ==========================================
# 11. QUANTUM SENSOR & ES-EKF ON SO(3)
# ==========================================
class MatterWaveColdAtomQuantumSensor:
    def __init__(self):
        self.k_eff = K_EFF_WAVEVECTOR
        self.T = RAMAN_PULSE_SEP_T

    def measure_phase_shift(self, true_accel_inertial):
        phase_shift = self.k_eff * true_accel_inertial * (self.T ** 2)
        phase_shift += np.random.normal(0, 0.002, 3).astype(np.float32)
        a_quantum = phase_shift / (self.k_eff * (self.T ** 2))
        return a_quantum, phase_shift

class ErrorStateEKF15:
    def __init__(self, init_pos):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.b_a = np.zeros(3, dtype=np.float32)
        self.b_g = np.zeros(3, dtype=np.float32)
        self.P = np.diag([0.15]*3 + [0.1]*3 + [0.02]*3 + [0.05]*3 + [0.01]*3).astype(np.float32)
        self.Q_d = np.diag([1e-4]*3 + [1e-3]*3 + [1e-4]*3 + [1e-5]*3 + [1e-6]*3).astype(np.float32)

    def predict(self, accel_raw, gyro_raw, dt=0.016):
        a_unbiased = accel_raw - self.b_a
        w_unbiased = gyro_raw - self.b_g
        R = quat_to_rot(self.q)
        f_inertial = R @ a_unbiased + GRAVITY_G

        self.p += self.v * dt + 0.5 * f_inertial * (dt ** 2)
        self.v += f_inertial * dt

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
        self.P = 0.5 * (self.P + self.P.T)

    def update_measurement(self, H, innovation, R_cov):
        S = H @ self.P @ H.T + R_cov
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        delta_x = (K @ innovation).flatten()
        delta_x = np.clip(delta_x, -3.5, 3.5)
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

# ==========================================
# 12. 100-DIM OBSERVATION SYNTHESIZER
# ==========================================
class DroneObservationSynthesizer100Dim:
    def __init__(self):
        self.last_tension = PAYLOAD_WEIGHT_N

    def synthesize(self, p_est, v_est, R_est, omega_est, target_p, lidar_16, conical_8,
                   v_tas, q_dyn, min_sonar, wind_est, ice_idx, cable_len, cable_vel,
                   tension_n, sway_2d, bms_soc, v_pack, bms_temp_c, r0_dyn,
                   chi2_val, jerk_discrepancy, is_spoofed, emf_rms, wire_dist,
                   agent_dist, ground_spl_dba, has_payload,
                   hook_a, hook_w, load_cell_3d, thermal_dt, dolp, ndvi,
                   esc_i_rms, esc_t_mosfet, alpha_deg, beta_deg, rh_pct, dust_pm10,
                   hdop, cn0_avg, rf_jam_pwr, mag_grad, dt=0.016):
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)

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

        # 12. 8-Channel ESC Telemetry (Currents & Temps) [76:92]
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

        return obs

# ==========================================
# 13. RESNET FLIGHT EXECUTIVE (5 HEADS)
# ==========================================
class ResNetBlock256(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.block(x)

class MultiModalFlightExecutiveNet(nn.Module):
    def __init__(self, in_features=OBSERVATION_DIM):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.SiLU()
        )
        self.res1 = ResNetBlock256(256)
        self.res2 = ResNetBlock256(256)

        self.guidance_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 4),
            nn.Tanh()
        )
        self.winch_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
        self.health_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 8),
            nn.Sigmoid()
        )
        self.trust_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
        self.executive_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, NUM_EXECUTIVE_STATES)
        )

    def forward(self, x):
        feat = self.input_proj(x)
        feat = self.res1(feat)
        feat = self.res2(feat)
        guidance = self.guidance_head(feat)
        winch_cmd = self.winch_head(feat)
        health_pred = self.health_head(feat)
        trust_scores = self.trust_head(feat)
        fsm_logits = self.executive_head(feat)
        return guidance, fsm_logits, winch_cmd, health_pred, trust_scores

# ==========================================
# 14. 2-STAGE MULTI-PROCESSING TRAINER
# ==========================================
def _worker_harvest_metropolitan(num_steps):
    np.random.seed()
    random.seed()
    world = MetropolitanCityGridWorld400m()
    synth = DroneObservationSynthesizer100Dim()
    lidar_engine = MacroPerceptionLiDARSuite(world)
    sensor_suite = AdvancedMultiModalSensorSuite(world)

    observations, teacher_guidance, teacher_fsm = [], [], []

    while len(observations) < num_steps:
        start_p = np.array([random.uniform(-160, 160), random.uniform(-160, 160), random.uniform(2.0, 18.0)], dtype=np.float32)
        # Select target from individual building addresses
        target_entry = random.choice(world.manifest_addresses)
        target_gnd = target_entry["pos"].copy()

        drone_p = start_p.copy()
        drone_v = np.zeros(3, dtype=np.float32)
        hover_dest = np.array([target_gnd[0], target_gnd[1], 2.2], dtype=np.float32)

        step_count = 0
        dt = 0.016

        # Randomize environmental conditions per episode
        weather_mode = random.choice(["CLEAR_NOMINAL", "CANYON_GALE", "FREEZING_RAIN", "MICROBURST_STORM"])
        if weather_mode == "CLEAR_NOMINAL":
            episode_wind = np.array([random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0), 0.0], dtype=np.float32)
            episode_temp_c = random.uniform(18.0, 26.0)
            ice_active = False
        elif weather_mode == "CANYON_GALE":
            episode_wind = np.array([random.uniform(-6.0, 6.0), random.uniform(-6.0, 6.0), 0.0], dtype=np.float32)
            episode_temp_c = random.uniform(10.0, 16.0)
            ice_active = False
        elif weather_mode == "FREEZING_RAIN":
            episode_wind = np.array([random.uniform(-3.0, 3.0), random.uniform(-3.0, 3.0), 0.0], dtype=np.float32)
            episode_temp_c = random.uniform(-6.0, -1.0)
            ice_active = True
        else:
            episode_wind = np.array([random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0), random.uniform(-3.0, 0.0)], dtype=np.float32)
            episode_temp_c = random.uniform(8.0, 14.0)
            ice_active = False

        episode_soc = random.uniform(0.20, 0.98)
        episode_v_pack = 18.5 + episode_soc * (25.2 - 18.5)
        is_spoofed_ep = (random.random() < 0.20)
        chi2_ep = float(random.uniform(20.0, 45.0)) if is_spoofed_ep else float(random.uniform(0.8, 3.0))

        sim_has_payload = True
        sim_cable_len = 0.08
        sim_cable_vel = 0.0
        sim_tension = PAYLOAD_WEIGHT_N

        while step_count < 350 and len(observations) < num_steps:
            d_horiz = float(np.linalg.norm(drone_p[:2] - target_gnd[:2]))

            if d_horiz > 0.85:
                fsm_token = 0  # TRANSIT
                v_des = ((hover_dest - drone_p) / max(1e-4, float(np.linalg.norm(hover_dest - drone_p)))) * 3.6
            elif sim_cable_len <= 0.10 and sim_has_payload:
                fsm_token = 1  # HOVER_LOCK
                v_des = np.zeros(3)
                if random.random() < 0.35: sim_cable_vel = 2.40
            elif sim_cable_len > 0.10 and sim_has_payload:
                cargo_z = drone_p[2] - sim_cable_len
                if cargo_z > 0.06:
                    fsm_token = 2  # WINCH_LOWER
                    sim_cable_len += sim_cable_vel * dt
                    sim_tension = PAYLOAD_WEIGHT_N + float(np.random.normal(0, 0.15))
                else:
                    fsm_token = 3  # TOUCHDOWN_RELEASE
                    sim_tension = 1.8
                    if random.random() < 0.40:
                        sim_has_payload = False
                        sim_cable_vel = -3.20
                v_des = np.zeros(3)
            elif not sim_has_payload and sim_cable_len > 0.08:
                fsm_token = 4  # WINCH_RETRACT
                sim_cable_len = max(0.08, sim_cable_len + sim_cable_vel * dt)
                sim_tension = 1.2
                v_des = np.zeros(3)
            else:
                fsm_token = 5  # AUTO_DISPATCH
                sim_cable_len = 0.08
                sim_tension = 1.2
                v_des = np.zeros(3)

            F_d = -np.array([6.5, 6.5, 11.5]) * (drone_p - hover_dest) - np.array([4.4, 4.4, 7.0]) * (drone_v - v_des) + MASS_KG * (-GRAVITY_G)
            guidance_vec = np.clip(F_d / 22.0, -1.0, 1.0).astype(np.float32)

            lidar_16 = lidar_engine.cast_360_lidar(drone_p, 0.0)
            conical_8, wire_clr = lidar_engine.cast_downward_conical(drone_p)

            hook_a, hook_w = sensor_suite.poll_smart_hook_imu(F_d/MASS_KG, np.zeros(3), np.zeros(2), sim_cable_len, sim_has_payload)
            load_3d = sensor_suite.poll_3axis_load_cell(sim_tension, np.zeros(2), sim_cable_len)
            th_dt, dolp, ndvi = sensor_suite.poll_thermal_and_polarization(drone_p[:2])
            i_rms, t_mosfet = sensor_suite.poll_esc_foc_telemetry(np.full(8, HOVER_OMEGA_NOMINAL), np.ones(8))
            alpha, beta, rh, dust = sensor_suite.poll_5port_air_data(drone_v, episode_wind, np.eye(3))
            hdop, cn0, rf_pwr, mag_gr = sensor_suite.poll_rtk_gnss_metrics(drone_p, is_spoofed_ep)

            obs_t = synth.synthesize(
                p_est=drone_p, v_est=drone_v, R_est=np.eye(3, dtype=np.float32), omega_est=np.zeros(3),
                target_p=hover_dest, lidar_16=lidar_16, conical_8=conical_8,
                v_tas=float(np.linalg.norm(drone_v - episode_wind)), q_dyn=12.0, min_sonar=max(0.1, drone_p[2]),
                wind_est=episode_wind, ice_idx=0.35 if ice_active else 0.0, cable_len=sim_cable_len, cable_vel=sim_cable_vel,
                tension_n=sim_tension, sway_2d=np.zeros(2, dtype=np.float32), bms_soc=episode_soc,
                v_pack=episode_v_pack, bms_temp_c=episode_temp_c, r0_dyn=0.0095, chi2_val=chi2_ep,
                jerk_discrepancy=0.5, is_spoofed=is_spoofed_ep, emf_rms=4.0, wire_dist=wire_clr,
                agent_dist=25.0, ground_spl_dba=55.0, has_payload=sim_has_payload,
                hook_a=hook_a, hook_w=hook_w, load_cell_3d=load_3d, thermal_dt=th_dt, dolp=dolp, ndvi=ndvi,
                esc_i_rms=i_rms, esc_t_mosfet=t_mosfet, alpha_deg=alpha, beta_deg=beta, rh_pct=rh, dust_pm10=dust,
                hdop=hdop, cn0_avg=cn0, rf_jam_pwr=rf_pwr, mag_grad=mag_gr, dt=dt
            )

            observations.append(obs_t)
            teacher_guidance.append(np.array([guidance_vec[0], guidance_vec[1], guidance_vec[2], 0.0], dtype=np.float32))
            teacher_fsm.append(fsm_token)

            drone_p += drone_v * dt
            drone_v += (F_d / MASS_KG) * dt
            step_count += 1

    return np.array(observations, dtype=np.float32), np.array(teacher_guidance, dtype=np.float32), np.array(teacher_fsm, dtype=np.int64)

def harvest_and_train_metropolitan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_SAMPLES = 800000              
    NUM_PRETRAIN_EPOCHS = 100         
    NUM_MAIN_EPOCHS = 250             
    BATCH_SIZE = 8192               
    LEARNING_RATE = 3e-3
    
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per = NUM_SAMPLES // num_workers

    print("\n" + "="*75)
    print(f"HARVESTING {NUM_SAMPLES:,} SAMPLES (METROPOLITAN GRID + WEATHER VARIATION)")
    print(f"Parallel Workers: {num_workers} processes | Compute Device: {device}")
    print("="*75)
    t0 = time.time()

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_harvest_metropolitan, [samples_per] * num_workers)

    X_all = np.concatenate([r[0] for r in results], axis=0)[:NUM_SAMPLES]
    y_g_all = np.concatenate([r[1] for r in results], axis=0)[:NUM_SAMPLES]
    y_f_all = np.concatenate([r[2] for r in results], axis=0)[:NUM_SAMPLES]
    print(f"Dataset compiled in {time.time()-t0:.2f}s! ({len(X_all):,} samples)")

    num_train = int(len(X_all) * 0.90)
    X_train = torch.tensor(X_all[:num_train], device=device)
    yg_train = torch.tensor(y_g_all[:num_train], device=device)
    yf_train = torch.tensor(y_f_all[:num_train], device=device)

    X_val = torch.tensor(X_all[num_train:], device=device)
    yg_val = torch.tensor(y_g_all[num_train:], device=device)
    yf_val = torch.tensor(y_f_all[num_train:], device=device)

    model = MultiModalFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
    criterion_g = nn.MSELoss()
    criterion_f = nn.CrossEntropyLoss()

    num_batches = (num_train + BATCH_SIZE - 1) // BATCH_SIZE

    # Stage 1: Pre-Training Foundation Backbone
    print("\n" + "-"*75)
    print(f"STAGE 1: FOUNDATION BACKBONE PRE-TRAINING ({NUM_PRETRAIN_EPOCHS} EPOCHS)")
    print("-"*75)

    pre_optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    for ep in range(NUM_PRETRAIN_EPOCHS):
        model.train()
        perm = torch.randperm(num_train, device=device)
        ep_loss = 0.0

        for b in range(0, num_train, BATCH_SIZE):
            idx = perm[b : b + BATCH_SIZE]
            bx, byg = X_train[idx], yg_train[idx]

            pre_optimizer.zero_grad(set_to_none=True)
            p_g, _, _, _, _ = model(bx)
            loss_pre = criterion_g(p_g, byg)
            loss_pre.backward()
            pre_optimizer.step()
            ep_loss += loss_pre.item()

        print(f"Pre-train Epoch [{ep+1:02d}/{NUM_PRETRAIN_EPOCHS:02d}] - Kinematic MSE: {ep_loss/num_batches:.5f}")

    # Stage 2: Multi-Hazard Executive Fine-Tuning
    print("\n" + "-"*75)
    print(f"STAGE 2: MULTI-HAZARD EXECUTIVE TRAINING ({NUM_MAIN_EPOCHS} EPOCHS)")
    print("-"*75)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    total_steps = num_batches * NUM_MAIN_EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=total_steps)

    for epoch in range(NUM_MAIN_EPOCHS):
        model.train()
        perm = torch.randperm(num_train, device=device)
        total_loss = 0.0

        for b in range(0, num_train, BATCH_SIZE):
            idx = perm[b : b + BATCH_SIZE]
            bx, byg, byf = X_train[idx], yg_train[idx], yf_train[idx]

            optimizer.zero_grad(set_to_none=True)
            p_g, p_f, _, _, _ = model(bx)
            loss = criterion_g(p_g, byg) + 0.40 * criterion_f(p_f, byf)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            vg, vf, _, _, _ = model(X_val)
            val_mse = criterion_g(vg, yg_val).item()
            val_acc = (vf.argmax(dim=-1) == yf_val).float().mean().item() * 100.0

        print(f"Epoch [{epoch+1:02d}/{NUM_MAIN_EPOCHS:02d}] - Train Loss: {total_loss/num_batches:.4f} | Val MSE: {val_mse:.5f} | Decision Acc: {val_acc:.1f}%")

    torch.save(model.state_dict(), MODEL_WEIGHTS_FILE)
    print("\n" + "="*75)
    print(f"SUCCESS: Model saved to {MODEL_WEIGHTS_FILE} in {time.time()-t0:.2f}s!")
    print("="*75 + "\n")
    return model, device

# ==========================================
# 15. CHASE PROJECTION & EXACT RAYCASTER
# ==========================================
def project_3d_chase(pt, cam_focus, cam_yaw, cam_pitch, center_screen, scale=28.0):
    rel = pt - cam_focus
    x1 = rel[0]*math.cos(cam_yaw) - rel[1]*math.sin(cam_yaw)
    y1 = rel[0]*math.sin(cam_yaw) + rel[1]*math.cos(cam_yaw)
    y2 = y1*math.cos(cam_pitch) - rel[2]*math.sin(cam_pitch)
    z2 = y1*math.sin(cam_pitch) + rel[2]*math.cos(cam_pitch)
    focal = max(2.5, 14.0 - y2)
    sx = int(center_screen[0] + (x1 / focal) * scale * 9.5)
    sy = int(center_screen[1] - (z2 / focal) * scale * 9.5)
    return sx, sy, y2

def screen_to_ground_raycast_exact(mouse_pos, cam_focus, cam_yaw, cam_pitch, center_screen, scale=28.0):
    mx, my = mouse_pos
    dx = (mx - center_screen[0]) / (scale * 9.5)
    dy = -(my - center_screen[1]) / (scale * 9.5)

    cp, sp = math.cos(cam_pitch), math.sin(cam_pitch)
    cy, sy = math.cos(cam_yaw), math.sin(cam_yaw)
    zf = cam_focus[2]

    denom = sp + dy * cp
    if abs(denom) < 1e-4: return cam_focus.copy()

    y1 = (14.0 * dy - zf * (dy * sp - cp)) / denom
    y2 = y1 * cp + zf * sp
    focal = max(2.5, 14.0 - y2)
    x1 = dx * focal

    x = x1 * cy + y1 * sy
    y = -x1 * sy + y1 * cy

    return np.array([
        np.clip(cam_focus[0] + x, -195.0, 195.0),
        np.clip(cam_focus[1] + y, -195.0, 195.0),
        0.0
    ], dtype=np.float32)

def draw_tactical_radar_minimap(screen, map_rect, world, drone_p, drone_yaw, target_p, drone_trail, manifest_idx, mission_phase, vehicles):
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
        pygame.draw.circle(screen, (255, 140, 40), (vx, vy), 2)

    bx, by = to_map(world.base_station[0], world.base_station[1])
    h_col = (0, 255, 140) if "BASE" in mission_phase else (0, 240, 255)
    pygame.draw.circle(screen, h_col, (bx, by), 6, 2)

    active_safe_idx = manifest_idx % len(world.manifest_addresses)
    for idx, order in enumerate(world.manifest_addresses):
        wx, wy = to_map(order["pos"][0], order["pos"][1])
        if idx == active_safe_idx and "DELIVERY" in mission_phase:
            pygame.draw.circle(screen, (0, 255, 140), (wx, wy), 4)
        else:
            pygame.draw.circle(screen, (60, 80, 100), (wx, wy), 2)

    if len(drone_trail) > 1:
        map_trail = [to_map(p[0], p[1]) for p in drone_trail]
        pygame.draw.lines(screen, (0, 180, 220), False, map_trail, 1)

    dx, dy = to_map(drone_p[0], drone_p[1])
    pygame.draw.circle(screen, (255, 255, 255), (dx, dy), 4)
    hx = int(dx + math.cos(drone_yaw) * 8.0)
    hy = int(dy - math.sin(drone_yaw) * 8.0)
    pygame.draw.line(screen, (255, 200, 40), (dx, dy), (hx, hy), 2)

# ==========================================
# 16. MAIN SIMULATION & FLIGHT COCKPIT
# ==========================================
def run_all_physics_simulator():
    if not os.path.exists(MODEL_WEIGHTS_FILE):
        model, device = harvest_and_train_metropolitan()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiModalFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
        try:
            model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE, map_location=device))
            print(f"\nLoaded verified 100-Dim Flight Executive from {MODEL_WEIGHTS_FILE}!")
        except Exception:
            print("Checkpoint mismatch. Re-compiling 100-dim model...")
            model, device = harvest_and_train_metropolitan()

    model.eval()
    pygame.init()
    VIEW_W, VIEW_H = 760, 740
    SIDEBAR_W = 440
    W, H = VIEW_W + SIDEBAR_W, VIEW_H

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Survival Core V17.3 — Metropolitan City Grid & Multi-Task Executive")
    clock = pygame.time.Clock()

    font_large = pygame.font.SysFont("consolas", 17, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 12, bold=True)

    world = MetropolitanCityGridWorld400m()
    weather_engine = DynamicUrbanWeatherEngine()
    traffic_engine = DynamicRoadVehicleTrafficEngine(world.avenue_x, world.street_y)

    init_coords = np.array([-155.0, -155.0, 1.8], dtype=np.float32)
    home_coords = np.array([-155.0, -155.0, 2.2], dtype=np.float32)

    drone = BEMTCoaxialAirframe8DOF(init_coords)
    ekf = ErrorStateEKF15(init_coords)
    quantum_sensor = MatterWaveColdAtomQuantumSensor()
    sensors = FullPhysicsSensorSuite()
    bms = SubZeroArrheniusBatteryECM()
    assurance = SurfaceAndWireAssuranceSuite(world)
    winch = AutonomousWinchDeliverySystem()
    synth = DroneObservationSynthesizer100Dim()
    mixer = RateDampedActuatorMixerWithCBF()

    lidar_suite = MacroPerceptionLiDARSuite(world)
    sensor_suite = AdvancedMultiModalSensorSuite(world)
    ground_agent = DynamicGroundIntruderAgent(init_pos=[-80.0, 80.0, 0.0])
    icing_model = MessingerRotorIcingModel()
    dryden_wind_gen = DrydenAtmosphericTurbulenceModel(w20_m_s=3.4)

    manifest_idx = 0
    num_addresses = len(world.manifest_addresses)
    active_order = world.manifest_addresses[manifest_idx % num_addresses]
    target_delivery_point = active_order["pos"].copy()
    
    mission_phase = "DELIVERY_OUTBOUND"
    depot_turnaround_timer = 0.0

    cam_yaw, cam_pitch = 0.75, 0.45
    center_viewport = (VIEW_W // 2, VIEW_H // 2 + 50)
    cam_focus = drone.p.copy()
    running = True

    gust_timer = 0.0
    active_gust = np.zeros(3, dtype=np.float32)
    inject_spoof = False
    inject_multipath = False
    hover_settle_timer = 0.0
    packages_delivered_count = 0
    delivered_boxes = []
    flight_time = 0.0
    drone_trail = deque(maxlen=180)
    current_token_idx = 0

    current_action_name = "TRANSIT"
    policy_latency_ms = 1.2
    confidence = 0.95

    wind_particles = [
        np.array([random.uniform(-180.0, 180.0), random.uniform(-180.0, 180.0), random.uniform(1.0, 30.0)], dtype=np.float32)
        for _ in range(55)
    ]

    print("\n================ V17.3 METROPOLITAN GRID SIMULATOR RUNNING ================")
    print(" [CITY GRID] : 400m District • 48 Buildings • Dedicated Courtyard Addresses")
    print(" [WEATHER]   : 4 Regimes (Clear, Canyon Gale, Freezing Rain, Storm) + Venturi")
    print(" [TRAFFIC]   : 8 Dynamic Road Patrol Vehicles with CBF Obstacle Buffers")
    print(" [CONTROLS]  : [CLICK] Retarget | [W] Cycle Weather | [1-8] Motor Kill | [H] RTB")
    print("               [A] Leash | [B] Low Batt | [C] Cold | [I] Ice | [J] Spoof | [R] Reset")
    print("===========================================================================\n")

    while running:
        dt = 0.016
        flight_time += dt

        weather_engine.step(dt=dt)
        traffic_engine.step(dt=dt)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if mx < VIEW_W:
                    click_gnd = screen_to_ground_raycast_exact((mx, my), cam_focus, cam_yaw, cam_pitch, center_viewport)
                    safe_gnd = assurance.snap_to_nearest_safe_ground(click_gnd)
                    target_delivery_point = safe_gnd
                    hover_settle_timer = 0.0
                    winch.reload_new_cargo()
                    mission_phase = "DELIVERY_OUTBOUND"
                    print(f">>> [DROPZONE SET] TARGET: [{safe_gnd[0]:.2f}, {safe_gnd[1]:.2f}]m (PARCEL LOADED) <<<")
            elif event.type == pygame.KEYDOWN:
                if pygame.K_1 <= event.key <= pygame.K_8:
                    m_idx = event.key - pygame.K_1
                    drone.motor_health[m_idx] = 0.0 if drone.motor_health[m_idx] == 1.0 else 1.0
                    print(f">>> MOTOR #{m_idx+1}: {'OFFLINE' if drone.motor_health[m_idx]==0.0 else 'ONLINE'}")
                elif event.key == pygame.K_w:
                    weather_engine.cycle_weather()
                elif event.key == pygame.K_a:
                    ground_agent.is_leashed = not ground_agent.is_leashed
                    print(f">>> GROUND AGENT: {'LEASHED' if ground_agent.is_leashed else 'ROAMING'} <<<")
                elif event.key == pygame.K_h:
                    target_delivery_point = home_coords.copy()
                    mission_phase = "RETURNING_TO_BASE"
                    hover_settle_timer = 0.0
                    print(">>> [RTH] RETURNING TO DEPOT HELIPAD [H] <<<")
                elif event.key == pygame.K_b:
                    bms.soc = 0.18
                    print(">>> LOW BATTERY TRIGGERED (18% SOC) -> RTH INITIATED <<<")
                elif event.key == pygame.K_c:
                    bms.temp_c = -5.0
                    print(">>> COLD-SOAK (-5°C) TRIGGERED: ARRHENIUS VOLTAGE SAG <<<")
                elif event.key == pygame.K_i:
                    icing_model.icing_active = not icing_model.icing_active
                    print(f">>> ROTOR ICING: {'ACTIVE' if icing_model.icing_active else 'OFF'}")
                elif event.key == pygame.K_m:
                    inject_multipath = not inject_multipath
                    print(f">>> GNSS MULTIPATH: {'INJECTED' if inject_multipath else 'CLEARED'}")
                elif event.key == pygame.K_j:
                    inject_spoof = not inject_spoof
                    print(f">>> GPS SPOOFING: {'ACTIVE' if inject_spoof else 'CLEARED'}")
                elif event.key == pygame.K_g:
                    gust_timer = 2.5
                    ang = random.uniform(0, 2*math.pi)
                    active_gust = np.array([math.cos(ang)*8.5, math.sin(ang)*8.5, 0.0], dtype=np.float32)
                    print(">>> SEVERE GUST MICROBURST TRIGGERED <<<")
                elif event.key == pygame.K_p:
                    drone.parachute_deployed = True
                    print(">>> EMERGENCY BALLISTIC PARACHUTE DEPLOYED! <<<")
                elif event.key == pygame.K_r:
                    drone.motor_health.fill(1.0)
                    drone.parachute_deployed = False
                    bms.soc = 0.96
                    bms.temp_c = 22.0
                    icing_model.ice_thickness_mm = 0.0
                    inject_spoof = False
                    inject_multipath = False
                    winch.reload_new_cargo()
                    mission_phase = "DELIVERY_OUTBOUND"
                    print(">>> SYSTEM FULL RESET PERFORMED <<<")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: cam_yaw -= 0.03
        if keys[pygame.K_RIGHT]: cam_yaw += 0.03
        if keys[pygame.K_UP]: cam_pitch = min(1.2, cam_pitch + 0.02)
        if keys[pygame.K_DOWN]: cam_pitch = max(0.15, cam_pitch - 0.02)

        cam_focus += (drone.p - cam_focus) * 5.0 * dt

        # Weather, Turbulence & Venturi Winds
        eff_wind, gust_scale, amb_temp, auto_ice = weather_engine.get_weather_parameters(drone.p, world.avenue_x)
        dryden_gust = dryden_wind_gen.step(altitude_m=drone.p[2], airspeed_m_s=np.linalg.norm(drone.v), dt=dt) * (gust_scale / 2.0)
        microburst = active_gust * (gust_timer / 2.5) if gust_timer > 0.0 else np.zeros(3, dtype=np.float32)
        if gust_timer > 0.0: gust_timer = max(0.0, gust_timer - dt)

        total_wind = eff_wind + dryden_gust + microburst

        for wp in wind_particles:
            wp += total_wind * 0.45 * dt
            if wp[0] > 200.0: wp[0] = -200.0
            if wp[0] < -200.0: wp[0] = 200.0
            if wp[1] > 200.0: wp[1] = -200.0
            if wp[1] < -200.0: wp[1] = 200.0

        # Icing & Ground Intruder Step
        icing_model.icing_active = auto_ice or icing_model.icing_active
        ice_mm, ice_idx = icing_model.step(airspeed=float(np.linalg.norm(drone.v - total_wind)), dt=dt)

        payload_tip_world = drone.p + np.array([0.0, 0.0, -0.08]) + np.array([0.0, 0.0, -1.0]) * winch.cable_len
        is_aborting = (winch.state == "ABORT_RETRACT")
        agent_pos, agent_breach, agent_dist = ground_agent.step(
            cargo_world_pos=payload_tip_world,
            is_winch_lowering=(winch.state == "LOWERING"),
            is_aborting=is_aborting,
            dt=dt
        )

        # 16-Beam LiDAR & 8-Beam Conical Rays (Including Dynamic Vehicles)
        drone_yaw = math.atan2(drone.R[1, 0], drone.R[0, 0])
        lidar_16 = lidar_suite.cast_360_lidar(drone.p, drone_yaw, traffic_engine.vehicles)
        conical_8, wire_clearance = lidar_suite.cast_downward_conical(drone.p)
        wire_hit = (wire_clearance < 1.4)

        surf_class, is_surf_safe, _ = assurance.audit_surface(target_delivery_point[:2])
        s_data = sensors.poll_all_physics_sensors(drone.p, drone.v, total_wind)

        # Quantum Sensing & ES-EKF on SO(3)
        accel_inertial_true = (drone.accel_linear - GRAVITY_G)
        a_quantum, phase_shift = quantum_sensor.measure_phase_shift(accel_inertial_true)

        gyro_raw = drone.omega + np.random.normal(0, 0.005, 3).astype(np.float32)
        ekf.predict(drone.R.T @ a_quantum, gyro_raw, dt=dt)

        H_baro = np.zeros((1, 15), dtype=np.float32); H_baro[0, 2] = 1.0
        ekf.update_measurement(H_baro, np.array([drone.p[2] + np.random.normal(0, 0.04) - ekf.p[2]], dtype=np.float32), np.array([[0.05]], dtype=np.float32))

        # GNSS Spoofing Check
        multipath_offset = np.array([4.2, -3.1, 0.8], dtype=np.float32) if inject_multipath else np.zeros(3)
        spoof_offset = np.array([3.5, 2.8, 0.5]) if inject_spoof else np.zeros(3)
        z_gps = drone.p + multipath_offset + spoof_offset + np.random.normal(0, 0.1, 3).astype(np.float32)

        a_inertial_measured = a_quantum + GRAVITY_G
        if not hasattr(ekf, "last_gps_p"):
            ekf.last_gps_p = z_gps.copy()
            ekf.last_gps_v = np.zeros(3, dtype=np.float32)

        v_gps_apparent = (z_gps - ekf.last_gps_p) / dt
        a_gps_apparent = (v_gps_apparent - ekf.last_gps_v) / dt
        ekf.last_gps_p = z_gps.copy()
        ekf.last_gps_v = v_gps_apparent.copy()

        jerk_discrepancy = float(np.linalg.norm(a_gps_apparent - a_inertial_measured))
        kinematic_violation = jerk_discrepancy > 8.5

        H_pos = np.zeros((3, 15), dtype=np.float32); H_pos[0:3, 0:3] = np.eye(3)
        y_innov = z_gps - ekf.p
        S_cov = H_pos @ ekf.P @ H_pos.T + np.eye(3, dtype=np.float32) * 0.08
        chi2_dist = float(y_innov.T @ np.linalg.inv(S_cov) @ y_innov)

        is_attack_detected = (chi2_dist > 16.27) or kinematic_violation or inject_spoof or inject_multipath

        if not is_attack_detected:
            ekf.update_measurement(H_pos, y_innov, np.eye(3, dtype=np.float32) * 0.05)
            gnss_status = "GNSS NOMINAL"
        else:
            cam_optical_drift = np.random.normal(0, 0.015, 3).astype(np.float32)
            p_vio_fused = drone.p + cam_optical_drift
            y_vio = p_vio_fused - ekf.p
            ekf.update_measurement(H_pos, y_vio, np.eye(3, dtype=np.float32) * 0.02)
            gnss_status = "SPOOF REJECTED: QUANTUM VIO LOCK"

        # Battery ECM
        bms.temp_c = amb_temp
        bms_soc, v_pack, i_draw, r0_dyn = bms.step(drone.step.__dict__.get("last_p_watts", 185.0), dt=dt)

        # Autonomous Logistics Settle
        R_est = quat_to_rot(ekf.q)
        omega_est = gyro_raw - ekf.b_g
        hover_target = np.array([target_delivery_point[0], target_delivery_point[1], 2.2], dtype=np.float32)
        active_target = home_coords if (bms_soc <= 0.20 or mission_phase in ["RETURNING_TO_BASE", "BASE_RELOAD"]) else hover_target

        d_horiz_est = float(np.linalg.norm(ekf.p[:2] - active_target[:2]))
        v_horiz_est = float(np.linalg.norm(ekf.v[:2]))

        is_centered = (d_horiz_est <= HOVER_LOCK_RADIUS_M) and (v_horiz_est <= HOVER_LOCK_SPEED_MAX_MS)
        if is_centered and winch.has_payload and mission_phase == "DELIVERY_OUTBOUND":
            hover_settle_timer += dt
        else:
            if winch.cable_len <= 0.10: hover_settle_timer = 0.0

        is_settled_for_winch = (hover_settle_timer >= HOVER_SETTLE_DURATION_S)

        # Multi-Modal Sensors
        hook_a, hook_w = sensor_suite.poll_smart_hook_imu(drone.accel_linear, drone.omega, winch.sway_offset, winch.cable_len, winch.has_payload)
        load_3d = sensor_suite.poll_3axis_load_cell(winch.tension_n, winch.sway_offset, winch.cable_len)
        th_dt, dolp, ndvi = sensor_suite.poll_thermal_and_polarization(target_delivery_point[:2])
        i_rms, t_mosfet = sensor_suite.poll_esc_foc_telemetry(drone.motor_omegas, drone.motor_health)
        alpha, beta, rh, dust = sensor_suite.poll_5port_air_data(drone.v, total_wind, drone.R)
        hdop, cn0, rf_pwr, mag_gr = sensor_suite.poll_rtk_gnss_metrics(drone.p, is_attack_detected)

        # Synthesize 100-Dim State
        obs_vec = synth.synthesize(
            p_est=ekf.p, v_est=ekf.v, R_est=R_est, omega_est=omega_est, target_p=active_target,
            lidar_16=lidar_16, conical_8=conical_8, v_tas=s_data["v_tas"], q_dyn=s_data["q_dyn"],
            min_sonar=s_data["min_sonar"], wind_est=total_wind, ice_idx=ice_idx, cable_len=winch.cable_len,
            cable_vel=winch.cable_vel, tension_n=winch.tension_n, sway_2d=winch.sway_offset,
            bms_soc=bms_soc, v_pack=v_pack, bms_temp_c=bms.temp_c, r0_dyn=r0_dyn, chi2_val=chi2_dist,
            jerk_discrepancy=jerk_discrepancy, is_spoofed=is_attack_detected, emf_rms=s_data["emf_rms"],
            wire_dist=wire_clearance, agent_dist=agent_dist, ground_spl_dba=58.0, has_payload=winch.has_payload,
            hook_a=hook_a, hook_w=hook_w, load_cell_3d=load_3d, thermal_dt=th_dt, dolp=dolp, ndvi=ndvi,
            esc_i_rms=i_rms, esc_t_mosfet=t_mosfet, alpha_deg=alpha, beta_deg=beta, rh_pct=rh, dust_pm10=dust,
            hdop=hdop, cn0_avg=cn0, rf_jam_pwr=rf_pwr, mag_grad=mag_gr, dt=dt
        )

        # Neural Network Forward Pass
        t_inf0 = time.time()
        with torch.no_grad():
            t_in = torch.from_numpy(obs_vec).float().unsqueeze(0).to(device)
            nn_guidance, nn_logits, nn_winch, nn_health, nn_trust = model(t_in)
            neural_guidance = nn_guidance[0].cpu().numpy()
            probs = F.softmax(nn_logits[0], dim=-1).cpu().numpy()
            token_idx = int(np.argmax(probs))

        policy_latency_ms = (time.time() - t_inf0) * 1000.0
        current_token_idx = token_idx
        current_action_name = ACTION_NAMES[token_idx]
        confidence = float(probs[token_idx])

        # Winch Step
        cable_l, tension, sway_2d = winch.step(
            drone_z=drone.p[2],
            is_settled=is_settled_for_winch,
            is_surface_safe=is_surf_safe,
            wire_hit=wire_hit,
            agent_breach=agent_breach,
            agent_dist=agent_dist,
            wind_2d=total_wind[:2],
            neural_token_idx=current_token_idx,
            dt=dt
        )

        active_order = world.manifest_addresses[manifest_idx % num_addresses]

        if winch.delivery_completed and mission_phase == "DELIVERY_OUTBOUND":
            winch.delivery_completed = False
            packages_delivered_count += 1
            delivered_boxes.append(target_delivery_point.copy())
            hover_settle_timer = 0.0

            mission_phase = "RETURNING_TO_BASE"
            target_delivery_point = home_coords.copy()
            print(f"\n>>> [DELIVERY COMPLETE] PARCEL #{packages_delivered_count} AT {active_order['bldg_id']} ({active_order['addr']})! <<<")

        dist_to_depot = float(np.linalg.norm(drone.p[:2] - world.base_station[:2]))
        is_docked_at_depot = (dist_to_depot < 1.35) and (v_horiz_est < 0.55)

        if mission_phase == "RETURNING_TO_BASE":
            if is_docked_at_depot:
                depot_turnaround_timer += dt
                if depot_turnaround_timer >= 0.8:
                    depot_turnaround_timer = 0.0
                    manifest_idx = (manifest_idx + 1) % num_addresses
                    active_order = world.manifest_addresses[manifest_idx]
                    target_delivery_point = active_order["pos"].copy()

                    winch.reload_new_cargo()  
                    bms.soc = 0.96            
                    mission_phase = "DELIVERY_OUTBOUND"
                    hover_settle_timer = 0.0
                    print(f">>> [DEPOT DOCKED] RELOADED! DISPATCHING TO: {active_order['bldg_id']} ({active_order['addr']}) <<<")
            else:
                depot_turnaround_timer = 0.0

        # CBF Obstacles: Buildings + Dynamic Vehicles
        obstacles_cbf = [{"pos": np.array([(h["aabb"][0]+h["aabb"][1])/2, (h["aabb"][2]+h["aabb"][3])/2, 0.0]), "radius": 11.5} for h in world.houses]
        obstacles_cbf.extend(traffic_engine.get_cbf_obstacles())

        # Mixer & Airframe Integration
        omega_cmds, f_thrust, a_safe = mixer.mix(
            drone_health=drone.motor_health, p_est=ekf.p, v_est=ekf.v, R_est=R_est, omega_est=omega_est,
            p_target=active_target, neural_guidance=neural_guidance, token_idx=token_idx,
            estimated_wind=total_wind, obstacles=obstacles_cbf, cable_len=winch.cable_len,
            sway_2d=sway_2d, load_cell_3d=load_3d, ige_fac=drone.step.__dict__.get("last_ige", 1.0),
            v_pack=v_pack, bms_temp_c=bms.temp_c, dt=dt
        )

        f_thrust_act, torques, p_watts, ige_fac = drone.step(
            motor_cmds=omega_cmds, current_wind=total_wind, rho_air=s_data["rho_air"],
            ice_index=ice_idx, dt=dt
        )
        drone.step.__dict__["last_p_watts"] = p_watts
        drone.step.__dict__["last_ige"] = ige_fac

        drone_yaw = math.atan2(drone.R[1, 0], drone.R[0, 0])
        if int(flight_time * 60) % 6 == 0:
            drone_trail.append(drone.p[:2].copy())

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
            pygame.draw.line(screen, (18, 24, 32), (sx1, sy1), (sx2, sy2), 1)

        # Asphalt Avenues & Streets
        for r in world.roads:
            if math.hypot(cam_focus[0] - (r["x_min"]+r["x_max"])/2, cam_focus[1] - (r["y_min"]+r["y_max"])/2) < 85.0:
                rc_pts = [
                    np.array([r["x_min"], r["y_min"], 0.0]), np.array([r["x_max"], r["y_min"], 0.0]),
                    np.array([r["x_max"], r["y_max"], 0.0]), np.array([r["x_min"], r["y_max"], 0.0])
                ]
                r_screens = [project_3d_chase(p, cam_focus, cam_yaw, cam_pitch, center_viewport)[:2] for p in rc_pts]
                pygame.draw.polygon(screen, (24, 30, 38), r_screens)

        # 48 Multi-Story Buildings with Address Signs
        for idx, h in enumerate(world.houses):
            hx1, hx2, hy1, hy2, hh = h["aabb"]
            h_center = np.array([(hx1+hx2)/2, (hy1+hy2)/2, 0.0])
            if np.linalg.norm(cam_focus[:2] - h_center[:2]) < 65.0:
                h_corners = [
                    np.array([hx1, hy1, 0.0]), np.array([hx2, hy1, 0.0]),
                    np.array([hx2, hy2, 0.0]), np.array([hx1, hy1, hh]),
                    np.array([hx2, hy1, hh]), np.array([hx2, hy2, hh]),
                    np.array([hx1, hy2, hh])
                ]
                h_screens = [project_3d_chase(p, cam_focus, cam_yaw, cam_pitch, center_viewport)[:2] for p in h_corners]
                roof_poly = [h_screens[3], h_screens[4], h_screens[5], h_screens[6]]
                pygame.draw.polygon(screen, (42, 50, 62), roof_poly)
                pygame.draw.lines(screen, (100, 120, 145), True, roof_poly, 2)
                for ci in range(3):
                    pygame.draw.line(screen, (65, 78, 95), h_screens[ci], h_screens[ci+3], 2)

                rf_cx = int((h_screens[3][0] + h_screens[5][0]) / 2)
                rf_cy = int((h_screens[3][1] + h_screens[5][1]) / 2)
                screen.blit(font_small.render(f"{world.manifest_addresses[idx]['bldg_id']}", True, (140, 170, 205)), (rf_cx - 16, rf_cy - 6))

        # Dynamic Road Vehicles
        for v in traffic_engine.vehicles:
            if np.linalg.norm(cam_focus[:2] - v["pos"][:2]) < 60.0:
                vx_s, vy_s, _ = project_3d_chase(v["pos"], cam_focus, cam_yaw, cam_pitch, center_viewport)
                pygame.draw.rect(screen, (255, 140, 40), (vx_s - 6, vy_s - 4, 12, 8), border_radius=2)
                pygame.draw.circle(screen, (255, 255, 200), (vx_s + 4, vy_s), 2)

        # Base Logistics Depot Helipad [H]
        if np.linalg.norm(cam_focus[:2] - world.base_station[:2]) < 70.0:
            hc_sx, hc_sy, _ = project_3d_chase(world.base_station, cam_focus, cam_yaw, cam_pitch, center_viewport)
            h_col = (0, 255, 140) if "BASE" in mission_phase else (0, 240, 255)
            pygame.draw.circle(screen, h_col, (hc_sx, hc_sy), 24, 2)
            screen.blit(font_large.render("H", True, h_col), (hc_sx - 6, hc_sy - 10))

        # Ground Intruder Agent
        if np.linalg.norm(cam_focus[:2] - agent_pos[:2]) < 45.0:
            ag_sx, ag_sy, _ = project_3d_chase(agent_pos, cam_focus, cam_yaw, cam_pitch, center_viewport)
            ag_col = (100, 220, 120) if ground_agent.is_leashed else ((255, 60, 80) if agent_breach else (255, 120, 40))
            pygame.draw.circle(screen, ag_col, (ag_sx, ag_sy), 7)
            pygame.draw.circle(screen, (100, 140, 180) if not agent_breach else (255, 60, 80), (ag_sx, ag_sy), int(ground_agent.personal_space_m * 18), 1)

        # Delivered Boxes on Ground
        for idx, pb in enumerate(delivered_boxes):
            if np.linalg.norm(cam_focus[:2] - pb[:2]) < 55.0:
                bx, by, _ = project_3d_chase(pb, cam_focus, cam_yaw, cam_pitch, center_viewport)
                pygame.draw.rect(screen, (60, 160, 90), (bx - 5, by - 5, 10, 10), border_radius=2)

        # Target Dropzone Ring
        tsx, tsy, _ = project_3d_chase(target_delivery_point, cam_focus, cam_yaw, cam_pitch, center_viewport)
        t_ring_col = (0, 220, 255) if "BASE" in mission_phase else ((0, 255, 140) if is_surf_safe else (255, 60, 80))
        pygame.draw.circle(screen, t_ring_col, (tsx, tsy), 16, 2)
        pygame.draw.circle(screen, (255, 255, 255), (tsx, tsy), 4)
        t_label = "BASE DEPOT [H]" if "BASE" in mission_phase else f"{active_order['bldg_id']} ({active_order['addr']})"
        screen.blit(font_small.render(t_label, True, t_ring_col), (tsx + 18, tsy - 6))

        # Wind Streamers
        for wp in wind_particles:
            px, py, _ = project_3d_chase(wp, cam_focus, cam_yaw, cam_pitch, center_viewport)
            if 0 < px < VIEW_W and 70 < py < H:
                pygame.draw.line(screen, (80, 140, 180), (px, py), (px + int(total_wind[0]*2.0), py - int(total_wind[1]*2.0)), 1)

        # Airframe & Rotors
        gsx, gsy, _ = project_3d_chase(np.array([drone.p[0], drone.p[1], 0.0]), cam_focus, cam_yaw, cam_pitch, center_viewport)
        shadow_r = int(max(6, 16 - drone.p[2] * 2.0))
        pygame.draw.ellipse(screen, (22, 32, 44), (gsx - shadow_r, gsy - shadow_r // 2, shadow_r * 2, shadow_r))

        dsx, dsy, _ = project_3d_chase(drone.p, cam_focus, cam_yaw, cam_pitch, center_viewport)
        pygame.draw.circle(screen, (0, 240, 255), (dsx, dsy), 8)

        arm_d = ARM_LENGTH_M * 1.8
        arm_ends_body = [
            np.array([ arm_d/1.414, -arm_d/1.414, 0.0]), np.array([ arm_d/1.414,  arm_d/1.414, 0.0]),
            np.array([-arm_d/1.414,  arm_d/1.414, 0.0]), np.array([-arm_d/1.414, -arm_d/1.414, 0.0]),
        ]
        arm_screens = [project_3d_chase(drone.p + drone.R @ a, cam_focus, cam_yaw, cam_pitch, center_viewport)[:2] for a in arm_ends_body]
        pygame.draw.line(screen, (160, 170, 185), arm_screens[0], arm_screens[2], 3)
        pygame.draw.line(screen, (160, 170, 185), arm_screens[1], arm_screens[3], 3)

        for m_i in range(8):
            a_i = [0, 0, 1, 1, 2, 2, 3, 3][m_i]
            mx, my = arm_screens[a_i]
            off_y = -6 if (m_i % 2 == 0) else 6
            if drone.motor_health[m_i] == 0.0:
                pygame.draw.line(screen, (255, 60, 80), (mx - 5, my + off_y - 5), (mx + 5, my + off_y + 5), 2)
                pygame.draw.line(screen, (255, 60, 80), (mx + 5, my + off_y - 5), (mx - 5, my + off_y + 5), 2)
            else:
                m_col = (0, 240, 255) if (m_i % 2 == 0) else (0, 255, 140)
                pygame.draw.circle(screen, m_col, (mx, my + off_y), 9, 1)

        # Winch & Cargo Box
        payload_3d = np.array([
            drone.p[0] + winch.sway_offset[0],
            drone.p[1] + winch.sway_offset[1],
            drone.p[2] - winch.cable_len
        ], dtype=np.float32)
        psx, psy, _ = project_3d_chase(payload_3d, cam_focus, cam_yaw, cam_pitch, center_viewport)

        if not winch.severed:
            if winch.cable_len > 0.10:
                pygame.draw.line(screen, (230, 240, 255), (dsx, dsy + 4), (psx, psy), 2)

            if winch.has_payload:
                box_x, box_y = (dsx, dsy + 6) if winch.cable_len <= 0.10 else (psx, psy)
                pygame.draw.rect(screen, (255, 195, 35), (box_x - 7, box_y - 7, 14, 14), border_radius=2)
                pygame.draw.rect(screen, (60, 45, 10), (box_x - 7, box_y - 7, 14, 14), 1)
                pygame.draw.line(screen, (50, 40, 10), (box_x - 7, box_y), (box_x + 7, box_y), 1)
                pygame.draw.line(screen, (50, 40, 10), (box_x, box_y - 7), (box_x, box_y + 7), 1)
                screen.blit(font_small.render("0.45kg PARCEL", True, (255, 215, 0)), (box_x + 9, box_y - 6))
            else:
                hook_x, hook_y = (dsx, dsy + 4) if winch.cable_len <= 0.10 else (psx, psy)
                pygame.draw.circle(screen, (200, 220, 240), (hook_x, hook_y), 3)

        # ==========================================
        # COMPLETE SIDEBAR DASHBOARD
        # ==========================================
        sb_x = VIEW_W + 14
        screen.blit(font_large.render("SURVIVAL CORE V17.3", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("City Grid • Dynamic Weather Regimes • Vehicle Traffic", True, (130, 160, 190)), (sb_x, 38))

        # Tactical Radar Minimap
        map_rect = (sb_x, 65, 210, 210)
        draw_tactical_radar_minimap(screen, map_rect, world, drone.p, drone_yaw, target_delivery_point, drone_trail, manifest_idx, mission_phase, traffic_engine.vehicles)

        # 48-Building Manifest Queue
        mf_x = sb_x + 224
        screen.blit(font_mono.render("48-BLDG QUEUE:", True, (240, 240, 240)), (mf_x, 65))
        current_safe_idx = manifest_idx % num_addresses
        for offset in range(5):
            m_idx = (current_safe_idx + offset) % num_addresses
            order = world.manifest_addresses[m_idx]
            if offset == 0:
                m_col = (0, 255, 140) if mission_phase == "DELIVERY_OUTBOUND" else (0, 220, 255)
                tag = "[ACTIVE]" if mission_phase == "DELIVERY_OUTBOUND" else "[RTB]"
            else:
                m_col = (140, 160, 180)
                tag = f"[Q #{offset+1}]"
            screen.blit(font_small.render(f"{tag} {order['bldg_id']} ({order['addr'][:12]})", True, m_col), (mf_x, 85 + offset * 18))

        # 1. District Manifest Status
        p1_y = 285
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p1_y - 8), (W - 14, p1_y - 8), 1)
        screen.blit(font_mono.render("1. 48-BUILDING DISTRICT MANIFEST:", True, (240, 240, 240)), (sb_x, p1_y))
        dist_dest = float(np.linalg.norm(drone.p[:2] - active_target[:2]))
        dest_tag = "Base Depot [H]" if "BASE" in mission_phase else f"{active_order['bldg_id']} ({active_order['addr']})"
        screen.blit(font_small.render(f"Target Waypoint:   {dest_tag[:22]} ({dist_dest:.1f}m)", True, (0, 220, 255)), (sb_x + 10, p1_y + 18))
        cargo_tag = "0.45kg PARCEL CLAMPED" if winch.has_payload else "EMPTY HOOK (RTB RELOAD)"
        screen.blit(font_small.render(f"Fuselage Cargo:    {cargo_tag}", True, (0, 255, 140) if winch.has_payload else (255, 200, 40)), (sb_x + 10, p1_y + 34))
        screen.blit(font_small.render(f"Manifest Status:   {packages_delivered_count} Completed (48 Address Queue)", True, (0, 255, 140)), (sb_x + 10, p1_y + 50))

        # 2. 100-Dim Sensory Telemetry
        p2_y = p1_y + 68
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (W - 14, p2_y), 1)
        screen.blit(font_mono.render("2. 100-DIM SENSORY TELEMETRY:", True, (240, 240, 240)), (sb_x, p2_y + 8))
        screen.blit(font_small.render(f"Smart Hook IMU:    Tumble Spin = [{hook_w[0]:.1f}, {hook_w[1]:.1f}] rad/s", True, (0, 220, 255)), (sb_x + 10, p2_y + 26))
        screen.blit(font_small.render(f"3-Axis Load Cell:  F_xyz = [{load_3d[0]:.1f}, {load_3d[1]:.1f}, {load_3d[2]:.1f}] N", True, (160, 180, 205)), (sb_x + 10, p2_y + 42))
        screen.blit(font_small.render(f"5-Port Air Probe:  Alpha = {alpha:.1f}° | Beta = {beta:.1f}° | PM10 = {dust:.0f}", True, (0, 255, 140)), (sb_x + 10, p2_y + 58))

        # 3. ESC Telemetry & Multi-Spectral
        p3_y = p2_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (W - 14, p3_y), 1)
        screen.blit(font_mono.render("3. ESC TELEMETRY & MULTI-SPECTRAL:", True, (240, 240, 240)), (sb_x, p3_y + 8))
        avg_i = float(np.mean(i_rms))
        max_t = float(np.max(t_mosfet))
        screen.blit(font_small.render(f"8-Ch ESC Currents: Avg I = {avg_i:.1f} A | Max MOSFET = {max_t:.1f}°C", True, (255, 180, 40) if max_t > 65.0 else (0, 220, 255)), (sb_x + 10, p3_y + 26))
        screen.blit(font_small.render(f"Thermal LWIR / DoLP:ΔT = {th_dt:.1f}°C | Pol = {dolp:.2f} | NDVI = {ndvi:.2f}", True, (0, 255, 140) if ndvi > 0.4 else (255, 60, 80)), (sb_x + 10, p3_y + 42))
        screen.blit(font_small.render(f"RTK GNSS Metrics:  HDOP = {hdop:.2f} | Avg C/N0 = {cn0:.1f} dB-Hz", True, (160, 180, 205)), (sb_x + 10, p3_y + 58))

        # 4. Urban Weather & Canyon Aerodynamics
        p4_y = p3_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (W - 14, p4_y), 1)
        screen.blit(font_mono.render("4. URBAN WEATHER & CANYON AERODYNAMICS:", True, (240, 240, 240)), (sb_x, p4_y + 8))
        w_mode = weather_engine.get_mode_name()
        w_col = (0, 255, 140) if "CLEAR" in w_mode else ((255, 180, 40) if "CANYON" in w_mode else (255, 60, 80))
        screen.blit(font_small.render(f"Atmospheric State: {w_mode} ([W] Cycle)", True, w_col), (sb_x + 10, p4_y + 26))
        screen.blit(font_small.render(f"Venturi Airspeed:  |V_tas| = {s_data['v_tas']:.2f} m/s | Wind: {np.linalg.norm(total_wind):.1f} m/s", True, (0, 220, 255)), (sb_x + 10, p4_y + 42))
        screen.blit(font_small.render(f"Rotor Ice Layer:   {ice_mm:.2f} mm | Temp = {bms.temp_c:.1f}°C", True, (160, 180, 205)), (sb_x + 10, p4_y + 58))

        # 5. GNSS Anti-Spoof & Quantum Sensing
        p5_y = p4_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p5_y), (W - 14, p5_y), 1)
        screen.blit(font_mono.render("5. GNSS ANTI-SPOOF & QUANTUM VIO:", True, (240, 240, 240)), (sb_x, p5_y + 8))
        screen.blit(font_small.render(f"Navigation State:  {gnss_status}", True, (255, 60, 80) if "REJECTED" in gnss_status else (0, 255, 140)), (sb_x + 10, p5_y + 26))
        screen.blit(font_small.render(f"Kinematic Discrep: Δa = {jerk_discrepancy:.2f} m/s² (Thresh = 8.5)", True, (160, 180, 205)), (sb_x + 10, p5_y + 42))
        screen.blit(font_small.render(f"Quantum Phase:     ΔΦ = {float(np.linalg.norm(phase_shift)):.3f} rad (Rb-87 Raman)", True, (0, 220, 255)), (sb_x + 10, p5_y + 58))

        # 6. Deep Neural Executive & Winch
        p6_y = p5_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p6_y), (W - 14, p6_y), 1)
        screen.blit(font_mono.render("6. NEURAL EXECUTIVE & WINCH DYNAMICS:", True, (240, 240, 240)), (sb_x, p6_y + 8))
        screen.blit(font_small.render(f"Primary Action:    {current_action_name} ({confidence*100:.1f}%) | Latency: {policy_latency_ms:.2f}ms", True, (0, 255, 140)), (sb_x + 10, p6_y + 26))
        screen.blit(font_small.render(f"Winch State:       {winch.state} (Cable: {winch.cable_len:.2f} m | Tension: {winch.tension_n:.1f} N)", True, (0, 220, 255)), (sb_x + 10, p6_y + 42))
        screen.blit(font_small.render(f"Traffic / Agent:   8 Vehicles Active | Ground Agent: {agent_dist:.1f}m", True, (160, 180, 205)), (sb_x + 10, p6_y + 58))

        # Bottom Command Strip
        bot_y = H - 38
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y - 8), (W - 14, bot_y - 8), 1)
        screen.blit(font_mono.render("[CLICK] Target | [W] Weather | [H] RTB | [J] Spoof | [I] Ice | [R] Reset", True, (255, 200, 40)), (sb_x + 8, bot_y))

        # Viewport Header
        status_banner = f"PHASE: {mission_phase} | Weather: {w_mode} | Delivered: {packages_delivered_count} | Target: {dest_tag[:20]}"
        screen.blit(font_large.render(status_banner, True, (240, 240, 240)), (15, 14))
        screen.blit(font_small.render("Metropolitan City Grid • 48 Dedicated Building Addresses • Dynamic Urban Weather", True, (0, 210, 255)), (15, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    mp.freeze_support()
    run_all_physics_simulator()