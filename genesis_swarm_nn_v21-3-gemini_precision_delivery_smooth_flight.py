"""
=============================================================================
SURVIVAL CORE V39.0 — POST-HUMAN SENSORIMOTOR DRONE SWARM SYSTEM
=============================================================================
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import sys
import time
import math
import random
from collections import deque
from pathlib import Path
import multiprocessing as mp
import tempfile
import shutil
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pygame

# =============================================================================
# 0. PHYSICAL AIRFRAME, SENSORY & SWARM CONSTANTS
# =============================================================================
DRY_MASS_KG = 1.35                   
CARGO_MASS_KG = 0.45                 
MASS_KG = DRY_MASS_KG + CARGO_MASS_KG 

GRAVITY_G = np.array([0.0, 0.0, -9.80665], dtype=np.float32)
G_ACCEL = 9.80665
ARM_LENGTH_M = 0.26
ROTOR_RADIUS_M = 0.127
ROTOR_DISK_AREA = math.pi * (ROTOR_RADIUS_M ** 2)

INERTIA_MATRIX = np.diag([0.022, 0.022, 0.038]).astype(np.float32)
INV_INERTIA = np.linalg.inv(INERTIA_MATRIX).astype(np.float32)

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
CBF_ALPHA_OBSTACLE = 2.4
CBF_BASE_SAFE_RADIUS_M = 0.85
SWARM_INTER_DRONE_SAFE_RADIUS_M = 2.6
DOWNWASH_CONE_EXPANSION = 0.38
DOWNWASH_HAZARD_DEPTH_M = 7.5

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

# Continuous Tether Constants (Zero FSM)
PAYLOAD_WEIGHT_N = CARGO_MASS_KG * G_ACCEL
GUILLOTINE_THRESHOLD_N = 48.0
MAX_WINCH_SPEED_MS = 3.5

# Feature Dimensions: 139 features x 3 history frames = 417 dims
BASE_OBS_DIM = 139
HISTORY_LEN = 3
OBSERVATION_DIM = BASE_OBS_DIM * HISTORY_LEN
NUM_EXECUTIVE_STATES = 7

NUM_SWARM_DRONES = 3
MODEL_WEIGHTS_FILE = "drone_master_brain_v39_posthuman.pth"

# =============================================================================
# 1. MATHEMATICAL UTILITIES & LIE ALGEBRA ON SO(3)
# =============================================================================
def skew_symmetric(v):
    return np.array([
        [0.0, -v[2], v[1]], 
        [v[2], 0.0, -v[0]], 
        [-v[1], v[0], 0.0]
    ], dtype=np.float32)

hat_map = skew_symmetric

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

def rot_to_euler(R):
    pitch = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return roll, pitch, yaw

# =============================================================================
# 2. ADAPTIVE ESTIMATORS & SPATIAL HASH ENGINE
# =============================================================================
class VectorizedSpatialHashGrid2D:
    def __init__(self, cell_size=30.0):
        self.cell_size = cell_size
        self.buckets = {}
        self.cached_aabbs = None
        self.cached_ids = None

    def insert(self, aabb, item):
        min_x, max_x, min_y, max_y = aabb[0], aabb[1], aabb[2], aabb[3]
        gx0, gx1 = int(min_x // self.cell_size), int(max_x // self.cell_size)
        gy0, gy1 = int(min_y // self.cell_size), int(max_y // self.cell_size)
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                key = (gx, gy)
                if key not in self.buckets:
                    self.buckets[key] = []
                self.buckets[key].append(item)

    def finalize_cache(self, all_houses):
        aabbs, ids = [], []
        for h in all_houses:
            aabbs.append(h["aabb"])
            ids.append(h["id"])
        self.cached_aabbs = np.array(aabbs, dtype=np.float32)
        self.cached_ids = ids

    def query_segment(self, p1, p2):
        found = []
        seen = set()
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        steps = max(1, int(dist / (self.cell_size * 0.5)))
        for s in range(steps + 1):
            tau = s / float(steps)
            x = p1[0] + tau * dx
            y = p1[1] + tau * dy
            key = (int(x // self.cell_size), int(y // self.cell_size))
            if key in self.buckets:
                for item in self.buckets[key]:
                    item_id = item["id"] if isinstance(item, dict) and "id" in item else id(item)
                    if item_id not in seen:
                        seen.add(item_id)
                        found.append(item)
        return found

class SwarmMicroclimateSharedMap:
    def __init__(self):
        self.grid = {}
        self.voxel_size_xy = 20.0
        self.voxel_size_z = 8.0

    def broadcast_wind(self, pos, wind_vec, timestamp):
        vx = int(pos[0] // self.voxel_size_xy)
        vy = int(pos[1] // self.voxel_size_xy)
        vz = int(max(0.0, pos[2]) // self.voxel_size_z)
        key = (vx, vy, vz)
        if key not in self.grid:
            self.grid[key] = [wind_vec.copy(), timestamp]
        else:
            old_w, _ = self.grid[key]
            self.grid[key] = [0.65 * old_w + 0.35 * wind_vec, timestamp]

    def query_wind_prior(self, pos, current_time, max_age_s=45.0):
        vx = int(pos[0] // self.voxel_size_xy)
        vy = int(pos[1] // self.voxel_size_xy)
        vz = int(max(0.0, pos[2]) // self.voxel_size_z)
        key = (vx, vy, vz)
        if key in self.grid:
            wind_data, t_stamp = self.grid[key]
            if (current_time - t_stamp) <= max_age_s:
                return wind_data.copy(), True
        return np.zeros(3, dtype=np.float32), False

    def query_vertical_wind_shear_profile(self, pos_xy, current_time, candidate_alts):
        vx = int(pos_xy[0] // self.voxel_size_xy)
        vy = int(pos_xy[1] // self.voxel_size_xy)
        profile = []
        for alt in candidate_alts:
            vz = int(max(0.0, alt) // self.voxel_size_z)
            key = (vx, vy, vz)
            if key in self.grid and (current_time - self.grid[key][1]) <= 60.0:
                profile.append(self.grid[key][0].copy())
            else:
                profile.append(np.zeros(3, dtype=np.float32))
        return profile

# =============================================================================
# 3. METROPOLITAN CITY GRID WORLD (400M SCALED ENVIRONMENT)
# =============================================================================
class MetropolitanCityGridWorld400m:
    def __init__(self):
        self.avenue_x = [-120.0, -60.0, 0.0, 60.0, 120.0]
        self.street_y = [-120.0, -60.0, 0.0, 60.0, 120.0]
        self.street_width = 10.0
        self.shared_microclimate_map = SwarmMicroclimateSharedMap()

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
        self.spatial_hash = VectorizedSpatialHashGrid2D(cell_size=30.0)
        self._build_48_buildings_with_rooftop_clutter()
        self.spatial_hash.finalize_cache(self.houses)

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

                clutter_elements = [
                    {"aabb": [bx1 + 1.5, bx1 + 4.5, by2 - 4.5, by2 - 1.5, height + 1.2], "type": "HVAC"},
                    {"aabb": [bx2 - 5.0, bx2 - 2.0, by1 + 2.0, by1 + 5.0, height + 2.2], "type": "BULKHEAD"}
                ]

                bldg_dict = {
                    "id": bldg_id,
                    "aabb": [bx1, bx2, by1, by2, height],
                    "clutter": clutter_elements
                }
                self.houses.append(bldg_dict)
                self.spatial_hash.insert([bx1 - 2.0, bx2 + 2.0, by1 - 2.0, by2 + 2.0], bldg_dict)

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
# 4. DYNAMIC WEATHER, TRAFFIC & INTRUDER ENGINES
# =============================================================================
class DynamicUrbanWeatherEngine:
    def __init__(self):
        self.weather_modes = ["CLEAR_NOMINAL", "CANYON_GALE", "FREEZING_RAIN", "MICROBURST_STORM"]
        self.current_mode_idx = 0
        self.target_mode_idx = 0
        self.regime_timer = 0.0
        self.regime_duration = 45.0
        self.blend_factor = 1.0

    def cycle_weather(self):
        self.target_mode_idx = (self.target_mode_idx + 1) % len(self.weather_modes)
        self.regime_timer = 0.0
        self.blend_factor = 0.0
        print(f"\n>>> [ATMOSPHERIC TRANSITION]: {self.weather_modes[self.target_mode_idx]} <<<")

    def get_mode_name(self):
        return self.weather_modes[self.target_mode_idx]

    def step(self, dt=0.016):
        self.regime_timer += dt
        if self.blend_factor < 1.0:
            self.blend_factor = min(1.0, self.blend_factor + dt / 2.5)
            if self.blend_factor >= 1.0:
                self.current_mode_idx = self.target_mode_idx

        if self.regime_timer >= self.regime_duration:
            self.cycle_weather()

    def _get_preset(self, mode_idx):
        mode = self.weather_modes[mode_idx]
        if mode == "CLEAR_NOMINAL":
            return np.array([1.8, -0.6, 0.0], dtype=np.float32), 0.6, 22.0, False, 0.0
        elif mode == "CANYON_GALE":
            return np.array([5.8, 2.4, 0.0], dtype=np.float32), 3.8, 14.0, False, 0.0
        elif mode == "FREEZING_RAIN":
            return np.array([2.4, -1.8, 0.0], dtype=np.float32), 1.5, -4.5, True, -0.4
        else:  # MICROBURST_STORM
            return np.array([-4.5, 4.2, 0.0], dtype=np.float32), 5.5, 11.0, False, -2.0

    def get_weather_parameters(self, drone_pos, avenues_x):
        px, py, pz = drone_pos
        w1, g1, t1, i1, m1 = self._get_preset(self.current_mode_idx)
        w2, g2, t2, i2, m2 = self._get_preset(self.target_mode_idx)

        b = self.blend_factor
        base_wind = (1.0 - b) * w1 + b * w2
        gust_scale = (1.0 - b) * g1 + b * g2
        temp_c = (1.0 - b) * t1 + b * t2
        auto_ice = i2 if b > 0.5 else i1
        microburst_w = (1.0 - b) * m1 + b * m2

        if microburst_w < -0.2:
            stagnation_factor = float(math.tanh(max(0.1, pz) / 4.5))
            microburst_w *= stagnation_factor

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
            ax = avenues_x[i % len(venues_x)] if (venues_x := avenues_x) else 0.0
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

    def step(self, cargo_world_pos, is_dropping, dt=0.016):
        if self.is_leashed:
            self.p = np.array([-145.0, -145.0, 0.0], dtype=np.float32)
            self.v.fill(0.0)
            return self.p.copy(), False, 99.0

        d_to_cargo = cargo_world_pos[:2] - self.p[:2]
        dist = float(np.linalg.norm(d_to_cargo))

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

    # =============================================================================
# 5. ROTOR ICING, DRYDEN TURBULENCE & SUB-ZERO BATTERY ECM
# =============================================================================
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
# 6. INFLOW HEALTH & 6-DOF MEMS IMU SENSORS
# =============================================================================
class OnlineRotorInflowAndHealthEstimator:
    def __init__(self, nominal_ct=C_T_CLEAN):
        self.nominal_ct = nominal_ct
        self.ct_estimated = nominal_ct
        self.aerodynamic_health = 1.0

    def update(self, current_mass, accel_z, motor_omegas, dt=0.016):
        sum_omega_sq = float(np.sum(motor_omegas ** 2))
        if sum_omega_sq > 1e4:
            t_req = current_mass * float(max(1.0, accel_z + G_ACCEL))
            instant_ct = t_req / sum_omega_sq
            alpha_ct = 0.01
            self.ct_estimated += alpha_ct * (instant_ct - self.ct_estimated)
            self.ct_estimated = float(np.clip(self.ct_estimated, self.nominal_ct * 0.55, self.nominal_ct * 1.35))
            self.aerodynamic_health = float(np.clip(self.ct_estimated / self.nominal_ct, 0.55, 1.0))
        return self.ct_estimated, self.aerodynamic_health

class Airframe6DOFMEMS_IMU:
    def __init__(self):
        self.b_a = np.array([0.02, -0.015, 0.03], dtype=np.float32)
        self.b_g = np.array([0.005, -0.004, 0.002], dtype=np.float32)
        self.accel_raw = np.zeros(3, dtype=np.float32)
        self.gyro_raw = np.zeros(3, dtype=np.float32)

    def poll(self, drone_R, accel_linear, omega_true, motor_omegas):
        f_inertial = accel_linear - GRAVITY_G
        f_body = drone_R.T @ f_inertial

        rpm_ratio = float(np.mean(motor_omegas)) / MAX_OMEGA
        vib_amp = 0.16 * rpm_ratio
        vib_accel = np.random.normal(0.0, vib_amp, 3).astype(np.float32)
        
        self.accel_raw = f_body + self.b_a + vib_accel + np.random.normal(0.0, 0.035, 3).astype(np.float32)

        vib_gyro = np.random.normal(0.0, 0.015 * rpm_ratio, 3).astype(np.float32)
        self.gyro_raw = omega_true + self.b_g + vib_gyro + np.random.normal(0.0, 0.004, 3).astype(np.float32)

        return self.accel_raw.copy(), self.gyro_raw.copy()

# =============================================================================
# 7. UNCONSTRAINED CONTINUOUS PBD TETHER DYNAMICS (ZERO FSM LOCKS)
# =============================================================================
class ContinuousWinchTetherSystem:
    """
    Position-Based Dynamics (PBD) Continuous Tether System.
    Operates without rigid FSM states. The neural policy directly controls continuous
    cable velocity, allowing dynamic touch-and-go drops and momentum exploitation.
    """
    def __init__(self, num_segments=4):
        self.num_segments = num_segments
        self.num_nodes = num_segments + 1
        
        self.cable_len = 0.08
        self.cable_vel = 0.0
        self.theta = np.zeros(2, dtype=np.float32)
        self.theta_dot = np.zeros(2, dtype=np.float32)
        self.sway_offset = np.zeros(2, dtype=np.float32)
        self.catenary_midpoint = np.zeros(2, dtype=np.float32)
        self.tension_n = PAYLOAD_WEIGHT_N
        self.prev_tension_n = PAYLOAD_WEIGHT_N
        self.tension_derivative = 0.0
        self.has_payload = True
        self.severed = False
        self.touchdown_timer = 0.0
        self.delivery_completed = False
        
        self.linear_density = 0.008
        self.cable_diameter_m = 0.0025
        
        self.nodes_pos = np.zeros((self.num_nodes, 3), dtype=np.float32)
        self.nodes_vel = np.zeros((self.num_nodes, 3), dtype=np.float32)
        
        self.cargo_yaw = 0.0
        self.cargo_yaw_rate = 0.0
        self.contact_normal_force = 0.0
        
        self.f_skid_left = 0.0
        self.f_skid_right = 0.0
        self.is_skid_grounded = False

    def reload_new_cargo(self):
        self.cable_len = 0.08
        self.cable_vel = 0.0
        self.theta.fill(0.0)
        self.theta_dot.fill(0.0)
        self.nodes_pos.fill(0.0)
        self.nodes_vel.fill(0.0)
        self.sway_offset.fill(0.0)
        self.catenary_midpoint.fill(0.0)
        self.cargo_yaw = 0.0
        self.cargo_yaw_rate = 0.0
        self.contact_normal_force = 0.0
        self.has_payload = True
        self.severed = False
        self.tension_n = PAYLOAD_WEIGHT_N
        self.prev_tension_n = PAYLOAD_WEIGHT_N
        self.tension_derivative = 0.0
        self.touchdown_timer = 0.0
        self.delivery_completed = False
        self.is_skid_grounded = False

    def step(self, continuous_winch_cmd, drone_p, drone_v, wind_3d, 
             surface_z=0.0, target_surface_z=0.0, dist_to_target_xy=999.0, dt=0.016):
        if self.severed:
            return self.cable_len, 0.0, np.zeros(2, dtype=np.float32), False

        # Continuous Winch Velocity: Command directly controls reel motor
        target_vel = float(np.clip(continuous_winch_cmd, -1.0, 1.0)) * MAX_WINCH_SPEED_MS
        
        # Automatic payload retraction once released
        if not self.has_payload:
            target_vel = -2.80

        # Physical safety overrides (Guillotine cut on extreme snag load)
        if self.tension_n > GUILLOTINE_THRESHOLD_N:
            self.severed = True
            self.has_payload = False
            return self.cable_len, 0.0, np.zeros(2, dtype=np.float32), False

        self.cable_vel += (target_vel - self.cable_vel) * min(1.0, dt / 0.05)
        self.cable_len = float(np.clip(self.cable_len + self.cable_vel * dt, 0.08, 14.0))

        if self.cable_len <= 0.081 and not self.has_payload:
            self.delivery_completed = True

        L = max(0.08, float(self.cable_len))
        seg_len = L / float(self.num_segments)

        self.nodes_pos[0] = drone_p + np.array([0.0, 0.0, -0.05], dtype=np.float32)
        self.nodes_vel[0] = drone_v.copy()

        if self.cable_len <= 0.10:
            for k in range(1, self.num_nodes):
                self.nodes_pos[k] = self.nodes_pos[0] + np.array([0.0, 0.0, -k * seg_len], dtype=np.float32)
                self.nodes_vel[k] = drone_v.copy()
            self.theta.fill(0.0)
            self.theta_dot.fill(0.0)
            self.tension_n = PAYLOAD_WEIGHT_N if self.has_payload else 1.2
            self.prev_tension_n = self.tension_n
            self.tension_derivative = 0.0
            self.sway_offset.fill(0.0)
            self.catenary_midpoint.fill(0.0)
            self.contact_normal_force = 0.0
            return self.cable_len, self.tension_n, self.sway_offset, False

        predicted_pos = np.zeros_like(self.nodes_pos)
        predicted_pos[0] = self.nodes_pos[0]

        # Cargo Fin Aerodynamics (Aligns payload with relative wind)
        v_rel_tip = self.nodes_vel[-1] - wind_3d
        v_rel_xy_speed = math.hypot(v_rel_tip[0], v_rel_tip[1])
        if v_rel_xy_speed > 0.2:
            wind_track_yaw = math.atan2(v_rel_tip[1], v_rel_tip[0])
            yaw_err = (self.cargo_yaw - wind_track_yaw + math.pi) % (2.0 * math.pi) - math.pi
            fin_align_torque = -1.8 * yaw_err * v_rel_xy_speed - 0.45 * self.cargo_yaw_rate
            self.cargo_yaw_rate += fin_align_torque * dt
            self.cargo_yaw += self.cargo_yaw_rate * dt

        # Discretized Node Dynamics (Aerodynamic Drag & Gravity)
        for k in range(1, self.num_nodes):
            m_k = CARGO_MASS_KG if (k == self.num_nodes - 1 and self.has_payload) else 0.04
            v_rel = self.nodes_vel[k] - wind_3d
            speed_rel = float(np.linalg.norm(v_rel))
            cd = 0.85 if (k == self.num_nodes - 1) else 1.15
            area = (CARGO_MASS_KG * 0.045) if (k == self.num_nodes - 1) else (self.cable_diameter_m * seg_len)
            f_drag = -0.5 * 1.225 * cd * area * speed_rel * v_rel

            a_k = GRAVITY_G + (f_drag / m_k)
            self.nodes_vel[k] += a_k * dt
            self.nodes_vel[k] *= 0.965
            predicted_pos[k] = self.nodes_pos[k] + self.nodes_vel[k] * dt

        # Position-Based Dynamics (PBD) Distance Constraints
        for _ in range(4):
            predicted_pos[0] = self.nodes_pos[0]
            for i in range(self.num_segments):
                delta = predicted_pos[i + 1] - predicted_pos[i]
                dist = float(np.linalg.norm(delta))
                if dist > seg_len and dist > 1e-5:
                    correction = (dist - seg_len) * (delta / dist)
                    if i == 0:
                        predicted_pos[i + 1] -= correction
                    else:
                        predicted_pos[i] += 0.35 * correction
                        predicted_pos[i + 1] -= 0.65 * correction

        # REAL SURFACE CONTACT: Only collide with the actual surface under the cargo
        contact_depth = surface_z + 0.06 - predicted_pos[-1][2]
        if contact_depth > 0.0:
            k_spring = 340.0
            d_damper = 14.0
            f_spring = k_spring * contact_depth - d_damper * min(0.0, float(self.nodes_vel[-1][2]))
            self.contact_normal_force = float(np.clip(f_spring, 0.0, 1.5 * PAYLOAD_WEIGHT_N))
            predicted_pos[-1][2] = surface_z + 0.06
            self.nodes_vel[-1][2] = 0.0
            self.nodes_vel[-1][:2] *= 0.85
        else:
            self.contact_normal_force = 0.0

        for k in range(1, self.num_nodes):
            self.nodes_vel[k] = (predicted_pos[k] - self.nodes_pos[k]) / dt
            self.nodes_pos[k] = predicted_pos[k].copy()

        self.sway_offset = self.nodes_pos[-1][:2] - self.nodes_pos[0][:2]
        
        theta_x = math.atan2(float(self.sway_offset[0]), L)
        theta_y = math.atan2(float(self.sway_offset[1]), L)
        self.theta_dot = (np.array([theta_x, theta_y], dtype=np.float32) - self.theta) / dt
        self.theta = np.array([theta_x, theta_y], dtype=np.float32)

        mid_node = self.num_nodes // 2
        straight_mid = 0.5 * (self.nodes_pos[0][:2] + self.nodes_pos[-1][:2])
        self.catenary_midpoint = self.nodes_pos[mid_node][:2] - straight_mid

        # DELIVERY CRITERIA:
        # 1. Must be horizontally over the target dropzone (<= 2.2m)
        # 2. Surface underneath must be the rooftop (abs diff <= 0.6m)
        # 3. Cargo must be touching down on that surface
        cargo_z = self.nodes_pos[-1][2]
        at_dropzone = (dist_to_target_xy <= 2.2) and (abs(surface_z - target_surface_z) <= 0.6)
        touchdown = at_dropzone and (abs(cargo_z - target_surface_z) <= 0.20 or self.contact_normal_force >= 0.40 * PAYLOAD_WEIGHT_N)

        released_this_step = False
        if self.has_payload and touchdown:
            self.touchdown_timer += dt
            if self.touchdown_timer >= 0.20:
                self.has_payload = False
                released_this_step = True
                self.touchdown_timer = 0.0
        else:
            self.touchdown_timer = max(0.0, self.touchdown_timer - dt)

        # Dynamic Cable Tension
        if self.has_payload:
            theta_mag = float(np.linalg.norm(self.theta))
            t_grav = max(0.5, CARGO_MASS_KG * (G_ACCEL * math.cos(theta_mag)) - self.contact_normal_force)
            t_centrif = CARGO_MASS_KG * L * float(np.dot(self.theta_dot, self.theta_dot))
            new_tension = float(np.clip(t_grav + t_centrif, 1.2, 45.0))
        else:
            new_tension = 1.2

        self.tension_derivative = (new_tension - self.prev_tension_n) / dt
        self.prev_tension_n = self.tension_n
        self.tension_n = new_tension

        return self.cable_len, self.tension_n, self.sway_offset, released_this_step

        predicted_pos = np.zeros_like(self.nodes_pos)
        predicted_pos[0] = self.nodes_pos[0]

        # Cargo Fin Aerodynamics (Aligns payload with relative wind)
        v_rel_tip = self.nodes_vel[-1] - wind_3d
        v_rel_xy_speed = math.hypot(v_rel_tip[0], v_rel_tip[1])
        if v_rel_xy_speed > 0.2:
            wind_track_yaw = math.atan2(v_rel_tip[1], v_rel_tip[0])
            yaw_err = (self.cargo_yaw - wind_track_yaw + math.pi) % (2.0 * math.pi) - math.pi
            fin_align_torque = -1.8 * yaw_err * v_rel_xy_speed - 0.45 * self.cargo_yaw_rate
            self.cargo_yaw_rate += fin_align_torque * dt
            self.cargo_yaw += self.cargo_yaw_rate * dt

        # Discretized Node Dynamics (Aerodynamic Drag & Gravity)
        for k in range(1, self.num_nodes):
            m_k = CARGO_MASS_KG if (k == self.num_nodes - 1 and self.has_payload) else 0.04
            v_rel = self.nodes_vel[k] - wind_3d
            speed_rel = float(np.linalg.norm(v_rel))
            cd = 0.85 if (k == self.num_nodes - 1) else 1.15
            area = (CARGO_MASS_KG * 0.045) if (k == self.num_nodes - 1) else (self.cable_diameter_m * seg_len)
            f_drag = -0.5 * 1.225 * cd * area * speed_rel * v_rel

            a_k = GRAVITY_G + (f_drag / m_k)
            self.nodes_vel[k] += a_k * dt
            self.nodes_vel[k] *= 0.965
            predicted_pos[k] = self.nodes_pos[k] + self.nodes_vel[k] * dt

        # Position-Based Dynamics (PBD) Distance Constraints
        for _ in range(4):
            predicted_pos[0] = self.nodes_pos[0]
            for i in range(self.num_segments):
                delta = predicted_pos[i + 1] - predicted_pos[i]
                dist = float(np.linalg.norm(delta))
                if dist > seg_len and dist > 1e-5:
                    correction = (dist - seg_len) * (delta / dist)
                    if i == 0:
                        predicted_pos[i + 1] -= correction
                    else:
                        predicted_pos[i] += 0.35 * correction
                        predicted_pos[i + 1] -= 0.65 * correction

        # Rooftop / Surface Contact Physics
        contact_depth = target_surface_z + 0.06 - predicted_pos[-1][2]
        if contact_depth > 0.0:
            k_spring = 340.0
            d_damper = 14.0
            f_spring = k_spring * contact_depth - d_damper * min(0.0, float(self.nodes_vel[-1][2]))
            self.contact_normal_force = float(np.clip(f_spring, 0.0, 1.5 * PAYLOAD_WEIGHT_N))
            predicted_pos[-1][2] = target_surface_z + 0.06
            self.nodes_vel[-1][2] = 0.0
            self.nodes_vel[-1][:2] *= 0.85
        else:
            self.contact_normal_force = 0.0

        for k in range(1, self.num_nodes):
            self.nodes_vel[k] = (predicted_pos[k] - self.nodes_pos[k]) / dt
            self.nodes_pos[k] = predicted_pos[k].copy()

        self.sway_offset = self.nodes_pos[-1][:2] - self.nodes_pos[0][:2]
        
        theta_x = math.atan2(float(self.sway_offset[0]), L)
        theta_y = math.atan2(float(self.sway_offset[1]), L)
        self.theta_dot = (np.array([theta_x, theta_y], dtype=np.float32) - self.theta) / dt
        self.theta = np.array([theta_x, theta_y], dtype=np.float32)

        mid_node = self.num_nodes // 2
        straight_mid = 0.5 * (self.nodes_pos[0][:2] + self.nodes_pos[-1][:2])
        self.catenary_midpoint = self.nodes_pos[mid_node][:2] - straight_mid

        # Dynamic Payload Release Check (Physical Touchdown Dwell)
        released_this_step = False
        cargo_clearance = (self.nodes_pos[-1][2] - target_surface_z)
        if self.has_payload and (cargo_clearance <= 0.12 or self.contact_normal_force >= 0.40 * PAYLOAD_WEIGHT_N):
            self.touchdown_timer += dt
            if self.touchdown_timer >= 0.18:
                self.has_payload = False
                released_this_step = True
                self.touchdown_timer = 0.0
        else:
            self.touchdown_timer = max(0.0, self.touchdown_timer - dt)

        # Dynamic Cable Tension
        if self.has_payload:
            theta_mag = float(np.linalg.norm(self.theta))
            t_grav = max(0.5, CARGO_MASS_KG * (G_ACCEL * math.cos(theta_mag)) - self.contact_normal_force)
            t_centrif = CARGO_MASS_KG * L * float(np.dot(self.theta_dot, self.theta_dot))
            new_tension = float(np.clip(t_grav + t_centrif, 1.2, 45.0))
        else:
            new_tension = 1.2

        self.tension_derivative = (new_tension - self.prev_tension_n) / dt
        self.prev_tension_n = self.tension_n
        self.tension_n = new_tension

        return self.cable_len, self.tension_n, self.sway_offset, released_this_step

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

AutonomousWinchDeliverySystem = ContinuousWinchTetherSystem

# =============================================================================
# 8. BEMT COAXIAL AIRFRAME 8-DOF WITH RK4 & WALL INTERACTION
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
        self.vrs_turb_intensity = 0.0

    def _compute_derivatives(self, p, v, R, omega, motor_omegas, current_wind, rho_air, 
                             ice_index, surface_z, building_walls, current_mass):
        z_h = max(0.02, float(p[2] - surface_z))
        ratio_zh_r = z_h / ROTOR_RADIUS_M
        v_h_hover = math.sqrt(max(0.1, (current_mass * G_ACCEL) / (2.0 * rho_air * ROTOR_DISK_AREA * 4.0)))
        advance_ratio_mu = float(np.linalg.norm(v[:2])) / max(1.0, 4.0 * v_h_hover)

        if ratio_zh_r >= 0.5:
            ige_static = 1.0 / (1.0 - (1.0 / (4.0 * ratio_zh_r)) ** 2)
            ige_static = float(np.clip(ige_static, 1.0, 1.28))
        else:
            ige_static = float(1.20 + 0.16 * (ratio_zh_r / 0.5))
        ige_factor = 1.0 + (ige_static - 1.0) / (1.0 + 4.5 * (advance_ratio_mu ** 2))

        v_z_descent = -v[2]
        vrs_factor = 1.0
        self.vrs_turb_intensity = 0.0
        if v_z_descent > 0.4 * v_h_hover and v_z_descent < 1.6 * v_h_hover and abs(v[0]) < 2.0 and abs(v[1]) < 2.0:
            v_ratio = (v_z_descent - 0.4 * v_h_hover) / (1.2 * v_h_hover)
            vrs_factor = 1.0 - 0.28 * math.sin(math.pi * v_ratio)
            self.vrs_turb_intensity = 0.35 * math.sin(math.pi * v_ratio)

        lift_loss_factor = (1.0 - 0.28 * ice_index) * vrs_factor
        drag_rise_factor = 1.0 + 1.80 * ice_index

        T = np.zeros(8, dtype=np.float32)
        Q = np.zeros(8, dtype=np.float32)
        for i in range(8):
            if self.motor_health[i] == 0.0: continue
            is_bottom = (i % 2 == 1)
            scale_c = (C_T_CLEAN * COAXIAL_EFFICIENCY) if is_bottom else C_T_CLEAN
            t_rot = scale_c * (motor_omegas[i] ** 2) * ige_factor * lift_loss_factor
            q_rot = C_Q_CLEAN * (motor_omegas[i] ** 2) * drag_rise_factor
            T[i] = max(0.0, t_rot)
            Q[i] = max(0.0, q_rot)

        f_thrust = float(np.sum(T))
        arm = ARM_LENGTH_M / math.sqrt(2.0)
        tau_x = arm * (- (T[0] + T[1]) + (T[2] + T[3]) + (T[4] + T[5]) - (T[6] + T[7]))
        tau_y = arm * (- (T[0] + T[1]) - (T[2] + T[3]) + (T[4] + T[5]) + (T[6] + T[7]))
        tau_z = -Q[0] + Q[1] + Q[2] - Q[3] - Q[4] + Q[5] + Q[6] - Q[7]

        if self.vrs_turb_intensity > 0.05:
            tau_x += float(np.random.normal(0, self.vrs_turb_intensity * 0.03))
            tau_y += float(np.random.normal(0, self.vrs_turb_intensity * 0.03))

        torques = np.array([tau_x, tau_y, tau_z], dtype=np.float32)

        f_wall_suction = np.zeros(3, dtype=np.float32)
        if building_walls:
            px, py, pz = p
            for w in building_walls:
                x1, x2, y1, y2, hh = w["aabb"]
                if pz <= hh + 1.0:
                    dx = max(0.0, max(x1 - px, px - x2))
                    dy = max(0.0, max(y1 - py, py - y2))
                    wall_dist = math.hypot(dx, dy)
                    if 0.2 < wall_dist < 3.2:
                        suction_mag = (0.5 * rho_air * (v_h_hover**2) * 0.08) * ((ROTOR_RADIUS_M / wall_dist) ** 2)
                        n_wall = np.array([(x1+x2)/2.0 - px, (y1+y2)/2.0 - py, 0.0], dtype=np.float32)
                        n_wall /= max(1e-4, float(np.linalg.norm(n_wall)))
                        f_wall_suction += suction_mag * n_wall

        v_air = v - current_wind
        speed_air = float(np.linalg.norm(v_air))
        f_drag = -0.5 * rho_air * np.array([0.22, 0.22, 0.38], dtype=np.float32) * v_air * max(1.0, speed_air)
        f_body = np.array([0.0, 0.0, f_thrust], dtype=np.float32)
        f_gravity = current_mass * GRAVITY_G

        a_linear = (R @ f_body + f_gravity + f_drag + f_wall_suction) / current_mass
        a_linear = np.clip(a_linear, -35.0, 35.0)

        gyroscopic = np.cross(omega, INERTIA_MATRIX @ omega)
        a_angular = INV_INERTIA @ (torques - gyroscopic)

        p_watts = float(np.sum(Q * motor_omegas)) / 0.85 + 24.0
        return a_linear, a_angular, f_thrust, torques, p_watts, ige_factor

    def step(self, motor_cmds, current_wind, rho_air, ice_index=0.0, surface_z=0.0, 
             has_payload=True, building_walls=None, dt=0.016):
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

        is_on_surface = (self.p[2] <= surface_z + 0.03)
        est_thrust = float(np.sum(C_T_CLEAN * (self.motor_omegas ** 2)))
        if is_on_surface and est_thrust < (0.45 * weight_force):
            self.p[2] = surface_z
            self.v.fill(0.0)
            self.omega.fill(0.0)
            self.accel_linear.fill(0.0)
            self.R = np.eye(3, dtype=np.float32)
            return 0.0, np.zeros(3, dtype=np.float32), 24.0, 1.0

        a1_lin, a1_ang, f_thrust, torques, p_watts, ige_fac = self._compute_derivatives(
            self.p, self.v, self.R, self.omega, self.motor_omegas, current_wind, 
            rho_air, ice_index, surface_z, building_walls, current_mass
        )
        k1_v, k1_p, k1_w = a1_lin, self.v, a1_ang

        p_k2 = self.p + 0.5 * dt * k1_p
        v_k2 = self.v + 0.5 * dt * k1_v
        w_k2 = self.omega + 0.5 * dt * k1_w
        a2_lin, a2_ang, _, _, _, _ = self._compute_derivatives(
            p_k2, v_k2, self.R, w_k2, self.motor_omegas, current_wind, 
            rho_air, ice_index, surface_z, building_walls, current_mass
        )
        k2_v, k2_p, k2_w = a2_lin, v_k2, a2_ang

        p_k3 = self.p + 0.5 * dt * k2_p
        v_k3 = self.v + 0.5 * dt * k2_v
        w_k3 = self.omega + 0.5 * dt * k2_w
        a3_lin, a3_ang, _, _, _, _ = self._compute_derivatives(
            p_k3, v_k3, self.R, w_k3, self.motor_omegas, current_wind, 
            rho_air, ice_index, surface_z, building_walls, current_mass
        )
        k3_v, k3_p, k3_w = a3_lin, v_k3, a3_ang

        p_k4 = self.p + dt * k3_p
        v_k4 = self.v + dt * k3_v
        w_k4 = self.omega + dt * k3_w
        a4_lin, a4_ang, _, _, _, _ = self._compute_derivatives(
            p_k4, v_k4, self.R, w_k4, self.motor_omegas, current_wind, 
            rho_air, ice_index, surface_z, building_walls, current_mass
        )
        k4_v, k4_p, k4_w = a4_lin, v_k4, a4_ang

        self.p += (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
        self.v += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        self.omega += (dt / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w)
        self.accel_linear = a1_lin

        if self.p[2] <= surface_z + 0.04 and self.v[2] < 0.0:
            self.p[2] = surface_z
            self.v[2] = 0.0
            self.v[0] *= 0.70
            self.v[1] *= 0.70
            self.omega[0] *= 0.10
            self.omega[1] *= 0.10
            self.omega[2] *= 0.20

        d_theta = float(np.linalg.norm(self.omega)) * dt
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

        self.last_p_watts = p_watts
        self.last_ige = ige_fac
        return f_thrust, torques, p_watts, ige_fac

# =============================================================================
# 9. MULTI-MODAL PHYSICAL SENSOR SUITE
# =============================================================================
class AdvancedMultiModalSensorSuite:
    def __init__(self, world: MetropolitanCityGridWorld400m):
        self.world = world
        self._wind_sensor_state = {}
        self._wind_sensor_bias = {}

    def poll_wind_vector(self, drone_v, current_wind, drone_R, dt=0.016, sensor_id=0, ground_velocity_est=None):
        key = int(sensor_id)
        v_ground_true = np.asarray(drone_v, dtype=np.float32).reshape(3)
        v_ground_for_sensor = (v_ground_true if ground_velocity_est is None else
                               np.asarray(ground_velocity_est, dtype=np.float32).reshape(3))
        wind_truth = np.asarray(current_wind, dtype=np.float32).reshape(3)
        rotation = np.asarray(drone_R, dtype=np.float32).reshape(3, 3)
        dt = float(np.clip(dt, 1e-4, 0.25))

        v_ground_meas = v_ground_for_sensor + np.random.normal(0.0, 0.045, 3).astype(np.float32)
        air_rel_world = v_ground_true - wind_truth
        air_rel_body = rotation.T @ air_rel_world
        bias = self._wind_sensor_bias.get(key)
        if bias is None:
            bias = np.zeros(3, dtype=np.float32)
        bias = (bias + np.random.normal(0.0, 0.0018 * math.sqrt(dt / 0.016), 3)).astype(np.float32)
        self._wind_sensor_bias[key] = bias
        air_rel_body_meas = air_rel_body + bias + np.random.normal(0.0, 0.14, 3).astype(np.float32)
        air_rel_world_meas = rotation @ air_rel_body_meas

        wind_raw = v_ground_meas - air_rel_world_meas
        wind_raw[:2] = np.clip(wind_raw[:2], -12.0, 12.0)
        wind_raw[2] = float(np.clip(wind_raw[2], -4.0, 2.0))
        previous = self._wind_sensor_state.get(key)
        if previous is None:
            estimate = wind_raw.copy()
        else:
            alpha = dt / (0.16 + dt)
            estimate = previous + alpha * (wind_raw - previous)
        estimate = np.asarray(estimate, dtype=np.float32)
        self._wind_sensor_state[key] = estimate.copy()

        horizontal_speed = float(np.linalg.norm(estimate[:2]))
        if horizontal_speed < 0.25:
            wind_from_deg = 0.0
        else:
            wind_from_deg = float((math.degrees(math.atan2(-estimate[0], -estimate[1])) + 360.0) % 360.0)
        quality = min(0.99, max(0.55, float(1.0 - 0.025 * np.linalg.norm(wind_raw - estimate))))
        return estimate.copy(), horizontal_speed, wind_from_deg, quality

    def poll_optical_flow(self, drone_p, drone_v, drone_R, drone_omega, surface_z=0.0):
        cos_tilt = max(0.35, float(drone_R[2, 2]))
        z_rel = max(0.15, float(drone_p[2] - surface_z))
        z_slant = z_rel / cos_tilt
        
        v_body = drone_R.T @ drone_v
        
        w_flow_x = (v_body[0] / z_slant) - drone_omega[1] + float(np.random.normal(0, 0.015))
        w_flow_y = (v_body[1] / z_slant) + drone_omega[0] + float(np.random.normal(0, 0.015))
        
        quality = 0.95 if z_rel < 35.0 else float(np.clip(1.0 - (z_rel - 35.0) / 25.0, 0.1, 0.95))
        return np.array([w_flow_x, w_flow_y], dtype=np.float32), quality, z_slant

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
        hdop = 3.8 if is_spoofed else float(np.clip(0.85 + float(np.random.normal(0, 0.05)), 0.6, 2.0))
        cn0_avg = 28.5 if is_spoofed else float(np.clip(44.0 + float(np.random.normal(0, 1.2)), 32.0, 48.0))
        rf_jam_pwr = -52.0 if is_spoofed else -110.0
        return hdop, cn0_avg, rf_jam_pwr, 0.02

# =============================================================================
# 10. VECTORIZED MACRO LIDAR & ASSURANCE SUITE
# =============================================================================
class MacroPerceptionLiDARSuite:
    def __init__(self, world: MetropolitanCityGridWorld400m):
        self.world = world

    def cast_360_lidar(self, drone_pos, yaw_rad, dynamic_vehicles=None):
        px, py, pz = drone_pos
        ranges = np.full(16, 30.0, dtype=np.float32)
        angles = yaw_rad + np.linspace(0, 2.0 * math.pi, 16, endpoint=False)
        dx_all = np.cos(angles).astype(np.float32)
        dy_all = np.sin(angles).astype(np.float32)

        aabbs = self.world.spatial_hash.cached_aabbs
        if aabbs is None or len(aabbs) == 0:
            return ranges

        valid_bldgs = aabbs[pz <= (aabbs[:, 4] + 0.1)]
        if len(valid_bldgs) == 0:
            return ranges

        bx1, bx2, by1, by2 = valid_bldgs[:, 0], valid_bldgs[:, 1], valid_bldgs[:, 2], valid_bldgs[:, 3]

        for beam_idx in range(16):
            dx, dy = dx_all[beam_idx], dy_all[beam_idx]

            if abs(dx) > 1e-5:
                tx1 = (bx1 - px) / dx
                tx2 = (bx2 - px) / dx
                t_min_x = np.minimum(tx1, tx2)
                t_max_x = np.maximum(tx1, tx2)
            else:
                x_in = (px >= bx1) & (px <= bx2)
                t_min_x = np.where(x_in, -1e6, 1e6)
                t_max_x = np.where(x_in, 1e6, -1e6)

            if abs(dy) > 1e-5:
                ty1 = (by1 - py) / dy
                ty2 = (by2 - py) / dy
                t_min_y = np.minimum(ty1, ty2)
                t_max_y = np.maximum(ty1, ty2)
            else:
                y_in = (py >= by1) & (py <= by2)
                t_min_y = np.where(y_in, -1e6, 1e6)
                t_max_y = np.where(y_in, 1e6, -1e6)

            t_enter = np.maximum(t_min_x, t_min_y)
            t_exit = np.minimum(t_max_x, t_max_y)

            hit_mask = (t_enter <= t_exit) & (t_exit > 0.0) & (t_enter < 30.0)
            if np.any(hit_mask):
                valid_hits = np.maximum(0.0, t_enter[hit_mask])
                ranges[beam_idx] = min(ranges[beam_idx], float(np.min(valid_hits)))

            if dynamic_vehicles:
                for v in dynamic_vehicles:
                    vx, vy = v["pos"][0], v["pos"][1]
                    d_veh = math.hypot(px - vx, py - vy)
                    if pz < 3.2 and d_veh < ranges[beam_idx]:
                        bearing = math.atan2(vy - py, vx - px)
                        diff = abs((bearing - angles[beam_idx] + math.pi) % (2.0 * math.pi) - math.pi)
                        if diff <= (math.pi / 16.0):
                            ranges[beam_idx] = float(d_veh)

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

        cone_angle = math.radians(25.0)
        sin_c, cos_c = math.sin(cone_angle), math.cos(cone_angle)

        for i in range(8):
            ang = 2.0 * math.pi * i / 8.0
            beam_offset_xy = np.array([sin_c * math.cos(ang), sin_c * math.sin(ang)], dtype=np.float32)
            sample_pt = drone_pos[:2] + beam_offset_xy * 1.5
            surf_h = self.world.get_surface_height(sample_pt[0], sample_pt[1])
            true_clearance = max(0.05, float(pz - surf_h))
            conical_ranges[i] = min(15.0, float(true_clearance / max(0.1, cos_c)))

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
                return np.array([float((x1 + x2) / 2.0), float((y1 + y2) / 2.0), hh], dtype=np.float32)

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
        return {
            "z_baro": z_baro, 
            "rho_air": rho_air, 
            "v_tas": v_tas, 
            "q_dyn": q_dyn, 
            "emf_rms": emf_rms, 
            "min_sonar": min_sonar
        }

    # =============================================================================
# 11. MATTER-WAVE QUANTUM SENSOR & REGULARIZED ERROR-STATE EKF ON SO(3)
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
        self.Q_base = np.diag([1e-4]*3 + [1e-3]*3 + [1e-4]*3 + [1e-5]*3 + [1e-6]*3).astype(np.float32)
        self.Q_d = self.Q_base.copy()
        self.residual_scale = 1.0

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

        # Sage-Husa adaptive process noise inflation
        self.Q_d = self.Q_base * max(1.0, float(self.residual_scale))
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

    def update_measurement(self, H, innovation, R_cov, chi2_threshold=11.34):
        S = H @ self.P @ H.T + R_cov
        S_reg = S + np.eye(S.shape[0], dtype=np.float32) * 1e-6

        try:
            PHt = self.P @ H.T
            K = np.linalg.solve(S_reg, PHt.T).T
            innov_flat = np.array(innovation).flatten()
            S_inv_innov = np.linalg.solve(S_reg, innov_flat)
            gamma = float(innov_flat @ S_inv_innov)
        except np.linalg.LinAlgError:
            return

        d_k = 0.05
        expected_gamma = float(len(innov_flat))
        self.residual_scale = (1.0 - d_k) * self.residual_scale + d_k * (gamma / max(1.0, expected_gamma))
        self.residual_scale = float(np.clip(self.residual_scale, 0.8, 5.0))

        if gamma > chi2_threshold:
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

    def update_heading(self, measured_yaw_rad, r_cov=0.02):
        R_curr = quat_to_rot(self.q)
        current_yaw = math.atan2(R_curr[1, 0], R_curr[0, 0])
        yaw_innov = (measured_yaw_rad - current_yaw + math.pi) % (2.0 * math.pi) - math.pi

        denom = max(1e-6, R_curr[0, 0]**2 + R_curr[1, 0]**2)
        H_yaw = np.zeros((1, 15), dtype=np.float32)
        H_yaw[0, 6] = 0.0
        H_yaw[0, 7] = (-R_curr[0, 0] * R_curr[1, 2] + R_curr[1, 0] * R_curr[0, 2]) / denom
        H_yaw[0, 8] = ( R_curr[0, 0] * R_curr[1, 1] - R_curr[1, 0] * R_curr[0, 1]) / denom

        self.update_measurement(H_yaw, np.array([yaw_innov], dtype=np.float32), np.array([[r_cov]], dtype=np.float32))

    def update_optical_flow_velocity_adaptive(self, v_flow_world, flow_quality=0.95, z_slant=1.0, r_cov_base=0.015):
        alt_factor = (min(45.0, z_slant) / 10.0) ** 2
        r_adaptive = (r_cov_base / max(0.05, flow_quality)) + alt_factor * 0.08
        r_adaptive = float(np.clip(r_adaptive, 0.02, 1.20))

        H_v = np.zeros((2, 15), dtype=np.float32)
        H_v[0, 3] = 1.0
        H_v[1, 4] = 1.0
        v_innov = v_flow_world[:2] - self.v[:2]
        self.update_measurement(H_v, v_innov, np.eye(2, dtype=np.float32) * r_adaptive)

    def update_baro_altitude_adaptive(self, z_baro, q_dyn=0.0, r_cov_base=0.05):
        r_adaptive = float(r_cov_base + 0.0022 * q_dyn)
        r_adaptive = float(np.clip(r_adaptive, 0.05, 0.40))

        H_baro = np.zeros((1, 15), dtype=np.float32)
        H_baro[0, 2] = 1.0
        innov = np.array([z_baro - self.p[2]], dtype=np.float32)
        self.update_measurement(H_baro, innov, np.array([[r_adaptive]], dtype=np.float32))

    def update_gps_position_adaptive(self, p_gps, hdop=0.85, cn0_avg=44.0, is_spoofed=False, r_cov_base=0.04):
        if is_spoofed:
            r_adaptive = 45.0
        else:
            r_adaptive = r_cov_base * (hdop ** 2) * (44.0 / max(15.0, cn0_avg))
        r_adaptive = float(np.clip(r_adaptive, 0.03, 50.0))

        H_pos = np.zeros((3, 15), dtype=np.float32)
        H_pos[0:3, 0:3] = np.eye(3, dtype=np.float32)
        innov = p_gps - self.p
        self.update_measurement(H_pos, innov, np.eye(3, dtype=np.float32) * r_adaptive)

# =============================================================================
# 12. PASSIVE INVARIANT SAFETY SHIELD & SE(3) ACTUATOR MIXER
# =============================================================================
class PassiveInterventionCBFShield:
    """
    Emergency Invariant Set Safety Shield.
    Unlike classical human planners, this shield remains completely passive and transparent.
    It does not restrict tilt, dictate paths, or throttle velocity. It intervenes only
    if an imminent structural or mesh collision boundary is violated.
    """
    def __init__(self):
        self.alpha_obs = CBF_ALPHA_OBSTACLE
        self.safe_radius = CBF_BASE_SAFE_RADIUS_M
        self.swarm_r = SWARM_INTER_DRONE_SAFE_RADIUS_M
        self.downwash_expansion = DOWNWASH_CONE_EXPANSION
        self.downwash_depth = DOWNWASH_HAZARD_DEPTH_M

    def evaluate_shield(self, drone_idx, drone_p, drone_v, continuous_accel_cmd, obstacles, swarm_states=None):
        A_rows = []
        b_rows = []

        # 1. Static and Dynamic Obstacle Boundaries
        for obs in obstacles:
            rel_pos = drone_p - obs["pos"]
            dist_xy = math.hypot(rel_pos[0], rel_pos[1])
            obs_h = obs.get("height", 0.0)

            n_xy = rel_pos[:2] / max(1e-4, dist_xy)
            v_closing = -float(np.dot(drone_v[:2], n_xy))

            # Only activate barrier if drone is actively penetrating safe boundary
            h_obs = dist_xy - (obs["radius"] + self.safe_radius)
            if drone_p[2] <= (obs_h - 0.2) and h_obs < 2.5:
                grad_h = np.array([n_xy[0], n_xy[1], 0.0], dtype=np.float32)
                l_fh = float(np.dot(grad_h, drone_v))
                A_rows.append(grad_h)
                b_rows.append(-l_fh - self.alpha_obs * h_obs)

        # 2. Inter-Drone Swarm Separation & Downwash Hazard
        if swarm_states:
            for j, other in enumerate(swarm_states):
                if j == drone_idx:
                    continue
                d_p = drone_p[:2] - other["p"][:2]
                dist_ij = float(np.linalg.norm(d_p))
                dz = drone_p[2] - other["p"][2]

                # Downwash cone barrier
                if -self.downwash_depth < dz < -0.4:
                    cone_r = self.swarm_r + self.downwash_expansion * abs(dz)
                    h_downwash = dist_ij - cone_r
                    if h_downwash < 1.8:
                        n_dw = np.array([d_p[0]/max(1e-4, dist_ij), d_p[1]/max(1e-4, dist_ij), 0.0], dtype=np.float32)
                        v_rel = drone_v[:2] - other["v"][:2]
                        A_rows.append(n_dw)
                        b_rows.append(-float(np.dot(n_dw[:2], v_rel)) - 3.0 * h_downwash)

                # Direct horizontal proximity barrier
                if abs(dz) < 2.5:
                    h_swarm = dist_ij - self.swarm_r
                    if h_swarm < 2.0:
                        n_ij = np.array([d_p[0]/max(1e-4, dist_ij), d_p[1]/max(1e-4, dist_ij), 0.0], dtype=np.float32)
                        v_rel = drone_v[:2] - other["v"][:2]
                        A_rows.append(n_ij)
                        b_rows.append(-float(np.dot(n_ij[:2], v_rel)) - 2.5 * h_swarm)

        # 3. Ground Plane Barrier
        h_gnd = drone_p[2] - 0.15
        if h_gnd < 1.0 and drone_p[2] > 0.18:
            A_rows.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
            b_rows.append(-3.0 * drone_v[2] - 4.0 * h_gnd)

        # If completely safe, pass the neural command through unmodified (100% transparent)
        if not A_rows:
            return continuous_accel_cmd

        # Minimal QP projection: find closest safe acceleration to neural command
        A = np.array(A_rows, dtype=np.float32)
        b = np.array(b_rows, dtype=np.float32)
        u = continuous_accel_cmd.copy()

        for _ in range(8):
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
    """
    Direct Geometric SE(3) Control Allocator with Passive Shielding.
    Translates continuous acceleration vectors into 8-rotor coaxial RPM commands.
    """
    def __init__(self):
        self.shield = PassiveInterventionCBFShield()
        self.inflow_estimator = OnlineRotorInflowAndHealthEstimator()
        self._last_motor_omegas = np.full(8, HOVER_OMEGA_NOMINAL, dtype=np.float32)

        arm = ARM_LENGTH_M / math.sqrt(2.0)
        cq_ct_top = C_Q_CLEAN / C_T_CLEAN
        cq_ct_bot = (C_Q_CLEAN / C_T_CLEAN) / COAXIAL_EFFICIENCY
        
        self.B_nominal = np.array([
            [ 1.0,         1.0,        1.0,         1.0,        1.0,         1.0,        1.0,         1.0],
            [-arm,        -arm,        arm,         arm,        arm,         arm,       -arm,        -arm],
            [-arm,        -arm,       -arm,        -arm,        arm,         arm,        arm,         arm],
            [-cq_ct_top, cq_ct_bot, cq_ct_top, -cq_ct_bot, -cq_ct_top, cq_ct_bot, cq_ct_top, -cq_ct_bot]
        ], dtype=np.float32)

    def mix(self, drone_health, p_est, v_est, R_est, omega_est, neural_accel_cmd,
            obstacles, swarm_states=None, drone_idx=0, has_payload=True, dt=0.016):

        current_mass = DRY_MASS_KG + (CARGO_MASS_KG if has_payload else 0.0)
        
        # 1. Apply Passive Safety Shield
        a_safe = self.shield.evaluate_shield(
            drone_idx=drone_idx, drone_p=p_est, drone_v=v_est,
            continuous_accel_cmd=neural_accel_cmd, obstacles=obstacles, swarm_states=swarm_states
        )
        
        # 2. Compute Desired Force Vector (Compensating Gravity)
        F_d = a_safe * current_mass + current_mass * (-GRAVITY_G)

        # 3. Geometric Attitude Formulation on SO(3)
        b3 = R_est[:, 2]
        f_thrust = float(np.dot(F_d, b3))
        f_thrust = float(np.clip(f_thrust, 0.0, 38.0))

        b3_d = F_d / max(1e-4, float(np.linalg.norm(F_d)))
        yaw_target = 0.0
        b1_c = np.array([math.cos(yaw_target), math.sin(yaw_target), 0.0], dtype=np.float32)
        b2_d = np.cross(b3_d, b1_c)
        b2_d /= max(1e-4, float(np.linalg.norm(b2_d)))
        b1_d = np.cross(b2_d, b3_d)
        R_d = np.column_stack((b1_d, b2_d, b3_d)).astype(np.float32)

        # Attitude error on Lie Algebra so(3)
        R_err_mat = R_d.T @ R_est - R_est.T @ R_d
        e_R = 0.5 * vee_map(R_err_mat)
        e_w = omega_est

        k_R = np.array([2.40, 2.40, 1.80], dtype=np.float32)
        k_w = np.array([1.20, 1.20, 0.85], dtype=np.float32)
        tau = -k_R * e_R - k_w * e_w
        tau = np.clip(tau, -0.75, 0.75)

        # 4. Actuator Pseudo-Inverse Allocation
        desired_wrench = np.array([f_thrust, tau[0], tau[1], tau[2]], dtype=np.float32)
        W_effective = drone_health.copy()
        W_diag = np.diag(W_effective).astype(np.float32)
        B_active = self.B_nominal @ W_diag

        try:
            B_pinv = np.linalg.pinv(B_active)
            T_cmds = B_pinv @ desired_wrench
        except np.linalg.LinAlgError:
            T_cmds = np.full(8, f_thrust / 8.0, dtype=np.float32)

        T_cmds = np.clip(T_cmds * W_effective, 0.0, 10.5)

        # Convert rotor thrust to commanded motor angular velocities
        omega_cmds = np.zeros(8, dtype=np.float32)
        for i in range(8):
            if W_effective[i] > 0.0:
                base_c = (C_T_CLEAN * COAXIAL_EFFICIENCY) if (i % 2 == 1) else C_T_CLEAN
                omega_cmds[i] = math.sqrt(T_cmds[i] / max(1e-6, base_c))

        self._last_motor_omegas = omega_cmds.copy()
        return omega_cmds, f_thrust, a_safe

# =============================================================================
# 13. FIXED 139-DIM CONTINUOUS SENSORIMOTOR OBSERVATION SYNTHESIZER
# =============================================================================
class DroneObservationSynthesizer139Dim:
    def __init__(self):
        self.dim = BASE_OBS_DIM

    def synthesize_instantaneous(self, p_est, v_est, R_est, omega_est, target_p, lidar_16, conical_8,
                                 v_tas, q_dyn, min_sonar, wind_est, ice_idx, cable_len, cable_vel,
                                 tension_n, sway_2d, bms_soc, v_pack, bms_temp_c, r0_dyn,
                                 chi2_val, jerk_discrepancy, is_spoofed, emf_rms, wire_dist,
                                 agent_dist, ground_spl_dba, has_payload, hook_a, hook_w,
                                 load_cell_3d, thermal_dt, dolp, ndvi, esc_i_rms, esc_t_mosfet,
                                 alpha_deg, beta_deg, rh_pct, dust_pm10, hdop, cn0_avg,
                                 rf_jam_pwr, mag_grad, airframe_accel, airframe_gyro,
                                 peer_swarm_kinematics, optical_flow_2d, flow_quality,
                                 aruco_offset_3d, aruco_locked, skid_force_left, skid_force_right,
                                 is_skid_grounded, siren_angle, siren_freq, siren_spl,
                                 adsb_rel_pos, adsb_closing_spd, dt=0.016):
        out = np.zeros(self.dim, dtype=np.float32)

        out[0:3] = (target_p - p_est) / 50.0
        out[3:6] = p_est / 200.0
        out[6:9] = v_est / 8.0
        out[9:18] = R_est.flatten()
        out[18:21] = omega_est / 10.0
        out[21:37] = lidar_16 / 30.0
        out[37] = min_sonar / 15.0
        out[38] = v_tas / 20.0
        out[39] = q_dyn / 100.0
        out[40] = wire_dist / 50.0
        out[41] = agent_dist / 50.0
        out[42:45] = wind_est / 8.0
        out[45:53] = conical_8 / 15.0
        out[53] = cable_len / 15.0
        out[54] = cable_vel / 5.0
        out[55] = tension_n / 50.0
        out[56] = sway_2d[0] / 2.0
        out[57] = sway_2d[1] / 2.0
        out[58] = 1.0 if has_payload else 0.0
        out[59] = skid_force_left / 20.0
        out[60] = skid_force_right / 20.0
        out[61] = bms_soc
        out[62] = v_pack / 25.2
        out[63] = bms_temp_c / 60.0
        out[64] = r0_dyn / 0.05
        out[65] = ice_idx
        out[66] = 1.0 if is_skid_grounded else 0.0
        out[67] = chi2_val / 10.0
        out[68] = jerk_discrepancy / 16.0
        out[69:77] = esc_i_rms / 25.0
        out[77:85] = esc_t_mosfet / 100.0
        out[85:88] = hook_a / 20.0
        out[88:91] = hook_w / 10.0
        out[91:94] = load_cell_3d / 50.0
        out[94] = alpha_deg / 45.0
        out[95] = beta_deg / 45.0
        out[96] = rh_pct / 100.0
        out[97] = dust_pm10 / 100.0
        out[98] = thermal_dt / 10.0
        out[99] = dolp
        out[100] = hdop / 5.0
        out[101] = cn0_avg / 50.0
        out[102] = (rf_jam_pwr + 120.0) / 70.0
        out[103] = mag_grad / 1.0
        out[104] = optical_flow_2d[0] / 5.0
        out[105] = optical_flow_2d[1] / 5.0
        out[106] = flow_quality
        out[107] = 1.0 if is_spoofed else 0.0
        out[108:111] = aruco_offset_3d / 10.0
        out[111] = aruco_locked
        out[112] = siren_angle / math.pi
        out[113] = siren_spl / 100.0
        out[114:117] = adsb_rel_pos / 200.0
        out[117] = adsb_closing_spd / 30.0

        p_idx = 118
        for peer in peer_swarm_kinematics[:2]:
            out[p_idx:p_idx+3] = peer["p_rel"] / 50.0
            out[p_idx+3:p_idx+6] = peer["v_rel"] / 10.0
            p_idx += 6

        out[130:133] = airframe_accel / 30.0
        out[133:136] = airframe_gyro / 10.0
        out[136] = ndvi
        out[137] = emf_rms / 100.0
        out[138] = ground_spl_dba / 100.0
        return out

    def stack_history(self, history_q, current_obs):
        if len(history_q) == 0:
            for _ in range(history_q.maxlen or HISTORY_LEN):
                history_q.append(current_obs.copy())
        else:
            history_q.append(current_obs.copy())
        return np.concatenate(list(history_q), axis=0).astype(np.float32)

# =============================================================================
# 14. HIGH-THROUGHPUT VECTORIZED GPU TENSOR SWARM ENVIRONMENT
# =============================================================================
class BatchedPyTorchSwarmEnv:
    """
    Massively Parallel Post-Human Swarm Environment on GPU Tensors.
    Eliminates all human FSM locks: policy outputs continuous 3D acceleration
    and continuous cable rate simultaneously. Drones can execute dynamic drops.
    """
    def __init__(self, num_envs=4096, num_drones=NUM_SWARM_DRONES, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.B = num_envs
        self.D = num_drones
        self.device = torch.device(device)
        self.dt = 0.016
        
        self.gravity = torch.tensor([0.0, 0.0, -G_ACCEL], device=self.device)
        self.dry_mass = DRY_MASS_KG
        self.cargo_mass = CARGO_MASS_KG
        self.cda = torch.tensor([0.22, 0.22, 0.38], device=self.device)
        
        self.pos = torch.zeros((self.B, self.D, 3), device=self.device)
        self.vel = torch.zeros((self.B, self.D, 3), device=self.device)
        self.accel_prev = torch.zeros((self.B, self.D, 3), device=self.device)
        
        self.cable_len = torch.full((self.B, self.D), 0.08, device=self.device)
        self.cable_vel = torch.zeros((self.B, self.D), device=self.device)
        self.has_payload = torch.ones((self.B, self.D), dtype=torch.bool, device=self.device)
        self.touchdown_timer = torch.zeros((self.B, self.D), device=self.device)
        
        self.target_pos = torch.zeros((self.B, self.D, 3), device=self.device)
        self.roof_heights = torch.zeros((self.B, self.D), device=self.device)
        self.battery_soc = torch.full((self.B, self.D), 0.98, device=self.device)
        self.prev_dist = torch.zeros((self.B, self.D), device=self.device)
        # Timer to track how many steps each drone has flown:
        self.step_counts = torch.zeros((self.B, self.D), device=self.device, dtype=torch.long)
        
        self.wind = torch.zeros((self.B, 3), device=self.device)
        self.obs_history = torch.zeros((self.B, self.D, HISTORY_LEN, BASE_OBS_DIM), device=self.device)
        
        world_ref = MetropolitanCityGridWorld400m()
        aabbs = [h["aabb"] for h in world_ref.houses]
        self.building_aabbs = torch.tensor(np.array(aabbs, dtype=np.float32), device=self.device)
        self.depot_helipads = torch.tensor(np.array(world_ref.depot_helipads, dtype=np.float32), device=self.device)
        
        self.peer0_idx = torch.tensor([1, 0, 0], device=self.device, dtype=torch.long)
        self.peer1_idx = torch.tensor([2, 2, 1], device=self.device, dtype=torch.long)
        self.lidar_angles = torch.arange(16, dtype=torch.float32, device=self.device) * (2.0 * math.pi / 16.0)
        self.lidar_rx = torch.cos(self.lidar_angles).reshape(1, 1, 16, 1)
        self.lidar_ry = torch.sin(self.lidar_angles).reshape(1, 1, 16, 1)
        self.lidar_rx = torch.where(self.lidar_rx.abs() < 1e-6, torch.full_like(self.lidar_rx, 1e-6), self.lidar_rx)
        self.lidar_ry = torch.where(self.lidar_ry.abs() < 1e-6, torch.full_like(self.lidar_ry, 1e-6), self.lidar_ry)

        self.reset_all()

    def reset_all(self):
        roof_indices = torch.randint(0, len(self.building_aabbs), (self.B, self.D), device=self.device)
        chosen_bldgs = self.building_aabbs[roof_indices]
        
        # 1. Set the target rooftop coordinates first
        self.target_pos[..., 0] = 0.5 * (chosen_bldgs[..., 0] + chosen_bldgs[..., 1])
        self.target_pos[..., 1] = 0.5 * (chosen_bldgs[..., 2] + chosen_bldgs[..., 3])
        self.target_pos[..., 2] = chosen_bldgs[..., 4] + 10.0
        self.roof_heights = chosen_bldgs[..., 4]

        # 2. Spawn drones 20m–30m near the target building instead of 240m away at depot
        offset_xy = torch.randn((self.B, self.D, 2), device=self.device) * 20.0
        self.pos[..., :2] = self.target_pos[..., :2] + offset_xy
        self.pos[..., 2] = self.roof_heights + 12.0  # Float 12 meters above roof level
            
        self.vel.zero_()
        self.cable_len.fill_(0.08)
        self.cable_vel.zero_()
        self.has_payload.fill_(True)
        self.battery_soc.fill_(0.98)
        self.obs_history.zero_()
        self.step_counts.zero_()
        
        self.prev_dist = torch.linalg.vector_norm(self.target_pos - self.pos, dim=-1)
        return self._get_stacked_obs()

    def _vectorized_lidar_16(self):
        px = self.pos[..., 0, None, None]
        py = self.pos[..., 1, None, None]
        pz = self.pos[..., 2, None, None]
        
        xmin = self.building_aabbs[:, 0].reshape(1, 1, 1, -1)
        xmax = self.building_aabbs[:, 1].reshape(1, 1, 1, -1)
        ymin = self.building_aabbs[:, 2].reshape(1, 1, 1, -1)
        ymax = self.building_aabbs[:, 3].reshape(1, 1, 1, -1)
        top  = self.building_aabbs[:, 4].reshape(1, 1, 1, -1)
        
        tx1, tx2 = (xmin - px) / self.lidar_rx, (xmax - px) / self.lidar_rx
        ty1, ty2 = (ymin - py) / self.lidar_ry, (ymax - py) / self.lidar_ry
        t_near = torch.maximum(torch.minimum(tx1, tx2), torch.minimum(ty1, ty2))
        t_far  = torch.minimum(torch.maximum(tx1, tx2), torch.maximum(ty1, ty2))
        
        valid = (t_far >= torch.clamp(t_near, min=0.0)) & (pz < top) & (t_far >= 0.0)
        hits = torch.where(valid, torch.clamp(t_near, min=0.1), torch.full_like(t_near, 30.0))
        return torch.clamp(hits.min(dim=-1).values, 0.1, 30.0)

    def _get_stacked_obs(self):
        lidar = self._vectorized_lidar_16()
        current_obs = torch.zeros((self.B, self.D, BASE_OBS_DIM), device=self.device)
        
        current_obs[..., 0:3] = (self.target_pos - self.pos) / 50.0
        current_obs[..., 3:6] = self.pos / 200.0
        current_obs[..., 6:9] = self.vel / 8.0
        current_obs[..., 9] = 1.0
        current_obs[..., 13] = 1.0
        current_obs[..., 17] = 1.0
        current_obs[..., 21:37] = lidar / 30.0
        
        sonar = torch.clamp(self.pos[..., 2], min=0.05, max=15.0)
        current_obs[..., 37] = sonar / 15.0
        v_air = self.vel - self.wind[:, None, :]
        v_tas = torch.linalg.vector_norm(v_air, dim=-1)
        current_obs[..., 38] = v_tas / 20.0
        current_obs[..., 39] = (0.5 * 1.225 * v_tas.square()) / 100.0
        current_obs[..., 42:45] = self.wind[:, None, :].expand(-1, self.D, -1) / 8.0
        
        conical = torch.clamp(sonar[..., None].expand(-1, -1, 8) / 0.9063, 0.05, 15.0)
        current_obs[..., 45:53] = conical / 15.0
        current_obs[..., 53] = self.cable_len / 15.0
        current_obs[..., 54] = self.cable_vel / 5.0
        current_obs[..., 55] = torch.where(self.has_payload, torch.full_like(self.cable_len, PAYLOAD_WEIGHT_N / 50.0), torch.full_like(self.cable_len, 1.2 / 50.0))
        current_obs[..., 58] = self.has_payload.float()
        current_obs[..., 61] = self.battery_soc
        current_obs[..., 104:106] = self.vel[..., :2] / 5.0
        current_obs[..., 108:111] = (self.target_pos - self.pos) / 10.0
        current_obs[..., 111] = (torch.linalg.vector_norm(self.target_pos - self.pos, dim=-1) < 15.0).float()
        current_obs[..., 130:133] = self.accel_prev / 30.0
        
        # Zero-loop peer kinematics
        p_peer0 = self.pos[:, self.peer0_idx]
        v_peer0 = self.vel[:, self.peer0_idx]
        p_peer1 = self.pos[:, self.peer1_idx]
        v_peer1 = self.vel[:, self.peer1_idx]
        
        current_obs[..., 118:121] = (p_peer0 - self.pos) / 50.0
        current_obs[..., 121:124] = (v_peer0 - self.vel) / 10.0
        current_obs[..., 124:127] = (p_peer1 - self.pos) / 50.0
        current_obs[..., 127:130] = (v_peer1 - self.vel) / 10.0
                
        self.obs_history = torch.cat((self.obs_history[:, :, 1:, :], current_obs.unsqueeze(2)), dim=2)
        return self.obs_history.reshape(self.B, self.D, OBSERVATION_DIM)

    def step(self, actions_continuous_accel, actions_continuous_winch):
        """
        Advances the unconstrained sensorimotor dynamics.
        actions_continuous_accel: continuous 3D acceleration [ax, ay, az]
        actions_continuous_winch: continuous reel rate [-1.0, 1.0]
        """
        m_eff = self.dry_mass + self.cargo_mass * self.has_payload.float()
        a_cmd = torch.clamp(actions_continuous_accel, -6.0, 6.0)
        
        v_air = self.vel - self.wind[:, None, :]
        v_tas = torch.linalg.vector_norm(v_air, dim=-1, keepdim=True).clamp(min=0.1)
        f_drag = -0.5 * 1.225 * self.cda * v_tas * v_air
        
        # 1. Rotors cancel gravity; net acceleration is a_cmd plus aerodynamic drag
        a_net = a_cmd + (f_drag / m_eff[..., None])
        self.accel_prev.copy_(a_net)
        
        # 2. Integrate velocity and position
        self.vel.add_(a_net * self.dt).clamp_(-15.0, 15.0)
        self.pos.add_(self.vel * self.dt)
        
        # 3. Ground plane: clamp height to 0.06m and only zero DOWNWARD velocity
        ground_mask = self.pos[..., 2] <= 0.06
        self.pos[..., 2].clamp_(min=0.06)
        self.vel[..., 2] = torch.where(ground_mask & (self.vel[..., 2] < 0.0), torch.zeros_like(self.vel[..., 2]), self.vel[..., 2])
        self.vel[..., :2] = torch.where(ground_mask.unsqueeze(-1), self.vel[..., :2] * 0.85, self.vel[..., :2])
        
        # Continuous Winch Dynamics (Controlled purely by policy current)
        target_winch_vel = torch.clamp(actions_continuous_winch, -1.0, 1.0) * MAX_WINCH_SPEED_MS
        target_winch_vel = torch.where(~self.has_payload, torch.full_like(target_winch_vel, -2.8), target_winch_vel)
        self.cable_vel.copy_(target_winch_vel)
        self.cable_len = torch.clamp(self.cable_len + self.cable_vel * self.dt, min=0.08, max=14.0)
        
        # Continuous Parabolic / Touch-and-Go Payload Release
        cargo_z = self.pos[..., 2] - self.cable_len
        dist_to_target_xy = torch.linalg.vector_norm(self.pos[..., :2] - self.target_pos[..., :2], dim=-1)
        touchdown_envelope = (dist_to_target_xy < 2.2) & (cargo_z <= self.roof_heights + 0.30) & self.has_payload
        
        self.touchdown_timer = torch.where(touchdown_envelope, self.touchdown_timer + self.dt, torch.zeros_like(self.touchdown_timer))
        released = touchdown_envelope & (self.touchdown_timer >= 0.18)
        self.has_payload = torch.where(released, torch.zeros_like(self.has_payload), self.has_payload)
        
        # Holistic Potential-Based Reward Function
        dist_to_wp = torch.linalg.vector_norm(self.target_pos - self.pos, dim=-1)
        speed = torch.linalg.vector_norm(self.vel, dim=-1)
        
        r_progress = 4.0 * (self.prev_dist - dist_to_wp)
        self.prev_dist = dist_to_wp.clone()
        
        r_dist = r_progress - 0.02 * dist_to_wp
        r_hover = torch.where(dist_to_wp < 1.8, 0.45 / (1.0 + 1.5 * speed), torch.zeros_like(dist_to_wp))
        r_delivery = torch.where(released, torch.full_like(dist_to_wp, 85.0), torch.zeros_like(dist_to_wp))
        
        lidar = self._vectorized_lidar_16()
        min_lidar = lidar.min(dim=-1).values
        r_collision = torch.where(min_lidar < 2.0, -5.0 * (2.0 - min_lidar).square(), torch.zeros_like(min_lidar))
        r_effort = -0.01 * torch.linalg.vector_norm(a_cmd, dim=-1)
        
        rewards = r_dist + r_hover + r_delivery + r_collision + r_effort
        
        self.battery_soc = torch.clamp(self.battery_soc - 0.00015 * (1.0 + 0.1 * speed), min=0.0)

        # Count 1 step; if the drone flies for 350 steps (~5.5 seconds) without delivering, mark as timed out
        self.step_counts += 1
        timeout = self.step_counts >= 350

        # Mark done if battery dies, drone goes too high, crashes, delivers, OR times out:
        dones = (self.battery_soc <= 0.05) | (self.pos[..., 2] > CEILING_HEIGHT_M) | (min_lidar < 0.40) | released | timeout
        
        if dones.any():
            reset_mask = dones.unsqueeze(-1)

            # Pick a new random building for the drones that finished
            new_roof_idx = torch.randint(0, len(self.building_aabbs), (self.B, self.D), device=self.device)
            new_bldgs = self.building_aabbs[new_roof_idx]
            new_tx = 0.5 * (new_bldgs[..., 0] + new_bldgs[..., 1])
            new_ty = 0.5 * (new_bldgs[..., 2] + new_bldgs[..., 3])
            new_tz = new_bldgs[..., 4] + 10.0

            # Update targets for the reset drones only
            self.target_pos[..., 0] = torch.where(dones, new_tx, self.target_pos[..., 0])
            self.target_pos[..., 1] = torch.where(dones, new_ty, self.target_pos[..., 1])
            self.target_pos[..., 2] = torch.where(dones, new_tz, self.target_pos[..., 2])
            self.roof_heights = torch.where(dones, new_bldgs[..., 4], self.roof_heights)

            # Respawn close to the new target (20-30m away, 12m above roof)
            offset_xy = torch.randn((self.B, self.D, 2), device=self.device) * 20.0
            spawn_pos = torch.stack([
                self.target_pos[..., 0] + offset_xy[..., 0],
                self.target_pos[..., 1] + offset_xy[..., 1],
                self.roof_heights + 12.0
            ], dim=-1)

            self.pos = torch.where(reset_mask, spawn_pos, self.pos)
            self.vel = torch.where(reset_mask, torch.zeros_like(self.vel), self.vel)
            self.battery_soc = torch.where(dones, torch.full_like(self.battery_soc, 0.98), self.battery_soc)
            self.has_payload = torch.where(dones, torch.ones_like(self.has_payload), self.has_payload)
            self.cable_len = torch.where(dones, torch.full_like(self.cable_len, 0.08), self.cable_len)

            new_dist = torch.linalg.vector_norm(self.target_pos - self.pos, dim=-1)
            self.prev_dist = torch.where(dones, new_dist, self.prev_dist)

            # Reset the timer back to 0 only for the drones that finished
            self.step_counts = torch.where(dones, torch.zeros_like(self.step_counts), self.step_counts)

        next_obs = self._get_stacked_obs()
        return next_obs, rewards, dones, {"delivered": released}

    # =============================================================================
# 15. SWARM TRANSFORMER & SENSORIMOTOR ACTOR-CRITIC NETWORK
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

class SwarmCrossAttentionBlock(nn.Module):
    """
    Decentralized Swarm Cross-Attention Block.
    Replaces human bid matrices: drones communicate latent intention tokens
    through multi-head self/cross-attention across peer agents.
    """
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm_mlp = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (Batch, Num_Drones, Embed_Dim)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm_mlp(x + self.mlp(x))
        return x

class SwarmTransformerActorCriticNet(nn.Module):
    """
    Post-Human Swarm Transformer Policy (~2.6M Parameters).
    Eliminates rigid FSM states. Simultaneously outputs continuous 3D acceleration
    and continuous cable rate, conditioned on latent cross-drone attention.
    """
    def __init__(self, in_features=OBSERVATION_DIM, num_drones=NUM_SWARM_DRONES):
        super().__init__()
        self.D = num_drones
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.SiLU()
        )
        
        self.res1 = DenseResBlock512(dim=512, hidden_dim=768)
        self.res2 = DenseResBlock512(dim=512, hidden_dim=768)
        self.swarm_attn = SwarmCrossAttentionBlock(embed_dim=512, num_heads=8)
        self.res3 = DenseResBlock512(dim=512, hidden_dim=768)

        # Actor Continuous 3D Guidance Head: [ax, ay, az]
        self.actor_guidance_mu = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
            nn.Tanh()
        )
        self.guidance_logstd = nn.Parameter(torch.full((3,), -0.85))

        # Actor Continuous Winch Rate Head: [w_rate]
        self.actor_winch_mu = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        self.winch_logstd = nn.Parameter(torch.full((1,), -0.70))

        # Categorical Intent Head (Maintained for HUD & Diagnostics)
        self.actor_fsm_logits = nn.Sequential(
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, NUM_EXECUTIVE_STATES)
        )

        # Critic Scalar State-Value V(s)
        self.critic_value = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # Multi-task auxiliary prediction heads
        self.aux_health = nn.Sequential(nn.Linear(512, 64), nn.SiLU(), nn.Linear(64, 8), nn.Sigmoid())
        self.aux_trust  = nn.Sequential(nn.Linear(512, 64), nn.SiLU(), nn.Linear(64, 2), nn.Sigmoid())

    def _forward_backbone(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        is_batched_swarm = (x.ndim == 3)
        
        if not is_batched_swarm:
            # Reshape (B*D, Obs) into (B, D, Obs) if divisible
            total_items = x.shape[0]
            if total_items % self.D == 0:
                B = total_items // self.D
                x_swarm = x.reshape(B, self.D, -1)
                feat = self.input_proj(x_swarm)
                feat = self.res1(feat)
                feat = self.res2(feat)
                feat = self.swarm_attn(feat)
                feat = self.res3(feat)
                return feat.reshape(total_items, -1)
            else:
                feat = self.input_proj(x)
                feat = self.res1(feat)
                feat = self.res2(feat)
                feat = self.res3(feat)
                return feat
        else:
            feat = self.input_proj(x)
            feat = self.res1(feat)
            feat = self.res2(feat)
            feat = self.swarm_attn(feat)
            feat = self.res3(feat)
            return feat

    def forward(self, x):
        feat = self._forward_backbone(x)
        g_mu = self.actor_guidance_mu(feat)
        zeros_pad = torch.zeros((*g_mu.shape[:-1], 1), device=g_mu.device, dtype=g_mu.dtype)
        guidance = torch.cat((g_mu, zeros_pad), dim=-1)
        
        fsm_logits = self.actor_fsm_logits(feat)
        w_mu = self.actor_winch_mu(feat)
        winch_cmd = torch.cat((w_mu, torch.zeros_like(w_mu)), dim=-1)
        
        health_pred = self.aux_health(feat)
        trust_scores = self.aux_trust(feat)
        value = self.critic_value(feat)
        return guidance, fsm_logits, winch_cmd, health_pred, trust_scores, value

    def get_value(self, x):
        feat = self._forward_backbone(x)
        return self.critic_value(feat).squeeze(-1)

    def forward_policy(self, x, deterministic=False):
        feat = self._forward_backbone(x)

        g_mu = self.actor_guidance_mu(feat) * 6.0
        g_std = torch.exp(self.guidance_logstd).expand_as(g_mu)
        dist_g = torch.distributions.Normal(g_mu, g_std)

        w_mu = self.actor_winch_mu(feat)
        w_std = torch.exp(self.winch_logstd).expand_as(w_mu)
        dist_w = torch.distributions.Normal(w_mu, w_std)

        fsm_logits = self.actor_fsm_logits(feat)
        dist_fsm = torch.distributions.Categorical(logits=fsm_logits)

        if deterministic:
            act_g, act_w, act_fsm = g_mu, w_mu.squeeze(-1), fsm_logits.argmax(dim=-1)
        else:
            act_g = dist_g.sample()
            act_w = dist_w.sample().squeeze(-1)
            act_fsm = dist_fsm.sample()

        log_prob = (dist_g.log_prob(act_g).sum(dim=-1) + 
                    dist_w.log_prob(act_w.unsqueeze(-1)).sum(dim=-1) + 
                    dist_fsm.log_prob(act_fsm))

        return act_g, act_w, act_fsm, log_prob

    def evaluate_actions(self, x, act_g, act_w, act_fsm):
        feat = self._forward_backbone(x)

        g_mu = self.actor_guidance_mu(feat) * 6.0
        g_std = torch.exp(self.guidance_logstd).expand_as(g_mu)
        dist_g = torch.distributions.Normal(g_mu, g_std)

        w_mu = self.actor_winch_mu(feat)
        w_std = torch.exp(self.winch_logstd).expand_as(w_mu)
        dist_w = torch.distributions.Normal(w_mu, w_std)

        fsm_logits = self.actor_fsm_logits(feat)
        dist_fsm = torch.distributions.Categorical(logits=fsm_logits)

        log_prob = (dist_g.log_prob(act_g).sum(dim=-1) + 
                    dist_w.log_prob(act_w.unsqueeze(-1)).sum(dim=-1) + 
                    dist_fsm.log_prob(act_fsm))

        entropy = dist_g.entropy().sum(dim=-1) + dist_w.entropy().sum(dim=-1) + dist_fsm.entropy()
        value = self.critic_value(feat).squeeze(-1)

        return log_prob, entropy, value

DeepDenseFlightExecutiveNet = SwarmTransformerActorCriticNet
PPOActorCriticFlightNet = SwarmTransformerActorCriticNet
MultiModalFlightExecutiveNet = SwarmTransformerActorCriticNet
MultiModalPPOActorCriticNet = SwarmTransformerActorCriticNet

# =============================================================================
# 16. HIGH-THROUGHPUT PPO ACTOR-CRITIC OPTIMIZATION ENGINE
# =============================================================================
class PPOTrainer:
    def __init__(self, env: BatchedPyTorchSwarmEnv, policy_net: DeepDenseFlightExecutiveNet, 
                 lr=3.0e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.20):
        self.env = env
        self.net = policy_net
        self.device = env.device
        self.optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-5)
        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        
        self.rollout_steps = 256
        self.ppo_epochs = 20
        self.minibatch_size = 8192

    def train_iteration(self):
        self.net.eval()
        B, D = self.env.B, self.env.D
        total_samples = self.rollout_steps * B * D
        
        b_obs       = torch.zeros((self.rollout_steps, B, D, OBSERVATION_DIM), device=self.device)
        b_act_g     = torch.zeros((self.rollout_steps, B, D, 3), device=self.device)
        b_act_w     = torch.zeros((self.rollout_steps, B, D), device=self.device)
        b_act_fsm   = torch.zeros((self.rollout_steps, B, D), dtype=torch.long, device=self.device)
        b_log_prob  = torch.zeros((self.rollout_steps, B, D), device=self.device)
        b_rewards   = torch.zeros((self.rollout_steps, B, D), device=self.device)
        b_dones     = torch.zeros((self.rollout_steps, B, D), dtype=torch.bool, device=self.device)
        b_values    = torch.zeros((self.rollout_steps, B, D), device=self.device)
        
        ep_deliveries = 0
        ep_crashes = 0
        mean_dist_sum = 0.0
        
        obs = self.env._get_stacked_obs()
        
        with torch.no_grad():
            for t in range(self.rollout_steps):
                flat_obs = obs.reshape(B * D, OBSERVATION_DIM)
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    act_g, act_w, act_fsm, log_p = self.net.forward_policy(flat_obs)
                    val = self.net.get_value(flat_obs)
                
                b_obs[t] = obs
                b_act_g[t] = act_g.reshape(B, D, 3)
                b_act_w[t] = act_w.reshape(B, D)
                b_act_fsm[t] = act_fsm.reshape(B, D)
                b_log_prob[t] = log_p.reshape(B, D)
                b_values[t] = val.reshape(B, D)
                
                next_obs, rewards, dones, info = self.env.step(b_act_g[t], b_act_w[t])
                
                ep_deliveries += int(info.get("delivered", torch.zeros(1)).sum().item())
                cur_min_lidar = self.env._vectorized_lidar_16().min(dim=-1).values
                ep_crashes += int((cur_min_lidar < 0.40).sum().item())
                mean_dist_sum += float(torch.linalg.vector_norm(self.env.target_pos - self.env.pos, dim=-1).mean().item())
                
                b_rewards[t] = rewards
                b_dones[t] = dones
                obs = next_obs
                
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                last_val = self.net.get_value(obs.reshape(B * D, OBSERVATION_DIM)).reshape(B, D)

        b_advantages = torch.zeros_like(b_rewards)
        last_gae = 0.0
        for t in reversed(range(self.rollout_steps)):
            next_non_terminal = (~b_dones[t]).float()
            next_v = last_val if t == self.rollout_steps - 1 else b_values[t + 1]
            delta = b_rewards[t] + self.gamma * next_v * next_non_terminal - b_values[t]
            b_advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            
        b_returns = b_advantages + b_values

        f_obs       = b_obs.reshape(total_samples, OBSERVATION_DIM)
        f_act_g     = b_act_g.reshape(total_samples, 3)
        f_act_w     = b_act_w.reshape(total_samples)
        f_act_fsm   = b_act_fsm.reshape(total_samples)
        f_log_prob  = b_log_prob.reshape(total_samples)
        f_advantages= b_advantages.reshape(total_samples)
        f_returns   = b_returns.reshape(total_samples)
        
        y_var = torch.var(f_returns)
        explained_var = (1.0 - (torch.var(f_returns - b_values.reshape(-1)) / (y_var + 1e-8))).item()
        f_advantages = (f_advantages - f_advantages.mean()) / (f_advantages.std() + 1e-8)

        self.net.train()
        total_p_loss, total_v_loss = 0.0, 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        
        for epoch in range(self.ppo_epochs):
            perm = torch.randperm(total_samples, device=self.device)
            for start in range(0, total_samples, self.minibatch_size):
                idx = perm[start : start + self.minibatch_size]
                
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    cur_log_prob, cur_entropy, cur_val = self.net.evaluate_actions(
                        f_obs[idx], f_act_g[idx], f_act_w[idx], f_act_fsm[idx]
                    )
                    
                    log_ratio = cur_log_prob - f_log_prob[idx]
                    ratio = torch.exp(log_ratio)
                    
                    with torch.no_grad():
                        total_kl += ((ratio - 1.0) - log_ratio).mean().item()
                        total_clip_frac += ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
                    
                    surr1 = ratio * f_advantages[idx]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * f_advantages[idx]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    value_loss = 0.5 * F.mse_loss(cur_val, f_returns[idx])
                    entropy_loss = -0.01 * cur_entropy.mean()
                    loss = policy_loss + 0.5 * value_loss + entropy_loss

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.8)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.8)
                    self.optimizer.step()
                
                total_p_loss += policy_loss.item()
                total_v_loss += value_loss.item()
                total_entropy += cur_entropy.mean().item()
                
        num_batches = (total_samples // self.minibatch_size) * self.ppo_epochs
        
        metrics = {
            "p_loss": total_p_loss / max(1, num_batches),
            "v_loss": total_v_loss / max(1, num_batches),
            "entropy": total_entropy / max(1, num_batches),
            "approx_kl": total_kl / max(1, num_batches),
            "clip_frac": total_clip_frac / max(1, num_batches),
            "explained_var": explained_var,
            "mean_reward": b_rewards.mean().item(),
            "deliveries": ep_deliveries,
            "crashes": ep_crashes,
            "mean_dist": mean_dist_sum / self.rollout_steps
        }
        return metrics

def train_ppo_swarm_policy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = int(os.environ.get("SURVIVAL_PPO_ENVS", "4096"))
    
    print("\n" + "="*75)
    print("STARTING POST-HUMAN PPO SWARM TRANSFORMER TRAINING")
    print(f"Parallel Envs: {num_envs:,} ({num_envs * NUM_SWARM_DRONES:,} Drones) | Device: {device}")
    print("="*75)
    
    vec_env = BatchedPyTorchSwarmEnv(num_envs=num_envs, num_drones=NUM_SWARM_DRONES, device=device)
    ppo_net = SwarmTransformerActorCriticNet(in_features=OBSERVATION_DIM).to(device)
    
    if hasattr(torch, "compile") and device.type == "cuda" and sys.platform != "win32":
        try:
            ppo_net = torch.compile(ppo_net)
            print("[INFO] torch.compile active: Kernel fusion enabled.")
        except Exception:
            pass

    trainer = PPOTrainer(vec_env, ppo_net, lr=3.0e-4)
    t0 = time.time()
    
    for iteration in range(1, 151):
        m = trainer.train_iteration()
        if iteration % 5 == 0 or iteration == 1:
            total_steps = trainer.rollout_steps * vec_env.B * vec_env.D * iteration
            fps = total_steps / (time.time() - t0)
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
            
            print(f"PPO Iter [{iteration:03d}/150] | Rew: {m['mean_reward']:+6.2f} | "
                  f"Deliveries: {m['deliveries']:4d} | Crashes: {m['crashes']:4d} | "
                  f"Dist to Goal: {m['mean_dist']:5.1f}m")
            print(f"  > Swarm Policy Health: KL: {m['approx_kl']:.4f} | Clip: {m['clip_frac']*100:4.1f}% | "
                  f"Entropy: {m['entropy']:.2f} | ExplVar: {m['explained_var']:+5.2f} | "
                  f"{fps:,.0f} steps/s | Peak VRAM: {peak_vram:.2f} GB")
                  
    torch.save(ppo_net.state_dict(), MODEL_WEIGHTS_FILE)
    print(f"\n[PPO SUCCESS] Swarm Policy saved to {MODEL_WEIGHTS_FILE} in {time.time()-t0:.2f}s!\n")
    return ppo_net, device

harvest_and_train_metropolitan = train_ppo_swarm_policy

# =============================================================================
# 17. EMERGENT SWARM DISPATCHER & SWARM AGENT RUNTIME
# =============================================================================
class EmergentSwarmDispatcher:
    """
    Decentralized Potential-Based Swarm Dispatcher.
    Replaces rigid CBBA bid-lock negotiation with spatial-energy potential matching.
    """
    def __init__(self, manifest_addresses, depot_helipads, num_drones=NUM_SWARM_DRONES):
        self.manifest = list(manifest_addresses)
        self.depot_helipads = [np.asarray(pad, dtype=np.float32) for pad in depot_helipads]
        self.num_drones = num_drones
        self.tasks = []
        
        for idx, item in enumerate(self.manifest):
            self.tasks.append({
                "task_id": idx,
                "bldg_id": item["bldg_id"],
                "addr": item["addr"],
                "pos": np.asarray(item["pos"], dtype=np.float32),
                "roof_h": float(item["height"]),
                "tag_id": item["tag_id"],
                "priority": 1.0,
                "completed": False
            })
            
        self.total_completed = 0
        self.total_failed = 0

    def inject_custom_waypoint(self, pos_xyz, priority=5.0, bldg_id="CUSTOM_DROP", addr="Emergency Retarget"):
        new_id = len(self.tasks)
        task_dict = {
            "task_id": new_id,
            "bldg_id": bldg_id,
            "addr": addr,
            "pos": np.asarray(pos_xyz, dtype=np.float32),
            "roof_h": float(pos_xyz[2]),
            "tag_id": 999 + new_id,
            "priority": float(priority),
            "completed": False
        }
        self.tasks.append(task_dict)
        print(f"\n>>> [EMERGENT INJECTION] High Priority Potential Well @ {pos_xyz} <<<")
        return task_dict

    def dispatch_next_mission(self, drone_id, drone_soc=0.98, drone_pos=None, microclimate_map=None, sim_time=0.0):
        home_pad = self.depot_helipads[drone_id % len(self.depot_helipads)]
        current_p = home_pad if drone_pos is None else drone_pos

        # Rank uncompleted tasks by energy-distance potential
        available = [t for t in self.tasks if not t["completed"]]
        if not available:
            # Reopen manifest cyclically
            for t in self.tasks:
                t["completed"] = False
            available = self.tasks

        best_task = None
        best_potential = -1e9
        for t in available:
            dist = float(np.linalg.norm(current_p[:2] - t["pos"][:2]))
            potential = (t["priority"] * 100.0) / (1.0 + 0.05 * dist) * max(0.2, drone_soc)
            if potential > best_potential:
                best_potential = potential
                best_task = t

        assigned_alt = 42.0 + (drone_id * 3.0)
        mission = {
            "order_idx": best_task["task_id"],
            "target_pos": best_task["pos"].copy(),
            "bldg_id": best_task["bldg_id"],
            "addr": best_task["addr"],
            "tag_id": best_task["tag_id"],
            "home_pad": home_pad.copy(),
            "roof_height": float(best_task["roof_h"]),
            "assigned_cruise_alt": assigned_alt,
            "drone_id": drone_id
        }
        return mission

    def report_completed(self, drone_id, task_id=None):
        self.total_completed += 1
        if task_id is not None and task_id < len(self.tasks):
            self.tasks[task_id]["completed"] = True

    def report_failed(self, drone_id):
        self.total_failed += 1

CBBADispatcher = EmergentSwarmDispatcher
SwarmMissionDispatcher = EmergentSwarmDispatcher

class SwarmDroneInstance:
    """
    Physical Runtime Agent Instance for the 60 FPS Interactive Cockpit.
    """
    def __init__(self, drone_id, init_pos, world, dispatcher):
        self.drone_id = drone_id
        self.world = world
        self.dispatcher = dispatcher
        self.airframe = BEMTCoaxialAirframe8DOF(init_pos)
        self.ekf = ErrorStateEKF15(init_pos)
        self.quantum = MatterWaveColdAtomQuantumSensor()
        self.bms = SubZeroArrheniusBatteryECM()
        self.winch = ContinuousWinchTetherSystem()
        self.icing = MessingerRotorIcingModel()
        self.imu = Airframe6DOFMEMS_IMU()

        self.mixer = RateDampedActuatorMixerWithCBF()
        self.synth = DroneObservationSynthesizer139Dim()
        self.history_q = deque(maxlen=HISTORY_LEN)

        self.mission = self.dispatcher.dispatch_next_mission(
            self.drone_id, self.bms.soc, init_pos, 
            microclimate_map=world.shared_microclimate_map
        )
        self.home_coords = self.mission["home_pad"].copy()
        self.target = self.mission["target_pos"].copy()
        self.assigned_cruise_alt = self.mission["assigned_cruise_alt"]

        self.trail = deque(maxlen=160)
        self.packages_delivered = 0
        self.delivery_attempts = 0
        self.delivery_failures = 0
        self.rth_active = False

    def evaluate_dynamic_rth(self, live_wind):
        dist_home = float(np.linalg.norm(self.airframe.p[:2] - self.home_coords[:2]))
        if dist_home < 4.0:
            return False

        dir_to_home = (self.home_coords[:2] - self.airframe.p[:2]) / max(0.1, dist_home)
        headwind = -float(np.dot(live_wind[:2], dir_to_home))
        v_ground = max(2.5, 7.5 - headwind)
        time_needed_s = (dist_home / v_ground) + 16.0

        power_needed_watts = self.airframe.last_p_watts * 1.15
        energy_needed = time_needed_s * power_needed_watts
        soc_needed = energy_needed / (PACK_CAPACITY_AH * 3600.0 * max(18.0, self.bms.v_pack))
        return self.bms.soc <= float(np.clip(soc_needed + 0.08, 0.15, 0.58))

def compute_vector_guidance(current_p, current_v, target_p, max_speed_xy=9.5, max_climb=2.40, 
                            max_descend=1.20, wind_vector=None):
    delta_p = target_p - current_p
    dist_xy = float(math.hypot(delta_p[0], delta_p[1]))
    dist_z = float(delta_p[2])
    
    if dist_xy > 0.05 and wind_vector is not None:
        track_unit = delta_p[:2] / dist_xy
        headwind = -float(np.dot(wind_vector[:2], track_unit))
        adaptive_cruise_speed = (max_speed_xy + 0.40 * headwind) if headwind > 0 else max(4.5, max_speed_xy + 0.22 * headwind)
    else:
        adaptive_cruise_speed = max_speed_xy

    if dist_xy <= 0.05:
        v_cmd_xy = np.zeros(2, dtype=np.float32)
    else:
        v_xy_mag = min(adaptive_cruise_speed, math.sqrt(2.0 * 2.0 * dist_xy))
        v_cmd_xy = (delta_p[:2] / dist_xy) * v_xy_mag

    if dist_z > 0.0:
        v_cmd_z = min(max_climb, math.sqrt(max(0.0, 2.0 * 1.2 * dist_z)))
    else:
        v_cmd_z = -min(max_descend, math.sqrt(max(0.0, 2.0 * 0.8 * abs(dist_z))))
        
    return target_p.copy(), np.array([v_cmd_xy[0], v_cmd_xy[1], v_cmd_z], dtype=np.float32)

# =============================================================================
# 18. GENESIS 3D MIRROR, PERSPECTIVE FALLBACK & OCCLUSION RESOLVER
# =============================================================================
def _rotation_matrix_to_quaternion_wxyz(R):
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        q = math.sqrt(max(1e-12, 1.0 + tr)) * 2.0
        quat = [0.25*q, (R[2,1]-R[1,2])/q, (R[0,2]-R[2,0])/q, (R[1,0]-R[0,1])/q]
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        q = math.sqrt(max(1e-12, 1.0 + R[0,0]-R[1,1]-R[2,2])) * 2.0
        quat = [(R[2,1]-R[1,2])/q, 0.25*q, (R[0,1]+R[1,0])/q, (R[0,2]+R[2,0])/q]
    elif R[1,1] > R[2,2]:
        q = math.sqrt(max(1e-12, 1.0 + R[1,1]-R[0,0]-R[2,2])) * 2.0
        quat = [(R[0,2]-R[2,0])/q, (R[0,1]+R[1,0])/q, 0.25*q, (R[1,2]+R[2,1])/q]
    else:
        q = math.sqrt(max(1e-12, 1.0 + R[2,2]-R[0,0]-R[1,1])) * 2.0
        quat = [(R[1,0]-R[0,1])/q, (R[0,2]+R[2,0])/q, (R[1,2]+R[2,1])/q, 0.25*q]
    quat = np.asarray(quat, dtype=np.float32)
    return quat / max(1e-8, float(np.linalg.norm(quat)))

def _genesis_pose(entity, pos, quat_wxyz=None):
    p = np.asarray(pos, dtype=np.float32).reshape(3)
    if not np.all(np.isfinite(p)):
        return
    entity.set_pos(tuple(float(v) for v in p))
    if quat_wxyz is not None:
        q = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
        if np.all(np.isfinite(q)):
            entity.set_quat(tuple(float(v) for v in q))

def _generate_v99_visual_urdf(filepath):
    arm = float(ARM_LENGTH_M)
    d = arm / math.sqrt(2.0)
    r = float(ROTOR_RADIUS_M)
    lines = [
        '<?xml version="1.0"?>', '<robot name="v99_coaxial_x8_visual">',
        '  <link name="base_link">',
        f'    <inertial><origin xyz="0 0 0"/><mass value="{DRY_MASS_KG}"/><inertia ixx="0.022" ixy="0" ixz="0" iyy="0.022" iyz="0" izz="0.038"/></inertial>',
        '    <visual><origin xyz="0 0 0"/><geometry><box size="0.22 0.16 0.08"/></geometry><material name="fuselage"><color rgba="0.12 0.15 0.19 1"/></material></visual>',
        '    <collision><origin xyz="0 0 0"/><geometry><box size="0.24 0.18 0.09"/></geometry></collision>',
        '    <visual><origin xyz="0.12 0 0.01"/><geometry><box size="0.06 0.04 0.03"/></geometry><material name="nose"><color rgba="0.95 0.25 0.25 1"/></material></visual>',
        '    <visual><origin xyz="0 0 0.055"/><geometry><cylinder radius="0.045" length="0.03"/></geometry><material name="gps"><color rgba="0 0.85 0.95 1"/></material></visual>',
    ]
    arms = [((d/2,d/2), math.pi/4), ((d/2,-d/2),-math.pi/4), ((-d/2,d/2),-math.pi/4), ((-d/2,-d/2),math.pi/4)]
    for idx, ((x,y), yaw) in enumerate(arms):
        lines.append(f'    <visual><origin xyz="{x} {y} 0.01" rpy="0 0 {yaw}"/><geometry><box size="{arm} 0.024 0.024"/></geometry><material name="arm{idx}"><color rgba="0.22 0.24 0.27 1"/></material></visual>')
    for idx, (x,y) in enumerate([(d,d),(d,-d),(-d,d),(-d,-d)]):
        rgba = '0.95 0.55 0.1 0.82' if idx < 2 else '0 0.8 0.9 0.82'
        lines.append(f'    <visual><origin xyz="{x} {y} 0.035"/><geometry><cylinder radius="{r}" length="0.008"/></geometry><material name="rotor_top_{idx}"><color rgba="{rgba}"/></material></visual>')
        lines.append(f'    <visual><origin xyz="{x} {y} -0.02"/><geometry><cylinder radius="{r}" length="0.008"/></geometry><material name="rotor_bottom_{idx}"><color rgba="0.1 0.1 0.1 0.65"/></material></visual>')
    lines += [
        '    <visual><origin xyz="0 0.08 -0.065" rpy="0 1.5708 0"/><geometry><cylinder radius="0.007" length="0.22"/></geometry><material name="skid1"><color rgba="0.3 0.35 0.4 1"/></material></visual>',
        '    <visual><origin xyz="0 -0.08 -0.065" rpy="0 1.5708 0"/><geometry><cylinder radius="0.007" length="0.22"/></geometry><material name="skid2"><color rgba="0.3 0.35 0.4 1"/></material></visual>',
        '  </link>', '</robot>'
    ]
    Path(filepath).write_text('\n'.join(lines), encoding='utf-8')
    return filepath

def _camera_render_resolution(view_w, view_h):
    try:
        requested_width = int(os.environ.get("SURVIVAL_RENDER_WIDTH", "512"))
    except (TypeError, ValueError):
        requested_width = 512
    width = max(320, min(int(view_w), requested_width))
    height = max(240, int(round(width * int(view_h) / max(1, int(view_w)))))
    return width, height

def _camera_render_fps():
    try:
        requested = float(os.environ.get("SURVIVAL_RENDER_FPS", "60"))
    except (TypeError, ValueError):
        requested = 60.0
    return float(np.clip(requested, 10.0, 60.0))

def build_genesis_3d_mirror(world, drones, traffic_engine, ground_agent, view_w, view_h):
    try:
        import genesis as gs
    except ImportError as exc:
        raise RuntimeError("Genesis not installed; using 3D software projection.") from exc
    try:
        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu)
    except Exception as exc:
        raise RuntimeError(f"Genesis init failed: {exc}") from exc

    sim_options = gs.options.SimOptions(dt=1.0/60.0, substeps=1, gravity=(0.0,0.0,0.0))
    try:
        vis_options = gs.options.VisOptions(
            ambient_light=(0.30, 0.32, 0.36),
            lights=[gs.options.vis.DirectionalLight(
                dir=(-0.45, -0.55, -1.0), color=(1.0, 0.97, 0.90), intensity=4.0
            )],
        )
        scene = gs.Scene(sim_options=sim_options, vis_options=vis_options, show_viewer=False)
    except (AttributeError, TypeError):
        scene = gs.Scene(sim_options=sim_options, show_viewer=False)

    def add_box(size, pos, color, roughness=0.72, friction=0.7, fixed=True):
        return scene.add_entity(morph=gs.morphs.Box(size=tuple(float(x) for x in size), pos=tuple(float(x) for x in pos), fixed=fixed, collision=False),
            material=gs.materials.Rigid(friction=friction),
            surface=gs.surfaces.Rough(color=tuple(float(c) for c in color), roughness=roughness))
    def add_cylinder(radius, height, pos, color, fixed=True):
        return scene.add_entity(morph=gs.morphs.Cylinder(radius=float(radius), height=float(height), pos=tuple(float(x) for x in pos), fixed=fixed, collision=False),
            material=gs.materials.Rigid(friction=0.75), surface=gs.surfaces.Default(color=tuple(float(c) for c in color)))
    def add_sphere(radius, pos, color, fixed=False):
        return scene.add_entity(morph=gs.morphs.Sphere(radius=float(radius), pos=tuple(float(x) for x in pos), fixed=fixed, collision=False),
            material=gs.materials.Rigid(friction=0.4), surface=gs.surfaces.Rough(color=tuple(float(c) for c in color), roughness=0.8))

    scene.add_entity(morph=gs.morphs.Plane(collision=False), material=gs.materials.Rigid(friction=0.8),
        surface=gs.surfaces.Rough(color=(0.16,0.18,0.21), roughness=0.9))
        
    for road in world.roads:
        sx = max(0.1, float(road['x_max']-road['x_min']))
        sy = max(0.1, float(road['y_max']-road['y_min']))
        add_box((sx,sy,0.025), ((road['x_min']+road['x_max'])/2, (road['y_min']+road['y_max'])/2, 0.0125),
                (0.075,0.085,0.10), roughness=0.95, friction=0.9, fixed=True)

    for idx, h in enumerate(world.houses):
        x1,x2,y1,y2,top = [float(v) for v in h['aabb']]
        cx,cy = (x1+x2)/2, (y1+y2)/2
        shade = 0.28 + 0.025*(idx%4)
        add_box((x2-x1,y2-y1,top), (cx,cy,top/2), (shade,shade+0.035,shade+0.09), fixed=True)
        add_cylinder(2.2,0.08,(cx,cy,top+0.04),(0.0,0.72,0.50),fixed=True)
        for c in h.get('clutter',[]):
            cx1,cx2,cy1,cy2,ctop = [float(v) for v in c['aabb']]
            clutter_h = max(0.15, ctop-top)
            add_box((cx2-cx1,cy2-cy1,clutter_h), ((cx1+cx2)/2,(cy1+cy2)/2,top+clutter_h/2), (0.48,0.50,0.54), roughness=0.55, fixed=True)

    palette = [(0.10,0.65,0.95),(0.95,0.55,0.15),(0.75,0.25,0.95)]
    for i,pad in enumerate(world.depot_helipads):
        add_cylinder(2.8,0.08,(float(pad[0]),float(pad[1]),0.04),palette[i%len(palette)],fixed=True)
    for line in getattr(world,'powerlines',[]):
        a,b = np.asarray(line['p1'],dtype=np.float32), np.asarray(line['p2'],dtype=np.float32)
        d = b-a
        add_box((max(0.1,float(np.linalg.norm(d[:2]))),0.08,0.08), tuple((a+b)/2), (0.95,0.20,0.20), roughness=0.4, fixed=True)
    for tree in world.trees:
        tx,ty = float(tree['pos'][0]),float(tree['pos'][1])
        radius,top = float(tree.get('r',3.2)),float(tree.get('h',8.0))
        trunk_h = min(5.0,top*0.7)
        add_cylinder(0.45,trunk_h,(tx,ty,trunk_h/2),(0.40,0.25,0.15),fixed=True)
        add_sphere(radius,(tx,ty,top-radius/2),(0.15,0.50,0.20),fixed=True)

    urdf_path = _generate_v99_visual_urdf(os.path.abspath('drone_coaxial_x8_v99_visual.urdf'))
    drone_entities, cargo_entities, cable_entities = [], [], []
    for d in drones:
        p = np.asarray(d.airframe.p,dtype=np.float32)
        drone_entities.append(scene.add_entity(morph=gs.morphs.URDF(file=urdf_path,pos=tuple(float(v) for v in p),fixed=True,collision=False)))
        cargo_entities.append(add_box((0.22,0.22,0.18),(p[0],p[1],max(0.12,p[2]-max(0.08,d.winch.cable_len))),(0.95,0.75,0.15),roughness=0.5,fixed=True))
        cable_entities.append([add_sphere(0.025,(p[0],p[1],max(0.12,p[2]-0.1)),(0.78,0.82,0.88),fixed=True) for _ in range(max(2,int(d.winch.num_nodes)))])
        
    vehicle_entities=[]
    for v in traffic_engine.vehicles:
        emergency = bool(v.get('is_emergency',False))
        vehicle_entities.append(add_box((3.0,1.35,1.0 if emergency else 0.9),tuple(v['pos']),
            (0.95,0.18,0.18) if emergency else (0.95,0.42,0.12),roughness=0.45,fixed=True))
            
    plane_entity = add_box((4.0,1.7,0.55),tuple(world.civil_aircraft['pos']),(0.95,0.78,0.15),roughness=0.35,fixed=True)
    intruder_pos = np.asarray(ground_agent.p,dtype=np.float32)+np.array([0,0,0.75],dtype=np.float32)
    intruder_entity = add_cylinder(0.24,1.5,tuple(intruder_pos),(0.90,0.26,0.18),fixed=True)
    target_entity = add_sphere(0.55,tuple(drones[0].target),(0.1,0.95,0.62),fixed=True)
    delivered_entities = [add_box((0.30,0.30,0.18),(0,0,-25-i*0.2),(0.15,0.60,0.28),fixed=True) for i in range(120)]
    
    initial_p = np.asarray(drones[0].airframe.p,dtype=np.float32)
    initial_focus = np.asarray(initial_p + np.array([0.0, 0.0, 2.0], dtype=np.float32))
    initial_camera = initial_focus + np.array([34.0, -34.0, 26.0], dtype=np.float32)
    render_w, render_h = _camera_render_resolution(view_w, view_h)
    
    camera_kwargs = dict(
        res=(render_w, render_h), pos=tuple(float(v) for v in initial_camera),
        lookat=tuple(float(v) for v in initial_focus), fov=48, GUI=False,
        near=0.05, far=1600.0,
    )
    try:
        camera = scene.add_camera(**camera_kwargs)
    except TypeError:
        camera_kwargs.pop('near', None); camera_kwargs.pop('far', None)
        camera = scene.add_camera(**camera_kwargs)
        
    scene.build(n_envs=1)
    try:
        scene.reset()
    except Exception:
        pass
        
    hidden_delivery_positions = [np.array([0.0, 0.0, -25.0 - i * 0.2], dtype=np.float32)
                                 for i in range(len(delivered_entities))]
                                 
    mirror = {'scene':scene,'camera':camera,'drone_entities':drone_entities,'cargo_entities':cargo_entities,
        'cable_entities':cable_entities,'vehicle_entities':vehicle_entities,'plane_entity':plane_entity,
        'intruder_entity':intruder_entity,'target_entity':target_entity,'delivered_entities':delivered_entities,
        'delivered_positions':hidden_delivery_positions,
        'camera_pos':initial_camera.copy(), 'camera_lookat':initial_focus.copy(),
        'render_width':render_w, 'render_height':render_h,
        'render_fps':_camera_render_fps(), 'last_view_render_time':0.0,
        'cached_view_surface':None, 'cached_view_key':None, 'render_frame_count':0,
        'render_call_count':0, 'last_render_call':0,
        'segmentation_check_interval':90,
        'last_render_error':0.0, 'last_render_ok':False}
        
    for _ in range(2):
        try:
            camera.render(rgb=True, force_render=True)
        except TypeError:
            try: camera.render(rgb=True)
            except Exception: break
        except Exception:
            break
    return mirror

def _ray_aabb_first_hit(origin, direction, bounds_min, bounds_max):
    t_min,t_max=-float('inf'),float('inf')
    for axis in range(3):
        o,d,lo,hi=float(origin[axis]),float(direction[axis]),float(bounds_min[axis]),float(bounds_max[axis])
        if abs(d)<1e-9:
            if o<lo or o>hi: return None
            continue
        t1,t2=(lo-o)/d,(hi-o)/d
        if t1>t2: t1,t2=t2,t1
        t_min,t_max=max(t_min,t1),min(t_max,t2)
        if t_max<t_min: return None
    if t_max<0: return None
    return t_min if t_min>=0 else t_max

def resolve_genesis_camera_occlusion(camera_pos, camera_lookat, world, clearance=0.9, minimum_distance=0.65):
    target=np.asarray(camera_lookat,dtype=np.float64).reshape(3)
    desired=np.asarray(camera_pos,dtype=np.float64).reshape(3)
    ray=desired-target
    desired_dist=float(np.linalg.norm(ray))
    if not np.isfinite(desired_dist) or desired_dist<1e-6:
        return desired.astype(np.float32)
    direction=ray/desired_dist
    nearest=desired_dist

    for h in world.houses:
        x1,x2,y1,y2,top=[float(v) for v in h['aabb']]
        t=_ray_aabb_first_hit(target,direction,(x1,y1,0.0),(x2,y2,top))
        if t is not None and 1.25<float(t)<nearest:
            nearest=float(t)
        for c in h.get('clutter',[]):
            cx1,cx2,cy1,cy2,ctop=[float(v) for v in c['aabb']]
            t=_ray_aabb_first_hit(target,direction,(cx1,cy1,top),(cx2,cy2,ctop))
            if t is not None and 1.25<float(t)<nearest:
                nearest=float(t)

    for tree in getattr(world,'trees',[]):
        centre=np.array([float(tree['pos'][0]),float(tree['pos'][1]),float(tree.get('h',8.0))-float(tree.get('r',3.2))/2.0])
        radius=float(tree.get('r',3.2))+0.75
        oc=target-centre
        b=float(np.dot(oc,direction))
        c=float(np.dot(oc,oc)-radius*radius)
        disc=b*b-c
        if disc>=0.0:
            t=-b-math.sqrt(disc)
            if 1.25<t<nearest:
                nearest=t

    for line in getattr(world,'powerlines',[]):
        a=np.asarray(line['p1'],dtype=np.float64); b=np.asarray(line['p2'],dtype=np.float64)
        lo=np.minimum(a,b)-np.array([0.5,0.65,0.65])
        hi=np.maximum(a,b)+np.array([0.5,0.65,0.65])
        t=_ray_aabb_first_hit(target,direction,lo,hi)
        if t is not None and 1.25<float(t)<nearest:
            nearest=t

    safe_distance=max(float(minimum_distance),nearest-float(clearance))
    if nearest < desired_dist and safe_distance < desired_dist:
        desired=target+direction*safe_distance
    desired[2]=max(0.85,float(desired[2]))
    return desired.astype(np.float32)

def screen_to_world_raycast_genesis(mouse_pos,camera_pos,camera_lookat,view_w,view_h,world,fov_deg=48.0):
    origin=np.asarray(camera_pos,dtype=np.float64).reshape(3)
    lookat=np.asarray(camera_lookat,dtype=np.float64).reshape(3)
    forward=lookat-origin; forward/=max(1e-9,float(np.linalg.norm(forward)))
    up_world=np.array([0.0,0.0,1.0])
    right=np.cross(forward,up_world)
    if np.linalg.norm(right)<1e-7: right=np.array([1.0,0.0,0.0])
    right/=max(1e-9,float(np.linalg.norm(right)))
    up=np.cross(right,forward); up/=max(1e-9,float(np.linalg.norm(up)))
    mx,my=float(mouse_pos[0]),float(mouse_pos[1])
    aspect=float(view_w)/max(1.0,float(view_h)); th=math.tan(math.radians(float(fov_deg))*0.5)
    x=((2*mx/max(1.0,float(view_w)))-1)*aspect*th
    y=(1-(2*my/max(1.0,float(view_h))))*th
    direction=forward+right*x+up*y; direction/=max(1e-9,float(np.linalg.norm(direction)))
    best_t,best_point=float('inf'),None
    for h in world.houses:
        x1,x2,y1,y2,top=[float(v) for v in h['aabb']]
        t=_ray_aabb_first_hit(origin,direction,(x1,y1,0.0),(x2,y2,top))
        if t is not None and 0<=t<best_t: best_t,best_point=t,origin+t*direction
        for c in h.get('clutter',[]):
            cx1,cx2,cy1,cy2,ctop=[float(v) for v in c['aabb']]
            tc=_ray_aabb_first_hit(origin,direction,(cx1,cy1,top),(cx2,cy2,ctop))
            if tc is not None and 0<=tc<best_t: best_t,best_point=tc,origin+tc*direction
    if direction[2]<-1e-8:
        tg=-origin[2]/direction[2]
        if 0<=tg<best_t: best_t,best_point=tg,origin+tg*direction
    if best_point is None or not np.all(np.isfinite(best_point)):
        return np.asarray(world.depot_helipads[0],dtype=np.float32).copy()
    return np.asarray(best_point,dtype=np.float32)

def compute_v99_camera_pose(world, drone, cam_mode, cam_yaw, cam_pitch, cam_focus, cam_zoom=1.0):
    p = np.asarray(drone.airframe.p, dtype=np.float32).reshape(3)
    target = np.asarray(drone.target, dtype=np.float32).reshape(3)
    zoom = float(np.clip(cam_zoom, 0.35, 2.8))

    if cam_mode == "CITY_OVERVIEW":
        focus = np.array([0.0, 0.0, 14.0], dtype=np.float32)
        distance = 720.0 * zoom
        az = float(cam_yaw) + 0.25
        elevation = float(np.clip(cam_pitch + 0.15, 0.25, 1.10))
        horizontal = distance * math.cos(elevation)
        cam = focus + np.array([horizontal * math.cos(az), horizontal * math.sin(az), distance * math.sin(elevation)], dtype=np.float32)
        return cam, focus

    if cam_mode == "DEPOT_OVERVIEW":
        depot = np.mean(np.asarray(world.depot_helipads, dtype=np.float32), axis=0)
        if world.manifest_addresses:
            nearest = min(world.manifest_addresses,
                          key=lambda item: float(np.linalg.norm(np.asarray(item['pos'])[:2] - depot[:2])))
            nearest_pos = np.asarray(nearest['pos'], dtype=np.float32)
            focus = 0.58 * depot + 0.42 * nearest_pos
        else:
            focus = depot.copy()
        focus[2] = max(7.0, float(focus[2]))
        distance = 178.0 * zoom
        az = float(cam_yaw) + 0.15
        elevation = float(np.clip(cam_pitch + 0.02, 0.22, 1.05))
        horizontal = distance * math.cos(elevation)
        cam = focus + np.array([horizontal * math.cos(az), horizontal * math.sin(az), distance * math.sin(elevation)], dtype=np.float32)
        return cam, focus

    if cam_mode == "BELLY_GIMBAL":
        R = np.asarray(drone.airframe.R, dtype=np.float32).reshape(3, 3)
        cam = p + R @ np.array([-2.4, 0.0, 1.2], dtype=np.float32)
        cargo_z = max(0.18, float(p[2]) - max(0.25, float(drone.winch.cable_len) * 0.58))
        focus = np.array([p[0] + drone.winch.sway_offset[0],
                          p[1] + drone.winch.sway_offset[1], cargo_z], dtype=np.float32)
        cam[2] = max(1.0, float(cam[2]))
        return cam, focus

    delta = target[:2] - p[:2]
    distance_to_target = float(np.linalg.norm(delta))
    if distance_to_target > 1.0:
        forward = np.array([delta[0] / distance_to_target, delta[1] / distance_to_target, 0.0], dtype=np.float32)
    else:
        velocity = np.asarray(drone.airframe.v[:2], dtype=np.float32)
        speed = float(np.linalg.norm(velocity))
        forward = np.array([velocity[0] / speed, velocity[1] / speed, 0.0], dtype=np.float32) if speed > 0.6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

    if cam_mode == "ORBIT":
        focus = p + np.array([0.0, 0.0, 1.1], dtype=np.float32)
        radius = float(np.clip(20.0 * zoom, 7.0, 85.0))
        elevation = float(np.clip(cam_pitch, 0.15, 1.30))
        azimuth = float(cam_yaw)
        cam = focus + np.array([radius * math.cos(elevation) * math.cos(azimuth),
                                radius * math.cos(elevation) * math.sin(azimuth),
                                radius * math.sin(elevation)], dtype=np.float32)
        return cam, focus

    yaw_offset = float(cam_yaw) - 0.75
    c, sn = math.cos(yaw_offset), math.sin(yaw_offset)
    route_forward = np.array([forward[0] * c - forward[1] * sn,
                              forward[0] * sn + forward[1] * c, 0.0], dtype=np.float32)
    right = np.array([-route_forward[1], route_forward[0], 0.0], dtype=np.float32)
    radius = float(np.clip(31.0 * zoom, 12.0, 110.0))
    look_ahead = min(65.0, max(18.0, distance_to_target * 0.38))
    focus = p + route_forward * look_ahead + np.array([0.0, 0.0, 1.5], dtype=np.float32)
    cam = p - route_forward * radius + right * (radius * 0.18)
    cam[2] = float(p[2]) + max(8.0, radius * math.sin(float(np.clip(cam_pitch, 0.20, 1.25))) + 2.5)
    return cam.astype(np.float32), focus.astype(np.float32)

# =============================================================================
# 19. PERSPECTIVE 3D SOFTWARE FALLBACK PROJECTION ENGINE
# =============================================================================
_SOFTWARE_RENDER_FONT_CACHE = {}

def _cached_software_font(name="consolas", size=12, bold=True):
    key = (name, int(size), bool(bold))
    font = _SOFTWARE_RENDER_FONT_CACHE.get(key)
    if font is None:
        font = pygame.font.SysFont(name, int(size), bold=bool(bold))
        _SOFTWARE_RENDER_FONT_CACHE[key] = font
    return font

def _software_render_v99_3d(screen, world, drones, camera_pos, camera_lookat, view_w, view_h,
                             selected_drone_idx, active_target, cam_mode, fov_deg=48.0):
    width, height = int(view_w), int(view_h)
    screen.fill((10, 16, 26), (0, 0, width, height))
    for y in range(height):
        t = y / max(1, height - 1)
        col = (int(17 - 8*t), int(32 - 12*t), int(51 - 15*t))
        pygame.draw.line(screen, col, (0, y), (width, y))

    origin = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    look = np.asarray(camera_lookat, dtype=np.float64).reshape(3)
    forward = look - origin
    forward /= max(1e-9, float(np.linalg.norm(forward)))
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right /= max(1e-9, float(np.linalg.norm(right)))
    up = np.cross(right, forward)
    up /= max(1e-9, float(np.linalg.norm(up)))
    focal = (height * 0.5) / math.tan(math.radians(float(fov_deg)) * 0.5)
    near = 0.25

    def camera_point(pt):
        d = np.asarray(pt, dtype=np.float64).reshape(3) - origin
        return np.array([np.dot(d, right), np.dot(d, up), np.dot(d, forward)], dtype=np.float64)

    def project(pt):
        cp = camera_point(pt)
        if not np.all(np.isfinite(cp)) or cp[2] <= near:
            return None
        x = width * 0.5 + focal * cp[0] / cp[2]
        y = height * 0.5 - focal * cp[1] / cp[2]
        return (int(round(x)), int(round(y)), float(cp[2]))

    draw_items = []
    def project_clipped_polygon(points):
        cps = [camera_point(p) for p in points]
        clipped = []
        for i, current in enumerate(cps):
            nxt = cps[(i + 1) % len(cps)]
            in_current, in_next = current[2] >= near, nxt[2] >= near
            if in_current and in_next:
                clipped.append(nxt)
            elif in_current and not in_next:
                t = (near - current[2]) / (nxt[2] - current[2])
                clipped.append(current + t * (nxt - current))
            elif not in_current and in_next:
                t = (near - current[2]) / (nxt[2] - current[2])
                clipped.append(current + t * (nxt - current))
                clipped.append(nxt)
        if len(clipped) < 3:
            return None
        xy = []
        for cp in clipped:
            x = width * 0.5 + focal * cp[0] / cp[2]
            y = height * 0.5 - focal * cp[1] / cp[2]
            if not math.isfinite(x) or not math.isfinite(y):
                return None
            xy.append((int(round(x)), int(round(y))))
        depth = sum(float(cp[2]) for cp in clipped) / len(clipped)
        return depth, xy

    def add_polygon(points, color, outline=(13, 20, 28), width_px=1):
        projected = project_clipped_polygon(points)
        if projected is None:
            return
        depth, xy = projected
        draw_items.append((depth, xy, color, outline, width_px))

    def draw_background_polygon(points, color):
        projected = project_clipped_polygon(points)
        if projected is not None:
            pygame.draw.polygon(screen, color, projected[1])

    def add_box(x1, x2, y1, y2, z0, z1, color, outline=(18, 24, 32)):
        verts = [np.array(v, dtype=np.float64) for v in [
            (x1,y1,z0),(x2,y1,z0),(x2,y2,z0),(x1,y2,z0),
            (x1,y1,z1),(x2,y1,z1),(x2,y2,z1),(x1,y2,z1)]]
        faces = [
            ([0,1,5,4],0.72),([1,2,6,5],0.86),([2,3,7,6],0.62),([3,0,4,7],0.78),([4,5,6,7],1.22)
        ]
        for inds, factor in faces:
            c = tuple(max(0,min(255,int(v*factor))) for v in color)
            add_polygon([verts[i] for i in inds], c, outline)

    draw_background_polygon([(-240.0,-240.0,-0.02),(240.0,-240.0,-0.02),
                             (240.0,240.0,-0.02),(-240.0,240.0,-0.02)], (39,48,58))

    for road in world.roads:
        draw_background_polygon([
            (float(road['x_min']), float(road['y_min']), 0.025),
            (float(road['x_max']), float(road['y_min']), 0.025),
            (float(road['x_max']), float(road['y_max']), 0.025),
            (float(road['x_min']), float(road['y_max']), 0.025),
        ], (31, 38, 47))

    for idx, pad in enumerate(world.depot_helipads):
        p=np.asarray(pad,dtype=np.float64)
        pts=[(p[0]+2.8*math.cos(a),p[1]+2.8*math.sin(a),0.12) for a in np.linspace(0,2*math.pi,16,endpoint=False)]
        projected=[project(q) for q in pts]
        if all(q is not None for q in projected):
            pygame.draw.polygon(screen,[(30,170,225),(236,134,37),(180,75,225)][idx%3],[q[:2] for q in projected])
            pygame.draw.polygon(screen,(210,238,250),[q[:2] for q in projected],1)

    for tree in world.trees:
        tx,ty = float(tree['pos'][0]),float(tree['pos'][1])
        radius,top = float(tree.get('r',3.2)),float(tree.get('h',8.0))
        trunk_bottom, trunk_top = project((tx,ty,0.1)),project((tx,ty,min(5.0,top*0.7)))
        canopy = project((tx,ty,top-radius/2))
        if trunk_bottom and trunk_top:
            pygame.draw.line(screen,(104,67,40),trunk_bottom[:2],trunk_top[:2],max(2,int(focal*0.45/max(1,trunk_bottom[2]))))
        if canopy:
            rad=int(np.clip(focal*radius/max(1.0,canopy[2]),3,55))
            pygame.draw.circle(screen,(26,112,55),canopy[:2],rad)
            pygame.draw.circle(screen,(69,156,83),canopy[:2],rad,1)

    for line in getattr(world,'powerlines',[]):
        a,b=project(line['p1']),project(line['p2'])
        if a and b:
            pygame.draw.line(screen,(238,74,82),a[:2],b[:2],2)

    for idx, house in enumerate(world.houses):
        x1,x2,y1,y2,top = [float(v) for v in house['aabb']]
        shade = int(88 + (idx % 5) * 7)
        add_box(x1,x2,y1,y2,0.05,top,(shade,shade+8,min(190,shade+28)))
        radius = 2.2
        cx,cy = (x1+x2)/2, (y1+y2)/2
        pad_pts = [(cx+radius*math.cos(a),cy+radius*math.sin(a),top+0.12) for a in np.linspace(0,2*math.pi,16,endpoint=False)]
        add_polygon(pad_pts,(5,190,146),(10,75,66),1)
        for clutter in house.get('clutter',[]):
            ax,bx,ay,by,ctop = [float(v) for v in clutter['aabb']]
            add_box(ax,bx,ay,by,top+0.05,max(top+0.2,ctop),(128,139,151))

    draw_items.sort(key=lambda it: it[0], reverse=True)
    for _, points, color, outline, width_px in draw_items:
        pygame.draw.polygon(screen, color, points)
        if width_px:
            pygame.draw.polygon(screen, outline, points, width_px)

    colors=[(40,235,255),(255,180,40),(232,118,255)]
    for idx, drone in enumerate(drones):
        pp=np.asarray(drone.airframe.p,dtype=np.float64).copy()
        if pp[2] < 0.22:
            pp[2] = 0.32
        q=project(pp)
        if q is None: continue
        x,y,z=q
        rr=int(np.clip(focal*2.1/max(z,1.0),9,18))
        color=colors[idx%len(colors)]
        pygame.draw.circle(screen,(10,18,28),(x,y),rr+4)
        pygame.draw.circle(screen,color,(x,y),rr,2 if idx==selected_drone_idx else 1)
        pygame.draw.line(screen,color,(x-rr-3,y),(x+rr+3,y),2)
        pygame.draw.line(screen,color,(x,y-rr-3),(x,y+rr+3),2)
        pygame.draw.circle(screen,(255,245,190),(x,y),max(2,rr//3))
        
        label=f'D{idx+1}  {float(drone.airframe.p[2]):.1f}m'
        font=_cached_software_font('consolas',12,bold=True)
        textsurf=font.render(label,True,color)
        tag=pygame.Rect(x+rr+5+(idx%2)*3,y-10+idx*23,textsurf.get_width()+8,20)
        pygame.draw.rect(screen,(7,14,23),tag,border_radius=3)
        pygame.draw.rect(screen,color,tag,1,border_radius=3)
        screen.blit(textsurf,(tag.x+4,tag.y+3))

        cable_len = float(getattr(drone.winch, 'cable_len', 0.0))
        if cable_len > 0.35:
            sway = np.asarray(getattr(drone.winch, 'sway_offset', np.zeros(2)), dtype=np.float64)
            parcel = np.array([pp[0]+sway[0], pp[1]+sway[1], max(0.14,float(drone.airframe.p[2])-cable_len)])
            cable_top = project(pp)
            cable_bottom = project(parcel)
            if cable_top and cable_bottom:
                pygame.draw.line(screen,(215,222,232),cable_top[:2],cable_bottom[:2],2)
                size=max(3,min(8,rr//2))
                parcel_rect=pygame.Rect(cable_bottom[0]-size,cable_bottom[1]-size,size*2,size*2)
                pygame.draw.rect(screen,(244,184,38),parcel_rect)
                pygame.draw.rect(screen,(70,48,12),parcel_rect,1)

    target = project(np.asarray(active_target,dtype=np.float64))
    if target:
        x,y,_=target
        pygame.draw.circle(screen,(68,255,160),(x,y),9,2)
        pygame.draw.line(screen,(68,255,160),(x-13,y),(x+13,y),1)
        pygame.draw.line(screen,(68,255,160),(x,y-13),(x,y+13),1)

    label=f'3D SOFTWARE VIEW | {cam_mode} | Genesis frame unavailable'
    font=_cached_software_font('consolas',12,bold=True)
    surf=font.render(label,True,(235,245,255))
    bg=pygame.Rect(10,10,surf.get_width()+12,24)
    pygame.draw.rect(screen,(7,14,23),bg,border_radius=4)
    pygame.draw.rect(screen,(255,181,64),bg,1,border_radius=4)
    screen.blit(surf,(16,16))

def _software_render_v99_3d_scaled(mirror, screen, world, drones, camera_pos, camera_lookat,
                                  view_w, view_h, selected_drone_idx, active_target, cam_mode,
                                  fov_deg=48.0):
    render_w, render_h = _camera_render_resolution(view_w, view_h)
    if render_w >= int(view_w) and render_h >= int(view_h):
        _software_render_v99_3d(screen, world, drones, camera_pos, camera_lookat,
                                 view_w, view_h, selected_drone_idx, active_target,
                                 cam_mode, fov_deg=fov_deg)
        return
    low = mirror.get('software_render_surface')
    if low is None or low.get_size() != (render_w, render_h):
        low = pygame.Surface((render_w, render_h))
        mirror['software_render_surface'] = low
    scaled = mirror.get('software_scaled_surface')
    if scaled is None or scaled.get_size() != (int(view_w), int(view_h)):
        scaled = pygame.Surface((int(view_w), int(view_h)))
        mirror['software_scaled_surface'] = scaled
    _software_render_v99_3d(
        low, world, drones, camera_pos, camera_lookat, render_w, render_h,
        selected_drone_idx, active_target, cam_mode, fov_deg=fov_deg
    )
    pygame.transform.scale(low, (int(view_w), int(view_h)), scaled)
    screen.blit(scaled, (0, 0))

def _render_genesis_mirror_impl(mirror,world,drones,traffic_engine,agent_pos,delivered_boxes,selected_drone_idx,
                          cam_mode,cam_yaw,cam_pitch,cam_focus,active_target,screen,view_w,view_h,font_small):
    d_active = drones[selected_drone_idx]
    p = np.asarray(d_active.airframe.p, dtype=np.float32)
    zoom = float(mirror.get('cam_zoom', 1.0))
    cam_pos, cam_lookat = compute_v99_camera_pose(world, d_active, cam_mode, cam_yaw, cam_pitch, cam_focus, zoom)
    cam_pos = np.asarray(cam_pos, dtype=np.float32)
    cam_lookat = np.asarray(cam_lookat, dtype=np.float32)
    if not np.all(np.isfinite(cam_pos)) or not np.all(np.isfinite(cam_lookat)):
        cam_lookat = p + np.array([0.0, 0.0, 1.5], dtype=np.float32)
        cam_pos = p + np.array([24.0, -24.0, 15.0], dtype=np.float32)
    cam_pos[2] = max(1.0, float(cam_pos[2]))
    if cam_mode not in ("CITY_OVERVIEW", "DEPOT_OVERVIEW"):
        cam_pos=resolve_genesis_camera_occlusion(cam_pos,cam_lookat,world)
    mirror['camera_pos'],mirror['camera_lookat']=cam_pos.copy(),cam_lookat.copy()
    if mirror.get('software_only', False):
        _software_render_v99_3d_scaled(mirror, screen, world, drones, cam_pos, cam_lookat,
                                        view_w, view_h, selected_drone_idx, active_target,
                                        cam_mode, fov_deg=48.0)
        return False
    mirror['camera'].set_pose(pos=tuple(float(x) for x in cam_pos),lookat=tuple(float(x) for x in cam_lookat))

    for d,e,cargo_e,cables in zip(drones,mirror['drone_entities'],mirror['cargo_entities'],mirror['cable_entities']):
        proxy_p = np.asarray(d.airframe.p, dtype=np.float32).copy()
        if proxy_p[2] < 0.22:
            proxy_p[2] = 0.32
        _genesis_pose(e, proxy_p, _rotation_matrix_to_quaternion_wxyz(d.airframe.R))
        cp=np.array([d.airframe.p[0]+d.winch.sway_offset[0],d.airframe.p[1]+d.winch.sway_offset[1],max(0.12,d.airframe.p[2]-max(0.08,d.winch.cable_len))],dtype=np.float32)
        _genesis_pose(cargo_e,cp)
        nodes=np.asarray(d.winch.nodes_pos,dtype=np.float32)
        for j,node_e in enumerate(cables):
            if nodes.ndim==2 and nodes.shape[1]==3 and len(nodes):
                ix=min(len(nodes)-1,round(j*(len(nodes)-1)/max(1,len(cables)-1)))
                node=nodes[ix]
            else:
                node=d.airframe.p+np.array([0,0,-0.08-d.winch.cable_len*(j+1)/len(cables)],dtype=np.float32)
            if not np.all(np.isfinite(node)) or np.linalg.norm(node)<1e-6:
                node=d.airframe.p+np.array([0,0,-0.08-d.winch.cable_len*(j+1)/len(cables)],dtype=np.float32)
            if node[2] < 0.12: node[2] = 0.12
            _genesis_pose(node_e,node)
            
    for v,e in zip(traffic_engine.vehicles,mirror['vehicle_entities']):
        vel=np.asarray(v['vel'],dtype=np.float32); yaw=math.atan2(float(vel[1]),float(vel[0])) if np.linalg.norm(vel[:2])>1e-4 else 0.0
        _genesis_pose(e,v['pos'],np.array([math.cos(yaw/2),0,0,math.sin(yaw/2)],dtype=np.float32))
        
    plane=world.civil_aircraft; yaw=math.atan2(float(plane['vel'][1]),float(plane['vel'][0]))
    _genesis_pose(mirror['plane_entity'],plane['pos'],np.array([math.cos(yaw/2),0,0,math.sin(yaw/2)],dtype=np.float32))
    ip=np.asarray(agent_pos,dtype=np.float32).copy(); ip[2]+=0.75
    _genesis_pose(mirror['intruder_entity'],ip)
    _genesis_pose(mirror['target_entity'],active_target)
    delivered=list(delivered_boxes)
    delivered_positions = mirror.get('delivered_positions')
    if delivered_positions is None:
        delivered_positions = [np.array([0.0,0.0,-25.0-i*0.2],dtype=np.float32)
                               for i in range(len(mirror['delivered_entities']))]
        mirror['delivered_positions'] = delivered_positions
    for i in range(min(len(delivered), len(mirror['delivered_entities']))):
        new_pos=np.asarray(delivered[i],dtype=np.float32).reshape(3)
        cached_pos=delivered_positions[i]
        if cached_pos is None or float(np.linalg.norm(new_pos-np.asarray(cached_pos,dtype=np.float32))) > 0.01:
            _genesis_pose(mirror['delivered_entities'][i],new_pos)
            delivered_positions[i]=new_pos.copy()
            
    try:
        render_count = int(mirror.get('render_frame_count', 0))
        check_interval = max(1, int(mirror.get('segmentation_check_interval', 90)))
        check_segmentation = (render_count % check_interval) == 0
        if check_segmentation:
            try:
                rendered=mirror['camera'].render(rgb=True,segmentation=True,force_render=True)
            except TypeError:
                try:
                    rendered=mirror['camera'].render(rgb=True,segmentation=True)
                except TypeError:
                    rendered=mirror['camera'].render(rgb=True,force_render=True)
        else:
            try:
                rendered=mirror['camera'].render(rgb=True,force_render=True)
            except TypeError:
                rendered=mirror['camera'].render(rgb=True)
        rgb = rendered[0] if isinstance(rendered, (tuple, list)) else rendered
        segmentation = (rendered[2] if check_segmentation and isinstance(rendered, (tuple, list)) and len(rendered) > 2 else None)
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        rgb = np.asarray(rgb)
        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.shape[0] < 2 or rgb.shape[1] < 2:
            raise RuntimeError(f'Unexpected Genesis RGB array shape: {rgb.shape}')
        rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            sample_max = float(np.nanmax(rgb[::16, ::16, :]))
            rgb = np.clip(rgb * 255.0 if sample_max <= 1.0 else rgb, 0, 255).astype(np.uint8)
        diagnostic = rgb[::8, ::8, :]
        pixel_std = float(np.std(diagnostic))
        diagnostic_f = diagnostic.astype(np.float32)
        chroma = float(np.mean(np.max(diagnostic_f, axis=2) - np.min(diagnostic_f, axis=2) > 12.0))
        if not np.isfinite(pixel_std) or (pixel_std < 6.0 and chroma < 0.01):
            raise RuntimeError(f'Genesis returned a nearly blank frame (std={pixel_std:.2f}, chroma={chroma:.4f})')
        if segmentation is not None:
            if isinstance(segmentation, torch.Tensor):
                segmentation = segmentation.detach().cpu().numpy()
            seg = np.asarray(segmentation)
            if seg.ndim == 4:
                seg = seg[0]
            if seg.ndim > 2:
                seg = np.squeeze(seg)
            if seg.ndim == 2 and seg.size:
                labels = np.unique(seg)
                visible_label_count = int(labels.size)
                if visible_label_count < 4:
                    raise RuntimeError(
                        f'Genesis frame has only {visible_label_count} visible segmentation labels; '
                        'switching to the software 3D renderer'
                    )
        surf = pygame.image.frombuffer(rgb.tobytes(), (int(rgb.shape[1]), int(rgb.shape[0])), 'RGB')
        if surf.get_size() != (int(view_w), int(view_h)):
            surf = pygame.transform.smoothscale(surf, (int(view_w), int(view_h)))
        screen.blit(surf, (0, 0))
        mirror['last_render_ok'] = True
        banner = f'GENESIS 3D | {cam_mode} | D{selected_drone_idx+1} | ALT {p[2]:.1f} m'
        bs = font_small.render(banner, True, (240,248,255))
        pygame.draw.rect(screen, (10,16,24), (10,10,bs.get_width()+12,24), border_radius=4)
        pygame.draw.rect(screen, (0,220,255), (10,10,bs.get_width()+12,24), 1, border_radius=4)
        screen.blit(bs, (16,16))
        return True
    except Exception as exc:
        now = time.monotonic()
        if now - float(mirror.get('last_render_error', 0.0)) >= 3.0:
            print(f'[GENESIS RENDER WARNING] {type(exc).__name__}: {exc}; using perspective software renderer')
            mirror['last_render_error'] = now
        mirror['last_render_ok'] = False
        _software_render_v99_3d_scaled(mirror, screen, world, drones, cam_pos, cam_lookat,
                                        view_w, view_h, selected_drone_idx, active_target,
                                        cam_mode, fov_deg=48.0)
        return False

def render_genesis_mirror(mirror, world, drones, traffic_engine, agent_pos, delivered_boxes, selected_drone_idx,
                          cam_mode, cam_yaw, cam_pitch, cam_focus, active_target, screen, view_w, view_h, font_small):
    now = time.monotonic()
    cache_key = (str(cam_mode), int(selected_drone_idx), bool(mirror.get('software_only', False)))
    cached_surface = mirror.get('cached_view_surface')
    fps = float(mirror.get('render_fps', _camera_render_fps()))
    render_stride = max(1, int(round((1.0 / 0.016) / max(1.0, fps))))
    call_count = int(mirror.get('render_call_count', 0)) + 1
    mirror['render_call_count'] = call_count
    key_changed = cache_key != mirror.get('cached_view_key')
    last_render_call = int(mirror.get('last_render_call', 0))

    if cached_surface is not None and not key_changed and (call_count - last_render_call) < render_stride:
        screen.blit(cached_surface, (0, 0))
        return bool(mirror.get('last_render_ok', False))

    rendered_ok = False
    try:
        rendered_ok = _render_genesis_mirror_impl(
            mirror, world, drones, traffic_engine, agent_pos, delivered_boxes, selected_drone_idx,
            cam_mode, cam_yaw, cam_pitch, cam_focus, active_target, screen, view_w, view_h, font_small
        )
    except Exception as exc:
        if now - float(mirror.get('last_render_error', 0.0)) >= 3.0:
            print(f'[3D VIEW WARNING] {type(exc).__name__}: {exc}; showing software-rendered 3D scene instead.')
            mirror['last_render_error'] = now
        mirror['last_render_ok'] = False
        drone = drones[int(np.clip(selected_drone_idx, 0, len(drones)-1))]
        try:
            cam_pos, lookat = compute_v99_camera_pose(
                world, drone, cam_mode, cam_yaw, cam_pitch, cam_focus, mirror.get('cam_zoom', 1.0)
            )
            cam_pos = np.asarray(cam_pos, dtype=np.float32)
            lookat = np.asarray(lookat, dtype=np.float32)
            cam_pos[2] = max(1.0, float(cam_pos[2]))
            if cam_mode not in ('CITY_OVERVIEW', 'DEPOT_OVERVIEW'):
                cam_pos = resolve_genesis_camera_occlusion(cam_pos, lookat, world)
        except Exception:
            p = np.asarray(drone.airframe.p, dtype=np.float32).reshape(3)
            lookat = p + np.array([0.0, 0.0, 2.0], dtype=np.float32)
            cam_pos = p + np.array([30.0, -30.0, 22.0], dtype=np.float32)
        mirror['camera_pos'], mirror['camera_lookat'] = cam_pos.copy(), lookat.copy()
        _software_render_v99_3d_scaled(
            mirror, screen, world, drones, cam_pos, lookat, view_w, view_h,
            selected_drone_idx, active_target, cam_mode, fov_deg=48.0
        )
        rendered_ok = False

    try:
        mirror['cached_view_surface'] = screen.subsurface((0, 0, int(view_w), int(view_h))).copy()
    except Exception:
        mirror['cached_view_surface'] = None
    mirror['cached_view_key'] = cache_key
    mirror['last_view_render_time'] = time.monotonic()
    mirror['last_render_call'] = call_count
    mirror['render_frame_count'] = int(mirror.get('render_frame_count', 0)) + 1
    mirror['last_render_ok'] = bool(rendered_ok)
    return bool(rendered_ok)

# =============================================================================
# 20. COCKPIT PROJECTION ENGINES, ARTIFICIAL HORIZON & RADAR MINIMAP
# =============================================================================
def draw_pfd_artificial_horizon(screen, rect, roll, pitch, alt, airspeed, vspeed, font):
    rx, ry, rw, rh = rect
    cx, cy = rx + rw // 2, ry + rh // 2
    sub = screen.subsurface(rect)
    sub.fill((16, 22, 32))
    pygame.draw.rect(screen, (0, 220, 255), rect, 1)

    pitch_px = int(math.degrees(pitch) * 1.5)
    for deg in range(-90, 91, 10):
        y_pos = cy - pitch_px + int(deg * 1.5 * math.cos(roll))
        x_off = int(24 * math.cos(roll))
        y_off = int(24 * math.sin(roll))
        if ry <= y_pos <= ry + rh:
            pygame.draw.line(screen, (100, 150, 190), (cx - x_off, y_pos - y_off), (cx + x_off, y_pos + y_off), 1)

    hz_dx = int(rw * 0.5 * math.cos(roll))
    hz_dy = int(rw * 0.5 * math.sin(roll))
    pygame.draw.line(screen, (0, 255, 140), (cx - hz_dx, cy - pitch_px - hz_dy), (cx + hz_dx, cy - pitch_px + hz_dy), 2)
    pygame.draw.circle(screen, (255, 220, 40), (cx, cy), 3)

    txt_alt = font.render(f"{alt:.1f}m", True, (0, 220, 255))
    txt_spd = font.render(f"{airspeed:.1f}m/s", True, (255, 180, 40))
    screen.blit(txt_spd, (rx + 4, ry + 4))
    screen.blit(txt_alt, (rx + rw - txt_alt.get_width() - 4, ry + 4))

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
# 21. MASTER INTERACTIVE COCKPIT LOOP (60 FPS) & MAIN ENTRY POINT
# =============================================================================
def _load_model_checkpoint(model, checkpoint_path, device):
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[CHECKPOINT INFO] Initialized missing RL parameters: {missing}")
    return model

def run_all_physics_simulator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_mode = os.environ.get("SURVIVAL_TRAIN_MODE", "PPO").strip().upper()
    force_retrain = os.environ.get("SURVIVAL_FORCE_RETRAIN", "0").strip().lower() in {"1", "true", "yes"}

    if force_retrain or not os.path.exists(MODEL_WEIGHTS_FILE):
        print(f"[POLICY] Training Swarm Transformer Policy via {train_mode}...", flush=True)
        model, device = train_ppo_swarm_policy()
    else:
        model = SwarmTransformerActorCriticNet(in_features=OBSERVATION_DIM).to(device)
        try:
            _load_model_checkpoint(model, MODEL_WEIGHTS_FILE, device)
            print(f"\nLoaded Swarm Policy model from {MODEL_WEIGHTS_FILE}.", flush=True)
        except Exception as e:
            print(f"[WARNING] Loading failed ({e}); initiating PPO training...", flush=True)
            model, device = train_ppo_swarm_policy()

    model.eval()
    pygame.init()
    VIEW_W, VIEW_H = 760, 740
    SIDEBAR_W = 440
    W, H = VIEW_W + SIDEBAR_W, VIEW_H

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Survival Core V39.0 — Post-Human Sensorimotor Drone Swarm")
    clock = pygame.time.Clock()

    font_large = pygame.font.SysFont("consolas", 17, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 12, bold=True)

    world = MetropolitanCityGridWorld400m()
    weather_engine = DynamicUrbanWeatherEngine()
    traffic_engine = DynamicRoadVehicleTrafficEngine(world.avenue_x, world.street_y)
    sensors_physics = FullPhysicsSensorSuite()
    assurance = SurfaceAndWireAssuranceSuite(world)
    lidar_suite = MacroPerceptionLiDARSuite(world)
    sensor_suite = AdvancedMultiModalSensorSuite(world)
    ground_agent = DynamicGroundIntruderAgent(init_pos=[-80.0, 80.0, 0.0])
    dryden_wind_gens = [DrydenAtmosphericTurbulenceModel(w20_m_s=3.4) for _ in range(NUM_SWARM_DRONES)]

    dispatcher = EmergentSwarmDispatcher(world.manifest_addresses, world.depot_helipads, num_drones=NUM_SWARM_DRONES)

    drones = []
    swarm_colors = [(0, 240, 255), (255, 180, 40), (220, 100, 255)]
    for i in range(NUM_SWARM_DRONES):
        init_pos = world.depot_helipads[i].copy()
        init_pos[2] = 0.06
        d_inst = SwarmDroneInstance(i, init_pos, world, dispatcher)
        drones.append(d_inst)

    try:
        genesis_mirror = build_genesis_3d_mirror(
            world=world, drones=drones, traffic_engine=traffic_engine,
            ground_agent=ground_agent, view_w=VIEW_W, view_h=VIEW_H
        )
    except Exception as exc:
        print(f"[3D VIEW WARNING] Genesis init fallback: {exc}")
        genesis_mirror = {
            'software_only': True, 'cam_zoom': 1.0,
            'camera_pos': np.array([0.0, 0.0, 100.0], dtype=np.float32),
            'camera_lookat': np.array([0.0, 0.0, 0.0], dtype=np.float32),
            'render_fps': _camera_render_fps(), 'last_view_render_time': 0.0,
            'render_width': _camera_render_resolution(VIEW_W, VIEW_H)[0],
            'render_height': _camera_render_resolution(VIEW_W, VIEW_H)[1],
            'cached_view_surface': None, 'cached_view_key': None, 'render_frame_count': 0,
            'render_call_count': 0, 'last_render_call': 0,
            'last_render_error': 0.0, 'last_render_ok': False,
        }

    selected_drone_idx = 0
    cam_yaw, cam_pitch = 0.75, 0.45
    cam_zoom = 1.0
    camera_dragging = False
    camera_last_mouse = None
    genesis_mirror['cam_zoom'] = cam_zoom
    cam_focus = drones[0].airframe.p.copy()
    cam_mode = "DEPOT_OVERVIEW"
    running = True

    gust_timer = 0.0
    active_gust = np.zeros(3, dtype=np.float32)
    inject_spoof = False
    flight_time = 0.0
    delivered_boxes = deque(maxlen=120)

    print("\n================ V39.0 POST-HUMAN SENSORIMOTOR DRONE SWARM ONLINE ================")
    print(" [ARCHITECTURE]       : End-to-End Swarm Cross-Attention Transformer (Zero FSM)")
    print(" [CONTROL ACTION]     : Unconstrained 3D Acceleration + Continuous Winch Velocity")
    print(" [EMERGENT FLOCKING]  : Latent Token Passing over Peer Drones (Replacing Auctions)")
    print(" [INVARIANT SAFETY]   : Passive Control Barrier Shield (No Artificial Tilt Limits)")
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
                elif event.key == pygame.K_c:
                    cam_modes = ["DEPOT_OVERVIEW", "CHASE", "ORBIT", "BELLY_GIMBAL", "CITY_OVERVIEW"]
                    cam_mode = cam_modes[(cam_modes.index(cam_mode) + 1) % len(cam_modes)]
                elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                    cam_zoom = max(0.35, cam_zoom * 0.88)
                    genesis_mirror['cam_zoom'] = cam_zoom
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    cam_zoom = min(2.8, cam_zoom * 1.12)
                    genesis_mirror['cam_zoom'] = cam_zoom
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
                    cur_d.target = cur_d.home_coords.copy()
                    cur_d.mission["target_pos"] = cur_d.home_coords.copy()
                    cur_d.rth_active = True
                    print(f"\n>>> [EMERGENCY OVERRIDE] Drone #{cur_d.drone_id+1} Returning to Base! <<<")
                elif event.key == pygame.K_b:
                    drones[selected_drone_idx].bms.soc = 0.20
                elif event.key == pygame.K_i:
                    d = drones[selected_drone_idx]
                    d.icing.icing_active = not d.icing.icing_active
                elif event.key == pygame.K_j:
                    inject_spoof = not inject_spoof
                elif event.key == pygame.K_g:
                    gust_timer = 2.5
                    ang = random.uniform(0, 2*math.pi)
                    active_gust = np.array([math.cos(ang)*8.5, math.sin(ang)*8.5, 0.0], dtype=np.float32)
                elif event.key == pygame.K_r:
                    for d in drones:
                        d.airframe.motor_health.fill(1.0)
                        d.bms.soc = 0.96
                        d.winch.reload_new_cargo()
                        d.rth_active = False
                    inject_spoof = False

            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    cam_zoom = max(0.35, cam_zoom * (0.88 ** event.y))
                elif event.y < 0:
                    cam_zoom = min(2.8, cam_zoom * (1.12 ** (-event.y)))
                genesis_mirror['cam_zoom'] = cam_zoom
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                camera_dragging = True
                camera_last_mouse = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                camera_dragging = False
                camera_last_mouse = None
            elif event.type == pygame.MOUSEMOTION and camera_dragging and camera_last_mouse is not None:
                mx, my = event.pos
                lx, ly = camera_last_mouse
                cam_yaw += float(mx - lx) * 0.008
                cam_pitch = float(np.clip(cam_pitch + float(ly - my) * 0.006, 0.15, 1.30))
                camera_last_mouse = event.pos
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if mx < VIEW_W:
                    click_world = screen_to_world_raycast_genesis(
                        (mx, my), genesis_mirror["camera_pos"], genesis_mirror["camera_lookat"], 
                        VIEW_W, VIEW_H, world, fov_deg=48.0
                    )
                    safe_pad = assurance.snap_to_nearest_safe_ground(click_world[:2])
                    
                    # Inject goal potential well
                    new_task = dispatcher.inject_custom_waypoint(
                        pos_xyz=safe_pad,
                        priority=5.0,
                        bldg_id=f"VIP_DROP_{random.randint(100, 999)}",
                        addr="Cockpit Click Target"
                    )
                    
                    # Allocate to optimal drone based on energy-distance potential
                    best_drone_idx = min(
                        range(len(drones)),
                        key=lambda i: np.linalg.norm(drones[i].airframe.p[:2] - safe_pad[:2]) / max(0.1, drones[i].bms.soc)
                    )
                    winning_drone = drones[best_drone_idx]
                    winning_drone.target = safe_pad.copy()
                    winning_drone.mission["target_pos"] = safe_pad.copy()
                    winning_drone.mission["roof_height"] = float(safe_pad[2])
                    winning_drone.mission["bldg_id"] = new_task["bldg_id"]
                    winning_drone.mission["addr"] = "VIP Target"
                    winning_drone.winch.reload_new_cargo()
                    print(f">>> [EMERGENT ROUTING] Drone #{best_drone_idx + 1} acquired VIP Target! <<<")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: cam_yaw -= 0.03
        if keys[pygame.K_RIGHT]: cam_yaw += 0.03
        if keys[pygame.K_UP]: cam_pitch = min(1.30, cam_pitch + 0.02)
        if keys[pygame.K_DOWN]: cam_pitch = max(0.15, cam_pitch - 0.02)

        if cam_mode == "CHASE":
            cam_focus += (drones[selected_drone_idx].airframe.p - cam_focus) * 5.0 * dt
        elif cam_mode == "BELLY_GIMBAL":
            cam_focus = drones[selected_drone_idx].airframe.p.copy()
        else:
            cam_focus += (np.array([-150.0, -155.0, 4.0], dtype=np.float32) - cam_focus) * 5.0 * dt

        active_drone = drones[selected_drone_idx]
        microburst = active_gust * (gust_timer / 2.5) if gust_timer > 0.0 else np.zeros(3, dtype=np.float32)
        if gust_timer > 0.0:
            gust_timer = max(0.0, gust_timer - dt)

        drone_winds = []
        drone_ambient_temps = []
        for wind_idx, wind_drone in enumerate(drones):
            local_wind, local_gust_scale, local_ambient_temp, _ = weather_engine.get_weather_parameters(
                wind_drone.airframe.p, world.avenue_x
            )
            local_dryden = dryden_wind_gens[wind_idx].step(
                altitude_m=wind_drone.airframe.p[2],
                airspeed_m_s=float(np.linalg.norm(wind_drone.airframe.v)), dt=dt
            ) * (local_gust_scale / 2.0)
            drone_winds.append((local_wind + local_dryden + microburst).astype(np.float32))
            drone_ambient_temps.append(float(local_ambient_temp))

        active_tip = active_drone.airframe.p + np.array([0, 0, -0.08 - active_drone.winch.cable_len])
        agent_pos, global_agent_breach, _ = ground_agent.step(
            cargo_world_pos=active_tip,
            is_dropping=(active_drone.winch.cable_vel > 0.5),
            dt=dt
        )

        obstacles_cbf = [
            {"pos": np.array([(h["aabb"][0]+h["aabb"][1])/2, (h["aabb"][2]+h["aabb"][3])/2, 0.0]), "radius": 11.5, "height": h["aabb"][4], "id": h["id"]} 
            for h in world.houses
        ]
        obstacles_cbf.extend(traffic_engine.get_cbf_obstacles())
        swarm_states = [{"p": d.airframe.p.copy(), "v": d.airframe.v.copy()} for d in drones]

        selected_diag = {}
        drone_step_data = []
        batch_observations = []

        # ==========================================
        # 1. PER-DRONE SENSORY & OBSERVATION STACK
        # ==========================================
        for d_idx, d in enumerate(drones):
            drone_wind = drone_winds[d_idx]
            estimated_wind, sensed_wind_speed, sensed_wind_from_deg, sensed_wind_quality = sensor_suite.poll_wind_vector(
                d.airframe.v, drone_wind, d.airframe.R, dt=dt, sensor_id=d_idx,
                ground_velocity_est=d.ekf.v
            )
            
            # Dynamic Return-To-Home check
            if not d.rth_active and d.evaluate_dynamic_rth(estimated_wind):
                d.rth_active = True
                d.target = d.home_coords.copy()
                d.mission["target_pos"] = d.home_coords.copy()

            current_surf_z = world.get_surface_height(d.airframe.p[0], d.airframe.p[1])
            s_data = sensors_physics.poll_all_physics_sensors(d.airframe.p, d.airframe.v, drone_wind, surface_z=current_surf_z)

            meas_accel, meas_gyro = d.imu.poll(d.airframe.R, d.airframe.accel_linear, d.airframe.omega, d.airframe.motor_omegas)
            a_quantum_body, phase_shift = d.quantum.measure_phase_shift(meas_accel)
            d.ekf.predict(a_quantum_body, meas_gyro, dt=dt)

            compass_yaw = math.atan2(d.airframe.R[1, 0], d.airframe.R[0, 0]) + float(np.random.normal(0, 0.008))
            d.ekf.update_heading(compass_yaw, r_cov=0.02)

            flow_2d, flow_q, z_slant = sensor_suite.poll_optical_flow(d.airframe.p, d.airframe.v, d.airframe.R, d.airframe.omega, current_surf_z)
            v_flow_body = np.array([(flow_2d[0] + meas_gyro[1]) * z_slant, (flow_2d[1] - meas_gyro[0]) * z_slant, 0.0], dtype=np.float32)
            v_flow_world = quat_to_rot(d.ekf.q) @ v_flow_body
            d.ekf.update_optical_flow_velocity_adaptive(v_flow_world, flow_quality=flow_q, z_slant=z_slant)
            d.ekf.update_baro_altitude_adaptive(s_data["z_baro"], q_dyn=s_data["q_dyn"])
            
            z_gps = d.airframe.p + np.random.normal(0, 0.08, 3).astype(np.float32)
            hdop_val, cn0_val, _, _ = sensor_suite.poll_rtk_gnss_metrics(d.airframe.p, inject_spoof)
            d.ekf.update_gps_position_adaptive(z_gps, hdop=hdop_val, cn0_avg=cn0_val, is_spoofed=inject_spoof)

            bms_soc, v_pack, _, r0_dyn = d.bms.step(d.airframe.last_p_watts, amb_temp=drone_ambient_temps[d_idx], dt=dt)
            R_est = quat_to_rot(d.ekf.q)
            omega_est = meas_gyro - d.ekf.b_g

            weight_force = (DRY_MASS_KG + (CARGO_MASS_KG if d.winch.has_payload else 0.0)) * G_ACCEL
            is_on_surf = (d.airframe.p[2] <= current_surf_z + 0.05)
            d.winch.update_skid_strain_gauges(is_on_surf, weight_force, d.airframe.last_p_watts * 0.05)

            drone_yaw = math.atan2(d.airframe.R[1, 0], d.airframe.R[0, 0])
            lidar_16 = lidar_suite.cast_360_lidar(d.airframe.p, drone_yaw)
            conical_8, wire_clr = lidar_suite.cast_downward_conical(d.airframe.p)
            aruco_3d, aruco_lk = sensor_suite.poll_aruco_fiducial_detector(d.airframe.p, d.airframe.R, d.mission["target_pos"], d.mission["tag_id"])
            siren_a, siren_f, siren_s = sensor_suite.poll_directional_acoustic_array(d.airframe.p, traffic_engine.vehicles, sim_time=flight_time)
            adsb_p, adsb_cs, adsb_d = sensor_suite.poll_adsb_civil_transponder(d.airframe.p, d.airframe.v, world.civil_aircraft)
            hook_a, hook_w = sensor_suite.poll_smart_hook_imu(d.airframe.accel_linear, d.airframe.omega, d.winch.sway_offset, d.winch.cable_len, d.winch.has_payload)
            load_3d = sensor_suite.poll_3axis_load_cell(d.winch.tension_n, d.winch.sway_offset, d.winch.cable_len)
            th_dt, dolp, ndvi = sensor_suite.poll_thermal_and_polarization(d.target[:2])
            i_rms, t_mosfet = sensor_suite.poll_esc_foc_telemetry(d.airframe.motor_omegas, d.airframe.motor_health)
            alpha, beta, rh, dust = sensor_suite.poll_5port_air_data(d.airframe.v, drone_wind, d.airframe.R)

            real_peer_info = []
            for j_idx, other_d in enumerate(drones):
                if j_idx == d_idx: continue
                p_rel = other_d.airframe.p - d.airframe.p
                v_rel = other_d.airframe.v - d.airframe.v
                dist_p = float(np.linalg.norm(p_rel))
                real_peer_info.append({"p_rel": p_rel, "v_rel": v_rel, "dist": dist_p, "has_payload": other_d.winch.has_payload})

            obs_139 = d.synth.synthesize_instantaneous(
                p_est=d.ekf.p, v_est=d.ekf.v, R_est=R_est, omega_est=omega_est,
                target_p=d.target, lidar_16=lidar_16, conical_8=conical_8,
                v_tas=s_data["v_tas"], q_dyn=s_data["q_dyn"], min_sonar=s_data["min_sonar"],
                wind_est=estimated_wind, ice_idx=d.icing.ice_thickness_mm / 3.0,
                cable_len=d.winch.cable_len, cable_vel=d.winch.cable_vel,
                tension_n=d.winch.tension_n, sway_2d=d.winch.sway_offset,
                bms_soc=bms_soc, v_pack=v_pack, bms_temp_c=d.bms.temp_c, r0_dyn=r0_dyn,
                chi2_val=1.2, jerk_discrepancy=float(np.linalg.norm(meas_accel - d.airframe.accel_linear)),
                is_spoofed=inject_spoof, emf_rms=s_data["emf_rms"], wire_dist=wire_clr,
                agent_dist=float(np.linalg.norm(agent_pos[:2] - d.airframe.p[:2])),
                ground_spl_dba=55.0, has_payload=d.winch.has_payload,
                hook_a=hook_a, hook_w=hook_w, load_cell_3d=load_3d,
                thermal_dt=th_dt, dolp=dolp, ndvi=ndvi,
                esc_i_rms=i_rms, esc_t_mosfet=t_mosfet,
                alpha_deg=alpha, beta_deg=beta, rh_pct=rh, dust_pm10=dust,
                hdop=hdop_val, cn0_avg=cn0_val, rf_jam_pwr=-52.0 if inject_spoof else -110.0,
                mag_grad=0.02, airframe_accel=meas_accel, airframe_gyro=meas_gyro,
                peer_swarm_kinematics=real_peer_info, optical_flow_2d=flow_2d,
                flow_quality=flow_q, aruco_offset_3d=aruco_3d, aruco_locked=aruco_lk,
                skid_force_left=d.winch.f_skid_left, skid_force_right=d.winch.f_skid_right,
                is_skid_grounded=d.winch.is_skid_grounded, siren_angle=siren_a,
                siren_freq=siren_f, siren_spl=siren_s, adsb_rel_pos=adsb_p,
                adsb_closing_spd=adsb_cs, dt=dt
            )
            obs_417 = d.synth.stack_history(d.history_q, obs_139)
            batch_observations.append(obs_417)

            drone_step_data.append({
                "R_est": R_est, "omega_est": omega_est,
                "current_surf_z": current_surf_z, "meas_accel": meas_accel,
                "s_data": s_data, "flow_2d": flow_2d, "wind_speed": sensed_wind_speed,
                "wind_from_deg": sensed_wind_from_deg, "wind_quality": sensed_wind_quality,
                "adsb_d": adsb_d, "siren_s": siren_s, "phase_shift": phase_shift,
                "v_pack": v_pack, "bms_soc": bms_soc
            })

        # ==========================================
        # 2. SWARM TRANSFORMER CROSS-ATTENTION INFERENCE
        # ==========================================
        t0_inf = time.time()
        # Shape: (1, Num_Drones, Obs_Dim)
        batch_t = torch.from_numpy(np.stack(batch_observations)).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                b_guidance, b_fsm, b_winch, b_health, b_trust, _ = model(batch_t)
        inf_latency_ms = (time.time() - t0_inf) * 1000.0

        b_guidance_np = (b_guidance.squeeze(0)[..., :3] * 6.0).cpu().numpy()
        b_winch_np = b_winch.squeeze(0).squeeze(-1).cpu().numpy()

        # ==========================================
        # 3. UNCONSTRAINED SENSORIMOTOR STEPPING
        # ==========================================
        for d_idx, d in enumerate(drones):
            drone_wind = drone_winds[d_idx]
            gd = drone_step_data[d_idx]
            continuous_accel = b_guidance_np[d_idx]
            
            # Extract the first element (winch rate) safely from the array
            continuous_winch = float(np.clip(b_winch_np[d_idx].flatten()[0], -1.0, 1.0))
            
            roof_h = float(d.mission["roof_height"])

           # Actual surface height directly under the cargo payload
            cargo_xy = d.airframe.p[:2] + d.winch.sway_offset
            actual_surf_z = world.get_surface_height(cargo_xy[0], cargo_xy[1])
            dist_to_target_xy = float(np.linalg.norm(d.airframe.p[:2] - d.target[:2]))

            # Step continuous winch tether with real ground and dropzone verification
            cable_l, tension, sway, released = d.winch.step(
                continuous_winch_cmd=continuous_winch,
                drone_p=d.airframe.p, drone_v=d.airframe.v,
                wind_3d=drone_wind,
                surface_z=actual_surf_z,
                target_surface_z=roof_h,
                dist_to_target_xy=dist_to_target_xy,
                dt=dt
            )

            # Handle dynamic touch-and-go delivery
            if released:
                d.packages_delivered += 1
                delivered_boxes.append(np.array([d.target[0], d.target[1], roof_h], dtype=np.float32))
                dispatcher.report_completed(d.drone_id, d.mission["order_idx"])
                # Immediately acquire next potential goal without stopping
                d.mission = dispatcher.dispatch_next_mission(d.drone_id, d.bms.soc, d.airframe.p)
                d.target = d.mission["target_pos"].copy()
                d.winch.reload_new_cargo()
                print(f">>> [DYNAMIC DELIVERY] Drone #{d.drone_id+1} dropped payload at {d.mission['bldg_id']}! Swooping to next task. <<<")

            # Step geometric SE(3) actuator mixer with passive barrier shield
            omega_cmds, f_thrust, a_safe = d.mixer.mix(
                drone_health=d.airframe.motor_health, p_est=d.ekf.p, v_est=d.ekf.v,
                R_est=gd["R_est"], omega_est=gd["omega_est"],
                neural_accel_cmd=continuous_accel, obstacles=obstacles_cbf,
                swarm_states=swarm_states, drone_idx=d_idx, has_payload=d.winch.has_payload, dt=dt
            )

            # Step high-fidelity 8-DOF BEMT RK4 dynamics
            d.airframe.step(
                motor_cmds=omega_cmds, current_wind=drone_wind, rho_air=gd["s_data"]["rho_air"],
                ice_index=d.icing.ice_thickness_mm / 3.0, surface_z=gd["current_surf_z"],
                has_payload=d.winch.has_payload, dt=dt
            )

            if int(flight_time * 60) % 6 == 0:
                d.trail.append(d.airframe.p[:2].copy())

            if d_idx == selected_drone_idx:
                r_r, r_p, r_y = rot_to_euler(gd["R_est"])
                selected_diag = {
                    "v_pack": gd["v_pack"], "soc": gd["bms_soc"], "temp_c": d.bms.temp_c,
                    "thrust": f_thrust, "tension": d.winch.tension_n,
                    "cable_l": d.winch.cable_len, "cable_v": d.winch.cable_vel,
                    "accel_cmd": continuous_accel, "latency": inf_latency_ms,
                    "gnss_status": "GNSS REJECTED [SPOOFING ACTIVE]" if inject_spoof else "RTK FIXED [NOMINAL]",
                    "jerk": float(np.linalg.norm(gd["meas_accel"] - d.airframe.accel_linear)),
                    "phase": float(gd["phase_shift"][0]),
                    "v_tas": gd["s_data"]["v_tas"], "flow_2d": gd["flow_2d"],
                    "adsb_d": gd["adsb_d"], "siren_s": gd["siren_s"],
                    "roll": r_r, "pitch": r_p, "yaw": r_y, "alt": d.airframe.p[2],
                    "airspeed": float(np.linalg.norm(d.airframe.v)), "vspeed": d.airframe.v[2],
                    "wind_speed": gd["wind_speed"], "wind_from_deg": gd["wind_from_deg"]
                }

        # ==========================================
        # 4. GRAPHICS RENDERING & TELEMETRY HUD
        # ==========================================
        screen.fill((10, 13, 20))
        act_d = drones[selected_drone_idx]
        act_tgt = act_d.target

        render_genesis_mirror(
            mirror=genesis_mirror, world=world, drones=drones,
            traffic_engine=traffic_engine, agent_pos=agent_pos,
            delivered_boxes=delivered_boxes, selected_drone_idx=selected_drone_idx,
            cam_mode=cam_mode, cam_yaw=cam_yaw, cam_pitch=cam_pitch,
            cam_focus=cam_focus, active_target=act_tgt, screen=screen,
            view_w=VIEW_W, view_h=VIEW_H, font_small=font_small
        )
        pygame.draw.line(screen, (35, 45, 60), (VIEW_W, 0), (VIEW_W, H), 2)

        sb_x = VIEW_W + 14
        screen.blit(font_large.render("SURVIVAL CORE V39.0", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Post-Human Sensorimotor Swarm • Continuous PPO", True, (130, 160, 190)), (sb_x, 38))

        pfd_rect = (sb_x, 65, 140, 110)
        draw_pfd_artificial_horizon(
            screen, pfd_rect, 
            selected_diag.get("roll", 0.0), selected_diag.get("pitch", 0.0),
            selected_diag.get("alt", 0.0), selected_diag.get("airspeed", 0.0),
            selected_diag.get("vspeed", 0.0), font_small
        )

        map_rect = (sb_x + 148, 65, 260, 150)
        draw_tactical_radar_minimap(screen, map_rect, world, drones, selected_drone_idx, traffic_engine.vehicles)

        fl_y = 225
        screen.blit(font_mono.render("SWARM FLEET [TAB] | CAM [C]:", True, (240, 240, 240)), (sb_x, fl_y))
        for d_i, d in enumerate(drones):
            c = swarm_colors[d_i % len(swarm_colors)]
            tag = f"[D{d_i+1}]" + ("*" if d_i == selected_drone_idx else " ")
            stat = "RTH" if d.rth_active else "DELIVER"
            screen.blit(font_small.render(f"{tag} {stat:7s} | {d.bms.soc*100:.0f}% | Drops:{d.packages_delivered}", True, c), (sb_x + 10, fl_y + 18 + d_i * 18))

        cur_mission = act_d.mission
        tot_deliv = dispatcher.total_completed
        screen.blit(font_mono.render("ACTIVE TARGET WELL:", True, (240, 240, 240)), (sb_x + 220, fl_y))
        screen.blit(font_small.render(f"Goal: {cur_mission['bldg_id']} (Tag #{cur_mission['tag_id']})", True, (0, 220, 255)), (sb_x + 220, fl_y + 18))
        screen.blit(font_small.render(f"Addr: {cur_mission['addr'][:16]}", True, (0, 255, 140)), (sb_x + 220, fl_y + 36))
        screen.blit(font_small.render(f"Mode: Continuous Flight", True, (255, 200, 40)), (sb_x + 220, fl_y + 54))

        p1_y = 300
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p1_y - 8), (W - 14, p1_y - 8), 1)
        screen.blit(font_mono.render("1. SWARM TRANSFORMER CROSS-ATTENTION:", True, (240, 240, 240)), (sb_x, p1_y))
        dist_dest = float(np.linalg.norm(act_d.airframe.p[:2] - act_d.target[:2]))
        screen.blit(font_small.render(f"Distance to Goal:  {dist_dest:.1f} m (Potential Field Match)", True, (0, 220, 255)), (sb_x + 10, p1_y + 18))
        screen.blit(font_small.render(f"Total Deliveries:  {tot_deliv} Swoop Touchdowns", True, (0, 255, 140)), (sb_x + 10, p1_y + 34))

        p2_y = p1_y + 54
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (W - 14, p2_y), 1)
        screen.blit(font_mono.render("2. CONTINUOUS SENSORIMOTOR ACTION:", True, (240, 240, 240)), (sb_x, p2_y + 8))
        a_cmd = selected_diag.get("accel_cmd", np.zeros(3))
        screen.blit(font_small.render(f"Neural Accel:      [{a_cmd[0]:+.2f}, {a_cmd[1]:+.2f}, {a_cmd[2]:+.2f}] m/s²", True, (0, 220, 255)), (sb_x + 10, p2_y + 26))
        screen.blit(font_small.render(f"Continuous Winch:  {selected_diag.get('cable_v', 0.0):+.2f} m/s (Len: {selected_diag.get('cable_l', 0.08):.2f}m)", True, (0, 255, 140)), (sb_x + 10, p2_y + 42))

        p3_y = p2_y + 60
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (W - 14, p3_y), 1)
        screen.blit(font_mono.render("3. PASSIVE CONTROL BARRIER SHIELD:", True, (240, 240, 240)), (sb_x, p3_y + 8))
        screen.blit(font_small.render(f"Barrier Status:    PASSIVE [100% UNCONSTRAINED FLIGHT]", True, (0, 255, 140)), (sb_x + 10, p3_y + 26))
        screen.blit(font_small.render(f"Tether Snag Limit: {selected_diag.get('tension', 1.2):.1f} N / 48.0 N Cutoff", True, (0, 220, 255)), (sb_x + 10, p3_y + 42))

        p4_y = p3_y + 60
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (W - 14, p4_y), 1)
        screen.blit(font_mono.render("4. ATMOSPHERIC STATE & WIND SURFING:", True, (240, 240, 240)), (sb_x, p4_y + 8))
        w_mode = weather_engine.get_mode_name()
        screen.blit(font_small.render(f"Regime:            {w_mode} ([W] Shift)", True, (255, 180, 40)), (sb_x + 10, p4_y + 26))
        screen.blit(font_small.render(f"Wind Sensor:       {selected_diag.get('wind_speed', 0.0):.1f} m/s @ {selected_diag.get('wind_from_deg', 0.0):.0f}°", True, (0, 220, 255)), (sb_x + 10, p4_y + 42))

        p5_y = p4_y + 60
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p5_y), (W - 14, p5_y), 1)
        screen.blit(font_mono.render("5. QUANTUM INERTIAL NAVIGATION:", True, (240, 240, 240)), (sb_x, p5_y + 8))
        g_stat = selected_diag.get("gnss_status", "NOMINAL")
        screen.blit(font_small.render(f"GPS State:         {g_stat}", True, (255, 60, 80) if "REJECTED" in g_stat else (0, 255, 140)), (sb_x + 10, p5_y + 26))
        screen.blit(font_small.render(f"Quantum Phase:     ΔΦ = {selected_diag.get('phase', 0.0):.3f} rad (Rb-87 Raman)", True, (0, 220, 255)), (sb_x + 10, p5_y + 42))

        bot_y = H - 38
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y - 8), (W - 14, bot_y - 8), 1)
        screen.blit(font_mono.render("[TAB] Drone | [C] Camera | [W] Weather | [CLICK] Goal Well", True, (255, 200, 40)), (sb_x + 8, bot_y))

        vert_spd = act_d.airframe.v[2]
        status_banner = f"ACTIVE: D{selected_drone_idx+1} | Speed: {selected_diag.get('airspeed', 0.0):.1f}m/s (Vz: {vert_spd:+.2f}m/s) | Deliveries: {tot_deliv}"
        screen.blit(font_large.render(status_banner, True, (240, 240, 240)), (15, 14))
        screen.blit(font_small.render(f"V39.0 Post-Human Swarm Transformer • Continuous PPO • View: {cam_mode}", True, (0, 210, 255)), (15, 40))

        pygame.display.flip()

        target_render_fps = float(os.environ.get("SURVIVAL_RENDER_FPS", 60.0))
        clock.tick(max(1, int(np.clip(target_render_fps, 1.0, 60.0))))

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    mp.freeze_support()
    run_all_physics_simulator()