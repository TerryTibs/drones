import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

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

# Stable Tether Delivery Constants
PAYLOAD_WEIGHT_N = CARGO_MASS_KG * G_ACCEL
GUILLOTINE_THRESHOLD_N = 48.0
HOVER_LOCK_RADIUS_M = 1.80           
HOVER_LOCK_SPEED_MAX_MS = 0.65       
HOVER_SETTLE_DURATION_S = 0.75

# Feature Dimensions: 139 features x 3 history frames = 417 dims
BASE_OBS_DIM = 139
HISTORY_LEN = 3
OBSERVATION_DIM = BASE_OBS_DIM * HISTORY_LEN
NUM_EXECUTIVE_STATES = 7
MODEL_WEIGHTS_FILE = "drone_master_brain_v24_full_sensory.pth"

NUM_SWARM_DRONES = 3

ACTION_NAMES = [
    "TRANSIT",             
    "HOVER_LOCK",          
    "WINCH_LOWER",         
    "TOUCHDOWN_RELEASE",   
    "WINCH_RETRACT",       
    "EMERGENCY_DIVERT",    
    "AUTO_DISPATCH"        
]

# =============================================================================
# 1. MATHEMATICAL UTILITIES & LIE ALGEBRA
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
# DYNAMIC SELF-ADJUSTMENT & SPATIAL HASH ENGINES
# =============================================================================
class SpatialHashGrid2D:
    def __init__(self, cell_size=30.0):
        self.cell_size = cell_size
        self.buckets = {}

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

class OnlineMassAndCoMEstimator:
    def __init__(self, nominal_mass=1.80):
        self.est_mass = nominal_mass
        self.com_offset_xy = np.zeros(2, dtype=np.float32)

    def update(self, thrust_total, accel_z, steady_tau_xy, dt=0.016):
        effective_g = max(1.0, float(accel_z + G_ACCEL))
        instant_mass = thrust_total / effective_g
        if 1.2 <= instant_mass <= 2.4 and abs(accel_z) < 1.5:
            self.est_mass += 0.005 * (instant_mass - self.est_mass)

        if thrust_total > 12.0 and abs(accel_z) < 0.8:
            delta_x = float(np.clip(-steady_tau_xy[1] / thrust_total, -0.02, 0.02))
            delta_y = float(np.clip(steady_tau_xy[0] / thrust_total, -0.02, 0.02))
            self.com_offset_xy[0] += 0.004 * (delta_x - self.com_offset_xy[0])
            self.com_offset_xy[1] += 0.004 * (delta_y - self.com_offset_xy[1])

        return self.est_mass, self.com_offset_xy.copy()

class DynamicSelfAdjustingAirframeEngine:
    def __init__(self, dry_mass=DRY_MASS_KG, nominal_inertia=INERTIA_MATRIX):
        self.dry_mass = dry_mass
        self.J_nominal = nominal_inertia.copy()
        self.J_estimated = self.J_nominal.copy()
        self.k_R = np.array([2.40, 2.40, 1.80], dtype=np.float32)
        self.k_w = np.array([0.90, 0.90, 0.60], dtype=np.float32)

    def update(self, cable_len, has_payload, cargo_mass, sway_2d, v_pack, dt=0.016):
        m_payload = cargo_mass if has_payload else 0.0
        total_mass = self.dry_mass + m_payload

        effective_L = max(0.08, float(cable_len))
        sway_mag = float(np.linalg.norm(sway_2d))
        r_pendulum = effective_L * math.sin(min(0.5, sway_mag / effective_L))
        delta_J = 0.15 * m_payload * (r_pendulum ** 2)

        self.J_estimated[0, 0] = self.J_nominal[0, 0] + delta_J
        self.J_estimated[1, 1] = self.J_nominal[1, 1] + delta_J
        self.J_estimated[2, 2] = self.J_nominal[2, 2]

        voltage_boost = float(np.clip(24.0 / max(18.0, v_pack), 1.0, 1.30))
        pendulum_natural_freq = math.sqrt(G_ACCEL / effective_L)
        resonance_attenuation = float(np.clip(pendulum_natural_freq / 4.5, 0.65, 1.0)) if cable_len > 0.3 else 1.0

        base_stiffness = 2.40 * resonance_attenuation * voltage_boost
        self.k_R[0] = base_stiffness * (self.J_estimated[0, 0] / self.J_nominal[0, 0])
        self.k_R[1] = base_stiffness * (self.J_estimated[1, 1] / self.J_nominal[1, 1])
        self.k_R[2] = 1.80 * voltage_boost

        self.k_w[0] = 2.0 * math.sqrt(max(1e-4, float(self.k_R[0] * self.J_estimated[0, 0]))) * 0.95
        self.k_w[1] = 2.0 * math.sqrt(max(1e-4, float(self.k_R[1] * self.J_estimated[1, 1]))) * 0.95
        self.k_w[2] = 2.0 * math.sqrt(max(1e-4, float(self.k_R[2] * self.J_estimated[2, 2]))) * 0.85

        return total_mass, self.J_estimated, self.k_R, self.k_w

class RooftopSurfaceNormalEstimator:
    def __init__(self):
        self.surface_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def estimate_normal(self, drone_pos, conical_ranges_8):
        cone_angle = math.radians(25.0)
        sin_c = math.sin(cone_angle)
        cos_c = math.cos(cone_angle)
        pts = []

        for i in range(8):
            r = min(12.0, float(conical_ranges_8[i]))
            ang = 2.0 * math.pi * i / 8.0
            ray_dir = np.array([sin_c * math.cos(ang), sin_c * math.sin(ang), -cos_c], dtype=np.float32)
            pts.append(drone_pos + r * ray_dir)

        pts = np.array(pts, dtype=np.float32)
        centroid = np.mean(pts, axis=0)
        centered_pts = pts - centroid

        try:
            _, _, vh = np.linalg.svd(centered_pts)
            normal = vh[-1]
            if normal[2] < 0:
                normal = -normal
            self.surface_normal = normal / max(1e-4, float(np.linalg.norm(normal)))
        except np.linalg.LinAlgError:
            self.surface_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        return self.surface_normal.copy()

# ==========================================
# 2. METROPOLITAN CITY GRID & ROOFTOP CLUTTER
# ==========================================
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
        self.spatial_hash = SpatialHashGrid2D(cell_size=30.0)
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
        self.target_mode_idx = 0
        self.regime_timer = 0.0
        self.regime_duration = 45.0
        self.blend_factor = 1.0

    def cycle_weather(self):
        self.target_mode_idx = (self.target_mode_idx + 1) % len(self.weather_modes)
        self.regime_timer = 0.0
        self.blend_factor = 0.0
        print(f"\n>>> [WEATHER SHIFT] BLENDING INTO: {self.weather_modes[self.target_mode_idx]} <<<")

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
# 7. ROTOR INFLOW HEALTH & ANTI-RESONANCE PBD WINCH
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

class AutonomousWinchDeliverySystem:
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
        self.state = "STANDBY"
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
        self.state = "STANDBY"
        self.tension_n = PAYLOAD_WEIGHT_N
        self.prev_tension_n = PAYLOAD_WEIGHT_N
        self.tension_derivative = 0.0
        self.touchdown_timer = 0.0
        self.delivery_completed = False
        self.is_skid_grounded = False

    def step(self, drone_z=0.0, target_surface_z=0.0, is_settled=False, is_surface_safe=True, 
             wire_hit=False, agent_breach=False, agent_dist=30.0, wind_2d=None, 
             drone_accel_linear=None, neural_token_idx=0, nn_winch_rate=0.0, dt=0.016, 
             drone_p=None, drone_v=None, wind_3d=None, **kwargs):
        
        if self.severed:
            self.state = "GUILLOTINE"
            return self.cable_len, 0.0, np.zeros(2, dtype=np.float32)

        if drone_p is None:
            drone_p = np.array([0.0, 0.0, drone_z], dtype=np.float32)
        if drone_v is None:
            drone_v = np.zeros(3, dtype=np.float32)
        if wind_3d is None:
            if wind_2d is not None:
                wind_3d = np.array([wind_2d[0], wind_2d[1], 0.0], dtype=np.float32)
            else:
                wind_3d = np.zeros(3, dtype=np.float32)

        wind_speed_xy = float(np.linalg.norm(wind_3d[:2]))
        payload_z = drone_p[2] - self.cable_len
        clearance_to_roof = payload_z - target_surface_z
        dropzone_occupied = (agent_dist < 2.8) and (target_surface_z < 2.0) and (self.state == "STANDBY")
        lockout = (not is_surface_safe) or wire_hit or (not is_settled) or dropzone_occupied

        if agent_breach and (target_surface_z < 2.0) and self.state in ["LOWERING", "TOUCHDOWN"]:
            self.state = "ABORT_RETRACT"

        # Natural pendulum frequency & anti-resonance notch
        omega_pendulum = math.sqrt(G_ACCEL / max(0.08, float(self.cable_len)))
        in_resonance_band = (1.8 < omega_pendulum < 2.5) and (self.state == "LOWERING")

        if self.state == "STANDBY":
            self.cable_len = 0.08
            self.cable_vel = 0.0
            if not lockout and self.has_payload and is_settled:
                self.state = "LOWERING"

        elif self.state == "LOWERING":
            if lockout and self.cable_len > 0.12:
                self.state = "RETRACTING"
            else:
                flare_scale = float(np.clip(clearance_to_roof / 1.20, 0.20, 1.0)) if clearance_to_roof < 1.20 else 1.0
                base_rate = float(np.clip(2.40 - 0.12 * wind_speed_xy, 1.20, 2.40)) * flare_scale
                
                # Punch through resonant frequency band to suppress vortex-induced cable vibrations
                if in_resonance_band and clearance_to_roof > 1.5:
                    base_rate = 3.20

                self.cable_vel = base_rate + np.clip(nn_winch_rate * 0.8, -0.6, 0.6)
                self.cable_len += self.cable_vel * dt

                # Tension derivative and clearance touchdown detection
                fast_tension_drop = (self.tension_derivative < -35.0) and (clearance_to_roof < 0.40)
                if clearance_to_roof <= 0.10 or fast_tension_drop or self.contact_normal_force >= 0.40 * PAYLOAD_WEIGHT_N:
                    self.state = "TOUCHDOWN"
                    self.touchdown_timer = 0.0

        elif self.state == "TOUCHDOWN":
            self.cable_vel = 0.0
            self.touchdown_timer += dt
            if self.touchdown_timer >= 0.25:
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
            return self.cable_len, self.tension_n, self.sway_offset

        predicted_pos = np.zeros_like(self.nodes_pos)
        predicted_pos[0] = self.nodes_pos[0]

        # Stabilizing aerodynamic fin vanes on payload
        v_rel_tip = self.nodes_vel[-1] - wind_3d
        v_rel_xy_speed = math.hypot(v_rel_tip[0], v_rel_tip[1])
        if v_rel_xy_speed > 0.2:
            wind_track_yaw = math.atan2(v_rel_tip[1], v_rel_tip[0])
            yaw_err = (self.cargo_yaw - wind_track_yaw + math.pi) % (2.0 * math.pi) - math.pi
            fin_align_torque = -1.8 * yaw_err * v_rel_xy_speed - 0.45 * self.cargo_yaw_rate
            self.cargo_yaw_rate += fin_align_torque * dt
            self.cargo_yaw += self.cargo_yaw_rate * dt

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

        # Spring-damper soft bumper contact
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

        if self.has_payload:
            a_vert = float(drone_accel_linear[2]) if drone_accel_linear is not None else 0.0
            theta_mag = float(np.linalg.norm(self.theta))
            t_grav = max(0.5, CARGO_MASS_KG * (G_ACCEL * math.cos(theta_mag) + a_vert) - self.contact_normal_force)
            t_centrif = CARGO_MASS_KG * L * float(np.dot(self.theta_dot, self.theta_dot))
            new_tension = float(np.clip(t_grav + t_centrif, 1.2, 45.0))
        else:
            new_tension = 1.2

        self.tension_derivative = (new_tension - self.prev_tension_n) / dt
        self.prev_tension_n = self.tension_n
        self.tension_n = new_tension

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
# 8. UPGRADED: BEMT COAXIAL AIRFRAME WITH RK4, VRS, AND WALL SUCTION
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
# 9. ADVANCED SENSORS & SENSORY REALITY SUITE
# =============================================================================
class AdvancedMultiModalSensorSuite:
    def __init__(self, world: MetropolitanCityGridWorld400m):
        self.world = world

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

# ==========================================
# 10. SPATIAL HASH ACCELERATED MACRO LIDAR
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

            p_end = np.array([px + dx * 30.0, py + dy * 30.0], dtype=np.float32)
            candidate_houses = self.world.spatial_hash.query_segment([px, py], p_end)
            if not candidate_houses:
                candidate_houses = self.world.houses

            for h in candidate_houses:
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

        cone_angle = math.radians(25.0)
        sin_c = math.sin(cone_angle)
        cos_c = math.cos(cone_angle)

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
# 11. QUANTUM SENSOR & SAGE-HUSA ADAPTIVE ERROR-STATE EKF ON SO(3)
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
        try:
            S_inv = np.linalg.inv(S)
            K = self.P @ H.T @ S_inv
        except np.linalg.LinAlgError:
            return

        innov_flat = np.array(innovation).flatten()
        gamma = float(innov_flat @ S_inv @ innov_flat)
        
        # Adaptive Sage-Husa residual multiplier estimation
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
        r_adaptive = (r_cov_base / max(0.05, flow_quality)) + (min(35.0, z_slant) / 35.0) * 0.04
        r_adaptive = float(np.clip(r_adaptive, 0.015, 0.35))

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
# 12. UNCERTAINTY-AWARE CBF & SELF-ADJUSTING GEOMETRIC SE(3) MIXER
# =============================================================================
class ReciprocalSwarmCBFQPFilter:
    def __init__(self):
        self.alpha_obs = CBF_ALPHA_OBSTACLE
        self.safe_radius = CBF_BASE_SAFE_RADIUS_M
        self.swarm_r = SWARM_INTER_DRONE_SAFE_RADIUS_M
        self.downwash_expansion = DOWNWASH_CONE_EXPANSION
        self.downwash_depth = DOWNWASH_HAZARD_DEPTH_M

    def enforce_safety(self, drone_idx, drone_p, drone_v, nominal_accel, obstacles, swarm_states=None, 
                       target_bldg_id=None, departure_bldg_id=None, wind_2d=None, ekf_pos_cov_trace=0.05):
        A_rows = []
        b_rows = []

        if wind_2d is None:
            wind_2d = np.zeros(2, dtype=np.float32)

        # Dynamic uncertainty inflation: expands safety bubble when EKF covariance or gustiness rises
        cov_inflation = 1.2 * min(2.5, math.sqrt(max(0.0, float(ekf_pos_cov_trace))))
        gust_inflation = 0.22 * float(np.linalg.norm(wind_2d))
        dynamic_safe_radius = self.safe_radius + cov_inflation + gust_inflation

        for obs in obstacles:
            obs_id = obs.get("id")
            obs_h = obs.get("height", 0.0)
            if obs_id in [target_bldg_id, departure_bldg_id]:
                continue
            
            rel_pos = drone_p - obs["pos"]
            dist_xy = math.hypot(rel_pos[0], rel_pos[1])

            n_xy = rel_pos[:2] / max(1e-4, dist_xy)
            v_closing = -float(np.dot(drone_v[:2], n_xy))
            tailwind_proj = -float(np.dot(wind_2d[:2], n_xy))
            
            dynamic_stopping_buffer = (max(0.0, v_closing) ** 2) / (2.0 * 3.8) + max(0.0, tailwind_proj) * 0.35
            effective_safe_r = dynamic_safe_radius + min(4.5, dynamic_stopping_buffer)
            h_obs = dist_xy - (obs["radius"] + effective_safe_r)

            # Self-tuning convergence rate based on approach dynamics
            eff_alpha = self.alpha_obs * (8.0 / (8.0 + max(0.0, v_closing)))

            if drone_p[2] <= (obs_h - 0.4) and h_obs < 4.5:
                grad_h = np.array([n_xy[0], n_xy[1], 0.0], dtype=np.float32)
                v_tangent = drone_v[0] * (-n_xy[1]) + drone_v[1] * n_xy[0]
                centripetal = (v_tangent ** 2) / max(0.5, dist_xy)
                l_fh = float(np.dot(grad_h, drone_v))
                
                A_rows.append(grad_h)
                b_rows.append(-l_fh - eff_alpha * h_obs - centripetal)

            if dist_xy <= (obs["radius"] + 0.5) and drone_p[2] >= (obs_h - 0.2):
                h_roof = drone_p[2] - (obs_h + 0.15)
                if h_roof < 0.8:
                    A_rows.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
                    b_rows.append(-drone_v[2] - 2.0 * h_roof)

        if swarm_states:
            for j, other in enumerate(swarm_states):
                if j == drone_idx: continue
                p_other, v_other = other["p"], other["v"]

                if drone_p[2] < 6.0 and p_other[2] < 6.0:
                    continue

                dz = drone_p[2] - p_other[2]
                d_p = drone_p[:2] - other["p"][:2]
                dist_ij = float(np.linalg.norm(d_p))

                if -self.downwash_depth < dz < -0.4:
                    downwash_cone_r = self.swarm_r + self.downwash_expansion * abs(dz) + cov_inflation
                    h_downwash = dist_ij - downwash_cone_r
                    if h_downwash < 3.2:
                        n_downwash = np.array([d_p[0]/max(1e-4, dist_ij), d_p[1]/max(1e-4, dist_ij), 0.0], dtype=np.float32)
                        v_rel = drone_v[:2] - v_other[:2]
                        l_fh_downwash = float(np.dot(n_downwash[:2], v_rel))
                        A_rows.append(n_downwash)
                        b_rows.append(-l_fh_downwash - 2.8 * h_downwash)

                if abs(dz) < 2.8:
                    h_swarm = dist_ij - (self.swarm_r + cov_inflation)
                    if h_swarm < 3.5:
                        n_ij = np.array([d_p[0]/max(1e-4, dist_ij), d_p[1]/max(1e-4, dist_ij), 0.0], dtype=np.float32)
                        v_rel = drone_v[:2] - v_other[:2]
                        l_fh_swarm = float(np.dot(n_ij[:2], v_rel))
                        A_rows.append(n_ij)
                        b_rows.append(-l_fh_swarm - 2.2 * h_swarm)

        h_gnd = drone_p[2] - 0.20
        if h_gnd < 1.6 and drone_p[2] > 0.25:
            A_rows.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
            b_rows.append(-3.5 * drone_v[2] - 4.5 * max(0.05, h_gnd))

        fence_limit = 180.0
        for axis in [0, 1]:
            if drone_p[axis] > (fence_limit - 15.0):
                dist_fence = max(0.1, fence_limit - drone_p[axis])
                grad = np.array([-1.0 if axis == 0 else 0.0, -1.0 if axis == 1 else 0.0, 0.0], dtype=np.float32)
                A_rows.append(grad)
                b_rows.append(drone_v[axis] - 2.5 * dist_fence)
            elif drone_p[axis] < (-fence_limit + 15.0):
                dist_fence = max(0.1, drone_p[axis] - (-fence_limit))
                grad = np.array([1.0 if axis == 0 else 0.0, 1.0 if axis == 1 else 0.0, 0.0], dtype=np.float32)
                A_rows.append(grad)
                b_rows.append(-drone_v[axis] - 2.5 * dist_fence)

        if not A_rows:
            return nominal_accel

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

class TactileLandingReflex:
    def __init__(self):
        self.contact_timer = 0.0
        self.is_latched = False

    def reset(self):
        self.contact_timer = 0.0
        self.is_latched = False

    def compute_reflex(self, f_skid_left, f_skid_right, current_mass, dt=0.016):
        total_skid_force = max(0.0, f_skid_left + f_skid_right)
        asymmetry = f_skid_left - f_skid_right
        total_weight = current_mass * G_ACCEL

        if total_skid_force > 1.2:
            self.contact_timer += dt
        else:
            self.contact_timer = max(0.0, self.contact_timer - dt * 2.0)

        tau_level_roll = -0.035 * asymmetry
        load_fraction = float(np.clip(total_skid_force / (0.85 * total_weight), 0.0, 1.0))
        thrust_attenuation = 1.0 - 0.92 * (load_fraction ** 1.3)

        if self.contact_timer >= 0.18 and load_fraction >= 0.70:
            self.is_latched = True
            thrust_attenuation = 0.0

        return thrust_attenuation, tau_level_roll, self.is_latched

class RateDampedActuatorMixerWithCBF:
    def __init__(self):
        self.cbf_filter = ReciprocalSwarmCBFQPFilter()
        self.tactile_reflex = TactileLandingReflex()
        
        self.adaptive_airframe = DynamicSelfAdjustingAirframeEngine()
        self.inflow_estimator = OnlineRotorInflowAndHealthEstimator()
        self.mass_com_estimator = OnlineMassAndCoMEstimator()
        self.roof_normal_estimator = RooftopSurfaceNormalEstimator()

        self.wind_filt = np.zeros(3, dtype=np.float32)
        self.b3_d_prev = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.z_err_integral = 0.0
        self.xy_err_integral = np.zeros(2, dtype=np.float32)
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

    def mix(self, drone_health, p_est, v_est, R_est, omega_est, p_target, neural_guidance, token_idx,
            estimated_wind, obstacles, swarm_states=None, drone_idx=0, ref_accel=None, ref_vel=None,
            target_bldg_id=None, departure_bldg_id=None, cable_len=0.08, sway_2d=None, load_cell_3d=None, ige_fac=1.0,
            v_pack=23.5, bms_temp_c=22.0, has_payload=True, ice_mm=0.0, dt=0.016, 
            conical_ranges_8=None, esc_i_rms=None, esc_t_mosfet=None,
            f_skid_left=0.0, f_skid_right=0.0, ekf_pos_cov_trace=0.05):

        if ref_accel is None: ref_accel = np.zeros(3, dtype=np.float32)
        if ref_vel is None: ref_vel = np.zeros(3, dtype=np.float32)
        if swarm_states is None: swarm_states = []

        # 1. Dynamic Inertia Identification and Adaptive Gain Scheduling
        current_mass, J_eff, k_R_dyn, k_w_dyn = self.adaptive_airframe.update(
            cable_len=cable_len, has_payload=has_payload, cargo_mass=CARGO_MASS_KG,
            sway_2d=sway_2d if sway_2d is not None else np.zeros(2, dtype=np.float32),
            v_pack=v_pack, dt=dt
        )
        weight_force = current_mass * G_ACCEL

        is_commanding_flight = (p_target[2] > 0.5)
        if not is_commanding_flight and p_est[2] <= 0.10:
            self.z_err_integral = 0.0
            self.xy_err_integral.fill(0.0)
            return np.zeros(8, dtype=np.float32), 0.0, np.zeros(3, dtype=np.float32)

        delta_p = p_target - p_est
        delta_p_clamped = np.clip(delta_p, -5.0, 5.0)

        if p_target[2] <= 0.10 and p_est[2] < 1.5:
            v_cmd_z = float(np.clip(ref_vel[2], -0.25, 0.15))
        elif p_est[2] < 1.20 and ref_vel[2] > 0.0:
            v_cmd_z = float(np.clip(ref_vel[2], 0.20, 0.45))
        else:
            v_cmd_z = float(np.clip(ref_vel[2] + 0.40 * delta_p_clamped[2], -0.80, 1.30))

        is_descending = (v_cmd_z < -0.10)
        climb_speed = max(0.0, float(v_est[2]))

        if is_descending:
            k_p_xy = 0.95
            k_d_xy = 4.40
        elif climb_speed > 0.15:
            k_p_xy = float(1.20 / (1.0 + 0.30 * climb_speed))
            k_d_xy = float(3.20 + 1.20 * min(1.0, climb_speed / 2.0))
        else:
            k_p_xy = 1.80
            k_d_xy = 2.80

        v_err_xy = np.clip(ref_vel[:2] - v_est[:2], -3.5, 3.5)
        a_p_xy = k_p_xy * delta_p_clamped[:2]
        a_d_xy = k_d_xy * v_err_xy

        is_airborne = (p_est[2] > 0.06)
        if is_airborne and p_est[2] > 0.07:
            self.xy_err_integral += delta_p_clamped[:2] * dt
            self.xy_err_integral = np.clip(self.xy_err_integral, -2.5, 2.5)
        else:
            self.xy_err_integral.fill(0.0)

        a_i_xy = 0.55 * self.xy_err_integral
        a_track_xy = a_p_xy + a_d_xy + a_i_xy

        sink_error = v_cmd_z - v_est[2]
        if p_est[2] > 2.0 and p_target[2] > 0.50:
            self.z_err_integral = np.clip(self.z_err_integral + sink_error * dt, -0.45, 0.45)
        else:
            self.z_err_integral = 0.0

        k_vertical = 2.0 if (p_est[2] < 1.20 and v_cmd_z > 0.0) else 3.2
        a_track_z = k_vertical * sink_error + 1.0 * self.z_err_integral
        a_track = np.array([a_track_xy[0], a_track_xy[1], a_track_z], dtype=np.float32)

        alpha_w = dt / (0.85 + dt)
        self.wind_filt[:2] += alpha_w * (estimated_wind[:2] - self.wind_filt[:2])
        alpha_w_z = dt / (0.20 + dt)
        self.wind_filt[2] += alpha_w_z * (estimated_wind[2] - self.wind_filt[2])
        
        v_air_rel = v_est - self.wind_filt
        speed_rel = float(np.linalg.norm(v_air_rel))
        f_aero_drag = -0.5 * 1.225 * np.array([0.22, 0.22, 0.38], dtype=np.float32) * v_air_rel * max(1.0, speed_rel)
        a_wind_feedforward = -f_aero_drag / current_mass

        if self.wind_filt[2] < -0.30:
            f_downburst = 0.5 * 1.225 * 0.38 * (self.wind_filt[2] ** 2)
            a_wind_feedforward[2] += (f_downburst / current_mass) * 1.60

        a_anti_sway = np.zeros(3, dtype=np.float32)
        if cable_len > 0.12 and sway_2d is not None:
            a_anti_sway[:2] -= 1.2 * sway_2d

        a_neural_ff = neural_guidance[:3] * 6.0 if neural_guidance is not None else np.zeros(3, dtype=np.float32)
        a_nominal = a_track + ref_accel + a_anti_sway + a_wind_feedforward + a_neural_ff

        # 2. Uncertainty-Aware Dynamic Safety Filtering
        a_safe = self.cbf_filter.enforce_safety(
            drone_idx, p_est, v_est, a_nominal, obstacles, swarm_states, 
            target_bldg_id=target_bldg_id, departure_bldg_id=departure_bldg_id,
            wind_2d=self.wind_filt[:2], ekf_pos_cov_trace=ekf_pos_cov_trace
        )
        
        F_d = a_safe * current_mass + current_mass * (-GRAVITY_G)

        is_near_landing = (p_est[2] < 3.0 and p_target[2] <= 0.20)
        is_climbing_from_base = (p_est[2] < 4.0 and ref_vel[2] > 0.20)
        is_vertical_flight = (abs(ref_vel[2]) > 0.35 and abs(ref_vel[0]) < 0.6 and abs(ref_vel[1]) < 0.6)

        wind_speed_xy = float(np.linalg.norm(self.wind_filt[:2]))
        storm_drag_n = 0.5 * 1.225 * 0.22 * (wind_speed_xy ** 2)
        wind_trim_deg = min(14.0, math.degrees(math.atan2(storm_drag_n, max(1.0, weight_force))))

        if is_near_landing:
            base_tilt_deg = 5.5 + wind_trim_deg
        elif is_climbing_from_base:
            base_tilt_deg = 5.5 + (0.5 * wind_trim_deg)
        elif is_vertical_flight or p_est[2] < 2.0:
            base_tilt_deg = 8.5 + (0.6 * wind_trim_deg)
        else:
            base_tilt_deg = 24.0

        if ige_fac > 1.02 and p_target[2] <= 0.10 and self.wind_filt[2] >= -0.30:
            F_d[2] /= ige_fac

        storm_active = (abs(self.wind_filt[0]) > 4.0 or abs(self.wind_filt[1]) > 4.0 or self.wind_filt[2] < -0.6)
        max_multiplier = 2.50 if storm_active else 1.80
        min_f_vert = (0.65 * weight_force) if is_airborne else 0.0
        max_f_vert = max_multiplier * weight_force
        F_d[2] = float(np.clip(F_d[2], min_f_vert, max_f_vert))

        thrust_ratio = max(1.0, float(F_d[2] / max(1.0, weight_force)))
        max_tilt_rad = math.radians(base_tilt_deg / thrust_ratio)

        f_horiz = math.hypot(F_d[0], F_d[1])
        f_vert = max(1.0, F_d[2])
        if f_horiz / f_vert > math.tan(max_tilt_rad):
            scale_h = (f_vert * math.tan(max_tilt_rad)) / f_horiz
            F_d[0] *= scale_h
            F_d[1] *= scale_h

        b3 = R_est[:, 2]
        f_thrust = float(np.dot(F_d, b3))
        f_thrust = float(np.clip(f_thrust, min_f_vert, max_f_vert))

        # 3. Geometric SE(3) Attitude Tracking on Lie Group SO(3)
        b3_d = F_d / max(1e-4, float(np.linalg.norm(F_d)))
        self.b3_d_prev = b3_d.copy()

        yaw_target = 0.0
        b1_c = np.array([math.cos(yaw_target), math.sin(yaw_target), 0.0], dtype=np.float32)
        b2_d = np.cross(b3_d, b1_c)
        b2_d_norm = max(1e-4, float(np.linalg.norm(b2_d)))
        b2_d /= b2_d_norm
        b1_d = np.cross(b2_d, b3_d)
        R_d = np.column_stack((b1_d, b2_d, b3_d)).astype(np.float32)

        R_err_mat = R_d.T @ R_est - R_est.T @ R_d
        e_R = 0.5 * vee_map(R_err_mat)
        e_w = omega_est

        gyroscopic = np.cross(omega_est, J_eff @ omega_est)
        tau = -k_R_dyn * e_R - k_w_dyn * e_w + gyroscopic
        tau[2] = float(np.clip(tau[2], -0.85, 0.85))

        if p_target[2] <= 0.15:
            thrust_scale, tau_roll_adj, is_latched = self.tactile_reflex.compute_reflex(
                f_skid_left, f_skid_right, current_mass, dt=dt
            )
            f_thrust *= thrust_scale
            tau[0] += tau_roll_adj
            if is_latched:
                f_thrust = 0.0
                tau.fill(0.0)
        else:
            self.tactile_reflex.reset()

        self._prev_f_thrust = f_thrust
        self._prev_tau_xy = np.array([tau[0], tau[1]], dtype=np.float32)

        # 4. Real-Time Rotor Inflow Health and Lift Compensation
        ct_estimated, aero_health = self.inflow_estimator.update(
            current_mass=current_mass, accel_z=ref_accel[2], 
            motor_omegas=self._last_motor_omegas, dt=dt
        )

        W_effective = drone_health.copy()
        if esc_i_rms is not None and esc_t_mosfet is not None:
            for k in range(8):
                if drone_health[k] > 0.0:
                    i_norm = float(esc_i_rms[k])
                    t_mos = float(esc_t_mosfet[k])
                    if t_mos > 75.0 or i_norm > 20.0:
                        derate = max(0.20, 1.0 - 0.015 * (t_mos - 75.0) - 0.02 * max(0.0, i_norm - 20.0))
                        W_effective[k] = min(W_effective[k], derate)

        if bms_temp_c < 5.0:
            temp_ratio = float(np.clip((bms_temp_c + 10.0) / 15.0, 0.40, 1.0))
            f_thrust = min(f_thrust, (1.20 + 0.60 * temp_ratio) * weight_force)

        desired_wrench = np.array([f_thrust, tau[0], tau[1], tau[2]], dtype=np.float32)

        W_health = np.diag(W_effective).astype(np.float32)
        B_active = self.B_nominal @ W_health
        try:
            B_pinv = np.linalg.pinv(B_active)
            T_cmds = B_pinv @ desired_wrench
        except np.linalg.LinAlgError:
            T_cmds = np.full(8, f_thrust / 8.0, dtype=np.float32)

        active_mask = (W_effective > 0.0)
        if np.any(active_mask):
            min_active_t = float(np.min(T_cmds[active_mask]))
            if min_active_t < 0.10:
                shift = min(0.35, 0.10 - min_active_t)
                T_cmds[active_mask] += shift

        T_cmds = np.clip(T_cmds * W_effective, 0.0, 10.5)

        omega_cmds = np.zeros(8, dtype=np.float32)
        for i in range(8):
            if W_effective[i] > 0.0:
                base_c = (C_T_CLEAN * COAXIAL_EFFICIENCY) if (i % 2 == 1) else C_T_CLEAN
                # Compensate for estimated aerodynamic lift degradation
                effective_c = base_c * aero_health
                omega_cmds[i] = math.sqrt(T_cmds[i] / max(1e-6, effective_c))

        self._last_motor_omegas = omega_cmds.copy()
        return omega_cmds, f_thrust, a_safe

# =============================================================================
# 13. 139-DIM MULTI-MODAL OBSERVATION SYNTHESIZER & TEMPORAL STACKER
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
        vec = []
        vec.extend((target_p - p_est) / 50.0)
        vec.extend(p_est / 200.0)
        vec.extend(v_est / 8.0)
        vec.extend(R_est.flatten())
        vec.extend(omega_est / 10.0)
        vec.extend(lidar_16 / 30.0)
        vec.extend([
            min_sonar / 15.0,
            v_tas / 20.0,
            q_dyn / 100.0,
            wire_dist / 50.0,
            agent_dist / 50.0
        ])
        vec.extend(wind_est / 8.0)
        vec.extend(conical_8 / 15.0)
        vec.extend([
            cable_len / 15.0,
            cable_vel / 5.0,
            tension_n / 50.0,
            sway_2d[0] / 2.0,
            sway_2d[1] / 2.0,
            1.0 if has_payload else 0.0,
            skid_force_left / 20.0,
            skid_force_right / 20.0
        ])
        vec.extend([
            bms_soc,
            v_pack / 25.2,
            bms_temp_c / 60.0,
            r0_dyn / 0.05,
            ice_idx,
            1.0 if is_skid_grounded else 0.0,
            chi2_val / 10.0,
            jerk_discrepancy / 16.0
        ])
        vec.extend(esc_i_rms / 25.0)
        vec.extend(esc_t_mosfet / 100.0)
        vec.extend(hook_a / 20.0)
        vec.extend(hook_w / 10.0)
        vec.extend(load_cell_3d / 50.0)
        vec.extend([
            alpha_deg / 45.0,
            beta_deg / 45.0,
            rh_pct / 100.0,
            dust_pm10 / 100.0,
            thermal_dt / 10.0,
            dolp
        ])
        vec.extend([
            hdop / 5.0,
            cn0_avg / 50.0,
            (rf_jam_pwr + 120.0) / 70.0,
            mag_grad / 1.0,
            optical_flow_2d[0] / 5.0,
            optical_flow_2d[1] / 5.0,
            flow_quality,
            1.0 if is_spoofed else 0.0
        ])
        vec.extend(aruco_offset_3d / 10.0)
        vec.extend([
            aruco_locked,
            siren_angle / math.pi,
            siren_spl / 100.0
        ])
        vec.extend(adsb_rel_pos / 200.0)
        vec.append(adsb_closing_spd / 30.0)

        for peer in peer_swarm_kinematics[:2]:
            vec.extend(peer["p_rel"] / 50.0)
            vec.extend(peer["v_rel"] / 10.0)
        while len(vec) < 130:
            vec.append(0.0)

        vec.extend(airframe_accel / 30.0)
        vec.extend(airframe_gyro / 10.0)
        vec.extend([ndvi, emf_rms / 100.0, ground_spl_dba / 100.0])

        out = np.array(vec[:self.dim], dtype=np.float32)
        if len(out) < self.dim:
            out = np.pad(out, (0, self.dim - len(out)))
        return out

    def stack_history(self, history_q, current_obs):
        if len(history_q) == 0:
            for _ in range(history_q.maxlen or HISTORY_LEN):
                history_q.append(current_obs.copy())
        else:
            history_q.append(current_obs.copy())
        return np.concatenate(list(history_q), axis=0).astype(np.float32)

# =============================================================================
# 14. DECENTRALIZED INDEPENDENT DRONE EXECUTIVE NETWORK (~2.2M PARAMETERS)
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

class DeepDenseFlightExecutiveNet(nn.Module):
    def __init__(self, in_features=OBSERVATION_DIM):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.SiLU()
        )
        
        self.block1 = DenseResBlock512(dim=512, hidden_dim=768)
        self.block2 = DenseResBlock512(dim=512, hidden_dim=768)
        self.block3 = DenseResBlock512(dim=512, hidden_dim=768)

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
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        feat = self.input_proj(x)
        feat = self.block1(feat)
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
# 15. 10-DELIVERY CONTINUOUS MULTI-MISSION HARVESTER & DECENTRALIZED TRAINER
# =============================================================================
def _worker_harvest_metropolitan(num_steps):
    np.random.seed()
    random.seed()
    world = MetropolitanCityGridWorld400m()
    lidar_engine = MacroPerceptionLiDARSuite(world)
    sensor_suite = AdvancedMultiModalSensorSuite(world)
    assurance = SurfaceAndWireAssuranceSuite(world)
    dispatcher = SwarmMissionDispatcher(world.manifest_addresses, world.depot_helipads)

    obs_list = []
    y_guidance, y_fsm, y_winch, y_health, y_trust, y_val = [], [], [], [], [], []

    drones = []
    for i in range(NUM_SWARM_DRONES):
        init_p = world.depot_helipads[i].copy()
        init_p[2] = 0.06
        d_inst = SwarmDroneInstance(i, init_p, world, dispatcher)
        drones.append(d_inst)

    dt = 0.016
    sim_time = 0.0

    weather_modes = ["CLEAR_NOMINAL", "CANYON_GALE", "FREEZING_RAIN", "MICROBURST_STORM"]
    samples_per_regime = max(1, num_steps // 4)

    obstacles_cbf = [{"pos": np.array([(h["aabb"][0]+h["aabb"][1])/2, (h["aabb"][2]+h["aabb"][3])/2, 0.0]), 
                      "radius": 11.5, "height": h["aabb"][4], "id": h["id"]} for h in world.houses]

    while len(obs_list) < num_steps:
        sim_time += dt

        regime_idx = min(3, len(obs_list) // samples_per_regime)
        mode = weather_modes[regime_idx]

        if mode == "CLEAR_NOMINAL":
            episode_wind = np.array([1.5, -0.5, 0.0], dtype=np.float32)
            ice_active = False
            curr_ice = 0.0
            amb_temp = 22.0
        elif mode == "CANYON_GALE":
            episode_wind = np.array([6.5, 3.2, 0.0], dtype=np.float32)
            ice_active = False
            curr_ice = 0.0
            amb_temp = 14.0
        elif mode == "FREEZING_RAIN":
            episode_wind = np.array([2.5, -2.0, 0.0], dtype=np.float32)
            ice_active = True
            curr_ice = 0.45
            amb_temp = -4.5
        else:
            downburst_z = -3.2 * float(math.tanh(max(0.1, drones[0].airframe.p[2]) / 4.5))
            episode_wind = np.array([-5.0, 4.2, downburst_z], dtype=np.float32)
            ice_active = False
            curr_ice = 0.0
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

            dist_tgt_xy = float(np.linalg.norm(d.airframe.p[:2] - tgt_xy))
            dist_home_xy = float(np.linalg.norm(d.airframe.p[:2] - home_xy))

            if d.mission_phase == "1_TAKEOFF":
                wp_target = np.array([home_xy[0], home_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.0, max_climb=2.40)
                fsm_token = 0
                winch_rate_cmd = 0.0
                if d.airframe.p[2] >= (cruise_z - 2.0):
                    d.mission_phase = "2_CRUISE_OUTBOUND"

            elif d.mission_phase == "2_CRUISE_OUTBOUND":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=8.0, max_climb=2.40)
                fsm_token = 0
                winch_rate_cmd = 0.0
                if dist_tgt_xy <= 2.5:
                    d.mission_phase = "3_CHIMNEY_DESCENT"

            elif d.mission_phase == "3_CHIMNEY_DESCENT":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], high_hover_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=1.2, max_descend=0.65, wind_vector=episode_wind)
                fsm_token = 0
                winch_rate_cmd = 0.0
                if abs(d.airframe.p[2] - high_hover_z) <= 1.8 and dist_tgt_xy <= 3.2:
                    d.mission_phase = "4_DELIVERY_HOVER"
                    d.hover_settle_timer = 0.0

            elif d.mission_phase == "4_DELIVERY_HOVER":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], high_hover_z], dtype=np.float32)
                p_ref, v_ref = wp_target.copy(), np.zeros(3, dtype=np.float32)
                d.hover_settle_timer += dt

                cable_stowed = (d.winch.cable_len <= 0.10)
                if d.winch.has_payload and cable_stowed:
                    fsm_token = 1
                    winch_rate_cmd = 0.0
                elif d.winch.has_payload and not cable_stowed:
                    cargo_z = d.airframe.p[2] - d.winch.cable_len
                    if cargo_z > (roof_h + 0.18):
                        fsm_token = 2
                        winch_rate_cmd = 1.0
                    else:
                        fsm_token = 3
                        winch_rate_cmd = 0.0
                elif not d.winch.has_payload and not cable_stowed:
                    fsm_token = 4
                    winch_rate_cmd = -1.0
                else:
                    fsm_token = 0
                    winch_rate_cmd = 0.0

                if d.hover_settle_timer >= 8.0 and not d.winch.delivery_completed:
                    d.winch.has_payload = False
                    d.winch.state = "RETRACTING"

                if (d.winch.delivery_completed and cable_stowed) or (d.hover_settle_timer >= 12.0 and cable_stowed):
                    d.winch.delivery_completed = False
                    d.hover_settle_timer = 0.0
                    d.packages_delivered += 1
                    d.dispatcher.report_completed(d.drone_id)
                    d.mission_phase = "5_CHIMNEY_ASCENT"
                    d.target = d.mission["home_pad"].copy()

            elif d.mission_phase == "5_CHIMNEY_ASCENT":
                wp_target = np.array([d.airframe.p[0], d.airframe.p[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.0, max_climb=2.40, wind_vector=episode_wind)
                fsm_token = 0
                winch_rate_cmd = 0.0
                if d.airframe.p[2] >= (cruise_z - 2.0):
                    d.mission_phase = "6_CRUISE_INBOUND"

            elif d.mission_phase == "6_CRUISE_INBOUND":
                wp_target = np.array([home_xy[0], home_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=8.0, max_climb=2.40, wind_vector=episode_wind)
                fsm_token = 0
                winch_rate_cmd = 0.0
                
                wind_spd = float(np.linalg.norm(episode_wind[:2]))
                inbound_gate = min(4.5, 2.2 + 0.20 * wind_spd)
                if dist_home_xy <= inbound_gate:
                    d.mission_phase = "7_DEPOT_DESCENT"

            elif d.mission_phase == "7_DEPOT_DESCENT":
                wp_target = np.array([home_xy[0], home_xy[1], 0.02], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.35, max_descend=0.22, wind_vector=episode_wind)
                fsm_token = 5
                winch_rate_cmd = 0.0

                is_on_helipad = (dist_home_xy <= 1.5)
                has_touched_down = (d.airframe.p[2] <= 0.08 or d.winch.is_skid_grounded)
                if is_on_helipad and has_touched_down:
                    d.depot_timer = 0.0
                    d.mission_phase = "8_DEPOT_SERVICE"
                    d.airframe.v.fill(0.0)

            else:
                wp_target = np.array([home_xy[0], home_xy[1], 0.06], dtype=np.float32)
                p_ref, v_ref = wp_target.copy(), np.zeros(3, dtype=np.float32)
                fsm_token = 5
                winch_rate_cmd = 0.0
                d.airframe.v.fill(0.0)

                d.depot_timer += dt
                if d.depot_timer >= 1.0:
                    d.depot_timer = 0.0
                    d.mission = d.dispatcher.dispatch_next_mission(d.drone_id, d.bms.soc, d.airframe.p)
                    d.home_coords = d.mission["home_pad"].copy()
                    d.target = d.mission["target_pos"].copy()
                    d.assigned_cruise_alt = d.mission["assigned_cruise_alt"]
                    d.winch.reload_new_cargo()
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
            flow_2d, flow_q, _ = sensor_suite.poll_optical_flow(d.airframe.p, d.airframe.v, d.airframe.R, d.airframe.omega, current_surf_z)
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

            obs_139 = d.synth.synthesize_instantaneous(
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

            obs_417 = d.synth.stack_history(d.history_q, obs_139)
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
            current_dist = dist_home_xy if ("INBOUND" in d.mission_phase or "DEPOT" in d.mission_phase) else dist_tgt_xy
            step_yv_triad.append(np.array([np.clip(-current_dist / 30.0, -1.0, 1.0)], dtype=np.float32))

            d.winch.step(
                drone_z=d.airframe.p[2], target_surface_z=roof_h, is_settled=(d.mission_phase == "4_DELIVERY_HOVER"),
                is_surface_safe=is_surface_safe, wire_hit=False, agent_breach=False, agent_dist=30.0,
                wind_2d=episode_wind[:2], drone_accel_linear=d.airframe.accel_linear,
                neural_token_idx=fsm_token, nn_winch_rate=winch_rate_cmd, dt=dt,
                drone_p=d.airframe.p, drone_v=d.airframe.v, wind_3d=episode_wind
            )

            omega_cmds, _, _ = d.mixer.mix(
                drone_health=d.airframe.motor_health, p_est=d.airframe.p, v_est=d.airframe.v, R_est=d.airframe.R,
                omega_est=d.airframe.omega, p_target=p_ref, neural_guidance=guidance_vec, token_idx=fsm_token,
                estimated_wind=episode_wind, obstacles=obstacles_cbf, swarm_states=swarm_states, drone_idx=d_idx,
                ref_vel=v_ref, target_bldg_id=d.mission["bldg_id"] if ("OUTBOUND" in d.mission_phase or "DELIVER" in d.mission_phase or "CHIMNEY" in d.mission_phase) else None,
                departure_bldg_id=d.mission["bldg_id"] if ("ASCENT" in d.mission_phase or ("INBOUND" in d.mission_phase and dist_tgt_xy < 25.0)) else None,
                cable_len=d.winch.cable_len, sway_2d=d.winch.sway_offset, load_cell_3d=load_3d,
                ige_fac=d.airframe.last_ige, v_pack=d.bms.v_pack, bms_temp_c=d.bms.temp_c,
                has_payload=d.winch.has_payload, dt=dt,
                f_skid_left=d.winch.f_skid_left, f_skid_right=d.winch.f_skid_right,
                ekf_pos_cov_trace=0.05
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
    NUM_SAMPLES = 250000              
    NUM_PRETRAIN_EPOCHS = 20         
    NUM_MAIN_EPOCHS = 40             
    BATCH_SIZE = 4104
    LEARNING_RATE = 1.2e-3
    
    num_workers = min(14, max(1, mp.cpu_count() - 2))
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

    num_train = int(len(X_all) * 0.90)

    X_train = torch.from_numpy(X_all[:num_train]).to(device)
    yg_tr = torch.from_numpy(yg_all[:num_train]).to(device)
    yf_tr = torch.as_tensor(yf_all[:num_train], dtype=torch.long, device=device)
    yw_tr = torch.from_numpy(yw_all[:num_train]).to(device)
    yh_tr = torch.from_numpy(yh_all[:num_train]).to(device)
    yt_tr = torch.from_numpy(yt_all[:num_train]).to(device)
    yv_tr = torch.from_numpy(yv_all[:num_train]).to(device)

    X_val = torch.from_numpy(X_all[num_train:]).to(device)
    yg_va = torch.from_numpy(yg_all[num_train:]).to(device)
    yf_va = torch.as_tensor(yf_all[num_train:], dtype=torch.long, device=device)

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
        perm = torch.randperm(num_train_samples, device=device)
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
        perm = torch.randperm(num_train_samples, device=device)
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

            c_idx = (HISTORY_LEN - 1) * BASE_OBS_DIM
            v_drone = bx[:, (c_idx + 6):(c_idx + 9)] * 8.0
            wind_est = bx[:, (c_idx + 42):(c_idx + 45)] * 8.0
            v_rel = v_drone - wind_est
            speed_rel = torch.norm(v_rel, dim=-1, keepdim=True)

            a_drag_physics = -0.5 * 1.225 * 0.28 * v_rel * torch.clamp(speed_rel, min=1.0) / 1.80
            physics_target_accel = byg[:, :3] - (a_drag_physics / 12.0)
            loss_physics = crit_mse(p_g[:, :3], physics_target_accel)

            loss = (0.70 * crit_mse(p_g, byg) +
                    0.30 * loss_physics +
                    0.35 * crit_ce(p_f, byf) +
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

# =============================================================================
# 16. CHASE & GIMBAL PROJECTIONS, PFD, AND MINIMAP
# =============================================================================
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

def project_belly_gimbal(pt, drone_p, drone_R, center_screen, scale=32.0):
    rel = pt - drone_p
    rel_b = drone_R.T @ rel
    z_down = -rel_b[2]
    if z_down <= 0.2:
        return None, None, z_down
    focal = max(2.0, z_down)
    sx = int(center_screen[0] + (rel_b[1] / focal) * scale * 9.5)
    sy = int(center_screen[1] + (rel_b[0] / focal) * scale * 9.5)
    return sx, sy, z_down

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
# 17. DISTRIBUTED CBBA AUCTION & BREGUET ENERGY ROUTING DISPATCHER
# =============================================================================
class SwarmMissionDispatcher:
    def __init__(self, manifest_addresses, depot_helipads):
        self.manifest = list(manifest_addresses)
        self.depot_helipads = depot_helipads
        self.pending_orders = deque(range(len(self.manifest)))
        self.active_missions = {}
        self.total_completed = 0

    def dispatch_next_mission(self, drone_id, drone_soc=0.98, drone_pos=None, microclimate_map=None, sim_time=0.0):
        if not self.pending_orders:
            self.pending_orders.extend(range(len(self.manifest)))

        home_pad = self.depot_helipads[drone_id % len(self.depot_helipads)]
        current_p = home_pad if drone_pos is None else drone_pos

        best_queue_idx = 0
        best_cost = 1e9

        for q_idx, order_idx in enumerate(self.pending_orders):
            candidate = self.manifest[order_idx]
            d_outbound = float(np.linalg.norm(current_p[:2] - candidate["pos"][:2]))
            d_return = float(np.linalg.norm(candidate["pos"][:2] - home_pad[:2]))
            total_flight_dist = d_outbound + d_return

            battery_penalty = 1.0 + (1.0 - drone_soc) * 2.2
            cost = total_flight_dist * battery_penalty

            if cost < best_cost:
                best_cost = cost
                best_queue_idx = q_idx

        chosen_order_idx = self.pending_orders[best_queue_idx]
        del self.pending_orders[best_queue_idx]

        order = self.manifest[chosen_order_idx]

        # Breguet Wind Layer Routing: Select optimal energy cruise altitude
        candidate_alts = [32.0, 38.0, 44.0, 50.0, 56.0]
        assigned_cruise_alt = 42.0 + (drone_id * 3.0)

        if microclimate_map is not None:
            track_dir = candidate["pos"][:2] - current_p[:2]
            track_dist = max(1.0, float(np.linalg.norm(track_dir)))
            u_track = track_dir / track_dist

            wind_profile = microclimate_map.query_vertical_wind_shear_profile(current_p[:2], sim_time, candidate_alts)
            best_tail_val = -999.0
            best_alt = assigned_cruise_alt
            for idx, alt in enumerate(candidate_alts):
                if alt > float(order["height"]) + 8.0:
                    tail_comp = float(np.dot(wind_profile[idx][:2], u_track))
                    if tail_comp > best_tail_val:
                        best_tail_val = tail_comp
                        best_alt = alt
            assigned_cruise_alt = best_alt

        mission = {
            "order_idx": chosen_order_idx,
            "target_pos": order["pos"].copy(),
            "bldg_id": order["bldg_id"],
            "addr": order["addr"],
            "tag_id": order["tag_id"],
            "home_pad": home_pad.copy(),
            "roof_height": float(order["height"]),
            "assigned_cruise_alt": assigned_cruise_alt,
            "drone_id": drone_id
        }
        self.active_missions[drone_id] = mission
        return mission

    def re_auction_task(self, abandoned_mission, candidate_drones):
        best_drone_id = None
        best_bid = -1e9
        for d in candidate_drones:
            if d.drone_id == abandoned_mission.get("drone_id", -1) or d.rth_active:
                continue
            d_to_task = float(np.linalg.norm(d.airframe.p[:2] - abandoned_mission["target_pos"][:2]))
            health_score = float(np.mean(d.airframe.motor_health))
            bid = (d.bms.soc * health_score * 100.0) / max(10.0, d_to_task)
            if bid > best_bid:
                best_bid = bid
                best_drone_id = d.drone_id
        return best_drone_id

    def report_completed(self, drone_id):
        self.total_completed += 1

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

        self.mission_phase = "1_TAKEOFF"
        self.trail = deque(maxlen=160)
        self.packages_delivered = 0
        self.hover_settle_timer = 0.0
        self.depot_timer = 0.0
        self.rth_active = False

    def get_high_hover_z(self):
        return self.mission["roof_height"] + 10.0

    def evaluate_dynamic_rth_threshold(self, live_wind):
        dist_home = float(np.linalg.norm(self.airframe.p[:2] - self.home_coords[:2]))
        if dist_home < 4.0:
            return False

        dir_to_home = (self.home_coords[:2] - self.airframe.p[:2]) / max(0.1, dist_home)
        headwind_to_home = -float(np.dot(live_wind[:2], dir_to_home))
        
        v_ground_est = max(2.5, 7.5 - headwind_to_home)
        time_needed_s = (dist_home / v_ground_est) + 16.0

        power_needed_watts = self.airframe.last_p_watts * 1.15
        energy_joules_needed = time_needed_s * power_needed_watts
        soc_needed = energy_joules_needed / (PACK_CAPACITY_AH * 3600.0 * max(18.0, self.bms.v_pack))

        dynamic_abort_limit = float(np.clip(soc_needed + 0.08, 0.15, 0.58))
        return self.bms.soc <= dynamic_abort_limit

# =============================================================================
# 18. DIRECT 3D VECTOR GUIDANCE & LIVE REALITY SIMULATOR COCKPIT
# =============================================================================
class QuinticMinimumJerkTrajectory:
    def __init__(self, p0, v0, a0, pf, vf=None, af=None, duration=4.0):
        self.p0 = np.array(p0, dtype=np.float32)
        self.v0 = np.array(v0, dtype=np.float32)
        self.a0 = np.array(a0, dtype=np.float32)
        self.pf = np.array(pf, dtype=np.float32)
        self.vf = np.zeros(3, dtype=np.float32) if vf is None else np.array(vf, dtype=np.float32)
        self.af = np.zeros(3, dtype=np.float32) if af is None else np.array(af, dtype=np.float32)
        self.T = max(0.5, float(duration))

        T = self.T
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        delta_p = self.pf - self.p0 - self.v0 * T - 0.5 * self.a0 * T2
        delta_v = self.vf - self.v0 - self.a0 * T
        delta_a = self.af - self.a0

        self.c0 = self.p0
        self.c1 = self.v0
        self.c2 = 0.5 * self.a0
        self.c3 = (10.0 * delta_p) / T3 - (4.0 * delta_v) / T2 + (0.5 * delta_a) / T
        self.c4 = (-15.0 * delta_p) / T4 + (7.0 * delta_v) / T3 - (1.0 * delta_a) / T2
        self.c5 = (6.0 * delta_p) / T5 - (3.0 * delta_v) / T4 + (0.5 * delta_a) / T3

    def evaluate(self, t):
        t_clamped = float(np.clip(t, 0.0, self.T))
        t2 = t_clamped * t_clamped
        t3 = t2 * t_clamped
        t4 = t3 * t_clamped
        t5 = t4 * t_clamped

        p = self.c0 + self.c1 * t_clamped + self.c2 * t2 + self.c3 * t3 + self.c4 * t4 + self.c5 * t5
        v = self.c1 + 2.0 * self.c2 * t_clamped + 3.0 * self.c3 * t2 + 4.0 * self.c4 * t3 + 5.0 * self.c5 * t4
        a = 2.0 * self.c2 + 6.0 * self.c3 * t_clamped + 12.0 * self.c4 * t2 + 20.0 * self.c5 * t3
        return p.astype(np.float32), v.astype(np.float32), a.astype(np.float32)

def compute_vector_guidance(current_p, current_v, target_p, max_speed_xy=8.0, max_climb=2.40, 
                            max_descend=0.65, wind_speed=0.0, wind_vector=None, **kwargs):
    delta_p = target_p - current_p
    dist_xy = float(math.hypot(delta_p[0], delta_p[1]))
    dist_z = float(delta_p[2])
    
    if dist_xy > 0.05 and wind_vector is not None:
        track_unit = delta_p[:2] / dist_xy
        headwind = -float(np.dot(wind_vector[:2], track_unit))
        adaptive_cruise_speed = (max_speed_xy + 0.40 * headwind) if headwind > 0 else max(4.5, max_speed_xy + 0.22 * headwind)
    else:
        adaptive_cruise_speed = max_speed_xy

    if max_speed_xy <= 0.05:
        v_cmd_xy = np.zeros(2, dtype=np.float32)
    else:
        adaptive_max_xy = float(np.clip(adaptive_cruise_speed, 2.0, max(2.0, max_speed_xy)))
        v_xy_mag = min(adaptive_max_xy, math.sqrt(2.0 * 1.5 * dist_xy)) if dist_xy > 1.8 else 1.2 * dist_xy
        v_cmd_xy = (delta_p[:2] / dist_xy) * v_xy_mag if dist_xy > 0.03 else np.zeros(2, dtype=np.float32)

    if target_p[2] <= 0.15:
        z_clearance = max(0.01, float(current_p[2] - target_p[2]))
        v_cmd_z = -float(np.clip(0.35 * math.sqrt(z_clearance), 0.06, max_descend))
    else:
        if dist_z > 0.0:
            if current_p[2] < 2.5:
                climb_ramp_ceiling = float(np.clip(0.65 + 0.70 * current_p[2], 0.65, max_climb))
            else:
                climb_ramp_ceiling = max_climb
            v_cmd_z = min(climb_ramp_ceiling, math.sqrt(max(0.0, 2.0 * 1.0 * dist_z)))
        else:
            v_cmd_z = -min(max_descend, math.sqrt(max(0.0, 2.0 * 0.50 * abs(dist_z))))
        
    v_des = np.array([v_cmd_xy[0], v_cmd_xy[1], v_cmd_z], dtype=np.float32)
    return target_p.copy(), v_des

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
    pygame.display.set_caption("Survival Core V37.0 — Adaptive Autonomous Drone Swarm Cockpit")
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
    cam_mode = "CHASE"
    running = True

    gust_timer = 0.0
    active_gust = np.zeros(3, dtype=np.float32)
    inject_spoof = False
    flight_time = 0.0
    delivered_boxes = deque(maxlen=120)

    wind_particles = [
        np.array([random.uniform(-180.0, 180.0), random.uniform(-180.0, 180.0), random.uniform(1.0, 35.0)], dtype=np.float32)
        for _ in range(55)
    ]

    print("\n================ V37.0 ADAPTIVE AUTONOMOUS DRONE SWARM ONLINE ================")
    print(" [1. GEOMETRIC SE(3)] : Almost-Global Exponential Lie-Group Attitude Tracking")
    print(" [2. NEURAL ARBITER] : Active Hierarchical FSM & Fault Override Mechanism")
    print(" [3. CBBA AUCTION]   : Dynamic Distributed Consensus Task Re-Auctioning")
    print(" [4. SPATIAL HASH]   : O(1) Accelerated 360-LiDAR Raycasting Buckets")
    print(" [5. TETHER VANES]   : Aerodynamic Stabilizer Fins & Soft Ground Bumper")
    print(" [6. BREGUET ROUTING]: Wind-Shear Energy-Optimal Cruise Altitude Selection")
    print(" [7. COCKPIT PFD]    : Primary Flight Display (ADI) & Multi-Camera Views")
    print("===============================================================================\n")

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
                    cam_modes = ["CHASE", "BELLY_GIMBAL", "DEPOT_OVERVIEW"]
                    cam_mode = cam_modes[(cam_modes.index(cam_mode) + 1) % len(cam_modes)]
                    print(f">>> [CAMERA PERSPECTIVE]: {cam_mode} <<<")
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
                    cur_d.hover_settle_timer = 0.0
                    cur_d.winch.state = "STANDBY"
                    cur_d.winch.has_payload = False
                    cur_d.winch.cable_len = 0.08
                    print(f"\n>>> [MANUAL OVERRIDE] Drone #{cur_d.drone_id+1} Returning to Base! <<<")
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
                        d.mission_phase = "1_TAKEOFF"
                    inject_spoof = False

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

        if cam_mode == "CHASE":
            cam_focus += (drones[selected_drone_idx].airframe.p - cam_focus) * 5.0 * dt
        elif cam_mode == "BELLY_GIMBAL":
            cam_focus = drones[selected_drone_idx].airframe.p.copy()
        else: # DEPOT_OVERVIEW
            cam_focus += (np.array([-150.0, -155.0, 4.0], dtype=np.float32) - cam_focus) * 5.0 * dt

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
        drone_step_data = []
        wind_speed_xy = float(np.linalg.norm(total_wind[:2]))

        for d_idx, d in enumerate(drones):
            tgt_xy = d.mission["target_pos"][:2]
            home_xy = d.mission["home_pad"][:2]
            roof_h = d.mission["roof_height"]
            cruise_z = d.assigned_cruise_alt
            
            storm_standoff = min(3.5, 0.35 * wind_speed_xy)
            high_hover_z = roof_h + 10.0 + storm_standoff

            dist_tgt_xy = float(math.hypot(d.airframe.p[0] - tgt_xy[0], d.airframe.p[1] - tgt_xy[1]))
            dist_home_xy = float(math.hypot(d.airframe.p[0] - home_xy[0], d.airframe.p[1] - home_xy[1]))

            gate_tol_xy = min(5.5, 3.2 + 0.25 * wind_speed_xy)
            gate_tol_z = min(2.5, 1.4 + 0.12 * wind_speed_xy)
            is_solidly_over_roof = False

            if not d.rth_active and d.mission_phase in ["1_TAKEOFF", "2_CRUISE_OUTBOUND", "3_CHIMNEY_DESCENT", "4_DELIVERY_HOVER"]:
                if d.evaluate_dynamic_rth_threshold(total_wind):
                    d.rth_active = True
                    d.mission_phase = "5_CHIMNEY_ASCENT"
                    d.target = d.mission["home_pad"].copy()
                    reassigned_id = dispatcher.re_auction_task(d.mission, drones)
                    if reassigned_id is not None:
                        drones[reassigned_id].mission = d.mission.copy()
                        drones[reassigned_id].target = d.mission["target_pos"].copy()
                        print(f"\n>>> [CBBA AUCTION] Task {d.mission['bldg_id']} dynamically reassigned to Drone #{reassigned_id+1}! <<<")

            if d.mission_phase == "1_TAKEOFF":
                wp_target = np.array([home_xy[0], home_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.0, max_climb=2.40, wind_vector=total_wind)
                if d.airframe.p[2] >= (cruise_z - 2.0):
                    d.mission_phase = "2_CRUISE_OUTBOUND"

            elif d.mission_phase == "2_CRUISE_OUTBOUND":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=9.5, max_climb=2.40, wind_vector=total_wind)
                if dist_tgt_xy <= gate_tol_xy:
                    d.mission_phase = "3_CHIMNEY_DESCENT"

            elif d.mission_phase == "3_CHIMNEY_DESCENT":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], high_hover_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=4.5, max_descend=0.85, wind_vector=total_wind)
                chimney_reach = 3.5
                dist_z = abs(d.airframe.p[2] - high_hover_z)
                if dist_z <= (gate_tol_z + 0.5) and dist_tgt_xy <= chimney_reach:
                    d.mission_phase = "4_DELIVERY_HOVER"
                    d.hover_settle_timer = 0.0

            elif d.mission_phase == "4_DELIVERY_HOVER":
                sway_trim = np.clip(d.winch.sway_offset, -2.5, 2.5) if d.winch.cable_len > 0.12 else np.zeros(2, dtype=np.float32)
                wp_target = np.array([tgt_xy[0] - sway_trim[0], tgt_xy[1] - sway_trim[1], high_hover_z], dtype=np.float32)
                p_ref, v_ref = wp_target.copy(), np.zeros(3, dtype=np.float32)

                payload_xy = d.airframe.p[:2] + d.winch.sway_offset
                dist_payload_to_target = float(math.hypot(payload_xy[0] - tgt_xy[0], payload_xy[1] - tgt_xy[1]))
                drone_spd_xy = float(math.hypot(d.airframe.v[0], d.airframe.v[1]))

                surf_under_drone = world.get_surface_height(d.airframe.p[0], d.airframe.p[1])
                surf_under_payload = world.get_surface_height(payload_xy[0], payload_xy[1])
                is_solidly_over_roof = (surf_under_drone >= roof_h - 1.2) and (surf_under_payload >= roof_h - 1.2)

                is_aligned_for_drop = (dist_payload_to_target <= 2.2) and (drone_spd_xy <= 1.1) and is_solidly_over_roof

                if is_aligned_for_drop or (d.winch.state in ["LOWERING", "TOUCHDOWN"]):
                    d.hover_settle_timer += dt
                else:
                    d.hover_settle_timer = max(0.0, d.hover_settle_timer - dt * 0.5)

                cable_stowed = (d.winch.cable_len <= 0.12)
                if d.hover_settle_timer >= 14.0 and not d.winch.delivery_completed:
                    d.winch.has_payload = False
                    d.winch.state = "RETRACTING"

                if (d.winch.delivery_completed and cable_stowed) or (d.hover_settle_timer >= 18.0 and cable_stowed):
                    d.winch.delivery_completed = False
                    d.hover_settle_timer = 0.0
                    d.packages_delivered += 1
                    delivered_boxes.append(np.array([tgt_xy[0], tgt_xy[1], roof_h], dtype=np.float32))
                    d.dispatcher.report_completed(d.drone_id)
                    d.mission_phase = "5_CHIMNEY_ASCENT"
                    d.target = d.mission["home_pad"].copy()

            elif d.mission_phase == "5_CHIMNEY_ASCENT":
                wp_target = np.array([d.airframe.p[0], d.airframe.p[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.0, max_climb=2.40, wind_vector=total_wind)
                if d.airframe.p[2] >= (cruise_z - 2.0):
                    d.mission_phase = "6_CRUISE_INBOUND"

            elif d.mission_phase == "6_CRUISE_INBOUND":
                wp_target = np.array([home_xy[0], home_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=9.5, max_climb=2.40, wind_vector=total_wind)
                if dist_home_xy <= 3.5:
                    d.mission_phase = "7_DEPOT_DESCENT"

            elif d.mission_phase == "7_DEPOT_DESCENT":
                wp_target = np.array([home_xy[0], home_xy[1], 0.02], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.6, max_descend=0.45, wind_vector=total_wind)
                is_on_helipad = (dist_home_xy <= 1.8)
                if is_on_helipad and (d.airframe.p[2] <= 0.08 or d.winch.is_skid_grounded):
                    d.depot_timer = 0.0
                    d.mission_phase = "8_DEPOT_SERVICE"
                    d.airframe.v.fill(0.0)
                    d.airframe.accel_linear.fill(0.0)

            else:
                wp_target = np.array([home_xy[0], home_xy[1], 0.06], dtype=np.float32)
                p_ref, v_ref = wp_target.copy(), np.zeros(3, dtype=np.float32)
                d.airframe.v.fill(0.0)
                d.airframe.accel_linear.fill(0.0)

                d.depot_timer += dt
                if d.depot_timer >= 1.0:
                    d.depot_timer = 0.0
                    d.rth_active = False
                    d.mission = d.dispatcher.dispatch_next_mission(
                        d.drone_id, d.bms.soc, d.airframe.p, 
                        microclimate_map=world.shared_microclimate_map, sim_time=flight_time
                    )
                    d.home_coords = d.mission["home_pad"].copy()
                    d.target = d.mission["target_pos"].copy()
                    d.assigned_cruise_alt = d.mission["assigned_cruise_alt"]
                    d.winch.reload_new_cargo()
                    d.icing.ice_thickness_mm = 0.0
                    d.bms.soc = 0.98
                    d.bms.v_pack = 25.0
                    d.airframe.p[2] = 0.06
                    d.mission_phase = "1_TAKEOFF"

            current_surf_z = world.get_surface_height(d.airframe.p[0], d.airframe.p[1])
            target_surface_z = float(d.mission["roof_height"] if ("DELIVER" in d.mission_phase or "CHIMNEY" in d.mission_phase) else 0.0)
            surf_class, is_surf_safe, _ = assurance.audit_surface(d.target[:2], target_z=target_surface_z)
            s_data = sensors_physics.poll_all_physics_sensors(d.airframe.p, d.airframe.v, total_wind, surface_z=current_surf_z)

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

            bms_soc, v_pack, i_draw, r0_dyn = d.bms.step(d.airframe.last_p_watts, amb_temp=amb_temp, dt=dt)
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
            alpha, beta, rh, dust = sensor_suite.poll_5port_air_data(d.airframe.v, total_wind, d.airframe.R)

            real_peer_info = []
            for j_idx, other_d in enumerate(drones):
                if j_idx == d_idx: continue
                p_rel = other_d.airframe.p - d.airframe.p
                v_rel = other_d.airframe.v - d.airframe.v
                dist_p = float(np.linalg.norm(p_rel))
                real_peer_info.append({"p_rel": p_rel, "v_rel": v_rel, "dist": dist_p, "has_payload": other_d.winch.has_payload})

            obs_139 = d.synth.synthesize_instantaneous(
                p_est=d.ekf.p, v_est=d.ekf.v, R_est=R_est, omega_est=omega_est,
                target_p=wp_target, lidar_16=lidar_16, conical_8=conical_8,
                v_tas=s_data["v_tas"], q_dyn=s_data["q_dyn"], min_sonar=s_data["min_sonar"],
                wind_est=total_wind, ice_idx=d.icing.ice_thickness_mm / 3.0,
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

            with torch.no_grad():
                obs_t = torch.from_numpy(obs_417).unsqueeze(0).to(device)
                t0_inf = time.time()
                p_guidance, p_fsm, p_winch, p_health, p_trust, _ = model(obs_t)
                inf_latency_ms = (time.time() - t0_inf) * 1000.0

                nn_guidance = p_guidance[0].cpu().numpy() * 0.15
                nn_fsm_token = int(p_fsm[0].argmax().item())
                nn_winch_rate = float(p_winch[0, 0].item())
                nn_conf = float(F.softmax(p_fsm[0], dim=-1).max().item())

            # Winch Hysteresis & Target Verification
            is_delivering = (d.mission_phase == "4_DELIVERY_HOVER")
            if d.winch.state == "STANDBY":
                is_winch_settled = is_delivering and (d.hover_settle_timer >= 0.50) and is_aligned_for_drop
            else:
                is_winch_settled = is_delivering and is_solidly_over_roof and (dist_payload_to_target <= 5.5)

            d.winch.step(
                drone_z=d.airframe.p[2], target_surface_z=roof_h, 
                is_settled=is_winch_settled,
                is_surface_safe=(is_surf_safe and is_solidly_over_roof), 
                wire_hit=False, agent_breach=global_agent_breach,
                agent_dist=float(np.linalg.norm(agent_pos[:2] - d.airframe.p[:2])),
                wind_2d=total_wind[:2], drone_accel_linear=d.airframe.accel_linear,
                neural_token_idx=0, nn_winch_rate=0.0, dt=dt,
                drone_p=d.airframe.p, drone_v=d.airframe.v, wind_3d=total_wind
            )

            ekf_pos_cov_trace = float(np.trace(d.ekf.P[0:3, 0:3]))

            drone_step_data.append({
                "p_ref": p_ref, "v_ref": v_ref, "R_est": R_est, "omega_est": omega_est,
                "load_3d": load_3d, "v_pack": v_pack, "bms_soc": bms_soc,
                "current_surf_z": current_surf_z, "meas_accel": meas_accel, "meas_gyro": meas_gyro,
                "s_data": s_data, "flow_2d": flow_2d, "is_surf_safe": is_surf_safe,
                "conical_8": conical_8, "i_rms": i_rms, "t_mosfet": t_mosfet,
                "nn_guidance": nn_guidance, "nn_fsm_token": nn_fsm_token, "nn_conf": nn_conf,
                "inf_latency_ms": inf_latency_ms, "adsb_d": adsb_d, "siren_s": siren_s,
                "phase_shift": phase_shift, "agent_dist": float(np.linalg.norm(agent_pos[:2] - d.airframe.p[:2])),
                "ekf_pos_cov_trace": ekf_pos_cov_trace
            })

        for d_idx, d in enumerate(drones):
            gd = drone_step_data[d_idx]

            world.shared_microclimate_map.broadcast_wind(d.airframe.p, total_wind, flight_time)
            wind_prior, has_prior = world.shared_microclimate_map.query_wind_prior(gd["p_ref"], flight_time)
            effective_wind_feed = (0.7 * total_wind + 0.3 * wind_prior) if has_prior else total_wind

            omega_cmds, f_thrust, a_safe = d.mixer.mix(
                drone_health=d.airframe.motor_health, p_est=d.ekf.p, v_est=d.ekf.v, R_est=gd["R_est"], omega_est=gd["omega_est"],
                p_target=gd["p_ref"], neural_guidance=gd["nn_guidance"], token_idx=gd["nn_fsm_token"],
                estimated_wind=effective_wind_feed, obstacles=obstacles_cbf, swarm_states=swarm_states, drone_idx=d_idx,
                ref_accel=np.zeros(3, dtype=np.float32), ref_vel=gd["v_ref"], 
                target_bldg_id=d.mission["bldg_id"] if ("OUTBOUND" in d.mission_phase or "DELIVER" in d.mission_phase or "CHIMNEY" in d.mission_phase) else None,
                departure_bldg_id=d.mission["bldg_id"] if ("ASCENT" in d.mission_phase or ("INBOUND" in d.mission_phase and dist_tgt_xy < 25.0)) else None,
                cable_len=d.winch.cable_len, sway_2d=d.winch.sway_offset, load_cell_3d=gd["load_3d"], 
                ige_fac=d.airframe.last_ige, v_pack=gd["v_pack"], bms_temp_c=d.bms.temp_c, 
                has_payload=d.winch.has_payload, ice_mm=d.icing.ice_thickness_mm, dt=dt,
                conical_ranges_8=gd["conical_8"], esc_i_rms=gd["i_rms"], esc_t_mosfet=gd["t_mosfet"],
                f_skid_left=d.winch.f_skid_left, f_skid_right=d.winch.f_skid_right,
                ekf_pos_cov_trace=gd["ekf_pos_cov_trace"]
            )

            f_act, torques, p_watts, ige_fac = d.airframe.step(
                motor_cmds=omega_cmds, current_wind=total_wind, rho_air=gd["s_data"]["rho_air"],
                ice_index=d.icing.ice_thickness_mm / 3.0, surface_z=gd["current_surf_z"], has_payload=d.winch.has_payload, dt=dt
            )

            if int(flight_time * 60) % 6 == 0:
                d.trail.append(d.airframe.p[:2].copy())

            if d_idx == selected_drone_idx:
                r_r, r_p, r_y = rot_to_euler(gd["R_est"])
                selected_diag = {
                    "v_pack": gd["v_pack"], "soc": gd["bms_soc"], "temp_c": d.bms.temp_c,
                    "action": ACTION_NAMES[gd["nn_fsm_token"]], "v_crit": 0.0,
                    "thrust": f_thrust, "tension": d.winch.tension_n,
                    "cable_l": d.winch.cable_len, "load_3d": gd["load_3d"],
                    "ice_mm": d.icing.ice_thickness_mm, "wind_speed": np.linalg.norm(total_wind),
                    "confidence": gd["nn_conf"], "latency": gd["inf_latency_ms"],
                    "gnss_status": "GNSS REJECTED [SPOOF DETECTED]" if inject_spoof else "GNSS NOMINAL (RTK FIXED)",
                    "jerk": float(np.linalg.norm(gd["meas_accel"] - d.airframe.accel_linear)),
                    "phase": float(gd["phase_shift"][0]),
                    "i_rms": gd["i_rms"], "t_mosfet": gd["t_mosfet"],
                    "th_dt": 0.5, "dolp": 0.2, "ndvi": 0.3,
                    "hdop": 3.8 if inject_spoof else 0.85, "cn0": 28.5 if inject_spoof else 44.0,
                    "alpha": 0.0, "beta": 0.0, "dust": 20.0, "v_tas": gd["s_data"]["v_tas"],
                    "accel_raw": gd["meas_accel"], "gyro_raw": gd["meas_gyro"],
                    "flow_2d": gd["flow_2d"], "aruco_locked": 1.0 if "DELIVERY" in d.mission_phase else 0.0,
                    "is_grounded": d.winch.is_skid_grounded,
                    "adsb_d": gd["adsb_d"], "siren_s": gd["siren_s"],
                    "local_agent_dist": gd["agent_dist"], "rth_active": d.rth_active,
                    "roll": r_r, "pitch": r_p, "yaw": r_y, "alt": d.airframe.p[2],
                    "airspeed": float(np.linalg.norm(d.airframe.v)), "vspeed": d.airframe.v[2]
                }

        # ==========================================
        # RENDERING (3D CITY GRID & COCKPIT PFD)
        # ==========================================
        screen.fill((10, 13, 20))
        pygame.draw.line(screen, (35, 45, 60), (VIEW_W, 0), (VIEW_W, H), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (W, 70), 2)

        def view_proj(p_3d):
            if cam_mode == "BELLY_GIMBAL":
                return project_belly_gimbal(p_3d, active_drone.airframe.p, active_drone.airframe.R, center_viewport)
            return project_3d_chase(p_3d, cam_focus, cam_yaw, cam_pitch, center_viewport)

        for gx in range(-8, 9):
            p1 = np.array([cam_focus[0] + gx * 8.0, cam_focus[1] - 45.0, 0.0], dtype=np.float32)
            p2 = np.array([cam_focus[0] + gx * 8.0, cam_focus[1] + 45.0, 0.0], dtype=np.float32)
            sx1, sy1, _ = view_proj(p1)
            sx2, sy2, _ = view_proj(p2)
            if sx1 is not None and sx2 is not None:
                pygame.draw.line(screen, (18, 24, 32), (sx1, sy1), (sx2, sy2), 1)

        for r in world.roads:
            if math.hypot(cam_focus[0] - (r["x_min"]+r["x_max"])/2, cam_focus[1] - (r["y_min"]+r["y_max"])/2) < 85.0:
                rc_pts = [
                    np.array([r["x_min"], r["y_min"], 0.0]), np.array([r["x_max"], r["y_min"], 0.0]),
                    np.array([r["x_max"], r["y_max"], 0.0]), np.array([r["x_min"], r["y_max"], 0.0])
                ]
                r_screens = [view_proj(p)[:2] for p in rc_pts]
                if all(pt[0] is not None for pt in r_screens):
                    pygame.draw.polygon(screen, (24, 30, 38), r_screens)

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
                h_screens = [view_proj(p)[:2] for p in h_corners]
                
                if all(h_screens[i][0] is not None for i in range(4, 8)):
                    roof_poly = [h_screens[4], h_screens[5], h_screens[6], h_screens[7]]
                    pygame.draw.polygon(screen, (38, 46, 58), roof_poly)
                    pygame.draw.lines(screen, (100, 120, 145), True, roof_poly, 2)
                    
                    roof_center = np.array([h_center_xy[0], h_center_xy[1], hh], dtype=np.float32)
                    rf_sx, rf_sy, _ = view_proj(roof_center)
                    if rf_sx is not None:
                        pygame.draw.circle(screen, (0, 220, 255), (rf_sx, rf_sy), 12, 1)
                        screen.blit(font_small.render("H", True, (0, 220, 255)), (rf_sx - 3, rf_sy - 6))

        for v in traffic_engine.vehicles:
            if math.hypot(cam_focus[0] - v["pos"][0], cam_focus[1] - v["pos"][1]) < 60.0:
                vx_s, vy_s, _ = view_proj(v["pos"])
                if vx_s is not None:
                    col = (255, 45, 65) if v.get("is_emergency", False) else (255, 140, 40)
                    pygame.draw.rect(screen, col, (vx_s - 6, vy_s - 4, 12, 8), border_radius=2)

        cv_pos = world.civil_aircraft["pos"]
        if math.hypot(cam_focus[0] - cv_pos[0], cam_focus[1] - cv_pos[1]) < 90.0:
            cx_s, cy_s, _ = view_proj(cv_pos)
            if cx_s is not None:
                pygame.draw.circle(screen, (255, 220, 40), (cx_s, cy_s), 7, 2)

        for p_i, pad in enumerate(world.depot_helipads):
            if math.hypot(cam_focus[0] - pad[0], cam_focus[1] - pad[1]) < 70.0:
                hc_sx, hc_sy, _ = view_proj(pad)
                if hc_sx is not None:
                    pad_color = swarm_colors[p_i % len(swarm_colors)]
                    pygame.draw.circle(screen, pad_color, (hc_sx, hc_sy), 16, 2)
                    screen.blit(font_large.render(f"H{p_i+1}", True, pad_color), (hc_sx - 10, hc_sy - 9))

        if math.hypot(cam_focus[0] - agent_pos[0], cam_focus[1] - agent_pos[1]) < 45.0:
            ag_sx, ag_sy, _ = view_proj(agent_pos)
            if ag_sx is not None:
                ag_col = (100, 220, 120) if ground_agent.is_leashed else ((255, 60, 80) if global_agent_breach else (255, 120, 40))
                pygame.draw.circle(screen, ag_col, (ag_sx, ag_sy), 7)

        for pb in delivered_boxes:
            if math.hypot(cam_focus[0] - pb[0], cam_focus[1] - pb[1]) < 65.0:
                bx, by, _ = view_proj(pb)
                if bx is not None:
                    pygame.draw.rect(screen, (60, 160, 90), (bx - 5, by - 5, 10, 10), border_radius=2)

        for d_i, d in enumerate(drones):
            col = swarm_colors[d_i % len(swarm_colors)]
            dsx, dsy, _ = view_proj(d.airframe.p)
            if dsx is None: continue

            # Central hub
            pygame.draw.circle(screen, col, (dsx, dsy), 8 if d_i == selected_drone_idx else 5)
            screen.blit(font_small.render(f"D{d_i+1}", True, col), (dsx + 12, dsy - 12))

            # Cross-arms & 8 coaxial propellers
            arm_d = ARM_LENGTH_M * 1.8
            arm_ends_body = [
                np.array([ arm_d/1.414, -arm_d/1.414, 0.0], dtype=np.float32),
                np.array([ arm_d/1.414,  arm_d/1.414, 0.0], dtype=np.float32),
                np.array([-arm_d/1.414,  arm_d/1.414, 0.0], dtype=np.float32),
                np.array([-arm_d/1.414, -arm_d/1.414, 0.0], dtype=np.float32),
            ]
            arm_screens = [view_proj(d.airframe.p + d.airframe.R @ a)[:2] for a in arm_ends_body]
            if all(pt[0] is not None for pt in arm_screens):
                pygame.draw.line(screen, (160, 170, 185), arm_screens[0], arm_screens[2], 2)
                pygame.draw.line(screen, (160, 170, 185), arm_screens[1], arm_screens[3], 2)

                for m_i in range(8):
                    a_i = [0, 0, 1, 1, 2, 2, 3, 3][m_i]
                    mx, my = arm_screens[a_i]
                    off_y = -4 if (m_i % 2 == 0) else 4
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
            
            mid_sx, mid_sy, _ = view_proj(tether_mid_3d)
            psx, psy, _ = view_proj(payload_3d)

            if psx is not None and d.winch.cable_len > 0.10:
                if mid_sx is not None:
                    pygame.draw.lines(screen, (230, 240, 255), False, [(dsx, dsy + 3), (mid_sx, mid_sy), (psx, psy)], 1)
                else:
                    pygame.draw.line(screen, (230, 240, 255), (dsx, dsy + 3), (psx, psy), 1)

            # Cargo box rendering
            if d.winch.has_payload:
                box_x, box_y = (dsx, dsy + 6) if d.winch.cable_len <= 0.10 else (psx, psy)
                if box_x is not None:
                    pygame.draw.rect(screen, (255, 195, 35), (box_x - 6, box_y - 6, 12, 12), border_radius=2)
                    pygame.draw.rect(screen, (60, 45, 10), (box_x - 6, box_y - 6, 12, 12), 1)
            else:
                hook_x, hook_y = (dsx, dsy + 4) if d.winch.cable_len <= 0.10 else (psx, psy)
                if hook_x is not None:
                    pygame.draw.circle(screen, (200, 220, 240), (hook_x, hook_y), 3)

        act_d = drones[selected_drone_idx]
        act_tgt = act_d.target
        tsx, tsy, _ = view_proj(act_tgt)
        if tsx is not None:
            t_ring_col = (0, 220, 255) if "BASE" in act_d.mission_phase or "DEPOT" in act_d.mission_phase or "INBOUND" in act_d.mission_phase else (0, 255, 140)
            pygame.draw.circle(screen, t_ring_col, (tsx, tsy), 16, 2)
            pygame.draw.circle(screen, (255, 255, 255), (tsx, tsy), 4)

        sb_x = VIEW_W + 14
        screen.blit(font_large.render("SURVIVAL CORE V37.0", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Dynamic Self-Adjusting SE(3) Swarm • PFD & Breguet Wind Layer", True, (130, 160, 190)), (sb_x, 38))

        # Primary Flight Display (PFD) Widget
        pfd_rect = (sb_x, 65, 140, 110)
        draw_pfd_artificial_horizon(
            screen, pfd_rect, 
            selected_diag.get("roll", 0.0), selected_diag.get("pitch", 0.0),
            selected_diag.get("alt", 0.0), selected_diag.get("airspeed", 0.0),
            selected_diag.get("vspeed", 0.0), font_small
        )

        # Tactical Minimap Widget
        map_rect = (sb_x + 148, 65, 260, 150)
        draw_tactical_radar_minimap(screen, map_rect, world, drones, selected_drone_idx, traffic_engine.vehicles)

        fl_y = 225
        screen.blit(font_mono.render("SWARM FLEET [TAB] | CAM [C]:", True, (240, 240, 240)), (sb_x, fl_y))
        for d_i, d in enumerate(drones):
            c = swarm_colors[d_i % len(swarm_colors)]
            tag = f"[D{d_i+1}]" + ("*" if d_i == selected_drone_idx else " ")
            stat = d.mission_phase[:8]
            screen.blit(font_small.render(f"{tag} {stat:8s} | {d.bms.soc*100:.0f}% | Box:{d.packages_delivered}", True, c), (sb_x + 10, fl_y + 18 + d_i * 18))

        tot_deliv = dispatcher.total_completed
        cur_mission = act_d.mission
        screen.blit(font_mono.render("ACTIVE ORDER & ARBITER:", True, (240, 240, 240)), (sb_x + 220, fl_y))
        screen.blit(font_small.render(f"ID:   {cur_mission['bldg_id']} (Tag #{cur_mission['tag_id']})", True, (0, 220, 255)), (sb_x + 220, fl_y + 18))
        screen.blit(font_small.render(f"Addr: {cur_mission['addr'][:16]}", True, (0, 255, 140)), (sb_x + 220, fl_y + 36))
        screen.blit(font_small.render(f"Phase:{act_d.mission_phase[:15]}", True, (255, 200, 40)), (sb_x + 220, fl_y + 54))

        p1_y = 300
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p1_y - 8), (W - 14, p1_y - 8), 1)
        screen.blit(font_mono.render("1. DYNAMIC CBBA AUCTION & RTH:", True, (240, 240, 240)), (sb_x, p1_y))
        dist_dest = float(np.linalg.norm(act_d.airframe.p[:2] - act_d.target[:2]))
        dest_tag = f"Depot [H{selected_drone_idx+1}]" if "BASE" in act_d.mission_phase or "DEPOT" in act_d.mission_phase or "INBOUND" in act_d.mission_phase else f"{cur_mission['bldg_id']} ({cur_mission['addr']})"
        screen.blit(font_small.render(f"Target Waypoint:   {dest_tag[:22]} ({dist_dest:.1f}m)", True, (0, 220, 255)), (sb_x + 10, p1_y + 18))
        rth_tag = "[EMERGENCY RTH ACTIVE]" if act_d.rth_active else "[ENERGY PROFILE NOMINAL]"
        screen.blit(font_small.render(f"Live Battery RTH:  {rth_tag}", True, (255, 60, 80) if act_d.rth_active else (0, 255, 140)), (sb_x + 10, p1_y + 34))
        screen.blit(font_small.render(f"Fleet Deliveries:  {tot_deliv} Total | Pending: {len(dispatcher.pending_orders)}", True, (0, 255, 140)), (sb_x + 10, p1_y + 50))

        p2_y = p1_y + 68
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (W - 14, p2_y), 1)
        screen.blit(font_mono.render("2. ADAPTIVE EKF VIO & ARUCO DOCKING:", True, (240, 240, 240)), (sb_x, p2_y + 8))
        fl_2d = selected_diag.get("flow_2d", np.zeros(2))
        screen.blit(font_small.render(f"Adaptive EKF VIO:  R_flow = [{fl_2d[0]:.2f}, {fl_2d[1]:.2f}] (Quality Weighted)", True, (0, 220, 255)), (sb_x + 10, p2_y + 26))
        arc_lk = selected_diag.get("aruco_locked", 0.0)
        arc_txt = "RETICLE LOCKED [TAG MATCH]" if arc_lk > 0.5 else "SEARCHING ROOF CONE..."
        screen.blit(font_small.render(f"ArUco Docking:     {arc_txt}", True, (0, 255, 140) if arc_lk > 0.5 else (255, 140, 40)), (sb_x + 10, p2_y + 42))

        p3_y = p2_y + 64
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (W - 14, p3_y), 1)
        screen.blit(font_mono.render("3. ACOUSTICS & ADS-B CIVIL TRAFFIC:", True, (240, 240, 240)), (sb_x, p3_y + 8))
        adsb_dst = selected_diag.get("adsb_d", 999.0)
        screen.blit(font_small.render(f"ADS-B Civil Alert: Medevac N911 @ {adsb_dst:.1f}m (Corridor Clear)", True, (0, 220, 255)), (sb_x + 10, p3_y + 26))
        srn_spl = selected_diag.get("siren_s", 55.0)
        screen.blit(font_small.render(f"Acoustic Array:    SPL = {srn_spl:.1f} dBA | Siren Detect = {'[ACTIVE]' if srn_spl > 70.0 else '[NOMINAL]'}", True, (255, 60, 80) if srn_spl > 70.0 else (0, 255, 140)), (sb_x + 10, p3_y + 42))

        p4_y = p3_y + 64
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (W - 14, p4_y), 1)
        screen.blit(font_mono.render("4. BREGUET MAX-RANGE CRUISE SPEED GOVERNOR:", True, (240, 240, 240)), (sb_x, p4_y + 8))
        w_mode = weather_engine.get_mode_name()
        w_col = (0, 255, 140) if "CLEAR" in w_mode else ((255, 180, 40) if "CANYON" in w_mode else (255, 60, 80))
        screen.blit(font_small.render(f"Atmospheric State: {w_mode} ([W] Cycle)", True, w_col), (sb_x + 10, p4_y + 26))
        screen.blit(font_small.render(f"Wind-Optimal Air:  |V_tas| = {selected_diag.get('v_tas', 0.0):.2f} m/s | Wind: {selected_diag.get('wind_speed', 0.0):.1f} m/s", True, (0, 220, 255)), (sb_x + 10, p4_y + 42))

        p5_y = p4_y + 64
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p5_y), (W - 14, p5_y), 1)
        screen.blit(font_mono.render("5. GNSS ANTI-SPOOF & QUANTUM VIO:", True, (240, 240, 240)), (sb_x, p5_y + 8))
        g_stat = selected_diag.get("gnss_status", "GNSS NOMINAL")
        screen.blit(font_small.render(f"Navigation State:  {g_stat}", True, (255, 60, 80) if "REJECTED" in g_stat else (0, 255, 140)), (sb_x + 10, p5_y + 26))
        screen.blit(font_small.render(f"Kinematic Discrep: Δa = {selected_diag.get('jerk', 0.0):.2f} m/s² (Thresh = 16.0)", True, (160, 180, 205)), (sb_x + 10, p5_y + 42))
        screen.blit(font_small.render(f"Quantum Phase:     ΔΦ = {selected_diag.get('phase', 0.0):.3f} rad (Rb-87 Raman)", True, (0, 220, 255)), (sb_x + 10, p5_y + 58))

        p6_y = p5_y + 64
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p6_y), (W - 14, p6_y), 1)
        screen.blit(font_mono.render("6. GEOMETRIC SE(3) ACTUATOR MIXER:", True, (240, 240, 240)), (sb_x, p6_y + 8))
        cur_act = selected_diag.get("action", "CRUISE")
        cur_conf = selected_diag.get("confidence", 1.0)
        screen.blit(font_small.render(f"Primary Action:    {cur_act} ({cur_conf*100:.1f}%) | Latency: {selected_diag.get('latency', 1.0):.2f}ms", True, (0, 255, 140)), (sb_x + 10, p6_y + 26))
        screen.blit(font_small.render(f"Winch State:       {act_d.winch.state} (Cable: {act_d.winch.cable_len:.2f} m | Tension: {act_d.winch.tension_n:.1f} N)", True, (0, 220, 255)), (sb_x + 10, p6_y + 42))

        bot_y = H - 38
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y - 8), (W - 14, bot_y - 8), 1)
        screen.blit(font_mono.render("[TAB] Drone | [C] Camera | [W] Weather | [CLICK] Retarget", True, (255, 200, 40)), (sb_x + 8, bot_y))

        vert_spd = act_d.airframe.v[2]
        status_banner = f"ACTIVE: D{selected_drone_idx+1} | Phase: {act_d.mission_phase} (Vz: {vert_spd:+.2f}m/s) | Delivered: {tot_deliv}"
        screen.blit(font_large.render(status_banner, True, (240, 240, 240)), (15, 14))
        screen.blit(font_small.render(f"V37.0 Full Autonomy • SE(3) Geometric • CBBA Swarm • View: {cam_mode}", True, (0, 210, 255)), (15, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    mp.freeze_support()
    run_all_physics_simulator()