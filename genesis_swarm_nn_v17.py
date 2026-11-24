"""
=============================================================================
SURVIVAL CORE V37.0 — FULL ADAPTIVE AUTONOMOUS DRONE SWARM SYSTEM
PART 1 OF 5: SYSTEM CONSTANTS, CAM MODES, WATERTIGHT URDF & 400M METROPOLITAN GRID
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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pygame

# =============================================================================
# 0. PHYSICAL AIRFRAME, SENSORY & MISSION CONSTANTS
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

# Feature Dimensions: Exactly 139 features x 3 history frames = 417 dims
BASE_OBS_DIM = 139
HISTORY_LEN = 3
OBSERVATION_DIM = BASE_OBS_DIM * HISTORY_LEN
NUM_EXECUTIVE_STATES = 7
MODEL_WEIGHTS_FILE = "drone_master_brain_v24_full_sensory.pth"

NUM_SWARM_DRONES = 3

ACTION_NAMES = [
    "TRANSIT",             # 0
    "HOVER_LOCK",          # 1
    "WINCH_LOWER",         # 2
    "TOUCHDOWN_RELEASE",   # 3
    "WINCH_RETRACT",       # 4
    "DEPOT_SERVICE",       # 5
    "EMERGENCY_DIVERT"     # 6
]

# Upgraded 7-Mode Camera Perspectives
CAMERA_MODES = [
    "CHASE",               # 0: Dynamic 7.5m third-person chase
    "NOSE_FPV",            # 1: Cockpit first-person forward view
    "BELLY_GIMBAL",        # 2: Downward tether & winch cargo camera
    "TOP_DOWN",            # 3: 40m tactical satellite nadir view
    "DEPOT_OVERVIEW",      # 4: Elevated depot command view
    "CITY_PANORAMA",       # 5: Long-shot wide cinematic vista of entire 400m city
    "FREE_ORBIT"           # 6: Interactive 360-deg mouse drag orbit
]

# =============================================================================
# 1. PROCEDURAL WATERTIGHT VISUAL URDF GENERATOR
# =============================================================================
def generate_v99_visual_urdf(filepath: str = "drone_coaxial_x8_v99_visual.urdf"):
    """
    Builds a visual coaxial X8 proxy.
    Omits physical <collision> tags so Genesis allocates zero contact DOFs,
    preventing constraint solver warnings and stuttering.
    """
    arm = float(ARM_LENGTH_M)
    d = arm / math.sqrt(2.0)
    r = float(ROTOR_RADIUS_M)
    lines = [
        '<?xml version="1.0"?>', '<robot name="v99_coaxial_x8_visual">',
        '  <link name="base_link">',
        f'    <inertial><origin xyz="0 0 0"/><mass value="{DRY_MASS_KG}"/><inertia ixx="0.022" ixy="0" ixz="0" iyy="0.022" iyz="0" izz="0.038"/></inertial>',
        '    <visual><origin xyz="0 0 0"/><geometry><box size="0.22 0.16 0.08"/></geometry><material name="fuselage"><color rgba="0.12 0.15 0.19 1"/></material></visual>',
        '    <visual><origin xyz="0.12 0 0.01"/><geometry><box size="0.06 0.04 0.03"/></geometry><material name="nose"><color rgba="0.95 0.25 0.25 1"/></material></visual>',
        '    <visual><origin xyz="0 0 0.055"/><geometry><cylinder radius="0.045" length="0.03"/></geometry><material name="gps"><color rgba="0 0.85 0.95 1"/></material></visual>',
    ]
    arms = [((d/2, d/2), math.pi/4), ((d/2, -d/2), -math.pi/4), ((-d/2, d/2), -math.pi/4), ((-d/2, -d/2), math.pi/4)]
    for idx, ((x, y), yaw) in enumerate(arms):
        lines.append(f'    <visual><origin xyz="{x} {y} 0.01" rpy="0 0 {yaw}"/><geometry><box size="{arm} 0.024 0.024"/></geometry><material name="arm{idx}"><color rgba="0.22 0.24 0.27 1"/></material></visual>')
    for idx, (x, y) in enumerate([(d, d), (d, -d), (-d, d), (-d, -d)]):
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

# =============================================================================
# 2. MATHEMATICAL UTILITIES & LIE ALGEBRA ON SO(3)
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
# 3. SPATIAL HASH ENGINE & 400M METROPOLITAN GRID WORLD
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
        aabbs = []
        ids = []
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

        # Powerlines explicitly measured
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
                bx1, bx2 = cx - bldg_w / 2.0, cx + bldg_w / 2.0
                by1, by2 = ry - bldg_d / 2.0, ry + bldg_d / 2.0

                tier = (c_idx + r_idx) % 4
                height = [8.5, 14.5, 21.0, 28.0][tier]
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

    """
=============================================================================
SURVIVAL CORE V37.0 — FULL ADAPTIVE AUTONOMOUS DRONE SWARM SYSTEM
PART 2 OF 5: ADAPTIVE AIRFRAME ENGINES, 3D GLIDE-SLOPE PLANNER,
             DYNAMIC WEATHER, TRAFFIC, INTRUDER, ICING & BATTERY ECM
=============================================================================
"""

# =============================================================================
# 4. ONLINE MASS, COM & DYNAMIC AIRFRAME ESTIMATORS
# =============================================================================
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

# =============================================================================
# 5. 3D GLIDE-SLOPE CORRIDOR & URBAN VERTICAL CHIMNEY PLANNER
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

# =============================================================================
# 6. DYNAMIC WEATHER, ROAD VEHICLES & INTRUDER AGENT ENGINES
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
        else:
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

# =============================================================================
# 7. ROTOR ICING, DRYDEN TURBULENCE & SUB-ZERO BATTERY ECM
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
# 8. ROTOR INFLOW HEALTH & 6-DOF MEMS IMU NOISE SIMULATOR
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

    """
=============================================================================
SURVIVAL CORE V37.0 — FULL ADAPTIVE AUTONOMOUS DRONE SWARM SYSTEM
PART 3 OF 5: POSITION-BASED DYNAMICS WINCH, 8-DOF BEMT COAXIAL AIRFRAME,
             ADVANCED MULTI-MODAL SENSORS, VECTORIZED LIDAR & ASSURANCE SUITE
=============================================================================
"""

# =============================================================================
# 9. AUTONOMOUS WINCH DELIVERY SYSTEM (PBD CABLE & REACTION SENSORS)
# =============================================================================
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
                
                if in_resonance_band and clearance_to_roof > 1.5:
                    base_rate = 3.20

                self.cable_vel = base_rate + np.clip(nn_winch_rate * 0.8, -0.6, 0.6)
                self.cable_len += self.cable_vel * dt

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
# 10. BEMT COAXIAL AIRFRAME 8-DOF WITH RK4 & WALL INTERACTION
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
        self.v += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_p + k4_p)
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
# 11. MULTI-MODAL SENSOR SUITE & PHYSICAL INSTRUMENTATION
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

# =============================================================================
# 12. VECTORIZED MACRO LIDAR & ASSURANCE SUITE
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

        bx1 = valid_bldgs[:, 0]
        bx2 = valid_bldgs[:, 1]
        by1 = valid_bldgs[:, 2]
        by2 = valid_bldgs[:, 3]

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
                min_hit = float(np.min(valid_hits))
                ranges[beam_idx] = min(ranges[beam_idx], min_hit)

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
        return {
            "z_baro": z_baro, 
            "rho_air": rho_air, 
            "v_tas": v_tas, 
            "q_dyn": q_dyn, 
            "emf_rms": emf_rms, 
            "min_sonar": min_sonar
        }

    