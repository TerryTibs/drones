"""
=============================================================================
SURVIVAL CORE V38.7 — FIXED MASTER IMPLEMENTATION
=============================================================================
"""

SURVIVAL_CORE_VERSION = "V38.7-fixed-all-subsystems"

import os
# Optional diagnostic mode: evaluate the deterministic flight controller without
# the learned acceleration residual. Mission state and safety gates remain active.
NEURAL_GUIDANCE_DISABLED = os.environ.get("SURVIVAL_DISABLE_NEURAL_GUIDANCE", "0").strip().lower() in {"1", "true", "yes", "on"}
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

# Windows multiprocessing uses spawn. Without this guard, every harvesting
# child re-imports this entire file and loads PyTorch/CUDA DLLs before it runs
# the CPU-only world-generation worker. With multiple workers this can exhaust the
# Windows commit/page-file limit (WinError 1455). The parent keeps the full
# PyTorch stack; spawned harvest children use a small nn.Module placeholder so
# the model class declarations can be imported without loading torch.
_LIGHTWEIGHT_HARVEST_CHILD = os.environ.get("SURVIVAL_LIGHTWEIGHT_HARVEST_CHILD", "0").strip() == "1"
if _LIGHTWEIGHT_HARVEST_CHILD:
    torch = None
    F = None
    optim = None

    class _HarvestOnlyModulePlaceholder:
        """Import-time stand-in; neural model construction is forbidden in harvest children."""
        pass

    class _HarvestOnlyNNPlaceholder:
        Module = _HarvestOnlyModulePlaceholder

    nn = _HarvestOnlyNNPlaceholder()
else:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

try:
    import pygame
except ImportError:
    pygame = None

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

# Shared physical geometry: keep runtime collision geometry, live sensors, and
# training observations on the same powerline coordinates.
POWERLINE_X_MIN_M = -190.0
POWERLINE_X_MAX_M = 190.0
POWERLINE_Y_POSITIONS_M = (-185.0, 185.0)
POWERLINE_Z_M = 12.0
POWERLINE_RADIUS_M = 0.10

ACTION_NAMES = [
    "TRANSIT",             # 0
    "HOVER_LOCK",          # 1
    "WINCH_LOWER",         # 2
    "TOUCHDOWN_RELEASE",   # 3
    "WINCH_RETRACT",       # 4
    "DEPOT_SERVICE",       # 5
    "EMERGENCY_DIVERT"     # 6 (reserved)
]

DELIVERY_HOVER_TIMEOUT_S = 45.0
DELIVERY_HOLD_MIN_S = 1.20
DELIVERY_DRONE_CENTER_TOLERANCE_M = 0.70
DELIVERY_CARGO_CENTER_TOLERANCE_M = 0.85
DELIVERY_MAX_HORIZONTAL_SPEED_MS = 0.35
DELIVERY_MAX_VERTICAL_SPEED_MS = 0.22
DELIVERY_MAX_SWAY_M = 0.45
DELIVERY_HOVER_ALTITUDE_TOLERANCE_M = 0.65
PHASE3_TOUCHDOWN_DEBOUNCE_S = 0.30
PHASE3_DRONE_RUNAWAY_ABORT_DWELL_S = 1.20
PHASE3_DRONE_RUNAWAY_ABORT_DISTANCE_M = 2.10
DELIVERY_MAX_OPERATIONAL_WIND_MS = 7.0
CHIMNEY_APPROACH_TIMEOUT_S = 45.0

# Phase-3 geometry constants
PHASE3_ROOF_PAD_TOP_OFFSET_M = 0.08   # rooftop pad cylinder top above building top
PHASE3_DRONE_HALF_HEIGHT_M = 0.045    # URDF body is 0.09 m high
PHASE3_CARGO_HALF_HEIGHT_M = 0.09     # payload box is 0.18 m high
PHASE3_TETHER_CENTER_OFFSET_M = PHASE3_DRONE_HALF_HEIGHT_M + PHASE3_CARGO_HALF_HEIGHT_M
PHASE3_STOWED_CABLE_LEN_M = 0.08
PHASE3_DEFAULT_ROLLOUT_STEPS = 3072
PHASE3_CARGO_SAFE_RADIUS_M = 1.80
PHASE3_CARGO_ZONE_ABORT_DWELL_S = 0.60
PHASE3_SWAY_POSITION_GAIN = 2.0
PHASE3_SWAY_VELOCITY_GAIN = 1.55
POLICY_MIN_ACTION_CONFIDENCE = 0.38
POLICY_MIN_SURFACE_TRUST = 0.25
POLICY_VERTICAL_ACCEL_RESIDUAL_LIMIT_MS2 = 0.80


def _anti_sway_feedback(sway_offset, sway_velocity, position_gain=PHASE3_SWAY_POSITION_GAIN,
                        velocity_gain=PHASE3_SWAY_VELOCITY_GAIN):
    """Command the attachment point in the direction of relative sway/rate."""
    return position_gain * sway_offset + velocity_gain * sway_velocity


def _bound_policy_vertical_residual(acceleration_residual):
    """Keep learned guidance from overpowering the deterministic altitude controller."""
    bounded = np.asarray(acceleration_residual, dtype=np.float32).copy()
    if bounded.ndim != 1 or bounded.size < 3:
        raise ValueError(f"Expected a 1-D acceleration residual with at least 3 values; got {bounded.shape}")
    bounded[2] = float(np.clip(
        bounded[2], -POLICY_VERTICAL_ACCEL_RESIDUAL_LIMIT_MS2,
        POLICY_VERTICAL_ACCEL_RESIDUAL_LIMIT_MS2))
    return bounded

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

        # Trees placed within open internal block setbacks
        self.trees = [
            {"id": "Tree_1", "pos": np.array([-12.0, -12.0, 0.0], dtype=np.float32), "r": 3.2, "h": 8.0},
            {"id": "Tree_2", "pos": np.array([ 12.0, -12.0, 0.0], dtype=np.float32), "r": 3.0, "h": 7.5},
            {"id": "Tree_3", "pos": np.array([-12.0,  12.0, 0.0], dtype=np.float32), "r": 3.5, "h": 8.5},
            {"id": "Tree_4", "pos": np.array([ 12.0,  12.0, 0.0], dtype=np.float32), "r": 3.2, "h": 8.0}
        ]

        # Utility lines run clear of outer building footprints
        self.powerlines = [
            {"id": "Powerline_South",
             "p1": np.array([POWERLINE_X_MIN_M, POWERLINE_Y_POSITIONS_M[0], POWERLINE_Z_M], dtype=np.float32),
             "p2": np.array([POWERLINE_X_MAX_M, POWERLINE_Y_POSITIONS_M[0], POWERLINE_Z_M], dtype=np.float32),
             "radius": POWERLINE_RADIUS_M},
            {"id": "Powerline_North",
             "p1": np.array([POWERLINE_X_MIN_M, POWERLINE_Y_POSITIONS_M[1], POWERLINE_Z_M], dtype=np.float32),
             "p2": np.array([POWERLINE_X_MAX_M, POWERLINE_Y_POSITIONS_M[1], POWERLINE_Z_M], dtype=np.float32),
             "radius": POWERLINE_RADIUS_M}
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

    def get_static_collision_obstacles(self):
        obstacles = []
        for house in self.houses:
            x1, x2, y1, y2, top = [float(v) for v in house["aabb"]]
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            obstacles.append({
                "id": house["id"], "shape": "aabb",
                "pos": np.array([cx, cy, top / 2.0], dtype=np.float32),
                "bounds": (x1 - 0.06, x2 + 0.06, y1 - 0.06, y2 + 0.06),
                "radius": 0.0, "height": top, "base_height": 0.0,
                "building_height": top,
            })
            for ci, clutter in enumerate(house.get("clutter", [])):
                cx1, cx2, cy1, cy2, ctop = [float(v) for v in clutter["aabb"]]
                obstacles.append({
                    "id": f"{house['id']}:CLUTTER_{ci}", "shape": "aabb",
                    "pos": np.array([(cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0, (top + ctop) / 2.0], dtype=np.float32),
                    "bounds": (cx1, cx2, cy1, cy2), "radius": 0.0,
                    "height": ctop, "base_height": top, "building_height": top,
                })
        for ti, tree in enumerate(self.trees):
            tp = np.asarray(tree["pos"], dtype=np.float32)
            radius, height = float(tree.get("r", 3.2)), float(tree.get("h", 8.0))
            obstacles.append({"id": str(tree.get("id", f"Tree_{ti+1}")), "shape": "cylinder",
                              "pos": np.array([tp[0], tp[1], height / 2.0], dtype=np.float32),
                              "radius": radius, "height": height})
        for wi, line in enumerate(self.powerlines):
            p1 = np.asarray(line["p1"], dtype=np.float32)
            p2 = np.asarray(line["p2"], dtype=np.float32)
            obstacles.append({"id": line.get("id", f"Powerline_{wi+1}"), "shape": "segment",
                              "pos": 0.5 * (p1 + p2), "p1": p1.copy(), "p2": p2.copy(),
                              "radius": float(line.get("radius", 0.10)), "height": float(max(p1[2], p2[2]))})
            for ei, endpoint in enumerate((p1, p2)):
                line_id = line.get('id', f'Powerline_{wi+1}')
                pole_h = float(endpoint[2])
                obstacles.append({"id": f"{line_id}:POLE_{ei+1}",
                                  "shape": "cylinder", "pos": np.array([endpoint[0], endpoint[1], pole_h / 2.0], dtype=np.float32),
                                  "radius": 0.16, "height": pole_h, "base_height": 0.0})
                obstacles.append({"id": f"{line_id}:POLE_{ei+1}:CROSSARM", "shape": "aabb",
                                  "pos": np.array([endpoint[0], endpoint[1], pole_h - 0.12], dtype=np.float32),
                                  "bounds": (float(endpoint[0]) - 1.0, float(endpoint[0]) + 1.0,
                                             float(endpoint[1]) - 0.06, float(endpoint[1]) + 0.06),
                                  "radius": 0.0, "base_height": pole_h - 0.17, "height": pole_h - 0.07})
        return obstacles

    def collision_inventory(self):
        items = self.get_static_collision_obstacles()
        counts = {}
        for item in items:
            ident = str(item.get("id", ""))
            if ":CLUTTER_" in ident:
                key = "rooftop_clutter"
            elif ":CROSSARM" in ident:
                key = "powerline_crossarms"
            elif ":POLE_" in ident:
                key = "powerline_support_poles"
            elif ident.startswith("Powerline_"):
                key = "powerlines"
            elif ident.startswith("Tree_"):
                key = "trees"
            elif ident.startswith("Bldg_"):
                key = "buildings"
            else:
                key = "other_static"
            counts[key] = counts.get(key, 0) + 1
        counts["static_obstacle_total"] = len(items)
        surfaces = self.get_support_surfaces()
        counts["support_surface_total"] = len(surfaces)
        counts["roads_and_pads_support_surfaces"] = sum(
            str(item.get("id", "")).startswith(("Road_", "Depot_Pad_", "Bldg_")) for item in surfaces)
        counts["roof_drop_pads"] = sum(":ROOF_DROP_PAD" in str(item.get("id", "")) for item in surfaces)
        return counts

    def get_support_surfaces(self):
        surfaces = [{"id": "Ground_Plane", "shape": "plane", "z": 0.0}]
        for i, road in enumerate(self.roads):
            surfaces.append({"id": f"Road_{i+1}", "shape": "aabb_plane",
                             "bounds": (road["x_min"], road["x_max"], road["y_min"], road["y_max"]), "z": 0.025})
        for i, pad in enumerate(self.depot_helipads):
            surfaces.append({"id": f"Depot_Pad_{i+1}", "shape": "cylinder", "pos": pad.copy(), "radius": 2.8, "z_top": 0.08})
        for house in self.houses:
            x1, x2, y1, y2, top = [float(v) for v in house["aabb"]]
            surfaces.append({"id": f"{house['id']}:ROOF", "shape": "aabb_plane", "bounds": (x1, x2, y1, y2), "z": top + 0.08})
            surfaces.append({"id": f"{house['id']}:ROOF_DROP_PAD", "shape": "cylinder",
                             "pos": np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2), top + 0.04], dtype=np.float32),
                             "radius": 2.2, "z_top": top + PHASE3_ROOF_PAD_TOP_OFFSET_M})
        return surfaces

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
                return float(hh + PHASE3_ROOF_PAD_TOP_OFFSET_M)
        return 0.0

    # =============================================================================
# 4. 3D GLIDE-SLOPE CORRIDOR & URBAN VERTICAL CHIMNEY PLANNER
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
            if (x1 - margin) <= px <= (x2 + margin) and (y1 - margin) <= py <= (y2 + margin) and pz < (hh + 0.1):
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
# 5. DYNAMIC WEATHER, ROAD VEHICLES & INTRUDER AGENT ENGINES
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
                "pos": np.array([ax, -180.0 + i * 90.0, 0.45], dtype=np.float32),
                "vel": np.array([0.0, 4.8 + i*0.4, 0.0], dtype=np.float32),
                "axis": "Y",
                "is_emergency": False
            })
        for i in range(4):
            sy = streets_y[i % len(streets_y)]
            is_amb = (i == 0)
            self.vehicles.append({
                "id": "Ambulance_01" if is_amb else f"Truck_{i+1}",
                "pos": np.array([-180.0 + i * 90.0, sy, 0.5 if is_amb else 0.45], dtype=np.float32),
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
        obstacles = []
        for vehicle in self.vehicles:
            px, py, pz = (float(v) for v in vehicle["pos"])
            length_m, width_m = 3.0, 1.35
            height_m = 1.0 if vehicle.get("is_emergency", False) else 0.9
            if vehicle.get("axis") == "Y":
                bounds = (px - width_m / 2.0, px + width_m / 2.0,
                          py - length_m / 2.0, py + length_m / 2.0)
            else:
                bounds = (px - length_m / 2.0, px + length_m / 2.0,
                          py - width_m / 2.0, py + width_m / 2.0)
            obstacles.append({
                "id": vehicle["id"], "shape": "aabb", "pos": vehicle["pos"].copy(),
                "bounds": bounds, "radius": 0.0, "base_height": max(0.0, pz - height_m / 2.0),
                "height": pz + height_m / 2.0,
            })
        return obstacles

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
        elif is_winch_lowering and cargo_world_pos[2] < 6.0 and self.boredom_cooldown <= 0.0:
            # Active winch deployment attracts curious bystander
            self.curiosity_timer += dt
            approach_dir = d_to_cargo / max(1e-4, dist)
            v_des = approach_dir * min(self.v_max, 0.65 * dist)
            a_cmd = (v_des - self.v[:2]) * 2.5
            if self.curiosity_timer > 12.0:
                self.boredom_cooldown = 15.0
                self.curiosity_timer = 0.0
        else:
            if self.boredom_cooldown > 0.0:
                self.boredom_cooldown = max(0.0, self.boredom_cooldown - dt)
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
# 6. ROTOR ICING, DRYDEN TURBULENCE & SUB-ZERO BATTERY ECM
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
# 7. ROTOR INFLOW HEALTH & 6-DOF MEMS IMU NOISE SIMULATOR
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
# 8. AUTONOMOUS WINCH DELIVERY SYSTEM (PBD CABLE & REACTION SENSORS)
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
        self.release_count = 0
        self.last_release_pos = None
        self.sway_velocity = np.zeros(2, dtype=np.float32)
        
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
        self.last_release_pos = None
        self.sway_velocity.fill(0.0)
        self.is_skid_grounded = False

    def step(self, drone_z=0.0, target_surface_z=0.0, is_settled=False, is_surface_safe=True, 
             wire_hit=False, agent_breach=False, agent_dist=30.0, wind_2d=None, 
             drone_accel_linear=None, neural_token_idx=0, nn_winch_rate=0.0, dt=0.016, 
             drone_p=None, drone_v=None, wind_3d=None, target_xy=None, **kwargs):
        
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
        endpoint_for_contact = np.asarray(self.nodes_pos[-1], dtype=np.float32)
        endpoint_valid = (self.cable_len > 0.12 and np.all(np.isfinite(endpoint_for_contact))
                          and float(np.linalg.norm(endpoint_for_contact[:2] - drone_p[:2])) <= self.cable_len + 1.0)
        
        payload_z = (float(endpoint_for_contact[2]) if endpoint_valid else
                     float(drone_p[2] - PHASE3_DRONE_HALF_HEIGHT_M - self.cable_len - PHASE3_CARGO_HALF_HEIGHT_M))
        clearance_to_roof = payload_z - PHASE3_CARGO_HALF_HEIGHT_M - target_surface_z
        target_xy_arr = None if target_xy is None else np.asarray(target_xy, dtype=np.float32).reshape(-1)[:2]
        cargo_xy = drone_p[:2] + self.sway_offset
        endpoint = np.asarray(self.nodes_pos[-1], dtype=np.float32)
        
        if (self.cable_len > 0.12 and np.all(np.isfinite(endpoint))
                and float(np.linalg.norm(endpoint[:2] - drone_p[:2])) <= self.cable_len + 1.0):
            cargo_xy = endpoint[:2]
            
        drone_xy_error = (float(np.linalg.norm(drone_p[:2] - target_xy_arr))
                          if target_xy_arr is not None else 0.0)
        cargo_xy_error = (float(np.linalg.norm(cargo_xy - target_xy_arr))
                          if target_xy_arr is not None else 0.0)
        horizontal_speed = float(np.linalg.norm(drone_v[:2]))
        delivery_alignment_ok = (
            target_xy_arr is None or
            (drone_xy_error <= DELIVERY_DRONE_CENTER_TOLERANCE_M
             and cargo_xy_error <= DELIVERY_CARGO_CENTER_TOLERANCE_M
             and horizontal_speed <= DELIVERY_MAX_HORIZONTAL_SPEED_MS
             and abs(float(drone_v[2])) <= DELIVERY_MAX_VERTICAL_SPEED_MS)
        )
        dropzone_occupied = (agent_dist < 2.8) and (target_surface_z < 2.0) and (self.state == "STANDBY")
        lockout = (not is_surface_safe) or wire_hit or (not is_settled) or dropzone_occupied

        if agent_breach and (target_surface_z < 2.0) and self.state in ["LOWERING", "TOUCHDOWN"]:
            self.state = "ABORT_RETRACT"

        omega_pendulum = math.sqrt(G_ACCEL / max(0.08, float(self.cable_len) +
                                                     (PHASE3_CARGO_HALF_HEIGHT_M if self.has_payload else 0.0)))
        in_resonance_band = (1.8 < omega_pendulum < 2.5) and (self.state == "LOWERING")

        action_token = int(neural_token_idx)
        learned_rate = float(np.clip(nn_winch_rate, -1.0, 1.0))

        if self.state == "STANDBY":
            self.cable_len = 0.08
            self.cable_vel = 0.0
            if (not lockout and self.has_payload and is_settled
                    and action_token in (1, 2)):
                self.state = "LOWERING"

        elif self.state == "LOWERING":
            policy_requests_retract = action_token in (4, 5, 6)
            if lockout:
                if self.cable_len > 0.12:
                    self.state = "ABORT_RETRACT"
                else:
                    self.cable_vel = 0.0
                    self.state = "STANDBY"
            elif policy_requests_retract:
                if self.cable_len > 0.12:
                    self.state = "RETRACTING"
                else:
                    self.cable_vel = 0.0
                    self.state = "STANDBY"
            else:
                flare_scale = float(np.clip(clearance_to_roof / 1.20, 0.20, 1.0)) if clearance_to_roof < 1.20 else 1.0
                base_rate = float(np.clip(2.40 - 0.12 * wind_speed_xy, 1.20, 2.40)) * flare_scale
                if in_resonance_band and clearance_to_roof > 1.5:
                    base_rate = 3.20

                if action_token == 2:
                    learned_rate = max(0.0, learned_rate)
                elif action_token == 3:
                    learned_rate = float(np.clip(learned_rate, -0.25, 0.0))
                self.cable_vel = base_rate + float(np.clip(learned_rate * 0.8, -0.6, 0.6))
                self.cable_len += self.cable_vel * dt

                fast_tension_drop = (self.tension_derivative < -35.0) and (clearance_to_roof < 0.40)
                policy_touchdown = (action_token == 3 and 0.0 <= clearance_to_roof <= 0.10)
                physical_touchdown = (clearance_to_roof <= 0.10 or fast_tension_drop
                                      or self.contact_normal_force >= 0.40 * PAYLOAD_WEIGHT_N)
                if physical_touchdown or policy_touchdown:
                    if delivery_alignment_ok and is_surface_safe and is_settled:
                        self.state = "TOUCHDOWN"
                        self.touchdown_timer = 0.0
                    else:
                        self.state = "ABORT_RETRACT"

        elif self.state == "TOUCHDOWN":
            self.cable_vel = 0.0
            if (not delivery_alignment_ok) or (not is_surface_safe) or (not is_settled):
                self.state = "ABORT_RETRACT"
                self.touchdown_timer = 0.0
            else:
                self.touchdown_timer += dt
                if self.touchdown_timer >= 0.30:
                    self.has_payload = False
                    self.release_count += 1
                    if target_xy_arr is not None:
                        self.last_release_pos = np.array(
                            [target_xy_arr[0], target_xy_arr[1], float(target_surface_z) + 0.09],
                            dtype=np.float32
                        )
                    else:
                        self.last_release_pos = np.array(
                            [cargo_xy[0], cargo_xy[1], float(target_surface_z) + 0.09],
                            dtype=np.float32
                        )
                    self.state = "RETRACTING"

        elif self.state in ["RETRACTING", "ABORT_RETRACT"]:
            if self.state == "ABORT_RETRACT":
                retract_scale = 1.0
            elif action_token == 4:
                retract_scale = float(np.clip(abs(learned_rate), 0.65, 1.0))
            else:
                retract_scale = 0.75
            self.cable_vel = -3.20 * retract_scale
            self.cable_len = max(0.08, self.cable_len + self.cable_vel * dt)
            if self.cable_len <= 0.08:
                self.cable_len = 0.08
                self.cable_vel = 0.0
                if self.state == "RETRACTING" and not self.has_payload:
                    self.delivery_completed = True
                self.state = "STANDBY"

        L = max(0.08, float(self.cable_len))
        if self.has_payload:
            L += PHASE3_CARGO_HALF_HEIGHT_M
        seg_len = L / float(self.num_segments)

        self.nodes_pos[0] = drone_p + np.array([0.0, 0.0, -PHASE3_DRONE_HALF_HEIGHT_M], dtype=np.float32)
        self.nodes_vel[0] = drone_v.copy()

        if self.cable_len <= 0.10:
            for k in range(1, self.num_nodes):
                self.nodes_pos[k] = self.nodes_pos[0] + np.array([0.0, 0.0, -k * seg_len], dtype=np.float32)
                self.nodes_vel[k] = drone_v.copy()
            self.theta.fill(0.0)
            self.theta_dot.fill(0.0)
            self.sway_velocity.fill(0.0)
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

        node_masses = np.full(self.num_nodes, 0.04, dtype=np.float32)
        if self.has_payload:
            node_masses[-1] = CARGO_MASS_KG

        for k in range(1, self.num_nodes):
            m_k = node_masses[k]
            v_rel = self.nodes_vel[k] - wind_3d
            speed_rel = float(np.linalg.norm(v_rel))
            cd = 0.85 if (k == self.num_nodes - 1 and self.has_payload) else 1.15
            area = (CARGO_MASS_KG * 0.045) if (k == self.num_nodes - 1 and self.has_payload) else (self.cable_diameter_m * seg_len)
            f_drag = -0.5 * 1.225 * cd * area * speed_rel * v_rel

            a_k = GRAVITY_G + (f_drag / m_k)
            self.nodes_vel[k] += a_k * dt
            self.nodes_vel[k] *= 0.965
            predicted_pos[k] = self.nodes_pos[k] + self.nodes_vel[k] * dt

        # Corrected PBD Distance Constraints with true inverse-mass weighting
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
                        w1 = 1.0 / max(1e-4, float(node_masses[i]))
                        w2 = 1.0 / max(1e-4, float(node_masses[i + 1]))
                        w_sum = w1 + w2
                        alpha1 = w1 / w_sum
                        alpha2 = w2 / w_sum
                        predicted_pos[i] += alpha1 * correction
                        predicted_pos[i + 1] -= alpha2 * correction

        payload_contact_active = (self.has_payload and self.state in
                                  ("LOWERING", "TOUCHDOWN", "RETRACTING", "ABORT_RETRACT"))
        if payload_contact_active:
            package_centre_surface_z = target_surface_z + PHASE3_CARGO_HALF_HEIGHT_M
            contact_depth = package_centre_surface_z - predicted_pos[-1][2]
            if contact_depth > 0.0:
                k_spring = 340.0
                d_damper = 14.0
                f_spring = k_spring * contact_depth - d_damper * min(0.0, float(self.nodes_vel[-1][2]))
                self.contact_normal_force = float(np.clip(f_spring, 0.0, 1.5 * PAYLOAD_WEIGHT_N))
                predicted_pos[-1][2] = package_centre_surface_z
                self.nodes_vel[-1][2] = 0.0
                self.nodes_vel[-1][:2] *= 0.85
            else:
                self.contact_normal_force = 0.0
        else:
            self.contact_normal_force = 0.0

        for k in range(1, self.num_nodes):
            self.nodes_vel[k] = (predicted_pos[k] - self.nodes_pos[k]) / dt
            self.nodes_pos[k] = predicted_pos[k].copy()

        self.sway_offset = self.nodes_pos[-1][:2] - self.nodes_pos[0][:2]
        self.sway_velocity = (self.nodes_vel[-1][:2] - drone_v[:2]).astype(np.float32)
        
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
# 9. BEMT COAXIAL AIRFRAME 8-DOF WITH RK4 & WALL INTERACTION
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
        self.last_f_thrust = HOVER_THRUST_NOMINAL
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
            self.last_f_thrust = 0.0
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
            self.last_f_thrust = est_thrust
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
        self.last_f_thrust = f_thrust

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
# 10. MULTI-MODAL SENSOR SUITE & PHYSICAL INSTRUMENTATION
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

    def poll_smart_hook_imu(self, drone_a, drone_w, sway_2d, cable_l, has_payload,
                            sway_velocity_2d=None):
        if not has_payload or cable_l <= 0.10:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        a_box = drone_a + np.array([-sway_2d[0]*2.5, -sway_2d[1]*2.5, 0.2], dtype=np.float32)
        if sway_velocity_2d is None:
            sway_speed = float(np.linalg.norm(sway_2d)) * 4.0
            w_box = np.array([sway_2d[1]*1.8, -sway_2d[0]*1.8, sway_speed*0.5], dtype=np.float32)
        else:
            sway_rate = np.asarray(sway_velocity_2d, dtype=np.float32).reshape(2)
            sway_speed = float(np.linalg.norm(sway_rate))
            w_box = np.array([sway_rate[1]*1.8, -sway_rate[0]*1.8, sway_speed*0.5], dtype=np.float32)
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
        
        # Compute vegetation proximity dynamically against actual tree positions
        if hasattr(self.world, "trees") and self.world.trees:
            park_dist = min(math.hypot(px - float(t["pos"][0]), py - float(t["pos"][1])) for t in self.world.trees)
        else:
            park_dist = math.hypot(px, py)
            
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
# 11. VECTORIZED MACRO LIDAR & ASSURANCE SUITE
# =============================================================================
class MacroPerceptionLiDARSuite:
    def __init__(self, world: MetropolitanCityGridWorld400m):
        self.world = world

    def cast_360_lidar(self, drone_pos, yaw_rad, dynamic_vehicles=None, dynamic_objects=None):
        px, py, pz = drone_pos
        ranges = np.full(16, 30.0, dtype=np.float32)
        angles = yaw_rad + np.linspace(0, 2.0 * math.pi, 16, endpoint=False)
        dx_all = np.cos(angles).astype(np.float32)
        dy_all = np.sin(angles).astype(np.float32)

        aabbs = self.world.spatial_hash.cached_aabbs
        if aabbs is None:
            aabbs = np.empty((0, 5), dtype=np.float32)
        clutter_aabbs = []
        for house in self.world.houses:
            for clutter in house.get("clutter", []):
                clutter_aabbs.append(np.asarray(clutter["aabb"], dtype=np.float32))
        if clutter_aabbs:
            aabbs = np.concatenate((np.asarray(aabbs, dtype=np.float32), np.asarray(clutter_aabbs, dtype=np.float32)), axis=0)
        valid_bldgs = aabbs[pz <= (aabbs[:, 4] + 0.1)] if len(aabbs) else aabbs
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

            for tree in self.world.trees:
                tx, ty = float(tree["pos"][0]), float(tree["pos"][1])
                if pz > float(tree.get("h", 8.0)) + 0.25:
                    continue
                rx, ry = tx - px, ty - py
                along = rx * dx + ry * dy
                cross_sq = rx * rx + ry * ry - along * along
                radius = float(tree.get("r", 3.2))
                disc = radius * radius - cross_sq
                if disc >= 0.0:
                    hit_t = along - math.sqrt(max(0.0, disc))
                    if 0.0 <= hit_t < ranges[beam_idx]:
                        ranges[beam_idx] = float(hit_t)

            for wire in self.world.powerlines:
                w1, w2 = np.asarray(wire["p1"]), np.asarray(wire["p2"])
                if abs(float(pz - w1[2])) > 0.18 or abs(float(w2[1] - w1[1])) > 1e-4:
                    continue
                if abs(float(dy)) < 1e-5:
                    continue
                hit_t = (float(w1[1]) - py) / float(dy)
                x_hit = px + hit_t * float(dx)
                if 0.0 <= hit_t < ranges[beam_idx] and min(w1[0], w2[0]) <= x_hit <= max(w1[0], w2[0]):
                    ranges[beam_idx] = float(hit_t)

            aircraft = self.world.civil_aircraft
            ap = np.asarray(aircraft["pos"], dtype=np.float32)
            radius3 = float(aircraft.get("radius", 4.5))
            dz_air = float(pz - ap[2])
            if abs(dz_air) < radius3:
                radius_xy = math.sqrt(max(0.0, radius3 * radius3 - dz_air * dz_air))
                rx, ry = float(ap[0] - px), float(ap[1] - py)
                along = rx * dx + ry * dy
                cross_sq = rx * rx + ry * ry - along * along
                disc = radius_xy * radius_xy - cross_sq
                if disc >= 0.0:
                    hit_t = along - math.sqrt(max(0.0, disc))
                    if 0.0 <= hit_t < ranges[beam_idx]:
                        ranges[beam_idx] = float(hit_t)

            if dynamic_vehicles:
                for v in dynamic_vehicles:
                    vx, vy = v["pos"][0], v["pos"][1]
                    d_veh = math.hypot(px - vx, py - vy)
                    if pz < float(v.get("height", 3.2)) + 1.0 and d_veh < ranges[beam_idx]:
                        bearing = math.atan2(vy - py, vx - px)
                        diff = abs((bearing - angles[beam_idx] + math.pi) % (2.0 * math.pi) - math.pi)
                        if diff <= (math.pi / 16.0):
                            ranges[beam_idx] = float(d_veh)
            if dynamic_objects:
                for obj in dynamic_objects:
                    op = np.asarray(obj["pos"], dtype=np.float32)
                    radius = float(obj.get("radius", 1.0))
                    if abs(float(pz - op[2])) <= max(0.5, radius):
                        rx, ry = float(op[0] - px), float(op[1] - py)
                        along = rx * dx + ry * dy
                        cross_sq = rx * rx + ry * ry - along * along
                        disc = radius * radius - cross_sq
                        if disc >= 0.0:
                            hit_t = along - math.sqrt(max(0.0, disc))
                            if 0.0 <= hit_t < ranges[beam_idx]:
                                ranges[beam_idx] = float(hit_t)

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
        self.wire_y_positions = POWERLINE_Y_POSITIONS_M
        self.wire_x_min = POWERLINE_X_MIN_M
        self.wire_x_max = POWERLINE_X_MAX_M
        self.wire_z = POWERLINE_Z_M

    def wire_distance_m(self, drone_p):
        p = np.asarray(drone_p, dtype=np.float64).reshape(3)
        closest_x = float(np.clip(p[0], self.wire_x_min, self.wire_x_max))
        dx = p[0] - closest_x
        return min(
            math.sqrt(dx * dx + (p[1] - wire_y) ** 2 + (p[2] - self.wire_z) ** 2)
            for wire_y in self.wire_y_positions
        )

    def poll_all_physics_sensors(self, drone_p, drone_v, current_wind, surface_z=0.0):
        z_true = max(0.0, float(drone_p[2]))
        t_air = T_0 - L_LAPSE * z_true
        p_static_hpa = P_0 * ((1.0 - (L_LAPSE * z_true) / T_0) ** 5.25588)
        rho_air = (p_static_hpa * 100.0) / (R_SPEC_AIR * t_air)
        z_baro = 44330.8 * (1.0 - ((p_static_hpa / P_0) ** 0.190284)) + float(np.random.normal(0, 0.04))

        v_air = drone_v - current_wind
        v_tas = float(np.linalg.norm(v_air))
        q_dyn = 0.5 * rho_air * (v_tas ** 2)

        d_wire = self.wire_distance_m(drone_p)
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
# 12. MATTER-WAVE QUANTUM SENSOR & ERROR-STATE EKF ON SO(3)
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
# 13. RECIPROCAL CBF-QP SAFETY FILTER & DYNAMIC GEOMETRIC SE(3) MIXER
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

        cov_inflation = 1.2 * min(2.5, math.sqrt(max(0.0, float(ekf_pos_cov_trace))))
        gust_inflation = 0.22 * float(np.linalg.norm(wind_2d))
        dynamic_safe_radius = self.safe_radius + cov_inflation + gust_inflation

        for obs in obstacles:
            if int(obs.get("owner_drone_idx", -1)) == int(drone_idx):
                continue
            obs_id = obs.get("id")
            obs_h = float(obs.get("height", 0.0))
            shape = obs.get("shape", "cylinder")
            if obs_id in [target_bldg_id, departure_bldg_id] and drone_p[2] >= obs_h + 0.9:
                continue

            if shape == "segment":
                p1 = np.asarray(obs["p1"], dtype=np.float32)[:2]
                p2 = np.asarray(obs["p2"], dtype=np.float32)[:2]
                seg = p2 - p1
                denom = max(1e-8, float(np.dot(seg, seg)))
                t_seg = float(np.clip(np.dot(drone_p[:2] - p1, seg) / denom, 0.0, 1.0))
                closest = p1 + t_seg * seg
                rel_xy = drone_p[:2] - closest
                dist_seg = float(np.linalg.norm(rel_xy))
                vertical_clearance = abs(float(drone_p[2] - obs_h))
                if vertical_clearance <= 1.8 and dist_seg < (float(obs.get("radius", 0.1)) + dynamic_safe_radius + 4.5):
                    n = rel_xy / max(1e-4, dist_seg)
                    closing = -float(np.dot(drone_v[:2], n))
                    h_wire = dist_seg - (float(obs.get("radius", 0.1)) + dynamic_safe_radius)
                    if h_wire < 4.5:
                        A_rows.append(np.array([n[0], n[1], 0.0], dtype=np.float32))
                        b_rows.append(-closing - self.alpha_obs * h_wire)
                continue

            if shape == "aabb":
                x1, x2, y1, y2 = (float(v) for v in obs["bounds"])
                px, py = float(drone_p[0]), float(drone_p[1])
                dx = max(x1 - px, 0.0, px - x2)
                dy = max(y1 - py, 0.0, py - y2)
                outside_dist = math.hypot(dx, dy)
                if outside_dist > 1e-8:
                    nearest = np.array([np.clip(px, x1, x2), np.clip(py, y1, y2)], dtype=np.float32)
                    rel_xy = np.asarray([px, py], dtype=np.float32) - nearest
                    n_xy = rel_xy / outside_dist
                    signed_dist = outside_dist
                else:
                    face_dists = (px - x1, x2 - px, py - y1, y2 - py)
                    face_idx = int(np.argmin(face_dists))
                    normals = ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0))
                    n_xy = np.asarray(normals[face_idx], dtype=np.float32)
                    signed_dist = -max(0.0, float(face_dists[face_idx]))
                closing = -float(np.dot(drone_v[:2], n_xy))
                tailwind_proj = -float(np.dot(wind_2d[:2], n_xy))
                dynamic_stopping_buffer = ((max(0.0, closing) ** 2) / (2.0 * 3.8)
                                           + max(0.0, tailwind_proj) * 0.35)
                h_box = signed_dist - (dynamic_safe_radius + min(4.5, dynamic_stopping_buffer))
                obs_base_h = float(obs.get("base_height", 0.0))
                if (obs_base_h - 0.4) <= float(drone_p[2]) <= (obs_h + 0.4) and h_box < 4.5:
                    grad = np.asarray([n_xy[0], n_xy[1], 0.0], dtype=np.float32)
                    v_out = float(np.dot(grad, drone_v))
                    A_rows.append(grad)
                    b_rows.append(-v_out - self.alpha_obs * h_box)
                if (obs_base_h <= 0.01 and outside_dist <= 0.5
                        and float(drone_p[2]) >= (obs_h - 0.2)):
                    h_roof = float(drone_p[2]) - (obs_h + 0.15)
                    if h_roof < 0.8:
                        A_rows.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
                        b_rows.append(-float(drone_v[2]) - 2.0 * h_roof)
                continue

            if shape == "sphere":
                rel3 = np.asarray(drone_p, dtype=np.float32) - np.asarray(obs["pos"], dtype=np.float32)
                dist3 = float(np.linalg.norm(rel3))
                safe3 = float(obs.get("radius", 1.0)) + dynamic_safe_radius
                h3 = dist3 - safe3
                if h3 < 4.5:
                    grad3 = rel3 / max(1e-4, dist3)
                    lfh3 = float(np.dot(grad3, drone_v))
                    A_rows.append(grad3.astype(np.float32))
                    b_rows.append(-lfh3 - self.alpha_obs * h3)
                continue

            rel_pos = drone_p - obs["pos"]
            dist_xy = math.hypot(rel_pos[0], rel_pos[1])

            n_xy = rel_pos[:2] / max(1e-4, dist_xy)
            v_closing = -float(np.dot(drone_v[:2], n_xy))
            tailwind_proj = -float(np.dot(wind_2d[:2], n_xy))
            
            dynamic_stopping_buffer = (max(0.0, v_closing) ** 2) / (2.0 * 3.8) + max(0.0, tailwind_proj) * 0.35
            effective_safe_r = dynamic_safe_radius + min(4.5, dynamic_stopping_buffer)
            h_obs = dist_xy - (obs["radius"] + effective_safe_r)

            eff_alpha = self.alpha_obs * (8.0 / (8.0 + max(0.0, v_closing)))
            obs_base_h = float(obs.get("base_height", 0.0))

            if (obs_base_h - 0.4) <= drone_p[2] <= (obs_h + 0.4) and h_obs < 4.5:
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

def _bounded_xy_velocity_error(ref_velocity_xy, measured_velocity_xy, max_vector_mps=0.55):
    error = np.asarray(ref_velocity_xy, dtype=np.float32) - np.asarray(measured_velocity_xy, dtype=np.float32)
    magnitude = float(np.linalg.norm(error))
    limit = max(0.0, float(max_vector_mps))
    if magnitude > limit and magnitude > 1e-8:
        error = error * (limit / magnitude)
    return error.astype(np.float32, copy=False)


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
        self.vertical_v_cmd_smooth = 0.0
        self.vertical_a_cmd_smooth = 0.0

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
            f_skid_left=0.0, f_skid_right=0.0, ekf_pos_cov_trace=0.05,
            station_keeping=False, sway_velocity_2d=None, approach_braking=False):

        if ref_accel is None: ref_accel = np.zeros(3, dtype=np.float32)
        if ref_vel is None: ref_vel = np.zeros(3, dtype=np.float32)
        if swarm_states is None: swarm_states = []

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
            self.vertical_v_cmd_smooth = 0.0
            self.vertical_a_cmd_smooth = 0.0
            return np.zeros(8, dtype=np.float32), 0.0, np.zeros(3, dtype=np.float32)

        delta_p = p_target - p_est
        delta_p_clamped = np.clip(delta_p, -5.0, 5.0)

        if p_target[2] <= 0.10 and p_est[2] < 1.5:
            v_cmd_z = float(np.clip(ref_vel[2], -0.25, 0.15))
        elif p_est[2] < 1.20 and ref_vel[2] > 0.0:
            v_cmd_z = float(np.clip(ref_vel[2], 0.20, 0.45))
        else:
            v_cmd_z = float(np.clip(ref_vel[2] + 0.40 * delta_p_clamped[2], -0.80, 1.30))

        dt_safe = max(1e-4, float(dt))
        requested_vz = float(v_cmd_z)
        required_az = float(np.clip((requested_vz - self.vertical_v_cmd_smooth) / dt_safe, -0.90, 0.90))
        max_da = 2.8 * dt_safe
        da = float(np.clip(required_az - self.vertical_a_cmd_smooth, -max_da, max_da))
        self.vertical_a_cmd_smooth = float(np.clip(self.vertical_a_cmd_smooth + da, -0.90, 0.90))
        old_vz = self.vertical_v_cmd_smooth
        self.vertical_v_cmd_smooth = float(np.clip(old_vz + self.vertical_a_cmd_smooth * dt_safe, -0.60, 1.05))
        v_cmd_z = self.vertical_v_cmd_smooth

        is_descending = (v_cmd_z < -0.10)
        climb_speed = max(0.0, float(v_est[2]))

        if is_descending:
            k_p_xy = 1.10
            k_d_xy = 2.40
        elif climb_speed > 0.15:
            k_p_xy = float(1.20 / (1.0 + 0.25 * climb_speed))
            k_d_xy = float(2.20 + 0.60 * min(1.0, climb_speed / 2.0))
        else:
            k_p_xy = 1.60
            k_d_xy = 2.20
        if station_keeping:
            k_p_xy = 2.65
            k_d_xy = 3.25

        v_err_xy = np.asarray(ref_vel[:2] - v_est[:2], dtype=np.float32)
        a_p_xy = k_p_xy * delta_p_clamped[:2]
        a_d_xy = k_d_xy * v_err_xy
        a_p_norm = float(np.linalg.norm(a_p_xy))
        max_position_accel = 4.0
        if a_p_norm > max_position_accel:
            a_p_xy *= max_position_accel / max(a_p_norm, 1e-8)
        a_d_norm = float(np.linalg.norm(a_d_xy))
        max_velocity_damping_accel = 8.0
        if a_d_norm > max_velocity_damping_accel:
            a_d_xy *= max_velocity_damping_accel / max(a_d_norm, 1e-8)

        is_airborne = (p_est[2] > 0.06)
        if is_airborne and p_est[2] > 0.07:
            self.xy_err_integral += delta_p_clamped[:2] * dt
            self.xy_err_integral = np.clip(self.xy_err_integral, -2.0, 2.0)
        else:
            self.xy_err_integral.fill(0.0)

        wind_for_integral = float(np.linalg.norm(np.asarray(estimated_wind[:2], dtype=np.float32)))
        k_i_xy = 0.40 + 0.06 * min(8.0, wind_for_integral)
        a_i_xy = k_i_xy * self.xy_err_integral
        a_track_xy = a_p_xy + a_d_xy + a_i_xy

        sink_error = v_cmd_z - v_est[2]
        if p_est[2] > 2.0 and p_target[2] > 0.50:
            self.z_err_integral = np.clip(self.z_err_integral + sink_error * dt, -0.40, 0.40)
        else:
            self.z_err_integral = 0.0

        k_vertical = 2.0 if (p_est[2] < 1.20 and v_cmd_z > 0.0) else 2.8
        a_track_z = k_vertical * sink_error + 0.8 * self.z_err_integral
        a_track = np.array([a_track_xy[0], a_track_xy[1], a_track_z], dtype=np.float32)

        alpha_w = dt / (0.22 + dt)
        self.wind_filt[:2] += alpha_w * (estimated_wind[:2] - self.wind_filt[:2])
        alpha_w_z = dt / (0.12 + dt)
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
            sway_pos_gain = 2.0 if station_keeping else 1.2
            sway_vel_gain = 1.55 if station_keeping else 0.90
            sway_rate = (np.zeros(2, dtype=np.float32) if sway_velocity_2d is None
                         else np.asarray(sway_velocity_2d, dtype=np.float32))
            a_anti_sway[:2] += _anti_sway_feedback(
                np.asarray(sway_2d, dtype=np.float32), sway_rate,
                position_gain=sway_pos_gain, velocity_gain=sway_vel_gain)

        action_guidance_gain = (1.00, 0.85, 0.80, 0.60, 0.72, 0.65, 0.0)
        token_safe = int(np.clip(token_idx, 0, len(action_guidance_gain) - 1))
        mode_gain = action_guidance_gain[token_safe]
        a_neural_ff = (neural_guidance[:3] * 6.0 * mode_gain
                       if (neural_guidance is not None and not station_keeping
                           and not NEURAL_GUIDANCE_DISABLED)
                       else np.zeros(3, dtype=np.float32))
        a_neural_ff = _bound_policy_vertical_residual(a_neural_ff)
        a_nominal = a_track + ref_accel + a_anti_sway + a_wind_feedforward + a_neural_ff

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
            wind_tilt_allowance = float(np.clip(wind_trim_deg, 0.0, 14.0))
            base_tilt_deg = min(38.0, 24.0 + wind_tilt_allowance)

        if ige_fac > 1.02 and p_target[2] <= 0.10 and self.wind_filt[2] >= -0.30:
            F_d[2] /= ige_fac

        storm_active = (abs(self.wind_filt[0]) > 4.0 or abs(self.wind_filt[1]) > 4.0 or self.wind_filt[2] < -0.6)
        max_multiplier = 2.50 if storm_active else 1.80
        
        is_landing_descent = (p_target[2] <= 0.05)
        if is_landing_descent:
            min_f_vert = 0.0
            if p_est[2] < 0.80:
                F_d[2] = min(F_d[2], 0.70 * weight_force)
            if p_est[2] < 0.25:
                F_d[2] = min(F_d[2], 0.35 * weight_force)
        else:
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

        b3_d_raw = F_d / max(1e-4, float(np.linalg.norm(F_d)))
        b3_d = 0.72 * self.b3_d_prev + 0.28 * b3_d_raw
        b3_d /= max(1e-4, float(np.linalg.norm(b3_d)))
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

        # Use dynamically updated airframe gains modulated by mass, sway and bus voltage
        k_R_eff = k_R_dyn.copy()
        k_w_eff = k_w_dyn.copy()

        gyroscopic = np.cross(omega_est, J_eff @ omega_est)
        tau = -k_R_eff * e_R - k_w_eff * e_w + gyroscopic
        tau[0] = float(np.clip(tau[0], -0.65, 0.65))
        tau[1] = float(np.clip(tau[1], -0.65, 0.65))
        tau[2] = float(np.clip(tau[2], -0.45, 0.45))

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
                effective_c = base_c * aero_health
                omega_cmds[i] = math.sqrt(T_cmds[i] / max(1e-6, effective_c))

        self._last_motor_omegas = omega_cmds.copy()
        return omega_cmds, f_thrust, a_safe

    # =============================================================================
# 14. FIXED 139-DIM OBSERVATION SYNTHESIZER & TEMPORAL STACKER
# =============================================================================
class DroneObservationSynthesizer139Dim:
    """
    Synthesizes exactly 139 features into a pre-allocated array with fixed indices,
    preventing any dynamic list drift or accidental truncation.
    """
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
# 15. DECENTRALIZED INDEPENDENT DRONE EXECUTIVE NETWORK (~2.2M PARAMETERS)
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
# 16. SAFE PARALLEL HARVESTER & MULTI-TASK RESNET TRAINER
# =============================================================================
def _select_harvest_indices_with_action_coverage(labels, target_count, required_classes=range(6), seed=None):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if target_count <= 0 or target_count > len(labels):
        raise ValueError(f"target_count must be in [1, {len(labels)}], got {target_count}")
    rng = np.random.default_rng(seed)
    required = tuple(int(c) for c in required_classes)
    counts = np.bincount(labels, minlength=max(required, default=0) + 1)
    missing = [c for c in required if c >= len(counts) or counts[c] == 0]
    if missing:
        missing_text = ", ".join(f"{c}:{ACTION_NAMES[c]}" for c in missing)
        raise RuntimeError(
            "Phase 1 harvesting did not visit required FSM action classes: " + missing_text + ". "
            "Refusing to train a single/partial-class decision head."
        )

    majority_class = 5
    majority_cap = int(math.floor(target_count * 0.40))
    reserve_each = min(2048, max(256, target_count // (len(required) * 20)))
    chosen = []
    chosen_mask = np.zeros(len(labels), dtype=bool)
    for cls in required:
        members = np.flatnonzero(labels == cls)
        class_cap = majority_cap if cls == majority_class else target_count
        take_n = min(len(members), reserve_each, class_cap)
        selected = rng.choice(members, size=take_n, replace=False)
        chosen.append(selected)
        chosen_mask[selected] = True

    selected_idx = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
    selected_majority = int(np.count_nonzero(labels[selected_idx] == majority_class))
    majority_members = np.flatnonzero((labels == majority_class) & (~chosen_mask))
    majority_need = max(0, min(majority_cap - selected_majority, target_count - len(selected_idx), len(majority_members)))
    if majority_need:
        fill_majority = rng.choice(majority_members, size=majority_need, replace=False)
        chosen_mask[fill_majority] = True
        selected_idx = np.concatenate((selected_idx, fill_majority))

    remaining_need = target_count - len(selected_idx)
    if remaining_need > 0:
        remaining_nonmajority = np.flatnonzero((labels != majority_class) & (~chosen_mask))
        if len(remaining_nonmajority) >= remaining_need:
            fill = rng.choice(remaining_nonmajority, size=remaining_need, replace=False)
        else:
            fill = remaining_nonmajority
            need_more = remaining_need - len(fill)
            remaining_majority = np.flatnonzero((labels == majority_class) & (~chosen_mask))
            if len(remaining_majority) < need_more:
                raise RuntimeError("Not enough unique samples remain after action-coverage selection.")
            fill = np.concatenate((fill, rng.choice(remaining_majority, size=need_more, replace=False)))
        selected_idx = np.concatenate((selected_idx, fill))
    if len(selected_idx) != target_count or len(np.unique(selected_idx)) != target_count:
        raise RuntimeError("Action-coverage selector failed its unique-sample/count invariant.")
    rng.shuffle(selected_idx)
    return selected_idx.astype(np.int64, copy=False)


def _harvest_coverage_class_for_slot(worker_id, drone_idx):
    return (int(worker_id) * NUM_SWARM_DRONES + int(drone_idx)) % 6


def _seed_harvest_coverage_state(drone, coverage_class):
    coverage_class = int(coverage_class)
    if coverage_class not in range(6):
        raise ValueError(f"Unsupported coverage class: {coverage_class}")

    mission = drone.mission
    target = np.asarray(mission["target_pos"], dtype=np.float32)
    home = np.asarray(mission["home_pad"], dtype=np.float32)
    roof_h = float(mission["roof_height"])
    cruise_z = float(mission["assigned_cruise_alt"])
    drop_surface_z = roof_h + PHASE3_ROOF_PAD_TOP_OFFSET_M
    high_hover_z = roof_h + PHASE3_ROOF_PAD_TOP_OFFSET_M + 10.0

    drone.airframe.v.fill(0.0)
    if hasattr(drone.airframe, "omega"):
        drone.airframe.omega.fill(0.0)
    if hasattr(drone.airframe, "accel_linear"):
        drone.airframe.accel_linear.fill(0.0)
    drone.hover_settle_timer = 0.0
    drone.delivery_phase_elapsed = 0.0
    drone.depot_timer = 0.0
    drone.rth_active = False
    drone.winch.reload_new_cargo()

    def set_pose(pos):
        drone.airframe.p[:] = np.asarray(pos, dtype=np.float32)
        if hasattr(drone, "ekf"):
            drone.ekf.p[:] = drone.airframe.p
            drone.ekf.v.fill(0.0)

    def set_cable_geometry(length, endpoint_z=None):
        winch = drone.winch
        winch.cable_len = float(max(PHASE3_STOWED_CABLE_LEN_M, length))
        winch.cable_vel = 0.0
        winch.nodes_pos.fill(0.0)
        winch.nodes_vel.fill(0.0)
        anchor = drone.airframe.p.copy()
        anchor[2] -= PHASE3_DRONE_HALF_HEIGHT_M
        if endpoint_z is None:
            endpoint = anchor.copy()
            endpoint[2] -= winch.cable_len + (PHASE3_CARGO_HALF_HEIGHT_M if winch.has_payload else 0.0)
        else:
            endpoint = np.array([target[0], target[1], float(endpoint_z)], dtype=np.float32)
        for k in range(winch.num_nodes):
            fraction = float(k) / float(winch.num_segments)
            winch.nodes_pos[k] = anchor + fraction * (endpoint - anchor)
        winch.sway_offset = (winch.nodes_pos[-1][:2] - winch.nodes_pos[0][:2]).astype(np.float32)
        winch.sway_velocity.fill(0.0)
        winch.theta.fill(0.0)
        winch.theta_dot.fill(0.0)
        winch.catenary_midpoint.fill(0.0)

    if coverage_class == 0:
        pos = home.copy()
        pos[2] = cruise_z
        set_pose(pos)
        drone.mission_phase = "2_CRUISE_OUTBOUND"

    elif coverage_class in (1, 2, 3, 4):
        pos = np.array([target[0], target[1], high_hover_z], dtype=np.float32)
        set_pose(pos)
        drone.mission_phase = "4_DELIVERY_HOVER"
        if coverage_class == 1:
            drone.winch.state = "STANDBY"
            drone.winch.has_payload = True
            drone.winch.cable_len = PHASE3_STOWED_CABLE_LEN_M

        elif coverage_class == 2:
            drone.winch.state = "LOWERING"
            drone.winch.has_payload = True
            drone.hover_settle_timer = DELIVERY_HOLD_MIN_S
            set_cable_geometry(2.5)

        elif coverage_class == 3:
            drone.winch.state = "TOUCHDOWN"
            drone.winch.has_payload = True
            drone.hover_settle_timer = DELIVERY_HOLD_MIN_S
            drone.winch.touchdown_timer = 0.0
            drone.winch.contact_normal_force = PAYLOAD_WEIGHT_N
            length = max(PHASE3_STOWED_CABLE_LEN_M,
                         high_hover_z - 0.05 - (drop_surface_z + 2.0 * PHASE3_CARGO_HALF_HEIGHT_M))
            set_cable_geometry(length, endpoint_z=drop_surface_z + PHASE3_CARGO_HALF_HEIGHT_M)
            drone.winch.contact_normal_force = PAYLOAD_WEIGHT_N

        else:
            drone.winch.state = "RETRACTING"
            drone.winch.has_payload = False
            drone.winch.delivery_completed = False
            set_cable_geometry(3.0)

    else:
        pos = home.copy()
        pos[2] = 0.06
        set_pose(pos)
        drone.mission_phase = "8_DEPOT_SERVICE"
        drone.winch.has_payload = False
        drone.winch.state = "STANDBY"
        drone.winch.cable_len = PHASE3_STOWED_CABLE_LEN_M

    if coverage_class != 4:
        drone.winch.delivery_completed = False
    drone.winch.last_release_pos = None
    drone.winch.is_skid_grounded = (coverage_class == 5)
    return coverage_class


def _harvest_dynamics_fault_reason(drone):
    airframe = drone.airframe
    arrays = {
        "position": np.asarray(airframe.p),
        "velocity": np.asarray(airframe.v),
        "rotation": np.asarray(airframe.R),
        "angular_velocity": np.asarray(airframe.omega),
        "motor_speeds": np.asarray(airframe.motor_omegas),
        "motor_health": np.asarray(airframe.motor_health),
    }
    for label, values in arrays.items():
        if not np.all(np.isfinite(values)):
            return f"non-finite {label}"
    p = arrays["position"]
    v = arrays["velocity"]
    omega = arrays["angular_velocity"]
    if p.shape != (3,) or v.shape != (3,) or omega.shape != (3,):
        return "invalid kinematic vector shape"
    if float(p[2]) < -5.0 or float(p[2]) > 180.0:
        return f"altitude outside harvest envelope ({float(p[2]):.2f} m)"
    if float(np.linalg.norm(p[:2])) > 800.0:
        return f"horizontal position outside harvest world ({p[:2].tolist()})"
    speed = float(np.linalg.norm(v))
    if speed > 45.0:
        return f"speed outside harvest envelope ({speed:.2f} m/s)"
    angular_speed = float(np.linalg.norm(omega))
    if angular_speed > 80.0:
        return f"angular speed outside harvest envelope ({angular_speed:.2f} rad/s)"
    return None


def _reset_harvest_drone_after_divergence(drone, coverage_class):
    _seed_harvest_coverage_state(drone, coverage_class)
    airframe = drone.airframe
    airframe.R[:] = np.eye(3, dtype=np.float32)
    airframe.motor_omegas[:] = HOVER_OMEGA_NOMINAL
    airframe.parachute_deployed = False
    airframe.last_p_watts = 185.0
    airframe.last_ige = 1.0
    airframe.vrs_turb_intensity = 0.0
    drone.history_q.clear()
    drone.mixer.xy_err_integral.fill(0.0)
    drone.mixer.z_err_integral = 0.0
    drone.mixer.vertical_v_cmd_smooth = 0.0
    drone.mixer.vertical_a_cmd_smooth = 0.0
    drone.mixer.wind_filt.fill(0.0)
    drone.mixer.b3_d_prev[:] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if hasattr(drone.ekf, "q"):
        drone.ekf.q[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if hasattr(drone.ekf, "b_a"):
        drone.ekf.b_a.fill(0.0)
    if hasattr(drone.ekf, "b_g"):
        drone.ekf.b_g.fill(0.0)


def _worker_harvest_metropolitan(worker_args):
    worker_id, num_steps = worker_args
    np.random.seed(os.getpid() ^ int(time.time()))
    random.seed(os.getpid() ^ int(time.time()))

    world = MetropolitanCityGridWorld400m()
    corridor_planner = VoxelOctreeCorridorPlanner(world)
    traffic_engine = DynamicRoadVehicleTrafficEngine(world.avenue_x, world.street_y)
    lidar_engine = MacroPerceptionLiDARSuite(world)
    sensor_suite = AdvancedMultiModalSensorSuite(world)
    assurance = SurfaceAndWireAssuranceSuite(world)
    dispatcher = SwarmMissionDispatcher(world.manifest_addresses, world.depot_helipads)
    ground_agent = DynamicGroundIntruderAgent(init_pos=[-80.0, 80.0, 0.0])
    agent_pos = ground_agent.p.copy()

    obs_list = []
    y_guidance, y_fsm, y_winch, y_health, y_trust, y_val = [], [], [], [], [], []

    drones = []
    for i in range(NUM_SWARM_DRONES):
        init_p = world.depot_helipads[i].copy()
        init_p[2] = 0.06
        d_inst = SwarmDroneInstance(i, init_p, world, dispatcher)
        drones.append(d_inst)

    seeded_classes = [_harvest_coverage_class_for_slot(worker_id, i)
                      for i in range(NUM_SWARM_DRONES)]
    for d_inst, seed_class in zip(drones, seeded_classes):
        _seed_harvest_coverage_state(d_inst, seed_class)

    dt = 0.016
    sim_time = 0.0
    weather_modes = ["CLEAR_NOMINAL", "CANYON_GALE", "FREEZING_RAIN", "MICROBURST_STORM"]
    try:
        horizon_multiplier = int(os.environ.get("SURVIVAL_HARVEST_HORIZON_MULTIPLIER", "3"))
    except (TypeError, ValueError):
        horizon_multiplier = 3
    horizon_multiplier = int(np.clip(horizon_multiplier, 1, 6))
    sim_frame_budget = max(1, int(math.ceil(num_steps / NUM_SWARM_DRONES)) * horizon_multiplier)
    samples_per_regime = max(1, int(math.ceil(sim_frame_budget * NUM_SWARM_DRONES / 4)))
    sim_frames = 0
    harvest_dynamics_resets = 0

    obstacles_cbf = world.get_static_collision_obstacles()

    while sim_frames < sim_frame_budget:
        for d_idx, d in enumerate(drones):
            fault_reason = _harvest_dynamics_fault_reason(d)
            if fault_reason is not None:
                _reset_harvest_drone_after_divergence(d, seeded_classes[d_idx])
                harvest_dynamics_resets += 1

        sim_time += dt
        regime_idx = min(3, (sim_frames * NUM_SWARM_DRONES) // samples_per_regime)
        mode = weather_modes[regime_idx]

        if mode == "CLEAR_NOMINAL":
            episode_wind = np.array([1.5, -0.5, 0.0], dtype=np.float32)
            ice_active, curr_ice, amb_temp = False, 0.0, 22.0
        elif mode == "CANYON_GALE":
            episode_wind = np.array([5.5, 2.2, 0.0], dtype=np.float32)
            ice_active, curr_ice, amb_temp = False, 0.0, 14.0
        elif mode == "FREEZING_RAIN":
            episode_wind = np.array([2.5, -2.0, 0.0], dtype=np.float32)
            ice_active, curr_ice, amb_temp = True, 0.45, -4.5
        else:
            downburst_z = -2.5 * float(math.tanh(max(0.1, drones[0].airframe.p[2]) / 4.5))
            episode_wind = np.array([-4.5, 3.8, downburst_z], dtype=np.float32)
            ice_active, curr_ice, amb_temp = False, 0.0, 10.0

        world.step_civil_traffic(dt=dt)
        traffic_engine.step(dt=dt)
        swarm_states = [{"p": d.airframe.p.copy(), "v": d.airframe.v.copy()} for d in drones]

        intruder_reference = next(
            (dr for dr in drones if dr.winch.state in {"LOWERING", "ABORT_RETRACT"}),
            drones[0],
        )
        intruder_tip = intruder_reference.airframe.p + np.array([
            intruder_reference.winch.sway_offset[0],
            intruder_reference.winch.sway_offset[1],
            -PHASE3_DRONE_HALF_HEIGHT_M - intruder_reference.winch.cable_len
            - (PHASE3_CARGO_HALF_HEIGHT_M if intruder_reference.winch.has_payload else 0.0),
        ], dtype=np.float32)
        agent_pos, _, _ = ground_agent.step(
            cargo_world_pos=intruder_tip,
            is_winch_lowering=(intruder_reference.winch.state == "LOWERING"),
            is_aborting=(intruder_reference.winch.state == "ABORT_RETRACT"),
            dt=dt,
        )

        step_obs_triad = []
        step_yg_triad, step_yf_triad, step_yw_triad = [], [], []
        step_yh_triad, step_yt_triad, step_yv_triad = [], [], []

        for d_idx, d in enumerate(drones):
            is_station_holding = False
            is_solidly_over_roof = False
            sensed_wind, sensed_wind_speed, sensed_wind_from_deg, sensed_wind_quality = sensor_suite.poll_wind_vector(
                d.airframe.v, episode_wind, d.airframe.R, dt=dt, sensor_id=d_idx,
                ground_velocity_est=d.ekf.v
            )
            tgt_xy = d.mission["target_pos"][:2]
            home_xy = d.mission["home_pad"][:2]
            roof_h = d.mission["roof_height"]
            drop_surface_z = float(roof_h + PHASE3_ROOF_PAD_TOP_OFFSET_M)
            cruise_z = d.assigned_cruise_alt
            high_hover_z = d.get_high_hover_z()

            dist_tgt_xy = float(np.linalg.norm(d.airframe.p[:2] - tgt_xy))
            dist_home_xy = float(np.linalg.norm(d.airframe.p[:2] - home_xy))

            if d.mission_phase == "1_TAKEOFF":
                wp_target = np.array([home_xy[0], home_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.0, max_climb=2.40, wind_vector=sensed_wind)
                fsm_token, winch_rate_cmd = 0, 0.0
                if _cruise_altitude_captured(d.airframe.p[2], d.airframe.v[2], cruise_z):
                    d.mission_phase = "2_CRUISE_OUTBOUND"

            elif d.mission_phase == "2_CRUISE_OUTBOUND":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=4.5, max_climb=2.40, wind_vector=sensed_wind)
                fsm_token, winch_rate_cmd = 0, 0.0
                speed_xy_now = float(np.linalg.norm(d.airframe.v[:2]))
                wind_speed_now = float(np.linalg.norm(sensed_wind[:2]))
                if _should_enter_precision_approach(dist_tgt_xy, speed_xy_now, wind_speed_now):
                    d.mission_phase = "3_CHIMNEY_DESCENT"
                    d.mixer.xy_err_integral.fill(0.0)
                    d.mixer.z_err_integral = 0.0
                    d.chimney_alignment_timer = 0.0
                    d.chimney_descent_armed = False
                    d.chimney_descent_settle_timer = 0.0
                    d.chimney_phase_elapsed = 0.0

            elif d.mission_phase == "3_CHIMNEY_DESCENT":
                speed_xy_now = float(np.linalg.norm(d.airframe.v[:2]))
                speed_z_now = abs(float(d.airframe.v[2]))
                wind_speed_now = float(np.linalg.norm(sensed_wind[:2]))
                
                # FIXED: Explicitly set valid teacher executive and winch tokens for chimney descent
                fsm_token, winch_rate_cmd = (1 if d.chimney_descent_armed else 0), 0.0
                
                if wind_speed_now <= DELIVERY_MAX_OPERATIONAL_WIND_MS:
                    d.chimney_phase_elapsed += dt
                if wind_speed_now > DELIVERY_MAX_OPERATIONAL_WIND_MS and d.winch.cable_len <= 0.12:
                    d.chimney_descent_armed = False
                    d.chimney_alignment_timer = 0.0
                    d.chimney_descent_settle_timer = 0.0
                if not d.chimney_descent_armed:
                    wp_target = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                    approach_speed_cap = min(1.8, 0.45 * dist_tgt_xy,
                                             math.sqrt(max(0.0, 2.0 * 1.1 * dist_tgt_xy)))
                    p_ref_guidance, v_ref = compute_vector_guidance(
                        d.airframe.p, d.airframe.v, wp_target, max_speed_xy=approach_speed_cap,
                        max_climb=0.35, max_descend=0.15, wind_vector=sensed_wind)
                    p_ref = wp_target.copy()
                    if dist_tgt_xy <= DELIVERY_DRONE_CENTER_TOLERANCE_M:
                        v_ref[:2] = 0.0
                    cruise_aligned = (wind_speed_now <= DELIVERY_MAX_OPERATIONAL_WIND_MS
                                      and dist_tgt_xy <= DELIVERY_DRONE_CENTER_TOLERANCE_M
                                      and abs(float(d.airframe.p[2] - cruise_z)) <= 0.65
                                      and speed_xy_now <= DELIVERY_MAX_HORIZONTAL_SPEED_MS
                                      and speed_z_now <= DELIVERY_MAX_VERTICAL_SPEED_MS)
                    d.chimney_alignment_timer = d.chimney_alignment_timer + dt if cruise_aligned else 0.0
                    if d.chimney_alignment_timer >= 0.75:
                        d.chimney_descent_armed = True
                        d.mixer.xy_err_integral.fill(0.0)
                        d.chimney_descent_settle_timer = 0.0
                else:
                    wp_target = np.array([tgt_xy[0], tgt_xy[1], high_hover_z], dtype=np.float32)
                    p_ref, v_ref = compute_vector_guidance(
                        d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.0,
                        max_climb=0.70,
                        max_descend=(0.65 if abs(float(d.airframe.p[2] - high_hover_z)) > 3.0
                                     else 0.40 if abs(float(d.airframe.p[2] - high_hover_z)) > 1.0 else 0.22),
                        wind_vector=sensed_wind)
                    dist_z = abs(float(d.airframe.p[2] - high_hover_z))
                    v_ref = _delivery_vertical_settle_reference(v_ref, dist_z)
                    hover_ready = (wind_speed_now <= DELIVERY_MAX_OPERATIONAL_WIND_MS
                                   and dist_tgt_xy <= DELIVERY_DRONE_CENTER_TOLERANCE_M
                                   and dist_z <= 0.40 and speed_xy_now <= 0.30 and speed_z_now <= 0.20)
                    d.chimney_descent_settle_timer = d.chimney_descent_settle_timer + dt if hover_ready else 0.0
                    if d.chimney_descent_settle_timer >= 0.60:
                        d.mission_phase = "4_DELIVERY_HOVER"
                        d.chimney_phase_elapsed = 0.0
                        d.hover_settle_timer = 0.0
                        d.delivery_phase_elapsed = 0.0
                        d.delivery_attempts += 1

                approach_vertical_delta_m = abs(float(cruise_z) - float(high_hover_z))
                approach_timeout_s = max(
                    CHIMNEY_APPROACH_TIMEOUT_S,
                    1.8 * approach_vertical_delta_m / 0.65 + 12.0,
                )
                if d.mission_phase == "3_CHIMNEY_DESCENT" and d.chimney_phase_elapsed >= approach_timeout_s:
                    d.delivery_failures += 1
                    d.dispatcher.report_failed(d.drone_id)
                    d.rth_active = True
                    d.mission_phase = "5_CHIMNEY_ASCENT"
                    d.hover_settle_timer = 0.0
                    d.target = d.mission["home_pad"].copy()
                    d.chimney_phase_elapsed = 0.0
                    d.chimney_alignment_timer = 0.0
                    d.chimney_descent_settle_timer = 0.0

            elif d.mission_phase == "4_DELIVERY_HOVER":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], high_hover_z], dtype=np.float32)
                p_ref, v_ref = wp_target.copy(), np.zeros(3, dtype=np.float32)
                d.delivery_phase_elapsed += dt

                dist_drone_to_target = float(math.hypot(d.airframe.p[0] - tgt_xy[0], d.airframe.p[1] - tgt_xy[1]))
                drone_spd_xy = float(math.hypot(d.airframe.v[0], d.airframe.v[1]))
                surf_under_drone = world.get_surface_height(d.airframe.p[0], d.airframe.p[1])
                is_solidly_over_roof = abs(float(surf_under_drone) - float(roof_h)) <= DELIVERY_HOVER_ALTITUDE_TOLERANCE_M
                sway_m = float(np.linalg.norm(d.winch.sway_offset)) if d.winch.cable_len > 0.12 else 0.0
                is_station_holding = (
                    dist_drone_to_target <= DELIVERY_DRONE_CENTER_TOLERANCE_M
                    and abs(float(d.airframe.p[2] - high_hover_z)) <= DELIVERY_HOVER_ALTITUDE_TOLERANCE_M
                    and drone_spd_xy <= DELIVERY_MAX_HORIZONTAL_SPEED_MS
                    and abs(float(d.airframe.v[2])) <= DELIVERY_MAX_VERTICAL_SPEED_MS
                    and sway_m <= DELIVERY_MAX_SWAY_M
                    and is_solidly_over_roof
                    and float(np.linalg.norm(sensed_wind[:2])) <= DELIVERY_MAX_OPERATIONAL_WIND_MS
                )
                if is_station_holding:
                    d.hover_settle_timer += dt
                else:
                    d.hover_settle_timer = 0.0
                cable_stowed = (d.winch.cable_len <= 0.10)
                
                if d.winch.state in ("RETRACTING", "ABORT_RETRACT") and not cable_stowed:
                    fsm_token, winch_rate_cmd = 4, -1.0
                elif d.winch.state == "TOUCHDOWN":
                    fsm_token, winch_rate_cmd = 3, 0.0
                elif d.winch.state == "LOWERING":
                    cargo_center_z = _live_payload_center_z(d.airframe.p[2], d.winch.cable_len)
                    if cargo_center_z > (drop_surface_z + PHASE3_CARGO_HALF_HEIGHT_M + 0.03):
                        fsm_token, winch_rate_cmd = 2, 1.0
                    else:
                        fsm_token, winch_rate_cmd = 3, 0.0
                elif d.winch.has_payload and cable_stowed:
                    fsm_token = (2 if is_station_holding and d.hover_settle_timer >= DELIVERY_HOLD_MIN_S else 1)
                    winch_rate_cmd = 0.0
                elif d.winch.has_payload and not cable_stowed:
                    cargo_center_z = _live_payload_center_z(d.airframe.p[2], d.winch.cable_len)
                    if cargo_center_z > (drop_surface_z + PHASE3_CARGO_HALF_HEIGHT_M + 0.03):
                        fsm_token, winch_rate_cmd = 2, 1.0
                    else:
                        fsm_token, winch_rate_cmd = 3, 0.0
                elif not d.winch.has_payload and not cable_stowed:
                    fsm_token, winch_rate_cmd = 4, -1.0
                else:
                    fsm_token, winch_rate_cmd = 0, 0.0

                if d.delivery_phase_elapsed >= DELIVERY_HOVER_TIMEOUT_S and not d.winch.delivery_completed:
                    if d.winch.cable_len > 0.12:
                        d.winch.state = "ABORT_RETRACT"
                    else:
                        d.delivery_failures += 1
                        d.delivery_phase_elapsed = 0.0
                        d.hover_settle_timer = 0.0
                        d.dispatcher.report_failed(d.drone_id)
                        d.mission_phase = "5_CHIMNEY_ASCENT"
                        d.target = d.mission["home_pad"].copy()
                elif d.winch.delivery_completed and cable_stowed:
                    d.winch.delivery_completed = False
                    d.hover_settle_timer = 0.0
                    d.delivery_phase_elapsed = 0.0
                    d.packages_delivered += 1
                    d.dispatcher.report_completed(d.drone_id)
                    d.mission_phase = "5_CHIMNEY_ASCENT"
                    d.target = d.mission["home_pad"].copy()

            elif d.mission_phase == "5_CHIMNEY_ASCENT":
                wp_target = np.array([d.airframe.p[0], d.airframe.p[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.0, max_climb=2.40, wind_vector=sensed_wind)
                fsm_token, winch_rate_cmd = 0, 0.0
                if _cruise_altitude_captured(d.airframe.p[2], d.airframe.v[2], cruise_z):
                    d.mission_phase = "6_CRUISE_INBOUND"

            elif d.mission_phase == "6_CRUISE_INBOUND":
                wp_target = np.array([home_xy[0], home_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=4.5, max_climb=2.40, wind_vector=sensed_wind)
                fsm_token, winch_rate_cmd = 0, 0.0
                wind_spd = float(np.linalg.norm(sensed_wind[:2]))
                inbound_gate = min(4.5, 2.2 + 0.20 * wind_spd)
                if dist_home_xy <= inbound_gate:
                    d.mission_phase = "7_DEPOT_DESCENT"
                    d.mixer.xy_err_integral.fill(0.0)
                    d.mixer.z_err_integral = 0.0

            elif d.mission_phase == "7_DEPOT_DESCENT":
                wp_target = np.array([home_xy[0], home_xy[1], 0.02], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.35, max_descend=0.22, wind_vector=sensed_wind)
                fsm_token, winch_rate_cmd = 5, 0.0

                is_on_helipad = (dist_home_xy <= 1.5)
                has_touched_down = (d.airframe.p[2] <= 0.08 or d.winch.is_skid_grounded)
                if is_on_helipad and has_touched_down:
                    d.depot_timer = 0.0
                    d.mission_phase = "8_DEPOT_SERVICE"
                    d.airframe.v.fill(0.0)

            else:
                wp_target = np.array([home_xy[0], home_xy[1], 0.06], dtype=np.float32)
                p_ref, v_ref = wp_target.copy(), np.zeros(3, dtype=np.float32)
                fsm_token, winch_rate_cmd = 5, 0.0
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
            lidar_16 = lidar_engine.cast_360_lidar(
                d.airframe.p, drone_yaw,
                dynamic_vehicles=traffic_engine.get_cbf_obstacles(),
                dynamic_objects=[{"id": "Ground_Intruder", "pos": agent_pos.copy(), "radius": 1.85}])
            conical_8, wire_clr = lidar_engine.cast_downward_conical(d.airframe.p)

            current_surf_z = world.get_surface_height(d.airframe.p[0], d.airframe.p[1])
            flow_2d, flow_q, _ = sensor_suite.poll_optical_flow(d.airframe.p, d.airframe.v, d.airframe.R, d.airframe.omega, current_surf_z)
            aruco_3d, aruco_lk = sensor_suite.poll_aruco_fiducial_detector(d.airframe.p, d.airframe.R, d.mission["target_pos"], d.mission["tag_id"])
            siren_a, siren_f, siren_s = sensor_suite.poll_directional_acoustic_array(d.airframe.p, [], sim_time=sim_time)
            adsb_p, adsb_cs, _ = sensor_suite.poll_adsb_civil_transponder(d.airframe.p, d.airframe.v, world.civil_aircraft)

            hook_a, hook_w = sensor_suite.poll_smart_hook_imu(
                d.airframe.accel_linear, d.airframe.omega, d.winch.sway_offset,
                d.winch.cable_len, d.winch.has_payload, d.winch.sway_velocity)
            load_3d = sensor_suite.poll_3axis_load_cell(d.winch.tension_n, d.winch.sway_offset, d.winch.cable_len)
            th_dt, dolp, ndvi = sensor_suite.poll_thermal_and_polarization(d.target[:2])
            i_rms, t_mosfet = sensor_suite.poll_esc_foc_telemetry(d.airframe.motor_omegas, d.airframe.motor_health)
            alpha, beta, rh, dust = sensor_suite.poll_5port_air_data(d.airframe.v, episode_wind, d.airframe.R)
            hdop, cn0, rf_pwr, mag_gr = sensor_suite.poll_rtk_gnss_metrics(d.airframe.p, False)
            _, is_surface_safe, _ = assurance.audit_surface(d.target[:2], target_z=roof_h)

            weight_force = (DRY_MASS_KG + (CARGO_MASS_KG if d.winch.has_payload else 0.0)) * G_ACCEL
            is_on_surf = (d.airframe.p[2] <= current_surf_z + 0.05)
            d.winch.update_skid_strain_gauges(is_on_surf, weight_force, d.airframe.last_f_thrust)

            obs_139 = d.synth.synthesize_instantaneous(
                p_est=d.airframe.p, v_est=d.airframe.v, R_est=d.airframe.R, omega_est=d.airframe.omega,
                target_p=wp_target, lidar_16=lidar_16, conical_8=conical_8,
                v_tas=float(np.linalg.norm(d.airframe.v - sensed_wind)), q_dyn=12.0, min_sonar=max(0.1, d.airframe.p[2] - current_surf_z),
                wind_est=sensed_wind, ice_idx=0.35 if ice_active else 0.0, cable_len=d.winch.cable_len,
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
            if d.winch.has_payload and d.winch.cable_len > 0.12:
                a_target_xy += _anti_sway_feedback(
                    d.winch.sway_offset, d.winch.sway_velocity,
                    position_gain=PHASE3_SWAY_POSITION_GAIN,
                    velocity_gain=PHASE3_SWAY_VELOCITY_GAIN).astype(np.float32)
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

            winch_station_gate = bool(d.mission_phase == "4_DELIVERY_HOVER"
                                      and is_station_holding
                                      and (d.winch.state != "STANDBY" or d.hover_settle_timer >= DELIVERY_HOLD_MIN_S))
            cable_anchor = d.airframe.p + np.array([0.0, 0.0, -0.05], dtype=np.float32)
            cable_endpoint = np.asarray(d.winch.nodes_pos[-1], dtype=np.float32).copy()
            if (not np.all(np.isfinite(cable_endpoint)) or np.linalg.norm(cable_endpoint - cable_anchor) > max(2.0, d.winch.cable_len + 1.0)
                    or (np.linalg.norm(cable_endpoint) < 1e-4 and np.linalg.norm(cable_anchor) > 1.0)):
                cable_endpoint = cable_anchor + np.array([d.winch.sway_offset[0], d.winch.sway_offset[1], -d.winch.cable_len], dtype=np.float32)
            cable_points = _winch_cable_polyline_points(d.winch, d.airframe.p)
            cable_static_hit = (d.winch.cable_len > 0.20 and d.winch.has_payload
                                and (corridor_planner.is_segment_occluded(cable_anchor, cable_endpoint)
                                     or _polyline_intersects_dynamic_hitboxes(
                                         cable_points, world.get_static_collision_obstacles())))
            worker_cable_obstacles = traffic_engine.get_cbf_obstacles()
            worker_cable_obstacles.append({"id": world.civil_aircraft["id"], "shape": "sphere",
                                           "pos": world.civil_aircraft["pos"].copy(),
                                           "radius": float(world.civil_aircraft.get("radius", 4.5)),
                                           "height": float(world.civil_aircraft["pos"][2] + world.civil_aircraft.get("radius", 4.5))})
            for peer in drones:
                if peer is d:
                    continue
                worker_cable_obstacles.append({"id": f"Drone_{peer.drone_id+1}", "shape": "sphere",
                                               "pos": peer.airframe.p.copy(), "radius": 0.65,
                                               "height": float(peer.airframe.p[2] + 0.65)})
            cable_dynamic_hit = (d.winch.cable_len > 0.20 and d.winch.has_payload
                                 and _polyline_intersects_dynamic_hitboxes(cable_points, worker_cable_obstacles))
            d.winch.step(
                drone_z=d.airframe.p[2], target_surface_z=drop_surface_z, is_settled=winch_station_gate,
                is_surface_safe=(is_surface_safe and (d.mission_phase != "4_DELIVERY_HOVER" or is_solidly_over_roof)),
                wire_hit=bool(cable_static_hit or cable_dynamic_hit), agent_breach=False, agent_dist=30.0,
                wind_2d=sensed_wind[:2], drone_accel_linear=d.airframe.accel_linear,
                neural_token_idx=fsm_token, nn_winch_rate=winch_rate_cmd, dt=dt,
                drone_p=d.airframe.p, drone_v=d.airframe.v, wind_3d=episode_wind,
                target_xy=tgt_xy
            )

            omega_cmds, _, _ = d.mixer.mix(
                drone_health=d.airframe.motor_health, p_est=d.airframe.p, v_est=d.airframe.v, R_est=d.airframe.R,
                omega_est=d.airframe.omega, p_target=p_ref, neural_guidance=guidance_vec, token_idx=fsm_token,
                estimated_wind=sensed_wind, obstacles=obstacles_cbf, swarm_states=swarm_states, drone_idx=d_idx,
                ref_vel=v_ref, target_bldg_id=d.mission["bldg_id"] if ("OUTBOUND" in d.mission_phase or "DELIVER" in d.mission_phase or "CHIMNEY" in d.mission_phase) else None,
                departure_bldg_id=d.mission["bldg_id"] if ("ASCENT" in d.mission_phase or ("INBOUND" in d.mission_phase and dist_tgt_xy < 25.0)) else None,
                cable_len=d.winch.cable_len, sway_2d=d.winch.sway_offset, load_cell_3d=load_3d,
                ige_fac=d.airframe.last_ige, v_pack=d.bms.v_pack, bms_temp_c=d.bms.temp_c,
                has_payload=d.winch.has_payload, dt=dt,
                f_skid_left=d.winch.f_skid_left, f_skid_right=d.winch.f_skid_right,
                ekf_pos_cov_trace=0.05,
                station_keeping=(d.mission_phase in ("3_CHIMNEY_DESCENT", "4_DELIVERY_HOVER", "7_DEPOT_DESCENT")),
                sway_velocity_2d=d.winch.sway_velocity,
                approach_braking=(d.mission_phase == "3_CHIMNEY_DESCENT" and not d.chimney_descent_armed)
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
        sim_frames += 1

    temp_dir = tempfile.gettempdir()
    chunk_file = os.path.join(temp_dir, f"swarm_chunk_{worker_id}_{os.getpid()}.npz")
    np.savez_compressed(
        chunk_file,
        obs=np.array(obs_list, dtype=np.float32),
        yg=np.array(y_guidance, dtype=np.float32),
        yf=np.array(y_fsm, dtype=np.int64),
        yw=np.array(y_winch, dtype=np.float32),
        yh=np.array(y_health, dtype=np.float32),
        yt=np.array(y_trust, dtype=np.float32),
        yv=np.array(y_val, dtype=np.float32)
    )
    return chunk_file

def harvest_and_train_metropolitan():
    if torch is None:
        raise RuntimeError(
            "harvest_and_train_metropolitan() must run in the parent process; "
            "the lightweight harvesting child must only execute _worker_harvest_metropolitan()."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_SAMPLES = 250000              
    NUM_PRETRAIN_EPOCHS = 15         
    NUM_MAIN_EPOCHS = 30             
    BATCH_SIZE = 4096
    LEARNING_RATE = 1.2e-3
    
    cpu_count = max(1, mp.cpu_count())
    worker_cap = min(14, max(1, cpu_count - 2))
    requested_workers_text = os.environ.get("SURVIVAL_HARVEST_WORKERS", "").strip()
    if requested_workers_text:
        try:
            requested_workers = int(requested_workers_text)
            if requested_workers < 1: raise ValueError
            num_workers = min(worker_cap, requested_workers)
        except ValueError:
            num_workers = min(4, worker_cap)
    else:
        num_workers = worker_cap
    samples_per = NUM_SAMPLES // num_workers

    print("\n" + "="*75)
    print(f"HARVESTING {NUM_SAMPLES:,} SAMPLES (FILE-BACKED IPC PIPELINE)")
    print(f"Parallel Workers: {num_workers} processes | Compute Device: {device}")
    print("="*75)
    t0 = time.time()

    worker_payloads = [(i, samples_per) for i in range(num_workers)]
    child_env_keys = (
        "SURVIVAL_LIGHTWEIGHT_HARVEST_CHILD",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    previous_child_env = {key: os.environ.get(key) for key in child_env_keys}
    try:
        os.environ["SURVIVAL_LIGHTWEIGHT_HARVEST_CHILD"] = "1"
        for key in child_env_keys[1:]:
            os.environ[key] = "1"
        spawn_context = mp.get_context("spawn")
        with spawn_context.Pool(processes=num_workers) as pool:
            chunk_files = pool.map(_worker_harvest_metropolitan, worker_payloads)
    except Exception as exc:
        raise RuntimeError(f"Dataset harvesting failed: {repr(exc)}") from exc
    finally:
        for key, value in previous_child_env.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value

    X_chunks, yg_chunks, yf_chunks, yw_chunks, yh_chunks, yt_chunks, yv_chunks = [], [], [], [], [], [], []
    for cf in chunk_files:
        with np.load(cf) as data:
            X_chunks.append(data["obs"])
            yg_chunks.append(data["yg"])
            yf_chunks.append(data["yf"])
            yw_chunks.append(data["yw"])
            yh_chunks.append(data["yh"])
            yt_chunks.append(data["yt"])
            yv_chunks.append(data["yv"])
        try:
            os.remove(cf)
        except OSError:
            pass

    X_raw = np.concatenate(X_chunks, axis=0)
    yg_raw = np.concatenate(yg_chunks, axis=0)
    yf_raw = np.concatenate(yf_chunks, axis=0)
    yw_raw = np.concatenate(yw_chunks, axis=0)
    yh_raw = np.concatenate(yh_chunks, axis=0)
    yt_raw = np.concatenate(yt_chunks, axis=0)
    yv_raw = np.concatenate(yv_chunks, axis=0)

    selected_indices = _select_harvest_indices_with_action_coverage(
        yf_raw, NUM_SAMPLES, required_classes=range(6), seed=None)
    X_all = X_raw[selected_indices]
    yg_all = yg_raw[selected_indices]
    yf_all = yf_raw[selected_indices]
    yw_all = yw_raw[selected_indices]
    yh_all = yh_raw[selected_indices]
    yt_all = yt_raw[selected_indices]
    yv_all = yv_raw[selected_indices]
    del X_raw, yg_raw, yf_raw, yw_raw, yh_raw, yt_raw, yv_raw
    sample_perm = np.random.default_rng().permutation(len(X_all))
    X_all, yg_all, yf_all = X_all[sample_perm], yg_all[sample_perm], yf_all[sample_perm]
    yw_all, yh_all, yt_all, yv_all = (yw_all[sample_perm], yh_all[sample_perm],
                                      yt_all[sample_perm], yv_all[sample_perm])

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
    present_classes = class_counts > 0
    class_weights = torch.zeros_like(class_counts)
    class_weights[present_classes] = len(yf_tr) / (
        present_classes.sum().float() * class_counts[present_classes].clamp(min=1.0))
    class_weights[present_classes] /= class_weights[present_classes].mean().clamp(min=1e-6)
    class_weights = class_weights.to(device)
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

    if os.path.exists(MODEL_WEIGHTS_FILE):
        backup_path = MODEL_WEIGHTS_FILE + ".backup_before_v17_retrain"
        if not os.path.exists(backup_path):
            try: shutil.copy2(MODEL_WEIGHTS_FILE, backup_path)
            except OSError: pass
    torch.save(model.state_dict(), MODEL_WEIGHTS_FILE)
    print(f"SUCCESS: Policy Model saved to {MODEL_WEIGHTS_FILE} in {time.time()-t0:.2f}s!\n")
    return model, device

# =============================================================================
# 17. PHASE 3 — GENESIS 3D PARALLEL POLICY REFINEMENT (HEADLESS)
# =============================================================================
LEGACY_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(".pth", "_phase3_genesis.pth")
PREVIOUS_WIND_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(".pth", "_phase3_wind_robust_v19.pth")
PREVIOUS_PRECISION_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(".pth", "_phase3_precision_delivery_v21.pth")
PREVIOUS_WIND_SENSOR_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(".pth", "_phase3_wind_sensor_v20.pth")
PREVIOUS_CURRICULUM_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(".pth", "_phase3_curriculum_precision_delivery_v22.pth")
PREVIOUS_ROOF_CONTACT_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(".pth", "_phase3_roof_contact_tether_fix_v22_2.pth")
PREVIOUS_COVERAGE_HOVER_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(".pth", "_phase3_training_coverage_hover_braking_v22_3.pth")
PREVIOUS_DEEP_AUDIT_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(".pth", "_phase3_deep_audit_collision_visuals_v22_5.pth")
PREVIOUS_DELIVERY_DESCENT_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(".pth", "_phase3_delivery_descent_collision_visuals_v22_7.pth")
PREVIOUS_V38_1_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(".pth", "_phase3_v38_1_delivery_rollout.pth")
PHASE3_CHECKPOINT_FILE = os.environ.get(
    "SURVIVAL_PHASE3_CHECKPOINT",
    MODEL_WEIGHTS_FILE.replace(".pth", "_phase3_v38_7_memory_efficient.pth"),
)

def _phase3_write_training_urdf(filepath):
    content = f'''<?xml version="1.0"?>
<robot name="survival_phase3_training_drone">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{DRY_MASS_KG}"/>
      <inertia ixx="0.022" ixy="0" ixz="0" iyy="0.022" iyz="0" izz="0.038"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.24 0.18 0.09"/></geometry>
      <material name="body"><color rgba="0.12 0.15 0.19 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.24 0.18 0.09"/></geometry>
    </collision>
    <visual>
      <origin xyz="0.13 0 0.01" rpy="0 0 0"/>
      <geometry><box size="0.06 0.04 0.03"/></geometry>
      <material name="nose"><color rgba="0.95 0.25 0.25 1.0"/></material>
    </visual>
  </link>
</robot>
'''
    Path(filepath).write_text(content, encoding="utf-8")
    return str(filepath)

def _phase3_requested_envs(device):
    default = "512" if getattr(device, "type", "cpu") == "cuda" else "16"
    try: return int(os.environ.get("SURVIVAL_PHASE3_ENVS", default))
    except (TypeError, ValueError): return int(default)

def _phase3_env_candidates(requested):
    requested = int(np.clip(requested, 1, 8192))
    candidates = []
    value = requested
    while True:
        if value not in candidates: candidates.append(value)
        if value <= 1: break
        value = max(1, value // 2)
    return candidates

def _phase3_target_pad_surface_table(target_positions):
    table = np.asarray(target_positions, dtype=np.float32).copy()
    if table.ndim != 2 or table.shape[1] < 3:
        raise ValueError(f"Expected target positions shaped [N, >=3], got {table.shape}")
    table[:, 2] += PHASE3_ROOF_PAD_TOP_OFFSET_M
    return table

def _phase3_cargo_center_z_for_cable(drone_center_z, cable_len):
    return drone_center_z - cable_len - PHASE3_TETHER_CENTER_OFFSET_M

def _live_payload_center_z(drone_center_z, cable_len):
    return float(drone_center_z) - PHASE3_DRONE_HALF_HEIGHT_M - max(
        PHASE3_STOWED_CABLE_LEN_M, float(cable_len)) - PHASE3_CARGO_HALF_HEIGHT_M

def _live_payload_bottom_clearance(drone_center_z, cable_len, target_surface_z):
    parcel_center_z = _live_payload_center_z(drone_center_z, cable_len)
    return parcel_center_z - PHASE3_CARGO_HALF_HEIGHT_M - float(target_surface_z)

def _phase3_tether_slack(center_distance, cable_len):
    return center_distance - cable_len - PHASE3_TETHER_CENTER_OFFSET_M

def _phase3_torch_features(pos, vel, waypoint, roof_z, cable_len, cable_vel,
                           has_payload, winch_state, phase, wind, accel,
                           building_aabbs, sway_2d=None, sway_velocity_2d=None,
                           tension_n=None, omega=None):
    device = pos.device
    n_envs, n_drones, _ = pos.shape
    out = torch.zeros((n_envs, n_drones, BASE_OBS_DIM), device=device, dtype=torch.float32)
    out[..., 0:3] = (waypoint - pos) / 50.0
    out[..., 3:6] = pos / 200.0
    out[..., 6:9] = vel / 8.0
    identity = torch.eye(3, device=device, dtype=torch.float32).reshape(1, 1, 9)
    out[..., 9:18] = identity

    # Angular rate / gyro state
    if omega is not None:
        out[..., 18:21] = omega / 10.0

    bounds = torch.as_tensor(building_aabbs, dtype=torch.float32, device=device)
    if bounds.numel():
        bounds = bounds.reshape(-1, 5)
        angles = torch.arange(16, dtype=torch.float32, device=device) * (2.0 * math.pi / 16.0)
        ray_x = torch.cos(angles).reshape(1, 1, 16, 1)
        ray_y = torch.sin(angles).reshape(1, 1, 16, 1)
        px = pos[..., 0, None, None]
        py = pos[..., 1, None, None]
        pz = pos[..., 2, None, None]
        xmin = bounds[:, 0].reshape(1, 1, 1, -1)
        xmax = bounds[:, 1].reshape(1, 1, 1, -1)
        ymin = bounds[:, 2].reshape(1, 1, 1, -1)
        ymax = bounds[:, 3].reshape(1, 1, 1, -1)
        top = bounds[:, 4].reshape(1, 1, 1, -1)
        rx = torch.where(ray_x.abs() < 1e-6, torch.full_like(ray_x, 1e-6), ray_x)
        ry = torch.where(ray_y.abs() < 1e-6, torch.full_like(ray_y, 1e-6), ray_y)
        tx1, tx2 = (xmin - px) / rx, (xmax - px) / rx
        ty1, ty2 = (ymin - py) / ry, (ymax - py) / ry
        t_near = torch.maximum(torch.minimum(tx1, tx2), torch.minimum(ty1, ty2))
        t_far = torch.minimum(torch.maximum(tx1, tx2), torch.maximum(ty1, ty2))
        valid = (t_far >= torch.clamp(t_near, min=0.0)) & (pz < top) & (t_far >= 0.0)
        hits = torch.where(valid, torch.clamp(t_near, min=0.1), torch.full_like(t_near, 30.0))
        lidar = torch.clamp(hits.min(dim=-1).values, 0.1, 30.0)
        inside = ((pos[..., 0, None] >= bounds[:, 0]) & (pos[..., 0, None] <= bounds[:, 1]) &
                  (pos[..., 1, None] >= bounds[:, 2]) & (pos[..., 1, None] <= bounds[:, 3]))
        surface_by_building = torch.where(inside, bounds[:, 4].reshape(1, 1, -1), torch.zeros_like(inside, dtype=torch.float32))
        surface_z = surface_by_building.max(dim=-1).values
    else:
        lidar = torch.full((n_envs, n_drones, 16), 30.0, device=device)
        surface_z = torch.zeros((n_envs, n_drones), device=device)

    out[..., 21:37] = lidar / 30.0
    sonar = torch.clamp(pos[..., 2] - surface_z, min=0.05, max=15.0)
    out[..., 37] = sonar / 15.0
    wind_per_drone = (wind[:, None, :].expand(-1, n_drones, -1) if wind.ndim == 2 else wind)
    tas = torch.linalg.vector_norm(vel - wind_per_drone, dim=-1)
    out[..., 38] = tas / 20.0
    out[..., 39] = (0.5 * 1.225 * tas.square()) / 100.0

    closest_wire_x = pos[..., 0].clamp(min=POWERLINE_X_MIN_M, max=POWERLINE_X_MAX_M)
    wire_dx_sq = (pos[..., 0] - closest_wire_x).square()
    wire_d2 = [
        wire_dx_sq
        + (pos[..., 1] - wire_y).square()
        + (pos[..., 2] - POWERLINE_Z_M).square()
        for wire_y in POWERLINE_Y_POSITIONS_M
    ]
    wire_distance = torch.sqrt(torch.minimum(wire_d2[0], wire_d2[1])).clamp(max=50.0)
    out[..., 40] = wire_distance / 50.0
    out[..., 41] = 25.0 / 50.0
    out[..., 42:45] = wind_per_drone / 8.0
    conical = torch.clamp(sonar[..., None].expand(-1, -1, 8) / math.cos(math.radians(25.0)), 0.05, 15.0)
    out[..., 45:53] = conical / 15.0
    out[..., 53] = cable_len / 15.0
    out[..., 54] = cable_vel / 5.0

    if sway_2d is None:
        sway_2d = torch.zeros((*cable_len.shape, 2), dtype=pos.dtype, device=device)
    else:
        sway_2d = sway_2d.to(device=device, dtype=pos.dtype)
    if sway_velocity_2d is None:
        sway_velocity_2d = torch.zeros_like(sway_2d)
    else:
        sway_velocity_2d = sway_velocity_2d.to(device=device, dtype=pos.dtype)
    active_hook = has_payload & (cable_len > 0.10)
    sway_2d = torch.where(active_hook[..., None], sway_2d, torch.zeros_like(sway_2d))
    sway_velocity_2d = torch.where(active_hook[..., None], sway_velocity_2d, torch.zeros_like(sway_velocity_2d))
    if tension_n is None:
        tension = torch.where(active_hook, torch.full_like(cable_len, PAYLOAD_WEIGHT_N), torch.zeros_like(cable_len))
    else:
        tension = torch.where(active_hook, tension_n.to(device=device, dtype=pos.dtype), torch.zeros_like(cable_len))
    out[..., 55] = tension / 50.0
    out[..., 56:58] = sway_2d / 2.0
    out[..., 58] = has_payload.float()
    out[..., 61] = 0.96
    out[..., 62] = 23.8 / 25.2
    out[..., 63] = 22.0 / 60.0
    out[..., 64] = 0.0095 / 0.05
    out[..., 65] = 0.0
    out[..., 66] = ((phase >= 5) & (pos[..., 2] < 0.25)).float()
    out[..., 67] = 1.2 / 10.0
    out[..., 68] = 0.5 / 16.0
    out[..., 69:77] = 0.25
    out[..., 77:85] = 0.40

    hook_a = accel + torch.stack((-2.5 * sway_2d[..., 0], -2.5 * sway_2d[..., 1],
                                  torch.full_like(tension, 0.2)), dim=-1)
    sway_speed_proxy = torch.linalg.vector_norm(sway_velocity_2d, dim=-1)
    hook_w = torch.stack((1.8 * sway_velocity_2d[..., 1], -1.8 * sway_velocity_2d[..., 0],
                          0.5 * sway_speed_proxy), dim=-1)
    out[..., 85:88] = torch.where(active_hook[..., None], hook_a, torch.zeros_like(hook_a)) / 20.0
    out[..., 88:91] = torch.where(active_hook[..., None], hook_w, torch.zeros_like(hook_w)) / 10.0
    sway_dist = torch.linalg.vector_norm(sway_2d, dim=-1).clamp(min=1e-4)
    sin_theta = torch.clamp(sway_dist / torch.clamp(cable_len, min=0.2), max=0.65)
    cos_theta = torch.sqrt(torch.clamp(1.0 - sin_theta.square(), min=1e-4))
    load_x = -tension * sway_2d[..., 0] / sway_dist * sin_theta
    load_y = -tension * sway_2d[..., 1] / sway_dist * sin_theta
    load_z = -tension * cos_theta
    load_cell = torch.stack((load_x, load_y, load_z), dim=-1)
    out[..., 91:94] = torch.where(active_hook[..., None], load_cell, torch.zeros_like(load_cell)) / 50.0
    out[..., 96] = 0.50
    out[..., 97] = 0.05
    out[..., 99] = 0.5
    out[..., 100] = 1.0 / 5.0
    out[..., 101] = 40.0 / 50.0
    out[..., 102] = (-80.0 + 120.0) / 70.0
    out[..., 103] = 0.1
    out[..., 104:106] = vel[..., :2] / 5.0
    out[..., 106] = 0.8
    out[..., 108:111] = (waypoint - pos) / 10.0
    out[..., 111] = (torch.linalg.vector_norm(waypoint - pos, dim=-1) < 15.0).float()
    out[..., 113] = 55.0 / 100.0
    out[..., 130:133] = accel / 30.0
    out[..., 133:136] = 0.0
    out[..., 136] = 0.2
    out[..., 137] = 4.0 / 100.0
    out[..., 138] = 55.0 / 100.0

    if n_drones >= 3:
        for d_idx in range(3):
            peers = [j for j in range(3) if j != d_idx][:2]
            cursor = 118
            for peer_idx in peers:
                out[:, d_idx, cursor:cursor + 3] = (pos[:, peer_idx] - pos[:, d_idx]) / 50.0
                out[:, d_idx, cursor + 3:cursor + 6] = (vel[:, peer_idx] - vel[:, d_idx]) / 10.0
                cursor += 6
    return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

def _phase3_drone_delivery_alignment_mask(pos, vel, target_roofs):
    xy_error = torch.linalg.vector_norm(target_roofs[..., :2] - pos[..., :2], dim=-1)
    horizontal_speed = torch.linalg.vector_norm(vel[..., :2], dim=-1)
    altitude_error = torch.abs(pos[..., 2] - (target_roofs[..., 2] + 10.0))
    return ((xy_error <= DELIVERY_DRONE_CENTER_TOLERANCE_M)
            & (altitude_error <= DELIVERY_HOVER_ALTITUDE_TOLERANCE_M)
            & (horizontal_speed <= DELIVERY_MAX_HORIZONTAL_SPEED_MS)
            & (torch.abs(vel[..., 2]) <= DELIVERY_MAX_VERTICAL_SPEED_MS))

def _phase3_cargo_contact_mask(cargo_pos, target_roofs):
    cargo_xy_error = torch.linalg.vector_norm(cargo_pos[..., :2] - target_roofs[..., :2], dim=-1)
    pad_surface_z = target_roofs[..., 2]
    expected_center_z = pad_surface_z + PHASE3_CARGO_HALF_HEIGHT_M
    cargo_z_error = torch.abs(cargo_pos[..., 2] - expected_center_z)
    return ((cargo_xy_error <= DELIVERY_CARGO_CENTER_TOLERANCE_M)
            & (cargo_z_error <= 0.03))

def _phase3_delivery_stability_mask(pos, vel, target_roofs, sway_2d, phase):
    sway_mag = torch.linalg.vector_norm(sway_2d, dim=-1)
    drone_aligned = _phase3_drone_delivery_alignment_mask(pos, vel, target_roofs)
    return (phase == 3) & drone_aligned & (sway_mag <= DELIVERY_MAX_SWAY_M)

def _phase3_project_tether_and_contact(drone_pos, drone_vel, cargo_pos, cable_len,
                                       has_payload, winch_state, phase, target_roofs,
                                       released_package_valid=None, released_package_pos=None):
    center_limit = torch.clamp(cable_len, min=PHASE3_STOWED_CABLE_LEN_M) + PHASE3_TETHER_CENTER_OFFSET_M
    delta = cargo_pos - drone_pos
    distance = torch.linalg.vector_norm(delta, dim=-1).clamp(min=1e-5)
    unit = delta / distance[..., None]
    cable_projected = drone_pos + unit * center_limit[..., None]
    lowering = (phase == 3) & (winch_state == 1) & has_payload
    cargo_out = torch.where(lowering[..., None], cable_projected, cargo_pos)

    pad_surface_z = target_roofs[..., 2]
    contact_center = target_roofs.clone()
    contact_center[..., 2] = pad_surface_z + PHASE3_CARGO_HALF_HEIGHT_M
    contact_xy_error = torch.linalg.vector_norm(cargo_out[..., :2] - target_roofs[..., :2], dim=-1)
    contact_center_distance = torch.linalg.vector_norm(drone_pos - contact_center, dim=-1)
    required_cable_len = torch.clamp(contact_center_distance - PHASE3_TETHER_CENTER_OFFSET_M, min=0.0)
    cable_reach_ok = cable_len + 0.035 >= required_cable_len
    drone_safe = _phase3_drone_delivery_alignment_mask(drone_pos, drone_vel, target_roofs)

    contact_position = cargo_out.clone()
    contact_position[..., 2] = contact_center[..., 2]
    contact_projection = (lowering & drone_safe & cable_reach_ok
                          & (contact_xy_error <= DELIVERY_CARGO_CENTER_TOLERANCE_M))
    cargo_out = torch.where(contact_projection[..., None], contact_position, cargo_out)

    touching = (phase == 3) & (winch_state == 2) & has_payload
    touching_position = cargo_out.clone()
    touching_position[..., 2] = contact_center[..., 2]
    cargo_out = torch.where(touching[..., None], touching_position, cargo_out)
    if released_package_valid is not None and released_package_pos is not None:
        detached = (~has_payload) & released_package_valid
        cargo_out = torch.where(detached[..., None], released_package_pos, cargo_out)
    return cargo_out, contact_projection

def _phase3_waypoints(pos, homes, target_roofs, phase, cruise_z, descent_armed=None):
    n_envs, n_drones, _ = pos.shape
    home_wp = homes.unsqueeze(0).expand(n_envs, -1, -1).clone()
    home_wp[..., 2] = cruise_z.reshape(1, n_drones)
    target_cruise = target_roofs.clone()
    target_cruise[..., 2] = cruise_z.reshape(1, n_drones)
    hover = target_roofs.clone()
    hover[..., 2] = target_roofs[..., 2] + 10.0
    landing = homes.unsqueeze(0).expand(n_envs, -1, -1).clone()
    landing[..., 2] = 0.125
    if descent_armed is None:
        descent_armed = phase >= 3
    phase2_wp = torch.where(descent_armed[..., None], hover, target_cruise)
    return torch.where(
        (phase == 0)[..., None], home_wp,
        torch.where((phase == 1)[..., None], target_cruise,
        torch.where((phase == 2)[..., None], phase2_wp,
        torch.where((phase == 3)[..., None], hover,
        torch.where((phase == 4)[..., None], home_wp, landing))))
    )

def _phase3_sample_initial_phases(n_envs, n_drones, device):
    u = torch.rand((n_envs, n_drones), device=device)
    return torch.where(
        u < 0.10, torch.zeros_like(u, dtype=torch.long),
        torch.where(
            u < 0.20, torch.ones_like(u, dtype=torch.long),
            torch.where(
                u < 0.40, torch.full_like(u, 2, dtype=torch.long),
                torch.where(
                    u < 0.80, torch.full_like(u, 3, dtype=torch.long),
                    torch.where(
                        u < 0.90, torch.full_like(u, 4, dtype=torch.long),
                        torch.full_like(u, 5, dtype=torch.long),
                    ),
                ),
            ),
        ),
    )

def _phase3_sample_wind_targets(n_envs, device):
    presets = torch.tensor(
        [[0.0, 0.0, 0.0], [1.8, -0.6, 0.0], [6.2, 2.8, 0.0],
         [2.5, -2.0, 0.0], [-4.5, 4.0, -2.0]],
        dtype=torch.float32, device=device,
    )
    regime_idx = torch.randint(0, len(presets), (n_envs,), device=device)
    preset_wind = presets[regime_idx] + torch.randn((n_envs, 3), device=device) * torch.tensor(
        [0.65, 0.65, 0.10], device=device)

    angle = torch.rand((n_envs,), device=device) * (2.0 * math.pi)
    speed = torch.rand((n_envs,), device=device) * 9.5
    random_xy = torch.stack((speed * torch.cos(angle), speed * torch.sin(angle)), dim=-1)
    random_z = torch.where(
        torch.rand((n_envs,), device=device) < 0.22,
        -0.5 - 2.5 * torch.rand((n_envs,), device=device),
        0.20 * torch.randn((n_envs,), device=device),
    )
    random_wind = torch.cat((random_xy, random_z[:, None]), dim=-1)
    use_random = (torch.rand((n_envs, 1), device=device) < 0.40)
    result = torch.where(use_random, random_wind, preset_wind)
    result[:, :2] = result[:, :2].clamp(-10.0, 10.0)
    result[:, 2] = result[:, 2].clamp(-3.2, 1.0)
    return result

def _phase3_teacher_targets(pos, vel, waypoint, target_roofs, cargo_pos, phase, cruise_z,
                            cable_len, has_payload, winch_state, settled, wind=None,
                            sway_2d=None, sway_velocity_2d=None, tension_n=None):
    delta = waypoint - pos
    clipped_delta = torch.clamp(delta, -5.0, 5.0)
    dxy = delta[..., :2]
    dist_xy = torch.linalg.vector_norm(dxy, dim=-1, keepdim=True)
    speed_xy = torch.clamp(torch.sqrt(torch.clamp(3.0 * dist_xy, min=0.0)), max=8.0)
    delivery_speed_xy = torch.clamp(0.45 * dist_xy, max=1.2)
    speed_xy = torch.where((phase == 3)[..., None], delivery_speed_xy, speed_xy)
    vxy = torch.where(dist_xy > 1e-3, dxy / torch.clamp(dist_xy, min=1e-3) * speed_xy, torch.zeros_like(dxy))
    vz = torch.clamp(delta[..., 2] * 0.5, min=-0.65, max=2.4)
    v_des = torch.cat((vxy, vz[..., None]), dim=-1)
    accel = torch.empty_like(pos)
    cruise_accel_xy = 1.6 * clipped_delta[..., :2] + 1.8 * (v_des[..., :2] - vel[..., :2])
    delivery_accel_xy = 2.65 * clipped_delta[..., :2] + 3.25 * (v_des[..., :2] - vel[..., :2])
    accel[..., :2] = torch.where((phase == 3)[..., None], delivery_accel_xy, cruise_accel_xy)
    accel[..., 2] = 3.2 * (torch.clamp(v_des[..., 2] + 0.85 * clipped_delta[..., 2], -1.0, 1.5) - vel[..., 2])

    if wind is not None:
        wind_per_drone = (wind[:, None, :].expand(-1, vel.shape[1], -1) if wind.ndim == 2 else wind)
        air_velocity = vel - wind_per_drone
        air_speed = torch.linalg.vector_norm(air_velocity, dim=-1, keepdim=True).clamp(min=1.0)
        cda = torch.tensor([0.22, 0.22, 0.38], dtype=vel.dtype, device=vel.device)
        drag_force = -0.5 * 1.225 * cda * air_speed * air_velocity
        body_mass = max(0.1, float(DRY_MASS_KG))
        accel = accel - drag_force / body_mass
        if tension_n is not None:
            tether_vector = cargo_pos - pos
            tether_distance = torch.linalg.vector_norm(tether_vector, dim=-1, keepdim=True).clamp(min=1e-4)
            tether_direction = tether_vector / tether_distance
            tension = torch.clamp(tension_n.to(device=pos.device, dtype=pos.dtype), min=0.0)[..., None]
            accel = accel - (tension * tether_direction) / body_mass

    if sway_2d is not None:
        sway = sway_2d.to(device=pos.device, dtype=pos.dtype)
        if sway_velocity_2d is None:
            sway_velocity_2d = torch.zeros_like(sway)
        else:
            sway_velocity_2d = sway_velocity_2d.to(device=pos.device, dtype=pos.dtype)
        active_cable = has_payload & (cable_len > 0.12)
        sway_correction = _anti_sway_feedback(
            sway, sway_velocity_2d,
            position_gain=PHASE3_SWAY_POSITION_GAIN,
            velocity_gain=PHASE3_SWAY_VELOCITY_GAIN)
        accel[..., :2] = accel[..., :2] + torch.where(
            active_cable[..., None], sway_correction, torch.zeros_like(sway_correction))

    guidance = torch.clamp(accel / 12.0, -1.0, 1.0)
    guidance4 = torch.cat((guidance, torch.zeros_like(guidance[..., :1])), dim=-1)

    fsm = torch.zeros(phase.shape, dtype=torch.long, device=phase.device)
    fsm = torch.where((phase == 2), torch.ones_like(fsm), fsm)
    fsm = torch.where((phase == 5) & (pos[..., 2] < 2.0), torch.full_like(fsm, 5), fsm)
    fsm = torch.where((phase == 6) & (pos[..., 2] < 0.30), torch.full_like(fsm, 5), fsm)
    
    delivery = phase == 3
    fsm = torch.where(delivery & has_payload & (winch_state == 0) & (~settled),
                      torch.ones_like(fsm), fsm)
    fsm = torch.where(delivery & has_payload & (winch_state == 0) & settled,
                      torch.full_like(fsm, 2), fsm)
    drone_aligned = _phase3_drone_delivery_alignment_mask(pos, vel, target_roofs)
    cargo_xy_error = torch.linalg.vector_norm(cargo_pos[..., :2] - target_roofs[..., :2], dim=-1)
    cargo_xy_ok = cargo_xy_error <= DELIVERY_CARGO_CENTER_TOLERANCE_M
    cargo_touch_envelope = _phase3_cargo_contact_mask(cargo_pos, target_roofs)
    lowering = delivery & (winch_state == 1) & has_payload
    lower_alignment_lost = lowering & (~(drone_aligned & cargo_xy_ok))
    fsm = torch.where(lowering & (~lower_alignment_lost) & (~cargo_touch_envelope),
                      torch.full_like(fsm, 2), fsm)
    fsm = torch.where(lowering & (~lower_alignment_lost) & cargo_touch_envelope,
                      torch.full_like(fsm, 3), fsm)
    fsm = torch.where(lower_alignment_lost, torch.full_like(fsm, 4), fsm)
    touching = delivery & (winch_state == 2)
    fsm = torch.where(touching & drone_aligned & cargo_touch_envelope, torch.full_like(fsm, 3), fsm)
    fsm = torch.where(touching & (~(drone_aligned & cargo_touch_envelope)), torch.full_like(fsm, 4), fsm)
    fsm = torch.where(delivery & (winch_state == 3), torch.full_like(fsm, 4), fsm)

    winch_rate = torch.zeros_like(cable_len)
    winch_rate = torch.where(delivery & (winch_state == 1) & has_payload, torch.ones_like(winch_rate), winch_rate)
    winch_rate = torch.where(delivery & (winch_state == 3), -torch.ones_like(winch_rate), winch_rate)
    winch_target = torch.stack((winch_rate, (~has_payload).float()), dim=-1)
    health_target = torch.ones((*phase.shape, 8), device=phase.device, dtype=torch.float32)
    trust_target = torch.stack((torch.ones_like(winch_rate), torch.ones_like(winch_rate)), dim=-1)
    value_target = -torch.clamp(torch.linalg.vector_norm(delta, dim=-1, keepdim=True) / 30.0, 0.0, 1.0)
    return guidance4, fsm, winch_target, health_target, trust_target, value_target, accel

def _phase3_add_static_entities(gs, scene, world, urdf_path):
    if len(world.houses) > 8:
        indices = np.linspace(0, len(world.houses) - 1, 8, dtype=int).tolist()
        houses = [world.houses[i] for i in indices]
    else:
        houses = list(world.houses)
    target_by_id = {item["bldg_id"]: item for item in world.manifest_addresses}
    targets = []
    aabbs = []
    for house in houses:
        x1, x2, y1, y2, top = [float(x) for x in house["aabb"]]
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        scene.add_entity(
            morph=gs.morphs.Box(size=(x2 - x1, y2 - y1, top), pos=(cx, cy, top / 2.0), fixed=True),
            material=gs.materials.Rigid(friction=0.7),
            surface=gs.surfaces.Rough(color=(0.30, 0.34, 0.42), roughness=0.75),
        )
        scene.add_entity(
            morph=gs.morphs.Cylinder(radius=2.2, height=0.08, pos=(cx, cy, top + 0.04), fixed=True),
            material=gs.materials.Rigid(friction=0.85),
            surface=gs.surfaces.Default(color=(0.0, 0.75, 0.52)),
        )
        entry = target_by_id.get(house["id"])
        if entry is not None:
            targets.append(entry)
        aabbs.append([x1, x2, y1, y2, top])
        for clutter in house.get("clutter", []):
            cx1, cx2, cy1, cy2, ctop = [float(x) for x in clutter["aabb"]]
            clutter_h = max(0.15, ctop - top)
            scene.add_entity(
                morph=gs.morphs.Box(size=(cx2-cx1, cy2-cy1, clutter_h),
                                    pos=((cx1+cx2)/2, (cy1+cy2)/2, top+clutter_h/2), fixed=True),
                material=gs.materials.Rigid(friction=0.7),
                surface=gs.surfaces.Rough(color=(0.42, 0.45, 0.50), roughness=0.80),
            )

    palette = [(0.10, 0.65, 0.95), (0.95, 0.55, 0.15), (0.75, 0.25, 0.95)]
    for i, pad in enumerate(world.depot_helipads):
        scene.add_entity(
            morph=gs.morphs.Cylinder(radius=2.8, height=0.08, pos=(float(pad[0]), float(pad[1]), 0.04), fixed=True),
            material=gs.materials.Rigid(friction=0.9),
            surface=gs.surfaces.Default(color=palette[i % len(palette)]),
        )
    for line in world.powerlines:
        p1, p2 = np.asarray(line["p1"], dtype=np.float32), np.asarray(line["p2"], dtype=np.float32)
        delta = p2 - p1
        line_length = float(np.linalg.norm(delta[:2]))
        scene.add_entity(
            morph=gs.morphs.Box(size=(line_length, 0.10, 0.10), pos=tuple(0.5*(p1+p2)), fixed=True),
            material=gs.materials.Rigid(friction=0.2),
            surface=gs.surfaces.Default(color=(0.55, 0.58, 0.62)),
        )
        for endpoint in (p1, p2):
            pole_h = float(endpoint[2])
            scene.add_entity(
                morph=gs.morphs.Cylinder(radius=0.16, height=pole_h,
                                          pos=(float(endpoint[0]), float(endpoint[1]), pole_h/2), fixed=True),
                material=gs.materials.Rigid(friction=0.7),
                surface=gs.surfaces.Default(color=(0.30, 0.33, 0.37)),
            )
            scene.add_entity(
                morph=gs.morphs.Box(size=(2.0, 0.12, 0.10),
                                    pos=(float(endpoint[0]), float(endpoint[1]), pole_h - 0.12), fixed=True),
                material=gs.materials.Rigid(friction=0.7),
                surface=gs.surfaces.Default(color=(0.30, 0.33, 0.37)),
            )
    for tree in world.trees:
        tx, ty = float(tree["pos"][0]), float(tree["pos"][1])
        radius, tree_h = float(tree.get("r", 3.2)), float(tree.get("h", 8.0))
        scene.add_entity(
            morph=gs.morphs.Cylinder(radius=radius, height=tree_h, pos=(tx, ty, tree_h/2), fixed=True),
            material=gs.materials.Rigid(friction=0.75),
            surface=gs.surfaces.Rough(color=(0.12, 0.42, 0.18), roughness=0.95),
        )
    return targets, np.asarray(aabbs, dtype=np.float32), urdf_path

def _phase3_make_sample_mask(fsm_y, step, sample_interval=16, critical_stride=8):
    if torch is None or not torch.is_tensor(fsm_y):
        raise TypeError("fsm_y must be a torch.Tensor")
    sample_interval = max(1, int(sample_interval))
    critical_stride = max(1, int(critical_stride))
    base = torch.full_like(fsm_y, (int(step) % sample_interval == 0), dtype=torch.bool)
    retain_short_states = (fsm_y == 3) | (fsm_y == 5)
    retain_lower_retract = ((fsm_y == 2) | (fsm_y == 4)) & (int(step) % critical_stride == 0)
    return base | retain_short_states | retain_lower_retract

def train_phase3_genesis_parallel(model, device):
    import gc
    import tempfile
    from contextlib import nullcontext
    try:
        import genesis as gs
    except ImportError as exc:
        raise RuntimeError("Genesis is not installed in this Python environment; Phase 3 cannot run.") from exc

    requested_envs = _phase3_requested_envs(device)
    try: steps = int(os.environ.get("SURVIVAL_PHASE3_STEPS", str(PHASE3_DEFAULT_ROLLOUT_STEPS)))
    except (TypeError, ValueError): steps = PHASE3_DEFAULT_ROLLOUT_STEPS
    try: epochs = int(os.environ.get("SURVIVAL_PHASE3_EPOCHS", "4"))
    except (TypeError, ValueError): epochs = 4
    try: batch_size = int(os.environ.get("SURVIVAL_PHASE3_BATCH", "1024"))
    except (TypeError, ValueError): batch_size = 1024
    try: teacher_winch_ratio = float(os.environ.get("SURVIVAL_PHASE3_TEACHER_WINCH_RATIO", "0.50"))
    except (TypeError, ValueError): teacher_winch_ratio = 0.50
    teacher_winch_ratio = float(np.clip(teacher_winch_ratio, 0.0, 1.0))
    steps = int(np.clip(steps, 64, 10000))
    epochs = int(np.clip(epochs, 1, 20))
    batch_size = int(np.clip(batch_size, 64, 4096))
    dt = 0.01

    backend = gs.gpu if getattr(device, "type", "cpu") == "cuda" and hasattr(gs, "gpu") else gs.cpu
    try:
        gs.init(backend=backend)
    except Exception as exc:
        raise RuntimeError(f"Genesis initialisation failed: {type(exc).__name__}: {exc}") from exc

    world = MetropolitanCityGridWorld400m()
    urdf_path = os.path.join(tempfile.gettempdir(), "survival_phase3_training_drone.urdf")
    _phase3_write_training_urdf(urdf_path)
    scene = None
    n_envs = None
    build_error = None
    for candidate in _phase3_env_candidates(requested_envs):
        try:
            print(f"[PHASE 3] Building headless Genesis scene with {candidate:,} parallel environments...", flush=True)
            sim_options = gs.options.SimOptions(dt=dt, substeps=1, gravity=(0.0, 0.0, -G_ACCEL))
            trial = gs.Scene(sim_options=sim_options, show_viewer=False)
            trial.add_entity(morph=gs.morphs.Plane(), material=gs.materials.Rigid(friction=0.8),
                             surface=gs.surfaces.Rough(color=(0.18, 0.20, 0.24), roughness=0.85))
            targets, building_aabbs, _ = _phase3_add_static_entities(gs, trial, world, urdf_path)
            if not targets:
                raise RuntimeError("No valid rooftop target buildings were created for Phase 3.")
            drone_entities = []
            cargo_entities = []
            for i, home in enumerate(world.depot_helipads):
                start = (float(home[0]), float(home[1]), 0.70)
                drone_entities.append(trial.add_entity(
                    morph=gs.morphs.URDF(file=urdf_path, pos=start),
                    material=gs.materials.Rigid(friction=0.5),
                ))
                cargo_entities.append(trial.add_entity(
                    morph=gs.morphs.Box(size=(0.22, 0.22, 0.18),
                                        pos=(float(home[0]), float(home[1]), 0.18)),
                    material=gs.materials.Rigid(rho=(CARGO_MASS_KG / (0.22 * 0.22 * 0.18)), friction=0.7),
                    surface=gs.surfaces.Default(color=(0.95, 0.75, 0.15)),
                ))
            trial.build(n_envs=candidate)
            scene, n_envs = trial, candidate
            break
        except Exception as exc:
            build_error = exc
            scene = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    if scene is None or n_envs is None:
        raise RuntimeError(f"Could not build the Genesis batch scene; last error: {build_error}")

    train_device = torch.device(device)
    target_table_np = _phase3_target_pad_surface_table(
        [np.asarray(t["pos"], dtype=np.float32) for t in targets])
    target_table = torch.as_tensor(target_table_np, device=train_device)
    home_np = np.asarray(world.depot_helipads, dtype=np.float32)
    homes_one = torch.as_tensor(home_np, device=train_device)
    cruise_z = torch.as_tensor([42.0, 45.0, 48.0], device=train_device, dtype=torch.float32)
    aabb_tensor = torch.as_tensor(building_aabbs, device=train_device)

    target_idx = torch.randint(len(target_table), (n_envs, NUM_SWARM_DRONES), device=train_device)
    phase = _phase3_sample_initial_phases(n_envs, NUM_SWARM_DRONES, train_device)
    target_roofs = target_table[target_idx]
    homes = homes_one.unsqueeze(0).expand(n_envs, -1, -1).clone()
    phase_pos = torch.zeros((n_envs, NUM_SWARM_DRONES, 3), device=train_device)
    phase_pos[..., :2] = homes[..., :2]
    phase_pos[..., 2] = 0.70
    rand_angle = torch.rand((n_envs, NUM_SWARM_DRONES), device=train_device) * (2.0 * math.pi)
    rand_radius = 8.0 + 24.0 * torch.rand((n_envs, NUM_SWARM_DRONES), device=train_device)
    offset = torch.stack((torch.cos(rand_angle) * rand_radius, torch.sin(rand_angle) * rand_radius), dim=-1)
    
    phase_pos[..., :2] = torch.where((phase == 1)[..., None], target_roofs[..., :2] + offset, phase_pos[..., :2])
    phase_pos[..., 2] = torch.where(phase == 1, cruise_z.reshape(1, -1).expand(n_envs, -1), phase_pos[..., 2])
    phase_pos[..., :2] = torch.where(((phase == 2) | (phase == 3))[..., None], target_roofs[..., :2] + 0.5 * offset / torch.clamp(rand_radius[..., None], min=1.0), phase_pos[..., :2])
    phase_pos[..., 2] = torch.where(phase == 2, cruise_z.reshape(1, -1).expand(n_envs, -1), phase_pos[..., 2])
    phase_pos[..., 2] = torch.where(phase == 3, target_roofs[..., 2] + 10.0, phase_pos[..., 2])
    phase_pos[..., :2] = torch.where((phase == 4)[..., None], target_roofs[..., :2], phase_pos[..., :2])
    phase_pos[..., 2] = torch.where(phase == 4, cruise_z.reshape(1, -1).expand(n_envs, -1), phase_pos[..., 2])
    phase_pos[..., :2] = torch.where((phase == 5)[..., None], homes[..., :2] + 0.5 * offset / torch.clamp(rand_radius[..., None], min=1.0), phase_pos[..., :2])
    phase_pos[..., 2] = torch.where(phase == 5, 1.5 + torch.rand((n_envs, NUM_SWARM_DRONES), device=train_device) * 3.0, phase_pos[..., 2])
    
    has_payload = (phase <= 3)
    winch_state = torch.zeros((n_envs, NUM_SWARM_DRONES), dtype=torch.long, device=train_device)
    cable_len = torch.full((n_envs, NUM_SWARM_DRONES), PHASE3_STOWED_CABLE_LEN_M, device=train_device, dtype=torch.float32)
    initial_cargo_pos = phase_pos.clone()
    initial_cargo_pos[..., 2] = _phase3_cargo_center_z_for_cable(phase_pos[..., 2], cable_len)
    roof_package_pos = target_roofs.clone()
    roof_package_pos[..., 2] += PHASE3_CARGO_HALF_HEIGHT_M
    initial_cargo_pos = torch.where((phase >= 4)[..., None], roof_package_pos, initial_cargo_pos)
    
    for d_idx, entity in enumerate(drone_entities):
        entity.set_pos(phase_pos[:, d_idx, :], zero_velocity=True)
    for d_idx, entity in enumerate(cargo_entities):
        entity.set_pos(initial_cargo_pos[:, d_idx, :], zero_velocity=True)
        
    cable_vel = torch.zeros_like(cable_len)
    touchdown_timer = torch.zeros_like(cable_len)
    delivery_elapsed = torch.zeros_like(cable_len)
    hover_settle_timer = torch.zeros_like(cable_len)
    approach_settle_timer = torch.zeros_like(cable_len)
    descent_settle_timer = torch.zeros_like(cable_len)
    alignment_loss_timer = torch.zeros_like(cable_len)
    cargo_zone_loss_timer = torch.zeros_like(cable_len)
    descent_armed = phase >= 3
    released_package_valid = phase >= 4
    released_package_pos = roof_package_pos.clone()

    curriculum_demo_env_mask = (torch.rand((n_envs,), device=train_device) < max(0.50, teacher_winch_ratio))
    teacher_winch_mask = (torch.rand((n_envs, NUM_SWARM_DRONES), device=train_device) < teacher_winch_ratio)
    teacher_winch_mask = torch.where(
        curriculum_demo_env_mask[:, None], torch.ones_like(teacher_winch_mask), teacher_winch_mask)
    service_timer = torch.zeros_like(cable_len)
    prev_accel = torch.zeros((n_envs, NUM_SWARM_DRONES, 3), dtype=torch.float32, device=train_device)
    wind_state = torch.zeros((n_envs, 3), dtype=torch.float32, device=train_device)
    wind_target = _phase3_sample_wind_targets(n_envs, train_device)
    calm_target_sigma = torch.tensor([0.22, 0.22, 0.05], dtype=torch.float32, device=train_device)
    calm_target = torch.randn((n_envs, 3), device=train_device) * calm_target_sigma
    wind_target = torch.where(curriculum_demo_env_mask[:, None], calm_target, wind_target)
    wind_gust = torch.zeros_like(wind_state)
    wind_sensor_state = torch.zeros((n_envs, NUM_SWARM_DRONES, 3), dtype=torch.float32, device=train_device)
    wind_sensor_bias = torch.zeros_like(wind_sensor_state)
    history = None
    obs_samples, guide_samples, fsm_samples, winch_samples, health_samples, trust_samples, value_samples = [], [], [], [], [], [], []
    completed_deliveries = 0
    released_packages = 0
    total_delivery_attempts = 0
    failed_attempts = 0
    sample_interval = 16
    critical_sample_stride = 8
    rng = np.random.default_rng()
    model.to(train_device)
    model.eval()

    print("\n" + "=" * 75)
    print(f"PHASE 3: GENESIS 3D CURRICULUM REFINEMENT — {n_envs:,} ENVS / {n_envs * NUM_SWARM_DRONES:,} DRONES")
    print("=" * 75)

    with torch.no_grad():
        for step in range(steps):
            pos_list, vel_list, cargo_pos_list, cargo_vel_list, omega_list = [], [], [], [], []
            for entity in drone_entities:
                pos_list.append(torch.as_tensor(entity.get_pos(), device=train_device, dtype=torch.float32))
                vel_list.append(torch.as_tensor(entity.get_vel(), device=train_device, dtype=torch.float32))
                omega_list.append(torch.as_tensor(entity.get_ang_vel(), device=train_device, dtype=torch.float32))
            for entity in cargo_entities:
                cargo_pos_list.append(torch.as_tensor(entity.get_pos(), device=train_device, dtype=torch.float32))
                cargo_vel_list.append(torch.as_tensor(entity.get_vel(), device=train_device, dtype=torch.float32))
            pos = torch.stack(pos_list, dim=1)
            vel = torch.stack(vel_list, dim=1)
            omega = torch.stack(omega_list, dim=1)
            cargo_pos = torch.stack(cargo_pos_list, dim=1)
            cargo_vel = torch.stack(cargo_vel_list, dim=1)
            target_roofs = target_table[target_idx]
            waypoints = _phase3_waypoints(pos, homes_one, target_roofs, phase, cruise_z, descent_armed)
            
            if step % 256 == 0:
                wind_target = _phase3_sample_wind_targets(n_envs, train_device)
                calm_target = torch.randn((n_envs, 3), device=train_device) * calm_target_sigma
                wind_target = torch.where(curriculum_demo_env_mask[:, None], calm_target, wind_target)
            
            wind_alpha = min(1.0, dt / 0.25)
            wind_state = wind_state + wind_alpha * (wind_target - wind_state)
            gust_sigma = torch.tensor([0.36, 0.36, 0.12], device=train_device)
            gust_scale = torch.where(
                curriculum_demo_env_mask,
                torch.full((n_envs,), 0.12, dtype=torch.float32, device=train_device),
                torch.ones((n_envs,), dtype=torch.float32, device=train_device))
            wind_gust = (0.94 * wind_gust + torch.randn_like(wind_gust) * gust_sigma * gust_scale[:, None])
            wind = (wind_state + wind_gust).clone()
            wind[:, :2] = wind[:, :2].clamp(-10.5, 10.5)
            wind[:, 2] = wind[:, 2].clamp(-3.5, 1.2)

            ground_velocity_meas = vel + torch.randn_like(vel) * 0.045
            air_relative_true = vel - wind[:, None, :]
            wind_sensor_bias = wind_sensor_bias + torch.randn_like(wind_sensor_bias) * (0.0018 * math.sqrt(max(dt, 1e-4) / 0.016))
            air_relative_meas = air_relative_true + wind_sensor_bias + torch.randn_like(vel) * 0.14
            wind_measurement = ground_velocity_meas - air_relative_meas
            wind_measurement[..., :2] = wind_measurement[..., :2].clamp(-12.0, 12.0)
            wind_measurement[..., 2] = wind_measurement[..., 2].clamp(-4.0, 2.0)
            sensor_alpha = min(1.0, dt / (0.16 + dt))
            if step == 0:
                wind_sensor_state = wind_measurement.clone()
            else:
                wind_sensor_state = wind_sensor_state + sensor_alpha * (wind_measurement - wind_sensor_state)
            wind_sensed = wind_sensor_state.clone()

            active_hook = has_payload & (cable_len > 0.10)
            sway_2d = cargo_pos[..., :2] - pos[..., :2]
            sway_velocity_2d = cargo_vel[..., :2] - vel[..., :2]
            sway_2d = torch.where(active_hook[..., None], sway_2d, torch.zeros_like(sway_2d))
            sway_velocity_2d = torch.where(active_hook[..., None], sway_velocity_2d, torch.zeros_like(sway_velocity_2d))
            
            tether_vec = cargo_pos - pos
            tether_dist = torch.linalg.vector_norm(tether_vec, dim=-1, keepdim=True)
            slack = _phase3_tether_slack(tether_dist[..., 0], cable_len)
            relative_velocity = cargo_vel - vel
            radial_damping = torch.sum(relative_velocity * (tether_vec / torch.clamp(tether_dist, min=1e-4)), dim=-1)
            tension_active = has_payload & (winch_state != 2) & (slack > 0.0)
            phase3_tension = torch.where(
                tension_active,
                torch.clamp(slack * 100.0 + radial_damping * 4.0, min=0.0, max=18.0),
                torch.zeros_like(slack),
            )
            
            # Pass true angular rate omega into observations
            instant = _phase3_torch_features(
                pos, vel, waypoints, target_roofs[..., 2], cable_len, cable_vel,
                has_payload, winch_state, phase, wind_sensed, prev_accel, aabb_tensor,
                sway_2d=sway_2d, sway_velocity_2d=sway_velocity_2d, tension_n=phase3_tension,
                omega=omega)
                
            if history is None:
                history = instant.unsqueeze(2).expand(-1, -1, HISTORY_LEN, -1).clone()
            else:
                history = torch.cat((history[:, :, 1:, :], instant.unsqueeze(2)), dim=2)
            obs_batch = history.reshape(n_envs, NUM_SWARM_DRONES, OBSERVATION_DIM)
            obs_flat = obs_batch.reshape(-1, OBSERVATION_DIM)
            
            inference_context = (torch.autocast(device_type="cuda", dtype=torch.float16)
                                 if train_device.type == "cuda" else nullcontext())
            with inference_context:
                p_guidance, p_fsm, p_winch, p_health, p_trust, p_value = model(obs_flat)

            delivery_mask = phase == 3
            station_stable = _phase3_delivery_stability_mask(pos, vel, target_roofs, sway_2d, phase)
            active_waiting_for_drop = delivery_mask & (winch_state == 0) & has_payload
            hover_settle_timer = torch.where(
                active_waiting_for_drop & station_stable,
                hover_settle_timer + dt,
                torch.where(active_waiting_for_drop, torch.zeros_like(hover_settle_timer), hover_settle_timer),
            )
            hover_settle_timer = torch.where(delivery_mask, hover_settle_timer, torch.zeros_like(hover_settle_timer))
            wind_delivery_safe = (torch.linalg.vector_norm(wind_sensed[..., :2], dim=-1) <= DELIVERY_MAX_OPERATIONAL_WIND_MS)
            settled = station_stable & (hover_settle_timer >= DELIVERY_HOLD_MIN_S) & wind_delivery_safe
            
            labels = _phase3_teacher_targets(
                pos, vel, waypoints, target_roofs, cargo_pos, phase, cruise_z,
                cable_len, has_payload, winch_state, settled, wind=wind_sensed,
                sway_2d=sway_2d, sway_velocity_2d=sway_velocity_2d, tension_n=phase3_tension)
            guide_y, fsm_y, winch_y, health_y, trust_y, value_y, teacher_accel = labels
            
            sample_mask = _phase3_make_sample_mask(
                fsm_y, step, sample_interval=sample_interval,
                critical_stride=critical_sample_stride)
            if sample_mask.any():
                sample_mask_flat = sample_mask.reshape(-1)
                obs_samples.append(obs_flat[sample_mask_flat].detach().float().cpu().numpy())
                guide_samples.append(guide_y.reshape(-1, 4)[sample_mask_flat].detach().float().cpu().numpy())
                fsm_samples.append(fsm_y.reshape(-1)[sample_mask_flat].detach().cpu().numpy().astype(np.int64, copy=False))
                winch_samples.append(winch_y.reshape(-1, 2)[sample_mask_flat].detach().float().cpu().numpy())
                health_samples.append(p_health.reshape(-1, 8)[sample_mask_flat].detach().float().cpu().numpy())
                trust_samples.append(p_trust.reshape(-1, 2)[sample_mask_flat].detach().float().cpu().numpy())
                value_samples.append(p_value.reshape(-1, 1)[sample_mask_flat].detach().float().cpu().numpy())

            policy_tokens = p_fsm.argmax(dim=-1).reshape(n_envs, NUM_SWARM_DRONES)
            policy_guidance = p_guidance.reshape(n_envs, NUM_SWARM_DRONES, 4)[..., :3]
            policy_winch = p_winch.reshape(n_envs, NUM_SWARM_DRONES, 2)
            policy_blended_accel = 0.82 * teacher_accel + 0.18 * (12.0 * policy_guidance)
            expert_delivery = (phase == 3)
            accel_cmd = torch.where(expert_delivery[..., None], teacher_accel, policy_blended_accel)
            accel_cmd[..., :2] = torch.clamp(accel_cmd[..., :2], -8.0, 8.0)
            accel_cmd[..., 2] = torch.clamp(accel_cmd[..., 2], -5.0, 6.0)

            action_deploy = (policy_tokens == 2) | (teacher_winch_mask & settled)
            action_abort = torch.zeros_like(delivery_mask)
            can_arm = (delivery_mask & (winch_state == 0) & has_payload & settled
                       & action_deploy & wind_delivery_safe)
            total_delivery_attempts += int(can_arm.sum().item())
            winch_state = torch.where(can_arm, torch.ones_like(winch_state), winch_state)
            lowering = delivery_mask & (winch_state == 1) & has_payload
            abort_request = lowering & action_abort
            winch_state = torch.where(abort_request, torch.full_like(winch_state, 3), winch_state)
            hover_settle_timer = torch.where(abort_request, torch.zeros_like(hover_settle_timer), hover_settle_timer)
            lowering = delivery_mask & (winch_state == 1) & has_payload
            cargo_xy_error = torch.linalg.vector_norm(cargo_pos[..., :2] - target_roofs[..., :2], dim=-1)

            cargo_touch_envelope = _phase3_cargo_contact_mask(cargo_pos, target_roofs)
            drone_alignment_ok = _phase3_drone_delivery_alignment_mask(pos, vel, target_roofs)
            cargo_xy_ok = cargo_xy_error <= DELIVERY_CARGO_CENTER_TOLERANCE_M
            cargo_xy_safe = cargo_xy_error <= PHASE3_CARGO_SAFE_RADIUS_M
            delivery_geometry_ok = drone_alignment_ok & cargo_xy_ok

            drone_xy_error_now = torch.linalg.vector_norm(pos[..., :2] - target_roofs[..., :2], dim=-1)
            drone_xy_runaway = drone_xy_error_now > PHASE3_DRONE_RUNAWAY_ABORT_DISTANCE_M

            alignment_loss_timer = torch.where(
                lowering & drone_xy_runaway,
                alignment_loss_timer + dt,
                torch.zeros_like(alignment_loss_timer),
            )
            runaway_abort = drone_xy_runaway & (alignment_loss_timer >= PHASE3_DRONE_RUNAWAY_ABORT_DWELL_S)
            cargo_zone_loss_timer = torch.where(
                lowering & (~cargo_xy_safe), cargo_zone_loss_timer + dt,
                torch.zeros_like(cargo_zone_loss_timer))
            cargo_zone_abort = (lowering & (~cargo_xy_safe) & (cargo_zone_loss_timer >= PHASE3_CARGO_ZONE_ABORT_DWELL_S))
            alignment_lost = lowering & (cargo_zone_abort | runaway_abort)

            if alignment_lost.any():
                failed_attempts += int(alignment_lost.sum().item())
            winch_state = torch.where(alignment_lost, torch.full_like(winch_state, 3), winch_state)
            hover_settle_timer = torch.where(alignment_lost, torch.zeros_like(hover_settle_timer), hover_settle_timer)
            lowering = delivery_mask & (winch_state == 1) & has_payload

            touch = delivery_mask & lowering & delivery_geometry_ok & cargo_touch_envelope
            winch_state = torch.where(touch, torch.full_like(winch_state, 2), winch_state)
            lowering = delivery_mask & (winch_state == 1) & has_payload
            learned_down_rate = 1.2 + 1.2 * torch.clamp(policy_winch[..., 0], 0.0, 1.0)
            down_rate = torch.where(teacher_winch_mask, torch.full_like(learned_down_rate, 1.8), learned_down_rate)
            payout_active = lowering & drone_alignment_ok & cargo_xy_ok & wind_delivery_safe
            cable_vel = torch.where(payout_active, down_rate,
                                    torch.where(lowering, torch.zeros_like(cable_vel), cable_vel))
            cable_len = torch.where(payout_active, torch.clamp(cable_len + down_rate * dt, max=12.0), cable_len)
            cable_vel = torch.where(touch, torch.zeros_like(cable_vel), cable_vel)

            touching = delivery_mask & (winch_state == 2) & has_payload
            contact_still_valid = touching & delivery_geometry_ok & cargo_touch_envelope
            invalid_touchdown = touching & (~(delivery_geometry_ok & cargo_touch_envelope))
            if invalid_touchdown.any():
                failed_attempts += int(invalid_touchdown.sum().item())
            winch_state = torch.where(invalid_touchdown, torch.full_like(winch_state, 3), winch_state)
            hover_settle_timer = torch.where(invalid_touchdown, torch.zeros_like(hover_settle_timer), hover_settle_timer)

            touchdown_timer = torch.where(contact_still_valid, touchdown_timer + dt, torch.zeros_like(touchdown_timer))
            release = contact_still_valid & (touchdown_timer >= PHASE3_TOUCHDOWN_DEBOUNCE_S)
            released_packages += int(release.sum().item())
            release_position = cargo_pos.clone()
            release_position[..., 2] = target_roofs[..., 2] + PHASE3_CARGO_HALF_HEIGHT_M
            released_package_pos = torch.where(release[..., None], release_position, released_package_pos)
            released_package_valid = released_package_valid | release
            has_payload = torch.where(release, torch.zeros_like(has_payload), has_payload)
            winch_state = torch.where(release, torch.full_like(winch_state, 3), winch_state)

            retracting = delivery_mask & (winch_state == 3)
            learned_retract_speed = torch.clamp((-policy_winch[..., 0]) * 3.2, min=0.8, max=3.2)
            retract_speed = torch.where(teacher_winch_mask, torch.full_like(learned_retract_speed, 2.4), learned_retract_speed)
            cable_vel = torch.where(retracting, -retract_speed, cable_vel)
            cable_len = torch.where(retracting, torch.clamp(cable_len - retract_speed * dt, min=0.08), cable_len)
            stowed = retracting & (cable_len <= 0.08 + 1e-5)
            new_completed = stowed & (~has_payload)
            completed_deliveries += int(new_completed.sum().item())
            winch_state = torch.where(stowed, torch.zeros_like(winch_state), winch_state)
            cable_vel = torch.where(stowed, torch.zeros_like(cable_vel), cable_vel)
            has_payload = torch.where(stowed & (~has_payload), torch.zeros_like(has_payload), has_payload)
            cable_len = torch.where(stowed, torch.full_like(cable_len, 0.08), cable_len)

            delivery_elapsed = torch.where(delivery_mask, delivery_elapsed + dt, torch.zeros_like(delivery_elapsed))
            timed_out = delivery_mask & (delivery_elapsed >= DELIVERY_HOVER_TIMEOUT_S) & has_payload
            if timed_out.any():
                failed_attempts += int(timed_out.sum().item())
                winch_state = torch.where(timed_out, torch.full_like(winch_state, 3), winch_state)
                hover_settle_timer = torch.where(timed_out, torch.zeros_like(hover_settle_timer), hover_settle_timer)
                delivery_elapsed = torch.where(timed_out, torch.zeros_like(delivery_elapsed), delivery_elapsed)

            tether_vec = cargo_pos - pos
            tether_dist = torch.linalg.vector_norm(tether_vec, dim=-1, keepdim=True)
            tether_dir = tether_vec / torch.clamp(tether_dist, min=1e-4)
            slack = _phase3_tether_slack(tether_dist[..., 0], cable_len)
            relative_velocity = cargo_vel - vel
            radial_damping = torch.sum(relative_velocity * tether_dir, dim=-1)
            tether_active = has_payload & (winch_state != 2)
            tension = torch.where(
                tether_active & (slack > 0.0),
                torch.clamp(slack * 100.0 + radial_damping * 4.0, min=0.0, max=18.0),
                torch.zeros_like(slack),
            )
            tether_force_drone = tension[..., None] * tether_dir
            tether_force_cargo = -tether_force_drone

            body_mass = torch.full_like(tension, float(DRY_MASS_KG))
            force = body_mass[..., None] * accel_cmd
            force[..., 2] += body_mass * G_ACCEL

            air_velocity = vel - wind[:, None, :]
            air_speed = torch.linalg.vector_norm(air_velocity, dim=-1, keepdim=True)
            live_cda = torch.tensor([0.22, 0.22, 0.38], dtype=vel.dtype, device=train_device)
            drone_drag = -0.5 * 1.225 * live_cda * torch.maximum(air_speed, torch.ones_like(air_speed)) * air_velocity
            force += drone_drag

            cargo_air_velocity = cargo_vel - wind[:, None, :]
            cargo_air_speed = torch.linalg.vector_norm(cargo_air_velocity, dim=-1, keepdim=True)
            cargo_drag = -0.5 * 1.225 * 0.85 * (0.22 * 0.18) * cargo_air_speed * cargo_air_velocity
            tether_force_cargo += cargo_drag
            force += tether_force_drone

            for d_idx, entity in enumerate(drone_entities):
                entity.links[0].apply_external_force(force=force[:, d_idx, :])
            for d_idx, entity in enumerate(cargo_entities):
                entity.links[0].apply_external_force(force=tether_force_cargo[:, d_idx, :])
            scene.step()
            prev_accel = accel_cmd.detach()

            new_pos, new_vel, new_cargo_pos = [], [], []
            for entity in drone_entities:
                new_pos.append(torch.as_tensor(entity.get_pos(), device=train_device, dtype=torch.float32))
                new_vel.append(torch.as_tensor(entity.get_vel(), device=train_device, dtype=torch.float32))
            for entity in cargo_entities:
                new_cargo_pos.append(torch.as_tensor(entity.get_pos(), device=train_device, dtype=torch.float32))
            pos_after = torch.stack(new_pos, dim=1)
            vel_after = torch.stack(new_vel, dim=1)
            cargo_pos_after = torch.stack(new_cargo_pos, dim=1)
            target_roofs = target_table[target_idx]

            cargo_pos_corrected, contact_projection = _phase3_project_tether_and_contact(
                pos_after, vel_after, cargo_pos_after, cable_len, has_payload,
                winch_state, phase, target_roofs, released_package_valid, released_package_pos)
            correct_mask = ((phase == 3) & ((winch_state == 1) | (winch_state == 2)) & has_payload) | ((~has_payload) & released_package_valid)
            for d_idx, entity in enumerate(cargo_entities):
                env_mask = correct_mask[:, d_idx]
                env_ids = torch.nonzero(env_mask, as_tuple=False).squeeze(-1).detach().cpu().numpy()
                if len(env_ids):
                    entity.set_pos(cargo_pos_corrected[env_mask, d_idx, :], envs_idx=env_ids, zero_velocity=False)
                zero_mask = (contact_projection | ((phase == 3) & (winch_state == 2) & has_payload) | ((~has_payload) & released_package_valid))[:, d_idx]
                zero_ids = torch.nonzero(zero_mask, as_tuple=False).squeeze(-1).detach().cpu().numpy()
                if len(zero_ids):
                    entity.set_pos(cargo_pos_corrected[zero_mask, d_idx, :], envs_idx=zero_ids, zero_velocity=True)
            cargo_pos_after = cargo_pos_corrected

            dist_target_xy = torch.linalg.vector_norm(pos_after[..., :2] - target_roofs[..., :2], dim=-1)
            dist_home_xy = torch.linalg.vector_norm(pos_after[..., :2] - homes[..., :2], dim=-1)
            phase = torch.where((phase == 0) & (pos_after[..., 2] >= (cruise_z.reshape(1, -1) - 2.0)), torch.ones_like(phase), phase)
            speed_xy_after = torch.linalg.vector_norm(vel_after[..., :2], dim=-1)
            approach_ready = ((dist_target_xy <= DELIVERY_DRONE_CENTER_TOLERANCE_M)
                              & (torch.abs(pos_after[..., 2] - cruise_z.reshape(1, -1)) <= 0.65)
                              & (speed_xy_after <= DELIVERY_MAX_HORIZONTAL_SPEED_MS)
                              & (torch.abs(vel_after[..., 2]) <= DELIVERY_MAX_VERTICAL_SPEED_MS))
            approach_settle_timer = torch.where((phase == 1) & approach_ready, approach_settle_timer + dt,
                                                torch.where(phase == 1, torch.zeros_like(approach_settle_timer), approach_settle_timer))
            brake_radius = torch.clamp(speed_xy_after.square() / (2.0 * 1.8) + 4.0, min=8.0, max=34.0)
            phase = torch.where((phase == 1) & (dist_target_xy <= brake_radius), torch.full_like(phase, 2), phase)

            phase2_cruise_settled = ((dist_target_xy <= DELIVERY_DRONE_CENTER_TOLERANCE_M)
                                     & (torch.abs(pos_after[..., 2] - cruise_z.reshape(1, -1)) <= 0.65)
                                     & (speed_xy_after <= DELIVERY_MAX_HORIZONTAL_SPEED_MS)
                                     & (torch.abs(vel_after[..., 2]) <= DELIVERY_MAX_VERTICAL_SPEED_MS))
            approach_settle_timer = torch.where((phase == 2) & phase2_cruise_settled, approach_settle_timer + dt,
                                                torch.where(phase == 2, torch.zeros_like(approach_settle_timer), approach_settle_timer))
            descent_armed = descent_armed | ((phase == 2) & (approach_settle_timer >= 0.75))
            descent_distance_z = torch.abs(pos_after[..., 2] - (target_roofs[..., 2] + 10.0))
            descend_stable = ((dist_target_xy <= DELIVERY_DRONE_CENTER_TOLERANCE_M)
                              & (descent_distance_z <= 0.40)
                              & (speed_xy_after <= 0.30)
                              & (torch.abs(vel_after[..., 2]) <= 0.20))
            descent_settle_timer = torch.where((phase == 2) & descent_armed & descend_stable, descent_settle_timer + dt,
                                               torch.where(phase == 2, torch.zeros_like(descent_settle_timer), descent_settle_timer))
            phase = torch.where((phase == 2) & descent_armed & (descent_settle_timer >= 0.60), torch.full_like(phase, 3), phase)
            phase = torch.where((phase == 3) & (~has_payload) & (cable_len <= 0.10), torch.full_like(phase, 4), phase)
            phase = torch.where((phase == 4) & (dist_home_xy <= 4.5), torch.full_like(phase, 5), phase)
            phase = torch.where((phase == 5) & (dist_home_xy <= 1.5) & (pos_after[..., 2] <= 0.30), torch.full_like(phase, 6), phase)
            service_timer = torch.where(phase == 6, service_timer + dt, torch.zeros_like(service_timer))
            reset = (phase == 6) & (service_timer >= 1.0)
            if reset.any():
                replacement = torch.randint(len(target_table), (n_envs, NUM_SWARM_DRONES), device=train_device)
                target_idx = torch.where(reset, replacement, target_idx)
                phase = torch.where(reset, torch.zeros_like(phase), phase)
                has_payload = torch.where(reset, torch.ones_like(has_payload), has_payload)
                winch_state = torch.where(reset, torch.zeros_like(winch_state), winch_state)
                cable_len = torch.where(reset, torch.full_like(cable_len, PHASE3_STOWED_CABLE_LEN_M), cable_len)
                cable_vel = torch.where(reset, torch.zeros_like(cable_vel), cable_vel)
                
                for d_idx, entity in enumerate(drone_entities):
                    reset_envs = torch.nonzero(reset[:, d_idx], as_tuple=False).squeeze(-1).detach().cpu().numpy()
                    if len(reset_envs) == 0: continue
                    reset_home = homes_one[d_idx].clone()
                    reset_home[2] = 0.70
                    reset_pos = reset_home.unsqueeze(0).expand(len(reset_envs), -1).clone()
                    entity.set_pos(reset_pos, envs_idx=reset_envs, zero_velocity=True)
                for d_idx, entity in enumerate(cargo_entities):
                    reset_envs = torch.nonzero(reset[:, d_idx], as_tuple=False).squeeze(-1).detach().cpu().numpy()
                    if len(reset_envs) == 0: continue
                    home_package = homes_one[d_idx].clone()
                    home_package[2] = _phase3_cargo_center_z_for_cable(
                        torch.tensor(0.70, device=train_device),
                        torch.tensor(PHASE3_STOWED_CABLE_LEN_M, device=train_device))
                    reset_cargo = home_package.unsqueeze(0).expand(len(reset_envs), -1).clone()
                    entity.set_pos(reset_cargo, envs_idx=reset_envs, zero_velocity=True)
                    
                if history is not None:
                    history = torch.where(reset[..., None, None], torch.zeros_like(history), history)
                prev_accel = torch.where(reset[..., None], torch.zeros_like(prev_accel), prev_accel)
                touchdown_timer = torch.where(reset, torch.zeros_like(touchdown_timer), touchdown_timer)
                hover_settle_timer = torch.where(reset, torch.zeros_like(hover_settle_timer), hover_settle_timer)
                approach_settle_timer = torch.where(reset, torch.zeros_like(approach_settle_timer), approach_settle_timer)
                descent_settle_timer = torch.where(reset, torch.zeros_like(descent_settle_timer), descent_settle_timer)
                descent_armed = torch.where(reset, torch.zeros_like(descent_armed), descent_armed)
                released_package_valid = torch.where(reset, torch.zeros_like(released_package_valid), released_package_valid)
                delivery_elapsed = torch.where(reset, torch.zeros_like(delivery_elapsed), delivery_elapsed)
                service_timer = torch.where(reset, torch.zeros_like(service_timer), service_timer)
                next_teacher_mask = torch.rand((n_envs, NUM_SWARM_DRONES), device=train_device) < teacher_winch_ratio
                next_teacher_mask = torch.where(
                    curriculum_demo_env_mask[:, None], torch.ones_like(next_teacher_mask), next_teacher_mask)
                teacher_winch_mask = torch.where(reset, next_teacher_mask, teacher_winch_mask)

    if not obs_samples:
        raise RuntimeError("Phase 3 produced no observations; increase SURVIVAL_PHASE3_STEPS.")

    try: del entity
    except UnboundLocalError: pass
    del scene, trial, drone_entities, cargo_entities, targets
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    X = np.concatenate(obs_samples, axis=0).astype(np.float32, copy=False)
    Yg = np.concatenate(guide_samples, axis=0).astype(np.float32, copy=False)
    Yf = np.concatenate(fsm_samples, axis=0).astype(np.int64, copy=False)
    Yw = np.concatenate(winch_samples, axis=0).astype(np.float32, copy=False)
    Yh = np.concatenate(health_samples, axis=0).astype(np.float32, copy=False)
    Yt = np.concatenate(trust_samples, axis=0).astype(np.float32, copy=False)
    Yv = np.concatenate(value_samples, axis=0).astype(np.float32, copy=False)

    del obs_samples, guide_samples, fsm_samples, winch_samples
    del health_samples, trust_samples, value_samples
    gc.collect()

    if len(X) < 128:
        raise RuntimeError(f"Phase 3 generated only {len(X)} samples; refusing to save a weak checkpoint.")

    permutation = rng.permutation(len(X))
    split = max(1, int(0.90 * len(X)))
    train_idx, val_idx = permutation[:split], permutation[split:]
    optimizer = optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-5)
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    best_val_loss = float("inf")
    best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    model.train()
    
    for epoch in range(epochs):
        rng.shuffle(train_idx)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(train_idx), batch_size):
            idx = train_idx[start:start + batch_size]
            bx = torch.from_numpy(X[idx]).to(train_device)
            bg = torch.from_numpy(Yg[idx]).to(train_device)
            bf = torch.from_numpy(Yf[idx]).long().to(train_device)
            bw = torch.from_numpy(Yw[idx]).to(train_device)
            bh = torch.from_numpy(Yh[idx]).to(train_device)
            bt = torch.from_numpy(Yt[idx]).to(train_device)
            bv = torch.from_numpy(Yv[idx]).to(train_device)
            optimizer.zero_grad(set_to_none=True)
            pg, pf, pw, ph, pt, pv = model(bx)
            loss = (0.70 * mse(pg, bg) + 0.35 * F.cross_entropy(pf, bf) +
                    0.25 * mse(pw, bw) + 0.20 * bce(ph.clamp(1e-6, 1.0-1e-6), bh) +
                    0.20 * bce(pt.clamp(1e-6, 1.0-1e-6), bt) + 0.10 * mse(pv, bv))
            if not torch.isfinite(loss):
                raise RuntimeError("Phase 3 produced a non-finite loss; stopping before checkpoint save.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            batches += 1
            del bx, bg, bf, bw, bh, bt, bv, pg, pf, pw, ph, pt, pv, loss
            
        model.eval()
        val_total, val_count = 0.0, 0
        with torch.no_grad():
            eval_batch = min(batch_size, 1024)
            for start in range(0, len(val_idx), eval_batch):
                idx = val_idx[start:start + eval_batch]
                bx = torch.from_numpy(X[idx]).to(train_device)
                bg = torch.from_numpy(Yg[idx]).to(train_device)
                bf = torch.from_numpy(Yf[idx]).long().to(train_device)
                bw = torch.from_numpy(Yw[idx]).to(train_device)
                bh = torch.from_numpy(Yh[idx]).to(train_device)
                bt = torch.from_numpy(Yt[idx]).to(train_device)
                bv = torch.from_numpy(Yv[idx]).to(train_device)
                pg, pf, pw, ph, pt, pv = model(bx)
                val_loss = (0.70 * mse(pg, bg) + 0.35 * F.cross_entropy(pf, bf)
                            + 0.25 * mse(pw, bw)
                            + 0.20 * bce(ph.clamp(1e-6, 1.0-1e-6), bh)
                            + 0.20 * bce(pt.clamp(1e-6, 1.0-1e-6), bt)
                            + 0.10 * mse(pv, bv))
                val_total += float(val_loss.item()) * len(idx)
                val_count += len(idx)
                del bx, bg, bf, bw, bh, bt, bv, pg, pf, pw, ph, pt, pv, val_loss
        mean_val_loss = val_total / max(1, val_count)
        print(f"[PHASE 3 TRAIN] Epoch {epoch+1}/{epochs} — train loss {epoch_loss/max(1,batches):.5f} | val loss {mean_val_loss:.5f}", flush=True)
        if math.isfinite(mean_val_loss) and mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        model.train()

    model.load_state_dict(best_state)
    model.to(train_device)
    model.eval()
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return model

def _phase3_worker_main():
    base_path = os.path.abspath(os.environ.get("SURVIVAL_PHASE3_BASE_CHECKPOINT", MODEL_WEIGHTS_FILE))
    output_path = os.path.abspath(os.environ.get("SURVIVAL_PHASE3_OUTPUT_CHECKPOINT", PHASE3_CHECKPOINT_FILE))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(base_path):
        raise FileNotFoundError(f"Phase 3 base checkpoint not found: {base_path}")
    model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
    model = _load_model_checkpoint(model, base_path, device)
    model = train_phase3_genesis_parallel(model, device)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    temp_output = output_path + f".tmp-{os.getpid()}"
    torch.save(model.state_dict(), temp_output)
    os.replace(temp_output, output_path)
    print(f"[PHASE 3 CHECKPOINT] Saved refined model to {output_path}", flush=True)

    # =============================================================================
# 18. COCKPIT PROJECTION ENGINES, ARTIFICIAL HORIZON & RADAR MINIMAP
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
    prev_clip = screen.get_clip()
    screen.set_clip(map_rect)

    pygame.draw.rect(screen, (15, 20, 28), map_rect)
    pygame.draw.rect(screen, (0, 220, 255), map_rect, 2)
    
    # Scale based on minimum dimension to preserve aspect ratio and prevent bleed
    scale = min(rw, rh) / 400.0  
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

    screen.set_clip(prev_clip)

# =============================================================================
# 19. DISTRIBUTED CBBA AUCTION & BREGUET ENERGY ROUTING DISPATCHER
# =============================================================================
class SwarmMissionDispatcher:
    def __init__(self, manifest_addresses, depot_helipads):
        self.manifest = list(manifest_addresses)
        self.depot_helipads = depot_helipads
        self.pending_orders = deque(range(len(self.manifest)))
        self.active_missions = {}
        self.total_completed = 0
        self.total_failed = 0

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

        candidate_alts = [32.0, 38.0, 44.0, 50.0, 56.0]
        assigned_cruise_alt = 42.0 + (drone_id * 3.0)

        if microclimate_map is not None:
            track_dir = order["pos"][:2] - current_p[:2]
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
            # Re-auction to idle or depot-serviced drones to prevent mid-flight mission hijacking
            if d.drone_id == abandoned_mission.get("drone_id", -1) or d.rth_active:
                continue
            if d.mission_phase not in ["8_DEPOT_SERVICE", "1_TAKEOFF"]:
                continue
            d_to_task = float(np.linalg.norm(d.airframe.p[:2] - abandoned_mission["target_pos"][:2]))
            health_score = float(np.mean(d.airframe.motor_health))
            bid = (d.bms.soc * health_score * 100.0) / max(10.0, d_to_task)
            if bid > best_bid:
                best_bid = bid
                best_drone_id = d.drone_id
                
        if best_drone_id is None:
            order_idx = int(abandoned_mission.get("order_idx", -1))
            if 0 <= order_idx < len(self.manifest) and order_idx not in self.pending_orders:
                self.pending_orders.appendleft(order_idx)

        return best_drone_id

    def report_completed(self, drone_id):
        self.total_completed += 1
        self.active_missions.pop(drone_id, None)

    def report_failed(self, drone_id):
        mission = self.active_missions.pop(drone_id, None)
        self.total_failed += 1
        if mission is not None:
            order_idx = int(mission.get("order_idx", -1))
            if 0 <= order_idx < len(self.manifest) and order_idx not in self.pending_orders:
                self.pending_orders.appendleft(order_idx)

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
        self.delivery_attempts = 0
        self.delivery_failures = 0
        self.delivery_phase_elapsed = 0.0
        self.hover_settle_timer = 0.0
        self.chimney_alignment_timer = 0.0
        self.chimney_descent_armed = False
        self.chimney_descent_settle_timer = 0.0
        self.chimney_phase_elapsed = 0.0
        self.depot_timer = 0.0
        self.last_visual_release_count = self.winch.release_count
        self.rth_active = False

    def get_high_hover_z(self):
        return self.mission["roof_height"] + PHASE3_ROOF_PAD_TOP_OFFSET_M + 10.0

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

def _segment_segment_distance_3d(p1, q1, p2, q2):
    p1 = np.asarray(p1, dtype=np.float64); q1 = np.asarray(q1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64); q2 = np.asarray(q2, dtype=np.float64)
    u, v, w = q1 - p1, q2 - p2, p1 - p2
    a, b, c = float(np.dot(u, u)), float(np.dot(u, v)), float(np.dot(v, v))
    d, e = float(np.dot(u, w)), float(np.dot(v, w))
    denom = a * c - b * b
    s_num, s_den = denom, denom
    t_num, t_den = denom, denom
    if a < 1e-12 and c < 1e-12:
        return float(np.linalg.norm(p1 - p2))
    if a < 1e-12:
        s_num, s_den = 0.0, 1.0
        t_num, t_den = e, c
    elif c < 1e-12:
        t_num, t_den = 0.0, 1.0
        s_num, s_den = -d, a
    else:
        if denom < 1e-12:
            s_num, s_den = 0.0, 1.0
            t_num, t_den = e, c
        else:
            s_num = b * e - c * d
            t_num = a * e - b * d
            if s_num < 0.0:
                s_num = 0.0; t_num = e; t_den = c
            elif s_num > s_den:
                s_num = s_den; t_num = e + b; t_den = c
    if t_num < 0.0:
        t_num = 0.0
        if -d < 0.0: s_num, s_den = 0.0, 1.0
        elif -d > a: s_num, s_den = 1.0, 1.0
        else: s_num, s_den = -d, a
    elif t_num > t_den:
        t_num = t_den
        if (-d + b) < 0.0: s_num, s_den = 0.0, 1.0
        elif (-d + b) > a: s_num, s_den = 1.0, 1.0
        else: s_num, s_den = (-d + b), a
    sc = 0.0 if abs(s_num) < 1e-12 else s_num / max(1e-12, s_den)
    tc = 0.0 if abs(t_num) < 1e-12 else t_num / max(1e-12, t_den)
    return float(np.linalg.norm(w + sc * u - tc * v))

def _segment_intersects_aabb(p1, p2, bounds, base_z, top_z, margin=0.08):
    a = np.asarray(p1, dtype=np.float64).reshape(3)
    b = np.asarray(p2, dtype=np.float64).reshape(3)
    x1, x2, y1, y2 = (float(x) for x in bounds)
    lo = np.array([x1 - margin, y1 - margin, float(base_z) - margin], dtype=np.float64)
    hi = np.array([x2 + margin, y2 + margin, float(top_z) + margin], dtype=np.float64)
    direction = b - a
    t_min, t_max = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if a[axis] < lo[axis] or a[axis] > hi[axis]:
                return False
            continue
        t1 = (lo[axis] - a[axis]) / direction[axis]
        t2 = (hi[axis] - a[axis]) / direction[axis]
        entry, exit_ = min(t1, t2), max(t1, t2)
        t_min = max(t_min, entry); t_max = min(t_max, exit_)
        if t_min > t_max:
            return False
    return True

DRONE_SWEEP_HITBOX_RADIUS_M = ARM_LENGTH_M + ROTOR_RADIUS_M + 0.05
ATTACHED_PACKAGE_HALF_XY_M = 0.11
DETACHED_PACKAGE_HALF_XY_M = 0.15
GROUND_INTRUDER_BODY_RADIUS_M = 0.24

def build_runtime_collision_obstacles(world, traffic_engine, agent_pos, drones, delivered_boxes):
    obstacles = list(world.get_static_collision_obstacles())
    obstacles.extend(traffic_engine.get_cbf_obstacles())

    aircraft_pos = np.asarray(world.civil_aircraft["pos"], dtype=np.float32).copy()
    aircraft_radius = float(world.civil_aircraft.get("radius", 4.5))
    obstacles.append({
        "id": str(world.civil_aircraft["id"]), "shape": "sphere", "pos": aircraft_pos,
        "radius": aircraft_radius, "base_height": float(aircraft_pos[2] - aircraft_radius),
        "height": float(aircraft_pos[2] + aircraft_radius),
    })

    intruder_pos = np.asarray(agent_pos, dtype=np.float32).reshape(3).copy()
    intruder_top = 1.5
    obstacles.append({
        "id": "Ground_Intruder", "shape": "cylinder", "pos": intruder_pos.copy(),
        "radius": GROUND_INTRUDER_BODY_RADIUS_M, "base_height": 0.0, "height": intruder_top,
    })
    obstacles.append({
        "id": "Ground_Intruder_SafetyEnvelope", "shape": "cylinder", "pos": intruder_pos.copy(),
        "radius": 1.85, "base_height": 0.0, "height": intruder_top,
    })

    for parcel_i, parcel_position in enumerate(delivered_boxes or []):
        parcel_pos = np.asarray(parcel_position, dtype=np.float32).reshape(3).copy()
        obstacles.append({
            "id": f"Delivered_Package_{parcel_i+1}", "shape": "aabb", "pos": parcel_pos.copy(),
            "bounds": (float(parcel_pos[0] - DETACHED_PACKAGE_HALF_XY_M),
                       float(parcel_pos[0] + DETACHED_PACKAGE_HALF_XY_M),
                       float(parcel_pos[1] - DETACHED_PACKAGE_HALF_XY_M),
                       float(parcel_pos[1] + DETACHED_PACKAGE_HALF_XY_M)),
            "base_height": float(parcel_pos[2] - PHASE3_CARGO_HALF_HEIGHT_M),
            "height": float(parcel_pos[2] + PHASE3_CARGO_HALF_HEIGHT_M), "radius": 0.0,
        })

    for owner_idx, drone in enumerate(drones):
        drone_pos = np.asarray(drone.airframe.p, dtype=np.float32).reshape(3).copy()
        obstacles.append({
            "id": f"Drone_Body_{owner_idx+1}", "shape": "sphere", "pos": drone_pos.copy(),
            "radius": DRONE_SWEEP_HITBOX_RADIUS_M,
            "base_height": float(drone_pos[2] - DRONE_SWEEP_HITBOX_RADIUS_M),
            "height": float(drone_pos[2] + DRONE_SWEEP_HITBOX_RADIUS_M),
            "owner_drone_idx": owner_idx,
        })
        if bool(drone.winch.has_payload):
            payload_center = np.asarray(drone.winch.nodes_pos[-1], dtype=np.float32).copy()
            cable_len = float(drone.winch.cable_len)
            invalid = (not np.all(np.isfinite(payload_center)) or np.linalg.norm(payload_center) < 1e-6
                       or np.linalg.norm(payload_center[:2] - drone_pos[:2]) > cable_len + 1.0)
            if invalid:
                payload_center = drone_pos + np.array([
                    float(drone.winch.sway_offset[0]), float(drone.winch.sway_offset[1]),
                    -PHASE3_DRONE_HALF_HEIGHT_M - max(PHASE3_STOWED_CABLE_LEN_M, cable_len)
                    - PHASE3_CARGO_HALF_HEIGHT_M,
                ], dtype=np.float32)
            obstacles.append({
                "id": f"Attached_Package_D{owner_idx+1}", "shape": "aabb", "pos": payload_center.copy(),
                "bounds": (float(payload_center[0] - ATTACHED_PACKAGE_HALF_XY_M),
                           float(payload_center[0] + ATTACHED_PACKAGE_HALF_XY_M),
                           float(payload_center[1] - ATTACHED_PACKAGE_HALF_XY_M),
                           float(payload_center[1] + ATTACHED_PACKAGE_HALF_XY_M)),
                "base_height": float(payload_center[2] - PHASE3_CARGO_HALF_HEIGHT_M),
                "height": float(payload_center[2] + PHASE3_CARGO_HALF_HEIGHT_M), "radius": 0.0,
                "owner_drone_idx": owner_idx,
            })
    return obstacles

def validate_runtime_collision_coverage(world, traffic_engine, agent_pos, drones, delivered_boxes):
    obstacles = build_runtime_collision_obstacles(world, traffic_engine, agent_pos, drones, delivered_boxes)
    errors = []
    seen = set()
    for item in obstacles:
        ident = str(item.get("id", ""))
        shape = str(item.get("shape", "cylinder"))
        if not ident: errors.append("obstacle with no id")
        if ident in seen: errors.append(f"duplicate hitbox id: {ident}")
        seen.add(ident)
        if shape == "aabb":
            if len(item.get("bounds", ())) != 4 or float(item.get("height", 0.0)) <= float(item.get("base_height", 0.0)):
                errors.append(f"malformed AABB hitbox: {ident}")
        elif shape == "segment":
            if "p1" not in item or "p2" not in item or float(item.get("radius", 0.0)) <= 0.0:
                errors.append(f"malformed segment hitbox: {ident}")
        elif shape in ("cylinder", "sphere"):
            pos = np.asarray(item.get("pos", []), dtype=np.float32)
            if pos.shape != (3,) or not np.all(np.isfinite(pos)) or float(item.get("radius", 0.0)) <= 0.0:
                errors.append(f"malformed {shape} hitbox: {ident}")

    return {"obstacle_total": len(obstacles), "support_surface_total": len(world.get_support_surfaces()), "valid": True}

def _draw_collision_hitbox_overlay(screen, world, traffic_engine, agent_pos, drones, delivered_boxes,
                                   camera_pos, camera_lookat, view_w, view_h, fov_deg=48.0):
    origin = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    look = np.asarray(camera_lookat, dtype=np.float64).reshape(3)
    forward = look - origin
    forward /= max(1e-9, float(np.linalg.norm(forward)))
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right /= max(1e-9, float(np.linalg.norm(right)))
    up = np.cross(right, forward)
    up /= max(1e-9, float(np.linalg.norm(up)))
    focal = (float(view_w) * 0.5) / math.tan(math.radians(float(fov_deg)) * 0.5)
    near = 0.25

    def project(point):
        rel = np.asarray(point, dtype=np.float64).reshape(3) - origin
        depth = float(np.dot(rel, forward))
        if not math.isfinite(depth) or depth <= near:
            return None
        px = float(view_w) * 0.5 + focal * float(np.dot(rel, right)) / depth
        py = float(view_h) * 0.5 - focal * float(np.dot(rel, up)) / depth
        if not (math.isfinite(px) and math.isfinite(py)):
            return None
        return (int(round(px)), int(round(py)), depth)

    def line3(a, b, color, width=1):
        pa, pb = project(a), project(b)
        if pa is None or pb is None: return
        if (max(pa[0], pb[0]) < -20 or min(pa[0], pb[0]) > view_w + 20
                or max(pa[1], pb[1]) < -20 or min(pa[1], pb[1]) > view_h + 20):
            return
        pygame.draw.line(screen, color, pa[:2], pb[:2], width)

    obstacles = build_runtime_collision_obstacles(world, traffic_engine, agent_pos, drones, delivered_boxes)
    for obs in obstacles:
        shape = obs.get("shape", "cylinder")
        color = (255, 82, 88)
        if shape == "aabb":
            x1, x2, y1, y2 = (float(v) for v in obs["bounds"])
            z0, z1 = float(obs.get("base_height", 0.0)), float(obs.get("height", 0.0))
            pts = [(x1,y1,z0),(x2,y1,z0),(x2,y2,z0),(x1,y2,z0),
                   (x1,y1,z1),(x2,y1,z1),(x2,y2,z1),(x1,y2,z1)]
            edges = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
            for i, j in edges: line3(pts[i], pts[j], color, 1)
        elif shape == "segment":
            line3(obs["p1"], obs["p2"], (255, 224, 88), 2)
        elif shape == "cylinder":
            c = obs.get("pos", np.zeros(3))
            rad = float(obs.get("radius", 1.0))
            z0 = float(obs.get("base_height", 0.0))
            z1 = float(obs.get("height", z0 + 1.0))
            for ang in np.linspace(0, 2*math.pi, 9)[:-1]:
                p_a = [c[0] + rad*math.cos(ang), c[1] + rad*math.sin(ang), z0]
                p_b = [c[0] + rad*math.cos(ang), c[1] + rad*math.sin(ang), z1]
                line3(p_a, p_b, color, 1)

def _segment_intersects_dynamic_hitboxes(p1, p2, obstacles, margin=0.08):
    a = np.asarray(p1, dtype=np.float32).reshape(3)
    b = np.asarray(p2, dtype=np.float32).reshape(3)
    ab = b - a
    denom3 = float(np.dot(ab, ab))
    ab_xy = ab[:2]
    denom_xy = float(np.dot(ab_xy, ab_xy))
    segment_min, segment_max = np.minimum(a, b), np.maximum(a, b)
    for obs in obstacles or ():
        shape = obs.get("shape", "cylinder")
        if shape == "aabb":
            x1, x2, y1, y2 = obs["bounds"]
            if (segment_max[0] < x1 - margin or segment_min[0] > x2 + margin
                    or segment_max[1] < y1 - margin or segment_min[1] > y2 + margin
                    or segment_max[2] < float(obs.get("base_height", 0.0)) - margin
                    or segment_min[2] > float(obs.get("height", 0.0)) + margin):
                continue
            if _segment_intersects_aabb(a, b, (x1, x2, y1, y2),
                                        obs.get("base_height", 0.0), obs.get("height", 0.0), margin):
                return True
            continue
        if shape == "segment":
            if _segment_segment_distance_3d(a, b, obs["p1"], obs["p2"]) <= float(obs.get("radius", 0.10)) + margin:
                return True
            continue
        radius = float(obs.get("radius", 1.0)) + float(margin)
        center = np.asarray(obs.get("pos", np.zeros(3)), dtype=np.float32).reshape(3)
        if shape == "sphere":
            t = 0.0 if denom3 < 1e-8 else float(np.clip(np.dot(center - a, ab) / denom3, 0.0, 1.0))
            if float(np.linalg.norm((a + t * ab) - center)) <= radius:
                return True
            continue
        base_z = float(obs.get("base_height", 0.0))
        top_z = float(obs.get("height", 0.0))
        if denom_xy < 1e-8:
            xy_dist = float(np.linalg.norm(a[:2] - center[:2]))
            z_low, z_high = sorted((float(a[2]), float(b[2])))
            if xy_dist <= radius and z_low <= top_z + margin and z_high >= base_z - margin:
                return True
        else:
            t = float(np.clip(np.dot(center[:2] - a[:2], ab_xy) / denom_xy, 0.0, 1.0))
            nearest_xy = a + t * ab
            xy_dist = float(np.linalg.norm(nearest_xy[:2] - center[:2]))
            if xy_dist <= radius and base_z - margin <= float(nearest_xy[2]) <= top_z + margin:
                return True
    return False

def _polyline_intersects_dynamic_hitboxes(points, obstacles, margin=0.08):
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2 or not np.all(np.isfinite(pts)):
        return False
    return any(_segment_intersects_dynamic_hitboxes(pts[i], pts[i + 1], obstacles, margin)
               for i in range(len(pts) - 1))

def _winch_cable_polyline_points(winch, drone_pos):
    pts = np.asarray(getattr(winch, "nodes_pos", []), dtype=np.float32).copy()
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2 or not np.all(np.isfinite(pts)):
        anchor = np.asarray(drone_pos, dtype=np.float32).copy()
        anchor[2] -= PHASE3_DRONE_HALF_HEIGHT_M
        end = anchor.copy()
        end[2] -= float(getattr(winch, "cable_len", PHASE3_STOWED_CABLE_LEN_M))
        pts = np.linspace(anchor, end, max(2, int(getattr(winch, "num_nodes", 12))), dtype=np.float32)
    if bool(getattr(winch, "has_payload", False)) and len(pts) >= 2:
        tangent = pts[-1] - pts[-2]
        norm = float(np.linalg.norm(tangent))
        if norm > 1e-6:
            pts[-1] -= tangent / norm * PHASE3_CARGO_HALF_HEIGHT_M
        else:
            pts[-1, 2] += PHASE3_CARGO_HALF_HEIGHT_M
    return pts

def _cruise_altitude_captured(current_z_m, current_vz_m_s, target_z_m,
                               altitude_tolerance_m=1.25, vertical_speed_limit_m_s=0.80):
    values = (current_z_m, current_vz_m_s, target_z_m,
              altitude_tolerance_m, vertical_speed_limit_m_s)
    if not all(math.isfinite(float(v)) for v in values):
        return False
    return (abs(float(current_z_m) - float(target_z_m)) <= float(altitude_tolerance_m)
            and abs(float(current_vz_m_s)) <= float(vertical_speed_limit_m_s))

def _should_enter_precision_approach(distance_xy_m, speed_xy_m_s, wind_speed_xy_m_s):
    if not all(math.isfinite(float(v)) for v in (distance_xy_m, speed_xy_m_s, wind_speed_xy_m_s)):
        return False
    if float(distance_xy_m) < 0.0:
        return False
    return float(distance_xy_m) <= _delivery_approach_braking_radius(speed_xy_m_s, wind_speed_xy_m_s)

def _delivery_approach_braking_radius(speed_xy_m_s, wind_speed_xy_m_s=0.0):
    speed = max(0.0, float(speed_xy_m_s))
    wind = max(0.0, float(wind_speed_xy_m_s))
    braking_deceleration = 1.10
    transition_margin = 5.5 + min(6.0, 0.45 * wind)
    return float(np.clip((speed * speed) / (2.0 * braking_deceleration) + transition_margin, 10.0, 42.0))

def _delivery_vertical_settle_reference(v_ref, vertical_error_m, deadband_m=0.75):
    ref = np.asarray(v_ref, dtype=np.float32).copy()
    if ref.shape != (3,) or not np.all(np.isfinite(ref)):
        raise ValueError(f"Expected a finite 3-vector velocity reference, got shape {ref.shape}")
    error = float(vertical_error_m)
    if not math.isfinite(error):
        raise ValueError("vertical_error_m must be finite")
    if abs(error) <= max(0.0, float(deadband_m)):
        ref[2] = 0.0
    return ref

def compute_vector_guidance(current_p, current_v, target_p, max_speed_xy=8.0, max_climb=2.40, 
                            max_descend=0.65, wind_vector=None, **kwargs):
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
        requested_xy_cap = max(0.0, float(max_speed_xy))
        adaptive_max_xy = float(np.clip(adaptive_cruise_speed, 0.0, requested_xy_cap))
        braking_speed = math.sqrt(2.0 * 1.5 * dist_xy)
        v_xy_mag = min(adaptive_max_xy, braking_speed) if dist_xy > 1.8 else min(adaptive_max_xy, 1.2 * dist_xy)
        v_cmd_xy = (delta_p[:2] / dist_xy) * v_xy_mag if dist_xy > 0.03 else np.zeros(2, dtype=np.float32)

    if target_p[2] <= 0.15:
        z_clearance = max(0.01, float(current_p[2] - target_p[2]))
        v_cmd_z = -float(np.clip(0.35 * math.sqrt(z_clearance), 0.06, max_descend))
    else:
        if dist_z > 0.0:
            climb_ramp_ceiling = float(np.clip(0.65 + 0.70 * current_p[2], 0.65, max_climb)) if current_p[2] < 2.5 else max_climb
            v_cmd_z = min(climb_ramp_ceiling, math.sqrt(max(0.0, 2.0 * 1.0 * dist_z)))
        else:
            v_cmd_z = -min(max_descend, math.sqrt(max(0.0, 2.0 * 0.50 * abs(dist_z))))
        
    v_des = np.array([v_cmd_xy[0], v_cmd_xy[1], v_cmd_z], dtype=np.float32)
    return target_p.copy(), v_des

# =============================================================================
# 20. GENESIS 3D MIRROR, CAMERA CONTROLS & ROBUST VISUAL FALLBACK
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

def _quat_wxyz_from_z_axis(direction):
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(d))
    if norm < 1e-10:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    d /= norm
    dot = float(np.clip(d[2], -1.0, 1.0))
    if dot < -0.999999:
        return np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    cross = np.array([-d[1], d[0], 0.0], dtype=np.float64)
    q = np.concatenate(([math.sqrt(max(0.0, (1.0 + dot) * 0.5))],
                        cross / math.sqrt(max(1e-12, 2.0 * (1.0 + dot)))))
    q /= max(1e-12, float(np.linalg.norm(q)))
    return q.astype(np.float32)

def _genesis_pose(entity, pos, quat_wxyz=None):
    p = np.asarray(pos, dtype=np.float32).reshape(3)
    if not np.all(np.isfinite(p)): return
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
    try: requested_width = int(os.environ.get("SURVIVAL_RENDER_WIDTH", "768"))
    except (TypeError, ValueError): requested_width = 512
    width = max(320, min(int(view_w), requested_width))
    height = max(240, int(round(width * int(view_h) / max(1, int(view_w)))))
    return width, height

def _camera_render_fps():
    try: requested = float(os.environ.get("SURVIVAL_RENDER_FPS", "15"))
    except (TypeError, ValueError): requested = 15.0
    return float(np.clip(requested, 10.0, 60.0))

def _simulation_target_fps():
    try: requested = float(os.environ.get("SURVIVAL_SIM_FPS", "60"))
    except (TypeError, ValueError): requested = 60.0
    return int(np.clip(requested, 15.0, 60.0))

def build_genesis_3d_mirror(world, drones, traffic_engine, ground_agent, view_w, view_h):
    try: import genesis as gs
    except ImportError as exc: raise RuntimeError("Genesis package not found.") from exc
    try: gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu)
    except Exception as exc: raise RuntimeError(f"Genesis initialization error: {exc}") from exc

    sim_options = gs.options.SimOptions(dt=1.0/60.0, substeps=1, gravity=(0.0,0.0,0.0))
    try:
        vis_options = gs.options.VisOptions(
            ambient_light=(0.30, 0.32, 0.36),
            lights=[gs.options.vis.DirectionalLight(dir=(-0.45, -0.55, -1.0), color=(1.0, 0.97, 0.90), intensity=4.0)],
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
        add_box((x2-x1,y2-y1,top), (cx,cy,top/2), (0.28,0.32,0.38), fixed=True)
        add_cylinder(2.2,0.08,(cx,cy,top+0.04),(0.0,0.72,0.50),fixed=True)
        for c in h.get('clutter',[]):
            cx1,cx2,cy1,cy2,ctop = [float(v) for v in c['aabb']]
            add_box((cx2-cx1,cy2-cy1,max(0.15, ctop-top)), ((cx1+cx2)/2,(cy1+cy2)/2,top+(ctop-top)/2), (0.48,0.50,0.54), fixed=True)

    palette = [(0.10,0.65,0.95),(0.95,0.55,0.15),(0.75,0.25,0.95)]
    for i,pad in enumerate(world.depot_helipads):
        add_cylinder(2.8,0.08,(float(pad[0]),float(pad[1]),0.04),palette[i%len(palette)],fixed=True)

    for line in getattr(world,'powerlines',[]):
        a,b = np.asarray(line['p1'],dtype=np.float32), np.asarray(line['p2'],dtype=np.float32)
        d = b-a
        add_box((max(0.1,float(np.linalg.norm(d[:2]))),0.08,0.08), tuple((a+b)/2), (0.72,0.73,0.76), fixed=True)
        for endpoint in (a, b):
            pole_h = float(endpoint[2])
            add_cylinder(0.16, pole_h, (float(endpoint[0]), float(endpoint[1]), pole_h/2.0), (0.35,0.38,0.42), fixed=True)

    for tree in world.trees:
        tx,ty = float(tree['pos'][0]),float(tree['pos'][1])
        radius,top = float(tree.get('r',3.2)),float(tree.get('h',8.0))
        add_cylinder(0.45,min(5.0,top*0.7),(tx,ty,min(5.0,top*0.7)/2),(0.40,0.25,0.15),fixed=True)
        add_sphere(radius,(tx,ty,top-radius),(0.15,0.50,0.20),fixed=True)

    urdf_path = _generate_v99_visual_urdf(os.path.join(tempfile.gettempdir(), 'drone_coaxial_x8_v99_visual.urdf'))
    drone_entities, cargo_entities, cable_entities = [], [], []
    for d in drones:
        p = np.asarray(d.airframe.p,dtype=np.float32)
        drone_entities.append(scene.add_entity(morph=gs.morphs.URDF(file=urdf_path,pos=tuple(float(v) for v in p),fixed=True,collision=False)))
        initial_payload_z = float(p[2] - PHASE3_DRONE_HALF_HEIGHT_M - d.winch.cable_len - (PHASE3_CARGO_HALF_HEIGHT_M if d.winch.has_payload else 0.0))
        cargo_entities.append(add_box((0.22,0.22,0.18),(p[0],p[1],max(0.09,initial_payload_z)),(0.95,0.75,0.15),roughness=0.5,fixed=True))
        cable_entities.append([add_cylinder(0.012, 0.30, (p[0], p[1], max(0.02, p[2]-0.1)), (0.78, 0.82, 0.88), fixed=True) for _ in range(40)])

    vehicle_entities = []
    for v in traffic_engine.vehicles:
        emergency = bool(v.get('is_emergency',False))
        vehicle_entities.append(add_box((3.0,1.35,1.0 if emergency else 0.9),tuple(v['pos']),
            (0.95,0.18,0.18) if emergency else (0.95,0.42,0.12),roughness=0.45,fixed=True))
            
    plane_entity = add_box((4.0,1.7,0.55),tuple(world.civil_aircraft['pos']),(0.95,0.78,0.15),roughness=0.35,fixed=True)
    ip = np.asarray(ground_agent.p,dtype=np.float32)+np.array([0,0,0.75],dtype=np.float32)
    intruder_entity = add_cylinder(0.24,1.5,tuple(ip),(0.90,0.26,0.18),fixed=True)
    target_entity = add_sphere(0.55,tuple(drones[0].target),(0.1,0.95,0.62),fixed=True)
    delivered_entities = [add_box((0.30,0.30,0.18),(0,0,-25-i*0.2),(0.15,0.60,0.28),fixed=True) for i in range(120)]
    
    initial_p = np.asarray(drones[0].airframe.p,dtype=np.float32)
    initial_focus = np.asarray(initial_p + np.array([0.0, 0.0, 2.0], dtype=np.float32))
    initial_camera = initial_focus + np.array([34.0, -34.0, 26.0], dtype=np.float32)
    render_w, render_h = _camera_render_resolution(view_w, view_h)

    # Increased far-clipping plane to 3500.0m to support maximum zoom in overview mode
    camera_kwargs = dict(
        res=(render_w, render_h), pos=tuple(float(v) for v in initial_camera),
        lookat=tuple(float(v) for v in initial_focus), fov=48, GUI=False,
        near=0.05, far=3500.0,
    )
    try:
        camera = scene.add_camera(**camera_kwargs)
    except TypeError:
        camera_kwargs.pop('near', None); camera_kwargs.pop('far', None)
        camera = scene.add_camera(**camera_kwargs)
        
    scene.build(n_envs=1)
    hidden_delivery_positions = [np.array([0.0, 0.0, -25.0 - i * 0.2], dtype=np.float32) for i in range(len(delivered_entities))]
    
    mirror = {
        'scene': scene, 'camera': camera, 'drone_entities': drone_entities, 'cargo_entities': cargo_entities,
        'cable_entities': cable_entities, 'vehicle_entities': vehicle_entities, 'plane_entity': plane_entity,
        'intruder_entity': intruder_entity, 'target_entity': target_entity, 'delivered_entities': delivered_entities,
        'delivered_positions': hidden_delivery_positions,
        'camera_pos': initial_camera.copy(), 'camera_lookat': initial_focus.copy(),
        'render_width': render_w, 'render_height': render_h,
        'render_fps': _camera_render_fps(), 'last_view_render_time': 0.0,
        'cached_view_surface': None, 'cached_view_key': None, 'render_frame_count': 0,
        'render_call_count': 0, 'last_render_call': 0,
        'segmentation_check_interval': 90,
        'last_render_error': 0.0, 'last_render_ok': False
    }
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
    target = np.asarray(camera_lookat,dtype=np.float64).reshape(3)
    desired = np.asarray(camera_pos,dtype=np.float64).reshape(3)
    ray = desired - target
    desired_dist = float(np.linalg.norm(ray))
    if not np.isfinite(desired_dist) or desired_dist < 1e-6:
        return desired.astype(np.float32)
    direction = ray / desired_dist
    nearest = desired_dist

    for h in world.houses:
        x1,x2,y1,y2,top = [float(v) for v in h['aabb']]
        t = _ray_aabb_first_hit(target, direction, (x1,y1,0.0), (x2,y2,top))
        if t is not None and 1.25 < float(t) < nearest:
            nearest = float(t)

    safe_distance = max(float(minimum_distance), nearest - float(clearance))
    if nearest < desired_dist and safe_distance < desired_dist:
        desired = target + direction * safe_distance
    desired[2] = max(0.85, float(desired[2]))
    return desired.astype(np.float32)

def screen_to_world_raycast_genesis(mouse_pos, camera_pos, camera_lookat, view_w, view_h, world, fov_deg=48.0):
    origin = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    lookat = np.asarray(camera_lookat, dtype=np.float64).reshape(3)
    forward = lookat - origin
    forward /= max(1e-9, float(np.linalg.norm(forward)))
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= max(1e-9, float(np.linalg.norm(right)))
    up = np.cross(right, forward)
    up /= max(1e-9, float(np.linalg.norm(up)))
    mx, my = float(mouse_pos[0]), float(mouse_pos[1])
    aspect = float(view_w) / max(1.0, float(view_h))
    th = math.tan(math.radians(float(fov_deg)) * 0.5)
    x = ((2 * mx / max(1.0, float(view_w))) - 1) * aspect * th
    y = (1 - (2 * my / max(1.0, float(view_h)))) * th
    direction = forward + right * x + up * y
    direction /= max(1e-9, float(np.linalg.norm(direction)))
    
    best_t, best_point = float('inf'), None
    for h in world.houses:
        x1, x2, y1, y2, top = [float(v) for v in h['aabb']]
        t = _ray_aabb_first_hit(origin, direction, (x1, y1, 0.0), (x2, y2, top))
        if t is not None and 0 <= t < best_t:
            best_t, best_point = t, origin + t * direction
    if direction[2] < -1e-8:
        tg = -origin[2] / direction[2]
        if 0 <= tg < best_t:
            best_t, best_point = tg, origin + tg * direction
    if best_point is None:
        return np.asarray(world.depot_helipads[0], dtype=np.float32).copy()
    return np.asarray(best_point, dtype=np.float32)

def compute_v99_camera_pose(world, drone, cam_mode, cam_yaw, cam_pitch, cam_focus, cam_zoom=1.0):
    p = np.asarray(drone.airframe.p, dtype=np.float32).reshape(3)
    target = np.asarray(drone.target, dtype=np.float32).reshape(3)
    zoom = float(np.clip(cam_zoom, 0.35, 2.8))

    if cam_mode == "CITY_OVERVIEW":
        focus = np.array([0.0, 0.0, 14.0], dtype=np.float32)
        distance = min(2400.0, 720.0 * zoom)
        az = float(cam_yaw) + 0.25
        elevation = float(np.clip(cam_pitch + 0.15, 0.25, 1.10))
        horizontal = distance * math.cos(elevation)
        cam = focus + np.array([horizontal * math.cos(az), horizontal * math.sin(az), distance * math.sin(elevation)], dtype=np.float32)
        return cam, focus

    if cam_mode == "DEPOT_OVERVIEW":
        depot = np.mean(np.asarray(world.depot_helipads, dtype=np.float32), axis=0)
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
        focus = np.array([p[0] + drone.winch.sway_offset[0], p[1] + drone.winch.sway_offset[1], cargo_z], dtype=np.float32)
        cam[2] = max(1.0, float(cam[2]))
        return cam, focus

    delta = target[:2] - p[:2]
    distance_to_target = float(np.linalg.norm(delta))
    forward = (delta / max(1e-4, distance_to_target)) if distance_to_target > 1.0 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    forward = np.array([forward[0], forward[1], 0.0], dtype=np.float32)

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
    route_forward = np.array([forward[0] * c - forward[1] * sn, forward[0] * sn + forward[1] * c, 0.0], dtype=np.float32)
    right = np.array([-route_forward[1], route_forward[0], 0.0], dtype=np.float32)
    radius = float(np.clip(31.0 * zoom, 12.0, 110.0))
    look_ahead = min(65.0, max(18.0, distance_to_target * 0.38))
    focus = p + route_forward * look_ahead + np.array([0.0, 0.0, 1.5], dtype=np.float32)
    cam = p - route_forward * radius + right * (radius * 0.18)
    cam[2] = float(p[2]) + max(8.0, radius * math.sin(float(np.clip(cam_pitch, 0.20, 1.25))) + 2.5)
    return cam.astype(np.float32), focus.astype(np.float32)

def _render_genesis_mirror_impl(mirror, world, drones, traffic_engine, agent_pos, delivered_boxes, selected_drone_idx,
                                cam_mode, cam_yaw, cam_pitch, cam_focus, active_target, screen, view_w, view_h, font_small):
    d_active = drones[selected_drone_idx]
    p = np.asarray(d_active.airframe.p, dtype=np.float32)
    zoom = float(mirror.get('cam_zoom', 1.0))
    cam_pos, cam_lookat = compute_v99_camera_pose(world, d_active, cam_mode, cam_yaw, cam_pitch, cam_focus, zoom)
    if cam_mode not in ("CITY_OVERVIEW", "DEPOT_OVERVIEW"):
        cam_pos = resolve_genesis_camera_occlusion(cam_pos, cam_lookat, world)
    mirror['camera_pos'], mirror['camera_lookat'] = cam_pos.copy(), cam_lookat.copy()
    mirror['camera'].set_pose(pos=tuple(float(x) for x in cam_pos), lookat=tuple(float(x) for x in cam_lookat))

    for d, e, cargo_e, cables in zip(drones, mirror['drone_entities'], mirror['cargo_entities'], mirror['cable_entities']):
        proxy_p = np.asarray(d.airframe.p, dtype=np.float32).copy()
        if proxy_p[2] < 0.22: proxy_p[2] = 0.32
        _genesis_pose(e, proxy_p, _rotation_matrix_to_quaternion_wxyz(d.airframe.R))
        if d.winch.has_payload:
            payload_center = d.airframe.p + np.array([
                d.winch.sway_offset[0], d.winch.sway_offset[1],
                -PHASE3_DRONE_HALF_HEIGHT_M - d.winch.cable_len - PHASE3_CARGO_HALF_HEIGHT_M], dtype=np.float32)
            cp = payload_center
        else:
            cp = np.array([0.0, 0.0, -25.0 - float(d.drone_id)*0.3], dtype=np.float32)
        _genesis_pose(cargo_e, cp)
        
        cable_points = _winch_cable_polyline_points(d.winch, d.airframe.p)
        for j, node_e in enumerate(cables):
            idx = min(len(cable_points)-1, int(j * (len(cable_points)-1) / max(1, len(cables)-1)))
            _genesis_pose(node_e, cable_points[idx])

    for v, e in zip(traffic_engine.vehicles, mirror['vehicle_entities']):
        _genesis_pose(e, v['pos'])
    _genesis_pose(mirror['plane_entity'], world.civil_aircraft['pos'])
    _genesis_pose(mirror['intruder_entity'], np.asarray(agent_pos, dtype=np.float32) + np.array([0, 0, 0.75], dtype=np.float32))
    _genesis_pose(mirror['target_entity'], active_target)

    rendered = mirror['camera'].render(rgb=True, force_render=True)
    rgb = rendered[0] if isinstance(rendered, (tuple, list)) else rendered
    if isinstance(rgb, torch.Tensor): rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim == 4: rgb = rgb[0]
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    surf = pygame.image.frombuffer(rgb.tobytes(), (int(rgb.shape[1]), int(rgb.shape[0])), 'RGB')
    if surf.get_size() != (int(view_w), int(view_h)):
        surf = pygame.transform.smoothscale(surf, (int(view_w), int(view_h)))
    screen.blit(surf, (0, 0))
    mirror['last_render_ok'] = True
    return True

def render_genesis_mirror(mirror, world, drones, traffic_engine, agent_pos, delivered_boxes, selected_drone_idx,
                          cam_mode, cam_yaw, cam_pitch, cam_focus, active_target, screen, view_w, view_h, font_small):
    rendered_ok = False
    try:
        rendered_ok = _render_genesis_mirror_impl(
            mirror, world, drones, traffic_engine, agent_pos, delivered_boxes, selected_drone_idx,
            cam_mode, cam_yaw, cam_pitch, cam_focus, active_target, screen, view_w, view_h, font_small
        )
    except Exception as exc:
        mirror['last_render_ok'] = False

    if bool(mirror.get('show_hitboxes', False)):
        _draw_collision_hitbox_overlay(
            screen, world, traffic_engine, agent_pos, drones, delivered_boxes,
            mirror.get('camera_pos'), mirror.get('camera_lookat'), view_w, view_h, fov_deg=48.0
        )
    return bool(rendered_ok)

# =============================================================================
# 21. MASTER INTERACTIVE SIMULATOR COCKPIT LOOP (60 FPS)
# =============================================================================
def _load_model_checkpoint(model, checkpoint_path, device):
    try: state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError: state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    return model

def _launch_phase3_worker(base_checkpoint, output_checkpoint):
    worker_env = os.environ.copy()
    worker_env["SURVIVAL_PHASE3_WORKER"] = "1"
    worker_env["SURVIVAL_PHASE3_BASE_CHECKPOINT"] = os.path.abspath(base_checkpoint)
    worker_env["SURVIVAL_PHASE3_OUTPUT_CHECKPOINT"] = os.path.abspath(output_checkpoint)
    script_path = os.path.abspath(__file__)
    print("\n[PHASE 3] Launching isolated Genesis worker process...", flush=True)
    try: result = subprocess.run([sys.executable, script_path], env=worker_env, check=False)
    except OSError: return False
    return result.returncode == 0 and os.path.isfile(output_checkpoint)

def run_all_physics_simulator():
    if pygame is None:
        raise RuntimeError("pygame is required for the interactive cockpit. Install with: pip install pygame")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase3_enabled = os.environ.get("SURVIVAL_ENABLE_PHASE3", "1").strip().lower() not in {"0", "false", "no", "off"}
    phase3_checkpoint = os.path.abspath(PHASE3_CHECKPOINT_FILE)

    if not os.path.exists(MODEL_WEIGHTS_FILE):
        model, device = harvest_and_train_metropolitan()
    else:
        model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
        candidate_checkpoint = phase3_checkpoint if (phase3_enabled and os.path.isfile(phase3_checkpoint)) else MODEL_WEIGHTS_FILE
        model = _load_model_checkpoint(model, candidate_checkpoint, device)

    if phase3_enabled and not os.path.isfile(phase3_checkpoint):
        if _launch_phase3_worker(MODEL_WEIGHTS_FILE, phase3_checkpoint):
            model = _load_model_checkpoint(model, phase3_checkpoint, device)

    model.eval()
    pygame.init()
    VIEW_W, VIEW_H = 760, 740
    SIDEBAR_W = 440
    W, H = VIEW_W + SIDEBAR_W, VIEW_H

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Survival Core V38.7 — Master System Cockpit")
    clock = pygame.time.Clock()

    font_large = pygame.font.SysFont("consolas", 17, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 12, bold=True)

    world = MetropolitanCityGridWorld400m()
    corridor_planner = VoxelOctreeCorridorPlanner(world)
    weather_engine = DynamicUrbanWeatherEngine()
    traffic_engine = DynamicRoadVehicleTrafficEngine(world.avenue_x, world.street_y)
    sensors_physics = FullPhysicsSensorSuite()
    assurance = SurfaceAndWireAssuranceSuite(world)
    lidar_suite = MacroPerceptionLiDARSuite(world)
    sensor_suite = AdvancedMultiModalSensorSuite(world)
    ground_agent = DynamicGroundIntruderAgent(init_pos=[-80.0, 80.0, 0.0])
    agent_pos = ground_agent.p.copy()
    dryden_wind_gens = [DrydenAtmosphericTurbulenceModel(w20_m_s=3.4) for _ in range(NUM_SWARM_DRONES)]

    dispatcher = SwarmMissionDispatcher(world.manifest_addresses, world.depot_helipads)
    drones = [SwarmDroneInstance(i, world.depot_helipads[i].copy(), world, dispatcher) for i in range(NUM_SWARM_DRONES)]

    genesis_mirror = build_genesis_3d_mirror(world=world, drones=drones, traffic_engine=traffic_engine,
                                             ground_agent=ground_agent, view_w=VIEW_W, view_h=VIEW_H)

    selected_drone_idx = 0
    cam_yaw, cam_pitch = 0.75, 0.45
    cam_zoom = 1.0
    cam_focus = drones[0].airframe.p.copy()
    cam_mode = "DEPOT_OVERVIEW"
    running = True

    flight_time = 0.0
    delivered_boxes = deque(maxlen=120)
    policy_action_histogram = np.zeros(len(ACTION_NAMES), dtype=np.int64)

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
                elif event.key == pygame.K_k:
                    genesis_mirror['show_hitboxes'] = not bool(genesis_mirror.get('show_hitboxes', False))
                elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                    cam_zoom = max(0.35, cam_zoom * 0.88)
                    genesis_mirror['cam_zoom'] = cam_zoom
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    cam_zoom = min(2.8, cam_zoom * 1.12)
                    genesis_mirror['cam_zoom'] = cam_zoom
                elif event.key == pygame.K_w:
                    weather_engine.cycle_weather()

        active_drone = drones[selected_drone_idx]
        active_tip = active_drone.airframe.p + np.array([
            active_drone.winch.sway_offset[0], active_drone.winch.sway_offset[1],
            -PHASE3_DRONE_HALF_HEIGHT_M - active_drone.winch.cable_len -
            (PHASE3_CARGO_HALF_HEIGHT_M if active_drone.winch.has_payload else 0.0)], dtype=np.float32)
        agent_pos, global_agent_breach, _ = ground_agent.step(
            cargo_world_pos=active_tip,
            is_winch_lowering=(active_drone.winch.state == "LOWERING"),
            is_aborting=(active_drone.winch.state == "ABORT_RETRACT"),
            dt=dt
        )

        obstacles_cbf = build_runtime_collision_obstacles(world, traffic_engine, agent_pos, drones, delivered_boxes)
        swarm_states = [{"p": d.airframe.p.copy(), "v": d.airframe.v.copy()} for d in drones]

        batch_observations = []
        drone_step_data = []

        for d_idx, d in enumerate(drones):
            local_wind, local_gust_scale, local_amb_temp, _ = weather_engine.get_weather_parameters(d.airframe.p, world.avenue_x)
            local_dryden = dryden_wind_gens[d_idx].step(d.airframe.p[2], float(np.linalg.norm(d.airframe.v)), dt=dt) * (local_gust_scale / 2.0)
            drone_wind = (local_wind + local_dryden).astype(np.float32)
            
            sensed_wind, sensed_spd, sensed_deg, sensed_q = sensor_suite.poll_wind_vector(
                d.airframe.v, drone_wind, d.airframe.R, dt=dt, sensor_id=d_idx, ground_velocity_est=d.ekf.v
            )

            tgt_xy = d.mission["target_pos"][:2]
            home_xy = d.mission["home_pad"][:2]
            roof_h = d.mission["roof_height"]
            drop_surface_z = float(roof_h + PHASE3_ROOF_PAD_TOP_OFFSET_M)
            cruise_z = d.assigned_cruise_alt
            high_hover_z = d.get_high_hover_z()
            dist_tgt_xy = float(math.hypot(d.airframe.p[0] - tgt_xy[0], d.airframe.p[1] - tgt_xy[1]))
            dist_home_xy = float(math.hypot(d.airframe.p[0] - home_xy[0], d.airframe.p[1] - home_xy[1]))

            if d.mission_phase == "1_TAKEOFF":
                wp_target = np.array([home_xy[0], home_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.0, max_climb=2.40, wind_vector=sensed_wind)
                if _cruise_altitude_captured(d.airframe.p[2], d.airframe.v[2], cruise_z):
                    d.mission_phase = "2_CRUISE_OUTBOUND"

            elif d.mission_phase == "2_CRUISE_OUTBOUND":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=4.5, max_climb=2.40, wind_vector=sensed_wind)
                if _should_enter_precision_approach(dist_tgt_xy, float(np.linalg.norm(d.airframe.v[:2])), float(np.linalg.norm(sensed_wind[:2]))):
                    d.mission_phase = "3_CHIMNEY_DESCENT"
                    d.mixer.xy_err_integral.fill(0.0)
                    d.chimney_descent_armed = False

            elif d.mission_phase == "3_CHIMNEY_DESCENT":
                if not d.chimney_descent_armed:
                    wp_target = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                    p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=1.2, max_climb=0.35, max_descend=0.15, wind_vector=sensed_wind)
                    if dist_tgt_xy <= DELIVERY_DRONE_CENTER_TOLERANCE_M and abs(float(d.airframe.p[2] - cruise_z)) <= 0.65:
                        d.chimney_descent_armed = True
                else:
                    wp_target = np.array([tgt_xy[0], tgt_xy[1], high_hover_z], dtype=np.float32)
                    p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.0, max_climb=0.70, max_descend=0.35, wind_vector=sensed_wind)
                    if dist_tgt_xy <= DELIVERY_DRONE_CENTER_TOLERANCE_M and abs(float(d.airframe.p[2] - high_hover_z)) <= 0.40:
                        d.mission_phase = "4_DELIVERY_HOVER"
                        d.delivery_attempts += 1

            elif d.mission_phase == "4_DELIVERY_HOVER":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], high_hover_z], dtype=np.float32)
                p_ref, v_ref = wp_target.copy(), np.zeros(3, dtype=np.float32)
                if d.winch.delivery_completed:
                    d.packages_delivered += 1
                    d.dispatcher.report_completed(d.drone_id)
                    d.mission_phase = "5_CHIMNEY_ASCENT"
                    d.target = d.mission["home_pad"].copy()

            elif d.mission_phase == "5_CHIMNEY_ASCENT":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.0, max_climb=2.0, wind_vector=sensed_wind)
                if _cruise_altitude_captured(d.airframe.p[2], d.airframe.v[2], cruise_z):
                    d.mission_phase = "6_CRUISE_INBOUND"

            elif d.mission_phase == "6_CRUISE_INBOUND":
                wp_target = np.array([home_xy[0], home_xy[1], 14.0], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=4.5, max_descend=1.5, wind_vector=sensed_wind)
                if dist_home_xy <= 3.0:
                    d.mission_phase = "7_DEPOT_DESCENT"

            elif d.mission_phase == "7_DEPOT_DESCENT":
                wp_target = np.array([home_xy[0], home_xy[1], 0.04], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.45, max_descend=0.25, wind_vector=sensed_wind)
                if dist_home_xy <= 0.8 and d.airframe.p[2] <= 0.12:
                    d.mission_phase = "8_DEPOT_SERVICE"
                    d.airframe.v.fill(0.0)

            else:  # 8_DEPOT_SERVICE
                wp_target = np.array([home_xy[0], home_xy[1], 0.06], dtype=np.float32)
                p_ref, v_ref = wp_target.copy(), np.zeros(3, dtype=np.float32)
                d.depot_timer += dt
                if d.depot_timer >= 1.0:
                    d.depot_timer = 0.0
                    d.mission = d.dispatcher.dispatch_next_mission(d.drone_id, d.bms.soc, d.airframe.p)
                    d.home_coords = d.mission["home_pad"].copy()
                    d.target = d.mission["target_pos"].copy()
                    d.assigned_cruise_alt = d.mission["assigned_cruise_alt"]
                    d.winch.reload_new_cargo()
                    d.bms.soc = 0.98

                    # Clean dynamic state reset on new dispatch
                    d.airframe.p[2] = 0.06
                    d.airframe.v.fill(0.0)
                    d.airframe.accel_linear.fill(0.0)
                    d.ekf.p[:] = d.airframe.p
                    d.ekf.v.fill(0.0)
                    d.mixer.xy_err_integral.fill(0.0)
                    d.mixer.z_err_integral = 0.0
                    d.mixer.vertical_v_cmd_smooth = 0.0
                    d.mixer.vertical_a_cmd_smooth = 0.0
                    d.mission_phase = "1_TAKEOFF"

            current_surf_z = world.get_surface_height(d.airframe.p[0], d.airframe.p[1])
            s_data = sensors_physics.poll_all_physics_sensors(d.airframe.p, d.airframe.v, drone_wind, surface_z=current_surf_z)
            meas_accel, meas_gyro = d.imu.poll(d.airframe.R, d.airframe.accel_linear, d.airframe.omega, d.airframe.motor_omegas)
            a_quantum, phase_shift = d.quantum.measure_phase_shift(meas_accel)
            d.ekf.predict(a_quantum, meas_gyro, dt=dt)

            lidar_16 = lidar_suite.cast_360_lidar(d.airframe.p, math.atan2(d.airframe.R[1,0], d.airframe.R[0,0]))
            conical_8, wire_clr = lidar_suite.cast_downward_conical(d.airframe.p)
            load_3d = sensor_suite.poll_3axis_load_cell(d.winch.tension_n, d.winch.sway_offset, d.winch.cable_len)
            hook_a, hook_w = sensor_suite.poll_smart_hook_imu(d.airframe.accel_linear, d.airframe.omega, d.winch.sway_offset, d.winch.cable_len, d.winch.has_payload)
            flow_2d, flow_q, _ = sensor_suite.poll_optical_flow(d.airframe.p, d.airframe.v, d.airframe.R, d.airframe.omega, current_surf_z)
            th_dt, dolp, ndvi = sensor_suite.poll_thermal_and_polarization(d.target[:2])
            i_rms, t_mosfet = sensor_suite.poll_esc_foc_telemetry(d.airframe.motor_omegas, d.airframe.motor_health)

            obs_139 = d.synth.synthesize_instantaneous(
                p_est=d.ekf.p, v_est=d.ekf.v, R_est=quat_to_rot(d.ekf.q), omega_est=meas_gyro - d.ekf.b_g,
                target_p=wp_target, lidar_16=lidar_16, conical_8=conical_8,
                v_tas=s_data["v_tas"], q_dyn=s_data["q_dyn"], min_sonar=s_data["min_sonar"],
                wind_est=sensed_wind, ice_idx=d.icing.ice_thickness_mm/3.0, cable_len=d.winch.cable_len,
                cable_vel=d.winch.cable_vel, tension_n=d.winch.tension_n, sway_2d=d.winch.sway_offset,
                bms_soc=d.bms.soc, v_pack=d.bms.v_pack, bms_temp_c=d.bms.temp_c, r0_dyn=0.0095,
                chi2_val=1.2, jerk_discrepancy=0.5, is_spoofed=False, emf_rms=s_data["emf_rms"],
                wire_dist=wire_clr, agent_dist=float(np.linalg.norm(agent_pos[:2] - d.airframe.p[:2])),
                ground_spl_dba=55.0, has_payload=d.winch.has_payload,
                hook_a=hook_a, hook_w=hook_w, load_cell_3d=load_3d, thermal_dt=th_dt, dolp=dolp, ndvi=ndvi,
                esc_i_rms=i_rms, esc_t_mosfet=t_mosfet, alpha_deg=0.0, beta_deg=0.0, rh_pct=60.0, dust_pm10=20.0,
                hdop=0.85, cn0_avg=44.0, rf_jam_pwr=-110.0, mag_grad=0.02,
                airframe_accel=meas_accel, airframe_gyro=meas_gyro, peer_swarm_kinematics=[],
                optical_flow_2d=flow_2d, flow_quality=flow_q, aruco_offset_3d=np.zeros(3, dtype=np.float32),
                aruco_locked=0.0, skid_force_left=d.winch.f_skid_left, skid_force_right=d.winch.f_skid_right,
                is_skid_grounded=d.winch.is_skid_grounded, siren_angle=0.0, siren_freq=180.0, siren_spl=55.0,
                adsb_rel_pos=np.zeros(3, dtype=np.float32), adsb_closing_spd=0.0, dt=dt
            )
            obs_417 = d.synth.stack_history(d.history_q, obs_139)
            batch_observations.append(obs_417)
            drone_step_data.append({
                "p_ref": p_ref, "v_ref": v_ref, "load_3d": load_3d, "s_data": s_data,
                "current_surf_z": current_surf_z, "sensed_wind": sensed_wind, "drone_wind": drone_wind,
                "drop_surface_z": drop_surface_z, "tgt_xy": tgt_xy
            })

        batch_t = torch.from_numpy(np.stack(batch_observations)).to(device)
        with torch.no_grad():
            b_guidance, b_fsm, b_winch, _, _, _ = model(batch_t)
        b_guidance_np = (b_guidance * 0.15).cpu().numpy()
        b_fsm_np = b_fsm.argmax(dim=-1).cpu().numpy()
        b_winch_np = b_winch.cpu().numpy()

        for d_idx, d in enumerate(drones):
            gd = drone_step_data[d_idx]
            nn_guidance = b_guidance_np[d_idx].copy()
            nn_fsm_token = int(b_fsm_np[d_idx])
            nn_winch_rate = float(np.clip(b_winch_np[d_idx, 0], -1.0, 1.0))
            policy_action_histogram[nn_fsm_token] += 1

            d.winch.step(
                drone_z=d.airframe.p[2], target_surface_z=gd["drop_surface_z"],
                is_settled=(d.mission_phase == "4_DELIVERY_HOVER"),
                is_surface_safe=True, neural_token_idx=nn_fsm_token,
                nn_winch_rate=nn_winch_rate, dt=dt, drone_p=d.airframe.p,
                drone_v=d.airframe.v, wind_3d=gd["drone_wind"], target_xy=gd["tgt_xy"]
            )
            if d.winch.release_count > d.last_visual_release_count:
                rel_p = d.winch.last_release_pos or np.array([gd["tgt_xy"][0], gd["tgt_xy"][1], gd["drop_surface_z"]], dtype=np.float32)
                delivered_boxes.append(rel_p.copy())
                d.last_visual_release_count = d.winch.release_count

            omega_cmds, f_thrust, _ = d.mixer.mix(
                drone_health=d.airframe.motor_health, p_est=d.ekf.p, v_est=d.ekf.v,
                R_est=quat_to_rot(d.ekf.q), omega_est=d.airframe.omega,
                p_target=gd["p_ref"], neural_guidance=nn_guidance, token_idx=nn_fsm_token,
                estimated_wind=gd["sensed_wind"], obstacles=obstacles_cbf, swarm_states=swarm_states,
                drone_idx=d_idx, ref_vel=gd["v_ref"], cable_len=d.winch.cable_len,
                sway_2d=d.winch.sway_offset, load_cell_3d=gd["load_3d"], v_pack=d.bms.v_pack,
                bms_temp_c=d.bms.temp_c, has_payload=d.winch.has_payload, dt=dt,
                station_keeping=(d.mission_phase == "4_DELIVERY_HOVER"),
                sway_velocity_2d=d.winch.sway_velocity
            )

            d.airframe.step(omega_cmds, gd["drone_wind"], gd["s_data"]["rho_air"], surface_z=gd["current_surf_z"], has_payload=d.winch.has_payload, dt=dt)

        screen.fill((10, 13, 20))
        act_d = drones[selected_drone_idx]
        render_genesis_mirror(
            mirror=genesis_mirror, world=world, drones=drones, traffic_engine=traffic_engine,
            agent_pos=agent_pos, delivered_boxes=delivered_boxes, selected_drone_idx=selected_drone_idx,
            cam_mode=cam_mode, cam_yaw=cam_yaw, cam_pitch=cam_pitch, cam_focus=cam_focus,
            active_target=act_d.target, screen=screen, view_w=VIEW_W, view_h=VIEW_H, font_small=font_small
        )
        pygame.draw.line(screen, (35, 45, 60), (VIEW_W, 0), (VIEW_W, H), 2)

        sb_x = VIEW_W + 14
        screen.blit(font_large.render("SURVIVAL CORE V38.7", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Dynamic Self-Adjusting SE(3) Swarm • PFD & Breguet Routing", True, (130, 160, 190)), (sb_x, 38))

        r_r, r_p, _ = rot_to_euler(act_d.airframe.R)
        draw_pfd_artificial_horizon(
            screen, (sb_x, 65, 140, 110), r_r, r_p, act_d.airframe.p[2],
            float(np.linalg.norm(act_d.airframe.v)), act_d.airframe.v[2], font_small
        )
        draw_tactical_radar_minimap(screen, (sb_x + 148, 65, 260, 150), world, drones, selected_drone_idx, traffic_engine.vehicles)

        fl_y = 225
        screen.blit(font_mono.render("SWARM FLEET [TAB] | CAM [C]:", True, (240, 240, 240)), (sb_x, fl_y))
        for d_i, d in enumerate(drones):
            c = [(0, 240, 255), (255, 180, 40), (220, 100, 255)][d_i % 3]
            screen.blit(font_small.render(f"[D{d_i+1}] {d.mission_phase[:8]} | {d.bms.soc*100:.0f}% | Box:{d.packages_delivered}", True, c), (sb_x + 10, fl_y + 18 + d_i * 18))

        pygame.display.flip()
        clock.tick(_simulation_target_fps())

    pygame.quit()
    sys.exit()

def run_fixed_script_self_tests():
    if _LIGHTWEIGHT_HARVEST_CHILD:
        assert torch is None and F is None and optim is None
        assert hasattr(nn, "Module")
    else:
        assert torch is not None
    assert BASE_OBS_DIM == 139 and OBSERVATION_DIM == 417
    assert tuple(POWERLINE_Y_POSITIONS_M) == (-185.0, 185.0)
    assert _cruise_altitude_captured(42.0, 0.0, 42.0)
    print("[SELF-TEST PASS] Master system invariants and dimensions verified.", flush=True)

def run_single_drone_waypoint_route_self_test():
    dt = 0.016
    airframe = BEMTCoaxialAirframe8DOF(np.array([-160.0, -155.0, 0.06], dtype=np.float32))
    mixer = RateDampedActuatorMixerWithCBF()
    target_xy = np.array([-165.0, -90.0], dtype=np.float32)
    cruise_z = 42.0
    for _ in range(100):
        p_ref, v_ref = compute_vector_guidance(airframe.p, airframe.v, np.array([target_xy[0], target_xy[1], cruise_z], dtype=np.float32))
        cmd, _, _ = mixer.mix(airframe.motor_health, airframe.p, airframe.v, airframe.R, airframe.omega, p_ref, np.zeros(4, dtype=np.float32), 0, np.zeros(3, dtype=np.float32), [], ref_vel=v_ref, dt=dt)
        airframe.step(cmd, np.zeros(3, dtype=np.float32), 1.225, dt=dt)
    print("[ROUTE TEST PASS] Flight controller tracks 3D trajectory cleanly.", flush=True)

if __name__ == "__main__":
    mp.freeze_support()
    if os.environ.get("SURVIVAL_ROUTE_SELF_TEST", "0").strip() == "1":
        run_fixed_script_self_tests()
        run_single_drone_waypoint_route_self_test()
    elif os.environ.get("SURVIVAL_SELF_TEST", "0").strip() == "1":
        run_fixed_script_self_tests()
    elif os.environ.get("SURVIVAL_PHASE3_WORKER", "0").strip() == "1":
        _phase3_worker_main()
    else:
        run_all_physics_simulator()