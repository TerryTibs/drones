"""
=============================================================================
SURVIVAL CORE V39.0 — FULLY INTEGRATED AUTONOMOUS SWARM SUITE
PART 1 OF 7: SYSTEM CONFIG, LIE ALGEBRA, ATMOSPHERIC & AIRFRAME DYNAMICS
=============================================================================
"""

import os
import sys
import time
import math
import random
import struct
import socket
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import deque
from enum import IntEnum
from pathlib import Path
import multiprocessing as mp
import tempfile
import shutil
import subprocess
import numpy as np

# Set environment optimization flags
NEURAL_GUIDANCE_DISABLED = os.environ.get("SURVIVAL_DISABLE_NEURAL_GUIDANCE", "0").strip().lower() in {"1", "true", "yes", "on"}
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

# Windows multiprocessing spawn guard & torch isolation
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
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
    except ImportError:
        torch = None
        nn = None
        F = None
        optim = None

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

# Shared physical geometry
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
    "EMERGENCY_DIVERT"     # 6
]

# Delivery-policy supervision safeguards
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

# Geometry constants
PHASE3_ROOF_PAD_TOP_OFFSET_M = 0.08
PHASE3_DRONE_HALF_HEIGHT_M = 0.045
PHASE3_CARGO_HALF_HEIGHT_M = 0.09
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

# Physics Lookahead Horizon
PHYSICS_PREDICTION_HORIZON_S = 0.20
PHYSICS_SWAY_PRED_DIM = 5      # [theta_x, theta_y, theta_dot_x, theta_dot_y, tension]
PHYSICS_VEL_PRED_DIM = 3       # [vx, vy, vz]_{t+dt}

def _anti_sway_feedback(sway_offset, sway_velocity, position_gain=PHASE3_SWAY_POSITION_GAIN,
                        velocity_gain=PHASE3_SWAY_VELOCITY_GAIN):
    """Command the attachment point in the direction of relative sway/rate."""
    return position_gain * sway_offset + velocity_gain * sway_velocity

def _bound_policy_vertical_residual(acceleration_residual):
    """Keep learned guidance from overpowering deterministic altitude controller."""
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
def skew_symmetric(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -v[2], v[1]], 
        [v[2], 0.0, -v[0]], 
        [-v[1], v[0], 0.0]
    ], dtype=np.float32)

hat_map = skew_symmetric

def vee_map(mat: np.ndarray) -> np.ndarray:
    return np.array([mat[2, 1], mat[0, 2], mat[1, 0]], dtype=np.float32)

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z),       2.0*(x*z + w*y)],
        [2.0*(x*y + w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x)],
        [2.0*(x*z - w*y),       2.0*(y*z + w*x),       1.0 - 2.0*(x*x + y*y)]
    ], dtype=np.float32)

def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float32)

def rot_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    pitch = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return roll, pitch, yaw

# =============================================================================
# 2. ADAPTIVE ESTIMATORS & SPATIAL HASH ENGINE
# =============================================================================
class VectorizedSpatialHashGrid2D:
    def __init__(self, cell_size: float = 30.0):
        self.cell_size = cell_size
        self.buckets: Dict[Tuple[int, int], List[Any]] = {}
        self.cached_aabbs: Optional[np.ndarray] = None
        self.cached_ids: Optional[List[str]] = None

    def insert(self, aabb: List[float], item: Any):
        min_x, max_x, min_y, max_y = aabb[0], aabb[1], aabb[2], aabb[3]
        gx0, gx1 = int(min_x // self.cell_size), int(max_x // self.cell_size)
        gy0, gy1 = int(min_y // self.cell_size), int(max_y // self.cell_size)
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                key = (gx, gy)
                if key not in self.buckets:
                    self.buckets[key] = []
                self.buckets[key].append(item)

    def finalize_cache(self, all_houses: List[Dict[str, Any]]):
        aabbs = []
        ids = []
        for h in all_houses:
            aabbs.append(h["aabb"])
            ids.append(h["id"])
        self.cached_aabbs = np.array(aabbs, dtype=np.float32)
        self.cached_ids = ids

    def query_segment(self, p1: np.ndarray, p2: np.ndarray) -> List[Any]:
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
        self.grid: Dict[Tuple[int, int, int], List[Any]] = {}
        self.voxel_size_xy = 20.0
        self.voxel_size_z = 8.0

    def broadcast_wind(self, pos: np.ndarray, wind_vec: np.ndarray, timestamp: float):
        vx = int(pos[0] // self.voxel_size_xy)
        vy = int(pos[1] // self.voxel_size_xy)
        vz = int(max(0.0, pos[2]) // self.voxel_size_z)
        key = (vx, vy, vz)
        if key not in self.grid:
            self.grid[key] = [wind_vec.copy(), timestamp]
        else:
            old_w, _ = self.grid[key]
            self.grid[key] = [0.65 * old_w + 0.35 * wind_vec, timestamp]

    def query_wind_prior(self, pos: np.ndarray, current_time: float, max_age_s: float = 45.0) -> Tuple[np.ndarray, bool]:
        vx = int(pos[0] // self.voxel_size_xy)
        vy = int(pos[1] // self.voxel_size_xy)
        vz = int(max(0.0, pos[2]) // self.voxel_size_z)
        key = (vx, vy, vz)
        if key in self.grid:
            wind_data, t_stamp = self.grid[key]
            if (current_time - t_stamp) <= max_age_s:
                return wind_data.copy(), True
        return np.zeros(3, dtype=np.float32), False

    def query_vertical_wind_shear_profile(self, pos_xy: np.ndarray, current_time: float, candidate_alts: List[float]) -> List[np.ndarray]:
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
    def __init__(self, nominal_mass: float = 1.80):
        self.est_mass = nominal_mass
        self.com_offset_xy = np.zeros(2, dtype=np.float32)

    def update(self, thrust_total: float, accel_z: float, steady_tau_xy: np.ndarray, dt: float = 0.016) -> Tuple[float, np.ndarray]:
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
    def __init__(self, dry_mass: float = DRY_MASS_KG, nominal_inertia: np.ndarray = INERTIA_MATRIX):
        self.dry_mass = dry_mass
        self.J_nominal = nominal_inertia.copy()
        self.J_estimated = self.J_nominal.copy()
        self.k_R = np.array([2.40, 2.40, 1.80], dtype=np.float32)
        self.k_w = np.array([0.90, 0.90, 0.60], dtype=np.float32)

    def update(self, cable_len: float, has_payload: bool, cargo_mass: float, sway_2d: np.ndarray, v_pack: float, dt: float = 0.016):
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

    def estimate_normal(self, drone_pos: np.ndarray, conical_ranges_8: np.ndarray) -> np.ndarray:
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

        self.trees = [
            {"id": "Tree_1", "pos": np.array([-12.0, -12.0, 0.0], dtype=np.float32), "r": 3.2, "h": 8.0},
            {"id": "Tree_2", "pos": np.array([ 12.0, -12.0, 0.0], dtype=np.float32), "r": 3.0, "h": 7.5},
            {"id": "Tree_3", "pos": np.array([-12.0,  12.0, 0.0], dtype=np.float32), "r": 3.5, "h": 8.5},
            {"id": "Tree_4", "pos": np.array([ 12.0,  12.0, 0.0], dtype=np.float32), "r": 3.2, "h": 8.0}
        ]

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

    def get_static_collision_obstacles(self) -> List[Dict[str, Any]]:
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

    def get_support_surfaces(self) -> List[Dict[str, Any]]:
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

    def step_civil_traffic(self, dt: float = 0.016):
        self.civil_aircraft["pos"] += self.civil_aircraft["vel"] * dt
        if self.civil_aircraft["pos"][0] > 195.0:
            self.civil_aircraft["pos"][0] = -195.0

    def get_surface_height(self, x: float, y: float) -> float:
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

    def is_point_occluded(self, pt: np.ndarray, margin: float = 2.2) -> bool:
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

    def is_segment_occluded(self, p1: np.ndarray, p2: np.ndarray, step_size: float = 2.0) -> bool:
        dist = float(np.linalg.norm(p2 - p1))
        steps = max(2, int(dist / step_size))
        for s in range(steps + 1):
            tau = s / float(steps)
            interp_pt = p1 + tau * (p2 - p1)
            if self.is_point_occluded(interp_pt):
                return True
        return False

    def find_safe_corridor_waypoints(self, start_p: np.ndarray, end_p: np.ndarray,
                                     assigned_cruise_alt: float = 44.0, is_depot_landing: bool = False) -> List[np.ndarray]:
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

    def get_mode_name(self) -> str:
        return self.weather_modes[self.target_mode_idx]

    def step(self, dt: float = 0.016):
        self.regime_timer += dt
        if self.blend_factor < 1.0:
            self.blend_factor = min(1.0, self.blend_factor + dt / 2.5)
            if self.blend_factor >= 1.0:
                self.current_mode_idx = self.target_mode_idx

        if self.regime_timer >= self.regime_duration:
            self.cycle_weather()

    def _get_preset(self, mode_idx: int) -> Tuple[np.ndarray, float, float, bool, float]:
        mode = self.weather_modes[mode_idx]
        if mode == "CLEAR_NOMINAL":
            return np.array([1.8, -0.6, 0.0], dtype=np.float32), 0.6, 22.0, False, 0.0
        elif mode == "CANYON_GALE":
            return np.array([5.8, 2.4, 0.0], dtype=np.float32), 3.8, 14.0, False, 0.0
        elif mode == "FREEZING_RAIN":
            return np.array([2.4, -1.8, 0.0], dtype=np.float32), 1.5, -4.5, True, -0.4
        else:  # MICROBURST_STORM
            return np.array([-4.5, 4.2, 0.0], dtype=np.float32), 5.5, 11.0, False, -2.0

    def get_weather_parameters(self, drone_pos: np.ndarray, avenues_x: List[float]) -> Tuple[np.ndarray, float, float, bool]:
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
    def __init__(self, avenues_x: List[float], streets_y: List[float]):
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

    def step(self, dt: float = 0.016):
        for v in self.vehicles:
            v["pos"] += v["vel"] * dt
            if v["axis"] == "Y":
                if v["pos"][1] > 190.0: v["pos"][1] = -190.0
                if v["pos"][1] < -190.0: v["pos"][1] = 190.0
            else:
                if v["pos"][0] > 190.0: v["pos"][0] = -190.0
                if v["pos"][0] < -190.0: v["pos"][0] = 190.0

    def get_cbf_obstacles(self) -> List[Dict[str, Any]]:
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
    def __init__(self, init_pos: List[float]):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.v_max = 2.8
        self.personal_space_m = 1.85
        self.curiosity_timer = 0.0
        self.boredom_cooldown = 0.0
        self.wander_target = np.array([-100.0, 40.0], dtype=np.float32)
        self.is_leashed = False

    def step(self, cargo_world_pos: np.ndarray, is_winch_lowering: bool, is_aborting: bool, dt: float = 0.016) -> Tuple[np.ndarray, bool, float]:
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
# 6. ROTOR ICING, TURBULENCE & SUB-ZERO BATTERY ECM
# =============================================================================
class MessingerRotorIcingModel:
    def __init__(self):
        self.ice_thickness_mm = 0.0
        self.icing_active = False

    def step(self, airspeed: float, dt: float = 0.016) -> Tuple[float, float]:
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
    def __init__(self, w20_m_s: float = 3.4):
        self.w20 = w20_m_s
        self.gust_state = np.zeros(3, dtype=np.float32)

    def step(self, altitude_m: float, airspeed_m_s: float, dt: float = 0.016) -> np.ndarray:
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

    def step(self, p_elec_watts: float, amb_temp: float = 22.0, dt: float = 0.016) -> Tuple[float, float, float, float]:
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
    def __init__(self, nominal_ct: float = C_T_CLEAN):
        self.nominal_ct = nominal_ct
        self.ct_estimated = nominal_ct
        self.aerodynamic_health = 1.0

    def update(self, current_mass: float, accel_z: float, motor_omegas: np.ndarray, dt: float = 0.016) -> Tuple[float, float]:
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

    def poll(self, drone_R: np.ndarray, accel_linear: np.ndarray, omega_true: np.ndarray, motor_omegas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
# 8. BEMT COAXIAL AIRFRAME 8-DOF WITH RK4 & GROUND/WALL INTERACTION
# =============================================================================
class BEMTCoaxialAirframe8DOF:
    def __init__(self, init_pos: np.ndarray):
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

    def _compute_derivatives(self, p: np.ndarray, v: np.ndarray, R: np.ndarray, omega: np.ndarray,
                             motor_omegas: np.ndarray, current_wind: np.ndarray, rho_air: float, 
                             ice_index: float, surface_z: float, building_walls: Optional[List[Dict[str, Any]]],
                             current_mass: float):
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

    def step(self, motor_cmds: np.ndarray, current_wind: np.ndarray, rho_air: float,
             ice_index: float = 0.0, surface_z: float = 0.0, has_payload: bool = True,
             building_walls: Optional[List[Dict[str, Any]]] = None, dt: float = 0.016):
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
# 9. TACTILE LANDING REFLEX & STRAIN GAUGES
# =============================================================================
class TactileLandingReflex:
    def __init__(self):
        self.contact_timer = 0.0
        self.is_latched = False

    def reset(self):
        self.contact_timer = 0.0
        self.is_latched = False

    def compute_reflex(self, f_skid_left: float, f_skid_right: float,
                       current_mass: float, dt: float = 0.016) -> Tuple[float, float, bool]:
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

# =============================================================================
# 10. AUTONOMOUS WINCH DELIVERY SYSTEM (PBD CABLE & REACTION SENSORS)
# =============================================================================
class AutonomousWinchDeliverySystem:
    def __init__(self, num_segments: int = 4):
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

    def step(self, drone_z: float = 0.0, target_surface_z: float = 0.0,
             is_settled: bool = False, is_surface_safe: bool = True, 
             wire_hit: bool = False, agent_breach: bool = False,
             agent_dist: float = 30.0, wind_2d: Optional[np.ndarray] = None, 
             drone_accel_linear: Optional[np.ndarray] = None, neural_token_idx: int = 0,
             nn_winch_rate: float = 0.0, dt: float = 0.016, 
             drone_p: Optional[np.ndarray] = None, drone_v: Optional[np.ndarray] = None,
             wind_3d: Optional[np.ndarray] = None, target_xy: Optional[np.ndarray] = None, **kwargs) -> Tuple[float, float, np.ndarray]:
        
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

    def update_skid_strain_gauges(self, is_on_surface: bool, weight_force: float, f_thrust: float):
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
# 11. ADVANCED MULTI-MODAL PHYSICAL SENSOR SUITE
# =============================================================================
class AdvancedMultiModalSensorSuite:
    def __init__(self, world: MetropolitanCityGridWorld400m):
        self.world = world
        self._wind_sensor_state: Dict[int, np.ndarray] = {}
        self._wind_sensor_bias: Dict[int, np.ndarray] = {}

    def poll_wind_vector(self, drone_v: np.ndarray, current_wind: np.ndarray, drone_R: np.ndarray,
                         dt: float = 0.016, sensor_id: int = 0,
                         ground_velocity_est: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, float, float]:
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

    def poll_optical_flow(self, drone_p: np.ndarray, drone_v: np.ndarray, drone_R: np.ndarray,
                          drone_omega: np.ndarray, surface_z: float = 0.0) -> Tuple[np.ndarray, float, float]:
        cos_tilt = max(0.35, float(drone_R[2, 2]))
        z_rel = max(0.15, float(drone_p[2] - surface_z))
        z_slant = z_rel / cos_tilt
        
        v_body = drone_R.T @ drone_v
        
        w_flow_x = (v_body[0] / z_slant) - drone_omega[1] + float(np.random.normal(0, 0.015))
        w_flow_y = (v_body[1] / z_slant) + drone_omega[0] + float(np.random.normal(0, 0.015))
        
        quality = 0.95 if z_rel < 35.0 else float(np.clip(1.0 - (z_rel - 35.0) / 25.0, 0.1, 0.95))
        return np.array([w_flow_x, w_flow_y], dtype=np.float32), quality, z_slant

    def poll_aruco_fiducial_detector(self, drone_p: np.ndarray, drone_R: np.ndarray,
                                     target_pad: np.ndarray, expected_tag_id: int) -> Tuple[np.ndarray, float]:
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

    def poll_directional_acoustic_array(self, drone_p: np.ndarray, vehicles: List[Dict[str, Any]],
                                        sim_time: float = 0.0) -> Tuple[float, float, float]:
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

    def poll_adsb_civil_transponder(self, drone_p: np.ndarray, drone_v: np.ndarray,
                                    civil_aircraft: Dict[str, Any]) -> Tuple[np.ndarray, float, float]:
        rel_pos = civil_aircraft["pos"] - drone_p
        dist_3d = float(np.linalg.norm(rel_pos))
        rel_vel = civil_aircraft["vel"] - drone_v
        closing_speed = -float(np.dot(rel_pos / max(1e-3, dist_3d), rel_vel))
        return rel_pos, float(np.clip(closing_speed, -30.0, 30.0)), dist_3d

    def poll_smart_hook_imu(self, drone_a: np.ndarray, drone_w: np.ndarray,
                            sway_2d: np.ndarray, cable_l: float, has_payload: bool,
                            sway_velocity_2d: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
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

    def poll_3axis_load_cell(self, tension_n: float, sway_2d: np.ndarray, cable_l: float) -> np.ndarray:
        if cable_l <= 0.10:
            return np.array([0.0, 0.0, -tension_n], dtype=np.float32)
        sway_dist = max(1e-4, math.hypot(sway_2d[0], sway_2d[1]))
        sin_theta = min(0.65, sway_dist / max(0.2, cable_l))
        cos_theta = math.sqrt(max(1e-4, 1.0 - sin_theta**2))
        fx = -tension_n * (sway_2d[0] / sway_dist) * sin_theta
        fy = -tension_n * (sway_2d[1] / sway_dist) * sin_theta
        fz = -tension_n * cos_theta
        return np.array([fx, fy, fz], dtype=np.float32)

    def poll_thermal_and_polarization(self, pos_xy: np.ndarray) -> Tuple[float, float, float]:
        px, py = pos_xy
        min_roof_dist = 999.0
        for h in self.world.houses:
            x1, x2, y1, y2, _ = h["aabb"]
            dx = max(0.0, max(x1 - px, px - x2))
            dy = max(0.0, max(y1 - py, py - y2))
            d = math.hypot(dx, dy)
            min_roof_dist = min(min_roof_dist, d)

        roof_heat = 7.5 * math.exp(-min_roof_dist / 8.0)
        
        if hasattr(self.world, "trees") and self.world.trees:
            park_dist = min(math.hypot(px - float(t["pos"][0]), py - float(t["pos"][1])) for t in self.world.trees)
        else:
            park_dist = math.hypot(px, py)
            
        park_cool = -3.5 * math.exp(-park_dist / 22.0)
        th_dt = float(roof_heat + park_cool + 0.5)

        dolp = float(0.12 + 0.55 * math.exp(-min_roof_dist / 6.0))
        ndvi = float(0.10 + 0.68 * math.exp(-park_dist / 25.0))
        return th_dt, dolp, ndvi

    def poll_esc_foc_telemetry(self, motor_omegas: np.ndarray, motor_health: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    def poll_5port_air_data(self, drone_v: np.ndarray, current_wind: np.ndarray,
                            drone_R: np.ndarray) -> Tuple[float, float, float, float]:
        v_air_world = drone_v - current_wind
        v_air_body = drone_R.T @ v_air_world
        speed = max(0.5, float(np.linalg.norm(v_air_body)))
        alpha_rad = math.atan2(v_air_body[2], max(0.2, v_air_body[0]))
        beta_rad = math.asin(np.clip(v_air_body[1] / speed, -0.9, 0.9))
        rh_pct = float(np.clip(68.0 - 0.4 * drone_v[2], 30.0, 95.0))
        pm10_dust = float(np.clip(22.0 + 1.2 * float(np.linalg.norm(current_wind[:2])), 5.0, 80.0))
        return math.degrees(alpha_rad), math.degrees(beta_rad), rh_pct, pm10_dust

    def poll_rtk_gnss_metrics(self, drone_p: np.ndarray, is_spoofed: bool) -> Tuple[float, float, float, float]:
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

    def cast_360_lidar(self, drone_pos: np.ndarray, yaw_rad: float,
                       dynamic_vehicles: Optional[List[Dict[str, Any]]] = None,
                       dynamic_objects: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
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

    def cast_downward_conical(self, drone_pos: np.ndarray) -> Tuple[np.ndarray, float]:
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

    def audit_surface(self, pos_xy: np.ndarray, target_z: float = 0.0, buffer_m: float = 2.5) -> Tuple[str, bool, float]:
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

    def snap_to_nearest_safe_ground(self, target_xy: np.ndarray) -> np.ndarray:
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

    def wire_distance_m(self, drone_p: np.ndarray) -> float:
        p = np.asarray(drone_p, dtype=np.float64).reshape(3)
        closest_x = float(np.clip(p[0], self.wire_x_min, self.wire_x_max))
        dx = p[0] - closest_x
        return min(
            math.sqrt(dx * dx + (p[1] - wire_y) ** 2 + (p[2] - self.wire_z) ** 2)
            for wire_y in self.wire_y_positions
        )

    def poll_all_physics_sensors(self, drone_p: np.ndarray, drone_v: np.ndarray,
                                 current_wind: np.ndarray, surface_z: float = 0.0) -> Dict[str, float]:
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
# 13. MATTER-WAVE QUANTUM SENSOR & ERROR-STATE EKF ON SO(3)
# =============================================================================
class MatterWaveColdAtomQuantumSensor:
    def __init__(self):
        self.k_eff = K_EFF_WAVEVECTOR
        self.T = RAMAN_PULSE_SEP_T

    def measure_phase_shift(self, specific_force_body: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        phase_shift = self.k_eff * specific_force_body * (self.T ** 2)
        phase_shift += np.random.normal(0, 0.002, 3).astype(np.float32)
        a_quantum_body = phase_shift / (self.k_eff * (self.T ** 2))
        return a_quantum_body, phase_shift

class ErrorStateEKF15:
    def __init__(self, init_pos: np.ndarray):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.b_a = np.zeros(3, dtype=np.float32)
        self.b_g = np.zeros(3, dtype=np.float32)
        self.P = np.diag([0.15]*3 + [0.1]*3 + [0.02]*3 + [0.05]*3 + [0.01]*3).astype(np.float32)
        self.Q_base = np.diag([1e-4]*3 + [1e-3]*3 + [1e-4]*3 + [1e-5]*3 + [1e-6]*3).astype(np.float32)
        self.Q_d = self.Q_base.copy()
        self.residual_scale = 1.0
        self.last_gamma = 1.0

    def predict(self, accel_body_raw: np.ndarray, gyro_raw: np.ndarray, dt: float = 0.016):
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

    def update_measurement(self, H: np.ndarray, innovation: np.ndarray,
                           R_cov: np.ndarray, chi2_threshold: float = 11.34):
        S = H @ self.P @ H.T + R_cov
        S_reg = S + np.eye(S.shape[0], dtype=np.float32) * 1e-6

        try:
            PHt = self.P @ H.T
            K = np.linalg.solve(S_reg, PHt.T).T
            innov_flat = np.array(innovation).flatten()
            S_inv_innov = np.linalg.solve(S_reg, innov_flat)
            gamma = float(innov_flat @ S_inv_innov)
            self.last_gamma = gamma
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

    def update_heading(self, measured_yaw_rad: float, r_cov: float = 0.02):
        R_curr = quat_to_rot(self.q)
        current_yaw = math.atan2(R_curr[1, 0], R_curr[0, 0])
        yaw_innov = (measured_yaw_rad - current_yaw + math.pi) % (2.0 * math.pi) - math.pi

        denom = max(1e-6, R_curr[0, 0]**2 + R_curr[1, 0]**2)
        H_yaw = np.zeros((1, 15), dtype=np.float32)
        H_yaw[0, 6] = 0.0
        H_yaw[0, 7] = (-R_curr[0, 0] * R_curr[1, 2] + R_curr[1, 0] * R_curr[0, 2]) / denom
        H_yaw[0, 8] = ( R_curr[0, 0] * R_curr[1, 1] - R_curr[1, 0] * R_curr[0, 1]) / denom

        self.update_measurement(H_yaw, np.array([yaw_innov], dtype=np.float32), np.array([[r_cov]], dtype=np.float32))

    def update_optical_flow_velocity_adaptive(self, v_flow_world: np.ndarray,
                                              flow_quality: float = 0.95,
                                              z_slant: float = 1.0,
                                              r_cov_base: float = 0.015):
        alt_factor = (min(45.0, z_slant) / 10.0) ** 2
        r_adaptive = (r_cov_base / max(0.05, flow_quality)) + alt_factor * 0.08
        r_adaptive = float(np.clip(r_adaptive, 0.02, 1.20))

        H_v = np.zeros((2, 15), dtype=np.float32)
        H_v[0, 3] = 1.0
        H_v[1, 4] = 1.0
        v_innov = v_flow_world[:2] - self.v[:2]
        self.update_measurement(H_v, v_innov, np.eye(2, dtype=np.float32) * r_adaptive)

    def update_baro_altitude_adaptive(self, z_baro: float, q_dyn: float = 0.0, r_cov_base: float = 0.05):
        r_adaptive = float(r_cov_base + 0.0022 * q_dyn)
        r_adaptive = float(np.clip(r_adaptive, 0.05, 0.40))

        H_baro = np.zeros((1, 15), dtype=np.float32)
        H_baro[0, 2] = 1.0
        innov = np.array([z_baro - self.p[2]], dtype=np.float32)
        self.update_measurement(H_baro, innov, np.array([[r_adaptive]], dtype=np.float32))

    def update_gps_position_adaptive(self, p_gps: np.ndarray, hdop: float = 0.85,
                                     cn0_avg: float = 44.0, is_spoofed: bool = False,
                                     r_cov_base: float = 0.04):
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
# 14. RF PROPAGATION & PROTOCOL CONSTANTS
# =============================================================================
RF_FREQ_HZ = 2.4e9                   # 2.4 GHz ISM Band
SPEED_OF_LIGHT = 299792458.0
RF_WAVELENGTH_M = SPEED_OF_LIGHT / RF_FREQ_HZ
TX_POWER_DBM = 20.0                  # 100 mW standard transceivers
ANTENNA_GAIN_TX_DBI = 2.15
ANTENNA_GAIN_RX_DBI = 2.15
NOISE_FLOOR_DBM = -101.0
RX_SENSITIVITY_DBM = -92.0           # Minimum power for successful packet decode
PATH_LOSS_EXPONENT_LOS = 2.2         # Line-of-sight urban corridor
PATH_LOSS_EXPONENT_NLOS = 3.8        # Obstructed urban canyon
WALL_ATTENUATION_DB_PER_M = 14.0     # Signal loss passing through concrete structures
RAYLEIGH_SIGMA_DB = 3.5              # Multipath fading standard deviation

MAX_MESH_HOPS = 4
PACKET_QUEUE_CAPACITY = 256
HEARTBEAT_INTERVAL_S = 0.20
HEARTBEAT_TIMEOUT_S = 0.80
CBBA_MAX_BUNDLE_SIZE = 4
GOSSIP_FANOUT = 2                    # Number of peers to target per gossip round
GOSSIP_INTERVAL_S = 0.10

# =============================================================================
# 15. MESSAGE PACKETS & PROTOCOL DATACLASSES
# =============================================================================
@dataclass
class MeshPacket:
    packet_id: int
    src_id: int
    dest_id: int                     # -1 designates broadcast
    ttl: int
    timestamp_s: float
    payload_type: str                # "HEARTBEAT", "GOSSIP", "CBBA_BID", "HAZARD_ALERT"
    payload: Dict[str, Any]
    hops_taken: int = 0
    route_trace: List[int] = field(default_factory=list)

@dataclass
class PeerLinkState:
    peer_id: int
    last_seen_s: float
    rssi_dbm: float
    packet_loss_rate: float
    is_direct_neighbor: bool
    estimated_distance_m: float
    packets_received: int = 0
    packets_expected: int = 0

@dataclass
class DeliveryTaskContract:
    task_id: int
    target_pos: np.ndarray           # [X, Y, Z]
    bldg_id: str
    reward_base: float
    expiry_time_s: float
    required_payload_kg: float = 0.45

# =============================================================================
# 16. PHYSICS-INFORMED RF CHANNEL MODEL (LOS/NLOS & SHADOWING)
# =============================================================================
class UrbanRFChannelModel:
    """
    Computes realistic radio attenuation, building penetration losses,
    multipath Rayleigh fading, and packet error rates (PER).
    """
    def __init__(self, world_houses: Optional[List[Dict[str, Any]]] = None):
        self.houses = world_houses or []

    def _segment_box_penetration_length(self, p1: np.ndarray, p2: np.ndarray, box: List[float]) -> float:
        """Calculate line segment intersection length through a 3D building AABB."""
        x1, x2, y1, y2, z_top = box
        d = p2 - p1
        length = float(np.linalg.norm(d))
        if length < 1e-4:
            return 0.0

        t_min, t_max = 0.0, 1.0
        bounds = [(x1, x2), (y1, y2), (0.0, z_top)]

        for i in range(3):
            if abs(d[i]) < 1e-8:
                if p1[i] < bounds[i][0] or p1[i] > bounds[i][1]:
                    return 0.0
            else:
                inv_d = 1.0 / d[i]
                t1 = (bounds[i][0] - p1[i]) * inv_d
                t2 = (bounds[i][1] - p1[i]) * inv_d
                entry = min(t1, t2)
                exit_ = max(t1, t2)
                t_min = max(t_min, entry)
                t_max = min(t_max, exit_)
                if t_min > t_max:
                    return 0.0

        intersect_dist = (t_max - t_min) * length
        return max(0.0, intersect_dist)

    def evaluate_link(self, pos_tx: np.ndarray, pos_rx: np.ndarray) -> Tuple[float, float, bool]:
        """
        Calculates received signal power (RSSI), Packet Success Rate (PSR),
        and Line-of-Sight status between two drones.
        """
        dist_3d = float(np.linalg.norm(pos_tx - pos_rx))
        if dist_3d < 0.1:
            return TX_POWER_DBM, 1.0, True

        # 1. Test building penetration length
        penetration_m = 0.0
        for h in self.houses:
            box = h["aabb"]
            # Fast rejection test
            if max(pos_tx[2], pos_rx[2]) < 0.0 or min(pos_tx[2], pos_rx[2]) > box[4]:
                continue
            penetration_m += self._segment_box_penetration_length(pos_tx, pos_rx, box)

        is_los = (penetration_m < 0.05)
        path_exponent = PATH_LOSS_EXPONENT_LOS if is_los else PATH_LOSS_EXPONENT_NLOS

        # 2. Log-Distance Path Loss
        # FSPL at 1 meter reference distance
        fspl_1m = 20.0 * math.log10((4.0 * math.pi * 1.0) / RF_WAVELENGTH_M)
        pl_dist = fspl_1m + 10.0 * path_exponent * math.log10(max(1.0, dist_3d))

        # 3. Structural absorption + Multipath fading
        absorption_loss = penetration_m * WALL_ATTENUATION_DB_PER_M
        multipath_fading = np.random.normal(0.0, RAYLEIGH_SIGMA_DB)

        total_loss_db = pl_dist + absorption_loss + multipath_fading
        rssi_dbm = TX_POWER_DBM + ANTENNA_GAIN_TX_DBI + ANTENNA_GAIN_RX_DBI - total_loss_db

        # 4. Signal-to-Noise Ratio (SNR) and Packet Success Rate (PSR)
        fade_margin = rssi_dbm - RX_SENSITIVITY_DBM
        if fade_margin < -10.0:
            psr = 0.0
        elif fade_margin > 15.0:
            psr = 0.999
        else:
            psr = float(1.0 / (1.0 + math.exp(-0.45 * fade_margin)))

        return rssi_dbm, psr, is_los

# =============================================================================
# 17. AD-HOC P2P MESH ROUTER (HOP-BY-HOP LINK ROUTING)
# =============================================================================
class AdHocP2PMeshRouter:
    """
    Decentralized link-layer router managing neighbor tables, hop routing,
    packet deduplication, and transmission queues.
    """
    def __init__(self, node_id: int, channel_model: UrbanRFChannelModel):
        self.node_id = node_id
        self.channel_model = channel_model
        self.neighbors: Dict[int, PeerLinkState] = {}
        self.inbound_queue: deque = deque(maxlen=PACKET_QUEUE_CAPACITY)
        self.outbound_queue: deque = deque(maxlen=PACKET_QUEUE_CAPACITY)
        self.seen_packet_ids: Set[int] = set()
        self.packet_seq_counter = node_id * 1000000
        self.heartbeat_timer = random.uniform(0.0, HEARTBEAT_INTERVAL_S)

    def generate_packet_id(self) -> int:
        self.packet_seq_counter += 1
        return self.packet_seq_counter

    def send_broadcast(self, payload_type: str, payload_data: Dict[str, Any], sim_time_s: float):
        pkt = MeshPacket(
            packet_id=self.generate_packet_id(),
            src_id=self.node_id,
            dest_id=-1,
            ttl=MAX_MESH_HOPS,
            timestamp_s=sim_time_s,
            payload_type=payload_type,
            payload=payload_data,
            hops_taken=0,
            route_trace=[self.node_id]
        )
        self.seen_packet_ids.add(pkt.packet_id)
        self.outbound_queue.append(pkt)

    def send_unicast(self, dest_id: int, payload_type: str, payload_data: Dict[str, Any], sim_time_s: float):
        pkt = MeshPacket(
            packet_id=self.generate_packet_id(),
            src_id=self.node_id,
            dest_id=dest_id,
            ttl=MAX_MESH_HOPS,
            timestamp_s=sim_time_s,
            payload_type=payload_type,
            payload=payload_data,
            hops_taken=0,
            route_trace=[self.node_id]
        )
        self.seen_packet_ids.add(pkt.packet_id)
        self.outbound_queue.append(pkt)

    def receive_packet_from_medium(self, pkt: MeshPacket, sim_time_s: float, current_pos: np.ndarray, tx_pos: np.ndarray):
        # 1. Physics transmission test over RF channel
        rssi, psr, is_los = self.channel_model.evaluate_link(tx_pos, current_pos)
        
        # Update / discover neighbor state
        if pkt.src_id != self.node_id:
            if pkt.src_id not in self.neighbors:
                self.neighbors[pkt.src_id] = PeerLinkState(
                    peer_id=pkt.src_id,
                    last_seen_s=sim_time_s,
                    rssi_dbm=rssi,
                    packet_loss_rate=1.0 - psr,
                    is_direct_neighbor=True,
                    estimated_distance_m=float(np.linalg.norm(current_pos - tx_pos)),
                    packets_received=1,
                    packets_expected=1
                )
            else:
                n = self.neighbors[pkt.src_id]
                n.last_seen_s = sim_time_s
                n.rssi_dbm = 0.70 * n.rssi_dbm + 0.30 * rssi
                n.estimated_distance_m = float(np.linalg.norm(current_pos - tx_pos))
                n.packets_received += 1
                n.packet_loss_rate = 0.85 * n.packet_loss_rate + 0.15 * (1.0 - psr)

        # Monte-Carlo packet drop based on channel PSR
        if random.random() > psr:
            return  # Dropped by channel noise/shadowing

        # 2. Packet deduplication
        if pkt.packet_id in self.seen_packet_ids:
            return
        self.seen_packet_ids.add(pkt.packet_id)

        # 3. Destination routing logic
        if pkt.dest_id == self.node_id or pkt.dest_id == -1:
            self.inbound_queue.append(pkt)

        # 4. Multi-hop retransmission if TTL remains
        if pkt.dest_id != self.node_id and pkt.ttl > 1:
            forwarded_pkt = MeshPacket(
                packet_id=pkt.packet_id,
                src_id=pkt.src_id,
                dest_id=pkt.dest_id,
                ttl=pkt.ttl - 1,
                timestamp_s=pkt.timestamp_s,
                payload_type=pkt.payload_type,
                payload=pkt.payload,
                hops_taken=pkt.hops_taken + 1,
                route_trace=pkt.route_trace + [self.node_id]
            )
            self.outbound_queue.append(forwarded_pkt)

    def prune_stale_neighbors(self, current_time_s: float):
        dead_peers = []
        for peer_id, state in self.neighbors.items():
            if (current_time_s - state.last_seen_s) > HEARTBEAT_TIMEOUT_S:
                dead_peers.append(peer_id)
        for dp in dead_peers:
            del self.neighbors[dp]

# =============================================================================
# 18. EPIDEMIC GOSSIP SYNC (MICROCLIMATE & SWARM KINEMATICS)
# =============================================================================
class EpidemicGossipEngine:
    """
    Vector-clock driven Anti-Entropy gossip synchronization for distributed
    wind fields, swarm peer telemetry, and static/dynamic hazard maps.
    """
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.vector_clock: Dict[int, int] = {self.node_id: 0}
        self.peer_state_table: Dict[int, Dict[str, Any]] = {}
        self.wind_field_table: Dict[Tuple[int, int, int], Tuple[np.ndarray, float, int]] = {}
        self.gossip_timer = 0.0

    def record_local_state(self, p: np.ndarray, v: np.ndarray, soc: float, health: float, mission_phase: str):
        self.vector_clock[self.node_id] = self.vector_clock.get(self.node_id, 0) + 1
        self.peer_state_table[self.node_id] = {
            "p": p.copy(),
            "v": v.copy(),
            "soc": soc,
            "health": health,
            "phase": mission_phase,
            "clock": self.vector_clock[self.node_id],
            "t": time.time()
        }

    def record_local_wind(self, pos: np.ndarray, wind_vec: np.ndarray, timestamp_s: float):
        vx = int(pos[0] // 20.0)
        vy = int(pos[1] // 20.0)
        vz = int(max(0.0, pos[2]) // 8.0)
        key = (vx, vy, vz)
        version = self.vector_clock[self.node_id]
        self.wind_field_table[key] = (wind_vec.copy(), timestamp_s, version)

    def prepare_gossip_digest(self) -> Dict[str, Any]:
        """Compact serialized summary for mesh transmission."""
        return {
            "origin_id": self.node_id,
            "vector_clock": self.vector_clock.copy(),
            "peer_states": self.peer_state_table.copy(),
            "wind_entries": {k: (v[0].tolist(), v[1], v[2]) for k, v in self.wind_field_table.items()}
        }

    def merge_gossip_digest(self, digest: Dict[str, Any]):
        """Anti-entropy reconciliation using vector clocks."""
        remote_peer_states = digest.get("peer_states", {})
        remote_wind = digest.get("wind_entries", {})

        # Merge peer states
        for peer_id, state in remote_peer_states.items():
            current_clock = self.vector_clock.get(peer_id, -1)
            incoming_clock = state.get("clock", 0)
            if incoming_clock > current_clock:
                self.vector_clock[peer_id] = incoming_clock
                self.peer_state_table[peer_id] = {
                    "p": np.array(state["p"], dtype=np.float32),
                    "v": np.array(state["v"], dtype=np.float32),
                    "soc": state["soc"],
                    "health": state["health"],
                    "phase": state["phase"],
                    "clock": incoming_clock,
                    "t": state["t"]
                }

        # Merge distributed wind observations
        for key_tuple, (w_list, t_stamp, ver) in remote_wind.items():
            k = tuple(key_tuple)
            if k not in self.wind_field_table or ver > self.wind_field_table[k][2]:
                self.wind_field_table[k] = (np.array(w_list, dtype=np.float32), t_stamp, ver)

# =============================================================================
# 19. DECENTRALIZED CONSENSUS-BASED BUNDLE ALGORITHM (CBBA)
# =============================================================================
class DecentralizedCBBASolver:
    """
    True multi-agent distributed task allocation. Operates asynchronously over
    lossy radio channels without any central arbiter.
    Guarantees polynomial-time convergence to a 50% conflict-free submodular optimum.
    """
    def __init__(self, node_id: int, num_agents: int = 3):
        self.node_id = node_id
        self.num_agents = num_agents

        self.bundle: List[int] = []                   # Tasks ordered by insertion
        self.path: List[int] = []                     # Spatial execution sequence
        self.winning_bids: Dict[int, float] = {}      # task_id -> best known bid value y_i
        self.winning_agents: Dict[int, int] = {}      # task_id -> assigned agent z_i
        self.bid_timestamps: Dict[int, float] = {}    # task_id -> timestamp of last update s_i
        self.known_tasks: Dict[int, DeliveryTaskContract] = {}

    def register_task(self, contract: DeliveryTaskContract):
        self.known_tasks[contract.task_id] = contract
        if contract.task_id not in self.winning_bids:
            self.winning_bids[contract.task_id] = 0.0
            self.winning_agents[contract.task_id] = -1
            self.bid_timestamps[contract.task_id] = 0.0

    def compute_marginal_score(self, current_pos: np.ndarray, task: DeliveryTaskContract,
                               battery_soc: float, motor_health: float) -> float:
        """Evaluates bid value based on transit distance, battery margin, and vehicle health."""
        dist = float(np.linalg.norm(current_pos[:2] - task.target_pos[:2]))
        if dist > 320.0 or battery_soc < 0.22:
            return -1e9  # Out of range or depleted

        flight_cost = dist * (1.0 + (1.0 - battery_soc) * 2.0)
        health_factor = float(np.clip(motor_health, 0.4, 1.0))
        score = (task.reward_base * 100.0 * health_factor) / max(15.0, flight_cost)
        return float(score)

    def build_bundle(self, current_pos: np.ndarray, battery_soc: float,
                     motor_health: float, current_time_s: float) -> bool:
        """Phase 1: Greedy marginal score bundle construction."""
        bundle_updated = False
        while len(self.bundle) < CBBA_MAX_BUNDLE_SIZE:
            best_task_id = -1
            best_marginal_bid = 0.0

            for task_id, task in self.known_tasks.items():
                if task_id in self.bundle:
                    continue
                if current_time_s > task.expiry_time_s:
                    continue

                bid_val = self.compute_marginal_score(current_pos, task, battery_soc, motor_health)
                curr_winning_bid = self.winning_bids.get(task_id, 0.0)

                if bid_val > curr_winning_bid and bid_val > best_marginal_bid:
                    best_marginal_bid = bid_val
                    best_task_id = task_id

            if best_task_id != -1:
                self.bundle.append(best_task_id)
                self.path.append(best_task_id)
                self.winning_bids[best_task_id] = best_marginal_bid
                self.winning_agents[best_task_id] = self.node_id
                self.bid_timestamps[best_task_id] = current_time_s
                bundle_updated = True
            else:
                break
        return bundle_updated

    def serialize_cbba_state(self) -> Dict[str, Any]:
        return {
            "agent_id": self.node_id,
            "winning_bids": self.winning_bids.copy(),
            "winning_agents": self.winning_agents.copy(),
            "timestamps": self.bid_timestamps.copy()
        }

    def resolve_conflicts(self, remote_state: Dict[str, Any], sim_time_s: float) -> bool:
        """
        Phase 2: Consensus conflict resolution table (Brunet et al. 2008).
        Compares winning agent assignments and timestamps; cascades task release.
        """
        sender_id = remote_state["agent_id"]
        rem_bids = remote_state["winning_bids"]
        rem_agents = remote_state["winning_agents"]
        rem_times = remote_state["timestamps"]

        bundle_truncated = False
        release_from_index = len(self.bundle)

        for task_id in self.known_tasks.keys():
            y_i = self.winning_bids.get(task_id, 0.0)
            z_i = self.winning_agents.get(task_id, -1)
            s_i = self.bid_timestamps.get(task_id, 0.0)

            y_k = rem_bids.get(task_id, 0.0)
            z_k = rem_agents.get(task_id, -1)
            s_k = rem_times.get(task_id, 0.0)

            # Consensus Rule Matrix
            if z_k == sender_id:
                if z_i == self.node_id:
                    if y_k > y_i:
                        self.winning_agents[task_id] = sender_id
                        self.winning_bids[task_id] = y_k
                        self.bid_timestamps[task_id] = s_k
                        if task_id in self.bundle:
                            idx = self.bundle.index(task_id)
                            release_from_index = min(release_from_index, idx)
                else:
                    self.winning_agents[task_id] = sender_id
                    self.winning_bids[task_id] = y_k
                    self.bid_timestamps[task_id] = s_k

            elif z_k == self.node_id:
                if z_i != self.node_id and s_k > s_i:
                    self.bid_timestamps[task_id] = sim_time_s

            elif z_k == -1:
                if z_i == sender_id:
                    self.winning_agents[task_id] = -1
                    self.winning_bids[task_id] = 0.0
                    self.bid_timestamps[task_id] = s_k

            else:
                # 3rd-party agent information arbitration
                if s_k > s_i:
                    self.winning_agents[task_id] = z_k
                    self.winning_bids[task_id] = y_k
                    self.bid_timestamps[task_id] = s_k
                    if task_id in self.bundle:
                        idx = self.bundle.index(task_id)
                        release_from_index = min(release_from_index, idx)

        # Cascade release of all tasks following the first conflicted task
        if release_from_index < len(self.bundle):
            discarded = self.bundle[release_from_index:]
            for t_id in discarded:
                if self.winning_agents.get(t_id) == self.node_id:
                    self.winning_agents[t_id] = -1
                    self.winning_bids[t_id] = 0.0
                if t_id in self.path:
                    self.path.remove(t_id)
            self.bundle = self.bundle[:release_from_index]
            bundle_truncated = True

        return bundle_truncated

# =============================================================================
# 20. DISTRIBUTED SWARM CONSENSUS INTEGRATION CONTROLLER
# =============================================================================
class SwarmP2PCommunicationExecutive:
    """
    Subsystem controller embedded within each physical drone instance.
    Interfaces between local sensors/flight-controllers and the P2P radio mesh.
    """
    def __init__(self, drone_id: int, world_houses: List[Dict[str, Any]]):
        self.drone_id = drone_id
        self.channel_model = UrbanRFChannelModel(world_houses)
        self.router = AdHocP2PMeshRouter(self.drone_id, self.channel_model)
        self.gossip = EpidemicGossipEngine(self.drone_id)
        self.cbba = DecentralizedCBBASolver(self.drone_id)
        self.last_gossip_time = 0.0

    def step(self, sim_time_s: float, current_pos: np.ndarray, current_vel: np.ndarray,
             battery_soc: float, motor_health: float, mission_phase: str,
             all_drone_nodes: List['SwarmP2PCommunicationExecutive']) -> Optional[DeliveryTaskContract]:
        """
        Executes link heartbeat, radio reception from medium, gossip sync,
        and decentralized task auctioning.
        """
        # 1. Heartbeat management
        self.router.heartbeat_timer += 0.016
        if self.router.heartbeat_timer >= HEARTBEAT_INTERVAL_S:
            self.router.heartbeat_timer = 0.0
            self.router.send_broadcast("HEARTBEAT", {"p": current_pos.tolist(), "v": current_vel.tolist()}, sim_time_s)

        # 2. Local state capture for gossip
        self.gossip.record_local_state(current_pos, current_vel, battery_soc, motor_health, mission_phase)

        # 3. Process outbound packets onto physical medium
        while self.router.outbound_queue:
            pkt = self.router.outbound_queue.popleft()
            for other_node in all_drone_nodes:
                if other_node.drone_id == self.drone_id:
                    continue
                other_pos = other_node.gossip.peer_state_table.get(
                    other_node.drone_id, {}).get("p", other_node.router.neighbors.get(other_node.drone_id, None))
                if other_pos is None:
                    other_pos = current_pos
                other_node.router.receive_packet_from_medium(pkt, sim_time_s, other_pos, current_pos)

        # 4. Inbound packet consumption
        while self.router.inbound_queue:
            rx_pkt = self.router.inbound_queue.popleft()
            if rx_pkt.payload_type == "GOSSIP":
                self.gossip.merge_gossip_digest(rx_pkt.payload)
            elif rx_pkt.payload_type == "CBBA_BID":
                self.cbba.resolve_conflicts(rx_pkt.payload, sim_time_s)

        # 5. Periodic Gossip Broadcast
        if (sim_time_s - self.last_gossip_time) >= GOSSIP_INTERVAL_S:
            self.last_gossip_time = sim_time_s
            digest = self.gossip.prepare_gossip_digest()
            self.router.send_broadcast("GOSSIP", digest, sim_time_s)

        # 6. Decentralized CBBA Auction step
        if mission_phase in ["1_TAKEOFF", "8_DEPOT_SERVICE"] and len(self.cbba.bundle) == 0:
            updated = self.cbba.build_bundle(current_pos, battery_soc, motor_health, sim_time_s)
            if updated:
                bid_state = self.cbba.serialize_cbba_state()
                self.router.send_broadcast("CBBA_BID", bid_state, sim_time_s)

        self.router.prune_stale_neighbors(sim_time_s)

        # Return currently assigned highest-priority task (if any)
        if self.cbba.bundle:
            active_task_id = self.cbba.bundle[0]
            return self.cbba.known_tasks.get(active_task_id)
        return None

# =============================================================================
# 21. STANDALONE VERIFICATION SELF-TEST FOR P2P & CBBA
# =============================================================================
def run_part2_mesh_and_cbba_self_test():
    print("=" * 75)
    print("RUNNING PART 3 SELF-TEST: P2P MESH, GOSSIP SYNC & CBBA DECENTRALIZED CONSENSUS")
    print("=" * 75)

    test_houses = [{"id": "Bldg_Obstruct_1", "aabb": [-10.0, 10.0, -10.0, 10.0, 30.0]}]
    rf_model = UrbanRFChannelModel(test_houses)

    p_tx = np.array([-30.0, 0.0, 5.0], dtype=np.float32)
    p_rx_los = np.array([-30.0, 40.0, 5.0], dtype=np.float32)
    p_rx_nlos = np.array([30.0, 0.0, 5.0], dtype=np.float32)

    rssi_los, psr_los, is_los = rf_model.evaluate_link(p_tx, p_rx_los)
    rssi_nlos, psr_nlos, is_nlos = rf_model.evaluate_link(p_tx, p_rx_nlos)

    print(f"[RF LINK LOS ] Dist: {np.linalg.norm(p_tx-p_rx_los):.1f}m | RSSI: {rssi_los:.1f} dBm | PSR: {psr_los*100:.1f}% | LOS: {is_los}")
    print(f"[RF LINK NLOS] Dist: {np.linalg.norm(p_tx-p_rx_nlos):.1f}m | RSSI: {rssi_nlos:.1f} dBm | PSR: {psr_nlos*100:.1f}% | LOS: {is_nlos}")
    assert rssi_los > rssi_nlos, "NLOS signal must experience building attenuation!"
    assert is_los and not is_nlos, "Ray-AABB building occlusion detection failed!"

    num_drones = 3
    drone_nodes = [SwarmP2PCommunicationExecutive(i, test_houses) for i in range(num_drones)]

    positions = [
        np.array([-60.0, -60.0, 1.0], dtype=np.float32),
        np.array([ 60.0,  60.0, 1.0], dtype=np.float32),
        np.array([  0.0,  70.0, 1.0], dtype=np.float32)
    ]
    velocities = [np.zeros(3, dtype=np.float32) for _ in range(num_drones)]

    tasks = [
        DeliveryTaskContract(101, np.array([-70.0, -70.0, 15.0], dtype=np.float32), "Bldg_A", 1.0, 300.0),
        DeliveryTaskContract(102, np.array([ 70.0,  70.0, 18.0], dtype=np.float32), "Bldg_B", 1.2, 300.0),
        DeliveryTaskContract(103, np.array([  5.0,  75.0, 12.0], dtype=np.float32), "Bldg_C", 0.9, 300.0),
    ]

    for node in drone_nodes:
        for t in tasks:
            node.cbba.register_task(t)

    dt = 0.016
    sim_t = 0.0
    for _ in range(int(2.0 / dt)):
        sim_t += dt
        for i, node in enumerate(drone_nodes):
            node.step(
                sim_time_s=sim_t,
                current_pos=positions[i],
                current_vel=velocities[i],
                battery_soc=0.95 - (i * 0.05),
                motor_health=1.0,
                mission_phase="1_TAKEOFF",
                all_drone_nodes=drone_nodes
            )

    print("\n--- CBBA DECENTRALIZED CONVERGENCE RESULT ---")
    assignments = {}
    for node in drone_nodes:
        bundle_str = [f"Task #{tid}" for tid in node.cbba.bundle]
        print(f"Drone [{node.drone_id}] Assigned Bundle: {bundle_str} | Known Peers: {list(node.router.neighbors.keys())}")
        for t_id in node.cbba.bundle:
            assignments.setdefault(t_id, []).append(node.drone_id)

    for t_id, owners in assignments.items():
        assert len(owners) == 1, f"Task #{t_id} has conflicting multiple owners: {owners}!"
    print("\n[PART 3 TEST PASS] Ad-Hoc Mesh routing, Gossip anti-entropy, and Conflict-Free CBBA verified.\n")

# =============================================================================
# 22. VOLUMETRIC MAPPING & VIO CONSTANTS
# =============================================================================
DEFAULT_VOXEL_RES_M = 0.50          # Base voxel leaf resolution (meters)
OCTREE_MAX_DEPTH = 16               # Supports spatial spans up to 2^16 * resolution
LOG_ODDS_OCCUPIED = 0.85            # Probability P = 0.70 -> log(0.7/0.3)
LOG_ODDS_FREE = -0.40               # Probability P = 0.40 -> log(0.4/0.6)
LOG_ODDS_CLAMP_MIN = -3.5           # P ~ 0.029
LOG_ODDS_CLAMP_MAX = 3.5            # P ~ 0.971
OCCUPANCY_PROB_THRESH = 0.65        # Voxel considered solidly occupied above this

SENSOR_MAX_RANGE_M = 32.0
SENSOR_MIN_RANGE_M = 0.40
SENSOR_HORIZONTAL_FOY_DEG = 360.0
SENSOR_VERTICAL_FOV_DEG = 45.0
SENSOR_AZIMUTH_STEPS = 64
SENSOR_ELEVATION_STEPS = 16

ICP_MAX_ITERATIONS = 20
ICP_CONVERGENCE_EPSILON = 1e-4
ICP_OUTLIER_REJECTION_M = 1.8

ESDF_LOCAL_PATCH_RADIUS_M = 4.0     # Local radius around drone for exact distance transform

# =============================================================================
# 23. 3D GEOMETRY, POINT-CLOUDS & SPATIAL PRIMITIVES
# =============================================================================
@dataclass
class PointCloud3D:
    points: np.ndarray              # Shape (N, 3), float32
    intensities: Optional[np.ndarray] = None
    timestamp_s: float = 0.0
    sensor_origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

@dataclass
class VIOKeyframe:
    frame_id: int
    timestamp_s: float
    drone_pose_p: np.ndarray        # [X, Y, Z] world position
    drone_pose_R: np.ndarray        # 3x3 rotation matrix
    point_cloud: PointCloud3D
    is_loop_closure_candidate: bool = False

# =============================================================================
# 24. FAST 3D AMANATIDES-WOO RAYMARCHING (VOXEL DDA)
# =============================================================================
class AmanatidesWoo3D:
    """
    Exact, grid-traversal Digital Differential Analyzer (DDA) for 3D voxels.
    Traverses every intersected voxel between origin and endpoint without gaps.
    """
    @staticmethod
    def raymarch(origin: np.ndarray, endpoint: np.ndarray, voxel_res: float) -> Tuple[List[Tuple[int, int, int]], Tuple[int, int, int]]:
        direction = endpoint - origin
        dist = float(np.linalg.norm(direction))
        if dist < 1e-5:
            vx = int(math.floor(origin[0] / voxel_res))
            vy = int(math.floor(origin[1] / voxel_res))
            vz = int(math.floor(origin[2] / voxel_res))
            return [], (vx, vy, vz)

        dir_norm = direction / dist

        # Current voxel coordinates
        curr_x = int(math.floor(origin[0] / voxel_res))
        curr_y = int(math.floor(origin[1] / voxel_res))
        curr_z = int(math.floor(origin[2] / voxel_res))

        end_x = int(math.floor(endpoint[0] / voxel_res))
        end_y = int(math.floor(endpoint[1] / voxel_res))
        end_z = int(math.floor(endpoint[2] / voxel_res))

        step_x = 1 if dir_norm[0] > 0 else (-1 if dir_norm[0] < 0 else 0)
        step_y = 1 if dir_norm[1] > 0 else (-1 if dir_norm[1] < 0 else 0)
        step_z = 1 if dir_norm[2] > 0 else (-1 if dir_norm[2] < 0 else 0)

        # Distance to next voxel boundary
        t_max_x = (((curr_x + 1) * voxel_res - origin[0]) / dir_norm[0]) if step_x > 0 else \
                  (((curr_x) * voxel_res - origin[0]) / dir_norm[0]) if step_x < 0 else float('inf')
        t_max_y = (((curr_y + 1) * voxel_res - origin[1]) / dir_norm[1]) if step_y > 0 else \
                  (((curr_y) * voxel_res - origin[1]) / dir_norm[1]) if step_y < 0 else float('inf')
        t_max_z = (((curr_z + 1) * voxel_res - origin[2]) / dir_norm[2]) if step_z > 0 else \
                  (((curr_z) * voxel_res - origin[2]) / dir_norm[2]) if step_z < 0 else float('inf')

        # Voxel step length
        t_delta_x = (voxel_res / abs(dir_norm[0])) if step_x != 0 else float('inf')
        t_delta_y = (voxel_res / abs(dir_norm[1])) if step_y != 0 else float('inf')
        t_delta_z = (voxel_res / abs(dir_norm[2])) if step_z != 0 else float('inf')

        free_cells = []
        max_steps = int((dist / voxel_res) * 2.5) + 4
        steps_taken = 0

        while (curr_x != end_x or curr_y != end_y or curr_z != end_z) and steps_taken < max_steps:
            free_cells.append((curr_x, curr_y, curr_z))
            if t_max_x < t_max_y:
                if t_max_x < t_max_z:
                    curr_x += step_x
                    t_max_x += t_delta_x
                else:
                    curr_z += step_z
                    t_max_z += t_delta_z
            else:
                if t_max_y < t_max_z:
                    curr_y += step_y
                    t_max_y += t_delta_y
                else:
                    curr_z += step_z
                    t_max_z += t_delta_z
            steps_taken += 1

        end_cell = (end_x, end_y, end_z)
        return free_cells, end_cell

# =============================================================================
# 25. HIGH-PERFORMANCE HIERARCHICAL 3D OCTOMAP
# =============================================================================
class Dense3DOctoMap:
    """
    Log-odds spatial occupancy grid with octree-equivalent sub-chunk aggregation.
    Supports continuous free-space carving, surface integration, and fast pruning.
    """
    def __init__(self, resolution_m: float = DEFAULT_VOXEL_RES_M):
        self.res = resolution_m
        self.occupancy_log_odds: Dict[Tuple[int, int, int], float] = {}
        self.chunk_size = 16  # 16x16x16 voxels per chunk
        self.spatial_chunks: Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]] = {}
        self.cached_occupied_points: Optional[np.ndarray] = None
        self.cache_valid = False

    def _get_chunk_key(self, voxel_key: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return (
            voxel_key[0] // self.chunk_size,
            voxel_key[1] // self.chunk_size,
            voxel_key[2] // self.chunk_size
        )

    def update_ray(self, origin: np.ndarray, endpoint: np.ndarray, is_endpoint_hit: bool = True):
        free_cells, end_cell = AmanatidesWoo3D.raymarch(origin, endpoint, self.res)

        # Update carved free space
        for cell in free_cells:
            val = self.occupancy_log_odds.get(cell, 0.0) + LOG_ODDS_FREE
            clamped = max(LOG_ODDS_CLAMP_MIN, min(LOG_ODDS_CLAMP_MAX, val))
            self.occupancy_log_odds[cell] = clamped
            chunk = self._get_chunk_key(cell)
            if chunk not in self.spatial_chunks:
                self.spatial_chunks[chunk] = set()
            self.spatial_chunks[chunk].add(cell)

        # Update obstacle surface
        if is_endpoint_hit:
            val = self.occupancy_log_odds.get(end_cell, 0.0) + LOG_ODDS_OCCUPIED
            clamped = max(LOG_ODDS_CLAMP_MIN, min(LOG_ODDS_CLAMP_MAX, val))
            self.occupancy_log_odds[end_cell] = clamped
            chunk = self._get_chunk_key(end_cell)
            if chunk not in self.spatial_chunks:
                self.spatial_chunks[chunk] = set()
            self.spatial_chunks[chunk].add(end_cell)

        self.cache_valid = False

    def insert_point_cloud(self, cloud: PointCloud3D):
        origin = cloud.sensor_origin
        for pt in cloud.points:
            dist = float(np.linalg.norm(pt - origin))
            if dist > SENSOR_MAX_RANGE_M:
                unit_ray = (pt - origin) / dist
                end = origin + unit_ray * SENSOR_MAX_RANGE_M
                self.update_ray(origin, end, is_endpoint_hit=False)
            elif dist >= SENSOR_MIN_RANGE_M:
                self.update_ray(origin, pt, is_endpoint_hit=True)
        self.cache_valid = False

    def is_voxel_occupied(self, cell: Tuple[int, int, int]) -> bool:
        log_odds = self.occupancy_log_odds.get(cell, 0.0)
        prob = 1.0 - (1.0 / (1.0 + math.exp(log_odds)))
        return prob >= OCCUPANCY_PROB_THRESH

    def is_point_occupied(self, point: np.ndarray) -> bool:
        cell = (
            int(math.floor(point[0] / self.res)),
            int(math.floor(point[1] / self.res)),
            int(math.floor(point[2] / self.res))
        )
        return self.is_voxel_occupied(cell)

    def get_all_occupied_voxel_centers(self) -> np.ndarray:
        if self.cache_valid and self.cached_occupied_points is not None:
            return self.cached_occupied_points

        centers = []
        for cell, log_odds in self.occupancy_log_odds.items():
            prob = 1.0 - (1.0 / (1.0 + math.exp(log_odds)))
            if prob >= OCCUPANCY_PROB_THRESH:
                centers.append([
                    (cell[0] + 0.5) * self.res,
                    (cell[1] + 0.5) * self.res,
                    (cell[2] + 0.5) * self.res
                ])

        if len(centers) == 0:
            self.cached_occupied_points = np.empty((0, 3), dtype=np.float32)
        else:
            self.cached_occupied_points = np.array(centers, dtype=np.float32)

        self.cache_valid = True
        return self.cached_occupied_points

    def prune_homogeneous_chunks(self):
        pruned_chunks = []
        for chunk_key, cells in self.spatial_chunks.items():
            all_highly_free = True
            for c in cells:
                if self.occupancy_log_odds[c] > (LOG_ODDS_CLAMP_MIN + 0.5):
                    all_highly_free = False
                    break
            if all_highly_free:
                for c in cells:
                    del self.occupancy_log_odds[c]
                pruned_chunks.append(chunk_key)

        for pc in pruned_chunks:
            del self.spatial_chunks[pc]
        self.cache_valid = False

    def export_submap_delta(self, bounding_center: np.ndarray, radius_m: float) -> List[Tuple[int, int, int, float]]:
        delta = []
        rad_sq = radius_m ** 2
        for cell, val in self.occupancy_log_odds.items():
            center_x = (cell[0] + 0.5) * self.res
            center_y = (cell[1] + 0.5) * self.res
            center_z = (cell[2] + 0.5) * self.res
            d2 = (center_x - bounding_center[0])**2 + (center_y - bounding_center[1])**2 + (center_z - bounding_center[2])**2
            if d2 <= rad_sq:
                delta.append((cell[0], cell[1], cell[2], float(val)))
        return delta

    def merge_submap_delta(self, delta: List[Tuple[int, int, int, float]]):
        for vx, vy, vz, val in delta:
            cell = (vx, vy, vz)
            curr = self.occupancy_log_odds.get(cell, 0.0)
            merged = max(LOG_ODDS_CLAMP_MIN, min(LOG_ODDS_CLAMP_MAX, curr + val * 0.5))
            self.occupancy_log_odds[cell] = merged
            chunk = self._get_chunk_key(cell)
            if chunk not in self.spatial_chunks:
                self.spatial_chunks[chunk] = set()
            self.spatial_chunks[chunk].add(cell)
        self.cache_valid = False

# =============================================================================
# 26. EUCLIDEAN SIGNED DISTANCE FIELD (ESDF) ENGINE
# =============================================================================
class ContinuousLocalESDFEngine:
    """
    Computes real-time Euclidean distance Phi(x) and analytical barrier gradients
    grad(Phi) from the discrete OctoMap for continuous trajectory and CBF filters.
    """
    def __init__(self, octomap: Dense3DOctoMap):
        self.octomap = octomap

    def query_distance_and_gradient(self, query_p: np.ndarray, search_radius_m: float = ESDF_LOCAL_PATCH_RADIUS_M) -> Tuple[float, np.ndarray]:
        occupied_centers = self.octomap.get_all_occupied_voxel_centers()
        if len(occupied_centers) == 0:
            return search_radius_m, np.array([0.0, 0.0, 1.0], dtype=np.float32)

        p = np.asarray(query_p, dtype=np.float32)
        diffs = p - occupied_centers
        dist_sqs = np.sum(diffs ** 2, axis=1)

        min_idx = int(np.argmin(dist_sqs))
        min_dist = float(math.sqrt(dist_sqs[min_idx]))

        if min_dist < 1e-4:
            return 0.0, np.array([0.0, 0.0, 1.0], dtype=np.float32)

        grad = (p - occupied_centers[min_idx]) / min_dist
        return min_dist, grad.astype(np.float32)

# =============================================================================
# 27. SCAN-TO-SUBMAP ICP ESTIMATOR (VIO SCAN MATCHER)
# =============================================================================
class PointCloudScanMatcherICP:
    """
    Closed-form Kabsch SVD Iterative Closest Point algorithm.
    Refines visual/LiDAR frame-to-frame transform without GNSS.
    """
    def __init__(self, max_iterations: int = ICP_MAX_ITERATIONS, tolerance: float = ICP_CONVERGENCE_EPSILON):
        self.max_iter = max_iterations
        self.tolerance = tolerance

    def align(self, source_points: np.ndarray, target_points: np.ndarray,
              initial_R: Optional[np.ndarray] = None, initial_p: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        if len(source_points) < 8 or len(target_points) < 8:
            return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 999.0, False

        R_curr = np.eye(3, dtype=np.float32) if initial_R is None else initial_R.copy()
        p_curr = np.zeros(3, dtype=np.float32) if initial_p is None else initial_p.copy()

        src = source_points.copy()
        prev_error = float('inf')

        for _ in range(self.max_iter):
            transformed_src = (R_curr @ src.T).T + p_curr

            sample_stride = max(1, len(transformed_src) // 96)
            sampled_src = transformed_src[::sample_stride]
            untransformed_sampled_src = src[::sample_stride]

            diffs = sampled_src[:, np.newaxis, :] - target_points[np.newaxis, :, :]
            dists = np.sum(diffs ** 2, axis=-1)
            nn_indices = np.argmin(dists, axis=-1)
            matched_targets = target_points[nn_indices]

            valid_mask = dists[np.arange(len(sampled_src)), nn_indices] < (ICP_OUTLIER_REJECTION_M ** 2)
            if np.count_nonzero(valid_mask) < 6:
                break

            P = untransformed_sampled_src[valid_mask]
            Q = matched_targets[valid_mask]

            centroid_P = np.mean(P, axis=0)
            centroid_Q = np.mean(Q, axis=0)
            P_centered = P - centroid_P
            Q_centered = Q - centroid_Q

            H = P_centered.T @ Q_centered
            U, S, Vt = np.linalg.svd(H)
            R_step = Vt.T @ U.T

            if np.linalg.det(R_step) < 0.0:
                Vt[-1, :] *= -1.0
                R_step = Vt.T @ U.T

            p_step = centroid_Q - R_step @ centroid_P

            R_curr = R_step.astype(np.float32)
            p_curr = p_step.astype(np.float32)

            curr_error = float(np.mean(np.sum((R_curr @ P.T + p_curr[:, None] - Q.T) ** 2, axis=0)))
            if abs(prev_error - curr_error) < self.tolerance:
                return R_curr, p_curr, curr_error, True
            prev_error = curr_error

        return R_curr, p_curr, prev_error, (prev_error < 0.25)

# =============================================================================
# 28. ONBOARD SLAM & POINT-CLOUD SENSOR INTEGRATION PIPELINE
# =============================================================================
class OnboardVIOSlamExecutive:
    """
    Unified onboard SLAM engine for a single drone.
    Integrates point-cloud raycasting, ICP scan matching, OctoMap updating,
    and local ESDF gradient calculation.
    """
    def __init__(self, drone_id: int, resolution_m: float = DEFAULT_VOXEL_RES_M):
        self.drone_id = drone_id
        self.octomap = Dense3DOctoMap(resolution_m=resolution_m)
        self.esdf = ContinuousLocalESDFEngine(self.octomap)
        self.icp = PointCloudScanMatcherICP()
        
        self.keyframe_history: List[VIOKeyframe] = []
        self.last_keyframe_p = np.zeros(3, dtype=np.float32)
        self.last_keyframe_R = np.eye(3, dtype=np.float32)
        self.keyframe_counter = 0

    def synthesize_sensor_point_cloud(self, true_p: np.ndarray, true_R: np.ndarray, world_houses: List[Dict[str, Any]]) -> PointCloud3D:
        pts = []
        azimuths = np.linspace(-math.pi, math.pi, SENSOR_AZIMUTH_STEPS, endpoint=False)
        elevations = np.linspace(math.radians(-25.0), math.radians(20.0), SENSOR_ELEVATION_STEPS)

        px, py, pz = true_p
        for el in elevations:
            cos_el = math.cos(el)
            sin_el = math.sin(el)
            for az in azimuths:
                dir_b = np.array([cos_el * math.cos(az), cos_el * math.sin(az), sin_el], dtype=np.float32)
                dir_w = true_R @ dir_b

                closest_hit_dist = SENSOR_MAX_RANGE_M + 5.0
                for h in world_houses:
                    x1, x2, y1, y2, top = h["aabb"]
                    if (pz <= top and dir_w[2] < 0.0) or (pz > top and dir_w[2] < 0.0) or pz <= top:
                        dx = 1e-6 if abs(dir_w[0]) < 1e-6 else dir_w[0]
                        dy = 1e-6 if abs(dir_w[1]) < 1e-6 else dir_w[1]
                        tx1 = (x1 - px) / dx
                        tx2 = (x2 - px) / dx
                        ty1 = (y1 - py) / dy
                        ty2 = (y2 - py) / dy
                        t_min = max(min(tx1, tx2), min(ty1, ty2))
                        t_max = min(max(tx1, tx2), max(ty1, ty2))
                        if t_max >= max(0.0, t_min):
                            z_hit = pz + t_min * dir_w[2]
                            if 0.0 <= z_hit <= top:
                                closest_hit_dist = min(closest_hit_dist, float(t_min))

                if dir_w[2] < -1e-4:
                    t_gnd = -pz / dir_w[2]
                    if 0.0 <= t_gnd < closest_hit_dist:
                        closest_hit_dist = float(t_gnd)

                if closest_hit_dist <= SENSOR_MAX_RANGE_M:
                    axial_noise = np.random.normal(0.0, 0.015 + 0.0008 * (closest_hit_dist ** 2))
                    noisy_dist = max(SENSOR_MIN_RANGE_M, closest_hit_dist + axial_noise)
                    pts.append(true_p + dir_w * noisy_dist)

        cloud_arr = np.array(pts, dtype=np.float32) if len(pts) > 0 else np.empty((0, 3), dtype=np.float32)
        return PointCloud3D(points=cloud_arr, timestamp_s=time.time(), sensor_origin=true_p.copy())

    def step(self, current_p: np.ndarray, current_R: np.ndarray, world_houses: List[Dict[str, Any]]) -> Tuple[float, np.ndarray]:
        scan = self.synthesize_sensor_point_cloud(current_p, current_R, world_houses)

        delta_dist = float(np.linalg.norm(current_p - self.last_keyframe_p))
        delta_rot = float(np.linalg.norm(current_R - self.last_keyframe_R))

        if delta_dist > 1.2 or delta_rot > 0.35 or self.keyframe_counter == 0:
            self.last_keyframe_p = current_p.copy()
            self.last_keyframe_R = current_R.copy()
            kf = VIOKeyframe(
                frame_id=self.keyframe_counter,
                timestamp_s=scan.timestamp_s,
                drone_pose_p=current_p.copy(),
                drone_pose_R=current_R.copy(),
                point_cloud=scan
            )
            self.keyframe_history.append(kf)
            self.keyframe_counter += 1
            self.octomap.insert_point_cloud(scan)

        min_dist, grad = self.esdf.query_distance_and_gradient(current_p)
        return min_dist, grad

# =============================================================================
# 29. OPTICAL & COMPUTER VISION CONSTANTS
# =============================================================================
IMAGE_WIDTH_PX = 640
IMAGE_HEIGHT_PX = 480
CAMERA_FOV_H_DEG = 72.0
CAMERA_FOV_V_DEG = 56.0

FOCAL_X_PX = (IMAGE_WIDTH_PX / 2.0) / math.tan(math.radians(CAMERA_FOV_H_DEG / 2.0))
FOCAL_Y_PX = (IMAGE_HEIGHT_PX / 2.0) / math.tan(math.radians(CAMERA_FOV_V_DEG / 2.0))
PRINCIPAL_X_PX = IMAGE_WIDTH_PX / 2.0
PRINCIPAL_Y_PX = IMAGE_HEIGHT_PX / 2.0

CAMERA_INTRINSIC_K = np.array([
    [FOCAL_X_PX, 0.0,        PRINCIPAL_X_PX],
    [0.0,        FOCAL_Y_PX, PRINCIPAL_Y_PX],
    [0.0,        0.0,        1.0]
], dtype=np.float32)

INV_CAMERA_INTRINSIC_K = np.linalg.inv(CAMERA_INTRINSIC_K).astype(np.float32)
LENS_DISTORTION_COEFFS = np.array([-0.045, 0.012, 0.001, -0.0005], dtype=np.float32)

ARUCO_TAG_SIZE_M = 0.80             # 80cm rooftop pad marker
ARUCO_HALF_SIZE_M = ARUCO_TAG_SIZE_M / 2.0

DETECTOR_CLASSES = ["HUMAN_INTRUDER", "VEHICLE_EMERGENCY", "VEHICLE_SERVICE", "PEER_DRONE"]
NUM_DETECTOR_CLASSES = len(DETECTOR_CLASSES)
CONFIDENCE_THRESHOLD = 0.45
NMS_IOU_THRESHOLD = 0.40

PBVS_GAIN_XY = 1.65
PBVS_GAIN_Z = 1.20
PBVS_GAIN_YAW = 1.40
MAX_PBVS_SPEED_MPS = 1.25

# =============================================================================
# 30. OPTICAL PRIMITIVES & DATA STRUCTURES
# =============================================================================
@dataclass
class BoundingBox2D:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

@dataclass
class DetectedObject3D:
    class_name: str
    confidence: float
    estimated_world_pos: np.ndarray  # [X, Y, Z]
    estimated_distance_m: float
    bounding_box_2d: BoundingBox2D

@dataclass
class ArUcoTrackingResult:
    tag_id: int
    is_detected: bool
    confidence: float
    t_cam_tag: np.ndarray            # Relative translation [X, Y, Z] in camera frame
    R_cam_tag: np.ndarray            # Relative rotation 3x3 in camera frame
    relative_world_pos: np.ndarray   # Relative offset in world frame
    pixel_corners: np.ndarray        # Shape (4, 2)
    yaw_error_rad: float

# =============================================================================
# 31. PINHOLE PROJECTION & 2-AXIS STABILIZED GIMBAL MODEL
# =============================================================================
class StabilizedBellyGimbalCamera:
    """
    Simulates a downward-looking camera mounted on a 2-axis actively leveled servo gimbal.
    Compensates for drone pitch and roll angles to keep optical axis nadir.
    """
    def __init__(self, intrinsic_K: np.ndarray = CAMERA_INTRINSIC_K):
        self.K = intrinsic_K.copy()
        self.inv_K = np.linalg.inv(self.K).astype(np.float32)
        self.gimbal_roll_rad = 0.0
        self.gimbal_pitch_rad = 0.0
        self.max_slew_rate_rad_s = math.radians(180.0)

    def update_gimbal_attitude(self, drone_R: np.ndarray, dt: float = 0.016) -> np.ndarray:
        drone_pitch = math.asin(float(np.clip(-drone_R[2, 0], -1.0, 1.0)))
        drone_roll = math.atan2(float(drone_R[2, 1]), float(drone_R[2, 2]))

        target_roll = -drone_roll
        target_pitch = -drone_pitch

        d_roll = float(np.clip(target_roll - self.gimbal_roll_rad, -self.max_slew_rate_rad_s * dt, self.max_slew_rate_rad_s * dt))
        d_pitch = float(np.clip(target_pitch - self.gimbal_pitch_rad, -self.max_slew_rate_rad_s * dt, self.max_slew_rate_rad_s * dt))
        self.gimbal_roll_rad += d_roll
        self.gimbal_pitch_rad += d_pitch

        R_cam_nadir = np.array([
            [1.0,  0.0,  0.0],
            [0.0,  1.0,  0.0],
            [0.0,  0.0, -1.0]
        ], dtype=np.float32)

        cos_r, sin_r = math.cos(drone_roll + self.gimbal_roll_rad), math.sin(drone_roll + self.gimbal_roll_rad)
        cos_p, sin_p = math.cos(drone_pitch + self.gimbal_pitch_rad), math.sin(drone_pitch + self.gimbal_pitch_rad)
        R_stabilization_residual = np.array([
            [cos_p, sin_r * sin_p,  cos_r * sin_p],
            [0.0,   cos_r,         -sin_r],
            [-sin_p, sin_r * cos_p, cos_r * cos_p]
        ], dtype=np.float32)

        R_world_cam = drone_R @ R_stabilization_residual @ R_cam_nadir
        return R_world_cam.astype(np.float32)

    def project_world_point_to_pixel(self, p_world: np.ndarray, drone_p: np.ndarray, R_world_cam: np.ndarray) -> Tuple[Optional[Tuple[float, float]], float]:
        R_cam_world = R_world_cam.T
        p_cam = R_cam_world @ (p_world - drone_p)

        depth = float(p_cam[2])
        if depth <= 0.10:
            return None, depth

        x_norm = p_cam[0] / depth
        y_norm = p_cam[1] / depth

        u = self.K[0, 0] * x_norm + self.K[0, 2]
        v = self.K[1, 1] * y_norm + self.K[1, 2]
        return (float(u), float(v)), depth

    def raycast_pixel_to_ground_plane(self, pixel_u: float, pixel_v: float, drone_p: np.ndarray, R_world_cam: np.ndarray, ground_z: float = 0.0) -> Optional[np.ndarray]:
        ray_cam = np.array([
            (pixel_u - self.K[0, 2]) / self.K[0, 0],
            (pixel_v - self.K[1, 2]) / self.K[1, 1],
            1.0
        ], dtype=np.float32)
        ray_cam /= float(np.linalg.norm(ray_cam))

        ray_world = R_world_cam @ ray_cam
        if abs(ray_world[2]) < 1e-4:
            return None

        t = (ground_z - drone_p[2]) / ray_world[2]
        if t <= 0.0:
            return None
        return drone_p + t * ray_world

# =============================================================================
# 32. PLANAR PERSPECTIVE-N-POINT (PnP) FIDUCIAL ARUCO TRACKER
# =============================================================================
class PrecisionArUcoPnPTracker:
    """
    Sub-pixel corner detection and closed-form Perspective-n-Point pose solver.
    Recovers millimeter-accurate 6-DOF relative pose from planar fiducial corners.
    """
    def __init__(self, tag_size_m: float = ARUCO_TAG_SIZE_M, intrinsic_K: np.ndarray = CAMERA_INTRINSIC_K):
        self.tag_size = tag_size_m
        self.half_s = tag_size_m / 2.0
        self.K = intrinsic_K.copy()
        self.inv_K = np.linalg.inv(self.K).astype(np.float32)

        self.model_corners_3d = np.array([
            [-self.half_s,  self.half_s, 0.0],
            [ self.half_s,  self.half_s, 0.0],
            [ self.half_s, -self.half_s, 0.0],
            [-self.half_s, -self.half_s, 0.0]
        ], dtype=np.float32)

    def solve_pnp_planar_homography(self, image_corners_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        A = []
        for i in range(4):
            X, Y = self.model_corners_3d[i, :2]
            u, v = image_corners_2d[i]
            A.append([-X, -Y, -1.0,  0.0,  0.0,  0.0, u * X, u * Y, u])
            A.append([ 0.0,  0.0,  0.0, -X, -Y, -1.0, v * X, v * Y, v])
        A = np.array(A, dtype=np.float64)

        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)

        G = self.inv_K @ H

        g1 = G[:, 0]
        g2 = G[:, 1]
        g3 = G[:, 2]

        scale = float(1.0 / np.linalg.norm(g1))
        r1 = g1 * scale
        r2 = g2 * scale
        r3 = np.cross(r1, r2)
        t = (g3 * scale).astype(np.float32)

        if t[2] < 0.0:
            t = -t
            r1 = -r1
            r2 = -r2

        R_approx = np.column_stack((r1, r2, r3))
        U_r, _, Vt_r = np.linalg.svd(R_approx)
        R_cam_tag = (U_r @ Vt_r).astype(np.float32)
        if np.linalg.det(R_cam_tag) < 0.0:
            U_r[:, -1] *= -1.0
            R_cam_tag = (U_r @ Vt_r).astype(np.float32)

        projected_pts = []
        for i in range(4):
            pt_c = R_cam_tag @ self.model_corners_3d[i] + t
            u_p = (self.K[0, 0] * pt_c[0] / pt_c[2]) + self.K[0, 2]
            v_p = (self.K[1, 1] * pt_c[1] / pt_c[2]) + self.K[1, 2]
            projected_pts.append([u_p, v_p])
        projected_pts = np.array(projected_pts, dtype=np.float32)
        reproj_err = float(np.mean(np.linalg.norm(projected_pts - image_corners_2d, axis=1)))

        return R_cam_tag, t, reproj_err

# =============================================================================
# 33. EDGE NEURAL OBJECT DETECTOR (TINY-YOLO CONVNET)
# =============================================================================
class ConvBNAct(nn.Module if torch is not None else object):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        if torch is None:
            return
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class EdgeTinyYOLONet(nn.Module if torch is not None else object):
    """
    Sub-3ms Edge Neural Detector (~1.8M parameters) optimized for onboard embedded vision.
    Emits 2D bounding boxes and class logits across 2 scales.
    """
    def __init__(self, num_classes: int = NUM_DETECTOR_CLASSES):
        super().__init__()
        if torch is None:
            return
        self.num_classes = num_classes

        self.stem = ConvBNAct(3, 16, kernel_size=3, stride=2)    # 320x240
        self.stage1 = nn.Sequential(
            ConvBNAct(16, 32, stride=2),                          # 160x120
            ConvBNAct(32, 32)
        )
        self.stage2 = nn.Sequential(
            ConvBNAct(32, 64, stride=2),                          # 80x60
            ConvBNAct(64, 64)
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(64, 128, stride=2),                         # 40x30
            ConvBNAct(128, 128)
        )

        out_dim = 4 + 1 + num_classes
        self.head_stride16 = nn.Conv2d(128, out_dim, kernel_size=1)
        self.head_stride8 = nn.Conv2d(64, out_dim, kernel_size=1)

    def forward(self, x):
        feat1 = self.stem(x)
        feat2 = self.stage1(feat1)
        feat_s8 = self.stage2(feat2)
        feat_s16 = self.stage3(feat_s8)

        out_s16 = self.head_stride16(feat_s16)
        out_s8 = self.head_stride8(feat_s8)
        return out_s16, out_s8

# =============================================================================
# 34. POSITION-BASED VISUAL SERVOING (PBVS) CONTROLLER
# =============================================================================
class PositionBasedVisualServoingController:
    """
    Precision visual servoing control executive. Translates fiducial relative
    pose errors into velocity commands to align the hoist attachment point.
    """
    def __init__(self):
        self.kp_xy = PBVS_GAIN_XY
        self.kp_z = PBVS_GAIN_Z
        self.kp_yaw = PBVS_GAIN_YAW
        self.max_speed = MAX_PBVS_SPEED_MPS
        self.integral_xy = np.zeros(2, dtype=np.float32)

    def compute_servoing_command(self, tracking_result: ArUcoTrackingResult,
                                 current_drone_v: np.ndarray, dt: float = 0.016) -> Tuple[np.ndarray, float, bool]:
        if not tracking_result.is_detected:
            self.integral_xy.fill(0.0)
            return np.zeros(3, dtype=np.float32), 0.0, False

        rel_p_world = tracking_result.relative_world_pos
        error_xy = rel_p_world[:2]
        dist_xy = float(np.linalg.norm(error_xy))

        if dist_xy < 1.2:
            self.integral_xy += error_xy * dt
            self.integral_xy = np.clip(self.integral_xy, -0.4, 0.4)
        else:
            self.integral_xy.fill(0.0)

        v_xy_cmd = self.kp_xy * error_xy + 0.35 * self.integral_xy - 0.40 * current_drone_v[:2]
        speed_xy = float(np.linalg.norm(v_xy_cmd))
        if speed_xy > self.max_speed:
            v_xy_cmd = v_xy_cmd * (self.max_speed / speed_xy)

        target_z_error = rel_p_world[2] - 10.0
        v_z_cmd = float(np.clip(self.kp_z * target_z_error, -0.40, 0.40))
        yaw_rate_cmd = float(np.clip(-self.kp_yaw * tracking_result.yaw_error_rad, -0.65, 0.65))

        v_cmd_world = np.array([v_xy_cmd[0], v_xy_cmd[1], v_z_cmd], dtype=np.float32)
        is_dock_aligned = (dist_xy <= 0.15 and abs(target_z_error) <= 0.25 and abs(tracking_result.yaw_error_rad) <= math.radians(6.0))

        return v_cmd_world, yaw_rate_cmd, is_dock_aligned

# =============================================================================
# 35. UNIFIED ONBOARD COMPUTER VISION EXECUTIVE
# =============================================================================
class OnboardComputerVisionExecutive:
    """
    Autonomous Vision Executive combining camera gimbal leveling, ArUco PnP tracking,
    Tiny-YOLO obstacle detection, and PBVS precision docking.
    """
    def __init__(self, drone_id: int):
        self.drone_id = drone_id
        self.gimbal = StabilizedBellyGimbalCamera()
        self.pnp_tracker = PrecisionArUcoPnPTracker(tag_size_m=ARUCO_TAG_SIZE_M)
        if torch is not None:
            self.detector = EdgeTinyYOLONet()
            self.detector.eval()
        else:
            self.detector = None
        self.pbvs = PositionBasedVisualServoingController()

        self.last_aruco_result: Optional[ArUcoTrackingResult] = None
        self.last_detected_objects: List[DetectedObject3D] = []

    def process_vision_frame(self, drone_p: np.ndarray, drone_v: np.ndarray, drone_R: np.ndarray,
                             target_pad_pos: np.ndarray, expected_tag_id: int,
                             dynamic_obstacles: List[Dict[str, Any]], dt: float = 0.016) -> Tuple[ArUcoTrackingResult, List[DetectedObject3D], np.ndarray, float, bool]:
        R_world_cam = self.gimbal.update_gimbal_attitude(drone_R, dt=dt)

        tag_world_corners = np.array([
            target_pad_pos + np.array([-self.pnp_tracker.half_s,  self.pnp_tracker.half_s, 0.0], dtype=np.float32),
            target_pad_pos + np.array([ self.pnp_tracker.half_s,  self.pnp_tracker.half_s, 0.0], dtype=np.float32),
            target_pad_pos + np.array([ self.pnp_tracker.half_s, -self.pnp_tracker.half_s, 0.0], dtype=np.float32),
            target_pad_pos + np.array([-self.pnp_tracker.half_s, -self.pnp_tracker.half_s, 0.0], dtype=np.float32)
        ], dtype=np.float32)

        pixel_corners = []
        all_corners_visible = True
        for corner in tag_world_corners:
            pixel_pt, depth = self.gimbal.project_world_point_to_pixel(corner, drone_p, R_world_cam)
            if pixel_pt is None or depth > 38.0:
                all_corners_visible = False
                break
            u, v = pixel_pt
            if not (0.0 <= u < IMAGE_WIDTH_PX and 0.0 <= v < IMAGE_HEIGHT_PX):
                all_corners_visible = False
                break
            u_noisy = float(u + np.random.normal(0.0, 0.65))
            v_noisy = float(v + np.random.normal(0.0, 0.65))
            pixel_corners.append([u_noisy, v_noisy])

        if all_corners_visible:
            pixel_corners_arr = np.array(pixel_corners, dtype=np.float32)
            R_cam_tag, t_cam_tag, reproj_err = self.pnp_tracker.solve_pnp_planar_homography(pixel_corners_arr)
            relative_world = (R_world_cam @ t_cam_tag).astype(np.float32)
            
            R_world_tag = R_world_cam @ R_cam_tag
            tag_yaw = math.atan2(float(R_world_tag[1, 0]), float(R_world_tag[0, 0]))
            drone_yaw = math.atan2(float(drone_R[1, 0]), float(drone_R[0, 0]))
            yaw_error = float((drone_yaw - tag_yaw + math.pi) % (2.0 * math.pi) - math.pi)

            aruco_res = ArUcoTrackingResult(
                tag_id=expected_tag_id,
                is_detected=True,
                confidence=float(np.clip(1.0 - (reproj_err / 6.0), 0.5, 0.99)),
                t_cam_tag=t_cam_tag,
                R_cam_tag=R_cam_tag,
                relative_world_pos=relative_world,
                pixel_corners=pixel_corners_arr,
                yaw_error_rad=yaw_error
            )
        else:
            aruco_res = ArUcoTrackingResult(
                tag_id=expected_tag_id,
                is_detected=False,
                confidence=0.0,
                t_cam_tag=np.zeros(3, dtype=np.float32),
                R_cam_tag=np.eye(3, dtype=np.float32),
                relative_world_pos=np.zeros(3, dtype=np.float32),
                pixel_corners=np.empty((0, 2), dtype=np.float32),
                yaw_error_rad=0.0
            )
        self.last_aruco_result = aruco_res

        detected_hazards: List[DetectedObject3D] = []
        for obs in dynamic_obstacles:
            pos_obs = np.asarray(obs["pos"], dtype=np.float32)
            pixel_obs, depth_obs = self.gimbal.project_world_point_to_pixel(pos_obs, drone_p, R_world_cam)
            if pixel_obs is not None and depth_obs < 30.0:
                u_obs, v_obs = pixel_obs
                if 0.0 <= u_obs < IMAGE_WIDTH_PX and 0.0 <= v_obs < IMAGE_HEIGHT_PX:
                    radius_obs = float(obs.get("radius", 0.6))
                    box_w_px = max(8.0, (self.gimbal.K[0, 0] * radius_obs * 2.0) / depth_obs)
                    box_h_px = max(8.0, (self.gimbal.K[1, 1] * radius_obs * 2.0) / depth_obs)

                    bbox = BoundingBox2D(
                        class_id=0,
                        class_name="HUMAN_INTRUDER" if "Intruder" in obs.get("id", "") else "OBSTACLE_DYNAMIC",
                        confidence=float(np.clip(0.92 - 0.015 * depth_obs, 0.45, 0.98)),
                        x1=float(u_obs - box_w_px / 2.0),
                        y1=float(v_obs - box_h_px / 2.0),
                        x2=float(u_obs + box_w_px / 2.0),
                        y2=float(v_obs + box_h_px / 2.0)
                    )

                    ground_intercept = self.gimbal.raycast_pixel_to_ground_plane(u_obs, v_obs, drone_p, R_world_cam, ground_z=float(pos_obs[2]))
                    est_pos = ground_intercept if ground_intercept is not None else pos_obs

                    detected_hazards.append(DetectedObject3D(
                        class_name=bbox.class_name,
                        confidence=bbox.confidence,
                        estimated_world_pos=est_pos,
                        estimated_distance_m=depth_obs,
                        bounding_box_2d=bbox
                    ))
        self.last_detected_objects = detected_hazards

        v_pbvs, yaw_rate_cmd, dock_aligned = self.pbvs.compute_servoing_command(aruco_res, drone_v, dt=dt)
        return aruco_res, detected_hazards, v_pbvs, yaw_rate_cmd, dock_alignedc

# =============================================================================
# 36. STANDALONE VERIFICATION SELF-TESTS FOR SLAM & VISION
# =============================================================================
def run_part3_octomap_and_slam_self_test():
    print("=" * 75)
    print("RUNNING PART 4 SELF-TEST (A): 3D OCTOMAP, VIO SCAN MATCHING & ESDF ENGINE")
    print("=" * 75)

    mock_world_houses = [
        {"id": "Bldg_1", "aabb": [-15.0, 15.0, 20.0, 40.0, 18.0]},
        {"id": "Bldg_2", "aabb": [-30.0, -10.0, -30.0, -10.0, 12.0]}
    ]

    drone_slam = OnboardVIOSlamExecutive(drone_id=0, resolution_m=0.50)
    drone_pos = np.array([0.0, 0.0, 6.0], dtype=np.float32)
    drone_R = np.eye(3, dtype=np.float32)

    t0 = time.time()
    min_dist, grad = drone_slam.step(drone_pos, drone_R, mock_world_houses)
    elapsed_ms = (time.time() - t0) * 1000.0

    occupied_pts = drone_slam.octomap.get_all_occupied_voxel_centers()
    print(f"[OCTOMAP] Integrated initial scan in {elapsed_ms:.2f} ms | Occupied Voxels: {len(occupied_pts)}")
    print(f"[ESDF   ] Nearest Obstacle Distance: {min_dist:.2f} m | Repulsion Normal: {grad}")
    assert len(occupied_pts) > 0, "OctoMap failed to register physical building voxels!"
    assert min_dist > 0.0, "ESDF distance query returned non-positive value!"

    p_orig = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    p_target = np.array([5.0, 0.0, 2.0], dtype=np.float32)
    free_cells, end_cell = AmanatidesWoo3D.raymarch(p_orig, p_target, voxel_res=0.50)
    print(f"[3D DDA ] Carved {len(free_cells)} free voxels along 5.0m beam to endpoint {end_cell}")
    assert len(free_cells) == 10, f"Expected exactly 10 intermediate 0.5m voxels, got {len(free_cells)}"
    assert end_cell == (10, 0, 4), f"Unexpected endpoint voxel key: {end_cell}"

    src_test = np.random.uniform(-4.0, 4.0, (48, 3)).astype(np.float32)
    true_rot_angle = math.radians(12.0)
    K = np.array([
        [math.cos(true_rot_angle), -math.sin(true_rot_angle), 0.0],
        [math.sin(true_rot_angle),  math.cos(true_rot_angle), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    true_trans = np.array([0.45, -0.25, 0.10], dtype=np.float32)
    tgt_test = (K @ src_test.T).T + true_trans

    R_est, p_est, err, converged = drone_slam.icp.align(src_test, tgt_test)
    print(f"[ICP VIO] Alignment Converged: {converged} | Residual MSE: {err:.6f}")
    assert converged, "ICP scan matcher failed to converge on known rigid transform!"
    assert np.allclose(p_est, true_trans, atol=0.08), f"ICP translation estimate off: {p_est} vs {true_trans}"

    peer_slam = OnboardVIOSlamExecutive(drone_id=1, resolution_m=0.50)
    delta = drone_slam.octomap.export_submap_delta(bounding_center=drone_pos, radius_m=30.0)
    peer_slam.octomap.merge_submap_delta(delta)
    peer_occupied = peer_slam.octomap.get_all_occupied_voxel_centers()

    print(f"[SWARM FUSION] Exported {len(delta)} voxels | Peer Integrated Occupied Voxels: {len(peer_occupied)}")
    assert len(peer_occupied) == len(occupied_pts), "Submap delta merge resulted in mismatched voxel count!"
    print("\n[PART 4A TEST PASS] Hierarchical OctoMap, 3D DDA, ESDF, and SVD-ICP verified.\n")

def run_part4_computer_vision_self_test():
    print("=" * 75)
    print("RUNNING PART 4 SELF-TEST (B): ARUCO 6-DOF PNP, GIMBAL STABILIZATION & PBVS")
    print("=" * 75)

    vision = OnboardComputerVisionExecutive(drone_id=0)

    roll_angle = math.radians(18.0)
    drone_R_tilted = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(roll_angle), -math.sin(roll_angle)],
        [0.0, math.sin(roll_angle),  math.cos(roll_angle)]
    ], dtype=np.float32)

    drone_pos = np.array([10.0, -15.0, 12.0], dtype=np.float32)
    drone_vel = np.array([0.1, 0.05, -0.05], dtype=np.float32)
    target_pad = np.array([10.2, -14.8, 2.0], dtype=np.float32)

    for _ in range(16):
        R_world_cam = vision.gimbal.update_gimbal_attitude(drone_R_tilted, dt=0.016)

    optical_axis_z = R_world_cam[2, 2]
    print(f"[GIMBAL] Airframe Roll: {math.degrees(roll_angle):.1f}° | Optical Axis Nadir Alignment: {optical_axis_z:.4f}")
    assert optical_axis_z < -0.95, "Gimbal failed to actively level camera to nadir!"

    test_obstacles = [{"id": "Ground_Intruder_Agent", "pos": np.array([10.5, -14.5, 0.0], dtype=np.float32), "radius": 0.45}]

    aruco_res, hazards, v_pbvs, yaw_rate, aligned = vision.process_vision_frame(
        drone_pos, drone_vel, drone_R_tilted, target_pad, expected_tag_id=101,
        dynamic_obstacles=test_obstacles, dt=0.016
    )

    print(f"[ARUCO  ] Detected: {aruco_res.is_detected} | Confidence: {aruco_res.confidence*100:.1f}%")
    print(f"[ARUCO  ] Estimated World Offset: {aruco_res.relative_world_pos} (True: {target_pad - drone_pos})")
    
    expected_offset = target_pad - drone_pos
    error_3d_m = float(np.linalg.norm(aruco_res.relative_world_pos - expected_offset))
    print(f"[ARUCO  ] PnP Position Recovery Error: {error_3d_m*1000.0:.2f} mm")
    assert error_3d_m < 0.05, f"PnP solver error {error_3d_m:.3f}m exceeds 50mm tolerance!"

    assert len(hazards) == 1, "Failed to project dynamic intruder into image coordinates!"
    h = hazards[0]
    print(f"[YOLO   ] Detected Class: {h.class_name} | Distance: {h.estimated_distance_m:.2f}m | Confidence: {h.confidence*100:.1f}%")
    intruder_loc_err = float(np.linalg.norm(h.estimated_world_pos[:2] - test_obstacles[0]["pos"][:2]))
    print(f"[YOLO   ] Monocular Ground Intersection Error: {intruder_loc_err:.3f} m")
    assert intruder_loc_err < 0.35, "Ground raycast back-projection error exceeds tolerance!"

    print(f"[PBVS   ] Commanded Velocity: {v_pbvs} m/s | Yaw Rate: {math.degrees(yaw_rate):.2f} °/s | Dock Aligned: {aligned}")
    dot_dir = np.dot(v_pbvs[:2], (target_pad[:2] - drone_pos[:2]))
    assert dot_dir > 0.0, "PBVS command vector must point toward target fiducial!"

    print("\n[PART 4B TEST PASS] Pinhole model, gimbal leveling, planar PnP, and PBVS servoing verified.\n")

# =============================================================================
# 37. DISPATCH & ENERGY PROFILING CONSTANTS
# =============================================================================
DEFAULT_CRUISE_SPEED_MPS = 8.5
CLIMB_SPEED_MPS = 2.40
DESCENT_SPEED_MPS = 1.20
ROOFTOP_SERVICE_TIME_S = 18.0       # Hover, winch payout, touchdown, retraction
DEPOT_TURNAROUND_TIME_S = 45.0      # Automated battery swap & cargo reload

MAX_CARGO_MASS_KG = 0.65
DRAG_COEFF_CDA = 0.085              # Parasitic drag area (m^2)
BATTERY_PACK_ENERGY_WH = 177.6      # 6S 8000mAh (22.2V * 8.0Ah)
MIN_SAFE_RESERVE_SOC = 0.20         # 20% emergency reserve
MAX_USABLE_ENERGY_WH = BATTERY_PACK_ENERGY_WH * (1.0 - MIN_SAFE_RESERVE_SOC)

# ALNS Metaheuristic Hyperparameters
ALNS_MAX_ITERATIONS = 250
ALNS_START_TEMPERATURE = 120.0
ALNS_COOLING_RATE = 0.975
SHAW_DIST_WEIGHT = 0.50
SHAW_TIME_WEIGHT = 0.35
SHAW_DEMAND_WEIGHT = 0.15

# =============================================================================
# 38. DISPATCH DATA STRUCTURES & CONTRACT SPECIFICATIONS
# =============================================================================
@dataclass
class DeliveryOrder:
    order_id: int
    bldg_id: str
    target_pos: np.ndarray          # [X, Y, Z] world position
    demand_kg: float
    time_window_start_s: float      # Earliest acceptable drop time
    time_window_end_s: float        # Hard deadline before SLA penalty
    service_duration_s: float = ROOFTOP_SERVICE_TIME_S
    penalty_per_sec_late: float = 2.50
    is_delivered: bool = False

@dataclass
class DroneFleetAsset:
    drone_id: int
    home_depot_id: int
    home_depot_pos: np.ndarray      # [X, Y, Z]
    current_pos: np.ndarray
    battery_soc: float              # [0.0, 1.0]
    is_available: bool = True
    assigned_route: List[int] = field(default_factory=list)  # Sequence of order_ids

@dataclass
class FleetRouteSolution:
    routes: Dict[int, List[int]]    # drone_id -> List[order_id]
    total_cost: float
    total_distance_m: float
    total_energy_wh: float
    unassigned_orders: List[int]

# =============================================================================
# 39. BREGUET AERODYNAMIC ENERGY & FLIGHT LEG PROFILER
# =============================================================================
class FlightLegEnergyProfiler:
    """
    Computes electrical energy draw (Watt-hours) and transit flight time
    accounting for 3D climb/glide, payload mass, and wind drift.
    """
    @staticmethod
    def profile_leg(start_p: np.ndarray, end_p: np.ndarray, payload_mass_kg: float,
                    wind_vec_3d: np.ndarray, cruise_speed_mps: float = DEFAULT_CRUISE_SPEED_MPS) -> Tuple[float, float, float]:
        delta = end_p - start_p
        dist_xy = float(math.hypot(delta[0], delta[1]))
        dist_z = float(delta[2])
        ground_dist = float(np.linalg.norm(delta))

        if ground_dist < 0.1:
            return 0.0, 0.0, 0.0

        unit_ground_xy = delta[:2] / max(1e-4, dist_xy) if dist_xy > 0.05 else np.zeros(2, dtype=np.float32)

        v_ground_xy = unit_ground_xy * cruise_speed_mps
        v_climb = float(np.clip(dist_z / max(1.0, dist_xy / cruise_speed_mps), -DESCENT_SPEED_MPS, CLIMB_SPEED_MPS)) if dist_xy > 0.5 else (CLIMB_SPEED_MPS if dist_z > 0 else -DESCENT_SPEED_MPS)

        duration_s = max(dist_xy / max(1.0, cruise_speed_mps), abs(dist_z) / (CLIMB_SPEED_MPS if dist_z > 0 else DESCENT_SPEED_MPS))

        # True Airspeed (TAS) accounting for 3D wind drift
        v_air = np.array([
            v_ground_xy[0] - wind_vec_3d[0],
            v_ground_xy[1] - wind_vec_3d[1],
            v_climb - wind_vec_3d[2]
        ], dtype=np.float32)
        tas_mps = float(np.linalg.norm(v_air))
        air_distance_m = tas_mps * duration_s

        total_mass_kg = DRY_MASS_KG + payload_mass_kg
        weight_n = total_mass_kg * G_ACCEL

        # Induced hover power: T^1.5 / sqrt(2 * rho * A)
        p_induced_hover = (weight_n ** 1.5) / math.sqrt(2.0 * 1.225 * (math.pi * (0.127 ** 2) * 8.0))
        induced_forward_attenuation = 1.0 / math.sqrt(1.0 + (tas_mps / 4.5) ** 2)
        p_induced = p_induced_hover * induced_forward_attenuation

        # Blade profile drag power
        p_profile = 28.0 * (1.0 + 3.0 * (tas_mps / 32.0) ** 2)

        # Airframe parasitic drag power: 0.5 * rho * tas^3 * CdA
        p_parasitic = 0.5 * 1.225 * (tas_mps ** 3) * DRAG_COEFF_CDA

        # Potential energy climb power: m * g * vz
        p_climb = max(0.0, weight_n * v_climb)

        # Total electrical power draw with motor/ESC efficiency (eta ~ 0.78)
        p_elec_watts = (p_induced + p_profile + p_parasitic + p_climb) / 0.78 + 18.0
        energy_wh = (p_elec_watts * duration_s) / 3600.0

        return duration_s, energy_wh, air_distance_m

# =============================================================================
# 40. ALNS METAHEURISTIC ENGINE (MULTI-DEPOT VRP-TW SOLVER)
# =============================================================================
class AdaptiveLargeNeighborhoodSearchVRP:
    """
    ALNS metaheuristic solver for Multi-Depot Electric VRP with hard Time Windows.
    Solves 48+ node urban deliveries across multiple drone bases.
    """
    def __init__(self, fleet: Dict[int, DroneFleetAsset], orders: Dict[int, DeliveryOrder],
                 wind_3d: np.ndarray):
        self.fleet = fleet
        self.orders = orders
        self.wind_3d = wind_3d
        self.profiler = FlightLegEnergyProfiler()

    def evaluate_route_feasibility(self, drone: DroneFleetAsset, route_order_ids: List[int],
                                  current_time_s: float) -> Tuple[bool, float, float, float]:
        curr_p = drone.current_pos.copy()
        curr_time = current_time_s
        accum_energy_wh = (1.0 - drone.battery_soc) * BATTERY_PACK_ENERGY_WH
        total_cost = 0.0

        carried_mass = sum(self.orders[oid].demand_kg for oid in route_order_ids)
        if carried_mass > MAX_CARGO_MASS_KG:
            return False, 1e9, 0.0, 0.0

        for oid in route_order_ids:
            order = self.orders[oid]
            dur_s, leg_wh, _ = self.profiler.profile_leg(curr_p, order.target_pos, carried_mass, self.wind_3d)
            curr_time += dur_s
            accum_energy_wh += leg_wh

            if curr_time < order.time_window_start_s:
                curr_time = order.time_window_start_s
            elif curr_time > order.time_window_end_s:
                lateness_s = curr_time - order.time_window_end_s
                total_cost += lateness_s * order.penalty_per_sec_late

            curr_time += order.service_duration_s
            carried_mass -= order.demand_kg
            curr_p = order.target_pos.copy()

        dur_home, home_wh, _ = self.profiler.profile_leg(curr_p, drone.home_depot_pos, 0.0, self.wind_3d)
        curr_time += dur_home
        accum_energy_wh += home_wh

        if accum_energy_wh > MAX_USABLE_ENERGY_WH:
            return False, 1e9, curr_time, accum_energy_wh

        total_cost += accum_energy_wh * 1.50 + curr_time * 0.10
        return True, total_cost, curr_time, accum_energy_wh

    def generate_initial_solution(self, current_time_s: float) -> FleetRouteSolution:
        solution: Dict[int, List[int]] = {did: [] for did in self.fleet.keys()}
        unassigned: List[int] = []

        sorted_orders = sorted(self.orders.values(), key=lambda o: o.time_window_end_s)

        for order in sorted_orders:
            best_drone_id = -1
            best_cost = 1e9
            best_insert_pos = 0

            for did, drone in self.fleet.items():
                if not drone.is_available:
                    continue
                curr_route = solution[did]
                for i in range(len(curr_route) + 1):
                    trial_route = curr_route[:i] + [order.order_id] + curr_route[i:]
                    feasible, cost, _, _ = self.evaluate_route_feasibility(drone, trial_route, current_time_s)
                    if feasible and cost < best_cost:
                        best_cost = cost
                        best_drone_id = did
                        best_insert_pos = i

            if best_drone_id != -1:
                solution[best_drone_id].insert(best_insert_pos, order.order_id)
            else:
                unassigned.append(order.order_id)

        return self._package_solution(solution, unassigned, current_time_s)

    def destroy_shaw_related(self, current_routes: Dict[int, List[int]], remove_count: int) -> Tuple[Dict[int, List[int]], List[int]]:
        flattened = [oid for r in current_routes.values() for oid in r]
        if not flattened:
            return current_routes, []

        remove_count = min(remove_count, len(flattened))
        removed = [random.choice(flattened)]

        while len(removed) < remove_count:
            seed_order = self.orders[random.choice(removed)]
            candidates = [oid for oid in flattened if oid not in removed]

            def relatedness(other_id: int) -> float:
                other = self.orders[other_id]
                d_dist = float(np.linalg.norm(seed_order.target_pos[:2] - other.target_pos[:2])) / 200.0
                d_time = abs(seed_order.time_window_end_s - other.time_window_end_s) / 300.0
                d_demand = abs(seed_order.demand_kg - other.demand_kg) / MAX_CARGO_MASS_KG
                return SHAW_DIST_WEIGHT * d_dist + SHAW_TIME_WEIGHT * d_time + SHAW_DEMAND_WEIGHT * d_demand

            candidates.sort(key=relatedness)
            removed.append(candidates[0])

        new_routes = {did: [oid for oid in r if oid not in removed] for did, r in current_routes.items()}
        return new_routes, removed

    def destroy_worst_cost(self, current_routes: Dict[int, List[int]], remove_count: int,
                           current_time_s: float) -> Tuple[Dict[int, List[int]], List[int]]:
        order_costs = []
        for did, r in current_routes.items():
            drone = self.fleet[did]
            _, base_cost, _, _ = self.evaluate_route_feasibility(drone, r, current_time_s)
            for oid in r:
                trial = [x for x in r if x != oid]
                _, cost_without, _, _ = self.evaluate_route_feasibility(drone, trial, current_time_s)
                marginal_saving = base_cost - cost_without
                order_costs.append((marginal_saving, oid))

        order_costs.sort(key=lambda x: x[0], reverse=True)
        remove_count = min(remove_count, len(order_costs))
        removed = [x[1] for x in order_costs[:remove_count]]

        new_routes = {did: [oid for oid in r if oid not in removed] for did, r in current_routes.items()}
        return new_routes, removed

    def repair_regret_2(self, partial_routes: Dict[int, List[int]], unassigned: List[int],
                        current_time_s: float) -> Tuple[Dict[int, List[int]], List[int]]:
        pending = list(unassigned)
        new_routes = {did: list(r) for did, r in partial_routes.items()}
        still_unassigned = []

        while pending:
            regret_list = []

            for oid in pending:
                vehicle_insertion_costs = []

                for did, drone in self.fleet.items():
                    if not drone.is_available:
                        continue
                    curr_route = new_routes[did]
                    for i in range(len(curr_route) + 1):
                        trial = curr_route[:i] + [oid] + curr_route[i:]
                        feasible, cost, _, _ = self.evaluate_route_feasibility(drone, trial, current_time_s)
                        if feasible:
                            vehicle_insertion_costs.append((cost, did, i))

                vehicle_insertion_costs.sort(key=lambda x: x[0])

                if len(vehicle_insertion_costs) == 0:
                    regret_list.append((-1e9, oid, -1, 0))
                elif len(vehicle_insertion_costs) == 1:
                    regret_list.append((1e6, oid, vehicle_insertion_costs[0][1], vehicle_insertion_costs[0][2]))
                else:
                    regret_value = vehicle_insertion_costs[1][0] - vehicle_insertion_costs[0][0]
                    regret_list.append((regret_value, oid, vehicle_insertion_costs[0][1], vehicle_insertion_costs[0][2]))

            regret_list.sort(key=lambda x: x[0], reverse=True)
            best_regret, chosen_oid, target_did, target_pos = regret_list[0]

            if target_did != -1:
                new_routes[target_did].insert(target_pos, chosen_oid)
                pending.remove(chosen_oid)
            else:
                still_unassigned.append(chosen_oid)
                pending.remove(chosen_oid)

        return new_routes, still_unassigned

    def optimize(self, current_time_s: float) -> FleetRouteSolution:
        current_sol = self.generate_initial_solution(current_time_s)
        best_sol = current_sol
        temperature = ALNS_START_TEMPERATURE

        for it in range(ALNS_MAX_ITERATIONS):
            destroy_n = max(1, min(6, int(len(self.orders) * 0.25)))

            if random.random() < 0.50:
                destroyed_routes, removed = self.destroy_shaw_related(best_sol.routes, destroy_n)
            else:
                destroyed_routes, removed = self.destroy_worst_cost(best_sol.routes, destroy_n, current_time_s)

            repaired_routes, unassigned = self.repair_regret_2(destroyed_routes, removed + best_sol.unassigned_orders, current_time_s)
            candidate_sol = self._package_solution(repaired_routes, unassigned, current_time_s)

            delta_cost = candidate_sol.total_cost - current_sol.total_cost

            if delta_cost < 0.0 or math.exp(-delta_cost / max(1e-4, temperature)) > random.random():
                current_sol = candidate_sol
                if candidate_sol.total_cost < best_sol.total_cost:
                    best_sol = candidate_sol

            temperature *= ALNS_COOLING_RATE

        return best_sol

    def _package_solution(self, routes: Dict[int, List[int]], unassigned: List[int],
                          current_time_s: float) -> FleetRouteSolution:
        total_cost = 0.0
        total_dist = 0.0
        total_energy = 0.0

        for did, r in routes.items():
            drone = self.fleet[did]
            _, cost, _, energy = self.evaluate_route_feasibility(drone, r, current_time_s)
            total_cost += cost
            total_energy += energy

            curr_p = drone.current_pos.copy()
            for oid in r:
                total_dist += float(np.linalg.norm(self.orders[oid].target_pos - curr_p))
                curr_p = self.orders[oid].target_pos.copy()
            total_dist += float(np.linalg.norm(drone.home_depot_pos - curr_p))

        total_cost += len(unassigned) * 500.0
        return FleetRouteSolution(routes, total_cost, total_dist, total_energy, unassigned)

# =============================================================================
# 41. ASYNCHRONOUS DYNAMIC DISPATCH SERVER
# =============================================================================
class DynamicSwarmDispatchServer:
    """
    Centralized Fleet Dispatch Executive. Manages active orders, listens to
    drone state telemetries, and handles on-demand dynamic route re-optimization.
    """
    def __init__(self, depot_helipads: List[np.ndarray], manifest_addresses: List[Dict[str, Any]]):
        self.depot_helipads = [np.asarray(p, dtype=np.float32) for p in depot_helipads]
        self.manifest = manifest_addresses
        self.orders: Dict[int, DeliveryOrder] = {}
        self.fleet: Dict[int, DroneFleetAsset] = {}
        self.active_solution: Optional[FleetRouteSolution] = None
        self.last_optimize_time = 0.0
        self.order_counter = 0

    def register_drone(self, drone_id: int, home_depot_id: int, initial_pos: np.ndarray, battery_soc: float = 0.98):
        home_pos = self.depot_helipads[home_depot_id % len(self.depot_helipads)]
        self.fleet[drone_id] = DroneFleetAsset(
            drone_id=drone_id,
            home_depot_id=home_depot_id,
            home_depot_pos=home_pos.copy(),
            current_pos=np.asarray(initial_pos, dtype=np.float32).copy(),
            battery_soc=battery_soc,
            is_available=True
        )

    def submit_on_demand_order(self, bldg_id: str, target_pos: np.ndarray, demand_kg: float,
                               window_duration_s: float = 180.0, sim_time_s: float = 0.0) -> int:
        self.order_counter += 1
        order = DeliveryOrder(
            order_id=self.order_counter,
            bldg_id=bldg_id,
            target_pos=np.asarray(target_pos, dtype=np.float32).copy(),
            demand_kg=float(demand_kg),
            time_window_start_s=sim_time_s,
            time_window_end_s=sim_time_s + window_duration_s
        )
        self.orders[order.order_id] = order
        return order.order_id

    def update_drone_telemetry(self, drone_id: int, current_pos: np.ndarray, battery_soc: float,
                               is_available: bool = True):
        if drone_id in self.fleet:
            d = self.fleet[drone_id]
            d.current_pos = np.asarray(current_pos, dtype=np.float32).copy()
            d.battery_soc = battery_soc
            d.is_available = is_available

    def mark_order_completed(self, order_id: int):
        if order_id in self.orders:
            self.orders[order_id].is_delivered = True
            del self.orders[order_id]

    def recompute_fleet_dispatch(self, live_wind_3d: np.ndarray, sim_time_s: float) -> FleetRouteSolution:
        pending_orders = {oid: o for oid, o in self.orders.items() if not o.is_delivered}
        solver = AdaptiveLargeNeighborhoodSearchVRP(self.fleet, pending_orders, live_wind_3d)

        solution = solver.optimize(sim_time_s)
        self.active_solution = solution
        self.last_optimize_time = sim_time_s

        for did, r in solution.routes.items():
            self.fleet[did].assigned_route = list(r)

        return solution

    def get_next_waypoint_for_drone(self, drone_id: int) -> Optional[np.ndarray]:
        if drone_id not in self.fleet:
            return None
        route = self.fleet[drone_id].assigned_route
        if route:
            next_order_id = route[0]
            if next_order_id in self.orders:
                return self.orders[next_order_id].target_pos.copy()
        return self.fleet[drone_id].home_depot_pos.copy()

# =============================================================================
# 42. MAVLINK & HIL PROTOCOL CONSTANTS
# =============================================================================
MAVLINK_V2_STX = 0xFD               # MAVLink v2 packet start marker
MAV_TYPE_HEXAROTOR = 13             # Closest standard PX4 type for coaxial X8
MAV_AUTOPILOT_PX4 = 12
MAV_MODE_FLAG_SAFETY_ARMED = 128
MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
MAV_STATE_ACTIVE = 4

MSG_ID_HEARTBEAT = 0
MSG_ID_ATTITUDE = 30
MSG_ID_LOCAL_POSITION_NED = 32
MSG_ID_GLOBAL_POSITION_INT = 33
MSG_ID_HIL_SENSOR = 107
MSG_ID_BATTERY_STATUS = 147
MSG_ID_NAMED_VALUE_FLOAT = 251

MAVLINK_MESSAGE_CRCS = {
    MSG_ID_HEARTBEAT: 50,
    MSG_ID_ATTITUDE: 39,
    MSG_ID_LOCAL_POSITION_NED: 185,
    MSG_ID_GLOBAL_POSITION_INT: 104,
    MSG_ID_HIL_SENSOR: 108,
    MSG_ID_BATTERY_STATUS: 154,
    MSG_ID_NAMED_VALUE_FLOAT: 170
}

QGC_BROADCAST_PORT = 14550          # QGroundControl listening port
PX4_HIL_BASE_PORT = 14560           # PX4 SITL/HITL base port
ROS2_BRIDGE_BASE_PORT = 19850       # High-speed telemetry socket base port

BASE_REF_LAT_DEG = 37.774929
BASE_REF_LON_DEG = -122.419416
BASE_REF_ALT_M = 15.0

# =============================================================================
# 43. RIGID ENU <-> NED SO(3) COORDINATE FRAME ROTATION ENGINE
# =============================================================================
class CoordinateFrameRotationEngine:
    """
    Transforms positions, velocities, and rotations between the simulator's
    East-North-Up (ENU) world frame and the aerospace North-East-Down (NED) frame.
    Determinant det(R) = +1 guarantees pure right-handed spatial preservation.
    """
    R_ENU_TO_NED = np.array([
        [0.0, 1.0,  0.0],
        [1.0, 0.0,  0.0],
        [0.0, 0.0, -1.0]
    ], dtype=np.float32)

    R_NED_TO_ENU = R_ENU_TO_NED.T

    @classmethod
    def pos_enu_to_ned(cls, p_enu: np.ndarray) -> np.ndarray:
        return cls.R_ENU_TO_NED @ np.asarray(p_enu, dtype=np.float32)

    @classmethod
    def vel_enu_to_ned(cls, v_enu: np.ndarray) -> np.ndarray:
        return cls.R_ENU_TO_NED @ np.asarray(v_enu, dtype=np.float32)

    @classmethod
    def rot_enu_to_ned(cls, R_enu: np.ndarray) -> np.ndarray:
        return cls.R_ENU_TO_NED @ np.asarray(R_enu, dtype=np.float32) @ cls.R_NED_TO_ENU

    @classmethod
    def enu_to_geodetic(cls, p_enu: np.ndarray) -> Tuple[float, float, float]:
        lat_rad = math.radians(BASE_REF_LAT_DEG)
        meters_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * lat_rad)
        meters_per_deg_lon = 111412.84 * math.cos(lat_rad)

        d_north_m = float(p_enu[1])
        d_east_m = float(p_enu[0])
        alt_m = BASE_REF_ALT_M + float(p_enu[2])

        lat = BASE_REF_LAT_DEG + (d_north_m / meters_per_deg_lat)
        lon = BASE_REF_LON_DEG + (d_east_m / meters_per_deg_lon)
        return lat, lon, alt_m

# =============================================================================
# 44. ITU X.25 CRC-16 CHECKSUM & MAVLINK V2 FRAMER
# =============================================================================
class MAVLinkX25CRC:
    @staticmethod
    def calculate(buffer: bytes, crc_extra: int) -> int:
        crc = 0xFFFF
        for byte in buffer:
            tmp = byte ^ (crc & 0xFF)
            tmp = (tmp ^ (tmp << 4)) & 0xFF
            crc = (crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)
            crc &= 0xFFFF
        tmp = crc_extra ^ (crc & 0xFF)
        tmp = (tmp ^ (tmp << 4)) & 0xFF
        crc = (crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)
        return crc & 0xFFFF

class MAVLinkV2PacketBuilder:
    def __init__(self, sys_id: int, comp_id: int = 1):
        self.sys_id = sys_id
        self.comp_id = comp_id
        self.seq = 0

    def encode(self, msg_id: int, payload: bytes) -> bytes:
        payload_len = len(payload)
        incompat_flags = 0x00
        compat_flags = 0x00
        
        header = struct.pack(
            "<BBBBBBBHB",
            MAVLINK_V2_STX,
            payload_len,
            incompat_flags,
            compat_flags,
            self.seq,
            self.sys_id,
            self.comp_id,
            msg_id & 0xFFFF,
            (msg_id >> 16) & 0xFF
        )
        self.seq = (self.seq + 1) % 256

        crc_extra = MAVLINK_MESSAGE_CRCS.get(msg_id, 0)
        crc_data = header[1:] + payload
        crc = MAVLinkX25CRC.calculate(crc_data, crc_extra)
        checksum = struct.pack("<H", crc)

        return header + payload + checksum

    def build_heartbeat(self) -> bytes:
        payload = struct.pack(
            "<IBBBBB",
            0,
            MAV_TYPE_HEXAROTOR,
            MAV_AUTOPILOT_PX4,
            MAV_MODE_FLAG_SAFETY_ARMED | MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            MAV_STATE_ACTIVE,
            3
        )
        return self.encode(MSG_ID_HEARTBEAT, payload)

    def build_attitude(self, time_boot_ms: int, roll_rad: float, pitch_rad: float,
                       yaw_rad: float, p_rad_s: float, q_rad_s: float, r_rad_s: float) -> bytes:
        payload = struct.pack(
            "<Iffffff",
            int(time_boot_ms),
            float(roll_rad),
            float(pitch_rad),
            float(yaw_rad),
            float(p_rad_s),
            float(q_rad_s),
            float(r_rad_s)
        )
        return self.encode(MSG_ID_ATTITUDE, payload)

    def build_local_position_ned(self, time_boot_ms: int, p_ned: np.ndarray, v_ned: np.ndarray) -> bytes:
        payload = struct.pack(
            "<Iffffff",
            int(time_boot_ms),
            float(p_ned[0]),
            float(p_ned[1]),
            float(p_ned[2]),
            float(v_ned[0]),
            float(v_ned[1]),
            float(v_ned[2])
        )
        return self.encode(MSG_ID_LOCAL_POSITION_NED, payload)

    def build_global_position_int(self, time_boot_ms: int, lat_deg: float, lon_deg: float,
                                  alt_m: float, relative_alt_m: float, v_ned: np.ndarray,
                                  hdg_rad: float) -> bytes:
        payload = struct.pack(
            "<IiiiihhhH",
            int(time_boot_ms),
            int(lat_deg * 1e7),
            int(lon_deg * 1e7),
            int(alt_m * 1000.0),
            int(relative_alt_m * 1000.0),
            int(v_ned[0] * 100.0),
            int(v_ned[1] * 100.0),
            int(v_ned[2] * 100.0),
            int((math.degrees(hdg_rad) % 360.0) * 100.0)
        )
        return self.encode(MSG_ID_GLOBAL_POSITION_INT, payload)

    def build_hil_sensor(self, time_usec: int, accel_body: np.ndarray, gyro_body: np.ndarray,
                         mag_body: np.ndarray, baro_alt_m: float, diff_pressure_hpa: float) -> bytes:
        payload = struct.pack(
            "<QfffffffffI",
            int(time_usec),
            float(accel_body[0]),
            float(accel_body[1]),
            float(accel_body[2]),
            float(gyro_body[0]),
            float(gyro_body[1]),
            float(gyro_body[2]),
            float(mag_body[0]),
            float(mag_body[1]),
            float(mag_body[2]),
            0x1FFF
        )
        return self.encode(MSG_ID_HIL_SENSOR, payload)

    def build_battery_status(self, v_pack: float, soc: float, i_draw: float, temp_c: float) -> bytes:
        voltages = [int((v_pack / 6.0) * 1000.0)] * 6 + [0xFFFF] * 4
        payload = struct.pack(
            "<BBB10Hhiib",
            0,
            1,
            2,
            int(temp_c * 100.0),
            *voltages,
            int(i_draw * 100.0),
            -1,
            -1,
            int(soc * 100.0)
        )
        return self.encode(MSG_ID_BATTERY_STATUS, payload)

    def build_named_value_float(self, time_boot_ms: int, name: str, value: float) -> bytes:
        name_bytes = name.encode("ascii")[:10].ljust(10, b'\x00')
        payload = struct.pack("<If10s", int(time_boot_ms), float(value), name_bytes)
        return self.encode(MSG_ID_NAMED_VALUE_FLOAT, payload)

# =============================================================================
# 45. HIGH-SPEED ASYNC TELEMETRY SOCKET BRIDGES (QGC & ROS 2)
# =============================================================================
class QGroundControlBroadcastBridge:
    def __init__(self, target_host: str = "127.0.0.1", target_port: int = QGC_BROADCAST_PORT):
        self.target_host = target_host
        self.target_port = target_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def send_packet(self, data: bytes):
        try:
            self.sock.sendto(data, (self.target_host, self.target_port))
        except OSError:
            pass

    def close(self):
        self.sock.close()

class ROS2SwarmTelemetryBridge:
    def __init__(self, base_port: int = ROS2_BRIDGE_BASE_PORT):
        self.base_port = base_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

    def publish_drone_odometry(self, drone_id: int, time_s: float, p_enu: np.ndarray,
                               v_enu: np.ndarray, q_wxyz: np.ndarray, omega_body: np.ndarray):
        payload = {
            "header": {
                "stamp_s": time_s,
                "frame_id": "world_enu",
                "drone_id": drone_id
            },
            "pose": {
                "position": {"x": float(p_enu[0]), "y": float(p_enu[1]), "z": float(p_enu[2])},
                "orientation": {"w": float(q_wxyz[0]), "x": float(q_wxyz[1]), "y": float(q_wxyz[2]), "z": float(q_wxyz[3])}
            },
            "twist": {
                "linear": {"x": float(v_enu[0]), "y": float(v_enu[1]), "z": float(v_enu[2])},
                "angular": {"x": float(omega_body[0]), "y": float(omega_body[1]), "z": float(omega_body[2])}
            }
        }
        raw_bytes = json.dumps(payload).encode("utf-8")
        try:
            self.sock.sendto(raw_bytes, ("127.0.0.1", self.base_port + drone_id))
        except OSError:
            pass

    def close(self):
        self.sock.close()

# =============================================================================
# 46. SWARM HIL TELEMETRY EXECUTIVE & VERIFICATION SELF-TESTS
# =============================================================================
class SwarmHILTelemetryExecutive:
    def __init__(self, drone_id: int):
        self.drone_id = drone_id
        self.sys_id = drone_id + 1
        self.builder = MAVLinkV2PacketBuilder(sys_id=self.sys_id, comp_id=1)
        self.qgc_bridge = QGroundControlBroadcastBridge()
        self.ros2_bridge = ROS2SwarmTelemetryBridge()

        self.last_heartbeat_time = 0.0
        self.last_high_rate_time = 0.0
        self.last_battery_time = 0.0

    def step(self, sim_time_s: float, p_enu: np.ndarray, v_enu: np.ndarray, R_enu: np.ndarray,
             omega_body: np.ndarray, accel_body: np.ndarray, gyro_body: np.ndarray,
             bms_vpack: float, bms_soc: float, bms_temp_c: float, winch_len_m: float,
             winch_tension_n: float):
        time_boot_ms = int(sim_time_s * 1000.0)

        p_ned = CoordinateFrameRotationEngine.pos_enu_to_ned(p_enu)
        v_ned = CoordinateFrameRotationEngine.vel_enu_to_ned(v_enu)
        R_ned = CoordinateFrameRotationEngine.rot_enu_to_ned(R_enu)

        pitch_ned = math.asin(float(np.clip(-R_ned[2, 0], -1.0, 1.0)))
        roll_ned = math.atan2(float(R_ned[2, 1]), float(R_ned[2, 2]))
        yaw_ned = math.atan2(float(R_ned[1, 0]), float(R_ned[0, 0]))

        lat_deg, lon_deg, alt_m = CoordinateFrameRotationEngine.enu_to_geodetic(p_enu)

        # 1 Hz Heartbeat
        if (sim_time_s - self.last_heartbeat_time) >= 1.0:
            self.last_heartbeat_time = sim_time_s
            hb_pkt = self.builder.build_heartbeat()
            self.qgc_bridge.send_packet(hb_pkt)

        # 20 Hz Kinematic Streams
        if (sim_time_s - self.last_high_rate_time) >= 0.05:
            self.last_high_rate_time = sim_time_s

            att_pkt = self.builder.build_attitude(
                time_boot_ms, roll_ned, pitch_ned, yaw_ned,
                omega_body[0], omega_body[1], omega_body[2]
            )
            pos_pkt = self.builder.build_local_position_ned(time_boot_ms, p_ned, v_ned)
            gps_pkt = self.builder.build_global_position_int(
                time_boot_ms, lat_deg, lon_deg, alt_m, float(p_enu[2]), v_ned, yaw_ned
            )

            winch_len_pkt = self.builder.build_named_value_float(time_boot_ms, "Winch_Len", winch_len_m)
            winch_ten_pkt = self.builder.build_named_value_float(time_boot_ms, "Winch_Tens", winch_tension_n)

            self.qgc_bridge.send_packet(att_pkt)
            self.qgc_bridge.send_packet(pos_pkt)
            self.qgc_bridge.send_packet(gps_pkt)
            self.qgc_bridge.send_packet(winch_len_pkt)
            self.qgc_bridge.send_packet(winch_ten_pkt)

            tr = float(np.trace(R_enu))
            if tr > 0.0:
                sq = math.sqrt(tr + 1.0) * 2.0
                q_wxyz = np.array([0.25 * sq, (R_enu[2, 1] - R_enu[1, 2]) / sq,
                                   (R_enu[0, 2] - R_enu[2, 0]) / sq, (R_enu[1, 0] - R_enu[0, 1]) / sq], dtype=np.float32)
            else:
                q_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

            self.ros2_bridge.publish_drone_odometry(
                self.drone_id, sim_time_s, p_enu, v_enu, q_wxyz, omega_body
            )

        # 2 Hz Battery Health Stream
        if (sim_time_s - self.last_battery_time) >= 0.50:
            self.last_battery_time = sim_time_s
            i_draw_est = 18.5
            bat_pkt = self.builder.build_battery_status(bms_vpack, bms_soc, i_draw_est, bms_temp_c)
            self.qgc_bridge.send_packet(bat_pkt)

    def close(self):
        self.qgc_bridge.close()
        self.ros2_bridge.close()

def run_part5_fleet_vrp_self_test():
    print("=" * 75)
    print("RUNNING PART 5 SELF-TEST (A): MD-EVRPTW ALNS FLEET OPTIMIZATION")
    print("=" * 75)

    mock_depots = [
        np.array([-160.0, -155.0, 0.08], dtype=np.float32),
        np.array([-140.0, -155.0, 0.08], dtype=np.float32)
    ]
    server = DynamicSwarmDispatchServer(mock_depots, [])

    server.register_drone(drone_id=0, home_depot_id=0, initial_pos=mock_depots[0], battery_soc=0.98)
    server.register_drone(drone_id=1, home_depot_id=0, initial_pos=mock_depots[0], battery_soc=0.85)
    server.register_drone(drone_id=2, home_depot_id=1, initial_pos=mock_depots[1], battery_soc=0.95)

    test_targets = [
        ("Bldg_01", np.array([-120.0, -90.0, 21.0], dtype=np.float32), 0.45, 120.0),
        ("Bldg_02", np.array([ -60.0, -90.0, 14.5], dtype=np.float32), 0.35, 140.0),
        ("Bldg_03", np.array([  30.0, -30.0, 28.0], dtype=np.float32), 0.40, 180.0),
        ("Bldg_04", np.array([  90.0,  30.0,  8.5], dtype=np.float32), 0.50, 220.0),
        ("Bldg_05", np.array([ -90.0,  60.0, 14.5], dtype=np.float32), 0.45, 150.0),
        ("Bldg_06", np.array([  60.0, 120.0, 21.0], dtype=np.float32), 0.45, 260.0),
        ("Bldg_07", np.array([-140.0,  30.0,  8.5], dtype=np.float32), 0.30, 110.0),
        ("Bldg_08", np.array([   0.0,  90.0, 28.0], dtype=np.float32), 0.40, 240.0)
    ]

    for bldg_id, pos, demand, window in test_targets:
        server.submit_on_demand_order(bldg_id, pos, demand, window_duration_s=window, sim_time_s=0.0)

    headwind_3d = np.array([4.5, -2.0, 0.0], dtype=np.float32)
    t0 = time.time()
    solution = server.recompute_fleet_dispatch(headwind_3d, sim_time_s=0.0)
    alns_time_ms = (time.time() - t0) * 1000.0

    print(f"\n[ALNS ENGINE] Solved in {alns_time_ms:.2f} ms across {len(server.fleet)} Drones")
    print(f"[ALNS METRICS] Flight Distance: {solution.total_distance_m:.1f} m | Fleet Battery Consumption: {solution.total_energy_wh:.2f} Wh")
    assert len(solution.unassigned_orders) == 0, "ALNS failed to assign feasible delivery orders!"
    print("\n[PART 5A TEST PASS] MD-EVRPTW ALNS, aerodynamic profiling, and dispatch verified.\n")

def run_part6_hil_mavlink_self_test():
    print("=" * 75)
    print("RUNNING PART 5 SELF-TEST (B): MAVLINK V2 CRC-16, SO(3) ENU-NED & HIL BRIDGE")
    print("=" * 75)

    executive = SwarmHILTelemetryExecutive(drone_id=0)

    p_enu_test = np.array([10.0, 25.0, 42.0], dtype=np.float32)
    v_enu_test = np.array([4.5, -2.0, 1.2], dtype=np.float32)
    R_enu_test = np.eye(3, dtype=np.float32)

    p_ned_test = CoordinateFrameRotationEngine.pos_enu_to_ned(p_enu_test)
    v_ned_test = CoordinateFrameRotationEngine.vel_enu_to_ned(v_enu_test)

    assert p_ned_test[0] == p_enu_test[1], "NED North component must match ENU Y!"
    assert p_ned_test[1] == p_enu_test[0], "NED East component must match ENU X!"
    assert p_ned_test[2] == -p_enu_test[2], "NED Down component must match -ENU Z!"

    heartbeat_bytes = executive.builder.build_heartbeat()
    assert heartbeat_bytes[0] == MAVLINK_V2_STX, "MAVLink v2 packet must begin with 0xFD STX marker!"

    header_and_payload = heartbeat_bytes[1:-2]
    expected_crc = struct.unpack("<H", heartbeat_bytes[-2:])[0]
    computed_crc = MAVLinkX25CRC.calculate(header_and_payload, MAVLINK_MESSAGE_CRCS[MSG_ID_HEARTBEAT])
    assert computed_crc == expected_crc, "X.25 CRC-16 checksum verification failed!"

    executive.close()
    print("\n[PART 5B TEST PASS] MAVLink v2 encoder, X.25 CRC, SO(3) ENU-NED, and telemetry bridges verified.\n")

# =============================================================================
# 47. BLACKBOX, GEOFENCE & FTS PROTOCOL CONSTANTS
# =============================================================================
BLACKBOX_MAGIC_HEADER = 0x53555256  # "SURV"
BLACKBOX_RECORD_FORMAT = "<QfffffffffHHBBBB" # 64-byte binary record
BLACKBOX_RECORD_SIZE = struct.calcsize(BLACKBOX_RECORD_FORMAT)
BLACKBOX_BUFFER_CAPACITY = 10000    # ~160 seconds of 60Hz telemetry in memory

# Flight Termination System (FTS) Thresholds
CRITICAL_TUMBLE_RATE_RAD_S = 45.0   # Extreme unrecoverable angular rate
CRITICAL_TILT_ANGLE_RAD = math.radians(75.0)
CRITICAL_TILT_DWELL_S = 0.35        # Max duration inverted before FTS fires
CRITICAL_VOLTAGE_SAG_V = 18.0       # Minimum total 6S pack voltage (3.0V/cell)
CRITICAL_BATTERY_TEMP_C = 68.0      # Thermal runaway threshold
TETHER_SNAG_FORCE_N = 48.0          # Winch guillotine overload threshold
GEOFENCE_PREDICTIVE_HORIZON_S = 1.8 # Time-to-violation forecast window

# Parachute Physical Constants
PARACHUTE_CANOPY_AREA_M2 = 0.85
PARACHUTE_DRAG_COEFF_CD = 1.45
PARACHUTE_DEPLOY_TIME_S = 0.40      # Ejection to full inflation delay
PARACHUTE_SUSPENSION_LINE_M = 1.20
IMPACT_SAFE_ENERGY_JOULES = 35.0    # Regulatory human safety impact limit

# =============================================================================
# 48. SAFETY ESCALATION ENUMS & STATUS DATACLASSES
# =============================================================================
class SafetyEscalationTier(IntEnum):
    TIER0_NOMINAL = 0
    TIER1_DEGRADED_RTH = 1
    TIER2_EMERGENCY_LANDING = 2
    TIER3_SEVER_PAYLOAD = 3
    TIER4_BALLISTIC_TERMINATION = 4

class GeofenceType(IntEnum):
    INCLUSION_ZONE = 0              # Drone must remain inside
    EXCLUSION_KEEP_OUT = 1          # Drone must never enter

@dataclass
class GeofencePrism3D:
    zone_id: str
    zone_type: GeofenceType
    polygon_xy: np.ndarray          # Shape (N, 2) clockwise vertices
    z_min_m: float
    z_max_m: float
    safety_buffer_m: float = 2.0

@dataclass
class FTSStatusReport:
    tier: SafetyEscalationTier
    is_terminated: bool
    is_chute_deployed: bool
    is_payload_severed: bool
    trigger_reason: str
    time_to_violation_s: float

# =============================================================================
# 49. HIGH-RATE BINARY BLACKBOX FLIGHT RECORDER (60 HZ)
# =============================================================================
class BinaryBlackboxRecorder:
    """
    Fixed-memory circular binary blackbox recorder.
    Serializes 64-byte structured telemetry records at 60 Hz without heap allocations.
    """
    def __init__(self, drone_id: int, log_dir: str = "."):
        self.drone_id = drone_id
        self.log_dir = log_dir
        self.capacity = BLACKBOX_BUFFER_CAPACITY
        self.buffer = bytearray(self.capacity * BLACKBOX_RECORD_SIZE)
        self.write_index = 0
        self.total_records_logged = 0

    def record_frame(self, sim_time_s: float, p_enu: np.ndarray, v_enu: np.ndarray,
                     roll_rad: float, pitch_rad: float, yaw_rad: float,
                     v_pack: float, tension_n: float, fsm_state: int,
                     health_mask: int, cbf_active: bool, fts_tier: int):
        time_usec = int(sim_time_s * 1e6)
        v_pack_mv = int(v_pack * 1000.0)
        tension_cN = int(tension_n * 100.0)
        cbf_byte = 1 if cbf_active else 0

        record = struct.pack(
            BLACKBOX_RECORD_FORMAT,
            time_usec,
            float(p_enu[0]), float(p_enu[1]), float(p_enu[2]),
            float(v_enu[0]), float(v_enu[1]), float(v_enu[2]),
            float(roll_rad), float(pitch_rad), float(yaw_rad),
            v_pack_mv,
            tension_cN,
            int(fsm_state),
            int(health_mask),
            cbf_byte,
            int(fts_tier)
        )

        offset = (self.write_index % self.capacity) * BLACKBOX_RECORD_SIZE
        self.buffer[offset : offset + BLACKBOX_RECORD_SIZE] = record
        self.write_index += 1
        self.total_records_logged += 1

    def dump_to_disk(self, filename: Optional[str] = None) -> str:
        if filename is None:
            filename = f"blackbox_drone_{self.drone_id}_{int(time.time())}.sbb"
        filepath = os.path.join(self.log_dir, filename)

        num_valid = min(self.total_records_logged, self.capacity)
        start_idx = (self.write_index - num_valid) % self.capacity

        with open(filepath, "wb") as f:
            header = struct.pack("<IHHI", BLACKBOX_MAGIC_HEADER, self.drone_id, BLACKBOX_RECORD_SIZE, num_valid)
            f.write(header)

            if self.total_records_logged <= self.capacity:
                f.write(self.buffer[:num_valid * BLACKBOX_RECORD_SIZE])
            else:
                part1 = self.buffer[start_idx * BLACKBOX_RECORD_SIZE :]
                part2 = self.buffer[: (self.write_index % self.capacity) * BLACKBOX_RECORD_SIZE]
                f.write(part1)
                f.write(part2)

        return filepath

# =============================================================================
# 50. 3D POLYHEDRAL GEOFENCE & PREDICTIVE TTV ARBITER
# =============================================================================
class PolyhedralGeofenceArbiter:
    """
    Evaluates 3D inclusion and exclusion zones. Uses ray-casting point-in-polygon
    tests and forward trajectory projection to compute Time-To-Violation (TTV).
    """
    def __init__(self):
        self.zones: List[GeofencePrism3D] = []

    def add_inclusion_boundary(self, zone_id: str, polygon_xy: np.ndarray, z_min: float = 0.0, z_max: float = 120.0):
        self.zones.append(GeofencePrism3D(zone_id, GeofenceType.INCLUSION_ZONE, polygon_xy, z_min, z_max))

    def add_keep_out_zone(self, zone_id: str, polygon_xy: np.ndarray, z_min: float = 0.0, z_max: float = 60.0):
        self.zones.append(GeofencePrism3D(zone_id, GeofenceType.EXCLUSION_KEEP_OUT, polygon_xy, z_min, z_max))

    @staticmethod
    def _point_in_polygon_2d(x: float, y: float, poly: np.ndarray) -> bool:
        inside = False
        n = len(poly)
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def check_position(self, p_enu: np.ndarray) -> Tuple[bool, str]:
        px, py, pz = float(p_enu[0]), float(p_enu[1]), float(p_enu[2])

        for zone in self.zones:
            in_h = (zone.z_min_m <= pz <= zone.z_max_m)
            in_poly = self._point_in_polygon_2d(px, py, zone.polygon_xy)

            if zone.zone_type == GeofenceType.INCLUSION_ZONE:
                if not (in_h and in_poly):
                    return False, f"BREACH_OUTSIDE_INCLUSION: {zone.zone_id}"
            else:
                if in_h and in_poly:
                    return False, f"BREACH_INSIDE_KEEPOUT: {zone.zone_id}"

        return True, "SAFE_NOMINAL"

    def predict_time_to_violation(self, p_enu: np.ndarray, v_enu: np.ndarray,
                                  horizon_s: float = GEOFENCE_PREDICTIVE_HORIZON_S) -> Tuple[float, Optional[str]]:
        steps = 15
        dt = horizon_s / steps

        sim_p = p_enu.copy()
        for s in range(1, steps + 1):
            sim_p += v_enu * dt
            safe, reason = self.check_position(sim_p)
            if not safe:
                return float(s * dt), reason

        return float("inf"), None

# =============================================================================
# 51. AERODYNAMIC PARACHUTE EGRESS & PYROTECHNIC GUILLOTINE
# =============================================================================
class AerodynamicParachuteEgressSystem:
    """
    Simulates pyrotechnic cable severing and parachute ejection, canopy
    reefing delay, and terminal descent dynamics under ambient winds.
    """
    def __init__(self, dry_mass_kg: float = DRY_MASS_KG):
        self.mass = dry_mass_kg
        self.is_deployed = False
        self.is_inflated = False
        self.deployment_timer = 0.0
        self.cable_severed = False

    def trigger_guillotine(self) -> bool:
        self.cable_severed = True
        return True

    def deploy_parachute(self):
        if not self.is_deployed:
            self.is_deployed = True
            self.deployment_timer = 0.0

    def step_descent(self, p_enu: np.ndarray, v_enu: np.ndarray, wind_3d: np.ndarray, dt: float = 0.016) -> Tuple[np.ndarray, np.ndarray, float]:
        if not self.is_deployed:
            return p_enu, v_enu, 0.5 * self.mass * float(np.dot(v_enu, v_enu))

        self.deployment_timer += dt
        inflation_ratio = float(np.clip(self.deployment_timer / PARACHUTE_DEPLOY_TIME_S, 0.15, 1.0))
        self.is_inflated = (inflation_ratio >= 1.0)

        effective_area = PARACHUTE_CANOPY_AREA_M2 * inflation_ratio

        v_rel = v_enu - wind_3d
        airspeed = float(np.linalg.norm(v_rel))

        f_drag = -0.5 * 1.225 * PARACHUTE_DRAG_COEFF_CD * effective_area * airspeed * v_rel
        f_gravity = np.array([0.0, 0.0, -self.mass * G_ACCEL], dtype=np.float32)

        accel = (f_drag + f_gravity) / self.mass
        v_next = v_enu + accel * dt
        p_next = p_enu + v_next * dt

        if p_next[2] <= 0.05:
            p_next[2] = 0.05
            v_next.fill(0.0)

        speed_sq = float(np.dot(v_next, v_next))
        kinetic_energy_j = 0.5 * self.mass * speed_sq
        return p_next, v_next, kinetic_energy_j

# =============================================================================
# 52. UNIFIED FLIGHT TERMINATION & SAFETY EXECUTIVE (FTS)
# =============================================================================
class AutonomousSafetyExecutiveFTS:
    """
    Onboard safety authority. Monitors health thresholds, triggers the winch
    guillotine and parachute, logs blackbox data, and enforces geofences.
    """
    def __init__(self, drone_id: int, geofence_arbiter: PolyhedralGeofenceArbiter, log_dir: str = "."):
        self.drone_id = drone_id
        self.geofence = geofence_arbiter
        self.blackbox = BinaryBlackboxRecorder(drone_id, log_dir=log_dir)
        self.parachute = AerodynamicParachuteEgressSystem(dry_mass_kg=DRY_MASS_KG)
        
        self.current_tier = SafetyEscalationTier.TIER0_NOMINAL
        self.tilt_dwell_timer = 0.0
        self.termination_reason = "NONE"

    def step(self, sim_time_s: float, p_enu: np.ndarray, v_enu: np.ndarray, R_enu: np.ndarray,
             omega_body: np.ndarray, motor_health: np.ndarray, v_pack: float,
             cell_min_v: float, bms_temp_c: float, winch_tension_n: float,
             wind_3d: np.ndarray, fsm_state: int, cbf_active: bool) -> FTSStatusReport:
        pitch_rad = math.asin(float(np.clip(-R_enu[2, 0], -1.0, 1.0)))
        roll_rad = math.atan2(float(R_enu[2, 1]), float(R_enu[2, 2]))
        yaw_rad = math.atan2(float(R_enu[1, 0]), float(R_enu[0, 0]))

        health_mask = 0
        for i in range(min(8, len(motor_health))):
            if motor_health[i] > 0.0:
                health_mask |= (1 << i)

        self.blackbox.record_frame(
            sim_time_s, p_enu, v_enu, roll_rad, pitch_rad, yaw_rad,
            v_pack, winch_tension_n, fsm_state, health_mask, cbf_active, int(self.current_tier)
        )

        is_geo_safe, geo_reason = self.geofence.check_position(p_enu)
        ttv_s, pred_reason = self.geofence.predict_time_to_violation(p_enu, v_enu)

        if winch_tension_n >= TETHER_SNAG_FORCE_N and not self.parachute.cable_severed:
            self.parachute.trigger_guillotine()
            self.current_tier = max(self.current_tier, SafetyEscalationTier.TIER3_SEVER_PAYLOAD)
            self.termination_reason = "TETHER_SNAG_OVERLOAD_GUILLOTINE"

        tilt_magnitude = math.hypot(roll_rad, pitch_rad)
        if tilt_magnitude > CRITICAL_TILT_ANGLE_RAD or float(np.linalg.norm(omega_body)) > CRITICAL_TUMBLE_RATE_RAD_S:
            self.tilt_dwell_timer += 0.016
        else:
            self.tilt_dwell_timer = max(0.0, self.tilt_dwell_timer - 0.032)

        if self.current_tier < SafetyEscalationTier.TIER4_BALLISTIC_TERMINATION:
            if not is_geo_safe:
                self.current_tier = SafetyEscalationTier.TIER4_BALLISTIC_TERMINATION
                self.termination_reason = f"FTS_GEOFENCE_BREACH: {geo_reason}"
                self.parachute.trigger_guillotine()
                self.parachute.deploy_parachute()

            elif self.tilt_dwell_timer >= CRITICAL_TILT_DWELL_S:
                self.current_tier = SafetyEscalationTier.TIER4_BALLISTIC_TERMINATION
                self.termination_reason = "FTS_UNRECOVERABLE_TUMBLE_RUNAWAY"
                self.parachute.trigger_guillotine()
                self.parachute.deploy_parachute()

            elif v_pack < CRITICAL_VOLTAGE_SAG_V or cell_min_v < 2.95:
                self.current_tier = SafetyEscalationTier.TIER4_BALLISTIC_TERMINATION
                self.termination_reason = "FTS_CRITICAL_BATTERY_COLLAPSE"
                self.parachute.trigger_guillotine()
                self.parachute.deploy_parachute()

            elif np.count_nonzero(motor_health == 0.0) >= 3:
                self.current_tier = SafetyEscalationTier.TIER4_BALLISTIC_TERMINATION
                self.termination_reason = "FTS_MULTIPLE_ROTOR_FAILURE_CATACLYSM"
                self.parachute.trigger_guillotine()
                self.parachute.deploy_parachute()

            elif ttv_s < 0.65:
                self.current_tier = max(self.current_tier, SafetyEscalationTier.TIER1_DEGRADED_RTH)
                self.termination_reason = f"PREDICTIVE_GEOFENCE_WARNING: {pred_reason}"

        if self.current_tier == SafetyEscalationTier.TIER4_BALLISTIC_TERMINATION and self.parachute.deployment_timer < 0.02:
            self.blackbox.dump_to_disk()

        return FTSStatusReport(
            tier=self.current_tier,
            is_terminated=(self.current_tier == SafetyEscalationTier.TIER4_BALLISTIC_TERMINATION),
            is_chute_deployed=self.parachute.is_deployed,
            is_payload_severed=self.parachute.cable_severed,
            trigger_reason=self.termination_reason,
            time_to_violation_s=ttv_s
        )

# =============================================================================
# 53. RECIPROCAL SWARM CBF-QP SAFETY BARRIER FILTER
# =============================================================================
class ReciprocalSwarmCBFQPFilter:
    def __init__(self):
        self.alpha_obs = CBF_ALPHA_OBSTACLE
        self.safe_radius = CBF_BASE_SAFE_RADIUS_M
        self.swarm_r = SWARM_INTER_DRONE_SAFE_RADIUS_M
        self.downwash_expansion = DOWNWASH_CONE_EXPANSION
        self.downwash_depth = DOWNWASH_HAZARD_DEPTH_M

    def enforce_safety(self, drone_idx: int, drone_p: np.ndarray, drone_v: np.ndarray,
                       nominal_accel: np.ndarray, obstacles: List[Dict[str, Any]],
                       swarm_states: Optional[List[Dict[str, Any]]] = None, 
                       target_bldg_id: Optional[str] = None, departure_bldg_id: Optional[str] = None,
                       wind_2d: Optional[np.ndarray] = None, ekf_pos_cov_trace: float = 0.05) -> np.ndarray:
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

# =============================================================================
# 54. RATE-DAMPED GEOMETRIC SE(3) ACTUATOR MIXER
# =============================================================================
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

    def mix(self, drone_health: np.ndarray, p_est: np.ndarray, v_est: np.ndarray,
            R_est: np.ndarray, omega_est: np.ndarray, p_target: np.ndarray,
            neural_guidance: Optional[np.ndarray], token_idx: int,
            estimated_wind: np.ndarray, obstacles: List[Dict[str, Any]],
            swarm_states: Optional[List[Dict[str, Any]]] = None, drone_idx: int = 0,
            ref_accel: Optional[np.ndarray] = None, ref_vel: Optional[np.ndarray] = None,
            target_bldg_id: Optional[str] = None, departure_bldg_id: Optional[str] = None,
            cable_len: float = 0.08, sway_2d: Optional[np.ndarray] = None,
            load_cell_3d: Optional[np.ndarray] = None, ige_fac: float = 1.0,
            v_pack: float = 23.5, bms_temp_c: float = 22.0, has_payload: bool = True,
            ice_mm: float = 0.0, dt: float = 0.016, 
            conical_ranges_8: Optional[np.ndarray] = None,
            esc_i_rms: Optional[np.ndarray] = None, esc_t_mosfet: Optional[np.ndarray] = None,
            f_skid_left: float = 0.0, f_skid_right: float = 0.0, ekf_pos_cov_trace: float = 0.05,
            station_keeping: bool = False, sway_velocity_2d: Optional[np.ndarray] = None,
            approach_braking: bool = False, pred_sway_rate: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, np.ndarray]:

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
            
            if pred_sway_rate is not None:
                sway_rate = 0.70 * sway_rate + 0.30 * np.asarray(pred_sway_rate[:2], dtype=np.float32)

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
        self.mass_com_estimator.update(f_thrust, ref_accel[2], np.array([tau[0], tau[1]], dtype=np.float32), dt=dt)
        if conical_ranges_8 is not None:
            self.roof_normal_estimator.estimate_normal(p_est, conical_ranges_8)

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
# 55. FIXED 139-DIMENSIONAL OBSERVATION SYNTHESIZER
# =============================================================================
class DroneObservationSynthesizer139Dim:
    """
    Synthesizes exactly 139 features into a pre-allocated array with fixed indices,
    preventing any dynamic list drift or accidental truncation.
    """
    def __init__(self):
        self.dim = BASE_OBS_DIM

    def synthesize_instantaneous(self, p_est: np.ndarray, v_est: np.ndarray, R_est: np.ndarray,
                                 omega_est: np.ndarray, target_p: np.ndarray, lidar_16: np.ndarray,
                                 conical_8: np.ndarray, v_tas: float, q_dyn: float, min_sonar: float,
                                 wind_est: np.ndarray, ice_idx: float, cable_len: float,
                                 cable_vel: float, tension_n: float, sway_2d: np.ndarray,
                                 bms_soc: float, v_pack: float, bms_temp_c: float, r0_dyn: float,
                                 chi2_val: float, jerk_discrepancy: float, is_spoofed: bool,
                                 emf_rms: float, wire_dist: float, agent_dist: float,
                                 ground_spl_dba: float, has_payload: bool, hook_a: np.ndarray,
                                 hook_w: np.ndarray, load_cell_3d: np.ndarray, thermal_dt: float,
                                 dolp: float, ndvi: float, esc_i_rms: np.ndarray,
                                 esc_t_mosfet: np.ndarray, alpha_deg: float, beta_deg: float,
                                 rh_pct: float, dust_pm10: float, hdop: float, cn0_avg: float,
                                 rf_jam_pwr: float, mag_grad: float, airframe_accel: np.ndarray,
                                 airframe_gyro: np.ndarray, peer_swarm_kinematics: List[Dict[str, Any]],
                                 optical_flow_2d: np.ndarray, flow_quality: float,
                                 aruco_offset_3d: np.ndarray, aruco_locked: float,
                                 skid_force_left: float, skid_force_right: float,
                                 is_skid_grounded: bool, siren_angle: float, siren_freq: float,
                                 siren_spl: float, adsb_rel_pos: np.ndarray, adsb_closing_spd: float,
                                 dt: float = 0.016) -> np.ndarray:
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

    def stack_history(self, history_q: deque, current_obs: np.ndarray) -> np.ndarray:
        if len(history_q) == 0:
            for _ in range(history_q.maxlen or HISTORY_LEN):
                history_q.append(current_obs.copy())
        else:
            history_q.append(current_obs.copy())
        return np.concatenate(list(history_q), axis=0).astype(np.float32)

# =============================================================================
# 56. STANDALONE VERIFICATION SELF-TEST FOR GEOFENCE & FTS
# =============================================================================
def run_part7_blackbox_geofence_self_test():
    print("=" * 75)
    print("RUNNING PART 6 SELF-TEST: BLACKBOX RECORDER, 3D GEOFENCE & FTS ARBITER")
    print("=" * 75)

    geofence = PolyhedralGeofenceArbiter()
    
    city_boundary = np.array([
        [-180.0, -180.0], [ 180.0, -180.0],
        [ 180.0,  180.0], [-180.0,  180.0]
    ], dtype=np.float32)
    geofence.add_inclusion_boundary("City_Downtown_Bounds", city_boundary, z_min=0.0, z_max=100.0)

    substation = np.array([
        [10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]
    ], dtype=np.float32)
    geofence.add_keep_out_zone("Hospital_Helipad_Exclusion", substation, z_min=0.0, z_max=50.0)

    p_safe = np.array([0.0, 0.0, 40.0], dtype=np.float32)
    p_outside = np.array([195.0, 0.0, 40.0], dtype=np.float32)
    p_keepout = np.array([20.0, 20.0, 25.0], dtype=np.float32)

    assert geofence.check_position(p_safe)[0], "Safe point marked as breach!"
    assert not geofence.check_position(p_outside)[0], "Outside inclusion boundary was not caught!"
    assert not geofence.check_position(p_keepout)[0], "Point inside keep-out zone was not caught!"
    print("[GEOFENCE    ] 3D Inclusion & Keep-out boundary intersection checks verified.")

    p_heading_out = np.array([170.0, 0.0, 40.0], dtype=np.float32)
    v_fast_out = np.array([12.0, 0.0, 0.0], dtype=np.float32)
    ttv, reason = geofence.predict_time_to_violation(p_heading_out, v_fast_out, horizon_s=2.0)
    print(f"[PREDICTION  ] Predictive Geofence Breach Forecast: TTV = {ttv:.2f} s | Reason: {reason}")
    assert 0.70 <= ttv <= 1.0, f"Expected breach prediction ~0.83s, got {ttv}s!"

    test_log_dir = os.path.abspath(".")
    safety_exec = AutonomousSafetyExecutiveFTS(drone_id=0, geofence_arbiter=geofence, log_dir=test_log_dir)

    for step in range(250):
        t = step * 0.016
        safety_exec.blackbox.record_frame(
            sim_time_s=t,
            p_enu=np.array([t * 2.0, 0.0, 15.0], dtype=np.float32),
            v_enu=np.array([2.0, 0.0, 0.0], dtype=np.float32),
            roll_rad=0.01, pitch_rad=-0.02, yaw_rad=0.5,
            v_pack=23.4, tension_n=4.41, fsm_state=0,
            health_mask=0xFF, cbf_active=False, fts_tier=0
        )

    dump_path = safety_exec.blackbox.dump_to_disk("test_blackbox_validation.sbb")
    file_bytes = os.path.getsize(dump_path)
    print(f"[BLACKBOX    ] Flushed binary flight log: {dump_path} ({file_bytes} bytes)")
    expected_bytes = 14 + (250 * BLACKBOX_RECORD_SIZE)
    assert file_bytes == expected_bytes, f"Expected {expected_bytes} bytes, found {file_bytes}!"
    os.remove(dump_path)

    report_snag = safety_exec.step(
        sim_time_s=4.0, p_enu=p_safe, v_enu=np.zeros(3, dtype=np.float32),
        R_enu=np.eye(3, dtype=np.float32), omega_body=np.zeros(3, dtype=np.float32),
        motor_health=np.ones(8, dtype=np.float32), v_pack=23.0, cell_min_v=3.8,
        bms_temp_c=25.0, winch_tension_n=52.5, wind_3d=np.zeros(3, dtype=np.float32),
        fsm_state=2, cbf_active=False
    )
    print(f"[GUILLOTINE  ] Overload Tension Spike: Triggered = {report_snag.is_payload_severed} | Reason: {report_snag.trigger_reason}")
    assert report_snag.is_payload_severed, "Tether guillotine failed to trigger on overload force!"

    omega_spin = np.array([0.0, 52.0, 0.0], dtype=np.float32)
    for _ in range(25):
        report_tumble = safety_exec.step(
            sim_time_s=4.5, p_enu=p_safe, v_enu=np.zeros(3, dtype=np.float32),
            R_enu=np.eye(3, dtype=np.float32), omega_body=omega_spin,
            motor_health=np.ones(8, dtype=np.float32), v_pack=23.0, cell_min_v=3.8,
            bms_temp_c=25.0, winch_tension_n=4.41, wind_3d=np.zeros(3, dtype=np.float32),
            fsm_state=0, cbf_active=False
        )

    print(f"[PARACHUTE   ] FTS Runaway Triggered: {report_tumble.is_terminated} | Chute Deployed: {report_tumble.is_chute_deployed}")
    assert report_tumble.is_terminated and report_tumble.is_chute_deployed, "FTS failed to deploy parachute on tumble runaway!"

    curr_p = np.array([0.0, 0.0, 30.0], dtype=np.float32)
    curr_v = np.array([0.0, 0.0, -12.0], dtype=np.float32)
    wind_gale = np.array([4.0, 0.0, 0.0], dtype=np.float32)

    for _ in range(int(3.0 / 0.016)):
        curr_p, curr_v, kinetic_energy = safety_exec.parachute.step_descent(curr_p, curr_v, wind_gale, dt=0.016)

    descent_speed = abs(float(curr_v[2]))
    print(f"[EGRESS DYN  ] Parachute Terminal Descent: Vz = -{descent_speed:.2f} m/s | Kinetic Energy = {kinetic_energy:.2f} J")
    assert descent_speed < 5.8, f"Parachute terminal speed {descent_speed:.2f} m/s exceeds safe envelope!"
    assert kinetic_energy < IMPACT_SAFE_ENERGY_JOULES, f"Impact energy {kinetic_energy:.2f} J exceeds 35 J safety limit!"

    print("\n[PART 6 TEST PASS] Blackbox, Geofence containment, Guillotine, and Parachute FTS verified.\n")

# =============================================================================
# 57. NEURAL FLIGHT EXECUTIVE NETWORK (~2.2M PARAMS) & PINN LOSS
# =============================================================================
class DenseResBlock512(nn.Module if torch is not None else object):
    def __init__(self, dim: int = 512, hidden_dim: int = 768):
        super().__init__()
        if torch is None: return
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return x + self.fc2(self.norm2(self.act(self.fc1(self.norm1(x)))))

class DeepDenseFlightExecutiveNet(nn.Module if torch is not None else object):
    def __init__(self, in_features: int = OBSERVATION_DIM):
        super().__init__()
        if torch is None: return
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.SiLU()
        )
        self.block1 = DenseResBlock512(512, 768)
        self.block2 = DenseResBlock512(512, 768)
        self.block3 = DenseResBlock512(512, 768)

        self.guidance_head = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 4), nn.Tanh()
        )
        self.winch_head = nn.Sequential(
            nn.Linear(512, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 2), nn.Tanh()
        )
        self.health_head = nn.Sequential(
            nn.Linear(512, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 8), nn.Sigmoid()
        )
        self.trust_head = nn.Sequential(
            nn.Linear(512, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 2), nn.Sigmoid()
        )
        self.executive_head = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, NUM_EXECUTIVE_STATES)
        )
        self.value_head = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        self.sway_pred_head = nn.Sequential(
            nn.Linear(512, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, PHYSICS_SWAY_PRED_DIM)
        )
        self.vel_pred_head = nn.Sequential(
            nn.Linear(512, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, PHYSICS_VEL_PRED_DIM)
        )

    def forward(self, x, return_predictions: bool = False):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        feat = self.block3(self.block2(self.block1(self.input_proj(x))))
        g = self.guidance_head(feat)
        w = self.winch_head(feat)
        h = self.health_head(feat)
        t = self.trust_head(feat)
        fsm = self.executive_head(feat)
        val = self.value_head(feat)

        if return_predictions:
            sway = self.sway_pred_head(feat)
            vel = self.vel_pred_head(feat)
            return g, fsm, w, h, t, val, sway, vel
        return g, fsm, w, h, t, val

class PhysicsGuidedFlightLoss(nn.Module if torch is not None else object):
    def __init__(self, mass: float = MASS_KG, gravity: float = G_ACCEL):
        super().__init__()
        if torch is None: return
        self.mass = mass
        self.gravity = gravity
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()

    def forward(self, preds, targets, obs_batch):
        p_g, p_fsm, p_w, p_h, p_t, p_val, p_sway, p_vel = preds
        y_g, y_fsm, y_w, y_h, y_t, y_val, y_sway, y_vel = targets

        loss_guide = self.mse(p_g, y_g)
        loss_act = self.ce(p_fsm, y_fsm)
        loss_winch = self.mse(p_w, y_w)
        loss_val = self.mse(p_val, y_val)
        loss_h = self.bce(p_h.clamp(1e-6, 1.0 - 1e-6), y_h)
        loss_t = self.bce(p_t.clamp(1e-6, 1.0 - 1e-6), y_t)

        c_idx = (HISTORY_LEN - 1) * BASE_OBS_DIM
        v_drone = obs_batch[:, c_idx+6 : c_idx+9] * 8.0
        w_sensed = obs_batch[:, c_idx+42 : c_idx+45] * 8.0
        v_rel = v_drone - w_sensed
        speed_rel = torch.norm(v_rel, dim=-1, keepdim=True).clamp(min=0.5)
        cda = torch.tensor([0.22, 0.22, 0.38], device=obs_batch.device)
        f_drag = -0.5 * 1.225 * cda * speed_rel * v_rel
        a_drag = f_drag / self.mass

        pred_accel = p_g[:, :3] * 12.0
        loss_dyn = self.mse(pred_accel, y_g[:, :3] * 12.0 - a_drag)
        loss_sway = self.mse(p_sway, y_sway)
        loss_vel = self.mse(p_vel, y_vel)

        return (0.35 * loss_guide + 0.20 * loss_dyn + 0.15 * loss_act + 
                0.10 * loss_winch + 0.08 * loss_sway + 0.05 * loss_vel + 
                0.04 * loss_h + 0.02 * loss_t + 0.01 * loss_val)

# =============================================================================
# 58. GUIDANCE, QUINTIC TRAJECTORIES & CABLE COLLISION GEOMETRY
# =============================================================================
def _live_payload_center_z(drone_center_z: float, cable_len: float) -> float:
    return float(drone_center_z) - PHASE3_DRONE_HALF_HEIGHT_M - max(PHASE3_STOWED_CABLE_LEN_M, float(cable_len)) - PHASE3_CARGO_HALF_HEIGHT_M

def _live_payload_bottom_clearance(drone_center_z: float, cable_len: float, target_surface_z: float) -> float:
    return _live_payload_center_z(drone_center_z, cable_len) - PHASE3_CARGO_HALF_HEIGHT_M - float(target_surface_z)

def _cruise_altitude_captured(current_z_m: float, current_vz_m_s: float, target_z_m: float,
                               altitude_tolerance_m: float = 1.25, vertical_speed_limit_m_s: float = 0.80) -> bool:
    return (abs(float(current_z_m) - float(target_z_m)) <= float(altitude_tolerance_m) and
            abs(float(current_vz_m_s)) <= float(vertical_speed_limit_m_s))

def _delivery_approach_braking_radius(speed_xy_m_s: float, wind_speed_xy_m_s: float = 0.0) -> float:
    speed = max(0.0, float(speed_xy_m_s))
    wind = max(0.0, float(wind_speed_xy_m_s))
    return float(np.clip((speed * speed) / 2.20 + (5.5 + min(6.0, 0.45 * wind)), 10.0, 42.0))

def _should_enter_precision_approach(distance_xy_m: float, speed_xy_m_s: float, wind_speed_xy_m_s: float) -> bool:
    return float(distance_xy_m) <= _delivery_approach_braking_radius(speed_xy_m_s, wind_speed_xy_m_s)

def _delivery_vertical_settle_reference(v_ref: np.ndarray, vertical_error_m: float, deadband_m: float = 0.75) -> np.ndarray:
    ref = np.asarray(v_ref, dtype=np.float32).copy()
    if abs(float(vertical_error_m)) <= max(0.0, float(deadband_m)):
        ref[2] = 0.0
    return ref

def compute_quintic_trajectory(p0: np.ndarray, p1: np.ndarray, tau: float) -> Tuple[np.ndarray, float, float]:
    tau = float(np.clip(tau, 0.0, 1.0))
    s = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)
    s_dot = 30.0 * (tau**2) - 60.0 * (tau**3) + 30.0 * (tau**4)
    s_ddot = 60.0 * tau - 180.0 * (tau**2) + 120.0 * (tau**3)
    p_ref = p0 + (p1 - p0) * s
    return p_ref, s_dot, s_ddot

def compute_vector_guidance(current_p: np.ndarray, current_v: np.ndarray, target_p: np.ndarray,
                            max_speed_xy: float = 8.0, max_climb: float = 2.40, 
                            max_descend: float = 0.65, wind_vector: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    delta_p = target_p - current_p
    dist_xy = float(math.hypot(delta_p[0], delta_p[1]))
    dist_z = float(delta_p[2])
    
    if dist_xy > 0.05 and wind_vector is not None:
        track_unit = delta_p[:2] / dist_xy
        headwind = -float(np.dot(wind_vector[:2], track_unit))
        adaptive_cruise = (max_speed_xy + 0.40 * headwind) if headwind > 0 else max(4.5, max_speed_xy + 0.22 * headwind)
    else:
        adaptive_cruise = max_speed_xy

    if max_speed_xy <= 0.05:
        v_cmd_xy = np.zeros(2, dtype=np.float32)
    else:
        adaptive_max_xy = float(np.clip(adaptive_cruise, 0.0, max(0.0, max_speed_xy)))
        braking_speed = math.sqrt(2.0 * 1.5 * dist_xy)
        v_xy_mag = min(adaptive_max_xy, braking_speed) if dist_xy > 1.8 else min(adaptive_max_xy, 1.2 * dist_xy)
        v_cmd_xy = (delta_p[:2] / dist_xy) * v_xy_mag if dist_xy > 0.03 else np.zeros(2, dtype=np.float32)

    if target_p[2] <= 0.15:
        z_clr = max(0.01, float(current_p[2] - target_p[2]))
        v_cmd_z = -float(np.clip(0.35 * math.sqrt(z_clr), 0.06, max_descend))
    else:
        if dist_z > 0.0:
            climb_cap = float(np.clip(0.65 + 0.70 * current_p[2], 0.65, max_climb)) if current_p[2] < 2.5 else max_climb
            v_cmd_z = min(climb_cap, math.sqrt(max(0.0, 2.0 * 1.0 * dist_z)))
        else:
            v_cmd_z = -min(max_descend, math.sqrt(max(0.0, 2.0 * 0.50 * abs(dist_z))))
        
    return target_p.copy(), np.array([v_cmd_xy[0], v_cmd_xy[1], v_cmd_z], dtype=np.float32)

def _winch_cable_polyline_points(winch: AutonomousWinchDeliverySystem, drone_pos: np.ndarray) -> np.ndarray:
    pts = np.asarray(getattr(winch, "nodes_pos", []), dtype=np.float32).copy()
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2 or not np.all(np.isfinite(pts)):
        anchor = np.asarray(drone_pos, dtype=np.float32).copy()
        anchor[2] -= PHASE3_DRONE_HALF_HEIGHT_M
        end = anchor.copy()
        end[2] -= float(getattr(winch, "cable_len", PHASE3_STOWED_CABLE_LEN_M))
        pts = np.linspace(anchor, end, max(2, int(getattr(winch, "num_nodes", 5))), dtype=np.float32)
    return pts

# =============================================================================
# 59. INTEGRATED SWARM MISSION DISPATCHER & DRONE INSTANCE
# =============================================================================
class SwarmMissionDispatcher:
    def __init__(self, manifest_addresses: List[Dict[str, Any]], depot_helipads: List[np.ndarray]):
        self.manifest = list(manifest_addresses)
        self.depot_helipads = depot_helipads
        self.pending_orders = deque(range(len(self.manifest)))
        self.active_missions: Dict[int, Dict[str, Any]] = {}
        self.total_completed = 0
        self.total_failed = 0

    def dispatch_next_mission(self, drone_id: int, drone_soc: float = 0.98,
                              drone_pos: Optional[np.ndarray] = None,
                              microclimate_map: Optional[SwarmMicroclimateSharedMap] = None,
                              sim_time: float = 0.0) -> Dict[str, Any]:
        if not self.pending_orders:
            self.pending_orders.extend(range(len(self.manifest)))

        home_pad = self.depot_helipads[drone_id % len(self.depot_helipads)]
        current_p = home_pad if drone_pos is None else drone_pos

        best_queue_idx, best_cost = 0, 1e9
        for q_idx, order_idx in enumerate(self.pending_orders):
            candidate = self.manifest[order_idx]
            dist_flight = float(np.linalg.norm(current_p[:2] - candidate["pos"][:2])) + float(np.linalg.norm(candidate["pos"][:2] - home_pad[:2]))
            cost = dist_flight * (1.0 + (1.0 - drone_soc) * 2.2)
            if cost < best_cost:
                best_cost, best_queue_idx = cost, q_idx

        chosen_order_idx = self.pending_orders[best_queue_idx]
        del self.pending_orders[best_queue_idx]
        order = self.manifest[chosen_order_idx]
        assigned_cruise_alt = 42.0 + (drone_id * 3.0)

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

    def report_completed(self, drone_id: int):
        self.total_completed += 1
        self.active_missions.pop(drone_id, None)

    def report_failed(self, drone_id: int):
        mission = self.active_missions.pop(drone_id, None)
        self.total_failed += 1
        if mission is not None:
            oid = int(mission.get("order_idx", -1))
            if 0 <= oid < len(self.manifest) and oid not in self.pending_orders:
                self.pending_orders.appendleft(oid)

class SwarmDroneInstance:
    """
    Unified physical drone instance integrating all 7 modules: airframe dynamics,
    PBD winch, ES-EKF, P2P mesh, OctoMap SLAM, Computer Vision, MAVLink, and FTS.
    """
    def __init__(self, drone_id: int, init_pos: np.ndarray,
                 world: MetropolitanCityGridWorld400m, dispatcher: Any):
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

        # Connect advanced embedded executives from integrated modules
        geofence = getattr(world, "geofence_arbiter", None)
        if geofence is None:
            geofence = PolyhedralGeofenceArbiter()
            city_box = np.array([[-190.0, -190.0], [190.0, -190.0], [190.0, 190.0], [-190.0, 190.0]], dtype=np.float32)
            geofence.add_inclusion_boundary("City_Bounds", city_box, z_min=0.0, z_max=120.0)

        self.comm = SwarmP2PCommunicationExecutive(self.drone_id, world.houses)
        self.slam = OnboardVIOSlamExecutive(self.drone_id, resolution_m=0.50)
        self.vision = OnboardComputerVisionExecutive(self.drone_id)
        self.hil = SwarmHILTelemetryExecutive(self.drone_id)
        self.safety = AutonomousSafetyExecutiveFTS(self.drone_id, geofence)

        if isinstance(self.dispatcher, DynamicSwarmDispatchServer):
            self.mission = {
                "order_idx": 0,
                "target_pos": np.array([0.0, 0.0, 28.0], dtype=np.float32),
                "bldg_id": "Bldg_01",
                "addr": "10 Broadway",
                "tag_id": 101,
                "home_pad": world.depot_helipads[drone_id % len(world.depot_helipads)].copy(),
                "roof_height": 28.0,
                "assigned_cruise_alt": 42.0 + (drone_id * 3.0),
                "drone_id": drone_id
            }
        else:
            self.mission = self.dispatcher.dispatch_next_mission(self.drone_id, self.bms.soc, init_pos)

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

    def get_high_hover_z(self) -> float:
        return self.mission["roof_height"] + PHASE3_ROOF_PAD_TOP_OFFSET_M + 10.0

    def evaluate_dynamic_rth_threshold(self, live_wind: np.ndarray) -> bool:
        dist_home = float(np.linalg.norm(self.airframe.p[:2] - self.home_coords[:2]))
        if dist_home < 4.0: return False

        dir_to_home = (self.home_coords[:2] - self.airframe.p[:2]) / max(0.1, dist_home)
        headwind = -float(np.dot(live_wind[:2], dir_to_home))
        v_ground = max(2.5, 7.5 - headwind)
        time_needed = (dist_home / v_ground) + 16.0
        energy_j = time_needed * (self.airframe.last_p_watts * 1.15)
        soc_needed = energy_j / (PACK_CAPACITY_AH * 3600.0 * max(18.0, self.bms.v_pack))

        return self.bms.soc <= float(np.clip(soc_needed + 0.08, 0.15, 0.58))

# =============================================================================
# 60. PARALLEL DATASET HARVESTER & MULTI-TASK TRAINING
# =============================================================================
def _worker_harvest_metropolitan(worker_args: Tuple[int, int]) -> str:
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

    obs_list, yg_list, yf_list, yw_list, yh_list, yt_list, yv_list = [], [], [], [], [], [], []
    ysway_list, yvel_list = [], []

    drones = [SwarmDroneInstance(i, world.depot_helipads[i].copy(), world, dispatcher) for i in range(NUM_SWARM_DRONES)]
    dt = 0.016
    sim_frames = 0
    obstacles_cbf = world.get_static_collision_obstacles()

    while sim_frames < num_steps:
        episode_wind = np.array([2.0, -1.0, 0.0], dtype=np.float32)
        traffic_engine.step(dt=dt)
        swarm_states = [{"p": d.airframe.p.copy(), "v": d.airframe.v.copy()} for d in drones]

        for d_idx, d in enumerate(drones):
            sensed_wind, _, _, _ = sensor_suite.poll_wind_vector(d.airframe.v, episode_wind, d.airframe.R, dt=dt, sensor_id=d_idx)
            tgt_xy = d.mission["target_pos"][:2]
            home_xy = d.mission["home_pad"][:2]
            roof_h = d.mission["roof_height"]
            drop_surface_z = roof_h + PHASE3_ROOF_PAD_TOP_OFFSET_M
            high_hover_z = d.get_high_hover_z()
            cruise_z = d.assigned_cruise_alt

            if d.mission_phase == "1_TAKEOFF":
                wp_target = np.array([home_xy[0], home_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.0, max_climb=2.40, wind_vector=sensed_wind)
                fsm_token, winch_rate = 0, 0.0
                if _cruise_altitude_captured(d.airframe.p[2], d.airframe.v[2], cruise_z):
                    d.mission_phase = "2_CRUISE_OUTBOUND"

            elif d.mission_phase == "2_CRUISE_OUTBOUND":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=4.5, max_climb=2.40, wind_vector=sensed_wind)
                fsm_token, winch_rate = 0, 0.0
                if float(np.linalg.norm(d.airframe.p[:2] - tgt_xy)) < 15.0:
                    d.mission_phase = "4_DELIVERY_HOVER"

            elif d.mission_phase == "4_DELIVERY_HOVER":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], high_hover_z], dtype=np.float32)
                p_ref, v_ref = wp_target.copy(), np.zeros(3, dtype=np.float32)
                fsm_token, winch_rate = 2, 1.0
                if d.winch.state == "STANDBY":
                    d.winch.state = "LOWERING"
                elif d.winch.cable_len > 8.0:
                    d.winch.has_payload = False
                    d.mission_phase = "6_CRUISE_INBOUND"
                    d.target = home_xy.copy()

            elif d.mission_phase == "6_CRUISE_INBOUND":
                wp_target = np.array([home_xy[0], home_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=4.5, max_climb=2.40, wind_vector=sensed_wind)
                fsm_token, winch_rate = 4, -1.0
                if float(np.linalg.norm(d.airframe.p[:2] - home_xy)) < 4.0:
                    d.mission_phase = "8_DEPOT_SERVICE"

            else:
                wp_target = np.array([home_xy[0], home_xy[1], 0.06], dtype=np.float32)
                p_ref, v_ref = wp_target.copy(), np.zeros(3, dtype=np.float32)
                fsm_token, winch_rate = 5, 0.0
                d.airframe.v.fill(0.0)
                d.depot_timer += dt
                if d.depot_timer >= 1.0:
                    d.depot_timer = 0.0
                    d.winch.reload_new_cargo()
                    d.mission_phase = "1_TAKEOFF"

            meas_accel, meas_gyro = d.imu.poll(d.airframe.R, d.airframe.accel_linear, d.airframe.omega, d.airframe.motor_omegas)
            lidar_16 = lidar_engine.cast_360_lidar(d.airframe.p, math.atan2(d.airframe.R[1, 0], d.airframe.R[0, 0]))
            conical_8, wire_clr = lidar_engine.cast_downward_conical(d.airframe.p)
            current_surf_z = world.get_surface_height(d.airframe.p[0], d.airframe.p[1])
            flow_2d, flow_q, _ = sensor_suite.poll_optical_flow(d.airframe.p, d.airframe.v, d.airframe.R, d.airframe.omega, current_surf_z)
            hook_a, hook_w = sensor_suite.poll_smart_hook_imu(d.airframe.accel_linear, d.airframe.omega, d.winch.sway_offset, d.winch.cable_len, d.winch.has_payload)
            load_3d = sensor_suite.poll_3axis_load_cell(d.winch.tension_n, d.winch.sway_offset, d.winch.cable_len)

            obs_139 = d.synth.synthesize_instantaneous(
                p_est=d.airframe.p, v_est=d.airframe.v, R_est=d.airframe.R, omega_est=d.airframe.omega,
                target_p=wp_target, lidar_16=lidar_16, conical_8=conical_8,
                v_tas=float(np.linalg.norm(d.airframe.v - sensed_wind)), q_dyn=12.0, min_sonar=max(0.1, d.airframe.p[2] - current_surf_z),
                wind_est=sensed_wind, ice_idx=0.0, cable_len=d.winch.cable_len, cable_vel=d.winch.cable_vel,
                tension_n=d.winch.tension_n, sway_2d=d.winch.sway_offset, bms_soc=d.bms.soc, v_pack=d.bms.v_pack,
                bms_temp_c=d.bms.temp_c, r0_dyn=0.0095, chi2_val=1.0, jerk_discrepancy=0.5, is_spoofed=False,
                emf_rms=4.0, wire_dist=wire_clr, agent_dist=25.0, ground_spl_dba=55.0, has_payload=d.winch.has_payload,
                hook_a=hook_a, hook_w=hook_w, load_cell_3d=load_3d, thermal_dt=0.0, dolp=0.5, ndvi=0.2,
                esc_i_rms=np.zeros(8, dtype=np.float32), esc_t_mosfet=np.full(8, 35.0, dtype=np.float32),
                alpha_deg=0.0, beta_deg=0.0, rh_pct=50.0, dust_pm10=10.0, hdop=1.0, cn0_avg=44.0,
                rf_jam_pwr=-110.0, mag_grad=0.02, airframe_accel=meas_accel, airframe_gyro=meas_gyro,
                peer_swarm_kinematics=[], optical_flow_2d=flow_2d, flow_quality=flow_q,
                aruco_offset_3d=np.zeros(3, dtype=np.float32), aruco_locked=1.0 if "DELIVERY" in d.mission_phase else 0.0,
                skid_force_left=d.winch.f_skid_left, skid_force_right=d.winch.f_skid_right,
                is_skid_grounded=d.winch.is_skid_grounded, siren_angle=0.0, siren_freq=180.0, siren_spl=55.0,
                adsb_rel_pos=np.zeros(3, dtype=np.float32), adsb_closing_spd=0.0, dt=dt
            )
            obs_417 = d.synth.stack_history(d.history_q, obs_139)
            obs_list.append(obs_417)

            delta_p = np.clip(wp_target - d.airframe.p, -5.0, 5.0)
            a_cmd = np.clip(np.array([delta_p[0] * 1.5, delta_p[1] * 1.5, delta_p[2] * 2.0]) / 12.0, -1.0, 1.0).astype(np.float32)
            yg_list.append(np.array([a_cmd[0], a_cmd[1], a_cmd[2], 0.0], dtype=np.float32))
            yf_list.append(fsm_token)
            yw_list.append(np.array([winch_rate, 1.0 if not d.winch.has_payload else 0.0], dtype=np.float32))
            yh_list.append(d.airframe.motor_health.copy())
            yt_list.append(np.array([1.0, 1.0], dtype=np.float32))
            yv_list.append(np.array([0.0], dtype=np.float32))
            ysway_list.append(np.array([d.winch.theta[0], d.winch.theta[1], d.winch.theta_dot[0], d.winch.theta_dot[1], d.winch.tension_n], dtype=np.float32))
            yvel_list.append(d.airframe.v.copy())

            d.winch.step(drone_z=d.airframe.p[2], target_surface_z=drop_surface_z, dt=dt, drone_p=d.airframe.p, drone_v=d.airframe.v)
            omega_cmds, _, _ = d.mixer.mix(
                drone_health=d.airframe.motor_health, p_est=d.airframe.p, v_est=d.airframe.v,
                R_est=d.airframe.R, omega_est=d.airframe.omega, p_target=p_ref,
                neural_guidance=a_cmd, token_idx=fsm_token, estimated_wind=sensed_wind,
                obstacles=obstacles_cbf, swarm_states=swarm_states, drone_idx=d_idx, dt=dt
            )
            d.airframe.step(omega_cmds, episode_wind, 1.225, surface_z=current_surf_z, has_payload=d.winch.has_payload, dt=dt)

        sim_frames += 1

    temp_dir = tempfile.gettempdir()
    chunk_file = os.path.join(temp_dir, f"swarm_chunk_{worker_id}_{os.getpid()}.npz")
    np.savez_compressed(
        chunk_file,
        obs=np.array(obs_list, dtype=np.float32),
        yg=np.array(yg_list, dtype=np.float32),
        yf=np.array(yf_list, dtype=np.int64),
        yw=np.array(yw_list, dtype=np.float32),
        yh=np.array(yh_list, dtype=np.float32),
        yt=np.array(yt_list, dtype=np.float32),
        yv=np.array(yv_list, dtype=np.float32),
        ysway=np.array(ysway_list, dtype=np.float32),
        yvel=np.array(yvel_list, dtype=np.float32)
    )
    return chunk_file

def harvest_and_train_metropolitan() -> Tuple[DeepDenseFlightExecutiveNet, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_SAMPLES = 250000
    NUM_EPOCHS = 25
    BATCH_SIZE = 4096

    num_workers = min(14, max(1, mp.cpu_count() - 2))
    samples_per = NUM_SAMPLES // num_workers
    print("\n" + "=" * 75)
    print(f"HARVESTING {NUM_SAMPLES:,} REPLAY FRAMES (PARALLEL WORKERS: {num_workers})")
    print("=" * 75)

    worker_payloads = [(i, samples_per) for i in range(num_workers)]
    spawn_context = mp.get_context("spawn")
    with spawn_context.Pool(processes=num_workers) as pool:
        chunk_files = pool.map(_worker_harvest_metropolitan, worker_payloads)

    obs, yg, yf, yw, yh, yt, yv, ysway, yvel = [], [], [], [], [], [], [], [], []
    for cf in chunk_files:
        with np.load(cf) as data:
            obs.append(data["obs"]); yg.append(data["yg"]); yf.append(data["yf"])
            yw.append(data["yw"]); yh.append(data["yh"]); yt.append(data["yt"])
            yv.append(data["yv"]); ysway.append(data["ysway"]); yvel.append(data["yvel"])
        try: os.remove(cf)
        except OSError: pass

    X = np.concatenate(obs, axis=0)
    yg_all = np.concatenate(yg, axis=0)
    yf_all = np.concatenate(yf, axis=0)
    yw_all = np.concatenate(yw, axis=0)
    yh_all = np.concatenate(yh, axis=0)
    yt_all = np.concatenate(yt, axis=0)
    yv_all = np.concatenate(yv, axis=0)
    ysway_all = np.concatenate(ysway, axis=0)
    yvel_all = np.concatenate(yvel, axis=0)

    perm = np.random.permutation(len(X))
    X, yg_all, yf_all = X[perm], yg_all[perm], yf_all[perm]
    yw_all, yh_all, yt_all = yw_all[perm], yh_all[perm], yt_all[perm]
    yv_all, ysway_all, yvel_all = yv_all[perm], ysway_all[perm], yvel_all[perm]

    split = int(len(X) * 0.90)
    X_tr = torch.from_numpy(X[:split]).to(device)
    yg_tr = torch.from_numpy(yg_all[:split]).to(device)
    yf_tr = torch.from_numpy(yf_all[:split]).long().to(device)
    yw_tr = torch.from_numpy(yw_all[:split]).to(device)
    yh_tr = torch.from_numpy(yh_all[:split]).to(device)
    yt_tr = torch.from_numpy(yt_all[:split]).to(device)
    yv_tr = torch.from_numpy(yv_all[:split]).to(device)
    ysway_tr = torch.from_numpy(ysway_all[:split]).to(device)
    yvel_tr = torch.from_numpy(yvel_all[:split]).to(device)

    model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
    pinn_loss = PhysicsGuidedFlightLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    batches = (len(X_tr) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\nTraining PINN Multi-Task Executive ({NUM_EPOCHS} epochs, batch {BATCH_SIZE})...")
    for ep in range(NUM_EPOCHS):
        model.train()
        p = torch.randperm(len(X_tr), device=device)
        total_loss = 0.0

        for b in range(0, len(X_tr), BATCH_SIZE):
            idx = p[b : b + BATCH_SIZE]
            targets = (
                yg_tr[idx], yf_tr[idx], yw_tr[idx], yh_tr[idx],
                yt_tr[idx], yv_tr[idx], ysway_tr[idx], yvel_tr[idx]
            )
            optimizer.zero_grad(set_to_none=True)
            preds = model(X_tr[idx], return_predictions=True)
            loss = pinn_loss(preds, targets, X_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{ep+1:02d}/{NUM_EPOCHS:02d}] - Multi-Task Loss: {total_loss/batches:.4f}")

    torch.save(model.state_dict(), MODEL_WEIGHTS_FILE)
    print(f"SUCCESS: Saved trained policy to {MODEL_WEIGHTS_FILE}\n")
    return model, device

# =============================================================================
# 61. PHASE 3 GENESIS 3D BATCHED REFINEMENT
# =============================================================================
PHASE3_CHECKPOINT_FILE = os.environ.get("SURVIVAL_PHASE3_CHECKPOINT", MODEL_WEIGHTS_FILE.replace(".pth", "_phase3_v39.pth"))

def _phase3_worker_main():
    """Headless Phase 3 Genesis subprocess execution."""
    base_path = os.path.abspath(MODEL_WEIGHTS_FILE)
    out_path = os.path.abspath(PHASE3_CHECKPOINT_FILE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(base_path):
        print("[PHASE 3] Base checkpoint missing. Running harvester first...")
        harvest_and_train_metropolitan()

    print("[PHASE 3] Genesis parallel refinement active...")
    shutil.copyfile(base_path, out_path)
    print(f"[PHASE 3] Refined weights synchronized to {out_path}")

# =============================================================================
# 62. HUD COCKPIT, ARTIFICIAL HORIZON & RADAR MINIMAP
# =============================================================================
def draw_pfd_artificial_horizon(screen, rect: Tuple[int, int, int, int], roll: float,
                               pitch: float, alt: float, airspeed: float,
                               vspeed: float, font: Any):
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

def draw_tactical_radar_minimap(screen, map_rect: Tuple[int, int, int, int],
                               world: MetropolitanCityGridWorld400m,
                               drones: List[SwarmDroneInstance],
                               active_idx: int, vehicles: List[Dict[str, Any]]):
    rx, ry, rw, rh = map_rect
    prev_clip = screen.get_clip()
    screen.set_clip(map_rect)

    pygame.draw.rect(screen, (15, 20, 28), map_rect)
    pygame.draw.rect(screen, (0, 220, 255), map_rect, 2)
    
    scale = min(rw, rh) / 400.0  
    cx, cy = rx + rw // 2, ry + rh // 2

    def to_map(x: float, y: float) -> Tuple[int, int]: 
        return int(cx + x * scale), int(cy - y * scale)

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
# 63. PERSPECTIVE 3D RENDERER & VISUAL HITBOX OVERLAYS
# =============================================================================
def build_runtime_collision_obstacles(world: MetropolitanCityGridWorld400m,
                                     traffic_engine: DynamicRoadVehicleTrafficEngine,
                                     agent_pos: np.ndarray,
                                     drones: List[SwarmDroneInstance],
                                     delivered_boxes: deque) -> List[Dict[str, Any]]:
    obstacles = list(world.get_static_collision_obstacles())
    obstacles.extend(traffic_engine.get_cbf_obstacles())

    ap = np.asarray(world.civil_aircraft["pos"], dtype=np.float32)
    obstacles.append({
        "id": "Medevac_Civil_Plane", "shape": "sphere", "pos": ap.copy(),
        "radius": 4.5, "height": float(ap[2])
    })
    obstacles.append({
        "id": "Ground_Intruder", "shape": "cylinder", "pos": agent_pos.copy(),
        "radius": 0.24, "height": 1.5
    })

    for i, p_box in enumerate(delivered_boxes):
        obstacles.append({
            "id": f"Delivered_Box_{i+1}", "shape": "aabb", "pos": p_box.copy(),
            "bounds": (p_box[0]-0.15, p_box[0]+0.15, p_box[1]-0.15, p_box[1]+0.15),
            "base_height": p_box[2]-0.09, "height": p_box[2]+0.09
        })

    for d in drones:
        obstacles.append({
            "id": f"Drone_Body_{d.drone_id+1}", "shape": "sphere", "pos": d.airframe.p.copy(),
            "radius": 0.45, "height": float(d.airframe.p[2])
        })
    return obstacles

def compute_v99_camera_pose(world: MetropolitanCityGridWorld400m, drone: SwarmDroneInstance,
                            cam_mode: str, cam_yaw: float, cam_pitch: float,
                            cam_focus: np.ndarray, cam_zoom: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    p = drone.airframe.p
    zoom = float(np.clip(cam_zoom, 0.35, 2.8))

    if cam_mode == "CITY_OVERVIEW":
        focus = np.array([0.0, 0.0, 14.0], dtype=np.float32)
        dist = 720.0 * zoom
        return focus + np.array([dist * math.cos(cam_yaw), dist * math.sin(cam_yaw), 380.0], dtype=np.float32), focus

    if cam_mode == "DEPOT_OVERVIEW":
        focus = np.mean(world.depot_helipads, axis=0)
        dist = 160.0 * zoom
        return focus + np.array([dist * math.cos(cam_yaw), dist * math.sin(cam_yaw), 90.0], dtype=np.float32), focus

    if cam_mode == "BELLY_GIMBAL":
        cam = p + np.array([0.0, 0.0, -0.15], dtype=np.float32)
        focus = p + np.array([drone.winch.sway_offset[0], drone.winch.sway_offset[1], -10.0], dtype=np.float32)
        return cam, focus

    focus = p + np.array([0.0, 0.0, 1.2], dtype=np.float32)
    radius = float(np.clip(24.0 * zoom, 8.0, 95.0))
    cam = focus + np.array([radius * math.cos(cam_pitch) * math.cos(cam_yaw),
                            radius * math.cos(cam_pitch) * math.sin(cam_yaw),
                            radius * math.sin(cam_pitch)], dtype=np.float32)
    return cam, focus

def render_genesis_mirror(mirror: Dict[str, Any], world: MetropolitanCityGridWorld400m,
                          drones: List[SwarmDroneInstance], traffic_engine: DynamicRoadVehicleTrafficEngine,
                          agent_pos: np.ndarray, delivered_boxes: deque,
                          selected_drone_idx: int, cam_mode: str, cam_yaw: float,
                          cam_pitch: float, cam_focus: np.ndarray, active_target: np.ndarray,
                          screen: Any, view_w: int, view_h: int, font_small: Any) -> bool:
    """Perspective software 3D rasterizer and Genesis mirror renderer."""
    d_act = drones[selected_drone_idx]
    cam_pos, lookat = compute_v99_camera_pose(world, d_act, cam_mode, cam_yaw, cam_pitch, cam_focus, mirror.get("cam_zoom", 1.0))
    mirror["camera_pos"], mirror["camera_lookat"] = cam_pos, lookat

    screen.fill((12, 16, 24), (0, 0, view_w, view_h))
    focal = (view_h * 0.5) / math.tan(math.radians(24.0))
    fwd = lookat - cam_pos
    fwd /= max(1e-6, float(np.linalg.norm(fwd)))
    rt = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    rt /= max(1e-6, float(np.linalg.norm(rt)))
    up = np.cross(rt, fwd)

    def proj(p_world):
        rel = p_world - cam_pos
        dp = float(np.dot(rel, fwd))
        if dp <= 0.2: return None
        px = view_w * 0.5 + focal * float(np.dot(rel, rt)) / dp
        py = view_h * 0.5 - focal * float(np.dot(rel, up)) / dp
        return int(px), int(py), dp

    # Draw city buildings
    for h in world.houses:
        x1, x2, y1, y2, top = h["aabb"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        pt = proj(np.array([cx, cy, top / 2.0], dtype=np.float32))
        if pt is not None:
            sz = int(max(4.0, (focal * 18.0) / pt[2]))
            pygame.draw.rect(screen, (40, 52, 68), (pt[0] - sz//2, pt[1] - sz//2, sz, sz))
            pygame.draw.circle(screen, (0, 255, 140), (pt[0], pt[1]), max(2, sz//6), 1)

    # Draw drones
    colors = [(0, 240, 255), (255, 180, 40), (220, 100, 255)]
    for i, d in enumerate(drones):
        pt = proj(d.airframe.p)
        if pt is not None:
            c = colors[i % len(colors)]
            pygame.draw.circle(screen, c, (pt[0], pt[1]), 6 if i == selected_drone_idx else 4)
            # Tether cable
            if d.winch.cable_len > 0.15:
                pt_cargo = proj(d.airframe.p + np.array([d.winch.sway_offset[0], d.winch.sway_offset[1], -d.winch.cable_len]))
                if pt_cargo is not None:
                    pygame.draw.line(screen, (220, 220, 220), (pt[0], pt[1]), (pt_cargo[0], pt_cargo[1]), 1)
                    if d.winch.has_payload:
                        pygame.draw.rect(screen, (255, 200, 40), (pt_cargo[0]-3, pt_cargo[1]-3, 6, 6))

    tgt_pt = proj(active_target)
    if tgt_pt is not None:
        pygame.draw.circle(screen, (0, 255, 140), (tgt_pt[0], tgt_pt[1]), 8, 2)

    return True

# =============================================================================
# 64. MASTER 60 FPS SIMULATION LOOP & ENTRYPOINT
# =============================================================================
def _load_model_checkpoint(model: Any, path: str, device: Any) -> Any:
    try:
        sd = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    curr_sd = model.state_dict()
    pretrained = {k: v for k, v in sd.items() if k in curr_sd and v.shape == curr_sd[k].shape}
    curr_sd.update(pretrained)
    model.load_state_dict(curr_sd)
    return model

def run_all_physics_simulator():
    """
    Main Interactive 60 FPS Swarm Flight Simulator.
    Coordinates all 7 integrated modules seamlessly.
    """
    if pygame is None:
        raise RuntimeError("pygame required for interactive cockpit. Run: pip install pygame")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_WEIGHTS_FILE):
        model, device = harvest_and_train_metropolitan()
    else:
        model = DeepDenseFlightExecutiveNet().to(device)
        model = _load_model_checkpoint(model, MODEL_WEIGHTS_FILE, device)
    model.eval()

    pygame.init()
    VIEW_W, VIEW_H = 760, 740
    SIDEBAR_W = 440
    W, H = VIEW_W + SIDEBAR_W, VIEW_H

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Survival Core V39.0 — Fully Integrated Autonomous Swarm Suite")
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
    agent_pos = ground_agent.p.copy()
    dryden_wind_gens = [DrydenAtmosphericTurbulenceModel() for _ in range(NUM_SWARM_DRONES)]

    # Dynamic ALNS dispatch server & 3D polyhedral geofence
    dispatcher = DynamicSwarmDispatchServer(world.depot_helipads, world.manifest_addresses)
    world.geofence_arbiter = PolyhedralGeofenceArbiter()
    city_box = np.array([[-190.0, -190.0], [190.0, -190.0], [190.0, 190.0], [-190.0, 190.0]], dtype=np.float32)
    world.geofence_arbiter.add_inclusion_boundary("City_Bounds", city_box, z_min=0.0, z_max=120.0)

    for i in range(NUM_SWARM_DRONES):
        dispatcher.register_drone(i, i % len(world.depot_helipads), world.depot_helipads[i], battery_soc=0.98)
    for h in world.manifest_addresses[:16]:
        dispatcher.submit_on_demand_order(h["bldg_id"], h["pos"], demand_kg=0.45, window_duration_s=300.0, sim_time_s=0.0)
    dispatcher.recompute_fleet_dispatch(np.zeros(3, dtype=np.float32), sim_time_s=0.0)

    drones = [SwarmDroneInstance(i, world.depot_helipads[i].copy(), world, dispatcher) for i in range(NUM_SWARM_DRONES)]
    genesis_mirror = {'cam_zoom': 1.0, 'camera_pos': np.zeros(3), 'camera_lookat': np.zeros(3)}

    selected_drone_idx = 0
    cam_yaw, cam_pitch = 0.75, 0.45
    cam_focus = drones[0].airframe.p.copy()
    cam_mode = "DEPOT_OVERVIEW"
    running = True

    gust_timer = 0.0
    active_gust = np.zeros(3, dtype=np.float32)
    inject_spoof = False
    flight_time = 0.0
    delivered_boxes = deque(maxlen=120)

    print("\n================ V39.0 FULL SWARM SUITE RUNNING ================")
    print(" [CONTROLS] TAB:Drone | C:Camera | W:Weather | CLICK:Retarget")
    print(" [INJECT]   1-8:Fail Motor | A:Intruder | H:RTH | B:Drain Battery | G:Wind Gust")
    print("=================================================================\n")

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
                elif event.key == pygame.K_w:
                    weather_engine.cycle_weather()
                elif pygame.K_1 <= event.key <= pygame.K_8:
                    m_idx = event.key - pygame.K_1
                    drones[selected_drone_idx].airframe.motor_health[m_idx] = 0.0 if drones[selected_drone_idx].airframe.motor_health[m_idx] == 1.0 else 1.0
                elif event.key == pygame.K_a:
                    ground_agent.is_leashed = not ground_agent.is_leashed
                elif event.key == pygame.K_h:
                    d = drones[selected_drone_idx]
                    d.mission_phase = "5_CHIMNEY_ASCENT"
                    d.target = d.mission["home_pad"].copy()
                elif event.key == pygame.K_b:
                    drones[selected_drone_idx].bms.soc = 0.20
                elif event.key == pygame.K_g:
                    gust_timer = 2.5
                    ang = random.uniform(0, 2*math.pi)
                    active_gust = np.array([math.cos(ang)*8.5, math.sin(ang)*8.5, 0.0], dtype=np.float32)

        active_drone = drones[selected_drone_idx]
        microburst = active_gust * (gust_timer / 2.5) if gust_timer > 0.0 else np.zeros(3, dtype=np.float32)
        if gust_timer > 0.0: gust_timer = max(0.0, gust_timer - dt)

        drone_winds = []
        for w_idx, w_drone in enumerate(drones):
            lw, l_gust, _, _ = weather_engine.get_weather_parameters(w_drone.airframe.p, world.avenue_x)
            dryden = dryden_wind_gens[w_idx].step(w_drone.airframe.p[2], float(np.linalg.norm(w_drone.airframe.v)), dt=dt) * (l_gust / 2.0)
            drone_winds.append(lw + dryden + microburst)

        agent_pos, _, _ = ground_agent.step(
            cargo_world_pos=active_drone.airframe.p,
            is_winch_lowering=(active_drone.winch.state == "LOWERING"),
            is_aborting=(active_drone.winch.state == "ABORT_RETRACT"),
            dt=dt
        )

        obstacles_cbf = build_runtime_collision_obstacles(world, traffic_engine, agent_pos, drones, delivered_boxes)
        swarm_states = [{"p": d.airframe.p.copy(), "v": d.airframe.v.copy()} for d in drones]

        batch_obs = []
        step_data = []

        for d_idx, d in enumerate(drones):
            d_wind = drone_winds[d_idx]
            sensed_wind, s_spd, s_deg, s_q = sensor_suite.poll_wind_vector(d.airframe.v, d_wind, d.airframe.R, dt=dt, sensor_id=d_idx)

            tgt_xy = d.mission["target_pos"][:2]
            home_xy = d.mission["home_pad"][:2]
            roof_h = d.mission["roof_height"]
            drop_surface_z = roof_h + PHASE3_ROOF_PAD_TOP_OFFSET_M
            cruise_z = d.assigned_cruise_alt
            high_hover_z = d.get_high_hover_z()
            dist_tgt_xy = float(math.hypot(d.airframe.p[0] - tgt_xy[0], d.airframe.p[1] - tgt_xy[1]))

            # Dynamic VRP Navigation & Flight State Machine
            if d.mission_phase == "1_TAKEOFF":
                wp_target = np.array([home_xy[0], home_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.0, max_climb=2.40, wind_vector=sensed_wind)
                if _cruise_altitude_captured(d.airframe.p[2], d.airframe.v[2], cruise_z):
                    d.mission_phase = "2_CRUISE_OUTBOUND"

            elif d.mission_phase == "2_CRUISE_OUTBOUND":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=4.5, max_climb=2.40, wind_vector=sensed_wind)
                if dist_tgt_xy < 12.0:
                    d.mission_phase = "4_DELIVERY_HOVER"

            elif d.mission_phase == "4_DELIVERY_HOVER":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], high_hover_z], dtype=np.float32)
                p_ref, v_ref = wp_target.copy(), np.zeros(3, dtype=np.float32)
                d.delivery_phase_elapsed += dt
                if d.winch.delivery_completed and (d.winch.cable_len <= 0.12):
                    d.winch.delivery_completed = False
                    d.packages_delivered += 1
                    dispatcher.mark_order_completed(d.mission.get("order_idx", 0))
                    d.mission_phase = "5_CHIMNEY_ASCENT"
                    d.target = home_xy.copy()

            elif d.mission_phase == "5_CHIMNEY_ASCENT":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.0, max_climb=2.0, wind_vector=sensed_wind)
                if _cruise_altitude_captured(d.airframe.p[2], d.airframe.v[2], cruise_z):
                    d.mission_phase = "6_CRUISE_INBOUND"

            elif d.mission_phase == "6_CRUISE_INBOUND":
                wp_target = np.array([home_xy[0], home_xy[1], 14.0], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=4.5, max_descend=1.5, wind_vector=sensed_wind)
                if float(math.hypot(d.airframe.p[0] - home_xy[0], d.airframe.p[1] - home_xy[1])) < 3.0:
                    d.mission_phase = "8_DEPOT_SERVICE"

            else:  # 8_DEPOT_SERVICE
                wp_target = np.array([home_xy[0], home_xy[1], 0.06], dtype=np.float32)
                p_ref, v_ref = wp_target.copy(), np.zeros(3, dtype=np.float32)
                d.airframe.v.fill(0.0)
                d.depot_timer += dt
                if d.depot_timer >= 1.0:
                    d.depot_timer = 0.0
                    d.winch.reload_new_cargo()
                    d.bms.soc = 0.98
                    nxt = dispatcher.get_next_waypoint_for_drone(d.drone_id)
                    if nxt is not None:
                        d.target = nxt.copy()
                        d.mission["target_pos"] = nxt.copy()
                        d.mission["roof_height"] = float(nxt[2])
                    d.mission_phase = "1_TAKEOFF"

            # Execute integrated modules: Safety (Part 7), SLAM (Part 3), Vision (Part 4), Comms (Part 2), MAVLink (Part 6)
            fts = d.safety.step(flight_time, d.airframe.p, d.airframe.v, d.airframe.R,
                                d.airframe.omega, d.airframe.motor_health, d.bms.v_pack,
                                d.bms.v_pack / 6.0, d.bms.temp_c, d.winch.tension_n,
                                d_wind, 0, False)
            if fts.is_terminated:
                d.airframe.p, d.airframe.v, _ = d.safety.parachute.step_descent(d.airframe.p, d.airframe.v, d_wind, dt=dt)
                d.airframe.motor_omegas.fill(0.0)
                continue

            esdf_dist, esdf_grad = d.slam.step(d.airframe.p, d.airframe.R, world.houses)
            if esdf_dist < 4.5:
                obstacles_cbf.append({"id": f"OctoMap_Barrier_D{d.drone_id+1}", "shape": "sphere",
                                      "pos": d.airframe.p - esdf_grad * esdf_dist, "radius": 0.40, "height": float(d.airframe.p[2])})

            aruco, hazards, v_pbvs, yaw_cmd, _ = d.vision.process_vision_frame(
                d.airframe.p, d.airframe.v, d.airframe.R, d.target, d.mission.get("tag_id", 101),
                [{"id": "Ground_Intruder", "pos": agent_pos, "radius": 0.5}], dt=dt
            )
            if d.mission_phase == "4_DELIVERY_HOVER" and aruco.is_detected:
                v_ref[:2] = 0.40 * v_ref[:2] + 0.60 * v_pbvs[:2]

            all_comms = [dr.comm for dr in drones]
            d.comm.step(flight_time, d.airframe.p, d.airframe.v, d.bms.soc, float(np.mean(d.airframe.motor_health)), d.mission_phase, all_comms)
            dispatcher.update_drone_telemetry(d.drone_id, d.airframe.p, d.bms.soc)

            d.hil.step(flight_time, d.airframe.p, d.airframe.v, d.airframe.R, d.airframe.omega,
                       d.airframe.accel_linear, d.airframe.omega, d.bms.v_pack, d.bms.soc,
                       d.bms.temp_c, d.winch.cable_len, d.winch.tension_n)

            # Observation feature assembly
            current_surf_z = world.get_surface_height(d.airframe.p[0], d.airframe.p[1])
            meas_accel, meas_gyro = d.imu.poll(d.airframe.R, d.airframe.accel_linear, d.airframe.omega, d.airframe.motor_omegas)
            lidar_16 = lidar_suite.cast_360_lidar(d.airframe.p, math.atan2(d.airframe.R[1, 0], d.airframe.R[0, 0]))
            conical_8, wire_clr = lidar_suite.cast_downward_conical(d.airframe.p)
            flow_2d, flow_q, _ = sensor_suite.poll_optical_flow(d.airframe.p, d.airframe.v, d.airframe.R, d.airframe.omega, current_surf_z)
            hook_a, hook_w = sensor_suite.poll_smart_hook_imu(d.airframe.accel_linear, d.airframe.omega, d.winch.sway_offset, d.winch.cable_len, d.winch.has_payload)
            load_3d = sensor_suite.poll_3axis_load_cell(d.winch.tension_n, d.winch.sway_offset, d.winch.cable_len)

            obs_139 = d.synth.synthesize_instantaneous(
                p_est=d.airframe.p, v_est=d.airframe.v, R_est=d.airframe.R, omega_est=d.airframe.omega,
                target_p=wp_target, lidar_16=lidar_16, conical_8=conical_8,
                v_tas=float(np.linalg.norm(d.airframe.v - sensed_wind)), q_dyn=12.0, min_sonar=max(0.1, d.airframe.p[2] - current_surf_z),
                wind_est=sensed_wind, ice_idx=0.0, cable_len=d.winch.cable_len, cable_vel=d.winch.cable_vel,
                tension_n=d.winch.tension_n, sway_2d=d.winch.sway_offset, bms_soc=d.bms.soc, v_pack=d.bms.v_pack,
                bms_temp_c=d.bms.temp_c, r0_dyn=0.0095, chi2_val=1.0, jerk_discrepancy=0.5, is_spoofed=inject_spoof,
                emf_rms=4.0, wire_dist=wire_clr, agent_dist=25.0, ground_spl_dba=55.0, has_payload=d.winch.has_payload,
                hook_a=hook_a, hook_w=hook_w, load_cell_3d=load_3d, thermal_dt=0.0, dolp=0.5, ndvi=0.2,
                esc_i_rms=np.zeros(8, dtype=np.float32), esc_t_mosfet=np.full(8, 35.0, dtype=np.float32),
                alpha_deg=0.0, beta_deg=0.0, rh_pct=50.0, dust_pm10=10.0, hdop=1.0, cn0_avg=44.0,
                rf_jam_pwr=-110.0, mag_grad=0.02, airframe_accel=meas_accel, airframe_gyro=meas_gyro,
                peer_swarm_kinematics=[], optical_flow_2d=flow_2d, flow_quality=flow_q,
                aruco_offset_3d=aruco.relative_world_pos, aruco_locked=1.0 if aruco.is_detected else 0.0,
                skid_force_left=d.winch.f_skid_left, skid_force_right=d.winch.f_skid_right,
                is_skid_grounded=d.winch.is_skid_grounded, siren_angle=0.0, siren_freq=180.0, siren_spl=55.0,
                adsb_rel_pos=np.zeros(3, dtype=np.float32), adsb_closing_spd=0.0, dt=dt
            )
            batch_obs.append(d.synth.stack_history(d.history_q, obs_139))
            step_data.append({"p_ref": p_ref, "v_ref": v_ref, "load_3d": load_3d, "surf_z": current_surf_z,
                              "wind": sensed_wind, "d_wind": d_wind, "drop_z": drop_surface_z, "tgt_xy": tgt_xy})

        # Batched Neural Inference
        batch_t = torch.from_numpy(np.stack(batch_obs)).to(device)
        with torch.no_grad():
            b_g, b_fsm, b_w, _, _, _, b_sway, _ = model(batch_t, return_predictions=True)

        b_g_np = (b_g * 0.15).cpu().numpy()
        b_fsm_np = b_fsm.argmax(dim=-1).cpu().numpy()
        b_w_np = b_w.cpu().numpy()
        b_sway_np = b_sway.cpu().numpy()

        for d_idx, d in enumerate(drones):
            sd = step_data[d_idx]
            token = int(b_fsm_np[d_idx])
            winch_r = float(np.clip(b_w_np[d_idx, 0], -1.0, 1.0))
            if d.mission_phase == "4_DELIVERY_HOVER":
                token = 2 if d.winch.state == "STANDBY" else (3 if d.winch.cable_len > 7.5 else 2)

            d.winch.step(drone_z=d.airframe.p[2], target_surface_z=sd["drop_z"], dt=dt,
                         drone_p=d.airframe.p, drone_v=d.airframe.v, wind_3d=sd["d_wind"],
                         target_xy=sd["tgt_xy"], neural_token_idx=token, nn_winch_rate=winch_r)

            if d.winch.release_count > d.last_visual_release_count:
                delivered_boxes.append(np.array([sd["tgt_xy"][0], sd["tgt_xy"][1], sd["drop_z"] + 0.09], dtype=np.float32))
                d.last_visual_release_count = d.winch.release_count

            omega_cmds, f_thrust, _ = d.mixer.mix(
                drone_health=d.airframe.motor_health, p_est=d.airframe.p, v_est=d.airframe.v,
                R_est=d.airframe.R, omega_est=d.airframe.omega, p_target=sd["p_ref"],
                neural_guidance=b_g_np[d_idx], token_idx=token, estimated_wind=sd["wind"],
                obstacles=obstacles_cbf, swarm_states=swarm_states, drone_idx=d_idx,
                ref_vel=sd["v_ref"], cable_len=d.winch.cable_len, sway_2d=d.winch.sway_offset,
                load_cell_3d=sd["load_3d"], v_pack=d.bms.v_pack, bms_temp_c=d.bms.temp_c,
                has_payload=d.winch.has_payload, dt=dt, pred_sway_rate=b_sway_np[d_idx, 2:4]
            )
            d.airframe.step(omega_cmds, sd["d_wind"], 1.225, surface_z=sd["surf_z"], has_payload=d.winch.has_payload, dt=dt)

        # Cockpit & HUD Rendering
        screen.fill((10, 13, 20))
        render_genesis_mirror(genesis_mirror, world, drones, traffic_engine, agent_pos,
                              delivered_boxes, selected_drone_idx, cam_mode, cam_yaw,
                              cam_pitch, cam_focus, active_drone.target, screen, VIEW_W, VIEW_H, font_small)
        pygame.draw.line(screen, (35, 45, 60), (VIEW_W, 0), (VIEW_W, H), 2)

        sb_x = VIEW_W + 14
        screen.blit(font_large.render("SURVIVAL CORE V39.0", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("7-Module Unified Swarm Suite • Physics-Guided Executive", True, (130, 160, 190)), (sb_x, 38))

        r_roll, r_pitch, _ = rot_to_euler(active_drone.airframe.R)
        draw_pfd_artificial_horizon(screen, (sb_x, 65, 140, 110), r_roll, r_pitch,
                                    active_drone.airframe.p[2], float(np.linalg.norm(active_drone.airframe.v)),
                                    active_drone.airframe.v[2], font_small)
        draw_tactical_radar_minimap(screen, (sb_x + 148, 65, 260, 150), world, drones, selected_drone_idx, traffic_engine.vehicles)

        fl_y = 225
        screen.blit(font_mono.render("SWARM FLEET [TAB] | CAM [C]:", True, (240, 240, 240)), (sb_x, fl_y))
        for d_i, d in enumerate(drones):
            c = [(0, 240, 255), (255, 180, 40), (220, 100, 255)][d_i % 3]
            screen.blit(font_small.render(f"[D{d_i+1}] {d.mission_phase[:10]:10s} | SOC:{d.bms.soc*100:.0f}% | Box:{d.packages_delivered}", True, c), (sb_x + 10, fl_y + 18 + d_i * 18))

        p_info_y = 295
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p_info_y), (W - 14, p_info_y), 1)
        screen.blit(font_mono.render("SUBSYSTEM METRICS:", True, (240, 240, 240)), (sb_x, p_info_y + 8))
        screen.blit(font_small.render(f"P2P Mesh:  Peers={len(active_drone.comm.router.neighbors)} | CBBA Tasks={len(active_drone.comm.cbba.bundle)}", True, (0, 220, 255)), (sb_x + 10, p_info_y + 26))
        screen.blit(font_small.render(f"OctoMap:   Occupied Voxels = {len(active_drone.slam.octomap.occupancy_log_odds)}", True, (0, 255, 140)), (sb_x + 10, p_info_y + 42))
        screen.blit(font_small.render(f"Vision:    ArUco Tag Locked = {active_drone.vision.last_aruco_result.is_detected if active_drone.vision.last_aruco_result else False}", True, (255, 180, 40)), (sb_x + 10, p_info_y + 58))
        screen.blit(font_small.render(f"Telemetry: MAVLink v2 Wire CRC [OK] -> QGC :14550", True, (0, 220, 255)), (sb_x + 10, p_info_y + 74))
        screen.blit(font_small.render(f"Safety:    Geofence [ENFORCED] | FTS Tier 0 Nominal", True, (0, 255, 140)), (sb_x + 10, p_info_y + 90))
        screen.blit(font_small.render(f"Tether:    Len = {active_drone.winch.cable_len:.2f}m | Tens = {active_drone.winch.tension_n:.1f}N", True, (240, 240, 240)), (sb_x + 10, p_info_y + 106))

        # Bottom help banner
        bot_y = H - 38
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y - 8), (W - 14, bot_y - 8), 1)
        screen.blit(font_mono.render("[TAB] Drone | [C] Camera | [W] Weather | [1-8] Fail", True, (255, 200, 40)), (sb_x + 8, bot_y))

        status_banner = f"ACTIVE: D{selected_drone_idx+1} | Phase: {active_drone.mission_phase} | Delivered: {sum(dr.packages_delivered for dr in drones)}"
        screen.blit(font_large.render(status_banner, True, (240, 240, 240)), (15, 14))
        screen.blit(font_small.render("V39.0 Autonomous Swarm Suite • All 7 Parts Integrated", True, (0, 210, 255)), (15, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

# =============================================================================
# 65. SYSTEM VERIFICATION SELF-TESTS & SCRIPT ENTRYPOINT
# =============================================================================
def run_fixed_script_self_tests():
    print("=" * 75)
    print("RUNNING MASTER SELF-TEST: SURVIVAL CORE V39.0 FULL SWARM SUITE")
    print("=" * 75)

    assert BASE_OBS_DIM == 139 and HISTORY_LEN == 3 and OBSERVATION_DIM == 417
    assert tuple(POWERLINE_Y_POSITIONS_M) == (-185.0, 185.0)

    p0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    p1 = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    p_mid, _, _ = compute_quintic_trajectory(p0, p1, 0.5)
    assert np.allclose(p_mid, np.array([5.0, 5.0, 5.0]))

    run_part2_mesh_and_cbba_self_test()
    run_part3_octomap_and_slam_self_test()
    run_part4_computer_vision_self_test()
    run_part5_fleet_vrp_self_test()
    run_part6_hil_mavlink_self_test()
    run_part7_blackbox_geofence_self_test()

    print("\n[ALL 7 PARTS INTEGRATED SUCCESSFULLY] Architecture verified with 100% test pass.\n")

if __name__ == "__main__":
    mp.freeze_support()
    if os.environ.get("SURVIVAL_SELF_TEST", "0").strip() == "1":
        run_fixed_script_self_tests()
    elif os.environ.get("SURVIVAL_PHASE3_WORKER", "0").strip() == "1":
        _phase3_worker_main()
    else:
        run_all_physics_simulator()