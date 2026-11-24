"""
=============================================================================
SURVIVAL CORE V38.42 — TRANSITION CONTRACT / BASELINE PARITY AUDIT
=============================================================================
V38.42 fixes the live constant-velocity baseline to use the same observation-derived velocity as offline validation, adds train/holdout/live action-input distribution diagnostics, and exposes phase-gated guidance. Predictive control remains unauthorized.
"""

SURVIVAL_CORE_VERSION = "V38.42-transition-contract-and-baseline-parity-audit"

# V38.40 adds opt-in, passive live validation: it predicts the one-step state delta
# from the already-selected high-level action, then compares that prediction with the
# authoritative post-step BEMT/RK4 state. It cannot try alternate actions or change
# actuation, and it is not multi-step MPC validation.

# Runtime-stability notes:
# - V38.28 adds prediction uncertainty estimation; outputs are diagnostic-only and cannot control flight.
# - V38.31 adds a paired-action one-step transition model; action-conditioned predictions remain offline/diagnostic only.
# - V38.32 compares held-out one-step transition RMSE against a constant-velocity baseline.
# - V38.34 breaks transition errors down by commanded FSM action, winch command, and sensed-wind bin, and probes action-input sensitivity.
# - V38.34 reports a conservative readiness gate; V38.35 adds held-out, teacher-forced accumulated multi-step transition evaluation.
# - V38.37 recursively updates estimated wind, cable-length/velocity, icing, cargo sway/rate/tension and their explicitly reconstructed sensor proxies, but still is not a complete sensor/world simulator or live MPC.
# - V38.30 adds a strictly offline finite-candidate predictive-control evaluator using an analytic surrogate; no live-control wiring.
# - V38.22 stabilised capture entry; V38.23 adds explicit weather-triggered depot recapture to a safe staging height.
# - Set SURVIVAL_PHASE3_TRAIN_DEVICE=cuda only to opt back into CUDA fine-tuning.
# - A harvest vertical speed above 8 m/s or altitude above 80 m is quarantined.
#   These are dataset guards, not proof that the underlying controller has been fixed.

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
import json
from collections import deque
from pathlib import Path
import multiprocessing as mp
import tempfile
import shutil
import subprocess
import numpy as np


def _configured_genesis_logging_level():
    """Genesis log level; default to warning so per-entity INFO logs cannot flood the terminal."""
    level = os.environ.get("SURVIVAL_GENESIS_LOG_LEVEL", "warning").strip().lower()
    allowed = {"debug", "info", "warning", "error", "critical"}
    return level if level in allowed else "warning"


def _capture_per_drone_mission_context(mission, current_position):
    """Copy mission coordinates and distances so separate control passes cannot share loop locals."""
    target = np.asarray(mission["target_pos"], dtype=np.float32).reshape(-1)
    home = np.asarray(mission["home_pad"], dtype=np.float32).reshape(-1)
    position = np.asarray(current_position, dtype=np.float32).reshape(-1)
    if target.size < 2 or home.size < 2 or position.size < 2:
        raise ValueError("Mission target, home pad, and current position must contain XY coordinates")
    if not (np.all(np.isfinite(target[:2])) and np.all(np.isfinite(home[:2]))
            and np.all(np.isfinite(position[:2]))):
        raise ValueError("Mission context coordinates must be finite")
    target_xy = target[:2].copy()
    home_xy = home[:2].copy()
    return {
        "mission_target_xy": target_xy,
        "mission_home_xy": home_xy,
        "mission_roof_height": float(mission["roof_height"]),
        "mission_dist_target_xy": float(np.linalg.norm(position[:2] - target_xy)),
        "mission_dist_home_xy": float(np.linalg.norm(position[:2] - home_xy)),
    }


# Windows multiprocessing uses spawn. Without this guard, every harvesting
# child re-imports this entire file and loads PyTorch/CUDA DLLs before it runs
# the CPU-only world-generation worker. With 14 workers this can exhaust the
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
    # Headless self-tests and training workers can still import the module.
    # The interactive simulator reports the missing dependency at launch.
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
# training observations on the same powerline coordinates.  Wires are finite
# segments spanning x=[-190, 190] m at y=+/-185 m and z=12 m.
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
    "EMERGENCY_DIVERT"     # 6 (reserved; not currently represented in teacher labels)
]

# Delivery-policy supervision and evaluation safeguards. The learned policy may
# request an action, but the physical proximity/surface checks remain authoritative.
DELIVERY_HOVER_TIMEOUT_S = 45.0
DELIVERY_MAX_ABORT_RETRIES = 1  # one retry after the first safely retracted aborted drop
DELIVERY_ABORT_RECOVERY_S = 2.0
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
DELIVERY_MAX_OPERATIONAL_WIND_MS = 7.0  # hard active-cycle boundary; above this, fail safe
DELIVERY_MAX_DEPLOYMENT_WIND_MS = 4.5   # do not start lowering in gusty conditions
DELIVERY_PAYOUT_WIND_PAUSE_MS = 4.8     # latch a payout hold at/above this wind
DELIVERY_PAYOUT_WIND_RESUME_MS = 4.0    # hysteresis: resume only after a real lull
CHIMNEY_APPROACH_TIMEOUT_S = 45.0  # includes cruise alignment and near-hover settling
# Depot landing is split into lateral capture/settling followed by a vertical descent.
# The craft may not command descent until it has held the pad centre and low velocity
# continuously; lateral drift during the descent pauses it above the pad for recapture.
DEPOT_LANDING_CAPTURE_RADIUS_M = 0.50
DEPOT_LANDING_RECAPTURE_RADIUS_M = 0.80
DEPOT_LANDING_MAX_XY_SPEED_MS = 0.22
DEPOT_LANDING_MAX_Z_SPEED_MS = 0.22
DEPOT_LANDING_SETTLE_DWELL_S = 0.60
DEPOT_LANDING_DESCEND_SPEED_MS = 0.42
DEPOT_LANDING_FINAL_DESCEND_SPEED_MS = 0.12
DEPOT_LANDING_PAUSE_HEIGHT_M = 0.35
DEPOT_LANDING_DESCENT_HOLD_XY_SPEED_MS = 0.30
DEPOT_LANDING_DESCENT_HOLD_XY_ERROR_M = 0.50
# Gust policy: pause vertical landing above 5.5 m/s estimated horizontal wind;
# a descent that is actively armed is disarmed if wind exceeds 6.5 m/s.
# The aircraft continues active horizontal/vertical hold while waiting for a lull.
DEPOT_LANDING_WIND_RESUME_MS = 5.5
DEPOT_LANDING_WIND_HARD_HOLD_MS = 6.5
# Vertical gusts can destabilise a pad landing even when horizontal wind is modest.
# These are conservative landing interlocks, not limits on normal flight.
DEPOT_LANDING_VERTICAL_WIND_PAUSE_MS = 1.25
DEPOT_LANDING_VERTICAL_WIND_HARD_HOLD_MS = 2.00
# Do not switch from inbound flight to final landing while carrying large momentum.
DEPOT_CAPTURE_STAGING_ALTITUDE_M = 24.0
DEPOT_CAPTURE_ALTITUDE_TOLERANCE_M = 0.70
DEPOT_CAPTURE_MAX_XY_SPEED_MS = 0.85
DEPOT_CAPTURE_MAX_VERTICAL_SPEED_MS = 0.65
# Use the staging position target when the craft has drifted far from its latched height.
DEPOT_STAGING_POSITION_FEEDBACK_THRESHOLD_M = 0.50
# In the final metre, tolerate small controlled lateral motion so tiny sensor/wind
# fluctuations do not keep an otherwise aligned aircraft suspended indefinitely.
DEPOT_LANDING_LOW_ALT_HOLD_HEIGHT_M = 1.50
DEPOT_LANDING_LOW_ALT_HOLD_XY_ERROR_M = 0.65
DEPOT_LANDING_LOW_ALT_HOLD_XY_SPEED_MS = 0.45
# Start the final pad-capture controller well before the pad. The previous 3 m
# transition was too late: compute_vector_guidance was still requesting ~3 m/s
# laterally at 3 m range, leaving insufficient margin for a vertical landing.
DEPOT_APPROACH_CAPTURE_RADIUS_M = 24.0
DEPOT_APPROACH_BRAKE_MARGIN_M = 24.0
DEPOT_APPROACH_BRAKE_ACCEL_MS2 = 1.30
DEPOT_ACTIVE_STATION_DEPARTURE_M = 2.0
DEPOT_ACTIVE_STATION_SPEED_MS = 1.25
DELIVERY_ACTIVE_DEPARTURE_VERTICAL_MS = 0.90
# Phase-3 Genesis geometry constants. Cargo/airframe positions are rigid-body
# centres, while the winch cable length runs between the drone underside and
# the cargo top. The training physics must account for both half-heights.
PHASE3_ROOF_PAD_TOP_OFFSET_M = 0.08   # rooftop pad cylinder top above building top
PHASE3_DRONE_HALF_HEIGHT_M = 0.045    # Phase-3 URDF body is 0.09 m high
PHASE3_CARGO_HALF_HEIGHT_M = 0.09     # payload box is 0.18 m high
PHASE3_TETHER_CENTER_OFFSET_M = PHASE3_DRONE_HALF_HEIGHT_M + PHASE3_CARGO_HALF_HEIGHT_M
PHASE3_STOWED_CABLE_LEN_M = 0.08
PHASE3_DEFAULT_ROLLOUT_STEPS = 3072  # 30.72 simulated seconds at dt=0.01s
# A parcel must be tightly centred to release, but while attached it needs room
# to swing and recover. Do not confuse the strict release radius with the wider
# physical safety envelope of the 2.2 m rooftop pad.
PHASE3_CARGO_SAFE_RADIUS_M = 1.80
PHASE3_CARGO_ZONE_ABORT_DWELL_S = 0.60
PHASE3_SWAY_POSITION_GAIN = 2.0
PHASE3_SWAY_VELOCITY_GAIN = 1.55
POLICY_MIN_ACTION_CONFIDENCE = 0.38
POLICY_MIN_SURFACE_TRUST = 0.25
# Learned guidance is a residual on top of deterministic closed-loop tracking.
# Bound its vertical contribution so an immature policy cannot overpower the
# altitude/vertical-velocity controller with a multi-m/s^2 climb or descent command.
POLICY_VERTICAL_ACCEL_RESIDUAL_LIMIT_MS2 = 0.80


def _anti_sway_feedback(sway_offset, sway_velocity, position_gain=PHASE3_SWAY_POSITION_GAIN,
                        velocity_gain=PHASE3_SWAY_VELOCITY_GAIN):
    """Command the attachment point in the direction of relative sway/rate.

    `sway_offset = cargo_xy - drone_xy` and `sway_velocity = cargo_v_xy - drone_v_xy`.
    Moving the attachment point with the relative displacement/rate reduces relative
    motion; subtracting these terms, as older versions did, can reinforce the swing.
    Works with NumPy arrays and Torch tensors.
    """
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


def _delivery_winch_continuation_authorized(mission_phase, winch_state, has_payload, wind_speed_xy_m_s):
    """Keep an already-authorized lowering/contact cycle active through small tracking fluctuations.

    This deliberately does NOT bypass independent surface, collision, cable-obstacle, or
    ground-intruder checks in AutonomousWinchDeliverySystem.step(). Wind must remain inside
    the validated delivery envelope, and a payload must still be attached. The strict
    alignment/sway/velocity gates are evaluated again before contact and release.
    """
    try:
        wind_speed = float(wind_speed_xy_m_s)
    except (TypeError, ValueError):
        return False
    return bool(
        mission_phase == "4_DELIVERY_HOVER"
        and winch_state in {"LOWERING", "TOUCHDOWN"}
        and has_payload
        and math.isfinite(wind_speed)
        and wind_speed <= DELIVERY_MAX_OPERATIONAL_WIND_MS
    )

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
# 2. ADAPTIVE ESTIMATORS & OPTIMIZED SPATIAL HASH ENGINE
# =============================================================================
class VectorizedSpatialHashGrid2D:
    """
    Spatial hash grid maintaining pre-stacked contiguous arrays for 
    vectorized Liang-Barsky and slab-intersection raycasts.
    """
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
        """Precomputes contiguous matrix buffers for O(1) vectorized ray intersection."""
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

        # Trees sit in the open internal block setbacks. The old +/-25 m positions
        # intersected the 18 m footprints of buildings centred at +/-30 m.
        self.trees = [
            {"id": "Tree_1", "pos": np.array([-12.0, -12.0, 0.0], dtype=np.float32), "r": 3.2, "h": 8.0},
            {"id": "Tree_2", "pos": np.array([ 12.0, -12.0, 0.0], dtype=np.float32), "r": 3.0, "h": 7.5},
            {"id": "Tree_3", "pos": np.array([-12.0,  12.0, 0.0], dtype=np.float32), "r": 3.5, "h": 8.5},
            {"id": "Tree_4", "pos": np.array([ 12.0,  12.0, 0.0], dtype=np.float32), "r": 3.2, "h": 8.0}
        ]

        # Utility lines run outside the outer building footprints (the buildings
        # centred at y=+/-165 occupy +/-9 m around their centres). The old +/-170 m
        # placement bisected those buildings by up to 4 m.
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
        """Return conservative analytic hitboxes for every static world obstacle.

        The Genesis viewport is a non-physical mirror. These descriptors are consumed
        by the authoritative V99 control-barrier filter; support surfaces (ground,
        roadway and pads) are modelled separately by get_surface_height/landing logic.
        """
        obstacles = []
        for house in self.houses:
            x1, x2, y1, y2, top = [float(v) for v in house["aabb"]]
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            # Axis-aligned building footprint is the actual geometry. The previous
            # half-diagonal cylinder expanded an 18 m square to an 18 m radius and
            # caused needlessly early braking/avoidance around rooftops.
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
            # A support pole is part of the authoritative collision model as well as the render.
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
        """Summarise geometry represented by the authoritative collision model."""
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
        """Describe support/contact hitboxes not treated as no-fly volumes."""
        surfaces = [{"id": "Ground_Plane", "shape": "plane", "z": 0.0}]
        for i, road in enumerate(self.roads):
            surfaces.append({"id": f"Road_{i+1}", "shape": "aabb_plane",
                             "bounds": (road["x_min"], road["x_max"], road["y_min"], road["y_max"]), "z": 0.025})
        for i, pad in enumerate(self.depot_helipads):
            surfaces.append({"id": f"Depot_Pad_{i+1}", "shape": "cylinder", "pos": pad.copy(), "radius": 2.8, "z_top": 0.08})
        for house in self.houses:
            x1, x2, y1, y2, top = [float(v) for v in house["aabb"]]
            surfaces.append({"id": f"{house['id']}:ROOF", "shape": "aabb_plane", "bounds": (x1, x2, y1, y2), "z": top + 0.08})
            # Dedicated cylinder matches the visible 2.2 m-radius rooftop delivery pad.
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
                # The roof collision/support surface and visible drop pad top are
                # 8 cm above the building's structural top. Return the same physical
                # contact height used by the winch and landing-pad geometry.
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
        """Return footprint-aligned road-vehicle hitboxes in the authoritative world frame.

        Vehicle centres now sit at half body height; the AABBs match the 3.0 x 1.35 m
        rendered bodies and distinguish vehicles travelling along X from those along Y.
        CBF uncertainty/stopping buffers are added by the controller, not baked into the
        physical vehicle dimensions.
        """
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
        # Monotonic event counter lets the renderer detach the parcel at the exact
        # release moment instead of drawing it at the end of the retracting cable.
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
        self.last_abort_reason = None
        self.payout_wind_paused = False
        self.last_payout_pause_reason = None

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
        self.last_abort_reason = None
        self.payout_wind_paused = False
        self.last_payout_pause_reason = None

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

        # `wind_3d` is the physical disturbance used by the tether/cargo dynamics;
        # control gates must use the sensor estimate passed separately as `wind_2d`.
        # This keeps the winch controller from seeing ground-truth wind unavailable to
        # the vehicle. If the sensor estimate is missing or invalid, fail closed.
        wind_gate_xy = (np.asarray(wind_2d, dtype=np.float32).reshape(-1)[:2]
                        if wind_2d is not None else np.asarray(wind_3d[:2], dtype=np.float32))
        wind_speed_xy = (float(np.linalg.norm(wind_gate_xy))
                         if wind_gate_xy.size >= 2 and np.all(np.isfinite(wind_gate_xy))
                         else float("inf"))
        endpoint_for_contact = np.asarray(self.nodes_pos[-1], dtype=np.float32)
        endpoint_valid = (self.cable_len > 0.12 and np.all(np.isfinite(endpoint_for_contact))
                          and float(np.linalg.norm(endpoint_for_contact[:2] - drone_p[:2])) <= self.cable_len + 1.0
                          and (abs(float(endpoint_for_contact[0])) + abs(float(endpoint_for_contact[1])) > 1e-5))
        # Cable length is measured from the underside anchor to the package top;
        # the last PBD node is the package centre (another half-height below top).
        payload_z = (float(endpoint_for_contact[2]) if endpoint_valid else
                     float(drone_p[2] - PHASE3_DRONE_HALF_HEIGHT_M - self.cable_len - PHASE3_CARGO_HALF_HEIGHT_M))
        clearance_to_roof = payload_z - PHASE3_CARGO_HALF_HEIGHT_M - target_surface_z
        target_xy_arr = None if target_xy is None else np.asarray(target_xy, dtype=np.float32).reshape(-1)[:2]
        cargo_xy = drone_p[:2] + self.sway_offset
        endpoint = np.asarray(self.nodes_pos[-1], dtype=np.float32)
        # Ignore uninitialised endpoint state; otherwise use the cable simulation's
        # measured payload endpoint for a more faithful drop-zone check.
        if (self.cable_len > 0.12 and np.all(np.isfinite(endpoint))
                and float(np.linalg.norm(endpoint[:2] - drone_p[:2])) <= self.cable_len + 1.0
                and (abs(float(endpoint[0])) + abs(float(endpoint[1])) > 1e-5)):
            cargo_xy = endpoint[:2]
        drone_xy_error = (float(np.linalg.norm(drone_p[:2] - target_xy_arr))
                          if target_xy_arr is not None else 0.0)
        cargo_xy_error = (float(np.linalg.norm(cargo_xy - target_xy_arr))
                          if target_xy_arr is not None else 0.0)
        horizontal_speed = float(np.linalg.norm(drone_v[:2]))
        vertical_speed_abs = abs(float(drone_v[2]))
        sway_m = float(np.linalg.norm(self.sway_offset)) if self.cable_len > 0.12 else 0.0
        # Do not keep payout running if the aircraft physically departs the drop
        # location. Previously the continuation latch ignored horizontal displacement
        # throughout LOWERING, so the cable could reach its contact height tens of
        # metres away and only then report a failed placement.
        active_station_departure = (
            self.state in ("LOWERING", "TOUCHDOWN")
            and target_xy_arr is not None
            and (
                drone_xy_error > DEPOT_ACTIVE_STATION_DEPARTURE_M
                or horizontal_speed > DEPOT_ACTIVE_STATION_SPEED_MS
                or vertical_speed_abs > DELIVERY_ACTIVE_DEPARTURE_VERTICAL_MS
            )
        )
        delivery_alignment_failures = []
        if target_xy_arr is not None:
            if drone_xy_error > DELIVERY_DRONE_CENTER_TOLERANCE_M:
                delivery_alignment_failures.append(
                    f"drone_xy={drone_xy_error:.2f}m>{DELIVERY_DRONE_CENTER_TOLERANCE_M:.2f}m")
            if cargo_xy_error > DELIVERY_CARGO_CENTER_TOLERANCE_M:
                delivery_alignment_failures.append(
                    f"cargo_xy={cargo_xy_error:.2f}m>{DELIVERY_CARGO_CENTER_TOLERANCE_M:.2f}m")
            if horizontal_speed > DELIVERY_MAX_HORIZONTAL_SPEED_MS:
                delivery_alignment_failures.append(
                    f"vxy={horizontal_speed:.2f}m/s>{DELIVERY_MAX_HORIZONTAL_SPEED_MS:.2f}m/s")
            if vertical_speed_abs > DELIVERY_MAX_VERTICAL_SPEED_MS:
                delivery_alignment_failures.append(
                    f"|vz|={vertical_speed_abs:.2f}m/s>{DELIVERY_MAX_VERTICAL_SPEED_MS:.2f}m/s")
            if self.cable_len > 0.12 and sway_m > DELIVERY_MAX_SWAY_M:
                delivery_alignment_failures.append(
                    f"sway={sway_m:.2f}m>{DELIVERY_MAX_SWAY_M:.2f}m")
        delivery_alignment_ok = not delivery_alignment_failures
        dropzone_occupied = (agent_dist < 2.8) and (target_surface_z < 2.0) and (self.state == "STANDBY")
        lockout = ((not is_surface_safe) or wire_hit or (not is_settled) or
                   dropzone_occupied or active_station_departure)
        lockout_reasons = []
        if not is_surface_safe:
            lockout_reasons.append("surface/safety gate")
        if wire_hit:
            lockout_reasons.append("cable/obstacle collision")
        if not is_settled:
            lockout_reasons.append("deployment authorization/wind gate")
        if dropzone_occupied:
            lockout_reasons.append("drop zone occupied")
        if active_station_departure:
            lockout_reasons.append(
                "aircraft departed delivery envelope "
                f"(drone_xy={drone_xy_error:.2f}m, vxy={horizontal_speed:.2f}m/s, "
                f"|vz|={vertical_speed_abs:.2f}m/s)"
            )

        if agent_breach and (target_surface_z < 2.0) and self.state in ["LOWERING", "TOUCHDOWN"]:
            self.last_abort_reason = "ground intruder breach"
            self.state = "ABORT_RETRACT"

        omega_pendulum = math.sqrt(G_ACCEL / max(0.08, float(self.cable_len) +
                                                     (PHASE3_CARGO_HALF_HEIGHT_M if self.has_payload else 0.0)))
        in_resonance_band = (1.8 < omega_pendulum < 2.5) and (self.state == "LOWERING")

        # The trained executive now participates in winch decisions. Physical
        # safety/clearance checks remain authoritative and an abort always retracts.
        action_token = int(neural_token_idx)
        learned_rate = float(np.clip(nn_winch_rate, -1.0, 1.0))

        if self.state == "STANDBY":
            self.cable_len = 0.08
            self.cable_vel = 0.0
            # HOVER_LOCK/WINCH_LOWER are the only trained actions allowed to arm
            # a new deployment. The caller has already checked model confidence,
            # predicted surface trust, station-keeping, and actual surface safety.
            if (not lockout and self.has_payload and is_settled
                    and action_token in (1, 2)):
                self.last_abort_reason = None
                self.payout_wind_paused = False
                self.last_payout_pause_reason = None
                self.state = "LOWERING"

        elif self.state == "LOWERING":
            # A gust during deployment should stop further payout, not automatically
            # trigger a retract with the package still attached. Hysteresis avoids
            # rapid pause/resume chatter around a single wind threshold. The existing
            # 7 m/s operational boundary remains a hard lockout/abort above this point.
            if wind_speed_xy >= DELIVERY_PAYOUT_WIND_PAUSE_MS:
                self.payout_wind_paused = True
            elif wind_speed_xy <= DELIVERY_PAYOUT_WIND_RESUME_MS:
                self.payout_wind_paused = False

            # Once the cable is deployed, a TRANSIT (0) or HOVER_LOCK (1) prediction
            # is not interpreted as a retract command. The original classifier is
            # dominated by those classes; treating either as an abort repeatedly
            # cancels drops before contact. Explicit RETRACT/SERVICE/DIVERT actions
            # still request a controlled retract; physical lockouts always override.
            policy_requests_retract = action_token in (4, 5, 6)
            if lockout:
                if self.cable_len > 0.12:
                    # A physical hazard overrides the policy and requests a fast abort.
                    self.last_abort_reason = "; ".join(lockout_reasons) or "unspecified lockout"
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
                payout_hold_reasons = []
                if self.payout_wind_paused:
                    payout_hold_reasons.append(
                        f"wind hold {wind_speed_xy:.2f}m/s (resume <= {DELIVERY_PAYOUT_WIND_RESUME_MS:.1f}m/s)")
                if not delivery_alignment_ok:
                    payout_hold_reasons.extend(delivery_alignment_failures)

                # Match the Phase-3 teacher's payout pause semantics in the live
                # simulator: never keep extending cable while the drone, cargo centre,
                # swing amplitude, or velocity is outside the release envelope. The
                # anti-sway controller continues operating during this hold so the
                # attached parcel can settle before descent continues.
                payout_authorized = not payout_hold_reasons
                self.last_payout_pause_reason = ("; ".join(payout_hold_reasons)
                                                 if not payout_authorized else None)
                if payout_authorized:
                    flare_scale = float(np.clip(clearance_to_roof / 1.20, 0.20, 1.0)) if clearance_to_roof < 1.20 else 1.0
                    base_rate = float(np.clip(2.40 - 0.12 * wind_speed_xy, 1.20, 2.40)) * flare_scale
                    if in_resonance_band and clearance_to_roof > 1.5:
                        base_rate = 3.20

                    # Label 1 of the winch head is the normalized lowering/retracting
                    # request. It adjusts speed but can never override clearance checks.
                    if action_token == 2:
                        learned_rate = max(0.0, learned_rate)
                    elif action_token == 3:
                        learned_rate = float(np.clip(learned_rate, -0.25, 0.0))
                    self.cable_vel = base_rate + float(np.clip(learned_rate * 0.8, -0.6, 0.6))
                    self.cable_len += self.cable_vel * dt
                else:
                    # Hold current cable length; do not retract or keep lowering merely
                    # because transient wind/sway made strict placement unsafe.
                    self.cable_vel = 0.0

                fast_tension_drop = (self.tension_derivative < -35.0) and (clearance_to_roof < 0.40)
                policy_touchdown = (action_token == 3 and 0.0 <= clearance_to_roof <= 0.10)
                physical_touchdown = (clearance_to_roof <= 0.10 or fast_tension_drop
                                      or self.contact_normal_force >= 0.40 * PAYLOAD_WEIGHT_N)
                if physical_touchdown or policy_touchdown:
                    # Contact outside the designated delivery envelope is a failed
                    # placement, not permission to release. Retract with the package.
                    if delivery_alignment_ok and is_surface_safe and is_settled:
                        self.state = "TOUCHDOWN"
                        self.touchdown_timer = 0.0
                    else:
                        failures = []
                        if not delivery_alignment_ok:
                            failures.append("placement gate [" + ", ".join(delivery_alignment_failures) + "]")
                        if not is_surface_safe:
                            failures.append("surface gate")
                        if not is_settled:
                            failures.append("authorization/wind gate")
                        self.last_abort_reason = "; ".join(failures) or "touchdown rejected"
                        self.state = "ABORT_RETRACT"

        elif self.state == "TOUCHDOWN":
            self.cable_vel = 0.0
            # Keep the aircraft and parcel aligned for the full contact debounce.
            # If either drifts out of the drop envelope, abort and retain the parcel.
            if (not delivery_alignment_ok) or (not is_surface_safe) or (not is_settled):
                failures = []
                if not delivery_alignment_ok:
                    failures.append("placement/contact gate [" + ", ".join(delivery_alignment_failures) + "]")
                if not is_surface_safe:
                    failures.append("surface gate")
                if not is_settled:
                    failures.append("authorization/wind gate")
                self.last_abort_reason = "; ".join(failures) or "contact debounce rejected"
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
                # Continue retracting safely if the policy is uncertain or chooses
                # another action; a weak prediction must never strand an extended cable.
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
        # When loaded, the winch payout is anchor-to-package-top length; the final
        # PBD node tracks the package centre, so the geometric span includes one
        # cargo half-height. After release, the empty hook uses the cable length only.
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
                        # Drone cable attachment is kinematic/fixed for this solve.
                        predicted_pos[i + 1] -= correction
                    else:
                        # Position-Based Dynamics correction is proportional to inverse
                        # mass. The loaded payload at the terminal node is heavier than
                        # a cable node, so it must absorb LESS—not 65%—of the correction.
                        mass_i = CARGO_MASS_KG if (i == self.num_nodes - 1 and self.has_payload) else 0.04
                        mass_j = CARGO_MASS_KG if (i + 1 == self.num_nodes - 1 and self.has_payload) else 0.04
                        inv_i = 1.0 / max(1e-6, float(mass_i))
                        inv_j = 1.0 / max(1e-6, float(mass_j))
                        inv_sum = inv_i + inv_j
                        alpha_i = inv_i / inv_sum
                        alpha_j = inv_j / inv_sum
                        predicted_pos[i] += alpha_i * correction
                        predicted_pos[i + 1] -= alpha_j * correction

        # The cable endpoint represents the parcel centre, so its contact height is
        # the pad top plus half the package height. Do not apply this ground clamp to
        # an empty cable after release: that pins the hook to the roof during retract
        # and injects an impossible constraint which can produce the observed swing.
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
# 10. MULTI-MODAL SENSOR SUITE & PHYSICAL INSTRUMENTATION
# =============================================================================
class AdvancedMultiModalSensorSuite:
    def __init__(self, world: MetropolitanCityGridWorld400m):
        self.world = world
        # Per-aircraft simulated pitot/multi-hole air-data sensor state. Each drone
        # has independent bias and filtering; no camera-selection coupling.
        self._wind_sensor_state = {}
        self._wind_sensor_bias = {}

    def poll_wind_vector(self, drone_v, current_wind, drone_R, dt=0.016, sensor_id=0, ground_velocity_est=None):
        """Estimate local 3-D wind from GNSS/INS ground velocity and air-data flow.

        `current_wind` is used only to synthesize the physical air-relative velocity
        seen by the simulated probe. The controller receives the noisy, biased,
        low-pass-filtered estimate—not the simulator's ground-truth wind vector.

        Returns (wind_xyz_m_s, horizontal_speed_m_s, meteorological_from_deg, quality).
        Direction is undefined in near-calm conditions and is reported as 0 degrees.
        """
        key = int(sensor_id)
        v_ground_true = np.asarray(drone_v, dtype=np.float32).reshape(3)
        v_ground_for_sensor = (v_ground_true if ground_velocity_est is None else
                               np.asarray(ground_velocity_est, dtype=np.float32).reshape(3))
        wind_truth = np.asarray(current_wind, dtype=np.float32).reshape(3)
        rotation = np.asarray(drone_R, dtype=np.float32).reshape(3, 3)
        dt = float(np.clip(dt, 1e-4, 0.25))

        # Noisy inertial/GNSS ground-velocity measurement. When available, use the
        # estimated velocity supplied by the EKF rather than simulator truth.
        v_ground_meas = v_ground_for_sensor + np.random.normal(0.0, 0.045, 3).astype(np.float32)
        # Five-port/pitot air-data measurement in body coordinates, with realistic
        # measurement noise and a slowly varying sensor bias.
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
            # Respond quickly enough for feedback/feed-forward control during weather
            # transitions. Noise remains bounded by the air-data model and this low-pass.
            alpha = dt / (0.09 + dt)
            estimate = previous + alpha * (wind_raw - previous)
        estimate = np.asarray(estimate, dtype=np.float32)
        self._wind_sensor_state[key] = estimate.copy()

        horizontal_speed = float(np.linalg.norm(estimate[:2]))
        if horizontal_speed < 0.25:
            wind_from_deg = 0.0
        else:
            # Wind vector points toward where air is moving; meteorological
            # direction is the bearing the wind comes FROM, clockwise from north.
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
        """Simulate the hook IMU with swing displacement and angular-rate cues.

        The optional velocity argument preserves backward compatibility for callers
        outside this script; V22's harvest and live loops pass the cable solver's
        relative sway velocity explicitly.
        """
        if not has_payload or cable_l <= 0.10:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        a_box = drone_a + np.array([-sway_2d[0]*2.5, -sway_2d[1]*2.5, 0.2], dtype=np.float32)
        if sway_velocity_2d is None:
            # Compatibility fallback for any external caller using the V21 signature.
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
        # Vegetation cues must follow the actual world tree coordinates; a stale
        # (-25,-25) m park centre no longer matches the four modelled trees at +/-12 m.
        tree_positions = [np.asarray(tree["pos"], dtype=np.float32)[:2] for tree in self.world.trees]
        if tree_positions:
            vegetation_dist = min(float(np.linalg.norm(np.asarray([px, py], dtype=np.float32) - tree_xy))
                                  for tree_xy in tree_positions)
        else:
            vegetation_dist = 999.0
        park_cool = -3.5 * math.exp(-vegetation_dist / 22.0)
        th_dt = float(roof_heat + park_cool + 0.5)

        dolp = float(0.12 + 0.55 * math.exp(-min_roof_dist / 6.0))
        ndvi = float(0.10 + 0.68 * math.exp(-vegetation_dist / 25.0))
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

            # Trunk/foliage cylinders: intersect the horizontal ray with each radius.
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

            # Horizontal powerline segment intersections are visible to any ray at wire altitude.
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

            # Civil aircraft is a 3-D sphere, projected to its altitude slice for a lidar ray.
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
        # Expose the same geometry used by the world and the training features.
        self.wire_y_positions = POWERLINE_Y_POSITIONS_M
        self.wire_x_min = POWERLINE_X_MIN_M
        self.wire_x_max = POWERLINE_X_MAX_M
        self.wire_z = POWERLINE_Z_M

    def wire_distance_m(self, drone_p):
        """Shortest Euclidean distance to either finite powerline segment."""
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
# 12. MATTER-WAVE QUANTUM SENSOR & REGULARIZED ERROR-STATE EKF ON SO(3)
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
        """Numerically stable measurement update with Tikhonov-regularized solver."""
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
# 13. RECIPROCAL CBF-QP SAFETY FILTER & GEOMETRIC SE(3) ACTUATOR MIXER
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
                # A drone must not avoid its own attached package/body collider.
                continue
            obs_id = obs.get("id")
            obs_h = float(obs.get("height", 0.0))
            shape = obs.get("shape", "cylinder")
            # Never exempt a building merely because it is the target. The exemption
            # is valid only once the drone is safely above its roof; below roof level
            # its footprint remains an active collision constraint.
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
                    # Inside an obstacle footprint: use the closest face as the
                    # signed-distance gradient to drive the aircraft back outside.
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
                # Flat roofs are support surfaces, not vertical walls above their top.
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
            # Keep the controller's vertical influence volume above the object's top
            # as well as below it. Using `obs_h - 0.4` clipped the envelope short at
            # the top of trees/intruders and could stop avoiding an object before the
            # drone's rotor sweep had cleared it. AABB and cylinder branches now share
            # the same +0.4 m top margin.
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
    """Bound horizontal velocity error by vector magnitude for smooth braking.

    Component-wise clipping allows the diagonal acceleration magnitude to reach
    sqrt(2) times the intended bound. For precision approach we instead cap the
    error vector itself, making the derivative term's acceleration predictable.
    """
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
        # Jerk/acceleration-limited vertical velocity target; prevents command steps
        # at phase boundaries from exciting a large vertical oscillation.
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
            station_keeping=False, sway_velocity_2d=None, approach_braking=False,
            rho_air=1.225):

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

        # Smooth vertical velocity demand with bounded acceleration and jerk. The
        # final position controller still tracks the requested altitude, but the
        # commanded vertical speed cannot jump instantly at a mission-phase change.
        dt_safe = max(1e-4, float(dt))
        requested_vz = float(v_cmd_z)
        required_az = float(np.clip((requested_vz - self.vertical_v_cmd_smooth) / dt_safe, -0.90, 0.90))
        max_da = 2.8 * dt_safe
        da = float(np.clip(required_az - self.vertical_a_cmd_smooth, -max_da, max_da))
        self.vertical_a_cmd_smooth = float(np.clip(self.vertical_a_cmd_smooth + da, -0.90, 0.90))
        old_vz = self.vertical_v_cmd_smooth
        self.vertical_v_cmd_smooth = float(np.clip(old_vz + self.vertical_a_cmd_smooth * dt_safe, -0.60, 1.05))
        # Do not snap to requested_vz on target crossing: that shortcut changes the
        # acceleration state in one frame and violates the jerk bound during reversals.
        # The rate-limited loop converges without a discontinuous command step.
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
            # Tight position/velocity loop for rooftop delivery and pad approach.
            # Neural feed-forward is suppressed below so the learned policy cannot
            # push the aircraft out of its physical station-keeping envelope.
            k_p_xy = 2.65
            k_d_xy = 3.25

        # Do not clamp velocity error before feedback. The old +/-1.8..2.8 m/s
        # clipping destroyed the braking signal once a drone overshot its desired
        # speed; the saturated position term could then keep accelerating toward
        # the waypoint. Bound position and damping *acceleration vectors* instead.
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

        # Increase integral correction gradually with measured horizontal wind. The
        # integrator is already bounded to +/-2 m; this compensates residual drag/model
        # errors without making the controller depend solely on neural output.
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

        # Wind-disturbance feed-forward: the simulator supplies a local wind estimate
        # each frame, so use a faster low-pass than the old 0.85 s filter. The filter
        # still suppresses gust noise, but does not leave the drone uncompensated
        # through most of a short rooftop hover.
        # Two serial 0.2 s filters left a noticeable gap between a gust and the
        # compensating force. Use a faster, still stable controller-side filter during
        # station keeping/precision landing; cruise keeps the previous smoother response.
        wind_tau_xy = 0.075 if station_keeping else 0.22
        alpha_w = dt / (wind_tau_xy + dt)
        self.wind_filt[:2] += alpha_w * (estimated_wind[:2] - self.wind_filt[:2])
        alpha_w_z = dt / (0.10 + dt)
        self.wind_filt[2] += alpha_w_z * (estimated_wind[2] - self.wind_filt[2])
        
        v_air_rel = v_est - self.wind_filt
        speed_rel = float(np.linalg.norm(v_air_rel))
        rho_control = float(rho_air) if math.isfinite(float(rho_air)) else 1.225
        rho_control = float(np.clip(rho_control, 0.75, 1.45))
        f_aero_drag = -0.5 * rho_control * np.array([0.22, 0.22, 0.38], dtype=np.float32) * v_air_rel * max(1.0, speed_rel)
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

        # Action-head output now selects a conservative control mode. These gains
        # modulate only the learned correction; position/velocity tracking and the
        # CBF safety filter remain active for every action.
        action_guidance_gain = (1.00, 0.85, 0.80, 0.60, 0.72, 0.65, 0.0)
        token_safe = int(np.clip(token_idx, 0, len(action_guidance_gain) - 1))
        mode_gain = action_guidance_gain[token_safe]
        a_neural_ff = (neural_guidance[:3] * 6.0 * mode_gain
                       if (neural_guidance is not None and not station_keeping
                           and not NEURAL_GUIDANCE_DISABLED)
                       else np.zeros(3, dtype=np.float32))
        # The policy is a residual, not the primary altitude controller. Large learned
        # Z corrections can overwhelm the deterministic vertical-velocity feedback and
        # produce runaway climb/descent (the previous live logs reached ~14.85 m/s).
        # Keep XY guidance intact, but tightly bound only the learned vertical term.
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
            # A fixed ~19.5 degree cap could be below the tilt required to resist
            # crosswind drag near touchdown. Landing wind holds prevent descent in the
            # strongest gusts; this bounded extra trim preserves disturbance authority
            # during moderate wind without allowing unlimited low-altitude bank.
            base_tilt_deg = min(28.0, 5.5 + 1.55 * wind_trim_deg)
        elif is_climbing_from_base:
            base_tilt_deg = 5.5 + (0.5 * wind_trim_deg)
        elif is_vertical_flight or p_est[2] < 2.0:
            base_tilt_deg = 8.5 + (0.6 * wind_trim_deg)
        else:
            # In cruise and rooftop station-keeping, the previous implementation
            # calculated wind_trim_deg but ignored it, leaving a hard 24-degree cap.
            # Permit more tilt only when dynamic pressure requires it, capped at 38 deg.
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

        k_R_tuned = np.array([1.90, 1.90, 1.60], dtype=np.float32)
        k_w_tuned = np.array([1.35, 1.35, 0.95], dtype=np.float32)

        gyroscopic = np.cross(omega_est, J_eff @ omega_est)
        tau = -k_R_tuned * e_R - k_w_tuned * e_w + gyroscopic
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

# Normalization contracts for the auxiliary suspended-cargo prediction target.
# The learned values are dimensionless; conversion to SI is used for validation logs only.
CARGO_DYNAMICS_SWAY_SCALE_M = 5.0
CARGO_DYNAMICS_SWAY_RATE_SCALE_MS = 8.0
CARGO_DYNAMICS_TENSION_SCALE_N = 45.0
KINEMATIC_EQUATION_RESIDUAL_WEIGHT = 0.05
# The force-residual head predicts an SI force correction normalised to the force
# corresponding to 12 m/s^2 on the Phase-3 Genesis drone mass. Its training targets
# are constructed from measured Genesis acceleration, commanded acceleration, and
# the same drag+tether forces actually applied to the rigid body.
FORCE_BALANCE_RESIDUAL_SCALE_N = float(DRY_MASS_KG * 12.0)
FORCE_BALANCE_RESIDUAL_WEIGHT = 0.05
FORCE_BALANCE_MAX_RESIDUAL_ACCEL_MS2 = 30.0
FORCE_BALANCE_MIN_AIRBORNE_ALTITUDE_M = 1.5
# Rotational Euler residual is trained on Phase-1/2 BEMT transitions where the
# simulator exposes body-frame motor torques and angular velocity. The reduced
# Genesis Phase-3 model has no propeller torque actuation, so it is deliberately
# not used to synthesize these targets.
ROTATIONAL_DYNAMICS_RESIDUAL_SCALE_NM = 0.50
ROTATIONAL_DYNAMICS_RESIDUAL_WEIGHT = 0.05
ROTATIONAL_DYNAMICS_MAX_ANGULAR_SPEED_RAD_S = 80.0
ROTATIONAL_DYNAMICS_MAX_ANGULAR_ACCEL_RAD_S2 = 200.0
ROTATIONAL_DYNAMICS_MAX_RESIDUAL_NM = 1.50
ROTATIONAL_DYNAMICS_MIN_SURFACE_CLEARANCE_M = 0.45
# Uncertainty head estimates log variance in normalized target coordinates for:
# kinematics[0:6], cargo[6:11], force residual[11:14], rotational residual[14:17].
# Features and prediction means are detached from this objective: calibration cannot
# backpropagate into the control policy or the primary mean-prediction heads.
PHYSICS_UNCERTAINTY_DIM = 17
PHYSICS_UNCERTAINTY_LOGVAR_MIN = -6.0
PHYSICS_UNCERTAINTY_LOGVAR_MAX = 3.0
PHYSICS_UNCERTAINTY_NLL_WEIGHT = 0.02

# Forecast true future states after fixed numbers of BEMT frames (dt=0.016 s).
# Each horizon predicts world displacement from the source state and future velocity.
# The final 3 target columns are independent validity masks, one per horizon.
TRAJECTORY_FORECAST_HORIZON_FRAMES = (16, 32, 62)
TRAJECTORY_FORECAST_HORIZONS_S = tuple(0.016 * n for n in TRAJECTORY_FORECAST_HORIZON_FRAMES)
TRAJECTORY_FORECAST_HORIZON_COUNT = len(TRAJECTORY_FORECAST_HORIZON_FRAMES)
TRAJECTORY_FORECAST_VALUES_PER_HORIZON = 6
TRAJECTORY_FORECAST_DIM = TRAJECTORY_FORECAST_HORIZON_COUNT * TRAJECTORY_FORECAST_VALUES_PER_HORIZON
TRAJECTORY_FORECAST_TARGET_DIM = TRAJECTORY_FORECAST_DIM + TRAJECTORY_FORECAST_HORIZON_COUNT
TRAJECTORY_FORECAST_POSITION_SCALE_M = 10.0
TRAJECTORY_FORECAST_VELOCITY_SCALE_MS = 8.0
TRAJECTORY_FORECAST_WEIGHT = 0.08
TRAJECTORY_FORECAST_MAX_SPEED_MS = 15.0
TRAJECTORY_FORECAST_LOSS_HORIZON_WEIGHTS = (1.0, 1.0, 1.0)
# Action-conditioned one-step transition model. Action = [3 normalized guidance
# acceleration components, one-hot FSM command, normalized winch payout command].
# Targets are measured delta position [m], delta velocity [m/s], delta body-rate [rad/s],
# then a validity mask. These are targets from a single dt=0.016 s BEMT transition.
ACTION_TRANSITION_ACTION_DIM = 3 + NUM_EXECUTIVE_STATES + 1
ACTION_TRANSITION_OUTPUT_DIM = 9
ACTION_TRANSITION_TARGET_DIM = ACTION_TRANSITION_OUTPUT_DIM + 1
ACTION_TRANSITION_POSITION_SCALE_M = 0.25
ACTION_TRANSITION_VELOCITY_SCALE_MS = 1.0
ACTION_TRANSITION_ANGULAR_RATE_SCALE_RADS = 2.0
ACTION_TRANSITION_WEIGHT = 0.06
ACTION_TRANSITION_MAX_DELTA_POSITION_M = 0.90
ACTION_TRANSITION_MAX_DELTA_VELOCITY_MS = 6.0
ACTION_TRANSITION_MAX_DELTA_OMEGA_RADS = 8.0
# Action-conditioned environment-state forecast for next-step normalized sensor channels:
# wind xyz (features 42:45), cable length (53), cable rate (54), and icing (65).
# Targets are read from the next per-drone observation row (3 rows later in frame-major data).
ENVIRONMENT_TRANSITION_FEATURE_INDICES = (42, 43, 44, 53, 54, 65)
ENVIRONMENT_TRANSITION_DIM = len(ENVIRONMENT_TRANSITION_FEATURE_INDICES)
ENVIRONMENT_TRANSITION_TARGET_DIM = ENVIRONMENT_TRANSITION_DIM + 1
ENVIRONMENT_TRANSITION_WEIGHT = 0.035
ENVIRONMENT_TRANSITION_MAX_WIND_DELTA_NORM = 0.65  # normalized feature units, 5.2 m/s
ENVIRONMENT_TRANSITION_MAX_CABLE_LENGTH_DELTA = 0.06  # normalized feature units, 0.9 m
ENVIRONMENT_TRANSITION_MAX_CABLE_RATE_DELTA = 1.25  # normalized feature units, 6.25 m/s
_QUINTIC_MAX_DS = 1.875
_QUINTIC_MAX_ABS_DDS = 10.0 / math.sqrt(3.0)


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
        # Predict [average world velocity over next dt / 8 m/s,
        #          observed world acceleration over next dt / 12 m/s^2].
        # This branch is auxiliary: no live actuator or safety decision consumes it.
        self.kinematics_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 6)
        )
        # Predict next-step relative cargo sway x/y (per 5 m), relative sway
        # velocity x/y (per 8 m/s), and cable tension (per 45 N). This head is
        # auxiliary only: no live actuator, release or safety decision consumes it.
        self.cargo_dynamics_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 5)
        )
        # Predict the SI force-balance residual correction [Fx,Fy,Fz]/scale.
        # This head is trained only from paired Genesis transitions in Phase 3.
        # The persistent boolean buffer distinguishes a trained head from random
        # weights saved by Phase 1/2; it is deliberately not an optimizer parameter.
        self.force_balance_residual_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 3)
        )
        self.register_buffer("_force_balance_head_trained_state", torch.tensor(False, dtype=torch.bool))
        # Predict the torque residual left by the rigid-body Euler equation, scaled
        # by 0.5 N*m. This head is trained from the authoritative BEMT harvester,
        # where actual motor torques and body-frame angular velocity are available.
        self.rotational_dynamics_residual_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 3)
        )
        self.register_buffer("_rotational_dynamics_head_trained_state", torch.tensor(False, dtype=torch.bool))
        # Heteroscedastic log-variance estimate for all auxiliary physics targets.
        # Feeding detached features prevents this diagnostic branch from changing the
        # shared policy representation; the means are detached again in its loss.
        self.physics_uncertainty_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, PHYSICS_UNCERTAINTY_DIM)
        )
        nn.init.constant_(self.physics_uncertainty_head[-1].bias, -2.0)
        self.register_buffer("_physics_uncertainty_base_trained_state", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("_force_uncertainty_trained_state", torch.tensor(False, dtype=torch.bool))
        # Multi-horizon measured future displacement/velocity forecast. Its output is
        # diagnostic only; no live actuator or safety decision consumes it.
        self.trajectory_forecast_head = nn.Sequential(
            nn.Linear(512, 192), nn.LayerNorm(192), nn.SiLU(),
            nn.Linear(192, 96), nn.SiLU(), nn.Linear(96, TRAJECTORY_FORECAST_DIM)
        )
        self.register_buffer("_trajectory_forecast_head_trained_state", torch.tensor(False, dtype=torch.bool))
        # Action-conditioned one-step dynamics: the latent observation representation
        # is combined with the high-level command actually passed to the mixer/winch
        # teacher path, and predicts delta-position, delta-velocity and delta-body-rate.
        # The head is diagnostic/offline-only until held-out performance warrants use.
        self.action_transition_action_encoder = nn.Sequential(
            nn.Linear(ACTION_TRANSITION_ACTION_DIM, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Linear(64, 48), nn.SiLU())
        self.action_conditioned_transition_head = nn.Sequential(
            nn.Linear(512 + 48, 192), nn.LayerNorm(192), nn.SiLU(),
            nn.Linear(192, 96), nn.SiLU(), nn.Linear(96, ACTION_TRANSITION_OUTPUT_DIM))
        self.register_buffer("_action_conditioned_transition_head_trained_state", torch.tensor(False, dtype=torch.bool))
        # Predict next observed normalized wind/cable/ice channels conditioned on
        # the same high-level action as the kinematic transition head. Output is
        # diagnostic/offline-only; it never directly drives the live controller.
        self.environment_observation_transition_head = nn.Sequential(
            nn.Linear(512 + 48, 160), nn.LayerNorm(160), nn.SiLU(),
            nn.Linear(160, 96), nn.SiLU(), nn.Linear(96, ENVIRONMENT_TRANSITION_DIM))
        self.register_buffer("_environment_observation_transition_trained_state", torch.tensor(False, dtype=torch.bool))

    def forward(self, x, return_kinematics=False, return_physics=False, return_force_residual=False,
                return_rotational_residual=False, return_uncertainty=False,
                return_trajectory_forecast=False, return_action_transition=False,
                return_environment_transition=False, transition_action=None):
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

        if return_action_transition or return_environment_transition:
            if transition_action is None:
                raise ValueError("transition_action is required when return_action_transition=True")
            if (transition_action.ndim != 2 or transition_action.shape[0] != x.shape[0]
                    or transition_action.shape[1] != ACTION_TRANSITION_ACTION_DIM):
                raise ValueError(
                    f"transition_action must have shape [N,{ACTION_TRANSITION_ACTION_DIM}], "
                    f"got {tuple(transition_action.shape)}")
            transition_action = torch.nan_to_num(transition_action.to(dtype=feat.dtype, device=feat.device),
                                                 nan=0.0, posinf=1.0, neginf=-1.0)

        def _pack_outputs(*values):
            packed = list(values)
            action_embed = None
            if return_action_transition or return_environment_transition:
                action_embed = self.action_transition_action_encoder(transition_action)
            if return_action_transition:
                action_transition = self.action_conditioned_transition_head(
                    torch.cat((feat, action_embed), dim=-1))
                packed.append(action_transition)
            if return_uncertainty:
                log_variance = self.physics_uncertainty_head(feat.detach())
                packed.append(log_variance)
            if return_environment_transition:
                environment_transition = self.environment_observation_transition_head(
                    torch.cat((feat, action_embed), dim=-1))
                packed.append(environment_transition)
            return tuple(packed)

        if return_force_residual and return_rotational_residual:
            kinematics = self.kinematics_head(feat)
            cargo_dynamics = self.cargo_dynamics_head(feat)
            force_balance_residual = self.force_balance_residual_head(feat)
            rotational_residual = self.rotational_dynamics_residual_head(feat)
            return _pack_outputs(guidance, fsm_logits, winch_cmd, health_pred, trust_scores, value,
                                 kinematics, cargo_dynamics, force_balance_residual, rotational_residual)
        if return_force_residual:
            kinematics = self.kinematics_head(feat)
            cargo_dynamics = self.cargo_dynamics_head(feat)
            force_balance_residual = self.force_balance_residual_head(feat)
            return _pack_outputs(guidance, fsm_logits, winch_cmd, health_pred, trust_scores, value,
                                 kinematics, cargo_dynamics, force_balance_residual)
        if return_rotational_residual:
            kinematics = self.kinematics_head(feat)
            cargo_dynamics = self.cargo_dynamics_head(feat)
            rotational_residual = self.rotational_dynamics_residual_head(feat)
            if return_trajectory_forecast:
                trajectory_forecast = self.trajectory_forecast_head(feat)
                return _pack_outputs(guidance, fsm_logits, winch_cmd, health_pred, trust_scores, value,
                                     kinematics, cargo_dynamics, rotational_residual, trajectory_forecast)
            return _pack_outputs(guidance, fsm_logits, winch_cmd, health_pred, trust_scores, value,
                                 kinematics, cargo_dynamics, rotational_residual)
        if return_physics:
            kinematics = self.kinematics_head(feat)
            cargo_dynamics = self.cargo_dynamics_head(feat)
            return _pack_outputs(guidance, fsm_logits, winch_cmd, health_pred, trust_scores, value,
                                 kinematics, cargo_dynamics)
        if return_kinematics:
            kinematics = self.kinematics_head(feat)
            return _pack_outputs(guidance, fsm_logits, winch_cmd, health_pred, trust_scores, value, kinematics)
        return _pack_outputs(guidance, fsm_logits, winch_cmd, health_pred, trust_scores, value)

MultiModalFlightExecutiveNet = DeepDenseFlightExecutiveNet
MultiModalPPOActorCriticNet = DeepDenseFlightExecutiveNet


def _masked_gaussian_uncertainty_nll(prediction, log_variance, target_with_validity, width):
    """Heteroscedastic Gaussian NLL in normalized target units, masked by validity.

    This estimates conditional uncertainty only: the prediction mean is detached and
    the uncertainty head receives detached features, so this objective cannot train
    the primary prediction mean or alter the policy/backbone through the uncertainty
    branch. Last target column is a 1-valid/0-invalid mask.
    """
    width = int(width)
    if prediction.ndim != 2 or prediction.shape[-1] != width:
        raise ValueError(f"uncertainty mean must have shape [N,{width}], got {tuple(prediction.shape)}")
    if log_variance.shape != prediction.shape:
        raise ValueError(f"log variance shape {tuple(log_variance.shape)} must match prediction {tuple(prediction.shape)}")
    if target_with_validity.ndim != 2 or target_with_validity.shape[-1] != width + 1:
        raise ValueError(f"uncertainty target must have shape [N,{width+1}], got {tuple(target_with_validity.shape)}")
    if prediction.shape[0] != target_with_validity.shape[0]:
        raise ValueError("uncertainty prediction and target batch lengths must match")
    valid = ((target_with_validity[:, width] > 0.5)
             & torch.isfinite(target_with_validity[:, :width]).all(dim=-1)
             & torch.isfinite(prediction.detach()).all(dim=-1)
             & torch.isfinite(log_variance).all(dim=-1))
    if not bool(valid.any()):
        return log_variance.sum() * 0.0
    mean_error = prediction.detach()[valid] - target_with_validity[valid, :width]
    logvar = log_variance[valid].clamp(PHYSICS_UNCERTAINTY_LOGVAR_MIN,
                                      PHYSICS_UNCERTAINTY_LOGVAR_MAX)
    return 0.5 * (logvar + mean_error.square() * torch.exp(-logvar)).mean()


def _physics_uncertainty_phase12_loss(log_variance, pkin, pcargo, prot, ykin, ycargo, yrot):
    """Calibrate kinematics, cargo and BEMT-rotational uncertainty in Phase 1/2."""
    pieces = [
        _masked_gaussian_uncertainty_nll(pkin, log_variance[:, 0:6], ykin, 6),
        _masked_gaussian_uncertainty_nll(pcargo, log_variance[:, 6:11], ycargo, 5),
        _masked_gaussian_uncertainty_nll(prot, log_variance[:, 14:17], yrot, 3),
    ]
    return torch.stack(pieces).mean()


def _physics_uncertainty_phase3_loss(log_variance, pkin, pcargo, pforce, ykin, ycargo, yforce):
    """Calibrate kinematics, cargo and Genesis force-residual uncertainty in Phase 3."""
    pieces = [
        _masked_gaussian_uncertainty_nll(pkin, log_variance[:, 0:6], ykin, 6),
        _masked_gaussian_uncertainty_nll(pcargo, log_variance[:, 6:11], ycargo, 5),
        _masked_gaussian_uncertainty_nll(pforce, log_variance[:, 11:14], yforce, 3),
    ]
    return torch.stack(pieces).mean()


def _uncertainty_calibration_stats(prediction, log_variance, target_with_validity, width):
    """Return masked normalized NLL, component coverage at 1σ/2σ and row count."""
    width = int(width)
    valid = ((target_with_validity[:, width] > 0.5)
             & torch.isfinite(target_with_validity[:, :width]).all(dim=-1)
             & torch.isfinite(prediction).all(dim=-1)
             & torch.isfinite(log_variance).all(dim=-1))
    if not bool(valid.any()):
        return {"nll": float("nan"), "coverage_1sigma": float("nan"),
                "coverage_2sigma": float("nan"), "count": 0}
    err = (prediction.detach()[valid] - target_with_validity[valid, :width]).abs()
    lv = log_variance[valid].clamp(PHYSICS_UNCERTAINTY_LOGVAR_MIN,
                                  PHYSICS_UNCERTAINTY_LOGVAR_MAX)
    sigma = torch.exp(0.5 * lv)
    nll = 0.5 * (lv + err.square() * torch.exp(-lv))
    return {"nll": float(nll.mean().detach().item()),
            "coverage_1sigma": float((err <= sigma).float().mean().detach().item()),
            "coverage_2sigma": float((err <= (2.0 * sigma)).float().mean().detach().item()),
            "count": int(valid.sum().item())}


def _uncertainty_summary_text(label, prediction, log_variance, target, width):
    stats = _uncertainty_calibration_stats(prediction, log_variance, target, width)
    if stats["count"] == 0:
        return f"{label}: n/a"
    return (f"{label}: NLL={stats['nll']:.4f}, 1sigma={100.0*stats['coverage_1sigma']:.1f}%"
            f", 2sigma={100.0*stats['coverage_2sigma']:.1f}%, n={stats['count']}")


def _kinematics_prediction_loss(prediction, target_with_validity):
    """Masked MSE for one-step [mean velocity/8, acceleration/12] targets.

    The final target column is 1 only when the measured simulator transition stayed
    inside the harvesting validity envelope. Divergent transitions are retained for
    action learning but do not corrupt the auxiliary physical-prediction objective.
    """
    if prediction.ndim != 2 or prediction.shape[-1] != 6:
        raise ValueError(f"kinematics prediction must have shape [N,6], got {tuple(prediction.shape)}")
    if target_with_validity.ndim != 2 or target_with_validity.shape[-1] != 7:
        raise ValueError(f"kinematics target must have shape [N,7], got {tuple(target_with_validity.shape)}")
    valid = (target_with_validity[:, 6] > 0.5) & torch.isfinite(target_with_validity[:, :6]).all(dim=-1)
    if not bool(valid.any()):
        # Differentiable zero keeps callers' training loops simple on rare all-invalid batches.
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[valid], target_with_validity[valid, :6])


def _kinematic_equation_residual_error(prediction, observation, target_with_validity, dt):
    """Return normalized p_dot=v integration-residual error and a validity mask.

    For one transition, the finite-step mean velocity should be related to the
    starting velocity and acceleration. We do not force the residual to zero:
    the supervised target records the actual residual of each simulator/integrator,
    which can be non-zero under changing acceleration and discrete integration.

    prediction columns are [mean_velocity / 8, acceleration / 12]. The observation
    is the full HISTORY_LEN x BASE_OBS_DIM stack, with latest-frame velocity at
    feature indices 6:9 normalized by 8 m/s. `dt` MUST match the source simulator.
    Returns component-wise residual error normalized by 8 m/s, plus row validity.
    """
    if prediction.ndim != 2 or prediction.shape[-1] != 6:
        raise ValueError(f"kinematics prediction must have shape [N,6], got {tuple(prediction.shape)}")
    if observation.ndim != 2 or observation.shape[-1] != OBSERVATION_DIM:
        raise ValueError(f"observation must have shape [N,{OBSERVATION_DIM}], got {tuple(observation.shape)}")
    if target_with_validity.ndim != 2 or target_with_validity.shape[-1] != 7:
        raise ValueError(f"kinematics target must have shape [N,7], got {tuple(target_with_validity.shape)}")
    if prediction.shape[0] != observation.shape[0] or prediction.shape[0] != target_with_validity.shape[0]:
        raise ValueError("prediction, observation, and kinematics target batch lengths must match")
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0.0 or dt > 0.1:
        raise ValueError(f"dt must be finite and in (0, 0.1] seconds, got {dt!r}")

    latest = (HISTORY_LEN - 1) * BASE_OBS_DIM
    v_before = observation[:, latest + 6:latest + 9] * 8.0
    pred_vbar = prediction[:, :3] * 8.0
    pred_accel = prediction[:, 3:6] * 12.0
    target_vbar = target_with_validity[:, :3] * 8.0
    target_accel = target_with_validity[:, 3:6] * 12.0

    predicted_residual = (pred_vbar - v_before - 0.5 * dt * pred_accel) / 8.0
    measured_residual = (target_vbar - v_before - 0.5 * dt * target_accel) / 8.0
    residual_error = predicted_residual - measured_residual
    valid = ((target_with_validity[:, 6] > 0.5)
             & torch.isfinite(target_with_validity[:, :6]).all(dim=-1)
             & torch.isfinite(prediction).all(dim=-1)
             & torch.isfinite(observation[:, latest + 6:latest + 9]).all(dim=-1))
    return residual_error, valid


def _kinematic_equation_residual_loss(prediction, observation, target_with_validity, dt):
    """Soft supervised consistency loss for the finite-step kinematic equation."""
    residual_error, valid = _kinematic_equation_residual_error(
        prediction, observation, target_with_validity, dt)
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return F.mse_loss(residual_error[valid], torch.zeros_like(residual_error[valid]))


def _kinematic_equation_residual_rmse_mps(prediction, observation, target_with_validity, dt):
    """Physical-unit RMSE between predicted and measured integration residuals."""
    residual_error, valid = _kinematic_equation_residual_error(
        prediction, observation, target_with_validity, dt)
    if not bool(valid.any()):
        return float("nan")
    return float((torch.sqrt(torch.mean(residual_error[valid] ** 2)) * 8.0).detach().item())


def _numpy_one_step_kinematics_target(position_before, velocity_before, position_after,
                                       velocity_after, dt, angular_velocity=None):
    """Create a clipped, masked transition target from two authoritative sim states."""
    dt = max(1e-6, float(dt))
    pb = np.asarray(position_before, dtype=np.float64).reshape(3)
    vb = np.asarray(velocity_before, dtype=np.float64).reshape(3)
    pa = np.asarray(position_after, dtype=np.float64).reshape(3)
    va = np.asarray(velocity_after, dtype=np.float64).reshape(3)
    avg_velocity = (pa - pb) / dt
    acceleration = (va - vb) / dt
    valid = (np.all(np.isfinite(np.concatenate([pb, vb, pa, va, avg_velocity, acceleration])))
                     and -5.0 <= pa[2] <= 80.0
                     and abs(float(va[2])) <= 8.0
                     and float(np.linalg.norm(va)) <= 45.0
                     and float(np.linalg.norm(acceleration)) <= 60.0)
    values = np.concatenate([np.clip(avg_velocity / 8.0, -2.0, 2.0),
                             np.clip(acceleration / 12.0, -3.0, 3.0)])
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.concatenate([values, np.array([1.0 if valid else 0.0], dtype=np.float32)])


def _build_action_condition_tensor(guidance, fsm, winch):
    """Encode the paired high-level action applied during a measured transition."""
    if guidance.ndim != 2 or guidance.shape[1] < 3:
        raise ValueError(f"guidance must have shape [N,>=3], got {tuple(guidance.shape)}")
    if fsm.ndim != 1 or fsm.shape[0] != guidance.shape[0]:
        raise ValueError("FSM action labels must be a length-N vector aligned to guidance")
    if winch.ndim != 2 or winch.shape[0] != guidance.shape[0] or winch.shape[1] < 1:
        raise ValueError("winch labels must have shape [N,>=1] and align with guidance")
    fsm_index = torch.clamp(fsm.to(torch.long), 0, NUM_EXECUTIVE_STATES - 1)
    fsm_onehot = F.one_hot(fsm_index, num_classes=NUM_EXECUTIVE_STATES).to(dtype=guidance.dtype)
    action = torch.cat((guidance[:, :3], fsm_onehot, torch.clamp(winch[:, :1], -1.0, 1.0)), dim=-1)
    return torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)


def _numpy_one_step_action_transition_target(position_before, velocity_before, omega_before_body,
                                              position_after, velocity_after, omega_after_body,
                                              dt, transition_valid, contact_free):
    """Create a paired one-step delta-state target with explicit mask.

    Target order: delta position / 0.25 m, delta world velocity / 1 m/s,
    delta body angular rate / 2 rad/s, valid. It pairs the current observation and
    high-level action with the immediately following authoritative BEMT state.
    """
    dt = float(dt)
    vectors = [np.asarray(v, dtype=np.float64).reshape(3) for v in
               (position_before, velocity_before, omega_before_body,
                position_after, velocity_after, omega_after_body)]
    p0, v0, w0, p1, v1, w1 = vectors
    dp, dv, dw = p1 - p0, v1 - v0, w1 - w0
    finite = np.all(np.isfinite(np.concatenate(vectors + [dp, dv, dw])))
    valid = bool(
        transition_valid and contact_free and finite
        and math.isfinite(dt) and 0.0 < dt <= 0.05
        and -5.0 <= p0[2] <= 80.0 and -5.0 <= p1[2] <= 80.0
        and float(np.linalg.norm(v0)) <= 45.0 and float(np.linalg.norm(v1)) <= 45.0
        and float(np.linalg.norm(w0)) <= 80.0 and float(np.linalg.norm(w1)) <= 80.0
        and float(np.linalg.norm(dp)) <= ACTION_TRANSITION_MAX_DELTA_POSITION_M
        and float(np.linalg.norm(dv)) <= ACTION_TRANSITION_MAX_DELTA_VELOCITY_MS
        and float(np.linalg.norm(dw)) <= ACTION_TRANSITION_MAX_DELTA_OMEGA_RADS
    )
    values = np.concatenate((
        np.clip(dp / ACTION_TRANSITION_POSITION_SCALE_M, -4.0, 4.0),
        np.clip(dv / ACTION_TRANSITION_VELOCITY_SCALE_MS, -6.0, 6.0),
        np.clip(dw / ACTION_TRANSITION_ANGULAR_RATE_SCALE_RADS, -4.0, 4.0)))
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if not valid:
        values.fill(0.0)
    return np.concatenate((values, np.array([1.0 if valid else 0.0], dtype=np.float32)))


def _action_conditioned_transition_loss(prediction, target_with_validity):
    """Masked one-step state-delta loss, normalized to the target scales."""
    if prediction.ndim != 2 or prediction.shape[-1] != ACTION_TRANSITION_OUTPUT_DIM:
        raise ValueError(f"action-conditioned transition prediction must be [N,{ACTION_TRANSITION_OUTPUT_DIM}]")
    if target_with_validity.ndim != 2 or target_with_validity.shape[-1] != ACTION_TRANSITION_TARGET_DIM:
        raise ValueError(f"action-conditioned transition target must be [N,{ACTION_TRANSITION_TARGET_DIM}]")
    valid = ((target_with_validity[:, -1] > 0.5)
             & torch.isfinite(target_with_validity[:, :-1]).all(dim=-1)
             & torch.isfinite(prediction).all(dim=-1))
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[valid], target_with_validity[valid, :ACTION_TRANSITION_OUTPUT_DIM])


def _action_conditioned_transition_validation_stats(prediction, target_with_validity):
    """Validation RMSE in SI units for delta-position, delta-velocity and delta-body-rate."""
    valid = ((target_with_validity[:, -1] > 0.5)
             & torch.isfinite(target_with_validity[:, :-1]).all(dim=-1)
             & torch.isfinite(prediction).all(dim=-1))
    count = int(valid.sum().item())
    if not count:
        return {"count": 0, "delta_position_rmse_m": float("nan"),
                "delta_velocity_rmse_mps": float("nan"), "delta_omega_rmse_rads": float("nan")}
    errors = prediction[valid] - target_with_validity[valid, :ACTION_TRANSITION_OUTPUT_DIM]
    pos_rmse = float(torch.sqrt(torch.mean(errors[:, 0:3].square())).item()
                     * ACTION_TRANSITION_POSITION_SCALE_M)
    vel_rmse = float(torch.sqrt(torch.mean(errors[:, 3:6].square())).item()
                     * ACTION_TRANSITION_VELOCITY_SCALE_MS)
    omega_rmse = float(torch.sqrt(torch.mean(errors[:, 6:9].square())).item()
                        * ACTION_TRANSITION_ANGULAR_RATE_SCALE_RADS)
    return {"count": count, "delta_position_rmse_m": pos_rmse,
            "delta_velocity_rmse_mps": vel_rmse, "delta_omega_rmse_rads": omega_rmse}


def _action_conditioned_transition_constant_velocity_baseline_stats(
        observation, target_with_validity, dt=0.016):
    """Evaluate a transparent one-step baseline on the exact same held-out rows.

    Baseline assumption: position advances at the current observed world velocity;
    velocity and body angular rate remain unchanged over this one 16 ms step.
    Returns the same SI-unit metric schema as the learned transition head.
    This is a comparison baseline, not a physically complete simulator.
    """
    if observation.ndim != 2 or observation.shape[1] != OBSERVATION_DIM:
        raise ValueError(f"observation must be [N,{OBSERVATION_DIM}]")
    if target_with_validity.ndim != 2 or target_with_validity.shape != (observation.shape[0], ACTION_TRANSITION_TARGET_DIM):
        raise ValueError("transition target must be [N,ACTION_TRANSITION_TARGET_DIM] and align with observations")
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0.0 or dt > 0.1:
        raise ValueError(f"dt must be finite and in (0, 0.1], got {dt!r}")
    current = (HISTORY_LEN - 1) * BASE_OBS_DIM
    velocity = observation[:, current + 6:current + 9] * 8.0
    baseline = torch.zeros((observation.shape[0], ACTION_TRANSITION_OUTPUT_DIM),
                           dtype=observation.dtype, device=observation.device)
    # Position-delta output is normalized by 0.25 m; the observed velocity feature is normalized by 8 m/s.
    baseline[:, 0:3] = (velocity * dt) / ACTION_TRANSITION_POSITION_SCALE_M
    baseline = torch.where(torch.isfinite(velocity).all(dim=-1, keepdim=True), baseline,
                           torch.full_like(baseline, float("nan")))
    return _action_conditioned_transition_validation_stats(baseline, target_with_validity)


def _numpy_action_transition_constant_velocity_baseline_si(observation, dt=0.016):
    """Return the SI-unit one-step constant-velocity baseline from policy-visible features.

    This deliberately decodes the latest observation velocity feature, matching
    `_action_conditioned_transition_constant_velocity_baseline_stats`. It must not
    use the simulator's hidden true velocity: doing so would give the live baseline
    information unavailable to the learned predictor and bias the comparison.
    Returns an [N, 9] array: [delta-position m, delta-velocity m/s, delta-body-rate rad/s].
    """
    obs = np.asarray(observation, dtype=np.float64)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if obs.ndim != 2 or obs.shape[1] != OBSERVATION_DIM:
        raise ValueError(f"observation must be [N,{OBSERVATION_DIM}] or [{OBSERVATION_DIM}], got {obs.shape}")
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0.0 or dt > 0.1:
        raise ValueError(f"dt must be finite and in (0, 0.1], got {dt!r}")
    current = (HISTORY_LEN - 1) * BASE_OBS_DIM
    velocity_obs = obs[:, current + 6:current + 9] * 8.0
    baseline = np.zeros((obs.shape[0], ACTION_TRANSITION_OUTPUT_DIM), dtype=np.float64)
    baseline[:, 0:3] = velocity_obs * dt
    invalid = ~np.isfinite(velocity_obs).all(axis=1)
    baseline[invalid] = np.nan
    return baseline


def _action_transition_action_input_stats(action_condition, target_with_validity):
    """Summarize action-input coverage on valid paired transition rows.

    Values describe numeric inputs supplied to the transition model, not physical
    actuator outputs. The distinction matters because live guidance is scaled and
    phase-gated differently from teacher guidance during harvest.
    """
    action_np = (action_condition.detach().cpu().numpy() if torch.is_tensor(action_condition)
                 else np.asarray(action_condition))
    target_np = (target_with_validity.detach().cpu().numpy() if torch.is_tensor(target_with_validity)
                 else np.asarray(target_with_validity))
    if action_np.ndim != 2 or action_np.shape[1] != ACTION_TRANSITION_ACTION_DIM:
        raise ValueError(f"action_condition must be [N,{ACTION_TRANSITION_ACTION_DIM}], got {action_np.shape}")
    if target_np.ndim != 2 or target_np.shape != (action_np.shape[0], ACTION_TRANSITION_TARGET_DIM):
        raise ValueError("target_with_validity must align one-to-one with action_condition")
    finite = np.isfinite(action_np).all(axis=1) & np.isfinite(target_np[:, :-1]).all(axis=1)
    valid = (target_np[:, -1] > 0.5) & finite
    guidance = np.asarray(action_np[valid, :3], dtype=np.float64)
    fsm_values = np.asarray(action_np[valid, 3:3 + NUM_EXECUTIVE_STATES], dtype=np.float64)
    winch = np.asarray(action_np[valid, -1], dtype=np.float64)
    count = int(guidance.shape[0])
    if count:
        norms = np.linalg.norm(guidance, axis=1)
        fsm_ids = np.argmax(fsm_values, axis=1)
        fsm_counts = {ACTION_NAMES[int(i)]: int(np.sum(fsm_ids == i)) for i in range(NUM_EXECUTIVE_STATES)
                      if np.any(fsm_ids == i)}
        guidance_stats = {
            "norm_mean": float(np.mean(norms)),
            "norm_p50": float(np.percentile(norms, 50)),
            "norm_p95": float(np.percentile(norms, 95)),
            "norm_max": float(np.max(norms)),
            "exact_zero_fraction": float(np.mean(norms <= 1e-9)),
            "component_abs_p95": float(np.percentile(np.abs(guidance), 95)),
        }
        winch_stats = {"mean": float(np.mean(winch)), "min": float(np.min(winch)),
                       "max": float(np.max(winch)), "payout_fraction": float(np.mean(winch > 0.05)),
                       "retract_fraction": float(np.mean(winch < -0.05)),
                       "neutral_fraction": float(np.mean(np.abs(winch) <= 0.05))}
    else:
        guidance_stats = {"norm_mean": None, "norm_p50": None, "norm_p95": None,
                          "norm_max": None, "exact_zero_fraction": None, "component_abs_p95": None}
        winch_stats = {"mean": None, "min": None, "max": None,
                       "payout_fraction": None, "retract_fraction": None, "neutral_fraction": None}
        fsm_counts = {}
    return {
        "rows_total": int(action_np.shape[0]),
        "valid_paired_rows": count,
        "invalid_or_masked_rows": int(action_np.shape[0] - count),
        "guidance_features": guidance_stats,
        "fsm_action_counts": fsm_counts,
        "winch_input": winch_stats,
    }


def _action_transition_relative_gain_pct(baseline_rmse, learned_rmse):
    """Positive percentages mean the learned head reduced RMSE versus baseline."""
    try:
        baseline_rmse = float(baseline_rmse)
        learned_rmse = float(learned_rmse)
    except (TypeError, ValueError):
        return float("nan")
    if not (math.isfinite(baseline_rmse) and math.isfinite(learned_rmse)):
        return float("nan")
    if baseline_rmse <= 1e-9:
        return 0.0 if learned_rmse <= 1e-9 else float("-inf")
    return 100.0 * (baseline_rmse - learned_rmse) / baseline_rmse



def _action_transition_stratified_validation_stats(
        prediction, target_with_validity, observation, action_condition,
        min_group_samples=32, dt=0.016):
    """Break held-out transition errors down by FSM action, winch command and sensed wind.

    All groups use the same held-out rows and the same constant-velocity baseline as
    the overall metric. Small groups are reported as insufficient rather than silently
    treated as evidence. Wind is read from the current 139-D observation frame's
    documented normalized XYZ wind features (indices 42:45, scaled by 8 m/s).
    """
    n = int(prediction.shape[0]) if getattr(prediction, "ndim", 0) == 2 else -1
    if n < 0 or target_with_validity.shape != (n, ACTION_TRANSITION_TARGET_DIM):
        raise ValueError("stratified transition validation requires aligned [N,output/target] tensors")
    if observation.ndim != 2 or observation.shape != (n, OBSERVATION_DIM):
        raise ValueError("stratified validation observation shape mismatch")
    if action_condition.ndim != 2 or action_condition.shape != (n, ACTION_TRANSITION_ACTION_DIM):
        raise ValueError("stratified validation action-condition shape mismatch")
    min_group_samples = max(1, int(min_group_samples))
    valid = ((target_with_validity[:, -1] > 0.5)
             & torch.isfinite(target_with_validity[:, :-1]).all(dim=-1)
             & torch.isfinite(prediction).all(dim=-1))

    fsm_start = 3
    fsm_end = fsm_start + NUM_EXECUTIVE_STATES
    fsm = action_condition[:, fsm_start:fsm_end].argmax(dim=-1)
    winch_cmd = action_condition[:, -1]
    c_idx = (HISTORY_LEN - 1) * BASE_OBS_DIM
    wind_xyz = observation[:, c_idx + 42:c_idx + 45] * 8.0
    wind_speed = torch.linalg.vector_norm(wind_xyz, dim=-1)

    groups = []
    for action_idx in range(NUM_EXECUTIVE_STATES):
        action_name = ACTION_NAMES[action_idx] if action_idx < len(ACTION_NAMES) else f"FSM_{action_idx}"
        groups.append(("fsm", action_name, fsm == action_idx))
    groups.extend([
        ("winch", "retract", winch_cmd < -0.05),
        ("winch", "neutral", (winch_cmd >= -0.05) & (winch_cmd <= 0.05)),
        ("winch", "payout", winch_cmd > 0.05),
        ("wind", "calm_lt_2p5mps", wind_speed < 2.5),
        ("wind", "moderate_2p5_to_5p5mps", (wind_speed >= 2.5) & (wind_speed < 5.5)),
        ("wind", "strong_ge_5p5mps", wind_speed >= 5.5),
    ])

    results = []
    for group_type, group_name, member_mask in groups:
        mask = valid & member_mask & torch.isfinite(wind_speed)
        count = int(mask.sum().item())
        entry = {"group_type": group_type, "group": group_name, "count": count,
                 "min_group_samples": min_group_samples, "adequate": count >= min_group_samples}
        if count < min_group_samples:
            results.append(entry)
            continue
        pred_subset = prediction[mask]
        target_subset = target_with_validity[mask]
        obs_subset = observation[mask]
        model_stats = _action_conditioned_transition_validation_stats(pred_subset, target_subset)
        baseline_stats = _action_conditioned_transition_constant_velocity_baseline_stats(
            obs_subset, target_subset, dt=dt)
        entry["learned"] = model_stats
        entry["constant_velocity_baseline"] = baseline_stats
        entry["gain_vs_baseline_pct"] = {
            "position": _action_transition_relative_gain_pct(
                baseline_stats["delta_position_rmse_m"], model_stats["delta_position_rmse_m"]),
            "velocity": _action_transition_relative_gain_pct(
                baseline_stats["delta_velocity_rmse_mps"], model_stats["delta_velocity_rmse_mps"]),
            "angular_rate": _action_transition_relative_gain_pct(
                baseline_stats["delta_omega_rmse_rads"], model_stats["delta_omega_rmse_rads"]),
        }
        results.append(entry)
    return results


def _action_transition_action_sensitivity_probe(model, observation, action_condition,
                                                target_with_validity=None, max_samples=128):
    """Measure whether model outputs respond to action-input perturbations.

    This is a sensitivity diagnostic, not a correctness test: counterfactual labels
    for the perturbed action are unavailable. It must not be interpreted as proof that
    changing a command will produce the predicted response in the real simulator.
    """
    if observation.ndim != 2 or observation.shape[1] != OBSERVATION_DIM:
        raise ValueError("sensitivity probe observations must be [N, OBSERVATION_DIM]")
    if action_condition.shape != (observation.shape[0], ACTION_TRANSITION_ACTION_DIM):
        raise ValueError("sensitivity probe actions must align with observations")
    if target_with_validity is not None:
        if target_with_validity.shape != (observation.shape[0], ACTION_TRANSITION_TARGET_DIM):
            raise ValueError("sensitivity probe targets must align with observations")
        valid = ((target_with_validity[:, -1] > 0.5)
                 & torch.isfinite(target_with_validity[:, :-1]).all(dim=-1)
                 & torch.isfinite(observation).all(dim=-1)
                 & torch.isfinite(action_condition).all(dim=-1))
        indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
    else:
        valid = torch.isfinite(observation).all(dim=-1) & torch.isfinite(action_condition).all(dim=-1)
        indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if indices.numel() == 0:
        return {"count": 0, "guidance_perturbation": None, "fsm_perturbation": None}
    take = min(max(1, int(max_samples)), int(indices.numel()))
    # Evenly spaced rows reduce a bias toward the earliest part of a long validation rollout.
    picks = torch.linspace(0, indices.numel() - 1, steps=take, device=indices.device).long()
    indices = indices[picks]
    obs = observation[indices]
    actions = action_condition[indices].clone()

    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            def predict(action_tensor):
                result = model(
                    obs, return_physics=True, return_rotational_residual=True,
                    return_uncertainty=True, return_trajectory_forecast=True,
                    return_action_transition=True, transition_action=action_tensor)
                return result[-2]

            baseline = predict(actions)
            guidance_probe = actions.clone()
            row_sign = torch.where(
                torch.arange(take, device=obs.device) % 2 == 0,
                torch.ones(take, dtype=obs.dtype, device=obs.device),
                -torch.ones(take, dtype=obs.dtype, device=obs.device))
            guidance_probe[:, 0] = torch.clamp(guidance_probe[:, 0] + 0.25 * row_sign, -1.0, 1.0)
            guidance_changed = predict(guidance_probe)

            fsm_probe = actions.clone()
            fsm_start = 3
            fsm_end = fsm_start + NUM_EXECUTIVE_STATES
            original_fsm = fsm_probe[:, fsm_start:fsm_end].argmax(dim=-1)
            next_fsm = (original_fsm + 1) % max(1, min(6, NUM_EXECUTIVE_STATES))
            fsm_probe[:, fsm_start:fsm_end] = F.one_hot(
                next_fsm, num_classes=NUM_EXECUTIVE_STATES).to(dtype=fsm_probe.dtype)
            fsm_changed = predict(fsm_probe)

            def response_metrics(changed):
                delta = changed - baseline
                dp = delta[:, 0:3] * ACTION_TRANSITION_POSITION_SCALE_M
                dv = delta[:, 3:6] * ACTION_TRANSITION_VELOCITY_SCALE_MS
                dw = delta[:, 6:9] * ACTION_TRANSITION_ANGULAR_RATE_SCALE_RADS
                return {
                    "predicted_delta_position_response_rms_m": float(torch.sqrt(torch.mean(torch.sum(dp.square(), dim=-1))).item()),
                    "predicted_delta_velocity_response_rms_mps": float(torch.sqrt(torch.mean(torch.sum(dv.square(), dim=-1))).item()),
                    "predicted_delta_angular_rate_response_rms_rads": float(torch.sqrt(torch.mean(torch.sum(dw.square(), dim=-1))).item()),
                }
            return {"count": int(take), "guidance_perturbation": response_metrics(guidance_changed),
                    "fsm_perturbation": response_metrics(fsm_changed),
                    "interpretation": "response magnitude only; not counterfactual accuracy"}
    finally:
        model.train(was_training)


def _action_transition_rollout_readiness_gate(
        valid_transition_rows, overall_gain_vs_baseline_pct, stratified_groups,
        action_sensitivity, min_overall_rows=1000, min_group_samples=32,
        min_overall_gain_pct=0.0, min_critical_group_gain_pct=0.0,
        sensitivity_floor=1e-7):
    """Conservatively decide whether to begin OFFLINE multi-step rollout evaluation.

    Passing only authorizes offline evaluation, not controller deployment. Gates require
    enough paired held-out samples, non-negative gains over the constant-velocity baseline
    in every overall output family and each critical state group, plus non-collapsed action
    sensitivity. Sensitivity is not counterfactual correctness and is used only as a basic
    collapse check.
    """
    reasons = []
    checks = {}
    try:
        n = int(valid_transition_rows)
    except (TypeError, ValueError, OverflowError):
        n = -1
    checks["overall_sample_count"] = {"value": n, "minimum": int(min_overall_rows),
                                      "passed": n >= int(min_overall_rows)}
    if n < int(min_overall_rows):
        reasons.append(f"valid held-out transition count {n} is below {int(min_overall_rows)}")

    metric_gains = {}
    for metric in ("position", "velocity", "angular_rate"):
        try:
            gain = float(overall_gain_vs_baseline_pct.get(metric, float("nan")))
        except (AttributeError, TypeError, ValueError):
            gain = float("nan")
        passed = math.isfinite(gain) and gain >= float(min_overall_gain_pct)
        metric_gains[metric] = gain if math.isfinite(gain) else None
        checks[f"overall_gain_{metric}"] = {
            "value_pct": metric_gains[metric], "minimum_pct": float(min_overall_gain_pct), "passed": passed}
        if not passed:
            reasons.append(f"overall {metric} gain is unavailable or below {float(min_overall_gain_pct):.1f}%")

    required_groups = {
        ("fsm", "HOVER_LOCK"),
        ("fsm", "WINCH_LOWER"),
        ("fsm", "WINCH_RETRACT"),
        ("fsm", "DEPOT_SERVICE"),
        ("winch", "payout"),
        ("winch", "retract"),
        ("wind", "strong_ge_5p5mps"),
    }
    by_key = {}
    for entry in stratified_groups if isinstance(stratified_groups, (list, tuple)) else []:
        if isinstance(entry, dict):
            by_key[(str(entry.get("group_type", "")), str(entry.get("group", "")))] = entry

    critical_report = {}
    for key in sorted(required_groups):
        entry = by_key.get(key)
        label = f"{key[0]}:{key[1]}"
        if (not isinstance(entry, dict) or not entry.get("adequate", False)
                or int(entry.get("count", 0)) < int(min_group_samples)):
            count = int(entry.get("count", 0)) if isinstance(entry, dict) else 0
            critical_report[label] = {"count": count, "passed": False, "reason": "missing or insufficient validation samples"}
            reasons.append(f"critical subgroup {label} is missing or has insufficient samples")
            continue
        gains = entry.get("gain_vs_baseline_pct", {})
        try:
            gp = float(gains.get("position", float("nan")))
            gv = float(gains.get("velocity", float("nan")))
            gw = float(gains.get("angular_rate", float("nan")))
        except (AttributeError, TypeError, ValueError):
            gp = gv = gw = float("nan")
        pass_each = {
            "position": math.isfinite(gp) and gp >= float(min_critical_group_gain_pct),
            "velocity": math.isfinite(gv) and gv >= float(min_critical_group_gain_pct),
            "angular_rate": math.isfinite(gw) and gw >= float(min_critical_group_gain_pct),
        }
        group_pass = all(pass_each.values())
        critical_report[label] = {
            "count": int(entry.get("count", 0)),
            "gain_vs_baseline_pct": {
                "position": gp if math.isfinite(gp) else None,
                "velocity": gv if math.isfinite(gv) else None,
                "angular_rate": gw if math.isfinite(gw) else None,
            },
            "metric_pass": pass_each,
            "passed": group_pass,
        }
        if not group_pass:
            reasons.append(f"critical subgroup {label} regresses or lacks a finite metric")

    sensitivity_checks = {}
    for perturbation_name in ("guidance_perturbation", "fsm_perturbation"):
        perturbation = action_sensitivity.get(perturbation_name, {}) if isinstance(action_sensitivity, dict) else {}
        metric_names = (
            "predicted_delta_position_response_rms_m",
            "predicted_delta_velocity_response_rms_mps",
            "predicted_delta_angular_rate_response_rms_rads",
        )
        values = []
        for name in metric_names:
            try:
                value = float(perturbation.get(name, float("nan")))
            except (AttributeError, TypeError, ValueError):
                value = float("nan")
            if math.isfinite(value):
                values.append(max(0.0, value))
        response = max(values) if values else float("nan")
        passed = math.isfinite(response) and response > float(sensitivity_floor)
        sensitivity_checks[perturbation_name] = {
            "max_response_rms": response if math.isfinite(response) else None,
            "minimum_response_rms": float(sensitivity_floor), "passed": passed}
        if not passed:
            reasons.append(f"action sensitivity probe {perturbation_name} is missing or collapsed")

    ready = len(reasons) == 0
    return {
        "status": "READY_FOR_OFFLINE_MULTISTEP_EVALUATION" if ready else "BLOCKED",
        "ready_for_offline_multistep_evaluation": ready,
        "live_control_authorized": False,
        "minimum_overall_rows": int(min_overall_rows),
        "minimum_group_samples": int(min_group_samples),
        "minimum_overall_gain_pct": float(min_overall_gain_pct),
        "minimum_critical_group_gain_pct": float(min_critical_group_gain_pct),
        "overall_gain_vs_baseline_pct": metric_gains,
        "checks": checks,
        "critical_subgroups": critical_report,
        "action_sensitivity_checks": sensitivity_checks,
        "reasons": reasons,
        "limitations": [
            "Passing permits offline multi-step evaluation only; it never authorizes live control.",
            "Action sensitivity is not counterfactual accuracy.",
            "One-step validation does not prove recursive multi-step accuracy.",
            "A complete trained V38.35 validation run is required to compute a real readiness result."
        ]
    }


def _action_transition_json_safe(value):
    """Convert nested report values to strict JSON types; non-finite metrics become null."""
    if isinstance(value, dict):
        return {str(k): _action_transition_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_action_transition_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        fvalue = float(value)
        return fvalue if math.isfinite(fvalue) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (int, str)) or value is None:
        return value
    return str(value)


def _numpy_rotational_dynamics_residual_target(omega_before_body, omega_after_body,
                                               modeled_motor_torque_body, dt,
                                               transition_valid, contact_free):
    """Build [Euler torque residual / scale_Nm, valid] from an authoritative BEMT step.

    Rigid-body Euler equation in body coordinates:
        tau_motor + tau_unmodeled = J * alpha + omega x (J * omega)
    The stored target is tau_unmodeled = J * alpha_observed + omega x (J*omega)
    - tau_motor. `alpha_observed` is the finite-step change in body-frame angular
    velocity divided by dt, so the target intentionally includes small RK4/finite-
    step mismatch. This is a simulator residual, not a guarantee about real hardware.
    All torques are body-frame N*m and omega is body-frame rad/s.
    """
    dt = float(dt)
    w0 = np.asarray(omega_before_body, dtype=np.float64).reshape(3)
    w1 = np.asarray(omega_after_body, dtype=np.float64).reshape(3)
    tau_motor = np.asarray(modeled_motor_torque_body, dtype=np.float64).reshape(3)
    inertia = np.diag(INERTIA_MATRIX).astype(np.float64)
    if not math.isfinite(dt) or dt <= 0.0 or dt > 0.1:
        raise ValueError(f"dt must be finite and in (0, 0.1] seconds, got {dt!r}")
    alpha_observed = (w1 - w0) / dt
    angular_momentum = inertia * w0
    torque_required = inertia * alpha_observed + np.cross(w0, angular_momentum)
    residual_nm = torque_required - tau_motor
    finite = np.all(np.isfinite(np.concatenate((w0, w1, tau_motor, alpha_observed, residual_nm))))
    valid = bool(
        transition_valid and contact_free and finite
        and float(np.linalg.norm(w0)) <= ROTATIONAL_DYNAMICS_MAX_ANGULAR_SPEED_RAD_S
        and float(np.linalg.norm(w1)) <= ROTATIONAL_DYNAMICS_MAX_ANGULAR_SPEED_RAD_S
        and float(np.linalg.norm(alpha_observed)) <= ROTATIONAL_DYNAMICS_MAX_ANGULAR_ACCEL_RAD_S2
        and float(np.linalg.norm(residual_nm)) <= ROTATIONAL_DYNAMICS_MAX_RESIDUAL_NM
    )
    scaled = np.clip(residual_nm / ROTATIONAL_DYNAMICS_RESIDUAL_SCALE_NM, -3.0, 3.0)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if not valid:
        scaled.fill(0.0)
    return np.concatenate((scaled, np.array([1.0 if valid else 0.0], dtype=np.float32)))


def _rotational_dynamics_residual_prediction_loss(prediction, target_with_validity):
    """Masked MSE for predicted Euler torque residual [tau_x,y,z]/0.5 N*m."""
    if prediction.ndim != 2 or prediction.shape[-1] != 3:
        raise ValueError(f"rotational residual prediction must have shape [N,3], got {tuple(prediction.shape)}")
    if target_with_validity.ndim != 2 or target_with_validity.shape[-1] != 4:
        raise ValueError(f"rotational residual target must have shape [N,4], got {tuple(target_with_validity.shape)}")
    if prediction.shape[0] != target_with_validity.shape[0]:
        raise ValueError("rotational residual prediction and target batch lengths must match")
    valid = ((target_with_validity[:, 3] > 0.5)
             & torch.isfinite(target_with_validity[:, :3]).all(dim=-1)
             & torch.isfinite(prediction).all(dim=-1))
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[valid], target_with_validity[valid, :3])


def _rotational_dynamics_residual_error_nm(prediction, target_with_validity):
    """Return normalized-head error and target in N*m plus a valid-row mask."""
    if prediction.ndim != 2 or prediction.shape[-1] != 3:
        raise ValueError(f"rotational residual prediction must have shape [N,3], got {tuple(prediction.shape)}")
    if target_with_validity.ndim != 2 or target_with_validity.shape[-1] != 4:
        raise ValueError(f"rotational residual target must have shape [N,4], got {tuple(target_with_validity.shape)}")
    if prediction.shape[0] != target_with_validity.shape[0]:
        raise ValueError("rotational residual prediction and target batch lengths must match")
    valid = ((target_with_validity[:, 3] > 0.5)
             & torch.isfinite(target_with_validity[:, :3]).all(dim=-1)
             & torch.isfinite(prediction).all(dim=-1))
    scale = ROTATIONAL_DYNAMICS_RESIDUAL_SCALE_NM
    prediction_nm = torch.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0) * scale
    target_nm = torch.nan_to_num(target_with_validity[:, :3], nan=0.0, posinf=0.0, neginf=0.0) * scale
    return prediction_nm - target_nm, target_nm, valid


def _rotational_dynamics_residual_rmse_nm(prediction, target_with_validity):
    error_nm, _, valid = _rotational_dynamics_residual_error_nm(prediction, target_with_validity)
    if not bool(valid.any()):
        return float("nan")
    return float(torch.sqrt(torch.mean(error_nm[valid] ** 2)).detach().item())



def _trajectory_forecast_loss(prediction, target_with_validity):
    """Masked multi-horizon future displacement/velocity loss in normalized units."""
    if prediction.ndim != 2 or prediction.shape[-1] != TRAJECTORY_FORECAST_DIM:
        raise ValueError(f"trajectory forecast must have shape [N,{TRAJECTORY_FORECAST_DIM}], got {tuple(prediction.shape)}")
    if target_with_validity.ndim != 2 or target_with_validity.shape[-1] != TRAJECTORY_FORECAST_TARGET_DIM:
        raise ValueError(f"trajectory target must have shape [N,{TRAJECTORY_FORECAST_TARGET_DIM}], got {tuple(target_with_validity.shape)}")
    if prediction.shape[0] != target_with_validity.shape[0]:
        raise ValueError("trajectory forecast prediction and target batch lengths must match")
    horizon_losses = []
    weight_sum = 0.0
    for hidx, weight in enumerate(TRAJECTORY_FORECAST_LOSS_HORIZON_WEIGHTS):
        lo = hidx * TRAJECTORY_FORECAST_VALUES_PER_HORIZON
        hi = lo + TRAJECTORY_FORECAST_VALUES_PER_HORIZON
        valid = ((target_with_validity[:, TRAJECTORY_FORECAST_DIM+hidx] > 0.5)
                 & torch.isfinite(target_with_validity[:, lo:hi]).all(dim=-1))
        if bool(valid.any().item()):
            horizon_losses.append(float(weight) * F.mse_loss(prediction[valid, lo:hi], target_with_validity[valid, lo:hi]))
            weight_sum += float(weight)
    if not horizon_losses:
        return prediction.sum() * 0.0
    return torch.stack(horizon_losses).sum() / max(weight_sum, 1e-12)


def _trajectory_forecast_validation_stats(prediction, target_with_validity):
    """Physical-unit RMSE for future displacement (m) and velocity (m/s), per horizon."""
    if prediction.ndim != 2 or prediction.shape[-1] != TRAJECTORY_FORECAST_DIM:
        raise ValueError(f"trajectory forecast must have shape [N,{TRAJECTORY_FORECAST_DIM}], got {tuple(prediction.shape)}")
    if target_with_validity.ndim != 2 or target_with_validity.shape[-1] != TRAJECTORY_FORECAST_TARGET_DIM:
        raise ValueError(f"trajectory target must have shape [N,{TRAJECTORY_FORECAST_TARGET_DIM}], got {tuple(target_with_validity.shape)}")
    result = []
    for hidx, horizon_s in enumerate(TRAJECTORY_FORECAST_HORIZONS_S):
        lo = hidx * TRAJECTORY_FORECAST_VALUES_PER_HORIZON
        hi = lo + TRAJECTORY_FORECAST_VALUES_PER_HORIZON
        valid = ((target_with_validity[:, TRAJECTORY_FORECAST_DIM+hidx] > 0.5)
                 & torch.isfinite(target_with_validity[:, lo:hi]).all(dim=-1)
                 & torch.isfinite(prediction[:, lo:hi]).all(dim=-1))
        count = int(valid.sum().item())
        if count:
            dpe = (prediction[valid, lo:lo+3] - target_with_validity[valid, lo:lo+3]) * TRAJECTORY_FORECAST_POSITION_SCALE_M
            dve = (prediction[valid, lo+3:hi] - target_with_validity[valid, lo+3:hi]) * TRAJECTORY_FORECAST_VELOCITY_SCALE_MS
            pos_rmse = float(torch.sqrt(torch.mean(dpe.square())).item())
            vel_rmse = float(torch.sqrt(torch.mean(dve.square())).item())
        else:
            pos_rmse = vel_rmse = float("nan")
        result.append({"horizon_s": float(horizon_s), "count": count,
                       "position_delta_rmse_m": pos_rmse, "velocity_rmse_mps": vel_rmse})
    return result


def quintic_time_scaling(time_s, duration_s):
    """Return quintic smooth-step s and its first three time derivatives."""
    duration = float(duration_s)
    time_value = float(time_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be finite and > 0")
    if not math.isfinite(time_value):
        raise ValueError("time_s must be finite")
    u = min(1.0, max(0.0, time_value/duration))
    u2, u3, u4, u5 = u*u, u*u*u, u*u*u*u, u*u*u*u*u
    scale = 10.0*u3 - 15.0*u4 + 6.0*u5
    dscale = (30.0*u2 - 60.0*u3 + 30.0*u4)/duration
    ddscale = (60.0*u - 180.0*u2 + 120.0*u3)/(duration*duration)
    dddscale = (60.0 - 360.0*u + 360.0*u2)/(duration*duration*duration)
    return scale, dscale, ddscale, dddscale


def quintic_trajectory_reference(start_position, target_position, duration_s, time_s):
    """Offline 3D quintic reference; not a replacement for live flight guidance."""
    p0 = np.asarray(start_position, dtype=np.float64).reshape(3)
    p1 = np.asarray(target_position, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(p0)) or not np.all(np.isfinite(p1)):
        raise ValueError("trajectory endpoints must be finite 3-vectors")
    scale, dscale, ddscale, dddscale = quintic_time_scaling(time_s, duration_s)
    delta = p1-p0
    return p0+delta*scale, delta*dscale, delta*ddscale, delta*dddscale


def quintic_duration_for_limits(start_position, target_position, max_speed_mps,
                                max_acceleration_mps2, min_duration_s=0.25):
    """Compute duration from quintic peak speed and acceleration bounds."""
    p0 = np.asarray(start_position, dtype=np.float64).reshape(3)
    p1 = np.asarray(target_position, dtype=np.float64).reshape(3)
    vmax, amax, tmin = float(max_speed_mps), float(max_acceleration_mps2), float(min_duration_s)
    if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
        raise ValueError("trajectory endpoints must be finite")
    if not math.isfinite(vmax) or vmax <= 0 or not math.isfinite(amax) or amax <= 0:
        raise ValueError("max speed and acceleration must be finite and > 0")
    if not math.isfinite(tmin) or tmin <= 0:
        raise ValueError("min_duration_s must be finite and > 0")
    distance = float(np.linalg.norm(p1-p0))
    return max(tmin, _QUINTIC_MAX_DS*distance/vmax,
               math.sqrt(_QUINTIC_MAX_ABS_DDS*distance/amax))


def evaluate_quintic_trajectory(start_position, target_position, duration_s, sample_count=201):
    """Offline analytic profile quality metrics; not a closed-loop mission test."""
    n, duration = int(sample_count), float(duration_s)
    if n < 3 or not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("sample_count must be >= 3 and duration_s finite and > 0")
    p0 = np.asarray(start_position, dtype=np.float64).reshape(3)
    p1 = np.asarray(target_position, dtype=np.float64).reshape(3)
    times = np.linspace(0.0, duration, n)
    states = [quintic_trajectory_reference(p0, p1, duration, t) for t in times]
    positions = np.stack([v[0] for v in states]); velocities = np.stack([v[1] for v in states])
    accelerations = np.stack([v[2] for v in states]); jerks = np.stack([v[3] for v in states])
    delta = p1-p0; distance_sq = max(1e-12, float(np.dot(delta, delta)))
    progress = ((positions-p0) @ delta)/distance_sq
    return {"duration_s": duration, "endpoint_error_m": float(np.linalg.norm(positions[-1]-p1)),
            "start_speed_mps": float(np.linalg.norm(velocities[0])),
            "end_speed_mps": float(np.linalg.norm(velocities[-1])),
            "start_acceleration_mps2": float(np.linalg.norm(accelerations[0])),
            "end_acceleration_mps2": float(np.linalg.norm(accelerations[-1])),
            "peak_speed_mps": float(np.linalg.norm(velocities, axis=1).max()),
            "peak_acceleration_mps2": float(np.linalg.norm(accelerations, axis=1).max()),
            "peak_jerk_mps3": float(np.linalg.norm(jerks, axis=1).max()),
            "monotonic_progress": bool(np.all(np.diff(progress) >= -1e-8)),
            "finite": bool(np.all(np.isfinite(positions)) and np.all(np.isfinite(velocities))
                           and np.all(np.isfinite(accelerations)) and np.all(np.isfinite(jerks)))}


# -----------------------------------------------------------------------------
# V38.30 OFFLINE PREDICTIVE-CONTROL EVALUATOR
# -----------------------------------------------------------------------------
# This sandbox intentionally does NOT use the learned V38.29 forecast head to rank
# candidate actions: that head predicts from the current observation but is not
# action-conditioned. Instead, candidate accelerations are rolled through a small,
# explicit, tunable surrogate. This can test candidate generation, objective weights,
# and constraint plumbing, but is not a replacement for the authoritative BEMT/RK4
# airframe + PBD tether. It must not be connected to the live actuator path without
# cross-validation against the full simulator.
OFFLINE_MPC_DT_S = 0.05
OFFLINE_MPC_HORIZON_S = 1.0
OFFLINE_MPC_MAX_HORIZON_STEPS = 24
OFFLINE_MPC_ACTUATOR_TAU_S = 0.16
OFFLINE_MPC_WIND_DRAG_COEFF = 0.012  # effective 1/m coefficient, surrogate only
OFFLINE_MPC_CABLE_DAMPING_RATIO = 0.12
OFFLINE_MPC_PAYLOAD_TO_DRONE_MASS_RATIO = 0.08
OFFLINE_MPC_DEFAULT_PAYLOAD_MASS_KG = 0.80
OFFLINE_MPC_MAX_HORIZONTAL_ACCEL_MS2 = 2.8
OFFLINE_MPC_MAX_VERTICAL_ACCEL_MS2 = 1.8
OFFLINE_MPC_MAX_SPEED_MS = 8.0
OFFLINE_MPC_MIN_ALTITUDE_M = 0.75
# This is only a broad surrogate-feasibility bound for evaluating en-route actions;
# it is NOT a delivery/release threshold. The live payload-release gates remain stricter.
OFFLINE_MPC_MAX_SWAY_M = 2.50
OFFLINE_MPC_MAX_TENSION_N = 25.0
OFFLINE_MPC_POSITION_COST_WEIGHT = 3.0
OFFLINE_MPC_VELOCITY_COST_WEIGHT = 1.1
OFFLINE_MPC_SWAY_COST_WEIGHT = 3.0
OFFLINE_MPC_SWAY_RATE_COST_WEIGHT = 0.60
OFFLINE_MPC_CONTROL_COST_WEIGHT = 0.035
OFFLINE_MPC_SMOOTHNESS_COST_WEIGHT = 0.10
OFFLINE_MPC_TERMINAL_POSITION_WEIGHT = 5.0
OFFLINE_MPC_TERMINAL_VELOCITY_WEIGHT = 2.0
OFFLINE_MPC_TERMINAL_SWAY_WEIGHT = 5.0
OFFLINE_MPC_NOMINAL_DEVIATION_COST_WEIGHT = 1.5


def _offline_mpc_limit_action(action_accel_mps2):
    """Bound an offline world-frame acceleration-guidance candidate, not motor thrust."""
    action = np.asarray(action_accel_mps2, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(action)):
        raise ValueError("offline MPC action must be a finite 3-vector")
    xy_norm = float(np.linalg.norm(action[:2]))
    if xy_norm > OFFLINE_MPC_MAX_HORIZONTAL_ACCEL_MS2:
        action[:2] *= OFFLINE_MPC_MAX_HORIZONTAL_ACCEL_MS2 / max(xy_norm, 1e-12)
    action[2] = float(np.clip(action[2], -OFFLINE_MPC_MAX_VERTICAL_ACCEL_MS2,
                              OFFLINE_MPC_MAX_VERTICAL_ACCEL_MS2))
    return action


def _offline_mpc_initial_state(position, velocity, acceleration=None, sway_xy=(0.0, 0.0),
                               sway_rate_xy=(0.0, 0.0), previous_action=None):
    """Validate and package state for the isolated analytic surrogate."""
    p = np.asarray(position, dtype=np.float64).reshape(3)
    v = np.asarray(velocity, dtype=np.float64).reshape(3)
    a = np.zeros(3, dtype=np.float64) if acceleration is None else np.asarray(acceleration, dtype=np.float64).reshape(3)
    q = np.asarray(sway_xy, dtype=np.float64).reshape(2)
    qd = np.asarray(sway_rate_xy, dtype=np.float64).reshape(2)
    u_prev = np.zeros(3, dtype=np.float64) if previous_action is None else np.asarray(previous_action, dtype=np.float64).reshape(3)
    if not all(np.all(np.isfinite(x)) for x in (p, v, a, q, qd, u_prev)):
        raise ValueError("offline MPC state vectors must be finite")
    return {"position": p.copy(), "velocity": v.copy(), "acceleration": a.copy(),
            "sway_xy": q.copy(), "sway_rate_xy": qd.copy(), "previous_action": u_prev.copy()}


def _offline_mpc_surrogate_step(state, action_accel_mps2, wind_xyz_mps, dt_s,
                                cable_length_m, payload_mass_kg,
                                payload_to_drone_mass_ratio=OFFLINE_MPC_PAYLOAD_TO_DRONE_MASS_RATIO):
    """One step of a small action-conditioned surrogate for offline evaluator testing.

    This is a smooth point-mass model plus small-angle suspended-cargo dynamics.
    It is not authoritative flight physics; coefficients require calibration against
    paired BEMT/RK4 + PBD episodes before the model can be used for real MPC.
    """
    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0 or dt > 0.25:
        raise ValueError("offline MPC dt_s must be finite and in (0, 0.25]")
    wind = np.asarray(wind_xyz_mps, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(wind)):
        raise ValueError("offline MPC wind must be a finite 3-vector")
    cable_length = float(cable_length_m)
    payload_mass = float(payload_mass_kg)
    mass_ratio = float(payload_to_drone_mass_ratio)
    if not (math.isfinite(cable_length) and cable_length >= 0.0
            and math.isfinite(payload_mass) and payload_mass >= 0.0
            and math.isfinite(mass_ratio) and 0.0 <= mass_ratio <= 1.0):
        raise ValueError("offline MPC cable/payload parameters are invalid")

    out = {key: np.array(value, dtype=np.float64, copy=True) for key, value in state.items()}
    action = _offline_mpc_limit_action(action_accel_mps2)
    p, v, a_response = out["position"], out["velocity"], out["acceleration"]
    q, qd = out["sway_xy"], out["sway_rate_xy"]

    response_alpha = min(1.0, dt / OFFLINE_MPC_ACTUATOR_TAU_S)
    a_response += response_alpha * (action - a_response)
    relative_air_velocity = v - wind
    relative_air_speed = float(np.linalg.norm(relative_air_velocity))
    drag_accel = -OFFLINE_MPC_WIND_DRAG_COEFF * min(relative_air_speed, 25.0) * relative_air_velocity

    deployed_cargo = cable_length >= 0.30 and payload_mass > 0.0
    cable_accel_xy = np.zeros(2, dtype=np.float64)
    if deployed_cargo:
        cable_accel_xy = mass_ratio * 9.80665 * q / max(cable_length, 0.30)
        cable_accel_xy = np.clip(cable_accel_xy, -2.5, 2.5)
    net_accel = a_response + drag_accel
    if deployed_cargo:
        net_accel[:2] += cable_accel_xy
    # Numerical guard for the evaluator; it is a model bound, not a flight-system limit.
    net_accel = np.clip(np.nan_to_num(net_accel, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)

    p_next = p + v * dt + 0.5 * net_accel * dt * dt
    v_next = v + net_accel * dt
    tension_n = 0.0
    if deployed_cargo:
        natural_freq = math.sqrt(9.80665 / max(cable_length, 0.30))
        qdd = -(9.80665 / max(cable_length, 0.30)) * q
        qdd -= 2.0 * OFFLINE_MPC_CABLE_DAMPING_RATIO * natural_freq * qd
        qdd -= net_accel[:2]
        qd_next = qd + qdd * dt
        q_next = q + qd_next * dt
        vertical_specific_force = max(0.1, 9.80665 + net_accel[2])
        lateral_specific_force = 9.80665 * float(np.linalg.norm(q_next)) / max(cable_length, 0.30)
        tension_n = payload_mass * math.sqrt(vertical_specific_force**2 + lateral_specific_force**2)
    else:
        q_next = np.zeros(2, dtype=np.float64)
        qd_next = np.zeros(2, dtype=np.float64)

    out["position"] = p_next
    out["velocity"] = v_next
    out["acceleration"] = a_response
    out["sway_xy"] = q_next
    out["sway_rate_xy"] = qd_next
    out["previous_action"] = action.copy()
    out["net_acceleration"] = net_accel.copy()
    out["tension_n"] = float(tension_n)
    return out


def _offline_mpc_nominal_action(state, trajectory_start_position, trajectory_target_position,
                                trajectory_duration_s, trajectory_elapsed_s,
                                cable_length_m=0.0):
    """PD-plus-feedforward baseline with a small-cable-sway stabilisation term.

    This is still an evaluator-only surrogate controller, not the live flight
    controller. The sway term moves the drone toward the displaced payload and adds
    damping from relative sway velocity; both baseline and candidate-search rollouts
    use the same future nominal controller for a fairer comparison.
    """
    p_ref, v_ref, a_ref, _ = quintic_trajectory_reference(
        trajectory_start_position, trajectory_target_position,
        trajectory_duration_s, trajectory_elapsed_s)
    p = state["position"]
    v = state["velocity"]
    desired = a_ref + 2.0 * (p_ref - p) + 1.8 * (v_ref - v)
    cable_len = float(cable_length_m)
    if cable_len >= 0.30:
        q = np.asarray(state.get("sway_xy", np.zeros(2)), dtype=np.float64).reshape(2)
        qd = np.asarray(state.get("sway_rate_xy", np.zeros(2)), dtype=np.float64).reshape(2)
        natural_freq_sq = 9.80665 / max(cable_len, 0.30)
        desired[:2] += 0.55 * natural_freq_sq * q + 1.05 * qd
    return _offline_mpc_limit_action(desired)


def evaluate_offline_predictive_action(state, trajectory_start_position,
                                       trajectory_target_position, trajectory_duration_s,
                                       trajectory_elapsed_s, estimated_wind_xyz_mps=(0.0, 0.0, 0.0),
                                       cable_length_m=0.0,
                                       payload_mass_kg=OFFLINE_MPC_DEFAULT_PAYLOAD_MASS_KG,
                                       horizon_s=OFFLINE_MPC_HORIZON_S, dt_s=OFFLINE_MPC_DT_S,
                                       max_candidates=27):
    """Evaluate bounded constant acceleration-guidance candidates over a short horizon.

    Returns the selected candidate and diagnostic scores. It is an offline prototype:
    no map/obstacle constraints, motor/tilt model, learned action-conditioned transition
    model, or authoritative BEMT/PBD evaluation is present. It must not control live flight.
    """
    start = np.asarray(trajectory_start_position, dtype=np.float64).reshape(3)
    target = np.asarray(trajectory_target_position, dtype=np.float64).reshape(3)
    elapsed = float(trajectory_elapsed_s)
    duration = float(trajectory_duration_s)
    horizon = float(horizon_s)
    dt = float(dt_s)
    if not (np.all(np.isfinite(start)) and np.all(np.isfinite(target))):
        raise ValueError("trajectory endpoints must be finite 3-vectors")
    if not math.isfinite(elapsed) or elapsed < 0.0 or not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("trajectory times must be finite with duration > 0 and elapsed >= 0")
    if not math.isfinite(horizon) or horizon <= 0.0 or not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("horizon_s and dt_s must be finite and > 0")
    n_steps = int(math.ceil(horizon / dt))
    if n_steps < 1 or n_steps > OFFLINE_MPC_MAX_HORIZON_STEPS:
        raise ValueError(f"offline MPC horizon must be 1..{OFFLINE_MPC_MAX_HORIZON_STEPS} steps")
    if int(max_candidates) < 1:
        raise ValueError("max_candidates must be >= 1")
    wind = np.asarray(estimated_wind_xyz_mps, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(wind)):
        raise ValueError("estimated wind must be finite")

    nominal = _offline_mpc_nominal_action(state, start, target, duration, elapsed, cable_length_m)
    offsets = []
    for ix in (-1.0, 0.0, 1.0):
        for iy in (-1.0, 0.0, 1.0):
            for iz in (-1.0, 0.0, 1.0):
                offsets.append(np.array([1.35*ix, 1.35*iy, 0.70*iz], dtype=np.float64))
    candidates = []
    seen = set()
    for offset in offsets:
        candidate = _offline_mpc_limit_action(nominal + offset)
        key = tuple(np.round(candidate, 5).tolist())
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
        if len(candidates) >= int(max_candidates):
            break
    # Keep the current baseline action explicitly even if saturation caused deduplication.
    nominal_key = tuple(np.round(nominal, 5).tolist())
    if nominal_key not in seen and len(candidates) < int(max_candidates):
        candidates.append(nominal.copy())

    candidate_records = []
    for candidate_idx, candidate in enumerate(candidates):
        rollout_state = {key: np.array(value, dtype=np.float64, copy=True)
                         for key, value in state.items() if isinstance(value, np.ndarray)}
        total_cost = 0.0
        feasible = True
        max_sway = 0.0
        max_tension = 0.0
        max_speed = float(np.linalg.norm(rollout_state["velocity"]))
        max_accel = 0.0
        failure_reason = ""
        previous_action = np.asarray(state.get("previous_action", np.zeros(3)), dtype=np.float64)
        for step_idx in range(1, n_steps + 1):
            time_before_step = min(duration, elapsed + (step_idx - 1) * dt)
            # Receding-horizon approximation: score a candidate for the immediate
            # action, then simulate the same feedback nominal policy for future steps.
            # Holding one candidate for the entire horizon made the comparison unfair
            # against the baseline, which updates its command every timestep.
            if step_idx == 1:
                action_this_step = candidate
            else:
                action_this_step = _offline_mpc_nominal_action(
                    rollout_state, start, target, duration, time_before_step, cable_length_m)
            rollout_state = _offline_mpc_surrogate_step(
                rollout_state, action_this_step, wind, dt, cable_length_m, payload_mass_kg)
            time_future = min(duration, elapsed + step_idx * dt)
            p_ref, v_ref, _, _ = quintic_trajectory_reference(start, target, duration, time_future)
            pos_err = rollout_state["position"] - p_ref
            vel_err = rollout_state["velocity"] - v_ref
            sway = rollout_state["sway_xy"]
            sway_rate = rollout_state["sway_rate_xy"]
            speed = float(np.linalg.norm(rollout_state["velocity"]))
            accel_norm = float(np.linalg.norm(rollout_state["net_acceleration"]))
            tension = float(rollout_state["tension_n"])
            sway_norm = float(np.linalg.norm(sway))
            max_sway = max(max_sway, sway_norm)
            max_tension = max(max_tension, tension)
            max_speed = max(max_speed, speed)
            max_accel = max(max_accel, accel_norm)
            state_finite = all(np.all(np.isfinite(rollout_state[key])) for key in (
                "position", "velocity", "acceleration", "sway_xy", "sway_rate_xy", "net_acceleration"))
            if not state_finite:
                feasible, failure_reason = False, "non-finite surrogate state"
                break
            if float(rollout_state["position"][2]) < OFFLINE_MPC_MIN_ALTITUDE_M:
                feasible, failure_reason = False, "predicted ground/low-altitude violation"
                break
            if speed > OFFLINE_MPC_MAX_SPEED_MS:
                feasible, failure_reason = False, "predicted speed limit violation"
                break
            if sway_norm > OFFLINE_MPC_MAX_SWAY_M:
                feasible, failure_reason = False, "predicted cargo-sway limit violation"
                break
            if tension > OFFLINE_MPC_MAX_TENSION_N:
                feasible, failure_reason = False, "predicted cable-tension limit violation"
                break
            du = action_this_step - previous_action
            total_cost += dt * (
                OFFLINE_MPC_POSITION_COST_WEIGHT * float(np.dot(pos_err, pos_err))
                + OFFLINE_MPC_VELOCITY_COST_WEIGHT * float(np.dot(vel_err, vel_err))
                + OFFLINE_MPC_SWAY_COST_WEIGHT * float(np.dot(sway, sway))
                + OFFLINE_MPC_SWAY_RATE_COST_WEIGHT * float(np.dot(sway_rate, sway_rate))
                + OFFLINE_MPC_CONTROL_COST_WEIGHT * float(np.dot(action_this_step, action_this_step))
                + OFFLINE_MPC_SMOOTHNESS_COST_WEIGHT * float(np.dot(du, du)))
            previous_action = action_this_step
        if feasible:
            # Penalise needless deviation from the baseline feedback command. With a
            # one-action receding-horizon approximation, this prevents tiny surrogate
            # cost changes from selecting large guidance offsets that the forecast
            # cannot robustly justify.
            nominal_deviation = candidate - nominal
            total_cost += OFFLINE_MPC_NOMINAL_DEVIATION_COST_WEIGHT * float(np.dot(nominal_deviation, nominal_deviation))
            p_ref_end, v_ref_end, _, _ = quintic_trajectory_reference(
                start, target, duration, min(duration, elapsed + n_steps * dt))
            final_pos_err = rollout_state["position"] - p_ref_end
            final_vel_err = rollout_state["velocity"] - v_ref_end
            total_cost += (OFFLINE_MPC_TERMINAL_POSITION_WEIGHT * float(np.dot(final_pos_err, final_pos_err))
                           + OFFLINE_MPC_TERMINAL_VELOCITY_WEIGHT * float(np.dot(final_vel_err, final_vel_err))
                           + OFFLINE_MPC_TERMINAL_SWAY_WEIGHT * float(np.dot(rollout_state["sway_xy"], rollout_state["sway_xy"])))
        else:
            total_cost = float("inf")
        candidate_records.append({
            "index": candidate_idx, "action_acceleration_mps2": candidate.copy(),
            "feasible": feasible, "score": float(total_cost), "failure_reason": failure_reason,
            "endpoint_position_m": rollout_state["position"].copy(),
            "endpoint_velocity_mps": rollout_state["velocity"].copy(),
            "max_sway_m": float(max_sway), "max_tension_n": float(max_tension),
            "max_speed_mps": float(max_speed), "max_acceleration_mps2": float(max_accel)})

    feasible_records = [record for record in candidate_records if record["feasible"] and math.isfinite(record["score"])]
    if not feasible_records:
        return {"status": "NO_FEASIBLE_CANDIDATE", "action_acceleration_mps2": None,
                "nominal_action_acceleration_mps2": nominal, "evaluated_candidates": len(candidate_records),
                "feasible_candidates": 0, "best_score": float("inf"),
                "candidate_records": candidate_records,
                "model_scope": "analytic surrogate only; not connected to live control"}
    best = min(feasible_records, key=lambda r: r["score"])
    return {"status": "FEASIBLE", "action_acceleration_mps2": best["action_acceleration_mps2"].copy(),
            "nominal_action_acceleration_mps2": nominal, "evaluated_candidates": len(candidate_records),
            "feasible_candidates": len(feasible_records), "best_score": float(best["score"]),
            "best_candidate": best, "candidate_records": candidate_records,
            "model_scope": "analytic surrogate only; not connected to live control"}


def _offline_mpc_rollout_scenario(use_candidate_search, wind_xyz_mps, dt_s=0.05,
                                  trajectory_duration_s=5.0, horizon_s=1.0):
    """Run one synthetic scenario using the same approximate surrogate for comparison."""
    p0 = np.array([0.0, 0.0, 12.0], dtype=np.float64)
    p1 = np.array([8.0, 4.0, 12.0], dtype=np.float64)
    state = _offline_mpc_initial_state(p0, np.zeros(3), sway_xy=(0.18, -0.10),
                                       sway_rate_xy=(0.0, 0.0))
    cable_length = 6.0
    payload_mass = OFFLINE_MPC_DEFAULT_PAYLOAD_MASS_KG
    steps = int(math.ceil(trajectory_duration_s / dt_s))
    pos_errors, vel_errors, sway_history = [], [], []
    candidate_evaluations = 0
    no_safe_candidate = 0
    max_speed = 0.0
    max_accel = 0.0
    for step_idx in range(steps):
        elapsed = min(trajectory_duration_s, step_idx * dt_s)
        nominal = _offline_mpc_nominal_action(state, p0, p1, trajectory_duration_s, elapsed, cable_length)
        action = nominal
        if use_candidate_search:
            result = evaluate_offline_predictive_action(
                state, p0, p1, trajectory_duration_s, elapsed,
                estimated_wind_xyz_mps=wind_xyz_mps, cable_length_m=cable_length,
                payload_mass_kg=payload_mass, horizon_s=horizon_s, dt_s=dt_s)
            candidate_evaluations += result["evaluated_candidates"]
            if result["status"] == "FEASIBLE":
                action = result["action_acceleration_mps2"]
            else:
                # Offline fallback: retain the nominal feedback action instead of
                # injecting a zero command. This makes the comparison conservative and
                # avoids making the candidate-search baseline look worse solely because
                # the proxy found no candidate satisfying all horizon constraints.
                # No live actuation consumes this result or fallback.
                no_safe_candidate += 1
                action = nominal.copy()
        state = _offline_mpc_surrogate_step(
            state, action, wind_xyz_mps, dt_s, cable_length, payload_mass)
        ref_p, ref_v, _, _ = quintic_trajectory_reference(
            p0, p1, trajectory_duration_s, min(trajectory_duration_s, (step_idx + 1) * dt_s))
        pos_errors.append(float(np.linalg.norm(state["position"] - ref_p)))
        vel_errors.append(float(np.linalg.norm(state["velocity"] - ref_v)))
        sway_history.append(float(np.linalg.norm(state["sway_xy"])))
        max_speed = max(max_speed, float(np.linalg.norm(state["velocity"])))
        max_accel = max(max_accel, float(np.linalg.norm(state.get("net_acceleration", np.zeros(3)))))
    return {"controller": "candidate-search" if use_candidate_search else "nominal-feedback",
            "wind_xyz_mps": np.asarray(wind_xyz_mps, dtype=np.float64).tolist(),
            "position_rmse_m": float(np.sqrt(np.mean(np.square(pos_errors)))),
            "velocity_rmse_mps": float(np.sqrt(np.mean(np.square(vel_errors)))),
            "terminal_position_error_m": float(np.linalg.norm(state["position"] - p1)),
            "terminal_velocity_mps": float(np.linalg.norm(state["velocity"])),
            "max_cargo_sway_m": float(max(sway_history) if sway_history else 0.0),
            "max_speed_mps": float(max_speed), "max_acceleration_mps2": float(max_accel),
            "candidate_evaluations": int(candidate_evaluations),
            "no_feasible_action_steps": int(no_safe_candidate),
            "steps": int(steps), "dt_s": float(dt_s), "model_scope": "synthetic analytic surrogate only"}


def run_offline_predictive_control_benchmark():
    """Compare nominal PD with finite-candidate horizon scoring on a proxy model.

    This is an offline algorithm benchmark; it does not use the learned forecast head,
    does not call Genesis/BEMT, and does not demonstrate live flight or delivery safety.
    """
    print("=" * 78, flush=True)
    print("OFFLINE PREDICTIVE-CONTROL EVALUATION — V38.35 (analytic surrogate unchanged)", flush=True)
    print("Model: analytic action-conditioned surrogate; no live actuator connection.", flush=True)
    print("Note: V38.35 adds a gated teacher-forced accumulated transition evaluator. This V38.30 analytic surrogate remains separate; it still does not use the learned transition model. Readiness and sensitivity diagnostics are not counterfactual proof.", flush=True)
    print("Candidate action is evaluated for the first step; future rollout uses the same nominal feedback for both controllers.", flush=True)
    print("=" * 78, flush=True)
    scenarios = [("calm", np.array([0.0, 0.0, 0.0])),
                 ("crosswind", np.array([4.0, -1.0, 0.0]))]
    all_results = []
    for scenario_name, wind in scenarios:
        baseline = _offline_mpc_rollout_scenario(False, wind)
        candidate = _offline_mpc_rollout_scenario(True, wind)
        all_results.extend([baseline, candidate])
        print(f"[OFFLINE MPC] scenario={scenario_name} controller={baseline['controller']} "
              f"pos_RMSE={baseline['position_rmse_m']:.3f}m vel_RMSE={baseline['velocity_rmse_mps']:.3f}m/s "
              f"terminal_err={baseline['terminal_position_error_m']:.3f}m "
              f"max_sway={baseline['max_cargo_sway_m']:.3f}m", flush=True)
        print(f"[OFFLINE MPC] scenario={scenario_name} controller={candidate['controller']} "
              f"pos_RMSE={candidate['position_rmse_m']:.3f}m vel_RMSE={candidate['velocity_rmse_mps']:.3f}m/s "
              f"terminal_err={candidate['terminal_position_error_m']:.3f}m "
              f"max_sway={candidate['max_cargo_sway_m']:.3f}m "
              f"candidates={candidate['candidate_evaluations']} no-safe={candidate['no_feasible_action_steps']}", flush=True)
    print("[OFFLINE MPC COMPLETE] Results compare algorithms only on the same surrogate; "
          "no claim of BEMT/Genesis or live-flight performance is supported.", flush=True)
    return all_results


def _torch_force_balance_residual_target(measured_acceleration, commanded_acceleration,
                                          modeled_external_force, body_mass,
                                          transition_valid, contact_free):
    """Build masked [force-balance residual / scale_N, valid] Genesis targets.

    The batched Genesis model explicitly applies m*a_command, gravity compensation,
    aerodynamic drag, and tether reaction. With Genesis gravity cancelling the added
    +m*g compensation, the nominal free-flight acceleration is:
        a_nominal = a_command + (F_drag + F_tether) / m
    Therefore this target is the remaining correction force:
        F_residual = m*(a_measured - a_command) - (F_drag + F_tether).

    This is a simulator-specific residual, not a universal real-drone force model.
    Rows near the ground/depot, during touchdown, or outside the valid transition
    envelope are masked to avoid learning collision/impact impulses as smooth forces.
    All vector inputs are [...,3]; body_mass and masks are [...].
    """
    mass = torch.as_tensor(body_mass, dtype=measured_acceleration.dtype,
                           device=measured_acceleration.device)
    mass_safe = torch.clamp(torch.nan_to_num(mass, nan=0.0, posinf=0.0, neginf=0.0), min=1e-6)
    accel_meas = torch.nan_to_num(measured_acceleration, nan=0.0, posinf=0.0, neginf=0.0)
    accel_cmd = torch.nan_to_num(commanded_acceleration, nan=0.0, posinf=0.0, neginf=0.0)
    external_force = torch.nan_to_num(modeled_external_force, nan=0.0, posinf=0.0, neginf=0.0)
    force_residual_n = mass_safe.unsqueeze(-1) * (accel_meas - accel_cmd) - external_force
    accel_residual = force_residual_n / mass_safe.unsqueeze(-1)
    finite = (torch.isfinite(measured_acceleration).all(dim=-1)
              & torch.isfinite(commanded_acceleration).all(dim=-1)
              & torch.isfinite(modeled_external_force).all(dim=-1)
              & torch.isfinite(mass) & (mass > 0.0)
              & torch.isfinite(force_residual_n).all(dim=-1))
    transition_ok = torch.as_tensor(transition_valid, device=mass.device, dtype=torch.bool)
    contact_ok = torch.as_tensor(contact_free, device=mass.device, dtype=torch.bool)
    force_limit_n = FORCE_BALANCE_RESIDUAL_SCALE_N * 3.0
    valid = (transition_ok & contact_ok & finite
             & (torch.linalg.vector_norm(accel_residual, dim=-1) <= FORCE_BALANCE_MAX_RESIDUAL_ACCEL_MS2)
             & (torch.linalg.vector_norm(force_residual_n, dim=-1) <= force_limit_n))
    scaled = (force_residual_n / FORCE_BALANCE_RESIDUAL_SCALE_N).clamp(-3.0, 3.0)
    scaled = torch.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = torch.where(valid.unsqueeze(-1), scaled, torch.zeros_like(scaled))
    return torch.cat((scaled, valid.to(scaled.dtype).unsqueeze(-1)), dim=-1)


def _force_balance_residual_prediction_loss(prediction, target_with_validity):
    """Masked MSE for predicted force correction [Fx,Fy,Fz]/scale_N.

    Target column 3 is validity. This loss is used in Phase 3 only; no force head
    output is consumed by the live flight controller.
    """
    if prediction.ndim != 2 or prediction.shape[-1] != 3:
        raise ValueError(f"force residual prediction must have shape [N,3], got {tuple(prediction.shape)}")
    if target_with_validity.ndim != 2 or target_with_validity.shape[-1] != 4:
        raise ValueError(f"force residual target must have shape [N,4], got {tuple(target_with_validity.shape)}")
    if prediction.shape[0] != target_with_validity.shape[0]:
        raise ValueError("force residual prediction and target batch lengths must match")
    valid = ((target_with_validity[:, 3] > 0.5)
             & torch.isfinite(target_with_validity[:, :3]).all(dim=-1)
             & torch.isfinite(prediction).all(dim=-1))
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[valid], target_with_validity[valid, :3])


def _force_balance_residual_error_n(prediction, target_with_validity):
    """Return prediction error N, observed residual N, and a valid-row mask."""
    if prediction.ndim != 2 or prediction.shape[-1] != 3:
        raise ValueError(f"force residual prediction must have shape [N,3], got {tuple(prediction.shape)}")
    if target_with_validity.ndim != 2 or target_with_validity.shape[-1] != 4:
        raise ValueError(f"force residual target must have shape [N,4], got {tuple(target_with_validity.shape)}")
    if prediction.shape[0] != target_with_validity.shape[0]:
        raise ValueError("force residual prediction and target batch lengths must match")
    valid = ((target_with_validity[:, 3] > 0.5)
             & torch.isfinite(target_with_validity[:, :3]).all(dim=-1)
             & torch.isfinite(prediction).all(dim=-1))
    prediction_n = torch.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0) * FORCE_BALANCE_RESIDUAL_SCALE_N
    target_n = torch.nan_to_num(target_with_validity[:, :3], nan=0.0, posinf=0.0, neginf=0.0) * FORCE_BALANCE_RESIDUAL_SCALE_N
    return prediction_n - target_n, target_n, valid


def _force_balance_residual_rmse_n(prediction, target_with_validity):
    """Vector RMSE in N (sqrt mean over valid rows of squared 3-D force error)."""
    error_n, _, valid = _force_balance_residual_error_n(prediction, target_with_validity)
    if not bool(valid.any()):
        return float("nan")
    return float(torch.sqrt(torch.sum(error_n[valid] ** 2, dim=-1).mean()).detach().item())


def _cargo_dynamics_prediction_loss(prediction, target_with_validity):
    """Masked MSE for [sway_xy/5m, relative_sway_velocity_xy/8m/s, tension/45N].

    Target column 5 is the validity mask. Samples without a payload on an extended
    cable are excluded rather than teaching the head that absent cargo always means
    zero sway. Extreme/non-finite transitions also remain excluded from this loss.
    """
    if prediction.ndim != 2 or prediction.shape[-1] != 5:
        raise ValueError(f"cargo dynamics prediction must have shape [N,5], got {tuple(prediction.shape)}")
    if target_with_validity.ndim != 2 or target_with_validity.shape[-1] != 6:
        raise ValueError(f"cargo dynamics target must have shape [N,6], got {tuple(target_with_validity.shape)}")
    valid = (target_with_validity[:, 5] > 0.5) & torch.isfinite(target_with_validity[:, :5]).all(dim=-1)
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[valid], target_with_validity[valid, :5])


def _numpy_one_step_cargo_dynamics_target(sway_after, sway_velocity_after, tension_after,
                                          payload_before, cable_len_before,
                                          payload_after, cable_len_after,
                                          transition_valid=True):
    """Build a one-step cargo target from measured post-step state and a strict mask.

    The head learns only when an attached package is on a deployed cable both before
    and after the transition; stowed/empty-hook rows cannot dominate the loss.
    """
    sway = np.asarray(sway_after, dtype=np.float64).reshape(2)
    sway_rate = np.asarray(sway_velocity_after, dtype=np.float64).reshape(2)
    try:
        tension = float(tension_after)
        cable_before = float(cable_len_before)
        cable_after = float(cable_len_after)
    except (TypeError, ValueError, OverflowError):
        tension, cable_before, cable_after = float("nan"), float("nan"), float("nan")
    finite = (np.all(np.isfinite(sway)) and np.all(np.isfinite(sway_rate))
              and math.isfinite(tension) and math.isfinite(cable_before)
              and math.isfinite(cable_after))
    active = bool(payload_before) and bool(payload_after) and cable_before > 0.12 and cable_after > 0.12
    valid = (bool(transition_valid) and active and finite
             and float(np.linalg.norm(sway)) <= 10.0
             and float(np.linalg.norm(sway_rate)) <= 3.0 * CARGO_DYNAMICS_SWAY_RATE_SCALE_MS
             and 0.0 <= tension <= 45.0
             and cable_before <= 12.5 and cable_after <= 12.5)
    values = np.concatenate((
        np.clip(sway / CARGO_DYNAMICS_SWAY_SCALE_M, -2.0, 2.0),
        np.clip(sway_rate / CARGO_DYNAMICS_SWAY_RATE_SCALE_MS, -3.0, 3.0),
        np.array([np.clip(tension / CARGO_DYNAMICS_TENSION_SCALE_N, 0.0, 1.0)]),
    )).astype(np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((values, np.array([1.0 if valid else 0.0], dtype=np.float32)))


def _numpy_one_step_environment_transition_targets(observations, kinematics_targets,
                                                  drone_count=NUM_SWARM_DRONES):
    """Pair each row with the next frame's same-drone observation.

    The worker writes rows frame-major, then drone-major, so same-drone t+1 is
    row i+drone_count. Targets are the six normalized next-state channels listed
    in ENVIRONMENT_TRANSITION_FEATURE_INDICES plus a strict transition-valid mask.
    The existing BEMT transition validity mask rejects divergent/reset transitions;
    additional channel-delta limits reject implausible sensor jumps.
    """
    obs = np.asarray(observations, dtype=np.float32)
    kin = np.asarray(kinematics_targets, dtype=np.float32)
    drone_count = int(drone_count)
    if obs.ndim != 2 or obs.shape[1] != OBSERVATION_DIM:
        raise ValueError(f"observations must be [N,{OBSERVATION_DIM}]")
    if kin.shape != (len(obs), 7):
        raise ValueError("kinematics_targets must have shape [N,7]")
    if drone_count < 1 or len(obs) % drone_count != 0:
        raise ValueError("rows must be frame-major with a fixed positive drone_count")
    out = np.zeros((len(obs), ENVIRONMENT_TRANSITION_TARGET_DIM), dtype=np.float32)
    latest = (HISTORY_LEN - 1) * BASE_OBS_DIM
    idx = np.asarray(ENVIRONMENT_TRANSITION_FEATURE_INDICES, dtype=np.int64)
    for i in range(max(0, len(obs) - drone_count)):
        j = i + drone_count
        if kin[i, 6] <= 0.5 or not np.isfinite(kin[i, :6]).all():
            continue
        if not np.isfinite(obs[i]).all() or not np.isfinite(obs[j]).all():
            continue
        current = obs[i, latest + idx]
        future = obs[j, latest + idx]
        delta = future - current
        if not np.isfinite(current).all() or not np.isfinite(future).all():
            continue
        if float(np.linalg.norm(delta[:3])) > ENVIRONMENT_TRANSITION_MAX_WIND_DELTA_NORM:
            continue
        if abs(float(delta[3])) > ENVIRONMENT_TRANSITION_MAX_CABLE_LENGTH_DELTA:
            continue
        if abs(float(delta[4])) > ENVIRONMENT_TRANSITION_MAX_CABLE_RATE_DELTA:
            continue
        if abs(float(delta[5])) > 0.60:
            continue
        # Targets are absolute normalized future channel values, not deltas. This
        # lets recursive evaluation replace these channels directly with predictions.
        out[i, :ENVIRONMENT_TRANSITION_DIM] = np.clip(future, [-2,-2,-2,0,-2,0], [2,2,2,1.0,2,1.0])
        out[i, -1] = 1.0
    return out


def _environment_transition_prediction_loss(prediction, target_with_validity):
    """Masked MSE over next normalized wind/cable/icing observation channels."""
    if prediction.ndim != 2 or prediction.shape[-1] != ENVIRONMENT_TRANSITION_DIM:
        raise ValueError(f"environment transition prediction must be [N,{ENVIRONMENT_TRANSITION_DIM}]")
    if target_with_validity.ndim != 2 or target_with_validity.shape[-1] != ENVIRONMENT_TRANSITION_TARGET_DIM:
        raise ValueError(f"environment transition target must be [N,{ENVIRONMENT_TRANSITION_TARGET_DIM}]")
    valid = ((target_with_validity[:, -1] > 0.5)
             & torch.isfinite(target_with_validity[:, :-1]).all(dim=-1)
             & torch.isfinite(prediction).all(dim=-1))
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[valid], target_with_validity[valid, :-1])


def _environment_transition_validation_stats(prediction, target_with_validity):
    """Per-channel RMSE in meaningful units for valid held-out examples."""
    if prediction.ndim != 2 or prediction.shape[-1] != ENVIRONMENT_TRANSITION_DIM:
        raise ValueError("invalid environment-transition predictions")
    valid = ((target_with_validity[:, -1] > 0.5)
             & torch.isfinite(target_with_validity[:, :-1]).all(dim=-1)
             & torch.isfinite(prediction).all(dim=-1))
    n = int(valid.sum().item())
    if n == 0:
        return {"count": 0, "wind_vector_rmse_mps": float("nan"),
                "cable_length_rmse_m": float("nan"), "cable_rate_rmse_mps": float("nan"),
                "icing_rmse_index": float("nan"), "normalized_mse": float("nan")}
    err = prediction[valid] - target_with_validity[valid, :-1]
    wind_rmse = float((torch.sqrt(torch.mean(torch.sum(err[:, :3] ** 2, dim=-1))) * 8.0).item())
    cable_length_rmse = float((torch.sqrt(torch.mean(err[:, 3] ** 2)) * 15.0).item())
    cable_rate_rmse = float((torch.sqrt(torch.mean(err[:, 4] ** 2)) * 5.0).item())
    icing_rmse = float(torch.sqrt(torch.mean(err[:, 5] ** 2)).item())
    return {"count": n, "wind_vector_rmse_mps": wind_rmse,
            "cable_length_rmse_m": cable_length_rmse, "cable_rate_rmse_mps": cable_rate_rmse,
            "icing_rmse_index": icing_rmse, "normalized_mse": float(torch.mean(err ** 2).item())}

"""
=============================================================================
SURVIVAL CORE V37.2 — WIND-SENSOR-INTEGRATED, WIND-ROBUST ADAPTIVE DRONE SWARM SYSTEM
PART 5 OF 5: DEADLOCK-FREE HARVESTER & TRAINER, GENESIS 3D MIRROR,
             CAMERA SYSTEM 2, CBBA DISPATCHER & 60 FPS INTERACTIVE COCKPIT
=============================================================================
"""

# =============================================================================
# 16. SAFE PARALLEL HARVESTER & MULTI-TASK RESNET TRAINER
# =============================================================================
def _select_harvest_indices_with_action_coverage(labels, target_count, required_classes=range(6), seed=None):
    """Select real samples with class coverage and cap the service-state majority.

    Samples are selected without replacement; action labels are never fabricated or
    duplicated. The pre-V22.5 selector reserved a small quota for rare classes, then
    filled the rest completely at random. In the user's run that left DEPOT_SERVICE
    at 71.6% of the selected data. This selector caps that single class at 40% and
    fills the rest from real samples in the other classes. Inverse-frequency CE
    weights still address the remaining unavoidable imbalance (especially contact).
    Classes 0..5 are required; class 6 remains deliberately untrained and must be
    handled by the independent runtime safety supervisor rather than neural prediction.
    """
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
            "Refusing to train a single/partial-class decision head. Increase the harvest horizon, "
            "or inspect mission/winch progression before retraining."
        )

    majority_class = 5  # DEPOT_SERVICE is the known over-represented class.
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
    """Cycle deterministically through real teacher states across worker/drone slots."""
    return (int(worker_id) * NUM_SWARM_DRONES + int(drone_idx)) % 6


def _seed_harvest_coverage_state(drone, coverage_class):
    """Reset one simulated drone to a physically consistent, reachable mission state.

    This removes dependence on every worker naturally completing a full mission before
    rare action labels can be observed. The actual teacher and winch state machine still
    generate the labels after reset; no action labels are directly manufactured.
    """
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
        anchor[2] -= PHASE3_DRONE_HALF_HEIGHT_M  # same attachment point used by AutonomousWinchDeliverySystem.step
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

    if coverage_class == 0:  # TRANSIT: outbound, at cruise altitude with payload stowed
        pos = home.copy()
        pos[2] = cruise_z
        set_pose(pos)
        drone.mission_phase = "2_CRUISE_OUTBOUND"

    elif coverage_class in (1, 2, 3, 4):
        pos = np.array([target[0], target[1], high_hover_z], dtype=np.float32)
        set_pose(pos)
        drone.mission_phase = "4_DELIVERY_HOVER"
        if coverage_class == 1:  # HOVER_LOCK: payload attached and cable stowed
            drone.winch.state = "STANDBY"
            drone.winch.has_payload = True
            drone.winch.cable_len = PHASE3_STOWED_CABLE_LEN_M

        elif coverage_class == 2:  # WINCH_LOWER: cable already deployed, package still attached
            drone.winch.state = "LOWERING"
            drone.winch.has_payload = True
            drone.hover_settle_timer = DELIVERY_HOLD_MIN_S
            set_cable_geometry(2.5)

        elif coverage_class == 3:  # TOUCHDOWN_RELEASE: valid payload contact at the pad centre
            drone.winch.state = "TOUCHDOWN"
            drone.winch.has_payload = True
            drone.hover_settle_timer = DELIVERY_HOLD_MIN_S
            drone.winch.touchdown_timer = 0.0
            drone.winch.contact_normal_force = PAYLOAD_WEIGHT_N
            # Cable payout is measured from the drone underside to the parcel top;
            # the PBD endpoint is the parcel centre at pad_top + half-height.
            length = max(PHASE3_STOWED_CABLE_LEN_M,
                         high_hover_z - 0.05 - (drop_surface_z + 2.0 * PHASE3_CARGO_HALF_HEIGHT_M))
            set_cable_geometry(length, endpoint_z=drop_surface_z + PHASE3_CARGO_HALF_HEIGHT_M)
            drone.winch.contact_normal_force = PAYLOAD_WEIGHT_N

        else:  # WINCH_RETRACT: package has been released, empty cable is being recovered
            drone.winch.state = "RETRACTING"
            drone.winch.has_payload = False
            drone.winch.delivery_completed = False
            set_cable_geometry(3.0)

    else:  # DEPOT_SERVICE: on pad, empty winch, ready for next dispatch
        pos = home.copy()
        pos[2] = 0.06
        set_pose(pos)
        drone.mission_phase = "8_DEPOT_SERVICE"
        drone.winch.has_payload = False
        drone.winch.state = "STANDBY"
        drone.winch.cable_len = PHASE3_STOWED_CABLE_LEN_M

    # Ensure a seeded mission phase cannot inherit completion flags from a prior state.
    if coverage_class != 4:
        drone.winch.delivery_completed = False
    drone.winch.last_release_pos = None
    drone.winch.is_skid_grounded = (coverage_class == 5)
    return coverage_class


def _harvest_dynamics_fault_reason(drone):
    """Return a reason when a harvest-only airframe state has left the valid envelope.

    The city missions use roof/cruise altitudes under roughly 60 m. States hundreds of
    metres high, non-finite vectors, or extreme speeds are numerical/controller divergence,
    not useful demonstrations. The guard is intentionally limited to dataset harvesting.
    """
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
    # The city, roof and cruise missions occupy roughly 0–60 m. Catch vertical
    # runaway before it climbs to the old 180 m ceiling and contaminates many frames.
    vertical_speed = abs(float(v[2]))
    if vertical_speed > 8.0:
        return f"vertical speed outside harvest envelope ({float(v[2]):.2f} m/s)"
    if float(p[2]) < -5.0 or float(p[2]) > 80.0:
        return f"altitude outside harvest envelope ({float(p[2]):.2f} m; limit 80.0 m)"
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
    """Reset a divergent harvesting episode to its known reachable seed state."""
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
    # Reset persistent mixer estimates as well as its integrators. Otherwise a
    # divergent episode can carry stale rotor/CoM/attitude estimates into the
    # freshly seeded mission and trigger the same runaway again.
    drone.mixer._last_motor_omegas[:] = HOVER_OMEGA_NOMINAL
    if hasattr(drone.mixer, "_prev_f_thrust"):
        drone.mixer._prev_f_thrust = 0.0
    if hasattr(drone.mixer, "_prev_tau_xy"):
        drone.mixer._prev_tau_xy.fill(0.0)
    if hasattr(drone.mixer, "adaptive_airframe"):
        engine = drone.mixer.adaptive_airframe
        engine.J_estimated[:] = engine.J_nominal
    if hasattr(drone.mixer, "inflow_estimator"):
        estimator = drone.mixer.inflow_estimator
        estimator.ct_estimated = estimator.nominal_ct
        estimator.aerodynamic_health = 1.0
    if hasattr(drone.mixer, "mass_com_estimator"):
        estimator = drone.mixer.mass_com_estimator
        estimator.est_mass = float(DRY_MASS_KG + (CARGO_MASS_KG if drone.winch.has_payload else 0.0))
        estimator.com_offset_xy.fill(0.0)
    if hasattr(drone.mixer, "roof_normal_estimator"):
        drone.mixer.roof_normal_estimator.surface_normal[:] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if hasattr(drone.mixer, "tactile_reflex"):
        drone.mixer.tactile_reflex.reset()

    # Episode timers and flags must agree with the newly seeded FSM/winch state.
    drone.chimney_alignment_timer = 0.0
    drone.chimney_descent_armed = False
    drone.chimney_descent_settle_timer = 0.0
    drone.chimney_phase_elapsed = 0.0
    drone.delivery_abort_retries = 0
    drone.delivery_retry_cooldown = 0.0
    drone.last_winch_state = drone.winch.state

    if hasattr(drone.ekf, "p"):
        drone.ekf.p[:] = drone.airframe.p
    if hasattr(drone.ekf, "v"):
        drone.ekf.v.fill(0.0)
    if hasattr(drone.ekf, "q"):
        drone.ekf.q[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if hasattr(drone.ekf, "P") and hasattr(drone.ekf, "Q_base"):
        drone.ekf.P[:] = np.diag([0.15]*3 + [0.1]*3 + [0.02]*3 + [0.05]*3 + [0.01]*3).astype(np.float32)
        drone.ekf.Q_d[:] = drone.ekf.Q_base
        drone.ekf.residual_scale = 1.0
    if hasattr(drone.ekf, "b_a"):
        drone.ekf.b_a.fill(0.0)
    if hasattr(drone.ekf, "b_g"):
        drone.ekf.b_g.fill(0.0)


def _worker_harvest_metropolitan(worker_args):
    """
    Worker process saves arrays directly to temporary binary storage to avoid
    multiprocessing pipe buffer deadlocks when generating large datasets.
    """
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
    # Harvesting has its own intruder state; define agent_pos before the first
    # sensor observation so no worker path relies on a locals()-based existence test.
    ground_agent = DynamicGroundIntruderAgent(init_pos=[-80.0, 80.0, 0.0])
    agent_pos = ground_agent.p.copy()

    obs_list = []
    y_guidance, y_fsm, y_winch, y_health, y_trust, y_val, y_kinematics, y_cargo_dynamics, y_rotational_dynamics = [], [], [], [], [], [], [], [], []
    y_trajectory_forecast = []
    y_action_transition = []

    drones = []
    for i in range(NUM_SWARM_DRONES):
        init_p = world.depot_helipads[i].copy()
        init_p[2] = 0.06
        d_inst = SwarmDroneInstance(i, init_p, world, dispatcher)
        drones.append(d_inst)

    # Deterministic coverage reset: 14 workers × 3 drones span all six trained
    # actions. The simulator still records observations and teacher labels normally.
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
    # `num_steps` is a sample count, while every simulation frame emits one sample
    # per drone. The old loop therefore gave each worker only ~num_steps/3 frames,
    # often ending before any delivery state was ever visited. Run a longer rollout,
    # then the parent selects a balanced, shuffled 250k subset.
    sim_frame_budget = max(1, int(math.ceil(num_steps / NUM_SWARM_DRONES)) * horizon_multiplier)
    samples_per_regime = max(1, int(math.ceil(sim_frame_budget * NUM_SWARM_DRONES / 4)))
    sim_frames = 0
    pending_trajectory_forecasts = [deque() for _ in range(NUM_SWARM_DRONES)]
    harvest_dynamics_resets = 0
    harvest_reported_fault_kinds = set()
    harvest_fault_reason_counts = {}

    obstacles_cbf = world.get_static_collision_obstacles()

    while sim_frames < sim_frame_budget:
        # Quarantine physically impossible states before they affect sensors, the intruder,
        # peer-avoidance observations, or teacher labels. This is a harvest-only guard;
        # it does not teleport aircraft in the live simulator.
        for d_idx, d in enumerate(drones):
            fault_reason = _harvest_dynamics_fault_reason(d)
            if fault_reason is not None:
                bad_p = np.asarray(d.airframe.p, dtype=np.float64).copy()
                bad_v = np.asarray(d.airframe.v, dtype=np.float64).copy()
                _reset_harvest_drone_after_divergence(d, seeded_classes[d_idx])
                pending_trajectory_forecasts[d_idx].clear()
                harvest_dynamics_resets += 1
                fault_kind = fault_reason.split(" (", 1)[0]
                harvest_fault_reason_counts[fault_kind] = harvest_fault_reason_counts.get(fault_kind, 0) + 1
                # One representative sample per failure kind per worker is enough;
                # the final summary preserves full counts without filling the terminal.
                if fault_kind not in harvest_reported_fault_kinds:
                    harvest_reported_fault_kinds.add(fault_kind)
                    print(
                        f"[HARVEST DYNAMICS GUARD] worker={worker_id} sample_drone={d_idx + 1} "
                        f"reason={fault_reason}; state reset before observation/label generation.",
                        flush=True,
                    )
        sim_time += dt
        regime_idx = min(3, (sim_frames * NUM_SWARM_DRONES) // samples_per_regime)
        mode = weather_modes[regime_idx]

        if mode == "CLEAR_NOMINAL":
            episode_wind = np.array([1.5, -0.5, 0.0], dtype=np.float32)
            ice_active, curr_ice, amb_temp = False, 0.0, 22.0
        elif mode == "CANYON_GALE":
            episode_wind = np.array([6.5, 3.2, 0.0], dtype=np.float32)
            ice_active, curr_ice, amb_temp = False, 0.0, 14.0
        elif mode == "FREEZING_RAIN":
            episode_wind = np.array([2.5, -2.0, 0.0], dtype=np.float32)
            ice_active, curr_ice, amb_temp = True, 0.45, -4.5
        else:
            downburst_z = -3.2 * float(math.tanh(max(0.1, drones[0].airframe.p[2]) / 4.5))
            episode_wind = np.array([-5.0, 4.2, downburst_z], dtype=np.float32)
            ice_active, curr_ice, amb_temp = False, 0.0, 10.0

        world.step_civil_traffic(dt=dt)
        traffic_engine.step(dt=dt)
        swarm_states = [{"p": d.airframe.p.copy(), "v": d.airframe.v.copy()} for d in drones]

        # Advance the same ground-intruder model used by live flight once per
        # simulation tick. Prefer the drone whose winch is active; otherwise use
        # drone 0 as the stable reference for the intruder's proximity response.
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
        step_yh_triad, step_yt_triad, step_yv_triad, step_ykin_triad, step_ycargo_triad, step_yrot_triad = [], [], [], [], [], []
        step_ytrajectory_triad = []
        step_yaction_transition_triad = []

        for d_idx, d in enumerate(drones):
            forecast_queue = pending_trajectory_forecasts[d_idx]
            for pending in forecast_queue:
                age_frames = sim_frames - pending["frame"]
                for hidx, required_frames in enumerate(TRAJECTORY_FORECAST_HORIZON_FRAMES):
                    target_row = pending["target"]
                    mask_idx = TRAJECTORY_FORECAST_DIM + hidx
                    if target_row[mask_idx] > 0.5 or age_frames != required_frames:
                        continue
                    delta_future = np.asarray(d.airframe.p, dtype=np.float64) - pending["position"]
                    future_velocity = np.asarray(d.airframe.v, dtype=np.float64)
                    horizon_s = TRAJECTORY_FORECAST_HORIZONS_S[hidx]
                    displacement_limit = max(3.0, 12.0*horizon_s)
                    valid_forecast = bool(
                        np.all(np.isfinite(delta_future)) and np.all(np.isfinite(future_velocity))
                        and float(np.linalg.norm(delta_future)) <= displacement_limit
                        and float(np.linalg.norm(future_velocity)) <= TRAJECTORY_FORECAST_MAX_SPEED_MS
                        and abs(float(future_velocity[2])) <= 8.0
                        and -1.0 <= float(d.airframe.p[2]) <= 80.0)
                    if valid_forecast:
                        start = hidx*TRAJECTORY_FORECAST_VALUES_PER_HORIZON
                        target_row[start:start+3] = np.clip(delta_future/TRAJECTORY_FORECAST_POSITION_SCALE_M, -3.0, 3.0)
                        target_row[start+3:start+6] = np.clip(future_velocity/TRAJECTORY_FORECAST_VELOCITY_SCALE_MS, -2.0, 2.0)
                        target_row[mask_idx] = 1.0
            while forecast_queue and (sim_frames-forecast_queue[0]["frame"]) >= TRAJECTORY_FORECAST_HORIZON_FRAMES[-1]:
                forecast_queue.popleft()
            # Every sample starts with explicit safe labels. Python loop variables
            # persist across drones/frames, so leaving a state unassigned can leak an
            # unrelated prior FSM label into the supervised dataset.
            fsm_token, winch_rate_cmd = 0, 0.0
            is_station_holding = False
            is_solidly_over_roof = False
            # The teacher/controller sees only the simulated wind-sensor estimate.
            # True episode_wind remains reserved for physical dynamics and sensor synthesis.
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
                # Hand over before the target to the precision-capture state even if
                # altitude or wind is not yet settled. That state holds cruise altitude
                # and refuses to arm descent until all safety gates pass. Requiring the
                # altitude/wind gates here let the drone fly past the target in transit.
                if _should_enter_precision_approach(dist_tgt_xy, speed_xy_now, wind_speed_now):
                    d.mission_phase = "3_CHIMNEY_DESCENT"
                    # Cruise integrator memory is not valid for precision station keeping.
                    d.mixer.xy_err_integral.fill(0.0)
                    d.mixer.z_err_integral = 0.0
                    d.chimney_alignment_timer = 0.0
                    d.chimney_descent_armed = False
                    d.chimney_descent_settle_timer = 0.0
                    d.chimney_phase_elapsed = 0.0

            elif d.mission_phase == "3_CHIMNEY_DESCENT":
                # This state is aircraft precision capture/descent, not winch payout.
                # Teach transit while aligning at cruise; use HOVER_LOCK once the
                # precision descent is armed. Never inherit another state's token.
                fsm_token = 1 if d.chimney_descent_armed else 0
                winch_rate_cmd = 0.0
                speed_xy_now = float(np.linalg.norm(d.airframe.v[:2]))
                speed_z_now = abs(float(d.airframe.v[2]))
                # In the harvesting worker the per-frame sensor output is named
                # sensed_wind. estimated_wind is the alias used by the live loop.
                wind_speed_now = float(np.linalg.norm(sensed_wind[:2]))
                # Pause the deadline while wind makes precision delivery unavailable;
                # otherwise the safety interlock itself can force a timeout abort.
                if wind_speed_now <= DELIVERY_MAX_OPERATIONAL_WIND_MS:
                    d.chimney_phase_elapsed += dt
                # If wind exceeds the delivery envelope while the cable is still stowed,
                # suspend vertical descent and regain the cruise-height hold. This prevents
                # the aircraft from committing to a rooftop approach it cannot stabilise.
                if wind_speed_now > DELIVERY_MAX_OPERATIONAL_WIND_MS and d.winch.cable_len <= 0.12:
                    d.chimney_descent_armed = False
                    d.chimney_alignment_timer = 0.0
                    d.chimney_descent_settle_timer = 0.0
                if not d.chimney_descent_armed:
                    # First stop over the target at cruise altitude. Do not begin
                    # descending while horizontal approach speed is still high.
                    wp_target = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                    approach_speed_cap = min(1.8, 0.45 * dist_tgt_xy,
                                             math.sqrt(max(0.0, 2.0 * 1.1 * dist_tgt_xy)))
                    p_ref_guidance, v_ref = compute_vector_guidance(
                        d.airframe.p, d.airframe.v, wp_target, max_speed_xy=approach_speed_cap,
                        max_climb=0.35, max_descend=0.15, wind_vector=sensed_wind)
                    # Match live and teacher control: fixed target position with a
                    # speed-limited reference velocity, not a waypoint that moves
                    # with the current aircraft position.
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
                    # Vertical descent follows only after the cruise-height hover is
                    # established. XY target is held fixed and descent is capped.
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
                        d.delivery_abort_retries = 0
                        d.delivery_retry_cooldown = 0.0
                        d.last_winch_state = d.winch.state
                        d.delivery_attempts += 1

                # The approach phase includes both cruise-height alignment and a
                # potentially large vertical move to the roof hover point. A fixed
                # 24 s timeout can be physically impossible for higher cruise-to-roof
                # differences. Scale the deadline to the commanded vertical travel,
                # with margin for acceleration, braking and the required settle dwell.
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
                    xy_error_abort = float(np.linalg.norm(d.airframe.p[:2] - tgt_xy))
                    xy_speed_abort = float(np.linalg.norm(d.airframe.v[:2]))
                    z_error_abort = abs(float(d.airframe.p[2]) - float(high_hover_z))
                    z_speed_abort = abs(float(d.airframe.v[2]))
                    print(
                        f"[APPROACH ABORT] Drone {d.drone_id + 1}: stability gate not achieved "
                        f"within {approach_timeout_s:.1f}s (vertical travel budget "
                        f"{approach_vertical_delta_m:.1f}m); retaining cargo and returning to base. "
                        f"xy_error={xy_error_abort:.2f}m, xy_speed={xy_speed_abort:.2f}m/s, "
                        f"z_error={z_error_abort:.2f}m, z_speed={z_speed_abort:.2f}m/s, "
                        f"wind={wind_speed_now:.2f}m/s, descent_armed={d.chimney_descent_armed}, "
                        f"align_dwell={d.chimney_alignment_timer:.2f}s, "
                        f"settle_dwell={d.chimney_descent_settle_timer:.2f}s."
                    )

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
                    and float(np.linalg.norm(sensed_wind[:2])) <= DELIVERY_MAX_DEPLOYMENT_WIND_MS
                )
                if is_station_holding:
                    d.hover_settle_timer += dt
                else:
                    d.hover_settle_timer = 0.0
                cable_stowed = (d.winch.cable_len <= 0.10)
                # Teach distinct actions for state transitions, not just cable length:
                # once STANDBY has armed the winch, LOWERING must continue extending
                # even before the cable has grown past the stowed-length threshold.
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

                # Do not manufacture a successful delivery when a timer expires.
                # If deployment is stuck, retract with the payload still attached and
                # report a failed attempt only once the cable is safely stowed.
                if d.delivery_phase_elapsed >= DELIVERY_HOVER_TIMEOUT_S and not d.winch.delivery_completed:
                    if d.winch.cable_len > 0.12:
                        d.winch.state = "ABORT_RETRACT"
                    else:
                        d.delivery_failures += 1
                        d.delivery_phase_elapsed = 0.0
                        d.hover_settle_timer = 0.0
                        d.dispatcher.report_failed(d.drone_id)
                        d.mission_phase = "5_CHIMNEY_ASCENT"

                        d.hover_settle_timer = 0.0
                        d.target = d.mission["home_pad"].copy()
                elif d.winch.delivery_completed and cable_stowed:
                    d.winch.delivery_completed = False
                    d.hover_settle_timer = 0.0
                    d.delivery_phase_elapsed = 0.0
                    d.packages_delivered += 1
                    d.dispatcher.report_completed(d.drone_id)
                    d.mission_phase = "5_CHIMNEY_ASCENT"

                    d.hover_settle_timer = 0.0
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
                    # Do not carry cruise tracking integral into the slow landing controller.
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
            d.winch.update_skid_strain_gauges(is_on_surf, weight_force, d.airframe.last_p_watts * 0.05)

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
            trajectory_target = np.zeros(TRAJECTORY_FORECAST_TARGET_DIM, dtype=np.float32)
            step_ytrajectory_triad.append(trajectory_target)
            pending_trajectory_forecasts[d_idx].append({
                "frame": int(sim_frames), "position": np.asarray(d.airframe.p, dtype=np.float64).copy(),
                "target": trajectory_target})
            # Capture exactly the state represented by obs_417 before its paired
            # high-level guidance/FSM/winch action is applied to the simulator.
            action_transition_state_before = (
                np.asarray(d.airframe.p, dtype=np.float64).copy(),
                np.asarray(d.airframe.v, dtype=np.float64).copy(),
                np.asarray(d.airframe.omega, dtype=np.float64).copy(),
                float(current_surf_z), bool(d.airframe.parachute_deployed))
            action_transition_target_row = np.zeros(ACTION_TRANSITION_TARGET_DIM, dtype=np.float32)
            step_yaction_transition_triad.append(action_transition_target_row)

            delta_p_clamped = np.clip(wp_target - d.airframe.p, -5.0, 5.0)
            a_target_xy = np.array([1.6, 1.6], dtype=np.float32) * delta_p_clamped[:2] + np.array([1.8, 1.8], dtype=np.float32) * (v_ref[:2] - d.airframe.v[:2])
            if d.winch.has_payload and d.winch.cable_len > 0.12:
                # Include the same displacement/rate damping used by the live mixer
                # in the teacher target, so data collection reinforces anti-sway control.
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

            if d.winch.state in ("LOWERING", "TOUCHDOWN"):
                # Once armed, stay authorized through ordinary station/sway changes
                # up to the hard operational wind boundary. The winch itself pauses
                # payout at moderate gusts; returning False here would wrongly force
                # an immediate abort/retract on every such gust.
                winch_station_gate = _delivery_winch_continuation_authorized(
                    d.mission_phase, d.winch.state, d.winch.has_payload,
                    float(np.linalg.norm(sensed_wind[:2])))
            else:
                winch_station_gate = bool(
                    d.mission_phase == "4_DELIVERY_HOVER"
                    and is_station_holding
                    and (d.winch.state != "STANDBY"
                         or d.hover_settle_timer >= DELIVERY_HOLD_MIN_S))
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
            winch_state_before_step = d.winch.state
            cargo_payload_before_step = bool(d.winch.has_payload)
            cargo_cable_len_before_step = float(d.winch.cable_len)
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
            kin_p_before = d.airframe.p.copy()
            kin_v_before = d.airframe.v.copy()
            rot_omega_before = d.airframe.omega.copy()
            _, rot_motor_torque_body, _, _ = d.airframe.step(
                omega_cmds, episode_wind, 1.225, ice_index=curr_ice,
                surface_z=current_surf_z, has_payload=d.winch.has_payload, dt=dt)
            kin_target = _numpy_one_step_kinematics_target(
                kin_p_before, kin_v_before, d.airframe.p, d.airframe.v, dt,
                angular_velocity=d.airframe.omega)
            step_ykin_triad.append(kin_target)
            action_p0, action_v0, action_omega0, action_surface_z, action_chute_before = action_transition_state_before
            action_contact_free = bool(
                action_p0[2] > action_surface_z + ROTATIONAL_DYNAMICS_MIN_SURFACE_CLEARANCE_M
                and float(d.airframe.p[2]) > action_surface_z + ROTATIONAL_DYNAMICS_MIN_SURFACE_CLEARANCE_M
                and not action_chute_before and not bool(d.airframe.parachute_deployed))
            step_yaction_transition_triad[-1][:] = _numpy_one_step_action_transition_target(
                action_p0, action_v0, action_omega0,
                d.airframe.p, d.airframe.v, d.airframe.omega, dt,
                transition_valid=bool(kin_target[6] > 0.5), contact_free=action_contact_free)
            rot_contact_free = bool(
                kin_p_before[2] > current_surf_z + ROTATIONAL_DYNAMICS_MIN_SURFACE_CLEARANCE_M
                and d.airframe.p[2] > current_surf_z + ROTATIONAL_DYNAMICS_MIN_SURFACE_CLEARANCE_M
                and not d.airframe.parachute_deployed
            )
            step_yrot_triad.append(_numpy_rotational_dynamics_residual_target(
                rot_omega_before, d.airframe.omega, rot_motor_torque_body, dt,
                transition_valid=bool(kin_target[6] > 0.5), contact_free=rot_contact_free))
            # Recompute relative cargo state against the post-step airframe pose;
            # the winch's PBD nodes were advanced inside d.winch.step().
            cargo_sway_after = (np.asarray(d.winch.nodes_pos[-1], dtype=np.float64)[:2]
                                - np.asarray(d.airframe.p, dtype=np.float64)[:2])
            cargo_sway_rate_after = (np.asarray(d.winch.nodes_vel[-1], dtype=np.float64)[:2]
                                     - np.asarray(d.airframe.v, dtype=np.float64)[:2])
            step_ycargo_triad.append(_numpy_one_step_cargo_dynamics_target(
                cargo_sway_after, cargo_sway_rate_after, d.winch.tension_n,
                cargo_payload_before_step, cargo_cable_len_before_step,
                bool(d.winch.has_payload), float(d.winch.cable_len),
                transition_valid=bool(kin_target[6] > 0.5)))

        obs_list.extend(step_obs_triad)
        y_guidance.extend(step_yg_triad)
        y_fsm.extend(step_yf_triad)
        y_winch.extend(step_yw_triad)
        y_health.extend(step_yh_triad)
        y_trust.extend(step_yt_triad)
        y_val.extend(step_yv_triad)
        y_kinematics.extend(step_ykin_triad)
        y_cargo_dynamics.extend(step_ycargo_triad)
        y_rotational_dynamics.extend(step_yrot_triad)
        y_trajectory_forecast.extend(step_ytrajectory_triad)
        y_action_transition.extend(step_yaction_transition_triad)
        sim_frames += 1

    if harvest_dynamics_resets:
        reasons_text = ", ".join(
            f"{name}={count}" for name, count in sorted(harvest_fault_reason_counts.items())
        )
        print(
            f"[HARVEST DYNAMICS SUMMARY] worker={worker_id} total_resets={harvest_dynamics_resets}; "
            f"reasons: {reasons_text}. All invalid states were reset before observation/label generation.",
            flush=True,
        )

    temp_dir = tempfile.gettempdir()
    chunk_file = os.path.join(temp_dir, f"swarm_chunk_{worker_id}_{os.getpid()}.npz")
    obs_worker_np = np.asarray(obs_list, dtype=np.float32)
    ykin_worker_np = np.asarray(y_kinematics, dtype=np.float32)
    yenv_worker_np = _numpy_one_step_environment_transition_targets(obs_worker_np, ykin_worker_np)
    np.savez_compressed(
        chunk_file,
        obs=obs_worker_np,
        yg=np.array(y_guidance, dtype=np.float32),
        yf=np.array(y_fsm, dtype=np.int64),
        yw=np.array(y_winch, dtype=np.float32),
        yh=np.array(y_health, dtype=np.float32),
        yt=np.array(y_trust, dtype=np.float32),
        yv=np.array(y_val, dtype=np.float32),
        ykin=ykin_worker_np,
        yenvironment=yenv_worker_np,
        ycargo=np.array(y_cargo_dynamics, dtype=np.float32),
        yrot=np.array(y_rotational_dynamics, dtype=np.float32),
        ytrajectory=np.array(y_trajectory_forecast, dtype=np.float32),
        yaction_transition=np.array(y_action_transition, dtype=np.float32)
    )
    return chunk_file

def _save_transition_rollout_sequence_artifact(chunk_files, heldout_worker_ids, output_path,
                                                sequence_length=128, segments_per_drone=8):
    """Save ordered per-drone sequences from genuinely held-out harvest workers.

    Worker chunks are frame-major and then drone-major: every simulation frame emits
    NUM_SWARM_DRONES consecutive rows. Sequence windows are sampled evenly through
    each held-out episode so evaluation covers more than its opening state. Each row
    contains the real 417-D observation, the paired high-level action, and measured
    one-step transition target. This artifact is specifically for teacher-forced
    accumulated multi-step evaluation; it is not a free-running observation simulator.
    """
    path = os.path.abspath(os.fspath(output_path))
    if heldout_worker_ids is None:
        return {"status": "BLOCKED", "path": path,
                "reason": "worker-held-out validation unavailable; row-random fallback cannot support independent sequence evaluation"}
    worker_ids = sorted({int(x) for x in np.asarray(heldout_worker_ids).reshape(-1).tolist()})
    if not worker_ids:
        return {"status": "BLOCKED", "path": path, "reason": "no held-out worker IDs supplied"}
    sequence_length = int(np.clip(int(sequence_length), 64, 512))
    segments_per_drone = int(np.clip(int(segments_per_drone), 1, 16))

    observations_out = []
    actions_out = []
    targets_out = []
    worker_id_out = []
    drone_id_out = []
    start_frame_out = []
    valid_out = []
    used_workers = []
    total_rows = 0
    valid_rows = 0

    for worker_id in worker_ids:
        if worker_id < 0 or worker_id >= len(chunk_files):
            continue
        chunk_path = chunk_files[worker_id]
        try:
            with np.load(chunk_path, allow_pickle=False) as chunk:
                obs = np.asarray(chunk["obs"], dtype=np.float32)
                guidance = np.asarray(chunk["yg"], dtype=np.float32)
                fsm = np.asarray(chunk["yf"], dtype=np.int64)
                winch = np.asarray(chunk["yw"], dtype=np.float32)
                targets = np.asarray(chunk["yaction_transition"], dtype=np.float32)
        except (OSError, KeyError, ValueError) as exc:
            print(f"[TRANSITION ROLLOUT SEQUENCES WARNING] worker={worker_id} cannot read chunk: {exc}", flush=True)
            continue
        n = len(obs)
        if (n < NUM_SWARM_DRONES * sequence_length or n % NUM_SWARM_DRONES != 0
                or guidance.shape[0] != n or fsm.shape[0] != n or winch.shape[0] != n
                or targets.shape != (n, ACTION_TRANSITION_TARGET_DIM)
                or obs.shape != (n, OBSERVATION_DIM)
                or guidance.ndim != 2 or guidance.shape[1] < 3
                or winch.ndim != 2 or winch.shape[1] < 1):
            print(f"[TRANSITION ROLLOUT SEQUENCES WARNING] worker={worker_id} invalid shapes/order; skipped", flush=True)
            continue
        frames = n // NUM_SWARM_DRONES
        obs3 = obs.reshape(frames, NUM_SWARM_DRONES, OBSERVATION_DIM).transpose(1, 0, 2)
        guidance3 = guidance.reshape(frames, NUM_SWARM_DRONES, -1).transpose(1, 0, 2)
        fsm3 = np.clip(fsm.reshape(frames, NUM_SWARM_DRONES).transpose(1, 0),
                       0, NUM_EXECUTIVE_STATES - 1)
        winch3 = winch.reshape(frames, NUM_SWARM_DRONES, -1).transpose(1, 0, 2)
        target3 = targets.reshape(frames, NUM_SWARM_DRONES, ACTION_TRANSITION_TARGET_DIM).transpose(1, 0, 2)
        fsm_onehot = np.eye(NUM_EXECUTIVE_STATES, dtype=np.float32)[fsm3]
        action3 = np.concatenate((
            np.nan_to_num(guidance3[..., :3], nan=0.0, posinf=1.0, neginf=-1.0),
            fsm_onehot,
            np.clip(winch3[..., :1], -1.0, 1.0)), axis=-1).astype(np.float32)
        if action3.shape[-1] != ACTION_TRANSITION_ACTION_DIM:
            continue

        actual_length = min(sequence_length, frames)
        if actual_length < 64:
            continue
        # Evenly space windows over the whole held-out rollout. Each window remains
        # in native temporal order and is never stitched across separate episodes.
        max_start = frames - actual_length
        starts = np.unique(np.linspace(0, max_start, num=segments_per_drone, dtype=np.int64))
        worker_added_any = False
        for drone_id in range(NUM_SWARM_DRONES):
            for start in starts.tolist():
                stop = int(start) + actual_length
                ob = obs3[drone_id, int(start):stop]
                ac = action3[drone_id, int(start):stop]
                ta = target3[drone_id, int(start):stop]
                row_valid = ((ta[:, -1] > 0.5)
                             & np.isfinite(ta[:, :-1]).all(axis=1)
                             & np.isfinite(ob).all(axis=1)
                             & np.isfinite(ac).all(axis=1))
                if int(row_valid.sum()) < 32:
                    continue
                # Fixed-length artifact simplifies batch validation. If a chunk is
                # shorter than the requested length, pad with invalid zero rows.
                if actual_length < sequence_length:
                    pad = sequence_length - actual_length
                    ob = np.pad(ob, ((0, pad), (0, 0)))
                    ac = np.pad(ac, ((0, pad), (0, 0)))
                    ta = np.pad(ta, ((0, pad), (0, 0)))
                    ta[actual_length:, -1] = 0.0
                observations_out.append(ob.astype(np.float32, copy=False))
                actions_out.append(ac.astype(np.float32, copy=False))
                targets_out.append(ta.astype(np.float32, copy=False))
                worker_id_out.append(worker_id)
                drone_id_out.append(drone_id)
                start_frame_out.append(int(start))
                valid_out.append(int(row_valid.sum()))
                total_rows += int(len(row_valid))
                valid_rows += int(row_valid.sum())
                worker_added_any = True
        if worker_added_any:
            used_workers.append(worker_id)

    if not observations_out:
        return {"status": "BLOCKED", "path": path,
                "reason": "no sequence segment had at least 32 valid transitions; increase worker rollout coverage or inspect transition masks"}
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temp_path = path + f".tmp-{os.getpid()}"
    metadata = {
        "schema_version": 1,
        "sequence_length": sequence_length,
        "action_condition_dim": ACTION_TRANSITION_ACTION_DIM,
        "observation_dim": OBSERVATION_DIM,
        "transition_target_dim": ACTION_TRANSITION_TARGET_DIM,
        "dt_s": 0.016,
        "split": "whole worker held out from Phase 1/2 train rows",
        "evaluation_mode": "teacher-forced accumulated transition error; not free-running MPC",
        "worker_ids": used_workers,
        "source_worker_count": len(chunk_files),
    }
    try:
        with open(temp_path, "wb") as out_file:
            np.savez_compressed(
                out_file,
                observations=np.stack(observations_out).astype(np.float32),
                actions=np.stack(actions_out).astype(np.float32),
                targets=np.stack(targets_out).astype(np.float32),
                worker_ids=np.asarray(worker_id_out, dtype=np.int16),
                drone_ids=np.asarray(drone_id_out, dtype=np.int8),
                start_frames=np.asarray(start_frame_out, dtype=np.int32),
                valid_counts=np.asarray(valid_out, dtype=np.int16),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        os.replace(temp_path, path)
    finally:
        try:
            if os.path.isfile(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
    return {"status": "SAVED", "path": path, "sequence_count": len(observations_out),
            "worker_ids": used_workers, "valid_rows": valid_rows, "total_rows": total_rows,
            "sequence_length": sequence_length, "segments_per_drone_requested": segments_per_drone,
            "evaluation_mode": "teacher-forced accumulated transition error; not free-running MPC"}


def _transition_multistep_rollout_metrics(observations, actions, targets, predictions,
                                          sequence_lengths, horizons_steps=(1, 2, 4, 8, 16, 32, 64),
                                          dt_s=0.016):
    """Evaluate accumulated delta prediction across actual held-out sequences.

    The model is evaluated on the actual observation and actual recorded action at
    every timestep, then predicted deltas are accumulated over each horizon. This
    quantifies accumulated one-step model error under teacher forcing; it is NOT a
    free-running rollout because observations are not rebuilt from predicted states.
    """
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    predictions = np.asarray(predictions, dtype=np.float32)
    if observations.ndim != 3 or observations.shape[-1] != OBSERVATION_DIM:
        raise ValueError("observations must have shape [sequence,time,OBSERVATION_DIM]")
    nseq, steps, _ = observations.shape
    if actions.shape != (nseq, steps, ACTION_TRANSITION_ACTION_DIM):
        raise ValueError("actions must align with observations and action-conditioning schema")
    if targets.shape != (nseq, steps, ACTION_TRANSITION_TARGET_DIM):
        raise ValueError("targets must align with observations and action-transition target schema")
    if predictions.shape != (nseq, steps, ACTION_TRANSITION_OUTPUT_DIM):
        raise ValueError("predictions must align with targets and action-transition output schema")
    sequence_lengths = np.asarray(sequence_lengths, dtype=np.int64).reshape(nseq)
    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0 or dt > 0.1:
        raise ValueError("dt_s must be finite and in (0,0.1]")

    metric_results = []
    scales = np.asarray([ACTION_TRANSITION_POSITION_SCALE_M] * 3
                        + [ACTION_TRANSITION_VELOCITY_SCALE_MS] * 3
                        + [ACTION_TRANSITION_ANGULAR_RATE_SCALE_RADS] * 3, dtype=np.float64)
    cidx = (HISTORY_LEN - 1) * BASE_OBS_DIM
    for horizon in sorted({int(h) for h in horizons_steps if int(h) >= 1}):
        learned_errors = []
        baseline_errors = []
        windows = 0
        for seq_idx in range(nseq):
            length = int(np.clip(sequence_lengths[seq_idx], 0, steps))
            if length < horizon:
                continue
            obs = observations[seq_idx, :length].astype(np.float64, copy=False)
            ac = actions[seq_idx, :length].astype(np.float64, copy=False)
            tar = targets[seq_idx, :length].astype(np.float64, copy=False)
            pred = predictions[seq_idx, :length].astype(np.float64, copy=False)
            row_valid = ((tar[:, -1] > 0.5)
                         & np.isfinite(obs).all(axis=1)
                         & np.isfinite(ac).all(axis=1)
                         & np.isfinite(tar[:, :-1]).all(axis=1)
                         & np.isfinite(pred).all(axis=1))
            if not row_valid.any():
                continue
            predicted_delta = pred * scales.reshape(1, -1)
            actual_delta = tar[:, :ACTION_TRANSITION_OUTPUT_DIM] * scales.reshape(1, -1)
            velocity = obs[:, cidx + 6:cidx + 9] * 8.0
            baseline_delta = np.zeros_like(actual_delta)
            baseline_delta[:, 0:3] = velocity * dt
            pred_cum = np.vstack((np.zeros((1, ACTION_TRANSITION_OUTPUT_DIM)), np.cumsum(predicted_delta, axis=0)))
            actual_cum = np.vstack((np.zeros((1, ACTION_TRANSITION_OUTPUT_DIM)), np.cumsum(actual_delta, axis=0)))
            baseline_cum = np.vstack((np.zeros((1, ACTION_TRANSITION_OUTPUT_DIM)), np.cumsum(baseline_delta, axis=0)))
            valid_prefix = np.concatenate(([0], np.cumsum(row_valid.astype(np.int32))))
            starts = np.arange(0, length - horizon + 1, dtype=np.int64)
            all_valid = (valid_prefix[starts + horizon] - valid_prefix[starts]) == horizon
            starts = starts[all_valid]
            if starts.size == 0:
                continue
            p_err = pred_cum[starts + horizon] - pred_cum[starts] - (actual_cum[starts + horizon] - actual_cum[starts])
            b_err = baseline_cum[starts + horizon] - baseline_cum[starts] - (actual_cum[starts + horizon] - actual_cum[starts])
            learned_errors.append(p_err)
            baseline_errors.append(b_err)
            windows += int(starts.size)
        if windows:
            le = np.concatenate(learned_errors, axis=0)
            be = np.concatenate(baseline_errors, axis=0)
            learned_rmse = {
                "position_delta_rmse_m": float(np.sqrt(np.mean(le[:, 0:3] ** 2))),
                "velocity_delta_rmse_mps": float(np.sqrt(np.mean(le[:, 3:6] ** 2))),
                "angular_rate_delta_rmse_rads": float(np.sqrt(np.mean(le[:, 6:9] ** 2))),
            }
            baseline_rmse = {
                "position_delta_rmse_m": float(np.sqrt(np.mean(be[:, 0:3] ** 2))),
                "velocity_delta_rmse_mps": float(np.sqrt(np.mean(be[:, 3:6] ** 2))),
                "angular_rate_delta_rmse_rads": float(np.sqrt(np.mean(be[:, 6:9] ** 2))),
            }
            gain = {key: _action_transition_relative_gain_pct(baseline_rmse[key], learned_rmse[key])
                    for key in learned_rmse}
        else:
            learned_rmse = {"position_delta_rmse_m": None, "velocity_delta_rmse_mps": None,
                            "angular_rate_delta_rmse_rads": None}
            baseline_rmse = dict(learned_rmse)
            gain = dict(learned_rmse)
        metric_results.append({
            "horizon_steps": horizon, "horizon_s": horizon * dt, "valid_windows": windows,
            "learned_teacher_forced_accumulation": learned_rmse,
            "constant_velocity_accumulation_baseline": baseline_rmse,
            "gain_vs_baseline_pct": gain,
            "sufficient_samples": windows >= 32,
        })
    if not metric_results:
        raise ValueError("no valid horizons were requested")
    return {
        "status": "COMPLETED" if all(x["sufficient_samples"] for x in metric_results) else "COMPLETED_WITH_INSUFFICIENT_HORIZON_DATA",
        "evaluation_mode": "teacher-forced accumulated multi-step error",
        "free_running_rollout": False,
        "live_control_authorized": False,
        "dt_s": dt,
        "horizons": metric_results,
        "limitations": [
            "Uses the measured observation and recorded action at every step; prediction errors do not feed into subsequent observations.",
            "Measures accumulated one-step prediction error, not autonomous recursive rollout accuracy.",
            "Recorded actions are not counterfactual actions selected from predicted states.",
            "Passing metrics does not authorize live motor, winch, release, or safety control."
        ]
    }


def run_offline_transition_rollout_evaluation():
    """Run a gated multi-step accumulated transition evaluation offline only."""
    report_path = os.path.abspath(os.environ.get(
        "SURVIVAL_ACTION_TRANSITION_REPORT", "action_transition_validation_v38_42.json"))
    data_path = os.path.abspath(os.environ.get(
        "SURVIVAL_TRANSITION_ROLLOUT_DATASET", "transition_rollout_sequences_v38_40.npz"))
    output_path = os.path.abspath(os.environ.get(
        "SURVIVAL_TRANSITION_ROLLOUT_REPORT", "action_transition_rollout_eval_v38_40.json"))
    checkpoint_path = os.path.abspath(os.environ.get(
        "SURVIVAL_TRANSITION_ROLLOUT_CHECKPOINT", MODEL_WEIGHTS_FILE))

    if not os.path.isfile(report_path):
        raise FileNotFoundError(f"One-step readiness report is missing: {report_path}. Run V38.35 training first.")
    with open(report_path, "r", encoding="utf-8") as src:
        readiness_report = json.load(src)
    gate = readiness_report.get("rollout_readiness_gate", {}) if isinstance(readiness_report, dict) else {}
    if readiness_report.get("version") != SURVIVAL_CORE_VERSION:
        raise RuntimeError(
            f"Readiness report version mismatch: expected {SURVIVAL_CORE_VERSION}, got {readiness_report.get('version')!r}; rerun this version's training pipeline.")
    if gate.get("status") != "READY_FOR_OFFLINE_MULTISTEP_EVALUATION" or gate.get("live_control_authorized", True) is not False:
        reasons = gate.get("reasons", []) if isinstance(gate, dict) else []
        raise RuntimeError("Multi-step evaluation BLOCKED by one-step readiness gate: " + "; ".join(map(str, reasons[:12])))
    artifact_record = readiness_report.get("rollout_sequence_artifact", {})
    if not isinstance(artifact_record, dict) or artifact_record.get("status") != "SAVED":
        raise RuntimeError("Multi-step evaluation BLOCKED: the readiness report does not record a successfully saved held-out sequence artifact for this run.")
    recorded_data_path = artifact_record.get("path")
    if recorded_data_path and os.path.abspath(str(recorded_data_path)) != data_path:
        raise RuntimeError(f"Sequence dataset path differs from training report: expected {recorded_data_path}, got {data_path}")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Held-out ordered sequence artifact is missing: {data_path}. Run V38.35 training first.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Transition-model checkpoint is missing: {checkpoint_path}")

    device_text = os.environ.get("SURVIVAL_TRANSITION_ROLLOUT_DEVICE", "cpu").strip().lower()
    device = torch.device("cuda" if device_text == "cuda" and torch.cuda.is_available() else "cpu")
    if device_text == "cuda" and device.type != "cuda":
        print("[TRANSITION ROLLOUT] CUDA requested but unavailable; using CPU.", flush=True)
    model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
    _load_model_checkpoint(model, checkpoint_path, device)
    if not bool(getattr(model, "_action_conditioned_transition_head_checkpoint_trained", False)):
        raise RuntimeError("Multi-step evaluation BLOCKED: checkpoint does not contain a trained action-conditioned transition head.")
    model.eval()

    with np.load(data_path, allow_pickle=False) as dataset:
        observations = np.asarray(dataset["observations"], dtype=np.float32)
        actions = np.asarray(dataset["actions"], dtype=np.float32)
        targets = np.asarray(dataset["targets"], dtype=np.float32)
        metadata = json.loads(str(np.asarray(dataset["metadata_json"]).item()))
        seq_len = int(metadata.get("sequence_length", observations.shape[1]))
        worker_ids = np.asarray(dataset["worker_ids"], dtype=np.int16)
        drone_ids = np.asarray(dataset["drone_ids"], dtype=np.int8)
        start_frames = np.asarray(dataset["start_frames"], dtype=np.int32)
    if observations.ndim != 3:
        raise ValueError(f"Sequence artifact observations must be [N,T,D], got {observations.shape}")
    if observations.shape[1] != seq_len:
        raise ValueError("Sequence artifact length does not match metadata")
    if actions.shape != (observations.shape[0], seq_len, ACTION_TRANSITION_ACTION_DIM):
        raise ValueError("Sequence artifact action shape mismatch")
    if targets.shape != (observations.shape[0], seq_len, ACTION_TRANSITION_TARGET_DIM):
        raise ValueError("Sequence artifact target shape mismatch")
    if worker_ids.shape != (observations.shape[0],) or drone_ids.shape != worker_ids.shape or start_frames.shape != worker_ids.shape:
        raise ValueError("Sequence artifact identity arrays do not align with sequences")

    nseq, steps, _ = observations.shape
    flat_obs = observations.reshape(-1, OBSERVATION_DIM)
    flat_actions = actions.reshape(-1, ACTION_TRANSITION_ACTION_DIM)
    prediction_flat = np.zeros((len(flat_obs), ACTION_TRANSITION_OUTPUT_DIM), dtype=np.float32)
    batch_size = int(np.clip(int(os.environ.get("SURVIVAL_TRANSITION_ROLLOUT_BATCH_SIZE", "512")), 32, 4096))
    with torch.no_grad():
        for start in range(0, len(flat_obs), batch_size):
            stop = min(len(flat_obs), start + batch_size)
            out = model(
                torch.from_numpy(flat_obs[start:stop]).to(device),
                return_action_transition=True,
                transition_action=torch.from_numpy(flat_actions[start:stop]).to(device))
            prediction_flat[start:stop] = out[-1].detach().cpu().numpy().astype(np.float32, copy=False)
    predictions = prediction_flat.reshape(nseq, steps, ACTION_TRANSITION_OUTPUT_DIM)
    lengths = np.full(nseq, steps, dtype=np.int32)
    metrics = _transition_multistep_rollout_metrics(
        observations, actions, targets, predictions, lengths,
        horizons_steps=(1, 2, 4, 8, 16, 32, 64), dt_s=float(metadata.get("dt_s", 0.016)))
    metrics.update({
        "version": SURVIVAL_CORE_VERSION,
        "readiness_report": report_path,
        "sequence_dataset": data_path,
        "checkpoint": checkpoint_path,
        "sequence_count": int(nseq),
        "unique_workers": sorted(set(int(x) for x in worker_ids.tolist())),
        "samples_per_sequence": int(steps),
        "worker_ids": worker_ids.tolist(),
        "drone_ids": drone_ids.tolist(),
        "start_frames": start_frames.tolist(),
        "live_control_authorized": False,
    })
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    temp_path = output_path + f".tmp-{os.getpid()}"
    try:
        with open(temp_path, "w", encoding="utf-8") as dst:
            json.dump(_action_transition_json_safe(metrics), dst, indent=2, sort_keys=True, allow_nan=False)
        os.replace(temp_path, output_path)
    finally:
        try:
            if os.path.isfile(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
    print(f"[TRANSITION MULTISTEP EVAL] status={metrics['status']} sequences={nseq} horizons="
          + ", ".join(f"{x['horizon_s']:.3f}s:n={x['valid_windows']}" for x in metrics["horizons"]), flush=True)
    print(f"[TRANSITION MULTISTEP REPORT] {output_path}; teacher-forced accumulation only; live_control_authorized=False", flush=True)
    return metrics



# =============================================================================
# V38.37 — RECURSIVE KINEMATIC ROLLOUT EVALUATION (OFFLINE ONLY)
# =============================================================================
RECURSIVE_ROLLOUT_VERSION = "V38.40-authoritative-live-transition-shadow-validation"
RECURSIVE_ROLLOUT_HORIZONS = (1, 2, 4, 8, 16, 32, 64)
RECURSIVE_ROLLOUT_LIVE_CONTROL_AUTHORIZED = False


def _so3_exp_batch(rotation_vectors):
    """Vectorised SO(3) exponential map for body-frame rotation vectors [N,3]."""
    vec = np.asarray(rotation_vectors, dtype=np.float64)
    if vec.ndim != 2 or vec.shape[1] != 3:
        raise ValueError("rotation_vectors must have shape [N,3]")
    n = vec.shape[0]
    theta = np.linalg.norm(vec, axis=1)
    k = np.zeros((n, 3, 3), dtype=np.float64)
    k[:, 0, 1] = -vec[:, 2]
    k[:, 0, 2] = vec[:, 1]
    k[:, 1, 0] = vec[:, 2]
    k[:, 1, 2] = -vec[:, 0]
    k[:, 2, 0] = -vec[:, 1]
    k[:, 2, 1] = vec[:, 0]
    theta2 = theta * theta
    a = np.ones_like(theta)
    b = np.full_like(theta, 0.5)
    nz = theta > 1e-7
    a[nz] = np.sin(theta[nz]) / theta[nz]
    b[nz] = (1.0 - np.cos(theta[nz])) / theta2[nz]
    k2 = np.matmul(k, k)
    ident = np.broadcast_to(np.eye(3, dtype=np.float64), (n, 3, 3))
    return ident + a[:, None, None] * k + b[:, None, None] * k2


def _recursive_update_observation_batch(observation_batch, predicted_delta_normalized, dt_s=0.016,
                                        predicted_environment_state=None, predicted_cargo_state=None):
    """Update reconstructable kinematics plus learned wind/cable/ice and cargo state.

    Transition output is normalized [delta-p/0.25m, delta-v/1m/s, delta-omega/2rad/s].
    Rebuilds target-relative position, absolute position, velocity, rotation, body rate,
    and derived accelerometer/gyro features. Optional learned predictions update normalized
    wind, cable length/velocity, icing and loaded-cargo sway/tension. Hook IMU and load-cell
    proxies are regenerated using the observation synthesizer's simple equations. Lidar,
    peers, battery/ESC/thermal state and unavailable world channels remain frozen: this is
    still a partial-state rollout, not a complete sensor/world simulator.
    """
    obs = np.asarray(observation_batch, dtype=np.float32)
    pred = np.asarray(predicted_delta_normalized, dtype=np.float32)
    dt_s = float(dt_s)
    if obs.ndim != 2 or obs.shape[1] != OBSERVATION_DIM:
        raise ValueError(f"observation_batch must be [N,{OBSERVATION_DIM}]")
    if pred.shape != (obs.shape[0], ACTION_TRANSITION_OUTPUT_DIM):
        raise ValueError(f"predicted_delta_normalized must be [N,{ACTION_TRANSITION_OUTPUT_DIM}]")
    env_pred = None if predicted_environment_state is None else np.asarray(predicted_environment_state, dtype=np.float32)
    cargo_pred = None if predicted_cargo_state is None else np.asarray(predicted_cargo_state, dtype=np.float32)
    if env_pred is not None and env_pred.shape != (obs.shape[0], ENVIRONMENT_TRANSITION_DIM):
        raise ValueError(f"predicted_environment_state must be [N,{ENVIRONMENT_TRANSITION_DIM}]")
    if cargo_pred is not None and cargo_pred.shape != (obs.shape[0], 5):
        raise ValueError("predicted_cargo_state must be [N,5]")
    if not math.isfinite(dt_s) or not (0.0 < dt_s <= 0.05):
        raise ValueError("dt_s must be finite and in (0,0.05]")
    if not np.isfinite(obs).all() or not np.isfinite(pred).all():
        raise ValueError("recursive observation update requires finite inputs")

    frames = obs.reshape(-1, HISTORY_LEN, BASE_OBS_DIM)
    current = frames[:, -1, :].copy()
    dp = pred[:, 0:3] * ACTION_TRANSITION_POSITION_SCALE_M
    dv = pred[:, 3:6] * ACTION_TRANSITION_VELOCITY_SCALE_MS
    dw = pred[:, 6:9] * ACTION_TRANSITION_ANGULAR_RATE_SCALE_RADS

    p0 = current[:, 3:6].astype(np.float64) * 200.0
    v0 = current[:, 6:9].astype(np.float64) * 8.0
    omega0 = current[:, 18:21].astype(np.float64) * 10.0
    p1 = p0 + dp.astype(np.float64)
    v1 = v0 + dv.astype(np.float64)
    omega1 = omega0 + dw.astype(np.float64)

    # The normalized target-relative vector and absolute position encode a fixed local
    # waypoint. Restricting rollout windows to a single FSM phase reduces, but cannot
    # completely rule out, target changes within that phase.
    target_p = p0 + current[:, 0:3].astype(np.float64) * 50.0
    current[:, 0:3] = ((target_p - p1) / 50.0).astype(np.float32)
    current[:, 3:6] = (p1 / 200.0).astype(np.float32)
    current[:, 6:9] = (v1 / 8.0).astype(np.float32)

    # Observation R is a flattened body-to-world rotation matrix. Integrate body rates
    # with a midpoint-rate exponential map: R_next = R_now Exp(omega_mid * dt).
    rotations = current[:, 9:18].reshape(-1, 3, 3).astype(np.float64)
    omega_mid = 0.5 * (omega0 + omega1)
    delta_R = _so3_exp_batch(omega_mid * dt_s)
    rotations_next = np.matmul(rotations, delta_R)
    current[:, 9:18] = rotations_next.reshape(-1, 9).astype(np.float32)
    current[:, 18:21] = (omega1 / 10.0).astype(np.float32)

    # These sensor features are tied directly to aircraft acceleration/gyro and can be
    # refreshed from the recursive estimate. Other sensor channels remain frozen.
    current[:, 130:133] = ((dv.astype(np.float64) / dt_s) / 30.0).astype(np.float32)
    current[:, 133:136] = (omega1 / 10.0).astype(np.float32)

    # The environment head predicts normalized next-state values for the six
    # explicitly supervised channels: sensed wind xyz, cable length, cable rate, ice.
    if env_pred is not None:
        if not np.isfinite(env_pred).all():
            raise ValueError("environment prediction contains non-finite values")
        current[:, 42:45] = np.clip(env_pred[:, 0:3], -2.0, 2.0)
        current[:, 53] = np.clip(env_pred[:, 3], 0.0, 1.0)
        current[:, 54] = np.clip(env_pred[:, 4], -2.0, 2.0)
        current[:, 65] = np.clip(env_pred[:, 5], 0.0, 1.0)

    # The existing cargo head predicts sway/rate/tension in the trained SI scales.
    # Do not hallucinate payload state: only apply loaded-cable forecasts where the
    # discrete observed payload flag remains active in the replayed mission context.
    if cargo_pred is not None:
        if not np.isfinite(cargo_pred).all():
            raise ValueError("cargo prediction contains non-finite values")
        active_cargo = (current[:, 58] > 0.5) & (current[:, 53] > (0.12 / 15.0))
        current[~active_cargo, 55:58] = 0.0
        if np.any(active_cargo):
            sway_xy = np.clip(cargo_pred[:, 0:2] * CARGO_DYNAMICS_SWAY_SCALE_M, -10.0, 10.0)
            sway_rate_xy = np.clip(cargo_pred[:, 2:4] * CARGO_DYNAMICS_SWAY_RATE_SCALE_MS, -24.0, 24.0)
            tension_n = np.clip(cargo_pred[:, 4] * CARGO_DYNAMICS_TENSION_SCALE_N, 0.0, 45.0)
            current[active_cargo, 56:58] = (sway_xy[active_cargo] / 2.0)
            current[active_cargo, 55] = tension_n[active_cargo] / 50.0
            # Reconstruct the simple hook-IMU sensor proxies from the same equations
            # as poll_smart_hook_imu, using recursively predicted acceleration/sway.
            accel_world = (dv.astype(np.float64) / dt_s)
            hook_accel = accel_world.copy()
            hook_accel[:, 0] -= sway_xy[:, 0] * 2.5
            hook_accel[:, 1] -= sway_xy[:, 1] * 2.5
            hook_accel[:, 2] += 0.2
            sway_speed = np.linalg.norm(sway_rate_xy, axis=1)
            hook_gyro = np.stack((sway_rate_xy[:, 1] * 1.8,
                                  -sway_rate_xy[:, 0] * 1.8,
                                  sway_speed * 0.5), axis=1)
            current[active_cargo, 85:88] = hook_accel[active_cargo] / 20.0
            current[active_cargo, 88:91] = hook_gyro[active_cargo] / 10.0
            sway_dist = np.maximum(1e-4, np.linalg.norm(sway_xy, axis=1))
            sin_theta = np.minimum(0.65, sway_dist / np.maximum(0.2, current[:, 53] * 15.0))
            fx = -tension_n * (sway_xy[:, 0] / sway_dist) * sin_theta
            fy = -tension_n * (sway_xy[:, 1] / sway_dist) * sin_theta
            fz = tension_n * np.sqrt(np.maximum(0.0, 1.0 - sin_theta * sin_theta))
            current[active_cargo, 91:94] = np.stack((fx, fy, fz), axis=1)[active_cargo] / 50.0

    next_frames = np.concatenate((frames[:, 1, :], frames[:, 2, :], current), axis=1)
    if not np.isfinite(next_frames).all():
        raise FloatingPointError("recursive observation update produced non-finite features")
    # State extracted from the predicted latest base frame for horizon metrics.
    latest = next_frames.reshape(-1, HISTORY_LEN, BASE_OBS_DIM)[:, -1, :]
    state = np.concatenate((latest[:, 3:6] * 200.0,
                            latest[:, 6:9] * 8.0,
                            latest[:, 18:21] * 10.0), axis=1).astype(np.float64)
    return next_frames.astype(np.float32), state


def _recursive_rollout_state_from_observations(observations):
    """Extract world position, world velocity and body angular rate from each 417-D row."""
    x = np.asarray(observations, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != OBSERVATION_DIM:
        raise ValueError("observations must be [N,OBSERVATION_DIM]")
    c = x[:, (HISTORY_LEN - 1) * BASE_OBS_DIM:]
    return np.concatenate((c[:, 3:6] * 200.0, c[:, 6:9] * 8.0, c[:, 18:21] * 10.0), axis=1).astype(np.float64)


def _recursive_rollout_metrics_from_errors(learned_errors, baseline_errors, horizon, candidate_count,
                                           completed_count, divergent_count):
    """Metric helper with explicit units and no promotion from sparse/partial runs."""
    if completed_count:
        le = np.asarray(learned_errors, dtype=np.float64)
        be = np.asarray(baseline_errors, dtype=np.float64)
        learned = {
            "position_rmse_m": float(np.sqrt(np.mean(np.sum(le[:, 0:3] ** 2, axis=1)))),
            "velocity_rmse_mps": float(np.sqrt(np.mean(np.sum(le[:, 3:6] ** 2, axis=1)))),
            "angular_rate_rmse_rads": float(np.sqrt(np.mean(np.sum(le[:, 6:9] ** 2, axis=1)))),
        }
        baseline = {
            "position_rmse_m": float(np.sqrt(np.mean(np.sum(be[:, 0:3] ** 2, axis=1)))),
            "velocity_rmse_mps": float(np.sqrt(np.mean(np.sum(be[:, 3:6] ** 2, axis=1)))),
            "angular_rate_rmse_rads": float(np.sqrt(np.mean(np.sum(be[:, 6:9] ** 2, axis=1)))),
        }
        gain = {k: _action_transition_relative_gain_pct(baseline[k], learned[k]) for k in learned}
    else:
        learned = {"position_rmse_m": None, "velocity_rmse_mps": None, "angular_rate_rmse_rads": None}
        baseline = dict(learned)
        gain = dict(learned)
    return {
        "horizon_steps": int(horizon), "horizon_s": float(horizon * 0.016),
        "candidate_windows": int(candidate_count), "completed_windows": int(completed_count),
        "divergent_windows": int(divergent_count), "partial_recursive_learned_model": learned,
        "constant_velocity_baseline_from_initial_state": baseline,
        "gain_vs_baseline_pct": gain, "sufficient_samples": int(completed_count) >= 32,
    }


def _recursive_error_magnitude_summary(error_vectors, seed=0, bootstrap_replicates=250):
    """Summarise endpoint vector-error magnitudes with deterministic RMSE uncertainty.

    The bootstrap interval quantifies sampling uncertainty of the empirical RMSE. It is
    NOT a predictive interval and does not imply that the model's uncertainty head is
    calibrated. Input rows are independent evaluation windows only approximately; windows
    from a common held-out episode can remain correlated, so this interval is descriptive.
    """
    arr = np.asarray(error_vectors, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("error_vectors must have shape [N,D]")
    if arr.shape[0] == 0:
        return {"count": 0, "rmse": None, "mean_norm": None, "median_norm": None,
                "p95_norm": None, "max_norm": None, "bootstrap_95pct_ci_rmse": None}
    finite = np.isfinite(arr).all(axis=1)
    arr = arr[finite]
    if not len(arr):
        return {"count": 0, "rmse": None, "mean_norm": None, "median_norm": None,
                "p95_norm": None, "max_norm": None, "bootstrap_95pct_ci_rmse": None}
    norms = np.linalg.norm(arr, axis=1)
    rmse = float(np.sqrt(np.mean(norms ** 2)))
    ci = None
    n = len(norms)
    if n >= 2 and bootstrap_replicates > 0:
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        boot = np.empty(int(bootstrap_replicates), dtype=np.float64)
        # Chunk draws to keep temporary memory bounded on large held-out artifacts.
        chunk = max(1, min(32, int(bootstrap_replicates)))
        for start in range(0, int(bootstrap_replicates), chunk):
            stop = min(int(bootstrap_replicates), start + chunk)
            indices = rng.integers(0, n, size=(stop-start, n))
            boot[start:stop] = np.sqrt(np.mean(norms[indices] ** 2, axis=1))
        ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    return {"count": int(n), "rmse": rmse,
            "mean_norm": float(np.mean(norms)),
            "median_norm": float(np.quantile(norms, 0.50)),
            "p95_norm": float(np.quantile(norms, 0.95)),
            "max_norm": float(np.max(norms)),
            "bootstrap_95pct_ci_rmse": ci}


def _recursive_error_distribution_report(learned_errors, baseline_errors, horizon_steps):
    """Per-horizon error distributions and bootstrap RMSE intervals in physical units."""
    learned = np.asarray(learned_errors, dtype=np.float64)
    baseline = np.asarray(baseline_errors, dtype=np.float64)
    if learned.ndim != 2 or learned.shape[1] != 9:
        if learned.size == 0:
            learned = np.zeros((0, 9), dtype=np.float64)
        else:
            raise ValueError("learned_errors must have shape [N,9]")
    if baseline.ndim != 2 or baseline.shape != learned.shape:
        if learned.shape[0] == 0 and baseline.size == 0:
            baseline = np.zeros((0, 9), dtype=np.float64)
        else:
            raise ValueError("baseline_errors must match learned_errors [N,9]")
    axes = {
        "position_m": (0, 3),
        "velocity_mps": (3, 6),
        "angular_rate_rads": (6, 9),
    }
    result = {}
    for metric_name, (lo, hi) in axes.items():
        result[metric_name] = {
            "learned": _recursive_error_magnitude_summary(learned[:, lo:hi],
                seed=381_000 + int(horizon_steps) * 31 + lo),
            "constant_velocity_baseline": _recursive_error_magnitude_summary(baseline[:, lo:hi],
                seed=382_000 + int(horizon_steps) * 31 + lo),
        }
    return result


def _recursive_context_groups(context):
    """Map a window's start condition to grounded regime labels."""
    groups = []
    wind = float(context.get("wind_mps", float("nan")))
    if math.isfinite(wind):
        groups.append(("wind", "calm_lt3mps" if wind < 3.0 else
                       "moderate_3_to_5p5mps" if wind < 5.5 else "strong_ge5p5mps"))
    groups.append(("payload", "payload_active" if bool(context.get("payload_active", False)) else "payload_inactive"))
    groups.append(("winch", str(context.get("winch_regime", "unknown"))))
    groups.append(("fsm", str(context.get("fsm_name", "unknown"))))
    return groups


def _recursive_stratified_metrics(contexts, completed_indices, learned_errors, baseline_errors,
                                  min_group_samples=32):
    """Compare learned and baseline errors for completed windows, retaining divergence counts."""
    all_contexts = list(contexts)
    completed_indices = np.asarray(completed_indices, dtype=np.int64)
    learned_errors = np.asarray(learned_errors, dtype=np.float64).reshape(-1, 9)
    baseline_errors = np.asarray(baseline_errors, dtype=np.float64).reshape(-1, 9)
    if len(completed_indices) != len(learned_errors) or len(learned_errors) != len(baseline_errors):
        raise ValueError("stratified metric indices and errors are not aligned")
    completed_map = {int(idx): row for row, idx in enumerate(completed_indices.tolist())}
    group_candidates = {}
    for idx, ctx in enumerate(all_contexts):
        for group_type, group_name in _recursive_context_groups(ctx):
            group_candidates.setdefault((group_type, group_name), []).append(idx)
    output = []
    for (group_type, group_name), candidate_indices in sorted(group_candidates.items()):
        completed_rows = [completed_map[idx] for idx in candidate_indices if idx in completed_map]
        n_candidate = len(candidate_indices)
        n_completed = len(completed_rows)
        if n_completed:
            le = learned_errors[completed_rows]
            be = baseline_errors[completed_rows]
            def rmses(error_arr):
                return {
                    "position_rmse_m": float(np.sqrt(np.mean(np.sum(error_arr[:, 0:3] ** 2, axis=1)))),
                    "velocity_rmse_mps": float(np.sqrt(np.mean(np.sum(error_arr[:, 3:6] ** 2, axis=1)))),
                    "angular_rate_rmse_rads": float(np.sqrt(np.mean(np.sum(error_arr[:, 6:9] ** 2, axis=1)))),
                }
            le_metrics, be_metrics = rmses(le), rmses(be)
            gain = {k: _action_transition_relative_gain_pct(be_metrics[k], le_metrics[k])
                    for k in le_metrics}
        else:
            le_metrics = {k: None for k in ("position_rmse_m", "velocity_rmse_mps", "angular_rate_rmse_rads")}
            be_metrics = dict(le_metrics)
            gain = dict(le_metrics)
        output.append({
            "group_type": group_type, "group": group_name,
            "candidate_windows": int(n_candidate), "completed_windows": int(n_completed),
            "divergent_windows": int(n_candidate - n_completed),
            "completion_rate": float(n_completed / n_candidate) if n_candidate else None,
            "learned_rmse": le_metrics, "baseline_rmse": be_metrics,
            "gain_vs_baseline_pct": gain,
            "sufficient_samples": bool(n_completed >= int(min_group_samples)),
            "interpretation": "descriptive holdout subgroup; windows may be correlated within held-out episodes",
        })
    return output


def _recursive_evaluation_gate(horizon_results, groups_by_horizon, min_horizon_windows=32,
                               min_group_windows=32):
    """Never call a partial-world replay 'MPC ready'; distinguish results from authorization."""
    blockers = [
        "observation channels remain frozen in the partial world model",
        "future high-level actions are replayed from the recorded trajectory, not recomputed counterfactually",
        "prediction uncertainty for the action-conditioned transition head is not calibrated as a predictive interval",
        "the evaluator is not connected to the authoritative world simulator or live controller",
    ]
    horizon_adequacy = all(int(r.get("completed_windows", 0)) >= int(min_horizon_windows)
                           for r in horizon_results)
    group_failures = []
    for horizon, groups in zip(horizon_results, groups_by_horizon):
        for item in groups:
            if item["group_type"] in ("wind", "payload", "winch", "fsm") and item["candidate_windows"] >= min_group_windows:
                if item["completed_windows"] < min_group_windows:
                    group_failures.append({"horizon_s": horizon.get("horizon_s"),
                                           "group_type": item["group_type"], "group": item["group"],
                                           "completed_windows": item["completed_windows"],
                                           "candidate_windows": item["candidate_windows"]})
    return {
        "status": "BLOCKED_FOR_MPC",
        "recursive_eval_sufficient_horizon_coverage": bool(horizon_adequacy),
        "minimum_horizon_windows": int(min_horizon_windows),
        "minimum_group_windows": int(min_group_windows),
        "subgroup_coverage_failures": group_failures,
        "live_control_authorized": False,
        "reasons": blockers,
        "note": "A completed evaluation is not a controller-readiness decision; this partial-observation, recorded-action evaluator remains research-only.",
    }


def run_offline_recursive_transition_rollout_evaluation():
    """Readiness-gated, partial-observation recursive rollout; never used for live actuation."""
    report_path = os.path.abspath(os.environ.get(
        "SURVIVAL_ACTION_TRANSITION_REPORT", "action_transition_validation_v38_42.json"))
    data_path = os.path.abspath(os.environ.get(
        "SURVIVAL_TRANSITION_ROLLOUT_DATASET", "transition_rollout_sequences_v38_40.npz"))
    checkpoint_path = os.path.abspath(os.environ.get(
        "SURVIVAL_TRANSITION_ROLLOUT_CHECKPOINT", MODEL_WEIGHTS_FILE))
    output_path = os.path.abspath(os.environ.get(
        "SURVIVAL_TRANSITION_RECURSIVE_REPORT", "action_transition_recursive_rollout_eval_v38_40.json"))

    if not os.path.isfile(report_path):
        raise FileNotFoundError(f"Current-version readiness report is missing: {report_path}; run this version training pipeline first.")
    with open(report_path, "r", encoding="utf-8") as src:
        readiness = json.load(src)
    gate = readiness.get("rollout_readiness_gate", {}) if isinstance(readiness, dict) else {}
    if readiness.get("version") != SURVIVAL_CORE_VERSION:
        raise RuntimeError(f"Readiness version mismatch; expected {SURVIVAL_CORE_VERSION}, got {readiness.get('version')!r}")
    if (gate.get("status") != "READY_FOR_OFFLINE_MULTISTEP_EVALUATION"
            or gate.get("live_control_authorized", True) is not False):
        reasons = gate.get("reasons", []) if isinstance(gate, dict) else []
        raise RuntimeError("Recursive rollout BLOCKED by readiness gate: " + "; ".join(map(str, reasons[:12])))
    art = readiness.get("rollout_sequence_artifact", {})
    if not isinstance(art, dict) or art.get("status") != "SAVED":
        raise RuntimeError("Recursive rollout BLOCKED: no fresh successful held-out sequence artifact in readiness report.")
    if art.get("path") and os.path.abspath(str(art["path"])) != data_path:
        raise RuntimeError(f"Sequence path mismatch: readiness reports {art['path']}, requested {data_path}")
    if not os.path.isfile(data_path) or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("Held-out sequence artifact or trained checkpoint is missing; run the V38.35 training pipeline first.")

    with np.load(data_path, allow_pickle=False) as ds:
        observations = np.asarray(ds["observations"], dtype=np.float32)
        actions = np.asarray(ds["actions"], dtype=np.float32)
        targets = np.asarray(ds["targets"], dtype=np.float32)
        metadata = json.loads(str(np.asarray(ds["metadata_json"]).item()))
        worker_ids = np.asarray(ds["worker_ids"], dtype=np.int32)
        drone_ids = np.asarray(ds["drone_ids"], dtype=np.int32)
        starts_frame = np.asarray(ds["start_frames"], dtype=np.int32)

    if observations.ndim != 3:
        raise ValueError("Sequence observations must be [sequence,time,417]")
    nseq, steps, odim = observations.shape
    if odim != OBSERVATION_DIM or actions.shape != (nseq, steps, ACTION_TRANSITION_ACTION_DIM):
        raise ValueError("Sequence observation/action schema mismatch")
    if targets.shape != (nseq, steps, ACTION_TRANSITION_TARGET_DIM):
        raise ValueError("Sequence target schema mismatch")
    if any(arr.shape != (nseq,) for arr in (worker_ids, drone_ids, starts_frame)):
        raise ValueError("Sequence identity metadata mismatch")
    dt_s = float(metadata.get("dt_s", 0.016))
    if not math.isfinite(dt_s) or abs(dt_s - 0.016) > 1e-6:
        raise ValueError(f"This evaluator expects 0.016s BEMT transitions, got {dt_s}")

    device_text = os.environ.get("SURVIVAL_TRANSITION_ROLLOUT_DEVICE", "cpu").strip().lower()
    device = torch.device("cuda" if device_text == "cuda" and torch.cuda.is_available() else "cpu")
    if device_text == "cuda" and device.type != "cuda":
        print("[RECURSIVE ROLLOUT] CUDA requested but unavailable; using CPU", flush=True)
    model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
    _load_model_checkpoint(model, checkpoint_path, device)
    if not bool(getattr(model, "_action_conditioned_transition_head_checkpoint_trained", False)):
        raise RuntimeError("Recursive rollout BLOCKED: checkpoint lacks the trained action-conditioned transition head.")
    if not bool(getattr(model, "_environment_observation_transition_checkpoint_trained", False)):
        raise RuntimeError("Recursive rollout BLOCKED: checkpoint lacks the trained environment-observation transition head.")
    if not bool(getattr(model, "_cargo_dynamics_head_checkpoint_trained", False)):
        raise RuntimeError("Recursive rollout BLOCKED: checkpoint lacks the trained cargo dynamics head needed for cargo-state updates.")
    model.eval()

    max_windows_per_sequence = int(np.clip(int(os.environ.get("SURVIVAL_RECURSIVE_WINDOWS_PER_SEQUENCE", "8")), 1, 32))
    batch_size = int(np.clip(int(os.environ.get("SURVIVAL_RECURSIVE_BATCH_SIZE", "256")), 8, 1024))
    horizons = tuple(h for h in RECURSIVE_ROLLOUT_HORIZONS if h < steps)
    results = []
    total_selected = 0

    for horizon in horizons:
        cohorts_obs = []
        cohorts_actions = []
        cohorts_targets = []
        cohorts_true_endpoint = []
        cohorts_initial_state = []
        cohorts_context = []
        seen_keys = set()
        # Each window must be a contiguous valid segment with constant FSM phase.
        # This avoids rolling a fixed target-relative encoding across mission-phase changes.
        for si in range(nseq):
            last_start = steps - horizon - 1
            candidates = []
            for start in range(max(0, last_start + 1)):
                segment_t = targets[si, start:start+horizon, :]
                segment_a = actions[si, start:start+horizon, :]
                if len(segment_t) != horizon or len(segment_a) != horizon:
                    continue
                if not np.all(segment_t[:, -1] > 0.5) or not np.isfinite(segment_t[:, :-1]).all():
                    continue
                if not np.isfinite(observations[si, start:start+horizon+1]).all() or not np.isfinite(segment_a).all():
                    continue
                fsm = np.argmax(segment_a[:, 3:3+NUM_EXECUTIVE_STATES], axis=1)
                if not np.all(fsm == fsm[0]):
                    continue
                key = (int(worker_ids[si]), int(drone_ids[si]), int(starts_frame[si]) + int(start))
                if key in seen_keys:
                    continue
                candidates.append((int(start), key))
            if not candidates:
                continue
            if len(candidates) > max_windows_per_sequence:
                pick_indices = np.unique(np.linspace(0, len(candidates)-1, num=max_windows_per_sequence, dtype=np.int64))
                candidates = [candidates[int(i)] for i in pick_indices]
            for start, key in candidates:
                seen_keys.add(key)
                cohorts_obs.append(observations[si, start].copy())
                cohorts_actions.append(actions[si, start:start+horizon].copy())
                cohorts_targets.append(targets[si, start:start+horizon].copy())
                cohorts_true_endpoint.append(observations[si, start+horizon].copy())
                cohorts_initial_state.append(_recursive_rollout_state_from_observations(observations[si, start:start+1])[0])
                frame = observations[si, start].reshape(HISTORY_LEN, BASE_OBS_DIM)[-1]
                sensed_wind_mps = float(np.linalg.norm(frame[42:45]) * 8.0)
                cable_length_m = float(frame[53] * 15.0)
                payload_active = bool(frame[58] > 0.5)
                winch_cmd = float(segment_a[0, -1])
                winch_regime = "payout" if winch_cmd > 0.05 else ("retract" if winch_cmd < -0.05 else "neutral")
                fsm_idx = int(np.argmax(segment_a[0, 3:3+NUM_EXECUTIVE_STATES]))
                fsm_names = ("TRANSIT", "HOVER_LOCK", "WINCH_LOWER", "TOUCHDOWN_RELEASE",
                             "WINCH_RETRACT", "DEPOT_SERVICE", "EMERGENCY_DIVERT")
                cohorts_context.append({
                    "wind_mps": sensed_wind_mps,
                    "cable_length_m": cable_length_m,
                    "payload_active": payload_active,
                    "winch_regime": winch_regime,
                    "fsm_name": fsm_names[fsm_idx] if fsm_idx < len(fsm_names) else "unknown",
                })

        total_selected += len(cohorts_obs)
        if not cohorts_obs:
            empty_result = _recursive_rollout_metrics_from_errors([], [], horizon, 0, 0, 0)
            empty_result["error_distributions"] = _recursive_error_distribution_report([], [], horizon)
            empty_result["stratified_groups"] = []
            empty_result["divergent_window_fraction"] = None
            results.append(empty_result)
            continue

        roll_obs = np.stack(cohorts_obs).astype(np.float32)
        roll_actions = np.stack(cohorts_actions).astype(np.float32)
        initial_state = np.stack(cohorts_initial_state).astype(np.float64)
        true_endpoint_state = _recursive_rollout_state_from_observations(np.stack(cohorts_true_endpoint))
        alive = np.ones(len(roll_obs), dtype=bool)
        for step_idx in range(horizon):
            active_indices = np.flatnonzero(alive)
            if active_indices.size == 0:
                break
            pred_all = np.zeros((len(active_indices), ACTION_TRANSITION_OUTPUT_DIM), dtype=np.float32)
            pred_environment = np.zeros((len(active_indices), ENVIRONMENT_TRANSITION_DIM), dtype=np.float32)
            pred_cargo = np.zeros((len(active_indices), 5), dtype=np.float32)
            with torch.no_grad():
                for b0 in range(0, len(active_indices), batch_size):
                    batch_idx = active_indices[b0:b0+batch_size]
                    out = model(torch.from_numpy(roll_obs[batch_idx]).to(device),
                                return_physics=True, return_action_transition=True,
                                return_environment_transition=True,
                                transition_action=torch.from_numpy(roll_actions[batch_idx, step_idx]).to(device))
                    n_batch = len(batch_idx)
                    pred_cargo[b0:b0+n_batch] = out[7].detach().cpu().numpy().astype(np.float32, copy=False)
                    pred_all[b0:b0+n_batch] = out[-2].detach().cpu().numpy().astype(np.float32, copy=False)
                    pred_environment[b0:b0+n_batch] = out[-1].detach().cpu().numpy().astype(np.float32, copy=False)
            dp = pred_all[:, 0:3] * ACTION_TRANSITION_POSITION_SCALE_M
            dv = pred_all[:, 3:6] * ACTION_TRANSITION_VELOCITY_SCALE_MS
            dw = pred_all[:, 6:9] * ACTION_TRANSITION_ANGULAR_RATE_SCALE_RADS
            physically_bounded = (np.isfinite(pred_all).all(axis=1)
                                  & (np.linalg.norm(dp, axis=1) <= ACTION_TRANSITION_MAX_DELTA_POSITION_M)
                                  & (np.linalg.norm(dv, axis=1) <= ACTION_TRANSITION_MAX_DELTA_VELOCITY_MS)
                                  & (np.linalg.norm(dw, axis=1) <= ACTION_TRANSITION_MAX_DELTA_OMEGA_RADS))
            failed_indices = active_indices[~physically_bounded]
            alive[failed_indices] = False
            good_indices = active_indices[physically_bounded]
            good_pred = pred_all[physically_bounded]
            if good_indices.size:
                try:
                    local_good = physically_bounded
                    roll_obs[good_indices], _ = _recursive_update_observation_batch(
                        roll_obs[good_indices], good_pred, dt_s=dt_s,
                        predicted_environment_state=pred_environment[local_good],
                        predicted_cargo_state=pred_cargo[local_good])
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    alive[good_indices] = False

        completed = np.flatnonzero(alive)
        learned_errors = np.zeros((0, 9), dtype=np.float64)
        baseline_errors = np.zeros((0, 9), dtype=np.float64)
        if completed.size:
            predicted_endpoint_state = _recursive_rollout_state_from_observations(roll_obs[completed])
            actual = true_endpoint_state[completed]
            learned_errors = predicted_endpoint_state - actual
            init = initial_state[completed]
            baseline_endpoint = np.concatenate((init[:, 0:3] + init[:, 3:6] * (horizon * dt_s),
                                                init[:, 3:6], init[:, 6:9]), axis=1)
            baseline_errors = baseline_endpoint - actual
        result = _recursive_rollout_metrics_from_errors(
            learned_errors, baseline_errors, horizon, len(roll_obs), len(completed), int((~alive).sum()))
        result["horizon_s"] = float(horizon * dt_s)
        result["divergent_window_fraction"] = float((~alive).sum() / len(alive)) if len(alive) else None
        result["error_distributions"] = _recursive_error_distribution_report(
            learned_errors, baseline_errors, horizon)
        result["stratified_groups"] = _recursive_stratified_metrics(
            cohorts_context, completed, learned_errors, baseline_errors)
        results.append(result)

    if not results:
        raise RuntimeError("No recursive horizons could be evaluated")
    adequate_horizons = [r for r in results if r["completed_windows"] >= 32]
    if adequate_horizons and len(adequate_horizons) == len(results):
        status = "EVALUATION_COMPLETED_RESEARCH_ONLY"
    elif any(r["completed_windows"] > 0 for r in results):
        status = "EVALUATION_COMPLETED_WITH_INSUFFICIENT_DATA"
    else:
        status = "BLOCKED_NO_VALID_WINDOWS"
    readiness_decision = _recursive_evaluation_gate(
        results, [r.get("stratified_groups", []) for r in results])
    output = {
        "evaluator_version": RECURSIVE_ROLLOUT_VERSION,
        "training_pipeline_version": SURVIVAL_CORE_VERSION,
        "status": status,
        "readiness_decision": readiness_decision,
        "uncertainty_semantics": "bootstrap confidence intervals quantify sampling uncertainty of empirical RMSE only; they are not calibrated neural predictive intervals",
        "evaluation_mode": "partial-observation autoregressive rollout with learned kinematic, environment-state, and cargo updates; recorded future actions",
        "rollout_is_recursive": True,
        "action_sequence_is_recorded_not_counterfactual": True,
        "full_observation_world_reconstruction": False,
        "live_control_authorized": False,
        "live_control_connected": False,
        "dt_s": dt_s,
        "horizons": results,
        "sequence_count": int(nseq),
        "selected_windows_total_across_horizons": int(total_selected),
        "unique_workers": sorted(set(int(x) for x in worker_ids.tolist())),
        "checkpoint": checkpoint_path,
        "readiness_report": report_path,
        "sequence_dataset": data_path,
        "state_features_recursively_updated": [
            "target-relative-position (assumed fixed within constant FSM windows)",
            "world position", "world velocity", "body-to-world rotation matrix",
            "body angular rate", "derived acceleration feature", "derived gyro feature",
            "estimated wind xyz (learned next-state head)", "cable length and cable rate (learned next-state head)",
            "icing index (learned next-state head)", "cargo sway and tension (existing cargo head when payload remains attached)",
            "hook IMU and load-cell proxies regenerated from simplified sensor equations"],
        "state_features_frozen": [
            "discrete payload flag and FSM semantics", "lidar/conical ranges", "battery/ESC/thermal state",
            "peer telemetry", "world geometry and changing target context", "other environmental sensor channels"],
        "limitations": [
            "This is a genuinely recursive update for the explicitly listed kinematic features, but not a complete 139-feature sensor/world simulator.",
            "Only the directly supervised wind/cable/ice channels and existing cargo forecast channels are updated; other sensor/world features are frozen, so this is still not a full simulator.",
            "The next-step environment head predicts observation channels from recorded-action transitions; recursive outputs can drift from physically realizable wind or cable dynamics outside the training distribution.",
            "Recorded future actions are replayed open-loop and were chosen on the real trajectory, not recomputed from predicted states.",
            "Windows are restricted to a constant FSM command to reduce target/mission-phase discontinuities.",
            "The evaluator cannot authorize live motor, winch, release or landing control.",
            "The reported bootstrap intervals quantify sampling uncertainty of empirical endpoint RMSE, not calibrated per-sample predictive uncertainty.",
            "Subgroup metrics use start-of-window sensed wind, payload flag, winch command and FSM command; future regime changes within a window are not reclassified."],
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp = output_path + f".tmp-{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as dst:
            json.dump(_action_transition_json_safe(output), dst, indent=2, sort_keys=True, allow_nan=False)
        os.replace(tmp, output_path)
    finally:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass
    print(f"[RECURSIVE TRANSITION EVAL] status={status} horizons=" + ", ".join(
        f"{r['horizon_s']:.3f}s:{r['completed_windows']}/{r['candidate_windows']}"
        for r in results), flush=True)
    print(f"[RECURSIVE TRANSITION REPORT] {output_path}; partial observation state; live_control_authorized=False", flush=True)
    return output



# =============================================================================
# V38.40 — CLOSED-LOOP SHADOW-POLICY ROLLOUT (OFFLINE ONLY)
# =============================================================================
def _build_shadow_policy_action(guidance, fsm_logits, winch_prediction, fixed_fsm_indices):
    """Build the trained transition model's action vector from policy outputs.

    Guidance and winch commands are recomputed from the current recursively predicted
    observation. FSM is held to the starting phase for this local rollout so the fixed
    waypoint encoded in the observation remains semantically valid. Raw policy FSM
    predictions are returned as a diagnostic; they do not change mission phase here.
    """
    if torch is None or not all(torch.is_tensor(x) for x in (guidance, fsm_logits, winch_prediction)):
        raise TypeError("shadow action builder expects Torch tensors")
    if guidance.ndim != 2 or guidance.shape[1] < 3:
        raise ValueError("guidance must be [N,>=3]")
    n = guidance.shape[0]
    if fsm_logits.shape != (n, NUM_EXECUTIVE_STATES):
        raise ValueError("fsm_logits shape mismatch")
    if winch_prediction.ndim != 2 or winch_prediction.shape[0] != n or winch_prediction.shape[1] < 1:
        raise ValueError("winch_prediction must be [N,>=1]")
    phase = torch.as_tensor(fixed_fsm_indices, dtype=torch.long, device=guidance.device).reshape(-1)
    if phase.shape != (n,) or bool(((phase < 0) | (phase >= NUM_EXECUTIVE_STATES)).any().item()):
        raise ValueError("fixed_fsm_indices must contain one valid FSM index per row")
    guidance_cmd = torch.clamp(torch.nan_to_num(guidance[:, :3], nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    winch_cmd = torch.clamp(torch.nan_to_num(winch_prediction[:, :1], nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    phase_onehot = F.one_hot(phase, num_classes=NUM_EXECUTIVE_STATES).to(dtype=guidance_cmd.dtype)
    action = torch.cat((guidance_cmd, phase_onehot, winch_cmd), dim=-1)
    predicted_fsm = torch.argmax(torch.nan_to_num(fsm_logits, nan=0.0, posinf=1.0, neginf=-1.0), dim=-1)
    if action.shape != (n, ACTION_TRANSITION_ACTION_DIM):
        raise RuntimeError("shadow action builder produced wrong schema")
    return action, predicted_fsm


def _shadow_summary(values, unit):
    """Finite distribution summary for model-generated shadow metrics."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None, "unit": unit}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "unit": unit,
    }


def _shadow_policy_rollout_self_test():
    """Ensure guidance/winch are bounded and the local mission phase remains fixed."""
    guidance = torch.tensor([[2.0, -2.0, 0.25, 0.0], [-0.2, 0.4, 0.1, 0.0]], dtype=torch.float32)
    logits = torch.zeros((2, NUM_EXECUTIVE_STATES), dtype=torch.float32)
    logits[0, 4] = 4.0
    logits[1, 2] = 4.0
    winch = torch.tensor([[1.5, 0.0], [-0.4, 1.0]], dtype=torch.float32)
    fixed_fsm = torch.tensor([2, 5], dtype=torch.long)
    action, predicted_mode = _build_shadow_policy_action(guidance, logits, winch, fixed_fsm)
    assert action.shape == (2, ACTION_TRANSITION_ACTION_DIM)
    assert torch.allclose(action[:, :3], torch.tensor([[1.0, -1.0, 0.25], [-0.2, 0.4, 0.1]]))
    assert torch.equal(action[:, 3:3 + NUM_EXECUTIVE_STATES].argmax(dim=-1), fixed_fsm)
    assert torch.allclose(action[:, -1], torch.tensor([1.0, -0.4]))
    assert torch.equal(predicted_mode, torch.tensor([4, 2]))
    # A static FSM must not be confused with the policy's raw mode proposal.
    assert int((predicted_mode != fixed_fsm).sum().item()) == 2
    return True


def run_offline_shadow_policy_rollout_evaluation():
    """Recompute guidance/winch from predicted state and assess model-only proxies.

    This deliberately does not compare the counterfactual endpoint with the recorded
    future, because the recorded future followed different commands. Metrics are
    tracking/smoothness/divergence proxies inside the partial learned model only.
    """
    report_path = os.path.abspath(os.environ.get(
        "SURVIVAL_ACTION_TRANSITION_REPORT", "action_transition_validation_v38_42.json"))
    data_path = os.path.abspath(os.environ.get(
        "SURVIVAL_TRANSITION_ROLLOUT_DATASET", "transition_rollout_sequences_v38_40.npz"))
    checkpoint_path = os.path.abspath(os.environ.get(
        "SURVIVAL_TRANSITION_ROLLOUT_CHECKPOINT", MODEL_WEIGHTS_FILE))
    output_path = os.path.abspath(os.environ.get(
        "SURVIVAL_SHADOW_POLICY_REPORT", "action_transition_shadow_policy_rollout_v38_40.json"))

    for name, path in (("readiness report", report_path), ("sequence dataset", data_path), ("checkpoint", checkpoint_path)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Shadow-policy rollout blocked: missing {name}: {path}. Run this version's training pipeline first.")
    with open(report_path, "r", encoding="utf-8") as src:
        readiness = json.load(src)
    gate = readiness.get("rollout_readiness_gate", {}) if isinstance(readiness, dict) else {}
    if readiness.get("version") != SURVIVAL_CORE_VERSION:
        raise RuntimeError(f"Readiness report version mismatch: expected {SURVIVAL_CORE_VERSION}, got {readiness.get('version')!r}")
    if gate.get("status") != "READY_FOR_OFFLINE_MULTISTEP_EVALUATION" or gate.get("live_control_authorized", True) is not False:
        reasons = gate.get("reasons", []) if isinstance(gate, dict) else []
        raise RuntimeError("Shadow-policy rollout blocked by one-step gate: " + "; ".join(map(str, reasons[:12])))
    artifact = readiness.get("rollout_sequence_artifact", {})
    if not isinstance(artifact, dict) or artifact.get("status") != "SAVED":
        raise RuntimeError("Shadow-policy rollout blocked: readiness report lacks a saved held-out sequence artifact")
    if artifact.get("path") and os.path.abspath(str(artifact["path"])) != data_path:
        raise RuntimeError(f"Held-out sequence path mismatch: report={artifact['path']} requested={data_path}")

    with np.load(data_path, allow_pickle=False) as ds:
        observations = np.asarray(ds["observations"], dtype=np.float32)
        actions = np.asarray(ds["actions"], dtype=np.float32)
        targets = np.asarray(ds["targets"], dtype=np.float32)
        metadata = json.loads(str(np.asarray(ds["metadata_json"]).item()))
        worker_ids = np.asarray(ds["worker_ids"], dtype=np.int32)
        drone_ids = np.asarray(ds["drone_ids"], dtype=np.int32)
        start_frames = np.asarray(ds["start_frames"], dtype=np.int32)
    if observations.ndim != 3:
        raise ValueError("held-out observations must be [sequence,time,OBSERVATION_DIM]")
    nseq, steps, odim = observations.shape
    if odim != OBSERVATION_DIM or actions.shape != (nseq, steps, ACTION_TRANSITION_ACTION_DIM):
        raise ValueError("shadow rollout observation/action schema mismatch")
    if targets.shape != (nseq, steps, ACTION_TRANSITION_TARGET_DIM):
        raise ValueError("shadow rollout target schema mismatch")
    if any(x.shape != (nseq,) for x in (worker_ids, drone_ids, start_frames)):
        raise ValueError("shadow rollout sequence identity metadata mismatch")
    dt_s = float(metadata.get("dt_s", 0.016))
    if not math.isfinite(dt_s) or abs(dt_s - 0.016) > 1e-6:
        raise ValueError(f"shadow rollout requires 0.016 s transitions, got {dt_s}")

    max_horizon = int(np.clip(int(os.environ.get("SURVIVAL_SHADOW_POLICY_HORIZON_STEPS", "64")), 1, 64))
    horizons = tuple(h for h in RECURSIVE_ROLLOUT_HORIZONS if h <= max_horizon and h < steps)
    if not horizons:
        raise ValueError("held-out sequences are too short for requested shadow rollout horizons")
    max_horizon = max(horizons)
    windows_per_sequence = int(np.clip(int(os.environ.get("SURVIVAL_SHADOW_POLICY_WINDOWS_PER_SEQUENCE", "2")), 1, 16))
    max_windows = int(np.clip(int(os.environ.get("SURVIVAL_SHADOW_POLICY_MAX_WINDOWS", "128")), 8, 1024))

    # Sample held-out starts. FSM phase and encoded target must remain fixed in the
    # recorded reference window; the model's own predicted FSM is still logged.
    candidates = []
    fsm_start = 3
    fsm_end = fsm_start + NUM_EXECUTIVE_STATES
    latest_offset = (HISTORY_LEN - 1) * BASE_OBS_DIM
    for si in range(nseq):
        max_start = steps - max_horizon - 1
        if max_start < 0:
            continue
        starts = np.unique(np.linspace(0, max_start, num=min(windows_per_sequence, max_start + 1), dtype=np.int64))
        for start in starts.tolist():
            stop = start + max_horizon
            ob = observations[si, start:stop + 1]
            ac = actions[si, start:stop]
            tg = targets[si, start:stop]
            if len(ob) != max_horizon + 1 or len(ac) != max_horizon or len(tg) != max_horizon:
                continue
            if not np.isfinite(ob).all() or not np.isfinite(ac).all() or not np.isfinite(tg).all():
                continue
            if not np.all(tg[:, -1] > 0.5):
                continue
            phases = np.argmax(ac[:, fsm_start:fsm_end], axis=1)
            if not np.all(phases == phases[0]):
                continue
            # Ensure the recorded target remains the same within this local window.
            latest = ob.reshape(max_horizon + 1, HISTORY_LEN, BASE_OBS_DIM)[:, -1, :]
            pos_m = latest[:, 3:6] * 200.0
            target_m = pos_m + latest[:, 0:3] * 50.0
            target_drift = np.linalg.norm(target_m - target_m[0], axis=1)
            if float(np.max(target_drift)) > 0.5:
                continue
            fixed_fsm = int(phases[0])
            # Restrict to known executive classes; retain source ids for reproducibility.
            candidates.append({
                "sequence_index": int(si), "start": int(start), "worker_id": int(worker_ids[si]),
                "drone_id": int(drone_ids[si]), "start_frame": int(start_frames[si]) + int(start),
                "fixed_fsm": fixed_fsm, "observation": observations[si, start].copy(),
                "target_position_m": target_m[0].astype(np.float64),
                "target_drift_max_m": float(np.max(target_drift)),
            })
    if len(candidates) > max_windows:
        take = np.unique(np.linspace(0, len(candidates) - 1, num=max_windows, dtype=np.int64))
        candidates = [candidates[int(i)] for i in take]
    if len(candidates) < 8:
        raise RuntimeError(f"Shadow-policy rollout blocked: only {len(candidates)} valid held-out windows; at least 8 required")

    device_text = os.environ.get("SURVIVAL_SHADOW_POLICY_DEVICE", os.environ.get("SURVIVAL_TRANSITION_ROLLOUT_DEVICE", "cpu")).strip().lower()
    device = torch.device("cuda" if device_text == "cuda" and torch.cuda.is_available() else "cpu")
    if device_text == "cuda" and device.type != "cuda":
        print("[SHADOW POLICY] CUDA requested but unavailable; using CPU", flush=True)
    model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
    _load_model_checkpoint(model, checkpoint_path, device)
    required_heads = {
        "action_transition": bool(getattr(model, "_action_conditioned_transition_head_checkpoint_trained", False)),
        "environment_transition": bool(getattr(model, "_environment_observation_transition_checkpoint_trained", False)),
        "cargo_dynamics": bool(getattr(model, "_cargo_dynamics_head_checkpoint_trained", False)),
    }
    if not all(required_heads.values()):
        raise RuntimeError(f"Shadow-policy rollout blocked: required trained checkpoint heads missing/untrained: {required_heads}")
    model.eval()

    n = len(candidates)
    roll_obs = np.stack([x["observation"] for x in candidates]).astype(np.float32)
    fixed_fsm = np.asarray([x["fixed_fsm"] for x in candidates], dtype=np.int64)
    target_pos = np.stack([x["target_position_m"] for x in candidates]).astype(np.float64)
    alive = np.ones(n, dtype=bool)
    failure_reason = np.full(n, "", dtype=object)
    previous_guidance = np.zeros((n, 3), dtype=np.float64)
    has_previous_guidance = np.zeros(n, dtype=bool)
    previous_winch_sign = np.zeros(n, dtype=np.int8)
    guidance_norm_sum = np.zeros(n, dtype=np.float64)
    guidance_delta_sum = np.zeros(n, dtype=np.float64)
    guidance_delta_count = np.zeros(n, dtype=np.int32)
    fsm_disagree_sum = np.zeros(n, dtype=np.int32)
    winch_positive = np.zeros(n, dtype=np.int32)
    winch_negative = np.zeros(n, dtype=np.int32)
    winch_neutral = np.zeros(n, dtype=np.int32)
    winch_sign_flips = np.zeros(n, dtype=np.int32)
    steps_seen = np.zeros(n, dtype=np.int32)
    batch_size = int(np.clip(int(os.environ.get("SURVIVAL_SHADOW_POLICY_BATCH_SIZE", "64")), 8, 512))
    fsm_names = ("TRANSIT", "HOVER_LOCK", "WINCH_LOWER", "TOUCHDOWN_RELEASE", "WINCH_RETRACT", "DEPOT_SERVICE", "EMERGENCY_DIVERT")
    records = []
    divergence_counts = {"transition_nonfinite_or_bound": 0, "observation_update_failure": 0, "state_envelope": 0}
    horizons_set = set(horizons)

    for step_idx in range(1, max_horizon + 1):
        active = np.flatnonzero(alive)
        if active.size == 0:
            for h in horizons:
                if h == step_idx:
                    records.append({"horizon_steps": h, "horizon_s": h * dt_s, "candidate_windows": n,
                                    "completed_windows": 0, "divergent_windows": n, "status": "NO_COMPLETED_WINDOWS"})
            continue
        next_obs_for_active = roll_obs[active].copy()
        command_guidance = np.zeros((len(active), 3), dtype=np.float32)
        command_winch = np.zeros(len(active), dtype=np.float32)
        raw_policy_fsm = np.zeros(len(active), dtype=np.int64)
        action_vectors = None
        transition_pred = np.zeros((len(active), ACTION_TRANSITION_OUTPUT_DIM), dtype=np.float32)
        env_pred = np.zeros((len(active), ENVIRONMENT_TRANSITION_DIM), dtype=np.float32)
        cargo_pred = np.zeros((len(active), 5), dtype=np.float32)
        try:
            with torch.no_grad():
                for b0 in range(0, len(active), batch_size):
                    sl = slice(b0, min(len(active), b0 + batch_size))
                    rows = active[sl]
                    obs_t = torch.from_numpy(roll_obs[rows]).to(device)
                    policy = model(obs_t)
                    action_t, raw_fsm_t = _build_shadow_policy_action(
                        policy[0], policy[1], policy[2], fixed_fsm[rows])
                    out = model(obs_t, return_physics=True, return_action_transition=True,
                                return_environment_transition=True, transition_action=action_t)
                    command_guidance[sl] = action_t[:, :3].detach().cpu().numpy()
                    command_winch[sl] = action_t[:, -1].detach().cpu().numpy()
                    raw_policy_fsm[sl] = raw_fsm_t.detach().cpu().numpy()
                    transition_pred[sl] = out[-2].detach().cpu().numpy().astype(np.float32, copy=False)
                    env_pred[sl] = out[-1].detach().cpu().numpy().astype(np.float32, copy=False)
                    cargo_pred[sl] = out[7].detach().cpu().numpy().astype(np.float32, copy=False)
        except Exception as exc:
            raise RuntimeError(f"Shadow-policy model inference failed at step {step_idx}: {exc}") from exc

        guidance_norm = np.linalg.norm(command_guidance, axis=1)
        for local, global_idx in enumerate(active):
            guidance_norm_sum[global_idx] += float(guidance_norm[local])
            if has_previous_guidance[global_idx]:
                guidance_delta_sum[global_idx] += float(np.linalg.norm(command_guidance[local] - previous_guidance[global_idx]))
                guidance_delta_count[global_idx] += 1
            previous_guidance[global_idx] = command_guidance[local].astype(np.float64)
            has_previous_guidance[global_idx] = True
            fsm_disagree_sum[global_idx] += int(raw_policy_fsm[local] != fixed_fsm[global_idx])
            sign = int(np.sign(command_winch[local])) if abs(float(command_winch[local])) > 0.05 else 0
            if sign > 0:
                winch_positive[global_idx] += 1
            elif sign < 0:
                winch_negative[global_idx] += 1
            else:
                winch_neutral[global_idx] += 1
            if sign != 0 and previous_winch_sign[global_idx] != 0 and sign != previous_winch_sign[global_idx]:
                winch_sign_flips[global_idx] += 1
            if sign != 0:
                previous_winch_sign[global_idx] = sign
            steps_seen[global_idx] += 1

        dp = transition_pred[:, 0:3] * ACTION_TRANSITION_POSITION_SCALE_M
        dv = transition_pred[:, 3:6] * ACTION_TRANSITION_VELOCITY_SCALE_MS
        dw = transition_pred[:, 6:9] * ACTION_TRANSITION_ANGULAR_RATE_SCALE_RADS
        bounded = (np.isfinite(transition_pred).all(axis=1)
                   & (np.linalg.norm(dp, axis=1) <= ACTION_TRANSITION_MAX_DELTA_POSITION_M)
                   & (np.linalg.norm(dv, axis=1) <= ACTION_TRANSITION_MAX_DELTA_VELOCITY_MS)
                   & (np.linalg.norm(dw, axis=1) <= ACTION_TRANSITION_MAX_DELTA_OMEGA_RADS))
        if np.any(~bounded):
            failed = active[~bounded]
            alive[failed] = False
            failure_reason[failed] = "transition_nonfinite_or_bound"
            divergence_counts["transition_nonfinite_or_bound"] += int(len(failed))
        good_local = np.flatnonzero(bounded)
        good_global = active[good_local]
        if good_global.size:
            try:
                updated, _ = _recursive_update_observation_batch(
                    roll_obs[good_global], transition_pred[good_local], dt_s=dt_s,
                    predicted_environment_state=env_pred[good_local], predicted_cargo_state=cargo_pred[good_local])
                predicted_state = _recursive_rollout_state_from_observations(updated)
                speed = np.linalg.norm(predicted_state[:, 3:6], axis=1)
                z = predicted_state[:, 2]
                state_ok = (np.isfinite(predicted_state).all(axis=1) & (speed <= 35.0) & (z >= -5.0) & (z <= 85.0))
                roll_obs[good_global] = updated
                if np.any(~state_ok):
                    failed = good_global[~state_ok]
                    alive[failed] = False
                    failure_reason[failed] = "state_envelope"
                    divergence_counts["state_envelope"] += int(len(failed))
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                alive[good_global] = False
                failure_reason[good_global] = "observation_update_failure"
                divergence_counts["observation_update_failure"] += int(len(good_global))

        if step_idx in horizons_set:
            completed = np.flatnonzero(alive)
            state = _recursive_rollout_state_from_observations(roll_obs[completed]) if completed.size else np.zeros((0, 9))
            target_error = np.linalg.norm(state[:, :3] - target_pos[completed], axis=1) if completed.size else np.zeros(0)
            speed = np.linalg.norm(state[:, 3:6], axis=1) if completed.size else np.zeros(0)
            omega = np.linalg.norm(state[:, 6:9], axis=1) if completed.size else np.zeros(0)
            mean_guidance_per_window = guidance_norm_sum[completed] / np.maximum(steps_seen[completed], 1)
            mean_guidance_delta_per_window = guidance_delta_sum[completed] / np.maximum(guidance_delta_count[completed], 1)
            fsm_disagreement = fsm_disagree_sum[completed] / np.maximum(steps_seen[completed], 1)
            total_winch = np.maximum(steps_seen[completed], 1)
            rec = {
                "horizon_steps": int(step_idx), "horizon_s": float(step_idx * dt_s),
                "candidate_windows": int(n), "completed_windows": int(len(completed)),
                "divergent_windows": int(n - len(completed)),
                "divergence_fraction": float((n - len(completed)) / n),
                "target_tracking_error_m": _shadow_summary(target_error, "m"),
                "predicted_speed_mps": _shadow_summary(speed, "m/s"),
                "predicted_body_rate_magnitude_rads": _shadow_summary(omega, "rad/s"),
                "mean_normalized_guidance_magnitude": _shadow_summary(mean_guidance_per_window, "normalized acceleration command"),
                "mean_normalized_guidance_step_change": _shadow_summary(mean_guidance_delta_per_window, "normalized command change"),
                "raw_policy_fsm_disagreement_fraction": _shadow_summary(fsm_disagreement, "fraction of steps"),
                "winch_positive_payout_fraction": _shadow_summary(winch_positive[completed] / total_winch, "fraction of steps"),
                "winch_negative_retract_fraction": _shadow_summary(winch_negative[completed] / total_winch, "fraction of steps"),
                "winch_neutral_fraction": _shadow_summary(winch_neutral[completed] / total_winch, "fraction of steps"),
                "winch_sign_flips_per_window": _shadow_summary(winch_sign_flips[completed], "count"),
                "status": "COMPLETED_MODEL_PROXY_ONLY" if len(completed) else "NO_COMPLETED_WINDOWS",
            }
            records.append(rec)

    output = {
        "version": SURVIVAL_CORE_VERSION,
        "evaluation_mode": "offline authoritative live same-action transition validation and closed-loop shadow-policy rollout using learned partial observation model",
        "action_generation": "guidance and winch commands recomputed from each recursively predicted observation; FSM held fixed per segment to preserve target semantics; raw policy FSM proposals logged as disagreement diagnostics",
        "readiness_report": report_path,
        "sequence_dataset": data_path,
        "checkpoint": checkpoint_path,
        "sequence_count": int(nseq),
        "candidate_windows": int(n),
        "unique_workers": sorted(set(int(c["worker_id"]) for c in candidates)),
        "fixed_fsm_modes": {name: int(sum(1 for c in candidates if fsm_names[c["fixed_fsm"]] == name)) for name in fsm_names},
        "horizons": records,
        "divergence_reason_counts": divergence_counts,
        "counterfactual_ground_truth_available": False,
        "state_reconstruction_complete": False,
        "live_control_authorized": False,
        "live_control_connected": False,
        "mpc_readiness": "BLOCKED_FOR_MPC",
        "limitations": [
            "The evaluator updates selected kinematic, wind, cable, icing and cargo channels but freezes other sensor/world features.",
            "The FSM phase is fixed per window because target and mission phase transitions cannot yet be reconstructed faithfully; raw policy FSM proposals are logged separately.",
            "The future trajectory is generated by the learned model under recomputed guidance/winch commands. No authoritative simulator is in the loop, so these target-tracking values are model-internal proxies, not counterfactual accuracy measurements.",
            "No learned action from this evaluator is sent to live motors, winch, release or depot landing control.",
            "The run does not establish that the model improves delivery completion or physical stability."],
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp = output_path + f".tmp-{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as dst:
            json.dump(_action_transition_json_safe(output), dst, indent=2, sort_keys=True, allow_nan=False)
        os.replace(tmp, output_path)
    finally:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass
    print(f"[SHADOW POLICY ROLLOUT] candidates={n} completed={int(alive.sum())}/{n} horizons="
          + ", ".join(f"{r['horizon_s']:.3f}s:{r.get('completed_windows', 0)}/{n}" for r in records), flush=True)
    print(f"[SHADOW POLICY REPORT] {output_path}; model-proxy-only; live_control_authorized=False; MPC blocked", flush=True)
    return output


def _recursive_rollout_self_test():
    """Small deterministic test of physical scaling, temporal stacking and SO(3) update."""
    obs = np.zeros((2, OBSERVATION_DIM), dtype=np.float32)
    for i in range(2):
        fr = obs[i].reshape(HISTORY_LEN, BASE_OBS_DIM)
        for t in range(HISTORY_LEN):
            fr[t, 3:6] = np.array([1.0, -0.5, 0.16], dtype=np.float32)  # [200,-100,32] m
            fr[t, 6:9] = np.array([0.25, 0.0, 0.0], dtype=np.float32)   # [2,0,0] m/s
            fr[t, 9:18] = np.eye(3, dtype=np.float32).reshape(-1)
            fr[t, 18:21] = np.array([0.0, 0.0, 0.1], dtype=np.float32)
            fr[t, 42:45] = np.array([0.25, 0.0, 0.0], dtype=np.float32)
        fr[-1, 0:3] = np.array([0.5, 0.0, 0.0], dtype=np.float32)  # target at [300,-100,32]
        fr[-1, 70] = 0.123  # frozen arbitrary non-kinematic feature
    pred = np.zeros((2, ACTION_TRANSITION_OUTPUT_DIM), dtype=np.float32)
    pred[:, 0] = 0.10 / ACTION_TRANSITION_POSITION_SCALE_M
    pred[:, 3] = 0.5 / ACTION_TRANSITION_VELOCITY_SCALE_MS
    pred[:, 8] = 0.1 / ACTION_TRANSITION_ANGULAR_RATE_SCALE_RADS
    nxt, st = _recursive_update_observation_batch(obs, pred, 0.016)
    cur = nxt.reshape(2, HISTORY_LEN, BASE_OBS_DIM)[:, -1, :]
    assert np.allclose(cur[:, 3:6] * 200.0, np.array([[200.1, -100.0, 32.0]] * 2), atol=2e-5)
    assert np.allclose(cur[:, 6:9] * 8.0, np.array([[2.5, 0.0, 0.0]] * 2), atol=2e-5)
    assert np.allclose(cur[:, 0:3] * 50.0, np.array([[24.9, 0.0, 0.0]] * 2), atol=2e-5)
    assert np.allclose(cur[:, 70], 0.123)
    assert np.allclose(cur[:, 42:45], np.array([[0.25, 0.0, 0.0]] * 2))
    for r in cur[:, 9:18].reshape(-1, 3, 3):
        assert np.allclose(r.T @ r, np.eye(3), atol=1e-5)
        assert abs(np.linalg.det(r) - 1.0) < 1e-5
    assert np.allclose(st[:, 0:3], np.array([[200.1, -100.0, 32.0]] * 2), atol=2e-5)
    # Metric helper must correctly detect a zero-error model beating a nonzero baseline.
    metric = _recursive_rollout_metrics_from_errors(
        np.zeros((32, 9), dtype=np.float64), np.ones((32, 9), dtype=np.float64),
        horizon=4, candidate_count=32, completed_count=32, divergent_count=0)
    assert metric["sufficient_samples"] and metric["completed_windows"] == 32
    assert metric["gain_vs_baseline_pct"]["position_rmse_m"] > 99.0
    # Empirical error-distribution diagnostics must report finite values and a CI when n>=2.
    errors = np.zeros((40, 9), dtype=np.float64)
    errors[:, 0] = np.linspace(0.0, 0.39, 40)
    baseline_errors = errors.copy()
    baseline_errors[:, 0] += 1.0
    dist = _recursive_error_distribution_report(errors, baseline_errors, horizon_steps=4)
    assert dist["position_m"]["learned"]["count"] == 40
    assert dist["position_m"]["learned"]["bootstrap_95pct_ci_rmse"] is not None
    assert dist["position_m"]["learned"]["p95_norm"] > dist["position_m"]["learned"]["median_norm"]
    contexts = [{"wind_mps": 6.0, "payload_active": True, "winch_regime": "payout", "fsm_name": "WINCH_LOWER"} for _ in range(40)]
    stratified = _recursive_stratified_metrics(contexts, np.arange(40), errors, baseline_errors)
    by_group = {(x["group_type"], x["group"]): x for x in stratified}
    assert by_group[("wind", "strong_ge5p5mps")]["completed_windows"] == 40
    assert by_group[("winch", "payout")]["sufficient_samples"]
    gate = _recursive_evaluation_gate([{"completed_windows": 40, "horizon_s": 0.1}], [stratified])
    assert gate["status"] == "BLOCKED_FOR_MPC" and gate["live_control_authorized"] is False
    return True

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
            if requested_workers < 1:
                raise ValueError
            num_workers = min(worker_cap, requested_workers)
            if requested_workers > worker_cap:
                print(
                    f"[HARVEST RESOURCE POLICY] Requested {requested_workers} workers, "
                    f"capped at {worker_cap} (CPU limit).", flush=True)
        except ValueError:
            num_workers = min(4, worker_cap)
            print(
                f"[HARVEST RESOURCE POLICY] Invalid SURVIVAL_HARVEST_WORKERS="
                f"{requested_workers_text!r}; using {num_workers} workers.", flush=True)
    else:
        num_workers = worker_cap
    samples_per = NUM_SAMPLES // num_workers

    print("\n" + "="*75)
    print(f"HARVESTING {NUM_SAMPLES:,} SAMPLES (FILE-BACKED IPC PIPELINE)")
    print(f"Parallel Workers: {num_workers} processes | Compute Device: {device}")
    print("Windows spawn fix: worker children skip the PyTorch/CUDA import; only the parent loads the training stack.")
    print("="*75)
    t0 = time.time()

    worker_payloads = [(i, samples_per) for i in range(num_workers)]
    # A spawned Python worker re-executes this file as __mp_main__. Advertise a
    # lightweight import mode before creating the pool so it does not load
    # PyTorch's large CUDA DLL set in every child. Restore all environment
    # variables after pool creation/use to avoid changing the training parent.
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
        raise RuntimeError(
            "Dataset harvesting failed while starting or running CPU-only worker processes. "
            "The worker mode no longer imports PyTorch/CUDA. If Windows still reports "
            "WinError 1455, try `$env:SURVIVAL_HARVEST_WORKERS='2'` and ensure the Windows "
            "paging file is system-managed. Original error: " + repr(exc)
        ) from exc
    finally:
        for key, value in previous_child_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    X_chunks, yg_chunks, yf_chunks, yw_chunks, yh_chunks, yt_chunks, yv_chunks, ykin_chunks, yenv_chunks, ycargo_chunks, yrot_chunks, ytrajectory_chunks, yaction_transition_chunks = [], [], [], [], [], [], [], [], [], [], [], [], []
    source_worker_chunks = []
    for source_worker_id, cf in enumerate(chunk_files):
        with np.load(cf) as data:
            X_chunks.append(data["obs"])
            yg_chunks.append(data["yg"])
            yf_chunks.append(data["yf"])
            yw_chunks.append(data["yw"])
            yh_chunks.append(data["yh"])
            yt_chunks.append(data["yt"])
            yv_chunks.append(data["yv"])
            ykin_chunks.append(data["ykin"])
            yenv_chunks.append(data["yenvironment"])
            ycargo_chunks.append(data["ycargo"])
            yrot_chunks.append(data["yrot"])
            ytrajectory_chunks.append(data["ytrajectory"])
            yaction_transition_chunks.append(data["yaction_transition"])
            source_worker_chunks.append(np.full(len(data["obs"]), source_worker_id, dtype=np.int16))
        # V38.35 retains these disk-backed worker chunks until the worker-held-out
        # ordered sequence artifact is written below. They are removed immediately
        # after extraction to avoid leaking large temporary files.

    # Combine the longer rollouts before sampling; slicing the first 250k rows used
    # to retain early transit samples and discard later mission phases from each worker.
    X_raw = np.concatenate(X_chunks, axis=0)
    yg_raw = np.concatenate(yg_chunks, axis=0)
    yf_raw = np.concatenate(yf_chunks, axis=0)
    yw_raw = np.concatenate(yw_chunks, axis=0)
    yh_raw = np.concatenate(yh_chunks, axis=0)
    yt_raw = np.concatenate(yt_chunks, axis=0)
    yv_raw = np.concatenate(yv_chunks, axis=0)
    ykin_raw = np.concatenate(ykin_chunks, axis=0)
    yenv_raw = np.concatenate(yenv_chunks, axis=0)
    ycargo_raw = np.concatenate(ycargo_chunks, axis=0)
    yrot_raw = np.concatenate(yrot_chunks, axis=0)
    ytrajectory_raw = np.concatenate(ytrajectory_chunks, axis=0)
    yaction_transition_raw = np.concatenate(yaction_transition_chunks, axis=0)
    source_worker_raw = np.concatenate(source_worker_chunks, axis=0)
    del X_chunks, yg_chunks, yf_chunks, yw_chunks, yh_chunks, yt_chunks, yv_chunks
    del ykin_chunks, yenv_chunks, ycargo_chunks, yrot_chunks, ytrajectory_chunks, yaction_transition_chunks, source_worker_chunks
    if not (len(X_raw) == len(yg_raw) == len(yf_raw) == len(yw_raw) == len(yh_raw) == len(yt_raw) == len(yv_raw) == len(ykin_raw) == len(yenv_raw) == len(ycargo_raw) == len(yrot_raw) == len(ytrajectory_raw) == len(yaction_transition_raw) == len(source_worker_raw)):
        raise RuntimeError("Harvest data alignment failure: physics/trajectory/source-group labels do not match observations.")
    valid_kin = int((ykin_raw[:, 6] > 0.5).sum())
    valid_cargo = int((ycargo_raw[:, 5] > 0.5).sum())
    valid_rot = int((yrot_raw[:, 3] > 0.5).sum())
    valid_env = int((yenv_raw[:, -1] > 0.5).sum())
    print(f"[ENVIRONMENT TRANSITION DATA] same-drone wind/cable/ice targets valid={valid_env:,}/{len(yenv_raw):,} ({100.0*valid_env/max(1,len(yenv_raw)):.2f}%); reset and implausible transitions masked.")
    print(f"[PHYSICS-GUIDED DATA] one-step kinematics targets valid={valid_kin:,}/{len(ykin_raw):,} ({100.0*valid_kin/max(1,len(ykin_raw)):.2f}%); invalid transitions masked from this loss.")
    print(f"[CARGO-DYNAMICS DATA] measured sway/rate/tension targets valid={valid_cargo:,}/{len(ycargo_raw):,} ({100.0*valid_cargo/max(1,len(ycargo_raw)):.2f}%); only loaded extended-cable transitions contribute.")
    print(f"[ROTATIONAL EULER DATA] valid BEMT torque-residual targets={valid_rot:,}/{len(yrot_raw):,} ({100.0*valid_rot/max(1,len(yrot_raw)):.2f}%); grounded, parachute and invalid/high-dynamics rows are masked.")
    valid_action_transition = int((yaction_transition_raw[:, -1] > 0.5).sum())
    print(f"[ACTION-CONDITIONED TRANSITIONS] valid paired BEMT state/action transitions={valid_action_transition:,}/{len(yaction_transition_raw):,} ({100.0*valid_action_transition/max(1,len(yaction_transition_raw)):.2f}%).")
    future_counts = [int((ytrajectory_raw[:, TRAJECTORY_FORECAST_DIM+h] > 0.5).sum()) for h in range(TRAJECTORY_FORECAST_HORIZON_COUNT)]
    future_summary = ", ".join(f"{horizon:.3f}s={count:,}" for horizon, count in zip(TRAJECTORY_FORECAST_HORIZONS_S, future_counts))
    print(f"[TRAJECTORY FORECAST DATA] valid future-state pairs per horizon: {future_summary}; reset and implausible transitions masked.")
    print(f"Dataset compiled in {time.time()-t0:.2f}s! ({len(X_raw):,} candidate samples)")
    raw_fsm_counts = np.bincount(np.asarray(yf_raw, dtype=np.int64), minlength=NUM_EXECUTIVE_STATES)
    raw_fsm_total = max(1, int(raw_fsm_counts.sum()))
    raw_fsm_summary = ", ".join(
        f"{idx}:{ACTION_NAMES[idx]}={int(count)} ({100.0*count/raw_fsm_total:.2f}%)"
        for idx, count in enumerate(raw_fsm_counts))
    print(f"[RAW DATASET FSM LABELS] {raw_fsm_summary}")
    selected_indices = _select_harvest_indices_with_action_coverage(
        yf_raw, NUM_SAMPLES, required_classes=range(6), seed=None)
    X_all = X_raw[selected_indices]
    yg_all = yg_raw[selected_indices]
    yf_all = yf_raw[selected_indices]
    yw_all = yw_raw[selected_indices]
    yh_all = yh_raw[selected_indices]
    yt_all = yt_raw[selected_indices]
    yv_all = yv_raw[selected_indices]
    ykin_all = ykin_raw[selected_indices]
    yenv_all = yenv_raw[selected_indices]
    ycargo_all = ycargo_raw[selected_indices]
    yrot_all = yrot_raw[selected_indices]
    ytrajectory_all = ytrajectory_raw[selected_indices]
    yaction_transition_all = yaction_transition_raw[selected_indices]
    source_worker_all = source_worker_raw[selected_indices]
    # Free oversized candidate arrays before moving the selected dataset to CUDA.
    del X_raw, yg_raw, yf_raw, yw_raw, yh_raw, yt_raw, yv_raw, ykin_raw, yenv_raw, ycargo_raw, yrot_raw, ytrajectory_raw, yaction_transition_raw, source_worker_raw
    sample_perm = np.random.default_rng().permutation(len(X_all))
    X_all, yg_all, yf_all = X_all[sample_perm], yg_all[sample_perm], yf_all[sample_perm]
    yw_all, yh_all, yt_all, yv_all = (yw_all[sample_perm], yh_all[sample_perm],
                                      yt_all[sample_perm], yv_all[sample_perm])
    ykin_all = ykin_all[sample_perm]
    yenv_all = yenv_all[sample_perm]
    ycargo_all = ycargo_all[sample_perm]
    yrot_all = yrot_all[sample_perm]
    ytrajectory_all = ytrajectory_all[sample_perm]
    yaction_transition_all = yaction_transition_all[sample_perm]
    source_worker_all = source_worker_all[sample_perm]
    selected_action_transition_valid = int((yaction_transition_all[:, -1] > 0.5).sum())
    print(f"[ACTION-CONDITIONED TRANSITIONS SELECTED] valid={selected_action_transition_valid:,}/{len(yaction_transition_all):,}.", flush=True)
    if selected_action_transition_valid < 256:
        raise RuntimeError("Too few valid paired action-conditioned transitions were selected; refusing to mark the transition head trained.")
    selected_env_valid = int((yenv_all[:, -1] > 0.5).sum())
    print(f"[ENVIRONMENT TRANSITION SELECTED] valid next wind/cable/ice targets={selected_env_valid:,}/{len(yenv_all):,}.", flush=True)
    if selected_env_valid < 256:
        raise RuntimeError(f"Only {selected_env_valid} valid dynamic environment transitions were selected; refusing to mark the environment transition head trained.")
    selected_cargo_valid = int((ycargo_all[:, 5] > 0.5).sum())
    print(f"[CARGO-DYNAMICS SELECTED] valid loaded-cable targets={selected_cargo_valid:,}/{len(ycargo_all):,}.", flush=True)
    if selected_cargo_valid < 128:
        raise RuntimeError(
            f"Only {selected_cargo_valid} valid loaded-cargo transitions were selected; "
            "refusing to mark the suspended-cargo prediction head as trained. Check lowering/retract coverage and cable-state alignment.")
    selected_rot_valid = int((yrot_all[:, 3] > 0.5).sum())
    if selected_rot_valid < 128:
        raise RuntimeError(
            f"Only {selected_rot_valid} valid rotational-dynamics transitions were selected; "
            "refusing to mark the Euler-residual head as trained. Check airborne sampling and BEMT torque targets.")
    selected_trajectory_counts = [int((ytrajectory_all[:, TRAJECTORY_FORECAST_DIM+h] > 0.5).sum())
                                  for h in range(TRAJECTORY_FORECAST_HORIZON_COUNT)]
    print("[TRAJECTORY FORECAST SELECTED] " + ", ".join(
        f"{h:.3f}s={n:,}" for h, n in zip(TRAJECTORY_FORECAST_HORIZONS_S, selected_trajectory_counts)), flush=True)
    if any(n < 128 for n in selected_trajectory_counts):
        raise RuntimeError(
            f"Insufficient valid future-state labels per trajectory horizon: {selected_trajectory_counts}; "
            "refusing to mark the forecast head trained. Check worker frame pairing, rollout duration and reset masking.")
    print(f"[DATASET FSM LABELS SELECTED] {len(X_all):,} shuffled samples with verified coverage for actions 0–5; action 6 remains untrained and is handled by hard safety gates.")
    fsm_counts = np.bincount(np.asarray(yf_all, dtype=np.int64), minlength=NUM_EXECUTIVE_STATES)
    fsm_total = max(1, int(fsm_counts.sum()))
    fsm_summary = ", ".join(
        f"{idx}:{ACTION_NAMES[idx]}={int(count)} ({100.0*count/fsm_total:.2f}%)"
        for idx, count in enumerate(fsm_counts))
    print(f"[DATASET FSM LABELS] {fsm_summary}", flush=True)

    # Keep whole worker rollouts out of validation to reduce temporal leakage: future
    # labels from neighbouring frames of the same simulated episode must not straddle
    # train and validation when a multi-horizon forecast is being evaluated.
    unique_workers = np.unique(source_worker_all)
    rollout_heldout_workers = None
    if len(unique_workers) >= 5:
        val_worker_count = max(1, int(math.ceil(0.20 * len(unique_workers))))
        split_rng = np.random.default_rng(202910)
        split = None
        for _ in range(2048):
            val_workers = split_rng.choice(unique_workers, size=val_worker_count, replace=False)
            val_mask_np = np.isin(source_worker_all, val_workers)
            train_mask_np = ~val_mask_np
            train_classes = set(np.unique(yf_all[train_mask_np]).tolist())
            val_forecast_counts = [int((ytrajectory_all[val_mask_np, TRAJECTORY_FORECAST_DIM+h] > 0.5).sum())
                                   for h in range(TRAJECTORY_FORECAST_HORIZON_COUNT)]
            val_action_transition_count = int((yaction_transition_all[val_mask_np, -1] > 0.5).sum())
            if (all(c in train_classes for c in range(6)) and all(c >= 128 for c in val_forecast_counts)
                    and val_action_transition_count >= 128):
                split = (train_mask_np, val_mask_np, np.sort(val_workers), val_forecast_counts, val_action_transition_count)
                # Prefer a held-out worker group that covers all six supervised actions,
                # but do not sacrifice a clean episode split if a rare action is absent.
                val_classes = set(np.unique(yf_all[val_mask_np]).tolist())
                if all(c in val_classes for c in range(6)):
                    break
        if split is None:
            for _chunk_file in chunk_files:
                try:
                    os.remove(_chunk_file)
                except OSError:
                    pass
            raise RuntimeError("Could not create worker-held-out train/validation split with complete training action coverage and valid multi-horizon labels; refusing to report a leakage-prone forecast metric.")
        train_mask_np, val_mask_np, val_workers, val_forecast_counts, val_action_transition_count = split
        rollout_heldout_workers = np.asarray(val_workers, dtype=np.int64).copy()
        train_indices = np.flatnonzero(train_mask_np)
        val_indices = np.flatnonzero(val_mask_np)
        print(f"[VALIDATION SPLIT] held-out harvest workers={val_workers.tolist()} train={len(train_indices):,} validation={len(val_indices):,}; future-target validation counts={val_forecast_counts}; valid action transitions={val_action_transition_count}", flush=True)
    else:
        # Low-worker fallback preserves small-memory runs but warns that the validation
        # set is row-random rather than fully episode-independent.
        split_perm = np.random.default_rng(202910).permutation(len(X_all))
        num_train_fallback = int(len(X_all)*0.90)
        train_indices, val_indices = split_perm[:num_train_fallback], split_perm[num_train_fallback:]
        print(f"[VALIDATION SPLIT WARNING] only {len(unique_workers)} harvest worker(s); using row-random 90/10 split, so trajectory metrics may be optimistic. Increase SURVIVAL_HARVEST_WORKERS to at least 5 for worker-held-out validation.", flush=True)

    rollout_sequence_artifact_path = os.environ.get(
        "SURVIVAL_TRANSITION_ROLLOUT_DATASET", "transition_rollout_sequences_v38_40.npz")
    try:
        rollout_sequence_artifact_info = _save_transition_rollout_sequence_artifact(
            chunk_files=chunk_files,
            heldout_worker_ids=rollout_heldout_workers,
            output_path=rollout_sequence_artifact_path,
            sequence_length=int(os.environ.get("SURVIVAL_TRANSITION_ROLLOUT_SEQUENCE_LENGTH", "128")),
            segments_per_drone=int(os.environ.get("SURVIVAL_TRANSITION_ROLLOUT_SEGMENTS_PER_DRONE", "8")),
        )
    except Exception as sequence_exc:
        rollout_sequence_artifact_info = {
            "status": "BLOCKED", "path": os.path.abspath(rollout_sequence_artifact_path),
            "reason": f"sequence artifact creation failed: {type(sequence_exc).__name__}: {sequence_exc}"}
        print(f"[TRANSITION ROLLOUT SEQUENCES WARNING] {rollout_sequence_artifact_info['reason']}", flush=True)
    finally:
        for _chunk_file in chunk_files:
            try:
                os.remove(_chunk_file)
            except OSError:
                pass
    if rollout_sequence_artifact_info.get("status") != "SAVED":
        try:
            if os.path.isfile(rollout_sequence_artifact_path):
                os.remove(rollout_sequence_artifact_path)
        except OSError:
            pass
    if rollout_sequence_artifact_info.get("status") == "SAVED":
        print(f"[TRANSITION ROLLOUT SEQUENCES] saved={rollout_sequence_artifact_info['path']} "
              f"sequences={rollout_sequence_artifact_info['sequence_count']} "
              f"valid_rows={rollout_sequence_artifact_info['valid_rows']}/"
              f"{rollout_sequence_artifact_info['total_rows']}", flush=True)
    else:
        print(f"[TRANSITION ROLLOUT SEQUENCES BLOCKED] {rollout_sequence_artifact_info.get('reason', 'no independent worker-held-out sequence data')}", flush=True)

    num_train = int(len(train_indices))
    X_train = torch.from_numpy(X_all[train_indices]).to(device)
    yg_tr = torch.from_numpy(yg_all[train_indices]).to(device)
    yf_tr = torch.as_tensor(yf_all[train_indices], dtype=torch.long, device=device)
    yw_tr = torch.from_numpy(yw_all[train_indices]).to(device)
    yh_tr = torch.from_numpy(yh_all[train_indices]).to(device)
    yt_tr = torch.from_numpy(yt_all[train_indices]).to(device)
    yv_tr = torch.from_numpy(yv_all[train_indices]).to(device)
    ykin_tr = torch.from_numpy(ykin_all[train_indices]).to(device)
    yenv_tr = torch.from_numpy(yenv_all[train_indices]).to(device)
    ycargo_tr = torch.from_numpy(ycargo_all[train_indices]).to(device)
    yrot_tr = torch.from_numpy(yrot_all[train_indices]).to(device)
    ytrajectory_tr = torch.from_numpy(ytrajectory_all[train_indices]).to(device)
    yaction_transition_tr = torch.from_numpy(yaction_transition_all[train_indices]).to(device)
    action_condition_tr = _build_action_condition_tensor(yg_tr, yf_tr, yw_tr)

    X_val = torch.from_numpy(X_all[val_indices]).to(device)
    yg_va = torch.from_numpy(yg_all[val_indices]).to(device)
    yf_va = torch.as_tensor(yf_all[val_indices], dtype=torch.long, device=device)
    yw_va = torch.from_numpy(yw_all[val_indices]).to(device)
    action_condition_va = _build_action_condition_tensor(yg_va, yf_va, yw_va)
    yaction_transition_va = torch.from_numpy(yaction_transition_all[val_indices]).to(device)
    ykin_va = torch.from_numpy(ykin_all[val_indices]).to(device)
    yenv_va = torch.from_numpy(yenv_all[val_indices]).to(device)
    ycargo_va = torch.from_numpy(ycargo_all[val_indices]).to(device)
    yrot_va = torch.from_numpy(yrot_all[val_indices]).to(device)
    ytrajectory_va = torch.from_numpy(ytrajectory_all[val_indices]).to(device)

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
            p_g, _, _, _, _, p_v, p_kin, p_cargo, p_rot, p_trajectory, p_action_transition, p_logvar, p_env_transition = model(
                bx, return_physics=True, return_rotational_residual=True, return_uncertainty=True,
                return_trajectory_forecast=True, return_action_transition=True,
                return_environment_transition=True, transition_action=action_condition_tr[idx])
            loss_pre = (crit_mse(p_g, byg) + 0.15 * crit_mse(p_v, byv)
                        + 0.20 * _kinematics_prediction_loss(p_kin, ykin_tr[idx])
                        + 0.10 * _cargo_dynamics_prediction_loss(p_cargo, ycargo_tr[idx])
                        + KINEMATIC_EQUATION_RESIDUAL_WEIGHT * _kinematic_equation_residual_loss(
                            p_kin, bx, ykin_tr[idx], dt=0.016)
                        + ROTATIONAL_DYNAMICS_RESIDUAL_WEIGHT * _rotational_dynamics_residual_prediction_loss(
                            p_rot, yrot_tr[idx])
                        + PHYSICS_UNCERTAINTY_NLL_WEIGHT * _physics_uncertainty_phase12_loss(
                            p_logvar, p_kin, p_cargo, p_rot, ykin_tr[idx], ycargo_tr[idx], yrot_tr[idx])
                        + TRAJECTORY_FORECAST_WEIGHT * _trajectory_forecast_loss(p_trajectory, ytrajectory_tr[idx])
                        + ACTION_TRANSITION_WEIGHT * _action_conditioned_transition_loss(
                            p_action_transition, yaction_transition_tr[idx])
                        + ENVIRONMENT_TRANSITION_WEIGHT * _environment_transition_prediction_loss(
                            p_env_transition, yenv_tr[idx]))
            loss_pre.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            pre_optimizer.step()
            ep_loss += loss_pre.item()

        print(f"Pre-train Epoch [{ep+1:02d}/{NUM_PRETRAIN_EPOCHS:02d}] - Composite physics-guided training loss: {ep_loss/num_batches:.5f}")

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
            p_g, p_f, p_w, p_h, p_t, p_v, p_kin, p_cargo, p_rot, p_trajectory, p_action_transition, p_logvar, p_env_transition = model(
                bx, return_physics=True, return_rotational_residual=True, return_uncertainty=True,
                return_trajectory_forecast=True, return_action_transition=True,
                return_environment_transition=True, transition_action=action_condition_tr[idx])

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
                    0.10 * crit_mse(p_v, byv) +
                    0.20 * _kinematics_prediction_loss(p_kin, ykin_tr[idx]) +
                    0.10 * _cargo_dynamics_prediction_loss(p_cargo, ycargo_tr[idx]) +
                    KINEMATIC_EQUATION_RESIDUAL_WEIGHT * _kinematic_equation_residual_loss(
                        p_kin, bx, ykin_tr[idx], dt=0.016)
                    + ROTATIONAL_DYNAMICS_RESIDUAL_WEIGHT * _rotational_dynamics_residual_prediction_loss(
                        p_rot, yrot_tr[idx])
                    + PHYSICS_UNCERTAINTY_NLL_WEIGHT * _physics_uncertainty_phase12_loss(
                        p_logvar, p_kin, p_cargo, p_rot, ykin_tr[idx], ycargo_tr[idx], yrot_tr[idx])
                    + TRAJECTORY_FORECAST_WEIGHT * _trajectory_forecast_loss(p_trajectory, ytrajectory_tr[idx])
                    + ACTION_TRANSITION_WEIGHT * _action_conditioned_transition_loss(
                        p_action_transition, yaction_transition_tr[idx])
                    + ENVIRONMENT_TRANSITION_WEIGHT * _environment_transition_prediction_loss(
                        p_env_transition, yenv_tr[idx]))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_preds_g = []
            val_preds_f = []
            val_preds_kin = []
            val_preds_cargo = []
            val_preds_rot = []
            val_preds_trajectory = []
            val_preds_uncertainty = []
            val_preds_action_transition = []
            val_preds_environment_transition = []
            for vb in range(0, X_val.shape[0], 300):
                v_bx = X_val[vb:vb+300]
                vg_sub, vf_sub, _, _, _, _, vkin_sub, vcargo_sub, vrot_sub, vtrajectory_sub, vaction_transition_sub, vlogvar_sub, venv_sub = model(
                    v_bx, return_physics=True, return_rotational_residual=True, return_uncertainty=True,
                    return_trajectory_forecast=True, return_action_transition=True,
                    return_environment_transition=True, transition_action=action_condition_va[vb:vb+300])
                val_preds_g.append(vg_sub)
                val_preds_f.append(vf_sub)
                val_preds_kin.append(vkin_sub)
                val_preds_cargo.append(vcargo_sub)
                val_preds_rot.append(vrot_sub)
                val_preds_trajectory.append(vtrajectory_sub)
                val_preds_uncertainty.append(vlogvar_sub)
                val_preds_action_transition.append(vaction_transition_sub)
                val_preds_environment_transition.append(venv_sub)

            vg = torch.cat(val_preds_g, dim=0)
            vf = torch.cat(val_preds_f, dim=0)
            val_mse = crit_mse(vg, yg_va).item()
            vkin = torch.cat(val_preds_kin, dim=0)
            val_kin_mse = float(_kinematics_prediction_loss(vkin, ykin_va).item())
            val_kin_residual_rmse = _kinematic_equation_residual_rmse_mps(
                vkin, X_val, ykin_va, dt=0.016)
            vcargo = torch.cat(val_preds_cargo, dim=0)
            val_cargo_mse = float(_cargo_dynamics_prediction_loss(vcargo, ycargo_va).item())
            vrot = torch.cat(val_preds_rot, dim=0)
            vtrajectory = torch.cat(val_preds_trajectory, dim=0)
            trajectory_stats = _trajectory_forecast_validation_stats(vtrajectory, ytrajectory_va)
            vtrajectory_loss = float(_trajectory_forecast_loss(vtrajectory, ytrajectory_va).item())
            vaction_transition = torch.cat(val_preds_action_transition, dim=0)
            venvironment_transition = torch.cat(val_preds_environment_transition, dim=0)
            environment_transition_loss = float(_environment_transition_prediction_loss(
                venvironment_transition, yenv_va).item())
            environment_transition_stats = _environment_transition_validation_stats(
                venvironment_transition, yenv_va)
            action_transition_loss = float(_action_conditioned_transition_loss(
                vaction_transition, yaction_transition_va).item())
            action_transition_stats = _action_conditioned_transition_validation_stats(
                vaction_transition, yaction_transition_va)
            action_transition_baseline_stats = _action_conditioned_transition_constant_velocity_baseline_stats(
                X_val, yaction_transition_va, dt=0.016)
            action_transition_gain_pos = _action_transition_relative_gain_pct(
                action_transition_baseline_stats["delta_position_rmse_m"],
                action_transition_stats["delta_position_rmse_m"])
            action_transition_gain_vel = _action_transition_relative_gain_pct(
                action_transition_baseline_stats["delta_velocity_rmse_mps"],
                action_transition_stats["delta_velocity_rmse_mps"])
            action_transition_gain_omega = _action_transition_relative_gain_pct(
                action_transition_baseline_stats["delta_omega_rmse_rads"],
                action_transition_stats["delta_omega_rmse_rads"])
            vlogvar = torch.cat(val_preds_uncertainty, dim=0)
            val_unc_kin = _uncertainty_calibration_stats(vkin, vlogvar[:, 0:6], ykin_va, 6)
            val_unc_cargo = _uncertainty_calibration_stats(vcargo, vlogvar[:, 6:11], ycargo_va, 5)
            val_unc_rot = _uncertainty_calibration_stats(vrot, vlogvar[:, 14:17], yrot_va, 3)
            val_rot_mse = float(_rotational_dynamics_residual_prediction_loss(vrot, yrot_va).item())
            val_rot_rmse_nm = _rotational_dynamics_residual_rmse_nm(vrot, yrot_va)
            val_pred_cls = vf.argmax(dim=-1)
            val_acc = (val_pred_cls == yf_va).float().mean().item() * 100.0
            present_val_classes = torch.unique(yf_va)
            per_class_acc = []
            for cls in present_val_classes:
                cls_mask = yf_va == cls
                per_class_acc.append((val_pred_cls[cls_mask] == cls).float().mean())
            balanced_val_acc = (torch.stack(per_class_acc).mean().item() * 100.0
                                if per_class_acc else float("nan"))

        uncertainty_text = (
            f"Uncertainty NLL(k/c/r)={val_unc_kin['nll']:.3f}/{val_unc_cargo['nll']:.3f}/{val_unc_rot['nll']:.3f} "
            f"1sigma={100*val_unc_kin['coverage_1sigma']:.1f}/{100*val_unc_cargo['coverage_1sigma']:.1f}/{100*val_unc_rot['coverage_1sigma']:.1f}%")
        trajectory_text = "; ".join(
            f"{entry['horizon_s']:.3f}s:posRMSE={entry['position_delta_rmse_m']:.3f}m velRMSE={entry['velocity_rmse_mps']:.3f}m/s n={entry['count']}"
            for entry in trajectory_stats)
        environment_transition_text = (
            f"Environment-state next-step loss={environment_transition_loss:.5f} "
            f"windRMSE={environment_transition_stats['wind_vector_rmse_mps']:.3f}m/s "
            f"cableLenRMSE={environment_transition_stats['cable_length_rmse_m']:.3f}m "
            f"cableRateRMSE={environment_transition_stats['cable_rate_rmse_mps']:.3f}m/s "
            f"iceRMSE={environment_transition_stats['icing_rmse_index']:.4f} "
            f"n={environment_transition_stats['count']}")
        action_transition_text = (
            f"Action-conditioned 1-step loss={action_transition_loss:.5f} "
            f"dposRMSE={action_transition_stats['delta_position_rmse_m']:.5f}m "
            f"dvelRMSE={action_transition_stats['delta_velocity_rmse_mps']:.4f}m/s "
            f"domegaRMSE={action_transition_stats['delta_omega_rmse_rads']:.4f}rad/s "
            f"n={action_transition_stats['count']} | CV-baseline "
            f"dpos={action_transition_baseline_stats['delta_position_rmse_m']:.5f}m "
            f"dvel={action_transition_baseline_stats['delta_velocity_rmse_mps']:.4f}m/s "
            f"domega={action_transition_baseline_stats['delta_omega_rmse_rads']:.4f}rad/s "
            f"| learned gain vs CV={action_transition_gain_pos:+.1f}% pos/"
            f"{action_transition_gain_vel:+.1f}% vel/"
            f"{action_transition_gain_omega:+.1f}% omega")
        print(f"Epoch [{epoch+1:02d}/{NUM_MAIN_EPOCHS:02d}] - Multi-Task Loss: {total_loss/num_batches:.4f} | Val MSE: {val_mse:.5f} | Kinematics Val MSE: {val_kin_mse:.5f} | Kinematic Eq-Residual RMSE: {val_kin_residual_rmse:.4f}m/s | Cargo Val MSE: {val_cargo_mse:.5f} | Rotational Euler Residual RMSE: {val_rot_rmse_nm:.5f}N*m | Trajectory forecast loss={vtrajectory_loss:.5f} [{trajectory_text}] | {action_transition_text} | {environment_transition_text} | {uncertainty_text} | Decision Acc: {val_acc:.1f}% | Balanced Action Acc: {balanced_val_acc:.1f}%")

    # V38.34: subgroup evaluation is a post-training diagnostic, not a promotion
    # gate. Overall mean metrics can hide failures in rare FSM commands, winch
    # states, or high-wind observations, so quantify each group on the same holdout.
    model.eval()
    action_transition_group_stats = _action_transition_stratified_validation_stats(
        vaction_transition, yaction_transition_va, X_val, action_condition_va,
        min_group_samples=int(os.environ.get("SURVIVAL_TRANSITION_MIN_GROUP_SAMPLES", "32")),
        dt=0.016)
    for group_entry in action_transition_group_stats:
        if not group_entry.get("adequate", False):
            print(f"[ACTION TRANSITION STRATIFIED] {group_entry['group_type']}={group_entry['group']} "
                  f"n={group_entry['count']} insufficient (need {group_entry['min_group_samples']}); no score claimed.", flush=True)
            continue
        learned = group_entry["learned"]
        gains = group_entry["gain_vs_baseline_pct"]
        print(f"[ACTION TRANSITION STRATIFIED] {group_entry['group_type']}={group_entry['group']} n={group_entry['count']} "
              f"pos={learned['delta_position_rmse_m']:.5f}m ({gains['position']:+.1f}% vs CV) "
              f"vel={learned['delta_velocity_rmse_mps']:.4f}m/s ({gains['velocity']:+.1f}%) "
              f"omega={learned['delta_omega_rmse_rads']:.4f}rad/s ({gains['angular_rate']:+.1f}%).", flush=True)
    action_transition_sensitivity = _action_transition_action_sensitivity_probe(
        model, X_val, action_condition_va, yaction_transition_va, max_samples=128)
    print("[ACTION TRANSITION SENSITIVITY] " + json.dumps(
        _action_transition_json_safe(action_transition_sensitivity), sort_keys=True, allow_nan=False), flush=True)
    transition_readiness = _action_transition_rollout_readiness_gate(
        valid_transition_rows=int(action_transition_stats["count"]),
        overall_gain_vs_baseline_pct={
            "position": action_transition_gain_pos,
            "velocity": action_transition_gain_vel,
            "angular_rate": action_transition_gain_omega,
        },
        stratified_groups=action_transition_group_stats,
        action_sensitivity=action_transition_sensitivity,
        min_overall_rows=int(os.environ.get("SURVIVAL_TRANSITION_GATE_MIN_OVERALL", "1000")),
        min_group_samples=int(os.environ.get("SURVIVAL_TRANSITION_MIN_GROUP_SAMPLES", "32")),
        min_overall_gain_pct=float(os.environ.get("SURVIVAL_TRANSITION_GATE_MIN_GAIN_PCT", "0.0")),
        min_critical_group_gain_pct=float(os.environ.get("SURVIVAL_TRANSITION_GATE_MIN_CRITICAL_GAIN_PCT", "0.0")),
    )
    gate_status = transition_readiness["status"]
    print(f"[ACTION TRANSITION ROLLOUT READINESS] {gate_status}; "
          f"offline multi-step evaluation only; live_control_authorized=False; "
          f"failed_checks={len(transition_readiness['reasons'])}", flush=True)
    for gate_reason in transition_readiness["reasons"]:
        print(f"[ACTION TRANSITION ROLLOUT BLOCKER] {gate_reason}", flush=True)
    action_input_distribution = {
        "training_teacher_actions": _action_transition_action_input_stats(action_condition_tr, yaction_transition_tr),
        "heldout_teacher_actions": _action_transition_action_input_stats(action_condition_va, yaction_transition_va),
        "contract": {
            "harvest_guidance": "Yg[:3] teacher guidance coordinates generated from target acceleration divided by 12, then passed to the mixer",
            "live_guidance": "model guidance output is multiplied by 0.15; selected mission phases then hard-zero guidance before the mixer",
            "comparison_note": "Training and live effective guidance inputs can be distribution-shifted. These stats expose the mismatch; they do not normalize or alter actuation.",
            "fsm_and_winch": "condition features use the FSM token and winch command passed to the transition model; report counts are conditioned on valid paired targets",
        },
    }
    validation_report = {
        "version": SURVIVAL_CORE_VERSION,
        "validation_rows": int(X_val.shape[0]),
        "valid_transition_rows": int(action_transition_stats["count"]),
        "minimum_group_samples": int(os.environ.get("SURVIVAL_TRANSITION_MIN_GROUP_SAMPLES", "32")),
        "overall_learned": action_transition_stats,
        "overall_constant_velocity_baseline": action_transition_baseline_stats,
        "action_input_distribution": action_input_distribution,
        "overall_gain_vs_baseline_pct": {
            "position": action_transition_gain_pos,
            "velocity": action_transition_gain_vel,
            "angular_rate": action_transition_gain_omega,
        },
        "stratified_groups": action_transition_group_stats,
        "action_input_sensitivity": action_transition_sensitivity,
        "rollout_readiness_gate": transition_readiness,
        "rollout_sequence_artifact": rollout_sequence_artifact_info,
        "limitations": [
            "Action sensitivity is response magnitude only, not counterfactual accuracy.",
            "Single-step validation does not establish multi-step rollout accuracy.",
            "A READY gate only authorizes a further offline evaluation, not live controller use.",
            "No live control uses the learned action-conditioned transition model."
        ]
    }
    report_path = Path(os.environ.get(
        "SURVIVAL_ACTION_TRANSITION_REPORT", "action_transition_validation_v38_42.json"))
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as report_file:
            json.dump(_action_transition_json_safe(validation_report), report_file,
                      indent=2, sort_keys=True, allow_nan=False)
        print(f"[ACTION TRANSITION VALIDATION REPORT] saved={report_path.resolve()}", flush=True)
    except (OSError, TypeError, ValueError) as report_exc:
        print(f"[WARNING] Could not write action-transition report to {report_path}: {report_exc}", flush=True)

    # Preserve an existing checkpoint if the user explicitly retrains with the
    # corrected teacher data. The backup is created only once and is never replaced.
    if os.path.exists(MODEL_WEIGHTS_FILE):
        backup_path = MODEL_WEIGHTS_FILE + ".backup_before_v17_retrain"
        if not os.path.exists(backup_path):
            try:
                shutil.copy2(MODEL_WEIGHTS_FILE, backup_path)
                print(f"[CHECKPOINT BACKUP] Existing weights preserved at {backup_path}")
            except OSError as backup_exc:
                print(f"[WARNING] Could not create checkpoint backup: {backup_exc}")
    model._kinematics_head_checkpoint_trained = True
    model._cargo_dynamics_head_checkpoint_trained = True
    model._rotational_dynamics_head_trained_state.fill_(True)
    model._rotational_dynamics_head_checkpoint_trained = True
    model._physics_uncertainty_base_trained_state.fill_(True)
    model._physics_uncertainty_base_checkpoint_trained = True
    model._trajectory_forecast_head_trained_state.fill_(True)
    model._trajectory_forecast_head_checkpoint_trained = True
    model._action_conditioned_transition_head_trained_state.fill_(True)
    model._action_conditioned_transition_head_checkpoint_trained = True
    model._environment_observation_transition_trained_state.fill_(True)
    model._environment_observation_transition_checkpoint_trained = True
    print("[ENVIRONMENT TRANSITION TRAINING] Action-conditioned next observation features (wind, cable length/rate, icing) trained on paired same-drone next-frame targets; diagnostic-only.")
    print("[TRAJECTORY FORECAST TRAINING] Multi-horizon future position/velocity head trained from delayed paired BEMT states; diagnostic-only.")
    print("[ACTION-CONDITIONED TRANSITION TRAINING] One-step delta-position/velocity/body-rate head trained on paired teacher guidance, FSM, winch commands and observed BEMT transitions; diagnostic-only.")
    print("[ROTATIONAL EULER TRAINING] Rotational residual head trained on paired BEMT motor-torque transitions.")
    print("[UNCERTAINTY TRAINING] Kinematics, loaded-cargo and rotational predictive uncertainty calibrated on masked validation targets; force uncertainty awaits Phase 3.")
    torch.save(model.state_dict(), MODEL_WEIGHTS_FILE)
    print("\n" + "="*75)
    print(f"SUCCESS: Policy Model saved to {MODEL_WEIGHTS_FILE} in {time.time()-t0:.2f}s!")
    print("="*75 + "\n")
    return model, device

# =============================================================================
# 17. PHASE 3 — GENESIS 3D PARALLEL POLICY REFINEMENT (HEADLESS)
# =============================================================================
# This phase deliberately fine-tunes the Phase-2 network rather than creating a
# new network. It uses batched Genesis dynamics, policy-guided rollouts and
# teacher labels (DAgger-style supervised refinement). It is NOT PPO/RL: the
# supplied V8 reference also performs supervised learning, not reward-gradient
# optimisation. Mission waypoints and the safety envelope remain teacher-owned.

# Keep prior checkpoints untouched. V38 writes a dedicated reliability-rebuild
# checkpoint so it cannot silently reuse an old V37 Phase-3 result as its final model.
LEGACY_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(".pth", "_phase3_genesis.pth")
PREVIOUS_WIND_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(".pth", "_phase3_wind_robust_v19.pth")
PREVIOUS_PRECISION_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(".pth", "_phase3_precision_delivery_v21.pth")
PREVIOUS_WIND_SENSOR_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(".pth", "_phase3_wind_sensor_v20.pth")
PREVIOUS_CURRICULUM_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(
    ".pth", "_phase3_curriculum_precision_delivery_v22.pth")
PREVIOUS_ROOF_CONTACT_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(
    ".pth", "_phase3_roof_contact_tether_fix_v22_2.pth")
PREVIOUS_COVERAGE_HOVER_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(
    ".pth", "_phase3_training_coverage_hover_braking_v22_3.pth")
PREVIOUS_DEEP_AUDIT_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(
    ".pth", "_phase3_deep_audit_collision_visuals_v22_5.pth")
PREVIOUS_DELIVERY_DESCENT_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(
    ".pth", "_phase3_delivery_descent_collision_visuals_v22_7.pth")
PREVIOUS_V38_1_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(
    ".pth", "_phase3_v38_1_delivery_rollout.pth")
PREVIOUS_V38_7_MEMORY_EFFICIENT_PHASE3_CHECKPOINT_FILE = MODEL_WEIGHTS_FILE.replace(
    ".pth", "_phase3_v38_7_memory_efficient.pth")
PHASE3_CHECKPOINT_FILE = os.environ.get(
    "SURVIVAL_PHASE3_CHECKPOINT",
    MODEL_WEIGHTS_FILE.replace(".pth", "_phase3_v38_9_delivery_latch.pth"),
)


def _phase3_write_training_urdf(filepath):
    """Write a small collidable drone body for fast batched Genesis rollouts."""
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
    """Default to 512 batched CUDA environments so each rollout can run longer."""
    default = "512" if getattr(device, "type", "cpu") == "cuda" else "16"
    try:
        return int(os.environ.get("SURVIVAL_PHASE3_ENVS", default))
    except (TypeError, ValueError):
        return int(default)


def _phase3_env_candidates(requested):
    """Return descending batch-size fallback candidates without exceeding the request."""
    requested = int(np.clip(requested, 1, 8192))
    candidates = []
    value = requested
    while True:
        if value not in candidates:
            candidates.append(value)
        if value <= 1:
            break
        value = max(1, value // 2)
    return candidates


def _phase3_target_pad_surface_table(target_positions):
    """Copy manifest rooftop positions and raise z to the actual pad top surface.

    The physical pad cylinder is 0.08 m high and centred at building_top+0.04;
    therefore its top surface is building_top+0.08. Using the manifest building
    top as the touchdown surface makes the cargo-contact test impossible for a
    correctly resting 0.18 m high parcel.
    """
    table = np.asarray(target_positions, dtype=np.float32).copy()
    if table.ndim != 2 or table.shape[1] < 3:
        raise ValueError(f"Expected target positions shaped [N, >=3], got {table.shape}")
    table[:, 2] += PHASE3_ROOF_PAD_TOP_OFFSET_M
    return table


def _phase3_cargo_center_z_for_cable(drone_center_z, cable_len):
    """Cargo centre z consistent with a cable measured between physical surfaces."""
    return drone_center_z - cable_len - PHASE3_TETHER_CENTER_OFFSET_M


def _live_payload_center_z(drone_center_z, cable_len):
    """Estimate parcel-centre Z from drone centre and surface-to-surface payout length."""
    return float(drone_center_z) - PHASE3_DRONE_HALF_HEIGHT_M - max(
        PHASE3_STOWED_CABLE_LEN_M, float(cable_len)) - PHASE3_CARGO_HALF_HEIGHT_M


def _live_payload_bottom_clearance(drone_center_z, cable_len, target_surface_z):
    """Clearance from parcel bottom to the actual pad top; positive means above the pad."""
    parcel_center_z = _live_payload_center_z(drone_center_z, cable_len)
    return parcel_center_z - PHASE3_CARGO_HALF_HEIGHT_M - float(target_surface_z)


def _phase3_tether_slack(center_distance, cable_len):
    """Return positive extension beyond cable length, accounting for body half-heights."""
    return center_distance - cable_len - PHASE3_TETHER_CENTER_OFFSET_M


def _phase3_torch_rotation_from_quaternion(quat_wxyz):
    """Convert Genesis wxyz quaternions into body-to-world rotation matrices."""
    if quat_wxyz.shape[-1] != 4:
        raise ValueError(f"Expected wxyz quaternion final dimension 4; got {tuple(quat_wxyz.shape)}")
    q = quat_wxyz / torch.linalg.vector_norm(quat_wxyz, dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q.unbind(dim=-1)
    r00 = 1.0 - 2.0 * (y*y + z*z)
    r01 = 2.0 * (x*y - w*z)
    r02 = 2.0 * (x*z + w*y)
    r10 = 2.0 * (x*y + w*z)
    r11 = 1.0 - 2.0 * (x*x + z*z)
    r12 = 2.0 * (y*z - w*x)
    r20 = 2.0 * (x*z - w*y)
    r21 = 2.0 * (y*z + w*x)
    r22 = 1.0 - 2.0 * (x*x + y*y)
    return torch.stack((r00, r01, r02, r10, r11, r12, r20, r21, r22), dim=-1).reshape(
        *quat_wxyz.shape[:-1], 3, 3
    )


def _phase3_torch_features(pos, vel, waypoint, roof_z, cable_len, cable_vel,
                           has_payload, winch_state, phase, wind, accel,
                           building_aabbs, sway_2d=None, sway_velocity_2d=None,
                           tension_n=None, rotation_matrix=None, omega_body=None):
    """Build 139 features with the V17 feature index/scaling contract.

    The batched training world exposes exact kinematics, roof ranges, nearby
    building ray ranges, cable state and peer-relative state. Unsimulated avionics
    fields use the same bounded nominal placeholders used by the teacher harvester.
    """
    device = pos.device
    n_envs, n_drones, _ = pos.shape
    out = torch.zeros((n_envs, n_drones, BASE_OBS_DIM), device=device, dtype=torch.float32)
    out[..., 0:3] = (waypoint - pos) / 50.0
    out[..., 3:6] = pos / 200.0
    out[..., 6:9] = vel / 8.0
    if rotation_matrix is None:
        rotation_matrix = torch.eye(3, device=device, dtype=torch.float32).reshape(1, 1, 3, 3).expand(
            n_envs, n_drones, 3, 3
        )
    else:
        rotation_matrix = rotation_matrix.to(device=device, dtype=torch.float32)
        if tuple(rotation_matrix.shape) != (n_envs, n_drones, 3, 3):
            raise ValueError(f"rotation_matrix must have shape {(n_envs, n_drones, 3, 3)}, got {tuple(rotation_matrix.shape)}")
    out[..., 9:18] = rotation_matrix.reshape(n_envs, n_drones, 9)
    if omega_body is None:
        omega_body = torch.zeros_like(pos)
    else:
        omega_body = omega_body.to(device=device, dtype=torch.float32)
        if tuple(omega_body.shape) != (n_envs, n_drones, 3):
            raise ValueError(f"omega_body must have shape {(n_envs, n_drones, 3)}, got {tuple(omega_body.shape)}")
    out[..., 18:21] = omega_body / 10.0

    # Vectorised horizontal 16-ray slab intersections against the training towers.
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
    # Exact distance to the same finite segments used by the runtime world.
    # Clamping x to the segment endpoints matters for positions beyond +/-190 m.
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
    # These slots match the live observation synthesizer's horizontal sway features.
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
    # Approximate the live smart-hook IMU and 3-axis load-cell values from the
    # same per-drone sway/tension estimate used by the controller.
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

    # Match the live synthesizer's two peer slots for each of the three drones.
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
    """Live-equivalent drone geometry/velocity gate, excluding cable sway itself."""
    xy_error = torch.linalg.vector_norm(target_roofs[..., :2] - pos[..., :2], dim=-1)
    horizontal_speed = torch.linalg.vector_norm(vel[..., :2], dim=-1)
    altitude_error = torch.abs(pos[..., 2] - (target_roofs[..., 2] + 10.0))
    return ((xy_error <= DELIVERY_DRONE_CENTER_TOLERANCE_M)
            & (altitude_error <= DELIVERY_HOVER_ALTITUDE_TOLERANCE_M)
            & (horizontal_speed <= DELIVERY_MAX_HORIZONTAL_SPEED_MS)
            & (torch.abs(vel[..., 2]) <= DELIVERY_MAX_VERTICAL_SPEED_MS))


def _phase3_cargo_contact_mask(cargo_pos, target_roofs):
    """Cargo-centre must be within 3 cm of the actual pad-top contact height."""
    cargo_xy_error = torch.linalg.vector_norm(cargo_pos[..., :2] - target_roofs[..., :2], dim=-1)
    # target_roofs is the pad's physical top surface, because the target table is
    # built by _phase3_target_pad_surface_table(). Do not add the pad offset twice.
    pad_surface_z = target_roofs[..., 2]
    expected_center_z = pad_surface_z + PHASE3_CARGO_HALF_HEIGHT_M
    cargo_z_error = torch.abs(cargo_pos[..., 2] - expected_center_z)
    return ((cargo_xy_error <= DELIVERY_CARGO_CENTER_TOLERANCE_M)
            & (cargo_z_error <= 0.03))


def _phase3_delivery_stability_mask(pos, vel, target_roofs, sway_2d, phase):
    """Return the geometry/velocity/sway envelope required for precision station keeping.

    The separate dwell timer is applied by the rollout loop. This helper only reports
    whether the current frame is stable enough to contribute to that continuous dwell.
    """
    sway_mag = torch.linalg.vector_norm(sway_2d, dim=-1)
    drone_aligned = _phase3_drone_delivery_alignment_mask(pos, vel, target_roofs)
    return (phase == 3) & drone_aligned & (sway_mag <= DELIVERY_MAX_SWAY_M)


def _phase3_project_tether_and_contact(drone_pos, drone_vel, cargo_pos, cable_len,
                                       has_payload, winch_state, phase, target_roofs,
                                       released_package_valid=None, released_package_pos=None):
    """Enforce tether geometry and project the parcel onto the physical pad top.

    The input is the pad-top table produced by _phase3_target_pad_surface_table,
    not raw building heights. Contact is gated by available cable length and safe
    geometry; the old circular test required the parcel to be at contact height
    before allowing the tether projection to bring it there.
    """
    center_limit = torch.clamp(cable_len, min=PHASE3_STOWED_CABLE_LEN_M) + PHASE3_TETHER_CENTER_OFFSET_M
    delta = cargo_pos - drone_pos
    distance = torch.linalg.vector_norm(delta, dim=-1).clamp(min=1e-5)
    unit = delta / distance[..., None]
    cable_projected = drone_pos + unit * center_limit[..., None]
    lowering = (phase == 3) & (winch_state == 1) & has_payload
    cargo_out = torch.where(lowering[..., None], cable_projected, cargo_pos)

    # target_roofs already stores the physical pad-top Z.
    pad_surface_z = target_roofs[..., 2]
    contact_center = target_roofs.clone()
    contact_center[..., 2] = pad_surface_z + PHASE3_CARGO_HALF_HEIGHT_M
    contact_xy_error = torch.linalg.vector_norm(cargo_out[..., :2] - target_roofs[..., :2], dim=-1)
    contact_center_distance = torch.linalg.vector_norm(drone_pos - contact_center, dim=-1)
    required_cable_len = torch.clamp(contact_center_distance - PHASE3_TETHER_CENTER_OFFSET_M, min=0.0)
    cable_reach_ok = cable_len + 0.035 >= required_cable_len
    drone_safe = _phase3_drone_delivery_alignment_mask(drone_pos, drone_vel, target_roofs)
    # Preserve the parcel's measured XY within the legal drop-zone radius; only
    # project Z to the contact surface. This avoids a nonphysical horizontal snap.
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
    """Waypoint state machine: transit, cruise-height brake/align, then vertical approach."""
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
    """Sample an initial-state curriculum with enough time for full deliveries.

    State probabilities are: transit 10%, cruise/brake 10%, pre-descent 20%,
    delivery 40%, return 10%, depot landing/service 10%. Every mission phase
    remains represented, but delivery demonstrations are no longer diluted to
    1/6 of agents by a uniform draw.
    """
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
    """Sample live-regime and randomized strong winds for robust Phase-3 training.

    Horizontal wind covers calm conditions through roughly 9.5 m/s; vertical
    components include microburst downdrafts. Targets are subsequently low-pass
    transitioned and perturbed by a correlated gust process in the rollout.
    """
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
    """Teacher labels for guidance/FSM/winch plus shape-compatible auxiliary placeholders.

    Phase-3 fitting distils the existing model's health/trust/value heads instead of
    fitting those heads to synthetic constants from this reduced physics scene.
    """
    delta = waypoint - pos
    clipped_delta = torch.clamp(delta, -5.0, 5.0)
    dxy = delta[..., :2]
    dist_xy = torch.linalg.vector_norm(dxy, dim=-1, keepdim=True)
    speed_xy = torch.clamp(torch.sqrt(torch.clamp(3.0 * dist_xy, min=0.0)), max=8.0)
    # Final delivery is a station-keeping task, not a fast waypoint pass. Reduce
    # desired lateral speed linearly with error near the target to avoid overshoot.
    delivery_speed_xy = torch.clamp(0.45 * dist_xy, max=1.2)
    speed_xy = torch.where((phase == 3)[..., None], delivery_speed_xy, speed_xy)
    vxy = torch.where(dist_xy > 1e-3, dxy / torch.clamp(dist_xy, min=1e-3) * speed_xy, torch.zeros_like(dxy))
    vz = torch.clamp(delta[..., 2] * 0.5, min=-0.65, max=2.4)
    v_des = torch.cat((vxy, vz[..., None]), dim=-1)
    accel = torch.empty_like(pos)
    cruise_accel_xy = 1.6 * clipped_delta[..., :2] + 1.8 * (v_des[..., :2] - vel[..., :2])
    # During a delivery the drone must remain over the pad for the entire winch
    # payout. Use the same stronger position/velocity gains as the live controller;
    # the old cruise gains allowed gust-driven drift to trip the 1.4 m abort gate
    # within a fraction of a second, before the cable could reach the roof.
    delivery_accel_xy = 2.65 * clipped_delta[..., :2] + 3.25 * (v_des[..., :2] - vel[..., :2])
    accel[..., :2] = torch.where((phase == 3)[..., None], delivery_accel_xy, cruise_accel_xy)
    accel[..., 2] = 3.2 * (torch.clamp(v_des[..., 2] + 0.85 * clipped_delta[..., 2], -1.0, 1.5) - vel[..., 2])

    # Physics-informed feed-forward label. Translational drag is
    # F_D = -0.5 * rho * CdA * |v - wind| * (v - wind); the acceleration command
    # must cancel that force (minus F_D / m) while holding position. Use the same
    # CdA coefficients as BEMTCoaxialAirframe8DOF and the live mixer.
    if wind is not None:
        wind_per_drone = (wind[:, None, :].expand(-1, vel.shape[1], -1) if wind.ndim == 2 else wind)
        air_velocity = vel - wind_per_drone
        air_speed = torch.linalg.vector_norm(air_velocity, dim=-1, keepdim=True).clamp(min=1.0)
        cda = torch.tensor([0.22, 0.22, 0.38], dtype=vel.dtype, device=vel.device)
        drag_force = -0.5 * 1.225 * cda * air_speed * air_velocity
        # Genesis training URDF mass is DRY_MASS_KG. Use that same physical
        # mass for force-to-acceleration conversion; using dry+cargo here while
        # applying force to a dry-mass body caused a systematic wind-compensation
        # mismatch. The suspended-load force is compensated separately below.
        body_mass = max(0.1, float(DRY_MASS_KG))
        accel = accel - drag_force / body_mass
        if tension_n is not None:
            tether_vector = cargo_pos - pos
            tether_distance = torch.linalg.vector_norm(tether_vector, dim=-1, keepdim=True).clamp(min=1e-4)
            tether_direction = tether_vector / tether_distance
            tension = torch.clamp(tension_n.to(device=pos.device, dtype=pos.dtype), min=0.0)[..., None]
            accel = accel - (tension * tether_direction) / body_mass

    # Match the live mixer: counteract both the lateral cable displacement and
    # relative swing velocity while the payload is attached and cable is deployed.
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

    # Match V17 ACTION_NAMES, not the mission-phase numbers: 0=TRANSIT,
    # 1=HOVER_LOCK, 2=WINCH_LOWER, 3=TOUCHDOWN_RELEASE,
    # 4=WINCH_RETRACT, 5=DEPOT_SERVICE, 6=EMERGENCY_DIVERT.
    fsm = torch.zeros(phase.shape, dtype=torch.long, device=phase.device)
    fsm = torch.where((phase == 2), torch.ones_like(fsm), fsm)  # brake/align/settle at cruise altitude
    fsm = torch.where((phase == 6) & (pos[..., 2] < 0.30), torch.full_like(fsm, 5), fsm)
    delivery = phase == 3
    # At delivery, teach HOVER_LOCK until the dwell is complete, then lower.
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
    # Do not teach TOUCHDOWN_RELEASE just because the payload is low: the drone
    # itself must also still satisfy the live delivery envelope.
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
    """Build a light 3D training city: 8 spaced towers, actual depot/roof pads and wires."""
    # Select spatially dispersed towers to retain a meaningful city-scale route
    # while keeping hundreds/thousands of clones feasible in GPU memory.
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
        # Roof plant is physical in the training scene, matching the live hitbox map.
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
        line_yaw = math.atan2(float(delta[1]), float(delta[0]))
        # These training-scene wires are fixed collision bodies. V22.5 uses the
        # same +/-185 m position as the live world, not stale +/-170 m coordinates.
        wire = scene.add_entity(
            morph=gs.morphs.Box(size=(line_length, 0.10, 0.10), pos=tuple(0.5*(p1+p2)), fixed=True),
            material=gs.materials.Rigid(friction=0.2),
            surface=gs.surfaces.Default(color=(0.55, 0.58, 0.62)),
        )
        # Four pole colliders bound the ends of the two line spans.
        for endpoint in (p1, p2):
            pole_h = float(endpoint[2])
            scene.add_entity(
                morph=gs.morphs.Cylinder(radius=0.16, height=pole_h,
                                          pos=(float(endpoint[0]), float(endpoint[1]), pole_h/2), fixed=True),
                material=gs.materials.Rigid(friction=0.7),
                surface=gs.surfaces.Default(color=(0.30, 0.33, 0.37)),
            )
            # Crossarm is also a collision body; its visible box is not just render art.
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
    """Bound Phase-3 dataset growth without discarding brief release/service states.

    Regular actions are sampled every sample_interval rollout steps. The dominant
    lower/retract labels are retained at a coarser stride, while touchdown/release
    and depot-service labels are retained on every frame so short transitions do
    not disappear from the training set.
    """
    if torch is None or not torch.is_tensor(fsm_y):
        raise TypeError("fsm_y must be a torch.Tensor")
    sample_interval = max(1, int(sample_interval))
    critical_stride = max(1, int(critical_stride))
    base = torch.full_like(fsm_y, (int(step) % sample_interval == 0), dtype=torch.bool)
    retain_short_states = (fsm_y == 3) | (fsm_y == 5)
    # Element-wise expression (rather than Python bool) for batched tensor labels.
    retain_lower_retract = ((fsm_y == 2) | (fsm_y == 4)) & (int(step) % critical_stride == 0)
    return base | retain_short_states | retain_lower_retract


# -----------------------------------------------------------------------------
# V38.42 TRANSITION CONTRACT / BASELINE PARITY AUDIT
# Live shadow baseline uses observation-derived velocity exactly like offline scoring;
# guidance pre/post phase-gate distributions are now reported for diagnosing shift.

# V38.41 PERSISTENT DAgger REPLAY CORPUS
# -----------------------------------------------------------------------------
# Phase 3 already collects policy-visited Genesis states with expert labels. Prior
# versions discarded those examples after fine-tuning and randomly split correlated
# rows, which can leak adjacent samples from one environment into both train/holdout.
# V38.41 persists training groups only and holds out complete Genesis environment
# groups for each run. Replay remains supervised data, not RL or control authority.
PHASE3_DAGGER_SCHEMA_VERSION = 1
PHASE3_DAGGER_ARRAY_KEYS = ("X", "Yg", "Yf", "Yw", "Yh", "Yt", "Yv", "Ykin", "Ycargo", "Yforce")
PHASE3_DAGGER_DEFAULT_MAX_SAMPLES = 500_000

def _phase3_dagger_replay_path():
    default_path = PHASE3_CHECKPOINT_FILE.replace(".pth", "_dagger_replay_v1.npz")
    return os.path.abspath(os.environ.get("SURVIVAL_DAGGER_REPLAY_FILE", default_path))

def _phase3_dagger_expected_shapes(arrays):
    expected = {}
    for key in PHASE3_DAGGER_ARRAY_KEYS:
        if key not in arrays:
            raise ValueError(f"DAgger corpus missing required array {key}")
        value = np.asarray(arrays[key])
        if value.ndim < 1:
            raise ValueError(f"DAgger array {key} must have a sample dimension")
        expected[key] = tuple(value.shape[1:])
    return expected

def _phase3_load_dagger_replay(path, expected_arrays):
    """Load only a validated training replay corpus; corrupt/stale files are rejected."""
    if not path or not os.path.isfile(path):
        return None
    expected_shapes = _phase3_dagger_expected_shapes(expected_arrays)
    try:
        with np.load(path, allow_pickle=False) as data:
            if "metadata_json" not in data or "source_group_ids" not in data:
                raise ValueError("missing metadata_json/source_group_ids")
            metadata = json.loads(str(data["metadata_json"].item()))
            if int(metadata.get("schema_version", -1)) != PHASE3_DAGGER_SCHEMA_VERSION:
                raise ValueError(f"unsupported schema_version={metadata.get('schema_version')}")
            if int(metadata.get("observation_dim", -1)) != OBSERVATION_DIM:
                raise ValueError("observation dimension mismatch")
            arrays = {}
            sample_count = None
            for key in PHASE3_DAGGER_ARRAY_KEYS:
                if key not in data:
                    raise ValueError(f"missing array {key}")
                dtype = np.int64 if key == "Yf" else np.float32
                arr = np.asarray(data[key], dtype=dtype)
                if arr.ndim < 1 or tuple(arr.shape[1:]) != expected_shapes[key]:
                    raise ValueError(f"shape mismatch for {key}: got {arr.shape}, expected (*,{expected_shapes[key]})")
                if sample_count is None:
                    sample_count = len(arr)
                elif len(arr) != sample_count:
                    raise ValueError("sample-count mismatch across arrays")
                if key != "Yf" and not np.isfinite(arr).all():
                    raise ValueError(f"non-finite values in {key}")
                arrays[key] = arr.copy()
            groups = np.asarray(data["source_group_ids"], dtype=np.int64)
            if groups.shape != (int(sample_count or 0),):
                raise ValueError("source_group_ids length mismatch")
            if sample_count == 0:
                raise ValueError("empty replay corpus")
        arrays["source_group_ids"] = groups.copy()
        print(f"[DAgger REPLAY] loaded {len(groups):,} prior training rows from {path}; "
              f"groups={len(np.unique(groups)):,}; holdout remains current-run-only.", flush=True)
        return arrays
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"[DAgger REPLAY WARNING] rejected replay corpus {path}: {type(exc).__name__}: {exc}. "
              "Continuing with current-run data only.", flush=True)
        return None

def _phase3_group_heldout_split(source_env_ids, rng, val_fraction=0.10):
    """Return row indices with whole source environments held out for validation."""
    groups = np.asarray(source_env_ids, dtype=np.int64).reshape(-1)
    unique_groups = np.unique(groups)
    fraction = float(val_fraction)
    if len(unique_groups) < 2:
        raise ValueError("at least two independent source environment groups are required")
    if not math.isfinite(fraction) or not (0.0 < fraction < 1.0):
        raise ValueError("val_fraction must be finite and strictly between zero and one")
    shuffled = rng.permutation(unique_groups)
    n_val_groups = max(1, int(math.ceil(fraction * len(shuffled))))
    n_val_groups = min(n_val_groups, len(shuffled) - 1)
    val_groups = shuffled[:n_val_groups]
    val_mask = np.isin(groups, val_groups)
    train_idx = np.flatnonzero(~val_mask).astype(np.int64, copy=False)
    val_idx = np.flatnonzero(val_mask).astype(np.int64, copy=False)
    if not len(train_idx) or not len(val_idx):
        raise ValueError("group-held-out split produced an empty partition")
    if np.intersect1d(np.unique(groups[train_idx]), np.unique(groups[val_idx])).size:
        raise AssertionError("source environment group leaked across train/validation")
    return train_idx, val_idx, np.asarray(val_groups, dtype=np.int64)

def _phase3_dagger_cap_indices(labels, max_samples):
    """Deterministically cap corpus size while preserving rare nonempty FSM classes."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    n, cap = len(labels), int(max_samples)
    if cap <= 0 or n <= cap:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(3841)
    classes, counts = np.unique(labels, return_counts=True)
    allocation = {int(c): 1 for c in classes if counts[np.where(classes == c)[0][0]] > 0}
    remaining = max(0, cap - len(allocation))
    total = max(1, int(counts.sum()))
    quotas = {int(c): remaining * int(nc) / total for c, nc in zip(classes, counts)}
    for c in allocation:
        allocation[c] += int(quotas[c])
    used = sum(allocation.values())
    # Distribute rounding remainder to classes with the largest fractional quota.
    order = sorted((int(c) for c in classes), key=lambda c: quotas[c] - int(quotas[c]), reverse=True)
    cursor = 0
    while used < cap and order:
        c = order[cursor % len(order)]
        class_count = int(np.sum(labels == c))
        if allocation[c] < class_count:
            allocation[c] += 1; used += 1
        cursor += 1
        if cursor > cap * max(1, len(order)):
            break
    selected = []
    for c in classes:
        ids = np.flatnonzero(labels == c)
        take = min(len(ids), allocation[int(c)])
        if take:
            selected.append(rng.choice(ids, size=take, replace=False))
    out = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
    if len(out) > cap:
        out = rng.choice(out, size=cap, replace=False)
    elif len(out) < cap:
        remainder = np.setdiff1d(np.arange(n, dtype=np.int64), out, assume_unique=False)
        if len(remainder):
            out = np.concatenate((out, rng.choice(remainder, size=min(cap-len(out), len(remainder)), replace=False)))
    rng.shuffle(out)
    return out.astype(np.int64, copy=False)

def _phase3_save_dagger_replay(path, arrays, source_group_ids, max_samples=PHASE3_DAGGER_DEFAULT_MAX_SAMPLES):
    """Atomically save a bounded, training-only replay corpus with an explicit schema."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n = len(np.asarray(arrays["X"]))
    groups = np.asarray(source_group_ids, dtype=np.int64).reshape(-1)
    if n < 1 or groups.shape != (n,):
        raise ValueError("DAgger save requires aligned nonempty arrays and source_group_ids")
    normalized = {}
    for key in PHASE3_DAGGER_ARRAY_KEYS:
        arr = np.asarray(arrays[key], dtype=np.int64 if key == "Yf" else np.float32)
        if len(arr) != n:
            raise ValueError(f"DAgger save length mismatch for {key}")
        if key != "Yf" and not np.isfinite(arr).all():
            raise ValueError(f"DAgger save rejected non-finite values in {key}")
        normalized[key] = arr
    cap = int(os.environ.get("SURVIVAL_DAGGER_REPLAY_MAX_SAMPLES", max_samples))
    cap = max(1, cap)
    selected = _phase3_dagger_cap_indices(normalized["Yf"], cap)
    normalized = {key: value[selected] for key, value in normalized.items()}
    groups = groups[selected]
    metadata = {"schema_version": PHASE3_DAGGER_SCHEMA_VERSION, "observation_dim": OBSERVATION_DIM,
                "sample_count": int(len(groups)), "source_group_count": int(len(np.unique(groups))),
                "selection": "FSM-stratified deterministic cap", "split_method": "whole Genesis environment groups; validation rows excluded",
                "target_families": list(PHASE3_DAGGER_ARRAY_KEYS), "version": SURVIVAL_CORE_VERSION,
                "created_unix_s": float(time.time())}
    temp_path = path + f".tmp-{os.getpid()}"
    try:
        with open(temp_path, "wb") as handle:
            np.savez_compressed(handle, **normalized, source_group_ids=groups,
                                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except OSError: pass
    print(f"[DAgger REPLAY] atomically saved {len(groups):,} training rows / "
          f"{len(np.unique(groups)):,} source groups to {path} (cap={cap:,}).", flush=True)

def train_phase3_genesis_parallel(model, device):
    """Phase 3: headless Genesis 3D batched rollouts + conservative supervised fine-tuning.

    Environment variables:
      SURVIVAL_PHASE3_ENVS   (default 512 on CUDA; 16 on CPU; auto-halves on build failure)
      SURVIVAL_PHASE3_STEPS  (default 3072 simulation steps / 30.72 simulated seconds)
      SURVIVAL_PHASE3_EPOCHS (default 4 fine-tuning epochs)
      SURVIVAL_PHASE3_BATCH  (default 1024)
      SURVIVAL_PHASE3_CRITICAL_SAMPLE_STRIDE (default 8; retain lower/retract frames every 8 rollout steps)

    The Phase-2 model is used during rollouts and its own weights are fine-tuned.
    Teacher guidance keeps trajectories in a safe, solvable envelope. This stage is
    supervised on-policy refinement, not PPO/reinforcement learning.
    """
    import gc
    import tempfile
    from contextlib import nullcontext

    source_device = torch.device(device)
    # Keep a CPU copy of the unmodified Phase-2 policy. The Genesis rollout is
    # supervised data collection, not parameter training, so this is the exact
    # starting checkpoint needed if post-rollout fine-tuning moves to another device.
    phase3_start_state_cpu = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    default_train_device = "cpu" if source_device.type == "cuda" else source_device.type
    requested_train_device = os.environ.get(
        "SURVIVAL_PHASE3_TRAIN_DEVICE", default_train_device
    ).strip().lower()
    if requested_train_device not in {"cpu", "cuda"}:
        print(
            f"[PHASE 3 WARNING] Invalid SURVIVAL_PHASE3_TRAIN_DEVICE={requested_train_device!r}; "
            f"using {default_train_device}.", flush=True
        )
        requested_train_device = default_train_device
    if requested_train_device == "cuda" and not torch.cuda.is_available():
        print("[PHASE 3 WARNING] CUDA fine-tuning requested but CUDA is unavailable; using CPU.", flush=True)
        requested_train_device = "cpu"

    try:
        import genesis as gs
    except ImportError as exc:
        raise RuntimeError("Genesis is not installed in this Python environment; Phase 3 cannot run.") from exc

    requested_envs = _phase3_requested_envs(device)
    try:
        steps = int(os.environ.get("SURVIVAL_PHASE3_STEPS", str(PHASE3_DEFAULT_ROLLOUT_STEPS)))
    except (TypeError, ValueError):
        steps = PHASE3_DEFAULT_ROLLOUT_STEPS
    try:
        epochs = int(os.environ.get("SURVIVAL_PHASE3_EPOCHS", "4"))
    except (TypeError, ValueError):
        epochs = 4
    try:
        batch_size = int(os.environ.get("SURVIVAL_PHASE3_BATCH", "1024"))
    except (TypeError, ValueError):
        batch_size = 1024
    try:
        teacher_winch_ratio = float(os.environ.get("SURVIVAL_PHASE3_TEACHER_WINCH_RATIO", "0.50"))
    except (TypeError, ValueError):
        teacher_winch_ratio = 0.50
    teacher_winch_ratio = float(np.clip(teacher_winch_ratio, 0.0, 1.0))
    steps = int(np.clip(steps, 64, 10000))
    epochs = int(np.clip(epochs, 1, 20))
    batch_size = int(np.clip(batch_size, 64, 4096))
    dt = 0.01

    backend = gs.gpu if getattr(device, "type", "cpu") == "cuda" and hasattr(gs, "gpu") else gs.cpu
    try:
        gs.init(backend=backend, logging_level=_configured_genesis_logging_level())
    except Exception as exc:
        # A new subprocess is used for Phase 3, so an init error is a real error,
        # not something to swallow silently.
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
                # Start the drone just above its pad with the payload on the stowed cable,
                # matching the source system's loaded-departure state.
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
            print(f"[PHASE 3 WARNING] {candidate:,} environments failed to build: {type(exc).__name__}: {exc}", flush=True)
            scene = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if scene is None or n_envs is None:
        raise RuntimeError(f"Could not build the Genesis batch scene; last error: {build_error}")
    if n_envs != requested_envs:
        print(f"[PHASE 3] Reduced environment count to {n_envs:,} after the requested build failed.", flush=True)

    train_device = torch.device(device)
    target_table_np = _phase3_target_pad_surface_table(
        [np.asarray(t["pos"], dtype=np.float32) for t in targets])
    target_table = torch.as_tensor(target_table_np, device=train_device)
    roofs_table = target_table[:, 2]
    home_np = np.asarray(world.depot_helipads, dtype=np.float32)
    homes_one = torch.as_tensor(home_np, device=train_device)
    cruise_z = torch.as_tensor([42.0, 45.0, 48.0], device=train_device, dtype=torch.float32)
    aabb_tensor = torch.as_tensor(building_aabbs, device=train_device)

    # Different targets and mission phases in each clone give the policy varied
    # on-policy 3D states rather than 1,024 copies of one deterministic trajectory.
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
    target_cruise = target_roofs.clone()
    target_cruise[..., 2] = cruise_z.reshape(1, -1)
    hover_pos = target_roofs.clone()
    hover_pos[..., 2] = target_roofs[..., 2] + 10.0
    # Populate each initial phase near its corresponding valid starting region.
    phase_pos[..., :2] = torch.where((phase == 1)[..., None], target_roofs[..., :2] + offset, phase_pos[..., :2])
    phase_pos[..., 2] = torch.where(phase == 1, cruise_z.reshape(1, -1).expand(n_envs, -1), phase_pos[..., 2])
    phase_pos[..., :2] = torch.where(((phase == 2) | (phase == 3))[..., None], target_roofs[..., :2] + 0.5 * offset / torch.clamp(rand_radius[..., None], min=1.0), phase_pos[..., :2])
    # Phase 2 starts at cruise height, close enough to brake but not already
    # descending. It must settle over the target before the vertical approach begins.
    phase_pos[..., 2] = torch.where(phase == 2, cruise_z.reshape(1, -1).expand(n_envs, -1), phase_pos[..., 2])
    phase_pos[..., 2] = torch.where(phase == 3, target_roofs[..., 2] + 10.0, phase_pos[..., 2])
    phase_pos[..., :2] = torch.where((phase == 4)[..., None], target_roofs[..., :2], phase_pos[..., :2])
    phase_pos[..., 2] = torch.where(phase == 4, cruise_z.reshape(1, -1).expand(n_envs, -1), phase_pos[..., 2])
    phase_pos[..., :2] = torch.where((phase == 5)[..., None], homes[..., :2] + 0.5 * offset / torch.clamp(rand_radius[..., None], min=1.0), phase_pos[..., :2])
    phase_pos[..., 2] = torch.where(phase == 5, 1.5 + torch.rand((n_envs, NUM_SWARM_DRONES), device=train_device) * 3.0, phase_pos[..., 2])
    has_payload = (phase <= 3)
    winch_state = torch.zeros((n_envs, NUM_SWARM_DRONES), dtype=torch.long, device=train_device)  # 0 standby, 1 lowering, 2 touchdown, 3 retract
    # Cable length is measured surface-to-surface. Convert it to centre separation
    # so the cargo box does not initially overlap the drone's collision body.
    cable_len = torch.full((n_envs, NUM_SWARM_DRONES), PHASE3_STOWED_CABLE_LEN_M,
                           device=train_device, dtype=torch.float32)
    initial_cargo_pos = phase_pos.clone()
    initial_cargo_pos[..., 2] = _phase3_cargo_center_z_for_cable(phase_pos[..., 2], cable_len)
    # For already-returning episodes, place the delivered parcel on the actual pad
    # top surface: its centre is one cargo half-height above that surface.
    roof_package_pos = target_roofs.clone()
    roof_package_pos[..., 2] += PHASE3_CARGO_HALF_HEIGHT_M
    initial_cargo_pos = torch.where((phase >= 4)[..., None], roof_package_pos, initial_cargo_pos)
    for d_idx, entity in enumerate(drone_entities):
        # Genesis supports a batched ([n_envs, 3]) pose. Do not continue with
        # identical default poses if the required per-environment reset fails.
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
    # Keep already released cargo at its rooftop position; a newly released parcel
    # is added to this set when the touchdown debounce succeeds.
    released_package_valid = phase >= 4
    released_package_pos = roof_package_pos.clone()
    # Half the clones default to safe teacher-forced winch sequencing to ensure
    # the rollout dataset contains lowering/contact/release examples even when the
    # warm-start policy has not learned the new strict deployment gate yet.
    # Give part of the curriculum a controlled low-wind demonstration regime.
    # This is still a physical Genesis rollout (not a fabricated label): the teacher
    # must fly, lower, contact, release and retract the rigid-body package. Other
    # environments retain the full randomized wind distribution for robustness.
    curriculum_demo_env_mask = (torch.rand((n_envs,), device=train_device)
                                < max(0.50, teacher_winch_ratio))
    teacher_winch_mask = (torch.rand((n_envs, NUM_SWARM_DRONES), device=train_device)
                          < teacher_winch_ratio)
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
    # Each clone/drone has a separate noisy air-data + GNSS/INS wind estimate.
    # This state is used for observations and teacher feed-forward, never for physics.
    wind_sensor_state = torch.zeros((n_envs, NUM_SWARM_DRONES, 3), dtype=torch.float32, device=train_device)
    wind_sensor_bias = torch.zeros_like(wind_sensor_state)
    history = None
    obs_samples, guide_samples, fsm_samples, winch_samples, health_samples, trust_samples, value_samples, kinematics_samples, cargo_dynamics_samples, force_balance_samples = [], [], [], [], [], [], [], [], [], []
    source_env_samples = []
    completed_deliveries = 0
    released_packages = 0
    total_delivery_attempts = 0
    failed_attempts = 0
    peak_sway_observed_m = 0.0
    max_cable_len_seen_m = float(PHASE3_STOWED_CABLE_LEN_M)
    min_cargo_surface_clearance_m = float("inf")
    max_lowering_drone_xy_error_m = 0.0
    contact_diagnostics = {
        key: torch.zeros((), dtype=torch.int64, device=train_device)
        for key in ("lowering_frames", "aligned_frames", "cargo_xy_ok_frames",
                    "cargo_xy_fail_frames", "cargo_z_ok_frames", "sway_over_limit_frames",
                    "contact_candidates", "touchdown_frames", "touchdown_valid_frames",
                    "projected_contact_frames", "drone_runaway_frames", "runaway_aborts",
                    "cable_reach_ok_frames", "cable_reach_fail_frames",
                    "cable_reach_blocked_at_max_frames",
                    "drone_gate_fail_frames", "cargo_xy_projection_fail_frames",
                    "wind_gate_fail_frames", "payout_paused_frames",
                    "cargo_safe_zone_frames", "cargo_zone_fail_frames", "cargo_zone_aborts")
    }
    sample_interval = 16
    try:
        critical_sample_stride = int(os.environ.get("SURVIVAL_PHASE3_CRITICAL_SAMPLE_STRIDE", "8"))
    except (TypeError, ValueError):
        critical_sample_stride = 8
    critical_sample_stride = int(np.clip(critical_sample_stride, 2, 64))
    sample_count_collected = 0
    rng = np.random.default_rng()
    model.to(train_device)
    model.eval()

    print("\n" + "=" * 75, flush=True)
    print(f"PHASE 3: GENESIS 3D CURRICULUM REFINEMENT — {n_envs:,} ENVS / {n_envs * NUM_SWARM_DRONES:,} DRONES", flush=True)
    print(f"Delivery gate: {DELIVERY_DRONE_CENTER_TOLERANCE_M:.2f} m XY, {DELIVERY_HOVER_ALTITUDE_TOLERANCE_M:.2f} m altitude, "
          f"{DELIVERY_MAX_HORIZONTAL_SPEED_MS:.2f} m/s horizontal, {DELIVERY_MAX_VERTICAL_SPEED_MS:.2f} m/s vertical, "
          f"{DELIVERY_HOLD_MIN_S:.2f} s continuous dwell", flush=True)
    print(f"Rollout steps: {steps:,} | dt={dt:.3f}s | fine-tuning epochs: {epochs} | "
          f"teacher-forced winch clones: {teacher_winch_ratio:.0%} | "
          f"low-wind demo envs: {int(curriculum_demo_env_mask.sum().item())}/{n_envs} | device={train_device}", flush=True)
    print("=" * 75, flush=True)

    with torch.no_grad():
        for step in range(steps):
            pos_list, vel_list, quat_list, omega_world_list, cargo_pos_list, cargo_vel_list = [], [], [], [], [], []
            for entity in drone_entities:
                pos_list.append(torch.as_tensor(entity.get_pos(), device=train_device, dtype=torch.float32))
                vel_list.append(torch.as_tensor(entity.get_vel(), device=train_device, dtype=torch.float32))
                try:
                    quat_list.append(torch.as_tensor(entity.get_quat(), device=train_device, dtype=torch.float32))
                    omega_world_list.append(torch.as_tensor(entity.get_ang(), device=train_device, dtype=torch.float32))
                except (AttributeError, TypeError) as exc:
                    raise RuntimeError(
                        "Phase 3 needs Genesis entity.get_quat() and entity.get_ang() "
                        "to preserve observation features 9:21; refusing to silently train on zeroed attitude features."
                    ) from exc
            for entity in cargo_entities:
                cargo_pos_list.append(torch.as_tensor(entity.get_pos(), device=train_device, dtype=torch.float32))
                cargo_vel_list.append(torch.as_tensor(entity.get_vel(), device=train_device, dtype=torch.float32))
            pos = torch.stack(pos_list, dim=1)
            vel = torch.stack(vel_list, dim=1)
            quat_wxyz = torch.stack(quat_list, dim=1)
            rotation_matrix = _phase3_torch_rotation_from_quaternion(quat_wxyz)
            omega_world = torch.stack(omega_world_list, dim=1)
            # Genesis exposes angular velocity in world coordinates; the Phase-1/2
            # gyro contract is body-frame angular velocity, so rotate it into body axes.
            omega_body = torch.matmul(rotation_matrix.transpose(-1, -2), omega_world.unsqueeze(-1)).squeeze(-1)
            cargo_pos = torch.stack(cargo_pos_list, dim=1)
            cargo_vel = torch.stack(cargo_vel_list, dim=1)
            target_roofs = target_table[target_idx]
            waypoints = _phase3_waypoints(pos, homes_one, target_roofs, phase, cruise_z, descent_armed)
            if step % 256 == 0:
                wind_target = _phase3_sample_wind_targets(n_envs, train_device)
                calm_target = torch.randn((n_envs, 3), device=train_device) * calm_target_sigma
                wind_target = torch.where(curriculum_demo_env_mask[:, None], calm_target, wind_target)
            # Correlated wind transitions prevent discontinuous, one-frame impulses.
            # Each clone sees realistic regimes and stronger storms throughout Phase 3.
            wind_alpha = min(1.0, dt / 0.25)
            wind_state = wind_state + wind_alpha * (wind_target - wind_state)
            gust_sigma = torch.tensor([0.36, 0.36, 0.12], device=train_device)
            gust_scale = torch.where(
                curriculum_demo_env_mask,
                torch.full((n_envs,), 0.12, dtype=torch.float32, device=train_device),
                torch.ones((n_envs,), dtype=torch.float32, device=train_device))
            wind_gust = (0.94 * wind_gust
                         + torch.randn_like(wind_gust) * gust_sigma * gust_scale[:, None])
            wind = (wind_state + wind_gust).clone()
            wind[:, :2] = wind[:, :2].clamp(-10.5, 10.5)
            wind[:, 2] = wind[:, 2].clamp(-3.5, 1.2)
            # Sensor model: noisy GNSS/INS ground velocity minus noisy 3-D air-relative
            # velocity from a multi-hole pitot probe. Bias drifts slowly and the output
            # is low-pass filtered. `wind` remains the actual disturbance in physics.
            ground_velocity_meas = vel + torch.randn_like(vel) * 0.045
            air_relative_true = vel - wind[:, None, :]
            wind_sensor_bias = wind_sensor_bias + torch.randn_like(wind_sensor_bias) * (0.0018 * math.sqrt(max(dt, 1e-4) / 0.016))
            air_relative_meas = air_relative_true + wind_sensor_bias + torch.randn_like(vel) * 0.14
            wind_measurement = ground_velocity_meas - air_relative_meas
            wind_measurement[..., :2] = wind_measurement[..., :2].clamp(-12.0, 12.0)
            wind_measurement[..., 2] = wind_measurement[..., 2].clamp(-4.0, 2.0)
            sensor_alpha = min(1.0, dt / (0.16 + dt))
            if step == 0:
                # A real sensor supplies a measurement immediately; do not train the
                # policy on an artificial zero-wind startup transient.
                wind_sensor_state = wind_measurement.clone()
            else:
                wind_sensor_state = wind_sensor_state + sensor_alpha * (wind_measurement - wind_sensor_state)
            wind_sensed = wind_sensor_state.clone()
            # In this reduced batched physics model, cargo-to-airframe relative
            # position/velocity are the observable cable sway estimates. They are
            # independent for every environment/drone and are zero when no attached
            # package is being carried on an extended cable.
            active_hook = has_payload & (cable_len > 0.10)
            cargo_active_before = active_hook.clone()
            sway_2d = cargo_pos[..., :2] - pos[..., :2]
            sway_velocity_2d = cargo_vel[..., :2] - vel[..., :2]
            sway_2d = torch.where(active_hook[..., None], sway_2d, torch.zeros_like(sway_2d))
            sway_velocity_2d = torch.where(active_hook[..., None], sway_velocity_2d,
                                           torch.zeros_like(sway_velocity_2d))
            if sway_2d.numel():
                peak_sway_observed_m = max(peak_sway_observed_m,
                                           float(torch.linalg.vector_norm(sway_2d, dim=-1).max().item()))
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
            instant = _phase3_torch_features(
                pos, vel, waypoints, target_roofs[..., 2], cable_len, cable_vel,
                has_payload, winch_state, phase, wind_sensed, prev_accel, aabb_tensor,
                sway_2d=sway_2d, sway_velocity_2d=sway_velocity_2d, tension_n=phase3_tension,
                rotation_matrix=rotation_matrix, omega_body=omega_body)
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
            # Teacher-generated targets for on-policy fine-tuning. The live policy
            # still affects guidance and winch transitions during this rollout.
            delivery_mask = phase == 3
            station_stable = _phase3_delivery_stability_mask(pos, vel, target_roofs, sway_2d, phase)
            active_waiting_for_drop = delivery_mask & (winch_state == 0) & has_payload
            hover_settle_timer = torch.where(
                active_waiting_for_drop & station_stable,
                hover_settle_timer + dt,
                torch.where(active_waiting_for_drop, torch.zeros_like(hover_settle_timer), hover_settle_timer),
            )
            hover_settle_timer = torch.where(delivery_mask, hover_settle_timer, torch.zeros_like(hover_settle_timer))
            # Training deploy/payout gate mirrors the live controller: new deployment
            # and continued downward payout are reserved for a calm-enough window.
            wind_delivery_safe = (torch.linalg.vector_norm(wind_sensed[..., :2], dim=-1)
                                  <= DELIVERY_MAX_DEPLOYMENT_WIND_MS)
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
                # Distil the existing Phase-2 auxiliary heads on the Phase-3
                # state distribution; don't synthesize health/trust targets.
                health_samples.append(p_health.reshape(-1, 8)[sample_mask_flat].detach().float().cpu().numpy())
                trust_samples.append(p_trust.reshape(-1, 2)[sample_mask_flat].detach().float().cpu().numpy())
                value_samples.append(p_value.reshape(-1, 1)[sample_mask_flat].detach().float().cpu().numpy())
                sample_count_collected += int(obs_samples[-1].shape[0])
                env_ids_grid = torch.arange(n_envs, device=train_device, dtype=torch.int64)[:, None].expand(-1, NUM_SWARM_DRONES)
                source_env_samples.append(env_ids_grid[sample_mask].detach().cpu().numpy().astype(np.int64, copy=False))

            policy_tokens = p_fsm.argmax(dim=-1).reshape(n_envs, NUM_SWARM_DRONES)
            policy_guidance = p_guidance.reshape(n_envs, NUM_SWARM_DRONES, 4)[..., :3]
            policy_winch = p_winch.reshape(n_envs, NUM_SWARM_DRONES, 2)
            # Guidance residual is deliberately bounded; expert feedback supplies
            # the stable nominal trajectory while the policy influences dynamics.
            policy_blended_accel = 0.82 * teacher_accel + 0.18 * (12.0 * policy_guidance)
            # In the designated teacher-forced delivery clones, retain the expert
            # station-keeping law all the way through lower/contact/retract. The
            # other half remains policy-influenced, preserving on-policy diversity.
            # During delivery, a physically valid teacher controls the simplified
            # aircraft while the policy continues to learn from the observed states.
            # Letting an immature policy perturb the drone after lowering starts was
            # responsible for repeated geometry aborts despite a valid cargo footprint.
            expert_delivery = (phase == 3)
            accel_cmd = torch.where(expert_delivery[..., None], teacher_accel, policy_blended_accel)
            accel_cmd[..., :2] = torch.clamp(accel_cmd[..., :2], -8.0, 8.0)
            accel_cmd[..., 2] = torch.clamp(accel_cmd[..., 2], -5.0, 6.0)

            # Policy-controlled winch requests, with physical contact and timeout
            # guards. State transitions are vectorised across every environment.
            delivery_mask = phase == 3
            action_deploy = (policy_tokens == 2) | (teacher_winch_mask & settled)
            # Teacher-forced clones demonstrate the complete safe delivery sequence;
            # non-forced clones use the policy's explicit WINCH_LOWER / RETRACT tokens.
            # Physical alignment, contact and timeout gates override both paths.
            policy_abort = (policy_tokens == 4) | (policy_tokens == 5) | (policy_tokens == 6)
            # A not-yet-trained action head must not abort a physically valid drop.
            # During Phase 3, only geometric loss, physical contact failure, or the
            # episode timeout may cause a retract. The policy learns retract labels
            # from resulting states, rather than controlling this safety-critical gate.
            # `lowering` is defined below, so it cannot be used to allocate this
            # mask yet. Use the already-defined delivery mask, which has the same
            # [n_envs, NUM_SWARM_DRONES] shape and boolean dtype. Phase 3 deliberately
            # disables policy-requested aborts; physical safety gates below remain active.
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
            if lowering.any():
                max_cable_len_seen_m = max(max_cable_len_seen_m, float(cable_len[lowering].max().item()))
                min_cargo_surface_clearance_m = min(
                    min_cargo_surface_clearance_m,
                    float((cargo_pos[..., 2][lowering] - target_roofs[..., 2][lowering]
                           - PHASE3_CARGO_HALF_HEIGHT_M).min().item()))
                max_lowering_drone_xy_error_m = max(
                    max_lowering_drone_xy_error_m,
                    float(torch.linalg.vector_norm(pos[..., :2] - target_roofs[..., :2], dim=-1)[lowering].max().item()))
            cargo_touch_envelope = _phase3_cargo_contact_mask(cargo_pos, target_roofs)
            # Use the same distinct checks as the live winch: drone pose/velocity
            # must be safe, and the cargo footprint must remain inside the drop zone.
            # Swing is not itself an abort trigger if the package remains in-bounds;
            # it is already constrained during the pre-lowering station-settle dwell.
            drone_alignment_ok = _phase3_drone_delivery_alignment_mask(pos, vel, target_roofs)
            cargo_xy_ok = cargo_xy_error <= DELIVERY_CARGO_CENTER_TOLERANCE_M
            cargo_xy_safe = cargo_xy_error <= PHASE3_CARGO_SAFE_RADIUS_M
            sway_over_limit = torch.linalg.vector_norm(sway_2d, dim=-1) > DELIVERY_MAX_SWAY_M
            delivery_geometry_ok = drone_alignment_ok & cargo_xy_ok & (~sway_over_limit)
            contact_diagnostics["cargo_safe_zone_frames"] += (lowering & cargo_xy_safe).sum()
            contact_diagnostics["cargo_zone_fail_frames"] += (lowering & (~cargo_xy_safe)).sum()

            # Instrument the actual projection gates. In V37.3.2 the counter
            # cable_reach_ok_frames was allocated and printed but never updated,
            # so its reported zero was not evidence that reach was physically false.
            contact_center_for_diag = target_roofs.clone()
            contact_center_for_diag[..., 2] += PHASE3_CARGO_HALF_HEIGHT_M
            required_cable_for_diag = torch.clamp(
                torch.linalg.vector_norm(pos - contact_center_for_diag, dim=-1)
                - PHASE3_TETHER_CENTER_OFFSET_M, min=0.0)
            cable_reach_ok_diag = (cable_len + 0.035) >= required_cable_for_diag
            wind_ok_for_payout = wind_delivery_safe
            contact_diagnostics["cable_reach_ok_frames"] += (lowering & cable_reach_ok_diag).sum()
            # A fail during ordinary payout is expected until the cable grows long
            # enough. Count a separate failure only when payout has reached its
            # physical 12 m limit and the target still cannot be reached.
            cable_at_max = cable_len >= (12.0 - 1e-4)
            contact_diagnostics["cable_reach_fail_frames"] += (lowering & (~cable_reach_ok_diag)).sum()
            contact_diagnostics["cable_reach_blocked_at_max_frames"] += (
                lowering & (~cable_reach_ok_diag) & cable_at_max).sum()
            contact_diagnostics["drone_gate_fail_frames"] += (lowering & (~drone_alignment_ok)).sum()
            contact_diagnostics["cargo_xy_projection_fail_frames"] += (lowering & (~cargo_xy_ok)).sum()
            contact_diagnostics["wind_gate_fail_frames"] += (lowering & (~wind_ok_for_payout)).sum()
            # Once a deployment has been safely armed, small transient drone-position
            # errors do not instantly abort the whole drop if the parcel remains over
            # the legal zone. Keep strict cargo XY and physical contact checks.
            # Abort if the parcel itself leaves the drop zone or the drone is severely
            # outside the local safe envelope (>2x placement tolerance).
            drone_xy_error_now = torch.linalg.vector_norm(pos[..., :2] - target_roofs[..., :2], dim=-1)
            drone_xy_runaway = drone_xy_error_now > PHASE3_DRONE_RUNAWAY_ABORT_DISTANCE_M
            # Do not abort a long-cable drop on one transient frame of drone position
            # error. Let the expert station-keeping controller recover for 0.60 s;
            # cargo leaving the legal drop zone remains an immediate abort. Release
            # still requires the strict live alignment/velocity gate.
            alignment_loss_timer = torch.where(
                lowering & drone_xy_runaway,
                alignment_loss_timer + dt,
                torch.zeros_like(alignment_loss_timer),
            )
            runaway_abort = drone_xy_runaway & (alignment_loss_timer >= PHASE3_DRONE_RUNAWAY_ABORT_DWELL_S)
            cargo_zone_loss_timer = torch.where(
                lowering & (~cargo_xy_safe), cargo_zone_loss_timer + dt,
                torch.zeros_like(cargo_zone_loss_timer))
            cargo_zone_abort = (lowering & (~cargo_xy_safe)
                                & (cargo_zone_loss_timer >= PHASE3_CARGO_ZONE_ABORT_DWELL_S))
            contact_diagnostics["cargo_zone_aborts"] += cargo_zone_abort.sum()
            # Strict centring is required for contact/release, not for continuing to
            # hold an attached package safely. Brief swing outside the release radius
            # pauses payout and allows anti-sway feedback to recover it; only a sustained
            # excursion beyond the physical pad safety radius aborts the attempt.
            alignment_lost = lowering & (cargo_zone_abort | runaway_abort)
            contact_diagnostics["lowering_frames"] += lowering.sum()
            contact_diagnostics["aligned_frames"] += (lowering & drone_alignment_ok).sum()
            contact_diagnostics["cargo_xy_ok_frames"] += (lowering & cargo_xy_ok).sum()
            contact_diagnostics["cargo_xy_fail_frames"] += (lowering & (~cargo_xy_ok)).sum()
            contact_diagnostics["cargo_z_ok_frames"] += (lowering & (torch.abs(
                cargo_pos[..., 2] - (target_roofs[..., 2] + PHASE3_CARGO_HALF_HEIGHT_M)) <= 0.03)).sum()
            contact_diagnostics["sway_over_limit_frames"] += (lowering & sway_over_limit).sum()
            contact_diagnostics["drone_runaway_frames"] += (lowering & drone_xy_runaway).sum()
            contact_diagnostics["runaway_aborts"] += (lowering & runaway_abort).sum()
            if alignment_lost.any():
                failed_attempts += int(alignment_lost.sum().item())
            winch_state = torch.where(alignment_lost, torch.full_like(winch_state, 3), winch_state)
            hover_settle_timer = torch.where(alignment_lost, torch.zeros_like(hover_settle_timer), hover_settle_timer)
            lowering = delivery_mask & (winch_state == 1) & has_payload
            # Contact only counts when the payload is actually over the pad and
            # the aircraft still satisfies the live position/velocity envelope.
            touch = delivery_mask & lowering & delivery_geometry_ok & cargo_touch_envelope
            contact_diagnostics["contact_candidates"] += touch.sum()
            winch_state = torch.where(touch, torch.full_like(winch_state, 2), winch_state)
            lowering = delivery_mask & (winch_state == 1) & has_payload
            learned_down_rate = 1.2 + 1.2 * torch.clamp(policy_winch[..., 0], 0.0, 1.0)
            # On teacher-forced clones, use a controlled physical rate instead of
            # allowing an untrained winch head to starve/overshoot the drop sequence.
            down_rate = torch.where(teacher_winch_mask,
                                    torch.full_like(learned_down_rate, 1.8),
                                    learned_down_rate)
            # Do not continue paying out cable while the aircraft is outside the
            # same alignment/velocity envelope required for physical contact, or
            # while wind exceeds the configured operational limit. V37.3.2 kept
            # extending the cable in those states even though contact projection
            # was forbidden by the drone safety gate; this let the payload swing
            # while making successful contact progressively less likely.
            payout_active = (lowering & drone_alignment_ok & cargo_xy_ok
                             & (~sway_over_limit) & wind_delivery_safe)
            payout_paused = lowering & (~(drone_alignment_ok & cargo_xy_ok
                                          & (~sway_over_limit) & wind_delivery_safe))
            contact_diagnostics["payout_paused_frames"] += payout_paused.sum()
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
            contact_diagnostics["touchdown_frames"] += touching.sum()
            contact_diagnostics["touchdown_valid_frames"] += contact_still_valid.sum()
            touchdown_timer = torch.where(contact_still_valid, touchdown_timer + dt,
                                          torch.zeros_like(touchdown_timer))
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
            retract_speed = torch.where(teacher_winch_mask,
                                        torch.full_like(learned_retract_speed, 2.4),
                                        learned_retract_speed)
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

            # A taut cable transfers a fraction of the payload load to the drone.
            # Estimate that fraction from simulated tension, rather than treating a
            # slack payload as if its full weight were already on the airframe.
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
            # The Genesis training URDF has fixed mass DRY_MASS_KG; force scaling
            # must use that same body mass. The tether force is applied explicitly
            # below and is cancelled in the teacher feed-forward, avoiding the old
            # effective-mass mismatch that caused large unmodelled acceleration.
            body_mass = torch.full_like(tension, float(DRY_MASS_KG))
            force = body_mass[..., None] * accel_cmd
            force[..., 2] += body_mass * G_ACCEL
            # Apply aerodynamic drag/disturbance to both physical rigid bodies so
            # the persistent randomized wind changes the actual Genesis trajectories.
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

            # Teacher-owned navigation state transitions. Policy action/continuous
            # guidance influence the physical route and delivery attempt; the expert
            # mission manager still prevents arbitrary state skipping.
            new_pos, new_vel, new_cargo_pos, new_cargo_vel = [], [], [], []
            for entity in drone_entities:
                new_pos.append(torch.as_tensor(entity.get_pos(), device=train_device, dtype=torch.float32))
                new_vel.append(torch.as_tensor(entity.get_vel(), device=train_device, dtype=torch.float32))
            for entity in cargo_entities:
                new_cargo_pos.append(torch.as_tensor(entity.get_pos(), device=train_device, dtype=torch.float32))
                new_cargo_vel.append(torch.as_tensor(entity.get_vel(), device=train_device, dtype=torch.float32))
            pos_after = torch.stack(new_pos, dim=1)
            vel_after = torch.stack(new_vel, dim=1)
            cargo_pos_after = torch.stack(new_cargo_pos, dim=1)
            cargo_vel_after = torch.stack(new_cargo_vel, dim=1)
            # Pair the pre-step observation with the actual post-step Genesis transition.
            # Position delta estimates average world velocity over dt; velocity delta
            # provides measured acceleration. No hidden action labels are fabricated.
            avg_velocity_target = (pos_after - pos) / max(dt, 1e-6) / 8.0
            acceleration_target = (vel_after - vel) / max(dt, 1e-6) / 12.0
            transition_valid = (torch.isfinite(pos_after).all(dim=-1)
                                & torch.isfinite(vel_after).all(dim=-1)
                                & (pos_after[..., 2] >= -5.0) & (pos_after[..., 2] <= 80.0)
                                & (torch.abs(vel_after[..., 2]) <= 8.0)
                                & (torch.linalg.vector_norm(vel_after, dim=-1) <= 45.0)
                                & (torch.linalg.vector_norm((vel_after - vel) / max(dt, 1e-6), dim=-1) <= 60.0))
            transition_target = torch.cat((avg_velocity_target.clamp(-2.0, 2.0),
                                           acceleration_target.clamp(-3.0, 3.0),
                                           transition_valid.to(torch.float32).unsqueeze(-1)), dim=-1)
            if sample_mask.any():
                transition_flat = transition_target.reshape(-1, 7)
                kinematics_samples.append(transition_flat[sample_mask.reshape(-1)].detach().float().cpu().numpy())

                # Newtonian force-balance residual audit: subtract the known command,
                # drag, and tether force from measured Genesis acceleration. Only
                # free-flight/airborne rows are eligible; depot/ground and touchdown
                # transitions are masked rather than interpreted as smooth forces.
                measured_accel = (vel_after - vel) / max(dt, 1e-6)
                contact_free_force_mask = ((pos[..., 2] > FORCE_BALANCE_MIN_AIRBORNE_ALTITUDE_M)
                                           & (phase < 5) & (winch_state != 2))
                force_balance_target = _torch_force_balance_residual_target(
                    measured_accel, accel_cmd, drone_drag + tether_force_drone, body_mass,
                    transition_valid, contact_free_force_mask)
                force_balance_flat = force_balance_target.reshape(-1, 4)
                force_balance_samples.append(
                    force_balance_flat[sample_mask.reshape(-1)].detach().float().cpu().numpy())

                # Predict measured next-step cargo state relative to the airframe. The
                # target is the physical Genesis post-step state before contact projection;
                # release/stowed/empty-hook rows are masked rather than treated as zero swing.
                cargo_sway_after = cargo_pos_after[..., :2] - pos_after[..., :2]
                cargo_sway_rate_after = cargo_vel_after[..., :2] - vel_after[..., :2]
                post_tether_vec = cargo_pos_after - pos_after
                post_tether_dist = torch.linalg.vector_norm(post_tether_vec, dim=-1)
                post_slack = _phase3_tether_slack(post_tether_dist, cable_len)
                post_relative_velocity = cargo_vel_after - vel_after
                post_tether_dir = post_tether_vec / torch.clamp(post_tether_dist[..., None], min=1e-4)
                post_radial_damping = torch.sum(post_relative_velocity * post_tether_dir, dim=-1)
                cargo_tension_after = torch.where(
                    has_payload & (winch_state != 2) & (post_slack > 0.0),
                    torch.clamp(post_slack * 100.0 + post_radial_damping * 4.0, min=0.0, max=18.0),
                    torch.zeros_like(post_slack))
                cargo_transition_valid = (cargo_active_before & has_payload & (cable_len > 0.10)
                    & torch.isfinite(cargo_sway_after).all(dim=-1)
                    & torch.isfinite(cargo_sway_rate_after).all(dim=-1)
                    & torch.isfinite(cargo_tension_after)
                    & (torch.linalg.vector_norm(cargo_sway_after, dim=-1) <= 10.0)
                    & (torch.linalg.vector_norm(cargo_sway_rate_after, dim=-1) <= 3.0 * CARGO_DYNAMICS_SWAY_RATE_SCALE_MS)
                    & (cargo_tension_after >= 0.0) & (cargo_tension_after <= 45.0)
                    & (cable_len <= 12.5))
                cargo_target = torch.cat((
                    (cargo_sway_after / CARGO_DYNAMICS_SWAY_SCALE_M).clamp(-2.0, 2.0),
                    (cargo_sway_rate_after / CARGO_DYNAMICS_SWAY_RATE_SCALE_MS).clamp(-3.0, 3.0),
                    (cargo_tension_after / CARGO_DYNAMICS_TENSION_SCALE_N).clamp(0.0, 1.0).unsqueeze(-1),
                    cargo_transition_valid.to(torch.float32).unsqueeze(-1)), dim=-1)
                cargo_target_flat = cargo_target.reshape(-1, 6)
                cargo_dynamics_samples.append(
                    cargo_target_flat[sample_mask.reshape(-1)].detach().float().cpu().numpy())
            target_roofs = target_table[target_idx]

            # Position-based tether constraint and contact persistence. Without this,
            # the simplified spring lets the cable go slack during payout and cargo
            # never reaches the pad even though the winch reports lowering.
            cargo_pos_corrected, contact_projection = _phase3_project_tether_and_contact(
                pos_after, vel_after, cargo_pos_after, cable_len, has_payload,
                winch_state, phase, target_roofs, released_package_valid, released_package_pos)
            contact_diagnostics["projected_contact_frames"] += contact_projection.sum()
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
                # A new package is placed on each depot pad and the drone is
                # repositioned to its original loaded-departure state in reset envs.
                for d_idx, entity in enumerate(drone_entities):
                    reset_envs = torch.nonzero(reset[:, d_idx], as_tuple=False).squeeze(-1).detach().cpu().numpy()
                    if len(reset_envs) == 0:
                        continue
                    reset_home = homes_one[d_idx].clone()
                    reset_home[2] = 0.70
                    reset_pos = reset_home.unsqueeze(0).expand(len(reset_envs), -1).clone()
                    # Crucial: envs_idx prevents set_pos from zeroing velocities in
                    # every non-reset environment (Genesis zeroes velocity by default).
                    entity.set_pos(reset_pos, envs_idx=reset_envs, zero_velocity=True)
                for d_idx, entity in enumerate(cargo_entities):
                    reset_envs = torch.nonzero(reset[:, d_idx], as_tuple=False).squeeze(-1).detach().cpu().numpy()
                    if len(reset_envs) == 0:
                        continue
                    home_package = homes_one[d_idx].clone()
                    home_package[2] = _phase3_cargo_center_z_for_cable(
                        torch.tensor(0.70, device=train_device),
                        torch.tensor(PHASE3_STOWED_CABLE_LEN_M, device=train_device))
                    reset_cargo = home_package.unsqueeze(0).expand(len(reset_envs), -1).clone()
                    entity.set_pos(reset_cargo, envs_idx=reset_envs, zero_velocity=True)
                # New episode = new temporal context and no previous-episode acceleration.
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

            if (step + 1) % 512 == 0:
                waypoint_dist = torch.linalg.vector_norm(waypoints[..., :2] - pos[..., :2], dim=-1)
                mean_dist = waypoint_dist.mean().item()
                phase_counts = [int((phase == k).sum().item()) for k in range(7)]
                delivery_dist_mask = (phase == 2) | (phase == 3)
                delivery_dist = (waypoint_dist[delivery_dist_mask].mean().item()
                                 if delivery_dist_mask.any() else float("nan"))
                phase_summary = "/".join(str(v) for v in phase_counts)
                delivery_dist_text = f"{delivery_dist:.2f}m" if math.isfinite(delivery_dist) else "n/a"
                print(f"[PHASE 3 ROLLOUT] {step+1:4d}/{steps} steps | mean XY waypoint distance all={mean_dist:6.1f} m, delivery-phases={delivery_dist_text} | phase counts 0..6={phase_summary} | releases={released_packages} completed={completed_deliveries} failed={failed_attempts}", flush=True)

    if not obs_samples:
        raise RuntimeError("Phase 3 produced no observations; increase SURVIVAL_PHASE3_STEPS.")

    # The rollout is complete: release Genesis native scene/entities before large
    # host-memory dataset consolidation and fine-tuning. Keeping the scene alive
    # alongside multi-gigabyte feature buffers needlessly raises peak commit usage.
    try:
        del entity
    except UnboundLocalError:
        pass
    del scene, trial, drone_entities, cargo_entities, targets
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            # A CUDA allocator/backend error after the rollout must not prevent
            # CPU-side dataset consolidation and checkpoint-safe fine-tuning.
            print(f"[PHASE 3 MEMORY] CUDA cache release warning: {type(exc).__name__}: {exc}", flush=True)
    print("[PHASE 3 MEMORY] Released Genesis scene/entities before dataset consolidation.", flush=True)

    # Concatenate one target family at a time and immediately release its source
    # chunks. Releasing all seven lists only after seven concatenations temporarily
    # duplicates most of the dataset in Windows commit memory.
    X = np.concatenate(obs_samples, axis=0).astype(np.float32, copy=False)
    del obs_samples
    gc.collect()
    source_env_ids = np.concatenate(source_env_samples, axis=0).astype(np.int64, copy=False)
    del source_env_samples
    if len(source_env_ids) != len(X):
        raise RuntimeError(f"DAgger source-group alignment failure: observations={len(X):,}, env_ids={len(source_env_ids):,}.")
    Yg = np.concatenate(guide_samples, axis=0).astype(np.float32, copy=False)
    del guide_samples
    gc.collect()
    Yf = np.concatenate(fsm_samples, axis=0).astype(np.int64, copy=False)
    del fsm_samples
    gc.collect()
    Yw = np.concatenate(winch_samples, axis=0).astype(np.float32, copy=False)
    del winch_samples
    gc.collect()
    Yh = np.concatenate(health_samples, axis=0).astype(np.float32, copy=False)
    del health_samples
    gc.collect()
    Yt = np.concatenate(trust_samples, axis=0).astype(np.float32, copy=False)
    del trust_samples
    gc.collect()
    Yv = np.concatenate(value_samples, axis=0).astype(np.float32, copy=False)
    del value_samples
    gc.collect()
    if not kinematics_samples:
        raise RuntimeError("Phase 3 produced no paired one-step kinematics targets; refusing physics-guided refinement.")
    Ykin = np.concatenate(kinematics_samples, axis=0).astype(np.float32, copy=False)
    del kinematics_samples
    gc.collect()
    if not cargo_dynamics_samples:
        raise RuntimeError("Phase 3 produced no paired cargo-dynamics targets; refusing physics-guided refinement.")
    Ycargo = np.concatenate(cargo_dynamics_samples, axis=0).astype(np.float32, copy=False)
    del cargo_dynamics_samples
    gc.collect()
    if not force_balance_samples:
        raise RuntimeError("Phase 3 produced no paired force-balance targets; refusing physics-guided refinement.")
    Yforce = np.concatenate(force_balance_samples, axis=0).astype(np.float32, copy=False)
    del force_balance_samples
    gc.collect()
    if not (len(X) == len(Ykin) == len(Ycargo) == len(Yforce)):
        raise RuntimeError(
            f"Phase 3 transition alignment failure: observations={len(X):,}, kinematics_targets={len(Ykin):,}, "
            f"cargo_targets={len(Ycargo):,}, force_targets={len(Yforce):,}.")
    valid_kin_count = int((Ykin[:, 6] > 0.5).sum())
    valid_cargo_count = int((Ycargo[:, 5] > 0.5).sum())
    valid_force_count = int((Yforce[:, 3] > 0.5).sum())
    print(f"[PHYSICS-GUIDED DATA] Phase 3 measured transitions valid={valid_kin_count:,}/{len(Ykin):,}.", flush=True)
    print(f"[CARGO-DYNAMICS DATA] Phase 3 loaded extended-cable targets valid={valid_cargo_count:,}/{len(Ycargo):,}.", flush=True)
    if valid_force_count:
        force_residual_values_n = Yforce[Yforce[:, 3] > 0.5, :3] * FORCE_BALANCE_RESIDUAL_SCALE_N
        force_residual_norms_n = np.linalg.norm(force_residual_values_n, axis=1)
        baseline_force_rmse_n = float(np.sqrt(np.mean(np.square(force_residual_norms_n))))
        baseline_force_p95_n = float(np.percentile(force_residual_norms_n, 95.0))
        force_bias_n = np.mean(force_residual_values_n, axis=0)
        print(
            f"[FORCE-BALANCE AUDIT] free-flight residual rows={valid_force_count:,}/{len(Yforce):,} | "
            f"baseline vector-RMSE={baseline_force_rmse_n:.4f}N "
            f"({baseline_force_rmse_n / max(DRY_MASS_KG, 1e-6):.4f}m/s^2 equivalent) | "
            f"P95 norm={baseline_force_p95_n:.4f}N | "
            f"bias xyz=({force_bias_n[0]:+.4f},{force_bias_n[1]:+.4f},{force_bias_n[2]:+.4f})N",
            flush=True)
        if baseline_force_rmse_n < 0.05:
            print("[FORCE-BALANCE AUDIT NOTE] Residual target is nearly zero on this free-flight sample; the model is already force-consistent under the Phase-3 equation, so meaningful correction improvement may be small.", flush=True)
    else:
        baseline_force_rmse_n = float("nan")
        print("[FORCE-BALANCE AUDIT WARNING] No valid free-flight force residuals were collected.", flush=True)
    if valid_cargo_count < 128:
        raise RuntimeError(
            f"Phase 3 produced only {valid_cargo_count} valid cargo-dynamics transitions; "
            "refusing to save a checkpoint that falsely marks the cargo predictor as trained.")
    if valid_force_count < 128:
        raise RuntimeError(
            f"Phase 3 produced only {valid_force_count} valid free-flight force-balance transitions; "
            "refusing to mark the force-residual head as trained. Check free-flight/contact masking and sample coverage.")

    # Genesis GPU simulation and PyTorch backward used the same CUDA context in
    # V38.9. The reported allocator failure occurred exactly at loss.backward().
    # Use CPU fine-tuning by default after a CUDA-backed rollout; users may opt in
    # to CUDA explicitly once their Genesis/PyTorch memory setup is known stable.
    train_device = torch.device(requested_train_device)
    current_model_device = next(model.parameters()).device
    if current_model_device != train_device:
        del model
        gc.collect()
        model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(train_device)
        model.load_state_dict({name: tensor.to(train_device) for name, tensor in phase3_start_state_cpu.items()})
    model.eval()
    print(f"[PHASE 3 TRAIN DEVICE] supervised fine-tuning device={train_device}; "
          f"rollout device={source_device}.", flush=True)
    phase3_dataset_bytes = sum(array.nbytes for array in (X, Yg, Yf, Yw, Yh, Yt, Yv, Ykin, Ycargo, Yforce))
    print(f"[PHASE 3 MEMORY] retained_dataset={phase3_dataset_bytes / (1024**2):.1f} MiB "
          f"samples={len(X):,} batch_size={batch_size}; source chunk buffers released.", flush=True)
    if len(X) < 128:
        raise RuntimeError(f"Phase 3 generated only {len(X)} samples; refusing to save a weak checkpoint.")
    print(f"[PHASE 3 DATASET] {len(X):,} on-policy Genesis states retained from {n_envs:,} parallel environments "
          f"(sample_interval={sample_interval}, critical_stride={critical_sample_stride}, collected={sample_count_collected:,}).", flush=True)
    phase3_fsm_counts = np.bincount(Yf, minlength=NUM_EXECUTIVE_STATES)
    phase3_fsm_total = max(1, int(phase3_fsm_counts.sum()))
    phase3_fsm_summary = ", ".join(
        f"{idx}:{ACTION_NAMES[idx]}={int(count)} ({100.0*count/phase3_fsm_total:.2f}%)"
        for idx, count in enumerate(phase3_fsm_counts))
    print(f"[PHASE 3 FSM LABELS] {phase3_fsm_summary}", flush=True)
    if np.count_nonzero(phase3_fsm_counts) < 4:
        print("[PHASE 3 WARNING] Fewer than four FSM labels were captured; inspect rollout state coverage before trusting the refined checkpoint.", flush=True)
    if len(phase3_fsm_counts) > 6 and int(phase3_fsm_counts[6]) == 0:
        print("[PHASE 3 SAFETY] EMERGENCY_DIVERT has no teacher labels; hard safety gates remain authoritative and this action is not considered learned.", flush=True)
    pending_retraction = max(0, released_packages - completed_deliveries)
    print(f"[PHASE 3 OUTCOME] attempts armed={total_delivery_attempts} | packages released={released_packages} | deliveries fully retracted={completed_deliveries} | released awaiting full retraction={pending_retraction} | failed/aborted attempts={failed_attempts} | peak observed cable sway={peak_sway_observed_m:.2f} m", flush=True)
    min_clearance_text = (f"{min_cargo_surface_clearance_m:.3f}m"
                          if math.isfinite(min_cargo_surface_clearance_m) else "n/a")
    print("[PHASE 3 CONTACT GATES] " + " | ".join(
        f"{key}={int(value.item())}" for key, value in contact_diagnostics.items())
        + f" | max_cable_len={max_cable_len_seen_m:.3f}m"
        + f" | min_cargo_surface_clearance={min_clearance_text}"
        + f" | max_lowering_drone_xy_error={max_lowering_drone_xy_error_m:.3f}m", flush=True)
    if os.environ.get("SURVIVAL_PHASE3_ALLOW_UNSUCCESSFUL_ROLLOUT", "0").strip().lower() not in {"1", "true", "yes"}:
        # Keep this hard gate: a lower loss cannot turn a zero-delivery rollout into
        # a better model. The diagnostic line above now includes cable extension and
        # minimum cargo clearance so the next failure points to a measurable cause.
        if released_packages <= 0 or completed_deliveries <= 0:
            raise RuntimeError(
                "Phase 3 quality gate rejected the rollout: it produced no successfully released-and-retracted "
                "delivery. No new checkpoint will be saved. Inspect [PHASE 3 CONTACT GATES] to identify "
                "whether alignment, cargo zone, cable reach, cargo height or contact debounce blocked delivery. "
                "Set SURVIVAL_PHASE3_ALLOW_UNSUCCESSFUL_ROLLOUT=1 only for an intentional debugging run."
            )

    # Hold out complete parallel Genesis environment groups. A row-wise random split
    # leaks adjacent states from the same rollout into train and validation and can
    # make the Phase-3 holdout look artificially easy.
    try:
        fresh_train_idx, val_idx, val_groups = _phase3_group_heldout_split(source_env_ids, rng, val_fraction=0.10)
    except ValueError as split_exc:
        raise RuntimeError(f"Phase-3 group-held-out split failed: {split_exc}") from split_exc
    print(f"[PHASE 3 GROUP SPLIT] whole-env holdout: train_groups={len(np.unique(source_env_ids[fresh_train_idx])):,} "
          f"train_rows={len(fresh_train_idx):,}; validation_groups={len(val_groups):,} "
          f"validation_rows={len(val_idx):,}; no environment group crosses the split.", flush=True)

    # Prior replay contains training rows only; current-run validation remains strictly
    # held out and is never appended to replay. The corpus is opt-in-load / default-save
    # so the first upgraded run starts cleanly and later runs aggregate policy-visited data.
    replay_path = _phase3_dagger_replay_path()
    fresh_arrays = {key: value for key, value in (("X", X), ("Yg", Yg), ("Yf", Yf), ("Yw", Yw),
                    ("Yh", Yh), ("Yt", Yt), ("Yv", Yv), ("Ykin", Ykin), ("Ycargo", Ycargo), ("Yforce", Yforce))}
    replay = _phase3_load_dagger_replay(replay_path, fresh_arrays)
    fresh_group_ids = np.empty(len(X), dtype=np.int64)
    for group_offset, local_group in enumerate(np.unique(source_env_ids)):
        fresh_group_ids[source_env_ids == local_group] = group_offset
    if replay is not None:
        old_n = len(replay["X"])
        next_group_id = int(replay["source_group_ids"].max()) + 1
        fresh_group_ids += next_group_id
        for key in PHASE3_DAGGER_ARRAY_KEYS:
            fresh_arrays[key] = np.concatenate((replay[key], fresh_arrays[key]), axis=0)
        combined_groups = np.concatenate((replay["source_group_ids"], fresh_group_ids))
        train_idx = np.concatenate((np.arange(old_n, dtype=np.int64), fresh_train_idx + old_n))
        val_idx = val_idx + old_n
        X, Yg, Yf, Yw, Yh, Yt, Yv, Ykin, Ycargo, Yforce = (fresh_arrays[key] for key in PHASE3_DAGGER_ARRAY_KEYS)
        source_group_ids = combined_groups
        print(f"[DAgger REPLAY] training rows={old_n:,} prior + {len(fresh_train_idx):,} current; "
              f"validation is {len(val_idx):,} current-run rows only.", flush=True)
    else:
        train_idx = fresh_train_idx
        source_group_ids = fresh_group_ids
        print("[DAgger REPLAY] no compatible prior corpus found; starting from current-run training groups.", flush=True)
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
            bkin = torch.from_numpy(Ykin[idx]).to(train_device)
            bcargo = torch.from_numpy(Ycargo[idx]).to(train_device)
            bforce = torch.from_numpy(Yforce[idx]).to(train_device)
            optimizer.zero_grad(set_to_none=True)
            pg, pf, pw, ph, pt, pv, pkin, pcargo, pforce, plogvar = model(
                bx, return_physics=True, return_force_residual=True, return_uncertainty=True)
            loss = (0.70 * mse(pg, bg) + 0.35 * F.cross_entropy(pf, bf) +
                    0.25 * mse(pw, bw) + 0.20 * bce(ph.clamp(1e-6, 1.0-1e-6), bh) +
                    0.20 * bce(pt.clamp(1e-6, 1.0-1e-6), bt) + 0.10 * mse(pv, bv) +
                    0.20 * _kinematics_prediction_loss(pkin, bkin) +
                    0.10 * _cargo_dynamics_prediction_loss(pcargo, bcargo) +
                    KINEMATIC_EQUATION_RESIDUAL_WEIGHT * _kinematic_equation_residual_loss(
                        pkin, bx, bkin, dt=dt) +
                    FORCE_BALANCE_RESIDUAL_WEIGHT * _force_balance_residual_prediction_loss(pforce, bforce) +
                    PHYSICS_UNCERTAINTY_NLL_WEIGHT * _physics_uncertainty_phase3_loss(
                        plogvar, pkin, pcargo, pforce, bkin, bcargo, bforce))
            if not torch.isfinite(loss):
                raise RuntimeError("Phase 3 produced a non-finite loss; stopping before checkpoint save.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            batches += 1
            del bx, bg, bf, bw, bh, bt, bv, bkin, bcargo, bforce, pg, pf, pw, ph, pt, pv, pkin, pcargo, pforce, plogvar, loss
        model.eval()
        val_total, val_count = 0.0, 0
        val_kin_residual_weighted = 0.0
        val_force_error_sq_sum = 0.0
        val_force_baseline_sq_sum = 0.0
        val_force_row_count = 0
        val_uncertainty_nll_sum = 0.0
        val_uncertainty_rows = 0
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
                bkin = torch.from_numpy(Ykin[idx]).to(train_device)
                bcargo = torch.from_numpy(Ycargo[idx]).to(train_device)
                bforce = torch.from_numpy(Yforce[idx]).to(train_device)
                pg, pf, pw, ph, pt, pv, pkin, pcargo, pforce, plogvar = model(
                    bx, return_physics=True, return_force_residual=True, return_uncertainty=True)
                val_loss = (0.70 * mse(pg, bg) + 0.35 * F.cross_entropy(pf, bf)
                            + 0.25 * mse(pw, bw)
                            + 0.20 * bce(ph.clamp(1e-6, 1.0-1e-6), bh)
                            + 0.20 * bce(pt.clamp(1e-6, 1.0-1e-6), bt)
                            + 0.10 * mse(pv, bv)
                            + 0.20 * _kinematics_prediction_loss(pkin, bkin)
                            + 0.10 * _cargo_dynamics_prediction_loss(pcargo, bcargo)
                            + KINEMATIC_EQUATION_RESIDUAL_WEIGHT * _kinematic_equation_residual_loss(
                                pkin, bx, bkin, dt=dt)
                            + FORCE_BALANCE_RESIDUAL_WEIGHT * _force_balance_residual_prediction_loss(pforce, bforce)
                            + PHYSICS_UNCERTAINTY_NLL_WEIGHT * _physics_uncertainty_phase3_loss(
                                plogvar, pkin, pcargo, pforce, bkin, bcargo, bforce))
                val_total += float(val_loss.item()) * len(idx)
                unc_stats = [
                    _uncertainty_calibration_stats(pkin, plogvar[:, 0:6], bkin, 6),
                    _uncertainty_calibration_stats(pcargo, plogvar[:, 6:11], bcargo, 5),
                    _uncertainty_calibration_stats(pforce, plogvar[:, 11:14], bforce, 3),
                ]
                unc_valid_stats = [v for v in unc_stats if v["count"] > 0 and math.isfinite(v["nll"])]
                if unc_valid_stats:
                    val_uncertainty_nll_sum += sum(v["nll"] * v["count"] for v in unc_valid_stats)
                    val_uncertainty_rows += sum(v["count"] for v in unc_valid_stats)
                batch_residual_rmse = _kinematic_equation_residual_rmse_mps(
                    pkin, bx, bkin, dt=dt)
                if math.isfinite(batch_residual_rmse):
                    val_kin_residual_weighted += batch_residual_rmse * len(idx)
                force_error_n, force_target_n, force_valid_mask = _force_balance_residual_error_n(pforce, bforce)
                if bool(force_valid_mask.any()):
                    val_force_error_sq_sum += float(torch.sum(force_error_n[force_valid_mask] ** 2).item())
                    val_force_baseline_sq_sum += float(torch.sum(force_target_n[force_valid_mask] ** 2).item())
                    val_force_row_count += int(force_valid_mask.sum().item())
                val_count += len(idx)
                del bx, bg, bf, bw, bh, bt, bv, bkin, bcargo, bforce, pg, pf, pw, ph, pt, pv, pkin, pcargo, pforce, plogvar, val_loss, force_error_n, force_target_n, force_valid_mask
        mean_val_loss = val_total / max(1, val_count)
        if val_force_row_count:
            force_holdout_rmse_n = math.sqrt(val_force_error_sq_sum / val_force_row_count)
            force_nominal_rmse_n = math.sqrt(val_force_baseline_sq_sum / val_force_row_count)
            force_holdout_rmse_ms2 = force_holdout_rmse_n / max(DRY_MASS_KG, 1e-6)
            force_nominal_rmse_ms2 = force_nominal_rmse_n / max(DRY_MASS_KG, 1e-6)
            force_metric_text = (f"holdout force-correction RMSE {force_holdout_rmse_n:.3f}N "
                                 f"({force_holdout_rmse_ms2:.3f}m/s^2) vs uncorrected {force_nominal_rmse_n:.3f}N "
                                 f"({force_nominal_rmse_ms2:.3f}m/s^2); valid={val_force_row_count}")
        else:
            force_metric_text = "holdout force-correction RMSE n/a (no valid rows)"
        uncertainty_metric_text = (f"holdout uncertainty NLL={val_uncertainty_nll_sum/max(1,val_uncertainty_rows):.4f} "
                                   f"across {val_uncertainty_rows} valid task-rows"
                                   if val_uncertainty_rows else "holdout uncertainty NLL=n/a (no valid rows)")
        print(f"[PHASE 3 TRAIN] Epoch {epoch+1}/{epochs} — train multi-task loss {epoch_loss/max(1,batches):.5f} | holdout multi-task loss {mean_val_loss:.5f} | holdout kinematic equation residual RMSE {val_kin_residual_weighted/max(1,val_count):.4f} m/s | {force_metric_text} | {uncertainty_metric_text}", flush=True)
        if math.isfinite(mean_val_loss) and mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        model.train()

    model.load_state_dict(best_state)
    model.to(train_device)
    model.eval()
    model._kinematics_head_checkpoint_trained = True
    model._cargo_dynamics_head_checkpoint_trained = True
    model._force_balance_head_trained_state.fill_(True)
    model._force_balance_head_checkpoint_trained = True
    model._force_uncertainty_trained_state.fill_(True)
    model._force_uncertainty_checkpoint_trained = True
    print(f"[PHASE 3 VALIDATION] Restored best internal group-held-out checkpoint (multi-task loss={best_val_loss:.5f}); force-residual and force-uncertainty heads marked trained.", flush=True)
    # Persist only training membership; never serialize this run's held-out groups.
    try:
        train_arrays = {key: value[train_idx] for key, value in (("X", X), ("Yg", Yg), ("Yf", Yf),
                        ("Yw", Yw), ("Yh", Yh), ("Yt", Yt), ("Yv", Yv), ("Ykin", Ykin),
                        ("Ycargo", Ycargo), ("Yforce", Yforce))}
        _phase3_save_dagger_replay(replay_path, train_arrays, source_group_ids[train_idx])
    except (OSError, ValueError, MemoryError) as replay_exc:
        print(f"[DAgger REPLAY WARNING] model training completed but replay corpus was not updated: "
              f"{type(replay_exc).__name__}: {replay_exc}", flush=True)
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[PHASE 3 MEMORY] Final CUDA cache release warning: {type(exc).__name__}: {exc}", flush=True)
    return model


def _phase3_worker_main():
    """Subprocess entry: isolate Genesis training so the later GUI can initialise cleanly."""
    base_path = os.path.abspath(os.environ.get("SURVIVAL_PHASE3_BASE_CHECKPOINT", MODEL_WEIGHTS_FILE))
    output_path = os.path.abspath(os.environ.get("SURVIVAL_PHASE3_OUTPUT_CHECKPOINT", PHASE3_CHECKPOINT_FILE))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(base_path):
        raise FileNotFoundError(f"Phase 3 base checkpoint not found: {base_path}")
    model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
    try:
        state = torch.load(base_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(base_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    incompat = model.load_state_dict(state, strict=False)
    allowed_missing = ("kinematics_head.", "cargo_dynamics_head.", "force_balance_residual_head.",
                       "rotational_dynamics_residual_head.", "physics_uncertainty_head.",
                       "trajectory_forecast_head.", "action_transition_action_encoder.",
                       "action_conditioned_transition_head.", "environment_observation_transition_head.")
    missing = [key for key in incompat.missing_keys
               if not (key.startswith(allowed_missing) or key in ("_force_balance_head_trained_state",
                                                                  "_rotational_dynamics_head_trained_state",
                                                                  "_physics_uncertainty_base_trained_state",
                                                                  "_force_uncertainty_trained_state",
                                                                  "_trajectory_forecast_head_trained_state",
                                                                  "_action_conditioned_transition_head_trained_state",
                                                                  "_environment_observation_transition_trained_state"))]
    if incompat.unexpected_keys or missing:
        raise RuntimeError(f"Phase 3 base checkpoint incompatible: unexpected={incompat.unexpected_keys[:8]}, missing={missing[:8]}")
    model._kinematics_head_checkpoint_trained = not any(k.startswith("kinematics_head.") for k in incompat.missing_keys)
    model._cargo_dynamics_head_checkpoint_trained = not any(k.startswith("cargo_dynamics_head.") for k in incompat.missing_keys)
    has_force_head = not any(k.startswith("force_balance_residual_head.") for k in incompat.missing_keys)
    model._force_balance_head_checkpoint_trained = bool(has_force_head and model._force_balance_head_trained_state.item())
    has_rot_head = not any(k.startswith("rotational_dynamics_residual_head.") for k in incompat.missing_keys)
    model._rotational_dynamics_head_checkpoint_trained = bool(
        has_rot_head and model._rotational_dynamics_head_trained_state.item())
    has_uncertainty_head = not any(k.startswith("physics_uncertainty_head.") for k in incompat.missing_keys)
    model._physics_uncertainty_base_checkpoint_trained = bool(
        has_uncertainty_head and model._physics_uncertainty_base_trained_state.item())
    model._force_uncertainty_checkpoint_trained = bool(
        has_uncertainty_head and model._force_uncertainty_trained_state.item())
    has_trajectory_head = not any(k.startswith("trajectory_forecast_head.") for k in incompat.missing_keys)
    model._trajectory_forecast_head_checkpoint_trained = bool(
        has_trajectory_head and model._trajectory_forecast_head_trained_state.item())
    has_action_transition_head = not any(
        k.startswith(("action_transition_action_encoder.", "action_conditioned_transition_head."))
        for k in incompat.missing_keys)
    model._action_conditioned_transition_head_checkpoint_trained = bool(
        has_action_transition_head and model._action_conditioned_transition_head_trained_state.item())
    has_environment_transition_head = not any(
        k.startswith("environment_observation_transition_head.") for k in incompat.missing_keys)
    model._environment_observation_transition_checkpoint_trained = bool(
        has_environment_transition_head and model._environment_observation_transition_trained_state.item())
    if incompat.missing_keys or not model._force_balance_head_checkpoint_trained:
        print("[PHASE 3] Base checkpoint has an untrained/missing force-balance residual head or other auxiliary physics heads; Phase 3 will train Genesis-supervised heads. The rotational head must already be trained from paired BEMT transitions.", flush=True)
    model = train_phase3_genesis_parallel(model, device)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    temp_output = output_path + f".tmp-{os.getpid()}"
    torch.save(model.state_dict(), temp_output)
    os.replace(temp_output, output_path)
    print(f"[PHASE 3 CHECKPOINT] Saved refined model to {output_path}", flush=True)


# =============================================================================
# 17. COCKPIT PROJECTION ENGINES, ARTIFICIAL HORIZON & RADAR MINIMAP
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

def _draw_tactical_radar_minimap_unclipped(screen, map_rect, world, drones, active_idx, vehicles):
    rx, ry, rw, rh = map_rect
    pygame.draw.rect(screen, (15, 20, 28), map_rect)
    pygame.draw.rect(screen, (0, 220, 255), map_rect, 2)
    # Preserve the 400 x 400 m world aspect ratio and fit the entire map inside
    # the allocated panel; width-only scaling used to spill north/south features
    # over the HUD when the panel was only 150 px high.
    scale = max(0.01, (min(rw, rh) - 10.0) / 400.0)
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


def draw_tactical_radar_minimap(screen, map_rect, world, drones, active_idx, vehicles):
    """Draw the radar strictly inside its HUD panel, restoring the caller clip."""
    previous_clip = screen.get_clip()
    try:
        screen.set_clip(map_rect)
        _draw_tactical_radar_minimap_unclipped(screen, map_rect, world, drones, active_idx, vehicles)
    finally:
        screen.set_clip(previous_clip)

# =============================================================================
# 18. DISTRIBUTED CBBA AUCTION & BREGUET ENERGY ROUTING DISPATCHER
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
        """Choose only a genuinely idle, landed, cable-stowed depot drone.

        Active missions are never overwritten in flight. Current live RTH handling
        queues the interrupted order for later dispatch; this helper remains safe for
        future callers that may want to accelerate reassignment.
        """
        best_drone_id = None
        best_bid = -1e9
        for d in candidate_drones:
            if d.drone_id == abandoned_mission.get("drone_id", -1) or d.rth_active:
                continue
            if d.mission_phase != "8_DEPOT_SERVICE":
                continue
            if d.winch.has_payload or float(d.winch.cable_len) > 0.10:
                continue
            if not d.winch.is_skid_grounded or float(d.airframe.p[2]) > 0.10:
                continue
            home = np.asarray(d.mission.get("home_pad", d.home_coords), dtype=np.float32)
            if float(np.linalg.norm(d.airframe.p[:2] - home[:2])) > 0.75:
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
        self.active_missions.pop(drone_id, None)

    def report_failed(self, drone_id):
        """Requeue an actually failed order; failures never count as deliveries."""
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
        self.delivery_abort_retries = 0
        self.delivery_retry_cooldown = 0.0
        self.last_winch_state = self.winch.state
        self.hover_settle_timer = 0.0
        self.chimney_alignment_timer = 0.0
        self.chimney_descent_armed = False
        self.chimney_descent_settle_timer = 0.0
        self.chimney_phase_elapsed = 0.0
        self.depot_timer = 0.0
        self.depot_alignment_timer = 0.0
        self.depot_descent_armed = False
        self.depot_staging_z = float(init_pos[2])
        self.depot_vertical_hold_active = False
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
    """Minimum distance between two finite 3-D segments."""
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
    """Slab-test a finite cable segment against an expanded axis-aligned box."""
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
    """Build the single source of truth for live CBF/cable hitboxes.

    Static architecture, moving road vehicles, the civil aircraft, ground intruder,
    drone bodies, attached payloads and detached payloads are returned here. Road/roof/
    depot support surfaces are maintained separately by world.get_support_surfaces().
    Decorative mesh details are inside their parent object's hitbox and need no duplicate
    collider. Cable lines themselves are tested as polylines against this obstacle set.
    """
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
    # Physical body collider and personal-space avoidance envelope are separate on
    # purpose: the visual body is small, while the controller keeps a larger buffer.
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
    """Fail fast if a current world object is missing a well-formed hitbox."""
    obstacles = build_runtime_collision_obstacles(world, traffic_engine, agent_pos, drones, delivered_boxes)
    errors = []
    seen = set()
    for item in obstacles:
        ident = str(item.get("id", ""))
        shape = str(item.get("shape", "cylinder"))
        if not ident:
            errors.append("obstacle with no id")
        if ident in seen:
            errors.append(f"duplicate hitbox id: {ident}")
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
        else:
            errors.append(f"unsupported hitbox shape {shape}: {ident}")

    expected = {
        "buildings": len(world.houses),
        "rooftop_clutter": sum(len(h.get("clutter", [])) for h in world.houses),
        "trees": len(world.trees),
        "powerlines": len(world.powerlines),
        "powerline_support_poles": 2 * len(world.powerlines),
        "powerline_crossarms": 2 * len(world.powerlines),
        "civil_aircraft": 1,
        "ground_intruder_body": 1,
        "ground_intruder_safety_envelope": 1,
        "road_vehicles": len(traffic_engine.vehicles),
        "drone_bodies": len(drones),
        "attached_packages": sum(bool(d.winch.has_payload) for d in drones),
        "detached_packages": len(delivered_boxes or []),
    }
    counts = {
        "buildings": sum(x["id"].startswith("Bldg_") and ":CLUTTER_" not in x["id"] for x in obstacles),
        "rooftop_clutter": sum(":CLUTTER_" in x["id"] for x in obstacles),
        "trees": sum(x["id"].startswith("Tree_") for x in obstacles),
        "powerlines": sum(x["id"].startswith("Powerline_") and ":POLE_" not in x["id"] for x in obstacles),
        "powerline_support_poles": sum(":POLE_" in x["id"] and ":CROSSARM" not in x["id"] for x in obstacles),
        "powerline_crossarms": sum(":CROSSARM" in x["id"] for x in obstacles),
        "civil_aircraft": sum(x["id"] == str(world.civil_aircraft["id"]) for x in obstacles),
        "ground_intruder_body": sum(x["id"] == "Ground_Intruder" for x in obstacles),
        "ground_intruder_safety_envelope": sum(x["id"] == "Ground_Intruder_SafetyEnvelope" for x in obstacles),
        "road_vehicles": sum(x["id"] in {v["id"] for v in traffic_engine.vehicles} for x in obstacles),
        "drone_bodies": sum(x["id"].startswith("Drone_Body_") for x in obstacles),
        "attached_packages": sum(x["id"].startswith("Attached_Package_D") for x in obstacles),
        "detached_packages": sum(x["id"].startswith("Delivered_Package_") for x in obstacles),
    }
    for key, expected_count in expected.items():
        if counts.get(key) != expected_count:
            errors.append(f"{key}: expected {expected_count} hitboxes, found {counts.get(key)}")
    if len(world.get_support_surfaces()) < 1 + len(world.roads) + len(world.depot_helipads) + 2 * len(world.houses):
        errors.append("one or more ground/road/depot/roof support surfaces are missing")
    if errors:
        raise RuntimeError("Collision coverage audit failed: " + "; ".join(errors))
    return {"obstacle_total": len(obstacles), "support_surface_total": len(world.get_support_surfaces()),
            "counts": counts, "cable_polyline_checks": len(drones), "valid": True}


def _draw_collision_hitbox_overlay(screen, world, traffic_engine, agent_pos, drones, delivered_boxes,
                                   camera_pos, camera_lookat, view_w, view_h, fov_deg=48.0):
    """Draw the exact analytic CBF/cable hitboxes over the rendered 3D camera view.

    Red/orange/yellow outlines are collision obstacles; cyan/green outlines are support
    surfaces. This is a display overlay only and never changes physics or control.
    """
    import pygame
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
        if pa is None or pb is None:
            return
        if (max(pa[0], pb[0]) < -20 or min(pa[0], pb[0]) > view_w + 20
                or max(pa[1], pb[1]) < -20 or min(pa[1], pb[1]) > view_h + 20):
            return
        pygame.draw.line(screen, color, pa[:2], pb[:2], width)

    obstacles = build_runtime_collision_obstacles(world, traffic_engine, agent_pos, drones, delivered_boxes)
    static_red = (255, 82, 88)
    clutter_orange = (255, 151, 66)
    wire_yellow = (255, 224, 88)
    dynamic_orange = (255, 174, 62)
    payload_gold = (255, 216, 70)
    drone_cyan = (72, 242, 255)
    person_magenta = (255, 88, 214)

    for obs in obstacles:
        ident = str(obs["id"])
        shape = obs.get("shape", "cylinder")
        if ":CLUTTER_" in ident:
            color = clutter_orange
        elif "Powerline_" in ident or ":CROSSARM" in ident or ":POLE_" in ident:
            color = wire_yellow
        elif ident.startswith("Drone_Body_"):
            color = drone_cyan
        elif "Package" in ident:
            color = payload_gold
        elif "Ground_Intruder" in ident:
            color = person_magenta
        elif any(ident.startswith(v["id"]) for v in traffic_engine.vehicles):
            color = dynamic_orange
        elif ident == str(world.civil_aircraft["id"]):
            color = (255, 116, 222)
        else:
            color = static_red

        if shape == "aabb":
            x1, x2, y1, y2 = (float(v) for v in obs["bounds"])
            z0, z1 = float(obs.get("base_height", 0.0)), float(obs.get("height", 0.0))
            if z1 <= z0:
                continue
            pts = [(x1,y1,z0),(x2,y1,z0),(x2,y2,z0),(x1,y2,z0),
                   (x1,y1,z1),(x2,y1,z1),(x2,y2,z1),(x1,y2,z1)]
            edges = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                     (0,4),(1,5),(2,6),(3,7))
            for i,j in edges:
                line3(pts[i], pts[j], color, 1)
        elif shape == "segment":
            line3(obs["p1"], obs["p2"], color, 2)
            # Small cross-lines visualise the wire radius at both endpoints.
            for endpoint in (np.asarray(obs["p1"], dtype=np.float64), np.asarray(obs["p2"], dtype=np.float64)):
                rad = float(obs.get("radius", 0.1))
                line3(endpoint + np.array([0, -rad, 0]), endpoint + np.array([0, rad, 0]), color, 2)
                line3(endpoint + np.array([0, 0, -rad]), endpoint + np.array([0, 0, rad]), color, 2)
        else:
            centre = np.asarray(obs.get("pos", np.zeros(3)), dtype=np.float64)
            radius = float(obs.get("radius", 0.0))
            if radius <= 0.0:
                continue
            if shape == "sphere":
                for plane in ("xy", "xz", "yz"):
                    ring = []
                    for angle in np.linspace(0.0, 2.0 * math.pi, 25):
                        c, sn = math.cos(float(angle)), math.sin(float(angle))
                        if plane == "xy":
                            ring.append(centre + [radius*c, radius*sn, 0.0])
                        elif plane == "xz":
                            ring.append(centre + [radius*c, 0.0, radius*sn])
                        else:
                            ring.append(centre + [0.0, radius*c, radius*sn])
                    for a, b in zip(ring[:-1], ring[1:]):
                        line3(a, b, color, 1)
            else:  # cylinder and conservative personal-space envelopes
                z0 = float(obs.get("base_height", 0.0))
                z1 = float(obs.get("height", z0 + 1.0))
                low_ring, high_ring = [], []
                for angle in np.linspace(0.0, 2.0 * math.pi, 25):
                    a = float(angle)
                    low_ring.append(np.array([centre[0] + radius*math.cos(a), centre[1] + radius*math.sin(a), z0]))
                    high_ring.append(np.array([centre[0] + radius*math.cos(a), centre[1] + radius*math.sin(a), z1]))
                for i in range(len(low_ring)-1):
                    line3(low_ring[i], low_ring[i+1], color, 1)
                    line3(high_ring[i], high_ring[i+1], color, 1)
                for i in (0, 6, 12, 18):
                    line3(low_ring[i], high_ring[i], color, 1)

    # Support/contact surfaces are not no-fly volumes; show them in cyan/green.
    support_color = (65, 244, 174)
    for surface in world.get_support_surfaces():
        kind = surface.get("shape")
        ident = str(surface.get("id", ""))
        if kind == "aabb_plane":
            x1, x2, y1, y2 = (float(v) for v in surface["bounds"])
            z = float(surface.get("z", 0.0))
            corners = [(x1,y1,z),(x2,y1,z),(x2,y2,z),(x1,y2,z)]
            for i in range(4):
                line3(corners[i], corners[(i+1) % 4], support_color, 1)
        elif kind == "cylinder":
            centre = np.asarray(surface.get("pos", np.zeros(3)), dtype=np.float64).copy()
            centre[2] = float(surface.get("z_top", centre[2]))
            radius = float(surface.get("radius", 0.0))
            ring = [centre + [radius*math.cos(float(a)), radius*math.sin(float(a)), 0.0]
                    for a in np.linspace(0.0, 2.0*math.pi, 33)]
            for a,b in zip(ring[:-1],ring[1:]):
                line3(a,b,support_color,1)

    legend = _cached_software_font("consolas", 11, True)
    msg = "HITBOX OVERLAY  K:OFF  red=buildings  orange=clutter/vehicles  cyan=drones/support  gold=packages"
    surf = legend.render(msg, True, (242, 248, 255))
    panel = pygame.Rect(10, int(view_h) - 28, min(int(view_w)-20, surf.get_width()+12), 22)
    pygame.draw.rect(screen, (6, 14, 23), panel, border_radius=4)
    pygame.draw.rect(screen, (55, 235, 205), panel, 1, border_radius=4)
    screen.blit(surf, (panel.x+6, panel.y+4), area=pygame.Rect(0,0,max(1,panel.width-12),max(1,panel.height-8)))


def _segment_intersects_dynamic_hitboxes(p1, p2, obstacles, margin=0.08):
    """Test a cable segment against cylinder, sphere, exact AABB and wire hitboxes."""
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
    """Check each actual cable segment, rather than its potentially unsafe end chord."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2 or not np.all(np.isfinite(pts)):
        return False
    return any(_segment_intersects_dynamic_hitboxes(pts[i], pts[i + 1], obstacles, margin)
               for i in range(len(pts) - 1))


def _winch_cable_polyline_points(winch, drone_pos):
    """Return cable points trimmed to the parcel-top attachment when carrying cargo."""
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
    """True only when altitude is near its target and vertical motion is settled."""
    values = (current_z_m, current_vz_m_s, target_z_m,
              altitude_tolerance_m, vertical_speed_limit_m_s)
    if not all(math.isfinite(float(v)) for v in values):
        return False
    return (abs(float(current_z_m) - float(target_z_m)) <= float(altitude_tolerance_m)
            and abs(float(current_vz_m_s)) <= float(vertical_speed_limit_m_s))


def _should_enter_precision_approach(distance_xy_m, speed_xy_m_s, wind_speed_xy_m_s):
    """Enter target capture early; altitude and wind are handled by the safe approach state.

    The prior combined gate kept the drone in cruise transit if altitude or wind was
    outside limits exactly when it crossed the braking radius. That allowed it to pass
    the delivery target without ever entering the controller designed to settle over it.
    Entering this state does not arm descent: the separate altitude, velocity and wind
    checks inside the state remain authoritative.
    """
    if not all(math.isfinite(float(v)) for v in
               (distance_xy_m, speed_xy_m_s, wind_speed_xy_m_s)):
        return False
    if float(distance_xy_m) < 0.0:
        return False
    return float(distance_xy_m) <= _delivery_approach_braking_radius(
        speed_xy_m_s, wind_speed_xy_m_s)


def _delivery_approach_braking_radius(speed_xy_m_s, wind_speed_xy_m_s=0.0):
    """Brake before the target using the same conservative deceleration as the approach profile."""
    speed = max(0.0, float(speed_xy_m_s))
    wind = max(0.0, float(wind_speed_xy_m_s))
    # The approach profile limits speed using sqrt(2 * 1.1 * distance). The old
    # radius assumed 1.8 m/s^2, so it handed over too late for its own speed profile.
    # Add room for sensing, control lag, and wind without exceeding city-scale limits.
    braking_deceleration = 1.10
    transition_margin = 5.5 + min(6.0, 0.45 * wind)
    return float(np.clip((speed * speed) / (2.0 * braking_deceleration) + transition_margin, 10.0, 42.0))


def _delivery_vertical_settle_reference(v_ref, vertical_error_m, deadband_m=0.75):
    """Remove continued vertical feed-forward close to the delivery hover altitude.

    compute_vector_guidance intentionally requests downward velocity while a target is
    below the aircraft. Close to a rooftop hover target, that can keep requesting descent
    even as the package/drone should be settling, preventing the vertical-speed gate from
    ever becoming true. Position feedback remains active through p_ref; this only zeros
    the commanded vertical reference inside a small capture band.
    """
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
            if current_p[2] < 2.5:
                climb_ramp_ceiling = float(np.clip(0.65 + 0.70 * current_p[2], 0.65, max_climb))
            else:
                climb_ramp_ceiling = max_climb
            v_cmd_z = min(climb_ramp_ceiling, math.sqrt(max(0.0, 2.0 * 1.0 * dist_z)))
        else:
            v_cmd_z = -min(max_descend, math.sqrt(max(0.0, 2.0 * 0.50 * abs(dist_z))))
        
    v_des = np.array([v_cmd_xy[0], v_cmd_xy[1], v_cmd_z], dtype=np.float32)
    return target_p.copy(), v_des


def _depot_inbound_speed_cap(distance_to_pad_m):
    """Speed cap with an explicit braking buffer before the depot pad.

    The cap reaches zero at DEPOT_APPROACH_BRAKE_MARGIN_M, allowing the aircraft
    to enter pad capture slowly instead of still requesting 3-4 m/s at 3 m range.
    """
    distance = max(0.0, float(distance_to_pad_m))
    braking_distance = max(0.0, distance - DEPOT_APPROACH_BRAKE_MARGIN_M)
    return min(9.5, math.sqrt(2.0 * DEPOT_APPROACH_BRAKE_ACCEL_MS2 * braking_distance))


def _should_enter_depot_capture(distance_to_pad_m, current_p, current_v):
    """Only enter final pad-capture logic when position and momentum are capturable.

    The inbound controller first targets the 24 m staging altitude inside the pad
    capture radius. This gate prevents a high-speed crossing from latching the landing
    controller at an arbitrary height with substantial residual vertical momentum.
    """
    p = np.asarray(current_p, dtype=np.float64).reshape(3)
    v = np.asarray(current_v, dtype=np.float64).reshape(3)
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(v))
            and math.isfinite(float(distance_to_pad_m))):
        return False
    return bool(
        float(distance_to_pad_m) <= DEPOT_APPROACH_CAPTURE_RADIUS_M
        and abs(float(p[2]) - DEPOT_CAPTURE_STAGING_ALTITUDE_M) <= DEPOT_CAPTURE_ALTITUDE_TOLERANCE_M
        and float(np.linalg.norm(v[:2])) <= DEPOT_CAPTURE_MAX_XY_SPEED_MS
        and abs(float(v[2])) <= DEPOT_CAPTURE_MAX_VERTICAL_SPEED_MS
    )


def _velocity_damped_xy_reference(target_xy, current_xy, current_vxy,
                                  position_gain=0.85, velocity_gain=1.50,
                                  max_speed=1.80, max_position_speed=0.90):
    """Return a bounded XY velocity command with explicit momentum damping.

    A position-only speed cap tends to zero at the target; that also caps braking
    authority precisely when the vehicle crosses the target with residual velocity.
    This command separately bounds the position attraction, subtracts measured
    lateral velocity, then applies only a global actuator-safe speed limit.
    """
    target = np.asarray(target_xy, dtype=np.float32).reshape(2)
    position = np.asarray(current_xy, dtype=np.float32).reshape(2)
    velocity = np.asarray(current_vxy, dtype=np.float32).reshape(2)
    if not (np.all(np.isfinite(target)) and np.all(np.isfinite(position))
            and np.all(np.isfinite(velocity))):
        return np.zeros(2, dtype=np.float32)
    delta = target - position
    attraction = float(position_gain) * delta
    attraction_norm = float(np.linalg.norm(attraction))
    attraction_cap = max(0.0, float(max_position_speed))
    if attraction_norm > attraction_cap and attraction_norm > 1e-8:
        attraction *= attraction_cap / attraction_norm
    command = attraction - max(0.0, float(velocity_gain)) * velocity
    command_norm = float(np.linalg.norm(command))
    command_cap = max(0.0, float(max_speed))
    if command_norm > command_cap and command_norm > 1e-8:
        command *= command_cap / command_norm
    return command.astype(np.float32)


def _seed_vertical_command_smoother(mixer, measured_vz):
    """Re-seed vertical command history at a precision-hold boundary.

    The mixer jerk-limits vertical-velocity commands across mission phases. If a
    cruise descent leaves a negative smoothed command/acceleration behind, merely
    changing ref_vel to zero does not immediately produce an altitude hold. Seed
    the command from measured motion (within actuator command bounds) and clear
    the old acceleration so the feedback loop brakes current momentum rather than
    replaying stale descent intent.
    """
    vz = float(measured_vz)
    if not math.isfinite(vz):
        vz = 0.0
    mixer.vertical_v_cmd_smooth = float(np.clip(vz, -0.60, 1.05))
    mixer.vertical_a_cmd_smooth = 0.0


def _depot_recovery_staging_altitude(current_z_m, current_staging_z_m):
    """Choose a stable recovery height after wind interrupts depot descent.

    Above the near-ground band, return to the existing staging target (normally 24 m),
    rather than latching whatever altitude happened to coincide with the gust. Inside
    the last 1.5 m, recover to 1.5 m so a near-touchdown gust does not trigger a needless
    climb back to full staging altitude.
    """
    current_z = float(current_z_m)
    stage_z = float(current_staging_z_m)
    if not math.isfinite(current_z):
        return DEPOT_CAPTURE_STAGING_ALTITUDE_M
    if current_z <= DEPOT_LANDING_LOW_ALT_HOLD_HEIGHT_M:
        return float(DEPOT_LANDING_LOW_ALT_HOLD_HEIGHT_M)
    if math.isfinite(stage_z) and stage_z > DEPOT_LANDING_LOW_ALT_HOLD_HEIGHT_M:
        return stage_z
    return float(DEPOT_CAPTURE_STAGING_ALTITUDE_M)


def _depot_landing_guidance(current_p, current_v, home_xy, staging_z,
                            descent_armed, alignment_timer, dt, wind_vector=None):
    """Capture the depot pad and explicitly recover from weather interruptions.

    Any wind condition that makes descent unsafe disarms the descent and starts a new
    recapture cycle. The drone is guided horizontally back to the pad and vertically to
    its last safe staging target (normally 24 m, or 1.5 m in the final near-ground band).
    Only a fresh low-wind, centred, low-speed dwell can re-arm descent. Crucially, the
    controller does not replace the known staging target with an arbitrary gust-time
    altitude, which previously allowed a displaced drone to remain at an unsuitable height.

    Returns (p_ref, v_ref, armed, timer, staging_z, armed_now, paused_now).
    """
    p = np.asarray(current_p, dtype=np.float32).reshape(3)
    v = np.asarray(current_v, dtype=np.float32).reshape(3)
    home = np.asarray(home_xy, dtype=np.float32).reshape(2)
    staging_z = float(staging_z)
    dt = max(0.0, float(dt))
    xy_error = float(np.linalg.norm(p[:2] - home))
    xy_speed = float(np.linalg.norm(v[:2]))
    z_speed = abs(float(v[2]))
    wind3 = (np.asarray(wind_vector, dtype=np.float32).reshape(-1)[:3]
             if wind_vector is not None else np.zeros(3, dtype=np.float32))
    if wind3.size < 3 or not np.all(np.isfinite(wind3)):
        wind_speed_xy = float("inf")
        wind_speed_z_abs = float("inf")
    else:
        wind_speed_xy = float(np.linalg.norm(wind3[:2]))
        wind_speed_z_abs = abs(float(wind3[2]))
    # Hold vertical landing progress for either excessive horizontal wind or a vertical
    # gust. Use the sensed vector only; ground-truth wind remains confined to physics.
    wind_pause = (wind_speed_xy > DEPOT_LANDING_WIND_RESUME_MS
                  or wind_speed_z_abs > DEPOT_LANDING_VERTICAL_WIND_PAUSE_MS)
    armed = bool(descent_armed)
    timer = max(0.0, float(alignment_timer))
    armed_now = False
    paused_now = False
    entered_hold_this_step = False

    # Weather is treated as a recapture event, not as a reason to remain indefinitely
    # in an armed-but-paused descent. Disarm on the pause envelope (not only the harder
    # emergency threshold), restore a known recovery altitude, and require a fresh settle
    # dwell before descending again. Keep the staging target stable across gusts.
    if armed and wind_pause and p[2] > 0.08:
        armed = False
        timer = 0.0
        staging_z = _depot_recovery_staging_altitude(p[2], staging_z)
        paused_now = True
        entered_hold_this_step = True
    elif armed and p[2] > DEPOT_LANDING_PAUSE_HEIGHT_M and (
            xy_error > DEPOT_LANDING_RECAPTURE_RADIUS_M
            or (xy_error > DEPOT_LANDING_CAPTURE_RADIUS_M and xy_speed > 0.65)):
        # A significant lateral departure also restarts pad capture. Restore the stable
        # recovery height rather than locking the vertical target to the displaced state.
        armed = False
        timer = 0.0
        staging_z = _depot_recovery_staging_altitude(p[2], staging_z)
        paused_now = True
        entered_hold_this_step = True

    target_xy = home
    if not armed:
        wp_target = np.array([home[0], home[1], staging_z], dtype=np.float32)
        p_ref, v_ref = compute_vector_guidance(
            p, v, wp_target, max_speed_xy=0.0,
            max_climb=0.30, max_descend=0.30, wind_vector=wind_vector)
        # Bounded position attraction plus measured-velocity damping. Keep the stopping
        # command effective even when position error is near zero; the mixer also has
        # its own feedback, so avoid feeding it a duplicate altitude-position error.
        v_ref[:2] = _velocity_damped_xy_reference(
            target_xy, p[:2], v[:2], position_gain=1.05, velocity_gain=1.85,
            max_speed=1.60, max_position_speed=1.00)
        staging_z_error = float(staging_z - p[2])
        # Keep ordinary hold velocity-led near the staging point. If a gust has displaced
        # the craft materially from its latched altitude, re-enable position feedback and
        # allow a stronger bounded recovery request instead of taking tens of seconds to
        # recover at the previous +/-0.32 m/s cap.
        if abs(staging_z_error) > DEPOT_STAGING_POSITION_FEEDBACK_THRESHOLD_M:
            p_ref[2] = float(staging_z)
        else:
            p_ref[2] = float(p[2])
        v_ref[2] = float(np.clip(
            0.65 * staging_z_error - 1.15 * float(v[2]), -0.55, 0.55))
        if entered_hold_this_step:
            # One-frame bumpless transfer: stop replaying the old descent request. On
            # the next frame, normal feedback actively restores the safe recovery height.
            p_ref[2] = float(p[2])
            v_ref[2] = 0.0
        paused_now = bool(entered_hold_this_step or wind_pause)

        stable = (
            xy_error <= DEPOT_LANDING_CAPTURE_RADIUS_M
            and xy_speed <= DEPOT_LANDING_MAX_XY_SPEED_MS
            and z_speed <= DEPOT_LANDING_MAX_Z_SPEED_MS
            and abs(float(p[2]) - staging_z) <= 0.35
            and not wind_pause
        )
        timer = timer + dt if stable else 0.0
        if timer >= DEPOT_LANDING_SETTLE_DWELL_S:
            armed = True
            armed_now = True
            timer = 0.0
    else:
        # Hold off the vertical axis above the pad when the wind is over the landing
        # envelope. Keep descent authorised only for moderate wind; a hard gust above
        # the threshold already disarmed it at the top of this function.
        p_ref = np.array([home[0], home[1], -0.05], dtype=np.float32)
        if p[2] > 4.0:
            descent_cap = DEPOT_LANDING_DESCEND_SPEED_MS
        elif p[2] > DEPOT_LANDING_LOW_ALT_HOLD_HEIGHT_M:
            descent_cap = 0.28
        elif p[2] > DEPOT_LANDING_PAUSE_HEIGHT_M:
            descent_cap = DEPOT_LANDING_FINAL_DESCEND_SPEED_MS
        else:
            descent_cap = 0.10
        # Use slow descent as a function of height. The mixer sees p_target_z <= 0.05
        # and therefore retains its touchdown-specific thrust limits.
        height_above_pad = max(0.0, float(p[2]) - 0.04)
        braking_speed = math.sqrt(max(0.0, 2.0 * 0.30 * height_above_pad))
        vertical_descent = min(descent_cap, max(0.06 if p[2] > 0.35 else 0.04, braking_speed))
        v_ref = np.zeros(3, dtype=np.float32)
        v_ref[:2] = _velocity_damped_xy_reference(
            target_xy, p[:2], v[:2], position_gain=0.70, velocity_gain=1.75,
            max_speed=0.90, max_position_speed=0.50)
        v_ref[2] = -vertical_descent

        if p[2] > DEPOT_LANDING_PAUSE_HEIGHT_M:
            if p[2] <= DEPOT_LANDING_LOW_ALT_HOLD_HEIGHT_M:
                xy_hold_error = DEPOT_LANDING_LOW_ALT_HOLD_XY_ERROR_M
                xy_hold_speed = DEPOT_LANDING_LOW_ALT_HOLD_XY_SPEED_MS
            else:
                xy_hold_error = DEPOT_LANDING_DESCENT_HOLD_XY_ERROR_M
                xy_hold_speed = DEPOT_LANDING_DESCENT_HOLD_XY_SPEED_MS
            lateral_hold = (
                wind_pause
                or xy_error > xy_hold_error
                or xy_speed > xy_hold_speed
            )
        else:
            lateral_hold = wind_pause

        if lateral_hold:
            p_ref[2] = float(p[2])
            v_ref[2] = float(np.clip(-1.15 * float(v[2]), -0.30, 0.30))
            paused_now = True

    return p_ref, v_ref, armed, timer, staging_z, armed_now, paused_now

# =============================================================================
# 19. GENESIS 3D MIRROR, CAMERA CONTROLS & ROBUST VISUAL FALLBACK
# Camera/rendering changes only: V16 flight simulation and learning core preserved.
# =============================================================================

# V99's original teacher training, policy, sensor calculations, missions, BEMT
# dynamics, actuator mixing, safety handling, and winch logic remain authoritative.
# Genesis renders the same city and mirrors V99 state; it does not run a second
# competing flight physics model. This is the behaviour-preserving 3D upgrade.

def _rotation_matrix_to_quaternion_wxyz(R):
    """Convert a 3x3 rotation matrix to a normalised Genesis wxyz quaternion."""
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
    """Quaternion rotating a local +Z cylinder axis onto a world-space segment."""
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
    """Synchronise a render proxy to the original simulation state."""
    p = np.asarray(pos, dtype=np.float32).reshape(3)
    if not np.all(np.isfinite(p)):
        return
    entity.set_pos(tuple(float(v) for v in p))
    if quat_wxyz is not None:
        q = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
        if np.all(np.isfinite(q)):
            entity.set_quat(tuple(float(v) for v in q))
    # This scene is a visual mirror only and scene.step() is never called.
    # set_pos/set_quat are therefore sufficient; clearing velocity and angular
    # velocity every render was redundant work for every proxy entity.


def _generate_v99_visual_urdf(filepath):
    """Create a visual coaxial X8 proxy using V99's original 0.26 m arm length."""
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
    """Return a configurable, aspect-preserving camera resolution.

    The Pygame cockpit keeps its original size; only the off-screen Genesis
    image is rendered smaller and then scaled to the viewport.
    """
    try:
        requested_width = int(os.environ.get("SURVIVAL_RENDER_WIDTH", "768"))
    except (TypeError, ValueError):
        requested_width = 512
    width = max(320, min(int(view_w), requested_width))
    height = max(240, int(round(width * int(view_h) / max(1, int(view_w)))))
    return width, height


def _camera_render_fps():
    """Target camera/viewport rendering rate; physics continues at its own rate."""
    try:
        requested = float(os.environ.get("SURVIVAL_RENDER_FPS", "15"))
    except (TypeError, ValueError):
        requested = 15.0
    return float(np.clip(requested, 10.0, 60.0))


def _simulation_target_fps():
    """Main physics-loop pacing; independent from the cached camera render rate."""
    try:
        requested = float(os.environ.get("SURVIVAL_SIM_FPS", "60"))
    except (TypeError, ValueError):
        requested = 60.0
    # 60 Hz is close to real time for the authoritative fixed dt=0.016 s.
    return int(np.clip(requested, 15.0, 60.0))


def build_genesis_3d_mirror(world, drones, traffic_engine, ground_agent, view_w, view_h):
    """Build a Genesis 3D rendering scene at the original V99 world coordinates."""
    try:
        import genesis as gs
    except ImportError as exc:
        raise RuntimeError("The 3D environment requires the Genesis package installed in this Python environment.") from exc
    try:
        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu,
                logging_level=_configured_genesis_logging_level())
    except Exception as exc:
        raise RuntimeError(f"Genesis initialisation failed: {type(exc).__name__}: {exc}") from exc

    # This Genesis scene is a render mirror only: V99 remains the authoritative
    # flight-physics simulator. Disable mirror collisions and gravity so visual
    # proxy bodies can be teleported to V99 state without a second physics model.
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
        # Older Genesis builds may expose fewer visualization options. Keep the
        # supported scene constructor rather than failing the entire simulator.
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

    # Use a closed, fixed render-only box instead of an open Plane mesh. The
    # visual scene has zero gravity and collision=False throughout; the top face
    # remains at z=0, matching the old plane. A 1 km square covers the full city
    # and camera overview while avoiding the non-watertight Plane inertia warning.
    add_box((1000.0, 1000.0, 0.10), (0.0, 0.0, -0.05),
            (0.16, 0.18, 0.21), roughness=0.9, fixed=True)
    for road in world.roads:
        sx = max(0.1, float(road['x_max']-road['x_min']))
        sy = max(0.1, float(road['y_max']-road['y_min']))
        add_box((sx,sy,0.025), ((road['x_min']+road['x_max'])/2, (road['y_min']+road['y_max'])/2, 0.0125),
                (0.075,0.085,0.10), roughness=0.95, friction=0.9, fixed=True)
    # Subtle dashed centre markings improve city-scale depth perception. These are
    # render-only meshes; road support/collision remains in the authoritative map.
    for x in world.avenue_x:
        for y in np.arange(-188.0, 189.0, 28.0):
            add_box((0.12, 5.0, 0.018), (float(x), float(y), 0.034), (0.78, 0.68, 0.34), roughness=0.9, fixed=True)
    for y in world.street_y:
        for x in np.arange(-188.0, 189.0, 28.0):
            add_box((5.0, 0.12, 0.018), (float(x), float(y), 0.034), (0.78, 0.68, 0.34), roughness=0.9, fixed=True)

    for idx, h in enumerate(world.houses):
        x1,x2,y1,y2,top = [float(v) for v in h['aabb']]
        cx,cy = (x1+x2)/2, (y1+y2)/2
        facade_palette = [(0.26,0.31,0.39), (0.34,0.32,0.30), (0.25,0.36,0.38), (0.38,0.37,0.34), (0.29,0.32,0.36)]
        facade = facade_palette[idx % len(facade_palette)]
        add_box((x2-x1,y2-y1,top), (cx,cy,top/2), facade, fixed=True)
        # Two facing sides get restrained blue glass strips at regular floor levels.
        floor_levels = int(max(1, min(6, math.floor(top / 4.5))))
        for floor_i in range(floor_levels):
            wz = min(top - 1.0, 2.0 + floor_i * 4.0)
            if wz <= 0.5:
                continue
            # Four-sided window treatment; each strip remains inside the building's
            # AABB, so the building collider fully covers all facade details.
            add_box((12.5, 0.055, 0.72), (cx, y2 - 0.035, wz), (0.10, 0.23, 0.31), roughness=0.3, fixed=True)
            add_box((12.5, 0.055, 0.72), (cx, y1 + 0.035, wz), (0.10, 0.23, 0.31), roughness=0.3, fixed=True)
            add_box((0.055, 12.5, 0.72), (x2 - 0.035, cy, wz), (0.10, 0.23, 0.31), roughness=0.3, fixed=True)
            add_box((0.055, 12.5, 0.72), (x1 + 0.035, cy, wz), (0.10, 0.23, 0.31), roughness=0.3, fixed=True)
        add_box((x2-x1+0.12,0.12,0.10),(cx,y2,top-0.05),(0.48,0.51,0.56),roughness=0.5,fixed=True)
        add_cylinder(2.2,0.08,(cx,cy,top+0.04),(0.0,0.72,0.50),fixed=True)
        # High-contrast visual drop reticle; its geometry sits on the existing pad
        # support surface and does not introduce an independent collision volume.
        add_box((2.45, 0.055, 0.014), (cx, cy, top + PHASE3_ROOF_PAD_TOP_OFFSET_M + 0.009),
                (0.80, 0.95, 0.94), roughness=0.35, fixed=True)
        add_box((0.055, 2.45, 0.014), (cx, cy, top + PHASE3_ROOF_PAD_TOP_OFFSET_M + 0.009),
                (0.80, 0.95, 0.94), roughness=0.35, fixed=True)
        for c in h.get('clutter',[]):
            cx1,cx2,cy1,cy2,ctop = [float(v) for v in c['aabb']]
            clutter_h = max(0.15, ctop-top)
            add_box((cx2-cx1,cy2-cy1,clutter_h), ((cx1+cx2)/2,(cy1+cy2)/2,top+clutter_h/2), (0.48,0.50,0.54), roughness=0.55, fixed=True)

    palette = [(0.10,0.65,0.95),(0.95,0.55,0.15),(0.75,0.25,0.95)]
    for i,pad in enumerate(world.depot_helipads):
        add_cylinder(2.8,0.08,(float(pad[0]),float(pad[1]),0.04),palette[i%len(palette)],fixed=True)
        add_box((2.2, 0.06, 0.014), (float(pad[0]), float(pad[1]), 0.087),
                (0.88, 0.92, 0.95), roughness=0.35, fixed=True)
        add_box((0.06, 2.2, 0.014), (float(pad[0]), float(pad[1]), 0.087),
                (0.88, 0.92, 0.95), roughness=0.35, fixed=True)
    for wi, line in enumerate(getattr(world,'powerlines',[])):
        a,b = np.asarray(line['p1'],dtype=np.float32), np.asarray(line['p2'],dtype=np.float32)
        d = b-a
        yaw = math.atan2(float(d[1]),float(d[0]))
        # Lines are outside the city block footprint; their end poles match the
        # corresponding analytic CBF hitboxes in get_static_collision_obstacles().
        add_box((max(0.1,float(np.linalg.norm(d[:2]))),0.08,0.08), tuple((a+b)/2), (0.72,0.73,0.76), roughness=0.4, fixed=True)
        for endpoint in (a, b):
            pole_h = float(endpoint[2])
            add_cylinder(0.16, pole_h, (float(endpoint[0]), float(endpoint[1]), pole_h/2.0), (0.35,0.38,0.42), fixed=True)
            add_box((2.0,0.12,0.10), (float(endpoint[0]), float(endpoint[1]), pole_h-0.12), (0.35,0.38,0.42), fixed=True)
    for tree in world.trees:
        tx,ty = float(tree['pos'][0]),float(tree['pos'][1])
        radius,top = float(tree.get('r',3.2)),float(tree.get('h',8.0))
        trunk_h = min(5.0,top*0.7)
        add_cylinder(0.45,trunk_h,(tx,ty,trunk_h/2),(0.40,0.25,0.15),fixed=True)
        add_sphere(radius,(tx,ty,top-radius),(0.15,0.50,0.20),fixed=True)

    urdf_path = _generate_v99_visual_urdf(os.path.abspath('drone_coaxial_x8_v99_visual.urdf'))
    drone_entities, cargo_entities, cable_entities = [], [], []
    for d in drones:
        p = np.asarray(d.airframe.p,dtype=np.float32)
        drone_entities.append(scene.add_entity(morph=gs.morphs.URDF(file=urdf_path,pos=tuple(float(v) for v in p),fixed=True,collision=False)))
        initial_payload_z = float(p[2] - PHASE3_DRONE_HALF_HEIGHT_M - d.winch.cable_len -
                                  (PHASE3_CARGO_HALF_HEIGHT_M if d.winch.has_payload else 0.0))
        cargo_entities.append(add_box((0.22,0.22,0.18),(p[0],p[1],max(0.09,initial_payload_z)),(0.95,0.75,0.15),roughness=0.5,fixed=True))
        # Overlapping short cylinders provide a visually continuous cable while
        # remaining inexpensive for the cached 30-FPS viewport.
        cable_entities.append([add_cylinder(0.012, 0.30, (p[0], p[1], max(0.02, p[2]-0.1)),
                                           (0.78, 0.82, 0.88), fixed=True)
                               for _ in range(40)])
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
        near=0.05, far=3000.0,
    )
    try:
        camera = scene.add_camera(**camera_kwargs)
    except TypeError:
        # Compatibility with older Genesis releases without near/far keywords.
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
    # Prime graphics buffers; some Genesis renderer/backend combinations return
    # a blank first frame immediately after scene.build().
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
    """Move the camera toward its look-at target when static world geometry blocks the view."""
    target=np.asarray(camera_lookat,dtype=np.float64).reshape(3)
    desired=np.asarray(camera_pos,dtype=np.float64).reshape(3)
    ray=desired-target
    desired_dist=float(np.linalg.norm(ray))
    if not np.isfinite(desired_dist) or desired_dist<1e-6:
        return desired.astype(np.float32)
    direction=ray/desired_dist
    nearest=desired_dist

    # Tower volumes and roof clutter use the same AABBs as V99's authoritative world.
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

    # Conservative sphere tests for tree canopies.
    for tree in getattr(world,'trees',[]):
        centre=np.array([float(tree['pos'][0]),float(tree['pos'][1]),float(tree.get('h',8.0))-float(tree.get('r',3.2))])
        radius=float(tree.get('r',3.2))+0.75
        oc=target-centre
        b=float(np.dot(oc,direction))
        c=float(np.dot(oc,oc)-radius*radius)
        disc=b*b-c
        if disc>=0.0:
            t=-b-math.sqrt(disc)
            if 1.25<t<nearest:
                nearest=t

    # Current V99 lines are horizontal along X; expanded bounds include camera clearance.
    for line in getattr(world,'powerlines',[]):
        a=np.asarray(line['p1'],dtype=np.float64); b=np.asarray(line['p2'],dtype=np.float64)
        lo=np.minimum(a,b)-np.array([0.5,0.65,0.65])
        hi=np.maximum(a,b)+np.array([0.5,0.65,0.65])
        t=_ray_aabb_first_hit(target,direction,lo,hi)
        if t is not None and 1.25<float(t)<nearest:
            nearest=float(t)

    safe_distance=max(float(minimum_distance),nearest-float(clearance))
    if nearest < desired_dist and safe_distance < desired_dist:
        desired=target+direction*safe_distance
    desired[2]=max(0.85,float(desired[2]))
    return desired.astype(np.float32)


def screen_to_world_raycast_genesis(mouse_pos,camera_pos,camera_lookat,view_w,view_h,world,fov_deg=48.0):
    """Perspective click ray matching the Genesis camera, returning nearest 3D surface hit."""
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
    """Stable camera framing for the 400 m V99 world; never changes physics."""
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
        # View the payload without placing the virtual camera below ground.
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

    # CHASE: compose the drone in the near lower-middle and aim down the route.
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


_SOFTWARE_RENDER_FONT_CACHE = {}


def _cached_software_font(name="consolas", size=12, bold=True):
    import pygame
    key = (name, int(size), bool(bold))
    font = _SOFTWARE_RENDER_FONT_CACHE.get(key)
    if font is None:
        font = pygame.font.SysFont(name, int(size), bold=bool(bold))
        _SOFTWARE_RENDER_FONT_CACHE[key] = font
    return font


def _software_render_v99_3d(screen, world, drones, camera_pos, camera_lookat, view_w, view_h,
                             selected_drone_idx, active_target, cam_mode, fov_deg=48.0,
                             delivered_boxes=None):
    """Perspective Pygame fallback used when Genesis returns a blank/error frame.

    This is visualisation only. It draws the same world coordinates and V99 drone
    states; it does not feed back into flight dynamics, observations, or rewards.
    """
    import pygame
    width, height = int(view_w), int(view_h)
    screen.fill((10, 16, 26), (0, 0, width, height))
    # Deep-blue sky gradient gives the city silhouette clear separation.
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
        # Clip polygons to the near plane instead of discarding an entire tower
        # face when only one corner crosses the camera plane.
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
        if max(x for x, _ in xy) < -2*width or min(x for x, _ in xy) > 3*width:
            return None
        if max(y for _, y in xy) < -2*height or min(y for _, y in xy) > 3*height:
            return None
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

    # Ground slab: fills the whole city footprint under roads and towers.
    draw_background_polygon([(-240.0,-240.0,-0.02),(240.0,-240.0,-0.02),
                             (240.0,240.0,-0.02),(-240.0,240.0,-0.02)], (39,48,58))

    # Road ribbons are background surfaces, not depth-sorted boxes. Sorting a
    # 400 m road prism by its average depth can incorrectly paint it over roofs.
    for road in world.roads:
        draw_background_polygon([
            (float(road['x_min']), float(road['y_min']), 0.025),
            (float(road['x_max']), float(road['y_min']), 0.025),
            (float(road['x_max']), float(road['y_max']), 0.025),
            (float(road['x_min']), float(road['y_max']), 0.025),
        ], (31, 38, 47))

    # Dashed road centrelines, projected through the same perspective camera.
    for x in world.avenue_x:
        for y0 in np.arange(-188.0, 189.0, 28.0):
            a, b = project((float(x), float(y0-2.5), 0.045)), project((float(x), float(y0+2.5), 0.045))
            if a and b:
                pygame.draw.line(screen, (190, 169, 105), a[:2], b[:2], 2)
    for y0 in world.street_y:
        for x0 in np.arange(-188.0, 189.0, 28.0):
            a, b = project((float(x0-2.5), float(y0), 0.045)), project((float(x0+2.5), float(y0), 0.045))
            if a and b:
                pygame.draw.line(screen, (190, 169, 105), a[:2], b[:2], 2)

    # Depot pads are also ground-level surfaces, so draw them before the towers.
    for idx, pad in enumerate(world.depot_helipads):
        p=np.asarray(pad,dtype=np.float64)
        pts=[(p[0]+2.8*math.cos(a),p[1]+2.8*math.sin(a),0.08) for a in np.linspace(0,2*math.pi,16,endpoint=False)]
        projected=[project(q) for q in pts]
        if all(q is not None for q in projected):
            pygame.draw.polygon(screen,[(30,170,225),(236,134,37),(180,75,225)][idx%3],[q[:2] for q in projected])
            pygame.draw.polygon(screen,(210,238,250),[q[:2] for q in projected],1)

    # Trees and wires go behind tower faces so a tower can visibly occlude them.
    for tree in world.trees:
        tx,ty = float(tree['pos'][0]),float(tree['pos'][1])
        radius,top = float(tree.get('r',3.2)),float(tree.get('h',8.0))
        trunk_bottom, trunk_top = project((tx,ty,0.1)),project((tx,ty,min(5.0,top*0.7)))
        canopy = project((tx,ty,top-radius))
        if trunk_bottom and trunk_top:
            pygame.draw.line(screen,(104,67,40),trunk_bottom[:2],trunk_top[:2],max(2,int(focal*0.45/max(1,trunk_bottom[2]))))
        if canopy:
            rad=int(np.clip(focal*radius/max(1.0,canopy[2]),3,55))
            pygame.draw.circle(screen,(26,112,55),canopy[:2],rad)
            pygame.draw.circle(screen,(69,156,83),canopy[:2],rad,1)
    for line in getattr(world,'powerlines',[]):
        a,b=project(line['p1']),project(line['p2'])
        if a and b:
            pygame.draw.line(screen,(190,198,207),a[:2],b[:2],2)
        for endpoint in (line['p1'], line['p2']):
            pole_bottom = project((float(endpoint[0]), float(endpoint[1]), 0.0))
            pole_top = project(endpoint)
            if pole_bottom and pole_top:
                pygame.draw.line(screen,(111,122,135),pole_bottom[:2],pole_top[:2],max(2,int(focal*0.16/max(pole_bottom[2],1.0))))

    # Buildings and actual roof clutter from the original V99 environment.
    for idx, house in enumerate(world.houses):
        x1,x2,y1,y2,top = [float(v) for v in house['aabb']]
        shade = int(88 + (idx % 5) * 7)
        facade_palette = [(56,76,96),(72,70,67),(48,82,91),(82,83,86),(61,74,91)]
        facade = facade_palette[idx % len(facade_palette)]
        add_box(x1,x2,y1,y2,0.05,top,facade)
        # Subtle blue-glass bands and floor-line details add depth without changing geometry.
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        levels = int(max(1, min(6, math.floor(top / 4.5))))
        for floor_i in range(levels):
            wz = min(top - 1.0, 2.0 + floor_i * 4.0)
            if wz <= 0.5:
                continue
            add_box(x1+1.1,x2-1.1,y2-0.035,y2+0.01,wz-0.36,wz+0.36,(29,74,97),outline=(19,38,48))
            add_box(x2-0.01,x2+0.035,y1+1.1,y2-1.1,wz-0.36,wz+0.36,(29,74,97),outline=(19,38,48))
        add_box(x1,x2,y2-0.13,y2+0.02,top-0.10,top-0.02,(111,124,137),outline=(45,56,66))
        # Roof helipad -- a visible turquoise disc built from a projected polygon.
        radius = 2.2
        cx,cy = (x1+x2)/2, (y1+y2)/2
        pad_pts = [(cx+radius*math.cos(a),cy+radius*math.sin(a),top+PHASE3_ROOF_PAD_TOP_OFFSET_M) for a in np.linspace(0,2*math.pi,16,endpoint=False)]
        add_polygon(pad_pts,(5,190,146),(10,75,66),1)
        for clutter in house.get('clutter',[]):
            ax,bx,ay,by,ctop = [float(v) for v in clutter['aabb']]
            add_box(ax,bx,ay,by,top+0.05,max(top+0.2,ctop),(128,139,151))

    # Paint all world faces back-to-front.
    draw_items.sort(key=lambda it: it[0], reverse=True)
    for _, points, color, outline, width_px in draw_items:
        pygame.draw.polygon(screen, color, points)
        if width_px:
            pygame.draw.polygon(screen, outline, points, width_px)

    # Render drone models with a readable visual halo. The halo is a screen-space
    # aid only; true airframe dimensions and coordinates remain unchanged.
    colors=[(40,235,255),(255,180,40),(232,118,255)]
    for idx, drone in enumerate(drones):
        pp=np.asarray(drone.airframe.p,dtype=np.float64).copy()
        if pp[2] < 0.22:
            pp[2] = 0.32  # visual lift only, prevents a docked model blending into the ground
        q=project(pp)
        if q is None: continue
        x,y,z=q
        rr=int(np.clip(focal*2.1/max(z,1.0),9,18))
        color=colors[idx%len(colors)]
        pygame.draw.circle(screen,(10,18,28),(x,y),rr+4)
        pygame.draw.circle(screen,color,(x,y),rr,2 if idx==selected_drone_idx else 1)
        # Crossed arm silhouette; enlarged in screen space only for readability.
        pygame.draw.line(screen,color,(x-rr-3,y),(x+rr+3,y),2)
        pygame.draw.line(screen,color,(x,y-rr-3),(x,y+rr+3),2)
        pygame.draw.circle(screen,(255,245,190),(x,y),max(2,rr//3))
        label=f'D{idx+1}  {float(drone.airframe.p[2]):.1f}m'
        font=_cached_software_font('consolas',12,bold=True)
        textsurf=font.render(label,True,color)
        # Stagger HUD labels in screen space so the three docked drones remain
        # individually identifiable in wide overview modes without moving their
        # actual world positions or flight states.
        tag=pygame.Rect(x+rr+5+(idx%2)*3,y-10+idx*23,textsurf.get_width()+8,20)
        pygame.draw.rect(screen,(7,14,23),tag,border_radius=3)
        pygame.draw.rect(screen,color,tag,1,border_radius=3)
        screen.blit(textsurf,(tag.x+4,tag.y+3))
        # Make an extended winch cable and suspended parcel visible in fallback mode.
        cable_len = float(getattr(drone.winch, 'cable_len', 0.0))
        if cable_len > 0.35:
            sway = np.asarray(getattr(drone.winch, 'sway_offset', np.zeros(2)), dtype=np.float64)
            hook = np.array([pp[0]+sway[0], pp[1]+sway[1], max(0.14,float(drone.airframe.p[2])-cable_len)])
            cable_top = project(pp)
            cable_bottom = project(hook)
            if cable_top and cable_bottom:
                pygame.draw.line(screen,(215,222,232),cable_top[:2],cable_bottom[:2],2)
                if bool(getattr(drone.winch, 'has_payload', True)):
                    size=max(3,min(8,rr//2))
                    parcel_rect=pygame.Rect(cable_bottom[0]-size,cable_bottom[1]-size,size*2,size*2)
                    pygame.draw.rect(screen,(244,184,38),parcel_rect)
                    pygame.draw.rect(screen,(70,48,12),parcel_rect,1)
                else:
                    # Empty hook only: released parcel remains on the roof below.
                    pygame.draw.circle(screen,(220,225,232),cable_bottom[:2],max(2,rr//5),1)

    # Detached delivered packages are independent world objects and remain on roofs.
    for delivered_pos in list(delivered_boxes or []):
        box = project(np.asarray(delivered_pos, dtype=np.float64))
        if box is None:
            continue
        bx, by, bz = box
        size = int(np.clip(focal * 0.18 / max(bz, 1.0), 2, 7))
        rect = pygame.Rect(bx-size, by-size, 2*size, 2*size)
        pygame.draw.rect(screen, (244,184,38), rect)
        pygame.draw.rect(screen, (70,48,12), rect, 1)

    target = project(np.asarray(active_target,dtype=np.float64))
    if target:
        x,y,_=target
        pygame.draw.circle(screen,(68,255,160),(x,y),9,2)
        pygame.draw.line(screen,(68,255,160),(x-13,y),(x+13,y),1)
        pygame.draw.line(screen,(68,255,160),(x,y-13),(x,y+13),1)

    # Keep visual fallback explicit rather than silently showing an empty image.
    label=f'3D SOFTWARE VIEW | {cam_mode} | Genesis frame unavailable'
    font=_cached_software_font('consolas',12,bold=True)
    surf=font.render(label,True,(235,245,255))
    bg=pygame.Rect(10,10,surf.get_width()+12,24)
    pygame.draw.rect(screen,(7,14,23),bg,border_radius=4)
    pygame.draw.rect(screen,(255,181,64),bg,1,border_radius=4)
    screen.blit(surf,(16,16))


def _software_render_v99_3d_scaled(mirror, screen, world, drones, camera_pos, camera_lookat,
                                  view_w, view_h, selected_drone_idx, active_target, cam_mode,
                                  fov_deg=48.0, delivered_boxes=None):
    """Run the software fallback at camera resolution, then scale into the viewport.

    Reuses both surfaces across frames so the fallback also benefits from a lower
    pixel count instead of drawing all geometry directly at the full cockpit size.
    """
    import pygame
    render_w, render_h = _camera_render_resolution(view_w, view_h)
    if render_w >= int(view_w) and render_h >= int(view_h):
        _software_render_v99_3d(screen, world, drones, camera_pos, camera_lookat,
                                 view_w, view_h, selected_drone_idx, active_target,
                                 cam_mode, fov_deg=fov_deg, delivered_boxes=delivered_boxes)
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
        selected_drone_idx, active_target, cam_mode, fov_deg=fov_deg, delivered_boxes=delivered_boxes
    )
    pygame.transform.scale(low, (int(view_w), int(view_h)), scaled)
    screen.blit(scaled, (0, 0))


def _render_genesis_mirror_impl(mirror,world,drones,traffic_engine,agent_pos,delivered_boxes,selected_drone_idx,
                          cam_mode,cam_yaw,cam_pitch,cam_focus,active_target,screen,view_w,view_h,font_small):
    """Synchronise Genesis scene proxies and blit its rendered 3D image."""
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
    # Wide overview shots intentionally look across the city. Applying a near-
    # target obstruction clamp to those shots collapses a 400 m panorama into a
    # camera position a few metres from the first tower along the view ray.
    # Let overview views use natural visual occlusion; only pull back close-follow
    # cameras when their view segment intersects geometry.
    if cam_mode not in ("CITY_OVERVIEW", "DEPOT_OVERVIEW"):
        cam_pos=resolve_genesis_camera_occlusion(cam_pos,cam_lookat,world)
    mirror['camera_pos'],mirror['camera_lookat']=cam_pos.copy(),cam_lookat.copy()
    if mirror.get('software_only', False):
        _software_render_v99_3d_scaled(mirror, screen, world, drones, cam_pos, cam_lookat,
                                        view_w, view_h, selected_drone_idx, active_target,
                                        cam_mode, fov_deg=48.0, delivered_boxes=delivered_boxes)
        return False
    mirror['camera'].set_pose(pos=tuple(float(x) for x in cam_pos),lookat=tuple(float(x) for x in cam_lookat))

    for d,e,cargo_e,cables in zip(drones,mirror['drone_entities'],mirror['cargo_entities'],mirror['cable_entities']):
        proxy_p = np.asarray(d.airframe.p, dtype=np.float32).copy()
        if proxy_p[2] < 0.22:
            proxy_p[2] = 0.32  # display-only docked-height correction; V99 state is untouched
        _genesis_pose(e, proxy_p, _rotation_matrix_to_quaternion_wxyz(d.airframe.R))
        if d.winch.has_payload:
            cable_points = _winch_cable_polyline_points(d.winch, d.airframe.p)
            payload_center = np.asarray(d.winch.nodes_pos[-1], dtype=np.float32).copy()
            if (not np.all(np.isfinite(payload_center)) or np.linalg.norm(payload_center) < 1e-6
                    or np.linalg.norm(payload_center[:2] - d.airframe.p[:2]) > d.winch.cable_len + 1.0):
                payload_center = d.airframe.p + np.array([
                    d.winch.sway_offset[0], d.winch.sway_offset[1],
                    -PHASE3_DRONE_HALF_HEIGHT_M - d.winch.cable_len - PHASE3_CARGO_HALF_HEIGHT_M], dtype=np.float32)
            # Apply the same display-only docked-height offset to the parcel as to
            # the drone so a stowed package does not render under the ground plane.
            if d.airframe.p[2] < 0.22:
                payload_center[2] = max(PHASE3_CARGO_HALF_HEIGHT_M,
                                        float(proxy_p[2] - 0.05 - d.winch.cable_len - PHASE3_CARGO_HALF_HEIGHT_M))
            cp = payload_center
        else:
            cable_points = _winch_cable_polyline_points(d.winch, d.airframe.p)
            # This visual proxy represents only the attached payload. Move it out
            # of view as soon as release is confirmed; the delivered-box entity is
            # rendered at the persistent world position.
            cp=np.array([0.0,0.0,-25.0-float(d.drone_id)*0.3],dtype=np.float32)
        _genesis_pose(cargo_e,cp)
        # Resample along the *actual curved cable* so the rendered line follows sway.
        seg_lengths = np.linalg.norm(np.diff(cable_points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        total_length = float(cumulative[-1])
        if total_length > 1e-6:
            sample_distances = np.linspace(0.0, total_length, len(cables), dtype=np.float32)
            cable_samples = []
            for sample_d in sample_distances:
                idx = min(len(seg_lengths)-1, int(np.searchsorted(cumulative, sample_d, side="right") - 1))
                frac = float((sample_d - cumulative[idx]) / max(1e-8, seg_lengths[idx]))
                cable_samples.append(cable_points[idx] + frac * (cable_points[idx + 1] - cable_points[idx]))
            cable_samples = np.asarray(cable_samples, dtype=np.float32)
        else:
            cable_samples = np.repeat(cable_points[:1], len(cables), axis=0)
        for j, node_e in enumerate(cables):
            center = cable_samples[j].copy()
            # Move cylinders halfway between adjacent samples, orienting the local
            # Z axis along each segment; their 0.30 m length overlaps the sample gap.
            if j < len(cable_samples)-1:
                direction = cable_samples[j+1] - cable_samples[j]
                center = 0.5 * (cable_samples[j+1] + cable_samples[j])
            else:
                direction = cable_samples[j] - cable_samples[j-1]
            center[2] = max(0.02, float(center[2]))
            _genesis_pose(node_e, center, _quat_wxyz_from_z_axis(direction))
    for v,e in zip(traffic_engine.vehicles,mirror['vehicle_entities']):
        vel=np.asarray(v['vel'],dtype=np.float32); yaw=math.atan2(float(vel[1]),float(vel[0])) if np.linalg.norm(vel[:2])>1e-4 else 0.0
        _genesis_pose(e,v['pos'],np.array([math.cos(yaw/2),0,0,math.sin(yaw/2)],dtype=np.float32))
    plane=world.civil_aircraft; yaw=math.atan2(float(plane['vel'][1]),float(plane['vel'][0]))
    _genesis_pose(mirror['plane_entity'],plane['pos'],np.array([math.cos(yaw/2),0,0,math.sin(yaw/2)],dtype=np.float32))
    ip=np.asarray(agent_pos,dtype=np.float32).copy(); ip[2]+=0.75
    _genesis_pose(mirror['intruder_entity'],ip)
    active_target_marker = np.asarray(active_target, dtype=np.float32).copy()
    active_target_marker[2] += PHASE3_ROOF_PAD_TOP_OFFSET_M + 0.30
    _genesis_pose(mirror['target_entity'], active_target_marker)
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
        # RGB is enough for normal frames. Segmentation is expensive to transfer
        # and analyse, so request it only periodically for blank-frame diagnostics.
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
            # Check the dynamic range on a coarse sample instead of scanning the
            # whole frame; the subsequent uint8 conversion necessarily visits all pixels.
            sample_max = float(np.nanmax(rgb[::16, ::16, :]))
            rgb = np.clip(rgb * 255.0 if sample_max <= 1.0 else rgb, 0, 255).astype(np.uint8)
        # Sample the image for blank-frame diagnostics. Full-frame statistics are
        # unnecessarily expensive and do not improve this heuristic materially.
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
                # A depot view should include ground, three drones, and nearby
                # towers; a city overview should include many independent objects.
                # Fewer than four visible labels strongly suggests the camera is
                # looking at an empty backdrop or only the ground plane.
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
                                        cam_mode, fov_deg=48.0, delivered_boxes=delivered_boxes)
        return False



def render_genesis_mirror(mirror, world, drones, traffic_engine, agent_pos, delivered_boxes, selected_drone_idx,
                          cam_mode, cam_yaw, cam_pitch, cam_focus, active_target, screen, view_w, view_h, font_small):
    """Render the camera at a capped rate while the authoritative simulation keeps stepping.

    The cached image covers only the camera viewport; the PFD, status panels and
    minimap are still drawn by the cockpit loop every simulation frame.
    Set SURVIVAL_RENDER_FPS (10..60, default 15) to trade visual update rate for speed.
    """
    now = time.monotonic()
    cache_key = (str(cam_mode), int(selected_drone_idx), bool(mirror.get('software_only', False)),
                 bool(mirror.get('show_hitboxes', False)))
    cached_surface = mirror.get('cached_view_surface')
    fps = float(mirror.get('render_fps', _camera_render_fps()))
    # The authoritative loop uses a fixed 0.016 s step (about 62.5 iterations/s).
    # Deterministically render once every N simulation iterations. Unlike a pure
    # wall-clock limiter this still skips frames when rendering itself is slow.
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

    if bool(mirror.get('show_hitboxes', False)):
        try:
            _draw_collision_hitbox_overlay(
                screen, world, traffic_engine, agent_pos, drones, delivered_boxes,
                mirror.get('camera_pos'), mirror.get('camera_lookat'), view_w, view_h, fov_deg=48.0
            )
        except Exception as overlay_exc:
            print(f"[HITBOX OVERLAY WARNING] {type(overlay_exc).__name__}: {overlay_exc}")

    # Cache the scene together with the optional overlay for the frames we skip.
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
# 20. MASTER INTERACTIVE SIMULATOR COCKPIT LOOP (60 FPS)
# =============================================================================
def _load_model_checkpoint(model, checkpoint_path, device):
    """Load a state-dict checkpoint with compatibility for older Torch versions."""
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    allowed_missing = ("kinematics_head.", "cargo_dynamics_head.", "force_balance_residual_head.",
                       "rotational_dynamics_residual_head.", "physics_uncertainty_head.",
                       "trajectory_forecast_head.", "action_transition_action_encoder.",
                       "action_conditioned_transition_head.", "environment_observation_transition_head.")
    unexpected_missing = [key for key in missing
                          if not (key.startswith(allowed_missing) or key in ("_force_balance_head_trained_state",
                                                                            "_rotational_dynamics_head_trained_state",
                                                                            "_physics_uncertainty_base_trained_state",
                                                                            "_force_uncertainty_trained_state",
                                                                            "_trajectory_forecast_head_trained_state",
                                                                            "_action_conditioned_transition_head_trained_state",
                                                                  "_environment_observation_transition_trained_state"))]
    if unexpected or unexpected_missing:
        raise RuntimeError(
            f"Checkpoint architecture mismatch in {checkpoint_path}: "
            f"unexpected={unexpected[:8]}, missing={unexpected_missing[:8]}")
    model._kinematics_head_checkpoint_trained = not any(k.startswith("kinematics_head.") for k in missing)
    model._cargo_dynamics_head_checkpoint_trained = not any(k.startswith("cargo_dynamics_head.") for k in missing)
    has_force_head = not any(k.startswith("force_balance_residual_head.") for k in missing)
    model._force_balance_head_checkpoint_trained = bool(has_force_head and model._force_balance_head_trained_state.item())
    has_rot_head = not any(k.startswith("rotational_dynamics_residual_head.") for k in missing)
    model._rotational_dynamics_head_checkpoint_trained = bool(
        has_rot_head and model._rotational_dynamics_head_trained_state.item())
    has_uncertainty_head = not any(k.startswith("physics_uncertainty_head.") for k in missing)
    model._physics_uncertainty_base_checkpoint_trained = bool(
        has_uncertainty_head and model._physics_uncertainty_base_trained_state.item())
    model._force_uncertainty_checkpoint_trained = bool(
        has_uncertainty_head and model._force_uncertainty_trained_state.item())
    has_trajectory_head = not any(k.startswith("trajectory_forecast_head.") for k in missing)
    model._trajectory_forecast_head_checkpoint_trained = bool(
        has_trajectory_head and model._trajectory_forecast_head_trained_state.item())
    has_action_transition_head = not any(
        k.startswith(("action_transition_action_encoder.", "action_conditioned_transition_head."))
        for k in missing)
    model._action_conditioned_transition_head_checkpoint_trained = bool(
        has_action_transition_head and model._action_conditioned_transition_head_trained_state.item())
    has_environment_transition_head = not any(k.startswith("environment_observation_transition_head.") for k in missing)
    model._environment_observation_transition_checkpoint_trained = bool(
        has_environment_transition_head and model._environment_observation_transition_trained_state.item())
    if (missing or not model._force_balance_head_checkpoint_trained
            or not model._rotational_dynamics_head_checkpoint_trained
            or not model._physics_uncertainty_base_checkpoint_trained
            or not model._trajectory_forecast_head_checkpoint_trained
            or not model._action_conditioned_transition_head_checkpoint_trained
            or not model._environment_observation_transition_checkpoint_trained):
        print("[CHECKPOINT COMPAT] One or more auxiliary physics heads are missing or untrained; Phase 3 and, when needed, Phase 1/2 BEMT transition/rotational training are required before this checkpoint is fully physics-ready.", flush=True)
    return model


def _launch_phase3_worker(base_checkpoint, output_checkpoint):
    """Run Genesis Phase 3 in a child process to isolate GPU/Genesis lifecycle from the GUI."""
    worker_env = os.environ.copy()
    worker_env["SURVIVAL_PHASE3_WORKER"] = "1"
    worker_env["SURVIVAL_PHASE3_BASE_CHECKPOINT"] = os.path.abspath(base_checkpoint)
    worker_env["SURVIVAL_PHASE3_OUTPUT_CHECKPOINT"] = os.path.abspath(output_checkpoint)
    script_path = os.path.abspath(__file__)
    print("\n[PHASE 3] Launching isolated Genesis worker process...", flush=True)
    try:
        result = subprocess.run([sys.executable, script_path], env=worker_env, check=False)
    except OSError as exc:
        print(f"[PHASE 3 WARNING] Could not launch worker process: {exc}", flush=True)
        return False
    return result.returncode == 0 and os.path.isfile(output_checkpoint)


def run_all_physics_simulator():
    if pygame is None:
        raise RuntimeError("pygame is required for the interactive simulator. Install it with: python -m pip install pygame")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    force_retrain = os.environ.get("SURVIVAL_FORCE_RETRAIN", "0").strip().lower() in {"1", "true", "yes"}
    phase3_only = os.environ.get("SURVIVAL_PHASE3_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
    if phase3_only:
        if not os.path.isfile(MODEL_WEIGHTS_FILE):
            raise FileNotFoundError(
                f"SURVIVAL_PHASE3_ONLY=1 requires an existing Phase-2 checkpoint: {MODEL_WEIGHTS_FILE}")
        if force_retrain:
            print("[POLICY] SURVIVAL_PHASE3_ONLY overrides a stale SURVIVAL_FORCE_RETRAIN flag; Phase 1/2 will be skipped.", flush=True)
        force_retrain = False
        print(f"[POLICY] Phase-3-only mode: reusing existing supervised checkpoint {MODEL_WEIGHTS_FILE}.", flush=True)
    phase3_enabled = os.environ.get("SURVIVAL_ENABLE_PHASE3", "1").strip().lower() not in {"0", "false", "no", "off"}
    force_phase3 = os.environ.get("SURVIVAL_FORCE_PHASE3", "0").strip().lower() in {"1", "true", "yes"}
    force_phase3 = force_phase3 or force_retrain
    phase3_checkpoint = os.path.abspath(PHASE3_CHECKPOINT_FILE)
    legacy_phase3_checkpoint = os.path.abspath(LEGACY_PHASE3_CHECKPOINT_FILE)
    # Reuse the latest accepted V38.7 refinement as the warm-start for this
    # state-machine-only merge when present; never overwrite that file. If absent,
    # use the accepted V38.1 refinement or fall back to the Phase-2 supervised model.
    v38_1_checkpoint = os.path.abspath(PREVIOUS_V38_1_PHASE3_CHECKPOINT_FILE)
    v38_7_checkpoint = os.path.abspath(PREVIOUS_V38_7_MEMORY_EFFICIENT_PHASE3_CHECKPOINT_FILE)
    if not force_retrain and os.path.isfile(phase3_checkpoint):
        phase3_base_checkpoint = phase3_checkpoint
    elif not force_retrain and os.path.isfile(v38_7_checkpoint):
        phase3_base_checkpoint = v38_7_checkpoint
        print(f"[PHASE 3] Warm-starting V38.9 from the accepted V38.7 memory-efficient checkpoint: {v38_7_checkpoint}", flush=True)
    elif not force_retrain and os.path.isfile(v38_1_checkpoint):
        phase3_base_checkpoint = v38_1_checkpoint
        print(f"[PHASE 3] Warm-starting V38.9 from accepted V38.1 checkpoint: {v38_1_checkpoint}", flush=True)
    else:
        # Phase-3-only recovery after an interrupted fine-tune starts from the
        # existing Phase-2 model if no older accepted refinement is available.
        phase3_base_checkpoint = os.path.abspath(MODEL_WEIGHTS_FILE)
    used_phase3_checkpoint = False

    if force_retrain or not os.path.exists(MODEL_WEIGHTS_FILE):
        if force_retrain:
            print("[POLICY] SURVIVAL_FORCE_RETRAIN is enabled: running Phase 1 harvesting and Phase 2 supervised training.", flush=True)
        model, device = harvest_and_train_metropolitan()
    else:
        model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
        if phase3_enabled and os.path.isfile(phase3_checkpoint) and not force_phase3:
            candidate_checkpoint = phase3_checkpoint
        elif phase3_enabled and not force_retrain:
            # Do not fall back to a Phase-2 checkpoint known to have learned only
            # TRANSIT if the new Phase-3 rollout is rejected by its delivery gate.
            # Load the newest existing compatible refinement as the live fallback,
            # while still running the V38 Phase 3 rollout when its dedicated checkpoint
            # does not yet exist.
            # Resolve fallback entries through the constants declared in section 17.
            # Older lowercase aliases were undefined and could crash startup after training.
            safe_fallback_candidates = [
                phase3_checkpoint,
                os.path.abspath(PREVIOUS_V38_7_MEMORY_EFFICIENT_PHASE3_CHECKPOINT_FILE),
                os.path.abspath(PREVIOUS_V38_1_PHASE3_CHECKPOINT_FILE),
                os.path.abspath(PREVIOUS_DELIVERY_DESCENT_PHASE3_CHECKPOINT_FILE),
                os.path.abspath(PREVIOUS_DEEP_AUDIT_PHASE3_CHECKPOINT_FILE),
                os.path.abspath(PREVIOUS_COVERAGE_HOVER_PHASE3_CHECKPOINT_FILE),
                os.path.abspath(PREVIOUS_ROOF_CONTACT_PHASE3_CHECKPOINT_FILE),
                os.path.abspath(PREVIOUS_CURRICULUM_PHASE3_CHECKPOINT_FILE),
                os.path.abspath(PREVIOUS_PRECISION_PHASE3_CHECKPOINT_FILE),
                os.path.abspath(PREVIOUS_WIND_SENSOR_PHASE3_CHECKPOINT_FILE),
                os.path.abspath(PREVIOUS_WIND_PHASE3_CHECKPOINT_FILE),
                legacy_phase3_checkpoint,
            ]
            candidate_checkpoint = next(
                (p for p in safe_fallback_candidates if os.path.isfile(p)),
                MODEL_WEIGHTS_FILE,
            )
        else:
            candidate_checkpoint = MODEL_WEIGHTS_FILE
        try:
            _load_model_checkpoint(model, candidate_checkpoint, device)
            # A checkpoint counts as ready only if every required auxiliary physics
            # head has trained weights. V38.23 checkpoints predate cargo prediction;
            # loading one must trigger Phase 3 rather than silently skipping the new head.
            used_phase3_checkpoint = (
                os.path.abspath(candidate_checkpoint) == phase3_checkpoint
                and not force_phase3
                and bool(getattr(model, "_kinematics_head_checkpoint_trained", False))
                and bool(getattr(model, "_cargo_dynamics_head_checkpoint_trained", False))
                and bool(getattr(model, "_force_balance_head_checkpoint_trained", False))
                and bool(getattr(model, "_rotational_dynamics_head_checkpoint_trained", False))
                and bool(getattr(model, "_physics_uncertainty_base_checkpoint_trained", False))
                and bool(getattr(model, "_force_uncertainty_checkpoint_trained", False))
                and bool(getattr(model, "_trajectory_forecast_head_checkpoint_trained", False))
                and bool(getattr(model, "_action_conditioned_transition_head_checkpoint_trained", False))
                and bool(getattr(model, "_environment_observation_transition_checkpoint_trained", False))
            )
            if used_phase3_checkpoint:
                print(f"\nLoaded current V38 Genesis-refined model from {candidate_checkpoint}.", flush=True)
            elif os.path.abspath(candidate_checkpoint) == os.path.abspath(MODEL_WEIGHTS_FILE):
                print(f"\nLoaded Phase-2 supervised model from {MODEL_WEIGHTS_FILE}.", flush=True)
            else:
                print(f"\nLoaded existing refinement as checkpoint fallback from {candidate_checkpoint}; Phase 3 will still run.", flush=True)
        except Exception as e:
            if os.path.abspath(candidate_checkpoint) != os.path.abspath(MODEL_WEIGHTS_FILE):
                print(f"[PHASE 3 WARNING] Existing Phase-3 checkpoint could not be loaded ({e}); falling back to Phase 2.", flush=True)
                try:
                    _load_model_checkpoint(model, MODEL_WEIGHTS_FILE, device)
                except Exception as base_exc:
                    print(f"[WARNING] Base weights also failed ({base_exc}); running Phases 1 and 2 again.", flush=True)
                    model, device = harvest_and_train_metropolitan()
                    force_phase3 = True
            else:
                print(f"[WARNING] Weights mismatch ({e}); running Phases 1 and 2 again.", flush=True)
                model, device = harvest_and_train_metropolitan()
                force_phase3 = True

    # Rotational residual targets require the authoritative BEMT harvester's actual
    # motor-torque output. Genesis Phase 3's force-only proxy cannot train this head.
    # If an older checkpoint lacks the head or its persistent trained marker, rebuild
    # Phase 1/2 instead of repeatedly running Phase 3 with random rotational weights.
    if phase3_enabled and not bool(getattr(model, "_rotational_dynamics_head_checkpoint_trained", False)):
        print("[ROTATIONAL EULER TRAINING] Loaded checkpoint lacks a trained BEMT rotational-residual head; rerunning Phase 1/2 to collect correctly aligned torque targets.", flush=True)
        model, device = harvest_and_train_metropolitan()
        force_phase3 = True
        used_phase3_checkpoint = False
        phase3_base_checkpoint = os.path.abspath(MODEL_WEIGHTS_FILE)

    if phase3_enabled and not bool(getattr(model, "_physics_uncertainty_base_checkpoint_trained", False)):
        print("[UNCERTAINTY TRAINING] Loaded checkpoint lacks calibrated Phase-1/2 uncertainty weights; rerunning source-matched supervised training.", flush=True)
        model, device = harvest_and_train_metropolitan()
        force_phase3 = True
        used_phase3_checkpoint = False
        phase3_base_checkpoint = os.path.abspath(MODEL_WEIGHTS_FILE)

    if phase3_enabled and not bool(getattr(model, "_trajectory_forecast_head_checkpoint_trained", False)):
        print("[TRAJECTORY FORECAST TRAINING] Loaded checkpoint lacks a trained multi-horizon forecast head; rerunning Phase 1/2 to collect paired future-state targets.", flush=True)
        model, device = harvest_and_train_metropolitan()
        force_phase3 = True
        used_phase3_checkpoint = False
        phase3_base_checkpoint = os.path.abspath(MODEL_WEIGHTS_FILE)

    if phase3_enabled and not bool(getattr(model, "_action_conditioned_transition_head_checkpoint_trained", False)):
        print("[ACTION-CONDITIONED TRANSITION TRAINING] Loaded checkpoint lacks a trained paired-action transition head; rerunning Phase 1/2 before Phase 3.", flush=True)
        model, device = harvest_and_train_metropolitan()
        force_phase3 = True
        used_phase3_checkpoint = False
        phase3_base_checkpoint = os.path.abspath(MODEL_WEIGHTS_FILE)

    if phase3_enabled and not bool(getattr(model, "_environment_observation_transition_checkpoint_trained", False)):
        print("[ENVIRONMENT TRANSITION TRAINING] Loaded checkpoint lacks trained next-step wind/cable/icing predictions; rerunning Phase 1/2 with aligned same-drone observations.", flush=True)
        model, device = harvest_and_train_metropolitan()
        force_phase3 = True
        used_phase3_checkpoint = False
        phase3_base_checkpoint = os.path.abspath(MODEL_WEIGHTS_FILE)

    if phase3_enabled and not used_phase3_checkpoint:
        kinematics_needs_training = not bool(getattr(model, "_kinematics_head_checkpoint_trained", False))
        cargo_dynamics_needs_training = not bool(getattr(model, "_cargo_dynamics_head_checkpoint_trained", False))
        force_balance_needs_training = not bool(getattr(model, "_force_balance_head_checkpoint_trained", False))
        rotational_dynamics_needs_training = not bool(getattr(model, "_rotational_dynamics_head_checkpoint_trained", False))
        force_uncertainty_needs_training = not bool(getattr(model, "_force_uncertainty_checkpoint_trained", False))
        trajectory_forecast_needs_training = not bool(getattr(model, "_trajectory_forecast_head_checkpoint_trained", False))
        action_transition_needs_training = not bool(getattr(model, "_action_conditioned_transition_head_checkpoint_trained", False))
        environment_transition_needs_training = not bool(getattr(model, "_environment_observation_transition_checkpoint_trained", False))
        predictor_needs_training = (kinematics_needs_training or cargo_dynamics_needs_training
                                    or force_balance_needs_training or rotational_dynamics_needs_training
                                    or force_uncertainty_needs_training or trajectory_forecast_needs_training
                                    or action_transition_needs_training or environment_transition_needs_training)
        phase3_needed = force_phase3 or not os.path.isfile(phase3_checkpoint) or predictor_needs_training
        if predictor_needs_training and phase3_enabled:
            missing_heads = []
            if kinematics_needs_training:
                missing_heads.append("aircraft kinematics")
            if cargo_dynamics_needs_training:
                missing_heads.append("suspended-cargo dynamics")
            if force_balance_needs_training:
                missing_heads.append("Newtonian force-balance residual")
            if rotational_dynamics_needs_training:
                missing_heads.append("rotational Euler residual (requires Phase 1/2 BEMT targets)")
            if force_uncertainty_needs_training:
                missing_heads.append("force-residual uncertainty (requires Phase 3 targets)")
            if trajectory_forecast_needs_training:
                missing_heads.append("multi-horizon future position/velocity forecast (requires Phase 1/2 delayed BEMT targets)")
            if action_transition_needs_training:
                missing_heads.append("action-conditioned one-step state transition (requires paired guidance/FSM/winch actions and measured BEMT deltas)")
            if environment_transition_needs_training:
                missing_heads.append("next-step wind/cable/icing transition (requires valid consecutive same-drone observations)")
            print(f"[PHYSICS-GUIDED TRAINING] Loaded checkpoint needs training for {', '.join(missing_heads)} prediction head(s); running physics-guided training with source-matched targets.", flush=True)
        if phase3_needed:
            if os.path.isfile(phase3_checkpoint):
                backup = phase3_checkpoint + ".backup_before_phase3_retrain"
                if not os.path.isfile(backup):
                    try:
                        shutil.copy2(phase3_checkpoint, backup)
                        print(f"[CHECKPOINT BACKUP] Previous Phase-3 model copied to {backup}", flush=True)
                    except OSError as exc:
                        print(f"[PHASE 3 WARNING] Could not back up existing Phase-3 model: {exc}", flush=True)
            # V38.9 writes a separate checkpoint. Warm-start from its own accepted
            # refinement, then the accepted V38.1 checkpoint, or Phase-2 if neither exists.
            # All older checkpoints remain untouched and are not silently promoted.
            succeeded = _launch_phase3_worker(phase3_base_checkpoint, phase3_checkpoint)
            if succeeded:
                try:
                    _load_model_checkpoint(model, phase3_checkpoint, device)
                    used_phase3_checkpoint = True
                    print("[PHASE 3] Genesis refinement complete; using the refined policy for live flight.", flush=True)
                except Exception as exc:
                    print(f"[PHASE 3 WARNING] Refined checkpoint could not be loaded ({exc}); keeping the Phase-2 policy.", flush=True)
            else:
                print("[PHASE 3 WARNING] Genesis refinement did not produce a new checkpoint. Keeping the policy already loaded in memory (compatible prior refinement when available, otherwise Phase 2); check the worker traceback.", flush=True)
                print("[STARTUP NOTE] Phase 3 did not produce an accepted checkpoint. The visual simulator will open in fallback/debug mode after the worker has fully exited; this is not a successful Phase 3 training result.", flush=True)
        else:
            print(f"[PHASE 3] Existing refined checkpoint is ready: {phase3_checkpoint}", flush=True)
    elif not phase3_enabled:
        print("[PHASE 3] Disabled by SURVIVAL_ENABLE_PHASE3=0; using the currently loaded checkpoint without Genesis refinement.", flush=True)
        print("[STARTUP NOTE] The Pygame window will open now because Phase 3 was explicitly skipped. Remove SURVIVAL_ENABLE_PHASE3 from the environment to run the full configured training/refinement pipeline before opening the simulator.", flush=True)

    model.eval()
    # Define viewport dimensions without opening the visible Pygame window yet.
    # Genesis scene/kernel/camera initialisation must finish before the cockpit appears.
    VIEW_W, VIEW_H = 760, 740
    SIDEBAR_W = 440
    W, H = VIEW_W + SIDEBAR_W, VIEW_H

    world = MetropolitanCityGridWorld400m()
    corridor_planner = VoxelOctreeCorridorPlanner(world)
    collision_inventory = world.collision_inventory()
    print(f"[COLLISION AUDIT] Static hitboxes: {collision_inventory}", flush=True)
    if collision_inventory.get("buildings") != len(world.houses):
        raise RuntimeError("Collision audit failed: one or more buildings lack a hitbox.")
    expected_clutter = sum(len(h.get("clutter", [])) for h in world.houses)
    if collision_inventory.get("rooftop_clutter") != expected_clutter:
        raise RuntimeError("Collision audit failed: one or more rooftop objects lack a hitbox.")
    if (collision_inventory.get("trees") != len(world.trees)
            or collision_inventory.get("powerlines") != len(world.powerlines)
            or collision_inventory.get("powerline_support_poles") != 2 * len(world.powerlines)):
        raise RuntimeError("Collision audit failed: a tree, powerline or support pole lacks a hitbox.")
    expected_supports = 1 + len(world.roads) + len(world.depot_helipads) + 2 * len(world.houses)
    if collision_inventory.get("support_surface_total") != expected_supports:
        raise RuntimeError("Collision audit failed: a ground, road, depot pad, roof or rooftop drop-pad surface is missing.")
    weather_engine = DynamicUrbanWeatherEngine()
    traffic_engine = DynamicRoadVehicleTrafficEngine(world.avenue_x, world.street_y)
    print(f"[COLLISION AUDIT] Dynamic objects prepared: road_vehicles={len(traffic_engine.vehicles)} | civil_aircraft=1 | "
          f"ground_intruder=1 body+1 safety envelope | swarm_drones={NUM_SWARM_DRONES} | "
          f"cable_polylines={NUM_SWARM_DRONES} | detached_package_hitboxes=spawned_on_release", flush=True)

    sensors_physics = FullPhysicsSensorSuite()
    assurance = SurfaceAndWireAssuranceSuite(world)
    lidar_suite = MacroPerceptionLiDARSuite(world)
    sensor_suite = AdvancedMultiModalSensorSuite(world)
    ground_agent = DynamicGroundIntruderAgent(init_pos=[-80.0, 80.0, 0.0])
    # Define a valid initial position before any startup validator, renderer, or
    # sensor/diagnostic path can consume it. The loop refreshes it from step().
    agent_pos = ground_agent.p.copy()
    # Independent turbulence state per aircraft. A selected camera drone must not
    # determine the weather disturbance experienced by every other drone.
    dryden_wind_gens = [DrydenAtmosphericTurbulenceModel(w20_m_s=3.4)
                        for _ in range(NUM_SWARM_DRONES)]

    dispatcher = SwarmMissionDispatcher(world.manifest_addresses, world.depot_helipads)

    drones = []
    swarm_colors = [(0, 240, 255), (255, 180, 40), (220, 100, 255)]
    for i in range(NUM_SWARM_DRONES):
        init_pos = world.depot_helipads[i].copy()
        init_pos[2] = 0.06
        d_inst = SwarmDroneInstance(i, init_pos, world, dispatcher)
        drones.append(d_inst)

    startup_collision_report = validate_runtime_collision_coverage(
        world, traffic_engine, ground_agent.p, drones, delivered_boxes=[]
    )
    print("[COLLISION AUDIT] Runtime hitboxes validated: "
          f"{startup_collision_report['obstacle_total']} obstacle colliders + "
          f"{startup_collision_report['support_surface_total']} support surfaces; "
          f"families={startup_collision_report['counts']}", flush=True)

    # Camera/render-only startup guard: the V16 simulation can run even if Genesis
    # cannot initialise; the perspective Pygame renderer then draws the same world.
    try:
        genesis_mirror = build_genesis_3d_mirror(
            world=world, drones=drones, traffic_engine=traffic_engine,
            ground_agent=ground_agent, view_w=VIEW_W, view_h=VIEW_H
        )
    except Exception as exc:
        print(f"[3D VIEW WARNING] Genesis could not initialise ({type(exc).__name__}: {exc}); using software 3D renderer.")
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

    # The visible cockpit starts only after Genesis scene build, reset and its initial
    # camera priming attempt have returned (or the software renderer fallback was set).
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Survival Core V38.35 — Gated Multi-Step Transition Evaluation")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("consolas", 17, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 12, bold=True)
    print("[VISUALIZER START] Genesis initialization/fallback setup complete; opening cockpit now.", flush=True)

    selected_drone_idx = 0
    cam_yaw, cam_pitch = 0.75, 0.45
    cam_zoom = 1.0
    camera_dragging = False
    camera_last_mouse = None
    genesis_mirror['cam_zoom'] = cam_zoom
    genesis_mirror['show_hitboxes'] = False
    cam_focus = drones[0].airframe.p.copy()
    cam_mode = "DEPOT_OVERVIEW"
    running = True

    gust_timer = 0.0
    active_gust = np.zeros(3, dtype=np.float32)
    inject_spoof = False
    flight_time = 0.0
    # Passive validation of the auxiliary one-step predictor. Keep accumulators
    # per drone: this code runs inside the drone loop, so scalar accumulators were
    # reset once per drone and produced one pooled metric followed by samples=1.
    # These errors are measured after the physical step and never control flight.
    kinematics_eval_count = np.zeros(len(drones), dtype=np.int64)
    kinematics_vel_sq_error_sum = np.zeros(len(drones), dtype=np.float64)
    kinematics_accel_sq_error_sum = np.zeros(len(drones), dtype=np.float64)
    kinematics_equation_resid_sq_error_sum = np.zeros(len(drones), dtype=np.float64)
    cargo_eval_count = np.zeros(len(drones), dtype=np.int64)
    cargo_sway_sq_error_sum = np.zeros(len(drones), dtype=np.float64)
    cargo_sway_rate_sq_error_sum = np.zeros(len(drones), dtype=np.float64)
    cargo_tension_sq_error_sum = np.zeros(len(drones), dtype=np.float64)
    uncertainty_eval_count = np.zeros(len(drones), dtype=np.int64)
    # sigma sums: vel m/s, accel m/s^2, cargo sway m, cargo sway-rate m/s,
    # cable tension N, force residual N, rotational torque residual N*m.
    uncertainty_sigma_sum = np.zeros((len(drones), 7), dtype=np.float64)
    uncertainty_base_ready = bool(getattr(model, "_physics_uncertainty_base_checkpoint_trained", False))
    force_uncertainty_ready = bool(getattr(model, "_force_uncertainty_checkpoint_trained", False))
    kinematics_predictor_ready = bool(getattr(model, "_kinematics_head_checkpoint_trained", False))
    cargo_dynamics_predictor_ready = bool(getattr(model, "_cargo_dynamics_head_checkpoint_trained", False))
    force_balance_predictor_ready = bool(getattr(model, "_force_balance_head_checkpoint_trained", False))
    rotational_dynamics_predictor_ready = bool(getattr(model, "_rotational_dynamics_head_checkpoint_trained", False))
    # V38.40 optional passive same-action validation against the authoritative live
    # BEMT/RK4 step. It conditions the transition model on the exact post-gating
    # guidance/FSM/winch command currently passed to the controller, then compares its
    # prediction with the measured next state. This is not a counterfactual alternate-
    # action test and never changes commands. Disabled by default to avoid added latency.
    authoritative_shadow_requested = os.environ.get(
        "SURVIVAL_AUTHORITATIVE_TRANSITION_SHADOW", "0").strip().lower() in {"1", "true", "yes", "on"}
    authoritative_shadow_stride = int(np.clip(int(os.environ.get(
        "SURVIVAL_AUTHORITATIVE_SHADOW_STRIDE", "30")), 1, 6000))
    authoritative_shadow_output_path = os.path.abspath(os.environ.get(
        "SURVIVAL_AUTHORITATIVE_SHADOW_REPORT", "authoritative_transition_shadow_v38_42.json"))
    authoritative_shadow_head_ready = bool(getattr(
        model, "_action_conditioned_transition_head_checkpoint_trained", False))
    authoritative_shadow_enabled = bool(authoritative_shadow_requested and authoritative_shadow_head_ready)
    authoritative_shadow_block_reason = None
    if authoritative_shadow_requested and not authoritative_shadow_head_ready:
        authoritative_shadow_block_reason = "checkpoint has no trained action-conditioned transition head"
        print(f"[AUTHORITATIVE SHADOW VALIDATION BLOCKED] {authoritative_shadow_block_reason}; live control remains unchanged.", flush=True)
    elif authoritative_shadow_enabled:
        print(f"[AUTHORITATIVE SHADOW VALIDATION] enabled; per-drone stride={authoritative_shadow_stride} steps; "
              f"report={authoritative_shadow_output_path}; passive only, no actuation connection.", flush=True)
    authoritative_shadow_eval_count = np.zeros(len(drones), dtype=np.int64)
    authoritative_shadow_invalid_count = np.zeros(len(drones), dtype=np.int64)
    # Count scheduled probes separately from valid/failed samples. This makes a
    # zero-sample report distinguish "never probed" from "probed but rejected".
    authoritative_shadow_probe_count = np.zeros(len(drones), dtype=np.int64)
    authoritative_shadow_sq_error = np.zeros((len(drones), 3), dtype=np.float64)
    authoritative_shadow_sq_baseline = np.zeros((len(drones), 3), dtype=np.float64)
    authoritative_shadow_action_norm_sum = np.zeros(len(drones), dtype=np.float64)
    authoritative_shadow_pre_gate_guidance_norm_sum = np.zeros(len(drones), dtype=np.float64)
    authoritative_shadow_phase_gate_sample_count = np.zeros(len(drones), dtype=np.int64)
    authoritative_shadow_phase_gate_suppressed_count = np.zeros(len(drones), dtype=np.int64)
    authoritative_shadow_wind_sum = np.zeros(len(drones), dtype=np.float64)
    authoritative_shadow_loaded_count = np.zeros(len(drones), dtype=np.int64)
    authoritative_shadow_phase_counts = [dict() for _ in drones]
    authoritative_shadow_fsm_counts = [dict() for _ in drones]
    authoritative_shadow_winch_counts = [dict() for _ in drones]
    authoritative_shadow_windows = []

    def _write_authoritative_shadow_report(status="RUNNING"):
        """Atomically persist only observed paired metrics; JSON is useful mid-run too."""
        payload = {
            "version": SURVIVAL_CORE_VERSION,
            "status": status,
            "evaluation_mode": "passive live same-action one-step validation against authoritative BEMT/RK4 state transitions",
            "enabled": bool(authoritative_shadow_enabled),
            "requested": bool(authoritative_shadow_requested),
            "trained_action_transition_head": bool(authoritative_shadow_head_ready),
            "report_path": authoritative_shadow_output_path,
            "physics_timestep_s": 0.016,
            "probe_stride_steps_per_drone": int(authoritative_shadow_stride),
            "metric_contract": {
                "baseline_prediction_source": "latest observed velocity features from the same flattened observation passed to the learned model; decoded by multiplying features by 8.0",
                "shared_scoring_rule": "learned transition and baseline are scored against the same valid paired authoritative post-step target rows; RMSE is per vector component",
                "position_baseline": "observed velocity times physics timestep; zero predicted velocity delta and body-rate delta",
                "hidden_true_velocity_used_by_baseline": False,
            },
            "action_input_contract": {
                "training_guidance": "teacher guidance target coordinates are acceleration targets divided by 12 before being passed to the mixer",
                "live_guidance": "live model guidance output is scaled by 0.15, then zeroed in controller-owned mission phases before being passed to the mixer",
                "runtime_diagnostics": "each scored window reports pre-phase-gate guidance residual norm, effective guidance norm, and whether a nonzero residual was suppressed by the phase gate",
                "actuation_modified_by_audit": False,
            },
            "windows": list(authoritative_shadow_windows),
            "total_probe_attempts": int(sum(int(w.get("probe_attempts", 0)) for w in authoritative_shadow_windows)
                                           + int(np.sum(authoritative_shadow_probe_count))),
            "total_valid_samples": int(sum(int(w.get("valid_samples", 0)) for w in authoritative_shadow_windows)
                                          + int(np.sum(authoritative_shadow_eval_count))),
            "total_invalid_or_masked_samples": int(sum(int(w.get("masked_or_failed_probes", 0)) for w in authoritative_shadow_windows)
                                                      + int(np.sum(authoritative_shadow_invalid_count))),
            "latest_window_by_drone": {
                str(i + 1): next((w for w in reversed(authoritative_shadow_windows)
                                  if int(w.get("drone_id", -1)) == i + 1), None)
                for i in range(len(drones))
            },
            "live_control_authorized": False,
            "actuation_modified_by_evaluator": False,
            "counterfactual_alternate_action_test": False,
            "limitations": [
                "The transition prediction is compared with the next measured state after the exact high-level guidance/FSM/winch inputs passed by the live loop; it does not test a different action on a cloned simulator state.",
                "Teacher guidance features are acceleration targets divided by 12; live guidance output is scaled by 0.15 and hard-zeroed in controller-owned phases. Per-window raw/effective norms and gate suppression counts expose the resulting action-distribution shift.",
                "The constant-velocity baseline uses observed velocity features, not hidden simulator velocity, to avoid giving the baseline privileged state information.",
                "The physical target comes from the authoritative custom BEMT/RK4 airframe step in the actual simulator; Genesis is used as a visual mirror, not as this transition target source.",
                "Samples near surfaces, invalid states, parachute transitions and transitions rejected by the same target validity mask are excluded.",
                "This validation is one-step only; it does not prove multi-step rollout accuracy or authorize predictive control."
            ]
        }
        try:
            parent = os.path.dirname(authoritative_shadow_output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            temp_path = authoritative_shadow_output_path + f".tmp-{os.getpid()}"
            with open(temp_path, "w", encoding="utf-8") as dst:
                json.dump(_action_transition_json_safe(payload), dst, indent=2, sort_keys=True, allow_nan=False)
            os.replace(temp_path, authoritative_shadow_output_path)
        except (OSError, TypeError, ValueError) as exc:
            print(f"[AUTHORITATIVE SHADOW REPORT WARNING] could not write report: {exc}", flush=True)

    if authoritative_shadow_requested:
        _write_authoritative_shadow_report("BLOCKED" if authoritative_shadow_block_reason else "RUNNING")

    simulation_wall_start = time.perf_counter()
    last_metrics_wall_time = simulation_wall_start
    last_metrics_sim_time = 0.0
    simulation_step_count = 0
    last_metrics_step_count = 0
    delivered_boxes = deque(maxlen=120)
    # Track the raw model class separately from the safety-filtered command. The
    # latter legitimately defaults to TRANSIT in non-delivery phases when model
    # confidence is low; conflating them can make a healthy fallback look like a
    # collapsed policy.
    policy_raw_action_histogram = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    policy_action_histogram = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    last_policy_metrics_time = 0.0

    print("\n================ V38.40 AUTHORITATIVE LIVE TRANSITION SHADOW VALIDATION + PHYSICS-GUIDED CONTROLS ONLINE ================")
    print(f"[CONTROL MODE] neural guidance residual: {'DISABLED (deterministic controller test)' if NEURAL_GUIDANCE_DISABLED else 'enabled'}", flush=True)
    print(" [AUTHORITATIVE CORE] : 8-DOF BEMT RK4 Dynamics & PBD Tether Simulation")
    print(" [PHYSICS-GUIDED HEAD]: Kinematics/cargo/physics residuals and multi-horizon forecast; diagnostic only")
    print(" [ACTION TRANSITION]: Paired high-level action -> measured one-step state delta; diagnostic only")
    print(f" [FORCE RESIDUAL STATUS]: {'Phase-3-trained (diagnostic only)' if force_balance_predictor_ready else 'UNTRAINED — Phase 3 required before physics-ready checkpoint'}")
    print(f" [ROTATIONAL RESIDUAL STATUS]: {'BEMT-trained (diagnostic only)' if rotational_dynamics_predictor_ready else 'UNTRAINED — Phase 1/2 BEMT torque targets required'}")
    print(f" [PREDICTION UNCERTAINTY]: {'Phase 1/2 trained; holdout calibration must be inspected (diagnostic only)' if uncertainty_base_ready else 'UNTRAINED — do not interpret sigma outputs'}; force uncertainty {'Phase-3 trained; cross-simulator calibration unverified' if force_uncertainty_ready else 'not yet trained'}")
    print(" [139-DIM SENSORS]    : Live Quantum Gravimeter, EKF15, LiDAR & Acoustics")
    print(" [WIND VECTOR SENSOR] : Per-drone 3-D air-data + GNSS/INS estimate; policy/controller use sensed wind, not truth")
    print(" [CAMERA SYSTEM 3]   : 5 Perspective Modes, Zoom, Orbit & Robust 3D Fallback")
    print(" [COLLISION VISUAL]  : Press K to toggle authoritative hitboxes/support surfaces")
    render_w = int(genesis_mirror.get('render_width', _camera_render_resolution(VIEW_W, VIEW_H)[0]))
    render_h = int(genesis_mirror.get('render_height', _camera_render_resolution(VIEW_W, VIEW_H)[1]))
    render_fps = float(genesis_mirror.get('render_fps', _camera_render_fps()))
    simulation_target_fps = _simulation_target_fps()
    print(f" [RENDER OPTIMIZER]   : {render_fps:.0f} FPS cached viewport; off-screen {render_w}x{render_h}")
    print(f" [SIM CLOCK]          : dt=0.016 s; loop cap={simulation_target_fps} steps/s; viewport render cap={render_fps:.0f} FPS")
    print(" [SIMULATION CORE]    : Physics timestep preserved at dt=0.016 s; real-time factor is reported below")
    print("===============================================================================\n")

    while running:
        dt = 0.016
        flight_time += dt
        simulation_step_count += 1

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
                    genesis_mirror['cached_view_surface'] = None
                    print(f">>> [CAMERA PERSPECTIVE]: {cam_mode} <<<")
                elif event.key == pygame.K_k:
                    genesis_mirror['show_hitboxes'] = not bool(genesis_mirror.get('show_hitboxes', False))
                    genesis_mirror['cached_view_surface'] = None
                    genesis_mirror['cached_view_key'] = None
                    print(f">>> [COLLISION HITBOX OVERLAY]: {'ON' if genesis_mirror['show_hitboxes'] else 'OFF'} <<<")
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
                        d.delivery_abort_retries = 0
                        d.delivery_retry_cooldown = 0.0
                        d.last_winch_state = d.winch.state
                        d.delivery_phase_elapsed = 0.0
                        d.rth_active = False
                        d.mission_phase = "1_TAKEOFF"
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
                # Left-clicking the viewport must never silently retask a live drone.
                # The old handler rewrote mission["target_pos"], changed the roof height,
                # reloaded cargo and forced phase 2 without checking whether the aircraft
                # was airborne or the cable was deployed. That could invalidate the
                # delivery gate while a drone was lowering. Target selection is now
                # telemetry-only; missions are owned by the dispatcher.
                print(
                    f"[INPUT] Ignored map click for mission retargeting; "
                    f"Drone {selected_drone_idx + 1} mission remains dispatcher-owned.",
                    flush=True,
                )

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

        # Local wind is evaluated independently at each drone's position/altitude.
        # The same global weather regime is shared, but Venturi effects and Dryden
        # turbulence are local to each aircraft, not to the currently selected camera.
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
        # Preserve total_wind/amb_temp aliases for selected-drone display and log output.
        total_wind = drone_winds[selected_drone_idx]
        amb_temp = drone_ambient_temps[selected_drone_idx]

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

        # One shared hitbox inventory feeds flight avoidance, cable intersection tests,
        # the startup coverage audit and the optional visual collision overlay.
        obstacles_cbf = build_runtime_collision_obstacles(
            world, traffic_engine, agent_pos, drones, delivered_boxes
        )
        swarm_states = [{"p": d.airframe.p.copy(), "v": d.airframe.v.copy()} for d in drones]

        selected_diag = {}
        drone_step_data = []
        batch_observations = []
        # ==========================================
        # STEP 1: AUTHORITATIVE SENSOR AUDIT & STACKING
        # ==========================================
        for d_idx, d in enumerate(drones):
            drone_wind = drone_winds[d_idx]  # physical wind applied by the simulator
            estimated_wind, sensed_wind_speed, sensed_wind_from_deg, sensed_wind_quality = sensor_suite.poll_wind_vector(
                d.airframe.v, drone_wind, d.airframe.R, dt=dt, sensor_id=d_idx,
                ground_velocity_est=d.ekf.v
            )
            wind_speed_xy = float(np.linalg.norm(estimated_wind[:2]))
            tgt_xy = d.mission["target_pos"][:2]
            home_xy = d.mission["home_pad"][:2]
            roof_h = d.mission["roof_height"]
            drop_surface_z = float(roof_h + PHASE3_ROOF_PAD_TOP_OFFSET_M)
            cruise_z = d.assigned_cruise_alt

            # The station-holding predicate is recomputed inside the delivery branch,
            # but is consumed later for every mission phase when building winch_context.
            # Default it to False on every drone/step so takeoff, transit, ascent and
            # depot phases cannot read a value left unbound by a skipped branch.
            is_station_holding = False

            # Keep the delivery altitude fixed throughout approach, hover and winch
            # operation. The previous wind-dependent setpoint jumped by up to 3.5 m
            # when gust strength changed, contradicting the Phase-1/Phase-3 teacher.
            high_hover_z = d.get_high_hover_z()

            dist_tgt_xy = float(math.hypot(d.airframe.p[0] - tgt_xy[0], d.airframe.p[1] - tgt_xy[1]))
            dist_home_xy = float(math.hypot(d.airframe.p[0] - home_xy[0], d.airframe.p[1] - home_xy[1]))

            gate_tol_xy = min(5.5, 3.2 + 0.25 * wind_speed_xy)
            gate_tol_z = min(2.5, 1.4 + 0.12 * wind_speed_xy)

            # A failed contact/safety gate retracts without releasing the payload.
            # Without an explicit retry policy, the next STANDBY frame could lower
            # again immediately, producing the observed endless lower/raise loop.
            d.delivery_retry_cooldown = max(0.0, float(d.delivery_retry_cooldown) - dt)
            observed_winch_state = d.winch.state
            aborted_cycle_finished = (
                d.mission_phase == "4_DELIVERY_HOVER"
                and d.winch.has_payload
                and observed_winch_state == "STANDBY"
                and d.last_winch_state in ("ABORT_RETRACT", "RETRACTING")
            )
            if aborted_cycle_finished:
                d.delivery_abort_retries += 1
                d.hover_settle_timer = 0.0
                d.delivery_retry_cooldown = DELIVERY_ABORT_RECOVERY_S
                print(
                    f"[WINCH RECOVERY] Drone {d.drone_id + 1}: aborted drop retracted safely; "
                    f"retry {d.delivery_abort_retries}/{DELIVERY_MAX_ABORT_RETRIES + 1}, "
                    f"cooldown={DELIVERY_ABORT_RECOVERY_S:.1f}s, payload retained.", flush=True)
                if d.delivery_abort_retries > DELIVERY_MAX_ABORT_RETRIES:
                    d.delivery_failures += 1
                    d.delivery_phase_elapsed = 0.0
                    d.hover_settle_timer = 0.0
                    d.rth_active = True
                    dispatcher.report_failed(d.drone_id)
                    d.mission_phase = "5_CHIMNEY_ASCENT"
                    d.target = d.mission["home_pad"].copy()
                    d.chimney_phase_elapsed = 0.0
                    d.chimney_alignment_timer = 0.0
                    d.chimney_descent_settle_timer = 0.0
                    print(
                        f"[DELIVERY RETRIES EXHAUSTED] Drone {d.drone_id + 1}: "
                        "two aborted lower/retract cycles; retaining cargo and returning to base.",
                        flush=True)
            d.last_winch_state = observed_winch_state

            if not d.rth_active and d.mission_phase in ["1_TAKEOFF", "2_CRUISE_OUTBOUND", "3_CHIMNEY_DESCENT", "4_DELIVERY_HOVER"]:
                if d.evaluate_dynamic_rth_threshold(estimated_wind):
                    d.rth_active = True
                    d.mission_phase = "5_CHIMNEY_ASCENT"

                    d.hover_settle_timer = 0.0
                    d.target = d.mission["home_pad"].copy()
                    # Keep order ownership consistent: a dynamic-RTH task is returned
                    # to the dispatcher's pending queue. Never overwrite another drone's
                    # in-flight target while its cable/package may be deployed.
                    failed_mission_id = d.mission.get("bldg_id", "unknown")
                    dispatcher.report_failed(d.drone_id)
                    print(
                        f"\n>>> [RTH QUEUE] Drone #{d.drone_id + 1} returning to base; "
                        f"task {failed_mission_id} requeued for a later safe dispatch. <<<",
                        flush=True,
                    )

            if d.mission_phase == "1_TAKEOFF":
                wp_target = np.array([home_xy[0], home_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.0, max_climb=2.40, wind_vector=estimated_wind)
                if _cruise_altitude_captured(d.airframe.p[2], d.airframe.v[2], cruise_z):
                    d.mission_phase = "2_CRUISE_OUTBOUND"

            elif d.mission_phase == "2_CRUISE_OUTBOUND":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(d.airframe.p, d.airframe.v, wp_target, max_speed_xy=4.5, max_climb=2.40, wind_vector=estimated_wind)
                speed_xy_now = float(np.linalg.norm(d.airframe.v[:2]))
                # Capture the target early even if vertical speed or wind is not yet
                # inside the delivery envelope. The precision-approach controller holds
                # cruise altitude and will not arm descent until its independent safety
                # gates pass. Waiting here let aircraft coast through and past the target.
                if _should_enter_precision_approach(dist_tgt_xy, speed_xy_now, wind_speed_xy):
                    d.mission_phase = "3_CHIMNEY_DESCENT"
                    print(
                        f"[APPROACH CAPTURE] Drone {d.drone_id + 1}: entering precision target capture "
                        f"at xy_error={dist_tgt_xy:.2f}m, altitude_error="
                        f"{abs(float(d.airframe.p[2]) - float(cruise_z)):.2f}m, "
                        f"vz={d.airframe.v[2]:+.2f}m/s, wind={wind_speed_xy:.2f}m/s. "
                        f"Descent remains disarmed until all delivery gates pass.", flush=True)
                    # Cruise integrator memory is not valid for precision station keeping.
                    d.mixer.xy_err_integral.fill(0.0)
                    d.mixer.z_err_integral = 0.0
                    d.chimney_alignment_timer = 0.0
                    d.chimney_descent_armed = False
                    d.chimney_descent_settle_timer = 0.0
                    d.chimney_phase_elapsed = 0.0

            elif d.mission_phase == "3_CHIMNEY_DESCENT":
                approach_speed_xy = float(np.linalg.norm(d.airframe.v[:2]))
                approach_speed_z = abs(float(d.airframe.v[2]))
                wind_speed_now = float(np.linalg.norm(estimated_wind[:2]))
                # Pause the deadline while wind makes precision delivery unavailable;
                # otherwise the safety interlock itself can force a timeout abort.
                if wind_speed_now <= DELIVERY_MAX_OPERATIONAL_WIND_MS:
                    d.chimney_phase_elapsed += dt
                # Suspend deployment when sensed wind exceeds the validated precision-drop envelope.
                # If the winch is still stowed, return to cruise-height braking rather than descending.
                if wind_speed_now > DELIVERY_MAX_OPERATIONAL_WIND_MS and d.winch.cable_len <= 0.12:
                    d.chimney_descent_armed = False
                    d.chimney_alignment_timer = 0.0
                    d.chimney_descent_settle_timer = 0.0
                if not d.chimney_descent_armed:
                    # Brake and settle over the delivery XY while holding cruise altitude.
                    wp_target = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                    # Limit the requested XY speed as target error shrinks. The old
                    # fixed 1.25 m/s cap still requested up to 0.84 m/s at 0.70 m
                    # error, while descent arming requires <=0.35 m/s. That makes
                    # the gate oscillatory: the controller passes through its own
                    # alignment zone too quickly to accumulate the 0.75 s dwell.
                    _, v_ref = compute_vector_guidance(
                        d.airframe.p, d.airframe.v, wp_target,
                        max_speed_xy=0.0, max_climb=0.35, max_descend=0.15,
                        wind_vector=estimated_wind)
                    # Keep a fixed world-frame target, and retain full braking authority
                    # even when position error is small. A position-distance speed cap
                    # allowed the drone to orbit the 0.7 m capture zone at ~0.7 m/s.
                    p_ref = wp_target.copy()
                    v_ref[:2] = _velocity_damped_xy_reference(
                        tgt_xy, d.airframe.p[:2], d.airframe.v[:2],
                        position_gain=0.85, velocity_gain=1.50,
                        max_speed=1.60, max_position_speed=1.20)
                    cruise_aligned = (wind_speed_now <= DELIVERY_MAX_OPERATIONAL_WIND_MS
                                      and dist_tgt_xy <= DELIVERY_DRONE_CENTER_TOLERANCE_M
                                      and abs(float(d.airframe.p[2] - cruise_z)) <= 0.65
                                      and approach_speed_xy <= DELIVERY_MAX_HORIZONTAL_SPEED_MS
                                      and approach_speed_z <= DELIVERY_MAX_VERTICAL_SPEED_MS)
                    d.chimney_alignment_timer = d.chimney_alignment_timer + dt if cruise_aligned else 0.0
                    if d.chimney_alignment_timer >= 0.75:
                        d.chimney_descent_armed = True
                        d.mixer.xy_err_integral.fill(0.0)
                        d.chimney_descent_settle_timer = 0.0
                else:
                    # Descend vertically, with XY station-keeping and low vertical speed.
                    wp_target = np.array([tgt_xy[0], tgt_xy[1], high_hover_z], dtype=np.float32)
                    p_ref_guidance, v_ref = compute_vector_guidance(
                        d.airframe.p, d.airframe.v, wp_target,
                        max_speed_xy=0.0, max_climb=0.70,
                        max_descend=(0.65 if abs(float(d.airframe.p[2] - high_hover_z)) > 3.0
                                     else 0.40 if abs(float(d.airframe.p[2] - high_hover_z)) > 1.0 else 0.22),
                        wind_vector=estimated_wind)
                    # Hold a fixed target XY while descending. Using the current XY
                    # as p_ref every frame nulls the position-error term and lets gusts
                    # walk the drone away from the rooftop. The fixed reference restores
                    # a restoring force while zero lateral velocity damps oscillation.
                    p_ref = np.array([tgt_xy[0], tgt_xy[1], high_hover_z], dtype=np.float32)
                    v_ref[:2] = 0.0
                    dist_z = abs(float(d.airframe.p[2] - high_hover_z))
                    v_ref = _delivery_vertical_settle_reference(v_ref, dist_z)
                    at_hover = (wind_speed_now <= DELIVERY_MAX_OPERATIONAL_WIND_MS
                                and dist_tgt_xy <= DELIVERY_DRONE_CENTER_TOLERANCE_M
                                and dist_z <= 0.40
                                and approach_speed_xy <= 0.30 and approach_speed_z <= 0.20)
                    d.chimney_descent_settle_timer = d.chimney_descent_settle_timer + dt if at_hover else 0.0
                    if d.chimney_descent_settle_timer >= 0.60:
                        d.mission_phase = "4_DELIVERY_HOVER"
                        d.chimney_phase_elapsed = 0.0
                        d.hover_settle_timer = 0.0
                        d.delivery_phase_elapsed = 0.0
                        d.delivery_abort_retries = 0
                        d.delivery_retry_cooldown = 0.0
                        d.last_winch_state = d.winch.state
                        d.delivery_attempts += 1

                # The approach phase includes both cruise-height alignment and a
                # potentially large vertical move to the roof hover point. A fixed
                # 24 s timeout can be physically impossible for higher cruise-to-roof
                # differences. Scale the deadline to the commanded vertical travel,
                # with margin for acceleration, braking and the required settle dwell.
                approach_vertical_delta_m = abs(float(cruise_z) - float(high_hover_z))
                approach_timeout_s = max(
                    CHIMNEY_APPROACH_TIMEOUT_S,
                    1.8 * approach_vertical_delta_m / 0.65 + 12.0,
                )
                # This trace is deliberately observational. It captures the commanded
                # altitude references as well as the real altitude error, which makes
                # an approach that drifts several metres away from cruise height
                # diagnosable without extending the timeout or changing control.
                if d.mission_phase == "3_CHIMNEY_DESCENT" and simulation_step_count % 300 == d_idx:
                    print(
                        f"[APPROACH TRACK] Drone {d.drone_id + 1} "
                        f"elapsed={d.chimney_phase_elapsed:.1f}/{approach_timeout_s:.1f}s "
                        f"xy_error={dist_tgt_xy:.2f}m vxy={approach_speed_xy:.2f}m/s "
                        f"z={d.airframe.p[2]:.2f}m cruise_z={cruise_z:.2f}m "
                        f"z_error_cruise={float(d.airframe.p[2] - cruise_z):+.2f}m "
                        f"hover_z={high_hover_z:.2f}m z_error_hover={float(d.airframe.p[2] - high_hover_z):+.2f}m "
                        f"vz={d.airframe.v[2]:+.2f}m/s p_ref_z={p_ref[2]:.2f}m "
                        f"v_ref_z={v_ref[2]:+.2f}m/s wind={wind_speed_now:.2f}m/s "
                        f"descent_armed={d.chimney_descent_armed} "
                        f"align_dwell={d.chimney_alignment_timer:.2f}s "
                        f"settle_dwell={d.chimney_descent_settle_timer:.2f}s "
                        f"ice={d.icing.ice_thickness_mm:.2f}mm.", flush=True)
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
                    xy_error_abort = float(np.linalg.norm(d.airframe.p[:2] - tgt_xy))
                    xy_speed_abort = float(np.linalg.norm(d.airframe.v[:2]))
                    z_error_abort = abs(float(d.airframe.p[2]) - float(high_hover_z))
                    z_speed_abort = abs(float(d.airframe.v[2]))
                    print(
                        f"[APPROACH ABORT] Drone {d.drone_id + 1}: stability gate not achieved "
                        f"within {approach_timeout_s:.1f}s (vertical travel budget "
                        f"{approach_vertical_delta_m:.1f}m); retaining cargo and returning to base. "
                        f"xy_error={xy_error_abort:.2f}m, xy_speed={xy_speed_abort:.2f}m/s, "
                        f"z_error={z_error_abort:.2f}m, z_speed={z_speed_abort:.2f}m/s, "
                        f"wind={wind_speed_now:.2f}m/s, descent_armed={d.chimney_descent_armed}, "
                        f"align_dwell={d.chimney_alignment_timer:.2f}s, "
                        f"settle_dwell={d.chimney_descent_settle_timer:.2f}s."
                    )

            elif d.mission_phase == "4_DELIVERY_HOVER":
                wp_target = np.array([tgt_xy[0], tgt_xy[1], high_hover_z], dtype=np.float32)
                p_ref, v_ref = wp_target.copy(), np.zeros(3, dtype=np.float32)

                dist_drone_to_target = float(math.hypot(d.airframe.p[0] - tgt_xy[0], d.airframe.p[1] - tgt_xy[1]))
                drone_spd_xy = float(math.hypot(d.airframe.v[0], d.airframe.v[1]))
                surf_under_drone = world.get_surface_height(d.airframe.p[0], d.airframe.p[1])
                is_solidly_over_roof = abs(float(surf_under_drone) - float(roof_h)) <= 0.65
                sway_m = float(np.linalg.norm(d.winch.sway_offset)) if d.winch.cable_len > 0.12 else 0.0
                is_station_holding = (
                    dist_drone_to_target <= DELIVERY_DRONE_CENTER_TOLERANCE_M
                    and abs(float(d.airframe.p[2] - high_hover_z)) <= DELIVERY_HOVER_ALTITUDE_TOLERANCE_M
                    and drone_spd_xy <= DELIVERY_MAX_HORIZONTAL_SPEED_MS
                    and abs(float(d.airframe.v[2])) <= DELIVERY_MAX_VERTICAL_SPEED_MS
                    and sway_m <= DELIVERY_MAX_SWAY_M
                    and is_solidly_over_roof
                    and float(np.linalg.norm(estimated_wind[:2])) <= DELIVERY_MAX_DEPLOYMENT_WIND_MS
                )

                d.delivery_phase_elapsed += dt
                # Require one continuous period of stable station-keeping before
                # deployment. Do not accumulate time while drifting outside the zone.
                if is_station_holding and d.delivery_retry_cooldown <= 0.0:
                    d.hover_settle_timer += dt
                else:
                    d.hover_settle_timer = 0.0

                cable_stowed = (d.winch.cable_len <= 0.12)
                # Failed drops are never scored as completed. Abort/retract preserves
                # the payload, returns the drone safely, and requeues the order.
                if d.delivery_phase_elapsed >= DELIVERY_HOVER_TIMEOUT_S and not d.winch.delivery_completed:
                    if not cable_stowed:
                        d.winch.state = "ABORT_RETRACT"
                    else:
                        d.delivery_failures += 1
                        d.delivery_phase_elapsed = 0.0
                        d.hover_settle_timer = 0.0
                        d.rth_active = True
                        d.dispatcher.report_failed(d.drone_id)
                        d.mission_phase = "5_CHIMNEY_ASCENT"

                        d.hover_settle_timer = 0.0
                        d.target = d.mission["home_pad"].copy()
                        print(f"[DELIVERY FAILED] Drone {d.drone_id + 1}: {d.mission['bldg_id']} timed out; returning with payload and requeuing.")

                if d.winch.delivery_completed and cable_stowed:
                    d.winch.delivery_completed = False
                    d.hover_settle_timer = 0.0
                    d.delivery_phase_elapsed = 0.0
                    d.packages_delivered += 1
                    d.dispatcher.report_completed(d.drone_id)
                    d.mission_phase = "5_CHIMNEY_ASCENT"

                    d.hover_settle_timer = 0.0
                    d.target = d.mission["home_pad"].copy()

            elif d.mission_phase == "5_CHIMNEY_ASCENT":
                # Hold near the delivery target while recovering the assigned cruise
                # altitude. Never start horizontal return because a timer expired or
                # because the aircraft merely passed the target altitude while still
                # climbing fast; both behaviours made drones leave after a short hop.
                wp_target = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(
                    d.airframe.p, d.airframe.v, wp_target, max_speed_xy=0.0,
                    max_climb=2.0, max_descend=0.65, wind_vector=estimated_wind)
                d.hover_settle_timer += dt
                if _cruise_altitude_captured(d.airframe.p[2], d.airframe.v[2], cruise_z):
                    d.hover_settle_timer = 0.0
                    d.mission_phase = "6_CRUISE_INBOUND"
                    print(
                        f"[ASCENT COMPLETE] Drone {d.drone_id + 1}: cruise altitude captured "
                        f"at z={d.airframe.p[2]:.2f}m, vz={d.airframe.v[2]:+.2f}m/s; "
                        f"starting inbound return.", flush=True)
                elif d.hover_settle_timer >= 8.0:
                    print(
                        f"[ASCENT HOLD] Drone {d.drone_id + 1}: not leaving delivery area; "
                        f"waiting for cruise altitude capture (z={d.airframe.p[2]:.2f}/"
                        f"{cruise_z:.2f}m, vz={d.airframe.v[2]:+.2f}m/s).", flush=True)
                    d.hover_settle_timer = 0.0

            elif d.mission_phase == "6_CRUISE_INBOUND":
                # Keep the aircraft on the 24 m staging altitude once inside the pad's
                # capture radius. Continuing toward the old 14 m gate while waiting for
                # XY/vertical momentum to decay caused captures at arbitrary altitude.
                inside_depot_capture_zone = dist_home_xy <= DEPOT_APPROACH_CAPTURE_RADIUS_M
                depot_gate_z = (DEPOT_CAPTURE_STAGING_ALTITUDE_M
                                if inside_depot_capture_zone else 14.0)
                wp_target = np.array([home_xy[0], home_xy[1], depot_gate_z], dtype=np.float32)
                inbound_speed = _depot_inbound_speed_cap(dist_home_xy)
                p_ref, v_ref = compute_vector_guidance(
                    d.airframe.p, d.airframe.v, wp_target,
                    max_speed_xy=inbound_speed,
                    max_climb=(1.10 if inside_depot_capture_zone else 2.40),
                    max_descend=(0.90 if inside_depot_capture_zone else 1.80),
                    wind_vector=estimated_wind)
                # Do not transfer a moving aircraft into final pad capture. The inbound
                # position/velocity loop continues braking at the 24 m staging target
                # until both horizontal and vertical speeds are inside the capture gate.
                if _should_enter_depot_capture(dist_home_xy, d.airframe.p, d.airframe.v):
                    d.mission_phase = "7_DEPOT_DESCENT"
                    d.depot_alignment_timer = 0.0
                    d.depot_descent_armed = False
                    d.depot_staging_z = float(DEPOT_CAPTURE_STAGING_ALTITUDE_M)
                    d.depot_vertical_hold_active = False
                    d.mixer.xy_err_integral.fill(0.0)
                    d.mixer.z_err_integral = 0.0
                    _seed_vertical_command_smoother(d.mixer, d.airframe.v[2])
                    print(
                        f"[DEPOT CAPTURE] Drone {d.drone_id + 1}: entered pad capture "
                        f"at range={dist_home_xy:.2f}m, vxy={np.linalg.norm(d.airframe.v[:2]):.2f}m/s, "
                        f"vz={d.airframe.v[2]:+.2f}m/s, z={d.depot_staging_z:.2f}m, "
                        f"wind_xy={np.linalg.norm(estimated_wind[:2]):.2f}m/s, "
                        f"wind_z={estimated_wind[2]:+.2f}m/s; vertical descent remains disarmed.", flush=True)
                elif inside_depot_capture_zone and simulation_step_count % 300 == d_idx:
                    print(
                        f"[DEPOT INBOUND CAPTURE] Drone {d.drone_id + 1}: waiting for stable entry; "
                        f"range={dist_home_xy:.2f}m vxy={np.linalg.norm(d.airframe.v[:2]):.2f}m/s "
                        f"z={d.airframe.p[2]:.2f}/{DEPOT_CAPTURE_STAGING_ALTITUDE_M:.2f}m "
                        f"vz={d.airframe.v[2]:+.2f}m/s wind_xy={np.linalg.norm(estimated_wind[:2]):.2f}m/s "
                        f"wind_z={estimated_wind[2]:+.2f}m/s; holding at staging altitude.", flush=True)

            elif d.mission_phase == "7_DEPOT_DESCENT":
                p_ref, v_ref, d.depot_descent_armed, d.depot_alignment_timer, \
                    d.depot_staging_z, depot_armed_now, depot_paused_now = _depot_landing_guidance(
                        d.airframe.p, d.airframe.v, home_xy, d.depot_staging_z,
                        d.depot_descent_armed, d.depot_alignment_timer, dt,
                        wind_vector=estimated_wind)
                if depot_paused_now and not d.depot_vertical_hold_active:
                    # One-shot transition action: remove the previous descent
                    # acceleration while preserving measured vertical momentum.
                    _seed_vertical_command_smoother(d.mixer, d.airframe.v[2])
                    d.mixer.z_err_integral = 0.0
                # Keep the hold flag through wind-triggered disarming as well, so
                # smoother history is re-seeded once per transition, not every frame.
                d.depot_vertical_hold_active = bool(depot_paused_now)
                if depot_armed_now:
                    print(
                        f"[DEPOT DESCENT ARMED] Drone {d.drone_id + 1}: centred over pad "
                        f"for {DEPOT_LANDING_SETTLE_DWELL_S:.2f}s; commencing vertical descent.",
                        flush=True)
                elif depot_paused_now and simulation_step_count % 180 == d_idx:
                    _depot_wind_now = float(np.linalg.norm(np.asarray(estimated_wind[:2], dtype=np.float32)))
                    _depot_wind_z = float(estimated_wind[2])
                    _depot_wind_pause = (_depot_wind_now > DEPOT_LANDING_WIND_RESUME_MS
                                         or abs(_depot_wind_z) > DEPOT_LANDING_VERTICAL_WIND_PAUSE_MS)
                    _depot_wind_hard = (_depot_wind_now > DEPOT_LANDING_WIND_HARD_HOLD_MS
                                        or abs(_depot_wind_z) > DEPOT_LANDING_VERTICAL_WIND_HARD_HOLD_MS)
                    pause_label = ("DEPOT WIND RECOVERY" if _depot_wind_pause and _depot_wind_hard
                                   else "DEPOT WIND HOLD" if _depot_wind_pause
                                   else "DEPOT DESCENT PAUSED" if not d.depot_descent_armed
                                   else "DEPOT DESCENT HOLD")
                    print(
                        f"[{pause_label}] Drone {d.drone_id + 1}: xy_error="
                        f"{np.linalg.norm(d.airframe.p[:2] - home_xy):.2f}m, "
                        f"vxy={np.linalg.norm(d.airframe.v[:2]):.2f}m/s, "
                        f"vz={d.airframe.v[2]:+.2f}m/s wind_xy={_depot_wind_now:.2f}m/s "
                        f"wind_z={_depot_wind_z:+.2f}m/s; descent inhibited, "
                        f"recapturing pad and recovery altitude={d.depot_staging_z:.2f}m.", flush=True)
                if (not d.depot_descent_armed) and simulation_step_count % 180 == d_idx:
                    _depot_xy = float(np.linalg.norm(d.airframe.p[:2] - home_xy))
                    _depot_vxy = float(np.linalg.norm(d.airframe.v[:2]))
                    _depot_zerr_signed = float(d.depot_staging_z - d.airframe.p[2])
                    print(
                        f"[DEPOT HOLD] Drone {d.drone_id + 1}: range={_depot_xy:.2f}m "
                        f"vxy={_depot_vxy:.2f}m/s vz={d.airframe.v[2]:+.2f}m/s "
                        f"z={d.airframe.p[2]:.2f}m stage_z={d.depot_staging_z:.2f}m "
                        f"stage_error={_depot_zerr_signed:+.2f}m "
                        f"settle_dwell={d.depot_alignment_timer:.2f}/"
                        f"{DEPOT_LANDING_SETTLE_DWELL_S:.2f}s "
                        f"vz_cmd={d.mixer.vertical_v_cmd_smooth:+.2f}m/s "
                        f"az_cmd={d.mixer.vertical_a_cmd_smooth:+.2f}m/s^2 "
                        f"wind_xy={np.linalg.norm(estimated_wind[:2]):.2f}m/s "
                        f"wind_z={estimated_wind[2]:+.2f}m/s.", flush=True)
                dist_home_xy = float(np.linalg.norm(d.airframe.p[:2] - home_xy))
                is_centered_over_pad = (dist_home_xy <= 0.45)
                has_touched_pad = (d.airframe.p[2] <= 0.08 or d.winch.is_skid_grounded)
                is_settled_on_pad = (float(np.linalg.norm(d.airframe.v[:2])) <= 0.22
                                     and abs(float(d.airframe.v[2])) <= 0.18)
                if is_centered_over_pad and has_touched_pad and is_settled_on_pad:
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
                    d.delivery_abort_retries = 0
                    d.delivery_retry_cooldown = 0.0
                    d.last_winch_state = d.winch.state
                    d.delivery_phase_elapsed = 0.0
                    d.depot_alignment_timer = 0.0
                    d.depot_descent_armed = False
                    d.depot_staging_z = float(d.airframe.p[2])
                    d.depot_vertical_hold_active = False
                    d.icing.ice_thickness_mm = 0.0
                    d.bms.soc = 0.98
                    d.bms.v_pack = 25.0
                    d.airframe.p[2] = 0.06
                    # Depot turnaround is a verified stationary calibration point. Keep
                    # true state, EKF state and controller memory in agreement before
                    # loading the next mission so stale landing integrators cannot kick
                    # the aircraft sideways/upward on its first takeoff frame.
                    d.airframe.v.fill(0.0)
                    d.airframe.accel_linear.fill(0.0)
                    d.airframe.omega.fill(0.0)
                    d.airframe.R = np.eye(3, dtype=np.float32)
                    d.ekf.p[:] = d.airframe.p
                    d.ekf.v.fill(0.0)
                    d.ekf.q[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    d.ekf.b_a.fill(0.0)
                    d.ekf.b_g.fill(0.0)
                    d.ekf.P = np.diag([0.15]*3 + [0.1]*3 + [0.02]*3 + [0.05]*3 + [0.01]*3).astype(np.float32)
                    d.ekf.Q_d = d.ekf.Q_base.copy()
                    d.ekf.residual_scale = 1.0
                    d.mixer.xy_err_integral.fill(0.0)
                    d.mixer.z_err_integral = 0.0
                    d.mixer.vertical_v_cmd_smooth = 0.0
                    d.mixer.vertical_a_cmd_smooth = 0.0
                    d.mixer.wind_filt.fill(0.0)
                    d.mixer.b3_d_prev = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    d.mixer._last_motor_omegas = d.airframe.motor_omegas.copy()
                    d.chimney_alignment_timer = 0.0
                    d.chimney_descent_settle_timer = 0.0
                    d.chimney_phase_elapsed = 0.0
                    d.delivery_phase_elapsed = 0.0
                    d.hover_settle_timer = 0.0
                    d.winch.is_skid_grounded = False
                    d.mission_phase = "1_TAKEOFF"

            current_surf_z = world.get_surface_height(d.airframe.p[0], d.airframe.p[1])
            target_surface_z = float(d.mission["roof_height"] if ("DELIVER" in d.mission_phase or "CHIMNEY" in d.mission_phase) else 0.0)
            surf_class, is_surf_safe, _ = assurance.audit_surface(d.target[:2], target_z=target_surface_z)
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
            lidar_16 = lidar_suite.cast_360_lidar(
                d.airframe.p, drone_yaw,
                dynamic_vehicles=traffic_engine.get_cbf_obstacles(),
                dynamic_objects=[{"id": "Ground_Intruder", "pos": agent_pos.copy(), "radius": 1.85}])
            conical_8, wire_clr = lidar_suite.cast_downward_conical(d.airframe.p)
            aruco_3d, aruco_lk = sensor_suite.poll_aruco_fiducial_detector(d.airframe.p, d.airframe.R, d.mission["target_pos"], d.mission["tag_id"])
            siren_a, siren_f, siren_s = sensor_suite.poll_directional_acoustic_array(d.airframe.p, traffic_engine.vehicles, sim_time=flight_time)
            adsb_p, adsb_cs, adsb_d = sensor_suite.poll_adsb_civil_transponder(d.airframe.p, d.airframe.v, world.civil_aircraft)
            hook_a, hook_w = sensor_suite.poll_smart_hook_imu(
                d.airframe.accel_linear, d.airframe.omega, d.winch.sway_offset,
                d.winch.cable_len, d.winch.has_payload, d.winch.sway_velocity)
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
                target_p=wp_target, lidar_16=lidar_16, conical_8=conical_8,
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

            is_delivering = (d.mission_phase == "4_DELIVERY_HOVER")
            surf_now = world.get_surface_height(d.airframe.p[0], d.airframe.p[1])
            is_solidly_over_roof = abs(float(surf_now) - float(roof_h)) <= 0.65
            is_winch_settled = bool(
                is_delivering and is_station_holding and
                (d.winch.state != "STANDBY" or d.hover_settle_timer >= DELIVERY_HOLD_MIN_S)
            )

            # Defer winch state advancement until after policy inference so the trained
            # action and winch heads can actually choose whether/how to deploy/retract.
            winch_agent_dist = float(np.linalg.norm(agent_pos[:2] - d.airframe.p[:2]))
            winch_context = {
                "is_settled_geom": bool(is_winch_settled),
                "is_station_holding": bool(is_station_holding) if is_delivering else False,
                "is_solidly_over_roof": bool(is_solidly_over_roof),
                "dist_tgt_xy": float(dist_tgt_xy),
                "is_surface_safe": bool(is_surf_safe),
                "wind_speed_xy": float(wind_speed_xy),
                "agent_breach": bool(global_agent_breach),
                "agent_dist": winch_agent_dist,
                "target_surface_z": float(drop_surface_z),
            }

            drone_step_data.append({
                # Snapshot all mission coordinates/derived distances for this drone.
                # The policy/actuation pass is a second loop; Python loop locals such
                # as tgt_xy otherwise retain the FINAL drone's values and contaminate
                # earlier drones' release gates and obstacle-departure logic.
                **_capture_per_drone_mission_context(d.mission, d.airframe.p),
                "winch_context": winch_context,
                "kinematics_p_before": np.asarray(d.airframe.p, dtype=np.float32).copy(),
                "kinematics_v_before": np.asarray(d.airframe.v, dtype=np.float32).copy(),
                # Exact pre-actuation state for optional V38.40 same-action shadow validation.
                # These snapshots align the current observation/action with the immediately
                # following authoritative BEMT/RK4 state; the evaluator cannot alter control.
                "action_transition_p_before": np.asarray(d.airframe.p, dtype=np.float64).copy(),
                "action_transition_v_before": np.asarray(d.airframe.v, dtype=np.float64).copy(),
                "action_transition_omega_before": np.asarray(d.airframe.omega, dtype=np.float64).copy(),
                "action_transition_surface_z_before": float(current_surf_z),
                "action_transition_parachute_before": bool(d.airframe.parachute_deployed),
                "action_transition_phase_before": str(d.mission_phase),
                "cargo_payload_before": bool(d.winch.has_payload),
                "cargo_cable_len_before": float(d.winch.cable_len),
                "p_ref": p_ref, "v_ref": v_ref, "R_est": R_est, "omega_est": omega_est,
                "load_3d": load_3d, "v_pack": v_pack, "bms_soc": bms_soc,
                "current_surf_z": current_surf_z, "meas_accel": meas_accel, "meas_gyro": meas_gyro,
                "s_data": s_data, "flow_2d": flow_2d, "is_surf_safe": is_surf_safe,
                "wind_estimate": estimated_wind.copy(), "wind_speed": sensed_wind_speed,
                "wind_from_deg": sensed_wind_from_deg, "wind_quality": sensed_wind_quality,
                "conical_8": conical_8, "i_rms": i_rms, "t_mosfet": t_mosfet,
                "adsb_d": adsb_d, "siren_s": siren_s, "phase_shift": phase_shift,
                "agent_dist": float(np.linalg.norm(agent_pos[:2] - d.airframe.p[:2])),
                "ekf_pos_cov_trace": float(np.trace(d.ekf.P[0:3, 0:3]))
            })

        # ==========================================
        # STEP 2: PARALLEL GPU RESNET INFERENCE
        # ==========================================
        t0_inf = time.time()
        batch_t = torch.from_numpy(np.stack(batch_observations)).to(device)
        with torch.no_grad():
            b_guidance, b_fsm, b_winch, b_health, b_trust, _, b_kinematics, b_cargo_dynamics, b_uncertainty = model(
                batch_t, return_physics=True, return_uncertainty=True)
        inf_latency_ms = (time.time() - t0_inf) * 1000.0

        b_guidance_np = (b_guidance * 0.15).cpu().numpy()
        b_kinematics_np = b_kinematics.cpu().numpy()
        b_cargo_dynamics_np = b_cargo_dynamics.cpu().numpy()
        b_uncertainty_np = b_uncertainty.detach().cpu().numpy()
        b_fsm_np = b_fsm.argmax(dim=-1).cpu().numpy()
        b_conf_np = F.softmax(b_fsm, dim=-1).max(dim=-1).values.cpu().numpy()
        b_winch_np = b_winch.cpu().numpy()
        b_trust_np = b_trust.cpu().numpy()

        # ==========================================
        # STEP 3: AUTHORITATIVE SE(3) ACTUATION STEP
        # ==========================================
        for d_idx, d in enumerate(drones):
            drone_wind = drone_winds[d_idx]  # actual environment disturbance
            gd = drone_step_data[d_idx]
            # Never read mission-specific loop locals left behind by STEP 1.
            # Each drone's own copied context is authoritative for winch geometry,
            # fallback release coordinates, and building-departure collision logic.
            tgt_xy = gd["mission_target_xy"]
            home_xy = gd["mission_home_xy"]
            roof_h = gd["mission_roof_height"]
            dist_tgt_xy = gd["mission_dist_target_xy"]
            dist_home_xy = gd["mission_dist_home_xy"]
            estimated_wind = gd["wind_estimate"]
            winch_ctx = gd["winch_context"]
            nn_guidance = b_guidance_np[d_idx].copy()
            nn_guidance_before_phase_gate = nn_guidance.copy()
            guidance_phase_gate_active = d.mission_phase in {
                "1_TAKEOFF", "3_CHIMNEY_DESCENT", "4_DELIVERY_HOVER",
                "5_CHIMNEY_ASCENT", "7_DEPOT_DESCENT", "8_DEPOT_SERVICE"
            }
            if guidance_phase_gate_active:
                nn_guidance.fill(0.0)
            predicted_token = int(b_fsm_np[d_idx])
            policy_raw_action_histogram[int(np.clip(predicted_token, 0, len(ACTION_NAMES) - 1))] += 1
            nn_conf = float(b_conf_np[d_idx])
            nn_winch_rate = float(np.clip(b_winch_np[d_idx, 0], -1.0, 1.0))
            predicted_surface_trust = float(np.clip(b_trust_np[d_idx, 1], 0.0, 1.0))

            # Class 6 (EMERGENCY_DIVERT) has no positive examples in the teacher
            # dataset yet; never treat an untrained logit as a reliable command.
            # Below the confidence threshold, use a state-aware, conservative fallback.
            if nn_conf < POLICY_MIN_ACTION_CONFIDENCE or predicted_token == 6:
                if d.mission_phase == "4_DELIVERY_HOVER":
                    if d.winch.state == "STANDBY":
                        nn_fsm_token = 1
                    elif d.winch.state == "LOWERING":
                        cargo_clearance = _live_payload_bottom_clearance(
                        d.airframe.p[2], d.winch.cable_len, winch_ctx["target_surface_z"])
                        nn_fsm_token = 2 if cargo_clearance > 0.10 else 3
                    elif d.winch.state == "TOUCHDOWN":
                        nn_fsm_token = 3
                    elif d.winch.state in ("RETRACTING", "ABORT_RETRACT"):
                        nn_fsm_token = 4
                    else:
                        nn_fsm_token = 1
                elif d.mission_phase in ("7_DEPOT_DESCENT", "8_DEPOT_SERVICE"):
                    nn_fsm_token = 5
                else:
                    nn_fsm_token = 0
            else:
                nn_fsm_token = predicted_token

            # Precision delivery is a safety-critical state machine. Do not allow a
            # fluctuating learned class to cancel a valid lowering cycle or request
            # retraction before contact; the network remains active in flight phases.
            if d.mission_phase == "4_DELIVERY_HOVER":
                if d.winch.state == "STANDBY":
                    nn_fsm_token = 2 if (d.hover_settle_timer >= DELIVERY_HOLD_MIN_S
                                         and winch_ctx["is_station_holding"]) else 1
                elif d.winch.state == "LOWERING":
                    cargo_clearance = _live_payload_bottom_clearance(
                        d.airframe.p[2], d.winch.cable_len, winch_ctx["target_surface_z"])
                    nn_fsm_token = 2 if cargo_clearance > 0.10 else 3
                elif d.winch.state == "TOUCHDOWN":
                    nn_fsm_token = 3
                elif d.winch.state in ("RETRACTING", "ABORT_RETRACT"):
                    nn_fsm_token = 4
                else:
                    nn_fsm_token = 1
                nn_winch_rate = 0.0

            # Precision approach and depot landing are controller-owned safety phases.
            # The policy remains trained/used for the mission, but cannot select TRANSIT
            # or an unrelated action while the vehicle is settling over a roof/pad.
            if d.mission_phase == "3_CHIMNEY_DESCENT":
                nn_fsm_token = 1  # HOVER_LOCK: deterministic precision capture/vertical approach
                nn_winch_rate = 0.0
            elif d.mission_phase in ("7_DEPOT_DESCENT", "8_DEPOT_SERVICE"):
                nn_fsm_token = 5  # DEPOT_SERVICE: deterministic landing/grounded turnaround
                nn_winch_rate = 0.0
            if d.mission_phase in ("3_CHIMNEY_DESCENT", "7_DEPOT_DESCENT", "8_DEPOT_SERVICE"):
                # Do not let learned acceleration residuals fight the hard-gated
                # velocity/position controller in the last precision phase.
                nn_guidance = np.zeros_like(nn_guidance)

            # Only the actual delivery phase may deploy cargo. A neural action can
            # request it, but physical safety, roof location and drop-zone occupancy
            # are still checked by the winch system itself.
            geom_settled = bool(winch_ctx["is_settled_geom"])
            if d.winch.state == "STANDBY":
                # The start gate is intentionally strict: require continuous stable
                # station-keeping, safe roof geometry, and no retry cooldown.
                winch_is_settled = (geom_settled and winch_ctx["is_station_holding"]
                                    and d.delivery_retry_cooldown <= 0.0)
            elif d.winch.state in ("LOWERING", "TOUCHDOWN"):
                # MERGED DELIVERY FIX: once a deployment was legitimately armed, do not
                # reinterpret a transient loss of station-holding (often caused by the
                # cable beginning to swing) as a request to abort/retract. This mirrors
                # the uploaded fixed-master's active-cycle semantics. Independent hard
                # gates are still passed separately below: roof/surface safety, cable
                # collision, agent breach, and operational wind. At contact/release the
                # winch re-checks precise XY, velocity, and sway limits.
                winch_is_settled = _delivery_winch_continuation_authorized(
                    d.mission_phase, d.winch.state, d.winch.has_payload,
                    winch_ctx.get("wind_speed_xy", float("inf")))
            else:
                # Retraction must never depend on station-holding state.
                winch_is_settled = True

            cable_anchor = d.airframe.p + np.array([0.0, 0.0, -0.05], dtype=np.float32)
            cable_endpoint = np.asarray(d.winch.nodes_pos[-1], dtype=np.float32).copy()
            if (not np.all(np.isfinite(cable_endpoint)) or np.linalg.norm(cable_endpoint - cable_anchor) > max(2.0, d.winch.cable_len + 1.0)
                    or (np.linalg.norm(cable_endpoint) < 1e-4 and np.linalg.norm(cable_anchor) > 1.0)):
                cable_endpoint = cable_anchor + np.array([d.winch.sway_offset[0], d.winch.sway_offset[1], -d.winch.cable_len], dtype=np.float32)
            cable_points = _winch_cable_polyline_points(d.winch, d.airframe.p)
            cable_obstacles = [obs for obs in obstacles_cbf
                               if obs.get("owner_drone_idx", -1) != d_idx]
            cable_static_hit = (d.winch.cable_len > 0.20 and d.winch.has_payload
                                and (corridor_planner.is_segment_occluded(cable_anchor, cable_endpoint)
                                     or _polyline_intersects_dynamic_hitboxes(
                                         cable_points, world.get_static_collision_obstacles())))
            cable_dynamic_hit = (d.winch.cable_len > 0.20 and d.winch.has_payload
                                 and _polyline_intersects_dynamic_hitboxes(cable_points, cable_obstacles))
            # Periodic evidence for the delivery drift seen in V38.12. Log both true
            # and estimated state alongside the frozen mission target so a target-frame
            # error can be distinguished from EKF/controller divergence.
            if (d.mission_phase == "4_DELIVERY_HOVER"
                    and d.winch.state in ("LOWERING", "TOUCHDOWN", "RETRACTING", "ABORT_RETRACT")
                    and simulation_step_count % 120 == d_idx):
                _trace_p = np.asarray(d.airframe.p, dtype=np.float32)
                _trace_v = np.asarray(d.airframe.v, dtype=np.float32)
                _trace_target = np.asarray(d.mission["target_pos"], dtype=np.float32)[:2]
                _trace_xy = float(np.linalg.norm(_trace_p[:2] - _trace_target))
                print(
                    f"[DELIVERY TRACK] Drone {d.drone_id + 1} phase={d.mission_phase} "
                    f"winch={d.winch.state} task={d.mission.get('bldg_id', 'unknown')} "
                    f"p=({_trace_p[0]:.2f},{_trace_p[1]:.2f},{_trace_p[2]:.2f}) "
                    f"target=({_trace_target[0]:.2f},{_trace_target[1]:.2f}) "
                    f"xy_error={_trace_xy:.2f}m vxy={np.linalg.norm(_trace_v[:2]):.2f}m/s "
                    f"vz={_trace_v[2]:+.2f}m/s ekf_xy_error="
                    f"{np.linalg.norm(d.ekf.p[:2] - _trace_target):.2f}m "
                    f"p_ref=({gd['p_ref'][0]:.2f},{gd['p_ref'][1]:.2f},{gd['p_ref'][2]:.2f}) "
                    f"cable={d.winch.cable_len:.2f}m "
                    f"payout_hold={(d.winch.last_payout_pause_reason if d.winch.state == 'LOWERING' else None) or 'none'}",
                    flush=True,
                )

            # This snapshot must be local to the live-simulator step. The similarly
            # named variable in the harvest worker is in another function/scope.
            # Optional V38.40 shadow prediction. The action is the exact high-level command
            # selected after confidence fallback and hard phase/safety overrides, and it is
            # the same numeric guidance/FSM/winch command passed to the live controller below.
            # The extra inference is sparsely sampled and never feeds back into actuation.
            if authoritative_shadow_enabled and (simulation_step_count % authoritative_shadow_stride) == d_idx:
                authoritative_shadow_probe_count[d_idx] += 1
                try:
                    _shadow_guidance = torch.as_tensor(
                        np.asarray(nn_guidance[:3], dtype=np.float32).reshape(1, 3), device=device)
                    _shadow_fsm = torch.as_tensor([int(nn_fsm_token)], dtype=torch.long, device=device)
                    _shadow_winch = torch.as_tensor([[float(nn_winch_rate)]], dtype=torch.float32, device=device)
                    _shadow_action = _build_action_condition_tensor(_shadow_guidance, _shadow_fsm, _shadow_winch)
                    with torch.no_grad():
                        _shadow_outputs = model(
                            batch_t[d_idx:d_idx + 1], return_action_transition=True,
                            transition_action=_shadow_action)
                    _shadow_pred_norm = _shadow_outputs[-1][0].detach().cpu().numpy().astype(np.float64, copy=False)
                    _shadow_scale = np.asarray(
                        [ACTION_TRANSITION_POSITION_SCALE_M] * 3
                        + [ACTION_TRANSITION_VELOCITY_SCALE_MS] * 3
                        + [ACTION_TRANSITION_ANGULAR_RATE_SCALE_RADS] * 3, dtype=np.float64)
                    _shadow_pred_si = _shadow_pred_norm * _shadow_scale
                    if _shadow_pred_si.shape == (ACTION_TRANSITION_OUTPUT_DIM,) and np.all(np.isfinite(_shadow_pred_si)):
                        gd["authoritative_shadow_prediction_si"] = _shadow_pred_si.copy()
                        gd["authoritative_shadow_guidance_norm"] = float(np.linalg.norm(np.asarray(nn_guidance[:3], dtype=np.float64)))
                        gd["authoritative_shadow_pre_gate_guidance_norm"] = float(np.linalg.norm(np.asarray(nn_guidance_before_phase_gate[:3], dtype=np.float64)))
                        gd["authoritative_shadow_phase_gate_active"] = bool(guidance_phase_gate_active)
                        gd["authoritative_shadow_phase_gate_suppressed_nonzero"] = bool(
                            guidance_phase_gate_active and np.linalg.norm(np.asarray(nn_guidance_before_phase_gate[:3], dtype=np.float64)) > 1e-9)
                        gd["authoritative_shadow_fsm"] = int(nn_fsm_token)
                        gd["authoritative_shadow_winch"] = float(nn_winch_rate)
                        gd["authoritative_shadow_wind_mps"] = float(np.linalg.norm(np.asarray(estimated_wind, dtype=np.float64)))
                        gd["authoritative_shadow_payload"] = bool(d.winch.has_payload)
                        gd["authoritative_shadow_phase"] = str(gd.get("action_transition_phase_before", d.mission_phase))
                        gd["authoritative_shadow_probe"] = True
                    else:
                        authoritative_shadow_invalid_count[d_idx] += 1
                except (RuntimeError, ValueError, TypeError, IndexError, FloatingPointError) as shadow_exc:
                    authoritative_shadow_invalid_count[d_idx] += 1
                    if simulation_step_count % 600 == d_idx:
                        print(f"[AUTHORITATIVE SHADOW VALIDATION WARNING] Drone {d_idx + 1}: "
                              f"prediction probe failed ({type(shadow_exc).__name__}: {shadow_exc}); ignored for control.", flush=True)

            winch_state_before_step = d.winch.state
            d.winch.step(
                drone_z=d.airframe.p[2],
                target_surface_z=winch_ctx["target_surface_z"],
                is_settled=winch_is_settled,
                is_surface_safe=(winch_ctx["is_surface_safe"] and winch_ctx["is_solidly_over_roof"]),
                wire_hit=bool(cable_static_hit or cable_dynamic_hit),
                agent_breach=winch_ctx["agent_breach"],
                agent_dist=winch_ctx["agent_dist"],
                wind_2d=estimated_wind[:2],
                drone_accel_linear=d.airframe.accel_linear,
                neural_token_idx=nn_fsm_token,
                nn_winch_rate=nn_winch_rate,
                dt=dt,
                drone_p=d.airframe.p,
                drone_v=d.airframe.v,
                wind_3d=drone_wind,
                target_xy=tgt_xy
            )
            if d.winch.state == "ABORT_RETRACT" and winch_state_before_step != "ABORT_RETRACT":
                _abort_target_xy = np.asarray(d.mission["target_pos"], dtype=np.float32)[:2]
                _abort_xy_error = float(np.linalg.norm(d.airframe.p[:2] - _abort_target_xy))
                print(
                    f"[WINCH ABORT] Drone {d.drone_id + 1}: "
                    f"{d.winch.last_abort_reason or 'unspecified physical/contact gate'}; "
                    f"p=({d.airframe.p[0]:.2f},{d.airframe.p[1]:.2f},{d.airframe.p[2]:.2f}) "
                    f"target=({_abort_target_xy[0]:.2f},{_abort_target_xy[1]:.2f}) "
                    f"xy_error_now={_abort_xy_error:.2f}m "
                    f"vxy={np.linalg.norm(d.airframe.v[:2]):.2f}m/s vz={d.airframe.v[2]:+.2f}m/s; "
                    f"cable={d.winch.cable_len:.2f}m, payload_retained={d.winch.has_payload}.",
                    flush=True,
                )
            if d.winch.release_count > d.last_visual_release_count:
                # The package is now detached and stays on the rooftop while the
                # empty cable retracts. Store its static display pose once per release.
                for _ in range(d.winch.release_count - d.last_visual_release_count):
                    release_pos = d.winch.last_release_pos
                    if release_pos is None:
                        release_pos = np.array([tgt_xy[0], tgt_xy[1], roof_h + 0.09], dtype=np.float32)
                    delivered_boxes.append(np.asarray(release_pos, dtype=np.float32).copy())
                d.last_visual_release_count = d.winch.release_count
            policy_action_histogram[nn_fsm_token] += 1

            world.shared_microclimate_map.broadcast_wind(d.airframe.p, estimated_wind, flight_time)
            # The flight controller receives only the onboard sensor estimate. The
            # ground-truth wind is reserved for the physics engine and sensor synthesis.
            effective_wind_feed = estimated_wind

            omega_cmds, f_thrust, _ = d.mixer.mix(
                drone_health=d.airframe.motor_health, p_est=d.ekf.p, v_est=d.ekf.v, R_est=gd["R_est"], omega_est=gd["omega_est"],
                p_target=gd["p_ref"], neural_guidance=nn_guidance, token_idx=nn_fsm_token,
                estimated_wind=effective_wind_feed, obstacles=obstacles_cbf, swarm_states=swarm_states, drone_idx=d_idx,
                ref_accel=np.zeros(3, dtype=np.float32), ref_vel=gd["v_ref"], 
                target_bldg_id=d.mission["bldg_id"] if ("OUTBOUND" in d.mission_phase or "DELIVER" in d.mission_phase or "CHIMNEY" in d.mission_phase) else None,
                departure_bldg_id=d.mission["bldg_id"] if ("ASCENT" in d.mission_phase or ("INBOUND" in d.mission_phase and dist_tgt_xy < 25.0)) else None,
                cable_len=d.winch.cable_len, sway_2d=d.winch.sway_offset, load_cell_3d=gd["load_3d"], 
                ige_fac=d.airframe.last_ige, v_pack=gd["v_pack"], bms_temp_c=d.bms.temp_c, 
                has_payload=d.winch.has_payload, ice_mm=d.icing.ice_thickness_mm, dt=dt,
                conical_ranges_8=gd["conical_8"], esc_i_rms=gd["i_rms"], esc_t_mosfet=gd["t_mosfet"],
                f_skid_left=d.winch.f_skid_left, f_skid_right=d.winch.f_skid_right,
                ekf_pos_cov_trace=gd["ekf_pos_cov_trace"],
                station_keeping=(d.mission_phase in ("3_CHIMNEY_DESCENT", "4_DELIVERY_HOVER", "7_DEPOT_DESCENT")),
                sway_velocity_2d=d.winch.sway_velocity,
                approach_braking=(d.mission_phase == "3_CHIMNEY_DESCENT" and not d.chimney_descent_armed),
                rho_air=gd["s_data"].get("rho_air", 1.225)
            )

            d.airframe.step(
                motor_cmds=omega_cmds, current_wind=drone_wind, rho_air=gd["s_data"]["rho_air"],
                ice_index=d.icing.ice_thickness_mm / 3.0, surface_z=gd["current_surf_z"], has_payload=d.winch.has_payload, dt=dt
            )

            # Match the prediction made just before actuation with this exact authoritative
            # BEMT/RK4 transition. This is online, same-action validation—not a counterfactual
            # test. We use the existing target validity mask to avoid scoring contacts/resets.
            if authoritative_shadow_enabled and gd.get("authoritative_shadow_probe", False):
                try:
                    _p0 = np.asarray(gd["action_transition_p_before"], dtype=np.float64)
                    _v0 = np.asarray(gd["action_transition_v_before"], dtype=np.float64)
                    _w0 = np.asarray(gd["action_transition_omega_before"], dtype=np.float64)
                    _p1 = np.asarray(d.airframe.p, dtype=np.float64)
                    _v1 = np.asarray(d.airframe.v, dtype=np.float64)
                    _w1 = np.asarray(d.airframe.omega, dtype=np.float64)
                    _surface = float(gd["action_transition_surface_z_before"])
                    _contact_free = bool(
                        _p0[2] > _surface + ROTATIONAL_DYNAMICS_MIN_SURFACE_CLEARANCE_M
                        and _p1[2] > _surface + ROTATIONAL_DYNAMICS_MIN_SURFACE_CLEARANCE_M
                        and not bool(gd["action_transition_parachute_before"])
                        and not bool(d.airframe.parachute_deployed))
                    _actual_norm = _numpy_one_step_action_transition_target(
                        _p0, _v0, _w0, _p1, _v1, _w1, dt,
                        transition_valid=True, contact_free=_contact_free)
                    if bool(_actual_norm[-1] > 0.5):
                        _scale = np.asarray(
                            [ACTION_TRANSITION_POSITION_SCALE_M] * 3
                            + [ACTION_TRANSITION_VELOCITY_SCALE_MS] * 3
                            + [ACTION_TRANSITION_ANGULAR_RATE_SCALE_RADS] * 3, dtype=np.float64)
                        _actual_si = np.asarray(_actual_norm[:-1], dtype=np.float64) * _scale
                        _pred_si = np.asarray(gd["authoritative_shadow_prediction_si"], dtype=np.float64)
                        # V38.42 parity fix: use the same observation-derived velocity
                        # convention as offline validation, not hidden true simulator v0.
                        _baseline_si = _numpy_action_transition_constant_velocity_baseline_si(
                            np.asarray(batch_observations[d_idx], dtype=np.float64), dt=dt)[0]
                        if not np.all(np.isfinite(_baseline_si)):
                            raise ValueError("observation-derived constant-velocity baseline is non-finite")
                        _parts = (slice(0, 3), slice(3, 6), slice(6, 9))
                        for _axis, _sl in enumerate(_parts):
                            authoritative_shadow_sq_error[d_idx, _axis] += float(np.sum((_pred_si[_sl] - _actual_si[_sl]) ** 2))
                            authoritative_shadow_sq_baseline[d_idx, _axis] += float(np.sum((_baseline_si[_sl] - _actual_si[_sl]) ** 2))
                        authoritative_shadow_eval_count[d_idx] += 1
                        authoritative_shadow_action_norm_sum[d_idx] += float(gd.get("authoritative_shadow_guidance_norm", 0.0))
                        authoritative_shadow_pre_gate_guidance_norm_sum[d_idx] += float(
                            gd.get("authoritative_shadow_pre_gate_guidance_norm", 0.0))
                        authoritative_shadow_phase_gate_sample_count[d_idx] += int(
                            bool(gd.get("authoritative_shadow_phase_gate_active", False)))
                        authoritative_shadow_phase_gate_suppressed_count[d_idx] += int(
                            bool(gd.get("authoritative_shadow_phase_gate_suppressed_nonzero", False)))
                        authoritative_shadow_wind_sum[d_idx] += float(gd.get("authoritative_shadow_wind_mps", 0.0))
                        authoritative_shadow_loaded_count[d_idx] += int(bool(gd.get("authoritative_shadow_payload", False)))
                        _phase = str(gd.get("authoritative_shadow_phase", "UNKNOWN"))
                        _fsm_name = ACTION_NAMES[int(np.clip(gd.get("authoritative_shadow_fsm", 0), 0, len(ACTION_NAMES) - 1))]
                        _winch_value = float(gd.get("authoritative_shadow_winch", 0.0))
                        _winch_name = "PAYOUT" if _winch_value > 0.10 else ("RETRACT" if _winch_value < -0.10 else "NEUTRAL")
                        for _map, _key in ((authoritative_shadow_phase_counts[d_idx], _phase),
                                           (authoritative_shadow_fsm_counts[d_idx], _fsm_name),
                                           (authoritative_shadow_winch_counts[d_idx], _winch_name)):
                            _map[_key] = int(_map.get(_key, 0)) + 1
                    else:
                        authoritative_shadow_invalid_count[d_idx] += 1
                except (ValueError, TypeError, IndexError, FloatingPointError) as shadow_eval_exc:
                    authoritative_shadow_invalid_count[d_idx] += 1
                    if simulation_step_count % 600 == d_idx:
                        print(f"[AUTHORITATIVE SHADOW VALIDATION WARNING] Drone {d_idx + 1}: "
                              f"paired target invalid ({type(shadow_eval_exc).__name__}: {shadow_eval_exc}); ignored for control.", flush=True)

            # Passive uncertainty monitoring: all outputs are diagnostics only. Sigma values
            # are transformed from normalized head units into approximate SI standard deviations.
            if uncertainty_base_ready:
                try:
                    lv = np.clip(np.asarray(b_uncertainty_np[d_idx], dtype=np.float64),
                                 PHYSICS_UNCERTAINTY_LOGVAR_MIN, PHYSICS_UNCERTAINTY_LOGVAR_MAX)
                    sig = np.exp(0.5 * lv)
                    if np.all(np.isfinite(sig)) and sig.shape == (PHYSICS_UNCERTAINTY_DIM,):
                        uncertainty_sigma_sum[d_idx, 0] += float(np.mean(sig[0:3]) * 8.0)
                        uncertainty_sigma_sum[d_idx, 1] += float(np.mean(sig[3:6]) * 12.0)
                        uncertainty_sigma_sum[d_idx, 2] += float(np.mean(sig[6:8]) * CARGO_DYNAMICS_SWAY_SCALE_M)
                        uncertainty_sigma_sum[d_idx, 3] += float(np.mean(sig[8:10]) * CARGO_DYNAMICS_SWAY_RATE_SCALE_MS)
                        uncertainty_sigma_sum[d_idx, 4] += float(sig[10] * CARGO_DYNAMICS_TENSION_SCALE_N)
                        uncertainty_sigma_sum[d_idx, 5] += float(np.mean(sig[11:14]) * FORCE_BALANCE_RESIDUAL_SCALE_N)
                        uncertainty_sigma_sum[d_idx, 6] += float(np.mean(sig[14:17]) * ROTATIONAL_DYNAMICS_RESIDUAL_SCALE_NM)
                        uncertainty_eval_count[d_idx] += 1
                except (ValueError, IndexError, FloatingPointError):
                    pass

            # Evaluate one-step forecast quality against the observed physical transition.
            # This monitoring path is deliberately one-way: it never alters commands.
            if kinematics_predictor_ready:
                p_before = gd["kinematics_p_before"]
                v_before = gd["kinematics_v_before"]
                p_after = np.asarray(d.airframe.p, dtype=np.float64)
                v_after = np.asarray(d.airframe.v, dtype=np.float64)
                observed_avg_velocity = (p_after - p_before) / max(dt, 1e-6)
                observed_acceleration = (v_after - v_before) / max(dt, 1e-6)
                pred = np.asarray(b_kinematics_np[d_idx], dtype=np.float64)
                if (np.all(np.isfinite(pred)) and np.all(np.isfinite(observed_avg_velocity))
                        and np.all(np.isfinite(observed_acceleration))
                        and abs(float(v_after[2])) <= 8.0
                        and float(np.linalg.norm(v_after)) <= 45.0
                        and float(np.linalg.norm(observed_acceleration)) <= 60.0):
                    pred_velocity = pred[:3] * 8.0
                    pred_acceleration = pred[3:6] * 12.0
                    kinematics_vel_sq_error_sum[d_idx] += float(np.sum((pred_velocity - observed_avg_velocity) ** 2))
                    kinematics_accel_sq_error_sum[d_idx] += float(np.sum((pred_acceleration - observed_acceleration) ** 2))
                    pred_integral_residual = pred_velocity - v_before - 0.5 * dt * pred_acceleration
                    observed_integral_residual = observed_avg_velocity - v_before - 0.5 * dt * observed_acceleration
                    kinematics_equation_resid_sq_error_sum[d_idx] += float(
                        np.sum((pred_integral_residual - observed_integral_residual) ** 2))
                    kinematics_eval_count[d_idx] += 1

            # Passive loaded-cargo forecast validation. Both the prediction and target
            # are relative to the airframe; this path cannot alter the live controller.
            if (cargo_dynamics_predictor_ready
                    and gd["cargo_payload_before"] and gd["cargo_cable_len_before"] > 0.12
                    and d.winch.has_payload and d.winch.cable_len > 0.12):
                cargo_sway_observed = (np.asarray(d.winch.nodes_pos[-1], dtype=np.float64)[:2]
                                       - np.asarray(d.airframe.p, dtype=np.float64)[:2])
                cargo_rate_observed = (np.asarray(d.winch.nodes_vel[-1], dtype=np.float64)[:2]
                                       - np.asarray(d.airframe.v, dtype=np.float64)[:2])
                cargo_tension_observed = float(d.winch.tension_n)
                cargo_pred = np.asarray(b_cargo_dynamics_np[d_idx], dtype=np.float64)
                cargo_ok = (cargo_pred.shape == (5,) and np.all(np.isfinite(cargo_pred))
                    and np.all(np.isfinite(cargo_sway_observed))
                    and np.all(np.isfinite(cargo_rate_observed))
                    and math.isfinite(cargo_tension_observed)
                    and float(np.linalg.norm(cargo_sway_observed)) <= 10.0
                    and float(np.linalg.norm(cargo_rate_observed)) <= 3.0 * CARGO_DYNAMICS_SWAY_RATE_SCALE_MS
                    and 0.0 <= cargo_tension_observed <= 45.0)
                if cargo_ok:
                    pred_sway = cargo_pred[:2] * CARGO_DYNAMICS_SWAY_SCALE_M
                    pred_sway_rate = cargo_pred[2:4] * CARGO_DYNAMICS_SWAY_RATE_SCALE_MS
                    pred_tension = float(np.clip(cargo_pred[4], 0.0, 1.0) * CARGO_DYNAMICS_TENSION_SCALE_N)
                    cargo_sway_sq_error_sum[d_idx] += float(np.sum((pred_sway - cargo_sway_observed) ** 2))
                    cargo_sway_rate_sq_error_sum[d_idx] += float(np.sum((pred_sway_rate - cargo_rate_observed) ** 2))
                    cargo_tension_sq_error_sum[d_idx] += float((pred_tension - cargo_tension_observed) ** 2)
                    cargo_eval_count[d_idx] += 1

            if simulation_step_count % 600 == 0:
                if kinematics_predictor_ready and kinematics_eval_count[d_idx] > 0:
                    kinematics_vel_rmse = math.sqrt(
                        kinematics_vel_sq_error_sum[d_idx] / (3.0 * kinematics_eval_count[d_idx]))
                    kinematics_accel_rmse = math.sqrt(
                        kinematics_accel_sq_error_sum[d_idx] / (3.0 * kinematics_eval_count[d_idx]))
                    kinematics_equation_resid_rmse = math.sqrt(
                        kinematics_equation_resid_sq_error_sum[d_idx] /
                        (3.0 * kinematics_eval_count[d_idx]))
                    print(f"[KINEMATICS PREDICTION] Drone {d_idx + 1} last_10_sim_s "
                          f"samples={kinematics_eval_count[d_idx]} "
                          f"velocity_RMSE={kinematics_vel_rmse:.3f}m/s "
                          f"acceleration_RMSE={kinematics_accel_rmse:.3f}m/s^2 "
                          f"kinematic_eq_residual_RMSE={kinematics_equation_resid_rmse:.3f}m/s "
                          "(per-drone; diagnostic only; no control feedback).", flush=True)
                elif not kinematics_predictor_ready and d_idx == 0:
                    print("[KINEMATICS PREDICTION] auxiliary head is untrained; forecast errors are not reported and it has no control authority.", flush=True)
                if uncertainty_base_ready and uncertainty_eval_count[d_idx] > 0:
                    uc = int(uncertainty_eval_count[d_idx])
                    umean = uncertainty_sigma_sum[d_idx] / max(1, uc)
                    uncertainty_fields = (f"sigma_vel={umean[0]:.3f}m/s sigma_accel={umean[1]:.3f}m/s^2 "
                                          f"sigma_sway={umean[2]:.3f}m sigma_sway_rate={umean[3]:.3f}m/s "
                                          f"sigma_tension={umean[4]:.3f}N")
                    if force_uncertainty_ready:
                        uncertainty_fields += f" sigma_force={umean[5]:.3f}N"
                    if rotational_dynamics_predictor_ready:
                        uncertainty_fields += f" sigma_torque={umean[6]:.4f}N*m"
                    print(f"[PREDICTION UNCERTAINTY] Drone {d_idx + 1} last_10_sim_s samples={uc} "
                          f"{uncertainty_fields} (estimated 1-sigma; diagnostic only, no control feedback).", flush=True)
                uncertainty_eval_count[d_idx] = 0
                uncertainty_sigma_sum[d_idx].fill(0.0)
                # Reset only this drone's window, not the other drones' metrics.
                kinematics_eval_count[d_idx] = 0
                kinematics_vel_sq_error_sum[d_idx] = 0.0
                kinematics_accel_sq_error_sum[d_idx] = 0.0
                kinematics_equation_resid_sq_error_sum[d_idx] = 0.0
                if authoritative_shadow_enabled:
                    _n_shadow = int(authoritative_shadow_eval_count[d_idx])
                    if _n_shadow > 0:
                        _denom_shadow = max(1.0, 3.0 * float(_n_shadow))
                        _shadow_names = ("position_delta_rmse_m", "velocity_delta_rmse_mps", "angular_rate_delta_rmse_rads")
                        _shadow_learned = {name: float(math.sqrt(max(0.0, authoritative_shadow_sq_error[d_idx, axis]) / _denom_shadow))
                                           for axis, name in enumerate(_shadow_names)}
                        _shadow_baseline = {name: float(math.sqrt(max(0.0, authoritative_shadow_sq_baseline[d_idx, axis]) / _denom_shadow))
                                            for axis, name in enumerate(_shadow_names)}
                        _shadow_gain = {name: _action_transition_relative_gain_pct(_shadow_baseline[name], _shadow_learned[name])
                                        for name in _shadow_names}
                        _shadow_record = {
                            "window_end_sim_time_s": float(flight_time), "drone_id": int(d_idx + 1),
                            "valid_samples": _n_shadow,
                            "probe_attempts": int(authoritative_shadow_probe_count[d_idx]),
                            "masked_or_failed_probes": int(authoritative_shadow_invalid_count[d_idx]),
                            "learned_transition_rmse": _shadow_learned,
                            "constant_velocity_baseline_rmse": _shadow_baseline,
                            "gain_vs_constant_velocity_baseline_pct": _shadow_gain,
                            "mean_live_guidance_input_norm": float(authoritative_shadow_action_norm_sum[d_idx] / _n_shadow),
                            "mean_pre_phase_gate_model_guidance_residual_norm": float(
                                authoritative_shadow_pre_gate_guidance_norm_sum[d_idx] / _n_shadow),
                            "phase_gate_active_samples": int(authoritative_shadow_phase_gate_sample_count[d_idx]),
                            "phase_gate_suppressed_nonzero_guidance_samples": int(
                                authoritative_shadow_phase_gate_suppressed_count[d_idx]),
                            "phase_gate_suppressed_nonzero_guidance_fraction": float(
                                authoritative_shadow_phase_gate_suppressed_count[d_idx] / _n_shadow),
                            "mean_sensed_wind_mps": float(authoritative_shadow_wind_sum[d_idx] / _n_shadow),
                            "loaded_payload_fraction": float(authoritative_shadow_loaded_count[d_idx] / _n_shadow),
                            "mission_phase_counts": dict(authoritative_shadow_phase_counts[d_idx]),
                            "fsm_action_counts": dict(authoritative_shadow_fsm_counts[d_idx]),
                            "winch_command_counts": dict(authoritative_shadow_winch_counts[d_idx]),
                            "target_source": "paired live custom BEMT/RK4 post-step state",
                            "control_changed_by_probe": False,
                        }
                        authoritative_shadow_windows.append(_shadow_record)
                        print(f"[AUTHORITATIVE SHADOW VALIDATION] Drone {d_idx + 1} last_10_sim_s n={_n_shadow} "
                              f"dposRMSE={_shadow_learned['position_delta_rmse_m']:.5f}m "
                              f"dvelRMSE={_shadow_learned['velocity_delta_rmse_mps']:.4f}m/s "
                              f"domegaRMSE={_shadow_learned['angular_rate_delta_rmse_rads']:.4f}rad/s "
                              f"| CV gain={_shadow_gain['position_delta_rmse_m']:+.1f}% pos/"
                              f"{_shadow_gain['velocity_delta_rmse_mps']:+.1f}% vel/"
                              f"{_shadow_gain['angular_rate_delta_rmse_rads']:+.1f}% omega; "
                              f"mean_wind={_shadow_record['mean_sensed_wind_mps']:.2f}m/s; "
                              "passive same-action test, no control feedback.", flush=True)
                    else:
                        _shadow_record = {
                            "window_end_sim_time_s": float(flight_time), "drone_id": int(d_idx + 1),
                            "valid_samples": 0,
                            "probe_attempts": int(authoritative_shadow_probe_count[d_idx]),
                            "masked_or_failed_probes": int(authoritative_shadow_invalid_count[d_idx]),
                            "status": ("NO_VALID_SAMPLES_IN_WINDOW" if authoritative_shadow_probe_count[d_idx] > 0
                                       else "NO_PROBES_IN_WINDOW"),
                            "control_changed_by_probe": False,
                        }
                        authoritative_shadow_windows.append(_shadow_record)
                    _write_authoritative_shadow_report("RUNNING")
                    authoritative_shadow_eval_count[d_idx] = 0
                    authoritative_shadow_invalid_count[d_idx] = 0
                    authoritative_shadow_probe_count[d_idx] = 0
                    authoritative_shadow_sq_error[d_idx].fill(0.0)
                    authoritative_shadow_sq_baseline[d_idx].fill(0.0)
                    authoritative_shadow_action_norm_sum[d_idx] = 0.0
                    authoritative_shadow_pre_gate_guidance_norm_sum[d_idx] = 0.0
                    authoritative_shadow_phase_gate_sample_count[d_idx] = 0
                    authoritative_shadow_phase_gate_suppressed_count[d_idx] = 0
                    authoritative_shadow_wind_sum[d_idx] = 0.0
                    authoritative_shadow_loaded_count[d_idx] = 0
                    authoritative_shadow_phase_counts[d_idx].clear()
                    authoritative_shadow_fsm_counts[d_idx].clear()
                    authoritative_shadow_winch_counts[d_idx].clear()
                if cargo_dynamics_predictor_ready and cargo_eval_count[d_idx] > 0:
                    cargo_sway_rmse = math.sqrt(cargo_sway_sq_error_sum[d_idx] / (2.0 * cargo_eval_count[d_idx]))
                    cargo_rate_rmse = math.sqrt(cargo_sway_rate_sq_error_sum[d_idx] / (2.0 * cargo_eval_count[d_idx]))
                    cargo_tension_rmse = math.sqrt(cargo_tension_sq_error_sum[d_idx] / cargo_eval_count[d_idx])
                    print(f"[CARGO DYNAMICS PREDICTION] Drone {d_idx + 1} last_10_sim_s "
                          f"active_samples={cargo_eval_count[d_idx]} sway_RMSE={cargo_sway_rmse:.3f}m "
                          f"sway_rate_RMSE={cargo_rate_rmse:.3f}m/s tension_RMSE={cargo_tension_rmse:.3f}N "
                          "(diagnostic only; no control feedback).", flush=True)
                elif not cargo_dynamics_predictor_ready and d_idx == 0:
                    print("[CARGO DYNAMICS PREDICTION] auxiliary loaded-cargo head is untrained; no cargo forecast is reported and it has no control authority.", flush=True)
                cargo_eval_count[d_idx] = 0
                cargo_sway_sq_error_sum[d_idx] = 0.0
                cargo_sway_rate_sq_error_sum[d_idx] = 0.0
                cargo_tension_sq_error_sum[d_idx] = 0.0

            if int(flight_time * 60) % 6 == 0:
                d.trail.append(d.airframe.p[:2].copy())

            if d_idx == selected_drone_idx:
                r_r, r_p, r_y = rot_to_euler(gd["R_est"])
                selected_diag = {
                    "v_pack": gd["v_pack"], "soc": gd["bms_soc"], "temp_c": d.bms.temp_c,
                    "action": ACTION_NAMES[nn_fsm_token],
                    "thrust": f_thrust, "tension": d.winch.tension_n,
                    "cable_l": d.winch.cable_len, "load_3d": gd["load_3d"],
                    "ice_mm": d.icing.ice_thickness_mm, "wind_speed": float(gd["wind_speed"]),
                    "wind_from_deg": float(gd["wind_from_deg"]), "wind_quality": float(gd["wind_quality"]),
                    "wind_estimate_xyz": estimated_wind.copy(),
                    "confidence": nn_conf, "latency": inf_latency_ms,
                    "gnss_status": "GNSS REJECTED [SPOOF DETECTED]" if inject_spoof else "GNSS NOMINAL (RTK FIXED)",
                    "jerk": float(np.linalg.norm(gd["meas_accel"] - d.airframe.accel_linear)),
                    "phase": float(gd["phase_shift"][0]),
                    "i_rms": gd["i_rms"], "t_mosfet": gd["t_mosfet"],
                    "hdop": 3.8 if inject_spoof else 0.85, "cn0": 28.5 if inject_spoof else 44.0,
                    "v_tas": gd["s_data"]["v_tas"],
                    "flow_2d": gd["flow_2d"], "aruco_locked": 1.0 if "DELIVERY" in d.mission_phase else 0.0,
                    "adsb_d": gd["adsb_d"], "siren_s": gd["siren_s"],
                    "roll": r_r, "pitch": r_p, "yaw": r_y, "alt": d.airframe.p[2],
                    "airspeed": float(np.linalg.norm(d.airframe.v)), "vspeed": d.airframe.v[2]
                }

        # ==========================================
        # STEP 4: GENESIS 3D VIEWPORT & HUD BLIT
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
        screen.blit(font_large.render("SURVIVAL CORE V38.25", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Dynamic Self-Adjusting SE(3) Swarm • PFD & Breguet Routing", True, (130, 160, 190)), (sb_x, 38))

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
            stat = d.mission_phase[:8]
            screen.blit(font_small.render(f"{tag} {stat:8s} | {d.bms.soc*100:.0f}% | Box:{d.packages_delivered}", True, c), (sb_x + 10, fl_y + 18 + d_i * 18))

        tot_deliv = dispatcher.total_completed
        total_attempts = sum(d.delivery_attempts for d in drones)
        total_drone_failures = sum(d.delivery_failures for d in drones)
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
        screen.blit(font_small.render(f"Deliveries: {tot_deliv} OK | Attempts: {total_attempts} | Failed: {total_drone_failures}", True, (0, 255, 140) if total_drone_failures == 0 else (255, 180, 40)), (sb_x + 10, p1_y + 50))

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
        screen.blit(font_small.render(f"Wind Sensor: {selected_diag.get('wind_speed', 0.0):.1f} m/s FROM {selected_diag.get('wind_from_deg', 0.0):.0f} deg | Q={selected_diag.get('wind_quality', 0.0):.2f}", True, (0, 220, 255)), (sb_x + 10, p4_y + 42))

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
        screen.blit(font_small.render(f"Primary Action:    {cur_act} ({cur_conf*100:.1f}%) | Latency: {selected_diag.get('latency', 1.0):.2f}ms (Batched)", True, (0, 255, 140)), (sb_x + 10, p6_y + 26))
        screen.blit(font_small.render(f"Winch State:       {act_d.winch.state} (Cable: {act_d.winch.cable_len:.2f} m | Tension: {act_d.winch.tension_n:.1f} N)", True, (0, 220, 255)), (sb_x + 10, p6_y + 42))

        bot_y = H - 38
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y - 8), (W - 14, bot_y - 8), 1)
        screen.blit(font_mono.render("[TAB] Drone | [C] Camera | [W] Weather | [CLICK] Retarget", True, (255, 200, 40)), (sb_x + 8, bot_y))

        vert_spd = act_d.airframe.v[2]
        status_banner = f"ACTIVE: D{selected_drone_idx+1} | Phase: {act_d.mission_phase} (Vz: {vert_spd:+.2f}m/s) | Delivered: {tot_deliv}"
        screen.blit(font_large.render(status_banner, True, (240, 240, 240)), (15, 14))
        screen.blit(font_small.render(f"V38.35 Transition Evaluation • Stable Landing • K:Hitboxes • View: {cam_mode}", True, (0, 210, 255)), (15, 40))

        pygame.display.flip()

        # Summarize policy behaviour at a low rate. This reveals whether the action
        # head ever chooses winch/hover actions and separates success from timeout.
        if flight_time - last_policy_metrics_time >= 10.0:
            actions_total = int(policy_action_histogram.sum())
            action_pct = (policy_action_histogram / max(1, actions_total) * 100.0).round(1).tolist()
            raw_actions_total = int(policy_raw_action_histogram.sum())
            raw_action_pct = (policy_raw_action_histogram / max(1, raw_actions_total) * 100.0).round(1).tolist()
            success_rate = 100.0 * dispatcher.total_completed / max(1, total_attempts)
            holding = [d for d in drones if d.mission_phase == "4_DELIVERY_HOVER"]
            phase_counts = {}
            for drone in drones:
                phase_counts[drone.mission_phase] = phase_counts.get(drone.mission_phase, 0) + 1
            phase_summary = ",".join(f"{phase}:{count}" for phase, count in sorted(phase_counts.items()))
            if holding:
                hold_errors = [float(np.linalg.norm(d.airframe.p[:2] - d.mission["target_pos"][:2])) for d in holding]
                hold_speeds = [float(np.linalg.norm(d.airframe.v[:2])) for d in holding]
                hold_alt_errors = [abs(float(d.airframe.p[2] - d.get_high_hover_z())) for d in holding]
                hold_stable_n = sum(
                    (hold_errors[i] <= DELIVERY_DRONE_CENTER_TOLERANCE_M
                     and hold_alt_errors[i] <= DELIVERY_HOVER_ALTITUDE_TOLERANCE_M
                     and hold_speeds[i] <= DELIVERY_MAX_HORIZONTAL_SPEED_MS
                     and abs(float(drone.airframe.v[2])) <= DELIVERY_MAX_VERTICAL_SPEED_MS)
                    for i, drone in enumerate(holding)
                )
                mean_hold_error = float(np.mean(hold_errors))
                mean_hold_speed = float(np.mean(hold_speeds))
                mean_hold_alt_error = float(np.mean(hold_alt_errors))
                hold_metrics = (f"hover_err={mean_hold_error:.2f}m hover_speed={mean_hold_speed:.2f}m/s "
                                f"hover_alt_err={mean_hold_alt_error:.2f}m station_ok={hold_stable_n}/{len(holding)}")
            else:
                # No drone has entered delivery hover in this metrics window yet.
                hold_metrics = "hover_err=n/a hover_speed=n/a hover_alt_err=n/a station_ok=0/0"
            chimney = [d for d in drones if d.mission_phase == "3_CHIMNEY_DESCENT"]
            if chimney:
                chimney_xy_errors = [float(np.linalg.norm(d.airframe.p[:2] - d.mission["target_pos"][:2])) for d in chimney]
                chimney_xy_speeds = [float(np.linalg.norm(d.airframe.v[:2])) for d in chimney]
                chimney_abs_vz = [abs(float(d.airframe.v[2])) for d in chimney]
                chimney_alt_errors = [
                    abs(float(d.airframe.p[2] - (d.get_high_hover_z() if d.chimney_descent_armed else d.assigned_cruise_alt)))
                    for d in chimney
                ]
                chimney_armed = sum(bool(d.chimney_descent_armed) for d in chimney)
                chimney_metrics = (
                    f"chimney={len(chimney)} armed={chimney_armed}/{len(chimney)} "
                    f"xy_err={float(np.mean(chimney_xy_errors)):.2f}m "
                    f"vxy={float(np.mean(chimney_xy_speeds)):.2f}m/s "
                    f"|vz|={float(np.mean(chimney_abs_vz)):.2f}m/s "
                    f"alt_err={float(np.mean(chimney_alt_errors)):.2f}m"
                )
            else:
                chimney_metrics = "chimney=n/a armed=0/0 xy_err=n/a vxy=n/a |vz|=n/a alt_err=n/a"
            print(
                f"[POLICY EVAL] sim_t={flight_time:.0f}s | deliveries={dispatcher.total_completed} "
                f"attempts={total_attempts} failed={total_drone_failures} "
                f"success/attempt={success_rate:.1f}% | wind_sensor={selected_diag.get('wind_speed', 0.0):.1f}m/s "
                f"from={selected_diag.get('wind_from_deg', 0.0):.0f}deg "
                f"{chimney_metrics} {hold_metrics} phases=[{phase_summary}] "
                f"| raw_head % [TRANSIT,HOVER,LOWER,TOUCHDOWN,RETRACT,SERVICE,UNTRAINED_DIVERT]={raw_action_pct} "
                f"| selected % [TRANSIT,HOVER,LOWER,TOUCHDOWN,RETRACT,SERVICE,UNTRAINED_DIVERT]={action_pct}"
            )
            policy_raw_action_histogram.fill(0)
            policy_action_histogram.fill(0)
            last_policy_metrics_time = flight_time

            wall_now = time.perf_counter()
            wall_window_s = max(1e-6, wall_now - last_metrics_wall_time)
            sim_window_s = max(0.0, flight_time - last_metrics_sim_time)
            steps_window = max(1, simulation_step_count - last_metrics_step_count)
            realtime_factor = sim_window_s / wall_window_s
            mean_step_ms = 1000.0 * wall_window_s / steps_window
            measured_step_rate = steps_window / wall_window_s
            total_wall_s = max(1e-6, wall_now - simulation_wall_start)
            total_rtf = flight_time / total_wall_s
            print(
                f"[SIM PERFORMANCE] window_sim={sim_window_s:.1f}s window_wall={wall_window_s:.1f}s "
                f"RTF={realtime_factor:.2f}x overall_RTF={total_rtf:.2f}x "
                f"step_rate={measured_step_rate:.1f}/s mean_step={mean_step_ms:.1f}ms "
                f"loop_cap={simulation_target_fps}/s viewport_cap={render_fps:.0f}fps",
                flush=True)
            last_metrics_wall_time = wall_now
            last_metrics_sim_time = flight_time
            last_metrics_step_count = simulation_step_count

        # Physics runs up to ~60 fixed steps/s independently of the lower-rate
        # cached Genesis viewport. The old shared 30 FPS cap limited simulation
        # advancement to at most 0.48x real time even before expensive frame work.
        clock.tick(simulation_target_fps)

    if authoritative_shadow_requested and authoritative_shadow_enabled:
        # Do not lose the final partial reporting window when the user closes the
        # simulation between 600-step metric boundaries. Also make a no-probe run
        # explicit rather than emitting an empty windows list with zero counters.
        for _d_idx in range(len(drones)):
            _n_shadow = int(authoritative_shadow_eval_count[_d_idx])
            _attempts = int(authoritative_shadow_probe_count[_d_idx])
            _invalid = int(authoritative_shadow_invalid_count[_d_idx])
            _has_prior = any(int(w.get("drone_id", -1)) == _d_idx + 1
                             for w in authoritative_shadow_windows)
            if _n_shadow <= 0 and _attempts <= 0 and _invalid <= 0 and _has_prior:
                continue
            _record = {
                "window_end_sim_time_s": float(flight_time),
                "drone_id": int(_d_idx + 1),
                "valid_samples": _n_shadow,
                "probe_attempts": _attempts,
                "masked_or_failed_probes": _invalid,
                "control_changed_by_probe": False,
                "final_partial_window": True,
            }
            if _n_shadow > 0:
                _denom_shadow = max(1.0, 3.0 * float(_n_shadow))
                _shadow_names = ("position_delta_rmse_m", "velocity_delta_rmse_mps", "angular_rate_delta_rmse_rads")
                _shadow_learned = {name: float(math.sqrt(max(0.0, authoritative_shadow_sq_error[_d_idx, axis]) / _denom_shadow))
                                   for axis, name in enumerate(_shadow_names)}
                _shadow_baseline = {name: float(math.sqrt(max(0.0, authoritative_shadow_sq_baseline[_d_idx, axis]) / _denom_shadow))
                                    for axis, name in enumerate(_shadow_names)}
                _record.update({
                    "learned_transition_rmse": _shadow_learned,
                    "constant_velocity_baseline_rmse": _shadow_baseline,
                    "gain_vs_constant_velocity_baseline_pct": {
                        name: _action_transition_relative_gain_pct(_shadow_baseline[name], _shadow_learned[name])
                        for name in _shadow_names
                    },
                    "mean_live_guidance_input_norm": float(authoritative_shadow_action_norm_sum[_d_idx] / _n_shadow),
                    "mean_pre_phase_gate_model_guidance_residual_norm": float(
                        authoritative_shadow_pre_gate_guidance_norm_sum[_d_idx] / _n_shadow),
                    "phase_gate_active_samples": int(authoritative_shadow_phase_gate_sample_count[_d_idx]),
                    "phase_gate_suppressed_nonzero_guidance_samples": int(
                        authoritative_shadow_phase_gate_suppressed_count[_d_idx]),
                    "phase_gate_suppressed_nonzero_guidance_fraction": float(
                        authoritative_shadow_phase_gate_suppressed_count[_d_idx] / _n_shadow),
                    "mean_sensed_wind_mps": float(authoritative_shadow_wind_sum[_d_idx] / _n_shadow),
                    "loaded_payload_fraction": float(authoritative_shadow_loaded_count[_d_idx] / _n_shadow),
                    "mission_phase_counts": dict(authoritative_shadow_phase_counts[_d_idx]),
                    "fsm_action_counts": dict(authoritative_shadow_fsm_counts[_d_idx]),
                    "winch_command_counts": dict(authoritative_shadow_winch_counts[_d_idx]),
                    "target_source": "paired live custom BEMT/RK4 post-step state",
                    "status": "PARTIAL_WINDOW_FLUSHED_AT_SHUTDOWN",
                })
            else:
                _record["status"] = "NO_VALID_SAMPLES_IN_FINAL_WINDOW" if (_attempts or _invalid) else "NO_PROBES_OBSERVED"
            authoritative_shadow_windows.append(_record)
            # These counts are now represented in the persisted window; clear the
            # pending accumulators so report totals do not count them twice.
            authoritative_shadow_eval_count[_d_idx] = 0
            authoritative_shadow_invalid_count[_d_idx] = 0
            authoritative_shadow_probe_count[_d_idx] = 0
            authoritative_shadow_sq_error[_d_idx].fill(0.0)
            authoritative_shadow_sq_baseline[_d_idx].fill(0.0)
            authoritative_shadow_action_norm_sum[_d_idx] = 0.0
            authoritative_shadow_pre_gate_guidance_norm_sum[_d_idx] = 0.0
            authoritative_shadow_phase_gate_sample_count[_d_idx] = 0
            authoritative_shadow_phase_gate_suppressed_count[_d_idx] = 0
            authoritative_shadow_wind_sum[_d_idx] = 0.0
            authoritative_shadow_loaded_count[_d_idx] = 0
            authoritative_shadow_phase_counts[_d_idx].clear()
            authoritative_shadow_fsm_counts[_d_idx].clear()
            authoritative_shadow_winch_counts[_d_idx].clear()
            print(f"[AUTHORITATIVE SHADOW VALIDATION] final flush Drone {_d_idx + 1}: "
                  f"attempts={_attempts} valid={_n_shadow} failed_or_masked={_invalid}; "
                  "passive only, no control feedback.", flush=True)
        _write_authoritative_shadow_report("COMPLETED")
    elif authoritative_shadow_requested:
        _write_authoritative_shadow_report("BLOCKED")
    pygame.quit()
    sys.exit()

def run_fixed_script_self_tests():
    """Fast regression checks; does not start training or the live simulation."""
    # The normal parent must have Torch; spawned harvest children must not load it.
    if _LIGHTWEIGHT_HARVEST_CHILD:
        assert torch is None and F is None and optim is None
        assert hasattr(nn, "Module"), "lightweight child must supply the model-class import shim"
    else:
        assert torch is not None, "normal application import must load PyTorch"
    # The observation contract is a hard checkpoint-compatibility constraint.
    assert BASE_OBS_DIM == 139, f"Unexpected base observation size: {BASE_OBS_DIM}"
    assert HISTORY_LEN == 3 and OBSERVATION_DIM == 417, (
        f"Unexpected temporal observation contract: {HISTORY_LEN} x {BASE_OBS_DIM} = {OBSERVATION_DIM}"
    )
    assert tuple(POWERLINE_Y_POSITIONS_M) == (-185.0, 185.0)
    assert POWERLINE_X_MIN_M < POWERLINE_X_MAX_M
    assert PHASE3_DEFAULT_ROLLOUT_STEPS == 3072
    # V38.40 authoritative live-shadow metric sanity: exact one-step prediction has
    # zero RMSE, the constant-velocity baseline has a finite positive acceleration error.
    _shadow_dt = 0.016
    _shadow_p0 = np.array([0.0, 0.0, 10.0], dtype=np.float64)
    _shadow_v0 = np.array([1.0, -0.5, 0.2], dtype=np.float64)
    _shadow_w0 = np.array([0.1, 0.0, -0.1], dtype=np.float64)
    _shadow_p1 = _shadow_p0 + _shadow_v0 * _shadow_dt + np.array([0.0, 0.0, 0.0005])
    _shadow_v1 = _shadow_v0 + np.array([0.20, -0.10, 0.05]) * _shadow_dt
    _shadow_w1 = _shadow_w0 + np.array([0.02, -0.03, 0.01]) * _shadow_dt
    _shadow_target = _numpy_one_step_action_transition_target(
        _shadow_p0, _shadow_v0, _shadow_w0, _shadow_p1, _shadow_v1, _shadow_w1,
        _shadow_dt, transition_valid=True, contact_free=True)
    assert _shadow_target[-1] == 1.0
    _shadow_scales = np.asarray([ACTION_TRANSITION_POSITION_SCALE_M] * 3
        + [ACTION_TRANSITION_VELOCITY_SCALE_MS] * 3
        + [ACTION_TRANSITION_ANGULAR_RATE_SCALE_RADS] * 3)
    _shadow_actual_si = _shadow_target[:-1] * _shadow_scales
    assert np.all(np.isfinite(_shadow_actual_si)) and _shadow_actual_si.shape == (9,)
    assert PHASE3_DEFAULT_ROLLOUT_STEPS * 0.01 >= 30.0

    # V38.30 offline predictive-control candidate evaluator: bounded deterministic
    # action choice, valid rollout metrics, and explicit no-feasible-action status.
    mpc_start = np.array([0.0, 0.0, 12.0], dtype=np.float64)
    mpc_target = np.array([4.0, 2.0, 12.0], dtype=np.float64)
    mpc_state = _offline_mpc_initial_state(mpc_start, np.zeros(3), sway_xy=(0.1, -0.05),
                                           sway_rate_xy=(0.0, 0.0))
    mpc_result = evaluate_offline_predictive_action(
        mpc_state, mpc_start, mpc_target, 4.0, 0.0,
        estimated_wind_xyz_mps=(0.0, 0.0, 0.0), cable_length_m=5.0,
        payload_mass_kg=OFFLINE_MPC_DEFAULT_PAYLOAD_MASS_KG,
        horizon_s=0.5, dt_s=0.05)
    assert mpc_result["status"] == "FEASIBLE", mpc_result["status"]
    assert mpc_result["evaluated_candidates"] >= 1
    assert mpc_result["feasible_candidates"] >= 1
    mpc_action = np.asarray(mpc_result["action_acceleration_mps2"], dtype=np.float64)
    assert np.all(np.isfinite(mpc_action))
    assert np.linalg.norm(mpc_action[:2]) <= OFFLINE_MPC_MAX_HORIZONTAL_ACCEL_MS2 + 1e-8
    assert abs(mpc_action[2]) <= OFFLINE_MPC_MAX_VERTICAL_ACCEL_MS2 + 1e-8
    mpc_repeat = evaluate_offline_predictive_action(
        mpc_state, mpc_start, mpc_target, 4.0, 0.0,
        estimated_wind_xyz_mps=(0.0, 0.0, 0.0), cable_length_m=5.0,
        payload_mass_kg=OFFLINE_MPC_DEFAULT_PAYLOAD_MASS_KG,
        horizon_s=0.5, dt_s=0.05)
    assert mpc_repeat["status"] == mpc_result["status"]
    assert np.allclose(mpc_repeat["action_acceleration_mps2"], mpc_action)
    mpc_next = _offline_mpc_surrogate_step(mpc_state, mpc_action, np.zeros(3), 0.05, 5.0, 0.8)
    assert np.all(np.isfinite(mpc_next["position"])) and np.all(np.isfinite(mpc_next["sway_xy"]))
    low_state = _offline_mpc_initial_state([0.0, 0.0, 0.20], [0.0, 0.0, 0.0])
    low_result = evaluate_offline_predictive_action(
        low_state, [0.0, 0.0, 0.20], [1.0, 0.0, 0.20], 2.0, 0.0,
        cable_length_m=0.0, horizon_s=0.5, dt_s=0.05)
    assert low_result["status"] == "NO_FEASIBLE_CANDIDATE"
    assert low_result["action_acceleration_mps2"] is None

    if not _LIGHTWEIGHT_HARVEST_CHILD:
        test_model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM)
        with torch.no_grad():
            test_input = torch.zeros((2, OBSERVATION_DIM), dtype=torch.float32)
            outputs = test_model(test_input)
            outputs_with_kin = test_model(test_input, return_kinematics=True)
            outputs_with_physics = test_model(test_input, return_physics=True)
            outputs_with_force = test_model(test_input, return_physics=True, return_force_residual=True)
            outputs_with_rot = test_model(test_input, return_physics=True, return_rotational_residual=True)
            outputs_with_force_rot = test_model(test_input, return_physics=True, return_force_residual=True,
                                                return_rotational_residual=True)
            outputs_with_uncertainty = test_model(test_input, return_physics=True, return_uncertainty=True)
            outputs_with_rot_uncertainty = test_model(test_input, return_physics=True,
                                                       return_rotational_residual=True, return_uncertainty=True)
            outputs_with_trajectory = test_model(test_input, return_physics=True,
                                                  return_rotational_residual=True, return_uncertainty=True,
                                                  return_trajectory_forecast=True)
            test_action_guidance = torch.tensor([[0.1, -0.2, 0.3, 0.0], [-0.1, 0.2, -0.3, 0.0]])
            test_action_fsm = torch.tensor([0, 2], dtype=torch.long)
            test_action_winch = torch.tensor([[0.0, 1.0], [0.4, 1.0]])
            test_action_condition = _build_action_condition_tensor(
                test_action_guidance, test_action_fsm, test_action_winch)
            outputs_with_action_transition = test_model(
                test_input, return_physics=True, return_rotational_residual=True,
                return_uncertainty=True, return_trajectory_forecast=True,
                return_action_transition=True, return_environment_transition=True,
                transition_action=test_action_condition)
        assert len(outputs) == 6 and len(outputs_with_kin) == 7 and len(outputs_with_physics) == 8
        assert len(outputs_with_uncertainty) == 9 and tuple(outputs_with_uncertainty[-1].shape) == (2, PHYSICS_UNCERTAINTY_DIM)
        assert len(outputs_with_rot_uncertainty) == 10 and tuple(outputs_with_rot_uncertainty[-1].shape) == (2, PHYSICS_UNCERTAINTY_DIM)
        assert len(outputs_with_trajectory) == 11 and tuple(outputs_with_trajectory[-2].shape) == (2, TRAJECTORY_FORECAST_DIM)
        assert tuple(outputs_with_trajectory[-1].shape) == (2, PHYSICS_UNCERTAINTY_DIM)
        assert len(outputs_with_force) == 9 and tuple(outputs_with_force[-1].shape) == (2, 3)
        assert len(outputs_with_rot) == 9 and tuple(outputs_with_rot[-1].shape) == (2, 3)
        assert len(outputs_with_force_rot) == 10 and tuple(outputs_with_force_rot[-1].shape) == (2, 3)
        assert tuple(outputs_with_kin[-1].shape) == (2, 6)
        assert tuple(outputs_with_physics[-2].shape) == (2, 6)
        assert tuple(outputs_with_physics[-1].shape) == (2, 5)
        assert not bool(test_model._force_balance_head_trained_state.item())
        assert not bool(test_model._rotational_dynamics_head_trained_state.item())
        assert not bool(test_model._physics_uncertainty_base_trained_state.item())
        assert not bool(test_model._force_uncertainty_trained_state.item())
        assert not bool(test_model._trajectory_forecast_head_trained_state.item())
        assert len(outputs_with_action_transition) == 13
        assert tuple(outputs_with_action_transition[-3].shape) == (2, ACTION_TRANSITION_OUTPUT_DIM)
        assert tuple(outputs_with_action_transition[-2].shape) == (2, PHYSICS_UNCERTAINTY_DIM)
        assert tuple(outputs_with_action_transition[-1].shape) == (2, ENVIRONMENT_TRANSITION_DIM)
        assert not bool(test_model._environment_observation_transition_trained_state.item())
        assert test_action_condition.shape == (2, ACTION_TRANSITION_ACTION_DIM)
        assert not bool(test_model._action_conditioned_transition_head_trained_state.item())
        uncertainty_target = torch.zeros((4, 4), dtype=torch.float32)
        uncertainty_target[:, 3] = torch.tensor([1.0, 1.0, 0.0, 1.0])
        uncertainty_prediction = torch.tensor([[0.1, -0.1, 0.2], [0.2, 0.1, -0.2],
                                               [8.0, 8.0, 8.0], [-0.1, 0.2, 0.1]], dtype=torch.float32,
                                              requires_grad=True)
        uncertainty_logvar = torch.zeros((4, 3), dtype=torch.float32, requires_grad=True)
        uncertainty_nll = _masked_gaussian_uncertainty_nll(
            uncertainty_prediction, uncertainty_logvar, uncertainty_target, 3)
        assert torch.isfinite(uncertainty_nll).item()
        uncertainty_nll.backward()
        assert torch.isfinite(uncertainty_logvar.grad).all().item()
        assert uncertainty_prediction.grad is None, "uncertainty loss must not backpropagate into mean predictions"
        uncertainty_stats = _uncertainty_calibration_stats(
            torch.zeros((2, 3)), torch.zeros((2, 3)), torch.cat((torch.zeros((2, 3)), torch.ones((2, 1))), dim=-1), 3)
        assert uncertainty_stats["count"] == 2 and uncertainty_stats["coverage_1sigma"] == 1.0
        test_targets = torch.zeros((2, 7), dtype=torch.float32)
        test_targets[:, 6] = torch.tensor([1.0, 0.0])
        test_prediction = torch.ones((2, 6), dtype=torch.float32, requires_grad=True)
        kin_loss = _kinematics_prediction_loss(test_prediction, test_targets)
        assert torch.isfinite(kin_loss).item()
        kin_loss.backward()
        assert torch.isfinite(test_prediction.grad).all().item()
        # The physical equation residual must be finite and exactly zero when the
        # predicted kinematic target matches the measured target, at both pipeline dt values.
        consistency_obs = torch.zeros((2, OBSERVATION_DIM), dtype=torch.float32)
        consistency_target = torch.zeros((2, 7), dtype=torch.float32)
        consistency_target[:, 6] = 1.0
        consistency_target[:, :3] = torch.tensor([[0.1, -0.2, 0.03], [0.0, 0.2, -0.1]])
        consistency_target[:, 3:6] = torch.tensor([[0.2, 0.4, -0.1], [-0.2, 0.1, 0.3]])
        consistency_pred = consistency_target[:, :6].clone().requires_grad_(True)
        for consistency_dt in (0.016, 0.010):
            consistency_loss = _kinematic_equation_residual_loss(
                consistency_pred, consistency_obs, consistency_target, dt=consistency_dt)
            consistency_rmse = _kinematic_equation_residual_rmse_mps(
                consistency_pred, consistency_obs, consistency_target, dt=consistency_dt)
            assert torch.isfinite(consistency_loss).item() and abs(float(consistency_loss.item())) < 1e-10
            assert math.isfinite(consistency_rmse) and abs(consistency_rmse) < 1e-5
        consistency_bad = consistency_pred.detach().clone()
        consistency_bad[:, 3] += 0.25
        residual_loss = _kinematic_equation_residual_loss(
            consistency_bad, consistency_obs, consistency_target, dt=0.016)
        assert torch.isfinite(residual_loss).item() and float(residual_loss.item()) > 0.0
        consistency_grad_test = consistency_pred.detach().clone().requires_grad_(True)
        consistency_grad_loss = _kinematic_equation_residual_loss(
            consistency_grad_test, consistency_obs, consistency_target, dt=0.016)
        consistency_grad_loss.backward()
        assert torch.isfinite(consistency_grad_test.grad).all().item()
        cargo_targets = torch.zeros((2, 6), dtype=torch.float32)
        cargo_targets[:, 5] = torch.tensor([1.0, 0.0])
        cargo_prediction = torch.ones((2, 5), dtype=torch.float32, requires_grad=True)
        cargo_loss = _cargo_dynamics_prediction_loss(cargo_prediction, cargo_targets)
        assert torch.isfinite(cargo_loss).item()
        cargo_loss.backward()
        assert torch.isfinite(cargo_prediction.grad).all().item()

        # Newtonian force balance: a_meas = a_cmd + F_external/m + F_residual/m.
        # A synthetic force-free residual must be zero and the head loss must backpropagate.
        test_mass = torch.full((3,), float(DRY_MASS_KG), dtype=torch.float32)
        test_accel_cmd = torch.tensor([[1.0, -0.5, 0.3], [0.0, 0.0, 0.0], [-1.0, 0.2, -0.4]], dtype=torch.float32)
        test_external_force = torch.tensor([[0.4, -0.2, 0.1], [1.0, 0.0, -0.5], [-0.3, 0.4, 0.2]], dtype=torch.float32)
        test_measured_accel = test_accel_cmd + test_external_force / test_mass[:, None]
        force_target = _torch_force_balance_residual_target(
            test_measured_accel, test_accel_cmd, test_external_force, test_mass,
            torch.ones(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool))
        assert force_target.shape == (3, 4) and torch.all(force_target[:, 3] == 1.0).item()
        assert float(torch.max(torch.abs(force_target[:, :3])).item()) < 1e-5
        force_target_bad = _torch_force_balance_residual_target(
            test_measured_accel, test_accel_cmd, test_external_force, test_mass,
            torch.tensor([True, False, True]), torch.tensor([True, True, False]))
        assert force_target_bad[:, 3].tolist() == [1.0, 0.0, 0.0]
        force_prediction = torch.ones((3, 3), dtype=torch.float32, requires_grad=True)
        force_loss = _force_balance_residual_prediction_loss(force_prediction, force_target)
        assert torch.isfinite(force_loss).item() and float(force_loss.item()) > 0.0
        force_loss.backward()
        assert torch.isfinite(force_prediction.grad).all().item()
        force_rmse_test = _force_balance_residual_rmse_n(torch.zeros((3, 3)), force_target)
        assert math.isfinite(force_rmse_test) and force_rmse_test < 1e-5

        # Euler-equation residual: a synthetic torque-free trajectory satisfying
        # J*alpha + omega x (J*omega) = tau has approximately zero residual.
        inertia_diag_test = np.diag(INERTIA_MATRIX).astype(np.float64)
        omega_before_test = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        alpha_expected_test = -np.cross(omega_before_test, inertia_diag_test * omega_before_test) / inertia_diag_test
        omega_after_test = omega_before_test + alpha_expected_test * 0.016
        rotational_target_test = _numpy_rotational_dynamics_residual_target(
            omega_before_test, omega_after_test, np.zeros(3), 0.016, True, True)
        assert rotational_target_test.shape == (4,) and rotational_target_test[3] == 1.0
        assert float(np.linalg.norm(rotational_target_test[:3])) < 1e-5
        rotational_target_invalid = _numpy_rotational_dynamics_residual_target(
            omega_before_test, omega_after_test, np.zeros(3), 0.016, False, True)
        assert rotational_target_invalid[3] == 0.0 and np.all(rotational_target_invalid[:3] == 0.0)
        rotational_targets_test = torch.as_tensor(
            np.stack([rotational_target_test, rotational_target_invalid]), dtype=torch.float32)
        rotational_prediction_test = torch.ones((2, 3), dtype=torch.float32, requires_grad=True)
        rotational_loss_test = _rotational_dynamics_residual_prediction_loss(
            rotational_prediction_test, rotational_targets_test)
        assert torch.isfinite(rotational_loss_test).item() and float(rotational_loss_test.item()) > 0.0
        rotational_loss_test.backward()
        assert torch.isfinite(rotational_prediction_test.grad).all().item()
        rotational_rmse_test = _rotational_dynamics_residual_rmse_nm(
            torch.as_tensor(rotational_target_test[:3]).reshape(1, 3),
            torch.as_tensor(rotational_target_test).reshape(1, 4))
        assert math.isfinite(rotational_rmse_test) and rotational_rmse_test < 1e-5

        valid_cargo_target = _numpy_one_step_cargo_dynamics_target(
            [0.5, -0.25], [0.8, -0.4], 12.0, True, 2.5, True, 2.52, True)
        invalid_cargo_target = _numpy_one_step_cargo_dynamics_target(
            [0.5, -0.25], [0.8, -0.4], 12.0, False, 0.08, False, 0.08, True)
        assert valid_cargo_target.shape == (6,) and valid_cargo_target[5] == 1.0
        assert invalid_cargo_target.shape == (6,) and invalid_cargo_target[5] == 0.0
        valid_target = _numpy_one_step_kinematics_target(
            [0, 0, 1], [0, 0, 0], [0, 0, 1.016], [0, 0, 0.5], 0.016)
        invalid_target = _numpy_one_step_kinematics_target(
            [0, 0, 1], [0, 0, 0], [0, 0, 1.2], [0, 0, 9.0], 0.016)
        assert valid_target.shape == (7,) and valid_target[6] == 1.0
        assert invalid_target[6] == 0.0
        legacy_state = {key: val for key, val in test_model.state_dict().items()
                        if not (key.startswith("kinematics_head.") or key.startswith("cargo_dynamics_head.")
                                or key.startswith("force_balance_residual_head.")
                                or key.startswith("rotational_dynamics_residual_head.")
                                or key in ("_force_balance_head_trained_state", "_rotational_dynamics_head_trained_state"))}
        compatibility_model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM)
        compatibility = compatibility_model.load_state_dict(legacy_state, strict=False)
        assert not compatibility.unexpected_keys
        assert compatibility.missing_keys and all(
            key.startswith("kinematics_head.") or key.startswith("cargo_dynamics_head.")
            or key.startswith("force_balance_residual_head.") or key.startswith("rotational_dynamics_residual_head.")
            or key in ("_force_balance_head_trained_state", "_rotational_dynamics_head_trained_state")
            for key in compatibility.missing_keys)

        # Exercise checkpoint loading for a V38.23-style checkpoint and older
        # checkpoints. A present force head without its trained marker stays untrained.
        import tempfile as _selftest_tempfile
        with _selftest_tempfile.TemporaryDirectory(prefix="survival_v3825_ckpt_test_") as test_dir:
            v3823_path = os.path.join(test_dir, "v38_23_checkpoint.pth")
            torch.save({key: val for key, val in test_model.state_dict().items()
                        if not key.startswith("cargo_dynamics_head.")}, v3823_path)
            v3823_loaded = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM)
            _load_model_checkpoint(v3823_loaded, v3823_path, torch.device("cpu"))
            assert bool(v3823_loaded._kinematics_head_checkpoint_trained)
            assert not bool(v3823_loaded._cargo_dynamics_head_checkpoint_trained)
            assert not bool(v3823_loaded._force_balance_head_checkpoint_trained)
            assert not bool(v3823_loaded._rotational_dynamics_head_checkpoint_trained)
            assert not bool(v3823_loaded._physics_uncertainty_base_checkpoint_trained)
            assert not bool(v3823_loaded._force_uncertainty_checkpoint_trained)

            old_path = os.path.join(test_dir, "older_checkpoint.pth")
            torch.save(legacy_state, old_path)
            old_loaded = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM)
            _load_model_checkpoint(old_loaded, old_path, torch.device("cpu"))
            assert not bool(old_loaded._kinematics_head_checkpoint_trained)
            assert not bool(old_loaded._cargo_dynamics_head_checkpoint_trained)
            assert not bool(old_loaded._force_balance_head_checkpoint_trained)
            assert not bool(old_loaded._rotational_dynamics_head_checkpoint_trained)

            # The persistent marker must survive save/load once Phase 3 actually trains it.
            trained_marker_path = os.path.join(test_dir, "force_head_trained_checkpoint.pth")
            trained_marker_model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM)
            trained_marker_model._force_balance_head_trained_state.fill_(True)
            torch.save(trained_marker_model.state_dict(), trained_marker_path)
            trained_marker_loaded = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM)
            _load_model_checkpoint(trained_marker_loaded, trained_marker_path, torch.device("cpu"))
            assert bool(trained_marker_loaded._force_balance_head_checkpoint_trained)

            trained_rot_marker_path = os.path.join(test_dir, "rotational_head_trained_checkpoint.pth")
            trained_rot_marker_model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM)
            trained_rot_marker_model._rotational_dynamics_head_trained_state.fill_(True)
            torch.save(trained_rot_marker_model.state_dict(), trained_rot_marker_path)
            trained_rot_marker_loaded = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM)
            _load_model_checkpoint(trained_rot_marker_loaded, trained_rot_marker_path, torch.device("cpu"))
            assert bool(trained_rot_marker_loaded._rotational_dynamics_head_checkpoint_trained)
        del compatibility_model, test_model
    # A vehicle entering the braking region must transition into precision capture
    # even when wind/vertical speed is not yet safe; the phase's independent descent
    # gate then keeps the winch disarmed until stability is achieved.
    assert _should_enter_precision_approach(8.0, 4.5, 9.0)
    assert not _should_enter_precision_approach(25.0, 1.0, 0.0)
    assert not _should_enter_precision_approach(float("nan"), 1.0, 0.0)
    # A deployed cable must not abort solely because transient swing breaks the
    # station-holding predicate; mission, payload and wind gates still apply.
    assert _delivery_winch_continuation_authorized("4_DELIVERY_HOVER", "LOWERING", True, 3.0)
    assert _delivery_winch_continuation_authorized("4_DELIVERY_HOVER", "TOUCHDOWN", True, 3.0)
    assert not _delivery_winch_continuation_authorized("2_CRUISE_OUTBOUND", "LOWERING", True, 3.0)
    assert not _delivery_winch_continuation_authorized("4_DELIVERY_HOVER", "LOWERING", False, 3.0)
    assert not _delivery_winch_continuation_authorized("4_DELIVERY_HOVER", "LOWERING", True, DELIVERY_MAX_OPERATIONAL_WIND_MS + 0.1)
    assert not _delivery_winch_continuation_authorized("4_DELIVERY_HOVER", "LOWERING", True, float("nan"))

    # Regression: an active lowering cycle must abort early if the aircraft has
    # departed the delivery envelope, rather than unwinding to the roof from afar.
    drift_guard = AutonomousWinchDeliverySystem()
    drift_guard.state = "LOWERING"
    drift_guard.cable_len = 1.0
    drift_guard.has_payload = True
    drift_guard.nodes_pos.fill(0.0)
    drift_guard.nodes_vel.fill(0.0)
    drift_guard.step(
        drone_z=10.0, target_surface_z=0.0, is_settled=True,
        is_surface_safe=True, wire_hit=False, agent_breach=False,
        agent_dist=30.0, wind_2d=np.zeros(2, dtype=np.float32),
        wind_3d=np.zeros(3, dtype=np.float32),
        drone_accel_linear=np.zeros(3, dtype=np.float32),
        neural_token_idx=2, nn_winch_rate=0.0, dt=0.016,
        drone_p=np.array([3.0, 0.0, 10.0], dtype=np.float32),
        drone_v=np.zeros(3, dtype=np.float32),
        target_xy=np.array([0.0, 0.0], dtype=np.float32))
    assert drift_guard.state == "ABORT_RETRACT", drift_guard.state
    assert "aircraft departed delivery envelope" in (drift_guard.last_abort_reason or "")

    # Wind gusts pause cable payout without aborting or changing cable length. A
    # sustained lull clears the hysteresis latch and lets the still-aligned cycle resume.
    wind_payout_guard = AutonomousWinchDeliverySystem()
    wind_payout_guard.state = "LOWERING"
    wind_payout_guard.cable_len = 1.0
    wind_payout_guard.has_payload = True
    wind_payout_guard.nodes_pos.fill(0.0)
    wind_payout_guard.nodes_vel.fill(0.0)
    wind_payout_guard.step(
        drone_z=10.0, target_surface_z=0.0, is_settled=True,
        is_surface_safe=True, wire_hit=False, agent_breach=False,
        agent_dist=30.0, wind_2d=np.array([5.2, 0.0], dtype=np.float32),
        wind_3d=np.array([3.5, 0.0, 0.0], dtype=np.float32),
        drone_accel_linear=np.zeros(3, dtype=np.float32),
        neural_token_idx=2, nn_winch_rate=0.0, dt=0.016,
        drone_p=np.array([0.0, 0.0, 10.0], dtype=np.float32),
        drone_v=np.zeros(3, dtype=np.float32),
        target_xy=np.array([0.0, 0.0], dtype=np.float32))
    assert wind_payout_guard.state == "LOWERING", wind_payout_guard.state
    assert abs(wind_payout_guard.cable_len - 1.0) < 1e-6, wind_payout_guard.cable_len
    assert wind_payout_guard.payout_wind_paused
    assert "wind hold" in (wind_payout_guard.last_payout_pause_reason or "")
    wind_payout_guard.step(
        drone_z=10.0, target_surface_z=0.0, is_settled=True,
        is_surface_safe=True, wire_hit=False, agent_breach=False,
        agent_dist=30.0, wind_2d=np.array([3.5, 0.0], dtype=np.float32),
        wind_3d=np.array([5.2, 0.0, 0.0], dtype=np.float32),
        drone_accel_linear=np.zeros(3, dtype=np.float32),
        neural_token_idx=2, nn_winch_rate=0.0, dt=0.016,
        drone_p=np.array([0.0, 0.0, 10.0], dtype=np.float32),
        drone_v=np.zeros(3, dtype=np.float32),
        target_xy=np.array([0.0, 0.0], dtype=np.float32))
    assert not wind_payout_guard.payout_wind_paused
    assert wind_payout_guard.cable_len > 1.0

    # A cargo sway excursion also pauses payout even if the drone centre remains on target.
    sway_payout_guard = AutonomousWinchDeliverySystem()
    sway_payout_guard.state = "LOWERING"
    sway_payout_guard.cable_len = 1.0
    sway_payout_guard.has_payload = True
    sway_payout_guard.sway_offset[:] = np.array([0.60, 0.0], dtype=np.float32)
    sway_payout_guard.nodes_pos.fill(0.0)
    sway_payout_guard.nodes_vel.fill(0.0)
    sway_payout_guard.step(
        drone_z=10.0, target_surface_z=0.0, is_settled=True,
        is_surface_safe=True, wire_hit=False, agent_breach=False,
        agent_dist=30.0, wind_2d=np.zeros(2, dtype=np.float32),
        wind_3d=np.zeros(3, dtype=np.float32),
        drone_accel_linear=np.zeros(3, dtype=np.float32),
        neural_token_idx=2, nn_winch_rate=0.0, dt=0.016,
        drone_p=np.array([0.0, 0.0, 10.0], dtype=np.float32),
        drone_v=np.zeros(3, dtype=np.float32),
        target_xy=np.array([0.0, 0.0], dtype=np.float32))
    assert sway_payout_guard.state == "LOWERING", sway_payout_guard.state
    assert abs(sway_payout_guard.cable_len - 1.0) < 1e-6, sway_payout_guard.cable_len
    assert "sway=" in (sway_payout_guard.last_payout_pause_reason or "")

    # A valid close-in delivery state must still be allowed to lower normally.
    hold_guard = AutonomousWinchDeliverySystem()
    hold_guard.state = "LOWERING"
    hold_guard.cable_len = 1.0
    hold_guard.has_payload = True
    hold_guard.nodes_pos.fill(0.0)
    hold_guard.nodes_vel.fill(0.0)
    hold_guard.step(
        drone_z=10.0, target_surface_z=0.0, is_settled=True,
        is_surface_safe=True, wire_hit=False, agent_breach=False,
        agent_dist=30.0, wind_2d=np.zeros(2, dtype=np.float32),
        wind_3d=np.zeros(3, dtype=np.float32),
        drone_accel_linear=np.zeros(3, dtype=np.float32),
        neural_token_idx=2, nn_winch_rate=0.0, dt=0.016,
        drone_p=np.array([0.5, 0.0, 10.0], dtype=np.float32),
        drone_v=np.zeros(3, dtype=np.float32),
        target_xy=np.array([0.0, 0.0], dtype=np.float32))
    assert hold_guard.state == "LOWERING", hold_guard.state

    assert _cruise_altitude_captured(42.0, 0.0, 42.0)
    assert _cruise_altitude_captured(42.8, -0.4, 42.0)
    assert not _cruise_altitude_captured(39.0, 0.0, 42.0)
    assert not _cruise_altitude_captured(42.0, 1.1, 42.0)
    # Near the delivery hover altitude, the target-position controller remains active
    # but vertical velocity feed-forward must no longer keep commanding descent.
    settle_ref = _delivery_vertical_settle_reference(
        np.array([0.0, 0.0, -0.22], dtype=np.float32), 0.16)
    assert float(settle_ref[2]) == 0.0, settle_ref
    outside_settle_ref = _delivery_vertical_settle_reference(
        np.array([0.0, 0.0, -0.22], dtype=np.float32), 0.90)
    assert abs(float(outside_settle_ref[2]) + 0.22) < 1e-6, outside_settle_ref
    assert CHIMNEY_APPROACH_TIMEOUT_S >= 45.0
    assert _simulation_target_fps() >= 30
    assert _configured_genesis_logging_level() in {"debug", "info", "warning", "error", "critical"}
    # Each drone must retain its own mission coordinates; no mutable shared/last-loop target.
    ctx_a = _capture_per_drone_mission_context(
        {"target_pos": [10.0, 20.0, 30.0], "home_pad": [1.0, 2.0, 0.0], "roof_height": 12.0},
        [8.0, 18.0, 15.0])
    ctx_b = _capture_per_drone_mission_context(
        {"target_pos": [100.0, -5.0, 25.0], "home_pad": [-20.0, -10.0, 0.0], "roof_height": 21.0},
        [95.0, -2.0, 17.0])
    assert np.allclose(ctx_a["mission_target_xy"], [10.0, 20.0])
    assert np.allclose(ctx_b["mission_target_xy"], [100.0, -5.0])
    assert not np.shares_memory(ctx_a["mission_target_xy"], ctx_b["mission_target_xy"])
    assert abs(ctx_a["mission_dist_target_xy"] - math.sqrt(8.0)) < 1e-5
    assert abs(ctx_b["mission_dist_target_xy"] - math.sqrt(34.0)) < 1e-5
    # Check the delivery-biased initial-state distribution still includes all six
    # represented mission phases, without being dominated by unrelated states.
    phase_test = _phase3_sample_initial_phases(4000, 3, torch.device("cpu"))
    phase_counts_test = torch.bincount(phase_test.reshape(-1), minlength=6)
    assert len(phase_counts_test) == 6 and bool((phase_counts_test > 0).all().item())
    phase3_fraction_test = float(phase_counts_test[3].item()) / float(phase_test.numel())
    assert 0.37 <= phase3_fraction_test <= 0.43, phase3_fraction_test
    # Phase-3 memory safeguard: dominant lower/retract frames are thinned, while
    # short touchdown/release and service states are retained at every rollout step.
    sample_test_labels = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    mask_off_stride = _phase3_make_sample_mask(sample_test_labels, step=1, sample_interval=16, critical_stride=8)
    assert mask_off_stride.tolist() == [[False, False, False, True, False, True]], mask_off_stride
    mask_critical_stride = _phase3_make_sample_mask(sample_test_labels, step=8, sample_interval=16, critical_stride=8)
    assert mask_critical_stride.tolist() == [[False, False, True, True, True, True]], mask_critical_stride
    mask_regular_stride = _phase3_make_sample_mask(sample_test_labels, step=16, sample_interval=16, critical_stride=8)
    assert bool(mask_regular_stride.all().item()), mask_regular_stride

    sensors = FullPhysicsSensorSuite()
    # Equal-distance points at the north and south wires must produce equal
    # distances and EMF readings; the old single +170 m wire model failed this.
    north = np.array([0.0, POWERLINE_Y_POSITIONS_M[1], POWERLINE_Z_M], dtype=np.float32)
    south = np.array([0.0, POWERLINE_Y_POSITIONS_M[0], POWERLINE_Z_M], dtype=np.float32)
    dn, ds = sensors.wire_distance_m(north), sensors.wire_distance_m(south)
    assert abs(dn) < 1e-6 and abs(ds) < 1e-6, (dn, ds)
    far_north = np.array([0.0, 200.0, POWERLINE_Z_M], dtype=np.float32)
    far_south = np.array([0.0, -200.0, POWERLINE_Z_M], dtype=np.float32)
    assert abs(sensors.wire_distance_m(far_north) - 15.0) < 1e-5
    assert abs(sensors.wire_distance_m(far_south) - 15.0) < 1e-5

    # A finite-segment check ensures x beyond the wire endpoints is not treated
    # as though an infinitely long wire existed there.
    beyond = np.array([200.0, 185.0, POWERLINE_Z_M], dtype=np.float32)
    assert abs(sensors.wire_distance_m(beyond) - 10.0) < 1e-5

    # Verify the instantiated world uses the same geometry, not merely the constants.
    world = MetropolitanCityGridWorld400m()
    assert len(world.powerlines) == len(POWERLINE_Y_POSITIONS_M)
    actual_y = tuple(sorted(float(line["p1"][1]) for line in world.powerlines))
    assert actual_y == tuple(sorted(POWERLINE_Y_POSITIONS_M)), (actual_y, POWERLINE_Y_POSITIONS_M)
    for line in world.powerlines:
        assert abs(float(line["p1"][0]) - POWERLINE_X_MIN_M) < 1e-6
        assert abs(float(line["p2"][0]) - POWERLINE_X_MAX_M) < 1e-6
        assert abs(float(line["p1"][2]) - POWERLINE_Z_M) < 1e-6
        assert abs(float(line["p2"][2]) - POWERLINE_Z_M) < 1e-6

    # Exercise the actual batched training feature builder and check feature 40
    # at the wire, 15 m from a wire, and 10 m beyond a segment endpoint.
    test_pos = torch.tensor([[[0.0, 185.0, 12.0], [0.0, -200.0, 12.0],
                              [200.0, 185.0, 12.0]]], dtype=torch.float32)
    zeros3 = torch.zeros_like(test_pos)
    zeros2 = torch.zeros((1, 3), dtype=torch.float32)
    false_payload = torch.zeros((1, 3), dtype=torch.bool)
    test_features = _phase3_torch_features(
        pos=test_pos, vel=zeros3, waypoint=zeros3, roof_z=zeros2,
        cable_len=zeros2, cable_vel=zeros2, has_payload=false_payload,
        winch_state=zeros2, phase=zeros2, wind=zeros3, accel=zeros3,
        building_aabbs=np.empty((0, 5), dtype=np.float32))
    assert tuple(test_features.shape) == (1, 3, BASE_OBS_DIM), tuple(test_features.shape)
    expected_feature40 = torch.tensor([0.0, 15.0 / 50.0, 10.0 / 50.0])
    assert torch.allclose(test_features[0, :, 40], expected_feature40, atol=1e-5), (
        test_features[0, :, 40], expected_feature40)

    # A physically impossible 2.7 km altitude from a harvesting worker must be
    # detected before it can contaminate later sensor/teacher samples.
    from types import SimpleNamespace
    fake_airframe = SimpleNamespace(
        p=np.array([0.0, 0.0, 2745.90], dtype=np.float32),
        v=np.array([0.0, 0.0, 3.93], dtype=np.float32),
        R=np.eye(3, dtype=np.float32),
        omega=np.zeros(3, dtype=np.float32),
        motor_omegas=np.full(8, HOVER_OMEGA_NOMINAL, dtype=np.float32),
        motor_health=np.ones(8, dtype=np.float32),
    )
    assert "altitude outside harvest envelope" in _harvest_dynamics_fault_reason(
        SimpleNamespace(airframe=fake_airframe)
    )
    fake_airframe.p[2] = 42.0
    assert _harvest_dynamics_fault_reason(SimpleNamespace(airframe=fake_airframe)) is None
    fake_airframe.p[2] = 55.0
    fake_airframe.v[2] = 8.1
    assert "vertical speed outside harvest envelope" in _harvest_dynamics_fault_reason(
        SimpleNamespace(airframe=fake_airframe)
    )
    fake_airframe.v[2] = 0.0
    fake_airframe.p[2] = 81.0
    assert "altitude outside harvest envelope" in _harvest_dynamics_fault_reason(
        SimpleNamespace(airframe=fake_airframe)
    )

    # Anti-sway sign regression: with sway defined as cargo minus drone, the
    # attachment-point command must follow both relative offset and relative velocity
    # so relative motion is damped. A negative sign reinforced the swing.
    sway_np = np.array([1.0, -0.5], dtype=np.float32)
    sway_v_np = np.array([0.4, -0.2], dtype=np.float32)
    feedback_np = _anti_sway_feedback(sway_np, sway_v_np)
    assert float(np.dot(feedback_np, sway_np)) > 0.0, (feedback_np, sway_np)
    assert float(np.dot(feedback_np, sway_v_np)) > 0.0, (feedback_np, sway_v_np)
    sway_t = torch.tensor([[[1.0, -0.5]]], dtype=torch.float32)
    sway_v_t = torch.tensor([[[0.4, -0.2]]], dtype=torch.float32)
    feedback_t = _anti_sway_feedback(sway_t, sway_v_t)
    assert float((feedback_t * sway_t).sum().item()) > 0.0
    assert float((feedback_t * sway_v_t).sum().item()) > 0.0

    # Near a target, braking authority must not collapse with position error.
    _near_target_cmd = _velocity_damped_xy_reference(
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([0.19, 0.0], dtype=np.float32),
        np.array([0.40, 0.0], dtype=np.float32),
        position_gain=0.90, velocity_gain=1.65, max_speed=1.25, max_position_speed=0.85)
    assert float(_near_target_cmd[0]) < -0.40, _near_target_cmd

    # Inbound speed must ramp down to zero across the braking buffer; the old
    # distance-only schedule requested about 3 m/s at 3 m range.
    assert _depot_inbound_speed_cap(3.0) == 0.0
    assert _depot_inbound_speed_cap(24.0) == 0.0
    assert 0.0 < _depot_inbound_speed_cap(30.0) < _depot_inbound_speed_cap(60.0) <= 9.5
    # Final capture must reject high horizontal/vertical momentum and altitude offsets.
    assert _should_enter_depot_capture(10.0, [0, 0, 24.0], [0.2, 0.1, 0.1])
    assert not _should_enter_depot_capture(10.0, [0, 0, 24.0], [1.2, 0.0, 0.0])
    assert not _should_enter_depot_capture(10.0, [0, 0, 24.0], [0.0, 0.0, 1.1])
    assert not _should_enter_depot_capture(10.0, [0, 0, 22.0], [0.0, 0.0, 0.0])
    assert not _should_enter_depot_capture(30.0, [0, 0, 24.0], [0.0, 0.0, 0.0])
    # Depot landing must hold altitude while aligning, then command no lateral
    # velocity during descent; lateral drift while airborne must pause the descent.
    depot_home = np.array([0.0, 0.0], dtype=np.float32)
    p_align = np.array([1.0, 0.0, 12.0], dtype=np.float32)
    v_align = np.zeros(3, dtype=np.float32)
    p_ref, v_ref, armed, timer, staging_z, armed_now, paused_now = _depot_landing_guidance(
        p_align, v_align, depot_home, 12.0, False, 0.0, 0.1, np.zeros(3, dtype=np.float32))
    assert not armed and abs(float(p_ref[2]) - 12.0) < 1e-6 and abs(float(v_ref[2])) < 1e-6
    # If the craft is approaching the pad too quickly, desired velocity must oppose
    # its current motion rather than repeatedly chasing and overshooting the centre.
    p_fast = np.array([1.0, 0.0, 12.0], dtype=np.float32)
    v_fast = np.array([4.0, 0.0, 0.0], dtype=np.float32)
    _, v_fast_ref, fast_armed, *_ = _depot_landing_guidance(
        p_fast, v_fast, depot_home, 12.0, False, 0.0, 0.016, np.zeros(3, dtype=np.float32))
    assert not fast_armed and float(v_fast_ref[0]) < 0.0, v_fast_ref
    p_centered = np.array([0.10, 0.05, 12.0], dtype=np.float32)
    v_centered = np.zeros(3, dtype=np.float32)
    p_ref, v_ref, armed, timer, staging_z, armed_now, paused_now = _depot_landing_guidance(
        p_centered, v_centered, depot_home, 12.0, True, 0.0, 0.016, np.zeros(3, dtype=np.float32))
    assert armed and np.all(np.isfinite(v_ref)) and float(v_ref[2]) < 0.0
    assert float(np.dot(v_ref[:2], depot_home - p_centered[:2])) >= 0.0
    # A brief lateral speed spike while near the pad must hold vertical motion and
    # issue an opposing velocity reference, without restarting alignment forever.
    p_near = np.array([0.19, 0.0, 12.0], dtype=np.float32)
    v_near = np.array([0.40, 0.0, 0.0], dtype=np.float32)
    p_ref, v_ref, armed, timer, staging_z, armed_now, paused_now = _depot_landing_guidance(
        p_near, v_near, depot_home, 12.0, True, 0.0, 0.016, np.zeros(3, dtype=np.float32))
    assert armed and paused_now and float(v_ref[0]) < 0.0 and abs(float(v_ref[2])) < 1e-8
    p_drifted = np.array([0.90, 0.0, 8.0], dtype=np.float32)
    p_ref, v_ref, armed, timer, staging_z, armed_now, paused_now = _depot_landing_guidance(
        p_drifted, np.array([0.0, 0.0, -0.1], dtype=np.float32), depot_home,
        12.0, True, 0.0, 0.016, np.zeros(3, dtype=np.float32))
    assert paused_now and not armed and abs(float(staging_z) - 12.0) < 1e-6
    assert abs(float(p_ref[2]) - 8.0) < 1e-6 and abs(float(v_ref[2])) < 1e-6

    # Keep the learned Z residual from overpowering deterministic altitude/velocity
    # tracking, while preserving full learned XY guidance authority.
    test_neural_residual = _bound_policy_vertical_residual(
        np.array([5.0, -4.0, 6.0], dtype=np.float32) * 6.0)
    assert abs(float(test_neural_residual[2])) <= POLICY_VERTICAL_ACCEL_RESIDUAL_LIMIT_MS2 + 1e-6
    assert float(test_neural_residual[0]) == 30.0 and float(test_neural_residual[1]) == -24.0
    try:
        _bound_policy_vertical_residual(np.zeros(2, dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("Vertical residual limiter must reject a malformed acceleration vector")

    # Staging altitude hold must command against measured vertical motion even
    # when the position error is almost zero.
    p_rising = np.array([0.02, -0.03, 24.08], dtype=np.float32)
    v_rising = np.array([0.0, 0.0, 0.55], dtype=np.float32)
    _, v_rising_ref, still_armed, *_ = _depot_landing_guidance(
        p_rising, v_rising, depot_home, 24.0, False, 0.0, 0.016,
        np.zeros(3, dtype=np.float32))
    assert not still_armed and float(v_rising_ref[2]) < 0.0, v_rising_ref
    p_falling = np.array([0.02, -0.03, 23.92], dtype=np.float32)
    v_falling = np.array([0.0, 0.0, -0.55], dtype=np.float32)
    _, v_falling_ref, still_armed, *_ = _depot_landing_guidance(
        p_falling, v_falling, depot_home, 24.0, False, 0.0, 0.016,
        np.zeros(3, dtype=np.float32))
    assert not still_armed and float(v_falling_ref[2]) > 0.0, v_falling_ref

    # Low-altitude landing should tolerate a small lateral error/speed and continue a
    # slow descent rather than repeatedly hovering just above the pad.
    p_low_drift = np.array([0.51, 0.0, 1.20], dtype=np.float32)
    v_low_drift = np.array([0.15, 0.0, 0.0], dtype=np.float32)
    _, v_low_drift_ref, low_armed, _, _, _, low_paused = _depot_landing_guidance(
        p_low_drift, v_low_drift, depot_home, 1.20, True, 0.0, 0.016,
        np.zeros(3, dtype=np.float32))
    assert low_armed and not low_paused and float(v_low_drift_ref[2]) < 0.0, v_low_drift_ref

    # A near-ground gust initiates a recapture to the 1.5 m near-ground hold.
    p_windy = np.array([0.10, 0.05, 1.20], dtype=np.float32)

    # A moderate gust now starts a full recapture cycle. Near the ground the safe
    # recovery target is 1.5 m; the transition frame itself holds smoothly.
    _, v_windy_ref, windy_armed, windy_timer, windy_z, _, windy_paused = _depot_landing_guidance(
        p_windy, np.zeros(3, dtype=np.float32), depot_home, 24.0, True, 0.4, 0.016,
        np.array([6.0, 0.0, 0.0], dtype=np.float32))
    assert not windy_armed and windy_paused and windy_timer == 0.0
    assert abs(float(windy_z) - DEPOT_LANDING_LOW_ALT_HOLD_HEIGHT_M) < 1e-6
    assert abs(float(v_windy_ref[2])) < 1e-8

    # Strong gust has the same re-capture semantics, not an arbitrary current-altitude
    # latch. The next calm frame requests movement back to the 1.5 m near-ground hold.
    _, v_hard_wind_ref, hard_wind_armed, hard_wind_timer, hard_wind_z, _, hard_wind_paused = _depot_landing_guidance(
        p_windy, np.zeros(3, dtype=np.float32), depot_home, 24.0, True, 0.4, 0.016,
        np.array([8.0, 0.0, 0.0], dtype=np.float32))
    assert not hard_wind_armed and hard_wind_paused and hard_wind_timer == 0.0, (hard_wind_armed, hard_wind_timer)
    assert abs(float(hard_wind_z) - DEPOT_LANDING_LOW_ALT_HOLD_HEIGHT_M) < 1e-6
    assert abs(float(v_hard_wind_ref[2])) < 1e-8

    # A gust at normal descent height must restore the original staging target, while
    # the next calm frame commands both pad-centering and climb-back to staging altitude.
    p_mid_descent = np.array([2.0, -1.0, 18.0], dtype=np.float32)
    v_mid_descent = np.array([0.2, -0.1, -0.4], dtype=np.float32)
    _, _, mid_armed, _, mid_stage_z, _, mid_paused = _depot_landing_guidance(
        p_mid_descent, v_mid_descent, depot_home, 24.0, True, 0.3, 0.016,
        np.array([6.0, 0.0, 0.0], dtype=np.float32))
    assert not mid_armed and mid_paused and abs(float(mid_stage_z) - 24.0) < 1e-6
    p_recap_ref, v_recap_ref, recap_armed, recap_timer, recap_stage_z, _, recap_paused = _depot_landing_guidance(
        p_mid_descent, np.zeros(3, dtype=np.float32), depot_home, mid_stage_z, False, 0.0, 0.016,
        np.zeros(3, dtype=np.float32))
    assert not recap_armed and abs(float(recap_stage_z) - 24.0) < 1e-6
    assert float(v_recap_ref[2]) > 0.0, v_recap_ref
    assert float(np.dot(v_recap_ref[:2], depot_home - p_mid_descent[:2])) > 0.0, v_recap_ref

    # A large wind-induced staging-altitude error must re-enable position feedback
    # and request bounded recovery rather than remaining at the old +/-0.32 m/s cap.
    p_high_stage_error = np.array([0.10, 0.10, 36.0], dtype=np.float32)
    _, v_stage_recover, stage_recover_armed, *_ = _depot_landing_guidance(
        p_high_stage_error, np.zeros(3, dtype=np.float32), depot_home, 24.0,
        False, 0.0, 0.016, np.zeros(3, dtype=np.float32))
    assert not stage_recover_armed and abs(float(v_stage_recover[2])) >= 0.54

    # Vertical gusts must initiate recapture even when horizontal wind is calm.
    _, v_vertical_gust, vertical_gust_armed, vertical_gust_timer, vertical_gust_z, _, vertical_gust_paused = _depot_landing_guidance(
        p_windy, np.zeros(3, dtype=np.float32), depot_home, 24.0, True, 0.0, 0.016,
        np.array([0.0, 0.0, 1.50], dtype=np.float32))
    assert not vertical_gust_armed and vertical_gust_paused and vertical_gust_timer == 0.0
    assert abs(float(vertical_gust_z) - DEPOT_LANDING_LOW_ALT_HOLD_HEIGHT_M) < 1e-6
    assert abs(float(v_vertical_gust[2])) < 1e-8
    _, v_vertical_hard, vertical_hard_armed, vertical_hard_timer, vertical_hard_z, _, vertical_hard_paused = _depot_landing_guidance(
        p_windy, np.zeros(3, dtype=np.float32), depot_home, 24.0, True, 0.4, 0.016,
        np.array([0.0, 0.0, 2.50], dtype=np.float32))
    assert not vertical_hard_armed and vertical_hard_paused and vertical_hard_timer == 0.0
    assert abs(float(vertical_hard_z) - DEPOT_LANDING_LOW_ALT_HOLD_HEIGHT_M) < 1e-6
    assert abs(float(v_vertical_hard[2])) < 1e-8

    class _VerticalSmootherProbe:
        vertical_v_cmd_smooth = -0.42
        vertical_a_cmd_smooth = -0.55
    _smoother_probe = _VerticalSmootherProbe()
    _seed_vertical_command_smoother(_smoother_probe, 0.31)
    assert abs(_smoother_probe.vertical_v_cmd_smooth - 0.31) < 1e-8
    assert _smoother_probe.vertical_a_cmd_smooth == 0.0
    _seed_vertical_command_smoother(_smoother_probe, float("nan"))
    assert _smoother_probe.vertical_v_cmd_smooth == 0.0
    assert _smoother_probe.vertical_a_cmd_smooth == 0.0

    # Contact projection regression: if the drone is settled over the pad and the
    # cable is long enough, a centred cargo parcel must be projected to exact contact
    # height and satisfy the same release geometry gate.
    test_roofs = torch.zeros((1, 3, 3), dtype=torch.float32)
    test_roofs[..., 0] = torch.tensor([0.0, 10.0, 20.0])
    test_roofs[..., 2] = 5.0
    test_drone = test_roofs.clone()
    test_drone[..., 2] += 10.0
    test_vel = torch.zeros_like(test_drone)
    test_cargo = test_drone.clone()
    test_cargo[..., 2] -= 4.0
    test_cable = torch.full((1, 3), 10.0, dtype=torch.float32)
    test_payload = torch.ones((1, 3), dtype=torch.bool)
    test_winch = torch.ones((1, 3), dtype=torch.long)
    test_phase = torch.full((1, 3), 3, dtype=torch.long)
    projected, projection_mask = _phase3_project_tether_and_contact(
        test_drone, test_vel, test_cargo, test_cable, test_payload,
        test_winch, test_phase, test_roofs)
    assert bool(projection_mask.all().item()), projection_mask
    assert bool(_phase3_cargo_contact_mask(projected, test_roofs).all().item()), projected

    # Distinguish tight release centring from the broader physical pad safety zone.
    assert DELIVERY_CARGO_CENTER_TOLERANCE_M < PHASE3_CARGO_SAFE_RADIUS_M < 2.2
    assert PHASE3_CARGO_ZONE_ABORT_DWELL_S > 0.0
    assert all(math.isfinite(v) for v in (POWERLINE_X_MIN_M, POWERLINE_X_MAX_M, POWERLINE_Z_M))

    # Phase-3 now carries the live rotation and gyro feature contract instead of
    # hardcoding identity orientation and zero angular velocity.
    q90 = torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], dtype=torch.float32)
    R90 = _phase3_torch_rotation_from_quaternion(q90.reshape(1, 1, 4).expand(1, 3, 4))
    omega_test = torch.zeros_like(test_pos)
    omega_test[0, 0, 2] = 1.0
    rot_features = _phase3_torch_features(
        pos=test_pos, vel=zeros3, waypoint=zeros3, roof_z=zeros2,
        cable_len=zeros2, cable_vel=zeros2, has_payload=false_payload,
        winch_state=zeros2, phase=zeros2, wind=zeros3, accel=zeros3,
        building_aabbs=np.empty((0, 5), dtype=np.float32),
        rotation_matrix=R90, omega_body=omega_test)
    assert abs(float(rot_features[0, 0, 10]) + 1.0) < 1e-5, rot_features[0, 0, 9:18]
    assert abs(float(rot_features[0, 0, 12]) - 1.0) < 1e-5, rot_features[0, 0, 9:18]
    assert abs(float(rot_features[0, 0, 20]) - 0.1) < 1e-5, rot_features[0, 0, 18:21]

    # Inverse-mass cable correction allocates more constraint motion to the light
    # line node than to the loaded terminal package.
    cable_mass, payload_mass = 0.04, CARGO_MASS_KG
    inv_cable, inv_payload = 1.0/cable_mass, 1.0/payload_mass
    frac_cable = inv_cable / (inv_cable + inv_payload)
    frac_payload = inv_payload / (inv_cable + inv_payload)
    assert frac_cable > frac_payload and abs(frac_cable + frac_payload - 1.0) < 1e-6

    # New thermal/NDVI cues should peak at the actual tree locations, not the stale
    # historical park coordinate. The sensor class is tied to a real world instance.
    sensor_world = MetropolitanCityGridWorld400m()
    thermal_sensor = AdvancedMultiModalSensorSuite(sensor_world)
    tree_xy = sensor_world.trees[0]["pos"][:2]
    nearby_thermal, _, nearby_ndvi = thermal_sensor.poll_thermal_and_polarization(tree_xy)
    far_thermal, _, far_ndvi = thermal_sensor.poll_thermal_and_polarization(np.array([150.0, 150.0]))
    assert nearby_ndvi > far_ndvi, (nearby_ndvi, far_ndvi)
    # Thermal delta also includes the separate rooftop heat term, so its total is
    # not required to be monotonic with distance from vegetation alone.
    assert math.isfinite(nearby_thermal) and math.isfinite(far_thermal)

    q0 = np.array([0.0, -3.0, 2.0], dtype=np.float64)
    q1 = np.array([20.0, 5.0, 7.0], dtype=np.float64)
    qdur = quintic_duration_for_limits(q0, q1, 4.5, 1.2)
    qmetrics = evaluate_quintic_trajectory(q0, q1, qdur, sample_count=201)
    q_start = quintic_trajectory_reference(q0, q1, qdur, 0.0)
    q_end = quintic_trajectory_reference(q0, q1, qdur, qdur)
    assert np.allclose(q_start[0], q0) and np.allclose(q_end[0], q1)
    assert np.linalg.norm(q_start[1]) < 1e-9 and np.linalg.norm(q_end[1]) < 1e-9
    assert np.linalg.norm(q_start[2]) < 1e-9 and np.linalg.norm(q_end[2]) < 1e-8
    assert qmetrics["finite"] and qmetrics["monotonic_progress"] and qmetrics["endpoint_error_m"] < 1e-8
    assert qmetrics["peak_speed_mps"] <= 4.5+0.03 and qmetrics["peak_acceleration_mps2"] <= 1.2+0.03
    try:
        quintic_trajectory_reference(q0, q1, 0.0, 0.2)
        raise AssertionError("invalid quintic duration should fail")
    except ValueError:
        pass
    action_target_valid = _numpy_one_step_action_transition_target(
        np.array([0.0, 0.0, 10.0]), np.array([1.0, 0.0, 0.0]), np.zeros(3),
        np.array([0.016, 0.0, 10.0]), np.array([1.02, 0.0, 0.0]), np.array([0.0, 0.01, 0.0]),
        0.016, transition_valid=True, contact_free=True)
    assert action_target_valid.shape == (ACTION_TRANSITION_TARGET_DIM,) and action_target_valid[-1] == 1.0
    action_target_invalid = _numpy_one_step_action_transition_target(
        np.zeros(3), np.zeros(3), np.zeros(3), np.array([2.0, 0.0, 1.0]),
        np.zeros(3), np.zeros(3), 0.016, transition_valid=True, contact_free=False)
    assert action_target_invalid[-1] == 0.0 and np.allclose(action_target_invalid[:-1], 0.0)
    if not _LIGHTWEIGHT_HARVEST_CHILD:
        action_transition_targets = torch.zeros((4, ACTION_TRANSITION_TARGET_DIM), dtype=torch.float32)
        action_transition_targets[:, -1] = torch.tensor([1.0, 1.0, 0.0, 1.0])
        action_transition_prediction = torch.zeros((4, ACTION_TRANSITION_OUTPUT_DIM), dtype=torch.float32, requires_grad=True)
        action_transition_loss = _action_conditioned_transition_loss(
            action_transition_prediction, action_transition_targets)
        assert torch.isfinite(action_transition_loss).item()
        action_transition_loss.backward()
        assert action_transition_prediction.grad is not None
        assert torch.isfinite(action_transition_prediction.grad).all().item()
        action_transition_stats = _action_conditioned_transition_validation_stats(
            action_transition_prediction.detach(), action_transition_targets)
        assert action_transition_stats["count"] == 3
        assert action_transition_stats["delta_position_rmse_m"] == 0.0
        # A constant-velocity baseline must be exact for an ideal no-acceleration transition.
        baseline_obs = torch.zeros((2, OBSERVATION_DIM), dtype=torch.float32)
        baseline_c_idx = (HISTORY_LEN - 1) * BASE_OBS_DIM
        baseline_velocity = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 0.25, 0.0]], dtype=torch.float32)
        baseline_obs[:, baseline_c_idx+6:baseline_c_idx+9] = baseline_velocity / 8.0
        baseline_targets = torch.zeros((2, ACTION_TRANSITION_TARGET_DIM), dtype=torch.float32)
        baseline_targets[:, 0:3] = baseline_velocity * 0.016 / ACTION_TRANSITION_POSITION_SCALE_M
        baseline_targets[:, -1] = 1.0
        baseline_stats = _action_conditioned_transition_constant_velocity_baseline_stats(
            baseline_obs, baseline_targets, dt=0.016)
        assert baseline_stats["count"] == 2
        assert baseline_stats["delta_position_rmse_m"] < 1e-7
        assert baseline_stats["delta_velocity_rmse_mps"] == 0.0
        assert baseline_stats["delta_omega_rmse_rads"] == 0.0
        # V38.42: live and offline baseline must decode the same observed velocity,
        # never the simulator's hidden true velocity.
        baseline_numpy_si = _numpy_action_transition_constant_velocity_baseline_si(baseline_obs, dt=0.016)
        expected_baseline_dp = baseline_velocity.numpy() * 0.016
        assert baseline_numpy_si.shape == (2, ACTION_TRANSITION_OUTPUT_DIM)
        assert np.allclose(baseline_numpy_si[:, 0:3], expected_baseline_dp, atol=1e-9)
        assert np.allclose(baseline_numpy_si[:, 3:9], 0.0, atol=1e-12)
        contract_targets = torch.zeros((2, ACTION_TRANSITION_TARGET_DIM), dtype=torch.float32)
        contract_targets[:, -1] = 1.0
        contract_stats = _action_transition_action_input_stats(test_action_condition, contract_targets)
        assert contract_stats["valid_paired_rows"] == 2
        assert sum(contract_stats["fsm_action_counts"].values()) == 2
        assert contract_stats["guidance_features"]["norm_mean"] is not None
        assert _action_transition_relative_gain_pct(2.0, 1.0) == 50.0
        assert _action_transition_relative_gain_pct(1.0, 2.0) == -100.0
        # Stratified metrics must match labels and correctly handle low-count groups.
        strat_n = 36
        strat_obs = torch.zeros((strat_n, OBSERVATION_DIM), dtype=torch.float32)
        strat_actions_guidance = torch.zeros((strat_n, 4), dtype=torch.float32)
        strat_actions_fsm = torch.arange(strat_n, dtype=torch.long) % NUM_EXECUTIVE_STATES
        strat_actions_winch = torch.zeros((strat_n, 2), dtype=torch.float32)
        strat_actions_winch[:, 0] = torch.tensor([-1.0, 0.0, 1.0] * 12)
        strat_actions = _build_action_condition_tensor(
            strat_actions_guidance, strat_actions_fsm, strat_actions_winch)
        strat_cidx = (HISTORY_LEN - 1) * BASE_OBS_DIM
        for i in range(strat_n):
            wind_speed_test = (0.0, 3.0, 7.0)[i % 3]
            strat_obs[i, strat_cidx+42:strat_cidx+45] = torch.tensor([wind_speed_test / 8.0, 0.0, 0.0])
        strat_target = torch.zeros((strat_n, ACTION_TRANSITION_TARGET_DIM), dtype=torch.float32)
        strat_target[:, -1] = 1.0
        strat_pred = torch.zeros((strat_n, ACTION_TRANSITION_OUTPUT_DIM), dtype=torch.float32)
        stratified_test = _action_transition_stratified_validation_stats(
            strat_pred, strat_target, strat_obs, strat_actions, min_group_samples=4)
        assert len(stratified_test) == NUM_EXECUTIVE_STATES + 6
        assert all("count" in entry and entry["count"] >= 4 for entry in stratified_test)
        assert all(entry.get("adequate", False) for entry in stratified_test)
        assert all(math.isfinite(entry["learned"]["delta_position_rmse_m"]) for entry in stratified_test)
        # V38.34 readiness must require adequate core groups and non-regressive metrics,
        # and must never authorize live actuation even when every offline gate passes.
        gate_required = [
            ("fsm", "HOVER_LOCK"), ("fsm", "WINCH_LOWER"), ("fsm", "WINCH_RETRACT"),
            ("fsm", "DEPOT_SERVICE"), ("winch", "payout"), ("winch", "retract"),
            ("wind", "strong_ge_5p5mps")]
        gate_groups = [{"group_type": gt, "group": gn, "count": 8, "adequate": True,
                        "gain_vs_baseline_pct": {"position": 2.0, "velocity": 1.0, "angular_rate": 0.5}}
                       for gt, gn in gate_required]
        gate_sensitivity = {name: {"predicted_delta_position_response_rms_m": 0.01,
                                   "predicted_delta_velocity_response_rms_mps": 0.02,
                                   "predicted_delta_angular_rate_response_rms_rads": 0.01}
                            for name in ("guidance_perturbation", "fsm_perturbation")}
        passing_gate = _action_transition_rollout_readiness_gate(
            2048, {"position": 2.0, "velocity": 1.0, "angular_rate": 0.5},
            gate_groups, gate_sensitivity, min_overall_rows=1000, min_group_samples=8)
        assert passing_gate["ready_for_offline_multistep_evaluation"]
        assert passing_gate["live_control_authorized"] is False
        regressed_groups = [dict(entry, gain_vs_baseline_pct={"position": -1.0, "velocity": 1.0,
                                                               "angular_rate": 0.5})
                            if entry["group_type"] == "wind" else entry for entry in gate_groups]
        blocked_gate = _action_transition_rollout_readiness_gate(
            2048, {"position": 2.0, "velocity": 1.0, "angular_rate": 0.5},
            regressed_groups, gate_sensitivity, min_overall_rows=1000, min_group_samples=8)
        assert not blocked_gate["ready_for_offline_multistep_evaluation"]
        assert any("strong_ge_5p5mps" in reason for reason in blocked_gate["reasons"])
        insufficient_gate = _action_transition_rollout_readiness_gate(
            10, {"position": 2.0, "velocity": 1.0, "angular_rate": 0.5},
            gate_groups[:-1], gate_sensitivity, min_overall_rows=1000, min_group_samples=8)
        assert not insufficient_gate["ready_for_offline_multistep_evaluation"]
        safe_json = _action_transition_json_safe({"nan": float("nan"), "inf": float("inf"), "ok": 1.0})
        assert safe_json == {"nan": None, "inf": None, "ok": 1.0}
        # Exercise the actual network branch, not only the standalone loss tensor:
        # both the action encoder and transition head must receive finite gradients.
        transition_gradient_model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM)
        transition_gradient_model.zero_grad(set_to_none=True)
        transition_model_outputs = transition_gradient_model(
            test_input, return_action_transition=True,
            transition_action=test_action_condition)
        transition_model_prediction = transition_model_outputs[-1]
        transition_model_targets = torch.zeros((2, ACTION_TRANSITION_TARGET_DIM), dtype=torch.float32)
        transition_model_targets[:, -1] = 1.0
        transition_model_loss = _action_conditioned_transition_loss(
            transition_model_prediction, transition_model_targets)
        transition_model_loss.backward()
        action_encoder_grad = transition_gradient_model.action_transition_action_encoder[0].weight.grad
        transition_head_grad = transition_gradient_model.action_conditioned_transition_head[-1].weight.grad
        assert action_encoder_grad is not None and torch.isfinite(action_encoder_grad).all().item()
        assert transition_head_grad is not None and torch.isfinite(transition_head_grad).all().item()
        sensitivity_test = _action_transition_action_sensitivity_probe(
            transition_gradient_model, test_input, test_action_condition,
            transition_model_targets, max_samples=2)
        assert sensitivity_test["count"] == 2
        assert set(sensitivity_test["guidance_perturbation"]) == {
            "predicted_delta_position_response_rms_m",
            "predicted_delta_velocity_response_rms_mps",
            "predicted_delta_angular_rate_response_rms_rads"}
        assert all(math.isfinite(float(v)) for v in sensitivity_test["guidance_perturbation"].values())
        assert all(math.isfinite(float(v)) for v in sensitivity_test["fsm_perturbation"].values())
        del transition_gradient_model
        all_invalid_action_transition_targets = torch.zeros((2, ACTION_TRANSITION_TARGET_DIM), dtype=torch.float32)
        all_invalid_action_transition_prediction = torch.zeros((2, ACTION_TRANSITION_OUTPUT_DIM), dtype=torch.float32, requires_grad=True)
        all_invalid_action_transition_loss = _action_conditioned_transition_loss(
            all_invalid_action_transition_prediction, all_invalid_action_transition_targets)
        assert all_invalid_action_transition_loss.item() == 0.0 and all_invalid_action_transition_loss.requires_grad
        trajectory_targets = torch.zeros((5, TRAJECTORY_FORECAST_TARGET_DIM), dtype=torch.float32)
        for hidx in range(TRAJECTORY_FORECAST_HORIZON_COUNT):
            trajectory_targets[:, TRAJECTORY_FORECAST_DIM+hidx] = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
        trajectory_prediction = torch.zeros((5, TRAJECTORY_FORECAST_DIM), dtype=torch.float32, requires_grad=True)
        trajectory_loss = _trajectory_forecast_loss(trajectory_prediction, trajectory_targets)
        assert torch.isfinite(trajectory_loss).item()
        trajectory_loss.backward()
        assert trajectory_prediction.grad is not None and torch.isfinite(trajectory_prediction.grad).all().item()
        trajectory_stats = _trajectory_forecast_validation_stats(trajectory_prediction.detach(), trajectory_targets)
        assert len(trajectory_stats) == 3 and trajectory_stats[0]["count"] == 4
        all_invalid_targets = torch.zeros((2, TRAJECTORY_FORECAST_TARGET_DIM), dtype=torch.float32)
        all_invalid_prediction = torch.zeros((2, TRAJECTORY_FORECAST_DIM), dtype=torch.float32, requires_grad=True)
        all_invalid_loss = _trajectory_forecast_loss(all_invalid_prediction, all_invalid_targets)
        assert all_invalid_loss.item() == 0.0 and all_invalid_loss.requires_grad

    # V38.35 held-out teacher-forced accumulated multi-step transition metric.
    rollout_test_obs = np.zeros((1, 80, OBSERVATION_DIM), dtype=np.float32)
    rollout_test_actions = np.zeros((1, 80, ACTION_TRANSITION_ACTION_DIM), dtype=np.float32)
    rollout_test_targets = np.zeros((1, 80, ACTION_TRANSITION_TARGET_DIM), dtype=np.float32)
    rollout_test_predictions = np.zeros((1, 80, ACTION_TRANSITION_OUTPUT_DIM), dtype=np.float32)
    rollout_test_targets[0, :, 0] = 0.4  # 0.1 m observed position delta per step
    rollout_test_predictions[0, :, 0] = 0.4
    rollout_test_targets[0, :, -1] = 1.0
    rollout_test_stats = _transition_multistep_rollout_metrics(
        rollout_test_obs, rollout_test_actions, rollout_test_targets,
        rollout_test_predictions, np.asarray([80]), horizons_steps=(1, 4, 8), dt_s=0.016)
    assert rollout_test_stats["status"] == "COMPLETED"
    assert [x["valid_windows"] for x in rollout_test_stats["horizons"]] == [80, 77, 73]
    assert rollout_test_stats["horizons"][1]["learned_teacher_forced_accumulation"]["position_delta_rmse_m"] < 1e-7
    assert rollout_test_stats["horizons"][1]["constant_velocity_accumulation_baseline"]["position_delta_rmse_m"] > 0.0
    assert rollout_test_stats["free_running_rollout"] is False

    # V38.37 same-drone next-observation target pairing and masked dynamic-state loss.
    env_obs = np.zeros((NUM_SWARM_DRONES * 4, OBSERVATION_DIM), dtype=np.float32)
    env_kin = np.zeros((len(env_obs), 7), dtype=np.float32)
    env_kin[:, 6] = 1.0
    latest_idx = (HISTORY_LEN - 1) * BASE_OBS_DIM
    for frame in range(4):
        for drone_idx in range(NUM_SWARM_DRONES):
            row = frame * NUM_SWARM_DRONES + drone_idx
            base = env_obs[row, latest_idx:latest_idx + BASE_OBS_DIM]
            base[42:45] = np.array([0.10, -0.05, 0.0], dtype=np.float32) + frame * np.array([0.01, 0.0, 0.0], dtype=np.float32)
            base[53] = 0.20 + 0.001 * frame
            base[54] = 0.10
            base[65] = 0.0
    env_targets = _numpy_one_step_environment_transition_targets(env_obs, env_kin)
    assert env_targets.shape == (len(env_obs), ENVIRONMENT_TRANSITION_TARGET_DIM)
    assert int((env_targets[:, -1] > 0.5).sum()) == len(env_obs) - NUM_SWARM_DRONES
    assert np.allclose(env_targets[0, 0], env_obs[NUM_SWARM_DRONES, latest_idx + 42])
    env_pred = torch.zeros((4, ENVIRONMENT_TRANSITION_DIM), dtype=torch.float32, requires_grad=True)
    env_targ = torch.zeros((4, ENVIRONMENT_TRANSITION_TARGET_DIM), dtype=torch.float32)
    env_targ[:, -1] = torch.tensor([1.0, 1.0, 0.0, 1.0])
    env_loss = _environment_transition_prediction_loss(env_pred, env_targ)
    assert torch.isfinite(env_loss).item()
    env_loss.backward()
    assert env_pred.grad is not None and torch.isfinite(env_pred.grad).all().item()
    env_stats = _environment_transition_validation_stats(env_pred.detach(), env_targ)
    assert env_stats["count"] == 3
    # Recursive updater must write only the direct channels represented by its heads.
    env_state = np.tile(np.array([[0.25, -0.10, 0.05, 0.20, -0.10, 0.35]], dtype=np.float32), (2, 1))
    cargo_state = np.tile(np.array([[0.10, -0.05, 0.02, -0.01, 0.20]], dtype=np.float32), (2, 1))
    obs_for_env = np.zeros((2, OBSERVATION_DIM), dtype=np.float32)
    frames_for_env = obs_for_env.reshape(2, HISTORY_LEN, BASE_OBS_DIM)
    frames_for_env[:, -1, 9:18] = np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (2, 1))
    frames_for_env[:, -1, 58] = 1.0
    frames_for_env[:, -1, 53] = 0.20
    env_updated, _ = _recursive_update_observation_batch(
        obs_for_env, np.zeros((2, ACTION_TRANSITION_OUTPUT_DIM), dtype=np.float32),
        predicted_environment_state=env_state, predicted_cargo_state=cargo_state)
    env_latest = env_updated.reshape(2, HISTORY_LEN, BASE_OBS_DIM)[:, -1, :]
    assert np.allclose(env_latest[:, 42:45], env_state[:, 0:3])
    assert np.allclose(env_latest[:, 53:55], env_state[:, 3:5])
    assert np.allclose(env_latest[:, 65], env_state[:, 5])
    assert np.allclose(env_latest[:, 56:58], cargo_state[:, 0:2] * CARGO_DYNAMICS_SWAY_SCALE_M / 2.0)
    assert np.allclose(env_latest[:, 55], cargo_state[:, 4] * CARGO_DYNAMICS_TENSION_SCALE_N / 50.0)
    assert _recursive_rollout_self_test()
    assert _shadow_policy_rollout_self_test()
    # V38.41 persistent DAgger corpus round-trip and whole-environment holdout.
    _test_groups = np.repeat(np.arange(10, dtype=np.int64), 3)
    _test_train, _test_val, _test_val_groups = _phase3_group_heldout_split(
        _test_groups, np.random.default_rng(41), val_fraction=0.2)
    assert len(_test_train) > 0 and len(_test_val) > 0
    assert not set(np.unique(_test_groups[_test_train])).intersection(set(np.unique(_test_groups[_test_val])))
    assert set(_test_val_groups) == set(np.unique(_test_groups[_test_val]))
    _test_n = 30
    _test_arrays = {
        "X": np.zeros((_test_n, OBSERVATION_DIM), dtype=np.float32),
        "Yg": np.zeros((_test_n, 4), dtype=np.float32),
        "Yf": (np.arange(_test_n) % max(1, NUM_EXECUTIVE_STATES)).astype(np.int64),
        "Yw": np.zeros((_test_n, 2), dtype=np.float32),
        "Yh": np.zeros((_test_n, 8), dtype=np.float32),
        "Yt": np.zeros((_test_n, 2), dtype=np.float32),
        "Yv": np.zeros((_test_n, 1), dtype=np.float32),
        "Ykin": np.zeros((_test_n, 7), dtype=np.float32),
        "Ycargo": np.zeros((_test_n, 6), dtype=np.float32),
        "Yforce": np.zeros((_test_n, 4), dtype=np.float32),
    }
    _test_old_cap = os.environ.pop("SURVIVAL_DAGGER_REPLAY_MAX_SAMPLES", None)
    try:
        with tempfile.TemporaryDirectory(prefix="survival_dagger_test_") as _test_dir:
            _test_replay_path = os.path.join(_test_dir, "dagger.npz")
            _phase3_save_dagger_replay(_test_replay_path, _test_arrays, _test_groups, max_samples=17)
            _test_loaded = _phase3_load_dagger_replay(_test_replay_path, _test_arrays)
            assert _test_loaded is not None
            assert len(_test_loaded["X"]) == 17
            assert len(_test_loaded["source_group_ids"]) == 17
            assert set(np.unique(_test_loaded["Yf"])).issuperset(set(np.unique(_test_arrays["Yf"])))
    finally:
        if _test_old_cap is not None:
            os.environ["SURVIVAL_DAGGER_REPLAY_MAX_SAMPLES"] = _test_old_cap
    print("[SELF-TEST PASS] V38.42 observation-only live/offline baseline parity and action-input contract telemetry; V38.41 group-held-out DAgger replay schema/cap and V38.40 recomputed-guidance/winch shadow action construction and fixed-FSM target-preservation tests; V38.38 recursive rollout error distribution/bootstrapped RMSE intervals/stratified subgroup metrics and MPC authorization block; V38.37 same-drone dynamic-observation target alignment/masking and environment prediction gradients; action-conditioned overall and stratified validation; action-input sensitivity probe; JSON metric sanitisation; observation contract; live/Genesis attitude features; inverse-mass cable correction; stale vegetation coordinates; powerline geometry; finite-segment distance; radar aspect/clip; anti-sway signs; bounded vertical policy residual; cable-reach/contact projection; weighted delivery curriculum; target capture; active-winched departure interlock; depot inbound braking; wind-gated/low-altitude depot landing; staged velocity-aware vertical depot landing; stable depot capture momentum gates; vertical-wind landing interlocks; weather-triggered depot recovery; per-drone mission context isolation; quiet Genesis logging; depot reset contracts; safe task ownership; memory-bounded Phase-3 sampling; kinematic equation-residual loss/masks; Newtonian force-balance residual target/loss/masks and checkpoint marker; rotational Euler residual target/loss/masks and trained-head checkpoint marker; detached heteroscedastic uncertainty head, masked Gaussian NLL, calibration coverage and checkpoint markers; multi-horizon trajectory forecast target masks/loss/physical-unit metrics; action-conditioned paired transition targets/loss/metrics/action encoder and checkpoint marker; V38.35 worker-held-out sequence artifact schema and teacher-forced multistep accumulated-error evaluator; quintic endpoint/speed/acceleration and monotonicity tests; offline finite-candidate predictive-control surrogate, constraints, deterministic scoring and no-feasible-action handling.", flush=True)


def run_single_drone_waypoint_route_self_test():
    """Test the real airframe and mixer on one calm-air waypoint/approach route.

    This is a controller/dynamics integration test only. It does not exercise the
    Pygame world, obstacle field, Genesis rollout, winch, contact or package release.
    """
    dt = 0.016
    wind = np.zeros(3, dtype=np.float32)
    airframe = BEMTCoaxialAirframe8DOF(np.array([-160.0, -155.0, 0.06], dtype=np.float32))
    mixer = RateDampedActuatorMixerWithCBF()
    target_xy = np.array([-165.0, -90.0], dtype=np.float32)
    roof_height_m = 14.5
    cruise_z = 42.0
    hover_z = roof_height_m + PHASE3_ROOF_PAD_TOP_OFFSET_M + 10.0
    phase = "1_TAKEOFF"
    descent_armed = False
    align_dwell = 0.0
    settle_dwell = 0.0
    last_logged_phase = None
    max_time_s = 120.0

    for step_idx in range(int(max_time_s / dt)):
        p = airframe.p.copy()
        v = airframe.v.copy()
        xy_error = float(np.linalg.norm(p[:2] - target_xy))
        xy_speed = float(np.linalg.norm(v[:2]))
        z_speed = abs(float(v[2]))
        if phase == "1_TAKEOFF":
            wp_target = np.array([-160.0, -155.0, cruise_z], dtype=np.float32)
            p_ref, v_ref = compute_vector_guidance(
                p, v, wp_target, max_speed_xy=0.0, max_climb=2.40, wind_vector=wind)
            if _cruise_altitude_captured(p[2], v[2], cruise_z):
                phase = "2_CRUISE_OUTBOUND"
        elif phase == "2_CRUISE_OUTBOUND":
            wp_target = np.array([target_xy[0], target_xy[1], cruise_z], dtype=np.float32)
            p_ref, v_ref = compute_vector_guidance(
                p, v, wp_target, max_speed_xy=4.5, max_climb=2.40, wind_vector=wind)
            if _should_enter_precision_approach(xy_error, xy_speed, 0.0):
                phase = "3_CHIMNEY_DESCENT"
                align_dwell = 0.0
                settle_dwell = 0.0
                mixer.xy_err_integral.fill(0.0)
                mixer.z_err_integral = 0.0
        elif phase == "3_CHIMNEY_DESCENT":
            if not descent_armed:
                wp_target = np.array([target_xy[0], target_xy[1], cruise_z], dtype=np.float32)
                speed_cap = min(1.8, 0.45 * xy_error,
                                math.sqrt(max(0.0, 2.0 * 1.1 * xy_error)))
                _, v_ref = compute_vector_guidance(
                    p, v, wp_target, max_speed_xy=speed_cap, max_climb=0.35,
                    max_descend=0.15, wind_vector=wind)
                # Fixed world-frame target, not a target that moves with the drone.
                p_ref = wp_target.copy()
                if xy_error <= DELIVERY_DRONE_CENTER_TOLERANCE_M:
                    v_ref[:2] = 0.0
                aligned = (xy_error <= DELIVERY_DRONE_CENTER_TOLERANCE_M
                           and abs(float(p[2]) - cruise_z) <= 0.65
                           and xy_speed <= DELIVERY_MAX_HORIZONTAL_SPEED_MS
                           and z_speed <= DELIVERY_MAX_VERTICAL_SPEED_MS)
                align_dwell = align_dwell + dt if aligned else 0.0
                if align_dwell >= 0.75:
                    descent_armed = True
                    mixer.xy_err_integral.fill(0.0)
            else:
                wp_target = np.array([target_xy[0], target_xy[1], hover_z], dtype=np.float32)
                p_ref, v_ref = compute_vector_guidance(
                    p, v, wp_target, max_speed_xy=0.0, max_climb=0.70,
                    max_descend=(0.65 if abs(float(p[2] - hover_z)) > 3.0
                                 else 0.40 if abs(float(p[2] - hover_z)) > 1.0 else 0.22),
                    wind_vector=wind)
                p_ref = np.array([target_xy[0], target_xy[1], hover_z], dtype=np.float32)
                v_ref[:2] = 0.0
                v_ref = _delivery_vertical_settle_reference(v_ref, abs(float(p[2] - hover_z)))
                at_hover = (xy_error <= DELIVERY_DRONE_CENTER_TOLERANCE_M
                            and abs(float(p[2] - hover_z)) <= 0.40
                            and xy_speed <= 0.30 and z_speed <= 0.20)
                settle_dwell = settle_dwell + dt if at_hover else 0.0
                if settle_dwell >= 0.60:
                    phase = "4_DELIVERY_HOVER"
        else:
            break

        if phase != last_logged_phase:
            print(f"[ROUTE TEST] t={step_idx * dt:.2f}s phase={phase} xy_error={xy_error:.2f}m altitude={p[2]:.2f}m speed_xy={xy_speed:.2f}m/s", flush=True)
            last_logged_phase = phase
        station_keeping = phase in ("3_CHIMNEY_DESCENT", "4_DELIVERY_HOVER")
        motor_cmd, _, _ = mixer.mix(
            drone_health=airframe.motor_health, p_est=airframe.p, v_est=airframe.v,
            R_est=airframe.R, omega_est=airframe.omega, p_target=p_ref,
            neural_guidance=np.zeros(4, dtype=np.float32), token_idx=0,
            estimated_wind=wind, obstacles=[], ref_vel=v_ref,
            has_payload=True, dt=dt, station_keeping=station_keeping,
            approach_braking=(phase == "3_CHIMNEY_DESCENT" and not descent_armed))
        airframe.step(motor_cmd, wind, 1.225, surface_z=0.0, has_payload=True, dt=dt)
        if phase == "4_DELIVERY_HOVER":
            final_xy_error = float(np.linalg.norm(airframe.p[:2] - target_xy))
            final_z_error = abs(float(airframe.p[2]) - hover_z)
            final_xy_speed = float(np.linalg.norm(airframe.v[:2]))
            final_z_speed = abs(float(airframe.v[2]))
            assert final_xy_error <= DELIVERY_DRONE_CENTER_TOLERANCE_M, final_xy_error
            assert final_z_error <= 0.40, final_z_error
            assert final_xy_speed <= 0.30, final_xy_speed
            assert final_z_speed <= 0.20, final_z_speed
            print(
                f"[ROUTE TEST PASS] reached and settled at target in {step_idx * dt:.2f}s; "
                f"xy_error={final_xy_error:.3f}m z_error={final_z_error:.3f}m "
                f"xy_speed={final_xy_speed:.3f}m/s vz={airframe.v[2]:+.3f}m/s. "
                "This test does not cover winch/contact/release or the full GUI/Genesis simulation.",
                flush=True)
            return

    raise RuntimeError(
        f"Single-drone waypoint route regression failed after {max_time_s:.1f}s: "
        f"phase={phase}, position={airframe.p.tolist()}, velocity={airframe.v.tolist()}, "
        f"xy_error={float(np.linalg.norm(airframe.p[:2] - target_xy)):.3f}m, "
        f"alignment_dwell={align_dwell:.2f}s, settle_dwell={settle_dwell:.2f}s")


if __name__ == "__main__":
    mp.freeze_support()
    if os.environ.get("SURVIVAL_SHADOW_POLICY_EVAL", "0").strip() == "1":
        run_fixed_script_self_tests()
        run_offline_shadow_policy_rollout_evaluation()
    elif os.environ.get("SURVIVAL_TRANSITION_RECURSIVE_EVAL", "0").strip() == "1":
        run_fixed_script_self_tests()
        run_offline_recursive_transition_rollout_evaluation()
    elif os.environ.get("SURVIVAL_TRANSITION_ROLLOUT_EVAL", "0").strip() == "1":
        run_fixed_script_self_tests()
        run_offline_transition_rollout_evaluation()
    elif os.environ.get("SURVIVAL_OFFLINE_MPC_EVAL", "0").strip() == "1":
        run_fixed_script_self_tests()
        run_offline_predictive_control_benchmark()
    elif os.environ.get("SURVIVAL_ROUTE_SELF_TEST", "0").strip() == "1":
        run_fixed_script_self_tests()
        run_single_drone_waypoint_route_self_test()
    elif os.environ.get("SURVIVAL_SELF_TEST", "0").strip() == "1":
        run_fixed_script_self_tests()
    elif os.environ.get("SURVIVAL_PHASE3_WORKER", "0").strip() == "1":
        _phase3_worker_main()
    else:
        run_all_physics_simulator()
