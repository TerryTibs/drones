"""
=============================================================================
UNIFIED MULTIMODAL HYDROFOIL AUTONOMY ENGINE (NCM-CBF-DSWP-BTT) -- FULL BUILD
=============================================================================
RESEARCH SIMULATION STACK (not flight-qualified):
  - 32-wave directional JONSWAP/PM sea with deep-water orbital kinematics
  - 15-state error-state EKF (IMU strapdown + mag + DVL + optical flow + GNSS)
  - FMCW radar / 3D boresight raycast / piezoresistive pressure-tap sensors
  - Active T-foil elevons + strut rudder, WIG wing, planing hull, PAR cushion
  - 4 independently 2-axis tilting lift rotors (0-15 deg omni-directional
    hub gimbal, Pitt-Peters inflow) + axial cruise pusher prop
  - 6S electro-thermal battery powering lift rotors, pusher, and sculling fin
  - CBF / contraction-metric safety filter (differentiable + embedded QP)
  - 20-state neural pilot + neural contraction metric (5D controllable subspace)
  - Morphogenetic airframe co-evolution genome

INCLUDED FIXES & UPGRADES:
  - Corrected pitch/mixer/EKF Jacobians, non-negative tap Cp, and causal L1 DOB
  - Thrust-preserving yaw-bounded mixer + 2-axis 0-15 deg omni rotor hub gimbals
  - Axial cruise pusher + faired foil sizing for level foil-borne cruise trim
  - Multi-mode & upset-recovery training across all 7 flight modes
  - Multi-timestep, plant-aware 5D NCM certification across full rollout
  - Causal FMCW radar crest bias, 3D boresight raycast, and 66.7 Hz sensor cadence
  - Full training & cockpit diagnostics + world-snapped scrolling 3D ocean grid
=============================================================================
"""

import argparse
import contextlib
import math
import os
import sys
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
try:
    import pygame
except ImportError:
    pygame = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


# =============================================================================
# LIVE SYNCHRONOUS LINE-BUFFERED DUAL-STREAM TELEMETRY LOGGER
# =============================================================================
class DualLogger:
    """Mirrors terminal stdout to disk, flushing on every write. Delegates unknown
    attributes (isatty, fileno, encoding...) to the real terminal stream."""
    def __init__(self, filename="flight_training_telemetry.txt"):
        self.terminal = sys.stdout
        self.log_file = open(filename, "w", encoding="utf-8")
        self.log_file.write(f"=== AEROMARINE AUTONOMY TELEMETRY SESSION INITIALIZED: {time.ctime()} ===\n\n")
        self.log_file.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        try:
            self.log_file.flush()
            self.log_file.close()
        except Exception:
            pass

    def isatty(self):
        try:
            return self.terminal.isatty()
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self.terminal, name)


# =============================================================================
# UNIVERSAL AMP & GRADSCALER COMPATIBILITY HELPERS
# =============================================================================
def make_grad_scaler(device, enabled=True, init_scale=128.0):
    if device.type != "cuda" or not enabled:
        class _DummyScaler:
            def scale(self, loss): return loss
            def unscale_(self, opt): pass
            def step(self, opt): opt.step()
            def update(self): pass
            def get_scale(self): return 1.0
        return _DummyScaler()

    try:
        return torch.amp.GradScaler("cuda", enabled=enabled, init_scale=init_scale)
    except (TypeError, ValueError, AttributeError):
        try:
            return torch.amp.GradScaler(device="cuda", enabled=enabled, init_scale=init_scale)
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=enabled, init_scale=init_scale)


def get_amp_autocast(device_str, enabled=True):
    """Returns an autocast context manager compatible across all PyTorch versions."""
    if device_str == "cuda" and torch.cuda.is_available():
        try:
            return torch.amp.autocast("cuda", enabled=enabled)
        except Exception:
            return torch.cuda.amp.autocast(enabled=enabled)
    elif device_str == "cpu":
        try:
            return torch.amp.autocast("cpu", enabled=enabled)
        except Exception:
            return contextlib.nullcontext()
    return contextlib.nullcontext()


def make_cosine(opt, lr, t_max, eta_min=1e-4):
    """CosineAnnealingLR reuses group['initial_lr']; reset it so restarts really restart."""
    for g in opt.param_groups:
        g["lr"] = lr
        g["initial_lr"] = lr
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, t_max), eta_min=eta_min)


# =============================================================================
# GLOBAL PHYSICAL, HYDRODYNAMIC & VEHICLE CONSTANTS
# =============================================================================
GRAVITY = 9.80665
RHO_AIR = 1.225         # kg/m^3
RHO_WATER_NOM = 1025.0  # kg/m^3
P_ATM = 101325.0        # Pa
P_VAPOR_WATER = 2338.0  # Pa
SIGMA_CAV_CRIT = 0.85

VEHICLE_MASS_NOM = 2.45     # kg
HOVER_WEIGHT_NOM = VEHICLE_MASS_NOM * GRAVITY  # ~24.03 N

# Base Nominal Aerodynamic Wing-in-Ground (WIG) Geometry
WING_SPAN_NOM = 0.92        # m
WING_CHORD_NOM = 0.22       # m
WING_AREA_NOM = WING_SPAN_NOM * WING_CHORD_NOM
WING_CL0 = 0.24
WING_CD0 = 0.022
WING_OSWALD = 0.82

# Base Nominal Ventral T-Foil Hydrofoil Geometry & Active Surfaces
FOIL_SPAN_NOM = 0.32        # m
FOIL_CHORD_NOM = 0.042      # m  (AR ~ 7.86 at nominal evolved span 0.33 m)
FOIL_AREA_NOM = FOIL_SPAN_NOM * FOIL_CHORD_NOM
FOIL_STRUT_LENGTH_NOM = 0.25
FOIL_CL_ALPHA = 4.50        # rad^-1 (finite-span corrected)
FOIL_CD0 = 0.012            # faired hydrofoil profile drag
HYDRO_SLAM_COEFF = 2.2
FOIL_LIFT_CLAMP_PER_SIDE = 16.0  # N
FOIL_DECALAGE_DEG = (1.1, 1.4)   # trims level flight at ~17-19 N foil lift
INIT_PITCH_NOSE_DOWN_DEG = 0.0   # level start; axial pusher carries cruise drag

# Submerged Hydro-Surface Effectiveness (Dual Elevons & Strut Rudder)
HYDRO_ELEVON_CHORD_FRAC = 0.30
HYDRO_CL_DELTA = 2.60
HYDRO_CD_DELTA = 0.35
HYDRO_RUDDER_AREA_NOM = 0.0075
HYDRO_RUDDER_CY_DELTA = 2.40
MAX_HYDRO_DEFLECTION = math.radians(15.0)
SERVO_RATE_LIMIT = 8.0       # rad/s

# Displacement Hull Geometry & Hydrodynamic Center
HULL_LENGTH = 0.65
HULL_BEAM = 0.18
HULL_DRAFT_MAX = 0.08
HULL_VOLUME_TOTAL = 0.0036
BOW_CENTER_OFFSET_X = 0.12

# Multi-Rotor Aeromechanics, 2-Axis Hub Gimbal, Axial Cruise Pusher & Power
PROP_RADIUS = 0.10
ROTOR_DISK_AREA = math.pi * (PROP_RADIUS**2)
TOTAL_ROTOR_AREA = 4.0 * ROTOR_DISK_AREA
V_INDUCED_HOVER = math.sqrt(HOVER_WEIGHT_NOM / (2.0 * RHO_AIR * TOTAL_ROTOR_AREA))
ARM_LENGTH = 0.24
NUM_BLADES = 2
BLADE_CHORD = 0.016
ROTOR_SOLIDITY = (NUM_BLADES * BLADE_CHORD) / (math.pi * PROP_RADIUS)
BLADE_CD0 = 0.012
LOCK_NUMBER = 5.2
BLADE_FLAP_FREQ = 1.10
BLADE_PITCH_ZERO = 0.16
MOTOR_POLE_PAIRS = 7

MAX_THRUST_PER_MOTOR = 14.5
MOTOR_IDLE_FLOOR = 0.80
TORQUE_TO_THRUST_RATIO = 0.015
MOTOR_TORQUE_CONSTANT = 0.020
MOTOR_COIL_RESISTANCE = 0.052
NO_LOAD_CURRENT = 0.70
MAX_JERK_ACCEL = 35.0

# Independent 2-Axis Omnidirectional Rotor Hub Gimbal (0 to 15 deg)
ROTOR_MAX_TILT_DEG = 15.0
ROTOR_MAX_TILT_RAD = math.radians(ROTOR_MAX_TILT_DEG)
ROTOR_MAX_TILT_SIN = math.sin(ROTOR_MAX_TILT_RAD)
ROTOR_MAX_TILT_COS = math.cos(ROTOR_MAX_TILT_RAD)
ROTOR_MAX_TILT_TAN = math.tan(ROTOR_MAX_TILT_RAD)
GIMBAL_TAU = 0.025           # s (1st-order servo time constant)

# Rear Axial Cruise Pusher Thruster Parameters
MAX_PUSHER_THRUST = 14.0     # N
PUSHER_RADIUS = 0.085        # m
PUSHER_DISK_AREA = math.pi * (PUSHER_RADIUS**2)
PUSHER_PROP_EFF = 0.72       # combined aerodynamic + motor/ESC efficiency

# 6S LiPo Electro-Thermal Thevenin Parameters
BATTERY_V0 = 25.2
BATTERY_CAPACITY_COULOMBS = 3.3 * 3600.0
BATTERY_R0_NOM = 0.022
BATTERY_R1 = 0.014
BATTERY_C1 = 15.0
BATTERY_R2 = 0.020
BATTERY_C2 = 500.0
BATTERY_THERMAL_CAP = 220.0
BATTERY_THERMAL_RES = 3.2
ARRHENIUS_COEFF = 450.0
T_AMBIENT_K = 298.15
AVIONICS_CURRENT = 1.4

# Flight Envelopes & Boundaries
RIDE_HEIGHT_OPTIMAL = 0.185
RIDE_HEIGHT_SOARING = 0.165
RIDE_HEIGHT_REPRIME = 0.160
RIDE_HEIGHT_MIN = 0.060
RIDE_HEIGHT_FOILING_MAX = 0.235
RIDE_HEIGHT_MAX = 0.235
RIDE_HEIGHT_WIG_MAX = 0.380
FREE_FLIGHT_ALTITUDE = 1.500
DSWP_LOOKAHEAD_TIME = 1.250
CONTROL_LOOP_DT = 0.005         # 200 Hz
FORWARD_RAY_DEPRESSION_RAD = math.radians(14.0)

# Reference step used to define dt-scaled filter/delay constants and 66.7 Hz sensor cadence
REF_DT = 0.015
GUIDANCE_DELAY_SECONDS = 0.030

# Frontier Idea 1 & 3 Physical Constants
PAR_PLENUM_EFF = 0.68
PAR_SEAL_GAIN = 1.95
SOARING_KATZMAYR_COEFF = 0.82

# Frontier Idea D: Caudal Fin Sculling Propulsion Constants (St ~ 0.24 at 5.2 m/s)
SCULL_FREQ_NOM = 22.0
SCULL_AMP_NOM = math.radians(18.0)
SCULL_LEVER_ARM = 0.090
SCULL_SERVO_POWER_COEFF = 0.45

# Hybrid Discrete Automaton Modes
MODE_LOITER = 0
MODE_TAKEOFF = 1
MODE_FOILING = 2
MODE_EGRESS = 3
MODE_FREE_FLIGHT = 4
MODE_INGRESS = 5
MODE_SOARING = 6

# Local Geomagnetic Field Vector (World NWU: [North, West, Up])
MAG_FIELD_WORLD = torch.tensor([0.42, 0.00, -0.90], device=DEVICE).view(1, 3)


def safe_norm(x, dim=-1, keepdim=True, eps=1e-8):
    """Euclidean norm preserving singleton dimensions."""
    return torch.sqrt(torch.sum(x * x, dim=dim, keepdim=keepdim) + eps)


def quat_to_rot(q):
    """Batch of unit quaternions [w, x, y, z] -> rotation matrices."""
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    r00 = 1.0 - 2.0 * (y**2 + z**2)
    r01 = 2.0 * (x*y - z*w)
    r02 = 2.0 * (x*z + y*w)
    r10 = 2.0 * (x*y + z*w)
    r11 = 1.0 - 2.0 * (x**2 + z**2)
    r12 = 2.0 * (y*z - x*w)
    r20 = 2.0 * (x*z - y*w)
    r21 = 2.0 * (y*z + x*w)
    r22 = 1.0 - 2.0 * (x**2 + y**2)
    row0 = torch.cat([r00, r01, r02], dim=-1)
    row1 = torch.cat([r10, r11, r12], dim=-1)
    row2 = torch.cat([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=1)


def quat_mult(q1, q2):
    """Batched quaternion multiplication q1 * q2."""
    w1, x1, y1, z1 = q1[:, 0:1], q1[:, 1:2], q1[:, 2:3], q1[:, 3:4]
    w2, x2, y2, z2 = q2[:, 0:1], q2[:, 1:2], q2[:, 2:3], q2[:, 3:4]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.cat([w, x, y, z], dim=-1)


def euler_to_rot(phi, theta, psi):
    """Right-hand Z(psi) Y(theta) X(phi) rotation. NOTE: +theta is NOSE-DOWN (x_body tilts toward -z)."""
    cp, sp = torch.cos(phi), torch.sin(phi)
    ct, st = torch.cos(theta), torch.sin(theta)
    cy, sy = torch.cos(psi), torch.sin(psi)

    r00 = cy * ct
    r01 = cy * st * sp - sy * cp
    r02 = cy * st * cp + sy * sp

    r10 = sy * ct
    r11 = sy * st * sp + cy * cp
    r12 = sy * st * cp - cy * sp

    r20 = -st
    r21 = ct * sp
    r22 = ct * cp

    row0 = torch.cat([r00, r01, r02], dim=-1)
    row1 = torch.cat([r10, r11, r12], dim=-1)
    row2 = torch.cat([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=1)


# =============================================================================
# FRONTIER IDEA E: MORPHOGENETIC AIRFRAME CO-EVOLUTION GENOME
# =============================================================================
class VehicleMorphologyGenome(nn.Module):
    """Differentiable CAD morphology genome (wing span/chord, foil span, strut, caudal area, anhedral)."""
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        self.raw_wing_span = nn.Parameter(torch.zeros(1, device=device))
        self.raw_wing_chord = nn.Parameter(torch.zeros(1, device=device))
        self.raw_foil_span = nn.Parameter(torch.zeros(1, device=device))
        self.raw_strut_len = nn.Parameter(torch.zeros(1, device=device))
        self.raw_caudal_area = nn.Parameter(torch.zeros(1, device=device))
        self.raw_anhedral_rad = nn.Parameter(torch.zeros(1, device=device))

    def forward(self):
        wing_span = 0.75 + 0.45 * torch.sigmoid(self.raw_wing_span)
        wing_chord = 0.16 + 0.12 * torch.sigmoid(self.raw_wing_chord)
        foil_span = 0.24 + 0.18 * torch.sigmoid(self.raw_foil_span)
        strut_len = 0.20 + 0.12 * torch.sigmoid(self.raw_strut_len)
        caudal_area = 0.004 + 0.010 * torch.sigmoid(self.raw_caudal_area)
        anhedral_rad = -0.08 + 0.16 * torch.sigmoid(self.raw_anhedral_rad)

        wing_area = wing_span * wing_chord
        foil_chord = FOIL_CHORD_NOM
        foil_area = foil_span * foil_chord
        aspect_ratio = (wing_span**2) / wing_area

        base_fixed_mass = 1.65
        structural_skin_density = 2.85
        total_mass = base_fixed_mass + structural_skin_density * (wing_area + foil_area + 2.0 * strut_len * 0.04)

        return {
            "wing_span": wing_span,
            "wing_chord": wing_chord,
            "wing_area": wing_area,
            "aspect_ratio": aspect_ratio,
            "foil_span": foil_span,
            "foil_chord": foil_chord,
            "foil_area": foil_area,
            "strut_length": strut_len,
            "caudal_area": caudal_area,
            "anhedral_rad": anhedral_rad,
            "total_mass": total_mass
        }


# =============================================================================
# 1. 32-WAVE OCEAN (PHYSICAL ENVIRONMENT TRUTH)
# =============================================================================
class DirectionalOceanWaveField:
    def __init__(self, dt, airspeed=5.20, num_waves=32, device=DEVICE):
        self.dt = dt
        self.device = device
        self.V = airspeed
        self.num_waves = num_waves

        freqs = torch.linspace(0.45, 2.40, num_waves, device=device)
        df = freqs[1] - freqs[0]

        alpha_jonswap = 0.0081
        s_pm = (alpha_jonswap * (GRAVITY**2) / (freqs**5)) * torch.exp(
            -0.74 * ((GRAVITY / (self.V * freqs))**4)
        )

        self.amplitudes_base = torch.sqrt(2.0 * s_pm * df).view(1, num_waves, 1)
        self.k = (freqs**2 / GRAVITY).view(1, num_waves, 1)
        self.omega = freqs.view(1, num_waves, 1)

        spread_angles = torch.linspace(-0.35, 0.35, num_waves, device=device)
        self.angles = spread_angles.view(1, num_waves, 1)
        self.phases = (torch.rand(num_waves, device=device) * (2.0 * math.pi)).view(1, num_waves, 1)

        self.kx = self.k * torch.cos(self.angles)
        self.ky = self.k * torch.sin(self.angles)

        self.L_w = 4.0
        self.sigma_w = 0.08 * self.V
        self.a_w = math.exp(-self.dt * self.V / self.L_w)
        self.b_w = self.sigma_w * math.sqrt(1.0 - self.a_w**2)
        self.a_g = math.exp(-self.dt / 0.18)
        self.b_g = 0.08 * math.sqrt((1.0 - self.a_g**2) / (1.0 - 0.92**2))
        self.gust_state = torch.zeros((1, 3), device=device)

        self.ray_body_vec = torch.tensor(
            [math.cos(FORWARD_RAY_DEPRESSION_RAD), 0.0, -math.sin(FORWARD_RAY_DEPRESSION_RAD)],
            device=device
        ).view(1, 3, 1)

    def reset(self, num_envs):
        self.gust_state = torch.zeros((num_envs, 3), device=self.device)
        self.phases = torch.rand((num_envs, self.num_waves, 1), device=self.device) * (2.0 * math.pi)

    def step_gust(self):
        N = self.gust_state.shape[0]
        white_noise = torch.randn((N, 3), device=self.device)
        u_g = self.a_g * self.gust_state[:, 0:1] + self.b_g * white_noise[:, 0:1]
        v_g = self.a_g * self.gust_state[:, 1:2] + self.b_g * white_noise[:, 1:2]
        w_g = self.a_w * self.gust_state[:, 2:3] + self.b_w * white_noise[:, 2:3]
        self.gust_state = torch.cat([u_g, v_g, w_g], dim=-1)
        return self.gust_state

    def evaluate_wave_kinematics(self, pos, t, wave_scale=1.0):
        px = pos[:, 0:1].unsqueeze(1)
        py = pos[:, 1:2].unsqueeze(1)
        pz = pos[:, 2:3]

        t_term = t.unsqueeze(1) if (torch.is_tensor(t) and t.ndim == 2) else t

        amps = self.amplitudes_base * wave_scale
        phase_angle = (self.kx * px + self.ky * py - self.omega * t_term + self.phases)
        sin_p = torch.sin(phase_angle)
        cos_p = torch.cos(phase_angle)

        eta = torch.sum(amps * cos_p, dim=1)
        deta_dx = -torch.sum(amps * self.kx * sin_p, dim=1)
        deta_dy = -torch.sum(amps * self.ky * sin_p, dim=1)
        deta_dt = torch.sum(amps * self.omega * sin_p, dim=1)

        clearance = F.relu(pz - eta)
        decay_air = torch.exp(-self.k * clearance.unsqueeze(1))
        w_air_updraft = torch.sum(amps * self.omega * sin_p * decay_air, dim=1)

        depth_sub = F.relu(eta - pz)
        decay_water = torch.exp(-self.k * depth_sub.unsqueeze(1))
        u_water = torch.sum(amps * self.omega * torch.cos(self.angles) * cos_p * decay_water, dim=1)
        v_water = torch.sum(amps * self.omega * torch.sin(self.angles) * cos_p * decay_water, dim=1)
        w_water = torch.sum(amps * self.omega * sin_p * decay_water, dim=1)
        v_water_orbital = torch.cat([u_water, v_water, w_water], dim=-1)

        return eta, deta_dx, deta_dy, deta_dt, w_air_updraft, v_water_orbital

    def raycast_forward_surface_pose(self, pos, rot, eta_nadir_true, t, wave_scale=1.0):
        """True 3D boresight ray-wave intersection from the vehicle's actual 6-DoF pose.
        Returns (eta_forward_meas, valid_mask)."""
        ray_w = torch.matmul(rot, self.ray_body_vec).squeeze(-1)
        down_slope = (-ray_w[:, 2:3]).clamp(min=0.06)
        s = torch.clamp((pos[:, 2:3] - eta_nadir_true) / down_slope, min=0.25, max=6.0)

        t_term = t.unsqueeze(1) if (torch.is_tensor(t) and t.ndim == 2) else t
        amps = self.amplitudes_base * wave_scale

        eta_hit = eta_nadir_true
        for _ in range(2):
            p_hit = pos + s * ray_w
            px_fwd = p_hit[:, 0:1].unsqueeze(1)
            py_fwd = p_hit[:, 1:2].unsqueeze(1)
            phase_angle = (self.kx * px_fwd + self.ky * py_fwd - self.omega * t_term + self.phases)
            eta_hit = torch.sum(amps * torch.cos(phase_angle), dim=1)
            s = torch.clamp((pos[:, 2:3] - eta_hit) / down_slope, min=0.25, max=6.0)

        range_noise = torch.randn_like(eta_hit) * 0.025
        valid = (torch.rand_like(eta_hit) >= 0.03) & (ray_w[:, 2:3] < -0.05) & (s < 5.8)
        eta_forward_meas = torch.where(valid, eta_hit + range_noise, torch.zeros_like(eta_hit))
        return eta_forward_meas, valid


# =============================================================================
# 2. 15-STATE AEROMARINE ERROR-STATE EKF (ES-EKF / MEKF)
# =============================================================================
class AeromarineErrorStateEKF:
    """
    15-State Error-State EKF.  Nominal: [p(3), v(3), q(4), b_a(3), b_w(3)]; error: 15D.
    """
    def __init__(self, num_envs, dt, device=DEVICE):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device

        self.p_hat = torch.zeros((num_envs, 3), device=device)
        self.v_hat = torch.zeros((num_envs, 3), device=device)
        self.q_hat = torch.zeros((num_envs, 4), device=device)
        self.q_hat[:, 0] = 1.0
        self.ba_hat = torch.zeros((num_envs, 3), device=device)
        self.bw_hat = torch.zeros((num_envs, 3), device=device)

        self.P = torch.eye(15, device=device).unsqueeze(0).repeat(num_envs, 1, 1) * 0.02
        self.P[:, 6:9, 6:9] = torch.eye(3, device=device).unsqueeze(0) * 0.002
        self.P[:, 9:12, 9:12] = torch.eye(3, device=device).unsqueeze(0) * 0.001
        self.P[:, 12:15, 12:15] = torch.eye(3, device=device).unsqueeze(0) * 0.0001

        self.q_var_accel = 0.08
        self.q_var_gyro = 0.015
        self.q_var_ba = 2.0e-5
        self.q_var_bw = 1.0e-6
        self.Q_d = torch.eye(15, device=device).unsqueeze(0).repeat(num_envs, 1, 1)
        self.Q_d[:, 0:3, 0:3] *= (self.q_var_accel * (dt**2))
        self.Q_d[:, 3:6, 3:6] *= (self.q_var_accel * dt)
        self.Q_d[:, 6:9, 6:9] *= (self.q_var_gyro * dt)
        self.Q_d[:, 9:12, 9:12] *= (self.q_var_ba * dt)
        self.Q_d[:, 12:15, 12:15] *= (self.q_var_bw * dt)

    def reset(self, num_envs, init_p, init_v, init_rot):
        self.num_envs = num_envs
        self.p_hat = init_p.clone()
        self.v_hat = init_v.clone()
        self.ba_hat = torch.zeros((num_envs, 3), device=self.device)
        self.bw_hat = torch.zeros((num_envs, 3), device=self.device)

        R = init_rot
        tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        qw = 0.5 * torch.sqrt(torch.clamp(1.0 + tr, min=1e-6))
        qx = (R[:, 2, 1] - R[:, 1, 2]) / (4.0 * qw)
        qy = (R[:, 0, 2] - R[:, 2, 0]) / (4.0 * qw)
        qz = (R[:, 1, 0] - R[:, 0, 1]) / (4.0 * qw)
        q = torch.cat([qw.unsqueeze(-1), qx.unsqueeze(-1), qy.unsqueeze(-1), qz.unsqueeze(-1)], dim=-1)
        self.q_hat = q / safe_norm(q, dim=-1, keepdim=True)

        self.P = torch.eye(15, device=self.device).unsqueeze(0).repeat(num_envs, 1, 1) * 0.02
        self.P[:, 6:9, 6:9] = torch.eye(3, device=self.device).unsqueeze(0) * 0.002
        self.P[:, 9:12, 9:12] = torch.eye(3, device=self.device).unsqueeze(0) * 0.001
        self.P[:, 12:15, 12:15] = torch.eye(3, device=self.device).unsqueeze(0) * 0.0001

    def detach_state(self):
        self.p_hat = self.p_hat.detach()
        self.v_hat = self.v_hat.detach()
        self.q_hat = self.q_hat.detach()
        self.ba_hat = self.ba_hat.detach()
        self.bw_hat = self.bw_hat.detach()
        self.P = self.P.detach()

    def propagate(self, accel_meas, gyro_meas):
        """Strapdown mechanization and covariance propagation (strict FP32)."""
        device_str = "cuda" if accel_meas.is_cuda else "cpu"
        with get_amp_autocast(device_str, enabled=False):
            N = self.num_envs
            dt = self.dt

            a_b = (accel_meas - self.ba_hat).float()
            w_b = (gyro_meas - self.bw_hat).float()

            R_hat = quat_to_rot(self.q_hat.float())
            omega_mag = safe_norm(w_b, dim=-1, keepdim=True).clamp(min=1e-6)
            delta_angle = omega_mag * dt
            axis = w_b / omega_mag
            dq_w = torch.cos(0.5 * delta_angle)
            dq_xyz = axis * torch.sin(0.5 * delta_angle)
            dq = torch.cat([dq_w, dq_xyz], dim=-1)
            self.q_hat = quat_mult(self.q_hat.float(), dq)
            self.q_hat = self.q_hat / safe_norm(self.q_hat, dim=-1, keepdim=True)

            grav_vec = torch.tensor([0.0, 0.0, -GRAVITY], device=self.device, dtype=torch.float32).view(1, 3).repeat(N, 1)
            a_world = torch.matmul(R_hat, a_b.unsqueeze(-1)).squeeze(-1) + grav_vec
            self.p_hat = self.p_hat.float() + self.v_hat.float() * dt + 0.5 * a_world * (dt**2)
            self.v_hat = self.v_hat.float() + a_world * dt

            Phi = torch.eye(15, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)
            Phi[:, 0:3, 3:6] = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1) * dt

            ax, ay, az = a_b[:, 0:1], a_b[:, 1:2], a_b[:, 2:3]
            zv = torch.zeros_like(ax)
            skew_a = torch.stack([
                torch.cat([zv, -az, ay], dim=-1),
                torch.cat([az, zv, -ax], dim=-1),
                torch.cat([-ay, ax, zv], dim=-1)
            ], dim=1)

            Phi[:, 3:6, 6:9] = -torch.matmul(R_hat, skew_a) * dt
            Phi[:, 3:6, 9:12] = -R_hat * dt

            R_dq = quat_to_rot(dq)
            Phi[:, 6:9, 6:9] = R_dq.transpose(-1, -2)
            Phi[:, 6:9, 12:15] = -torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1) * dt

            P_prop = torch.bmm(torch.bmm(Phi, self.P.float()), Phi.transpose(-1, -2)) + self.Q_d.float()
            P_sym = 0.5 * (P_prop + P_prop.transpose(-1, -2))
            self.P = torch.nan_to_num(P_sym, nan=0.05, posinf=2.0, neginf=0.0).clamp(-2.0, 2.0)
            diag_idx = torch.arange(15, device=self.device)
            self.P[:, diag_idx, diag_idx] = torch.clamp(self.P[:, diag_idx, diag_idx], min=1e-4, max=2.0)

    def update_measurement(self, y, H, R_cov, active_mask=None):
        """Joseph-stabilized Kalman update with error-state injection (strict FP32)."""
        device_str = "cuda" if y.is_cuda else "cpu"
        with get_amp_autocast(device_str, enabled=False):
            y = y.float()
            H = H.float()
            R_cov = R_cov.float()
            self.P = self.P.float()

            N = self.num_envs
            P_HT = torch.bmm(self.P, H.transpose(-1, -2)).float()
            S = (torch.bmm(H, P_HT) + R_cov).float()

            S_reg = (S + torch.eye(S.shape[-1], device=self.device, dtype=torch.float32).unsqueeze(0) * 1e-5).float()
            P_HT_T = P_HT.transpose(-1, -2).float()

            K = torch.linalg.solve(S_reg, P_HT_T).transpose(-1, -2).float()

            delta_x = torch.bmm(K, y.unsqueeze(-1)).squeeze(-1).float()
            delta_x = torch.nan_to_num(delta_x, nan=0.0, posinf=0.0, neginf=0.0)

            if active_mask is not None:
                mask_1d = active_mask.bool().view(N, 1)
                delta_x = torch.where(mask_1d, delta_x, torch.zeros_like(delta_x))

            delta_x = torch.nan_to_num(delta_x, nan=0.0, posinf=0.0, neginf=0.0)

            self.p_hat = (self.p_hat.float() + delta_x[:, 0:3]).float()
            self.v_hat = (self.v_hat.float() + delta_x[:, 3:6]).float()

            d_theta = torch.clamp(delta_x[:, 6:9], -0.05, 0.05)
            dq_corr = torch.cat([torch.ones((N, 1), device=self.device, dtype=torch.float32), 0.5 * d_theta], dim=-1)
            self.q_hat = quat_mult(self.q_hat.float(), dq_corr)
            self.q_hat = (self.q_hat / safe_norm(self.q_hat, dim=-1, keepdim=True)).float()

            self.ba_hat = torch.clamp(self.ba_hat.float() + delta_x[:, 9:12], -0.08, 0.08).float()
            self.bw_hat = torch.clamp(self.bw_hat.float() + delta_x[:, 12:15], -0.03, 0.03).float()

            I_KH = (torch.eye(15, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1) - torch.bmm(K, H)).float()
            P_new = torch.bmm(torch.bmm(I_KH, self.P), I_KH.transpose(-1, -2)).float() + torch.bmm(torch.bmm(K, R_cov), K.transpose(-1, -2)).float()
            P_sym = 0.5 * (P_new + P_new.transpose(-1, -2))

            if active_mask is not None:
                mask_3d = active_mask.bool().view(N, 1, 1)
                self.P = torch.where(mask_3d, P_sym, self.P).clamp(-2.0, 2.0)
            else:
                self.P = P_sym.clamp(-2.0, 2.0)
            self.P = torch.nan_to_num(self.P, nan=0.05, posinf=2.0, neginf=0.0)
            diag_idx = torch.arange(15, device=self.device)
            self.P[:, diag_idx, diag_idx] = torch.clamp(self.P[:, diag_idx, diag_idx], min=1e-4, max=2.0)

    def update_magnetometer(self, mag_meas_body):
        """Corrects yaw drift using the geomagnetic reference vector."""
        N = self.num_envs
        R_hat = quat_to_rot(self.q_hat)
        m_ref_w = MAG_FIELD_WORLD.to(self.device).repeat(N, 1)
        m_pred_b = torch.bmm(R_hat.transpose(-1, -2), m_ref_w.unsqueeze(-1)).squeeze(-1)

        y_mag = mag_meas_body - m_pred_b
        mx, my, mz = m_pred_b[:, 0:1], m_pred_b[:, 1:2], m_pred_b[:, 2:3]
        zv = torch.zeros_like(mx)
        skew_m = torch.stack([
            torch.cat([zv, -mz, my], dim=-1),
            torch.cat([mz, zv, -mx], dim=-1),
            torch.cat([-my, mx, zv], dim=-1)
        ], dim=1)

        H = torch.zeros((N, 3, 15), device=self.device, dtype=torch.float32)
        H[:, :, 6:9] = skew_m
        R_cov = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1) * 0.04
        self.update_measurement(y_mag, H, R_cov)

    def update_dvl(self, dvl_meas_body, is_foil_submerged, dvl_lost_track):
        """DVL water-track velocity with attitude Jacobian [v_b]x."""
        N = self.num_envs
        R_hat = quat_to_rot(self.q_hat)
        v_pred_b = torch.bmm(R_hat.transpose(-1, -2), self.v_hat.unsqueeze(-1)).squeeze(-1)

        y_dvl = dvl_meas_body - v_pred_b
        H = torch.zeros((N, 3, 15), device=self.device, dtype=torch.float32)
        H[:, :, 3:6] = R_hat.transpose(-1, -2).float()
        vx_b, vy_b, vz_b = v_pred_b[:, 0:1], v_pred_b[:, 1:2], v_pred_b[:, 2:3]
        zv_ = torch.zeros_like(vx_b)
        H[:, :, 6:9] = torch.stack([
            torch.cat([zv_, -vz_b, vy_b], dim=-1),
            torch.cat([vz_b, zv_, -vx_b], dim=-1),
            torch.cat([-vy_b, vx_b, zv_], dim=-1)
        ], dim=1)

        R_cov = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1) * 0.10
        active_mask = (is_foil_submerged.float() * (1.0 - dvl_lost_track.float())).squeeze(-1)
        self.update_measurement(y_dvl, H, R_cov, active_mask=active_mask)

    def update_optical_flow(self, flow_rates_body, clearance_est, celerity_water, is_airborne):
        """Optical flow translational update with attitude Jacobian."""
        N = self.num_envs
        R_hat = quat_to_rot(self.q_hat)
        R_T = R_hat.transpose(-1, -2)
        h_clamped = torch.clamp(clearance_est, min=0.08, max=3.0)

        u_body = torch.bmm(R_T, self.v_hat.unsqueeze(-1)).squeeze(-1)

        flow_pred_x = -(u_body[:, 1:2]) / h_clamped
        flow_pred_y = +(u_body[:, 0:1] - celerity_water) / h_clamped
        flow_pred = torch.cat([flow_pred_x, flow_pred_y], dim=-1)

        y_flow = (flow_rates_body - flow_pred).float()
        H = torch.zeros((N, 2, 15), device=self.device, dtype=torch.float32)
        inv_h = (1.0 / h_clamped).view(N, 1)

        H[:, 0, 3:6] = -inv_h * R_T[:, 1, :]
        H[:, 1, 3:6] = +inv_h * R_T[:, 0, :]
        ux_b, uy_b, uz_b = u_body[:, 0:1], u_body[:, 1:2], u_body[:, 2:3]
        zu_ = torch.zeros_like(ux_b)
        row0 = torch.cat([zu_, -uz_b, uy_b], dim=-1)
        row1 = torch.cat([uz_b, zu_, -ux_b], dim=-1)
        H[:, 0, 6:9] = -inv_h * row1
        H[:, 1, 6:9] = +inv_h * row0

        R_cov = torch.eye(2, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1) * 0.08
        active_mask = (is_airborne.float() * (clearance_est < 1.8).float()).squeeze(-1)
        self.update_measurement(y_flow, H, R_cov, active_mask=active_mask)

    def update_gnss(self, gnss_pos_w, gnss_vel_w, clearance_est):
        """GNSS with spray-attenuation covariance inflation."""
        N = self.num_envs
        y_pos = gnss_pos_w - self.p_hat
        y_vel = gnss_vel_w - self.v_hat

        H_pos = torch.zeros((N, 3, 15), device=self.device, dtype=torch.float32)
        H_pos[:, 0:3, 0:3] = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)

        H_vel = torch.zeros((N, 3, 15), device=self.device, dtype=torch.float32)
        H_vel[:, 0:3, 3:6] = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)

        clr_safe = torch.clamp(clearance_est, min=0.01, max=10.0)
        spray_inflation = 1.0 + 8.0 * torch.exp(-clr_safe / 0.25).unsqueeze(-1)
        R_pos = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1) * (0.0025 * spray_inflation)
        R_vel = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1) * (0.0009 * spray_inflation)

        self.update_measurement(y_pos, H_pos, R_pos)
        self.update_measurement(y_vel, H_vel, R_vel)

    def update_altimeter_fusion(self, *args, **kwargs):
        raise NotImplementedError("Altimeter-to-MSL fusion was removed; see FIX-05.")


# =============================================================================
# 3. SPRAY-ADAPTIVE RADAR & HYDROSTATIC PERCEPTION SUITE
# =============================================================================
class HardwarePerceptionSuite:
    def __init__(self, dt, device):
        self.dt = dt
        self.device = device
        self.cant_angle = math.radians(22.0)
        self.radar_num_bins = 128
        self.radar_r_max = 4.50
        self.radar_dr = self.radar_r_max / (self.radar_num_bins - 1)
        self.airframe_exclusion_m = 0.135
        self.cfar_threshold_factor = 2.15

        self.bins = torch.linspace(0.0, self.radar_r_max, self.radar_num_bins, device=device).view(1, -1)
        sponson_dist = 0.092
        mast_dist = 0.118
        self.p_airframe = 32.0 * torch.exp(-((self.bins - sponson_dist)**2) / (2 * (0.018**2))) + \
                          24.0 * torch.exp(-((self.bins - mast_dist)**2) / (2 * (0.015**2)))
        self.spatial_mask = (self.bins > self.airframe_exclusion_m).float()

    def process_fmcw_radar_dsp(self, true_clearance, deta_dx, deta_dy, fwd_speed, vz_speed):
        """Returns (detected_range, spray_dropout, is_target_locked, multipath_rejected)
        without adding back the true wave-slope crest_bias."""
        N = true_clearance.shape[0]
        r_footprint = true_clearance * 0.140
        wave_slope_mag = torch.sqrt(deta_dx**2 + deta_dy**2 + 1e-6)
        crest_bias = wave_slope_mag * (r_footprint * 0.40)
        true_range_water = (true_clearance - crest_bias).clamp(min=0.03, max=self.radar_r_max - 0.05)

        spray_dropout = (torch.rand_like(true_clearance) < 0.025).float()
        r_water_batch = true_range_water.view(N, 1)
        p_water_amp = (42.0 * (1.0 - spray_dropout)).clamp(min=2.0)
        p_water = p_water_amp * torch.exp(-((self.bins - r_water_batch)**2) / (2 * (0.028**2)))

        p_noise = torch.randn((N, self.radar_num_bins), device=self.device) * 1.5 + 8.0
        raw_range_fft = self.p_airframe + p_water + p_noise
        gated_range_fft = raw_range_fft * self.spatial_mask

        radial_closure = fwd_speed * math.sin(self.cant_angle * 0.1) - vz_speed
        doppler_attenuation = torch.where((torch.abs(radial_closure) < 0.02) & (fwd_speed < 0.3), 0.40, 1.0)

        cleaned_spectrum = F.relu(gated_range_fft * doppler_attenuation - self.p_airframe)

        peak_idx = torch.argmax(cleaned_spectrum, dim=-1, keepdim=True)
        idx_l = (peak_idx - 1).clamp(min=0)
        idx_r = (peak_idx + 1).clamp(max=self.radar_num_bins - 1)
        y_l = torch.gather(cleaned_spectrum, -1, idx_l)
        y_c = torch.gather(cleaned_spectrum, -1, peak_idx)
        y_r = torch.gather(cleaned_spectrum, -1, idx_r)
        denom = y_l - 2.0 * y_c + y_r
        offset = torch.where(denom.abs() > 1e-6, 0.5 * (y_l - y_r) / denom, torch.zeros_like(denom)).clamp(-0.5, 0.5)
        detected_clearance = (peak_idx.float() + offset) * self.radar_dr

        peak_power = y_c
        cfar_noise_floor = torch.mean(p_noise, dim=-1, keepdim=True)
        first_valid_bin = int(math.floor(self.airframe_exclusion_m / self.radar_dr)) + 1
        not_edge_clipped = (peak_idx > first_valid_bin)
        is_target_locked = (
            (peak_power > (cfar_noise_floor * self.cfar_threshold_factor))
            & (spray_dropout < 0.5)
            & not_edge_clipped
        )

        multipath_rejected = (peak_idx != torch.argmax(raw_range_fft, dim=-1, keepdim=True)).float()

        return detected_clearance, spray_dropout, is_target_locked, multipath_rejected

    def sample_piezoresistive_tap(self, d_sub_physical, v_foil_rel_body_true, v_body_est, rho_water_true):
        """Physical transducer + onboard DSP decoupling using EKF velocity estimates."""
        device_str = "cuda" if self.device.type == "cuda" else "cpu"
        with get_amp_autocast(device_str, enabled=False):
            d_sub_physical = d_sub_physical.float()
            v_foil_rel_body_true = v_foil_rel_body_true.float()
            v_body_est = v_body_est.float()

            is_tap_submerged_true = (d_sub_physical > 0.005)
            rho_fluid_true = torch.where(is_tap_submerged_true, rho_water_true, torch.full_like(rho_water_true, RHO_AIR))

            u_true = v_foil_rel_body_true[:, 0:1]
            v_true = v_foil_rel_body_true[:, 1:2]
            w_true = v_foil_rel_body_true[:, 2:3]
            q_true = (0.5 * rho_fluid_true * (u_true**2 + v_true**2 + w_true**2)).clamp(max=15000.0)

            alpha_strut_true = torch.atan2(w_true, u_true.clamp(min=0.2))
            beta_strut_true = torch.atan2(v_true, u_true.clamp(min=0.2))
            cp_actual = torch.clamp(0.045 + 0.420 * torch.abs(alpha_strut_true) + 0.680 * torch.abs(beta_strut_true), min=0.0, max=0.80)

            p_hydrostatic_true = rho_water_true * GRAVITY * d_sub_physical
            p_dynamic_error_true = cp_actual * q_true
            noise_p = torch.randn_like(d_sub_physical) * 5.0
            p_measured_raw = float(P_ATM) + p_hydrostatic_true + p_dynamic_error_true + noise_p

            u_est = v_body_est[:, 0:1]
            v_est = v_body_est[:, 1:2]
            w_est = v_body_est[:, 2:3]
            q_est = (0.5 * RHO_WATER_NOM * (u_est**2 + v_est**2 + w_est**2)).clamp(max=15000.0)

            alpha_strut_est = torch.atan2(w_est, u_est.clamp(min=0.2))
            beta_strut_est = torch.atan2(v_est, u_est.clamp(min=0.2))

            cp_calibration = torch.clamp((0.045 + 0.420 * torch.abs(alpha_strut_est) + 0.680 * torch.abs(beta_strut_est)) * 1.05, min=0.0, max=0.85)
            p_static_decoupled = torch.minimum(p_measured_raw, p_measured_raw - (cp_calibration * q_est))

            d_sub_calc = F.relu((p_static_decoupled - float(P_ATM)) / (RHO_WATER_NOM * GRAVITY)).clamp(max=0.35)
            is_tap_submerged_meas = d_sub_calc > 0.005
            d_sub_hydrostatic = torch.where(is_tap_submerged_meas, d_sub_calc, torch.zeros_like(d_sub_calc))
            bernoulli_bias_cm = (d_sub_physical - d_sub_hydrostatic) * 100.0

            return d_sub_hydrostatic, p_static_decoupled, p_dynamic_error_true, bernoulli_bias_cm


# =============================================================================
# 4. ONLINE LINEAR DISPERSION OBSERVER
# =============================================================================
class RealSeaWaveDispersionObserver:
    def __init__(self, dt, device):
        self.dt = dt
        self.device = device
        self.k_dom = torch.full((1, 1), 0.45, device=device)
        self.omega_dom = torch.sqrt(GRAVITY * self.k_dom)
        self.q_quad = torch.zeros((1, 1), device=device)
        self.last_eta_nadir = torch.zeros((1, 1), device=device)
        self.deta_dx_nadir = torch.zeros((1, 1), device=device)
        self.w_up_nadir = torch.zeros((1, 1), device=device)
        self.ms_eta = torch.full((1, 1), 0.01, device=device)
        self.ms_slope = torch.full((1, 1), 0.01 * (0.45**2), device=device)

    def reset(self, num_envs):
        self.k_dom = torch.full((num_envs, 1), 0.45, device=self.device)
        self.omega_dom = torch.sqrt(GRAVITY * self.k_dom)
        self.q_quad = torch.zeros((num_envs, 1), device=self.device)
        self.last_eta_nadir = torch.zeros((num_envs, 1), device=self.device)
        self.deta_dx_nadir = torch.zeros((num_envs, 1), device=self.device)
        self.w_up_nadir = torch.zeros((num_envs, 1), device=self.device)
        self.ms_eta = torch.full((num_envs, 1), 0.01, device=self.device)
        self.ms_slope = torch.full((num_envs, 1), 0.01 * (0.45**2), device=self.device)

    def update_and_predict(self, clearance_est, craft_z, eta_forward_meas, valid, lookahead_dist, fwd_speed, preview_time=1.25):
        clearance_est = clearance_est.detach()
        craft_z = craft_z.detach()
        eta_forward_meas = eta_forward_meas.detach()

        eta_nadir_est = (craft_z - clearance_est).clamp(-0.6, 0.6)
        self.last_eta_nadir = eta_nadir_est
        delta_x = torch.clamp(lookahead_dist.detach(), min=0.25)
        spatial_slope = ((eta_forward_meas - eta_nadir_est) / delta_x).clamp(-0.6, 0.6)

        dt_obs = max(self.dt, REF_DT)
        a_ms = math.exp(-dt_obs / 0.45)
        ms_eta_new = a_ms * self.ms_eta + (1.0 - a_ms) * (eta_nadir_est**2)
        ms_slope_new = a_ms * self.ms_slope + (1.0 - a_ms) * (spatial_slope**2)
        self.ms_eta = torch.where(valid, ms_eta_new, self.ms_eta)
        self.ms_slope = torch.where(valid, ms_slope_new, self.ms_slope)

        k_est = torch.sqrt(self.ms_slope / (self.ms_eta + 0.002))
        a_k = math.exp(-dt_obs / 0.25)
        k_new = torch.clamp(a_k * self.k_dom + (1.0 - a_k) * k_est, min=0.08, max=3.5)
        self.k_dom = torch.where(valid, k_new, self.k_dom)
        self.omega_dom = torch.sqrt(GRAVITY * self.k_dom)

        q_meas = (-spatial_slope / self.k_dom).clamp(-0.4, 0.4)
        a_q = math.exp(-dt_obs / 0.15)
        q_new = a_q * self.q_quad + (1.0 - a_q) * q_meas
        self.q_quad = torch.where(valid, q_new, self.q_quad)

        self.deta_dx_nadir = -self.k_dom * self.q_quad
        self.w_up_nadir = self.omega_dom * self.q_quad

        delta = self.k_dom * (fwd_speed.detach() * preview_time) - self.omega_dom * preview_time
        c_d, s_d = torch.cos(delta), torch.sin(delta)
        eta_preview = eta_nadir_est * c_d - self.q_quad * s_d
        quad_ahead = self.q_quad * c_d + eta_nadir_est * s_d
        deta_dx_preview = -self.k_dom * quad_ahead
        w_up_preview = self.omega_dom * quad_ahead

        return eta_preview, deta_dx_preview, w_up_preview


# =============================================================================
# 5. MULTIMODAL AEROHYDRODYNAMICS WITH ACTIVE T-FOIL, PAR CUSHION & SCULLING
# =============================================================================
class MultimodalAeroHydroDynamics(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, pos, vel, rot, omega, eta, deta_dt, w_air_updraft, v_water_orbital, ambient_gust,
                is_vent_L, is_vent_R, foil_decalage, rho_water, hydro_surfaces, motor_thrusts_total,
                morphology_dict, sim_time, is_soaring, scull_throttle=None):
        N = pos.shape[0]

        w_span = morphology_dict["wing_span"]
        w_chord = morphology_dict["wing_chord"]
        w_area = morphology_dict["wing_area"]
        f_span = morphology_dict["foil_span"]
        f_area = morphology_dict["foil_area"]
        strut_len = morphology_dict["strut_length"]
        caudal_area = morphology_dict["caudal_area"]
        anhedral_rad = morphology_dict["anhedral_rad"]

        # 1. WIG wing & PAR cushion
        h_wing = (pos[:, 2:3] - eta).clamp(min=0.03)
        h_over_chord = h_wing / w_chord
        h_over_span = h_wing / w_span

        z0 = 0.0002
        wind_shear_factor = torch.clamp(
            torch.log((h_wing + z0) / z0) / math.log(10.0 / z0),
            min=0.18, max=1.25
        )
        ambient_wind_local = ambient_gust * wind_shear_factor

        anhedral_plenum_boost = 1.0 + 1.2 * torch.clamp(anhedral_rad, min=0.0)
        phi_lift_wig = (1.0 + 0.35 / (0.10 + 2.2 * torch.pow(h_over_chord, 1.15))) * anhedral_plenum_boost
        sigma_drag_wig = (1.0 - torch.exp(-2.48 * torch.pow(2.0 * h_over_span, 0.78))).clamp(min=0.08, max=1.0)

        downwash_vel = torch.sqrt(torch.clamp(motor_thrusts_total / (2.0 * RHO_AIR * TOTAL_ROTOR_AREA), min=0.0))
        par_pressure = (PAR_PLENUM_EFF * 0.5 * RHO_AIR * (downwash_vel**2) * F.relu(1.0 - h_wing / 0.25)).clamp(max=120.0)
        f_par_z = par_pressure * w_area

        updraft_field = torch.cat([torch.zeros((N, 2), device=self.device), w_air_updraft], dim=-1)
        v_air_effective = vel - ambient_wind_local - updraft_field
        v_air_mag = safe_norm(v_air_effective, dim=-1, keepdim=True).clamp(min=1.0)
        q_air = 0.5 * RHO_AIR * (v_air_mag**2)

        rot_T = rot.transpose(-1, -2)
        v_air_body = torch.matmul(rot_T, v_air_effective.unsqueeze(-1)).squeeze(-1)
        alpha_wing = torch.atan2(-v_air_body[:, 2:3], v_air_body[:, 0:1].clamp(min=0.5)) + 0.03

        cl_wing = torch.clamp((WING_CL0 + 4.5 * alpha_wing) * phi_lift_wig, -0.3, 1.4)
        cd_induced = (cl_wing**2) / (math.pi * morphology_dict["aspect_ratio"] * WING_OSWALD) * sigma_drag_wig
        cd_wing = WING_CD0 + cd_induced

        f_wing_z = q_air * w_area * cl_wing * torch.cos(alpha_wing) + f_par_z
        f_wing_x = -q_air * w_area * cd_wing * torch.sign(v_air_effective[:, 0:1] + 1e-4)
        f_wing_world = torch.cat([f_wing_x, torch.zeros((N, 1), device=self.device), f_wing_z], dim=-1)

        q_rate = omega[:, 1:2]
        tau_aero_pitch = q_air * w_area * w_chord * (+0.85 * alpha_wing - 2.8 * (q_rate * w_chord / (2.0 * v_air_mag)))

        # 2. Ventral hydrofoil with elevons, strut rudder, caudal sculling
        v_foil_rel = vel - v_water_orbital
        v_foil_rel_body = torch.matmul(rot_T, v_foil_rel.unsqueeze(-1)).squeeze(-1)

        v_foil_u_body = v_foil_rel_body[:, 0:1]
        v_strut_y_body = v_foil_rel_body[:, 1:2]
        v_foil_w_body = v_foil_rel_body[:, 2:3]

        v_foil_speed = safe_norm(v_foil_rel_body[:, :2], dim=-1, keepdim=True).clamp(min=0.1, max=20.0)
        q_water = (0.5 * rho_water * (v_foil_speed**2)).clamp(max=30000.0)

        y_station_offset = f_span * 0.25
        z_tip_L = pos[:, 2:3] + rot[:, 2, 1:2] * y_station_offset - rot[:, 2, 2:3] * strut_len
        z_tip_R = pos[:, 2:3] - rot[:, 2, 1:2] * y_station_offset - rot[:, 2, 2:3] * strut_len

        d_sub_L = (eta - z_tip_L).clamp(min=0.0, max=0.22)
        d_sub_R = (eta - z_tip_R).clamp(min=0.0, max=0.22)
        immersion_depth_mean = 0.5 * (d_sub_L + d_sub_R)

        delta_e_L = hydro_surfaces[:, 0:1]
        delta_e_R = hydro_surfaces[:, 1:2]
        delta_rud = hydro_surfaces[:, 2:3]

        scull_active = is_soaring.float() * (immersion_depth_mean > 0.020).float()
        if scull_throttle is None:
            scull_throttle_eff = scull_active
        else:
            scull_throttle_eff = scull_active * scull_throttle.clamp(0.0, 1.0)
        scull_freq = SCULL_FREQ_NOM
        scull_amp = SCULL_AMP_NOM
        delta_rud_eff = delta_rud

        linear_tail_amp = SCULL_LEVER_ARM * math.sin(scull_amp)
        strouhal_st = (2.0 * scull_freq * linear_tail_amp / v_foil_speed).clamp(max=2.0)
        strouhal_prop_eff = torch.exp(-((strouhal_st - 0.30)**2) / (2.0 * (0.08**2))) * 0.84

        v_scull_tip = 2.0 * math.pi * scull_freq * scull_amp * SCULL_LEVER_ARM
        v_scull_effective = torch.sqrt(v_foil_speed**2 + (v_scull_tip**2) * scull_active)
        q_scull = 0.5 * rho_water * (v_scull_effective**2)
        f_scull_avail = torch.clamp(
            q_scull * caudal_area * HYDRO_RUDDER_CY_DELTA * math.sin(scull_amp) * strouhal_prop_eff * scull_active,
            min=0.0, max=22.0
        )
        f_scull_thrust = f_scull_avail * scull_throttle_eff
        p_scull_actuator_watts = torch.clamp(f_scull_thrust * SCULL_SERVO_POWER_COEFF * (scull_freq / SCULL_FREQ_NOM), min=0.0, max=45.0)

        is_vent_prior = torch.maximum(is_vent_L, is_vent_R)
        delta_foil_vent_relief = torch.where(is_vent_prior.bool(), -math.radians(2.0), 0.0)

        alpha_flow = torch.atan2(-v_foil_w_body, v_foil_u_body.clamp(min=0.5))
        alpha_base = alpha_flow + foil_decalage + delta_foil_vent_relief

        alpha_foil_L = alpha_base + (HYDRO_CL_DELTA / FOIL_CL_ALPHA) * delta_e_L
        alpha_foil_R = alpha_base + (HYDRO_CL_DELTA / FOIL_CL_ALPHA) * delta_e_R

        cp_suction = torch.clamp(2.4 * F.relu(alpha_base), min=0.0, max=3.5)
        suction_dep_L = cp_suction * q_water
        suction_dep_R = cp_suction * q_water

        active_seal_boost = PAR_SEAL_GAIN * par_pressure
        seal_strength_L = rho_water * GRAVITY * d_sub_L + 120.0 + active_seal_boost
        seal_strength_R = rho_water * GRAVITY * d_sub_R + 120.0 + active_seal_boost

        vent_trigger_L = (suction_dep_L > 2.2 * seal_strength_L) & (d_sub_L < 0.030) & (alpha_foil_L > math.radians(2.2))
        reprime_trigger_L = (d_sub_L >= 0.040) & (alpha_foil_L < math.radians(4.0))
        is_airborne_L = (d_sub_L < 0.005)

        is_vent_L_next = torch.where(
            is_airborne_L, torch.zeros_like(is_vent_L),
            torch.where(is_vent_L.bool(),
                torch.where(reprime_trigger_L, torch.zeros_like(is_vent_L), torch.ones_like(is_vent_L)),
                torch.where(vent_trigger_L, torch.ones_like(is_vent_L), torch.zeros_like(is_vent_L))
            )
        )
        vent_factor_L = torch.where(is_vent_L_next.bool(), 0.40, 1.0)

        vent_trigger_R = (suction_dep_R > 2.2 * seal_strength_R) & (d_sub_R < 0.030) & (alpha_foil_R > math.radians(2.2))
        reprime_trigger_R = (d_sub_R >= 0.040) & (alpha_foil_R < math.radians(4.0))
        is_airborne_R = (d_sub_R < 0.005)

        is_vent_R_next = torch.where(
            is_airborne_R, torch.zeros_like(is_vent_R),
            torch.where(is_vent_R.bool(),
                torch.where(reprime_trigger_R, torch.zeros_like(is_vent_R), torch.ones_like(is_vent_R)),
                torch.where(vent_trigger_R, torch.ones_like(is_vent_R), torch.zeros_like(is_vent_R))
            )
        )
        vent_factor_R = torch.where(is_vent_R_next.bool(), 0.40, 1.0)

        cl_foil_L = torch.clamp(FOIL_CL_ALPHA * alpha_foil_L * vent_factor_L, -0.10, 1.30)
        cl_foil_R = torch.clamp(FOIL_CL_ALPHA * alpha_foil_R * vent_factor_R, -0.10, 1.30)

        cd_foil = torch.clamp(
            FOIL_CD0 + 0.05 * (cl_foil_L**2 + cl_foil_R**2) * 0.5 + HYDRO_CD_DELTA * (delta_e_L**2 + delta_e_R**2) * 0.5,
            0.010, 0.45
        )

        sub_frac_L = (d_sub_L / 0.040).clamp(min=0.0, max=1.0)
        sub_frac_R = (d_sub_R / 0.040).clamp(min=0.0, max=1.0)
        fwd_mask = (v_foil_u_body > 0.1).float()

        area_half = f_area * 0.5
        f_lift_L = torch.clamp(q_water * (area_half * sub_frac_L) * cl_foil_L * fwd_mask, min=-0.5, max=FOIL_LIFT_CLAMP_PER_SIDE)
        f_lift_R = torch.clamp(q_water * (area_half * sub_frac_R) * cl_foil_R * fwd_mask, min=-0.5, max=FOIL_LIFT_CLAMP_PER_SIDE)
        f_hydro_lift = f_lift_L + f_lift_R

        flow_angle_w = torch.clamp(-v_foil_w_body / v_foil_speed, -0.5, 0.5)
        f_soar_surge = torch.clamp(SOARING_KATZMAYR_COEFF * f_hydro_lift * torch.sin(flow_angle_w), min=-8.0, max=18.0)

        f_hydro_drag_mag = torch.clamp(
            q_water * (f_area * 0.5 * (sub_frac_L + sub_frac_R)) * cd_foil - f_soar_surge - f_scull_thrust,
            min=-25.0, max=25.0
        )
        f_hydro_drag_x = -f_hydro_drag_mag * torch.sign(v_foil_u_body + 1e-4)

        tau_hydro_surfaces_roll = (f_lift_L - f_lift_R) * y_station_offset

        f_rudder_side = q_water * caudal_area * (immersion_depth_mean / 0.08).clamp(max=1.0) * (
            -2.8 * (v_strut_y_body / v_foil_speed) + HYDRO_RUDDER_CY_DELTA * delta_rud_eff
        )
        f_strut_lateral_body_y = torch.clamp(f_rudder_side, -8.0, 8.0)

        arm_strut_z = strut_len - 0.5 * immersion_depth_mean
        tau_strut_roll = f_strut_lateral_body_y * arm_strut_z + tau_hydro_surfaces_roll
        strut_root_moment = torch.abs(f_strut_lateral_body_y * strut_len) + torch.abs(tau_hydro_surfaces_roll)

        immersion_rate = F.relu(deta_dt - vel[:, 2:3]) * (immersion_depth_mean > 0.001).float()
        f_slam_z = torch.clamp(0.5 * rho_water * HYDRO_SLAM_COEFF * f_span * FOIL_CHORD_NOM * (immersion_rate**2), max=25.0)

        # 3. Hull planing & buoyancy
        z_hull_bottom = pos[:, 2:3] - HULL_DRAFT_MAX
        hull_immersion = (eta - z_hull_bottom).clamp(min=0.0, max=HULL_DRAFT_MAX * 2.0)
        f_buoy_z = rho_water * GRAVITY * (hull_immersion / HULL_DRAFT_MAX).clamp(max=1.5) * HULL_VOLUME_TOTAL

        trim_angle = F.relu(alpha_wing + 0.02)
        f_planing_z = (0.5 * rho_water * (HULL_BEAM * HULL_LENGTH) * (v_foil_speed**2) * 0.14 * torch.sin(trim_angle) * (hull_immersion > 0.005).float() * fwd_mask).clamp(max=40.0)
        f_entry_drag_z = (-0.5 * rho_water * (HULL_BEAM * HULL_LENGTH) * 1.8 * vel[:, 2:3] * torch.abs(vel[:, 2:3]) * (hull_immersion > 0.002).float()).clamp(min=-40.0, max=40.0)

        fn = v_foil_speed / math.sqrt(GRAVITY * HULL_LENGTH)
        froude_hump = torch.exp(-((fn - 0.45)**2) / 0.04) * 1.8
        f_hull_drag_mag = (0.5 * rho_water * (HULL_BEAM * hull_immersion) * (v_foil_speed**2) * (0.04 + froude_hump)).clamp(max=40.0)
        f_hull_drag_x = -f_hull_drag_mag * torch.sign(v_foil_u_body + 1e-4)

        f_water_body = torch.cat([f_hydro_drag_x + f_hull_drag_x, f_strut_lateral_body_y, f_hydro_lift + f_slam_z + f_buoy_z + f_planing_z + f_entry_drag_z], dim=-1)
        f_hydro_world = torch.matmul(rot, f_water_body.unsqueeze(-1)).squeeze(-1)

        tau_bow_planing = (f_planing_z + f_buoy_z) * BOW_CENTER_OFFSET_X
        tau_hydro_pitch = (-f_hydro_drag_x * strut_len) + (-f_hull_drag_x * HULL_DRAFT_MAX) + tau_aero_pitch - tau_bow_planing
        tau_hydro_roll = tau_strut_roll
        tau_hydro_yaw = -f_strut_lateral_body_y * 0.08

        is_ventilated_overall = torch.maximum(is_vent_L_next, is_vent_R_next)

        sub_mean = 0.5 * (sub_frac_L + sub_frac_R)
        mass_est = morphology_dict["total_mass"].detach()
        damp_x = -((RHO_AIR * w_area * cd_wing * v_air_mag + rho_water * f_area * cd_foil * sub_mean * v_foil_speed) / mass_est).clamp(min=-8.0, max=0.0)
        damp_y = -((rho_water * caudal_area * 2.8 * (immersion_depth_mean / 0.08).clamp(max=1.0) * v_foil_speed + 0.3 * RHO_AIR * w_area * v_air_mag) / mass_est).clamp(min=-8.0, max=0.0)
        damp_z = -((0.5 * rho_water * f_area * sub_mean * FOIL_CL_ALPHA * v_foil_speed * fwd_mask + 0.5 * RHO_AIR * w_area * 4.5 * v_air_mag) / mass_est).clamp(min=-12.0, max=0.0)
        plant_damp_diag = torch.cat([damp_x, damp_y, damp_z], dim=-1).detach()

        aoa_tensors = {
            "alpha_foil": alpha_base,
            "strut_moment_nm": strut_root_moment,
            "q_water": q_water,
            "f_planing_z": f_planing_z,
            "tau_hydro_roll": tau_hydro_roll,
            "tau_hydro_yaw": tau_hydro_yaw,
            "f_par_z": f_par_z,
            "par_pressure": par_pressure,
            "f_soar_surge": f_soar_surge,
            "f_scull_thrust": f_scull_thrust,
            "p_scull_actuator_watts": p_scull_actuator_watts,
            "strouhal_st": strouhal_st,
            "plant_damp_diag": plant_damp_diag
        }

        return (f_wing_world, f_hydro_world, tau_hydro_pitch, tau_hydro_roll, tau_hydro_yaw, h_wing,
                immersion_depth_mean, f_slam_z, f_buoy_z, is_vent_L_next, is_vent_R_next,
                is_ventilated_overall, wind_shear_factor, strut_root_moment, aoa_tensors, alpha_base)


# =============================================================================
# 6. ROTOR AEROMECHANICS & PITT-PETERS DYNAMIC INFLOW
# =============================================================================
class RotorAeromechanics(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.R = PROP_RADIUS
        self.M_inflow = torch.tensor([8.0 * self.R / (3.0 * math.pi), 16.0 * self.R / (45.0 * math.pi), 16.0 * self.R / (45.0 * math.pi)], device=device).view(1, 1, 3)
        self.L_diag = torch.tensor([2.0, 4.0, 4.0], device=device).view(1, 1, 3)

    def step_inflow_and_flapping(self, nu_states, motors, v_air_hub_body, omega, dt):
        omega_props = torch.sqrt((F.relu(motors) + 1e-6) / 1.1e-5).clamp(min=50.0, max=1800.0)

        u_h = v_air_hub_body[:, :, 0:1]
        v_h = v_air_hub_body[:, :, 1:2]
        w_h = v_air_hub_body[:, :, 2:3]

        v_eff = torch.sqrt(u_h**2 + v_h**2 + (w_h + nu_states[:, :, 0:1])**2 + 1e-4)
        c_T = (motors / (RHO_AIR * ROTOR_DISK_AREA)).unsqueeze(-1)
        tip_speed = (omega_props * self.R).unsqueeze(-1).clamp(min=5.0)

        mu_x = (u_h / tip_speed).clamp(min=-0.4, max=0.4)
        mu_y = (v_h / tip_speed).clamp(min=-0.4, max=0.4)

        c_L = -0.5 * mu_y * c_T
        c_M = 0.5 * mu_x * c_T
        c_aero = torch.cat([c_T, c_L, c_M], dim=-1)

        L_inv = v_eff * self.L_diag
        nu_ss = c_aero / L_inv.clamp(min=1e-3)
        lambda_decay = (L_inv / self.M_inflow).clamp(max=2500.0)
        decay_factor = torch.exp(-lambda_decay * dt)

        nu_states_next = nu_states * decay_factor + nu_ss * (1.0 - decay_factor)
        nu_0 = nu_states_next[:, :, 0]

        gamma_8 = LOCK_NUMBER / 8.0
        nu_beta_sq_minus_1 = BLADE_FLAP_FREQ**2 - 1.0
        p_rate = omega[:, 0:1].unsqueeze(1)
        q_rate = omega[:, 1:2].unsqueeze(1)

        f_a = gamma_8 * (
            mu_x.squeeze(-1) * ((8.0 / 3.0) * BLADE_PITCH_ZERO - 2.0 * (nu_0 / tip_speed.squeeze(-1)))
            - (16.0 / (LOCK_NUMBER * omega_props)) * q_rate.squeeze(-1)
        )
        f_b = gamma_8 * (
            mu_y.squeeze(-1) * ((8.0 / 3.0) * BLADE_PITCH_ZERO - 2.0 * (nu_0 / tip_speed.squeeze(-1)))
            - (16.0 / (LOCK_NUMBER * omega_props)) * p_rate.squeeze(-1)
        )

        delta_det = nu_beta_sq_minus_1**2 + gamma_8**2
        a_1 = ((nu_beta_sq_minus_1 * f_a - gamma_8 * f_b) / delta_det).clamp(-0.20, 0.20)
        b_1 = ((gamma_8 * f_a + nu_beta_sq_minus_1 * f_b) / delta_det).clamp(-0.20, 0.20)

        return nu_states_next, a_1, b_1, omega_props


# =============================================================================
# 7. BATTERY ELECTRO-THERMAL POWERTRAIN (ROTORS + PUSHER + SCULLING)
# =============================================================================
class ElectroThermalBatteryPowertrain(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.tau1 = BATTERY_R1 * BATTERY_C1
        self.tau2 = BATTERY_R2 * BATTERY_C2

    def step(self, motor_thrusts, ambient_inflow_wz, nu_0, omega_props, u1, u2, soc, t_cell, dt, p_scull_load,
             pusher_thrust=None, v_air_x=None):
        thrust_clamped = F.relu(motor_thrusts)
        p_induced = thrust_clamped * F.relu(nu_0 - ambient_inflow_wz)
        q_profile = (1.0 / 8.0) * RHO_AIR * ROTOR_SOLIDITY * BLADE_CD0 * ROTOR_DISK_AREA * PROP_RADIUS * (omega_props * PROP_RADIUS)**2
        q_induced = p_induced / omega_props.clamp(min=50.0)

        motor_currents = ((q_profile + q_induced) / MOTOR_TORQUE_CONSTANT) + NO_LOAD_CURRENT

        if pusher_thrust is not None and v_air_x is not None:
            t_p = F.relu(pusher_thrust)
            va = F.relu(v_air_x)
            vi_p = -0.5 * va + torch.sqrt((0.5 * va)**2 + t_p / (2.0 * RHO_AIR * PUSHER_DISK_AREA) + 1e-6)
            p_pusher_elec = (t_p * (va + vi_p)) / PUSHER_PROP_EFF + torch.where(t_p > 0.05, 3.0, 0.0)
        else:
            p_pusher_elec = torch.zeros_like(u1)

        scull_current = p_scull_load / BATTERY_V0
        pusher_current = p_pusher_elec / BATTERY_V0
        total_current = torch.sum(motor_currents, dim=-1, keepdim=True) + scull_current + pusher_current + AVIONICS_CURRENT

        r0_temp = BATTERY_R0_NOM * torch.exp(ARRHENIUS_COEFF * (1.0 / t_cell.clamp(min=270.0) - 1.0 / T_AMBIENT_K))
        soc_next = (soc - (total_current * dt) / BATTERY_CAPACITY_COULOMBS).clamp(min=0.05, max=1.0)
        v_ocv = 21.0 + 3.8 * soc_next + 0.4 * (soc_next**2)

        exp_1 = math.exp(-dt / self.tau1)
        exp_2 = math.exp(-dt / self.tau2)

        u1_next = u1 * exp_1 + total_current * BATTERY_R1 * (1.0 - exp_1)
        u2_next = u2 * exp_2 + total_current * BATTERY_R2 * (1.0 - exp_2)
        v_terminal = v_ocv - total_current * r0_temp - u1_next - u2_next

        p_profile = q_profile * omega_props
        p_coils = torch.sum((motor_currents**2) * MOTOR_COIL_RESISTANCE, dim=-1, keepdim=True)
        p_batt_heat = (total_current**2) * r0_temp + total_current * (u1_next + u2_next)
        p_no_load_avionics = (4.0 * NO_LOAD_CURRENT + AVIONICS_CURRENT) * v_terminal
        p_electrical_total = (
            torch.sum(p_induced + p_profile, dim=-1, keepdim=True)
            + p_coils + p_batt_heat + p_no_load_avionics + p_scull_load + p_pusher_elec
        )

        t_cell_next = t_cell + (p_batt_heat - (t_cell - T_AMBIENT_K) / BATTERY_THERMAL_RES) * (dt / BATTERY_THERMAL_CAP)
        return p_electrical_total, v_terminal, total_current, u1_next, u2_next, soc_next, t_cell_next


# =============================================================================
# 8. DIFFERENTIABLE DYKSTRA-STYLE QP & FAST REGULARIZED DUAL QP SOLVER
# =============================================================================
class DifferentiableDykstraQP(nn.Module):
    """Soft, differentiable projection minimizing 0.5||u - u_nom||^2 + 0.5*rho||s||^2 s.t. A u - s <= b."""
    def __init__(self, max_iters=8, rho_slack=25.0, beta_smooth=18.0):
        super().__init__()
        self.max_iters = max_iters
        self.rho = rho_slack
        self.beta = beta_smooth

    def forward(self, u_nom, u_prev, A, b, dt):
        da_max = float(MAX_JERK_ACCEL * dt)
        u_clamped = torch.max(torch.min(u_nom, u_prev + da_max), u_prev - da_max)

        N, M_c, _ = A.shape
        u = u_clamped.clone()
        z_list = [torch.zeros((N, 1, 3), device=u.device, dtype=u.dtype) for _ in range(M_c)]
        p_list = [torch.zeros((N, 1, 1), device=u.device, dtype=u.dtype) for _ in range(M_c)]

        A_norm_sq = torch.sum(A * A, dim=-1, keepdim=True).clamp(min=1e-6)
        rho_inv = 1.0 / self.rho

        for _ in range(self.max_iters):
            for i in range(M_c):
                a_i = A[:, i:i+1, :]
                b_i = b[:, i:i+1, :]
                z_i = z_list[i]

                u_p = u.unsqueeze(1) - z_i
                dot_val = torch.sum(a_i * u_p, dim=-1, keepdim=True)
                viol = dot_val - b_i

                step = F.softplus(viol, beta=self.beta) / (A_norm_sq[:, i:i+1, :] + rho_inv)
                delta_u = step * a_i
                s_i_new = step * rho_inv

                u_new = (u_p - delta_u).squeeze(1)
                z_list[i] = u_new.unsqueeze(1) - u_p
                p_list[i] = F.relu(s_i_new)
                u = u_new

        p = torch.cat(p_list, dim=1)
        viol_final = torch.matmul(A, u.unsqueeze(-1)) - b
        slack_tensor = F.relu(viol_final).squeeze(-1)
        override_tensor = safe_norm(u - u_nom, dim=-1, keepdim=True)
        active_barrier_idx = torch.argmax(p.squeeze(-1), dim=-1, keepdim=True)
        active_barrier_val = torch.max(p.squeeze(-1), dim=-1, keepdim=True).values
        is_jerk_active = safe_norm(u_clamped - u_nom, dim=-1, keepdim=True) > 1e-3

        return u, override_tensor, slack_tensor, active_barrier_idx, active_barrier_val, is_jerk_active


class EmbeddedDeterministicQP:
    def __init__(self, max_iter=8, rho_reg=0.08):
        self.max_iter = max_iter
        self.rho_reg = rho_reg

    def solve_tensor(self, u_nom, A, b):
        N, M, _ = A.shape
        Q = torch.bmm(A, A.transpose(-1, -2))
        eye_M = torch.eye(M, device=u_nom.device, dtype=u_nom.dtype).unsqueeze(0).repeat(N, 1, 1)
        Q_reg = Q + self.rho_reg * eye_M

        c = torch.bmm(A, u_nom.unsqueeze(-1)) - b
        lam = torch.zeros((N, M, 1), device=u_nom.device, dtype=u_nom.dtype)
        diag_Q = torch.diagonal(Q_reg, dim1=-2, dim2=-1).unsqueeze(-1)

        for _ in range(self.max_iter):
            for i in range(M):
                q_i = Q_reg[:, i:i+1, :]
                grad_i = torch.bmm(q_i, lam) - c[:, i:i+1, :]
                lam_i_new = F.relu(lam[:, i:i+1, :] - grad_i / diag_Q[:, i:i+1, :])
                lam[:, i:i+1, :] = lam_i_new

        delta_u = torch.bmm(A.transpose(-1, -2), lam).squeeze(-1)
        delta_u_clamped = torch.clamp(delta_u, -12.0, 12.0)
        u_opt = u_nom - delta_u_clamped
        override = safe_norm(delta_u_clamped, dim=-1, keepdim=True)
        active_idx = torch.argmax(lam.squeeze(-1), dim=-1, keepdim=True)
        return u_opt, override, active_idx


# =============================================================================
# 9. FAST L1 ADAPTIVE DISTURBANCE OBSERVER
# =============================================================================
class FastL1AdaptiveDisturbanceObserver:
    """Single-lag disturbance observer, tau = 0.25 s."""
    def __init__(self, gamma=24.0, alpha_l1=0.85, device=DEVICE, tau=0.25, sigma_limit=6.0):
        self.gamma = gamma
        self.alpha = alpha_l1
        self.tau = tau
        self.sigma_limit = sigma_limit
        self.device = device
        self.sigma_hat = torch.zeros((1, 3), device=device)
        self.sigma_filtered = torch.zeros((1, 3), device=device)

    def reset(self, num_envs):
        self.sigma_hat = torch.zeros((num_envs, 3), device=self.device)
        self.sigma_filtered = torch.zeros((num_envs, 3), device=self.device)

    def detach_state(self):
        self.sigma_hat = self.sigma_hat.detach()
        self.sigma_filtered = self.sigma_filtered.detach()

    def step(self, accel_measured, accel_commanded, dt, sat_mask=None):
        a = math.exp(-dt / self.tau)
        d_hat = torch.clamp(accel_measured - accel_commanded + self.sigma_filtered, -15.0, 15.0)
        sigma_updated = torch.clamp(a * self.sigma_filtered + (1.0 - a) * d_hat, -self.sigma_limit, self.sigma_limit)
        if sat_mask is not None:
            sigma_decay = self.sigma_filtered * math.exp(-dt / 0.50)
            self.sigma_hat = torch.where(sat_mask, torch.zeros_like(d_hat), d_hat)
            self.sigma_filtered = torch.where(sat_mask, sigma_decay, sigma_updated)
        else:
            self.sigma_hat = d_hat
            self.sigma_filtered = sigma_updated
        return self.sigma_filtered


# =============================================================================
# 10. 20-STATE NEURAL PILOT & BLOCK-DECOUPLED NEURAL CONTRACTION METRIC (NCM)
# =============================================================================
class FullNeuralWavePilot(nn.Module):
    def __init__(self, state_dim=20, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, obs):
        raw = torch.tanh(self.net(obs))
        e_speed = obs[:, 0:1]
        e_height = obs[:, 1:2]
        e_int_h = obs[:, 2:3]
        e_y = obs[:, 3:4]
        h_dot = obs[:, 4:5]
        v_y = obs[:, 9:10]

        a_x = raw[:, 0:1] * 1.2 - 1.8 * e_speed
        a_y = raw[:, 1:2] * 0.9 - 2.4 * v_y - 2.8 * e_y
        a_z = raw[:, 2:3] * 1.2 - 9.5 * e_height - 3.5 * e_int_h - 4.2 * h_dot
        return torch.cat([a_x, a_y, a_z], dim=-1)


class NeuralContractionMetric(nn.Module):
    """Block-decoupled SPD contraction metric over the 5D controllable subspace:
    3x3 horizontal block on [e_y, e_vx, e_vy] and 2x2 vertical block on [e_h, e_vz],
    embedded in (N, 6, 6) so position-velocity cross-terms match both loop gains."""
    def __init__(self, state_dim=6, alpha=0.10):
        super().__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(state_dim - 1, 48),
            nn.SiLU(),
            nn.Linear(48, 48),
            nn.SiLU(),
            nn.Linear(48, 9),
        )

    def forward(self, state_error):
        device_str = "cuda" if state_error.is_cuda else "cpu"
        with get_amp_autocast(device_str, enabled=False):
            err_5 = state_error[:, 1:].float()
            params = self.net(err_5)
            z = torch.zeros_like(params[:, 0:1])
            o = torch.ones_like(params[:, 0:1])

            # 3x3 Cholesky block for [e_y, e_vx, e_vy]
            l00 = F.softplus(params[:, 0:1]) + 0.45
            l10 = params[:, 1:2]
            l11 = F.softplus(params[:, 2:3]) + 0.45
            l20 = params[:, 3:4]
            l21 = params[:, 4:5]
            l22 = F.softplus(params[:, 5:6]) + 0.45

            L_xy = torch.stack([
                torch.cat([l00, z, z], dim=-1),
                torch.cat([l10, l11, z], dim=-1),
                torch.cat([l20, l21, l22], dim=-1),
            ], dim=1)
            eye3 = torch.eye(3, device=state_error.device, dtype=L_xy.dtype).unsqueeze(0)
            M_xy = torch.matmul(L_xy, L_xy.transpose(-1, -2)) + 0.25 * eye3

            # 2x2 Cholesky block for [e_h, e_vz]
            lz00 = F.softplus(params[:, 6:7]) + 0.45
            lz10 = params[:, 7:8]
            lz11 = F.softplus(params[:, 8:9]) + 0.45

            L_z = torch.stack([
                torch.cat([lz00, z], dim=-1),
                torch.cat([lz10, lz11], dim=-1),
            ], dim=1)
            eye2 = torch.eye(2, device=state_error.device, dtype=L_z.dtype).unsqueeze(0)
            M_z = torch.matmul(L_z, L_z.transpose(-1, -2)) + 0.25 * eye2

            # Embed into (N, 6, 6) matching state_err = [0, e_y, e_h, e_vx, e_vy, e_vz]
            row0 = torch.cat([o, z, z, z, z, z], dim=-1)
            row1 = torch.cat([z, M_xy[:, 0:1, 0], z, M_xy[:, 0:1, 1], M_xy[:, 0:1, 2], z], dim=-1)
            row2 = torch.cat([z, z, M_z[:, 0:1, 0], z, z, M_z[:, 0:1, 1]], dim=-1)
            row3 = torch.cat([z, M_xy[:, 1:2, 0], z, M_xy[:, 1:2, 1], M_xy[:, 1:2, 2], z], dim=-1)
            row4 = torch.cat([z, M_xy[:, 2:3, 0], z, M_xy[:, 2:3, 1], M_xy[:, 2:3, 2], z], dim=-1)
            row5 = torch.cat([z, z, M_z[:, 1:2, 0], z, z, M_z[:, 1:2, 1]], dim=-1)
            return torch.stack([row0, row1, row2, row3, row4, row5], dim=1)


# =============================================================================
# 11. 6-DoF ROBOT DYNAMICS & 2-AXIS ROTOR GIMBAL HYBRID WRENCH ALLOCATOR
# =============================================================================
class WaveRidingAerialRobot:
    def __init__(self, dt, device):
        self.dt = dt
        self.device = device
        self.J = torch.tensor([[0.045, 0.052, 0.088]], device=device)
        self.J_inv = 1.0 / self.J
        self.d = ARM_LENGTH / math.sqrt(2.0)
        self.c_q = TORQUE_TO_THRUST_RATIO
        self.rotor_aero = RotorAeromechanics(device)

        d = self.d
        self.r_x = torch.tensor([[+d, +d, -d, -d]], device=device)
        self.r_y = torch.tensor([[-d, +d, +d, -d]], device=device)
        self.yaw_tx = -self.r_y / ARM_LENGTH
        self.yaw_ty = +self.r_x / ARM_LENGTH
        self.spin_signs = torch.tensor([[-1.0, +1.0, -1.0, +1.0]], device=device)

    def allocate_wrench_hybrid(self, Fx_cmd, Fy_cmd, Fz_cmd, torques_desired, v_batt, ige_gain,
                               d_sub, airspeed, is_soaring, morphology_dict, delta_collective=None):
        """Hybrid allocator with 4 independently 2-axis tilting rotors (0-15 deg omni cone)
        and low-speed hydro-surface fadeout so surfaces center cleanly near hover."""
        if not torch.is_tensor(v_batt):
            v_batt = torch.tensor(v_batt, device=Fz_cmd.device)
        thrust_ceiling = MAX_THRUST_PER_MOTOR * (v_batt / BATTERY_V0)**2

        q_water = 0.5 * RHO_WATER_NOM * (airspeed.clamp(min=0.5)**2)
        sub_frac = (d_sub / 0.040).clamp(min=0.0, max=1.0)
        hydro_speed_fade = ((airspeed - 0.30) / 0.70).clamp(min=0.0, max=1.0)
        eff_sub_frac = sub_frac * hydro_speed_fade

        f_span = morphology_dict["foil_span"]
        f_area = morphology_dict["foil_area"]
        caudal_area = morphology_dict["caudal_area"]

        max_hydro_roll_torque = 1.00 * eff_sub_frac
        max_hydro_yaw_torque = 0.50 * eff_sub_frac

        hydro_roll_gain = 2.0 * (f_span * 0.25) * q_water * (f_area * 0.5) * HYDRO_CL_DELTA * sub_frac
        hydro_yaw_gain = q_water * caudal_area * HYDRO_RUDDER_CY_DELTA * sub_frac * 0.08

        tx_des = torch.clamp(torques_desired[:, 0:1], -3.5, 3.5)
        ty_des = torch.clamp(torques_desired[:, 1:2], -3.5, 3.5)
        tz_des = torch.clamp(torques_desired[:, 2:3], -1.2, 1.2)

        tau_hydro_roll_alloc = torch.clamp(tx_des, -max_hydro_roll_torque, max_hydro_roll_torque)
        tau_hydro_yaw_alloc = torch.clamp(tz_des, -max_hydro_yaw_torque, max_hydro_yaw_torque)

        delta_roll = torch.clamp(tau_hydro_roll_alloc / (hydro_roll_gain + 0.15), -MAX_HYDRO_DEFLECTION, MAX_HYDRO_DEFLECTION) * hydro_speed_fade
        delta_rud = -torch.clamp(tau_hydro_yaw_alloc / (hydro_yaw_gain + 0.10), -MAX_HYDRO_DEFLECTION, MAX_HYDRO_DEFLECTION) * hydro_speed_fade

        if delta_collective is None:
            delta_collective = torch.zeros_like(delta_roll)
        delta_collective_eff = delta_collective * hydro_speed_fade
        delta_e_L = torch.clamp(delta_roll + delta_collective_eff, -MAX_HYDRO_DEFLECTION, MAX_HYDRO_DEFLECTION)
        delta_e_R = torch.clamp(-delta_roll + delta_collective_eff, -MAX_HYDRO_DEFLECTION, MAX_HYDRO_DEFLECTION)

        tx_residual = tx_des - tau_hydro_roll_alloc
        ty_residual = ty_des
        tz_after_hydro = tz_des - tau_hydro_yaw_alloc

        Fz_compensated = Fz_cmd / torch.clamp(ige_gain, min=1.0, max=1.6)
        thrust_floor_per_motor = torch.where(is_soaring.bool(), torch.full_like(Fz_compensated, MOTOR_IDLE_FLOOR), torch.full_like(Fz_compensated, 0.90))
        f_floor_tensor = thrust_floor_per_motor * 4.0
        f_ceil_tensor = (thrust_ceiling - 1.20) * 4.0
        Fz_base = torch.minimum(torch.maximum(Fz_compensated, f_floor_tensor), f_ceil_tensor)

        d = self.d
        c_q = self.c_q

        f_nom = 0.25 * Fz_base
        df_x = 0.25 * (tx_residual / d)
        df_y = -0.25 * (ty_residual / d)

        df_rp = torch.cat([-df_x + df_y, +df_x + df_y, +df_x - df_y, -df_x - df_y], dim=-1)
        raw_rp = f_nom + df_rp
        min_rp = torch.min(raw_rp, dim=-1, keepdim=True).values
        rp_shifted = raw_rp + F.relu(MOTOR_IDLE_FLOOR - min_rp)
        max_rp = torch.max(rp_shifted, dim=-1, keepdim=True).values
        fz_rotors = torch.minimum((rp_shifted - F.relu(max_rp - thrust_ceiling)).clamp(min=MOTOR_IDLE_FLOOR), thrust_ceiling)

        f_lat_max = fz_rotors * ROTOR_MAX_TILT_TAN
        f_lat_sum = torch.sum(f_lat_max, dim=-1, keepdim=True).clamp(min=1e-4)
        tz_gimbal_max = ARM_LENGTH * f_lat_sum
        tz_gimbal_alloc = torch.clamp(tz_after_hydro, -0.65 * tz_gimbal_max, 0.65 * tz_gimbal_max)

        f_yaw_mag = (f_lat_max / f_lat_sum) * (tz_gimbal_alloc / ARM_LENGTH)
        f_yaw_x = f_yaw_mag * self.yaw_tx
        f_yaw_y = f_yaw_mag * self.yaw_ty

        fx_needed = Fx_cmd - torch.sum(f_yaw_x, dim=-1, keepdim=True)
        fy_needed = Fy_cmd - torch.sum(f_yaw_y, dim=-1, keepdim=True)
        fz_weights = fz_rotors / torch.sum(fz_rotors, dim=-1, keepdim=True).clamp(min=1e-4)

        f_x_raw = f_yaw_x + fz_weights * fx_needed
        f_y_raw = f_yaw_y + fz_weights * fy_needed

        f_lat_raw = torch.sqrt(f_x_raw**2 + f_y_raw**2 + 1e-8)
        cone_scale = torch.clamp(f_lat_max / f_lat_raw, max=1.0)
        f_x_rot = f_x_raw * cone_scale
        f_y_rot = f_y_raw * cone_scale

        tz_gimbal_actual = torch.sum(self.r_x * f_y_rot - self.r_y * f_x_rot, dim=-1, keepdim=True)
        tz_residual = tz_after_hydro - tz_gimbal_actual
        df_z = 0.25 * (tz_residual / c_q)

        f_total_pre_yaw = torch.sqrt(f_x_rot**2 + f_y_rot**2 + fz_rotors**2)
        headroom_low = (f_total_pre_yaw - MOTOR_IDLE_FLOOR).clamp(min=0.0)
        headroom_high = (thrust_ceiling - f_total_pre_yaw).clamp(min=0.0)
        max_df_z = torch.min(torch.minimum(headroom_low, headroom_high), dim=-1, keepdim=True).values
        df_z_clamped = torch.clamp(df_z, -max_df_z, max_df_z)
        yaw_pattern = self.spin_signs * df_z_clamped

        clamped_motors = torch.minimum((f_total_pre_yaw + yaw_pattern).clamp(min=MOTOR_IDLE_FLOOR), thrust_ceiling)
        f_mag_actual = torch.sqrt(f_x_rot**2 + f_y_rot**2 + fz_rotors**2).clamp(min=1e-4)
        u_x_cmd = f_x_rot / f_mag_actual
        u_y_cmd = f_y_rot / f_mag_actual
        target_tilts = torch.stack([u_x_cmd, u_y_cmd], dim=-1)

        fx_gimbal_alloc = torch.sum(f_x_rot, dim=-1, keepdim=True)
        fy_gimbal_alloc = torch.sum(f_y_rot, dim=-1, keepdim=True)

        is_clipped_min = (clamped_motors <= (MOTOR_IDLE_FLOOR + 0.01))
        is_clipped_max = (clamped_motors >= thrust_ceiling - 1e-3)
        headroom_min = min_rp - MOTOR_IDLE_FLOOR
        tz_delivered = tz_gimbal_actual + 4.0 * c_q * df_z_clamped
        torque_scale = torch.where(
            torch.abs(tz_after_hydro) > 1e-3,
            (torch.abs(tz_delivered) / (torch.abs(tz_after_hydro) + 1e-6)).clamp(max=1.0),
            torch.ones_like(headroom_min)
        )
        hydro_surfaces = torch.cat([delta_e_L, delta_e_R, delta_rud], dim=-1)

        return (clamped_motors, target_tilts, hydro_surfaces, headroom_min, is_clipped_min,
                is_clipped_max, torque_scale, fx_gimbal_alloc, fy_gimbal_alloc)

    def integrate(self, pos, vel, rot, omega, motors, rotor_tilts, pusher_thrust, nu_states,
                  target_motors, target_tilts, target_pusher, f_wing, f_hydro,
                  tau_hydro_pitch, tau_hydro_roll, tau_hydro_yaw, ambient_wind, eta, wave_slope,
                  mass, motor_tau_up, motor_tau_down):
        """Newtonian 6-DoF integration with independent 2-axis rotor hub gimbals (0-15 deg)."""
        N = pos.shape[0]
        c_q = self.c_q

        tau_m = torch.where(target_motors >= motors, motor_tau_up, motor_tau_down)
        motors_next = motors + (self.dt / (tau_m + self.dt)) * (target_motors - motors)
        motors_next = torch.nan_to_num(motors_next, nan=MOTOR_IDLE_FLOOR).clamp(min=MOTOR_IDLE_FLOOR)

        tilts_next = rotor_tilts + (self.dt / (GIMBAL_TAU + self.dt)) * (target_tilts - rotor_tilts)
        tilts_next = torch.nan_to_num(tilts_next, nan=0.0)
        lat_sin = torch.sqrt(tilts_next[:, :, 0]**2 + tilts_next[:, :, 1]**2 + 1e-10)
        lat_scale = torch.clamp(ROTOR_MAX_TILT_SIN / lat_sin, max=1.0).unsqueeze(-1)
        tilts_next = tilts_next * lat_scale

        ux = tilts_next[:, :, 0]
        uy = tilts_next[:, :, 1]
        uz = torch.sqrt((1.0 - ux**2 - uy**2).clamp(min=ROTOR_MAX_TILT_COS**2))

        tau_p = torch.where(target_pusher >= pusher_thrust, motor_tau_up, motor_tau_down)
        pusher_next = pusher_thrust + (self.dt / (tau_p + self.dt)) * (target_pusher - pusher_thrust)
        pusher_next = torch.nan_to_num(pusher_next, nan=0.0).clamp(min=0.0, max=MAX_PUSHER_THRUST)

        h_rotors = (pos[:, 2:3] - eta).clamp(min=0.04)
        ige_rotor_gain = 1.0 / (1.0 - torch.clamp((PROP_RADIUS / (4.0 * h_rotors))**2, max=0.35))
        spray_derate = 1.0 + 0.15 * F.relu(0.14 - h_rotors) * wave_slope.clamp(max=0.5)

        f_rotor_x = motors_next * ux
        f_rotor_y = motors_next * uy
        f_rotor_z = motors_next * uz

        f_body_total = torch.cat([
            torch.sum(f_rotor_x, dim=-1, keepdim=True) + pusher_next,
            torch.sum(f_rotor_y, dim=-1, keepdim=True),
            (torch.sum(f_rotor_z, dim=-1, keepdim=True) * ige_rotor_gain) / spray_derate
        ], dim=-1)
        f_rotors_world = torch.matmul(rot, f_body_total.unsqueeze(-1)).squeeze(-1)

        v_air_hub_world = vel.unsqueeze(1) - ambient_wind.unsqueeze(1)
        rot_T = rot.transpose(-1, -2).unsqueeze(1)
        v_air_hub_body = torch.matmul(rot_T, v_air_hub_world.unsqueeze(-1)).squeeze(-1)

        nu_states_next, a_1, b_1, omega_props = self.rotor_aero.step_inflow_and_flapping(
            nu_states, motors_next, v_air_hub_body, omega, self.dt
        )

        total_force = (
            f_rotors_world
            + f_wing
            + f_hydro
            + torch.cat([torch.zeros((N, 2), device=self.device), -mass * GRAVITY], dim=-1)
        )
        total_force = torch.nan_to_num(total_force, nan=0.0, posinf=60.0, neginf=-60.0)

        accel = total_force / mass
        vel_next = (vel + accel * self.dt).clamp(-20.0, 20.0)
        pos_next = pos + vel_next * self.dt

        tx = torch.sum(self.r_y * f_rotor_z, dim=-1, keepdim=True) + tau_hydro_roll
        ty = -torch.sum(self.r_x * f_rotor_z, dim=-1, keepdim=True) + tau_hydro_pitch
        tz = (
            torch.sum(self.r_x * f_rotor_y - self.r_y * f_rotor_x, dim=-1, keepdim=True)
            + c_q * torch.sum(self.spin_signs * f_rotor_z, dim=-1, keepdim=True)
            + tau_hydro_yaw
        )

        gyroscopic = torch.cross(omega, omega * self.J, dim=-1)
        omega_dot = (torch.cat([tx, ty, tz], dim=-1) - gyroscopic) * self.J_inv
        omega_next = torch.nan_to_num(omega + omega_dot * self.dt, nan=0.0).clamp(-10.0, 10.0)

        v_rot = omega_next * self.dt
        theta = safe_norm(v_rot, dim=-1, keepdim=True).clamp(min=1e-6)
        vx, vy, vz = v_rot[:, 0:1], v_rot[:, 1:2], v_rot[:, 2:3]
        zv = torch.zeros_like(vx)

        K = torch.stack([
            torch.cat([zv, -vz, vy], dim=-1),
            torch.cat([vz, zv, -vx], dim=-1),
            torch.cat([-vy, vx, zv], dim=-1),
        ], dim=1)

        eye3 = torch.eye(3, device=self.device).unsqueeze(0)
        s_th = torch.sin(theta) / theta
        c_th = (1.0 - torch.cos(theta)) / (theta**2)
        dR = eye3 + s_th.unsqueeze(-1) * K + c_th.unsqueeze(-1) * torch.matmul(K, K)
        rot_next = torch.matmul(rot, dR)

        c0 = rot_next[:, :, 0] / safe_norm(rot_next[:, :, 0], dim=-1, keepdim=True)
        c2 = torch.cross(c0, rot_next[:, :, 1], dim=-1)
        c2 = c2 / safe_norm(c2, dim=-1, keepdim=True)
        c1 = torch.cross(c2, c0, dim=-1)
        rot_clean = torch.stack([c0, c1, c2], dim=-1)

        return (pos_next, vel_next, rot_clean, omega_next, motors_next, tilts_next,
                pusher_next, nu_states_next, omega_props, accel, ige_rotor_gain)

# =============================================================================
# 12. SIM-TO-REAL MULTI-ENVIRONMENT (WITH CO-EVOLUTIONARY MORPHOLOGY)
# =============================================================================
class WaveRidingAutonomyEnv:
    def __init__(self, num_envs=1024, dt=0.015, device=DEVICE, use_embedded_qp=False):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.use_embedded_qp = use_embedded_qp

        self.morphology = VehicleMorphologyGenome(device)

        self.waves = DirectionalOceanWaveField(dt, airspeed=5.20, num_waves=32, device=device)
        self.aero_hydro = MultimodalAeroHydroDynamics(device)
        self.robot = WaveRidingAerialRobot(dt, device)
        self.powertrain = ElectroThermalBatteryPowertrain(device)
        self.safety_filter = DifferentiableDykstraQP(max_iters=8, rho_slack=25.0, beta_smooth=18.0)
        self.embedded_qp = EmbeddedDeterministicQP(max_iter=8, rho_reg=0.08)

        self.perception = HardwarePerceptionSuite(dt, device)
        self.dispersion_observer = RealSeaWaveDispersionObserver(dt, device)
        self.l1_observer = FastL1AdaptiveDisturbanceObserver(gamma=24.0, alpha_l1=0.85, device=device)
        self.ekf = AeromarineErrorStateEKF(num_envs, dt, device=device)

        self.guidance_delay_steps = max(1, int(round(GUIDANCE_DELAY_SECONDS / dt)))
        self.guidance_cmd_buffer = deque(maxlen=self.guidance_delay_steps + 1)
        self.pilot_filter_a = math.exp(-dt / 0.0164)
        self.sensor_period_steps = max(1, int(round(REF_DT / dt)))
        self.gnss_period_steps = max(1, int(round(0.10 / dt)))
        self.step_count = 0

        self.current_mode = torch.full((num_envs, 1), MODE_FOILING, dtype=torch.long, device=device)
        self.target_speed = 5.20
        self.wave_curriculum_scale = 1.0

        self.h_target_filtered = torch.full((num_envs, 1), RIDE_HEIGHT_OPTIMAL, device=device)
        self.v_target_filtered = torch.full((num_envs, 1), self.target_speed, device=device)
        self.phi_des_filtered = torch.zeros((num_envs, 1), device=device)
        self.psi_des_filtered = torch.zeros((num_envs, 1), device=device)
        self.e_int_height = torch.zeros((num_envs, 1), device=device)
        self.e_int_y = torch.zeros((num_envs, 1), device=device)

        self.k_R = torch.tensor([[7.5, 8.0, 4.0]], device=device)
        self.k_omega = torch.tensor([[1.25, 1.35, 0.65]], device=device)

        self.hydro_surfaces = torch.zeros((num_envs, 3), device=device)
        self.hydro_surfaces_actual = torch.zeros((num_envs, 3), device=device)
        self.rotor_tilts = torch.zeros((num_envs, 4, 2), device=device)
        self.auto_mission_active = False
        self.auto_mission_timer = 0.0

        self.A_cbf_crash = torch.tensor([[0.0, 0.0, -1.0]], device=self.device).repeat(num_envs, 1, 1)
        self.A_cbf_ceil = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).repeat(num_envs, 1, 1)

        self.reset()

    def set_mode(self, new_mode):
        self.current_mode = torch.full((self.num_envs, 1), new_mode, dtype=torch.long, device=self.device)

    def set_wave_curriculum(self, scale):
        self.wave_curriculum_scale = scale

    def detach_states(self):
        """Truncated BPTT boundary: detaches all stateful tensors between rollout windows."""
        self.pos = self.pos.detach()
        self.vel = self.vel.detach()
        self.rot = self.rot.detach()
        self.omega = self.omega.detach()
        self.motors = self.motors.detach()
        self.rotor_tilts = self.rotor_tilts.detach()
        self.pusher_thrust = self.pusher_thrust.detach()
        self.nu_states = self.nu_states.detach()
        self.u1 = self.u1.detach()
        self.u2 = self.u2.detach()
        self.soc = self.soc.detach()
        self.t_cell = self.t_cell.detach()
        self.v_batt = self.v_batt.detach()
        self.h_target_filtered = self.h_target_filtered.detach()
        self.v_target_filtered = self.v_target_filtered.detach()
        self.phi_des_filtered = self.phi_des_filtered.detach()
        self.psi_des_filtered = self.psi_des_filtered.detach()
        self.e_int_height = self.e_int_height.detach()
        self.e_int_y = self.e_int_y.detach()
        self.a_pilot_prev = self.a_pilot_prev.detach()
        self.a_pilot_filtered = self.a_pilot_filtered.detach()
        self.a_pre_dob_prev = self.a_pre_dob_prev.detach()
        self.l1_sat_prev = self.l1_sat_prev.detach()
        self.scull_throttle_cmd = self.scull_throttle_cmd.detach()
        self.last_deta_dx_nadir = self.last_deta_dx_nadir.detach()
        self.last_w_up_nadir = self.last_w_up_nadir.detach()
        self.hydro_surfaces_actual = self.hydro_surfaces_actual.detach()
        self.hydro_surfaces = self.hydro_surfaces.detach()
        self.is_vent_L = self.is_vent_L.detach()
        self.is_vent_R = self.is_vent_R.detach()
        self.clearance_hold = self.clearance_hold.detach()
        self.follower_accel_prev = self.follower_accel_prev.detach()
        detached_buf = [t.detach() for t in self.guidance_cmd_buffer]
        self.guidance_cmd_buffer.clear()
        for t in detached_buf:
            self.guidance_cmd_buffer.append(t)
        self.ekf.detach_state()
        self.l1_observer.detach_state()

    def reset(self, mode_mix=False):
        N = self.num_envs
        self.sim_time = torch.zeros((N, 1), device=self.device)
        self.step_count = 0
        self.waves.reset(N)
        self.l1_observer.reset(N)
        self.dispersion_observer.reset(N)

        self.mass_noise = torch.empty((N, 1), device=self.device).uniform_(0.96, 1.04)
        with torch.no_grad():
            m_evolved = self.morphology()["total_mass"]
        self.mass = (m_evolved.repeat(N, 1) * self.mass_noise).detach()
        self.foil_decalage = torch.empty((N, 1), device=self.device).uniform_(
            math.radians(FOIL_DECALAGE_DEG[0]), math.radians(FOIL_DECALAGE_DEG[1])
        )
        self.motor_tau_up = torch.empty((N, 1), device=self.device).uniform_(0.018, 0.026)
        self.motor_tau_down = torch.empty((N, 1), device=self.device).uniform_(0.032, 0.044)
        self.rho_water_local = torch.empty((N, 1), device=self.device).uniform_(1000.0, 1028.0)

        self.true_ba = (torch.randn((N, 3), device=self.device) * 0.03).clamp(-0.08, 0.08)
        self.true_bw = (torch.randn((N, 3), device=self.device) * 0.01).clamp(-0.03, 0.03)

        init_x = torch.rand((N, 1), device=self.device) * 40.0

        if not mode_mix:
            self.current_mode = torch.full((N, 1), MODE_FOILING, dtype=torch.long, device=self.device)
            init_y = torch.randn((N, 1), device=self.device) * (0.25 if N > 1 else 0.05)
            init_clearance = torch.full((N, 1), RIDE_HEIGHT_OPTIMAL, device=self.device)
            vx0 = torch.full((N, 1), self.target_speed, device=self.device)
            vy0 = torch.zeros((N, 1), device=self.device)
            vz0 = torch.zeros((N, 1), device=self.device)
            phi0 = torch.zeros((N, 1), device=self.device)
            theta0 = torch.full((N, 1), math.radians(INIT_PITCH_NOSE_DOWN_DEG), device=self.device)
            psi0 = torch.zeros((N, 1), device=self.device)
            omega0 = torch.zeros((N, 3), device=self.device)
            self.h_target_filtered.fill_(RIDE_HEIGHT_OPTIMAL)
            self.v_target_filtered.fill_(self.target_speed)
            self.motors = torch.full((N, 4), 1.1, device=self.device)
            self.pusher_thrust = torch.full((N, 1), 2.5, device=self.device)
        else:
            u_mode = torch.rand((N, 1), device=self.device)
            mode_t = torch.full((N, 1), MODE_FOILING, dtype=torch.long, device=self.device)
            mode_t = torch.where(u_mode >= 0.30, torch.full_like(mode_t, MODE_LOITER), mode_t)
            mode_t = torch.where(u_mode >= 0.45, torch.full_like(mode_t, MODE_TAKEOFF), mode_t)
            mode_t = torch.where(u_mode >= 0.60, torch.full_like(mode_t, MODE_EGRESS), mode_t)
            mode_t = torch.where(u_mode >= 0.68, torch.full_like(mode_t, MODE_FREE_FLIGHT), mode_t)
            mode_t = torch.where(u_mode >= 0.75, torch.full_like(mode_t, MODE_INGRESS), mode_t)
            mode_t = torch.where(u_mode >= 0.90, torch.full_like(mode_t, MODE_SOARING), mode_t)
            self.current_mode = mode_t

            init_y = torch.empty((N, 1), device=self.device).uniform_(-1.2, 1.2)
            init_clearance = torch.empty((N, 1), device=self.device).uniform_(0.08, 0.34)
            vx0 = torch.empty((N, 1), device=self.device).uniform_(4.0, 6.2)
            vy0 = torch.empty((N, 1), device=self.device).uniform_(-0.7, 0.7)
            vz0 = torch.empty((N, 1), device=self.device).uniform_(-0.5, 0.5)
            phi0 = torch.empty((N, 1), device=self.device).uniform_(-math.radians(12.0), math.radians(12.0))
            theta0 = torch.empty((N, 1), device=self.device).uniform_(-math.radians(8.0), math.radians(8.0))
            psi0 = torch.empty((N, 1), device=self.device).uniform_(-math.radians(12.0), math.radians(12.0))
            omega0 = torch.randn((N, 3), device=self.device) * 0.35

            is_l = (mode_t == MODE_LOITER)
            is_to = (mode_t == MODE_TAKEOFF)
            is_eg = (mode_t == MODE_EGRESS)
            is_ff = (mode_t == MODE_FREE_FLIGHT)
            is_in = (mode_t == MODE_INGRESS)
            is_so = (mode_t == MODE_SOARING)

            init_clearance = torch.where(is_l, torch.empty_like(init_clearance).uniform_(0.05, 0.30), init_clearance)
            init_clearance = torch.where(is_to, torch.empty_like(init_clearance).uniform_(0.05, 0.16), init_clearance)
            init_clearance = torch.where(is_eg, torch.empty_like(init_clearance).uniform_(0.16, 0.90), init_clearance)
            init_clearance = torch.where(is_ff, torch.empty_like(init_clearance).uniform_(1.10, 1.90), init_clearance)
            init_clearance = torch.where(is_in, torch.empty_like(init_clearance).uniform_(0.50, 1.70), init_clearance)
            init_clearance = torch.where(is_so, torch.empty_like(init_clearance).uniform_(0.11, 0.22), init_clearance)

            vx0 = torch.where(is_l, torch.empty_like(vx0).uniform_(0.0, 5.2), vx0)
            vx0 = torch.where(is_to, torch.empty_like(vx0).uniform_(0.5, 4.2), vx0)
            vx0 = torch.where(is_eg, torch.empty_like(vx0).uniform_(4.5, 6.5), vx0)
            vx0 = torch.where(is_ff, torch.empty_like(vx0).uniform_(6.5, 9.5), vx0)
            vx0 = torch.where(is_in, torch.empty_like(vx0).uniform_(4.8, 8.5), vx0)
            vx0 = torch.where(is_so, torch.empty_like(vx0).uniform_(4.5, 5.8), vx0)

            self.h_target_filtered = torch.where(is_eg | is_ff, torch.full_like(init_clearance, FREE_FLIGHT_ALTITUDE),
                                     torch.where(is_so, torch.full_like(init_clearance, RIDE_HEIGHT_SOARING),
                                     torch.where(is_l & (vx0 < 1.5), torch.full_like(init_clearance, 0.04),
                                     torch.full_like(init_clearance, RIDE_HEIGHT_OPTIMAL))))
            self.v_target_filtered = torch.where(is_l, torch.zeros_like(vx0),
                                     torch.where(is_ff, torch.full_like(vx0, 8.50),
                                     torch.full_like(vx0, self.target_speed)))

            airborne_ic = (init_clearance > 0.24)
            m_init = torch.where(airborne_ic, torch.full_like(vx0, HOVER_WEIGHT_NOM / 4.0), torch.full_like(vx0, 1.1))
            self.motors = m_init.repeat(1, 4)
            self.pusher_thrust = torch.where((vx0 > 3.0) & (~is_so) & (~is_l), torch.full_like(vx0, 2.5), torch.zeros_like(vx0))

        self.clearance_hold = init_clearance.clone().clamp(0.04, self.perception.radar_r_max)

        eta0, _, _, _, _, _ = self.waves.evaluate_wave_kinematics(
            torch.cat([init_x, init_y, torch.zeros_like(init_x)], dim=-1),
            self.sim_time,
            wave_scale=self.wave_curriculum_scale
        )

        init_z = eta0 + init_clearance
        self.pos = torch.cat([init_x, init_y, init_z], dim=-1)
        self.vel = torch.cat([vx0, vy0, vz0], dim=-1)
        self.omega = omega0
        self.rot = euler_to_rot(phi0, theta0, psi0)

        self.rotor_tilts = torch.zeros((N, 4, 2), device=self.device)
        self.nu_states = torch.zeros((N, 4, 3), device=self.device)
        self.nu_states[:, :, 0] = V_INDUCED_HOVER * 0.20

        self.ekf.reset(N, self.pos, self.vel, self.rot)
        self.follower_accel_prev = torch.zeros((N, 3), device=self.device)
        self.a_pilot_prev = torch.zeros((N, 3), device=self.device)
        self.a_pilot_filtered = torch.zeros((N, 3), device=self.device)
        self.a_pre_dob_prev = torch.zeros((N, 3), device=self.device)
        self.l1_sat_prev = torch.zeros((N, 3), dtype=torch.bool, device=self.device)
        self.scull_throttle_cmd = torch.zeros((N, 1), device=self.device)

        self.guidance_cmd_buffer.clear()
        for _ in range(self.guidance_delay_steps + 1):
            self.guidance_cmd_buffer.append(torch.zeros((N, 3), device=self.device))

        self.is_vent_L = torch.zeros((N, 1), device=self.device)
        self.is_vent_R = torch.zeros((N, 1), device=self.device)
        self.hydro_surfaces = torch.zeros((N, 3), device=self.device)
        self.hydro_surfaces_actual = torch.zeros((N, 3), device=self.device)
        self.phi_des_filtered.zero_()
        self.psi_des_filtered.zero_()
        self.e_int_height.zero_()
        self.e_int_y.zero_()

        self.last_d_sub_hydrostatic = torch.zeros((N, 1), device=self.device)
        self.last_p_static_decoupled = torch.full((N, 1), float(P_ATM), device=self.device)
        self.last_bernoulli_bias_cm = torch.zeros((N, 1), device=self.device)
        self.last_radar_raw = init_clearance.clone()
        self.last_radar_locked = torch.ones((N, 1), dtype=torch.bool, device=self.device)
        self.last_spray_dropout = torch.zeros((N, 1), device=self.device)
        self.last_multipath_rejected = torch.zeros((N, 1), device=self.device)
        self.last_dvl_active = torch.zeros((N, 1), dtype=torch.bool, device=self.device)
        self.last_flow_active = torch.zeros((N, 1), dtype=torch.bool, device=self.device)
        self.last_eta_fwd_meas = torch.zeros((N, 1), device=self.device)
        self.last_eta_fwd_valid = torch.ones((N, 1), dtype=torch.bool, device=self.device)
        self.last_eta_prev = torch.zeros((N, 1), device=self.device)
        self.last_deta_dx_prev = torch.zeros((N, 1), device=self.device)
        self.last_w_up_prev = torch.zeros((N, 1), device=self.device)
        self.last_deta_dx_nadir = torch.zeros((N, 1), device=self.device)
        self.last_w_up_nadir = torch.zeros((N, 1), device=self.device)
        self.last_look_dist = torch.full((N, 1), 0.75, device=self.device)

        self.u1 = torch.zeros((N, 1), device=self.device)
        self.u2 = torch.zeros((N, 1), device=self.device)
        self.soc = torch.full((N, 1), 0.95, device=self.device)
        self.t_cell = torch.full((N, 1), T_AMBIENT_K, device=self.device)
        self.v_batt = torch.full((N, 1), BATTERY_V0, device=self.device)
        self.metrics = {}

    def step(self, actor_net, ncm_net, export_diagnostics=False):
        N = self.num_envs
        self.sim_time = self.sim_time + self.dt
        self.step_count += 1
        ambient_gust = self.waves.step_gust()

        morph_dict = self.morphology()
        self.mass = morph_dict["total_mass"].repeat(N, 1) * self.mass_noise

        # 1. Physical ocean truth
        eta, deta_dx, deta_dy, deta_dt, w_updraft, v_water_orb = self.waves.evaluate_wave_kinematics(
            self.pos, self.sim_time, wave_scale=self.wave_curriculum_scale
        )

        is_soaring = (self.current_mode == MODE_SOARING)

        # 2. Physical multi-phase fluid dynamics
        motor_thrusts_total = torch.sum(self.motors, dim=-1, keepdim=True)
        (
            f_wing, f_hydro, tau_hydro_pitch, tau_hydro_roll, tau_hydro_yaw, h_wing, d_sub_physical, f_slam, f_buoy,
            self.is_vent_L, self.is_vent_R, is_ventilated_overall, wind_shear_factor, strut_root_moment, aoa_tensors, alpha_foil
        ) = self.aero_hydro(
            self.pos, self.vel, self.rot, self.omega, eta, deta_dt, w_updraft, v_water_orb, ambient_gust,
            self.is_vent_L, self.is_vent_R, self.foil_decalage, self.rho_water_local, self.hydro_surfaces,
            motor_thrusts_total, morph_dict, self.sim_time, is_soaring, scull_throttle=self.scull_throttle_cmd
        )

        # 3 & 4. Simulated sensor hardware + dt-invariant ES-EKF fusion
        with torch.no_grad():
            rot_T_true = self.rot.transpose(-1, -2)
            v_water_rel_world = self.vel - v_water_orb
            v_water_rel_body = torch.matmul(rot_T_true, v_water_rel_world.unsqueeze(-1)).squeeze(-1)

            accel_body_true = torch.matmul(
                rot_T_true,
                (self.follower_accel_prev + torch.tensor([0.0, 0.0, GRAVITY], device=self.device)).unsqueeze(-1)
            ).squeeze(-1)
            meas_accel_body = accel_body_true + self.true_ba + torch.randn_like(accel_body_true) * 0.02
            meas_gyro_body = self.omega + self.true_bw + torch.randn_like(self.omega) * 0.008

            self.ekf.propagate(meas_accel_body, meas_gyro_body)

            if (self.step_count - 1) % self.sensor_period_steps == 0:
                R_hat_prior = quat_to_rot(self.ekf.q_hat)
                v_body_est = torch.matmul(R_hat_prior.transpose(-1, -2), self.ekf.v_hat.unsqueeze(-1)).squeeze(-1)

                d_sub_hydrostatic, p_static_decoupled, _, bernoulli_bias_cm = self.perception.sample_piezoresistive_tap(
                    d_sub_physical, v_water_rel_body, v_body_est, self.rho_water_local
                )
                radar_raw, spray_dropout, radar_locked, multipath_rejected = self.perception.process_fmcw_radar_dsp(
                    h_wing, deta_dx, deta_dy, v_water_rel_body[:, 0:1], v_water_rel_body[:, 2:3]
                )
                self.last_d_sub_hydrostatic = d_sub_hydrostatic
                self.last_p_static_decoupled = p_static_decoupled
                self.last_bernoulli_bias_cm = bernoulli_bias_cm
                self.last_radar_raw = radar_raw
                self.last_radar_locked = radar_locked
                self.last_spray_dropout = spray_dropout
                self.last_multipath_rejected = multipath_rejected

                meas_radar = torch.where(radar_locked, radar_raw, self.clearance_hold)

                mag_meas_body = torch.matmul(rot_T_true, MAG_FIELD_WORLD.to(self.device).repeat(N, 1).unsqueeze(-1)).squeeze(-1) + torch.randn_like(self.omega) * 0.02
                meas_dvl_body = v_water_rel_body + torch.randn_like(v_water_rel_body) * 0.03

                celerity_est = self.dispersion_observer.omega_dom / torch.clamp(self.dispersion_observer.k_dom, min=0.1)
                h_flow_true = torch.clamp(h_wing, min=0.08)
                flow_rates_body = torch.cat([
                    -(v_water_rel_body[:, 1:2]) / h_flow_true,
                    +(v_water_rel_body[:, 0:1] - celerity_est) / h_flow_true
                ], dim=-1) + torch.randn((N, 2), device=self.device) * 0.04

                self.ekf.update_magnetometer(mag_meas_body)
                is_submerged_sensor = (d_sub_hydrostatic > 0.015)
                dvl_acoustic_lock = (~is_ventilated_overall.bool()) & (torch.rand_like(is_ventilated_overall) > 0.01)
                self.ekf.update_dvl(meas_dvl_body, is_submerged_sensor, (~dvl_acoustic_lock).float())
                self.ekf.update_optical_flow(flow_rates_body, meas_radar, celerity_est, (d_sub_hydrostatic <= 0.015))
                self.last_dvl_active = is_submerged_sensor & dvl_acoustic_lock
                self.last_flow_active = (~is_submerged_sensor) & (meas_radar < 1.8)
            else:
                d_sub_hydrostatic = self.last_d_sub_hydrostatic
                p_static_decoupled = self.last_p_static_decoupled
                bernoulli_bias_cm = self.last_bernoulli_bias_cm
                radar_raw = self.last_radar_raw
                radar_locked = self.last_radar_locked
                spray_dropout = self.last_spray_dropout
                multipath_rejected = self.last_multipath_rejected
                meas_radar = torch.where(radar_locked, radar_raw, self.clearance_hold)
                is_submerged_sensor = (d_sub_hydrostatic > 0.015)

            dvl_active = self.last_dvl_active
            flow_active = self.last_flow_active

            if self.step_count % self.gnss_period_steps == 0:
                infl_true = 1.0 + 8.0 * torch.exp(-torch.clamp(h_wing, min=0.01, max=10.0) / 0.25)
                gnss_pos = self.pos + torch.randn_like(self.pos) * 0.05 * torch.sqrt(infl_true)
                gnss_vel = self.vel + torch.randn_like(self.vel) * 0.03 * torch.sqrt(infl_true)
                self.ekf.update_gnss(gnss_pos, gnss_vel, meas_radar)

            # 5. Navigation estimates & synchronized EKF error diagnostics
            est_pos = self.ekf.p_hat
            est_vel = self.ekf.v_hat
            est_rot = quat_to_rot(self.ekf.q_hat)
            est_omega = meas_gyro_body - self.ekf.bw_hat

            grav_w = torch.tensor([0.0, 0.0, -GRAVITY], device=self.device).view(1, 3)
            a_est_world = torch.matmul(est_rot, (meas_accel_body - self.ekf.ba_hat).unsqueeze(-1)).squeeze(-1) + grav_w

            cos_tilt_est = est_rot[:, 2, 2:3].clamp(min=0.30)
            h_hydro_projected = F.relu(morph_dict["strut_length"].detach() * cos_tilt_est - d_sub_hydrostatic)
            est_clearance = torch.where(
                radar_locked,
                meas_radar,
                torch.where(is_submerged_sensor, h_hydro_projected, torch.clamp(est_pos[:, 2:3], min=0.04, max=10.0))
            )
            self.clearance_hold = est_clearance.detach().clamp(0.04, self.perception.radar_r_max)
            est_h_dot = est_vel[:, 2:3]

            ekf_err_h = est_clearance - h_wing
            ekf_err_vx = est_vel[:, 0:1] - self.vel[:, 0:1]
            R_ekf_err = torch.bmm(self.rot.transpose(-1, -2), est_rot)
            tr_ekf = R_ekf_err[:, 0, 0] + R_ekf_err[:, 1, 1] + R_ekf_err[:, 2, 2]
            ekf_err_att_deg = (torch.acos(((tr_ekf - 1.0) * 0.5).clamp(-0.99999, 0.99999)) * (180.0 / math.pi)).unsqueeze(-1)

            # 6. True 3D boresight raycast + causal EKF lookahead dispersion observer
            if (self.step_count - 1) % self.sensor_period_steps == 0:
                est_ray_w = torch.matmul(est_rot, self.waves.ray_body_vec).squeeze(-1)
                est_s = torch.clamp(est_clearance / (-est_ray_w[:, 2:3]).clamp(min=0.06), min=0.25, max=6.0)
                lookahead_dist = safe_norm(est_s * est_ray_w[:, :2], dim=-1, keepdim=True)

                eta_forward_meas, eta_forward_valid = self.waves.raycast_forward_surface_pose(
                    self.pos, self.rot, eta, self.sim_time, self.wave_curriculum_scale
                )
                eta_prev, deta_dx_prev, w_up_prev = self.dispersion_observer.update_and_predict(
                    est_clearance, est_pos[:, 2:3], eta_forward_meas, eta_forward_valid, lookahead_dist,
                    est_vel[:, 0:1], preview_time=1.25
                )
                self.last_eta_fwd_meas = eta_forward_meas
                self.last_eta_fwd_valid = eta_forward_valid
                self.last_eta_prev = eta_prev
                self.last_deta_dx_prev = deta_dx_prev
                self.last_w_up_prev = w_up_prev
                self.last_deta_dx_nadir = self.dispersion_observer.deta_dx_nadir
                self.last_w_up_nadir = self.dispersion_observer.w_up_nadir
                self.last_look_dist = lookahead_dist
            else:
                eta_forward_meas = self.last_eta_fwd_meas
                eta_prev = self.last_eta_prev
                deta_dx_prev = self.last_deta_dx_prev
                w_up_prev = self.last_w_up_prev
                lookahead_dist = self.last_look_dist

            sigma_l1 = self.l1_observer.step(a_est_world, self.a_pre_dob_prev, self.dt, sat_mask=self.l1_sat_prev)

        # 7. Discrete mode automaton
        is_loiter = (self.current_mode == MODE_LOITER)
        is_takeoff = (self.current_mode == MODE_TAKEOFF)
        is_foiling = (self.current_mode == MODE_FOILING)
        is_egress = (self.current_mode == MODE_EGRESS)
        is_flight = (self.current_mode == MODE_FREE_FLIGHT)
        is_ingress = (self.current_mode == MODE_INGRESS)
        is_soaring = (self.current_mode == MODE_SOARING)

        v_target_raw = torch.where(is_loiter, 0.0,
                       torch.where(is_takeoff, 5.20,
                       torch.where(is_flight, 8.50, self.target_speed)))

        loiter_descent_ready = is_loiter & (est_vel[:, 0:1] < 1.5)
        h_loiter_tgt = torch.where(loiter_descent_ready, 0.04, 0.18)

        is_ventilated_sensor_est = (d_sub_hydrostatic > 0.020) & (a_est_world[:, 2:3] < -1.8)

        h_foiling_cmd = torch.where(
            is_ventilated_sensor_est & is_submerged_sensor,
            torch.full_like(h_loiter_tgt, RIDE_HEIGHT_REPRIME),
            torch.full_like(h_loiter_tgt, RIDE_HEIGHT_OPTIMAL)
        )

        h_target_raw = torch.where(is_loiter, h_loiter_tgt,
                       torch.where(is_takeoff, 0.185,
                       torch.where(is_foiling, h_foiling_cmd,
                       torch.where(is_soaring, RIDE_HEIGHT_SOARING,
                       torch.where(is_egress, FREE_FLIGHT_ALTITUDE,
                       torch.where(is_flight, FREE_FLIGHT_ALTITUDE, RIDE_HEIGHT_OPTIMAL))))))

        alpha_blend = self.dt / (0.15 + self.dt)
        self.h_target_filtered = self.h_target_filtered + alpha_blend * (h_target_raw - self.h_target_filtered)
        self.v_target_filtered = self.v_target_filtered + alpha_blend * (v_target_raw - self.v_target_filtered)

        takeoff_complete = is_takeoff & (est_vel[:, 0:1] > 4.6) & is_submerged_sensor
        self.current_mode = torch.where(takeoff_complete, torch.full_like(self.current_mode, MODE_FOILING), self.current_mode)

        egress_complete = is_egress & (est_clearance > (FREE_FLIGHT_ALTITUDE - 0.15))
        self.current_mode = torch.where(egress_complete, torch.full_like(self.current_mode, MODE_FREE_FLIGHT), self.current_mode)

        ingress_complete = is_ingress & is_submerged_sensor
        self.current_mode = torch.where(ingress_complete, torch.full_like(self.current_mode, MODE_FOILING), self.current_mode)

        # 8. Neural pilot & safety filter
        e_speed = est_vel[:, 0:1] - self.v_target_filtered
        e_height = est_clearance - self.h_target_filtered
        self.e_int_height = torch.where(
            (is_foiling | is_takeoff | is_soaring),
            torch.clamp(self.e_int_height + e_height * self.dt, -0.12, 0.12),
            self.e_int_height * 0.95
        )
        y_int_active = (is_foiling | is_takeoff | is_soaring) & (est_vel[:, 0:1] > 1.0)
        self.e_int_y = torch.where(
            y_int_active,
            torch.clamp(self.e_int_y + est_pos[:, 1:2] * self.dt, -1.5, 1.5),
            self.e_int_y * 0.95
        )

        actor_obs = torch.cat([
            e_speed, e_height, self.e_int_height, est_pos[:, 1:2],
            est_h_dot, self.v_target_filtered / 8.50, d_sub_hydrostatic, self.h_target_filtered,
            est_vel, est_omega, est_rot[:, 2, :],
            eta_prev - (est_pos[:, 2:3] - est_clearance), deta_dx_prev, w_up_prev
        ], dim=-1)

        a_neural = actor_net(actor_obs)
        self.guidance_cmd_buffer.append(a_neural)
        a_guidance_delayed = self.guidance_cmd_buffer[0]
        self.a_pilot_filtered = self.pilot_filter_a * self.a_pilot_filtered + (1.0 - self.pilot_filter_a) * a_guidance_delayed

        state_err = torch.cat([
            torch.zeros((N, 1), device=self.device), est_pos[:, 1:2], e_height,
            est_vel[:, 0:1] - self.v_target_filtered, est_vel[:, 1:2], est_h_dot
        ], dim=-1)
        M_metric = ncm_net(state_err)
        M_5 = M_metric[:, 1:, 1:]

        v_wave_apparent_est = self.last_deta_dx_nadir * est_vel[:, 0:1] + self.last_w_up_nadir
        h_dot_relative_est = est_vel[:, 2:3] - v_wave_apparent_est
        h_min_mode = torch.where(loiter_descent_ready | is_takeoff, 0.01, RIDE_HEIGHT_MIN)

        b_cbf_crash = (4.00 * h_dot_relative_est + 12.0 * (est_clearance - h_min_mode)).unsqueeze(-1)
        h_max_mode = torch.where(is_egress | is_flight | is_ingress, 4.00, torch.where(is_foiling | is_soaring, RIDE_HEIGHT_FOILING_MAX, RIDE_HEIGHT_WIG_MAX))
        h_margin = h_max_mode - est_clearance
        b_cbf_ceil = (-4.00 * h_dot_relative_est + 12.0 * torch.clamp(h_margin, min=-0.5, max=1.0)).unsqueeze(-1)

        # Full Riemannian contraction constraints (including M_pv * e_p drift)
        e_xy = state_err[:, [1, 3, 4]]
        M_xy = M_5[:, [0, 2, 3], :][:, :, [0, 2, 3]]
        m_e_xy = torch.bmm(M_xy, e_xy.unsqueeze(-1)).squeeze(-1)
        g_xy = m_e_xy[:, 1:3]
        V_xy = torch.sum(e_xy * m_e_xy, dim=-1, keepdim=True)
        kin_xy = m_e_xy[:, 0:1] * e_xy[:, 2:3]
        A_ncm_xy = torch.cat([g_xy, torch.zeros((N, 1), device=self.device)], dim=-1).unsqueeze(1)
        b_ncm_xy = (-0.15 * V_xy - kin_xy).unsqueeze(-1)

        e_z = state_err[:, [2, 5]]
        M_z = M_5[:, [1, 4], :][:, :, [1, 4]]
        m_e_z = torch.bmm(M_z, e_z.unsqueeze(-1)).squeeze(-1)
        g_z = m_e_z[:, 1:2]
        V_z = torch.sum(e_z * m_e_z, dim=-1, keepdim=True)
        kin_z = m_e_z[:, 0:1] * e_z[:, 1:2]
        A_ncm_z = torch.cat([torch.zeros((N, 2), device=self.device), g_z], dim=-1).unsqueeze(1)
        b_ncm_z = (-0.15 * V_z - kin_z).unsqueeze(-1)

        A_all = torch.cat([self.A_cbf_crash, self.A_cbf_ceil, A_ncm_xy, A_ncm_z], dim=1)
        b_all = torch.cat([b_cbf_crash, b_cbf_ceil, b_ncm_xy, b_ncm_z], dim=1)

        if self.use_embedded_qp and not torch.is_grad_enabled():
            da_max = float(MAX_JERK_ACCEL * self.dt)
            a_nom_jl = torch.max(torch.min(self.a_pilot_filtered, self.a_pilot_prev + da_max), self.a_pilot_prev - da_max)
            a_safe, qp_override, active_qp_barrier_idx = self.embedded_qp.solve_tensor(a_nom_jl, A_all, b_all)
            slack = torch.zeros((N, A_all.shape[1]), device=self.device)
            active_qp_barrier_val = qp_override.clone()
            is_jerk_active = torch.zeros((N, 1), dtype=torch.bool, device=self.device)
        else:
            a_safe, qp_override, slack, active_qp_barrier_idx, active_qp_barrier_val, is_jerk_active = self.safety_filter(
                self.a_pilot_filtered, self.a_pilot_prev, A_all, b_all, self.dt
            )
        self.a_pilot_prev = a_safe

        # 9. Guidance & inner loops with 2-axis omnidirectional rotor hub gimbals (0-15 deg)
        submerged_factor_est = torch.clamp(d_sub_hydrostatic / 0.04, min=0.0, max=1.0)
        phi_max_allowed = submerged_factor_est * math.radians(6.5) + (1.0 - submerged_factor_est) * math.radians(18.0)

        v_horiz_est = safe_norm(est_vel[:, :2], dim=-1, keepdim=True).clamp(min=0.5)
        q_water_est = 0.5 * RHO_WATER_NOM * (v_horiz_est**2)
        q_air_est = 0.5 * RHO_AIR * (v_horiz_est**2)

        f_hydro_drag_est = q_water_est * morph_dict["foil_area"] * FOIL_CD0 * submerged_factor_est
        f_wing_drag_est = q_air_est * morph_dict["wing_area"] * (WING_CD0 + 0.03)
        f_drag_est = f_hydro_drag_est + f_wing_drag_est

        nom_mass = float(morph_dict["total_mass"].detach().item())

        dec_mean = math.radians(0.5 * (FOIL_DECALAGE_DEG[0] + FOIL_DECALAGE_DEG[1]))
        v_b_est = torch.matmul(est_rot.transpose(-1, -2), est_vel.unsqueeze(-1)).squeeze(-1)
        alpha_foil_est = torch.atan2(-v_b_est[:, 2:3], v_b_est[:, 0:1].clamp(min=0.5)) + dec_mean
        q_A_CLa = q_water_est * morph_dict["foil_area"] * FOIL_CL_ALPHA
        f_hydro_lift_unclamped = q_A_CLa * alpha_foil_est * submerged_factor_est

        h_est_safe = est_clearance.clamp(min=0.03)
        phi_wig_est = 1.0 + 0.35 / (0.10 + 2.2 * torch.pow(h_est_safe / morph_dict["wing_chord"], 1.15))
        alpha_wing_est = torch.atan2(-v_b_est[:, 2:3], v_b_est[:, 0:1].clamp(min=0.5)) + 0.03
        cl_wing_est = torch.clamp((WING_CL0 + 4.5 * alpha_wing_est) * phi_wig_est, -0.3, 1.4)
        f_wing_lift_est = (q_air_est * morph_dict["wing_area"] * cl_wing_est).clamp(min=0.0, max=15.0)

        tau_hydro_pitch_est = f_hydro_drag_est * morph_dict["strut_length"]

        min_rotor_floor = torch.where(is_soaring, torch.full((N, 1), 5.6, device=self.device), torch.full((N, 1), 6.8, device=self.device))
        desired_fz_total = nom_mass * (GRAVITY + a_safe[:, 2:3] - sigma_l1[:, 2:3])
        lift_cap = desired_fz_total - min_rotor_floor - f_wing_lift_est
        over_lift_est = F.relu(f_hydro_lift_unclamped - lift_cap)
        q_A_CLd = q_water_est * morph_dict["foil_area"] * HYDRO_CL_DELTA * submerged_factor_est
        coll_max = 0.8 * MAX_HYDRO_DEFLECTION
        foil_active_gate = (submerged_factor_est > 0.05).float() * (est_vel[:, 0:1] > 1.0).float()
        delta_collective = -torch.clamp(over_lift_est / (q_A_CLd + 1.0), max=coll_max) * foil_active_gate
        over_after_collective = F.relu(over_lift_est - q_A_CLd * coll_max)
        theta_shed_lift = torch.clamp(over_after_collective / (q_A_CLa + 50.0), max=math.radians(6.0)) * foil_active_gate

        f_hydro_lift_est = torch.clamp(
            f_hydro_lift_unclamped + q_A_CLd * delta_collective,
            min=-1.0, max=2.0 * FOIL_LIFT_CLAMP_PER_SIDE
        )

        raw_f_rotors = desired_fz_total - f_hydro_lift_est - f_wing_lift_est
        f_rotors_needed = torch.maximum(raw_f_rotors, min_rotor_floor).clamp(max=float(nom_mass * GRAVITY + 6.0))

        psi_est_rad = torch.atan2(est_rot[:, 1:2, 0], est_rot[:, 0:1, 0])
        c_psi, s_psi = torch.cos(psi_est_rad), torch.sin(psi_est_rad)
        ax_w = a_safe[:, 0:1] - sigma_l1[:, 0:1]
        ay_w = a_safe[:, 1:2] - 1.4 * self.e_int_y - sigma_l1[:, 1:2]
        Fx_dyn = nom_mass * (c_psi * ax_w + s_psi * ay_w)
        Fy_dyn = nom_mass * (-s_psi * ax_w + c_psi * ay_w)

        f_gimbal_horiz_cap = f_rotors_needed * (ROTOR_MAX_TILT_TAN * 0.70)
        Fx_gimbal_est = torch.where(is_soaring, torch.zeros_like(Fx_dyn), torch.clamp(Fx_dyn, -f_gimbal_horiz_cap, f_gimbal_horiz_cap))

        soar_prop_req = torch.where(
            is_soaring & is_submerged_sensor,
            torch.clamp(Fx_dyn + f_drag_est, min=0.0, max=22.0),
            torch.zeros_like(Fx_dyn)
        )
        self.scull_throttle_cmd = (soar_prop_req / 22.0).clamp(0.0, 1.0).detach()

        pusher_disabled = is_loiter & (est_vel[:, 0:1] < 0.8)
        target_pusher = torch.where(
            pusher_disabled | is_soaring,
            torch.zeros_like(Fx_dyn),
            torch.clamp(Fx_dyn + f_drag_est - Fx_gimbal_est, min=0.0, max=MAX_PUSHER_THRUST)
        )
        Fx_cmd_gimbal = Fx_dyn + f_drag_est - target_pusher - soar_prop_req
        Fx_residual = Fx_cmd_gimbal - torch.clamp(Fx_cmd_gimbal, -f_gimbal_horiz_cap, f_gimbal_horiz_cap)

        Fy_cmd_gimbal = Fy_dyn
        Fy_residual = Fy_dyn - torch.clamp(Fy_dyn, -f_gimbal_horiz_cap, f_gimbal_horiz_cap)

        f_vertical_net = (nom_mass * GRAVITY - f_hydro_lift_est - f_wing_lift_est).clamp(min=4.0)
        raw_phi_btt = torch.atan2(-Fy_residual, f_vertical_net)
        beta_fluid = torch.atan2(est_vel[:, 1:2], est_vel[:, 0:1].clamp(min=1.0))
        lateral_damping = 0.22 * torch.clamp(beta_fluid, -math.radians(10.0), math.radians(10.0))
        phi_target = torch.minimum(torch.maximum(raw_phi_btt + lateral_damping, -phi_max_allowed), phi_max_allowed)

        alpha_phi = self.dt / (0.15 + self.dt)
        self.phi_des_filtered = self.phi_des_filtered + alpha_phi * (phi_target - self.phi_des_filtered)
        phi_des = self.phi_des_filtered

        theta_tilt_residual = torch.atan2(Fx_residual, f_vertical_net)
        theta_bias = torch.where(is_egress, -math.radians(4.5),
                     torch.where(is_ingress, -math.radians(1.5),
                     torch.zeros_like(theta_tilt_residual)))
        theta_des = torch.clamp(theta_tilt_residual + theta_bias + theta_shed_lift, -math.radians(14.0), math.radians(24.0))

        self.a_pre_dob_prev = torch.cat([
            a_safe[:, 0:1],
            a_safe[:, 1:2] - 1.4 * self.e_int_y,
            a_safe[:, 2:3]
        ], dim=-1).detach()
        sat_x = (theta_des <= math.radians(-13.8)) | (theta_des >= math.radians(23.8)) | (pusher_disabled & (Fx_dyn > f_gimbal_horiz_cap))
        sat_y = torch.abs(raw_phi_btt + lateral_damping) > phi_max_allowed
        sat_z = (raw_f_rotors < min_rotor_floor) | (raw_f_rotors > float(nom_mass * GRAVITY + 6.0))
        self.l1_sat_prev = torch.cat([sat_x, sat_y, sat_z], dim=-1).detach()

        yaw_steer_gate = ((est_vel[:, 0:1] - 0.5) / 1.5).clamp(min=0.0, max=1.0)
        psi_target = -torch.clamp(
            est_pos[:, 1:2] * 0.12 + 0.15 * est_vel[:, 1:2],
            -math.radians(5.0), math.radians(5.0)
        ) * yaw_steer_gate
        alpha_psi = self.dt / (0.45 + self.dt)
        self.psi_des_filtered = self.psi_des_filtered + alpha_psi * (psi_target - self.psi_des_filtered)
        psi_des = self.psi_des_filtered

        Rd = euler_to_rot(phi_des, theta_des, psi_des)
        R_err = torch.matmul(Rd.transpose(-1, -2), est_rot) - torch.matmul(est_rot.transpose(-1, -2), Rd)

        tau_des = torch.cat([
            -self.k_R[0, 0] * R_err[:, 2, 1:2] * 0.5 - self.k_omega[0, 0] * est_omega[:, 0:1],
            -self.k_R[0, 1] * R_err[:, 0, 2:3] * 0.5 - self.k_omega[0, 1] * est_omega[:, 1:2] - 1.00 * tau_hydro_pitch_est,
            -self.k_R[0, 2] * R_err[:, 1, 0:1] * 0.5 - self.k_omega[0, 2] * est_omega[:, 2:3]
        ], dim=-1)

        h_rotors_est = torch.clamp(est_clearance, min=0.04)
        ige_est = 1.0 / (1.0 - torch.clamp((PROP_RADIUS / (4.0 * h_rotors_est))**2, max=0.35))

        # 10. Hybrid wrench allocator (rotors + 2-axis gimbals + hydro surfaces)
        (
            target_motors, target_tilts, hydro_surfaces_cmd, headroom_min, is_clipped_min,
            is_clipped_max, torque_scale, fx_gimbal_alloc, fy_gimbal_alloc
        ) = self.robot.allocate_wrench_hybrid(
            Fx_cmd_gimbal, Fy_cmd_gimbal, f_rotors_needed, tau_des, self.v_batt, ige_est,
            d_sub_hydrostatic, est_vel[:, 0:1], is_soaring, morph_dict, delta_collective=delta_collective
        )

        max_d_delta = SERVO_RATE_LIMIT * self.dt
        self.hydro_surfaces_actual = self.hydro_surfaces_actual + torch.clamp(
            hydro_surfaces_cmd - self.hydro_surfaces_actual, -max_d_delta, max_d_delta
        )
        self.hydro_surfaces = self.hydro_surfaces_actual

        wave_slope_mag = safe_norm(torch.cat([deta_dx, deta_dy], dim=-1), dim=-1, keepdim=True)

        # 11. Integrate ground-truth 6-DoF physics
        (
            self.pos, self.vel, self.rot, self.omega, self.motors, self.rotor_tilts, self.pusher_thrust,
            self.nu_states, omega_props, accel, ige_rotor_gain
        ) = self.robot.integrate(
            self.pos, self.vel, self.rot, self.omega, self.motors, self.rotor_tilts, self.pusher_thrust,
            self.nu_states, target_motors, target_tilts, target_pusher, f_wing, f_hydro,
            tau_hydro_pitch, tau_hydro_roll, tau_hydro_yaw,
            ambient_gust * wind_shear_factor, eta, wave_slope_mag,
            self.mass, self.motor_tau_up, self.motor_tau_down
        )
        self.follower_accel_prev = accel.detach()

        eta_post = eta + deta_dt * self.dt
        clearance_eval = (self.pos[:, 2:3] - eta_post).clamp(min=0.01)
        z_tip_L_post = self.pos[:, 2:3] + self.rot[:, 2, 1:2] * (morph_dict["foil_span"] * 0.25) - self.rot[:, 2, 2:3] * morph_dict["strut_length"]
        z_tip_R_post = self.pos[:, 2:3] - self.rot[:, 2, 1:2] * (morph_dict["foil_span"] * 0.25) - self.rot[:, 2, 2:3] * morph_dict["strut_length"]
        immersion_eval = 0.5 * (
            (eta_post - z_tip_L_post).clamp(min=0.0, max=0.22)
            + (eta_post - z_tip_R_post).clamp(min=0.0, max=0.22)
        )
        phi_true_deg = torch.atan2(self.rot[:, 2, 1:2], self.rot[:, 2, 2:3]) * (180.0 / math.pi)

        p_scull_load = aoa_tensors["p_scull_actuator_watts"]
        v_air_x_body = torch.matmul(self.rot.transpose(-1, -2), self.vel.unsqueeze(-1)).squeeze(-1)[:, 0:1]
        p_elec, self.v_batt, i_batt, self.u1, self.u2, self.soc, self.t_cell = self.powertrain.step(
            self.motors, w_updraft, self.nu_states[:, :, 0], omega_props,
            self.u1, self.u2, self.soc, self.t_cell, self.dt, p_scull_load,
            pusher_thrust=self.pusher_thrust, v_air_x=v_air_x_body
        )

        if export_diagnostics:
            hover_motors = torch.full_like(self.motors, HOVER_WEIGHT_NOM / 4.0)
            omega_props_hover = torch.sqrt((hover_motors + 1e-6) / 1.1e-5).clamp(min=50.0, max=1800.0)
            p_base_isolated, _, _, _, _, _, _ = self.powertrain.step(
                hover_motors, torch.zeros_like(w_updraft),
                torch.full_like(self.nu_states[:, :, 0], V_INDUCED_HOVER), omega_props_hover,
                torch.zeros_like(self.u1), torch.zeros_like(self.u2), torch.full_like(self.soc, 0.95),
                torch.full_like(self.t_cell, T_AMBIENT_K), self.dt, torch.zeros_like(p_scull_load),
                pusher_thrust=torch.zeros_like(self.pusher_thrust), v_air_x=torch.zeros_like(v_air_x_body)
            )
            power_saved_pct = ((p_base_isolated - p_elec) / p_base_isolated) * 100.0
        else:
            power_saved_pct = torch.zeros((N, 1), device=self.device)

        tilt_sin_mag = torch.sqrt(self.rotor_tilts[:, :, 0]**2 + self.rotor_tilts[:, :, 1]**2 + 1e-10).clamp(max=1.0)
        rotor_tilt_deg = torch.asin(tilt_sin_mag) * (180.0 / math.pi)

        self.metrics = {
            "mode": self.current_mode,
            "v_target_mode": self.v_target_filtered,
            "h_target_mode": self.h_target_filtered,
            "plant_damp_diag": aoa_tensors["plant_damp_diag"],
            "phi_true_deg": phi_true_deg,
            "ekf_err_h": ekf_err_h,
            "ekf_err_vx": ekf_err_vx,
            "ekf_err_att_deg": ekf_err_att_deg,
            "sigma_l1_vec": sigma_l1,
            "e_int_y": self.e_int_y,
            "fx_gimbal": fx_gimbal_alloc,
            "fy_gimbal": fy_gimbal_alloc,
            "clearance_eval": clearance_eval,
            "immersion_eval": immersion_eval,
            "f_hydro_lift_eval": f_hydro[:, 2:3],
            "vel_eval": self.vel,
            "pos_eval": self.pos,
            "omega_eval": self.omega,
            "h_wing": h_wing,
            "clearance_est": est_clearance,
            "immersion_depth": d_sub_hydrostatic,
            "f_slam": f_slam,
            "f_buoy": f_buoy,
            "f_hydro_lift": f_hydro[:, 2:3],
            "f_wing_lift": f_wing[:, 2:3],
            "f_rotors": torch.sum(self.motors, dim=-1, keepdim=True),
            "f_pusher": self.pusher_thrust,
            "motors": self.motors,
            "rotor_tilts": self.rotor_tilts,
            "rotor_tilt_deg": rotor_tilt_deg,
            "hydro_surfaces": self.hydro_surfaces,
            "hydro_deflections": (
                math.degrees(self.hydro_surfaces[0, 0].item()),
                math.degrees(self.hydro_surfaces[0, 1].item()),
                math.degrees(self.hydro_surfaces[0, 2].item())
            ),
            "cell_temp_c": self.t_cell[0, 0].item() - 273.15,
            "headroom_min": headroom_min,
            "is_clipped_min": is_clipped_min,
            "is_clipped_max": is_clipped_max,
            "power_saved_pct": power_saved_pct,
            "p_elec": p_elec,
            "v_batt": self.v_batt,
            "i_batt": i_batt,
            "soc": self.soc * 100.0,
            "qp_override": qp_override,
            "max_slack": slack,
            "actor_obs": actor_obs,
            "a_safe": a_safe,
            "a_neural": a_neural,
            "state_err": state_err,
            "pos": self.pos,
            "vel": self.vel,
            "eta": eta,
            "omega": self.omega,
            "e_R": torch.cat([R_err[:, 2, 1:2] * 0.5, R_err[:, 0, 2:3] * 0.5, R_err[:, 1, 0:1] * 0.5], dim=-1),
            "e_int_height": self.e_int_height,
            "is_vent_L": self.is_vent_L,
            "is_vent_R": self.is_vent_R,
            "is_ventilated": is_ventilated_overall[0, 0].item() > 0.5 if export_diagnostics else False,
            "h_dot_est": est_h_dot,
            "radar_dropout": spray_dropout[0, 0].item() > 0.5 if export_diagnostics else False,
            "eta_preview": eta_prev,
            "eta_forward_meas": eta_forward_meas,
            "eta_nadir_est": self.dispersion_observer.last_eta_nadir,
            "look_dist": lookahead_dist,
            "k_dom": self.dispersion_observer.k_dom,
            "omega_dom": self.dispersion_observer.omega_dom,
            "sigma_l1_norm": safe_norm(sigma_l1, dim=-1, keepdim=True),
            "strut_moment_nm": strut_root_moment,
            "radar_locked": bool(radar_locked[0, 0].item() > 0.5) if export_diagnostics else True,
            "multipath_rejected": bool(multipath_rejected[0, 0].item() > 0.5) if export_diagnostics else False,
            "delta_collective_deg": math.degrees(delta_collective[0, 0].item()),
            "foil_lift_est": f_hydro_lift_est,
            "status_banner": "[NOMINAL]"
        }

        if export_diagnostics:
            phi_deg = math.degrees(math.atan2(est_rot[0, 2, 1].item(), est_rot[0, 2, 2].item()))
            theta_deg = math.degrees(math.asin(float(np.clip(-est_rot[0, 2, 0].item(), -0.9999, 0.9999))))
            psi_actual_deg = math.degrees(math.atan2(est_rot[0, 1, 0].item(), est_rot[0, 0, 0].item()))
            psi_des_deg = math.degrees(psi_des[0, 0].item())
            psi_err_deg = psi_actual_deg - psi_des_deg
            beta_sideslip_deg = math.degrees(math.atan2(est_vel[0, 1].item(), max(0.1, est_vel[0, 0].item())))

            k_val = max(0.01, self.dispersion_observer.k_dom[0, 0].item())
            wavelength_m = (2.0 * math.pi) / k_val
            celerity_mps = self.dispersion_observer.omega_dom[0, 0].item() / k_val

            min_clipped = is_clipped_min[0].cpu().numpy()
            max_clipped = is_clipped_max[0].cpu().numpy()
            if any(min_clipped) and any(max_clipped):
                sat_str = "[MIN+MAX CLIPPED]"
            elif any(min_clipped):
                sat_str = "[MIN FLOOR]"
            elif any(max_clipped):
                sat_str = "[MAX CEIL]"
            else:
                sat_str = "[CLEAN]"

            torque_auth_pct = torque_scale[0, 0].item() * 100.0
            cell_temp_c = self.t_cell[0, 0].item() - 273.15
            i_batt_val = i_batt[0, 0].item()
            r0_val = BATTERY_R0_NOM * math.exp(ARRHENIUS_COEFF * (1.0 / max(270.0, self.t_cell[0, 0].item()) - 1.0 / T_AMBIENT_K))
            v_sag = i_batt_val * r0_val

            battery_ah = BATTERY_CAPACITY_COULOMBS / 3600.0
            rem_wh = max(0.0, self.soc[0, 0].item() - 0.05) * battery_ah * max(20.0, self.v_batt[0, 0].item())
            p_elec_w = max(1.0, p_elec[0, 0].item())
            endurance_min = (rem_wh / p_elec_w) * 60.0
            vx_kmh = max(0.2, self.vel[0, 0].item() * 3.6)
            wh_per_km = p_elec_w / vx_kmh

            with torch.no_grad():
                nn_raw_vec = torch.tanh(actor_net.net(actor_obs[0:1]))[0].cpu().numpy()

            mode_names = ["LOITER", "TAKEOFF", "FOILING", "EGRESS", "FLIGHT", "INGRESS", "SOARING"]
            current_mode_name = mode_names[self.current_mode[0, 0].item()]

            d_eL_deg = math.degrees(self.hydro_surfaces[0, 0].item())
            d_eR_deg = math.degrees(self.hydro_surfaces[0, 1].item())
            d_rud_deg = math.degrees(self.hydro_surfaces[0, 2].item())

            q_water_val = max(1.0, aoa_tensors["q_water"][0, 0].item())
            p_static_val = max(100.0, p_static_decoupled[0, 0].item())
            alpha_foil_val = aoa_tensors["alpha_foil"][0, 0].item()
            peak_suction = 2.4 * max(0.0, alpha_foil_val)
            cavitation_sigma = (p_static_val - P_VAPOR_WATER) / (q_water_val * (1.0 + peak_suction))
            froude_fn = est_vel[0, 0].item() / math.sqrt(GRAVITY * HULL_LENGTH)
            stall_margin_deg = math.degrees(math.radians(12.5) - abs(alpha_foil_val))

            sigma_z_cm = math.sqrt(max(1e-6, self.ekf.P[0, 2, 2].item())) * 100.0
            bias_omega_deg = tuple(math.degrees(b) for b in self.ekf.bw_hat[0].cpu().numpy())

            if qp_override[0, 0].item() < 0.10:
                barrier_name = "[FREE (Unconstrained)]"
            elif active_qp_barrier_idx[0, 0].item() == 0:
                barrier_name = "[CRASH BARRIER (Floor)]"
            elif active_qp_barrier_idx[0, 0].item() == 1:
                barrier_name = "[BREACH BARRIER (Ceiling)]"
            else:
                barrier_name = "[NCM CONTRACTION METRIC]"

            f_soar_val = aoa_tensors["f_soar_surge"][0, 0].item()
            f_scull_val = aoa_tensors["f_scull_thrust"][0, 0].item()
            f_par_val = aoa_tensors["f_par_z"][0, 0].item()
            st_val = aoa_tensors["strouhal_st"][0, 0].item()

            if is_soaring[0, 0].item() and (f_soar_val + f_scull_val) > 1.0:
                status_banner = "[WAVE SOARING + SCULLING]"
            elif abs(phi_deg) > 28.0:
                status_banner = "[CAPSIZE RISK - HIGH ROLL]"
            elif est_clearance[0, 0].item() < HULL_DRAFT_MAX + 0.01:
                status_banner = "[HULL CONTACT - PLANING]"
            elif f_hydro[0, 2].item() > 8.0:
                status_banner = "[OPTIMAL WAVE FOILING]"
            else:
                status_banner = "[AIRBORNE FLIGHT]"

            self.metrics.update({
                "status_banner": status_banner,
                "psi_des_deg": psi_des_deg,
                "psi_err_deg": psi_err_deg,
                "beta_sideslip_deg": beta_sideslip_deg,
                "wavelength_m": wavelength_m,
                "celerity_mps": celerity_mps,
                "f_slam_z": f_slam[0, 0].item(),
                "f_planing_z": (f_buoy[0, 0] + aoa_tensors["f_planing_z"][0, 0]).item(),
                "sat_str": sat_str,
                "torque_auth_pct": torque_auth_pct,
                "cell_temp_c": cell_temp_c,
                "v_sag": v_sag,
                "endurance_min": endurance_min,
                "wh_per_km": wh_per_km,
                "nn_raw": tuple(float(v) for v in nn_raw_vec),
                "sigma_l1_xyz": tuple(float(v) for v in sigma_l1[0].cpu().numpy()),
                "ba_hat_0": tuple(float(v) for v in self.ekf.ba_hat[0].cpu().numpy()),
                "a_neural_0": tuple(float(v) for v in a_neural[0].cpu().numpy()),
                "a_safe_0": tuple(float(v) for v in a_safe[0].cpu().numpy()),
                "e_int_hy": (self.e_int_height[0, 0].item(), self.e_int_y[0, 0].item()),
                "gimbal_fxy": (fx_gimbal_alloc[0, 0].item(), fy_gimbal_alloc[0, 0].item()),
                "ekf_err_0": (ekf_err_h[0, 0].item() * 100.0, ekf_err_vx[0, 0].item(), ekf_err_att_deg[0, 0].item()),
                "bernoulli_bias_cm_0": bernoulli_bias_cm[0, 0].item(),
                "sensor_locks": (bool(dvl_active[0, 0].item()), bool(flow_active[0, 0].item())),
                "targets_0": (self.h_target_filtered[0, 0].item(), self.v_target_filtered[0, 0].item()),
                "omega_deg_0": tuple(math.degrees(w) for w in self.omega[0].cpu().numpy()),
                "euler_deg": (phi_deg, theta_deg, psi_actual_deg),
                "phi_btt_deg": math.degrees(phi_des[0, 0].item()),
                "mode_name": current_mode_name,
                "hydro_deflections": (d_eL_deg, d_eR_deg, d_rud_deg),
                "margin_ceil_cm": (h_max_mode[0, 0].item() - est_clearance[0, 0].item()) * 100.0,
                "margin_floor_cm": (est_clearance[0, 0].item() - h_min_mode[0, 0].item()) * 100.0,
                "alpha_total_deg": math.degrees(alpha_foil_val),
                "theta_shed_deg": math.degrees(theta_shed_lift[0, 0].item()),
                "cavitation_sigma": cavitation_sigma,
                "froude_fn": froude_fn,
                "stall_margin_deg": stall_margin_deg,
                "sigma_z_cm": sigma_z_cm,
                "bias_omega_deg": bias_omega_deg,
                "barrier_name": barrier_name,
                "f_soar_surge": f_soar_val,
                "f_scull_thrust": f_scull_val,
                "f_par_z": f_par_val,
                "strouhal_st": st_val,
                "morph_span": morph_dict["wing_span"].item(),
                "morph_ar": morph_dict["aspect_ratio"].item(),
                "morph_mass": morph_dict["total_mass"].item(),
                "morph_anhedral": math.degrees(morph_dict["anhedral_rad"].item())
            })

        if not torch.is_grad_enabled():
            self.detach_states()

        return self.metrics

# =============================================================================
# 13. STAGE 0 PRE-TRAINING ENGINE (RICCATI METRIC & PHYSICAL TRIM PRIOR)
# =============================================================================
def pretrain_aeromarine_stack(actor_net, ncm_net, device=DEVICE):
    print("\n" + "=" * 114)
    print(" [STAGE 0 PRE-TRAINING] CO-SYNTHESIS OF RICCATI RIEMANNIAN METRIC & PHYSICAL TRIM PRIOR")
    print("=" * 114)

    M_xy_nom = torch.tensor([
        [3.5, 0.0, 0.7],
        [0.0, 2.5, 0.0],
        [0.7, 0.0, 1.0],
    ], device=device).unsqueeze(0).repeat(512, 1, 1)
    M_z_nom = torch.tensor([
        [7.0, 0.8],
        [0.8, 1.0],
    ], device=device).unsqueeze(0).repeat(512, 1, 1)

    ncm_pre_opt = torch.optim.Adam(ncm_net.parameters(), lr=4.0e-3)
    for _ in range(300):
        dummy_state = torch.randn((512, 6), device=device) * 0.15
        dummy_state[:, 0] = 0.0
        M_pred = ncm_net(dummy_state)
        M_xy_pred = M_pred[:, [1, 3, 4], :][:, :, [1, 3, 4]]
        M_z_pred = M_pred[:, [2, 5], :][:, :, [2, 5]]
        loss_val = F.mse_loss(M_xy_pred, M_xy_nom) + F.mse_loss(M_z_pred, M_z_nom)
        perturb = torch.randn_like(dummy_state) * 0.02
        perturb[:, 0] = 0.0
        M_pert = ncm_net(dummy_state + perturb)
        loss_flat = F.mse_loss(M_pert, M_pred) * 2.0
        ncm_pre_opt.zero_grad()
        (loss_val + loss_flat).backward()
        ncm_pre_opt.step()
    print("  ├─ 1. RIEMANNIAN METRIC PRE-TRAIN: Stabilizing block-decoupled 5D Riccati metric.")

    actor_pre_opt = torch.optim.Adam(actor_net.parameters(), lr=3.0e-3)
    batch_size = 1024
    for _ in range(300):
        e_speed = torch.empty((batch_size, 1), device=device).uniform_(-3.0, 3.0)
        e_height = torch.empty((batch_size, 1), device=device).uniform_(-0.25, 0.50)
        e_int_h = torch.empty((batch_size, 1), device=device).uniform_(-0.08, 0.08)
        e_y = torch.empty((batch_size, 1), device=device).uniform_(-1.00, 1.00)
        h_dot = torch.empty((batch_size, 1), device=device).uniform_(-0.60, 0.60)
        v_tgt_norm = torch.empty((batch_size, 1), device=device).uniform_(0.0, 1.0)
        h_tgt_m = torch.empty((batch_size, 1), device=device).uniform_(0.04, 1.50)
        d_sub = torch.clamp(0.065 - e_height, min=0.0, max=0.15)
        vel = torch.cat([
            (v_tgt_norm * 8.50 + e_speed).clamp(min=0.0, max=12.0),
            torch.empty((batch_size, 1), device=device).uniform_(-0.50, 0.50),
            h_dot
        ], dim=-1)
        omega = torch.randn((batch_size, 3), device=device) * 0.15
        rot_r2 = torch.cat([torch.zeros((batch_size, 2), device=device), torch.ones((batch_size, 1), device=device)], dim=-1)
        eta_prev_err = torch.empty((batch_size, 1), device=device).uniform_(-0.12, 0.12)
        deta_dx_prev = torch.empty((batch_size, 1), device=device).uniform_(-0.18, 0.18)
        w_up_prev = torch.empty((batch_size, 1), device=device).uniform_(-0.3, 0.3)

        obs_batch = torch.cat([
            e_speed, e_height, e_int_h, e_y, h_dot, v_tgt_norm,
            d_sub, h_tgt_m, vel, omega, rot_r2,
            eta_prev_err, deta_dx_prev, w_up_prev
        ], dim=-1)

        target_ax = -1.8 * e_speed
        target_ay = -2.2 * vel[:, 1:2] - 2.8 * e_y
        a_ff_wave = +vel[:, 0:1] * 1.4 * deta_dx_prev * (d_sub > 0.01).float()
        target_az = a_ff_wave - 9.5 * e_height - 3.5 * e_int_h - 4.2 * h_dot
        target_actions = torch.cat([target_ax, target_ay, target_az], dim=-1)

        loss_actor_bc = F.mse_loss(actor_net(obs_batch), target_actions)
        actor_pre_opt.zero_grad()
        loss_actor_bc.backward()
        torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=1.0)
        actor_pre_opt.step()
    print("  └─ 2. ACTOR TRIM PRIOR PRE-TRAIN: Multi-mode & upset trim prior conditioned.\n")


# =============================================================================
# 14. MULTI-STAGE BI-LEVEL MORPHOGENETIC TRAINING ENGINE
# =============================================================================
def train_wave_riding_pilot(epochs=96, num_envs=2048, rollout_steps=200, tbptt_steps=10):
    import warnings

    print("\n" + "=" * 114)
    print(" TRAINING MULTIMODAL AEROMARINE CONTROLLER WITH BI-LEVEL MORPHOGENETIC EVOLUTION")
    print(f" Envs: {num_envs:,} | Steps: {rollout_steps} | TBTT: {tbptt_steps} | All 7 Modes + Upsets | Multi-Step Plant-Aware NCM")
    print("=" * 114)

    env = WaveRidingAutonomyEnv(num_envs=num_envs, dt=0.015, device=DEVICE, use_embedded_qp=False)
    actor_net = FullNeuralWavePilot(state_dim=20, action_dim=3).to(DEVICE)
    ncm_net = NeuralContractionMetric(state_dim=6, alpha=0.10).to(DEVICE)

    pretrain_aeromarine_stack(actor_net, ncm_net, device=DEVICE)

    optimizer_actor = torch.optim.AdamW(actor_net.parameters(), lr=1.5e-3, weight_decay=1e-5)
    optimizer_ncm = torch.optim.AdamW(ncm_net.parameters(), lr=2.5e-3, weight_decay=1e-5)
    optimizer_morph = torch.optim.Adam(env.morphology.parameters(), lr=8.0e-4)

    ep_s1 = max(1, (epochs * 28) // 96)
    ep_s2 = max(ep_s1 + 1, (epochs * 56) // 96)

    scheduler_actor = make_cosine(optimizer_actor, 1.5e-3, ep_s1)
    scheduler_ncm = make_cosine(optimizer_ncm, 2.5e-3, ep_s1)

    scaler = make_grad_scaler(DEVICE, enabled=False)
    device_str = "cuda" if DEVICE.type == "cuda" else "cpu"

    ncm_sample_count = min(256, num_envs)
    delta_v, delta_p = 0.03, 0.03
    pert_mask = torch.zeros((10, 1, 20), device=DEVICE)
    pert_mask[0, 0, 0] = +delta_v; pert_mask[0, 0, 8] = +delta_v
    pert_mask[1, 0, 0] = -delta_v; pert_mask[1, 0, 8] = -delta_v
    pert_mask[2, 0, 9] = +delta_v
    pert_mask[3, 0, 9] = -delta_v
    pert_mask[4, 0, 4] = +delta_v; pert_mask[4, 0, 10] = +delta_v
    pert_mask[5, 0, 4] = -delta_v; pert_mask[5, 0, 10] = -delta_v
    pert_mask[6, 0, 3] = +delta_p
    pert_mask[7, 0, 3] = -delta_p
    pert_mask[8, 0, 1] = +delta_p
    pert_mask[9, 0, 1] = -delta_p

    kin_top_base = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ], device=DEVICE)
    diag_jitter_5 = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=DEVICE) * 1e-4).unsqueeze(0)
    mode_short_names = ["LOIT", "TAKE", "FOIL", "EGRS", "FFLT", "INGR", "SOAR"]

    for epoch in range(epochs):
        t_ep_start = time.perf_counter()
        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(DEVICE)

        optimizer_actor.zero_grad()
        optimizer_ncm.zero_grad()
        optimizer_morph.zero_grad()

        if epoch == ep_s1:
            scheduler_actor = make_cosine(optimizer_actor, 1.0e-3, max(1, ep_s2 - ep_s1))
            scheduler_ncm = make_cosine(optimizer_ncm, 2.0e-3, max(1, ep_s2 - ep_s1))
        elif epoch == ep_s2:
            scheduler_actor = make_cosine(optimizer_actor, 8.0e-4, max(1, epochs - ep_s2))
            scheduler_ncm = make_cosine(optimizer_ncm, 1.5e-3, max(1, epochs - ep_s2))

        if epoch < ep_s1:
            wave_scale = 0.0; alpha_t = 0.05
            curriculum_stage = "Stage 1 (Flat Water Multi-Mode)"
        elif epoch < ep_s2:
            wave_scale = 0.4; alpha_t = 0.04
            curriculum_stage = "Stage 2 (Gentle Swells + Upsets)"
        else:
            wave_scale = 1.0; alpha_t = 0.025
            curriculum_stage = "Stage 3 (Full Ocean Swells + Upsets)"

        env.set_wave_curriculum(wave_scale)
        env.reset(mode_mix=True)

        c_h_acc = c_ceil_acc = c_def_acc = c_spd_acc = c_drift_acc = 0.0
        c_omega_acc = c_clip_acc = c_vent_acc = c_slack_acc = c_stress_acc = 0.0
        foil_slam_sum = foil_breach_sum = foil_total_count = 0.0
        capsize_acc = clip_acc = 0.0
        nan_encountered = False

        ncm_err_samples = []
        ncm_obs_samples = []
        ncm_asafe_samples = []
        ncm_damp_samples = []

        with get_amp_autocast(device_str, enabled=False):
            window_loss = torch.zeros(1, device=DEVICE)

            for step_idx in range(rollout_steps):
                if step_idx == 80 and epoch >= max(1, ep_s1 // 2):
                    with torch.no_grad():
                        upset_mask = (torch.rand((num_envs, 1), device=DEVICE) < 0.50).float()
                        env.vel = (env.vel + upset_mask * (torch.randn_like(env.vel) * torch.tensor([0.7, 0.9, 0.45], device=DEVICE))).clamp(-15.0, 15.0)
                        env.omega = (env.omega + upset_mask * (torch.randn_like(env.omega) * 0.65)).clamp(-6.0, 6.0)
                elif step_idx == 100:
                    with torch.no_grad():
                        sw_mask = (torch.rand((num_envs, 1), device=DEVICE) < 0.25)
                        high_alt = (env.current_mode == MODE_EGRESS) | (env.current_mode == MODE_FREE_FLIGHT)
                        rand_modes = torch.randint(0, 7, (num_envs, 1), device=DEVICE)
                        target_sw = torch.where(high_alt, torch.full_like(rand_modes, MODE_INGRESS), rand_modes)
                        env.current_mode = torch.where(sw_mask, target_sw, env.current_mode)

                m = env.step(actor_net, ncm_net, export_diagnostics=False)

                is_foil_bool = (m["mode"] == MODE_FOILING) | (m["mode"] == MODE_SOARING)
                is_foil_mode = is_foil_bool.float()
                is_air_mode = ((m["mode"] == MODE_EGRESS) | (m["mode"] == MODE_FREE_FLIGHT) | (m["mode"] == MODE_INGRESS)).float()

                h_lower_gate = torch.where(
                    (m["immersion_eval"] < 0.040),
                    torch.full_like(m["clearance_eval"], 0.145),
                    torch.full_like(m["clearance_eval"], 0.160)
                )
                h_eval_safe = torch.clamp(m["clearance_eval"], min=0.01, max=2.50)
                low_altitude_penalty = (
                    torch.mean((F.relu(h_lower_gate - h_eval_safe)**2) * is_foil_mode) * 2000.0
                    + torch.mean((F.relu(0.12 - h_eval_safe)**2) * is_air_mode) * 1000.0
                )
                high_altitude_penalty = torch.mean((F.relu(h_eval_safe - 0.215)**2) * is_foil_mode) * 2000.0
                e_h = torch.clamp(h_eval_safe - m["h_target_mode"], min=-0.8, max=0.8)
                loss_h_base = torch.mean(e_h**2) * 350.0 + torch.mean(m["e_int_height"]**2) * 80.0
                loss_height = loss_h_base + high_altitude_penalty + low_altitude_penalty

                loss_airborne = (
                    torch.mean(F.relu(0.018 - m["immersion_eval"]) * is_foil_mode) * 150.0
                    + torch.mean((F.relu(0.008 - m["immersion_eval"])**2) * is_foil_mode) * 1500.0
                )
                loss_deep_hull = torch.mean((F.relu(m["immersion_eval"] - 0.090)**2) * is_foil_mode) * 2000.0
                loss_foil_lift = torch.mean(F.relu(11.5 - m["f_hydro_lift_eval"]) * is_foil_mode) * 1.5

                loss_speed = torch.mean((m["vel_eval"][:, 0:1] - m["v_target_mode"])**2) * 2.00
                loss_drift = torch.mean(m["vel_eval"][:, 1:2]**2) * 1.50 + torch.mean(m["pos_eval"][:, 1:2]**2) * 3.00
                loss_strut_stress = torch.mean(m["strut_moment_nm"]) * 0.05
                loss_vent = torch.mean((m["is_vent_L"].float() + m["is_vent_R"].float()) * is_foil_mode) * 8.0
                loss_clipping = torch.mean(F.relu(0.0 - m["headroom_min"])**2) * 15.0
                loss_omega = torch.mean(m["omega_eval"]**2) * 0.30
                loss_safety_slack = torch.mean(m["max_slack"]) * 5.0
                loss_power = torch.mean(m["p_elec"]) * 0.002
                loss_residual = torch.mean(torch.tanh(actor_net.net(m["actor_obs"]))**2) * 0.40

                step_cost = (
                    loss_height + loss_airborne + loss_deep_hull + loss_foil_lift + loss_speed +
                    loss_drift + loss_strut_stress + loss_vent + loss_clipping + loss_omega +
                    loss_safety_slack + loss_power + loss_residual
                )

                c_h_acc += loss_h_base.item()
                c_ceil_acc += (high_altitude_penalty + low_altitude_penalty).item()
                c_def_acc += (loss_airborne + loss_deep_hull + loss_foil_lift).item()
                c_spd_acc += loss_speed.item()
                c_drift_acc += loss_drift.item()
                c_omega_acc += loss_omega.item()
                c_clip_acc += loss_clipping.item()
                c_vent_acc += loss_vent.item()
                c_slack_acc += loss_safety_slack.item()
                c_stress_acc += loss_strut_stress.item()

                with torch.no_grad():
                    foil_cnt = is_foil_mode.sum().item()
                    if foil_cnt > 0:
                        foil_slam_sum += ((m["clearance_eval"] < 0.080) & is_foil_bool).float().sum().item()
                        foil_breach_sum += ((m["clearance_eval"] > 0.235) & is_foil_bool).float().sum().item()
                        foil_total_count += foil_cnt
                    capsize_acc += (m["phi_true_deg"].abs() > 25.0).float().mean().item()
                    clip_acc += (m["is_clipped_min"].any(dim=-1) | m["is_clipped_max"].any(dim=-1)).float().mean().item()

                window_loss = window_loss + step_cost / rollout_steps

                if (step_idx + 1) % 20 == 0:
                    perm_idx = torch.randperm(num_envs, device=DEVICE)[:ncm_sample_count]
                    ncm_err_samples.append(m["state_err"][perm_idx].detach().float())
                    ncm_obs_samples.append(m["actor_obs"][perm_idx].detach().float())
                    ncm_asafe_samples.append(m["a_safe"][perm_idx].detach().float())
                    ncm_damp_samples.append(m["plant_damp_diag"][perm_idx].detach().float())

                if ((step_idx + 1) % tbptt_steps == 0) or (step_idx == rollout_steps - 1):
                    if torch.isnan(window_loss).any() or torch.isinf(window_loss).any():
                        nan_encountered = True
                        break
                    scaler.scale(window_loss).backward()
                    env.detach_states()
                    window_loss = torch.zeros(1, device=DEVICE)

            if nan_encountered:
                print(f">> [WARN] NaN/Inf loss encountered at Epoch {epoch+1}. Skipping optimizer step.")
                optimizer_actor.zero_grad()
                optimizer_ncm.zero_grad()
                optimizer_morph.zero_grad()
                env.detach_states()
                continue

            c_h_scalar = c_h_acc / rollout_steps
            c_ceil_scalar = c_ceil_acc / rollout_steps
            c_def_scalar = c_def_acc / rollout_steps
            c_spd_scalar = c_spd_acc / rollout_steps
            c_drift_scalar = c_drift_acc / rollout_steps
            c_omega_scalar = c_omega_acc / rollout_steps
            c_clip_scalar = c_clip_acc / rollout_steps
            c_vent_scalar = c_vent_acc / rollout_steps
            c_slack_scalar = c_slack_acc / rollout_steps
            c_stress_scalar = c_stress_acc / rollout_steps

            state_err_s = torch.cat(ncm_err_samples, dim=0)
            obs_base_s = torch.cat(ncm_obs_samples, dim=0)
            a_flight_s = torch.cat(ncm_asafe_samples, dim=0)
            damp_diag_s = torch.cat(ncm_damp_samples, dim=0)
            M_batch = state_err_s.shape[0]

            with torch.no_grad():
                stacked_obs = (obs_base_s.unsqueeze(0) + pert_mask).view(-1, 20)
            stacked_actions = actor_net(stacked_obs)
            splits = torch.split(stacked_actions, M_batch, dim=0)
            J_v = torch.stack([(splits[2*k] - splits[2*k+1]) / (2.0 * delta_v) for k in range(3)], dim=-1)
            J_p_yz = torch.stack([(splits[6 + 2*k] - splits[6 + 2*k+1]) / (2.0 * delta_p) for k in range(2)], dim=-1)
            b_eff = 0.85
            kin_top_s = kin_top_base.unsqueeze(0).repeat(M_batch, 1, 1)
            A_vv = torch.diag_embed(damp_diag_s) + b_eff * J_v
            row_bot_5 = torch.cat([b_eff * J_p_yz, A_vv], dim=-1)
            A_cl_5 = torch.cat([kin_top_s, row_bot_5], dim=-2)

            M_full = ncm_net(state_err_s)
            M_5 = M_full[:, 1:, 1:]

            e_v_s = state_err_s[:, 3:6]
            e_dot_s = torch.cat([torch.zeros_like(e_v_s[:, 0:1]), e_v_s[:, 1:3], a_flight_s + damp_diag_s * e_v_s], dim=-1)

            eps_lie = 2.0e-3
            M_plus = ncm_net(state_err_s + eps_lie * e_dot_s)[:, 1:, 1:]
            M_minus = ncm_net(state_err_s - eps_lie * e_dot_s)[:, 1:, 1:]
            M_dot_5 = torch.clamp((M_plus - M_minus) / (2.0 * eps_lie), -25.0, 25.0)

            R_5 = M_dot_5 + torch.matmul(A_cl_5.transpose(-1, -2), M_5) + torch.matmul(M_5, A_cl_5) + 2.0 * alpha_t * M_5
            R_sym = 0.5 * (R_5 + R_5.transpose(-1, -2))
            R_reg = torch.nan_to_num(R_sym + diag_jitter_5, nan=0.0, posinf=1e3, neginf=-1e3)

            _, eigvecs_R = torch.linalg.eigh(R_reg)
            v_det = eigvecs_R.detach()
            eigvals_R = torch.sum(v_det * torch.matmul(R_reg, v_det), dim=-2)
            violation_all = F.relu(eigvals_R + 0.015)
            loss_viol_all = F.smooth_l1_loss(violation_all, torch.zeros_like(violation_all), beta=0.20, reduction='mean') * 10.0

            M_sym = 0.5 * (M_5 + M_5.transpose(-1, -2))
            M_reg = torch.nan_to_num(M_sym + diag_jitter_5, nan=0.0, posinf=1e3, neginf=-1e3)
            _, eigvecs_M = torch.linalg.eigh(M_reg)
            v_M_det = eigvecs_M.detach()
            eigvals_M = torch.sum(v_M_det * torch.matmul(M_reg, v_M_det), dim=-2)
            cond_M_tensor = (eigvals_M.max(dim=-1).values / eigvals_M.min(dim=-1).values.clamp(min=1e-3)).mean()

            w_contraction = min(1.0, (epoch + 1) / 6.0)
            ncm_loss = (loss_viol_all * 3.0 * w_contraction) + cond_M_tensor * 0.005

        if torch.isnan(ncm_loss).any() or torch.isinf(ncm_loss).any():
            print(f">> [WARN] NaN/Inf NCM loss encountered at Epoch {epoch+1}. Skipping optimizer step.")
            optimizer_actor.zero_grad()
            optimizer_ncm.zero_grad()
            optimizer_morph.zero_grad()
            continue

        scaler.scale(ncm_loss).backward()
        scaler.unscale_(optimizer_actor)
        scaler.unscale_(optimizer_ncm)
        scaler.unscale_(optimizer_morph)

        grad_norm_actor = torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=1.0).item()
        grad_norm_ncm = torch.nn.utils.clip_grad_norm_(ncm_net.parameters(), max_norm=1.0).item()
        grad_norm_morph = torch.nn.utils.clip_grad_norm_(env.morphology.parameters(), max_norm=0.5).item()

        if math.isnan(grad_norm_actor) or math.isnan(grad_norm_ncm) or math.isnan(grad_norm_morph):
            print(f">> [WARN] NaN gradient detected at Epoch {epoch+1}. Skipping optimizer step.")
            optimizer_actor.zero_grad()
            optimizer_ncm.zero_grad()
            optimizer_morph.zero_grad()
            scaler.update()
            continue

        scale_before = scaler.get_scale()
        scaler.step(optimizer_actor)
        scaler.step(optimizer_ncm)
        scaler.step(optimizer_morph)
        scaler.update()
        scale_after = scaler.get_scale()

        if scale_before <= scale_after:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                scheduler_actor.step()
                scheduler_ncm.step()

        if (epoch + 1) % 8 == 0 or epoch == 0:
            ep_secs = time.perf_counter() - t_ep_start
            steps_per_sec = (num_envs * rollout_steps) / max(1e-4, ep_secs)
            peak_vram_mb = (torch.cuda.max_memory_allocated(DEVICE) / (1024.0**2)) if DEVICE.type == "cuda" else 0.0

            lam_max_samples = eigvals_R.max(dim=-1).values
            lam_mean = lam_max_samples.mean().item()
            lam_p95 = torch.quantile(lam_max_samples, 0.95).item()
            max_lambda_R = lam_max_samples.max().item()
            min_lambda_R = eigvals_R.min().item()
            min_lambda_M = eigvals_M.min().item()
            pct_cert = (lam_max_samples <= 0.0).float().mean().item() * 100.0
            cert_status = f"SAMPLED OK ({pct_cert:.0f}%)" if max_lambda_R <= 0.0 else f"{pct_cert:.0f}% <= 0"

            foil_mask = ((m["mode"] == MODE_FOILING) | (m["mode"] == MODE_SOARING)).squeeze(-1)
            if foil_mask.any():
                h_m = m["h_wing"][foil_mask].mean().item()
                h_est = m["clearance_est"][foil_mask].mean().item()
                d_sub_cm = m["immersion_depth"][foil_mask].mean().item() * 100.0
                f_lift_foil = m["f_hydro_lift"][foil_mask].mean().item()
                f_rot_foil = m["f_rotors"][foil_mask].mean().item()
                f_push_foil = m["f_pusher"][foil_mask].mean().item()
                mean_gimbal_deg = m["rotor_tilt_deg"][foil_mask].mean().item()
                p_elec_scalar = m["p_elec"][foil_mask].mean().item()
                flow_status = "VENTILATED" if m["is_vent_L"][foil_mask].float().mean().item() > 0.5 else "PRIME (Attached)"
            else:
                h_m = m["h_wing"].mean().item()
                h_est = m["clearance_est"].mean().item()
                d_sub_cm = m["immersion_depth"].mean().item() * 100.0
                f_lift_foil = m["f_hydro_lift"].mean().item()
                f_rot_foil = m["f_rotors"].mean().item()
                f_push_foil = m["f_pusher"].mean().item()
                mean_gimbal_deg = m["rotor_tilt_deg"].mean().item()
                p_elec_scalar = m["p_elec"].mean().item()
                flow_status = "PRIME (Attached)"

            with torch.no_grad():
                raw_res = torch.tanh(actor_net.net(m["actor_obs"]))
                raw_rms = torch.sqrt(torch.mean(raw_res**2, dim=0)).cpu().numpy()
                raw_sat = ((raw_res.abs() > 0.95).float().mean(dim=0) * 100.0).cpu().numpy()

            ekf_rmse_h_cm = torch.sqrt(torch.mean(m["ekf_err_h"]**2)).item() * 100.0
            ekf_rmse_vx = torch.sqrt(torch.mean(m["ekf_err_vx"]**2)).item()
            ekf_rmse_att = torch.sqrt(torch.mean(m["ekf_err_att_deg"]**2)).item()
            l1_mean = m["sigma_l1_norm"].mean().item()
            l1_max = m["sigma_l1_norm"].max().item()

            slam_pct = (foil_slam_sum / max(1.0, foil_total_count)) * 100.0
            breach_pct = (foil_breach_sum / max(1.0, foil_total_count)) * 100.0
            capsize_pct = (capsize_acc / rollout_steps) * 100.0
            clip_pct = (clip_acc / rollout_steps) * 100.0

            mode_cards = []
            for k in range(7):
                mk = (m["mode"].squeeze(-1) == k)
                if mk.any():
                    dh_cm = (m["clearance_eval"][mk] - m["h_target_mode"][mk]).abs().mean().item() * 100.0
                    dvx = (m["vel_eval"][mk, 0:1] - m["v_target_mode"][mk]).abs().mean().item()
                    dy = m["pos_eval"][mk, 1:2].abs().mean().item()
                    mode_cards.append(f"{mode_short_names[k]}({mk.sum().item()}):Δh={dh_cm:.1f}cm,ΔV={dvx:.2f},|y|={dy:.2f}")
                else:
                    mode_cards.append(f"{mode_short_names[k]}(0):-")
            mode_line_1 = " | ".join(mode_cards[:4])
            mode_line_2 = " | ".join(mode_cards[4:])

            v_batt_scalar = m["v_batt"].mean().item()
            i_batt_scalar = m["i_batt"].mean().item()
            soc_scalar = m["soc"].mean().item()
            qp_override_scalar = m["qp_override"].mean().item()

            lr_act = scheduler_actor.get_last_lr()[0]
            lr_ncm = scheduler_ncm.get_last_lr()[0]

            m_dict = env.morphology()
            span_val = m_dict["wing_span"].item()
            ar_val = m_dict["aspect_ratio"].item()
            f_span_val = m_dict["foil_span"].item()
            strut_val = m_dict["strut_length"].item()
            mass_val = m_dict["total_mass"].item()

            print(
                f"\n[Epoch {epoch+1:02d}/{epochs} | {curriculum_stage} | α={alpha_t:.3f} | {ep_secs:.2f}s ({steps_per_sec/1000:.1f}k steps/s) | VRAM: {peak_vram_mb:.0f}MB] ───────\n"
                f"  ├─ 1. WAVE CLEARANCE (FOIL): True h: {h_m:.3f}m | Est h_hat: {h_est:.3f}m | EKF RMSE: Δh={ekf_rmse_h_cm:.1f}cm, ΔVx={ekf_rmse_vx:.2f}m/s, ΔAtt={ekf_rmse_att:.2f}°\n"
                f"  ├─ 2. HYDROFOIL (FOIL SUB):  Immersion: {d_sub_cm:.1f}cm | Foil Lift: {f_lift_foil:.1f}N | Rotors: {f_rot_foil:.1f}N (Gimbal: {mean_gimbal_deg:.1f}°) | Pusher: {f_push_foil:.1f}N | [{flow_status}]\n"
                f"  ├─ 3. MODE SCORECARD (1/2):  {mode_line_1}\n"
                f"  ├─ 4. MODE SCORECARD (2/2):  {mode_line_2}\n"
                f"  ├─ 5. LOSS DECOMPOSITION:    L_h: {c_h_scalar:.2f} | L_ceil: {c_ceil_scalar:.2f} | L_def: {c_def_scalar:.2f} | L_spd: {c_spd_scalar:.2f} | L_drft: {c_drift_scalar:.2f} | L_rate: {c_omega_scalar:.2f} | L_clip: {c_clip_scalar:.2f} | L_vent: {c_vent_scalar:.2f} | L_slk: {c_slack_scalar:.2f} | L_str: {c_stress_scalar:.2f}\n"
                f"  ├─ 6. NN RESIDUAL & L1 DOB:  Raw RMS: ({raw_rms[0]:.2f}, {raw_rms[1]:.2f}, {raw_rms[2]:.2f}) | Sat(>95%): ({raw_sat[0]:.1f}%, {raw_sat[1]:.1f}%, {raw_sat[2]:.1f}%) | ||σ_L1|| mean/max: {l1_mean:.2f}/{l1_max:.2f} m/s²\n"
                f"  ├─ 7. TAIL FAILURE RATES:    Slam(<8cm): {slam_pct:.2f}% | Breach(>23.5cm): {breach_pct:.2f}% | Capsize(|φ|>25°): {capsize_pct:.2f}% | Motor Clip: {clip_pct:.2f}%\n"
                f"  ├─ 8. RIEMANNIAN SPECTRUM5D: λ_max(R5) mean/p95/max: {lam_mean:+.3f} / {lam_p95:+.3f} / {max_lambda_R:+.3f} [{cert_status}] | λ_min(M5): {min_lambda_M:.3f} | κ(M5): {cond_M_tensor.item():.1f} | Viol: {loss_viol_all.item():.4f} (N={M_batch:,})\n"
                f"  ├─ 9. ATTITUDE & SAFETY:     Rate ||ω||: {safe_norm(m['omega'], dim=-1).mean().item()*57.3:.1f}°/s | AttErr: {safe_norm(m['e_R'], dim=-1).mean().item()*57.3:.2f}° | QP Override: {qp_override_scalar:.3f}m/s² | Slack: {m['max_slack'].mean().item():.4f}\n"
                f"  ├─ 10. POWER & MORPHOLOGY:   Power(Foil): {p_elec_scalar:.1f}W | V_batt: {v_batt_scalar:.2f}V ({i_batt_scalar:.1f}A, SoC:{soc_scalar:.1f}%) | Span: {span_val:.2f}m(AR:{ar_val:.1f}) | Foil: {f_span_val:.2f}m(Str:{strut_val:.2f}m) | Mass: {mass_val:.2f}kg\n"
                f"  └─ 11. OPTIMIZATION HEALTH:  Actor Grad: {grad_norm_actor:.3f} | NCM Grad: {grad_norm_ncm:.3f} | Morph Grad: {grad_norm_morph:.3f} | LRs: [{lr_act:.1e}, {lr_ncm:.1e}]"
            )

    print("\n" + "=" * 114)
    print(">> MULTI-STAGE BI-LEVEL MORPHOGENETIC TRAINING COMPLETE")
    final_morph = env.morphology()
    print(f"   [FINAL EVOLVED CAD SPECIFICATIONS]")
    print(f"   - Optimal Wing Span:        {final_morph['wing_span'].item():.3f} m")
    print(f"   - Optimal Wing Chord:       {final_morph['wing_chord'].item():.3f} m")
    print(f"   - Aspect Ratio:             {final_morph['aspect_ratio'].item():.2f}")
    print(f"   - Hydrofoil Span:           {final_morph['foil_span'].item():.3f} m")
    print(f"   - Strut Length:             {final_morph['strut_length'].item():.3f} m")
    print(f"   - Caudal Fin Area:          {final_morph['caudal_area'].item()*10000:.1f} cm^2")
    print(f"   - Wing Anhedral Trap Angle: {math.degrees(final_morph['anhedral_rad'].item()):+.2f} deg")
    print(f"   - Total Structural Mass:    {final_morph['total_mass'].item():.3f} kg")
    print("=" * 114 + "\n")

    return actor_net, ncm_net, env.morphology


# =============================================================================
# 14b. STATIC TRIM REPORT (run with --trim)
# =============================================================================
def static_trim_report():
    """Prints foil lift, drag, WIG lift, rotor trim, and pusher power at cruise."""
    with torch.no_grad():
        morph = VehicleMorphologyGenome(torch.device("cpu"))()
    f_area = morph["foil_area"].item()
    w_area = morph["wing_area"].item()
    w_chord = morph["wing_chord"].item()
    ar = morph["aspect_ratio"].item()
    mass = morph["total_mass"].item()
    weight = mass * GRAVITY
    strut = morph["strut_length"].item()
    v = 5.20
    q_w = 0.5 * 1014.0 * v**2
    q_a = 0.5 * RHO_AIR * v**2
    h_opt = RIDE_HEIGHT_OPTIMAL
    phi_wig = 1.0 + 0.35 / (0.10 + 2.2 * (h_opt / w_chord)**1.15)
    cl_wig = (WING_CL0 + 4.5 * 0.03) * phi_wig
    wig_lift = q_a * w_area * cl_wig
    wig_drag = q_a * w_area * (WING_CD0 + cl_wig**2 / (math.pi * ar * WING_OSWALD) * 0.65)

    print("=" * 88)
    print(f" STATIC TRIM REPORT @ {v:.2f} m/s | mass {mass:.3f} kg | weight {weight:.1f} N | q_water {q_w/1000:.1f} kPa")
    print(f" foil area {f_area:.4f} m^2 | strut {strut:.3f} m | WIG lift {wig_lift:.2f} N | immersion {(strut-h_opt)*100:.1f} cm")
    print("-" * 88)
    print(f" {'alpha_total[deg]':>16} | {'CL':>6} | {'foil lift[N]':>12} | {'rotors[N]':>10} | {'drag/pusher[N]':>15}")
    for a_deg in [0.5, 1.0, 1.25, 1.5, 2.0, 3.0]:
        cl = min(max(FOIL_CL_ALPHA * math.radians(a_deg), -0.10), 1.30)
        l_foil = 2.0 * min(q_w * (f_area * 0.5) * cl, FOIL_LIFT_CLAMP_PER_SIDE)
        cd_foil = FOIL_CD0 + 0.05 * cl**2
        d_total = q_w * f_area * cd_foil + wig_drag
        f_rot = max(4.4, weight - wig_lift - l_foil)
        print(f" {a_deg:16.2f} | {cl:6.3f} | {l_foil:12.1f} | {f_rot:10.1f} | {d_total:15.2f}")
    a_hold = (weight - wig_lift - 4.4) / (2.0 * q_w * (f_area * 0.5) * FOIL_CL_ALPHA)
    print("-" * 88)
    print(f" alpha_total for balanced level cruise (4.4 N rotor control floor): {math.degrees(a_hold):.2f} deg")
    print(f" configured foil decalage range: {FOIL_DECALAGE_DEG[0]:.2f}-{FOIL_DECALAGE_DEG[1]:.2f} deg | gimbal max: {ROTOR_MAX_TILT_DEG:.1f} deg | pusher max: {MAX_PUSHER_THRUST:.1f} N")
    print("=" * 88)

    # =============================================================================
# 15. AUTOMATED ONNX EXPORT & LATENCY BENCHMARK
# =============================================================================
def export_flight_software_onnx(actor_net, filename="wave_pilot_hardware_fp32.onnx"):
    actor_net.eval()
    dummy_input = torch.randn(1, 20, device=DEVICE)
    try:
        torch.onnx.export(
            actor_net,
            dummy_input,
            filename,
            input_names=["actor_obs_20state"],
            output_names=["accel_cmd_3axis"],
            dynamic_axes={"actor_obs_20state": {0: "batch"}, "accel_cmd_3axis": {0: "batch"}},
            opset_version=14,
        )
        print(f">> [ONNX DEPLOYMENT] 20-State Pilot network compiled successfully to: {filename}")
        times = []
        with torch.no_grad():
            for _ in range(150):
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = actor_net(dummy_input)
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
        mean_lat = np.mean(times[20:])
        print(f">> [ONNX BENCHMARK] Mean network-only inference latency: {mean_lat:.3f} ms "
              f"(the full control step incl. EKF/QP is much larger; see cockpit step timer)\n")
    except Exception as e:
        print(f">> [ONNX NOTICE] Edge ONNX export skipped ({e}).")


# =============================================================================
# 16. 3D INTERACTIVE OCEAN COCKPIT
# =============================================================================
def project_3d_batch(pts_3d, cam_pos, cam_lookat, screen_w, screen_h, focal):
    fwd = cam_lookat - cam_pos
    fwd_len = safe_norm(fwd)
    fwd = fwd / (fwd_len + 1e-6)

    world_up = torch.tensor([0.0, 0.0, 1.0], device=pts_3d.device)
    right = torch.linalg.cross(fwd, world_up)
    rlen = safe_norm(right)
    right = torch.where(rlen < 1e-4, torch.tensor([1.0, 0.0, 0.0], device=pts_3d.device), right / (rlen + 1e-6))
    up = torch.linalg.cross(right, fwd)

    delta = pts_3d - cam_pos.unsqueeze(0)
    depth = torch.sum(delta * fwd.unsqueeze(0), dim=-1, keepdim=True)
    valid_mask = (depth > 0.20).squeeze(-1)

    safe_depth = torch.clamp(depth, min=0.20)
    px = screen_w * 0.5 + focal * torch.sum(delta * right.unsqueeze(0), dim=-1, keepdim=True) / safe_depth
    py = screen_h * 0.5 - focal * torch.sum(delta * up.unsqueeze(0), dim=-1, keepdim=True) / safe_depth

    coords_2d = torch.cat([px, py], dim=-1)
    coords_2d = torch.nan_to_num(coords_2d, nan=-9999.0, posinf=9999.0, neginf=-9999.0)
    coords_2d = torch.clamp(coords_2d, -3000, 5000)
    return coords_2d.cpu().numpy(), valid_mask.cpu().numpy(), depth.squeeze(-1).cpu().numpy()


def to_screen_pt(coord):
    try:
        x, y = float(coord[0]), float(coord[1])
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            return None
        return (int(round(x)), int(round(y)))
    except Exception:
        return None


def run_interactive_cockpit_visualizer(actor_net, ncm_net, morphology_genome=None):
    if pygame is None:
        print("[ERROR] Pygame required for visualization. Run: pip install pygame")
        return

    torch.set_num_threads(1)
    pygame.init()
    SCREEN_W, SCREEN_H = 1260, 800
    try:
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    except pygame.error as e:
        print(f"[ERROR] Display unavailable ({e}). Cockpit skipped.")
        return

    pygame.display.set_caption("Aeromarine Autonomous Flight Engine (simulated 200 Hz control loop)")
    clock = pygame.time.Clock()
    font_hud = pygame.font.SysFont("consolas", 12, bold=True)
    font_chart = pygame.font.SysFont("consolas", 10)

    visualizer_device = torch.device("cpu")
    env = WaveRidingAutonomyEnv(num_envs=1, dt=CONTROL_LOOP_DT, device=visualizer_device, use_embedded_qp=True)
    env.set_wave_curriculum(1.0)
    actor_net.to(visualizer_device).eval()
    ncm_net.to(visualizer_device).eval()
    if morphology_genome is not None:
        env.morphology.load_state_dict(morphology_genome.state_dict())
    env.morphology.to(visualizer_device).eval()

    cam_yaw, cam_pitch, cam_dist = 0.75, 0.32, 5.5
    running = True
    frame_count = 0
    ema_latency_us = 420.0

    history_len = 160
    hist_clearance = deque([0.185] * history_len, maxlen=history_len)
    hist_wave = deque([0.0] * history_len, maxlen=history_len)
    hist_lift = deque([18.0] * history_len, maxlen=history_len)
    hist_rotors = deque([4.4] * history_len, maxlen=history_len)

    gx = np.linspace(-10.0, 10.0, 15)
    gy = np.linspace(-8.0, 8.0, 13)
    dx_grid = float(gx[1] - gx[0])
    dy_grid = float(gy[1] - gy[0])
    grid_offsets = torch.tensor([[x, y, 0.0] for x in gx for y in gy], device=visualizer_device, dtype=torch.float32)
    pts_tensor = grid_offsets.clone()

    with torch.no_grad():
        m_cad = env.morphology()
    cad_w_span = m_cad["wing_span"].item()
    cad_w_chord = m_cad["wing_chord"].item()
    cad_ar = m_cad["aspect_ratio"].item()
    cad_f_span = m_cad["foil_span"].item()
    cad_strut = m_cad["strut_length"].item()
    cad_mass = m_cad["total_mass"].item()
    cad_anhed = math.degrees(m_cad["anhedral_rad"].item())
    d_arm = ARM_LENGTH / math.sqrt(2.0)

    body_points_local_base = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -HULL_DRAFT_MAX],
        [0.0, 0.0, -cad_strut],
        [0.0, cad_w_span * 0.5, 0.0],
        [0.0, -cad_w_span * 0.5, 0.0],
        [0.0, -cad_f_span * 0.5, -cad_strut],
        [0.0, cad_f_span * 0.5, -cad_strut]
    ], device=visualizer_device, dtype=torch.float32)

    hubs_local = torch.tensor([
        [d_arm, -d_arm, 0.0], [d_arm, d_arm, 0.0],
        [-d_arm, d_arm, 0.0], [-d_arm, -d_arm, 0.0]
    ], device=visualizer_device, dtype=torch.float32)

    focal_len = (SCREEN_H * 0.5) / math.tan(math.radians(52.0) * 0.5)
    mode_names = ["LOITER", "TAKEOFF", "FOILING", "EGRESS", "FLIGHT", "INGRESS", "SOARING"]

    button_specs = [
        ("1:LOITER", MODE_LOITER, "MODE_LOITER (Decelerate & Settle)"),
        ("2:TAKEOFF", MODE_TAKEOFF, "MODE_TAKEOFF (Accelerate to 5.2 m/s)"),
        ("3:FOIL", MODE_FOILING, "MODE_FOILING (Optimal WIG Wave-Ride)"),
        ("4:EGRESS", MODE_EGRESS, "MODE_EGRESS (Punch to Free Air)"),
        ("5:INGRESS", MODE_INGRESS, "MODE_INGRESS (Flare & Water Re-Entry)"),
        ("6:SOAR", MODE_SOARING, "MODE_SOARING (Aquatic Sculling)"),
        ("A:AUTO", None, "AUTONOMOUS SORTIE MISSION"),
    ]
    btn_w, btn_h, btn_gap = 128, 34, 12
    total_btn_w = len(button_specs) * btn_w + (len(button_specs) - 1) * btn_gap
    btn_start_x = (SCREEN_W - total_btn_w) // 2
    btn_y = SCREEN_H - 50
    button_rects = [
        pygame.Rect(btn_start_x + i * (btn_w + btn_gap), btn_y, btn_w, btn_h)
        for i in range(len(button_specs))
    ]

    def trigger_mode_action(mode_val, label_desc):
        if mode_val is None:
            env.auto_mission_active = not env.auto_mission_active
            env.auto_mission_timer = 0.0
            print(f">> [AUTONOMY] Autonomous Sortie Mission Engaged = {env.auto_mission_active}")
        else:
            env.auto_mission_active = False
            env.set_mode(mode_val)
            print(f">> [MODE TRANSITION] Switched to {label_desc}")

    print("\n" + "=" * 114)
    print(" COCKPIT ACTIVE (simulation runs at dt = 5 ms per rendered frame, i.e. slower than real time)")
    print(f"   CAD: Wing {cad_w_span:.2f}m x {cad_w_chord:.2f}m (AR {cad_ar:.1f}, Anhed {cad_anhed:+.1f}°) | Foil {cad_f_span:.2f}m (Strut {cad_strut:.2f}m) | Mass {cad_mass:.2f}kg")
    print("   Click HUD mode buttons or press [1]-[6] (top row / numpad) | [A] -> AUTONOMOUS SORTIE MISSION")
    print("=" * 114 + "\n")

    wall_t0 = time.perf_counter()

    while running:
        clock.tick(60)
        frame_count += 1
        do_diagnostics = (frame_count % 45 == 0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                cam_dist = np.clip(cam_dist - event.y * 0.8, 2.5, 18.0)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for idx, rect in enumerate(button_rects):
                    if rect.collidepoint(event.pos):
                        trigger_mode_action(button_specs[idx][1], button_specs[idx][2])
                        pygame.mouse.get_rel()
                        break
            elif event.type == pygame.KEYDOWN:
                ch = event.unicode.lower() if event.unicode else ""
                if event.key in (pygame.K_1, pygame.K_KP1) or ch == "1":
                    trigger_mode_action(button_specs[0][1], button_specs[0][2])
                elif event.key in (pygame.K_2, pygame.K_KP2) or ch == "2":
                    trigger_mode_action(button_specs[1][1], button_specs[1][2])
                elif event.key in (pygame.K_3, pygame.K_KP3) or ch == "3":
                    trigger_mode_action(button_specs[2][1], button_specs[2][2])
                elif event.key in (pygame.K_4, pygame.K_KP4) or ch == "4":
                    trigger_mode_action(button_specs[3][1], button_specs[3][2])
                elif event.key in (pygame.K_5, pygame.K_KP5) or ch == "5":
                    trigger_mode_action(button_specs[4][1], button_specs[4][2])
                elif event.key in (pygame.K_6, pygame.K_KP6) or ch == "6":
                    trigger_mode_action(button_specs[5][1], button_specs[5][2])
                elif event.key == pygame.K_a or ch == "a":
                    trigger_mode_action(button_specs[6][1], button_specs[6][2])

        mouse = pygame.mouse.get_pressed()
        m_pos = pygame.mouse.get_pos()
        if (mouse[0] or mouse[2]) and m_pos[1] < (btn_y - 6):
            dx, dy = pygame.mouse.get_rel()
            cam_yaw += dx * 0.008
            cam_pitch = np.clip(cam_pitch + dy * 0.008, 0.05, 1.45)
        else:
            pygame.mouse.get_rel()

        if env.auto_mission_active:
            env.auto_mission_timer += CONTROL_LOOP_DT
            t_m = env.auto_mission_timer
            if t_m < 3.0:
                env.set_mode(MODE_TAKEOFF)
            elif t_m < 15.0:
                env.set_mode(MODE_FOILING)
            elif t_m < 25.0:
                env.set_mode(MODE_SOARING)
            elif t_m < 28.0:
                env.set_mode(MODE_EGRESS)
            elif t_m < 38.0:
                env.set_mode(MODE_FREE_FLIGHT)
            elif t_m < 42.0:
                env.set_mode(MODE_INGRESS)
            elif t_m < 55.0:
                env.set_mode(MODE_FOILING)
            else:
                env.set_mode(MODE_LOITER)
                env.auto_mission_timer = 0.0

        t_loop_start = time.perf_counter()
        with torch.no_grad():
            m = env.step(actor_net, ncm_net, export_diagnostics=do_diagnostics)
        step_latency = (time.perf_counter() - t_loop_start) * 1e6

        if not do_diagnostics:
            ema_latency_us = 0.95 * ema_latency_us + 0.05 * step_latency

        sim_t = env.sim_time[0, 0].item()
        clearance_val = m["clearance_est"][0, 0].item()
        wave_val = m["eta"][0, 0].item()
        lift_val = m["f_hydro_lift"][0, 0].item()
        rotors_val = m["f_rotors"][0, 0].item()
        pusher_val = m["f_pusher"][0, 0].item()
        cur_mode_idx = int(env.current_mode[0, 0].item())
        cur_mode_name = mode_names[cur_mode_idx]
        g_vals = m["rotor_tilt_deg"][0].cpu().numpy()

        hist_clearance.append(clearance_val)
        hist_wave.append(wave_val)
        hist_lift.append(lift_val)
        hist_rotors.append(rotors_val)

        if do_diagnostics:
            rt_factor = sim_t / max(1e-3, time.perf_counter() - wall_t0)
            cpu_budget_x = 5000.0 / max(1.0, ema_latency_us)
            flow_str = "VENTILATED" if m["is_ventilated"] else "PRIME (Attached)"
            radar_status = "LOCKED" if m["radar_locked"] else ("SPRAY DROPOUT" if m["radar_dropout"] else "SEARCHING")
            multipath_str = "REJECTED (Masked)" if m["multipath_rejected"] else "CLEAN"
            dvl_on, flow_on = m["sensor_locks"]
            sensor_handoff_str = f"DVL:{'LOCK' if dvl_on else 'OFF'} | Flow:{'ON' if flow_on else 'OFF'}"
            m_vals = m["motors"][0].cpu().numpy()
            m_disp = f"M0:{m_vals[0]:.2f}N | M1:{m_vals[1]:.2f}N | M2:{m_vals[2]:.2f}N | M3:{m_vals[3]:.2f}N | Push:{pusher_val:.2f}N"
            g_disp = f"G0:{g_vals[0]:.1f}° | G1:{g_vals[1]:.1f}° | G2:{g_vals[2]:.1f}° | G3:{g_vals[3]:.1f}°"
            gf_x, gf_y = m["gimbal_fxy"]
            d_surf = m["hydro_deflections"]
            eul = m["euler_deg"]
            om_d = m["omega_deg_0"]
            h_tgt_v, v_tgt_v = m["targets_0"]
            eh_i, ey_i = m["e_int_hy"]
            ekf_dh, ekf_dvx, ekf_datt = m["ekf_err_0"]
            ba = m["ba_hat_0"]
            b_om = m["bias_omega_deg"]
            ann = m["a_neural_0"]
            aqp = m["a_safe_0"]
            nraw = m["nn_raw"]
            sl1 = m["sigma_l1_xyz"]

            if env.auto_mission_active:
                t_m = env.auto_mission_timer
                if t_m < 3.0: phase_str = f"SORTIE (T+{t_m:.1f}s/3.0s -> FOIL)"
                elif t_m < 15.0: phase_str = f"SORTIE (T+{t_m:.1f}s/15.0s -> SCULL)"
                elif t_m < 25.0: phase_str = f"SORTIE (T+{t_m:.1f}s/25.0s -> EGRESS)"
                elif t_m < 28.0: phase_str = f"SORTIE (T+{t_m:.1f}s/28.0s -> FLIGHT)"
                elif t_m < 38.0: phase_str = f"SORTIE (T+{t_m:.1f}s/38.0s -> INGRESS)"
                elif t_m < 42.0: phase_str = f"SORTIE (T+{t_m:.1f}s/42.0s -> FOIL)"
                else: phase_str = f"SORTIE (T+{t_m:.1f}s/55.0s -> LOITER)"
            else:
                phase_str = "MANUAL PILOT"

            cav_str = f"{m['cavitation_sigma']:.2f} [SAFE]" if m['cavitation_sigma'] > SIGMA_CAV_CRIT else f"{m['cavitation_sigma']:.2f} [CAVITATING!]"
            budget_str = "" if ema_latency_us < 5000.0 else " [OVER 5 ms BUDGET]"

            print(
                f"\n[T={sim_t:05.2f}s | Mode: {m['mode_name']:<8} | Mission: {phase_str} | Status: {m['status_banner']} | Step: {ema_latency_us:05.1f}µs ({cpu_budget_x:.2f}x CPU, {rt_factor:.2f}x RT){budget_str}] " + "─" * 8 + "\n"
                f"  ├─ 1. CORRIDOR & TARGETS:    Vx: {m['vel'][0,0].item():.2f}m/s (Tgt:{v_tgt_v:.2f}) | Y-Dev: {m['pos'][0,1].item():+.2f}m (Vy:{m['vel'][0,1].item():+.2f}, IntY:{ey_i:+.2f}) | Sideslip β: {m['beta_sideslip_deg']:+.1f}° | YawErr: {m['psi_err_deg']:+.1f}°\n"
                f"  ├─ 2. ATTITUDE & RATES:      φ: {eul[0]:+.1f}° (cmd {m['phi_btt_deg']:+.1f}°) | θ: {eul[1]:+.1f}° (shed {m['theta_shed_deg']:+.1f}°) | ψ: {eul[2]:+.1f}° (cmd {m['psi_des_deg']:+.1f}°) | ω=({om_d[0]:+.1f}, {om_d[1]:+.1f}, {om_d[2]:+.1f})°/s\n"
                f"  ├─ 3. ACTIVE HYDRO-SURFACES: Elevon L: {d_surf[0]:+.1f}° | Elevon R: {d_surf[1]:+.1f}° | Rudder: {d_surf[2]:+.1f}° | δ_coll: {m['delta_collective_deg']:+.1f}° | Foil α: {m['alpha_total_deg']:+.2f}° (Stall: {m['stall_margin_deg']:.1f}°) | Mom: {m['strut_moment_nm'][0,0].item():.2f}N·m\n"
                f"  ├─ 4. ES-EKF & SENSORS:      True h: {m['h_wing'][0,0].item():.3f}m | Est h: {clearance_val:.3f}m (Tgt:{h_tgt_v:.3f}, σ_z:±{m['sigma_z_cm']:.1f}cm) | Err(Δh:{ekf_dh:+.1f}cm, ΔV:{ekf_dvx:+.2f}, ΔAtt:{ekf_datt:.2f}°) | TapBias:{m['bernoulli_bias_cm_0']:+.1f}cm | {sensor_handoff_str}\n"
                f"  ├─ 5. RADAR & IMU BIASES:    Radar: {radar_status} | Multipath: {multipath_str} | Gyro Bias: ({b_om[0]:+.2f}, {b_om[1]:+.2f}, {b_om[2]:+.2f})°/s | Accel Bias: ({ba[0]:+.3f}, {ba[1]:+.3f}, {ba[2]:+.3f})m/s²\n"
                f"  ├─ 6. SEA DISPERSION:        Fwd Raycast: {m['eta_forward_meas'][0,0].item():+.3f}m | Nadir η̂: {m['eta_nadir_est'][0,0].item():+.3f}m (true {m['eta'][0,0].item():+.3f}m) | Dominant λ: {m['wavelength_m']:.1f}m (cp: {m['celerity_mps']:.1f}m/s) | Preview (1.25s): {m['eta_preview'][0,0].item():+.3f}m\n"
                f"  ├─ 7. FLUID & AERO-HYDRO:    Immersion: {m['immersion_depth'][0,0].item()*100:.1f}cm [{flow_str}] | Foil Lift: {lift_val:04.1f}N | Wing: {m['f_wing_lift'][0,0].item():04.1f}N (PAR:{m['f_par_z']:.1f}N) | Hull: {m['f_planing_z']:04.1f}N (Fn:{m['froude_fn']:.2f}) | Rotors: {rotors_val:04.1f}N | Push: {pusher_val:04.1f}N\n"
                f"  ├─ 8. SCULLING & SOARING:    Caudal Thrust: {m['f_scull_thrust']:.2f}N (St: {m['strouhal_st']:.2f}) | Wave Surge: {m['f_soar_surge']:+.2f}N | Cavitation σ: {cav_str}\n"
                f"  ├─ 9. MOTOR & GIMBAL ALLOC:  [{m_disp}] | Tilts: [{g_disp}] | Gimbal F=({gf_x:+.2f}, {gf_y:+.2f})N | Auth: {m['torque_auth_pct']:.0f}% | Headroom: {m['headroom_min'][0,0].item():+.2f}N | [{m['sat_str']}]\n"
                f"  ├─ 10. BATTERY & ENDURANCE:  Power: {m['p_elec'][0,0].item():04.1f}W ({m['power_saved_pct'][0,0].item():+.1f}% SAVED) | SoC: {m['soc'][0,0].item():.1f}% | V_batt: {m['v_batt'][0,0].item():.2f}V (Sag:{m['v_sag']:.2f}V, {m['i_batt'][0,0].item():.1f}A) | Temp: {m['cell_temp_c']:.1f}°C | Endur: {m['endurance_min']:.1f}m ({m['wh_per_km']:.1f}Wh/km)\n"
                f"  └─ 11. CONTROL & SAFETY:     a_NN=({ann[0]:+.2f},{ann[1]:+.2f},{ann[2]:+.2f}) -> a_QP=({aqp[0]:+.2f},{aqp[1]:+.2f},{aqp[2]:+.2f}) | Raw=({nraw[0]:+.2f},{nraw[1]:+.2f},{nraw[2]:+.2f}) | σ_L1=({sl1[0]:+.2f},{sl1[1]:+.2f},{sl1[2]:+.2f}) | IntH:{eh_i:+.3f} | QP:{m['qp_override'][0,0].item():.3f} {m['barrier_name']} | Ceil:{m['margin_ceil_cm']:+.1f}cm Flr:{m['margin_floor_cm']:+.1f}cm"
            )

        robot_p_t = env.pos[0]
        cam_lookat = robot_p_t.clone()
        cam_x = cam_lookat[0] - cam_dist * math.cos(cam_pitch) * math.cos(cam_yaw)
        cam_y = cam_lookat[1] - cam_dist * math.cos(cam_pitch) * math.sin(cam_yaw)
        cam_z = max(0.5, float(cam_lookat[2].item() + cam_dist * math.sin(cam_pitch)))
        cam_pos = torch.tensor([float(cam_x), float(cam_y), cam_z], device=visualizer_device, dtype=torch.float32)

        screen.fill((8, 14, 22))

        snap_x = math.floor(float(robot_p_t[0].item()) / dx_grid) * dx_grid
        snap_y = math.floor(float(robot_p_t[1].item()) / dy_grid) * dy_grid
        pts_tensor.copy_(grid_offsets)
        pts_tensor[:, 0] += snap_x
        pts_tensor[:, 1] += snap_y
        t_tensor = env.sim_time.repeat(len(pts_tensor), 1)
        with torch.no_grad():
            eta_grid, _, _, _, _, _ = env.waves.evaluate_wave_kinematics(pts_tensor, t_tensor, wave_scale=1.0)
        ocean_3d = pts_tensor.clone()
        ocean_3d[:, 2:3] = eta_grid.view(-1, 1)

        grid_2d, grid_valid, _ = project_3d_batch(
            ocean_3d, cam_pos, cam_lookat, SCREEN_W, SCREEN_H, focal_len
        )

        nx, ny = len(gx), len(gy)
        for ix in range(nx - 1):
            for iy in range(ny - 1):
                idx = ix * ny + iy
                idx_x = (ix + 1) * ny + iy
                idx_y = ix * ny + (iy + 1)
                if grid_valid[idx] and grid_valid[idx_x]:
                    p1 = to_screen_pt(grid_2d[idx])
                    p2 = to_screen_pt(grid_2d[idx_x])
                    if p1 and p2: pygame.draw.line(screen, (22, 58, 92), p1, p2, 1)
                if grid_valid[idx] and grid_valid[idx_y]:
                    p1 = to_screen_pt(grid_2d[idx])
                    p3 = to_screen_pt(grid_2d[idx_y])
                    if p1 and p3: pygame.draw.line(screen, (22, 58, 92), p1, p3, 1)

        rot_t = env.rot[0]
        body_points_7 = body_points_local_base[:7, :]
        body_3d_craft = robot_p_t.unsqueeze(0) + torch.matmul(body_points_7, rot_t.T)

        preview_world = torch.tensor([
            robot_p_t[0].item() + env.vel[0, 0].item() * 1.25,
            robot_p_t[1].item(),
            m["eta_preview"][0, 0].item()
        ], device=visualizer_device, dtype=torch.float32).unsqueeze(0)

        body_3d = torch.cat([body_3d_craft, preview_world], dim=0)
        craft_2d, craft_valid, craft_depth = project_3d_batch(
            body_3d, cam_pos, cam_lookat, SCREEN_W, SCREEN_H, focal_len
        )

        p_c, p_hull, p_foil, p_wl, p_wr, p_fl, p_fr, p_prev = [to_screen_pt(craft_2d[i]) for i in range(8)]
        v_c, v_hull, v_foil, v_wl, v_wr, v_fl, v_fr, v_prev = [craft_valid[i] for i in range(8)]

        if v_wl and v_wr and p_wl and p_wr: pygame.draw.line(screen, (240, 245, 255), p_wl, p_wr, 4)
        if v_c and v_hull and p_c and p_hull: pygame.draw.line(screen, (180, 220, 255), p_c, p_hull, 6)
        if v_hull and v_foil and p_hull and p_foil: pygame.draw.line(screen, (255, 60, 60), p_hull, p_foil, 3)
        if v_foil and p_foil:
            foil_color = (0, 255, 200) if lift_val > 5.0 else (255, 180, 40)
            pygame.draw.circle(screen, foil_color, p_foil, 6)
            if v_fl and v_fr and p_fl and p_fr: pygame.draw.line(screen, foil_color, p_fl, p_fr, 3)
        if v_c and v_prev and p_c and p_prev:
            pygame.draw.line(screen, (0, 255, 255), p_c, p_prev, 1)
            pygame.draw.circle(screen, (0, 255, 255), p_prev, 5, 1)

        tilts_now = m["rotor_tilts"][0]
        ux_v = tilts_now[:, 0]
        uy_v = tilts_now[:, 1]
        uz_v = torch.sqrt((1.0 - ux_v**2 - uy_v**2).clamp(min=ROTOR_MAX_TILT_COS**2))
        thrust_vecs_local = torch.stack([ux_v, uy_v, uz_v], dim=-1) * 0.22
        hubs_3d = robot_p_t.unsqueeze(0) + torch.matmul(hubs_local, rot_t.T)
        tips_3d = hubs_3d + torch.matmul(thrust_vecs_local, rot_t.T)

        hubs_2d, hubs_valid, hubs_depth = project_3d_batch(
            hubs_3d, cam_pos, cam_lookat, SCREEN_W, SCREEN_H, focal_len
        )
        tips_2d, tips_valid, _ = project_3d_batch(
            tips_3d, cam_pos, cam_lookat, SCREEN_W, SCREEN_H, focal_len
        )
        for i in range(4):
            if hubs_valid[i]:
                p_h = to_screen_pt(hubs_2d[i])
                if p_h:
                    rad = max(3, int(22.0 / max(0.1, float(hubs_depth[i]))))
                    pygame.draw.circle(screen, (255, 210, 40), p_h, rad, 1)
                    if tips_valid[i]:
                        p_tip = to_screen_pt(tips_2d[i])
                        if p_tip:
                            pygame.draw.line(screen, (0, 255, 160), p_h, p_tip, 2)

        hud = pygame.Rect(15, 15, 820, 270)
        pygame.draw.rect(screen, (10, 16, 26), hud, border_radius=6)
        pygame.draw.rect(screen, (0, 220, 255), hud, 1, border_radius=6)

        m_vals = m["motors"][0].cpu().numpy()
        m_str = f"M0:{m_vals[0]:.1f}N | M1:{m_vals[1]:.1f}N | M2:{m_vals[2]:.1f}N | M3:{m_vals[3]:.1f}N | Push:{pusher_val:.1f}N"
        g_str = f"G0:{g_vals[0]:.1f}° | G1:{g_vals[1]:.1f}° | G2:{g_vals[2]:.1f}° | G3:{g_vals[3]:.1f}°"
        d_surf = m["hydro_deflections"]
        mission_tag = "AUTO SORTIE" if env.auto_mission_active else "MANUAL"
        h_tgt_hud, v_tgt_hud = env.h_target_filtered[0, 0].item(), env.v_target_filtered[0, 0].item()

        stats = [
            (f"AERO-HYDRO AUTONOMY ENGINE | MODE: {cur_mode_name} [{mission_tag}]", (0, 240, 255)),
            (f"CONTROL STEP (EMA): {ema_latency_us:.1f} µs (Budget: 5000 µs @ 200 Hz)", (0, 255, 120) if ema_latency_us < 5000.0 else (255, 90, 90)),
            (f"ACTIVE HYDRO-SURFACES: Elevon L: {d_surf[0]:+.1f}° | Elevon R: {d_surf[1]:+.1f}° | Rudder: {d_surf[2]:+.1f}° | δ_coll: {m['delta_collective_deg']:+.1f}°", (255, 255, 100)),
            (f"TOTAL WATER LIFT GENERATED: {lift_val:.1f} N | ROTOR POWER SAVINGS: {m['power_saved_pct'][0,0].item():+.1f}%", (0, 255, 200)),
            (f"OMNI GIMBAL TILTS (0-15°): [{g_str}]", (120, 255, 180)),
            (f"NAVIGATION EKF: Clearance: {clearance_val:.3f}m (Tgt {h_tgt_hud:.2f}m) | Vx: {m['vel'][0,0].item():.2f}m/s (Tgt {v_tgt_hud:.2f}) | Y: {m['pos'][0,1].item():+.2f}m", (100, 255, 100)),
            (f"PROPULSION ALLOCATION: [{m_str}]", (255, 210, 40)),
            (f"BATTERY (6S): SoC: {m['soc'][0,0].item():.1f}% | {m['v_batt'][0,0].item():.2f}V | {m['i_batt'][0,0].item():.1f}A | Temp: {m['cell_temp_c']:.1f}°C", (200, 220, 255)),
        ]
        for i, s in enumerate(stats):
            screen.blit(font_hud.render(s[0], True, s[1]), (hud.x + 14, hud.y + 12 + i * 26))

        chart_box = pygame.Rect(SCREEN_W - 380, 15, 365, 270)
        pygame.draw.rect(screen, (10, 16, 26), chart_box, border_radius=6)
        pygame.draw.rect(screen, (0, 180, 220), chart_box, 1, border_radius=6)

        screen.blit(font_chart.render("LIVE TELEMETRY SCOPE", True, (0, 240, 255)), (chart_box.x + 10, chart_box.y + 8))

        c1_rect = pygame.Rect(chart_box.x + 10, chart_box.y + 28, 345, 100)
        pygame.draw.rect(screen, (14, 22, 34), c1_rect)
        pygame.draw.line(screen, (60, 80, 100), (c1_rect.x, c1_rect.centery), (c1_rect.right, c1_rect.centery), 1)

        pts_clr, pts_wav = [], []
        for idx in range(history_len):
            x_pos = c1_rect.x + (idx / (history_len - 1)) * c1_rect.width
            y_clr = c1_rect.bottom - (hist_clearance[idx] / 0.40) * c1_rect.height
            y_wav = c1_rect.centery - (hist_wave[idx] / 0.20) * (c1_rect.height * 0.4)
            pts_clr.append((x_pos, np.clip(y_clr, c1_rect.top, c1_rect.bottom)))
            pts_wav.append((x_pos, np.clip(y_wav, c1_rect.top, c1_rect.bottom)))

        if len(pts_wav) > 1: pygame.draw.lines(screen, (0, 120, 255), False, pts_wav, 1)
        if len(pts_clr) > 1: pygame.draw.lines(screen, (0, 255, 120), False, pts_clr, 2)
        screen.blit(font_chart.render("Clearance h (Green) vs Wave eta (Blue)", True, (200, 220, 240)), (c1_rect.x + 6, c1_rect.y + 4))

        c2_rect = pygame.Rect(chart_box.x + 10, chart_box.y + 145, 345, 100)
        pygame.draw.rect(screen, (14, 22, 34), c2_rect)

        pts_lft, pts_rot = [], []
        for idx in range(history_len):
            x_pos = c2_rect.x + (idx / (history_len - 1)) * c2_rect.width
            y_lft = c2_rect.bottom - (hist_lift[idx] / 32.0) * c2_rect.height
            y_rot = c2_rect.bottom - (hist_rotors[idx] / 32.0) * c2_rect.height
            pts_lft.append((x_pos, np.clip(y_lft, c2_rect.top, c2_rect.bottom)))
            pts_rot.append((x_pos, np.clip(y_rot, c2_rect.top, c2_rect.bottom)))

        if len(pts_lft) > 1: pygame.draw.lines(screen, (0, 240, 220), False, pts_lft, 2)
        if len(pts_rot) > 1: pygame.draw.lines(screen, (255, 180, 40), False, pts_rot, 1)
        screen.blit(font_chart.render("Water Lift (Cyan) vs Rotors (Orange) [0-32 N]", True, (200, 220, 240)), (c2_rect.x + 6, c2_rect.y + 4))

        for idx, rect in enumerate(button_rects):
            label, mode_val, _ = button_specs[idx]
            is_active = (env.auto_mission_active if mode_val is None else (not env.auto_mission_active and cur_mode_idx == mode_val))
            bg_col = (0, 110, 140) if is_active else (14, 22, 34)
            bd_col = (0, 255, 180) if is_active else (60, 110, 140)
            tx_col = (255, 255, 255) if is_active else (170, 200, 220)
            pygame.draw.rect(screen, bg_col, rect, border_radius=5)
            pygame.draw.rect(screen, bd_col, rect, 2 if is_active else 1, border_radius=5)
            txt_surf = font_hud.render(label, True, tx_col)
            screen.blit(txt_surf, (rect.centerx - txt_surf.get_width() // 2, rect.centery - txt_surf.get_height() // 2))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


# =============================================================================
# 17. MASTER SCRIPT ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import atexit

    parser = argparse.ArgumentParser(description="Hydrofoil autonomy engine (multi-mode + 2-axis gimbal build)")
    parser.add_argument("--epochs", type=int, default=96)
    parser.add_argument("--envs", type=int, default=2048, help="parallel environments (lower this on OOM)")
    parser.add_argument("--steps", type=int, default=200, help="rollout steps per epoch")
    parser.add_argument("--tbptt", type=int, default=10, help="truncated BPTT window length in steps")
    parser.add_argument("--no-viz", action="store_true", help="skip the pygame cockpit after training")
    parser.add_argument("--trim", action="store_true", help="print the static foil trim report and exit")
    parser.add_argument("--decalage", type=float, nargs=2, default=None, metavar=("MIN_DEG", "MAX_DEG"),
                        help="foil decalage range in degrees (default 1.1 1.4)")
    args = parser.parse_args()
    if args.decalage is not None:
        FOIL_DECALAGE_DEG = (min(args.decalage), max(args.decalage))

    if args.trim:
        static_trim_report()
        sys.exit(0)

    telemetry_log_filename = f"aeromarine_flight_telemetry_{int(time.time())}.txt"
    sys.stdout = DualLogger(telemetry_log_filename)
    atexit.register(sys.stdout.close)
    print(f">> [SYS-INIT] Live Synchronous Disk Logging Active: '{telemetry_log_filename}'\n")

    try:
        trained_pilot, trained_ncm, evolved_morphology = train_wave_riding_pilot(
            epochs=args.epochs, num_envs=args.envs, rollout_steps=args.steps, tbptt_steps=args.tbptt
        )
    except torch.cuda.OutOfMemoryError:
        print(">> [ERROR] CUDA out of memory. Reduce --envs, --steps, or --tbptt (e.g. --envs 512 --tbptt 5)")
        sys.exit(1)

    export_flight_software_onnx(trained_pilot, "wave_pilot_hardware_fp32.onnx")
    if not args.no_viz:
        run_interactive_cockpit_visualizer(trained_pilot, trained_ncm, evolved_morphology)