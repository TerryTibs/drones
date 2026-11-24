"""
=============================================================================
UNIFIED MULTIMODAL HYDROFOIL AUTONOMY ENGINE (NCM-CBF-DSWP-BTT) -- REAL PHYSICS
=============================================================================
RESEARCH SIMULATION & SIL VALIDATION STACK (not flight-qualified):
  - 32-wave directional JONSWAP/PM sea + 2nd-order Stokes + Stokes drift
  - 8-strip spanwise T-foil & 5-strip depth strut w/ Hoerner spray drag
  - Kirchhoff 6-DoF hydrodynamic added mass, added inertia & Munk moments
  - Low-Re + Viterna post-stall polars, Cm(alpha,delta), -Cp_min cavitation
  - Full 3D banked WIG aerodynamics, sideslip/anhedral moments, PAR cushion
  - 4x 2-axis tilting lift rotors (Pitt-Peters + flapping) + J-rolloff pusher
  - 15-state ES-EKF + FMCW radar / boresight raycast / pressure-tap DSP
  - Pure-NumPy < 1 ms 200 Hz FastFlightCore + 7-scenario Monte Carlo suite
=============================================================================
"""

import argparse
import atexit
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
    """Mirrors terminal stdout to disk, safely handling interpreter shutdown."""
    def __init__(self, filename="flight_training_telemetry.txt"):
        self.terminal = sys.stdout
        self.log_file = open(filename, "w", encoding="utf-8")
        self.log_file.write(f"=== AEROMARINE AUTONOMY TELEMETRY SESSION INITIALIZED: {time.ctime()} ===\n\n")
        self.log_file.flush()

    def write(self, message):
        self.terminal.write(message)
        if self.log_file is not None and not self.log_file.closed:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            if self.log_file is not None and not self.log_file.closed:
                self.log_file.flush()
        except Exception:
            pass

    def close(self):
        try:
            if self.log_file is not None and not self.log_file.closed:
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
    for g in opt.param_groups:
        g["lr"] = lr
        g["initial_lr"] = lr
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, t_max), eta_min=eta_min)


# =============================================================================
# GLOBAL PHYSICAL, HYDRODYNAMIC & VEHICLE CONSTANTS
# =============================================================================
GRAVITY = 9.80665
RHO_AIR = 1.225
MU_AIR = 1.81e-5
RHO_WATER_NOM = 1025.0
MU_WATER = 1.08e-3
P_ATM = 101325.0
P_VAPOR_WATER = 2338.0
SIGMA_CAV_CRIT = 0.85
STRUT_YIELD_LIMIT_NM = 12.0

VEHICLE_MASS_NOM = 2.45
HOVER_WEIGHT_NOM = VEHICLE_MASS_NOM * GRAVITY

WING_SPAN_NOM = 0.92
WING_CHORD_NOM = 0.22
WING_AREA_NOM = WING_SPAN_NOM * WING_CHORD_NOM
WING_CL0 = 0.24
WING_CD0 = 0.022
WING_OSWALD = 0.82

FOIL_SPAN_NOM = 0.32
FOIL_CHORD_NOM = 0.042
FOIL_AREA_NOM = FOIL_SPAN_NOM * FOIL_CHORD_NOM
FOIL_STRUT_LENGTH_NOM = 0.25
FOIL_CL_ALPHA = 4.50
FOIL_CD0 = 0.012
HYDRO_SLAM_COEFF = 2.2
FOIL_LIFT_CLAMP_PER_SIDE = 16.0
FOIL_DECALAGE_DEG = (1.1, 1.4)
INIT_PITCH_NOSE_DOWN_DEG = 0.0

HYDRO_ELEVON_CHORD_FRAC = 0.30
HYDRO_CL_DELTA = 2.60
HYDRO_CD_DELTA = 0.35
HYDRO_RUDDER_AREA_NOM = 0.0075
HYDRO_RUDDER_CY_DELTA = 2.40
MAX_HYDRO_DEFLECTION = math.radians(15.0)
SERVO_RATE_LIMIT = 8.0
SERVO_BACKLASH_RAD = math.radians(0.3)

HULL_LENGTH = 0.65
HULL_BEAM = 0.18
HULL_DRAFT_MAX = 0.08
HULL_VOLUME_TOTAL = 0.0036
BOW_CENTER_OFFSET_X = 0.12

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

ROTOR_MAX_TILT_DEG = 15.0
ROTOR_MAX_TILT_RAD = math.radians(ROTOR_MAX_TILT_DEG)
ROTOR_MAX_TILT_SIN = math.sin(ROTOR_MAX_TILT_RAD)
ROTOR_MAX_TILT_COS = math.cos(ROTOR_MAX_TILT_RAD)
ROTOR_MAX_TILT_TAN = math.tan(ROTOR_MAX_TILT_RAD)
GIMBAL_TAU = 0.025

MAX_PUSHER_THRUST = 14.0
PUSHER_RADIUS = 0.085
PUSHER_DISK_AREA = math.pi * (PUSHER_RADIUS**2)
PUSHER_PROP_EFF = 0.72
PUSHER_V_PITCH_ZERO = 16.5

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

RIDE_HEIGHT_OPTIMAL = 0.185
RIDE_HEIGHT_SOARING = 0.165
RIDE_HEIGHT_REPRIME = 0.160
RIDE_HEIGHT_MIN = 0.060
RIDE_HEIGHT_FOILING_MAX = 0.235
RIDE_HEIGHT_MAX = 0.235
RIDE_HEIGHT_WIG_MAX = 0.380
FREE_FLIGHT_ALTITUDE = 1.500
DSWP_LOOKAHEAD_TIME = 1.250
CONTROL_LOOP_DT = 0.005
FORWARD_RAY_DEPRESSION_RAD = math.radians(14.0)

REF_DT = 0.015
GUIDANCE_DELAY_SECONDS = 0.030

PAR_PLENUM_EFF = 0.68
PAR_SEAL_GAIN = 1.95
SOARING_KATZMAYR_COEFF = 0.82

SCULL_FREQ_NOM = 22.0
SCULL_AMP_NOM = math.radians(18.0)
SCULL_LEVER_ARM = 0.090
SCULL_ACTUATOR_EFF = 0.68
SCULL_SERVO_IDLE_WATTS = 2.5

MODE_LOITER = 0
MODE_TAKEOFF = 1
MODE_FOILING = 2
MODE_EGRESS = 3
MODE_FREE_FLIGHT = 4
MODE_INGRESS = 5
MODE_SOARING = 6

MAG_FIELD_WORLD = torch.tensor([0.42, 0.00, -0.90], device=DEVICE).view(1, 3)


def safe_norm(x, dim=-1, keepdim=True, eps=1e-8):
    return torch.sqrt(torch.sum(x * x, dim=dim, keepdim=keepdim) + eps)


def quat_to_rot(q):
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    row0 = torch.cat([1.0 - 2.0 * (y**2 + z**2), 2.0 * (x*y - z*w), 2.0 * (x*z + y*w)], dim=-1)
    row1 = torch.cat([2.0 * (x*y + z*w), 1.0 - 2.0 * (x**2 + z**2), 2.0 * (y*z - x*w)], dim=-1)
    row2 = torch.cat([2.0 * (x*z - y*w), 2.0 * (y*z + x*w), 1.0 - 2.0 * (x**2 + y**2)], dim=-1)
    return torch.stack([row0, row1, row2], dim=1)


def rot_to_quat(R):
    """Singularity-free Shepperd rotation-matrix-to-quaternion conversion (N, 3, 3) -> (N, 4)."""
    r00, r01, r02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    r10, r11, r12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    r20, r21, r22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    tr = r00 + r11 + r22

    s0 = 2.0 * torch.sqrt(torch.clamp(1.0 + tr, min=1e-6))
    q0 = torch.stack([0.25 * s0, (r21 - r12) / s0, (r02 - r20) / s0, (r10 - r01) / s0], dim=-1)

    s1 = 2.0 * torch.sqrt(torch.clamp(1.0 + r00 - r11 - r22, min=1e-6))
    q1 = torch.stack([(r21 - r12) / s1, 0.25 * s1, (r01 + r10) / s1, (r02 + r20) / s1], dim=-1)

    s2 = 2.0 * torch.sqrt(torch.clamp(1.0 + r11 - r00 - r22, min=1e-6))
    q2 = torch.stack([(r02 - r20) / s2, (r01 + r10) / s2, 0.25 * s2, (r12 + r21) / s2], dim=-1)

    s3 = 2.0 * torch.sqrt(torch.clamp(1.0 + r22 - r00 - r11, min=1e-6))
    q3 = torch.stack([(r10 - r01) / s3, (r02 + r20) / s3, (r12 + r21) / s3, 0.25 * s3], dim=-1)

    q = torch.where((r11 > r22).unsqueeze(-1), q2, q3)
    q = torch.where(((r00 > r11) & (r00 > r22)).unsqueeze(-1), q1, q)
    q = torch.where((tr > 0.0).unsqueeze(-1), q0, q)
    q = q / safe_norm(q, dim=-1, keepdim=True)
    return torch.where(q[:, 0:1] < 0.0, -q, q)


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1[:, 0:1], q1[:, 1:2], q1[:, 2:3], q1[:, 3:4]
    w2, x2, y2, z2 = q2[:, 0:1], q2[:, 1:2], q2[:, 2:3], q2[:, 3:4]
    return torch.cat([
        w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)


def euler_to_rot(phi, theta, psi):
    cp, sp = torch.cos(phi), torch.sin(phi)
    ct, st = torch.cos(theta), torch.sin(theta)
    cy, sy = torch.cos(psi), torch.sin(psi)
    row0 = torch.cat([cy * ct, cy * st * sp - sy * cp, cy * st * cp + sy * sp], dim=-1)
    row1 = torch.cat([sy * ct, sy * st * sp + cy * cp, sy * st * cp - cy * sp], dim=-1)
    row2 = torch.cat([-st, ct * sp, ct * cp], dim=-1)
    return torch.stack([row0, row1, row2], dim=1)


def eval_foil_polar(alpha, delta_flap, Re, is_water=True):
    """Smooth Re-dependent low-Re + Viterna post-stall polar returning (cl, cd, cm_aero, neg_cp_min).
    Note: cm_aero follows standard aerodynamic nose-up-positive convention."""
    delta_t = torch.as_tensor(delta_flap, dtype=alpha.dtype, device=alpha.device)
    if is_water:
        cl0, cl_a, cm0 = 0.0, FOIL_CL_ALPHA, 0.0
        a_eff = alpha + (HYDRO_CL_DELTA / FOIL_CL_ALPHA) * delta_t
        a_stall = math.radians(10.5) + math.radians(2.5) * torch.clamp((Re - 8.0e4) / 2.2e5, 0.0, 1.0)
        cd_min = 0.011 * torch.pow((2.0e5 / Re.clamp(min=2.0e4)), 0.20)
        neg_cp_min = 0.35 + 2.8 * torch.abs(a_eff) + 1.8 * torch.abs(delta_t)
    else:
        cl0, cl_a, cm0 = WING_CL0, 4.5, -0.045
        a_eff = alpha
        a_stall = math.radians(10.0) + math.radians(2.5) * torch.clamp((Re - 4.0e4) / 1.2e5, 0.0, 1.0)
        cd_min = 0.018 * torch.pow((1.0e5 / Re.clamp(min=1.5e4)), 0.20)
        neg_cp_min = 0.40 + 2.5 * torch.abs(a_eff)

    s_att = torch.sigmoid(18.0 * (a_stall - torch.abs(a_eff)))
    cl_att = cl0 + cl_a * a_eff
    cl_post = 1.05 * torch.sin(2.0 * a_eff)
    cl = s_att * cl_att + (1.0 - s_att) * cl_post

    cd_att = cd_min + 0.045 * ((cl_att - cl0)**2) + (HYDRO_CD_DELTA * (delta_t**2) if is_water else 0.0)
    cd_post = 0.05 + 1.85 * (torch.sin(a_eff)**2)
    cd = s_att * cd_att + (1.0 - s_att) * cd_post
    cm = cm0 - 0.12 * (1.0 - s_att) * torch.sin(a_eff) - (0.55 * delta_t if is_water else 0.0)
    return cl, cd, cm, neg_cp_min


# =============================================================================
# FRONTIER IDEA E: MORPHOGENETIC AIRFRAME CO-EVOLUTION GENOME
# =============================================================================
class VehicleMorphologyGenome(nn.Module):
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
        foil_area = foil_span * FOIL_CHORD_NOM
        aspect_ratio = (wing_span**2) / wing_area
        total_mass = 1.65 + 2.85 * (wing_area + foil_area + caudal_area + 2.0 * strut_len * 0.04)

        return {
            "wing_span": wing_span, "wing_chord": wing_chord, "wing_area": wing_area,
            "aspect_ratio": aspect_ratio, "foil_span": foil_span, "foil_chord": FOIL_CHORD_NOM,
            "foil_area": foil_area, "strut_length": strut_len, "caudal_area": caudal_area,
            "anhedral_rad": anhedral_rad, "total_mass": total_mass
        }


# =============================================================================
# 1. 32-WAVE OCEAN (2ND-ORDER STOKES + STOKES DRIFT + STRIP EVALUATION)
# =============================================================================
class DirectionalOceanWaveField:
    def __init__(self, dt, airspeed=5.20, num_waves=32, device=DEVICE):
        self.dt, self.device, self.V, self.num_waves = dt, device, airspeed, num_waves
        freqs = torch.linspace(0.45, 2.40, num_waves, device=device)
        df = freqs[1] - freqs[0]
        s_pm = (0.0081 * (GRAVITY**2) / (freqs**5)) * torch.exp(-0.74 * ((GRAVITY / (self.V * freqs))**4))
        self.amplitudes_base = torch.sqrt(2.0 * s_pm * df).view(1, num_waves, 1)
        self.k = (freqs**2 / GRAVITY).view(1, num_waves, 1)
        self.omega = freqs.view(1, num_waves, 1)
        self.angles = torch.linspace(-0.35, 0.35, num_waves, device=device).view(1, num_waves, 1)
        self.phases = (torch.rand(num_waves, device=device) * (2.0 * math.pi)).view(1, num_waves, 1)
        self.kx = self.k * torch.cos(self.angles)
        self.ky = self.k * torch.sin(self.angles)

        self.a_w = math.exp(-self.dt * self.V / 4.0)
        self.b_w = (0.08 * self.V) * math.sqrt(1.0 - self.a_w**2)
        self.a_g = math.exp(-self.dt / 0.18)
        self.b_g = 0.08 * math.sqrt((1.0 - self.a_g**2) / (1.0 - 0.92**2))
        self.gust_state = torch.zeros((1, 3), device=device)
        self.ray_body_vec = torch.tensor(
            [math.cos(FORWARD_RAY_DEPRESSION_RAD), 0.0, -math.sin(FORWARD_RAY_DEPRESSION_RAD)], device=device
        ).view(1, 3, 1)

    def reset(self, num_envs):
        self.gust_state = torch.zeros((num_envs, 3), device=self.device)
        self.phases = torch.rand((num_envs, self.num_waves, 1), device=self.device) * (2.0 * math.pi)

    def step_gust(self):
        wn = torch.randn((self.gust_state.shape[0], 3), device=self.device)
        self.gust_state = torch.cat([
            self.a_g * self.gust_state[:, 0:1] + self.b_g * wn[:, 0:1],
            self.a_g * self.gust_state[:, 1:2] + self.b_g * wn[:, 1:2],
            self.a_w * self.gust_state[:, 2:3] + self.b_w * wn[:, 2:3]
        ], dim=-1)
        return self.gust_state

    def evaluate_wave_kinematics(self, pos, t, wave_scale=1.0):
        px, py, pz = pos[:, 0:1].unsqueeze(1), pos[:, 1:2].unsqueeze(1), pos[:, 2:3]
        t_term = t.unsqueeze(1) if (torch.is_tensor(t) and t.ndim == 2) else t
        amps = self.amplitudes_base * wave_scale
        phase = self.kx * px + self.ky * py - self.omega * t_term + self.phases
        sin_p, cos_p = torch.sin(phase), torch.cos(phase)

        eta = torch.sum(amps * cos_p + 0.5 * self.k * (amps**2) * torch.cos(2.0 * phase), dim=1)
        deta_dx = -torch.sum(amps * self.kx * sin_p + self.k * self.kx * (amps**2) * torch.sin(2.0 * phase), dim=1)
        deta_dy = -torch.sum(amps * self.ky * sin_p + self.k * self.ky * (amps**2) * torch.sin(2.0 * phase), dim=1)
        deta_dt = torch.sum(amps * self.omega * sin_p + self.k * self.omega * (amps**2) * torch.sin(2.0 * phase), dim=1)

        decay_air = torch.exp(-self.k * F.relu(pz - eta).unsqueeze(1))
        w_air_updraft = torch.sum(amps * self.omega * sin_p * decay_air, dim=1)

        depth_sub = F.relu(eta - pz).unsqueeze(1)
        decay_w = torch.exp(-self.k * depth_sub)
        stokes_drift = self.omega * self.k * (amps**2) * torch.exp(-2.0 * self.k * depth_sub)
        v_water_orbital = torch.cat([
            torch.sum((amps * self.omega * cos_p * decay_w + stokes_drift) * torch.cos(self.angles), dim=1),
            torch.sum((amps * self.omega * cos_p * decay_w + stokes_drift) * torch.sin(self.angles), dim=1),
            torch.sum(amps * self.omega * sin_p * decay_w, dim=1)
        ], dim=-1)
        return eta, deta_dx, deta_dy, deta_dt, w_air_updraft, v_water_orbital

    def evaluate_wave_points(self, pts_3d, t, wave_scale=1.0):
        px = pts_3d[:, :, 0].unsqueeze(1)
        py = pts_3d[:, :, 1].unsqueeze(1)
        pz = pts_3d[:, :, 2]
        t_term = t.unsqueeze(1) if (torch.is_tensor(t) and t.ndim == 2) else t
        amps = self.amplitudes_base * wave_scale
        phase = self.kx * px + self.ky * py - self.omega * t_term + self.phases
        sin_p, cos_p = torch.sin(phase), torch.cos(phase)

        eta = torch.sum(amps * cos_p + 0.5 * self.k * (amps**2) * torch.cos(2.0 * phase), dim=1)
        deta_dt = torch.sum(amps * self.omega * sin_p + self.k * self.omega * (amps**2) * torch.sin(2.0 * phase), dim=1)

        depth_sub = F.relu(eta - pz).unsqueeze(1)
        decay_w = torch.exp(-self.k * depth_sub)
        stokes_drift = self.omega * self.k * (amps**2) * torch.exp(-2.0 * self.k * depth_sub)
        v_orb = torch.stack([
            torch.sum((amps * self.omega * cos_p * decay_w + stokes_drift) * torch.cos(self.angles), dim=1),
            torch.sum((amps * self.omega * cos_p * decay_w + stokes_drift) * torch.sin(self.angles), dim=1),
            torch.sum(amps * self.omega * sin_p * decay_w, dim=1)
        ], dim=-1)
        return eta, deta_dt, v_orb

    def raycast_forward_surface_pose(self, pos, rot, eta_nadir_true, t, wave_scale=1.0):
        ray_w = torch.matmul(rot, self.ray_body_vec).squeeze(-1)
        down_slope = (-ray_w[:, 2:3]).clamp(min=0.06)
        s = torch.clamp((pos[:, 2:3] - eta_nadir_true) / down_slope, min=0.25, max=6.0)
        t_term = t.unsqueeze(1) if (torch.is_tensor(t) and t.ndim == 2) else t
        amps = self.amplitudes_base * wave_scale

        eta_hit = eta_nadir_true
        for _ in range(2):
            p_hit = pos + s * ray_w
            phase = self.kx * p_hit[:, 0:1].unsqueeze(1) + self.ky * p_hit[:, 1:2].unsqueeze(1) - self.omega * t_term + self.phases
            eta_hit = torch.sum(amps * torch.cos(phase) + 0.5 * self.k * (amps**2) * torch.cos(2.0 * phase), dim=1)
            s = torch.clamp((pos[:, 2:3] - eta_hit) / down_slope, min=0.25, max=6.0)

        valid = (torch.rand_like(eta_hit) >= 0.03) & (ray_w[:, 2:3] < -0.05) & (s < 5.8)
        eta_forward_meas = torch.where(valid, eta_hit + torch.randn_like(eta_hit) * 0.025, torch.zeros_like(eta_hit))
        return eta_forward_meas, valid


# =============================================================================
# 2. 15-STATE AEROMARINE ERROR-STATE EKF (ES-EKF / MEKF)
# =============================================================================
class AeromarineErrorStateEKF:
    def __init__(self, num_envs, dt, device=DEVICE):
        self.num_envs, self.dt, self.device = num_envs, dt, device
        self.p_hat = torch.zeros((num_envs, 3), device=device)
        self.v_hat = torch.zeros((num_envs, 3), device=device)
        self.q_hat = torch.zeros((num_envs, 4), device=device)
        self.q_hat[:, 0] = 1.0
        self.ba_hat = torch.zeros((num_envs, 3), device=device)
        self.bw_hat = torch.zeros((num_envs, 3), device=device)
        self._init_cov(num_envs)
        self.Q_d = torch.eye(15, device=device).unsqueeze(0).repeat(num_envs, 1, 1)
        self.Q_d[:, 0:3, 0:3] *= (0.08 * (dt**2))
        self.Q_d[:, 3:6, 3:6] *= (0.08 * dt)
        self.Q_d[:, 6:9, 6:9] *= (0.015 * dt)
        self.Q_d[:, 9:12, 9:12] *= (2.0e-5 * dt)
        self.Q_d[:, 12:15, 12:15] *= (1.0e-6 * dt)

    def _init_cov(self, N):
        self.P = torch.eye(15, device=self.device).unsqueeze(0).repeat(N, 1, 1) * 0.02
        self.P[:, 6:9, 6:9] = torch.eye(3, device=self.device).unsqueeze(0) * 0.002
        self.P[:, 9:12, 9:12] = torch.eye(3, device=self.device).unsqueeze(0) * 0.001
        self.P[:, 12:15, 12:15] = torch.eye(3, device=self.device).unsqueeze(0) * 0.0001

    def reset(self, num_envs, init_p, init_v, init_rot):
        self.num_envs = num_envs
        self.p_hat, self.v_hat = init_p.clone(), init_v.clone()
        self.ba_hat = torch.zeros((num_envs, 3), device=self.device)
        self.bw_hat = torch.zeros((num_envs, 3), device=self.device)
        self.q_hat = rot_to_quat(init_rot)
        self._init_cov(num_envs)

    def detach_state(self):
        self.p_hat = self.p_hat.detach(); self.v_hat = self.v_hat.detach(); self.q_hat = self.q_hat.detach()
        self.ba_hat = self.ba_hat.detach(); self.bw_hat = self.bw_hat.detach(); self.P = self.P.detach()

    def propagate(self, accel_meas, gyro_meas):
        device_str = "cuda" if accel_meas.is_cuda else "cpu"
        with get_amp_autocast(device_str, enabled=False):
            N, dt = self.num_envs, self.dt
            a_b, w_b = (accel_meas - self.ba_hat).float(), (gyro_meas - self.bw_hat).float()
            R_hat = quat_to_rot(self.q_hat.float())
            om = safe_norm(w_b, dim=-1, keepdim=True).clamp(min=1e-6)
            da = om * dt
            dq = torch.cat([torch.cos(0.5 * da), (w_b / om) * torch.sin(0.5 * da)], dim=-1)
            self.q_hat = quat_mult(self.q_hat.float(), dq)
            self.q_hat = self.q_hat / safe_norm(self.q_hat, dim=-1, keepdim=True)
            grav = torch.tensor([0.0, 0.0, -GRAVITY], device=self.device, dtype=torch.float32).view(1, 3).repeat(N, 1)
            a_w = torch.matmul(R_hat, a_b.unsqueeze(-1)).squeeze(-1) + grav
            self.p_hat = self.p_hat.float() + self.v_hat.float() * dt + 0.5 * a_w * (dt**2)
            self.v_hat = self.v_hat.float() + a_w * dt

            Phi = torch.eye(15, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)
            Phi[:, 0:3, 3:6] = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0) * dt
            ax, ay, az = a_b[:, 0:1], a_b[:, 1:2], a_b[:, 2:3]
            zv = torch.zeros_like(ax)
            skew_a = torch.stack([torch.cat([zv, -az, ay], dim=-1), torch.cat([az, zv, -ax], dim=-1), torch.cat([-ay, ax, zv], dim=-1)], dim=1)
            Phi[:, 3:6, 6:9] = -torch.matmul(R_hat, skew_a) * dt
            Phi[:, 3:6, 9:12] = -R_hat * dt
            Phi[:, 6:9, 6:9] = quat_to_rot(dq).transpose(-1, -2)
            Phi[:, 6:9, 12:15] = -torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0) * dt
            P_prop = torch.bmm(torch.bmm(Phi, self.P.float()), Phi.transpose(-1, -2)) + self.Q_d.float()
            self.P = torch.nan_to_num(0.5 * (P_prop + P_prop.transpose(-1, -2)), nan=0.05, posinf=2.0, neginf=0.0).clamp(-2.0, 2.0)
            idx = torch.arange(15, device=self.device)
            self.P[:, idx, idx] = torch.clamp(self.P[:, idx, idx], min=1e-4, max=2.0)

    def update_measurement(self, y, H, R_cov, active_mask=None):
        device_str = "cuda" if y.is_cuda else "cpu"
        with get_amp_autocast(device_str, enabled=False):
            y, H, R_cov, self.P = y.float(), H.float(), R_cov.float(), self.P.float()
            N = self.num_envs
            P_HT = torch.bmm(self.P, H.transpose(-1, -2))
            S_reg = torch.bmm(H, P_HT) + R_cov + torch.eye(H.shape[1], device=self.device, dtype=torch.float32).unsqueeze(0) * 1e-5
            K = torch.linalg.solve(S_reg, P_HT.transpose(-1, -2)).transpose(-1, -2)
            dx = torch.nan_to_num(torch.bmm(K, y.unsqueeze(-1)).squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0)
            if active_mask is not None:
                dx = torch.where(active_mask.bool().view(N, 1), dx, torch.zeros_like(dx))

            self.p_hat = self.p_hat.float() + dx[:, 0:3]
            self.v_hat = self.v_hat.float() + dx[:, 3:6]
            dq_corr = torch.cat([torch.ones((N, 1), device=self.device, dtype=torch.float32), 0.5 * torch.clamp(dx[:, 6:9], -0.05, 0.05)], dim=-1)
            self.q_hat = quat_mult(self.q_hat.float(), dq_corr)
            self.q_hat = self.q_hat / safe_norm(self.q_hat, dim=-1, keepdim=True)
            self.ba_hat = torch.clamp(self.ba_hat.float() + dx[:, 9:12], -0.08, 0.08)
            self.bw_hat = torch.clamp(self.bw_hat.float() + dx[:, 12:15], -0.03, 0.03)

            I_KH = torch.eye(15, device=self.device, dtype=torch.float32).unsqueeze(0) - torch.bmm(K, H)
            P_new = torch.bmm(torch.bmm(I_KH, self.P), I_KH.transpose(-1, -2)) + torch.bmm(torch.bmm(K, R_cov), K.transpose(-1, -2))
            P_sym = 0.5 * (P_new + P_new.transpose(-1, -2))
            self.P = (torch.where(active_mask.bool().view(N, 1, 1), P_sym, self.P) if active_mask is not None else P_sym).clamp(-2.0, 2.0)
            self.P = torch.nan_to_num(self.P, nan=0.05, posinf=2.0, neginf=0.0)
            idx = torch.arange(15, device=self.device)
            self.P[:, idx, idx] = torch.clamp(self.P[:, idx, idx], min=1e-4, max=2.0)

    def update_magnetometer(self, mag_meas_body):
        N = self.num_envs
        R_hat = quat_to_rot(self.q_hat)
        m_pred_b = torch.bmm(R_hat.transpose(-1, -2), MAG_FIELD_WORLD.to(self.device).repeat(N, 1).unsqueeze(-1)).squeeze(-1)
        mx, my, mz = m_pred_b[:, 0:1], m_pred_b[:, 1:2], m_pred_b[:, 2:3]
        zv = torch.zeros_like(mx)
        H = torch.zeros((N, 3, 15), device=self.device, dtype=torch.float32)
        H[:, :, 6:9] = torch.stack([torch.cat([zv, -mz, my], dim=-1), torch.cat([mz, zv, -mx], dim=-1), torch.cat([-my, mx, zv], dim=-1)], dim=1)
        self.update_measurement(mag_meas_body - m_pred_b, H, torch.eye(3, device=self.device).unsqueeze(0).repeat(N, 1, 1) * 0.04)

    def update_dvl(self, dvl_meas_body, is_foil_submerged, dvl_lost_track):
        N = self.num_envs
        R_hat = quat_to_rot(self.q_hat)
        v_pred_b = torch.bmm(R_hat.transpose(-1, -2), self.v_hat.unsqueeze(-1)).squeeze(-1)
        H = torch.zeros((N, 3, 15), device=self.device, dtype=torch.float32)
        H[:, :, 3:6] = R_hat.transpose(-1, -2).float()
        vx, vy, vz = v_pred_b[:, 0:1], v_pred_b[:, 1:2], v_pred_b[:, 2:3]
        zv = torch.zeros_like(vx)
        H[:, :, 6:9] = torch.stack([torch.cat([zv, -vz, vy], dim=-1), torch.cat([vz, zv, -vx], dim=-1), torch.cat([-vy, vx, zv], dim=-1)], dim=1)
        self.update_measurement(dvl_meas_body - v_pred_b, H, torch.eye(3, device=self.device).unsqueeze(0).repeat(N, 1, 1) * 0.10, active_mask=(is_foil_submerged.float() * (1.0 - dvl_lost_track.float())).squeeze(-1))

    def update_optical_flow(self, flow_rates_body, clearance_est, celerity_water, is_airborne):
        N = self.num_envs
        R_T = quat_to_rot(self.q_hat).transpose(-1, -2)
        h_c = torch.clamp(clearance_est, min=0.08, max=3.0)
        u_b = torch.bmm(R_T, self.v_hat.unsqueeze(-1)).squeeze(-1)
        flow_pred = torch.cat([-(u_b[:, 1:2]) / h_c, +(u_b[:, 0:1] - celerity_water) / h_c], dim=-1)
        H = torch.zeros((N, 2, 15), device=self.device, dtype=torch.float32)
        inv_h = (1.0 / h_c).view(N, 1)
        H[:, 0, 3:6], H[:, 1, 3:6] = -inv_h * R_T[:, 1, :], +inv_h * R_T[:, 0, :]
        ux, uy, uz = u_b[:, 0:1], u_b[:, 1:2], u_b[:, 2:3]
        zu = torch.zeros_like(ux)
        H[:, 0, 6:9], H[:, 1, 6:9] = -inv_h * torch.cat([uz, zu, -ux], dim=-1), +inv_h * torch.cat([zu, -uz, uy], dim=-1)
        self.update_measurement((flow_rates_body - flow_pred).float(), H, torch.eye(2, device=self.device).unsqueeze(0).repeat(N, 1, 1) * 0.08, active_mask=(is_airborne.float() * (clearance_est < 1.8).float()).squeeze(-1))

    def update_gnss(self, gnss_pos_w, gnss_vel_w, clearance_est):
        N = self.num_envs
        H_pos, H_vel = torch.zeros((N, 3, 15), device=self.device, dtype=torch.float32), torch.zeros((N, 3, 15), device=self.device, dtype=torch.float32)
        H_pos[:, 0:3, 0:3] = torch.eye(3, device=self.device).unsqueeze(0)
        H_vel[:, 0:3, 3:6] = torch.eye(3, device=self.device).unsqueeze(0)
        infl = 1.0 + 8.0 * torch.exp(-torch.clamp(clearance_est, min=0.01, max=10.0) / 0.25).unsqueeze(-1)
        eye3 = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)
        self.update_measurement(gnss_pos_w - self.p_hat, H_pos, eye3 * (0.0025 * infl))
        self.update_measurement(gnss_vel_w - self.v_hat, H_vel, eye3 * (0.0009 * infl))


# =============================================================================
# 3. SPRAY-ADAPTIVE RADAR & HYDROSTATIC PERCEPTION SUITE
# =============================================================================
class HardwarePerceptionSuite:
    def __init__(self, dt, device):
        self.dt, self.device = dt, device
        self.cant_angle = math.radians(22.0)
        self.radar_num_bins, self.radar_r_max = 128, 4.50
        self.radar_dr = self.radar_r_max / (self.radar_num_bins - 1)
        self.airframe_exclusion_m, self.cfar_threshold_factor = 0.135, 2.15
        self.bins = torch.linspace(0.0, self.radar_r_max, self.radar_num_bins, device=device).view(1, -1)
        self.p_airframe = 32.0 * torch.exp(-((self.bins - 0.092)**2) / (2 * (0.018**2))) + 24.0 * torch.exp(-((self.bins - 0.118)**2) / (2 * (0.015**2)))
        self.spatial_mask = (self.bins > self.airframe_exclusion_m).float()

    def process_fmcw_radar_dsp(self, true_clearance, deta_dx, deta_dy, fwd_speed, vz_speed):
        N = true_clearance.shape[0]
        crest_bias = torch.sqrt(deta_dx**2 + deta_dy**2 + 1e-6) * (true_clearance * 0.140 * 0.40)
        true_range_water = (true_clearance - crest_bias).clamp(min=0.03, max=self.radar_r_max - 0.05)

        spray_dropout = (torch.rand_like(true_clearance) < 0.025).float()
        p_water = (42.0 * (1.0 - spray_dropout)).clamp(min=2.0) * torch.exp(-((self.bins - true_range_water.view(N, 1))**2) / (2 * (0.028**2)))
        p_noise = torch.randn((N, self.radar_num_bins), device=self.device) * 1.5 + 8.0
        raw_fft = self.p_airframe + p_water + p_noise
        doppler_att = torch.where((torch.abs(fwd_speed * math.sin(self.cant_angle * 0.1) - vz_speed) < 0.02) & (fwd_speed < 0.3), 0.40, 1.0)
        cleaned = F.relu((raw_fft * self.spatial_mask) * doppler_att - self.p_airframe)

        peak_idx = torch.argmax(cleaned, dim=-1, keepdim=True)
        y_l = torch.gather(cleaned, -1, (peak_idx - 1).clamp(min=0))
        y_c = torch.gather(cleaned, -1, peak_idx)
        y_r = torch.gather(cleaned, -1, (peak_idx + 1).clamp(max=self.radar_num_bins - 1))
        denom = y_l - 2.0 * y_c + y_r
        offset = torch.where(denom.abs() > 1e-6, 0.5 * (y_l - y_r) / denom, torch.zeros_like(denom)).clamp(-0.5, 0.5)
        detected_clearance = (peak_idx.float() + offset) * self.radar_dr

        cfar_floor = torch.mean(p_noise, dim=-1, keepdim=True)
        first_valid_bin = int(math.floor(self.airframe_exclusion_m / self.radar_dr)) + 1
        is_target_locked = (y_c > (cfar_floor * self.cfar_threshold_factor)) & (spray_dropout < 0.5) & (peak_idx > first_valid_bin)
        return detected_clearance, spray_dropout, is_target_locked, (peak_idx != torch.argmax(raw_fft, dim=-1, keepdim=True)).float()

    def sample_piezoresistive_tap(self, d_sub_physical, v_foil_rel_body_true, v_body_est, rho_water_true):
        device_str = "cuda" if self.device.type == "cuda" else "cpu"
        with get_amp_autocast(device_str, enabled=False):
            d_sub_physical, v_true_f, v_est_f = d_sub_physical.float(), v_foil_rel_body_true.float(), v_body_est.float()
            rho_fluid = torch.where(d_sub_physical > 0.005, rho_water_true, torch.full_like(rho_water_true, RHO_AIR))
            u_t, v_t, w_t = v_true_f[:, 0:1], v_true_f[:, 1:2], v_true_f[:, 2:3]
            q_t = (0.5 * rho_fluid * (u_t**2 + v_t**2 + w_t**2)).clamp(max=15000.0)
            cp_act = torch.clamp(0.045 + 0.420 * torch.abs(torch.atan2(w_t, u_t.clamp(min=0.2))) + 0.680 * torch.abs(torch.atan2(v_t, u_t.clamp(min=0.2))), 0.0, 0.80)
            p_dyn_err = cp_act * q_t
            p_meas_raw = float(P_ATM) + rho_water_true * GRAVITY * d_sub_physical + p_dyn_err + torch.randn_like(d_sub_physical) * 5.0

            u_e, v_e, w_e = v_est_f[:, 0:1], v_est_f[:, 1:2], v_est_f[:, 2:3]
            q_e = (0.5 * RHO_WATER_NOM * (u_e**2 + v_e**2 + w_e**2)).clamp(max=15000.0)
            cp_cal = torch.clamp((0.045 + 0.420 * torch.abs(torch.atan2(w_e, u_e.clamp(min=0.2))) + 0.680 * torch.abs(torch.atan2(v_e, u_e.clamp(min=0.2)))) * 1.05, 0.0, 0.85)
            p_static_dec = torch.minimum(p_meas_raw, p_meas_raw - (cp_cal * q_e))
            d_calc = F.relu((p_static_dec - float(P_ATM)) / (RHO_WATER_NOM * GRAVITY)).clamp(max=0.35)
            d_sub_hyd = torch.where(d_calc > 0.005, d_calc, torch.zeros_like(d_calc))
            return d_sub_hyd, p_static_dec, p_dyn_err, (d_sub_physical - d_sub_hyd) * 100.0


# =============================================================================
# 4. ONLINE LINEAR DISPERSION OBSERVER
# =============================================================================
class RealSeaWaveDispersionObserver:
    def __init__(self, dt, device):
        self.dt, self.device = dt, device
        self.reset(1)

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
        eta_nadir = (craft_z.detach() - clearance_est.detach()).clamp(-0.6, 0.6)
        self.last_eta_nadir = eta_nadir
        slope = ((eta_forward_meas.detach() - eta_nadir) / torch.clamp(lookahead_dist.detach(), min=0.25)).clamp(-0.6, 0.6)

        dt_obs = max(self.dt, REF_DT)
        a_ms = math.exp(-dt_obs / 0.45)
        self.ms_eta = torch.where(valid, a_ms * self.ms_eta + (1.0 - a_ms) * (eta_nadir**2), self.ms_eta)
        self.ms_slope = torch.where(valid, a_ms * self.ms_slope + (1.0 - a_ms) * (slope**2), self.ms_slope)

        k_est = torch.sqrt(self.ms_slope / (self.ms_eta + 0.002))
        a_k = math.exp(-dt_obs / 0.25)
        self.k_dom = torch.where(valid, torch.clamp(a_k * self.k_dom + (1.0 - a_k) * k_est, min=0.08, max=3.5), self.k_dom)
        self.omega_dom = torch.sqrt(GRAVITY * self.k_dom)

        a_q = math.exp(-dt_obs / 0.15)
        self.q_quad = torch.where(valid, a_q * self.q_quad + (1.0 - a_q) * (-slope / self.k_dom).clamp(-0.4, 0.4), self.q_quad)
        self.deta_dx_nadir = -self.k_dom * self.q_quad
        self.w_up_nadir = self.omega_dom * self.q_quad

        delta = self.k_dom * (fwd_speed.detach() * preview_time) - self.omega_dom * preview_time
        c_d, s_d = torch.cos(delta), torch.sin(delta)
        eta_prev = eta_nadir * c_d - self.q_quad * s_d
        quad_ahead = self.q_quad * c_d + eta_nadir * s_d
        return eta_prev, -self.k_dom * quad_ahead, self.omega_dom * quad_ahead


# =============================================================================
# 5. REAL-WORLD MULTIMODAL AEROHYDRODYNAMICS (STRIP THEORY, ADDED MASS, POLARS)
# =============================================================================
class MultimodalAeroHydroDynamics(nn.Module):
    def __init__(self, device, dt=0.015, harsh_physics=False, fixed_foil=False, disable_scull=False):
        super().__init__()
        self.device, self.dt = device, dt
        self.harsh_physics, self.fixed_foil, self.disable_scull = harsh_physics, fixed_foil, disable_scull
        self.n_foil_strips = 8
        self.n_strut_strips = 5
        self.y_frac_f = torch.linspace(-0.5 + 0.5 / 8, 0.5 - 0.5 / 8, 8, device=device).view(1, 8)
        self.z_frac_s = torch.linspace(0.1, 0.9, 5, device=device).view(1, 5)
        self.cl_lag_s = torch.zeros((1, 8), device=device)

    def reset(self, num_envs):
        self.cl_lag_s = torch.zeros((num_envs, self.n_foil_strips), device=self.device)

    def detach_state(self):
        self.cl_lag_s = self.cl_lag_s.detach()

    def forward(self, pos, vel, rot, omega, eta, deta_dt, w_air_updraft, v_water_orbital, ambient_gust,
                is_vent_L, is_vent_R, foil_decalage, rho_water, hydro_surfaces, motor_thrusts_total,
                morphology_dict, sim_time, is_soaring, scull_throttle=None, wave_field=None, wave_scale=1.0):
        N = pos.shape[0]
        if self.cl_lag_s.shape[0] != N:
            self.reset(N)

        w_span, w_chord, w_area = morphology_dict["wing_span"], morphology_dict["wing_chord"], morphology_dict["wing_area"]
        f_span, f_area, strut_len = morphology_dict["foil_span"], morphology_dict["foil_area"], morphology_dict["strut_length"]
        caudal_area, anhedral_rad = morphology_dict["caudal_area"], morphology_dict["anhedral_rad"]

        # 1. Full 3D WIG aerodynamics + sideslip + PAR cushion
        h_wing = (pos[:, 2:3] - eta).clamp(min=0.03)
        wind_shear = torch.clamp(torch.log((h_wing + 0.0002) / 0.0002) / math.log(10.0 / 0.0002), 0.18, 1.25)
        ambient_wind_local = ambient_gust * wind_shear

        phi_lift_wig = (1.0 + 0.35 / (0.10 + 2.2 * torch.pow(h_wing / w_chord, 1.15))) * (1.0 + 1.2 * torch.clamp(anhedral_rad, min=0.0))
        sigma_drag_wig = (1.0 - torch.exp(-2.48 * torch.pow(2.0 * (h_wing / w_span), 0.78))).clamp(0.08, 1.0)
        downwash_vel = torch.sqrt(torch.clamp(motor_thrusts_total / (2.0 * RHO_AIR * TOTAL_ROTOR_AREA), min=0.0))
        par_pressure = (PAR_PLENUM_EFF * 0.5 * RHO_AIR * (downwash_vel**2) * F.relu(1.0 - h_wing / 0.25)).clamp(max=120.0)
        f_par_z = par_pressure * w_area

        v_air_eff = vel - ambient_wind_local - torch.cat([torch.zeros((N, 2), device=self.device), w_air_updraft], dim=-1)
        v_air_mag = safe_norm(v_air_eff, dim=-1, keepdim=True).clamp(min=0.5)
        q_air = 0.5 * RHO_AIR * (v_air_mag**2)
        rot_T = rot.transpose(-1, -2)
        v_air_body = torch.matmul(rot_T, v_air_eff.unsqueeze(-1)).squeeze(-1)

        Re_air = (RHO_AIR * v_air_mag * w_chord / MU_AIR).clamp(1.5e4, 5.0e5)
        alpha_a = torch.atan2(-v_air_body[:, 2:3], v_air_body[:, 0:1].clamp(min=0.5))
        alpha_wing = alpha_a + 0.03
        beta_wing = torch.atan2(v_air_body[:, 1:2], safe_norm(v_air_body[:, [0, 2]], dim=-1, keepdim=True).clamp(min=0.5))

        cl_w_2d, cd_w_2d, cm_w, _ = eval_foil_polar(alpha_wing, 0.0, Re_air, is_water=False)
        cl_wing = cl_w_2d * phi_lift_wig
        cd_wing = cd_w_2d + ((cl_wing**2) / (math.pi * morphology_dict["aspect_ratio"] * WING_OSWALD)) * sigma_drag_wig

        L_wing = q_air * w_area * cl_wing
        D_wing = q_air * w_area * cd_wing
        f_w_bx = L_wing * torch.sin(alpha_a) - D_wing * torch.cos(alpha_a)
        f_w_by = -q_air * w_area * 0.65 * beta_wing
        f_w_bz = L_wing * torch.cos(alpha_a) + D_wing * torch.sin(alpha_a) + f_par_z
        f_wing_world = torch.matmul(rot, torch.cat([f_w_bx, f_w_by, f_w_bz], dim=-1).unsqueeze(-1)).squeeze(-1)

        # In +X fwd, +Y port/left, +Z up: +tau_y is nose-DOWN, +tau_x is roll-right (away from +Y wind)
        tau_aero_pitch = q_air * w_area * w_chord * (+0.85 * alpha_wing - cm_w - 2.8 * (omega[:, 1:2] * w_chord / (2.0 * v_air_mag)))
        tau_aero_roll = q_air * w_area * w_span * ((+0.08 - 0.85 * anhedral_rad) * beta_wing - 0.45 * (omega[:, 0:1] * w_span / (2.0 * v_air_mag)))
        tau_aero_yaw = q_air * w_area * w_span * (+0.04 * beta_wing - 0.18 * (omega[:, 2:3] * w_span / (2.0 * v_air_mag)))

        # 2. 8-strip spanwise hydrofoil + 5-strip depth strut + added mass
        y_s = self.y_frac_f * f_span
        ds_f = f_span / self.n_foil_strips
        r_f_loc = torch.stack([torch.zeros_like(y_s), y_s, -strut_len.expand_as(y_s)], dim=-1)
        p_f_w = pos.unsqueeze(1) + torch.matmul(r_f_loc, rot_T)

        if wave_field is not None:
            eta_f, deta_dt_f, v_orb_f = wave_field.evaluate_wave_points(p_f_w, sim_time, wave_scale=wave_scale)
        else:
            eta_f, deta_dt_f, v_orb_f = eta.expand(N, 8), deta_dt.expand(N, 8), v_water_orbital.unsqueeze(1).expand(N, 8, 3)

        omega_x_rf = torch.cross(omega.unsqueeze(1), r_f_loc.expand(N, -1, -1), dim=-1)
        v_f_w = vel.unsqueeze(1) + torch.matmul(omega_x_rf, rot_T)
        v_rel_f_b = torch.matmul(v_f_w - v_orb_f, rot)

        u_f, v_f, w_f = v_rel_f_b[:, :, 0], v_rel_f_b[:, :, 1], v_rel_f_b[:, :, 2]
        spd_f = torch.sqrt(u_f**2 + w_f**2 + 1e-4).clamp(min=0.1, max=20.0)
        q_w_f = (0.5 * rho_water * (spd_f**2)).clamp(max=45000.0 if self.harsh_physics else 30000.0)
        d_sub_s = (eta_f - p_f_w[:, :, 2]).clamp(min=0.0, max=0.22)

        if self.fixed_foil:
            delta_s = torch.zeros_like(d_sub_s)
            delta_rud = torch.zeros((N, 1), device=self.device)
        else:
            delta_s = torch.cat([hydro_surfaces[:, 1:2].expand(N, 4), hydro_surfaces[:, 0:1].expand(N, 4)], dim=-1)
            delta_rud = hydro_surfaces[:, 2:3]

        d_sub_L, d_sub_R = d_sub_s[:, 4:8].mean(dim=-1, keepdim=True), d_sub_s[:, 0:4].mean(dim=-1, keepdim=True)
        immersion_depth_mean = 0.5 * (d_sub_L + d_sub_R)
        q_w_mean = q_w_f.mean(dim=-1, keepdim=True)

        delta_vent_relief = torch.where(torch.maximum(is_vent_L, is_vent_R).bool(), -math.radians(2.0), 0.0)
        alpha_f_s = torch.atan2(-w_f, u_f.clamp(min=0.5)) + foil_decalage + delta_vent_relief
        alpha_base = alpha_f_s.mean(dim=-1, keepdim=True)

        Re_f = (rho_water * spd_f * FOIL_CHORD_NOM / MU_WATER).clamp(1.0e4, 2.0e6)
        cl_raw_s, cd_raw_s, cm_s, neg_cp_s = eval_foil_polar(alpha_f_s, delta_s, Re_f, is_water=True)

        suction_L = neg_cp_s[:, 4:8].mean(dim=-1, keepdim=True) * q_w_f[:, 4:8].mean(dim=-1, keepdim=True)
        suction_R = neg_cp_s[:, 0:4].mean(dim=-1, keepdim=True) * q_w_f[:, 0:4].mean(dim=-1, keepdim=True)
        seal_L = rho_water * GRAVITY * d_sub_L + 120.0 + PAR_SEAL_GAIN * par_pressure
        seal_R = rho_water * GRAVITY * d_sub_R + 120.0 + PAR_SEAL_GAIN * par_pressure
        is_air_L, is_air_R = (d_sub_L < 0.005), (d_sub_R < 0.005)
        alpha_L_m = (alpha_f_s[:, 4:8] + (HYDRO_CL_DELTA / FOIL_CL_ALPHA) * delta_s[:, 4:8]).mean(dim=-1, keepdim=True)
        alpha_R_m = (alpha_f_s[:, 0:4] + (HYDRO_CL_DELTA / FOIL_CL_ALPHA) * delta_s[:, 0:4]).mean(dim=-1, keepdim=True)

        is_vent_L_next = torch.where(is_air_L, torch.zeros_like(is_vent_L),
            torch.where(is_vent_L.bool(), torch.where((d_sub_L >= 0.040) & (alpha_L_m < math.radians(4.0)), torch.zeros_like(is_vent_L), torch.ones_like(is_vent_L)),
                        torch.where((suction_L > 2.2 * seal_L) & (d_sub_L < 0.030) & (alpha_L_m > math.radians(2.2)), torch.ones_like(is_vent_L), torch.zeros_like(is_vent_L))))
        is_vent_R_next = torch.where(is_air_R, torch.zeros_like(is_vent_R),
            torch.where(is_vent_R.bool(), torch.where((d_sub_R >= 0.040) & (alpha_R_m < math.radians(4.0)), torch.zeros_like(is_vent_R), torch.ones_like(is_vent_R)),
                        torch.where((suction_R > 2.2 * seal_R) & (d_sub_R < 0.030) & (alpha_R_m > math.radians(2.2)), torch.ones_like(is_vent_R), torch.zeros_like(is_vent_R))))

        vent_s = torch.cat([torch.where(is_vent_R_next.bool(), 0.40, 1.0).expand(N, 4), torch.where(is_vent_L_next.bool(), 0.40, 1.0).expand(N, 4)], dim=-1)
        fs_eff_s = 1.0 - 0.30 * torch.exp(-d_sub_s / (0.65 * FOIL_CHORD_NOM))

        sigma_cav_s = (P_ATM + rho_water * GRAVITY * d_sub_s - P_VAPOR_WATER) / (q_w_f.clamp(min=10.0) * (1.0 + neg_cp_s))
        sigma_cav_min = torch.min(sigma_cav_s, dim=-1, keepdim=True).values
        is_air_s = (d_sub_s < 0.005)

        if self.harsh_physics:
            cav_sev = torch.clamp((SIGMA_CAV_CRIT - sigma_cav_s) / 0.45, 0.0, 1.0) * (~is_air_s).float()
            cl_st_s = cl_raw_s * vent_s * fs_eff_s * (1.0 - 0.55 * cav_sev)
            cd_cav_s = 0.12 * cav_sev
            a_lag = self.dt / ((FOIL_CHORD_NOM / (0.5 * spd_f)).clamp(0.005, 0.08) + self.dt)
            cl_s = torch.where(is_air_s, cl_st_s, self.cl_lag_s + a_lag * (cl_st_s - self.cl_lag_s))
            self.cl_lag_s = torch.where(is_air_s, 0.5 * cl_st_s, cl_s)
        else:
            cl_s = cl_raw_s * vent_s * fs_eff_s
            cd_cav_s = torch.zeros_like(d_sub_s)

        cd_s = cd_raw_s + cd_cav_s
        sub_s = (d_sub_s / 0.040).clamp(0.0, 1.0)
        fwd_s = (u_f > 0.1).float()
        area_s = ds_f * FOIL_CHORD_NOM
        l_strip_cap = (26.0 if self.harsh_physics else FOIL_LIFT_CLAMP_PER_SIDE) / 4.0

        L_s = torch.clamp(q_w_f * area_s * sub_s * cl_s * fwd_s, -0.5 if self.harsh_physics else -0.15, l_strip_cap)
        D_s = q_w_f * area_s * sub_s * cd_s
        flow_ang_s = torch.atan2(-w_f, u_f.clamp(min=0.5))
        f_soar_s = torch.clamp(SOARING_KATZMAYR_COEFF * L_s * torch.sin(flow_ang_s.clamp(-0.5, 0.5)), -1.0, 2.25)

        scull_active = torch.zeros((N, 1), device=self.device) if (self.fixed_foil or self.disable_scull) else is_soaring.float() * (immersion_depth_mean > 0.020).float()
        scull_thr = scull_active if scull_throttle is None else scull_active * scull_throttle.clamp(0.0, 1.0)
        spd_mean = spd_f.mean(dim=-1, keepdim=True)
        strouhal_st = (2.0 * SCULL_FREQ_NOM * SCULL_LEVER_ARM * math.sin(SCULL_AMP_NOM) / spd_mean).clamp(max=2.0)
        strouhal_eff = torch.exp(-((strouhal_st - 0.30)**2) / (2.0 * (0.08**2))) * 0.84
        v_sc_tip = 2.0 * math.pi * SCULL_FREQ_NOM * SCULL_AMP_NOM * SCULL_LEVER_ARM
        q_sc = 0.5 * rho_water * (spd_mean**2 + (v_sc_tip**2) * scull_active)
        f_scull_thrust = torch.clamp(q_sc * caudal_area * HYDRO_RUDDER_CY_DELTA * math.sin(SCULL_AMP_NOM) * strouhal_eff * scull_active, 0.0, 22.0) * scull_thr
        p_scull_actuator_watts = torch.clamp(
            (f_scull_thrust * spd_mean) / SCULL_ACTUATOR_EFF + torch.where(scull_thr > 0.01, SCULL_SERVO_IDLE_WATTS, 0.0),
            0.0, 180.0
        )

        f_foil_bx = torch.sum(-D_s * torch.sign(u_f + 1e-4) + f_soar_s, dim=-1, keepdim=True) + f_scull_thrust
        f_foil_bz = torch.sum(L_s, dim=-1, keepdim=True)
        f_soar_surge = torch.sum(f_soar_s, dim=-1, keepdim=True)

        slam_cap, hull_cap = (350.0, 250.0) if self.harsh_physics else (25.0, 40.0)
        imm_rate_s = F.relu(deta_dt_f - v_f_w[:, :, 2]) * (d_sub_s > 0.001).float()
        f_slam_z = torch.clamp(torch.sum(0.5 * rho_water * HYDRO_SLAM_COEFF * ds_f * FOIL_CHORD_NOM * (imm_rate_s**2), dim=-1, keepdim=True), max=slam_cap)

        tau_foil_roll = torch.sum(y_s * L_s, dim=-1, keepdim=True)
        tau_foil_pitch = -torch.sum(q_w_f * area_s * FOIL_CHORD_NOM * sub_s * cm_s, dim=-1, keepdim=True) - f_foil_bx * strut_len

        # 5-strip depth strut + Hoerner spray drag
        ds_v = (strut_len - HULL_DRAFT_MAX).clamp(min=0.04) / self.n_strut_strips
        z_k = -HULL_DRAFT_MAX - self.z_frac_s * (strut_len - HULL_DRAFT_MAX).clamp(min=0.04)
        r_s_loc = torch.stack([torch.full_like(z_k, -0.08), torch.zeros_like(z_k), z_k], dim=-1)
        p_s_w = pos.unsqueeze(1) + torch.matmul(r_s_loc, rot_T)

        if wave_field is not None:
            eta_st, _, v_orb_st = wave_field.evaluate_wave_points(p_s_w, sim_time, wave_scale=wave_scale)
        else:
            eta_st, v_orb_st = eta.expand(N, 5), v_water_orbital.unsqueeze(1).expand(N, 5, 3)

        v_st_w = vel.unsqueeze(1) + torch.matmul(torch.cross(omega.unsqueeze(1), r_s_loc.expand(N, -1, -1), dim=-1), rot_T)
        v_rel_st_b = torch.matmul(v_st_w - v_orb_st, rot)
        u_st, v_st = v_rel_st_b[:, :, 0], v_rel_st_b[:, :, 1]
        spd_st = torch.sqrt(u_st**2 + v_st**2 + 1e-4).clamp(min=0.1, max=20.0)
        q_w_st = (0.5 * rho_water * (spd_st**2)).clamp(max=45000.0 if self.harsh_physics else 30000.0)
        w_frac_st = ((eta_st - p_s_w[:, :, 2]) / ds_v + 0.5).clamp(0.0, 1.0)
        strut_wet_len = torch.sum(w_frac_st, dim=-1, keepdim=True) * ds_v

        area_st_strip = caudal_area / 5.0 + 0.04 * ds_v
        fy_st = q_w_st * area_st_strip * w_frac_st * (-2.8 * (v_st / spd_st) + HYDRO_RUDDER_CY_DELTA * delta_rud * self.z_frac_s)
        fx_st = -q_w_st * area_st_strip * w_frac_st * 0.018 * torch.sign(u_st + 1e-4)

        is_piercing = ((strut_wet_len > 0.005) & (strut_wet_len < (strut_len - HULL_DRAFT_MAX - 0.005))).float()
        f_spray_x = -0.5 * rho_water * (spd_mean**2) * (0.012**2) * 0.24 * is_piercing * torch.sign(u_st.mean(dim=-1, keepdim=True) + 1e-4)

        lat_st_cap = 25.0 if self.harsh_physics else 8.0
        f_strut_lat_y = torch.clamp(torch.sum(fy_st, dim=-1, keepdim=True), -lat_st_cap, lat_st_cap)
        f_strut_drag_x = torch.sum(fx_st, dim=-1, keepdim=True) + f_spray_x
        tau_strut_roll = torch.sum(-z_k * fy_st, dim=-1, keepdim=True) + tau_foil_roll
        tau_strut_yaw = torch.sum(-0.08 * fy_st, dim=-1, keepdim=True)
        strut_root_moment = torch.abs(f_strut_lat_y * strut_len) + torch.abs(tau_foil_roll)

        # 3. Hull planing & buoyancy
        hull_imm = (eta - (pos[:, 2:3] - HULL_DRAFT_MAX)).clamp(0.0, HULL_DRAFT_MAX * 2.0)
        f_buoy_z = rho_water * GRAVITY * (hull_imm / HULL_DRAFT_MAX).clamp(max=1.5) * HULL_VOLUME_TOTAL
        f_planing_z = (0.5 * rho_water * (HULL_BEAM * HULL_LENGTH) * (spd_mean**2) * 0.14 * torch.sin(F.relu(alpha_wing + 0.02)) * (hull_imm > 0.005).float() * (u_f.mean(dim=-1, keepdim=True) > 0.1).float()).clamp(max=hull_cap)
        f_entry_drag_z = (-0.5 * rho_water * (HULL_BEAM * HULL_LENGTH) * 1.8 * vel[:, 2:3] * torch.abs(vel[:, 2:3]) * (hull_imm > 0.002).float()).clamp(-hull_cap, hull_cap)
        froude_hump = torch.exp(-(((spd_mean / math.sqrt(GRAVITY * HULL_LENGTH)) - 0.45)**2) / 0.04) * 1.8
        f_hull_drag_x = -(0.5 * rho_water * (HULL_BEAM * hull_imm) * (spd_mean**2) * (0.04 + froude_hump)).clamp(max=hull_cap) * torch.sign(u_f.mean(dim=-1, keepdim=True) + 1e-4)

        # 4. Hydrodynamic added mass, added inertia, Coriolis & Munk moments
        sub_mean = torch.mean(sub_s, dim=-1, keepdim=True)
        m_a_x = 0.04 * sub_mean * (rho_water / 1025.0)
        m_a_y = (math.pi / 4.0) * rho_water * (0.040**2) * strut_wet_len
        m_a_z = (math.pi / 4.0) * rho_water * (FOIL_CHORD_NOM**2) * f_span * sub_mean
        m_a_diag = torch.cat([m_a_x, m_a_y, m_a_z], dim=-1)

        I_a_x = (1.0 / 12.0) * m_a_z * (f_span**2) + m_a_y * ((0.5 * strut_len)**2)
        I_a_y = (math.pi / 128.0) * rho_water * (FOIL_CHORD_NOM**4) * f_span * sub_mean + m_a_x * (strut_len**2)
        I_a_z = m_a_y * (0.08**2)
        I_a_diag = torch.cat([I_a_x, I_a_y, I_a_z], dim=-1)

        v_r_m = v_rel_f_b.mean(dim=1)
        f_am_cor_world = torch.matmul(rot, (-torch.cross(omega, m_a_diag * v_r_m, dim=-1)).unsqueeze(-1)).squeeze(-1)

        tau_munk_x = torch.clamp((m_a_y - m_a_z) * v_r_m[:, 1:2] * v_r_m[:, 2:3], -5.0, 5.0)
        tau_munk_y = torch.clamp((m_a_z - m_a_x) * v_r_m[:, 0:1] * v_r_m[:, 2:3], -5.0, 5.0)
        tau_munk_z = torch.clamp(-(m_a_y - m_a_x) * v_r_m[:, 0:1] * v_r_m[:, 1:2], -5.0, 5.0)

        f_hydro_world = torch.matmul(rot, torch.cat([
            f_foil_bx + f_strut_drag_x + f_hull_drag_x,
            f_strut_lat_y,
            f_foil_bz + f_slam_z + f_buoy_z + f_planing_z + f_entry_drag_z
        ], dim=-1).unsqueeze(-1)).squeeze(-1)

        tau_hydro_pitch = tau_foil_pitch + (-f_hull_drag_x * HULL_DRAFT_MAX) + tau_aero_pitch - ((f_planing_z + f_buoy_z) * BOW_CENTER_OFFSET_X) + tau_munk_y
        tau_hydro_roll = tau_strut_roll + tau_aero_roll + tau_munk_x
        tau_hydro_yaw = tau_strut_yaw + tau_aero_yaw + tau_munk_z
        is_vent_overall = torch.maximum(is_vent_L_next, is_vent_R_next)

        mass_est = morphology_dict["total_mass"].detach()
        fwd_mean = (u_f.mean(dim=-1, keepdim=True) > 0.1).float()
        plant_damp_diag = torch.cat([
            -((RHO_AIR * w_area * cd_wing * v_air_mag + rho_water * f_area * cd_s.mean(dim=-1, keepdim=True) * sub_mean * spd_mean) / mass_est).clamp(-8.0, 0.0),
            -((rho_water * caudal_area * 2.8 * (immersion_depth_mean / 0.08).clamp(max=1.0) * spd_mean + 0.3 * RHO_AIR * w_area * v_air_mag) / mass_est).clamp(-8.0, 0.0),
            -((0.5 * rho_water * f_area * sub_mean * FOIL_CL_ALPHA * spd_mean * fwd_mean + 0.5 * RHO_AIR * w_area * 4.5 * v_air_mag) / mass_est).clamp(-12.0, 0.0)
        ], dim=-1).detach()

        aoa_tensors = {
            "alpha_foil": alpha_base, "strut_moment_nm": strut_root_moment, "q_water": q_w_mean,
            "f_planing_z": f_planing_z, "tau_hydro_roll": tau_hydro_roll, "tau_hydro_yaw": tau_hydro_yaw,
            "f_par_z": f_par_z, "par_pressure": par_pressure, "f_soar_surge": f_soar_surge,
            "f_scull_thrust": f_scull_thrust, "p_scull_actuator_watts": p_scull_actuator_watts,
            "strouhal_st": strouhal_st, "plant_damp_diag": plant_damp_diag,
            "sigma_cav_min": sigma_cav_min, "strut_yield_fail": (strut_root_moment > STRUT_YIELD_LIMIT_NM).float(),
            "cav_active": (sigma_cav_min < SIGMA_CAV_CRIT).float() * (~(is_air_L & is_air_R)).float(),
            "m_a_diag": m_a_diag, "I_a_diag": I_a_diag, "f_am_coriolis_world": f_am_cor_world
        }
        return (f_wing_world, f_hydro_world, tau_hydro_pitch, tau_hydro_roll, tau_hydro_yaw, h_wing,
                immersion_depth_mean, f_slam_z, f_buoy_z, is_vent_L_next, is_vent_R_next,
                is_vent_overall, wind_shear, strut_root_moment, aoa_tensors, alpha_base)


# =============================================================================
# 6. ROTOR AEROMECHANICS & PITT-PETERS DYNAMIC INFLOW
# =============================================================================
class RotorAeromechanics(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.R = PROP_RADIUS
        self.M_inflow = torch.tensor([8.0 * self.R / (3.0 * math.pi), 16.0 * self.R / (45.0 * math.pi), 16.0 * self.R / (45.0 * math.pi)], device=device).view(1, 1, 3)
        self.L_diag = torch.tensor([2.0, 4.0, 4.0], device=device).view(1, 1, 3)

    def step_inflow_and_flapping(self, nu_states, motors, v_air_hub_body, omega, dt):
        omega_props = torch.sqrt((F.relu(motors) + 1e-6) / 1.1e-5).clamp(min=50.0, max=1800.0)
        u_h, v_h, w_h = v_air_hub_body[:, :, 0:1], v_air_hub_body[:, :, 1:2], v_air_hub_body[:, :, 2:3]
        v_eff = torch.sqrt(u_h**2 + v_h**2 + (w_h + nu_states[:, :, 0:1])**2 + 1e-4)
        c_T = (motors / (RHO_AIR * ROTOR_DISK_AREA)).unsqueeze(-1)
        tip_speed = (omega_props * self.R).unsqueeze(-1).clamp(min=5.0)
        mu_x, mu_y = (u_h / tip_speed).clamp(-0.4, 0.4), (v_h / tip_speed).clamp(-0.4, 0.4)
        c_aero = torch.cat([c_T, -0.5 * mu_y * c_T, 0.5 * mu_x * c_T], dim=-1)

        L_inv = v_eff * self.L_diag
        decay = torch.exp(-(L_inv / self.M_inflow).clamp(max=2500.0) * dt)
        nu_next = nu_states * decay + (c_aero / L_inv.clamp(min=1e-3)) * (1.0 - decay)
        nu_0 = nu_next[:, :, 0]

        g8, nu_b = LOCK_NUMBER / 8.0, BLADE_FLAP_FREQ**2 - 1.0
        f_a = g8 * (mu_x.squeeze(-1) * ((8.0 / 3.0) * BLADE_PITCH_ZERO - 2.0 * (nu_0 / tip_speed.squeeze(-1))) - (16.0 / (LOCK_NUMBER * omega_props)) * omega[:, 1:2])
        f_b = g8 * (mu_y.squeeze(-1) * ((8.0 / 3.0) * BLADE_PITCH_ZERO - 2.0 * (nu_0 / tip_speed.squeeze(-1))) - (16.0 / (LOCK_NUMBER * omega_props)) * omega[:, 0:1])
        det = nu_b**2 + g8**2
        return nu_next, ((nu_b * f_a - g8 * f_b) / det).clamp(-0.20, 0.20), ((g8 * f_a + nu_b * f_b) / det).clamp(-0.20, 0.20), omega_props


# =============================================================================
# 7. BATTERY ELECTRO-THERMAL POWERTRAIN (ROTORS + PUSHER + SCULLING)
# =============================================================================
class ElectroThermalBatteryPowertrain(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.tau1, self.tau2 = BATTERY_R1 * BATTERY_C1, BATTERY_R2 * BATTERY_C2

    def step(self, motor_thrusts, ambient_inflow_wz, nu_0, omega_props, u1, u2, soc, t_cell, dt, p_scull_load, pusher_thrust=None, v_air_x=None):
        p_induced = F.relu(motor_thrusts) * F.relu(nu_0 - ambient_inflow_wz)
        q_profile = (1.0 / 8.0) * RHO_AIR * ROTOR_SOLIDITY * BLADE_CD0 * ROTOR_DISK_AREA * PROP_RADIUS * (omega_props * PROP_RADIUS)**2
        motor_currents = ((q_profile + p_induced / omega_props.clamp(min=50.0)) / MOTOR_TORQUE_CONSTANT) + NO_LOAD_CURRENT

        if pusher_thrust is not None and v_air_x is not None:
            t_p, va = F.relu(pusher_thrust), F.relu(v_air_x)
            vi_p = -0.5 * va + torch.sqrt((0.5 * va)**2 + t_p / (2.0 * RHO_AIR * PUSHER_DISK_AREA) + 1e-6)
            p_pusher_elec = (t_p * (va + vi_p)) / PUSHER_PROP_EFF + torch.where(t_p > 0.05, 3.0, 0.0)
        else:
            p_pusher_elec = torch.zeros_like(u1)

        total_current = torch.sum(motor_currents, dim=-1, keepdim=True) + (p_scull_load + p_pusher_elec) / BATTERY_V0 + AVIONICS_CURRENT
        r0_temp = BATTERY_R0_NOM * torch.exp(ARRHENIUS_COEFF * (1.0 / t_cell.clamp(min=270.0) - 1.0 / T_AMBIENT_K))
        soc_next = (soc - (total_current * dt) / BATTERY_CAPACITY_COULOMBS).clamp(0.05, 1.0)
        v_ocv = 21.0 + 3.8 * soc_next + 0.4 * (soc_next**2)

        e1, e2 = math.exp(-dt / self.tau1), math.exp(-dt / self.tau2)
        u1_next = u1 * e1 + total_current * BATTERY_R1 * (1.0 - e1)
        u2_next = u2 * e2 + total_current * BATTERY_R2 * (1.0 - e2)
        v_terminal = v_ocv - total_current * r0_temp - u1_next - u2_next

        p_coils = torch.sum((motor_currents**2) * MOTOR_COIL_RESISTANCE, dim=-1, keepdim=True)
        p_batt_heat = (total_current**2) * r0_temp + total_current * (u1_next + u2_next)
        p_elec = torch.sum(p_induced + q_profile * omega_props, dim=-1, keepdim=True) + p_coils + p_batt_heat + (4.0 * NO_LOAD_CURRENT + AVIONICS_CURRENT) * v_terminal + p_scull_load + p_pusher_elec
        t_cell_next = t_cell + (p_batt_heat - (t_cell - T_AMBIENT_K) / BATTERY_THERMAL_RES) * (dt / BATTERY_THERMAL_CAP)
        return p_elec, v_terminal, total_current, u1_next, u2_next, soc_next, t_cell_next


# =============================================================================
# 8. DIFFERENTIABLE DYKSTRA-STYLE QP & EMBEDDED DUAL QP SOLVER
# =============================================================================
class DifferentiableDykstraQP(nn.Module):
    def __init__(self, max_iters=8, rho_slack=25.0, beta_smooth=18.0):
        super().__init__()
        self.max_iters, self.rho, self.beta = max_iters, rho_slack, beta_smooth

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
                a_i, b_i, z_i = A[:, i:i+1, :], b[:, i:i+1, :], z_list[i]
                u_p = u.unsqueeze(1) - z_i
                step = F.softplus(torch.sum(a_i * u_p, dim=-1, keepdim=True) - b_i, beta=self.beta) / (A_norm_sq[:, i:i+1, :] + rho_inv)
                u_new = (u_p - step * a_i).squeeze(1)
                z_list[i] = u_new.unsqueeze(1) - u_p
                p_list[i] = F.relu(step * rho_inv)
                u = u_new

        p = torch.cat(p_list, dim=1)
        slack = F.relu(torch.matmul(A, u.unsqueeze(-1)) - b).squeeze(-1)
        return u, safe_norm(u - u_nom, dim=-1, keepdim=True), slack, torch.argmax(p.squeeze(-1), dim=-1, keepdim=True), torch.max(p.squeeze(-1), dim=-1, keepdim=True).values, safe_norm(u_clamped - u_nom, dim=-1, keepdim=True) > 1e-3


class EmbeddedDeterministicQP:
    def __init__(self, max_iter=8, rho_reg=0.08):
        self.max_iter, self.rho_reg = max_iter, rho_reg

    def solve_tensor(self, u_nom, A, b):
        N, M, _ = A.shape
        Q_reg = torch.bmm(A, A.transpose(-1, -2)) + self.rho_reg * torch.eye(M, device=u_nom.device, dtype=u_nom.dtype).unsqueeze(0)
        c = torch.bmm(A, u_nom.unsqueeze(-1)) - b
        lam = torch.zeros((N, M, 1), device=u_nom.device, dtype=u_nom.dtype)
        diag_Q = torch.diagonal(Q_reg, dim1=-2, dim2=-1).unsqueeze(-1)
        for _ in range(self.max_iter):
            for i in range(M):
                lam[:, i:i+1, :] = F.relu(lam[:, i:i+1, :] - (torch.bmm(Q_reg[:, i:i+1, :], lam) - c[:, i:i+1, :]) / diag_Q[:, i:i+1, :])
        du = torch.clamp(torch.bmm(A.transpose(-1, -2), lam).squeeze(-1), -12.0, 12.0)
        return u_nom - du, safe_norm(du, dim=-1, keepdim=True), torch.argmax(lam.squeeze(-1), dim=-1, keepdim=True)


# =============================================================================
# 9. FAST L1 ADAPTIVE DISTURBANCE OBSERVER
# =============================================================================
class FastL1AdaptiveDisturbanceObserver:
    def __init__(self, gamma=24.0, alpha_l1=0.85, device=DEVICE, tau=0.25, sigma_limit=6.0):
        self.gamma, self.alpha, self.tau, self.sigma_limit, self.device = gamma, alpha_l1, tau, sigma_limit, device
        self.reset(1)

    def reset(self, num_envs):
        self.sigma_hat = torch.zeros((num_envs, 3), device=self.device)
        self.sigma_filtered = torch.zeros((num_envs, 3), device=self.device)

    def detach_state(self):
        self.sigma_hat, self.sigma_filtered = self.sigma_hat.detach(), self.sigma_filtered.detach()

    def step(self, accel_measured, accel_commanded, dt, sat_mask=None):
        a = math.exp(-dt / self.tau)
        d_hat = torch.clamp(accel_measured - accel_commanded + self.sigma_filtered, -15.0, 15.0)
        sig_upd = torch.clamp(a * self.sigma_filtered + (1.0 - a) * d_hat, -self.sigma_limit, self.sigma_limit)
        if sat_mask is not None:
            self.sigma_hat = torch.where(sat_mask, torch.zeros_like(d_hat), d_hat)
            self.sigma_filtered = torch.where(sat_mask, self.sigma_filtered * math.exp(-dt / 0.50), sig_upd)
        else:
            self.sigma_hat, self.sigma_filtered = d_hat, sig_upd
        return self.sigma_filtered


# =============================================================================
# 10. 20-STATE NEURAL PILOT & BLOCK-DECOUPLED NEURAL CONTRACTION METRIC (NCM)
# =============================================================================
class FullNeuralWavePilot(nn.Module):
    def __init__(self, state_dim=20, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, obs):
        raw = torch.tanh(self.net(obs))
        return torch.cat([
            raw[:, 0:1] * 1.2 - 1.8 * obs[:, 0:1],
            raw[:, 1:2] * 0.9 - 2.4 * obs[:, 9:10] - 2.8 * obs[:, 3:4],
            raw[:, 2:3] * 1.2 - 9.5 * obs[:, 1:2] - 3.5 * obs[:, 2:3] - 4.2 * obs[:, 4:5]
        ], dim=-1)


class NeuralContractionMetric(nn.Module):
    def __init__(self, state_dim=6, alpha=0.10):
        super().__init__()
        self.alpha = alpha
        self.net = nn.Sequential(nn.Linear(state_dim - 1, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, 9))

    def forward(self, state_error):
        device_str = "cuda" if state_error.is_cuda else "cpu"
        with get_amp_autocast(device_str, enabled=False):
            p = self.net(state_error[:, 1:].float())
            z, o = torch.zeros_like(p[:, 0:1]), torch.ones_like(p[:, 0:1])
            L_xy = torch.stack([
                torch.cat([F.softplus(p[:, 0:1]) + 0.45, z, z], dim=-1),
                torch.cat([p[:, 1:2], F.softplus(p[:, 2:3]) + 0.45, z], dim=-1),
                torch.cat([p[:, 3:4], p[:, 4:5], F.softplus(p[:, 5:6]) + 0.45], dim=-1),
            ], dim=1)
            M_xy = torch.matmul(L_xy, L_xy.transpose(-1, -2)) + 0.25 * torch.eye(3, device=state_error.device, dtype=L_xy.dtype).unsqueeze(0)
            L_z = torch.stack([
                torch.cat([F.softplus(p[:, 6:7]) + 0.45, z], dim=-1),
                torch.cat([p[:, 7:8], F.softplus(p[:, 8:9]) + 0.45], dim=-1),
            ], dim=1)
            M_z = torch.matmul(L_z, L_z.transpose(-1, -2)) + 0.25 * torch.eye(2, device=state_error.device, dtype=L_z.dtype).unsqueeze(0)
            return torch.stack([
                torch.cat([o, z, z, z, z, z], dim=-1),
                torch.cat([z, M_xy[:, 0:1, 0], z, M_xy[:, 0:1, 1], M_xy[:, 0:1, 2], z], dim=-1),
                torch.cat([z, z, M_z[:, 0:1, 0], z, z, M_z[:, 0:1, 1]], dim=-1),
                torch.cat([z, M_xy[:, 1:2, 0], z, M_xy[:, 1:2, 1], M_xy[:, 1:2, 2], z], dim=-1),
                torch.cat([z, M_xy[:, 2:3, 0], z, M_xy[:, 2:3, 1], M_xy[:, 2:3, 2], z], dim=-1),
                torch.cat([z, z, M_z[:, 1:2, 0], z, z, M_z[:, 1:2, 1]], dim=-1),
            ], dim=1)


# =============================================================================
# 11. 6-DoF KIRCHHOFF ROBOT DYNAMICS (ADDED MASS, FLAPPING, J-ROLLOFF)
# =============================================================================
class WaveRidingAerialRobot:
    def __init__(self, dt, device, fixed_foil=False, gimbal_max_deg=15.0, mass_scale=1.0):
        self.dt, self.device = dt, device
        self.fixed_foil, self.gimbal_max_deg = fixed_foil, float(gimbal_max_deg)
        rad = math.radians(self.gimbal_max_deg)
        self.gimbal_sin, self.gimbal_cos, self.gimbal_tan = math.sin(rad), math.cos(rad), math.tan(rad)
        self.J = torch.tensor([[0.045, 0.052, 0.088]], device=device) * mass_scale
        self.J_inv = 1.0 / self.J
        self.d, self.c_q = ARM_LENGTH / math.sqrt(2.0), TORQUE_TO_THRUST_RATIO
        self.rotor_aero = RotorAeromechanics(device)
        d = self.d
        self.r_x = torch.tensor([[+d, +d, -d, -d]], device=device)
        self.r_y = torch.tensor([[-d, +d, +d, -d]], device=device)
        self.r_hubs = torch.stack([self.r_x, self.r_y, torch.zeros_like(self.r_x)], dim=-1)
        self.yaw_tx, self.yaw_ty = -self.r_y / ARM_LENGTH, +self.r_x / ARM_LENGTH
        self.spin_signs = torch.tensor([[-1.0, +1.0, -1.0, +1.0]], device=device)
        self.motor_fault_scale, self.gimbal_jammed = None, False

    def set_mass_scale(self, mass_scale):
        self.J = torch.tensor([[0.045, 0.052, 0.088]], device=self.device) * float(mass_scale)
        self.J_inv = 1.0 / self.J

    def allocate_wrench_hybrid(self, Fx_cmd, Fy_cmd, Fz_cmd, torques_desired, v_batt, ige_gain,
                               d_sub, airspeed, is_soaring, morphology_dict, delta_collective=None):
        if not torch.is_tensor(v_batt):
            v_batt = torch.tensor(v_batt, device=Fz_cmd.device)
        thrust_ceiling = MAX_THRUST_PER_MOTOR * (v_batt / BATTERY_V0)**2
        q_water = 0.5 * RHO_WATER_NOM * (airspeed.clamp(min=0.5)**2)
        sub_frac, h_fade = (d_sub / 0.040).clamp(0.0, 1.0), ((airspeed - 0.30) / 0.70).clamp(0.0, 1.0)
        tx_des, ty_des, tz_des = torch.clamp(torques_desired[:, 0:1], -3.5, 3.5), torch.clamp(torques_desired[:, 1:2], -3.5, 3.5), torch.clamp(torques_desired[:, 2:3], -1.2, 1.2)

        if self.fixed_foil:
            tau_hr, tau_hy = torch.zeros_like(tx_des), torch.zeros_like(tz_des)
            delta_e_L = delta_e_R = delta_rud = torch.zeros_like(tx_des)
        else:
            max_hr, max_hy = 1.00 * sub_frac * h_fade, 0.50 * sub_frac * h_fade
            hr_gain = 2.0 * (morphology_dict["foil_span"] * 0.25) * q_water * (morphology_dict["foil_area"] * 0.5) * HYDRO_CL_DELTA * sub_frac
            hy_gain = q_water * morphology_dict["caudal_area"] * HYDRO_RUDDER_CY_DELTA * sub_frac * 0.08
            tau_hr, tau_hy = torch.clamp(tx_des, -max_hr, max_hr), torch.clamp(tz_des, -max_hy, max_hy)
            d_roll = torch.clamp(tau_hr / (hr_gain + 0.15), -MAX_HYDRO_DEFLECTION, MAX_HYDRO_DEFLECTION) * h_fade
            delta_rud = -torch.clamp(tau_hy / (hy_gain + 0.10), -MAX_HYDRO_DEFLECTION, MAX_HYDRO_DEFLECTION) * h_fade
            d_coll = (torch.zeros_like(d_roll) if delta_collective is None else delta_collective) * h_fade
            delta_e_L, delta_e_R = torch.clamp(d_roll + d_coll, -MAX_HYDRO_DEFLECTION, MAX_HYDRO_DEFLECTION), torch.clamp(-d_roll + d_coll, -MAX_HYDRO_DEFLECTION, MAX_HYDRO_DEFLECTION)

        tx_res, ty_res, tz_after = tx_des - tau_hr, ty_des, tz_des - tau_hy
        Fz_comp = Fz_cmd / torch.clamp(ige_gain, min=1.0, max=1.6)
        f_floor = torch.where(is_soaring.bool(), torch.full_like(Fz_comp, MOTOR_IDLE_FLOOR), torch.full_like(Fz_comp, 0.90)) * 4.0
        Fz_base = torch.minimum(torch.maximum(Fz_comp, f_floor), (thrust_ceiling - 1.20) * 4.0)

        df_x, df_y = 0.25 * (tx_res / self.d), -0.25 * (ty_res / self.d)
        raw_rp = 0.25 * Fz_base + torch.cat([-df_x + df_y, +df_x + df_y, +df_x - df_y, -df_x - df_y], dim=-1)
        min_rp = torch.min(raw_rp, dim=-1, keepdim=True).values
        rp_sh = raw_rp + F.relu(MOTOR_IDLE_FLOOR - min_rp)
        fz_rotors = torch.minimum((rp_sh - F.relu(torch.max(rp_sh, dim=-1, keepdim=True).values - thrust_ceiling)).clamp(min=MOTOR_IDLE_FLOOR), thrust_ceiling)

        if self.gimbal_max_deg <= 1e-3:
            f_x_rot = f_y_rot = torch.zeros_like(fz_rotors); tz_g_act = torch.zeros_like(tz_after)
        else:
            f_lmax = fz_rotors * self.gimbal_tan
            f_lsum = torch.sum(f_lmax, dim=-1, keepdim=True).clamp(min=1e-4)
            tz_ga = torch.clamp(tz_after, -0.65 * ARM_LENGTH * f_lsum, 0.65 * ARM_LENGTH * f_lsum)
            fy_mag = (f_lmax / f_lsum) * (tz_ga / ARM_LENGTH)
            fy_x, fy_y = fy_mag * self.yaw_tx, fy_mag * self.yaw_ty
            fw = fz_rotors / torch.sum(fz_rotors, dim=-1, keepdim=True).clamp(min=1e-4)
            fx_raw, fy_raw = fy_x + fw * (Fx_cmd - torch.sum(fy_x, dim=-1, keepdim=True)), fy_y + fw * (Fy_cmd - torch.sum(fy_y, dim=-1, keepdim=True))
            scale = torch.clamp(f_lmax / torch.sqrt(fx_raw**2 + fy_raw**2 + 1e-8), max=1.0)
            f_x_rot, f_y_rot = fx_raw * scale, fy_raw * scale
            tz_g_act = torch.sum(self.r_x * f_y_rot - self.r_y * f_x_rot, dim=-1, keepdim=True)

        f_tot = torch.sqrt(f_x_rot**2 + f_y_rot**2 + fz_rotors**2)
        max_dfz = torch.min(torch.minimum((f_tot - MOTOR_IDLE_FLOOR).clamp(min=0.0), (thrust_ceiling - f_tot).clamp(min=0.0)), dim=-1, keepdim=True).values
        df_zc = torch.clamp(0.25 * ((tz_after - tz_g_act) / self.c_q), -max_dfz, max_dfz)
        clamped_motors = torch.minimum((f_tot + self.spin_signs * df_zc).clamp(min=MOTOR_IDLE_FLOOR), thrust_ceiling)
        f_mag = f_tot.clamp(min=1e-4)
        target_tilts = torch.stack([f_x_rot / f_mag, f_y_rot / f_mag], dim=-1)

        tz_deliv = tz_g_act + 4.0 * self.c_q * df_zc
        t_scale = torch.where(torch.abs(tz_after) > 1e-3, (torch.abs(tz_deliv) / (torch.abs(tz_after) + 1e-6)).clamp(max=1.0), torch.ones_like(min_rp))
        return (clamped_motors, target_tilts, torch.cat([delta_e_L, delta_e_R, delta_rud], dim=-1),
                min_rp - MOTOR_IDLE_FLOOR, clamped_motors <= (MOTOR_IDLE_FLOOR + 0.01),
                clamped_motors >= thrust_ceiling - 1e-3, t_scale,
                torch.sum(f_x_rot, dim=-1, keepdim=True), torch.sum(f_y_rot, dim=-1, keepdim=True))

    def integrate(self, pos, vel, rot, omega, motors, rotor_tilts, pusher_thrust, nu_states,
                  target_motors, target_tilts, target_pusher, f_wing, f_hydro,
                  tau_hydro_pitch, tau_hydro_roll, tau_hydro_yaw, ambient_wind, eta, wave_slope,
                  mass, motor_tau_up, motor_tau_down, m_a_diag=None, I_a_diag=None, f_am_cor_world=None, v_batt=None):
        N = pos.shape[0]
        v_rat = (v_batt / BATTERY_V0).clamp(0.7, 1.05) if v_batt is not None else torch.ones((N, 1), device=self.device)
        m_ceil = MAX_THRUST_PER_MOTOR * (v_rat**2)

        tau_m = torch.where(target_motors >= motors, motor_tau_up, motor_tau_down)
        motors_next = torch.minimum(torch.nan_to_num(motors + (self.dt / (tau_m + self.dt)) * (target_motors - motors), nan=MOTOR_IDLE_FLOOR).clamp(min=MOTOR_IDLE_FLOOR), m_ceil)

        if self.gimbal_jammed:
            tilts_next = rotor_tilts
        elif self.gimbal_max_deg <= 1e-3:
            tilts_next = torch.zeros_like(rotor_tilts)
        else:
            tilts_next = torch.nan_to_num(rotor_tilts + (self.dt / (GIMBAL_TAU + self.dt)) * (target_tilts - rotor_tilts), nan=0.0)
            tilts_next = tilts_next * torch.clamp(self.gimbal_sin / torch.sqrt(tilts_next[:, :, 0]**2 + tilts_next[:, :, 1]**2 + 1e-10), max=1.0).unsqueeze(-1)

        ux, uy = tilts_next[:, :, 0], tilts_next[:, :, 1]
        uz = torch.sqrt((1.0 - ux**2 - uy**2).clamp(min=max(1e-4, self.gimbal_cos**2)))

        v_cg_b = torch.matmul(rot.transpose(-1, -2), (vel - ambient_wind).unsqueeze(-1)).squeeze(-1)
        v_hub_b = v_cg_b.unsqueeze(1) + torch.cross(omega.unsqueeze(1), self.r_hubs.expand(N, -1, -1), dim=-1)
        v_ax = v_cg_b[:, 0:1].clamp(min=0.0)
        pusher_cap = MAX_PUSHER_THRUST * (v_rat**2) * torch.clamp(1.0 - torch.pow(v_ax / PUSHER_V_PITCH_ZERO, 1.4), min=0.12, max=1.0)
        tau_p = torch.where(target_pusher >= pusher_thrust, motor_tau_up, motor_tau_down)
        pusher_next = torch.minimum(torch.nan_to_num(pusher_thrust + (self.dt / (tau_p + self.dt)) * (target_pusher - pusher_thrust), nan=0.0).clamp(min=0.0), pusher_cap)

        h_rot = (pos[:, 2:3] - eta).clamp(min=0.04)
        ige_gain = 1.0 / (1.0 - torch.clamp((PROP_RADIUS / (4.0 * h_rot))**2, max=0.35))
        spray_derate = 1.0 + 0.15 * F.relu(0.14 - h_rot) * wave_slope.clamp(max=0.5)

        m_phys = motors_next * self.motor_fault_scale if self.motor_fault_scale is not None else motors_next
        nu_next, a_1, b_1, omega_props = self.rotor_aero.step_inflow_and_flapping(nu_states, m_phys, v_hub_b, omega, self.dt)

        fx, fy, fz = m_phys * ux, m_phys * uy, m_phys * uz
        f_flap_x, f_flap_y = -m_phys * torch.sin(a_1), +m_phys * torch.sin(b_1)
        f_rot_w = torch.matmul(rot, torch.cat([
            torch.sum(fx + f_flap_x, dim=-1, keepdim=True) + pusher_next,
            torch.sum(fy + f_flap_y, dim=-1, keepdim=True),
            (torch.sum(fz, dim=-1, keepdim=True) * ige_gain) / spray_derate
        ], dim=-1).unsqueeze(-1)).squeeze(-1)

        tot_f = f_rot_w + f_wing + f_hydro + torch.cat([torch.zeros((N, 2), device=self.device), -mass * GRAVITY], dim=-1)
        if f_am_cor_world is not None:
            tot_f = tot_f + f_am_cor_world
        tot_f = torch.nan_to_num(tot_f, nan=0.0, posinf=400.0, neginf=-400.0)

        if m_a_diag is not None:
            eye3 = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0)
            M_eff = mass.float().unsqueeze(-1) * eye3 + torch.bmm(rot.float() * m_a_diag.float().unsqueeze(1), rot.float().transpose(-1, -2))
            accel = torch.linalg.solve(M_eff, tot_f.float().unsqueeze(-1)).squeeze(-1)
        else:
            accel = tot_f / mass

        vel_next = (vel + accel * self.dt).clamp(-25.0, 25.0)
        pos_next = pos + vel_next * self.dt

        tx_flap = -0.025 * torch.sum(m_phys * torch.sin(b_1), dim=-1, keepdim=True)
        ty_flap = -0.025 * torch.sum(m_phys * torch.sin(a_1), dim=-1, keepdim=True)
        tx = torch.sum(self.r_y * fz, dim=-1, keepdim=True) + tau_hydro_roll + tx_flap
        ty = -torch.sum(self.r_x * fz, dim=-1, keepdim=True) + tau_hydro_pitch + ty_flap
        tz = torch.sum(self.r_x * fy - self.r_y * fx, dim=-1, keepdim=True) + self.c_q * torch.sum(self.spin_signs * fz, dim=-1, keepdim=True) + tau_hydro_yaw

        J_eff = (self.J + I_a_diag) if I_a_diag is not None else self.J
        omega_next = torch.nan_to_num(omega + ((torch.cat([tx, ty, tz], dim=-1) - torch.cross(omega, omega * J_eff, dim=-1)) / J_eff) * self.dt, nan=0.0).clamp(-12.0, 12.0)

        v_rot = omega_next * self.dt
        th = safe_norm(v_rot, dim=-1, keepdim=True).clamp(min=1e-6)
        vx, vy, vz = v_rot[:, 0:1], v_rot[:, 1:2], v_rot[:, 2:3]
        zv = torch.zeros_like(vx)
        K = torch.stack([torch.cat([zv, -vz, vy], dim=-1), torch.cat([vz, zv, -vx], dim=-1), torch.cat([-vy, vx, zv], dim=-1)], dim=1)
        dR = torch.eye(3, device=self.device).unsqueeze(0) + (torch.sin(th) / th).unsqueeze(-1) * K + ((1.0 - torch.cos(th)) / (th**2)).unsqueeze(-1) * torch.matmul(K, K)
        rot_next = torch.matmul(rot, dR)
        c0 = rot_next[:, :, 0] / safe_norm(rot_next[:, :, 0], dim=-1, keepdim=True)
        c2 = torch.cross(c0, rot_next[:, :, 1], dim=-1); c2 = c2 / safe_norm(c2, dim=-1, keepdim=True)
        rot_clean = torch.stack([c0, torch.cross(c2, c0, dim=-1), c2], dim=-1)

        return pos_next, vel_next, rot_clean, omega_next, motors_next, tilts_next, pusher_next, nu_next, omega_props, accel, ige_gain

        