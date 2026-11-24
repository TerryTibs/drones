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
SCULL_SERVO_POWER_COEFF = 0.45

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
    """Smooth Re-dependent low-Re + Viterna post-stall polar returning (cl, cd, cm, neg_cp_min)."""
    if is_water:
        cl0, cl_a, cm0 = 0.0, FOIL_CL_ALPHA, 0.0
        a_eff = alpha + (HYDRO_CL_DELTA / FOIL_CL_ALPHA) * delta_flap
        a_stall = math.radians(10.5) + math.radians(2.5) * torch.clamp((Re - 8.0e4) / 2.2e5, 0.0, 1.0)
        cd_min = 0.011 * torch.pow((2.0e5 / Re.clamp(min=2.0e4)), 0.20)
        neg_cp_min = 0.35 + 2.8 * torch.abs(a_eff) + 1.8 * torch.abs(delta_flap)
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

    cd_att = cd_min + 0.045 * ((cl_att - cl0)**2) + (HYDRO_CD_DELTA * (delta_flap**2) if is_water else 0.0)
    cd_post = 0.05 + 1.85 * (torch.sin(a_eff)**2)
    cd = s_att * cd_att + (1.0 - s_att) * cd_post
    cm = cm0 - 0.12 * (1.0 - s_att) * torch.sin(a_eff) - (0.55 * delta_flap if is_water else 0.0)
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
        total_mass = 1.65 + 2.85 * (wing_area + foil_area + 2.0 * strut_len * 0.04)

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
        """Vectorized evaluation over (N, S, 3) strip coordinates -> (eta (N,S), deta_dt (N,S), v_orb (N,S,3))."""
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
        R = init_rot
        qw = 0.5 * torch.sqrt(torch.clamp(1.0 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2], min=1e-6))
        q = torch.cat([
            qw.unsqueeze(-1),
            ((R[:, 2, 1] - R[:, 1, 2]) / (4.0 * qw)).unsqueeze(-1),
            ((R[:, 0, 2] - R[:, 2, 0]) / (4.0 * qw)).unsqueeze(-1),
            ((R[:, 1, 0] - R[:, 0, 1]) / (4.0 * qw)).unsqueeze(-1)
        ], dim=-1)
        self.q_hat = q / safe_norm(q, dim=-1, keepdim=True)
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

        # 1. FULL 3D WIG AERODYNAMICS + SIDESLIP + PAR CUSHION
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

        # Rotate aerodynamic lift & drag in body frame so banked turns tilt wing lift laterally!
        L_wing = q_air * w_area * cl_wing
        D_wing = q_air * w_area * cd_wing
        f_w_bx = L_wing * torch.sin(alpha_a) - D_wing * torch.cos(alpha_a)
        f_w_by = -q_air * w_area * 0.65 * beta_wing
        f_w_bz = L_wing * torch.cos(alpha_a) + D_wing * torch.sin(alpha_a) + f_par_z
        f_wing_world = torch.matmul(rot, torch.cat([f_w_bx, f_w_by, f_w_bz], dim=-1).unsqueeze(-1)).squeeze(-1)

        tau_aero_pitch = q_air * w_area * w_chord * (+0.85 * alpha_wing + cm_w - 2.8 * (omega[:, 1:2] * w_chord / (2.0 * v_air_mag)))
        tau_aero_roll = q_air * w_area * w_span * ((-0.08 - 0.85 * anhedral_rad) * beta_wing - 0.45 * (omega[:, 0:1] * w_span / (2.0 * v_air_mag)))
        tau_aero_yaw = q_air * w_area * w_span * (+0.04 * beta_wing - 0.18 * (omega[:, 2:3] * w_span / (2.0 * v_air_mag)))

        # 2. 8-STRIP SPANWISE HYDROFOIL + 5-STRIP DEPTH STRUT + ADDED MASS
        y_s = self.y_frac_f * f_span  # (1, 8)
        ds_f = f_span / self.n_foil_strips
        r_f_loc = torch.stack([torch.zeros_like(y_s), y_s, -strut_len.expand_as(y_s)], dim=-1)  # (1, 8, 3)
        p_f_w = pos.unsqueeze(1) + torch.matmul(r_f_loc, rot_T)  # (N, 8, 3)

        if wave_field is not None:
            eta_f, deta_dt_f, v_orb_f = wave_field.evaluate_wave_points(p_f_w, sim_time, wave_scale=wave_scale)
        else:
            eta_f, deta_dt_f, v_orb_f = eta.expand(N, 8), deta_dt.expand(N, 8), v_water_orbital.unsqueeze(1).expand(N, 8, 3)

        omega_x_rf = torch.cross(omega.unsqueeze(1), r_f_loc.expand(N, -1, -1), dim=-1)
        v_f_w = vel.unsqueeze(1) + torch.matmul(omega_x_rf, rot_T)
        v_rel_f_b = torch.matmul(v_f_w - v_orb_f, rot)  # (N, 8, 3)

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

        # Ventilation hysteresis on L/R semi-spans + Breslin shallow-submergence attenuation
        suction_L = neg_cp_s[:, 4:8].mean(dim=-1, keepdim=True) * q_w_f[:, 4:8].mean(dim=-1, keepdim=True)
        suction_R = neg_cp_s[:, 0:4].mean(dim=-1, keepdim=True) * q_w_f[:, 0:4].mean(dim=-1, keepdim=True)
        seal_L = rho_water * GRAVITY * d_sub_L + 120.0 + PAR_SEAL_GAIN * par_pressure
        seal_R = rho_water * GRAVITY * d_sub_R + 120.0 + PAR_SEAL_GAIN * par_pressure
        is_air_L, is_air_R = (d_sub_L < 0.005), (d_sub_R < 0.005)
        alpha_L_m, alpha_R_m = (alpha_f_s[:, 4:8] + (HYDRO_CL_DELTA / FOIL_CL_ALPHA) * delta_s[:, 4:8]).mean(dim=-1, keepdim=True), (alpha_f_s[:, 0:4] + (HYDRO_CL_DELTA / FOIL_CL_ALPHA) * delta_s[:, 0:4]).mean(dim=-1, keepdim=True)

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
            self.cl_lag_s = cl_s
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
        p_scull_actuator_watts = torch.clamp(f_scull_thrust * SCULL_SERVO_POWER_COEFF, 0.0, 45.0)

        f_foil_bx = torch.sum(-D_s * torch.sign(u_f + 1e-4) + f_soar_s, dim=-1, keepdim=True) + f_scull_thrust
        f_foil_bz = torch.sum(L_s, dim=-1, keepdim=True)
        f_soar_surge = torch.sum(f_soar_s, dim=-1, keepdim=True)

        slam_cap, hull_cap = (350.0, 250.0) if self.harsh_physics else (25.0, 40.0)
        imm_rate_s = F.relu(deta_dt_f - v_f_w[:, :, 2]) * (d_sub_s > 0.001).float()
        f_slam_z = torch.clamp(torch.sum(0.5 * rho_water * HYDRO_SLAM_COEFF * ds_f * FOIL_CHORD_NOM * (imm_rate_s**2), dim=-1, keepdim=True), max=slam_cap)

        tau_foil_roll = torch.sum(y_s * L_s, dim=-1, keepdim=True)
        tau_foil_pitch = torch.sum(q_w_f * area_s * FOIL_CHORD_NOM * sub_s * cm_s, dim=-1, keepdim=True) - f_foil_bx * strut_len

        # 5-Strip Vertical Strut (crossflow, rudder, rotational yaw/roll damping & Hoerner spray drag)
        ds_v = (strut_len - HULL_DRAFT_MAX).clamp(min=0.04) / self.n_strut_strips
        z_k = -HULL_DRAFT_MAX - self.z_frac_s * (strut_len - HULL_DRAFT_MAX).clamp(min=0.04)  # (1, 5)
        r_s_loc = torch.stack([torch.full_like(z_k, -0.04), torch.zeros_like(z_k), z_k], dim=-1)  # (1, 5, 3)
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

        rud_wt = self.z_frac_s  # rudder concentrated on lower strut
        fy_st = q_w_st * (caudal_area / 5.0 + 0.04 * ds_v) * w_frac_st * (-2.8 * (v_st / spd_st) + HYDRO_RUDDER_CY_DELTA * delta_rud * rud_wt)
        fx_st = -q_w_st * (0.04 * ds_v) * w_frac_st * 0.018 * torch.sign(u_st + 1e-4)

        # Hoerner free-surface spray drag when strut pierces the waterline
        is_piercing = ((strut_wet_len > 0.005) & (strut_wet_len < (strut_len - HULL_DRAFT_MAX - 0.005))).float()
        f_spray_x = -0.5 * rho_water * (spd_mean**2) * (0.012**2) * 0.24 * is_piercing * torch.sign(u_st.mean(dim=-1, keepdim=True) + 1e-4)

        lat_st_cap = 25.0 if self.harsh_physics else 8.0
        f_strut_lat_y = torch.clamp(torch.sum(fy_st, dim=-1, keepdim=True), -lat_st_cap, lat_st_cap)
        f_strut_drag_x = torch.sum(fx_st, dim=-1, keepdim=True) + f_spray_x
        tau_strut_roll = torch.sum(-z_k * fy_st, dim=-1, keepdim=True) + tau_foil_roll
        tau_strut_yaw = torch.sum(-0.04 * fy_st, dim=-1, keepdim=True) - f_strut_lat_y * 0.04
        strut_root_moment = torch.abs(f_strut_lat_y * strut_len) + torch.abs(tau_foil_roll)

        # 3. Hull planing & buoyancy
        hull_imm = (eta - (pos[:, 2:3] - HULL_DRAFT_MAX)).clamp(0.0, HULL_DRAFT_MAX * 2.0)
        f_buoy_z = rho_water * GRAVITY * (hull_imm / HULL_DRAFT_MAX).clamp(max=1.5) * HULL_VOLUME_TOTAL
        f_planing_z = (0.5 * rho_water * (HULL_BEAM * HULL_LENGTH) * (spd_mean**2) * 0.14 * torch.sin(F.relu(alpha_wing + 0.02)) * (hull_imm > 0.005).float() * (u_f.mean(dim=-1, keepdim=True) > 0.1).float()).clamp(max=hull_cap)
        f_entry_drag_z = (-0.5 * rho_water * (HULL_BEAM * HULL_LENGTH) * 1.8 * vel[:, 2:3] * torch.abs(vel[:, 2:3]) * (hull_imm > 0.002).float()).clamp(-hull_cap, hull_cap)
        froude_hump = torch.exp(-(((spd_mean / math.sqrt(GRAVITY * HULL_LENGTH)) - 0.45)**2) / 0.04) * 1.8
        f_hull_drag_x = -(0.5 * rho_water * (HULL_BEAM * hull_imm) * (spd_mean**2) * (0.04 + froude_hump)).clamp(max=hull_cap) * torch.sign(u_f.mean(dim=-1, keepdim=True) + 1e-4)

        # 4. Hydrodynamic Added Mass, Added Inertia, Coriolis & Munk Moments
        sub_mean = torch.mean(sub_s, dim=-1, keepdim=True)
        m_a_x = 0.04 * sub_mean * (rho_water / 1025.0)
        m_a_y = (math.pi / 4.0) * rho_water * (0.040**2) * strut_wet_len
        m_a_z = (math.pi / 4.0) * rho_water * (FOIL_CHORD_NOM**2) * f_span * sub_mean
        m_a_diag = torch.cat([m_a_x, m_a_y, m_a_z], dim=-1)

        I_a_x = (1.0 / 12.0) * m_a_z * (f_span**2) + m_a_y * ((0.5 * strut_len)**2)
        I_a_y = (math.pi / 128.0) * rho_water * (FOIL_CHORD_NOM**4) * f_span * sub_mean + m_a_x * (strut_len**2)
        I_a_z = m_a_y * (0.06**2)
        I_a_diag = torch.cat([I_a_x, I_a_y, I_a_z], dim=-1)

        v_r_m = v_rel_f_b.mean(dim=1)
        f_am_cor_b = -torch.cross(omega, m_a_diag * v_r_m, dim=-1)
        f_am_cor_world = torch.matmul(rot, f_am_cor_b.unsqueeze(-1)).squeeze(-1)

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
        self.yaw_tx, self.yaw_ty = -self.r_y / ARM_LENGTH, +self.r_x / ARM_LENGTH
        self.spin_signs = torch.tensor([[-1.0, +1.0, -1.0, +1.0]], device=device)
        self.motor_fault_scale, self.gimbal_jammed = None, False

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

        v_hub_b = torch.matmul(rot.transpose(-1, -2).unsqueeze(1), (vel - ambient_wind).unsqueeze(1).unsqueeze(-1)).squeeze(-1)
        v_ax = v_hub_b[:, 0, 0:1].clamp(min=0.0)
        pusher_cap = MAX_PUSHER_THRUST * (v_rat**2) * torch.clamp(1.0 - torch.pow(v_ax / PUSHER_V_PITCH_ZERO, 1.4), min=0.12, max=1.0)
        tau_p = torch.where(target_pusher >= pusher_thrust, motor_tau_up, motor_tau_down)
        pusher_next = torch.minimum(torch.nan_to_num(pusher_thrust + (self.dt / (tau_p + self.dt)) * (target_pusher - pusher_thrust), nan=0.0).clamp(min=0.0), pusher_cap)

        h_rot = (pos[:, 2:3] - eta).clamp(min=0.04)
        ige_gain = 1.0 / (1.0 - torch.clamp((PROP_RADIUS / (4.0 * h_rot))**2, max=0.35))
        spray_derate = 1.0 + 0.15 * F.relu(0.14 - h_rot) * wave_slope.clamp(max=0.5)

        m_phys = motors_next * self.motor_fault_scale if self.motor_fault_scale is not None else motors_next
        nu_next, a_1, b_1, omega_props = self.rotor_aero.step_inflow_and_flapping(nu_states, m_phys, v_hub_b, omega, self.dt)

        # Rotor thrust + flapping H-force drag and semi-rigid hub blowback moments
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

        # =============================================================================
# 12. SIM-TO-REAL MULTI-ENVIRONMENT (WITH REAL PHYSICS, ABLATIONS & OBSTACLES)
# =============================================================================
class WaveRidingAutonomyEnv:
    def __init__(self, num_envs=1024, dt=0.015, device=DEVICE, use_embedded_qp=False,
                 fixed_foil=False, disable_scull=False, gimbal_max_deg=15.0, mass_kg=None,
                 harsh_physics=False, enable_obstacles=False, gnss_denied_window=None,
                 fault_scenario=None, fault_time=4.0):
        self.num_envs, self.dt, self.device, self.use_embedded_qp = num_envs, dt, device, use_embedded_qp
        self.fixed_foil, self.disable_scull, self.gimbal_max_deg = fixed_foil, disable_scull, float(gimbal_max_deg)
        self.harsh_physics, self.enable_obstacles = harsh_physics, enable_obstacles
        self.gnss_denied_window, self.fault_scenario, self.fault_time = gnss_denied_window, fault_scenario, fault_time
        self.mass_scale = 1.0 if mass_kg is None else (float(mass_kg) / 2.371)

        self.morphology = VehicleMorphologyGenome(device)
        self.waves = DirectionalOceanWaveField(dt, airspeed=5.20, num_waves=32, device=device)
        self.aero_hydro = MultimodalAeroHydroDynamics(device, dt=dt, harsh_physics=harsh_physics, fixed_foil=fixed_foil, disable_scull=disable_scull)
        self.robot = WaveRidingAerialRobot(dt, device, fixed_foil=fixed_foil, gimbal_max_deg=gimbal_max_deg, mass_scale=self.mass_scale)
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
        self.target_speed, self.wave_curriculum_scale = 5.20, 1.0
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
        self.auto_mission_active, self.auto_mission_timer = False, 0.0

        self.A_cbf_crash = torch.tensor([[0.0, 0.0, -1.0]], device=device).repeat(num_envs, 1, 1)
        self.A_cbf_ceil = torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(num_envs, 1, 1)
        self.A_cbf_cav = torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(num_envs, 1, 1)
        self.reset()

    def set_mode(self, new_mode):
        self.current_mode = torch.full((self.num_envs, 1), new_mode, dtype=torch.long, device=self.device)

    def set_wave_curriculum(self, scale):
        self.wave_curriculum_scale = scale

    def detach_states(self):
        for attr in ("pos", "vel", "rot", "omega", "motors", "rotor_tilts", "pusher_thrust", "nu_states",
                     "u1", "u2", "soc", "t_cell", "v_batt", "h_target_filtered", "v_target_filtered",
                     "phi_des_filtered", "psi_des_filtered", "e_int_height", "e_int_y", "a_pilot_prev",
                     "a_pilot_filtered", "a_pre_dob_prev", "l1_sat_prev", "scull_throttle_cmd",
                     "last_deta_dx_nadir", "last_w_up_nadir", "hydro_surfaces_actual", "hydro_surfaces",
                     "is_vent_L", "is_vent_R", "clearance_hold", "follower_accel_prev", "y_ref"):
            setattr(self, attr, getattr(self, attr).detach())
        buf = [t.detach() for t in self.guidance_cmd_buffer]
        self.guidance_cmd_buffer.clear(); self.guidance_cmd_buffer.extend(buf)
        self.ekf.detach_state(); self.l1_observer.detach_state(); self.aero_hydro.detach_state()

    def reset(self, mode_mix=False):
        N = self.num_envs
        self.sim_time, self.step_count = torch.zeros((N, 1), device=self.device), 0
        self.waves.reset(N); self.l1_observer.reset(N); self.dispersion_observer.reset(N); self.aero_hydro.reset(N)
        self.robot.motor_fault_scale, self.robot.gimbal_jammed, self.fault_active = None, False, False

        self.mass_noise = torch.empty((N, 1), device=self.device).uniform_(0.96, 1.04)
        with torch.no_grad():
            m_evolved = self.morphology()["total_mass"]
        self.mass = (m_evolved.repeat(N, 1) * self.mass_noise * self.mass_scale).detach()
        self.foil_decalage = torch.empty((N, 1), device=self.device).uniform_(math.radians(FOIL_DECALAGE_DEG[0]), math.radians(FOIL_DECALAGE_DEG[1]))
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
            vx0, vy0, vz0 = torch.full((N, 1), self.target_speed, device=self.device), torch.zeros((N, 1), device=self.device), torch.zeros((N, 1), device=self.device)
            phi0, theta0, psi0 = torch.zeros((N, 1), device=self.device), torch.full((N, 1), math.radians(INIT_PITCH_NOSE_DOWN_DEG), device=self.device), torch.zeros((N, 1), device=self.device)
            omega0 = torch.zeros((N, 3), device=self.device)
            self.h_target_filtered.fill_(RIDE_HEIGHT_OPTIMAL); self.v_target_filtered.fill_(self.target_speed)
            self.motors = torch.full((N, 4), 1.5, device=self.device); self.pusher_thrust = torch.full((N, 1), 2.5, device=self.device)
        else:
            u_m = torch.rand((N, 1), device=self.device)
            mode_t = torch.full((N, 1), MODE_FOILING, dtype=torch.long, device=self.device)
            for thr, md in [(0.30, MODE_LOITER), (0.45, MODE_TAKEOFF), (0.60, MODE_EGRESS), (0.68, MODE_FREE_FLIGHT), (0.75, MODE_INGRESS), (0.90, MODE_SOARING)]:
                mode_t = torch.where(u_m >= thr, torch.full_like(mode_t, md), mode_t)
            self.current_mode = mode_t
            init_y = torch.empty((N, 1), device=self.device).uniform_(-1.2, 1.2)
            init_clearance = torch.empty((N, 1), device=self.device).uniform_(0.08, 0.34)
            vx0 = torch.empty((N, 1), device=self.device).uniform_(4.0, 6.2)
            vy0, vz0 = torch.empty((N, 1), device=self.device).uniform_(-0.7, 0.7), torch.empty((N, 1), device=self.device).uniform_(-0.5, 0.5)
            phi0 = torch.empty((N, 1), device=self.device).uniform_(-math.radians(12.0), math.radians(12.0))
            theta0 = torch.empty((N, 1), device=self.device).uniform_(-math.radians(8.0), math.radians(8.0))
            psi0 = torch.empty((N, 1), device=self.device).uniform_(-math.radians(12.0), math.radians(12.0))
            omega0 = torch.randn((N, 3), device=self.device) * 0.35

            is_l, is_to, is_eg, is_ff, is_in, is_so = (mode_t == MODE_LOITER), (mode_t == MODE_TAKEOFF), (mode_t == MODE_EGRESS), (mode_t == MODE_FREE_FLIGHT), (mode_t == MODE_INGRESS), (mode_t == MODE_SOARING)
            init_clearance = torch.where(is_l, torch.empty_like(init_clearance).uniform_(0.05, 0.30),
                             torch.where(is_to, torch.empty_like(init_clearance).uniform_(0.05, 0.16),
                             torch.where(is_eg, torch.empty_like(init_clearance).uniform_(0.16, 0.90),
                             torch.where(is_ff, torch.empty_like(init_clearance).uniform_(1.10, 1.90),
                             torch.where(is_in, torch.empty_like(init_clearance).uniform_(0.50, 1.70),
                             torch.where(is_so, torch.empty_like(init_clearance).uniform_(0.11, 0.22), init_clearance))))))
            vx0 = torch.where(is_l, torch.empty_like(vx0).uniform_(0.0, 5.2),
                  torch.where(is_to, torch.empty_like(vx0).uniform_(0.5, 4.2),
                  torch.where(is_eg, torch.empty_like(vx0).uniform_(4.5, 6.5),
                  torch.where(is_ff, torch.empty_like(vx0).uniform_(6.5, 9.5),
                  torch.where(is_in, torch.empty_like(vx0).uniform_(4.8, 8.5),
                  torch.where(is_so, torch.empty_like(vx0).uniform_(4.5, 5.8), vx0))))))

            self.h_target_filtered = torch.where(is_eg | is_ff, torch.full_like(init_clearance, FREE_FLIGHT_ALTITUDE),
                                     torch.where(is_so, torch.full_like(init_clearance, RIDE_HEIGHT_SOARING),
                                     torch.where(is_l & (vx0 < 1.5), torch.full_like(init_clearance, 0.04), torch.full_like(init_clearance, RIDE_HEIGHT_OPTIMAL))))
            self.v_target_filtered = torch.where(is_l, torch.zeros_like(vx0), torch.where(is_ff, torch.full_like(vx0, 8.50), torch.full_like(vx0, self.target_speed)))
            self.motors = torch.where(init_clearance > 0.24, torch.full_like(vx0, HOVER_WEIGHT_NOM * self.mass_scale / 4.0), torch.full_like(vx0, 1.5)).repeat(1, 4)
            self.pusher_thrust = torch.where((vx0 > 3.0) & (~is_so) & (~is_l), torch.full_like(vx0, 2.5), torch.zeros_like(vx0))

        self.clearance_hold = init_clearance.clone().clamp(0.04, self.perception.radar_r_max)
        eta0, _, _, _, _, _ = self.waves.evaluate_wave_kinematics(torch.cat([init_x, init_y, torch.zeros_like(init_x)], dim=-1), self.sim_time, wave_scale=self.wave_curriculum_scale)
        self.pos, self.vel, self.omega, self.rot = torch.cat([init_x, init_y, eta0 + init_clearance], dim=-1), torch.cat([vx0, vy0, vz0], dim=-1), omega0, euler_to_rot(phi0, theta0, psi0)

        self.rotor_tilts = torch.zeros((N, 4, 2), device=self.device)
        self.nu_states = torch.zeros((N, 4, 3), device=self.device); self.nu_states[:, :, 0] = V_INDUCED_HOVER * 0.20
        self.ekf.reset(N, self.pos, self.vel, self.rot)

        self.follower_accel_prev = torch.zeros((N, 3), device=self.device)
        self.a_pilot_prev = torch.zeros((N, 3), device=self.device); self.a_pilot_filtered = torch.zeros((N, 3), device=self.device)
        self.a_pre_dob_prev = torch.zeros((N, 3), device=self.device); self.l1_sat_prev = torch.zeros((N, 3), dtype=torch.bool, device=self.device)
        self.scull_throttle_cmd = torch.zeros((N, 1), device=self.device)
        self.guidance_cmd_buffer.clear()
        for _ in range(self.guidance_delay_steps + 1):
            self.guidance_cmd_buffer.append(torch.zeros((N, 3), device=self.device))

        self.is_vent_L = torch.zeros((N, 1), device=self.device); self.is_vent_R = torch.zeros((N, 1), device=self.device)
        self.hydro_surfaces = torch.zeros((N, 3), device=self.device); self.hydro_surfaces_actual = torch.zeros((N, 3), device=self.device)
        self.phi_des_filtered.zero_(); self.psi_des_filtered.zero_(); self.e_int_height.zero_(); self.e_int_y.zero_()

        self.last_d_sub_hydrostatic = torch.zeros((N, 1), device=self.device); self.last_p_static_decoupled = torch.full((N, 1), float(P_ATM), device=self.device)
        self.last_bernoulli_bias_cm = torch.zeros((N, 1), device=self.device); self.last_radar_raw = init_clearance.clone()
        self.last_radar_locked = torch.ones((N, 1), dtype=torch.bool, device=self.device); self.last_spray_dropout = torch.zeros((N, 1), device=self.device)
        self.last_multipath_rejected = torch.zeros((N, 1), device=self.device); self.last_dvl_active = torch.zeros((N, 1), dtype=torch.bool, device=self.device)
        self.last_flow_active = torch.zeros((N, 1), dtype=torch.bool, device=self.device); self.last_eta_fwd_meas = torch.zeros((N, 1), device=self.device)
        self.last_eta_fwd_valid = torch.ones((N, 1), dtype=torch.bool, device=self.device); self.last_eta_prev = torch.zeros((N, 1), device=self.device)
        self.last_deta_dx_prev = torch.zeros((N, 1), device=self.device); self.last_w_up_prev = torch.zeros((N, 1), device=self.device)
        self.last_deta_dx_nadir = torch.zeros((N, 1), device=self.device); self.last_w_up_nadir = torch.zeros((N, 1), device=self.device)
        self.last_look_dist = torch.full((N, 1), 0.75, device=self.device)

        self.radar_blind_steps = torch.zeros((N, 1), dtype=torch.long, device=self.device)
        self.dvl_blind_steps = torch.zeros((N, 1), dtype=torch.long, device=self.device)
        self.aeration_steps = torch.zeros((N, 1), dtype=torch.long, device=self.device)

        self.num_obs = 6
        self.obs_x = init_x + torch.tensor([[12.0, 24.0, 36.0, 48.0, 60.0, 72.0]], device=self.device).repeat(N, 1) + torch.empty((N, 6), device=self.device).uniform_(-2.0, 2.0)
        self.obs_y = torch.empty((N, 6), device=self.device).uniform_(-0.65, 0.65)
        self.obs_r = torch.empty((N, 6), device=self.device).uniform_(0.35, 0.55)
        self.y_ref = torch.zeros((N, 1), device=self.device)

        self.u1 = torch.zeros((N, 1), device=self.device); self.u2 = torch.zeros((N, 1), device=self.device)
        self.soc = torch.full((N, 1), 0.95, device=self.device); self.t_cell = torch.full((N, 1), T_AMBIENT_K, device=self.device)
        self.v_batt = torch.full((N, 1), BATTERY_V0, device=self.device); self.metrics = {}

    def step(self, actor_net, ncm_net, export_diagnostics=False):
        N = self.num_envs
        self.sim_time = self.sim_time + self.dt
        self.step_count += 1
        ambient_gust = self.waves.step_gust()

        t_cur = float(self.sim_time[0, 0].item())
        if self.fault_scenario is not None and t_cur >= self.fault_time:
            self.fault_active = True
            if self.fault_scenario == "motor_loss":
                mask = torch.ones((N, 4), device=self.device); mask[:, 2] = 0.65
                self.robot.motor_fault_scale = mask
            elif self.fault_scenario == "gimbal_jam":
                self.robot.gimbal_jammed = True
            elif self.fault_scenario == "elevon_jam" and not self.fixed_foil:
                self.hydro_surfaces[:, 0] = math.radians(10.0)
            elif self.fault_scenario == "radar_blind":
                self.radar_blind_steps.fill_(9999)

        morph_dict = self.morphology()
        self.mass = morph_dict["total_mass"].repeat(N, 1) * self.mass_noise * self.mass_scale

        eta_raw, deta_dx, deta_dy, deta_dt, w_updraft, v_water_orb = self.waves.evaluate_wave_kinematics(self.pos, self.sim_time, wave_scale=self.wave_curriculum_scale)
        motor_thrusts_total = torch.sum(self.motors, dim=-1, keepdim=True)
        dimple = (motor_thrusts_total / (2.0 * RHO_WATER_NOM * GRAVITY * TOTAL_ROTOR_AREA)) * torch.exp(-F.relu(self.pos[:, 2:3] - eta_raw) / 0.14)
        eta = eta_raw - dimple

        if self.harsh_physics:
            new_aer = (torch.rand((N, 1), device=self.device) < 0.008) & (self.aeration_steps == 0)
            self.aeration_steps = torch.where(new_aer, torch.randint(10, 30, (N, 1), device=self.device), (self.aeration_steps - 1).clamp(min=0))
            rho_eff = torch.where(self.aeration_steps > 0, self.rho_water_local * 0.55, self.rho_water_local)
        else:
            rho_eff = self.rho_water_local

        is_soaring = (self.current_mode == MODE_SOARING)
        (f_wing, f_hydro, tau_hydro_pitch, tau_hydro_roll, tau_hydro_yaw, h_wing, d_sub_physical, f_slam, f_buoy,
         self.is_vent_L, self.is_vent_R, is_ventilated_overall, wind_shear_factor, strut_root_moment, aoa_tensors, alpha_foil) = self.aero_hydro(
            self.pos, self.vel, self.rot, self.omega, eta, deta_dt, w_updraft, v_water_orb, ambient_gust,
            self.is_vent_L, self.is_vent_R, self.foil_decalage, rho_eff, self.hydro_surfaces,
            motor_thrusts_total, morph_dict, self.sim_time, is_soaring, scull_throttle=self.scull_throttle_cmd,
            wave_field=self.waves, wave_scale=self.wave_curriculum_scale
        )

        with torch.no_grad():
            rot_T = self.rot.transpose(-1, -2)
            v_rel_b = torch.matmul(rot_T, (self.vel - v_water_orb).unsqueeze(-1)).squeeze(-1)
            accel_b_true = torch.matmul(rot_T, (self.follower_accel_prev + torch.tensor([0.0, 0.0, GRAVITY], device=self.device)).unsqueeze(-1)).squeeze(-1)
            vib = 1.0 + 0.04 * motor_thrusts_total
            meas_accel_body = accel_b_true + self.true_ba + torch.randn_like(accel_b_true) * (0.02 * vib)
            meas_gyro_body = self.omega + self.true_bw + torch.randn_like(self.omega) * (0.008 * vib)
            self.ekf.propagate(meas_accel_body, meas_gyro_body)

            if (self.step_count - 1) % self.sensor_period_steps == 0:
                v_body_est = torch.matmul(quat_to_rot(self.ekf.q_hat).transpose(-1, -2), self.ekf.v_hat.unsqueeze(-1)).squeeze(-1)
                d_sub_hyd, p_stat_dec, _, bern_bias = self.perception.sample_piezoresistive_tap(d_sub_physical, v_rel_b, v_body_est, self.rho_water_local)
                radar_raw, spray_drop, radar_locked, mp_rej = self.perception.process_fmcw_radar_dsp(h_wing, deta_dx, deta_dy, v_rel_b[:, 0:1], v_rel_b[:, 2:3])

                if self.harsh_physics:
                    trig_r = ((f_slam > 15.0) | (torch.rand((N, 1), device=self.device) < 0.008)) & (self.radar_blind_steps == 0)
                    self.radar_blind_steps = torch.where(trig_r, torch.randint(33, 120, (N, 1), device=self.device), (self.radar_blind_steps - 1).clamp(min=0))
                    radar_locked = radar_locked & (self.radar_blind_steps == 0)
                    spray_drop = torch.where(self.radar_blind_steps > 0, torch.ones_like(spray_drop), spray_drop)
                    trig_d = ((is_ventilated_overall.bool()) | (f_buoy > 2.0) | (torch.rand((N, 1), device=self.device) < 0.005)) & (self.dvl_blind_steps == 0)
                    self.dvl_blind_steps = torch.where(trig_d, torch.randint(50, 150, (N, 1), device=self.device), (self.dvl_blind_steps - 1).clamp(min=0))
                elif self.fault_scenario == "radar_blind" and self.fault_active:
                    radar_locked = torch.zeros_like(radar_locked); spray_drop = torch.ones_like(spray_drop)

                self.last_d_sub_hydrostatic, self.last_p_static_decoupled, self.last_bernoulli_bias_cm = d_sub_hyd, p_stat_dec, bern_bias
                self.last_radar_raw, self.last_radar_locked, self.last_spray_dropout, self.last_multipath_rejected = radar_raw, radar_locked, spray_drop, mp_rej
                meas_radar = torch.where(radar_locked, radar_raw, self.clearance_hold)

                mag_meas = torch.matmul(rot_T, MAG_FIELD_WORLD.to(self.device).repeat(N, 1).unsqueeze(-1)).squeeze(-1) + torch.randn_like(self.omega) * 0.02
                dvl_meas = v_rel_b + torch.randn_like(v_rel_b) * 0.03
                cel = self.dispersion_observer.omega_dom / torch.clamp(self.dispersion_observer.k_dom, min=0.1)
                h_fl = torch.clamp(h_wing, min=0.08)
                flow_rates = torch.cat([-(v_rel_b[:, 1:2]) / h_fl, +(v_rel_b[:, 0:1] - cel) / h_fl], dim=-1) + torch.randn((N, 2), device=self.device) * 0.04

                self.ekf.update_magnetometer(mag_meas)
                is_sub_s = (d_sub_hyd > 0.015)
                dvl_lock = (~is_ventilated_overall.bool()) & (torch.rand_like(is_ventilated_overall) > 0.01) & (self.dvl_blind_steps == 0)
                self.ekf.update_dvl(dvl_meas, is_sub_s, (~dvl_lock).float())
                self.ekf.update_optical_flow(flow_rates, meas_radar, cel, (d_sub_hyd <= 0.015))
                self.last_dvl_active, self.last_flow_active = is_sub_s & dvl_lock, (~is_sub_s) & (meas_radar < 1.8)
            else:
                d_sub_hyd, p_stat_dec, bern_bias = self.last_d_sub_hydrostatic, self.last_p_static_decoupled, self.last_bernoulli_bias_cm
                radar_raw, radar_locked, spray_drop, mp_rej = self.last_radar_raw, self.last_radar_locked, self.last_spray_dropout, self.last_multipath_rejected
                meas_radar = torch.where(radar_locked, radar_raw, self.clearance_hold)
                is_sub_s = (d_sub_hyd > 0.015)

            dvl_active, flow_active = self.last_dvl_active, self.last_flow_active
            gnss_blocked = (self.gnss_denied_window is not None) and (self.gnss_denied_window[0] <= t_cur <= self.gnss_denied_window[1])
            if (self.step_count % self.gnss_period_steps == 0) and (not gnss_blocked):
                infl = 1.0 + 8.0 * torch.exp(-torch.clamp(h_wing, 0.01, 10.0) / 0.25)
                self.ekf.update_gnss(self.pos + torch.randn_like(self.pos) * 0.05 * torch.sqrt(infl), self.vel + torch.randn_like(self.vel) * 0.03 * torch.sqrt(infl), meas_radar)

            est_pos, est_vel, est_rot, est_omega = self.ekf.p_hat, self.ekf.v_hat, quat_to_rot(self.ekf.q_hat), meas_gyro_body - self.ekf.bw_hat
            a_est_world = torch.matmul(est_rot, (meas_accel_body - self.ekf.ba_hat).unsqueeze(-1)).squeeze(-1) + torch.tensor([0.0, 0.0, -GRAVITY], device=self.device).view(1, 3)

            h_hyd_proj = F.relu(morph_dict["strut_length"].detach() * est_rot[:, 2, 2:3].clamp(min=0.30) - d_sub_hyd)
            est_clearance = torch.where(radar_locked, meas_radar, torch.where(is_sub_s, h_hyd_proj, torch.clamp(est_pos[:, 2:3], 0.04, 10.0)))
            self.clearance_hold = est_clearance.detach().clamp(0.04, self.perception.radar_r_max)
            est_h_dot = est_vel[:, 2:3]

            ekf_err_h, ekf_err_vx = est_clearance - h_wing, est_vel[:, 0:1] - self.vel[:, 0:1]
            R_ekf = torch.bmm(self.rot.transpose(-1, -2), est_rot)
            ekf_err_att_deg = (torch.acos((((R_ekf[:, 0, 0] + R_ekf[:, 1, 1] + R_ekf[:, 2, 2]) - 1.0) * 0.5).clamp(-0.99999, 0.99999)) * (180.0 / math.pi)).unsqueeze(-1)

            if (self.step_count - 1) % self.sensor_period_steps == 0:
                ray_w = torch.matmul(est_rot, self.waves.ray_body_vec).squeeze(-1)
                est_s = torch.clamp(est_clearance / (-ray_w[:, 2:3]).clamp(min=0.06), 0.25, 6.0)
                look_dist = safe_norm(est_s * ray_w[:, :2], dim=-1, keepdim=True)
                eta_fwd_meas, eta_fwd_valid = self.waves.raycast_forward_surface_pose(self.pos, self.rot, eta, self.sim_time, self.wave_curriculum_scale)
                eta_prev, deta_dx_prev, w_up_prev = self.dispersion_observer.update_and_predict(est_clearance, est_pos[:, 2:3], eta_fwd_meas, eta_fwd_valid, look_dist, est_vel[:, 0:1], 1.25)
                self.last_eta_fwd_meas, self.last_eta_fwd_valid = eta_fwd_meas, eta_fwd_valid
                self.last_eta_prev, self.last_deta_dx_prev, self.last_w_up_prev = eta_prev, deta_dx_prev, w_up_prev
                self.last_deta_dx_nadir, self.last_w_up_nadir = self.dispersion_observer.deta_dx_nadir, self.dispersion_observer.w_up_nadir
                self.last_look_dist = look_dist
            else:
                eta_fwd_meas, eta_prev, deta_dx_prev, w_up_prev, look_dist = self.last_eta_fwd_meas, self.last_eta_prev, self.last_deta_dx_prev, self.last_w_up_prev, self.last_look_dist

            sigma_l1 = self.l1_observer.step(a_est_world, self.a_pre_dob_prev, self.dt, sat_mask=self.l1_sat_prev)

            obstacle_hit = torch.zeros((N, 1), dtype=torch.bool, device=self.device)
            obs_active_cbf = torch.zeros((N, 1), dtype=torch.bool, device=self.device)
            obs_avoid_dir, oy_n, or_n = torch.ones((N, 1), device=self.device), torch.zeros((N, 1), device=self.device), torch.full((N, 1), 0.4, device=self.device)
            if self.enable_obstacles:
                dist_true = torch.sqrt((self.obs_x - self.pos[:, 0:1])**2 + (self.obs_y - self.pos[:, 1:2])**2)
                obstacle_hit = ((dist_true < self.obs_r) & (d_sub_physical > 0.005)).any(dim=-1, keepdim=True)
                passed = self.obs_x < (self.pos[:, 0:1] - 3.0)
                if passed.any():
                    far_x = torch.max(self.obs_x, dim=-1, keepdim=True).values + 12.0 + torch.empty((N, self.num_obs), device=self.device).uniform_(0.0, 4.0)
                    self.obs_x = torch.where(passed, far_x, self.obs_x)
                    self.obs_y = torch.where(passed, torch.empty_like(self.obs_y).uniform_(-0.65, 0.65), self.obs_y)

                dx_e = self.obs_x - est_pos[:, 0:1]
                dx_m = torch.where((dx_e > 0.2) & (dx_e < 11.0) & (torch.rand_like(dx_e) < 0.90), dx_e, torch.full_like(dx_e, 1e6))
                idx_n = torch.argmin(dx_m, dim=-1, keepdim=True)
                dx_n, oy_n, or_n = torch.gather(dx_m, -1, idx_n), torch.gather(self.obs_y, -1, idx_n), torch.gather(self.obs_r, -1, idx_n)
                obs_avoid_dir = torch.where(oy_n >= 0.0, -torch.ones_like(oy_n), torch.ones_like(oy_n))
                need_av = (dx_n < 10.5) & (torch.abs(est_pos[:, 1:2] - oy_n) < (or_n + 0.65))
                self.y_ref = self.y_ref + (self.dt / (0.25 + self.dt)) * (torch.where(need_av, oy_n + obs_avoid_dir * (or_n + 0.80), torch.zeros_like(oy_n)) - self.y_ref)
                obs_active_cbf = (dx_n < 4.5) & is_sub_s & (torch.abs(est_pos[:, 1:2] - oy_n) < (or_n + 0.55))
            else:
                self.y_ref.zero_()

        is_loiter = (self.current_mode == MODE_LOITER); is_takeoff = (self.current_mode == MODE_TAKEOFF)
        is_foiling = (self.current_mode == MODE_FOILING); is_egress = (self.current_mode == MODE_EGRESS)
        is_flight = (self.current_mode == MODE_FREE_FLIGHT); is_ingress = (self.current_mode == MODE_INGRESS)
        is_soaring = (self.current_mode == MODE_SOARING)

        v_tgt_raw = torch.where(is_loiter, 0.0, torch.where(is_flight, 8.50, self.target_speed))
        loiter_desc = is_loiter & (est_vel[:, 0:1] < 1.5)
        h_foil_cmd = torch.where((d_sub_hyd > 0.020) & (a_est_world[:, 2:3] < -1.8) & is_sub_s, RIDE_HEIGHT_REPRIME, RIDE_HEIGHT_OPTIMAL)
        h_tgt_raw = torch.where(is_loiter, torch.where(loiter_desc, 0.04, 0.18),
                    torch.where(is_takeoff, 0.185, torch.where(is_foiling, h_foil_cmd,
                    torch.where(is_soaring, RIDE_HEIGHT_SOARING, torch.where(is_egress | is_flight, FREE_FLIGHT_ALTITUDE, RIDE_HEIGHT_OPTIMAL)))))

        a_bl = self.dt / (0.15 + self.dt)
        self.h_target_filtered += a_bl * (h_tgt_raw - self.h_target_filtered)
        self.v_target_filtered += a_bl * (v_tgt_raw - self.v_target_filtered)
        self.current_mode = torch.where(is_takeoff & (est_vel[:, 0:1] > 4.6) & is_sub_s, torch.full_like(self.current_mode, MODE_FOILING), self.current_mode)
        self.current_mode = torch.where(is_egress & (est_clearance > (FREE_FLIGHT_ALTITUDE - 0.15)), torch.full_like(self.current_mode, MODE_FREE_FLIGHT), self.current_mode)
        self.current_mode = torch.where(is_ingress & is_sub_s, torch.full_like(self.current_mode, MODE_FOILING), self.current_mode)

        e_speed, e_height, e_y_track = est_vel[:, 0:1] - self.v_target_filtered, est_clearance - self.h_target_filtered, est_pos[:, 1:2] - self.y_ref
        self.e_int_height = torch.where((is_foiling | is_takeoff | is_soaring), torch.clamp(self.e_int_height + e_height * self.dt, -0.12, 0.12), self.e_int_height * 0.95)
        self.e_int_y = torch.where((is_foiling | is_takeoff | is_soaring) & (est_vel[:, 0:1] > 1.0), torch.clamp(self.e_int_y + e_y_track * self.dt, -1.5, 1.5), self.e_int_y * 0.95)

        actor_obs = torch.cat([
            e_speed, e_height, self.e_int_height, e_y_track, est_h_dot, self.v_target_filtered / 8.50,
            d_sub_hyd, self.h_target_filtered, est_vel, est_omega, est_rot[:, 2, :],
            eta_prev - (est_pos[:, 2:3] - est_clearance), deta_dx_prev, w_up_prev
        ], dim=-1)

        a_neural = actor_net(actor_obs)
        self.guidance_cmd_buffer.append(a_neural)
        self.a_pilot_filtered = self.pilot_filter_a * self.a_pilot_filtered + (1.0 - self.pilot_filter_a) * self.guidance_cmd_buffer[0]

        state_err = torch.cat([torch.zeros((N, 1), device=self.device), e_y_track, e_height, e_speed, est_vel[:, 1:2], est_h_dot], dim=-1)
        M_5 = ncm_net(state_err)[:, 1:, 1:]

        h_dot_rel = est_vel[:, 2:3] - (self.last_deta_dx_nadir * est_vel[:, 0:1] + self.last_w_up_nadir)
        h_min_m = torch.where(loiter_desc | is_takeoff, 0.01, RIDE_HEIGHT_MIN)
        h_max_m = torch.where(is_egress | is_flight | is_ingress, 4.00, torch.where(is_foiling | is_soaring, RIDE_HEIGHT_FOILING_MAX, RIDE_HEIGHT_WIG_MAX))
        b_cbf_crash = (4.00 * h_dot_rel + 12.0 * (est_clearance - h_min_m)).unsqueeze(-1)
        b_cbf_ceil = (-4.00 * h_dot_rel + 12.0 * torch.clamp(h_max_m - est_clearance, -0.5, 1.0)).unsqueeze(-1)

        e_xy = state_err[:, [1, 3, 4]]; m_e_xy = torch.bmm(M_5[:, [0, 2, 3], :][:, :, [0, 2, 3]], e_xy.unsqueeze(-1)).squeeze(-1)
        A_ncm_xy = torch.cat([m_e_xy[:, 1:3], torch.zeros((N, 1), device=self.device)], dim=-1).unsqueeze(1)
        b_ncm_xy = (-0.15 * torch.sum(e_xy * m_e_xy, dim=-1, keepdim=True) - m_e_xy[:, 0:1] * e_xy[:, 2:3]).unsqueeze(-1)

        e_z = state_err[:, [2, 5]]; m_e_z = torch.bmm(M_5[:, [1, 4], :][:, :, [1, 4]], e_z.unsqueeze(-1)).squeeze(-1)
        A_ncm_z = torch.cat([torch.zeros((N, 2), device=self.device), m_e_z[:, 1:2]], dim=-1).unsqueeze(1)
        b_ncm_z = (-0.15 * torch.sum(e_z * m_e_z, dim=-1, keepdim=True) - m_e_z[:, 0:1] * e_z[:, 1:2]).unsqueeze(-1)

        c_A, c_b = [self.A_cbf_crash, self.A_cbf_ceil, A_ncm_xy, A_ncm_z], [b_cbf_crash, b_cbf_ceil, b_ncm_xy, b_ncm_z]
        if self.harsh_physics:
            v_be = torch.matmul(est_rot.transpose(-1, -2), est_vel.unsqueeze(-1)).squeeze(-1)
            a_fe = torch.atan2(-v_be[:, 2:3], v_be[:, 0:1].clamp(min=0.5)) + math.radians(0.5 * (FOIL_DECALAGE_DEG[0] + FOIL_DECALAGE_DEG[1]))
            v_cs = torch.sqrt(2.0 * (P_ATM - P_VAPOR_WATER) / (1.10 * RHO_WATER_NOM * (1.0 + 2.4 * F.relu(a_fe))))
            c_A.append(self.A_cbf_cav); c_b.append(torch.where(is_sub_s, 2.5 * (v_cs - est_vel[:, 0:1]), torch.full_like(v_cs, 20.0)).unsqueeze(-1))
        if self.enable_obstacles:
            A_obs = torch.cat([torch.zeros((N, 1), device=self.device), -obs_avoid_dir, torch.zeros((N, 1), device=self.device)], dim=-1).unsqueeze(1)
            b_obs = torch.where(obs_active_cbf, 4.0 * (obs_avoid_dir * est_vel[:, 1:2]) + 10.0 * (obs_avoid_dir * (est_pos[:, 1:2] - oy_n) - (or_n + 0.45)), torch.full_like(oy_n, 20.0)).unsqueeze(-1)
            c_A.append(A_obs); c_b.append(b_obs)

        A_all, b_all = torch.cat(c_A, dim=1), torch.cat(c_b, dim=1)
        if self.use_embedded_qp and not torch.is_grad_enabled():
            da_max = float(MAX_JERK_ACCEL * self.dt)
            a_safe, qp_override, active_qp_barrier_idx = self.embedded_qp.solve_tensor(torch.max(torch.min(self.a_pilot_filtered, self.a_pilot_prev + da_max), self.a_pilot_prev - da_max), A_all, b_all)
            slack = torch.zeros((N, A_all.shape[1]), device=self.device)
        else:
            a_safe, qp_override, slack, active_qp_barrier_idx, _, _ = self.safety_filter(self.a_pilot_filtered, self.a_pilot_prev, A_all, b_all, self.dt)
        self.a_pilot_prev = a_safe

        sub_fac = torch.clamp(d_sub_hyd / 0.04, 0.0, 1.0)
        phi_max = sub_fac * math.radians(6.5) + (1.0 - sub_fac) * math.radians(18.0)
        v_hor = safe_norm(est_vel[:, :2], dim=-1, keepdim=True).clamp(min=0.5)
        q_w_e, q_a_e = 0.5 * RHO_WATER_NOM * (v_hor**2), 0.5 * RHO_AIR * (v_hor**2)
        f_h_drag = q_w_e * morph_dict["foil_area"] * FOIL_CD0 * sub_fac
        f_drag_est = f_h_drag + q_a_e * morph_dict["wing_area"] * (WING_CD0 + 0.03)
        nom_mass = float(morph_dict["total_mass"].detach().item()) * self.mass_scale

        v_b_est = torch.matmul(est_rot.transpose(-1, -2), est_vel.unsqueeze(-1)).squeeze(-1)
        a_f_est = torch.atan2(-v_b_est[:, 2:3], v_b_est[:, 0:1].clamp(min=0.5)) + math.radians(0.5 * (FOIL_DECALAGE_DEG[0] + FOIL_DECALAGE_DEG[1]))
        q_ACLa = q_w_e * morph_dict["foil_area"] * FOIL_CL_ALPHA
        f_lift_unc = q_ACLa * a_f_est * sub_fac

        phi_wig = 1.0 + 0.35 / (0.10 + 2.2 * torch.pow(est_clearance.clamp(min=0.03) / morph_dict["wing_chord"], 1.15))
        f_wing_lift_est = (q_a_e * morph_dict["wing_area"] * torch.clamp((WING_CL0 + 4.5 * (torch.atan2(-v_b_est[:, 2:3], v_b_est[:, 0:1].clamp(min=0.5)) + 0.03)) * phi_wig, -0.3, 1.4)).clamp(0.0, 15.0)
        tau_h_pitch_est = f_h_drag * morph_dict["strut_length"]

        min_r_floor = torch.where(is_soaring, torch.full((N, 1), 5.6, device=self.device), torch.full((N, 1), 6.8, device=self.device))
        des_fz = nom_mass * (GRAVITY + a_safe[:, 2:3] - sigma_l1[:, 2:3])
        over_lift = F.relu(f_lift_unc - (des_fz - min_r_floor - f_wing_lift_est))
        q_ACLd = q_w_e * morph_dict["foil_area"] * HYDRO_CL_DELTA * sub_fac
        coll_max, foil_gate = 0.8 * MAX_HYDRO_DEFLECTION, (sub_fac > 0.05).float() * (est_vel[:, 0:1] > 1.0).float()

        if self.fixed_foil:
            delta_collective = torch.zeros_like(over_lift)
            theta_shed = torch.clamp(over_lift / (q_ACLa + 50.0), max=math.radians(6.0)) * foil_gate
            f_h_lift_est = torch.clamp(f_lift_unc, -1.0, 2.0 * FOIL_LIFT_CLAMP_PER_SIDE)
        else:
            delta_collective = -torch.clamp(over_lift / (q_ACLd + 1.0), max=coll_max) * foil_gate
            theta_shed = torch.clamp(F.relu(over_lift - q_ACLd * coll_max) / (q_ACLa + 50.0), max=math.radians(6.0)) * foil_gate
            f_h_lift_est = torch.clamp(f_lift_unc + q_ACLd * delta_collective, -1.0, 2.0 * FOIL_LIFT_CLAMP_PER_SIDE)

        raw_f_rotors = des_fz - f_h_lift_est - f_wing_lift_est
        f_rotors_needed = torch.maximum(raw_f_rotors, min_r_floor).clamp(max=float(nom_mass * GRAVITY + 6.0))

        psi_est = torch.atan2(est_rot[:, 1:2, 0], est_rot[:, 0:1, 0])
        c_p, s_p = torch.cos(psi_est), torch.sin(psi_est)
        ax_w, ay_w = a_safe[:, 0:1] - sigma_l1[:, 0:1], a_safe[:, 1:2] - 1.4 * self.e_int_y - sigma_l1[:, 1:2]
        Fx_dyn, Fy_dyn = nom_mass * (c_p * ax_w + s_p * ay_w), nom_mass * (-s_p * ax_w + c_p * ay_w)

        f_g_cap = f_rotors_needed * (self.robot.gimbal_tan * 0.70)
        Fx_g_est = torch.where(is_soaring, torch.zeros_like(Fx_dyn), torch.clamp(Fx_dyn, -f_g_cap, f_g_cap))
        soar_req = torch.zeros_like(Fx_dyn) if (self.fixed_foil or self.disable_scull) else torch.where(is_soaring & is_sub_s, torch.clamp(Fx_dyn + f_drag_est, 0.0, 22.0), torch.zeros_like(Fx_dyn))
        self.scull_throttle_cmd = (soar_req / 22.0).clamp(0.0, 1.0).detach()

        pusher_off = is_loiter & (est_vel[:, 0:1] < 0.8)
        target_pusher = torch.where(pusher_off | is_soaring, torch.zeros_like(Fx_dyn), torch.clamp(Fx_dyn + f_drag_est - Fx_g_est, 0.0, MAX_PUSHER_THRUST))
        Fx_cmd_g = Fx_dyn + f_drag_est - target_pusher - soar_req
        Fx_res, Fy_res = Fx_cmd_g - torch.clamp(Fx_cmd_g, -f_g_cap, f_g_cap), Fy_dyn - torch.clamp(Fy_dyn, -f_g_cap, f_g_cap)

        f_vnet = (nom_mass * GRAVITY - f_h_lift_est - f_wing_lift_est).clamp(min=4.0)
        raw_phi = torch.atan2(-Fy_res, f_vnet)
        lat_dmp = 0.22 * torch.clamp(torch.atan2(est_vel[:, 1:2], est_vel[:, 0:1].clamp(min=1.0)), -math.radians(10.0), math.radians(10.0))
        self.phi_des_filtered += (self.dt / (0.15 + self.dt)) * (torch.clamp(raw_phi + lat_dmp, -phi_max, phi_max) - self.phi_des_filtered)
        phi_des = self.phi_des_filtered
        theta_des = torch.clamp(torch.atan2(Fx_res, f_vnet) + torch.where(is_egress, -math.radians(4.5), torch.where(is_ingress, -math.radians(1.5), torch.zeros_like(Fx_res))) + theta_shed, -math.radians(14.0), math.radians(24.0))

        self.a_pre_dob_prev = torch.cat([a_safe[:, 0:1], a_safe[:, 1:2] - 1.4 * self.e_int_y, a_safe[:, 2:3]], dim=-1).detach()
        self.l1_sat_prev = torch.cat([
            (theta_des <= math.radians(-13.8)) | (theta_des >= math.radians(23.8)) | (pusher_off & (Fx_dyn > f_g_cap)),
            torch.abs(raw_phi + lat_dmp) > phi_max,
            (raw_f_rotors < min_r_floor) | (raw_f_rotors > float(nom_mass * GRAVITY + 6.0))
        ], dim=-1).detach()

        psi_tgt = -torch.clamp(e_y_track * 0.12 + 0.15 * est_vel[:, 1:2], -math.radians(5.0), math.radians(5.0)) * ((est_vel[:, 0:1] - 0.5) / 1.5).clamp(0.0, 1.0)
        self.psi_des_filtered += (self.dt / (0.45 + self.dt)) * (psi_tgt - self.psi_des_filtered)

        Rd = euler_to_rot(phi_des, theta_des, self.psi_des_filtered)
        R_err = torch.matmul(Rd.transpose(-1, -2), est_rot) - torch.matmul(est_rot.transpose(-1, -2), Rd)
        tau_des = torch.cat([
            -self.k_R[0, 0] * R_err[:, 2, 1:2] * 0.5 - self.k_omega[0, 0] * est_omega[:, 0:1],
            -self.k_R[0, 1] * R_err[:, 0, 2:3] * 0.5 - self.k_omega[0, 1] * est_omega[:, 1:2] - tau_h_pitch_est,
            -self.k_R[0, 2] * R_err[:, 1, 0:1] * 0.5 - self.k_omega[0, 2] * est_omega[:, 2:3]
        ], dim=-1)

        ige_est = 1.0 / (1.0 - torch.clamp((PROP_RADIUS / (4.0 * est_clearance.clamp(min=0.04)))**2, max=0.35))
        (target_motors, target_tilts, hydro_cmd, headroom_min, is_clipped_min, is_clipped_max,
         torque_scale, fx_gimbal_alloc, fy_gimbal_alloc) = self.robot.allocate_wrench_hybrid(
            Fx_cmd_g, Fy_dyn, f_rotors_needed, tau_des, self.v_batt, ige_est, d_sub_hyd, est_vel[:, 0:1], is_soaring, morph_dict, delta_collective=delta_collective
        )

        if self.fault_scenario == "elevon_jam" and self.fault_active and not self.fixed_foil:
            hydro_cmd[:, 0] = math.radians(10.0)

        d_surf_req = hydro_cmd - self.hydro_surfaces_actual
        d_surf_move = torch.where(torch.abs(d_surf_req) > SERVO_BACKLASH_RAD, torch.clamp(d_surf_req, -SERVO_RATE_LIMIT * self.dt, SERVO_RATE_LIMIT * self.dt), torch.zeros_like(d_surf_req))
        self.hydro_surfaces_actual = self.hydro_surfaces_actual + d_surf_move
        self.hydro_surfaces = self.hydro_surfaces_actual

        (self.pos, self.vel, self.rot, self.omega, self.motors, self.rotor_tilts, self.pusher_thrust,
         self.nu_states, omega_props, accel, _) = self.robot.integrate(
            self.pos, self.vel, self.rot, self.omega, self.motors, self.rotor_tilts, self.pusher_thrust,
            self.nu_states, target_motors, target_tilts, target_pusher, f_wing, f_hydro,
            tau_hydro_pitch, tau_hydro_roll, tau_hydro_yaw, ambient_gust * wind_shear_factor,
            eta, safe_norm(torch.cat([deta_dx, deta_dy], dim=-1), dim=-1, keepdim=True),
            self.mass, self.motor_tau_up, self.motor_tau_down,
            m_a_diag=aoa_tensors["m_a_diag"], I_a_diag=aoa_tensors["I_a_diag"],
            f_am_cor_world=aoa_tensors["f_am_coriolis_world"], v_batt=self.v_batt
        )
        self.follower_accel_prev = accel.detach()

        eta_post = eta + deta_dt * self.dt
        clearance_eval = (self.pos[:, 2:3] - eta_post).clamp(min=0.01)
        z_tip_L_p = self.pos[:, 2:3] + self.rot[:, 2, 1:2] * (morph_dict["foil_span"] * 0.25) - self.rot[:, 2, 2:3] * morph_dict["strut_length"]
        z_tip_R_p = self.pos[:, 2:3] - self.rot[:, 2, 1:2] * (morph_dict["foil_span"] * 0.25) - self.rot[:, 2, 2:3] * morph_dict["strut_length"]
        immersion_eval = 0.5 * ((eta_post - z_tip_L_p).clamp(0.0, 0.22) + (eta_post - z_tip_R_p).clamp(0.0, 0.22))
        phi_true_deg = torch.atan2(self.rot[:, 2, 1:2], self.rot[:, 2, 2:3]) * (180.0 / math.pi)

        v_air_x_b = torch.matmul(self.rot.transpose(-1, -2), self.vel.unsqueeze(-1)).squeeze(-1)[:, 0:1]
        p_elec, self.v_batt, i_batt, self.u1, self.u2, self.soc, self.t_cell = self.powertrain.step(
            self.motors, w_updraft, self.nu_states[:, :, 0], omega_props, self.u1, self.u2, self.soc, self.t_cell,
            self.dt, aoa_tensors["p_scull_actuator_watts"], pusher_thrust=self.pusher_thrust, v_air_x=v_air_x_b
        )

        if export_diagnostics:
            hm = torch.full_like(self.motors, (HOVER_WEIGHT_NOM * self.mass_scale) / 4.0)
            p_base, _, _, _, _, _, _ = self.powertrain.step(
                hm, torch.zeros_like(w_updraft), torch.full_like(self.nu_states[:, :, 0], V_INDUCED_HOVER),
                torch.sqrt((hm + 1e-6) / 1.1e-5).clamp(50.0, 1800.0), torch.zeros_like(self.u1), torch.zeros_like(self.u2),
                torch.full_like(self.soc, 0.95), torch.full_like(self.t_cell, T_AMBIENT_K), self.dt, torch.zeros_like(self.u1),
                pusher_thrust=torch.zeros_like(self.pusher_thrust), v_air_x=torch.zeros_like(v_air_x_b)
            )
            power_saved_pct = ((p_base - p_elec) / p_base) * 100.0
        else:
            power_saved_pct = torch.zeros((N, 1), device=self.device)

        rotor_tilt_deg = torch.asin(torch.sqrt(self.rotor_tilts[:, :, 0]**2 + self.rotor_tilts[:, :, 1]**2 + 1e-10).clamp(max=1.0)) * (180.0 / math.pi)
        self.metrics = {
            "mode": self.current_mode, "v_target_mode": self.v_target_filtered, "h_target_mode": self.h_target_filtered,
            "plant_damp_diag": aoa_tensors["plant_damp_diag"], "phi_true_deg": phi_true_deg,
            "ekf_err_h": ekf_err_h, "ekf_err_vx": ekf_err_vx, "ekf_err_att_deg": ekf_err_att_deg,
            "sigma_l1_vec": sigma_l1, "e_int_y": self.e_int_y, "fx_gimbal": fx_gimbal_alloc, "fy_gimbal": fy_gimbal_alloc,
            "clearance_eval": clearance_eval, "immersion_eval": immersion_eval, "f_hydro_lift_eval": f_hydro[:, 2:3],
            "vel_eval": self.vel, "pos_eval": self.pos, "omega_eval": self.omega, "h_wing": h_wing,
            "clearance_est": est_clearance, "immersion_depth": d_sub_hyd, "f_slam": f_slam, "f_buoy": f_buoy,
            "f_hydro_lift": f_hydro[:, 2:3], "f_wing_lift": f_wing[:, 2:3], "f_rotors": torch.sum(self.motors, dim=-1, keepdim=True),
            "f_pusher": self.pusher_thrust, "motors": self.motors, "rotor_tilts": self.rotor_tilts, "rotor_tilt_deg": rotor_tilt_deg,
            "hydro_surfaces": self.hydro_surfaces,
            "hydro_deflections": (math.degrees(self.hydro_surfaces[0, 0].item()), math.degrees(self.hydro_surfaces[0, 1].item()), math.degrees(self.hydro_surfaces[0, 2].item())),
            "cell_temp_c": self.t_cell[0, 0].item() - 273.15, "headroom_min": headroom_min,
            "is_clipped_min": is_clipped_min, "is_clipped_max": is_clipped_max, "power_saved_pct": power_saved_pct,
            "p_elec": p_elec, "v_batt": self.v_batt, "i_batt": i_batt, "soc": self.soc * 100.0,
            "qp_override": qp_override, "max_slack": slack, "actor_obs": actor_obs, "a_safe": a_safe, "a_neural": a_neural,
            "state_err": state_err, "pos": self.pos, "vel": self.vel, "eta": eta, "omega": self.omega,
            "e_R": torch.cat([R_err[:, 2, 1:2] * 0.5, R_err[:, 0, 2:3] * 0.5, R_err[:, 1, 0:1] * 0.5], dim=-1),
            "e_int_height": self.e_int_height, "is_vent_L": self.is_vent_L, "is_vent_R": self.is_vent_R,
            "is_ventilated": is_ventilated_overall[0, 0].item() > 0.5 if export_diagnostics else False,
            "h_dot_est": est_h_dot, "radar_dropout": spray_drop[0, 0].item() > 0.5 if export_diagnostics else False,
            "eta_preview": eta_prev, "eta_forward_meas": eta_fwd_meas, "eta_nadir_est": self.dispersion_observer.last_eta_nadir,
            "look_dist": look_dist, "k_dom": self.dispersion_observer.k_dom, "omega_dom": self.dispersion_observer.omega_dom,
            "sigma_l1_norm": safe_norm(sigma_l1, dim=-1, keepdim=True), "strut_moment_nm": strut_root_moment,
            "strut_yield_fail": aoa_tensors["strut_yield_fail"], "sigma_cav_min": aoa_tensors["sigma_cav_min"],
            "cav_active": aoa_tensors["cav_active"], "obstacle_hit": obstacle_hit, "y_ref": self.y_ref,
            "radar_locked": bool(radar_locked[0, 0].item() > 0.5) if export_diagnostics else True,
            "multipath_rejected": bool(mp_rej[0, 0].item() > 0.5) if export_diagnostics else False,
            "delta_collective_deg": math.degrees(delta_collective[0, 0].item()), "foil_lift_est": f_h_lift_est, "status_banner": "[NOMINAL]"
        }

        if export_diagnostics:
            phi_deg = math.degrees(math.atan2(est_rot[0, 2, 1].item(), est_rot[0, 2, 2].item()))
            theta_deg = math.degrees(math.asin(float(np.clip(-est_rot[0, 2, 0].item(), -0.9999, 0.9999))))
            psi_act_deg = math.degrees(math.atan2(est_rot[0, 1, 0].item(), est_rot[0, 0, 0].item()))
            psi_des_deg = math.degrees(self.psi_des_filtered[0, 0].item())
            k_val = max(0.01, self.dispersion_observer.k_dom[0, 0].item())
            min_cl, max_cl = is_clipped_min[0].cpu().numpy(), is_clipped_max[0].cpu().numpy()
            sat_str = "[MIN+MAX CLIPPED]" if (any(min_cl) and any(max_cl)) else ("[MIN FLOOR]" if any(min_cl) else ("[MAX CEIL]" if any(max_cl) else "[CLEAN]"))
            i_b = i_batt[0, 0].item()
            r0_v = BATTERY_R0_NOM * math.exp(ARRHENIUS_COEFF * (1.0 / max(270.0, self.t_cell[0, 0].item()) - 1.0 / T_AMBIENT_K))
            rem_wh = max(0.0, self.soc[0, 0].item() - 0.05) * (BATTERY_CAPACITY_COULOMBS / 3600.0) * max(20.0, self.v_batt[0, 0].item())
            p_w = max(1.0, p_elec[0, 0].item())
            with torch.no_grad():
                nn_raw = tuple(float(v) for v in torch.tanh(actor_net.net(actor_obs[0:1]))[0].cpu().numpy())
            mode_names = ["LOITER", "TAKEOFF", "FOILING", "EGRESS", "FLIGHT", "INGRESS", "SOARING"]
            a_f_v = aoa_tensors["alpha_foil"][0, 0].item()
            cav_sig = aoa_tensors["sigma_cav_min"][0, 0].item()
            b_idx = active_qp_barrier_idx[0, 0].item()
            b_name = "[FREE (Unconstrained)]" if qp_override[0, 0].item() < 0.10 else ("[CRASH BARRIER (Floor)]" if b_idx == 0 else ("[BREACH BARRIER (Ceiling)]" if b_idx == 1 else "[NCM CONTRACTION METRIC]"))
            f_soar_v, f_scull_v = aoa_tensors["f_soar_surge"][0, 0].item(), aoa_tensors["f_scull_thrust"][0, 0].item()
            banner = "[WAVE SOARING + SCULLING]" if (is_soaring[0, 0].item() and (f_soar_v + f_scull_v) > 1.0) else (
                     "[CAPSIZE RISK - HIGH ROLL]" if abs(phi_deg) > 28.0 else (
                     "[HULL CONTACT - PLANING]" if est_clearance[0, 0].item() < HULL_DRAFT_MAX + 0.01 else (
                     "[OPTIMAL WAVE FOILING]" if f_hydro[0, 2].item() > 8.0 else "[AIRBORNE FLIGHT]")))

            self.metrics.update({
                "status_banner": banner, "psi_des_deg": psi_des_deg, "psi_err_deg": psi_act_deg - psi_des_deg,
                "beta_sideslip_deg": math.degrees(math.atan2(est_vel[0, 1].item(), max(0.1, est_vel[0, 0].item()))),
                "wavelength_m": (2.0 * math.pi) / k_val, "celerity_mps": self.dispersion_observer.omega_dom[0, 0].item() / k_val,
                "f_slam_z": f_slam[0, 0].item(), "f_planing_z": (f_buoy[0, 0] + aoa_tensors["f_planing_z"][0, 0]).item(),
                "sat_str": sat_str, "torque_auth_pct": torque_scale[0, 0].item() * 100.0, "cell_temp_c": self.t_cell[0, 0].item() - 273.15,
                "v_sag": i_b * r0_v, "endurance_min": (rem_wh / p_w) * 60.0, "wh_per_km": p_w / max(0.2, self.vel[0, 0].item() * 3.6),
                "nn_raw": nn_raw, "sigma_l1_xyz": tuple(float(v) for v in sigma_l1[0].cpu().numpy()),
                "ba_hat_0": tuple(float(v) for v in self.ekf.ba_hat[0].cpu().numpy()),
                "a_neural_0": tuple(float(v) for v in a_neural[0].cpu().numpy()), "a_safe_0": tuple(float(v) for v in a_safe[0].cpu().numpy()),
                "e_int_hy": (self.e_int_height[0, 0].item(), self.e_int_y[0, 0].item()),
                "gimbal_fxy": (fx_gimbal_alloc[0, 0].item(), fy_gimbal_alloc[0, 0].item()),
                "ekf_err_0": (ekf_err_h[0, 0].item() * 100.0, ekf_err_vx[0, 0].item(), ekf_err_att_deg[0, 0].item()),
                "bernoulli_bias_cm_0": bern_bias[0, 0].item(), "sensor_locks": (bool(dvl_active[0, 0].item()), bool(flow_active[0, 0].item())),
                "targets_0": (self.h_target_filtered[0, 0].item(), self.v_target_filtered[0, 0].item()),
                "omega_deg_0": tuple(math.degrees(w) for w in self.omega[0].cpu().numpy()),
                "euler_deg": (phi_deg, theta_deg, psi_act_deg), "phi_btt_deg": math.degrees(phi_des[0, 0].item()),
                "mode_name": mode_names[self.current_mode[0, 0].item()],
                "margin_ceil_cm": (h_max_m[0, 0].item() - est_clearance[0, 0].item()) * 100.0,
                "margin_floor_cm": (est_clearance[0, 0].item() - h_min_m[0, 0].item()) * 100.0,
                "alpha_total_deg": math.degrees(a_f_v), "theta_shed_deg": math.degrees(theta_shed[0, 0].item()),
                "cavitation_sigma": cav_sig, "froude_fn": est_vel[0, 0].item() / math.sqrt(GRAVITY * HULL_LENGTH),
                "stall_margin_deg": math.degrees(math.radians(12.5) - abs(a_f_v)),
                "sigma_z_cm": math.sqrt(max(1e-6, self.ekf.P[0, 2, 2].item())) * 100.0,
                "bias_omega_deg": tuple(math.degrees(b) for b in self.ekf.bw_hat[0].cpu().numpy()),
                "barrier_name": b_name, "f_soar_surge": f_soar_v, "f_scull_thrust": f_scull_v,
                "f_par_z": aoa_tensors["f_par_z"][0, 0].item(), "strouhal_st": aoa_tensors["strouhal_st"][0, 0].item(),
                "morph_span": morph_dict["wing_span"].item(), "morph_ar": morph_dict["aspect_ratio"].item(),
                "morph_mass": morph_dict["total_mass"].item() * self.mass_scale,
                "morph_anhedral": math.degrees(morph_dict["anhedral_rad"].item())
            })

        if not torch.is_grad_enabled():
            self.detach_states()
        return self.metrics

        # =============================================================================
# 16. 3D INTERACTIVE OCEAN COCKPIT (WITH FASTFLIGHTCORE, REAL PHYSICS & OBSTACLES)
# =============================================================================
def project_3d_batch(pts_3d, cam_pos, cam_lookat, screen_w, screen_h, focal):
    fwd = cam_lookat - cam_pos
    fwd = fwd / (safe_norm(fwd) + 1e-6)
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
    coords_2d = torch.clamp(torch.nan_to_num(torch.cat([px, py], dim=-1), nan=-9999.0, posinf=9999.0, neginf=-9999.0), -3000, 5000)
    return coords_2d.cpu().numpy(), valid_mask.cpu().numpy(), depth.squeeze(-1).cpu().numpy()


def to_screen_pt(coord):
    try:
        x, y = float(coord[0]), float(coord[1])
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            return None
        return (int(round(x)), int(round(y)))
    except Exception:
        return None


def run_interactive_cockpit_visualizer(actor_net, ncm_net, morphology_genome=None,
                                       fixed_foil=False, disable_scull=False, gimbal_deg=15.0,
                                       mass_kg=None, harsh_sim=False, enable_obstacles=False,
                                       gnss_denied_window=None, fault_scenario=None, fault_time=4.0):
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

    pygame.display.set_caption("Aeromarine Autonomy & Real-Physics SIL Cockpit (200 Hz FastFlightCore)")
    clock = pygame.time.Clock()
    font_hud = pygame.font.SysFont("consolas", 12, bold=True)
    font_chart = pygame.font.SysFont("consolas", 10)

    visualizer_device = torch.device("cpu")
    env = WaveRidingAutonomyEnv(
        num_envs=1, dt=CONTROL_LOOP_DT, device=visualizer_device, use_embedded_qp=True,
        fixed_foil=fixed_foil, disable_scull=disable_scull, gimbal_max_deg=gimbal_deg,
        mass_kg=mass_kg, harsh_physics=harsh_sim, enable_obstacles=enable_obstacles,
        gnss_denied_window=gnss_denied_window, fault_scenario=fault_scenario, fault_time=fault_time
    )
    env.set_wave_curriculum(1.0)
    actor_net.to(visualizer_device).eval()
    ncm_net.to(visualizer_device).eval()
    if morphology_genome is not None:
        env.morphology.load_state_dict(morphology_genome.state_dict())
    env.morphology.to(visualizer_device).eval()
    env.reset(mode_mix=False)

    with torch.no_grad():
        m_cad = env.morphology()
    cad_w_span, cad_w_chord = m_cad["wing_span"].item(), m_cad["wing_chord"].item()
    cad_ar, cad_f_span, cad_strut = m_cad["aspect_ratio"].item(), m_cad["foil_span"].item(), m_cad["strut_length"].item()
    cad_mass, cad_anhed = m_cad["total_mass"].item() * env.mass_scale, math.degrees(m_cad["anhedral_rad"].item())

    fast_core = FastFlightCore(
        actor_net, ncm_net, m_cad, dt=CONTROL_LOOP_DT,
        fixed_foil=fixed_foil, disable_scull=disable_scull,
        gimbal_max_deg=gimbal_deg, mass_scale=env.mass_scale
    )
    fast_core.reset(env.pos[0].numpy(), env.vel[0].numpy(), env.rot[0].numpy(), mode=MODE_FOILING)

    cam_yaw, cam_pitch, cam_dist = 0.75, 0.32, 5.5
    running, frame_count = True, 0
    ema_sim_us, ema_core_us = 4200.0, 250.0

    history_len = 160
    hist_clearance = deque([0.185] * history_len, maxlen=history_len)
    hist_wave = deque([0.0] * history_len, maxlen=history_len)
    hist_lift = deque([14.0] * history_len, maxlen=history_len)
    hist_rotors = deque([6.8] * history_len, maxlen=history_len)

    gx, gy = np.linspace(-10.0, 10.0, 15), np.linspace(-8.0, 8.0, 13)
    dx_grid, dy_grid = float(gx[1] - gx[0]), float(gy[1] - gy[0])
    grid_offsets = torch.tensor([[x, y, 0.0] for x in gx for y in gy], device=visualizer_device, dtype=torch.float32)
    pts_tensor = grid_offsets.clone()

    d_arm = ARM_LENGTH / math.sqrt(2.0)
    body_points_local_base = torch.tensor([
        [0.0, 0.0, 0.0], [0.0, 0.0, -HULL_DRAFT_MAX], [0.0, 0.0, -cad_strut],
        [0.0, cad_w_span * 0.5, 0.0], [0.0, -cad_w_span * 0.5, 0.0],
        [0.0, -cad_f_span * 0.5, -cad_strut], [0.0, cad_f_span * 0.5, -cad_strut]
    ], device=visualizer_device, dtype=torch.float32)
    hubs_local = torch.tensor([[d_arm, -d_arm, 0.0], [d_arm, d_arm, 0.0], [-d_arm, d_arm, 0.0], [-d_arm, -d_arm, 0.0]], device=visualizer_device, dtype=torch.float32)
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
    btn_start_x = (SCREEN_W - (len(button_specs) * btn_w + (len(button_specs) - 1) * btn_gap)) // 2
    btn_y = SCREEN_H - 50
    button_rects = [pygame.Rect(btn_start_x + i * (btn_w + btn_gap), btn_y, btn_w, btn_h) for i in range(len(button_specs))]

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
    print(" COCKPIT ACTIVE (dt = 5 ms/frame | Stokes + 8-strip foil/5-strip strut + Kirchhoff added mass + FastFlightCore)")
    print(f"   CAD: Wing {cad_w_span:.2f}m x {cad_w_chord:.2f}m (AR {cad_ar:.1f}, Anhed {cad_anhed:+.1f}°) | Foil {cad_f_span:.2f}m (Strut {cad_strut:.2f}m) | Mass {cad_mass:.2f}kg")
    print(f"   SIL Config: FixedFoil={fixed_foil} | Gimbal={gimbal_deg:.0f}° | Harsh={harsh_sim} | Obstacles={enable_obstacles} | Fault={fault_scenario}")
    print("   Click HUD mode buttons or press [1]-[6] | [A] -> AUTONOMOUS SORTIE MISSION | [ESC] -> Quit")
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
                cam_dist = float(np.clip(cam_dist - event.y * 0.8, 2.5, 18.0))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for idx, rect in enumerate(button_rects):
                    if rect.collidepoint(event.pos):
                        trigger_mode_action(button_specs[idx][1], button_specs[idx][2])
                        pygame.mouse.get_rel()
                        break
            elif event.type == pygame.KEYDOWN:
                ch = event.unicode.lower() if event.unicode else ""
                kmap = {
                    pygame.K_1: 0, pygame.K_KP1: 0, pygame.K_2: 1, pygame.K_KP2: 1,
                    pygame.K_3: 2, pygame.K_KP3: 2, pygame.K_4: 3, pygame.K_KP4: 3,
                    pygame.K_5: 4, pygame.K_KP5: 4, pygame.K_6: 5, pygame.K_KP6: 5, pygame.K_a: 6
                }
                if event.key in kmap:
                    idx = kmap[event.key]
                    trigger_mode_action(button_specs[idx][1], button_specs[idx][2])
                elif ch in ("1", "2", "3", "4", "5", "6", "a"):
                    idx = 6 if ch == "a" else int(ch) - 1
                    trigger_mode_action(button_specs[idx][1], button_specs[idx][2])

        mouse, m_pos = pygame.mouse.get_pressed(), pygame.mouse.get_pos()
        if (mouse[0] or mouse[2]) and m_pos[1] < (btn_y - 6):
            dx, dy = pygame.mouse.get_rel()
            cam_yaw += dx * 0.008
            cam_pitch = float(np.clip(cam_pitch + dy * 0.008, 0.05, 1.45))
        else:
            pygame.mouse.get_rel()

        if env.auto_mission_active:
            env.auto_mission_timer += CONTROL_LOOP_DT
            tm = env.auto_mission_timer
            if tm < 3.0: env.set_mode(MODE_TAKEOFF)
            elif tm < 15.0: env.set_mode(MODE_FOILING)
            elif tm < 25.0: env.set_mode(MODE_SOARING)
            elif tm < 28.0: env.set_mode(MODE_EGRESS)
            elif tm < 38.0: env.set_mode(MODE_FREE_FLIGHT)
            elif tm < 42.0: env.set_mode(MODE_INGRESS)
            elif tm < 55.0: env.set_mode(MODE_FOILING)
            else:
                env.set_mode(MODE_LOITER)
                env.auto_mission_timer = 0.0

        t0 = time.perf_counter_ns()
        with torch.no_grad():
            m = env.step(actor_net, ncm_net, export_diagnostics=do_diagnostics)
        t1 = time.perf_counter_ns()

        t_c0 = time.perf_counter_ns()
        _ = fast_core.step(
            accel_meas=np.array([0.0, 0.0, GRAVITY], dtype=np.float64),
            gyro_meas=m["omega"][0].numpy(),
            run_fast_sensors=((env.step_count - 1) % env.sensor_period_steps == 0),
            mag_meas=np.array([0.42, 0.0, -0.90], dtype=np.float64),
            dvl_meas=m["vel"][0].numpy(), dvl_ok=True, flow_rates=np.zeros(2, dtype=np.float64),
            radar_raw=m["clearance_est"][0, 0].item(), radar_locked=True,
            d_sub_hydrostatic=m["immersion_depth"][0, 0].item(),
            eta_fwd_meas=m["eta_forward_meas"][0, 0].item(), eta_fwd_valid=True,
            run_gnss=(env.step_count % env.gnss_period_steps == 0),
            gnss_pos=m["pos"][0].numpy(), gnss_vel=m["vel"][0].numpy(),
            v_batt=m["v_batt"][0, 0].item(), mode_cmd=int(env.current_mode[0, 0].item()),
            y_ref=float(m["y_ref"][0, 0].item())
        )
        t_c1 = time.perf_counter_ns()

        if not do_diagnostics:
            ema_sim_us = 0.95 * ema_sim_us + 0.05 * ((t1 - t0) / 1000.0)
            ema_core_us = 0.95 * ema_core_us + 0.05 * ((t_c1 - t_c0) / 1000.0)

        sim_t = env.sim_time[0, 0].item()
        clearance_val, wave_val = m["clearance_est"][0, 0].item(), m["eta"][0, 0].item()
        lift_val, rotors_val, pusher_val = m["f_hydro_lift"][0, 0].item(), m["f_rotors"][0, 0].item(), m["f_pusher"][0, 0].item()
        cur_mode_idx = int(env.current_mode[0, 0].item())
        cur_mode_name = mode_names[cur_mode_idx]
        g_vals = m["rotor_tilt_deg"][0].cpu().numpy()

        hist_clearance.append(clearance_val); hist_wave.append(wave_val)
        hist_lift.append(lift_val); hist_rotors.append(rotors_val)

        if do_diagnostics:
            rt_factor = sim_t / max(1e-3, time.perf_counter() - wall_t0)
            cpu_budget_x = 5000.0 / max(1.0, ema_core_us)
            flow_str = "VENTILATED" if m["is_ventilated"] else "PRIME (Attached)"
            radar_status = "LOCKED" if m["radar_locked"] else ("SPRAY DROPOUT" if m["radar_dropout"] else "SEARCHING")
            multipath_str = "REJECTED (Masked)" if m["multipath_rejected"] else "CLEAN"
            dvl_on, flow_on = m["sensor_locks"]
            m_vals = m["motors"][0].cpu().numpy()
            m_disp = f"M0:{m_vals[0]:.2f}N | M1:{m_vals[1]:.2f}N | M2:{m_vals[2]:.2f}N | M3:{m_vals[3]:.2f}N | Push:{pusher_val:.2f}N"
            g_disp = f"G0:{g_vals[0]:.1f}° | G1:{g_vals[1]:.1f}° | G2:{g_vals[2]:.1f}° | G3:{g_vals[3]:.1f}°"
            gf_x, gf_y = m["gimbal_fxy"]
            d_surf, eul, om_d = m["hydro_deflections"], m["euler_deg"], m["omega_deg_0"]
            h_tgt_v, v_tgt_v = m["targets_0"]; eh_i, ey_i = m["e_int_hy"]
            ekf_dh, ekf_dvx, ekf_datt = m["ekf_err_0"]; ba, b_om = m["ba_hat_0"], m["bias_omega_deg"]
            ann, aqp, nraw, sl1 = m["a_neural_0"], m["a_safe_0"], m["nn_raw"], m["sigma_l1_xyz"]
            phase_str = f"SORTIE (T+{env.auto_mission_timer:.1f}s)" if env.auto_mission_active else "MANUAL PILOT"
            cav_str = f"{m['cavitation_sigma']:.2f} [SAFE]" if m["cavitation_sigma"] > SIGMA_CAV_CRIT else f"{m['cavitation_sigma']:.2f} [CAVITATING!]"

            print(
                f"\n[T={sim_t:05.2f}s | Mode: {m['mode_name']:<8} | Mission: {phase_str} | Status: {m['status_banner']} | FastCore: {ema_core_us:05.1f}µs ({cpu_budget_x:.1f}x CPU) | Sim: {ema_sim_us/1000:.2f}ms ({rt_factor:.2f}x RT)] " + "─" * 4 + "\n"
                f"  ├─ 1. CORRIDOR & TARGETS:    Vx: {m['vel'][0,0].item():.2f}m/s (Tgt:{v_tgt_v:.2f}) | Y-Dev: {m['pos'][0,1].item():+.2f}m (y_ref:{m['y_ref'][0,0].item():+.2f}, Vy:{m['vel'][0,1].item():+.2f}, IntY:{ey_i:+.2f}) | β:{m['beta_sideslip_deg']:+.1f}° | YawErr:{m['psi_err_deg']:+.1f}°\n"
                f"  ├─ 2. ATTITUDE & RATES:      φ: {eul[0]:+.1f}° (cmd {m['phi_btt_deg']:+.1f}°) | θ: {eul[1]:+.1f}° (shed {m['theta_shed_deg']:+.1f}°) | ψ: {eul[2]:+.1f}° (cmd {m['psi_des_deg']:+.1f}°) | ω=({om_d[0]:+.1f}, {om_d[1]:+.1f}, {om_d[2]:+.1f})°/s\n"
                f"  ├─ 3. ACTIVE HYDRO-SURFACES: Elevon L: {d_surf[0]:+.1f}° | Elevon R: {d_surf[1]:+.1f}° | Rudder: {d_surf[2]:+.1f}° | δ_coll: {m['delta_collective_deg']:+.1f}° | Foil α: {m['alpha_total_deg']:+.2f}° (Stall: {m['stall_margin_deg']:.1f}°) | Mom: {m['strut_moment_nm'][0,0].item():.2f}N·m\n"
                f"  ├─ 4. ES-EKF & SENSORS:      True h: {m['h_wing'][0,0].item():.3f}m | Est h: {clearance_val:.3f}m (Tgt:{h_tgt_v:.3f}, σ_z:±{m['sigma_z_cm']:.1f}cm) | Err(Δh:{ekf_dh:+.1f}cm, ΔV:{ekf_dvx:+.2f}, ΔAtt:{ekf_datt:.2f}°) | TapBias:{m['bernoulli_bias_cm_0']:+.1f}cm | DVL:{'LOCK' if dvl_on else 'OFF'} Flow:{'ON' if flow_on else 'OFF'}\n"
                f"  ├─ 5. RADAR & IMU BIASES:    Radar: {radar_status} | Multipath: {multipath_str} | Gyro Bias: ({b_om[0]:+.2f}, {b_om[1]:+.2f}, {b_om[2]:+.2f})°/s | Accel Bias: ({ba[0]:+.3f}, {ba[1]:+.3f}, {ba[2]:+.3f})m/s²\n"
                f"  ├─ 6. SEA DISPERSION:        Fwd Raycast: {m['eta_forward_meas'][0,0].item():+.3f}m | Nadir η̂: {m['eta_nadir_est'][0,0].item():+.3f}m (true {wave_val:+.3f}m) | λ: {m['wavelength_m']:.1f}m (cp: {m['celerity_mps']:.1f}m/s) | Prev(1.25s): {m['eta_preview'][0,0].item():+.3f}m\n"
                f"  ├─ 7. FLUID & AERO-HYDRO:    Immersion: {m['immersion_depth'][0,0].item()*100:.1f}cm [{flow_str}] | Foil Lift: {lift_val:04.1f}N | Wing: {m['f_wing_lift'][0,0].item():04.1f}N (PAR:{m['f_par_z']:.1f}N) | Hull: {m['f_planing_z']:04.1f}N (Fn:{m['froude_fn']:.2f}) | Rotors: {rotors_val:04.1f}N | Push: {pusher_val:04.1f}N\n"
                f"  ├─ 8. SCULLING & SOARING:    Caudal Thrust: {m['f_scull_thrust']:.2f}N (St: {m['strouhal_st']:.2f}) | Wave Surge: {m['f_soar_surge']:+.2f}N | Cavitation σ: {cav_str}\n"
                f"  ├─ 9. MOTOR & GIMBAL ALLOC:  [{m_disp}] | Tilts: [{g_disp}] | Gimbal F=({gf_x:+.2f}, {gf_y:+.2f})N | Auth: {m['torque_auth_pct']:.0f}% | Headroom: {m['headroom_min'][0,0].item():+.2f}N | [{m['sat_str']}]\n"
                f"  ├─ 10. BATTERY & ENDURANCE:  Power: {m['p_elec'][0,0].item():04.1f}W ({m['power_saved_pct'][0,0].item():+.1f}% SAVED) | SoC: {m['soc'][0,0].item():.1f}% | V_batt: {m['v_batt'][0,0].item():.2f}V (Sag:{m['v_sag']:.2f}V, {m['i_batt'][0,0].item():.1f}A) | Temp: {m['cell_temp_c']:.1f}°C | Endur: {m['endurance_min']:.1f}m ({m['wh_per_km']:.1f}Wh/km)\n"
                f"  └─ 11. CONTROL & SAFETY:     a_NN=({ann[0]:+.2f},{ann[1]:+.2f},{ann[2]:+.2f}) -> a_QP=({aqp[0]:+.2f},{aqp[1]:+.2f},{aqp[2]:+.2f}) | Raw=({nraw[0]:+.2f},{nraw[1]:+.2f},{nraw[2]:+.2f}) | σ_L1=({sl1[0]:+.2f},{sl1[1]:+.2f},{sl1[2]:+.2f}) | IntH:{eh_i:+.3f} | QP:{m['qp_override'][0,0].item():.3f} {m['barrier_name']} | Ceil:{m['margin_ceil_cm']:+.1f}cm Flr:{m['margin_floor_cm']:+.1f}cm"
            )

        robot_p_t = env.pos[0]
        cam_lookat = robot_p_t.clone()
        cam_pos = torch.tensor([
            float(cam_lookat[0] - cam_dist * math.cos(cam_pitch) * math.cos(cam_yaw)),
            float(cam_lookat[1] - cam_dist * math.cos(cam_pitch) * math.sin(cam_yaw)),
            max(0.5, float(cam_lookat[2].item() + cam_dist * math.sin(cam_pitch)))
        ], device=visualizer_device, dtype=torch.float32)

        screen.fill((8, 14, 22))
        snap_x = math.floor(float(robot_p_t[0].item()) / dx_grid) * dx_grid
        snap_y = math.floor(float(robot_p_t[1].item()) / dy_grid) * dy_grid
        pts_tensor.copy_(grid_offsets); pts_tensor[:, 0] += snap_x; pts_tensor[:, 1] += snap_y
        with torch.no_grad():
            eta_grid, _, _, _, _, _ = env.waves.evaluate_wave_kinematics(pts_tensor, env.sim_time.repeat(len(pts_tensor), 1), wave_scale=1.0)
        ocean_3d = pts_tensor.clone(); ocean_3d[:, 2:3] = eta_grid.view(-1, 1)
        grid_2d, grid_valid, _ = project_3d_batch(ocean_3d, cam_pos, cam_lookat, SCREEN_W, SCREEN_H, focal_len)

        nx, ny = len(gx), len(gy)
        for ix in range(nx - 1):
            for iy in range(ny - 1):
                idx = ix * ny + iy
                if grid_valid[idx] and grid_valid[(ix + 1) * ny + iy]:
                    p1, p2 = to_screen_pt(grid_2d[idx]), to_screen_pt(grid_2d[(ix + 1) * ny + iy])
                    if p1 and p2: pygame.draw.line(screen, (22, 58, 92), p1, p2, 1)
                if grid_valid[idx] and grid_valid[ix * ny + iy + 1]:
                    p1, p3 = to_screen_pt(grid_2d[idx]), to_screen_pt(grid_2d[ix * ny + iy + 1])
                    if p1 and p3: pygame.draw.line(screen, (22, 58, 92), p1, p3, 1)

        if enable_obstacles:
            obs_3d = torch.stack([env.obs_x[0], env.obs_y[0], torch.zeros_like(env.obs_x[0])], dim=-1)
            obs_2d, obs_val, obs_d = project_3d_batch(obs_3d, cam_pos, cam_lookat, SCREEN_W, SCREEN_H, focal_len)
            for oi in range(env.num_obs):
                if obs_val[oi]:
                    po = to_screen_pt(obs_2d[oi])
                    if po:
                        rad = max(5, int((env.obs_r[0, oi].item() * focal_len) / max(0.5, float(obs_d[oi]))))
                        pygame.draw.circle(screen, (255, 85, 60), po, rad, 2)

        rot_t = env.rot[0]
        body_3d_craft = robot_p_t.unsqueeze(0) + torch.matmul(body_points_local_base[:7, :], rot_t.T)
        preview_world = torch.tensor([robot_p_t[0].item() + env.vel[0, 0].item() * 1.25, robot_p_t[1].item(), m["eta_preview"][0, 0].item()], device=visualizer_device, dtype=torch.float32).unsqueeze(0)
        craft_2d, craft_valid, _ = project_3d_batch(torch.cat([body_3d_craft, preview_world], dim=0), cam_pos, cam_lookat, SCREEN_W, SCREEN_H, focal_len)

        pts_c = [to_screen_pt(craft_2d[i]) for i in range(8)]
        if craft_valid[3] and craft_valid[4] and pts_c[3] and pts_c[4]: pygame.draw.line(screen, (240, 245, 255), pts_c[3], pts_c[4], 4)
        if craft_valid[0] and craft_valid[1] and pts_c[0] and pts_c[1]: pygame.draw.line(screen, (180, 220, 255), pts_c[0], pts_c[1], 6)
        if craft_valid[1] and craft_valid[2] and pts_c[1] and pts_c[2]: pygame.draw.line(screen, (255, 60, 60), pts_c[1], pts_c[2], 3)
        if craft_valid[2] and pts_c[2]:
            fc = (0, 255, 200) if lift_val > 5.0 else (255, 180, 40)
            pygame.draw.circle(screen, fc, pts_c[2], 6)
            if craft_valid[5] and craft_valid[6] and pts_c[5] and pts_c[6]: pygame.draw.line(screen, fc, pts_c[5], pts_c[6], 3)
        if craft_valid[0] and craft_valid[7] and pts_c[0] and pts_c[7]:
            pygame.draw.line(screen, (0, 255, 255), pts_c[0], pts_c[7], 1)
            pygame.draw.circle(screen, (0, 255, 255), pts_c[7], 5, 1)

        tilts_now = m["rotor_tilts"][0]
        ux_v, uy_v = tilts_now[:, 0], tilts_now[:, 1]
        uz_v = torch.sqrt((1.0 - ux_v**2 - uy_v**2).clamp(min=ROTOR_MAX_TILT_COS**2))
        hubs_3d = robot_p_t.unsqueeze(0) + torch.matmul(hubs_local, rot_t.T)
        tips_3d = hubs_3d + torch.matmul(torch.stack([ux_v, uy_v, uz_v], dim=-1) * 0.22, rot_t.T)
        hubs_2d, hubs_val, hubs_d = project_3d_batch(hubs_3d, cam_pos, cam_lookat, SCREEN_W, SCREEN_H, focal_len)
        tips_2d, tips_val, _ = project_3d_batch(tips_3d, cam_pos, cam_lookat, SCREEN_W, SCREEN_H, focal_len)
        for i in range(4):
            if hubs_val[i]:
                ph = to_screen_pt(hubs_2d[i])
                if ph:
                    pygame.draw.circle(screen, (255, 210, 40), ph, max(3, int(22.0 / max(0.1, float(hubs_d[i])))), 1)
                    if tips_val[i]:
                        ptip = to_screen_pt(tips_2d[i])
                        if ptip: pygame.draw.line(screen, (0, 255, 160), ph, ptip, 2)

        hud = pygame.Rect(15, 15, 835, 278)
        pygame.draw.rect(screen, (10, 16, 26), hud, border_radius=6)
        pygame.draw.rect(screen, (0, 220, 255), hud, 1, border_radius=6)
        m_vals = m["motors"][0].cpu().numpy()
        d_surf = m["hydro_deflections"]
        mission_tag = "AUTO SORTIE" if env.auto_mission_active else "MANUAL"
        sil_tag = f" | FixedFoil={fixed_foil} | Harsh={harsh_sim} | Obs={enable_obstacles} | Fault={fault_scenario or 'None'}"
        h_tgt_hud, v_tgt_hud = env.h_target_filtered[0, 0].item(), env.v_target_filtered[0, 0].item()

        stats = [
            (f"AERO-HYDRO AUTONOMY | MODE: {cur_mode_name} [{mission_tag}]{sil_tag}", (0, 240, 255)),
            (f"FAST FLIGHT CORE (NumPy 200Hz): {ema_core_us:.1f} µs (<5000 µs OK) | PyTorch Sim: {ema_sim_us/1000:.2f} ms", (0, 255, 120) if ema_core_us < 5000.0 else (255, 90, 90)),
            (f"HYDRO-SURFACES: Elev L:{d_surf[0]:+.1f}° R:{d_surf[1]:+.1f}° Rud:{d_surf[2]:+.1f}° δ_coll:{m['delta_collective_deg']:+.1f}° | StrutMom:{m['strut_moment_nm'][0,0].item():.2f}/{STRUT_YIELD_LIMIT_NM:.0f}N·m", (255, 255, 100)),
            (f"WATER LIFT: {lift_val:.1f} N | Cav σ: {m['sigma_cav_min'][0,0].item():.2f} | Power: {m['p_elec'][0,0].item():.1f} W ({m['power_saved_pct'][0,0].item():+.1f}% SAVED)", (0, 255, 200)),
            (f"OMNI GIMBAL TILTS (0-{gimbal_deg:.0f}°): [G0:{g_vals[0]:.1f}° | G1:{g_vals[1]:.1f}° | G2:{g_vals[2]:.1f}° | G3:{g_vals[3]:.1f}°]", (120, 255, 180)),
            (f"NAVIGATION EKF: Clr:{clearance_val:.3f}m (Tgt {h_tgt_hud:.2f}m) | Vx:{m['vel'][0,0].item():.2f}m/s (Tgt {v_tgt_hud:.2f}) | Y:{m['pos'][0,1].item():+.2f}m (y_ref:{m['y_ref'][0,0].item():+.2f})", (100, 255, 100)),
            (f"PROPULSION: [M0:{m_vals[0]:.1f}N | M1:{m_vals[1]:.1f}N | M2:{m_vals[2]:.1f}N | M3:{m_vals[3]:.1f}N | Push:{pusher_val:.1f}N]", (255, 210, 40)),
            (f"BATTERY (6S): SoC:{m['soc'][0,0].item():.1f}% | {m['v_batt'][0,0].item():.2f}V | {m['i_batt'][0,0].item():.1f}A | Temp:{m['cell_temp_c']:.1f}°C", (200, 220, 255)),
        ]
        for i, (txt, col) in enumerate(stats):
            screen.blit(font_hud.render(txt, True, col), (hud.x + 14, hud.y + 12 + i * 26))

        chart_box = pygame.Rect(SCREEN_W - 380, 15, 365, 278)
        pygame.draw.rect(screen, (10, 16, 26), chart_box, border_radius=6)
        pygame.draw.rect(screen, (0, 180, 220), chart_box, 1, border_radius=6)
        screen.blit(font_chart.render("LIVE TELEMETRY SCOPE", True, (0, 240, 255)), (chart_box.x + 10, chart_box.y + 8))

        c1_rect = pygame.Rect(chart_box.x + 10, chart_box.y + 28, 345, 102)
        pygame.draw.rect(screen, (14, 22, 34), c1_rect)
        pygame.draw.line(screen, (60, 80, 100), (c1_rect.x, c1_rect.centery), (c1_rect.right, c1_rect.centery), 1)
        pts_clr, pts_wav = [], []
        for idx in range(history_len):
            xp = c1_rect.x + (idx / (history_len - 1)) * c1_rect.width
            pts_clr.append((xp, np.clip(c1_rect.bottom - (hist_clearance[idx] / 0.40) * c1_rect.height, c1_rect.top, c1_rect.bottom)))
            pts_wav.append((xp, np.clip(c1_rect.centery - (hist_wave[idx] / 0.20) * (c1_rect.height * 0.4), c1_rect.top, c1_rect.bottom)))
        pygame.draw.lines(screen, (0, 120, 255), False, pts_wav, 1)
        pygame.draw.lines(screen, (0, 255, 120), False, pts_clr, 2)
        screen.blit(font_chart.render("Clearance h (Green) vs Wave eta (Blue)", True, (200, 220, 240)), (c1_rect.x + 6, c1_rect.y + 4))

        c2_rect = pygame.Rect(chart_box.x + 10, chart_box.y + 148, 345, 102)
        pygame.draw.rect(screen, (14, 22, 34), c2_rect)
        pts_lft, pts_rot = [], []
        for idx in range(history_len):
            xp = c2_rect.x + (idx / (history_len - 1)) * c2_rect.width
            pts_lft.append((xp, np.clip(c2_rect.bottom - (hist_lift[idx] / 32.0) * c2_rect.height, c2_rect.top, c2_rect.bottom)))
            pts_rot.append((xp, np.clip(c2_rect.bottom - (hist_rotors[idx] / 32.0) * c2_rect.height, c2_rect.top, c2_rect.bottom)))
        pygame.draw.lines(screen, (0, 240, 220), False, pts_lft, 2)
        pygame.draw.lines(screen, (255, 180, 40), False, pts_rot, 1)
        screen.blit(font_chart.render("Water Lift (Cyan) vs Rotors (Orange) [0-32 N]", True, (200, 220, 240)), (c2_rect.x + 6, c2_rect.y + 4))

        for idx, rect in enumerate(button_rects):
            label, mode_val, _ = button_specs[idx]
            is_act = (env.auto_mission_active if mode_val is None else (not env.auto_mission_active and cur_mode_idx == mode_val))
            pygame.draw.rect(screen, (0, 110, 140) if is_act else (14, 22, 34), rect, border_radius=5)
            pygame.draw.rect(screen, (0, 255, 180) if is_act else (60, 110, 140), rect, 2 if is_act else 1, border_radius=5)
            ts = font_hud.render(label, True, (255, 255, 255) if is_act else (170, 200, 220))
            screen.blit(ts, (rect.centerx - ts.get_width() // 2, rect.centery - ts.get_height() // 2))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


# =============================================================================
# 17. MASTER SCRIPT ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Aeromarine Hydrofoil Autonomy & Real-Physics SIL Suite")
    parser.add_argument("--epochs", type=int, default=96)
    parser.add_argument("--envs", type=int, default=2048, help="parallel training envs")
    parser.add_argument("--steps", type=int, default=200, help="rollout steps per epoch")
    parser.add_argument("--tbptt", type=int, default=10, help="truncated BPTT window length")
    parser.add_argument("--ckpt", type=str, default="aeromarine_ckpt.pt", help="checkpoint path to save/load")
    parser.add_argument("--train", action="store_true", help="force training and overwrite --ckpt")
    parser.add_argument("--skip-train", action="store_true", help="load --ckpt without retraining")
    parser.add_argument("--no-viz", action="store_true", help="skip the pygame cockpit")
    parser.add_argument("--trim", action="store_true", help="print static foil trim report and exit")
    parser.add_argument("--eval-suite", action="store_true", help="run 7-scenario Monte Carlo certification + FastFlightCore benchmark")
    parser.add_argument("--eval-envs", type=int, default=256)
    parser.add_argument("--eval-steps", type=int, default=350)
    parser.add_argument("--benchmark-core", action="store_true", help="run pure-NumPy 200 Hz FastFlightCore parity & timing benchmark")
    parser.add_argument("--fixed-foil", action="store_true", help="lock underwater elevons/rudder/sculling to 0 (passive carbon T-foil)")
    parser.add_argument("--disable-scull", action="store_true", help="disable 22 Hz caudal fin sculling")
    parser.add_argument("--gimbal-deg", type=float, default=15.0, help="rotor hub gimbal limit in degrees (0-15)")
    parser.add_argument("--mass-kg", type=float, default=None, help="override airframe mass in kg (e.g. 3.2)")
    parser.add_argument("--harsh-sim", action="store_true", help="enable 350N slam, Wagner lag, cavitation collapse & burst blackouts")
    parser.add_argument("--obstacles", action="store_true", help="enable synthetic subsurface hazard corridor")
    parser.add_argument("--fault-test", type=str, default=None, choices=["motor_loss", "elevon_jam", "gimbal_jam", "radar_blind"])
    parser.add_argument("--fault-time", type=float, default=4.0)
    parser.add_argument("--gnss-denied", type=float, nargs=2, default=None, metavar=("T_START", "T_END"))
    parser.add_argument("--decalage", type=float, nargs=2, default=None, metavar=("MIN_DEG", "MAX_DEG"))
    args = parser.parse_args()

    if args.decalage is not None:
        FOIL_DECALAGE_DEG = (min(args.decalage), max(args.decalage))

    if args.trim:
        static_trim_report(mass_kg=args.mass_kg)
        sys.exit(0)

    telemetry_log_filename = f"aeromarine_flight_telemetry_{int(time.time())}.txt"
    sys.stdout = DualLogger(telemetry_log_filename)
    atexit.register(sys.stdout.close)
    print(f">> [SYS-INIT] Live Synchronous Disk Logging Active: '{telemetry_log_filename}'\n")

    do_train = args.train or (not args.skip_train and not args.eval_suite and not args.benchmark_core)
    try:
        if do_train:
            trained_pilot, trained_ncm, evolved_morphology = train_wave_riding_pilot(
                epochs=args.epochs, num_envs=args.envs, rollout_steps=args.steps, tbptt_steps=args.tbptt,
                fixed_foil=args.fixed_foil, disable_scull=args.disable_scull, gimbal_deg=args.gimbal_deg,
                mass_kg=args.mass_kg, harsh_sim=args.harsh_sim, enable_obstacles=args.obstacles, save_ckpt=args.ckpt
            )
        else:
            trained_pilot, trained_ncm, evolved_morphology = load_or_train_checkpoint(
                args.ckpt, epochs=args.epochs, num_envs=args.envs, rollout_steps=args.steps, tbptt_steps=args.tbptt,
                fixed_foil=args.fixed_foil, disable_scull=args.disable_scull, gimbal_deg=args.gimbal_deg,
                mass_kg=args.mass_kg, harsh_sim=args.harsh_sim, enable_obstacles=args.obstacles
            )
    except torch.cuda.OutOfMemoryError:
        print(">> [ERROR] CUDA out of memory. Reduce --envs, --steps, or --tbptt (e.g. --envs 512 --tbptt 5)")
        sys.exit(1)

    if args.benchmark_core and not args.eval_suite:
        benchmark_fast_flight_core(trained_pilot, trained_ncm, evolved_morphology)

    if args.eval_suite:
        run_monte_carlo_suite(trained_pilot, trained_ncm, evolved_morphology, eval_envs=args.eval_envs, steps=args.eval_steps)

    export_flight_software_onnx(trained_pilot, "wave_pilot_hardware_fp32.onnx")
    if not args.no_viz:
        run_interactive_cockpit_visualizer(
            trained_pilot, trained_ncm, evolved_morphology,
            fixed_foil=args.fixed_foil, disable_scull=args.disable_scull, gimbal_deg=args.gimbal_deg,
            mass_kg=args.mass_kg, harsh_sim=args.harsh_sim, enable_obstacles=args.obstacles,
            gnss_denied_window=tuple(args.gnss_denied) if args.gnss_denied else None,
            fault_scenario=args.fault_test, fault_time=args.fault_time
        )