"""
=============================================================================
UNIFIED MULTIMODAL HYDROFOIL AUTONOMY ENGINE (PURE 16-ACTUATOR NN FLIGHT CTRL)
=============================================================================
RESEARCH SIMULATION STACK (not flight-qualified):
  - 32-wave directional JONSWAP/PM sea with deep-water orbital kinematics
  - 15-state error-state EKF (IMU strapdown + mag + DVL + optical flow + GNSS)
  - FMCW radar / forward raycast / piezoresistive pressure-tap sensor models
  - Active T-foil elevons + strut rudder, WIG wing, planing hull, PAR cushion
  - 4 independently 2-axis tilting lift rotors (0-15 deg omni-directional
    hub gimbal, Pitt-Peters inflow) + axial cruise pusher prop
  - 6S electro-thermal battery powering lift rotors, pusher, and sculling fin
  - Pure end-to-end 26-state -> 16-actuator Neural Pilot (4 motor thrusts,
    4x2 hub gimbal vectors, pusher thrust, 3 hydro-surface deflections)
  - Minimal vertical CBF safety shield + 5D Neural Contraction Metric (NCM)
  - Morphogenetic airframe co-evolution genome

CHANGES IN THIS BUILD (on top of FIX-01..FIX-40, PROP-1..PROP-2 & GIMBAL-1):
  [NN-CTRL] FullNeuralActuatorPilot (26 -> 16) directly drives all 16 physical
            actuators in step() and ONNX export. No hand-coded inner attitude
            PID, no BTT/pitch guidance law, no L1 compensator, and no runtime
            wrench mixer in the control loop. Stage 0 teacher BC initializes
            stabilized hover/foiling trim, then Stages 1-3 optimize all 16
            actuators end-to-end via truncated BPTT.
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

# Neural Pilot I/O Dimensions
NN_STATE_DIM = 26
NN_ACTION_DIM = 16

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

# Reference step used to define dt-scaled filter/delay constants
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

    def raycast_forward_surface(self, pos, lookahead_dist, t, wave_scale=1.0):
        """Forward lidar/radar elevation sample. `pos` must be the TRUE vehicle position (sensor sim)."""
        px_fwd = (pos[:, 0:1] + lookahead_dist).unsqueeze(1)
        py_fwd = pos[:, 1:2].unsqueeze(1)
        t_term = t.unsqueeze(1) if (torch.is_tensor(t) and t.ndim == 2) else t
        amps = self.amplitudes_base * wave_scale
        phase_angle = (self.kx * px_fwd + self.ky * py_fwd - self.omega * t_term + self.phases)
        eta_ideal = torch.sum(amps * torch.cos(phase_angle), dim=1)

        range_noise = torch.randn_like(eta_ideal) * 0.025
        valid = (torch.rand_like(eta_ideal) >= 0.03)
        eta_forward_meas = torch.where(valid, eta_ideal + range_noise, torch.zeros_like(eta_ideal))
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

        self.P = torch.eye(15, device=device).unsqueeze(0).repeat(num_envs, 1, 1) * 0.05
        self.P[:, 6:9, 6:9] *= 0.01

        self.q_var_accel = 0.08
        self.q_var_gyro = 0.015
        self.q_var_ba = 0.0005
        self.q_var_bw = 0.0001
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

        self.P = torch.eye(15, device=self.device).unsqueeze(0).repeat(num_envs, 1, 1) * 0.05
        self.P[:, 6:9, 6:9] *= 0.01

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
        """Returns (raw_detected_clearance, spray_dropout, is_target_locked, multipath_rejected)."""
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
        detected_range = (peak_idx.float() + offset) * self.radar_dr
        detected_clearance = detected_range + crest_bias

        peak_power = y_c
        cfar_noise_floor = torch.mean(p_noise, dim=-1, keepdim=True)
        is_target_locked = (peak_power > (cfar_noise_floor * self.cfar_threshold_factor)) & (spray_dropout < 0.5)

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

    def reset(self, num_envs):
        self.k_dom = torch.full((num_envs, 1), 0.45, device=self.device)
        self.omega_dom = torch.sqrt(GRAVITY * self.k_dom)
        self.q_quad = torch.zeros((num_envs, 1), device=self.device)
        self.last_eta_nadir = torch.zeros((num_envs, 1), device=self.device)

    def update_and_predict(self, clearance_est, craft_z, eta_forward_meas, valid, lookahead_dist, fwd_speed, preview_time=1.25):
        clearance_est = clearance_est.detach()
        craft_z = craft_z.detach()
        eta_forward_meas = eta_forward_meas.detach()

        eta_nadir_est = (craft_z - clearance_est).clamp(-0.6, 0.6)
        self.last_eta_nadir = eta_nadir_est
        delta_x = torch.clamp(lookahead_dist.detach(), min=0.30)
        spatial_slope = ((eta_forward_meas - eta_nadir_est) / delta_x).clamp(-0.6, 0.6)

        a_k = math.exp(-self.dt / 0.117)
        k_est = torch.abs(spatial_slope) / (torch.abs(eta_nadir_est) + 0.08)
        k_new = torch.clamp(a_k * self.k_dom + (1.0 - a_k) * k_est, min=0.08, max=3.5)
        self.k_dom = torch.where(valid, k_new, self.k_dom)
        self.omega_dom = torch.sqrt(GRAVITY * self.k_dom)

        q_meas = (-spatial_slope / self.k_dom).clamp(-0.4, 0.4)
        a_q = math.exp(-self.dt / 0.15)
        q_new = a_q * self.q_quad + (1.0 - a_q) * q_meas
        self.q_quad = torch.where(valid, q_new, self.q_quad)

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
                morphology_dict, sim_time, is_soaring):
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
        z_tip_L = pos[:, 2:3] + rot[:, 2, 1:2] * y_station_offset - strut_len
        z_tip_R = pos[:, 2:3] - rot[:, 2, 1:2] * y_station_offset - strut_len

        d_sub_L = (eta - z_tip_L).clamp(min=0.0, max=0.22)
        d_sub_R = (eta - z_tip_R).clamp(min=0.0, max=0.22)
        immersion_depth_mean = 0.5 * (d_sub_L + d_sub_R)

        delta_e_L = hydro_surfaces[:, 0:1]
        delta_e_R = hydro_surfaces[:, 1:2]
        delta_rud = hydro_surfaces[:, 2:3]

        scull_active = is_soaring.float() * (immersion_depth_mean > 0.020).float()
        scull_freq = SCULL_FREQ_NOM
        scull_amp = SCULL_AMP_NOM
        delta_rud_eff = delta_rud

        linear_tail_amp = SCULL_LEVER_ARM * math.sin(scull_amp)
        strouhal_st = (2.0 * scull_freq * linear_tail_amp / v_foil_speed).clamp(max=2.0)
        strouhal_prop_eff = torch.exp(-((strouhal_st - 0.30)**2) / (2.0 * (0.08**2))) * 0.84

        v_scull_tip = 2.0 * math.pi * scull_freq * scull_amp * SCULL_LEVER_ARM
        v_scull_effective = torch.sqrt(v_foil_speed**2 + (v_scull_tip**2) * scull_active)
        q_scull = 0.5 * rho_water * (v_scull_effective**2)
        f_scull_thrust = torch.clamp(
            q_scull * caudal_area * HYDRO_RUDDER_CY_DELTA * math.sin(scull_amp) * strouhal_prop_eff * scull_active,
            min=0.0, max=22.0
        )
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
            "strouhal_st": strouhal_st
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
        c_T = (motors / (2.0 * RHO_AIR * ROTOR_DISK_AREA)).unsqueeze(-1)
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

        p_coils = torch.sum((motor_currents**2) * MOTOR_COIL_RESISTANCE, dim=-1, keepdim=True)
        p_batt_heat = (total_current**2) * r0_temp + total_current * (u1_next + u2_next)
        p_electrical_total = torch.sum(p_induced, dim=-1, keepdim=True) + p_coils + p_batt_heat + p_scull_load + p_pusher_elec

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
        z = torch.zeros((N, M_c, 3), device=u.device, dtype=u.dtype)
        p = torch.zeros((N, M_c, 1), device=u.device, dtype=u.dtype)

        A_norm_sq = torch.sum(A * A, dim=-1, keepdim=True).clamp(min=1e-6)
        rho_inv = 1.0 / self.rho

        for _ in range(self.max_iters):
            for i in range(M_c):
                a_i = A[:, i:i+1, :]
                b_i = b[:, i:i+1, :]
                z_i = z[:, i:i+1, :]

                u_p = u.unsqueeze(1) - z_i
                dot_val = torch.sum(a_i * u_p, dim=-1, keepdim=True)
                viol = dot_val - b_i

                step = F.softplus(viol, beta=self.beta) / (A_norm_sq[:, i:i+1, :] + rho_inv)
                delta_u = step * a_i
                s_i_new = step * rho_inv

                u_new = (u_p - delta_u).squeeze(1)
                z = z.clone()
                z[:, i:i+1, :] = (u_new.unsqueeze(1) - u_p)
                p = p.clone()
                p[:, i:i+1, :] = F.relu(s_i_new)
                u = u_new

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
        viol = torch.bmm(A, u_opt.unsqueeze(-1)) - b
        active_idx = torch.argmax(viol.squeeze(-1), dim=-1, keepdim=True)
        return u_opt, override, active_idx


# =============================================================================
# 9. FAST L1 ADAPTIVE DISTURBANCE OBSERVER (DIAGNOSTIC / REFERENCE ONLY)
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

    def step(self, accel_measured, accel_commanded, dt):
        a = math.exp(-dt / self.tau)
        d_hat = torch.clamp(accel_measured - accel_commanded + self.sigma_filtered, -15.0, 15.0)
        self.sigma_hat = d_hat
        self.sigma_filtered = torch.clamp(a * self.sigma_filtered + (1.0 - a) * d_hat, -self.sigma_limit, self.sigma_limit)
        return self.sigma_filtered


# =============================================================================
# 10. PURE 16-ACTUATOR NEURAL FLIGHT CONTROLLER & 5D CONTRACTION METRIC (NCM)
# =============================================================================
class FullNeuralActuatorPilot(nn.Module):
    """[NN-CTRL] Pure end-to-end neural flight control system.
    Reads 26 EKF/sensor states and directly outputs all 16 physical actuators:
      [0:4]   4 lift rotor thrust magnitudes  in [MOTOR_IDLE_FLOOR, thrust_ceiling]
      [4:12]  4x2 omni hub gimbal vectors     in 0-15 deg circular cone
      [12:13] axial cruise pusher thrust      in [0, MAX_PUSHER_THRUST]
      [13:16] 3 hydro-surface deflections     in [-15 deg, +15 deg]
    No hardcoded PD backbone or mixer inside the network."""
    def __init__(self, state_dim=NN_STATE_DIM, action_dim=NN_ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 192),
            nn.SiLU(),
            nn.Linear(192, 192),
            nn.SiLU(),
            nn.Linear(192, 128),
            nn.SiLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs):
        raw = self.net(obs)
        v_batt_ratio = obs[:, 25:26].clamp(min=0.75, max=1.05)
        thrust_ceiling = MAX_THRUST_PER_MOTOR * (v_batt_ratio**2)

        motors = MOTOR_IDLE_FLOOR + (thrust_ceiling - MOTOR_IDLE_FLOOR) * torch.sigmoid(raw[:, 0:4])

        # Smooth isotropic radial projection into circular 15-deg gimbal cone per rotor
        z_tilts = raw[:, 4:12].view(-1, 4, 2)
        z_norm_sq = torch.sum(z_tilts * z_tilts, dim=-1, keepdim=True)
        tilts = (ROTOR_MAX_TILT_SIN * z_tilts / torch.sqrt(1.0 + z_norm_sq)).view(-1, 8)

        pusher = MAX_PUSHER_THRUST * torch.sigmoid(raw[:, 12:13])
        surfaces = MAX_HYDRO_DEFLECTION * torch.tanh(raw[:, 13:16])

        return torch.cat([motors, tilts, pusher, surfaces], dim=-1)


class NeuralContractionMetric(nn.Module):
    """Block-diagonal W between horizontal (x,y) and vertical (z) for 5D NCM certification."""
    def __init__(self, state_dim=6, alpha=0.10):
        super().__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(state_dim, 48),
            nn.SiLU(),
            nn.Linear(48, 48),
            nn.SiLU(),
            nn.Linear(48, 6),
        )

    def forward(self, state_error):
        device_str = "cuda" if state_error.is_cuda else "cpu"
        with get_amp_autocast(device_str, enabled=False):
            state_error_f = state_error.float()
            params = self.net(state_error_f)

            l00 = F.softplus(params[:, 0:1]) + 0.35
            l10 = params[:, 1:2]
            l11 = F.softplus(params[:, 2:3]) + 0.35
            l22 = F.softplus(params[:, 5:6]) + 0.35
            zeros = torch.zeros_like(l00)

            L_row0 = torch.cat([l00, zeros, zeros], dim=-1)
            L_row1 = torch.cat([l10, l11, zeros], dim=-1)
            L_row2 = torch.cat([zeros, zeros, l22], dim=-1)
            L = torch.stack([L_row0, L_row1, L_row2], dim=1)

            eye3 = torch.eye(3, device=state_error.device, dtype=L.dtype).unsqueeze(0)
            W = torch.matmul(L, L.transpose(-1, -2)) + 0.25 * eye3

            M_top = torch.cat([2.5 * W, 0.7 * W], dim=-1)
            M_bot = torch.cat([0.7 * W, 1.0 * W], dim=-1)
            return torch.cat([M_top, M_bot], dim=-2)


# =============================================================================
# 11. 6-DoF ROBOT DYNAMICS & STAGE-0 TEACHER ALLOCATOR
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
        """Used ONLY during Stage 0 teacher pre-training to generate supervision targets for the NN."""
        if not torch.is_tensor(v_batt):
            v_batt = torch.tensor(v_batt, device=Fz_cmd.device)
        thrust_ceiling = MAX_THRUST_PER_MOTOR * (v_batt / BATTERY_V0)**2

        q_water = 0.5 * RHO_WATER_NOM * (airspeed.clamp(min=0.5)**2)
        sub_frac = (d_sub / 0.040).clamp(min=0.0, max=1.0)

        f_span = morphology_dict["foil_span"]
        f_area = morphology_dict["foil_area"]
        caudal_area = morphology_dict["caudal_area"]

        max_hydro_roll_torque = 1.00 * sub_frac
        max_hydro_yaw_torque = 0.50 * sub_frac

        hydro_roll_gain = 2.0 * (f_span * 0.25) * q_water * (f_area * 0.5) * HYDRO_CL_DELTA * sub_frac
        hydro_yaw_gain = q_water * caudal_area * HYDRO_RUDDER_CY_DELTA * sub_frac * 0.08

        tx_des = torch.clamp(torques_desired[:, 0:1], -3.5, 3.5)
        ty_des = torch.clamp(torques_desired[:, 1:2], -3.5, 3.5)
        tz_des = torch.clamp(torques_desired[:, 2:3], -1.2, 1.2)

        tau_hydro_roll_alloc = torch.clamp(tx_des, -max_hydro_roll_torque, max_hydro_roll_torque)
        tau_hydro_yaw_alloc = torch.clamp(tz_des, -max_hydro_yaw_torque, max_hydro_yaw_torque)

        delta_roll = torch.clamp(tau_hydro_roll_alloc / (hydro_roll_gain + 0.15), -MAX_HYDRO_DEFLECTION, MAX_HYDRO_DEFLECTION)
        delta_rud = -torch.clamp(tau_hydro_yaw_alloc / (hydro_yaw_gain + 0.10), -MAX_HYDRO_DEFLECTION, MAX_HYDRO_DEFLECTION)

        if delta_collective is None:
            delta_collective = torch.zeros_like(delta_roll)
        delta_e_L = torch.clamp(delta_roll + delta_collective, -MAX_HYDRO_DEFLECTION, MAX_HYDRO_DEFLECTION)
        delta_e_R = torch.clamp(-delta_roll + delta_collective, -MAX_HYDRO_DEFLECTION, MAX_HYDRO_DEFLECTION)

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
# 12. SIM-TO-REAL MULTI-ENVIRONMENT (PURE 16-ACTUATOR NN CONTROL)
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

        self.gnss_period_steps = max(1, int(round(0.10 / dt)))
        self.step_count = 0

        self.current_mode = torch.full((num_envs, 1), MODE_FOILING, dtype=torch.long, device=device)
        self.target_speed = 5.20
        self.wave_curriculum_scale = 1.0

        self.h_target_filtered = torch.full((num_envs, 1), RIDE_HEIGHT_OPTIMAL, device=device)
        self.v_target_filtered = torch.full((num_envs, 1), self.target_speed, device=device)
        self.e_int_height = torch.zeros((num_envs, 1), device=device)

        self.hydro_surfaces = torch.zeros((num_envs, 3), device=device)
        self.hydro_surfaces_actual = torch.zeros((num_envs, 3), device=device)
        self.rotor_tilts = torch.zeros((num_envs, 4, 2), device=device)
        self.auto_mission_active = False
        self.auto_mission_timer = 0.0

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
        self.e_int_height = self.e_int_height.detach()
        self.hydro_surfaces_actual = self.hydro_surfaces_actual.detach()
        self.hydro_surfaces = self.hydro_surfaces.detach()
        self.act_motors_prev = self.act_motors_prev.detach()
        self.act_tilts_prev = self.act_tilts_prev.detach()
        self.act_surf_prev = self.act_surf_prev.detach()
        self.is_vent_L = self.is_vent_L.detach()
        self.is_vent_R = self.is_vent_R.detach()
        self.clearance_hold = self.clearance_hold.detach()
        self.follower_accel_prev = self.follower_accel_prev.detach()
        self.ekf.detach_state()
        self.l1_observer.detach_state()

    def reset(self):
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
        self.clearance_hold = torch.full((N, 1), RIDE_HEIGHT_OPTIMAL, device=self.device)

        init_x = torch.rand((N, 1), device=self.device) * 40.0
        init_y = torch.randn((N, 1), device=self.device) * (0.25 if N > 1 else 0.05)

        eta0, _, _, _, _, _ = self.waves.evaluate_wave_kinematics(
            torch.cat([init_x, init_y, torch.zeros_like(init_x)], dim=-1),
            self.sim_time,
            wave_scale=self.wave_curriculum_scale
        )

        init_clearance = torch.full((N, 1), RIDE_HEIGHT_OPTIMAL, device=self.device)
        vx0 = torch.full((N, 1), self.target_speed, device=self.device)
        vy0 = torch.zeros((N, 1), device=self.device)
        vz0 = torch.zeros((N, 1), device=self.device)
        phi0 = torch.zeros((N, 1), device=self.device)
        theta0 = torch.full((N, 1), math.radians(INIT_PITCH_NOSE_DOWN_DEG), device=self.device)
        psi0 = torch.zeros((N, 1), device=self.device)
        omega0 = torch.zeros((N, 3), device=self.device)

        init_z = eta0 + init_clearance
        self.pos = torch.cat([init_x, init_y, init_z], dim=-1)
        self.vel = torch.cat([vx0, vy0, vz0], dim=-1)
        self.omega = omega0
        self.rot = euler_to_rot(phi0, theta0, psi0)

        self.motors = torch.full((N, 4), 1.1, device=self.device)
        self.rotor_tilts = torch.zeros((N, 4, 2), device=self.device)
        self.pusher_thrust = torch.full((N, 1), 2.5, device=self.device)
        self.nu_states = torch.zeros((N, 4, 3), device=self.device)
        self.nu_states[:, :, 0] = V_INDUCED_HOVER * 0.20

        self.ekf.reset(N, self.pos, self.vel, self.rot)
        self.follower_accel_prev = torch.zeros((N, 3), device=self.device)

        self.is_vent_L = torch.zeros((N, 1), device=self.device)
        self.is_vent_R = torch.zeros((N, 1), device=self.device)
        self.hydro_surfaces = torch.zeros((N, 3), device=self.device)
        self.hydro_surfaces_actual = torch.zeros((N, 3), device=self.device)
        self.act_motors_prev = torch.full((N, 4), 1.1, device=self.device)
        self.act_tilts_prev = torch.zeros((N, 4, 2), device=self.device)
        self.act_surf_prev = torch.zeros((N, 3), device=self.device)
        self.h_target_filtered.fill_(RIDE_HEIGHT_OPTIMAL)
        self.v_target_filtered.fill_(self.target_speed)
        self.e_int_height.zero_()

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
            motor_thrusts_total, morph_dict, self.sim_time, is_soaring
        )

        # 3. Simulated sensor hardware
        with torch.no_grad():
            rot_T_true = self.rot.transpose(-1, -2)
            v_water_rel_world = self.vel - v_water_orb
            v_water_rel_body = torch.matmul(rot_T_true, v_water_rel_world.unsqueeze(-1)).squeeze(-1)

            R_hat_prior = quat_to_rot(self.ekf.q_hat)
            v_body_est = torch.matmul(R_hat_prior.transpose(-1, -2), self.ekf.v_hat.unsqueeze(-1)).squeeze(-1)

            d_sub_hydrostatic, p_static_decoupled, p_dynamic_error, bernoulli_bias_cm = self.perception.sample_piezoresistive_tap(
                d_sub_physical, v_water_rel_body, v_body_est, self.rho_water_local
            )

            radar_raw, spray_dropout, radar_locked, multipath_rejected = self.perception.process_fmcw_radar_dsp(
                h_wing, deta_dx, deta_dy, v_water_rel_body[:, 0:1], v_water_rel_body[:, 2:3]
            )
            meas_radar = torch.where(radar_locked, radar_raw, self.clearance_hold)

            accel_body_true = torch.matmul(
                rot_T_true,
                (self.follower_accel_prev + torch.tensor([0.0, 0.0, GRAVITY], device=self.device)).unsqueeze(-1)
            ).squeeze(-1)
            meas_accel_body = accel_body_true + self.true_ba + torch.randn_like(accel_body_true) * 0.02
            meas_gyro_body = self.omega + self.true_bw + torch.randn_like(self.omega) * 0.008

            mag_meas_body = torch.matmul(rot_T_true, MAG_FIELD_WORLD.to(self.device).repeat(N, 1).unsqueeze(-1)).squeeze(-1) + torch.randn_like(self.omega) * 0.02
            meas_dvl_body = v_water_rel_body + torch.randn_like(v_water_rel_body) * 0.03

            celerity_est = self.dispersion_observer.omega_dom / torch.clamp(self.dispersion_observer.k_dom, min=0.1)
            h_flow_true = torch.clamp(h_wing, min=0.08)
            flow_rates_body = torch.cat([
                -(v_water_rel_body[:, 1:2]) / h_flow_true,
                +(v_water_rel_body[:, 0:1] - celerity_est) / h_flow_true
            ], dim=-1) + torch.randn((N, 2), device=self.device) * 0.04

            infl_true = 1.0 + 8.0 * torch.exp(-torch.clamp(h_wing, min=0.01, max=10.0) / 0.25)
            gnss_pos = self.pos + torch.randn_like(self.pos) * 0.05 * torch.sqrt(infl_true)
            gnss_vel = self.vel + torch.randn_like(self.vel) * 0.03 * torch.sqrt(infl_true)

            # 4. ES-EKF
            self.ekf.propagate(meas_accel_body, meas_gyro_body)
            self.ekf.update_magnetometer(mag_meas_body)

            is_submerged_sensor = (d_sub_hydrostatic > 0.015)
            dvl_acoustic_lock = (~is_ventilated_overall.bool()) & (torch.rand_like(is_ventilated_overall) > 0.01)
            dvl_lost_track = (~dvl_acoustic_lock).float()
            self.ekf.update_dvl(meas_dvl_body, is_submerged_sensor, dvl_lost_track)

            is_airborne_sensor = (d_sub_hydrostatic <= 0.015)
            self.ekf.update_optical_flow(flow_rates_body, meas_radar, celerity_est, is_airborne_sensor)

            if self.step_count % self.gnss_period_steps == 0:
                self.ekf.update_gnss(gnss_pos, gnss_vel, meas_radar)

            # 5. Navigation estimates
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

            # 6. Wave preview
            lookahead_dist = est_clearance * math.tan(self.perception.cant_angle) + 0.85
            lookahead_true = h_wing * math.tan(self.perception.cant_angle) + 0.85
            eta_forward_meas, eta_forward_valid = self.waves.raycast_forward_surface(
                self.pos, lookahead_true, self.sim_time, self.wave_curriculum_scale
            )
            eta_prev, deta_dx_prev, w_up_prev = self.dispersion_observer.update_and_predict(
                est_clearance, est_pos[:, 2:3], eta_forward_meas, eta_forward_valid, lookahead_dist,
                est_vel[:, 0:1], preview_time=1.25
            )

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

        # 8. Pure 16-Actuator Neural Flight Control System (26 states -> 16 actuators)
        e_speed = est_vel[:, 0:1] - self.v_target_filtered
        e_height = est_clearance - self.h_target_filtered
        self.e_int_height = torch.where(
            (is_foiling | is_takeoff | is_soaring),
            torch.clamp(self.e_int_height + e_height * self.dt, -0.12, 0.12),
            self.e_int_height * 0.95
        )

        actor_obs = torch.cat([
            e_speed,                                           # 0
            e_height,                                          # 1
            self.e_int_height,                                 # 2
            est_pos[:, 1:2],                                   # 3
            est_h_dot,                                         # 4
            est_clearance,                                     # 5
            d_sub_hydrostatic,                                 # 6
            self.v_target_filtered,                            # 7
            est_vel,                                           # 8..10
            est_omega,                                         # 11..13
            est_rot[:, :, 0],                                  # 14..16 (body +x in world: pitch + yaw)
            est_rot[:, :, 2],                                  # 17..19 (body +z in world: roll + pitch)
            eta_prev - (est_pos[:, 2:3] - est_clearance),      # 20
            deta_dx_prev,                                      # 21
            w_up_prev,                                         # 22
            is_soaring.float(),                                # 23
            (is_egress | is_flight).float(),                   # 24
            self.v_batt / BATTERY_V0                           # 25
        ], dim=-1)

        act_nn = actor_net(actor_obs)
        nn_motors = act_nn[:, 0:4]
        target_tilts = act_nn[:, 4:12].view(N, 4, 2)
        target_pusher = act_nn[:, 12:13]
        hydro_surfaces_cmd = act_nn[:, 13:16]

        # 9. Minimal boundary vertical CBF safety shield (intervenes ONLY at crash/breach boundaries)
        state_err = torch.cat([
            torch.zeros((N, 1), device=self.device), est_pos[:, 1:2], e_height,
            est_vel[:, 0:1] - self.v_target_filtered, est_vel[:, 1:2], est_h_dot
        ], dim=-1)
        M_metric = ncm_net(state_err)

        v_wave_apparent_est = deta_dx_prev * est_vel[:, 0:1] + w_up_prev
        h_dot_relative_est = est_vel[:, 2:3] - v_wave_apparent_est
        h_min_mode = torch.where(loiter_descent_ready | is_takeoff, 0.01, RIDE_HEIGHT_MIN)
        h_max_mode = torch.where(is_egress | is_flight | is_ingress, 4.00, torch.where(is_foiling | is_soaring, RIDE_HEIGHT_FOILING_MAX, RIDE_HEIGHT_WIG_MAX))

        b_cbf_crash = 0.75 * h_dot_relative_est + 3.80 * (est_clearance - h_min_mode)
        b_cbf_ceil = -0.80 * h_dot_relative_est + 3.50 * torch.clamp(h_max_mode - est_clearance, min=-0.5, max=1.0)

        nom_mass = float(morph_dict["total_mass"].detach().item())
        h_rotors_est = torch.clamp(est_clearance, min=0.04)
        ige_est = 1.0 / (1.0 - torch.clamp((PROP_RADIUS / (4.0 * h_rotors_est))**2, max=0.35))
        cos_tilt_est = est_rot[:, 2, 2:3].clamp(min=0.30)

        uz_nn = torch.sqrt((1.0 - target_tilts[:, :, 0]**2 - target_tilts[:, :, 1]**2).clamp(min=ROTOR_MAX_TILT_COS**2))
        fz_rotors_nn = torch.sum(nn_motors * uz_nn, dim=-1, keepdim=True) * ige_est * cos_tilt_est

        submerged_factor_est = torch.clamp(d_sub_hydrostatic / 0.04, min=0.0, max=1.0)
        v_horiz_est = safe_norm(est_vel[:, :2], dim=-1, keepdim=True).clamp(min=0.5)
        q_water_est = 0.5 * RHO_WATER_NOM * (v_horiz_est**2)
        q_air_est = 0.5 * RHO_AIR * (v_horiz_est**2)
        dec_mean = math.radians(0.5 * (FOIL_DECALAGE_DEG[0] + FOIL_DECALAGE_DEG[1]))
        v_b_est = torch.matmul(est_rot.transpose(-1, -2), est_vel.unsqueeze(-1)).squeeze(-1)
        alpha_foil_est = torch.atan2(-v_b_est[:, 2:3], v_b_est[:, 0:1].clamp(min=0.5)) + dec_mean
        coll_avg = 0.5 * (hydro_surfaces_cmd[:, 0:1] + hydro_surfaces_cmd[:, 1:2])
        f_hydro_lift_est = torch.clamp(
            q_water_est * morph_dict["foil_area"] * (FOIL_CL_ALPHA * alpha_foil_est + HYDRO_CL_DELTA * coll_avg) * submerged_factor_est,
            min=-1.0, max=2.0 * FOIL_LIFT_CLAMP_PER_SIDE
        )
        phi_wig_est = 1.0 + 0.35 / (0.10 + 2.2 * torch.pow(est_clearance.clamp(min=0.03) / morph_dict["wing_chord"], 1.15))
        alpha_wing_est = torch.atan2(-v_b_est[:, 2:3], v_b_est[:, 0:1].clamp(min=0.5)) + 0.03
        f_wing_lift_est = (q_air_est * morph_dict["wing_area"] * torch.clamp((WING_CL0 + 4.5 * alpha_wing_est) * phi_wig_est, -0.3, 1.4)).clamp(min=0.0, max=15.0)

        a_z_est_nn = (fz_rotors_nn + f_hydro_lift_est + f_wing_lift_est) / nom_mass - GRAVITY
        a_z_min = -b_cbf_crash
        a_z_max = torch.maximum(b_cbf_ceil, a_z_min)
        a_z_safe = torch.minimum(torch.maximum(a_z_est_nn, a_z_min), a_z_max)

        thrust_ceiling = MAX_THRUST_PER_MOTOR * (self.v_batt / BATTERY_V0)**2
        delta_fz_cbf = ((a_z_safe - a_z_est_nn) * nom_mass) / (ige_est * cos_tilt_est).clamp(min=0.4)
        target_motors = torch.minimum((nn_motors + 0.25 * delta_fz_cbf).clamp(min=MOTOR_IDLE_FLOOR), thrust_ceiling)
        qp_override = torch.abs(a_z_safe - a_z_est_nn)
        slack = F.relu(a_z_est_nn - a_z_max) + F.relu(a_z_min - a_z_est_nn)

        max_d_delta = SERVO_RATE_LIMIT * self.dt
        self.hydro_surfaces_actual = self.hydro_surfaces_actual + torch.clamp(
            hydro_surfaces_cmd - self.hydro_surfaces_actual, -max_d_delta, max_d_delta
        )
        self.hydro_surfaces = self.hydro_surfaces_actual

        wave_slope_mag = safe_norm(torch.cat([deta_dx, deta_dy], dim=-1), dim=-1, keepdim=True)

        # 10. Integrate ground-truth 6-DoF physics
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

        # Diagnostic world acceleration estimate for NCM / L1 telemetry
        a_safe_diag = torch.cat([
            (torch.sum(nn_motors * target_tilts[:, :, 0], dim=-1, keepdim=True) + target_pusher) / nom_mass,
            torch.sum(nn_motors * target_tilts[:, :, 1], dim=-1, keepdim=True) / nom_mass,
            a_z_safe
        ], dim=-1)
        sigma_l1 = self.l1_observer.step(a_est_world, a_safe_diag.detach(), self.dt)

        eta_post = eta + deta_dt * self.dt
        clearance_eval = (self.pos[:, 2:3] - eta_post).clamp(min=0.01)
        z_tip_L_post = self.pos[:, 2:3] + self.rot[:, 2, 1:2] * (morph_dict["foil_span"] * 0.25) - morph_dict["strut_length"]
        z_tip_R_post = self.pos[:, 2:3] - self.rot[:, 2, 1:2] * (morph_dict["foil_span"] * 0.25) - morph_dict["strut_length"]
        immersion_eval = 0.5 * (
            (eta_post - z_tip_L_post).clamp(min=0.0, max=0.22)
            + (eta_post - z_tip_R_post).clamp(min=0.0, max=0.22)
        )

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
        headroom_min = (torch.min(nn_motors, dim=-1, keepdim=True).values - MOTOR_IDLE_FLOOR)

        self.metrics = {
            "clearance_eval": clearance_eval,
            "immersion_eval": immersion_eval,
            "f_hydro_lift_eval": f_hydro[:, 2:3],
            "vel_eval": self.vel,
            "pos_eval": self.pos,
            "rot_eval": self.rot,
            "omega_eval": self.omega,
            "act_motors": nn_motors,
            "act_tilts": target_tilts,
            "act_surf": hydro_surfaces_cmd,
            "act_motors_prev": self.act_motors_prev,
            "act_tilts_prev": self.act_tilts_prev,
            "act_surf_prev": self.act_surf_prev,
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
            "is_clipped_min": (self.motors <= (MOTOR_IDLE_FLOOR + 0.01)),
            "is_clipped_max": (self.motors >= thrust_ceiling - 1e-3),
            "power_saved_pct": power_saved_pct,
            "p_elec": p_elec,
            "v_batt": self.v_batt,
            "i_batt": i_batt,
            "soc": self.soc * 100.0,
            "qp_override": qp_override,
            "max_slack": slack,
            "actor_obs": actor_obs,
            "a_safe": a_safe_diag,
            "state_err": state_err,
            "pos": self.pos,
            "vel": self.vel,
            "eta": eta,
            "omega": self.omega,
            "e_R": torch.cat([self.rot[:, 2, 1:2], -self.rot[:, 2, 0:1], self.rot[:, 1, 0:1]], dim=-1),
            "h_target_mode": self.h_target_filtered,
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
            "foil_lift_est": f_hydro_lift_est,
            "status_banner": "[NOMINAL]"
        }

        self.act_motors_prev = nn_motors
        self.act_tilts_prev = target_tilts
        self.act_surf_prev = hydro_surfaces_cmd

        if export_diagnostics:
            phi_deg = math.degrees(math.atan2(est_rot[0, 2, 1].item(), est_rot[0, 2, 2].item()))
            theta_deg = math.degrees(math.asin(float(np.clip(-est_rot[0, 2, 0].item(), -0.9999, 0.9999))))
            psi_actual_deg = math.degrees(math.atan2(est_rot[0, 1, 0].item(), est_rot[0, 0, 0].item()))
            k_val = max(0.01, self.dispersion_observer.k_dom[0, 0].item())
            wavelength_m = (2.0 * math.pi) / k_val
            celerity_mps = self.dispersion_observer.omega_dom[0, 0].item() / k_val

            min_clipped = self.metrics["is_clipped_min"][0].cpu().numpy()
            max_clipped = self.metrics["is_clipped_max"][0].cpu().numpy()
            if any(min_clipped) and any(max_clipped):
                sat_str = "[MIN+MAX CLIPPED]"
            elif any(min_clipped):
                sat_str = "[MIN FLOOR]"
            elif any(max_clipped):
                sat_str = "[MAX CEIL]"
            else:
                sat_str = "[CLEAN]"

            cell_temp_c = self.t_cell[0, 0].item() - 273.15
            i_batt_val = i_batt[0, 0].item()
            r0_val = BATTERY_R0_NOM * math.exp(ARRHENIUS_COEFF * (1.0 / max(270.0, self.t_cell[0, 0].item()) - 1.0 / T_AMBIENT_K))
            v_sag = i_batt_val * r0_val

            mode_names = ["LOITER", "TAKEOFF", "FOILING", "EGRESS", "FLIGHT", "INGRESS", "SOARING"]
            current_mode_name = mode_names[self.current_mode[0, 0].item()]

            d_eL_deg = math.degrees(self.hydro_surfaces[0, 0].item())
            d_eR_deg = math.degrees(self.hydro_surfaces[0, 1].item())
            d_rud_deg = math.degrees(self.hydro_surfaces[0, 2].item())

            q_water_val = max(1.0, aoa_tensors["q_water"][0, 0].item())
            p_static_val = max(100.0, p_static_decoupled[0, 0].item())
            alpha_foil_val = aoa_tensors["alpha_foil"][0, 0].item()
            cavitation_sigma = (p_static_val - P_VAPOR_WATER) / (q_water_val * (1.0 + 2.4 * max(0.0, alpha_foil_val)))
            froude_fn = est_vel[0, 0].item() / math.sqrt(GRAVITY * HULL_LENGTH)
            stall_margin_deg = math.degrees(math.radians(12.5) - abs(alpha_foil_val))

            sigma_z_cm = math.sqrt(max(1e-6, self.ekf.P[0, 2, 2].item())) * 100.0
            bias_omega_deg = tuple(math.degrees(b) for b in self.ekf.bw_hat[0].cpu().numpy())

            if qp_override[0, 0].item() < 0.05:
                barrier_name = "[NN DIRECT (Unconstrained)]"
            elif a_z_est_nn[0, 0].item() < a_z_min[0, 0].item():
                barrier_name = "[CRASH BARRIER (Floor)]"
            else:
                barrier_name = "[BREACH BARRIER (Ceiling)]"

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
                "psi_des_deg": 0.0,
                "psi_err_deg": psi_actual_deg,
                "beta_sideslip_deg": math.degrees(math.atan2(est_vel[0, 1].item(), max(0.1, est_vel[0, 0].item()))),
                "wavelength_m": wavelength_m,
                "celerity_mps": celerity_mps,
                "f_slam_z": f_slam[0, 0].item(),
                "f_planing_z": (f_buoy[0, 0] + aoa_tensors["f_planing_z"][0, 0]).item(),
                "sat_str": sat_str,
                "torque_auth_pct": 100.0,
                "cell_temp_c": cell_temp_c,
                "v_sag": v_sag,
                "euler_deg": (phi_deg, theta_deg, psi_actual_deg),
                "mode_name": current_mode_name,
                "hydro_deflections": (d_eL_deg, d_eR_deg, d_rud_deg),
                "margin_ceil_cm": (h_max_mode[0, 0].item() - est_clearance[0, 0].item()) * 100.0,
                "margin_floor_cm": (est_clearance[0, 0].item() - h_min_mode[0, 0].item()) * 100.0,
                "alpha_total_deg": math.degrees(alpha_foil_val),
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
# 13. STAGE 0 PRE-TRAINING ENGINE (RICCATI METRIC & 16-ACTUATOR TEACHER IMITATION)
# =============================================================================
def compute_teacher_actuators(obs_batch, robot, morph_dict, device):
    """Offline Stage-0 teacher ONLY (never used at runtime or in cockpit).
    Generates stabilized 16-channel actuator targets from the 26D observation batch."""
    B = obs_batch.shape[0]
    e_speed = obs_batch[:, 0:1]
    e_height = obs_batch[:, 1:2]
    e_int_h = obs_batch[:, 2:3]
    e_y = obs_batch[:, 3:4]
    h_dot = obs_batch[:, 4:5]
    est_clearance = obs_batch[:, 5:6]
    d_sub = obs_batch[:, 6:7]
    v_target = obs_batch[:, 7:8]
    est_vel = obs_batch[:, 8:11]
    est_omega = obs_batch[:, 11:14]
    rot_c0 = obs_batch[:, 14:17]
    rot_c2 = obs_batch[:, 17:20]
    eta_prev_err = obs_batch[:, 20:21]
    deta_dx_prev = obs_batch[:, 21:22]
    is_soaring = obs_batch[:, 23:24] > 0.5
    is_air_mode = obs_batch[:, 24:25] > 0.5
    v_batt = obs_batch[:, 25:26] * BATTERY_V0

    rot_c1 = torch.cross(rot_c2, rot_c0, dim=-1)
    est_rot = torch.stack([rot_c0, rot_c1, rot_c2], dim=-1)

    a_x = -1.8 * e_speed
    a_y = -2.4 * est_vel[:, 1:2] - 2.8 * e_y
    a_z = -est_vel[:, 0:1] * 1.6 * deta_dx_prev - 9.5 * e_height - 3.5 * e_int_h - 4.2 * h_dot - 1.5 * eta_prev_err

    submerged_factor = torch.clamp(d_sub / 0.04, min=0.0, max=1.0)
    phi_max = submerged_factor * math.radians(6.5) + (1.0 - submerged_factor) * math.radians(18.0)
    v_horiz = safe_norm(est_vel[:, :2], dim=-1, keepdim=True).clamp(min=0.5)
    q_water = 0.5 * RHO_WATER_NOM * (v_horiz**2)
    q_air = 0.5 * RHO_AIR * (v_horiz**2)

    f_hydro_drag = q_water * morph_dict["foil_area"].detach() * FOIL_CD0 * submerged_factor
    f_drag = f_hydro_drag + q_air * morph_dict["wing_area"].detach() * (WING_CD0 + 0.03)
    nom_mass = float(morph_dict["total_mass"].detach().item())

    dec_mean = math.radians(0.5 * (FOIL_DECALAGE_DEG[0] + FOIL_DECALAGE_DEG[1]))
    v_b = torch.matmul(est_rot.transpose(-1, -2), est_vel.unsqueeze(-1)).squeeze(-1)
    alpha_foil = torch.atan2(-v_b[:, 2:3], v_b[:, 0:1].clamp(min=0.5)) + dec_mean
    q_A_CLa = q_water * morph_dict["foil_area"].detach() * FOIL_CL_ALPHA
    f_hydro_unclamped = q_A_CLa * alpha_foil * submerged_factor

    phi_wig = 1.0 + 0.35 / (0.10 + 2.2 * torch.pow(est_clearance.clamp(min=0.03) / morph_dict["wing_chord"].detach(), 1.15))
    alpha_wing = torch.atan2(-v_b[:, 2:3], v_b[:, 0:1].clamp(min=0.5)) + 0.03
    f_wing_lift = (q_air * morph_dict["wing_area"].detach() * torch.clamp((WING_CL0 + 4.5 * alpha_wing) * phi_wig, -0.3, 1.4)).clamp(min=0.0, max=15.0)
    tau_hydro_pitch = f_hydro_drag * morph_dict["strut_length"].detach()

    min_rotor_floor = torch.where(is_soaring, torch.full((B, 1), 4.0, device=device), torch.full((B, 1), 4.4, device=device))
    desired_fz = nom_mass * (GRAVITY + a_z)
    lift_cap = desired_fz - min_rotor_floor - f_wing_lift
    over_lift = F.relu(f_hydro_unclamped - lift_cap)
    q_A_CLd = q_water * morph_dict["foil_area"].detach() * HYDRO_CL_DELTA * submerged_factor
    coll_max = 0.8 * MAX_HYDRO_DEFLECTION
    delta_collective = -torch.clamp(over_lift / (q_A_CLd + 1.0), max=coll_max) * (submerged_factor > 0.05).float()
    theta_shed = torch.clamp(F.relu(over_lift - q_A_CLd * coll_max) / (q_A_CLa + 50.0), max=math.radians(6.0)) * (submerged_factor > 0.05).float()

    f_hydro_lift = torch.clamp(f_hydro_unclamped + q_A_CLd * delta_collective, min=-1.0, max=2.0 * FOIL_LIFT_CLAMP_PER_SIDE)
    f_rotors_needed = torch.maximum(desired_fz - f_hydro_lift - f_wing_lift, min_rotor_floor).clamp(max=float(nom_mass * GRAVITY + 6.0))

    psi_est = torch.atan2(est_rot[:, 1:2, 0], est_rot[:, 0:1, 0])
    c_psi, s_psi = torch.cos(psi_est), torch.sin(psi_est)
    Fx_dyn = nom_mass * (c_psi * a_x + s_psi * a_y)
    Fy_dyn = nom_mass * (-s_psi * a_x + c_psi * a_y)

    f_gimbal_cap = f_rotors_needed * (ROTOR_MAX_TILT_TAN * 0.70)
    Fx_gimbal_est = torch.where(is_soaring, torch.zeros_like(Fx_dyn), torch.clamp(Fx_dyn, -f_gimbal_cap, f_gimbal_cap))
    pusher_disabled = (v_target < 0.5) & (est_vel[:, 0:1] < 0.8)
    target_pusher = torch.where(
        pusher_disabled | is_soaring,
        torch.zeros_like(Fx_dyn),
        torch.clamp(Fx_dyn + f_drag - Fx_gimbal_est, min=0.0, max=MAX_PUSHER_THRUST)
    )
    Fx_cmd_gimbal = torch.where(is_soaring, torch.zeros_like(Fx_dyn), Fx_dyn + f_drag - target_pusher)
    Fx_residual = Fx_cmd_gimbal - torch.clamp(Fx_cmd_gimbal, -f_gimbal_cap, f_gimbal_cap)
    Fy_residual = Fy_dyn - torch.clamp(Fy_dyn, -f_gimbal_cap, f_gimbal_cap)

    f_vert_net = (nom_mass * GRAVITY - f_hydro_lift - f_wing_lift).clamp(min=4.0)
    beta_fluid = torch.atan2(est_vel[:, 1:2], est_vel[:, 0:1].clamp(min=1.0))
    phi_des = torch.clamp(torch.atan2(-Fy_residual, f_vert_net) + 0.22 * torch.clamp(beta_fluid, -math.radians(10.0), math.radians(10.0)), -phi_max, phi_max)
    theta_bias = torch.where(is_air_mode, -math.radians(2.0), torch.zeros_like(Fx_residual))
    theta_des = torch.clamp(torch.atan2(Fx_residual, f_vert_net) + theta_bias + theta_shed, -math.radians(14.0), math.radians(24.0))
    psi_des = -torch.clamp(e_y * 0.12 + 0.15 * est_vel[:, 1:2], -math.radians(5.0), math.radians(5.0))

    Rd = euler_to_rot(phi_des, theta_des, psi_des)
    R_err = torch.matmul(Rd.transpose(-1, -2), est_rot) - torch.matmul(est_rot.transpose(-1, -2), Rd)
    tau_des = torch.cat([
        -11.5 * R_err[:, 2, 1:2] * 0.5 - 1.6 * est_omega[:, 0:1],
        -12.0 * R_err[:, 0, 2:3] * 0.5 - 1.8 * est_omega[:, 1:2] - 1.00 * tau_hydro_pitch,
        -5.0 * R_err[:, 1, 0:1] * 0.5 - 0.8 * est_omega[:, 2:3]
    ], dim=-1)

    ige_est = 1.0 / (1.0 - torch.clamp((PROP_RADIUS / (4.0 * est_clearance.clamp(min=0.04)))**2, max=0.35))
    t_motors, t_tilts, t_surf, _, _, _, _, _, _ = robot.allocate_wrench_hybrid(
        Fx_cmd_gimbal, Fy_dyn, f_rotors_needed, tau_des, v_batt, ige_est,
        d_sub, est_vel[:, 0:1], is_soaring, morph_dict, delta_collective=delta_collective
    )
    return torch.cat([t_motors, t_tilts.view(B, 8), target_pusher, t_surf], dim=-1)


def pretrain_aeromarine_stack(actor_net, ncm_net, morph_dict, device=DEVICE):
    print("\n" + "=" * 114)
    print(" [STAGE 0 PRE-TRAINING] RICCATI RIEMANNIAN METRIC & 16-ACTUATOR NN FLIGHT CONTROLLER IMITATION")
    print("=" * 114)

    P_sub_nom = torch.diag(torch.tensor([2.8, 2.5, 3.4], device=device))
    ncm_pre_opt = torch.optim.Adam(ncm_net.parameters(), lr=4.0e-3)
    for _ in range(250):
        dummy_state = torch.randn((512, 6), device=device) * 0.15
        M_pred = ncm_net(dummy_state)
        loss_val = F.mse_loss(M_pred[:, 3:6, 3:6], P_sub_nom.unsqueeze(0).repeat(512, 1, 1))
        perturb = torch.randn_like(dummy_state) * 0.02
        M_pert = ncm_net(dummy_state + perturb)
        loss_flat = F.mse_loss(M_pert, M_pred) * 2.0
        ncm_pre_opt.zero_grad()
        (loss_val + loss_flat).backward()
        ncm_pre_opt.step()
    print("  ├─ 1. RIEMANNIAN METRIC PRE-TRAIN: Stabilizing block-decoupled Riccati metric.")

    robot_teacher = WaveRidingAerialRobot(0.015, device)
    actor_pre_opt = torch.optim.AdamW(actor_net.parameters(), lr=3.0e-3, weight_decay=1e-5)
    actor_sched = make_cosine(actor_pre_opt, 3.0e-3, 800, eta_min=2.0e-4)
    batch_size = 2048
    act_scale = torch.tensor(
        [10.0] * 4 + [ROTOR_MAX_TILT_SIN] * 8 + [MAX_PUSHER_THRUST] + [MAX_HYDRO_DEFLECTION] * 3,
        device=device
    ).view(1, NN_ACTION_DIM)

    last_bc_loss = 0.0
    for _ in range(800):
        mode_draw = torch.rand((batch_size, 1), device=device)
        is_loiter = (mode_draw < 0.12)
        is_soaring = (mode_draw >= 0.12) & (mode_draw < 0.24)
        is_air = (mode_draw >= 0.24) & (mode_draw < 0.40)

        v_target = torch.where(is_loiter, torch.zeros((batch_size, 1), device=device),
                   torch.where(is_air, torch.full((batch_size, 1), 8.50, device=device),
                                       torch.full((batch_size, 1), 5.20, device=device)))
        h_target = torch.where(is_air, torch.full((batch_size, 1), FREE_FLIGHT_ALTITUDE, device=device),
                   torch.where(is_soaring, torch.full((batch_size, 1), RIDE_HEIGHT_SOARING, device=device),
                                           torch.full((batch_size, 1), RIDE_HEIGHT_OPTIMAL, device=device)))

        e_speed = torch.empty((batch_size, 1), device=device).uniform_(-1.2, 1.2)
        e_height = torch.empty((batch_size, 1), device=device).uniform_(-0.12, 0.25)
        est_clearance = torch.clamp(h_target + e_height, min=0.04, max=2.5)
        e_int_h = torch.empty((batch_size, 1), device=device).uniform_(-0.08, 0.08)
        e_y = torch.empty((batch_size, 1), device=device).uniform_(-0.6, 0.6)
        h_dot = torch.empty((batch_size, 1), device=device).uniform_(-0.6, 0.6)
        d_sub = torch.where(is_air, torch.zeros_like(e_height), torch.clamp(0.258 - est_clearance, min=0.0, max=0.18))

        vx = torch.clamp(v_target + e_speed, min=0.0)
        vy = torch.empty((batch_size, 1), device=device).uniform_(-0.5, 0.5)
        est_vel = torch.cat([vx, vy, h_dot], dim=-1)
        est_omega = torch.randn((batch_size, 3), device=device) * 0.45

        phi = torch.empty((batch_size, 1), device=device).uniform_(-math.radians(14.0), math.radians(14.0))
        theta = torch.empty((batch_size, 1), device=device).uniform_(-math.radians(12.0), math.radians(14.0))
        psi = torch.empty((batch_size, 1), device=device).uniform_(-math.radians(18.0), math.radians(18.0))
        est_rot = euler_to_rot(phi, theta, psi)

        eta_prev_err = torch.empty((batch_size, 1), device=device).uniform_(-0.12, 0.12)
        deta_dx_prev = torch.empty((batch_size, 1), device=device).uniform_(-0.18, 0.18)
        w_up_prev = torch.empty((batch_size, 1), device=device).uniform_(-0.4, 0.4)
        v_batt_ratio = torch.empty((batch_size, 1), device=device).uniform_(0.88, 1.01)

        obs_batch = torch.cat([
            vx - v_target, est_clearance - h_target, e_int_h, e_y, h_dot,
            est_clearance, d_sub, v_target, est_vel, est_omega,
            est_rot[:, :, 0], est_rot[:, :, 2],
            eta_prev_err, deta_dx_prev, w_up_prev,
            is_soaring.float(), is_air.float(), v_batt_ratio
        ], dim=-1)

        with torch.no_grad():
            target_act = compute_teacher_actuators(obs_batch, robot_teacher, morph_dict, device)

        pred_act = actor_net(obs_batch)
        loss_actor_bc = F.mse_loss(pred_act / act_scale, target_act / act_scale)
        actor_pre_opt.zero_grad()
        loss_actor_bc.backward()
        torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=1.0)
        actor_pre_opt.step()
        actor_sched.step()
        last_bc_loss = loss_actor_bc.item()

    print(f"  └─ 2. 16-ACTUATOR NN PRE-TRAIN: Imitated 6-DoF + 15-deg gimbal trim across all 7 modes (norm MSE: {last_bc_loss:.5f}).\n")


# =============================================================================
# 14. MULTI-STAGE END-TO-END DIFFERENTIABLE PHYSICS TRAINING ENGINE
# =============================================================================
def estimate_nn_world_accel(act_tensor, obs_tensor, robot, morph_dict):
    """Differentiable instantaneous body-to-world rotor+pusher+foil acceleration model
    used to extract the true closed-loop Jacobian A_cl_5 of the 16-actuator NN."""
    motors = act_tensor[:, 0:4]
    tilts = act_tensor[:, 4:12].view(-1, 4, 2)
    pusher = act_tensor[:, 12:13]
    surf = act_tensor[:, 13:16]

    est_clearance = obs_tensor[:, 5:6]
    d_sub = obs_tensor[:, 6:7]
    est_vel = obs_tensor[:, 8:11]
    rot_c0 = obs_tensor[:, 14:17]
    rot_c2 = obs_tensor[:, 17:20]
    rot_c1 = torch.cross(rot_c2, rot_c0, dim=-1)
    est_rot = torch.stack([rot_c0, rot_c1, rot_c2], dim=-1)

    uz = torch.sqrt((1.0 - tilts[:, :, 0]**2 - tilts[:, :, 1]**2).clamp(min=ROTOR_MAX_TILT_COS**2))
    h_rot = torch.clamp(est_clearance, min=0.04)
    ige = 1.0 / (1.0 - torch.clamp((PROP_RADIUS / (4.0 * h_rot))**2, max=0.35))

    f_body = torch.cat([
        torch.sum(motors * tilts[:, :, 0], dim=-1, keepdim=True) + pusher,
        torch.sum(motors * tilts[:, :, 1], dim=-1, keepdim=True),
        torch.sum(motors * uz, dim=-1, keepdim=True) * ige
    ], dim=-1)

    sub_frac = (d_sub / 0.040).clamp(min=0.0, max=1.0)
    v_b = torch.matmul(est_rot.transpose(-1, -2), est_vel.unsqueeze(-1)).squeeze(-1)
    q_water = 0.5 * RHO_WATER_NOM * (safe_norm(est_vel[:, :2], dim=-1, keepdim=True).clamp(min=0.5)**2)
    dec_mean = math.radians(0.5 * (FOIL_DECALAGE_DEG[0] + FOIL_DECALAGE_DEG[1]))
    alpha_foil = torch.atan2(-v_b[:, 2:3], v_b[:, 0:1].clamp(min=0.5)) + dec_mean
    coll_avg = 0.5 * (surf[:, 0:1] + surf[:, 1:2])
    f_foil_z = torch.clamp(
        q_water * morph_dict["foil_area"].detach() * (FOIL_CL_ALPHA * alpha_foil + HYDRO_CL_DELTA * coll_avg) * sub_frac,
        min=-1.0, max=2.0 * FOIL_LIFT_CLAMP_PER_SIDE
    )
    f_body_total = f_body + torch.cat([torch.zeros_like(f_foil_z), torch.zeros_like(f_foil_z), f_foil_z], dim=-1)
    f_world = torch.matmul(est_rot, f_body_total.unsqueeze(-1)).squeeze(-1)
    nom_mass = float(morph_dict["total_mass"].detach().item())
    return f_world / nom_mass - torch.tensor([[0.0, 0.0, GRAVITY]], device=act_tensor.device)


def train_wave_riding_pilot(epochs=96, num_envs=2048, rollout_steps=200, tbptt_steps=10):
    import warnings

    print("\n" + "=" * 114)
    print(" TRAINING END-TO-END 16-ACTUATOR NEURAL FLIGHT CONTROLLER + BI-LEVEL MORPHOGENETICS")
    print(f" Envs: {num_envs:,} | Steps: {rollout_steps} | TBTT: {tbptt_steps} | Pure NN (26->16) | 5D NCM")
    print("=" * 114)

    env = WaveRidingAutonomyEnv(num_envs=num_envs, dt=0.015, device=DEVICE, use_embedded_qp=False)
    actor_net = FullNeuralActuatorPilot(state_dim=NN_STATE_DIM, action_dim=NN_ACTION_DIM).to(DEVICE)
    ncm_net = NeuralContractionMetric(state_dim=6, alpha=0.10).to(DEVICE)

    with torch.no_grad():
        init_morph_dict = env.morphology()
    pretrain_aeromarine_stack(actor_net, ncm_net, init_morph_dict, device=DEVICE)

    optimizer_actor = torch.optim.AdamW(actor_net.parameters(), lr=8.0e-4, weight_decay=1e-5)
    optimizer_ncm = torch.optim.AdamW(ncm_net.parameters(), lr=3.5e-3, weight_decay=1e-5)
    optimizer_morph = torch.optim.Adam(env.morphology.parameters(), lr=8.0e-4)

    scheduler_actor = make_cosine(optimizer_actor, 8.0e-4, 28)
    scheduler_ncm = make_cosine(optimizer_ncm, 3.5e-3, 28)

    scaler = make_grad_scaler(DEVICE, enabled=(DEVICE.type == "cuda"), init_scale=128.0)
    device_str = "cuda" if DEVICE.type == "cuda" else "cpu"

    delta_v, delta_p = 0.03, 0.03
    pert_mask = torch.zeros((10, 1, NN_STATE_DIM), device=DEVICE)
    pert_mask[0, 0, 0] = +delta_v; pert_mask[0, 0, 8] = +delta_v
    pert_mask[1, 0, 0] = -delta_v; pert_mask[1, 0, 8] = -delta_v
    pert_mask[2, 0, 9] = +delta_v
    pert_mask[3, 0, 9] = -delta_v
    pert_mask[4, 0, 4] = +delta_v; pert_mask[4, 0, 10] = +delta_v
    pert_mask[5, 0, 4] = -delta_v; pert_mask[5, 0, 10] = -delta_v
    pert_mask[6, 0, 3] = +delta_p
    pert_mask[7, 0, 3] = -delta_p
    pert_mask[8, 0, 1] = +delta_p; pert_mask[8, 0, 5] = +delta_p
    pert_mask[9, 0, 1] = -delta_p; pert_mask[9, 0, 5] = -delta_p

    kin_top_5 = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ], device=DEVICE).unsqueeze(0).repeat(num_envs, 1, 1)
    diag_jitter_5 = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=DEVICE) * 1e-4).unsqueeze(0)

    for epoch in range(epochs):
        optimizer_actor.zero_grad()
        optimizer_ncm.zero_grad()
        optimizer_morph.zero_grad()

        if epoch == 28:
            scheduler_actor = make_cosine(optimizer_actor, 6.0e-4, 28)
            scheduler_ncm = make_cosine(optimizer_ncm, 2.5e-3, 28)
        elif epoch == 56:
            scheduler_actor = make_cosine(optimizer_actor, 4.0e-4, epochs - 56)
            scheduler_ncm = make_cosine(optimizer_ncm, 2.0e-3, epochs - 56)

        if epoch < 28:
            wave_scale = 0.0; alpha_t = 0.05
            curriculum_stage = "Stage 1 (Flat Water Trim)"
        elif epoch < 56:
            wave_scale = 0.4; alpha_t = 0.04
            curriculum_stage = "Stage 2 (Gentle Swells)"
        else:
            wave_scale = 1.0; alpha_t = 0.025
            curriculum_stage = "Stage 3 (Full Ocean Swells)"

        env.set_wave_curriculum(wave_scale)
        env.reset()

        c_h_acc = 0.0
        c_ceil_acc = 0.0
        c_def_acc = 0.0
        c_drift_acc = 0.0
        c_omega_acc = 0.0
        nan_encountered = False

        with get_amp_autocast(device_str, enabled=False):
            window_loss = torch.zeros(1, device=DEVICE)

            for step_idx in range(rollout_steps):
                m = env.step(actor_net, ncm_net, export_diagnostics=False)

                h_lower_gate = torch.where(
                    (m["immersion_eval"] < 0.040),
                    torch.full_like(m["clearance_eval"], 0.145),
                    torch.full_like(m["clearance_eval"], 0.165)
                )
                h_eval_safe = torch.clamp(m["clearance_eval"], min=0.01, max=1.20)
                low_altitude_penalty = torch.mean(F.relu(h_lower_gate - h_eval_safe)**2) * 2000.0
                high_altitude_penalty = torch.mean(F.relu(h_eval_safe - 0.228)**2) * 800.0
                e_h = torch.clamp(h_eval_safe - m["h_target_mode"], min=-0.5, max=0.5)
                loss_h_base = torch.mean(e_h**2) * 350.0 + torch.mean(m["e_int_height"]**2) * 80.0
                loss_height = loss_h_base + high_altitude_penalty + low_altitude_penalty

                loss_airborne = torch.mean(F.relu(0.018 - m["immersion_eval"])) * 150.0 + \
                                torch.mean(F.relu(0.008 - m["immersion_eval"])**2) * 1500.0
                loss_deep_hull = torch.mean(F.relu(m["immersion_eval"] - 0.095)**2) * 1000.0
                loss_foil_lift = torch.mean(F.relu(14.0 - m["f_hydro_lift_eval"])) * 6.0

                loss_speed = torch.mean((m["vel_eval"][:, 0:1] - env.target_speed)**2) * 0.40
                loss_drift = torch.mean(m["vel_eval"][:, 1:2]**2) * 0.60 + torch.mean(m["pos_eval"][:, 1:2]**2) * 0.50
                loss_strut_stress = torch.mean(m["strut_moment_nm"]) * 0.05
                loss_vent = torch.mean(m["is_vent_L"].float() + m["is_vent_R"].float()) * 8.0
                loss_clipping = torch.mean(F.relu(0.05 - m["headroom_min"])**2) * 120.0

                # Direct NN attitude stabilization + rate damping + actuator smoothness + power efficiency
                loss_attitude = (
                    torch.mean(m["rot_eval"][:, 2, 1]**2) * 25.0
                    + torch.mean(m["rot_eval"][:, 2, 0]**2) * 10.0
                    + torch.mean(m["rot_eval"][:, 1, 0]**2) * 12.0
                )
                loss_omega = torch.mean(m["omega_eval"]**2) * 0.60
                loss_smooth = (
                    torch.mean((m["act_motors"] - m["act_motors_prev"])**2) * 0.04
                    + torch.mean((m["act_tilts"] - m["act_tilts_prev"])**2) * 2.5
                    + torch.mean((m["act_surf"] - m["act_surf_prev"])**2) * 1.0
                )
                loss_power = torch.mean(m["p_elec"]) * 0.004
                loss_safety_slack = torch.mean(m["max_slack"]) * 5.0

                step_cost = (
                    loss_height + loss_airborne + loss_deep_hull + loss_foil_lift + loss_speed +
                    loss_drift + loss_strut_stress + loss_vent + loss_clipping +
                    loss_attitude + loss_omega + loss_smooth + loss_power + loss_safety_slack
                )

                c_h_acc += loss_h_base.item()
                c_ceil_acc += (high_altitude_penalty + low_altitude_penalty).item()
                c_def_acc += (loss_airborne + loss_deep_hull + loss_foil_lift).item()
                c_drift_acc += loss_drift.item()
                c_omega_acc += (loss_omega + loss_attitude * 0.1).item()

                window_loss = window_loss + step_cost / rollout_steps

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
            c_drift_scalar = c_drift_acc / rollout_steps
            c_omega_scalar = c_omega_acc / rollout_steps

            # 5D NCM certification on the true closed-loop NN -> 16-actuator -> accel Jacobian
            state_err = m["state_err"].detach().float()
            obs_base = m["actor_obs"].detach().float()
            N = num_envs

            with torch.no_grad():
                morph_now = env.morphology()
                stacked_obs = (obs_base.unsqueeze(0) + pert_mask).view(-1, NN_STATE_DIM)
                stacked_acts = actor_net(stacked_obs)
                stacked_accels = estimate_nn_world_accel(stacked_acts, stacked_obs, env.robot, morph_now)
                splits = torch.split(stacked_accels, N, dim=0)
                J_v = torch.stack([(splits[2*k] - splits[2*k+1]) / (2.0 * delta_v) for k in range(3)], dim=-1)
                J_p_yz = torch.stack([(splits[6 + 2*k] - splits[6 + 2*k+1]) / (2.0 * delta_p) for k in range(2)], dim=-1)
                row_bot_5 = torch.cat([J_p_yz, J_v], dim=-1)
                A_cl_5 = torch.cat([kin_top_5, row_bot_5], dim=-2)

            M_full = ncm_net(state_err)
            M_5 = M_full[:, 1:, 1:]

            e_v = state_err[:, 3:6]
            a_flight_cmd = m["a_safe"].detach().float()
            e_dot = torch.cat([e_v, a_flight_cmd], dim=-1)

            eps_lie = 2.0e-3
            M_plus = ncm_net(state_err + eps_lie * e_dot)[:, 1:, 1:]
            M_minus = ncm_net(state_err - eps_lie * e_dot)[:, 1:, 1:]
            M_dot_5 = torch.clamp((M_plus - M_minus) / (2.0 * eps_lie), -25.0, 25.0)

            R_5 = M_dot_5 + torch.matmul(A_cl_5.transpose(-1, -2), M_5) + torch.matmul(M_5, A_cl_5) + 2.0 * alpha_t * M_5
            R_sym = 0.5 * (R_5 + R_5.transpose(-1, -2))
            R_reg = torch.nan_to_num(R_sym + diag_jitter_5, nan=0.0, posinf=1e3, neginf=-1e3)

            _, eigvecs_R = torch.linalg.eigh(R_reg)
            v_det = eigvecs_R.detach()
            eigvals_R = torch.sum(v_det * torch.matmul(R_reg, v_det), dim=-2)
            violation_all = F.relu(eigvals_R + 0.015)
            loss_viol_all = F.smooth_l1_loss(violation_all, torch.zeros_like(violation_all), beta=0.20, reduction='mean') * 50.0

            M_sym = 0.5 * (M_5 + M_5.transpose(-1, -2))
            M_reg = torch.nan_to_num(M_sym + diag_jitter_5, nan=0.0, posinf=1e3, neginf=-1e3)
            _, eigvecs_M = torch.linalg.eigh(M_reg)
            v_M_det = eigvecs_M.detach()
            eigvals_M = torch.sum(v_M_det * torch.matmul(M_reg, v_M_det), dim=-2)
            cond_M_tensor = (eigvals_M.max(dim=-1).values / eigvals_M.min(dim=-1).values.clamp(min=1e-3)).mean()

            w_contraction = min(1.0, (epoch + 1) / 6.0)
            ncm_loss = (loss_viol_all * 12.0 * w_contraction) + cond_M_tensor * 0.005

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
            max_lambda_R = eigvals_R.max().item()
            min_lambda_R = eigvals_R.min().item()
            min_lambda_M = eigvals_M.min().item()
            cert_status = "SAMPLED OK (<= 0)" if max_lambda_R <= 0.0 else "VIOLATED (> 0)"

            h_m = m["h_wing"].mean().item()
            h_est = m["clearance_est"].mean().item()
            d_sub_cm = m["immersion_depth"].mean().item() * 100.0
            p_elec_scalar = m["p_elec"].mean().item()
            v_batt_scalar = m["v_batt"].mean().item()
            i_batt_scalar = m["i_batt"].mean().item()
            soc_scalar = m["soc"].mean().item()
            qp_override_scalar = m["qp_override"].mean().item()
            flow_status = "VENTILATED" if m["is_vent_L"].float().mean().item() > 0.5 else "PRIME (Attached)"
            f_rot_mean = m["f_rotors"].mean().item()
            f_push_mean = m["f_pusher"].mean().item()
            mean_gimbal_deg = m["rotor_tilt_deg"].mean().item()

            lr_act = scheduler_actor.get_last_lr()[0]
            lr_ncm = scheduler_ncm.get_last_lr()[0]

            m_dict = env.morphology()
            span_val = m_dict["wing_span"].item()
            ar_val = m_dict["aspect_ratio"].item()
            f_span_val = m_dict["foil_span"].item()
            strut_val = m_dict["strut_length"].item()
            mass_val = m_dict["total_mass"].item()
            anhed_val = math.degrees(m_dict["anhedral_rad"].item())

            print(
                f"\n[Epoch {epoch+1:02d}/{epochs} | {curriculum_stage} | α={alpha_t:.3f}] ──────────────────────────────────────────────────────────\n"
                f"  ├─ 1. WAVE CLEARANCE:        True h: {h_m:.3f}m | Est h_hat: {h_est:.3f}m (Mode Target: {m['h_target_mode'].mean().item():.3f}m)\n"
                f"  ├─ 2. HYDROFOIL SENSORS:     Immersion: {d_sub_cm:.1f}cm | Foil Lift: {m['f_hydro_lift'].mean().item():.1f}N | Rotors: {f_rot_mean:.1f}N (Gimbal: {mean_gimbal_deg:.1f}°) | Pusher: {f_push_mean:.1f}N | [{flow_status}]\n"
                f"  ├─ 3. SUCTION MEMS TAP:      Lift Deficit Loss: {c_def_scalar:.2f} | Contraction Violation: {loss_viol_all.item():.4f}\n"
                f"  ├─ 4. SWARM DISTRIBUTION:    Active Swarm Envs: {num_envs:,} | Rollout Horizon: {rollout_steps} (TBTT={tbptt_steps})\n"
                f"  ├─ 5. LOSS DECOMPOSITION:    L_h: {c_h_scalar:.2f} | L_ceil: {c_ceil_scalar:.2f} | L_def: {c_def_scalar:.2f} | L_drift: {c_drift_scalar:.2f} | L_rate: {c_omega_scalar:.2f}\n"
                f"  ├─ 6. RIEMANNIAN SPECTRUM5D: λ_max(R5): {max_lambda_R:+.3f} [{cert_status}] | λ_min(R5): {min_lambda_R:+.3f} | λ_min(M5): {min_lambda_M:.3f} | κ(M5): {cond_M_tensor.item():.1f}\n"
                f"  ├─ 7. ATTITUDE & SAFETY:     Rate ||ω||: {safe_norm(m['omega'], dim=-1).mean().item()*57.3:.1f}°/s | TiltErr: {safe_norm(m['e_R'], dim=-1).mean().item()*57.3:.2f}° | CBF Shield: {qp_override_scalar:.3f}m/s² | Slack: {m['max_slack'].mean().item():.4f}\n"
                f"  ├─ 8. ELECTRICAL POWERTRAIN: Power: {p_elec_scalar:.1f}W | V_batt: {v_batt_scalar:.2f}V | Current: {i_batt_scalar:.1f}A (SoC: {soc_scalar:.1f}%)\n"
                f"  ├─ 9. EVOLVED CAD MORPHOLOGY:Span: {span_val:.2f}m (AR: {ar_val:.1f}) | Foil: {f_span_val:.2f}m (Strut: {strut_val:.2f}m) | Mass: {mass_val:.2f}kg | Anhedral: {anhed_val:+.1f}°\n"
                f"  └─ 10. OPTIMIZATION HEALTH:  Actor Grad: {grad_norm_actor:.3f} | NCM Grad: {grad_norm_ncm:.3f} | Morph Grad: {grad_norm_morph:.3f} | LRs: [{lr_act:.1e}, {lr_ncm:.1e}]"
            )

    print("\n" + "=" * 114)
    print(">> MULTI-STAGE END-TO-END NN TRAINING COMPLETE")
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
    dummy_input = torch.randn(1, NN_STATE_DIM, device=DEVICE)
    try:
        torch.onnx.export(
            actor_net,
            dummy_input,
            filename,
            input_names=["actor_obs_26state"],
            output_names=["actuators_16ch"],
            dynamic_axes={"actor_obs_26state": {0: "batch"}, "actuators_16ch": {0: "batch"}},
            opset_version=14,
        )
        print(f">> [ONNX DEPLOYMENT] 26-State -> 16-Actuator NN Pilot compiled successfully to: {filename}")
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
              f"(the full control step incl. EKF is much larger; see cockpit step timer)\n")
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

    pygame.display.set_caption("Aeromarine Autonomous Flight Engine (Pure 16-Actuator NN Control)")
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
    cad_f_span = m_cad["foil_span"].item()
    cad_strut = m_cad["strut_length"].item()
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
    print(" COCKPIT ACTIVE (Pure 16-Actuator NN Control | simulation runs at dt = 5 ms per frame)")
    print("   Click HUD mode buttons or press [1]-[6] (top row / numpad) | [A] -> AUTONOMOUS SORTIE MISSION")
    print("=" * 114 + "\n")

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
            flow_str = "VENTILATED" if m["is_ventilated"] else "PRIME (Attached)"
            radar_status = "LOCKED" if m["radar_locked"] else ("SPRAY DROPOUT" if m["radar_dropout"] else "SEARCHING")
            multipath_str = "REJECTED (Masked)" if m["multipath_rejected"] else "CLEAN"
            m_vals = m["motors"][0].cpu().numpy()
            m_disp = f"M0:{m_vals[0]:.2f}N | M1:{m_vals[1]:.2f}N | M2:{m_vals[2]:.2f}N | M3:{m_vals[3]:.2f}N | Push:{pusher_val:.2f}N"
            g_disp = f"G0:{g_vals[0]:.1f}° | G1:{g_vals[1]:.1f}° | G2:{g_vals[2]:.1f}° | G3:{g_vals[3]:.1f}°"
            d_surf = m["hydro_deflections"]

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

            b_om = m["bias_omega_deg"]
            cav_str = f"{m['cavitation_sigma']:.2f} [SAFE]" if m['cavitation_sigma'] > SIGMA_CAV_CRIT else f"{m['cavitation_sigma']:.2f} [CAVITATING!]"
            budget_str = "" if ema_latency_us < 5000.0 else " [OVER 5 ms BUDGET]"

            print(
                f"\n[T={sim_t:05.2f}s | Mode: {m['mode_name']:<8} | Mission: {phase_str} | Status: {m['status_banner']} | Step: {ema_latency_us:05.1f}µs{budget_str}] " + "─" * 15 + "\n"
                f"  ├─ 1. CORRIDOR & TRACKING:   Airspeed Vx: {m['vel'][0,0].item():.2f}m/s | Y-Dev: {m['pos'][0,1].item():+.2f}m (Vy: {m['vel'][0,1].item():+.2f}m/s) | Sideslip β: {m['beta_sideslip_deg']:+.1f}° | Yaw Err: {m['psi_err_deg']:+.1f}°\n"
                f"  ├─ 2. ACTIVE HYDRO-SURFACES: Elevon L: {d_surf[0]:+.1f}° | Elevon R: {d_surf[1]:+.1f}° | Rudder: {d_surf[2]:+.1f}° (Hydro Moment: {m['strut_moment_nm'][0,0].item():.2f}N·m)\n"
                f"  ├─ 3. ES-EKF:                True h: {m['h_wing'][0,0].item():.3f}m | Est h_hat: {clearance_val:.3f}m (σ_z: ±{m['sigma_z_cm']:.1f}cm) | Radar: {radar_status} | Multipath: {multipath_str} | Gyro Bias: ({b_om[0]:+.2f}°, {b_om[1]:+.2f}°, {b_om[2]:+.2f}°)/s\n"
                f"  ├─ 4. SEA DISPERSION:        Fwd Raycast: {m['eta_forward_meas'][0,0].item():+.3f}m | Nadir η̂: {m['eta_nadir_est'][0,0].item():+.3f}m (true {m['eta'][0,0].item():+.3f}m) | Dominant λ: {m['wavelength_m']:.1f}m (cp: {m['celerity_mps']:.1f}m/s) | Preview (1.25s): {m['eta_preview'][0,0].item():+.3f}m\n"
                f"  ├─ 5. FLUID TRANSDUCERS:     Immersion: {m['immersion_depth'][0,0].item()*100:.1f}cm | [{flow_str}]\n"
                f"  ├─ 6. AEROMARINE DYNAMICS:   Foil Lift: {lift_val:04.1f}N | Wing Lift: {m['f_wing_lift'][0,0].item():04.1f}N (PAR Cushion: {m['f_par_z']:.1f}N) | Hull: {m['f_planing_z']:04.1f}N | Rotors: {rotors_val:04.1f}N | Pusher: {pusher_val:04.1f}N\n"
                f"  ├─ 7. SCULLING & SOARING:    Caudal Thrust: {m['f_scull_thrust']:.2f}N (St: {m['strouhal_st']:.2f}) | Wave Surge: {m['f_soar_surge']:+.2f}N | Cavitation σ: {cav_str}\n"
                f"  ├─ 8. NN 16-ACTUATOR OUTPUT: [{m_disp}] | Tilts: [{g_disp}] | [{m['sat_str']}]\n"
                f"  ├─ 9. BATTERY & POWERTRAIN:  Power: {m['p_elec'][0,0].item():04.1f}W ({m['power_saved_pct'][0,0].item():+.1f}% SAVED) | V_batt: {m['v_batt'][0,0].item():.2f}V (Sag: {m['v_sag']:.2f}V) | Temp: {m['cell_temp_c']:.1f}°C\n"
                f"  └─ 10. SAFETY & DISTURBANCE: CBF Shield: {m['qp_override'][0,0].item():.3f}m/s² | Barrier: {m['barrier_name']} | Ceil Margin: {m['margin_ceil_cm']:+.1f}cm | Floor: {m['margin_floor_cm']:+.1f}cm"
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

        stats = [
            (f"PURE 16-ACTUATOR NN FLIGHT CONTROL | MODE: {cur_mode_name} [{mission_tag}]", (0, 240, 255)),
            (f"CONTROL STEP (EMA): {ema_latency_us:.1f} µs (Budget: 5000 µs @ 200 Hz)", (0, 255, 120) if ema_latency_us < 5000.0 else (255, 90, 90)),
            (f"NN HYDRO-SURFACES: Elevon L: {d_surf[0]:+.1f}° | Elevon R: {d_surf[1]:+.1f}° | Rudder: {d_surf[2]:+.1f}°", (255, 255, 100)),
            (f"TOTAL WATER LIFT GENERATED: {lift_val:.1f} N | ROTOR POWER SAVINGS: {m['power_saved_pct'][0,0].item():+.1f}%", (0, 255, 200)),
            (f"NN OMNI GIMBAL TILTS (0-15°): [{g_str}]", (120, 255, 180)),
            (f"NAVIGATION EKF: Clearance: {clearance_val:.3f}m | Airspeed: {m['vel'][0,0].item():.2f} m/s | CBF Shield: {m['qp_override'][0,0].item():.3f} m/s²", (100, 255, 100)),
            (f"NN PROPULSION ALLOCATION: [{m_str}]", (255, 210, 40)),
            (f"BATTERY DYNAMICS (6S): Voltage: {m['v_batt'][0,0].item():.2f}V | Current: {m['i_batt'][0,0].item():.1f}A | Temp: {m['cell_temp_c']:.1f}°C", (200, 220, 255)),
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
            y_lft = c2_rect.bottom - (hist_lift[idx] / 24.0) * c2_rect.height
            y_rot = c2_rect.bottom - (hist_rotors[idx] / 24.0) * c2_rect.height
            pts_lft.append((x_pos, np.clip(y_lft, c2_rect.top, c2_rect.bottom)))
            pts_rot.append((x_pos, np.clip(y_rot, c2_rect.top, c2_rect.bottom)))

        if len(pts_lft) > 1: pygame.draw.lines(screen, (0, 240, 220), False, pts_lft, 2)
        if len(pts_rot) > 1: pygame.draw.lines(screen, (255, 180, 40), False, pts_rot, 1)
        screen.blit(font_chart.render("Water Lift (Cyan) vs Rotors (Orange) [N]", True, (200, 220, 240)), (c2_rect.x + 6, c2_rect.y + 4))

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
    parser = argparse.ArgumentParser(description="Hydrofoil autonomy engine (pure 16-actuator NN control)")
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