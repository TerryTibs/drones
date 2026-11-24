"""
=============================================================================
MASTER AUTONOMOUS DRONE SYSTEM CORE (V4.4 - BASE STATION & WINCH RESTORED)
File: drone_autonomous_system_core_v4.py
=============================================================================
- Single Source of Truth Winch System: Starts Stowed at 8 cm against Fuselage
- Visible 2.0 kg Cargo Box Clamped to Belly -> Lowers on Wire -> Unlatches
- Bare Hook Reels Back UP to Belly -> Reloads -> Dispatches Next Target
- Illuminated Base Station Helipad [H] at [-3.5, -3.0] with Radio Mast & Depot
- Frameworks 1-5 Coupled: BEMT, IGE/ICE, CBF-QP Safety & Cold-Atom Quantum Sensing
=============================================================================
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

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
MASS_KG = 1.65                       # Coaxial X8 delivery airframe
GRAVITY_G = np.array([0.0, 0.0, -9.80665], dtype=np.float32)
G_ACCEL = 9.80665
ARM_LENGTH_M = 0.26
ROTOR_RADIUS_M = 0.127
INERTIA_MATRIX = np.diag([0.022, 0.022, 0.038]).astype(np.float32)
INV_INERTIA = np.linalg.inv(INERTIA_MATRIX)

C_T_CLEAN = 1.25e-5
C_Q_CLEAN = 2.40e-7
COAXIAL_EFFICIENCY = 0.88            # Bottom rotor efficiency
MOTOR_TAU_S = 0.030
MAX_RPM = 8800.0
MAX_OMEGA = (MAX_RPM * 2.0 * math.pi) / 60.0
HOVER_THRUST = MASS_KG * G_ACCEL
HOVER_OMEGA = math.sqrt((HOVER_THRUST / 8.0) / (C_T_CLEAN * 0.94))

# 1. BEMT Aerodynamics (Framework 1)
NUM_BLADE_ELEMENTS = 8
PRANDTL_TIP_FACTOR = 0.92
COAXIAL_DOWNWASH_INGESTION = 0.82

# 2. Ground & Ceiling Boundary Conditions (Framework 2)
IGE_HEIGHT_CUTOFF_M = 1.8 * ROTOR_RADIUS_M * 2.0
CEILING_HEIGHT_M = 14.0

# 3. Control Barrier Function Constants (Framework 4)
CBF_ALPHA_OBSTACLE = 1.8
CBF_ALPHA_CEILING = 2.2
CBF_SAFE_RADIUS_M = 0.65

# 4. Cold-Atom Quantum Sensing Constants (Framework 5)
K_EFF_WAVEVECTOR = 1.61e7            # Rubidium-87 Raman wavevector (m^-1)
RAMAN_PULSE_SEP_T = 0.015            # 15 ms pulse interrogation time

# Atmospheric & Sensor Defaults
P_0 = 1013.25
T_0 = 288.15
L_LAPSE = 0.0065
R_SPEC_AIR = 287.058
SPEED_OF_SOUND = 343.2
EMF_WARNING_THRESHOLD_UT = 18.0
POWERLINE_CURRENT_AMPS = 750.0
CHI2_SPOOF_THRESHOLD = 16.27

# 6S Battery ECM Constants
NUM_CELLS = 6
PACK_CAPACITY_AH = 4.5
PACK_ENERGY_WH = NUM_CELLS * 3.7 * PACK_CAPACITY_AH
CELL_R0_NOMINAL = 0.0085
CELL_R1 = 0.0065
CELL_C1 = 2800.0
ARRHENIUS_EA = 28000.0
GAS_CONST_R = 8.31446
T_REF_K = 298.15
RESERVE_SOC_FLOOR = 0.18

# Winch & Delivery Constants
PAYLOAD_WEIGHT_N = 2.0 * G_ACCEL
SNAG_THRESHOLD_N = 38.0
GUILLOTINE_THRESHOLD_N = 48.0
DROPZONE_TOLERANCE_M = 1.8

HOVER_LOCK_RADIUS_M = 0.45
HOVER_LOCK_SPEED_MAX_MS = 0.25
HOVER_SETTLE_DURATION_S = 0.75

OBSERVATION_DIM = 32
NUM_EXECUTIVE_STATES = 6
MODEL_WEIGHTS_FILE = "drone_v4_all_physics_brain.pth"

ACTION_NAMES = [
    "TRANSIT",             # 0: Cruising to dropzone
    "HOVER_LOCK",          # 1: Settling over target basket
    "WINCH_LOWER",         # 2: Motorized deployment of tether line
    "TOUCHDOWN_RELEASE",   # 3: Ground contact sensed, package unlatched
    "WINCH_RETRACT",       # 4: Empty hook reeling back to airframe belly
    "AUTO_DISPATCH"        # 5: Hook stowed, selecting next delivery target
]

# ==========================================
# 1. MATHEMATICAL UTILITIES & LIE ALGEBRA
# ==========================================
def skew_symmetric(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float32)

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
    r31 = R[2, 0]
    pitch = -math.asin(max(-1.0, min(1.0, r31)))
    if abs(math.cos(pitch)) > 1e-4:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

# ==========================================
# 2. CONTINUOUS 8-ROTOR BEMT COAXIAL AIRFRAME
# ==========================================
class BEMTCoaxialAirframe8DOF:
    def __init__(self, init_pos):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.R = np.eye(3, dtype=np.float32)
        self.omega = np.zeros(3, dtype=np.float32)
        self.motor_omegas = np.full(8, HOVER_OMEGA, dtype=np.float32)
        self.motor_health = np.ones(8, dtype=np.float32)
        self.accel_linear = np.zeros(3, dtype=np.float32)
        self.parachute_deployed = False

    def step(self, motor_cmds, current_wind, rho_air, dt=0.016):
        if self.parachute_deployed:
            self.motor_omegas.fill(0.0)
            self.v[0] *= 0.90
            self.v[1] *= 0.90
            self.v[2] += (-4.1 - self.v[2]) * 3.0 * dt
            self.p += (self.v + current_wind * 0.4) * dt
            if self.p[2] <= 0.0:
                self.p[2] = 0.0
                self.v.fill(0.0)
            return 0.0, np.zeros(3), 0.0, 1.0

        effective_cmds = np.clip(motor_cmds, 0.0, MAX_OMEGA) * self.motor_health
        self.motor_omegas += ((effective_cmds - self.motor_omegas) / MOTOR_TAU_S) * dt
        self.motor_omegas *= self.motor_health

        # Ground & Ceiling Effect (Method of Images)
        z_h = max(0.12, float(self.p[2]))
        ige_factor = 1.0 / max(0.35, (1.0 - (ROTOR_RADIUS_M / (4.0 * z_h)) ** 2))
        ige_factor = float(np.clip(ige_factor, 1.0, 1.35))
        d_ceil = max(0.3, CEILING_HEIGHT_M - z_h)
        ice_suction = float(1.0 + (ROTOR_RADIUS_M / (4.0 * d_ceil)) ** 2)

        # BEMT Integration Across Radial Elements
        T = np.zeros(8, dtype=np.float32)
        Q = np.zeros(8, dtype=np.float32)
        r_steps = np.linspace(0.15 * ROTOR_RADIUS_M, ROTOR_RADIUS_M, NUM_BLADE_ELEMENTS)
        dr = r_steps[1] - r_steps[0]

        for i in range(8):
            if self.motor_health[i] == 0.0: continue
            omega_i = self.motor_omegas[i]
            is_bottom = (i % 2 == 1)

            top_downwash = 0.0
            if is_bottom and self.motor_health[i - 1] > 0.0:
                top_downwash = 0.045 * (self.motor_omegas[i - 1] * ROTOR_RADIUS_M) * COAXIAL_DOWNWASH_INGESTION

            t_blade, q_blade = 0.0, 0.0
            for r in r_steps:
                v_axial = -self.v[2] + top_downwash
                v_tan = omega_i * r
                v_eff_sq = v_axial**2 + v_tan**2
                phi = math.atan2(max(0.1, v_axial), max(0.1, v_tan))

                theta_r = math.radians(14.0 - 8.0 * (r / ROTOR_RADIUS_M))
                alpha = theta_r - phi
                c_l = 2.0 * math.pi * alpha
                c_d = 0.012 + 0.04 * (alpha ** 2)

                exp_arg = -max(0.01, (ROTOR_RADIUS_M - r) / (2.0 * r * math.sin(max(0.01, phi))))
                acos_arg = min(1.0, max(-1.0, math.exp(exp_arg)))
                f_tip = max(0.2, min(1.0, (2.0 / math.pi) * math.acos(acos_arg)))

                dT = 0.5 * rho_air * v_eff_sq * (0.018) * (c_l * math.cos(phi) - c_d * math.sin(phi)) * f_tip * dr
                dQ = 0.5 * rho_air * v_eff_sq * (0.018) * (c_l * math.sin(phi) + c_d * math.cos(phi)) * f_tip * r * dr
                t_blade += dT
                q_blade += dQ

            T[i] = t_blade * 2.0 * ige_factor * ice_suction
            Q[i] = q_blade * 2.0

        f_thrust = float(np.sum(T))
        arm = ARM_LENGTH_M / math.sqrt(2.0)
        tau_x = arm * (- (T[0] + T[1]) + (T[2] + T[3]) + (T[4] + T[5]) - (T[6] + T[7]))
        tau_y = arm * (  (T[0] + T[1]) + (T[2] + T[3]) - (T[4] + T[5]) - (T[6] + T[7]))
        tau_z = -Q[0] + Q[1] + Q[2] - Q[3] - Q[4] + Q[5] + Q[6] - Q[7]
        torques = np.array([tau_x, tau_y, tau_z], dtype=np.float32)

        v_air = self.v - current_wind
        speed_air = float(np.linalg.norm(v_air))
        f_drag = -0.5 * rho_air * np.array([0.22, 0.22, 0.38], dtype=np.float32) * v_air * max(1.0, speed_air)
        f_body = np.array([0.0, 0.0, f_thrust], dtype=np.float32)
        f_gravity = MASS_KG * GRAVITY_G

        self.accel_linear = (self.R @ f_body + f_gravity + f_drag) / MASS_KG
        gyroscopic = np.cross(self.omega, INERTIA_MATRIX @ self.omega)
        accel_angular = INV_INERTIA @ (torques - gyroscopic)

        self.p += self.v * dt + 0.5 * self.accel_linear * (dt ** 2)
        self.v += self.accel_linear * dt
        if self.p[2] < 0.0:
            self.p[2] = 0.0
            self.v[2] = max(0.0, self.v[2])

        self.omega += accel_angular * dt
        d_theta = np.linalg.norm(self.omega) * dt
        if d_theta > 1e-6:
            axis = self.omega / np.linalg.norm(self.omega)
            K = skew_symmetric(axis)
            R_delta = np.eye(3) + math.sin(d_theta) * K + (1.0 - math.cos(d_theta)) * (K @ K)
            self.R = self.R @ R_delta

        U, _, Vt = np.linalg.svd(self.R)
        self.R = U @ Vt

        p_watts = float(np.sum(Q * self.motor_omegas)) / 0.85 + 24.0
        return f_thrust, torques, p_watts, ige_factor

# ==========================================
# 3. UNIFIED AUTONOMOUS WINCH & CARGO DELIVERY SYSTEM
# ==========================================
class AutonomousWinchDeliverySystem:
    """
    Single source of truth for tether payout, cargo clamping, touchdown detection,
    and automatic re-hoisting. Stowed tightly at 8 cm until commanded to lower.
    """
    def __init__(self):
        self.cable_len = 0.08               # Stowed tightly against belly clamp
        self.cable_vel = 0.0
        self.tension_n = PAYLOAD_WEIGHT_N
        self.has_payload = True             # Starts with package #1 loaded
        self.touchdown_timer = 0.0
        self.severed = False
        self.state = "STANDBY"              # STANDBY, LOWERING, TOUCHDOWN, RETRACTING, GUILLOTINE
        self.sway_offset = np.zeros(2, dtype=np.float32)
        self.delivery_completed = False
        self.snag_timer = 0.0

    def reload_new_cargo(self):
        """Reloads fresh cargo parcel from bay when hook docks at belly."""
        self.cable_len = 0.08
        self.cable_vel = 0.0
        self.has_payload = True
        self.state = "STANDBY"
        self.tension_n = PAYLOAD_WEIGHT_N
        self.touchdown_timer = 0.0
        self.delivery_completed = False

    def step(self, drone_z, is_settled, is_surface_safe, wire_hit, wind_2d, dt=0.016, force_snag=False):
        if self.severed:
            self.state = "GUILLOTINE"
            return self.cable_len, 0.0, np.zeros(2)

        # Pendular wind sway on suspended cable
        if self.cable_len > 0.12:
            target_sway = wind_2d * (self.cable_len * 0.06)
            self.sway_offset += (target_sway - self.sway_offset) * 3.5 * dt
        else:
            self.sway_offset.fill(0.0)

        # Safety interlock check
        lockout = (not is_surface_safe) or wire_hit or (not is_settled)

        # FSM State Transitions
        if self.state == "STANDBY":
            self.cable_len = 0.08
            self.cable_vel = 0.0
            self.tension_n = PAYLOAD_WEIGHT_N if self.has_payload else 1.2
            # Trigger lowering only when settled over safe dropzone with cargo loaded!
            if not lockout and self.has_payload:
                self.state = "LOWERING"
                print(">>> [DROPZONE CONFIRMED] LOWERING 2.0kg CARGO ON TETHER <<<")

        elif self.state == "LOWERING":
            if lockout:
                self.state = "RETRACTING"
            else:
                self.cable_vel = 1.25
                self.cable_len += self.cable_vel * dt
                self.tension_n = PAYLOAD_WEIGHT_N + float(np.random.normal(0, 0.25))

                # Touchdown check: Cargo parcel hits grass
                cargo_z = drone_z - self.cable_len
                if cargo_z <= 0.06:
                    self.state = "TOUCHDOWN"
                    self.touchdown_timer = 0.0
                    print(">>> [TOUCHDOWN] CARGO TOUCHDOWN ON GROUND (WEIGHT TRANSFERRED) <<<")

        elif self.state == "TOUCHDOWN":
            self.cable_vel = 0.0
            self.tension_n = 1.8 + float(np.random.normal(0, 0.1)) # Load cell tension drops!
            self.touchdown_timer += dt
            if self.touchdown_timer >= 1.0:
                # Mechanical latch opens: Package deposited on ground!
                self.has_payload = False
                self.state = "RETRACTING"
                print(">>> [LATCH OPEN] CARGO UNHOOKED! REELING BARE HOOK BACK UP <<<")

        elif self.state == "RETRACTING":
            self.cable_vel = -2.2 # High speed reel-in
            self.cable_len = max(0.08, self.cable_len + self.cable_vel * dt)
            self.tension_n = 1.2 + float(np.random.normal(0, 0.05)) # Empty hook drag

            # Hook fully docked at airframe belly!
            if self.cable_len <= 0.08:
                self.cable_len = 0.08
                self.cable_vel = 0.0
                self.state = "STANDBY"
                self.delivery_completed = True
                print(">>> [HOOK DOCKED] HOOK SECURED AT BELLY. AUTO-DISPATCH TRIGGERED <<<")

        if force_snag:
            self.tension_n = 58.0

        if self.tension_n >= GUILLOTINE_THRESHOLD_N:
            self.snag_timer += dt
            if self.snag_timer >= 0.25:
                self.severed = True
                self.state = "GUILLOTINE"
                print(">>> [SAFETY INTERLOCK] PYROTECHNIC GUILLOTINE FIRED: TETHER CUT <<<")
        else:
            self.snag_timer = max(0.0, self.snag_timer - dt)

        return self.cable_len, self.tension_n, self.sway_offset

# ==========================================
# 4. MATTER-WAVE QUANTUM ACCELEROMETER
# ==========================================
class MatterWaveColdAtomQuantumSensor:
    def __init__(self):
        self.k_eff = K_EFF_WAVEVECTOR
        self.T = RAMAN_PULSE_SEP_T
        self.phase_noise_std = 0.002

    def measure_phase_shift(self, true_accel_inertial):
        phase_shift = self.k_eff * true_accel_inertial * (self.T ** 2)
        phase_shift += np.random.normal(0, self.phase_noise_std, 3).astype(np.float32)
        inv_scale = 1.0 / (self.k_eff * (self.T ** 2))
        a_quantum = phase_shift * inv_scale
        return a_quantum, phase_shift

# ==========================================
# 5. CONTROL BARRIER FUNCTION (CBF-QP) FILTER
# ==========================================
class ControlBarrierFunctionQPFilter:
    def __init__(self):
        self.alpha_obs = CBF_ALPHA_OBSTACLE
        self.safe_radius = CBF_SAFE_RADIUS_M

    def enforce_safety(self, drone_p, drone_v, nominal_accel, obstacles):
        safe_accel = nominal_accel.copy()
        for obs in obstacles:
            rel_pos = drone_p - obs["pos"]
            dist_xy = math.hypot(rel_pos[0], rel_pos[1])
            h_obs = dist_xy - (obs["radius"] + self.safe_radius)
            if h_obs < 1.2:
                grad_h = np.array([rel_pos[0] / dist_xy, rel_pos[1] / dist_xy, 0.0], dtype=np.float32)
                l_fh = float(np.dot(grad_h, drone_v))
                b_cbf = -l_fh - self.alpha_obs * h_obs
                val = float(np.dot(grad_h, safe_accel))
                if val < b_cbf:
                    lambda_kkt = (b_cbf - val) / float(np.dot(grad_h, grad_h))
                    safe_accel += lambda_kkt * grad_h

        h_gnd = drone_p[2] - 0.20
        if h_gnd < 0.6:
            l_f_gnd = drone_v[2]
            b_gnd = -l_f_gnd - 2.5 * h_gnd
            if safe_accel[2] < b_gnd:
                safe_accel[2] = b_gnd

        return safe_accel

# ==========================================
# 6. DUAL-HEAD NEURAL POLICY ARCHITECTURE
# ==========================================
class MultiModalFlightExecutiveNet(nn.Module):
    def __init__(self, in_features=OBSERVATION_DIM):
        super().__init__()
        self.shared_backbone = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU()
        )
        self.skip_proj = nn.Linear(in_features, 128)

        self.guidance_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 4),
            nn.Tanh()
        )

        self.executive_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, NUM_EXECUTIVE_STATES)
        )

    def forward(self, x):
        latent = self.shared_backbone(x) + self.skip_proj(x)
        guidance = self.guidance_head(latent)
        logits = self.executive_head(latent)
        return guidance, logits

# ==========================================
# 7. OBSERVATION ENCODER
# ==========================================
class DroneObservationSynthesizer:
    def __init__(self):
        self.last_tension = 19.62

    def synthesize(self, p_est=None, v_est=None, R_est=None, omega_est=None, target_p=None,
                   v_tas=0.0, q_dyn=0.0, min_sonar=2.2, emf_rms=5.0, wind_est=None, adsb_t_cpa=50.0,
                   bms_soc=0.88, v_pack=22.8, cable_len=0.08, cable_vel=0.0, tension_n=19.62, has_payload=True,
                   chi2_val=1.8, is_spoofed=False, flight_time=0.0, total_plan_time=4.5, delivered_count=0, dt=0.016,
                   drone_p=None, drone_v=None, drone_R=None, drone_omega=None):
        p_in = p_est if p_est is not None else (drone_p if drone_p is not None else np.zeros(3, dtype=np.float32))
        v_in = v_est if v_est is not None else (drone_v if drone_v is not None else np.zeros(3, dtype=np.float32))
        R_in = R_est if R_est is not None else (drone_R if drone_R is not None else np.eye(3, dtype=np.float32))
        w_in = omega_est if omega_est is not None else (drone_omega if drone_omega is not None else np.zeros(3, dtype=np.float32))
        tgt_in = target_p if target_p is not None else np.zeros(3, dtype=np.float32)
        wind_in = wind_est if wind_est is not None else np.zeros(3, dtype=np.float32)

        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        delta_p = tgt_in - p_in
        obs[0:3] = np.clip(delta_p / 10.0, -1.0, 1.0)
        obs[3:6] = np.clip(v_in / 6.0, -1.0, 1.0)
        obs[6:9] = np.clip(R_in[:, 2], -1.0, 1.0)
        obs[9:12] = np.clip(w_in / 8.0, -1.0, 1.0)
        obs[12] = np.clip(v_tas / 15.0, 0.0, 1.0)
        obs[13] = np.clip(q_dyn / 100.0, 0.0, 1.0)
        obs[14] = np.clip(min_sonar / 8.0, 0.0, 1.0)
        obs[15] = np.clip(emf_rms / 40.0, 0.0, 1.0)
        obs[16:19] = np.clip(wind_in / 8.0, -1.0, 1.0)
        obs[19] = np.clip(adsb_t_cpa / 40.0, 0.0, 1.0)
        obs[20] = np.clip(bms_soc, 0.0, 1.0)
        obs[21] = np.clip((v_pack - 18.0) / (25.2 - 18.0), 0.0, 1.0)
        obs[22] = np.clip(cable_len / 10.0, 0.0, 1.0)
        obs[23] = np.clip(cable_vel / 3.0, -1.0, 1.0)
        d_tension = (tension_n - self.last_tension) / max(1e-4, dt)
        self.last_tension = tension_n
        obs[24] = np.clip(tension_n / 50.0, 0.0, 1.0)
        obs[25] = np.clip(d_tension / 25.0, -1.0, 1.0)
        obs[26] = 1.0 if has_payload else 0.0
        obs[27] = np.clip(chi2_val / 40.0, 0.0, 1.0)
        obs[28] = 1.0 if is_spoofed else 0.0
        progress = flight_time / max(0.1, total_plan_time)
        obs[29] = np.clip(progress, 0.0, 1.0)
        d_horiz = math.hypot(delta_p[0], delta_p[1])
        obs[30] = np.clip(d_horiz / 10.0, 0.0, 1.0)
        obs[31] = np.clip((delivered_count % 10) / 10.0, 0.0, 1.0)
        return obs

# ==========================================
# 8. SENSOR & SURFACE SUITES
# ==========================================
class SurfaceAndWireAssuranceSuite:
    def __init__(self):
        self.water_pond = {"center": np.array([2.5, 4.0, 0.0]), "radius": 2.2}
        self.deep_bushes = {"center": np.array([-2.5, 3.5, 0.0]), "radius": 1.8}
        self.overhead_wires = [
            {"p1": np.array([-5.0, 2.0, 3.2]), "p2": np.array([5.0, 2.0, 3.2]), "type": "UTILITY_LINE", "temp_c": 68.0, "dolp": 0.52}
        ]

    def audit_surface(self, pos_xy):
        px, py = pos_xy
        if math.hypot(px - self.water_pond["center"][0], py - self.water_pond["center"][1]) < self.water_pond["radius"]:
            return "LIQUID_WATER_HAZARD", 42.0, 0.25, False
        if math.hypot(px - self.deep_bushes["center"][0], py - self.deep_bushes["center"][1]) < self.deep_bushes["radius"]:
            return "POROUS_BUSH_CANOPY", 68.0, 1.85, False
        return "SOLID_TERRAIN_SAFE", 88.0, 0.35, True

    def scan_wires(self, drone_p, radius=1.6):
        for wire in self.overhead_wires:
            w1, w2 = wire["p1"], wire["p2"]
            if drone_p[2] > w1[2]:
                if min(w1[0], w2[0]) <= drone_p[0] <= max(w1[0], w2[0]):
                    if abs(drone_p[1] - w1[1]) < radius:
                        return True, drone_p[2] - w1[2], wire["type"], wire["dolp"]
        return False, 50.0, "NONE", 0.05

    def generate_verified_solid_dropzone(self, current_pos, min_distance=3.0):
        for _ in range(60):
            cand = np.array([random.uniform(-4.0, 4.0), random.uniform(-2.5, 6.0), 0.0], dtype=np.float32)
            if np.linalg.norm(cand[:2] - current_pos[:2]) < min_distance: continue
            _, _, _, is_safe = self.audit_surface(cand[:2])
            if is_safe: return cand
        return np.array([0.0, -1.0, 0.0], dtype=np.float32)

class FullPhysicsSensorSuite:
    def __init__(self):
        self.wire_y = 2.0
        self.wire_z = 3.2
        self.intruder_pos = np.array([450.0, 320.0, 85.0], dtype=np.float32)
        self.intruder_vel = np.array([-26.0, -18.0, 0.0], dtype=np.float32)

    def step_environment(self, dt):
        self.intruder_pos += self.intruder_vel * dt
        if self.intruder_pos[0] < -450.0:
            self.intruder_pos = np.array([500.0, 350.0, 85.0], dtype=np.float32)

    def poll_all_physics_sensors(self, drone_p, drone_v, current_wind):
        z_true = max(0.0, float(drone_p[2]))
        t_air = T_0 - L_LAPSE * z_true
        p_static_hpa = P_0 * ((1.0 - (L_LAPSE * z_true) / T_0) ** 5.25588)
        rho_air = (p_static_hpa * 100.0) / (R_SPEC_AIR * t_air)
        z_baro = 44330.8 * (1.0 - ((p_static_hpa / P_0) ** 0.190284)) + float(np.random.normal(0, 0.04))

        v_air = drone_v - current_wind
        v_tas = float(np.linalg.norm(v_air))
        q_dyn = 0.5 * rho_air * (v_tas ** 2)

        d_wire = math.hypot(drone_p[1] - self.wire_y, drone_p[2] - self.wire_z)
        emf_rms = float(150.0 / max(0.25, d_wire))

        dp = self.intruder_pos - drone_p
        dv = self.intruder_vel - drone_v
        dist_intruder = float(np.linalg.norm(dp))
        v_rel_sq = float(np.dot(dv, dv))
        t_cpa = -float(np.dot(dp, dv)) / max(1e-4, v_rel_sq) if v_rel_sq > 1e-4 else 0.0
        d_cpa = float(np.linalg.norm(dp + dv * t_cpa)) if t_cpa > 0 else dist_intruder
        adsb_t_cpa_clamped = max(0.0, t_cpa) if (t_cpa > 0 and d_cpa < 200.0) else 50.0

        min_sonar = max(0.05, min(8.0, z_true))
        return {
            "z_baro": z_baro, "rho_air": rho_air, "v_tas": v_tas, "q_dyn": q_dyn,
            "emf_rms": emf_rms, "dist_intruder": dist_intruder, "t_cpa": adsb_t_cpa_clamped,
            "min_sonar": min_sonar, "t_air": t_air
        }

class Battery1RCTheveninECM:
    def __init__(self):
        self.soc = 0.92
        self.temp_c = 18.0
        self.v1 = np.zeros(NUM_CELLS, dtype=np.float32)
        self.capacity_coulombs = PACK_CAPACITY_AH * 3600.0
        self.v_pack = 23.5

    def step(self, p_elec_watts, dt=0.016):
        i_draw = p_elec_watts / max(18.0, self.v_pack)
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
        self.temp_c += ((p_joule - 0.35 * (self.temp_c - 18.0)) / 350.0) * dt
        return self.soc, self.v_pack, i_draw

# ==========================================
# 9. 15-STATE ES-EKF ON SO(3)
# ==========================================
class ErrorStateEKF15:
    def __init__(self, init_pos):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.b_a = np.zeros(3, dtype=np.float32)
        self.b_g = np.zeros(3, dtype=np.float32)
        self.P = np.diag([0.15]*3 + [0.1]*3 + [0.02]*3 + [0.05]*3 + [0.01]*3).astype(np.float32)
        self.Q_d = np.diag([1e-4]*3 + [1e-3]*3 + [1e-4]*3 + [1e-5]*3 + [1e-6]*3).astype(np.float32)

    def predict(self, accel_raw, gyro_raw, dt=0.016):
        a_unbiased = accel_raw - self.b_a
        w_unbiased = gyro_raw - self.b_g
        R = quat_to_rot(self.q)
        f_inertial = R @ a_unbiased + GRAVITY_G

        self.p += self.v * dt + 0.5 * f_inertial * (dt ** 2)
        self.v += f_inertial * dt

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

        self.P = F_x @ self.P @ F_x.T + self.Q_d
        self.P = 0.5 * (self.P + self.P.T)

    def update_measurement(self, H, innovation, R_cov):
        S = H @ self.P @ H.T + R_cov
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        delta_x = (K @ innovation).flatten()
        delta_x = np.clip(delta_x, -3.5, 3.5)
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

# ==========================================
# 10. CBF-QP FILTER & COAXIAL ACTUATOR MIXER
# ==========================================
class RateDampedActuatorMixerWithCBF:
    def __init__(self):
        self.k_R = 1.95
        self.k_w = 0.38
        self.e_p_integral = np.zeros(3, dtype=np.float32)
        self.wind_force_hat = np.zeros(3, dtype=np.float32)
        self.cbf_filter = ControlBarrierFunctionQPFilter()

        arm = ARM_LENGTH_M / math.sqrt(2.0)
        cq_ct = C_Q_CLEAN / C_T_CLEAN
        self.B_nominal = np.array([
            [ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
            [-arm, -arm,  arm,  arm,  arm,  arm, -arm, -arm],
            [ arm,  arm,  arm,  arm, -arm, -arm, -arm, -arm],
            [-cq_ct, cq_ct, cq_ct, -cq_ct, -cq_ct, cq_ct, cq_ct, -cq_ct]
        ], dtype=np.float32)

    def mix(self, drone_health, p_est, v_est, R_est, omega_est, p_target, neural_guidance, token_idx, estimated_wind, obstacles, dt=0.016):
        delta_p = p_target - p_est
        d_horiz = float(np.linalg.norm(delta_p[:2]))

        if token_idx == 0:  # TRANSIT
            v_cruise = 2.8
            dir_norm = delta_p / max(1e-4, float(np.linalg.norm(delta_p)))
            v_des = dir_norm * v_cruise
            if d_horiz < 2.2:
                v_des[:2] *= (d_horiz / 2.2)
        else:
            v_des = np.zeros(3, dtype=np.float32)

        e_p = p_est - p_target
        e_v = v_est - v_des

        self.e_p_integral += e_p * dt
        self.e_p_integral = np.clip(self.e_p_integral, -1.5, 1.5)

        # Disturbance Observer for Wind Drag
        f_drag_est = 0.22 * estimated_wind * float(np.linalg.norm(estimated_wind)) * 0.35
        self.wind_force_hat += (f_drag_est - self.wind_force_hat) * 4.0 * dt

        f_neural_offset = neural_guidance[:3] * 6.5

        a_nominal = (-np.array([6.5, 6.5, 11.0]) * e_p - np.array([4.4, 4.4, 6.8]) * e_v - np.array([1.2, 1.2, 2.5]) * self.e_p_integral + f_neural_offset - self.wind_force_hat) / MASS_KG

        a_safe = self.cbf_filter.enforce_safety(p_est, v_est, a_nominal, obstacles)
        F_d = a_safe * MASS_KG + MASS_KG * (-GRAVITY_G)

        max_tilt_rad = math.radians(25.0)
        f_horiz = math.hypot(F_d[0], F_d[1])
        f_vert = max(1.0, F_d[2])
        if f_horiz / f_vert > math.tan(max_tilt_rad):
            scale_h = (f_vert * math.tan(max_tilt_rad)) / f_horiz
            F_d[0] *= scale_h
            F_d[1] *= scale_h

        b3 = R_est[:, 2]
        f_thrust = float(np.dot(F_d, b3))
        f_thrust = max(0.1, min(38.0, f_thrust))

        b3_d = F_d / (np.linalg.norm(F_d) + 1e-6)
        b1_c = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b2_d = np.cross(b3_d, b1_c); b2_d /= (np.linalg.norm(b2_d) + 1e-6)
        b1_d = np.cross(b2_d, b3_d)
        R_d = np.column_stack([b1_d, b2_d, b3_d]).astype(np.float32)

        R_err_mat = 0.5 * (R_d.T @ R_est - R_est.T @ R_d)
        e_R = vee_map(R_err_mat)
        tau = -self.k_R * e_R - self.k_w * omega_est + np.cross(omega_est, INERTIA_MATRIX @ omega_est)
        desired_wrench = np.array([f_thrust, tau[0], tau[1], tau[2]], dtype=np.float32)

        W_health = np.diag(drone_health).astype(np.float32)
        B_active = self.B_nominal @ W_health
        try:
            B_pinv = np.linalg.pinv(B_active)
            T_cmds = B_pinv @ desired_wrench
        except np.linalg.LinAlgError:
            T_cmds = np.full(8, f_thrust / 8.0, dtype=np.float32)

        T_cmds = np.clip(T_cmds * drone_health, 0.0, 9.8)

        omega_cmds = np.zeros(8, dtype=np.float32)
        for i in range(8):
            if drone_health[i] > 0.0:
                scale_c = C_T_CLEAN if (i % 2 == 0) else (C_T_CLEAN * COAXIAL_EFFICIENCY)
                omega_cmds[i] = math.sqrt(T_cmds[i] / scale_c)

        return omega_cmds, f_thrust, a_safe

# ==========================================
# 11. MULTI-HAZARD PARALLEL HARVESTER
# ==========================================
def _worker_harvest_v4(num_steps):
    np.random.seed()
    random.seed()
    synth = DroneObservationSynthesizer()

    observations, teacher_guidance, teacher_fsm = [], [], []

    while len(observations) < num_steps:
        start_p = np.array([random.uniform(-4, 4), random.uniform(-4, 4), random.uniform(1.2, 3)], dtype=np.float32)
        target_gnd = np.array([random.uniform(-4, 4), random.uniform(-2, 6), 0.0], dtype=np.float32)
        home_pos = np.array([-3.5, -3.0, 1.5], dtype=np.float32)
        drone_p = start_p.copy()
        drone_v = np.zeros(3, dtype=np.float32)
        hover_dest = np.array([target_gnd[0], target_gnd[1], 2.2], dtype=np.float32)

        step_count = 0
        flight_time = 0.0
        dt = 0.016

        episode_wind = np.array([random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0), 0.0], dtype=np.float32)
        episode_soc = random.uniform(0.15, 0.98)
        episode_v_pack = 18.0 + episode_soc * (25.2 - 18.0)
        is_spoofed_episode = (random.random() < 0.25)
        chi2_episode = float(random.uniform(22.0, 55.0)) if is_spoofed_episode else float(random.uniform(0.5, 3.5))

        has_wire_hazard = (random.random() < 0.20)
        emf_episode = float(random.uniform(22.0, 45.0)) if has_wire_hazard else float(random.uniform(2.0, 8.0))

        has_adsb_threat = (random.random() < 0.15)
        adsb_t_cpa_episode = float(random.uniform(5.0, 12.0)) if has_adsb_threat else float(random.uniform(35.0, 60.0))

        while step_count < 400 and len(observations) < num_steps:
            flight_time += dt
            d_horiz = float(np.linalg.norm(drone_p[:2] - target_gnd[:2]))
            v_des = np.zeros(3, dtype=np.float32)

            if has_adsb_threat:
                fsm_token = 1
                v_des[2] = -2.2
            elif has_wire_hazard:
                fsm_token = 1
                v_des[1] = 2.0
            elif episode_soc <= 0.20:
                fsm_token = 0
                delta_home = home_pos - drone_p
                v_des = (delta_home / max(1e-4, float(np.linalg.norm(delta_home)))) * 2.8
            elif d_horiz > 0.45:
                fsm_token = 0
                delta = hover_dest - drone_p
                v_des = (delta / max(1e-4, float(np.linalg.norm(delta)))) * 2.8
            else:
                fsm_token = 2

            e_p = drone_p - hover_dest
            e_v = drone_v - v_des
            f_drag_counter = -0.22 * episode_wind * float(np.linalg.norm(episode_wind)) * 0.35
            F_d = -np.array([6.5, 6.5, 11.0]) * e_p - np.array([4.4, 4.4, 6.8]) * e_v + MASS_KG * (-GRAVITY_G) + f_drag_counter

            guidance_target = np.clip(F_d / (MASS_KG * 15.0), -1.0, 1.0).astype(np.float32)
            guidance_vec = np.array([guidance_target[0], guidance_target[1], guidance_target[2], 0.0], dtype=np.float32)

            obs_t = synth.synthesize(
                p_est=drone_p, v_est=drone_v, R_est=np.eye(3), omega_est=np.zeros(3),
                target_p=hover_dest, v_tas=float(np.linalg.norm(drone_v - episode_wind)), q_dyn=12.0,
                min_sonar=max(0.1, min(8.0, drone_p[2])), emf_rms=emf_episode, wind_est=episode_wind,
                adsb_t_cpa=adsb_t_cpa_episode, bms_soc=episode_soc, v_pack=episode_v_pack, cable_len=0.08,
                cable_vel=0.0, tension_n=PAYLOAD_WEIGHT_N, has_payload=True,
                chi2_val=chi2_episode, is_spoofed=is_spoofed_episode, flight_time=flight_time,
                total_plan_time=4.5, delivered_count=1, dt=dt
            )

            observations.append(obs_t)
            teacher_guidance.append(guidance_vec)
            teacher_fsm.append(fsm_token)

            drone_p += drone_v * dt
            drone_v += (F_d / MASS_KG) * dt
            step_count += 1

    return np.array(observations, dtype=np.float32), np.array(teacher_guidance, dtype=np.float32), np.array(teacher_fsm, dtype=np.int64)

def harvest_and_train_v4():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = 80000
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per = num_samples // num_workers

    print("\n" + "="*70)
    print("HARVESTING 80,000 MULTI-PHYSICS TRAJECTORIES (BEMT + IGE + CBF + QUANTUM)")
    print(f"Parallel Workers: {num_workers} processes | Device: {device}")
    print("="*70)
    t0 = time.time()

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_harvest_v4, [samples_per] * num_workers)

    X_all = np.concatenate([r[0] for r in results], axis=0)[:num_samples]
    y_g_all = np.concatenate([r[1] for r in results], axis=0)[:num_samples]
    y_f_all = np.concatenate([r[2] for r in results], axis=0)[:num_samples]

    print(f"Dataset compiled in {time.time() - t0:.2f}s! ({num_samples:,} samples)")

    print("\n" + "="*70)
    print("TRAINING NEURAL AUTOPILOT ON ADVANCED PHYSICS (8 EPOCHS)")
    print("="*70)

    num_train = int(len(X_all) * 0.9)
    X_train = torch.tensor(X_all[:num_train], device=device)
    y_g_train = torch.tensor(y_g_all[:num_train], device=device)
    y_f_train = torch.tensor(y_f_all[:num_train], device=device)

    X_val = torch.tensor(X_all[num_train:], device=device)
    y_g_val = torch.tensor(y_g_all[num_train:], device=device)
    y_f_val = torch.tensor(y_f_all[num_train:], device=device)

    model = MultiModalFlightExecutiveNet().to(device)
    criterion_g = nn.MSELoss()
    criterion_f = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)

    batch_size = 2048
    epochs = 8
    num_batches = (num_train + batch_size - 1) // batch_size
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-3, steps_per_epoch=num_batches, epochs=epochs)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(num_train, device=device)
        total_loss = 0.0

        for b in range(0, num_train, batch_size):
            idx = perm[b : b + batch_size]
            bx, by_g, by_f = X_train[idx], y_g_train[idx], y_f_train[idx]

            optimizer.zero_grad(set_to_none=True)
            p_g, p_f = model(bx)
            loss = criterion_g(p_g, by_g) + 0.40 * criterion_f(p_f, by_f)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            v_g, v_f = model(X_val)
            val_acc = (v_f.argmax(dim=-1) == y_f_val).float().mean().item() * 100.0
            val_mse = criterion_g(v_g, y_g_val).item()

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] - Loss: {total_loss/num_batches:.4f} | Guidance MSE: {val_mse:.5f} | Decision Acc: {val_acc:.1f}%")

    torch.save(model.state_dict(), MODEL_WEIGHTS_FILE)
    print(f"\nModel compiled & saved to {MODEL_WEIGHTS_FILE} in {time.time() - t0:.2f}s!")
    return model, device

# ==========================================
# 12. 3D PROJECTION WITH DEFENSIVE SANITIZATION
# ==========================================
def project_3d(pt, cam, center, scale=46.0):
    yaw, pitch = cam
    x, y, z = pt
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return int(center[0]), int(center[1]), 10.0

    x1 = x * math.cos(yaw) - y * math.sin(yaw)
    y1 = x * math.sin(yaw) + y * math.cos(yaw)
    x2 = x1
    y2 = y1 * math.cos(pitch) - z * math.sin(pitch)
    z2 = y1 * math.sin(pitch) + z * math.cos(pitch)
    focal = max(2.5, 12.0 - y2)
    sx = int(np.clip(center[0] + (x2 / focal) * scale * 7.5, -2500, 2500))
    sy = int(np.clip(center[1] - (z2 / focal) * scale * 7.5, -2500, 2500))
    return sx, sy, y2

def screen_to_ground_raycast(mouse_pos, cam, center, scale=46.0):
    mx, my = mouse_pos
    yaw, pitch = cam
    dx = (mx - center[0]) / (scale * 7.5)
    dy = -(my - center[1]) / (scale * 7.5)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    denom = cp + dy * sp
    if abs(denom) < 1e-4: return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    y2 = (12.0 * dy * sp) / denom
    dist = max(2.5, 12.0 - y2)
    x2 = dx * dist
    z2 = 0.0
    y1 = (y2 * cp + z2 * sp)
    z1 = (-y2 * sp + z2 * cp)
    x = x2 * cy + y1 * sy
    y = -x2 * sy + y1 * cy
    return np.array([np.clip(x, -6.5, 6.5), np.clip(y, -5.0, 9.0), 0.0], dtype=np.float32)

# ==========================================
# 13. MAIN SIMULATOR & ALL-PHYSICS COCKPIT HUD
# ==========================================
def run_all_physics_simulator():
    if not os.path.exists(MODEL_WEIGHTS_FILE):
        model, device = harvest_and_train_v4()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiModalFlightExecutiveNet().to(device)
        try:
            model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE, map_location=device))
            print("\nLoaded verified Advanced Physics Neural Model from disk!")
        except Exception:
            print("Checkpoint mismatch. Re-training model...")
            model, device = harvest_and_train_v4()

    model.eval()

    pygame.init()
    VIEW_W, VIEW_H = 760, 740
    SIDEBAR_W = 440
    W, H = VIEW_W + SIDEBAR_W, VIEW_H

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Survival Core V4.4 — Base Station, Cargo Container & BEMT Physics")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("consolas", 17, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 12, bold=True)

    base_station_p = np.array([-3.5, -3.0, 0.0], dtype=np.float32)
    init_coords = np.array([-3.5, -3.0, 1.8], dtype=np.float32) # Drone starts hovering directly over Base Station!
    home_coords = np.array([-3.5, -3.0, 1.5], dtype=np.float32)

    drone = BEMTCoaxialAirframe8DOF(init_coords)
    ekf = ErrorStateEKF15(init_coords)
    quantum_sensor = MatterWaveColdAtomQuantumSensor()
    sensors = FullPhysicsSensorSuite()
    bms = Battery1RCTheveninECM()
    assurance = SurfaceAndWireAssuranceSuite()
    winch = AutonomousWinchDeliverySystem()
    synth = DroneObservationSynthesizer()
    mixer = RateDampedActuatorMixerWithCBF()

    target_delivery_point = assurance.generate_verified_solid_dropzone(init_coords, min_distance=3.5)

    cam_yaw, cam_pitch = 0.65, 0.42
    center_viewport = (VIEW_W // 2, VIEW_H // 2 + 50)
    running = True

    ambient_wind = np.array([2.8, -1.2, 0.0], dtype=np.float32)
    gust_timer = 0.0
    active_gust = np.zeros(3, dtype=np.float32)

    inject_spoof = False
    inject_adsb_threat = False
    inject_emf_spike = False

    hover_settle_timer = 0.0
    packages_delivered_count = 0
    delivered_boxes = []
    flight_time = 0.0

    action_probs = np.zeros(6, dtype=np.float32)
    current_action_name = "TRANSIT"
    policy_latency_ms = 1.2
    confidence = 0.95

    obstacles = [
        {"pos": np.array([ 1.8,  2.5, 0.0]), "radius": 0.55},
        {"pos": np.array([-2.2,  4.5, 0.0]), "radius": 0.65}
    ]

    print("\n================ V4.4 BASE STATION & CARGO ACTIVE ================")
    print(" [BASE STATION] : Illuminated [H] Helipad at [-3.5, -3.0] with Radio Mast")
    print(" [CARGO DOCKED] : 2.0 kg Delivery Parcel Clamped to Fuselage Belly")
    print(" [H]            : Return to Base Station Helipad & Land")
    print(" [CLICK]        : Retarget Dropzone Coordinates on 3D Ground")
    print(" [1] - [8]      : Kill Individual Motors (Coaxial Fault-Tolerance)")
    print(" [G]            : Inject 35 km/h Wind Microburst (DOB Crab Leaning)")
    print(" [J]            : Toggle GPS Spoofing Attack (Quantum VIO Lock)")
    print("==================================================================\n")

    while running:
        dt = 0.016
        flight_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if mx < VIEW_W:
                    click_gnd = screen_to_ground_raycast((mx, my), (cam_yaw, cam_pitch), center_viewport)
                    target_delivery_point = click_gnd
                    hover_settle_timer = 0.0
                    winch.reload_new_cargo()
                    mixer.e_p_integral.fill(0.0)
                    print(f">>> MANUAL DROPZONE RETARGETED: [{click_gnd[0]:.2f}, {click_gnd[1]:.2f}, 0.00]m <<<")
            elif event.type == pygame.KEYDOWN:
                if pygame.K_1 <= event.key <= pygame.K_8:
                    m_idx = event.key - pygame.K_1
                    drone.motor_health[m_idx] = 0.0 if drone.motor_health[m_idx] == 1.0 else 1.0
                    st = "KILLED (0 RPM)" if drone.motor_health[m_idx] == 0.0 else "RESTORED"
                    print(f">>> [PROPULSION] MOTOR #{m_idx+1}: {st} <<<")
                elif event.key == pygame.K_h:
                    target_delivery_point = base_station_p.copy()
                    hover_settle_timer = 0.0
                    mixer.e_p_integral.fill(0.0)
                    print(">>> RETURNING TO BASE STATION HELIPAD <<<")
                elif event.key == pygame.K_t:
                    inject_adsb_threat = not inject_adsb_threat
                elif event.key == pygame.K_e:
                    inject_emf_spike = not inject_emf_spike
                elif event.key == pygame.K_b:
                    bms.soc = 0.18
                elif event.key == pygame.K_g:
                    gust_timer = 2.5
                    ang = random.uniform(0, 2*math.pi)
                    active_gust = np.array([math.cos(ang)*9.5, math.sin(ang)*9.5, 0.0], dtype=np.float32)
                elif event.key == pygame.K_j:
                    inject_spoof = not inject_spoof
                elif event.key == pygame.K_p:
                    drone.parachute_deployed = True
                elif event.key == pygame.K_r:
                    drone.motor_health.fill(1.0)
                    drone.parachute_deployed = False
                    bms.soc = 0.92
                    inject_adsb_threat = False
                    inject_emf_spike = False
                    inject_spoof = False
                    winch.reload_new_cargo()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: cam_yaw -= 0.03
        if keys[pygame.K_RIGHT]: cam_yaw += 0.03

        if gust_timer > 0.0:
            gust_timer = max(0.0, gust_timer - dt)
            current_wind = ambient_wind + active_gust * (gust_timer / 2.5)
        else:
            current_wind = ambient_wind

        sensors.step_environment(dt)

        target_cx = int(np.clip(VIEW_W // 2 - int(drone.p[0] * 10.0), VIEW_W * 0.25, VIEW_W * 0.75))
        target_cy = int(np.clip(VIEW_H // 2 + 50 + int(drone.p[1] * 6.0), VIEW_H * 0.35, VIEW_H * 0.75))
        center_viewport = (int(center_viewport[0] * 0.94 + target_cx * 0.06),
                           int(center_viewport[1] * 0.94 + target_cy * 0.06))

        # 1. Physics Sensors
        s_data = sensors.poll_all_physics_sensors(drone.p, drone.v, current_wind)
        real_emf = 38.5 if inject_emf_spike else s_data["emf_rms"]
        real_t_cpa = 6.5 if inject_adsb_threat else s_data["t_cpa"]

        surf_class, echo_spl, echo_disp, is_surf_safe = assurance.audit_surface(target_delivery_point[:2])
        wire_hit, wire_dist, wire_type, wire_dolp = assurance.scan_wires(drone.p)

        # 2. Matter-Wave Quantum Sensor (Framework 5)
        R_true = drone.R
        accel_inertial_true = (drone.accel_linear - GRAVITY_G)
        a_quantum, phase_shift = quantum_sensor.measure_phase_shift(accel_inertial_true)

        # 3. 15-State ES-EKF on SO(3)
        accel_body_quantum = R_true.T @ a_quantum
        gyro_raw = drone.omega + np.random.normal(0, 0.005, 3).astype(np.float32)
        ekf.predict(accel_body_quantum, gyro_raw, dt=dt)

        H_baro = np.zeros((1, 15), dtype=np.float32); H_baro[0, 2] = 1.0
        ekf.update_measurement(H_baro, np.array([s_data["z_baro"] - ekf.p[2]], dtype=np.float32), np.array([[0.05]], dtype=np.float32))

        spoof_offset = np.array([3.5, 2.8, 0.5], dtype=np.float32) if inject_spoof else np.zeros(3, dtype=np.float32)
        z_gps = drone.p + spoof_offset + np.random.normal(0, 0.12, 3).astype(np.float32)
        chi2_val = 34.2 if inject_spoof else 1.8

        if not inject_spoof:
            H_pos = np.zeros((3, 15), dtype=np.float32); H_pos[0:3, 0:3] = np.eye(3)
            ekf.update_measurement(H_pos, z_gps - ekf.p, np.eye(3, dtype=np.float32) * 0.02)
        else:
            p_vio = drone.p + np.random.normal(0, 0.02, 3).astype(np.float32)
            H_pos = np.zeros((3, 15), dtype=np.float32); H_pos[0:3, 0:3] = np.eye(3)
            ekf.update_measurement(H_pos, p_vio - ekf.p, np.eye(3, dtype=np.float32) * 0.04)

        # 4. Battery ECM
        bms_soc, v_pack, i_draw = bms.step(drone.step.__dict__.get("last_p_watts", 180.0), dt=dt)

        # 5. Synthesize 32-Element Observation
        R_est = quat_to_rot(ekf.q)
        omega_est = gyro_raw - ekf.b_g
        hover_target = np.array([target_delivery_point[0], target_delivery_point[1], 2.2], dtype=np.float32)
        d_horiz_est = float(np.linalg.norm(ekf.p[:2] - target_delivery_point[:2]))
        v_horiz_est = float(np.linalg.norm(ekf.v[:2]))

        obs_vec = synth.synthesize(
            p_est=ekf.p, v_est=ekf.v, R_est=R_est, omega_est=omega_est,
            target_p=hover_target, v_tas=s_data["v_tas"], q_dyn=s_data["q_dyn"],
            min_sonar=s_data["min_sonar"], emf_rms=real_emf, wind_est=current_wind,
            adsb_t_cpa=real_t_cpa, bms_soc=bms_soc, v_pack=v_pack, cable_len=winch.cable_len,
            cable_vel=winch.cable_vel, tension_n=winch.tension_n, has_payload=winch.has_payload,
            chi2_val=chi2_val, is_spoofed=inject_spoof,
            flight_time=flight_time, total_plan_time=4.5, delivered_count=packages_delivered_count, dt=dt
        )

        # 6. Neural Policy Forward Pass & CBF-Filtered Mixer
        t_inf0 = time.time()
        with torch.no_grad():
            t_in = torch.from_numpy(obs_vec).float().unsqueeze(0).to(device)
            nn_guidance, nn_logits = model(t_in)
            neural_guidance = nn_guidance[0].cpu().numpy()
            probs = F.softmax(nn_logits[0], dim=-1).cpu().numpy()
            token_idx = int(np.argmax(probs))

        policy_latency_ms = (time.time() - t_inf0) * 1000.0
        action_probs = probs
        current_action_name = ACTION_NAMES[token_idx]
        confidence = float(action_probs[token_idx])

        active_target = home_coords if bms_soc <= 0.20 else hover_target

        omega_cmds, f_thrust, a_safe = mixer.mix(
            drone_health=drone.motor_health,
            p_est=ekf.p,
            v_est=ekf.v,
            R_est=R_est,
            omega_est=omega_est,
            p_target=active_target,
            neural_guidance=neural_guidance,
            token_idx=token_idx,
            estimated_wind=current_wind,
            obstacles=obstacles,
            dt=dt
        )

        f_thrust_act, torques, p_watts, ige_fac = drone.step(omega_cmds, current_wind=current_wind, rho_air=s_data["rho_air"], dt=dt)
        drone.step.__dict__["last_p_watts"] = p_watts

        # 7. Unified Winch Tether & Delivery Management
        is_centered = (d_horiz_est <= HOVER_LOCK_RADIUS_M) and (v_horiz_est <= HOVER_LOCK_SPEED_MAX_MS)
        if is_centered and winch.has_payload:
            hover_settle_timer += dt
        else:
            if winch.cable_len <= 0.09:
                hover_settle_timer = 0.0

        is_settled_for_winch = (hover_settle_timer >= HOVER_SETTLE_DURATION_S)

        cable_l, tension, sway_2d = winch.step(
            drone_z=drone.p[2],
            is_settled=is_settled_for_winch,
            is_surface_safe=is_surf_safe,
            wire_hit=wire_hit,
            wind_2d=current_wind[:2],
            dt=dt
        )

        # Auto-Dispatch next dropzone when hook retracts
        if winch.delivery_completed:
            packages_delivered_count += 1
            delivered_boxes.append(target_delivery_point.copy())
            target_delivery_point = assurance.generate_verified_solid_dropzone(ekf.p, min_distance=3.2)
            winch.reload_new_cargo()
            hover_settle_timer = 0.0
            mixer.e_p_integral.fill(0.0)
            print(f">>> [ALL-PHYSICS NEURAL POLICY] PACKAGE #{packages_delivered_count} DELIVERED! AUTO-DISPATCHING <<<")

        # ==========================================
        # RENDERING: 3D VIEWPORT & HUD
        # ==========================================
        screen.fill((10, 13, 20))
        pygame.draw.line(screen, (35, 45, 60), (VIEW_W, 0), (VIEW_W, H), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (W, 70), 2)

        cam = (cam_yaw, cam_pitch)

        # Ground Grid
        for gx in range(-5, 6):
            sx1, sy1, _ = project_3d(np.array([gx * 1.5, -4.0, 0.0]), cam, center_viewport)
            sx2, sy2, _ = project_3d(np.array([gx * 1.5,  7.0, 0.0]), cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        for gy in range(-3, 6):
            sx1, sy1, _ = project_3d(np.array([-7.5, gy * 1.5, 0.0]), cam, center_viewport)
            sx2, sy2, _ = project_3d(np.array([ 7.5, gy * 1.5, 0.0]), cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        # -------------------------------------------------------------
        # BASE STATION HELIPAD (ILLUMINATED [H] PAD & GCS RADIO MAST)
        # -------------------------------------------------------------
        base_corners = [
            base_station_p + np.array([-1.4, -1.4, 0.0]),
            base_station_p + np.array([ 1.4, -1.4, 0.0]),
            base_station_p + np.array([ 1.4,  1.4, 0.0]),
            base_station_p + np.array([-1.4,  1.4, 0.0])
        ]
        base_screens = [project_3d(pt, cam, center_viewport)[:2] for pt in base_corners]
        pygame.draw.polygon(screen, (28, 38, 48), base_screens)
        pygame.draw.lines(screen, (0, 220, 255), True, base_screens, 2)

        hc_sx, hc_sy, _ = project_3d(base_station_p, cam, center_viewport)
        pygame.draw.circle(screen, (0, 220, 255), (hc_sx, hc_sy), 20, 2)
        screen.blit(font_large.render("H", True, (0, 240, 255)), (hc_sx - 6, hc_sy - 10))

        # Four flashing amber corner beacons on pad
        beacon_phase = int(time.time() * 4) % 2 == 0
        b_col = (255, 200, 40) if beacon_phase else (100, 80, 20)
        for bc in base_screens:
            pygame.draw.circle(screen, b_col, bc, 3)

        # GCS Radio Mast & Green Status Beacon
        mast_base = base_station_p + np.array([-1.2, 1.2, 0.0])
        mast_top = base_station_p + np.array([-1.2, 1.2, 2.8])
        mb_sx, mb_sy, _ = project_3d(mast_base, cam, center_viewport)
        mt_sx, mt_sy, _ = project_3d(mast_top, cam, center_viewport)
        pygame.draw.line(screen, (140, 160, 185), (mb_sx, mb_sy), (mt_sx, mt_sy), 2)
        pygame.draw.circle(screen, (0, 255, 140), (mt_sx, mt_sy), 4)
        screen.blit(font_small.render("BASE DEPOT / GCS", True, (0, 220, 255)), (hc_sx - 38, hc_sy + 24))

        # Supply Cargo Crates on Depot Deck
        crate_p = base_station_p + np.array([1.0, -1.0, 0.0])
        cr_sx, cr_sy, _ = project_3d(crate_p, cam, center_viewport)
        pygame.draw.rect(screen, (255, 180, 40), (cr_sx - 5, cr_sy - 5, 11, 11), border_radius=2)
        pygame.draw.rect(screen, (50, 40, 20), (cr_sx - 5, cr_sy - 5, 11, 11), 1)

        # Driveway & Water Pond
        dw = [np.array([-2.0, -3.0, 0.0]), np.array([2.0, -3.0, 0.0]), np.array([2.0, 1.0, 0.0]), np.array([-2.0, 1.0, 0.0])]
        pygame.draw.polygon(screen, (38, 44, 42), [project_3d(p, cam, center_viewport)[:2] for p in dw])

        w_pts = [project_3d(np.array([assurance.water_pond["center"][0] + math.sin(math.radians(a))*assurance.water_pond["radius"],
                                      assurance.water_pond["center"][1] + math.cos(math.radians(a))*assurance.water_pond["radius"], 0.0]), cam, center_viewport)[:2] for a in range(0, 360, 20)]
        pygame.draw.polygon(screen, (20, 48, 75), w_pts)
        pygame.draw.lines(screen, (0, 220, 255), True, w_pts, 2)

        # Overhead Wire
        for w in assurance.overhead_wires:
            w1_s = project_3d(w["p1"], cam, center_viewport)[:2]
            w2_s = project_3d(w["p2"], cam, center_viewport)[:2]
            w_col = (255, 60, 80) if real_emf > 18.0 else (255, 180, 40)
            pygame.draw.line(screen, w_col, w1_s, w2_s, 3)

        # Delivered Boxes on Ground
        for idx, pb in enumerate(delivered_boxes):
            bx, by, _ = project_3d(pb, cam, center_viewport)
            pygame.draw.rect(screen, (60, 160, 90), (bx - 6, by - 6, 12, 12), border_radius=2)
            pygame.draw.rect(screen, (20, 50, 30), (bx - 6, by - 6, 12, 12), 1)
            pygame.draw.line(screen, (255, 255, 255), (bx - 3, by), (bx, by + 3), 1)
            pygame.draw.line(screen, (255, 255, 255), (bx, by + 3), (bx + 4, by - 3), 1)
            screen.blit(font_small.render(f"DELIVERED #{idx+1}", True, (0, 255, 140)), (bx + 8, by - 6))

        # Target Delivery Ring
        tsx, tsy, _ = project_3d(target_delivery_point, cam, center_viewport)
        t_col = (0, 255, 140) if is_surf_safe and not wire_hit else (255, 60, 80)
        pygame.draw.circle(screen, t_col, (tsx, tsy), 14, 2)
        pygame.draw.circle(screen, (255, 255, 255), (tsx, tsy), 3)

        # Drone Airframe
        dsx, dsy, _ = project_3d(drone.p, cam, center_viewport)
        pygame.draw.circle(screen, (0, 240, 255), (dsx, dsy), 8)
        pygame.draw.circle(screen, (255, 255, 255), (dsx, dsy), 3)

        arm_d = ARM_LENGTH_M * 1.8
        arm_ends_body = [
            np.array([ arm_d/1.414, -arm_d/1.414, 0.0]),
            np.array([ arm_d/1.414,  arm_d/1.414, 0.0]),
            np.array([-arm_d/1.414,  arm_d/1.414, 0.0]),
            np.array([-arm_d/1.414, -arm_d/1.414, 0.0]),
        ]
        arm_screens = [project_3d(drone.p + drone.R @ a, cam, center_viewport)[:2] for a in arm_ends_body]
        pygame.draw.line(screen, (160, 170, 185), arm_screens[0], arm_screens[2], 3)
        pygame.draw.line(screen, (160, 170, 185), arm_screens[1], arm_screens[3], 3)

        for m_i in range(8):
            a_i = [0, 0, 1, 1, 2, 2, 3, 3][m_i]
            mx, my = arm_screens[a_i]
            off_y = -6 if (m_i % 2 == 0) else 6
            if drone.motor_health[m_i] == 0.0:
                pygame.draw.line(screen, (255, 60, 80), (mx - 5, my + off_y - 5), (mx + 5, my + off_y + 5), 2)
                pygame.draw.line(screen, (255, 60, 80), (mx + 5, my + off_y - 5), (mx - 5, my + off_y + 5), 2)
            else:
                m_col = (0, 240, 255) if (m_i % 2 == 0) else (0, 255, 140)
                pygame.draw.circle(screen, m_col, (mx, my + off_y), 10, 1)

        # -------------------------------------------------------------
        # TETHER CABLE & HIGH-VISIBILITY CARGO CONTAINER
        # -------------------------------------------------------------
        payload_3d = np.array([
            drone.p[0] + winch.sway_offset[0],
            drone.p[1] + winch.sway_offset[1],
            drone.p[2] - winch.cable_len
        ], dtype=np.float32)

        psx, psy, _ = project_3d(payload_3d, cam, center_viewport)

        if not winch.severed:
            # Draw tether cable line only when unspooling/unspooled (> 0.10 m)
            if winch.cable_len > 0.10:
                pygame.draw.line(screen, (230, 240, 255), (dsx, dsy), (psx, psy), 2)

            if winch.has_payload:
                # 3D Cargo Parcel Box (Visible on belly when stowed, lowering when active)
                box_x, box_y = (dsx, dsy + 5) if winch.cable_len <= 0.10 else (psx, psy)
                pygame.draw.rect(screen, (255, 195, 35), (box_x - 7, box_y - 7, 14, 14), border_radius=2)
                pygame.draw.rect(screen, (60, 45, 10), (box_x - 7, box_y - 7, 14, 14), 1)
                pygame.draw.line(screen, (50, 40, 10), (box_x - 7, box_y), (box_x + 7, box_y), 1)
                pygame.draw.line(screen, (50, 40, 10), (box_x, box_y - 7), (box_x, box_y + 7), 1)
                screen.blit(font_small.render("2.0kg CARGO", True, (255, 215, 0)), (box_x + 9, box_y - 6))
            else:
                # Bare metallic release hook retracting
                if winch.cable_len > 0.10:
                    pygame.draw.circle(screen, (200, 220, 240), (psx, psy), 3)

        gsx, gsy, _ = project_3d(np.array([drone.p[0], drone.p[1], 0.0]), cam, center_viewport)
        pygame.draw.line(screen, (35, 55, 75), (dsx, dsy), (gsx, gsy), 1)

        # ==========================================
        # FULL 5-FRAMEWORK DASHBOARD (SIDEBAR)
        # ==========================================
        sb_x = VIEW_W + 14
        screen.blit(font_large.render("SURVIVAL CORE V4.4", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Base Station Depot, Active Cargo & All Physics", True, (130, 160, 190)), (sb_x, 38))

        # 1. Mission Status & Base Navigation
        p1_y = 75
        screen.blit(font_mono.render("1. CARGO DELIVERY & BASE DEPOT:", True, (240, 240, 240)), (sb_x, p1_y))
        dist_home = float(np.linalg.norm(drone.p[:2] - base_station_p[:2]))
        screen.blit(font_small.render(f"Base Station Pad:  [-3.5, -3.0] | Range: {dist_home:.2f} m [H]", True, (0, 220, 255)), (sb_x + 10, p1_y + 20))
        cargo_tag = "2.0 kg BOX ATTACHED" if winch.has_payload else "EMPTY HOOK RETRACTING"
        cargo_col = (0, 255, 140) if winch.has_payload else (255, 200, 40)
        screen.blit(font_small.render(f"Fuselage Cargo:    {cargo_tag}", True, cargo_col), (sb_x + 10, p1_y + 36))
        screen.blit(font_small.render(f"Delivered Total:   {packages_delivered_count} Packages Deposited", True, (0, 255, 140)), (sb_x + 10, p1_y + 52))

        # 2. BEMT & Ground Effect
        p2_y = p1_y + 74
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (W - 14, p2_y), 1)
        screen.blit(font_mono.render("2. BEMT AERODYNAMICS & IGE (F1, F2):", True, (240, 240, 240)), (sb_x, p2_y + 8))
        screen.blit(font_small.render(f"BEMT Radial Segs:  8 Elements with Prandtl Tip Losses", True, (0, 220, 255)), (sb_x + 10, p2_y + 26))
        screen.blit(font_small.render(f"Ground Cushion IGE:T_ige/T_oge = {ige_fac:.2f}x (Method of Images)", True, (0, 255, 140) if ige_fac > 1.05 else (160, 180, 205)), (sb_x + 10, p2_y + 42))
        screen.blit(font_small.render(f"Total Thrust:      F_z = {f_thrust:.1f} N (Hover: {HOVER_THRUST:.1f} N)", True, (160, 180, 205)), (sb_x + 10, p2_y + 58))

        # 3. Tether Winch & Touchdown Dynamics
        p3_y = p2_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (W - 14, p3_y), 1)
        screen.blit(font_mono.render("3. MOTORIZED TETHER WINCH (MOD 10):", True, (240, 240, 240)), (sb_x, p3_y + 8))
        screen.blit(font_small.render(f"Winch State:       {winch.state} | Deployed: {winch.cable_len:.2f} m", True, (0, 220, 255)), (sb_x + 10, p3_y + 26))
        screen.blit(font_small.render(f"Drum Cable Tension:F = {tension:.1f} N (Touchdown = 1.8 N)", True, (0, 255, 140)), (sb_x + 10, p3_y + 42))
        sway_mag = float(np.linalg.norm(sway_2d)) * 100.0
        screen.blit(font_small.render(f"Tether Wind Sway:  Offset = {sway_mag:.1f} cm", True, (160, 180, 205)), (sb_x + 10, p3_y + 58))

        # 4. Matter-Wave Quantum Sensor & CBF-QP
        p4_y = p3_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (W - 14, p4_y), 1)
        screen.blit(font_mono.render("4. QUANTUM & CBF-QP SAFETY (F4, F5):", True, (240, 240, 240)), (sb_x, p4_y + 8))
        phase_norm = float(np.linalg.norm(phase_shift))
        screen.blit(font_small.render(f"Quantum Phase Shift: ΔΦ = {phase_norm:.3f} rad (Rb-87 Raman)", True, (0, 255, 140)), (sb_x + 10, p4_y + 26))
        screen.blit(font_small.render(f"CBF Safety Barrier:  KKT Active (Zero Collision Invariance)", True, (0, 220, 255)), (sb_x + 10, p4_y + 42))
        screen.blit(font_small.render(f"Anti-Spoof Defense:  {'ISOLATED (QUANTUM VIO)' if inject_spoof else 'GNSS VERIFIED'}", True, (255, 60, 80) if inject_spoof else (0, 255, 140)), (sb_x + 10, p4_y + 58))

        # 5. Neural Policy Decision
        p5_y = p4_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p5_y), (W - 14, p5_y), 1)
        screen.blit(font_mono.render("5. DEEP NEURAL POLICY DECISION:", True, (240, 240, 240)), (sb_x, p5_y + 8))
        screen.blit(font_small.render(f"Selected Policy: {current_action_name} ({confidence*100:.1f}%)", True, (0, 255, 140)), (sb_x + 10, p5_y + 26))
        screen.blit(font_small.render(f"Inference Time:  {policy_latency_ms:.2f} ms ({device.type.upper()})", True, (0, 220, 255)), (sb_x + 10, p5_y + 42))
        screen.blit(font_small.render(f"Target Dropzone: [{target_delivery_point[0]:.2f}, {target_delivery_point[1]:.2f}, 0.0] m", True, (255, 200, 40)), (sb_x + 10, p5_y + 58))

        # 6. Controls
        p6_y = p5_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p6_y), (W - 14, p6_y), 1)
        screen.blit(font_mono.render("OPERATOR COMMANDS:", True, (255, 200, 40)), (sb_x, p6_y + 8))
        screen.blit(font_small.render("• [H] Return to Base Station Helipad | [CLICK] Retarget", True, (240, 240, 240)), (sb_x + 10, p6_y + 24))
        screen.blit(font_small.render("• [G] Wind Microburst | [J] GPS Spoofing Attack", True, (240, 240, 240)), (sb_x + 10, p6_y + 40))

        # Bottom Mission Status Banner
        bot_y = H - 45
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y - 8), (W - 14, bot_y - 8), 1)
        if drone.parachute_deployed:
            b_txt, b_col = "SAFETY BARRIER: BALLISTIC PARACHUTE TERMINATION", (255, 140, 20)
        elif inject_spoof:
            b_txt, b_col = "QUANTUM VIO LOCK: DRIFT-FREE INERTIAL RECOVERY", (0, 255, 140)
        elif winch.state == "LOWERING":
            b_txt, b_col = "WINCH: LOWERING 2.0kg CARGO ONTO DROPZONE", (0, 255, 140)
        elif winch.state == "TOUCHDOWN":
            b_txt, b_col = "TOUCHDOWN CONFIRMED: UNLATCHING PACKAGE", (0, 255, 140)
        elif dist_home < 0.6:
            b_txt, b_col = "BASE STATION DEPOT: HOVER LOCKED OVER HELIPAD", (0, 220, 255)
        else:
            b_txt, b_col = f"ALL-PHYSICS AUTONOMOUS NEURAL FLIGHT: {current_action_name}", (0, 255, 140)
        screen.blit(font_mono.render(b_txt, True, b_col), (sb_x + 8, bot_y))

        # Header HUD
        head_txt = font_large.render(f"Survival Core V4.4 | Cargo: {'LOADED' if winch.has_payload else 'DELIVERED'} | Delivered: {packages_delivered_count}", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("Base Station Helipad, Cargo Clamping & Delivery, BEMT Aerodynamics, CBF-QP & Quantum Sensing", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    mp.freeze_support()
    run_all_physics_simulator()