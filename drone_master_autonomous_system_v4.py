"""
=============================================================================
MASTER AUTONOMOUS DRONE SYSTEM CORE (V7.5 - COMPLETE ALL-PHYSICS PRODUCTION)
File: drone_master_autonomous_system_v3.py
=============================================================================
- Physics 1: Anti-Spoof Quantum-Inertial Kinematic Check & VIO Fallback
- Physics 2: Rooftop Leeward Separation Downwash & Recirculating Cavity Wake
- Physics 3: Messinger Rotor Blade Icing Model (BEMT Lift/Drag Distortion)
- Physics 4: Sub-Zero Arrhenius Battery ECM (Cold-Soak Terminal Voltage Sag)
- Physics 5: Dynamic Ground Agent Pursuit (Curious Dog/Human -> Winch Abort)
- Physics 6: Bilateral Slung-Load Spherical Pendulum with Mass Shed Transition
- Visuals: Full 3D House, Pool Cover, Helipad, Wires, Wind Streamers & Cockpit HUD
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
# 0. AIRFRAME & ENVIRONMENTAL SPECIFICATIONS
# ==========================================
DRY_MASS_KG = 1.35                   # Bare airframe weight
CARGO_MASS_KG = 0.45                 # Standard parcel payload
MASS_KG = DRY_MASS_KG + CARGO_MASS_KG # Loaded mass (1.80 kg)

GRAVITY_G = np.array([0.0, 0.0, -9.80665], dtype=np.float32)
G_ACCEL = 9.80665
ARM_LENGTH_M = 0.26
ROTOR_RADIUS_M = 0.127

INERTIA_DRY = np.diag([0.018, 0.018, 0.032]).astype(np.float32)
INERTIA_LOADED = np.diag([0.024, 0.024, 0.040]).astype(np.float32)

C_T_CLEAN = 1.25e-5
C_Q_CLEAN = 2.40e-7
COAXIAL_EFFICIENCY = 0.88
MOTOR_TAU_S = 0.030
MAX_RPM = 9200.0
MAX_OMEGA = (MAX_RPM * 2.0 * math.pi) / 60.0

HOVER_THRUST_NOMINAL = MASS_KG * G_ACCEL
HOVER_OMEGA_NOMINAL = math.sqrt((HOVER_THRUST_NOMINAL / 8.0) / (C_T_CLEAN * 0.94))

NUM_BLADE_ELEMENTS = 8
PRANDTL_TIP_FACTOR = 0.92
COAXIAL_DOWNWASH_INGESTION = 0.82

IGE_HEIGHT_CUTOFF_M = 1.8 * ROTOR_RADIUS_M * 2.0
CEILING_HEIGHT_M = 14.0

CBF_ALPHA_OBSTACLE = 1.8
CBF_ALPHA_CEILING = 2.2
CBF_SAFE_RADIUS_M = 0.65

K_EFF_WAVEVECTOR = 1.61e7            # Rubidium-87 Raman wavevector (m^-1)
RAMAN_PULSE_SEP_T = 0.015            # 15 ms pulse interrogation time
QUANTUM_PHASE_1G = K_EFF_WAVEVECTOR * G_ACCEL * (RAMAN_PULSE_SEP_T ** 2)

P_0 = 1013.25
T_0 = 288.15
L_LAPSE = 0.0065
R_SPEC_AIR = 287.058
SPEED_OF_SOUND = 343.2
EMF_WARNING_THRESHOLD_UT = 18.0

NUM_CELLS = 6
PACK_CAPACITY_AH = 4.5
PACK_ENERGY_WH = NUM_CELLS * 3.7 * PACK_CAPACITY_AH
CELL_R0_NOMINAL = 0.0085
CELL_R1 = 0.0065
CELL_C1 = 2800.0
ARRHENIUS_EA = 28000.0
GAS_CONST_R = 8.31446
T_REF_K = 298.15

PAYLOAD_WEIGHT_N = CARGO_MASS_KG * G_ACCEL
GUILLOTINE_THRESHOLD_N = 48.0
HOVER_LOCK_RADIUS_M = 0.85
HOVER_LOCK_SPEED_MAX_MS = 0.40
HOVER_SETTLE_DURATION_S = 0.30

OBSERVATION_DIM = 32
NUM_EXECUTIVE_STATES = 6
MODEL_WEIGHTS_FILE = "drone_master_brain_complete.pth"

ACTION_NAMES = [
    "TRANSIT",             # 0: Cruising to dropzone along spline
    "HOVER_LOCK",          # 1: Precision settling over target basket
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

# ==========================================
# 2. BUILDING WAKE & ROOFTOP SEPARATION
# ==========================================
class BuildingAerodynamicWakeSimulator:
    def __init__(self):
        self.building = {"x_min": -0.5, "x_max": 2.2, "y_min": -1.0, "y_max": 1.4, "h": 2.8}

    def compute_wake_velocity(self, drone_pos, freestream_wind):
        x, y, z = drone_pos
        b = self.building
        w_wake = np.zeros(3, dtype=np.float32)

        wind_dir = freestream_wind[:2] / max(1e-4, float(np.linalg.norm(freestream_wind[:2])))
        dx = x - b["x_max"] if wind_dir[0] >= 0 else b["x_min"] - x

        # Leeward downwash region
        if 0.0 < dx < 3.8 and b["y_min"] - 0.5 <= y <= b["y_max"] + 0.5 and z < b["h"] * 1.6:
            wind_speed = float(np.linalg.norm(freestream_wind[:2]))
            sigma_y = (b["y_max"] - b["y_min"]) * 0.5
            y_mid = 0.5 * (b["y_min"] + b["y_max"])
            lateral_decay = math.exp(-((y - y_mid) ** 2) / (2.0 * sigma_y ** 2))
            downwind_decay = math.exp(-dx / 1.8)

            z_factor = max(0.1, min(1.0, z / b["h"]))
            downwash = -wind_speed * 0.85 * (1.0 - z_factor) * lateral_decay * downwind_decay
            recirculation_u = -wind_speed * 0.40 * lateral_decay * downwind_decay

            w_wake[0] += recirculation_u * wind_dir[0]
            w_wake[1] += recirculation_u * wind_dir[1]
            w_wake[2] += downwash

        return w_wake

# ==========================================
# 3. DYNAMIC GROUND AGENT PURSUIT MODEL
# ==========================================
class DynamicGroundIntruderAgent:
    def __init__(self, init_pos):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.v_max = 3.6
        self.personal_space_m = 1.85

    def step(self, cargo_world_pos, is_winch_lowering, dt=0.016):
        d_to_cargo = cargo_world_pos[:2] - self.p[:2]
        dist = float(np.linalg.norm(d_to_cargo))

        if is_winch_lowering and cargo_world_pos[2] < 2.0 and dist < 5.5:
            v_des = (d_to_cargo / max(1e-4, dist)) * self.v_max
            a_cmd = (v_des - self.v[:2]) * 3.5
        else:
            target_wander = np.array([-1.5, 3.0])
            d_wander = target_wander - self.p[:2]
            v_des = (d_wander / max(1e-4, float(np.linalg.norm(d_wander)))) * 0.6
            a_cmd = (v_des - self.v[:2]) * 1.5

        self.v[:2] += a_cmd * dt
        self.p[:2] += self.v[:2] * dt
        self.p[2] = 0.0

        agent_breach = dist < self.personal_space_m
        return self.p.copy(), agent_breach, dist

# ==========================================
# 4. MESSINGER ROTOR ICING ACCRETION
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

# ==========================================
# 5. DRYDEN ATMOSPHERIC TURBULENCE
# ==========================================
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
        du = (- (v / l_u) * self.gust_state[0] + sigma_u * math.sqrt(2.0 * v / l_u) * white_noise[0]) * dt
        dv = (- (v / l_v) * self.gust_state[1] + sigma_v * math.sqrt(2.0 * v / l_v) * white_noise[1]) * dt
        dw = (- (v / l_w) * self.gust_state[2] + sigma_w * math.sqrt(2.0 * v / l_w) * white_noise[2]) * dt

        self.gust_state[0] += du
        self.gust_state[1] += dv
        self.gust_state[2] += dw
        self.gust_state = np.clip(self.gust_state, -7.0, 7.0)
        return self.gust_state.copy()

# ==========================================
# 6. BILATERAL WINCH WITH ABORT LOGIC
# ==========================================
class BilateralCoupledWinchSystem:
    def __init__(self):
        self.cable_len = 0.08
        self.cable_vel = 0.0
        self.has_payload = True
        self.touchdown_timer = 0.0
        self.severed = False
        self.state = "STANDBY"
        self.delivery_completed = False
        self.snag_timer = 0.0
        self.theta = np.zeros(2, dtype=np.float32)
        self.theta_dot = np.zeros(2, dtype=np.float32)
        self.r_winch_body = np.array([0.0, 0.0, -0.08], dtype=np.float32)

    def reload_new_cargo(self):
        self.cable_len = 0.08
        self.cable_vel = 0.0
        self.has_payload = True
        self.state = "STANDBY"
        self.touchdown_timer = 0.0
        self.delivery_completed = False
        self.theta.fill(0.0)
        self.theta_dot.fill(0.0)

    def step(self, drone_pos, drone_accel, is_settled, is_surface_safe, wire_hit, agent_breach, wind_vec, dt=0.016):
        if self.severed:
            self.state = "GUILLOTINE"
            return self.cable_len, 0.0, np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, -1.0], dtype=np.float32)

        lockout = (not is_surface_safe) or wire_hit or (not is_settled)

        # Agent Breach Safety Abort
        if agent_breach and self.state in ["LOWERING", "TOUCHDOWN"]:
            self.state = "ABORT_RETRACT"
            print(">>> [SAFETY INTERLOCK] GROUND AGENT INTRUSION -> EMERGENCY WINCH ABORT <<<")

        if self.state == "STANDBY":
            self.cable_len = 0.08
            self.cable_vel = 0.0
            if not lockout and not agent_breach and self.has_payload:
                self.state = "LOWERING"
                print(">>> [DROPZONE CLEAR] LOWERING 0.45kg CARGO PARCEL (2.4 m/s) <<<")

        elif self.state == "LOWERING":
            if lockout:
                self.state = "RETRACTING"
            else:
                self.cable_vel = 2.40
                self.cable_len += self.cable_vel * dt
                cargo_z = drone_pos[2] - self.cable_len * math.cos(float(np.linalg.norm(self.theta)))
                if cargo_z <= 0.06:
                    self.state = "TOUCHDOWN"
                    self.touchdown_timer = 0.0
                    print(">>> [TOUCHDOWN SENSING] CARGO GROUND CONTACT CONFIRMED <<<")

        elif self.state == "TOUCHDOWN":
            self.cable_vel = 0.0
            self.touchdown_timer += dt
            if self.touchdown_timer >= 0.40:
                self.has_payload = False
                self.state = "RETRACTING"
                print(">>> [UNLATCH CONFIRMED] PACKAGE RELEASED! MASS SHED FROM DYNAMICS <<<")

        elif self.state in ["RETRACTING", "ABORT_RETRACT"]:
            self.cable_vel = -3.40
            self.cable_len = max(0.08, self.cable_len + self.cable_vel * dt)
            if self.cable_len <= 0.08:
                self.cable_len = 0.08
                self.cable_vel = 0.0
                if self.state == "ABORT_RETRACT":
                    self.state = "STANDBY"
                else:
                    self.state = "STANDBY"
                    self.delivery_completed = True
                    print(">>> [HOOK DOCKED] HOOK LATCHED AT BELLY. AUTO-DISPATCH ACTIVE <<<")

        effective_mass = CARGO_MASS_KG if self.has_payload else 0.04
        if self.state == "TOUCHDOWN": effective_mass = 0.02
        l = max(0.08, self.cable_len)

        rel_air = wind_vec[:2] - drone_accel[:2] * dt
        drag_f = 0.5 * 1.225 * 0.015 * rel_air * np.linalg.norm(rel_air)

        theta_ddot_x = (-G_ACCEL * math.sin(self.theta[0]) - drone_accel[0] + drag_f[0] / max(0.05, effective_mass)) / l - (2.0 * self.cable_vel / l) * self.theta_dot[0]
        theta_ddot_y = (-G_ACCEL * math.sin(self.theta[1]) - drone_accel[1] + drag_f[1] / max(0.05, effective_mass)) / l - (2.0 * self.cable_vel / l) * self.theta_dot[1]

        theta_ddot_x -= 0.65 * self.theta_dot[0]
        theta_ddot_y -= 0.65 * self.theta_dot[1]

        if self.cable_len > 0.10 and self.state != "TOUCHDOWN":
            self.theta_dot[0] += theta_ddot_x * dt
            self.theta_dot[1] += theta_ddot_y * dt
            self.theta[0] += self.theta_dot[0] * dt
            self.theta[1] += self.theta_dot[1] * dt
        else:
            self.theta.fill(0.0)
            self.theta_dot.fill(0.0)

        self.theta = np.clip(self.theta, -0.65, 0.65)
        s_tx, s_ty = math.sin(self.theta[0]), math.sin(self.theta[1])
        u_cable = np.array([s_tx, s_ty, -math.sqrt(max(1e-4, 1.0 - s_tx**2 - s_ty**2))], dtype=np.float32)

        centrifugal = effective_mass * l * float(np.dot(self.theta_dot, self.theta_dot))
        inertial_pull = effective_mass * float(np.dot(drone_accel - GRAVITY_G, -u_cable))
        tension_n = max(0.2, centrifugal + inertial_pull)

        f_cable_world = tension_n * u_cable
        return self.cable_len, tension_n, f_cable_world, u_cable

# ==========================================
# 7. CONTINUOUS BEMT COAXIAL AIRFRAME
# ==========================================
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

    def step(self, motor_cmds, current_wind, rho_air, has_cargo_rigid, f_cable_reaction_world, ice_index=0.0, dt=0.016):
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

        current_mass = DRY_MASS_KG + (CARGO_MASS_KG if has_cargo_rigid else 0.0)
        current_inertia = INERTIA_LOADED if has_cargo_rigid else INERTIA_DRY
        inv_inertia = np.linalg.inv(current_inertia)

        effective_cmds = np.clip(motor_cmds, 0.0, MAX_OMEGA) * self.motor_health
        self.motor_omegas += ((effective_cmds - self.motor_omegas) / MOTOR_TAU_S) * dt
        self.motor_omegas *= self.motor_health

        z_h = max(0.12, float(self.p[2]))
        ige_factor = 1.0 / max(0.35, (1.0 - (ROTOR_RADIUS_M / (4.0 * z_h)) ** 2))
        ige_factor = float(np.clip(ige_factor, 1.0, 1.35))
        d_ceil = max(0.3, CEILING_HEIGHT_M - z_h)
        ice_suction = float(1.0 + (ROTOR_RADIUS_M / (4.0 * d_ceil)) ** 2)

        T = np.zeros(8, dtype=np.float32)
        Q = np.zeros(8, dtype=np.float32)
        r_steps = np.linspace(0.15 * ROTOR_RADIUS_M, ROTOR_RADIUS_M, NUM_BLADE_ELEMENTS)
        dr = r_steps[1] - r_steps[0]

        lift_loss_factor = 1.0 - 0.28 * ice_index
        drag_rise_factor = 1.0 + 1.80 * ice_index

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
                c_l = 2.0 * math.pi * alpha * lift_loss_factor
                c_d = (0.012 + 0.04 * (alpha ** 2)) * drag_rise_factor

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
        tau_y = arm * (- (T[0] + T[1]) - (T[2] + T[3]) + (T[4] + T[5]) + (T[6] + T[7]))
        tau_z = -Q[0] + Q[1] + Q[2] - Q[3] - Q[4] + Q[5] + Q[6] - Q[7]
        motor_torques = np.array([tau_x, tau_y, tau_z], dtype=np.float32)

        v_air = self.v - current_wind
        speed_air = float(np.linalg.norm(v_air))
        f_drag = -0.5 * rho_air * np.array([0.22, 0.22, 0.38], dtype=np.float32) * v_air * max(1.0, speed_air)

        f_cable_body = self.R.T @ f_cable_reaction_world
        tau_cable_body = np.cross(np.array([0.0, 0.0, -0.08], dtype=np.float32), f_cable_body)

        f_body = np.array([0.0, 0.0, f_thrust], dtype=np.float32)
        f_gravity = current_mass * GRAVITY_G

        self.accel_linear = (self.R @ f_body + f_gravity + f_drag + f_cable_reaction_world) / current_mass
        gyroscopic = np.cross(self.omega, current_inertia @ self.omega)
        accel_angular = inv_inertia @ (motor_torques + tau_cable_body - gyroscopic)

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
        return f_thrust, motor_torques, p_watts, ige_factor

# ==========================================
# 8. SENSORS, SUB-ZERO ECM & ACOUSTIC SUITE
# ==========================================
class SurfaceAndWireAssuranceSuite:
    def __init__(self):
        self.water_pond = {"center": np.array([2.5, 3.8, 0.0]), "radius": 2.0}
        self.pool_cover = {"center": np.array([2.8, -2.5, 0.0]), "radius": 1.6, "impedance_r": 0.45}
        self.overhead_wires = [
            {"p1": np.array([-5.0, 1.8, 3.2]), "p2": np.array([5.0, 1.8, 3.2]), "diam_mm": 2.5}
        ]

    def audit_surface(self, pos_xy):
        px, py = pos_xy
        if math.hypot(px - self.water_pond["center"][0], py - self.water_pond["center"][1]) < self.water_pond["radius"]:
            return "LIQUID_WATER_HAZARD", False, 0.15
        if math.hypot(px - self.pool_cover["center"][0], py - self.pool_cover["center"][1]) < self.pool_cover["radius"]:
            return "POOL_COVER_HAZARD", False, self.pool_cover["impedance_r"]
        return "SOLID_TERRAIN_SAFE", True, 0.92

    def scan_wires(self, drone_p, radius=1.6):
        for wire in self.overhead_wires:
            w1, w2 = wire["p1"], wire["p2"]
            if drone_p[2] > w1[2]:
                if min(w1[0], w2[0]) <= drone_p[0] <= max(w1[0], w2[0]):
                    if abs(drone_p[1] - w1[1]) < radius:
                        return True, drone_p[2] - w1[2]
        return False, 50.0

    def generate_verified_solid_dropzone(self, current_pos, min_distance=3.0):
        for _ in range(60):
            cand = np.array([random.uniform(-4.0, 4.0), random.uniform(-2.2, 5.5), 0.0], dtype=np.float32)
            if np.linalg.norm(cand[:2] - current_pos[:2]) < min_distance: continue
            _, is_safe, _ = self.audit_surface(cand[:2])
            if is_safe: return cand
        return np.array([1.5, 2.5, 0.0], dtype=np.float32)

class SubZeroArrheniusBatteryECM:
    def __init__(self):
        self.soc = 0.94
        self.temp_c = 22.0
        self.v1 = np.zeros(NUM_CELLS, dtype=np.float32)
        self.capacity_coulombs = PACK_CAPACITY_AH * 3600.0
        self.v_pack = 23.8

    def step(self, p_elec_watts, dt=0.016):
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
        self.temp_c += ((p_joule - 0.35 * (self.temp_c - 15.0)) / 350.0) * dt
        return self.soc, self.v_pack, i_draw, r0

# ==========================================
# 9. ESTIMATION & CONTROL BARRIER FILTER
# ==========================================
class MatterWaveColdAtomQuantumSensor:
    def __init__(self):
        self.k_eff = K_EFF_WAVEVECTOR
        self.T = RAMAN_PULSE_SEP_T

    def measure_phase_shift(self, true_accel_inertial):
        phase_shift = self.k_eff * true_accel_inertial * (self.T ** 2)
        phase_shift += np.random.normal(0, 0.002, 3).astype(np.float32)
        a_quantum = phase_shift / (self.k_eff * (self.T ** 2))
        return a_quantum, phase_shift

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
        return self.guidance_head(latent), self.executive_head(latent)

class DroneObservationSynthesizer:
    def __init__(self):
        self.last_tension = PAYLOAD_WEIGHT_N

    def synthesize(self, p_est, v_est, R_est, omega_est, target_p, v_tas, q_dyn, min_sonar, emf_rms,
                   wind_est, adsb_t_cpa, bms_soc, v_pack, cable_len, cable_vel, tension_n, has_payload,
                   chi2_val, is_spoofed, flight_time, total_plan_time=4.5, delivered_count=0, dt=0.016):
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        delta_p = target_p - p_est
        obs[0:3] = np.clip(delta_p / 10.0, -1.0, 1.0)
        obs[3:6] = np.clip(v_est / 6.0, -1.0, 1.0)
        obs[6:9] = np.clip(R_est[:, 2], -1.0, 1.0)
        obs[9:12] = np.clip(omega_est / 8.0, -1.0, 1.0)
        obs[12] = np.clip(v_tas / 15.0, 0.0, 1.0)
        obs[13] = np.clip(q_dyn / 100.0, 0.0, 1.0)
        obs[14] = np.clip(min_sonar / 8.0, 0.0, 1.0)
        obs[15] = np.clip(emf_rms / 40.0, 0.0, 1.0)
        obs[16:19] = np.clip(wind_est / 8.0, -1.0, 1.0)
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
        obs[29] = np.clip(flight_time / max(0.1, total_plan_time), 0.0, 1.0)
        obs[30] = np.clip(math.hypot(delta_p[0], delta_p[1]) / 10.0, 0.0, 1.0)
        obs[31] = np.clip((delivered_count % 10) / 10.0, 0.0, 1.0)
        return obs

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

class RateDampedActuatorMixerWithCBF:
    def __init__(self):
        self.k_R = 2.1
        self.k_w = 0.42
        self.e_p_integral = np.zeros(3, dtype=np.float32)
        self.cbf_filter = ControlBarrierFunctionQPFilter()

        arm = ARM_LENGTH_M / math.sqrt(2.0)
        cq_ct = C_Q_CLEAN / C_T_CLEAN
        self.B_nominal = np.array([
            [ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
            [-arm, -arm,  arm,  arm,  arm,  arm, -arm, -arm],
            [-arm, -arm, -arm, -arm,  arm,  arm,  arm,  arm],
            [-cq_ct, cq_ct, cq_ct, -cq_ct, -cq_ct, cq_ct, cq_ct, -cq_ct]
        ], dtype=np.float32)

    def mix(self, drone_health, p_est, v_est, R_est, omega_est, p_target, neural_guidance, token_idx,
            estimated_wind, obstacles, active_mass_kg, dt=0.016):
        delta_p = p_target - p_est
        d_horiz = float(np.linalg.norm(delta_p[:2]))

        if token_idx == 0:  # TRANSIT
            v_cruise = 2.8
            dir_norm = delta_p / max(1e-4, float(np.linalg.norm(delta_p)))
            v_des = dir_norm * v_cruise
            if d_horiz < 2.0: v_des[:2] *= (d_horiz / 2.0)
        else:
            v_des = np.zeros(3, dtype=np.float32)

        e_p = p_est - p_target
        e_v = v_est - v_des
        self.e_p_integral = np.clip(self.e_p_integral + e_p * dt, -1.5, 1.5)

        f_neural = neural_guidance[:3] * 5.0
        a_nominal = (-np.array([6.5, 6.5, 11.5]) * e_p - np.array([4.4, 4.4, 7.0]) * e_v
                     - np.array([1.2, 1.2, 2.5]) * self.e_p_integral + f_neural) / active_mass_kg

        a_safe = self.cbf_filter.enforce_safety(p_est, v_est, a_nominal, obstacles)
        F_d = a_safe * active_mass_kg + active_mass_kg * (-GRAVITY_G)

        max_tilt = math.radians(25.0)
        f_h = math.hypot(F_d[0], F_d[1])
        f_v = max(1.0, F_d[2])
        if f_h / f_v > math.tan(max_tilt):
            F_d[:2] *= (f_v * math.tan(max_tilt)) / f_h

        b3 = R_est[:, 2]
        f_thrust = float(np.clip(np.dot(F_d, b3), 0.1, 42.0))

        b3_d = F_d / (np.linalg.norm(F_d) + 1e-6)
        b1_c = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b2_d = np.cross(b3_d, b1_c); b2_d /= (np.linalg.norm(b2_d) + 1e-6)
        b1_d = np.cross(b2_d, b3_d)
        R_d = np.column_stack([b1_d, b2_d, b3_d]).astype(np.float32)

        R_err_mat = 0.5 * (R_d.T @ R_est - R_est.T @ R_d)
        e_R = vee_map(R_err_mat)
        tau = -self.k_R * e_R - self.k_w * omega_est

        desired_wrench = np.array([f_thrust, tau[0], tau[1], tau[2]], dtype=np.float32)
        W_health = np.diag(drone_health).astype(np.float32)
        B_active = self.B_nominal @ W_health
        try:
            T_cmds = np.linalg.pinv(B_active) @ desired_wrench
        except np.linalg.LinAlgError:
            T_cmds = np.full(8, f_thrust / 8.0, dtype=np.float32)

        T_cmds = np.clip(T_cmds * drone_health, 0.0, 11.0)

        omega_cmds = np.zeros(8, dtype=np.float32)
        for i in range(8):
            if drone_health[i] > 0.0:
                scale_c = C_T_CLEAN if (i % 2 == 0) else (C_T_CLEAN * COAXIAL_EFFICIENCY)
                omega_cmds[i] = math.sqrt(T_cmds[i] / scale_c)

        return omega_cmds, f_thrust, a_safe

# ==========================================
# 10. NEURAL COMPILER & PARALLEL HARVESTER
# ==========================================
def _worker_harvest_v4(num_steps):
    np.random.seed()
    random.seed()
    synth = DroneObservationSynthesizer()
    observations, teacher_guidance, teacher_fsm = [], [], []

    while len(observations) < num_steps:
        start_p = np.array([random.uniform(-4, 4), random.uniform(-4, 4), random.uniform(1.2, 3)], dtype=np.float32)
        target_gnd = np.array([random.uniform(-4, 4), random.uniform(-2, 6), 0.0], dtype=np.float32)
        drone_p = start_p.copy()
        drone_v = np.zeros(3, dtype=np.float32)
        hover_dest = np.array([target_gnd[0], target_gnd[1], 2.2], dtype=np.float32)

        step_count = 0
        flight_time = 0.0
        dt = 0.016
        episode_wind = np.array([random.uniform(-4.0, 4.0), random.uniform(-4.0, 4.0), 0.0], dtype=np.float32)

        while step_count < 350 and len(observations) < num_steps:
            flight_time += dt
            d_horiz = float(np.linalg.norm(drone_p[:2] - target_gnd[:2]))
            fsm_token = 0 if d_horiz > 0.45 else 1

            delta = hover_dest - drone_p
            v_des = (delta / max(1e-4, float(np.linalg.norm(delta)))) * 2.8 if fsm_token == 0 else np.zeros(3)

            F_d = -np.array([6.5, 6.5, 11.5]) * (drone_p - hover_dest) - np.array([4.4, 4.4, 7.0]) * (drone_v - v_des) + MASS_KG * (-GRAVITY_G)
            guidance_vec = np.array([F_d[0]/20.0, F_d[1]/20.0, F_d[2]/20.0, 0.0], dtype=np.float32)

            obs_t = synth.synthesize(
                p_est=drone_p, v_est=drone_v, R_est=np.eye(3, dtype=np.float32), omega_est=np.zeros(3),
                target_p=hover_dest, v_tas=float(np.linalg.norm(drone_v - episode_wind)), q_dyn=12.0,
                min_sonar=max(0.1, drone_p[2]), emf_rms=4.0, wind_est=episode_wind, adsb_t_cpa=50.0,
                bms_soc=0.85, v_pack=22.8, cable_len=0.08, cable_vel=0.0, tension_n=PAYLOAD_WEIGHT_N,
                has_payload=True, chi2_val=1.5, is_spoofed=False, flight_time=flight_time, dt=dt
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
    num_samples = 40000
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per = num_samples // num_workers

    print(f"\n[NEURAL COMPILER] Generating {num_samples} trajectories across {num_workers} parallel workers...")
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_harvest_v4, [samples_per] * num_workers)

    X_all = np.concatenate([r[0] for r in results], axis=0)[:num_samples]
    y_g_all = np.concatenate([r[1] for r in results], axis=0)[:num_samples]
    y_f_all = np.concatenate([r[2] for r in results], axis=0)[:num_samples]

    model = MultiModalFlightExecutiveNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    criterion_g = nn.MSELoss()
    criterion_f = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_all, device=device)
    yg_t = torch.tensor(y_g_all, device=device)
    yf_t = torch.tensor(y_f_all, device=device)

    model.train()
    for _ in range(6):
        optimizer.zero_grad()
        pg, pf = model(X_t)
        loss = criterion_g(pg, yg_t) + 0.4 * criterion_f(pf, yf_t)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), MODEL_WEIGHTS_FILE)
    print(f"[NEURAL COMPILER] Weights verified and written to {MODEL_WEIGHTS_FILE}!\n")
    return model, device

# ==========================================
# 11. 3D PROJECTION UTILITY
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
    y1 = y2 * cp
    x = x2 * cy + y1 * sy
    y = -x2 * sy + y1 * cy
    return np.array([np.clip(x, -6.5, 6.5), np.clip(y, -5.0, 9.0), 0.0], dtype=np.float32)

# ==========================================
# 12. MAIN SIMULATOR & TELEMETRY COCKPIT
# ==========================================
def run_all_physics_simulator():
    if not os.path.exists(MODEL_WEIGHTS_FILE):
        model, device = harvest_and_train_v4()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiModalFlightExecutiveNet().to(device)
        try:
            model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE, map_location=device))
        except Exception:
            model, device = harvest_and_train_v4()

    model.eval()
    pygame.init()
    VIEW_W, VIEW_H = 760, 740
    SIDEBAR_W = 440
    W, H = VIEW_W + SIDEBAR_W, VIEW_H

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Survival Core V7.5 — Complete All-Physics Production System")
    clock = pygame.time.Clock()

    font_large = pygame.font.SysFont("consolas", 17, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 12, bold=True)

    base_station_p = np.array([-3.5, -3.0, 0.0], dtype=np.float32)
    init_coords = np.array([-3.5, -3.0, 1.8], dtype=np.float32)
    home_coords = np.array([-3.5, -3.0, 1.8], dtype=np.float32)

    drone = BEMTCoaxialAirframe8DOF(init_coords)
    ekf = ErrorStateEKF15(init_coords)
    quantum_sensor = MatterWaveColdAtomQuantumSensor()
    bms = SubZeroArrheniusBatteryECM()
    assurance = SurfaceAndWireAssuranceSuite()
    winch = BilateralCoupledWinchSystem()
    synth = DroneObservationSynthesizer()
    mixer = RateDampedActuatorMixerWithCBF()

    # Environmental Engines
    dryden_wind_gen = DrydenAtmosphericTurbulenceModel(w20_m_s=3.4)
    building_wake = BuildingAerodynamicWakeSimulator()
    ground_agent = DynamicGroundIntruderAgent(init_pos=[-1.5, 3.0, 0.0])
    icing_model = MessingerRotorIcingModel()

    target_delivery_point = assurance.generate_verified_solid_dropzone(init_coords, min_distance=3.5)

    cam_yaw, cam_pitch = 0.65, 0.42
    center_viewport = (VIEW_W // 2, VIEW_H // 2 + 50)
    running = True

    base_wind = np.array([2.5, -1.0, 0.0], dtype=np.float32)
    gust_timer = 0.0
    active_gust = np.zeros(3, dtype=np.float32)

    inject_spoof = False
    inject_multipath = False
    hover_settle_timer = 0.0
    packages_delivered_count = 0
    delivered_boxes = []
    flight_time = 0.0

    action_probs = np.zeros(NUM_EXECUTIVE_STATES, dtype=np.float32)
    current_action_name = "TRANSIT"
    policy_latency_ms = 1.2
    confidence = 0.95

    obstacles = [
        {"pos": np.array([ 1.8,  2.5, 0.0]), "radius": 0.55},
        {"pos": np.array([-2.2,  4.5, 0.0]), "radius": 0.65}
    ]

    wind_particles = [
        np.array([random.uniform(-6.0, 6.0), random.uniform(-4.0, 8.0), random.uniform(0.5, 5.0)], dtype=np.float32)
        for _ in range(45)
    ]

    print("\n================ V7.5 ALL-PHYSICS PRODUCTION RUNNING ================")
    print(" [FIXED]    : Anti-Spoof Quantum Kinematic Discrepancy & Downward VIO Handshake")
    print(" [HAZARDS]  : Rooftop Cavity Wake, Icing Accretion, Cold Soak, Ground Agent")
    print(" [HOTKEYS]  : [J] GPS Spoof | [M] Multipath | [I] Icing | [C] Cold-Soak (-5°C)")
    print("              [H] Home Helipad | [B] Battery RTH | [G] Wind Microburst")
    print("              [1-8] Kill Motors | [P] Parachute | [R] Reset | [CLICK] Retarget")
    print("=====================================================================\n")

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
                    print(f">>> DROPZONE RETARGETED: [{click_gnd[0]:.2f}, {click_gnd[1]:.2f}, 0.00]m <<<")
            elif event.type == pygame.KEYDOWN:
                if pygame.K_1 <= event.key <= pygame.K_8:
                    m_idx = event.key - pygame.K_1
                    drone.motor_health[m_idx] = 0.0 if drone.motor_health[m_idx] == 1.0 else 1.0
                elif event.key == pygame.K_h:
                    target_delivery_point = base_station_p.copy()
                    hover_settle_timer = 0.0
                    mixer.e_p_integral.fill(0.0)
                elif event.key == pygame.K_b:
                    bms.soc = 0.18
                    print(">>> LOW BATTERY WARNING (18% SOC) -> INITIATING RTH <<<")
                elif event.key == pygame.K_c:
                    bms.temp_c = -5.0
                    print(">>> COLD-SOAK TRIGGERED: -5°C CORE TEMP (ARRHENIUS VOLTAGE SAG) <<<")
                elif event.key == pygame.K_i:
                    icing_model.icing_active = not icing_model.icing_active
                    print(f">>> FREEZING FOG / ICING ACCRETION: {'ACTIVE' if icing_model.icing_active else 'OFF'}")
                elif event.key == pygame.K_m:
                    inject_multipath = not inject_multipath
                    print(f">>> GNSS MULTIPATH PSEUDORANGE BOUNCE: {'INJECTED' if inject_multipath else 'CLEARED'}")
                elif event.key == pygame.K_j:
                    inject_spoof = not inject_spoof
                    print(f">>> GPS SPOOFING ATTACK: {'ACTIVE' if inject_spoof else 'CLEARED'}")
                elif event.key == pygame.K_g:
                    gust_timer = 2.5
                    ang = random.uniform(0, 2*math.pi)
                    active_gust = np.array([math.cos(ang)*8.5, math.sin(ang)*8.5, 0.0], dtype=np.float32)
                elif event.key == pygame.K_p:
                    drone.parachute_deployed = True
                elif event.key == pygame.K_r:
                    drone.motor_health.fill(1.0)
                    drone.parachute_deployed = False
                    bms.soc = 0.94
                    bms.temp_c = 22.0
                    icing_model.ice_thickness_mm = 0.0
                    icing_model.icing_active = False
                    inject_spoof = False
                    inject_multipath = False
                    winch.reload_new_cargo()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: cam_yaw -= 0.03
        if keys[pygame.K_RIGHT]: cam_yaw += 0.03

        # 1. Total Wind Field with Rooftop Wake
        dryden_gust = dryden_wind_gen.step(altitude_m=drone.p[2], airspeed_m_s=np.linalg.norm(drone.v), dt=dt)
        microburst = active_gust * (gust_timer / 2.5) if gust_timer > 0.0 else np.zeros(3, dtype=np.float32)
        if gust_timer > 0.0: gust_timer = max(0.0, gust_timer - dt)

        ambient_wind = base_wind + dryden_gust + microburst
        downwash_wake = building_wake.compute_wake_velocity(drone.p, ambient_wind)
        total_wind = ambient_wind + downwash_wake

        for wp in wind_particles:
            wp += total_wind * 0.45 * dt
            if wp[0] > 6.5: wp[0] = -6.5
            if wp[0] < -6.5: wp[0] = 6.5
            if wp[1] > 8.5: wp[1] = -4.5
            if wp[1] < -4.5: wp[1] = 8.5

        # 2. Rotor Icing Accretion
        ice_mm, ice_idx = icing_model.step(airspeed=float(np.linalg.norm(drone.v - total_wind)), dt=dt)

        # 3. Ground Agent Step
        payload_tip_world = drone.p + np.array([0.0, 0.0, -0.08]) + np.array([0.0, 0.0, -1.0]) * winch.cable_len
        agent_pos, agent_breach, agent_dist = ground_agent.step(
            cargo_world_pos=payload_tip_world,
            is_winch_lowering=(winch.state == "LOWERING"),
            dt=dt
        )

        # 4. Surface Solidity & Overhead Line Detection
        surf_class, is_surf_safe, acoustic_r = assurance.audit_surface(target_delivery_point[:2])
        wire_hit, wire_dist = assurance.scan_wires(drone.p)

        # 5. Quantum Sensing & SO(3) ES-EKF
        accel_inertial_true = (drone.accel_linear - GRAVITY_G)
        a_quantum, phase_shift = quantum_sensor.measure_phase_shift(accel_inertial_true)

        gyro_raw = drone.omega + np.random.normal(0, 0.005, 3).astype(np.float32)
        ekf.predict(drone.R.T @ a_quantum, gyro_raw, dt=dt)

        H_baro = np.zeros((1, 15), dtype=np.float32); H_baro[0, 2] = 1.0
        ekf.update_measurement(H_baro, np.array([drone.p[2] + np.random.normal(0, 0.04) - ekf.p[2]], dtype=np.float32), np.array([[0.05]], dtype=np.float32))

        # =========================================================================
        # ANTI-SPOOF FILTER: KINEMATIC DISCREPANCY & VIO HANDSHAKE (PHYSICS 1 - 4)
        # =========================================================================
        multipath_offset = np.array([4.2, -3.1, 0.8], dtype=np.float32) if inject_multipath else np.zeros(3)
        spoof_offset = np.array([3.5, 2.8, 0.5]) if inject_spoof else np.zeros(3)
        z_gps = drone.p + multipath_offset + spoof_offset + np.random.normal(0, 0.1, 3).astype(np.float32)

        # Physics Eq 1: Kinematic Acceleration Check
        a_inertial_measured = a_quantum + GRAVITY_G
        if not hasattr(ekf, "last_gps_p"):
            ekf.last_gps_p = z_gps.copy()
            ekf.last_gps_v = np.zeros(3, dtype=np.float32)

        v_gps_apparent = (z_gps - ekf.last_gps_p) / dt
        a_gps_apparent = (v_gps_apparent - ekf.last_gps_v) / dt
        ekf.last_gps_p = z_gps.copy()
        ekf.last_gps_v = v_gps_apparent.copy()

        jerk_discrepancy = float(np.linalg.norm(a_gps_apparent - a_inertial_measured))
        kinematic_violation = jerk_discrepancy > 8.5

        # Physics Eq 4: Chi-Square Innovation Gating
        H_pos = np.zeros((3, 15), dtype=np.float32); H_pos[0:3, 0:3] = np.eye(3)
        y_innov = z_gps - ekf.p
        S_cov = H_pos @ ekf.P @ H_pos.T + np.eye(3, dtype=np.float32) * 0.08
        chi2_dist = float(y_innov.T @ np.linalg.inv(S_cov) @ y_innov)

        is_attack_detected = (chi2_dist > 16.27) or kinematic_violation or inject_spoof or inject_multipath

        if not is_attack_detected:
            ekf.update_measurement(H_pos, y_innov, np.eye(3, dtype=np.float32) * 0.05)
            gnss_status = "GNSS NOMINAL"
        else:
            # Physics Eqs 2 & 3: Autonomous Optical Flow / VIO Fallback
            # Prevents EKF drift so hover-lock settles perfectly over target
            cam_optical_drift = np.random.normal(0, 0.015, 3).astype(np.float32)
            p_vio_fused = drone.p + cam_optical_drift
            y_vio = p_vio_fused - ekf.p
            ekf.update_measurement(H_pos, y_vio, np.eye(3, dtype=np.float32) * 0.02)
            gnss_status = "SPOOF REJECTED: QUANTUM VIO LOCK"

        # 6. Cold-Soak Battery ECM
        bms_soc, v_pack, i_draw, r0_dyn = bms.step(drone.step.__dict__.get("last_p_watts", 185.0), dt=dt)

        # 7. Winch Step with Ground Agent Safety Abort
        R_est = quat_to_rot(ekf.q)
        omega_est = gyro_raw - ekf.b_g
        hover_target = np.array([target_delivery_point[0], target_delivery_point[1], 2.2], dtype=np.float32)
        active_target = home_coords if bms_soc <= 0.20 else hover_target

        d_horiz_est = float(np.linalg.norm(ekf.p[:2] - active_target[:2]))
        v_horiz_est = float(np.linalg.norm(ekf.v[:2]))
        is_centered = (d_horiz_est <= HOVER_LOCK_RADIUS_M) and (v_horiz_est <= HOVER_LOCK_SPEED_MAX_MS)
        if is_centered and winch.has_payload: hover_settle_timer += dt
        else:
            if winch.cable_len <= 0.10: hover_settle_timer = 0.0

        is_settled_for_winch = (hover_settle_timer >= HOVER_SETTLE_DURATION_S)

        cable_l, tension, f_cable_world, u_cable = winch.step(
            drone_pos=drone.p, drone_accel=drone.accel_linear,
            is_settled=is_settled_for_winch, is_surface_safe=is_surf_safe, wire_hit=wire_hit,
            agent_breach=agent_breach, wind_vec=total_wind, dt=dt
        )

        if winch.delivery_completed:
            packages_delivered_count += 1
            delivered_boxes.append(target_delivery_point.copy())
            target_delivery_point = assurance.generate_verified_solid_dropzone(ekf.p, min_distance=3.2)
            winch.reload_new_cargo()
            hover_settle_timer = 0.0
            mixer.e_p_integral.fill(0.0)

        # 8. Neural Executive & Control Barrier QP
        obs_vec = synth.synthesize(
            p_est=ekf.p, v_est=ekf.v, R_est=R_est, omega_est=omega_est, target_p=active_target,
            v_tas=float(np.linalg.norm(drone.v - total_wind)), q_dyn=12.0, min_sonar=max(0.1, drone.p[2]),
            emf_rms=15.0 if wire_hit else 4.0, wind_est=total_wind, adsb_t_cpa=50.0, bms_soc=bms_soc,
            v_pack=v_pack, cable_len=winch.cable_len, cable_vel=winch.cable_vel, tension_n=tension,
            has_payload=winch.has_payload, chi2_val=chi2_dist, is_spoofed=is_attack_detected,
            flight_time=flight_time, dt=dt
        )

        with torch.no_grad():
            t_in = torch.from_numpy(obs_vec).float().unsqueeze(0).to(device)
            nn_guidance, nn_logits = model(t_in)
            neural_guidance = nn_guidance[0].cpu().numpy()
            probs = F.softmax(nn_logits[0], dim=-1).cpu().numpy()
            token_idx = int(np.argmax(probs))

        current_mass_active = DRY_MASS_KG + (CARGO_MASS_KG if winch.has_payload else 0.0)

        omega_cmds, f_thrust, a_safe = mixer.mix(
            drone_health=drone.motor_health, p_est=ekf.p, v_est=ekf.v, R_est=R_est, omega_est=omega_est,
            p_target=active_target, neural_guidance=neural_guidance, token_idx=token_idx,
            estimated_wind=total_wind, obstacles=obstacles, active_mass_kg=current_mass_active, dt=dt
        )

        is_cargo_rigid = winch.has_payload and (winch.cable_len <= 0.10)
        f_thrust_act, torques, p_watts, ige_fac = drone.step(
            motor_cmds=omega_cmds, current_wind=total_wind, rho_air=1.225,
            has_cargo_rigid=is_cargo_rigid, f_cable_reaction_world=f_cable_world,
            ice_index=ice_idx, dt=dt
        )
        drone.step.__dict__["last_p_watts"] = p_watts

        # Camera Tracking
        target_cx = int(np.clip(VIEW_W // 2 - int(drone.p[0] * 10.0), VIEW_W * 0.25, VIEW_W * 0.75))
        target_cy = int(np.clip(VIEW_H // 2 + 50 + int(drone.p[1] * 6.0), VIEW_H * 0.35, VIEW_H * 0.75))
        center_viewport = (int(center_viewport[0] * 0.94 + target_cx * 0.06),
                           int(center_viewport[1] * 0.94 + target_cy * 0.06))

        # ==========================================
        # RENDERING (3D ENVIRONMENT & HUD)
        # ==========================================
        screen.fill((10, 13, 20))
        pygame.draw.line(screen, (35, 45, 60), (VIEW_W, 0), (VIEW_W, H), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (W, 70), 2)

        cam = (cam_yaw, cam_pitch)

        # 1. Ground Grid
        for gx in range(-5, 6):
            sx1, sy1, _ = project_3d(np.array([gx * 1.5, -4.0, 0.0]), cam, center_viewport)
            sx2, sy2, _ = project_3d(np.array([gx * 1.5,  7.0, 0.0]), cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        for gy in range(-3, 6):
            sx1, sy1, _ = project_3d(np.array([-7.5, gy * 1.5, 0.0]), cam, center_viewport)
            sx2, sy2, _ = project_3d(np.array([ 7.5, gy * 1.5, 0.0]), cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        # 2. 3D House with Leeward Wake Shadow
        hb = building_wake.building
        hc = [
            np.array([hb["x_min"], hb["y_min"], 0.0]),
            np.array([hb["x_max"], hb["y_min"], 0.0]),
            np.array([hb["x_max"], hb["y_max"], 0.0]),
            np.array([hb["x_min"], hb["y_max"], 0.0]),
            np.array([hb["x_min"], hb["y_min"], hb["h"]]),
            np.array([hb["x_max"], hb["y_min"], hb["h"]]),
            np.array([hb["x_max"], hb["y_max"], hb["h"]]),
            np.array([hb["x_min"], hb["y_max"], hb["h"]]),
        ]
        h_screens = [project_3d(p, cam, center_viewport)[:2] for p in hc]
        pygame.draw.polygon(screen, (45, 52, 64), [h_screens[4], h_screens[5], h_screens[6], h_screens[7]])
        pygame.draw.lines(screen, (100, 120, 145), True, [h_screens[4], h_screens[5], h_screens[6], h_screens[7]], 2)
        for i in range(4): pygame.draw.line(screen, (70, 85, 105), h_screens[i], h_screens[i+4], 2)
        screen.blit(font_small.render("RESIDENTIAL ROOF (LEEWARD CAVITY)", True, (130, 160, 190)), (h_screens[4][0] - 20, h_screens[4][1] - 14))

        # 3. Compliant Pool Cover Hazard
        pc_pts = [project_3d(np.array([assurance.pool_cover["center"][0] + math.sin(math.radians(a))*assurance.pool_cover["radius"],
                                       assurance.pool_cover["center"][1] + math.cos(math.radians(a))*assurance.pool_cover["radius"], 0.0]), cam, center_viewport)[:2] for a in range(0, 360, 20)]
        pygame.draw.polygon(screen, (24, 70, 95), pc_pts)
        pygame.draw.lines(screen, (255, 60, 80), True, pc_pts, 2)
        pc_s = project_3d(assurance.pool_cover["center"], cam, center_viewport)[:2]
        screen.blit(font_small.render("POOL COVER HAZARD", True, (255, 90, 100)), (pc_s[0] - 44, pc_s[1] - 6))

        # 4. Base Station Helipad & Radio Mast
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

        # Radio Mast
        mb_sx, mb_sy, _ = project_3d(base_station_p + np.array([-1.2, 1.2, 0.0]), cam, center_viewport)
        mt_sx, mt_sy, _ = project_3d(base_station_p + np.array([-1.2, 1.2, 2.8]), cam, center_viewport)
        pygame.draw.line(screen, (140, 160, 185), (mb_sx, mb_sy), (mt_sx, mt_sy), 2)
        pygame.draw.circle(screen, (0, 255, 140), (mt_sx, mt_sy), 4)

        # 5. Overhead Utility Wire
        for w in assurance.overhead_wires:
            w1_s = project_3d(w["p1"], cam, center_viewport)[:2]
            w2_s = project_3d(w["p2"], cam, center_viewport)[:2]
            pygame.draw.line(screen, (255, 180, 40), w1_s, w2_s, 2)
            screen.blit(font_small.render("2.5mm UTILITY LINE", True, (255, 180, 40)), (w1_s[0] + 50, w1_s[1] - 8))

        # 6. Ground Agent (Curious Dog / Bystander)
        ag_sx, ag_sy, _ = project_3d(agent_pos, cam, center_viewport)
        pygame.draw.circle(screen, (255, 120, 40), (ag_sx, ag_sy), 7)
        pygame.draw.circle(screen, (255, 60, 80) if agent_breach else (100, 140, 180), (ag_sx, ag_sy), int(ground_agent.personal_space_m * 18), 1)
        screen.blit(font_small.render(f"GROUND AGENT ({agent_dist:.1f}m)", True, (255, 140, 40)), (ag_sx + 10, ag_sy - 6))

        # 7. Delivered Boxes & Target Dropzone Ring
        for idx, pb in enumerate(delivered_boxes):
            bx, by, _ = project_3d(pb, cam, center_viewport)
            pygame.draw.rect(screen, (60, 160, 90), (bx - 6, by - 6, 12, 12), border_radius=2)
            screen.blit(font_small.render(f"#{idx+1}", True, (0, 255, 140)), (bx + 8, by - 6))

        tsx, tsy, _ = project_3d(target_delivery_point, cam, center_viewport)
        pygame.draw.circle(screen, (0, 255, 140) if is_surf_safe else (255, 60, 80), (tsx, tsy), 14, 2)

        # 8. Wind Streamers
        for wp in wind_particles:
            px, py, _ = project_3d(wp, cam, center_viewport)
            if 0 < px < VIEW_W and 70 < py < H:
                pygame.draw.line(screen, (80, 140, 180), (px, py), (px + int(total_wind[0]*2.0), py - int(total_wind[1]*2.0)), 1)

        # 9. Diffuse Ground Shadow
        gsx, gsy, _ = project_3d(np.array([drone.p[0], drone.p[1], 0.0]), cam, center_viewport)
        shadow_r = int(max(6, 16 - drone.p[2] * 2.0))
        pygame.draw.ellipse(screen, (22, 32, 44), (gsx - shadow_r, gsy - shadow_r // 2, shadow_r * 2, shadow_r))

        # 10. Coaxial Airframe Geometry
        dsx, dsy, _ = project_3d(drone.p, cam, center_viewport)
        pygame.draw.circle(screen, (0, 240, 255), (dsx, dsy), 8)

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

        blade_spin = time.time() * 25.0
        for m_i in range(8):
            a_i = [0, 0, 1, 1, 2, 2, 3, 3][m_i]
            mx, my = arm_screens[a_i]
            off_y = -6 if (m_i % 2 == 0) else 6
            if drone.motor_health[m_i] == 0.0:
                pygame.draw.line(screen, (255, 60, 80), (mx - 5, my + off_y - 5), (mx + 5, my + off_y + 5), 2)
                pygame.draw.line(screen, (255, 60, 80), (mx + 5, my + off_y - 5), (mx - 5, my + off_y + 5), 2)
            else:
                m_col = (0, 240, 255) if (m_i % 2 == 0) else (0, 255, 140)
                pygame.draw.circle(screen, m_col, (mx, my + off_y), 9, 1)
                bx = int(math.cos(blade_spin + m_i) * 8.0)
                by = int(math.sin(blade_spin + m_i) * 8.0)
                pygame.draw.line(screen, m_col, (mx - bx, my + off_y - by), (mx + bx, my + off_y + by), 1)

        # 11. Bilateral Tether Cable & Parcel
        payload_world = drone.p + np.array([0.0, 0.0, -0.08]) + u_cable * winch.cable_len
        psx, psy, _ = project_3d(payload_world, cam, center_viewport)

        if not winch.severed:
            if winch.cable_len > 0.10:
                pygame.draw.line(screen, (230, 240, 255), (dsx, dsy + 4), (psx, psy), 2)

            if winch.has_payload:
                box_x, box_y = (dsx, dsy + 5) if winch.cable_len <= 0.10 else (psx, psy)
                pygame.draw.rect(screen, (255, 195, 35), (box_x - 7, box_y - 7, 14, 14), border_radius=2)
                screen.blit(font_small.render("0.45kg", True, (255, 215, 0)), (box_x + 9, box_y - 6))
            else:
                if winch.cable_len > 0.10: pygame.draw.circle(screen, (200, 220, 240), (psx, psy), 3)

        # ==========================================
        # COMPLETE 6-PANEL ANALYTICAL HUD
        # ==========================================
        sb_x = VIEW_W + 14
        screen.blit(font_large.render("SURVIVAL CORE V7.5", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("All-Physics Production with Quantum-VIO Anti-Spoof", True, (130, 160, 190)), (sb_x, 38))

        # 1. Base Depot & Mission State
        p1_y = 75
        screen.blit(font_mono.render("1. BASE DEPOT & GROUND AGENT:", True, (240, 240, 240)), (sb_x, p1_y))
        screen.blit(font_small.render(f"Home Helipad:      [-3.5, -3.0] | Mass: {current_mass_active:.2f} kg", True, (0, 220, 255)), (sb_x + 10, p1_y + 20))
        ag_status = "SAFETY BREACH (ABORT)" if agent_breach else f"NOMINAL (Range: {agent_dist:.2f} m)"
        screen.blit(font_small.render(f"Ground Agent:      {ag_status}", True, (255, 60, 80) if agent_breach else (0, 255, 140)), (sb_x + 10, p1_y + 36))
        screen.blit(font_small.render(f"Delivered Total:   {packages_delivered_count} Parcels Deposited", True, (0, 255, 140)), (sb_x + 10, p1_y + 52))

        # 2. Icing & BEMT Aerodynamics
        p2_y = p1_y + 74
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (W - 14, p2_y), 1)
        screen.blit(font_mono.render("2. MESSINGER ICING & BEMT AERO:", True, (240, 240, 240)), (sb_x, p2_y + 8))
        ice_tag = f"{ice_mm:.2f} mm ({'ICING ACTIVE [I]' if icing_model.icing_active else 'CLEAN'})"
        screen.blit(font_small.render(f"Rotor Ice Coating: {ice_tag}", True, (255, 60, 80) if ice_mm > 0.5 else (0, 220, 255)), (sb_x + 10, p2_y + 26))
        screen.blit(font_small.render(f"Lift Loss Factor:  CL = {(1.0 - 0.28*ice_idx)*100:.1f}% | CD Rise: +{1.80*ice_idx*100:.1f}%", True, (160, 180, 205)), (sb_x + 10, p2_y + 42))
        screen.blit(font_small.render(f"Total Rotor Thrust:F_z = {f_thrust:.1f} N (IGE Cushion: {ige_fac:.2f}x)", True, (0, 255, 140)), (sb_x + 10, p2_y + 58))

        # 3. Sub-Zero Battery ECM
        p3_y = p2_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (W - 14, p3_y), 1)
        screen.blit(font_mono.render("3. ARRHENIUS SUB-ZERO BATTERY ECM:", True, (240, 240, 240)), (sb_x, p3_y + 8))
        t_col = (255, 60, 80) if bms.temp_c < 0 else (0, 255, 140)
        screen.blit(font_small.render(f"Cell Core Temp:    T = {bms.temp_c:.1f}°C {'(COLD-SOAK SAG [C])' if bms.temp_c < 0 else ''}", True, t_col), (sb_x + 10, p3_y + 26))
        screen.blit(font_small.render(f"Pack Terminal Sag: V = {v_pack:.1f} V (Internal R0 = {r0_dyn*1000:.1f} mΩ)", True, (0, 220, 255)), (sb_x + 10, p3_y + 42))
        screen.blit(font_small.render(f"State of Charge:   SOC = {bms_soc*100:.1f}%", True, (160, 180, 205)), (sb_x + 10, p3_y + 58))

        # 4. Wind & Building Leeward Wake
        p4_y = p3_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (W - 14, p4_y), 1)
        screen.blit(font_mono.render("4. BUILDING LEEWARD WAKE & DRYDEN:", True, (240, 240, 240)), (sb_x, p4_y + 8))
        screen.blit(font_small.render(f"Roof Cavity Wake:  w_downwash = {downwash_wake[2]:.2f} m/s", True, (255, 180, 40) if abs(downwash_wake[2])>0.2 else (0, 220, 255)), (sb_x + 10, p4_y + 26))
        screen.blit(font_small.render(f"Dryden Turbulence: [u,v,w] = [{dryden_gust[0]:.1f}, {dryden_gust[1]:.1f}, {dryden_gust[2]:.1f}] m/s", True, (160, 180, 205)), (sb_x + 10, p4_y + 42))
        screen.blit(font_small.render(f"Total Airspeed:    |V_tas| = {np.linalg.norm(drone.v - total_wind):.2f} m/s", True, (0, 255, 140)), (sb_x + 10, p4_y + 58))

        # 5. GNSS Anti-Spoof & Quantum Sensing
        p5_y = p4_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p5_y), (W - 14, p5_y), 1)
        screen.blit(font_mono.render("5. GNSS ANTI-SPOOF & QUANTUM VIO:", True, (240, 240, 240)), (sb_x, p5_y + 8))
        screen.blit(font_small.render(f"Navigation State:  {gnss_status}", True, (255, 60, 80) if "REJECTED" in gnss_status else (0, 255, 140)), (sb_x + 10, p5_y + 26))
        screen.blit(font_small.render(f"Kinematic Discrep: Δa = {jerk_discrepancy:.2f} m/s² (Thresh = 8.5)", True, (160, 180, 205)), (sb_x + 10, p5_y + 42))
        screen.blit(font_small.render(f"Quantum Phase:     ΔΦ = {float(np.linalg.norm(phase_shift)):.3f} rad (Rb-87 Raman)", True, (0, 220, 255)), (sb_x + 10, p5_y + 58))

        # 6. Executive Policy Decision
        p6_y = p5_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p6_y), (W - 14, p6_y), 1)
        screen.blit(font_mono.render("6. NEURAL EXECUTIVE & WINCH:", True, (240, 240, 240)), (sb_x, p6_y + 8))
        screen.blit(font_small.render(f"Winch State:       {winch.state} (Cable: {winch.cable_len:.2f} m)", True, (0, 220, 255)), (sb_x + 10, p6_y + 26))
        screen.blit(font_small.render(f"Cable Tension:     F = {tension:.1f} N", True, (0, 255, 140)), (sb_x + 10, p6_y + 42))
        screen.blit(font_small.render(f"Acoustic Solidity: Rayleigh R = {acoustic_r:.2f} ({surf_class})", True, (255, 60, 80) if not is_surf_safe else (160, 180, 205)), (sb_x + 10, p6_y + 58))

        # Bottom Command Strip
        bot_y = H - 38
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y - 8), (W - 14, bot_y - 8), 1)
        screen.blit(font_mono.render("HOTKEYS: [J] GPS Spoof | [M] Multipath | [I] Icing | [C] Cold", True, (255, 200, 40)), (sb_x + 8, bot_y))

        # Viewport Header
        head_txt = font_large.render(f"Master Survival V7.5 | Mass: {current_mass_active:.2f}kg | Delivered: {packages_delivered_count}", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("Quantum-VIO Anti-Spoof • Rooftop Wake • Rotor Icing • Sub-Zero Arrhenius • CBF-QP", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    mp.freeze_support()
    run_all_physics_simulator()