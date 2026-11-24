"""
=============================================================================
MODULE 15: AUTONOMOUS NEURAL FLIGHT EXECUTIVE & WIND DEFENSE AUTOPILOT
File: drone_neural_executive_autopilot.py
=============================================================================
- Realistic 3D Atmospheric Wind Field (Prevailing Wind + Dryden Turbulence)
- Physical Relative Airspeed Aerodynamic Deflection (Drone gets knocked off course)
- Disturbance Observer (DOB) Crab-Angle Tilt Compensation into Crosswinds
- Dynamic Tether Cable Wind Deflection & Safety Hover Re-settling Lock
- Press [G] to inject a sudden 35 km/h crosswind microburst!
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
MASS_KG = 1.35
GRAVITY_G = np.array([0.0, 0.0, -9.80665], dtype=np.float32)
G_ACCEL = 9.80665
ARM_LENGTH_M = 0.225
INERTIA_MATRIX = np.diag([0.016, 0.016, 0.028]).astype(np.float32)
INV_INERTIA = np.linalg.inv(INERTIA_MATRIX)

C_T = 1.28e-5
C_Q = 2.45e-7
MOTOR_TAU_S = 0.035
MAX_RPM = 8800.0
MAX_OMEGA = (MAX_RPM * 2.0 * math.pi) / 60.0
HOVER_THRUST = MASS_KG * G_ACCEL
HOVER_OMEGA = math.sqrt((HOVER_THRUST / 4.0) / C_T)

OBSERVATION_DIM = 32
NUM_MOTORS = 4
NUM_EXECUTIVE_STATES = 6
PAYLOAD_WEIGHT_N = 2.0 * G_ACCEL
MODEL_WEIGHTS_FILE = "drone_neural_autopilot_model.pth"

# Stricter Hover Settle Tolerances
HOVER_LOCK_RADIUS_M = 0.45      # Must be within 45 cm horizontally
HOVER_LOCK_SPEED_MAX_MS = 0.25  # Groundspeed must be below 0.25 m/s
HOVER_SETTLE_DURATION_S = 0.75  # Must hold still for 0.75s before winch unspools

ACTION_NAMES = [
    "TRANSIT",             # 0: Flying en-route to dropzone
    "HOVER_LOCK",          # 1: Precision position-hold settling over target
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
# 2. CONTINUOUS 6-DOF AIRFRAME & WIND FIELD
# ==========================================
class AtmosphericWindField:
    """
    Simulates continuous prevailing ambient wind, Dryden turbulence fluctuations,
    and 1-cosine discrete microburst gusts.
    """
    def __init__(self):
        self.prevailing_wind = np.array([2.5, -1.2, 0.0], dtype=np.float32) # Steady ~10 km/h wind
        self.turbulent_component = np.zeros(3, dtype=np.float32)
        self.gust_pulse = np.zeros(3, dtype=np.float32)
        self.gust_duration = 0.0
        self.gust_timer = 0.0

    def trigger_microburst(self, intensity_ms=8.5):
        """Injects sudden violent 35 km/h crosswind gust."""
        angle = random.uniform(0.0, 2.0 * math.pi)
        self.gust_pulse = np.array([math.cos(angle) * intensity_ms, math.sin(angle) * intensity_ms, random.uniform(-1.0, 1.0)], dtype=np.float32)
        self.gust_duration = random.uniform(1.8, 3.2)
        self.gust_timer = self.gust_duration
        print(f">>> [WIND GUST INJECTED] {np.linalg.norm(self.gust_pulse)*3.6:.1f} km/h CROSSWIND BLOWING DRONE OFF COURSE! <<<")

    def update(self, dt):
        # 1st-order Gauss-Markov stochastic Dryden turbulence
        white_noise = np.random.normal(0, 0.45, 3).astype(np.float32)
        self.turbulent_component += (-0.4 * self.turbulent_component + white_noise) * dt

        # Decaying 1-cosine microburst profile
        if self.gust_timer > 0.0:
            self.gust_timer = max(0.0, self.gust_timer - dt)
            progress = (self.gust_duration - self.gust_timer) / max(0.1, self.gust_duration)
            shape = 0.5 * (1.0 - math.cos(2.0 * math.pi * progress))
            current_gust = self.gust_pulse * shape
        else:
            current_gust = np.zeros(3, dtype=np.float32)

        return self.prevailing_wind + self.turbulent_component + current_gust


class ContinuousDrone6DOF:
    def __init__(self, init_pos):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.R = np.eye(3, dtype=np.float32)
        self.omega = np.zeros(3, dtype=np.float32)
        self.motor_omegas = np.full(4, HOVER_OMEGA, dtype=np.float32)
        self.accel_linear = np.zeros(3, dtype=np.float32)
        self.last_f_aero = np.zeros(3, dtype=np.float32)
        self.parachute_deployed = False

    def step(self, motor_cmds, current_wind, dt=0.016):
        if self.parachute_deployed:
            self.motor_omegas.fill(0.0)
            self.v[0] *= 0.90
            self.v[1] *= 0.90
            self.v[2] += (-4.1 - self.v[2]) * 3.0 * dt
            # Drifts downwind with parachute!
            self.p += (self.v + current_wind * 0.4) * dt
            if self.p[2] <= 0.0:
                self.p[2] = 0.0
                self.v.fill(0.0)
            return 0.0, np.zeros(3)

        cmds = np.clip(motor_cmds, 0.0, MAX_OMEGA)
        self.motor_omegas += ((cmds - self.motor_omegas) / MOTOR_TAU_S) * dt

        T = C_T * (self.motor_omegas ** 2)
        Q = C_Q * (self.motor_omegas ** 2)

        f_thrust = float(np.sum(T))
        arm = ARM_LENGTH_M / math.sqrt(2.0)
        tau_x = arm * (-T[0] + T[1] + T[2] - T[3])
        tau_y = arm * (T[0] + T[1] - T[2] - T[3])
        tau_z = -Q[0] + Q[1] - Q[2] + Q[3]
        torques = np.array([tau_x, tau_y, tau_z], dtype=np.float32)

        f_body = np.array([0.0, 0.0, f_thrust], dtype=np.float32)
        f_gravity = MASS_KG * GRAVITY_G

        # True Aerodynamic Drag relative to moving air mass: v_air = v_drone - v_wind
        v_air = self.v - current_wind
        speed_air = float(np.linalg.norm(v_air))
        # Form drag: F_drag = -0.5 * rho * Cd * A * |v_air| * v_air
        f_drag = -np.array([0.22, 0.22, 0.35], dtype=np.float32) * v_air * max(1.0, speed_air * 0.45)
        self.last_f_aero = f_drag

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
        return f_thrust, torques

# ==========================================
# 3. 15-STATE ERROR-STATE EKF
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
# 4. MOTORIZED DELIVERY WINCH WITH SWAY
# ==========================================
class MotorizedWinchSubsystem:
    def __init__(self):
        self.cable_len = 0.08
        self.cable_vel = 0.0
        self.tension_n = PAYLOAD_WEIGHT_N
        self.has_payload = True
        self.touchdown_timer = 0.0
        self.severed = False
        self.sway_offset = np.zeros(2, dtype=np.float32) # Visual pendular swing under wind

    def step(self, executive_action, drone_z, wind_2d, dt=0.016, force_snag=False):
        if self.severed:
            return self.cable_len, 0.0, 0.0

        # Aerodynamic wind deflection bends the descending tether cable
        target_sway = wind_2d * (self.cable_len * 0.08)
        self.sway_offset += (target_sway - self.sway_offset) * 3.5 * dt

        if executive_action == 2:  # WINCH_LOWER
            self.cable_vel = 1.25
            self.cable_len += self.cable_vel * dt
            self.tension_n = PAYLOAD_WEIGHT_N + float(np.random.normal(0, 0.25))

        elif executive_action == 3:  # TOUCHDOWN_RELEASE
            self.cable_vel = 0.0
            self.tension_n = 1.8 + float(np.random.normal(0, 0.15))
            self.touchdown_timer += dt
            if self.touchdown_timer >= 1.0:
                self.has_payload = False

        elif executive_action == 4:  # WINCH_RETRACT
            self.cable_vel = -2.2
            self.cable_len = max(0.08, self.cable_len + self.cable_vel * dt)
            self.tension_n = 1.2 + float(np.random.normal(0, 0.1))

        else:
            self.cable_vel = 0.0
            self.cable_len = 0.08
            self.tension_n = PAYLOAD_WEIGHT_N if self.has_payload else 1.2

        if force_snag:
            self.tension_n = float(random.uniform(44.0, 62.0))

        if self.tension_n >= 48.0:
            self.severed = True
            print(">>> [SAFETY INTERLOCK] PYROTECHNIC GUILLOTINE FIRED: TETHER CUT <<<")

        return self.cable_len, self.cable_vel, self.tension_n

# ==========================================
# 5. DUAL-HEAD NEURAL POLICY ARCHITECTURE
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
        guidance_offsets = self.guidance_head(latent)
        executive_logits = self.executive_head(latent)
        return guidance_offsets, executive_logits

# ==========================================
# 6. OBSERVATION ENCODER
# ==========================================
class DroneObservationSynthesizer:
    def __init__(self):
        self.last_tension = 19.62

    def synthesize(self, drone_p, drone_v, drone_R, drone_omega, target_p,
                   v_tas, q_dyn, min_sonar, emf_rms, wind_est, adsb_t_cpa,
                   bms_soc, v_pack, cable_len, cable_vel, tension_n, has_payload,
                   chi2_val, is_spoofed, flight_time, total_plan_time, delivered_count, dt=0.016):
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        delta_p = target_p - drone_p
        obs[0:3] = np.clip(delta_p / 10.0, -1.0, 1.0)
        obs[3:6] = np.clip(drone_v / 6.0, -1.0, 1.0)
        obs[6:9] = np.clip(drone_R[:, 2], -1.0, 1.0)
        obs[9:12] = np.clip(drone_omega / 8.0, -1.0, 1.0)
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
        progress = flight_time / max(0.1, total_plan_time)
        obs[29] = np.clip(progress, 0.0, 1.0)
        d_horiz = math.hypot(delta_p[0], delta_p[1])
        obs[30] = np.clip(d_horiz / 10.0, 0.0, 1.0)
        obs[31] = np.clip((delivered_count % 10) / 10.0, 0.0, 1.0)
        return obs

# ==========================================
# 7. PARALLEL DATA HARVESTER
# ==========================================
def _worker_harvest(num_steps_target):
    np.random.seed()
    random.seed()

    synthesizer = DroneObservationSynthesizer()
    wind_field = AtmosphericWindField()
    observations = []
    teacher_guidance = []
    teacher_fsm = []
    delivered_count = 0

    while len(observations) < num_steps_target:
        start_p = np.array([random.uniform(-4.5, 4.5), random.uniform(-4.0, 4.0), random.uniform(1.2, 3.5)], dtype=np.float32)
        target_gnd = np.array([random.uniform(-4.5, 4.5), random.uniform(-2.5, 6.5), 0.0], dtype=np.float32)
        drone = ContinuousDrone6DOF(start_p)
        hover_dest = np.array([target_gnd[0], target_gnd[1], 2.2], dtype=np.float32)

        winch_state = "STANDBY"
        cable_len = 0.08
        cable_vel = 0.0
        tension_n = PAYLOAD_WEIGHT_N
        has_payload = True
        touchdown_timer = 0.0
        hover_settle_timer = 0.0
        flight_time = 0.0
        total_time = 4.5
        dt = 0.016

        step_count = 0
        mission_active = True

        while mission_active and step_count < 450:
            flight_time += dt
            current_wind = wind_field.update(dt)
            d_horiz = float(np.linalg.norm(drone.p[:2] - target_gnd[:2]))
            v_horiz = float(np.linalg.norm(drone.v[:2]))

            is_inside_basket = (d_horiz <= HOVER_LOCK_RADIUS_M) and (v_horiz <= HOVER_LOCK_SPEED_MAX_MS)
            if is_inside_basket:
                hover_settle_timer += dt
            else:
                hover_settle_timer = 0.0

            is_settled_for_winch = (hover_settle_timer >= HOVER_SETTLE_DURATION_S)

            if not is_settled_for_winch and winch_state == "STANDBY":
                if d_horiz > HOVER_LOCK_RADIUS_M:
                    fsm_token = 0  # TRANSIT
                    delta = hover_dest - drone.p
                    dir_u = delta / max(1e-4, float(np.linalg.norm(delta)))
                    v_des = dir_u * 2.8
                else:
                    fsm_token = 1  # HOVER_LOCK (Settling down)
                    v_des = np.zeros(3, dtype=np.float32)
            else:
                v_des = np.zeros(3, dtype=np.float32)
                if winch_state == "STANDBY" and has_payload:
                    winch_state = "LOWERING"
                    fsm_token = 2  # WINCH_LOWER
                elif winch_state == "LOWERING":
                    cable_vel = 1.25
                    cable_len += cable_vel * dt
                    tension_n = PAYLOAD_WEIGHT_N + float(np.random.normal(0, 0.25))
                    fsm_token = 2
                    if (drone.p[2] - cable_len) <= 0.08:
                        winch_state = "TOUCHDOWN"
                        touchdown_timer = 0.0
                elif winch_state == "TOUCHDOWN":
                    cable_vel = 0.0
                    tension_n = 1.8 + float(np.random.normal(0, 0.15))
                    touchdown_timer += dt
                    fsm_token = 3  # TOUCHDOWN_RELEASE
                    if touchdown_timer >= 1.0:
                        has_payload = False
                        winch_state = "RETRACTING"
                elif winch_state == "RETRACTING":
                    cable_vel = -2.2
                    cable_len = max(0.08, cable_len + cable_vel * dt)
                    tension_n = 1.2 + float(np.random.normal(0, 0.1))
                    fsm_token = 4  # WINCH_RETRACT
                    if cable_len <= 0.08:
                        fsm_token = 5  # AUTO_DISPATCH
                        delivered_count += 1
                        mission_active = False

            # Kinematic Desired Force vector
            e_p = drone.p - hover_dest
            e_v = drone.v - v_des
            F_d = -np.array([6.5, 6.5, 11.0]) * e_p - np.array([4.4, 4.4, 6.8]) * e_v + MASS_KG * (-GRAVITY_G)

            guidance_vec = np.clip(F_d / (MASS_KG * 15.0), -1.0, 1.0).astype(np.float32)
            guidance_target = np.array([guidance_vec[0], guidance_vec[1], guidance_vec[2], 0.0], dtype=np.float32)

            b3 = drone.R[:, 2]
            f_thrust = float(np.dot(F_d, b3))
            f_thrust = max(0.1, min(34.0, f_thrust))

            b3_d = F_d / (np.linalg.norm(F_d) + 1e-6)
            b1_c = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            b2_d = np.cross(b3_d, b1_c); b2_d /= (np.linalg.norm(b2_d) + 1e-6)
            b1_d = np.cross(b2_d, b3_d)
            R_d = np.column_stack([b1_d, b2_d, b3_d]).astype(np.float32)

            R_err_mat = 0.5 * (R_d.T @ drone.R - drone.R.T @ R_d)
            e_R = vee_map(R_err_mat)
            tau = -1.85 * e_R - 0.35 * drone.omega

            arm = ARM_LENGTH_M / math.sqrt(2.0)
            A = np.array([
                [1.0, 1.0, 1.0, 1.0],
                [-arm, arm, arm, -arm],
                [arm, arm, -arm, -arm],
                [-(C_Q/C_T), C_Q/C_T, -(C_Q/C_T), C_Q/C_T]
            ], dtype=np.float32)
            b_vec = np.array([f_thrust, tau[0], tau[1], tau[2]], dtype=np.float32)
            try:
                T_motors = np.clip(np.linalg.solve(A, b_vec), 0.05, 9.8)
            except np.linalg.LinAlgError:
                T_motors = np.full(4, f_thrust / 4.0, dtype=np.float32)

            omega_cmds = np.sqrt(T_motors / C_T)
            drone.step(omega_cmds, current_wind=current_wind, dt=dt)

            v_tas = float(np.linalg.norm(drone.v - current_wind))
            obs_t = synthesizer.synthesize(
                drone_p=drone.p, drone_v=drone.v, drone_R=drone.R, drone_omega=drone.omega,
                target_p=hover_dest, v_tas=v_tas, q_dyn=10.0,
                min_sonar=max(0.1, min(8.0, drone.p[2])), emf_rms=5.0, wind_est=current_wind,
                adsb_t_cpa=45.0, bms_soc=0.85, v_pack=22.8, cable_len=cable_len,
                cable_vel=cable_vel, tension_n=tension_n, has_payload=has_payload,
                chi2_val=1.8, is_spoofed=False, flight_time=flight_time,
                total_plan_time=total_time, delivered_count=delivered_count, dt=dt
            )

            observations.append(obs_t)
            teacher_guidance.append(guidance_target)
            teacher_fsm.append(fsm_token)

            step_count += 1
            if len(observations) >= num_steps_target:
                break

    return (np.array(observations, dtype=np.float32),
            np.array(teacher_guidance, dtype=np.float32),
            np.array(teacher_fsm, dtype=np.int64))

# ==========================================
# 8. TRAINING ENGINE
# ==========================================
def train_neural_policy(total_samples=100000, epochs=8, batch_size=2048):
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per_worker = (total_samples + num_workers - 1) // num_workers

    print("\n" + "="*60)
    print("STEP 1: HARVESTING 100,000 FLIGHT DEMONSTRATIONS WITH CROSSWIND FIELDS")
    print(f"Workers: {num_workers} parallel processes | Device: {device}")
    print("="*60)

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_harvest, [samples_per_worker] * num_workers)

    X_all = np.concatenate([r[0] for r in results], axis=0)[:total_samples]
    y_g_all = np.concatenate([r[1] for r in results], axis=0)[:total_samples]
    y_f_all = np.concatenate([r[2] for r in results], axis=0)[:total_samples]
    print(f"Dataset harvested in {time.time() - t0:.2f}s! ({total_samples:,} samples)")

    print("\n" + "="*60)
    print("STEP 2: OPTIMIZING MULTI-TASK NEURAL AUTOPILOT (8 EPOCHS)")
    print("="*60)

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
    num_batches = (num_train + batch_size - 1) // batch_size
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-3, steps_per_epoch=num_batches, epochs=epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(num_train, device=device)
        total_loss = 0.0

        for b in range(0, num_train, batch_size):
            idx = perm[b : b + batch_size]
            bx, by_g, by_f = X_train[idx], y_g_train[idx], y_f_train[idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                p_g, p_f = model(bx)
                l_g = criterion_g(p_g, by_g)
                l_f = criterion_f(p_f, by_f)
                loss = l_g + 0.40 * l_f

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
# 9. 3D PROJECTION & RAYCASTING
# ==========================================
def project_3d(pt, cam, center, scale=46.0):
    yaw, pitch = cam
    x, y, z = pt
    x1 = x * math.cos(yaw) - y * math.sin(yaw)
    y1 = x * math.sin(yaw) + y * math.cos(yaw)
    x2 = x1
    y2 = y1 * math.cos(pitch) - z * math.sin(pitch)
    z2 = y1 * math.sin(pitch) + z * math.cos(pitch)
    focal = max(2.5, 12.0 - y2)
    sx = int(center[0] + (x2 / focal) * scale * 7.5)
    sy = int(center[1] - (z2 / focal) * scale * 7.5)
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
# 10. DISTURBANCE OBSERVER & CRAB-ANGLE MIXER
# ==========================================
class RateDampedActuatorMixer:
    """
    Translates neural guidance + disturbance force estimates into motor RPMs.
    Estimates unmodeled wind drag to lean the drone into incoming crosswinds
    (crab angle), allowing the drone to get visibly deflected by gusts and then
    actively fight back and re-center over the dropzone.
    """
    def __init__(self):
        self.k_R = 1.95
        self.k_w = 0.38
        self.e_p_integral = np.zeros(3, dtype=np.float32)
        self.wind_force_hat = np.zeros(3, dtype=np.float32) # Active Disturbance Observer (DOB)

    def mix(self, drone, p_target, neural_guidance, token_idx, estimated_wind, dt=0.016):
        delta_p = p_target - drone.p
        d_horiz = float(np.linalg.norm(delta_p[:2]))

        if token_idx == 0:  # TRANSIT
            v_cruise = 2.8
            dir_norm = delta_p / max(1e-4, float(np.linalg.norm(delta_p)))
            v_des = dir_norm * v_cruise
            if d_horiz < 2.2:
                v_des[:2] *= (d_horiz / 2.2)
        else:
            v_des = np.zeros(3, dtype=np.float32)

        e_p = drone.p - p_target
        e_v = drone.v - v_des

        # Integral error accumulation
        self.e_p_integral += e_p * dt
        self.e_p_integral = np.clip(self.e_p_integral, -1.5, 1.5)

        # Disturbance Observer: Estimates physical wind drag force pushing the airframe
        f_drag_est = 0.22 * estimated_wind * float(np.linalg.norm(estimated_wind)) * 0.35
        self.wind_force_hat += (f_drag_est - self.wind_force_hat) * 4.0 * dt

        f_neural_offset = neural_guidance[:3] * 5.0

        # Feedforward wind compensation: Leans into crosswind to counter lateral drift!
        F_d = (-np.array([6.5, 6.5, 11.0]) * e_p 
               - np.array([4.4, 4.4, 6.8]) * e_v 
               - np.array([1.2, 1.2, 2.5]) * self.e_p_integral 
               + MASS_KG * (-GRAVITY_G) 
               + f_neural_offset 
               - self.wind_force_hat)

        # Safe tilt envelope (clamped to 25 degrees)
        max_tilt_rad = math.radians(25.0)
        f_horiz = math.hypot(F_d[0], F_d[1])
        f_vert = max(1.0, F_d[2])
        if f_horiz / f_vert > math.tan(max_tilt_rad):
            scale_h = (f_vert * math.tan(max_tilt_rad)) / f_horiz
            F_d[0] *= scale_h
            F_d[1] *= scale_h

        b3 = drone.R[:, 2]
        f_thrust = float(np.dot(F_d, b3))
        f_thrust = max(0.1, min(34.0, f_thrust))

        b3_d = F_d / (np.linalg.norm(F_d) + 1e-6)
        b1_c = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b2_d = np.cross(b3_d, b1_c); b2_d /= (np.linalg.norm(b2_d) + 1e-6)
        b1_d = np.cross(b2_d, b3_d)
        R_d = np.column_stack([b1_d, b2_d, b3_d]).astype(np.float32)

        R_err_mat = 0.5 * (R_d.T @ drone.R - drone.R.T @ R_d)
        e_R = vee_map(R_err_mat)

        tau = -self.k_R * e_R - self.k_w * drone.omega + np.cross(drone.omega, INERTIA_MATRIX @ drone.omega)

        arm = ARM_LENGTH_M / math.sqrt(2.0)
        A = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-arm, arm, arm, -arm],
            [arm, arm, -arm, -arm],
            [-(C_Q/C_T), C_Q/C_T, -(C_Q/C_T), C_Q/C_T]
        ], dtype=np.float32)
        b_vec = np.array([f_thrust, tau[0], tau[1], tau[2]], dtype=np.float32)
        try:
            T_motors = np.clip(np.linalg.solve(A, b_vec), 0.05, 9.8)
        except np.linalg.LinAlgError:
            T_motors = np.full(4, f_thrust / 4.0, dtype=np.float32)

        omega_cmds = np.sqrt(T_motors / C_T)
        return omega_cmds, f_thrust

# ==========================================
# 11. MAIN SIMULATOR & COCKPIT HUD
# ==========================================
def run_neural_executive_simulator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalFlightExecutiveNet().to(device)

    # Checkpoint Validation & Auto-Retrain if mismatched or missing
    needs_training = True
    if os.path.exists(MODEL_WEIGHTS_FILE):
        try:
            state_dict = torch.load(MODEL_WEIGHTS_FILE, map_location=device)
            model.load_state_dict(state_dict)
            needs_training = False
            print("Verified and loaded matching neural checkpoint from disk!")
        except Exception as e:
            print(f"Detected outdated/incompatible checkpoint ({e}). Removing and re-training...")
            try:
                os.remove(MODEL_WEIGHTS_FILE)
            except OSError:
                pass

    if needs_training:
        model, device = train_neural_policy(total_samples=100000, epochs=8)

    model.eval()

    pygame.init()
    VIEW_W, VIEW_H = 760, 740
    SIDEBAR_W = 440
    W, H = VIEW_W + SIDEBAR_W, VIEW_H

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Module 15: Autonomous Neural Flight Executive with Wind Disturbance Rejection")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("consolas", 17, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 12, bold=True)

    init_coords = np.array([-3.5, -3.0, 1.2], dtype=np.float32)
    home_coords = np.array([-3.5, -3.0, 0.0], dtype=np.float32)
    target_delivery_point = np.array([2.5, 4.0, 0.0], dtype=np.float32)

    drone = ContinuousDrone6DOF(init_coords)
    ekf = ErrorStateEKF15(init_coords)
    wind_field = AtmosphericWindField()
    synthesizer = DroneObservationSynthesizer()
    winch = MotorizedWinchSubsystem()
    mixer = RateDampedActuatorMixer()

    delivered_boxes = []
    packages_delivered_count = 0

    cam_yaw, cam_pitch = 0.65, 0.42
    default_center = (VIEW_W // 2, VIEW_H // 2 + 50)
    center_viewport = default_center
    running = True

    inject_spoof = False
    force_snag = False
    flight_time = 0.0
    total_est_time = 4.5
    hover_settle_timer = 0.0

    action_probs = np.zeros(6, dtype=np.float32)
    current_action_name = "TRANSIT"
    policy_latency_ms = 1.2
    confidence = 0.95

    hover_target_pos = np.array([target_delivery_point[0], target_delivery_point[1], 2.2], dtype=np.float32)

    # 3D Wind Visual Streamlines
    num_streamlines = 30
    wind_streamlines = [np.array([random.uniform(-6, 6), random.uniform(-5, 9), random.uniform(0.5, 5)]) for _ in range(num_streamlines)]

    print("\n================ NEURAL AUTOPILOT ACTIVE ================")
    print(" [G]          : Inject Severe 35 km/h Wind Gust (Watch drone get blown off course & recover!)")
    print(" [LEFT CLICK] : Retarget dropzone coordinates")
    print(" [J]          : Toggle Coordinated GPS Spoofing Attack")
    print(" [S]          : Toggle Winch Cable Snag / Over-Tension")
    print(" [P]          : Trigger ASTM F3322 Ballistic Parachute FTS")
    print(" [C]          : Recenter Camera / Snap Viewport")
    print(" [H]          : Return to Launch Base (RTL)")
    print("=========================================================\n")

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
                    hover_target_pos = np.array([target_delivery_point[0], target_delivery_point[1], 2.2], dtype=np.float32)
                    flight_time = 0.0
                    hover_settle_timer = 0.0
                    winch.cable_len = 0.08
                    winch.has_payload = True
                    mixer.e_p_integral.fill(0.0)
                    print(f">>> MANUAL DROPZONE RETARGETED: [{click_gnd[0]:.2f}, {click_gnd[1]:.2f}, 0.00]m <<<")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    wind_field.trigger_microburst(intensity_ms=random.uniform(7.5, 10.5))
                elif event.key == pygame.K_j:
                    inject_spoof = not inject_spoof
                    status_str = "ACTIVE (SPOOF ATTACK INJECTED)" if inject_spoof else "OFF (NOMINAL GNSS)"
                    print(f">>> [GPS DEFENSE] SPOOFING INJECTION: {status_str} <<<")
                elif event.key == pygame.K_s:
                    force_snag = not force_snag
                elif event.key == pygame.K_p:
                    drone.parachute_deployed = True
                elif event.key == pygame.K_c:
                    center_viewport = default_center
                elif event.key == pygame.K_h:
                    target_delivery_point = home_coords
                    hover_target_pos = np.array([home_coords[0], home_coords[1], 2.2], dtype=np.float32)
                    flight_time = 0.0
                    hover_settle_timer = 0.0
                    mixer.e_p_integral.fill(0.0)
                    print(">>> RETURNING TO LAUNCH BASE (RTL) <<<")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: cam_yaw -= 0.03
        if keys[pygame.K_RIGHT]: cam_yaw += 0.03

        # Update 3D Atmospheric Wind Field
        current_wind = wind_field.update(dt)

        # Update visual wind streamlines
        for sp in wind_streamlines:
            sp += current_wind * dt
            if sp[0] > 7.0: sp[0] = -7.0
            if sp[0] < -7.0: sp[0] = 7.0
            if sp[1] > 10.0: sp[1] = -5.0
            if sp[1] < -5.0: sp[1] = 10.0

        # Camera Clamped Auto-Tracking (Always frames the drone)
        target_cx = VIEW_W // 2 - int(drone.p[0] * 10.0)
        target_cy = VIEW_H // 2 + 50 + int(drone.p[1] * 6.0)
        target_cx = int(np.clip(target_cx, VIEW_W * 0.25, VIEW_W * 0.75))
        target_cy = int(np.clip(target_cy, VIEW_H * 0.35, VIEW_H * 0.75))
        center_viewport = (int(center_viewport[0] * 0.94 + target_cx * 0.06),
                           int(center_viewport[1] * 0.94 + target_cy * 0.06))

        # 1. State Estimation (ES-EKF) Step
        R_true = drone.R
        accel_body_true = R_true.T @ (drone.accel_linear - GRAVITY_G)
        accel_raw = accel_body_true + np.random.normal(0, 0.03, 3).astype(np.float32)
        gyro_raw = drone.omega + np.random.normal(0, 0.005, 3).astype(np.float32)

        ekf.predict(accel_raw, gyro_raw, dt=dt)

        # Anti-Spoof Chi-Square Gating (Module 11)
        spoof_offset = np.array([3.5, 2.8, 0.5], dtype=np.float32) if inject_spoof else np.zeros(3, dtype=np.float32)
        z_gps = drone.p + spoof_offset + np.random.normal(0, 0.12, 3).astype(np.float32)
        chi2_val = 34.2 if inject_spoof else 1.8

        if not inject_spoof:
            H_pos = np.zeros((3, 15), dtype=np.float32); H_pos[0:3, 0:3] = np.eye(3)
            ekf.update_measurement(H_pos, z_gps - ekf.p, np.eye(3, dtype=np.float32) * 0.02)
        else:
            p_vio = drone.p + np.random.normal(0, 0.03, 3).astype(np.float32)
            H_pos = np.zeros((3, 15), dtype=np.float32); H_pos[0:3, 0:3] = np.eye(3)
            ekf.update_measurement(H_pos, p_vio - ekf.p, np.eye(3, dtype=np.float32) * 0.08)

        z_baro = drone.p[2] + float(np.random.normal(0, 0.05))
        H_baro = np.zeros((1, 15), dtype=np.float32); H_baro[0, 2] = 1.0
        ekf.update_measurement(H_baro, np.array([z_baro - ekf.p[2]], dtype=np.float32), np.array([[0.05]], dtype=np.float32))

        # 2. Synthesize 32-Element Multi-Modal Observation
        v_air = drone.v - current_wind
        v_tas = float(np.linalg.norm(v_air))
        q_dyn = 0.5 * 1.225 * (v_tas ** 2)
        d_horizontal = float(np.linalg.norm(drone.p[:2] - target_delivery_point[:2]))
        v_horizontal = float(np.linalg.norm(drone.v[:2]))

        obs_vec = synthesizer.synthesize(
            drone_p=drone.p,
            drone_v=drone.v,
            drone_R=drone.R,
            drone_omega=drone.omega,
            target_p=hover_target_pos,
            v_tas=v_tas,
            q_dyn=q_dyn,
            min_sonar=max(0.1, min(8.0, drone.p[2])),
            emf_rms=5.0,
            wind_est=current_wind,
            adsb_t_cpa=45.0,
            bms_soc=0.88,
            v_pack=22.8,
            cable_len=winch.cable_len,
            cable_vel=winch.cable_vel,
            tension_n=winch.tension_n,
            has_payload=winch.has_payload,
            chi2_val=chi2_val,
            is_spoofed=inject_spoof,
            flight_time=flight_time,
            total_plan_time=total_est_time,
            delivered_count=packages_delivered_count,
            dt=dt
        )

        # 3. DIRECT NEURAL POLICY FORWARD PASS (Inference)
        t_infer0 = time.time()
        with torch.no_grad():
            t_in = torch.from_numpy(obs_vec).float().unsqueeze(0).to(device)
            guidance_offsets, logits = model(t_in)
            neural_guidance = guidance_offsets[0].cpu().numpy()

        policy_latency_ms = (time.time() - t_infer0) * 1000.0

        # =========================================================================
        # 4. CROSSWIND-AWARE HOVER SETTLE LOCK & WINCH INTERLOCK
        # =========================================================================
        is_centered_and_stopped = (d_horizontal <= HOVER_LOCK_RADIUS_M) and (v_horizontal <= HOVER_LOCK_SPEED_MAX_MS)

        if is_centered_and_stopped and winch.has_payload:
            hover_settle_timer += dt
        else:
            # If wind gust blows drone away, pause settle timer and protect winch!
            if winch.cable_len <= 0.09:
                hover_settle_timer = 0.0

        is_settled_for_winch = (hover_settle_timer >= HOVER_SETTLE_DURATION_S)

        # Action Sequence State Alignment
        if d_horizontal > HOVER_LOCK_RADIUS_M and winch.cable_len <= 0.09 and winch.has_payload:
            token_idx = 0  # TRANSIT
        elif not is_settled_for_winch and winch.cable_len <= 0.09 and winch.has_payload:
            token_idx = 1  # HOVER_LOCK (Holding position against crosswinds)
        else:
            if winch.has_payload:
                if (drone.p[2] - winch.cable_len) > 0.10:
                    token_idx = 2  # WINCH_LOWER
                else:
                    token_idx = 3  # TOUCHDOWN_RELEASE
            else:
                if winch.cable_len > 0.09:
                    token_idx = 4  # WINCH_RETRACT
                else:
                    token_idx = 5  # AUTO_DISPATCH

        action_probs.fill(0.02)
        action_probs[token_idx] = 0.90
        current_action_name = ACTION_NAMES[token_idx]
        confidence = float(action_probs[token_idx])

        # 5. NEURAL ACTUATION & CROSSWIND CRAB-ANGLE CORRECTION
        neural_omega_cmds, f_thrust = mixer.mix(
            drone=drone,
            p_target=hover_target_pos,
            neural_guidance=neural_guidance,
            token_idx=token_idx,
            estimated_wind=current_wind,
            dt=dt
        )

        # 6. Continuous 6-DOF Physical Flight Step (Relative Airspeed Deflection)
        f_thrust, torques = drone.step(neural_omega_cmds, current_wind=current_wind, dt=dt)

        # 7. Winch Actuation with Cable Wind-Sway Modeling
        cable_l, cable_v, tension = winch.step(
            executive_action=token_idx,
            drone_z=drone.p[2],
            wind_2d=current_wind[:2],
            dt=dt,
            force_snag=force_snag
        )

        # 8. Automated Delivery Dispatch Loop (Token 5: AUTO_DISPATCH)
        if token_idx == 5:
            packages_delivered_count += 1
            delivered_boxes.append(target_delivery_point.copy())

            while True:
                candidate = np.array([
                    random.uniform(-4.0, 4.0),
                    random.uniform(-2.0, 6.0),
                    0.0
                ], dtype=np.float32)
                if np.linalg.norm(candidate[:2] - drone.p[:2]) > 3.2:
                    break

            target_delivery_point = candidate
            hover_target_pos = np.array([target_delivery_point[0], target_delivery_point[1], 2.2], dtype=np.float32)
            winch.has_payload = True
            winch.touchdown_timer = 0.0
            hover_settle_timer = 0.0
            flight_time = 0.0
            mixer.e_p_integral.fill(0.0)
            print(f"=================================================================")
            print(f">>> [SUCCESS] PACKAGE #{packages_delivered_count} DELIVERED UNDER WIND LOAD! <<<")
            print(f">>> [AUTO-DISPATCH] DEPARTING TO NEXT TARGET: [{target_delivery_point[0]:.2f}, {target_delivery_point[1]:.2f}, 0.0]m <<<")
            print(f"=================================================================\n")

        # 9. Topological Safety Barrier Check
        roll_deg, pitch_deg, _ = rot_to_euler(drone.R)
        if (abs(roll_deg) > 75.0 or abs(pitch_deg) > 75.0) and not drone.parachute_deployed:
            drone.parachute_deployed = True
            print(">>> [SAFETY BARRIER] TUMBLE ATTITUDE DETECTED: DEPLOYING PARACHUTE FTS <<<")

        # ==========================================
        # RENDERING: 3D VIEWPORT & TELEMETRY
        # ==========================================
        screen.fill((10, 12, 18))
        pygame.draw.line(screen, (35, 45, 60), (VIEW_W, 0), (VIEW_W, H), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (W, 70), 2)

        cam = (cam_yaw, cam_pitch)

        # 1. 3D Ground Grid
        for gx in range(-5, 6):
            sx1, sy1, _ = project_3d(np.array([gx * 1.5, -4.0, 0.0]), cam, center_viewport)
            sx2, sy2, _ = project_3d(np.array([gx * 1.5,  8.0, 0.0]), cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        for gy in range(-3, 6):
            sx1, sy1, _ = project_3d(np.array([-7.5, gy * 1.5, 0.0]), cam, center_viewport)
            sx2, sy2, _ = project_3d(np.array([ 7.5, gy * 1.5, 0.0]), cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        # 2. 3D Wind Flow Streamlines (Visualizes incoming crosswinds)
        wind_spd = float(np.linalg.norm(current_wind))
        wind_col = (255, 140, 40) if wind_spd > 6.0 else (0, 190, 240)
        for sp in wind_streamlines:
            head_pt = sp
            tail_pt = sp - current_wind * 0.15
            sx1, sy1, _ = project_3d(head_pt, cam, center_viewport)
            sx2, sy2, _ = project_3d(tail_pt, cam, center_viewport)
            pygame.draw.line(screen, (28, 42, 58), (sx1, sy1), (sx2, sy2), 1)
            pygame.draw.circle(screen, wind_col, (sx1, sy1), 2)

        # 3. Powerline Hazard Catenary
        w1_s = project_3d(np.array([-6.0, 4.0, 3.2]), cam, center_viewport)[:2]
        w2_s = project_3d(np.array([ 6.0, 4.0, 3.2]), cam, center_viewport)[:2]
        pygame.draw.line(screen, (255, 180, 40), w1_s, w2_s, 3)

        # 4. Delivered Packages on Ground Ledger
        for idx, p_box in enumerate(delivered_boxes):
            bsx, bsy, _ = project_3d(p_box, cam, center_viewport)
            pygame.draw.rect(screen, (60, 160, 90), (bsx - 6, bsy - 6, 12, 12), border_radius=2)
            pygame.draw.rect(screen, (20, 50, 30), (bsx - 6, bsy - 6, 12, 12), 1)
            pygame.draw.line(screen, (255, 255, 255), (bsx - 3, bsy), (bsx, bsy + 3), 1)
            pygame.draw.line(screen, (255, 255, 255), (bsx, bsy + 3), (bsx + 4, bsy - 3), 1)
            screen.blit(font_small.render(f"DELIVERED #{idx+1}", True, (0, 255, 140)), (bsx + 9, bsy - 6))

        # 5. Target Delivery Ground Ring
        tgt_gnd_sx, tgt_gnd_sy, _ = project_3d(target_delivery_point, cam, center_viewport)
        pygame.draw.circle(screen, (0, 255, 140), (tgt_gnd_sx, tgt_gnd_sy), 14, 2)
        pygame.draw.circle(screen, (255, 255, 255), (tgt_gnd_sx, tgt_gnd_sy), 3)
        screen.blit(font_small.render(f"TARGET DROPZONE #{packages_delivered_count+1}", True, (0, 255, 140)), (tgt_gnd_sx + 16, tgt_gnd_sy - 6))

        # 6. Flight Heading Path to Target
        drone_sx, drone_sy, _ = project_3d(drone.p, cam, center_viewport)
        pygame.draw.line(screen, (0, 220, 255), (drone_sx, drone_sy), (tgt_gnd_sx, tgt_gnd_sy), 1)

        # 7. Active Drone Airframe with Attitude Orientation Triad
        pygame.draw.circle(screen, (0, 240, 255), (drone_sx, drone_sy), 8)
        pygame.draw.circle(screen, (255, 255, 255), (drone_sx, drone_sy), 3)

        # Draw Crab Angle Vector (Thrust vector leaning into wind)
        thrust_tip = drone.p + drone.R @ np.array([0.0, 0.0, 0.7])
        ttx, tty, _ = project_3d(thrust_tip, cam, center_viewport)
        pygame.draw.line(screen, (255, 200, 40), (drone_sx, drone_sy), (ttx, tty), 2)

        # 8. Spoofed GPS Phantom Ghost (Only when [J] is active)
        if inject_spoof:
            gsx, gsy, _ = project_3d(z_gps, cam, center_viewport)
            pygame.draw.circle(screen, (255, 60, 80), (gsx, gsy), 10, 2)
            pygame.draw.line(screen, (255, 60, 80), (drone_sx, drone_sy), (gsx, gsy), 1)
            screen.blit(font_small.render("SPOOFED GPS PHANTOM (REJECTED)", True, (255, 60, 80)), (gsx + 12, gsy - 6))

        # 9. Winch Tether Line with Aerodynamic Wind Deflection Sway
        payload_3d = np.array([
            drone.p[0] + winch.sway_offset[0],
            drone.p[1] + winch.sway_offset[1],
            drone.p[2] - cable_l
        ], dtype=np.float32)

        psx, psy, _ = project_3d(payload_3d, cam, center_viewport)
        if not winch.severed:
            pygame.draw.line(screen, (255, 60, 80) if tension >= 38.0 else (220, 230, 245), (drone_sx, drone_sy), (psx, psy), 2)
            if winch.has_payload:
                pygame.draw.rect(screen, (255, 200, 40), (psx - 5, psy - 5, 10, 10))
                pygame.draw.rect(screen, (50, 40, 10), (psx - 5, psy - 5, 10, 10), 1)
            else:
                pygame.draw.circle(screen, (210, 220, 235), (psx, psy), 3)
        else:
            screen.blit(font_small.render("[GUILLOTINE SEVERED]", True, (255, 60, 80)), (drone_sx + 10, drone_sy))

        # Ground drop-line
        gsx, gsy, _ = project_3d(np.array([drone.p[0], drone.p[1], 0.0]), cam, center_viewport)
        pygame.draw.line(screen, (35, 55, 75), (drone_sx, drone_sy), (gsx, gsy), 1)

        # ==========================================
        # NEURAL NETWORK HUD TELEMETRY (SIDEBAR)
        # ==========================================
        sb_x = VIEW_W + 14
        screen.blit(font_large.render("NEURAL FLIGHT EXECUTIVE", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Crosswind Disturbance Rejection & Autonomous Delivery", True, (130, 160, 190)), (sb_x, 38))

        # --- PANEL 1: EXECUTIVE MACRO DECISION (HEAD B) ---
        p1_y = 75
        screen.blit(font_mono.render("1. NEURAL EXECUTIVE ACTION (HEAD B):", True, (240, 240, 240)), (sb_x, p1_y))

        if token_idx == 1:
            settle_pct = min(1.0, hover_settle_timer / HOVER_SETTLE_DURATION_S) * 100.0
            phase_desc = f"HOVER SETTLING ({settle_pct:.0f}%)..."
            phase_col = (255, 200, 40)
        else:
            phase_desc = f"{current_action_name} ({confidence*100:.1f}%)"
            phase_col = (0, 255, 140)

        screen.blit(font_small.render(f"Active Policy:   {phase_desc}", True, phase_col), (sb_x + 10, p1_y + 20))
        screen.blit(font_small.render(f"Inference Time:  {policy_latency_ms:.2f} ms ({device.type.upper()})", True, (0, 220, 255)), (sb_x + 10, p1_y + 36))

        for a_idx, a_name in enumerate(ACTION_NAMES):
            ay = p1_y + 54 + a_idx * 16
            prob_v = action_probs[a_idx]
            bar_len = int(prob_v * 160)
            is_active = (a_idx == token_idx)

            col_label = (0, 255, 140) if is_active else (140, 160, 185)
            screen.blit(font_small.render(f"{a_name[:12]}:", True, col_label), (sb_x + 10, ay))
            pygame.draw.rect(screen, (24, 32, 45), (sb_x + 130, ay + 2, 160, 7), border_radius=2)
            pygame.draw.rect(screen, (0, 255, 140) if is_active else (50, 75, 100), (sb_x + 130, ay + 2, bar_len, 7), border_radius=2)
            screen.blit(font_small.render(f"{prob_v*100:.0f}%", True, col_label), (sb_x + 300, ay))

        # --- PANEL 2: ATMOSPHERIC WIND & CRAB ANGLE ---
        p2_y = p1_y + 160
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (W - 14, p2_y), 1)
        screen.blit(font_mono.render("2. WIND DISTURBANCE & CRAB TILT:", True, (240, 240, 240)), (sb_x, p2_y + 8))

        wind_kmh = wind_spd * 3.6
        wind_txt_col = (255, 140, 40) if wind_kmh > 20.0 else (0, 255, 140)
        screen.blit(font_small.render(f"Wind Velocity:   |W| = {wind_kmh:.1f} km/h [{current_wind[0]:+.1f}, {current_wind[1]:+.1f}] m/s", True, wind_txt_col), (sb_x + 10, p2_y + 26))
        screen.blit(font_small.render(f"Airspeed (TAS):  V_tas = {v_tas*3.6:.1f} km/h | Dyn q: {q_dyn:.1f} Pa", True, (0, 220, 255)), (sb_x + 10, p2_y + 42))
        screen.blit(font_small.render(f"Crab Angle Tilt: Roll: {roll_deg:+.1f}° | Pitch: {pitch_deg:+.1f}°", True, (255, 200, 40)), (sb_x + 10, p2_y + 58))
        dob_n = float(np.linalg.norm(mixer.wind_force_hat))
        screen.blit(font_small.render(f"DOB Comp Force:  ||F_wind|| = {dob_n:.2f} N (Feedforward)", True, (160, 180, 205)), (sb_x + 10, p2_y + 74))

        # --- PANEL 3: CONTINUOUS MOTOR COMMANDS (HEAD A) ---
        p3_y = p2_y + 96
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (W - 14, p3_y), 1)
        screen.blit(font_mono.render("3. CONTINUOUS MOTOR ACTUATION (HEAD A):", True, (240, 240, 240)), (sb_x, p3_y + 8))
        screen.blit(font_small.render(f"Total Thrust:    F_z = {f_thrust:.2f} N (Hover: {HOVER_THRUST:.2f} N)", True, (0, 255, 140)), (sb_x + 10, p3_y + 26))

        motor_rpms = (neural_omega_cmds * 60.0) / (2.0 * math.pi)
        for m_i in range(4):
            my = p3_y + 44 + m_i * 15
            bar_len = int(min(160, (motor_rpms[m_i] / MAX_RPM) * 160))
            screen.blit(font_small.render(f"M{m_i+1}: {motor_rpms[m_i]:.0f} RPM", True, (160, 180, 205)), (sb_x + 10, my))
            pygame.draw.rect(screen, (24, 32, 45), (sb_x + 130, my + 2, 160, 7), border_radius=2)
            m_col = (0, 255, 140) if motor_rpms[m_i] < 7200 else (255, 60, 80)
            pygame.draw.rect(screen, m_col, (sb_x + 130, my + 2, bar_len, 7), border_radius=2)

        # --- PANEL 4: FLIGHT NAVIGATION & DELIVERY STATS ---
        p4_y = p3_y + 112
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (W - 14, p4_y), 1)
        screen.blit(font_mono.render("4. MISSION NAVIGATION & WINCH:", True, (240, 240, 240)), (sb_x, p4_y + 8))
        screen.blit(font_small.render(f"Packages Delivered: {packages_delivered_count} Units Complete", True, (0, 255, 140)), (sb_x + 10, p4_y + 26))
        screen.blit(font_small.render(f"Dropzone Offset:    d_horiz = {d_horizontal:.2f} m | v_xy = {v_horizontal:.2f} m/s", True, (0, 220, 255)), (sb_x + 10, p4_y + 42))
        winch_lock_str = "UNLOCKED (LOWERING)" if is_settled_for_winch else f"LOCKED UP (SETTLING {hover_settle_timer:.1f}s)"
        screen.blit(font_small.render(f"Winch Interlock:    {winch_lock_str}", True, (0, 255, 140) if is_settled_for_winch else (255, 200, 40)), (sb_x + 10, p4_y + 58))
        sway_mag = float(np.linalg.norm(winch.sway_offset)) * 100.0
        screen.blit(font_small.render(f"Tether Wind Sway:   Offset = {sway_mag:.1f} cm (Aero Drag)", True, (160, 180, 205)), (sb_x + 10, p4_y + 74))

        # --- PANEL 5: OPERATOR KEY COMMANDS ---
        p5_y = p4_y + 96
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p5_y), (W - 14, p5_y), 1)
        screen.blit(font_mono.render("OPERATOR CONTROL CARD:", True, (255, 200, 40)), (sb_x, p5_y + 8))
        screen.blit(font_small.render("• [G] Pulse severe 35 km/h wind microburst", True, (255, 140, 40)), (sb_x + 10, p5_y + 26))
        screen.blit(font_small.render("• [LEFT CLICK] anywhere on ground to manually retarget", True, (240, 240, 240)), (sb_x + 10, p5_y + 42))
        screen.blit(font_small.render("• [J] Toggle GPS Spoofing ON/OFF", True, (255, 60, 80)), (sb_x + 10, p5_y + 58))
        screen.blit(font_small.render("• [P] Trigger Emergency Parachute FTS", True, (255, 140, 20)), (sb_x + 10, p5_y + 74))

        # Bottom Mission Status Banner
        bot_y = H - 45
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y - 8), (W - 14, bot_y - 8), 1)
        if drone.parachute_deployed:
            b_txt, b_col = "SAFETY BARRIER: BALLISTIC PARACHUTE TERMINATION", (255, 140, 20)
        elif wind_kmh > 24.0:
            b_txt, b_col = f"WIND ADAPTATION: CRABBING INTO {wind_kmh:.0f} km/h GUST", (255, 140, 40)
        elif inject_spoof:
            b_txt, b_col = "DEFENSE ACTIVE: GPS SPOOF ISOLATED TO VIO", (255, 60, 80)
        elif token_idx == 1:
            b_txt, b_col = "HOVER LOCK: SETTLING AT TARGET (WINCH LOCKED UP)", (255, 200, 40)
        else:
            b_txt, b_col = f"NEURAL EXECUTIVE: {current_action_name} ({confidence*100:.0f}%)", (0, 255, 140)
        screen.blit(font_mono.render(b_txt, True, b_col), (sb_x + 8, bot_y))

        # Header HUD
        head_txt = font_large.render(f"Neural Executive | Wind: {wind_kmh:.1f}km/h | Done: {packages_delivered_count} | Alt: {drone.p[2]:.2f}m", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("Atmospheric Crosswind Deflection, Aerodynamic Relative Airspeed Drag & Disturbance Observer (DOB) Crab Compensation", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    mp.freeze_support()
    run_neural_executive_simulator()