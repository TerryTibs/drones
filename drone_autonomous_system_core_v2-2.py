"""
=============================================================================
UNIFIED PRODUCTION AUTONOMOUS DRONE SYSTEM CORE (V2.2 - BULLETPROOF)
File: drone_autonomous_system_core_v2.py
=============================================================================
Bugfix Release:
- Hardened Anti-GPS Spoofing Defense with Bounded Phantom Drift
- Fail-Safe Camera Viewport Clamping (Camera can never fly off-screen)
- Visual "Spoofed GPS Ghost" Vector & Chi-Square (χ²) Trip Indicator
- Reliable Interactive Point-and-Click Autonomous Dropzone Dispatch
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
import pygame
import sys

# ==========================================
# 0. GLOBAL PHYSICAL & AIRFRAME CONSTANTS
# ==========================================
MASS_KG = 1.35
GRAVITY_G = np.array([0.0, 0.0, -9.80665], dtype=np.float32)
G_ACCEL = 9.80665
ARM_LENGTH_M = 0.225
INERTIA_MATRIX = np.diag([0.016, 0.016, 0.028]).astype(np.float32)
INV_INERTIA = np.linalg.inv(INERTIA_MATRIX)

C_T = 1.28e-5            # Thrust coeff: T = C_T * omega^2
C_Q = 2.45e-7            # Torque coeff: Q = C_Q * omega^2
MOTOR_TAU_S = 0.035      # 35ms motor response lag
MAX_RPM = 8800.0
MAX_OMEGA = (MAX_RPM * 2.0 * math.pi) / 60.0
HOVER_THRUST = MASS_KG * G_ACCEL
HOVER_OMEGA = math.sqrt((HOVER_THRUST / 4.0) / C_T)

# Atmospheric & Sensor Defaults
P_0 = 1013.25
SPEED_OF_SOUND = 343.2
EMF_WARNING_THRESHOLD_UT = 18.0
MAX_LEGAL_ALT_AGL = 120.0
CHI2_SPOOF_THRESHOLD = 16.27  # 99.9% Chi-Square boundary for 3-DOF

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

# Winch Delivery Constants
PAYLOAD_MASS_KG = 2.0
PAYLOAD_WEIGHT_N = PAYLOAD_MASS_KG * G_ACCEL
SNAG_THRESHOLD_N = 38.0
GUILLOTINE_THRESHOLD_N = 48.0

# ==========================================
# 1. MATHEMATICAL UTILITIES & LIE ALGEBRA
# ==========================================
def skew_symmetric(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float32)

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
# 2. CONTINUOUS 6-DOF FLIGHT DYNAMICS
# ==========================================
class DroneAirframe6DOF:
    def __init__(self, init_pos):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.R = np.eye(3, dtype=np.float32)
        self.omega = np.zeros(3, dtype=np.float32)
        self.motor_omegas = np.full(4, HOVER_OMEGA, dtype=np.float32)
        self.wind_gust = np.zeros(3, dtype=np.float32)
        self.last_accel_linear = np.zeros(3, dtype=np.float32)
        self.parachute_deployed = False

    def step(self, motor_cmds, dt=0.016):
        if self.parachute_deployed:
            self.motor_omegas.fill(0.0)
            self.v[0] *= 0.90
            self.v[1] *= 0.90
            self.v[2] += (-4.1 - self.v[2]) * 3.0 * dt
            self.p += self.v * dt
            if self.p[2] <= 0.0:
                self.p[2] = 0.0
                self.v.fill(0.0)
            return 0.0, np.zeros(3), self.v

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
        f_drag = -np.array([0.22, 0.22, 0.35], dtype=np.float32) * self.v
        f_gravity = MASS_KG * GRAVITY_G

        self.last_accel_linear = (self.R @ f_body + f_gravity + f_drag + self.wind_gust) / MASS_KG

        gyroscopic = np.cross(self.omega, INERTIA_MATRIX @ self.omega)
        accel_angular = INV_INERTIA @ (torques - gyroscopic)

        self.p += self.v * dt + 0.5 * self.last_accel_linear * (dt ** 2)
        self.v += self.last_accel_linear * dt
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
        self.wind_gust *= 0.992
        return f_thrust, torques, self.last_accel_linear

# ==========================================
# 3. 15-STATE ERROR-STATE EKF (SO3 MANIFOLD)
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
# 4. GEOMETRIC SE(3) TRACKING CONTROLLER
# ==========================================
class GeometricSE3Controller:
    def __init__(self):
        self.k_p = np.array([6.5, 6.5, 11.0], dtype=np.float32)
        self.k_v = np.array([4.4, 4.4, 6.8], dtype=np.float32)
        self.k_R = 1.85
        self.k_w = 0.35

    def compute(self, p_est, v_est, R_est, w_est, p_des, v_des, a_des, yaw_des=0.0):
        e_p = p_est - p_des
        e_v = v_est - v_des

        F_d = -self.k_p * e_p - self.k_v * e_v + MASS_KG * (-GRAVITY_G) + MASS_KG * a_des
        b3 = R_est[:, 2]
        f_thrust = float(np.dot(F_d, b3))
        f_thrust = max(0.1, min(34.0, f_thrust))

        b3_d = F_d / (np.linalg.norm(F_d) + 1e-6)
        yaw_rad = math.radians(yaw_des)
        b1_c = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=np.float32)
        b2_d = np.cross(b3_d, b1_c)
        b2_d /= (np.linalg.norm(b2_d) + 1e-6)
        b1_d = np.cross(b2_d, b3_d)
        R_d = np.column_stack([b1_d, b2_d, b3_d]).astype(np.float32)

        R_err_mat = 0.5 * (R_d.T @ R_est - R_est.T @ R_d)
        e_R = vee_map(R_err_mat)
        e_w = w_est

        tau = -self.k_R * e_R - self.k_w * e_w + np.cross(w_est, INERTIA_MATRIX @ w_est)

        arm = ARM_LENGTH_M / math.sqrt(2.0)
        A = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-arm, arm, arm, -arm],
            [arm, arm, -arm, -arm],
            [-(C_Q/C_T), C_Q/C_T, -(C_Q/C_T), C_Q/C_T]
        ], dtype=np.float32)
        b = np.array([f_thrust, tau[0], tau[1], tau[2]], dtype=np.float32)
        try:
            T_motors = np.clip(np.linalg.solve(A, b), 0.05, 9.8)
        except np.linalg.LinAlgError:
            T_motors = np.full(4, f_thrust / 4.0, dtype=np.float32)

        omega_cmds = np.sqrt(T_motors / C_T)
        return omega_cmds, f_thrust, tau

# ==========================================
# 5. PIECEWISE 7TH-ORDER MIN-SNAP PLANNER
# ==========================================
class MinimumSnapPlanner:
    def __init__(self):
        self.is_solved = False
        self.waypoints = []
        self.times = []
        self.total_time = 0.0

    def plan(self, waypoints, nominal_speed=2.8):
        self.waypoints = [np.array(w, dtype=np.float32) for w in waypoints]
        M = len(self.waypoints) - 1
        if M < 1: return False

        self.times = [max(1.4, float(np.linalg.norm(self.waypoints[k+1] - self.waypoints[k])) / nominal_speed) for k in range(M)]
        self.total_time = sum(self.times)

        N_vars = 8 * M
        A = np.zeros((N_vars, N_vars), dtype=np.float64)
        bx, by, bz = np.zeros(N_vars), np.zeros(N_vars), np.zeros(N_vars)
        row = 0

        def basis(t, d):
            c = np.zeros(8)
            for j in range(d, 8):
                fact = math.prod(range(j - d + 1, j + 1))
                c[j] = fact * (t ** (j - d))
            return c

        for d in range(4):
            A[row, 0:8] = basis(0.0, d)
            if d == 0: bx[row], by[row], bz[row] = self.waypoints[0]
            row += 1

        for k in range(1, M):
            t_prev = self.times[k-1]
            A[row, (k-1)*8 : k*8] = basis(t_prev, 0)
            bx[row], by[row], bz[row] = self.waypoints[k]
            row += 1
            A[row, k*8 : (k+1)*8] = basis(0.0, 0)
            bx[row], by[row], bz[row] = self.waypoints[k]
            row += 1
            for d in range(1, 7):
                A[row, (k-1)*8 : k*8] = basis(t_prev, d)
                A[row, k*8 : (k+1)*8] = -basis(0.0, d)
                row += 1

        t_last = self.times[-1]
        for d in range(4):
            A[row, (M-1)*8 : M*8] = basis(t_last, d)
            if d == 0: bx[row], by[row], bz[row] = self.waypoints[-1]
            row += 1

        try:
            self.cx = np.linalg.solve(A, bx).reshape(M, 8)
            self.cy = np.linalg.solve(A, by).reshape(M, 8)
            self.cz = np.linalg.solve(A, bz).reshape(M, 8)
            self.is_solved = True
            return True
        except np.linalg.LinAlgError:
            self.is_solved = False
            return False

    def eval(self, t):
        if not self.is_solved: return np.zeros(3), np.zeros(3), np.zeros(3)
        if t >= self.total_time:
            return self.waypoints[-1].copy(), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        t_clamped = max(0.0, min(self.total_time, t))
        accum_t, seg_idx, tau = 0.0, 0, 0.0
        for k, dur in enumerate(self.times):
            if t_clamped <= accum_t + dur or k == len(self.times) - 1:
                seg_idx = k
                tau = t_clamped - accum_t
                break
            accum_t += dur

        T_pow = np.array([tau**i for i in range(8)])
        dT_pow = np.array([0] + [i * tau**(i-1) for i in range(1, 8)])
        d2T_pow = np.array([0, 0] + [i * (i-1) * tau**(i-2) for i in range(2, 8)])

        cx, cy, cz = self.cx[seg_idx], self.cy[seg_idx], self.cz[seg_idx]
        pos = np.array([np.dot(cx, T_pow), np.dot(cy, T_pow), np.dot(cz, T_pow)], dtype=np.float32)
        vel = np.array([np.dot(cx, dT_pow), np.dot(cy, dT_pow), np.dot(cz, dT_pow)], dtype=np.float32)
        acc = np.array([np.dot(cx, d2T_pow), np.dot(cy, d2T_pow), np.dot(cz, d2T_pow)], dtype=np.float32)
        return pos, vel, acc

# ==========================================
# 6. BATTERY MANAGEMENT & DYNAMIC PSR (MOD 9)
# ==========================================
class BatteryManagementPSR:
    def __init__(self, init_soc=0.92, temp_c=18.0):
        self.soc = float(init_soc)
        self.temp_c = float(temp_c)
        self.v1 = np.zeros(NUM_CELLS, dtype=np.float32)
        self.capacity_coulombs = PACK_CAPACITY_AH * 3600.0

    def step(self, current_amps, dt=0.016):
        t_k = self.temp_c + 273.15
        r_scale = math.exp((ARRHENIUS_EA / GAS_CONST_R) * (1.0 / t_k - 1.0 / T_REF_K))
        r0 = CELL_R0_NOMINAL * max(0.85, r_scale)

        self.soc -= (current_amps * dt) / self.capacity_coulombs
        self.soc = max(0.0, min(1.0, self.soc))

        s = max(0.001, min(0.999, self.soc))
        ocv = 3.30 + 0.55 * s + 0.25 * math.log(s + 0.005) + 0.15 * math.exp(6.0 * (s - 0.88))
        ocv = float(np.clip(ocv, 3.20, 4.20))

        cell_volts = []
        for i in range(NUM_CELLS):
            self.v1[i] += (-(self.v1[i] / (CELL_R1 * CELL_C1)) + (current_amps / CELL_C1)) * dt
            cell_volts.append(ocv - current_amps * r0 - self.v1[i])

        v_pack = float(np.sum(cell_volts))
        p_loss = (current_amps ** 2) * (r0 * NUM_CELLS)
        self.temp_c += ((p_loss - 0.35 * (self.temp_c - 18.0)) / 350.0) * dt
        return v_pack, np.array(cell_volts, dtype=np.float32), r0

    def compute_psr(self, drone_pos, home_pos, wind_vec):
        d_home = float(np.linalg.norm(home_pos[:2] - drone_pos[:2]))
        p_return = 175.0 + 3.5 * float(np.linalg.norm(wind_vec))
        t_return = d_home / 5.5
        wh_needed = (p_return * (t_return / 3600.0)) + 1.2
        wh_usable = max(0.0, self.soc - RESERVE_SOC_FLOOR) * PACK_ENERGY_WH
        can_return = (wh_usable >= wh_needed)
        safe_radius_m = (wh_usable / max(1.0, p_return)) * 5.5 * 3600.0
        return wh_needed, wh_usable, can_return, safe_radius_m

# ==========================================
# 7. MOTORIZED DELIVERY WINCH & DROPZONE (MOD 10)
# ==========================================
class WinchDeliveryModule:
    def __init__(self):
        self.state = "STANDBY"
        self.cable_len = 0.08
        self.tension_n = PAYLOAD_WEIGHT_N
        self.touchdown_timer = 0.0
        self.severed = False

    def step(self, drone_z, ground_z=0.0, force_snag=False, dt=0.016):
        if self.severed:
            self.state = "GUILLOTINE"
            self.tension_n = 0.0
            return self.cable_len, self.tension_n

        payload_z = drone_z - self.cable_len
        if self.state == "LOWERING":
            self.cable_len += 1.25 * dt
            self.tension_n = PAYLOAD_WEIGHT_N + float(np.random.normal(0, 0.3))
            if payload_z <= (ground_z + 0.08):
                self.state = "TOUCHDOWN"
                self.touchdown_timer = 0.0
        elif self.state == "TOUCHDOWN":
            self.tension_n = float(np.random.normal(1.8, 0.2))
            self.touchdown_timer += dt
            if self.touchdown_timer >= 1.2:
                self.state = "RETRACTING"
        elif self.state == "RETRACTING":
            self.cable_len = max(0.08, self.cable_len - 2.2 * dt)
            self.tension_n = 1.2 + float(np.random.normal(0, 0.1))
            if self.cable_len <= 0.08:
                self.state = "STANDBY"

        if force_snag:
            self.tension_n = float(random.uniform(44.0, 62.0))

        if self.tension_n >= GUILLOTINE_THRESHOLD_N:
            self.severed = True
            self.state = "GUILLOTINE"
            print(">>> [SAFETY INTERLOCK] PYROTECHNIC GUILLOTINE FIRED: TETHER CUT <<<")

        return self.cable_len, self.tension_n

# ==========================================
# 8. BVLOS COMMS & ANTI-SPOOFING (MOD 11 - HARDENED)
# ==========================================
class BVLOSAntiSpoofModule:
    def __init__(self):
        self.link = "PRIMARY_5G"
        self.is_spoofed = False
        self.spoof_offset = np.zeros(3, dtype=np.float32)
        self.spoof_phase = 0.0

    def evaluate_gps(self, p_true, p_ekf, inject_spoof=False, dt=0.016):
        """
        Hardened Spoofing Generator:
        When inject_spoof is True, smoothly ramps a bounded phantom vector
        up to ~5.5m away (large enough to trip Chi-Square, bounded enough
        to stay on screen and visually demonstrate the spoof ghost!).
        """
        if inject_spoof:
            self.spoof_phase = min(1.0, self.spoof_phase + dt * 1.5)
            # Maximum 5.5 meter lateral displacement
            max_drift = np.array([4.2, 3.5, 0.8], dtype=np.float32)
            self.spoof_offset = max_drift * self.spoof_phase
        else:
            self.spoof_phase = max(0.0, self.spoof_phase - dt * 2.0)
            self.spoof_offset = np.array([4.2, 3.5, 0.8], dtype=np.float32) * self.spoof_phase

        z_gps = p_true + self.spoof_offset + np.random.normal(0, 0.15, 3).astype(np.float32)
        y = z_gps - p_ekf
        S = np.eye(3) * (0.15**2 + 0.10**2)
        chi2 = float(y.T @ np.linalg.inv(S) @ y)
        self.is_spoofed = (chi2 > CHI2_SPOOF_THRESHOLD)
        return z_gps, chi2, self.is_spoofed

# ==========================================
# 9. AVIATION COMPLIANCE & ADS-B (MOD 12)
# ==========================================
class AviationComplianceModule:
    def __init__(self):
        self.intruder_pos = np.array([850.0, 620.0, 95.0], dtype=np.float32)
        self.intruder_vel = np.array([-25.0, -18.0, 0.0], dtype=np.float32)

    def step(self, drone_pos, drone_vel, dt=0.016):
        self.intruder_pos += self.intruder_vel * dt
        if self.intruder_pos[0] < -600.0:
            self.intruder_pos = np.array([900.0, 650.0, 95.0], dtype=np.float32)

        dp = self.intruder_pos - drone_pos
        dv = self.intruder_vel - drone_vel
        dist = float(np.linalg.norm(dp))
        v_sq = float(np.dot(dv, dv))
        t_cpa = -float(np.dot(dp, dv)) / max(1e-4, v_sq)
        d_cpa = float(np.linalg.norm(dp + dv * t_cpa)) if t_cpa > 0 else dist

        is_ra = (dist < 150.0) or (0 < t_cpa < 12.0 and d_cpa < 120.0)
        return {
            "dist": dist, "t_cpa": max(0.0, t_cpa), "d_cpa": d_cpa,
            "is_ra": is_ra
        }

# ==========================================
# 10. VIBRATION FFT & PHM (MOD 13)
# ==========================================
class VibrationPHMModule:
    def __init__(self):
        self.buffer = deque(maxlen=128)
        for _ in range(128): self.buffer.append(0.0)
        self.fault_mode = "NOMINAL"

    def step(self):
        f0 = 85.0
        vib = 0.5 * math.sin(2.0 * math.pi * f0 * time.time())
        if self.fault_mode == "CHIPPED_PROP":
            vib += 7.8 * math.sin(2.0 * math.pi * f0 * time.time())
        elif self.fault_mode == "BEARING_WEAR":
            vib += 6.5 * math.sin(2.0 * math.pi * (f0 * 3.65) * time.time())
        vib += float(np.random.normal(0, 0.35))
        self.buffer.append(vib)

    def compute_metrics(self):
        sig = np.array(self.buffer, dtype=np.float32)
        fft_c = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        v_rms = float(np.sqrt(np.mean(sig ** 2)) * 1.8)
        return v_rms, fft_c

# ==========================================
# 11. 3D PERSPECTIVE PROJECTION & RAYCASTING
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
# 12. UNIFIED MISSION CONTROL ENGINE
# ==========================================
def run_autonomous_system_core_v2():
    pygame.init()
    VIEW_W, VIEW_H = 760, 740
    SIDEBAR_W = 440
    W, H = VIEW_W + SIDEBAR_W, VIEW_H

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Autonomous Drone System Core V2.2 — Hardened Anti-Spoof & Dropzone")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("consolas", 17, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 12, bold=True)

    # 1. Instantiate Autonomous Flight Core Modules
    init_coords = np.array([-3.5, -3.0, 1.2], dtype=np.float32)
    home_coords = np.array([-3.5, -3.0, 0.0], dtype=np.float32)
    target_delivery_point = np.array([3.0, 4.5, 0.0], dtype=np.float32)

    drone = DroneAirframe6DOF(init_coords)
    ekf = ErrorStateEKF15(init_coords)
    controller = GeometricSE3Controller()
    planner = MinimumSnapPlanner()

    # Modules 9-14 Operational Suite
    bms = BatteryManagementPSR(init_soc=0.92, temp_c=18.0)
    winch = WinchDeliveryModule()
    bvlos = BVLOSAntiSpoofModule()
    compliance = AviationComplianceModule()
    phm = VibrationPHMModule()

    def build_flight_plan(start_p, dest_ground_p):
        hover_dest = np.array([dest_ground_p[0], dest_ground_p[1], 2.2], dtype=np.float32)
        mid_pt = 0.5 * (start_p + hover_dest) + np.array([0.0, 0.0, 0.8], dtype=np.float32)
        wps = [start_p.copy(), mid_pt, hover_dest]
        planner.plan(wps, nominal_speed=2.8)

    build_flight_plan(drone.p, target_delivery_point)

    flight_time = 0.0
    cam_yaw, cam_pitch = 0.65, 0.42
    auto_cam_track = True
    default_center = (VIEW_W // 2, VIEW_H // 2 + 50)
    center_viewport = default_center
    running = True

    inject_spoof = False
    force_snag = False

    print("================ AUTONOMOUS DRONE SYSTEM CORE V2.2 ================")
    print(" [LEFT CLICK] : Click Anywhere on Ground to Set New Delivery Dropzone")
    print(" [SPACE]      : Deploy Winch Tether (Automatic Delivery at Target)")
    print(" [C]          : Recenter Camera / Toggle Smooth Auto-Tracking")
    print(" [H]          : Return to Launch Base (RTL)")
    print(" [J]          : Toggle Coordinated GPS Spoofing Attack")
    print(" [S]          : Toggle Winch Cable Snag / Over-Tension")
    print(" [P]          : Trigger ASTM F3322 Ballistic Parachute FTS")
    print(" [V]          : Cycle PHM Vibration Fault (Chipped Prop/Bearing)")
    print(" [G]          : Inject Severe Wind Gust Pulse")
    print(" [ARROW KEYS] : Orbit Perspective View Camera")
    print("===================================================================\n")

    while running:
        dt = 0.016 # 60 Hz physics step

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if mx < VIEW_W:
                    click_gnd = screen_to_ground_raycast((mx, my), (cam_yaw, cam_pitch), center_viewport)
                    target_delivery_point = click_gnd
                    build_flight_plan(drone.p, target_delivery_point)
                    flight_time = 0.0
                    winch.state = "STANDBY"
                    print(f">>> NEW DELIVERY DESTINATION SET: [{click_gnd[0]:.2f}, {click_gnd[1]:.2f}, 0.00]m <<<")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if winch.state == "STANDBY": winch.state = "LOWERING"
                elif event.key == pygame.K_c:
                    # Instant snap-back and recenter!
                    auto_cam_track = True
                    center_viewport = default_center
                    print(">>> CAMERA RECENTERED TO DEFAULT ORIGIN <<<")
                elif event.key == pygame.K_h:
                    target_delivery_point = home_coords
                    build_flight_plan(drone.p, home_coords)
                    flight_time = 0.0
                    print(">>> RETURNING TO LAUNCH BASE (RTL) <<<")
                elif event.key == pygame.K_j:
                    inject_spoof = not inject_spoof
                    status_str = "ACTIVE (SPOOF ATTACK INJECTED)" if inject_spoof else "OFF (NOMINAL GNSS)"
                    print(f">>> [GPS DEFENSE] SPOOFING INJECTION: {status_str} <<<")
                elif event.key == pygame.K_s:
                    force_snag = not force_snag
                elif event.key == pygame.K_p:
                    drone.parachute_deployed = True
                elif event.key == pygame.K_v:
                    modes = ["NOMINAL", "CHIPPED_PROP", "BEARING_WEAR"]
                    phm.fault_mode = modes[(modes.index(phm.fault_mode) + 1) % len(modes)]
                elif event.key == pygame.K_g:
                    drone.wind_gust = np.array([random.uniform(-4.5, 4.5), random.uniform(-4.5, 4.5), 0.0], dtype=np.float32)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: cam_yaw -= 0.03
        if keys[pygame.K_RIGHT]: cam_yaw += 0.03

        # Camera Clamped Auto-Tracking (Bounded to prevent off-screen runaway)
        if auto_cam_track:
            target_cx = VIEW_W // 2 - int(drone.p[0] * 10.0)
            target_cy = VIEW_H // 2 + 50 + int(drone.p[1] * 6.0)
            # Clamp viewport center within safe bounding box
            target_cx = int(np.clip(target_cx, VIEW_W * 0.25, VIEW_W * 0.75))
            target_cy = int(np.clip(target_cy, VIEW_H * 0.35, VIEW_H * 0.75))
            center_viewport = (int(center_viewport[0] * 0.94 + target_cx * 0.06),
                               int(center_viewport[1] * 0.94 + target_cy * 0.06))

        # 1. Trajectory Timeline & Position-Hold Latch
        flight_time = min(planner.total_time, flight_time + dt)
        p_des, v_des, a_des = planner.eval(flight_time)

        dest_hover_pos = np.array([target_delivery_point[0], target_delivery_point[1], 2.2], dtype=np.float32)
        d_to_dest = float(np.linalg.norm(drone.p - dest_hover_pos))

        # Arrival Latch
        is_arrived = (d_to_dest < 0.35) or (flight_time >= planner.total_time)
        if is_arrived:
            p_des = dest_hover_pos.copy()
            v_des.fill(0.0)
            a_des.fill(0.0)
            if winch.state == "STANDBY" and not winch.severed and d_to_dest < 0.4:
                winch.state = "LOWERING"

        # 2. ADS-B Compliance Step
        adsb_stat = compliance.step(drone.p, drone.v, dt=dt)
        if adsb_stat["is_ra"] and not drone.parachute_deployed:
            p_des[2] = min(p_des[2], 1.0)

        # 3. IMU Sensing & 15-State ES-EKF Step
        R_true = drone.R
        accel_body_true = R_true.T @ (drone.last_accel_linear - GRAVITY_G)
        accel_raw = accel_body_true + np.random.normal(0, 0.03, 3).astype(np.float32)
        gyro_raw = drone.omega + np.random.normal(0, 0.005, 3).astype(np.float32)

        ekf.predict(accel_raw, gyro_raw, dt=dt)

        # 4. Anti-GPS Spoofing Chi-Square Gating
        z_gps, chi2, is_spoofed = bvlos.evaluate_gps(drone.p, ekf.p, inject_spoof=inject_spoof, dt=dt)
        if not is_spoofed:
            # Trusted GNSS -> Apply Kalman measurement update
            H_pos = np.zeros((3, 15), dtype=np.float32)
            H_pos[0:3, 0:3] = np.eye(3)
            ekf.update_measurement(H_pos, z_gps - ekf.p, np.eye(3, dtype=np.float32) * 0.02)
        # When spoofed: Update rejected. Drone stays on true internal estimate!

        # Barometer measurement update
        z_baro = drone.p[2] + float(np.random.normal(0, 0.05))
        H_baro = np.zeros((1, 15), dtype=np.float32); H_baro[0, 2] = 1.0
        ekf.update_measurement(H_baro, np.array([z_baro - ekf.p[2]], dtype=np.float32), np.array([[0.05]], dtype=np.float32))

        # 5. Geometric SE(3) Control on Estimated State
        R_est = quat_to_rot(ekf.q)
        w_est = gyro_raw - ekf.b_g
        omega_cmds, f_thrust, tau = controller.compute(ekf.p, ekf.v, R_est, w_est, p_des, v_des, a_des)

        # 6. Physical 6-DOF Integration
        drone.step(omega_cmds, dt=dt)

        # 7. Battery Management System (BMS) Step
        current_draw = max(11.0, f_thrust * 1.80)
        v_pack, cell_v, r0 = bms.step(current_draw, dt=dt)
        wh_need, wh_use, can_rth, psr_r = bms.compute_psr(drone.p, home_coords, drone.wind_gust)

        # 8. Winch Delivery Step
        cable_l, tension = winch.step(drone.p[2], ground_z=0.0, force_snag=force_snag, dt=dt)

        # 9. Vibration FFT Step
        phm.step()
        v_rms, fft_spec = phm.compute_metrics()

        # ==========================================
        # RENDERING: 3D VIEWPORT & GAUGES
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

        # 2. Powerline Hazard Catenary
        w1_s = project_3d(np.array([-6.0, 4.0, 3.2]), cam, center_viewport)[:2]
        w2_s = project_3d(np.array([ 6.0, 4.0, 3.2]), cam, center_viewport)[:2]
        d_wire = math.hypot(drone.p[1] - 4.0, drone.p[2] - 3.2)
        is_emf_alert = (150.0 / max(0.3, d_wire)) >= EMF_WARNING_THRESHOLD_UT
        pygame.draw.line(screen, (255, 60, 80) if is_emf_alert else (255, 180, 40), w1_s, w2_s, 3)

        # 3. Minimum-Snap Spline Curve
        spline_pts = [project_3d(planner.eval((s/50.0)*planner.total_time)[0], cam, center_viewport)[:2] for s in range(51)]
        pygame.draw.lines(screen, (0, 220, 255), False, spline_pts, 2)

        # 4. Target Delivery Ground Ring
        tgt_gnd_sx, tgt_gnd_sy, _ = project_3d(target_delivery_point, cam, center_viewport)
        pygame.draw.circle(screen, (0, 255, 140), (tgt_gnd_sx, tgt_gnd_sy), 14, 2)
        pygame.draw.circle(screen, (255, 255, 255), (tgt_gnd_sx, tgt_gnd_sy), 3)
        screen.blit(font_small.render("DELIVERY DROPZONE", True, (0, 255, 140)), (tgt_gnd_sx + 16, tgt_gnd_sy - 6))

        # 5. Active Drone Airframe
        dsx, dsy, _ = project_3d(drone.p, cam, center_viewport)
        pygame.draw.circle(screen, (0, 240, 255), (dsx, dsy), 8)
        pygame.draw.circle(screen, (255, 255, 255), (dsx, dsy), 3)

        # 6. Spoofed GPS Phantom Ghost (Visualizing Anti-Spoof Defense in Action)
        if inject_spoof or is_spoofed:
            gsx, gsy, _ = project_3d(z_gps, cam, center_viewport)
            # Pulsing red dashed circle
            pygame.draw.circle(screen, (255, 60, 80), (gsx, gsy), 10, 2)
            pygame.draw.line(screen, (255, 60, 80), (dsx, dsy), (gsx, gsy), 1)
            screen.blit(font_small.render("SPOOFED GPS PHANTOM (REJECTED)", True, (255, 60, 80)), (gsx + 12, gsy - 6))

        # Winch cable and payload box
        payload_z = drone.p[2] - cable_l
        psx, psy, _ = project_3d(np.array([drone.p[0], drone.p[1], payload_z]), cam, center_viewport)
        if not winch.severed:
            pygame.draw.line(screen, (255, 60, 80) if tension >= SNAG_THRESHOLD_N else (220, 230, 245), (dsx, dsy), (psx, psy), 2)
            pygame.draw.rect(screen, (255, 200, 40), (psx - 5, psy - 5, 10, 10))
        else:
            screen.blit(font_small.render("[GUILLOTINE SEVERED]", True, (255, 60, 80)), (dsx + 10, dsy))

        # Ground drop-line
        gsx, gsy, _ = project_3d(np.array([drone.p[0], drone.p[1], 0.0]), cam, center_viewport)
        pygame.draw.line(screen, (35, 55, 75), (dsx, dsy), (gsx, gsy), 1)

        # ==========================================
        # PiP INSETS (LOWER VIEWPORT)
        # ==========================================
        pip_w, pip_h = 160, 110
        pip_y = H - pip_h - 18

        # PiP 1: Edge Privacy Redaction Stream
        pygame.draw.rect(screen, (16, 22, 30), (20, pip_y, pip_w, pip_h), border_radius=4)
        pygame.draw.rect(screen, (0, 255, 140), (20, pip_y, pip_w, pip_h), 1)
        screen.blit(font_small.render("EDGE PRIVACY STREAM", True, (0, 255, 140)), (25, pip_y + 6))
        for mx in range(65, 95, 7):
            for my in range(pip_y + 35, pip_y + 70, 7):
                cv = random.randint(110, 190)
                pygame.draw.rect(screen, (cv, cv, cv), (mx, my, 6, 6))
        screen.blit(font_small.render("PII REDACTED (GDPR)", True, (0, 255, 140)), (25, pip_y + 88))

        # PiP 2: Vibration FFT Spectrum
        fft_box_x = 200
        pygame.draw.rect(screen, (16, 22, 30), (fft_box_x, pip_y, pip_w, pip_h), border_radius=4)
        pygame.draw.rect(screen, (35, 50, 70), (fft_box_x, pip_y, pip_w, pip_h), 1)
        screen.blit(font_small.render("1000Hz VIBRATION FFT", True, (0, 220, 255)), (fft_box_x + 6, pip_y + 6))
        for bi in range(min(20, len(fft_spec))):
            bh = int(min(55, fft_spec[bi] * 4.5))
            pygame.draw.line(screen, (0, 240, 255), (fft_box_x + 10 + bi * 7, pip_y + 85), (fft_box_x + 10 + bi * 7, pip_y + 85 - bh), 2)
        screen.blit(font_small.render(f"v_rms: {v_rms:.2f} mm/s", True, (255, 200, 40)), (fft_box_x + 6, pip_y + 88))

        # ==========================================
        # SIDEBAR TELEMETRY HUD
        # ==========================================
        sb_x = VIEW_W + 14
        screen.blit(font_large.render("AUTONOMOUS CORE V2.2", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Hardened Anti-GPS Spoof & Interactive Dropzone", True, (130, 160, 190)), (sb_x, 38))

        # 1. Navigation & Hover State
        p1_y = 75
        screen.blit(font_mono.render("1. NAVIGATION & MISSION STAGE:", True, (240, 240, 240)), (sb_x, p1_y))
        status_txt = "HOVER LOCKED OVER DROPZONE" if is_arrived else "TRANSIT: MIN-SNAP EN-ROUTE"
        status_col = (0, 255, 140) if is_arrived else (0, 220, 255)
        screen.blit(font_small.render(f"Mission Status: {status_txt}", True, status_col), (sb_x + 10, p1_y + 20))
        screen.blit(font_small.render(f"Dest Coordinate: [{target_delivery_point[0]:.2f}, {target_delivery_point[1]:.2f}, 0.00] m", True, (255, 200, 40)), (sb_x + 10, p1_y + 36))
        screen.blit(font_small.render(f"Range to Drop:   d = {d_to_dest:.2f} m | Alt: {drone.p[2]:.2f}m", True, (0, 255, 140)), (sb_x + 10, p1_y + 52))

        # 2. Anti-GPS Spoofing & Chi-Square Innovation
        p2_y = p1_y + 72
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (W - 14, p2_y), 1)
        screen.blit(font_mono.render("2. ANTI-GPS SPOOFING (CHI-SQUARE GATING):", True, (240, 240, 240)), (sb_x, p2_y + 8))
        chi_col = (255, 60, 80) if is_spoofed else (0, 255, 140)
        spoof_tag = f"ATTACK REJECTED! (χ² = {chi2:.1f} > 16.27)" if is_spoofed else f"GNSS VERIFIED (χ² = {chi2:.2f})"
        screen.blit(font_small.render(f"Innovation Gate: {spoof_tag}", True, chi_col), (sb_x + 10, p2_y + 26))
        nav_mode_str = "VIO DEAD-RECKONING (GPS ISOLATED)" if is_spoofed else "FUSED (IMU + GNSS + BARO)"
        screen.blit(font_small.render(f"Navigation State:{nav_mode_str}", True, chi_col), (sb_x + 10, p2_y + 42))

        # 3. Battery BMS & PSR (Module 9)
        p3_y = p2_y + 68
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (W - 14, p3_y), 1)
        screen.blit(font_mono.render("3. BATTERY BMS & PSR ENVELOPE:", True, (240, 240, 240)), (sb_x, p3_y + 8))
        soc_pct = bms.soc * 100.0
        screen.blit(font_small.render(f"Battery SoC:    {soc_pct:.1f}% | V_pack: {v_pack:.2f}V", True, (0, 255, 140)), (sb_x + 10, p3_y + 26))
        screen.blit(font_small.render(f"PSR Radius:     R_safe = {psr_r:.0f}m (Usable: {wh_use:.1f}Wh)", True, (0, 220, 255)), (sb_x + 10, p3_y + 42))

        # 4. Winch Delivery (Module 10)
        p4_y = p3_y + 68
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (W - 14, p4_y), 1)
        screen.blit(font_mono.render("4. WINCH TETHER & TOUCHDOWN (MOD 10):", True, (240, 240, 240)), (sb_x, p4_y + 8))
        screen.blit(font_small.render(f"Winch State:    {winch.state} | Deployed: {cable_l:.2f}m", True, (0, 220, 255)), (sb_x + 10, p4_y + 26))
        t_col = (255, 60, 80) if tension >= SNAG_THRESHOLD_N else (0, 255, 140)
        screen.blit(font_small.render(f"Cable Tension:  F = {tension:.1f} N (Load-Cell)", True, t_col), (sb_x + 10, p4_y + 42))

        # 5. Interactive Instruction Card
        p5_y = p4_y + 68
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p5_y), (W - 14, p5_y), 1)
        screen.blit(font_mono.render("OPERATOR CONTROL CARD:", True, (255, 200, 40)), (sb_x, p5_y + 8))
        screen.blit(font_small.render("• [LEFT CLICK] anywhere on ground to set destination", True, (240, 240, 240)), (sb_x + 10, p5_y + 26))
        screen.blit(font_small.render("• [J] Toggle GPS Spoofing (Watch drone reject phantom)", True, (255, 60, 80)), (sb_x + 10, p5_y + 42))
        screen.blit(font_small.render("• [C] Recenter camera / toggle auto-tracking", True, (0, 220, 255)), (sb_x + 10, p5_y + 58))
        screen.blit(font_small.render("• [H] Return to Launch Base (RTL)", True, (0, 255, 140)), (sb_x + 10, p5_y + 74))

        # Bottom Mission Status Banner
        bot_y = H - 50
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y - 8), (W - 14, bot_y - 8), 1)
        if drone.parachute_deployed:
            b_txt, b_col = "EMERGENCY: PARACHUTE TERMINATION FLIGHT", (255, 140, 20)
        elif is_spoofed:
            b_txt, b_col = "ANTI-SPOOF ACTIVE: GPS REJECTED -> VIO LOCK", (255, 60, 80)
        elif is_arrived:
            b_txt, b_col = "ARRIVAL CONFIRMED: HOVER LOCKED OVER TARGET", (0, 255, 140)
        else:
            b_txt, b_col = "AUTONOMOUS FLIGHT: MIN-SNAP TO DESTINATION", (0, 220, 255)
        screen.blit(font_mono.render(b_txt, True, b_col), (sb_x + 8, bot_y))

        # Header HUD
        head_txt = font_large.render(f"Core V2.2 | Dest Dist: {d_to_dest:.2f}m | Alt: {drone.p[2]:.2f}m | Anti-Spoof Active", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("Bounded Phantom Anti-Spoof Defense, Fail-Safe Bounded Camera Tracker & Interactive Dropzone Dispatch", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    mp.freeze_support()
    print("Launching Unified Autonomous Drone System Core V2.2 (Hardened)...")
    run_autonomous_system_core_v2()