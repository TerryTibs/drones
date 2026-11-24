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
import pygame
import sys

# ==========================================
# 0. PHYSICAL AIRFRAME & MISSION CONSTANTS
# ==========================================
MASS_KG = 1.25
GRAVITY_G = np.array([0.0, 0.0, -9.80665], dtype=np.float32)
ARM_LENGTH_M = 0.22
INERTIA_MATRIX = np.diag([0.015, 0.015, 0.025]).astype(np.float32)
INV_INERTIA = np.linalg.inv(INERTIA_MATRIX)

C_T = 1.25e-5            # Thrust coeff: T = C_T * omega^2
C_Q = 2.40e-7            # Torque coeff: Q = C_Q * omega^2
MOTOR_TAU_S = 0.035      # Motor response time lag
MAX_RPM = 8600.0
MAX_OMEGA = (MAX_RPM * 2.0 * math.pi) / 60.0
HOVER_THRUST = MASS_KG * 9.80665
HOVER_OMEGA = math.sqrt((HOVER_THRUST / 4.0) / C_T)

# Atmospheric & Sensor Defaults
P_0 = 1013.25
T_0 = 288.15
L_LAPSE = 0.0065
SPEED_OF_SOUND = 343.2
B_EARTH_TOTAL_UT = 52.0
EMF_WARNING_THRESHOLD_UT = 18.0

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

    def step(self, motor_cmds, dt=0.01):
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
        f_drag = -np.array([0.15, 0.15, 0.25], dtype=np.float32) * self.v
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
        self.wind_gust *= 0.995
        return f_thrust, torques, self.last_accel_linear

# ==========================================
# 3. 15-STATE ERROR-STATE EXTENDED KALMAN FILTER
# ==========================================
class ErrorStateEKF15:
    def __init__(self, init_pos):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.b_a = np.zeros(3, dtype=np.float32)
        self.b_g = np.zeros(3, dtype=np.float32)

        self.P = np.diag([0.2, 0.2, 0.2,  0.1, 0.1, 0.1,  0.02, 0.02, 0.02,  0.05, 0.05, 0.05,  0.01, 0.01, 0.01]).astype(np.float32)
        self.Q_d = np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-6, 1e-6, 1e-6]).astype(np.float32)

    def predict(self, accel_raw, gyro_raw, dt=0.01):
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
        # Maintain symmetry
        self.P = 0.5 * (self.P + self.P.T)

    def update_measurement(self, H, innovation, R_cov):
        S = H @ self.P @ H.T + R_cov
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        delta_x = (K @ innovation).flatten()
        delta_x = np.clip(delta_x, -5.0, 5.0)

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
# 4. GEOMETRIC SE(3) NON-LINEAR TRACKING CONTROL
# ==========================================
class GeometricSE3Controller:
    def __init__(self):
        self.k_p = np.array([5.5, 5.5, 9.0], dtype=np.float32)
        self.k_v = np.array([3.8, 3.8, 5.8], dtype=np.float32)
        self.k_R = 1.75
        self.k_w = 0.32

    def compute(self, p_est, v_est, R_est, w_est, p_des, v_des, a_des, yaw_des=0.0):
        e_p = p_est - p_des
        e_v = v_est - v_des

        F_d = -self.k_p * e_p - self.k_v * e_v + MASS_KG * (-GRAVITY_G) + MASS_KG * a_des
        b3 = R_est[:, 2]
        f_thrust = float(np.dot(F_d, b3))
        f_thrust = max(0.1, min(30.0, f_thrust))

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
            T_motors = np.clip(np.linalg.solve(A, b), 0.05, 9.0)
        except np.linalg.LinAlgError:
            T_motors = np.full(4, f_thrust / 4.0, dtype=np.float32)

        omega_cmds = np.sqrt(T_motors / C_T)
        return omega_cmds, f_thrust, tau, e_p, e_R

# ==========================================
# 5. PIECEWISE 7TH-ORDER MINIMUM-SNAP PLANNER
# ==========================================
class MinimumSnapPlanner:
    def __init__(self):
        self.is_solved = False
        self.waypoints = []
        self.times = []
        self.total_time = 0.0

    def plan(self, waypoints, nominal_speed=3.2):
        self.waypoints = [np.array(w, dtype=np.float32) for w in waypoints]
        M = len(self.waypoints) - 1
        if M < 1: return False

        self.times = [max(1.2, float(np.linalg.norm(self.waypoints[k+1] - self.waypoints[k])) / nominal_speed) for k in range(M)]
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
# 6. MULTI-MODAL SENSORY SIMULATION
# ==========================================
class UnifiedSensorSuite:
    def __init__(self):
        self.powerline_y = 4.0
        self.powerline_z = 3.2
        self.obstacles = [
            {"pos": np.array([ 2.2, 2.5, 1.8]), "radius": 0.6},
            {"pos": np.array([-2.5, 4.5, 2.0]), "radius": 0.7}
        ]

    def poll_all(self, pos_true, vel_true, R_true, dt=0.01):
        # 1. Barometer
        z_true = max(0.0, float(pos_true[2]))
        p_stat = P_0 * ((1.0 - (L_LAPSE * z_true) / T_0) ** 5.256)
        z_baro = 44330.8 * (1.0 - ((p_stat / P_0) ** 0.190284)) + float(np.random.normal(0.0, 0.08))

        # 2. Pitot Airspeed
        rho = (p_stat * 100.0) / (287.058 * (T_0 - L_LAPSE * z_true))
        v_air = vel_true
        q_dyn = 0.5 * rho * (np.linalg.norm(v_air) ** 2)
        v_tas = math.sqrt(2.0 * max(0.0, q_dyn) / rho)

        # 3. 60 Hz Powerline EMF
        d_wire = math.hypot(pos_true[1] - self.powerline_y, pos_true[2] - self.powerline_z)
        emf_rms = float(150.0 / max(0.3, d_wire))
        is_emf_alert = (emf_rms >= EMF_WARNING_THRESHOLD_UT)

        # 4. Sonar ToF Clearances (Front, Rear, Left, Right, Up, Down)
        tofs = []
        cardinals = [np.array([0, 1, 0]), np.array([0, -1, 0]), np.array([-1, 0, 0]), np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, 0, -1])]
        for card in cardinals:
            d_hit = 8.0
            for obs in self.obstacles:
                oc = pos_true - obs["pos"]
                b = float(np.dot(oc, card))
                c = float(np.dot(oc, oc)) - obs["radius"]**2
                disc = b**2 - c
                if disc >= 0:
                    d_c = -b - math.sqrt(disc)
                    if 0 < d_c < d_hit: d_hit = d_c
            tofs.append(float(np.clip(d_hit, 0.1, 8.0)))

        return {
            "z_baro": z_baro, "v_tas": v_tas, "rho": rho,
            "emf_rms": emf_rms, "is_emf_alert": is_emf_alert, "tofs": tofs,
            "min_sonar": min(tofs)
        }

# ==========================================
# 7. UNIFIED AUTONOMOUS MISSION HUD & CONTROL
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

def run_unified_drone_system():
    pygame.init()
    VIEW_W, VIEW_H = 740, 720
    SIDEBAR_W = 380
    W, H = VIEW_W + SIDEBAR_W, VIEW_H

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Module 8: Unified Autonomous Drone System Core Simulation")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("consolas", 18, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 13, bold=True)

    # 1. Instantiate Autonomous Flight Core Modules
    init_coords = np.array([-4.0, -3.5, 1.2], dtype=np.float32)
    drone = DroneAirframe6DOF(init_coords)
    ekf = ErrorStateEKF15(init_coords)
    controller = GeometricSE3Controller()
    planner = MinimumSnapPlanner()
    sensors = UnifiedSensorSuite()

    # Pre-planned Autonomous Mission Waypoints
    mission_waypoints = [
        init_coords,
        np.array([-2.0,  0.5, 2.8]),
        np.array([ 1.5,  2.5, 3.8]),
        np.array([-1.2,  4.5, 2.0]),
        np.array([ 3.5,  5.5, 1.5])  # Destination Coordinate
    ]
    planner.plan(mission_waypoints, nominal_speed=3.0)

    flight_time = 0.0
    cam_yaw, cam_pitch = 0.65, 0.42
    center_viewport = (VIEW_W // 2, VIEW_H // 2 + 50)
    running = True
    pulse_phase = 0.0

    print("================ AUTONOMOUS DRONE SYSTEM ONLINE ================")
    print(" [N]          : Dispatch Random Destination Coordinates")
    print(" [G]          : Inject 35 km/h Microburst Wind Gust")
    print(" [H]          : Return to Launch Base (RTL)")
    print(" [SPACE]      : Emergency Flight Hold / Pause")
    print(" [ARROW KEYS] : Orbit 3D Sensor Camera")
    print("=================================================================\n")

    while running:
        dt = 0.016  # Consistent 60 Hz physics step
        pulse_phase += dt * 4.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    drone.wind_gust = np.array([random.uniform(-4.5, 4.5), random.uniform(-4.5, 4.5), random.uniform(-1.5, 1.5)], dtype=np.float32)
                elif event.key == pygame.K_n:
                    new_dest = np.array([random.uniform(-4.0, 4.0), random.uniform(-2.0, 6.0), random.uniform(1.2, 4.0)], dtype=np.float32)
                    mid_pt = 0.5 * (drone.p + new_dest) + np.array([random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5), 0.8], dtype=np.float32)
                    mission_waypoints = [drone.p.copy(), mid_pt, new_dest]
                    planner.plan(mission_waypoints, nominal_speed=3.0)
                    flight_time = 0.0
                elif event.key == pygame.K_h:
                    mission_waypoints = [drone.p.copy(), np.array([0.0, 0.0, 2.0], dtype=np.float32)]
                    planner.plan(mission_waypoints, nominal_speed=2.5)
                    flight_time = 0.0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: cam_yaw -= 0.03
        if keys[pygame.K_RIGHT]: cam_yaw += 0.03

        # Advance autonomous mission timeline
        flight_time = min(planner.total_time, flight_time + dt)
        p_des, v_des, a_des = planner.eval(flight_time)

        # 1. Multi-Modal Sensor Readings
        R_true = drone.R
        sens = sensors.poll_all(drone.p, drone.v, R_true, dt=dt)

        # 2. Proper Accelerometer Specific Force (Body Frame)
        accel_body_true = R_true.T @ (drone.last_accel_linear - GRAVITY_G)
        accel_raw = accel_body_true + np.random.normal(0, 0.04, 3).astype(np.float32)
        gyro_raw = drone.omega + np.random.normal(0, 0.005, 3).astype(np.float32)

        # 3. 15-State ES-EKF High-Rate Strapdown Propagation
        ekf.predict(accel_raw, gyro_raw, dt=dt)

        # Apply Sensor Measurement Updates
        ekf.update_measurement(np.array([[0,0,1] + [0]*12], dtype=np.float32), np.array([sens["z_baro"] - ekf.p[2]]), np.array([[0.06]], dtype=np.float32))
        p_meas = drone.p + np.random.normal(0, 0.10, 3).astype(np.float32)
        H_pos = np.zeros((3, 15), dtype=np.float32); H_pos[0:3, 0:3] = np.eye(3)
        ekf.update_measurement(H_pos, p_meas - ekf.p, np.eye(3, dtype=np.float32) * 0.02)

        # 4. Geometric SE(3) Control on Estimated State (with Rate Damping)
        R_est = quat_to_rot(ekf.q)
        omega_est = gyro_raw - ekf.b_g
        omega_cmds, f_thrust, tau, e_p, e_R = controller.compute(ekf.p, ekf.v, R_est, omega_est, p_des, v_des, a_des)

        # 5. Continuous 6-DOF Physical Flight Step
        drone.step(omega_cmds, dt=dt)

        # ==========================================
        # RENDERING VIEWPORT
        # ==========================================
        screen.fill((10, 12, 18))
        pygame.draw.line(screen, (35, 45, 60), (VIEW_W, 0), (VIEW_W, H), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (W, 70), 2)

        cam = (cam_yaw, cam_pitch)

        # Ground Grid
        for gx in range(-5, 6):
            sx1, sy1, _ = project_3d(np.array([gx * 1.5, -4.0, 0.0]), cam, center_viewport)
            sx2, sy2, _ = project_3d(np.array([gx * 1.5, 8.0, 0.0]), cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        # Powerline Hazard Catenary (500 kV)
        w1_s = project_3d(np.array([-6.0, sensors.powerline_y, sensors.powerline_z]), cam, center_viewport)[:2]
        w2_s = project_3d(np.array([ 6.0, sensors.powerline_y, sensors.powerline_z]), cam, center_viewport)[:2]
        line_col = (255, 60, 80) if sens["is_emf_alert"] else (255, 180, 40)
        pygame.draw.line(screen, line_col, w1_s, w2_s, 3)

        # Spline Trajectory Path
        spline_pts = [project_3d(planner.eval((s/100.0)*planner.total_time)[0], cam, center_viewport)[:2] for s in range(101)]
        pygame.draw.lines(screen, (0, 220, 255), False, spline_pts, 2)

        # Target Destination
        dest_sx, dest_sy, _ = project_3d(mission_waypoints[-1], cam, center_viewport)
        pygame.draw.circle(screen, (255, 200, 40), (dest_sx, dest_sy), 8)
        pygame.draw.circle(screen, (255, 255, 255), (dest_sx, dest_sy), 3)

        # 3D Drone & EKF Uncertainty Ellipsoid
        dsx, dsy, _ = project_3d(drone.p, cam, center_viewport)
        sigma_r = int(min(60, max(6, 3.0 * math.sqrt(max(1e-4, float(ekf.P[0,0]))) * 46.0)))
        pygame.draw.circle(screen, (255, 200, 40), (dsx, dsy), sigma_r, 1)
        pygame.draw.circle(screen, (0, 240, 255), (dsx, dsy), 8)
        pygame.draw.circle(screen, (255, 255, 255), (dsx, dsy), 3)

        # Ground drop-line
        ground_pt = np.array([drone.p[0], drone.p[1], 0.0])
        gsx, gsy, _ = project_3d(ground_pt, cam, center_viewport)
        pygame.draw.line(screen, (35, 55, 75), (dsx, dsy), (gsx, gsy), 1)

        # PiP 1: Multi-Spectral Thermal LWIR Camera (Bottom Left)
        pip_w, pip_h = 160, 110
        pygame.draw.rect(screen, (16, 22, 30), (20, H - pip_h - 20, pip_w, pip_h), border_radius=4)
        pygame.draw.rect(screen, (40, 55, 75), (20, H - pip_h - 20, pip_w, pip_h), 1)
        screen.blit(font_small.render("FLIR THERMAL LWIR", True, (255, 180, 40)), (25, H - pip_h - 15))
        t_wire_txt = "WIRE HOTSPOT (78°C)!" if sens["is_emf_alert"] else "AMBIENT CLEAR"
        screen.blit(font_small.render(t_wire_txt, True, (255, 60, 80) if sens["is_emf_alert"] else (0, 255, 140)), (25, H - pip_h + 3))

        # PiP 2: Ultrasonic Sonar Radar Scope
        scope_cx, scope_cy = 250, H - 75
        pygame.draw.circle(screen, (16, 24, 34), (scope_cx, scope_cy), 48)
        pygame.draw.circle(screen, (35, 55, 75), (scope_cx, scope_cy), 48, 1)
        for i, tof in enumerate(sens["tofs"][:4]):
            ang = i * (math.pi / 2.0)
            dist_px = int((tof / 8.0) * 44.0)
            pygame.draw.circle(screen, (0, 240, 255), (int(scope_cx + math.sin(ang)*dist_px), int(scope_cy - math.cos(ang)*dist_px)), 3)
        screen.blit(font_small.render("SONAR ToF SCOPE", True, (130, 160, 190)), (scope_cx - 40, H - 20))

        # ==========================================
        # FULL MISSION CONTROL HUD (SIDEBAR)
        # ==========================================
        sb_x = VIEW_W + 14
        screen.blit(font_large.render("AUTONOMOUS DRONE CORE", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Full-Stack Multi-Spectral Autonomous Navigation", True, (130, 160, 190)), (sb_x, 38))

        # Mission Status
        d_target = float(np.linalg.norm(drone.p - mission_waypoints[-1]))
        screen.blit(font_mono.render(f"FLIGHT PHASE: {'EN-ROUTE' if d_target > 0.25 else 'WAYPOINT ARRIVAL'}", True, (0, 255, 140)), (sb_x + 10, 85))
        screen.blit(font_small.render(f"Destination:  [{mission_waypoints[-1][0]:.2f}, {mission_waypoints[-1][1]:.2f}, {mission_waypoints[-1][2]:.2f}] m", True, (255, 200, 40)), (sb_x + 10, 105))
        screen.blit(font_small.render(f"Range to Go:  d = {d_target:.2f} m", True, (0, 220, 255)), (sb_x + 10, 122))

        # EKF State Estimates
        pygame.draw.line(screen, (38, 44, 60), (sb_x, 145), (W - 14, 145), 1)
        screen.blit(font_mono.render("15-STATE ERROR EKF NAVIGATION:", True, (240, 240, 240)), (sb_x, 155))
        r_d, p_d, y_d = rot_to_euler(R_est)
        pos_err_cm = float(np.linalg.norm(drone.p - ekf.p)) * 100.0
        screen.blit(font_small.render(f"Est Pos (X,Y,Z): [{ekf.p[0]:.2f}, {ekf.p[1]:.2f}, {ekf.p[2]:.2f}] m", True, (0, 255, 140)), (sb_x + 10, 175))
        screen.blit(font_small.render(f"Est Vel |v|:     {float(np.linalg.norm(ekf.v)):.2f} m/s", True, (0, 220, 255)), (sb_x + 10, 192))
        screen.blit(font_small.render(f"Attitude (RPY):  [{r_d:+.1f}°, {p_d:+.1f}°, {y_d:+.1f}°]", True, (160, 180, 205)), (sb_x + 10, 209))
        screen.blit(font_small.render(f"EKF Tracking:    Err = {pos_err_cm:.1f} cm (3σ: ±{sigma_r/46.0:.2f}m)", True, (0, 255, 140)), (sb_x + 10, 226))

        # Atmospheric & Sensors
        pygame.draw.line(screen, (38, 44, 60), (sb_x, 248), (W - 14, 248), 1)
        screen.blit(font_mono.render("ENVIRONMENT & ATMOSPHERE:", True, (240, 240, 240)), (sb_x, 258))
        screen.blit(font_small.render(f"Baro Altitude:   z = {sens['z_baro']:.2f} m (Hypsometric)", True, (0, 220, 255)), (sb_x + 10, 278))
        screen.blit(font_small.render(f"True Airspeed:   TAS = {sens['v_tas']:.2f} m/s (Pitot q)", True, (0, 255, 140)), (sb_x + 10, 295))
        emf_col = (255, 60, 80) if sens["is_emf_alert"] else (0, 255, 140)
        screen.blit(font_small.render(f"60 Hz Power EMF: B = {sens['emf_rms']:.1f} uT (Biot-Savart)", True, emf_col), (sb_x + 10, 312))
        screen.blit(font_small.render(f"Min Sonar Echo:  d = {sens['min_sonar']:.2f} m (40 kHz ToF)", True, (160, 180, 205)), (sb_x + 10, 329))

        # Actuation & Motors
        pygame.draw.line(screen, (38, 44, 60), (sb_x, 350), (W - 14, 350), 1)
        screen.blit(font_mono.render("6-DOF SE(3) MOTOR DISPATCH:", True, (240, 240, 240)), (sb_x, 360))
        rpms = [(w * 60.0) / (2.0 * math.pi) for w in drone.motor_omegas]
        for m_i in range(4):
            my = 380 + m_i * 16
            bar_len = int(min(170, (rpms[m_i] / MAX_RPM) * 170))
            screen.blit(font_small.render(f"M{m_i+1} RPM: {rpms[m_i]:.0f}", True, (160, 180, 205)), (sb_x + 10, my))
            pygame.draw.rect(screen, (24, 32, 45), (sb_x + 120, my + 2, 170, 7), border_radius=2)
            m_col = (0, 255, 140) if rpms[m_i] < 7200 else (255, 65, 80)
            pygame.draw.rect(screen, m_col, (sb_x + 120, my + 2, bar_len, 7), border_radius=2)

        # Flight Director Banner
        bot_y = H - 50
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y - 10), (W - 14, bot_y - 10), 1)
        if sens["is_emf_alert"]:
            screen.blit(font_mono.render("ALERT: POWERLINE EMF REPULSION ACTIVE", True, (255, 60, 80)), (sb_x + 10, bot_y))
        elif d_target < 0.25:
            screen.blit(font_mono.render("MISSION ACCOMPLISHED: HOVER LOCKED", True, (0, 255, 140)), (sb_x + 10, bot_y))
        else:
            screen.blit(font_mono.render("AUTONOMOUS TRANSIT: MIN-SNAP OPTIMAL", True, (0, 220, 255)), (sb_x + 10, bot_y))

        # Header UI
        head_txt = font_large.render(f"Autonomous Drone | Dest Range: {d_target:.2f}m | Alt: {drone.p[2]:.2f}m", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("Integrated System: Min-Snap Spline Planner -> Multi-Spectral/EMF/Sonar -> 15-State EKF -> Geometric SE(3) Control", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

# ==========================================
# 8. MAIN ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    mp.freeze_support()
    print("Initializing Unified Autonomous Drone Control System...")
    run_unified_drone_system()