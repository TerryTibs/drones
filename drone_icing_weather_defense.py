"""
=============================================================================
PHASE 2: EXTREME WEATHER AERODYNAMICS & IN-FLIGHT ICE DEFENSE
File: drone_icing_weather_defense.py
=============================================================================
- Real-Time Propeller Leading-Edge Rime Ice Accretion Model (dm/dt)
- Aerodynamic Camber Ruin: Progressive c_T Loss (-40%) & c_Q Torque Surge (+180%)
- Aero-Thermal Figure-of-Merit (FOM) Observer for Autonomous Ice Detection
- Centrifugal Shearing Pulse Countermeasure & Piezoelectric Hub Vibration
- Interactive Cloud Injection [C], De-Icing Toggle [D], and Flight Simulator
=============================================================================
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

import time
import math
import random
from collections import deque
import numpy as np
import pygame
import sys

# ==========================================
# 0. PHYSICAL AIRFRAME & ICING CONSTANTS
# ==========================================
MASS_KG = 1.45                       # Base airframe mass
GRAVITY_G = np.array([0.0, 0.0, -9.80665], dtype=np.float32)
G_ACCEL = 9.80665
ARM_LENGTH_M = 0.23
INERTIA_MATRIX = np.diag([0.018, 0.018, 0.030]).astype(np.float32)
INV_INERTIA = np.linalg.inv(INERTIA_MATRIX)

# Clean Propeller Aerodynamics
C_T_CLEAN = 1.30e-5                  # Clean thrust coefficient
C_Q_CLEAN = 2.40e-7                  # Clean torque coefficient
MOTOR_TAU_S = 0.035
MAX_RPM = 8800.0
MAX_OMEGA = (MAX_RPM * 2.0 * math.pi) / 60.0
HOVER_THRUST = MASS_KG * G_ACCEL

# Freezing Cloud & Atmospheric Parameters
FREEZING_TEMP_C = -4.5               # Sub-zero ambient temperature
FREEZING_RH_PCT = 94.0               # High relative humidity
LIQUID_WATER_CONTENT = 0.45          # LWC in g/m^3 (Moderate-to-severe freezing fog)
MAX_ICE_MASS_PER_PROP_G = 14.5       # Maximum ice buildup before catastrophic stall

# Icing Thresholds
ICE_DETECTION_EFF_FLOOR = 0.58       # Aerodynamic Figure of Merit threshold for alarm
SHEAR_FORCE_SHED_N = 18.5            # Centrifugal shear force required to dislodge ice

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
# 2. IN-FLIGHT PROPELLER ICE ACCRETION MODEL
# ==========================================
class PropellerIcingDynamics:
    """
    Simulates:
    1. Continuous leading-edge rime ice accumulation dm/dt.
    2. Degradation of c_T and amplification of profile drag c_Q.
    3. Structural mass penalty and vibration unbalance.
    4. Centrifugal shedding and active pulse de-icing.
    """
    def __init__(self):
        # Ice mass per rotor in grams
        self.ice_mass_g = np.zeros(4, dtype=np.float32)
        self.in_freezing_cloud = False
        self.deicing_active = False
        self.deicing_timer = 0.0
        self.shed_event_count = 0

    def step(self, motor_omegas, ambient_temp_c, dt=0.016):
        # 1. Ice Accretion: Occurs when in freezing cloud and temperature <= 1.0 C
        if self.in_freezing_cloud and ambient_temp_c <= 1.0:
            for i in range(4):
                tip_speed = motor_omegas[i] * 0.127 # ~60-120 m/s
                # dm/dt = collision_efficiency * LWC * v_tip * Area
                d_mass = 0.0035 * LIQUID_WATER_CONTENT * (tip_speed / 80.0) * dt
                self.ice_mass_g[i] = min(MAX_ICE_MASS_PER_PROP_G, self.ice_mass_g[i] + d_mass)

        # 2. Centrifugal Shearing & Active De-Icing Pulses
        if self.deicing_active:
            self.deicing_timer += dt
            for i in range(4):
                # Centrifugal force: F_c = m * omega^2 * r
                omega = motor_omegas[i]
                r_effective = 0.09 # meters along blade
                m_kg = (self.ice_mass_g[i] / 1000.0)
                f_centrifugal = m_kg * (omega ** 2) * r_effective

                # If RPM pulse + ultrasonic vibration exceeds adhesive strength, shed ice!
                if f_centrifugal > SHEAR_FORCE_SHED_N or self.deicing_timer > 0.35:
                    if self.ice_mass_g[i] > 0.5:
                        self.ice_mass_g[i] = max(0.0, self.ice_mass_g[i] - 12.0 * dt)
                        self.shed_event_count += 1

            if self.deicing_timer > 1.4:
                self.deicing_active = False
                self.deicing_timer = 0.0
                print(">>> [DE-ICING COMPLETE] CENTRIFUGAL PULSES CONCLUDED: ROTORS SHED <<<")

        # 3. Aerodynamic Coefficients as Functions of Ice Mass
        # Max ice mass reduces c_T by 40% and increases c_Q by 180%
        c_t_current = np.zeros(4, dtype=np.float32)
        c_q_current = np.zeros(4, dtype=np.float32)

        for i in range(4):
            ice_ratio = min(1.0, self.ice_mass_g[i] / MAX_ICE_MASS_PER_PROP_G)
            c_t_current[i] = C_T_CLEAN * (1.0 - 0.40 * ice_ratio)
            c_q_current[i] = C_Q_CLEAN * (1.0 + 1.80 * (ice_ratio ** 1.3))

        return c_t_current, c_q_current, self.ice_mass_g

# ==========================================
# 3. 6-DOF AIRFRAME WITH ICED BLADE MECHANICS
# ==========================================
class IcingAwareDroneAirframe:
    def __init__(self, init_pos):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.R = np.eye(3, dtype=np.float32)
        self.omega = np.zeros(3, dtype=np.float32)
        self.motor_omegas = np.full(4, 520.0, dtype=np.float32)
        self.icing = PropellerIcingDynamics()
        self.accel_linear = np.zeros(3, dtype=np.float32)
        self.wind_gust = np.zeros(3, dtype=np.float32)
        self.total_power_watts = 160.0

    def step(self, motor_cmds, dt=0.016):
        # First-order motor dynamic response
        cmds = np.clip(motor_cmds, 0.0, MAX_OMEGA)
        self.motor_omegas += ((cmds - self.motor_omegas) / MOTOR_TAU_S) * dt

        # Step Icing Model
        c_t_blades, c_q_blades, ice_masses = self.icing.step(self.motor_omegas, FREEZING_TEMP_C, dt=dt)

        # Individual rotor thrust and torque with degraded coefficients
        T = c_t_blades * (self.motor_omegas ** 2)
        Q = c_q_blades * (self.motor_omegas ** 2)

        f_thrust = float(np.sum(T))
        arm = ARM_LENGTH_M / math.sqrt(2.0)
        tau_x = arm * (-T[0] + T[1] + T[2] - T[3])
        tau_y = arm * (T[0] + T[1] - T[2] - T[3])
        tau_z = -Q[0] + Q[1] - Q[2] + Q[3]
        torques = np.array([tau_x, tau_y, tau_z], dtype=np.float32)

        # Additional ice weight added to airframe
        total_ice_mass_kg = float(np.sum(ice_masses)) / 1000.0
        current_mass = MASS_KG + total_ice_mass_kg

        f_body = np.array([0.0, 0.0, f_thrust], dtype=np.float32)
        f_drag = -np.array([0.22, 0.22, 0.35], dtype=np.float32) * self.v
        f_gravity = current_mass * GRAVITY_G

        self.accel_linear = (self.R @ f_body + f_gravity + f_drag + self.wind_gust) / current_mass
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

        # Electrical power consumption surge under ice drag
        p_mech = float(np.sum(Q * self.motor_omegas))
        self.total_power_watts = (p_mech / 0.85) + 18.0

        return f_thrust, torques, self.total_power_watts, total_ice_mass_kg

# ==========================================
# 4. AERO-THERMAL EFFICIENCY OBSERVER & CONTROLLER
# ==========================================
class IcingDefenseController:
    """
    Monitors Aero-Propulsive Figure of Merit (FOM).
    When icing degrades efficiency below threshold, commands automatic
    centrifugal shedding pulse cycles and adapts hover attitude.
    """
    def __init__(self):
        self.k_p = np.array([6.5, 6.5, 11.5], dtype=np.float32)
        self.k_v = np.array([4.4, 4.4, 7.0], dtype=np.float32)
        self.k_i = np.array([1.2, 1.2, 2.5], dtype=np.float32)
        self.k_R = 1.90
        self.k_w = 0.38
        self.e_p_integral = np.zeros(3, dtype=np.float32)

        self.fom_efficiency = 0.76   # Nominal clean figure of merit
        self.ice_detected = False
        self.shed_cooldown = 0.0

    def compute(self, drone, p_des, v_des, a_des, dt=0.016):
        # 1. Aero-Propulsive Figure of Merit (FOM) Observer:
        # eta_aero = T^(3/2) / (P_elec * sqrt(2 * rho * A_disk))
        f_thrust_meas = float(np.sum(C_T_CLEAN * (drone.motor_omegas ** 2))) # Expected nominal thrust
        p_elec = max(20.0, drone.total_power_watts)
        rho = 1.225
        a_disk = 4.0 * math.pi * (0.127 ** 2)

        raw_fom = (f_thrust_meas ** 1.5) / (p_elec * math.sqrt(2.0 * rho * a_disk) + 1e-4)
        # Low-pass filter to smooth turbulence
        self.fom_efficiency += (raw_fom - self.fom_efficiency) * 3.0 * dt
        self.fom_efficiency = float(np.clip(self.fom_efficiency, 0.25, 0.85))

        # Ice detection trigger: Low FOM efficiency + cold ambient air
        if self.fom_efficiency < ICE_DETECTION_EFF_FLOOR and FREEZING_TEMP_C <= 1.0:
            self.ice_detected = True
        else:
            self.ice_detected = False

        # Autonomous Centrifugal Shearing Trigger
        self.shed_cooldown = max(0.0, self.shed_cooldown - dt)
        if self.ice_detected and not drone.icing.deicing_active and self.shed_cooldown <= 0.0:
            drone.icing.deicing_active = True
            self.shed_cooldown = 4.5
            print(">>> [AERO OBSERVER] EFFICIENCY COLLAPSE DETECTED: ENGAGING ACTIVE DE-ICING PULSES <<<")

        # 2. Geometric SE(3) Position Control
        e_p = drone.p - p_des
        e_v = drone.v - v_des
        self.e_p_integral += e_p * dt
        self.e_p_integral = np.clip(self.e_p_integral, -1.2, 1.2)

        F_d = -self.k_p * e_p - self.k_v * e_v - self.k_i * self.e_p_integral + (MASS_KG * (-GRAVITY_G)) + MASS_KG * a_des
        b3 = drone.R[:, 2]
        f_thrust = float(np.dot(F_d, b3))
        f_thrust = max(0.1, min(38.0, f_thrust))

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
            [-0.018, 0.018, -0.018, 0.018]
        ], dtype=np.float32)
        b = np.array([f_thrust, tau[0], tau[1], tau[2]], dtype=np.float32)
        try:
            T_motors = np.clip(np.linalg.solve(A, b), 0.05, 9.8)
        except np.linalg.LinAlgError:
            T_motors = np.full(4, f_thrust / 4.0, dtype=np.float32)

        # Baseline nominal RPM
        omega_cmds = np.sqrt(T_motors / C_T_CLEAN)

        # 3. Superimpose High-Frequency Centrifugal De-Icing Pulses
        if drone.icing.deicing_active:
            # Alternating diagonal pairs pulse at 12 Hz (+/- 180 rad/s)
            t_pulse = drone.icing.deicing_timer
            pulse_val = 160.0 * math.sin(2.0 * math.pi * 12.0 * t_pulse)
            omega_cmds[0] += pulse_val
            omega_cmds[2] += pulse_val
            omega_cmds[1] -= pulse_val
            omega_cmds[3] -= pulse_val

        return omega_cmds, f_thrust, self.fom_efficiency

# ==========================================
# 5. 3D VISUALIZATION & VIEWPORT
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

# ==========================================
# 6. MAIN SIMULATOR & COCKPIT HUD
# ==========================================
def run_icing_defense_simulator():
    pygame.init()
    VIEW_W, VIEW_H = 760, 740
    SIDEBAR_W = 440
    W, H = VIEW_W + SIDEBAR_W, VIEW_H

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Phase 2: Extreme Weather Aerodynamics & In-Flight Ice Defense")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("consolas", 17, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 12, bold=True)

    init_coords = np.array([0.0, 0.0, 2.2], dtype=np.float32)
    drone = IcingAwareDroneAirframe(init_coords)
    controller = IcingDefenseController()

    target_pos = np.array([0.0, 0.0, 2.2], dtype=np.float32)
    cam_yaw, cam_pitch = 0.65, 0.42
    center_viewport = (VIEW_W // 2, VIEW_H // 2 + 50)
    running = True

    # Freezing Cloud Water Droplet Particles
    num_droplets = 60
    fog_droplets = [np.array([random.uniform(-5.0, 5.0), random.uniform(-4.0, 7.0), random.uniform(0.5, 4.5)]) for _ in range(num_droplets)]

    print("\n================ IN-FLIGHT ICE DEFENSE ACTIVE ================")
    print(" [C]       : Toggle Freezing Cloud / Fog Entry (T = -4.5°C, RH = 94%)")
    print(" [D]       : Manually Trigger High-G Centrifugal De-Icing Pulses")
    print(" [R]       : Thaw and Clear All Propeller Ice to 0.0g")
    print(" [W/A/S/D] : Navigate Drone (Target X/Y)")
    print(" [I/K]     : Climb / Descend Altitude")
    print(" [ARROWS]  : Orbit 3D Perspective View Camera")
    print("==============================================================\n")

    while running:
        dt = 0.016

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    drone.icing.in_freezing_cloud = not drone.icing.in_freezing_cloud
                    c_str = "INSIDE FREEZING CLOUD (ACCRETION ACTIVE)" if drone.icing.in_freezing_cloud else "CLEARED CLOUD (DRY AIR)"
                    print(f">>> [ATMOSPHERE] {c_str} <<<")
                elif event.key == pygame.K_d:
                    drone.icing.deicing_active = True
                    print(">>> [MANUAL OVERRIDE] CENTRIFUGAL DE-ICING SHEDDING PULSES FIRED <<<")
                elif event.key == pygame.K_r:
                    drone.icing.ice_mass_g.fill(0.0)
                    controller.fom_efficiency = 0.76
                    print(">>> PROPELLERS THAWED & RESTORED TO ZERO ICE ACCRETION <<<")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: target_pos[1] += 0.04
        if keys[pygame.K_s]: target_pos[1] -= 0.04
        if keys[pygame.K_d]: target_pos[0] += 0.04
        if keys[pygame.K_a]: target_pos[0] -= 0.04
        if keys[pygame.K_i]: target_pos[2] = min(6.0, target_pos[2] + 0.03)
        if keys[pygame.K_k]: target_pos[2] = max(0.4, target_pos[2] - 0.03)

        if keys[pygame.K_LEFT]: cam_yaw -= 0.03
        if keys[pygame.K_RIGHT]: cam_yaw += 0.03

        # Update freezing fog particle drift
        for dp in fog_droplets:
            dp[0] += 0.015
            dp[1] -= 0.008
            if dp[0] > 5.5: dp[0] = -5.5
            if dp[1] < -4.5: dp[1] = 7.5

        # 1. Compute Icing Defense Control & De-Icing Pulses
        v_des = np.zeros(3, dtype=np.float32)
        a_des = np.zeros(3, dtype=np.float32)
        omega_cmds, f_thrust, fom = controller.compute(drone, target_pos, v_des, a_des, dt=dt)

        # 2. Step 6-DOF Physical Drone Dynamics with Camber Loss
        f_thrust_actual, torques, p_watts, total_ice_kg = drone.step(omega_cmds, dt=dt)

        # ==========================================
        # RENDERING: 3D VIEWPORT & FREEZING CLOUD
        # ==========================================
        screen.fill((10, 13, 20))
        pygame.draw.line(screen, (35, 45, 60), (VIEW_W, 0), (VIEW_W, H), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (W, 70), 2)

        cam = (cam_yaw, cam_pitch)

        # 1. 3D Ground Grid
        for gx in range(-5, 6):
            sx1, sy1, _ = project_3d(np.array([gx * 1.5, -4.0, 0.0]), cam, center_viewport)
            sx2, sy2, _ = project_3d(np.array([gx * 1.5,  6.0, 0.0]), cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        for gy in range(-3, 5):
            sx1, sy1, _ = project_3d(np.array([-7.5, gy * 1.5, 0.0]), cam, center_viewport)
            sx2, sy2, _ = project_3d(np.array([ 7.5, gy * 1.5, 0.0]), cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        # 2. Freezing Cloud Droplet Fog (Visible when in cloud)
        if drone.icing.in_freezing_cloud:
            for dp in fog_droplets:
                fsx, fsy, _ = project_3d(dp, cam, center_viewport)
                pygame.draw.circle(screen, (160, 210, 245), (fsx, fsy), 2)

        # 3. Target Waypoint
        tx, ty, _ = project_3d(target_pos, cam, center_viewport)
        pygame.draw.circle(screen, (0, 255, 140), (tx, ty), 6, 1)
        pygame.draw.circle(screen, (255, 255, 255), (tx, ty), 2)

        # 4. Draw Airframe with Progressive Propeller Ice Rims
        dp = drone.p
        dsx, dsy, _ = project_3d(dp, cam, center_viewport)

        arm_d = ARM_LENGTH_M * 1.8
        rotors_body = [
            np.array([ arm_d / 1.414, -arm_d / 1.414, 0.0]), # M1 (FR)
            np.array([ arm_d / 1.414,  arm_d / 1.414, 0.0]), # M2 (FL)
            np.array([-arm_d / 1.414,  arm_d / 1.414, 0.0]), # M3 (RL)
            np.array([-arm_d / 1.414, -arm_d / 1.414, 0.0]), # M4 (RR)
        ]
        rotors_world = [dp + drone.R @ r for r in rotors_body]
        rotor_screens = [project_3d(pt, cam, center_viewport)[:2] for pt in rotors_world]

        # Booms
        pygame.draw.line(screen, (160, 170, 185), rotor_screens[0], rotor_screens[2], 3)
        pygame.draw.line(screen, (160, 170, 185), rotor_screens[1], rotor_screens[3], 3)

        # Fuselage Pod
        pygame.draw.circle(screen, (0, 240, 255), (dsx, dsy), 8)
        pygame.draw.circle(screen, (255, 255, 255), (dsx, dsy), 3)

        # Draw 4 Propellers with Dynamic Ice Build-up & Vibration Shake
        for i, (rx, ry) in enumerate(rotor_screens):
            ice_g = drone.icing.ice_mass_g[i]
            ice_norm = min(1.0, ice_g / MAX_ICE_MASS_PER_PROP_G)

            # Rotor blade blur disc
            pygame.draw.circle(screen, (0, 240, 255), (rx, ry), 15, 1)
            pygame.draw.circle(screen, (240, 240, 240), (rx, ry), 3)

            # Ice crust buildup on propeller ring (White/Ice-Blue frost rim)
            if ice_g > 0.4:
                ice_thickness = int(1 + ice_norm * 4)
                pygame.draw.circle(screen, (215, 240, 255), (rx, ry), 15, ice_thickness)
                # Centrifugal shedding burst sparks
                if drone.icing.deicing_active:
                    spark_x = rx + random.randint(-8, 8)
                    spark_y = ry + random.randint(-8, 8)
                    pygame.draw.circle(screen, (255, 255, 255), (spark_x, spark_y), 2)

        # Ground drop-line
        gsx, gsy, _ = project_3d(np.array([dp[0], dp[1], 0.0]), cam, center_viewport)
        pygame.draw.line(screen, (35, 55, 75), (dsx, dsy), (gsx, gsy), 1)

        # ==========================================
        # SIDEBAR TELEMETRY & AERO-THERMAL AUDIT
        # ==========================================
        sb_x = VIEW_W + 14
        screen.blit(font_large.render("IN-FLIGHT ICE DEFENSE", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Rime Accretion, Camber Degradation & De-Icing", True, (130, 160, 190)), (sb_x, 38))

        # --- PANEL 1: ATMOSPHERIC & FREEZING CLOUD STATE ---
        p1_y = 75
        screen.blit(font_mono.render("FREEZING CLOUD & METEOROLOGY:", True, (240, 240, 240)), (sb_x, p1_y))

        c_stat = "FREEZING CLOUD ACTIVE [C]" if drone.icing.in_freezing_cloud else "CLEAR AIRSPACE (DRY)"
        c_col = (0, 220, 255) if drone.icing.in_freezing_cloud else (0, 255, 140)
        screen.blit(font_mono.render(c_stat, True, c_col), (sb_x + 10, p1_y + 20))
        screen.blit(font_small.render(f"Ambient Temperature: T = {FREEZING_TEMP_C:+.1f} °C (Sub-Zero)", True, (0, 220, 255)), (sb_x + 10, p1_y + 40))
        screen.blit(font_small.render(f"Relative Humidity:   RH = {FREEZING_RH_PCT:.1f}% (Supercooled)", True, (160, 180, 205)), (sb_x + 10, p1_y + 56))
        screen.blit(font_small.render(f"Liquid Water Content:LWC= {LIQUID_WATER_CONTENT:.2f} g/m^3", True, (160, 180, 205)), (sb_x + 10, p1_y + 72))

        # --- PANEL 2: PROPELLER ICE ACCRETION & POWER SURGE ---
        p2_y = p1_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (W - 14, p2_y), 1)
        screen.blit(font_mono.render("PROPELLER ICE ACCRETION (dm/dt):", True, (240, 240, 240)), (sb_x, p2_y + 8))

        tot_ice_g = total_ice_kg * 1000.0
        ice_col = (255, 60, 80) if tot_ice_g > 25.0 else ((255, 180, 40) if tot_ice_g > 6.0 else (0, 255, 140))
        screen.blit(font_small.render(f"Total Ice Accreted:  m_ice = {tot_ice_g:.1f} g (Airframe load)", True, ice_col), (sb_x + 10, p2_y + 28))
        screen.blit(font_small.render(f"Electrical Power:    P_elec= {p_watts:.1f} W (Drag surge)", True, (255, 200, 40)), (sb_x + 10, p2_y + 44))

        # Individual Blade Ice Gauges
        for b_i in range(4):
            by = p2_y + 64 + b_i * 15
            m_g = drone.icing.ice_mass_g[b_i]
            bar_w = int(min(120, (m_g / MAX_ICE_MASS_PER_PROP_G) * 120))
            screen.blit(font_small.render(f"Prop {b_i+1} Ice:", True, (160, 180, 205)), (sb_x + 10, by))
            screen.blit(font_small.render(f"{m_g:.1f}g", True, ice_col), (sb_x + 95, by))

            pygame.draw.rect(screen, (24, 32, 45), (sb_x + 140, by + 2, 120, 7), border_radius=2)
            pygame.draw.rect(screen, (215, 240, 255), (sb_x + 140, by + 2, bar_w, 7), border_radius=2)

        # --- PANEL 3: AERO-PROPULSIVE FIGURE OF MERIT (FOM) ---
        p3_y = p2_y + 135
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (W - 14, p3_y), 1)
        screen.blit(font_mono.render("AERO-PROPULSIVE EFFICIENCY (FOM):", True, (240, 240, 240)), (sb_x, p3_y + 8))

        fom_col = (0, 255, 140) if fom >= ICE_DETECTION_EFF_FLOOR else (255, 60, 80)
        alarm_txt = "NOMINAL (CLEAN BLADES)" if fom >= ICE_DETECTION_EFF_FLOOR else "SEVERE ICING ALARM (STALL RISK!)"
        screen.blit(font_small.render(f"Propulsive Efficiency: eta = {fom:.2f} (Clean: 0.76)", True, fom_col), (sb_x + 10, p3_y + 28))
        screen.blit(font_small.render(f"Observer Status:       {alarm_txt}", True, fom_col), (sb_x + 10, p3_y + 44))
        screen.blit(font_small.render(f"Thrust Margin Loss:    c_T Loss: -{(1.0 - (fom/0.76))*40:.0f}%", True, (255, 180, 40)), (sb_x + 10, p3_y + 60))

        # Efficiency Bar Gauge
        bar_x = sb_x + 10
        bar_y = p3_y + 78
        pygame.draw.rect(screen, (24, 32, 45), (bar_x, bar_y, 320, 8), border_radius=2)
        fill_w = int(320 * min(1.0, max(0.0, (fom - 0.25) / 0.55)))
        pygame.draw.rect(screen, fom_col, (bar_x, bar_y, fill_w, 8), border_radius=2)

        # Threshold marker
        thresh_px = bar_x + int(320 * ((ICE_DETECTION_EFF_FLOOR - 0.25) / 0.55))
        pygame.draw.line(screen, (255, 60, 80), (thresh_px, bar_y - 2), (thresh_px, bar_y + 10), 2)
        screen.blit(font_small.render("Stall Floor", True, (255, 60, 80)), (thresh_px - 32, bar_y + 11))

        # --- PANEL 4: ACTIVE DE-ICING COUNTERMEASURE ---
        p4_y = bar_y + 26
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (W - 14, p4_y), 1)
        screen.blit(font_mono.render("CENTRIFUGAL SHEDDING SYSTEM:", True, (240, 240, 240)), (sb_x, p4_y + 8))

        deice_str = "ACTIVE (12 Hz RPM PULSING)" if drone.icing.deicing_active else "STANDBY (MONITORING)"
        deice_col = (255, 140, 20) if drone.icing.deicing_active else (0, 255, 140)

        screen.blit(font_small.render(f"De-Ice Countermeasure: {deice_str}", True, deice_col), (sb_x + 10, p4_y + 28))
        screen.blit(font_small.render(f"Shearing Mechanism:    Centrifugal Pulse (+/- 1800 RPM)", True, (0, 220, 255)), (sb_x + 10, p4_y + 44))
        screen.blit(font_small.render(f"Piezoelectric Hub:     28 kHz Ultrasonic Delamination", True, (160, 180, 205)), (sb_x + 10, p4_y + 60))
        screen.blit(font_small.render(f"Cumulative Ice Shed:   {drone.icing.shed_event_count} Fragments Cleared", True, (0, 255, 140)), (sb_x + 10, p4_y + 76))

        # --- PANEL 5: OPERATOR KEY COMMANDS ---
        p5_y = p4_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p5_y), (W - 14, p5_y), 1)
        screen.blit(font_mono.render("OPERATOR TEST BENCH:", True, (255, 200, 40)), (sb_x, p5_y + 8))
        screen.blit(font_small.render("• Press [C] to toggle entry into freezing fog cloud", True, (240, 240, 240)), (sb_x + 10, p5_y + 28))
        screen.blit(font_small.render("• Press [D] to force manual high-G de-icing pulses", True, (255, 140, 20)), (sb_x + 10, p5_y + 44))
        screen.blit(font_small.render("• Press [R] to instantly thaw and clear all ice", True, (0, 255, 140)), (sb_x + 10, p5_y + 60))
        screen.blit(font_small.render("• Use [W/A/S/D] and [I/K] to maneuver drone", True, (160, 180, 205)), (sb_x + 10, p5_y + 76))

        # Bottom Mission Status Banner
        bot_y = H - 45
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y - 8), (W - 14, bot_y - 8), 1)
        if drone.icing.deicing_active:
            b_txt, b_col = "DE-ICING: HIGH-G CENTRIFUGAL SHEDDING PULSE ENGAGED", (255, 140, 20)
        elif controller.ice_detected:
            b_txt, b_col = "ICING HAZARD: BLADE CAMBER RUIN DETECTED BY OBSERVER", (255, 60, 80)
        elif drone.icing.in_freezing_cloud:
            b_txt, b_col = "ATMOSPHERIC TRANSIT: FREEZING CLOUD (ACCRETING ICE)", (0, 220, 255)
        else:
            b_txt, b_col = "PROPULSION EFFICIENCY: NOMINAL (CLEAN PROPELLERS)", (0, 255, 140)
        screen.blit(font_mono.render(b_txt, True, b_col), (sb_x + 8, bot_y))

        # Header HUD
        head_txt = font_large.render(f"Ice Defense | Mass: {tot_ice_g:.1f}g | FOM: {fom:.2f} | Power: {p_watts:.0f}W", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("Rime Ice Accretion Model, Aero-Thermal Figure-of-Merit Observer & Centrifugal Shearing Defense", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_icing_defense_simulator()