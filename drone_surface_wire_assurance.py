"""
=============================================================================
PHASE 3: DUAL-BAND DROPZONE SURFACE ASSURANCE & OVERHEAD WIRE DEFENSE
File: drone_surface_wire_assurance.py
=============================================================================
- Dual-Band Surface Assurance (Optical Flow Texture vs Acoustic Pulse Dispersion)
- Liquid Water Specular Reflection vs Solid Concrete vs Porous Vegetation Auditing
- Overhead Thin Wire & Clothesline Detection (Rayleigh Skylight Polarization + LWIR)
- Automatic Winch Lockout & Obstacle-Free Delivery Basket Relocation
- Interactive Hazard Injection: Water Ponds [W], Thin Wires [T], and Driveways [D]
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
# 0. PHYSICAL AIRFRAME & SENSOR CONSTANTS
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

SPEED_OF_SOUND = 343.2               # m/s
DROPZONE_TOLERANCE_M = 1.8           # Safe delivery radius
MAX_ACCEPTABLE_SLOPE_DEG = 14.0

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
# 2. CONTINUOUS 6-DOF AIRFRAME SIMULATOR
# ==========================================
class DroneAirframe6DOF:
    def __init__(self, init_pos):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.R = np.eye(3, dtype=np.float32)
        self.omega = np.zeros(3, dtype=np.float32)
        self.motor_omegas = np.full(4, HOVER_OMEGA, dtype=np.float32)
        self.accel_linear = np.zeros(3, dtype=np.float32)
        self.wind_gust = np.zeros(3, dtype=np.float32)

    def step(self, motor_cmds, dt=0.016):
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

        self.accel_linear = (self.R @ f_body + f_gravity + f_drag + self.wind_gust) / MASS_KG
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
        self.wind_gust *= 0.995
        return f_thrust, torques

# ==========================================
# 3. DUAL-BAND SURFACE & WIRE ASSURANCE AUDITOR
# ==========================================
class SurfaceAndWireAssuranceSuite:
    """
    Simulates:
    1. Ground surface terrain classification: Solid lawn/driveway vs Liquid water pond vs Porous deep bushes.
    2. Multi-echo acoustic dispersion W_echo & Optic texture variance.
    3. Overhead thin guy-wires / clotheslines with Rayleigh linear polarization & thermal contrast.
    """
    def __init__(self):
        # 1. Surface Hazards in World Coordinates
        self.water_pond = {"center": np.array([2.5, 4.0, 0.0]), "radius": 2.2} # Water body
        self.deep_bushes = {"center": np.array([-2.5, 3.5, 0.0]), "radius": 1.8} # Porous vegetation
        self.solid_driveway = {"center": np.array([0.0, -1.0, 0.0]), "size": np.array([4.0, 5.0])}

        # 2. Suspended Thin Overhead Wires (1-3mm clotheslines / utility drop lines)
        self.overhead_wires = [
            {"p1": np.array([-5.0, 2.0, 3.2]), "p2": np.array([5.0, 2.0, 3.2]), "type": "COPPER_UTILITY", "temp_c": 68.0, "dolp": 0.52},
            {"p1": np.array([-4.0, 5.5, 2.5]), "p2": np.array([4.0, 5.5, 2.5]), "type": "NYLON_CLOTHESLINE", "temp_c": 18.0, "dolp": 0.65}
        ]

    def audit_ground_surface(self, pos_xy):
        """
        Audits downward acoustic and optic returns directly beneath the drone:
        Returns: surface_class, echo_spl_db, echo_dispersion_ms, optic_variance, is_safe_for_touchdown
        """
        px, py = pos_xy

        # Check Liquid Water Pond (Swimming Pool / Pond)
        d_water = math.hypot(px - self.water_pond["center"][0], py - self.water_pond["center"][1])
        if d_water < self.water_pond["radius"]:
            # Specular reflection: Acoustic pulse reflects away, optic texture is glassy flat
            echo_spl_db = 42.0 + float(np.random.normal(0, 1.5)) # Low return (loss > 35dB)
            echo_dispersion_ms = 0.25 # Sharp but faint specular bounce
            optic_variance = 0.04 # Glassy water mirror surface
            surface_class = "LIQUID_WATER_HAZARD"
            is_safe = False
            return surface_class, echo_spl_db, echo_dispersion_ms, optic_variance, is_safe

        # Check Porous Vegetation (Bushes / Deep Weeds)
        d_bush = math.hypot(px - self.deep_bushes["center"][0], py - self.deep_bushes["center"][1])
        if d_bush < self.deep_bushes["radius"]:
            # Multi-echo scattering: Pulse spreads across canopy depth, high optic texture variance
            echo_spl_db = 68.0 + float(np.random.normal(0, 2.0))
            echo_dispersion_ms = 1.85 + float(np.random.normal(0, 0.15)) # Multi-echo > 1.2ms!
            optic_variance = 0.85 # Highly textured foliage
            surface_class = "POROUS_VEGETATION_UNSTABLE"
            is_safe = False
            return surface_class, echo_spl_db, echo_dispersion_ms, optic_variance, is_safe

        # Solid Terrain (Flat Grass / Driveway / Tarmac)
        echo_spl_db = 88.0 + float(np.random.normal(0, 1.2)) # High diffuse return
        echo_dispersion_ms = 0.35 + float(np.random.normal(0, 0.05)) # Crisp, narrow reflection
        optic_variance = 0.45 # Moderate texture
        surface_class = "SOLID_TERRAIN_SAFE"
        is_safe = True
        return surface_class, echo_spl_db, echo_dispersion_ms, optic_variance, is_safe

    def scan_overhead_wires(self, drone_p, clearance_radius=1.6):
        """
        Scans for thin wires crossing the vertical cylinder directly beneath the aircraft.
        Returns: wire_detected, wire_distance, wire_type, polar_dolp, thermal_temp_c
        """
        for wire in self.overhead_wires:
            w1 = wire["p1"]
            w2 = wire["p2"]
            wire_z = w1[2]

            # Check if drone is vertically above wire and within horizontal cylinder
            if drone_p[2] > wire_z:
                # Segment distance in XY plane
                y_wire = w1[1]
                if w1[0] <= drone_p[0] <= w2[0] or w2[0] <= drone_p[0] <= w1[0]:
                    d_horiz = abs(drone_p[1] - y_wire)
                    if d_horiz < clearance_radius:
                        vert_dist = drone_p[2] - wire_z
                        return True, vert_dist, wire["type"], wire["dolp"], wire["temp_c"]

        return False, 50.0, "NONE", 0.05, 18.0

# ==========================================
# 4. GEOMETRIC POSITION CONTROLLER WITH RELOCATION
# ==========================================
class SurfaceAssuranceFlightController:
    def __init__(self):
        self.k_p = np.array([6.5, 6.5, 11.5], dtype=np.float32)
        self.k_v = np.array([4.4, 4.4, 7.0], dtype=np.float32)
        self.k_i = np.array([1.2, 1.2, 2.5], dtype=np.float32)
        self.k_R = 1.95
        self.k_w = 0.38
        self.e_p_integral = np.zeros(3, dtype=np.float32)

    def compute(self, drone, p_target, dt=0.016):
        delta_p = p_target - drone.p
        d_horiz = float(np.linalg.norm(delta_p[:2]))

        if d_horiz > 0.4:
            v_cruise = 2.8
            dir_norm = delta_p / max(1e-4, float(np.linalg.norm(delta_p)))
            v_des = dir_norm * v_cruise
            if d_horiz < 2.0:
                v_des[:2] *= (d_horiz / 2.0)
        else:
            v_des = np.zeros(3, dtype=np.float32)

        e_p = drone.p - p_target
        e_v = drone.v - v_des

        self.e_p_integral += e_p * dt
        self.e_p_integral = np.clip(self.e_p_integral, -1.2, 1.2)

        F_d = -self.k_p * e_p - self.k_v * e_v - self.k_i * self.e_p_integral + MASS_KG * (-GRAVITY_G)

        # Enforce tilt clamp <= 25 deg
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
            [-0.018, 0.018, -0.018, 0.018]
        ], dtype=np.float32)
        b_vec = np.array([f_thrust, tau[0], tau[1], tau[2]], dtype=np.float32)
        try:
            T_motors = np.clip(np.linalg.solve(A, b_vec), 0.05, 9.8)
        except np.linalg.LinAlgError:
            T_motors = np.full(4, f_thrust / 4.0, dtype=np.float32)

        omega_cmds = np.sqrt(T_motors / C_T)
        return omega_cmds, f_thrust

# ==========================================
# 5. 3D PROJECTION UTILITY
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
def run_surface_wire_simulator():
    pygame.init()
    VIEW_W, VIEW_H = 760, 740
    SIDEBAR_W = 440
    W, H = VIEW_W + SIDEBAR_W, VIEW_H

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Phase 3: Dual-Band Dropzone Surface Assurance & Overhead Wire Defense")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("consolas", 17, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 12, bold=True)

    init_coords = np.array([-3.0, -2.5, 3.8], dtype=np.float32)
    target_delivery_point = np.array([2.5, 4.0, 0.0], dtype=np.float32) # Starts directly over water pond!

    drone = DroneAirframe6DOF(init_coords)
    auditor = SurfaceAndWireAssuranceSuite()
    controller = SurfaceAssuranceFlightController()

    cam_yaw, cam_pitch = 0.65, 0.42
    center_viewport = (VIEW_W // 2, VIEW_H // 2 + 50)
    running = True

    cable_len = 0.08
    winch_state = "STOWED"
    winch_lockout_active = True
    lockout_reason = "SEARCHING"

    relocation_timer = 0.0
    packages_delivered_count = 0
    delivered_boxes = []

    print("\n================ SURFACE & WIRE ASSURANCE ACTIVE ================")
    print(" [1] : Send Drone to Water Pond Hazard [2.5, 4.0]")
    print(" [2] : Send Drone Directly Over Overhead Utility Wire [0.0, 2.0]")
    print(" [3] : Send Drone Over Deep Bush Vegetation [-2.5, 3.5]")
    print(" [4] : Send Drone to Solid Safe Driveway [0.0, -1.0]")
    print(" [W/A/S/D] : Navigate Drone Manually")
    print(" [ARROWS]  : Orbit 3D Perspective View Camera")
    print("=================================================================\n")

    while running:
        dt = 0.016

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    target_delivery_point = np.array([2.5, 4.0, 0.0], dtype=np.float32)
                    print(">>> [TEST DISPATCH] ROUTED TO RETENTION WATER POND HAZARD <<<")
                elif event.key == pygame.K_2:
                    target_delivery_point = np.array([0.0, 2.0, 0.0], dtype=np.float32)
                    print(">>> [TEST DISPATCH] ROUTED DIRECTLY UNDER THIN OVERHEAD GUY-WIRE <<<")
                elif event.key == pygame.K_3:
                    target_delivery_point = np.array([-2.5, 3.5, 0.0], dtype=np.float32)
                    print(">>> [TEST DISPATCH] ROUTED OVER DEEP BUSH / VEGETATION CANOPY <<<")
                elif event.key == pygame.K_4:
                    target_delivery_point = np.array([0.0, -1.0, 0.0], dtype=np.float32)
                    print(">>> [TEST DISPATCH] ROUTED TO CERTIFIED SOLID DRY DRIVEWAY <<<")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: target_delivery_point[1] += 0.05
        if keys[pygame.K_s]: target_delivery_point[1] -= 0.05
        if keys[pygame.K_d]: target_delivery_point[0] += 0.05
        if keys[pygame.K_a]: target_delivery_point[0] -= 0.05
        if keys[pygame.K_LEFT]: cam_yaw -= 0.03
        if keys[pygame.K_RIGHT]: cam_yaw += 0.03

        # Camera Clamped Auto-Tracking
        target_cx = VIEW_W // 2 - int(drone.p[0] * 10.0)
        target_cy = VIEW_H // 2 + 50 + int(drone.p[1] * 6.0)
        target_cx = int(np.clip(target_cx, VIEW_W * 0.25, VIEW_W * 0.75))
        target_cy = int(np.clip(target_cy, VIEW_H * 0.35, VIEW_H * 0.75))
        center_viewport = (int(center_viewport[0] * 0.94 + target_cx * 0.06),
                           int(center_viewport[1] * 0.94 + target_cy * 0.06))

        # 1. Dual-Band Surface & Overhead Wire Audit
        surf_class, echo_spl, echo_disp, optic_var, is_surface_safe = auditor.audit_ground_surface(drone.p[:2])
        wire_hit, wire_dist, wire_type, wire_dolp, wire_temp = auditor.scan_overhead_wires(drone.p)

        # 2. Automated Safety Interlock & Relocation Engine
        d_horiz = float(np.linalg.norm(drone.p[:2] - target_delivery_point[:2]))
        hover_height = 2.4

        if wire_hit:
            winch_lockout_active = True
            lockout_reason = f"OVERHEAD WIRE ({wire_type})"
            # Proactive lateral clearance nudge away from wire
            target_delivery_point[1] += 0.08
        elif not is_surface_safe:
            winch_lockout_active = True
            lockout_reason = f"HAZARD: {surf_class}"
            # Automated Relocation: Shift dropzone search vector to find safe solid ground!
            relocation_timer += dt
            if relocation_timer > 0.8:
                relocation_timer = 0.0
                target_delivery_point[0] += random.choice([-1.2, 1.2])
                target_delivery_point[1] += random.choice([-1.2, 1.2])
                print(f">>> [AUTO-RELOCATE] SURFACE UNSAFE ({surf_class}): SEARCHING SAFE GROUND AT [{target_delivery_point[0]:.1f}, {target_delivery_point[1]:.1f}] <<<")
        else:
            winch_lockout_active = False
            lockout_reason = "DROPZONE CERTIFIED (SAFE TO LOWER)"

        # 3. Winch State Machine Governed by Surface Assurance
        if d_horiz < 0.45 and not winch_lockout_active:
            if winch_state == "STOWED":
                winch_state = "LOWERING"
        elif winch_lockout_active and winch_state == "LOWERING":
            # Emergency re-hoist if wire or water suddenly enters corridor!
            winch_state = "EMERGENCY_RETRACT"

        if winch_state == "LOWERING":
            cable_len += 1.2 * dt
            if (drone.p[2] - cable_len) <= 0.08:
                winch_state = "TOUCHDOWN"
        elif winch_state == "TOUCHDOWN":
            packages_delivered_count += 1
            delivered_boxes.append(target_delivery_point.copy())
            winch_state = "RETRACTING"
            # Pick new delivery location
            target_delivery_point = np.array([random.uniform(-3.5, 3.5), random.uniform(-2.0, 5.0), 0.0], dtype=np.float32)
        elif winch_state in ["RETRACTING", "EMERGENCY_RETRACT"]:
            cable_len = max(0.08, cable_len - 2.2 * dt)
            if cable_len <= 0.08:
                winch_state = "STOWED"

        # 4. Step Flight Controller & 6-DOF Airframe
        hover_target = np.array([target_delivery_point[0], target_delivery_point[1], hover_height], dtype=np.float32)
        omega_cmds, f_thrust = controller.compute(drone, hover_target, dt=dt)
        drone.step(omega_cmds, dt=dt)

        # ==========================================
        # RENDERING: 3D VIEWPORT & HAZARD MAPPING
        # ==========================================
        screen.fill((10, 13, 20))
        pygame.draw.line(screen, (35, 45, 60), (VIEW_W, 0), (VIEW_W, H), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (W, 70), 2)

        cam = (cam_yaw, cam_pitch)

        # 1. 3D Ground Grid
        for gx in range(-5, 6):
            sx1, sy1, _ = project_3d(np.array([gx * 1.5, -4.0, 0.0]), cam, center_viewport)
            sx2, sy2, _ = project_3d(np.array([gx * 1.5,  7.0, 0.0]), cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        for gy in range(-3, 6):
            sx1, sy1, _ = project_3d(np.array([-7.5, gy * 1.5, 0.0]), cam, center_viewport)
            sx2, sy2, _ = project_3d(np.array([ 7.5, gy * 1.5, 0.0]), cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        # 2. Render Solid Tarmac Driveway (Safe Zone)
        dw_corners = [
            np.array([-2.0, -3.5, 0.0]), np.array([2.0, -3.5, 0.0]),
            np.array([2.0,  1.5, 0.0]), np.array([-2.0,  1.5, 0.0])
        ]
        dw_screens = [project_3d(pt, cam, center_viewport)[:2] for pt in dw_corners]
        pygame.draw.polygon(screen, (38, 44, 42), dw_screens)
        pygame.draw.lines(screen, (0, 255, 140), True, dw_screens, 1)
        screen.blit(font_small.render("CERTIFIED SOLID DRIVEWAY", True, (0, 255, 140)), (dw_screens[0][0] + 10, dw_screens[0][1] - 8))

        # 3. Render Liquid Water Retention Pond (Specular Mirror Hazard)
        water_pts = []
        for a in range(0, 360, 20):
            rad = math.radians(a)
            wx = auditor.water_pond["center"][0] + math.sin(rad) * auditor.water_pond["radius"]
            wy = auditor.water_pond["center"][1] + math.cos(rad) * auditor.water_pond["radius"]
            wsx, wsy, _ = project_3d(np.array([wx, wy, 0.0]), cam, center_viewport)
            water_pts.append((wsx, wsy))
        pygame.draw.polygon(screen, (20, 48, 75), water_pts)
        pygame.draw.lines(screen, (0, 220, 255), True, water_pts, 2)
        screen.blit(font_small.render("LIQUID WATER POND (SPECULAR)", True, (0, 220, 255)), (water_pts[0][0] - 40, water_pts[0][1] - 12))

        # 4. Render Porous Deep Bushes / Vegetation Canopy
        bush_pts = []
        for a in range(0, 360, 20):
            rad = math.radians(a)
            bx = auditor.deep_bushes["center"][0] + math.sin(rad) * auditor.deep_bushes["radius"]
            by = auditor.deep_bushes["center"][1] + math.cos(rad) * auditor.deep_bushes["radius"]
            bsx, bsy, _ = project_3d(np.array([bx, by, 0.0]), cam, center_viewport)
            bush_pts.append((bsx, bsy))
        pygame.draw.polygon(screen, (45, 55, 30), bush_pts)
        pygame.draw.lines(screen, (160, 180, 50), True, bush_pts, 2)
        screen.blit(font_small.render("POROUS BUSH CANOPY", True, (160, 180, 50)), (bush_pts[0][0] - 30, bush_pts[0][1] - 12))

        # 5. Render Suspended Overhead Thin Wires (Clotheslines / Utility drop cables)
        for wire in auditor.overhead_wires:
            w1_s = project_3d(wire["p1"], cam, center_viewport)[:2]
            w2_s = project_3d(wire["p2"], cam, center_viewport)[:2]
            # Polarized reflection glow
            wire_col = (255, 60, 80) if wire_hit and wire_type == wire["type"] else (255, 180, 40)
            pygame.draw.line(screen, wire_col, w1_s, w2_s, 2)
            screen.blit(font_small.render(f"THIN WIRE ({wire['type']} DoLP: {wire['dolp']:.2f})", True, wire_col), (w1_s[0] + 15, w1_s[1] - 8))

        # 6. Delivered Packages
        for p_box in delivered_boxes:
            bx, by, _ = project_3d(p_box, cam, center_viewport)
            pygame.draw.rect(screen, (60, 160, 90), (bx - 5, by - 5, 10, 10))

        # 7. Target Delivery Ring
        tsx, tsy, _ = project_3d(target_delivery_point, cam, center_viewport)
        t_col = (0, 255, 140) if is_surface_safe and not wire_hit else (255, 60, 80)
        pygame.draw.circle(screen, t_col, (tsx, tsy), 14, 2)
        pygame.draw.circle(screen, (255, 255, 255), (tsx, tsy), 3)

        # 8. Active Drone Airframe & Downward Sensing Cones
        dsx, dsy, _ = project_3d(drone.p, cam, center_viewport)
        pygame.draw.circle(screen, (0, 240, 255), (dsx, dsy), 8)
        pygame.draw.circle(screen, (255, 255, 255), (dsx, dsy), 3)

        # Downward Multi-Echo Acoustic Sensing Beam Cone (Yellow/Cyan cone)
        gnd_beneath = np.array([drone.p[0], drone.p[1], 0.0])
        gsx, gsy, _ = project_3d(gnd_beneath, cam, center_viewport)
        cone_col = (0, 255, 140) if is_surface_safe else (255, 60, 80)
        pygame.draw.line(screen, cone_col, (dsx - 6, dsy), (gsx - 12, gsy), 1)
        pygame.draw.line(screen, cone_col, (dsx + 6, dsy), (gsx + 12, gsy), 1)
        pygame.draw.ellipse(screen, cone_col, (gsx - 14, gsy - 6, 28, 12), 1)

        # Winch cable
        payload_z = drone.p[2] - cable_len
        psx, psy, _ = project_3d(np.array([drone.p[0], drone.p[1], payload_z]), cam, center_viewport)
        pygame.draw.line(screen, (240, 240, 240), (dsx, dsy), (psx, psy), 1)
        pygame.draw.rect(screen, (255, 200, 40), (psx - 5, psy - 5, 10, 10))

        # ==========================================
        # DUAL-BAND SURFACE & WIRE TELEMETRY (SIDEBAR)
        # ==========================================
        sb_x = VIEW_W + 14
        screen.blit(font_large.render("SURFACE & WIRE ASSURANCE", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Phase 3: Specular Water, Mud & Wire Polarimetry", True, (130, 160, 190)), (sb_x, 38))

        # --- PANEL 1: DOWNWARD SURFACE AUDIT ---
        p1_y = 75
        screen.blit(font_mono.render("1. DUAL-BAND SURFACE AUDIT:", True, (240, 240, 240)), (sb_x, p1_y))

        surf_col = (0, 255, 140) if is_surface_safe else (255, 60, 80)
        screen.blit(font_mono.render(f"Surface Class: {surf_class}", True, surf_col), (sb_x + 10, p1_y + 20))
        screen.blit(font_small.render(f"Acoustic Return SPL: {echo_spl:.1f} dB (Water loss > 35dB)", True, (0, 220, 255)), (sb_x + 10, p1_y + 40))
        disp_col = (255, 180, 40) if echo_disp > 1.2 else (0, 255, 140)
        screen.blit(font_small.render(f"Pulse Dispersion:    W_echo = {echo_disp:.2f} ms (Bushes > 1.2ms)", True, disp_col), (sb_x + 10, p1_y + 56))
        screen.blit(font_small.render(f"Optic Texture Var:   sigma^2 = {optic_var:.2f} (Mirror water < 0.1)", True, (160, 180, 205)), (sb_x + 10, p1_y + 72))

        # --- PANEL 2: OVERHEAD THIN WIRE SCANNER ---
        p2_y = p1_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (W - 14, p2_y), 1)
        screen.blit(font_mono.render("2. OVERHEAD WIRE & GUY-LINE SCANNER:", True, (240, 240, 240)), (sb_x, p2_y + 8))

        wire_txt = f"HAZARD DETECTED ({wire_type})!" if wire_hit else "CLEAR AIRSPACE COLUMN"
        wire_txt_col = (255, 60, 80) if wire_hit else (0, 255, 140)
        screen.blit(font_mono.render(wire_txt, True, wire_txt_col), (sb_x + 10, p2_y + 26))
        screen.blit(font_small.render(f"Linear Polarization: DoLP = {wire_dolp:.2f} (Wire spike > 0.45)", True, (255, 200, 40)), (sb_x + 10, p2_y + 46))
        screen.blit(font_small.render(f"Radiometric LWIR:    T_wire = {wire_temp:.1f} °C (Joule heating)", True, (0, 220, 255)), (sb_x + 10, p2_y + 62))
        screen.blit(font_small.render(f"Clearance Distance:  d_vert = {wire_dist:.2f} m beneath belly", True, (160, 180, 205)), (sb_x + 10, p2_y + 78))

        # --- PANEL 3: WINCH INTERLOCK & TOUCHDOWN STATUS ---
        p3_y = p2_y + 102
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (W - 14, p3_y), 1)
        screen.blit(font_mono.render("3. WINCH SAFETY INTERLOCK STATE:", True, (240, 240, 240)), (sb_x, p3_y + 8))

        lock_col = (255, 60, 80) if winch_lockout_active else (0, 255, 140)
        screen.blit(font_mono.render(f"Winch Interlock: {'LOCKED (DEPLOY PROHIBITED)' if winch_lockout_active else 'UNLOCKED (CLEAR TO DROP)'}", True, lock_col), (sb_x + 10, p3_y + 26))
        screen.blit(font_small.render(f"Lockout Reason:  {lockout_reason}", True, lock_col), (sb_x + 10, p3_y + 46))
        screen.blit(font_small.render(f"Tether State:    {winch_state} | Length: {cable_len:.2f} m", True, (0, 220, 255)), (sb_x + 10, p3_y + 62))
        screen.blit(font_small.render(f"Delivered Total: {packages_delivered_count} Packages Deposited", True, (0, 255, 140)), (sb_x + 10, p3_y + 78))

        # --- PANEL 4: RELOCATION BENCH ---
        p4_y = p3_y + 102
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (W - 14, p4_y), 1)
        screen.blit(font_mono.render("4. TEST BENCH & HAZARD DISPATCH:", True, (255, 200, 40)), (sb_x, p4_y + 8))
        screen.blit(font_small.render("• Press [1] to send drone to Water Retention Pond", True, (0, 220, 255)), (sb_x + 10, p4_y + 28))
        screen.blit(font_small.render("• Press [2] to hover directly over Thin Utility Wire", True, (255, 180, 40)), (sb_x + 10, p4_y + 44))
        screen.blit(font_small.render("• Press [3] to target Porous Deep Bush Vegetation", True, (160, 180, 50)), (sb_x + 10, p4_y + 60))
        screen.blit(font_small.render("• Press [4] to route to Certified Solid Driveway", True, (0, 255, 140)), (sb_x + 10, p4_y + 76))
        screen.blit(font_small.render("• Use [W/A/S/D] to adjust dropzone coordinates manually", True, (160, 180, 205)), (sb_x + 10, p4_y + 92))

        # Bottom Mission Status Banner
        bot_y = H - 45
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y - 8), (W - 14, bot_y - 8), 1)
        if wire_hit:
            b_txt, b_col = "OVERHEAD WIRE INTERCEPT: DEPLOYMENT LOCKED (RELOCATING)", (255, 60, 80)
        elif not is_surface_safe:
            b_txt, b_col = f"UNSAFE SURFACE ({surf_class}): AUTONOMOUS SEARCH ACTIVE", (255, 140, 20)
        elif winch_state == "LOWERING":
            b_txt, b_col = "WINCH: LOWERING PACKAGE ONTO CERTIFIED SOLID GROUND", (0, 255, 140)
        else:
            b_txt, b_col = "CORRIDOR VERIFIED: SOLID SURFACE & ZERO WIRE OBSTACLES", (0, 220, 255)
        screen.blit(font_mono.render(b_txt, True, b_col), (sb_x + 8, bot_y))

        # Header HUD
        head_txt = font_large.render(f"Surface Assurance | State: {surf_class} | Wire: {'HAZARD' if wire_hit else 'CLEAR'}", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("Dual-Band Specular Water Detection, Porous Canopy Acoustic Dispersion & Polarimetric Wire Scanner", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_surface_wire_simulator()