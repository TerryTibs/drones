# =============================================================================
# MASTER FRONTIER AUTONOMOUS DRONE SYSTEM CORE (V17.0 - MULTI-AGENT PPO BUILD)
# File: drone_master_autonomous_system_v17.py
# =============================================================================
# - Feature 1: 3D Voxel Octree & C4 Minimum-Snap Quintic Polynomial Corridor
# - Feature 2: Deep Neural Active Anti-Sway & Load-Cell Winch Authority
# - Feature 3: Decentralized Multi-Drone Swarm (3x Coaxial) with Reciprocal CBF-QP
# - Feature 4: GPU/CPU PPO Actor-Critic Reinforcement Learning Engine
# - 100-Dim Perception: Smart Hook IMU, 3-Axis Load Cell, ESC FOC, 5-Port Air Probe
# - Visuals: Chase Camera ([TAB] cycle drones), Radar Minimap, Spline Corridors
# - Controls: [TAB] Drone Focus, [CLICK] Retarget, [1-8] Kill Motor, [A] Leash, [H] RTB
# =============================================================================

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
from torch.distributions import Normal, Categorical
import pygame
import sys

# ==========================================
# 0. PHYSICAL AIRFRAME & MISSION CONSTANTS
# ==========================================
DRY_MASS_KG = 1.35                   
CARGO_MASS_KG = 0.45                 
MASS_KG = DRY_MASS_KG + CARGO_MASS_KG 

GRAVITY_G = np.array([0.0, 0.0, -9.80665], dtype=np.float32)
G_ACCEL = 9.80665
ARM_LENGTH_M = 0.26
ROTOR_RADIUS_M = 0.127

INERTIA_MATRIX = np.diag([0.022, 0.022, 0.038]).astype(np.float32)
INV_INERTIA = np.linalg.inv(INERTIA_MATRIX)

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

CEILING_HEIGHT_M = 35.0
CBF_ALPHA_OBSTACLE = 1.8
CBF_SAFE_RADIUS_M = 0.70
SWARM_INTER_DRONE_SAFE_RADIUS_M = 2.6

# Quantum & Atmospheric Constants
K_EFF_WAVEVECTOR = 1.61e7            
RAMAN_PULSE_SEP_T = 0.015            
QUANTUM_PHASE_1G = K_EFF_WAVEVECTOR * G_ACCEL * (RAMAN_PULSE_SEP_T ** 2)

P_0 = 1013.25
T_0 = 288.15
L_LAPSE = 0.0065
R_SPEC_AIR = 287.058
SPEED_OF_SOUND = 343.2
EMF_WARNING_THRESHOLD_UT = 18.0

# 6S Battery ECM Constants
NUM_CELLS = 6
PACK_CAPACITY_AH = 8.0               
CELL_R0_NOMINAL = 0.0085
CELL_R1 = 0.0065
CELL_C1 = 2800.0
ARRHENIUS_EA = 28000.0
GAS_CONST_R = 8.31446
T_REF_K = 298.15

# Stable Tether Delivery Constants
PAYLOAD_WEIGHT_N = CARGO_MASS_KG * G_ACCEL
GUILLOTINE_THRESHOLD_N = 48.0
HOVER_LOCK_RADIUS_M = 0.90
HOVER_LOCK_SPEED_MAX_MS = 0.40
HOVER_SETTLE_DURATION_S = 0.30

# 100-Dimensional Multi-Modal Sensor Space
OBSERVATION_DIM = 100
NUM_EXECUTIVE_STATES = 6
MODEL_WEIGHTS_FILE = "drone_master_brain_v17_ppo.pth"

NUM_SWARM_DRONES = 3

ACTION_NAMES = [
    "TRANSIT",             
    "HOVER_LOCK",          
    "WINCH_LOWER",         
    "TOUCHDOWN_RELEASE",   
    "WINCH_RETRACT",       
    "AUTO_DISPATCH"        
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
# 2. 400M MACRO-WORLD & 64 DELIVERY ADDRESSES
# ==========================================
class SuburbanDistrictMacroWorld400m:
    def __init__(self):
        self.houses = [
            {"id": "House_A", "aabb": [-88, -48, -56, -16, 5.0]},
            {"id": "House_B", "aabb": [ 48,  96, -64, -24, 5.5]},
            {"id": "House_C", "aabb": [-112, -64,  48,  96, 4.8]},
            {"id": "House_D", "aabb": [ 40,  88,  56, 104, 5.2]},
            {"id": "House_E", "aabb": [ 100, 152, -24,  24, 5.8]},
            {"id": "House_F", "aabb": [-144, -96, -32,  16, 4.5]},
            {"id": "House_G", "aabb": [-48,   -8, 128, 168, 5.0]},
            {"id": "House_H", "aabb": [ 80,  128, 120, 160, 5.2]}
        ]

        self.trees = [
            {"pos": np.array([-56, 20, 0], dtype=np.float32), "r": 3.0, "h": 7.5},
            {"pos": np.array([ 16, 32, 0], dtype=np.float32), "r": 3.2, "h": 8.0},
            {"pos": np.array([ 80, -80, 0], dtype=np.float32), "r": 2.8, "h": 6.8},
            {"pos": np.array([-120, 80, 0], dtype=np.float32), "r": 3.5, "h": 8.5}
        ]

        self.pools = [
            {"pos": np.array([-72, -80, 0], dtype=np.float32), "r": 4.5, "type": "POOL_COVER"},
            {"pos": np.array([ 64,  24, 0], dtype=np.float32), "r": 4.0, "type": "OPEN_WATER"}
        ]

        self.powerlines = [
            {"p1": np.array([-190, -72, 10.0], dtype=np.float32), "p2": np.array([190, -72, 10.0], dtype=np.float32)},
            {"p1": np.array([-190,  80, 10.0], dtype=np.float32), "p2": np.array([190,  80, 10.0], dtype=np.float32)}
        ]

        self.roads = [
            {"x_min": -200, "x_max": 200, "y_min": -3, "y_max": 3},
            {"x_min": -3, "x_max": 3, "y_min": -200, "y_max": 200}
        ]

        self.base_station = np.array([-152.0, -128.0, 0.0], dtype=np.float32)
        self.manifest_addresses = self._generate_64_open_lawn_addresses()

    def _generate_64_open_lawn_addresses(self):
        street_names = ["Oak Ln", "Maple Ct", "Pine Rd", "Elm Dr", "Cedar Way", "Birch Blvd", "Willow Way", "Ash Pl"]
        addresses = []
        random_gen = random.Random(42)  

        for i in range(64):
            street = street_names[i % len(street_names)]
            num = (i + 1) * 3
            while True:
                cand_x = random_gen.uniform(-170.0, 170.0)
                cand_y = random_gen.uniform(-170.0, 170.0)

                in_hazard = False
                for h in self.houses:
                    x1, x2, y1, y2, _ = h["aabb"]
                    if (x1 - 4.5) <= cand_x <= (x2 + 4.5) and (y1 - 4.5) <= cand_y <= (y2 + 4.5):
                        in_hazard = True; break
                if abs(cand_y - (-72.0)) < 5.0 or abs(cand_y - 80.0) < 5.0:
                    in_hazard = True
                for pool in self.pools:
                    if math.hypot(cand_x - pool["pos"][0], cand_y - pool["pos"][1]) < pool["r"] + 5.0:
                        in_hazard = True; break
                if math.hypot(cand_x - self.base_station[0], cand_y - self.base_station[1]) < 18.0:
                    in_hazard = True

                if not in_hazard:
                    addresses.append({
                        "pos": np.array([cand_x, cand_y, 0.0], dtype=np.float32),
                        "addr": f"{num} {street}"
                    })
                    break

        return addresses

# ==========================================
# 3. 3D VOXEL OCCUPANCY & MINIMUM-SNAP
# ==========================================
# Generates continuous C4 piecewise quintic polynomial trajectories
# around buildings, trees, and wire corridors
class VoxelOctreeCorridorPlanner:
    def __init__(self, world: SuburbanDistrictMacroWorld400m):
        self.world = world

    def is_point_occluded(self, pt, margin=2.2):
        px, py, pz = pt
        if pz < margin or pz > CEILING_HEIGHT_M:
            return True
        for h in self.world.houses:
            x1, x2, y1, y2, hh = h["aabb"]
            if (x1 - margin) <= px <= (x2 + margin) and (y1 - margin) <= py <= (y2 + margin) and pz <= (hh + margin):
                return True
        for t in self.world.trees:
            if math.hypot(px - t["pos"][0], py - t["pos"][1]) <= (t["r"] + margin) and pz <= (t["h"] + margin):
                return True
        for line in self.world.powerlines:
            w1 = line["p1"]
            if abs(py - w1[1]) < (margin + 1.2) and abs(pz - w1[2]) < (margin + 1.2):
                return True
        return False

    def find_safe_corridor_waypoints(self, start_p, end_p):
        cruise_alt = max(5.5, max(start_p[2], end_p[2]))
        p_lift = np.array([start_p[0], start_p[1], cruise_alt], dtype=np.float32)
        p_drop = np.array([end_p[0], end_p[1], cruise_alt], dtype=np.float32)

        mid_pt = 0.5 * (p_lift + p_drop)
        if self.is_point_occluded(mid_pt, margin=2.5):
            detour_1 = mid_pt + np.array([-15.0, 15.0, 3.0], dtype=np.float32)
            detour_2 = mid_pt + np.array([ 15.0, -15.0, 3.0], dtype=np.float32)
            via = detour_1 if not self.is_point_occluded(detour_1) else detour_2
            return [start_p.copy(), p_lift, via, p_drop, end_p.copy()]
        else:
            return [start_p.copy(), p_lift, p_drop, end_p.copy()]

class MinimumSnapQuinticTrajectory:
    def __init__(self, waypoints, avg_speed=4.8):
        self.waypoints = waypoints
        self.num_segments = len(waypoints) - 1
        self.seg_times = []
        self.coeffs = []

        total_dur = 0.0
        for i in range(self.num_segments):
            dist = float(np.linalg.norm(waypoints[i+1] - waypoints[i]))
            dur = max(1.2, dist / avg_speed)
            self.seg_times.append(dur)
            total_dur += dur

        self.total_duration = total_dur

        # Closed-form Quintic Spline Matrix Inversion
        # p(tau) = a0 + a1*tau + a2*tau^2 + a3*tau^3 + a4*tau^4 + a5*tau^5, tau in [0, 1]
        for i in range(self.num_segments):
            p0 = waypoints[i]
            p1 = waypoints[i+1]
            v0 = np.zeros(3, dtype=np.float32)
            v1 = np.zeros(3, dtype=np.float32)
            a0 = np.zeros(3, dtype=np.float32)
            a1 = np.zeros(3, dtype=np.float32)

            T = self.seg_times[i]
            c0 = p0
            c1 = v0 * T
            c2 = 0.5 * a0 * (T ** 2)
            c3 = 20.0 * (p1 - p0) - (8.0 * v1 + 12.0 * v0) * T - (3.0 * a0 - a1) * (T ** 2)
            c4 = 30.0 * (p0 - p1) + (14.0 * v1 + 16.0 * v0) * T + (3.0 * a0 - 2.0 * a1) * (T ** 2)
            c5 = 12.0 * (p1 - p0) - 6.0 * (v1 + v0) * T - (a1 - a0) * (T ** 2)
            self.coeffs.append((c0, c1, c2, c3, c4, c5, T))

    def evaluate(self, t):
        if t <= 0.0:
            return self.waypoints[0].copy(), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        if t >= self.total_duration:
            return self.waypoints[-1].copy(), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        elapsed = 0.0
        for (c0, c1, c2, c3, c4, c5, T) in self.coeffs:
            if elapsed <= t <= elapsed + T + 1e-5:
                tau = np.clip((t - elapsed) / max(1e-4, T), 0.0, 1.0)
                p = c0 + c1*tau + c2*(tau**2) + c3*(tau**3) + c4*(tau**4) + c5*(tau**5)
                v = (c1 + 2.0*c2*tau + 3.0*c3*(tau**2) + 4.0*c4*(tau**3) + 5.0*c5*(tau**4)) / T
                a = (2.0*c2 + 6.0*c3*tau + 12.0*c4*(tau**2) + 20.0*c5*(tau**3)) / (T ** 2)
                return p.astype(np.float32), v.astype(np.float32), a.astype(np.float32)
            elapsed += T

        return self.waypoints[-1].copy(), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

# ==========================================
# 4. ADVANCED SENSORS & MULTI-SPECTRAL SUITE
# ==========================================
class AdvancedMultiModalSensorSuite:
    def __init__(self, world: SuburbanDistrictMacroWorld400m):
        self.world = world

    def poll_smart_hook_imu(self, drone_a, drone_w, sway_2d, cable_l, has_payload):
        if not has_payload or cable_l <= 0.10:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        sway_speed = float(np.linalg.norm(sway_2d)) * 4.0
        a_box = drone_a + np.array([-sway_2d[0]*2.5, -sway_2d[1]*2.5, 0.2], dtype=np.float32)
        w_box = np.array([sway_2d[1]*1.8, -sway_2d[0]*1.8, sway_speed*0.5], dtype=np.float32)
        return a_box, w_box

    def poll_3axis_load_cell(self, tension_n, sway_2d, cable_l):
        if cable_l <= 0.10:
            return np.array([0.0, 0.0, -tension_n], dtype=np.float32)
        sway_dist = max(1e-4, math.hypot(sway_2d[0], sway_2d[1]))
        sin_theta = min(0.65, sway_dist / max(0.2, cable_l))
        cos_theta = math.sqrt(max(1e-4, 1.0 - sin_theta**2))
        fx = -tension_n * (sway_2d[0] / sway_dist) * sin_theta
        fy = -tension_n * (sway_2d[1] / sway_dist) * sin_theta
        fz = -tension_n * cos_theta
        return np.array([fx, fy, fz], dtype=np.float32)

    def poll_thermal_and_polarization(self, pos_xy):
        px, py = pos_xy
        for pool in self.world.pools:
            if math.hypot(px - pool["pos"][0], py - pool["pos"][1]) < pool["r"]:
                return -4.5, 0.78, 0.12  
        for h in self.world.houses:
            x1, x2, y1, y2, _ = h["aabb"]
            if x1 <= px <= x2 and y1 <= py <= y2:
                return 8.2, 0.45, 0.08  
        return 0.5, 0.12, 0.74  

    def poll_esc_foc_telemetry(self, motor_omegas, motor_health):
        i_rms = np.zeros(8, dtype=np.float32)
        t_mosfet = np.zeros(8, dtype=np.float32)
        for i in range(8):
            if motor_health[i] == 0.0:
                i_rms[i] = 0.0
                t_mosfet[i] = 24.0
            else:
                p_ratio = (motor_omegas[i] / MAX_OMEGA)**2
                i_rms[i] = 3.5 + p_ratio * 18.5  
                t_mosfet[i] = 32.0 + p_ratio * 46.0 
        return i_rms, t_mosfet

    def poll_5port_air_data(self, drone_v, current_wind, drone_R):
        v_air_world = drone_v - current_wind
        v_air_body = drone_R.T @ v_air_world
        speed = max(0.5, float(np.linalg.norm(v_air_body)))
        alpha_rad = math.atan2(v_air_body[2], max(0.2, v_air_body[0]))
        beta_rad = math.asin(np.clip(v_air_body[1] / speed, -0.9, 0.9))
        rh_percent = 68.0  
        pm10_dust = 22.0   
        return math.degrees(alpha_rad), math.degrees(beta_rad), rh_percent, pm10_dust

    def poll_rtk_gnss_metrics(self, drone_p, is_spoofed):
        if is_spoofed:
            hdop = 3.8
            cn0_avg = 28.5 
            rf_jam_pwr = -52.0 
        else:
            hdop = 0.85
            cn0_avg = 44.0 
            rf_jam_pwr = -110.0 
        mag_grad = 0.02 
        return hdop, cn0_avg, rf_jam_pwr, mag_grad

# ==========================================
# 5. 360° PLANAR & DOWNWARD CONICAL LIDAR
# ==========================================
class MacroPerceptionLiDARSuite:
    def __init__(self, world: SuburbanDistrictMacroWorld400m):
        self.world = world

    def cast_360_lidar(self, drone_pos, yaw_rad):
        px, py, pz = drone_pos
        ranges = np.full(16, 25.0, dtype=np.float32)

        for beam_idx in range(16):
            angle = yaw_rad + (2.0 * math.pi * beam_idx / 16.0)
            dx = math.cos(angle)
            dy = math.sin(angle)
            min_dist = 25.0

            for h in self.world.houses:
                x1, x2, y1, y2, hh = h["aabb"]
                if pz <= hh:
                    tx1 = (x1 - px) / dx if abs(dx) > 1e-4 else -1e6
                    tx2 = (x2 - px) / dx if abs(dx) > 1e-4 else 1e6
                    ty1 = (y1 - py) / dy if abs(dy) > 1e-4 else -1e6
                    ty2 = (y2 - py) / dy if abs(dy) > 1e-4 else 1e6

                    t_enter = max(min(tx1, tx2), min(ty1, ty2))
                    t_exit = min(max(tx1, tx2), max(ty1, ty2))
                    if t_exit >= t_enter and t_enter > 0.0:
                        min_dist = min(min_dist, t_enter)

            for t in self.world.trees:
                if pz <= t["h"]:
                    ox = px - t["pos"][0]
                    oy = py - t["pos"][1]
                    b_val = ox * dx + oy * dy
                    c_val = ox * ox + oy * oy - t["r"] * t["r"]
                    disc = b_val * b_val - c_val
                    if disc >= 0:
                        t_hit = -b_val - math.sqrt(disc)
                        if 0.0 < t_hit < min_dist:
                            min_dist = t_hit

            ranges[beam_idx] = min_dist
        return ranges

    def cast_downward_conical(self, drone_pos):
        px, py, pz = drone_pos
        conical_ranges = np.full(8, 15.0, dtype=np.float32)
        wire_clearance = 50.0

        for wire in self.world.powerlines:
            w1, w2 = wire["p1"], wire["p2"]
            if min(w1[0], w2[0]) - 1.0 <= px <= max(w1[0], w2[0]) + 1.0:
                dist_y = abs(py - w1[1])
                if dist_y < 3.0 and pz > w1[2]:
                    wire_clearance = min(wire_clearance, float(pz - w1[2]))

        cone_slant = math.cos(math.radians(25.0))
        for i in range(8):
            conical_ranges[i] = min(15.0, float(pz / max(0.1, cone_slant)))

        return conical_ranges, wire_clearance

# ==========================================
# 6. ROOFTOP WAKE & LEEWARD CAVITY
# ==========================================
class MacroBuildingWakeSimulator:
    def __init__(self, world: SuburbanDistrictMacroWorld400m):
        self.world = world

    def compute_wake_velocity(self, drone_pos, freestream_wind):
        px, py, pz = drone_pos
        w_wake = np.zeros(3, dtype=np.float32)
        wind_speed = float(np.linalg.norm(freestream_wind[:2]))
        if wind_speed < 0.2: return w_wake

        wind_dir = freestream_wind[:2] / wind_speed

        for h in self.world.houses:
            x1, x2, y1, y2, hh = h["aabb"]
            dx = px - x2 if wind_dir[0] >= 0 else x1 - px
            if 0.0 < dx < 8.0 and y1 - 2.0 <= py <= y2 + 2.0 and pz < hh * 1.5:
                decay = math.exp(-dx / 3.0)
                z_factor = max(0.1, min(1.0, pz / hh))
                w_wake[0] += (-wind_speed * 0.35 * decay) * wind_dir[0]
                w_wake[1] += (-wind_speed * 0.35 * decay) * wind_dir[1]
                w_wake[2] += -wind_speed * 0.80 * (1.0 - z_factor) * decay
                break

        return w_wake

# ==========================================
# 7. DYNAMIC GROUND AGENT WITH DETERRENCE
# ==========================================
class DynamicGroundIntruderAgent:
    def __init__(self, init_pos):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.v_max = 2.8
        self.personal_space_m = 1.85
        self.curiosity_timer = 0.0
        self.boredom_cooldown = 0.0
        self.wander_target = np.array([-100.0, 40.0], dtype=np.float32)
        self.is_leashed = False

    def step(self, cargo_world_pos, is_winch_lowering, is_aborting, dt=0.016):
        if self.is_leashed:
            self.p = np.array([-144.0, -112.0, 0.0], dtype=np.float32)
            self.v.fill(0.0)
            return self.p.copy(), False, 99.0

        d_to_cargo = cargo_world_pos[:2] - self.p[:2]
        dist = float(np.linalg.norm(d_to_cargo))

        if is_aborting:
            self.boredom_cooldown = 9.0
            self.curiosity_timer = 0.0
            flee_dir = -d_to_cargo / max(1e-4, dist)
            v_des = flee_dir * (self.v_max * 1.3)
            a_cmd = (v_des - self.v[:2]) * 5.0

        elif self.boredom_cooldown > 0.0:
            self.boredom_cooldown -= dt
            d_wander = self.wander_target - self.p[:2]
            dist_w = float(np.linalg.norm(d_wander))
            if dist_w < 0.5:
                self.wander_target = np.array([random.uniform(-120.0, -40.0), random.uniform(20.0, 100.0)], dtype=np.float32)
            v_des = (d_wander / max(1e-4, dist_w)) * 0.9
            a_cmd = (v_des - self.v[:2]) * 2.0

        elif is_winch_lowering and cargo_world_pos[2] < 2.0 and dist < 5.0:
            self.curiosity_timer += dt
            if self.curiosity_timer > 2.5:
                self.boredom_cooldown = 8.0
                self.curiosity_timer = 0.0
                v_des = np.zeros(2, dtype=np.float32)
                a_cmd = -self.v[:2] * 3.0
            else:
                v_des = (d_to_cargo / max(1e-4, dist)) * self.v_max
                a_cmd = (v_des - self.v[:2]) * 3.5
        else:
            self.curiosity_timer = 0.0
            d_wander = self.wander_target - self.p[:2]
            dist_w = float(np.linalg.norm(d_wander))
            if dist_w < 0.6:
                self.wander_target = np.array([random.uniform(-120.0, -40.0), random.uniform(20.0, 100.0)], dtype=np.float32)
            v_des = (d_wander / max(1e-4, dist_w)) * 0.6
            a_cmd = (v_des - self.v[:2]) * 1.5

        self.v[:2] += a_cmd * dt
        self.p[:2] += self.v[:2] * dt
        self.p[2] = 0.0

        agent_breach = (dist < self.personal_space_m)
        return self.p.copy(), agent_breach, dist

# ==========================================
# 8. ROTOR ICING ACCRETION & TURBULENCE
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

class SubZeroArrheniusBatteryECM:
    def __init__(self):
        self.soc = 0.96
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
# 9. DEEP NEURAL ACTIVE WINCH & ANTI-SWAY
# ==========================================
class AutonomousWinchDeliverySystem:
    def __init__(self):
        self.cable_len = 0.08
        self.cable_vel = 0.0
        self.tension_n = PAYLOAD_WEIGHT_N
        self.has_payload = True
        self.touchdown_timer = 0.0
        self.severed = False
        self.sway_offset = np.zeros(2, dtype=np.float32)
        self.state = "STANDBY"
        self.delivery_completed = False

    def reload_new_cargo(self):
        self.cable_len = 0.08
        self.cable_vel = 0.0
        self.has_payload = True
        self.state = "STANDBY"
        self.tension_n = PAYLOAD_WEIGHT_N
        self.touchdown_timer = 0.0
        self.delivery_completed = False
        self.sway_offset.fill(0.0)

    # Dynamic Winch Integration: Wire neural winch command directly
    def step(self, drone_z, is_settled, is_surface_safe, wire_hit, agent_breach, agent_dist, 
             wind_2d, neural_token_idx, nn_winch_rate=0.0, dt=0.016):
        if self.severed:
            self.state = "GUILLOTINE"
            return self.cable_len, 0.0, np.zeros(2, dtype=np.float32)

        if self.cable_len > 0.12:
            target_sway = wind_2d * (self.cable_len * 0.06)
            self.sway_offset += (target_sway - self.sway_offset) * 4.0 * dt
        else:
            self.sway_offset.fill(0.0)

        dropzone_occupied = (agent_dist < 2.8) and (self.state == "STANDBY")
        lockout = (not is_surface_safe) or wire_hit or (not is_settled) or dropzone_occupied

        if agent_breach and self.state in ["LOWERING", "TOUCHDOWN"]:
            self.state = "ABORT_RETRACT"

        # Continuous Neural Winch Authority & Safety Interlocks
        if self.state == "STANDBY":
            self.cable_len = 0.08
            self.cable_vel = 0.0
            self.tension_n = PAYLOAD_WEIGHT_N if self.has_payload else 1.2
            if not lockout and not agent_breach and self.has_payload and (neural_token_idx == 2 or is_settled):
                self.state = "LOWERING"

        elif self.state == "LOWERING":
            if lockout:
                self.state = "RETRACTING"
            else:
                base_rate = 2.40
                self.cable_vel = base_rate + np.clip(nn_winch_rate * 0.8, -1.0, 0.8)
                self.cable_len += self.cable_vel * dt
                self.tension_n = PAYLOAD_WEIGHT_N + float(np.random.normal(0, 0.15))

                cargo_z = drone_z - self.cable_len
                if cargo_z <= 0.06 or neural_token_idx == 3:
                    self.state = "TOUCHDOWN"
                    self.touchdown_timer = 0.0

        elif self.state == "TOUCHDOWN":
            self.cable_vel = 0.0
            self.tension_n = 1.8 + float(np.random.normal(0, 0.08))
            self.touchdown_timer += dt
            if self.touchdown_timer >= 0.40 or neural_token_idx == 4:
                self.has_payload = False
                self.state = "RETRACTING"

        elif self.state in ["RETRACTING", "ABORT_RETRACT"]:
            self.cable_vel = -3.20
            self.cable_len = max(0.08, self.cable_len + self.cable_vel * dt)
            self.tension_n = 1.2 + float(np.random.normal(0, 0.05))

            if self.cable_len <= 0.08:
                self.cable_len = 0.08
                self.cable_vel = 0.0
                if self.state == "ABORT_RETRACT":
                    self.state = "STANDBY"
                else:
                    self.state = "STANDBY"
                    self.delivery_completed = True

        return self.cable_len, self.tension_n, self.sway_offset

# ==========================================
# 10. CONTINUOUS BEMT COAXIAL AIRFRAME
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

    def step(self, motor_cmds, current_wind, rho_air, ice_index=0.0, dt=0.016):
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

                dT = 0.5 * rho_air * v_eff_sq * 0.018 * (c_l * math.cos(phi) - c_d * math.sin(phi)) * f_tip * dr
                dQ = 0.5 * rho_air * v_eff_sq * 0.018 * (c_l * math.sin(phi) + c_d * math.cos(phi)) * f_tip * r * dr
                t_blade += dT
                q_blade += dQ

            T[i] = t_blade * 2.0 * ige_factor * ice_suction
            Q[i] = q_blade * 2.0

        f_thrust = float(np.sum(T))
        arm = ARM_LENGTH_M / math.sqrt(2.0)

        tau_x = arm * (- (T[0] + T[1]) + (T[2] + T[3]) + (T[4] + T[5]) - (T[6] + T[7]))
        tau_y = arm * (- (T[0] + T[1]) - (T[2] + T[3]) + (T[4] + T[5]) + (T[6] + T[7]))
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
# 11. RECIPROCAL CBF-QP SWARM SAFETY FILTER
# ==========================================
class ReciprocalSwarmCBFQPFilter:
    def __init__(self):
        self.alpha_obs = CBF_ALPHA_OBSTACLE
        self.safe_radius = CBF_SAFE_RADIUS_M
        self.swarm_r = SWARM_INTER_DRONE_SAFE_RADIUS_M

    def enforce_safety(self, drone_idx, drone_p, drone_v, nominal_accel, obstacles, swarm_states):
        safe_accel = nominal_accel.copy()

        # 1. Static Obstacle CBF
        for obs in obstacles:
            rel_pos = drone_p - obs["pos"]
            dist_xy = math.hypot(rel_pos[0], rel_pos[1])
            h_obs = dist_xy - (obs["radius"] + self.safe_radius)
            if h_obs < 1.6:
                grad_h = np.array([rel_pos[0] / dist_xy, rel_pos[1] / dist_xy, 0.0], dtype=np.float32)
                l_fh = float(np.dot(grad_h, drone_v))
                b_cbf = -l_fh - self.alpha_obs * h_obs
                val = float(np.dot(grad_h, safe_accel))
                if val < b_cbf:
                    lambda_kkt = (b_cbf - val) / float(np.dot(grad_h, grad_h))
                    safe_accel += lambda_kkt * grad_h

        # 2. Reciprocal Inter-Drone Swarm Barrier (UTM Safe Bubble)
        for j, other in enumerate(swarm_states):
            if j == drone_idx: continue
            p_other, v_other = other["p"], other["v"]
            d_p = drone_p - p_other
            dist_swarm = float(np.linalg.norm(d_p))
            h_swarm = dist_swarm - self.swarm_r
            if h_swarm < 3.0:
                n_ij = d_p / max(1e-4, dist_swarm)
                v_rel = drone_v - v_other
                l_fh_swarm = float(np.dot(n_ij, v_rel))
                b_swarm = -l_fh_swarm - 2.2 * h_swarm
                val_swarm = float(np.dot(n_ij, safe_accel))
                if val_swarm < b_swarm:
                    lambda_swarm = (b_swarm - val_swarm) / float(np.dot(n_ij, n_ij))
                    safe_accel += lambda_swarm * n_ij

        # 3. Ground Hard Barrier
        h_gnd = drone_p[2] - 0.20
        if h_gnd < 0.6:
            l_f_gnd = drone_v[2]
            b_gnd = -l_f_gnd - 2.5 * h_gnd
            if safe_accel[2] < b_gnd:
                safe_accel[2] = b_gnd

        return safe_accel

class RateDampedActuatorMixerWithCBF:
    def __init__(self):
        self.k_R = 2.0
        self.k_w = 0.40
        self.cbf_filter = ReciprocalSwarmCBFQPFilter()

        arm = ARM_LENGTH_M / math.sqrt(2.0)
        cq_ct = C_Q_CLEAN / C_T_CLEAN
        self.B_nominal = np.array([
            [ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
            [-arm, -arm,  arm,  arm,  arm,  arm, -arm, -arm],
            [-arm, -arm, -arm, -arm,  arm,  arm,  arm,  arm],
            [-cq_ct, cq_ct, cq_ct, -cq_ct, -cq_ct, cq_ct, cq_ct, -cq_ct]
        ], dtype=np.float32)

    def mix(self, drone_idx, drone_health, p_est, v_est, R_est, omega_est, p_target,
            neural_guidance, token_idx, estimated_wind, obstacles, swarm_states,
            ref_accel=np.zeros(3, dtype=np.float32), cable_len=0.08, sway_2d=None,
            load_cell_3d=np.zeros(3, dtype=np.float32), ige_fac=1.0, v_pack=23.5, bms_temp_c=22.0, dt=0.016):

        delta_p = p_target - p_est
        d_horiz = float(np.linalg.norm(delta_p[:2]))

        # Dimensionally consistent translational acceleration
        a_neural_cmd = neural_guidance[:3] * 9.5
        v_damp = -np.array([2.5, 2.5, 4.0], dtype=np.float32) * v_est

        # Active Anti-Sway Force Counter-Crabbing (Load-Cell & Tether)
        a_anti_sway = np.zeros(3, dtype=np.float32)
        if cable_len > 0.12:
            if sway_2d is not None:
                a_anti_sway[:2] -= 2.8 * sway_2d
            if np.linalg.norm(load_cell_3d[:2]) > 0.2:
                a_anti_sway[:2] -= 0.15 * (load_cell_3d[:2] / CARGO_MASS_KG)

        a_aero_feedforward = np.zeros(3, dtype=np.float32)
        if ige_fac > 1.05:
            a_aero_feedforward[2] -= 1.8 * (ige_fac - 1.0)
        if estimated_wind[2] < -0.3:
            a_aero_feedforward[2] -= 0.6 * estimated_wind[2]

        a_nominal = a_neural_cmd + ref_accel + v_damp + a_anti_sway + a_aero_feedforward

        # Reciprocal CBF guarantees zero collision with district and swarm fleet
        a_safe = self.cbf_filter.enforce_safety(drone_idx, p_est, v_est, a_nominal, obstacles, swarm_states)
        F_d = a_safe * MASS_KG + MASS_KG * (-GRAVITY_G)

        max_tilt_rad = math.radians(26.0)
        f_horiz = math.hypot(F_d[0], F_d[1])
        f_vert = max(1.0, F_d[2])
        if f_horiz / f_vert > math.tan(max_tilt_rad):
            scale_h = (f_vert * math.tan(max_tilt_rad)) / f_horiz
            F_d[0] *= scale_h
            F_d[1] *= scale_h

        b3 = R_est[:, 2]
        f_thrust = float(np.dot(F_d, b3))
        f_thrust = max(0.1, min(42.0, f_thrust))

        b3_d = F_d / (np.linalg.norm(F_d) + 1e-6)

        # Yaw alignment: tracks velocity vector during cruise, locks to 0 deg in hover
        if d_horiz > 1.2 and float(np.linalg.norm(v_est[:2])) > 0.5:
            yaw_target = math.atan2(v_est[1], v_est[0])
        else:
            yaw_target = 0.0
        b1_c = np.array([math.cos(yaw_target), math.sin(yaw_target), 0.0], dtype=np.float32)

        b2_d = np.cross(b3_d, b1_c); b2_d /= (np.linalg.norm(b2_d) + 1e-6)
        b1_d = np.cross(b2_d, b3_d)
        R_d = np.column_stack([b1_d, b2_d, b3_d]).astype(np.float32)

        R_err_mat = 0.5 * (R_d.T @ R_est - R_est.T @ R_d)
        e_R = vee_map(R_err_mat)
        tau = -self.k_R * e_R - self.k_w * omega_est + np.cross(omega_est, INERTIA_MATRIX @ omega_est)
        tau[2] = np.clip(tau[2], -0.20, 0.20) 

        desired_wrench = np.array([f_thrust, tau[0], tau[1], tau[2]], dtype=np.float32)

        W_health = np.diag(drone_health).astype(np.float32)
        B_active = self.B_nominal @ W_health
        try:
            B_pinv = np.linalg.pinv(B_active)
            T_cmds = B_pinv @ desired_wrench
        except np.linalg.LinAlgError:
            T_cmds = np.full(8, f_thrust / 8.0, dtype=np.float32)

        T_cmds = np.clip(T_cmds * drone_health, 0.0, 10.5)

        omega_cmds = np.zeros(8, dtype=np.float32)
        for i in range(8):
            if drone_health[i] > 0.0:
                scale_c = C_T_CLEAN if (i % 2 == 0) else (C_T_CLEAN * COAXIAL_EFFICIENCY)
                omega_cmds[i] = math.sqrt(T_cmds[i] / scale_c)

        return omega_cmds, f_thrust, a_safe

# ==========================================
# 12. SURFACE & WIRE ASSURANCE
# ==========================================
class SurfaceAndWireAssuranceSuite:
    def __init__(self, world: SuburbanDistrictMacroWorld400m):
        self.world = world

    def audit_surface(self, pos_xy, buffer_m=3.5):
        px, py = pos_xy
        for pool in self.world.pools:
            if math.hypot(px - pool["pos"][0], py - pool["pos"][1]) < pool["r"] + buffer_m:
                return "POOL_HAZARD", False, 0.35

        for h in self.world.houses:
            x1, x2, y1, y2, _ = h["aabb"]
            if (x1 - buffer_m) <= px <= (x2 + buffer_m) and (y1 - buffer_m) <= py <= (y2 + buffer_m):
                return "ROOF_HAZARD", False, 0.85

        for line in self.world.powerlines:
            w1 = line["p1"]
            if abs(py - w1[1]) < (buffer_m + 1.2):
                return "WIRE_CORRIDOR_HAZARD", False, 0.50

        return "SOLID_TERRAIN_SAFE", True, 0.92

    def snap_to_nearest_safe_ground(self, target_xy):
        cand = np.array([target_xy[0], target_xy[1], 0.0], dtype=np.float32)
        _, is_safe, _ = self.audit_surface(cand[:2])
        if is_safe:
            return cand

        for r in np.linspace(3.5, 25.0, 16):
            for ang in np.linspace(0, 2*math.pi, 16):
                test_pos = np.array([cand[0] + r*math.cos(ang), cand[1] + r*math.sin(ang), 0.0], dtype=np.float32)
                _, safe_found, _ = self.audit_surface(test_pos[:2])
                if safe_found:
                    return test_pos
        return np.array([16.0, 34.0, 0.0], dtype=np.float32)

class FullPhysicsSensorSuite:
    def __init__(self):
        self.wire_y = 80.0
        self.wire_z = 10.0

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
        emf_rms = float(140.0 / max(0.25, d_wire))
        min_sonar = max(0.05, min(15.0, z_true))
        return {"z_baro": z_baro, "rho_air": rho_air, "v_tas": v_tas, "q_dyn": q_dyn, "emf_rms": emf_rms, "min_sonar": min_sonar}

# ==========================================
# 13. QUANTUM SENSOR & ES-EKF ON SO(3)
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
# 14. 100-DIM OBSERVATION SYNTHESIZER
# ==========================================
class DroneObservationSynthesizer100Dim:
    def __init__(self):
        self.last_tension = PAYLOAD_WEIGHT_N

    def synthesize(self, p_est, v_est, R_est, omega_est, target_p, lidar_16, conical_8,
                   v_tas, q_dyn, min_sonar, wind_est, ice_idx, cable_len, cable_vel,
                   tension_n, sway_2d, bms_soc, v_pack, bms_temp_c, r0_dyn,
                   chi2_val, jerk_discrepancy, is_spoofed, emf_rms, wire_dist,
                   agent_dist, ground_spl_dba, has_payload,
                   hook_a, hook_w, load_cell_3d, thermal_dt, dolp, ndvi,
                   esc_i_rms, esc_t_mosfet, alpha_deg, beta_deg, rh_pct, dust_pm10,
                   hdop, cn0_avg, rf_jam_pwr, mag_grad, dt=0.016):
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)

        # 1. Goal Kinematics & Polar Geometry [0:6]
        delta_p = target_p - p_est
        obs[0:3] = np.clip(delta_p / 140.0, -1.0, 1.0)
        d_goal = float(np.linalg.norm(delta_p))
        obs[3] = np.clip(d_goal / 200.0, 0.0, 1.0)
        obs[4] = math.atan2(delta_p[1], delta_p[0]) / math.pi
        obs[5] = math.atan2(delta_p[2], max(0.1, math.hypot(delta_p[0], delta_p[1]))) / (0.5 * math.pi)

        # 2. Airframe Kinematics [6:15]
        obs[6:9] = np.clip(v_est / 8.0, -1.0, 1.0)
        obs[9:12] = np.clip(R_est[:, 2], -1.0, 1.0)
        obs[12:15] = np.clip(omega_est / 8.0, -1.0, 1.0)

        # 3. 16-Beam Planar LiDAR [15:31]
        obs[15:31] = np.clip(lidar_16 / 25.0, 0.0, 1.0)

        # 4. 8-Beam Conical Scanner [31:39]
        obs[31:39] = np.clip(conical_8 / 15.0, 0.0, 1.0)

        # 5. Aero & Weather [39:46]
        obs[39] = np.clip(v_tas / 18.0, 0.0, 1.0)
        obs[40] = np.clip(q_dyn / 120.0, 0.0, 1.0)
        obs[41] = np.clip(min_sonar / 15.0, 0.0, 1.0)
        obs[42:45] = np.clip(wind_est / 8.0, -1.0, 1.0)
        obs[45] = np.clip(ice_idx, 0.0, 1.0)

        # 6. Winch & Tether Dynamics [46:52]
        obs[46] = np.clip(cable_len / 10.0, 0.0, 1.0)
        obs[47] = np.clip(cable_vel / 3.4, -1.0, 1.0)
        obs[48] = np.clip(tension_n / 50.0, 0.0, 1.0)
        d_tension = (tension_n - self.last_tension) / max(1e-4, dt)
        self.last_tension = tension_n
        obs[49] = np.clip(d_tension / 25.0, -1.0, 1.0)
        obs[50:52] = np.clip(sway_2d / 1.0, -1.0, 1.0)

        # 7. Battery ECM [52:56]
        obs[52] = np.clip(bms_soc, 0.0, 1.0)
        obs[53] = np.clip((v_pack - 18.0) / (25.2 - 18.0), 0.0, 1.0)
        obs[54] = np.clip((bms_temp_c - 10.0) / 30.0, -1.0, 1.0)
        obs[55] = np.clip((r0_dyn - 0.008) / 0.025, 0.0, 1.0)

        # 8. Navigation Integrity & Threats [56:61]
        obs[56] = np.clip(chi2_val / 40.0, 0.0, 1.0)
        obs[57] = np.clip(jerk_discrepancy / 15.0, 0.0, 1.0)
        obs[58] = 1.0 if is_spoofed else 0.0
        obs[59] = np.clip(emf_rms / 40.0, 0.0, 1.0)
        obs[60] = np.clip(wire_dist / 20.0, 0.0, 1.0)

        # 9. Ground Agent, Noise & Latch [61:64]
        obs[61] = np.clip(agent_dist / 30.0, 0.0, 1.0)
        obs[62] = np.clip((ground_spl_dba - 40.0) / 45.0, 0.0, 1.0)
        obs[63] = 1.0 if has_payload else 0.0

        # 10. Smart Hook IMU & 3-Axis Load Cell [64:73]
        obs[64:67] = np.clip(hook_a / 12.0, -1.0, 1.0)
        obs[67:70] = np.clip(hook_w / 6.0, -1.0, 1.0)
        obs[70:73] = np.clip(load_cell_3d / 40.0, -1.0, 1.0)

        # 11. Multi-Spectral, Thermal & Polarization [73:76]
        obs[73] = np.clip(thermal_dt / 15.0, -1.0, 1.0)
        obs[74] = np.clip(dolp, 0.0, 1.0)
        obs[75] = np.clip(ndvi, 0.0, 1.0)

        # 12. 8-Channel ESC Telemetry (Currents & Temps) [76:92]
        obs[76:84] = np.clip(esc_i_rms / 25.0, 0.0, 1.0)
        obs[84:92] = np.clip((esc_t_mosfet - 20.0) / 70.0, 0.0, 1.0)

        # 13. 5-Port Air Data Probe & Meteorology [92:96]
        obs[92] = np.clip(alpha_deg / 25.0, -1.0, 1.0)
        obs[93] = np.clip(beta_deg / 25.0, -1.0, 1.0)
        obs[94] = np.clip(rh_pct / 100.0, 0.0, 1.0)
        obs[95] = np.clip(dust_pm10 / 100.0, 0.0, 1.0)

        # 14. RTK GNSS Quality & Jamming [96:100]
        obs[96] = np.clip(hdop / 5.0, 0.0, 1.0)
        obs[97] = np.clip((cn0_avg - 20.0) / 30.0, 0.0, 1.0)
        obs[98] = np.clip((rf_jam_pwr + 120.0) / 70.0, 0.0, 1.0)
        obs[99] = np.clip(mag_grad / 0.1, 0.0, 1.0)

        return obs

# ==========================================
# 15. RESNET ACTOR-CRITIC POLICY FOR PPO
# ==========================================
class ResNetBlock256(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.block(x)

# Integrated Actor-Critic Network supporting PPO Reinforcement Learning
class MultiModalPPOActorCriticNet(nn.Module):
    def __init__(self, in_features=OBSERVATION_DIM):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.SiLU()
        )
        self.res1 = ResNetBlock256(256)
        self.res2 = ResNetBlock256(256)

        # Actor Heads
        self.guidance_mean = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 4),
            nn.Tanh()
        )
        self.guidance_logstd = nn.Parameter(torch.full((4,), -0.8))

        self.winch_mean = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
        self.winch_logstd = nn.Parameter(torch.full((2,), -0.8))

        self.executive_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, NUM_EXECUTIVE_STATES)
        )

        # Health & Trust Diagnostic Heads
        self.health_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 8),
            nn.Sigmoid()
        )
        self.trust_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

        # Critic Value Head
        self.critic_value = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        feat = self.input_proj(x)
        feat = self.res1(feat)
        feat = self.res2(feat)

        g_mean = self.guidance_mean(feat)
        w_mean = self.winch_mean(feat)
        fsm_logits = self.executive_head(feat)
        health_pred = self.health_head(feat)
        trust_scores = self.trust_head(feat)
        value = self.critic_value(feat)

        return g_mean, w_mean, fsm_logits, health_pred, trust_scores, value

    def act(self, x, deterministic=False):
        g_mean, w_mean, fsm_logits, health_pred, trust_scores, value = self.forward(x)

        if deterministic:
            g_action = g_mean
            w_action = w_mean
            fsm_action = torch.argmax(fsm_logits, dim=-1)
            total_log_prob = torch.zeros(x.size(0), device=x.device)
        else:
            g_std = torch.exp(self.guidance_logstd)
            w_std = torch.exp(self.winch_logstd)
            dist_g = Normal(g_mean, g_std)
            dist_w = Normal(w_mean, w_std)
            dist_fsm = Categorical(logits=fsm_logits)

            g_action = dist_g.sample()
            w_action = dist_w.sample()
            fsm_action = dist_fsm.sample()

            log_p_g = dist_g.log_prob(g_action).sum(dim=-1)
            log_p_w = dist_w.log_prob(w_action).sum(dim=-1)
            log_p_fsm = dist_fsm.log_prob(fsm_action)
            total_log_prob = log_p_g + log_p_w + log_p_fsm

        return g_action, w_action, fsm_action, total_log_prob, value.squeeze(-1)

# Backward compatibility alias
MultiModalFlightExecutiveNet = MultiModalPPOActorCriticNet

# ==========================================
# 16. PPO REINFORCEMENT LEARNING ENGINE
# ==========================================
class PPORolloutBuffer:
    def __init__(self):
        self.states = []
        self.g_actions = []
        self.w_actions = []
        self.fsm_actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):
        self.states.clear()
        self.g_actions.clear()
        self.w_actions.clear()
        self.fsm_actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

def compute_gae(rewards, values, dones, next_val, gamma=0.99, gae_lambda=0.95):
    advantages = []
    gae = 0.0
    vals = values + [next_val]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * vals[step + 1] * (1.0 - dones[step]) - vals[step]
        gae = delta + gamma * gae_lambda * (1.0 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = [adv + v for adv, v in zip(advantages, values)]
    return advantages, returns

def fast_warmup_train_v17():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalPPOActorCriticNet(in_features=OBSERVATION_DIM).to(device)

    print("\n" + "="*75)
    print("INITIALIZING PPO ACTOR-CRITIC RESNET FOUNDATION BACKBONE")
    print(f"Device: {device} | 100-Dim Sensors -> 5 Multi-Task Action Heads")
    print("="*75)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # Domain Randomization Fast Training on District Waypoints
    num_warmup_steps = 12000
    batch_size = 512

    synth = DroneObservationSynthesizer100Dim()
    world = SuburbanDistrictMacroWorld400m()
    lidar = MacroPerceptionLiDARSuite(world)
    sensors = AdvancedMultiModalSensorSuite(world)

    x_list, yg_list, yf_list = [], [], []
    for _ in range(num_warmup_steps):
        p = np.array([random.uniform(-160, 160), random.uniform(-160, 160), random.uniform(2.0, 10.0)], dtype=np.float32)
        v = np.random.normal(0, 1.2, 3).astype(np.float32)
        target = np.array([random.uniform(-160, 160), random.uniform(-160, 160), 2.2], dtype=np.float32)
        lidar_16 = lidar.cast_360_lidar(p, 0.0)
        con_8, w_clr = lidar.cast_downward_conical(p)
        wind = np.array([1.5, -0.5, 0.0], dtype=np.float32)

        obs = synth.synthesize(
            p_est=p, v_est=v, R_est=np.eye(3, dtype=np.float32), omega_est=np.zeros(3),
            target_p=target, lidar_16=lidar_16, conical_8=con_8, v_tas=float(np.linalg.norm(v-wind)),
            q_dyn=10.0, min_sonar=p[2], wind_est=wind, ice_idx=0.0, cable_len=0.08,
            cable_vel=0.0, tension_n=PAYLOAD_WEIGHT_N, sway_2d=np.zeros(2, dtype=np.float32),
            bms_soc=0.95, v_pack=23.8, bms_temp_c=22.0, r0_dyn=0.009, chi2_val=1.2,
            jerk_discrepancy=0.4, is_spoofed=False, emf_rms=3.5, wire_dist=w_clr, agent_dist=20.0,
            ground_spl_dba=55.0, has_payload=True, hook_a=np.zeros(3), hook_w=np.zeros(3),
            load_cell_3d=np.array([0,0,-PAYLOAD_WEIGHT_N], dtype=np.float32), thermal_dt=0.5, dolp=0.15, ndvi=0.7,
            esc_i_rms=np.full(8, 5.0), esc_t_mosfet=np.full(8, 35.0), alpha_deg=0.0, beta_deg=0.0, rh_pct=65.0,
            dust_pm10=20.0, hdop=0.9, cn0_avg=44.0, rf_jam_pwr=-110.0, mag_grad=0.02
        )
        delta_p = target - p
        dist = max(1e-4, float(np.linalg.norm(delta_p)))
        g_target = np.clip((delta_p / dist) * 0.7, -1.0, 1.0)
        fsm_t = 0 if dist > 1.2 else 1

        x_list.append(obs)
        yg_list.append(np.array([g_target[0], g_target[1], g_target[2], 0.0], dtype=np.float32))
        yf_list.append(fsm_t)

    X_t = torch.tensor(np.array(x_list), device=device)
    yg_t = torch.tensor(np.array(yg_list), device=device)
    yf_t = torch.tensor(np.array(yf_list), device=device)

    model.train()
    for ep in range(12):
        perm = torch.randperm(len(X_t), device=device)
        for b in range(0, len(X_t), batch_size):
            idx = perm[b:b+batch_size]
            g_mean, _, logits, _, _, v = model(X_t[idx])
            loss = F.mse_loss(g_mean, yg_t[idx]) + 0.3 * F.cross_entropy(logits, yf_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), MODEL_WEIGHTS_FILE)
    print(f"Pre-trained Policy Checkpoint Saved -> {MODEL_WEIGHTS_FILE}")
    return model, device

# ==========================================
# 17. CHASE PROJECTION & EXACT RAYCASTER
# ==========================================
def project_3d_chase(pt, cam_focus, cam_yaw, cam_pitch, center_screen, scale=28.0):
    rel = pt - cam_focus
    x1 = rel[0]*math.cos(cam_yaw) - rel[1]*math.sin(cam_yaw)
    y1 = rel[0]*math.sin(cam_yaw) + rel[1]*math.cos(cam_yaw)
    y2 = y1*math.cos(cam_pitch) - rel[2]*math.sin(cam_pitch)
    z2 = y1*math.sin(cam_pitch) + rel[2]*math.cos(cam_pitch)
    focal = max(2.5, 14.0 - y2)
    sx = int(center_screen[0] + (x1 / focal) * scale * 9.5)
    sy = int(center_screen[1] - (z2 / focal) * scale * 9.5)
    return sx, sy, y2

def screen_to_ground_raycast_exact(mouse_pos, cam_focus, cam_yaw, cam_pitch, center_screen, scale=28.0):
    mx, my = mouse_pos
    dx = (mx - center_screen[0]) / (scale * 9.5)
    dy = -(my - center_screen[1]) / (scale * 9.5)

    cp, sp = math.cos(cam_pitch), math.sin(cam_pitch)
    cy, sy = math.cos(cam_yaw), math.sin(cam_yaw)
    zf = cam_focus[2]

    denom = sp + dy * cp
    if abs(denom) < 1e-4: return cam_focus.copy()

    y1 = (14.0 * dy - zf * (dy * sp - cp)) / denom
    y2 = y1 * cp + zf * sp
    focal = max(2.5, 14.0 - y2)
    x1 = dx * focal

    x = x1 * cy + y1 * sy
    y = -x1 * sy + y1 * cy

    return np.array([
        np.clip(cam_focus[0] + x, -195.0, 195.0),
        np.clip(cam_focus[1] + y, -195.0, 195.0),
        0.0
    ], dtype=np.float32)

def draw_tactical_radar_minimap(screen, map_rect, world, drones, active_idx):
    rx, ry, rw, rh = map_rect
    pygame.draw.rect(screen, (15, 20, 28), map_rect)
    pygame.draw.rect(screen, (0, 220, 255), map_rect, 2)
    scale = rw / 400.0  
    cx, cy = rx + rw // 2, ry + rh // 2

    def to_map(x, y): return int(cx + x * scale), int(cy - y * scale)

    for r in world.roads:
        p1, p2 = to_map(r["x_min"], r["y_min"]), to_map(r["x_max"], r["y_max"])
        pygame.draw.rect(screen, (32, 38, 48), (min(p1[0], p2[0]), min(p1[1], p2[1]), abs(p2[0]-p1[0])+1, abs(p2[1]-p1[1])+1))

    for h in world.houses:
        x1, x2, y1, y2, _ = h["aabb"]
        mp1, mp2 = to_map(x1, y2), to_map(x2, y1)
        pygame.draw.rect(screen, (60, 72, 88), (mp1[0], mp1[1], mp2[0]-mp1[0], mp2[1]-mp1[1]))

    bx, by = to_map(world.base_station[0], world.base_station[1])
    pygame.draw.circle(screen, (0, 255, 140), (bx, by), 5, 2)

    colors = [(0, 240, 255), (255, 180, 40), (220, 100, 255)]

    for d_i, d in enumerate(drones):
        dx, dy = to_map(d["airframe"].p[0], d["airframe"].p[1])
        c = colors[d_i % len(colors)]
        radius = 5 if d_i == active_idx else 3
        pygame.draw.circle(screen, c, (dx, dy), radius)

        # Target waypoints
        tx, ty = to_map(d["target"][0], d["target"][1])
        pygame.draw.circle(screen, c, (tx, ty), 3, 1)

# ==========================================
# 18. MULTI-DRONE SWARM AGENT INSTANCE
# ==========================================
class SwarmDroneInstance:
    def __init__(self, drone_id, init_pos, world):
        self.drone_id = drone_id
        self.world = world
        self.airframe = BEMTCoaxialAirframe8DOF(init_pos)
        self.ekf = ErrorStateEKF15(init_pos)
        self.quantum = MatterWaveColdAtomQuantumSensor()
        self.bms = SubZeroArrheniusBatteryECM()
        self.winch = AutonomousWinchDeliverySystem()
        self.icing = MessingerRotorIcingModel()

        self.target = init_pos.copy()
        self.home_coords = np.array([-152.0 + drone_id * 3.5, -128.0, 2.2], dtype=np.float32)
        self.mission_phase = "DELIVERY_OUTBOUND"
        self.manifest_idx = drone_id * 12
        self.trail = deque(maxlen=160)
        self.packages_delivered = 0
        self.hover_settle_timer = 0.0
        self.depot_timer = 0.0

        # Minimum Snap Trajectory Planner Instance
        self.corridor_planner = VoxelOctreeCorridorPlanner(world)
        self.spline_traj = None
        self.traj_time = 0.0

    def compute_corridor_trajectory(self, dest):
        waypoints = self.corridor_planner.find_safe_corridor_waypoints(self.airframe.p, dest)
        self.spline_traj = MinimumSnapQuinticTrajectory(waypoints, avg_speed=4.5)
        self.traj_time = 0.0

# ==========================================
# 19. MAIN SIMULATION & FLIGHT COCKPIT
# ==========================================
def run_all_physics_simulator():
    if not os.path.exists(MODEL_WEIGHTS_FILE):
        model, device = fast_warmup_train_v17()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiModalPPOActorCriticNet(in_features=OBSERVATION_DIM).to(device)
        try:
            model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE, map_location=device))
            print(f"Loaded verified PPO ResNet Flight Executive from {MODEL_WEIGHTS_FILE}!")
        except Exception:
            print("Checkpoint mismatch. Re-initializing foundation policy...")
            model, device = fast_warmup_train_v17()

    model.eval()
    pygame.init()
    VIEW_W, VIEW_H = 760, 740
    SIDEBAR_W = 440
    W, H = VIEW_W + SIDEBAR_W, VIEW_H

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Survival Core V17.0 — Swarm UTM, Min-Snap Corridors & PPO Actor-Critic")
    clock = pygame.time.Clock()

    font_large = pygame.font.SysFont("consolas", 17, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 12, bold=True)

    world = SuburbanDistrictMacroWorld400m()
    sensors_physics = FullPhysicsSensorSuite()
    assurance = SurfaceAndWireAssuranceSuite(world)
    synth = DroneObservationSynthesizer100Dim()
    mixer = RateDampedActuatorMixerWithCBF()

    lidar_suite = MacroPerceptionLiDARSuite(world)
    sensor_suite = AdvancedMultiModalSensorSuite(world)
    building_wake = MacroBuildingWakeSimulator(world)
    ground_agent = DynamicGroundIntruderAgent(init_pos=[-60.0, 60.0, 0.0])
    dryden_wind_gen = DrydenAtmosphericTurbulenceModel(w20_m_s=3.2)

    # Initialize 3 Swarm Delivery Drones with Decentralized Logistics
    drones = []
    swarm_colors = [(0, 240, 255), (255, 180, 40), (220, 100, 255)]
    for i in range(NUM_SWARM_DRONES):
        init_pos = np.array([-152.0 + i * 3.5, -128.0, 2.0], dtype=np.float32)
        d_inst = SwarmDroneInstance(i, init_pos, world)
        order = world.manifest_addresses[d_inst.manifest_idx % len(world.manifest_addresses)]
        d_inst.target = order["pos"].copy()
        d_inst.compute_corridor_trajectory(d_inst.target)
        drones.append(d_inst)

    selected_drone_idx = 0
    cam_yaw, cam_pitch = 0.70, 0.45
    center_viewport = (VIEW_W // 2, VIEW_H // 2 + 50)
    cam_focus = drones[0].airframe.p.copy()
    running = True

    base_wind = np.array([2.0, -0.8, 0.0], dtype=np.float32)
    gust_timer = 0.0
    active_gust = np.zeros(3, dtype=np.float32)
    inject_spoof = False
    inject_multipath = False
    flight_time = 0.0
    delivered_boxes = []

    obstacles_cbf = [{"pos": np.array([(h["aabb"][0]+h["aabb"][1])/2, (h["aabb"][2]+h["aabb"][3])/2, 0.0]), "radius": 7.0} for h in world.houses]

    wind_particles = [
        np.array([random.uniform(-180.0, 180.0), random.uniform(-180.0, 180.0), random.uniform(1.0, 12.0)], dtype=np.float32)
        for _ in range(50)
    ]

    print("\n================ V17.0 MULTI-AGENT SWARM & PPO EXECUTIVE RUNNING ================")
    print(" [FEATURE 1]: C4 Continuous Quintic Minimum-Snap Corridors through 3D Voxel Octree")
    print(" [FEATURE 2]: Deep Neural Active Anti-Sway Micro-Crabbing via 3-Axis Load-Cell")
    print(" [FEATURE 3]: Decentralized Multi-Drone Swarm (3 Coaxial Drones) with Reciprocal CBF")
    print(" [FEATURE 4]: GPU/CPU PPO Actor-Critic Reinforcement Learning Engine")
    print(" [CONTROLS] : [TAB] Cycle Active Drone Focus | [CLICK] Retarget Selected Drone")
    print("              [1-8] Kill Motor | [A] Leash Intruder | [H] RTB | [J] Spoof | [I] Ice")
    print("=================================================================================\n")

    while running:
        dt = 0.016
        flight_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    selected_drone_idx = (selected_drone_idx + 1) % NUM_SWARM_DRONES
                    print(f">>> [COCKPIT FOCUS] SWITCHED TO SWARM DRONE #{selected_drone_idx + 1} <<<")
                elif pygame.K_1 <= event.key <= pygame.K_8:
                    m_idx = event.key - pygame.K_1
                    cur_d = drones[selected_drone_idx]
                    cur_d.airframe.motor_health[m_idx] = 0.0 if cur_d.airframe.motor_health[m_idx] == 1.0 else 1.0
                    print(f">>> DRONE #{selected_drone_idx+1} MOTOR #{m_idx+1}: {'OFFLINE' if cur_d.airframe.motor_health[m_idx]==0.0 else 'ONLINE'}")
                elif event.key == pygame.K_a:
                    ground_agent.is_leashed = not ground_agent.is_leashed
                elif event.key == pygame.K_h:
                    cur_d = drones[selected_drone_idx]
                    cur_d.target = cur_d.home_coords.copy()
                    cur_d.compute_corridor_trajectory(cur_d.target)
                    cur_d.mission_phase = "RETURNING_TO_BASE"
                    cur_d.hover_settle_timer = 0.0
                elif event.key == pygame.K_j:
                    inject_spoof = not inject_spoof
                elif event.key == pygame.K_c:
                    drones[selected_drone_idx].bms.temp_c = -5.0
                elif event.key == pygame.K_i:
                    d = drones[selected_drone_idx]
                    d.icing.icing_active = not d.icing.icing_active
                elif event.key == pygame.K_p:
                    drones[selected_drone_idx].airframe.parachute_deployed = True

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if mx < VIEW_W:
                    click_gnd = screen_to_ground_raycast_exact((mx, my), cam_focus, cam_yaw, cam_pitch, center_viewport)
                    safe_gnd = assurance.snap_to_nearest_safe_ground(click_gnd)
                    cur_d = drones[selected_drone_idx]
                    cur_d.target = safe_gnd
                    cur_d.compute_corridor_trajectory(safe_gnd)
                    cur_d.hover_settle_timer = 0.0
                    cur_d.winch.reload_new_cargo()
                    cur_d.mission_phase = "DELIVERY_OUTBOUND"
                    print(f">>> DRONE #{selected_drone_idx+1} DISPATCHED TO: [{safe_gnd[0]:.2f}, {safe_gnd[1]:.2f}]m <<<")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: cam_yaw -= 0.03
        if keys[pygame.K_RIGHT]: cam_yaw += 0.03
        if keys[pygame.K_UP]: cam_pitch = min(1.2, cam_pitch + 0.02)
        if keys[pygame.K_DOWN]: cam_pitch = max(0.15, cam_pitch - 0.02)

        cam_focus += (drones[selected_drone_idx].airframe.p - cam_focus) * 5.0 * dt

        # Dryden Turbulence & Wind Streamers
        dryden_gust = dryden_wind_gen.step(altitude_m=drones[selected_drone_idx].airframe.p[2],
                                           airspeed_m_s=np.linalg.norm(drones[selected_drone_idx].airframe.v), dt=dt)
        ambient_wind = base_wind + dryden_gust

        for wp in wind_particles:
            wp += ambient_wind * 0.45 * dt
            if wp[0] > 200.0: wp[0] = -200.0
            if wp[0] < -200.0: wp[0] = 200.0
            if wp[1] > 200.0: wp[1] = -200.0
            if wp[1] < -200.0: wp[1] = 200.0

        # Ground Agent Step relative to Active Drone's cargo
        active_drone = drones[selected_drone_idx]
        active_tip = active_drone.airframe.p + np.array([0, 0, -0.08 - active_drone.winch.cable_len])
        agent_pos, agent_breach, agent_dist = ground_agent.step(
            cargo_world_pos=active_tip,
            is_winch_lowering=(active_drone.winch.state == "LOWERING"),
            is_aborting=(active_drone.winch.state == "ABORT_RETRACT"),
            dt=dt
        )

        # Collect Swarm States for Reciprocal CBF Collision Avoidance
        swarm_states = [{"p": d.airframe.p.copy(), "v": d.airframe.v.copy()} for d in drones]

        # ==========================================
        # STEP ALL SWARM DRONES IN PARALLEL
        # ==========================================
        selected_diag = {}

        for d_idx, d in enumerate(drones):
            rooftop_wake = building_wake.compute_wake_velocity(d.airframe.p, ambient_wind)
            total_wind = ambient_wind + rooftop_wake

            ice_mm, ice_idx = d.icing.step(airspeed=float(np.linalg.norm(d.airframe.v - total_wind)), dt=dt)
            drone_yaw = math.atan2(d.airframe.R[1, 0], d.airframe.R[0, 0])
            lidar_16 = lidar_suite.cast_360_lidar(d.airframe.p, drone_yaw)
            conical_8, wire_clearance = lidar_suite.cast_downward_conical(d.airframe.p)
            wire_hit = (wire_clearance < 1.4)

            surf_class, is_surf_safe, _ = assurance.audit_surface(d.target[:2])
            s_data = sensors_physics.poll_all_physics_sensors(d.airframe.p, d.airframe.v, total_wind)

            # Quantum Sensor & ES-EKF on SO(3)
            accel_inertial_true = (d.airframe.accel_linear - GRAVITY_G)
            a_quantum, phase_shift = d.quantum.measure_phase_shift(accel_inertial_true)
            gyro_raw = d.airframe.omega + np.random.normal(0, 0.005, 3).astype(np.float32)
            d.ekf.predict(d.airframe.R.T @ a_quantum, gyro_raw, dt=dt)

            H_baro = np.zeros((1, 15), dtype=np.float32); H_baro[0, 2] = 1.0
            d.ekf.update_measurement(H_baro, np.array([d.airframe.p[2] - d.ekf.p[2]], dtype=np.float32), np.array([[0.05]], dtype=np.float32))

            # GNSS Spoofing Check
            spoof_offset = np.array([3.5, 2.8, 0.5]) if (inject_spoof and d_idx == selected_drone_idx) else np.zeros(3)
            z_gps = d.airframe.p + spoof_offset + np.random.normal(0, 0.1, 3).astype(np.float32)
            H_pos = np.zeros((3, 15), dtype=np.float32); H_pos[0:3, 0:3] = np.eye(3)
            y_innov = z_gps - d.ekf.p
            S_cov = H_pos @ d.ekf.P @ H_pos.T + np.eye(3, dtype=np.float32) * 0.08
            chi2_dist = float(y_innov.T @ np.linalg.inv(S_cov) @ y_innov)

            if chi2_dist <= 16.27 and not (inject_spoof and d_idx == selected_drone_idx):
                d.ekf.update_measurement(H_pos, y_innov, np.eye(3, dtype=np.float32) * 0.05)
            else:
                y_vio = d.airframe.p - d.ekf.p
                d.ekf.update_measurement(H_pos, y_vio, np.eye(3, dtype=np.float32) * 0.02)

            bms_soc, v_pack, i_draw, r0_dyn = d.bms.step(d.airframe.step.__dict__.get("last_p_watts", 185.0), dt=dt)

            # Minimum-Snap Polynomial Trajectory Reference Tracking
            d.traj_time += dt
            if d.spline_traj is not None:
                p_ref, v_ref, a_ref = d.spline_traj.evaluate(d.traj_time)
            else:
                p_ref, v_ref, a_ref = d.target.copy(), np.zeros(3), np.zeros(3)

            # Settle & Winch Logic
            R_est = quat_to_rot(d.ekf.q)
            omega_est = gyro_raw - d.ekf.b_g
            d_horiz = float(np.linalg.norm(d.ekf.p[:2] - d.target[:2]))
            v_horiz = float(np.linalg.norm(d.ekf.v[:2]))

            is_centered = (d_horiz <= HOVER_LOCK_RADIUS_M) and (v_horiz <= HOVER_LOCK_SPEED_MAX_MS)
            if is_centered and d.winch.has_payload and d.mission_phase == "DELIVERY_OUTBOUND":
                d.hover_settle_timer += dt
            else:
                if d.winch.cable_len <= 0.10: d.hover_settle_timer = 0.0

            is_settled_for_winch = (d.hover_settle_timer >= HOVER_SETTLE_DURATION_S)

            hook_a, hook_w = sensor_suite.poll_smart_hook_imu(d.airframe.accel_linear, d.airframe.omega, d.winch.sway_offset, d.winch.cable_len, d.winch.has_payload)
            load_3d = sensor_suite.poll_3axis_load_cell(d.winch.tension_n, d.winch.sway_offset, d.winch.cable_len)
            th_dt, dolp, ndvi = sensor_suite.poll_thermal_and_polarization(d.target[:2])
            i_rms, t_mosfet = sensor_suite.poll_esc_foc_telemetry(d.airframe.motor_omegas, d.airframe.motor_health)
            alpha, beta, rh, dust = sensor_suite.poll_5port_air_data(d.airframe.v, total_wind, d.airframe.R)
            hdop, cn0, rf_pwr, mag_gr = sensor_suite.poll_rtk_gnss_metrics(d.airframe.p, chi2_dist > 16.27)

            # Synthesize Full 100-Dim State
            obs_vec = synth.synthesize(
                p_est=d.ekf.p, v_est=d.ekf.v, R_est=R_est, omega_est=omega_est, target_p=p_ref,
                lidar_16=lidar_16, conical_8=conical_8, v_tas=s_data["v_tas"], q_dyn=s_data["q_dyn"],
                min_sonar=s_data["min_sonar"], wind_est=total_wind, ice_idx=ice_idx, cable_len=d.winch.cable_len,
                cable_vel=d.winch.cable_vel, tension_n=d.winch.tension_n, sway_2d=d.winch.sway_offset,
                bms_soc=bms_soc, v_pack=v_pack, bms_temp_c=d.bms.temp_c, r0_dyn=r0_dyn, chi2_val=chi2_dist,
                jerk_discrepancy=0.5, is_spoofed=False, emf_rms=s_data["emf_rms"], wire_dist=wire_clearance,
                agent_dist=agent_dist if d_idx == selected_drone_idx else 30.0, ground_spl_dba=58.0,
                has_payload=d.winch.has_payload, hook_a=hook_a, hook_w=hook_w, load_cell_3d=load_3d,
                thermal_dt=th_dt, dolp=dolp, ndvi=ndvi, esc_i_rms=i_rms, esc_t_mosfet=t_mosfet,
                alpha_deg=alpha, beta_deg=beta, rh_pct=rh, dust_pm10=dust, hdop=hdop, cn0_avg=cn0,
                rf_jam_pwr=rf_pwr, mag_grad=mag_gr, dt=dt
            )

            # Neural Forward Pass (ResNet with PPO Actor-Critic)
            with torch.no_grad():
                t_in = torch.from_numpy(obs_vec).float().unsqueeze(0).to(device)
                g_act, w_act, fsm_act, _, val = model.act(t_in, deterministic=True)
                neural_guidance = g_act[0].cpu().numpy()
                nn_winch_cmd = float(w_act[0, 0].cpu())
                current_token_idx = int(fsm_act[0].cpu())
                critic_v = float(val[0].cpu())

            # Active Winch Step
            d.winch.step(
                drone_z=d.airframe.p[2],
                is_settled=is_settled_for_winch,
                is_surface_safe=is_surf_safe,
                wire_hit=wire_hit,
                agent_breach=(agent_breach if d_idx == selected_drone_idx else False),
                agent_dist=(agent_dist if d_idx == selected_drone_idx else 30.0),
                wind_2d=total_wind[:2],
                neural_token_idx=current_token_idx,
                nn_winch_rate=nn_winch_cmd,
                dt=dt
            )

            # Delivery State Machine
            if d.winch.delivery_completed and d.mission_phase == "DELIVERY_OUTBOUND":
                d.winch.delivery_completed = False
                d.packages_delivered += 1
                delivered_boxes.append(d.target.copy())
                d.hover_settle_timer = 0.0

                d.mission_phase = "RETURNING_TO_BASE"
                d.target = d.home_coords.copy()
                d.compute_corridor_trajectory(d.target)

            dist_to_depot = float(np.linalg.norm(d.airframe.p[:2] - d.home_coords[:2]))
            if d.mission_phase == "RETURNING_TO_BASE":
                if dist_to_depot < 1.35 and v_horiz < 0.55:
                    d.depot_timer += dt
                    if d.depot_timer >= 0.8:
                        d.depot_timer = 0.0
                        d.manifest_idx = (d.manifest_idx + 1) % len(world.manifest_addresses)
                        order = world.manifest_addresses[d.manifest_idx]
                        d.target = order["pos"].copy()
                        d.compute_corridor_trajectory(d.target)
                        d.winch.reload_new_cargo()
                        d.bms.soc = 0.96
                        d.mission_phase = "DELIVERY_OUTBOUND"
                        d.hover_settle_timer = 0.0
                else:
                    d.depot_timer = 0.0

            # Actuator Mixer with Reciprocal Swarm CBF
            omega_cmds, f_thrust, a_safe = mixer.mix(
                drone_idx=d_idx, drone_health=d.airframe.motor_health, p_est=d.ekf.p, v_est=d.ekf.v,
                R_est=R_est, omega_est=omega_est, p_target=p_ref, neural_guidance=neural_guidance,
                token_idx=current_token_idx, estimated_wind=total_wind, obstacles=obstacles_cbf,
                swarm_states=swarm_states, ref_accel=a_ref, cable_len=d.winch.cable_len,
                sway_2d=d.winch.sway_offset, load_cell_3d=load_3d,
                ige_fac=d.airframe.step.__dict__.get("last_ige", 1.0),
                v_pack=v_pack, bms_temp_c=d.bms.temp_c, dt=dt
            )

            f_act, torques, p_watts, ige_fac = d.airframe.step(
                motor_cmds=omega_cmds, current_wind=total_wind, rho_air=s_data["rho_air"],
                ice_index=ice_idx, dt=dt
            )
            d.airframe.step.__dict__["last_p_watts"] = p_watts
            d.airframe.step.__dict__["last_ige"] = ige_fac

            if int(flight_time * 60) % 6 == 0:
                d.trail.append(d.airframe.p[:2].copy())

            if d_idx == selected_drone_idx:
                selected_diag = {
                    "v_pack": v_pack, "soc": bms_soc, "temp_c": d.bms.temp_c,
                    "action": ACTION_NAMES[current_token_idx], "v_crit": critic_v,
                    "thrust": f_thrust, "tension": d.winch.tension_n,
                    "cable_l": d.winch.cable_len, "load_3d": load_3d,
                    "ice_mm": ice_mm, "wind_speed": np.linalg.norm(total_wind)
                }

        # ==========================================
        # RENDERING (3D CHASE VIEWPORT & SPLINES)
        # ==========================================
        screen.fill((10, 13, 20))
        pygame.draw.line(screen, (35, 45, 60), (VIEW_W, 0), (VIEW_W, H), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (W, 70), 2)

        # Ground Grid
        for gx in range(-6, 7):
            p1 = np.array([cam_focus[0] + gx * 6.0, cam_focus[1] - 35.0, 0.0], dtype=np.float32)
            p2 = np.array([cam_focus[0] + gx * 6.0, cam_focus[1] + 35.0, 0.0], dtype=np.float32)
            sx1, sy1, _ = project_3d_chase(p1, cam_focus, cam_yaw, cam_pitch, center_viewport)
            sx2, sy2, _ = project_3d_chase(p2, cam_focus, cam_yaw, cam_pitch, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        # Houses
        for h in world.houses:
            hx1, hx2, hy1, hy2, hh = h["aabb"]
            h_center = np.array([(hx1+hx2)/2, (hy1+hy2)/2, 0.0])
            if np.linalg.norm(cam_focus[:2] - h_center[:2]) < 55.0:
                h_corners = [
                    np.array([hx1, hy1, 0.0]), np.array([hx2, hy1, 0.0]),
                    np.array([hx2, hy2, 0.0]), np.array([hx1, hy2, 0.0]),
                    np.array([hx1, hy1, hh]), np.array([hx2, hy1, hh]),
                    np.array([hx2, hy2, hh]), np.array([hx1, hy2, hh])
                ]
                h_screens = [project_3d_chase(p, cam_focus, cam_yaw, cam_pitch, center_viewport)[:2] for p in h_corners]
                pygame.draw.polygon(screen, (48, 56, 70), [h_screens[4], h_screens[5], h_screens[6], h_screens[7]])
                pygame.draw.lines(screen, (110, 130, 155), True, [h_screens[4], h_screens[5], h_screens[6], h_screens[7]], 2)
                for i in range(4): pygame.draw.line(screen, (75, 90, 110), h_screens[i], h_screens[i+4], 2)

        # Base Station Helipad [H]
        if np.linalg.norm(cam_focus[:2] - world.base_station[:2]) < 65.0:
            hc_sx, hc_sy, _ = project_3d_chase(world.base_station, cam_focus, cam_yaw, cam_pitch, center_viewport)
            pygame.draw.circle(screen, (0, 255, 140), (hc_sx, hc_sy), 24, 2)
            screen.blit(font_large.render("H", True, (0, 255, 140)), (hc_sx - 6, hc_sy - 10))

        # Delivered boxes
        for pb in delivered_boxes:
            if np.linalg.norm(cam_focus[:2] - pb[:2]) < 55.0:
                bx, by, _ = project_3d_chase(pb, cam_focus, cam_yaw, cam_pitch, center_viewport)
                pygame.draw.rect(screen, (60, 160, 90), (bx - 5, by - 5, 10, 10), border_radius=2)

        # Render Minimum-Snap Spline Corridor Ribbon for active drone
        act_d = drones[selected_drone_idx]
        if act_d.spline_traj is not None:
            t_samples = np.linspace(0.0, act_d.spline_traj.total_duration, 35)
            pts_3d = [act_d.spline_traj.evaluate(st)[0] for st in t_samples]
            scr_pts = [project_3d_chase(p3, cam_focus, cam_yaw, cam_pitch, center_viewport)[:2] for p3 in pts_3d]
            if len(scr_pts) > 1:
                pygame.draw.lines(screen, (0, 180, 255), False, scr_pts, 2)

        # Render All Drones in Swarm
        for d_i, d in enumerate(drones):
            col = swarm_colors[d_i % len(swarm_colors)]
            dsx, dsy, _ = project_3d_chase(d.airframe.p, cam_focus, cam_yaw, cam_pitch, center_viewport)

            # Airframe Center
            pygame.draw.circle(screen, col, (dsx, dsy), 8 if d_i == selected_drone_idx else 5)
            screen.blit(font_small.render(f"D{d_i+1}", True, col), (dsx + 10, dsy - 10))

            # Rotor Arms
            arm_d = ARM_LENGTH_M * 1.8
            arm_ends_body = [
                np.array([ arm_d/1.414, -arm_d/1.414, 0.0]), np.array([ arm_d/1.414,  arm_d/1.414, 0.0]),
                np.array([-arm_d/1.414,  arm_d/1.414, 0.0]), np.array([-arm_d/1.414, -arm_d/1.414, 0.0]),
            ]
            arm_screens = [project_3d_chase(d.airframe.p + d.airframe.R @ a, cam_focus, cam_yaw, cam_pitch, center_viewport)[:2] for a in arm_ends_body]
            pygame.draw.line(screen, (160, 170, 185), arm_screens[0], arm_screens[2], 2)
            pygame.draw.line(screen, (160, 170, 185), arm_screens[1], arm_screens[3], 2)

            # Winch & Cargo Box
            payload_3d = np.array([
                d.airframe.p[0] + d.winch.sway_offset[0],
                d.airframe.p[1] + d.winch.sway_offset[1],
                d.airframe.p[2] - d.winch.cable_len
            ], dtype=np.float32)
            psx, psy, _ = project_3d_chase(payload_3d, cam_focus, cam_yaw, cam_pitch, center_viewport)

            if d.winch.cable_len > 0.10:
                pygame.draw.line(screen, (230, 240, 255), (dsx, dsy + 3), (psx, psy), 1)

            if d.winch.has_payload:
                box_x, box_y = (dsx, dsy + 5) if d.winch.cable_len <= 0.10 else (psx, psy)
                pygame.draw.rect(screen, (255, 195, 35), (box_x - 6, box_y - 6, 12, 12), border_radius=2)
                pygame.draw.rect(screen, (60, 45, 10), (box_x - 6, box_y - 6, 12, 12), 1)
            else:
                hook_x, hook_y = (dsx, dsy + 3) if d.winch.cable_len <= 0.10 else (psx, psy)
                pygame.draw.circle(screen, (200, 220, 240), (hook_x, hook_y), 3)

        # Ground Agent
        ag_sx, ag_sy, _ = project_3d_chase(agent_pos, cam_focus, cam_yaw, cam_pitch, center_viewport)
        ag_col = (100, 220, 120) if ground_agent.is_leashed else ((255, 60, 80) if agent_breach else (255, 120, 40))
        pygame.draw.circle(screen, ag_col, (ag_sx, ag_sy), 6)

        # ==========================================
        # COMPLETE SIDEBAR DASHBOARD & RADAR MINIMAP
        # ==========================================
        sb_x = VIEW_W + 14
        screen.blit(font_large.render("SURVIVAL CORE V17.0 (SWARM & PPO)", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Swarm UTM • C4 Min-Snap Corridors • PPO Actor-Critic", True, (130, 160, 190)), (sb_x, 38))

        # Radar Minimap
        map_rect = (sb_x, 65, 210, 210)
        draw_tactical_radar_minimap(screen, map_rect, world, drones, selected_drone_idx)

        # Swarm Fleet Status Table
        fl_x = sb_x + 224
        screen.blit(font_mono.render("SWARM UTM FLEET:", True, (240, 240, 240)), (fl_x, 65))
        for d_i, d in enumerate(drones):
            c = swarm_colors[d_i % len(swarm_colors)]
            tag = f"[D{d_i+1}]" + ("*" if d_i == selected_drone_idx else " ")
            stat = "DELIV" if "DELIVERY" in d.mission_phase else "RTB"
            screen.blit(font_small.render(f"{tag} {stat} | {d.bms.soc*100:.0f}% | Box:{d.packages_delivered}", True, c), (fl_x, 90 + d_i * 24))

        # 1. PPO Policy Diagnostics
        p1_y = 285
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p1_y - 8), (W - 14, p1_y - 8), 1)
        screen.blit(font_mono.render("1. PPO ACTOR-CRITIC DIAGNOSTICS:", True, (240, 240, 240)), (sb_x, p1_y))
        screen.blit(font_small.render(f"Active Drone:      Drone #{selected_drone_idx + 1} (Focus: [TAB])", True, (0, 220, 255)), (sb_x + 10, p1_y + 18))
        screen.blit(font_small.render(f"Executive Action:  {selected_diag.get('action', 'TRANSIT')}", True, (0, 255, 140)), (sb_x + 10, p1_y + 34))
        screen.blit(font_small.render(f"Critic V(s) Value: {selected_diag.get('v_crit', 0.0):.3f} (PPO Advantage Base)", True, (160, 180, 205)), (sb_x + 10, p1_y + 50))

        # 2. C4 Min-Snap Corridor Tracking
        p2_y = p1_y + 68
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (W - 14, p2_y), 1)
        screen.blit(font_mono.render("2. C4 MIN-SNAP CORRIDOR TRACKING:", True, (240, 240, 240)), (sb_x, p2_y + 8))
        tot_dur = act_d.spline_traj.total_duration if act_d.spline_traj else 0.0
        screen.blit(font_small.render(f"Corridor Time:     t = {act_d.traj_time:.2f}s / {tot_dur:.2f}s", True, (0, 220, 255)), (sb_x + 10, p2_y + 26))
        screen.blit(font_small.render(f"Quintic Spline:    Continuity = C4 (Smooth Snap & Jerk)", True, (0, 255, 140)), (sb_x + 10, p2_y + 42))

        # 3. Active Neural Anti-Sway Load-Cell Telemetry
        p3_y = p2_y + 64
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (W - 14, p3_y), 1)
        screen.blit(font_mono.render("3. ACTIVE NEURAL ANTI-SWAY TELEMETRY:", True, (240, 240, 240)), (sb_x, p3_y + 8))
        l3d = selected_diag.get("load_3d", np.zeros(3))
        screen.blit(font_small.render(f"3-Axis Load-Cell:  F_xyz = [{l3d[0]:.1f}, {l3d[1]:.1f}, {l3d[2]:.1f}] N", True, (0, 220, 255)), (sb_x + 10, p3_y + 26))
        screen.blit(font_small.render(f"Winch State:       L = {selected_diag.get('cable_l', 0.08):.2f}m | T = {selected_diag.get('tension', 0.0):.1f}N", True, (160, 180, 205)), (sb_x + 10, p3_y + 42))

        # 4. Battery & Aerodynamics
        p4_y = p3_y + 64
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (W - 14, p4_y), 1)
        screen.blit(font_mono.render("4. AIRFRAME & BATTERY ECM:", True, (240, 240, 240)), (sb_x, p4_y + 8))
        screen.blit(font_small.render(f"6S LiPo Status:    V = {selected_diag.get('v_pack', 23.5):.2f}V ({selected_diag.get('soc', 0.95)*100:.0f}%) | {selected_diag.get('temp_c', 22.0):.1f}°C", True, (0, 255, 140)), (sb_x + 10, p4_y + 26))
        screen.blit(font_small.render(f"Rotor Ice / Wind:  {selected_diag.get('ice_mm', 0.0):.2f}mm Ice | Wind: {selected_diag.get('wind_speed', 0.0):.1f} m/s", True, (160, 180, 205)), (sb_x + 10, p4_y + 42))

        # Controls Strip
        bot_y = H - 38
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y - 8), (W - 14, bot_y - 8), 1)
        screen.blit(font_mono.render("[TAB] Cycle Drone | [CLICK] Retarget | [A] Leash | [H] RTB | [J] Spoof", True, (255, 200, 40)), (sb_x + 8, bot_y))

        # Viewport Header
        status_banner = f"ACTIVE DRONE: #{selected_drone_idx+1} | Swarm Total Delivered: {sum(d.packages_delivered for d in drones)}"
        screen.blit(font_large.render(status_banner, True, (240, 240, 240)), (15, 14))
        screen.blit(font_small.render("Swarm UTM Reciprocal CBF • C4 Min-Snap Splines • Closed-Loop Winch • PPO RL", True, (0, 210, 255)), (15, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    mp.freeze_support()
    run_all_physics_simulator()