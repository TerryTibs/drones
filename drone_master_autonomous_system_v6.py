"""
=============================================================================
MASTER AUTONOMOUS DRONE SYSTEM CORE (V9.0 - SUBURBAN DISTRICT & MACRO TRAINING)
File: drone_master_autonomous_system_v5.py
=============================================================================
- Macro-World: 120m x 120m Suburban District with 10 Houses, Roads & Catenary Grid
- Spatial Hash: O(1) Uniform Grid Partitioning for Fast Multi-Obstacle Raycasting
- Perception: 56-Dimensional State Vector with 16-Beam 360° LiDAR + 8-Beam Conical
- Cockpit: Dynamic 3D Chase Camera + Real-time Top-Down Tactical Radar Mini-Map
- Logistics: Sequential Multi-Stop Delivery Manifest with Automated Depot RTB
- Physics: BEMT, Icing, Sub-Zero Arrhenius, Rooftop Wake & Anti-Spoof Quantum VIO
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
DRY_MASS_KG = 1.35                   
CARGO_MASS_KG = 0.45                 
MASS_KG = DRY_MASS_KG + CARGO_MASS_KG 

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
CEILING_HEIGHT_M = 30.0

CBF_ALPHA_OBSTACLE = 1.8
CBF_ALPHA_CEILING = 2.2
CBF_SAFE_RADIUS_M = 0.70

K_EFF_WAVEVECTOR = 1.61e7            
RAMAN_PULSE_SEP_T = 0.015            
QUANTUM_PHASE_1G = K_EFF_WAVEVECTOR * G_ACCEL * (RAMAN_PULSE_SEP_T ** 2)

P_0 = 1013.25
T_0 = 288.15
L_LAPSE = 0.0065
R_SPEC_AIR = 287.058
SPEED_OF_SOUND = 343.2
EMF_WARNING_THRESHOLD_UT = 18.0

NUM_CELLS = 6
PACK_CAPACITY_AH = 6.0
CELL_R0_NOMINAL = 0.0085
CELL_R1 = 0.0065
CELL_C1 = 2800.0
ARRHENIUS_EA = 28000.0
GAS_CONST_R = 8.31446
T_REF_K = 298.15

PAYLOAD_WEIGHT_N = CARGO_MASS_KG * G_ACCEL
GUILLOTINE_THRESHOLD_N = 48.0
HOVER_LOCK_RADIUS_M = 0.90
HOVER_LOCK_SPEED_MAX_MS = 0.40
HOVER_SETTLE_DURATION_S = 0.30

# Perception & Manifest
OBSERVATION_DIM = 56                 # Expanded for 16-beam + 8-beam LiDAR
LIDAR_MAX_RANGE_M = 25.0
CONICAL_MAX_RANGE_M = 12.0
NUM_EXECUTIVE_STATES = 6
MODEL_WEIGHTS_FILE = "drone_master_brain_v9_complete.pth"

ACTION_NAMES = [
    "TRANSIT",             # 0: Cruising to address waypoint along spline
    "HOVER_LOCK",          # 1: Precision settling over customer dropzone
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
# 2. SPATIAL HASH GRID & SUBURBAN DISTRICT
# ==========================================
class SuburbanDistrictMacroWorld:
    """
    120m x 120m Macro-World with uniform spatial hash partitioning.
    Contains: 10 Residential Homes, Road Network, Swimming Pools,
    Tree Clusters, Catenary Overhead Transmission Grid, and Base Depot.
    """
    def __init__(self, sector_span_m=120.0, cell_size_m=10.0):
        self.sector_span = sector_span_m
        self.cell_size = cell_size_m
        self.grid = {}

        self.houses = [
            {"id": "House_A", "aabb": [-25, -15, -15, -5, 4.8], "roof_type": "GABLE"},
            {"id": "House_B", "aabb": [ 10,  22, -18, -6, 5.2], "roof_type": "HIP"},
            {"id": "House_C", "aabb": [-32, -20,  12,  24, 4.5], "roof_type": "GABLE"},
            {"id": "House_D", "aabb": [  8,  20,  15,  28, 5.0], "roof_type": "FLAT"},
            {"id": "House_E", "aabb": [ 30,  44,  -8,   4, 5.5], "roof_type": "GABLE"},
            {"id": "House_F", "aabb": [-48, -36,  -5,   6, 4.2], "roof_type": "HIP"},
            {"id": "House_G", "aabb": [-12,   0,  32,  44, 4.9], "roof_type": "GABLE"},
            {"id": "House_H", "aabb": [ 24,  36,  30,  42, 5.1], "roof_type": "HIP"},
            {"id": "House_I", "aabb": [-38, -26, -38, -26, 4.6], "roof_type": "GABLE"},
            {"id": "House_J", "aabb": [ 15,  28, -42, -30, 4.8], "roof_type": "FLAT"}
        ]

        self.trees = [
            {"pos": np.array([-18, 5, 0], dtype=np.float32), "r": 2.2, "h": 6.5},
            {"pos": np.array([  2, 8, 0], dtype=np.float32), "r": 2.5, "h": 7.0},
            {"pos": np.array([ 25, -22, 0], dtype=np.float32), "r": 2.0, "h": 5.8},
            {"pos": np.array([-40, 20, 0], dtype=np.float32), "r": 2.4, "h": 6.8}
        ]

        self.pools = [
            {"pos": np.array([-22, -22, 0], dtype=np.float32), "r": 2.8, "type": "POOL_COVER"},
            {"pos": np.array([ 15,  8, 0], dtype=np.float32), "r": 2.5, "type": "OPEN_WATER"}
        ]

        self.powerlines = [
            {"p1": np.array([-55, -20, 8.5], dtype=np.float32), "p2": np.array([55, -20, 8.5], dtype=np.float32)},
            {"p1": np.array([-55,  22, 8.5], dtype=np.float32), "p2": np.array([55,  22, 8.5], dtype=np.float32)}
        ]

        self.roads = [
            {"x_min": -60, "x_max": 60, "y_min": -2, "y_max": 2},   # Main East-West Boulevard
            {"x_min": -2, "x_max": 2, "y_min": -60, "y_max": 60}    # Central North-South Avenue
        ]

        self.base_station = np.array([-42.0, -38.0, 0.0], dtype=np.float32)

        # Delivery Manifest Waypoints across suburban addresses
        self.delivery_manifest = [
            {"id": "Order_101", "pos": np.array([-20.0, -10.0, 0.0], dtype=np.float32), "address": "12 Oak Lane"},
            {"id": "Order_102", "pos": np.array([ 16.0, -12.0, 0.0], dtype=np.float32), "address": "45 Maple Court"},
            {"id": "Order_103", "pos": np.array([-26.0,  18.0, 0.0], dtype=np.float32), "address": "78 Pine Ridge"},
            {"id": "Order_104", "pos": np.array([ 14.0,  22.0, 0.0], dtype=np.float32), "address": "104 Elm Drive"},
            {"id": "Order_105", "pos": np.array([ 36.0,  -2.0, 0.0], dtype=np.float32), "address": "22 Cedar Way"},
            {"id": "Order_106", "pos": np.array([-6.0,   38.0, 0.0], dtype=np.float32), "address": "15 Birch Boulevard"}
        ]

        self._build_spatial_hash()

    def _hash_coord(self, x, y):
        return (int(math.floor(x / self.cell_size)), int(math.floor(y / self.cell_size)))

    def _build_spatial_hash(self):
        for h in self.houses:
            x1, x2, y1, y2, _ = h["aabb"]
            cx1, cy1 = self._hash_coord(x1, y1)
            cx2, cy2 = self._hash_coord(x2, y2)
            for gx in range(cx1, cx2 + 1):
                for gy in range(cy1, cy2 + 1):
                    self.grid.setdefault((gx, gy), []).append({"type": "HOUSE", "data": h})

        for t in self.trees:
            cx, cy = self._hash_coord(t["pos"][0], t["pos"][1])
            for gx in range(cx - 1, cx + 2):
                for gy in range(cy - 1, cy + 2):
                    self.grid.setdefault((gx, gy), []).append({"type": "TREE", "data": t})

    def query_spatial_box(self, x, y, radius_m):
        cx, cy = self._hash_coord(x, y)
        span = int(math.ceil(radius_m / self.cell_size))
        candidates = []
        for gx in range(cx - span, cx + span + 1):
            for gy in range(cy - span, cy + span + 1):
                if (gx, gy) in self.grid:
                    candidates.extend(self.grid[(gx, gy)])
        return candidates

# ==========================================
# 3. 16-BEAM 360° LIDAR & CONICAL SOUNDER
# ==========================================
class MacroPerceptionLiDARSuite:
    """
    Simulates:
    1. 16-Beam Planar LiDAR (360° around yaw axis, slab AABB intersection).
    2. 8-Beam Downward Conical Acoustic Scanner (dropzone ground solidity & wires).
    """
    def __init__(self, world: SuburbanDistrictMacroWorld):
        self.world = world

    def cast_360_lidar(self, drone_pos, yaw_rad):
        px, py, pz = drone_pos
        ranges = np.full(16, LIDAR_MAX_RANGE_M, dtype=np.float32)
        candidates = self.world.query_spatial_box(px, py, LIDAR_MAX_RANGE_M)

        for beam_idx in range(16):
            angle = yaw_rad + (2.0 * math.pi * beam_idx / 16.0)
            dx = math.cos(angle)
            dy = math.sin(angle)

            min_dist = LIDAR_MAX_RANGE_M

            for item in candidates:
                if item["type"] == "HOUSE":
                    x1, x2, y1, y2, h = item["data"]["aabb"]
                    if pz <= h: # Height intersection
                        # Slab AABB test
                        tx1 = (x1 - px) / dx if abs(dx) > 1e-4 else -1e6
                        tx2 = (x2 - px) / dx if abs(dx) > 1e-4 else 1e6
                        ty1 = (y1 - py) / dy if abs(dy) > 1e-4 else -1e6
                        ty2 = (y2 - py) / dy if abs(dy) > 1e-4 else 1e6

                        tmin_x, tmax_x = min(tx1, tx2), max(tx1, tx2)
                        tmin_y, tmax_y = min(ty1, ty2), max(ty1, ty2)

                        t_enter = max(tmin_x, tmin_y)
                        t_exit = min(tmax_x, tmax_y)

                        if t_exit >= t_enter and t_enter > 0.0:
                            if t_enter < min_dist:
                                min_dist = t_enter

                elif item["type"] == "TREE":
                    t_pos = item["data"]["pos"]
                    tr = item["data"]["r"]
                    th = item["data"]["h"]
                    if pz <= th:
                        # 2D Ray-Circle test
                        ox = px - t_pos[0]
                        oy = py - t_pos[1]
                        b_val = ox * dx + oy * dy
                        c_val = ox * ox + oy * oy - tr * tr
                        disc = b_val * b_val - c_val
                        if disc >= 0:
                            t_hit = -b_val - math.sqrt(disc)
                            if 0.0 < t_hit < min_dist:
                                min_dist = t_hit

            ranges[beam_idx] = min_dist

        return ranges

    def cast_downward_conical(self, drone_pos):
        px, py, pz = drone_pos
        conical_ranges = np.full(8, CONICAL_MAX_RANGE_M, dtype=np.float32)
        wire_clearance = 50.0

        for wire in self.world.powerlines:
            w1, w2 = wire["p1"], wire["p2"]
            if min(w1[0], w2[0]) - 2.0 <= px <= max(w1[0], w2[0]) + 2.0:
                dist_y = abs(py - w1[1])
                if dist_y < 3.0 and pz > w1[2]:
                    wire_clearance = min(wire_clearance, float(pz - w1[2]))

        cone_slant = math.cos(math.radians(25.0))
        for i in range(8):
            raw_depth = pz / max(0.1, cone_slant)
            conical_ranges[i] = min(CONICAL_MAX_RANGE_M, float(raw_depth))

        return conical_ranges, wire_clearance

# ==========================================
# 4. ROOFTOP WAKE & LEEWARD CAVITY
# ==========================================
class MacroBuildingWakeEngine:
    def __init__(self, world: SuburbanDistrictMacroWorld):
        self.world = world

    def evaluate_wake(self, drone_pos, freestream_wind):
        px, py, pz = drone_pos
        w_wake = np.zeros(3, dtype=np.float32)
        wind_speed = float(np.linalg.norm(freestream_wind[:2]))
        if wind_speed < 0.2: return w_wake

        wind_dir = freestream_wind[:2] / wind_speed
        candidates = self.world.query_spatial_box(px, py, 12.0)

        for item in candidates:
            if item["type"] == "HOUSE":
                x1, x2, y1, y2, h = item["data"]["aabb"]
                dx = px - x2 if wind_dir[0] >= 0 else x1 - px
                if 0.0 < dx < 6.5 and y1 - 1.0 <= py <= y2 + 1.0 and pz < h * 1.5:
                    decay = math.exp(-dx / 2.5)
                    z_ratio = max(0.1, min(1.0, pz / h))
                    downwash = -wind_speed * 0.75 * (1.0 - z_ratio) * decay
                    recirc_u = -wind_speed * 0.35 * decay
                    w_wake[0] += recirc_u * wind_dir[0]
                    w_wake[1] += recirc_u * wind_dir[1]
                    w_wake[2] += downwash
                    break

        return w_wake

# ==========================================
# 5. DYNAMIC GROUND AGENT WITH DETERRENCE
# ==========================================
class MacroGroundIntruderAgent:
    def __init__(self, init_pos):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.v_max = 3.0
        self.personal_space_m = 1.85
        self.curiosity_timer = 0.0
        self.boredom_cooldown = 0.0
        self.is_leashed = False

    def step(self, cargo_pos, is_winch_lowering, is_aborting, dt=0.016):
        if self.is_leashed:
            self.p = np.array([-40.0, -32.0, 0.0], dtype=np.float32)
            self.v.fill(0.0)
            return self.p.copy(), False, 99.0

        d_vec = cargo_pos[:2] - self.p[:2]
        dist = float(np.linalg.norm(d_vec))

        if is_aborting:
            self.boredom_cooldown = 10.0
            self.curiosity_timer = 0.0
            flee = -d_vec / max(1e-4, dist)
            v_des = flee * (self.v_max * 1.3)
            a_cmd = (v_des - self.v[:2]) * 5.0

        elif self.boredom_cooldown > 0.0:
            self.boredom_cooldown -= dt
            v_des = np.array([0.5, -0.4])
            a_cmd = (v_des - self.v[:2]) * 2.0

        elif is_winch_lowering and cargo_pos[2] < 2.2 and dist < 6.5:
            self.curiosity_timer += dt
            if self.curiosity_timer > 2.5:
                self.boredom_cooldown = 8.0
                v_des = np.zeros(2)
                a_cmd = -self.v[:2] * 3.0
            else:
                v_des = (d_vec / max(1e-4, dist)) * self.v_max
                a_cmd = (v_des - self.v[:2]) * 3.5
        else:
            self.curiosity_timer = 0.0
            v_des = np.array([math.sin(time.time()*0.5)*0.6, math.cos(time.time()*0.5)*0.6])
            a_cmd = (v_des - self.v[:2]) * 1.5

        self.v[:2] += a_cmd * dt
        self.p[:2] += self.v[:2] * dt
        self.p[2] = 0.0

        breach = (dist < self.personal_space_m)
        return self.p.copy(), breach, dist

# ==========================================
# 6. ROTOR ICING ACCRETION MODEL
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
# 7. DRYDEN ATMOSPHERIC TURBULENCE
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
# 8. BILATERAL SLUNG WINCH WITH CLEARANCE
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

    def step(self, drone_pos, drone_accel, is_settled, is_surface_safe, wire_hit, agent_breach, agent_dist, wind_vec, dt=0.016):
        if self.severed:
            self.state = "GUILLOTINE"
            return self.cable_len, 0.0, np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, -1.0], dtype=np.float32)

        dropzone_occupied = (agent_dist < 2.8) and (self.state == "STANDBY")
        lockout = (not is_surface_safe) or wire_hit or (not is_settled) or dropzone_occupied

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
# 9. CONTINUOUS BEMT COAXIAL AIRFRAME
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
        return f_thrust, motor_torques, p_watts, ige_factor  # Explicitly returns ige_factor

# ==========================================
# 10. SUB-ZERO ARRHENIUS BATTERY ECM
# ==========================================
class SubZeroArrheniusBatteryECM:
    def __init__(self):
        self.soc = 0.96
        self.temp_c = 22.0
        self.v1 = np.zeros(NUM_CELLS, dtype=np.float32)
        self.capacity_coulombs = PACK_CAPACITY_AH * 3600.0
        self.v_pack = 24.2

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
# 11. QUANTUM SENSOR & ES-EKF ON SO(3)
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
# 12. CBF-QP & RATE-DAMPED MIXER
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

        if token_idx == 0:  # TRANSIT along macro-route
            v_cruise = 3.6  # 3.6 m/s rapid district transit
            dir_norm = delta_p / max(1e-4, float(np.linalg.norm(delta_p)))
            v_des = dir_norm * v_cruise
            if d_horiz < 3.0: v_des[:2] *= (d_horiz / 3.0)
        else:
            v_des = np.zeros(3, dtype=np.float32)

        e_p = p_est - p_target
        e_v = v_est - v_des
        self.e_p_integral = np.clip(self.e_p_integral + e_p * dt, -1.8, 1.8)

        f_neural = neural_guidance[:3] * 5.0
        a_nominal = (-np.array([6.5, 6.5, 11.5]) * e_p - np.array([4.4, 4.4, 7.0]) * e_v
                     - np.array([1.2, 1.2, 2.5]) * self.e_p_integral + f_neural) / active_mass_kg

        a_safe = self.cbf_filter.enforce_safety(p_est, v_est, a_nominal, obstacles)
        F_d = a_safe * active_mass_kg + active_mass_kg * (-GRAVITY_G)

        max_tilt = math.radians(28.0)
        f_h = math.hypot(F_d[0], F_d[1])
        f_v = max(1.0, F_d[2])
        if f_h / f_v > math.tan(max_tilt):
            F_d[:2] *= (f_v * math.tan(max_tilt)) / f_h

        b3 = R_est[:, 2]
        f_thrust = float(np.clip(np.dot(F_d, b3), 0.1, 45.0))

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

        T_cmds = np.clip(T_cmds * drone_health, 0.0, 11.5)

        omega_cmds = np.zeros(8, dtype=np.float32)
        for i in range(8):
            if drone_health[i] > 0.0:
                scale_c = C_T_CLEAN if (i % 2 == 0) else (C_T_CLEAN * COAXIAL_EFFICIENCY)
                omega_cmds[i] = math.sqrt(T_cmds[i] / scale_c)

        return omega_cmds, f_thrust, a_safe

# ==========================================
# 13. NEURAL NETWORK ARCHITECTURE & OBS ENCODER
# ==========================================
class MacroScaleFlightExecutiveNet(nn.Module):
    def __init__(self, in_features=OBSERVATION_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU()
        )
        self.guidance_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 4),
            nn.Tanh()
        )
        self.executive_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, NUM_EXECUTIVE_STATES)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.guidance_head(feat), self.executive_head(feat)

class MacroObservationSynthesizer:
    def __init__(self):
        self.last_tension = PAYLOAD_WEIGHT_N

    def synthesize(self, p_est, v_est, R_est, omega_est, target_p, lidar_16, conical_8,
                   v_tas, q_dyn, wind_est, bms_soc, v_pack, cable_len, cable_vel,
                   tension_n, has_payload, chi2_val, is_spoofed, jerk_discrepancy,
                   flight_time, completed_count, dt=0.016):
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)

        # 1. 16-Beam Planar LiDAR [0:16]
        obs[0:16] = np.clip(lidar_16 / LIDAR_MAX_RANGE_M, 0.0, 1.0)

        # 2. 8-Beam Conical Scanner [16:24]
        obs[16:24] = np.clip(conical_8 / CONICAL_MAX_RANGE_M, 0.0, 1.0)

        # 3. Kinematic States [24:36]
        delta_p = target_p - p_est
        obs[24:27] = np.clip(delta_p / 30.0, -1.0, 1.0)
        obs[27:30] = np.clip(v_est / 8.0, -1.0, 1.0)
        obs[30:33] = np.clip(R_est[:, 2], -1.0, 1.0)
        obs[33:36] = np.clip(omega_est / 8.0, -1.0, 1.0)

        # 4. Polar Goal Geometry [36:39]
        dist_goal = float(np.linalg.norm(delta_p))
        bearing_goal = math.atan2(delta_p[1], delta_p[0])
        elev_goal = math.atan2(delta_p[2], max(0.1, math.hypot(delta_p[0], delta_p[1])))
        obs[36] = np.clip(dist_goal / 60.0, 0.0, 1.0)
        obs[37] = bearing_goal / math.pi
        obs[38] = elev_goal / (0.5 * math.pi)

        # 5. Airspeed & Aerodynamics [39:44]
        obs[39] = np.clip(v_tas / 18.0, 0.0, 1.0)
        obs[40] = np.clip(q_dyn / 120.0, 0.0, 1.0)
        obs[41:44] = np.clip(wind_est / 8.0, -1.0, 1.0)

        # 6. Battery & Winch States [44:50]
        obs[44] = np.clip(bms_soc, 0.0, 1.0)
        obs[45] = np.clip((v_pack - 18.0) / (25.2 - 18.0), 0.0, 1.0)
        obs[46] = np.clip(cable_len / 10.0, 0.0, 1.0)
        obs[47] = np.clip(cable_vel / 3.4, -1.0, 1.0)
        obs[48] = np.clip(tension_n / 50.0, 0.0, 1.0)
        obs[49] = 1.0 if has_payload else 0.0

        # 7. Navigation Integrity & Attack Flags [50:53]
        obs[50] = np.clip(chi2_val / 40.0, 0.0, 1.0)
        obs[51] = 1.0 if is_spoofed else 0.0
        obs[52] = np.clip(jerk_discrepancy / 15.0, 0.0, 1.0)

        # 8. Manifest Logistics Progress [53:56]
        obs[53] = np.clip(flight_time / 120.0, 0.0, 1.0)
        obs[54] = np.clip((completed_count % 10) / 10.0, 0.0, 1.0)
        obs[55] = 1.0 if dist_goal < HOVER_LOCK_RADIUS_M else 0.0

        return obs

# ==========================================
# 14. PRE-TRAINING / DATASET HARVESTER
# ==========================================
def _worker_harvest_v9(num_steps):
    np.random.seed()
    random.seed()
    world = SuburbanDistrictMacroWorld()
    synth = MacroObservationSynthesizer()
    lidar_engine = MacroPerceptionLiDARSuite(world)

    observations, teacher_guidance, teacher_fsm = [], [], []

    while len(observations) < num_steps:
        drone_p = np.array([random.uniform(-50, 50), random.uniform(-50, 50), random.uniform(2, 8)], dtype=np.float32)
        target_p = np.array([random.uniform(-50, 50), random.uniform(-50, 50), 2.2], dtype=np.float32)
        drone_v = np.zeros(3, dtype=np.float32)
        dt = 0.016
        wind = np.array([random.uniform(-4, 4), random.uniform(-4, 4), 0.0], dtype=np.float32)

        for _ in range(120):
            delta = target_p - drone_p
            d_horiz = float(np.linalg.norm(delta[:2]))
            fsm_token = 0 if d_horiz > 0.85 else 1

            v_des = (delta / max(1e-4, float(np.linalg.norm(delta)))) * 3.5 if fsm_token == 0 else np.zeros(3)
            F_d = -np.array([6.5, 6.5, 11.5]) * (drone_p - target_p) - np.array([4.4, 4.4, 7.0]) * (drone_v - v_des) + MASS_KG * (-GRAVITY_G)
            guidance_vec = np.clip(F_d / 25.0, -1.0, 1.0).astype(np.float32)

            lidar_16 = lidar_engine.cast_360_lidar(drone_p, 0.0)
            conical_8, _ = lidar_engine.cast_downward_conical(drone_p)

            obs_t = synth.synthesize(
                p_est=drone_p, v_est=drone_v, R_est=np.eye(3, dtype=np.float32), omega_est=np.zeros(3),
                target_p=target_p, lidar_16=lidar_16, conical_8=conical_8, v_tas=float(np.linalg.norm(drone_v - wind)),
                q_dyn=12.0, wind_est=wind, bms_soc=0.88, v_pack=23.5, cable_len=0.08, cable_vel=0.0,
                tension_n=PAYLOAD_WEIGHT_N, has_payload=True, chi2_val=1.2, is_spoofed=False,
                jerk_discrepancy=0.5, flight_time=15.0, completed_count=1, dt=dt
            )

            observations.append(obs_t)
            teacher_guidance.append(np.array([guidance_vec[0], guidance_vec[1], guidance_vec[2], 0.0], dtype=np.float32))
            teacher_fsm.append(fsm_token)

            drone_p += drone_v * dt
            drone_v += (F_d / MASS_KG) * dt
            if len(observations) >= num_steps: break

    return np.array(observations, dtype=np.float32), np.array(teacher_guidance, dtype=np.float32), np.array(teacher_fsm, dtype=np.int64)

def harvest_and_train_v9():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = 40000
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per = num_samples // num_workers

    print(f"\n[NEURAL COMPILER] Generating {num_samples} trajectories across {num_workers} parallel workers...")
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_harvest_v9, [samples_per] * num_workers)

    X_all = np.concatenate([r[0] for r in results], axis=0)[:num_samples]
    y_g_all = np.concatenate([r[1] for r in results], axis=0)[:num_samples]
    y_f_all = np.concatenate([r[2] for r in results], axis=0)[:num_samples]

    model = MacroScaleFlightExecutiveNet().to(device)
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
    print(f"[NEURAL COMPILER] V9.0 Model compiled & saved to {MODEL_WEIGHTS_FILE}!\n")
    return model, device

# ==========================================
# 15. DYNAMIC CHASE CAMERA & TACTICAL MINIMAP
# ==========================================
def project_3d_chase(pt, cam_focus, cam_yaw, cam_pitch, center_screen, scale=28.0):
    rel = pt - cam_focus
    x1 = rel[0]*math.cos(cam_yaw) - rel[1]*math.sin(cam_yaw)
    y1 = rel[0]*math.sin(cam_yaw) + rel[1]*math.cos(cam_yaw)
    y2 = y1*math.cos(cam_pitch) - rel[2]*math.sin(cam_pitch)
    z2 = y1*math.sin(cam_pitch) + rel[2]*math.cos(cam_pitch)
    focal = max(3.0, 18.0 - y2)
    sx = int(center_screen[0] + (x1 / focal) * scale * 9.5)
    sy = int(center_screen[1] - (z2 / focal) * scale * 9.5)
    return sx, sy, y2

def draw_tactical_radar_minimap(screen, map_rect, world: SuburbanDistrictMacroWorld, drone_p, drone_yaw, manifest_idx, drone_trail):
    rx, ry, rw, rh = map_rect
    pygame.draw.rect(screen, (15, 20, 28), map_rect)
    pygame.draw.rect(screen, (0, 220, 255), map_rect, 2)

    # Scale: 120m maps to rw pixels
    scale = rw / 120.0
    cx, cy = rx + rw // 2, ry + rh // 2

    def to_map(x, y):
        mx = int(cx + x * scale)
        my = int(cy - y * scale)
        return mx, my

    # Roads
    for r in world.roads:
        p1 = to_map(r["x_min"], r["y_min"])
        p2 = to_map(r["x_max"], r["y_max"])
        pygame.draw.rect(screen, (32, 38, 48), (min(p1[0], p2[0]), min(p1[1], p2[1]), abs(p2[0]-p1[0])+1, abs(p2[1]-p1[1])+1))

    # Buildings
    for h in world.houses:
        x1, x2, y1, y2, _ = h["aabb"]
        mp1 = to_map(x1, y2)
        mp2 = to_map(x2, y1)
        pygame.draw.rect(screen, (60, 72, 88), (mp1[0], mp1[1], mp2[0]-mp1[0], mp2[1]-mp1[1]))

    # Base Station Helipad [H]
    bx, by = to_map(world.base_station[0], world.base_station[1])
    pygame.draw.circle(screen, (0, 240, 255), (bx, by), 5, 1)

    # Waypoints
    for idx, order in enumerate(world.delivery_manifest):
        wx, wy = to_map(order["pos"][0], order["pos"][1])
        w_col = (0, 255, 140) if idx == manifest_idx else ((100, 130, 150) if idx > manifest_idx else (40, 160, 80))
        pygame.draw.circle(screen, w_col, (wx, wy), 4)

    # Live Trail
    if len(drone_trail) > 1:
        map_trail = [to_map(p[0], p[1]) for p in drone_trail]
        pygame.draw.lines(screen, (0, 180, 220), False, map_trail, 1)

    # Drone blip
    dx, dy = to_map(drone_p[0], drone_p[1])
    pygame.draw.circle(screen, (255, 255, 255), (dx, dy), 4)
    hx = int(dx + math.cos(drone_yaw) * 8.0)
    hy = int(dy - math.sin(drone_yaw) * 8.0)
    pygame.draw.line(screen, (255, 200, 40), (dx, dy), (hx, hy), 2)

# ==========================================
# 16. MAIN SIMULATOR & COCKPIT HUD
# ==========================================
def run_all_physics_simulator():
    if not os.path.exists(MODEL_WEIGHTS_FILE):
        model, device = harvest_and_train_v9()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MacroScaleFlightExecutiveNet().to(device)
        try:
            model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE, map_location=device))
        except Exception:
            model, device = harvest_and_train_v9()

    model.eval()
    pygame.init()
    VIEW_W, VIEW_H = 760, 740
    SIDEBAR_W = 440
    W, H = VIEW_W + SIDEBAR_W, VIEW_H

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Survival Core V9.0 — 120m Macro Suburban District & Tactical Minimap")
    clock = pygame.time.Clock()

    font_large = pygame.font.SysFont("consolas", 17, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 12, bold=True)

    # Macro World Setup
    world = SuburbanDistrictMacroWorld(sector_span_m=120.0, cell_size_m=10.0)
    init_coords = np.array([-42.0, -38.0, 1.8], dtype=np.float32)
    home_coords = np.array([-42.0, -38.0, 2.2], dtype=np.float32)

    drone = BEMTCoaxialAirframe8DOF(init_coords)
    ekf = ErrorStateEKF15(init_coords)
    quantum_sensor = MatterWaveColdAtomQuantumSensor()
    bms = SubZeroArrheniusBatteryECM()
    winch = BilateralCoupledWinchSystem()
    synth = MacroObservationSynthesizer()
    mixer = RateDampedActuatorMixerWithCBF()

    lidar_suite = MacroPerceptionLiDARSuite(world)
    building_wake = MacroBuildingWakeEngine(world)
    ground_agent = MacroGroundIntruderAgent(init_pos=[-18.0, -8.0, 0.0])
    icing_model = MessingerRotorIcingModel()
    dryden_wind_gen = DrydenAtmosphericTurbulenceModel(w20_m_s=3.4)

    # Manifest Routing
    manifest_idx = 0
    active_order = world.delivery_manifest[manifest_idx]
    target_delivery_point = active_order["pos"].copy()

    cam_yaw, cam_pitch = 0.70, 0.45
    center_viewport = (VIEW_W // 2, VIEW_H // 2 + 50)
    running = True

    base_wind = np.array([2.2, -1.0, 0.0], dtype=np.float32)
    gust_timer = 0.0
    active_gust = np.zeros(3, dtype=np.float32)

    inject_spoof = False
    inject_multipath = False
    hover_settle_timer = 0.0
    packages_delivered_count = 0
    delivered_boxes = []
    flight_time = 0.0
    drone_trail = deque(maxlen=150)

    action_probs = np.zeros(NUM_EXECUTIVE_STATES, dtype=np.float32)
    current_action_name = "TRANSIT"
    policy_latency_ms = 1.2
    confidence = 0.95

    # Static Obstacles for CBF
    obstacles = [
        {"pos": np.array([-18.0,  5.0, 0.0]), "radius": 2.2},
        {"pos": np.array([  2.0,  8.0, 0.0]), "radius": 2.5},
        {"pos": np.array([ 25.0, -22.0, 0.0]), "radius": 2.0}
    ]

    wind_particles = [
        np.array([random.uniform(-50.0, 50.0), random.uniform(-50.0, 50.0), random.uniform(1.0, 10.0)], dtype=np.float32)
        for _ in range(50)
    ]

    print("\n================ V9.0 MACRO SUBURBAN DISTRICT RUNNING ================")
    print(" [WORLD]    : 120m x 120m Neighborhood with 10 Houses, Roads & Power Grid")
    print(" [SENSORS]  : 16-Beam 360° Planar LiDAR + 8-Beam Downward Conical Array")
    print(" [LOGISTICS]: Multi-Address Delivery Manifest with Automated Depot RTB")
    print(" [COCKPIT]  : Dynamic 3D Chase Camera + Tactical Top-Down Radar Mini-Map")
    print(" [HOTKEYS]  : [A] Leash Agent | [J] GPS Spoof | [M] Multipath | [I] Icing")
    print("              [C] Cold-Soak (-5°C) | [H] Return-to-Base Depot | [1-8] Motor Kill")
    print("======================================================================\n")

    while running:
        dt = 0.016
        flight_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if pygame.K_1 <= event.key <= pygame.K_8:
                    m_idx = event.key - pygame.K_1
                    drone.motor_health[m_idx] = 0.0 if drone.motor_health[m_idx] == 1.0 else 1.0
                elif event.key == pygame.K_a:
                    ground_agent.is_leashed = not ground_agent.is_leashed
                    print(f">>> GROUND AGENT: {'LEASHED' if ground_agent.is_leashed else 'ROAMING'} <<<")
                elif event.key == pygame.K_h:
                    target_delivery_point = home_coords.copy()
                    hover_settle_timer = 0.0
                    mixer.e_p_integral.fill(0.0)
                    print(">>> OPERATOR DIRECTIVE: IMMEDIATE RETURN TO BASE DEPOT [H] <<<")
                elif event.key == pygame.K_b:
                    bms.soc = 0.18
                    print(">>> LOW BATTERY WARNING (18% SOC) -> RTH INITIATED <<<")
                elif event.key == pygame.K_c:
                    bms.temp_c = -5.0
                    print(">>> COLD-SOAK (-5°C) TRIGGERED: ARRHENIUS VOLTAGE SAG <<<")
                elif event.key == pygame.K_i:
                    icing_model.icing_active = not icing_model.icing_active
                elif event.key == pygame.K_m:
                    inject_multipath = not inject_multipath
                elif event.key == pygame.K_j:
                    inject_spoof = not inject_spoof
                elif event.key == pygame.K_g:
                    gust_timer = 2.5
                    ang = random.uniform(0, 2*math.pi)
                    active_gust = np.array([math.cos(ang)*8.5, math.sin(ang)*8.5, 0.0], dtype=np.float32)
                elif event.key == pygame.K_p:
                    drone.parachute_deployed = True
                elif event.key == pygame.K_r:
                    drone.motor_health.fill(1.0)
                    drone.parachute_deployed = False
                    bms.soc = 0.96
                    bms.temp_c = 22.0
                    icing_model.ice_thickness_mm = 0.0
                    inject_spoof = False
                    inject_multipath = False
                    winch.reload_new_cargo()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: cam_yaw -= 0.03
        if keys[pygame.K_RIGHT]: cam_yaw += 0.03
        if keys[pygame.K_UP]: cam_pitch = min(1.2, cam_pitch + 0.02)
        if keys[pygame.K_DOWN]: cam_pitch = max(0.15, cam_pitch - 0.02)

        # 1. Total Wind Field with Macro Rooftop Cavity Wakes
        dryden_gust = dryden_wind_gen.step(altitude_m=drone.p[2], airspeed_m_s=np.linalg.norm(drone.v), dt=dt)
        microburst = active_gust * (gust_timer / 2.5) if gust_timer > 0.0 else np.zeros(3, dtype=np.float32)
        if gust_timer > 0.0: gust_timer = max(0.0, gust_timer - dt)

        ambient_wind = base_wind + dryden_gust + microburst
        rooftop_wake = building_wake.evaluate_wake(drone.p, ambient_wind)
        total_wind = ambient_wind + rooftop_wake

        for wp in wind_particles:
            wp += total_wind * 0.45 * dt
            if wp[0] > 60.0: wp[0] = -60.0
            if wp[0] < -60.0: wp[0] = 60.0
            if wp[1] > 60.0: wp[1] = -60.0
            if wp[1] < -60.0: wp[1] = 60.0

        # 2. Icing & Ground Agent Step
        ice_mm, ice_idx = icing_model.step(airspeed=float(np.linalg.norm(drone.v - total_wind)), dt=dt)

        payload_tip_world = drone.p + np.array([0.0, 0.0, -0.08]) + np.array([0.0, 0.0, -1.0]) * winch.cable_len
        is_aborting = (winch.state == "ABORT_RETRACT")
        agent_pos, agent_breach, agent_dist = ground_agent.step(
            cargo_pos=payload_tip_world,
            is_winch_lowering=(winch.state == "LOWERING"),
            is_aborting=is_aborting,
            dt=dt
        )

        # 3. 360° LiDAR & Downward Conical Raycasting
        drone_yaw = math.atan2(drone.R[1, 0], drone.R[0, 0])
        lidar_16 = lidar_suite.cast_360_lidar(drone.p, drone_yaw)
        conical_8, wire_clearance = lidar_suite.cast_downward_conical(drone.p)
        wire_hit = (wire_clearance < 1.4)

        # 4. Quantum Sensing & SO(3) ES-EKF
        accel_inertial_true = (drone.accel_linear - GRAVITY_G)
        a_quantum, phase_shift = quantum_sensor.measure_phase_shift(accel_inertial_true)

        gyro_raw = drone.omega + np.random.normal(0, 0.005, 3).astype(np.float32)
        ekf.predict(drone.R.T @ a_quantum, gyro_raw, dt=dt)

        H_baro = np.zeros((1, 15), dtype=np.float32); H_baro[0, 2] = 1.0
        ekf.update_measurement(H_baro, np.array([drone.p[2] + np.random.normal(0, 0.04) - ekf.p[2]], dtype=np.float32), np.array([[0.05]], dtype=np.float32))

        # 5. Anti-Spoof Kinematic & Downward VIO Fallback
        multipath_offset = np.array([4.2, -3.1, 0.8], dtype=np.float32) if inject_multipath else np.zeros(3)
        spoof_offset = np.array([3.5, 2.8, 0.5]) if inject_spoof else np.zeros(3)
        z_gps = drone.p + multipath_offset + spoof_offset + np.random.normal(0, 0.1, 3).astype(np.float32)

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

        H_pos = np.zeros((3, 15), dtype=np.float32); H_pos[0:3, 0:3] = np.eye(3)
        y_innov = z_gps - ekf.p
        S_cov = H_pos @ ekf.P @ H_pos.T + np.eye(3, dtype=np.float32) * 0.08
        chi2_dist = float(y_innov.T @ np.linalg.inv(S_cov) @ y_innov)

        is_attack_detected = (chi2_dist > 16.27) or kinematic_violation or inject_spoof or inject_multipath

        if not is_attack_detected:
            ekf.update_measurement(H_pos, y_innov, np.eye(3, dtype=np.float32) * 0.05)
            gnss_status = "GNSS NOMINAL"
        else:
            cam_optical_drift = np.random.normal(0, 0.015, 3).astype(np.float32)
            p_vio_fused = drone.p + cam_optical_drift
            y_vio = p_vio_fused - ekf.p
            ekf.update_measurement(H_pos, y_vio, np.eye(3, dtype=np.float32) * 0.02)
            gnss_status = "SPOOF REJECTED: QUANTUM VIO LOCK"

        # 6. Battery ECM
        bms_soc, v_pack, i_draw, r0_dyn = bms.step(drone.step.__dict__.get("last_p_watts", 185.0), dt=dt)

        # 7. Navigation Routing & Manifest Management
        R_est = quat_to_rot(ekf.q)
        omega_est = gyro_raw - ekf.b_g

        active_target = home_coords if (bms_soc <= 0.20 or manifest_idx >= len(world.delivery_manifest)) else target_delivery_point
        d_horiz_est = float(np.linalg.norm(ekf.p[:2] - active_target[:2]))
        v_horiz_est = float(np.linalg.norm(ekf.v[:2]))

        is_centered = (d_horiz_est <= HOVER_LOCK_RADIUS_M) and (v_horiz_est <= HOVER_LOCK_SPEED_MAX_MS)
        if is_centered and winch.has_payload: hover_settle_timer += dt
        else:
            if winch.cable_len <= 0.10: hover_settle_timer = 0.0

        is_settled_for_winch = (hover_settle_timer >= HOVER_SETTLE_DURATION_S)

        cable_l, tension, f_cable_world, u_cable = winch.step(
            drone_pos=drone.p, drone_accel=drone.accel_linear,
            is_settled=is_settled_for_winch, is_surface_safe=True, wire_hit=wire_hit,
            agent_breach=agent_breach, agent_dist=agent_dist, wind_vec=total_wind, dt=dt
        )

        # Auto-Dispatch to Next Address in Manifest
        if winch.delivery_completed:
            packages_delivered_count += 1
            delivered_boxes.append(target_delivery_point.copy())
            manifest_idx += 1
            if manifest_idx < len(world.delivery_manifest):
                active_order = world.delivery_manifest[manifest_idx]
                target_delivery_point = active_order["pos"].copy()
                winch.reload_new_cargo()
                hover_settle_timer = 0.0
                mixer.e_p_integral.fill(0.0)
                print(f">>> [DISPATCH] PACKAGE DELIVERED! ROUTING TO: {active_order['address']} <<<")
            else:
                target_delivery_point = home_coords.copy()
                print(">>> [MANIFEST COMPLETE] ALL PACKAGES DELIVERED. RETURNING TO BASE DEPOT [H] <<<")

        # 8. Synthesize 56-Dimensional Observation & Forward Policy
        obs_vec = synth.synthesize(
            p_est=ekf.p, v_est=ekf.v, R_est=R_est, omega_est=omega_est, target_p=active_target,
            lidar_16=lidar_16, conical_8=conical_8, v_tas=float(np.linalg.norm(drone.v - total_wind)),
            q_dyn=12.0, wind_est=total_wind, bms_soc=bms_soc, v_pack=v_pack, cable_len=winch.cable_len,
            cable_vel=winch.cable_vel, tension_n=tension, has_payload=winch.has_payload,
            chi2_val=chi2_dist, is_spoofed=is_attack_detected, jerk_discrepancy=jerk_discrepancy,
            flight_time=flight_time, completed_count=packages_delivered_count, dt=dt
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

        # Record trail
        if int(flight_time * 60) % 6 == 0:
            drone_trail.append(drone.p[:2].copy())

        # ==========================================
        # RENDERING (DYNAMIC CHASE CAM & RADAR)
        # ==========================================
        screen.fill((10, 13, 20))
        pygame.draw.line(screen, (35, 45, 60), (VIEW_W, 0), (VIEW_W, H), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (W, 70), 2)

        cam_focus = drone.p.copy()

        # 1. Ground Grid Lines around drone
        for gx in range(-5, 6):
            p1 = np.array([drone.p[0] + gx * 4.0, drone.p[1] - 25.0, 0.0], dtype=np.float32)
            p2 = np.array([drone.p[0] + gx * 4.0, drone.p[1] + 25.0, 0.0], dtype=np.float32)
            sx1, sy1, _ = project_3d_chase(p1, cam_focus, cam_yaw, cam_pitch, center_viewport)
            sx2, sy2, _ = project_3d_chase(p2, cam_focus, cam_yaw, cam_pitch, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        # 2. Roads
        for r in world.roads:
            if math.hypot(drone.p[0] - (r["x_min"]+r["x_max"])/2, drone.p[1] - (r["y_min"]+r["y_max"])/2) < 45.0:
                rc_pts = [
                    np.array([r["x_min"], r["y_min"], 0.0]), np.array([r["x_max"], r["y_min"], 0.0]),
                    np.array([r["x_max"], r["y_max"], 0.0]), np.array([r["x_min"], r["y_max"], 0.0])
                ]
                r_screens = [project_3d_chase(p, cam_focus, cam_yaw, cam_pitch, center_viewport)[:2] for p in rc_pts]
                pygame.draw.polygon(screen, (28, 34, 42), r_screens)

        # 3. Houses with Frustum Culling
        for h in world.houses:
            hx1, hx2, hy1, hy2, hh = h["aabb"]
            h_center = np.array([(hx1+hx2)/2, (hy1+hy2)/2, 0.0])
            if np.linalg.norm(drone.p[:2] - h_center[:2]) < 38.0:
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

        # 4. Base Station Helipad [H]
        if np.linalg.norm(drone.p[:2] - world.base_station[:2]) < 45.0:
            hc_sx, hc_sy, _ = project_3d_chase(world.base_station, cam_focus, cam_yaw, cam_pitch, center_viewport)
            pygame.draw.circle(screen, (0, 220, 255), (hc_sx, hc_sy), 22, 2)
            screen.blit(font_large.render("H", True, (0, 240, 255)), (hc_sx - 6, hc_sy - 10))

        # 5. Overhead Catenary Power Lines
        for line in world.powerlines:
            w1_s = project_3d_chase(line["p1"], cam_focus, cam_yaw, cam_pitch, center_viewport)[:2]
            w2_s = project_3d_chase(line["p2"], cam_focus, cam_yaw, cam_pitch, center_viewport)[:2]
            pygame.draw.line(screen, (255, 180, 40), w1_s, w2_s, 2)

        # 6. Active Waypoint Dropzone Ring
        tsx, tsy, _ = project_3d_chase(target_delivery_point, cam_focus, cam_yaw, cam_pitch, center_viewport)
        pygame.draw.circle(screen, (0, 255, 140), (tsx, tsy), 15, 2)
        screen.blit(font_small.render(f"WAYPOINT: {active_order['address']}", True, (0, 255, 140)), (tsx + 18, tsy - 6))

        # 7. Ground Agent
        if np.linalg.norm(drone.p[:2] - agent_pos[:2]) < 35.0:
            ag_sx, ag_sy, _ = project_3d_chase(agent_pos, cam_focus, cam_yaw, cam_pitch, center_viewport)
            pygame.draw.circle(screen, (255, 120, 40), (ag_sx, ag_sy), 7)
            screen.blit(font_small.render("GROUND AGENT", True, (255, 140, 40)), (ag_sx + 10, ag_sy - 6))

        # 8. Diffuse Ground Shadow & Airframe
        gsx, gsy, _ = project_3d_chase(np.array([drone.p[0], drone.p[1], 0.0]), cam_focus, cam_yaw, cam_pitch, center_viewport)
        shadow_r = int(max(6, 16 - drone.p[2] * 1.5))
        pygame.draw.ellipse(screen, (22, 32, 44), (gsx - shadow_r, gsy - shadow_r // 2, shadow_r * 2, shadow_r))

        dsx, dsy, _ = project_3d_chase(drone.p, cam_focus, cam_yaw, cam_pitch, center_viewport)
        pygame.draw.circle(screen, (0, 240, 255), (dsx, dsy), 8)

        # Arms & Counter-Rotating Rotors
        arm_d = ARM_LENGTH_M * 2.2
        arm_ends_body = [
            np.array([ arm_d/1.414, -arm_d/1.414, 0.0]), np.array([ arm_d/1.414,  arm_d/1.414, 0.0]),
            np.array([-arm_d/1.414,  arm_d/1.414, 0.0]), np.array([-arm_d/1.414, -arm_d/1.414, 0.0]),
        ]
        arm_screens = [project_3d_chase(drone.p + drone.R @ a, cam_focus, cam_yaw, cam_pitch, center_viewport)[:2] for a in arm_ends_body]
        pygame.draw.line(screen, (160, 170, 185), arm_screens[0], arm_screens[2], 3)
        pygame.draw.line(screen, (160, 170, 185), arm_screens[1], arm_screens[3], 3)

        blade_spin = time.time() * 25.0
        for m_i in range(8):
            a_i = [0, 0, 1, 1, 2, 2, 3, 3][m_i]
            mx, my = arm_screens[a_i]
            off_y = -6 if (m_i % 2 == 0) else 6
            if drone.motor_health[m_i] == 0.0:
                pygame.draw.line(screen, (255, 60, 80), (mx - 5, my + off_y - 5), (mx + 5, my + off_y + 5), 2)
            else:
                m_col = (0, 240, 255) if (m_i % 2 == 0) else (0, 255, 140)
                pygame.draw.circle(screen, m_col, (mx, my + off_y), 9, 1)

        # 9. Tether Cable & Parcel
        payload_world = drone.p + np.array([0.0, 0.0, -0.08]) + u_cable * winch.cable_len
        psx, psy, _ = project_3d_chase(payload_world, cam_focus, cam_yaw, cam_pitch, center_viewport)

        if not winch.severed:
            if winch.cable_len > 0.10:
                pygame.draw.line(screen, (230, 240, 255), (dsx, dsy + 4), (psx, psy), 2)
            if winch.has_payload:
                box_x, box_y = (dsx, dsy + 5) if winch.cable_len <= 0.10 else (psx, psy)
                pygame.draw.rect(screen, (255, 195, 35), (box_x - 7, box_y - 7, 14, 14), border_radius=2)
                screen.blit(font_small.render("0.45kg", True, (255, 215, 0)), (box_x + 9, box_y - 6))

        # ==========================================
        # COMPLETE SIDEBAR DASHBOARD & RADAR MINIMAP
        # ==========================================
        sb_x = VIEW_W + 14
        screen.blit(font_large.render("SURVIVAL CORE V9.0", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("120m Macro Suburban District & Tactical Minimap", True, (130, 160, 190)), (sb_x, 38))

        # Tactical Radar Mini-Map
        map_rect = (sb_x, 65, 210, 210)
        draw_tactical_radar_minimap(screen, map_rect, world, drone.p, drone_yaw, manifest_idx, drone_trail)

        # Manifest Status Panel
        mf_x = sb_x + 224
        screen.blit(font_mono.render("DELIVERY MANIFEST:", True, (240, 240, 240)), (mf_x, 65))
        for m_idx, order in enumerate(world.delivery_manifest[:5]):
            m_col = (0, 255, 140) if m_idx == manifest_idx else ((140, 160, 180) if m_idx > manifest_idx else (80, 120, 90))
            tag = "[ACTIVE]" if m_idx == manifest_idx else ("[DONE]" if m_idx < manifest_idx else "[QUEUED]")
            screen.blit(font_small.render(f"{tag} {order['address']}", True, m_col), (mf_x, 85 + m_idx * 16))

        # 1. Macro Logistics Telemetry
        p1_y = 285
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p1_y - 8), (W - 14, p1_y - 8), 1)
        screen.blit(font_mono.render("1. SUBURBAN LOGISTICS & DEPOT:", True, (240, 240, 240)), (sb_x, p1_y))
        dist_dest = float(np.linalg.norm(drone.p[:2] - active_target[:2]))
        screen.blit(font_small.render(f"Target Waypoint:   {active_order['address']} ({dist_dest:.1f}m)", True, (0, 220, 255)), (sb_x + 10, p1_y + 18))
        screen.blit(font_small.render(f"Airframe Mass:     {current_mass_active:.2f} kg ({'LOADED' if winch.has_payload else 'DRY SHED'})", True, (0, 255, 140) if winch.has_payload else (255, 200, 40)), (sb_x + 10, p1_y + 34))
        screen.blit(font_small.render(f"Manifest Deposited:{packages_delivered_count} / {len(world.delivery_manifest)} Addresses Completed", True, (0, 255, 140)), (sb_x + 10, p1_y + 50))

        # 2. 16-Beam LiDAR & Perception
        p2_y = p1_y + 68
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (W - 14, p2_y), 1)
        screen.blit(font_mono.render("2. 56-DIMENSIONAL PERCEPTION & LIDAR:", True, (240, 240, 240)), (sb_x, p2_y + 8))
        min_lidar = float(np.min(lidar_16))
        screen.blit(font_small.render(f"16-Beam Planar LiDAR:Min Clearance = {min_lidar:.1f} m", True, (255, 60, 80) if min_lidar < 3.0 else (0, 220, 255)), (sb_x + 10, p2_y + 26))
        screen.blit(font_small.render(f"Downward Conical:  Ground Depth = {conical_8[0]:.1f} m | Wire Clr = {wire_clearance:.1f} m", True, (160, 180, 205)), (sb_x + 10, p2_y + 42))

        # 3. Sub-Zero Battery & Navigation
        p3_y = p2_y + 60
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (W - 14, p3_y), 1)
        screen.blit(font_mono.render("3. BATTERY & ANTI-SPOOF NAVIGATION:", True, (240, 240, 240)), (sb_x, p3_y + 8))
        t_col = (255, 60, 80) if bms.temp_c < 0 else (0, 255, 140)
        screen.blit(font_small.render(f"Battery Core Temp: T = {bms.temp_c:.1f}°C (Arrhenius R0 = {r0_dyn*1000:.1f} mΩ)", True, t_col), (sb_x + 10, p3_y + 26))
        screen.blit(font_small.render(f"Navigation State:  {gnss_status}", True, (255, 60, 80) if "REJECTED" in gnss_status else (0, 255, 140)), (sb_x + 10, p3_y + 42))

        # 4. Aerodynamics & Weather
        p4_y = p3_y + 60
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (W - 14, p4_y), 1)
        screen.blit(font_mono.render("4. BEMT PROPULSION & ATMOSPHERE:", True, (240, 240, 240)), (sb_x, p4_y + 8))
        screen.blit(font_small.render(f"Rotor Icing:       {ice_mm:.2f} mm | Thrust: F_z = {f_thrust:.1f} N", True, (255, 60, 80) if ice_mm > 0.5 else (0, 220, 255)), (sb_x + 10, p4_y + 26))
        screen.blit(font_small.render(f"District Wind:     |V_wind| = {np.linalg.norm(total_wind):.2f} m/s (Dryden + Wake)", True, (160, 180, 205)), (sb_x + 10, p4_y + 42))

        # 5. Executive Winch
        p5_y = p4_y + 60
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p5_y), (W - 14, p5_y), 1)
        screen.blit(font_mono.render("5. BILATERAL WINCH DELIVERY:", True, (240, 240, 240)), (sb_x, p5_y + 8))
        screen.blit(font_small.render(f"Winch State:       {winch.state} (Cable: {winch.cable_len:.2f} m | Tension: {tension:.1f} N)", True, (0, 220, 255)), (sb_x + 10, p5_y + 26))
        screen.blit(font_small.render(f"Ground Agent:      {'LEASHED [A]' if ground_agent.is_leashed else f'RANGE = {agent_dist:.1f} m'}", True, (0, 255, 140) if ground_agent.is_leashed else (255, 120, 40)), (sb_x + 10, p5_y + 42))

        # Bottom Command Strip
        bot_y = H - 38
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y - 8), (W - 14, bot_y - 8), 1)
        screen.blit(font_mono.render("KEYS: [A] Leash | [H] RTB Depot | [J] GPS Spoof | [Arrows] Cam Orbit", True, (255, 200, 40)), (sb_x + 8, bot_y))

        # Viewport Header
        head_txt = font_large.render(f"Master Survival V9.0 | Manifest: {manifest_idx+1}/{len(world.delivery_manifest)} | Target: {active_order['address']}", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("120m Suburban District • 16-Beam LiDAR • Tactical Radar Minimap • Anti-Spoof VIO Lock", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    mp.freeze_support()
    run_all_physics_simulator()