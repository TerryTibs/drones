# =============================================================================
# MASTER FRONTIER AUTONOMOUS DRONE SYSTEM CORE (V35.2 - TRAINING STABILIZED)
# File: drone_master_autonomous_system_v33.py — [PART 1 OF 3]
# =============================================================================
# - TRAINING FIX 1: Auto-Padded SwarmFleetCrossAttention (Strict 3-token length)
# - TRAINING FIX 2: Gradient Norm Clipping (max_norm=1.0 stops Epoch 5 blowup)
# - TRAINING FIX 3: Inverse Class-Weighted Cross-Entropy (Solves 73.5% collapse)
# - TRAINING FIX 4: Stabilized Hyperparameters (BATCH_SIZE=1026, LR=1.2e-3)
# - HARVESTER FIX: Full Round-Trip Dataset (Outbound -> Drop -> Climb -> RTB -> Land)
# - AUTONOMY FIX: Lookahead "Carrot-Chasing" Path Follower (Zero Mid-City Diving)
# - AUTONOMY FIX: Strict Authority Separation (Spline = Altitude, Net = Horizontal)
# - AUTONOMY FIX: Rooftop Departure Masking in CBF (Zero Horizontal Kick on RTB)
# - AUTONOMY FIX: 8-Stage Mission FSM with Physical Depot Helipad Touchdown
# - OPTION A: 122-Feature Observation with Peer Swarm Relative Kinematics (366-Dim)
# - OPTION B: Multi-Agent Swarm Cross-Attention Transformer Layer (Fleet Reshaping)
# - OPTION C: Altitude-Stratified Corridors (30m / 35m / 40m) & Terminal Pad Queuing
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

C_T_CLEAN = 1.35e-5
C_Q_CLEAN = 2.40e-7
COAXIAL_EFFICIENCY = 0.92
MOTOR_TAU_S = 0.030
MAX_RPM = 9200.0
MAX_OMEGA = (MAX_RPM * 2.0 * math.pi) / 60.0

HOVER_THRUST_NOMINAL = MASS_KG * G_ACCEL
HOVER_OMEGA_NOMINAL = math.sqrt((HOVER_THRUST_NOMINAL / 8.0) / (C_T_CLEAN * 0.95))

NUM_BLADE_ELEMENTS = 8
PRANDTL_TIP_FACTOR = 0.92
COAXIAL_DOWNWASH_INGESTION = 0.65

CEILING_HEIGHT_M = 52.0
CBF_ALPHA_OBSTACLE = 1.8
CBF_SAFE_RADIUS_M = 0.85
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
HOVER_LOCK_RADIUS_M = 1.40
HOVER_LOCK_SPEED_MAX_MS = 0.95
HOVER_SETTLE_DURATION_S = 0.25

# Multi-Agent Swarm Feature Dimensions (122 base features x 3-frame history = 366)
BASE_OBS_DIM = 122
HISTORY_LEN = 3
OBSERVATION_DIM = BASE_OBS_DIM * HISTORY_LEN
NUM_EXECUTIVE_STATES = 6
MODEL_WEIGHTS_FILE = "drone_master_brain_v22_balanced.pth"

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
# 2. METROPOLITAN CITY GRID & ROOFTOP PADS
# ==========================================
class MetropolitanCityGridWorld400m:
    def __init__(self):
        self.avenue_x = [-120.0, -60.0, 0.0, 60.0, 120.0]
        self.street_y = [-120.0, -60.0, 0.0, 60.0, 120.0]
        self.street_width = 10.0

        self.roads = []
        for x in self.avenue_x:
            self.roads.append({"x_min": x - 5.0, "x_max": x + 5.0, "y_min": -200.0, "y_max": 200.0, "type": "AVENUE"})
        for y in self.street_y:
            self.roads.append({"x_min": -200.0, "x_max": 200.0, "y_min": y - 5.0, "y_max": y + 5.0, "type": "STREET"})

        self.depot_helipads = [
            np.array([-160.0, -155.0, 0.0], dtype=np.float32),
            np.array([-150.0, -155.0, 0.0], dtype=np.float32),
            np.array([-140.0, -155.0, 0.0], dtype=np.float32)
        ]
        self.base_station = self.depot_helipads[0]

        self.houses = []
        self.manifest_addresses = []
        self._build_48_buildings_with_rooftop_pads()

        self.trees = [
            {"pos": np.array([-25.0, -25.0, 0.0], dtype=np.float32), "r": 3.2, "h": 8.0},
            {"pos": np.array([ 25.0, -25.0, 0.0], dtype=np.float32), "r": 3.0, "h": 7.5},
            {"pos": np.array([-25.0,  25.0, 0.0], dtype=np.float32), "r": 3.5, "h": 8.5},
            {"pos": np.array([ 25.0,  25.0, 0.0], dtype=np.float32), "r": 3.2, "h": 8.0}
        ]

        self.powerlines = [
            {"p1": np.array([-190.0, -170.0, 12.0], dtype=np.float32), "p2": np.array([190.0, -170.0, 12.0], dtype=np.float32)},
            {"p1": np.array([-190.0,  170.0, 12.0], dtype=np.float32), "p2": np.array([190.0,  170.0, 12.0], dtype=np.float32)}
        ]

    def _build_48_buildings_with_rooftop_pads(self):
        street_names = ["Broadway", "5th Ave", "Park Ave", "Lexington", "Madison", "Elm St", "Market St", "Grand Blvd"]
        bldg_counter = 0

        col_candidates = [-165.0, -140.0, -90.0, -30.0, 30.0, 90.0, 140.0, 165.0]
        row_candidates = [-165.0, -140.0, -90.0, -30.0, 30.0, 90.0, 140.0, 165.0]

        for c_idx, cx in enumerate(col_candidates):
            for r_idx, ry in enumerate(row_candidates):
                if cx <= -130.0 and ry <= -130.0:
                    continue

                if bldg_counter >= 48:
                    break

                bldg_w = 18.0
                bldg_d = 18.0
                bx1 = cx - bldg_w / 2.0
                bx2 = cx + bldg_w / 2.0
                by1 = ry - bldg_d / 2.0
                by2 = ry + bldg_d / 2.0

                tier = (c_idx + r_idx) % 4
                if tier == 0:
                    height = 8.5
                elif tier == 1:
                    height = 14.5
                elif tier == 2:
                    height = 21.0
                else:
                    height = 28.0

                bldg_id = f"Bldg_{bldg_counter+1:02d}"
                self.houses.append({"id": bldg_id, "aabb": [bx1, bx2, by1, by2, height]})

                street = street_names[(c_idx + r_idx) % len(street_names)]
                num = (bldg_counter + 1) * 10 + 2

                drop_pos = np.array([cx, ry, height], dtype=np.float32)

                self.manifest_addresses.append({
                    "bldg_id": bldg_id,
                    "addr": f"{num} {street} [ROOFTOP]",
                    "pos": drop_pos,
                    "height": height
                })
                bldg_counter += 1

            if bldg_counter >= 48:
                break

    def get_surface_height(self, x, y):
        for h in self.houses:
            x1, x2, y1, y2, hh = h["aabb"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return float(hh)
        return 0.0

# =============================================================================
# 3. 3D GLIDE-SLOPE CORRIDOR & PROGRESS-AWARE QUINTIC SPLINE
# =============================================================================
class VoxelOctreeCorridorPlanner:
    def __init__(self, world: MetropolitanCityGridWorld400m):
        self.world = world

    def is_point_occluded(self, pt, margin=2.2):
        px, py, pz = pt
        if pz < 0.2 or pz > CEILING_HEIGHT_M:
            return True
        for h in self.world.houses:
            x1, x2, y1, y2, hh = h["aabb"]
            if (x1 - margin) <= px <= (x2 + margin) and (y1 - margin) <= py <= (y2 + margin) and pz < (hh - 0.2):
                return True
        for t in self.world.trees:
            if math.hypot(px - t["pos"][0], py - t["pos"][1]) <= (t["r"] + margin) and pz <= (t["h"] + margin):
                return True
        for line in self.world.powerlines:
            w1 = line["p1"]
            if abs(py - w1[1]) < (margin + 1.2) and abs(pz - w1[2]) < (margin + 1.2):
                return True
        return False

    def is_segment_occluded(self, p1, p2, step_size=4.0):
        dist = float(np.linalg.norm(p2 - p1))
        steps = max(2, int(dist / step_size))
        for s in range(steps + 1):
            tau = s / float(steps)
            interp_pt = p1 + tau * (p2 - p1)
            if self.is_point_occluded(interp_pt):
                return True
        return False

    def find_safe_corridor_waypoints(self, start_p, end_p, assigned_cruise_alt=34.0, is_depot_landing=False):
        cruise_alt = max(assigned_cruise_alt, max(start_p[2], end_p[2]) + 4.0)

        # Gentle vertical climbout gate above current surface
        local_start_h = self.world.get_surface_height(start_p[0], start_p[1])
        safe_climb_z = max(start_p[2] + 2.5, local_start_h + 3.5)
        p_climb_gate = np.array([start_p[0], start_p[1], safe_climb_z], dtype=np.float32)

        d_xy = end_p[:2] - start_p[:2]
        dist_total_xy = max(1.0, float(np.linalg.norm(d_xy)))
        dir_xy = d_xy / dist_total_xy

        climb_ramp_dist = min(35.0, dist_total_xy * 0.35)
        p_climb_top = np.array([
            start_p[0] + dir_xy[0] * climb_ramp_dist,
            start_p[1] + dir_xy[1] * climb_ramp_dist,
            cruise_alt
        ], dtype=np.float32)

        descent_ramp_dist = min(35.0, dist_total_xy * 0.35)
        p_descend_top = np.array([
            end_p[0] - dir_xy[0] * descent_ramp_dist,
            end_p[1] - dir_xy[1] * descent_ramp_dist,
            cruise_alt
        ], dtype=np.float32)

        flare_alt = end_p[2] + 2.5 if not is_depot_landing else 3.2
        p_flare = np.array([end_p[0], end_p[1], flare_alt], dtype=np.float32)

        if self.is_segment_occluded(p_climb_top, p_descend_top):
            mid_pt = 0.5 * (p_climb_top + p_descend_top)
            detour_1 = mid_pt + np.array([-20.0, 20.0, 5.0], dtype=np.float32)
            detour_2 = mid_pt + np.array([ 20.0, -20.0, 5.0], dtype=np.float32)
            via = detour_1 if not self.is_point_occluded(detour_1) else detour_2
            return [start_p.copy(), p_climb_gate, p_climb_top, via, p_descend_top, p_flare, end_p.copy()]
        else:
            return [start_p.copy(), p_climb_gate, p_climb_top, p_descend_top, p_flare, end_p.copy()]

# Analytical Quintic Spline with Mathematically Governed Derivative Speeds
class MinimumSnapQuinticTrajectory:
    def __init__(self, waypoints, avg_speed=5.0):
        self.waypoints = [np.array(wp, dtype=np.float32) for wp in waypoints]
        self.num_segments = len(self.waypoints) - 1
        self.seg_times = []
        self.coeffs = []

        total_dur = 0.0
        for i in range(self.num_segments):
            p_a = self.waypoints[i]
            p_b = self.waypoints[i+1]
            dist_horiz = float(np.hypot(p_b[0] - p_a[0], p_b[1] - p_a[1]))
            dist_vert = float(abs(p_b[2] - p_a[2]))

            if p_b[2] < p_a[2]:  # Descending
                if dist_horiz < 1.5:
                    t_vert = dist_vert / 0.30  # Flare descent: peak ~0.55 m/s
                    dur = max(3.0, t_vert)
                else:
                    t_vert = dist_vert / 0.55  # Glide descent: peak ~1.00 m/s
                    t_horiz = dist_horiz / avg_speed
                    dur = max(2.6, max(t_horiz, t_vert))
            else:  # Climbing
                if dist_horiz < 1.5:
                    t_vert = dist_vert / 0.65  # Vertical climbout: peak ~1.20 m/s
                    dur = max(2.6, t_vert)
                else:
                    t_vert = dist_vert / 0.80  # Ramped climb: peak ~1.50 m/s
                    t_horiz = dist_horiz / avg_speed
                    dur = max(2.4, max(t_horiz, t_vert))

            self.seg_times.append(dur)
            total_dur += dur

        self.total_duration = total_dur

        velocities = [np.zeros(3, dtype=np.float32) for _ in range(len(self.waypoints))]
        for i in range(1, len(self.waypoints) - 1):
            d_prev = self.waypoints[i] - self.waypoints[i-1]
            d_next = self.waypoints[i+1] - self.waypoints[i]
            t_span = (self.seg_times[i-1] + self.seg_times[i]) * 0.5
            v_knot = (d_prev + d_next) / max(1e-3, t_span)

            v_norm_xy = float(np.hypot(v_knot[0], v_knot[1]))
            if v_norm_xy > avg_speed:
                v_knot[:2] = v_knot[:2] * (avg_speed / v_norm_xy)

            v_knot[2] = np.clip(v_knot[2], -0.75, 1.10)
            velocities[i] = v_knot.astype(np.float32)

        for i in range(1, len(self.waypoints) - 1):
            d_prev_xy = float(np.hypot(self.waypoints[i][0] - self.waypoints[i-1][0], self.waypoints[i][1] - self.waypoints[i-1][1]))
            d_next_xy = float(np.hypot(self.waypoints[i+1][0] - self.waypoints[i][0], self.waypoints[i+1][1] - self.waypoints[i][1]))
            if d_prev_xy < 0.1 or d_next_xy < 0.1:
                velocities[i][:2] = 0.0

        velocities[0].fill(0.0)
        velocities[-1].fill(0.0)

        for i in range(self.num_segments):
            p0 = self.waypoints[i].copy()
            p1 = self.waypoints[i+1].copy()
            v0 = velocities[i].copy()
            v1 = velocities[i+1].copy()
            T = self.seg_times[i]
            delta_p = p1 - p0

            a0 = p0
            a1 = v0 * T
            a2 = np.zeros(3, dtype=np.float32)
            a3 = 10.0 * delta_p - (6.0 * v0 + 4.0 * v1) * T
            a4 = -15.0 * delta_p + (8.0 * v0 + 7.0 * v1) * T
            a5 = 6.0 * delta_p - 3.0 * (v0 + v1) * T
            self.coeffs.append((a0, a1, a2, a3, a4, a5, T))

    def evaluate(self, t):
        if t <= 0.0:
            return self.waypoints[0].copy(), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        if t >= self.total_duration:
            return self.waypoints[-1].copy(), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        elapsed = 0.0
        for (a0, a1, a2, a3, a4, a5, T) in self.coeffs:
            if elapsed <= t <= elapsed + T + 1e-5:
                tau = np.clip((t - elapsed) / max(1e-4, T), 0.0, 1.0)
                p = a0 + a1*tau + a2*(tau**2) + a3*(tau**3) + a4*(tau**4) + a5*(tau**5)
                v = (a1 + 2.0*a2*tau + 3.0*a3*(tau**2) + 4.0*a4*(tau**3) + 5.0*a5*(tau**4)) / T
                a = (2.0*a2 + 6.0*a3*tau + 12.0*a4*(tau**2) + 20.0*a5*(tau**3)) / (T ** 2)
                return p.astype(np.float32), v.astype(np.float32), a.astype(np.float32)
            elapsed += T

        return self.waypoints[-1].copy(), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

# ==========================================
# 4. DYNAMIC WEATHER & ROAD VEHICLES
# ==========================================
class DynamicUrbanWeatherEngine:
    def __init__(self):
        self.weather_modes = ["CLEAR_NOMINAL", "CANYON_GALE", "FREEZING_RAIN", "MICROBURST_STORM"]
        self.current_mode_idx = 0
        self.regime_timer = 0.0
        self.regime_duration = 45.0

    def cycle_weather(self):
        self.current_mode_idx = (self.current_mode_idx + 1) % len(self.weather_modes)
        self.regime_timer = 0.0
        print(f"\n>>> [WEATHER SHIFT] CURRENT REGIME: {self.get_mode_name()} <<<")

    def get_mode_name(self):
        return self.weather_modes[self.current_mode_idx]

    def step(self, dt=0.016):
        self.regime_timer += dt
        if self.regime_timer >= self.regime_duration:
            self.cycle_weather()

    def get_weather_parameters(self, drone_pos, avenues_x):
        mode = self.get_mode_name()
        px, py, pz = drone_pos

        if mode == "CLEAR_NOMINAL":
            base_wind = np.array([1.8, -0.6, 0.0], dtype=np.float32)
            gust_scale = 0.6
            temp_c = 22.0
            auto_ice = False
            microburst_w = 0.0
        elif mode == "CANYON_GALE":
            base_wind = np.array([5.8, 2.4, 0.0], dtype=np.float32)
            gust_scale = 3.8
            temp_c = 14.0
            auto_ice = False
            microburst_w = 0.0
        elif mode == "FREEZING_RAIN":
            base_wind = np.array([2.4, -1.8, 0.0], dtype=np.float32)
            gust_scale = 1.5
            temp_c = -4.5
            auto_ice = True
            microburst_w = -0.4
        else:  # MICROBURST_STORM
            base_wind = np.array([-4.5, 4.2, 0.0], dtype=np.float32)
            gust_scale = 5.5
            temp_c = 11.0
            auto_ice = False
            microburst_w = -2.0 if pz > 6.0 else -0.4

        venturi_boost = 1.0
        for ax in avenues_x:
            if abs(px - ax) < 7.0 and pz < 25.0:
                venturi_boost = 1.45
                break

        effective_wind = base_wind.copy()
        effective_wind[1] *= venturi_boost
        effective_wind[2] += microburst_w
        return effective_wind, gust_scale, temp_c, auto_ice

class DynamicRoadVehicleTrafficEngine:
    def __init__(self, avenues_x, streets_y):
        self.vehicles = []
        for i in range(4):
            ax = avenues_x[i % len(avenues_x)]
            self.vehicles.append({
                "id": f"Van_{i+1}",
                "pos": np.array([ax, -180.0 + i * 90.0, 0.9], dtype=np.float32),
                "vel": np.array([0.0, 4.8 + i*0.4, 0.0], dtype=np.float32),
                "axis": "Y"
            })
        for i in range(4):
            sy = streets_y[i % len(streets_y)]
            self.vehicles.append({
                "id": f"Truck_{i+1}",
                "pos": np.array([-180.0 + i * 90.0, sy, 1.1], dtype=np.float32),
                "vel": np.array([4.2 + i*0.5, 0.0, 0.0], dtype=np.float32),
                "axis": "X"
            })

    def step(self, dt=0.016):
        for v in self.vehicles:
            v["pos"] += v["vel"] * dt
            if v["axis"] == "Y":
                if v["pos"][1] > 190.0: v["pos"][1] = -190.0
                if v["pos"][1] < -190.0: v["pos"][1] = 190.0
            else:
                if v["pos"][0] > 190.0: v["pos"][0] = -190.0
                if v["pos"][0] < -190.0: v["pos"][0] = 190.0

    def get_cbf_obstacles(self):
        return [{"pos": v["pos"].copy(), "radius": 2.6, "height": 2.2, "id": v["id"]} for v in self.vehicles]

# ==========================================
# 5. DYNAMIC GROUND INTRUDER AGENT
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
            self.p = np.array([-145.0, -145.0, 0.0], dtype=np.float32)
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
        else:
            d_wander = self.wander_target - self.p[:2]
            dist_w = float(np.linalg.norm(d_wander))
            if dist_w < 0.6:
                self.wander_target = np.array([random.uniform(-140.0, 140.0), random.uniform(-140.0, 140.0)], dtype=np.float32)
            v_des = (d_wander / max(1e-4, dist_w)) * 0.8
            a_cmd = (v_des - self.v[:2]) * 1.8

        self.v[:2] += a_cmd * dt
        self.p[:2] += self.v[:2] * dt
        self.p[2] = 0.0
        return self.p.copy(), (dist < self.personal_space_m and cargo_world_pos[2] < 2.5), dist

# ==========================================
# 6. ROTOR ICING, TURBULENCE & BATTERY ECM
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
        self.gust_state = np.clip(self.gust_state, -6.0, 6.0)
        return self.gust_state.copy()

class SubZeroArrheniusBatteryECM:
    def __init__(self):
        self.soc = 0.96
        self.temp_c = 22.0
        self.v1 = np.zeros(NUM_CELLS, dtype=np.float32)
        self.capacity_coulombs = PACK_CAPACITY_AH * 3600.0
        self.v_pack = 23.8

    def step(self, p_elec_watts, amb_temp=22.0, dt=0.016):
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
        
        # Convective heat dissipation preserves internal Joule heating
        self.temp_c += ((p_joule - 0.40 * (self.temp_c - amb_temp)) / 350.0) * dt
        return self.soc, self.v_pack, i_draw, r0

# =============================================================================
# MASTER FRONTIER AUTONOMOUS DRONE SYSTEM CORE (V35.2 - TRAINING STABILIZED)
# File: drone_master_autonomous_system_v33.py — [PART 2 OF 3]
# =============================================================================

# =============================================================================
# 7. AIRFRAME 6-DOF IMU & CLAMPED 2-DOF PENDULUM WINCH
# =============================================================================
class Airframe6DOFMEMS_IMU:
    def __init__(self):
        self.b_a = np.array([0.02, -0.015, 0.03], dtype=np.float32)
        self.b_g = np.array([0.005, -0.004, 0.002], dtype=np.float32)
        self.accel_raw = np.zeros(3, dtype=np.float32)
        self.gyro_raw = np.zeros(3, dtype=np.float32)

    def poll(self, drone_R, accel_linear, omega_true, motor_omegas):
        # Specific force measured in body frame
        f_inertial = accel_linear - GRAVITY_G
        f_body = drone_R.T @ f_inertial

        rpm_ratio = float(np.mean(motor_omegas)) / MAX_OMEGA
        vib_amp = 0.16 * rpm_ratio
        vib_accel = np.random.normal(0.0, vib_amp, 3).astype(np.float32)
        
        self.accel_raw = f_body + self.b_a + vib_accel + np.random.normal(0.0, 0.035, 3).astype(np.float32)

        vib_gyro = np.random.normal(0.0, 0.015 * rpm_ratio, 3).astype(np.float32)
        self.gyro_raw = omega_true + self.b_g + vib_gyro + np.random.normal(0.0, 0.004, 3).astype(np.float32)

        return self.accel_raw.copy(), self.gyro_raw.copy()

# Clamped 2-DOF spherical pendulum: zero sway when docked at belly, derivative-damped when deployed
class AutonomousWinchDeliverySystem:
    def __init__(self):
        self.cable_len = 0.08
        self.cable_vel = 0.0
        self.theta = np.zeros(2, dtype=np.float32)
        self.theta_dot = np.zeros(2, dtype=np.float32)
        self.sway_offset = np.zeros(2, dtype=np.float32)
        self.tension_n = PAYLOAD_WEIGHT_N
        self.has_payload = True
        self.touchdown_timer = 0.0
        self.severed = False
        self.state = "STANDBY"
        self.delivery_completed = False

    def reload_new_cargo(self):
        self.cable_len = 0.08
        self.cable_vel = 0.0
        self.theta.fill(0.0)
        self.theta_dot.fill(0.0)
        self.sway_offset.fill(0.0)
        self.has_payload = True
        self.state = "STANDBY"
        self.tension_n = PAYLOAD_WEIGHT_N
        self.touchdown_timer = 0.0
        self.delivery_completed = False

    def step(self, drone_z, target_surface_z, is_settled, is_surface_safe, wire_hit, agent_breach, agent_dist, 
             wind_2d, drone_accel_linear, neural_token_idx, nn_winch_rate=0.0, dt=0.016):
        if self.severed:
            self.state = "GUILLOTINE"
            return self.cable_len, 0.0, np.zeros(2, dtype=np.float32)

        # Rigid belly fuselage clamp: when stowed (L <= 0.12m), parcel is rigidly clamped (ZERO SWAY)
        if self.cable_len > 0.12:
            L = max(0.14, self.cable_len)
            omega_n_sq = G_ACCEL / L
            coriolis_damping = (2.0 * self.cable_vel / L) * self.theta_dot
            a_drone_coupling = drone_accel_linear[:2] / L
            
            v_cargo_horiz = self.theta_dot * L
            f_wind_aero = 0.5 * 1.225 * 0.03 * (wind_2d - v_cargo_horiz) * np.linalg.norm(wind_2d - v_cargo_horiz + 1e-4)
            wind_torque = f_wind_aero / (CARGO_MASS_KG * L)
            
            heavy_damping = 1.65 * self.theta_dot / (CARGO_MASS_KG * L)
            theta_ddot = -omega_n_sq * self.theta - coriolis_damping - a_drone_coupling + wind_torque - heavy_damping

            self.theta_dot += theta_ddot * dt
            self.theta_dot = np.clip(self.theta_dot, -2.5, 2.5)
            self.theta += self.theta_dot * dt
            self.theta = np.clip(self.theta, -0.22, 0.22)
            self.sway_offset = L * np.sin(self.theta)

            theta_mag = float(np.linalg.norm(self.theta))
            theta_dot_sq = float(np.dot(self.theta_dot, self.theta_dot))
            a_vert = drone_accel_linear[2]
            
            if self.has_payload:
                t_centrif = CARGO_MASS_KG * L * theta_dot_sq
                t_grav = CARGO_MASS_KG * (G_ACCEL * math.cos(theta_mag) - a_vert)
                self.tension_n = max(1.5, t_grav + t_centrif)
            else:
                self.tension_n = 1.2
        else:
            self.theta.fill(0.0)
            self.theta_dot.fill(0.0)
            self.sway_offset.fill(0.0)
            self.tension_n = PAYLOAD_WEIGHT_N if self.has_payload else 1.2

        dropzone_occupied = (agent_dist < 2.8) and (target_surface_z < 2.0) and (self.state == "STANDBY")
        lockout = (not is_surface_safe) or wire_hit or (not is_settled) or dropzone_occupied

        if agent_breach and (target_surface_z < 2.0) and self.state in ["LOWERING", "TOUCHDOWN"]:
            self.state = "ABORT_RETRACT"

        # Autonomous Winch FSM
        if self.state == "STANDBY":
            self.cable_len = 0.08
            self.cable_vel = 0.0
            if not lockout and self.has_payload and (neural_token_idx == 2 or is_settled):
                self.state = "LOWERING"

        elif self.state == "LOWERING":
            if lockout and self.cable_len > 0.4:
                self.state = "RETRACTING"
            else:
                base_rate = 2.40
                self.cable_vel = base_rate + np.clip(nn_winch_rate * 0.8, -1.0, 0.8)
                self.cable_len += self.cable_vel * dt

                cargo_z = drone_z - self.cable_len
                if cargo_z <= (target_surface_z + 0.18) or neural_token_idx == 3:
                    self.state = "TOUCHDOWN"
                    self.touchdown_timer = 0.0

        elif self.state == "TOUCHDOWN":
            self.cable_vel = 0.0
            self.tension_n = 1.8 + float(np.random.normal(0, 0.08))
            self.touchdown_timer += dt
            if self.touchdown_timer >= 0.30 or neural_token_idx == 4:
                self.has_payload = False
                self.state = "RETRACTING"

        elif self.state in ["RETRACTING", "ABORT_RETRACT"]:
            self.cable_vel = -3.20
            self.cable_len = max(0.08, self.cable_len + self.cable_vel * dt)

            if self.cable_len <= 0.08:
                self.cable_len = 0.08
                self.cable_vel = 0.0
                if self.state == "RETRACTING" and not self.has_payload:
                    self.delivery_completed = True
                self.state = "STANDBY"

        return self.cable_len, self.tension_n, self.sway_offset

# ==========================================
# 8. CONTINUOUS BEMT COAXIAL AIRFRAME (8-DOF)
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

        self.last_p_watts = 185.0
        self.last_ige = 1.0

    def step(self, motor_cmds, current_wind, rho_air, ice_index=0.0, surface_z=0.0, has_payload=True, dt=0.016):
        current_mass = DRY_MASS_KG + (CARGO_MASS_KG if has_payload else 0.0)

        if self.parachute_deployed:
            self.motor_omegas.fill(0.0)
            self.v[0] *= 0.90
            self.v[1] *= 0.90
            self.v[2] += (-4.1 - self.v[2]) * 3.0 * dt
            self.p += (self.v + current_wind * 0.4) * dt
            if self.p[2] <= surface_z:
                self.p[2] = surface_z
                self.v.fill(0.0)
                self.omega.fill(0.0)
            return 0.0, np.zeros(3, dtype=np.float32), 0.0, 1.0

        effective_cmds = np.clip(motor_cmds, 0.0, MAX_OMEGA) * self.motor_health
        self.motor_omegas += ((effective_cmds - self.motor_omegas) / MOTOR_TAU_S) * dt
        self.motor_omegas *= self.motor_health

        z_h = max(0.12, float(self.p[2] - surface_z))
        ige_factor = 1.0 / max(0.35, (1.0 - (ROTOR_RADIUS_M / (4.0 * z_h)) ** 2))
        ige_factor = float(np.clip(ige_factor, 1.0, 1.35))
        d_ceil = max(0.3, CEILING_HEIGHT_M - self.p[2])
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
                top_downwash = 0.035 * (self.motor_omegas[i - 1] * ROTOR_RADIUS_M) * COAXIAL_DOWNWASH_INGESTION

            pitch_comp = math.radians(2.5) if is_bottom else 0.0

            t_blade, q_blade = 0.0, 0.0
            for r in r_steps:
                v_axial = np.clip(-self.v[2] + top_downwash, -8.0, 12.0)
                v_tan = omega_i * r
                v_eff_sq = v_axial**2 + v_tan**2
                phi = math.atan2(max(0.1, v_axial), max(0.1, v_tan))

                theta_r = math.radians(14.0 - 8.0 * (r / ROTOR_RADIUS_M)) + pitch_comp
                alpha = np.clip(theta_r - phi, -0.10, 0.42)
                c_l = 2.0 * math.pi * alpha * lift_loss_factor
                c_d = (0.012 + 0.04 * (alpha ** 2)) * drag_rise_factor

                exp_arg = -max(0.01, (ROTOR_RADIUS_M - r) / (2.0 * r * math.sin(max(0.01, phi))))
                acos_arg = min(1.0, max(-1.0, math.exp(exp_arg)))
                f_tip = max(0.2, min(1.0, (2.0 / math.pi) * math.acos(acos_arg)))

                dT = 0.5 * rho_air * v_eff_sq * 0.018 * (c_l * math.cos(phi) - c_d * math.sin(phi)) * f_tip * dr
                dQ = 0.5 * rho_air * v_eff_sq * 0.018 * (c_l * math.sin(phi) + c_d * math.cos(phi)) * f_tip * r * dr
                t_blade += dT
                q_blade += dQ

            T[i] = max(0.0, t_blade * 2.0 * ige_factor * ice_suction)
            Q[i] = max(0.0, q_blade * 2.0)

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
        f_gravity = current_mass * GRAVITY_G

        self.accel_linear = (self.R @ f_body + f_gravity + f_drag) / current_mass
        gyroscopic = np.cross(self.omega, INERTIA_MATRIX @ self.omega)
        accel_angular = INV_INERTIA @ (torques - gyroscopic)

        weight_force = current_mass * G_ACCEL
        is_on_surface = (self.p[2] <= surface_z + 0.04)

        if is_on_surface and (f_thrust < 0.98 * weight_force):
            self.p[2] = surface_z
            self.v.fill(0.0)
            self.omega.fill(0.0)
            accel_angular.fill(0.0)
            self.accel_linear.fill(0.0)
        else:
            self.p += self.v * dt + 0.5 * self.accel_linear * (dt ** 2)
            self.v += self.accel_linear * dt
            
            if self.p[2] < surface_z:
                self.p[2] = surface_z
                self.v[2] = max(0.0, self.v[2] * 0.05)
                self.v[0] *= 0.85
                self.v[1] *= 0.85

            self.omega += accel_angular * dt
            d_theta = np.linalg.norm(self.omega) * dt
            if d_theta > 1e-6:
                axis = self.omega / np.linalg.norm(self.omega)
                K = skew_symmetric(axis)
                R_delta = np.eye(3, dtype=np.float32) + math.sin(d_theta) * K + (1.0 - math.cos(d_theta)) * (K @ K)
                self.R = self.R @ R_delta

            U, _, Vt = np.linalg.svd(self.R)
            self.R = (U @ Vt).astype(np.float32)

        p_watts = float(np.sum(Q * self.motor_omegas)) / 0.85 + 24.0
        self.last_p_watts = p_watts
        self.last_ige = ige_factor
        return f_thrust, torques, p_watts, ige_factor

# ==========================================
# 9. ADVANCED SENSORS & SPATIAL GRADIENTS
# ==========================================
class AdvancedMultiModalSensorSuite:
    def __init__(self, world: MetropolitanCityGridWorld400m):
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
        min_roof_dist = 999.0
        for h in self.world.houses:
            x1, x2, y1, y2, _ = h["aabb"]
            dx = max(0.0, max(x1 - px, px - x2))
            dy = max(0.0, max(y1 - py, py - y2))
            d = math.hypot(dx, dy)
            min_roof_dist = min(min_roof_dist, d)

        roof_heat = 7.5 * math.exp(-min_roof_dist / 8.0)
        park_dist = math.hypot(px - (-25.0), py - (-25.0))
        park_cool = -3.5 * math.exp(-park_dist / 22.0)
        th_dt = float(roof_heat + park_cool + 0.5)

        dolp = float(0.12 + 0.55 * math.exp(-min_roof_dist / 6.0))
        ndvi = float(0.10 + 0.68 * math.exp(-park_dist / 25.0))
        return th_dt, dolp, ndvi

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
        rh_pct = float(np.clip(68.0 - 0.4 * drone_v[2], 30.0, 95.0))
        pm10_dust = float(np.clip(22.0 + 1.2 * float(np.linalg.norm(current_wind[:2])), 5.0, 80.0))
        return math.degrees(alpha_rad), math.degrees(beta_rad), rh_pct, pm10_dust

    def poll_rtk_gnss_metrics(self, drone_p, is_spoofed):
        hdop = 3.8 if is_spoofed else 0.85
        cn0_avg = 28.5 if is_spoofed else 44.0
        rf_jam_pwr = -52.0 if is_spoofed else -110.0
        return hdop, cn0_avg, rf_jam_pwr, 0.02

# ==========================================
# 10. MACRO PERCEPTION & ASSURANCE SUITE
# ==========================================
class MacroPerceptionLiDARSuite:
    def __init__(self, world: MetropolitanCityGridWorld400m):
        self.world = world

    def cast_360_lidar(self, drone_pos, yaw_rad, dynamic_vehicles=None):
        px, py, pz = drone_pos
        ranges = np.full(16, 30.0, dtype=np.float32)

        for beam_idx in range(16):
            angle = yaw_rad + (2.0 * math.pi * beam_idx / 16.0)
            dx, dy = math.cos(angle), math.sin(angle)
            min_dist = 30.0

            # Ray-AABB Slab intersection for buildings
            for h in self.world.houses:
                x1, x2, y1, y2, hh = h["aabb"]
                if pz <= (hh + 0.1):
                    if abs(dx) > 1e-5:
                        t1 = (x1 - px) / dx
                        t2 = (x2 - px) / dx
                        t_min_x, t_max_x = min(t1, t2), max(t1, t2)
                    else:
                        if px < x1 or px > x2: continue
                        t_min_x, t_max_x = -1e6, 1e6

                    if abs(dy) > 1e-5:
                        t1 = (y1 - py) / dy
                        t2 = (y2 - py) / dy
                        t_min_y, t_max_y = min(t1, t2), max(t1, t2)
                    else:
                        if py < y1 or py > y2: continue
                        t_min_y, t_max_y = -1e6, 1e6

                    t_enter = max(t_min_x, t_min_y)
                    t_exit = min(t_max_x, t_max_y)

                    if t_enter <= t_exit and t_exit > 0.0:
                        hit_dist = t_enter if t_enter > 0.0 else 0.0
                        if hit_dist < min_dist:
                            min_dist = hit_dist

            # Angular beam cone check for dynamic road vehicles
            if dynamic_vehicles:
                for v in dynamic_vehicles:
                    vx, vy = v["pos"][0], v["pos"][1]
                    d_veh = math.hypot(px - vx, py - vy)
                    if pz < 3.2 and d_veh < min_dist:
                        bearing = math.atan2(vy - py, vx - px)
                        diff = abs((bearing - angle + math.pi) % (2.0 * math.pi) - math.pi)
                        if diff <= (math.pi / 16.0):
                            min_dist = float(d_veh)

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
                if dist_y < 3.5 and pz > w1[2]:
                    wire_clearance = min(wire_clearance, float(pz - w1[2]))

        surf_h = self.world.get_surface_height(px, py)
        true_clearance = max(0.05, float(pz - surf_h))

        cone_slant = math.cos(math.radians(25.0))
        for i in range(8):
            conical_ranges[i] = min(15.0, float(true_clearance / max(0.1, cone_slant)))

        return conical_ranges, wire_clearance

class SurfaceAndWireAssuranceSuite:
    def __init__(self, world: MetropolitanCityGridWorld400m):
        self.world = world

    def audit_surface(self, pos_xy, target_z=0.0, buffer_m=2.5):
        px, py = pos_xy
        for h in self.world.houses:
            x1, x2, y1, y2, hh = h["aabb"]
            if x1 <= px <= x2 and y1 <= py <= y2:
                if abs(target_z - hh) < 1.4 or target_z > 5.0:
                    return "ROOFTOP_PAD_SAFE", True, 0.95
                else:
                    return "ROOF_HAZARD", False, 0.85

        for line in self.world.powerlines:
            w1 = line["p1"]
            if abs(py - w1[1]) < (buffer_m + 1.2):
                return "WIRE_CORRIDOR_HAZARD", False, 0.50

        return "SOLID_TERRAIN_SAFE", True, 0.92

    def snap_to_nearest_safe_ground(self, target_xy):
        cand = np.array([target_xy[0], target_xy[1], 0.0], dtype=np.float32)
        for h in self.world.houses:
            x1, x2, y1, y2, hh = h["aabb"]
            if x1 <= cand[0] <= x2 and y1 <= cand[1] <= y2:
                roof_x = float((x1 + x2) / 2.0)
                roof_y = float((y1 + y2) / 2.0)
                return np.array([roof_x, roof_y, hh], dtype=np.float32)

        _, is_safe, _ = self.audit_surface(cand[:2], target_z=0.0)
        if is_safe:
            return cand

        for r in np.linspace(3.5, 25.0, 16):
            for ang in np.linspace(0, 2*math.pi, 16):
                test_pos = np.array([cand[0] + r*math.cos(ang), cand[1] + r*math.sin(ang), 0.0], dtype=np.float32)
                _, safe_found, _ = self.audit_surface(test_pos[:2], target_z=0.0)
                if safe_found:
                    return test_pos

        return np.array([-155.0, -155.0, 0.0], dtype=np.float32)

class FullPhysicsSensorSuite:
    def __init__(self):
        self.wire_y = 170.0
        self.wire_z = 12.0

    def poll_all_physics_sensors(self, drone_p, drone_v, current_wind, surface_z=0.0):
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
        
        true_clearance = max(0.05, float(drone_p[2] - surface_z))
        min_sonar = max(0.05, min(15.0, true_clearance))
        return {"z_baro": z_baro, "rho_air": rho_air, "v_tas": v_tas, "q_dyn": q_dyn, "emf_rms": emf_rms, "min_sonar": min_sonar}

# =============================================================================
# 11. QUANTUM SENSOR & HEADING-AUGMENTED ERROR-STATE EKF ON SO(3)
# =============================================================================
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

        max_diag = np.array([100.0]*3 + [50.0]*3 + [0.5]*3 + [1.0]*3 + [0.2]*3, dtype=np.float32)
        diag_P = np.diag(self.P)
        for k in range(15):
            if diag_P[k] > max_diag[k]:
                scale = math.sqrt(max_diag[k] / diag_P[k])
                self.P[k, :] *= scale
                self.P[:, k] *= scale
        self.P = 0.5 * (self.P + self.P.T)

    def update_measurement(self, H, innovation, R_cov):
        S = H @ self.P @ H.T + R_cov
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        delta_x = (K @ innovation).flatten()
        delta_x[0:6] = np.clip(delta_x[0:6], -3.5, 3.5)
        delta_x[6:9] = np.clip(delta_x[6:9], -0.25, 0.25)
        delta_x[9:15] = np.clip(delta_x[9:15], -0.5, 0.5)

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

    def update_heading(self, measured_yaw_rad, r_cov=0.03):
        R_curr = quat_to_rot(self.q)
        current_yaw = math.atan2(R_curr[1, 0], R_curr[0, 0])
        yaw_innov = (measured_yaw_rad - current_yaw + math.pi) % (2.0 * math.pi) - math.pi

        H_yaw = np.zeros((1, 15), dtype=np.float32)
        H_yaw[0, 8] = 1.0
        self.update_measurement(H_yaw, np.array([yaw_innov], dtype=np.float32), np.array([[r_cov]], dtype=np.float32))

# =============================================================================
# 12. SMOOTH VELOCITY-GOVERNED MIXER & DECOUPLED SWARM CBF (FIX 2 & FIX 3)
# =============================================================================
class ReciprocalSwarmCBFQPFilter:
    def __init__(self):
        self.alpha_obs = CBF_ALPHA_OBSTACLE
        self.safe_radius = CBF_SAFE_RADIUS_M
        self.swarm_r = SWARM_INTER_DRONE_SAFE_RADIUS_M

    def enforce_safety(self, drone_idx, drone_p, drone_v, nominal_accel, obstacles, swarm_states=None, target_bldg_id=None, departure_bldg_id=None):
        safe_accel = nominal_accel.copy()

        # 1. Height-bounded 3D building barrier
        for obs in obstacles:
            obs_id = obs.get("id")
            obs_h = obs.get("height", 0.0)
            rel_pos = drone_p - obs["pos"]
            dist_xy = math.hypot(rel_pos[0], rel_pos[1])
            h_obs = dist_xy - (obs["radius"] + self.safe_radius)

            is_target_bldg = (target_bldg_id is not None and obs_id == target_bldg_id)
            is_departure_bldg = (departure_bldg_id is not None and obs_id == departure_bldg_id)

            # Fix 3: Rooftop Departure Masking (Disables horizontal repulsion during climbout until clear)
            if not is_target_bldg and not is_departure_bldg:
                if drone_p[2] <= (obs_h - 0.4):
                    if h_obs < 1.6:
                        grad_h = np.array([rel_pos[0] / max(1e-3, dist_xy), rel_pos[1] / max(1e-3, dist_xy), 0.0], dtype=np.float32)
                        l_fh = float(np.dot(grad_h, drone_v))
                        b_cbf = -l_fh - self.alpha_obs * h_obs
                        val = float(np.dot(grad_h, safe_accel))
                        if val < b_cbf:
                            lambda_kkt = (b_cbf - val) / float(np.dot(grad_h, grad_h))
                            safe_accel += lambda_kkt * grad_h
            
            # Above roof: soft deceleration cushion
            if dist_xy <= (obs["radius"] + 0.5) and drone_p[2] >= (obs_h - 0.2):
                h_roof = drone_p[2] - (obs_h + 0.15)
                if h_roof < 0.8:
                    l_f_roof = drone_v[2]
                    b_roof = -l_f_roof - 2.0 * h_roof
                    if safe_accel[2] < b_roof:
                        safe_accel[2] = b_roof

        # 2. Decoupled Horizontal Swarm Barrier
        if swarm_states:
            for j, other in enumerate(swarm_states):
                if j == drone_idx: continue
                p_other, v_other = other["p"], other["v"]
                
                dz = abs(drone_p[2] - p_other[2])
                if dz >= 2.8:
                    continue

                d_p_xy = drone_p[:2] - p_other[:2]
                dist_xy = float(np.linalg.norm(d_p_xy))
                h_swarm = dist_xy - self.swarm_r
                
                if h_swarm < 3.0:
                    n_xy = d_p_xy / max(1e-4, dist_xy)
                    n_ij = np.array([n_xy[0], n_xy[1], 0.0], dtype=np.float32)
                    
                    v_rel_xy = drone_v[:2] - v_other[:2]
                    l_fh_swarm = float(np.dot(n_xy, v_rel_xy))
                    b_swarm = -l_fh_swarm - 2.2 * h_swarm
                    
                    val_swarm = float(np.dot(n_ij, safe_accel))
                    if val_swarm < b_swarm:
                        lambda_swarm = (b_swarm - val_swarm) / float(np.dot(n_ij, n_ij))
                        safe_accel += lambda_swarm * n_ij

        # 3. Ground Hard Floor Barrier
        h_gnd = drone_p[2] - 0.20
        if h_gnd < 0.6:
            l_f_gnd = drone_v[2]
            b_gnd = -l_f_gnd - 2.0 * h_gnd
            if safe_accel[2] < b_gnd:
                safe_accel[2] = b_gnd

        return safe_accel

class RateDampedActuatorMixerWithCBF:
    def __init__(self):
        self.k_tilt = 2.4
        self.k_w = 0.45
        self.cbf_filter = ReciprocalSwarmCBFQPFilter()

        arm = ARM_LENGTH_M / math.sqrt(2.0)
        cq_ct = C_Q_CLEAN / C_T_CLEAN
        self.B_nominal = np.array([
            [ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
            [-arm, -arm,  arm,  arm,  arm,  arm, -arm, -arm],
            [-arm, -arm, -arm, -arm,  arm,  arm,  arm,  arm],
            [-cq_ct, cq_ct, cq_ct, -cq_ct, -cq_ct, cq_ct, cq_ct, -cq_ct]
        ], dtype=np.float32)

    def mix(self, drone_health, p_est, v_est, R_est, omega_est, p_target, neural_guidance, token_idx,
            estimated_wind, obstacles, swarm_states=None, drone_idx=0, ref_accel=None, ref_vel=None,
            target_bldg_id=None, departure_bldg_id=None, cable_len=0.08, sway_2d=None, load_cell_3d=None, ige_fac=1.0,
            v_pack=23.5, bms_temp_c=22.0, has_payload=True, dt=0.016):

        current_mass = DRY_MASS_KG + (CARGO_MASS_KG if has_payload else 0.0)
        weight_force = current_mass * G_ACCEL

        if ref_accel is None: ref_accel = np.zeros(3, dtype=np.float32)
        if ref_vel is None: ref_vel = np.zeros(3, dtype=np.float32)
        if swarm_states is None: swarm_states = []

        delta_p = p_target - p_est
        delta_p_clamped = np.clip(delta_p, -6.0, 6.0)

        # 1. Horizontal Tracking (XY plane)
        a_p_xy = np.array([2.0, 2.0], dtype=np.float32) * delta_p_clamped[:2]
        a_d_xy = np.array([2.2, 2.2], dtype=np.float32) * (ref_vel[:2] - v_est[:2])
        a_track_xy = a_p_xy + a_d_xy

        # 2. Smooth Velocity-Governed Vertical Tracking (Bounded elevator profile)
        v_cmd_z = float(np.clip(ref_vel[2] + 0.90 * delta_p_clamped[2], -1.05, 1.50))
        a_track_z = 3.20 * (v_cmd_z - v_est[2])
        a_track = np.array([a_track_xy[0], a_track_xy[1], a_track_z], dtype=np.float32)

        # Fix 2: Authority Separation - Vertical neural guidance is strictly ZEROED OUT
        # Altitude is 100% governed by the deterministic flight corridor
        a_neural_cmd = np.array([neural_guidance[0] * 1.5, neural_guidance[1] * 1.5, 0.0], dtype=np.float32)

        a_anti_sway = np.zeros(3, dtype=np.float32)
        if cable_len > 0.12 and sway_2d is not None:
            a_anti_sway[:2] -= 1.6 * sway_2d

        a_aero_feedforward = np.zeros(3, dtype=np.float32)
        if ige_fac > 1.05:
            a_aero_feedforward[2] -= 1.6 * (ige_fac - 1.0)
        if estimated_wind[2] < -0.3:
            a_aero_feedforward[2] -= 0.5 * estimated_wind[2]

        a_nominal = a_track + a_neural_cmd + ref_accel + a_anti_sway + a_aero_feedforward
        a_safe = self.cbf_filter.enforce_safety(
            drone_idx, p_est, v_est, a_nominal, obstacles, swarm_states, 
            target_bldg_id=target_bldg_id, departure_bldg_id=departure_bldg_id
        )
        
        F_d = a_safe * current_mass + current_mass * (-GRAVITY_G)

        # Dynamic Mass-Proportional Thrust Clamping
        # Floor (0.68x) prevents free-fall drops; Ceiling (1.55x) prevents rocket launches
        min_f_vert = 0.68 * weight_force
        max_f_vert = 1.55 * weight_force
        F_d[2] = float(np.clip(F_d[2], min_f_vert, max_f_vert))

        max_tilt_rad = math.radians(22.0)
        f_horiz = math.hypot(F_d[0], F_d[1])
        f_vert = max(1.0, F_d[2])
        if f_horiz / f_vert > math.tan(max_tilt_rad):
            scale_h = (f_vert * math.tan(max_tilt_rad)) / f_horiz
            F_d[0] *= scale_h
            F_d[1] *= scale_h

        b3 = R_est[:, 2]
        f_thrust = float(np.dot(F_d, b3))
        f_thrust = float(np.clip(f_thrust, min_f_vert, max_f_vert))

        b3_d = F_d / (np.linalg.norm(F_d) + 1e-6)

        # Restoring moments: positive cross product matches forward tilt kinematics
        e_tilt_world = np.cross(b3, b3_d)
        e_tilt_body = R_est.T @ e_tilt_world

        tau_x = self.k_tilt * e_tilt_body[0] - self.k_w * omega_est[0]
        tau_y = self.k_tilt * e_tilt_body[1] - self.k_w * omega_est[1]

        # Locked yaw control to North heading
        yaw_target = 0.0
        current_yaw = math.atan2(R_est[1, 0], R_est[0, 0])
        yaw_err = (current_yaw - yaw_target + math.pi) % (2.0 * math.pi) - math.pi
        tau_z = -1.8 * yaw_err - 0.35 * omega_est[2]
        tau_z = float(np.clip(tau_z, -0.15, 0.15))

        gyroscopic = np.cross(omega_est, INERTIA_MATRIX @ omega_est)
        tau = np.array([tau_x + gyroscopic[0], tau_y + gyroscopic[1], tau_z + gyroscopic[2]], dtype=np.float32)

        desired_wrench = np.array([f_thrust, tau[0], tau[1], tau[2]], dtype=np.float32)

        W_health = np.diag(drone_health).astype(np.float32)
        B_active = self.B_nominal @ W_health
        try:
            B_pinv = np.linalg.pinv(B_active)
            T_cmds = B_pinv @ desired_wrench
        except np.linalg.LinAlgError:
            T_cmds = np.full(8, f_thrust / 8.0, dtype=np.float32)

        active_mask = (drone_health > 0.0)
        if np.any(active_mask):
            min_active_t = float(np.min(T_cmds[active_mask]))
            if min_active_t < 0.10:
                shift = min(0.35, 0.10 - min_active_t)
                T_cmds[active_mask] += shift

        T_cmds = np.clip(T_cmds * drone_health, 0.0, 10.5)

        omega_cmds = np.zeros(8, dtype=np.float32)
        for i in range(8):
            if drone_health[i] > 0.0:
                scale_c = C_T_CLEAN if (i % 2 == 0) else (C_T_CLEAN * COAXIAL_EFFICIENCY)
                omega_cmds[i] = math.sqrt(T_cmds[i] / scale_c)

        return omega_cmds, f_thrust, a_safe

# =============================================================================
# 13. 3-FRAME TEMPORAL OBSERVATION SYNTHESIZER WITH SWARM OBSERVABLES (366-DIM)
# =============================================================================
class DroneObservationSynthesizer122Dim:
    def __init__(self):
        self.last_tension = PAYLOAD_WEIGHT_N

    def synthesize_instantaneous(self, p_est, v_est, R_est, omega_est, target_p, lidar_16, conical_8,
                                 v_tas, q_dyn, min_sonar, wind_est, ice_idx, cable_len, cable_vel,
                                 tension_n, sway_2d, bms_soc, v_pack, bms_temp_c, r0_dyn,
                                 chi2_val, jerk_discrepancy, is_spoofed, emf_rms, wire_dist,
                                 agent_dist, ground_spl_dba, has_payload,
                                 hook_a, hook_w, load_cell_3d, thermal_dt, dolp, ndvi,
                                 esc_i_rms, esc_t_mosfet, alpha_deg, beta_deg, rh_pct, dust_pm10,
                                 hdop, cn0_avg, rf_jam_pwr, mag_grad,
                                 airframe_accel=None, airframe_gyro=None,
                                 peer_swarm_kinematics=None, dt=0.016):
        obs = np.zeros(BASE_OBS_DIM, dtype=np.float32)

        if airframe_accel is None: airframe_accel = np.zeros(3, dtype=np.float32)
        if airframe_gyro is None: airframe_gyro = np.zeros(3, dtype=np.float32)

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
        obs[15:31] = np.clip(lidar_16 / 30.0, 0.0, 1.0)

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

        # 12. 8-Channel ESC Telemetry [76:92]
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

        # 15. Dedicated Airframe 6-DOF MEMS IMU [100:106]
        obs[100:103] = np.clip(airframe_accel / 15.0, -1.0, 1.0)
        obs[103:106] = np.clip(airframe_gyro / 8.0, -1.0, 1.0)

        # 16. Option A: Peer Swarm Relative Kinematics [106:122]
        if peer_swarm_kinematics is not None and len(peer_swarm_kinematics) > 0:
            for p_i, p_info in enumerate(peer_swarm_kinematics[:2]):
                base_idx = 106 + p_i * 8
                p_rel = p_info["p_rel"]
                v_rel = p_info["v_rel"]
                dist = p_info["dist"]
                obs[base_idx:base_idx+3] = np.clip(p_rel / 100.0, -1.0, 1.0)
                obs[base_idx+3:base_idx+6] = np.clip(v_rel / 10.0, -1.0, 1.0)
                obs[base_idx+6] = np.clip(dist / 150.0, 0.0, 1.0)
                obs[base_idx+7] = 1.0 if p_info.get("has_payload", True) else 0.0

        return obs

    def stack_history(self, history_deque, current_122_obs):
        history_deque.append(current_122_obs)
        while len(history_deque) < HISTORY_LEN:
            history_deque.append(current_122_obs)
        return np.concatenate(list(history_deque), axis=0).astype(np.float32)

DroneObservationSynthesizer318Dim = DroneObservationSynthesizer122Dim
DroneObservationSynthesizer366Dim = DroneObservationSynthesizer122Dim
DroneObservationSynthesizer106Dim = DroneObservationSynthesizer122Dim

# =============================================================================
# MASTER FRONTIER AUTONOMOUS DRONE SYSTEM CORE (V35.2 - TRAINING STABILIZED)
# File: drone_master_autonomous_system_v33.py — [PART 3 OF 3]
# =============================================================================

# =============================================================================
# 14. MULTI-AGENT SWARM CROSS-ATTENTION EXECUTIVE BACKBONE (~2.5M PARAMETERS)
# =============================================================================
class DenseResBlock512(nn.Module):
    def __init__(self, dim=512, hidden_dim=768):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.norm2(out)
        out = self.fc2(out)
        return residual + out

# Training Fix 1: Auto-Padded Swarm Cross-Attention (Strict length-3 attention token sequence)
class SwarmFleetCrossAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        if x.dim() == 2:
            B, D = x.shape
            rem = B % 3
            if rem != 0:
                # Pad to multiple of 3 to guarantee exact length-3 attention across all batch sizes
                pad = torch.zeros(3 - rem, D, device=x.device, dtype=x.dtype)
                x_padded = torch.cat([x, pad], dim=0)
                B_p = x_padded.shape[0]
                x_seq = x_padded.view(B_p // 3, 3, D)
                attn_out, _ = self.attn(x_seq, x_seq, x_seq)
                out = self.norm(x_seq + attn_out).view(B_p, D)
                return out[:B]
            else:
                x_seq = x.view(B // 3, 3, D)
                attn_out, _ = self.attn(x_seq, x_seq, x_seq)
                return self.norm(x_seq + attn_out).view(B, D)
        return x

class DeepDenseFlightExecutiveNet(nn.Module):
    def __init__(self, in_features=OBSERVATION_DIM):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.SiLU()
        )
        
        self.block1 = DenseResBlock512(dim=512, hidden_dim=768)
        self.swarm_attn = SwarmFleetCrossAttention(embed_dim=512, num_heads=4)
        self.block2 = DenseResBlock512(dim=512, hidden_dim=768)
        self.block3 = DenseResBlock512(dim=512, hidden_dim=768)

        # 6 Multi-Task Supervisory & Diagnostic Heads
        self.guidance_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 4),
            nn.Tanh()
        )
        self.winch_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
        self.health_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 8),
            nn.Sigmoid()
        )
        self.trust_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
        self.executive_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, NUM_EXECUTIVE_STATES)
        )
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        feat = self.input_proj(x)
        feat = self.block1(feat)
        feat = self.swarm_attn(feat)
        feat = self.block2(feat)
        feat = self.block3(feat)

        guidance = self.guidance_head(feat)
        winch_cmd = self.winch_head(feat)
        health_pred = self.health_head(feat)
        trust_scores = self.trust_head(feat)
        fsm_logits = self.executive_head(feat)
        value = self.value_head(feat)
        return guidance, fsm_logits, winch_cmd, health_pred, trust_scores, value

MultiModalFlightExecutiveNet = DeepDenseFlightExecutiveNet
MultiModalPPOActorCriticNet = DeepDenseFlightExecutiveNet

# =============================================================================
# 15. MULTI-LEG ROUND-TRIP MISSION HARVESTER & STABILIZED TRAINER
# =============================================================================
def _worker_harvest_metropolitan(num_steps):
    np.random.seed()
    random.seed()
    world = MetropolitanCityGridWorld400m()
    synth = DroneObservationSynthesizer122Dim()
    lidar_engine = MacroPerceptionLiDARSuite(world)
    sensor_suite = AdvancedMultiModalSensorSuite(world)
    assurance = SurfaceAndWireAssuranceSuite(world)
    mixer = RateDampedActuatorMixerWithCBF()

    obs_list = []
    y_guidance, y_fsm, y_winch, y_health, y_trust, y_val = [], [], [], [], [], []

    while len(obs_list) < num_steps:
        target_entry = random.choice(world.manifest_addresses)
        target_pad = target_entry["pos"].copy()
        roof_z = float(target_pad[2])
        depot_pad = random.choice(world.depot_helipads).copy()
        assigned_alt = random.choice([30.0, 35.0, 40.0])

        # Symmetrical round-trip flight regime initialization
        regime_roll = random.random()
        if regime_roll < 0.30:
            sim_regime = "OUTBOUND"
            start_p = np.array([
                random.uniform(-160.0, 160.0),
                random.uniform(-160.0, 160.0),
                random.uniform(assigned_alt - 2.0, assigned_alt + 2.0)
            ], dtype=np.float32)
            sim_winch = AutonomousWinchDeliverySystem()
            sim_winch.has_payload = True
        elif regime_roll < 0.45:
            sim_regime = "ROOF_DELIVERY"
            start_p = np.array([
                target_pad[0] + random.uniform(-0.8, 0.8),
                target_pad[1] + random.uniform(-0.8, 0.8),
                roof_z + 2.5 + random.uniform(-0.3, 0.3)
            ], dtype=np.float32)
            sim_winch = AutonomousWinchDeliverySystem()
            sim_winch.has_payload = True
        elif regime_roll < 0.60:
            sim_regime = "CLIMBOUT"
            start_p = np.array([
                target_pad[0] + random.uniform(-0.5, 0.5),
                target_pad[1] + random.uniform(-0.5, 0.5),
                roof_z + 2.5 + random.uniform(0.0, 2.0)
            ], dtype=np.float32)
            sim_winch = AutonomousWinchDeliverySystem()
            sim_winch.has_payload = False
        elif regime_roll < 0.85:
            sim_regime = "RTB_TRANSIT"
            start_p = np.array([
                random.uniform(-160.0, 160.0),
                random.uniform(-160.0, 160.0),
                random.uniform(assigned_alt - 2.0, assigned_alt + 2.0)
            ], dtype=np.float32)
            sim_winch = AutonomousWinchDeliverySystem()
            sim_winch.has_payload = False
        else:
            sim_regime = "DEPOT_TOUCHDOWN"
            start_p = np.array([
                depot_pad[0] + random.uniform(-1.0, 1.0),
                depot_pad[1] + random.uniform(-1.0, 1.0),
                random.uniform(1.2, 4.0)
            ], dtype=np.float32)
            sim_winch = AutonomousWinchDeliverySystem()
            sim_winch.has_payload = False

        airframe = BEMTCoaxialAirframe8DOF(start_p)
        airframe_imu = Airframe6DOFMEMS_IMU()

        motor_health = np.ones(8, dtype=np.float32)
        if random.random() < 0.18:
            fail_idx = random.randint(0, 7)
            motor_health[fail_idx] = 0.0
            if random.random() < 0.35:
                fail_idx2 = (fail_idx + 4) % 8
                motor_health[fail_idx2] = 0.0
        airframe.motor_health = motor_health.copy()

        weather_mode = random.choice(["CLEAR_NOMINAL", "CANYON_GALE", "FREEZING_RAIN", "MICROBURST_STORM"])
        if weather_mode == "CLEAR_NOMINAL":
            episode_wind = np.array([random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0), 0.0], dtype=np.float32)
            episode_temp_c = random.uniform(18.0, 26.0)
            ice_active = False
        elif weather_mode == "CANYON_GALE":
            episode_wind = np.array([random.uniform(-6.0, 6.0), random.uniform(-6.0, 6.0), 0.0], dtype=np.float32)
            episode_temp_c = random.uniform(10.0, 16.0)
            ice_active = False
        elif weather_mode == "FREEZING_RAIN":
            episode_wind = np.array([random.uniform(-3.0, 3.0), random.uniform(-3.0, 3.0), 0.0], dtype=np.float32)
            episode_temp_c = random.uniform(-6.0, -1.0)
            ice_active = True
        else:
            episode_wind = np.array([random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0), random.uniform(-2.5, 0.0)], dtype=np.float32)
            episode_temp_c = random.uniform(8.0, 14.0)
            ice_active = False

        episode_soc = random.uniform(0.20, 0.98)
        episode_v_pack = 18.5 + episode_soc * (25.2 - 18.5)
        is_spoofed_ep = (random.random() < 0.20)
        chi2_ep = float(random.uniform(22.0, 48.0)) if is_spoofed_ep else float(random.uniform(0.8, 3.0))

        history_q = deque(maxlen=HISTORY_LEN)
        peer_offsets = [
            np.array([random.uniform(-40.0, 40.0), random.uniform(-40.0, 40.0), random.choice([-5.0, 0.0, 5.0])], dtype=np.float32),
            np.array([random.uniform(-40.0, 40.0), random.uniform(-40.0, 40.0), random.choice([-10.0, 10.0])], dtype=np.float32)
        ]

        step_count = 0
        dt = 0.016

        # Step episode through round-trip closed-loop lifecycle
        while step_count < 220 and len(obs_list) < num_steps:
            d_dest_xy = float(np.linalg.norm(airframe.p[:2] - target_pad[:2]))
            d_depot_xy = float(np.linalg.norm(airframe.p[:2] - depot_pad[:2]))

            if sim_regime == "OUTBOUND":
                hover_dest = np.array([target_pad[0], target_pad[1], assigned_alt], dtype=np.float32)
                fsm_token = 0  # TRANSIT
                winch_rate_cmd = 0.0
                d_xy_vec = target_pad[:2] - airframe.p[:2]
                v_horiz_dir = d_xy_vec / max(1e-4, float(np.linalg.norm(d_xy_vec)))
                v_des = np.array([v_horiz_dir[0] * 4.5, v_horiz_dir[1] * 4.5, 0.0], dtype=np.float32)
                if d_dest_xy <= 1.4:
                    sim_regime = "ROOF_DELIVERY"

            elif sim_regime == "ROOF_DELIVERY":
                hover_dest = np.array([target_pad[0], target_pad[1], roof_z + 2.5], dtype=np.float32)
                v_des = np.zeros(3, dtype=np.float32)
                if sim_winch.has_payload and sim_winch.cable_len <= 0.10:
                    fsm_token = 1  # HOVER_LOCK
                    winch_rate_cmd = 0.0
                elif sim_winch.has_payload and sim_winch.cable_len > 0.10:
                    cargo_z = airframe.p[2] - sim_winch.cable_len
                    if cargo_z > (roof_z + 0.18):
                        fsm_token = 2  # WINCH_LOWER
                        winch_rate_cmd = 1.0
                    else:
                        fsm_token = 3  # TOUCHDOWN_RELEASE
                        winch_rate_cmd = 0.0
                elif not sim_winch.has_payload and sim_winch.cable_len > 0.08:
                    fsm_token = 4  # WINCH_RETRACT
                    winch_rate_cmd = -1.0
                else:
                    fsm_token = 5  # AUTO_DISPATCH -> Trigger climbout
                    winch_rate_cmd = 0.0
                    sim_regime = "CLIMBOUT"

            elif sim_regime == "CLIMBOUT":
                hover_dest = np.array([target_pad[0], target_pad[1], roof_z + 4.5], dtype=np.float32)
                fsm_token = 5  # AUTO_DISPATCH (vertical climbout)
                winch_rate_cmd = 0.0
                v_des = np.array([0.0, 0.0, 1.20], dtype=np.float32)
                if airframe.p[2] >= (roof_z + 4.0):
                    sim_regime = "RTB_TRANSIT"

            elif sim_regime == "RTB_TRANSIT":
                hover_dest = np.array([depot_pad[0], depot_pad[1], assigned_alt], dtype=np.float32)
                fsm_token = 0  # TRANSIT (unladen return cruise)
                winch_rate_cmd = 0.0
                d_xy_vec = depot_pad[:2] - airframe.p[:2]
                v_horiz_dir = d_xy_vec / max(1e-4, float(np.linalg.norm(d_xy_vec)))
                v_des = np.array([v_horiz_dir[0] * 4.5, v_horiz_dir[1] * 4.5, 0.0], dtype=np.float32)
                if d_depot_xy <= 2.0:
                    sim_regime = "DEPOT_TOUCHDOWN"

            elif sim_regime == "DEPOT_TOUCHDOWN":
                hover_dest = np.array([depot_pad[0], depot_pad[1], 0.06], dtype=np.float32)
                fsm_token = 5  # AUTO_DISPATCH (landing flare)
                winch_rate_cmd = 0.0
                v_des = np.array([0.0, 0.0, -0.35], dtype=np.float32)
                if airframe.p[2] <= 0.12:
                    sim_regime = "OUTBOUND"
                    sim_winch.reload_new_cargo()
                    target_entry = random.choice(world.manifest_addresses)
                    target_pad = target_entry["pos"].copy()
                    roof_z = float(target_pad[2])

            delta_p_clamped = np.clip(hover_dest - airframe.p, -6.0, 6.0)
            a_target_xy = np.array([2.0, 2.0], dtype=np.float32) * delta_p_clamped[:2] + np.array([2.2, 2.2], dtype=np.float32) * (v_des[:2] - airframe.v[:2])
            v_cmd_z = float(np.clip(v_des[2] + 0.90 * delta_p_clamped[2], -1.05, 1.50))
            a_target_z = 3.20 * (v_cmd_z - airframe.v[2])

            mock_peer_info = []
            for po in peer_offsets:
                p_rel = po + np.random.normal(0, 0.2, 3).astype(np.float32)
                v_rel = np.random.normal(0, 0.4, 3).astype(np.float32)
                dist = float(np.linalg.norm(p_rel))
                mock_peer_info.append({"p_rel": p_rel, "v_rel": v_rel, "dist": dist, "has_payload": True})
                if dist < 6.0:
                    n_repel = -p_rel[:2] / max(1e-3, dist)
                    a_target_xy += n_repel * (6.0 - dist) * 1.20

            a_target = np.array([a_target_xy[0], a_target_xy[1], a_target_z], dtype=np.float32)
            guidance_vec = np.clip(a_target / 12.0, -1.0, 1.0).astype(np.float32)

            meas_accel, meas_gyro = airframe_imu.poll(airframe.R, airframe.accel_linear, airframe.omega, airframe.motor_omegas)
            lidar_16 = lidar_engine.cast_360_lidar(airframe.p, math.atan2(airframe.R[1, 0], airframe.R[0, 0]))
            conical_8, wire_clr = lidar_engine.cast_downward_conical(airframe.p)

            hook_a, hook_w = sensor_suite.poll_smart_hook_imu(a_target, np.zeros(3), sim_winch.sway_offset, sim_winch.cable_len, sim_winch.has_payload)
            load_3d = sensor_suite.poll_3axis_load_cell(sim_winch.tension_n, sim_winch.sway_offset, sim_winch.cable_len)
            th_dt, dolp, ndvi = sensor_suite.poll_thermal_and_polarization(airframe.p[:2])
            i_rms, t_mosfet = sensor_suite.poll_esc_foc_telemetry(airframe.motor_omegas, airframe.motor_health)
            alpha, beta, rh, dust = sensor_suite.poll_5port_air_data(airframe.v, episode_wind, airframe.R)
            hdop, cn0, rf_pwr, mag_gr = sensor_suite.poll_rtk_gnss_metrics(airframe.p, is_spoofed_ep)
            _, is_surface_safe, _ = assurance.audit_surface(target_pad[:2], target_z=roof_z)

            obs_122 = synth.synthesize_instantaneous(
                p_est=airframe.p, v_est=airframe.v, R_est=airframe.R, omega_est=airframe.omega,
                target_p=hover_dest, lidar_16=lidar_16, conical_8=conical_8,
                v_tas=float(np.linalg.norm(airframe.v - episode_wind)), q_dyn=12.0, min_sonar=max(0.1, airframe.p[2] - roof_z),
                wind_est=episode_wind, ice_idx=0.35 if ice_active else 0.0, cable_len=sim_winch.cable_len,
                cable_vel=sim_winch.cable_vel, tension_n=sim_winch.tension_n, sway_2d=sim_winch.sway_offset,
                bms_soc=episode_soc, v_pack=episode_v_pack, bms_temp_c=episode_temp_c, r0_dyn=0.0095,
                chi2_val=chi2_ep, jerk_discrepancy=0.5, is_spoofed=is_spoofed_ep, emf_rms=4.0, wire_dist=wire_clr,
                agent_dist=25.0, ground_spl_dba=55.0, has_payload=sim_winch.has_payload,
                hook_a=hook_a, hook_w=hook_w, load_cell_3d=load_3d, thermal_dt=th_dt, dolp=dolp, ndvi=ndvi,
                esc_i_rms=i_rms, esc_t_mosfet=t_mosfet, alpha_deg=alpha, beta_deg=beta, rh_pct=rh, dust_pm10=dust,
                hdop=hdop, cn0_avg=cn0, rf_jam_pwr=rf_pwr, mag_grad=mag_gr,
                airframe_accel=meas_accel, airframe_gyro=meas_gyro,
                peer_swarm_kinematics=mock_peer_info, dt=dt
            )

            obs_366 = synth.stack_history(history_q, obs_122)
            obs_list.append(obs_366)

            y_guidance.append(np.array([guidance_vec[0], guidance_vec[1], guidance_vec[2], 0.0], dtype=np.float32))
            y_fsm.append(fsm_token)
            y_winch.append(np.array([winch_rate_cmd, 1.0 if not sim_winch.has_payload else 0.0], dtype=np.float32))
            y_health.append(airframe.motor_health.copy())
            y_trust.append(np.array([0.0 if is_spoofed_ep else 1.0, 1.0 if is_surface_safe else 0.0], dtype=np.float32))
            y_val.append(np.array([np.clip(-d_dest_xy / 25.0, -1.0, 1.0)], dtype=np.float32))

            sim_winch.step(
                drone_z=airframe.p[2], target_surface_z=roof_z, is_settled=(sim_regime == "ROOF_DELIVERY"),
                is_surface_safe=is_surface_safe, wire_hit=False, agent_breach=False, agent_dist=30.0,
                wind_2d=episode_wind[:2], drone_accel_linear=airframe.accel_linear,
                neural_token_idx=fsm_token, nn_winch_rate=winch_rate_cmd, dt=dt
            )

            obstacles_empty = []
            omega_cmds, _, _ = mixer.mix(
                drone_health=airframe.motor_health, p_est=airframe.p, v_est=airframe.v, R_est=airframe.R,
                omega_est=airframe.omega, p_target=hover_dest, neural_guidance=guidance_vec, token_idx=fsm_token,
                estimated_wind=episode_wind, obstacles=obstacles_empty, ref_vel=v_des,
                cable_len=sim_winch.cable_len, sway_2d=sim_winch.sway_offset, load_cell_3d=load_3d,
                has_payload=sim_winch.has_payload, dt=dt
            )

            curr_ice = 0.35 if ice_active else 0.0
            curr_surf = world.get_surface_height(airframe.p[0], airframe.p[1])
            airframe.step(omega_cmds, episode_wind, 1.225, ice_index=curr_ice, surface_z=curr_surf, has_payload=sim_winch.has_payload, dt=dt)
            step_count += 1

    return (np.array(obs_list, dtype=np.float32),
            np.array(y_guidance, dtype=np.float32),
            np.array(y_fsm, dtype=np.int64),
            np.array(y_winch, dtype=np.float32),
            np.array(y_health, dtype=np.float32),
            np.array(y_trust, dtype=np.float32),
            np.array(y_val, dtype=np.float32))

def harvest_and_train_metropolitan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_SAMPLES = 40000              
    NUM_PRETRAIN_EPOCHS = 5         
    NUM_MAIN_EPOCHS = 15             
    # Training Fix 4: BATCH_SIZE=1026 (exact multiple of 3) & stable learning rate
    BATCH_SIZE = 1026
    LEARNING_RATE = 1.2e-3
    
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per = NUM_SAMPLES // num_workers

    print("\n" + "="*75)
    print(f"HARVESTING {NUM_SAMPLES:,} SAMPLES (STABILIZED ROUND-TRIP DATASET)")
    print(f"Parallel Workers: {num_workers} processes | Compute Device: {device}")
    print("="*75)
    t0 = time.time()

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_harvest_metropolitan, [samples_per] * num_workers)

    X_all = np.concatenate([r[0] for r in results], axis=0)[:NUM_SAMPLES]
    yg_all = np.concatenate([r[1] for r in results], axis=0)[:NUM_SAMPLES]
    yf_all = np.concatenate([r[2] for r in results], axis=0)[:NUM_SAMPLES]
    yw_all = np.concatenate([r[3] for r in results], axis=0)[:NUM_SAMPLES]
    yh_all = np.concatenate([r[4] for r in results], axis=0)[:NUM_SAMPLES]
    yt_all = np.concatenate([r[5] for r in results], axis=0)[:NUM_SAMPLES]
    yv_all = np.concatenate([r[6] for r in results], axis=0)[:NUM_SAMPLES]
    print(f"Dataset compiled in {time.time()-t0:.2f}s! ({len(X_all):,} samples)")

    num_train = int(len(X_all) * 0.90)
    num_train = (num_train // 3) * 3
    num_val = ((len(X_all) - num_train) // 3) * 3

    X_train = torch.from_numpy(X_all[:num_train]).to(device)
    yg_tr = torch.from_numpy(yg_all[:num_train]).to(device)
    yf_tr = torch.as_tensor(yf_all[:num_train], dtype=torch.long, device=device)
    yw_tr = torch.from_numpy(yw_all[:num_train]).to(device)
    yh_tr = torch.from_numpy(yh_all[:num_train]).to(device)
    yt_tr = torch.from_numpy(yt_all[:num_train]).to(device)
    yv_tr = torch.from_numpy(yv_all[:num_train]).to(device)

    X_val = torch.from_numpy(X_all[num_train:num_train+num_val]).to(device)
    yg_va = torch.from_numpy(yg_all[num_train:num_train+num_val]).to(device)
    yf_va = torch.as_tensor(yf_all[num_train:num_train+num_val], dtype=torch.long, device=device)

    model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
    crit_mse = nn.MSELoss()
    crit_bce = nn.BCELoss()

    # Training Fix 3: Inverse Class-Weighted Cross-Entropy (Eliminates 73.5% TRANSIT collapse)
    class_counts = torch.bincount(yf_tr, minlength=NUM_EXECUTIVE_STATES).float()
    class_weights = len(yf_tr) / (len(class_counts) * torch.clamp(class_counts, min=1.0))
    class_weights = (class_weights / class_weights.mean()).to(device)
    crit_ce = nn.CrossEntropyLoss(weight=class_weights)

    num_batches = (num_train + BATCH_SIZE - 1) // BATCH_SIZE

    print("\n" + "-"*75)
    print(f"STAGE 1: 512-DIM RESNET BACKBONE PRE-TRAINING ({NUM_PRETRAIN_EPOCHS} EPOCHS)")
    print("-"*75)

    pre_optimizer = optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1e-4)

    for ep in range(NUM_PRETRAIN_EPOCHS):
        model.train()
        perm = torch.randperm(num_train, device=device)
        ep_loss = 0.0

        for b in range(0, num_train, BATCH_SIZE):
            idx = perm[b : b + BATCH_SIZE]
            bx, byg, byv = X_train[idx], yg_tr[idx], yv_tr[idx]

            pre_optimizer.zero_grad(set_to_none=True)
            p_g, _, _, _, _, p_v = model(bx)
            loss_pre = crit_mse(p_g, byg) + 0.15 * crit_mse(p_v, byv)
            loss_pre.backward()

            # Training Fix 2: Gradient Norm Clipping to prevent layer blowup
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            pre_optimizer.step()
            ep_loss += loss_pre.item()

        print(f"Pre-train Epoch [{ep+1:02d}/{NUM_PRETRAIN_EPOCHS:02d}] - Kinematic MSE: {ep_loss/num_batches:.5f}")

    print("\n" + "-"*75)
    print(f"STAGE 2: MULTI-TASK 6-HEAD SUPERVISED TRAINING ({NUM_MAIN_EPOCHS} EPOCHS)")
    print("-"*75)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    total_steps = num_batches * NUM_MAIN_EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=total_steps)

    for epoch in range(NUM_MAIN_EPOCHS):
        model.train()
        perm = torch.randperm(num_train, device=device)
        total_loss = 0.0

        for b in range(0, num_train, BATCH_SIZE):
            idx = perm[b : b + BATCH_SIZE]
            bx, byg, byf = X_train[idx], yg_tr[idx], yf_tr[idx]
            byw, byh, byt, byv = yw_tr[idx], yh_tr[idx], yt_tr[idx], yv_tr[idx]

            optimizer.zero_grad(set_to_none=True)
            p_g, p_f, p_w, p_h, p_t, p_v = model(bx)

            loss = (crit_mse(p_g, byg) +
                    0.40 * crit_ce(p_f, byf) +
                    0.25 * crit_mse(p_w, byw) +
                    0.30 * crit_bce(p_h, byh) +
                    0.20 * crit_bce(p_t, byt) +
                    0.10 * crit_mse(p_v, byv))

            loss.backward()
            # Training Fix 2: Gradient Norm Clipping on multi-head backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_preds_g = []
            val_preds_f = []
            for vb in range(0, num_val, 300):
                v_bx = X_val[vb:vb+300]
                vg_sub, vf_sub, _, _, _, _ = model(v_bx)
                val_preds_g.append(vg_sub)
                val_preds_f.append(vf_sub)

            vg = torch.cat(val_preds_g, dim=0)
            vf = torch.cat(val_preds_f, dim=0)
            val_mse = crit_mse(vg, yg_va).item()
            val_acc = (vf.argmax(dim=-1) == yf_va).float().mean().item() * 100.0

        print(f"Epoch [{epoch+1:02d}/{NUM_MAIN_EPOCHS:02d}] - Multi-Task Loss: {total_loss/num_batches:.4f} | Val MSE: {val_mse:.5f} | Decision Acc: {val_acc:.1f}%")

    torch.save(model.state_dict(), MODEL_WEIGHTS_FILE)
    print("\n" + "="*75)
    print(f"SUCCESS: Stabilized Model saved to {MODEL_WEIGHTS_FILE} in {time.time()-t0:.2f}s!")
    print("="*75 + "\n")
    return model, device

# ==========================================
# 16. CHASE PROJECTION & RADAR MINIMAP
# ==========================================
def project_3d_chase(pt, cam_focus, cam_yaw, cam_pitch, center_screen, scale=28.0):
    rel = pt - cam_focus
    x1 = rel[0]*math.cos(cam_yaw) - rel[1]*math.sin(cam_yaw)
    y1 = rel[0]*math.sin(cam_yaw) + rel[1]*math.cos(cam_yaw)
    y2 = y1*math.cos(cam_pitch) - rel[2]*math.sin(cam_pitch)
    z2 = y1*math.sin(cam_pitch) + rel[2]*math.cos(cam_pitch)

    if y2 >= 13.5:
        return None, None, y2

    focal = max(2.5, 14.0 - y2)
    sx = int(center_screen[0] + (x1 / focal) * scale * 9.5)
    sy = int(center_screen[1] - (z2 / focal) * scale * 9.5)
    return sx, sy, y2

def screen_to_world_raycast_exact(mouse_pos, cam_focus, cam_yaw, cam_pitch, center_screen, world, scale=28.0):
    mx, my = mouse_pos
    dx = (mx - center_screen[0]) / (scale * 9.5)
    dz = -(my - center_screen[1]) / (scale * 9.5)

    cp, sp = math.cos(cam_pitch), math.sin(cam_pitch)
    cy, sy = math.cos(cam_yaw), math.sin(cam_yaw)

    R_view = np.array([
        [cy, -sy, 0.0],
        [cp*sy, cp*cy, -sp],
        [sp*sy, sp*cy,  cp]
    ], dtype=np.float32)

    cam_eye = cam_focus + R_view.T @ np.array([0.0, -14.0, 0.0], dtype=np.float32)

    ray_dir_cam = np.array([dx, 1.0, dz], dtype=np.float32)
    ray_dir_cam /= max(1e-4, float(np.linalg.norm(ray_dir_cam)))
    ray_dir_world = R_view.T @ ray_dir_cam
    ray_dir_world /= max(1e-4, float(np.linalg.norm(ray_dir_world)))

    best_hit = None
    min_t = 1e6
    for h in sorted(world.houses, key=lambda b: b["aabb"][4], reverse=True):
        x1, x2, y1, y2, hh = h["aabb"]
        if abs(ray_dir_world[2]) > 1e-4:
            t = (hh - cam_eye[2]) / ray_dir_world[2]
            if 0.0 < t < min_t:
                hx = cam_eye[0] + t * ray_dir_world[0]
                hy = cam_eye[1] + t * ray_dir_world[1]
                if (x1 - 1.5) <= hx <= (x2 + 1.5) and (y1 - 1.5) <= hy <= (y2 + 1.5):
                    min_t = t
                    best_hit = np.array([(x1+x2)/2.0, (y1+y2)/2.0, hh], dtype=np.float32)

    if best_hit is not None:
        return best_hit

    if abs(ray_dir_world[2]) > 1e-4:
        t_gnd = -cam_eye[2] / ray_dir_world[2]
        if t_gnd > 0.0:
            return np.array([
                np.clip(cam_eye[0] + t_gnd * ray_dir_world[0], -195.0, 195.0),
                np.clip(cam_eye[1] + t_gnd * ray_dir_world[1], -195.0, 195.0),
                0.0
            ], dtype=np.float32)

    return cam_focus.copy()

def draw_tactical_radar_minimap(screen, map_rect, world, drones, active_idx, vehicles):
    rx, ry, rw, rh = map_rect
    pygame.draw.rect(screen, (15, 20, 28), map_rect)
    pygame.draw.rect(screen, (0, 220, 255), map_rect, 2)
    scale = rw / 400.0  
    cx, cy = rx + rw // 2, ry + rh // 2

    def to_map(x, y): return int(cx + x * scale), int(cy - y * scale)

    for r in world.roads:
        p1, p2 = to_map(r["x_min"], r["y_min"]), to_map(r["x_max"], r["y_max"])
        pygame.draw.rect(screen, (28, 34, 44), (min(p1[0], p2[0]), min(p1[1], p2[1]), abs(p2[0]-p1[0])+1, abs(p2[1]-p1[1])+1))

    for h in world.houses:
        x1, x2, y1, y2, _ = h["aabb"]
        mp1, mp2 = to_map(x1, y2), to_map(x2, y1)
        pygame.draw.rect(screen, (55, 65, 80), (mp1[0], mp1[1], mp2[0]-mp1[0], mp2[1]-mp1[1]))

    for v in vehicles:
        vx, vy = to_map(v["pos"][0], v["pos"][1])
        pygame.draw.circle(screen, (255, 140, 40), (vx, vy), 2)

    for pad in world.depot_helipads:
        bx, by = to_map(pad[0], pad[1])
        pygame.draw.circle(screen, (0, 255, 140), (bx, by), 4, 1)

    colors = [(0, 240, 255), (255, 180, 40), (220, 100, 255)]
    for d_i, d in enumerate(drones):
        dx, dy = to_map(d.airframe.p[0], d.airframe.p[1])
        c = colors[d_i % len(colors)]
        radius = 5 if d_i == active_idx else 3
        pygame.draw.circle(screen, c, (dx, dy), radius)

        tx, ty = to_map(d.target[0], d.target[1])
        pygame.draw.circle(screen, c, (tx, ty), 3, 1)

# =============================================================================
# 17. SWARM DRONE INSTANCE WITH LOOKAHEAD TRACKER & 8-STAGE MISSION FSM
# =============================================================================
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
        self.imu = Airframe6DOFMEMS_IMU()

        self.history_q = deque(maxlen=HISTORY_LEN)

        self.last_gps_p = None
        self.gps_v_filt = np.zeros(3, dtype=np.float32)
        self.gps_a_filt = np.zeros(3, dtype=np.float32)

        self.target = init_pos.copy()
        self.home_coords = world.depot_helipads[drone_id % len(world.depot_helipads)].copy()
        self.assigned_cruise_alt = 30.0 + (drone_id * 5.0)

        # 8-Stage Mission State Machine
        self.mission_phase = "DELIVERY_OUTBOUND"
        self.manifest_idx = drone_id * 14
        self.trail = deque(maxlen=160)
        self.packages_delivered = 0
        self.hover_settle_timer = 0.0
        self.depot_timer = 0.0
        
        # Rooftop Departure Immunity tracking
        self.last_delivery_bldg_id = None
        self.climbout_target_z = 0.0

        self.corridor_planner = VoxelOctreeCorridorPlanner(world)
        self.spline_traj = None
        self.traj_time = 0.0

    def compute_corridor_trajectory(self, dest, is_depot_landing=False):
        hover_dest = np.array([dest[0], dest[1], dest[2] + 2.5], dtype=np.float32)
        waypoints = self.corridor_planner.find_safe_corridor_waypoints(
            self.airframe.p, hover_dest, 
            assigned_cruise_alt=self.assigned_cruise_alt,
            is_depot_landing=is_depot_landing
        )
        self.spline_traj = MinimumSnapQuinticTrajectory(waypoints, avg_speed=5.0)
        self.traj_time = 0.0

# =============================================================================
# 18. MAIN FLIGHT COCKPIT & 8-STAGE AUTONOMOUS SIMULATION LOOP
# =============================================================================
def run_all_physics_simulator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(MODEL_WEIGHTS_FILE):
        model, device = harvest_and_train_metropolitan()
    else:
        model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
        try:
            model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE, map_location=device))
            print(f"\nLoaded verified 366-Dim Round-Trip Executive from {MODEL_WEIGHTS_FILE}!")
        except Exception:
            print("[WARNING] Weights file mismatch or corrupted. Retraining on Round-Trip Dataset...")
            model, device = harvest_and_train_metropolitan()

    model.eval()
    pygame.init()
    VIEW_W, VIEW_H = 760, 740
    SIDEBAR_W = 440
    W, H = VIEW_W + SIDEBAR_W, VIEW_H

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Survival Core V35.1 — Round-Trip Extended Swarm Fleet")
    clock = pygame.time.Clock()

    font_large = pygame.font.SysFont("consolas", 17, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 12, bold=True)

    world = MetropolitanCityGridWorld400m()
    weather_engine = DynamicUrbanWeatherEngine()
    traffic_engine = DynamicRoadVehicleTrafficEngine(world.avenue_x, world.street_y)

    sensors_physics = FullPhysicsSensorSuite()
    assurance = SurfaceAndWireAssuranceSuite(world)
    synth = DroneObservationSynthesizer122Dim()
    mixer = RateDampedActuatorMixerWithCBF()

    lidar_suite = MacroPerceptionLiDARSuite(world)
    sensor_suite = AdvancedMultiModalSensorSuite(world)
    ground_agent = DynamicGroundIntruderAgent(init_pos=[-80.0, 80.0, 0.0])
    dryden_wind_gen = DrydenAtmosphericTurbulenceModel(w20_m_s=3.4)

    drones = []
    swarm_colors = [(0, 240, 255), (255, 180, 40), (220, 100, 255)]
    for i in range(NUM_SWARM_DRONES):
        init_pos = world.depot_helipads[i].copy()
        init_pos[2] = 2.0
        d_inst = SwarmDroneInstance(i, init_pos, world)
        order = world.manifest_addresses[d_inst.manifest_idx % len(world.manifest_addresses)]
        d_inst.target = order["pos"].copy()
        d_inst.compute_corridor_trajectory(d_inst.target)
        drones.append(d_inst)

    selected_drone_idx = 0
    cam_yaw, cam_pitch = 0.75, 0.45
    center_viewport = (VIEW_W // 2, VIEW_H // 2 + 50)
    cam_focus = drones[0].airframe.p.copy()
    running = True

    gust_timer = 0.0
    active_gust = np.zeros(3, dtype=np.float32)
    inject_spoof = False
    inject_multipath = False
    flight_time = 0.0
    delivered_boxes = deque(maxlen=120)

    wind_particles = [
        np.array([random.uniform(-180.0, 180.0), random.uniform(-180.0, 180.0), random.uniform(1.0, 35.0)], dtype=np.float32)
        for _ in range(55)
    ]

    print("\n================ V35.1 ROUND-TRIP EXTENDED SIMULATOR RUNNING ================")
    print(" [HARVESTER] : Full round-trip dataset (Outbound, Winch, Climb, RTB, Landing)")
    print(" [TRAINING]  : Symmetrical regimes (30% Outbound, 15% Winch, 15% Climb, 25% RTB, 15% Land)")
    print(" [AUTONOMY]  : Lookahead carrot-chasing path tracker (Zero mid-city diving)")
    print(" [SAFETY]    : Authority separation (Spline = Altitude, Net = Horizontal Steering)")
    print(" [PHYSICS]   : Continuous pad touchdown (Z <= 0.10m), battery recharge & reload")
    print("=============================================================================\n")

    while running:
        dt = 0.016
        flight_time += dt

        weather_engine.step(dt=dt)
        traffic_engine.step(dt=dt)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    selected_drone_idx = (selected_drone_idx + 1) % NUM_SWARM_DRONES
                    print(f">>> [FOCUS] SWITCHED TO SWARM DRONE #{selected_drone_idx + 1} <<<")
                elif event.key == pygame.K_w:
                    weather_engine.cycle_weather()
                elif pygame.K_1 <= event.key <= pygame.K_8:
                    m_idx = event.key - pygame.K_1
                    cur_d = drones[selected_drone_idx]
                    cur_d.airframe.motor_health[m_idx] = 0.0 if cur_d.airframe.motor_health[m_idx] == 1.0 else 1.0
                elif event.key == pygame.K_a:
                    ground_agent.is_leashed = not ground_agent.is_leashed
                elif event.key == pygame.K_h:
                    cur_d = drones[selected_drone_idx]
                    cur_d.target = cur_d.home_coords.copy()
                    cur_d.compute_corridor_trajectory(cur_d.target, is_depot_landing=True)
                    cur_d.mission_phase = "RETURNING_TO_BASE"
                    cur_d.hover_settle_timer = 0.0
                elif event.key == pygame.K_b:
                    drones[selected_drone_idx].bms.soc = 0.18
                elif event.key == pygame.K_c:
                    drones[selected_drone_idx].bms.temp_c = -5.0
                elif event.key == pygame.K_i:
                    d = drones[selected_drone_idx]
                    d.icing.icing_active = not d.icing.icing_active
                elif event.key == pygame.K_m:
                    inject_multipath = not inject_multipath
                elif event.key == pygame.K_j:
                    inject_spoof = not inject_spoof
                elif event.key == pygame.K_g:
                    gust_timer = 2.5
                    ang = random.uniform(0, 2*math.pi)
                    active_gust = np.array([math.cos(ang)*8.5, math.sin(ang)*8.5, 0.0], dtype=np.float32)
                elif event.key == pygame.K_p:
                    drones[selected_drone_idx].airframe.parachute_deployed = True
                elif event.key == pygame.K_r:
                    for d in drones:
                        d.airframe.motor_health.fill(1.0)
                        d.airframe.parachute_deployed = False
                        d.bms.soc = 0.96
                        d.bms.temp_c = 22.0
                        d.icing.ice_thickness_mm = 0.0
                        d.winch.reload_new_cargo()
                        d.mission_phase = "DELIVERY_OUTBOUND"
                        d.last_delivery_bldg_id = None
                    inject_spoof = False
                    inject_multipath = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if mx < VIEW_W:
                    click_world = screen_to_world_raycast_exact((mx, my), cam_focus, cam_yaw, cam_pitch, center_viewport, world)
                    safe_pad = assurance.snap_to_nearest_safe_ground(click_world[:2])
                    cur_d = drones[selected_drone_idx]
                    cur_d.target = safe_pad
                    cur_d.compute_corridor_trajectory(safe_pad)
                    cur_d.hover_settle_timer = 0.0
                    cur_d.winch.reload_new_cargo()
                    cur_d.mission_phase = "DELIVERY_OUTBOUND"
                    cur_d.last_delivery_bldg_id = None

                    for m_idx, entry in enumerate(world.manifest_addresses):
                        if np.linalg.norm(entry["pos"][:2] - safe_pad[:2]) < 1.0:
                            cur_d.manifest_idx = m_idx
                            break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: cam_yaw -= 0.03
        if keys[pygame.K_RIGHT]: cam_yaw += 0.03
        if keys[pygame.K_UP]: cam_pitch = min(1.2, cam_pitch + 0.02)
        if keys[pygame.K_DOWN]: cam_pitch = max(0.15, cam_pitch - 0.02)

        cam_focus += (drones[selected_drone_idx].airframe.p - cam_focus) * 5.0 * dt

        active_drone = drones[selected_drone_idx]
        eff_wind, gust_scale, amb_temp, auto_ice = weather_engine.get_weather_parameters(active_drone.airframe.p, world.avenue_x)
        dryden_gust = dryden_wind_gen.step(altitude_m=active_drone.airframe.p[2], airspeed_m_s=np.linalg.norm(active_drone.airframe.v), dt=dt) * (gust_scale / 2.0)
        microburst = active_gust * (gust_timer / 2.5) if gust_timer > 0.0 else np.zeros(3, dtype=np.float32)
        if gust_timer > 0.0: gust_timer = max(0.0, gust_timer - dt)

        total_wind = eff_wind + dryden_gust + microburst

        for wp in wind_particles:
            wp += total_wind * 0.45 * dt
            if wp[0] > 200.0: wp[0] = -200.0
            if wp[0] < -200.0: wp[0] = 200.0
            if wp[1] > 200.0: wp[1] = -200.0
            if wp[1] < -200.0: wp[1] = 200.0

        active_tip = active_drone.airframe.p + np.array([0, 0, -0.08 - active_drone.winch.cable_len])
        agent_pos, agent_breach, agent_dist = ground_agent.step(
            cargo_world_pos=active_tip,
            is_winch_lowering=(active_drone.winch.state == "LOWERING"),
            is_aborting=(active_drone.winch.state == "ABORT_RETRACT"),
            dt=dt
        )

        obstacles_cbf = [{"pos": np.array([(h["aabb"][0]+h["aabb"][1])/2, (h["aabb"][2]+h["aabb"][3])/2, 0.0]), "radius": 11.5, "height": h["aabb"][4], "id": h["id"]} for h in world.houses]
        obstacles_cbf.extend(traffic_engine.get_cbf_obstacles())
        swarm_states = [{"p": d.airframe.p.copy(), "v": d.airframe.v.copy()} for d in drones]

        selected_diag = {}

        # =====================================================================
        # STEP ALL SWARM DRONES (HIGH-PERFORMANCE BATCHED INFERENCE)
        # =====================================================================
        swarm_obs_366 = []
        pre_step_states = []

        # --- Phase 1: Parallel Sensor Polling & 122-Dim Assembly ---
        for d_idx, d in enumerate(drones):
            d.icing.icing_active = auto_ice or d.icing.icing_active
            ice_mm, ice_idx = d.icing.step(airspeed=float(np.linalg.norm(d.airframe.v - total_wind)), dt=dt)

            drone_yaw = math.atan2(d.airframe.R[1, 0], d.airframe.R[0, 0])
            lidar_16 = lidar_suite.cast_360_lidar(d.airframe.p, drone_yaw, traffic_engine.vehicles)
            conical_8, wire_clearance = lidar_suite.cast_downward_conical(d.airframe.p)
            wire_hit = (wire_clearance < 1.4)

            current_surf_z = world.get_surface_height(d.airframe.p[0], d.airframe.p[1])
            target_surface_z = float(d.target[2])

            surf_class, is_surf_safe, _ = assurance.audit_surface(d.target[:2], target_z=target_surface_z)
            s_data = sensors_physics.poll_all_physics_sensors(d.airframe.p, d.airframe.v, total_wind, surface_z=current_surf_z)

            meas_accel, meas_gyro = d.imu.poll(d.airframe.R, d.airframe.accel_linear, d.airframe.omega, d.airframe.motor_omegas)

            accel_inertial_true = (d.airframe.accel_linear - GRAVITY_G)
            a_quantum, phase_shift = d.quantum.measure_phase_shift(accel_inertial_true)
            d.ekf.predict(d.airframe.R.T @ a_quantum, meas_gyro, dt=dt)

            compass_yaw = math.atan2(d.airframe.R[1, 0], d.airframe.R[0, 0]) + float(np.random.normal(0, 0.008))
            d.ekf.update_heading(compass_yaw, r_cov=0.02)

            H_baro = np.zeros((1, 15), dtype=np.float32); H_baro[0, 2] = 1.0
            d.ekf.update_measurement(H_baro, np.array([d.airframe.p[2] + np.random.normal(0, 0.04) - d.ekf.p[2]], dtype=np.float32), np.array([[0.05]], dtype=np.float32))

            spoof_offset = np.array([3.5, 2.8, 0.5]) if (inject_spoof and d_idx == selected_drone_idx) else np.zeros(3)
            multipath_offset = np.array([4.2, -3.1, 0.8]) if (inject_multipath and d_idx == selected_drone_idx) else np.zeros(3)
            z_gps = d.airframe.p + multipath_offset + spoof_offset + np.random.normal(0, 0.08, 3).astype(np.float32)

            if d.last_gps_p is None:
                d.last_gps_p = z_gps.copy()
                d.gps_v_filt = d.airframe.v.copy()
                d.gps_a_filt.fill(0.0)

            alpha_v = dt / (0.45 + dt)
            v_gps_raw = (z_gps - d.last_gps_p) / dt
            d.last_gps_p = z_gps.copy()
            
            v_prev_filt = d.gps_v_filt.copy()
            d.gps_v_filt += alpha_v * (v_gps_raw - d.gps_v_filt)
            
            alpha_a = dt / (0.35 + dt)
            a_gps_raw = (d.gps_v_filt - v_prev_filt) / dt
            d.gps_a_filt += alpha_a * (a_gps_raw - d.gps_a_filt)

            a_inertial_measured = a_quantum + GRAVITY_G
            jerk_discrepancy = float(np.linalg.norm(d.gps_a_filt - a_inertial_measured))
            kinematic_violation = (jerk_discrepancy > 16.0)

            H_pos = np.zeros((3, 15), dtype=np.float32); H_pos[0:3, 0:3] = np.eye(3)
            y_innov = z_gps - d.ekf.p
            S_cov = H_pos @ d.ekf.P @ H_pos.T + np.eye(3, dtype=np.float32) * 0.08
            chi2_dist = float(y_innov.T @ np.linalg.inv(S_cov) @ y_innov)

            is_attack_detected = (chi2_dist > 18.0) or kinematic_violation or (inject_spoof and d_idx == selected_drone_idx) or (inject_multipath and d_idx == selected_drone_idx)

            if not is_attack_detected:
                d.ekf.update_measurement(H_pos, y_innov, np.eye(3, dtype=np.float32) * 0.05)
                gnss_status = "GNSS NOMINAL"
            else:
                cam_optical_drift = np.random.normal(0, 0.015, 3).astype(np.float32)
                p_vio_fused = d.airframe.p + cam_optical_drift
                y_vio = p_vio_fused - d.ekf.p
                d.ekf.update_measurement(H_pos, y_vio, np.eye(3, dtype=np.float32) * 0.02)
                gnss_status = "SPOOF REJECTED: QUANTUM VIO LOCK"

            # Battery temperature floating dynamically with internal Joule heating and convective cooling
            bms_soc, v_pack, i_draw, r0_dyn = d.bms.step(d.airframe.last_p_watts, amb_temp=amb_temp, dt=dt)

            # Lookahead "Carrot-Chasing" Path Follower: clock progression tied to physical progress
            if d.spline_traj is not None:
                p_curr_ref, _, _ = d.spline_traj.evaluate(d.traj_time)
                lag_dist = float(np.linalg.norm(d.airframe.p[:2] - p_curr_ref[:2]))
                
                if lag_dist < 2.2:
                    d.traj_time += dt
                elif lag_dist < 4.5:
                    d.traj_time += dt * 0.35
                else:
                    d.traj_time += 0.0

                p_ref, v_ref, a_ref = d.spline_traj.evaluate(d.traj_time)
            else:
                p_ref = np.array([d.target[0], d.target[1], target_surface_z + 2.5], dtype=np.float32)
                v_ref, a_ref = np.zeros(3), np.zeros(3)

            # Descent prohibited until physically within 4.0m of the target pad
            dist_to_final_xy = float(np.linalg.norm(d.airframe.p[:2] - d.target[:2]))
            if dist_to_final_xy > 4.0 and p_ref[2] < (d.assigned_cruise_alt - 1.0) and d.mission_phase not in ["DEPOT_TOUCHDOWN", "CLIMBOUT", "DEPOT_SERVICE"]:
                p_ref[2] = max(p_ref[2], d.assigned_cruise_alt)
                v_ref[2] = max(0.0, v_ref[2])

            R_est = quat_to_rot(d.ekf.q)
            omega_est = meas_gyro - d.ekf.b_g
            d_horiz = float(np.linalg.norm(d.ekf.p[:2] - d.target[:2]))
            d_vert = abs(float(d.ekf.p[2] - (target_surface_z + 2.5)))
            v_horiz = float(np.linalg.norm(d.ekf.v[:2]))

            is_centered = (d_horiz <= HOVER_LOCK_RADIUS_M) and (d_vert <= 1.4) and (v_horiz <= HOVER_LOCK_SPEED_MAX_MS)
            if is_centered and d.winch.has_payload and d.mission_phase in ["DELIVERY_OUTBOUND", "HOVER_LOCK"]:
                d.hover_settle_timer += dt
            else:
                if d.winch.cable_len <= 0.10: d.hover_settle_timer = 0.0

            is_settled_for_winch = (d.hover_settle_timer >= HOVER_SETTLE_DURATION_S)

            hook_a, hook_w = sensor_suite.poll_smart_hook_imu(d.airframe.accel_linear, d.airframe.omega, d.winch.sway_offset, d.winch.cable_len, d.winch.has_payload)
            load_3d = sensor_suite.poll_3axis_load_cell(d.winch.tension_n, d.winch.sway_offset, d.winch.cable_len)
            th_dt, dolp, ndvi = sensor_suite.poll_thermal_and_polarization(d.target[:2])
            
            # --- FIXED LINE: Use d.airframe.motor_health directly ---
            i_rms, t_mosfet = sensor_suite.poll_esc_foc_telemetry(d.airframe.motor_omegas, d.airframe.motor_health)
            
            alpha, beta, rh, dust = sensor_suite.poll_5port_air_data(d.airframe.v, total_wind, d.airframe.R)
            hdop, cn0, rf_pwr, mag_gr = sensor_suite.poll_rtk_gnss_metrics(d.airframe.p, is_attack_detected)

            peer_swarm_kinematics = []
            for j_idx, other_d in enumerate(drones):
                if j_idx == d_idx: continue
                p_rel = other_d.airframe.p - d.airframe.p
                v_rel = other_d.airframe.v - d.airframe.v
                p_dist = float(np.linalg.norm(p_rel))
                peer_swarm_kinematics.append({
                    "p_rel": p_rel,
                    "v_rel": v_rel,
                    "dist": p_dist,
                    "has_payload": other_d.winch.has_payload
                })

            obs_goal = p_ref if (d.spline_traj is not None and dist_to_final_xy > 5.0) else np.array([d.target[0], d.target[1], target_surface_z + 2.5], dtype=np.float32)

            obs_122 = synth.synthesize_instantaneous(
                p_est=d.ekf.p, v_est=d.ekf.v, R_est=R_est, omega_est=omega_est, target_p=obs_goal,
                lidar_16=lidar_16, conical_8=conical_8, v_tas=s_data["v_tas"], q_dyn=s_data["q_dyn"],
                min_sonar=s_data["min_sonar"], wind_est=total_wind, ice_idx=ice_idx,
                cable_len=d.winch.cable_len, cable_vel=d.winch.cable_vel, tension_n=d.winch.tension_n,
                sway_2d=d.winch.sway_offset, bms_soc=bms_soc, v_pack=v_pack, bms_temp_c=d.bms.temp_c,
                r0_dyn=r0_dyn, chi2_val=chi2_dist, jerk_discrepancy=jerk_discrepancy, is_spoofed=is_attack_detected,
                emf_rms=s_data["emf_rms"], wire_dist=wire_clearance,
                agent_dist=agent_dist if d_idx == selected_drone_idx else 30.0, ground_spl_dba=58.0,
                has_payload=d.winch.has_payload, hook_a=hook_a, hook_w=hook_w, load_cell_3d=load_3d,
                thermal_dt=th_dt, dolp=dolp, ndvi=ndvi, esc_i_rms=i_rms, esc_t_mosfet=t_mosfet,
                alpha_deg=alpha, beta_deg=beta, rh_pct=rh, dust_pm10=dust, hdop=hdop, cn0_avg=cn0,
                rf_jam_pwr=rf_pwr, mag_grad=mag_gr, airframe_accel=meas_accel, airframe_gyro=meas_gyro,
                peer_swarm_kinematics=peer_swarm_kinematics, dt=dt
            )

            obs_366 = synth.stack_history(d.history_q, obs_122)
            swarm_obs_366.append(obs_366)

            pre_step_states.append({
                "target_surface_z": target_surface_z,
                "is_surf_safe": is_surf_safe,
                "wire_hit": wire_hit,
                "is_settled_for_winch": is_settled_for_winch,
                "p_ref": p_ref, "v_ref": v_ref, "a_ref": a_ref,
                "R_est": R_est, "omega_est": omega_est, "load_3d": load_3d,
                "current_surf_z": current_surf_z, "v_pack": v_pack, "bms_soc": bms_soc,
                "ice_mm": ice_mm, "ice_idx": ice_idx, "meas_accel": meas_accel,
                "meas_gyro": meas_gyro, "gnss_status": gnss_status,
                "jerk_discrepancy": jerk_discrepancy, "phase_shift": phase_shift,
                "i_rms": i_rms, "t_mosfet": t_mosfet, "th_dt": th_dt, "dolp": dolp,
                "ndvi": ndvi, "hdop": hdop, "cn0": cn0, "alpha": alpha, "beta": beta,
                "dust": dust, "s_data": s_data, "v_horiz": v_horiz,
                "dist_to_final_xy": dist_to_final_xy
            })

        # --- Phase 2: Single Batched Cross-Attention Forward Pass ---
        t_inf0 = time.time()
        with torch.no_grad():
            t_in_batch = torch.from_numpy(np.stack(swarm_obs_366)).float().to(device)
            nn_guidance_batch, nn_logits_batch, nn_winch_batch, nn_health_batch, nn_trust_batch, nn_val_batch = model(t_in_batch)

            # Single unified host-device transfer per frame (Zero pipeline stalls)
            guidance_all = nn_guidance_batch.cpu().numpy()
            probs_all = F.softmax(nn_logits_batch, dim=-1).cpu().numpy()
            winch_all = nn_winch_batch[:, 0].cpu().numpy()
            val_all = nn_val_batch[:, 0].cpu().numpy()

        policy_latency_ms = (time.time() - t_inf0) * 1000.0

        # --- Phase 3: Deterministic 8-Stage Mission FSM & Actuation ---
        for d_idx, d in enumerate(drones):
            st = pre_step_states[d_idx]
            neural_guidance = guidance_all[d_idx]
            probs = probs_all[d_idx]
            token_idx = int(np.argmax(probs))
            nn_winch_cmd = float(winch_all[d_idx])
            critic_v = float(val_all[d_idx])

            current_token_idx = token_idx
            current_action_name = ACTION_NAMES[token_idx]
            confidence = float(probs[token_idx])

            d.winch.step(
                drone_z=d.airframe.p[2],
                target_surface_z=st["target_surface_z"],
                is_settled=st["is_settled_for_winch"],
                is_surface_safe=st["is_surf_safe"],
                wire_hit=st["wire_hit"],
                agent_breach=(agent_breach if (d_idx == selected_drone_idx and st["target_surface_z"] < 2.0) else False),
                agent_dist=(agent_dist if d_idx == selected_drone_idx else 30.0),
                wind_2d=total_wind[:2],
                drone_accel_linear=d.airframe.accel_linear,
                neural_token_idx=current_token_idx,
                nn_winch_rate=nn_winch_cmd,
                dt=dt
            )

            # =========================================================
            # DETERMINISTIC 8-STAGE MISSION STATE MACHINE
            # =========================================================
            dist_to_dest_xy = float(np.linalg.norm(d.airframe.p[:2] - d.target[:2]))
            dist_to_depot_xy = float(np.linalg.norm(d.airframe.p[:2] - d.home_coords[:2]))

            # Stage 1: Outbound transit -> Hover Lock
            if d.mission_phase == "DELIVERY_OUTBOUND":
                if dist_to_dest_xy <= 1.40 and st["v_horiz"] < 0.90:
                    d.mission_phase = "HOVER_LOCK"

            # Stage 2: Hover Lock -> Winch Lower
            elif d.mission_phase == "HOVER_LOCK":
                if st["is_settled_for_winch"]:
                    d.mission_phase = "WINCH_DELIVERY"

            # Stage 3: Winch Delivery (Lower -> Touchdown -> Retract -> Climbout)
            elif d.mission_phase == "WINCH_DELIVERY":
                if d.winch.delivery_completed:
                    d.winch.delivery_completed = False
                    d.packages_delivered += 1
                    delivered_boxes.append(np.array([d.target[0], d.target[1], st["target_surface_z"]], dtype=np.float32))
                    d.hover_settle_timer = 0.0

                    # Rooftop Departure Masking: exempt current rooftop from CBF obstacle repulsion
                    for h in world.houses:
                        x1, x2, y1, y2, _ = h["aabb"]
                        if x1 <= d.target[0] <= x2 and y1 <= d.target[1] <= y2:
                            d.last_delivery_bldg_id = h["id"]
                            break

                    d.climbout_target_z = max(d.airframe.p[2] + 4.0, st["current_surf_z"] + 4.5)
                    d.mission_phase = "CLIMBOUT"

            # Stage 4: Rooftop Vertical Climbout
            elif d.mission_phase == "CLIMBOUT":
                st["p_ref"] = np.array([d.airframe.p[0], d.airframe.p[1], d.climbout_target_z], dtype=np.float32)
                st["v_ref"] = np.array([0.0, 0.0, 1.20], dtype=np.float32)
                st["a_ref"] = np.zeros(3, dtype=np.float32)

                if d.airframe.p[2] >= (d.climbout_target_z - 0.5):
                    d.target = d.home_coords.copy()
                    d.compute_corridor_trajectory(d.target, is_depot_landing=True)
                    d.mission_phase = "RETURNING_TO_BASE"

            # Stage 5: Return-to-Base (RTB) Transit Cruise
            elif d.mission_phase == "RETURNING_TO_BASE":
                # Clear departure exemption once safely at cruise altitude
                if d.last_delivery_bldg_id is not None and d.airframe.p[2] > (d.assigned_cruise_alt - 2.0):
                    d.last_delivery_bldg_id = None

                if dist_to_depot_xy <= 2.20 and d.airframe.p[2] <= 5.0:
                    d.mission_phase = "DEPOT_APPROACH"

            # Stage 6: Depot Approach & Terminal Pad Queuing Stack
            elif d.mission_phase == "DEPOT_APPROACH":
                pad_occupied = False
                for other_i, other_d in enumerate(drones):
                    if other_i == d_idx: continue
                    other_to_pad = float(np.linalg.norm(other_d.airframe.p[:2] - d.home_coords[:2]))
                    if other_to_pad < 2.5 and other_d.airframe.p[2] < 5.0:
                        pad_occupied = True
                        break

                if not pad_occupied:
                    d.mission_phase = "DEPOT_TOUCHDOWN"
                else:
                    # Hold in terminal hover stack at 4.5m until pad is clear
                    st["p_ref"] = np.array([d.home_coords[0], d.home_coords[1], 4.5], dtype=np.float32)
                    st["v_ref"] = np.zeros(3, dtype=np.float32)

            # Stage 7: Soft Helipad Descent Flare & Touchdown
            elif d.mission_phase == "DEPOT_TOUCHDOWN":
                st["p_ref"] = np.array([d.home_coords[0], d.home_coords[1], 0.05], dtype=np.float32)
                st["v_ref"] = np.array([0.0, 0.0, -0.35], dtype=np.float32)
                st["a_ref"] = np.zeros(3, dtype=np.float32)

                # Touchdown confirmed (Z <= 0.12m, skids firmly on pad)
                if d.airframe.p[2] <= 0.12 and abs(d.airframe.v[2]) < 0.35:
                    d.depot_timer = 0.0
                    d.mission_phase = "DEPOT_SERVICE"

            # Stage 8: Turnaround Servicing, Cargo Reload & Battery Reset
            elif d.mission_phase == "DEPOT_SERVICE":
                st["p_ref"] = np.array([d.home_coords[0], d.home_coords[1], 0.05], dtype=np.float32)
                st["v_ref"] = np.zeros(3, dtype=np.float32)
                st["a_ref"] = np.zeros(3, dtype=np.float32)
                d.depot_timer += dt

                if d.depot_timer >= 1.0:
                    d.depot_timer = 0.0
                    d.manifest_idx = (d.manifest_idx + 1) % len(world.manifest_addresses)
                    order = world.manifest_addresses[d.manifest_idx]
                    d.target = order["pos"].copy()
                    d.compute_corridor_trajectory(d.target)
                    d.winch.reload_new_cargo()
                    d.bms.soc = 0.98
                    d.bms.v_pack = 25.0
                    d.hover_settle_timer = 0.0
                    d.last_delivery_bldg_id = None
                    d.mission_phase = "DELIVERY_OUTBOUND"

            target_bldg_id = None
            if "DELIVERY" in d.mission_phase:
                for h in world.houses:
                    x1, x2, y1, y2, _ = h["aabb"]
                    if x1 <= d.target[0] <= x2 and y1 <= d.target[1] <= y2:
                        target_bldg_id = h["id"]
                        break

            # Mixer evaluation with Authority Separation & Rooftop Immunity
            omega_cmds, f_thrust, a_safe = mixer.mix(
                drone_health=d.airframe.motor_health, p_est=d.ekf.p, v_est=d.ekf.v, R_est=st["R_est"], omega_est=st["omega_est"],
                p_target=st["p_ref"], neural_guidance=neural_guidance, token_idx=current_token_idx,
                estimated_wind=total_wind, obstacles=obstacles_cbf, swarm_states=swarm_states, drone_idx=d_idx,
                ref_accel=st["a_ref"], ref_vel=st["v_ref"], 
                target_bldg_id=target_bldg_id, departure_bldg_id=d.last_delivery_bldg_id,
                cable_len=d.winch.cable_len, sway_2d=d.winch.sway_offset, load_cell_3d=st["load_3d"], 
                ige_fac=d.airframe.last_ige, v_pack=st["v_pack"], bms_temp_c=d.bms.temp_c, 
                has_payload=d.winch.has_payload, dt=dt
            )

            f_act, torques, p_watts, ige_fac = d.airframe.step(
                motor_cmds=omega_cmds, current_wind=total_wind, rho_air=st["s_data"]["rho_air"],
                ice_index=st["ice_idx"], surface_z=st["current_surf_z"], has_payload=d.winch.has_payload, dt=dt
            )

            if int(flight_time * 60) % 6 == 0:
                d.trail.append(d.airframe.p[:2].copy())

            if d_idx == selected_drone_idx:
                selected_diag = {
                    "v_pack": st["v_pack"], "soc": st["bms_soc"], "temp_c": d.bms.temp_c,
                    "action": current_action_name, "v_crit": critic_v,
                    "thrust": f_thrust, "tension": d.winch.tension_n,
                    "cable_l": d.winch.cable_len, "load_3d": st["load_3d"],
                    "ice_mm": st["ice_mm"], "wind_speed": np.linalg.norm(total_wind),
                    "confidence": confidence, "latency": policy_latency_ms,
                    "gnss_status": st["gnss_status"], "jerk": st["jerk_discrepancy"],
                    "phase": float(np.linalg.norm(st["phase_shift"])),
                    "i_rms": st["i_rms"], "t_mosfet": st["t_mosfet"], "th_dt": st["th_dt"],
                    "dolp": st["dolp"], "ndvi": st["ndvi"], "hdop": st["hdop"], "cn0": st["cn0"],
                    "alpha": st["alpha"], "beta": st["beta"], "dust": st["dust"], "v_tas": st["s_data"]["v_tas"],
                    "accel_raw": st["meas_accel"], "gyro_raw": st["meas_gyro"]
                }

        # ==========================================
        # RENDERING (3D CITY GRID & HUD COCKPIT)
        # ==========================================
        screen.fill((10, 13, 20))
        pygame.draw.line(screen, (35, 45, 60), (VIEW_W, 0), (VIEW_W, H), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (W, 70), 2)

        # Ground Grid Lines
        for gx in range(-8, 9):
            p1 = np.array([cam_focus[0] + gx * 8.0, cam_focus[1] - 45.0, 0.0], dtype=np.float32)
            p2 = np.array([cam_focus[0] + gx * 8.0, cam_focus[1] + 45.0, 0.0], dtype=np.float32)
            sx1, sy1, _ = project_3d_chase(p1, cam_focus, cam_yaw, cam_pitch, center_viewport)
            sx2, sy2, _ = project_3d_chase(p2, cam_focus, cam_yaw, cam_pitch, center_viewport)
            if sx1 is not None and sx2 is not None:
                pygame.draw.line(screen, (18, 24, 32), (sx1, sy1), (sx2, sy2), 1)

        # Asphalt Avenues & Streets
        for r in world.roads:
            if math.hypot(cam_focus[0] - (r["x_min"]+r["x_max"])/2, cam_focus[1] - (r["y_min"]+r["y_max"])/2) < 85.0:
                rc_pts = [
                    np.array([r["x_min"], r["y_min"], 0.0]), np.array([r["x_max"], r["y_min"], 0.0]),
                    np.array([r["x_max"], r["y_max"], 0.0]), np.array([r["x_min"], r["y_max"], 0.0])
                ]
                r_screens = [project_3d_chase(p, cam_focus, cam_yaw, cam_pitch, center_viewport)[:2] for p in rc_pts]
                if all(pt[0] is not None for pt in r_screens):
                    pygame.draw.polygon(screen, (24, 30, 38), r_screens)

        # Solid Buildings with Decoupled Roof / Strut Rendering
        for idx, h in enumerate(world.houses):
            hx1, hx2, hy1, hy2, hh = h["aabb"]
            h_center = np.array([(hx1+hx2)/2, (hy1+hy2)/2, 0.0])
            if np.linalg.norm(cam_focus[:2] - h_center[:2]) < 85.0:
                h_corners = [
                    np.array([hx1, hy1, 0.0]), np.array([hx2, hy1, 0.0]),
                    np.array([hx2, hy2, 0.0]), np.array([hx1, hy2, 0.0]),
                    np.array([hx1, hy1, hh]), np.array([hx2, hy1, hh]),
                    np.array([hx2, hy2, hh]), np.array([hx1, hy2, hh])
                ]
                h_screens = [project_3d_chase(p, cam_focus, cam_yaw, cam_pitch, center_viewport)[:2] for p in h_corners]
                
                if all(h_screens[i][0] is not None for i in range(4, 8)):
                    roof_poly = [h_screens[4], h_screens[5], h_screens[6], h_screens[7]]
                    pygame.draw.polygon(screen, (38, 46, 58), roof_poly)
                    pygame.draw.lines(screen, (100, 120, 145), True, roof_poly, 2)
                    
                    roof_center = np.array([(hx1+hx2)/2, (hy1+hy2)/2, hh], dtype=np.float32)
                    rf_sx, rf_sy, _ = project_3d_chase(roof_center, cam_focus, cam_yaw, cam_pitch, center_viewport)
                    if rf_sx is not None:
                        pygame.draw.circle(screen, (0, 220, 255), (rf_sx, rf_sy), 12, 1)
                        screen.blit(font_small.render("H", True, (0, 220, 255)), (rf_sx - 3, rf_sy - 6))
                        screen.blit(font_small.render(f"#{world.manifest_addresses[idx]['bldg_id']}", True, (140, 170, 205)), (rf_sx - 18, rf_sy + 10))

                for ci in range(4):
                    if h_screens[ci][0] is not None and h_screens[ci+4][0] is not None:
                        pygame.draw.line(screen, (65, 78, 95), h_screens[ci], h_screens[ci+4], 2)

        # Dynamic Road Vehicles
        for v in traffic_engine.vehicles:
            if np.linalg.norm(cam_focus[:2] - v["pos"][:2]) < 60.0:
                vx_s, vy_s, _ = project_3d_chase(v["pos"], cam_focus, cam_yaw, cam_pitch, center_viewport)
                if vx_s is not None:
                    pygame.draw.rect(screen, (255, 140, 40), (vx_s - 6, vy_s - 4, 12, 8), border_radius=2)
                    pygame.draw.circle(screen, (255, 255, 200), (vx_s + 4, vy_s), 2)

        # Logistics Depot Helipads [H1, H2, H3]
        for p_i, pad in enumerate(world.depot_helipads):
            if np.linalg.norm(cam_focus[:2] - pad[:2]) < 70.0:
                hc_sx, hc_sy, _ = project_3d_chase(pad, cam_focus, cam_yaw, cam_pitch, center_viewport)
                if hc_sx is not None:
                    pad_color = swarm_colors[p_i % len(swarm_colors)]
                    pygame.draw.circle(screen, pad_color, (hc_sx, hc_sy), 16, 2)
                    screen.blit(font_large.render(f"H{p_i+1}", True, pad_color), (hc_sx - 10, hc_sy - 9))

        # Ground Intruder Agent
        if np.linalg.norm(cam_focus[:2] - agent_pos[:2]) < 45.0:
            ag_sx, ag_sy, _ = project_3d_chase(agent_pos, cam_focus, cam_yaw, cam_pitch, center_viewport)
            if ag_sx is not None:
                ag_col = (100, 220, 120) if ground_agent.is_leashed else ((255, 60, 80) if agent_breach else (255, 120, 40))
                pygame.draw.circle(screen, ag_col, (ag_sx, ag_sy), 7)
                pygame.draw.circle(screen, (100, 140, 180) if not agent_breach else (255, 60, 80), (ag_sx, ag_sy), int(ground_agent.personal_space_m * 18), 1)

        # Delivered Boxes
        for pb in delivered_boxes:
            if np.linalg.norm(cam_focus[:2] - pb[:2]) < 65.0:
                bx, by, _ = project_3d_chase(pb, cam_focus, cam_yaw, cam_pitch, center_viewport)
                if bx is not None:
                    pygame.draw.rect(screen, (60, 160, 90), (bx - 5, by - 5, 10, 10), border_radius=2)

        # Quintic Spline Ribbon Trajectory
        act_d = drones[selected_drone_idx]
        if act_d.spline_traj is not None:
            t_samples = np.linspace(0.0, act_d.spline_traj.total_duration, 35)
            pts_3d = [act_d.spline_traj.evaluate(st)[0] for st in t_samples]
            scr_pts = [project_3d_chase(p3, cam_focus, cam_yaw, cam_pitch, center_viewport)[:2] for p3 in pts_3d]
            scr_pts_valid = [pt for pt in scr_pts if pt[0] is not None]
            if len(scr_pts_valid) > 1:
                pygame.draw.lines(screen, (0, 180, 255), False, scr_pts_valid, 2)

        # Render All Drones in Swarm
        for d_i, d in enumerate(drones):
            col = swarm_colors[d_i % len(swarm_colors)]
            dsx, dsy, _ = project_3d_chase(d.airframe.p, cam_focus, cam_yaw, cam_pitch, center_viewport)
            if dsx is None: continue

            pygame.draw.circle(screen, col, (dsx, dsy), 8 if d_i == selected_drone_idx else 5)
            screen.blit(font_small.render(f"D{d_i+1}", True, col), (dsx + 10, dsy - 10))

            arm_d = ARM_LENGTH_M * 1.8
            arm_ends_body = [
                np.array([ arm_d/1.414, -arm_d/1.414, 0.0]), np.array([ arm_d/1.414,  arm_d/1.414, 0.0]),
                np.array([-arm_d/1.414,  arm_d/1.414, 0.0]), np.array([-arm_d/1.414, -arm_d/1.414, 0.0]),
            ]
            arm_screens = [project_3d_chase(d.airframe.p + d.airframe.R @ a, cam_focus, cam_yaw, cam_pitch, center_viewport)[:2] for a in arm_ends_body]
            if all(pt[0] is not None for pt in arm_screens):
                pygame.draw.line(screen, (160, 170, 185), arm_screens[0], arm_screens[2], 2)
                pygame.draw.line(screen, (160, 170, 185), arm_screens[1], arm_screens[3], 2)

                for m_i in range(8):
                    a_i = [0, 0, 1, 1, 2, 2, 3, 3][m_i]
                    mx, my = arm_screens[a_i]
                    off_y = -5 if (m_i % 2 == 0) else 5
                    if d.airframe.motor_health[m_i] == 0.0:
                        pygame.draw.line(screen, (255, 60, 80), (mx - 4, my + off_y - 4), (mx + 4, my + off_y + 4), 2)
                        pygame.draw.line(screen, (255, 60, 80), (mx + 4, my + off_y - 4), (mx - 4, my + off_y + 4), 2)
                    else:
                        pygame.draw.circle(screen, col, (mx, my + off_y), 7, 1)

            payload_3d = np.array([
                d.airframe.p[0] + d.winch.sway_offset[0],
                d.airframe.p[1] + d.winch.sway_offset[1],
                d.airframe.p[2] - d.winch.cable_len
            ], dtype=np.float32)
            psx, psy, _ = project_3d_chase(payload_3d, cam_focus, cam_yaw, cam_pitch, center_viewport)

            if psx is not None:
                if d.winch.cable_len > 0.10:
                    pygame.draw.line(screen, (230, 240, 255), (dsx, dsy + 3), (psx, psy), 1)

                if d.winch.has_payload:
                    box_x, box_y = (dsx, dsy + 5) if d.winch.cable_len <= 0.10 else (psx, psy)
                    pygame.draw.rect(screen, (255, 195, 35), (box_x - 6, box_y - 6, 12, 12), border_radius=2)
                    pygame.draw.rect(screen, (60, 45, 10), (box_x - 6, box_y - 6, 12, 12), 1)
                else:
                    hook_x, hook_y = (dsx, dsy + 3) if d.winch.cable_len <= 0.10 else (psx, psy)
                    pygame.draw.circle(screen, (200, 220, 240), (hook_x, hook_y), 3)

        # Target Dropzone Ring
        act_tgt = act_d.target
        tsx, tsy, _ = project_3d_chase(act_tgt, cam_focus, cam_yaw, cam_pitch, center_viewport)
        if tsx is not None:
            t_ring_col = (0, 220, 255) if "BASE" in act_d.mission_phase or "DEPOT" in act_d.mission_phase else (0, 255, 140)
            pygame.draw.circle(screen, t_ring_col, (tsx, tsy), 16, 2)
            pygame.draw.circle(screen, (255, 255, 255), (tsx, tsy), 4)
            cur_order = world.manifest_addresses[act_d.manifest_idx % len(world.manifest_addresses)]
            t_tag = f"DEPOT [H{selected_drone_idx+1}]" if "BASE" in act_d.mission_phase or "DEPOT" in act_d.mission_phase else f"{cur_order['bldg_id']} ({cur_order['addr']})"
            screen.blit(font_small.render(t_tag, True, t_ring_col), (tsx + 18, tsy - 6))

        # Wind Streamers
        for wp in wind_particles:
            px, py, _ = project_3d_chase(wp, cam_focus, cam_yaw, cam_pitch, center_viewport)
            if px is not None and 0 < px < VIEW_W and 70 < py < H:
                pygame.draw.line(screen, (80, 140, 180), (px, py), (px + int(total_wind[0]*2.0), py - int(total_wind[1]*2.0)), 1)

        # ==========================================
        # COMPLETE SIDEBAR DASHBOARD
        # ==========================================
        sb_x = VIEW_W + 14
        screen.blit(font_large.render("SURVIVAL CORE V35.1", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Round-Trip Harvester • 8-Stage FSM • Swarm Attention", True, (130, 160, 190)), (sb_x, 38))

        # Tactical Radar Minimap
        map_rect = (sb_x, 65, 210, 210)
        draw_tactical_radar_minimap(screen, map_rect, world, drones, selected_drone_idx, traffic_engine.vehicles)

        fl_x = sb_x + 224
        screen.blit(font_mono.render("SWARM FLEET [TAB]:", True, (240, 240, 240)), (fl_x, 65))
        for d_i, d in enumerate(drones):
            c = swarm_colors[d_i % len(swarm_colors)]
            tag = f"[D{d_i+1}]" + ("*" if d_i == selected_drone_idx else " ")
            stat = d.mission_phase[:6]
            screen.blit(font_small.render(f"{tag} {stat:6s} | {d.bms.soc*100:.0f}% | Box:{d.packages_delivered}", True, c), (fl_x, 85 + d_i * 20))

        cur_order = world.manifest_addresses[act_d.manifest_idx % len(world.manifest_addresses)]
        screen.blit(font_mono.render("ACTIVE ORDER:", True, (240, 240, 240)), (fl_x, 155))
        screen.blit(font_small.render(f"ID:   {cur_order['bldg_id']}", True, (0, 220, 255)), (fl_x, 175))
        screen.blit(font_small.render(f"Addr: {cur_order['addr'][:16]}", True, (0, 255, 140)), (fl_x, 193))
        screen.blit(font_small.render(f"Corridor Alt: {act_d.assigned_cruise_alt:.0f}m", True, (160, 180, 205)), (fl_x, 211))
        screen.blit(font_small.render(f"Phase:{act_d.mission_phase[:15]}", True, (255, 200, 40)), (fl_x, 229))

        # 1. District Manifest Status
        p1_y = 285
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p1_y - 8), (W - 14, p1_y - 8), 1)
        screen.blit(font_mono.render("1. 48-BUILDING ROOFTOP MANIFEST:", True, (240, 240, 240)), (sb_x, p1_y))
        dist_dest = float(np.linalg.norm(act_d.airframe.p[:2] - act_d.target[:2]))
        dest_tag = f"Depot [H{selected_drone_idx+1}]" if "BASE" in act_d.mission_phase or "DEPOT" in act_d.mission_phase else f"{cur_order['bldg_id']} ({cur_order['addr']})"
        screen.blit(font_small.render(f"Target Waypoint:   {dest_tag[:22]} ({dist_dest:.1f}m)", True, (0, 220, 255)), (sb_x + 10, p1_y + 18))
        cargo_tag = "0.45kg PARCEL CLAMPED" if act_d.winch.has_payload else "EMPTY HOOK (RTB RELOAD)"
        screen.blit(font_small.render(f"Fuselage Cargo:    {cargo_tag}", True, (0, 255, 140) if act_d.winch.has_payload else (255, 200, 40)), (sb_x + 10, p1_y + 34))
        tot_deliv = sum(d.packages_delivered for d in drones)
        screen.blit(font_small.render(f"Swarm Deliveries:  {tot_deliv} Delivered (Continuous Fleet)", True, (0, 255, 140)), (sb_x + 10, p1_y + 50))

        # 2. 122-Dim Sensory Telemetry & Swarm Observables
        p2_y = p1_y + 68
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (W - 14, p2_y), 1)
        screen.blit(font_mono.render("2. 122-DIM MULTI-AGENT TELEMETRY (366-DIM):", True, (240, 240, 240)), (sb_x, p2_y + 8))
        raw_a = selected_diag.get("accel_raw", np.zeros(3))
        raw_w = selected_diag.get("gyro_raw", np.zeros(3))
        screen.blit(font_small.render(f"Airframe Accel:    [{raw_a[0]:.2f}, {raw_a[1]:.2f}, {raw_a[2]:.2f}] m/s²", True, (0, 220, 255)), (sb_x + 10, p2_y + 26))
        screen.blit(font_small.render(f"Airframe Gyro:     [{raw_w[0]:.2f}, {raw_w[1]:.2f}, {raw_w[2]:.2f}] rad/s", True, (0, 255, 140)), (sb_x + 10, p2_y + 42))
        
        min_peer_dist = 999.0
        for pi_idx, other_d in enumerate(drones):
            if pi_idx == selected_drone_idx: continue
            min_peer_dist = min(min_peer_dist, float(np.linalg.norm(other_d.airframe.p - act_d.airframe.p)))
        screen.blit(font_small.render(f"Swarm Peer Dist:   Closest Peer = {min_peer_dist:.1f} m (Corridor Safe)", True, (160, 180, 205)), (sb_x + 10, p2_y + 58))

        # 3. ESC Telemetry & Multi-Spectral
        p3_y = p2_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (W - 14, p3_y), 1)
        screen.blit(font_mono.render("3. ESC TELEMETRY & MULTI-SPECTRAL:", True, (240, 240, 240)), (sb_x, p3_y + 8))
        irms = selected_diag.get("i_rms", np.zeros(8))
        tmos = selected_diag.get("t_mosfet", np.zeros(8))
        screen.blit(font_small.render(f"8-Ch ESC Currents: Avg I = {float(np.mean(irms)):.1f} A | Max FET = {float(np.max(tmos)):.1f}°C", True, (0, 220, 255)), (sb_x + 10, p3_y + 26))
        screen.blit(font_small.render(f"Thermal LWIR/DoLP: ΔT = {selected_diag.get('th_dt', 0.0):.1f}°C | Pol = {selected_diag.get('dolp', 0.0):.2f} | NDVI = {selected_diag.get('ndvi', 0.0):.2f}", True, (0, 255, 140)), (sb_x + 10, p3_y + 42))
        screen.blit(font_small.render(f"RTK GNSS Metrics:  HDOP = {selected_diag.get('hdop', 0.85):.2f} | Avg C/N0 = {selected_diag.get('cn0', 44.0):.1f} dB-Hz", True, (160, 180, 205)), (sb_x + 10, p3_y + 58))

        # 4. Urban Weather & Canyon Aerodynamics
        p4_y = p3_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (W - 14, p4_y), 1)
        screen.blit(font_mono.render("4. URBAN WEATHER & CANYON AERODYNAMICS:", True, (240, 240, 240)), (sb_x, p4_y + 8))
        w_mode = weather_engine.get_mode_name()
        w_col = (0, 255, 140) if "CLEAR" in w_mode else ((255, 180, 40) if "CANYON" in w_mode else (255, 60, 80))
        screen.blit(font_small.render(f"Atmospheric State: {w_mode} ([W] Cycle)", True, w_col), (sb_x + 10, p4_y + 26))
        screen.blit(font_small.render(f"Venturi Airspeed:  |V_tas| = {selected_diag.get('v_tas', 0.0):.2f} m/s | Wind: {selected_diag.get('wind_speed', 0.0):.1f} m/s", True, (0, 220, 255)), (sb_x + 10, p4_y + 42))
        screen.blit(font_small.render(f"Rotor Ice Layer:   {selected_diag.get('ice_mm', 0.0):.2f} mm | Temp = {selected_diag.get('temp_c', 22.0):.1f}°C", True, (160, 180, 205)), (sb_x + 10, p4_y + 58))

        # 5. GNSS Anti-Spoof & Quantum Sensing
        p5_y = p4_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p5_y), (W - 14, p5_y), 1)
        screen.blit(font_mono.render("5. GNSS ANTI-SPOOF & QUANTUM VIO:", True, (240, 240, 240)), (sb_x, p5_y + 8))
        g_stat = selected_diag.get("gnss_status", "GNSS NOMINAL")
        screen.blit(font_small.render(f"Navigation State:  {g_stat}", True, (255, 60, 80) if "REJECTED" in g_stat else (0, 255, 140)), (sb_x + 10, p5_y + 26))
        screen.blit(font_small.render(f"Kinematic Discrep: Δa = {selected_diag.get('jerk', 0.0):.2f} m/s² (Thresh = 16.0)", True, (160, 180, 205)), (sb_x + 10, p5_y + 42))
        screen.blit(font_small.render(f"Quantum Phase:     ΔΦ = {selected_diag.get('phase', 0.0):.3f} rad (Rb-87 Raman)", True, (0, 220, 255)), (sb_x + 10, p5_y + 58))

        # 6. Swarm Cross-Attention Executive & Winch
        p6_y = p5_y + 78
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p6_y), (W - 14, p6_y), 1)
        screen.blit(font_mono.render("6. SWARM CROSS-ATTENTION EXECUTIVE (~2.5M):", True, (240, 240, 240)), (sb_x, p6_y + 8))
        screen.blit(font_small.render(f"Primary Action:    {current_action_name} ({confidence*100:.1f}%) | Latency: {selected_diag.get('latency', 1.0):.2f}ms", True, (0, 255, 140)), (sb_x + 10, p6_y + 26))
        screen.blit(font_small.render(f"Winch State:       {act_d.winch.state} (Cable: {act_d.winch.cable_len:.2f} m | Tension: {act_d.winch.tension_n:.1f} N)", True, (0, 220, 255)), (sb_x + 10, p6_y + 42))
        screen.blit(font_small.render(f"Traffic / Agent:   8 Vehicles Active | Ground Agent: {agent_dist:.1f}m", True, (160, 180, 205)), (sb_x + 10, p6_y + 58))

        # Bottom Command Strip
        bot_y = H - 38
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y - 8), (W - 14, bot_y - 8), 1)
        screen.blit(font_mono.render("[TAB] Drone | [W] Weather | [CLICK] Retarget | [H] RTB | [J] Spoof", True, (255, 200, 40)), (sb_x + 8, bot_y))

        # Viewport Header
        vert_spd = act_d.airframe.v[2]
        status_banner = f"ACTIVE: D{selected_drone_idx+1} | Phase: {act_d.mission_phase} (Vz: {vert_spd:+.2f}m/s) | Delivered: {tot_deliv}"
        screen.blit(font_large.render(status_banner, True, (240, 240, 240)), (15, 14))
        screen.blit(font_small.render("Round-Trip Mission Dataset • Lookahead Pursuit • Swarm Attention", True, (0, 210, 255)), (15, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    mp.freeze_support()
    run_all_physics_simulator()