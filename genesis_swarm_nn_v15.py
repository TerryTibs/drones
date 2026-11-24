"""
=============================================================================
SURVIVAL CORE V37.0 — UNIFIED GENESIS MULTI-PHYSICS MASTER SIMULATOR
=============================================================================
"""

import os
import sys
import math
import time
import random
from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pygame

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import genesis as gs

# =============================================================================
# 0. PHYSICAL AIRFRAME & MISSION CONSTANTS
# =============================================================================
DRY_MASS_KG = 1.35                   
CARGO_MASS_KG = 0.45                 
TOTAL_MASS_KG = DRY_MASS_KG + CARGO_MASS_KG  # 1.80 kg total mass
GRAVITY_VAL = 9.80665

ARM_LENGTH_M = 0.26 / math.sqrt(2.0)
ROTOR_RADIUS_M = 0.127
ROTOR_DISK_AREA = math.pi * (ROTOR_RADIUS_M ** 2)

INERTIA_MATRIX = np.diag([0.022, 0.022, 0.038]).astype(np.float32)

C_T_CLEAN = 1.35e-5
C_Q_CLEAN = 2.40e-7
COAXIAL_EFFICIENCY = 0.92
MOTOR_TAU_S = 0.030
MAX_RPM = 9200.0
MAX_OMEGA = (MAX_RPM * 2.0 * math.pi) / 60.0
HOVER_THRUST = TOTAL_MASS_KG * GRAVITY_VAL

MAX_CITY_OBSTACLE_HEIGHT_M = 28.0
MIN_SAFE_CRUISE_ALT_M = MAX_CITY_OBSTACLE_HEIGHT_M + 6.0
GLOBAL_CEILING_ALT_M = 60.0

BASE_OBS_DIM = 139
HISTORY_LEN = 3
OBSERVATION_DIM = BASE_OBS_DIM * HISTORY_LEN
NUM_FSM_STATES = 7
NUM_SWARM_DRONES = 3

MODEL_WEIGHTS_FILE = "drone_master_brain_genesis_master.pth"
URDF_ASSET_FILE = "drone_coaxial_x8.urdf"

FSM_NAMES = [
    "TAKEOFF",
    "CRUISE_OUTBOUND",
    "CHIMNEY_DESCENT",
    "DELIVERY_HOVER",
    "WINCH_RETRACT",
    "RETURN_BASE",
    "DEPOT_LANDING"
]

# =============================================================================
# 1. WATERTIGHT PROCEDURAL COAXIAL X8 DRONE URDF GENERATOR
# =============================================================================
def generate_coaxial_x8_urdf(filepath: str = "drone_coaxial_x8.urdf"):
    """
    Constructs a watertight, analytical collision airframe.
    All collision geometries use closed primitive boxes to avoid Genesis
    mesh non-watertight fallback warnings.
    """
    arm_l = ARM_LENGTH_M
    arm_angle = math.pi / 4.0
    dx = arm_l * math.cos(arm_angle)
    dy = arm_l * math.sin(arm_angle)

    urdf_content = f"""<?xml version="1.0"?>
<robot name="coaxial_x8_drone">

  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{DRY_MASS_KG}"/>
      <inertia ixx="0.022" ixy="0" ixz="0" iyy="0.022" iyz="0" izz="0.038"/>
    </inertial>
    
    <!-- Central Avionics Core -->
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.22 0.16 0.08"/></geometry>
      <material name="carbon_dark"><color rgba="0.12 0.14 0.18 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.22 0.16 0.08"/></geometry>
    </collision>

    <!-- Heading Arrow -->
    <visual>
      <origin xyz="0.12 0 0.01" rpy="0 0 0"/>
      <geometry><box size="0.06 0.04 0.03"/></geometry>
      <material name="nose_red"><color rgba="0.95 0.25 0.25 1.0"/></material>
    </visual>

    <!-- GPS Dome -->
    <visual>
      <origin xyz="0 0 0.055" rpy="0 0 0"/>
      <geometry><box size="0.07 0.07 0.03"/></geometry>
      <material name="cyan_accent"><color rgba="0.0 0.85 0.95 1.0"/></material>
    </visual>

    <!-- 4 Diagonal Arms (Watertight Box Colliders) -->
    <visual>
      <origin xyz="{dx/2} {dy/2} 0.01" rpy="0 0 {arm_angle}"/>
      <geometry><box size="{arm_l} 0.024 0.024"/></geometry>
      <material name="carbon_arm"><color rgba="0.2 0.22 0.25 1.0"/></material>
    </visual>
    <visual>
      <origin xyz="{dx/2} {-dy/2} 0.01" rpy="0 0 {-arm_angle}"/>
      <geometry><box size="{arm_l} 0.024 0.024"/></geometry>
      <material name="carbon_arm"><color rgba="0.2 0.22 0.25 1.0"/></material>
    </visual>
    <visual>
      <origin xyz="{-dx/2} {dy/2} 0.01" rpy="0 0 {math.pi - arm_angle}"/>
      <geometry><box size="{arm_l} 0.024 0.024"/></geometry>
      <material name="carbon_arm"><color rgba="0.2 0.22 0.25 1.0"/></material>
    </visual>
    <visual>
      <origin xyz="{-dx/2} {-dy/2} 0.01" rpy="0 0 {-math.pi + arm_angle}"/>
      <geometry><box size="{arm_l} 0.024 0.024"/></geometry>
      <material name="carbon_arm"><color rgba="0.2 0.22 0.25 1.0"/></material>
    </visual>

    <!-- Landing Skids -->
    <visual>
      <origin xyz="0 0.08 -0.055" rpy="0 0 0"/>
      <geometry><box size="0.22 0.014 0.014"/></geometry>
      <material name="skid_mat"><color rgba="0.3 0.35 0.4 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0.08 -0.055" rpy="0 0 0"/>
      <geometry><box size="0.22 0.014 0.014"/></geometry>
    </collision>
    <visual>
      <origin xyz="0 -0.08 -0.055" rpy="0 0 0"/>
      <geometry><box size="0.22 0.014 0.014"/></geometry>
      <material name="skid_mat"><color rgba="0.3 0.35 0.4 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 -0.08 -0.055" rpy="0 0 0"/>
      <geometry><box size="0.22 0.014 0.014"/></geometry>
    </collision>
  </link>
</robot>
"""
    with open(filepath, "w") as f:
        f.write(urdf_content)
    print(f"[URDF] Watertight Coaxial X8 model compiled to {filepath}")

# =============================================================================
# 2. VECTORIZED METROPOLITAN GRID DEFINITION
# =============================================================================
class MetropolitanCityGridWorld400m:
    def __init__(self):
        self.avenue_x = [-140.0, -90.0, -30.0, 30.0, 90.0, 140.0]
        self.street_y = [-140.0, -90.0, -30.0, 30.0, 90.0, 140.0]
        self.roads = []
        for x in self.avenue_x:
            self.roads.append({"x_min": x - 5.0, "x_max": x + 5.0, "y_min": -200.0, "y_max": 200.0, "type": "AVENUE"})
        for y in self.street_y:
            self.roads.append({"x_min": -200.0, "x_max": 200.0, "y_min": y - 5.0, "y_max": y + 5.0, "type": "STREET"})

        self.depot_helipads = [
            np.array([-120.0, -120.0, 0.08], dtype=np.float32),
            np.array([-120.0, -100.0, 0.08], dtype=np.float32),
            np.array([-100.0, -120.0, 0.08], dtype=np.float32)
        ]
        self.houses = []
        self.manifest_addresses = []
        self._build_manifest()

        self.trees = [
            {"pos": np.array([-25.0, -25.0, 0.0], dtype=np.float32), "r": 3.2, "h": 8.0},
            {"pos": np.array([ 25.0, -25.0, 0.0], dtype=np.float32), "r": 3.0, "h": 7.5},
            {"pos": np.array([-25.0,  25.0, 0.0], dtype=np.float32), "r": 3.5, "h": 8.5},
            {"pos": np.array([ 25.0,  25.0, 0.0], dtype=np.float32), "r": 3.2, "h": 8.0}
        ]
        self.powerlines = [
            {"p1": np.array([-170.0,  15.0, 12.0], dtype=np.float32), "p2": np.array([170.0,  15.0, 12.0], dtype=np.float32)},
            {"p1": np.array([-170.0, -15.0, 12.0], dtype=np.float32), "p2": np.array([170.0, -15.0, 12.0], dtype=np.float32)}
        ]

    def _build_manifest(self):
        col_candidates = [-150.0, -110.0, -70.0, -30.0, 30.0, 70.0, 110.0, 150.0]
        row_candidates = [-150.0, -110.0, -70.0, -30.0, 30.0, 70.0, 110.0, 150.0]
        bldg_idx = 0

        for c_i, cx in enumerate(col_candidates):
            for r_i, ry in enumerate(row_candidates):
                if cx <= -70.0 and ry <= -70.0:
                    continue
                if bldg_idx >= 48:
                    break

                tier = (c_i + r_i) % 4
                height = [8.5, 14.5, 21.0, 26.0][tier]
                bw, bd = 16.0, 16.0
                bx1, bx2 = cx - bw/2.0, cx + bw/2.0
                by1, by2 = ry - bd/2.0, ry + bd/2.0
                bldg_id = f"Bldg_{bldg_idx+1:02d}"

                clutter = [
                    {"aabb": [cx - 5.1, cx - 1.9, ry + 2.1, ry + 4.9, height + 1.4], "type": "HVAC"}
                ]

                bldg_dict = {
                    "id": bldg_id,
                    "pos": np.array([cx + 2.0, ry - 2.0, height + 0.08], dtype=np.float32),
                    "center": np.array([cx, ry, height / 2.0], dtype=np.float32),
                    "size": np.array([bw, bd, height], dtype=np.float32),
                    "aabb": [bx1, bx2, by1, by2, height],
                    "height": height,
                    "radius": 10.5,
                    "tag_id": bldg_idx + 101,
                    "clutter": clutter
                }
                self.houses.append(bldg_dict)
                self.manifest_addresses.append({
                    "id": bldg_id,
                    "bldg_id": bldg_id,
                    "addr": f"Tower {bldg_idx+1:02d} Pad",
                    "pos": bldg_dict["pos"].copy(),
                    "height": height,
                    "tag_id": bldg_idx + 101
                })
                bldg_idx += 1

    def get_surface_height(self, x, y):
        for h in self.houses:
            x1, x2, y1, y2, hh = h["aabb"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                for c in h.get("clutter", []):
                    cx1, cx2, cy1, cy2, ch = c["aabb"]
                    if cx1 <= x <= cx2 and cy1 <= y <= cy2:
                        return float(ch)
                return float(hh)
        return 0.0

def build_metropolitan_city_with_hazards(scene, world: MetropolitanCityGridWorld400m, device="cuda"):
    scene.add_entity(
        morph=gs.morphs.Plane(collision=True),
        material=gs.materials.Rigid(friction=0.85),
        surface=gs.surfaces.Rough(color=(0.18, 0.20, 0.24), roughness=0.8)
    )

    for r in world.roads:
        sx = max(0.1, float(r["x_max"] - r["x_min"]))
        sy = max(0.1, float(r["y_max"] - r["y_min"]))
        scene.add_entity(
            morph=gs.morphs.Box(size=(sx, sy, 0.025), pos=((r["x_min"]+r["x_max"])/2, (r["y_min"]+r["y_max"])/2, 0.0125), fixed=True, collision=False),
            material=gs.materials.Rigid(friction=0.9),
            surface=gs.surfaces.Rough(color=(0.08, 0.09, 0.11), roughness=0.95)
        )

    bldg_centers, bldg_half_extents, bldg_roof_heights = [], [], []

    for h in world.houses:
        bw, bd, bh = h["size"]
        cx, cy, _ = h["center"]
        tier = int(bh // 7.0)

        scene.add_entity(
            morph=gs.morphs.Box(size=(bw, bd, bh), pos=(cx, cy, bh / 2.0), fixed=True, collision=True),
            material=gs.materials.Rigid(friction=0.7),
            surface=gs.surfaces.Rough(color=(0.28 + 0.03 * tier, 0.32 + 0.03 * tier, 0.38 + 0.03 * tier), roughness=0.6)
        )
        for c in h.get("clutter", []):
            cx1, cx2, cy1, cy2, ch = c["aabb"]
            clutter_h = max(0.15, ch - bh)
            scene.add_entity(
                morph=gs.morphs.Box(size=(cx2 - cx1, cy2 - cy1, clutter_h), pos=((cx1+cx2)/2, (cy1+cy2)/2, bh + clutter_h/2), fixed=True, collision=True),
                material=gs.materials.Rigid(friction=0.6),
                surface=gs.surfaces.Rough(color=(0.55, 0.55, 0.58), roughness=0.5)
            )

        pad_pos = tuple(h["pos"])
        scene.add_entity(
            morph=gs.morphs.Cylinder(radius=2.2, height=0.08, pos=pad_pos, fixed=True, collision=False),
            material=gs.materials.Rigid(friction=0.85),
            surface=gs.surfaces.Default(color=(0.0, 0.85, 0.55))
        )

        bldg_centers.append([cx, cy, bh / 2.0])
        bldg_half_extents.append([bw / 2.0, bd / 2.0, bh / 2.0])
        bldg_roof_heights.append(bh)

    pad_colors = [(0.10, 0.65, 0.95), (0.95, 0.55, 0.15), (0.75, 0.25, 0.95)]
    for p_i, dp in enumerate(world.depot_helipads):
        scene.add_entity(
            morph=gs.morphs.Cylinder(radius=2.8, height=0.08, pos=tuple(dp), fixed=True, collision=False),
            material=gs.materials.Rigid(friction=0.9),
            surface=gs.surfaces.Default(color=pad_colors[p_i])
        )

    for line in world.powerlines:
        a, b = line["p1"], line["p2"]
        length = float(np.linalg.norm(b[:2] - a[:2]))
        scene.add_entity(
            morph=gs.morphs.Box(size=(length, 0.08, 0.08), pos=tuple((a + b) / 2.0), fixed=True, collision=False),
            material=gs.materials.Rigid(friction=0.2),
            surface=gs.surfaces.Default(color=(0.95, 0.20, 0.20))
        )

    for tree in world.trees:
        tx, ty = tree["pos"][0], tree["pos"][1]
        scene.add_entity(
            morph=gs.morphs.Cylinder(radius=0.45, height=5.0, pos=(tx, ty, 2.5), fixed=True, collision=False),
            material=gs.materials.Rigid(friction=0.7),
            surface=gs.surfaces.Rough(color=(0.40, 0.25, 0.15), roughness=0.8)
        )
        scene.add_entity(
            morph=gs.morphs.Sphere(radius=float(tree["r"]), pos=(tx, ty, 6.5), fixed=True, collision=False),
            material=gs.materials.Rigid(friction=0.6),
            surface=gs.surfaces.Rough(color=(0.15, 0.55, 0.20), roughness=0.7)
        )

    gpu_geom_cache = {
        "centers": torch.tensor(bldg_centers, device=device, dtype=torch.float32),
        "half_extents": torch.tensor(bldg_half_extents, device=device, dtype=torch.float32),
        "roof_heights": torch.tensor(bldg_roof_heights, device=device, dtype=torch.float32),
    }

    print(f"[WORLD] 400m City initialized on {device}.")
    return gpu_geom_cache

# =============================================================================
# 3. SWARM ENTITY SPAWNER (PHYSICAL DRONES & PACKAGES)
# =============================================================================
def spawn_swarm_drones_and_cargo(scene, depot_pads):
    drones = []
    cargos = []

    for i in range(NUM_SWARM_DRONES):
        pad_x, pad_y = float(depot_pads[i][0]), float(depot_pads[i][1])

        cargo_entity = scene.add_entity(
            morph=gs.morphs.Box(size=(0.22, 0.22, 0.18), pos=(pad_x, pad_y, 0.18)),
            material=gs.materials.Rigid(rho=45.0, friction=0.75),
            surface=gs.surfaces.Default(color=(0.95, 0.75, 0.15))
        )
        cargos.append(cargo_entity)

        drone_entity = scene.add_entity(
            morph=gs.morphs.URDF(file=URDF_ASSET_FILE, pos=(pad_x, pad_y, 0.70)),
            material=gs.materials.Rigid(friction=0.5)
        )
        drones.append(drone_entity)

    return drones, cargos

# =============================================================================
# 4. ATMOSPHERIC & 6S BATTERY EQUIVALENT CIRCUIT MODEL (ECM)
# =============================================================================
NUM_CELLS = 6
PACK_CAPACITY_AH = 8.0               
CELL_R0_NOMINAL = 0.0085
CELL_R1 = 0.0065
CELL_C1 = 2800.0
ARRHENIUS_EA = 28000.0
GAS_CONST_R = 8.31446
T_REF_K = 298.15

P_0 = 1013.25
T_0 = 288.15
L_LAPSE = 0.0065
R_SPEC_AIR = 287.058
PAYLOAD_WEIGHT_N = CARGO_MASS_KG * GRAVITY_VAL
GUILLOTINE_THRESHOLD_N = 48.0

class SubZeroArrheniusBatteryECM:
    def __init__(self, initial_soc=0.98, initial_temp_c=22.0):
        self.soc = initial_soc
        self.temp_c = initial_temp_c
        self.v1 = np.zeros(NUM_CELLS, dtype=np.float32)
        self.capacity_coulombs = PACK_CAPACITY_AH * 3600.0
        self.v_pack = 25.0

    def step(self, p_elec_watts, amb_temp_c=22.0, dt=0.005):
        i_draw = float(p_elec_watts) / max(16.0, self.v_pack)
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
        self.temp_c += ((p_joule - 0.45 * (self.temp_c - amb_temp_c)) / 350.0) * dt
        return self.soc, self.v_pack, i_draw, r0

class MessingerRotorIcingModel:
    def __init__(self):
        self.ice_thickness_mm = 0.0
        self.icing_active = False

    def step(self, airspeed, dt=0.005):
        if self.icing_active:
            lwc = 0.35
            catch_efficiency = 0.42
            d_ice = (lwc * max(1.0, airspeed) * catch_efficiency * 1e-3) * 0.08 * dt
            self.ice_thickness_mm = min(3.8, self.ice_thickness_mm + d_ice * 1000.0)
        else:
            self.ice_thickness_mm = max(0.0, self.ice_thickness_mm - 0.02 * dt)

        ice_index = float(np.clip(self.ice_thickness_mm / 3.0, 0.0, 1.0))
        lift_loss_factor = 1.0 - 0.28 * ice_index
        drag_rise_factor = 1.0 + 1.80 * ice_index
        return self.ice_thickness_mm, ice_index, lift_loss_factor, drag_rise_factor

class DrydenAtmosphericTurbulenceModel:
    def __init__(self, w20_m_s=3.4):
        self.w20 = w20_m_s
        self.gust_state = np.zeros(3, dtype=np.float32)

    def step(self, altitude_m, airspeed_m_s, dt=0.005):
        h = max(0.5, float(altitude_m))
        v = max(1.0, float(airspeed_m_s))
        sigma_w = 0.10 * self.w20
        sigma_u = sigma_w / ((0.177 + 0.000823 * h) ** 0.4)
        sigma_v = sigma_u
        l_w = h
        l_u = h / ((0.177 + 0.000823 * h) ** 1.2)
        l_v = l_u

        white_noise = np.random.normal(0.0, 1.0, 3).astype(np.float32)
        sq_dt = math.sqrt(dt)

        du = (- (v / l_u) * self.gust_state[0] * dt + sigma_u * math.sqrt(2.0 * v / l_u) * white_noise[0] * sq_dt)
        dv = (- (v / l_v) * self.gust_state[1] * dt + sigma_v * math.sqrt(2.0 * v / l_v) * white_noise[1] * sq_dt)
        dw = (- (v / l_w) * self.gust_state[2] * dt + sigma_w * math.sqrt(2.0 * v / l_w) * white_noise[2] * sq_dt)

        self.gust_state[0] += du
        self.gust_state[1] += dv
        self.gust_state[2] += dw
        self.gust_state = np.clip(self.gust_state, -6.5, 6.5)
        return self.gust_state.copy()

class DynamicUrbanWeatherEngine:
    def __init__(self, regime_duration=50.0):
        self.weather_modes = ["CLEAR_NOMINAL", "CANYON_GALE", "FREEZING_RAIN", "MICROBURST_STORM"]
        self.current_mode_idx = 0
        self.target_mode_idx = 0
        self.regime_timer = 0.0
        self.regime_duration = regime_duration
        self.blend_factor = 1.0

    def cycle_weather(self):
        self.target_mode_idx = (self.target_mode_idx + 1) % len(self.weather_modes)
        self.regime_timer = 0.0
        self.blend_factor = 0.0
        print(f"\n>>> [WEATHER SHIFT] Mode: {self.weather_modes[self.target_mode_idx]} <<<")

    def get_mode_name(self):
        return self.weather_modes[self.target_mode_idx]

    def step(self, dt=0.005):
        self.regime_timer += dt
        if self.blend_factor < 1.0:
            self.blend_factor = min(1.0, self.blend_factor + dt / 3.0)
            if self.blend_factor >= 1.0:
                self.current_mode_idx = self.target_mode_idx

        if self.regime_timer >= self.regime_duration:
            self.cycle_weather()

    def _get_preset(self, mode_idx):
        mode = self.weather_modes[mode_idx]
        if mode == "CLEAR_NOMINAL":
            return np.array([1.8, -0.6, 0.0], dtype=np.float32), 0.5, 22.0, False, 0.0
        elif mode == "CANYON_GALE":
            return np.array([6.2, 2.8, 0.0], dtype=np.float32), 3.5, 14.0, False, 0.0
        elif mode == "FREEZING_RAIN":
            return np.array([2.5, -2.0, 0.0], dtype=np.float32), 1.5, -4.5, True, -0.4
        else:
            return np.array([-4.5, 4.0, 0.0], dtype=np.float32), 5.5, 10.0, False, -2.8

    def get_weather_parameters(self, drone_pos, avenues_x=None):
        px, py, pz = float(drone_pos[0]), float(drone_pos[1]), float(drone_pos[2])
        w1, g1, t1, i1, m1 = self._get_preset(self.current_mode_idx)
        w2, g2, t2, i2, m2 = self._get_preset(self.target_mode_idx)

        b = self.blend_factor
        base_wind = (1.0 - b) * w1 + b * w2
        gust_scale = (1.0 - b) * g1 + b * g2
        temp_c = (1.0 - b) * t1 + b * t2
        auto_ice = i2 if b > 0.5 else i1
        microburst_w = (1.0 - b) * m1 + b * m2

        if microburst_w < -0.2:
            stagnation = float(math.tanh(max(0.1, pz) / 4.5))
            microburst_w *= stagnation

        venturi_boost = 1.0
        if avenues_x is not None:
            for ax in avenues_x:
                if abs(px - ax) < 8.0 and pz < 26.0:
                    venturi_boost = 1.45
                    break

        eff_wind = base_wind.copy()
        eff_wind[1] *= venturi_boost
        eff_wind[2] += microburst_w
        return eff_wind, gust_scale, temp_c, auto_ice

# =============================================================================
# 5. VECTORIZED GPU SKYSCRAPER WALL SUCTION (NO CPU ROUNDTRIP)
# =============================================================================
def compute_building_wall_suction_tensor(drone_pos_tensor, gpu_geom_cache, rho_air=1.225, v_h_hover=5.5):
    centers = gpu_geom_cache["centers"]             # (N, 3)
    half_extents = gpu_geom_cache["half_extents"]   # (N, 3)
    roof_heights = gpu_geom_cache["roof_heights"]   # (N,)

    num_drones = drone_pos_tensor.shape[0]
    total_suction = torch.zeros((num_drones, 3), device=drone_pos_tensor.device, dtype=torch.float32)

    for i in range(num_drones):
        p = drone_pos_tensor[i]
        pz = p[2]
        valid_mask = (pz <= (roof_heights + 1.0))
        if not torch.any(valid_mask):
            continue

        c = centers[valid_mask]
        h_ext = half_extents[valid_mask]

        x1, x2 = c[:, 0] - h_ext[:, 0], c[:, 0] + h_ext[:, 0]
        y1, y2 = c[:, 1] - h_ext[:, 1], c[:, 1] + h_ext[:, 1]

        dx = torch.clamp(torch.maximum(x1 - p[0], p[0] - x2), min=0.0)
        dy = torch.clamp(torch.maximum(y1 - p[1], p[1] - y2), min=0.0)
        wall_dist = torch.hypot(dx, dy)

        near_mask = (wall_dist > 0.2) & (wall_dist < 3.2)
        if torch.any(near_mask):
            d_near = wall_dist[near_mask].unsqueeze(-1)
            c_near = c[near_mask]

            suction_mag = (0.5 * rho_air * (v_h_hover ** 2) * 0.08) * ((ROTOR_RADIUS_M / d_near) ** 2)
            n_wall = torch.cat([c_near[:, :2] - p[:2], torch.zeros((c_near.shape[0], 1), device=p.device)], dim=-1)
            n_wall = n_wall / torch.clamp(torch.norm(n_wall, dim=-1, keepdim=True), min=1e-4)
            total_suction[i] = torch.sum(suction_mag * n_wall, dim=0)

    return total_suction

# =============================================================================
# 6. GPU-VECTORIZED BEMT COAXIAL MOTOR ALLOCATION ENGINE
# =============================================================================
class VectorizedBEMTEngine(nn.Module):
    def __init__(self, num_drones: int, device: str = "cuda"):
        super().__init__()
        self.num_drones = num_drones
        self.device = device
        self.motor_omegas = torch.full((num_drones, 8), 450.0, device=device)

        L = ARM_LENGTH_M
        cq_ct_top = C_Q_CLEAN / C_T_CLEAN
        cq_ct_bot = (C_Q_CLEAN / C_T_CLEAN) / COAXIAL_EFFICIENCY
        
        B_mat = torch.tensor([
            [ 1.0,         1.0,        1.0,         1.0,        1.0,         1.0,        1.0,         1.0],
            [-L,          -L,          L,           L,         -L,          -L,          L,           L],
            [-L,          -L,         -L,          -L,          L,           L,          L,           L],
            [-cq_ct_top, cq_ct_bot, cq_ct_top, -cq_ct_bot, -cq_ct_top, cq_ct_bot, cq_ct_top, -cq_ct_bot]
        ], device=device, dtype=torch.float32)

        self.register_buffer("B_mat", B_mat)
        self.register_buffer("B_pinv", torch.linalg.pinv(B_mat))

    def compute_aerodynamics(self, motor_cmds: torch.Tensor, pos: torch.Tensor, vel: torch.Tensor, target_surf_z: float, dt: float):
        effective_cmds = torch.clamp(motor_cmds, 0.0, MAX_OMEGA)
        self.motor_omegas += ((effective_cmds - self.motor_omegas) / MOTOR_TAU_S) * dt

        z_clearance = torch.clamp(pos[:, 2] - target_surf_z, min=0.06)
        ratio_zh_r = z_clearance / ROTOR_RADIUS_M
        ige_factor = torch.where(
            ratio_zh_r >= 0.5,
            torch.clamp(1.0 / (1.0 - (1.0 / (4.0 * ratio_zh_r)) ** 2), 1.0, 1.28),
            1.20 + 0.16 * (ratio_zh_r / 0.5)
        )

        v_h_hover = math.sqrt(HOVER_THRUST / (2.0 * 1.225 * ROTOR_DISK_AREA * 4.0))
        v_z_descent = -vel[:, 2]
        is_vrs_band = (v_z_descent > 0.4 * v_h_hover) & (v_z_descent < 1.6 * v_h_hover)
        vrs_factor = torch.ones(self.num_drones, device=self.device)
        if torch.any(is_vrs_band):
            ratio = (v_z_descent[is_vrs_band] - 0.4 * v_h_hover) / (1.2 * v_h_hover)
            vrs_factor[is_vrs_band] = 1.0 - 0.28 * torch.sin(math.pi * ratio)

        lift_scale = ige_factor * vrs_factor
        omega_sq = self.motor_omegas ** 2
        T = torch.zeros_like(self.motor_omegas)
        T[:, 0::2] = C_T_CLEAN * omega_sq[:, 0::2] * lift_scale.unsqueeze(-1)
        T[:, 1::2] = (C_T_CLEAN * COAXIAL_EFFICIENCY) * omega_sq[:, 1::2] * lift_scale.unsqueeze(-1)

        wrench = torch.matmul(self.B_mat, T.unsqueeze(-1)).squeeze(-1)
        return wrench[:, 0], wrench[:, 1:4]

    def allocate(self, thrust: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        w = torch.cat([thrust.unsqueeze(-1), tau], dim=-1)
        T_cmds = torch.clamp(torch.matmul(self.B_pinv, w.unsqueeze(-1)).squeeze(-1), 0.05, 10.5)
        omega_cmds = torch.zeros_like(T_cmds)
        omega_cmds[:, 0::2] = torch.sqrt(T_cmds[:, 0::2] / C_T_CLEAN)
        omega_cmds[:, 1::2] = torch.sqrt(T_cmds[:, 1::2] / (C_T_CLEAN * COAXIAL_EFFICIENCY))
        return omega_cmds

# =============================================================================
# 7. GPU BATCHED MACRO LIDAR & SURFACE ELEVATION ENGINE
# =============================================================================
class VectorizedMacroLiDAR:
    """
    Evaluates 16-beam radial LiDAR and 8-beam downward conical altimeters
    directly on CUDA tensors using slab ray-box intersection.
    """
    def __init__(self, gpu_geom_cache, device="cuda"):
        self.device = device
        self.centers = gpu_geom_cache["centers"]             # (N, 3)
        self.half_extents = gpu_geom_cache["half_extents"]   # (N, 3)
        self.roof_heights = gpu_geom_cache["roof_heights"]   # (N,)

        self.angles_16 = torch.linspace(0, 2.0 * math.pi, 16, device=device, endpoint=False)
        self.cone_angle = math.radians(25.0)
        self.sin_c = math.sin(self.cone_angle)
        self.cos_c = math.cos(self.cone_angle)
        self.conical_angles = torch.linspace(0, 2.0 * math.pi, 8, device=device, endpoint=False)

    def cast_360_lidar_batch(self, pos: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """
        pos: (B, 3), yaw: (B,)
        Returns: (B, 16) ranges clamped to 30.0m.
        """
        B = pos.shape[0]
        # (B, 16)
        ray_angles = yaw.unsqueeze(-1) + self.angles_16.unsqueeze(0)
        dx = torch.cos(ray_angles) # (B, 16)
        dy = torch.sin(ray_angles) # (B, 16)

        ranges = torch.full((B, 16), 30.0, device=self.device)
        bx1 = (self.centers[:, 0] - self.half_extents[:, 0]).unsqueeze(0).unsqueeze(0) # (1, 1, N)
        bx2 = (self.centers[:, 0] + self.half_extents[:, 0]).unsqueeze(0).unsqueeze(0)
        by1 = (self.centers[:, 1] - self.half_extents[:, 1]).unsqueeze(0).unsqueeze(0)
        by2 = (self.centers[:, 1] + self.half_extents[:, 1]).unsqueeze(0).unsqueeze(0)
        bh  = self.roof_heights.unsqueeze(0).unsqueeze(0)

        px = pos[:, 0].unsqueeze(-1).unsqueeze(-1) # (B, 1, 1)
        py = pos[:, 1].unsqueeze(-1).unsqueeze(-1)
        pz = pos[:, 2].unsqueeze(-1).unsqueeze(-1)

        dx_exp = dx.unsqueeze(-1) # (B, 16, 1)
        dy_exp = dy.unsqueeze(-1)

        # Altitude mask: building must be higher than drone altitude
        valid_bldgs = (pz <= (bh + 0.1)) # (B, 1, N)

        # Slab method
        tx1 = (bx1 - px) / torch.where(torch.abs(dx_exp) > 1e-4, dx_exp, torch.sign(dx_exp)*1e-4)
        tx2 = (bx2 - px) / torch.where(torch.abs(dx_exp) > 1e-4, dx_exp, torch.sign(dx_exp)*1e-4)
        ty1 = (by1 - py) / torch.where(torch.abs(dy_exp) > 1e-4, dy_exp, torch.sign(dy_exp)*1e-4)
        ty2 = (by2 - py) / torch.where(torch.abs(dy_exp) > 1e-4, dy_exp, torch.sign(dy_exp)*1e-4)

        t_min_x = torch.minimum(tx1, tx2)
        t_max_x = torch.maximum(tx1, tx2)
        t_min_y = torch.minimum(ty1, ty2)
        t_max_y = torch.maximum(ty1, ty2)

        t_enter = torch.maximum(t_min_x, t_min_y)
        t_exit = torch.minimum(t_max_x, t_max_y)

        hit_mask = (t_enter <= t_exit) & (t_exit > 0.0) & (t_enter < 30.0) & valid_bldgs
        t_enter_masked = torch.where(hit_mask, t_enter, torch.tensor(30.0, device=self.device))
        min_hits, _ = torch.min(t_enter_masked, dim=-1)
        return torch.clamp(min_hits, 0.0, 30.0)

    def cast_downward_conical_batch(self, pos: torch.Tensor) -> torch.Tensor:
        """
        pos: (B, 3)
        Returns: (B, 8) conical beam clearances clamped to 15.0m.
        """
        B = pos.shape[0]
        # (8,)
        sample_dx = self.sin_c * torch.cos(self.conical_angles) * 1.5
        sample_dy = self.sin_c * torch.sin(self.conical_angles) * 1.5

        sample_x = pos[:, 0].unsqueeze(-1) + sample_dx.unsqueeze(0) # (B, 8)
        sample_y = pos[:, 1].unsqueeze(-1) + sample_dy.unsqueeze(0) # (B, 8)
        pz = pos[:, 2].unsqueeze(-1)

        # Query surface altitude
        bx1 = (self.centers[:, 0] - self.half_extents[:, 0]).unsqueeze(0).unsqueeze(0) # (1, 1, N)
        bx2 = (self.centers[:, 0] + self.half_extents[:, 0]).unsqueeze(0).unsqueeze(0)
        by1 = (self.centers[:, 1] - self.half_extents[:, 1]).unsqueeze(0).unsqueeze(0)
        by2 = (self.centers[:, 1] + self.half_extents[:, 1]).unsqueeze(0).unsqueeze(0)
        bh  = self.roof_heights.unsqueeze(0).unsqueeze(0)

        sx = sample_x.unsqueeze(-1) # (B, 8, 1)
        sy = sample_y.unsqueeze(-1)

        inside = (sx >= bx1) & (sx <= bx2) & (sy >= by1) & (sy <= by2)
        roofs = torch.where(inside, bh, torch.tensor(0.0, device=self.device))
        surf_z, _ = torch.max(roofs, dim=-1) # (B, 8)

        clearance = torch.clamp(pz - surf_z, min=0.05)
        return torch.clamp(clearance / max(0.1, self.cos_c), 0.05, 15.0)

# =============================================================================
# 8. HIGH-PERFORMANCE PHYSICAL SENSOR SUITE
# =============================================================================
class FullPhysicsSensorSuite:
    def __init__(self):
        self.powerline_y = 15.0
        self.powerline_z = 12.0

    def poll(self, drone_p, drone_v, wind_3d, surface_z=0.0):
        z_true = max(0.0, float(drone_p[2]))
        t_air = T_0 - L_LAPSE * z_true
        p_static_hpa = P_0 * ((1.0 - (L_LAPSE * z_true) / T_0) ** 5.25588)
        rho_air = (p_static_hpa * 100.0) / (R_SPEC_AIR * t_air)
        z_baro = 44330.8 * (1.0 - ((p_static_hpa / P_0) ** 0.190284)) + float(np.random.normal(0, 0.04))

        v_air = drone_v - wind_3d
        v_tas = float(np.linalg.norm(v_air))
        q_dyn = 0.5 * rho_air * (v_tas ** 2)

        d_wire = math.hypot(drone_p[1] - self.powerline_y, drone_p[2] - self.powerline_z)
        emf_rms = float(140.0 / max(0.25, d_wire))
        clearance = max(0.05, float(drone_p[2] - surface_z))
        min_sonar = max(0.05, min(15.0, clearance))

        return {
            "z_baro": z_baro,
            "rho_air": rho_air,
            "v_tas": v_tas,
            "q_dyn": q_dyn,
            "emf_rms": emf_rms,
            "min_sonar": min_sonar
        }

class AdvancedMultiModalSensorSuite:
    def __init__(self, buildings_list):
        self.buildings = buildings_list

    def poll_optical_flow(self, drone_p, drone_v, drone_R, drone_omega, surface_z=0.0):
        cos_tilt = max(0.35, float(drone_R[2, 2]))
        z_rel = max(0.15, float(drone_p[2] - surface_z))
        z_slant = z_rel / cos_tilt
        v_body = drone_R.T @ drone_v
        w_flow_x = (v_body[0] / z_slant) - drone_omega[1] + float(np.random.normal(0, 0.015))
        w_flow_y = (v_body[1] / z_slant) + drone_omega[0] + float(np.random.normal(0, 0.015))
        quality = 0.95 if z_rel < 35.0 else float(np.clip(1.0 - (z_rel - 35.0) / 25.0, 0.1, 0.95))
        return np.array([w_flow_x, w_flow_y], dtype=np.float32), quality, z_slant

    def poll_aruco_detector(self, drone_p, drone_R, target_pad_xyz):
        rel_pos = target_pad_xyz - drone_p
        z_clearance = max(0.1, float(drone_p[2] - target_pad_xyz[2]))
        dist_xy = float(np.linalg.norm(rel_pos[:2]))
        fov_radius = z_clearance * math.tan(math.radians(35.0))
        is_visible = (dist_xy <= fov_radius) and (z_clearance <= 24.0)

        if is_visible:
            cam_offset = drone_R.T @ rel_pos
            noise = np.random.normal(0, 0.008, 2).astype(np.float32)
            d_psi = math.atan2(drone_R[1, 0], drone_R[0, 0])
            return np.array([cam_offset[0] + noise[0], cam_offset[1] + noise[1], d_psi], dtype=np.float32), 1.0
        return np.zeros(3, dtype=np.float32), 0.0

    def poll_smart_hook_and_load_cell(self, tension_n, sway_2d, cable_l, has_payload):
        if not has_payload or cable_l <= 0.10:
            hook_a = np.zeros(3, dtype=np.float32)
            hook_w = np.zeros(3, dtype=np.float32)
            load_3d = np.array([0.0, 0.0, -tension_n], dtype=np.float32)
            return hook_a, hook_w, load_3d

        hook_a = np.array([-sway_2d[0]*2.2, -sway_2d[1]*2.2, 0.15], dtype=np.float32)
        hook_w = np.array([sway_2d[1]*1.6, -sway_2d[0]*1.6, 0.0], dtype=np.float32)
        sway_dist = max(1e-4, math.hypot(sway_2d[0], sway_2d[1]))
        sin_theta = min(0.65, sway_dist / max(0.2, cable_l))
        cos_theta = math.sqrt(max(1e-4, 1.0 - sin_theta**2))
        fx = -tension_n * (sway_2d[0] / sway_dist) * sin_theta
        fy = -tension_n * (sway_2d[1] / sway_dist) * sin_theta
        fz = -tension_n * cos_theta
        return hook_a, hook_w, np.array([fx, fy, fz], dtype=np.float32)

# =============================================================================
# 9. VECTORIZED 139-DIM OBSERVATION SYNTHESIZER
# =============================================================================
class VectorizedObservationSynthesizer139:
    """
    Constructs the authoritative 139-feature avionics observation directly on GPU tensors.
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.dim = BASE_OBS_DIM

    def synthesize_batch(self, pos, vel, R_flat, omega, target_p, lidar_16, conical_8,
                         cable_len, cable_vel, tension_n, sway_2d, bms_soc, v_pack,
                         temp_c, ice_idx, raw_accel, raw_gyro, peer_rel_p, peer_rel_v):
        B = pos.shape[0]
        out = torch.zeros((B, self.dim), device=self.device, dtype=torch.float32)

        out[:, 0:3] = (target_p - pos) / 50.0
        out[:, 3:6] = pos / 200.0
        out[:, 6:9] = vel / 8.0
        out[:, 9:18] = R_flat
        out[:, 18:21] = omega / 10.0
        out[:, 21:37] = lidar_16 / 30.0
        out[:, 37] = torch.clamp(pos[:, 2], 0.0, 15.0) / 15.0
        out[:, 38] = torch.norm(vel, dim=-1) / 20.0
        out[:, 39] = 0.12
        out[:, 40] = 0.8
        out[:, 41] = 0.5
        out[:, 42:45] = 0.0 # wind
        out[:, 45:53] = conical_8 / 15.0
        out[:, 53] = cable_len / 15.0
        out[:, 54] = cable_vel / 5.0
        out[:, 55] = tension_n / 50.0
        out[:, 56:58] = sway_2d / 2.0
        out[:, 58] = 1.0 # has payload
        out[:, 59:61] = 0.0
        out[:, 61] = bms_soc
        out[:, 62] = v_pack / 25.2
        out[:, 63] = temp_c / 60.0
        out[:, 64] = 0.009 / 0.05
        out[:, 65] = ice_idx
        out[:, 66] = 0.0
        out[:, 67] = 0.12
        out[:, 68] = 0.05

        out[:, 118:121] = peer_rel_p[:, 0] / 50.0
        out[:, 121:124] = peer_rel_v[:, 0] / 10.0
        out[:, 124:127] = peer_rel_p[:, 1] / 50.0
        out[:, 127:130] = peer_rel_v[:, 1] / 10.0

        out[:, 130:133] = raw_accel / 30.0
        out[:, 133:136] = raw_gyro / 10.0
        return out

# =============================================================================
# 10. VECTORIZED LIE GROUP SO(3) OPERATIONS & GEOMETRIC SE(3) CONTROLLER
# =============================================================================
def quat_to_rot_batch(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    B = q.shape[0]
    R = torch.zeros((B, 3, 3), device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1.0 - 2.0 * (y*y + z*z)
    R[:, 0, 1] = 2.0 * (x*y - w*z)
    R[:, 0, 2] = 2.0 * (x*z + w*y)
    R[:, 1, 0] = 2.0 * (x*y + w*z)
    R[:, 1, 1] = 1.0 - 2.0 * (x*x + z*z)
    R[:, 1, 2] = 2.0 * (y*z - w*x)
    R[:, 2, 0] = 2.0 * (x*z - w*y)
    R[:, 2, 1] = 2.0 * (y*z + w*x)
    R[:, 2, 2] = 1.0 - 2.0 * (x*x + y*y)
    return R

def vee_map_batch(M: torch.Tensor) -> torch.Tensor:
    return torch.stack([M[:, 2, 1], M[:, 0, 2], M[:, 1, 0]], dim=-1)

class GeometricSE3Controller:
    def __init__(self, num_drones: int, device: str = "cuda"):
        self.num_drones = num_drones
        self.device = device
        self.k_v = torch.tensor([2.0, 2.0, 2.5], device=device)
        self.k_R = torch.tensor([2.1, 2.1, 1.7], device=device)
        self.k_w = torch.tensor([1.4, 1.4, 0.95], device=device)

    def compute_wrench(self, pos, vel, quat, ang_vel, target_pos, current_mass, nn_guidance=None):
        B = self.num_drones
        R = quat_to_rot_batch(quat)

        delta_p = target_pos - pos
        v_cmd_xy = torch.clamp(delta_p[:, :2] * 1.4, -6.0, 6.0)
        v_cmd_z = torch.clamp(delta_p[:, 2] * 1.2, -1.5, 2.4)
        v_des = torch.cat([v_cmd_xy, v_cmd_z.unsqueeze(-1)], dim=-1)

        a_des = self.k_v * (v_des - vel)
        a_des[:, 2] += GRAVITY_VAL
        if nn_guidance is not None:
            a_des += nn_guidance[:, :3]

        F_d = current_mass * torch.clamp(a_des, -15.0, 22.0)
        b3_d = F_d / torch.clamp(torch.norm(F_d, dim=-1, keepdim=True), min=1e-4)

        delta_xy = target_pos[:, :2] - pos[:, :2]
        dist_xy = torch.norm(delta_xy, dim=-1, keepdim=True)
        unit_track_xy = torch.where(
            dist_xy > 0.5, 
            delta_xy / torch.clamp(dist_xy, min=1e-4), 
            torch.tensor([1.0, 0.0], device=self.device)
        )
        b1_c = torch.cat([unit_track_xy, torch.zeros((B, 1), device=self.device)], dim=-1)

        b2_d = torch.linalg.cross(b3_d, b1_c, dim=-1)
        b2_d = b2_d / torch.clamp(torch.norm(b2_d, dim=-1, keepdim=True), min=1e-4)
        b1_d = torch.linalg.cross(b2_d, b3_d, dim=-1)
        R_d = torch.stack([b1_d, b2_d, b3_d], dim=-1)

        thrust_proj = torch.sum(F_d * R[:, :, 2], dim=-1)
        max_thrust = (2.0 * current_mass * GRAVITY_VAL).view(-1)
        f_thrust = torch.clamp(thrust_proj, min=torch.zeros_like(thrust_proj), max=max_thrust)

        R_err = torch.bmm(R_d.transpose(1, 2), R) - torch.bmm(R.transpose(1, 2), R_d)
        e_R = 0.5 * vee_map_batch(R_err)
        tau = torch.clamp(-self.k_R * e_R - self.k_w * ang_vel, -0.65, 0.65)
        return f_thrust, tau

# =============================================================================
# 11. 6-HEAD RESNET AVIONICS EXECUTIVE BRAIN (~2.2M PARAMS)
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
        return x + self.fc2(self.norm2(self.act(self.fc1(self.norm1(x)))))

class DeepDenseFlightExecutiveNet(nn.Module):
    def __init__(self, in_features=OBSERVATION_DIM):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.SiLU()
        )
        self.block1 = DenseResBlock512(512, 768)
        self.block2 = DenseResBlock512(512, 768)
        self.block3 = DenseResBlock512(512, 768)

        self.guidance_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 4),
            nn.Tanh()
        )
        self.executive_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, NUM_FSM_STATES)
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
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        feat = self.block3(self.block2(self.block1(self.input_proj(x))))

        guidance = self.guidance_head(feat) * 3.0
        fsm_logits = self.executive_head(feat)
        winch_cmd = self.winch_head(feat)
        health_pred = self.health_head(feat)
        trust_scores = self.trust_head(feat)
        value = self.value_head(feat)
        return guidance, fsm_logits, winch_cmd, health_pred, trust_scores, value

# =============================================================================
# 12. HIGH-THROUGHPUT GPU PARALLEL HARVESTING & RESNET TRAINER
# =============================================================================
def train_swarm_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_envs = 64  # 64 parallel world instances = 192 drones on GPU simultaneously

    print("\n" + "="*75)
    print(f"STAGE 1: HIGH-THROUGHPUT PURE-GPU HARVESTING ({n_envs} ENVS / 192 DRONES)")
    print("Zero CPU-GPU Pipeline Stalls Enabled")
    print("="*75)

    if not os.path.exists(URDF_ASSET_FILE):
        generate_coaxial_x8_urdf(URDF_ASSET_FILE)

    try:
        gs.init(backend=gs.gpu)
    except Exception:
        pass

    dt = 0.005
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=2, gravity=(0, 0, -GRAVITY_VAL)),
        show_viewer=False
    )

    world = MetropolitanCityGridWorld400m()
    gpu_geom_cache = build_metropolitan_city_with_hazards(scene, world, device="cuda")
    drones, cargos = spawn_swarm_drones_and_cargo(scene, world.depot_helipads)

    scene.build(n_envs=n_envs)
    print(f"[GPU PIPELINE] Scene built on {torch.cuda.get_device_name(0)}.")

    bemt_engines = [VectorizedBEMTEngine(num_drones=n_envs, device="cuda") for _ in range(3)]
    controllers = [GeometricSE3Controller(num_drones=n_envs, device="cuda") for _ in range(3)]
    lidar_engine = VectorizedMacroLiDAR(gpu_geom_cache, device="cuda")
    obs_synth = VectorizedObservationSynthesizer139(device="cuda")

    # Pre-allocated tensors on CUDA to saturate memory bandwidth
    NUM_HARVEST_STEPS = 1200
    TOTAL_SAMPLES = NUM_HARVEST_STEPS * n_envs
    obs_buffer = torch.zeros((TOTAL_SAMPLES, OBSERVATION_DIM), device="cuda", dtype=torch.float32)
    guidance_target_buf = torch.zeros((TOTAL_SAMPLES, 4), device="cuda", dtype=torch.float32)
    fsm_target_buf = torch.zeros((TOTAL_SAMPLES,), device="cuda", dtype=torch.long)
    winch_target_buf = torch.zeros((TOTAL_SAMPLES, 2), device="cuda", dtype=torch.float32)
    health_target_buf = torch.ones((TOTAL_SAMPLES, 8), device="cuda", dtype=torch.float32)
    trust_target_buf = torch.ones((TOTAL_SAMPLES, 2), device="cuda", dtype=torch.float32)
    dist_target_buf = torch.zeros((TOTAL_SAMPLES, 1), device="cuda", dtype=torch.float32)

    t_start = time.time()
    write_idx = 0

    target_cruise_alts = torch.tensor([34.0, 37.0, 40.0], device="cuda")
    pad_positions = torch.tensor(np.array(world.depot_helipads), device="cuda")

    # Rolling history queues on GPU (3 frames x 139 dims)
    history_tensor = torch.zeros((n_envs, HISTORY_LEN, BASE_OBS_DIM), device="cuda", dtype=torch.float32)

    cable_lens = torch.full((3, n_envs, 1), 0.55, device="cuda")
    has_payloads = torch.ones((3, n_envs), device="cuda", dtype=torch.bool)
    fsm_phases = torch.zeros((3, n_envs), device="cuda", dtype=torch.long)

    print("\n[GPU HARVEST] Stepping physical parallel environments...")

    for step in range(NUM_HARVEST_STEPS):
        d_pos = [drones[i].get_pos() for i in range(3)]     # Each: (n_envs, 3)
        d_vel = [drones[i].get_vel() for i in range(3)]
        d_quat = [drones[i].get_quat() for i in range(3)]
        d_ang = [drones[i].get_ang() for i in range(3)]

        c_pos = [cargos[i].get_pos() for i in range(3)]
        c_vel = [cargos[i].get_vel() for i in range(3)]

        for i in range(3):
            wp_target = pad_positions[i].unsqueeze(0).repeat(n_envs, 1)
            wp_target[:, 2] = target_cruise_alts[i]

            masses = torch.where(has_payloads[i].unsqueeze(-1), torch.tensor(TOTAL_MASS_KG, device="cuda"), torch.tensor(DRY_MASS_KG, device="cuda"))
            f_thrust, tau = controllers[i].compute_wrench(d_pos[i], d_vel[i], d_quat[i], d_ang[i], wp_target, masses)

            motor_cmds = bemt_engines[i].allocate(f_thrust, tau)
            act_thrust, act_tau = bemt_engines[i].compute_aerodynamics(motor_cmds, d_pos[i], d_vel[i], target_surf_z=0.0, dt=dt)
            R_curr = quat_to_rot_batch(d_quat[i])

            f_tether = torch.zeros((n_envs, 3), device="cuda")
            tether_vec = c_pos[i] - d_pos[i]
            tether_dist = torch.norm(tether_vec, dim=-1, keepdim=True)
            slack = tether_dist - cable_lens[i]
            is_taut = (slack > 0.0) & has_payloads[i].unsqueeze(-1)

            tether_dir = tether_vec / torch.clamp(tether_dist, min=1e-4)
            rel_vel = c_vel[i] - d_vel[i]
            damp = torch.sum(rel_vel * tether_dir, dim=-1, keepdim=True)
            t_mag = torch.where(is_taut, torch.clamp(slack * 120.0 + damp * 5.0, 0.0, 20.0), torch.zeros_like(slack))
            f_cargo = -t_mag * tether_dir
            f_tether = t_mag * tether_dir

            cargos[i].links[0].apply_external_force(force=f_cargo)
            tot_f = R_curr[:, :, 2] * act_thrust.unsqueeze(-1) + f_tether
            drones[i].links[0].apply_external_force(force=tot_f)
            drones[i].links[0].apply_external_torque(torque=act_tau)

        scene.step()

        # Pure GPU Sensor Raycasting & Feature Concatenation
        R_d0 = quat_to_rot_batch(d_quat[0])
        yaw_d0 = torch.atan2(R_d0[:, 1, 0], R_d0[:, 0, 0])
        lidar_16 = lidar_engine.cast_360_lidar_batch(d_pos[0], yaw_d0)
        conical_8 = lidar_engine.cast_downward_conical_batch(d_pos[0])

        peer_p = torch.stack([d_pos[1] - d_pos[0], d_pos[2] - d_pos[0]], dim=1) # (n_envs, 2, 3)
        peer_v = torch.stack([d_vel[1] - d_vel[0], d_vel[2] - d_vel[0]], dim=1)

        raw_a = torch.zeros((n_envs, 3), device="cuda")
        raw_a[:, 2] = 9.81
        sway_2d = (c_pos[0][:, :2] - d_pos[0][:, :2])

        obs_139 = obs_synth.synthesize_batch(
            pos=d_pos[0], vel=d_vel[0], R_flat=R_d0.view(n_envs, 9), omega=d_ang[0],
            target_p=wp_target, lidar_16=lidar_16, conical_8=conical_8,
            cable_len=cable_lens[0], cable_vel=torch.norm(c_vel[0] - d_vel[0], dim=-1, keepdim=True),
            tension_n=t_mag, sway_2d=sway_2d, bms_soc=torch.full((n_envs, 1), 0.95, device="cuda"),
            v_pack=torch.full((n_envs, 1), 24.8, device="cuda"), temp_c=torch.full((n_envs, 1), 22.0, device="cuda"),
            ice_idx=torch.zeros((n_envs, 1), device="cuda"), raw_accel=raw_a, raw_gyro=d_ang[0],
            peer_rel_p=peer_p, peer_rel_v=peer_v
        )

        history_tensor = torch.roll(history_tensor, shifts=-1, dims=1)
        history_tensor[:, -1, :] = obs_139
        obs_417 = history_tensor.view(n_envs, OBSERVATION_DIM)

        # Write straight into CUDA buffers
        idx_end = write_idx + n_envs
        obs_buffer[write_idx:idx_end] = obs_417
        guidance_target_buf[write_idx:idx_end, :3] = torch.clamp((wp_target - d_pos[0]) / 5.0, -1.0, 1.0)
        fsm_target_buf[write_idx:idx_end] = 1
        dist_target_buf[write_idx:idx_end] = torch.norm(wp_target - d_pos[0], dim=-1, keepdim=True)
        write_idx = idx_end

    print(f"\n[GPU HARVEST COMPLETE] Collected {TOTAL_SAMPLES:,} transitions in {time.time()-t_start:.2f}s!")

    # Train ResNet with batch size 4096 (High GPU compute saturation)
    model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)

    BATCH_SIZE = 4096
    EPOCHS = 30
    dataset_size = TOTAL_SAMPLES
    n_batches = (dataset_size + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"[POLICY TRAINING] Training with BATCH_SIZE={BATCH_SIZE} across {EPOCHS} epochs...")
    model.train()

    t_train_start = time.time()
    for epoch in range(EPOCHS):
        perm = torch.randperm(dataset_size, device=device)
        epoch_loss = 0.0

        for b in range(0, dataset_size, BATCH_SIZE):
            idx = perm[b:b+BATCH_SIZE]
            bx = obs_buffer[idx]
            bg = guidance_target_buf[idx]
            bf = fsm_target_buf[idx]
            bw = winch_target_buf[idx]
            bh = health_target_buf[idx]
            bt = trust_target_buf[idx]
            bv = dist_target_buf[idx]

            optimizer.zero_grad(set_to_none=True)
            p_g, p_f, p_w, p_h, p_t, p_v = model(bx)

            loss = (
                1.0 * F.mse_loss(p_g, bg) +
                0.5 * F.cross_entropy(p_f, bf) +
                0.3 * F.mse_loss(p_w, bw) +
                0.2 * F.binary_cross_entropy(p_h, bh) +
                0.2 * F.binary_cross_entropy(p_t, bt) +
                0.1 * F.mse_loss(p_v, bv)
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] — Loss: {epoch_loss / n_batches:.4f}")

    print(f"[COMPLETE] Training completed in {time.time()-t_train_start:.2f}s. Saving model...")
    torch.save(model.state_dict(), MODEL_WEIGHTS_FILE)
    print(f"[SUCCESS] Weights saved to '{MODEL_WEIGHTS_FILE}'.")

"""
=============================================================================
SURVIVAL CORE V37.0 — UNIFIED GENESIS MULTI-PHYSICS MASTER SIMULATOR
PART 5 OF 5: CAMERA SYSTEM 2, OCCLUSION SOLVER & 60+ FPS COCKPIT LOOP
=============================================================================
"""

# =============================================================================
# 13. CAMERA SYSTEM 2: GENESIS PERSPECTIVE & OCCLUSION RESOLUTION
# =============================================================================
def _ray_aabb_first_hit(origin, direction, bounds_min, bounds_max):
    t_min, t_max = -float('inf'), float('inf')
    for axis in range(3):
        o, d, lo, hi = float(origin[axis]), float(direction[axis]), float(bounds_min[axis]), float(bounds_max[axis])
        if abs(d) < 1e-9:
            if o < lo or o > hi:
                return None
            continue
        t1, t2 = (lo - o) / d, (hi - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min, t_max = max(t_min, t1), min(t_max, t2)
        if t_max < t_min:
            return None
    if t_max < 0:
        return None
    return t_min if t_min >= 0 else t_max

def resolve_genesis_camera_occlusion(camera_pos, camera_lookat, world, clearance=0.9, minimum_distance=0.65):
    """
    Slides camera forward along view ray if obstructed by buildings, trees, or powerlines.
    """
    target = np.asarray(camera_lookat, dtype=np.float64).reshape(3)
    desired = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    ray = desired - target
    desired_dist = float(np.linalg.norm(ray))
    if not np.isfinite(desired_dist) or desired_dist < 1e-6:
        return desired.astype(np.float32)
    direction = ray / desired_dist
    nearest = desired_dist

    for h in world.houses:
        x1, x2, y1, y2, top = [float(v) for v in h['aabb']]
        t = _ray_aabb_first_hit(target, direction, (x1, y1, 0.0), (x2, y2, top))
        if t is not None and 1.25 < float(t) < nearest:
            nearest = float(t)
        for c in h.get('clutter', []):
            cx1, cx2, cy1, cy2, ctop = [float(v) for v in c['aabb']]
            t = _ray_aabb_first_hit(target, direction, (cx1, cy1, top), (cx2, cy2, ctop))
            if t is not None and 1.25 < float(t) < nearest:
                nearest = float(t)

    for tree in getattr(world, 'trees', []):
        centre = np.array([float(tree['pos'][0]), float(tree['pos'][1]), float(tree.get('h', 8.0)) - float(tree.get('r', 3.2)) / 2.0])
        radius = float(tree.get('r', 3.2)) + 0.75
        oc = target - centre
        b = float(np.dot(oc, direction))
        c = float(np.dot(oc, oc) - radius * radius)
        disc = b * b - c
        if disc >= 0.0:
            t = -b - math.sqrt(disc)
            if 1.25 < t < nearest:
                nearest = t

    for line in getattr(world, 'powerlines', []):
        a = np.asarray(line['p1'], dtype=np.float64)
        b = np.asarray(line['p2'], dtype=np.float64)
        lo = np.minimum(a, b) - np.array([0.5, 0.65, 0.65])
        hi = np.maximum(a, b) + np.array([0.5, 0.65, 0.65])
        t = _ray_aabb_first_hit(target, direction, lo, hi)
        if t is not None and 1.25 < float(t) < nearest:
            nearest = float(t)

    safe_distance = max(float(minimum_distance), nearest - float(clearance))
    if nearest < desired_dist and safe_distance < desired_dist:
        desired = target + direction * safe_distance
    desired[2] = max(0.85, float(desired[2]))
    return desired.astype(np.float32)

def screen_to_world_raycast_genesis(mouse_pos, camera_pos, camera_lookat, view_w, view_h, world, fov_deg=48.0):
    origin = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    lookat = np.asarray(camera_lookat, dtype=np.float64).reshape(3)
    forward = lookat - origin
    forward /= max(1e-9, float(np.linalg.norm(forward)))
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world)
    if np.linalg.norm(right) < 1e-7:
        right = np.array([1.0, 0.0, 0.0])
    right /= max(1e-9, float(np.linalg.norm(right)))
    up = np.cross(right, forward)
    up /= max(1e-9, float(np.linalg.norm(up)))

    mx, my = float(mouse_pos[0]), float(mouse_pos[1])
    aspect = float(view_w) / max(1.0, float(view_h))
    th = math.tan(math.radians(float(fov_deg)) * 0.5)
    x = ((2 * mx / max(1.0, float(view_w))) - 1) * aspect * th
    y = (1 - (2 * my / max(1.0, float(view_h)))) * th
    direction = forward + right * x + up * y
    direction /= max(1e-9, float(np.linalg.norm(direction)))

    best_t, best_point = float('inf'), None
    for h in world.houses:
        x1, x2, y1, y2, top = [float(v) for v in h['aabb']]
        t = _ray_aabb_first_hit(origin, direction, (x1, y1, 0.0), (x2, y2, top))
        if t is not None and 0 <= t < best_t:
            best_t, best_point = t, origin + t * direction
        for c in h.get('clutter', []):
            cx1, cx2, cy1, cy2, ctop = [float(v) for v in c['aabb']]
            tc = _ray_aabb_first_hit(origin, direction, (cx1, cy1, top), (cx2, cy2, ctop))
            if tc is not None and 0 <= tc < best_t:
                best_t, best_point = tc, origin + tc * direction

    if direction[2] < -1e-8:
        tg = -origin[2] / direction[2]
        if 0 <= tg < best_t:
            best_t, best_point = tg, origin + tg * direction

    if best_point is None or not np.all(np.isfinite(best_point)):
        return np.asarray(world.depot_helipads[0], dtype=np.float32).copy()
    return np.asarray(best_point, dtype=np.float32)

class RobustSwarmCamera:
    def __init__(self):
        self.modes = ["CHASE", "BELLY_GIMBAL", "DEPOT_OVERVIEW", "FREE_ORBIT"]
        self.mode_idx = 0
        self.cam_yaw = 0.75
        self.cam_pitch = 0.45
        self.orbit_dist = 28.0
        self.cam_focus = np.array([-120.0, -120.0, 1.0], dtype=np.float32)
        self.snap_needed = True

    def cycle_mode(self):
        self.mode_idx = (self.mode_idx + 1) % len(self.modes)
        self.snap_needed = True
        print(f">>> [CAMERA PERSPECTIVE]: {self.get_current_mode()} <<<")

    def get_current_mode(self):
        return self.modes[self.mode_idx]

    def compute_pose(self, drone_pos, drone_vel, drone_R, target_pos, cable_len, sway_offset, world, dt=0.016):
        mode = self.get_current_mode()
        p = np.nan_to_num(np.asarray(drone_pos, dtype=np.float32), nan=0.0)
        tgt = np.nan_to_num(np.asarray(target_pos, dtype=np.float32), nan=0.0)

        delta = tgt[:2] - p[:2]
        dist = float(np.linalg.norm(delta))
        if dist > 1.0:
            fwd = np.array([delta[0] / dist, delta[1] / dist, 0.0], dtype=np.float32)
        else:
            vxy = np.asarray(drone_vel[:2], dtype=np.float32)
            vn = float(np.linalg.norm(vxy))
            fwd = np.array([vxy[0]/vn, vxy[1]/vn, 0.0], dtype=np.float32) if vn > 0.5 else np.array([1, 0, 0], dtype=np.float32)

        if mode == 'BELLY_GIMBAL':
            cam_pos = p + drone_R @ np.array([-1.10, 0.0, -0.20], dtype=np.float32)
            cam_pos[2] = max(0.85, float(cam_pos[2]))
            cam_lookat = np.array([p[0] + sway_offset[0], p[1] + sway_offset[1], p[2] - max(0.12, cable_len)], dtype=np.float32)
            self.cam_focus = p.copy()

        elif mode == 'DEPOT_OVERVIEW':
            depot_focus = np.array([-120.0, -120.0, 4.0], dtype=np.float32)
            if self.snap_needed:
                self.cam_focus = depot_focus.copy()
                self.snap_needed = False
            else:
                self.cam_focus += (depot_focus - self.cam_focus) * 5.0 * dt
            cam_lookat = self.cam_focus.copy()
            cam_pos = cam_lookat + np.array([38.0 * math.cos(self.cam_yaw), 38.0 * math.sin(self.cam_yaw), 28.0], dtype=np.float32)

        elif mode == 'FREE_ORBIT':
            if self.snap_needed:
                self.cam_focus = p.copy()
                self.snap_needed = False
            else:
                self.cam_focus += (p - self.cam_focus) * 5.0 * dt
            cam_lookat = self.cam_focus + np.array([0.0, 0.0, 1.5], dtype=np.float32)
            pitch = float(np.clip(self.cam_pitch, 0.15, 1.35))
            dx = self.orbit_dist * math.cos(pitch) * math.cos(self.cam_yaw)
            dy = self.orbit_dist * math.cos(pitch) * math.sin(self.cam_yaw)
            dz = self.orbit_dist * math.sin(pitch)
            cam_pos = cam_lookat + np.array([dx, dy, dz], dtype=np.float32)

        else:  # CHASE
            if self.snap_needed:
                self.cam_focus = p.copy()
                self.snap_needed = False
            else:
                self.cam_focus += (p - self.cam_focus) * 5.0 * dt

            yaw = math.atan2(float(fwd[1]), float(fwd[0])) + math.pi + (float(self.cam_yaw) - 0.75)
            pitch = float(np.clip(self.cam_pitch, 0.15, 1.20))
            radius = 28.0
            cam_pos = self.cam_focus + np.array([
                radius * math.cos(pitch) * math.cos(yaw),
                radius * math.cos(pitch) * math.sin(yaw),
                radius * math.sin(pitch) + 7.0
            ], dtype=np.float32)
            cam_lookat = self.cam_focus + np.array([0, 0, 2], dtype=np.float32) + 4.0 * fwd

        cam_pos[2] = max(0.85, float(cam_pos[2]))
        cam_pos = resolve_genesis_camera_occlusion(cam_pos, cam_lookat, world)
        return cam_pos, cam_lookat

# =============================================================================
# 14. HUD & RADAR MINIMAP DRAWING
# =============================================================================
def draw_pfd_artificial_horizon(surface, rect, roll, pitch, alt, airspeed, vspeed, font_small):
    rx, ry, rw, rh = rect
    cx, cy = rx + rw // 2, ry + rh // 2
    pygame.draw.rect(surface, (16, 22, 32), rect)
    pygame.draw.rect(surface, (0, 220, 255), rect, 1)

    pitch_px = int(math.degrees(pitch) * 1.4)
    for deg in range(-90, 91, 10):
        y_pos = cy - pitch_px + int(deg * 1.4 * math.cos(roll))
        x_off = int(22 * math.cos(roll))
        y_off = int(22 * math.sin(roll))
        if ry <= y_pos <= ry + rh:
            pygame.draw.line(surface, (90, 130, 170), (cx - x_off, y_pos - y_off), (cx + x_off, y_pos + y_off), 1)

    hz_dx = int(rw * 0.48 * math.cos(roll))
    hz_dy = int(rw * 0.48 * math.sin(roll))
    pygame.draw.line(surface, (0, 255, 140), (cx - hz_dx, cy - pitch_px - hz_dy), (cx + hz_dx, cy - pitch_px + hz_dy), 2)
    pygame.draw.circle(surface, (255, 220, 40), (cx, cy), 3)

    txt_alt = font_small.render(f"ALT:{alt:4.1f}m", True, (0, 220, 255))
    txt_spd = font_small.render(f"SPD:{airspeed:4.1f}m/s", True, (255, 180, 40))
    surface.blit(txt_spd, (rx + 4, ry + 4))
    surface.blit(txt_alt, (rx + rw - txt_alt.get_width() - 4, ry + 4))

def draw_tactical_radar_minimap(surface, map_rect, buildings, depot_pads, drone_positions, active_idx):
    rx, ry, rw, rh = map_rect
    pygame.draw.rect(surface, (15, 20, 28), map_rect)
    pygame.draw.rect(surface, (0, 220, 255), map_rect, 1)
    scale = rw / 360.0
    cx, cy = rx + rw // 2, ry + rh // 2

    def to_map(x, y): 
        return int(cx + x * scale), int(cy - y * scale)

    for b in buildings:
        bx, by = b["pos"][0], b["pos"][1]
        mp = to_map(bx, by)
        pygame.draw.rect(surface, (45, 55, 68), (mp[0] - 4, mp[1] - 4, 8, 8))

    for pad in depot_pads:
        px, py = to_map(pad[0], pad[1])
        pygame.draw.circle(surface, (0, 220, 255), (px, py), 4, 1)

    d_colors = [(0, 240, 255), (255, 180, 40), (220, 100, 255)]
    for i, dp in enumerate(drone_positions):
        dx, dy = to_map(dp[0], dp[1])
        radius = 5 if i == active_idx else 3
        pygame.draw.circle(surface, d_colors[i], (dx, dy), radius)

# =============================================================================
# 15. HIGH-FPS EVALUATION COCKPIT WITH CAMERA SYSTEM 2
# =============================================================================
def evaluate_swarm_system():
    if not os.path.exists(MODEL_WEIGHTS_FILE):
        print(f"\n[WARNING] Model weights '{MODEL_WEIGHTS_FILE}' not found! Running high-throughput training first...")
        train_swarm_system()

    generate_coaxial_x8_urdf(URDF_ASSET_FILE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE, map_location=device))
    model.eval()

    try:
        gs.init(backend=gs.gpu)
    except Exception:
        pass

    dt = 0.005  # 200 Hz physical simulation
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=2, gravity=(0, 0, -GRAVITY_VAL)),
        show_viewer=False
    )

    world = MetropolitanCityGridWorld400m()
    gpu_geom_cache = build_metropolitan_city_with_hazards(scene, world, device="cuda")
    drones, cargos = spawn_swarm_drones_and_cargo(scene, world.depot_helipads)

    VIEW_W, VIEW_H = 760, 740
    cam_sim = scene.add_camera(
        res=(VIEW_W, VIEW_H),
        pos=(45.0, -45.0, 35.0),
        lookat=(0.0, 0.0, 15.0),
        fov=48,
        GUI=False
    )

    scene.build(n_envs=1)

    bemt = VectorizedBEMTEngine(num_drones=3, device="cuda")
    ctrl = GeometricSE3Controller(num_drones=3, device="cuda")

    pad_positions = torch.tensor(np.array(world.depot_helipads), device="cuda")
    target_positions = pad_positions.clone()
    target_positions[:, 2] = 34.0

    cable_lens = torch.full((3, 1, 1), 0.55, device="cuda")
    has_payloads = torch.ones(3, dtype=torch.bool, device="cuda")

    pygame.init()
    WIN_W, WIN_H = 1200, 740
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Survival Core V37.0 — Camera System 2 Master Cockpit")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("consolas", 13, bold=True)
    font_bold = pygame.font.SysFont("consolas", 11, bold=True)
    font_small = pygame.font.SysFont("consolas", 10)

    camera_ctrl = RobustSwarmCamera()
    selected_drone_idx = 0
    mouse_dragging = False
    last_mouse_pos = (0, 0)
    running = True

    print("\n" + "="*75)
    print("V37.0 COCKPIT ONLINE WITH CAMERA SYSTEM 2 INTEGRATION")
    print("Zero Occlusion Penetration | Frustum Click-Raycast | 60+ FPS Display")
    print("="*75 + "\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if event.pos[0] < VIEW_W:
                        mouse_dragging = True
                        last_mouse_pos = event.pos
                        # 3D Screen click-to-world picker
                        clicked_world = screen_to_world_raycast_genesis(
                            mouse_pos=event.pos,
                            camera_pos=cam_sim_pos,
                            camera_lookat=cam_sim_lookat,
                            view_w=VIEW_W,
                            view_h=VIEW_H,
                            world=world,
                            fov_deg=48.0
                        )
                        target_positions[selected_drone_idx] = torch.tensor(clicked_world, device="cuda")
                        target_positions[selected_drone_idx, 2] = max(clicked_world[2] + 4.0, 34.0)
                        print(f">>> [RAYCAST TARGET] Drone {selected_drone_idx+1} dispatched to ({clicked_world[0]:.1f}, {clicked_world[1]:.1f}) <<<")
                elif event.button == 4:
                    camera_ctrl.orbit_dist = max(5.0, camera_ctrl.orbit_dist - 1.5)
                elif event.button == 5:
                    camera_ctrl.orbit_dist = min(50.0, camera_ctrl.orbit_dist + 1.5)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if mouse_dragging:
                    dx = event.pos[0] - last_mouse_pos[0]
                    dy = event.pos[1] - last_mouse_pos[1]
                    camera_ctrl.cam_yaw -= dx * 0.008
                    camera_ctrl.cam_pitch = float(np.clip(camera_ctrl.cam_pitch - dy * 0.008, 0.15, 1.35))
                    last_mouse_pos = event.pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    selected_drone_idx = (selected_drone_idx + 1) % 3
                    camera_ctrl.snap_needed = True
                elif event.key == pygame.K_c:
                    camera_ctrl.cycle_mode()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: camera_ctrl.cam_yaw -= 0.03
        if keys[pygame.K_RIGHT]: camera_ctrl.cam_yaw += 0.03
        if keys[pygame.K_UP]: camera_ctrl.cam_pitch = min(1.35, camera_ctrl.cam_pitch + 0.02)
        if keys[pygame.K_DOWN]: camera_ctrl.cam_pitch = max(0.15, camera_ctrl.cam_pitch - 0.02)

        # Physics Substepping (4 x 0.005 = 0.02s per frame)
        for _ in range(4):
            d_pos = [drones[i].get_pos() for i in range(3)]
            d_vel = [drones[i].get_vel() for i in range(3)]
            d_quat = [drones[i].get_quat() for i in range(3)]
            d_ang = [drones[i].get_ang() for i in range(3)]

            c_pos = [cargos[i].get_pos() for i in range(3)]
            c_vel = [cargos[i].get_vel() for i in range(3)]

            for i in range(3):
                masses = torch.tensor([[TOTAL_MASS_KG]], device="cuda")
                f_thrust, tau = ctrl.compute_wrench(d_pos[i], d_vel[i], d_quat[i], d_ang[i], target_positions[i].unsqueeze(0), masses)
                motor_cmds = bemt.allocate(f_thrust, tau)
                act_thrust, act_tau = bemt.compute_aerodynamics(motor_cmds, d_pos[i], d_vel[i], target_surf_z=0.0, dt=dt)

                R_curr = quat_to_rot_batch(d_quat[i])
                tether_vec = c_pos[i] - d_pos[i]
                tether_dist = torch.norm(tether_vec, dim=-1, keepdim=True)
                slack = tether_dist - cable_lens[i]
                is_taut = (slack > 0.0)

                tether_dir = tether_vec / torch.clamp(tether_dist, min=1e-4)
                rel_vel = c_vel[i] - d_vel[i]
                damp = torch.sum(rel_vel * tether_dir, dim=-1, keepdim=True)
                t_mag = torch.where(is_taut, torch.clamp(slack * 120.0 + damp * 5.0, 0.0, 20.0), torch.zeros_like(slack))
                f_cargo = -t_mag * tether_dir
                f_tether = t_mag * tether_dir

                cargos[i].links[0].apply_external_force(force=f_cargo)
                tot_f = R_curr[:, :, 2] * act_thrust.unsqueeze(-1) + f_tether
                drones[i].links[0].apply_external_force(force=tot_f)
                drones[i].links[0].apply_external_torque(torque=act_tau)

            scene.step()

        # Update Camera System 2
        active_p = d_pos[selected_drone_idx][0].cpu().numpy()
        active_v = d_vel[selected_drone_idx][0].cpu().numpy()
        active_R = quat_to_rot_batch(d_quat[selected_drone_idx])[0].cpu().numpy()
        active_tgt = target_positions[selected_drone_idx].cpu().numpy()
        sway_offset = (c_pos[selected_drone_idx][0][:2] - d_pos[selected_drone_idx][0][:2]).cpu().numpy()

        cam_sim_pos, cam_sim_lookat = camera_ctrl.compute_pose(
            drone_pos=active_p,
            drone_vel=active_v,
            drone_R=active_R,
            target_pos=active_tgt,
            cable_len=0.55,
            sway_offset=sway_offset,
            world=world,
            dt=0.016
        )

        cam_sim.set_pose(pos=tuple(cam_sim_pos), lookat=tuple(cam_sim_lookat))

        # Offscreen Buffer Render & Display
        try:
            rgb = cam_sim.render(rgb=True)[0]
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.detach().cpu().numpy()
            if rgb.ndim == 4:
                rgb = rgb[0]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * 255 if rgb.max() <= 1.0 else rgb, 0, 255).astype(np.uint8)
            sim_surf = pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), 'RGB')
            screen.blit(sim_surf, (0, 0))
        except Exception:
            pygame.draw.rect(screen, (10, 13, 20), (0, 0, VIEW_W, VIEW_H))

        # Right Panel: Avionics Telemetry
        pygame.draw.rect(screen, (16, 20, 28), (VIEW_W, 0, WIN_W - VIEW_W, WIN_H))
        pygame.draw.line(screen, (35, 45, 60), (VIEW_W, 0), (VIEW_W, WIN_H), 2)

        screen.blit(font_title.render("SURVIVAL CORE V37.0 HUD", True, (0, 220, 255)), (VIEW_W + 15, 15))
        screen.blit(font_small.render("Camera System 2 Active (Ray-AABB Occlusion)", True, (130, 150, 175)), (VIEW_W + 15, 34))

        roll = math.atan2(active_R[2, 1], active_R[2, 2])
        pitch = math.asin(float(np.clip(-active_R[2, 0], -1.0, 1.0)))

        draw_pfd_artificial_horizon(
            screen, (VIEW_W + 15, 55, 195, 135), roll, pitch, active_p[2], float(np.linalg.norm(active_v)), active_v[2], font_small
        )

        draw_tactical_radar_minimap(
            screen, (VIEW_W + 220, 55, 205, 135), world.houses, world.depot_helipads,
            [d_pos[j][0].cpu().numpy() for j in range(3)], selected_drone_idx
        )

        ty = 205
        pygame.draw.line(screen, (35, 45, 60), (VIEW_W + 15, ty), (WIN_W - 15, ty), 1)
        screen.blit(font_bold.render("CAMERA SYSTEM 2 TELEMETRY:", True, (240, 240, 240)), (VIEW_W + 15, ty + 6))
        screen.blit(font_small.render(f"Perspective Mode:   {camera_ctrl.get_current_mode()} [C]", True, (0, 220, 255)), (VIEW_W + 25, ty + 24))
        screen.blit(font_small.render(f"Camera Pos:         [{cam_sim_pos[0]:5.1f}, {cam_sim_pos[1]:5.1f}, {cam_sim_pos[2]:4.1f}] m", True, (160, 180, 205)), (VIEW_W + 25, ty + 40))
        screen.blit(font_small.render(f"LookAt Target:      [{cam_sim_lookat[0]:5.1f}, {cam_sim_lookat[1]:5.1f}, {cam_sim_lookat[2]:4.1f}] m", True, (160, 180, 205)), (VIEW_W + 25, ty + 56))
        screen.blit(font_small.render(f"Active Focus:       Drone #{selected_drone_idx+1} [TAB]", True, (0, 255, 140)), (VIEW_W + 25, ty + 72))

        ty += 96
        pygame.draw.line(screen, (35, 45, 60), (VIEW_W + 15, ty), (WIN_W - 15, ty), 1)
        screen.blit(font_bold.render("AVIONICS KINEMATICS:", True, (240, 240, 240)), (VIEW_W + 15, ty + 6))
        screen.blit(font_small.render(f"Drone Altitude:     {active_p[2]:4.1f} m", True, (0, 220, 255)), (VIEW_W + 25, ty + 24))
        screen.blit(font_small.render(f"Ground Speed:       {float(np.linalg.norm(active_v)):4.1f} m/s", True, (160, 180, 205)), (VIEW_W + 25, ty + 40))
        screen.blit(font_small.render(f"Target Waypoint:    [{active_tgt[0]:5.1f}, {active_tgt[1]:5.1f}, {active_tgt[2]:4.1f}] m", True, (255, 180, 40)), (VIEW_W + 25, ty + 56))

        # Viewport Status Overlay
        banner = f"FOCUS: D{selected_drone_idx+1} | {camera_ctrl.get_current_mode()} | ALT: {active_p[2]:.1f}m | CLICK VIEWPORT TO RETARGET"
        banner_surf = font_bold.render(banner, True, (240, 240, 240))
        pygame.draw.rect(screen, (16, 20, 28), (10, 10, banner_surf.get_width() + 16, 26), border_radius=4)
        pygame.draw.rect(screen, (0, 220, 255), (10, 10, banner_surf.get_width() + 16, 1), border_radius=4)
        screen.blit(banner_surf, (18, 16))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

# =============================================================================
# 16. ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_swarm_system()
    else:
        evaluate_swarm_system()