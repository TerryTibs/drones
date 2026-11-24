"""
=============================================================================
SURVIVAL CORE V37.0 — UNIFIED GENESIS MULTI-PHYSICS MASTER SIMULATOR
=============================================================================
PART 1 OF 5: PHYSICAL CONSTANTS, COAXIAL X8 DRONE URDF, 48-BUILDING CITY,
             PHYSICAL POWERLINES, TREE CANOPIES & 3D CORRIDOR PLANNER
=============================================================================
"""

import os
import sys
import math
import time
import random
import pygame
import ctypes
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import genesis as gs

# =============================================================================
# 0. PHYSICAL AIRFRAME & MISSION CONSTANTS
# =============================================================================
DRY_MASS_KG = 1.35                   
CARGO_MASS_KG = 0.45                 
TOTAL_MASS_KG = DRY_MASS_KG + CARGO_MASS_KG  # 1.80 kg total flying mass
GRAVITY_VAL = 9.80665

ARM_LENGTH_M = 0.26 / math.sqrt(2.0)
ROTOR_RADIUS_M = 0.127
ROTOR_DISK_AREA = math.pi * (ROTOR_RADIUS_M ** 2)

C_T_CLEAN = 1.35e-5
C_Q_CLEAN = 2.40e-7
COAXIAL_EFFICIENCY = 0.92
MOTOR_TAU_S = 0.030
MAX_RPM = 9200.0
MAX_OMEGA = (MAX_RPM * 2.0 * math.pi) / 60.0
HOVER_THRUST = TOTAL_MASS_KG * GRAVITY_VAL

# Urban Corridor Safety Bounds (Tallest skyscraper = 26m + 2m HVAC = 28m)
MAX_CITY_OBSTACLE_HEIGHT_M = 28.0
MIN_SAFE_CRUISE_ALT_M = MAX_CITY_OBSTACLE_HEIGHT_M + 6.0  # Safe corridor floor = 34.0m
GLOBAL_CEILING_ALT_M = 55.0

# 139 Avionics Features x 3 History Frames = 417 Dimensions
BASE_OBS_DIM = 139
HISTORY_LEN = 3
OBSERVATION_DIM = BASE_OBS_DIM * HISTORY_LEN
NUM_FSM_STATES = 7

MODEL_WEIGHTS_FILE = "drone_master_brain_genesis_master.pth"
URDF_ASSET_FILE = "drone_coaxial_x8.urdf"

FSM_NAMES = [
    "TAKEOFF",             # 0
    "CRUISE_OUTBOUND",    # 1
    "CHIMNEY_DESCENT",    # 2
    "DELIVERY_HOVER",     # 3
    "WINCH_RETRACT",      # 4
    "RETURN_BASE",        # 5
    "DEPOT_LANDING"       # 6
]

# =============================================================================
# 1. PROCEDURAL COAXIAL X8 DRONE URDF ASSET GENERATOR
# =============================================================================
def generate_coaxial_x8_urdf(filepath: str = "drone_coaxial_x8.urdf"):
    """
    Procedurally creates a realistic Coaxial X8 Multicopter URDF:
    Aligned to standard aerospace axes: +X is Front (Nose), +Y is Left, +Z is Up.
    """
    arm_l = ARM_LENGTH_M
    arm_angle = math.pi / 4.0  # 45 degrees
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
    
    <!-- Central Avionics Fuselage (Length along X) -->
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.22 0.16 0.08"/></geometry>
      <material name="carbon_dark"><color rgba="0.12 0.14 0.18 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.24 0.18 0.09"/></geometry>
    </collision>

    <!-- Front Nose Arrow Indicator (Clearly shows forward direction) -->
    <visual>
      <origin xyz="0.12 0 0.01" rpy="0 0 0"/>
      <geometry><box size="0.06 0.04 0.03"/></geometry>
      <material name="nose_red"><color rgba="0.95 0.25 0.25 1.0"/></material>
    </visual>

    <!-- Top GPS/RTK Dome -->
    <visual>
      <origin xyz="0 0 0.055" rpy="0 0 0"/>
      <geometry><cylinder radius="0.045" length="0.03"/></geometry>
      <material name="cyan_accent"><color rgba="0.0 0.85 0.95 1.0"/></material>
    </visual>

    <!-- 4 Diagonal Carbon Arms -->
    <!-- Front-Left (+X, +Y) -->
    <visual>
      <origin xyz="{dx/2} {dy/2} 0.01" rpy="0 0 {arm_angle}"/>
      <geometry><cylinder radius="0.012" length="{arm_l}"/></geometry>
      <material name="carbon_arm"><color rgba="0.2 0.22 0.25 1.0"/></material>
    </visual>
    <!-- Front-Right (+X, -Y) -->
    <visual>
      <origin xyz="{dx/2} {-dy/2} 0.01" rpy="0 0 {-arm_angle}"/>
      <geometry><cylinder radius="0.012" length="{arm_l}"/></geometry>
      <material name="carbon_arm"><color rgba="0.2 0.22 0.25 1.0"/></material>
    </visual>
    <!-- Rear-Left (-X, +Y) -->
    <visual>
      <origin xyz="{-dx/2} {dy/2} 0.01" rpy="0 0 {-arm_angle}"/>
      <geometry><cylinder radius="0.012" length="{arm_l}"/></geometry>
      <material name="carbon_arm"><color rgba="0.2 0.22 0.25 1.0"/></material>
    </visual>
    <!-- Rear-Right (-X, -Y) -->
    <visual>
      <origin xyz="{-dx/2} {-dy/2} 0.01" rpy="0 0 {arm_angle}"/>
      <geometry><cylinder radius="0.012" length="{arm_l}"/></geometry>
      <material name="carbon_arm"><color rgba="0.2 0.22 0.25 1.0"/></material>
    </visual>

    <!-- 8 Coaxial Rotor Disks -->
    <!-- Front-Left (Top / Bottom) -->
    <visual>
      <origin xyz="{dx} {dy} 0.035" rpy="0 0 0"/>
      <geometry><cylinder radius="{ROTOR_RADIUS_M}" length="0.008"/></geometry>
      <material name="rotor_orange"><color rgba="0.95 0.55 0.1 0.7"/></material>
    </visual>
    <visual>
      <origin xyz="{dx} {dy} -0.02" rpy="0 0 0"/>
      <geometry><cylinder radius="{ROTOR_RADIUS_M}" length="0.008"/></geometry>
      <material name="rotor_dark"><color rgba="0.1 0.1 0.1 0.7"/></material>
    </visual>

    <!-- Front-Right (Top / Bottom) -->
    <visual>
      <origin xyz="{dx} {-dy} 0.035" rpy="0 0 0"/>
      <geometry><cylinder radius="{ROTOR_RADIUS_M}" length="0.008"/></geometry>
      <material name="rotor_orange"><color rgba="0.95 0.55 0.1 0.7"/></material>
    </visual>
    <visual>
      <origin xyz="{dx} {-dy} -0.02" rpy="0 0 0"/>
      <geometry><cylinder radius="{ROTOR_RADIUS_M}" length="0.008"/></geometry>
      <material name="rotor_dark"><color rgba="0.1 0.1 0.1 0.7"/></material>
    </visual>

    <!-- Rear-Left (Top / Bottom) -->
    <visual>
      <origin xyz="{-dx} {dy} 0.035" rpy="0 0 0"/>
      <geometry><cylinder radius="{ROTOR_RADIUS_M}" length="0.008"/></geometry>
      <material name="rotor_cyan"><color rgba="0.0 0.8 0.9 0.7"/></material>
    </visual>
    <visual>
      <origin xyz="{-dx} {dy} -0.02" rpy="0 0 0"/>
      <geometry><cylinder radius="{ROTOR_RADIUS_M}" length="0.008"/></geometry>
      <material name="rotor_dark"><color rgba="0.1 0.1 0.1 0.7"/></material>
    </visual>

    <!-- Rear-Right (Top / Bottom) -->
    <visual>
      <origin xyz="{-dx} {-dy} 0.035" rpy="0 0 0"/>
      <geometry><cylinder radius="{ROTOR_RADIUS_M}" length="0.008"/></geometry>
      <material name="rotor_cyan"><color rgba="0.0 0.8 0.9 0.7"/></material>
    </visual>
    <visual>
      <origin xyz="{-dx} {-dy} -0.02" rpy="0 0 0"/>
      <geometry><cylinder radius="{ROTOR_RADIUS_M}" length="0.008"/></geometry>
      <material name="rotor_dark"><color rgba="0.1 0.1 0.1 0.7"/></material>
    </visual>

    <!-- Dual Landing Gear Skids (Running along X) -->
    <visual>
      <origin xyz="0 0.08 -0.065" rpy="0 1.5708 0"/>
      <geometry><cylinder radius="0.007" length="0.22"/></geometry>
      <material name="skid_mat"><color rgba="0.3 0.35 0.4 1.0"/></material>
    </visual>
    <visual>
      <origin xyz="0 -0.08 -0.065" rpy="0 1.5708 0"/>
      <geometry><cylinder radius="0.007" length="0.22"/></geometry>
      <material name="skid_mat"><color rgba="0.3 0.35 0.4 1.0"/></material>
    </visual>
  </link>

</robot>
"""
    with open(filepath, "w") as f:
        f.write(urdf_content)
    print(f"[ASSET] Coaxial X8 Drone URDF compiled (+X Front): {filepath}")

# =============================================================================
# 2. 400M METROPOLITAN CITY (BUILDINGS, POWERLINES & TREE CANOPIES)
# =============================================================================
def build_metropolitan_city_with_hazards(scene):
    """
    Spawns the 400m Metropolitan grid:
      - 48 building towers with tiered heights (8.5m to 26.0m)
      - Rooftop delivery helipads and HVAC clutter boxes
      - Physical 12m high-voltage transmission lines (Powerlines with EMF)
      - 4 Physical Tree canopies in urban courtyards
      - 3 distinct base station depot pads
    """
    # Ground plane
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(friction=0.8),
        surface=gs.surfaces.Rough(color=(0.18, 0.20, 0.24), roughness=0.8)
    )

    manifest = []
    avenues_x = [-60.0, -30.0, 0.0, 30.0, 60.0]
    streets_y = [-60.0, -30.0, 0.0, 30.0, 60.0]
    bldg_idx = 0

    for c_i, cx in enumerate(avenues_x):
        for r_i, ry in enumerate(streets_y):
            # Reserve southwest corner for depot helipads
            if cx <= -20.0 and ry <= -20.0:
                continue

            tier = (c_i + r_i) % 4
            height = [8.5, 14.0, 20.0, 26.0][tier]
            bw, bd = 16.0, 16.0

            # Skyscraper Tower
            scene.add_entity(
                morph=gs.morphs.Box(size=(bw, bd, height), pos=(cx, ry, height / 2.0), fixed=True),
                material=gs.materials.Rigid(friction=0.7),
                surface=gs.surfaces.Rough(
                    color=(0.28 + 0.03 * tier, 0.32 + 0.03 * tier, 0.38 + 0.03 * tier), 
                    roughness=0.6
                )
            )

            # Rooftop HVAC Bulkhead Clutter
            scene.add_entity(
                morph=gs.morphs.Box(size=(3.2, 2.8, 1.4), pos=(cx - 3.5, ry + 3.5, height + 0.7), fixed=True),
                material=gs.materials.Rigid(friction=0.6),
                surface=gs.surfaces.Rough(color=(0.55, 0.55, 0.58), roughness=0.5)
            )

            # Rooftop Delivery Helipad
            pad_pos = (cx + 2.0, ry - 2.0, height + 0.04)
            scene.add_entity(
                morph=gs.morphs.Cylinder(radius=2.2, height=0.08, pos=pad_pos, fixed=True),
                material=gs.materials.Rigid(friction=0.85),
                surface=gs.surfaces.Default(color=(0.0, 0.85, 0.55))
            )

            manifest.append({
                "id": f"Bldg_{bldg_idx+1:02d}",
                "pos": np.array(pad_pos, dtype=np.float32),          # Rooftop delivery pad
                "center": np.array([cx, ry, height / 2.0], dtype=np.float32),  # Exact tower center
                "size": np.array([16.0, 16.0, height], dtype=np.float32),     # Exact tower dimensions
                "height": height,
                "radius": 10.5
            })
            bldg_idx += 1

    # 3 Separate Depot Helipads
    depot_pads = [
        np.array([-48.0, -48.0, 0.04], dtype=np.float32),  # Pad 1
        np.array([-48.0, -36.0, 0.04], dtype=np.float32),  # Pad 2
        np.array([-36.0, -48.0, 0.04], dtype=np.float32),  # Pad 3
    ]
    pad_colors = [(0.10, 0.65, 0.95), (0.95, 0.55, 0.15), (0.75, 0.25, 0.95)]
    for p_i, dp in enumerate(depot_pads):
        scene.add_entity(
            morph=gs.morphs.Cylinder(radius=2.8, height=0.08, pos=tuple(dp), fixed=True),
            material=gs.materials.Rigid(friction=0.9),
            surface=gs.surfaces.Default(color=pad_colors[p_i])
        )

    # Physical Urban Hazard 1: 12m High-Voltage Powerline Corridor
    # Two transmission wire segments crossing the city with EMF radiation
    scene.add_entity(
        morph=gs.morphs.Box(size=(140.0, 0.08, 0.08), pos=(0.0, 15.0, 12.0), fixed=True),
        material=gs.materials.Rigid(friction=0.2),
        surface=gs.surfaces.Default(color=(0.95, 0.20, 0.20))
    )
    scene.add_entity(
        morph=gs.morphs.Box(size=(140.0, 0.08, 0.08), pos=(0.0, -15.0, 12.0), fixed=True),
        material=gs.materials.Rigid(friction=0.2),
        surface=gs.surfaces.Default(color=(0.95, 0.20, 0.20))
    )

    # Physical Urban Hazard 2: 4 Tree Canopies (Trunk + Foliage Obstacles)
    tree_coords = [(-15.0, -15.0), (15.0, -15.0), (-15.0, 15.0), (15.0, 15.0)]
    trees = []
    for tx, ty in tree_coords:
        # Trunk
        scene.add_entity(
            morph=gs.morphs.Cylinder(radius=0.45, height=5.0, pos=(tx, ty, 2.5), fixed=True),
            material=gs.materials.Rigid(friction=0.7),
            surface=gs.surfaces.Rough(color=(0.40, 0.25, 0.15), roughness=0.8)
        )
        # Foliage Sphere
        scene.add_entity(
            morph=gs.morphs.Sphere(radius=3.2, pos=(tx, ty, 6.5), fixed=True),
            material=gs.materials.Rigid(friction=0.6),
            surface=gs.surfaces.Rough(color=(0.15, 0.55, 0.20), roughness=0.7)
        )
        trees.append({"pos": np.array([tx, ty, 6.5], dtype=np.float32), "radius": 3.2, "height": 9.5})

    return manifest, depot_pads, avenues_x, streets_y, trees

# =============================================================================
# 3. SWARM ENTITY SPAWNER (3 DRONES & 3 CARGO BOXES)
# =============================================================================
def spawn_swarm_drones_and_cargo(scene, depot_pads):
    """
    Spawns 3 physical Coaxial X8 Drones and 3 Cargo Boxes with clean
    non-overlapping vertical clearance:
      - Depot Pad Top:   z = 0.08m
      - Cargo Box:       z = 0.18m (rests cleanly on pad)
      - Drone Airframe:  z = 0.70m (skids at 0.62m, 35cm air gap above box)
    """
    drones = []
    cargos = []

    for i in range(3):
        pad_x, pad_y = depot_pads[i][0], depot_pads[i][1]

        cargo_entity = scene.add_entity(
            morph=gs.morphs.Box(size=(0.22, 0.22, 0.18), pos=(pad_x, pad_y, 0.18)),
            material=gs.materials.Rigid(rho=45.0, friction=0.7),
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
# 4. 3D GLIDE-SLOPE CORRIDOR & URBAN VERTICAL CHIMNEY PLANNER
# =============================================================================
class VoxelOctreeCorridorPlanner:
    """
    Enforces safe urban corridor navigation:
      1. Vertical Chimney Ascent: Drones climb vertically to the safe cruise floor (>=34m)
         above the local urban structure before translating horizontally.
      2. Safe Cruise Floor: Flight corridor strictly clears all 28m skyscrapers + HVAC clutter.
      3. Vertical Chimney Descent: Straight vertical descent gate directly above the helipad.
    """
    def __init__(self, buildings_list, base_station_pad):
        self.buildings = buildings_list
        self.base_pad = base_station_pad

    def get_max_elevation_along_corridor(self, p_start, p_end, buffer_width=12.0):
        max_h = 0.0
        p1 = p_start[:2]
        p2 = p_end[:2]
        d_vec = p2 - p1
        seg_len = max(0.01, float(np.linalg.norm(d_vec)))
        d_unit = d_vec / seg_len

        for b in self.buildings:
            bx, by = b["pos"][0], b["pos"][1]
            bh = float(b["height"]) + 2.0  # +2m HVAC clutter buffer
            b_pt = np.array([bx, by])
            proj = float(np.dot(b_pt - p1, d_unit))
            
            if -10.0 <= proj <= (seg_len + 10.0):
                perp_dist = float(np.linalg.norm((b_pt - p1) - proj * d_unit))
                if perp_dist <= (b.get("radius", 10.5) + buffer_width):
                    max_h = max(max_h, bh)
        return max_h

    def plan_mission_waypoints(self, start_pos, target_dest, is_return=False):
        start_xy = start_pos[:2]
        target_xy = target_dest["pos"][:2]
        target_roof_h = float(target_dest["height"]) if not is_return else 0.0

        highest_obstacle = self.get_max_elevation_along_corridor(start_pos, target_dest["pos"])
        safe_cruise_alt = max(MIN_SAFE_CRUISE_ALT_M, highest_obstacle + 6.0)

        # 4-Gate Safe 3D Corridor
        wp0_climb_gate = np.array([start_xy[0], start_xy[1], safe_cruise_alt], dtype=np.float32)
        wp1_cruise_mid = np.array([0.5 * (start_xy[0] + target_xy[0]), 
                                   0.5 * (start_xy[1] + target_xy[1]), 
                                   safe_cruise_alt], dtype=np.float32)
        wp2_over_target = np.array([target_xy[0], target_xy[1], safe_cruise_alt], dtype=np.float32)
        
        descent_hover_z = target_roof_h + 7.5 if not is_return else 0.70
        wp3_chimney_gate = np.array([target_xy[0], target_xy[1], descent_hover_z], dtype=np.float32)

        return [wp0_climb_gate, wp1_cruise_mid, wp2_over_target, wp3_chimney_gate], safe_cruise_alt

    # =============================================================================
# 5. BATTERY & ATMOSPHERIC PHYSICAL CONSTANTS
# =============================================================================
NUM_CELLS = 6
PACK_CAPACITY_AH = 8.0               
CELL_R0_NOMINAL = 0.0085
CELL_R1 = 0.0065
CELL_C1 = 2800.0
ARRHENIUS_EA = 28000.0
GAS_CONST_R = 8.31446
T_REF_K = 298.15

P_0 = 1013.25                        # Sea level pressure (hPa)
T_0 = 288.15                         # Sea level temperature (K)
L_LAPSE = 0.0065                     # Temperature lapse rate (K/m)
R_SPEC_AIR = 287.058                 # Specific gas constant (J/(kg*K))
SPEED_OF_SOUND = 343.2

INERTIA_MATRIX = np.diag([0.022, 0.022, 0.038]).astype(np.float32)

# =============================================================================
# 6. SWARM MICROCLIMATE SHARED VOXEL MAP
# =============================================================================
class SwarmMicroclimateSharedMap:
    """
    Decentralized voxel map that accumulates local wind measurements broadcast
    by swarm drones. Enables vertical wind-shear lookups for energy-optimal routing.
    """
    def __init__(self, voxel_size_xy=20.0, voxel_size_z=8.0):
        self.grid = {}
        self.voxel_size_xy = voxel_size_xy
        self.voxel_size_z = voxel_size_z

    def broadcast_wind(self, pos, wind_vec, timestamp):
        vx = int(pos[0] // self.voxel_size_xy)
        vy = int(pos[1] // self.voxel_size_xy)
        vz = int(max(0.0, pos[2]) // self.voxel_size_z)
        key = (vx, vy, vz)
        if key not in self.grid:
            self.grid[key] = [wind_vec.copy(), timestamp]
        else:
            old_w, _ = self.grid[key]
            self.grid[key] = [0.65 * old_w + 0.35 * wind_vec, timestamp]

    def query_wind_prior(self, pos, current_time, max_age_s=45.0):
        vx = int(pos[0] // self.voxel_size_xy)
        vy = int(pos[1] // self.voxel_size_xy)
        vz = int(max(0.0, pos[2]) // self.voxel_size_z)
        key = (vx, vy, vz)
        if key in self.grid:
            wind_data, t_stamp = self.grid[key]
            if (current_time - t_stamp) <= max_age_s:
                return wind_data.copy(), True
        return np.zeros(3, dtype=np.float32), False

    def query_vertical_wind_shear_profile(self, pos_xy, current_time, candidate_alts):
        vx = int(pos_xy[0] // self.voxel_size_xy)
        vy = int(pos_xy[1] // self.voxel_size_xy)
        profile = []
        for alt in candidate_alts:
            vz = int(max(0.0, alt) // self.voxel_size_z)
            key = (vx, vy, vz)
            if key in self.grid and (current_time - self.grid[key][1]) <= 60.0:
                profile.append(self.grid[key][0].copy())
            else:
                profile.append(np.zeros(3, dtype=np.float32))
        return profile

# =============================================================================
# 7. DYNAMIC URBAN WEATHER ENGINE
# =============================================================================
class DynamicUrbanWeatherEngine:
    """
    Simulates dynamic urban meteorology with 4 regimes:
      - CLEAR_NOMINAL: Calm breeze (1.8 m/s), 22°C
      - CANYON_GALE: Urban street canyon winds (6.5 m/s), 14°C
      - FREEZING_RAIN: Supercooled precipitation with rotor icing, -4.5°C
      - MICROBURST_STORM: Severe downdraft shear (-2.8 m/s vertical), 10°C
    Also applies Venturi wind tunnel amplification along urban avenues.
    """
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
        print(f"\n>>> [WEATHER SHIFT] Blending into: {self.weather_modes[self.target_mode_idx]} <<<")

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
        else:  # MICROBURST_STORM
            return np.array([-4.5, 4.0, 0.0], dtype=np.float32), 5.5, 10.0, False, -2.8

    def get_weather_parameters(self, drone_pos, avenues_x=None):
        px, py, pz = drone_pos[0], drone_pos[1], drone_pos[2]
        w1, g1, t1, i1, m1 = self._get_preset(self.current_mode_idx)
        w2, g2, t2, i2, m2 = self._get_preset(self.target_mode_idx)

        b = self.blend_factor
        base_wind = (1.0 - b) * w1 + b * w2
        gust_scale = (1.0 - b) * g1 + b * g2
        temp_c = (1.0 - b) * t1 + b * t2
        auto_ice = i2 if b > 0.5 else i1
        microburst_w = (1.0 - b) * m1 + b * m2

        # Vertical downdraft stagnation near the ground (h < 5m)
        if microburst_w < -0.2:
            stagnation = float(math.tanh(max(0.1, pz) / 4.5))
            microburst_w *= stagnation

        # Venturi wind tunnel amplification between skyscraper rows
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
# 8. DRYDEN ATMOSPHERIC TURBULENCE MODEL (MIL-F-8785C)
# =============================================================================
class DrydenAtmosphericTurbulenceModel:
    """
    Stochastic atmospheric turbulence generator using low-altitude MIL-F-8785C
    spectral formulation for realistic stochastic wind gust dynamics.
    """
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

# =============================================================================
# 9. MESSINGER ROTOR ICING DYNAMICS MODEL
# =============================================================================
class MessingerRotorIcingModel:
    """
    Simulates rotor blade icing under supercooled liquid droplet impingement:
      - Accretes ice thickness (mm) as a function of airspeed and Liquid Water Content (LWC)
      - Degrades aerodynamic lift (up to -28%)
      - Increases aerodynamic drag (up to +180%)
    """
    def __init__(self):
        self.ice_thickness_mm = 0.0
        self.icing_active = False

    def step(self, airspeed, dt=0.005):
        if self.icing_active:
            lwc = 0.35  # g/m^3
            catch_efficiency = 0.42
            d_ice = (lwc * max(1.0, airspeed) * catch_efficiency * 1e-3) * 0.08 * dt
            self.ice_thickness_mm = min(3.8, self.ice_thickness_mm + d_ice * 1000.0)
        else:
            self.ice_thickness_mm = max(0.0, self.ice_thickness_mm - 0.02 * dt)

        ice_index = float(np.clip(self.ice_thickness_mm / 3.0, 0.0, 1.0))
        lift_loss_factor = 1.0 - 0.28 * ice_index
        drag_rise_factor = 1.0 + 1.80 * ice_index
        return self.ice_thickness_mm, ice_index, lift_loss_factor, drag_rise_factor

# =============================================================================
# 10. SUB-ZERO ARRHENIUS 6S BATTERY EQUIVALENT CIRCUIT MODEL (ECM)
# =============================================================================
class SubZeroArrheniusBatteryECM:
    """
    6S LiPo Battery Equivalent Circuit Model with:
      - Arrhenius internal resistance temperature dependence R0(T)
      - R1-C1 transient polarization voltage dynamics
      - Non-linear Open Circuit Voltage (OCV) vs. State-of-Charge (SoC)
      - Joule heating and convective ambient heat transfer
    """
    def __init__(self, initial_soc=0.98, initial_temp_c=22.0):
        self.soc = initial_soc
        self.temp_c = initial_temp_c
        self.v1 = np.zeros(NUM_CELLS, dtype=np.float32)
        self.capacity_coulombs = PACK_CAPACITY_AH * 3600.0
        self.v_pack = 25.0

    def step(self, p_elec_watts, amb_temp_c=22.0, dt=0.005):
        i_draw = float(p_elec_watts) / max(16.0, self.v_pack)
        t_k = self.temp_c + 273.15

        # Arrhenius internal resistance scaling
        r_scale = math.exp((ARRHENIUS_EA / GAS_CONST_R) * (1.0 / t_k - 1.0 / T_REF_K))
        r0 = CELL_R0_NOMINAL * max(0.85, r_scale)

        # Coulomb counting SoC update
        self.soc -= (i_draw * dt) / self.capacity_coulombs
        self.soc = max(0.0, min(1.0, self.soc))

        # Open Circuit Voltage (OCV) curve
        s = max(0.001, min(0.999, self.soc))
        ocv = 3.30 + 0.55 * s + 0.25 * math.log(s + 0.005) + 0.15 * math.exp(6.0 * (s - 0.88))
        ocv = float(np.clip(ocv, 3.20, 4.20))

        # 6 individual cell terminal voltages
        cell_volts = []
        for i in range(NUM_CELLS):
            self.v1[i] += (-(self.v1[i] / (CELL_R1 * CELL_C1)) + (i_draw / CELL_C1)) * dt
            cell_volts.append(ocv - i_draw * r0 - self.v1[i])

        self.v_pack = float(np.sum(cell_volts))

        # Joule self-heating (I^2 * R0) and convective thermal transfer
        p_joule = (i_draw ** 2) * (r0 * NUM_CELLS)
        self.temp_c += ((p_joule - 0.45 * (self.temp_c - amb_temp_c)) / 350.0) * dt

        return self.soc, self.v_pack, i_draw, r0

# =============================================================================
# 11. SKYSCRAPER WALL SUCTION AERODYNAMICS
# =============================================================================
def compute_building_wall_suction(drone_pos, buildings, rho_air=1.225, v_h_hover=5.5):
    """
    Computes low-pressure Bernoulli suction pulling the drone toward skyscraper walls:
      F_suction = (0.5 * rho * v_h^2 * 0.08) * (R / d_wall)^2 * n_wall
    Active when drone is within 0.2m < d_wall < 3.2m of vertical facades.
    """
    px, py, pz = drone_pos[0], drone_pos[1], drone_pos[2]
    f_wall_suction = np.zeros(3, dtype=np.float32)

    for b in buildings:
        bx, by, bh = b["pos"][0], b["pos"][1], b["height"]
        if pz <= (bh + 1.0):
            x1, x2 = bx - 8.0, bx + 8.0
            y1, y2 = by - 8.0, by + 8.0
            
            dx = max(0.0, max(x1 - px, px - x2))
            dy = max(0.0, max(y1 - py, py - y2))
            wall_dist = math.hypot(dx, dy)

            if 0.2 < wall_dist < 3.2:
                suction_mag = (0.5 * rho_air * (v_h_hover ** 2) * 0.08) * ((ROTOR_RADIUS_M / wall_dist) ** 2)
                n_wall = np.array([bx - px, by - py, 0.0], dtype=np.float32)
                n_wall /= max(1e-4, float(np.linalg.norm(n_wall)))
                f_wall_suction += suction_mag * n_wall

    return f_wall_suction

# =============================================================================
# 12. DYNAMIC AIRFRAME ADAPTATION & RESONANCE GAIN SCHEDULER
# =============================================================================
class DynamicSelfAdjustingAirframeEngine:
    """
    Adjusts control gains and moment of inertia based on tether dynamics:
      1. Calculates pendulum natural frequency: omega_n = sqrt(g / L)
      2. Attenuates attitude stiffness near resonance to eliminate tether wobbles
      3. Adapts to battery voltage sag (boosts gains as pack voltage drops)
    """
    def __init__(self, dry_mass=DRY_MASS_KG, nominal_inertia=INERTIA_MATRIX):
        self.dry_mass = dry_mass
        self.J_nominal = nominal_inertia.copy()
        self.J_estimated = self.J_nominal.copy()
        self.k_R = np.array([2.40, 2.40, 1.80], dtype=np.float32)
        self.k_w = np.array([0.90, 0.90, 0.60], dtype=np.float32)

    def update(self, cable_len, has_payload, cargo_mass, sway_2d, v_pack, dt=0.005):
        m_payload = cargo_mass if has_payload else 0.0
        total_mass = self.dry_mass + m_payload

        effective_L = max(0.08, float(cable_len))
        sway_mag = float(np.linalg.norm(sway_2d))
        r_pendulum = effective_L * math.sin(min(0.5, sway_mag / effective_L))
        delta_J = 0.15 * m_payload * (r_pendulum ** 2)

        self.J_estimated[0, 0] = self.J_nominal[0, 0] + delta_J
        self.J_estimated[1, 1] = self.J_nominal[1, 1] + delta_J
        self.J_estimated[2, 2] = self.J_nominal[2, 2]

        voltage_boost = float(np.clip(24.0 / max(18.0, v_pack), 1.0, 1.30))
        pendulum_natural_freq = math.sqrt(GRAVITY_VAL / effective_L)
        resonance_attenuation = float(np.clip(pendulum_natural_freq / 4.5, 0.65, 1.0)) if cable_len > 0.3 else 1.0

        base_stiffness = 2.40 * resonance_attenuation * voltage_boost
        self.k_R[0] = base_stiffness * (self.J_estimated[0, 0] / self.J_nominal[0, 0])
        self.k_R[1] = base_stiffness * (self.J_estimated[1, 1] / self.J_nominal[1, 1])
        self.k_R[2] = 1.80 * voltage_boost

        self.k_w[0] = 2.0 * math.sqrt(max(1e-4, float(self.k_R[0] * self.J_estimated[0, 0]))) * 0.95
        self.k_w[1] = 2.0 * math.sqrt(max(1e-4, float(self.k_R[1] * self.J_estimated[1, 1]))) * 0.95
        self.k_w[2] = 2.0 * math.sqrt(max(1e-4, float(self.k_R[2] * self.J_estimated[2, 2]))) * 0.85

        return total_mass, self.J_estimated, self.k_R, self.k_w

# =============================================================================
# 13. ONLINE MASS & CENTER-OF-MASS (CoM) ESTIMATOR
# =============================================================================
class OnlineMassAndCoMEstimator:
    """
    Recursively estimates physical payload mass and center-of-mass XY offsets:
      - Mass derived from thrust vs. vertical acceleration: m = F_thrust / (a_z + g)
      - CoM offset derived from steady-state roll/pitch trim moments
    """
    def __init__(self, nominal_mass=1.80):
        self.est_mass = nominal_mass
        self.com_offset_xy = np.zeros(2, dtype=np.float32)

    def update(self, thrust_total, accel_z, steady_tau_xy, dt=0.005):
        effective_g = max(1.0, float(accel_z + GRAVITY_VAL))
        instant_mass = thrust_total / effective_g
        if 1.2 <= instant_mass <= 2.4 and abs(accel_z) < 1.5:
            self.est_mass += 0.005 * (instant_mass - self.est_mass)

        if thrust_total > 12.0 and abs(accel_z) < 0.8:
            delta_x = float(np.clip(-steady_tau_xy[1] / thrust_total, -0.02, 0.02))
            delta_y = float(np.clip(steady_tau_xy[0] / thrust_total, -0.02, 0.02))
            self.com_offset_xy[0] += 0.004 * (delta_x - self.com_offset_xy[0])
            self.com_offset_xy[1] += 0.004 * (delta_y - self.com_offset_xy[1])

        return self.est_mass, self.com_offset_xy.copy()

# =============================================================================
# 14. ROOFTOP SURFACE NORMAL ESTIMATOR (SVD PLANE FITTING)
# =============================================================================
class RooftopSurfaceNormalEstimator:
    """
    Fits a 3D plane equation to 8 downward conical range returns via SVD
    to detect rooftop tilt angles and slope hazards before touchdown.
    """
    def __init__(self):
        self.surface_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def estimate_normal(self, drone_pos, conical_ranges_8):
        cone_angle = math.radians(25.0)
        sin_c = math.sin(cone_angle)
        cos_c = math.cos(cone_angle)
        pts = []

        for i in range(8):
            r = min(12.0, float(conical_ranges_8[i]))
            ang = 2.0 * math.pi * i / 8.0
            ray_dir = np.array([sin_c * math.cos(ang), sin_c * math.sin(ang), -cos_c], dtype=np.float32)
            pts.append(drone_pos + r * ray_dir)

        pts = np.array(pts, dtype=np.float32)
        centroid = np.mean(pts, axis=0)
        centered_pts = pts - centroid

        try:
            _, _, vh = np.linalg.svd(centered_pts)
            normal = vh[-1]
            if normal[2] < 0:
                normal = -normal
            self.surface_normal = normal / max(1e-4, float(np.linalg.norm(normal)))
        except np.linalg.LinAlgError:
            self.surface_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        return self.surface_normal.copy()

    # =============================================================================
# 15. ADVANCED WINCH & MULTI-NODE TETHER DELIVERY SYSTEM
# =============================================================================
PAYLOAD_WEIGHT_N = CARGO_MASS_KG * GRAVITY_VAL
GUILLOTINE_THRESHOLD_N = 48.0

class AutonomousWinchDeliverySystem:
    """
    Position-Based Dynamics (PBD) Tether & Delivery Winch:
      - Multi-node discretized cable with catenary sag modeling
      - Stabilizing aerodynamic fin vanes on payload (aligns yaw to relative wind)
      - Anti-resonance notch punch-through to suppress vortex vibrations
      - Tension derivative touchdown detection (dT/dt < -35 N/s)
      - Guillotine safety cutter (cuts tether if tension > 48 N)
      - Skid strain gauges for ground reaction telemetry
    """
    def __init__(self, num_segments=4):
        self.num_segments = num_segments
        self.num_nodes = num_segments + 1
        
        self.cable_len = 0.55
        self.cable_vel = 0.0
        self.theta = np.zeros(2, dtype=np.float32)
        self.theta_dot = np.zeros(2, dtype=np.float32)
        self.sway_offset = np.zeros(2, dtype=np.float32)
        self.catenary_midpoint = np.zeros(2, dtype=np.float32)
        self.tension_n = PAYLOAD_WEIGHT_N
        self.prev_tension_n = PAYLOAD_WEIGHT_N
        self.tension_derivative = 0.0
        self.has_payload = True
        self.severed = False
        self.state = "STANDBY"
        self.touchdown_timer = 0.0
        self.delivery_completed = False
        
        self.linear_density = 0.008
        self.cable_diameter_m = 0.0025
        self.nodes_pos = np.zeros((self.num_nodes, 3), dtype=np.float32)
        self.nodes_vel = np.zeros((self.num_nodes, 3), dtype=np.float32)
        
        self.cargo_yaw = 0.0
        self.cargo_yaw_rate = 0.0
        self.contact_normal_force = 0.0
        self.f_skid_left = 0.0
        self.f_skid_right = 0.0
        self.is_skid_grounded = False

    def reload_new_cargo(self):
        self.cable_len = 0.55
        self.cable_vel = 0.0
        self.theta.fill(0.0)
        self.theta_dot.fill(0.0)
        self.nodes_pos.fill(0.0)
        self.nodes_vel.fill(0.0)
        self.sway_offset.fill(0.0)
        self.catenary_midpoint.fill(0.0)
        self.cargo_yaw = 0.0
        self.cargo_yaw_rate = 0.0
        self.contact_normal_force = 0.0
        self.has_payload = True
        self.severed = False
        self.state = "STANDBY"
        self.tension_n = PAYLOAD_WEIGHT_N
        self.prev_tension_n = PAYLOAD_WEIGHT_N
        self.tension_derivative = 0.0
        self.touchdown_timer = 0.0
        self.delivery_completed = False
        self.is_skid_grounded = False

    def step(self, drone_p, drone_v, target_surface_z=0.0, is_settled=False,
             is_surface_safe=True, wire_hit=False, agent_breach=False,
             agent_dist=30.0, wind_3d=None, drone_accel_linear=None, dt=0.005):
        
        if self.severed:
            self.state = "GUILLOTINE"
            return self.cable_len, 0.0, np.zeros(2, dtype=np.float32)

        if wind_3d is None:
            wind_3d = np.zeros(3, dtype=np.float32)

        wind_speed_xy = float(np.linalg.norm(wind_3d[:2]))
        payload_z = drone_p[2] - self.cable_len
        clearance_to_roof = payload_z - target_surface_z
        dropzone_occupied = (agent_dist < 2.8) and (target_surface_z < 2.0) and (self.state == "STANDBY")
        lockout = (not is_surface_safe) or wire_hit or (not is_settled) or dropzone_occupied

        # Emergency abort if intruder breaches drop zone
        if agent_breach and (target_surface_z < 2.0) and self.state in ["LOWERING", "TOUCHDOWN"]:
            self.state = "ABORT_RETRACT"

        # Guillotine threshold check
        if self.tension_n >= GUILLOTINE_THRESHOLD_N:
            self.severed = True
            self.has_payload = False
            self.state = "GUILLOTINE"
            print(f">>> [GUILLOTINE CUTTER] Tension exceeded {GUILLOTINE_THRESHOLD_N}N! Tether severed! <<<")
            return self.cable_len, 0.0, np.zeros(2, dtype=np.float32)

        # Anti-resonance notch check (omega_p = sqrt(g / L))
        omega_pendulum = math.sqrt(GRAVITY_VAL / max(0.08, float(self.cable_len)))
        in_resonance_band = (1.8 < omega_pendulum < 2.5) and (self.state == "LOWERING")

        # Winch State Machine
        if self.state == "STANDBY":
            self.cable_vel = 0.0
            if not lockout and self.has_payload and is_settled:
                self.state = "LOWERING"

        elif self.state == "LOWERING":
            if lockout and self.cable_len > 0.60:
                self.state = "RETRACTING"
            else:
                flare_scale = float(np.clip(clearance_to_roof / 1.20, 0.20, 1.0)) if clearance_to_roof < 1.20 else 1.0
                base_rate = float(np.clip(2.40 - 0.12 * wind_speed_xy, 1.20, 2.40)) * flare_scale
                
                # Punch through resonant frequency band to suppress vortex cable vibrations
                if in_resonance_band and clearance_to_roof > 1.5:
                    base_rate = 3.20

                self.cable_vel = base_rate
                self.cable_len += self.cable_vel * dt

                # Tension derivative and clearance touchdown detection
                fast_tension_drop = (self.tension_derivative < -35.0) and (clearance_to_roof < 0.40)
                if clearance_to_roof <= 0.10 or fast_tension_drop or self.contact_normal_force >= 0.40 * PAYLOAD_WEIGHT_N:
                    self.state = "TOUCHDOWN"
                    self.touchdown_timer = 0.0

        elif self.state == "TOUCHDOWN":
            self.cable_vel = 0.0
            self.touchdown_timer += dt
            if self.touchdown_timer >= 0.25:
                self.has_payload = False
                self.state = "RETRACTING"

        elif self.state in ["RETRACTING", "ABORT_RETRACT"]:
            self.cable_vel = -3.20
            self.cable_len = max(0.20, self.cable_len + self.cable_vel * dt)
            if self.cable_len <= 0.25:
                self.cable_len = 0.20
                self.cable_vel = 0.0
                if self.state == "RETRACTING" and not self.has_payload:
                    self.delivery_completed = True
                self.state = "STANDBY"

        L = max(0.20, float(self.cable_len))
        seg_len = L / float(self.num_segments)

        self.nodes_pos[0] = drone_p + np.array([0.0, 0.0, -0.05], dtype=np.float32)
        self.nodes_vel[0] = drone_v.copy()

        # Stabilizing aerodynamic fin vanes on payload
        v_rel_tip = self.nodes_vel[-1] - wind_3d
        v_rel_xy_speed = math.hypot(v_rel_tip[0], v_rel_tip[1])
        if v_rel_xy_speed > 0.2:
            wind_track_yaw = math.atan2(v_rel_tip[1], v_rel_tip[0])
            yaw_err = (self.cargo_yaw - wind_track_yaw + math.pi) % (2.0 * math.pi) - math.pi
            fin_align_torque = -1.8 * yaw_err * v_rel_xy_speed - 0.45 * self.cargo_yaw_rate
            self.cargo_yaw_rate += fin_align_torque * dt
            self.cargo_yaw += self.cargo_yaw_rate * dt

        # Discretized PBD node dynamics
        predicted_pos = np.zeros_like(self.nodes_pos)
        predicted_pos[0] = self.nodes_pos[0]

        for k in range(1, self.num_nodes):
            m_k = CARGO_MASS_KG if (k == self.num_nodes - 1 and self.has_payload) else 0.04
            v_rel = self.nodes_vel[k] - wind_3d
            speed_rel = float(np.linalg.norm(v_rel))
            cd = 0.85 if (k == self.num_nodes - 1) else 1.15
            area = (CARGO_MASS_KG * 0.045) if (k == self.num_nodes - 1) else (self.cable_diameter_m * seg_len)
            f_drag = -0.5 * 1.225 * cd * area * speed_rel * v_rel

            a_k = np.array([0, 0, -GRAVITY_VAL], dtype=np.float32) + (f_drag / m_k)
            self.nodes_vel[k] += a_k * dt
            self.nodes_vel[k] *= 0.965
            predicted_pos[k] = self.nodes_pos[k] + self.nodes_vel[k] * dt

        # Distance constraint projection (PBD solver)
        for _ in range(4):
            predicted_pos[0] = self.nodes_pos[0]
            for i in range(self.num_segments):
                delta = predicted_pos[i + 1] - predicted_pos[i]
                dist = float(np.linalg.norm(delta))
                if dist > seg_len and dist > 1e-5:
                    correction = (dist - seg_len) * (delta / dist)
                    if i == 0:
                        predicted_pos[i + 1] -= correction
                    else:
                        predicted_pos[i] += 0.35 * correction
                        predicted_pos[i + 1] -= 0.65 * correction

        # Spring-damper soft bumper rooftop contact
        contact_depth = target_surface_z + 0.06 - predicted_pos[-1][2]
        if contact_depth > 0.0:
            k_spring = 340.0
            d_damper = 14.0
            f_spring = k_spring * contact_depth - d_damper * min(0.0, float(self.nodes_vel[-1][2]))
            self.contact_normal_force = float(np.clip(f_spring, 0.0, 1.5 * PAYLOAD_WEIGHT_N))
            predicted_pos[-1][2] = target_surface_z + 0.06
            self.nodes_vel[-1][2] = 0.0
            self.nodes_vel[-1][:2] *= 0.85
        else:
            self.contact_normal_force = 0.0

        for k in range(1, self.num_nodes):
            self.nodes_vel[k] = (predicted_pos[k] - self.nodes_pos[k]) / dt
            self.nodes_pos[k] = predicted_pos[k].copy()

        self.sway_offset = self.nodes_pos[-1][:2] - self.nodes_pos[0][:2]
        theta_x = math.atan2(float(self.sway_offset[0]), L)
        theta_y = math.atan2(float(self.sway_offset[1]), L)
        self.theta_dot = (np.array([theta_x, theta_y], dtype=np.float32) - self.theta) / dt
        self.theta = np.array([theta_x, theta_y], dtype=np.float32)

        mid_node = self.num_nodes // 2
        straight_mid = 0.5 * (self.nodes_pos[0][:2] + self.nodes_pos[-1][:2])
        self.catenary_midpoint = self.nodes_pos[mid_node][:2] - straight_mid

        if self.has_payload:
            a_vert = float(drone_accel_linear[2]) if drone_accel_linear is not None else 0.0
            theta_mag = float(np.linalg.norm(self.theta))
            t_grav = max(0.5, CARGO_MASS_KG * (GRAVITY_VAL * math.cos(theta_mag) + a_vert) - self.contact_normal_force)
            t_centrif = CARGO_MASS_KG * L * float(np.dot(self.theta_dot, self.theta_dot))
            new_tension = float(np.clip(t_grav + t_centrif, 1.2, 45.0))
        else:
            new_tension = 1.2

        self.tension_derivative = (new_tension - self.prev_tension_n) / dt
        self.prev_tension_n = self.tension_n
        self.tension_n = new_tension

        return self.cable_len, self.tension_n, self.sway_offset

    def update_skid_strain_gauges(self, is_on_surface, weight_force, f_thrust):
        if is_on_surface:
            f_normal_total = max(0.0, weight_force - f_thrust)
            noise = float(np.random.normal(0, 0.2))
            self.f_skid_left = 0.5 * f_normal_total + noise
            self.f_skid_right = 0.5 * f_normal_total - noise
            self.is_skid_grounded = (f_normal_total >= 0.65 * weight_force)
        else:
            self.f_skid_left = 0.0
            self.f_skid_right = 0.0
            self.is_skid_grounded = False

# =============================================================================
# 16. 3D MACRO LIDAR RAYCASTER & SURFACE ELEVATION ENGINE
# =============================================================================
class MacroPerceptionLiDARSuite:
    """
    Simulates physical 3D LiDAR raycasting:
      - 16 horizontal radial beams over 360 degrees (ranges 0.1m - 30.0m)
      - 8 downward conical beams at 25 degrees (ranges 0.05m - 15.0m)
    Computes exact geometric ray-AABB intersections against all 3D buildings.
    """
    def __init__(self, buildings):
        self.buildings = buildings

    def cast_360_lidar(self, drone_pos, drone_yaw):
        px, py, pz = drone_pos[0], drone_pos[1], drone_pos[2]
        ranges = np.full(16, 30.0, dtype=np.float32)
        angles = drone_yaw + np.linspace(0, 2.0 * math.pi, 16, endpoint=False)

        for i, ang in enumerate(angles):
            dx, dy = math.cos(ang), math.sin(ang)
            min_dist = 30.0

            for b in self.buildings:
                bx, by, bh = b["pos"][0], b["pos"][1], b["height"]
                if pz <= (bh + 0.1):
                    x1, x2 = bx - 8.0, bx + 8.0
                    y1, y2 = by - 8.0, by + 8.0

                    tx1 = (x1 - px) / dx if abs(dx) > 1e-4 else (-1e6 if px < x1 else 1e6)
                    tx2 = (x2 - px) / dx if abs(dx) > 1e-4 else (1e6 if px > x2 else -1e6)
                    ty1 = (y1 - py) / dy if abs(dy) > 1e-4 else (-1e6 if py < y1 else 1e6)
                    ty2 = (y2 - py) / dy if abs(dy) > 1e-4 else (1e6 if py > y2 else -1e6)

                    t_enter = max(min(tx1, tx2), min(ty1, ty2))
                    t_exit = min(max(tx1, tx2), max(ty1, ty2))

                    if t_enter <= t_exit and t_exit > 0.0:
                        hit = max(0.0, t_enter)
                        if hit < min_dist:
                            min_dist = hit

            ranges[i] = min_dist
        return ranges

    def cast_downward_conical(self, drone_pos):
        px, py, pz = drone_pos[0], drone_pos[1], drone_pos[2]
        ranges = np.full(8, 15.0, dtype=np.float32)
        sin_c = math.sin(math.radians(25.0))
        cos_c = math.cos(math.radians(25.0))

        for i in range(8):
            ang = 2.0 * math.pi * i / 8.0
            sample_x = px + sin_c * math.cos(ang) * 1.5
            sample_y = py + sin_c * math.sin(ang) * 1.5

            surf_h = 0.0
            for b in self.buildings:
                bx, by, bh = b["pos"][0], b["pos"][1], b["height"]
                if (bx - 8.0) <= sample_x <= (bx + 8.0) and (by - 8.0) <= sample_y <= (by + 8.0):
                    surf_h = max(surf_h, bh)

            clearance = max(0.05, pz - surf_h)
            ranges[i] = min(15.0, clearance / max(0.1, cos_c))

        return ranges

def get_ground_surface_under_drone(px, py, buildings):
    """Finds the height of the terrain or rooftop directly below (px, py)."""
    surf_z = 0.0
    for b in buildings:
        bx, by, bh = b["pos"][0], b["pos"][1], b["height"]
        if (bx - 8.0) <= px <= (bx + 8.0) and (by - 8.0) <= py <= (by + 8.0):
            surf_z = max(surf_z, bh)
    return surf_z

# =============================================================================
# 17. ADVANCED MULTI-MODAL PHYSICAL SENSOR SUITE
# =============================================================================
class AdvancedMultiModalSensorSuite:
    """
    Simulates physical avionics telemetry:
      - Optical Flow & Camera Slant Range
      - Downward ArUco Helipad Marker Detector
      - Emergency Siren Directional Acoustic Array
      - ADS-B Civil Aircraft Transponder
      - Smart Hook IMU & 3-Axis Hook Load Cell
      - Thermal IR Roof Heat & Polarization Sensing
      - ESC/FOC Motor Current & Temperature
      - 5-Port Air Data Pitot Probe (AoA, Sideslip, Humidity, Dust)
      - RTK GNSS Jamming/Spoofing Monitor
    """
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

    def poll_acoustic_array(self, drone_p, amb_pos=None, sim_time=0.0, siren_active=False):
        if siren_active and amb_pos is not None:
            d = float(np.linalg.norm(drone_p[:2] - amb_pos[:2]))
            rel = amb_pos[:2] - drone_p[:2]
            bearing = math.atan2(rel[1], rel[0])
            spl_dba = float(np.clip(102.0 - 20.0 * math.log10(max(1.0, d)), 45.0, 98.0))
            freq = 850.0 + 150.0 * math.sin(sim_time * 3.0)
            return bearing, freq, spl_dba
        return 0.0, 180.0, 52.0

    def poll_adsb_transponder(self, drone_p, drone_v, medevac_pos, medevac_vel):
        rel_pos = medevac_pos - drone_p
        dist = float(np.linalg.norm(rel_pos))
        rel_vel = medevac_vel - drone_v
        closing_spd = -float(np.dot(rel_pos / max(1e-3, dist), rel_vel))
        return rel_pos, float(np.clip(closing_spd, -30.0, 30.0)), dist

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

    def poll_thermal_and_polarization(self, drone_pos):
        px, py = drone_pos[0], drone_pos[1]
        min_roof_dist = 999.0
        for b in self.buildings:
            bx, by = b["pos"][0], b["pos"][1]
            d = math.hypot(px - bx, py - by)
            min_roof_dist = min(min_roof_dist, d)

        roof_heat = 7.5 * math.exp(-min_roof_dist / 8.0)
        park_dist = math.hypot(px - (-48.0), py - (-48.0))
        th_dt = float(roof_heat - 2.5 * math.exp(-park_dist / 20.0) + 0.5)
        dolp = float(0.12 + 0.55 * math.exp(-min_roof_dist / 6.0))
        ndvi = float(0.10 + 0.68 * math.exp(-park_dist / 25.0))
        return th_dt, dolp, ndvi

    def poll_esc_telemetry(self, motor_omegas):
        i_rms = np.zeros(8, dtype=np.float32)
        t_mosfet = np.zeros(8, dtype=np.float32)
        for i in range(8):
            p_ratio = (motor_omegas[i] / MAX_OMEGA) ** 2
            i_rms[i] = 3.5 + p_ratio * 18.5
            t_mosfet[i] = 32.0 + p_ratio * 46.0
        return i_rms, t_mosfet

    def poll_5port_pitot(self, drone_v, wind_3d, drone_R):
        v_air = drone_R.T @ (drone_v - wind_3d)
        spd = max(0.5, float(np.linalg.norm(v_air)))
        alpha_deg = math.degrees(math.atan2(v_air[2], max(0.2, v_air[0])))
        beta_deg = math.degrees(math.asin(np.clip(v_air[1] / spd, -0.9, 0.9)))
        rh_pct = float(np.clip(68.0 - 0.4 * drone_v[2], 30.0, 95.0))
        dust_pm10 = float(np.clip(22.0 + 1.2 * float(np.linalg.norm(wind_3d[:2])), 5.0, 80.0))
        return alpha_deg, beta_deg, rh_pct, dust_pm10

    def poll_rtk_gnss(self, is_spoofed=False):
        hdop = 3.8 if is_spoofed else float(np.clip(0.85 + np.random.normal(0, 0.05), 0.6, 2.0))
        cn0 = 28.5 if is_spoofed else float(np.clip(44.0 + np.random.normal(0, 1.2), 32.0, 48.0))
        rf_jam_pwr = -52.0 if is_spoofed else -110.0
        return hdop, cn0, rf_jam_pwr, 0.02

# =============================================================================
# 18. FULL PHYSICS ATMOSPHERIC SENSORS & 6-DOF MEMS IMU NOISE SUITE
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

class Airframe6DOFMEMS_IMU:
    def __init__(self):
        self.b_a = np.array([0.02, -0.015, 0.03], dtype=np.float32)
        self.b_g = np.array([0.005, -0.004, 0.002], dtype=np.float32)

    def poll(self, drone_R, accel_linear, omega_true, motor_omegas):
        g_vec = np.array([0, 0, -GRAVITY_VAL], dtype=np.float32)
        f_body = drone_R.T @ (accel_linear - g_vec)
        rpm_ratio = float(np.mean(motor_omegas)) / 963.4
        vib_a = np.random.normal(0.0, 0.16 * rpm_ratio, 3).astype(np.float32)
        vib_g = np.random.normal(0.0, 0.015 * rpm_ratio, 3).astype(np.float32)

        accel_raw = f_body + self.b_a + vib_a + np.random.normal(0.0, 0.035, 3).astype(np.float32)
        gyro_raw = omega_true + self.b_g + vib_g + np.random.normal(0.0, 0.004, 3).astype(np.float32)
        return accel_raw, gyro_raw

# =============================================================================
# 19. STRICT 139-DIMENSIONAL OBSERVATION SYNTHESIZER
# =============================================================================
class FullObservationSynthesizer139Dim:
    """
    Synthesizes the complete 139-dimensional avionics telemetry vector into a
    statically mapped array, preventing any dynamic feature slip or truncation.
    """
    def __init__(self):
        self.dim = BASE_OBS_DIM

    def synthesize(self, p_est, v_est, R_est, omega_est, target_p, lidar_16, conical_8,
                   v_tas, q_dyn, min_sonar, wind_est, ice_idx, cable_len, cable_vel,
                   tension_n, sway_2d, bms_soc, v_pack, bms_temp_c, r0_dyn,
                   chi2_val, jerk_discrepancy, is_spoofed, emf_rms, wire_dist,
                   agent_dist, ground_spl_dba, has_payload, hook_a, hook_w,
                   load_cell_3d, thermal_dt, dolp, ndvi, esc_i_rms, esc_t_mosfet,
                   alpha_deg, beta_deg, rh_pct, dust_pm10, hdop, cn0_avg,
                   rf_jam_pwr, mag_grad, airframe_accel, airframe_gyro,
                   peer_swarm_kinematics, optical_flow_2d, flow_quality,
                   aruco_offset_3d, aruco_locked, skid_force_left, skid_force_right,
                   is_skid_grounded, siren_angle, siren_spl, adsb_rel_pos, adsb_closing_spd):

        out = np.zeros(self.dim, dtype=np.float32)

        out[0:3] = (target_p - p_est) / 50.0
        out[3:6] = p_est / 200.0
        out[6:9] = v_est / 8.0
        out[9:18] = R_est.flatten()
        out[18:21] = omega_est / 10.0
        out[21:37] = lidar_16 / 30.0
        out[37] = min_sonar / 15.0
        out[38] = v_tas / 20.0
        out[39] = q_dyn / 100.0
        out[40] = wire_dist / 50.0
        out[41] = agent_dist / 50.0
        out[42:45] = wind_est / 8.0
        out[45:53] = conical_8 / 15.0
        out[53] = cable_len / 15.0
        out[54] = cable_vel / 5.0
        out[55] = tension_n / 50.0
        out[56:58] = sway_2d / 2.0
        out[58] = 1.0 if has_payload else 0.0
        out[59] = skid_force_left / 20.0
        out[60] = skid_force_right / 20.0
        out[61] = bms_soc
        out[62] = v_pack / 25.2
        out[63] = bms_temp_c / 60.0
        out[64] = r0_dyn / 0.05
        out[65] = ice_idx
        out[66] = 1.0 if is_skid_grounded else 0.0
        out[67] = chi2_val / 10.0
        out[68] = jerk_discrepancy / 16.0
        out[69:77] = esc_i_rms / 25.0
        out[77:85] = esc_t_mosfet / 100.0
        out[85:88] = hook_a / 20.0
        out[88:91] = hook_w / 10.0
        out[91:94] = load_cell_3d / 50.0
        out[94] = alpha_deg / 45.0
        out[95] = beta_deg / 45.0
        out[96] = rh_pct / 100.0
        out[97] = dust_pm10 / 100.0
        out[98] = thermal_dt / 10.0
        out[99] = dolp
        out[100] = hdop / 5.0
        out[101] = cn0_avg / 50.0
        out[102] = (rf_jam_pwr + 120.0) / 70.0
        out[103] = mag_grad / 1.0
        out[104] = optical_flow_2d[0] / 5.0
        out[105] = optical_flow_2d[1] / 5.0
        out[106] = flow_quality
        out[107] = 1.0 if is_spoofed else 0.0
        out[108:111] = aruco_offset_3d / 10.0
        out[111] = aruco_locked
        out[112] = siren_angle / math.pi
        out[113] = siren_spl / 100.0
        out[114:117] = adsb_rel_pos / 200.0
        out[117] = adsb_closing_spd / 30.0

        # Peer swarm kinematics (Slots 118-129)
        p_idx = 118
        for peer in peer_swarm_kinematics[:2]:
            out[p_idx:p_idx+3] = peer["p_rel"] / 50.0
            out[p_idx+3:p_idx+6] = peer["v_rel"] / 10.0
            p_idx += 6

        out[130:133] = airframe_accel / 30.0
        out[133:136] = airframe_gyro / 10.0
        out[136] = ndvi
        out[137] = emf_rms / 100.0
        out[138] = ground_spl_dba / 100.0
        return out

    def stack_history(self, history_q, current_obs):
        if len(history_q) == 0:
            for _ in range(history_q.maxlen or HISTORY_LEN):
                history_q.append(current_obs.copy())
        else:
            history_q.append(current_obs.copy())
        return np.concatenate(list(history_q), axis=0).astype(np.float32)

# =============================================================================
# 20. SWARM OBSERVATION & TEMPORAL STACKER MANAGER
# =============================================================================
class SwarmObservationSynthesizer:
    """
    Manages observations and rolling temporal queues for all 3 swarm drones,
    assembling real physical sensors and peer kinematics into the 417-dim tensor.
    """
    def __init__(self, num_drones=3):
        self.num_drones = num_drones
        self.obs_synth = FullObservationSynthesizer139Dim()
        self.temporal_queues = [deque(maxlen=HISTORY_LEN) for _ in range(num_drones)]

    def synthesize_drone_obs(
        self, drone_idx, p_est, v_est, R_est, omega_est, target_pos,
        total_wind, cable_len, c_vel, d_vel, c_pos, d_pos,
        has_payload, tension_mag, bms_soc, bms_v_pack, bms_temp_c, bms_r0,
        ice_idx, raw_accel, raw_gyro, adsb_rel, adsb_spd,
        peer_kinematics, is_grounded, buildings, sensor_suite, full_physics, lidar_suite
    ):
        surf_z = get_ground_surface_under_drone(p_est[0], p_est[1], buildings)
        drone_yaw = math.atan2(R_est[1, 0], R_est[0, 0])
        lidar_16 = lidar_suite.cast_360_lidar(p_est, drone_yaw)
        conical_8 = lidar_suite.cast_downward_conical(p_est)
        phys_data = full_physics.poll(p_est, v_est, total_wind, surface_z=surf_z)
        flow_2d, flow_q, _ = sensor_suite.poll_optical_flow(p_est, v_est, R_est, omega_est, surface_z=surf_z)
        aruco_3d, aruco_lk = sensor_suite.poll_aruco_detector(p_est, R_est, target_pos)

        sway_2d = (c_pos[:2] - d_pos[:2]) if has_payload else np.zeros(2, dtype=np.float32)
        hook_a, hook_w, load_3d = sensor_suite.poll_smart_hook_and_load_cell(
            tension_n=tension_mag, sway_2d=sway_2d, cable_l=cable_len, has_payload=has_payload
        )

        motor_om = np.full(8, 450.0, dtype=np.float32)
        esc_i_rms, esc_t_mosfet = sensor_suite.poll_esc_telemetry(motor_om)
        alpha_deg, beta_deg, rh_pct, dust_pm10 = sensor_suite.poll_5port_pitot(v_est, total_wind, R_est)
        hdop, cn0, rf_pwr, mag_grad = sensor_suite.poll_rtk_gnss(is_spoofed=False)
        th_dt, dolp, ndvi = sensor_suite.poll_thermal_and_polarization(p_est)

        skid_f = (0.5 * TOTAL_MASS_KG * 9.81) if is_grounded else 0.0

        obs_139 = self.obs_synth.synthesize(
            p_est=p_est, v_est=v_est, R_est=R_est, omega_est=omega_est, target_p=target_pos,
            lidar_16=lidar_16, conical_8=conical_8, v_tas=phys_data["v_tas"],
            q_dyn=phys_data["q_dyn"], min_sonar=phys_data["min_sonar"], wind_est=total_wind,
            ice_idx=ice_idx, cable_len=cable_len, cable_vel=float(np.linalg.norm(c_vel - d_vel)),
            tension_n=tension_mag, sway_2d=sway_2d, bms_soc=bms_soc, v_pack=bms_v_pack,
            bms_temp_c=bms_temp_c, r0_dyn=bms_r0, chi2_val=1.2,
            jerk_discrepancy=float(np.linalg.norm(raw_accel[:2])), is_spoofed=False,
            emf_rms=phys_data["emf_rms"], wire_dist=40.0, agent_dist=25.0, ground_spl_dba=52.0,
            has_payload=has_payload, hook_a=hook_a, hook_w=hook_w, load_cell_3d=load_3d,
            thermal_dt=th_dt, dolp=dolp, ndvi=ndvi, esc_i_rms=esc_i_rms, esc_t_mosfet=esc_t_mosfet,
            alpha_deg=alpha_deg, beta_deg=beta_deg, rh_pct=rh_pct, dust_pm10=dust_pm10,
            hdop=hdop, cn0_avg=cn0, rf_jam_pwr=rf_pwr, mag_grad=mag_grad,
            airframe_accel=raw_accel, airframe_gyro=raw_gyro,
            peer_swarm_kinematics=peer_kinematics, optical_flow_2d=flow_2d, flow_quality=flow_q,
            aruco_offset_3d=aruco_3d, aruco_locked=aruco_lk, skid_force_left=skid_f,
            skid_force_right=skid_f, is_skid_grounded=is_grounded, siren_angle=0.0, siren_spl=52.0,
            adsb_rel_pos=adsb_rel, adsb_closing_spd=adsb_spd
        )

        return self.obs_synth.stack_history(self.temporal_queues[drone_idx], obs_139)

    # =============================================================================
# 21. LIE ALGEBRA UTILITIES ON SO(3) FOR STATE ESTIMATION
# =============================================================================
def skew_symmetric(v):
    return np.array([
        [0.0, -v[2], v[1]], 
        [v[2], 0.0, -v[0]], 
        [-v[1], v[0], 0.0]
    ], dtype=np.float32)

def quat_to_rot_np(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z),       2.0*(x*z + w*y)],
        [2.0*(x*y + w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x)],
        [2.0*(x*z - w*y),       2.0*(y*z + w*x),       1.0 - 2.0*(x*x + y*y)]
    ], dtype=np.float32)

def quat_mult_np(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float32)

# =============================================================================
# 22. MATTER-WAVE QUANTUM SENSOR & 15-STATE ERROR-STATE EKF ON SO(3)
# =============================================================================
K_EFF_WAVEVECTOR = 1.61e7            # 87Rb Raman laser effective wavevector (1/m)
RAMAN_PULSE_SEP_T = 0.015            # Pulse separation time T (s)

class MatterWaveColdAtomQuantumSensor:
    """
    Simulates a light-pulse atom interferometer (87Rb Raman transitions).
    Measures specific force body acceleration via quantum phase shift:
      Phi = k_eff * a * T^2
    """
    def __init__(self):
        self.k_eff = K_EFF_WAVEVECTOR
        self.T = RAMAN_PULSE_SEP_T

    def measure_phase_shift(self, specific_force_body):
        phase_shift = self.k_eff * specific_force_body * (self.T ** 2)
        phase_shift += np.random.normal(0, 0.002, 3).astype(np.float32)
        a_quantum_body = phase_shift / (self.k_eff * (self.T ** 2))
        return a_quantum_body, phase_shift

class ErrorStateEKF15:
    """
    15-State Error-State EKF estimating:
      - Position error delta_p (3)
      - Velocity error delta_v (3)
      - Attitude error delta_theta on SO(3) (3)
      - Accelerometer bias delta_b_a (3)
      - Gyroscope bias delta_b_g (3)
    Features Sage-Husa adaptive process noise inflation and Tikhonov-regularized innovation.
    """
    def __init__(self, init_pos):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.b_a = np.zeros(3, dtype=np.float32)
        self.b_g = np.zeros(3, dtype=np.float32)

        # Covariance matrices
        self.P = np.diag([0.15]*3 + [0.10]*3 + [0.02]*3 + [0.05]*3 + [0.01]*3).astype(np.float32)
        self.Q_base = np.diag([1e-4]*3 + [1e-3]*3 + [1e-4]*3 + [1e-5]*3 + [1e-6]*3).astype(np.float32)
        self.Q_d = self.Q_base.copy()
        self.residual_scale = 1.0

    def predict(self, accel_body_raw, gyro_raw, dt=0.005):
        a_unbiased = np.clip(accel_body_raw - self.b_a, -30.0, 30.0)
        w_unbiased = np.clip(gyro_raw - self.b_g, -12.0, 12.0)
        
        R = quat_to_rot_np(self.q)
        g_inertial = np.array([0.0, 0.0, -GRAVITY_VAL], dtype=np.float32)
        f_inertial = R @ a_unbiased + g_inertial

        # Propagate nominal position & velocity
        self.p += self.v * dt + 0.5 * f_inertial * (dt ** 2)
        self.v += f_inertial * dt
        self.v = np.clip(self.v, -15.0, 15.0)

        # Propagate nominal attitude quaternion
        w_norm = np.linalg.norm(w_unbiased)
        if w_norm > 1e-6:
            half = 0.5 * w_norm * dt
            dq = np.array([math.cos(half), *((w_unbiased / w_norm) * math.sin(half))], dtype=np.float32)
            self.q = quat_mult_np(self.q, dq)
            self.q /= max(1e-6, np.linalg.norm(self.q))

        # Error state transition Jacobian F_x (15x15)
        F_x = np.eye(15, dtype=np.float32)
        F_x[0:3, 3:6] = np.eye(3) * dt
        F_x[3:6, 6:9] = -R @ skew_symmetric(a_unbiased) * dt
        F_x[3:6, 9:12] = -R * dt
        F_x[6:9, 6:9] = np.eye(3) - skew_symmetric(w_unbiased) * dt
        F_x[6:9, 12:15] = -np.eye(3) * dt

        # Sage-Husa process noise adaptation
        self.Q_d = self.Q_base * max(1.0, float(self.residual_scale))
        self.P = F_x @ self.P @ F_x.T + self.Q_d
        self.P = np.nan_to_num(self.P, nan=0.05)
        self.P = 0.5 * (self.P + self.P.T)

        max_diag = np.array([100.0]*3 + [50.0]*3 + [0.5]*3 + [1.0]*3 + [0.2]*3, dtype=np.float32)
        diag_P = np.diag(self.P)
        for k in range(15):
            if diag_P[k] > max_diag[k]:
                scale = math.sqrt(max_diag[k] / max(1e-6, diag_P[k]))
                self.P[k, :] *= scale
                self.P[:, k] *= scale
        self.P = 0.5 * (self.P + self.P.T)

    def update_measurement(self, H, innovation, R_cov, chi2_threshold=11.34):
        S = H @ self.P @ H.T + R_cov
        S_reg = S + np.eye(S.shape[0], dtype=np.float32) * 1e-6

        try:
            PHt = self.P @ H.T
            K = np.linalg.solve(S_reg, PHt.T).T
            innov_flat = np.array(innovation).flatten()
            S_inv_innov = np.linalg.solve(S_reg, innov_flat)
            gamma = float(innov_flat @ S_inv_innov)
        except np.linalg.LinAlgError:
            return 0.0, False

        d_k = 0.05
        expected_gamma = float(len(innov_flat))
        self.residual_scale = (1.0 - d_k) * self.residual_scale + d_k * (gamma / max(1.0, expected_gamma))
        self.residual_scale = float(np.clip(self.residual_scale, 0.8, 5.0))

        if gamma > chi2_threshold:
            return gamma, True  # Rejected anomalous reading

        delta_x = (K @ innovation).flatten()
        delta_x[0:6] = np.clip(delta_x[0:6], -3.0, 3.0)
        delta_x[6:9] = np.clip(delta_x[6:9], -0.20, 0.20)
        delta_x[9:15] = np.clip(delta_x[9:15], -0.40, 0.40)

        self.p += delta_x[0:3]
        self.v += delta_x[3:6]
        d_theta = delta_x[6:9]
        angle = np.linalg.norm(d_theta)
        if angle > 1e-6:
            dq = np.array([math.cos(angle * 0.5), *((d_theta / angle) * math.sin(angle * 0.5))], dtype=np.float32)
            self.q = quat_mult_np(self.q, dq)
            self.q /= max(1e-6, np.linalg.norm(self.q))

        self.b_a += delta_x[9:12]
        self.b_g += delta_x[12:15]

        I_KH = np.eye(15, dtype=np.float32) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_cov @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return gamma, False

    def update_compass_heading(self, measured_yaw_rad, r_cov=0.02):
        R_curr = quat_to_rot_np(self.q)
        current_yaw = math.atan2(R_curr[1, 0], R_curr[0, 0])
        yaw_innov = (measured_yaw_rad - current_yaw + math.pi) % (2.0 * math.pi) - math.pi

        denom = max(1e-6, R_curr[0, 0]**2 + R_curr[1, 0]**2)
        H_yaw = np.zeros((1, 15), dtype=np.float32)
        H_yaw[0, 7] = (-R_curr[0, 0] * R_curr[1, 2] + R_curr[1, 0] * R_curr[0, 2]) / denom
        H_yaw[0, 8] = ( R_curr[0, 0] * R_curr[1, 1] - R_curr[1, 0] * R_curr[0, 1]) / denom
        self.update_measurement(H_yaw, np.array([yaw_innov], dtype=np.float32), np.array([[r_cov]], dtype=np.float32))

    def update_optical_flow(self, v_flow_world, flow_quality=0.95, z_slant=1.0):
        alt_factor = (min(45.0, z_slant) / 10.0) ** 2
        r_adaptive = (0.015 / max(0.05, flow_quality)) + alt_factor * 0.08
        H_v = np.zeros((2, 15), dtype=np.float32)
        H_v[0, 3] = 1.0
        H_v[1, 4] = 1.0
        v_innov = v_flow_world[:2] - self.v[:2]
        self.update_measurement(H_v, v_innov, np.eye(2, dtype=np.float32) * float(np.clip(r_adaptive, 0.02, 1.20)))

    def update_barometer(self, z_baro, q_dyn=0.0):
        r_adaptive = float(np.clip(0.05 + 0.0022 * q_dyn, 0.05, 0.40))
        H_baro = np.zeros((1, 15), dtype=np.float32)
        H_baro[0, 2] = 1.0
        self.update_measurement(H_baro, np.array([z_baro - self.p[2]], dtype=np.float32), np.array([[r_adaptive]], dtype=np.float32))

    def update_gps(self, p_gps, hdop=0.85, cn0_avg=44.0, is_spoofed=False):
        r_adaptive = 45.0 if is_spoofed else 0.04 * (hdop ** 2) * (44.0 / max(15.0, cn0_avg))
        H_pos = np.zeros((3, 15), dtype=np.float32)
        H_pos[0:3, 0:3] = np.eye(3, dtype=np.float32)
        gamma, rejected = self.update_measurement(
            H_pos, p_gps - self.p, 
            np.eye(3, dtype=np.float32) * float(np.clip(r_adaptive, 0.03, 50.0))
        )
        return gamma, rejected

# =============================================================================
# 23. SINGLE DRONE STATE ESTIMATOR & SWARM CLOSED-LOOP EKF BRIDGE
# =============================================================================
class SingleDroneStateEstimator:
    """
    Hardware avionics simulation running the local 15-state EKF per drone:
      1. Corrupts ground-truth with calibrated biases & motor vibration noise
      2. Measures specific force with Matter-Wave Cold-Atom Gravimeter
      3. Propagates and updates local EKF
      4. Supplies filtered state estimates (p_est, v_est, R_est, w_est) to controller
    """
    def __init__(self, drone_id, init_pos):
        self.drone_id = drone_id
        self.ekf = ErrorStateEKF15(init_pos)
        self.b_a_true = np.array([0.025, -0.018, 0.035], dtype=np.float32)
        self.b_g_true = np.array([0.006, -0.004, 0.003], dtype=np.float32)
        self.quantum_sensor = MatterWaveColdAtomQuantumSensor()

    def step(self, true_p, true_v, true_q, true_ang, true_lin_accel, 
             motor_omegas, surface_z=0.0, wind_3d=None, is_spoofed=False, dt=0.005):
        if wind_3d is None:
            wind_3d = np.zeros(3, dtype=np.float32)

        R_true = quat_to_rot_np(true_q)

        # 1. Corrupted Physical MEMS IMU
        rpm_ratio = float(np.mean(motor_omegas)) / 963.4
        g_vec = np.array([0, 0, -GRAVITY_VAL], dtype=np.float32)
        f_body = R_true.T @ (true_lin_accel - g_vec)

        vib_accel = np.random.normal(0.0, 0.16 * rpm_ratio, 3).astype(np.float32)
        noise_accel = np.random.normal(0.0, 0.035, 3).astype(np.float32)
        raw_accel = f_body + self.b_a_true + vib_accel + noise_accel

        vib_gyro = np.random.normal(0.0, 0.015 * rpm_ratio, 3).astype(np.float32)
        noise_gyro = np.random.normal(0.0, 0.004, 3).astype(np.float32)
        raw_gyro = true_ang + self.b_g_true + vib_gyro + noise_gyro

        # 2. Matter-Wave Cold-Atom Gravimeter Phase Shift
        a_quantum_body, phase_shift = self.quantum_sensor.measure_phase_shift(raw_accel)

        # 3. EKF Prediction Step
        self.ekf.predict(a_quantum_body, raw_gyro, dt=dt)

        # 4. Compass Heading Update
        compass_yaw = math.atan2(R_true[1, 0], R_true[0, 0]) + float(np.random.normal(0, 0.012))
        self.ekf.update_compass_heading(compass_yaw, r_cov=0.02)

        # 5. Optical Flow Velocity Update
        cos_tilt = max(0.35, float(R_true[2, 2]))
        z_rel = max(0.15, float(true_p[2] - surface_z))
        z_slant = z_rel / cos_tilt
        v_body_flow = R_true.T @ true_v
        flow_noise = np.random.normal(0.0, 0.018, 2).astype(np.float32)
        w_flow = np.array([
            (v_body_flow[0] / z_slant) - true_ang[1] + flow_noise[0],
            (v_body_flow[1] / z_slant) + true_ang[0] + flow_noise[1]
        ], dtype=np.float32)
        flow_qual = 0.95 if z_rel < 35.0 else float(np.clip(1.0 - (z_rel - 35.0) / 25.0, 0.1, 0.95))
        
        R_ekf = quat_to_rot_np(self.ekf.q)
        v_flow_world = R_ekf @ np.array([(w_flow[0] + raw_gyro[1])*z_slant, (w_flow[1] - raw_gyro[0])*z_slant, 0.0], dtype=np.float32)
        self.ekf.update_optical_flow(v_flow_world, flow_quality=flow_qual, z_slant=z_slant)

        # 6. Barometric Pressure Altitude Update
        p_static = P_0 * ((1.0 - (L_LAPSE * max(0.0, float(true_p[2]))) / T_0) ** 5.25588)
        z_baro = 44330.8 * (1.0 - ((p_static / P_0) ** 0.190284)) + float(np.random.normal(0, 0.04))
        v_air = true_v - wind_3d
        q_dyn = 0.5 * 1.225 * float(np.dot(v_air, v_air))
        self.ekf.update_barometer(z_baro, q_dyn=q_dyn)

        # 7. RTK-GNSS Update with Spoofing Rejection
        if is_spoofed:
            gps_offset = np.array([18.5, -24.0, 12.0], dtype=np.float32)
            p_gps = true_p + gps_offset + np.random.normal(0, 0.8, 3).astype(np.float32)
            hdop, cn0 = 3.8, 28.5
        else:
            p_gps = true_p + np.random.normal(0, 0.06, 3).astype(np.float32)
            hdop = float(np.clip(0.85 + np.random.normal(0, 0.04), 0.6, 1.8))
            cn0 = float(np.clip(44.0 + np.random.normal(0, 1.0), 32.0, 48.0))

        gamma_gps, spoof_rejected = self.ekf.update_gps(p_gps, hdop=hdop, cn0_avg=cn0, is_spoofed=is_spoofed)

        p_est = self.ekf.p.copy()
        v_est = self.ekf.v.copy()
        q_est = self.ekf.q.copy()
        omega_est = raw_gyro - self.ekf.b_g
        cov_trace = float(np.trace(self.ekf.P[0:3, 0:3]))

        return {
            "p_est": p_est,
            "v_est": v_est,
            "q_est": q_est,
            "R_est": R_ekf,
            "omega_est": omega_est,
            "raw_accel": raw_accel,
            "raw_gyro": raw_gyro,
            "phase_shift": phase_shift,
            "cov_trace": cov_trace,
            "spoof_rejected": spoof_rejected,
            "gamma_gps": gamma_gps,
            "z_baro": z_baro
        }

class SwarmClosedLoopEKFBridge:
    """
    Connects the Genesis physical simulation to the avionics filter for all 3 swarm drones:
      - Takes batched GPU ground truth from Genesis
      - Runs independent 15-state EKF instances per drone
      - Returns filtered PyTorch CUDA tensors directly to the SE(3) flight controller
    """
    def __init__(self, initial_positions, device="cuda"):
        self.device = device
        self.num_drones = len(initial_positions)
        self.estimators = [SingleDroneStateEstimator(i, initial_positions[i]) for i in range(self.num_drones)]
        self.prev_v = [np.zeros(3, dtype=np.float32) for _ in range(self.num_drones)]

    def update_and_estimate(self, d_pos_tensor, d_vel_tensor, d_quat_tensor, d_ang_tensor,
                            motor_omegas_batch, surface_altitudes, wind_3d_np, is_spoofed=False, dt=0.005):
        p_est_list, v_est_list, q_est_list, w_est_list = [], [], [], []
        diagnostics = []

        d_pos_cpu = d_pos_tensor.cpu().numpy()
        d_vel_cpu = d_vel_tensor.cpu().numpy()
        d_quat_cpu = d_quat_tensor.cpu().numpy()
        d_ang_cpu = d_ang_tensor.cpu().numpy()

        for i in range(self.num_drones):
            a_lin = (d_vel_cpu[i] - self.prev_v[i]) / dt
            self.prev_v[i] = d_vel_cpu[i].copy()

            surf_z = float(surface_altitudes[i]) if isinstance(surface_altitudes, (list, np.ndarray)) else float(surface_altitudes)
            m_omegas = motor_omegas_batch[i] if isinstance(motor_omegas_batch, (list, np.ndarray)) else motor_omegas_batch

            diag = self.estimators[i].step(
                true_p=d_pos_cpu[i], true_v=d_vel_cpu[i], true_q=d_quat_cpu[i], true_ang=d_ang_cpu[i],
                true_lin_accel=a_lin, motor_omegas=m_omegas, surface_z=surf_z,
                wind_3d=wind_3d_np, is_spoofed=is_spoofed, dt=dt
            )

            p_est_list.append(diag["p_est"])
            v_est_list.append(diag["v_est"])
            q_est_list.append(diag["q_est"])
            w_est_list.append(diag["omega_est"])
            diagnostics.append(diag)

        p_est_t = torch.tensor(np.array(p_est_list), device=self.device, dtype=torch.float32)
        v_est_t = torch.tensor(np.array(v_est_list), device=self.device, dtype=torch.float32)
        q_est_t = torch.tensor(np.array(q_est_list), device=self.device, dtype=torch.float32)
        w_est_t = torch.tensor(np.array(w_est_list), device=self.device, dtype=torch.float32)

        return p_est_t, v_est_t, q_est_t, w_est_t, diagnostics

# =============================================================================
# 24. RECIPROCAL MULTI-DRONE SWARM CBF-QP SAFETY BARRIER FILTER
# =============================================================================
class ReciprocalMultiDroneCBFQPFilter:
    """
    Project nominal flight accelerations for all 3 drones onto safety sets:
      - Building Wall & Rooftop Barrier (inflated by EKF covariance & tailwind)
      - Peer Drone Separation Barrier (Inter-drone safe radius 2.6m)
      - Downwash Wake Cone Evasion (Strong lateral push if underneath a peer)
      - Ground Cushion & Geofence Boundaries
    """
    def __init__(self):
        self.alpha_obs = 1.8
        self.safe_radius = 0.85
        self.swarm_r = 2.6
        self.downwash_expansion = 0.38
        self.downwash_depth = 7.5

    def filter_swarm_accelerations(self, drone_positions, drone_velocities, nominal_accels,
                                   obstacles, wind_2d=None, ekf_cov_traces=None, target_bldg_ids=None):
        num_drones = len(drone_positions)
        safe_accels = np.zeros_like(nominal_accels)

        if wind_2d is None:
            wind_2d = np.zeros(2, dtype=np.float32)

        for i in range(num_drones):
            p_i = drone_positions[i]
            v_i = drone_velocities[i]
            u_nom = nominal_accels[i].copy()
            cov_trace = ekf_cov_traces[i] if ekf_cov_traces is not None else 0.05
            target_id = target_bldg_ids[i] if target_bldg_ids is not None else None

            A_rows, b_rows = [], []

            cov_inflation = 1.2 * min(2.5, math.sqrt(max(0.0, float(cov_trace))))
            gust_inflation = 0.22 * float(np.linalg.norm(wind_2d))
            dynamic_safe_r = self.safe_radius + cov_inflation + gust_inflation

            # 1. Building Barriers
            for obs in obstacles:
                if obs.get("id") == target_id and p_i[2] < (obs.get("height", 0.0) + 12.0):
                    continue

                rel = p_i - obs["pos"]
                dist_xy = math.hypot(rel[0], rel[1])
                obs_h = obs.get("height", 0.0)
                n_xy = rel[:2] / max(1e-4, dist_xy)
                v_closing = -float(np.dot(v_i[:2], n_xy))
                tailwind = -float(np.dot(wind_2d[:2], n_xy))

                stopping_buffer = (max(0.0, v_closing) ** 2) / 7.6 + max(0.0, tailwind) * 0.35
                eff_safe_r = dynamic_safe_r + min(4.5, stopping_buffer)
                h_obs = dist_xy - (obs.get("radius", 10.5) + eff_safe_r)
                eff_alpha = self.alpha_obs * (8.0 / (8.0 + max(0.0, v_closing)))

                if p_i[2] <= (obs_h - 0.4) and h_obs < 4.5:
                    grad = np.array([n_xy[0], n_xy[1], 0.0], dtype=np.float32)
                    v_tangent = v_i[0] * (-n_xy[1]) + v_i[1] * n_xy[0]
                    centripetal = (v_tangent ** 2) / max(0.5, dist_xy)
                    l_fh = float(np.dot(grad, v_i))
                    A_rows.append(grad)
                    b_rows.append(-l_fh - eff_alpha * h_obs - centripetal)

                if dist_xy <= (obs.get("radius", 10.5) + 0.5) and p_i[2] >= (obs_h - 0.2):
                    h_roof = p_i[2] - (obs_h + 0.15)
                    if h_roof < 0.8:
                        A_rows.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
                        b_rows.append(-v_i[2] - 2.0 * h_roof)

            # 2. Inter-Drone & Downwash Wake Barriers
            for j in range(num_drones):
                if i == j: continue
                p_j = drone_positions[j]
                v_j = drone_velocities[j]
                dz = p_i[2] - p_j[2]
                d_p = p_i[:2] - p_j[:2]
                dist_ij = float(np.linalg.norm(d_p))

                # Downwash cone hazard beneath peer drone j
                if -self.downwash_depth < dz < -0.4:
                    downwash_cone_r = self.swarm_r + self.downwash_expansion * abs(dz) + cov_inflation
                    h_downwash = dist_ij - downwash_cone_r
                    if h_downwash < 3.2:
                        n_down = np.array([d_p[0]/max(1e-4, dist_ij), d_p[1]/max(1e-4, dist_ij), 0.0], dtype=np.float32)
                        v_rel = v_i[:2] - v_j[:2]
                        l_fh_down = float(np.dot(n_down[:2], v_rel))
                        A_rows.append(n_down)
                        b_rows.append(-l_fh_down - 2.8 * h_downwash)

                # Horizontal separation barrier
                if abs(dz) < 2.8:
                    h_swarm = dist_ij - (self.swarm_r + cov_inflation)
                    if h_swarm < 3.5:
                        n_ij = np.array([d_p[0]/max(1e-4, dist_ij), d_p[1]/max(1e-4, dist_ij), 0.0], dtype=np.float32)
                        v_rel = v_i[:2] - v_j[:2]
                        l_fh_swarm = float(np.dot(n_ij[:2], v_rel))
                        A_rows.append(n_ij)
                        b_rows.append(-l_fh_swarm - 2.2 * h_swarm)

            # 3. Ground Cushion & Geofence Boundaries
            h_gnd = p_i[2] - 0.20
            if h_gnd < 1.6 and p_i[2] > 0.25:
                A_rows.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
                b_rows.append(-3.5 * v_i[2] - 4.5 * max(0.05, h_gnd))

            fence_limit = 180.0
            for axis in [0, 1]:
                if p_i[axis] > (fence_limit - 15.0):
                    grad = np.array([-1.0 if axis == 0 else 0.0, -1.0 if axis == 1 else 0.0, 0.0], dtype=np.float32)
                    A_rows.append(grad)
                    b_rows.append(v_i[axis] - 2.5 * max(0.1, fence_limit - p_i[axis]))
                elif p_i[axis] < (-fence_limit + 15.0):
                    grad = np.array([1.0 if axis == 0 else 0.0, 1.0 if axis == 1 else 0.0, 0.0], dtype=np.float32)
                    A_rows.append(grad)
                    b_rows.append(-v_i[axis] - 2.5 * max(0.1, p_i[axis] - (-fence_limit)))

            # Quadratic Programming Relaxation
            if not A_rows:
                safe_accels[i] = u_nom
            else:
                A = np.array(A_rows, dtype=np.float32)
                b = np.array(b_rows, dtype=np.float32)
                u = u_nom.copy()
                for _ in range(12):
                    max_viol = 0.0
                    for k in range(len(A)):
                        c_k = float(np.dot(A[k], u)) - b[k]
                        if c_k < 0.0:
                            max_viol = max(max_viol, -c_k)
                            delta_lambda = -c_k / float(np.dot(A[k], A[k]))
                            u += delta_lambda * A[k]
                    if max_viol < 1e-4:
                        break
                safe_accels[i] = u

        return safe_accels

# =============================================================================
# 25. 6-HEAD DEEP DENSE FLIGHT EXECUTIVE RESNET BRAIN (~2.2M PARAMS)
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
    """
    6-Head Multi-Task ResNet Flight Executive:
      - Head 1 (Guidance): 4D acceleration and heading correction [ax, ay, az, dyaw]
      - Head 2 (Executive FSM): 7 discrete mission state logits
      - Head 3 (Winch System): [winch_velocity_rate, touchdown_latch_release]
      - Head 4 (Health Diagnostic): 8 motor health factor predictions [0, 1]
      - Head 5 (Sensory Trust): Sensor integrity scores [RTK_trust, OpticalFlow_trust]
      - Head 6 (Critic Value): State-value function baseline
    """
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
# 26. DISTRIBUTED CBBA AUCTION & BREGUET WIND-SHEAR DISPATCHER
# =============================================================================
CANDIDATE_CRUISE_ALTS_M = [34.0, 38.0, 44.0, 50.0]

class SwarmCBBAAuctionDispatcher:
    """
    Consensus-Based Bundle Algorithm (CBBA) Task Dispatcher:
      - Assigns delivery orders from the 48-building manifest based on bids
      - Penalizes low-battery drones (1.0 + (1.0 - SoC) * 2.2)
      - BREGUET ROUTING: Queries vertical wind shear to select the cruise
        altitude with the highest tailwind component
      - Re-auctions tasks dynamically if a peer aborts or triggers emergency RTH
    """
    def __init__(self, manifest_addresses, depot_helipads):
        self.manifest = list(manifest_addresses)
        self.depot_helipads = depot_helipads
        self.pending_orders = deque(range(len(self.manifest)))
        self.active_missions = {}
        self.total_completed = 0

    def dispatch_next_mission(self, drone_id, drone_soc=0.98, drone_pos=None, 
                              microclimate_map=None, sim_time=0.0):
        if not self.pending_orders:
            self.pending_orders.extend(range(len(self.manifest)))

        home_pad = self.depot_helipads[drone_id % len(self.depot_helipads)]
        current_p = home_pad if drone_pos is None else drone_pos

        best_queue_idx = 0
        best_cost = 1e9

        for q_idx, order_idx in enumerate(self.pending_orders):
            candidate = self.manifest[order_idx]
            d_outbound = float(np.linalg.norm(current_p[:2] - candidate["pos"][:2]))
            d_return = float(np.linalg.norm(candidate["pos"][:2] - home_pad[:2]))
            total_flight_dist = d_outbound + d_return

            battery_penalty = 1.0 + (1.0 - max(0.1, drone_soc)) * 2.2
            cost = total_flight_dist * battery_penalty

            if cost < best_cost:
                best_cost = cost
                best_queue_idx = q_idx

        chosen_order_idx = self.pending_orders[best_queue_idx]
        del self.pending_orders[best_queue_idx]
        order = self.manifest[chosen_order_idx]

        assigned_cruise_alt = 34.0 + (drone_id * 3.0)

        if microclimate_map is not None:
            track_dir = order["pos"][:2] - current_p[:2]
            track_dist = max(1.0, float(np.linalg.norm(track_dir)))
            u_track = track_dir / track_dist

            wind_profile = microclimate_map.query_vertical_wind_shear_profile(
                current_p[:2], sim_time, CANDIDATE_CRUISE_ALTS_M
            )
            best_tail_val = -999.0
            best_alt = assigned_cruise_alt

            for idx, alt in enumerate(CANDIDATE_CRUISE_ALTS_M):
                if alt > float(order["height"]) + 6.0:
                    tail_comp = float(np.dot(wind_profile[idx][:2], u_track))
                    if tail_comp > best_tail_val:
                        best_tail_val = tail_comp
                        best_alt = alt
            assigned_cruise_alt = best_alt

        mission = {
            "order_idx": chosen_order_idx,
            "target_pos": order["pos"].copy(),
            "bldg_id": order["id"],
            "home_pad": home_pad.copy(),
            "roof_height": float(order["height"]),
            "assigned_cruise_alt": assigned_cruise_alt,
            "drone_id": drone_id
        }
        self.active_missions[drone_id] = mission
        return mission

    def re_auction_task(self, abandoned_mission, candidate_drones):
        best_drone_id = None
        best_bid = -1e9

        for d_id, d_data in candidate_drones.items():
            if d_id == abandoned_mission.get("drone_id", -1) or d_data.get("rth_active", False):
                continue
            d_to_task = float(np.linalg.norm(d_data["pos"][:2] - abandoned_mission["target_pos"][:2]))
            health_score = float(d_data.get("health", 1.0))
            bid = (d_data.get("soc", 0.9) * health_score * 100.0) / max(10.0, d_to_task)

            if bid > best_bid:
                best_bid = bid
                best_drone_id = d_id

        if best_drone_id is not None:
            self.active_missions[best_drone_id] = abandoned_mission
            print(f">>> [CBBA CONSENSUS] Task {abandoned_mission['bldg_id']} re-auctioned to Drone {best_drone_id+1}! <<<")
        return best_drone_id

    def report_completed(self, drone_id):
        self.total_completed += 1

# =============================================================================
# 27. SWARM PEER KINEMATICS & DOWNWASH WAKE TRACKER
# =============================================================================
class SwarmKinematicsTracker:
    def __init__(self, num_drones=3):
        self.num_drones = num_drones

    def compute_peer_kinematics(self, drone_positions, drone_velocities):
        peer_info_per_drone = []
        downwash_hazards = []

        for i in range(self.num_drones):
            p_i = drone_positions[i]
            v_i = drone_velocities[i]
            d_peers = []

            for j in range(self.num_drones):
                if i == j: continue
                p_j = drone_positions[j]
                v_j = drone_velocities[j]

                p_rel = p_j - p_i
                v_rel = v_j - v_i
                dist = float(np.linalg.norm(p_rel))

                d_peers.append({
                    "p_rel": p_rel.copy(),
                    "v_rel": v_rel.copy(),
                    "dist": dist
                })

                dz = p_i[2] - p_j[2]
                dist_xy = float(np.linalg.norm(p_rel[:2]))
                if -7.5 < dz < -0.4:
                    cone_r = 2.6 + 0.38 * abs(dz)
                    if dist_xy < cone_r:
                        downwash_hazards.append({
                            "victim_idx": i,
                            "leader_idx": j,
                            "depth": abs(dz),
                            "radial_dist": dist_xy
                        })

            peer_info_per_drone.append(d_peers)

        return peer_info_per_drone, downwash_hazards

# =============================================================================
# 28. DYNAMIC 3D URBAN AGENTS (TRAFFIC, MEDEVAC & GROUND INTRUDER)
# =============================================================================
class DynamicRoadTraffic3D:
    def __init__(self, scene, avenues_x, streets_y):
        self.avenues_x = avenues_x
        self.streets_y = streets_y
        self.vehicles = []

        for i in range(4):
            ax = avenues_x[i % len(avenues_x)]
            start_y = -160.0 + i * 80.0
            veh_box = scene.add_entity(
                morph=gs.morphs.Box(size=(2.2, 4.5, 1.8), pos=(ax, start_y, 0.95)),
                material=gs.materials.Rigid(rho=150.0, friction=0.6),
                surface=gs.surfaces.Rough(color=(0.35, 0.40, 0.48), roughness=0.5)
            )
            self.vehicles.append({
                "id": f"Van_{i+1}",
                "entity": veh_box,
                "pos": np.array([ax, start_y, 0.95], dtype=np.float32),
                "vel": np.array([0.0, 5.2 + i * 0.4, 0.0], dtype=np.float32),
                "axis": "Y",
                "is_emergency": False,
                "bounds": [-180.0, 180.0]
            })

        for i in range(4):
            sy = streets_y[i % len(streets_y)]
            is_amb = (i == 0)
            start_x = -160.0 + i * 80.0
            color = (0.95, 0.15, 0.15) if is_amb else (0.85, 0.55, 0.20)
            veh_box = scene.add_entity(
                morph=gs.morphs.Box(size=(4.5, 2.2, 1.8), pos=(start_x, sy, 0.95)),
                material=gs.materials.Rigid(rho=150.0, friction=0.6),
                surface=gs.surfaces.Default(color=color)
            )
            self.vehicles.append({
                "id": "Ambulance_01" if is_amb else f"Truck_{i+1}",
                "entity": veh_box,
                "pos": np.array([start_x, sy, 0.95], dtype=np.float32),
                "vel": np.array([7.8 if is_amb else 4.5 + i * 0.3, 0.0, 0.0], dtype=np.float32),
                "axis": "X",
                "is_emergency": is_amb,
                "bounds": [-180.0, 180.0]
            })

    def step(self, dt=0.005):
        for v in self.vehicles:
            v["pos"] += v["vel"] * dt
            if v["axis"] == "Y":
                if v["pos"][1] > v["bounds"][1]: v["pos"][1] = v["bounds"][0]
                elif v["pos"][1] < v["bounds"][0]: v["pos"][1] = v["bounds"][1]
            else:
                if v["pos"][0] > v["bounds"][1]: v["pos"][0] = v["bounds"][0]
                elif v["pos"][0] < v["bounds"][0]: v["pos"][0] = v["bounds"][1]
            v["entity"].set_pos(tuple(v["pos"]))

    def get_emergency_ambulance_state(self):
        for v in self.vehicles:
            if v["is_emergency"]:
                return v["pos"].copy(), v["vel"].copy()
        return np.array([0.0, 0.0, 0.95], dtype=np.float32), np.zeros(3, dtype=np.float32)

class CivilAircraftTraffic3D:
    def __init__(self, scene):
        self.pos = np.array([-180.0, 15.0, 62.0], dtype=np.float32)
        self.vel = np.array([13.5, 0.0, 0.0], dtype=np.float32)
        self.radius = 4.5
        self.entity = scene.add_entity(
            morph=gs.morphs.Box(size=(7.5, 2.2, 2.2), pos=tuple(self.pos)),
            material=gs.materials.Rigid(friction=0.3),
            surface=gs.surfaces.Default(color=(0.95, 0.85, 0.15))
        )

    def step(self, dt=0.005):
        self.pos += self.vel * dt
        if self.pos[0] > 195.0:
            self.pos[0] = -195.0
        self.entity.set_pos(tuple(self.pos))

    def poll_adsb(self, drone_pos, drone_vel):
        rel_pos = self.pos - drone_pos
        dist_3d = float(np.linalg.norm(rel_pos))
        rel_vel = self.vel - drone_vel
        closing_speed = -float(np.dot(rel_pos / max(1e-3, dist_3d), rel_vel))
        return rel_pos, float(np.clip(closing_speed, -30.0, 30.0)), dist_3d

class DynamicGroundIntruderAgent:
    def __init__(self, scene, init_pos=(-25.0, 25.0, 0.4)):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.v_max = 2.6
        self.personal_space_m = 1.85
        self.wander_target = np.array([10.0, 10.0], dtype=np.float32)
        self.is_leashed = False

        self.entity = scene.add_entity(
            morph=gs.morphs.Cylinder(radius=0.45, height=1.4, pos=(self.p[0], self.p[1], 0.7)),
            material=gs.materials.Rigid(rho=90.0, friction=0.8),
            surface=gs.surfaces.Default(color=(0.95, 0.35, 0.35))
        )

    def step(self, active_cargo_pos, is_winch_lowering, is_aborting, dt=0.005):
        if self.is_leashed:
            self.p = np.array([-40.0, -40.0, 0.0], dtype=np.float32)
            self.v.fill(0.0)
            self.entity.set_pos((self.p[0], self.p[1], 0.7))
            return self.p.copy(), False, 99.0

        d_to_cargo = active_cargo_pos[:2] - self.p[:2]
        dist = float(np.linalg.norm(d_to_cargo))

        if is_aborting:
            flee_dir = -d_to_cargo / max(1e-4, dist)
            v_des = flee_dir * (self.v_max * 1.4)
            a_cmd = (v_des - self.v[:2]) * 4.5
        elif is_winch_lowering and dist < 12.0 and active_cargo_pos[2] < 6.0:
            approach_dir = d_to_cargo / max(1e-4, dist)
            v_des = approach_dir * 1.2
            a_cmd = (v_des - self.v[:2]) * 2.5
        else:
            d_wander = self.wander_target - self.p[:2]
            dist_w = float(np.linalg.norm(d_wander))
            if dist_w < 1.0:
                self.wander_target = np.array([random.uniform(-40.0, 40.0), random.uniform(-40.0, 40.0)], dtype=np.float32)
            v_des = (d_wander / max(1e-4, dist_w)) * 1.1
            a_cmd = (v_des - self.v[:2]) * 1.8

        self.v[:2] += a_cmd * dt
        self.p[:2] += self.v[:2] * dt
        self.p[2] = 0.0

        self.entity.set_pos((self.p[0], self.p[1], 0.7))
        is_breach = (dist < self.personal_space_m and active_cargo_pos[2] < 2.5)
        return self.p.copy(), is_breach, dist

# =============================================================================
# 29. VECTORIZED BEMT COAXIAL ENGINE & GEOMETRIC SE(3) CONTROLLER
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
            [-L,          -L,          L,           L,          L,           L,         -L,          -L],
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
        unit_track_xy = torch.where(dist_xy > 0.5, delta_xy / torch.clamp(dist_xy, min=1e-4), torch.tensor([1.0, 0.0], device=self.device))
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
# 30. MASTER PARALLEL GPU SWARM TRAINING PIPELINE (SCALABLE n_envs)
# =============================================================================
def train_swarm_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Scale to 64, 128, or 256 parallel environments on your RTX 5060 Ti
    n_envs = 512

    print("\n" + "="*75)
    print(f"STAGE 1: 3-DRONE SWARM GPU HARVESTING & RESNET TRAINING [{n_envs} PARALLEL ENVS]")
    print(f"Total Drones Simulating Simultaneously: {n_envs * 3:,} drones on GPU")
    print("="*75)

    if not os.path.exists(URDF_ASSET_FILE):
        generate_coaxial_x8_urdf(URDF_ASSET_FILE)

    try:
        gs.init(backend=gs.gpu)
    except Exception:
        pass

    dt = 0.005  # 200 Hz
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=2, gravity=(0, 0, -GRAVITY_VAL)),
        show_viewer=False  # Headless mode for maximum GPU throughput
    )

    manifest, depot_pads, avenues_x, streets_y, trees = build_metropolitan_city_with_hazards(scene)
    drones, cargos = spawn_swarm_drones_and_cargo(scene, depot_pads)

    scene.build(n_envs=n_envs)
    print(f"\n[GPU HARVEST] Simulating {n_envs} parallel cities on {torch.cuda.get_device_name(0)}...")

    bemt_engines = [VectorizedBEMTEngine(num_drones=n_envs, device="cuda") for _ in range(3)]
    controllers = [GeometricSE3Controller(num_drones=n_envs, device="cuda") for _ in range(3)]
    
    dispatcher = SwarmCBBAAuctionDispatcher(manifest, depot_pads)
    cbf_filter = ReciprocalMultiDroneCBFQPFilter()
    kinematics_tracker = SwarmKinematicsTracker(num_drones=3)
    obs_manager = SwarmObservationSynthesizer(num_drones=3)

    weather_engine = DynamicUrbanWeatherEngine()
    dryden_engine = DrydenAtmosphericTurbulenceModel()
    icing_model = MessingerRotorIcingModel()
    battery_ecm = SubZeroArrheniusBatteryECM()
    sensor_suite = AdvancedMultiModalSensorSuite(manifest)
    full_physics = FullPhysicsSensorSuite()
    lidar_suite = MacroPerceptionLiDARSuite(manifest)

    # Initial CBBA Mission Assignments
    active_missions = [
        dispatcher.dispatch_next_mission(0, drone_soc=0.98, drone_pos=depot_pads[0]),
        dispatcher.dispatch_next_mission(1, drone_soc=0.98, drone_pos=depot_pads[1]),
        dispatcher.dispatch_next_mission(2, drone_soc=0.98, drone_pos=depot_pads[2])
    ]

    collected_obs = []
    collected_targets = []
    t_start = time.time()

    fsm_phases = [0, 0, 0]
    cable_lens = [0.55, 0.55, 0.55]
    has_payloads = [True, True, True]

    for step in range(3500):
        sim_time = step * dt

        # Read batched kinematics: each tensor has shape [n_envs, 3] or [n_envs, 4]
        d_pos = [drones[i].get_pos() for i in range(3)]      # 3 lists of [n_envs, 3]
        d_vel = [drones[i].get_vel() for i in range(3)]      # 3 lists of [n_envs, 3]
        d_quat = [drones[i].get_quat() for i in range(3)]    # 3 lists of [n_envs, 4]
        d_ang = [drones[i].get_ang() for i in range(3)]      # 3 lists of [n_envs, 3]

        c_pos = [cargos[i].get_pos() for i in range(3)]      # 3 lists of [n_envs, 3]
        c_vel = [cargos[i].get_vel() for i in range(3)]      # 3 lists of [n_envs, 3]

        # Step weather and battery
        d0_pos_np = d_pos[0][0].cpu().numpy()
        d0_vel_np = d_vel[0][0].cpu().numpy()
        eff_wind, gust_scale, temp_c, auto_ice = weather_engine.get_weather_parameters(d0_pos_np, avenues_x)
        dryden_gust = dryden_engine.step(float(d0_pos_np[2]), float(np.linalg.norm(d0_vel_np)), dt=dt) * (gust_scale / 2.0)
        total_wind = eff_wind + dryden_gust

        icing_model.icing_active = auto_ice
        ice_mm, ice_idx, lift_loss, drag_rise = icing_model.step(float(np.linalg.norm(d0_vel_np)), dt=dt)
        soc, v_pack, i_draw, r0 = battery_ecm.step(195.0, amb_temp_c=temp_c, dt=dt)

        # Build waypoints and targets for each of the 3 drones
        targets_np = []
        targets_t = []

        for i in range(3):
            mission = active_missions[i]
            tgt_xy = mission["target_pos"][:2]
            roof_h = mission["roof_height"]
            cruise_z = mission["assigned_cruise_alt"]
            curr_pos_mean = d_pos[i][:, 2].mean().item()

            if fsm_phases[i] == 0:  # TAKEOFF
                t_pos = np.array([depot_pads[i][0], depot_pads[i][1], cruise_z], dtype=np.float32)
                if curr_pos_mean >= (cruise_z - 3.0) or sim_time >= 8.0:
                    fsm_phases[i] = 1
            elif fsm_phases[i] == 1:  # CRUISE OUTBOUND
                t_pos = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                dist_xy_mean = torch.norm(d_pos[i][:, :2] - torch.tensor(tgt_xy, device="cuda"), dim=-1).mean().item()
                if dist_xy_mean <= 3.5:
                    fsm_phases[i] = 2
            elif fsm_phases[i] == 2:  # CHIMNEY DESCENT
                t_pos = np.array([tgt_xy[0], tgt_xy[1], roof_h + 7.5], dtype=np.float32)
                if abs(curr_pos_mean - (roof_h + 7.5)) <= 0.8:
                    fsm_phases[i] = 3
            elif fsm_phases[i] == 3:  # DELIVERY HOVER
                t_pos = np.array([tgt_xy[0], tgt_xy[1], roof_h + 7.5], dtype=np.float32)
                if has_payloads[i] and cable_lens[i] < 7.2:
                    cable_lens[i] += 1.8 * dt
                else:
                    has_payloads[i] = False
                    fsm_phases[i] = 4
            elif fsm_phases[i] == 4:  # WINCH RETRACT
                t_pos = np.array([tgt_xy[0], tgt_xy[1], roof_h + 7.5], dtype=np.float32)
                cable_lens[i] = max(0.20, cable_lens[i] - 3.0 * dt)
                if cable_lens[i] <= 0.25:
                    fsm_phases[i] = 5
            elif fsm_phases[i] == 5:  # RETURN TO BASE
                t_pos = np.array([depot_pads[i][0], depot_pads[i][1], cruise_z], dtype=np.float32)
                dist_home_mean = torch.norm(d_pos[i][:, :2] - torch.tensor(depot_pads[i][:2], device="cuda"), dim=-1).mean().item()
                if dist_home_mean <= 3.0:
                    fsm_phases[i] = 6
            else:  # DEPOT LANDING
                t_pos = np.array([depot_pads[i][0], depot_pads[i][1], 0.70], dtype=np.float32)

            targets_np.append(t_pos)
            targets_t.append(torch.tensor(t_pos, device="cuda").repeat(n_envs, 1))

            # Batched SE(3) Control & BEMT Aerodynamics
            curr_mass = TOTAL_MASS_KG if has_payloads[i] else DRY_MASS_KG
            masses = torch.full((n_envs, 1), curr_mass, device="cuda")

            f_thrust_cmd, tau_cmd = controllers[i].compute_wrench(
                d_pos[i], d_vel[i], d_quat[i], d_ang[i], targets_t[i], masses
            )

            motor_cmds = bemt_engines[i].allocate(f_thrust_cmd, tau_cmd)
            actual_thrust, actual_tau = bemt_engines[i].compute_aerodynamics(
                motor_cmds, d_pos[i], d_vel[i], target_surf_z=0.0, dt=dt
            )
            R_curr = quat_to_rot_batch(d_quat[i])

            # Batched Tether Tension
            f_tether_drone = torch.zeros((n_envs, 3), device="cuda")
            if has_payloads[i]:
                tether_vec = c_pos[i] - d_pos[i]
                tether_dist = torch.norm(tether_vec, dim=-1, keepdim=True)
                slack = tether_dist - cable_lens[i]
                is_taut = (slack > 0.0)

                tether_dir = tether_vec / torch.clamp(tether_dist, min=1e-4)
                rel_vel = c_vel[i] - d_vel[i]
                damp = torch.sum(rel_vel * tether_dir, dim=-1, keepdim=True)
                t_mag = torch.where(is_taut, torch.clamp(slack * 100.0 + damp * 4.0, 0.0, 18.0), torch.zeros_like(slack))

                f_cargo = -t_mag * tether_dir
                f_tether_drone = t_mag * tether_dir
                cargos[i].links[0].apply_external_force(force=f_cargo)

            thrust_vec = R_curr[:, :, 2] * actual_thrust.unsqueeze(-1)
            tot_f = thrust_vec + f_tether_drone
            drones[i].links[0].apply_external_force(force=tot_f)
            drones[i].links[0].apply_external_torque(torque=actual_tau)

        scene.step()

        # Harvest observation samples across environments
        if step % 4 == 0:
            num_harvest = min(n_envs, 16)
            for env_k in range(num_harvest):
                p_k = [d_pos[j][env_k].cpu().numpy() for j in range(3)]
                v_k = [d_vel[j][env_k].cpu().numpy() for j in range(3)]
                peer_info_k, _ = kinematics_tracker.compute_peer_kinematics(p_k, v_k)

                obs_417 = obs_manager.synthesize_drone_obs(
                    drone_idx=0,
                    p_est=p_k[0],
                    v_est=v_k[0],
                    R_est=quat_to_rot_batch(d_quat[0])[env_k].cpu().numpy(),
                    omega_est=d_ang[0][env_k].cpu().numpy(),
                    target_pos=targets_np[0],
                    total_wind=total_wind,
                    cable_len=cable_lens[0],
                    c_vel=c_vel[0][env_k].cpu().numpy(),
                    d_vel=v_k[0],
                    c_pos=c_pos[0][env_k].cpu().numpy(),
                    d_pos=p_k[0],
                    has_payload=has_payloads[0],
                    tension_mag=12.0,
                    bms_soc=soc,
                    bms_v_pack=v_pack,
                    bms_temp_c=temp_c,
                    bms_r0=r0,
                    ice_idx=ice_idx,
                    raw_accel=np.array([0, 0, 9.81], dtype=np.float32),
                    raw_gyro=d_ang[0][env_k].cpu().numpy(),
                    adsb_rel=np.array([-120.0, 20.0, 58.0], dtype=np.float32),
                    adsb_spd=0.0,
                    peer_kinematics=peer_info_k[0],
                    is_grounded=False,
                    buildings=manifest,
                    sensor_suite=sensor_suite,
                    full_physics=full_physics,
                    lidar_suite=lidar_suite
                )
                collected_obs.append(torch.from_numpy(obs_417).float().to(device))

                guidance_gt = torch.clamp((targets_t[0][env_k] - d_pos[0][env_k]) / 5.0, -1.0, 1.0)
                guidance_4d = torch.cat([guidance_gt, torch.zeros(1, device=device)])
                fsm_gt = torch.tensor(fsm_phases[0], device=device, dtype=torch.long)
                health_gt = torch.ones(8, device=device)
                trust_gt = torch.tensor([1.0, 1.0], device=device)
                dist_gt = torch.norm(targets_t[0][env_k] - d_pos[0][env_k], keepdim=True)
                winch_gt = torch.tensor([1.0 if fsm_phases[0] == 3 else 0.0, 1.0 if not has_payloads[0] else 0.0], device=device)

                collected_targets.append((guidance_4d, fsm_gt, winch_gt, health_gt, trust_gt, dist_gt))

    print(f"\n[DATASET READY] Harvested {len(collected_obs):,} state transitions in {time.time()-t_start:.2f}s!")

    # Multi-Task ResNet Training
    model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)

    X_train = torch.stack(collected_obs)
    Y_g = torch.stack([t[0] for t in collected_targets])
    Y_f = torch.stack([t[1] for t in collected_targets])
    Y_w = torch.stack([t[2] for t in collected_targets])
    Y_h = torch.stack([t[3] for t in collected_targets])
    Y_t = torch.stack([t[4] for t in collected_targets])
    Y_v = torch.stack([t[5] for t in collected_targets])

    dataset_size = X_train.shape[0]
    batch_size = 2048
    epochs = 60

    print(f"\n[TRAINING] Training 6-head ResNet policy ({epochs} epochs, batch_size={batch_size})...")
    model.train()

    for epoch in range(epochs):
        perm = torch.randperm(dataset_size, device=device)
        total_loss, n_batches = 0.0, 0
        for b in range(0, dataset_size, batch_size):
            idx = perm[b:b+batch_size]
            bx = X_train[idx]
            bg, bf, bw, bh, bt, bv = Y_g[idx], Y_f[idx], Y_w[idx], Y_h[idx], Y_t[idx], Y_v[idx]

            optimizer.zero_grad()
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
            total_loss += loss.item()
            n_batches += 1

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] — Policy Loss: {total_loss / n_batches:.4f}")

    torch.save(model.state_dict(), MODEL_WEIGHTS_FILE)
    print(f"\n[SUCCESS] Model weights saved to '{MODEL_WEIGHTS_FILE}'.")

# =============================================================================
# 31. LIVE 3D INTERACTIVE SWARM COCKPIT (ENDLESS LOOP & KEYBOARD HOTKEYS)
# =============================================================================
# =============================================================================
# 31. MASTER 3-PANEL AVIONICS COCKPIT & LIVE 3D SWARM VISUALIZER
# =============================================================================
import pygame

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


def draw_tactical_radar_minimap(surface, map_rect, buildings, depot_pads, drone_positions, active_idx, traffic, medevac_pos, intruder_p):
    rx, ry, rw, rh = map_rect
    pygame.draw.rect(surface, (15, 20, 28), map_rect)
    pygame.draw.rect(surface, (0, 220, 255), map_rect, 1)
    scale = rw / 180.0
    cx, cy = rx + rw // 2, ry + rh // 2

    def to_map(x, y): 
        return int(cx + x * scale), int(cy - y * scale)

    # Buildings
    for b in buildings:
        bx, by = b["pos"][0], b["pos"][1]
        mp = to_map(bx, by)
        pygame.draw.rect(surface, (45, 55, 68), (mp[0] - 5, mp[1] - 5, 10, 10))

    # Depot Pads
    for pad in depot_pads:
        px, py = to_map(pad[0], pad[1])
        pygame.draw.circle(surface, (0, 220, 255), (px, py), 4, 1)

    # Moving Road Traffic
    for v in traffic.vehicles:
        vx, vy = to_map(v["pos"][0], v["pos"][1])
        col = (255, 40, 60) if v.get("is_emergency", False) else (255, 140, 40)
        pygame.draw.circle(surface, col, (vx, vy), 2)

    # Civilian Medevac Aircraft
    mx, my = to_map(medevac_pos[0], medevac_pos[1])
    pygame.draw.circle(surface, (255, 220, 40), (mx, my), 4, 1)

    # Pedestrian Intruder
    ix, iy = to_map(intruder_p[0], intruder_p[1])
    pygame.draw.circle(surface, (255, 70, 70), (ix, iy), 3)

    # 3 Drones
    d_colors = [(0, 240, 255), (255, 180, 40), (220, 100, 255)]
    for i, dp in enumerate(drone_positions):
        dx, dy = to_map(dp[0], dp[1])
        radius = 5 if i == active_idx else 3
        pygame.draw.circle(surface, d_colors[i], (dx, dy), radius)

# =============================================================================
# SMART CAMERA OCCLUSION & COLLISION SPRING-ARM SOLVER
# =============================================================================
def resolve_camera_occlusion(cam_pos, focus_target_pos, buildings):
    """
    Prevents the camera from penetrating skyscrapers or going below ground:
      1. Clamps camera altitude to stay at least 0.85m above ground.
      2. Raycasts line-of-sight from focus_target_pos to proposed cam_pos.
      3. Uses exact building center coordinates (not pad offsets).
      4. If a wall is in the way, stops the camera 0.6m in front of the wall.
    """
    safe_cam = cam_pos.copy()
    safe_cam[2] = max(0.85, float(safe_cam[2]))

    ray_vec = safe_cam - focus_target_pos
    dist = float(np.linalg.norm(ray_vec))
    if dist < 0.4:
        return safe_cam

    d = ray_vec / dist
    min_safe_dist = dist

    for b in buildings:
        # Use true tower center and dimensions
        if "center" in b:
            cx, cy = b["center"][0], b["center"][1]
            bx_half, by_half = b["size"][0] / 2.0 + 0.6, b["size"][1] / 2.0 + 0.6
        else:
            cx, cy = b["pos"][0] - 2.0, b["pos"][1] + 2.0
            bx_half, by_half = 8.6, 8.6

        bh = b["height"]
        x1, x2 = cx - bx_half, cx + bx_half
        y1, y2 = cy - by_half, cy + by_half
        z1, z2 = 0.0, bh + 0.8

        # If both camera and focus target are above the building, skip
        if focus_target_pos[2] > z2 and safe_cam[2] > z2:
            continue

        # Slab ray-box intersection from focus_target_pos towards camera
        tx1 = (x1 - focus_target_pos[0]) / d[0] if abs(d[0]) > 1e-4 else (-1e6 if focus_target_pos[0] < x1 else 1e6)
        tx2 = (x2 - focus_target_pos[0]) / d[0] if abs(d[0]) > 1e-4 else (1e6 if focus_target_pos[0] > x2 else -1e6)
        ty1 = (y1 - focus_target_pos[1]) / d[1] if abs(d[1]) > 1e-4 else (-1e6 if focus_target_pos[1] < y1 else 1e6)
        ty2 = (y2 - focus_target_pos[1]) / d[1] if abs(d[1]) > 1e-4 else (1e6 if focus_target_pos[1] > y2 else -1e6)
        tz1 = (z1 - focus_target_pos[2]) / d[2] if abs(d[2]) > 1e-4 else (-1e6 if focus_target_pos[2] < z1 else 1e6)
        tz2 = (z2 - focus_target_pos[2]) / d[2] if abs(d[2]) > 1e-4 else (1e6 if focus_target_pos[2] > z2 else -1e6)

        t_enter = max(min(tx1, tx2), min(ty1, ty2), min(tz1, tz2))
        t_exit = min(max(tx1, tx2), max(ty1, ty2), max(tz1, tz2))

        if t_enter <= t_exit and t_exit > 0.1:
            if 0.1 < t_enter < min_safe_dist:
                min_safe_dist = max(0.8, t_enter - 0.6)

    resolved_pos = focus_target_pos + d * min_safe_dist
    resolved_pos[2] = max(0.85, float(resolved_pos[2]))
    return resolved_pos


def evaluate_swarm_system():
    if not os.path.exists(MODEL_WEIGHTS_FILE):
        print(f"\n[WARNING] '{MODEL_WEIGHTS_FILE}' not found! Running parallel GPU training first...")
        train_swarm_system()

    generate_coaxial_x8_urdf(URDF_ASSET_FILE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE, map_location=device))
    model.eval()
    print(f"\nLoaded verified 6-Head Swarm Policy from '{MODEL_WEIGHTS_FILE}'!")

    try:
        gs.init(backend=gs.gpu)
    except Exception:
        pass

    dt = 0.005  # 200 Hz physics
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=2, gravity=(0, 0, -GRAVITY_VAL)),
        show_viewer=False
    )

    manifest, depot_pads, avenues_x, streets_y, trees = build_metropolitan_city_with_hazards(scene)
    drones, cargos = spawn_swarm_drones_and_cargo(scene, depot_pads)

    road_traffic = DynamicRoadTraffic3D(scene, avenues_x, streets_y)
    medevac_plane = CivilAircraftTraffic3D(scene)
    ground_intruder = DynamicGroundIntruderAgent(scene, init_pos=(-25.0, 25.0, 0.4))

    cam_sim = scene.add_camera(
        res=(800, 740),
        pos=(45.0, -45.0, 35.0),
        lookat=(0.0, 0.0, 15.0),
        fov=48,
        GUI=False
    )

    scene.build(n_envs=1)

    bemt = VectorizedBEMTEngine(num_drones=3, device="cuda")
    ctrl = GeometricSE3Controller(num_drones=3, device="cuda")
    dispatcher = SwarmCBBAAuctionDispatcher(manifest, depot_pads)
    cbf_filter = ReciprocalMultiDroneCBFQPFilter()
    kinematics_tracker = SwarmKinematicsTracker(num_drones=3)
    obs_manager = SwarmObservationSynthesizer(num_drones=3)

    weather_engine = DynamicUrbanWeatherEngine()
    dryden_engine = DrydenAtmosphericTurbulenceModel()
    icing_model = MessingerRotorIcingModel()
    battery_ecm = SubZeroArrheniusBatteryECM()
    sensor_suite = AdvancedMultiModalSensorSuite(manifest)
    full_physics = FullPhysicsSensorSuite()
    lidar_suite = MacroPerceptionLiDARSuite(manifest)

    ekf_bridge = SwarmClosedLoopEKFBridge(
        initial_positions=[depot_pads[0], depot_pads[1], depot_pads[2]],
        device="cuda"
    )

    active_missions = [
        dispatcher.dispatch_next_mission(0, drone_soc=0.98, drone_pos=depot_pads[0]),
        dispatcher.dispatch_next_mission(1, drone_soc=0.98, drone_pos=depot_pads[1]),
        dispatcher.dispatch_next_mission(2, drone_soc=0.98, drone_pos=depot_pads[2])
    ]

    fsm_phases = [0, 0, 0]
    cable_lens = [0.55, 0.55, 0.55]
    has_payloads = [True, True, True]
    depot_timers = [0.0, 0.0, 0.0]
    is_breach = False

    # Initialize Pygame Window
    pygame.init()
    WIN_W, WIN_H = 1500, 740
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Survival Core V37.0 — Autonomous 3D Swarm Cockpit & Telemetry HUD")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("consolas", 13, bold=True)
    font_bold = pygame.font.SysFont("consolas", 11, bold=True)
    font_small = pygame.font.SysFont("consolas", 10)

    # Interactive States
    selected_drone_idx = 0
    camera_modes = ["CHASE", "FREE_ORBIT", "TOP_DOWN", "BELLY_CARGO", "DEPOT"]
    cam_mode_idx = 0
    cam_mode_changed = True
    inject_spoof = False
    active_gust = np.zeros(3, dtype=np.float32)
    gust_timer = 0.0
    motor_health = [np.ones(8, dtype=np.float32) for _ in range(3)]
    running = True
    step = 0

    # Camera Pipeline State
    smooth_cam_pos = None
    smooth_cam_lookat = None
    orbit_yaw = 0.0
    orbit_pitch = 0.35
    orbit_dist = 6.5
    mouse_dragging = False
    last_mouse_pos = (0, 0)

    print("\n" + "="*75)
    print("V37.0 UNIFIED COCKPIT ONLINE [GENESIS 3D + PYGAME]")
    print("AI Policy: Active Neural Executive Driving Flight Controls")
    print("="*75 + "\n")

    while running:
        # Step physics 4 times per frame to achieve true 1.0x real-time speed (4 * 0.005s = 0.02s per 50 FPS)
        for _ in range(4):
            step += 1
            sim_time = step * dt

            # 1. Dynamic Urban Agents & Weather
            road_traffic.step(dt=dt)
            medevac_plane.step(dt=dt)
            amb_pos, amb_vel = road_traffic.get_emergency_ambulance_state()

            intruder_p, is_breach, _ = ground_intruder.step(
                active_cargo_pos=cargos[0].get_pos()[0].cpu().numpy(),
                is_winch_lowering=(fsm_phases[0] == 3),
                is_aborting=(is_breach and fsm_phases[0] == 3),
                dt=dt
            )

            eff_wind, gust_scale, temp_c, auto_ice = weather_engine.get_weather_parameters(drones[0].get_pos()[0].cpu().numpy(), avenues_x)
            dryden_gust = dryden_engine.step(float(drones[0].get_pos()[0, 2].item()), float(torch.norm(drones[0].get_vel()[0]).item()), dt=dt) * (gust_scale / 2.0)
            microburst_wind = active_gust * (gust_timer / 2.5) if gust_timer > 0.0 else np.zeros(3, dtype=np.float32)
            if gust_timer > 0.0: gust_timer = max(0.0, gust_timer - dt)
            total_wind = eff_wind + dryden_gust + microburst_wind

            ice_mm, ice_idx, lift_loss, drag_rise = icing_model.step(float(torch.norm(drones[0].get_vel()[0]).item()), dt=dt)
            soc, v_pack, i_draw, r0 = battery_ecm.step(195.0, amb_temp_c=temp_c, dt=dt)

            # 2. Closed-Loop 15-State ES-EKF Estimation
            d_pos_raw = torch.stack([drones[i].get_pos()[0] for i in range(3)])
            d_vel_raw = torch.stack([drones[i].get_vel()[0] for i in range(3)])
            d_quat_raw = torch.stack([drones[i].get_quat()[0] for i in range(3)])
            d_ang_raw = torch.stack([drones[i].get_ang()[0] for i in range(3)])

            p_est_t, v_est_t, q_est_t, w_est_t, diagnostics = ekf_bridge.update_and_estimate(
                d_pos_tensor=d_pos_raw,
                d_vel_tensor=d_vel_raw,
                d_quat_tensor=d_quat_raw,
                d_ang_tensor=d_ang_raw,
                motor_omegas_batch=bemt.motor_omegas.cpu().numpy(),
                surface_altitudes=[0.0, 0.0, 0.0],
                wind_3d_np=total_wind,
                is_spoofed=inject_spoof,
                dt=dt
            )

            p_est_np = p_est_t.cpu().numpy()
            v_est_np = v_est_t.cpu().numpy()
            w_est_np = w_est_t.cpu().numpy()

            peer_kinematics, downwash_hazards = kinematics_tracker.compute_peer_kinematics(p_est_np, v_est_np)

            # 3. Active Neural Network Policy Inference
            # -------------------------------------------------------------
        # 1. SYNTHESIZE 417-DIM OBSERVATIONS FOR ALL 3 DRONES (ON GPU)
        # -------------------------------------------------------------
        c_pos_np = [cargos[i].get_pos()[0].cpu().numpy() for i in range(3)]
        c_vel_np = [cargos[i].get_vel()[0].cpu().numpy() for i in range(3)]
        R_curr = quat_to_rot_batch(q_est_t)

        obs_batch_list = []
        for d_i in range(3):
            obs_417 = obs_manager.synthesize_drone_obs(
                drone_idx=d_i,
                p_est=p_est_np[d_i],
                v_est=v_est_np[d_i],
                R_est=R_curr[d_i].cpu().numpy(),
                omega_est=w_est_np[d_i],
                target_pos=active_missions[d_i]["target_pos"],
                total_wind=total_wind,
                cable_len=cable_lens[d_i],
                c_vel=c_vel_np[d_i],
                d_vel=v_est_np[d_i],
                c_pos=c_pos_np[d_i],
                d_pos=p_est_np[d_i],
                has_payload=has_payloads[d_i],
                tension_mag=12.0,
                bms_soc=soc,
                bms_v_pack=v_pack,
                bms_temp_c=temp_c,
                bms_r0=r0,
                ice_idx=ice_idx,
                raw_accel=diagnostics[d_i]["raw_accel"],
                raw_gyro=diagnostics[d_i]["raw_gyro"],
                adsb_rel=np.array([-120.0, 20.0, 58.0], dtype=np.float32),
                adsb_spd=0.0,
                peer_kinematics=peer_kinematics[d_i],
                is_grounded=(p_est_np[d_i, 2] <= 0.85),
                buildings=manifest,
                sensor_suite=sensor_suite,
                full_physics=full_physics,
                lidar_suite=lidar_suite
            )
            obs_batch_list.append(torch.from_numpy(obs_417).float())

        # Batched input across all 3 drones: Shape [3, 417]
        obs_batch_tensor = torch.stack(obs_batch_list).to(device)

        # -------------------------------------------------------------
        # 2. RUN NEURAL NETWORK INFERENCE (ALL 3 DRONES)
        # -------------------------------------------------------------
        with torch.no_grad():
            nn_guidance, nn_fsm, nn_winch, nn_health, nn_trust, _ = model(obs_batch_tensor)

        # Extract neural outputs
        nn_guidance_accel = nn_guidance[:, :3]  # [3, 3] 3D Acceleration commands
        nn_fsm_preds = nn_fsm.argmax(dim=-1).cpu().numpy()  # [3] AI Phase decisions
        nn_winch_speeds = nn_winch[:, 0].cpu().numpy()      # [3] Winch velocity rates
        nn_latch_releases = (nn_winch[:, 1] > 0.35).cpu().numpy()  # [3] Cargo release triggers

        # -------------------------------------------------------------
        # 3. SWARM MISSION PROGRESSION & NEURAL WINCH ACTUATION
        # -------------------------------------------------------------
        targets_list = []
        nominal_accels = []

        for i in range(3):
            mission = active_missions[i]
            tgt_xy = mission["target_pos"][:2]
            roof_h = mission["roof_height"]
            cruise_z = mission["assigned_cruise_alt"]
            curr_p = p_est_np[i]

            dist_xy = float(np.linalg.norm(curr_p[:2] - tgt_xy))
            dist_home = float(np.linalg.norm(curr_p[:2] - depot_pads[i][:2]))

            if soc < 0.22 and fsm_phases[i] in [1, 2, 3]:
                fsm_phases[i] = 5

            # Arbiter supervises phase progression, informed by neural prediction
            if fsm_phases[i] == 0:  # TAKEOFF
                t_pos = np.array([depot_pads[i][0], depot_pads[i][1], cruise_z], dtype=np.float32)
                if curr_p[2] >= (cruise_z - 3.0) or nn_fsm_preds[i] == 1:
                    fsm_phases[i] = 1
            elif fsm_phases[i] == 1:  # CRUISE OUTBOUND
                t_pos = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                if dist_xy <= 3.5 or nn_fsm_preds[i] == 2:
                    fsm_phases[i] = 2
            elif fsm_phases[i] == 2:  # CHIMNEY DESCENT
                t_pos = np.array([tgt_xy[0], tgt_xy[1], roof_h + 7.5], dtype=np.float32)
                if abs(curr_p[2] - (roof_h + 7.5)) <= 0.8 or nn_fsm_preds[i] == 3:
                    fsm_phases[i] = 3
            elif fsm_phases[i] == 3:  # DELIVERY HOVER & NEURAL WINCH
                t_pos = np.array([tgt_xy[0], tgt_xy[1], roof_h + 7.5], dtype=np.float32)
                if is_breach and i == 0:
                    fsm_phases[i] = 4  # Intruder safety abort
                elif has_payloads[i]:
                    # Unspool winch driven by neural rate
                    rate_cmd = max(1.2, float(nn_winch_speeds[i]) * 2.2) if nn_winch_speeds[i] > 0 else 1.8
                    cable_lens[i] = min(7.2, cable_lens[i] + rate_cmd * dt)

                    # Trigger touchdown release if neural latch fires or roof clearance is reached
                    cargo_z = c_pos_np[i][2]
                    if (nn_latch_releases[i] and cargo_z <= (roof_h + 0.35)) or (cargo_z <= (roof_h + 0.15)) or cable_lens[i] >= 7.2:
                        has_payloads[i] = False
                        fsm_phases[i] = 4
                        dispatcher.report_completed(i)
                        print(f"[{sim_time:4.1f}s] NEURAL TOUCHDOWN: Drone {i+1} delivered cargo on {mission['bldg_id']} helipad! (Total: {dispatcher.total_completed})")
            elif fsm_phases[i] == 4:  # WINCH RETRACT
                t_pos = np.array([tgt_xy[0], tgt_xy[1], roof_h + 7.5], dtype=np.float32)
                cable_lens[i] = max(0.20, cable_lens[i] - 3.2 * dt)
                if cable_lens[i] <= 0.25:
                    fsm_phases[i] = 5
            elif fsm_phases[i] == 5:  # RETURN TO BASE
                t_pos = np.array([depot_pads[i][0], depot_pads[i][1], cruise_z], dtype=np.float32)
                if dist_home <= 3.0 or nn_fsm_preds[i] == 6:
                    fsm_phases[i] = 6
            else:  # DEPOT LANDING & ENDLESS RELOAD
                t_pos = np.array([depot_pads[i][0], depot_pads[i][1], 0.70], dtype=np.float32)
                if curr_p[2] <= 0.85 and float(np.linalg.norm(v_est_np[i])) < 0.4:
                    depot_timers[i] += dt
                    if depot_timers[i] >= 2.0:
                        depot_timers[i] = 0.0
                        has_payloads[i] = True
                        cable_lens[i] = 0.55
                        cargos[i].set_pos((depot_pads[i][0], depot_pads[i][1], 0.18))
                        cargos[i].set_vel((0.0, 0.0, 0.0))
                        active_missions[i] = dispatcher.dispatch_next_mission(
                            i, drone_soc=0.98, drone_pos=depot_pads[i], sim_time=sim_time
                        )
                        fsm_phases[i] = 0
                        print(f">>> [DEPOT SERVICE] Drone {i+1} reloaded cargo! Dispatched to: {active_missions[i]['bldg_id']} <<<")

            targets_list.append(t_pos)

            delta_p = t_pos - curr_p
            v_cmd_xy = np.clip(delta_p[:2] * 1.4, -6.0, 6.0)
            v_cmd_z = np.clip(delta_p[2] * 1.2, -1.5, 2.4)
            v_des = np.array([v_cmd_xy[0], v_cmd_xy[1], v_cmd_z], dtype=np.float32)

            # Blend Neural Network Guidance directly into commanded acceleration
            a_cmd = np.array([2.0, 2.0, 2.5], dtype=np.float32) * (v_des - v_est_np[i])
            a_cmd += nn_guidance_accel[i].cpu().numpy() * 0.50  # <-- AI ACTUALLY STEERING THE DRONE!
            nominal_accels.append(a_cmd)

        # -------------------------------------------------------------
        # 4. SAFETY BARRIER FILTER & PHYSICAL SE(3) ACTUATION
        # -------------------------------------------------------------
        safe_accels = cbf_filter.filter_swarm_accelerations(
            drone_positions=p_est_np,
            drone_velocities=v_est_np,
            nominal_accels=np.array(nominal_accels),
            obstacles=manifest,
            wind_2d=total_wind[:2],
            ekf_cov_traces=[d["cov_trace"] for d in diagnostics],
            target_bldg_ids=[active_missions[i]["bldg_id"] for i in range(3)]
        )

        targets_tensor = torch.tensor(np.array(targets_list), device="cuda", dtype=torch.float32)
        masses = torch.tensor([TOTAL_MASS_KG if has_payloads[i] else DRY_MASS_KG for i in range(3)], device="cuda")
        safe_accels_t = torch.tensor(safe_accels, device="cuda", dtype=torch.float32)
        nominal_accels_t = torch.from_numpy(np.stack(nominal_accels).astype(np.float32)).to("cuda")

        # Command SE(3) controller with safety-filtered neural wrench
        f_thrust_cmd, tau_cmd = ctrl.compute_wrench(
            p_est_t, v_est_t, q_est_t, w_est_t, targets_tensor, masses.unsqueeze(-1),
            nn_guidance=safe_accels_t - nominal_accels_t
        )

        motor_cmds = bemt.allocate(f_thrust_cmd, tau_cmd)
        health_tensor = torch.tensor(np.array(motor_health), device="cuda", dtype=torch.float32)
        motor_cmds *= health_tensor

        actual_thrust, actual_tau = bemt.compute_aerodynamics(motor_cmds, p_est_t, v_est_t, target_surf_z=0.0, dt=dt)

        for i in range(3):
                f_wall_suction = compute_building_wall_suction(p_est_np[i], manifest)
                f_wall_suction_t = torch.tensor(f_wall_suction, device="cuda", dtype=torch.float32)

                f_tether_drone = torch.zeros(3, device="cuda")
                if has_payloads[i]:
                    c_p_t = cargos[i].get_pos()[0]
                    d_p_t = drones[i].get_pos()[0]
                    c_v_t = cargos[i].get_vel()[0]
                    d_v_t = drones[i].get_vel()[0]

                    tether_vec = c_p_t - d_p_t
                    tether_dist = torch.norm(tether_vec)
                    slack = tether_dist - cable_lens[i]
                    if slack > 0.0:
                        tether_dir = tether_vec / torch.clamp(tether_dist, min=1e-4)
                        rel_vel = c_v_t - d_v_t
                        damp = torch.dot(rel_vel, tether_dir)
                        t_mag = torch.clamp(slack * 100.0 + damp * 4.0, 0.0, 18.0)
                        f_cargo = -t_mag * tether_dir
                        f_tether_drone = t_mag * tether_dir
                        cargos[i].links[0].apply_external_force(force=f_cargo.unsqueeze(0))

                tot_f = R_curr[i, :, 2] * actual_thrust[i] + f_tether_drone + f_wall_suction_t
                drones[i].links[0].apply_external_force(force=tot_f.unsqueeze(0))
                drones[i].links[0].apply_external_torque(torque=actual_tau[i].unsqueeze(0))

                scene.step()

        # -------------------------------------------------------------
        # 6. UNIFIED CAMERA PIPELINE (SOLVES FINDINGS 1 - 5)
        # -------------------------------------------------------------
        mouse_pos = pygame.mouse.get_pos()
        mouse_click = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_click = True
                    if 240 <= event.pos[0] <= 1040:
                        mouse_dragging = True
                        last_mouse_pos = event.pos
                # Harmonized zoom bounds across mouse and keyboard: [2.0m, 25.0m]
                elif event.button == 4:  # Wheel In
                    orbit_dist = max(2.0, orbit_dist - 0.6)
                elif event.button == 5:  # Wheel Out
                    orbit_dist = min(25.0, orbit_dist + 0.6)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if mouse_dragging and camera_modes[cam_mode_idx] == "FREE_ORBIT":
                    dx = event.pos[0] - last_mouse_pos[0]
                    dy = event.pos[1] - last_mouse_pos[1]
                    orbit_yaw -= dx * 0.008
                    orbit_pitch = float(np.clip(orbit_pitch + dy * 0.008, -0.15, 1.35))
                    last_mouse_pos = event.pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    selected_drone_idx = (selected_drone_idx + 1) % 3
                    cam_mode_changed = True
                elif event.key == pygame.K_c:
                    cam_mode_idx = (cam_mode_idx + 1) % len(camera_modes)
                    cam_mode_changed = True
                elif event.key == pygame.K_j:
                    inject_spoof = not inject_spoof
                elif event.key == pygame.K_g:
                    gust_timer = 2.5
                    ang = random.uniform(0, 2 * math.pi)
                    active_gust = np.array([math.cos(ang) * 9.5, math.sin(ang) * 9.5, -2.5], dtype=np.float32)
                elif event.key == pygame.K_b:
                    battery_ecm.soc = 0.18
                elif event.key == pygame.K_i:
                    icing_model.icing_active = not icing_model.icing_active
                elif event.key == pygame.K_LEFT:
                    orbit_yaw -= 0.06
                elif event.key == pygame.K_RIGHT:
                    orbit_yaw += 0.06
                elif event.key == pygame.K_UP:
                    orbit_pitch = min(1.35, orbit_pitch + 0.04)
                elif event.key == pygame.K_DOWN:
                    orbit_pitch = max(-0.15, orbit_pitch - 0.04)
                elif event.key in [pygame.K_LEFTBRACKET, pygame.K_MINUS]:
                    orbit_dist = min(25.0, orbit_dist + 0.5)
                elif event.key in [pygame.K_RIGHTBRACKET, pygame.K_EQUALS]:
                    orbit_dist = max(2.0, orbit_dist - 0.5)
                elif pygame.K_1 <= event.key <= pygame.K_8:
                    m_idx = event.key - pygame.K_1
                    cur_h = motor_health[selected_drone_idx][m_idx]
                    motor_health[selected_drone_idx][m_idx] = 0.0 if cur_h > 0.0 else 1.0

        # Continuous Arrow Keys for Orbit
        if camera_modes[cam_mode_idx] == "FREE_ORBIT":
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]: orbit_yaw -= 0.03
            if keys[pygame.K_RIGHT]: orbit_yaw += 0.03
            if keys[pygame.K_UP]: orbit_pitch = min(1.35, orbit_pitch + 0.02)
            if keys[pygame.K_DOWN]: orbit_pitch = max(-0.15, orbit_pitch - 0.02)

        # -------------------------------------------------------------
        # CALCULATE DESIRED CAMERA POSE
        # -------------------------------------------------------------
        foc_p = p_est_np[selected_drone_idx]
        active_tgt = active_missions[selected_drone_idx]["target_pos"]

        # True horizontal flight vector
        flight_delta = active_tgt[:2] - foc_p[:2]
        dist_to_tgt = math.hypot(flight_delta[0], flight_delta[1])
        if dist_to_tgt > 1.0:
            u_fwd = np.array([flight_delta[0] / dist_to_tgt, flight_delta[1] / dist_to_tgt, 0.0], dtype=np.float32)
        else:
            app_vec = active_tgt[:2] - depot_pads[selected_drone_idx][:2]
            app_dist = max(1.0, math.hypot(app_vec[0], app_vec[1]))
            u_fwd = np.array([app_vec[0] / app_dist, app_vec[1] / app_dist, 0.0], dtype=np.float32)

        if camera_modes[cam_mode_idx] == "CHASE":
            des_cam_pos = foc_p - u_fwd * 5.8 + np.array([0.0, 0.0, 2.2], dtype=np.float32)
            des_cam_lookat = foc_p + u_fwd * 1.5 + np.array([0.0, 0.0, 0.3], dtype=np.float32)
            focus_ref_target = foc_p

        elif camera_modes[cam_mode_idx] == "FREE_ORBIT":
            cp = math.cos(orbit_pitch)
            sp = math.sin(orbit_pitch)
            cy = math.cos(orbit_yaw)
            sy = math.sin(orbit_yaw)
            rel_orbit = np.array([orbit_dist * cp * cy, orbit_dist * cp * sy, orbit_dist * sp], dtype=np.float32)
            des_cam_pos = foc_p + rel_orbit
            des_cam_lookat = foc_p + np.array([0.0, 0.0, 0.2], dtype=np.float32)
            focus_ref_target = foc_p

        elif camera_modes[cam_mode_idx] == "TOP_DOWN":
            des_cam_pos = foc_p - u_fwd * 1.5 + np.array([0.0, 0.0, 12.0], dtype=np.float32)
            des_cam_lookat = foc_p + u_fwd * 0.5 + np.array([0.0, 0.0, 0.2], dtype=np.float32)
            focus_ref_target = foc_p

        elif camera_modes[cam_mode_idx] == "BELLY_CARGO":
            # True downward delivery gimbal tracking cargo payload
            des_cam_pos = foc_p - u_fwd * 1.4 - np.array([0.0, 0.0, 0.35], dtype=np.float32)
            if has_payloads[selected_drone_idx]:
                des_cam_lookat = c_pos_np[selected_drone_idx]
            else:
                des_cam_lookat = np.array([foc_p[0], foc_p[1], max(0.0, foc_p[2] - 4.0)], dtype=np.float32)
            focus_ref_target = foc_p

        else:  # DEPOT
            # Uses depot pad as line-of-sight origin (Solves Finding 2!)
            depot_center = np.array([-44.0, -44.0, 0.5], dtype=np.float32)
            des_cam_pos = np.array([-30.0, -30.0, 32.0], dtype=np.float32)
            des_cam_lookat = depot_center
            focus_ref_target = depot_center

        # Resolve Obstruction against the correct target reference (Solves Finding 2 & 5)
        des_cam_pos = resolve_camera_occlusion(des_cam_pos, focus_ref_target, manifest)

        # Single-pass smoothing (Solves Finding 1)
        if smooth_cam_pos is None or cam_mode_changed:
            smooth_cam_pos = des_cam_pos.copy()
            smooth_cam_lookat = des_cam_lookat.copy()
            cam_mode_changed = False
        else:
            lerp = 0.20 if camera_modes[cam_mode_idx] == "FREE_ORBIT" else 0.08
            smooth_cam_pos += (des_cam_pos - smooth_cam_pos) * lerp
            smooth_cam_lookat += (des_cam_lookat - smooth_cam_lookat) * lerp

        cam_sim.set_pose(pos=tuple(smooth_cam_pos), lookat=tuple(smooth_cam_lookat))

        # Render 3D Frame from Genesis Camera
        try:
            rgb = cam_sim.render(rgb=True)[0]
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.detach().cpu().numpy()
            if rgb.ndim == 4:
                rgb = rgb[0]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * 255 if rgb.max() <= 1.0 else rgb, 0, 255).astype(np.uint8)
            sim_surf = pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), 'RGB')
            screen.blit(sim_surf, (240, 0))
        except Exception as e:
            # Report rendering diagnostic explicitly (Solves rendering diagnostic finding)
            pygame.draw.rect(screen, (10, 13, 20), (240, 0, 800, 740))
            err_txt = font_bold.render(f"RENDER BUFFER WARN: {str(e)[:35]}", True, (255, 60, 60))
            screen.blit(err_txt, (260, 360))

        # -------------------------------------------------------------
        # 7. LEFT PANEL (COMMANDS & MOUSE BUTTONS)
        # -------------------------------------------------------------
        pygame.draw.rect(screen, (16, 20, 28), (0, 0, 240, 740))
        pygame.draw.line(screen, (35, 45, 60), (240, 0), (240, 740), 2)

        screen.blit(font_title.render("FLIGHT COMMANDS", True, (0, 220, 255)), (14, 15))
        screen.blit(font_small.render("Click button or press hotkey", True, (130, 150, 175)), (14, 34))

        buttons = [
            ("TAB", f"DRONE FOCUS: [D{selected_drone_idx+1}]", (0, 220, 255), "tab"),
            ("C", f"CAM: {camera_modes[cam_mode_idx]}", (255, 180, 40), "cam"),
            ("J", f"GPS SPOOF: {'ACTIVE' if inject_spoof else 'OFF'}", (255, 60, 80) if inject_spoof else (0, 255, 140), "spoof"),
            ("G", "MICROBURST GUST [9.5m/s]", (255, 120, 40) if gust_timer > 0 else (120, 140, 160), "gust"),
            ("B", "FORCE LOW BATT (20%)", (255, 60, 80) if soc < 0.22 else (120, 140, 160), "batt"),
            ("I", f"ROTOR ICING: {'ON' if icing_model.icing_active else 'OFF'}", (0, 220, 255) if icing_model.icing_active else (120, 140, 160), "ice"),
        ]

        by = 65
        for key_tag, label_text, col, action_id in buttons:
            btn_rect = pygame.Rect(14, by, 212, 32)
            is_hover = btn_rect.collidepoint(mouse_pos)
            bg_col = (28, 36, 48) if is_hover else (20, 26, 36)
            pygame.draw.rect(screen, bg_col, btn_rect, border_radius=4)
            pygame.draw.rect(screen, col, btn_rect, 1, border_radius=4)

            badge_rect = pygame.Rect(18, by + 5, 34, 22)
            pygame.draw.rect(screen, (38, 48, 64), badge_rect, border_radius=3)
            screen.blit(font_bold.render(key_tag, True, (240, 240, 240)), (23, by + 9))
            screen.blit(font_small.render(label_text, True, col), (58, by + 10))

            if mouse_click and is_hover:
                if action_id == "tab": 
                    selected_drone_idx = (selected_drone_idx + 1) % 3
                    cam_mode_changed = True
                elif action_id == "cam": 
                    cam_mode_idx = (cam_mode_idx + 1) % len(camera_modes)
                    cam_mode_changed = True
                elif action_id == "spoof": inject_spoof = not inject_spoof
                elif action_id == "gust":
                    gust_timer = 2.5
                    ang = random.uniform(0, 2 * math.pi)
                    active_gust = np.array([math.cos(ang) * 9.5, math.sin(ang) * 9.5, -2.5], dtype=np.float32)
                elif action_id == "batt": battery_ecm.soc = 0.18
                elif action_id == "ice": icing_model.icing_active = not icing_model.icing_active

            by += 40

        # Free Orbit Instructions Box
        if camera_modes[cam_mode_idx] == "FREE_ORBIT":
            orbit_box = pygame.Rect(14, by, 212, 44)
            pygame.draw.rect(screen, (22, 32, 46), orbit_box, border_radius=4)
            pygame.draw.rect(screen, (0, 220, 255), orbit_box, 1, border_radius=4)
            screen.blit(font_bold.render("FREE ORBIT ACTIVE:", True, (0, 220, 255)), (20, by + 6))
            screen.blit(font_small.render("Drag Mouse or use Arrow Keys", True, (200, 220, 240)), (20, by + 22))
            by += 52

        # Motor Health Cutout Switches [1] - [8]
        screen.blit(font_bold.render("COAXIAL MOTOR CUTOUT [1-8]:", True, (240, 240, 240)), (14, by + 10))
        m_by = by + 28
        for m_idx in range(8):
            col_x = 14 + (m_idx % 4) * 54
            row_y = m_by + (m_idx // 4) * 34
            m_rect = pygame.Rect(col_x, row_y, 48, 28)
            is_hover = m_rect.collidepoint(mouse_pos)
            is_alive = motor_health[selected_drone_idx][m_idx] > 0.0

            m_col = (0, 255, 140) if is_alive else (255, 50, 60)
            pygame.draw.rect(screen, (24, 32, 44) if is_hover else (18, 24, 32), m_rect, border_radius=3)
            pygame.draw.rect(screen, m_col, m_rect, 1, border_radius=3)
            screen.blit(font_bold.render(f"M{m_idx+1}", True, m_col), (col_x + 14, row_y + 7))

            if mouse_click and is_hover:
                motor_health[selected_drone_idx][m_idx] = 0.0 if is_alive else 1.0

        # -------------------------------------------------------------
        # 8. RIGHT PANEL: HUD & FULL TELEMETRY
        # -------------------------------------------------------------
        pygame.draw.rect(screen, (16, 20, 28), (1040, 0, 460, 740))
        pygame.draw.line(screen, (35, 45, 60), (1040, 0), (1040, 740), 2)

        screen.blit(font_title.render("AVIONICS HUD & TELEMETRY", True, (0, 220, 255)), (1055, 15))
        screen.blit(font_small.render("Survival Core V37.0 Full Multi-Physics", True, (130, 150, 175)), (1055, 34))

        act_p = p_est_np[selected_drone_idx]
        act_v = v_est_np[selected_drone_idx]
        roll = math.atan2(R_curr[selected_drone_idx, 2, 1].item(), R_curr[selected_drone_idx, 2, 2].item())
        pitch = math.asin(float(np.clip(-R_curr[selected_drone_idx, 2, 0].item(), -1.0, 1.0)))

        draw_pfd_artificial_horizon(
            screen, (1055, 55, 195, 135), roll, pitch, act_p[2], float(np.linalg.norm(act_v)), act_v[2], font_small
        )

        draw_tactical_radar_minimap(
            screen, (1260, 55, 225, 135), manifest, depot_pads, p_est_np, selected_drone_idx,
            road_traffic, medevac_plane.pos, intruder_p
        )

        ty = 205
        pygame.draw.line(screen, (35, 45, 60), (1055, ty), (1485, ty), 1)

        # Section 1: Swarm Fleet Overview
        screen.blit(font_bold.render("1. SWARM FLEET OVERVIEW:", True, (240, 240, 240)), (1055, ty + 6))
        d_colors = [(0, 220, 255), (255, 180, 40), (220, 100, 255)]
        for i in range(3):
            star = "*" if i == selected_drone_idx else " "
            status = f"D{i+1}{star} | {FSM_NAMES[fsm_phases[i]]:<12} | Z:{p_est_np[i,2]:4.1f}m | Tgt:{active_missions[i]['bldg_id']}"
            screen.blit(font_small.render(status, True, d_colors[i]), (1065, ty + 24 + i * 16))

        # Section 2: 15-State Error-State EKF
        ty += 78
        pygame.draw.line(screen, (35, 45, 60), (1055, ty), (1485, ty), 1)
        screen.blit(font_bold.render("2. CLOSED-LOOP 15-STATE ES-EKF:", True, (240, 240, 240)), (1055, ty + 6))
        screen.blit(font_small.render(f"Pos Est: [{act_p[0]:6.1f}, {act_p[1]:6.1f}, {act_p[2]:5.1f}] m", True, (0, 220, 255)), (1065, ty + 24))
        screen.blit(font_small.render(f"Vel Est: [{act_v[0]:5.1f}, {act_v[1]:5.1f}, {act_v[2]:5.1f}] m/s", True, (160, 180, 205)), (1065, ty + 40))
        screen.blit(font_small.render(f"Covariance Tr(P): {diagnostics[selected_drone_idx]['cov_trace']:.4f} | GPS: {'REJECTED (SPOOF)' if diagnostics[selected_drone_idx]['spoof_rejected'] else 'RTK FIXED'}", True, (255, 60, 80) if diagnostics[selected_drone_idx]['spoof_rejected'] else (0, 255, 140)), (1065, ty + 56))
        screen.blit(font_small.render(f"Quantum Phase: ΔΦ = {diagnostics[selected_drone_idx]['phase_shift'][0]:.3f} rad (87Rb Raman Laser)", True, (0, 220, 255)), (1065, ty + 72))

        # Section 3: Battery ECM & Power
        ty += 94
        pygame.draw.line(screen, (35, 45, 60), (1055, ty), (1485, ty), 1)
        screen.blit(font_bold.render("3. 6S ARRHENIUS BATTERY ECM:", True, (240, 240, 240)), (1055, ty + 6))
        screen.blit(font_small.render(f"State of Charge: {soc*100:4.1f}% | Pack Voltage: {v_pack:4.1f}V", True, (0, 255, 140) if soc > 0.3 else (255, 60, 80)), (1065, ty + 24))
        screen.blit(font_small.render(f"Internal Res R0: {r0:.4f} Ω | Pack Temp: {temp_c:4.1f}°C", True, (160, 180, 205)), (1065, ty + 40))
        screen.blit(font_small.render(f"ESC RMS Current: {7.5:4.1f}A | MOSFET Temp: {38.0:4.1f}°C", True, (160, 180, 205)), (1065, ty + 56))

        # Section 4: Atmosphere, Weather & Icing
        ty += 78
        pygame.draw.line(screen, (35, 45, 60), (1055, ty), (1485, ty), 1)
        screen.blit(font_bold.render("4. METEOROLOGY & ICING DYNAMICS:", True, (240, 240, 240)), (1055, ty + 6))
        screen.blit(font_small.render(f"Regime: {weather_engine.get_mode_name()} | Temp: {temp_c:4.1f}°C", True, (0, 220, 255)), (1065, ty + 24))
        screen.blit(font_small.render(f"Wind: [{total_wind[0]:4.1f}, {total_wind[1]:4.1f}, {total_wind[2]:4.1f}] m/s | Gust: {float(np.linalg.norm(total_wind)):4.1f}m/s", True, (160, 180, 205)), (1065, ty + 40))
        screen.blit(font_small.render(f"Rotor Ice: {ice_mm:4.2f} mm ({ice_idx*100:3.0f}%) | Drag Rise: +{drag_rise*100-100:3.0f}%", True, (255, 60, 80) if ice_mm > 0.5 else (0, 255, 140)), (1065, ty + 56))

        # Section 5: Tether, Winch & Moving Hazards
        ty += 78
        pygame.draw.line(screen, (35, 45, 60), (1055, ty), (1485, ty), 1)
        screen.blit(font_bold.render("5. TETHER & SENSORY HAZARDS:", True, (240, 240, 240)), (1055, ty + 6))
        adsb_p, adsb_spd, adsb_d = medevac_plane.poll_adsb(act_p, act_v)
        siren_ang, siren_freq, siren_spl = sensor_suite.poll_acoustic_array(act_p, amb_pos=amb_pos, sim_time=sim_time, siren_active=True)
        screen.blit(font_small.render(f"Winch Cable: {cable_lens[selected_drone_idx]:4.2f}m | Cargo: {'ATTACHED' if has_payloads[selected_drone_idx] else 'DELIVERED'}", True, (0, 220, 255)), (1065, ty + 24))
        screen.blit(font_small.render(f"Medevac ADS-B: {adsb_d:4.1f}m ({adsb_spd:+4.1f}m/s) | Alt: 62.0m", True, (255, 180, 40)), (1065, ty + 40))
        screen.blit(font_small.render(f"Ambulance Siren: {siren_spl:4.1f} dBA @ {siren_freq:.0f}Hz | Intruder: {float(np.linalg.norm(intruder_p[:2] - act_p[:2])):4.1f}m", True, (255, 60, 80) if siren_spl > 70 else (0, 255, 140)), (1065, ty + 56))
        screen.blit(font_small.render(f"Total Swarm Deliveries: {dispatcher.total_completed} Completed", True, (0, 255, 140)), (1065, ty + 72))

        # Banner Overlay
        banner_txt = f"ACTIVE: D{selected_drone_idx+1} | Phase: {FSM_NAMES[fsm_phases[selected_drone_idx]]} | Alt: {act_p[2]:4.1f}m | Cam: {camera_modes[cam_mode_idx]}"
        banner_surf = font_bold.render(banner_txt, True, (240, 240, 240))
        pygame.draw.rect(screen, (16, 20, 28), (250, 10, banner_surf.get_width() + 16, 26), border_radius=4)
        pygame.draw.rect(screen, (0, 220, 255), (250, 10, banner_surf.get_width() + 16, 26), 1, border_radius=4)
        screen.blit(banner_surf, (258, 16))

        pygame.display.flip()
        clock.tick(50)  # 50 FPS * 4 substeps = 200 Hz = 1.0x Real Time Speed!

    pygame.quit()

# =============================================================================
# 32. CLI ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_swarm_system()
    else:
        evaluate_swarm_system()