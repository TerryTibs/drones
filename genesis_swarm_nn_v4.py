"""
SURVIVAL CORE V37.0 — FULL GENESIS MULTI-PHYSICS UPGRADE
PART 1 OF 5: 3D COAXIAL X8 DRONE URDF GENERATOR & 3D URBAN CORRIDOR PLANNER
"""

import os
import math
import numpy as np
import torch
import genesis as gs

# =============================================================================
# 0. AIRFRAME & CORRIDOR CONSTANTS
# =============================================================================
DRY_MASS_KG = 1.35
CARGO_MASS_KG = 0.45
TOTAL_MASS_KG = DRY_MASS_KG + CARGO_MASS_KG
GRAVITY_VAL = 9.80665

ARM_LENGTH_M = 0.26 / math.sqrt(2.0)
ROTOR_RADIUS_M = 0.127
ROTOR_DISK_AREA = math.pi * (ROTOR_RADIUS_M ** 2)

# Urban corridor safety bounds: Tallest skyscraper is 26m + 2m HVAC clutter = 28m
MAX_CITY_OBSTACLE_HEIGHT_M = 28.0
MIN_SAFE_CRUISE_ALT_M = MAX_CITY_OBSTACLE_HEIGHT_M + 6.0  # Safe corridor floor = 34.0m
GLOBAL_CEILING_ALT_M = 55.0

# =============================================================================
# 1. PROCEDURAL COAXIAL X8 DRONE URDF GENERATOR
# =============================================================================
def generate_coaxial_x8_urdf(filepath: str = "drone_coaxial_x8.urdf"):
    """
    Generates a realistic Coaxial X8 Multicopter URDF file with:
      - Central avionics fuselage
      - 4 carbon-fiber tubular arms (X-configuration)
      - 8 coaxial rotor pods/disks (4 top, 4 bottom)
      - Dual landing gear skids
    """
    arm_l = ARM_LENGTH_M
    arm_angle = math.pi / 4.0  # 45 degrees
    dx = arm_l * math.cos(arm_angle)
    dy = arm_l * math.sin(arm_angle)

    urdf_content = f"""<?xml version="1.0"?>
<robot name="coaxial_x8_drone">

  <!-- Central Avionics Fuselage -->
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{DRY_MASS_KG}"/>
      <inertia ixx="0.022" ixy="0" ixz="0" iyy="0.022" iyz="0" izz="0.038"/>
    </inertial>
    
    <!-- Central Fuselage Body -->
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.22 0.18 0.08"/>
      </geometry>
      <material name="carbon_dark">
        <color rgba="0.12 0.14 0.18 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.24 0.20 0.09"/>
      </geometry>
    </collision>

    <!-- Top GPS / Avionics Dome -->
    <visual>
      <origin xyz="0 0 0.055" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.045" length="0.03"/>
      </geometry>
      <material name="cyan_accent">
        <color rgba="0.0 0.85 0.95 1.0"/>
      </material>
    </visual>

    <!-- Arm 1 (Front-Right: +X, +Y) -->
    <visual>
      <origin xyz="{dx/2} {dy/2} 0.01" rpy="0 0 {arm_angle}"/>
      <geometry>
        <cylinder radius="0.012" length="{arm_l}"/>
      </geometry>
      <material name="carbon_arm"><color rgba="0.2 0.22 0.25 1.0"/></material>
    </visual>
    <!-- Arm 2 (Front-Left: -X, +Y) -->
    <visual>
      <origin xyz="{-dx/2} {dy/2} 0.01" rpy="0 0 {-arm_angle}"/>
      <geometry>
        <cylinder radius="0.012" length="{arm_l}"/>
      </geometry>
      <material name="carbon_arm"><color rgba="0.2 0.22 0.25 1.0"/></material>
    </visual>
    <!-- Arm 3 (Rear-Left: -X, -Y) -->
    <visual>
      <origin xyz="{-dx/2} {-dy/2} 0.01" rpy="0 0 {arm_angle}"/>
      <geometry>
        <cylinder radius="0.012" length="{arm_l}"/>
      </geometry>
      <material name="carbon_arm"><color rgba="0.2 0.22 0.25 1.0"/></material>
    </visual>
    <!-- Arm 4 (Rear-Right: +X, -Y) -->
    <visual>
      <origin xyz="{dx/2} {-dy/2} 0.01" rpy="0 0 {-arm_angle}"/>
      <geometry>
        <cylinder radius="0.012" length="{arm_l}"/>
      </geometry>
      <material name="carbon_arm"><color rgba="0.2 0.22 0.25 1.0"/></material>
    </visual>

    <!-- Rotor Pods (4 Top, 4 Bottom) -->
    <!-- Front-Right Top / Bottom -->
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

    <!-- Front-Left Top / Bottom -->
    <visual>
      <origin xyz="{-dx} {dy} 0.035" rpy="0 0 0"/>
      <geometry><cylinder radius="{ROTOR_RADIUS_M}" length="0.008"/></geometry>
      <material name="rotor_orange"><color rgba="0.95 0.55 0.1 0.7"/></material>
    </visual>
    <visual>
      <origin xyz="{-dx} {dy} -0.02" rpy="0 0 0"/>
      <geometry><cylinder radius="{ROTOR_RADIUS_M}" length="0.008"/></geometry>
      <material name="rotor_dark"><color rgba="0.1 0.1 0.1 0.7"/></material>
    </visual>

    <!-- Rear-Left Top / Bottom -->
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

    <!-- Rear-Right Top / Bottom -->
    <visual>
      <origin xyz="{dx} {-dy} 0.035" rpy="0 0 0"/>
      <geometry><cylinder radius="{ROTOR_RADIUS_M}" length="0.008"/></geometry>
      <material name="rotor_cyan"><color rgba="0.0 0.8 0.9 0.7"/></material>
    </visual>
    <visual>
      <origin xyz="{dx} {-dy} -0.02" rpy="0 0 0"/>
      <geometry><cylinder radius="{ROTOR_RADIUS_M}" length="0.008"/></geometry>
      <material name="rotor_dark"><color rgba="0.1 0.1 0.1 0.7"/></material>
    </visual>

    <!-- Landing Skids Underneath -->
    <visual>
      <origin xyz="0 0.08 -0.065" rpy="1.5708 0 0"/>
      <geometry><cylinder radius="0.007" length="0.22"/></geometry>
      <material name="skid_mat"><color rgba="0.3 0.35 0.4 1.0"/></material>
    </visual>
    <visual>
      <origin xyz="0 -0.08 -0.065" rpy="1.5708 0 0"/>
      <geometry><cylinder radius="0.007" length="0.22"/></geometry>
      <material name="skid_mat"><color rgba="0.3 0.35 0.4 1.0"/></material>
    </visual>
  </link>

</robot>
"""
    with open(filepath, "w") as f:
        f.write(urdf_content)
    print(f"[ASSET] Coaxial X8 Drone URDF generated: {filepath}")

# =============================================================================
# 2. 3D GLIDE-SLOPE CORRIDOR & URBAN VERTICAL CHIMNEY PLANNER
# =============================================================================
class VoxelOctreeCorridorPlanner:
    """
    Prevents building collisions by enforcing:
      1. Vertical Chimney Ascent: Drone must climb above local building clearance
         before any horizontal translation begins.
      2. Global Safe Cruise Floor: Flight corridor is strictly planned above
         all intermediate skyscrapers along the trajectory.
      3. Vertical Chimney Descent: Straight descent gate directly over the helipad.
    """
    def __init__(self, buildings_list, base_station_pad):
        self.buildings = buildings_list
        self.base_pad = base_station_pad

    def get_max_elevation_along_corridor(self, p_start, p_end, buffer_width=12.0):
        """Finds the maximum obstacle elevation between two coordinates."""
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
                if perp_dist <= (b.get("radius", 8.0) + buffer_width):
                    max_h = max(max_h, bh)
        return max_h

    def plan_mission_waypoints(self, start_pos, target_dest, is_return=False):
        """
        Generates a 6-waypoint safe corridor:
          WP0: Current position
          WP1: Vertical Takeoff Gate (climb chimney)
          WP2: Cruise Entry Gate (above intermediate skyscrapers)
          WP3: Cruise Exit Gate (above target pad)
          WP4: Chimney Descent Gate (holding altitude above target)
          WP5: Precision Hover Lock (touchdown delivery altitude)
        """
        start_xy = start_pos[:2]
        target_xy = target_dest["pos"][:2]
        target_roof_h = float(target_dest["height"]) if not is_return else 0.0

        # 1. Calculate corridor clearance altitude
        highest_obstacle = self.get_max_elevation_along_corridor(start_pos, target_dest["pos"])
        safe_cruise_alt = max(MIN_SAFE_CRUISE_ALT_M, highest_obstacle + 6.0)

        # 2. Chimney gates
        wp1_climb_gate = np.array([start_xy[0], start_xy[1], safe_cruise_alt], dtype=np.float32)
        wp2_cruise_mid = np.array([0.5 * (start_xy[0] + target_xy[0]), 
                                   0.5 * (start_xy[1] + target_xy[1]), 
                                   safe_cruise_alt], dtype=np.float32)
        wp3_over_target = np.array([target_xy[0], target_xy[1], safe_cruise_alt], dtype=np.float32)
        
        # 3. Descent chimney
        descent_hover_z = target_roof_h + 7.5 if not is_return else 0.70
        wp4_chimney_gate = np.array([target_xy[0], target_xy[1], descent_hover_z], dtype=np.float32)

        return [wp1_climb_gate, wp2_cruise_mid, wp3_over_target, wp4_chimney_gate], safe_cruise_alt

    """
SURVIVAL CORE V37.0 — FULL GENESIS MULTI-PHYSICS UPGRADE
PART 2 OF 5: MICROCLIMATE MAP, URBAN WEATHER ENGINE, DRYDEN TURBULENCE,
             MESSINGER ROTOR ICING & SUB-ZERO 6S ARRHENIUS BATTERY ECM
"""

import math
import random
import numpy as np
import torch

# =============================================================================
# 0. BATTERY & ATMOSPHERIC PHYSICAL CONSTANTS
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
SPEED_OF_SOUND = 343.2

# =============================================================================
# 1. SWARM MICROCLIMATE SHARED VOXEL MAP
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
# 2. DYNAMIC URBAN WEATHER ENGINE
# =============================================================================
class DynamicUrbanWeatherEngine:
    """
    Simulates dynamic urban meteorology with 4 regimes:
      - CLEAR_NOMINAL: Calm breeze (1.8 m/s), 22°C
      - CANYON_GALE: Urban street canyon winds (6.5 m/s), 14°C
      - FREEZING_RAIN: Supercooled precipitation with rotor icing, -4.5°C
      - MICROBURST_STORM: Severe downdraft shear (-3.2 m/s vertical), 10°C
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
        print(f"\n>>> [WEATHER TRANSITION] Blending into: {self.weather_modes[self.target_mode_idx]} <<<")

    def get_mode_name(self):
        return self.weather_modes[self.target_mode_idx]

    def step(self, dt=0.01):
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
# 3. DRYDEN ATMOSPHERIC TURBULENCE MODEL (MIL-F-8785C)
# =============================================================================
class DrydenAtmosphericTurbulenceModel:
    """
    Stochastic atmospheric turbulence generator using low-altitude MIL-F-8785C
    spectral formulation for realistic wind gust dynamics.
    """
    def __init__(self, w20_m_s=3.4):
        self.w20 = w20_m_s
        self.gust_state = np.zeros(3, dtype=np.float32)

    def step(self, altitude_m, airspeed_m_s, dt=0.01):
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
# 4. MESSINGER ROTOR ICING DYNAMICS MODEL
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

    def step(self, airspeed, dt=0.01):
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
# 5. SUB-ZERO ARRHENIUS 6S BATTERY EQUIVALENT CIRCUIT MODEL (ECM)
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

    def step(self, p_elec_watts, amb_temp_c=22.0, dt=0.01):
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

    """
SURVIVAL CORE V37.0 — FULL GENESIS MULTI-PHYSICS UPGRADE
PART 3 OF 5: COLD-ATOM QUANTUM GRAVIMETER, 15-STATE ERROR-STATE EKF ON SO(3),
             FULL MULTI-MODAL SENSOR SUITE & EXACT 139-DIM SYNTHESIZER
"""

import math
import numpy as np
import torch

# =============================================================================
# 0. SENSORY PHYSICAL CONSTANTS
# =============================================================================
K_EFF_WAVEVECTOR = 1.61e7            # 87Rb Raman laser effective wavevector (1/m)
RAMAN_PULSE_SEP_T = 0.015            # Time separation between Raman pulses (s)
QUANTUM_PHASE_1G = K_EFF_WAVEVECTOR * 9.80665 * (RAMAN_PULSE_SEP_T ** 2)

BASE_OBS_DIM = 139
HISTORY_LEN = 3
OBSERVATION_DIM = BASE_OBS_DIM * HISTORY_LEN

# =============================================================================
# 1. LIE ALGEBRA UTILITIES FOR ERROR-STATE EKF
# =============================================================================
def skew_symmetric(v):
    return np.array([
        [0.0, -v[2], v[1]], 
        [v[2], 0.0, -v[0]], 
        [-v[1], v[0], 0.0]
    ], dtype=np.float32)

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

# =============================================================================
# 2. MATTER-WAVE QUANTUM SENSOR & 15-STATE ERROR-STATE EKF ON SO(3)
# =============================================================================
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
    15-State Error-State Extended Kalman Filter on SO(3):
      States: [delta_p (3), delta_v (3), delta_theta (3), delta_b_a (3), delta_b_g (3)]
    Features:
      - Tikhonov-regularized innovation solves (immune to matrix singularity)
      - Sage-Husa adaptive process noise inflation
      - Optical flow velocity adaptive fusion with slant-range weighting
      - Barometric altitude & RTK-GNSS spoofing rejection gating
    """
    def __init__(self, init_pos):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.b_a = np.zeros(3, dtype=np.float32)
        self.b_g = np.zeros(3, dtype=np.float32)
        self.P = np.diag([0.15]*3 + [0.1]*3 + [0.02]*3 + [0.05]*3 + [0.01]*3).astype(np.float32)
        self.Q_base = np.diag([1e-4]*3 + [1e-3]*3 + [1e-4]*3 + [1e-5]*3 + [1e-6]*3).astype(np.float32)
        self.Q_d = self.Q_base.copy()
        self.residual_scale = 1.0

    def predict(self, accel_body_raw, gyro_raw, dt=0.01):
        a_unbiased = np.clip(accel_body_raw - self.b_a, -30.0, 30.0)
        w_unbiased = np.clip(gyro_raw - self.b_g, -12.0, 12.0)
        R = quat_to_rot(self.q)
        f_inertial = R @ a_unbiased + np.array([0, 0, -9.80665], dtype=np.float32)

        self.p += self.v * dt + 0.5 * f_inertial * (dt ** 2)
        self.v += f_inertial * dt
        self.v = np.clip(self.v, -15.0, 15.0)

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

        self.Q_d = self.Q_base * max(1.0, float(self.residual_scale))
        self.P = F_x @ self.P @ F_x.T + self.Q_d
        self.P = np.nan_to_num(self.P, nan=0.05)
        self.P = 0.5 * (self.P + self.P.T)

        max_diag = np.array([100.0]*3 + [50.0]*3 + [0.5]*3 + [1.0]*3 + [0.2]*3, dtype=np.float32)
        diag_P = np.diag(self.P)
        for k in range(15):
            if diag_P[k] > max_diag[k]:
                scale = math.sqrt(max_diag[k] / diag_P[k])
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
            return

        d_k = 0.05
        expected_gamma = float(len(innov_flat))
        self.residual_scale = (1.0 - d_k) * self.residual_scale + d_k * (gamma / max(1.0, expected_gamma))
        self.residual_scale = float(np.clip(self.residual_scale, 0.8, 5.0))

        if gamma > chi2_threshold:
            return  # Reject anomalous measurement

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

    def update_heading(self, measured_yaw_rad, r_cov=0.02):
        R_curr = quat_to_rot(self.q)
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

    def update_baro(self, z_baro, q_dyn=0.0):
        r_adaptive = float(np.clip(0.05 + 0.0022 * q_dyn, 0.05, 0.40))
        H_baro = np.zeros((1, 15), dtype=np.float32)
        H_baro[0, 2] = 1.0
        self.update_measurement(H_baro, np.array([z_baro - self.p[2]], dtype=np.float32), np.array([[r_adaptive]], dtype=np.float32))

    def update_gps(self, p_gps, hdop=0.85, cn0_avg=44.0, is_spoofed=False):
        r_adaptive = 45.0 if is_spoofed else 0.04 * (hdop ** 2) * (44.0 / max(15.0, cn0_avg))
        H_pos = np.zeros((3, 15), dtype=np.float32)
        H_pos[0:3, 0:3] = np.eye(3, dtype=np.float32)
        self.update_measurement(H_pos, p_gps - self.p, np.eye(3, dtype=np.float32) * float(np.clip(r_adaptive, 0.03, 50.0)))

# =============================================================================
# 3. ADVANCED MULTI-MODAL PHYSICAL SENSOR SUITE
# =============================================================================
class AdvancedMultiModalSensorSuite:
    """
    Emulates the comprehensive avionics sensory reality:
      - Optical Flow & Camera Slant Range
      - Downward ArUco Helipad Marker Detector
      - Emergency Siren Directional Acoustic Array
      - ADS-B Civil Aircraft Transponder
      - Smart Hook IMU & 3-Axis Hook Load Cell
      - Thermal IR Roof Heat & Polarization Sensing
      - ESC / FOC Motor Current & Temperature
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

    def poll_acoustic_array(self, drone_p, sim_time=0.0, siren_active=False):
        if siren_active:
            amb_pos = np.array([-20.0, 15.0, 1.0], dtype=np.float32)
            d = float(np.linalg.norm(drone_p[:2] - amb_pos[:2]))
            rel = amb_pos[:2] - drone_p[:2]
            bearing = math.atan2(rel[1], rel[0])
            spl_dba = float(np.clip(102.0 - 20.0 * math.log10(max(1.0, d)), 45.0, 98.0))
            freq = 850.0 + 150.0 * math.sin(sim_time * 3.0)
            return bearing, freq, spl_dba
        return 0.0, 180.0, 52.0

    def poll_adsb_transponder(self, drone_p, drone_v):
        medevac_pos = np.array([-120.0, 20.0, 58.0], dtype=np.float32)
        medevac_vel = np.array([12.5, 0.0, 0.0], dtype=np.float32)
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
        park_dist = math.hypot(px - (-45.0), py - (-45.0))
        th_dt = float(roof_heat - 2.5 * math.exp(-park_dist / 20.0) + 0.5)
        dolp = float(0.12 + 0.55 * math.exp(-min_roof_dist / 6.0))
        ndvi = float(0.10 + 0.68 * math.exp(-park_dist / 25.0))
        return th_dt, dolp, ndvi

    def poll_esc_telemetry(self, motor_omegas):
        i_rms = np.zeros(8, dtype=np.float32)
        t_mosfet = np.zeros(8, dtype=np.float32)
        max_om = (9200.0 * 2.0 * math.pi) / 60.0
        for i in range(8):
            p_ratio = (motor_omegas[i] / max_om) ** 2
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
# 4. FULL PHYSICS SENSOR & MEMS IMU NOISE SIMULATOR
# =============================================================================
class FullPhysicsSensorSuite:
    def __init__(self):
        self.powerline_y = 15.0
        self.powerline_z = 12.0

    def poll(self, drone_p, drone_v, wind_3d, surface_z=0.0):
        z_true = max(0.0, float(drone_p[2]))
        t_air = 288.15 - 0.0065 * z_true
        p_static_hpa = 1013.25 * ((1.0 - (0.0065 * z_true) / 288.15) ** 5.25588)
        rho_air = (p_static_hpa * 100.0) / (287.058 * t_air)
        z_baro = 44330.8 * (1.0 - ((p_static_hpa / 1013.25) ** 0.190284)) + float(np.random.normal(0, 0.04))

        v_air = drone_v - wind_3d
        v_tas = float(np.linalg.norm(v_air))
        q_dyn = 0.5 * rho_air * (v_tas ** 2)

        d_wire = math.hypot(drone_p[1] - self.powerline_y, drone_p[2] - self.powerline_z)
        emf_rms = float(140.0 / max(0.25, d_wire))
        clearance = max(0.05, float(drone_p[2] - surface_z))
        min_sonar = max(0.05, min(15.0, clearance))

        return {"z_baro": z_baro, "rho_air": rho_air, "v_tas": v_tas, "q_dyn": q_dyn, "emf_rms": emf_rms, "min_sonar": min_sonar}

class Airframe6DOFMEMS_IMU:
    def __init__(self):
        self.b_a = np.array([0.02, -0.015, 0.03], dtype=np.float32)
        self.b_g = np.array([0.005, -0.004, 0.002], dtype=np.float32)

    def poll(self, drone_R, accel_linear, omega_true, motor_omegas):
        g_vec = np.array([0, 0, -9.80665], dtype=np.float32)
        f_body = drone_R.T @ (accel_linear - g_vec)
        rpm_ratio = float(np.mean(motor_omegas)) / 963.4
        vib_a = np.random.normal(0.0, 0.16 * rpm_ratio, 3).astype(np.float32)
        vib_g = np.random.normal(0.0, 0.015 * rpm_ratio, 3).astype(np.float32)

        accel_raw = f_body + self.b_a + vib_a + np.random.normal(0.0, 0.035, 3).astype(np.float32)
        gyro_raw = omega_true + self.b_g + vib_g + np.random.normal(0.0, 0.004, 3).astype(np.float32)
        return accel_raw, gyro_raw

# =============================================================================
# 5. STRICT 139-DIMENSIONAL OBSERVATION SYNTHESIZER & TEMPORAL STACKER
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
        out[56] = sway_2d[0] / 2.0
        out[57] = sway_2d[1] / 2.0
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

    """
SURVIVAL CORE V37.0 — FULL GENESIS MULTI-PHYSICS UPGRADE
PART 4 OF 5: RECIPROCAL SWARM CBF-QP SAFETY FILTER, TACTILE REFLEX,
             RATE-DAMPED SE(3) ACTUATOR MIXER & 6-HEAD RESNET BRAIN
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 0. SAFETY & ACTUATION CONSTANTS
# =============================================================================
CBF_ALPHA_OBSTACLE = 1.8
CBF_BASE_SAFE_RADIUS_M = 0.85
SWARM_INTER_DRONE_SAFE_RADIUS_M = 2.6
DOWNWASH_CONE_EXPANSION = 0.38
DOWNWASH_HAZARD_DEPTH_M = 7.5

ARM_LENGTH_M = 0.26 / math.sqrt(2.0)
C_T_CLEAN = 1.35e-5
C_Q_CLEAN = 2.40e-7
COAXIAL_EFFICIENCY = 0.92
MAX_RPM = 9200.0
MAX_OMEGA = (MAX_RPM * 2.0 * math.pi) / 60.0
G_ACCEL = 9.80665

BASE_OBS_DIM = 139
HISTORY_LEN = 3
OBSERVATION_DIM = BASE_OBS_DIM * HISTORY_LEN
NUM_EXECUTIVE_STATES = 7

# =============================================================================
# 1. RECIPROCAL SWARM CBF-QP SAFETY FILTER
# =============================================================================
class ReciprocalSwarmCBFQPFilter:
    """
    Control Barrier Function Quadratic Programming (CBF-QP) Filter.
    Guarantees safety by projecting nominal flight accelerations onto
    a forward-invariant safe set:
      - Building Wall & Rooftop avoidance (with EKF uncertainty inflation)
      - Swarm Downwash Cone avoidance (prevents flying into peer rotor wake)
      - Ground cushion and urban geofence enforcement
    """
    def __init__(self):
        self.alpha_obs = CBF_ALPHA_OBSTACLE
        self.safe_radius = CBF_BASE_SAFE_RADIUS_M
        self.swarm_r = SWARM_INTER_DRONE_SAFE_RADIUS_M
        self.downwash_expansion = DOWNWASH_CONE_EXPANSION
        self.downwash_depth = DOWNWASH_HAZARD_DEPTH_M

    def enforce_safety(self, drone_idx, drone_p, drone_v, nominal_accel, obstacles, swarm_states=None, 
                       target_bldg_id=None, departure_bldg_id=None, wind_2d=None, ekf_pos_cov_trace=0.05):
        A_rows = []
        b_rows = []

        if wind_2d is None:
            wind_2d = np.zeros(2, dtype=np.float32)

        # Dynamic uncertainty inflation based on EKF covariance trace & wind gustiness
        cov_inflation = 1.2 * min(2.5, math.sqrt(max(0.0, float(ekf_pos_cov_trace))))
        gust_inflation = 0.22 * float(np.linalg.norm(wind_2d))
        dynamic_safe_radius = self.safe_radius + cov_inflation + gust_inflation

        # 1. Building Obstacle Barriers
        for obs in obstacles:
            obs_id = obs.get("id")
            obs_h = obs.get("height", 0.0)
            if obs_id in [target_bldg_id, departure_bldg_id]:
                continue  # Target pads handled by precision descent
            
            rel_pos = drone_p - obs["pos"]
            dist_xy = math.hypot(rel_pos[0], rel_pos[1])

            n_xy = rel_pos[:2] / max(1e-4, dist_xy)
            v_closing = -float(np.dot(drone_v[:2], n_xy))
            tailwind_proj = -float(np.dot(wind_2d[:2], n_xy))
            
            dynamic_stopping_buffer = (max(0.0, v_closing) ** 2) / (2.0 * 3.8) + max(0.0, tailwind_proj) * 0.35
            effective_safe_r = dynamic_safe_radius + min(4.5, dynamic_stopping_buffer)
            h_obs = dist_xy - (obs.get("radius", 8.0) + effective_safe_r)
            eff_alpha = self.alpha_obs * (8.0 / (8.0 + max(0.0, v_closing)))

            # Wall barrier
            if drone_p[2] <= (obs_h - 0.4) and h_obs < 4.5:
                grad_h = np.array([n_xy[0], n_xy[1], 0.0], dtype=np.float32)
                v_tangent = drone_v[0] * (-n_xy[1]) + drone_v[1] * n_xy[0]
                centripetal = (v_tangent ** 2) / max(0.5, dist_xy)
                l_fh = float(np.dot(grad_h, drone_v))
                
                A_rows.append(grad_h)
                b_rows.append(-l_fh - eff_alpha * h_obs - centripetal)

            # Rooftop barrier
            if dist_xy <= (obs.get("radius", 8.0) + 0.5) and drone_p[2] >= (obs_h - 0.2):
                h_roof = drone_p[2] - (obs_h + 0.15)
                if h_roof < 0.8:
                    A_rows.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
                    b_rows.append(-drone_v[2] - 2.0 * h_roof)

        # 2. Inter-Drone Swarm & Downwash Wake Barriers
        if swarm_states:
            for j, other in enumerate(swarm_states):
                if j == drone_idx: continue
                p_other, v_other = other["p"], other["v"]

                if drone_p[2] < 6.0 and p_other[2] < 6.0:
                    continue  # Low altitude base maneuvers

                dz = drone_p[2] - p_other[2]
                d_p = drone_p[:2] - other["p"][:2]
                dist_ij = float(np.linalg.norm(d_p))

                # Downwash cone hazard below peer drone
                if -self.downwash_depth < dz < -0.4:
                    downwash_cone_r = self.swarm_r + self.downwash_expansion * abs(dz) + cov_inflation
                    h_downwash = dist_ij - downwash_cone_r
                    if h_downwash < 3.2:
                        n_downwash = np.array([d_p[0]/max(1e-4, dist_ij), d_p[1]/max(1e-4, dist_ij), 0.0], dtype=np.float32)
                        v_rel = drone_v[:2] - v_other[:2]
                        l_fh_downwash = float(np.dot(n_downwash[:2], v_rel))
                        A_rows.append(n_downwash)
                        b_rows.append(-l_fh_downwash - 2.8 * h_downwash)

                # Horizontal separation barrier
                if abs(dz) < 2.8:
                    h_swarm = dist_ij - (self.swarm_r + cov_inflation)
                    if h_swarm < 3.5:
                        n_ij = np.array([d_p[0]/max(1e-4, dist_ij), d_p[1]/max(1e-4, dist_ij), 0.0], dtype=np.float32)
                        v_rel = drone_v[:2] - v_other[:2]
                        l_fh_swarm = float(np.dot(n_ij[:2], v_rel))
                        A_rows.append(n_ij)
                        b_rows.append(-l_fh_swarm - 2.2 * h_swarm)

        # 3. Ground Cushion Barrier
        h_gnd = drone_p[2] - 0.20
        if h_gnd < 1.6 and drone_p[2] > 0.25:
            A_rows.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
            b_rows.append(-3.5 * drone_v[2] - 4.5 * max(0.05, h_gnd))

        # 4. Urban Geofence Boundary Barrier
        fence_limit = 180.0
        for axis in [0, 1]:
            if drone_p[axis] > (fence_limit - 15.0):
                dist_fence = max(0.1, fence_limit - drone_p[axis])
                grad = np.array([-1.0 if axis == 0 else 0.0, -1.0 if axis == 1 else 0.0, 0.0], dtype=np.float32)
                A_rows.append(grad)
                b_rows.append(drone_v[axis] - 2.5 * dist_fence)
            elif drone_p[axis] < (-fence_limit + 15.0):
                dist_fence = max(0.1, drone_p[axis] - (-fence_limit))
                grad = np.array([1.0 if axis == 0 else 0.0, 1.0 if axis == 1 else 0.0, 0.0], dtype=np.float32)
                A_rows.append(grad)
                b_rows.append(-drone_v[axis] - 2.5 * dist_fence)

        if not A_rows:
            return nominal_accel

        # Iterative QP Projection Loop
        A = np.array(A_rows, dtype=np.float32)
        b = np.array(b_rows, dtype=np.float32)
        u = nominal_accel.copy()

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

        return u

# =============================================================================
# 2. TACTILE LANDING REFLEX & ROTOR INFLOW HEALTH ESTIMATOR
# =============================================================================
class TactileLandingReflex:
    """Detects dual-skid strain gauge asymmetry, applies roll-leveling torque, and triggers touchdown."""
    def __init__(self):
        self.contact_timer = 0.0
        self.is_latched = False

    def reset(self):
        self.contact_timer = 0.0
        self.is_latched = False

    def compute_reflex(self, f_skid_left, f_skid_right, current_mass, dt=0.01):
        total_force = max(0.0, f_skid_left + f_skid_right)
        asymmetry = f_skid_left - f_skid_right
        total_weight = current_mass * G_ACCEL

        if total_force > 1.2:
            self.contact_timer += dt
        else:
            self.contact_timer = max(0.0, self.contact_timer - dt * 2.0)

        tau_roll_adj = -0.035 * asymmetry
        load_fraction = float(np.clip(total_force / (0.85 * total_weight), 0.0, 1.0))
        thrust_attenuation = 1.0 - 0.92 * (load_fraction ** 1.3)

        if self.contact_timer >= 0.18 and load_fraction >= 0.70:
            self.is_latched = True
            thrust_attenuation = 0.0

        return thrust_attenuation, tau_roll_adj, self.is_latched

class OnlineRotorInflowAndHealthEstimator:
    """Estimates real-time rotor thrust coefficient degradation due to damage or icing."""
    def __init__(self, nominal_ct=C_T_CLEAN):
        self.nominal_ct = nominal_ct
        self.ct_estimated = nominal_ct
        self.aerodynamic_health = 1.0

    def update(self, current_mass, accel_z, motor_omegas, dt=0.01):
        sum_sq = float(np.sum(motor_omegas ** 2))
        if sum_sq > 1e4:
            t_req = current_mass * float(max(1.0, accel_z + G_ACCEL))
            instant_ct = t_req / sum_sq
            self.ct_estimated += 0.01 * (instant_ct - self.ct_estimated)
            self.ct_estimated = float(np.clip(self.ct_estimated, self.nominal_ct * 0.55, self.nominal_ct * 1.35))
            self.aerodynamic_health = float(np.clip(self.ct_estimated / self.nominal_ct, 0.55, 1.0))
        return self.ct_estimated, self.aerodynamic_health

# =============================================================================
# 3. RATE-DAMPED ACTUATOR MIXER WITH CBF & SE(3) TRACKING
# =============================================================================
class RateDampedActuatorMixerWithCBF:
    """
    Translates reference guidance into low-pass filtered attitude commands on SO(3),
    applies the CBF safety filter, and allocates motor omegas via pseudo-inverse mixing.
    """
    def __init__(self):
        self.cbf_filter = ReciprocalSwarmCBFQPFilter()
        self.tactile_reflex = TactileLandingReflex()
        self.inflow_estimator = OnlineRotorInflowAndHealthEstimator()

        self.wind_filt = np.zeros(3, dtype=np.float32)
        self.b3_d_prev = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.z_err_integral = 0.0
        self.xy_err_integral = np.zeros(2, dtype=np.float32)
        self._last_motor_omegas = np.full(8, 450.0, dtype=np.float32)

        L = ARM_LENGTH_M
        cq_ct_top = C_Q_CLEAN / C_T_CLEAN
        cq_ct_bot = (C_Q_CLEAN / C_T_CLEAN) / COAXIAL_EFFICIENCY
        self.B_nominal = np.array([
            [ 1.0,         1.0,        1.0,         1.0,        1.0,         1.0,        1.0,         1.0],
            [-L,          -L,          L,           L,          L,           L,         -L,          -L],
            [-L,          -L,         -L,          -L,          L,           L,          L,           L],
            [-cq_ct_top, cq_ct_bot, cq_ct_top, -cq_ct_bot, -cq_ct_top, cq_ct_bot, cq_ct_top, -cq_ct_bot]
        ], dtype=np.float32)

    def mix(self, drone_health, p_est, v_est, R_est, omega_est, p_target, neural_guidance, token_idx,
            estimated_wind, obstacles, swarm_states=None, drone_idx=0, ref_vel=None,
            target_bldg_id=None, departure_bldg_id=None, cable_len=0.10, sway_2d=None, 
            has_payload=True, dt=0.01, f_skid_left=0.0, f_skid_right=0.0, ekf_pos_cov_trace=0.05):

        current_mass = 1.35 + (0.45 if has_payload else 0.0)
        weight_force = current_mass * G_ACCEL

        if ref_vel is None:
            ref_vel = np.zeros(3, dtype=np.float32)

        # 1. Translation Guidance & Anti-Windup Tracking
        delta_p = np.clip(p_target - p_est, -5.0, 5.0)
        v_err = np.clip(ref_vel - v_est, -3.0, 3.0)
        a_track = np.array([1.6, 1.6, 2.5], dtype=np.float32) * delta_p + np.array([2.0, 2.0, 2.8], dtype=np.float32) * v_err

        # Wind feedforward
        alpha_w = dt / (0.85 + dt)
        self.wind_filt += alpha_w * (estimated_wind - self.wind_filt)
        v_air = v_est - self.wind_filt
        f_drag = -0.5 * 1.225 * np.array([0.22, 0.22, 0.38], dtype=np.float32) * v_air * max(1.0, float(np.linalg.norm(v_air)))
        a_wind_ff = -f_drag / current_mass

        a_anti_sway = np.zeros(3, dtype=np.float32)
        if cable_len > 0.15 and sway_2d is not None:
            a_anti_sway[:2] -= 1.2 * sway_2d

        a_nn = neural_guidance[:3] if neural_guidance is not None else np.zeros(3, dtype=np.float32)
        a_nominal = a_track + a_wind_ff + a_anti_sway + a_nn

        # 2. CBF Safety Projection
        a_safe = self.cbf_filter.enforce_safety(
            drone_idx, p_est, v_est, a_nominal, obstacles, swarm_states,
            target_bldg_id=target_bldg_id, departure_bldg_id=departure_bldg_id,
            wind_2d=self.wind_filt[:2], ekf_pos_cov_trace=ekf_pos_cov_trace
        )

        F_d = current_mass * a_safe + np.array([0.0, 0.0, weight_force], dtype=np.float32)
        F_d[2] = float(np.clip(F_d[2], 0.0, 2.2 * weight_force))

        # 3. Low-Pass Filtered Attitude Tracking on SO(3)
        b3_d_raw = F_d / max(1e-4, float(np.linalg.norm(F_d)))
        b3_d = 0.72 * self.b3_d_prev + 0.28 * b3_d_raw
        b3_d /= max(1e-4, float(np.linalg.norm(b3_d)))
        self.b3_d_prev = b3_d.copy()

        b1_c = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b2_d = np.cross(b3_d, b1_c)
        b2_d /= max(1e-4, float(np.linalg.norm(b2_d)))
        b1_d = np.cross(b2_d, b3_d)
        R_d = np.column_stack((b1_d, b2_d, b3_d)).astype(np.float32)

        f_thrust = float(np.clip(np.dot(F_d, R_est[:, 2]), 0.0, 2.2 * weight_force))

        R_err = R_d.T @ R_est - R_est.T @ R_d
        e_R = 0.5 * np.array([R_err[2, 1], R_err[0, 2], R_err[1, 0]], dtype=np.float32)
        tau = -np.array([2.1, 2.1, 1.7], dtype=np.float32) * e_R - np.array([1.4, 1.4, 0.95], dtype=np.float32) * omega_est
        tau = np.clip(tau, -0.65, 0.65)

        # Tactile landing touchdown reflex
        if p_target[2] <= 0.15:
            scale, roll_adj, is_latched = self.tactile_reflex.compute_reflex(f_skid_left, f_skid_right, current_mass, dt=dt)
            f_thrust *= scale
            tau[0] += roll_adj
            if is_latched:
                f_thrust = 0.0
                tau.fill(0.0)
        else:
            self.tactile_reflex.reset()

        # 4. Motor Command Allocation
        W_health = np.diag(drone_health).astype(np.float32)
        B_active = self.B_nominal @ W_health
        desired_wrench = np.array([f_thrust, tau[0], tau[1], tau[2]], dtype=np.float32)

        try:
            T_cmds = np.linalg.pinv(B_active) @ desired_wrench
        except np.linalg.LinAlgError:
            T_cmds = np.full(8, f_thrust / 8.0, dtype=np.float32)

        T_cmds = np.clip(T_cmds * drone_health, 0.0, 10.5)

        omega_cmds = np.zeros(8, dtype=np.float32)
        for i in range(8):
            if drone_health[i] > 0.0:
                base_c = (C_T_CLEAN * COAXIAL_EFFICIENCY) if (i % 2 == 1) else C_T_CLEAN
                omega_cmds[i] = math.sqrt(T_cmds[i] / max(1e-6, base_c))

        self._last_motor_omegas = omega_cmds.copy()
        return omega_cmds, f_thrust, a_safe

# =============================================================================
# 4. 6-HEAD DEEP DENSE FLIGHT EXECUTIVE RESNET BRAIN (~2.2M PARAMETERS)
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
            nn.Linear(128, NUM_EXECUTIVE_STATES)
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

    """
SURVIVAL CORE V37.0 — FULL GENESIS MULTI-PHYSICS UPGRADE
PART 5 OF 5: MASTER SIMULATION ORCHESTRATION, PARALLEL GPU TRAINER
             & LIVE 3D INTERACTIVE FLIGHT COCKPIT
"""

import os
import sys
import math
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import genesis as gs


MODEL_WEIGHTS_FILE = "drone_master_brain_genesis_v2.pth"
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
# 1. BATCHED ROTATION & TENSOR UTILITIES
# =============================================================================
def quat_to_rot_batch(q: torch.Tensor) -> torch.Tensor:
    """Converts a batch of quaternions [B, 4] (w, x, y, z) into rotation matrices [B, 3, 3] on GPU."""
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

MOTOR_TAU_S = 0.030
MAX_RPM = 9200.0
MAX_OMEGA = (MAX_RPM * 2.0 * math.pi) / 60.0

# =============================================================================
# 2. VECTORIZED BEMT ENGINE & GEOMETRIC CONTROLLER
# =============================================================================
class VectorizedBEMTEngine(nn.Module):
    def __init__(self, n_envs: int, device: str = "cuda"):
        super().__init__()
        self.n_envs = n_envs
        self.device = device
        self.motor_omegas = torch.full((n_envs, 8), 450.0, device=device)

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

        hover_thrust = TOTAL_MASS_KG * GRAVITY_VAL
        v_h_hover = math.sqrt(hover_thrust / (2.0 * 1.225 * ROTOR_DISK_AREA * 4.0))
        v_z_descent = -vel[:, 2]
        is_vrs_band = (v_z_descent > 0.4 * v_h_hover) & (v_z_descent < 1.6 * v_h_hover)
        vrs_factor = torch.ones(self.n_envs, device=self.device)
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
    def __init__(self, n_envs: int, device: str = "cuda"):
        self.n_envs = n_envs
        self.device = device
        self.k_v = torch.tensor([2.0, 2.0, 2.5], device=device)
        self.k_R = torch.tensor([2.1, 2.1, 1.7], device=device)
        self.k_w = torch.tensor([1.4, 1.4, 0.95], device=device)

    def compute_wrench(self, pos, vel, quat, ang_vel, target_pos, current_mass, nn_guidance_accel=None):
        B = self.n_envs
        R = quat_to_rot_batch(quat)

        delta_p = target_pos - pos
        v_cmd_xy = torch.clamp(delta_p[:, :2] * 1.4, -6.0, 6.0)
        v_cmd_z = torch.clamp(delta_p[:, 2] * 1.2, -1.5, 2.4)
        v_des = torch.cat([v_cmd_xy, v_cmd_z.unsqueeze(-1)], dim=-1)

        a_des = self.k_v * (v_des - vel)
        a_des[:, 2] += GRAVITY_VAL
        if nn_guidance_accel is not None:
            a_des += nn_guidance_accel

        a_des = torch.clamp(a_des, -15.0, 22.0)
        F_d = current_mass * a_des
        b3_d = F_d / torch.clamp(torch.norm(F_d, dim=-1, keepdim=True), min=1e-4)

        b1_c = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(B, 1)
        b2_d = torch.linalg.cross(b3_d, b1_c, dim=-1)
        b2_d = b2_d / torch.clamp(torch.norm(b2_d, dim=-1, keepdim=True), min=1e-4)
        b1_d = torch.linalg.cross(b2_d, b3_d, dim=-1)
        R_d = torch.stack([b1_d, b2_d, b3_d], dim=-1)

        f_thrust = torch.clamp(torch.sum(F_d * R[:, :, 2], dim=-1), min=0.0, max=2.0 * (current_mass * GRAVITY_VAL))
        R_err = torch.bmm(R_d.transpose(1, 2), R) - torch.bmm(R.transpose(1, 2), R_d)
        e_R = 0.5 * vee_map_batch(R_err)
        tau = torch.clamp(-self.k_R * e_R - self.k_w * ang_vel, -0.65, 0.65)
        return f_thrust, tau

# =============================================================================
# 3. PROCEDURAL CITY BUILDER
# =============================================================================
def build_metropolitan_city(scene):
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(friction=0.8),
        surface=gs.surfaces.Rough(color=(0.18, 0.20, 0.24), roughness=0.8)
    )

    buildings = []
    avenues_x = [-60.0, -30.0, 0.0, 30.0, 60.0]
    streets_y = [-60.0, -30.0, 0.0, 30.0, 60.0]
    bldg_idx = 0

    for c_i, cx in enumerate(avenues_x):
        for r_i, ry in enumerate(streets_y):
            if cx <= -20.0 and ry <= -20.0:
                continue

            tier = (c_i + r_i) % 4
            height = [8.5, 14.0, 20.0, 26.0][tier]
            bw, bd = 16.0, 16.0

            scene.add_entity(
                morph=gs.morphs.Box(size=(bw, bd, height), pos=(cx, ry, height / 2.0), fixed=True),
                material=gs.materials.Rigid(friction=0.7),
                surface=gs.surfaces.Rough(color=(0.28 + 0.03 * tier, 0.32 + 0.03 * tier, 0.38 + 0.03 * tier), roughness=0.6)
            )

            scene.add_entity(
                morph=gs.morphs.Box(size=(3.2, 2.8, 1.4), pos=(cx - 3.5, ry + 3.5, height + 0.7), fixed=True),
                material=gs.materials.Rigid(friction=0.6),
                surface=gs.surfaces.Rough(color=(0.55, 0.55, 0.58), roughness=0.5)
            )

            pad_pos = (cx + 2.0, ry - 2.0, height + 0.04)
            scene.add_entity(
                morph=gs.morphs.Cylinder(radius=2.2, height=0.08, pos=pad_pos, fixed=True),
                material=gs.materials.Rigid(friction=0.85),
                surface=gs.surfaces.Default(color=(0.0, 0.85, 0.55))
            )

            buildings.append({
                "id": f"Bldg_{bldg_idx+1:02d}",
                "pos": np.array(pad_pos, dtype=np.float32),
                "height": height,
                "radius": 10.5
            })
            bldg_idx += 1

    depot_pad = np.array([-45.0, -45.0, 0.04], dtype=np.float32)
    scene.add_entity(
        morph=gs.morphs.Cylinder(radius=3.2, height=0.08, pos=tuple(depot_pad), fixed=True),
        material=gs.materials.Rigid(friction=0.9),
        surface=gs.surfaces.Default(color=(0.10, 0.65, 0.95))
    )

    return buildings, depot_pad, avenues_x

# =============================================================================
# 4. SENSOR PIPELINE & 3D LIDAR RAYCASTER
# =============================================================================
class MacroPerceptionLiDARSuite:
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
    surf_z = 0.0
    for b in buildings:
        bx, by, bh = b["pos"][0], b["pos"][1], b["height"]
        if (bx - 8.0) <= px <= (bx + 8.0) and (by - 8.0) <= py <= (by + 8.0):
            surf_z = max(surf_z, bh)
    return surf_z

def poll_all_real_sensors(
    d_pos_np, d_vel_np, d_R_np, d_ang_np,
    c_pos_np, c_vel_np,
    active_target_np, target_pad_xyz,
    total_wind_np, cable_len, has_payload, tension_mag,
    motor_omegas_np, bms_soc, bms_v_pack, bms_temp_c, bms_r0,
    ice_idx, sim_time, buildings,
    sensor_suite, full_physics, quantum_sensor, imu_sensor, lidar_suite, obs_synth
):
    surf_z = get_ground_surface_under_drone(d_pos_np[0], d_pos_np[1], buildings)
    drone_yaw = math.atan2(d_R_np[1, 0], d_R_np[0, 0])
    lidar_16 = lidar_suite.cast_360_lidar(d_pos_np, drone_yaw)
    conical_8 = lidar_suite.cast_downward_conical(d_pos_np)
    phys_data = full_physics.poll(d_pos_np, d_vel_np, total_wind_np, surface_z=surf_z)
    flow_2d, flow_q, _ = sensor_suite.poll_optical_flow(d_pos_np, d_vel_np, d_R_np, d_ang_np, surface_z=surf_z)
    aruco_3d, aruco_lk = sensor_suite.poll_aruco_detector(d_pos_np, d_R_np, target_pad_xyz)

    sway_2d = (c_pos_np[:2] - d_pos_np[:2]) if has_payload else np.zeros(2, dtype=np.float32)
    hook_a, hook_w, load_3d = sensor_suite.poll_smart_hook_and_load_cell(
        tension_n=tension_mag, sway_2d=sway_2d, cable_l=cable_len, has_payload=has_payload
    )

    esc_i_rms, esc_t_mosfet = sensor_suite.poll_esc_telemetry(motor_omegas_np)
    alpha_deg, beta_deg, rh_pct, dust_pm10 = sensor_suite.poll_5port_pitot(d_vel_np, total_wind_np, d_R_np)
    hdop, cn0, rf_pwr, mag_grad = sensor_suite.poll_rtk_gnss(is_spoofed=False)
    th_dt, dolp, ndvi = sensor_suite.poll_thermal_and_polarization(d_pos_np)
    siren_ang, siren_freq, siren_spl = sensor_suite.poll_acoustic_array(d_pos_np, sim_time=sim_time, siren_active=True)
    adsb_rel, adsb_spd, _ = sensor_suite.poll_adsb_transponder(d_pos_np, d_vel_np)

    meas_accel, meas_gyro = imu_sensor.poll(d_R_np, d_vel_np / 0.005, d_ang_np, motor_omegas_np)
    _, phase_shift = quantum_sensor.measure_phase_shift(meas_accel)

    is_grounded = bool((d_pos_np[2] <= surf_z + 0.12) and (d_vel_np[2] < 0.2))
    skid_f = 0.5 * (TOTAL_MASS_KG * 9.81) if is_grounded else 0.0

    return obs_synth.synthesize(
        p_est=d_pos_np, v_est=d_vel_np, R_est=d_R_np, omega_est=d_ang_np,
        target_p=active_target_np, lidar_16=lidar_16, conical_8=conical_8,
        v_tas=phys_data["v_tas"], q_dyn=phys_data["q_dyn"], min_sonar=phys_data["min_sonar"],
        wind_est=total_wind_np, ice_idx=ice_idx, cable_len=cable_len,
        cable_vel=float(np.linalg.norm(c_vel_np - d_vel_np)),
        tension_n=tension_mag, sway_2d=sway_2d,
        bms_soc=bms_soc, v_pack=bms_v_pack, bms_temp_c=bms_temp_c, r0_dyn=bms_r0,
        chi2_val=1.2, jerk_discrepancy=float(np.linalg.norm(meas_accel[:2])),
        is_spoofed=False, emf_rms=phys_data["emf_rms"], wire_dist=40.0,
        agent_dist=25.0, ground_spl_dba=52.0, has_payload=has_payload,
        hook_a=hook_a, hook_w=hook_w, load_cell_3d=load_3d,
        thermal_dt=th_dt, dolp=dolp, ndvi=ndvi,
        esc_i_rms=esc_i_rms, esc_t_mosfet=esc_t_mosfet,
        alpha_deg=alpha_deg, beta_deg=beta_deg, rh_pct=rh_pct, dust_pm10=dust_pm10,
        hdop=hdop, cn0_avg=cn0, rf_jam_pwr=rf_pwr, mag_grad=mag_grad,
        airframe_accel=meas_accel, airframe_gyro=meas_gyro,
        peer_swarm_kinematics=[], optical_flow_2d=flow_2d, flow_quality=flow_q,
        aruco_offset_3d=aruco_3d, aruco_locked=aruco_lk,
        skid_force_left=skid_f, skid_force_right=skid_f, is_skid_grounded=is_grounded,
        siren_angle=siren_ang, siren_spl=siren_spl,
        adsb_rel_pos=adsb_rel, adsb_closing_spd=adsb_spd
    )

# =============================================================================
# 5. PARALLEL GPU TRAINING PIPELINE
# =============================================================================
def train_full_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*75)
    print("STAGE 1: PARALLEL GPU HARVESTING & RESNET PRE-TRAINING (64 ENVS)")
    print("="*75)

    if not os.path.exists(URDF_ASSET_FILE):
        generate_coaxial_x8_urdf(URDF_ASSET_FILE)

    gs.init(backend=gs.gpu)
    dt = 0.005  # 200 Hz
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=2, gravity=(0, 0, -GRAVITY_VAL)),
        show_viewer=False
    )

    buildings, depot_pad, avenues_x = build_metropolitan_city(scene)
    corridor_planner = VoxelOctreeCorridorPlanner(buildings, depot_pad)

    cargo = scene.add_entity(
        morph=gs.morphs.Box(size=(0.22, 0.22, 0.18), pos=(-45.0, -45.0, 0.18)),
        material=gs.materials.Rigid(rho=45.0, friction=0.7),
        surface=gs.surfaces.Default(color=(0.95, 0.75, 0.15))
    )

    drone = scene.add_entity(
        morph=gs.morphs.URDF(file=URDF_ASSET_FILE, pos=(-45.0, -45.0, 0.70)),
        material=gs.materials.Rigid(friction=0.5)
    )

    n_envs = 2048
    scene.build(n_envs=n_envs)
    print(f"\n[GPU HARVEST] Simulating {n_envs} parallel multi-physics drones on {torch.cuda.get_device_name(0)}...")

    bemt = VectorizedBEMTEngine(n_envs=n_envs, device="cuda")
    ctrl = GeometricSE3Controller(n_envs=n_envs, device="cuda")
    weather_engine = DynamicUrbanWeatherEngine()
    dryden_engine = DrydenAtmosphericTurbulenceModel()
    icing_model = MessingerRotorIcingModel()
    battery_ecm = SubZeroArrheniusBatteryECM()
    sensor_suite = AdvancedMultiModalSensorSuite(buildings)
    full_physics = FullPhysicsSensorSuite()
    quantum_sensor = MatterWaveColdAtomQuantumSensor()
    imu_sensor = Airframe6DOFMEMS_IMU()
    lidar_suite = MacroPerceptionLiDARSuite(buildings)
    obs_synth = FullObservationSynthesizer139Dim()

    drone_link = drone.links[0]
    cargo_link = cargo.links[0]

    target_dest = buildings[5]
    # 1. Stack all 3D building positions and heights onto the GPU
    bldg_pos_np = np.stack([b["pos"] for b in buildings]).astype(np.float32)
    all_bldg_pos = torch.from_numpy(bldg_pos_np).to("cuda")

    bldg_h_np = np.array([b["height"] for b in buildings], dtype=np.float32)
    all_bldg_h = torch.from_numpy(bldg_h_np).to("cuda")

    # 2. Assign a random building to EACH environment
    target_indices = torch.randint(0, len(buildings), (n_envs,), device="cuda")
    tgt_pad = all_bldg_pos[target_indices]  # [n_envs, 3] - distinct rooftop coordinates for every drone
    roof_h = all_bldg_h[target_indices]    # [n_envs]    - distinct rooftop heights

    # 3. Universal safe cruise altitude clearing all skyscrapers in the city (>=34m)
    safe_cruise_alt = max(MIN_SAFE_CRUISE_ALT_M, float(all_bldg_h.max().item()) + 6.0)

    collected_obs = []
    collected_targets = []
    temporal_queues = [deque(maxlen=HISTORY_LEN) for _ in range(n_envs)]

    cable_len = 0.55
    has_payload = True
    fsm_phase = 0
    t_start = time.time()

    for step in range(10000):
        sim_time = step * dt
        d_pos, d_vel, d_quat, d_ang = drone.get_pos(), drone.get_vel(), drone.get_quat(), drone.get_ang()
        c_pos, c_vel = cargo.get_pos(), cargo.get_vel()
        curr_m = TOTAL_MASS_KG if has_payload else DRY_MASS_KG

        eff_wind, gust_scale, temp_c, auto_ice = weather_engine.get_weather_parameters(d_pos[0].cpu().numpy(), avenues_x)
        dryden_gust = dryden_engine.step(float(d_pos[0, 2].item()), float(torch.norm(d_vel[0]).item()), dt=dt) * (gust_scale / 2.0)
        total_wind = eff_wind + dryden_gust

        icing_model.icing_active = auto_ice
        ice_mm, ice_idx, lift_loss, drag_rise = icing_model.step(float(torch.norm(d_vel[0]).item()), dt=dt)
        soc, v_pack, i_draw, r0 = battery_ecm.step(195.0, amb_temp_c=temp_c, dt=dt)

        # -------------------------------------------------------------
        # CORRIDOR WAYPOINT STATE MACHINE (PER-ENVIRONMENT TARGETS)
        # -------------------------------------------------------------
        if fsm_phase == 0:  # Climb Chimney
            active_target = torch.tensor([-45.0, -45.0, safe_cruise_alt], device="cuda").repeat(n_envs, 1)
            fsm_label = 0
            winch_label = torch.tensor([0.0, 0.0], device="cuda").repeat(n_envs, 1)
            # Trigger cruise when reaching within 3.0m of cruise altitude or after 10 seconds
            if d_pos[:, 2].mean().item() >= (safe_cruise_alt - 3.0) or sim_time >= 10.0:
                fsm_phase = 1
                print(f"[{sim_time:.1f}s] HARVESTING: Transitioned to Phase 1 (CRUISE OUTBOUND)!")

        elif fsm_phase == 1:  # Cruise Over City to Assigned Rooftop
            # Each drone aims at its own building's XY coordinates at the safe cruise floor
            active_target = torch.stack([tgt_pad[:, 0], tgt_pad[:, 1], torch.full((n_envs,), safe_cruise_alt, device="cuda")], dim=-1)
            fsm_label = 1
            winch_label = torch.tensor([0.0, 0.0], device="cuda").repeat(n_envs, 1)
            dist_xy = torch.norm(d_pos[:, :2] - tgt_pad[:, :2], dim=-1).mean().item()
            if dist_xy <= 3.0:
                fsm_phase = 2

        elif fsm_phase == 2:  # Chimney Descent Over Rooftop
            active_target = torch.stack([tgt_pad[:, 0], tgt_pad[:, 1], roof_h + 7.5], dim=-1)
            fsm_label = 2
            winch_label = torch.tensor([0.0, 0.0], device="cuda").repeat(n_envs, 1)
            if abs((d_pos[:, 2] - (roof_h + 7.5)).mean().item()) <= 0.8:
                fsm_phase = 3

        elif fsm_phase == 3:  # Delivery Hover & Tether Lowering
            active_target = torch.stack([tgt_pad[:, 0], tgt_pad[:, 1], roof_h + 7.5], dim=-1)
            fsm_label = 3
            if has_payload and cable_len < 7.2:
                cable_len += 1.8 * dt
                winch_label = torch.tensor([1.0, 0.0], device="cuda").repeat(n_envs, 1)
            else:
                has_payload = False
                winch_label = torch.tensor([0.0, 1.0], device="cuda").repeat(n_envs, 1)
                fsm_phase = 4

        elif fsm_phase == 4:  # Winch Retraction
            active_target = torch.stack([tgt_pad[:, 0], tgt_pad[:, 1], roof_h + 7.5], dim=-1)
            fsm_label = 4
            cable_len = max(0.20, cable_len - 3.0 * dt)
            winch_label = torch.tensor([-1.0, 1.0], device="cuda").repeat(n_envs, 1)
            if cable_len <= 0.25:
                fsm_phase = 5

        elif fsm_phase == 5:  # Climb & Return to Depot
            active_target = torch.tensor([-45.0, -45.0, safe_cruise_alt], device="cuda").repeat(n_envs, 1)
            fsm_label = 5
            winch_label = torch.tensor([0.0, 1.0], device="cuda").repeat(n_envs, 1)
            dist_home = torch.norm(d_pos[:, :2] - torch.tensor([-45.0, -45.0], device="cuda"), dim=-1).mean().item()
            if dist_home <= 2.5:
                fsm_phase = 6

        else:  # Depot Helipad Landing
            active_target = torch.tensor([-45.0, -45.0, 0.70], device="cuda").repeat(n_envs, 1)
            fsm_label = 6
            winch_label = torch.tensor([0.0, 1.0], device="cuda").repeat(n_envs, 1)

        f_tether_drone = torch.zeros((n_envs, 3), device="cuda")
        t_mag = torch.zeros((n_envs, 1), device="cuda")
        if has_payload:
            tether_vec = c_pos - d_pos
            tether_dist = torch.norm(tether_vec, dim=-1, keepdim=True)
            slack = tether_dist - cable_len
            is_taut = (slack > 0.0)
            tether_dir = tether_vec / torch.clamp(tether_dist, min=1e-4)
            rel_vel = c_vel - d_vel
            damping_proj = torch.sum(rel_vel * tether_dir, dim=-1, keepdim=True)
            t_mag = torch.where(is_taut, torch.clamp(slack * 100.0 + damping_proj * 4.0, min=0.0, max=18.0), torch.zeros_like(slack))
            f_tether_cargo = -t_mag * tether_dir
            f_tether_drone = t_mag * tether_dir
            cargo_link.apply_external_force(force=f_tether_cargo)

        f_cmd, tau_cmd = ctrl.compute_wrench(d_pos, d_vel, d_quat, d_ang, active_target, curr_m)
        motor_cmds = bemt.allocate(f_cmd, tau_cmd)
        t_act, q_act = bemt.compute_aerodynamics(motor_cmds, d_pos, d_vel, roof_h if fsm_phase in [2, 3, 4] else 0.0, dt)

        R_batch = quat_to_rot_batch(d_quat)
        thrust_vec = R_batch[:, :, 2] * t_act.unsqueeze(-1)
        drone_link.apply_external_force(force=thrust_vec + f_tether_drone)
        drone_link.apply_external_torque(torque=q_act)
        scene.step()

        if step % 2 == 0:
            # Harvest transitions from 4 different environments visiting 4 different buildings
            num_harvest = min(n_envs, 4)
            for k in range(num_harvest):
                d_pk = d_pos[k].cpu().numpy()
                d_vk = d_vel[k].cpu().numpy()
                d_Rk = R_batch[k].cpu().numpy()
                d_wk = d_ang[k].cpu().numpy()
                c_pk = c_pos[k].cpu().numpy()
                c_vk = c_vel[k].cpu().numpy()
                motor_omk = bemt.motor_omegas[k].cpu().numpy()
                t_mag_val = float(t_mag[k].item()) if has_payload else 0.0

                assigned_bldg = buildings[target_indices[k].item()]

                obs_139 = poll_all_real_sensors(
                    d_pos_np=d_pk, d_vel_np=d_vk, d_R_np=d_Rk, d_ang_np=d_wk,
                    c_pos_np=c_pk, c_vel_np=c_vk,
                    active_target_np=active_target[k].cpu().numpy(),
                    target_pad_xyz=assigned_bldg["pos"],
                    total_wind_np=total_wind, cable_len=cable_len,
                    has_payload=has_payload, tension_mag=t_mag_val,
                    motor_omegas_np=motor_omk, bms_soc=soc,
                    bms_v_pack=v_pack, bms_temp_c=temp_c, bms_r0=r0,
                    ice_idx=ice_idx, sim_time=sim_time, buildings=buildings,
                    sensor_suite=sensor_suite, full_physics=full_physics,
                    quantum_sensor=quantum_sensor, imu_sensor=imu_sensor,
                    lidar_suite=lidar_suite, obs_synth=obs_synth
                )

                obs_417 = obs_synth.stack_history(temporal_queues[k], obs_139)
                collected_obs.append(torch.from_numpy(obs_417).float().to(device))

                guidance_gt = torch.clamp((active_target[k] - d_pos[k]) / 5.0, -1.0, 1.0)
                guidance_4d = torch.cat([guidance_gt, torch.zeros(1, device=device)])
                fsm_gt = torch.tensor(fsm_label, device=device, dtype=torch.long)
                health_gt = torch.ones(8, device=device)
                trust_gt = torch.tensor([1.0, 1.0], device=device)
                dist_gt = torch.norm(active_target[k] - d_pos[k], keepdim=True)

                collected_targets.append((guidance_4d, fsm_gt, winch_label[k], health_gt, trust_gt, dist_gt))

    print(f"\n[DATASET READY] Harvested {len(collected_obs):,} 417-dimensional state transitions in {time.time()-t_start:.2f}s!")

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
    batch_size = 128
    epochs = 15

    print(f"\n[TRAINING] Multi-task training 6-head ResNet executive ({epochs} epochs)...")
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

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] — Executive Loss: {total_loss / n_batches:.4f}")

    torch.save(model.state_dict(), MODEL_WEIGHTS_FILE)
    print(f"\n[SUCCESS] Model weights saved to '{MODEL_WEIGHTS_FILE}'.")

# =============================================================================
# 6. INTERACTIVE 3D EVALUATION SIMULATOR
# =============================================================================
def evaluate_full_system():
    if not os.path.exists(MODEL_WEIGHTS_FILE):
        print(f"\n[WARNING] '{MODEL_WEIGHTS_FILE}' not found. Training model first...")
        train_full_system()

    if not os.path.exists(URDF_ASSET_FILE):
        generate_coaxial_x8_urdf(URDF_ASSET_FILE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE, map_location=device))
    model.eval()
    print(f"\nLoaded verified 6-Head Flight Executive from '{MODEL_WEIGHTS_FILE}'!")

    gs.init(backend=gs.gpu)
    dt = 0.005  # 200 Hz
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=2, gravity=(0, 0, -GRAVITY_VAL)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(45.0, -45.0, 35.0),
            camera_lookat=(0.0, 0.0, 15.0),
            res=(1280, 720),
            refresh_rate=60,
        ),
        show_viewer=True
    )

    buildings, depot_pad, avenues_x = build_metropolitan_city(scene)
    corridor_planner = VoxelOctreeCorridorPlanner(buildings, depot_pad)

    cargo = scene.add_entity(
        morph=gs.morphs.Box(size=(0.22, 0.22, 0.18), pos=(-45.0, -45.0, 0.18)),
        material=gs.materials.Rigid(rho=45.0, friction=0.7),
        surface=gs.surfaces.Default(color=(0.95, 0.75, 0.15))
    )

    drone = scene.add_entity(
        morph=gs.morphs.URDF(file=URDF_ASSET_FILE, pos=(-45.0, -45.0, 0.70)),
        material=gs.materials.Rigid(friction=0.5)
    )

    n_envs = 4
    scene.build(n_envs=n_envs)

    bemt = VectorizedBEMTEngine(n_envs=n_envs, device="cuda")
    ctrl = GeometricSE3Controller(n_envs=n_envs, device="cuda")
    weather_engine = DynamicUrbanWeatherEngine()
    dryden_engine = DrydenAtmosphericTurbulenceModel()
    battery_ecm = SubZeroArrheniusBatteryECM()
    sensor_suite = AdvancedMultiModalSensorSuite(buildings)
    full_physics = FullPhysicsSensorSuite()
    quantum_sensor = MatterWaveColdAtomQuantumSensor()
    imu_sensor = Airframe6DOFMEMS_IMU()
    lidar_suite = MacroPerceptionLiDARSuite(buildings)
    obs_synth = FullObservationSynthesizer139Dim()

    drone_link = drone.links[0]
    cargo_link = cargo.links[0]

    target_dest = buildings[5]
    waypoints, safe_cruise_alt = corridor_planner.plan_mission_waypoints(depot_pad, target_dest)
    roof_h = float(target_dest["height"])

    cable_len = 0.55
    has_payload = True
    temporal_queue = deque(maxlen=HISTORY_LEN)

    print("\n" + "="*75)
    print("V37.0 FULL MULTI-PHYSICS NEURAL FLIGHT COCKPIT ONLINE [4 ENVS]")
    print(f"Destination: {target_dest['id']} @ {roof_h:.1f}m | Corridor Floor: {safe_cruise_alt:.1f}m")
    print("3D Asset: Real Coaxial X8 Drone URDF (Fuselage, Arms, 8 Rotors, Skids)")
    print("="*75 + "\n")

    try:
        for step in range(10000):
            if scene.viewer is not None and not scene.viewer.is_alive():
                print("\n[INFO] Viewer window closed by user. Exiting cleanly...")
                break

            d_pos, d_vel, d_quat, d_ang = drone.get_pos(), drone.get_vel(), drone.get_quat(), drone.get_ang()
            c_pos, c_vel = cargo.get_pos(), cargo.get_vel()
            sim_time = step * dt
            curr_m = TOTAL_MASS_KG if has_payload else DRY_MASS_KG

            eff_wind, gust_scale, temp_c, _ = weather_engine.get_weather_parameters(d_pos[0].cpu().numpy(), avenues_x)
            dryden_gust = dryden_engine.step(float(d_pos[0, 2].item()), float(torch.norm(d_vel[0]).item()), dt=dt) * (gust_scale / 2.0)
            total_wind = eff_wind + dryden_gust
            soc, v_pack, i_draw, r0 = battery_ecm.step(195.0, amb_temp_c=temp_c, dt=dt)

            d_p0 = d_pos[0].cpu().numpy()
            d_v0 = d_vel[0].cpu().numpy()
            d_R0 = quat_to_rot_batch(d_quat)[0].cpu().numpy()
            d_w0 = d_ang[0].cpu().numpy()
            c_p0 = c_pos[0].cpu().numpy()
            c_v0 = c_vel[0].cpu().numpy()
            motor_om0 = bemt.motor_omegas[0].cpu().numpy()

            f_tether_drone = torch.zeros((n_envs, 3), device="cuda")
            t_mag = torch.zeros((n_envs, 1), device="cuda")
            if has_payload:
                tether_vec = c_pos - d_pos
                tether_dist = torch.norm(tether_vec, dim=-1, keepdim=True)
                slack = tether_dist - cable_len
                is_taut = (slack > 0.0)
                tether_dir = tether_vec / torch.clamp(tether_dist, min=1e-4)
                rel_vel = c_vel - d_vel
                damping_proj = torch.sum(rel_vel * tether_dir, dim=-1, keepdim=True)
                t_mag = torch.where(is_taut, torch.clamp(slack * 100.0 + damping_proj * 4.0, min=0.0, max=18.0), torch.zeros_like(slack))
                f_tether_cargo = -t_mag * tether_dir
                f_tether_drone = t_mag * tether_dir
                cargo_link.apply_external_force(force=f_tether_cargo)

            t_mag_val = float(t_mag[0].item()) if has_payload else 0.0

            obs_139 = poll_all_real_sensors(
                d_pos_np=d_p0, d_vel_np=d_v0, d_R_np=d_R0, d_ang_np=d_w0,
                c_pos_np=c_p0, c_vel_np=c_v0,
                active_target_np=waypoints[2],
                target_pad_xyz=target_dest["pos"],
                total_wind_np=total_wind, cable_len=cable_len,
                has_payload=has_payload, tension_mag=t_mag_val,
                motor_omegas_np=motor_om0, bms_soc=soc,
                bms_v_pack=v_pack, bms_temp_c=temp_c, bms_r0=r0,
                ice_idx=0.0, sim_time=sim_time, buildings=buildings,
                sensor_suite=sensor_suite, full_physics=full_physics,
                quantum_sensor=quantum_sensor, imu_sensor=imu_sensor,
                lidar_suite=lidar_suite, obs_synth=obs_synth
            )

            obs_417 = obs_synth.stack_history(temporal_queue, obs_139)
            obs_tensor = torch.from_numpy(obs_417).float().unsqueeze(0).to(device).repeat(n_envs, 1)

            # -------------------------------------------------------------
            # 1. Neural Executive Inference
            # -------------------------------------------------------------
            with torch.no_grad():
                nn_guidance, nn_fsm, nn_winch, nn_health, nn_trust, _ = model(obs_tensor)

            fsm_idx = int(nn_fsm.argmax(dim=-1)[0].item())

            # -------------------------------------------------------------
            # 2. HIERARCHICAL ARBITER: Supervised Waypoint Progression
            # (Overrides / advances the phase if physical conditions are met)
            # -------------------------------------------------------------
            dist_to_bldg_xy = torch.norm(d_pos[:, :2] - torch.tensor(target_dest["pos"][:2], device="cuda"), dim=-1).mean().item()
            dist_to_depot_xy = torch.norm(d_pos[:, :2] - torch.tensor([-45.0, -45.0], device="cuda"), dim=-1).mean().item()
            current_alt = d_pos[:, 2].mean().item()

            if fsm_idx == 0:
                # If climbing and reached cruise floor (or after 8 seconds): advance to cruise
                if current_alt >= (safe_cruise_alt - 3.0) or sim_time > 8.0:
                    fsm_idx = 1
            elif fsm_idx == 1:
                # If cruising and arrived over the building: advance to chimney descent
                if dist_to_bldg_xy <= 3.5:
                    fsm_idx = 2
            elif fsm_idx == 2:
                # If descending and in hover position above roof: advance to delivery
                if abs(current_alt - (roof_h + 7.5)) <= 0.8:
                    fsm_idx = 3
            elif fsm_idx == 3:
                # If cargo was delivered/released: advance to winch retract
                if not has_payload:
                    fsm_idx = 4
            elif fsm_idx == 4:
                # If winch is retracted: advance to return flight
                if cable_len <= 0.25:
                    fsm_idx = 5
            elif fsm_idx == 5:
                # If arrived back over depot: advance to landing
                if dist_to_depot_xy <= 3.0:
                    fsm_idx = 6

            phase_name = FSM_NAMES[fsm_idx]

            # -------------------------------------------------------------
            # 3. 3D Glide-Slope Corridor Waypoint Target Assignment
            # -------------------------------------------------------------
            if fsm_idx == 0:
                active_target = torch.tensor(waypoints[0], device="cuda").repeat(n_envs, 1)
            elif fsm_idx == 1:
                active_target = torch.tensor(waypoints[2], device="cuda").repeat(n_envs, 1)
            elif fsm_idx in [2, 3, 4]:
                active_target = torch.tensor(waypoints[3], device="cuda").repeat(n_envs, 1)
            elif fsm_idx == 5:
                active_target = torch.tensor([-45.0, -45.0, safe_cruise_alt], device="cuda").repeat(n_envs, 1)
            else:
                active_target = torch.tensor([-45.0, -45.0, 0.70], device="cuda").repeat(n_envs, 1)

            latch_release = float(nn_winch[0, 1].item()) > 0.4
            if has_payload:
                if fsm_idx == 3: cable_len = min(7.2, cable_len + 1.8 * dt)
                if latch_release and c_pos[:, 2].mean().item() <= (roof_h + 0.25):
                    has_payload = False
                    print(f"[{sim_time:4.1f}s] TOUCHDOWN CONFIRMED: Cargo delivered on {target_dest['id']} helipad!")
            else:
                cable_len = max(0.20, cable_len - 3.2 * dt)

            f_cmd, tau_cmd = ctrl.compute_wrench(
                d_pos, d_vel, d_quat, d_ang, active_target, curr_m,
                nn_guidance_accel=nn_guidance[:, :3]
            )
            motor_cmds = bemt.allocate(f_cmd, tau_cmd)
            t_act, q_act = bemt.compute_aerodynamics(motor_cmds, d_pos, d_vel, roof_h if fsm_idx in [2, 3, 4] else 0.0, dt)

            R_batch = quat_to_rot_batch(d_quat)
            thrust_vec = R_batch[:, :, 2] * t_act.unsqueeze(-1)
            drone_link.apply_external_force(force=thrust_vec + f_tether_drone)
            drone_link.apply_external_torque(torque=q_act)

            scene.step()

            if step % 400 == 0:
                alt = d_pos[:, 2].mean().item()
                c_alt = c_pos[:, 2].mean().item()
                spd = torch.norm(d_vel[:, :2], dim=-1).mean().item()
                print(f"[{sim_time:4.1f}s] FSM: {phase_name:<16} | Drone Alt: {alt:5.1f}m (Cruise: {safe_cruise_alt:.1f}m) | Spd: {spd:4.1f}m/s | SoC: {soc*100:4.1f}% | V_pack: {v_pack:4.1f}V")

    except gs.GenesisException as e:
        if "Viewer closed" in str(e):
            print("\n[INFO] Viewer window closed by user. Simulation exited cleanly.")
        else:
            raise e

# =============================================================================
# 7. MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_full_system()
    else:
        evaluate_full_system()