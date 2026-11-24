"""
SURVIVAL CORE V37.0 — FULL GENESIS MULTI-PHYSICS UPGRADE
PART 1 OF 4: 3-DRONE SWARM SPAWNING, DYNAMIC ROAD TRAFFIC,
             CIVILIAN MEDEVAC AIRCRAFT & GROUND INTRUDER AGENT
"""

import math
import random
import numpy as np
import genesis as gs

URDF_ASSET_FILE = "drone_coaxial_x8.urdf"

# =============================================================================
# 1. PROCEDURAL 3D CITY BUILDER WITH 3 DEPOT PADS & 48 BUILDINGS
# =============================================================================
def build_metropolitan_city_with_swarm_depot(scene):
    """
    Spawns the 400m Metropolitan grid with 48 building towers, rooftop helipads,
    HVAC clutter, avenues, and 3 distinct base station depot pads.
    """
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
            # Reserve southwest corner for base station depot and clearway
            if cx <= -20.0 and ry <= -20.0:
                continue

            tier = (c_i + r_i) % 4
            height = [8.5, 14.0, 20.0, 26.0][tier]
            bw, bd = 16.0, 16.0

            # Building Tower Chassis
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
                "pos": np.array(pad_pos, dtype=np.float32),
                "height": height,
                "radius": 10.5
            })
            bldg_idx += 1

    # 3 Separate Depot Helipads for the 3 Swarm Drones
    depot_pads = [
        np.array([-48.0, -48.0, 0.04], dtype=np.float32),  # Pad for Drone 1
        np.array([-48.0, -36.0, 0.04], dtype=np.float32),  # Pad for Drone 2
        np.array([-36.0, -48.0, 0.04], dtype=np.float32),  # Pad for Drone 3
    ]
    pad_colors = [(0.10, 0.65, 0.95), (0.95, 0.55, 0.15), (0.75, 0.25, 0.95)]

    for p_i, dp in enumerate(depot_pads):
        scene.add_entity(
            morph=gs.morphs.Cylinder(radius=2.8, height=0.08, pos=tuple(dp), fixed=True),
            material=gs.materials.Rigid(friction=0.9),
            surface=gs.surfaces.Default(color=pad_colors[p_i])
        )

    return manifest, depot_pads, avenues_x, streets_y

# =============================================================================
# 2. SWARM ENTITY FACTORY (3 DRONES & 3 CARGO PAYLOADS)
# =============================================================================
def spawn_swarm_drones_and_cargo(scene, depot_pads):
    """
    Spawns 3 physical Coaxial X8 Drones and 3 corresponding physical Cargo Boxes
    with strict non-overlapping clearances.
    """
    drones = []
    cargos = []
    drone_colors = [(0.10, 0.65, 0.95), (0.95, 0.55, 0.15), (0.75, 0.25, 0.95)]

    for i in range(3):
        pad_x, pad_y = depot_pads[i][0], depot_pads[i][1]

        # Cargo payload resting cleanly on the helipad: z = 0.18m (bottom at 0.09m, pad top at 0.08m)
        cargo_entity = scene.add_entity(
            morph=gs.morphs.Box(size=(0.22, 0.22, 0.18), pos=(pad_x, pad_y, 0.18)),
            material=gs.materials.Rigid(rho=45.0, friction=0.7),
            surface=gs.surfaces.Default(color=(0.95, 0.75, 0.15))
        )
        cargos.append(cargo_entity)

        # Coaxial X8 Drone resting above cargo with initial cable slack: z = 0.70m
        drone_entity = scene.add_entity(
            morph=gs.morphs.URDF(file=URDF_ASSET_FILE, pos=(pad_x, pad_y, 0.70)),
            material=gs.materials.Rigid(friction=0.5)
        )
        drones.append(drone_entity)

    return drones, cargos

# =============================================================================
# 3. DYNAMIC ROAD TRAFFIC & MOVING AMBULANCE WITH SIREN
# =============================================================================
class DynamicRoadTraffic3D:
    """
    Manages 8 dynamic road vehicles moving through city corridors:
      - 4 Delivery Vans/Trucks traveling along north-south avenues
      - 3 Commercial Trucks traveling along east-west streets
      - 1 Emergency Ambulance (Ambulance_01) with an active directional siren
    """
    def __init__(self, scene, avenues_x, streets_y):
        self.avenues_x = avenues_x
        self.streets_y = streets_y
        self.vehicles = []

        # 4 Avenue Vehicles (North/South along Y)
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

        # 3 Commercial Trucks + 1 Emergency Ambulance (East/West along X)
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
        """Advances road traffic along urban street corridors."""
        for v in self.vehicles:
            v["pos"] += v["vel"] * dt
            if v["axis"] == "Y":
                if v["pos"][1] > v["bounds"][1]: v["pos"][1] = v["bounds"][0]
                elif v["pos"][1] < v["bounds"][0]: v["pos"][1] = v["bounds"][1]
            else:
                if v["pos"][0] > v["bounds"][1]: v["pos"][0] = v["bounds"][0]
                elif v["pos"][0] < v["bounds"][0]: v["pos"][0] = v["bounds"][1]
            
            # Sync kinematic state
            v["entity"].set_pos(tuple(v["pos"]))

    def get_emergency_ambulance_state(self):
        for v in self.vehicles:
            if v["is_emergency"]:
                return v["pos"].copy(), v["vel"].copy()
        return np.array([0.0, 0.0, 0.95], dtype=np.float32), np.zeros(3, dtype=np.float32)

# =============================================================================
# 4. CIVILIAN MEDEVAC AIRCRAFT (HIGH-ALTITUDE ADS-B TRAFFIC)
# =============================================================================
class CivilAircraftTraffic3D:
    """
    Simulates civilian air traffic (Medevac_N911) cruising through urban airspace:
      - Altitude: 62.0m (above drone cruise floors)
      - Speed: 13.5 m/s
      - Generates active ADS-B transponder closing speed & distance telemetry
    """
    def __init__(self, scene):
        self.pos = np.array([-180.0, 15.0, 62.0], dtype=np.float32)
        self.vel = np.array([13.5, 0.0, 0.0], dtype=np.float32)
        self.radius = 4.5

        # Visual fuselage for Medevac
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

# =============================================================================
# 5. DYNAMIC GROUND INTRUDER AGENT (DROPZONE SAFETY ABORT HAZARD)
# =============================================================================
class DynamicGroundIntruderAgent:
    """
    Simulates a wandering pedestrian/dog on the ground:
      - Wanders toward landing drop zones with curiosity
      - Personal space threshold (1.85m) triggers emergency delivery aborts
      - Flees when a cargo delivery aborts
    """
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

        # Flee if drone aborts, otherwise wander or investigate drop zone
        if is_aborting:
            flee_dir = -d_to_cargo / max(1e-4, dist)
            v_des = flee_dir * (self.v_max * 1.4)
            a_cmd = (v_des - self.v[:2]) * 4.5
        elif is_winch_lowering and dist < 12.0 and active_cargo_pos[2] < 6.0:
            # Curiously approach lowering payload
            approach_dir = d_to_cargo / max(1e-4, dist)
            v_des = approach_dir * 1.2
            a_cmd = (v_des - self.v[:2]) * 2.5
        else:
            # Random waypoint wandering
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

    """
SURVIVAL CORE V37.0 — FULL GENESIS MULTI-PHYSICS UPGRADE
PART 2 OF 4: CLOSED-LOOP 15-STATE ERROR-STATE EKF ON SO(3),
             QUANTUM PHASE ESTIMATOR & SWARM SENSOR FUSION BRIDGE
"""

import math
import numpy as np
import torch

# =============================================================================
# 0. SENSORY & ESTIMATION CONSTANTS
# =============================================================================
K_EFF_WAVEVECTOR = 1.61e7            # 87Rb Raman laser effective wavevector (1/m)
RAMAN_PULSE_SEP_T = 0.015            # Pulse separation time T (s)
GRAVITY_G = np.array([0.0, 0.0, -9.80665], dtype=np.float32)

P_0 = 1013.25                        # Sea level pressure (hPa)
T_0 = 288.15                         # Sea level temperature (K)
L_LAPSE = 0.0065                     # Temperature lapse rate (K/m)
R_SPEC_AIR = 287.058                 # Specific gas constant (J/(kg*K))

# =============================================================================
# 1. LIE ALGEBRA UTILITIES ON SO(3)
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
# 2. 15-STATE ERROR-STATE EKF ON SO(3)
# =============================================================================
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
        f_inertial = R @ a_unbiased + GRAVITY_G

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

        # Sage-Husa process noise update
        self.Q_d = self.Q_base * max(1.0, float(self.residual_scale))
        self.P = F_x @ self.P @ F_x.T + self.Q_d
        self.P = np.nan_to_num(self.P, nan=0.05)
        self.P = 0.5 * (self.P + self.P.T)

        # Covariance diagonal bounds clamp
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

        # Sage-Husa adaptation
        d_k = 0.05
        expected_gamma = float(len(innov_flat))
        self.residual_scale = (1.0 - d_k) * self.residual_scale + d_k * (gamma / max(1.0, expected_gamma))
        self.residual_scale = float(np.clip(self.residual_scale, 0.8, 5.0))

        # Chi-square innovation gating (Rejects spoofed GPS/sensor spikes)
        if gamma > chi2_threshold:
            return gamma, True  # Rejected

        delta_x = (K @ innovation).flatten()
        delta_x[0:6] = np.clip(delta_x[0:6], -3.0, 3.0)
        delta_x[6:9] = np.clip(delta_x[6:9], -0.20, 0.20)
        delta_x[9:15] = np.clip(delta_x[9:15], -0.40, 0.40)

        # Inject error state into nominal state
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

        # Joseph form covariance update
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
# 3. INDIVIDUAL DRONE PHYSICAL SENSOR & STATE ESTIMATOR UNIT
# =============================================================================
class SingleDroneStateEstimator:
    """
    Simulates the real hardware avionics and runs the local 15-state EKF:
      1. Corrupts ground-truth with physical sensor noise & biases
      2. Measures specific force with Matter-Wave Cold-Atom Gravimeter
      3. Propagates and updates the local EKF
      4. Supplies filtered state estimates (p_est, v_est, R_est, w_est) to the controller
    """
    def __init__(self, drone_id, init_pos):
        self.drone_id = drone_id
        self.ekf = ErrorStateEKF15(init_pos)

        # Hardware biases and calibration errors
        self.b_a_true = np.array([0.025, -0.018, 0.035], dtype=np.float32)
        self.b_g_true = np.array([0.006, -0.004, 0.003], dtype=np.float32)
        self.k_eff = K_EFF_WAVEVECTOR
        self.T = RAMAN_PULSE_SEP_T

    def step(self, true_p, true_v, true_q, true_ang, true_lin_accel, 
             motor_omegas, surface_z=0.0, wind_3d=None, is_spoofed=False, dt=0.005):
        if wind_3d is None:
            wind_3d = np.zeros(3, dtype=np.float32)

        R_true = quat_to_rot_np(true_q)

        # 1. Physical MEMS IMU Corrupted Measurement
        rpm_ratio = float(np.mean(motor_omegas)) / 963.4
        f_inertial = true_lin_accel - GRAVITY_G
        f_body = R_true.T @ f_inertial

        vib_accel = np.random.normal(0.0, 0.16 * rpm_ratio, 3).astype(np.float32)
        noise_accel = np.random.normal(0.0, 0.035, 3).astype(np.float32)
        raw_accel = f_body + self.b_a_true + vib_accel + noise_accel

        vib_gyro = np.random.normal(0.0, 0.015 * rpm_ratio, 3).astype(np.float32)
        noise_gyro = np.random.normal(0.0, 0.004, 3).astype(np.float32)
        raw_gyro = true_ang + self.b_g_true + vib_gyro + noise_gyro

        # 2. Matter-Wave Cold-Atom Gravimeter Phase Shift
        # Phase shift: Phi = k_eff * a_body * T^2
        phase_shift = self.k_eff * raw_accel * (self.T ** 2) + np.random.normal(0, 0.002, 3).astype(np.float32)
        a_quantum_body = phase_shift / (self.k_eff * (self.T ** 2))

        # 3. EKF Prediction Step using Cold-Atom Acceleration
        self.ekf.predict(a_quantum_body, raw_gyro, dt=dt)

        # 4. Compass Heading Measurement Update
        compass_yaw = math.atan2(R_true[1, 0], R_true[0, 0]) + float(np.random.normal(0, 0.012))
        self.ekf.update_compass_heading(compass_yaw, r_cov=0.02)

        # 5. Optical Flow Measurement Update
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
        v_flow_world = quat_to_rot_np(self.ekf.q) @ np.array([(w_flow[0] + raw_gyro[1])*z_slant, (w_flow[1] - raw_gyro[0])*z_slant, 0.0], dtype=np.float32)
        self.ekf.update_optical_flow(v_flow_world, flow_quality=flow_qual, z_slant=z_slant)

        # 6. Barometric Pressure Altitude Update
        p_static = P_0 * ((1.0 - (L_LAPSE * max(0.0, float(true_p[2]))) / T_0) ** 5.25588)
        z_baro = 44330.8 * (1.0 - ((p_static / P_0) ** 0.190284)) + float(np.random.normal(0, 0.04))
        v_air = true_v - wind_3d
        q_dyn = 0.5 * 1.225 * float(np.dot(v_air, v_air))
        self.ekf.update_barometer(z_baro, q_dyn=q_dyn)

        # 7. RTK-GNSS Measurement Update with Spoofing Rejection
        if is_spoofed:
            gps_offset = np.array([18.5, -24.0, 12.0], dtype=np.float32)  # Injected false offset
            p_gps = true_p + gps_offset + np.random.normal(0, 0.8, 3).astype(np.float32)
            hdop = 3.8
            cn0 = 28.5
        else:
            p_gps = true_p + np.random.normal(0, 0.06, 3).astype(np.float32)
            hdop = float(np.clip(0.85 + np.random.normal(0, 0.04), 0.6, 1.8))
            cn0 = float(np.clip(44.0 + np.random.normal(0, 1.0), 32.0, 48.0))

        gamma_gps, spoof_rejected = self.ekf.update_gps(p_gps, hdop=hdop, cn0_avg=cn0, is_spoofed=is_spoofed)

        # Extract filtered state
        p_est = self.ekf.p.copy()
        v_est = self.ekf.v.copy()
        q_est = self.ekf.q.copy()
        R_est = quat_to_rot_np(q_est)
        omega_est = raw_gyro - self.ekf.b_g
        cov_trace = float(np.trace(self.ekf.P[0:3, 0:3]))

        return {
            "p_est": p_est,
            "v_est": v_est,
            "q_est": q_est,
            "R_est": R_est,
            "omega_est": omega_est,
            "raw_accel": raw_accel,
            "raw_gyro": raw_gyro,
            "phase_shift": phase_shift,
            "cov_trace": cov_trace,
            "spoof_rejected": spoof_rejected,
            "gamma_gps": gamma_gps,
            "z_baro": z_baro,
            "w_flow": w_flow,
            "flow_qual": flow_qual
        }

# =============================================================================
# 4. SWARM CLOSED-LOOP EKF BRIDGE (MANAGES ALL 3 SWARM DRONES ON GPU)
# =============================================================================
class SwarmClosedLoopEKFBridge:
    """
    Connects the Genesis simulation and the avionics filter for all 3 swarm drones:
      - Takes batched GPU ground truth from Genesis
      - Runs independent 15-state EKF instances per drone
      - Returns filtered PyTorch CUDA tensors directly to the SE(3) flight controller
    """
    def __init__(self, initial_positions, device="cuda"):
        self.device = device
        self.num_drones = len(initial_positions)
        self.estimators = [
            SingleDroneStateEstimator(i, initial_positions[i]) 
            for i in range(self.num_drones)
        ]
        self.prev_v = [np.zeros(3, dtype=np.float32) for _ in range(self.num_drones)]

    def update_and_estimate(self, d_pos_tensor, d_vel_tensor, d_quat_tensor, d_ang_tensor,
                            motor_omegas_batch, surface_altitudes, wind_3d_np, is_spoofed=False, dt=0.005):
        """
        Accepts GPU tensors from Genesis:
          d_pos_tensor:  [num_drones, 3] (CUDA)
          d_vel_tensor:  [num_drones, 3] (CUDA)
          d_quat_tensor: [num_drones, 4] (CUDA)
          d_ang_tensor:  [num_drones, 3] (CUDA)
        Returns filtered estimated state tensors [num_drones, ...] on CUDA.
        """
        p_est_list, v_est_list, q_est_list, w_est_list = [], [], [], []
        diagnostics = []

        d_pos_cpu = d_pos_tensor.cpu().numpy()
        d_vel_cpu = d_vel_tensor.cpu().numpy()
        d_quat_cpu = d_quat_tensor.cpu().numpy()
        d_ang_cpu = d_ang_tensor.cpu().numpy()

        for i in range(self.num_drones):
            # Compute real kinematic linear acceleration from velocities
            a_lin = (d_vel_cpu[i] - self.prev_v[i]) / dt
            self.prev_v[i] = d_vel_cpu[i].copy()

            surf_z = float(surface_altitudes[i]) if isinstance(surface_altitudes, (list, np.ndarray)) else float(surface_altitudes)
            m_omegas = motor_omegas_batch[i] if isinstance(motor_omegas_batch, (list, np.ndarray)) else motor_omegas_batch

            # Run 15-state EKF on corrupted sensors
            diag = self.estimators[i].step(
                true_p=d_pos_cpu[i],
                true_v=d_vel_cpu[i],
                true_q=d_quat_cpu[i],
                true_ang=d_ang_cpu[i],
                true_lin_accel=a_lin,
                motor_omegas=m_omegas,
                surface_z=surf_z,
                wind_3d=wind_3d_np,
                is_spoofed=is_spoofed,
                dt=dt
            )

            p_est_list.append(diag["p_est"])
            v_est_list.append(diag["v_est"])
            q_est_list.append(diag["q_est"])
            w_est_list.append(diag["omega_est"])
            diagnostics.append(diag)

        # Convert filtered states back to PyTorch CUDA tensors for the flight controller
        p_est_tensor = torch.tensor(np.array(p_est_list), device=self.device, dtype=torch.float32)
        v_est_tensor = torch.tensor(np.array(v_est_list), device=self.device, dtype=torch.float32)
        q_est_tensor = torch.tensor(np.array(q_est_list), device=self.device, dtype=torch.float32)
        w_est_tensor = torch.tensor(np.array(w_est_list), device=self.device, dtype=torch.float32)

        return p_est_tensor, v_est_tensor, q_est_tensor, w_est_tensor, diagnostics

    """
SURVIVAL CORE V37.0 — FULL GENESIS MULTI-PHYSICS UPGRADE
PART 3 OF 4: DISTRIBUTED CBBA SWARM AUCTION DISPATCHER,
             PEER KINEMATIC TRACKER & RECIPROCAL MULTI-DRONE CBF-QP
"""

import math
from collections import deque
import numpy as np
import torch

# =============================================================================
# 0. SWARM & SAFETY CONSTANTS
# =============================================================================
CBF_ALPHA_OBSTACLE = 1.8
CBF_BASE_SAFE_RADIUS_M = 0.85
SWARM_INTER_DRONE_SAFE_RADIUS_M = 2.6
DOWNWASH_CONE_EXPANSION = 0.38
DOWNWASH_HAZARD_DEPTH_M = 7.5

CANDIDATE_CRUISE_ALTS_M = [34.0, 38.0, 44.0, 50.0]

# =============================================================================
# 1. DISTRIBUTED CBBA AUCTION & BREGUET WIND-SHEAR DISPATCHER
# =============================================================================
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

        # Evaluate candidate orders
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

        # Breguet Wind Layer Routing: Select optimal energy cruise altitude
        assigned_cruise_alt = 34.0 + (drone_id * 3.0)  # Safe default above all towers

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
        """Re-auctions an abandoned mission to the healthiest eligible peer drone."""
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
            print(f">>> [CBBA CONSENSUS] Task {abandoned_mission['bldg_id']} re-auctioned to Drone {best_drone_id+1} (Bid: {best_bid:.2f}) <<<")
        return best_drone_id

    def report_completed(self, drone_id):
        self.total_completed += 1

# =============================================================================
# 2. SWARM PEER KINEMATICS & DOWNWASH WAKE TRACKER
# =============================================================================
class SwarmKinematicsTracker:
    """
    Computes inter-drone spatial relationships across the 3 swarm drones:
      - Computes relative positions p_rel and velocities v_rel (for slots 118-129)
      - Computes downwash hazard cones beneath peer drones
    """
    def __init__(self, num_drones=3):
        self.num_drones = num_drones

    def compute_peer_kinematics(self, drone_positions, drone_velocities):
        """
        Accepts lists or numpy arrays of positions [num_drones, 3] and velocities [num_drones, 3].
        Returns:
          peer_info_per_drone: list of lists containing relative peer vectors for slots 118-129.
          downwash_hazards: list of active downwash hazard warnings.
        """
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

                # Check if drone i is in drone j's downwash wake
                dz = p_i[2] - p_j[2]
                dist_xy = float(np.linalg.norm(p_rel[:2]))
                if -DOWNWASH_HAZARD_DEPTH_M < dz < -0.4:
                    cone_r = SWARM_INTER_DRONE_SAFE_RADIUS_M + DOWNWASH_CONE_EXPANSION * abs(dz)
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
# 3. RECIPROCAL MULTI-DRONE SWARM CBF-QP SAFETY FILTER
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
        self.alpha_obs = CBF_ALPHA_OBSTACLE
        self.safe_radius = CBF_BASE_SAFE_RADIUS_M
        self.swarm_r = SWARM_INTER_DRONE_SAFE_RADIUS_M
        self.downwash_expansion = DOWNWASH_CONE_EXPANSION
        self.downwash_depth = DOWNWASH_HAZARD_DEPTH_M

    def filter_swarm_accelerations(self, drone_positions, drone_velocities, nominal_accels,
                                   obstacles, wind_2d=None, ekf_cov_traces=None,
                                   target_bldg_ids=None):
        """
        Accepts:
          drone_positions:  [3, 3] numpy array
          drone_velocities: [3, 3] numpy array
          nominal_accels:   [3, 3] numpy array
          obstacles:        list of building obstacle dicts
        Returns:
          safe_accels:      [3, 3] numpy array of safety-projected accelerations
        """
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

            A_rows = []
            b_rows = []

            cov_inflation = 1.2 * min(2.5, math.sqrt(max(0.0, float(cov_trace))))
            gust_inflation = 0.22 * float(np.linalg.norm(wind_2d))
            dynamic_safe_r = self.safe_radius + cov_inflation + gust_inflation

            # 1. Building Obstacle Barriers
            for obs in obstacles:
                if obs.get("id") == target_id and p_i[2] < (obs.get("height", 0.0) + 12.0):
                    continue  # Target helipad approached via chimney

                rel = p_i - obs["pos"]
                dist_xy = math.hypot(rel[0], rel[1])
                obs_h = obs.get("height", 0.0)
                n_xy = rel[:2] / max(1e-4, dist_xy)
                v_closing = -float(np.dot(v_i[:2], n_xy))
                tailwind = -float(np.dot(wind_2d[:2], n_xy))

                stopping_buffer = (max(0.0, v_closing) ** 2) / 7.6 + max(0.0, tailwind) * 0.35
                eff_safe_r = dynamic_safe_r + min(4.5, stopping_buffer)
                h_obs = dist_xy - (obs.get("radius", 8.0) + eff_safe_r)
                eff_alpha = self.alpha_obs * (8.0 / (8.0 + max(0.0, v_closing)))

                if p_i[2] <= (obs_h - 0.4) and h_obs < 4.5:
                    grad = np.array([n_xy[0], n_xy[1], 0.0], dtype=np.float32)
                    v_tangent = v_i[0] * (-n_xy[1]) + v_i[1] * n_xy[0]
                    centripetal = (v_tangent ** 2) / max(0.5, dist_xy)
                    l_fh = float(np.dot(grad, v_i))
                    A_rows.append(grad)
                    b_rows.append(-l_fh - eff_alpha * h_obs - centripetal)

                if dist_xy <= (obs.get("radius", 8.0) + 0.5) and p_i[2] >= (obs_h - 0.2):
                    h_roof = p_i[2] - (obs_h + 0.15)
                    if h_roof < 0.8:
                        A_rows.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
                        b_rows.append(-v_i[2] - 2.0 * h_roof)

            # 2. Inter-Drone Swarm & Downwash Barriers
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

            # 3. Ground Cushion & Geofence
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

            # Quadratic Programming Relaxation Loop
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

    """
SURVIVAL CORE V37.0 — COMPLETE GENESIS MULTI-PHYSICS UPGRADE
PART 4 OF 4: MASTER 3-DRONE SWARM EXECUTION ENGINE & LIVE 3D COCKPIT
Unites:
  - Part 1: swarm_agents_world.py (3 Drones, 3 Cargos, Traffic, Medevac, Intruder)
  - Part 2: closed_loop_ekf_bridge.py (Closed-Loop 15-State EKF & Quantum Sensing)
  - Part 3: swarm_cbba_cbf_coordination.py (CBBA Auction, Downwash & CBF-QP)
=============================================================================
Usage:
  Step 1 (Train):   python run_full_genesis_swarm.py --train
  Step 2 (View 3D): python run_full_genesis_swarm.py
=============================================================================
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



# Physical Constants
DRY_MASS_KG = 1.35
CARGO_MASS_KG = 0.45
TOTAL_MASS_KG = DRY_MASS_KG + CARGO_MASS_KG
GRAVITY_VAL = 9.80665
ARM_LENGTH_M = 0.26 / math.sqrt(2.0)
ROTOR_RADIUS_M = 0.127
ROTOR_DISK_AREA = math.pi * (ROTOR_RADIUS_M ** 2)
C_T_CLEAN = 1.35e-5
C_Q_CLEAN = 2.40e-7
COAXIAL_EFFICIENCY = 0.92
MOTOR_TAU_S = 0.030
MAX_OMEGA = (9200.0 * 2.0 * math.pi) / 60.0
HOVER_THRUST = TOTAL_MASS_KG * GRAVITY_VAL
MIN_SAFE_CRUISE_ALT_M = 34.0

BASE_OBS_DIM = 139
HISTORY_LEN = 3
OBSERVATION_DIM = BASE_OBS_DIM * HISTORY_LEN
NUM_FSM_STATES = 7
MODEL_FILE = "drone_master_brain_swarm_v1.pth"

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
# 1. GPU TENSOR ROTATION & BEMT ACTUATOR MIXER
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
# 2. 6-HEAD DEEP DENSE FLIGHT EXECUTIVE RESNET BRAIN
# =============================================================================
class DenseResBlock(nn.Module):
    def __init__(self, dim=256, hidden_dim=384):
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
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.SiLU()
        )
        self.block1 = DenseResBlock(256, 384)
        self.block2 = DenseResBlock(256, 384)

        self.guidance_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 4),
            nn.Tanh()
        )
        self.fsm_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, NUM_FSM_STATES)
        )
        self.winch_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
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
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        feat = self.block2(self.block1(self.input_proj(x)))
        guidance = self.guidance_head(feat) * 2.0
        fsm_logits = self.fsm_head(feat)
        winch_cmd = self.winch_head(feat)
        health_pred = self.health_head(feat)
        trust_scores = self.trust_head(feat)
        value = self.value_head(feat)
        return guidance, fsm_logits, winch_cmd, health_pred, trust_scores, value

# =============================================================================
# 3. 139-DIMENSIONAL OBSERVATION SYNTHESIZER
# =============================================================================
class SwarmObservationSynthesizer:
    def __init__(self, num_drones=3):
        self.num_drones = num_drones
        self.temporal_queues = [deque(maxlen=HISTORY_LEN) for _ in range(num_drones)]

    def synthesize_drone_obs(self, p_est, v_est, R_est, omega_est, target_pos,
                             total_wind, cable_len, c_vel, d_vel, c_pos, d_pos,
                             has_payload, tension_mag, bms_soc, bms_v_pack, bms_temp_c, bms_r0,
                             ice_idx, raw_accel, raw_gyro, adsb_rel, adsb_spd,
                             peer_kinematics, is_grounded):
        out = np.zeros(BASE_OBS_DIM, dtype=np.float32)

        out[0:3] = (target_pos - p_est) / 50.0
        out[3:6] = p_est / 200.0
        out[6:9] = v_est / 8.0
        out[9:18] = R_est.flatten()
        out[18:21] = omega_est / 10.0
        out[21:37] = 25.0 / 30.0  # LiDAR range surrogate
        out[37] = max(0.1, float(p_est[2])) / 15.0
        out[38] = float(np.linalg.norm(v_est - total_wind)) / 20.0
        out[39] = 14.0 / 100.0
        out[40] = 40.0 / 50.0
        out[41] = 25.0 / 50.0
        out[42:45] = total_wind / 8.0
        out[45:53] = 12.0 / 15.0
        out[53] = cable_len / 15.0
        out[54] = float(np.linalg.norm(c_vel - d_vel)) / 5.0
        out[55] = tension_mag / 50.0
        sway_2d = (c_pos[:2] - d_pos[:2]) if has_payload else np.zeros(2, dtype=np.float32)
        out[56:58] = sway_2d / 2.0
        out[58] = 1.0 if has_payload else 0.0
        out[59:61] = (0.5 * TOTAL_MASS_KG * 9.81 / 20.0) if is_grounded else 0.0
        out[61] = bms_soc
        out[62] = bms_v_pack / 25.2
        out[63] = bms_temp_c / 60.0
        out[64] = bms_r0 / 0.05
        out[65] = ice_idx
        out[66] = 1.0 if is_grounded else 0.0
        out[67] = 1.2 / 10.0
        out[68] = float(np.linalg.norm(raw_accel[:2])) / 16.0
        out[69:77] = 7.5 / 25.0   # ESC RMS Current
        out[77:85] = 38.0 / 100.0 # ESC MOSFET Temp
        out[85:88] = raw_accel / 20.0
        out[88:91] = raw_gyro / 10.0
        out[91:94] = np.array([0, 0, -tension_mag], dtype=np.float32) / 50.0
        out[94] = 0.0
        out[95] = 0.0
        out[96] = 65.0 / 100.0
        out[97] = 22.0 / 100.0
        out[98] = 0.6 / 10.0
        out[99] = 0.18
        out[100] = 0.85 / 5.0
        out[101] = 44.0 / 50.0
        out[102] = (-110.0 + 120.0) / 70.0
        out[103] = 0.02
        out[104:106] = 0.0
        out[106] = 0.95
        out[107] = 0.0
        out[108:111] = 0.0
        out[111] = 1.0 if p_est[2] < 15.0 else 0.0
        out[112] = 0.0
        out[113] = 52.0 / 100.0
        out[114:117] = adsb_rel / 200.0
        out[117] = adsb_spd / 30.0

        # Fill slots 118-129 with Peer Swarm Kinematics
        p_idx = 118
        for peer in peer_kinematics[:2]:
            out[p_idx:p_idx+3] = peer["p_rel"] / 50.0
            out[p_idx+3:p_idx+6] = peer["v_rel"] / 10.0
            p_idx += 6

        out[130:133] = raw_accel / 30.0
        out[133:136] = raw_gyro / 10.0
        out[136] = 0.25
        out[137] = 4.5 / 100.0
        out[138] = 52.0 / 100.0
        return out

    def update_history_stack(self, drone_idx, obs_139):
        q = self.temporal_queues[drone_idx]
        if len(q) == 0:
            for _ in range(HISTORY_LEN):
                q.append(obs_139.copy())
        else:
            q.append(obs_139.copy())
        return np.concatenate(list(q), axis=0).astype(np.float32)

# =============================================================================
# 4. PARALLEL GPU SWARM TRAINING PIPELINE (64 ENVS)
# =============================================================================
def train_swarm_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*75)
    print("STAGE 1: 3-DRONE SWARM MASSIVE GPU HARVESTING & RESNET PRE-TRAINING")
    print("="*75)

    gs.init(backend=gs.gpu)
    dt = 0.005  # 200 Hz
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=2, gravity=(0, 0, -GRAVITY_VAL)),
        show_viewer=False
    )

    manifest, depot_pads, avenues_x, streets_y = build_metropolitan_city_with_swarm_depot(scene)
    drones, cargos = spawn_swarm_drones_and_cargo(scene, depot_pads)

    n_envs = 64
    scene.build(n_envs=n_envs)
    print(f"\n[GPU HARVEST] Simulating 3-Drone Swarm across {n_envs} parallel environments on {torch.cuda.get_device_name(0)}...")

    bemt = VectorizedBEMTEngine(num_drones=3, device="cuda")
    ctrl = GeometricSE3Controller(num_drones=3, device="cuda")
    dispatcher = SwarmCBBAAuctionDispatcher(manifest, depot_pads)
    cbf_filter = ReciprocalMultiDroneCBFQPFilter()
    kinematics_tracker = SwarmKinematicsTracker(num_drones=3)
    obs_manager = SwarmObservationSynthesizer(num_drones=3)

    # Initial CBBA Mission Assignments for the 3 drones
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

        # Read Ground-Truth Kinematics across the 3 drones in env 0
        d_pos_list = [drones[i].get_pos()[0].cpu().numpy() for i in range(3)]
        d_vel_list = [drones[i].get_vel()[0].cpu().numpy() for i in range(3)]
        d_quat_list = [drones[i].get_quat()[0] for i in range(3)]
        d_ang_list = [drones[i].get_ang()[0] for i in range(3)]

        c_pos_list = [cargos[i].get_pos()[0].cpu().numpy() for i in range(3)]
        c_vel_list = [cargos[i].get_vel()[0].cpu().numpy() for i in range(3)]

        peer_kinematics, downwash_hazards = kinematics_tracker.compute_peer_kinematics(d_pos_list, d_vel_list)

        # Guidance targets per drone
        targets_gpu = []
        nominal_accels = []

        for i in range(3):
            mission = active_missions[i]
            tgt_xy = mission["target_pos"][:2]
            roof_h = mission["roof_height"]
            cruise_z = mission["assigned_cruise_alt"]
            curr_pos = d_pos_list[i]

            # FSM Waypoint Selection
            if fsm_phases[i] == 0:  # TAKEOFF
                t_pos = np.array([depot_pads[i][0], depot_pads[i][1], cruise_z], dtype=np.float32)
                if curr_pos[2] >= (cruise_z - 3.0) or sim_time >= 8.0:
                    fsm_phases[i] = 1
            elif fsm_phases[i] == 1:  # CRUISE OUTBOUND
                t_pos = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                if float(np.linalg.norm(curr_pos[:2] - tgt_xy)) <= 3.5:
                    fsm_phases[i] = 2
            elif fsm_phases[i] == 2:  # CHIMNEY DESCENT
                t_pos = np.array([tgt_xy[0], tgt_xy[1], roof_h + 7.5], dtype=np.float32)
                if abs(curr_pos[2] - (roof_h + 7.5)) <= 0.8:
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
                if float(np.linalg.norm(curr_pos[:2] - depot_pads[i][:2])) <= 3.0:
                    fsm_phases[i] = 6
            else:  # DEPOT LANDING
                t_pos = np.array([depot_pads[i][0], depot_pads[i][1], 0.70], dtype=np.float32)

            targets_gpu.append(t_pos)

            # Compute nominal acceleration
            delta_p = t_pos - curr_pos
            v_cmd_xy = np.clip(delta_p[:2] * 1.4, -6.0, 6.0)
            v_cmd_z = np.clip(delta_p[2] * 1.2, -1.5, 2.4)
            v_des = np.array([v_cmd_xy[0], v_cmd_xy[1], v_cmd_z], dtype=np.float32)
            a_nom = np.array([2.0, 2.0, 2.5], dtype=np.float32) * (v_des - d_vel_list[i])
            nominal_accels.append(a_nom)

        # Reciprocal Multi-Drone Swarm CBF-QP Safety Barrier Filter
        safe_accels = cbf_filter.filter_swarm_accelerations(
            drone_positions=np.array(d_pos_list),
            drone_velocities=np.array(d_vel_list),
            nominal_accels=np.array(nominal_accels),
            obstacles=manifest,
            wind_2d=np.array([1.5, -0.5], dtype=np.float32),
            target_bldg_ids=[active_missions[i]["bldg_id"] for i in range(3)]
        )

        # Execute Control & Physics for each drone
        targets_tensor = torch.tensor(np.array(targets_gpu), device="cuda", dtype=torch.float32)
        pos_t = torch.stack([drones[i].get_pos()[0] for i in range(3)])
        vel_t = torch.stack([drones[i].get_vel()[0] for i in range(3)])
        quat_t = torch.stack([d_quat_list[i] for i in range(3)])
        ang_t = torch.stack([d_ang_list[i] for i in range(3)])

        masses = torch.tensor([TOTAL_MASS_KG if has_payloads[i] else DRY_MASS_KG for i in range(3)], device="cuda")
        safe_accels_t = torch.tensor(safe_accels, device="cuda", dtype=torch.float32)

        f_thrust_cmd, tau_cmd = ctrl.compute_wrench(
            pos_t, vel_t, quat_t, ang_t, targets_tensor, masses.unsqueeze(-1),
            nn_guidance=safe_accels_t - torch.tensor(nominal_accels, device="cuda")
        )

        motor_cmds = bemt.allocate(f_thrust_cmd, tau_cmd)
        actual_thrust, actual_tau = bemt.compute_aerodynamics(motor_cmds, pos_t, vel_t, target_surf_z=0.0, dt=dt)
        R_curr = quat_to_rot_batch(quat_t)

        for i in range(3):
            # Tether tension
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

            tot_f = R_curr[i, :, 2] * actual_thrust[i] + f_tether_drone
            drones[i].links[0].apply_external_force(force=tot_f.unsqueeze(0))
            drones[i].links[0].apply_external_torque(torque=actual_tau[i].unsqueeze(0))

        scene.step()

        # Harvest observations from Drone 0
        if step % 2 == 0:
            obs_139 = obs_manager.synthesize_drone_obs(
                p_est=d_pos_list[0], v_est=d_vel_list[0],
                R_est=R_curr[0].cpu().numpy(), omega_est=d_ang_list[0].cpu().numpy(),
                target_pos=targets_gpu[0], total_wind=np.array([1.5, -0.5, 0.0], dtype=np.float32),
                cable_len=cable_lens[0], c_vel=c_vel_list[0], d_vel=d_vel_list[0],
                c_pos=c_pos_list[0], d_pos=d_pos_list[0], has_payload=has_payloads[0],
                tension_mag=12.0, bms_soc=0.95, bms_v_pack=24.8, bms_temp_c=22.0, bms_r0=0.009,
                ice_idx=0.0, raw_accel=np.array([0, 0, 9.81], dtype=np.float32),
                raw_gyro=d_ang_list[0].cpu().numpy(), adsb_rel=np.array([-120.0, 20.0, 58.0], dtype=np.float32),
                adsb_spd=0.0, peer_kinematics=peer_kinematics[0], is_grounded=False
            )
            obs_417 = obs_manager.update_history_stack(0, obs_139)
            collected_obs.append(torch.from_numpy(obs_417).float().to(device))

            guidance_gt = torch.clamp((targets_tensor[0] - pos_t[0]) / 5.0, -1.0, 1.0)
            guidance_4d = torch.cat([guidance_gt, torch.zeros(1, device=device)])
            fsm_gt = torch.tensor(fsm_phases[0], device=device, dtype=torch.long)
            health_gt = torch.ones(8, device=device)
            trust_gt = torch.tensor([1.0, 1.0], device=device)
            dist_gt = torch.norm(targets_tensor[0] - pos_t[0], keepdim=True)
            winch_gt = torch.tensor([1.0 if fsm_phases[0] == 3 else 0.0, 1.0 if not has_payloads[0] else 0.0], device=device)

            collected_targets.append((guidance_4d, fsm_gt, winch_gt, health_gt, trust_gt, dist_gt))

    print(f"\n[DATASET READY] Harvested {len(collected_obs):,} state transitions in {time.time()-t_start:.2f}s!")

    # Multi-Task 6-Head ResNet Training
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

    print(f"\n[TRAINING] Training 6-head ResNet policy ({epochs} epochs)...")
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

    torch.save(model.state_dict(), MODEL_FILE)
    print(f"\n[SUCCESS] Model weights saved to '{MODEL_FILE}'.")

# =============================================================================
# 5. LIVE 3D COCKPIT (3 DRONES IN SHARED CITY + MEDEVAC + TRAFFIC + EKF)
# =============================================================================
def evaluate_swarm_system():
    if not os.path.exists(MODEL_FILE):
        print(f"\n[WARNING] '{MODEL_FILE}' not found! Running parallel GPU training first...")
        train_swarm_system()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    print(f"\nLoaded verified 6-Head Swarm Policy from '{MODEL_FILE}'!")

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

    # 1. Spawn City, 3 Drones, 3 Cargos
    manifest, depot_pads, avenues_x, streets_y = build_metropolitan_city_with_swarm_depot(scene)
    drones, cargos = spawn_swarm_drones_and_cargo(scene, depot_pads)

    # 2. Spawn Dynamic Agents
    road_traffic = DynamicRoadTraffic3D(scene, avenues_x, streets_y)
    medevac_plane = CivilAircraftTraffic3D(scene)
    ground_intruder = DynamicGroundIntruderAgent(scene, init_pos=(-25.0, 25.0, 0.4))

    # All 3 drones share the exact same 3D city scene (n_envs=1)
    scene.build(n_envs=1)

    # 3. Instantiate Autonomy & Estimation Bridges
    bemt = VectorizedBEMTEngine(num_drones=3, device="cuda")
    ctrl = GeometricSE3Controller(num_drones=3, device="cuda")
    dispatcher = SwarmCBBAAuctionDispatcher(manifest, depot_pads)
    cbf_filter = ReciprocalMultiDroneCBFQPFilter()
    kinematics_tracker = SwarmKinematicsTracker(num_drones=3)
    obs_manager = SwarmObservationSynthesizer(num_drones=3)

    # Closed-Loop 15-State EKF Bridge for all 3 drones
    ekf_bridge = SwarmClosedLoopEKFBridge(
        initial_positions=[depot_pads[0], depot_pads[1], depot_pads[2]],
        device="cuda"
    )

    # Dispatch Initial Missions via CBBA Auction
    active_missions = [
        dispatcher.dispatch_next_mission(0, drone_soc=0.98, drone_pos=depot_pads[0]),
        dispatcher.dispatch_next_mission(1, drone_soc=0.98, drone_pos=depot_pads[1]),
        dispatcher.dispatch_next_mission(2, drone_soc=0.98, drone_pos=depot_pads[2])
    ]

    fsm_phases = [0, 0, 0]
    cable_lens = [0.55, 0.55, 0.55]
    has_payloads = [True, True, True]

    print("\n" + "="*75)
    print("V37.0 FULL 3-DRONE SWARM AIRSPACE ONLINE")
    print(f"D1 Target: {active_missions[0]['bldg_id']} | D2 Target: {active_missions[1]['bldg_id']} | D3 Target: {active_missions[2]['bldg_id']}")
    print("Dynamic Agents: Medevac N911 @ 62m | 8 Road Vehicles | Pedestrian Intruder")
    print("State Estimation: Closed-Loop 15-State Error-State EKF on SO(3)")
    print("="*75 + "\n")

    try:
        for step in range(5000):
            if scene.viewer is not None and not scene.viewer.is_alive():
                print("\n[INFO] Viewer window closed by user. Exiting cleanly...")
                break

            sim_time = step * dt

            # 1. Step Dynamic Urban Agents
            road_traffic.step(dt=dt)
            medevac_plane.step(dt=dt)
            amb_pos, amb_vel = road_traffic.get_emergency_ambulance_state()

            # Intruder approaches active drop zone of Drone 0
            intruder_p, is_breach, _ = ground_intruder.step(
                active_cargo_pos=cargos[0].get_pos()[0].cpu().numpy(),
                is_winch_lowering=(fsm_phases[0] == 3),
                is_aborting=(is_breach and fsm_phases[0] == 3),
                dt=dt
            )

            # 2. Read Ground Truth from Genesis
            d_pos_raw = torch.stack([drones[i].get_pos()[0] for i in range(3)])
            d_vel_raw = torch.stack([drones[i].get_vel()[0] for i in range(3)])
            d_quat_raw = torch.stack([drones[i].get_quat()[0] for i in range(3)])
            d_ang_raw = torch.stack([drones[i].get_ang()[0] for i in range(3)])

            # 3. CLOSED-LOOP 15-STATE EKF STATE ESTIMATION (Replaces Ground Truth!)
            p_est_t, v_est_t, q_est_t, w_est_t, diagnostics = ekf_bridge.update_and_estimate(
                d_pos_tensor=d_pos_raw,
                d_vel_tensor=d_vel_raw,
                d_quat_tensor=d_quat_raw,
                d_ang_tensor=d_ang_raw,
                motor_omegas_batch=bemt.motor_omegas.cpu().numpy(),
                surface_altitudes=[0.0, 0.0, 0.0],
                wind_3d_np=np.array([1.5, -0.5, 0.0], dtype=np.float32),
                is_spoofed=False,
                dt=dt
            )

            p_est_np = p_est_t.cpu().numpy()
            v_est_np = v_est_t.cpu().numpy()

            # 4. Peer Kinematics & Downwash Wake Tracking
            peer_kinematics, downwash_hazards = kinematics_tracker.compute_peer_kinematics(p_est_np, v_est_np)

            # 5. Mission State Machine & Target Determination per Drone
            targets_list = []
            nominal_accels = []

            for i in range(3):
                mission = active_missions[i]
                tgt_xy = mission["target_pos"][:2]
                roof_h = mission["roof_height"]
                cruise_z = mission["assigned_cruise_alt"]
                curr_p = p_est_np[i]

                # Hierarchical Arbiter Transitions
                dist_xy = float(np.linalg.norm(curr_p[:2] - tgt_xy))
                dist_home = float(np.linalg.norm(curr_p[:2] - depot_pads[i][:2]))

                if fsm_phases[i] == 0:  # TAKEOFF
                    t_pos = np.array([depot_pads[i][0], depot_pads[i][1], cruise_z], dtype=np.float32)
                    if curr_p[2] >= (cruise_z - 3.0) or sim_time > 8.0:
                        fsm_phases[i] = 1
                elif fsm_phases[i] == 1:  # CRUISE OUTBOUND
                    t_pos = np.array([tgt_xy[0], tgt_xy[1], cruise_z], dtype=np.float32)
                    if dist_xy <= 3.5:
                        fsm_phases[i] = 2
                elif fsm_phases[i] == 2:  # CHIMNEY DESCENT
                    t_pos = np.array([tgt_xy[0], tgt_xy[1], roof_h + 7.5], dtype=np.float32)
                    if abs(curr_p[2] - (roof_h + 7.5)) <= 0.8:
                        fsm_phases[i] = 3
                elif fsm_phases[i] == 3:  # DELIVERY HOVER
                    t_pos = np.array([tgt_xy[0], tgt_xy[1], roof_h + 7.5], dtype=np.float32)
                    # Abort if ground intruder breaches
                    if is_breach and i == 0:
                        fsm_phases[i] = 4  # Abort retract
                    elif has_payloads[i] and cable_lens[i] < 7.2:
                        cable_lens[i] += 1.8 * dt
                    else:
                        has_payloads[i] = False
                        fsm_phases[i] = 4
                        print(f"[{sim_time:4.1f}s] TOUCHDOWN! Drone {i+1} delivered cargo on {mission['bldg_id']}!")
                elif fsm_phases[i] == 4:  # WINCH RETRACT
                    t_pos = np.array([tgt_xy[0], tgt_xy[1], roof_h + 7.5], dtype=np.float32)
                    cable_lens[i] = max(0.20, cable_lens[i] - 3.2 * dt)
                    if cable_lens[i] <= 0.25:
                        fsm_phases[i] = 5
                elif fsm_phases[i] == 5:  # RETURN TO BASE
                    t_pos = np.array([depot_pads[i][0], depot_pads[i][1], cruise_z], dtype=np.float32)
                    if dist_home <= 3.0:
                        fsm_phases[i] = 6
                else:  # DEPOT LANDING
                    t_pos = np.array([depot_pads[i][0], depot_pads[i][1], 0.70], dtype=np.float32)

                targets_list.append(t_pos)

                delta_p = t_pos - curr_p
                v_cmd_xy = np.clip(delta_p[:2] * 1.4, -6.0, 6.0)
                v_cmd_z = np.clip(delta_p[2] * 1.2, -1.5, 2.4)
                v_des = np.array([v_cmd_xy[0], v_cmd_xy[1], v_cmd_z], dtype=np.float32)
                nominal_accels.append(np.array([2.0, 2.0, 2.5], dtype=np.float32) * (v_des - v_est_np[i]))

            # 6. Reciprocal Multi-Drone Swarm CBF-QP Safety Barrier Filter
            safe_accels = cbf_filter.filter_swarm_accelerations(
                drone_positions=p_est_np,
                drone_velocities=v_est_np,
                nominal_accels=np.array(nominal_accels),
                obstacles=manifest,
                wind_2d=np.array([1.5, -0.5], dtype=np.float32),
                ekf_cov_traces=[d["cov_trace"] for d in diagnostics],
                target_bldg_ids=[active_missions[i]["bldg_id"] for i in range(3)]
            )

            # 7. SE(3) Flight Control using ESTIMATED States
            targets_tensor = torch.tensor(np.array(targets_list), device="cuda", dtype=torch.float32)
            masses = torch.tensor([TOTAL_MASS_KG if has_payloads[i] else DRY_MASS_KG for i in range(3)], device="cuda")
            safe_accels_t = torch.tensor(safe_accels, device="cuda", dtype=torch.float32)

            f_thrust_cmd, tau_cmd = ctrl.compute_wrench(
                p_est_t, v_est_t, q_est_t, w_est_t, targets_tensor, masses.unsqueeze(-1),
                nn_guidance=safe_accels_t - torch.tensor(nominal_accels, device="cuda")
            )

            motor_cmds = bemt.allocate(f_thrust_cmd, tau_cmd)
            actual_thrust, actual_tau = bemt.compute_aerodynamics(motor_cmds, p_est_t, v_est_t, target_surf_z=0.0, dt=dt)
            R_curr = quat_to_rot_batch(q_est_t)

            # 8. Apply Forces & Tether Dynamics to Genesis
            for i in range(3):
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

                tot_f = R_curr[i, :, 2] * actual_thrust[i] + f_tether_drone
                drones[i].links[0].apply_external_force(force=tot_f.unsqueeze(0))
                drones[i].links[0].apply_external_torque(torque=actual_tau[i].unsqueeze(0))

            scene.step()

            # Console Telemetry Output
            if step % 400 == 0:
                adsb_p, adsb_spd, adsb_d = medevac_plane.poll_adsb(p_est_np[0], v_est_np[0])
                print(f"[{sim_time:4.1f}s] D1: {FSM_NAMES[fsm_phases[0]]:<12} ({p_est_np[0,2]:4.1f}m) | D2: {FSM_NAMES[fsm_phases[1]]:<12} ({p_est_np[1,2]:4.1f}m) | D3: {FSM_NAMES[fsm_phases[2]]:<12} ({p_est_np[2,2]:4.1f}m)")
                print(f"       ADS-B Medevac: {adsb_d:4.1f}m | Intruder: {intruder_p[0]:4.1f}, {intruder_p[1]:4.1f} | Downwash Alerts: {len(downwash_hazards)}")

    except gs.GenesisException as e:
        if "Viewer closed" in str(e):
            print("\n[INFO] Viewer window closed by user. Simulation exited cleanly.")
        else:
            raise e

# =============================================================================
# 6. MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_swarm_system()
    else:
        evaluate_swarm_system()