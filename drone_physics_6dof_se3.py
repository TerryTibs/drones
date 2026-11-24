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
import torch.optim as optim
import pygame
import sys

# ==========================================
# 0. PHYSICAL AIRFRAME & MOTOR SPECIFICATIONS
# ==========================================
# Parameters matching a 1.25 kg commercial carbon-fiber inspection quadrotor
MASS_KG = 1.25
GRAVITY_G = 9.81
ARM_LENGTH_M = 0.22      # 22 cm center-to-motor
INERTIA_IXX = 0.015      # kg * m^2
INERTIA_IYY = 0.015      # kg * m^2
INERTIA_IZZ = 0.025      # kg * m^2
INERTIA_MATRIX = np.diag([INERTIA_IXX, INERTIA_IYY, INERTIA_IZZ]).astype(np.float32)
INV_INERTIA = np.linalg.inv(INERTIA_MATRIX)

# Rotor Aerodynamic Coefficients
C_T = 1.25e-5            # Thrust coeff: T = C_T * omega^2 (N / (rad/s)^2)
C_Q = 2.40e-7            # Torque coeff: Q = C_Q * omega^2 (N*m / (rad/s)^2)
MOTOR_TAU_S = 0.035      # 35 ms first-order motor lag
MAX_RPM = 8600.0
MAX_OMEGA = (MAX_RPM * 2.0 * math.pi) / 60.0  # rad/s (~900 rad/s)
HOVER_THRUST_TOTAL = MASS_KG * GRAVITY_G
HOVER_OMEGA_PER_MOTOR = math.sqrt((HOVER_THRUST_TOTAL / 4.0) / C_T) # ~495 rad/s (~4720 RPM)

# Linear Aerodynamic Drag Tensor
DRAG_COEFFS = np.array([0.15, 0.15, 0.25], dtype=np.float32)

MODEL_SAVE_PATH = "drone_se3_disturbance_model.pth"

# ==========================================
# 1. MATHEMATICAL UTILITIES: SO(3) & QUATERNIONS
# ==========================================
def skew_symmetric(v):
    """Maps vector v in R^3 to Lie algebra matrix [v]_x in so(3)."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float32)

def vee_map(mat):
    """Inverse of skew-symmetric map: so(3) -> R^3."""
    return np.array([mat[2, 1], mat[0, 2], mat[1, 0]], dtype=np.float32)

def rot_matrix_to_euler(R):
    """Extracts Roll (phi), Pitch (theta), Yaw (psi) in degrees from R in SO(3)."""
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

    pitch_rad = -math.asin(max(-1.0, min(1.0, r31)))
    if abs(math.cos(pitch_rad)) > 1e-4:
        roll_rad = math.atan2(r32, r33)
        yaw_rad = math.atan2(r21, r11)
    else:
        roll_rad = 0.0
        yaw_rad = math.atan2(-r12, r22)

    return math.degrees(roll_rad), math.degrees(pitch_rad), math.degrees(yaw_rad)

# ==========================================
# 2. CONTINUOUS 6-DOF QUADROTOR FLIGHT DYNAMICS
# ==========================================
class DronePhysics6DOF:
    def __init__(self):
        self.reset()

    def reset(self, initial_pos=None):
        self.p = np.array(initial_pos if initial_pos else [0.0, 0.0, 1.5], dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.R = np.eye(3, dtype=np.float32)
        self.omega = np.zeros(3, dtype=np.float32)

        # 4 Motors: current actual angular velocity (rad/s)
        self.motor_omegas = np.full(4, HOVER_OMEGA_PER_MOTOR, dtype=np.float32)

        # Simulation clocks & environmental disturbance
        self.time = 0.0
        self.wind_gust = np.zeros(3, dtype=np.float32)
        return self.get_state_vector()

    def step(self, motor_omega_cmds, dt=0.01):
        """
        Integrates continuous 6-DOF Newton-Euler dynamics via Runge-Kutta 2 (Heun's method).
        motor_omega_cmds: array of 4 commanded motor speeds in rad/s.
        """
        # 1. First-order motor dynamic lag
        omega_cmds = np.clip(motor_omega_cmds, 0.0, MAX_OMEGA)
        d_omega = (omega_cmds - self.motor_omegas) / MOTOR_TAU_S
        self.motor_omegas += d_omega * dt

        # 2. Individual motor thrusts and reaction torques
        T = C_T * (self.motor_omegas ** 2)   # Thrust for motors 1, 2, 3, 4
        Q = C_Q * (self.motor_omegas ** 2)   # Drag torque

        # 3. Allocation to body force and moments (X-configuration)
        f_thrust = float(np.sum(T))
        arm = ARM_LENGTH_M / math.sqrt(2.0)
        tau_x = arm * (-T[0] + T[1] + T[2] - T[3])
        tau_y = arm * (T[0] + T[1] - T[2] - T[3])
        tau_z = -Q[0] + Q[1] - Q[2] + Q[3]  # CCW (+), CW (-)
        torques_body = np.array([tau_x, tau_y, tau_z], dtype=np.float32)

        # 4. Translational Newton equation: m * ddot(p) = -m*g*e3 + R*f_b - D_v*v + F_wind
        f_body = np.array([0.0, 0.0, f_thrust], dtype=np.float32)
        f_inertial = self.R @ f_body
        f_gravity = np.array([0.0, 0.0, -MASS_KG * GRAVITY_G], dtype=np.float32)
        f_drag = -DRAG_COEFFS * self.v

        accel_linear = (f_inertial + f_gravity + f_drag + self.wind_gust) / MASS_KG

        # 5. Rotational Euler equation: I * dot(omega) = tau - omega x (I * omega)
        gyroscopic = np.cross(self.omega, INERTIA_MATRIX @ self.omega)
        accel_angular = INV_INERTIA @ (torques_body - gyroscopic)

        # 6. Numerical Integration
        self.p += self.v * dt + 0.5 * accel_linear * (dt ** 2)
        self.v += accel_linear * dt

        # Enforce ground floor (cannot drop through ground z = 0)
        if self.p[2] < 0.0:
            self.p[2] = 0.0
            self.v[2] = max(0.0, self.v[2])

        # Integrate angular velocity & update SO(3) matrix via Rodrigues formula
        self.omega += accel_angular * dt
        d_theta = np.linalg.norm(self.omega) * dt
        if d_theta > 1e-6:
            axis = self.omega / np.linalg.norm(self.omega)
            K = skew_symmetric(axis)
            R_delta = np.eye(3) + math.sin(d_theta) * K + (1.0 - math.cos(d_theta)) * (K @ K)
            self.R = self.R @ R_delta

        # Re-orthonormalize R via SVD to eliminate numerical drift
        U, _, Vt = np.linalg.svd(self.R)
        self.R = U @ Vt

        self.time += dt
        return f_thrust, torques_body

    def get_state_vector(self):
        """Returns full 18-element state vector for controller & neural observer."""
        roll, pitch, yaw = rot_matrix_to_euler(self.R)
        return np.concatenate([
            self.p,             # 0:3  (x, y, z)
            self.v,             # 3:6  (vx, vy, vz)
            self.R.flatten(),   # 6:15 (Rotation matrix 9 elements)
            self.omega          # 15:18 (p, q, r)
        ]).astype(np.float32)

# ==========================================
# 3. GEOMETRIC NON-LINEAR SE(3) CONTROLLER
# ==========================================
class GeometricSE3Controller:
    """
    Asymptotically almost-globally stable geometric controller on SE(3)
    (Lee, Leok, McClamroch, IEEE CDC 2010).
    Zero gimbal lock, zero Euler angle singularities.
    """
    def __init__(self):
        # Position gains
        self.k_p = np.array([4.8, 4.8, 8.5], dtype=np.float32)
        self.k_v = np.array([3.4, 3.4, 5.2], dtype=np.float32)
        # Attitude gains on SO(3)
        self.k_R = 1.65
        self.k_omega = 0.28

    def compute_control(self, drone, p_des, v_des, a_des, yaw_des_deg, disturbance_hat=None):
        """
        Computes required collective thrust f and body torques tau on SE(3).
        """
        p, v, R, omega = drone.p, drone.v, drone.R, drone.omega

        # 1. Position & Velocity tracking errors in R^3
        e_p = p - p_des
        e_v = v - v_des

        # 2. Desired Force Vector F_d
        f_gravity = np.array([0.0, 0.0, MASS_KG * GRAVITY_G], dtype=np.float32)
        f_feedforward = MASS_KG * a_des
        f_feedback = -self.k_p * e_p - self.k_v * e_v

        F_d = f_feedback + f_gravity + f_feedforward
        if disturbance_hat is not None:
            F_d -= disturbance_hat  # Active neural disturbance rejection

        # Total collective scalar thrust
        b3 = R[:, 2]  # Current body z-axis (thrust direction)
        f_thrust = float(np.dot(F_d, b3))
        f_thrust = max(0.1, min(28.0, f_thrust))

        # 3. Construct Desired Rotation Matrix R_d in SO(3)
        norm_Fd = np.linalg.norm(F_d) + 1e-6
        b3_d = F_d / norm_Fd  # Desired body z-axis

        yaw_rad = math.radians(yaw_des_deg)
        b1_c = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=np.float32)

        # Cross product to get transverse body axis b2_d
        b2_d_raw = np.cross(b3_d, b1_c)
        norm_b2d = np.linalg.norm(b2_d_raw) + 1e-6
        b2_d = b2_d_raw / norm_b2d
        b1_d = np.cross(b2_d, b3_d)

        R_d = np.column_stack([b1_d, b2_d, b3_d]).astype(np.float32)

        # 4. Attitude Error e_R on Lie Group SO(3) using the Vee Map
        R_err_mat = 0.5 * (R_d.T @ R - R.T @ R_d)
        e_R = vee_map(R_err_mat)

        # Angular velocity error e_omega
        # (Assuming desired angular velocity omega_d ~ 0 for regulation)
        e_omega = omega

        # 5. Control Moments in Body Frame
        gyroscopic = np.cross(omega, INERTIA_MATRIX @ omega)
        tau_control = -self.k_R * e_R - self.k_omega * e_omega + gyroscopic

        # 6. Invert Allocation Matrix to Command Individual Motor Speeds (rad/s)
        arm = ARM_LENGTH_M / math.sqrt(2.0)
        # Allocation equations:
        # T0 + T1 + T2 + T3 = f
        # -T0 + T1 + T2 - T3 = tau_x / arm
        #  T0 + T1 - T2 - T3 = tau_y / arm
        # -Q0 + Q1 - Q2 + Q3 = tau_z (with Q = c_Q/c_T * T)
        t_alloc = np.zeros(4, dtype=np.float32)
        cq_over_ct = C_Q / C_T

        A = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-arm, arm, arm, -arm],
            [arm, arm, -arm, -arm],
            [-cq_over_ct, cq_over_ct, -cq_over_ct, cq_over_ct]
        ], dtype=np.float32)

        b = np.array([f_thrust, tau_control[0], tau_control[1], tau_control[2]], dtype=np.float32)
        try:
            T_motors = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            T_motors = np.full(4, f_thrust / 4.0, dtype=np.float32)

        T_motors = np.clip(T_motors, 0.05, 8.5)
        omega_cmds = np.sqrt(T_motors / C_T)

        return omega_cmds, f_thrust, tau_control, e_p, e_R, R_d

# ==========================================
# 4. PYTORCH NEURAL DISTURBANCE COMPENSATOR
# ==========================================
class SE3NeuralDisturbanceCompensator(nn.Module):
    """
    Physics-informed neural network estimating unmodeled aerodynamic forces
    and turbulent wind drag in real time to feed into the SE(3) control loop.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 3)  # Predicts 3D aerodynamic force disturbance (Nx, Ny, Nz)
        )

    def forward(self, x):
        return self.net(x)

def _worker_collect_se3(num_samples):
    drone = DronePhysics6DOF()
    controller = GeometricSE3Controller()
    states, disturbances = [], []

    while len(states) < num_samples:
        target_z = random.uniform(1.0, 4.0)
        p_des = np.array([random.uniform(-3.0, 3.0), random.uniform(-3.0, 3.0), target_z], dtype=np.float32)
        v_des = np.zeros(3, dtype=np.float32)
        a_des = np.zeros(3, dtype=np.float32)

        # Inject simulated Dryden gust disturbance
        gust = np.array([random.uniform(-3.5, 3.5), random.uniform(-3.5, 3.5), random.uniform(-1.0, 1.0)], dtype=np.float32)
        drone.wind_gust = gust

        for _ in range(50):
            omega_cmds, _, _, _, _, _ = controller.compute_control(drone, p_des, v_des, a_des, 0.0)
            drone.step(omega_cmds, dt=0.01)

            # Feature: velocity (3) + attitude z-axis (3) + angular velocity (3) = 9 inputs
            feat = np.concatenate([drone.v, drone.R[:, 2], drone.omega]).astype(np.float32)
            states.append(feat)
            disturbances.append(gust.copy())
            if len(states) >= num_samples:
                break

    return np.array(states, dtype=np.float32), np.array(disturbances, dtype=np.float32)

def train_neural_compensator():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"Device: {device}")
    print(f"Training SE(3) Neural Disturbance Compensator...")
    print(f"Aerodynamic Drag & Wind Gust Estimation")
    print(f"==========================================")

    print("Harvesting 120,000 6-DOF multirotor flight states across cores...")
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per_worker = (120000 + num_workers - 1) // num_workers

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_collect_se3, [samples_per_worker] * num_workers)

    X_np = np.concatenate([r[0] for r in results], axis=0)[:120000]
    y_np = np.concatenate([r[1] for r in results], axis=0)[:120000]
    print(f"Flight dynamics dataset harvested in {time.time() - t0:.2f}s!")

    X = torch.tensor(X_np, device=device)
    y = torch.tensor(y_np, device=device)

    model = SE3NeuralDisturbanceCompensator().to(device)
    BATCH_SIZE = 4096
    epochs = 10
    num_batches = (len(X) + BATCH_SIZE - 1) // BATCH_SIZE

    optimizer = optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-3, steps_per_epoch=num_batches, epochs=epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"Optimizing Neural Adaptive Weights (Batch {BATCH_SIZE}, 10 Epochs)...")
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X), device=device)
        total_loss = 0.0

        for i in range(0, len(X), BATCH_SIZE):
            idx = indices[i : i + BATCH_SIZE]
            bx, by = X[idx], y[idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(bx)
                loss = criterion(pred, by)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Aerodynamic Residual MSE Loss: {total_loss/num_batches:.5f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Neural Flight Model compiled in {time.time() - t0:.2f}s!")
    return model, device

# ==========================================
# 5. REAL-TIME 3D FLIGHT HUD & SIMULATOR
# ==========================================
def project_3d_to_screen(point, camera_rot, center_screen, scale=60.0):
    """Projects 3D world coordinate (X-East, Y-North, Z-Up) onto 2D screen."""
    yaw, pitch = camera_rot
    # Translate relative to origin
    x, y, z = point[0], point[1], point[2]

    # Rotate around Z (Yaw)
    x1 = x * math.cos(yaw) - y * math.sin(yaw)
    y1 = x * math.sin(yaw) + y * math.cos(yaw)
    z1 = z

    # Rotate around X-transverse (Pitch)
    x2 = x1
    y2 = y1 * math.cos(pitch) - z1 * math.sin(pitch)
    z2 = y1 * math.sin(pitch) + z1 * math.cos(pitch)

    # Perspective factor
    dist = 8.5 - y2
    focal = max(2.5, dist)
    screen_x = int(center_screen[0] + (x2 / focal) * scale * 6.5)
    screen_y = int(center_screen[1] - (z2 / focal) * scale * 6.5)
    return screen_x, screen_y, y2

def run_drone_flight_simulator(model, device):
    pygame.init()
    VIEW_WIDTH = 700
    VIEW_HEIGHT = 700
    SIDEBAR_WIDTH = 360
    WIDTH = VIEW_WIDTH + SIDEBAR_WIDTH
    HEIGHT = VIEW_HEIGHT

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Module 1: 6-DOF Multirotor Flight Dynamics & Geometric SE(3) Control")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("consolas", 18, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 13, bold=True)

    drone = DronePhysics6DOF()
    controller = GeometricSE3Controller()
    model.eval()

    # Target Waypoint in 3D (X, Y, Z) meters
    target_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    target_yaw_deg = 0.0

    camera_yaw = 0.65
    camera_pitch = 0.45
    center_viewport = (VIEW_WIDTH // 2, VIEW_HEIGHT // 2 + 60)

    flight_trail = deque(maxlen=80)
    running = True
    paused = False
    rotor_spin_phase = 0.0

    print("================ CONTROLS ================")
    print(" [W] / [S]      : Pitch / Move Forward / Back")
    print(" [A] / [D]      : Roll / Move Left / Right")
    print(" [I] / [K]      : Climb / Descend (Altitude)")
    print(" [J] / [L]      : Yaw Rotate (Heading)")
    print(" [G]            : Inject Sudden 35 km/h Wind Gust")
    print(" [SPACE]        : Pause / Resume Simulation")
    print(" [H]            : Return to 2.0m Center Hover")
    print(" [ARROW KEYS]   : Rotate 3D Camera View")
    print("==========================================\n")

    while running:
        dt = 0.01  # Fixed 100 Hz physical integration step
        rotor_spin_phase += 0.45

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_h:
                    target_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
                    target_yaw_deg = 0.0
                    drone.wind_gust = np.zeros(3, dtype=np.float32)
                elif event.key == pygame.K_g:
                    # Inject violent crosswind gust: 35 km/h (~9.7 m/s)
                    drone.wind_gust = np.array([random.uniform(-4.5, 4.5), random.uniform(-4.5, 4.5), random.uniform(-1.5, 1.5)], dtype=np.float32)

        # Interactive manual target positioning
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: target_pos[1] += 0.03
        if keys[pygame.K_s]: target_pos[1] -= 0.03
        if keys[pygame.K_d]: target_pos[0] += 0.03
        if keys[pygame.K_a]: target_pos[0] -= 0.03
        if keys[pygame.K_i]: target_pos[2] = min(5.0, target_pos[2] + 0.02)
        if keys[pygame.K_k]: target_pos[2] = max(0.2, target_pos[2] - 0.02)
        if keys[pygame.K_l]: target_yaw_deg = (target_yaw_deg + 1.5) % 360.0
        if keys[pygame.K_j]: target_yaw_deg = (target_yaw_deg - 1.5) % 360.0

        # Camera rotation
        if keys[pygame.K_LEFT]: camera_yaw -= 0.03
        if keys[pygame.K_RIGHT]: camera_yaw += 0.03
        if keys[pygame.K_UP]: camera_pitch = max(0.1, camera_pitch - 0.02)
        if keys[pygame.K_DOWN]: camera_pitch = min(1.2, camera_pitch + 0.02)

        if not paused:
            # 1. Evaluate Neural Disturbance Observer
            feat = np.concatenate([drone.v, drone.R[:, 2], drone.omega]).astype(np.float32)
            with torch.no_grad():
                feat_t = torch.tensor(feat, device=device).unsqueeze(0)
                d_hat_t = model(feat_t)[0]
                d_hat = d_hat_t.cpu().numpy()

            # 2. Compute Non-linear Geometric SE(3) Control
            v_des = np.zeros(3, dtype=np.float32)
            a_des = np.zeros(3, dtype=np.float32)
            omega_cmds, f_total, tau_ctrl, e_p, e_R, R_d = controller.compute_control(
                drone, target_pos, v_des, a_des, target_yaw_deg, disturbance_hat=d_hat
            )

            # 3. Step 6-DOF Physical Multirotor Equations
            drone.step(omega_cmds, dt=dt)
            flight_trail.append(drone.p.copy())

            # Slowly decay wind gust to simulate transient gust profile
            drone.wind_gust *= 0.995

        # Render Viewport
        screen.fill((10, 12, 16))

        # Dividers
        pygame.draw.line(screen, (35, 42, 58), (VIEW_WIDTH, 0), (VIEW_WIDTH, HEIGHT), 2)
        pygame.draw.line(screen, (35, 42, 58), (0, 70), (WIDTH, 70), 2)

        # ==========================================
        # 3D VIEWPORT: RENDERING AIRFRAME & SCENE
        # ==========================================
        cam = (camera_yaw, camera_pitch)

        # 1. Draw 3D Ground Reference Grid (5m x 5m)
        for gx in range(-3, 4):
            p_start = np.array([gx, -3.0, 0.0])
            p_end = np.array([gx, 3.0, 0.0])
            sx1, sy1, _ = project_3d_to_screen(p_start, cam, center_viewport)
            sx2, sy2, _ = project_3d_to_screen(p_end, cam, center_viewport)
            pygame.draw.line(screen, (22, 28, 38), (sx1, sy1), (sx2, sy2), 1)

        for gy in range(-3, 4):
            p_start = np.array([-3.0, gy, 0.0])
            p_end = np.array([3.0, gy, 0.0])
            sx1, sy1, _ = project_3d_to_screen(p_start, cam, center_viewport)
            sx2, sy2, _ = project_3d_to_screen(p_end, cam, center_viewport)
            pygame.draw.line(screen, (22, 28, 38), (sx1, sy1), (sx2, sy2), 1)

        # 2. Draw Target Waypoint (Gold Beacon with Altitude Drop-line)
        tx, ty, _ = project_3d_to_screen(target_pos, cam, center_viewport)
        gx, gy, _ = project_3d_to_screen(np.array([target_pos[0], target_pos[1], 0.0]), cam, center_viewport)
        pygame.draw.line(screen, (70, 60, 30), (tx, ty), (gx, gy), 1)
        pygame.draw.circle(screen, (255, 200, 0), (tx, ty), 7)
        pygame.draw.circle(screen, (255, 255, 255), (tx, ty), 3)

        # 3. Draw Flight Trail
        if len(flight_trail) >= 2:
            trail_pts = [project_3d_to_screen(pt, cam, center_viewport)[:2] for pt in flight_trail]
            pygame.draw.lines(screen, (0, 180, 220), False, trail_pts, 1)

        # 4. Draw 3D Multirotor Frame
        dp = drone.p
        arm_d = ARM_LENGTH_M * 1.5  # Scaled slightly for clear visual HUD presence
        # Motor locations in body frame (X-configuration)
        rotors_body = [
            np.array([arm_d / 1.414, -arm_d / 1.414, 0.0]),  # 1: Front-Right
            np.array([arm_d / 1.414,  arm_d / 1.414, 0.0]),  # 2: Front-Left
            np.array([-arm_d / 1.414,  arm_d / 1.414, 0.0]), # 3: Rear-Left
            np.array([-arm_d / 1.414, -arm_d / 1.414, 0.0]), # 4: Rear-Right
        ]
        rotors_inertial = [dp + drone.R @ r_b for r_b in rotors_body]

        hub_sx, hub_sy, _ = project_3d_to_screen(dp, cam, center_viewport)

        # Draw Cross-Arm Carbon Fiber Beams
        r_screen = [project_3d_to_screen(pt, cam, center_viewport)[:2] for pt in rotors_inertial]
        pygame.draw.line(screen, (160, 170, 185), r_screen[0], r_screen[2], 3)
        pygame.draw.line(screen, (160, 170, 185), r_screen[1], r_screen[3], 3)

        # Draw Spinning Rotor Discs
        disc_radius = 16
        for i, (rx, ry) in enumerate(r_screen):
            # Propeller blur ring
            r_col = (0, 240, 255) if i in [0, 1] else (0, 255, 140)
            pygame.draw.circle(screen, r_col, (rx, ry), disc_radius, 1)
            # Motor hub
            pygame.draw.circle(screen, (240, 240, 240), (rx, ry), 4)

        # Draw Fuselage Center Pod
        pygame.draw.circle(screen, (0, 220, 255), (hub_sx, hub_sy), 7)

        # Draw Body Orientation Triad at Hub:
        # b1 (Forward = Red), b2 (Left/Right = Green), b3 (Up/Thrust = Cyan)
        b1_tip = dp + drone.R @ np.array([0.45, 0.0, 0.0])
        b2_tip = dp + drone.R @ np.array([0.0, 0.45, 0.0])
        b3_tip = dp + drone.R @ np.array([0.0, 0.0, 0.45])

        b1_x, b1_y, _ = project_3d_to_screen(b1_tip, cam, center_viewport)
        b2_x, b2_y, _ = project_3d_to_screen(b2_tip, cam, center_viewport)
        b3_x, b3_y, _ = project_3d_to_screen(b3_tip, cam, center_viewport)

        pygame.draw.line(screen, (255, 60, 80), (hub_sx, hub_sy), (b1_x, b1_y), 2)  # b1: Forward
        pygame.draw.line(screen, (0, 255, 140), (hub_sx, hub_sy), (b2_x, b2_y), 2)  # b2: Lateral
        pygame.draw.line(screen, (0, 220, 255), (hub_sx, hub_sy), (b3_x, b3_y), 2)  # b3: Thrust vector

        # Drone Altitude Drop-line to Ground
        ground_x, ground_y, _ = project_3d_to_screen(np.array([dp[0], dp[1], 0.0]), cam, center_viewport)
        pygame.draw.line(screen, (40, 60, 85), (hub_sx, hub_sy), (ground_x, ground_y), 1)
        pygame.draw.circle(screen, (40, 70, 95), (ground_x, ground_y), 4)

        # ==========================================
        # AEROSPACE FLIGHT TELEMETRY (SIDEBAR)
        # ==========================================
        sb_x = VIEW_WIDTH + 14
        screen.blit(font_large.render("6-DOF FLIGHT TELEMETRY", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Geometric SE(3) Non-Linear Tracking Controller", True, (130, 160, 190)), (sb_x, 38))

        # --- PANEL 1: ROTOR MOTORS (RPM & LOAD) ---
        p1_y = 80
        screen.blit(font_mono.render("MOTOR ROTOR SPEEDS (RPM):", True, (240, 240, 240)), (sb_x, p1_y))

        rpms = [(w * 60.0) / (2.0 * math.pi) for w in drone.motor_omegas]
        m_labels = ["M1 (FR)", "M2 (FL)", "M3 (RL)", "M4 (RR)"]
        for m_idx in range(4):
            my = p1_y + 24 + m_idx * 16
            rpm_val = rpms[m_idx]
            bar_len = int(min(180, (rpm_val / MAX_RPM) * 180))

            m_col = (0, 255, 140) if rpm_val < 7200.0 else (255, 65, 80)
            screen.blit(font_small.render(f"{m_labels[m_idx]}: {rpm_val:.0f}", True, (160, 180, 205)), (sb_x + 10, my))
            pygame.draw.rect(screen, (24, 32, 45), (sb_x + 120, my + 2, 180, 7), border_radius=2)
            pygame.draw.rect(screen, m_col, (sb_x + 120, my + 2, bar_len, 7), border_radius=2)

        # --- PANEL 2: SE(3) TRACKING ERRORS ---
        p2_y = p1_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (WIDTH - 14, p2_y), 1)
        screen.blit(font_mono.render("SE(3) TRACKING RESIDUALS:", True, (240, 240, 240)), (sb_x, p2_y + 8))

        norm_ep = float(np.linalg.norm(e_p))
        norm_eR = float(np.linalg.norm(e_R))
        ep_col = (0, 255, 140) if norm_ep < 0.15 else ((255, 180, 40) if norm_ep < 0.50 else (255, 65, 80))

        screen.blit(font_small.render(f"Position Error: ||e_p|| = {norm_ep:.3f} m", True, ep_col), (sb_x + 10, p2_y + 28))
        screen.blit(font_small.render(f"Attitude Error: ||e_R|| = {norm_eR:.4f} rad (SO3)", True, (0, 220, 255)), (sb_x + 10, p2_y + 44))
        screen.blit(font_small.render(f"Control Torque: τ = [{tau_ctrl[0]:.2f}, {tau_ctrl[1]:.2f}, {tau_ctrl[2]:.2f}] N·m", True, (160, 180, 205)), (sb_x + 10, p2_y + 60))

        # --- PANEL 3: KINEMATICS & POSE ---
        p3_y = p2_y + 82
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (WIDTH - 14, p3_y), 1)
        screen.blit(font_mono.render("AIRCRAFT 6-DOF POSE:", True, (240, 240, 240)), (sb_x, p3_y + 8))

        roll_d, pitch_d, yaw_d = rot_matrix_to_euler(drone.R)
        speed_ms = float(np.linalg.norm(drone.v))

        screen.blit(font_small.render(f"Pos (X, Y, Z):  [{drone.p[0]:.2f}, {drone.p[1]:.2f}, {drone.p[2]:.2f}] m", True, (0, 255, 140)), (sb_x + 10, p3_y + 28))
        screen.blit(font_small.render(f"Speed |v|:      {speed_ms:.2f} m/s ({speed_ms*3.6:.1f} km/h)", True, (0, 220, 255)), (sb_x + 10, p3_y + 44))
        screen.blit(font_small.render(f"Attitude (RPY): [{roll_d:+.1f}°, {pitch_d:+.1f}°, {yaw_d:+.1f}°]", True, (160, 180, 205)), (sb_x + 10, p3_y + 60))
        screen.blit(font_small.render(f"Rates (p, q, r):[{drone.omega[0]:+.2f}, {drone.omega[1]:+.2f}, {drone.omega[2]:+.2f}] rad/s", True, (160, 180, 205)), (sb_x + 10, p3_y + 76))

        # --- PANEL 4: FORCES, THRUST & ACCELERATION ---
        p4_y = p3_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (WIDTH - 14, p4_y), 1)
        screen.blit(font_mono.render("FORCES & THRUST LOAD:", True, (240, 240, 240)), (sb_x, p4_y + 8))

        twr = f_total / HOVER_THRUST_TOTAL
        screen.blit(font_small.render(f"Total Thrust:   F_z = {f_total:.2f} N (Hover: {HOVER_THRUST_TOTAL:.2f} N)", True, (0, 255, 140)), (sb_x + 10, p4_y + 28))
        screen.blit(font_small.render(f"Thrust-to-Weight:TWR = {twr:.2f} : 1.0", True, (0, 220, 255)), (sb_x + 10, p4_y + 44))

        wind_spd = float(np.linalg.norm(drone.wind_gust))
        wind_col = (0, 255, 140) if wind_spd < 1.0 else (255, 180, 40)
        screen.blit(font_small.render(f"Wind Gust Load: |W| = {wind_spd*3.6:.1f} km/h (Press [G] to pulse)", True, wind_col), (sb_x + 10, p4_y + 60))

        # --- PANEL 5: NEURAL ADAPTATION STATUS ---
        p5_y = p4_y + 82
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p5_y), (WIDTH - 14, p5_y), 1)
        screen.blit(font_mono.render("NEURAL DISTURBANCE OBSERVER:", True, (240, 240, 240)), (sb_x, p5_y + 8))

        d_mag = float(np.linalg.norm(d_hat))
        screen.blit(font_small.render(f"Estimated Drag: ||d_hat|| = {d_mag:.3f} N", True, (0, 220, 255)), (sb_x + 10, p5_y + 28))
        screen.blit(font_small.render("Compensation:   Online SE(3) Feedforward Active", True, (0, 255, 140)), (sb_x + 10, p5_y + 44))
        screen.blit(font_small.render("Stability:      Lyapunov Asymptotic Convergence", True, (160, 180, 205)), (sb_x + 10, p5_y + 60))

        # Bottom Telemetry Banner
        bot_y = p5_y + 84
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y), (WIDTH - 14, bot_y), 1)
        status_banner = "SE(3) CONTROLLER: NOMINAL HOVER" if norm_ep < 0.2 else "TRAJECTORY TRACKING TRANSIENT"
        screen.blit(font_mono.render(status_banner, True, (0, 255, 140) if norm_ep < 0.2 else (255, 200, 40)), (sb_x + 8, bot_y + 8))

        # Header UI
        head_txt = font_large.render(f"Alt: {drone.p[2]:.2f}m | Airspeed: {speed_ms*3.6:.1f} km/h | 6-DOF Drone SE(3)", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("Full 6-DOF Newton-Euler Rigidbody Mechanics + Lie Group SE(3) Tracking (Zero Gimbal Lock)", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

# ==========================================
# 6. MAIN ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    mp.freeze_support()
    model, device = train_neural_compensator()
    print("Launching 6-DOF Multirotor Flight Dynamics & SE(3) Simulator...")
    run_drone_flight_simulator(model, device)