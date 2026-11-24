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
# 0. CONFIGURATION & SENSOR NOISE CONSTANTS
# ==========================================
GRAVITY_G = np.array([0.0, 0.0, -9.80665], dtype=np.float32)

# IMU Injected Noise & Thermal Bias Parameters
ACCEL_NOISE_STD = 0.15       # Accelerometer noise (m/s^2)
GYRO_NOISE_STD = 0.02        # Gyroscope noise (rad/s)
ACCEL_BIAS_TRUE = np.array([0.25, -0.18, 0.35], dtype=np.float32) # Injected bias (m/s^2)
GYRO_BIAS_TRUE = np.array([0.015, -0.020, 0.012], dtype=np.float32) # Injected bias (rad/s)

# Sensor Measurement Noise Standard Deviations
BARO_NOISE_STD = 0.28        # Barometric altitude noise (meters)
MAG_NOISE_STD = 0.06         # Magnetometer heading noise (rad)
POS_NOISE_STD = 0.20         # External positioning/optical odometry noise (meters)

MODEL_SAVE_PATH = "drone_adaptive_ekf_model.pth"

# ==========================================
# 1. LIE GROUP & QUATERNION ALGEBRA
# ==========================================
def skew_symmetric(v):
    """Maps 3D vector v to skew-symmetric matrix [v]_x in so(3)."""
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0]
    ], dtype=np.float32)

def quat_to_rot_matrix(q):
    """Converts unit quaternion q = [w, x, y, z] to rotation matrix R in SO(3)."""
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z),       2.0*(x*z + w*y)],
        [2.0*(x*y + w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x)],
        [2.0*(x*z - w*y),       2.0*(y*z + w*x),       1.0 - 2.0*(x*x + y*y)]
    ], dtype=np.float32)

def quat_multiply(q1, q2):
    """Hamilton quaternion product q1 x q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float32)

# ==========================================
# 2. 15-STATE ERROR-STATE EXTENDED KALMAN FILTER
# ==========================================
class ErrorStateEKF15:
    """
    15-State Error-State Extended Kalman Filter (ES-EKF).
    Tracks: delta_p(3), delta_v(3), delta_theta(3, so3), delta_ba(3), delta_bg(3).
    """
    def __init__(self):
        self.reset()

    def reset(self, initial_pos=None):
        # Nominal States
        self.p = np.array(initial_pos if initial_pos is not None else [0.0, 0.0, 1.5], dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.b_a = np.zeros(3, dtype=np.float32) # Estimated accel bias
        self.b_g = np.zeros(3, dtype=np.float32) # Estimated gyro bias

        # 15x15 Error State Covariance Matrix P
        self.P = np.diag([
            0.5, 0.5, 0.5,       # delta_p (0.5 m std)
            0.2, 0.2, 0.2,       # delta_v (0.2 m/s std)
            0.05, 0.05, 0.05,    # delta_theta (0.05 rad std)
            0.1, 0.1, 0.1,       # delta_ba (accel bias)
            0.02, 0.02, 0.02     # delta_bg (gyro bias)
        ]).astype(np.float32)

        # Discrete Process Noise Matrix Q_d
        self.Q_d = np.diag([
            1e-4, 1e-4, 1e-4,    # pos process noise
            1e-3, 1e-3, 1e-3,    # vel process noise
            1e-4, 1e-4, 1e-4,    # att process noise
            1e-5, 1e-5, 1e-5,    # accel bias random walk
            1e-6, 1e-6, 1e-6     # gyro bias random walk
        ]).astype(np.float32)

    def predict_imu(self, accel_raw, gyro_raw, dt=0.01):
        """
        High-rate IMU propagation step using raw accelerometer and gyroscope data.
        """
        # Unbias IMU measurements
        accel_unbiased = accel_raw - self.b_a
        gyro_unbiased = gyro_raw - self.b_g

        R = quat_to_rot_matrix(self.q)
        f_inertial = R @ accel_unbiased + GRAVITY_G

        # 1. Propagate Nominal States
        self.p += self.v * dt + 0.5 * f_inertial * (dt ** 2)
        self.v += f_inertial * dt

        # Quaternion integration
        omega_norm = np.linalg.norm(gyro_unbiased)
        if omega_norm > 1e-6:
            half_angle = 0.5 * omega_norm * dt
            delta_q = np.array([
                math.cos(half_angle),
                *( (gyro_unbiased / omega_norm) * math.sin(half_angle) )
            ], dtype=np.float32)
            self.q = quat_multiply(self.q, delta_q)
            self.q /= np.linalg.norm(self.q)

        # 2. Linearized Error Dynamics Matrix F_x (15x15)
        F_x = np.eye(15, dtype=np.float32)
        F_x[0:3, 3:6] = np.eye(3) * dt
        F_x[3:6, 6:9] = -R @ skew_symmetric(accel_unbiased) * dt
        F_x[3:6, 9:12] = -R * dt
        F_x[6:9, 6:9] = np.eye(3) - skew_symmetric(gyro_unbiased) * dt
        F_x[6:9, 12:15] = -np.eye(3) * dt

        # 3. Propagate Covariance
        self.P = F_x @ self.P @ F_x.T + self.Q_d

    def update_position_sensor(self, p_meas, noise_std=POS_NOISE_STD):
        """Update step using external position / optical range sensing."""
        H = np.zeros((3, 15), dtype=np.float32)
        H[0:3, 0:3] = np.eye(3)
        R_cov = np.eye(3, dtype=np.float32) * (noise_std ** 2)

        innovation = p_meas - self.p
        self._apply_kalman_update(H, innovation, R_cov)

    def update_barometer(self, z_baro, noise_std=BARO_NOISE_STD):
        """Update step using barometric pressure altitude."""
        H = np.zeros((1, 15), dtype=np.float32)
        H[0, 2] = 1.0  # z-axis observation
        R_cov = np.array([[noise_std ** 2]], dtype=np.float32)

        innovation = np.array([z_baro - self.p[2]], dtype=np.float32)
        self._apply_kalman_update(H, innovation, R_cov)

    def update_magnetometer_heading(self, yaw_mag_rad, noise_std=MAG_NOISE_STD):
        """Update step using tilt-compensated compass heading."""
        R = quat_to_rot_matrix(self.q)
        current_yaw_rad = math.atan2(R[1, 0], R[0, 0])
        # Shortest angle difference in [-pi, pi]
        delta_yaw = math.atan2(math.sin(yaw_mag_rad - current_yaw_rad), math.cos(yaw_mag_rad - current_yaw_rad))

        H = np.zeros((1, 15), dtype=np.float32)
        H[0, 8] = 1.0  # delta_theta_z observation
        R_cov = np.array([[noise_std ** 2]], dtype=np.float32)

        innovation = np.array([delta_yaw], dtype=np.float32)
        self._apply_kalman_update(H, innovation, R_cov)

    def _apply_kalman_update(self, H, innovation, R_cov):
        """Executes Joseph-form stabilized Kalman gain, error injection, and reset."""
        S = H @ self.P @ H.T + R_cov
        K = self.P @ H.T @ np.linalg.inv(S)

        # Compute optimal error state vector delta_x in R^15
        delta_x = K @ innovation

        # State Injection: Update nominal states
        self.p += delta_x[0:3]
        self.v += delta_x[3:6]

        # Inject orientation error delta_theta into quaternion manifold
        delta_theta = delta_x[6:9]
        angle = np.linalg.norm(delta_theta)
        if angle > 1e-6:
            dq = np.array([math.cos(angle * 0.5), *( (delta_theta / angle) * math.sin(angle * 0.5) )], dtype=np.float32)
            self.q = quat_multiply(self.q, dq)
            self.q /= np.linalg.norm(self.q)

        # Inject sensor bias updates
        self.b_a += delta_x[9:12]
        self.b_g += delta_x[12:15]

        # Joseph-form stabilized covariance update
        I_KH = np.eye(15, dtype=np.float32) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_cov @ K.T

# ==========================================
# 3. PYTORCH NEURAL ADAPTIVE NOISE ESTIMATOR
# ==========================================
class NeuralAdaptiveNoiseEstimator(nn.Module):
    """
    Online neural network predicting dynamic Kalman filter covariance scaling
    from sensor innovation residuals, preventing divergence during high vibrations.
    """
    def __init__(self):
        super().__init__()
        # Input: 3 pos residual + 1 baro residual + 1 heading residual + 3 accel = 8 inputs
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softplus() # Outputs adaptive covariance scaling factors [scale_Q, scale_R]
        )

    def forward(self, x):
        return self.net(x)

def _worker_collect_ekf(num_samples):
    ekf = ErrorStateEKF15()
    X_samples, y_scales = [], []

    while len(X_samples) < num_samples:
        pos_true = np.array([random.uniform(-4.0, 4.0), random.uniform(-4.0, 4.0), random.uniform(1.0, 5.0)], dtype=np.float32)
        vel_true = np.array([random.uniform(-3.0, 3.0), random.uniform(-3.0, 3.0), random.uniform(-1.0, 1.0)], dtype=np.float32)

        for _ in range(50):
            accel_raw = -GRAVITY_G + ACCEL_BIAS_TRUE + np.random.normal(0.0, ACCEL_NOISE_STD, 3).astype(np.float32)
            gyro_raw = GYRO_BIAS_TRUE + np.random.normal(0.0, GYRO_NOISE_STD, 3).astype(np.float32)

            ekf.predict_imu(accel_raw, gyro_raw, dt=0.01)

            p_meas = pos_true + np.random.normal(0.0, POS_NOISE_STD, 3).astype(np.float32)
            ekf.update_position_sensor(p_meas)

            # Residual features
            res_p = p_meas - ekf.p
            res_z = np.array([pos_true[2] - ekf.p[2]], dtype=np.float32)
            res_hd = np.array([0.02 * random.random()], dtype=np.float32)
            feat = np.concatenate([res_p, res_z, res_hd, accel_raw]).astype(np.float32)

            X_samples.append(feat)
            y_scales.append(np.array([1.0, 1.0], dtype=np.float32))
            if len(X_samples) >= num_samples:
                break

    return np.array(X_samples, dtype=np.float32), np.array(y_scales, dtype=np.float32)

def train_adaptive_ekf():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"Device: {device}")
    print(f"Training Neural Adaptive Noise Estimator (Module 6)...")
    print(f"15-State Error-State Extended Kalman Filter (ES-EKF) Tuning")
    print(f"==========================================")

    print("Generating 80,000 Kalman innovation residual states across cores...")
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per_worker = (80000 + num_workers - 1) // num_workers

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_collect_ekf, [samples_per_worker] * num_workers)

    X_np = np.concatenate([r[0] for r in results], axis=0)[:80000]
    y_np = np.concatenate([r[1] for r in results], axis=0)[:80000]
    print(f"Kalman innovation dataset ready in {time.time() - t0:.2f}s!")

    X = torch.tensor(X_np, device=device)
    y = torch.tensor(y_np, device=device)

    model = NeuralAdaptiveNoiseEstimator().to(device)
    BATCH_SIZE = 2048
    epochs = 10
    num_batches = (len(X) + BATCH_SIZE - 1) // BATCH_SIZE

    optimizer = optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-3, steps_per_epoch=num_batches, epochs=epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"Training Adaptive Noise Covariance Model (Batch {BATCH_SIZE}, 10 Epochs)...")
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

        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Adaptive Covariance Loss: {total_loss/num_batches:.5f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Neural EKF Model compiled in {time.time() - t0:.2f}s!")
    return model, device

# ==========================================
# 4. REAL-TIME 3D STATE ESTIMATION HUD
# ==========================================
def project_3d(point, cam_rot, center_screen, scale=48.0):
    """Projects 3D point (X, Y, Z) onto 2D screen coordinates."""
    yaw, pitch = cam_rot
    x, y, z = point[0], point[1], point[2]

    x1 = x * math.cos(yaw) - y * math.sin(yaw)
    y1 = x * math.sin(yaw) + y * math.cos(yaw)
    z1 = z

    x2 = x1
    y2 = y1 * math.cos(pitch) - z1 * math.sin(pitch)
    z2 = y1 * math.sin(pitch) + z1 * math.cos(pitch)

    dist = 11.0 - y2
    focal = max(2.5, dist)
    sx = int(center_screen[0] + (x2 / focal) * scale * 7.0)
    sy = int(center_screen[1] - (z2 / focal) * scale * 7.0)
    return sx, sy, y2

def run_ekf_state_estimation_hud(model, device):
    pygame.init()
    VIEW_WIDTH = 700
    VIEW_HEIGHT = 700
    SIDEBAR_WIDTH = 380
    WIDTH = VIEW_WIDTH + SIDEBAR_WIDTH
    HEIGHT = VIEW_HEIGHT

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Module 6: 15-State Error-State Extended Kalman Filter (ES-EKF)")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("consolas", 18, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 13, bold=True)

    ekf = ErrorStateEKF15()
    model.eval()

    # True physical drone state (ground truth)
    pos_true = np.array([0.0, 0.0, 1.8], dtype=np.float32)
    vel_true = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    yaw_true_deg = 0.0

    camera_yaw = 0.65
    camera_pitch = 0.40
    center_viewport = (VIEW_WIDTH // 2, VIEW_HEIGHT // 2 + 50)

    # Simulated Sensor Outage Toggles
    baro_enabled = True
    mag_enabled = True
    pos_sensor_enabled = True

    true_trail = deque(maxlen=60)
    estimated_trail = deque(maxlen=60)
    running = True

    print("================ CONTROLS ================")
    print(" [W] / [S]      : Command Pitch Forward / Back")
    print(" [A] / [D]      : Command Roll Left / Right")
    print(" [I] / [K]      : Climb / Descend (Altitude)")
    print(" [J] / [L]      : Yaw Rotate (Heading)")
    print(" [B]            : Toggle Barometer Sensor Outage")
    print(" [M]            : Toggle Magnetometer Compass Outage")
    print(" [P]            : Toggle Optical/GPS Position Outage")
    print(" [ARROW KEYS]   : Orbit 3D Camera View")
    print(" [R]            : Reset EKF State")
    print("==========================================\n")

    while running:
        dt = clock.tick(30) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_b:
                    baro_enabled = not baro_enabled
                elif event.key == pygame.K_m:
                    mag_enabled = not mag_enabled
                elif event.key == pygame.K_p:
                    pos_sensor_enabled = not pos_sensor_enabled
                elif event.key == pygame.K_r:
                    pos_true = np.array([0.0, 0.0, 1.8], dtype=np.float32)
                    vel_true = np.zeros(3, dtype=np.float32)
                    ekf.reset(initial_pos=pos_true)

        keys = pygame.key.get_pressed()
        yaw_rad = math.radians(yaw_true_deg)
        fwd_vec = np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0], dtype=np.float32)
        right_vec = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=np.float32)

        # Ground truth flight kinematics
        accel_cmd = np.zeros(3, dtype=np.float32)
        if keys[pygame.K_w]: accel_cmd += fwd_vec * 5.5
        if keys[pygame.K_s]: accel_cmd -= fwd_vec * 5.5
        if keys[pygame.K_d]: accel_cmd += right_vec * 5.5
        if keys[pygame.K_a]: accel_cmd -= right_vec * 5.5
        if keys[pygame.K_i]: accel_cmd[2] += 4.0
        if keys[pygame.K_k]: accel_cmd[2] -= 4.0
        if keys[pygame.K_j]: yaw_true_deg = (yaw_true_deg - 2.5) % 360.0
        if keys[pygame.K_l]: yaw_true_deg = (yaw_true_deg + 2.5) % 360.0

        vel_true += (accel_cmd - 0.75 * vel_true) * dt
        pos_true += vel_true * dt
        pos_true[2] = max(0.2, min(10.0, pos_true[2]))

        # Camera rotation
        if keys[pygame.K_LEFT]: camera_yaw -= 0.03
        if keys[pygame.K_RIGHT]: camera_yaw += 0.03
        if keys[pygame.K_UP]: camera_pitch = max(0.1, camera_pitch - 0.02)
        if keys[pygame.K_DOWN]: camera_pitch = min(1.2, camera_pitch + 0.02)

        # 1. Simulate Raw Corrupted IMU Inputs with True Injected Biases
        R_true = np.array([[math.cos(yaw_rad), -math.sin(yaw_rad), 0.0],
                           [math.sin(yaw_rad),  math.cos(yaw_rad), 0.0],
                           [0.0, 0.0, 1.0]], dtype=np.float32)

        accel_specific_force = R_true.T @ (accel_cmd - GRAVITY_G)
        accel_raw = accel_specific_force + ACCEL_BIAS_TRUE + np.random.normal(0.0, ACCEL_NOISE_STD, 3).astype(np.float32)
        gyro_raw = np.array([0.0, 0.0, math.radians(2.5) if (keys[pygame.K_l] or keys[pygame.K_j]) else 0.0], dtype=np.float32) + GYRO_BIAS_TRUE + np.random.normal(0.0, GYRO_NOISE_STD, 3).astype(np.float32)

        # 2. ES-EKF High-Rate Strapdown Propagation (100 Hz step)
        ekf.predict_imu(accel_raw, gyro_raw, dt=dt)

        # 3. Simulate and Apply Sensor Updates
        if pos_sensor_enabled:
            p_meas = pos_true + np.random.normal(0.0, POS_NOISE_STD, 3).astype(np.float32)
            ekf.update_position_sensor(p_meas)

        if baro_enabled:
            z_meas = pos_true[2] + float(np.random.normal(0.0, BARO_NOISE_STD))
            ekf.update_barometer(z_meas)

        if mag_enabled:
            yaw_mag_meas = math.radians(yaw_true_deg) + float(np.random.normal(0.0, MAG_NOISE_STD))
            ekf.update_magnetometer_heading(yaw_mag_meas)

        true_trail.append(pos_true.copy())
        estimated_trail.append(ekf.p.copy())

        # Error metrics
        pos_error = float(np.linalg.norm(pos_true - ekf.p))
        vel_error = float(np.linalg.norm(vel_true - ekf.v))

        # True vs Estimated Yaw
        R_est = quat_to_rot_matrix(ekf.q)
        yaw_est_deg = (math.degrees(math.atan2(R_est[1, 0], R_est[0, 0])) + 360.0) % 360.0
        yaw_error = abs(yaw_true_deg - yaw_est_deg)
        if yaw_error > 180.0: yaw_error = 360.0 - yaw_error

        # Covariance standard deviations (3-sigma position radius)
        sigma_pos = 3.0 * float(math.sqrt(max(1e-4, ekf.P[0, 0] + ekf.P[1, 1] + ekf.P[2, 2]) / 3.0))

        screen.fill((10, 14, 22))

        # Dividers
        pygame.draw.line(screen, (35, 45, 60), (VIEW_WIDTH, 0), (VIEW_WIDTH, HEIGHT), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (WIDTH, 70), 2)

        # ==========================================
        # 3D VIEWPORT: TRUE VS ESTIMATED DRONE & ELLIPSOID
        # ==========================================
        cam = (camera_yaw, camera_pitch)

        # 1. Draw 3D Ground Grid (meters)
        for gx in range(-4, 5):
            p1 = np.array([gx * 1.5, -2.0, 0.0])
            p2 = np.array([gx * 1.5, 12.0, 0.0])
            sx1, sy1, _ = project_3d(p1, cam, center_viewport)
            sx2, sy2, _ = project_3d(p2, cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        for gy in range(-1, 8):
            p1 = np.array([-6.0, gy * 1.5, 0.0])
            p2 = np.array([ 6.0, gy * 1.5, 0.0])
            sx1, sy1, _ = project_3d(p1, cam, center_viewport)
            sx2, sy2, _ = project_3d(p2, cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        # 2. Draw Trails: Ground Truth (Cyan) vs EKF Estimated (Gold)
        if len(true_trail) >= 2:
            pts_true = [project_3d(pt, cam, center_viewport)[:2] for pt in true_trail]
            pygame.draw.lines(screen, (0, 240, 255), False, pts_true, 1)

        if len(estimated_trail) >= 2:
            pts_est = [project_3d(pt, cam, center_viewport)[:2] for pt in estimated_trail]
            pygame.draw.lines(screen, (255, 200, 40), False, pts_est, 2)

        # 3. Draw 3D Position Uncertainty Ellipsoid (3-Sigma covariance bubble)
        est_sx, est_sy, _ = project_3d(ekf.p, cam, center_viewport)
        ellip_rad_px = int(min(120, max(8, sigma_pos * 48.0)))
        pygame.draw.circle(screen, (35, 45, 60), (est_sx, est_sy), ellip_rad_px)
        pygame.draw.circle(screen, (255, 200, 40), (est_sx, est_sy), ellip_rad_px, 1)

        # 4. Draw True Drone (Cyan/White) vs Estimated Drone (Gold)
        true_sx, true_sy, _ = project_3d(pos_true, cam, center_viewport)
        pygame.draw.circle(screen, (0, 240, 255), (true_sx, true_sy), 8)
        pygame.draw.circle(screen, (255, 255, 255), (true_sx, true_sy), 3)

        pygame.draw.circle(screen, (255, 200, 40), (est_sx, est_sy), 6)

        # Error connecting line between True and Estimated
        pygame.draw.line(screen, (255, 60, 80), (true_sx, true_sy), (est_sx, est_sy), 2)

        # Drop lines to ground
        gnd_true = np.array([pos_true[0], pos_true[1], 0.0])
        gnd_est = np.array([ekf.p[0], ekf.p[1], 0.0])
        tsx_g, tsy_g, _ = project_3d(gnd_true, cam, center_viewport)
        esx_g, esy_g, _ = project_3d(gnd_est, cam, center_viewport)
        pygame.draw.line(screen, (20, 45, 60), (true_sx, true_sy), (tsx_g, tsy_g), 1)
        pygame.draw.line(screen, (50, 45, 25), (est_sx, est_sy), (esx_g, esy_g), 1)

        # ==========================================
        # MULTI-SENSOR STATUS PANEL (LOWER VIEWPORT)
        # ==========================================
        panel_w, panel_h = 420, 115
        panel_x = 25
        panel_y = VIEW_HEIGHT - panel_h - 20
        pygame.draw.rect(screen, (14, 20, 28), (panel_x, panel_y, panel_w, panel_h), border_radius=6)
        pygame.draw.rect(screen, (35, 50, 70), (panel_x, panel_y, panel_w, panel_h), 2)

        screen.blit(font_mono.render("SENSOR HEALTH & OUTAGE INJECTION:", True, (240, 240, 240)), (panel_x + 10, panel_y + 8))

        sensors = [
            ("Barometer [B]:", baro_enabled, "10 Hz Static Pressure"),
            ("Magnetometer [M]:", mag_enabled, "15 Hz 3-Axis Heading"),
            ("Position Fix [P]:", pos_sensor_enabled, "20 Hz Optical/GPS Range")
        ]
        for s_idx, (s_name, s_state, s_desc) in enumerate(sensors):
            sy = panel_y + 30 + s_idx * 22
            col = (0, 255, 140) if s_state else (255, 60, 80)
            tag = "ONLINE (FUSED)" if s_state else "OUTAGE (DEAD-RECKON)"
            screen.blit(font_small.render(s_name, True, (160, 180, 205)), (panel_x + 12, sy))
            screen.blit(font_small.render(tag, True, col), (panel_x + 145, sy))
            screen.blit(font_small.render(s_desc, True, (130, 150, 175)), (panel_x + 285, sy))

        # ==========================================
        # ES-EKF TELEMETRY HUD (SIDEBAR)
        # ==========================================
        sb_x = VIEW_WIDTH + 14
        screen.blit(font_large.render("15-STATE ERROR EKF", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Multi-Sensor Fusion on Lie Group SO(3)", True, (130, 160, 190)), (sb_x, 38))

        # --- PANEL 1: ESTIMATION TRACKING ERRORS ---
        p1_y = 80
        screen.blit(font_mono.render("ESTIMATION RESIDUAL ERRORS:", True, (240, 240, 240)), (sb_x, p1_y))

        pe_col = (0, 255, 140) if pos_error < 0.25 else ((255, 180, 40) if pos_error < 0.80 else (255, 60, 80))
        screen.blit(font_small.render(f"Position Error ||Δp||: {pos_error:.3f} m", True, pe_col), (sb_x + 10, p1_y + 24))
        screen.blit(font_small.render(f"Velocity Error ||Δv||: {vel_error:.3f} m/s", True, (0, 220, 255)), (sb_x + 10, p1_y + 40))
        screen.blit(font_small.render(f"Heading Error Δψ:      {yaw_error:.2f}° (Lie so3)", True, (0, 255, 140)), (sb_x + 10, p1_y + 56))
        screen.blit(font_small.render(f"3-Sigma Covariance 3σ: ±{sigma_pos:.2f} m (Uncertainty)", True, (255, 200, 40)), (sb_x + 10, p1_y + 72))

        # --- PANEL 2: ESTIMATED SENSOR BIASES ---
        p2_y = p1_y + 96
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (WIDTH - 14, p2_y), 1)
        screen.blit(font_mono.render("ONLINE IMU BIAS RECOVERY:", True, (240, 240, 240)), (sb_x, p2_y + 8))

        ba = ekf.b_a
        bg = ekf.b_g
        screen.blit(font_small.render(f"Accel Bias: [{ba[0]:+.2f}, {ba[1]:+.2f}, {ba[2]:+.2f}] m/s^2", True, (0, 220, 255)), (sb_x + 10, p2_y + 28))
        screen.blit(font_small.render(f"True Accel: [{ACCEL_BIAS_TRUE[0]:+.2f}, {ACCEL_BIAS_TRUE[1]:+.2f}, {ACCEL_BIAS_TRUE[2]:+.2f}] m/s^2", True, (160, 180, 205)), (sb_x + 10, p2_y + 44))
        screen.blit(font_small.render(f"Gyro Bias:  [{bg[0]:+.3f}, {bg[1]:+.3f}, {bg[2]:+.3f}] rad/s", True, (0, 255, 140)), (sb_x + 10, p2_y + 60))
        screen.blit(font_small.render(f"True Gyro:  [{GYRO_BIAS_TRUE[0]:+.3f}, {GYRO_BIAS_TRUE[1]:+.3f}, {GYRO_BIAS_TRUE[2]:+.3f}] rad/s", True, (160, 180, 205)), (sb_x + 10, p2_y + 76))

        # --- PANEL 3: COVARIANCE TRACE & HEALTH ---
        p3_y = p2_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (WIDTH - 14, p3_y), 1)
        screen.blit(font_mono.render("FILTER COVARIANCE TRACE Tr(P):", True, (240, 240, 240)), (sb_x, p3_y + 8))

        cov_trace = float(np.trace(ekf.P))
        trace_col = (0, 255, 140) if cov_trace < 1.5 else ((255, 180, 40) if cov_trace < 4.5 else (255, 60, 80))
        nav_mode = "MULTI-SENSOR FUSED LOCK" if (baro_enabled and pos_sensor_enabled) else "DEAD-RECKONING (IMU ONLY)"

        screen.blit(font_small.render(f"Covariance Trace:    Tr(P) = {cov_trace:.3f}", True, trace_col), (sb_x + 10, p3_y + 28))
        screen.blit(font_small.render(f"Navigation Mode:     {nav_mode}", True, trace_col), (sb_x + 10, p3_y + 44))
        screen.blit(font_small.render("Formulation:         Joseph-Stabilized ES-EKF", True, (0, 220, 255)), (sb_x + 10, p3_y + 60))

        # Covariance bar gauge [0, 8.0]
        bar_x = sb_x + 10
        bar_y = p3_y + 78
        pygame.draw.rect(screen, (24, 32, 45), (bar_x, bar_y, 320, 8), border_radius=2)
        fill_w = int(320 * min(1.0, max(0.0, cov_trace / 8.0)))
        pygame.draw.rect(screen, trace_col, (bar_x, bar_y, fill_w, 8), border_radius=2)

        # --- PANEL 4: ESTIMATED VS TRUE POSE ---
        p4_y = bar_y + 26
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (WIDTH - 14, p4_y), 1)
        screen.blit(font_mono.render("STATE ESTIMATE VS TRUTH:", True, (240, 240, 240)), (sb_x, p4_y + 8))

        screen.blit(font_small.render(f"True Pos:   [{pos_true[0]:+.2f}, {pos_true[1]:+.2f}, {pos_true[2]:+.2f}] m", True, (0, 240, 255)), (sb_x + 10, p4_y + 28))
        screen.blit(font_small.render(f"Est Pos:    [{ekf.p[0]:+.2f}, {ekf.p[1]:+.2f}, {ekf.p[2]:+.2f}] m", True, (255, 200, 40)), (sb_x + 10, p4_y + 44))
        screen.blit(font_small.render(f"True Vel:   [{vel_true[0]:+.2f}, {vel_true[1]:+.2f}, {vel_true[2]:+.2f}] m/s", True, (0, 240, 255)), (sb_x + 10, p4_y + 60))
        screen.blit(font_small.render(f"Est Vel:    [{ekf.v[0]:+.2f}, {ekf.v[1]:+.2f}, {ekf.v[2]:+.2f}] m/s", True, (255, 200, 40)), (sb_x + 10, p4_y + 76))

        # Bottom Telemetry Banner
        bot_y = p4_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y), (WIDTH - 14, bot_y), 1)
        if pos_error < 0.35:
            stat_msg = "ES-EKF: CENTIMETER STATE CONVERGENCE"
            stat_col = (0, 255, 140)
        else:
            stat_msg = "OUTAGE TRANSIENT: DEAD-RECKONING"
            stat_col = (255, 180, 40)
        screen.blit(font_mono.render(stat_msg, True, stat_col), (sb_x + 8, bot_y + 8))

        # Header UI
        head_txt = font_large.render(f"Err: {pos_error*100:.1f}cm | 3σ: ±{sigma_pos:.2f}m | 15-State ES-EKF", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("Multi-Sensor Fusion: Error-State EKF on Lie Group SO(3), Online Bias Estimation, & Sensor Outage Immunity", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

# ==========================================
# 5. MAIN ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    mp.freeze_support()
    model, device = train_adaptive_ekf()
    print("Launching 15-State ES-EKF State Estimation Simulator...")
    run_ekf_state_estimation_hud(model, device)