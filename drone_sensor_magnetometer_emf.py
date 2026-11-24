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
# 0. CONFIGURATION & ELECTROMAGNETIC CONSTANTS
# ==========================================
# Earth's Geomagnetic Field Parameters (WMM Baseline)
B_EARTH_TOTAL_UT = 52.0    # Total magnetic intensity in microTesla (uT)
INCLINATION_DIP_DEG = 64.0 # Dip angle (inclination) in degrees
DECLINATION_DEG = -3.5     # Magnetic declination angle (West)

dip_rad = math.radians(INCLINATION_DIP_DEG)
dec_rad = math.radians(DECLINATION_DEG)
B_EARTH_NED = B_EARTH_TOTAL_UT * np.array([
    math.cos(dip_rad) * math.cos(dec_rad),
    math.cos(dip_rad) * math.sin(dec_rad),
    math.sin(dip_rad)
], dtype=np.float32)

# Uncalibrated Onboard Hard-Iron & Soft-Iron Distortion
HARD_IRON_BIAS = np.array([12.5, -9.0, 16.5], dtype=np.float32) # uT offset
SOFT_IRON_MATRIX = np.array([
    [1.12, 0.08, 0.02],
    [0.08, 0.94, 0.05],
    [0.02, 0.05, 1.06]
], dtype=np.float32)
INV_SOFT_IRON = np.linalg.inv(SOFT_IRON_MATRIX)

# High-Voltage 50/60 Hz Transmission Line (500 kV Catenary Wire)
GRID_FREQ_HZ = 60.0
LINE_CURRENT_AMPS = 750.0  # AC current (Amperes)
MU_0 = 4.0 * math.pi * 1e-7 # Vacuum permeability
EMF_WARNING_THRESHOLD_UT = 18.0 # Alert threshold (uT)
EMF_CRITICAL_THRESHOLD_UT = 38.0 # Critical proximity threshold (uT)

MODEL_SAVE_PATH = "drone_magnetometer_emf_model.pth"

# ==========================================
# 1. 3D GEOMAGNETIC & EMF SENSOR SUITE
# ==========================================
class MagnetometerEMFSuite:
    """
    Simulates:
    1. Earth's geomagnetic vector rotated into drone body frame R in SO(3).
    2. Onboard hard-iron and soft-iron sensor corruptions.
    3. 60 Hz Biot-Savart AC electromagnetic field radiated by transmission lines.
    """
    def __init__(self):
        # Power line catenary wires running across the world at Y = 5.0m, Z = 3.2m and 4.0m
        self.powerline_y = 5.0
        self.powerline_z_wires = [3.2, 4.0]
        self.ac_time = 0.0

    def poll_sensors(self, drone_pos, R_drone, dt=0.03):
        """
        Polls:
        - raw_mag: Corrupted 3-axis magnetometer vector [mx, my, mz] (uT)
        - cal_mag: Calibrated tilt-compensated magnetometer vector (uT)
        - emf_ac_rms: 60 Hz EMF inductive pickup intensity (uT)
        - heading_true: True geographic heading from tilt-compensated compass (deg)
        """
        self.ac_time += dt

        # 1. Geomagnetic Field rotated into body frame: B_body = R^T * B_NED
        B_body_ideal = R_drone.T @ B_EARTH_NED

        # 2. Add Hard-Iron and Soft-Iron Distortion
        m_raw = (SOFT_IRON_MATRIX @ B_body_ideal) + HARD_IRON_BIAS + np.random.normal(0.0, 0.4, 3).astype(np.float32)

        # 3. Apply Hard/Soft-Iron Calibration: m_cal = M_soft^-1 * (m_raw - b_hard)
        m_cal = INV_SOFT_IRON @ (m_raw - HARD_IRON_BIAS)

        # 4. Tilt-Compensated Electronic Compass Heading Calculation
        # Extract Euler roll (phi) and pitch (theta) from R
        r11, r12, r13 = R_drone[0, 0], R_drone[0, 1], R_drone[0, 2]
        r21, r22, r23 = R_drone[1, 0], R_drone[1, 1], R_drone[1, 2]
        r31, r32, r33 = R_drone[2, 0], R_drone[2, 1], R_drone[2, 2]

        pitch_rad = -math.asin(max(-1.0, min(1.0, r31)))
        roll_rad = math.atan2(r32, r33) if abs(math.cos(pitch_rad)) > 1e-4 else 0.0

        cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
        cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)

        # Tilt compensation projection onto horizontal plane
        bx_prime = m_cal[0] * cos_p + m_cal[1] * sin_r * sin_p + m_cal[2] * cos_r * sin_p
        by_prime = m_cal[1] * cos_r - m_cal[2] * sin_r

        # Magnetic heading + declination correction
        heading_mag_rad = math.atan2(-by_prime, bx_prime)
        heading_true_deg = (math.degrees(heading_mag_rad) + DECLINATION_DEG) % 360.0

        # 5. Biot-Savart 60 Hz Alternating Electromagnetic Field (Powerline)
        # Distance from drone to nearest energized wire
        min_wire_dist = 100.0
        for wz in self.powerline_z_wires:
            d = math.hypot(drone_pos[1] - self.powerline_y, drone_pos[2] - wz)
            min_wire_dist = min(min_wire_dist, d)

        # B_AC_RMS = (mu_0 * I_RMS) / (2 * pi * r) in microTesla (uT)
        # For 750 Amperes: mu_0 * I / (2 * pi) = 2e-7 * 750 = 1.5e-4 T*m = 150.0 uT*m
        emf_ac_rms = float(150.0 / max(0.25, min_wire_dist))
        # 60 Hz instantaneous oscillation
        ac_phase = 2.0 * math.pi * GRID_FREQ_HZ * self.ac_time
        emf_instantaneous = emf_ac_rms * math.sin(ac_phase)

        is_emf_warning = (emf_ac_rms >= EMF_WARNING_THRESHOLD_UT)
        is_emf_critical = (emf_ac_rms >= EMF_CRITICAL_THRESHOLD_UT)

        telemetry = {
            "m_raw": m_raw,
            "m_cal": m_cal,
            "heading_true": heading_true_deg,
            "emf_ac_rms": emf_ac_rms,
            "emf_instantaneous": emf_instantaneous,
            "min_wire_dist": min_wire_dist,
            "is_emf_warning": is_emf_warning,
            "is_emf_critical": is_emf_critical
        }
        return telemetry

# ==========================================
# 2. PYTORCH NEURAL EMF & CALIBRATION ESTIMATOR
# ==========================================
class EMFMagnetometerFusionNet(nn.Module):
    """
    Deep neural network processing raw magnetometer streams and AC EMF induction
    to predict uncorrupted compass heading and powerline proximity.
    """
    def __init__(self):
        super().__init__()
        # Inputs: 3 raw mag + 3 calibrated mag + 1 EMF RMS + 2 attitude (pitch, roll) = 9 inputs
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # [Heading_Sin, Heading_Cos] unit vector
        )

        self.wire_dist_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Softplus() # Distance to high-voltage wire >= 0
        )

    def forward(self, x):
        h = self.net[:-1](x)
        heading_vec = self.net[-1](h)
        norm = torch.norm(heading_vec, dim=-1, keepdim=True) + 1e-7
        unit_heading = heading_vec / norm
        wire_dist = self.wire_dist_head(h)
        return unit_heading, wire_dist

def _worker_collect_mag(num_samples):
    suite = MagnetometerEMFSuite()
    X_samples, y_heads, y_dists = [], [], []

    while len(X_samples) < num_samples:
        pos = np.array([random.uniform(-5.0, 5.0), random.uniform(-1.0, 11.0), random.uniform(0.5, 5.5)], dtype=np.float32)
        yaw = math.radians(random.uniform(-180.0, 180.0))
        pitch = math.radians(random.uniform(-15.0, 15.0))
        roll = math.radians(random.uniform(-15.0, 15.0))

        # Rotation matrix in SO(3)
        R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        R_y = np.array([[math.cos(pitch), 0.0, math.sin(pitch)], [0.0, 1.0, 0.0], [-math.sin(pitch), 0.0, math.cos(pitch)]], dtype=np.float32)
        R_x = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(roll), -math.sin(roll)], [0.0, math.sin(roll), math.cos(roll)]], dtype=np.float32)
        R = R_z @ R_y @ R_x

        telem = suite.poll_sensors(pos, R, dt=0.03)

        # Feature vector (9 dims)
        m_raw_norm = telem["m_raw"] / B_EARTH_TOTAL_UT
        m_cal_norm = telem["m_cal"] / B_EARTH_TOTAL_UT
        emf_norm = np.array([telem["emf_ac_rms"] / 50.0], dtype=np.float32)
        att_norm = np.array([pitch / 0.3, roll / 0.3], dtype=np.float32)

        feat = np.concatenate([m_raw_norm, m_cal_norm, emf_norm, att_norm]).astype(np.float32)

        head_rad = math.radians(telem["heading_true"])
        target_head_vec = np.array([math.sin(head_rad), math.cos(head_rad)], dtype=np.float32)

        X_samples.append(feat)
        y_heads.append(target_head_vec)
        y_dists.append(min(15.0, telem["min_wire_dist"]))

    return (np.array(X_samples, dtype=np.float32),
            np.array(y_heads, dtype=np.float32),
            np.array(y_dists, dtype=np.float32))

def train_magnetometer_brain():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"Device: {device}")
    print(f"Training Magnetometer & 60 Hz EMF Hazard Brain (Module 4)...")
    print(f"Hard/Soft-Iron Calibration & Biot-Savart Powerline Inversion")
    print(f"==========================================")

    print("Harvesting 80,000 electromagnetic states across CPU cores...")
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per_worker = (80000 + num_workers - 1) // num_workers

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_collect_mag, [samples_per_worker] * num_workers)

    X_np = np.concatenate([r[0] for r in results], axis=0)[:80000]
    y_hd_np = np.concatenate([r[1] for r in results], axis=0)[:80000]
    y_dst_np = np.concatenate([r[2] for r in results], axis=0)[:80000]
    print(f"Electromagnetic training dataset ready in {time.time() - t0:.2f}s!")

    X = torch.tensor(X_np, device=device)
    y_hd = torch.tensor(y_hd_np, device=device)
    y_dst = torch.tensor(y_dst_np, device=device).unsqueeze(1)

    model = EMFMagnetometerFusionNet().to(device)
    BATCH_SIZE = 2048
    epochs = 10
    num_batches = (len(X) + BATCH_SIZE - 1) // BATCH_SIZE

    optimizer = optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    criterion_hd = nn.MSELoss()
    criterion_dst = nn.MSELoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-3, steps_per_epoch=num_batches, epochs=epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"Optimizing Electromagnetic Inversion Layers (Batch {BATCH_SIZE}, 10 Epochs)...")
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X), device=device)
        total_loss = 0.0

        for i in range(0, len(X), BATCH_SIZE):
            idx = indices[i : i + BATCH_SIZE]
            bx, by_hd, by_dst = X[idx], y_hd[idx], y_dst[idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred_hd, pred_dst = model(bx)
                loss = criterion_hd(pred_hd, by_hd) + 0.2 * criterion_dst(pred_dst, by_dst)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Heading/Distance Loss: {total_loss/num_batches:.5f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Magnetometer/EMF Model compiled in {time.time() - t0:.2f}s!")
    return model, device

# ==========================================
# 3. REAL-TIME 3D MAGNETOMETER & EMF HUD
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

def run_magnetometer_emf_hud(model, device):
    pygame.init()
    VIEW_WIDTH = 700
    VIEW_HEIGHT = 700
    SIDEBAR_WIDTH = 380
    WIDTH = VIEW_WIDTH + SIDEBAR_WIDTH
    HEIGHT = VIEW_HEIGHT

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Module 4: Magnetometer Calibration & 60 Hz High-Voltage EMF Hazard Detection")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("consolas", 18, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 13, bold=True)

    suite = MagnetometerEMFSuite()
    model.eval()

    # Drone 6-DOF Coordinates in World Space (meters)
    drone_pos = np.array([0.0, 1.5, 2.2], dtype=np.float32)
    drone_yaw_deg = 0.0
    drone_pitch_deg = 0.0
    drone_roll_deg = 0.0

    camera_yaw = 0.55
    camera_pitch = 0.38
    center_viewport = (VIEW_WIDTH // 2, VIEW_HEIGHT // 2 + 50)

    running = True
    scatter_samples = deque(maxlen=60)
    emf_osc_history = deque(maxlen=80)

    print("================ CONTROLS ================")
    print(" [W] / [S]      : Fly Toward / Away from Powerline")
    print(" [A] / [D]      : Strafe Left / Right")
    print(" [I] / [K]      : Climb / Descend")
    print(" [J] / [L]      : Yaw Heading (Rotate Compass)")
    print(" [ARROW KEYS]   : Orbit 3D Camera View")
    print(" [R]          : Reset Drone to Base")
    print("==========================================\n")

    while running:
        dt = clock.tick(30) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    drone_pos = np.array([0.0, 1.5, 2.2], dtype=np.float32)
                    drone_yaw_deg = 0.0
                    drone_pitch_deg = 0.0

        keys = pygame.key.get_pressed()
        yaw_rad = math.radians(drone_yaw_deg)
        fwd_vec = np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0], dtype=np.float32)
        right_vec = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=np.float32)

        if keys[pygame.K_w]: drone_pos += fwd_vec * 0.14
        if keys[pygame.K_s]: drone_pos -= fwd_vec * 0.14
        if keys[pygame.K_d]: drone_pos += right_vec * 0.14
        if keys[pygame.K_a]: drone_pos -= right_vec * 0.14
        if keys[pygame.K_i]: drone_pos[2] = min(5.0, drone_pos[2] + 0.08)
        if keys[pygame.K_k]: drone_pos[2] = max(0.4, drone_pos[2] - 0.08)
        if keys[pygame.K_j]: drone_yaw_deg = (drone_yaw_deg - 2.5) % 360.0
        if keys[pygame.K_l]: drone_yaw_deg = (drone_yaw_deg + 2.5) % 360.0

        # Camera rotation
        if keys[pygame.K_LEFT]: camera_yaw -= 0.03
        if keys[pygame.K_RIGHT]: camera_yaw += 0.03
        if keys[pygame.K_UP]: camera_pitch = max(0.1, camera_pitch - 0.02)
        if keys[pygame.K_DOWN]: camera_pitch = min(1.2, camera_pitch + 0.02)

        # Construct Drone Rotation Matrix R in SO(3)
        R_drone = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0],
            [math.sin(yaw_rad),  math.cos(yaw_rad), 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # 1. Poll Electromagnetic Sensors
        telemetry = suite.poll_sensors(drone_pos, R_drone, dt=dt)
        scatter_samples.append((telemetry["m_raw"][0], telemetry["m_raw"][1], telemetry["m_cal"][0], telemetry["m_cal"][1]))
        emf_osc_history.append(telemetry["emf_instantaneous"])

        # 2. Run PyTorch Neural Magnetometer Estimator
        m_raw_norm = telemetry["m_raw"] / B_EARTH_TOTAL_UT
        m_cal_norm = telemetry["m_cal"] / B_EARTH_TOTAL_UT
        emf_norm = np.array([telemetry["emf_ac_rms"] / 50.0], dtype=np.float32)
        att_norm = np.array([drone_pitch_deg / 15.0, drone_roll_deg / 15.0], dtype=np.float32)
        feat_vec = np.concatenate([m_raw_norm, m_cal_norm, emf_norm, att_norm]).astype(np.float32)

        with torch.no_grad():
            feat_t = torch.tensor(feat_vec, device=device).unsqueeze(0)
            pred_hd, pred_dst = model(feat_t)
            hd_vec = pred_hd[0].cpu().numpy()
            nn_heading_deg = (math.degrees(math.atan2(hd_vec[0], hd_vec[1])) + 360.0) % 360.0
            nn_wire_dist = float(pred_dst[0, 0].item())

        screen.fill((10, 12, 18))

        # Dividers
        pygame.draw.line(screen, (35, 45, 60), (VIEW_WIDTH, 0), (VIEW_WIDTH, HEIGHT), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (WIDTH, 70), 2)

        # ==========================================
        # 3D VIEWPORT: POWER TRANSMISSION TOWERS & EMF
        # ==========================================
        cam = (camera_yaw, camera_pitch)

        # 1. Draw 3D Ground Grid (meters)
        for gx in range(-4, 5):
            p1 = np.array([gx * 1.5, -2.0, 0.0])
            p2 = np.array([gx * 1.5, 12.0, 0.0])
            sx1, sy1, _ = project_3d(p1, cam, center_viewport)
            sx2, sy2, _ = project_3d(p2, cam, center_viewport)
            pygame.draw.line(screen, (22, 28, 38), (sx1, sy1), (sx2, sy2), 1)

        for gy in range(-1, 8):
            p1 = np.array([-6.0, gy * 1.5, 0.0])
            p2 = np.array([ 6.0, gy * 1.5, 0.0])
            sx1, sy1, _ = project_3d(p1, cam, center_viewport)
            sx2, sy2, _ = project_3d(p2, cam, center_viewport)
            pygame.draw.line(screen, (22, 28, 38), (sx1, sy1), (sx2, sy2), 1)

        # 2. Draw 500 kV Transmission Towers (Pylons) & Overhead Cables
        pylon_x_positions = [-4.5, 4.5]
        for px in pylon_x_positions:
            base_pt = np.array([px, suite.powerline_y, 0.0])
            top_pt = np.array([px, suite.powerline_y, 4.8])
            cross_pt1 = np.array([px - 0.8, suite.powerline_y, 4.0])
            cross_pt2 = np.array([px + 0.8, suite.powerline_y, 4.0])

            sx_b, sy_b, _ = project_3d(base_pt, cam, center_viewport)
            sx_t, sy_t, _ = project_3d(top_pt, cam, center_viewport)
            sx_c1, sy_c1, _ = project_3d(cross_pt1, cam, center_viewport)
            sx_c2, sy_c2, _ = project_3d(cross_pt2, cam, center_viewport)

            # Steel lattice pylon legs
            pygame.draw.line(screen, (70, 85, 110), (sx_b, sy_b), (sx_t, sy_t), 4)
            pygame.draw.line(screen, (70, 85, 110), (sx_c1, sy_c1), (sx_c2, sy_c2), 3)

        # Draw 3-Phase Transmission Wires
        for wz in suite.powerline_z_wires:
            w1 = np.array([-6.0, suite.powerline_y, wz])
            w2 = np.array([ 6.0, suite.powerline_y, wz])
            wsx1, wsy1, _ = project_3d(w1, cam, center_viewport)
            wsx2, wsy2, _ = project_3d(w2, cam, center_viewport)

            # Wire color: Glowing warning orange if EMF high, else copper
            wire_col = (255, 60, 80) if telemetry["is_emf_warning"] else (255, 180, 40)
            pygame.draw.line(screen, wire_col, (wsx1, wsy1), (wsx2, wsy2), 3)

            # Draw EMF Radiation Induction Zone (Cylindrical Equipotential Shell)
            if telemetry["is_emf_warning"]:
                zone1 = np.array([-6.0, suite.powerline_y, wz + 0.8])
                zone2 = np.array([ 6.0, suite.powerline_y, wz + 0.8])
                zsx1, zsy1, _ = project_3d(zone1, cam, center_viewport)
                zsx2, zsy2, _ = project_3d(zone2, cam, center_viewport)
                pygame.draw.line(screen, (255, 120, 40), (zsx1, zsy1), (zsx2, zsy2), 1)

        # 3. Draw Drone & Body Frame Compass Needle
        hub_sx, hub_sy, _ = project_3d(drone_pos, cam, center_viewport)
        pygame.draw.circle(screen, (0, 240, 255), (hub_sx, hub_sy), 8)
        pygame.draw.circle(screen, (255, 255, 255), (hub_sx, hub_sy), 3)

        # Magnetic field vector needle from drone hub
        b_vec_world = R_drone @ telemetry["m_cal"]
        norm_b = np.linalg.norm(b_vec_world) + 1e-5
        needle_end = drone_pos + (b_vec_world / norm_b) * 1.2
        nsx, nsy, _ = project_3d(needle_end, cam, center_viewport)
        pygame.draw.line(screen, (0, 255, 140), (hub_sx, hub_sy), (nsx, nsy), 3)

        # ==========================================
        # AVIONICS COMPASS & CALIBRATION GAUGES
        # ==========================================
        # 1. Compass Rose Dial (Bottom-Left Viewport)
        comp_center = (110, VIEW_HEIGHT - 110)
        comp_radius = 65
        pygame.draw.circle(screen, (16, 22, 32), comp_center, comp_radius)
        pygame.draw.circle(screen, (35, 50, 70), comp_center, comp_radius, 2)

        # North-South-East-West ticks
        screen.blit(font_small.render("N", True, (0, 255, 140)), (comp_center[0] - 4, comp_center[1] - comp_radius + 4))
        screen.blit(font_small.render("S", True, (160, 180, 205)), (comp_center[0] - 4, comp_center[1] + comp_radius - 14))
        screen.blit(font_small.render("E", True, (160, 180, 205)), (comp_center[0] + comp_radius - 12, comp_center[1] - 6))
        screen.blit(font_small.render("W", True, (160, 180, 205)), (comp_center[0] - comp_radius + 4, comp_center[1] - 6))

        # True heading needle (Cyan)
        h_rad = math.radians(telemetry["heading_true"])
        hx = int(comp_center[0] + math.sin(h_rad) * (comp_radius - 14))
        hy = int(comp_center[1] - math.cos(h_rad) * (comp_radius - 14))
        pygame.draw.line(screen, (0, 240, 255), comp_center, (hx, hy), 3)
        pygame.draw.circle(screen, (0, 240, 255), (hx, hy), 4)

        screen.blit(font_mono.render(f"HDG: {telemetry['heading_true']:05.1f}°", True, (0, 240, 255)), (65, VIEW_HEIGHT - 32))

        # 2. Hard/Soft-Iron Calibration Ellipsoid Scatter (Bottom-Center Viewport)
        scat_center = (330, VIEW_HEIGHT - 110)
        scat_r = 65
        pygame.draw.circle(screen, (16, 22, 32), scat_center, scat_r)
        pygame.draw.circle(screen, (35, 50, 70), scat_center, scat_r, 1)

        # Plot uncalibrated points (Red ellipse) vs Calibrated points (Green sphere)
        for rx, ry, cx, cy in scatter_samples:
            # Scale uT to pixels
            raw_px = int(scat_center[0] + (rx / B_EARTH_TOTAL_UT) * 45)
            raw_py = int(scat_center[1] - (ry / B_EARTH_TOTAL_UT) * 45)
            cal_px = int(scat_center[0] + (cx / B_EARTH_TOTAL_UT) * 45)
            cal_py = int(scat_center[1] - (cy / B_EARTH_TOTAL_UT) * 45)

            pygame.draw.circle(screen, (255, 60, 80), (raw_px, raw_py), 1)
            pygame.draw.circle(screen, (0, 255, 140), (cal_px, cal_py), 1)

        screen.blit(font_small.render("ELLIPSOID CALIBRATION", True, (130, 160, 190)), (260, VIEW_HEIGHT - 32))
        screen.blit(font_small.render("Red=Raw Ellipsoid | Green=Calibrated", True, (160, 180, 205)), (230, VIEW_HEIGHT - 18))

        # 3. 60 Hz EMF Oscilloscope Waveform (Bottom-Right of Viewport)
        osc_x = 480
        osc_y = VIEW_HEIGHT - 150
        osc_w = 190
        osc_h = 75
        pygame.draw.rect(screen, (16, 22, 32), (osc_x, osc_y, osc_w, osc_h), border_radius=4)
        pygame.draw.rect(screen, (35, 50, 70), (osc_x, osc_y, osc_w, osc_h), 1)

        if len(emf_osc_history) >= 2:
            pts = []
            step_x = osc_w / max(1, len(emf_osc_history) - 1)
            for i, val in enumerate(emf_osc_history):
                px = int(osc_x + i * step_x)
                py = int(osc_y + osc_h // 2 - (val / 60.0) * (osc_h * 0.4))
                pts.append((px, py))
            osc_col = (255, 60, 80) if telemetry["is_emf_warning"] else (255, 180, 40)
            pygame.draw.lines(screen, osc_col, False, pts, 2)

        screen.blit(font_small.render("60 Hz AC EMF INDUCTION", True, (255, 180, 40)), (osc_x + 12, osc_y + 6))
        screen.blit(font_small.render(f"{telemetry['emf_ac_rms']:.1f} uT RMS", True, (0, 240, 255)), (osc_x + 12, osc_y + 20))

        # ==========================================
        # ELECTROMAGNETIC TELEMETRY (SIDEBAR)
        # ==========================================
        sb_x = VIEW_WIDTH + 14
        screen.blit(font_large.render("MAGNETOMETER & EMF", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Tri-Axial Fluxgate & 60 Hz Biot-Savart Pickup", True, (130, 160, 190)), (sb_x, 38))

        # --- PANEL 1: GEOMAGNETIC VECTOR & HEADING ---
        p1_y = 80
        screen.blit(font_mono.render("GEOMAGNETIC WMM HEADING:", True, (240, 240, 240)), (sb_x, p1_y))

        screen.blit(font_small.render(f"True Heading (Yaw):  {telemetry['heading_true']:.1f}° (Tilt-Compensated)", True, (0, 255, 140)), (sb_x + 10, p1_y + 24))
        screen.blit(font_small.render(f"Neural Heading Est:  {nn_heading_deg:.1f}° (AI Inversion)", True, (0, 220, 255)), (sb_x + 10, p1_y + 40))
        screen.blit(font_small.render(f"Declination Correction: {DECLINATION_DEG}° (WMM Model)", True, (160, 180, 205)), (sb_x + 10, p1_y + 56))
        screen.blit(font_small.render(f"Dip Angle (Inclin):  {INCLINATION_DIP_DEG:.1f}° (Downward Flux)", True, (160, 180, 205)), (sb_x + 10, p1_y + 72))

        # --- PANEL 2: HARD-IRON & SOFT-IRON CALIBRATION ---
        p2_y = p1_y + 96
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (WIDTH - 14, p2_y), 1)
        screen.blit(font_mono.render("SENSOR CALIBRATION MATRICES:", True, (240, 240, 240)), (sb_x, p2_y + 8))

        m_raw = telemetry["m_raw"]
        m_cal = telemetry["m_cal"]
        screen.blit(font_small.render(f"Raw Mag Vector:  [{m_raw[0]:+.1f}, {m_raw[1]:+.1f}, {m_raw[2]:+.1f}] uT", True, (255, 60, 80)), (sb_x + 10, p2_y + 28))
        screen.blit(font_small.render(f"Calibrated Mag:  [{m_cal[0]:+.1f}, {m_cal[1]:+.1f}, {m_cal[2]:+.1f}] uT", True, (0, 255, 140)), (sb_x + 10, p2_y + 44))
        screen.blit(font_small.render(f"Hard-Iron Offset:[{HARD_IRON_BIAS[0]:+.1f}, {HARD_IRON_BIAS[1]:+.1f}, {HARD_IRON_BIAS[2]:+.1f}] uT", True, (160, 180, 205)), (sb_x + 10, p2_y + 60))
        screen.blit(font_small.render("Algorithm:       Least-Squares Ellipsoid Fit", True, (0, 220, 255)), (sb_x + 10, p2_y + 76))

        # --- PANEL 3: 60 Hz HIGH-VOLTAGE POWERLINE EMF ---
        p3_y = p2_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (WIDTH - 14, p3_y), 1)
        screen.blit(font_mono.render("POWERLINE 60 Hz EMF HAZARD:", True, (240, 240, 240)), (sb_x, p3_y + 8))

        emf_rms = telemetry["emf_ac_rms"]
        w_dist = telemetry["min_wire_dist"]
        emf_col = (255, 50, 60) if telemetry["is_emf_critical"] else ((255, 180, 40) if telemetry["is_emf_warning"] else (0, 255, 140))
        emf_status = "CRITICAL PROXIMITY!" if telemetry["is_emf_critical"] else ("POWERLINE HAZARD ALERT" if telemetry["is_emf_warning"] else "CLEAR AIRSPACE")

        screen.blit(font_small.render(f"60 Hz AC Pickup:    B_rms = {emf_rms:.1f} uT (Biot-Savart)", True, emf_col), (sb_x + 10, p3_y + 28))
        screen.blit(font_small.render(f"Distance to Cable:  d_wire = {w_dist:.2f} m (NN Est: {nn_wire_dist:.2f}m)", True, (0, 220, 255)), (sb_x + 10, p3_y + 44))
        screen.blit(font_small.render(f"Line Parameters:    500 kV Catenary | 750 Amperes", True, (160, 180, 205)), (sb_x + 10, p3_y + 60))
        screen.blit(font_small.render(f"Hazard State:       {emf_status}", True, emf_col), (sb_x + 10, p3_y + 76))

        # EMF bar gauge [0, 50 uT]
        bar_x = sb_x + 10
        bar_y = p3_y + 94
        pygame.draw.rect(screen, (24, 32, 45), (bar_x, bar_y, 320, 8), border_radius=2)
        fill_w = int(320 * min(1.0, max(0.0, emf_rms / 50.0)))
        pygame.draw.rect(screen, emf_col, (bar_x, bar_y, fill_w, 8), border_radius=2)

        # Warning threshold marker at 18 uT
        warn_px = bar_x + int(320 * (EMF_WARNING_THRESHOLD_UT / 50.0))
        pygame.draw.line(screen, (255, 180, 40), (warn_px, bar_y - 2), (warn_px, bar_y + 10), 2)
        screen.blit(font_small.render("18uT Alert", True, (255, 180, 40)), (warn_px - 25, bar_y + 11))

        # --- PANEL 4: DRONE POSITION & SENSOR READINGS ---
        p4_y = bar_y + 26
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (WIDTH - 14, p4_y), 1)
        screen.blit(font_mono.render("AIRCRAFT FLIGHT DISPATCH:", True, (240, 240, 240)), (sb_x, p4_y + 8))

        screen.blit(font_small.render(f"Drone Coordinate:   [{drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f}] m", True, (0, 255, 140)), (sb_x + 10, p4_y + 28))
        screen.blit(font_small.render("Fly toward/away from Y=5.0m to test EMF induction", True, (160, 180, 205)), (sb_x + 10, p4_y + 44))

        # Bottom Telemetry Banner
        bot_y = p4_y + 70
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y), (WIDTH - 14, bot_y), 1)
        if telemetry["is_emf_critical"]:
            alert_str = "EMERGENCY: POWERLINE INVASION ZONE!"
            alert_col = (255, 50, 60)
        elif telemetry["is_emf_warning"]:
            alert_str = "CAUTION: ENERGIZED AC LINE DETECTED"
            alert_col = (255, 180, 40)
        else:
            alert_str = "MAGNETIC COMPASS: TILT-COMPENSATED LOCK"
            alert_col = (0, 255, 140)
        screen.blit(font_mono.render(alert_str, True, alert_col), (sb_x + 8, bot_y + 8))

        # Header UI
        head_txt = font_large.render(f"HDG: {telemetry['heading_true']:05.1f}° | EMF: {emf_rms:.1f}uT | Magnetometer", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("Geomagnetic Navigation: Hard/Soft-Iron Calibration, Tilt Compensation & 60 Hz Biot-Savart Induction", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

# ==========================================
# 4. MAIN ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    mp.freeze_support()
    model, device = train_magnetometer_brain()
    print("Launching Magnetometer Calibration & EMF Hazard Simulator...")
    run_magnetometer_emf_hud(model, device)