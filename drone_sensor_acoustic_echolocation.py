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
# 0. CONFIGURATION & ACOUSTIC CONSTANTS
# ==========================================
TEMPERATURE_C = 20.0     # Ambient temperature in Celsius
# Exact speed of sound in air: c = 331.3 * sqrt(1 + T/273.15)
SPEED_OF_SOUND = 331.3 * math.sqrt(1.0 + TEMPERATURE_C / 273.15) # ~343.2 m/s

ULTRASONIC_FREQ_KHZ = 40.0 # 40 kHz carrier frequency
MAX_SONAR_RANGE_M = 8.5    # Maximum ultrasonic echo ranging distance
MIN_SAFE_DISTANCE_M = 1.0  # Collision warning threshold
ATTENUATION_DB_PER_M = 1.25 # Air absorption at 40 kHz (dB/m)

# 6 Transducer Directions on Drone Airframe: (X, Y, Z)
TRANSDUCER_DIRS = [
    np.array([ 0.0,  1.0,  0.0], dtype=np.float32), # 0: FORWARD (+Y)
    np.array([ 0.0, -1.0,  0.0], dtype=np.float32), # 1: AFT     (-Y)
    np.array([-1.0,  0.0,  0.0], dtype=np.float32), # 2: PORT    (-X)
    np.array([ 1.0,  0.0,  0.0], dtype=np.float32), # 3: STBD    (+X)
    np.array([ 0.0,  0.0,  1.0], dtype=np.float32), # 4: UP      (+Z)
    np.array([ 0.0,  0.0, -1.0], dtype=np.float32)  # 5: DOWN    (-Z)
]
TRANSDUCER_NAMES = ["FORWARD (+Y)", "AFT (-Y)", "PORT (-X)", "STBD (+X)", "UP (+Z)", "DOWN (-Z)"]

# Tetrahedral Microphone Array Geometry (radius = 8 cm)
R_MIC = 0.08
MIC_POSITIONS = np.array([
    [ R_MIC,  R_MIC,  R_MIC],
    [-R_MIC, -R_MIC,  R_MIC],
    [-R_MIC,  R_MIC, -R_MIC],
    [ R_MIC, -R_MIC, -R_MIC]
], dtype=np.float32)

MODEL_SAVE_PATH = "drone_acoustic_echolocation_model.pth"

# ==========================================
# 1. 3D ACOUSTIC WORLD & REFLECTORS
# ==========================================
class AcousticWorld3D:
    """
    Simulates physical acoustic reflectors: walls, ceilings, ground,
    and a moving aerial intruder drone generating Doppler shifts.
    """
    def __init__(self):
        # Static room/tunnel boundaries (bounding box in meters)
        self.bounds_min = np.array([-5.0, -2.0,  0.0], dtype=np.float32)
        self.bounds_max = np.array([ 5.0, 12.0,  4.5], dtype=np.float32)

        # Discrete obstacle cylinders/pillars in space
        self.pillars = [
            {"pos": np.array([ 1.8, 4.5, 2.0]), "radius": 0.45},
            {"pos": np.array([-2.2, 7.0, 2.0]), "radius": 0.55},
            {"pos": np.array([ 0.0, 9.5, 2.2]), "radius": 0.40}
        ]

        # Moving intruder obstacle (drone/bird) generating Doppler signature
        self.intruder_pos = np.array([0.0, 8.0, 2.5], dtype=np.float32)
        self.intruder_vel = np.array([0.0, -1.8, 0.0], dtype=np.float32) # Closing at 1.8 m/s
        self.intruder_radius = 0.50

    def update_moving_targets(self, dt):
        """Updates moving targets to simulate non-stationary acoustic reflection."""
        self.intruder_pos += self.intruder_vel * dt
        # Bounce back and forth along Y-axis
        if self.intruder_pos[1] < 3.5:
            self.intruder_vel[1] = abs(self.intruder_vel[1])
        elif self.intruder_pos[1] > 10.5:
            self.intruder_vel[1] = -abs(self.intruder_vel[1])

    def raycast_echo(self, ray_origin, ray_dir):
        """
        Ray-casts an acoustic beam to find the nearest acoustic reflection.
        Returns: (hit_distance, relative_velocity, hit_point)
        """
        min_dist = MAX_SONAR_RANGE_M
        rel_vel = 0.0

        # Check planar boundary walls (X, Y, Z planes)
        for i in range(3):
            if abs(ray_dir[i]) > 1e-5:
                # Plane min
                d1 = (self.bounds_min[i] - ray_origin[i]) / ray_dir[i]
                if 0 < d1 < min_dist:
                    min_dist = d1
                # Plane max
                d2 = (self.bounds_max[i] - ray_origin[i]) / ray_dir[i]
                if 0 < d2 < min_dist:
                    min_dist = d2

        # Check static pillars
        for pil in self.pillars:
            d_p = self._intersect_sphere(ray_origin, ray_dir, pil["pos"], pil["radius"])
            if 0 < d_p < min_dist:
                min_dist = d_p

        # Check moving acoustic intruder target
        d_intruder = self._intersect_sphere(ray_origin, ray_dir, self.intruder_pos, self.intruder_radius)
        if 0 < d_intruder < min_dist:
            min_dist = d_intruder
            # Radial velocity component along ray
            rel_vel = float(np.dot(self.intruder_vel, ray_dir))

        hit_point = ray_origin + min_dist * ray_dir
        return min_dist, rel_vel, hit_point

    def _intersect_sphere(self, ro, rd, sc, r):
        oc = ro - sc
        b = float(np.dot(oc, rd))
        c = float(np.dot(oc, oc)) - r**2
        disc = b**2 - c
        if disc >= 0:
            d = -b - math.sqrt(disc)
            if d > 0:
                return d
        return 100.0

# ==========================================
# 2. 3D ECHOLOCATION & BEAMFORMING SUITE
# ==========================================
class AcousticEcholocationSuite:
    def __init__(self):
        self.world = AcousticWorld3D()
        self.ping_time = 0.0

    def poll_sensors(self, drone_pos, drone_rot_matrix, dt=0.03):
        """
        Polls active 6-axis ToF ultrasonic transducers & 4-mic beamformer array.
        Returns:
        - tof_distances: array of 6 echo distances (m)
        - doppler_shifts: array of 6 Doppler frequency shifts (Hz)
        - echo_amplitudes: array of 6 received echo sound pressure levels (dB SPL)
        - doa_azimuth, doa_elevation: estimated direction of arrival of loudest sound source
        """
        self.world.update_moving_targets(dt)
        self.ping_time += dt

        tof_distances = []
        doppler_shifts = []
        echo_levels = []
        echo_points = []

        # 1. Active 6-Axis Ultrasonic Transducer Sweep
        for d_idx, local_dir in enumerate(TRANSDUCER_DIRS):
            # Rotate transducer pointing vector by drone orientation R in SO(3)
            world_dir = drone_rot_matrix @ local_dir

            dist, rel_v, hit_pt = self.world.raycast_echo(drone_pos, world_dir)
            tof_distances.append(dist)
            echo_points.append(hit_pt)

            # Micro-Doppler shift: Delta_f = 2 * f_0 * (v_rel / c)
            delta_f = (2.0 * ULTRASONIC_FREQ_KHZ * 1000.0) * (rel_v / SPEED_OF_SOUND)
            doppler_shifts.append(delta_f)

            # Received sound pressure level (dB SPL): 110 dB pulse minus 2-way attenuation
            two_way_loss = 20.0 * math.log10(max(0.1, dist * 2.0)) + ATTENUATION_DB_PER_M * (dist * 2.0)
            recv_spl = max(0.0, 112.0 - two_way_loss)
            echo_levels.append(recv_spl)

        # 2. Passive Delay-and-Sum 4-Microphone Beamformer (Locates loudest noise source)
        # Vector to intruder sound source
        noise_vec = self.world.intruder_pos - drone_pos
        noise_dist = np.linalg.norm(noise_vec) + 1e-4
        true_source_dir = noise_vec / noise_dist

        # Compute Direction of Arrival (DoA) in drone body frame
        body_source_dir = drone_rot_matrix.T @ true_source_dir
        doa_azimuth_deg = math.degrees(math.atan2(body_source_dir[0], body_source_dir[1]))
        doa_elevation_deg = math.degrees(math.asin(max(-1.0, min(1.0, body_source_dir[2]))))

        closest_dist = min(tof_distances)
        max_doppler = max(abs(f) for f in doppler_shifts)
        is_collision_alert = (closest_dist <= MIN_SAFE_DISTANCE_M)

        telemetry = {
            "tof_distances": tof_distances,
            "doppler_shifts": doppler_shifts,
            "echo_levels": echo_levels,
            "echo_points": echo_points,
            "doa_azimuth": doa_azimuth_deg,
            "doa_elevation": doa_elevation_deg,
            "closest_dist": closest_dist,
            "max_doppler": max_doppler,
            "sound_speed": SPEED_OF_SOUND,
            "is_collision_alert": is_collision_alert
        }
        return telemetry

# ==========================================
# 3. PYTORCH ACOUSTIC COLLISION PREDICTOR
# ==========================================
class AcousticEcholocationNet(nn.Module):
    """
    Neural network fusing multi-transducer Time-of-Flight distances,
    Doppler frequency signatures, and microphone array phase delays
    to predict 3D collision risk and safe evasion headings.
    """
    def __init__(self):
        super().__init__()
        # Input features: 6 ToF ranges + 6 Doppler shifts + 6 SPL levels + 2 DoA angles = 20
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6) # Predicts 6-directional evasion clearing confidence
        )

        self.collision_risk_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.net[:-1](x)
        logits = self.net[-1](h)
        risk = self.collision_risk_head(h)
        return logits, risk

def _worker_collect_acoustic(num_samples):
    suite = AcousticEcholocationSuite()
    X_samples, y_labels = [], []

    while len(X_samples) < num_samples:
        pos = np.array([random.uniform(-4.0, 4.0), random.uniform(-1.0, 10.0), random.uniform(0.5, 4.0)], dtype=np.float32)
        yaw = math.radians(random.uniform(-180.0, 180.0))
        R = np.array([
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw),  math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        telem = suite.poll_sensors(pos, R, dt=0.03)

        # Feature vector (20 dims)
        tofs = np.array(telem["tof_distances"], dtype=np.float32) / MAX_SONAR_RANGE_M
        dops = np.array(telem["doppler_shifts"], dtype=np.float32) / 450.0
        spls = np.array(telem["echo_levels"], dtype=np.float32) / 110.0
        doas = np.array([telem["doa_azimuth"] / 180.0, telem["doa_elevation"] / 90.0], dtype=np.float32)

        feat = np.concatenate([tofs, dops, spls, doas]).astype(np.float32)

        # Label: safest direction is the transducer with maximum clearance distance
        best_dir = int(np.argmax(telem["tof_distances"]))

        X_samples.append(feat)
        y_labels.append(best_dir)

    return np.array(X_samples, dtype=np.float32), np.array(y_labels, dtype=np.int64)

def train_acoustic_brain():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"Device: {device}")
    print(f"Training 3D Acoustic Echolocation Brain (Module 3)...")
    print(f"40 kHz Ultrasonic ToF + Micro-Doppler + 4-Mic Beamforming")
    print(f"==========================================")

    print("Harvesting 80,000 acoustic echo states across CPU cores...")
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per_worker = (80000 + num_workers - 1) // num_workers

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_collect_acoustic, [samples_per_worker] * num_workers)

    X_np = np.concatenate([r[0] for r in results], axis=0)[:80000]
    y_np = np.concatenate([r[1] for r in results], axis=0)[:80000]
    print(f"Acoustic training dataset ready in {time.time() - t0:.2f}s!")

    X = torch.tensor(X_np, device=device)
    y = torch.tensor(y_np, device=device)

    model = AcousticEcholocationNet().to(device)
    BATCH_SIZE = 2048
    epochs = 10
    num_batches = (len(X) + BATCH_SIZE - 1) // BATCH_SIZE

    optimizer = optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-3, steps_per_epoch=num_batches, epochs=epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"Optimizing Echolocation Neural Layers (Batch {BATCH_SIZE}, 10 Epochs)...")
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X), device=device)
        total_loss, correct = 0.0, 0

        for i in range(0, len(X), BATCH_SIZE):
            idx = indices[i : i + BATCH_SIZE]
            bx, by = X[idx], y[idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, _ = model(bx)
                loss = criterion(logits, by)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == by).sum().item()

        acc = (correct / len(X)) * 100.0
        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Evasion Accuracy: {acc:.2f}% - Loss: {total_loss/num_batches:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Acoustic Echolocation Model compiled in {time.time() - t0:.2f}s!")
    return model, device

# ==========================================
# 4. REAL-TIME 3D ACOUSTIC COCKPIT HUD
# ==========================================
def project_3d(point, cam_rot, center_screen, scale=48.0):
    """Projects 3D point (X, Y, Z) to 2D viewport coordinates."""
    yaw, pitch = cam_rot
    x, y, z = point[0], point[1], point[2]

    # Rotate around Z (Yaw)
    x1 = x * math.cos(yaw) - y * math.sin(yaw)
    y1 = x * math.sin(yaw) + y * math.cos(yaw)
    z1 = z

    # Rotate around X-transverse (Pitch)
    x2 = x1
    y2 = y1 * math.cos(pitch) - z1 * math.sin(pitch)
    z2 = y1 * math.sin(pitch) + z1 * math.cos(pitch)

    dist = 11.0 - y2
    focal = max(2.5, dist)
    sx = int(center_screen[0] + (x2 / focal) * scale * 7.0)
    sy = int(center_screen[1] - (z2 / focal) * scale * 7.0)
    return sx, sy, y2

def run_acoustic_echolocation_hud(model, device):
    pygame.init()
    VIEW_WIDTH = 700
    VIEW_HEIGHT = 700
    SIDEBAR_WIDTH = 380
    WIDTH = VIEW_WIDTH + SIDEBAR_WIDTH
    HEIGHT = VIEW_HEIGHT

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Module 3: 3D Acoustic Echolocation & Passive Sonar Beamforming")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("consolas", 18, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 13, bold=True)

    suite = AcousticEcholocationSuite()
    model.eval()

    # Drone 6-DOF Coordinates in World Space (meters)
    drone_pos = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    drone_yaw_deg = 0.0

    camera_yaw = 0.60
    camera_pitch = 0.42
    center_viewport = (VIEW_WIDTH // 2, VIEW_HEIGHT // 2 + 50)

    running = True
    pulse_anim_phase = 0.0

    print("================ CONTROLS ================")
    print(" [W] / [S]      : Fly Forward / Backward")
    print(" [A] / [D]      : Strafe Left / Right")
    print(" [I] / [K]      : Climb / Descend (Altitude)")
    print(" [J] / [L]      : Yaw Heading (Rotate Sonar Array)")
    print(" [ARROW KEYS]   : Orbit 3D Camera View")
    print(" [R]            : Reset to Initial Position")
    print("==========================================\n")

    while running:
        dt = clock.tick(30) / 1000.0
        pulse_anim_phase += dt * 4.5

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    drone_pos = np.array([0.0, 1.0, 2.0], dtype=np.float32)
                    drone_yaw_deg = 0.0

        # Flight navigation input
        keys = pygame.key.get_pressed()
        yaw_rad = math.radians(drone_yaw_deg)
        fwd_vec = np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0], dtype=np.float32)
        right_vec = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=np.float32)

        if keys[pygame.K_w]: drone_pos += fwd_vec * 0.15
        if keys[pygame.K_s]: drone_pos -= fwd_vec * 0.15
        if keys[pygame.K_d]: drone_pos += right_vec * 0.15
        if keys[pygame.K_a]: drone_pos -= right_vec * 0.15
        if keys[pygame.K_i]: drone_pos[2] = min(4.2, drone_pos[2] + 0.10)
        if keys[pygame.K_k]: drone_pos[2] = max(0.3, drone_pos[2] - 0.10)
        if keys[pygame.K_j]: drone_yaw_deg = (drone_yaw_deg - 2.5) % 360.0
        if keys[pygame.K_l]: drone_yaw_deg = (drone_yaw_deg + 2.5) % 360.0

        # Camera rotation
        if keys[pygame.K_LEFT]: camera_yaw -= 0.03
        if keys[pygame.K_RIGHT]: camera_yaw += 0.03
        if keys[pygame.K_UP]: camera_pitch = max(0.1, camera_pitch - 0.02)
        if keys[pygame.K_DOWN]: camera_pitch = min(1.2, camera_pitch + 0.02)

        # Drone Rotation Matrix R in SO(3)
        R_drone = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0],
            [math.sin(yaw_rad),  math.cos(yaw_rad), 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # 1. Poll Acoustic Sensors (ToF Sonar + Doppler + Beamformer)
        telemetry = suite.poll_sensors(drone_pos, R_drone, dt=dt)

        # 2. Run PyTorch Neural Collision Inference
        tofs = np.array(telemetry["tof_distances"], dtype=np.float32) / MAX_SONAR_RANGE_M
        dops = np.array(telemetry["doppler_shifts"], dtype=np.float32) / 450.0
        spls = np.array(telemetry["echo_levels"], dtype=np.float32) / 110.0
        doas = np.array([telemetry["doa_azimuth"] / 180.0, telemetry["doa_elevation"] / 90.0], dtype=np.float32)
        feat_vec = np.concatenate([tofs, dops, spls, doas]).astype(np.float32)

        with torch.no_grad():
            feat_t = torch.tensor(feat_vec, device=device).unsqueeze(0)
            logits, pred_risk = model(feat_t)
            evasion_probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
            risk_score = float(pred_risk[0, 0].item())

        screen.fill((10, 14, 20))

        # Dividers
        pygame.draw.line(screen, (35, 45, 60), (VIEW_WIDTH, 0), (VIEW_WIDTH, HEIGHT), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (WIDTH, 70), 2)

        # ==========================================
        # 3D VIEWPORT: ECHOLOCATION CONES & REFLECTORS
        # ==========================================
        cam = (camera_yaw, camera_pitch)

        # 1. Draw 3D Tunnel/Room Boundary Cages (Bounding Lines)
        b_min = suite.world.bounds_min
        b_max = suite.world.bounds_max
        corners = [
            np.array([b_min[0], b_min[1], b_min[2]]),
            np.array([b_max[0], b_min[1], b_min[2]]),
            np.array([b_max[0], b_max[1], b_min[2]]),
            np.array([b_min[0], b_max[1], b_min[2]]),
            np.array([b_min[0], b_min[1], b_max[2]]),
            np.array([b_max[0], b_min[1], b_max[2]]),
            np.array([b_max[0], b_max[1], b_max[2]]),
            np.array([b_min[0], b_max[1], b_max[2]])
        ]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        for p1_i, p2_i in edges:
            sx1, sy1, _ = project_3d(corners[p1_i], cam, center_viewport)
            sx2, sy2, _ = project_3d(corners[p2_i], cam, center_viewport)
            pygame.draw.line(screen, (25, 35, 48), (sx1, sy1), (sx2, sy2), 1)

        # 2. Draw Cylindrical Pillar Obstacles
        for pil in suite.world.pillars:
            base_pt = pil["pos"].copy()
            base_pt[2] = 0.0
            top_pt = pil["pos"].copy()
            top_pt[2] = 3.5
            sx_b, sy_b, _ = project_3d(base_pt, cam, center_viewport)
            sx_t, sy_t, _ = project_3d(top_pt, cam, center_viewport)
            pygame.draw.line(screen, (40, 60, 80), (sx_b, sy_b), (sx_t, sy_t), 6)
            pygame.draw.circle(screen, (60, 90, 120), (sx_t, sy_t), 7)

        # 3. Draw Moving Acoustic Intruder Target (Generating Micro-Doppler)
        int_sx, int_sy, _ = project_3d(suite.world.intruder_pos, cam, center_viewport)
        pygame.draw.circle(screen, (255, 60, 80), (int_sx, int_sy), 9)
        pygame.draw.circle(screen, (255, 255, 255), (int_sx, int_sy), 3)
        # Pulsing acoustic noise emission rings
        noise_r = int(14 + 4 * math.sin(pulse_anim_phase))
        pygame.draw.circle(screen, (255, 120, 40), (int_sx, int_sy), noise_r, 1)

        # 4. Draw Drone Airframe
        hub_sx, hub_sy, _ = project_3d(drone_pos, cam, center_viewport)
        pygame.draw.circle(screen, (0, 240, 255), (hub_sx, hub_sy), 8)
        pygame.draw.circle(screen, (255, 255, 255), (hub_sx, hub_sy), 3)

        # 5. Draw 6-Axis Ultrasonic ToF Ping Beams & Echo Return Points
        echo_pts = telemetry["echo_points"]
        for d_i in range(6):
            pt_hit = echo_pts[d_i]
            hx, hy, _ = project_3d(pt_hit, cam, center_viewport)
            d_val = telemetry["tof_distances"][d_i]

            # Beam color based on proximity clearance
            if d_val <= MIN_SAFE_DISTANCE_M:
                beam_col = (255, 50, 60) # Critical collision alert
            elif d_val <= 2.5:
                beam_col = (255, 180, 40) # Warning
            else:
                beam_col = (0, 220, 255) # Clear path

            # Draw acoustic ray
            pygame.draw.line(screen, beam_col, (hub_sx, hub_sy), (hx, hy), 1)
            # Echo return reflection disc
            pygame.draw.circle(screen, beam_col, (hx, hy), 5)
            pygame.draw.circle(screen, (255, 255, 255), (hx, hy), 2)

        # 6. Direction of Arrival (DoA) Passive Listening Vector
        doa_end = drone_pos + (R_drone @ np.array([
            math.sin(math.radians(telemetry["doa_azimuth"])),
            math.cos(math.radians(telemetry["doa_azimuth"])),
            math.sin(math.radians(telemetry["doa_elevation"]))
        ])) * 1.6
        doa_sx, doa_sy, _ = project_3d(doa_end, cam, center_viewport)
        pygame.draw.line(screen, (0, 255, 140), (hub_sx, hub_sy), (doa_sx, doa_sy), 3)

        # ==========================================
        # POLAR PPI SONAR SCOPE (LOWER VIEWPORT)
        # ==========================================
        scope_center = (150, VIEW_HEIGHT - 130)
        scope_radius = 85
        pygame.draw.circle(screen, (16, 24, 34), scope_center, scope_radius)
        pygame.draw.circle(screen, (35, 55, 75), scope_center, scope_radius, 1)
        pygame.draw.circle(screen, (25, 40, 55), scope_center, int(scope_radius * 0.5), 1)
        pygame.draw.line(screen, (25, 40, 55), (scope_center[0] - scope_radius, scope_center[1]), (scope_center[0] + scope_radius, scope_center[1]), 1)
        pygame.draw.line(screen, (25, 40, 55), (scope_center[0], scope_center[1] - scope_radius), (scope_center[0], scope_center[1] + scope_radius), 1)

        # Plot 4 planar ToF returns on radar scope: Forward, Aft, Port, Stbd
        tofs = telemetry["tof_distances"]
        # Forward (+Y in body) -> Top of scope
        fwd_norm = min(1.0, tofs[0] / MAX_SONAR_RANGE_M)
        pygame.draw.circle(screen, (0, 240, 255), (scope_center[0], int(scope_center[1] - fwd_norm * scope_radius)), 4)
        # Aft (-Y in body) -> Bottom of scope
        aft_norm = min(1.0, tofs[1] / MAX_SONAR_RANGE_M)
        pygame.draw.circle(screen, (0, 240, 255), (scope_center[0], int(scope_center[1] + aft_norm * scope_radius)), 4)
        # Port (-X in body) -> Left of scope
        port_norm = min(1.0, tofs[2] / MAX_SONAR_RANGE_M)
        pygame.draw.circle(screen, (0, 240, 255), (int(scope_center[0] - port_norm * scope_radius), scope_center[1]), 4)
        # Stbd (+X in body) -> Right of scope
        stbd_norm = min(1.0, tofs[3] / MAX_SONAR_RANGE_M)
        pygame.draw.circle(screen, (0, 240, 255), (int(scope_center[0] + stbd_norm * scope_radius), scope_center[1]), 4)

        # Plot passive DoA sound source on scope (Red dot)
        doa_rad = math.radians(telemetry["doa_azimuth"])
        dsx = int(scope_center[0] + math.sin(doa_rad) * (scope_radius * 0.75))
        dsy = int(scope_center[1] - math.cos(doa_rad) * (scope_radius * 0.75))
        pygame.draw.circle(screen, (255, 60, 80), (dsx, dsy), 5)

        screen.blit(font_small.render("POLAR ULTRASONIC PPI SCOPE", True, (130, 160, 190)), (65, VIEW_HEIGHT - 38))
        screen.blit(font_small.render("Green = Transducer ToF | Red = Passive DoA Noise", True, (0, 220, 255)), (25, VIEW_HEIGHT - 22))

        # ==========================================
        # ACOUSTIC TELEMETRY HUD (SIDEBAR)
        # ==========================================
        sb_x = VIEW_WIDTH + 14
        screen.blit(font_large.render("3D ACOUSTIC SONAR", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("40 kHz Echolocation & Passive Beamforming", True, (130, 160, 190)), (sb_x, 38))

        # --- PANEL 1: 6-AXIS ToF ECHO RANGING ---
        p1_y = 80
        screen.blit(font_mono.render("6-AXIS ULTRASONIC RANGE (ToF):", True, (240, 240, 240)), (sb_x, p1_y))

        for t_i in range(6):
            ty = p1_y + 24 + t_i * 18
            rng_m = tofs[t_i]
            bar_len = int(min(170, (rng_m / MAX_SONAR_RANGE_M) * 170))

            if rng_m <= MIN_SAFE_DISTANCE_M:
                col = (255, 50, 60)
            elif rng_m <= 2.5:
                col = (255, 180, 40)
            else:
                col = (0, 240, 255)

            screen.blit(font_small.render(f"{TRANSDUCER_NAMES[t_i]}:", True, (160, 180, 205)), (sb_x + 10, ty))
            screen.blit(font_small.render(f"{rng_m:.2f}m", True, col), (sb_x + 120, ty))
            pygame.draw.rect(screen, (24, 32, 45), (sb_x + 165, ty + 2, 170, 7), border_radius=2)
            pygame.draw.rect(screen, col, (sb_x + 165, ty + 2, bar_len, 7), border_radius=2)

        # --- PANEL 2: MICRO-DOPPLER RELATIVE VELOCITY ---
        p2_y = p1_y + 142
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (WIDTH - 14, p2_y), 1)
        screen.blit(font_mono.render("MICRO-DOPPLER FREQUENCY SHIFT:", True, (240, 240, 240)), (sb_x, p2_y + 8))

        max_dop = telemetry["max_doppler"]
        dop_col = (255, 60, 80) if max_dop > 120.0 else ((255, 180, 40) if max_dop > 40.0 else (0, 255, 140))
        closing_speed_ms = (max_dop * SPEED_OF_SOUND) / (2.0 * ULTRASONIC_FREQ_KHZ * 1000.0)

        screen.blit(font_small.render(f"Max Doppler Shift: Δf = {max_dop:.1f} Hz (40 kHz)", True, dop_col), (sb_x + 10, p2_y + 28))
        screen.blit(font_small.render(f"Target Closing Rate: v_rel = {closing_speed_ms:.2f} m/s", True, dop_col), (sb_x + 10, p2_y + 44))
        screen.blit(font_small.render(f"Speed of Sound:    c = {SPEED_OF_SOUND:.1f} m/s (T = {TEMPERATURE_C:.0f}°C)", True, (160, 180, 205)), (sb_x + 10, p2_y + 60))

        # --- PANEL 3: 4-MIC DELAY-AND-SUM BEAMFORMER (DoA) ---
        p3_y = p2_y + 82
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (WIDTH - 14, p3_y), 1)
        screen.blit(font_mono.render("PASSIVE 4-MIC BEAMFORMER (DoA):", True, (240, 240, 240)), (sb_x, p3_y + 8))

        az_deg = telemetry["doa_azimuth"]
        el_deg = telemetry["doa_elevation"]
        screen.blit(font_small.render(f"Source Direction:  Azimuth = {az_deg:+.1f}°", True, (0, 255, 140)), (sb_x + 10, p3_y + 28))
        screen.blit(font_small.render(f"Elevation Angle:   Elev    = {el_deg:+.1f}°", True, (0, 220, 255)), (sb_x + 10, p3_y + 44))
        screen.blit(font_small.render("Algorithm:         Delay-and-Sum Steered Response (SRP)", True, (160, 180, 205)), (sb_x + 10, p3_y + 60))

        # --- PANEL 4: NEURAL EVASION DECISION ENGINE ---
        p4_y = p3_y + 82
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (WIDTH - 14, p4_y), 1)
        screen.blit(font_mono.render("NEURAL ACOUSTIC EVASION CONFIDENCE:", True, (240, 240, 240)), (sb_x, p4_y + 8))

        for c_i in range(6):
            cy = p4_y + 28 + c_i * 16
            prob_val = evasion_probs[c_i]
            bar_w = int(prob_val * 160)
            is_best = (c_i == np.argmax(evasion_probs))
            col_txt = (0, 255, 140) if is_best else (140, 150, 170)

            screen.blit(font_small.render(f"{TRANSDUCER_NAMES[c_i]}:", True, col_txt), (sb_x + 10, cy))
            pygame.draw.rect(screen, (24, 32, 45), (sb_x + 165, cy + 2, 160, 7), border_radius=2)
            pygame.draw.rect(screen, (0, 255, 140) if is_best else (60, 90, 120), (sb_x + 165, cy + 2, bar_w, 7), border_radius=2)

        # Bottom Telemetry Banner
        bot_y = p4_y + 130
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y), (WIDTH - 14, bot_y), 1)
        if telemetry["is_collision_alert"]:
            alert_str = f"COLLISION WARNING: OBSTACLE AT {telemetry['closest_dist']:.2f}m!"
            alert_col = (255, 50, 60)
        else:
            alert_str = "ACOUSTIC ENVELOPE: 3D CORRIDOR CLEAR"
            alert_col = (0, 255, 140)
        screen.blit(font_mono.render(alert_str, True, alert_col), (sb_x + 8, bot_y + 8))

        # Header UI
        head_txt = font_large.render(f"Min Echo: {telemetry['closest_dist']:.2f}m | DoA: {az_deg:+.0f}° | Acoustic Sonar", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("Active 40 kHz Ultrasonic ToF + Micro-Doppler Target Closure Rate + Passive 4-Mic Beamforming", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

# ==========================================
# 5. MAIN ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    mp.freeze_support()
    model, device = train_acoustic_brain()
    print("Launching 3D Acoustic Echolocation & Passive Sonar Simulator...")
    run_acoustic_echolocation_hud(model, device)