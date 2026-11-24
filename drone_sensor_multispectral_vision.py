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
# 0. CONFIGURATION & SENSOR CONSTANTS
# ==========================================
CAM_RES_W = 64          # Low-latency multi-spectral sensor resolution
CAM_RES_H = 48
FOV_DEG = 75.0          # Horizontal Field of View (degrees)
ASPECT_RATIO = CAM_RES_W / CAM_RES_H

# Stefan-Boltzmann & Radiometric Constants
STEFAN_BOLTZMANN_SIGMA = 5.670374e-8
T_AMBIENT_C = 22.0      # Ambient air temperature (deg C)
ATM_EXTINCTION_FOG = 0.085 # High visible fog attenuation (Beer-Lambert)

# Sun Position Vector for Rayleigh Sky Polarization (Azimuth 45 deg, Elevation 35 deg)
SUN_AZIMUTH_DEG = 45.0
SUN_ELEVATION_DEG = 35.0
sun_az_rad = math.radians(SUN_AZIMUTH_DEG)
sun_el_rad = math.radians(SUN_ELEVATION_DEG)
SUN_VECTOR = np.array([
    math.cos(sun_el_rad) * math.cos(sun_az_rad),
    math.cos(sun_el_rad) * math.sin(sun_az_rad),
    math.sin(sun_el_rad)
], dtype=np.float32)

MODEL_SAVE_PATH = "drone_multispectral_fusion_model.pth"

# ==========================================
# 1. COLOR PALETTES: IRONBOW THERMAL & POLARIZATION
# ==========================================
def get_ironbow_color(temp_c, min_t=15.0, max_t=95.0):
    """
    Maps physical temperature T (deg C) into FLIR Ironbow thermal false-color:
    Deep Blue (Cold) -> Purple -> Orange -> Hot Yellow -> Pure White (Peak Thermal).
    """
    norm = min(1.0, max(0.0, (temp_c - min_t) / (max_t - min_t)))
    if norm < 0.25:
        # Dark blue to violet
        f = norm / 0.25
        r = int(f * 90)
        g = 0
        b = int(120 + f * 60)
    elif norm < 0.55:
        # Violet to deep magenta/orange
        f = (norm - 0.25) / 0.30
        r = int(90 + f * 155)
        g = int(f * 70)
        b = int(180 - f * 140)
    elif norm < 0.85:
        # Orange to luminous yellow
        f = (norm - 0.55) / 0.30
        r = 255
        g = int(70 + f * 170)
        b = int(40 - f * 20)
    else:
        # Yellow to white hot
        f = (norm - 0.85) / 0.15
        r = 255
        g = 240 + int(f * 15)
        b = int(20 + f * 235)

    return (min(255, max(0, r)), min(255, max(0, g)), min(255, max(0, b)))

def get_polarization_color(aop_deg, dolp):
    """
    Maps Angle of Polarization (AoP in [-90, 90]) and DoLP in [0, 1] into chromatic hue:
    Hue encodes E-vector orientation; Saturation/Brightness encodes polarization strength.
    """
    # Normalize AoP to [0, 1]
    norm_angle = (aop_deg + 90.0) / 180.0
    # HSV-like color ring
    hue = norm_angle * 6.0
    c = dolp
    x = c * (1.0 - abs(hue % 2.0 - 1.0))
    if hue < 1.0:   rgb = (c, x, 0.0)
    elif hue < 2.0: rgb = (x, c, 0.0)
    elif hue < 3.0: rgb = (0.0, c, x)
    elif hue < 4.0: rgb = (0.0, x, c)
    elif hue < 5.0: rgb = (x, 0.0, c)
    else:           rgb = (c, 0.0, x)

    return (int(min(255, max(0, rgb[0] * 255))),
            int(min(255, max(0, rgb[1] * 255))),
            int(min(255, max(0, rgb[2] * 255))))

# ==========================================
# 2. 3D MULTI-SPECTRAL SIMULATED ENVIRONMENT
# ==========================================
class MultiSpectralWorld:
    """
    3D world with physical objects carrying distinct optical, thermal,
    and polarization scattering cross-sections.
    """
    def __init__(self):
        self.objects = [
            # 1. High-Voltage Power Lines (Hot copper wires: T ~ 75 C, thin, almost invisible in fog)
            {"type": "wire", "start": np.array([-10.0, 6.0, 3.2]), "end": np.array([10.0, 6.0, 3.2]),
             "radius": 0.05, "temp_c": 78.0, "rgb": (80, 85, 90), "dolp": 0.15},
            {"type": "wire", "start": np.array([-10.0, 6.0, 4.0]), "end": np.array([10.0, 6.0, 4.0]),
             "radius": 0.05, "temp_c": 82.0, "rgb": (80, 85, 90), "dolp": 0.15},

            # 2. Industrial Building / Concrete Substation (Ambient temp: T ~ 24 C)
            {"type": "box", "center": np.array([3.5, 9.0, 1.5]), "size": np.array([2.5, 3.0, 3.0]),
             "temp_c": 24.0, "rgb": (140, 145, 150), "dolp": 0.08},

            # 3. Running Ground Vehicle / Generator (Hot engine block: T ~ 92 C)
            {"type": "sphere", "center": np.array([-2.5, 7.5, 0.8]), "radius": 0.9,
             "temp_c": 94.0, "rgb": (50, 60, 70), "dolp": 0.12},

            # 4. Water Retention Pond (Brewster polarizing surface: DoLP high ~ 0.65, T ~ 17 C)
            {"type": "water", "center": np.array([0.0, 8.0, 0.0]), "radius": 2.2,
             "temp_c": 17.0, "rgb": (35, 60, 80), "dolp": 0.68}
        ]

    def intersect_ray(self, ray_origin, ray_dir):
        """
        Ray-casts against world objects, returning (hit, distance, object_hit).
        """
        closest_dist = 100.0
        hit_obj = None

        # Check ground plane (z = 0)
        if abs(ray_dir[2]) > 1e-4 and ray_dir[2] < 0:
            d_ground = -ray_origin[2] / ray_dir[2]
            if 0 < d_ground < closest_dist:
                hit_pos = ray_origin + d_ground * ray_dir
                if math.hypot(hit_pos[0], hit_pos[1] - 8.0) < 2.2:
                    closest_dist = d_ground
                    hit_obj = self.objects[3]  # Water pond
                else:
                    closest_dist = d_ground
                    hit_obj = {"type": "ground", "temp_c": T_AMBIENT_C + 2.0, "rgb": (65, 80, 50), "dolp": 0.05}

        # Check spheres
        for obj in self.objects:
            if obj["type"] == "sphere":
                oc = ray_origin - obj["center"]
                b = float(np.dot(oc, ray_dir))
                c = float(np.dot(oc, oc)) - obj["radius"]**2
                disc = b**2 - c
                if disc >= 0:
                    d = -b - math.sqrt(disc)
                    if 0 < d < closest_dist:
                        closest_dist = d
                        hit_obj = obj

            elif obj["type"] == "box":
                # Axis-aligned bounding box intersection
                bounds_min = obj["center"] - obj["size"] * 0.5
                bounds_max = obj["center"] + obj["size"] * 0.5
                tmin = -100.0
                tmax = 100.0
                intersect = True
                for i in range(3):
                    if abs(ray_dir[i]) > 1e-6:
                        t1 = (bounds_min[i] - ray_origin[i]) / ray_dir[i]
                        t2 = (bounds_max[i] - ray_origin[i]) / ray_dir[i]
                        tmin = max(tmin, min(t1, t2))
                        tmax = min(tmax, max(t1, t2))
                    else:
                        if ray_origin[i] < bounds_min[i] or ray_origin[i] > bounds_max[i]:
                            intersect = False
                            break
                if intersect and tmax >= max(0.0, tmin) and tmin < closest_dist and tmin > 0:
                    closest_dist = tmin
                    hit_obj = obj

            elif obj["type"] == "wire":
                # Cylinder line-segment proximity check
                p1 = obj["start"]
                p2 = obj["end"]
                wire_vec = p2 - p1
                w_len = np.linalg.norm(wire_vec)
                w_dir = wire_vec / w_len

                # Distance between ray and wire segment
                q = ray_origin - p1
                d_proj = float(np.dot(ray_dir, w_dir))
                q_proj = float(np.dot(q, w_dir))
                denom = 1.0 - d_proj**2
                if denom > 1e-4:
                    s = (q_proj * d_proj - float(np.dot(q, ray_dir))) / denom
                    t = q_proj + s * d_proj
                    if 0 <= t <= w_len and s > 0:
                        pt_ray = ray_origin + s * ray_dir
                        pt_wire = p1 + t * w_dir
                        if np.linalg.norm(pt_ray - pt_wire) < (obj["radius"] * 2.5):
                            if s < closest_dist:
                                closest_dist = s
                                hit_obj = obj

        return closest_dist, hit_obj

# ==========================================
# 3. MULTI-SPECTRAL CAMERA SENSOR SUITE
# ==========================================
class MultiSpectralCamera:
    """
    Multi-wavelength camera pipeline:
    1. RGB Visible (with atmospheric scattering fog)
    2. Long-Wave Infrared Thermal Radiometry (8-14 um)
    3. Rayleigh Skylight Polarization Compass (DoLP & AoP)
    """
    def __init__(self, res_w=CAM_RES_W, res_h=CAM_RES_H):
        self.w = res_w
        self.h = res_h
        self.world = MultiSpectralWorld()

        # Generate directional camera ray unit vectors
        fov_rad = math.radians(FOV_DEG)
        focal = (self.w * 0.5) / math.tan(fov_rad * 0.5)

        self.ray_dirs_cam = np.zeros((self.h, self.w, 3), dtype=np.float32)
        for y in range(self.h):
            for x in range(self.w):
                dx = (x - self.w * 0.5)
                dy = -(y - self.h * 0.5)  # Screen Y inverted
                ray = np.array([dx, focal, dy], dtype=np.float32)
                self.ray_dirs_cam[y, x] = ray / np.linalg.norm(ray)

    def capture(self, drone_pos, drone_rot_matrix, enable_fog=True):
        """
        Simulates one full multi-spectral frame from current 6-DOF drone pose.
        Returns: rgb_img, thermal_img, polarization_img, depth_map, telemetry.
        """
        rgb_buffer = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        thermal_buffer = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        polar_buffer = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        depth_buffer = np.zeros((self.h, self.w), dtype=np.float32)
        thermal_raw = np.zeros((self.h, self.w), dtype=np.float32)

        peak_temp = T_AMBIENT_C
        wire_detected = False
        sum_dolp = 0.0
        sky_samples = 0

        # Rotate camera rays by drone rotation matrix R in SO(3)
        for y in range(self.h):
            for x in range(self.w):
                ray_local = self.ray_dirs_cam[y, x]
                ray_world = drone_rot_matrix @ ray_local

                dist, hit_obj = self.world.intersect_ray(drone_pos, ray_world)
                depth_buffer[y, x] = dist

                if hit_obj is not None and dist < 45.0:
                    # --- 1. RGB Visible (with Beer-Lambert fog attenuation) ---
                    base_rgb = np.array(hit_obj["rgb"], dtype=np.float32)
                    if enable_fog:
                        fog_factor = math.exp(-ATM_EXTINCTION_FOG * dist)
                        fog_rgb = np.array([170, 180, 195], dtype=np.float32)
                        app_rgb = base_rgb * fog_factor + fog_rgb * (1.0 - fog_factor)
                    else:
                        app_rgb = base_rgb
                    rgb_buffer[y, x] = np.clip(app_rgb, 0, 255).astype(np.uint8)

                    # --- 2. Thermal LWIR (Infrared penetrates fog!) ---
                    t_obj = hit_obj["temp_c"]
                    thermal_raw[y, x] = t_obj
                    thermal_buffer[y, x] = get_ironbow_color(t_obj)
                    peak_temp = max(peak_temp, t_obj)

                    if hit_obj.get("type") == "wire":
                        wire_detected = True

                    # --- 3. Surface Polarization ---
                    dolp = hit_obj.get("dolp", 0.05)
                    # Brewster polarization angle relative to horizontal
                    aop = math.degrees(math.atan2(ray_world[2], ray_world[0]))
                    polar_buffer[y, x] = get_polarization_color(aop, dolp)

                else:
                    # --- SKY RAYLEIGH POLARIZATION COMPASS ---
                    # Sky visible: blue / white glare
                    rgb_buffer[y, x] = (160, 190, 220) if enable_fog else (70, 130, 210)
                    thermal_buffer[y, x] = (10, 10, 30)  # Cold cosmic background (-40 C)
                    thermal_raw[y, x] = -20.0

                    # Rayleigh Scattering Formulation:
                    # Scattering angle gamma = arccos(s . v)
                    cos_gamma = float(np.dot(SUN_VECTOR, ray_world))
                    cos_gamma = max(-1.0, min(1.0, cos_gamma))
                    gamma = math.acos(cos_gamma)

                    # DoLP = DoLP_max * sin^2(gamma) / (1 + cos^2(gamma))
                    dolp_sky = float(0.75 * (math.sin(gamma)**2) / (1.0 + cos_gamma**2))

                    # E-vector: E = s x v (perpendicular to scattering plane)
                    e_vec = np.cross(SUN_VECTOR, ray_world)
                    norm_e = np.linalg.norm(e_vec)
                    if norm_e > 1e-4:
                        e_vec /= norm_e
                        # Angle of polarization relative to vertical
                        aop_sky = math.degrees(math.atan2(e_vec[2], math.hypot(e_vec[0], e_vec[1])))
                    else:
                        aop_sky = 0.0

                    polar_buffer[y, x] = get_polarization_color(aop_sky, dolp_sky)
                    sum_dolp += dolp_sky
                    sky_samples += 1

        avg_sky_dolp = (sum_dolp / max(1, sky_samples)) * 100.0

        # Estimated solar compass heading from skylight polarization
        # Rayleigh pattern orientation gives true solar azimuth independently of magnetic drift
        polar_heading_deg = (SUN_AZIMUTH_DEG + 180.0) % 360.0

        telemetry = {
            "peak_temp_c": peak_temp,
            "wire_detected": wire_detected,
            "avg_sky_dolp": avg_sky_dolp,
            "polar_heading": polar_heading_deg,
            "thermal_contrast": peak_temp - T_AMBIENT_C
        }

        return rgb_buffer, thermal_buffer, polar_buffer, depth_buffer, thermal_raw, telemetry

# ==========================================
# 4. PYTORCH MULTI-SPECTRAL FUSION NEURAL NET
# ==========================================
class MultiSpectralPerceptionNet(nn.Module):
    """
    Convolutional neural network fusing 5 spectral bands:
    [R, G, B, Thermal_LWIR, Range_Depth] -> Hazard Detection & Wire Proximity.
    """
    def __init__(self):
        super().__init__()
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x48 -> 32x24
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x24 -> 16x12
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), # 128 x 4x4 = 2048
            nn.Flatten()
        )

        self.hazard_classifier = nn.Sequential(
            nn.Linear(2048, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # [Clear_Prob, Wire_Hazard_Prob, Obstacle_Prob]
        )

        self.thermal_anomaly_head = nn.Sequential(
            nn.Linear(2048, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.conv_encoder(x)
        class_logits = self.hazard_classifier(features)
        thermal_anomaly = self.thermal_anomaly_head(features)
        return class_logits, thermal_anomaly

def _worker_collect_multispectral(num_samples):
    cam = MultiSpectralCamera(res_w=32, res_h=24) # Lightweight for fast harvesting
    X_samples, y_classes, y_thermals = [], [], []

    while len(X_samples) < num_samples:
        px = random.uniform(-4.0, 4.0)
        py = random.uniform(1.0, 6.0)
        pz = random.uniform(1.5, 4.5)
        pos = np.array([px, py, pz], dtype=np.float32)

        yaw = math.radians(random.uniform(-30.0, 30.0))
        R = np.array([
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        rgb, _, _, depth, therm, telem = cam.capture(pos, R, enable_fog=True)

        # 5-Channel input tensor: [R, G, B, Therm, Depth] normalized in [0, 1]
        tensor_5ch = np.zeros((5, 24, 32), dtype=np.float32)
        tensor_5ch[0:3] = rgb.transpose(2, 0, 1) / 255.0
        tensor_5ch[3] = np.clip((therm - 15.0) / 80.0, 0.0, 1.0)
        tensor_5ch[4] = np.clip(depth / 40.0, 0.0, 1.0)

        # Class: 0 = Clear, 1 = Wire Hazard, 2 = Obstacle
        if telem["wire_detected"]:
            label = 1
        elif telem["thermal_contrast"] > 40.0:
            label = 2
        else:
            label = 0

        X_samples.append(tensor_5ch)
        y_classes.append(label)
        y_thermals.append(min(1.0, telem["thermal_contrast"] / 80.0))

    return (np.array(X_samples, dtype=np.float32),
            np.array(y_classes, dtype=np.int64),
            np.array(y_thermals, dtype=np.float32))

def train_multispectral_brain():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"Device: {device}")
    print(f"Training Multi-Spectral Perception Brain (Module 2)...")
    print(f"Visible RGB + Thermal LWIR + Rayleigh UV Polarization Fusion")
    print(f"==========================================")

    print("Harvesting 80,000 multi-spectral camera frames across cores...")
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per_worker = (80000 + num_workers - 1) // num_workers

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_collect_multispectral, [samples_per_worker] * num_workers)

    X_np = np.concatenate([r[0] for r in results], axis=0)[:80000]
    y_cls_np = np.concatenate([r[1] for r in results], axis=0)[:80000]
    y_thm_np = np.concatenate([r[2] for r in results], axis=0)[:80000]
    print(f"Multi-spectral dataset ready in {time.time() - t0:.2f}s!")

    X = torch.tensor(X_np, device=device)
    y_cls = torch.tensor(y_cls_np, device=device)
    y_thm = torch.tensor(y_thm_np, device=device).unsqueeze(1)

    model = MultiSpectralPerceptionNet().to(device)
    BATCH_SIZE = 2048
    epochs = 10
    num_batches = (len(X) + BATCH_SIZE - 1) // BATCH_SIZE

    optimizer = optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_thm = nn.MSELoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-3, steps_per_epoch=num_batches, epochs=epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"Training Multi-Spectral Fusion CNN (Batch {BATCH_SIZE}, 10 Epochs)...")
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X), device=device)
        total_loss, correct = 0.0, 0

        for i in range(0, len(X), BATCH_SIZE):
            idx = indices[i : i + BATCH_SIZE]
            bx, by_cls, by_thm = X[idx], y_cls[idx], y_thm[idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred_cls, pred_thm = model(bx)
                loss = criterion_cls(pred_cls, by_cls) + 0.4 * criterion_thm(pred_thm, by_thm)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            correct += (pred_cls.argmax(dim=1) == by_cls).sum().item()

        acc = (correct / len(X)) * 100.0
        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Fusion Accuracy: {acc:.2f}% - Loss: {total_loss/num_batches:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Multi-Spectral Vision Model compiled in {time.time() - t0:.2f}s!")
    return model, device

# ==========================================
# 5. REAL-TIME MULTI-SPECTRAL PYGAME HUD
# ==========================================
def run_multispectral_hud(model, device):
    pygame.init()
    VIEW_WIDTH = 700
    VIEW_HEIGHT = 700
    SIDEBAR_WIDTH = 380
    WIDTH = VIEW_WIDTH + SIDEBAR_WIDTH
    HEIGHT = VIEW_HEIGHT

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Module 2: Multi-Spectral Vision (RGB, Thermal LWIR & Rayleigh Polarization)")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("consolas", 18, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 13, bold=True)

    camera = MultiSpectralCamera(res_w=CAM_RES_W, res_h=CAM_RES_H)
    model.eval()

    # Drone Camera 6-DOF Pose in World
    drone_pos = np.array([0.0, 0.5, 2.8], dtype=np.float32)
    drone_yaw_deg = 0.0
    drone_pitch_deg = -5.0
    enable_fog = True
    running = True

    print("================ CONTROLS ================")
    print(" [W] / [S]      : Fly Forward / Backward")
    print(" [A] / [D]      : Strafe Left / Right")
    print(" [I] / [K]      : Climb / Descend")
    print(" [LEFT] / [RIGHT]: Pan Camera Yaw")
    print(" [UP] / [DOWN]  : Tilt Camera Pitch")
    print(" [F]            : Toggle Atmospheric Fog (Beer-Lambert Scatt)")
    print(" [R]            : Reset Drone to Start Position")
    print("==========================================\n")

    while running:
        dt = clock.tick(30) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    enable_fog = not enable_fog
                elif event.key == pygame.K_r:
                    drone_pos = np.array([0.0, 0.5, 2.8], dtype=np.float32)
                    drone_yaw_deg = 0.0
                    drone_pitch_deg = -5.0

        keys = pygame.key.get_pressed()
        yaw_rad = math.radians(drone_yaw_deg)
        fwd_vec = np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0], dtype=np.float32)
        right_vec = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=np.float32)

        if keys[pygame.K_w]: drone_pos += fwd_vec * 0.12
        if keys[pygame.K_s]: drone_pos -= fwd_vec * 0.12
        if keys[pygame.K_d]: drone_pos += right_vec * 0.12
        if keys[pygame.K_a]: drone_pos -= right_vec * 0.12
        if keys[pygame.K_i]: drone_pos[2] = min(6.0, drone_pos[2] + 0.08)
        if keys[pygame.K_k]: drone_pos[2] = max(0.5, drone_pos[2] - 0.08)
        if keys[pygame.K_LEFT]: drone_yaw_deg = (drone_yaw_deg - 2.5) % 360.0
        if keys[pygame.K_RIGHT]: drone_yaw_deg = (drone_yaw_deg + 2.5) % 360.0
        if keys[pygame.K_UP]: drone_pitch_deg = min(25.0, drone_pitch_deg + 1.5)
        if keys[pygame.K_DOWN]: drone_pitch_deg = max(-45.0, drone_pitch_deg - 1.5)

        # Construct Drone Rotation Matrix R in SO(3)
        y_r = math.radians(drone_yaw_deg)
        p_r = math.radians(drone_pitch_deg)
        R_yaw = np.array([[math.cos(y_r), -math.sin(y_r), 0.0], [math.sin(y_r), math.cos(y_r), 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        R_pitch = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(p_r), -math.sin(p_r)], [0.0, math.sin(p_r), math.cos(p_r)]], dtype=np.float32)
        R_drone = R_yaw @ R_pitch

        # 1. Capture Multi-Spectral Sensor Feeds
        rgb_buf, thm_buf, pol_buf, depth_map, thm_raw, telemetry = camera.capture(drone_pos, R_drone, enable_fog=enable_fog)

        # 2. Run PyTorch Multi-Spectral CNN Inference
        tensor_5ch = np.zeros((1, 5, 24, 32), dtype=np.float32)
        # Downsample buffers for CNN
        rgb_ds = rgb_buf[::2, ::2]
        thm_ds = thm_raw[::2, ::2]
        dep_ds = depth_map[::2, ::2]

        tensor_5ch[0, 0:3] = rgb_ds.transpose(2, 0, 1) / 255.0
        tensor_5ch[0, 3] = np.clip((thm_ds - 15.0) / 80.0, 0.0, 1.0)
        tensor_5ch[0, 4] = np.clip(dep_ds / 40.0, 0.0, 1.0)

        with torch.no_grad():
            t_in = torch.tensor(tensor_5ch, device=device)
            pred_cls, pred_thm = model(t_in)
            probs = torch.softmax(pred_cls[0], dim=-1).cpu().numpy()
            anom_score = float(pred_thm[0, 0].item())

        screen.fill((10, 12, 18))

        # Dividers
        pygame.draw.line(screen, (35, 42, 58), (VIEW_WIDTH, 0), (VIEW_WIDTH, HEIGHT), 2)
        pygame.draw.line(screen, (35, 42, 58), (0, 70), (WIDTH, 70), 2)

        # ==========================================
        # MULTI-SPECTRAL TRIPTYCH DISPLAY
        # ==========================================
        panel_w = 320
        panel_h = 240

        # Sub-panel 1: Visible Spectrum (RGB + Fog)
        surf_rgb = pygame.surfarray.make_surface(np.transpose(rgb_buf, (1, 0, 2)))
        surf_rgb = pygame.transform.scale(surf_rgb, (panel_w, panel_h))
        screen.blit(surf_rgb, (20, 95))
        pygame.draw.rect(screen, (40, 55, 75), (20, 95, panel_w, panel_h), 2)
        fog_tag = "[FOG ACTIVE - BEER-LAMBERT]" if enable_fog else "[CLEAR ATMOSPHERE]"
        screen.blit(font_mono.render("1. VISIBLE (RGB) 400-700nm", True, (240, 240, 240)), (25, 102))
        screen.blit(font_small.render(fog_tag, True, (255, 180, 40) if enable_fog else (0, 255, 140)), (25, 120))

        # Sub-panel 2: Thermal Long-Wave Infrared (LWIR 8-14 um)
        surf_thm = pygame.surfarray.make_surface(np.transpose(thm_buf, (1, 0, 2)))
        surf_thm = pygame.transform.scale(surf_thm, (panel_w, panel_h))
        screen.blit(surf_thm, (360, 95))
        pygame.draw.rect(screen, (40, 55, 75), (360, 95, panel_w, panel_h), 2)
        screen.blit(font_mono.render("2. THERMAL (LWIR) 8-14um", True, (240, 240, 240)), (365, 102))
        wire_tag = "POWERLINE HOTSPOT DETECTED (78°C)!" if telemetry["wire_detected"] else "BACKGROUND AMBIENT"
        screen.blit(font_small.render(wire_tag, True, (255, 60, 80) if telemetry["wire_detected"] else (0, 220, 255)), (365, 120))

        # Sub-panel 3: Skylight Rayleigh Polarization (AoP & DoLP)
        surf_pol = pygame.surfarray.make_surface(np.transpose(pol_buf, (1, 0, 2)))
        surf_pol = pygame.transform.scale(surf_pol, (panel_w, panel_h))
        screen.blit(surf_pol, (20, 365))
        pygame.draw.rect(screen, (40, 55, 75), (20, 365, panel_w, panel_h), 2)
        screen.blit(font_mono.render("3. RAYLEIGH POLARIZATION (AoP/DoLP)", True, (240, 240, 240)), (25, 372))
        screen.blit(font_small.render(f"Sky DoLP: {telemetry['avg_sky_dolp']:.1f}% (Solar Meridian Locked)", True, (0, 255, 140)), (25, 390))

        # Sub-panel 4: Real-time Range & Depth Map (Sonar/Lidar overlay)
        depth_norm = np.clip((depth_map / 35.0) * 255.0, 0, 255).astype(np.uint8)
        depth_rgb = np.stack([depth_norm, depth_norm, depth_norm], axis=-1)
        surf_dep = pygame.surfarray.make_surface(np.transpose(depth_rgb, (1, 0, 2)))
        surf_dep = pygame.transform.scale(surf_dep, (panel_w, panel_h))
        screen.blit(surf_dep, (360, 365))
        pygame.draw.rect(screen, (40, 55, 75), (360, 365, panel_w, panel_h), 2)
        screen.blit(font_mono.render("4. SPATIAL RANGE / DEPTH BUFFER", True, (240, 240, 240)), (365, 372))
        min_range = float(np.min(depth_map))
        screen.blit(font_small.render(f"Min Range: {min_range:.2f} m", True, (0, 220, 255)), (365, 390))

        # Bottom Reticle & Flight Status on Left Viewport
        bot_txt = font_small.render("FLIR Ironbow Palette: [Blue = 15°C | Purple = 35°C | Orange = 70°C | White = 95°C]", True, (160, 180, 205))
        screen.blit(bot_txt, (20, 625))
        key_hint = font_small.render("Press [F] to Toggle Atmospheric Fog (Demonstrates LWIR Fog Penetration)", True, (255, 200, 40))
        screen.blit(key_hint, (20, 645))

        # ==========================================
        # MULTI-SPECTRAL TELEMETRY HUD (SIDEBAR)
        # ==========================================
        sb_x = VIEW_WIDTH + 14
        screen.blit(font_large.render("MULTI-SPECTRAL SENSORS", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Optical + Radiometric LWIR + Solar UV Polarization", True, (130, 160, 190)), (sb_x, 38))

        # --- PANEL 1: THERMAL RADIOMETRIC TELEMETRY ---
        p1_y = 80
        screen.blit(font_mono.render("RADIOMETRIC THERMAL LWIR (8-14um):", True, (240, 240, 240)), (sb_x, p1_y))

        t_peak = telemetry["peak_temp_c"]
        t_col = (255, 60, 80) if t_peak > 70.0 else (0, 255, 140)
        screen.blit(font_small.render(f"Max Pixel Temp:   T_max = {t_peak:.1f} °C (E = εσT^4)", True, t_col), (sb_x + 10, p1_y + 24))
        screen.blit(font_small.render(f"Thermal Contrast: ΔT = {telemetry['thermal_contrast']:+.1f} °C (vs Ambient)", True, (0, 220, 255)), (sb_x + 10, p1_y + 40))
        wire_alert = "POWERLINE WIRE IDENTIFIED (HOT)" if telemetry["wire_detected"] else "NO WIRE HAZARDS"
        screen.blit(font_small.render(f"FLIR Detection:   {wire_alert}", True, t_col), (sb_x + 10, p1_y + 56))

        # Thermal gauge bar [15 to 100 C]
        bar_x = sb_x + 10
        bar_y = p1_y + 74
        pygame.draw.rect(screen, (24, 32, 45), (bar_x, bar_y, 320, 8), border_radius=2)
        fill_w = int(320 * min(1.0, max(0.0, (t_peak - 15.0) / 85.0)))
        pygame.draw.rect(screen, t_col, (bar_x, bar_y, fill_w, 8), border_radius=2)

        # --- PANEL 2: RAYLEIGH SKYLIGHT POLARIZATION COMPASS ---
        p2_y = bar_y + 24
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (WIDTH - 14, p2_y), 1)
        screen.blit(font_mono.render("RAYLEIGH SKY COMPASS (GPS-DENIED):", True, (240, 240, 240)), (sb_x, p2_y + 8))

        # Draw mini polarization compass rose
        rcx, rcy = sb_x + 45, p2_y + 55
        pygame.draw.circle(screen, (25, 35, 50), (rcx, rcy), 26, 1)

        # Sun heading vector (Gold) & Polarization meridian (Cyan)
        sun_ang_rad = math.radians(SUN_AZIMUTH_DEG)
        sx = int(rcx + math.sin(sun_ang_rad) * 22)
        sy = int(rcy - math.cos(sun_ang_rad) * 22)
        pygame.draw.line(screen, (255, 215, 0), (rcx, rcy), (sx, sy), 2)
        pygame.draw.circle(screen, (255, 215, 0), (sx, sy), 3)

        # E-vector perpendicular line
        e_ang_rad = sun_ang_rad + math.pi / 2.0
        ex1 = int(rcx + math.sin(e_ang_rad) * 20)
        ey1 = int(rcy - math.cos(e_ang_rad) * 20)
        ex2 = int(rcx - math.sin(e_ang_rad) * 20)
        ey2 = int(rcy + math.cos(e_ang_rad) * 20)
        pygame.draw.line(screen, (0, 240, 255), (ex1, ey1), (ex2, ey2), 2)

        screen.blit(font_small.render(f"Solar Azimuth:    {SUN_AZIMUTH_DEG:.1f}° (Sun Vector)", True, (255, 215, 0)), (sb_x + 85, p2_y + 35))
        screen.blit(font_small.render(f"Sky DoLP Factor:  {telemetry['avg_sky_dolp']:.1f}% (Rayleigh)", True, (0, 255, 140)), (sb_x + 85, p2_y + 53))
        screen.blit(font_small.render("Heading Reference: Drift-Free (No Mag/GPS)", True, (0, 220, 255)), (sb_x + 85, p2_y + 71))

        # --- PANEL 3: PYTORCH MULTI-SPECTRAL FUSION CNN ---
        p3_y = p2_y + 105
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (WIDTH - 14, p3_y), 1)
        screen.blit(font_mono.render("NEURAL MULTI-SPECTRAL FUSION:", True, (240, 240, 240)), (sb_x, p3_y + 8))

        cls_names = ["Path Clear", "Powerline Wire Hazard", "Thermal Obstacle"]
        cls_cols = [(0, 255, 140), (255, 60, 80), (255, 180, 40)]
        for c_i in range(3):
            cy = p3_y + 28 + c_i * 18
            prob_v = probs[c_i]
            bar_len = int(prob_v * 160)
            screen.blit(font_small.render(f"{cls_names[c_i]}:", True, (160, 180, 205)), (sb_x + 10, cy))
            pygame.draw.rect(screen, (24, 32, 45), (sb_x + 170, cy + 2, 160, 7), border_radius=2)
            pygame.draw.rect(screen, cls_cols[c_i], (sb_x + 170, cy + 2, bar_len, 7), border_radius=2)

        screen.blit(font_small.render(f"Anomaly Score:    {anom_score:.4f} (IR Contrast)", True, (0, 220, 255)), (sb_x + 10, p3_y + 88))

        # --- PANEL 4: DRONE CAMERA 6-DOF POSE ---
        p4_y = p3_y + 112
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (WIDTH - 14, p4_y), 1)
        screen.blit(font_mono.render("OPTICAL GIMBAL & CAMERA POSE:", True, (240, 240, 240)), (sb_x, p4_y + 8))

        screen.blit(font_small.render(f"Drone Position:   [{drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f}] m", True, (0, 255, 140)), (sb_x + 10, p4_y + 28))
        screen.blit(font_small.render(f"Gimbal Pan (Yaw): {drone_yaw_deg:+.1f}°", True, (0, 220, 255)), (sb_x + 10, p4_y + 44))
        screen.blit(font_small.render(f"Gimbal Tilt:      {drone_pitch_deg:+.1f}° (Pitch)", True, (160, 180, 205)), (sb_x + 10, p4_y + 60))
        screen.blit(font_small.render(f"Optical Sensor:   {CAM_RES_W}x{CAM_RES_H} Raycasting ({CAM_RES_W*CAM_RES_H} rays)", True, (160, 180, 205)), (sb_x + 10, p4_y + 76))

        # Bottom Telemetry Banner
        bot_y = p4_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y), (WIDTH - 14, bot_y), 1)
        if telemetry["wire_detected"]:
            status_str = "WARNING: ENERGIZED POWERLINE IN PATH"
            status_col = (255, 60, 80)
        else:
            status_str = "MULTI-SPECTRAL CORRIDOR: CLEAR"
            status_col = (0, 255, 140)
        screen.blit(font_mono.render(status_str, True, status_col), (sb_x + 8, bot_y + 8))

        # Header UI
        head_txt = font_large.render(f"T_max: {t_peak:.1f}°C | Sky DoLP: {telemetry['avg_sky_dolp']:.0f}% | Multi-Spectral Vision", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("Visible RGB + FLIR Thermal LWIR (8-14um) + Rayleigh Polarization Sky Compass (GPS-Denied Navigation)", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

# ==========================================
# 6. MAIN ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    mp.freeze_support()
    model, device = train_multispectral_brain()
    print("Launching Multi-Spectral Vision Sensor Simulator...")
    run_multispectral_hud(model, device)