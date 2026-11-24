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
# 0. CONFIGURATION & ATMOSPHERIC CONSTANTS
# ==========================================
# Standard Atmosphere at Sea Level (ISA Standard)
P_SEA_LEVEL_HPA = 1013.25   # Standard sea level pressure (hPa / mbar)
T_SEA_LEVEL_K = 288.15      # 15.0 deg C in Kelvin
RHO_SEA_LEVEL = 1.225       # kg/m^3 standard sea-level air density
LAPSE_RATE_K_PER_M = 0.0065 # Tropospheric temperature lapse rate (K/m)
GAS_CONST_AIR = 287.058     # Specific gas constant for dry air (J / kg*K)
G_GRAVITY = 9.80665

# Hypsometric exponent: (R * L) / (g_0 * M) ≈ 0.190284
HYPSO_EXPONENT = 0.190284
HYPSO_CONST = 44330.8       # meters

# Turbulence & Gust Limits
GUST_MAX_SPEED_MS = 9.5     # 34.2 km/h sudden microburst gust
TURBULENCE_INTENSITY = 0.85 # Dryden RMS turbulence intensity (m/s)

MODEL_SAVE_PATH = "drone_weather_estimator_model.pth"

# ==========================================
# 1. ATMOSPHERIC & SENSOR SUITE
# ==========================================
class AtmosphericSensorSuite:
    """
    Simulates:
    1. Precision barometric altimeter via the ISA hypsometric formula.
    2. Differential pitot-static tube dynamic pressure sensor.
    3. Ambient air density and density altitude variations.
    4. Dryden stochastic wind turbulence and discrete 1-cosine gusts.
    """
    def __init__(self):
        # Base prevailing wind vector: [North, East, Down] (m/s)
        self.base_wind = np.array([3.2, 1.8, 0.0], dtype=np.float32)
        self.turbulent_wind = np.zeros(3, dtype=np.float32)
        self.gust_active = False
        self.gust_timer = 0.0
        self.gust_duration = 2.5
        self.gust_amplitude = np.zeros(3, dtype=np.float32)

    def update_atmosphere(self, drone_pos, drone_vel, dt=0.03):
        """
        Updates atmospheric conditions at drone position:
        Returns:
        - p_static_hpa: Ambient static pressure (hPa)
        - z_baro_m: Barometric altitude (m)
        - q_dyn_pa: Pitot dynamic pressure (Pa)
        - v_tas_ms: True Airspeed (m/s)
        - v_ias_knots: Indicated Airspeed (knots)
        - rho_air: Air density (kg/m^3)
        - wind_current: Total 3D wind velocity vector (m/s)
        """
        z_true = max(0.0, float(drone_pos[2]))

        # 1. Temperature at Altitude: T(z) = T_0 - L * z
        t_current_k = T_SEA_LEVEL_K - LAPSE_RATE_K_PER_M * z_true
        t_current_c = t_current_k - 273.15

        # 2. Barometric Static Pressure via Hypsometric Formula
        # P(z) = P_0 * (1 - (L*z)/T_0)^(1 / 0.190284)
        p_static_hpa = P_SEA_LEVEL_HPA * ((1.0 - (LAPSE_RATE_K_PER_M * z_true) / T_SEA_LEVEL_K) ** (1.0 / HYPSO_EXPONENT))
        # Add realistic sensor noise (0.04 hPa RMS noise on high-precision MEMS barometer)
        p_measured_hpa = p_static_hpa + float(np.random.normal(0.0, 0.04))

        # Reconstructed barometric altitude from sensor measurement
        z_baro_m = HYPSO_CONST * (1.0 - ((p_measured_hpa / P_SEA_LEVEL_HPA) ** HYPSO_EXPONENT))

        # 3. Ambient Air Density rho = P / (R_spec * T)
        rho_air = (p_measured_hpa * 100.0) / (GAS_CONST_AIR * t_current_k)
        density_ratio = rho_air / RHO_SEA_LEVEL

        # 4. Dryden Atmospheric Wind & Gust Simulation
        # Continuous turbulence (1st-order Gauss-Markov shaping filter)
        white_noise = np.random.normal(0.0, 1.0, 3).astype(np.float32)
        d_turb = (-self.turbulent_wind * 0.4 + TURBULENCE_INTENSITY * white_noise) * dt
        self.turbulent_wind += d_turb

        # Discrete 1-Cosine Wind Gust
        if self.gust_active:
            self.gust_timer += dt
            if self.gust_timer < self.gust_duration:
                # 1 - cos(pi * t / T) profile
                gust_factor = 0.5 * (1.0 - math.cos(math.pi * (self.gust_timer / self.gust_duration)))
                active_gust = self.gust_amplitude * gust_factor
            else:
                self.gust_active = False
                active_gust = np.zeros(3, dtype=np.float32)
        else:
            active_gust = np.zeros(3, dtype=np.float32)

        total_wind = self.base_wind + self.turbulent_wind + active_gust

        # 5. Pitot-Static Aerodynamics: True Airspeed (TAS) vs Groundspeed
        # Relative airflow vector: v_air = v_ground - v_wind
        v_air_vector = drone_vel - total_wind
        v_tas_ms = float(np.linalg.norm(v_air_vector))

        # Dynamic Pressure q = 1/2 * rho * V_TAS^2 (Pascals)
        q_dyn_pa = 0.5 * rho_air * (v_tas_ms ** 2)
        # Add pitot tube pressure noise
        q_dyn_measured = max(0.0, q_dyn_pa + float(np.random.normal(0.0, 0.5)))

        # Indicated Airspeed (IAS) calibrated for standard sea-level density
        v_ias_ms = math.sqrt((2.0 * q_dyn_measured) / RHO_SEA_LEVEL)
        v_ias_knots = v_ias_ms * 1.94384

        # Vertical Speed Indicator (VSI): d(z)/dt in m/s
        vsi_ms = float(drone_vel[2])

        telemetry = {
            "p_static_hpa": p_measured_hpa,
            "z_baro_m": z_baro_m,
            "q_dyn_pa": q_dyn_measured,
            "v_tas_ms": v_tas_ms,
            "v_ias_knots": v_ias_knots,
            "vsi_ms": vsi_ms,
            "rho_air": rho_air,
            "density_ratio": density_ratio,
            "total_wind": total_wind,
            "temp_c": t_current_c,
            "is_gusting": self.gust_active
        }
        return telemetry

    def trigger_gust(self):
        """Triggers a severe atmospheric wind gust (35 km/h)."""
        self.gust_active = True
        self.gust_timer = 0.0
        # Gust along North-West direction
        self.gust_amplitude = np.array([random.uniform(4.5, 7.5), random.uniform(-5.5, -3.5), random.uniform(-1.5, 1.5)], dtype=np.float32)

# ==========================================
# 2. PYTORCH NEURAL ATMOSPHERIC ESTIMATOR
# ==========================================
class AtmosphericWeatherNet(nn.Module):
    """
    Neural network predicting 3D wind velocity vector and air density
    from differential pitot pressure, barometric pressure, and IMU motion.
    """
    def __init__(self):
        super().__init__()
        # Inputs: p_static, q_dyn, temp, ground_vel(3), attitude_z(3) = 9 inputs
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # Predicts 3D wind vector (w_x, w_y, w_z)
        )

        self.density_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid() # Predicts normalized air density rho / rho_0
        )

    def forward(self, x):
        h = self.net[:-1](x)
        wind_vec = self.net[-1](h)
        density_norm = self.density_head(h)
        return wind_vec, density_norm

def _worker_collect_weather(num_samples):
    suite = AtmosphericSensorSuite()
    X_samples, y_winds, y_rhos = [], [], []

    while len(X_samples) < num_samples:
        pos = np.array([random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0), random.uniform(0.5, 10.0)], dtype=np.float32)
        vel = np.array([random.uniform(-8.0, 8.0), random.uniform(-8.0, 8.0), random.uniform(-2.5, 2.5)], dtype=np.float32)
        yaw = math.radians(random.uniform(-180.0, 180.0))

        if random.random() < 0.25:
            suite.trigger_gust()

        telem = suite.update_atmosphere(pos, vel, dt=0.03)

        # Feature vector (9 dims)
        p_norm = (telem["p_static_hpa"] - 900.0) / 150.0
        q_norm = telem["q_dyn_pa"] / 150.0
        t_norm = (telem["temp_c"] - 10.0) / 25.0
        vel_norm = vel / 10.0
        att_z = np.array([math.sin(yaw), math.cos(yaw), 1.0], dtype=np.float32)

        feat = np.concatenate([[p_norm, q_norm, t_norm], vel_norm, att_z]).astype(np.float32)

        X_samples.append(feat)
        y_winds.append(telem["total_wind"])
        y_rhos.append(telem["density_ratio"])

    return (np.array(X_samples, dtype=np.float32),
            np.array(y_winds, dtype=np.float32),
            np.array(y_rhos, dtype=np.float32))

def train_weather_brain():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"Device: {device}")
    print(f"Training Atmospheric Sensing & Weather Estimator Brain (Module 5)...")
    print(f"Hypsometric Altimetry, Pitot-Static Aerodynamics & Dryden Gusts")
    print(f"==========================================")

    print("Harvesting 80,000 micro-meteorological atmospheric states across cores...")
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per_worker = (80000 + num_workers - 1) // num_workers

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_collect_weather, [samples_per_worker] * num_workers)

    X_np = np.concatenate([r[0] for r in results], axis=0)[:80000]
    y_wind_np = np.concatenate([r[1] for r in results], axis=0)[:80000]
    y_rho_np = np.concatenate([r[2] for r in results], axis=0)[:80000]
    print(f"Atmospheric training dataset ready in {time.time() - t0:.2f}s!")

    X = torch.tensor(X_np, device=device)
    y_wind = torch.tensor(y_wind_np, device=device)
    y_rho = torch.tensor(y_rho_np, device=device).unsqueeze(1)

    model = AtmosphericWeatherNet().to(device)
    BATCH_SIZE = 2048
    epochs = 10
    num_batches = (len(X) + BATCH_SIZE - 1) // BATCH_SIZE

    optimizer = optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    criterion_wind = nn.MSELoss()
    criterion_rho = nn.MSELoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-3, steps_per_epoch=num_batches, epochs=epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"Optimizing Neural Weather Observers (Batch {BATCH_SIZE}, 10 Epochs)...")
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X), device=device)
        total_loss = 0.0

        for i in range(0, len(X), BATCH_SIZE):
            idx = indices[i : i + BATCH_SIZE]
            bx, by_w, by_r = X[idx], y_wind[idx], y_rho[idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred_w, pred_r = model(bx)
                loss = criterion_wind(pred_w, by_w) + 0.3 * criterion_rho(pred_r, by_r)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Wind/Density MSE Loss: {total_loss/num_batches:.5f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Weather Estimator Model compiled in {time.time() - t0:.2f}s!")
    return model, device

# ==========================================
# 3. REAL-TIME 3D FLIGHT & PFD HUD
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

def run_weather_sensor_hud(model, device):
    pygame.init()
    VIEW_WIDTH = 700
    VIEW_HEIGHT = 700
    SIDEBAR_WIDTH = 380
    WIDTH = VIEW_WIDTH + SIDEBAR_WIDTH
    HEIGHT = VIEW_HEIGHT

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Module 5: Micro-Meteorological & Atmospheric Sensing Suite (Baro / Pitot / Dryden)")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("consolas", 18, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 13, bold=True)

    suite = AtmosphericSensorSuite()
    model.eval()

    # Drone 6-DOF Coordinates in World Space (meters)
    drone_pos = np.array([0.0, 1.0, 3.5], dtype=np.float32)
    drone_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    drone_yaw_deg = 0.0

    camera_yaw = 0.65
    camera_pitch = 0.40
    center_viewport = (VIEW_WIDTH // 2, VIEW_HEIGHT // 2 + 50)

    # 3D Atmospheric Wind Streamline Particles
    num_particles = 45
    stream_particles = [np.array([random.uniform(-5.0, 5.0), random.uniform(-2.0, 12.0), random.uniform(0.5, 6.0)]) for _ in range(num_particles)]

    running = True

    print("================ CONTROLS ================")
    print(" [W] / [S]      : Forward / Backward Flight (TAS)")
    print(" [A] / [D]      : Strafe Left / Right")
    print(" [I] / [K]      : Climb / Descend (Baro Altitude)")
    print(" [G]            : Trigger Severe 35 km/h Wind Gust")
    print(" [J] / [L]      : Yaw Heading")
    print(" [ARROW KEYS]   : Orbit 3D Camera View")
    print(" [R]            : Reset to Nominal Altitude")
    print("==========================================\n")

    while running:
        dt = clock.tick(30) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    suite.trigger_gust()
                elif event.key == pygame.K_r:
                    drone_pos = np.array([0.0, 1.0, 3.5], dtype=np.float32)
                    drone_vel = np.zeros(3, dtype=np.float32)

        keys = pygame.key.get_pressed()
        yaw_rad = math.radians(drone_yaw_deg)
        fwd_vec = np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0], dtype=np.float32)
        right_vec = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=np.float32)

        # Velocity commands
        accel = np.zeros(3, dtype=np.float32)
        if keys[pygame.K_w]: accel += fwd_vec * 6.5
        if keys[pygame.K_s]: accel -= fwd_vec * 6.5
        if keys[pygame.K_d]: accel += right_vec * 6.5
        if keys[pygame.K_a]: accel -= right_vec * 6.5
        if keys[pygame.K_i]: accel[2] += 4.5
        if keys[pygame.K_k]: accel[2] -= 4.5
        if keys[pygame.K_j]: drone_yaw_deg = (drone_yaw_deg - 2.5) % 360.0
        if keys[pygame.K_l]: drone_yaw_deg = (drone_yaw_deg + 2.5) % 360.0

        # Drag damping and velocity integration
        drone_vel += (accel - 0.75 * drone_vel) * dt
        drone_pos += drone_vel * dt
        drone_pos[2] = max(0.2, min(12.0, drone_pos[2]))

        # Camera rotation
        if keys[pygame.K_LEFT]: camera_yaw -= 0.03
        if keys[pygame.K_RIGHT]: camera_yaw += 0.03
        if keys[pygame.K_UP]: camera_pitch = max(0.1, camera_pitch - 0.02)
        if keys[pygame.K_DOWN]: camera_pitch = min(1.2, camera_pitch + 0.02)

        # 1. Poll Atmospheric Sensor Suite
        telemetry = suite.update_atmosphere(drone_pos, drone_vel, dt=dt)
        wind = telemetry["total_wind"]

        # Update wind streamline particles
        for p_i in range(num_particles):
            stream_particles[p_i] += wind * dt
            # Wrap around boundaries
            if stream_particles[p_i][1] > 12.0: stream_particles[p_i][1] = -2.0
            if stream_particles[p_i][0] > 5.0:  stream_particles[p_i][0] = -5.0
            if stream_particles[p_i][0] < -5.0: stream_particles[p_i][0] = 5.0

        # 2. Run PyTorch Neural Weather Inference
        p_norm = (telemetry["p_static_hpa"] - 900.0) / 150.0
        q_norm = telemetry["q_dyn_pa"] / 150.0
        t_norm = (telemetry["temp_c"] - 10.0) / 25.0
        vel_norm = drone_vel / 10.0
        att_z = np.array([math.sin(yaw_rad), math.cos(yaw_rad), 1.0], dtype=np.float32)
        feat_vec = np.concatenate([[p_norm, q_norm, t_norm], vel_norm, att_z]).astype(np.float32)

        with torch.no_grad():
            feat_t = torch.tensor(feat_vec, device=device).unsqueeze(0)
            pred_w, pred_r = model(feat_t)
            nn_wind = pred_w[0].cpu().numpy()
            nn_rho_ratio = float(pred_r[0, 0].item())

        screen.fill((10, 14, 22))

        # Dividers
        pygame.draw.line(screen, (35, 45, 60), (VIEW_WIDTH, 0), (VIEW_WIDTH, HEIGHT), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (WIDTH, 70), 2)

        # ==========================================
        # 3D VIEWPORT: AIRSPACE & WIND STREAMLINES
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

        # 2. Draw 3D Wind Flow Streamlines (Cyan vectors)
        wind_col = (255, 140, 20) if telemetry["is_gusting"] else (0, 220, 255)
        for sp in stream_particles:
            tail_pt = sp - wind * 0.15
            sx1, sy1, _ = project_3d(sp, cam, center_viewport)
            sx2, sy2, _ = project_3d(tail_pt, cam, center_viewport)
            pygame.draw.line(screen, (30, 45, 65), (sx1, sy1), (sx2, sy2), 1)
            pygame.draw.circle(screen, wind_col, (sx1, sy1), 2)

        # 3. Draw Drone Airframe
        hub_sx, hub_sy, _ = project_3d(drone_pos, cam, center_viewport)
        pygame.draw.circle(screen, (0, 240, 255), (hub_sx, hub_sy), 8)
        pygame.draw.circle(screen, (255, 255, 255), (hub_sx, hub_sy), 3)

        # Pitot tube forward pointing nose boom
        R_drone = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0],
            [math.sin(yaw_rad),  math.cos(yaw_rad), 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        pitot_end = drone_pos + R_drone @ np.array([0.0, 0.8, 0.0])
        psx, psy, _ = project_3d(pitot_end, cam, center_viewport)
        pygame.draw.line(screen, (255, 215, 0), (hub_sx, hub_sy), (psx, psy), 3)
        pygame.draw.circle(screen, (255, 255, 255), (psx, psy), 2)

        # Drop-line to ground
        ground_pt = np.array([drone_pos[0], drone_pos[1], 0.0])
        gsx, gsy, _ = project_3d(ground_pt, cam, center_viewport)
        pygame.draw.line(screen, (35, 55, 75), (hub_sx, hub_sy), (gsx, gsy), 1)
        pygame.draw.circle(screen, (40, 70, 95), (gsx, gsy), 3)

        # ==========================================
        # AVIATION GLASS COCKPIT PFD (LOWER VIEWPORT)
        # ==========================================
        pfd_w, pfd_h = 320, 160
        pfd_x = 25
        pfd_y = VIEW_HEIGHT - pfd_h - 25

        # PFD Housing
        pygame.draw.rect(screen, (14, 20, 28), (pfd_x, pfd_y, pfd_w, pfd_h), border_radius=6)
        pygame.draw.rect(screen, (35, 50, 70), (pfd_x, pfd_y, pfd_w, pfd_h), 2)

        # 1. Airspeed Tape (Left of PFD)
        spd_tape_x = pfd_x + 12
        spd_tape_y = pfd_y + 15
        pygame.draw.rect(screen, (20, 28, 40), (spd_tape_x, spd_tape_y, 55, pfd_h - 30), border_radius=3)
        screen.blit(font_small.render("IAS (KT)", True, (130, 160, 190)), (spd_tape_x + 6, spd_tape_y + 4))

        ias_val = telemetry["v_ias_knots"]
        screen.blit(font_mono.render(f"{ias_val:4.1f}", True, (0, 255, 140)), (spd_tape_x + 6, spd_tape_y + 55))
        screen.blit(font_small.render("KNOTS", True, (160, 180, 205)), (spd_tape_x + 10, spd_tape_y + 75))

        # 2. Artificial Horizon / Pitch Center
        ah_cx = pfd_x + pfd_w // 2
        ah_cy = pfd_y + pfd_h // 2
        pygame.draw.circle(screen, (24, 34, 48), (ah_cx, ah_cy), 45)
        pygame.draw.circle(screen, (40, 60, 85), (ah_cx, ah_cy), 45, 1)
        # Horizon pitch line
        pygame.draw.line(screen, (0, 220, 255), (ah_cx - 30, ah_cy), (ah_cx + 30, ah_cy), 2)
        pygame.draw.circle(screen, (255, 200, 0), (ah_cx, ah_cy), 3)

        # 3. Barometric Altitude Tape (Right of PFD)
        alt_tape_x = pfd_x + pfd_w - 70
        alt_tape_y = pfd_y + 15
        pygame.draw.rect(screen, (20, 28, 40), (alt_tape_x, alt_tape_y, 58, pfd_h - 30), border_radius=3)
        screen.blit(font_small.render("ALT (M)", True, (130, 160, 190)), (alt_tape_x + 8, alt_tape_y + 4))

        baro_alt = telemetry["z_baro_m"]
        screen.blit(font_mono.render(f"{baro_alt:5.1f}", True, (0, 220, 255)), (alt_tape_x + 6, alt_tape_y + 55))
        vsi = telemetry["vsi_ms"]
        vsi_col = (0, 255, 140) if vsi >= 0 else (255, 180, 40)
        screen.blit(font_small.render(f"{vsi:+.1f}m/s", True, vsi_col), (alt_tape_x + 8, alt_tape_y + 75))

        screen.blit(font_small.render("PRIMARY FLIGHT DISPLAY (PFD)", True, (130, 160, 190)), (pfd_x + 65, pfd_y + pfd_h - 18))

        # 4. Wind Vector Radar Compass (Beside PFD)
        w_dial_cx = 445
        w_dial_cy = VIEW_HEIGHT - 105
        w_radius = 55
        pygame.draw.circle(screen, (16, 24, 34), (w_dial_cx, w_dial_cy), w_radius)
        pygame.draw.circle(screen, (35, 50, 70), (w_dial_cx, w_dial_cy), w_radius, 1)

        # Draw Wind direction arrow
        wind_mag = float(np.linalg.norm(wind))
        if wind_mag > 0.1:
            w_dir_unit = wind[:2] / wind_mag
            w_end_x = int(w_dial_cx + w_dir_unit[0] * (w_radius - 12))
            w_end_y = int(w_dial_cy - w_dir_unit[1] * (w_radius - 12))
            pygame.draw.line(screen, wind_col, (w_dial_cx, w_dial_cy), (w_end_x, w_end_y), 3)
            pygame.draw.circle(screen, wind_col, (w_end_x, w_end_y), 4)

        screen.blit(font_small.render("WIND VECTOR", True, (130, 160, 190)), (w_dial_cx - 32, w_dial_cy + w_radius + 4))
        screen.blit(font_small.render(f"{wind_mag*3.6:4.1f} km/h", True, wind_col), (w_dial_cx - 28, w_dial_cy + w_radius + 18))

        # ==========================================
        # METEOROLOGICAL TELEMETRY (SIDEBAR)
        # ==========================================
        sb_x = VIEW_WIDTH + 14
        screen.blit(font_large.render("ATMOSPHERIC SENSORS", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Baro Altimetry + Pitot TAS + Dryden Turbulence", True, (130, 160, 190)), (sb_x, 38))

        # --- PANEL 1: BAROMETRIC ALTIMETRY & PRESSURE ---
        p1_y = 80
        screen.blit(font_mono.render("HYPSOMETRIC BARO ALTIMETRY:", True, (240, 240, 240)), (sb_x, p1_y))

        p_hpa = telemetry["p_static_hpa"]
        screen.blit(font_small.render(f"Static Pressure:    P = {p_hpa:.2f} hPa (mbar)", True, (0, 220, 255)), (sb_x + 10, p1_y + 24))
        screen.blit(font_small.render(f"Baro Altitude:      z = {baro_alt:.2f} m (AGL / MSL)", True, (0, 255, 140)), (sb_x + 10, p1_y + 40))
        screen.blit(font_small.render(f"Ambient Temp:       T = {telemetry['temp_c']:.1f} °C (Lapse: 6.5 K/km)", True, (160, 180, 205)), (sb_x + 10, p1_y + 56))

        # --- PANEL 2: PITOT-STATIC AIRSPEED (TAS & DYNAMIC PRESSURE) ---
        p2_y = p1_y + 80
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (WIDTH - 14, p2_y), 1)
        screen.blit(font_mono.render("PITOT-STATIC AIRSPEED (TAS):", True, (240, 240, 240)), (sb_x, p2_y + 8))

        q_dyn = telemetry["q_dyn_pa"]
        tas_ms = telemetry["v_tas_ms"]
        screen.blit(font_small.render(f"Dynamic Pressure:   q = {q_dyn:.1f} Pa (1/2 ρ V^2)", True, (0, 255, 140)), (sb_x + 10, p2_y + 28))
        screen.blit(font_small.render(f"True Airspeed TAS:  V = {tas_ms:.2f} m/s ({tas_ms*3.6:.1f} km/h)", True, (0, 220, 255)), (sb_x + 10, p2_y + 44))
        screen.blit(font_small.render(f"Indicated Airspeed: IAS = {ias_val:.1f} knots (Airfoil lift)", True, (255, 215, 0)), (sb_x + 10, p2_y + 60))

        # Airspeed bar gauge [0, 25 m/s]
        bar_x = sb_x + 10
        bar_y = p2_y + 78
        pygame.draw.rect(screen, (24, 32, 45), (bar_x, bar_y, 320, 8), border_radius=2)
        fill_w = int(320 * min(1.0, max(0.0, tas_ms / 25.0)))
        pygame.draw.rect(screen, (0, 220, 255), (bar_x, bar_y, fill_w, 8), border_radius=2)

        # --- PANEL 3: AIR DENSITY & ROTOR DEGRADATION ---
        p3_y = bar_y + 24
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (WIDTH - 14, p3_y), 1)
        screen.blit(font_mono.render("AIR DENSITY & ROTOR EFFICIENCY:", True, (240, 240, 240)), (sb_x, p3_y + 8))

        rho = telemetry["rho_air"]
        rho_ratio = telemetry["density_ratio"]
        screen.blit(font_small.render(f"Air Density:        ρ = {rho:.4f} kg/m^3", True, (0, 220, 255)), (sb_x + 10, p3_y + 28))
        screen.blit(font_small.render(f"Density Ratio:      ρ/ρ_0 = {rho_ratio*100:.1f}% of ISA Standard", True, (0, 255, 140)), (sb_x + 10, p3_y + 44))
        screen.blit(font_small.render(f"Rotor Thrust Load:  Scale = {1.0 / math.sqrt(rho_ratio):.3f}x Required RPM", True, (160, 180, 205)), (sb_x + 10, p3_y + 60))

        # --- PANEL 4: DRYDEN TURBULENCE & GUST DETECTION ---
        p4_y = p3_y + 80
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (WIDTH - 14, p4_y), 1)
        screen.blit(font_mono.render("DRYDEN TURBULENCE & GUSTS:", True, (240, 240, 240)), (sb_x, p4_y + 8))

        gust_col = (255, 60, 80) if telemetry["is_gusting"] else (0, 255, 140)
        gust_str = "SEVERE GUST PULSE ACTIVE!" if telemetry["is_gusting"] else "STOCHASTIC DRYDEN NORMAL"

        screen.blit(font_small.render(f"3D Wind Velocity:   [{wind[0]:+.1f}, {wind[1]:+.1f}, {wind[2]:+.1f}] m/s", True, gust_col), (sb_x + 10, p4_y + 28))
        screen.blit(font_small.render(f"NN Wind Estimate:   [{nn_wind[0]:+.1f}, {nn_wind[1]:+.1f}, {nn_wind[2]:+.1f}] m/s", True, (0, 220, 255)), (sb_x + 10, p4_y + 44))
        screen.blit(font_small.render(f"Atmospheric State:  {gust_str}", True, gust_col), (sb_x + 10, p4_y + 60))

        # Bottom Telemetry Banner
        bot_y = p4_y + 90
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y), (WIDTH - 14, bot_y), 1)
        if telemetry["is_gusting"]:
            alert_str = "CAUTION: MICROBURST SHEAR ACTIVE"
            alert_col = (255, 60, 80)
        else:
            alert_str = "AIR DATA: BARO & PITOT CALIBRATED"
            alert_col = (0, 255, 140)
        screen.blit(font_mono.render(alert_str, True, alert_col), (sb_x + 8, bot_y + 8))

        # Header UI
        head_txt = font_large.render(f"Alt: {baro_alt:4.1f}m | TAS: {tas_ms:4.1f}m/s | Weather Sensors", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("Barometric Hypsometric Altimetry, Pitot-Static Dynamic Pressure q=1/2*rho*V^2, & Dryden Gust Spectra", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

# ==========================================
# 4. MAIN ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    mp.freeze_support()
    model, device = train_weather_brain()
    print("Launching Atmospheric & Weather Sensing Simulator...")
    run_weather_sensor_hud(model, device)