"""
=============================================================================
MODULE 10: GROUND DROPZONE SAFETY & WINCH DELIVERY MECHANICS
File: drone_delivery_winch_dropzone.py
=============================================================================
Autonomous "Last 50 Feet" Package Delivery: Downward Terrain Slope & Hazard
Auditing, Motorized Tether Winch Dynamics, Load-Cell Touchdown Sensing,
and Pyrotechnic Emergency Line-Cutter (Guillotine) Failsafe System.
=============================================================================
"""

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
# 0. PHYSICAL AIRFRAME & WINCH SPECIFICATIONS
# ==========================================
GRAVITY_G = 9.80665
DRONE_HOVER_ALT_M = 15.0             # Safe standoff hover height above ground (m)
PAYLOAD_MASS_KG = 2.20               # Package mass (2.2 kg)
TETHER_MAX_LENGTH_M = 16.5           # Max available cable length on drum
CABLE_LINEAR_DENSITY = 0.008         # kg/m (Dyneema ultra-light braided line)
DRUM_MOMENT_INERTIA = 0.0035         # kg * m^2 (Lightweight aluminum spool)
WINCH_MOTOR_MAX_TORQUE = 1.8         # N*m
WINCH_DESCENT_SPEED_MS = 1.25        # Controlled descent speed (m/s)
WINCH_RETRACT_SPEED_MS = 2.40        # Empty hook retraction speed (m/s)

# Load-Cell Tension Limits (Newtons)
PAYLOAD_STATIC_WEIGHT_N = PAYLOAD_MASS_KG * GRAVITY_G # ~21.57 N
TOUCHDOWN_TENSION_FLOOR_N = 3.5      # Tension drops below this upon solid ground contact
SNAG_ALARM_TENSION_N = 38.0          # Snag/entanglement pull threshold
GUILLOTINE_TRIGGER_TENSION_N = 48.0  # Emergency line severance threshold

# Dropzone Spatial Clearance Boundaries
DROPZONE_RADIUS_M = 2.5              # 5-meter diameter landing circle
MAX_ACCEPTABLE_SLOPE_DEG = 14.0      # Maximum allowable surface incline

MODEL_SAVE_PATH = "drone_winch_dropzone_model.pth"

# ==========================================
# 1. DOWNWARD DROPZONE AUDITOR & HAZARD WORLD
# ==========================================
class DropzoneHazardWorld:
    """
    Simulates the physical ground terrain and transient dynamic obstacles:
    humans, domestic pets, tree branches, slope gradient, and ground elevation.
    """
    def __init__(self):
        self.ground_base_z = 0.0
        self.terrain_slope_deg = 4.5 # Nominal gentle lawn incline
        self.terrain_slope_heading_deg = 35.0

        # Transient dynamic obstacle (Human or Dog entering landing area)
        self.human_active = False
        self.human_pos = np.array([3.8, 2.5, 0.0], dtype=np.float32)
        self.human_vel = np.array([-0.6, -0.4, 0.0], dtype=np.float32)
        self.human_radius = 0.65

        # Static yard obstacles (Patio table, flowerbed, low branch)
        self.static_obstacles = [
            {"pos": np.array([ 1.8, -1.5, 0.4]), "radius": 0.55, "type": "PATIO_CHAIR"},
            {"pos": np.array([-2.2,  1.8, 1.2]), "radius": 0.70, "type": "LOW_CANOPY"}
        ]

    def update_actors(self, dt):
        """Simulates ground actors moving into or out of the delivery basket."""
        if self.human_active:
            self.human_pos += self.human_vel * dt
            # Rebound inside testing apron
            if abs(self.human_pos[0]) > 4.5 or abs(self.human_pos[1]) > 4.5:
                self.human_vel *= -1.0

    def query_terrain_height(self, x, y):
        """Returns physical ground elevation z for any (x, y) coordinate."""
        rad_slope = math.radians(self.terrain_slope_deg)
        rad_head = math.radians(self.terrain_slope_heading_deg)
        # Gradient plane
        dz_dx = math.sin(rad_slope) * math.cos(rad_head)
        dz_dy = math.sin(rad_slope) * math.sin(rad_head)
        return self.ground_base_z + dz_dx * x + dz_dy * y

    def audit_dropzone_footprint(self, center_xy, radius=DROPZONE_RADIUS_M):
        """
        Audits downward stereo optical raster and acoustic clearance:
        Returns:
        - min_clearance_dist: Distance to nearest solid obstruction inside basket.
        - max_slope_found_deg: Maximum slope detected within footprint.
        - human_incursion: True if human or animal detected inside basket.
        - static_hazard_count: Count of obstacles within dropzone.
        """
        cx, cy = center_xy
        min_clearance = 50.0
        human_incursion = False
        static_count = 0

        # Check human distance
        if self.human_active:
            d_h = math.hypot(self.human_pos[0] - cx, self.human_pos[1] - cy) - self.human_radius
            min_clearance = min(min_clearance, d_h)
            if d_h < radius:
                human_incursion = True

        # Check static obstacles
        for obs in self.static_obstacles:
            d_o = math.hypot(obs["pos"][0] - cx, obs["pos"][1] - cy) - obs["radius"]
            min_clearance = min(min_clearance, d_o)
            if d_o < radius:
                static_count += 1

        # Check slope
        slope_val = self.terrain_slope_deg

        is_safe = (not human_incursion) and (static_count == 0) and (slope_val <= MAX_ACCEPTABLE_SLOPE_DEG)
        return {
            "is_safe": is_safe,
            "min_clearance": min_clearance,
            "human_incursion": human_incursion,
            "static_count": static_count,
            "slope_deg": slope_val
        }

# ==========================================
# 2. MOTORIZED WINCH & LOAD-CELL DYNAMICS
# ==========================================
class MotorizedWinchDeliverySystem:
    """
    Simulates high-speed motorized winch tether deployment, cable elasticity,
    strain-gauge load-cell tension sensing, touchdown detection, and line guillotine.
    """
    def __init__(self):
        self.state = "STANDBY" # STANDBY, LOWERING, TOUCHDOWN, RELEASE, RETRACTING, GUILLOTINE_CUT, ABORT
        self.cable_deployed_len = 0.05       # 5 cm initial cable slack (m)
        self.cable_velocity = 0.0            # Rate of cable pay-out (m/s)
        self.payload_z = DRONE_HOVER_ALT_M - 0.05
        self.payload_attached = True
        self.line_severed = False

        # Load-cell strain gauge telemetry
        self.tension_n = 0.0
        self.tension_raw_history = deque(maxlen=40)
        self.touchdown_timer = 0.0
        self.snag_timer = 0.0
        self.sim_time = 0.0

    def step(self, drone_pos, ground_z_target, force_snag=False, dt=0.02):
        """
        Integrates cable physical kinematics, tension, and delivery state machine:
        - drone_pos: 3D coordinates of hovering drone airframe [X, Y, Z].
        - ground_z_target: Actual ground elevation directly beneath payload.
        """
        self.sim_time += dt
        drone_z = drone_pos[2]

        if self.line_severed:
            self.state = "GUILLOTINE_CUT"
            self.tension_n = 0.0
            if self.payload_z > ground_z_target:
                # Free-fall under gravity
                self.payload_z = max(ground_z_target, self.payload_z - 9.8 * (dt ** 2) * 5.0)
            return self.get_telemetry()

        # -----------------------------------------------------------------
        # 1. Finite State Machine (FSM) Autonomous Delivery Sequence
        # -----------------------------------------------------------------
        if self.state == "STANDBY":
            self.cable_deployed_len = 0.08
            self.payload_z = drone_z - self.cable_deployed_len
            self.tension_n = PAYLOAD_STATIC_WEIGHT_N if self.payload_attached else 0.8

        elif self.state == "LOWERING":
            # Controlled motorized drum unwind
            target_rate = WINCH_DESCENT_SPEED_MS
            # Accelerate cable payout
            self.cable_velocity += (target_rate - self.cable_velocity) * 8.0 * dt
            self.cable_deployed_len += self.cable_velocity * dt
            self.payload_z = drone_z - self.cable_deployed_len

            # Dynamic cable tension: m*(g - a_cable) + aerodrag + cable weight
            dyn_jitter = float(np.random.normal(0.0, 0.45))
            self.tension_n = PAYLOAD_STATIC_WEIGHT_N + dyn_jitter

            # Check if payload has touched the ground
            if self.payload_z <= (ground_z_target + 0.06):
                self.payload_z = ground_z_target
                self.state = "TOUCHDOWN"
                self.touchdown_timer = 0.0

        elif self.state == "TOUCHDOWN":
            # Payload is resting on the ground: cable tension drops sharply!
            # Drum continues slight slack payout (15 cm) to avoid pulling package over
            if self.cable_deployed_len < (drone_z - ground_z_target + 0.20):
                self.cable_deployed_len += 0.25 * dt

            self.tension_n = max(0.5, float(np.random.normal(1.8, 0.35))) # Minimal hook residual
            self.touchdown_timer += dt

            # Hold touchdown for 1.2s to confirm stable landing before release
            if self.touchdown_timer >= 1.2:
                self.payload_attached = False # Mechanical auto-latch releases package
                self.state = "RETRACTING"

        elif self.state == "RETRACTING":
            # High-speed cable reel-in with empty hook
            target_retract = -WINCH_RETRACT_SPEED_MS
            self.cable_velocity += (target_retract - self.cable_velocity) * 10.0 * dt
            self.cable_deployed_len += self.cable_velocity * dt
            self.payload_z = ground_z_target # Left behind on ground

            # Tension is just the light aerodynamic drag of the bare hook (~1.2 N)
            self.tension_n = max(0.2, 1.2 + float(np.random.normal(0.0, 0.15)))

            if self.cable_deployed_len <= 0.08:
                self.cable_deployed_len = 0.08
                self.cable_velocity = 0.0
                self.state = "COMPLETED"

        elif self.state == "ABORT":
            # Emergency package re-hoist: reel package back up to airframe belly
            target_retract = -WINCH_RETRACT_SPEED_MS * 0.75
            self.cable_velocity += (target_retract - self.cable_velocity) * 8.0 * dt
            self.cable_deployed_len += self.cable_velocity * dt
            self.payload_z = drone_z - self.cable_deployed_len

            # Higher dynamic tension during rapid upward acceleration
            self.tension_n = PAYLOAD_STATIC_WEIGHT_N * 1.35 + float(np.random.normal(0.0, 0.6))

            if self.cable_deployed_len <= 0.08:
                self.cable_deployed_len = 0.08
                self.cable_velocity = 0.0
                self.state = "STANDBY"

        # -----------------------------------------------------------------
        # 2. Snag, Entanglement & Pyrotechnic Guillotine Severance Logic
        # -----------------------------------------------------------------
        if force_snag and self.state in ["LOWERING", "TOUCHDOWN", "RETRACTING"]:
            # External party (person, pet, or fence) pulling vigorously on line
            self.tension_n = float(random.uniform(42.0, 68.0))

        # Check over-tension threshold
        if self.tension_n >= SNAG_ALARM_TENSION_N:
            self.snag_timer += dt
            if self.tension_n >= GUILLOTINE_TRIGGER_TENSION_N or self.snag_timer >= 0.45:
                # Fire pyrotechnic guillotine line cutter to save airframe!
                self.fire_guillotine()
        else:
            self.snag_timer = max(0.0, self.snag_timer - dt)

        self.tension_raw_history.append(self.tension_n)
        return self.get_telemetry()

    def start_delivery(self):
        """Commands autonomous lowering of payload."""
        if self.state == "STANDBY" and self.payload_attached and not self.line_severed:
            self.state = "LOWERING"

    def trigger_abort(self):
        """Commands rapid winch abort and package retrieval."""
        if self.state in ["LOWERING", "TOUCHDOWN"] and self.payload_attached and not self.line_severed:
            self.state = "ABORT"

    def fire_guillotine(self):
        """Fires redundant pyrotechnic cable cutter, severing tether in 30 ms."""
        self.line_severed = True
        self.state = "GUILLOTINE_CUT"
        print(">>> [SAFETY INTERLOCK] PYROTECHNIC GUILLOTINE ACTUATED: TETHER SEVERED TO PRESERVE AIRFRAME <<<")

    def get_telemetry(self):
        return {
            "state": self.state,
            "cable_len": self.cable_deployed_len,
            "cable_vel": self.cable_velocity,
            "payload_z": self.payload_z,
            "tension_n": self.tension_n,
            "payload_attached": self.payload_attached,
            "line_severed": self.line_severed,
            "is_snagged": (self.tension_n >= SNAG_ALARM_TENSION_N)
        }

# ==========================================
# 3. PYTORCH DROPZONE SAFETY PERCEPTION NET
# ==========================================
class DropzoneSafetyNet(nn.Module):
    """
    Neural perception model processing downward range clearances,
    ground slope gradient, load-cell tension slope, and optical bounding boxes
    to evaluate autonomous delivery readiness.
    """
    def __init__(self):
        super().__init__()
        # Inputs: Min Obstacle Range (1) + Slope Deg (1) + Tether Tension (1) + Drone Alt (1) + Tension Deriv (1) = 5
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # [SAFE_TO_LOWER, HOLD_CLEARING, ABORT_WAVE_OFF]
        )

    def forward(self, x):
        return self.net(x)

def _worker_collect_dropzone(num_samples):
    world = DropzoneHazardWorld()
    winch = MotorizedWinchDeliverySystem()
    X_samples, y_labels = [], []

    while len(X_samples) < num_samples:
        drone_pos = np.array([random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(12.0, 18.0)], dtype=np.float32)
        world.human_active = (random.random() < 0.35)
        world.terrain_slope_deg = random.uniform(1.0, 22.0)

        audit = world.audit_dropzone_footprint(drone_pos[:2])

        # Tension simulation
        is_snag = (random.random() < 0.10)
        t_sim = 60.0 if is_snag else (21.5 + random.uniform(-1.0, 1.0))
        d_tension = random.uniform(-5.0, 5.0)

        # Feature vector (5 dims)
        feat = np.array([
            min(15.0, audit["min_clearance"]) / 15.0,
            audit["slope_deg"] / 30.0,
            t_sim / 70.0,
            drone_pos[2] / 25.0,
            d_tension / 20.0
        ], dtype=np.float32)

        # Label: 0 = SAFE, 1 = HOLD (transient human/wind), 2 = ABORT (steep slope or snag)
        if is_snag or audit["slope_deg"] > MAX_ACCEPTABLE_SLOPE_DEG:
            label = 2 # ABORT
        elif audit["human_incursion"] or audit["static_count"] > 0:
            label = 1 # HOLD
        else:
            label = 0 # SAFE

        X_samples.append(feat)
        y_labels.append(label)

    return np.array(X_samples, dtype=np.float32), np.array(y_labels, dtype=np.int64)

def train_dropzone_brain():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"Device: {device}")
    print(f"Training Dropzone Safety & Winch Delivery Brain (Module 10)...")
    print(f"Terrain Audit, Dynamic Incursions & Load-Cell Tension Defense")
    print(f"==========================================")

    print("Generating 80,000 dropzone multi-modal delivery states across CPU cores...")
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per_worker = (80000 + num_workers - 1) // num_workers

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_collect_dropzone, [samples_per_worker] * num_workers)

    X_np = np.concatenate([r[0] for r in results], axis=0)[:80000]
    y_np = np.concatenate([r[1] for r in results], axis=0)[:80000]
    print(f"Dropzone training dataset harvested in {time.time() - t0:.2f}s!")

    X = torch.tensor(X_np, device=device)
    y = torch.tensor(y_np, device=device)

    model = DropzoneSafetyNet().to(device)
    BATCH_SIZE = 2048
    epochs = 10
    num_batches = (len(X) + BATCH_SIZE - 1) // BATCH_SIZE

    optimizer = optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-3, steps_per_epoch=num_batches, epochs=epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"Optimizing Dropzone Safety Classifier (Batch {BATCH_SIZE}, 10 Epochs)...")
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X), device=device)
        total_loss, correct = 0.0, 0

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
            correct += (pred.argmax(dim=1) == by).sum().item()

        acc = (correct / len(X)) * 100.0
        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Dropzone Classifier Accuracy: {acc:.2f}% - Loss: {total_loss/num_batches:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Dropzone Safety Model compiled in {time.time() - t0:.2f}s!")
    return model, device

# ==========================================
# 4. REAL-TIME 3D WINCH DELIVERY HUD
# ==========================================
def project_3d(point, cam_rot, center_screen, scale=36.0):
    """Projects 3D world coordinates onto 2D viewport."""
    yaw, pitch = cam_rot
    x, y, z = point[0], point[1], point[2]

    x1 = x * math.cos(yaw) - y * math.sin(yaw)
    y1 = x * math.sin(yaw) + y * math.cos(yaw)
    z1 = z

    x2 = x1
    y2 = y1 * math.cos(pitch) - z1 * math.sin(pitch)
    z2 = y1 * math.sin(pitch) + z1 * math.cos(pitch)

    dist = 22.0 - y2
    focal = max(2.5, dist)
    sx = int(center_screen[0] + (x2 / focal) * scale * 7.5)
    sy = int(center_screen[1] - (z2 / focal) * scale * 7.5)
    return sx, sy, y2

def run_winch_delivery_hud(model, device):
    pygame.init()
    VIEW_WIDTH = 700
    VIEW_HEIGHT = 700
    SIDEBAR_WIDTH = 380
    WIDTH = VIEW_WIDTH + SIDEBAR_WIDTH
    HEIGHT = VIEW_HEIGHT

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Module 10: Ground Dropzone Safety & Tether Winch Delivery Mechanics")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("consolas", 18, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 13, bold=True)

    world = DropzoneHazardWorld()
    winch = MotorizedWinchDeliverySystem()
    model.eval()

    # Drone Standoff Hover Position
    drone_pos = np.array([0.0, 0.0, DRONE_HOVER_ALT_M], dtype=np.float32)
    drone_yaw_deg = 0.0

    camera_yaw = 0.65
    camera_pitch = 0.42
    center_viewport = (VIEW_WIDTH // 2, VIEW_HEIGHT // 2 + 110)

    force_snag = False
    running = True

    print("================ WINCH DELIVERY CONTROLS ================")
    print(" [SPACE]        : Command Winch Deployment (Deliver Package)")
    print(" [A]            : Command Delivery Abort / Re-hoist Package")
    print(" [H]            : Toggle Human / Pet Ground Incursion")
    print(" [S]            : Inject Cable Snag / Over-Tension Event")
    print(" [C]            : Fire Emergency Pyrotechnic Line Guillotine")
    print(" [R]            : Reset Simulation & Re-arm Winch")
    print(" [ARROW KEYS]   : Orbit 3D Sensor Camera View")
    print("=========================================================\n")

    while running:
        dt = clock.tick(30) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    winch.start_delivery()
                elif event.key == pygame.K_a:
                    winch.trigger_abort()
                elif event.key == pygame.K_h:
                    world.human_active = not world.human_active
                    world.human_pos = np.array([3.2, 1.8, 0.0], dtype=np.float32)
                elif event.key == pygame.K_s:
                    force_snag = not force_snag
                elif event.key == pygame.K_c:
                    winch.fire_guillotine()
                elif event.key == pygame.K_r:
                    world = DropzoneHazardWorld()
                    winch = MotorizedWinchDeliverySystem()
                    force_snag = False

        # Camera rotation
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: camera_yaw -= 0.03
        if keys[pygame.K_RIGHT]: camera_yaw += 0.03
        if keys[pygame.K_UP]: camera_pitch = max(0.1, camera_pitch - 0.02)
        if keys[pygame.K_DOWN]: camera_pitch = min(1.2, camera_pitch + 0.02)

        # 1. Update Ground Actors & Footprint Auditing
        world.update_actors(dt)
        audit = world.audit_dropzone_footprint(drone_pos[:2])

        # Automatic abort if human rushes basket during descent
        if audit["human_incursion"] and winch.state == "LOWERING" and winch.payload_z > 3.0:
            winch.trigger_abort()
            print(">>> [SAFETY INTERLOCK] HUMAN INCURSION DETECTED IN DROPZONE: AUTOMATIC ABORT HOIST ACTIVATED <<<")

        # 2. Step Winch Physics & Load-Cell Tension
        ground_z_beneath = world.query_terrain_height(drone_pos[0], drone_pos[1])
        w_telem = winch.step(drone_pos, ground_z_beneath, force_snag=force_snag, dt=dt)

        # 3. Neural Dropzone Safety Inference
        feat_in = np.array([
            min(15.0, audit["min_clearance"]) / 15.0,
            audit["slope_deg"] / 30.0,
            w_telem["tension_n"] / 70.0,
            drone_pos[2] / 25.0,
            0.0
        ], dtype=np.float32)

        with torch.no_grad():
            feat_t = torch.tensor(feat_in, device=device).unsqueeze(0)
            logits = model(feat_t)[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        screen.fill((10, 13, 19))

        # Dividers
        pygame.draw.line(screen, (35, 45, 60), (VIEW_WIDTH, 0), (VIEW_WIDTH, HEIGHT), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (WIDTH, 70), 2)

        # ==========================================
        # 3D VIEWPORT: AIRFRAME, TETHER & DROPZONE
        # ==========================================
        cam = (camera_yaw, camera_pitch)

        # 1. Terrain Grid with Slope Incline
        for gx in range(-5, 6):
            p1 = np.array([gx * 1.5, -6.0, world.query_terrain_height(gx * 1.5, -6.0)])
            p2 = np.array([gx * 1.5,  6.0, world.query_terrain_height(gx * 1.5,  6.0)])
            sx1, sy1, _ = project_3d(p1, cam, center_viewport)
            sx2, sy2, _ = project_3d(p2, cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        for gy in range(-4, 5):
            p1 = np.array([-7.5, gy * 1.5, world.query_terrain_height(-7.5, gy * 1.5)])
            p2 = np.array([ 7.5, gy * 1.5, world.query_terrain_height( 7.5, gy * 1.5)])
            sx1, sy1, _ = project_3d(p1, cam, center_viewport)
            sx2, sy2, _ = project_3d(p2, cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        # 2. Draw 5-Meter Dropzone Safety Basket Perimeter Ring
        basket_pts = []
        for a_deg in range(0, 360, 20):
            rad = math.radians(a_deg)
            px = drone_pos[0] + math.sin(rad) * DROPZONE_RADIUS_M
            py = drone_pos[1] + math.cos(rad) * DROPZONE_RADIUS_M
            pz = world.query_terrain_height(px, py)
            sx, sy, _ = project_3d(np.array([px, py, pz]), cam, center_viewport)
            basket_pts.append((sx, sy))

        basket_col = (0, 255, 140) if audit["is_safe"] else (255, 60, 80)
        if len(basket_pts) >= 3:
            pygame.draw.lines(screen, basket_col, True, basket_pts, 2)

        # Target center crosshair on ground
        g_center = np.array([drone_pos[0], drone_pos[1], ground_z_beneath])
        csx, csy, _ = project_3d(g_center, cam, center_viewport)
        pygame.draw.circle(screen, basket_col, (csx, csy), 6, 1)

        # 3. Draw Static Obstacles
        for obs in world.static_obstacles:
            ox, oy, _ = project_3d(obs["pos"], cam, center_viewport)
            pygame.draw.circle(screen, (160, 120, 50), (ox, oy), 8)
            screen.blit(font_small.render(obs["type"], True, (180, 160, 120)), (ox + 10, oy - 5))

        # 4. Draw Human / Pet Incursion Actor
        if world.human_active:
            hx, hy, _ = project_3d(world.human_pos, cam, center_viewport)
            pygame.draw.circle(screen, (255, 60, 80), (hx, hy), 9)
            pygame.draw.circle(screen, (255, 255, 255), (hx, hy), 4)
            screen.blit(font_small.render("DYNAMIC HUMAN / PET", True, (255, 60, 80)), (hx + 12, hy - 6))

        # 5. Draw Hovering Drone Airframe (15m AGL)
        dsx, dsy, _ = project_3d(drone_pos, cam, center_viewport)
        pygame.draw.circle(screen, (0, 240, 255), (dsx, dsy), 9)
        pygame.draw.circle(screen, (255, 255, 255), (dsx, dsy), 3)

        # 6. Draw Winch Tether Cable & Payload
        payload_pos = np.array([drone_pos[0], drone_pos[1], w_telem["payload_z"]], dtype=np.float32)
        psx, psy, _ = project_3d(payload_pos, cam, center_viewport)

        if not w_telem["line_severed"]:
            # Active cable line
            cable_col = (255, 60, 80) if w_telem["is_snagged"] else (220, 230, 245)
            cable_width = 3 if w_telem["is_snagged"] else 1
            pygame.draw.line(screen, cable_col, (dsx, dsy), (psx, psy), cable_width)
        else:
            # Severed cable stub
            stub_end = drone_pos - np.array([0.0, 0.0, 0.8])
            ssx, ssy, _ = project_3d(stub_end, cam, center_viewport)
            pygame.draw.line(screen, (255, 60, 80), (dsx, dsy), (ssx, ssy), 2)
            screen.blit(font_small.render("[GUILLOTINE SEVERED]", True, (255, 60, 80)), (ssx + 8, ssy))

        # Payload Delivery Box (Yellow Carton)
        if w_telem["payload_attached"] or w_telem["state"] in ["TOUCHDOWN", "COMPLETED", "GUILLOTINE_CUT"]:
            box_col = (0, 255, 140) if w_telem["state"] == "COMPLETED" else (255, 200, 40)
            pygame.draw.rect(screen, box_col, (psx - 7, psy - 7, 14, 14), border_radius=2)
            pygame.draw.rect(screen, (30, 30, 30), (psx - 7, psy - 7, 14, 14), 1)

        # ==========================================
        # REAL-TIME WINCH LOAD-CELL OSCILLOSCOPE
        # ==========================================
        osc_x = 25
        osc_y = VIEW_HEIGHT - 135
        osc_w = 420
        osc_h = 115
        pygame.draw.rect(screen, (14, 20, 28), (osc_x, osc_y, osc_w, osc_h), border_radius=6)
        pygame.draw.rect(screen, (35, 50, 70), (osc_x, osc_y, osc_w, osc_h), 2)

        screen.blit(font_mono.render("STRAIN-GAUGE LOAD-CELL TENSION (NEWTONS):", True, (240, 240, 240)), (osc_x + 12, osc_y + 8))

        # Snag alert line at 38 N
        snag_py = osc_y + osc_h - int((SNAG_ALARM_TENSION_N / 70.0) * (osc_h - 35)) - 10
        pygame.draw.line(screen, (255, 60, 80), (osc_x + 10, snag_py), (osc_x + osc_w - 10, snag_py), 1)
        screen.blit(font_small.render("38N Snag Threshold", True, (255, 60, 80)), (osc_x + osc_w - 125, snag_py - 12))

        # Static package weight reference line at 21.6 N
        stat_py = osc_y + osc_h - int((PAYLOAD_STATIC_WEIGHT_N / 70.0) * (osc_h - 35)) - 10
        pygame.draw.line(screen, (0, 220, 255), (osc_x + 10, stat_py), (osc_x + osc_w - 10, stat_py), 1)
        screen.blit(font_small.render("21.6N Static Weight", True, (0, 220, 255)), (osc_x + 12, stat_py - 12))

        # Plot tension history waveform
        history = list(winch.tension_raw_history)
        if len(history) >= 2:
            pts = []
            dx_step = (osc_w - 24) / max(1, len(history) - 1)
            for idx, val in enumerate(history):
                px = int(osc_x + 12 + idx * dx_step)
                py = int(osc_y + osc_h - (min(70.0, val) / 70.0) * (osc_h - 35) - 10)
                pts.append((px, py))
            wave_col = (255, 60, 80) if w_telem["is_snagged"] else (0, 255, 140)
            pygame.draw.lines(screen, wave_col, False, pts, 2)

        # ==========================================
        # DROPZONE & WINCH TELEMETRY (SIDEBAR)
        # ==========================================
        sb_x = VIEW_WIDTH + 14
        screen.blit(font_large.render("DROPZONE & WINCH (BMS)", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("The 'Last 50 Feet' Automated Delivery Mechanics", True, (130, 160, 190)), (sb_x, 38))

        # --- PANEL 1: WINCH MECHANICAL STATE MACHINE ---
        p1_y = 80
        screen.blit(font_mono.render("WINCH DEPLOYMENT STATE:", True, (240, 240, 240)), (sb_x, p1_y))

        st_map = {
            "STANDBY": ("READY AT STANDOFF HOVER", (0, 255, 140)),
            "LOWERING": ("CONTROLLED DESCENT ACTIVE", (0, 220, 255)),
            "TOUCHDOWN": ("PACKAGE ON GROUND (LATCH REL)", (255, 200, 40)),
            "RETRACTING": ("REELING EMPTY HOOK TO BELLY", (0, 220, 255)),
            "ABORT": ("EMERGENCY RE-HOIST ABORT!", (255, 180, 40)),
            "GUILLOTINE_CUT": ("PYRO LINE CUTTER SEVERED!", (255, 60, 80)),
            "COMPLETED": ("DELIVERY ACCOMPLISHED (STOWED)", (0, 255, 140))
        }
        st_text, st_col = st_map.get(w_telem["state"], ("UNKNOWN", (160, 180, 205)))

        screen.blit(font_mono.render(f"FSM State:  {st_text}", True, st_col), (sb_x + 10, p1_y + 24))
        screen.blit(font_small.render(f"Cable Length:      L = {w_telem['cable_len']:.2f} m / {TETHER_MAX_LENGTH_M:.1f} m", True, (0, 220, 255)), (sb_x + 10, p1_y + 44))
        screen.blit(font_small.render(f"Cable Velocity:    v = {w_telem['cable_vel']:+.2f} m/s", True, (160, 180, 205)), (sb_x + 10, p1_y + 60))
        screen.blit(font_small.render(f"Payload Elevation: z = {w_telem['payload_z']:.2f} m AGL", True, (255, 200, 40)), (sb_x + 10, p1_y + 76))

        # Cable spool gauge bar [0 to 16.5m]
        bar_x = sb_x + 10
        bar_y = p1_y + 96
        pygame.draw.rect(screen, (24, 32, 45), (bar_x, bar_y, 320, 8), border_radius=2)
        fill_w = int(320 * min(1.0, max(0.0, w_telem["cable_len"] / TETHER_MAX_LENGTH_M)))
        pygame.draw.rect(screen, (0, 220, 255), (bar_x, bar_y, fill_w, 8), border_radius=2)

        # --- PANEL 2: LOAD-CELL STRAIN GAUGE & SNAG DEFENSE ---
        p2_y = bar_y + 24
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (WIDTH - 14, p2_y), 1)
        screen.blit(font_mono.render("STRAIN-GAUGE LOAD-CELL TELEMETRY:", True, (240, 240, 240)), (sb_x, p2_y + 8))

        t_val = w_telem["tension_n"]
        t_col = (255, 60, 80) if w_telem["is_snagged"] else ((255, 200, 40) if t_val < TOUCHDOWN_TENSION_FLOOR_N else (0, 255, 140))
        touch_txt = "TOUCHDOWN CONFIRMED" if (t_val < TOUCHDOWN_TENSION_FLOOR_N and w_telem["state"] in ["TOUCHDOWN", "RETRACTING", "COMPLETED"]) else "FREE-HANGING"

        screen.blit(font_small.render(f"Line Tension:      F = {t_val:.1f} N (Load-Cell)", True, t_col), (sb_x + 10, p2_y + 28))
        screen.blit(font_small.render(f"Touchdown Status:  {touch_txt}", True, (0, 220, 255)), (sb_x + 10, p2_y + 44))
        screen.blit(font_small.render(f"Snag Protection:   38 N Alarm | 48 N Guillotine", True, (160, 180, 205)), (sb_x + 10, p2_y + 60))
        cutter_txt = "SEVERED (AIRFRAME PROTECTED)" if w_telem["line_severed"] else "ARMED (PYROELECTRIC 30ms)"
        screen.blit(font_small.render(f"Line Guillotine:   {cutter_txt}", True, (255, 60, 80) if w_telem["line_severed"] else (0, 255, 140)), (sb_x + 10, p2_y + 76))

        # --- PANEL 3: DOWNWARD DROPZONE TERRAIN AUDIT ---
        p3_y = p2_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (WIDTH - 14, p3_y), 1)
        screen.blit(font_mono.render("DOWNWARD DROPZONE SCANNER:", True, (240, 240, 240)), (sb_x, p3_y + 8))

        slope_col = (0, 255, 140) if audit["slope_deg"] <= MAX_ACCEPTABLE_SLOPE_DEG else (255, 60, 80)
        human_col = (255, 60, 80) if audit["human_incursion"] else (0, 255, 140)

        screen.blit(font_small.render(f"Ground Incline:    {audit['slope_deg']:.1f}° (Limit ≤ 14.0°)", True, slope_col), (sb_x + 10, p3_y + 28))
        screen.blit(font_small.render(f"Dynamic Human/Pet: {'INCURSION DETECTED!' if audit['human_incursion'] else 'BASKET CLEAR'}", True, human_col), (sb_x + 10, p3_y + 44))
        screen.blit(font_small.render(f"Static Obstacles:  {audit['static_count']} Detected inside 5m Ring", True, (160, 180, 205)), (sb_x + 10, p3_y + 60))
        screen.blit(font_small.render(f"Safe Delivery OK:  {'YES (CLEAR TO LOWER)' if audit['is_safe'] else 'NO (HAZARD ACTIVE)'}", True, (0, 255, 140) if audit['is_safe'] else (255, 60, 80)), (sb_x + 10, p3_y + 76))

        # --- PANEL 4: PYTORCH DROPZONE SAFETY CLASSIFIER ---
        p4_y = p3_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (WIDTH - 14, p4_y), 1)
        screen.blit(font_mono.render("NEURAL DROPZONE DECISION ENGINE:", True, (240, 240, 240)), (sb_x, p4_y + 8))

        classes = ["Safe to Lower", "Hold / Wave Clearing", "Abort Mission"]
        cls_cols = [(0, 255, 140), (255, 200, 40), (255, 60, 80)]
        for c_i in range(3):
            cy = p4_y + 28 + c_i * 18
            prob_val = probs[c_i]
            bar_w = int(prob_val * 160)
            screen.blit(font_small.render(f"{classes[c_i]}:", True, (160, 180, 205)), (sb_x + 10, cy))
            pygame.draw.rect(screen, (24, 32, 45), (sb_x + 165, cy + 2, 160, 7), border_radius=2)
            pygame.draw.rect(screen, cls_cols[c_i], (sb_x + 165, cy + 2, bar_w, 7), border_radius=2)

        # Bottom Telemetry Banner
        bot_y = p4_y + 92
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y), (WIDTH - 14, bot_y), 1)
        if w_telem["line_severed"]:
            banner_str = "EMERGENCY: LINE GUILLOTINE FIRED"
            banner_col = (255, 60, 80)
        elif w_telem["is_snagged"]:
            banner_str = "ALERT: CABLE OVER-TENSION SNAG"
            banner_col = (255, 60, 80)
        elif audit["human_incursion"]:
            banner_str = "HOLD: LIVING BEING IN DELIVERY BASKET"
            banner_col = (255, 200, 40)
        else:
            banner_str = "DELIVERY ENVELOPE: OPTIMAL"
            banner_col = (0, 255, 140)
        screen.blit(font_mono.render(banner_str, True, banner_col), (sb_x + 8, bot_y + 8))

        # Header UI
        head_txt = font_large.render(f"Alt: {drone_pos[2]:.1f}m | Winch: {w_telem['cable_len']:.1f}m | Tens: {t_val:.1f}N | Dropzone", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("Tethered Winch Dynamics, Strain-Gauge Touchdown Sensing, Terrain Slope Audit & Guillotine Defense", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

# ==========================================
# 5. MAIN ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    mp.freeze_support()
    model, device = train_dropzone_brain()
    print("Launching Ground Dropzone Safety & Winch Delivery Simulator...")
    run_winch_delivery_hud(model, device)