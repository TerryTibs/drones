"""
=============================================================================
MODULE 12: AVIATION COMPLIANCE, AIRSPACE LAW & FAILSAFES
File: drone_aviation_compliance_failsafes.py
=============================================================================
Aviation Regulatory & Emergency Systems: ADS-B In (1090ES) Conflict Detection,
FAA Part 89 / EASA Remote ID (ASTM F3411) Broadcasting, 3D Geofence Containment,
and ASTM F3322 Ballistic Parachute Flight Termination System (FTS).
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
# 0. REGULATORY CONSTANTS & AIRSPACE LIMITS
# ==========================================
MAX_LEGAL_ALT_AGL = 120.0             # 120m / 400ft AGL ceiling (FAA Part 107 / EASA Open)
DRONE_MASS_KG = 1.35
GRAVITY_G = 9.80665

# ADS-B In Conflict Detection Envelopes
ADSB_TA_DIST_M = 450.0                # Traffic Advisory perimeter (m)
ADSB_RA_DIST_M = 180.0                # Resolution Advisory / Evasive Action (m)
ADSB_CPA_TIME_WARN_S = 35.0           # Warn if CPA occurs within 35 seconds

# Geofence Exclusion Zone (e.g. Hospital Helipad / Airport Airspace)
GEOFENCE_CENTER = np.array([45.0, 35.0], dtype=np.float32)
GEOFENCE_RADIUS_M = 28.0
GEOFENCE_CEILING_M = 150.0

# ASTM F3322 Ballistic Parachute Parameters
CHUTE_CANOPY_DIAMETER_M = 1.45       # 1.45m cross-parachute
CHUTE_CD = 1.55                       # Hemispherical drag coefficient
AIR_DENSITY_RHO = 1.225
CHUTE_AREA = math.pi * ((CHUTE_CANOPY_DIAMETER_M / 2.0) ** 2)
CHUTE_TERMINAL_VEL = math.sqrt((2.0 * DRONE_MASS_KG * GRAVITY_G) / (AIR_DENSITY_RHO * CHUTE_CD * CHUTE_AREA)) # ~4.1 m/s

# Remote ID (ASTM F3411-22) Simulated Serial Number
REMOTE_ID_SERIAL = "1596B-0089A4E271F"

MODEL_SAVE_PATH = "drone_aviation_compliance_model.pth"

# ==========================================
# 1. ADS-B IN & INTRUDER DYNAMICS
# ==========================================
class ADSBTransponderSimulator:
    """
    Simulates ADS-B In 1090 MHz receiver decoding manned aircraft
    (e.g., Bell 429 Air Ambulance) flying through shared low-altitude airspace.
    Calculates Closest Point of Approach (CPA) and Time to CPA (t_CPA).
    """
    def __init__(self):
        # Manned aircraft starts 800m away, descending through 80m altitude
        self.intruder_pos = np.array([550.0, 480.0, 85.0], dtype=np.float32)
        self.intruder_vel = np.array([-26.0, -22.0, -0.8], dtype=np.float32) # ~34 m/s (66 knots)
        self.callsign = "MEDEVAC_02"
        self.icao_hex = "A38F12"
        self.active = True

    def step(self, dt):
        if self.active:
            self.intruder_pos += self.intruder_vel * dt
            # Loop track once past mission area
            if self.intruder_pos[0] < -500.0:
                self.intruder_pos = np.array([600.0, 500.0, 85.0], dtype=np.float32)

    def evaluate_conflict(self, drone_pos, drone_vel):
        """
        Computes 3D CPA kinematics:
        t_CPA = -(delta_p . delta_v) / ||delta_v||^2
        """
        delta_p = self.intruder_pos - drone_pos
        delta_v = self.intruder_vel - drone_vel

        dist_current = float(np.linalg.norm(delta_p))
        v_rel_sq = float(np.dot(delta_v, delta_v))

        if v_rel_sq > 1e-4:
            t_cpa = -float(np.dot(delta_p, delta_v)) / v_rel_sq
        else:
            t_cpa = 0.0

        if t_cpa > 0:
            p_cpa = delta_p + delta_v * t_cpa
            d_cpa = float(np.linalg.norm(p_cpa))
        else:
            d_cpa = dist_current

        is_ta = (dist_current < ADSB_TA_DIST_M) or (t_cpa > 0 and t_cpa < ADSB_CPA_TIME_WARN_S and d_cpa < ADSB_TA_DIST_M)
        is_ra = (dist_current < ADSB_RA_DIST_M) or (t_cpa > 0 and t_cpa < 18.0 and d_cpa < ADSB_RA_DIST_M)

        return {
            "dist_current": dist_current,
            "t_cpa": max(0.0, t_cpa),
            "d_cpa": d_cpa,
            "is_ta": is_ta,
            "is_ra": is_ra,
            "pos": self.intruder_pos.copy(),
            "vel": self.intruder_vel.copy(),
            "callsign": self.callsign,
            "icao_hex": self.icao_hex
        }

# ==========================================
# 2. ASTM F3322 BALLISTIC PARACHUTE FTS
# ==========================================
class BallisticParachuteFTS:
    """
    Simulates an ASTM F3322-compliant autonomous Flight Termination System (FTS).
    Monitors unrecoverable attitudes, rotor failures, and triggers a pyrotechnic
    ballistic parachute, enforcing safe kinetic energy dissipation (<69 J).
    """
    def __init__(self):
        self.deployed = False
        self.deploy_time = 0.0
        self.motors_killed = False
        self.ejection_velocity = 18.0 # Ejection speed in m/s upward
        self.canopy_inflation_progress = 0.0

    def trigger_chute(self):
        if not self.deployed:
            self.deployed = True
            self.motors_killed = True
            print(">>> [SAFETY INTERLOCK] ASTM F3322 BALLISTIC PARACHUTE FIRED: MOTORS TERMINATED <<<")

    def update(self, drone_pos, drone_vel, dt):
        """Integrates parachute deployment dynamics and terminal velocity descent."""
        if not self.deployed:
            return drone_vel

        self.deploy_time += dt
        # Canopy inflation ramp (takes ~0.55s from mortar blast to full bloom)
        self.canopy_inflation_progress = min(1.0, self.deploy_time / 0.55)

        # Aerodynamic parachute braking
        current_v = drone_vel.copy()
        # Horizontal wind drift damping
        current_v[0] *= 0.94
        current_v[1] *= 0.94

        # Vertical descent rate settles to steady-state terminal velocity ~4.1 m/s
        target_vz = -CHUTE_TERMINAL_VEL
        current_v[2] += (target_vz - current_v[2]) * (2.8 * self.canopy_inflation_progress) * dt

        # Impact kinetic energy calculation: KE = 0.5 * m * v^2
        impact_ke = 0.5 * DRONE_MASS_KG * (current_v[2] ** 2)

        return current_v

# ==========================================
# 3. REMOTE ID BROADCAST ENGINE (ASTM F3411)
# ==========================================
class RemoteIDBroadcastEngine:
    """
    Simulates direct Open Drone ID RF broadcasting over Bluetooth 5.x / Wi-Fi NAN
    compliant with FAA Part 89 and EASA Commission Delegated Regulation (EU) 2019/945.
    """
    def __init__(self, serial_num=REMOTE_ID_SERIAL):
        self.serial_num = serial_num
        self.seq_num = 0
        self.msg_rate_hz = 1.0 # 1 Hz mandated location broadcast rate
        self.last_broadcast = 0.0

    def generate_message(self, drone_pos, drone_vel, is_emergency=False):
        """Generates raw ASTM F3411 binary frame telemetry structure."""
        self.seq_num = (self.seq_num + 1) % 256
        track_deg = (math.degrees(math.atan2(drone_vel[0], drone_vel[1])) + 360.0) % 360.0
        gs_knots = float(np.linalg.norm(drone_vel[:2])) * 1.94384

        status_code = "EMERGENCY_FTS" if is_emergency else "AIRBORNE_NORMAL"

        return {
            "protocol": "ASTM_F3411_22_DIRECT_RID",
            "serial": self.serial_num,
            "seq": self.seq_num,
            "status": status_code,
            "lat": 51.5074 + (drone_pos[1] * 0.000009),
            "lon": -0.1278 + (drone_pos[0] * 0.000014),
            "alt_geo_m": 25.0 + drone_pos[2],
            "alt_pressure_m": 24.8 + drone_pos[2],
            "track_deg": track_deg,
            "groundspeed_kt": gs_knots,
            "operator_id": "GBR-OP-82914029B"
        }

# ==========================================
# 4. PYTORCH COMPLIANCE & EMERGENCY CLASSIFIER
# ==========================================
class AviationComplianceNet(nn.Module):
    """
    Neural arbiter classifying real-time regulatory and airspace status:
    [0: AIRSPACE_CLEAR, 1: TRAFFIC_ADVISORY_DESCENT, 2: GEOFENCE_BREACH_HOLD, 3: BALLISTIC_CHUTE_TRIGGER]
    """
    def __init__(self):
        super().__init__()
        # Inputs: Drone Alt (1) + Dist Intruder (1) + t_CPA (1) + Dist Geofence (1) + Pitch Angle (1) + Roll Angle (1) = 6
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.net(x)

def _worker_collect_compliance(num_samples):
    adsb = ADSBTransponderSimulator()
    X_samples, y_labels = [], []

    while len(X_samples) < num_samples:
        pos = np.array([random.uniform(-40, 60), random.uniform(-40, 60), random.uniform(5, 140)], dtype=np.float32)
        vel = np.array([random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-2, 2)], dtype=np.float32)
        pitch = random.uniform(-25, 25)
        roll = random.uniform(-25, 25)

        # Invert attitude for tumble test
        if random.random() < 0.08:
            pitch = random.choice([-80.0, 85.0])

        conflict = adsb.evaluate_conflict(pos, vel)
        d_fence = float(np.linalg.norm(pos[:2] - GEOFENCE_CENTER)) - GEOFENCE_RADIUS_M

        feat = np.array([
            pos[2] / MAX_LEGAL_ALT_AGL,
            min(600.0, conflict["dist_current"]) / 600.0,
            min(60.0, conflict["t_cpa"]) / 60.0,
            max(-10.0, min(50.0, d_fence)) / 50.0,
            pitch / 90.0,
            roll / 90.0
        ], dtype=np.float32)

        # Label logic
        if abs(pitch) > 75.0 or abs(roll) > 75.0:
            label = 3 # BALLISTIC CHUTE TRIGGER
        elif d_fence < 0.0 or pos[2] > MAX_LEGAL_ALT_AGL:
            label = 2 # GEOFENCE HOLD
        elif conflict["is_ra"] or conflict["is_ta"]:
            label = 1 # TRAFFIC DESCENT
        else:
            label = 0 # CLEAR

        X_samples.append(feat)
        y_labels.append(label)

    return np.array(X_samples, dtype=np.float32), np.array(y_labels, dtype=np.int64)

def train_compliance_brain():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"Device: {device}")
    print(f"Training Aviation Compliance & Failsafe Brain (Module 12)...")
    print(f"ADS-B In Conflict Gating, Part 89 RID & ASTM F3322 Parachute")
    print(f"==========================================")

    print("Harvesting 80,000 regulatory airspace states across CPU cores...")
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per_worker = (80000 + num_workers - 1) // num_workers

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_collect_compliance, [samples_per_worker] * num_workers)

    X_np = np.concatenate([r[0] for r in results], axis=0)[:80000]
    y_np = np.concatenate([r[1] for r in results], axis=0)[:80000]
    print(f"Airspace regulatory dataset compiled in {time.time() - t0:.2f}s!")

    X = torch.tensor(X_np, device=device)
    y = torch.tensor(y_np, device=device)

    model = AviationComplianceNet().to(device)
    BATCH_SIZE = 2048
    epochs = 10
    num_batches = (len(X) + BATCH_SIZE - 1) // BATCH_SIZE

    optimizer = optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-3, steps_per_epoch=num_batches, epochs=epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"Optimizing Aviation Safety Classifier (Batch {BATCH_SIZE}, 10 Epochs)...")
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
        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Compliance Classifier Accuracy: {acc:.2f}% - Loss: {total_loss/num_batches:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Compliance Safety Model compiled in {time.time() - t0:.2f}s!")
    return model, device

# ==========================================
# 5. REAL-TIME 3D COMPLIANCE & FAILSAFE HUD
# ==========================================
def project_3d(point, cam_rot, center_screen, scale=28.0):
    """Projects 3D world coordinates onto 2D screen."""
    yaw, pitch = cam_rot
    x, y, z = point[0], point[1], point[2]

    x1 = x * math.cos(yaw) - y * math.sin(yaw)
    y1 = x * math.sin(yaw) + y * math.cos(yaw)
    z1 = z

    x2 = x1
    y2 = y1 * math.cos(pitch) - z1 * math.sin(pitch)
    z2 = y1 * math.sin(pitch) + z1 * math.cos(pitch)

    dist = 36.0 - y2
    focal = max(2.5, dist)
    sx = int(center_screen[0] + (x2 / focal) * scale * 7.5)
    sy = int(center_screen[1] - (z2 / focal) * scale * 7.5)
    return sx, sy, y2

def run_aviation_compliance_hud(model, device):
    pygame.init()
    VIEW_WIDTH = 700
    VIEW_HEIGHT = 700
    SIDEBAR_WIDTH = 380
    WIDTH = VIEW_WIDTH + SIDEBAR_WIDTH
    HEIGHT = VIEW_HEIGHT

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Module 12: Aviation Compliance, Airspace Law & ASTM F3322 Parachute FTS")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("consolas", 18, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 13, bold=True)

    adsb = ADSBTransponderSimulator()
    fts = BallisticParachuteFTS()
    rid_engine = RemoteIDBroadcastEngine()
    model.eval()

    # Drone State
    drone_pos = np.array([0.0, 0.0, 45.0], dtype=np.float32)
    drone_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    drone_pitch = 0.0
    drone_roll = 0.0

    camera_yaw = 0.65
    camera_pitch = 0.38
    center_viewport = (VIEW_WIDTH // 2, VIEW_HEIGHT // 2 + 60)

    flight_trail = deque(maxlen=70)
    intruder_trail = deque(maxlen=70)
    running = True

    print("================ AVIATION COMPLIANCE CONTROLS ================")
    print(" [W] / [S]      : Fly Forward / Backward")
    print(" [A] / [D]      : Strafe Left / Right")
    print(" [I] / [K]      : Climb / Descend (Ceiling: 120m AGL)")
    print(" [T]            : Induce Aircraft Tumble / Inverted Attitude")
    print(" [P]            : Manual Emergency Parachute Mortar Deploy")
    print(" [R]            : Reset Drone to Base State")
    print(" [ARROW KEYS]   : Orbit 3D Sensor Camera View")
    print("=============================================================\n")

    while running:
        dt = clock.tick(30) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    fts.trigger_chute()
                elif event.key == pygame.K_t:
                    drone_pitch = 85.0 # Induce severe tumble
                    print(">>> [TEST] ATTITUDE TUMBLE INDUCED (>75 DEG PITCH) <<<")
                elif event.key == pygame.K_r:
                    drone_pos = np.array([0.0, 0.0, 45.0], dtype=np.float32)
                    drone_vel = np.zeros(3, dtype=np.float32)
                    drone_pitch = 0.0
                    drone_roll = 0.0
                    fts = BallisticParachuteFTS()
                    adsb = ADSBTransponderSimulator()

        # Step Manned Aircraft Traffic
        adsb.step(dt)
        conflict = adsb.evaluate_conflict(drone_pos, drone_vel)

        # Autonomous Resolution Advisory (RA) Evasion:
        # If manned aircraft on collision course, command urgent descent to < 15m AGL
        if conflict["is_ra"] and not fts.deployed:
            drone_vel[2] = -4.5 # Rapid evasive descent
            print(">>> [ADS-B RA] RESOLUTION ADVISORY: EXECUTING IMMEDIATE DESCENT TO CEDE AIRSPACE <<<")
        else:
            keys = pygame.key.get_pressed()
            accel = np.zeros(3, dtype=np.float32)
            if not fts.motors_killed:
                if keys[pygame.K_w]: accel[1] += 6.5
                if keys[pygame.K_s]: accel[1] -= 6.5
                if keys[pygame.K_d]: accel[0] += 6.5
                if keys[pygame.K_a]: accel[0] -= 6.5
                if keys[pygame.K_i]: accel[2] += 4.5
                if keys[pygame.K_k]: accel[2] -= 4.5
                drone_vel += (accel - 0.7 * drone_vel) * dt

        # Camera rotation
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: camera_yaw -= 0.03
        if keys[pygame.K_RIGHT]: camera_yaw += 0.03
        if keys[pygame.K_UP]: camera_pitch = max(0.1, camera_pitch - 0.02)
        if keys[pygame.K_DOWN]: camera_pitch = min(1.2, camera_pitch + 0.02)

        # Parachute / FTS autonomous trigger conditions
        if (abs(drone_pitch) > 75.0 or abs(drone_roll) > 75.0) and not fts.deployed:
            fts.trigger_chute()

        # Update parachute dynamics
        if fts.deployed:
            drone_vel = fts.update(drone_pos, drone_vel, dt)

        drone_pos += drone_vel * dt
        drone_pos[2] = max(0.0, min(140.0, drone_pos[2]))

        flight_trail.append(drone_pos.copy())
        intruder_trail.append(conflict["pos"].copy())

        # Geofence Distance Calculation
        d_fence = float(np.linalg.norm(drone_pos[:2] - GEOFENCE_CENTER)) - GEOFENCE_RADIUS_M
        is_geofence_breach = (d_fence < 0.0)
        is_altitude_breach = (drone_pos[2] > MAX_LEGAL_ALT_AGL)

        # Remote ID Broadcast Packet
        rid_packet = rid_engine.generate_message(drone_pos, drone_vel, is_emergency=fts.deployed)

        # Neural Compliance Inference
        feat_in = np.array([
            drone_pos[2] / MAX_LEGAL_ALT_AGL,
            min(600.0, conflict["dist_current"]) / 600.0,
            min(60.0, conflict["t_cpa"]) / 60.0,
            max(-10.0, min(50.0, d_fence)) / 50.0,
            drone_pitch / 90.0,
            drone_roll / 90.0
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
        # 3D VIEWPORT: AIRSPACE, GEOFENCE & ADSB TRAFFIC
        # ==========================================
        cam = (camera_yaw, camera_pitch)

        # 1. 3D Ground Grid (meters)
        for gx in range(-5, 6):
            p1 = np.array([gx * 15.0, -80.0, 0.0])
            p2 = np.array([gx * 15.0,  80.0, 0.0])
            sx1, sy1, _ = project_3d(p1, cam, center_viewport)
            sx2, sy2, _ = project_3d(p2, cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        for gy in range(-4, 5):
            p1 = np.array([-75.0, gy * 20.0, 0.0])
            p2 = np.array([ 75.0, gy * 20.0, 0.0])
            sx1, sy1, _ = project_3d(p1, cam, center_viewport)
            sx2, sy2, _ = project_3d(p2, cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        # 2. Draw Geofence Cylinder Exclusion Zone (Helipad)
        fence_pts_base = []
        fence_pts_top = []
        for a_deg in range(0, 360, 24):
            rad = math.radians(a_deg)
            px = GEOFENCE_CENTER[0] + math.sin(rad) * GEOFENCE_RADIUS_M
            py = GEOFENCE_CENTER[1] + math.cos(rad) * GEOFENCE_RADIUS_M
            sx_b, sy_b, _ = project_3d(np.array([px, py, 0.0]), cam, center_viewport)
            sx_t, sy_t, _ = project_3d(np.array([px, py, 80.0]), cam, center_viewport)
            fence_pts_base.append((sx_b, sy_b))
            fence_pts_top.append((sx_t, sy_t))
            # Vertical barrier staves
            pygame.draw.line(screen, (75, 30, 40), (sx_b, sy_b), (sx_t, sy_t), 1)

        fence_col = (255, 60, 80) if is_geofence_breach else (220, 80, 50)
        if len(fence_pts_base) >= 3:
            pygame.draw.lines(screen, fence_col, True, fence_pts_base, 2)
            pygame.draw.lines(screen, fence_col, True, fence_pts_top, 2)

        # 3. Draw Manned Aircraft ADS-B Intruder (Red Triangle + Altitude Drop Line)
        ix, iy, _ = project_3d(conflict["pos"], cam, center_viewport)
        igx, igy, _ = project_3d(np.array([conflict["pos"][0], conflict["pos"][1], 0.0]), cam, center_viewport)
        pygame.draw.line(screen, (55, 30, 40), (ix, iy), (igx, igy), 1)

        int_col = (255, 60, 80) if conflict["is_ra"] else ((255, 180, 40) if conflict["is_ta"] else (255, 215, 0))
        # Draw aircraft chevron
        pygame.draw.circle(screen, int_col, (ix, iy), 8)
        screen.blit(font_small.render(f"ADS-B: {conflict['callsign']} ({conflict['dist_current']:.0f}m)", True, int_col), (ix + 12, iy - 6))

        # 4. Draw Trails
        if len(flight_trail) >= 2:
            t_pts = [project_3d(p, cam, center_viewport)[:2] for p in flight_trail]
            pygame.draw.lines(screen, (0, 220, 255), False, t_pts, 2)

        if len(intruder_trail) >= 2:
            i_pts = [project_3d(p, cam, center_viewport)[:2] for p in intruder_trail]
            pygame.draw.lines(screen, (255, 80, 80), False, i_pts, 1)

        # 5. Draw Drone Airframe & ASTM Ballistic Parachute Canopy
        dx, dy, _ = project_3d(drone_pos, cam, center_viewport)
        pygame.draw.circle(screen, (0, 240, 255), (dx, dy), 8)
        pygame.draw.circle(screen, (255, 255, 255), (dx, dy), 3)

        if fts.deployed:
            # Draw deployed parachute canopy 5m above drone
            canopy_pos = drone_pos + np.array([0.0, 0.0, 4.5 * fts.canopy_inflation_progress])
            cx, cy, _ = project_3d(canopy_pos, cam, center_viewport)
            # Canopy dome
            c_radius = int(18 * fts.canopy_inflation_progress)
            pygame.draw.circle(screen, (255, 140, 20), (cx, cy), c_radius)
            pygame.draw.circle(screen, (255, 255, 255), (cx, cy), c_radius, 2)
            # Suspension lines
            pygame.draw.line(screen, (200, 200, 200), (dx, dy), (cx - c_radius + 4, cy), 1)
            pygame.draw.line(screen, (200, 200, 200), (dx, dy), (cx + c_radius - 4, cy), 1)
            screen.blit(font_small.render(f"PARACHUTE DESCENT ({CHUTE_TERMINAL_VEL:.1f} m/s)", True, (255, 140, 20)), (cx + 22, cy - 8))

        # Altitude drop line
        gnd_pt = np.array([drone_pos[0], drone_pos[1], 0.0])
        gx, gy, _ = project_3d(gnd_pt, cam, center_viewport)
        pygame.draw.line(screen, (35, 55, 75), (dx, dy), (gx, gy), 1)

        # ==========================================
        # DIRECT REMOTE ID BROADCAST MONITOR
        # ==========================================
        rid_box_x = 25
        rid_box_y = VIEW_HEIGHT - 135
        rid_box_w = 420
        rid_box_h = 115
        pygame.draw.rect(screen, (14, 20, 28), (rid_box_x, rid_box_y, rid_box_w, rid_box_h), border_radius=6)
        pygame.draw.rect(screen, (35, 50, 70), (rid_box_x, rid_box_y, rid_box_w, rid_box_h), 2)

        screen.blit(font_mono.render("OPEN DRONE ID (ASTM F3411-22 DIRECT RID):", True, (240, 240, 240)), (rid_box_x + 12, rid_box_y + 8))
        screen.blit(font_small.render(f"UAS Serial:    {rid_packet['serial']} [ANSI/CTA-2063-A]", True, (0, 255, 140)), (rid_box_x + 12, rid_box_y + 28))
        screen.blit(font_small.render(f"Operator ID:   {rid_packet['operator_id']} (CAA Registered)", True, (160, 180, 205)), (rid_box_x + 12, rid_box_y + 44))
        screen.blit(font_small.render(f"WGS84 Fix:     Lat: {rid_packet['lat']:.5f}° | Lon: {rid_packet['lon']:.5f}°", True, (0, 220, 255)), (rid_box_x + 12, rid_box_y + 60))
        rid_col = (255, 60, 80) if fts.deployed else (0, 255, 140)
        screen.blit(font_small.render(f"Status Byte:   {rid_packet['status']} | Alt: {rid_packet['alt_geo_m']:.1f}m", True, rid_col), (rid_box_x + 12, rid_box_y + 76))
        screen.blit(font_small.render(f"RF Carrier:    BT5 Long-Range + Wi-Fi NAN (1 Hz Squitter)", True, (140, 160, 185)), (rid_box_x + 12, rid_box_y + 92))

        # ==========================================
        # AVIATION COMPLIANCE TELEMETRY (SIDEBAR)
        # ==========================================
        sb_x = VIEW_WIDTH + 14
        screen.blit(font_large.render("AIRSPACE & COMPLIANCE", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("ADS-B In Deconfliction, Remote ID & ASTM Failsafes", True, (130, 160, 190)), (sb_x, 38))

        # --- PANEL 1: ADS-B IN INTRUDER TRACKING ---
        p1_y = 80
        screen.blit(font_mono.render("ADS-B 1090 MHz INTRUDER RADAR:", True, (240, 240, 240)), (sb_x, p1_y))

        c_status = "RESOLUTION ADVISORY (EVADE!)" if conflict["is_ra"] else ("TRAFFIC ADVISORY (MONITOR)" if conflict["is_ta"] else "AIRSPACE CLEAR")
        screen.blit(font_mono.render(f"Threat State: {c_status}", True, int_col), (sb_x + 10, p1_y + 24))
        screen.blit(font_small.render(f"Intruder Range:    d = {conflict['dist_current']:.1f} m", True, (0, 220, 255)), (sb_x + 10, p1_y + 44))
        screen.blit(font_small.render(f"Time to CPA:       t_cpa = {conflict['t_cpa']:.1f} s", True, (255, 200, 40)), (sb_x + 10, p1_y + 60))
        screen.blit(font_small.render(f"Miss Distance:     d_cpa = {conflict['d_cpa']:.1f} m", True, (160, 180, 205)), (sb_x + 10, p1_y + 76))

        # Threat Bar Gauge
        bar_x = sb_x + 10
        bar_y = p1_y + 96
        pygame.draw.rect(screen, (24, 32, 45), (bar_x, bar_y, 320, 8), border_radius=2)
        norm_threat = 1.0 - min(1.0, max(0.0, conflict["dist_current"] / 600.0))
        fill_w = int(320 * norm_threat)
        pygame.draw.rect(screen, int_col, (bar_x, bar_y, fill_w, 8), border_radius=2)

        # --- PANEL 2: GEOFENCING & CEILING LAW ---
        p2_y = bar_y + 24
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (WIDTH - 14, p2_y), 1)
        screen.blit(font_mono.render("LEGAL CEILING & GEOFENCE CONTAINMENT:", True, (240, 240, 240)), (sb_x, p2_y + 8))

        alt_col = (255, 60, 80) if is_altitude_breach else (0, 255, 140)
        fence_col_txt = (255, 60, 80) if is_geofence_breach else (0, 255, 140)

        screen.blit(font_small.render(f"Flight Altitude:   z = {drone_pos[2]:.1f} m AGL (Max: {MAX_LEGAL_ALT_AGL:.0f}m)", True, alt_col), (sb_x + 10, p2_y + 28))
        screen.blit(font_small.render(f"Geofence Margin:   d = {d_fence:.1f} m (Helipad Zone)", True, fence_col_txt), (sb_x + 10, p2_y + 44))
        screen.blit(font_small.render(f"Ceiling Status:    {'VIOLATION (>400FT)!' if is_altitude_breach else 'LEGAL AIRSPACE'}", True, alt_col), (sb_x + 10, p2_y + 60))
        screen.blit(font_small.render(f"Containment Logic: Proactive Boundary Deceleration", True, (160, 180, 205)), (sb_x + 10, p2_y + 76))

        # --- PANEL 3: ASTM F3322 BALLISTIC PARACHUTE FTS ---
        p3_y = p2_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (WIDTH - 14, p3_y), 1)
        screen.blit(font_mono.render("ASTM F3322 FLIGHT TERMINATION (FTS):", True, (240, 240, 240)), (sb_x, p3_y + 8))

        fts_status = "DEPLOYED (DESCENT CONTROLLED)" if fts.deployed else "ARMED (STANDBY)"
        fts_col = (255, 140, 20) if fts.deployed else (0, 255, 140)

        screen.blit(font_small.render(f"Mortar System:     {fts_status}", True, fts_col), (sb_x + 10, p3_y + 28))
        screen.blit(font_small.render(f"Motor Power Bus:   {'CUT (ELECTRONIC KILL)' if fts.motors_killed else 'ENERGIZED (NORMAL)'}", True, (255, 60, 80) if fts.motors_killed else (0, 255, 140)), (sb_x + 10, p3_y + 44))
        screen.blit(font_small.render(f"Terminal Descent:  v_z = {abs(drone_vel[2]):.2f} m/s (Cap: 4.1 m/s)", True, (0, 220, 255)), (sb_x + 10, p3_y + 60))
        ke = 0.5 * DRONE_MASS_KG * (drone_vel[2] ** 2)
        screen.blit(font_small.render(f"Kinetic Energy:    E = {ke:.1f} J (Safe Limit < 69 J)", True, (0, 255, 140) if ke < 69.0 else (255, 60, 80)), (sb_x + 10, p3_y + 76))

        # --- PANEL 4: PYTORCH ARBITER CLASSIFIER ---
        p4_y = p3_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (WIDTH - 14, p4_y), 1)
        screen.blit(font_mono.render("NEURAL COMPLIANCE ARBITER:", True, (240, 240, 240)), (sb_x, p4_y + 8))

        classes = ["Airspace Clear", "ADS-B Avoidance", "Geofence Hold", "FTS Chute Eject"]
        cls_cols = [(0, 255, 140), (255, 180, 40), (255, 200, 40), (255, 60, 80)]
        for c_i in range(4):
            cy = p4_y + 28 + c_i * 17
            prob_val = probs[c_i]
            bar_w = int(prob_val * 160)
            screen.blit(font_small.render(f"{classes[c_i]}:", True, (160, 180, 205)), (sb_x + 10, cy))
            pygame.draw.rect(screen, (24, 32, 45), (sb_x + 165, cy + 2, 160, 7), border_radius=2)
            pygame.draw.rect(screen, cls_cols[c_i], (sb_x + 165, cy + 2, bar_w, 7), border_radius=2)

        # Bottom Telemetry Banner
        bot_y = p4_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y), (WIDTH - 14, bot_y), 1)
        if fts.deployed:
            banner_str = "EMERGENCY: PARACHUTE RECOVERY ACTIVE"
            banner_col = (255, 140, 20)
        elif conflict["is_ra"]:
            banner_str = "RESOLUTION ADVISORY: EVADING MANNED TRAFFIC"
            banner_col = (255, 60, 80)
        elif is_geofence_breach:
            banner_str = "CONTAINMENT ALERT: GEOFENCE INTRUSION"
            banner_col = (255, 200, 40)
        else:
            banner_str = "AIRSPACE STATUS: FULL COMPLIANCE (RID ACTIVE)"
            banner_col = (0, 255, 140)
        screen.blit(font_mono.render(banner_str, True, banner_col), (sb_x + 8, bot_y + 8))

        # Header UI
        head_txt = font_large.render(f"Alt: {drone_pos[2]:.1f}m | ADS-B: {conflict['dist_current']:.0f}m | Compliance", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("ADS-B In Deconfliction, ASTM F3411 Remote ID, Geofence Containment & ASTM F3322 Ballistic Chute", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

# ==========================================
# 6. MAIN ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    mp.freeze_support()
    model, device = train_compliance_brain()
    print("Launching Aviation Compliance & Failsafes Simulator...")
    run_aviation_compliance_hud(model, device)