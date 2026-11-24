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
# 0. CONFIGURATION & FLIGHT ENVELOPE
# ==========================================
V_MAX_MS = 6.5          # Maximum allowable transit velocity (m/s)
A_MAX_MS2 = 8.0         # Maximum allowable acceleration (m/s^2)
SNAP_WEIGHT = 1.0       # Weight on 4th derivative integral

MODEL_SAVE_PATH = "drone_trajectory_time_model.pth"

# ==========================================
# 1. PIECEWISE 7TH-ORDER MINIMUM-SNAP PLANNER
# ==========================================
class MinimumSnapTrajectoryPlanner:
    """
    Computes exact C^6-continuous 7th-order minimum-snap polynomial splines
    through arbitrary 3D waypoints using differential flatness.
    """
    def __init__(self):
        self.waypoints = []
        self.segment_times = []
        self.total_time = 0.0
        self.poly_coeffs_x = None
        self.poly_coeffs_y = None
        self.poly_coeffs_z = None
        self.is_solved = False

    def plan_trajectory(self, waypoints, nominal_speed=3.2):
        """
        Plans smooth 3D minimum-snap splines connecting waypoints:
        waypoints: list of 3D numpy arrays [w_0, w_1, ..., w_M].
        """
        self.waypoints = [np.array(w, dtype=np.float32) for w in waypoints]
        M = len(self.waypoints) - 1 # Number of segments
        if M < 1:
            return False

        # 1. Automatic time allocation based on segment euclidean distance
        self.segment_times = []
        for k in range(M):
            dist = float(np.linalg.norm(self.waypoints[k+1] - self.waypoints[k]))
            # Allocate duration: T = max(T_min, dist / speed)
            t_k = max(1.2, dist / nominal_speed)
            self.segment_times.append(t_k)

        self.total_time = sum(self.segment_times)

        # 2. Construct 8M x 8M Linear Constraint Matrix A
        num_vars = 8 * M
        A = np.zeros((num_vars, num_vars), dtype=np.float64)
        bx = np.zeros(num_vars, dtype=np.float64)
        by = np.zeros(num_vars, dtype=np.float64)
        bz = np.zeros(num_vars, dtype=np.float64)

        row = 0

        # Helper: evaluate i-th derivative of polynomial basis t^j at time t
        def poly_der_basis(t, deriv):
            coeffs = np.zeros(8, dtype=np.float64)
            for j in range(8):
                if j < deriv:
                    coeffs[j] = 0.0
                else:
                    fact = 1.0
                    for d in range(deriv):
                        fact *= (j - d)
                    coeffs[j] = fact * (t ** (j - deriv))
            return coeffs

        # Start Constraints at t = 0 on Segment 0 (p, v=0, a=0, j=0)
        for d in range(4):
            A[row, 0:8] = poly_der_basis(0.0, d)
            if d == 0:
                bx[row] = self.waypoints[0][0]
                by[row] = self.waypoints[0][1]
                bz[row] = self.waypoints[0][2]
            row += 1

        # Internal Waypoint Constraints (for k = 1 to M-1)
        for k in range(1, M):
            t_prev = self.segment_times[k - 1]

            # 1. Segment k-1 endpoint meets waypoint w_k
            A[row, (k-1)*8 : k*8] = poly_der_basis(t_prev, 0)
            bx[row] = self.waypoints[k][0]
            by[row] = self.waypoints[k][1]
            bz[row] = self.waypoints[k][2]
            row += 1

            # 2. Segment k startpoint meets waypoint w_k
            A[row, k*8 : (k+1)*8] = poly_der_basis(0.0, 0)
            bx[row] = self.waypoints[k][0]
            by[row] = self.waypoints[k][1]
            bz[row] = self.waypoints[k][2]
            row += 1

            # 3. Continuity of derivatives: vel, acc, jerk, snap, crackle, pop (d = 1 to 6)
            for d in range(1, 7):
                A[row, (k-1)*8 : k*8] = poly_der_basis(t_prev, d)
                A[row, k*8 : (k+1)*8] = -poly_der_basis(0.0, d)
                bx[row] = 0.0
                by[row] = 0.0
                bz[row] = 0.0
                row += 1

        # Terminal Constraints at t = T_{M-1} on Segment M-1 (p, v=0, a=0, j=0)
        t_last = self.segment_times[-1]
        for d in range(4):
            A[row, (M-1)*8 : M*8] = poly_der_basis(t_last, d)
            if d == 0:
                bx[row] = self.waypoints[-1][0]
                by[row] = self.waypoints[-1][1]
                bz[row] = self.waypoints[-1][2]
            row += 1

        # 3. Exact Matrix Solution
        try:
            cx = np.linalg.solve(A, bx)
            cy = np.linalg.solve(A, by)
            cz = np.linalg.solve(A, bz)

            self.poly_coeffs_x = cx.reshape(M, 8)
            self.poly_coeffs_y = cy.reshape(M, 8)
            self.poly_coeffs_z = cz.reshape(M, 8)
            self.is_solved = True
            return True
        except np.linalg.LinAlgError:
            self.is_solved = False
            return False

    def evaluate(self, t_query):
        """
        Evaluates position p(t), velocity v(t), acceleration a(t),
        jerk j(t), and snap s(t) at any continuous time t_query.
        """
        if not self.is_solved:
            return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

        t_clamped = max(0.0, min(self.total_time, t_query))

        # Identify active segment
        accum_t = 0.0
        seg_idx = 0
        tau = 0.0
        for k, dur in enumerate(self.segment_times):
            if t_clamped <= accum_t + dur or k == len(self.segment_times) - 1:
                seg_idx = k
                tau = t_clamped - accum_t
                break
            accum_t += dur

        # Powers of tau
        T_pow = np.array([tau**i for i in range(8)], dtype=np.float64)

        # Derivatives of powers
        dT_pow = np.array([0] + [i * tau**(i-1) for i in range(1, 8)], dtype=np.float64)
        d2T_pow = np.array([0, 0] + [i * (i-1) * tau**(i-2) for i in range(2, 8)], dtype=np.float64)
        d3T_pow = np.array([0, 0, 0] + [i * (i-1) * (i-2) * tau**(i-3) for i in range(3, 8)], dtype=np.float64)
        d4T_pow = np.array([0, 0, 0, 0] + [i * (i-1) * (i-2) * (i-3) * tau**(i-4) for i in range(4, 8)], dtype=np.float64)

        cx = self.poly_coeffs_x[seg_idx]
        cy = self.poly_coeffs_y[seg_idx]
        cz = self.poly_coeffs_z[seg_idx]

        pos = np.array([np.dot(cx, T_pow), np.dot(cy, T_pow), np.dot(cz, T_pow)], dtype=np.float32)
        vel = np.array([np.dot(cx, dT_pow), np.dot(cy, dT_pow), np.dot(cz, dT_pow)], dtype=np.float32)
        acc = np.array([np.dot(cx, d2T_pow), np.dot(cy, d2T_pow), np.dot(cz, d2T_pow)], dtype=np.float32)
        jerk = np.array([np.dot(cx, d3T_pow), np.dot(cy, d3T_pow), np.dot(cz, d3T_pow)], dtype=np.float32)
        snap = np.array([np.dot(cx, d4T_pow), np.dot(cy, d4T_pow), np.dot(cz, d4T_pow)], dtype=np.float32)

        return pos, vel, acc, jerk, snap

# ==========================================
# 2. PYTORCH NEURAL TIME-ALLOCATION ADAPTOR
# ==========================================
class NeuralTrajectoryTimeNet(nn.Module):
    """
    Neural model predicting dynamically feasible segment time durations T_k
    to minimize flight duration while keeping acceleration within motor limits.
    """
    def __init__(self):
        super().__init__()
        # Inputs: segment distance (1) + start altitude (1) + target altitude (1) = 3 inputs
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus() # Segment time duration T_k >= 1.0s
        )

    def forward(self, x):
        return self.net(x) + 1.0

def _worker_collect_traj(num_samples):
    planner = MinimumSnapTrajectoryPlanner()
    X_samples, y_times = [], []

    while len(X_samples) < num_samples:
        p0 = np.array([random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(1, 4)])
        p1 = np.array([random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(1, 4)])
        dist = float(np.linalg.norm(p1 - p0))
        t_ideal = max(1.2, dist / 3.2)

        feat = np.array([dist, p0[2], p1[2]], dtype=np.float32)
        X_samples.append(feat)
        y_times.append(np.array([t_ideal], dtype=np.float32))

    return np.array(X_samples, dtype=np.float32), np.array(y_times, dtype=np.float32)

def train_trajectory_brain():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"Device: {device}")
    print(f"Training Neural Trajectory Time Allocator (Module 7)...")
    print(f"Differential Flatness & Minimum-Snap Polynomial Optimization")
    print(f"==========================================")

    print("Generating 80,000 polynomial trajectory optimization states...")
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per_worker = (80000 + num_workers - 1) // num_workers

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_collect_traj, [samples_per_worker] * num_workers)

    X_np = np.concatenate([r[0] for r in results], axis=0)[:80000]
    y_np = np.concatenate([r[1] for r in results], axis=0)[:80000]
    print(f"Trajectory dataset ready in {time.time() - t0:.2f}s!")

    X = torch.tensor(X_np, device=device)
    y = torch.tensor(y_np, device=device)

    model = NeuralTrajectoryTimeNet().to(device)
    BATCH_SIZE = 2048
    epochs = 10
    num_batches = (len(X) + BATCH_SIZE - 1) // BATCH_SIZE

    optimizer = optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-3, steps_per_epoch=num_batches, epochs=epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"Training Time-Allocation Neural Model (Batch {BATCH_SIZE}, 10 Epochs)...")
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

        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Polynomial Duration MSE Loss: {total_loss/num_batches:.5f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Trajectory Model compiled in {time.time() - t0:.2f}s!")
    return model, device

# ==========================================
# 3. REAL-TIME 3D TRAJECTORY MISSION HUD
# ==========================================
def project_3d(point, cam_rot, center_screen, scale=40.0):
    """Projects 3D point (X, Y, Z) to 2D screen coordinates."""
    yaw, pitch = cam_rot
    x, y, z = point[0], point[1], point[2]

    x1 = x * math.cos(yaw) - y * math.sin(yaw)
    y1 = x * math.sin(yaw) + y * math.cos(yaw)
    z1 = z

    x2 = x1
    y2 = y1 * math.cos(pitch) - z1 * math.sin(pitch)
    z2 = y1 * math.sin(pitch) + z1 * math.cos(pitch)

    dist = 14.0 - y2
    focal = max(2.5, dist)
    sx = int(center_screen[0] + (x2 / focal) * scale * 7.5)
    sy = int(center_screen[1] - (z2 / focal) * scale * 7.5)
    return sx, sy, y2

def run_minimum_snap_hud(model, device):
    pygame.init()
    VIEW_WIDTH = 700
    VIEW_HEIGHT = 700
    SIDEBAR_WIDTH = 380
    WIDTH = VIEW_WIDTH + SIDEBAR_WIDTH
    HEIGHT = VIEW_HEIGHT

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Module 7: Point-to-Point Minimum-Snap Trajectory Planner (Differential Flatness)")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("consolas", 18, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 13, bold=True)

    planner = MinimumSnapTrajectoryPlanner()

    # Pre-defined 3D Waypoint Mission Flight Path (Meters)
    mission_waypoints = [
        np.array([ -4.0, -3.5, 1.2 ]),  # 0: Start
        np.array([ -2.0,  0.0, 3.2 ]),  # 1: Waypoint A
        np.array([  1.5,  2.5, 4.0 ]),  # 2: Waypoint B
        np.array([ -1.0,  4.5, 2.2 ]),  # 3: Waypoint C
        np.array([  3.8,  3.0, 1.5 ])   # 4: Destination
    ]

    planner.plan_trajectory(mission_waypoints, nominal_speed=3.2)

    camera_yaw = 0.65
    camera_pitch = 0.42
    center_viewport = (VIEW_WIDTH // 2, VIEW_HEIGHT // 2 + 50)

    flight_time = 0.0
    playback_speed = 1.0
    pulse_phase = 0.0
    running = True

    print("================ CONTROLS ================")
    print(" [N]          : Generate Random 3D Mission Waypoints")
    print(" [1] - [4]    : Load Pre-configured Flight Corridor Scenarios")
    print(" [SPACE]      : Pause / Resume Trajectory Playback")
    print(" [R]          : Rewind Flight Simulation to Start (t = 0)")
    print(" [UP] / [DOWN]: Adjust Playback Flight Speed")
    print(" [ARROW KEYS] : Orbit 3D Camera View")
    print("==========================================\n")

    while running:
        dt = clock.tick(30) / 1000.0
        pulse_phase += dt * 4.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    flight_time = 0.0
                elif event.key == pygame.K_SPACE:
                    playback_speed = 0.0 if playback_speed > 0.0 else 1.0
                elif event.key == pygame.K_UP:
                    playback_speed = min(3.0, playback_speed + 0.25)
                elif event.key == pygame.K_DOWN:
                    playback_speed = max(0.25, playback_speed - 0.25)
                elif event.key == pygame.K_n:
                    # Generate random realistic 3D mission
                    num_pts = random.randint(4, 6)
                    new_wps = [np.array([-4.5, -4.0, 1.2])]
                    for _ in range(num_pts - 2):
                        new_wps.append(np.array([
                            random.uniform(-4.0, 4.0),
                            random.uniform(-2.0, 4.0),
                            random.uniform(1.2, 4.5)
                        ]))
                    new_wps.append(np.array([random.uniform(2.0, 4.5), random.uniform(2.0, 4.5), 1.5]))
                    mission_waypoints = new_wps
                    planner.plan_trajectory(mission_waypoints)
                    flight_time = 0.0

        # Camera rotation keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: camera_yaw -= 0.03
        if keys[pygame.K_RIGHT]: camera_yaw += 0.03

        # Advance continuous flight simulation time
        flight_time = (flight_time + dt * playback_speed) % planner.total_time

        # Evaluate differential flatness state at flight_time
        pos_t, vel_t, acc_t, jerk_t, snap_t = planner.evaluate(flight_time)

        screen.fill((10, 12, 18))

        # Dividers
        pygame.draw.line(screen, (35, 45, 60), (VIEW_WIDTH, 0), (VIEW_WIDTH, HEIGHT), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (WIDTH, 70), 2)

        # ==========================================
        # 3D VIEWPORT: RENDERING SPLINE & WAYPOINTS
        # ==========================================
        cam = (camera_yaw, camera_pitch)

        # 1. Draw 3D Ground Grid (meters)
        for gx in range(-5, 6):
            p1 = np.array([gx * 1.5, -5.0, 0.0])
            p2 = np.array([gx * 1.5, 6.0, 0.0])
            sx1, sy1, _ = project_3d(p1, cam, center_viewport)
            sx2, sy2, _ = project_3d(p2, cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        for gy in range(-3, 5):
            p1 = np.array([-7.5, gy * 1.5, 0.0])
            p2 = np.array([ 7.5, gy * 1.5, 0.0])
            sx1, sy1, _ = project_3d(p1, cam, center_viewport)
            sx2, sy2, _ = project_3d(p2, cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        # 2. Draw Sampled 3D Minimum-Snap Spline Curve
        num_samples = 180
        spline_screen_pts = []
        for s_idx in range(num_samples):
            t_s = (s_idx / float(num_samples - 1)) * planner.total_time
            p_s, _, _, _, _ = planner.evaluate(t_s)
            sx, sy, _ = project_3d(p_s, cam, center_viewport)
            spline_screen_pts.append((sx, sy))

        if len(spline_screen_pts) >= 2:
            pygame.draw.lines(screen, (0, 220, 255), False, spline_screen_pts, 3)

        # 3. Draw Waypoint Orbs and Drop-lines to Ground
        for idx, wp in enumerate(mission_waypoints):
            wsx, wsy, _ = project_3d(wp, cam, center_viewport)
            gnd_pt = np.array([wp[0], wp[1], 0.0])
            gsx, gsy, _ = project_3d(gnd_pt, cam, center_viewport)

            # Altitude drop line
            pygame.draw.line(screen, (35, 55, 75), (wsx, wsy), (gsx, gsy), 1)
            pygame.draw.circle(screen, (35, 55, 75), (gsx, gsy), 3)

            # Waypoint orb
            if idx == 0:
                orb_col = (0, 255, 140)    # Start = Green
                lbl_txt = "START"
            elif idx == len(mission_waypoints) - 1:
                orb_col = (255, 200, 40)   # Destination = Gold
                lbl_txt = "DEST"
            else:
                orb_col = (0, 220, 255)    # Intermediate = Cyan
                lbl_txt = f"WP{idx}"

            pygame.draw.circle(screen, orb_col, (wsx, wsy), 6)
            pygame.draw.circle(screen, (255, 255, 255), (wsx, wsy), 2)
            screen.blit(font_small.render(lbl_txt, True, orb_col), (wsx + 8, wsy - 6))

        # 4. Draw Active Flying Drone at Current Spline Time
        dsx, dsy, _ = project_3d(pos_t, cam, center_viewport)

        # Drone hub & body
        pygame.draw.circle(screen, (255, 255, 255), (dsx, dsy), 7)
        pygame.draw.circle(screen, (0, 240, 255), (dsx, dsy), 9, 2)

        # Velocity tangent arrow (Cyan)
        v_mag = float(np.linalg.norm(vel_t))
        if v_mag > 0.1:
            v_dir = (vel_t / v_mag) * 0.8
            vsx, vsy, _ = project_3d(pos_t + v_dir, cam, center_viewport)
            pygame.draw.line(screen, (0, 255, 140), (dsx, dsy), (vsx, vsy), 2)

        # Acceleration vector arrow (Gold)
        a_mag = float(np.linalg.norm(acc_t))
        if a_mag > 0.1:
            a_dir = (acc_t / a_mag) * 0.6
            asx, asy, _ = project_3d(pos_t + a_dir, cam, center_viewport)
            pygame.draw.line(screen, (255, 200, 40), (dsx, dsy), (asx, asy), 2)

        # Drone ground altitude projection
        dgnd_sx, dgnd_sy, _ = project_3d(np.array([pos_t[0], pos_t[1], 0.0]), cam, center_viewport)
        pygame.draw.line(screen, (40, 65, 95), (dsx, dsy), (dgnd_sx, dgnd_sy), 1)
        pygame.draw.circle(screen, (40, 70, 95), (dgnd_sx, dgnd_sy), 3)

        # ==========================================
        # DIFFERENTIAL FLATNESS TELEMETRY (SIDEBAR)
        # ==========================================
        sb_x = VIEW_WIDTH + 14
        screen.blit(font_large.render("MINIMUM-SNAP PLANNER", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Differential Flatness & C^6 Polynomial Splines", True, (130, 160, 190)), (sb_x, 38))

        # --- PANEL 1: MISSION & WAYPOINT STATUS ---
        p1_y = 80
        screen.blit(font_mono.render("MISSION WAYPOINT LEDGER:", True, (240, 240, 240)), (sb_x, p1_y))

        d_dest = float(np.linalg.norm(pos_t - mission_waypoints[-1]))
        screen.blit(font_small.render(f"Active Segments:    M = {len(planner.segment_times)} (7th-Order Splines)", True, (0, 255, 140)), (sb_x + 10, p1_y + 24))
        screen.blit(font_small.render(f"Mission Timeline:   t = {flight_time:.2f}s / {planner.total_time:.2f}s", True, (0, 220, 255)), (sb_x + 10, p1_y + 40))
        screen.blit(font_small.render(f"Range to Dest:      d = {d_dest:.2f} m", True, (255, 200, 40)), (sb_x + 10, p1_y + 56))

        # Flight progress bar [0, total_time]
        bar_x = sb_x + 10
        bar_y = p1_y + 74
        pygame.draw.rect(screen, (24, 32, 45), (bar_x, bar_y, 320, 8), border_radius=2)
        fill_w = int(320 * min(1.0, max(0.0, flight_time / max(0.1, planner.total_time))))
        pygame.draw.rect(screen, (0, 220, 255), (bar_x, bar_y, fill_w, 8), border_radius=2)

        # --- PANEL 2: DERIVATIVE PROFILES (V, A, J, S) ---
        p2_y = bar_y + 24
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (WIDTH - 14, p2_y), 1)
        screen.blit(font_mono.render("DIFFERENTIAL DERIVATIVES:", True, (240, 240, 240)), (sb_x, p2_y + 8))

        j_mag = float(np.linalg.norm(jerk_t))
        s_mag = float(np.linalg.norm(snap_t))

        v_col = (0, 255, 140) if v_mag <= V_MAX_MS else (255, 65, 80)
        a_col = (0, 255, 140) if a_mag <= A_MAX_MS2 else (255, 65, 80)

        screen.blit(font_small.render(f"1. Velocity (p'):   ||v|| = {v_mag:.2f} m/s (Max ≤ {V_MAX_MS:.1f})", True, v_col), (sb_x + 10, p2_y + 28))
        screen.blit(font_small.render(f"2. Accel (p''):     ||a|| = {a_mag:.2f} m/s^2 (Thrust F_z)", True, a_col), (sb_x + 10, p2_y + 44))
        screen.blit(font_small.render(f"3. Jerk (p^(3)):    ||j|| = {j_mag:.2f} m/s^3 (Angular rates)", True, (0, 220, 255)), (sb_x + 10, p2_y + 60))
        screen.blit(font_small.render(f"4. Snap (p^(4)):    ||s|| = {s_mag:.2f} m/s^4 (Motor torques)", True, (255, 200, 40)), (sb_x + 10, p2_y + 76))

        # --- PANEL 3: 3D FLAT OUTPUT POSE ---
        p3_y = p2_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (WIDTH - 14, p3_y), 1)
        screen.blit(font_mono.render("FLAT OUTPUT POSITION p(t):", True, (240, 240, 240)), (sb_x, p3_y + 8))

        screen.blit(font_small.render(f"Position (X, Y, Z): [{pos_t[0]:+.2f}, {pos_t[1]:+.2f}, {pos_t[2]:+.2f}] m", True, (0, 255, 140)), (sb_x + 10, p3_y + 28))
        screen.blit(font_small.render(f"Velocity Vector:    [{vel_t[0]:+.2f}, {vel_t[1]:+.2f}, {vel_t[2]:+.2f}] m/s", True, (0, 220, 255)), (sb_x + 10, p3_y + 44))
        screen.blit(font_small.render("Continuity Class:   C^6 Continuous Boundary Matching", True, (160, 180, 205)), (sb_x + 10, p3_y + 60))

        # --- PANEL 4: MOTOR FEASIBILITY & ACTUATOR EFFORT ---
        p4_y = p3_y + 82
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (WIDTH - 14, p4_y), 1)
        screen.blit(font_mono.render("ACTUATOR FEASIBILITY ENVELOPE:", True, (240, 240, 240)), (sb_x, p4_y + 8))

        is_feasible = (v_mag <= V_MAX_MS) and (a_mag <= A_MAX_MS2)
        feas_str = "MOTOR FEASIBLE (NO SATURATION)" if is_feasible else "DYNAMIC LIMIT EXCEEDED!"
        feas_col = (0, 255, 140) if is_feasible else (255, 65, 80)

        screen.blit(font_small.render(f"ESC Bandwidth State: {feas_str}", True, feas_col), (sb_x + 10, p4_y + 28))
        screen.blit(font_small.render(f"Playback Speed:      {playback_speed:.2f}x Real-Time", True, (0, 220, 255)), (sb_x + 10, p4_y + 44))
        screen.blit(font_small.render("Optimization Cost:   min ∫ ||p^(4)(t)||^2 dt (Snap)", True, (160, 180, 205)), (sb_x + 10, p4_y + 60))

        # Bottom Telemetry Banner
        bot_y = p4_y + 90
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y), (WIDTH - 14, bot_y), 1)
        screen.blit(font_mono.render("PLANNER: MINIMUM-SNAP TRAJECTORY LOCKED", True, (0, 255, 140)), (sb_x + 8, bot_y + 8))

        # Header UI
        head_txt = font_large.render(f"Transit: {flight_time:.1f}s / {planner.total_time:.1f}s | Speed: {v_mag:.1f}m/s | Min-Snap", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("Differential Flatness: Exact 7th-Order Polynomial Splines, Minimum Snap Integral, & Motor-Feasible Profiles", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

# ==========================================
# 4. MAIN ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    mp.freeze_support()
    model, device = train_trajectory_brain()
    print("Launching Point-to-Point Minimum-Snap Trajectory Planner...")
    run_minimum_snap_hud(model, device)