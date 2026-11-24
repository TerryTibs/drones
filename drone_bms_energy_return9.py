"""
=============================================================================
MODULE 9: DYNAMIC ENERGY BUDGETING & POINT-OF-SAFE-RETURN (BMS)
File: drone_bms_energy_return.py
=============================================================================
Physics-Grounded Battery Management System (BMS), Dynamic Point-of-Safe-Return
(PSR), Headwind-Compensated Energy Expenditure, and 1-RC Electrochemical Model.
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
# 0. PHYSICAL CONSTANTS & CELL SPECIFICATIONS
# ==========================================
# Battery Configuration: 6S1P Lithium-Polymer (LiPo) Pack
NUM_CELLS_SERIES = 6
NOMINAL_CELL_CAPACITY_AH = 4.5       # 4500 mAh pack
PACK_NOMINAL_ENERGY_WH = NUM_CELLS_SERIES * 3.7 * NOMINAL_CELL_CAPACITY_AH # ~99.9 Wh
CELL_V_MAX = 4.20                     # 100% SoC
CELL_V_NOMINAL = 3.70                 # Nominal cell voltage
CELL_V_CUTOFF = 3.30                  # 0% SoC (Emergency floor)
CELL_R0_NOMINAL = 0.0085              # Ohms per cell at 25 deg C (8.5 mOhm)
CELL_R1 = 0.0065                      # Ohms (Electrochemical polarization)
CELL_C1 = 2800.0                      # Farads (Double-layer diffusion capacitance)

# Drone Physical Mass & Rotor Aerodynamics
DRONE_MASS_KG = 1.35                  # Base airframe + 6S battery mass
GRAVITY_G = 9.80665
AIR_DENSITY_RHO = 1.225               # kg/m^3
ROTOR_RADIUS_M = 0.127                # 5-inch propellers (0.127 m)
NUM_ROTORS = 4
TOTAL_DISK_AREA = NUM_ROTORS * math.pi * (ROTOR_RADIUS_M ** 2) # ~0.202 m^2
DRAG_AREA_CDA = 0.038                 # Flat-plate parasite drag area (m^2)

# Arrhenius Activation Energy for Electrolyte Derating
ARRHENIUS_EA = 28000.0                # J / mol
GAS_CONST_R = 8.31446                 # J / (mol * K)
T_REF_KELVIN = 298.15                 # 25 deg C

RESERVE_SOC_FLOOR = 0.18              # 18% Emergency Landing Reserve

MODEL_SAVE_PATH = "drone_bms_neural_model.pth"

# ==========================================
# 1. 1-RC EQUIVALENT CIRCUIT BATTERY MODEL
# ==========================================
class LithiumPolymerBatteryECM:
    """
    Simulates a non-linear 6S LiPo pack via an Arrhenius-temperature-derated
    1-RC Equivalent Circuit Model (ECM) with Coulomb counting and voltage sag.
    """
    def __init__(self, init_soc=0.95, temp_ambient_c=20.0):
        self.soc = float(init_soc)
        self.soh = 0.98               # State of Health (98% capacity retention)
        self.capacity_coulombs = NOMINAL_CELL_CAPACITY_AH * 3600.0 * self.soh
        self.v1 = np.zeros(NUM_CELLS_SERIES, dtype=np.float32) # Polarization capacitor voltages
        self.temp_core_c = float(temp_ambient_c)
        self.temp_ambient_c = float(temp_ambient_c)
        self.cell_soc_skew = np.random.normal(0.0, 0.006, NUM_CELLS_SERIES).astype(np.float32)

    def get_cell_ocv(self, cell_soc):
        """
        Non-linear Open Circuit Voltage (OCV) curve fitting a high-discharge LiPo cell:
        Characterized by a steep upper knee (4.2V - 4.0V), a flat plateau (3.85V - 3.65V),
        and a sharp depletion knee below 3.5V.
        """
        s = max(0.001, min(0.999, cell_soc))
        ocv = (CELL_V_CUTOFF +
               0.55 * s +
               0.25 * math.log(s + 0.005) +
               0.15 * math.exp(6.0 * (s - 0.88)) -
               0.08 * math.exp(-35.0 * s))
        return float(np.clip(ocv, 3.20, CELL_V_MAX))

    def step(self, current_amps, dt=0.02):
        """
        Simulates one electrical integration time step:
        - current_amps: Pack draw in Amperes (Positive = discharging).
        """
        # 1. Arrhenius Temperature Derating of Internal Resistance R0
        t_k = self.temp_core_c + 273.15
        r_scale = math.exp((ARRHENIUS_EA / GAS_CONST_R) * (1.0 / t_k - 1.0 / T_REF_KELVIN))
        # Cold temperature sharply increases R0
        r0_effective = CELL_R0_NOMINAL * max(0.85, r_scale)

        # 2. Coulomb Counting SoC Derivation
        delta_q = current_amps * dt
        self.soc -= delta_q / self.capacity_coulombs
        self.soc = max(0.0, min(1.0, self.soc))

        # 3. Individual Cell Voltages & Polarization Diffusion
        cell_voltages = []
        for i in range(NUM_CELLS_SERIES):
            # Individual cell SoC with manufacturing imbalance
            c_soc = max(0.0, min(1.0, self.soc + self.cell_soc_skew[i]))
            ocv_i = self.get_cell_ocv(c_soc)

            # dV1/dt = -V1/(R1*C1) + I/C1
            dv1 = (-(self.v1[i] / (CELL_R1 * CELL_C1)) + (current_amps / CELL_C1)) * dt
            self.v1[i] += dv1

            # V_terminal = OCV - I*R0 - V1
            v_term = ocv_i - current_amps * r0_effective - self.v1[i]
            cell_voltages.append(float(v_term))

        pack_terminal_voltage = float(np.sum(cell_voltages))

        # 4. Thermal Lumped-Capacitance Dynamics (Joule Heating + Dissipation)
        # P_loss = I^2 * R_total
        p_joule = (current_amps ** 2) * (r0_effective * NUM_CELLS_SERIES)
        q_dissipated = 0.35 * (self.temp_core_c - self.temp_ambient_c) # Convective cooling
        c_thermal = 350.0 # Pack thermal mass (J / K)
        self.temp_core_c += ((p_joule - q_dissipated) / c_thermal) * dt

        return pack_terminal_voltage, np.array(cell_voltages, dtype=np.float32), r0_effective

# ==========================================
# 2. AERODYNAMIC ENERGY BUDGET & PSR ENGINE
# ==========================================
class DynamicEnergyBudgetEngine:
    """
    Computes instantaneous mechanical and electrical power required for flight
    as a function of airspeed, climb rate, and environmental headwinds.
    Evaluates Point-of-Safe-Return (PSR) dynamic radius in real time.
    """
    def __init__(self, battery):
        self.battery = battery
        self.esc_efficiency = 0.88     # Motor/ESC electrical-to-mechanical efficiency
        self.avionics_aux_watts = 18.5 # Flight computer, radar, cameras, sensors

    def compute_power_expenditure(self, ground_vel, wind_vec):
        """
        Momentum-theory aerodynamic power consumption model:
        P_electrical = (P_induced + P_profile + P_parasitic + P_climb) / esc_eff + P_avionics
        """
        # Airspeed vector relative to local wind
        air_vel = ground_vel - wind_vec
        v_air = float(np.linalg.norm(air_vel))
        v_climb = float(ground_vel[2])

        # Required thrust to sustain hover + climb
        mass_thrust = DRONE_MASS_KG * GRAVITY_G
        # Body parasite drag: F_drag = 1/2 * rho * CdA * v_air^2
        parasite_drag_n = 0.5 * AIR_DENSITY_RHO * DRAG_AREA_CDA * (v_air ** 2)
        total_thrust_n = math.sqrt(mass_thrust**2 + parasite_drag_n**2)

        # Induced Velocity at hover: v_h = sqrt(T / (2 * rho * A))
        v_h = math.sqrt(total_thrust_n / (2.0 * AIR_DENSITY_RHO * TOTAL_DISK_AREA))

        # Glauert's high-speed induced power approximation:
        # P_induced = T * (-v_air/2 + sqrt((v_air/2)^2 + v_h^2)) (forward flight translational lift)
        v_ind = -0.5 * v_air + math.sqrt(max(0.01, (0.5 * v_air)**2 + v_h**2))
        p_induced = total_thrust_n * max(0.1, v_ind)

        # Profile Drag Power (Blade drag spinning in air)
        p_profile = 28.0 + 0.12 * (v_air ** 2)

        # Parasitic Power (Moving airframe through wind)
        p_parasitic = parasite_drag_n * v_air

        # Climb Power (Work done against gravity)
        p_climb = mass_thrust * max(0.0, v_climb)

        # Total Mechanical Rotor Shaft Power
        p_mechanical = p_induced + p_profile + p_parasitic + p_climb

        # Total Electrical Battery Pack Demand
        p_electrical = (p_mechanical / self.esc_efficiency) + self.avionics_aux_watts
        return p_electrical, p_mechanical, p_induced, p_parasitic

    def compute_safe_return_metrics(self, drone_pos, home_pos, wind_vec, nominal_ground_speed=6.0):
        """
        Calculates exact energy required to fly back to launch base (RTH)
        against the prevailing wind, returning the Point-of-Safe-Return (PSR) status.
        """
        to_home_vec = home_pos - drone_pos
        dist_to_home = float(np.linalg.norm(to_home_vec[:2])) # 2D horizontal range
        if dist_to_home < 0.1:
            return 0.0, 0.0, True, 100.0

        home_dir = to_home_vec[:2] / dist_to_home

        # Ground velocity flying toward base at desired ground speed
        desired_v_g = home_dir * nominal_ground_speed
        v_ground_3d = np.array([desired_v_g[0], desired_v_g[1], 0.0], dtype=np.float32)

        # Compute power demanded during return flight under wind conditions
        p_return_watts, _, _, _ = self.compute_power_expenditure(v_ground_3d, wind_vec)

        # Time to return to base
        t_return_sec = dist_to_home / nominal_ground_speed

        # Total Watt-Hours required for return journey + vertical descent
        wh_transit = (p_return_watts * (t_return_sec / 3600.0))
        wh_landing = (165.0 * (15.0 / 3600.0)) # 15s touchdown hover
        wh_return_total = wh_transit + wh_landing

        # Usable remaining battery capacity above emergency reserve
        soc_usable = max(0.0, self.battery.soc - RESERVE_SOC_FLOOR)
        wh_remaining_usable = (soc_usable * PACK_NOMINAL_ENERGY_WH)

        # Energy Margin
        energy_margin_pct = ((wh_remaining_usable - wh_return_total) / max(0.1, PACK_NOMINAL_ENERGY_WH)) * 100.0
        can_return_safely = (wh_remaining_usable >= wh_return_total)

        # Maximum Dynamic Safe Flight Radius (PSR Ring Radius)
        # R_max = (wh_usable / P_return) * v_ground * 3600
        safe_radius_m = (wh_remaining_usable / max(1.0, p_return_watts)) * nominal_ground_speed * 3600.0

        return wh_return_total, wh_remaining_usable, can_return_safely, safe_radius_m

# ==========================================
# 3. PYTORCH NEURAL BMS ENERGY ESTIMATOR
# ==========================================
class NeuralBMSEnergyNet(nn.Module):
    """
    Real-time neural observer estimating True State-of-Charge (SoC) and
    instantaneous power drain from voltage sag, current, and core temperature.
    """
    def __init__(self):
        super().__init__()
        # Input features: Pack Voltage (1) + Pack Current (1) + Core Temp (1) + Cell Min (1) + Cell Max (1) = 5
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # [Estimated_SoC (0-1), Estimated_Power_Watts]
        )

    def forward(self, x):
        out = self.net(x)
        soc_est = torch.sigmoid(out[:, 0:1]) # SoC bounded in [0, 1]
        power_est = torch.relu(out[:, 1:2]) * 800.0 # Watts >= 0
        return torch.cat([soc_est, power_est], dim=1)

def _worker_collect_bms(num_samples):
    bms = LithiumPolymerBatteryECM(init_soc=random.uniform(0.3, 0.98), temp_ambient_c=random.uniform(-5.0, 35.0))
    budget = DynamicEnergyBudgetEngine(bms)
    X_samples, y_targets = [], []

    while len(X_samples) < num_samples:
        speed = random.uniform(0.0, 14.0)
        v_g = np.array([speed, 0.0, random.uniform(-1.0, 1.0)], dtype=np.float32)
        wind = np.array([random.uniform(-6.0, 6.0), random.uniform(-6.0, 6.0), 0.0], dtype=np.float32)

        p_req, _, _, _ = budget.compute_power_expenditure(v_g, wind)
        v_pack_approx = max(19.0, bms.soc * 25.2)
        i_draw = p_req / v_pack_approx

        v_pack, cell_v, _ = bms.step(i_draw, dt=0.05)

        # Feature vector (5 dims)
        feat = np.array([
            v_pack / 25.2,
            i_draw / 35.0,
            bms.temp_core_c / 60.0,
            float(np.min(cell_v)) / 4.2,
            float(np.max(cell_v)) / 4.2
        ], dtype=np.float32)

        target = np.array([bms.soc, p_req], dtype=np.float32)

        X_samples.append(feat)
        y_targets.append(target)

    return np.array(X_samples, dtype=np.float32), np.array(y_targets, dtype=np.float32)

def train_bms_brain():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"Device: {device}")
    print(f"Training Neural Battery Observer (Module 9)...")
    print(f"1-RC Thevenin ECM & Aerodynamic Power Estimation")
    print(f"==========================================")

    print("Harvesting 80,000 electrochemical discharge states across CPU cores...")
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per_worker = (80000 + num_workers - 1) // num_workers

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_collect_bms, [samples_per_worker] * num_workers)

    X_np = np.concatenate([r[0] for r in results], axis=0)[:80000]
    y_np = np.concatenate([r[1] for r in results], axis=0)[:80000]
    print(f"BMS dataset generated in {time.time() - t0:.2f}s!")

    X = torch.tensor(X_np, device=device)
    y = torch.tensor(y_np, device=device)

    model = NeuralBMSEnergyNet().to(device)
    BATCH_SIZE = 2048
    epochs = 10
    num_batches = (len(X) + BATCH_SIZE - 1) // BATCH_SIZE

    optimizer = optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-3, steps_per_epoch=num_batches, epochs=epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"Optimizing Neural BMS Weights (Batch {BATCH_SIZE}, 10 Epochs)...")
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
                loss = criterion(pred[:, 0], by[:, 0]) + 0.001 * criterion(pred[:, 1], by[:, 1])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1:02d}/{epochs:02d} - BMS Observer MSE Loss: {total_loss/num_batches:.5f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"BMS Observer Model compiled in {time.time() - t0:.2f}s!")
    return model, device

# ==========================================
# 4. REAL-TIME 3D ENERGY & PSR COCKPIT HUD
# ==========================================
def project_3d(point, cam_rot, center_screen, scale=44.0):
    """Projects 3D world coordinates onto 2D screen."""
    yaw, pitch = cam_rot
    x, y, z = point[0], point[1], point[2]

    x1 = x * math.cos(yaw) - y * math.sin(yaw)
    y1 = x * math.sin(yaw) + y * math.cos(yaw)
    z1 = z

    x2 = x1
    y2 = y1 * math.cos(pitch) - z1 * math.sin(pitch)
    z2 = y1 * math.sin(pitch) + z1 * math.cos(pitch)

    dist = 16.0 - y2
    focal = max(2.5, dist)
    sx = int(center_screen[0] + (x2 / focal) * scale * 7.0)
    sy = int(center_screen[1] - (z2 / focal) * scale * 7.0)
    return sx, sy, y2

def run_bms_energy_hud(model, device):
    pygame.init()
    VIEW_WIDTH = 700
    VIEW_HEIGHT = 700
    SIDEBAR_WIDTH = 380
    WIDTH = VIEW_WIDTH + SIDEBAR_WIDTH
    HEIGHT = VIEW_HEIGHT

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Module 9: Dynamic Energy Budgeting & Point-of-Safe-Return (BMS Engine)")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("consolas", 18, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 13, bold=True)

    battery = LithiumPolymerBatteryECM(init_soc=0.88, temp_ambient_c=18.0)
    budget = DynamicEnergyBudgetEngine(battery)
    model.eval()

    # Home Launch Base Coordinates
    home_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Drone Physical Flight State
    drone_pos = np.array([0.0, 0.0, 2.5], dtype=np.float32)
    drone_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    drone_yaw_deg = 0.0

    # Atmospheric Wind Vector [Wx, Wy, Wz] (m/s)
    wind_vector = np.array([3.5, 0.0, 0.0], dtype=np.float32) # Prevailing easterly headwind

    camera_yaw = 0.62
    camera_pitch = 0.44
    center_viewport = (VIEW_WIDTH // 2, VIEW_HEIGHT // 2 + 50)

    flight_trail = deque(maxlen=70)
    current_history = deque(maxlen=60)
    auto_rth_active = False
    running = True

    print("================ BMS ENERGY CONTROLS ================")
    print(" [W] / [S]      : Fly Forward / Backward")
    print(" [A] / [D]      : Strafe Left / Right")
    print(" [I] / [K]      : Climb / Descend")
    print(" [G]            : Toggle High Headwind (+8.5 m/s North)")
    print(" [T]            : Inject Freezing Ambient Temp (-10°C)")
    print(" [H]            : Manual Return-to-Home (RTH) Abort")
    print(" [R]            : Recharge Battery to 100% SoC")
    print(" [ARROW KEYS]   : Orbit 3D Camera View")
    print("=====================================================\n")

    while running:
        dt = clock.tick(30) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    # Toggle severe headwind
                    if wind_vector[1] > 2.0:
                        wind_vector = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    else:
                        wind_vector = np.array([0.0, 8.5, 0.0], dtype=np.float32)
                elif event.key == pygame.K_t:
                    # Cycle freezing temperatures
                    battery.temp_ambient_c = -12.0 if battery.temp_ambient_c > 0.0 else 22.0
                elif event.key == pygame.K_h:
                    auto_rth_active = not auto_rth_active
                elif event.key == pygame.K_r:
                    battery = LithiumPolymerBatteryECM(init_soc=0.98, temp_ambient_c=battery.temp_ambient_c)
                    budget.battery = battery
                    auto_rth_active = False

        keys = pygame.key.get_pressed()
        yaw_rad = math.radians(drone_yaw_deg)
        fwd_vec = np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0], dtype=np.float32)
        right_vec = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=np.float32)

        # 1. Autonomous Return-to-Home (RTH) Controller or Manual Pilot
        if auto_rth_active:
            to_home = home_pos - drone_pos
            dist_h = float(np.linalg.norm(to_home[:2]))
            if dist_h > 0.3:
                rth_dir = to_home[:2] / dist_h
                target_vel = np.array([rth_dir[0] * 5.5, rth_dir[1] * 5.5, 0.0], dtype=np.float32)
            else:
                # Land
                target_vel = np.array([0.0, 0.0, -0.6], dtype=np.float32)
                if drone_pos[2] <= 0.2:
                    target_vel = np.zeros(3, dtype=np.float32)
            drone_vel += (target_vel - drone_vel) * 0.15
        else:
            accel = np.zeros(3, dtype=np.float32)
            if keys[pygame.K_w]: accel += fwd_vec * 6.0
            if keys[pygame.K_s]: accel -= fwd_vec * 6.0
            if keys[pygame.K_d]: accel += right_vec * 6.0
            if keys[pygame.K_a]: accel -= right_vec * 6.0
            if keys[pygame.K_i]: accel[2] += 4.0
            if keys[pygame.K_k]: accel[2] -= 4.0
            if keys[pygame.K_j]: drone_yaw_deg = (drone_yaw_deg - 2.5) % 360.0
            if keys[pygame.K_l]: drone_yaw_deg = (drone_yaw_deg + 2.5) % 360.0

            drone_vel += (accel - 0.70 * drone_vel) * dt

        drone_pos += drone_vel * dt
        drone_pos[2] = max(0.0, min(8.0, drone_pos[2]))
        flight_trail.append(drone_pos.copy())

        # Camera rotation
        if keys[pygame.K_LEFT]: camera_yaw -= 0.03
        if keys[pygame.K_RIGHT]: camera_yaw += 0.03
        if keys[pygame.K_UP]: camera_pitch = max(0.1, camera_pitch - 0.02)
        if keys[pygame.K_DOWN]: camera_pitch = min(1.2, camera_pitch + 0.02)

        # 2. Physics Energy Expenditure Calculation
        p_elec, p_mech, p_ind, p_par = budget.compute_power_expenditure(drone_vel, wind_vector)

        # Draw battery pack current: I = P / V
        v_est_pack = max(18.0, float(np.sum(battery.get_cell_ocv(battery.soc) for _ in range(NUM_CELLS_SERIES))))
        pack_current = p_elec / max(18.0, v_est_pack)

        # Step Battery ECM
        v_pack_term, cell_voltages, r0_derated = battery.step(pack_current, dt=dt)
        current_history.append(pack_current)

        # 3. Dynamic Point-of-Safe-Return (PSR) Evaluation
        wh_needed, wh_avail, is_safe_to_continue, psr_radius_m = budget.compute_safe_return_metrics(
            drone_pos, home_pos, wind_vector, nominal_ground_speed=5.5
        )

        # Automated Safety Fail-Safe: If energy depleted below RTH reserve, force return!
        if not is_safe_to_continue and not auto_rth_active and drone_pos[2] > 0.3:
            auto_rth_active = True
            print(">>> [SAFETY INTERLOCK] PSR BOUNDARY REACHED: INITIATING AUTONOMOUS RETURN-TO-HOME <<<")

        # 4. Neural BMS Inference
        feat_in = np.array([
            v_pack_term / 25.2,
            pack_current / 35.0,
            battery.temp_core_c / 60.0,
            float(np.min(cell_voltages)) / 4.2,
            float(np.max(cell_voltages)) / 4.2
        ], dtype=np.float32)

        with torch.no_grad():
            feat_t = torch.tensor(feat_in, device=device).unsqueeze(0)
            nn_out = model(feat_t)[0]
            nn_soc_est = float(nn_out[0].item())
            nn_power_est = float(nn_out[1].item())

        screen.fill((10, 13, 19))

        # Dividers
        pygame.draw.line(screen, (35, 45, 60), (VIEW_WIDTH, 0), (VIEW_WIDTH, HEIGHT), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (WIDTH, 70), 2)

        # ==========================================
        # 3D VIEWPORT: FLIGHT TRAIL & DYNAMIC PSR RING
        # ==========================================
        cam = (camera_yaw, camera_pitch)

        # 1. Ground Grid
        for gx in range(-6, 7):
            p1 = np.array([gx * 2.0, -8.0, 0.0])
            p2 = np.array([gx * 2.0, 14.0, 0.0])
            sx1, sy1, _ = project_3d(p1, cam, center_viewport)
            sx2, sy2, _ = project_3d(p2, cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        for gy in range(-4, 8):
            p1 = np.array([-12.0, gy * 2.0, 0.0])
            p2 = np.array([ 12.0, gy * 2.0, 0.0])
            sx1, sy1, _ = project_3d(p1, cam, center_viewport)
            sx2, sy2, _ = project_3d(p2, cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        # 2. Draw Launch Base (Home Marker)
        hx, hy, _ = project_3d(home_pos, cam, center_viewport)
        pygame.draw.circle(screen, (0, 255, 140), (hx, hy), 8)
        pygame.draw.circle(screen, (255, 255, 255), (hx, hy), 3)
        screen.blit(font_small.render("LAUNCH HOME", True, (0, 255, 140)), (hx + 10, hy - 6))

        # 3. Dynamic Point-of-Safe-Return (PSR) Abort Perimeter Ring
        # Sample ellipse/ring around home based on safe flight radius
        psr_pts = []
        r_clamped = min(18.0, psr_radius_m * 0.15) # Scaled for visual representation
        for a_deg in range(0, 360, 15):
            rad = math.radians(a_deg)
            # Headwind warps the safe return boundary into an asymmetric oval
            wind_warp = 1.0 - (np.dot(np.array([math.sin(rad), math.cos(rad)]), wind_vector[:2]) / 18.0)
            px = home_pos[0] + math.sin(rad) * r_clamped * wind_warp
            py = home_pos[1] + math.cos(rad) * r_clamped * wind_warp
            sx, sy, _ = project_3d(np.array([px, py, 0.0]), cam, center_viewport)
            psr_pts.append((sx, sy))

        psr_col = (0, 255, 140) if is_safe_to_continue else (255, 60, 80)
        if len(psr_pts) >= 3:
            pygame.draw.lines(screen, psr_col, True, psr_pts, 2)

        # 4. Flight Trail
        if len(flight_trail) >= 2:
            t_pts = [project_3d(p, cam, center_viewport)[:2] for p in flight_trail]
            pygame.draw.lines(screen, (0, 220, 255), False, t_pts, 1)

        # 5. Drone Airframe
        dx, dy, _ = project_3d(drone_pos, cam, center_viewport)
        pygame.draw.circle(screen, (0, 240, 255), (dx, dy), 8)
        pygame.draw.circle(screen, (255, 255, 255), (dx, dy), 3)

        # Airspeed / Groundspeed Vector
        v_scale = 0.4
        g_end = drone_pos + drone_vel * v_scale
        gx, gy, _ = project_3d(g_end, cam, center_viewport)
        pygame.draw.line(screen, (0, 255, 140), (dx, dy), (gx, gy), 2)

        # Drop-line to ground
        gnd_pt = np.array([drone_pos[0], drone_pos[1], 0.0])
        gsx, gsy, _ = project_3d(gnd_pt, cam, center_viewport)
        pygame.draw.line(screen, (35, 55, 75), (dx, dy), (gsx, gsy), 1)

        # ==========================================
        # 6S CELL VOLTAGE GAUGES (LOWER VIEWPORT)
        # ==========================================
        gauge_box_x = 25
        gauge_box_y = VIEW_HEIGHT - 135
        gauge_box_w = 420
        gauge_box_h = 115
        pygame.draw.rect(screen, (14, 20, 28), (gauge_box_x, gauge_box_y, gauge_box_w, gauge_box_h), border_radius=6)
        pygame.draw.rect(screen, (35, 50, 70), (gauge_box_x, gauge_box_y, gauge_box_w, gauge_box_h), 2)

        screen.blit(font_mono.render("6S LI-PO INDIVIDUAL CELL TELEMETRY (ECM):", True, (240, 240, 240)), (gauge_box_x + 12, gauge_box_y + 8))

        cell_w = 58
        for c_i in range(NUM_CELLS_SERIES):
            cx = gauge_box_x + 15 + c_i * 66
            cy = gauge_box_y + 32
            v_c = cell_voltages[c_i]

            # Cell bar height [3.20V to 4.20V]
            norm_v = min(1.0, max(0.0, (v_c - 3.20) / 1.00))
            bar_h = int(norm_v * 50)

            c_col = (0, 255, 140) if v_c > 3.65 else ((255, 180, 40) if v_c > 3.40 else (255, 60, 80))

            # Outer cell housing
            pygame.draw.rect(screen, (22, 30, 42), (cx, cy, cell_w, 55), border_radius=3)
            # Active voltage fill
            pygame.draw.rect(screen, c_col, (cx + 3, cy + 55 - bar_h, cell_w - 6, bar_h), border_radius=2)
            # Text label
            screen.blit(font_small.render(f"C{c_i+1}", True, (160, 180, 205)), (cx + 18, cy + 58))
            screen.blit(font_small.render(f"{v_c:.2f}V", True, (240, 240, 240)), (cx + 12, cy + 40))

        # ==========================================
        # ENERGY & BMS TELEMETRY (SIDEBAR)
        # ==========================================
        sb_x = VIEW_WIDTH + 14
        screen.blit(font_large.render("BMS & SAFE-RETURN (PSR)", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("1-RC Thevenin ECM + Aerodynamic Power Modeling", True, (130, 160, 190)), (sb_x, 38))

        # --- PANEL 1: PACK STATE OF CHARGE & HEALTH ---
        p1_y = 80
        screen.blit(font_mono.render("PACK STATE OF CHARGE (SoC):", True, (240, 240, 240)), (sb_x, p1_y))

        soc_pct = battery.soc * 100.0
        soc_col = (0, 255, 140) if soc_pct > 35.0 else ((255, 180, 40) if soc_pct > 20.0 else (255, 60, 80))
        screen.blit(font_small.render(f"Battery SoC:        {soc_pct:.1f}% (Neural: {nn_soc_est*100.0:.1f}%)", True, soc_col), (sb_x + 10, p1_y + 24))
        screen.blit(font_small.render(f"Pack Terminal Volt: V_pack = {v_pack_term:.2f} V (6S)", True, (0, 220, 255)), (sb_x + 10, p1_y + 40))
        screen.blit(font_small.render(f"Current Draw:       I_batt = {pack_current:.2f} A", True, (255, 200, 40)), (sb_x + 10, p1_y + 56))

        # SoC Bar Gauge with 18% Emergency Reserve Line
        bar_x = sb_x + 10
        bar_y = p1_y + 74
        bar_w = 320
        pygame.draw.rect(screen, (24, 32, 45), (bar_x, bar_y, bar_w, 9), border_radius=2)
        fill_w = int(bar_w * battery.soc)
        pygame.draw.rect(screen, soc_col, (bar_x, bar_y, fill_w, 9), border_radius=2)

        # Reserve marker
        res_x = bar_x + int(bar_w * RESERVE_SOC_FLOOR)
        pygame.draw.line(screen, (255, 60, 80), (res_x, bar_y - 2), (res_x, bar_y + 11), 2)
        screen.blit(font_small.render("18% RTH Reserve", True, (255, 60, 80)), (res_x - 30, bar_y + 12))

        # --- PANEL 2: POINT-OF-SAFE-RETURN (PSR) METRICS ---
        p2_y = bar_y + 28
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (WIDTH - 14, p2_y), 1)
        screen.blit(font_mono.render("POINT-OF-SAFE-RETURN (PSR):", True, (240, 240, 240)), (sb_x, p2_y + 8))

        dist_to_home = float(np.linalg.norm((home_pos - drone_pos)[:2]))
        psr_status = "SAFE AIRSPACE (WITHIN PSR)" if is_safe_to_continue else "CRITICAL: RTH REQUIRED!"
        psr_stat_col = (0, 255, 140) if is_safe_to_continue else (255, 60, 80)

        screen.blit(font_small.render(f"Range to Home:      d = {dist_to_home:.1f} m", True, (0, 220, 255)), (sb_x + 10, p2_y + 28))
        screen.blit(font_small.render(f"Return Energy Req:  E_rth = {wh_needed:.2f} Wh", True, (255, 200, 40)), (sb_x + 10, p2_y + 44))
        screen.blit(font_small.render(f"Remaining Usable:   E_use = {wh_avail:.2f} Wh", True, (0, 255, 140)), (sb_x + 10, p2_y + 60))
        screen.blit(font_small.render(f"Flight Status:      {psr_status}", True, psr_stat_col), (sb_x + 10, p2_y + 76))

        # --- PANEL 3: POWER DISTRIBUTION BREAKDOWN ---
        p3_y = p2_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (WIDTH - 14, p3_y), 1)
        screen.blit(font_mono.render("POWER LOAD EXPENDITURE (W):", True, (240, 240, 240)), (sb_x, p3_y + 8))

        screen.blit(font_small.render(f"Total Electrical:   P_total = {p_elec:.1f} W (NN: {nn_power_est:.1f}W)", True, (0, 255, 140)), (sb_x + 10, p3_y + 28))
        screen.blit(font_small.render(f"Induced Rotor Power:P_ind   = {p_ind:.1f} W (Hover lift)", True, (0, 220, 255)), (sb_x + 10, p3_y + 44))
        screen.blit(font_small.render(f"Parasite Airframe:  P_par   = {p_par:.1f} W (Headwind drag)", True, (255, 180, 40)), (sb_x + 10, p3_y + 60))
        screen.blit(font_small.render(f"Avionics & Sensors: P_aux   = {budget.avionics_aux_watts:.1f} W (Static load)", True, (160, 180, 205)), (sb_x + 10, p3_y + 76))

        # --- PANEL 4: ARRHENIUS THERMAL DERATING ---
        p4_y = p3_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (WIDTH - 14, p4_y), 1)
        screen.blit(font_mono.render("THERMAL & INTERNAL RESISTANCE:", True, (240, 240, 240)), (sb_x, p4_y + 8))

        t_amb = battery.temp_ambient_c
        t_core = battery.temp_core_c
        amb_col = (255, 60, 80) if t_amb < 0.0 else (0, 255, 140)

        screen.blit(font_small.render(f"Ambient Temperature:T_amb   = {t_amb:+.1f} °C ([T] to toggle)", True, amb_col), (sb_x + 10, p4_y + 28))
        screen.blit(font_small.render(f"Battery Core Temp:  T_core  = {t_core:+.1f} °C (Joule heated)", True, (0, 220, 255)), (sb_x + 10, p4_y + 44))
        screen.blit(font_small.render(f"Internal Resis R0:  R_cell  = {r0_derated*1000.0:.1f} mΩ (Arrhenius)", True, (255, 200, 40)), (sb_x + 10, p4_y + 60))

        # Wind Vector Telemetry
        wind_spd = float(np.linalg.norm(wind_vector))
        screen.blit(font_small.render(f"Wind Condition:     |W| = {wind_spd*3.6:.1f} km/h ([G] to toggle)", True, (160, 180, 205)), (sb_x + 10, p4_y + 76))

        # Bottom Telemetry Banner
        bot_y = p4_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y), (WIDTH - 14, bot_y), 1)
        if auto_rth_active:
            banner_txt = "EMERGENCY: AUTONOMOUS RTH TRANSIT"
            banner_col = (255, 60, 80)
        elif is_safe_to_continue:
            banner_txt = "MISSION ENERGY BUDGET: NOMINAL"
            banner_col = (0, 255, 140)
        else:
            banner_txt = "CAUTION: APPROACHING RETURN THRESHOLD"
            banner_col = (255, 180, 40)
        screen.blit(font_mono.render(banner_txt, True, banner_col), (sb_x + 8, bot_y + 8))

        # Header UI
        head_txt = font_large.render(f"SoC: {soc_pct:.1f}% | Power: {p_elec:.0f}W | Range: {dist_to_home:.1f}m | BMS PSR", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("Dynamic Point-of-Safe-Return (PSR), Arrhenius Cold Derating, 1-RC ECM, & Glauert Induced Power", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

# ==========================================
# 5. MAIN ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    mp.freeze_support()
    model, device = train_bms_brain()
    print("Launching Dynamic Energy Budgeting & Point-of-Safe-Return Simulator...")
    run_bms_energy_hud(model, device)