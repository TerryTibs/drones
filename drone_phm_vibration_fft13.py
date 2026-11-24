"""
=============================================================================
MODULE 13: PREDICTIVE HEALTH MAINTENANCE (PHM) & VIBRATION FFT
File: drone_phm_vibration_fft.py
=============================================================================
High-Frequency (1000 Hz) Accelerometer FFT Spectral Analysis, Harmonic
Decomposition (1x Unbalance, Blade Passing Frequency, Bearing BPFO/BPFI),
ISO 10816 Severity Gauging, and Neural Mechanical Fault Diagnostics.
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
# 0. PHYSICAL VIBRATION & SPECTRAL CONSTANTS
# ==========================================
SAMPLE_RATE_HZ = 1000.0              # 1 kHz sampling rate
FFT_WINDOW_SIZE = 256                # 256 samples per spectral frame (~256 ms)
NYQUIST_FREQ_HZ = SAMPLE_RATE_HZ / 2.0 # 500 Hz Nyquist limit
FREQ_BIN_RESOLUTION = SAMPLE_RATE_HZ / FFT_WINDOW_SIZE # ~3.9 Hz/bin

# Nominal Motor Operational Envelope
MOTOR_NOMINAL_RPM = 5100.0           # ~85 Hz shaft rotation frequency (1x)
MOTOR_NUM_BLADES = 2                 # 2-blade carbon props -> BPF = 170 Hz
BEARING_BPFO_MULT = 3.65             # Ball Pass Frequency Outer Race multiplier (~310 Hz)
AIRFRAME_RESONANCE_HZ = 42.0         # Structural arm natural resonance (Hz)

# ISO 10816 Vibration Velocity Severity Thresholds (mm/s RMS)
ISO_SEV_GOOD = 1.8                   # Class A: Unrestricted flight
ISO_SEV_ACCEPTABLE = 4.5             # Class B: Marginal (Scheduled maintenance)
ISO_SEV_UNSATISFACTORY = 7.1         # Class C: Restricted envelope (Speed limited)
# Above 7.1 mm/s = Class D: Danger (Imminent structural/motor failure -> RTH)

MODEL_SAVE_PATH = "drone_phm_vibration_model.pth"

# ==========================================
# 1. 1000 HZ VIBRATION SIGNAL SYNTHESIZER
# ==========================================
class MechanicalVibrationSimulator:
    """
    Simulates high-rate tri-axial accelerometer signals corrupted by:
    - Fundamental rotational shaft unbalance (1x)
    - Aerodynamic Blade Passing Frequency (BPF = 2x)
    - Ball bearing race micro-impacts (BPFO / BPFI transients)
    - Structural airframe resonance flutter (loose motor mount)
    - Sensor background pink/white noise
    """
    def __init__(self):
        self.rpm = MOTOR_NOMINAL_RPM
        self.fault_mode = "HEALTHY" # HEALTHY, CHIPPED_PROP, BEARING_WEAR, LOOSE_ARM
        self.time = 0.0
        self.sample_buffer = deque(maxlen=FFT_WINDOW_SIZE)
        # Pre-fill buffer with nominal baseline noise
        for _ in range(FFT_WINDOW_SIZE):
            self.sample_buffer.append(float(np.random.normal(0.0, 0.25)))

    def step(self, dt=0.001):
        """Generates one accelerometer sample at 1000 Hz."""
        self.time += dt
        f0 = self.rpm / 60.0 # Fundamental shaft rotational frequency (~85 Hz)

        # Baseline healthy airframe vibration: low-amplitude 1x and 2x
        vib = 0.65 * math.sin(2.0 * math.pi * f0 * self.time)
        vib += 0.35 * math.sin(2.0 * math.pi * (2.0 * f0) * self.time)

        # -------------------------------------------------------------
        # Inject Mechanical Fault Profiles
        # -------------------------------------------------------------
        if self.fault_mode == "CHIPPED_PROP":
            # Violent 1x fundamental unbalance due to missing mass on blade tip
            unbalance_amp = 8.5
            vib += unbalance_amp * math.sin(2.0 * math.pi * f0 * self.time)
            # Add aerodynamic flutter harmonic (2x BPF)
            vib += 3.2 * math.sin(2.0 * math.pi * (2.0 * f0) * self.time + 0.4)

        elif self.fault_mode == "BEARING_WEAR":
            # High-frequency BPFO impacts (~310 Hz) with impulsive ringing
            bpfo_freq = f0 * BEARING_BPFO_MULT
            impact_carrier = math.sin(2.0 * math.pi * bpfo_freq * self.time)
            # Modulate with rotational frequency envelope
            modulator = max(0.0, math.sin(2.0 * math.pi * f0 * self.time)) ** 4
            vib += 7.2 * impact_carrier * modulator
            # High-frequency bearing raceway noise
            vib += float(np.random.normal(0.0, 1.8))

        elif self.fault_mode == "LOOSE_ARM":
            # Low-frequency structural resonance around 42 Hz
            res_freq = AIRFRAME_RESONANCE_HZ + math.sin(self.time * 2.0) * 3.0
            vib += 6.8 * math.sin(2.0 * math.pi * res_freq * self.time)
            # Sub-harmonic rattling
            vib += 2.5 * math.sin(2.0 * math.pi * (res_freq * 0.5) * self.time)

        # Ambient sensor white noise
        vib += float(np.random.normal(0.0, 0.45))
        self.sample_buffer.append(vib)
        return vib

    def compute_fft(self):
        """
        Applies a Hann window to the rolling sample buffer, computes
        the real Fast Fourier Transform (rFFT), and calculates ISO severity.
        """
        signal = np.array(self.sample_buffer, dtype=np.float32)
        # Apply Hann window to suppress spectral leakage
        hann_window = np.hanning(len(signal))
        windowed_signal = signal * hann_window

        # Compute Fast Fourier Transform
        fft_complex = np.fft.rfft(windowed_signal)
        psd = (np.abs(fft_complex) ** 2) / (FFT_WINDOW_SIZE / 2.0) # Power Spectral Density
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / SAMPLE_RATE_HZ)

        # Vibration Velocity RMS (ISO 10816 Metric) via spectral integration
        # v(omega) = a(omega) / omega
        omega = 2.0 * math.pi * np.maximum(freqs, 1.0)
        vel_spectrum = np.abs(fft_complex) / omega
        v_rms_mms = float(np.sqrt(np.mean(vel_spectrum ** 2)) * 1000.0) # Convert to mm/s

        # Spectral Kurtosis (impulsiveness metric: healthy ~3.0, damaged bearing > 6.0)
        mean_val = np.mean(signal)
        std_val = np.std(signal) + 1e-5
        kurtosis = float(np.mean(((signal - mean_val) / std_val) ** 4))

        # Peak dominant frequency
        peak_idx = int(np.argmax(psd[1:])) + 1
        peak_freq = float(freqs[peak_idx])

        return freqs, psd, v_rms_mms, kurtosis, peak_freq

# ==========================================
# 2. PYTORCH NEURAL SPECTRAL FAULT CLASSIFIER
# ==========================================
class NeuralVibrationClassifierNet(nn.Module):
    """
    1D Convolutional Neural Network operating on the 129-bin FFT Power
    Spectral Density (PSD) vector to classify airframe health in real time.
    """
    def __init__(self):
        super().__init__()
        # Input: 129-bin power spectral density vector
        self.conv_stack = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2), # 129 -> 64
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2), # 64 -> 32
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8), # 64 x 8 = 512
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 4) # [HEALTHY, CHIPPED_PROP, BEARING_WEAR, LOOSE_ARM]
        )

    def forward(self, x):
        feat = self.conv_stack(x)
        return self.classifier(feat)

def _worker_collect_vibrations(num_samples):
    sim = MechanicalVibrationSimulator()
    X_samples, y_labels = [], []
    fault_map = {"HEALTHY": 0, "CHIPPED_PROP": 1, "BEARING_WEAR": 2, "LOOSE_ARM": 3}

    while len(X_samples) < num_samples:
        mode = random.choice(["HEALTHY", "CHIPPED_PROP", "BEARING_WEAR", "LOOSE_ARM"])
        sim.fault_mode = mode
        sim.rpm = random.uniform(4600.0, 5600.0)

        # Run 256 samples to fill buffer
        for _ in range(FFT_WINDOW_SIZE):
            sim.step(dt=0.001)

        _, psd, _, _, _ = sim.compute_fft()
        # Normalize PSD feature vector (129 elements)
        norm_psd = np.log1p(psd) / 8.0

        X_samples.append(norm_psd.astype(np.float32))
        y_labels.append(fault_map[mode])

    return np.array(X_samples, dtype=np.float32), np.array(y_labels, dtype=np.int64)

def train_phm_brain():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"Device: {device}")
    print(f"Training Predictive Health Maintenance (PHM) Brain (Module 13)...")
    print(f"1000 Hz Accelerometer FFT Spectral Analysis & ISO 10816 Classifier")
    print(f"==========================================")

    print("Harvesting 80,000 multi-harmonic vibration spectra across CPU cores...")
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per_worker = (80000 + num_workers - 1) // num_workers

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_collect_vibrations, [samples_per_worker] * num_workers)

    X_np = np.concatenate([r[0] for r in results], axis=0)[:80000]
    y_np = np.concatenate([r[1] for r in results], axis=0)[:80000]
    print(f"Vibration spectral dataset ready in {time.time() - t0:.2f}s!")

    # Shape: (N, 1, 129)
    X = torch.tensor(X_np, device=device).unsqueeze(1)
    y = torch.tensor(y_np, device=device)

    model = NeuralVibrationClassifierNet().to(device)
    BATCH_SIZE = 2048
    epochs = 10
    num_batches = (len(X) + BATCH_SIZE - 1) // BATCH_SIZE

    optimizer = optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-3, steps_per_epoch=num_batches, epochs=epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"Optimizing Spectral CNN Weights (Batch {BATCH_SIZE}, 10 Epochs)...")
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
        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Spectral Diagnostics Accuracy: {acc:.2f}% - Loss: {total_loss/num_batches:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"PHM Spectral Model compiled in {time.time() - t0:.2f}s!")
    return model, device

# ==========================================
# 3. REAL-TIME 3D PHM & VIBRATION FFT HUD
# ==========================================
def project_3d(point, cam_rot, center_screen, scale=36.0):
    """Projects 3D point (X, Y, Z) to 2D screen coordinates."""
    yaw, pitch = cam_rot
    x, y, z = point[0], point[1], point[2]

    x1 = x * math.cos(yaw) - y * math.sin(yaw)
    y1 = x * math.sin(yaw) + y * math.cos(yaw)
    z1 = z

    x2 = x1
    y2 = y1 * math.cos(pitch) - z1 * math.sin(pitch)
    z2 = y1 * math.sin(pitch) + z1 * math.cos(pitch)

    dist = 18.0 - y2
    focal = max(2.5, dist)
    sx = int(center_screen[0] + (x2 / focal) * scale * 7.5)
    sy = int(center_screen[1] - (z2 / focal) * scale * 7.5)
    return sx, sy, y2

def run_phm_vibration_hud(model, device):
    pygame.init()
    VIEW_WIDTH = 700
    VIEW_HEIGHT = 700
    SIDEBAR_WIDTH = 380
    WIDTH = VIEW_WIDTH + SIDEBAR_WIDTH
    HEIGHT = VIEW_HEIGHT

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Module 13: Predictive Health Maintenance (PHM) & Vibration FFT Diagnostics")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("consolas", 18, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 13, bold=True)

    sim = MechanicalVibrationSimulator()
    model.eval()

    camera_yaw = 0.65
    camera_pitch = 0.42
    center_viewport = (VIEW_WIDTH // 2, VIEW_HEIGHT // 2 + 50)

    drone_pos = np.array([0.0, 0.0, 2.5], dtype=np.float32)
    arm_length = 0.35

    waterfall_history = deque(maxlen=24) # Spectral waterfall depth
    time_domain_wave = deque(maxlen=160)
    pulse_phase = 0.0
    running = True

    print("================ PHM VIBRATION CONTROLS ================")
    print(" [1]            : Inject Nominal State (Healthy Airframe)")
    print(" [2]            : Inject Chipped Propeller Tip (1x Dynamic Unbalance)")
    print(" [3]            : Inject Motor Bearing Wear (BPFO High-Freq Spall)")
    print(" [4]            : Inject Loose Motor Arm Mount (42 Hz Structural Resonance)")
    print(" [UP] / [DOWN]  : Throttle Motor RPM (4200 - 6200 RPM)")
    print(" [ARROW KEYS]   : Orbit 3D Sensor Camera View")
    print("========================================================\n")

    while running:
        dt = clock.tick(30) / 1000.0
        pulse_phase += dt * 6.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    sim.fault_mode = "HEALTHY"
                    print(">>> [PHM INJECT] AIRFRAME RESTORED TO HEALTHY NOMINAL <<<")
                elif event.key == pygame.K_2:
                    sim.fault_mode = "CHIPPED_PROP"
                    print(">>> [PHM INJECT] CHIPPED PROPELLER UNBALANCE ACTIVE (1x HARMONIC) <<<")
                elif event.key == pygame.K_3:
                    sim.fault_mode = "BEARING_WEAR"
                    print(">>> [PHM INJECT] MOTOR BEARING SPALLING ACTIVE (BPFO IMPACTS) <<<")
                elif event.key == pygame.K_4:
                    sim.fault_mode = "LOOSE_ARM"
                    print(">>> [PHM INJECT] LOOSE ARM STRUCTURAL RESONANCE ACTIVE (42 Hz) <<<")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: sim.rpm = min(6200.0, sim.rpm + 25.0)
        if keys[pygame.K_DOWN]: sim.rpm = max(4200.0, sim.rpm - 25.0)

        # Camera rotation
        if keys[pygame.K_LEFT]: camera_yaw -= 0.03
        if keys[pygame.K_RIGHT]: camera_yaw += 0.03

        # 1. Step 1000 Hz Physical Vibration Synthesizer
        # In a 30 Hz display frame, step 33 times to maintain real-time 1000 Hz physics
        for _ in range(33):
            raw_sample = sim.step(dt=0.001)
            time_domain_wave.append(raw_sample)

        # 2. Compute Fast Fourier Transform (FFT) & ISO 10816 Metrics
        freqs, psd, v_rms_mms, kurtosis, peak_freq = sim.compute_fft()
        waterfall_history.append(psd.copy())

        # 3. Neural Fault Diagnostics Inference
        norm_psd = np.log1p(psd) / 8.0
        with torch.no_grad():
            t_in = torch.tensor(norm_psd, device=device).unsqueeze(0).unsqueeze(0).float()
            logits = model(t_in)[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        screen.fill((10, 13, 19))

        # Dividers
        pygame.draw.line(screen, (35, 45, 60), (VIEW_WIDTH, 0), (VIEW_WIDTH, HEIGHT), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (WIDTH, 70), 2)

        # ==========================================
        # 3D VIEWPORT: AIRFRAME STRESS & MOTOR PODS
        # ==========================================
        cam = (camera_yaw, camera_pitch)

        # 1. Ground Grid
        for gx in range(-4, 5):
            p1 = np.array([gx * 1.5, -4.0, 0.0])
            p2 = np.array([gx * 1.5,  4.0, 0.0])
            sx1, sy1, _ = project_3d(p1, cam, center_viewport)
            sx2, sy2, _ = project_3d(p2, cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        for gy in range(-3, 4):
            p1 = np.array([-6.0, gy * 1.5, 0.0])
            p2 = np.array([ 6.0, gy * 1.5, 0.0])
            sx1, sy1, _ = project_3d(p1, cam, center_viewport)
            sx2, sy2, _ = project_3d(p2, cam, center_viewport)
            pygame.draw.line(screen, (20, 26, 36), (sx1, sy1), (sx2, sy2), 1)

        # 2. Draw 4 Carbon Fiber Motor Arms in X-Configuration
        dp = drone_pos
        arm_d = arm_length * 2.2
        motor_locs = [
            dp + np.array([ arm_d / 1.414,  arm_d / 1.414, 0.0]), # M1 (Faulty Motor Pod)
            dp + np.array([-arm_d / 1.414,  arm_d / 1.414, 0.0]), # M2
            dp + np.array([-arm_d / 1.414, -arm_d / 1.414, 0.0]), # M3
            dp + np.array([ arm_d / 1.414, -arm_d / 1.414, 0.0])  # M4
        ]

        hsx, hsy, _ = project_3d(dp, cam, center_viewport)
        m_screens = [project_3d(m, cam, center_viewport)[:2] for m in motor_locs]

        # Draw structural spar beams
        pygame.draw.line(screen, (160, 170, 185), m_screens[0], m_screens[2], 3)
        pygame.draw.line(screen, (160, 170, 185), m_screens[1], m_screens[3], 3)

        # Draw Fuselage Center Pod (IMU Location)
        pygame.draw.circle(screen, (0, 240, 255), (hsx, hsy), 8)
        pygame.draw.circle(screen, (255, 255, 255), (hsx, hsy), 3)

        # Draw Motor Pods with Fault Thermal/Vibration Glow
        for idx, (mx, my) in enumerate(m_screens):
            if idx == 0 and sim.fault_mode != "HEALTHY":
                # Pulsing red warning glow on faulty front-right motor/propeller pod
                pulse_r = int(18 + 5 * math.sin(pulse_phase))
                pygame.draw.circle(screen, (255, 60, 80), (mx, my), pulse_r, 2)
                pygame.draw.circle(screen, (255, 60, 80), (mx, my), 8)
                screen.blit(font_small.render(f"FAULT POD (M1): {sim.fault_mode}", True, (255, 60, 80)), (mx + 12, my - 6))
            else:
                # Nominal green rotor ring
                pygame.draw.circle(screen, (0, 255, 140), (mx, my), 14, 1)
                pygame.draw.circle(screen, (240, 240, 240), (mx, my), 4)

        # Drop line to ground
        gnd_pt = np.array([dp[0], dp[1], 0.0])
        gsx, gsy, _ = project_3d(gnd_pt, cam, center_viewport)
        pygame.draw.line(screen, (35, 55, 75), (hsx, hsy), (gsx, gsy), 1)

        # ==========================================
        # REAL-TIME 1000 HZ FFT SPECTRUM ANALYZER
        # ==========================================
        fft_x = 25
        fft_y = VIEW_HEIGHT - 170
        fft_w = 420
        fft_h = 145
        pygame.draw.rect(screen, (14, 20, 28), (fft_x, fft_y, fft_w, fft_h), border_radius=6)
        pygame.draw.rect(screen, (35, 50, 70), (fft_x, fft_y, fft_w, fft_h), 2)

        screen.blit(font_mono.render("POWER SPECTRAL DENSITY (0 - 500 Hz rFFT):", True, (240, 240, 240)), (fft_x + 12, fft_y + 8))

        # Grid lines for 100, 200, 300, 400 Hz
        for mark_f in [100, 200, 300, 400]:
            px = int(fft_x + 15 + (mark_f / NYQUIST_FREQ_HZ) * (fft_w - 30))
            pygame.draw.line(screen, (25, 35, 48), (px, fft_y + 28), (px, fft_y + fft_h - 12), 1)
            screen.blit(font_small.render(f"{mark_f}Hz", True, (110, 130, 150)), (px - 14, fft_y + fft_h - 22))

        # Plot PSD Frequency Spectrum Bars
        num_bins = len(psd)
        dx_bin = (fft_w - 30) / float(num_bins)
        for b_i in range(1, num_bins):
            bx = int(fft_x + 15 + b_i * dx_bin)
            bar_height = int(min(fft_h - 45, (np.log1p(psd[b_i]) / 9.0) * (fft_h - 45)))
            by = fft_y + fft_h - 24 - bar_height

            # Color bars by frequency zone (Low=Cyan, Mid=Gold, High=Magenta)
            f_val = freqs[b_i]
            if f_val < 120.0:    b_col = (0, 220, 255) # 1x Shaft zone
            elif f_val < 260.0:  b_col = (255, 200, 40) # 2x Blade passing zone
            else:                b_col = (255, 60, 160) # Bearing defect zone

            pygame.draw.line(screen, b_col, (bx, fft_y + fft_h - 24), (bx, by), 2)

        # Fundamental 1x Shaft and BPF Markers
        f0_hz = sim.rpm / 60.0
        f0_px = int(fft_x + 15 + (f0_hz / NYQUIST_FREQ_HZ) * (fft_w - 30))
        pygame.draw.line(screen, (0, 255, 140), (f0_px, fft_y + 26), (f0_px, fft_y + 36), 2)
        screen.blit(font_small.render("1x", True, (0, 255, 140)), (f0_px - 6, fft_y + 38))

        bpf_px = int(fft_x + 15 + ((2.0 * f0_hz) / NYQUIST_FREQ_HZ) * (fft_w - 30))
        pygame.draw.line(screen, (255, 200, 40), (bpf_px, fft_y + 26), (bpf_px, fft_y + 36), 2)
        screen.blit(font_small.render("BPF", True, (255, 200, 40)), (bpf_px - 8, fft_y + 38))

        # ==========================================
        # PHM TELEMETRY & DIAGNOSTICS (SIDEBAR)
        # ==========================================
        sb_x = VIEW_WIDTH + 14
        screen.blit(font_large.render("VIBRATION FFT & PHM", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("1000 Hz Accelerometer Spectrum & Mechanical Health", True, (130, 160, 190)), (sb_x, 38))

        # --- PANEL 1: ISO 10816 VIBRATION SEVERITY ---
        p1_y = 80
        screen.blit(font_mono.render("ISO 10816 VIBRATION VELOCITY:", True, (240, 240, 240)), (sb_x, p1_y))

        if v_rms_mms < ISO_SEV_GOOD:
            iso_class, iso_col = "CLASS A: GOOD (UNRESTRICTED)", (0, 255, 140)
        elif v_rms_mms < ISO_SEV_ACCEPTABLE:
            iso_class, iso_col = "CLASS B: MARGINAL (SCHEDULED)", (255, 200, 40)
        elif v_rms_mms < ISO_SEV_UNSATISFACTORY:
            iso_class, iso_col = "CLASS C: UNSATISFACTORY", (255, 140, 20)
        else:
            iso_class, iso_col = "CLASS D: DANGER (IMMINENT FAILURE!)", (255, 60, 80)

        screen.blit(font_mono.render(iso_class, True, iso_col), (sb_x + 10, p1_y + 24))
        screen.blit(font_small.render(f"Overall Velocity:   v_rms = {v_rms_mms:.2f} mm/s RMS", True, (0, 220, 255)), (sb_x + 10, p1_y + 44))
        screen.blit(font_small.render(f"Spectral Kurtosis:  K     = {kurtosis:.2f} (Bearing Spall > 6)", True, (160, 180, 205)), (sb_x + 10, p1_y + 60))
        screen.blit(font_small.render(f"Peak Frequency:     f_pk  = {peak_freq:.1f} Hz (Dominant)", True, (255, 200, 40)), (sb_x + 10, p1_y + 76))

        # Vibration Velocity Gauge Bar [0 to 10 mm/s]
        bar_x = sb_x + 10
        bar_y = p1_y + 96
        pygame.draw.rect(screen, (24, 32, 45), (bar_x, bar_y, 320, 8), border_radius=2)
        norm_v = min(1.0, max(0.0, v_rms_mms / 10.0))
        fill_w = int(320 * norm_v)
        pygame.draw.rect(screen, iso_col, (bar_x, bar_y, fill_w, 8), border_radius=2)

        # ISO Severity Threshold markers
        mark_b = bar_x + int(320 * (ISO_SEV_GOOD / 10.0))
        mark_c = bar_x + int(320 * (ISO_SEV_ACCEPTABLE / 10.0))
        mark_d = bar_x + int(320 * (ISO_SEV_UNSATISFACTORY / 10.0))
        pygame.draw.line(screen, (255, 200, 40), (mark_b, bar_y - 2), (mark_b, bar_y + 10), 1)
        pygame.draw.line(screen, (255, 140, 20), (mark_c, bar_y - 2), (mark_c, bar_y + 10), 1)
        pygame.draw.line(screen, (255, 60, 80), (mark_d, bar_y - 2), (mark_d, bar_y + 10), 1)

        # --- PANEL 2: HARMONIC DECOMPOSITION ---
        p2_y = bar_y + 24
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (WIDTH - 14, p2_y), 1)
        screen.blit(font_mono.render("HARMONIC COMPONENT AUDIT:", True, (240, 240, 240)), (sb_x, p2_y + 8))

        f_shaft = sim.rpm / 60.0
        f_bpf = 2.0 * f_shaft
        f_bpfo = f_shaft * BEARING_BPFO_MULT

        screen.blit(font_small.render(f"Rotor Shaft (1x):  f_0  = {f_shaft:.1f} Hz ({sim.rpm:.0f} RPM)", True, (0, 255, 140)), (sb_x + 10, p2_y + 28))
        screen.blit(font_small.render(f"Blade Passing BPF: f_bpf= {f_bpf:.1f} Hz (2-Blade Prop)", True, (0, 220, 255)), (sb_x + 10, p2_y + 44))
        screen.blit(font_small.render(f"Bearing Race BPFO: f_bpfo={f_bpfo:.1f} Hz (Outer Ring)", True, (255, 200, 40)), (sb_x + 10, p2_y + 60))
        screen.blit(font_small.render(f"Frame Resonance:   f_res= {AIRFRAME_RESONANCE_HZ:.1f} Hz (Carbon Arm)", True, (160, 180, 205)), (sb_x + 10, p2_y + 76))

        # --- PANEL 3: TIME-DOMAIN ACCELEROMETER OSCILLOSCOPE ---
        p3_y = p2_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (WIDTH - 14, p3_y), 1)
        screen.blit(font_mono.render("IMU RAW ACCELERATION WAVEFORM:", True, (240, 240, 240)), (sb_x, p3_y + 8))

        osc_pts = []
        w_list = list(time_domain_wave)
        if len(w_list) >= 2:
            dx_w = 320.0 / max(1, len(w_list) - 1)
            for idx, val in enumerate(w_list):
                px = int(sb_x + 10 + idx * dx_w)
                py = int(p3_y + 55 - (val / 16.0) * 22)
                osc_pts.append((px, py))
            pygame.draw.lines(screen, (0, 220, 255), False, osc_pts, 1)

        # --- PANEL 4: PYTORCH FAULT CLASSIFIER ---
        p4_y = p3_y + 88
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (WIDTH - 14, p4_y), 1)
        screen.blit(font_mono.render("NEURAL PHM DIAGNOSTIC ARBITER:", True, (240, 240, 240)), (sb_x, p4_y + 8))

        classes = ["Healthy Nominal", "Chipped Prop Unbalance", "Motor Bearing Wear", "Loose Arm Resonance"]
        cls_cols = [(0, 255, 140), (255, 60, 80), (255, 140, 20), (255, 200, 40)]
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
        if sim.fault_mode != "HEALTHY":
            banner_str = f"ALERT: MECHANICAL ANOMALY [{sim.fault_mode}]"
            banner_col = (255, 60, 80)
        else:
            banner_str = "STRUCTURAL HEALTH: ALL MOTORS NOMINAL"
            banner_col = (0, 255, 140)
        screen.blit(font_mono.render(banner_str, True, banner_col), (sb_x + 8, bot_y + 8))

        # Header UI
        head_txt = font_large.render(f"Vib: {v_rms_mms:.2f}mm/s | f_0: {f_shaft:.0f}Hz | RPM: {sim.rpm:.0f} | PHM Diagnostics", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("1000 Hz Accelerometer FFT Spectral Analysis, ISO 10816 Severity, Harmonic Decomposition & CNN Diagnostics", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

# ==========================================
# 4. MAIN ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    mp.freeze_support()
    model, device = train_phm_brain()
    print("Launching Predictive Health Maintenance & Vibration FFT Simulator...")
    run_phm_vibration_hud(model, device)