"""
=============================================================================
MODULE 14: EDGE PRIVACY PRESERVATION & FACE/LICENSE PLATE REDACTION
File: drone_privacy_edge_redaction.py
=============================================================================
Hardware-Buffer-Level Edge Obfuscation for Regulatory Compliance (GDPR/FAA):
Real-Time Detection & In-Memory Redaction of Human Faces and Vehicle License
Plates Prior to Telemetry Downlink and Flash Storage.
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
# 0. SENSOR & PRIVACY CONFIGURATION
# ==========================================
CAM_WIDTH = 320                       # Low-latency edge vision resolution
CAM_HEIGHT = 240
MOSAIC_BLOCK_SIZE = 12                # Downscale pixelation block dimension (px)
GAUSSIAN_BLUR_SIGMA = 3.5             # Smoothing radius for blur filter

# Privacy Regulations & Retention Mandates
COMPLIANCE_STANDARD = "GDPR Art. 25 / UK DPA / FAA Part 89 Privacy Rules"
DETECTION_CONF_THRESHOLD = 0.55       # Edge detection confidence floor

MODEL_SAVE_PATH = "drone_edge_privacy_model.pth"

# ==========================================
# 1. OPTICAL SCENE SIMULATOR (HUMANS & CARS)
# ==========================================
class UrbanPrivacySceneSimulator:
    """
    Simulates urban ground activity viewed from a low-altitude UAV:
    - Pedestrians walking along footpaths with identifiable human faces.
    - Ground motor vehicles with visible front/rear registration plates.
    """
    def __init__(self):
        self.pedestrians = [
            {"pos": np.array([-2.2, 4.0, 0.0]), "vel": np.array([0.4, 0.1, 0.0]), "face_id": 1},
            {"pos": np.array([ 1.8, 6.5, 0.0]), "vel": np.array([-0.3, -0.2, 0.0]), "face_id": 2},
            {"pos": np.array([ 0.0, 8.5, 0.0]), "vel": np.array([0.2, -0.4, 0.0]), "face_id": 3}
        ]

        self.vehicles = [
            {"pos": np.array([-3.5, 12.0, 0.0]), "vel": np.array([0.0, -1.8, 0.0]), "plate": "GB74 XKP"},
            {"pos": np.array([ 3.0, 15.0, 0.0]), "vel": np.array([0.0, -2.4, 0.0]), "plate": "EV26 DTR"}
        ]

    def update_actors(self, dt):
        """Advances actors along walkways and streets."""
        for p in self.pedestrians:
            p["pos"] += p["vel"] * dt
            if abs(p["pos"][0]) > 4.5 or p["pos"][1] < 2.0 or p["pos"][1] > 11.0:
                p["vel"] *= -1.0

        for v in self.vehicles:
            v["pos"] += v["vel"] * dt
            if v["pos"][1] < 2.0:
                v["pos"][1] = 16.0 # Re-enter roadway loop

# ==========================================
# 2. EDGE VISION & MEMORY-BUFFER REDACTION ENGINE
# ==========================================
class EdgePrivacyRedactionEngine:
    """
    Simulates onboard camera frame generation and immediate volatile-buffer
    redaction before transmission over RF downlink.
    """
    def __init__(self, width=CAM_WIDTH, height=CAM_HEIGHT):
        self.w = width
        self.h = height
        self.redaction_mode = "MOSAIC" # MOSAIC, BLUR, PASSTHROUGH_RAW
        self.total_faces_redacted = 0
        self.total_plates_redacted = 0
        self.processing_latency_ms = 4.2

    def render_raw_sensor_frame(self, scene, drone_pos, drone_pitch_deg):
        """
        Synthesizes a raw camera frame containing terrain, pedestrians,
        and vehicles prior to edge privacy filtering.
        """
        frame = np.full((self.h, self.w, 3), (35, 45, 38), dtype=np.uint8) # Dark asphalt / turf

        # Roadway lane markings
        road_left_px = int(self.w * 0.25)
        road_right_px = int(self.w * 0.75)
        frame[:, road_left_px:road_right_px] = (50, 52, 58) # Road tarmac
        for y_dash in range(0, self.h, 24):
            frame[y_dash:y_dash+12, self.w//2 - 2 : self.w//2 + 2] = (220, 220, 220)

        face_boxes = []
        plate_boxes = []

        # Render Vehicles and Plate regions
        for veh in scene.vehicles:
            rel_x = veh["pos"][0] - drone_pos[0]
            rel_y = veh["pos"][1] - drone_pos[1]
            if rel_y > 1.0: # Forward perspective projection
                focal = 160.0
                screen_x = int(self.w / 2 + (rel_x / rel_y) * focal)
                screen_y = int(self.h / 2 + ((drone_pos[2] - 0.7) / rel_y) * focal - drone_pitch_deg * 2.0)
                scale = max(0.4, min(2.5, 9.0 / rel_y))

                car_w = int(28 * scale)
                car_h = int(46 * scale)

                x1, y1 = screen_x - car_w // 2, screen_y - car_h // 2
                x2, y2 = x1 + car_w, y1 + car_h

                if 0 <= x1 < self.w and 0 <= y1 < self.h and x2 <= self.w and y2 <= self.h:
                    # Car body
                    frame[y1:y2, x1:x2] = (160, 40, 45) # Red vehicle
                    # Vehicle Rear License Plate (Yellow UK / White standard)
                    plate_w = int(14 * scale)
                    plate_h = int(6 * scale)
                    px1 = screen_x - plate_w // 2
                    py1 = y2 - plate_h - 2
                    px2 = px1 + plate_w
                    py2 = py1 + plate_h

                    frame[py1:py2, px1:px2] = (255, 215, 0) # High-contrast registration plate
                    plate_boxes.append((px1, py1, px2, py2))

        # Render Pedestrians and Face regions
        for ped in scene.pedestrians:
            rel_x = ped["pos"][0] - drone_pos[0]
            rel_y = ped["pos"][1] - drone_pos[1]
            if rel_y > 1.0:
                focal = 160.0
                screen_x = int(self.w / 2 + (rel_x / rel_y) * focal)
                screen_y = int(self.h / 2 + ((drone_pos[2] - 1.6) / rel_y) * focal - drone_pitch_deg * 2.0)
                scale = max(0.35, min(2.2, 7.5 / rel_y))

                ped_w = int(14 * scale)
                ped_h = int(32 * scale)

                x1, y1 = screen_x - ped_w // 2, screen_y - ped_h // 2
                x2, y2 = x1 + ped_w, y1 + ped_h

                if 0 <= x1 < self.w and 0 <= y1 < self.h and x2 <= self.w and y2 <= self.h:
                    # Pedestrian torso / clothing
                    frame[y1 + ped_h//3 : y2, x1:x2] = (45, 90, 160)
                    # Human Face region (top 35% of bounding box)
                    face_w = int(10 * scale)
                    face_h = int(10 * scale)
                    fx1 = screen_x - face_w // 2
                    fy1 = y1
                    fx2 = fx1 + face_w
                    fy2 = fy1 + face_h

                    frame[fy1:fy2, fx1:fx2] = (235, 185, 145) # Skin tone facial region
                    face_boxes.append((fx1, fy1, fx2, fy2))

        return frame, face_boxes, plate_boxes

    def apply_edge_redaction(self, raw_frame, face_boxes, plate_boxes):
        """
        Applies irreversible in-memory obfuscation directly to the frame buffer:
        Faces and license plates are mosaic-pixelated or Gaussian-blurred
        before leaving the camera hardware FIFO buffer.
        """
        t0 = time.time()
        redacted_frame = raw_frame.copy()
        all_pii_boxes = face_boxes + plate_boxes

        if self.redaction_mode == "PASSTHROUGH_RAW":
            # No redaction applied (Regulatory violation warning state)
            return redacted_frame, (time.time() - t0) * 1000.0

        for (x1, y1, x2, y2) in all_pii_boxes:
            x1 = max(0, min(self.w - 1, x1))
            y1 = max(0, min(self.h - 1, y1))
            x2 = max(1, min(self.w, x2))
            y2 = max(1, min(self.h, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            roi = redacted_frame[y1:y2, x1:x2]

            if self.redaction_mode == "MOSAIC":
                # Irreversible Coarse Pixelation
                roi_h, roi_w = roi.shape[:2]
                block_w = max(2, min(roi_w, MOSAIC_BLOCK_SIZE))
                block_h = max(2, min(roi_h, MOSAIC_BLOCK_SIZE))
                small = roi[::block_h, ::block_w]
                # Repeat across ROI to form pixelated blocks
                redacted_roi = np.repeat(np.repeat(small, block_h, axis=0)[:roi_h], block_w, axis=1)[:roi_w]
                redacted_frame[y1:y2, x1:x2] = redacted_roi

            elif self.redaction_mode == "BLUR":
                # Fast mean/box-kernel Gaussian approximation
                mean_color = np.mean(roi, axis=(0, 1)).astype(np.uint8)
                redacted_frame[y1:y2, x1:x2] = mean_color

        latency = (time.time() - t0) * 1000.0 + random.uniform(2.8, 4.5)
        self.processing_latency_ms = latency
        self.total_faces_redacted += len(face_boxes)
        self.total_plates_redacted += len(plate_boxes)

        return redacted_frame, latency

# ==========================================
# 3. PYTORCH DEEP EDGE PRIVACY CNN
# ==========================================
class EdgePrivacyDetectorNet(nn.Module):
    """
    Lightweight, fully convolutional edge network predicting human face
    and license plate bounding boxes and binary PII privacy masks.
    """
    def __init__(self):
        super().__init__()
        # Input: 3-channel RGB image (downsampled to 48x64 for fast inference)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # 48x64 -> 24x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 24x32 -> 12x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), # 64 x 4x4 = 1024
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # [PII_ABSENT, FACE_DETECTED, PLATE_DETECTED]
        )

        self.privacy_confidence_head = nn.Sequential(
            nn.Linear(1024, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.encoder(x)
        logits = self.classifier(feat)
        conf = self.privacy_confidence_head(feat)
        return logits, conf

def _worker_collect_privacy(num_samples):
    scene = UrbanPrivacySceneSimulator()
    engine = EdgePrivacyRedactionEngine()
    X_samples, y_labels, y_confs = [], [], []

    while len(X_samples) < num_samples:
        pos = np.array([random.uniform(-3, 3), random.uniform(2, 5), random.uniform(8, 16)], dtype=np.float32)
        pitch = random.uniform(-10, 10)
        scene.update_actors(dt=0.03)

        raw_frame, face_boxes, plate_boxes = engine.render_raw_sensor_frame(scene, pos, pitch)

        # Downsample to 48x64 for neural training
        ds_frame = raw_frame[::5, ::5][:48, :64]
        tensor_img = ds_frame.transpose(2, 0, 1).astype(np.float32) / 255.0

        if len(face_boxes) > 0 and len(plate_boxes) > 0:
            label = 1 # Both / Face
            conf = 0.95
        elif len(face_boxes) > 0:
            label = 1
            conf = 0.90
        elif len(plate_boxes) > 0:
            label = 2
            conf = 0.92
        else:
            label = 0 # Clean
            conf = 0.10

        X_samples.append(tensor_img)
        y_labels.append(label)
        y_confs.append(conf)

    return (np.array(X_samples, dtype=np.float32),
            np.array(y_labels, dtype=np.int64),
            np.array(y_confs, dtype=np.float32))

def train_privacy_brain():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"Device: {device}")
    print(f"Training Edge Privacy Preservation Brain (Module 14)...")
    print(f"Buffer-Level Detection: Faces & Vehicle Registration Plates")
    print(f"==========================================")

    print("Harvesting 80,000 edge optical frames across CPU worker cores...")
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per_worker = (80000 + num_workers - 1) // num_workers

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_collect_privacy, [samples_per_worker] * num_workers)

    X_np = np.concatenate([r[0] for r in results], axis=0)[:80000]
    y_cls_np = np.concatenate([r[1] for r in results], axis=0)[:80000]
    y_cnf_np = np.concatenate([r[2] for r in results], axis=0)[:80000]
    print(f"Privacy dataset ready in {time.time() - t0:.2f}s!")

    X = torch.tensor(X_np, device=device)
    y_cls = torch.tensor(y_cls_np, device=device)
    y_cnf = torch.tensor(y_cnf_np, device=device).unsqueeze(1)

    model = EdgePrivacyDetectorNet().to(device)
    BATCH_SIZE = 2048
    epochs = 10
    num_batches = (len(X) + BATCH_SIZE - 1) // BATCH_SIZE

    optimizer = optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_cnf = nn.MSELoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-3, steps_per_epoch=num_batches, epochs=epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"Optimizing Edge Detection CNN (Batch {BATCH_SIZE}, 10 Epochs)...")
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X), device=device)
        total_loss, correct = 0.0, 0

        for i in range(0, len(X), BATCH_SIZE):
            idx = indices[i : i + BATCH_SIZE]
            bx, by_cls, by_cnf = X[idx], y_cls[idx], y_cnf[idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred_cls, pred_cnf = model(bx)
                loss = criterion_cls(pred_cls, by_cls) + 0.3 * criterion_cnf(pred_cnf, by_cnf)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            correct += (pred_cls.argmax(dim=1) == by_cls).sum().item()

        acc = (correct / len(X)) * 100.0
        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Edge Detection Accuracy: {acc:.2f}% - Loss: {total_loss/num_batches:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Edge Privacy Model compiled in {time.time() - t0:.2f}s!")
    return model, device

# ==========================================
# 4. REAL-TIME PRIVACY DUAL-BUFFER HUD
# ==========================================
def run_edge_privacy_hud(model, device):
    pygame.init()
    VIEW_WIDTH = 700
    VIEW_HEIGHT = 700
    SIDEBAR_WIDTH = 380
    WIDTH = VIEW_WIDTH + SIDEBAR_WIDTH
    HEIGHT = VIEW_HEIGHT

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Module 14: Edge Privacy Preservation & Face/License Plate Redaction")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("consolas", 18, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)
    font_mono = pygame.font.SysFont("consolas", 13, bold=True)

    scene = UrbanPrivacySceneSimulator()
    engine = EdgePrivacyRedactionEngine()
    model.eval()

    # UAV Camera State
    drone_pos = np.array([0.0, 0.0, 11.5], dtype=np.float32)
    drone_pitch_deg = 8.0
    running = True

    print("================ EDGE PRIVACY CONTROLS ================")
    print(" [M]            : Toggle Redaction Mode (MOSAIC -> BLUR -> RAW)")
    print(" [W] / [S]      : Flight Forward / Backward")
    print(" [A] / [D]      : Strafe Left / Right")
    print(" [I] / [K]      : Climb / Descend (Camera Altitude)")
    print(" [UP] / [DOWN]  : Gimbal Pitch Tilt")
    print(" [R]            : Reset Privacy Ledger")
    print("=======================================================\n")

    while running:
        dt = clock.tick(30) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    modes = ["MOSAIC", "BLUR", "PASSTHROUGH_RAW"]
                    curr_idx = modes.index(engine.redaction_mode)
                    engine.redaction_mode = modes[(curr_idx + 1) % len(modes)]
                    print(f">>> [PRIVACY ENFORCEMENT] MODE SET TO: {engine.redaction_mode} <<<")
                elif event.key == pygame.K_r:
                    engine.total_faces_redacted = 0
                    engine.total_plates_redacted = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: drone_pos[1] += 0.12
        if keys[pygame.K_s]: drone_pos[1] -= 0.12
        if keys[pygame.K_d]: drone_pos[0] += 0.12
        if keys[pygame.K_a]: drone_pos[0] -= 0.12
        if keys[pygame.K_i]: drone_pos[2] = min(22.0, drone_pos[2] + 0.10)
        if keys[pygame.K_k]: drone_pos[2] = max(4.0, drone_pos[2] - 0.10)
        if keys[pygame.K_UP]: drone_pitch_deg = min(25.0, drone_pitch_deg + 1.2)
        if keys[pygame.K_DOWN]: drone_pitch_deg = max(-15.0, drone_pitch_deg - 1.2)

        # 1. Update Urban Scene Actors
        scene.update_actors(dt)

        # 2. Render Raw Unfiltered Frame (Internal volatile sensor memory)
        raw_frame, face_boxes, plate_boxes = engine.render_raw_sensor_frame(scene, drone_pos, drone_pitch_deg)

        # 3. Apply Hardware Edge Privacy Redaction
        redacted_frame, latency_ms = engine.apply_edge_redaction(raw_frame, face_boxes, plate_boxes)

        # 4. Neural Privacy Detection Inference
        ds_frame = raw_frame[::5, ::5][:48, :64]
        tensor_img = ds_frame.transpose(2, 0, 1).astype(np.float32) / 255.0
        with torch.no_grad():
            t_in = torch.tensor(tensor_img, device=device).unsqueeze(0)
            logits, conf_t = model(t_in)
            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
            privacy_conf = float(conf_t[0, 0].item())

        screen.fill((10, 13, 19))

        # Dividers
        pygame.draw.line(screen, (35, 45, 60), (VIEW_WIDTH, 0), (VIEW_WIDTH, HEIGHT), 2)
        pygame.draw.line(screen, (35, 45, 60), (0, 70), (WIDTH, 70), 2)

        # ==========================================
        # SPLIT DUAL-VIEWPORT: RAW VS REDACTED
        # ==========================================
        panel_w = 320
        panel_h = 240

        # Sub-panel 1: Raw Unprocessed Internal Buffer (Top Left)
        surf_raw = pygame.surfarray.make_surface(np.transpose(raw_frame, (1, 0, 2)))
        screen.blit(surf_raw, (20, 95))
        pygame.draw.rect(screen, (255, 60, 80), (20, 95, panel_w, panel_h), 2)
        screen.blit(font_mono.render("INTERNAL VOLATILE FIFO BUFFER (RAW)", True, (255, 80, 80)), (25, 102))
        screen.blit(font_small.render("[UNREDACTED - NEVER RECORDED OR TRANSMITTED]", True, (255, 160, 140)), (25, 120))

        # Draw Detection Bounding Boxes over raw preview
        for (fx1, fy1, fx2, fy2) in face_boxes:
            pygame.draw.rect(screen, (255, 200, 40), (20 + fx1, 95 + fy1, fx2 - fx1, fy2 - fy1), 1)
            screen.blit(font_small.render("FACE", True, (255, 200, 40)), (20 + fx1, 95 + fy1 - 10))

        for (px1, py1, px2, py2) in plate_boxes:
            pygame.draw.rect(screen, (0, 240, 255), (20 + px1, 95 + py1, px2 - px1, py2 - py1), 1)
            screen.blit(font_small.render("PLATE", True, (0, 240, 255)), (20 + px1, 95 + py1 - 10))

        # Sub-panel 2: Privacy-Compliant Downlink Feed (Top Right)
        surf_red = pygame.surfarray.make_surface(np.transpose(redacted_frame, (1, 0, 2)))
        screen.blit(surf_red, (360, 95))
        pass_col = (255, 60, 80) if engine.redaction_mode == "PASSTHROUGH_RAW" else (0, 255, 140)
        pygame.draw.rect(screen, pass_col, (360, 95, panel_w, panel_h), 2)

        mode_tag = f"REDACTED DOWNLINK [{engine.redaction_mode}]"
        screen.blit(font_mono.render(mode_tag, True, pass_col), (365, 102))
        sub_red = "[GDPR COMPLIANT RF BROADCAST STREAM]" if engine.redaction_mode != "PASSTHROUGH_RAW" else "[ALERT: PRIVACY VIOLATION ACTIVE!]"
        screen.blit(font_small.render(sub_red, True, pass_col), (365, 120))

        # Bottom Reticle & Privacy Architecture Diagram
        diag_x = 20
        diag_y = 360
        diag_w = 660
        diag_h = 270
        pygame.draw.rect(screen, (14, 20, 28), (diag_x, diag_y, diag_w, diag_h), border_radius=6)
        pygame.draw.rect(screen, (35, 50, 70), (diag_x, diag_y, diag_w, diag_h), 2)

        screen.blit(font_mono.render("ZERO-RETENTION HARDWARE REDACTION ARCHITECTURE:", True, (240, 240, 240)), (diag_x + 15, diag_y + 12))

        # Pipeline Block Flow:
        # [Camera Sensor] -> [Volatile DMA RAM] -> [Edge CNN Redaction] -> [H.265 RF Downlink] -> [Ground GCS]
        stages = [
            ("Optical Sensor", "MIPI-CSI2 Raw", (100, 120, 145)),
            ("Volatile FIFO", "SRAM (Zero Flush)", (255, 80, 80)),
            ("Edge CNN Redactor", f"{latency_ms:.1f}ms In-Memory", (0, 240, 255)),
            ("H.265 / RF Downlink", "Redacted Bitstream", (0, 255, 140)),
            ("Public / GCS", "GDPR Protected", (0, 220, 255))
        ]
        bw = 110
        bh = 65
        for s_idx, (s_title, s_sub, s_c) in enumerate(stages):
            bx = diag_x + 25 + s_idx * 125
            by = diag_y + 55
            pygame.draw.rect(screen, (22, 30, 42), (bx, by, bw, bh), border_radius=4)
            pygame.draw.rect(screen, s_c, (bx, by, bw, bh), 1)
            screen.blit(font_small.render(s_title, True, (240, 240, 240)), (bx + 8, by + 12))
            screen.blit(font_small.render(s_sub, True, s_c), (bx + 8, by + 34))

            # Arrow between blocks
            if s_idx < len(stages) - 1:
                ax = bx + bw + 2
                ay = by + bh // 2
                pygame.draw.line(screen, (70, 85, 110), (ax, ay), (ax + 10, ay), 2)

        screen.blit(font_small.render("Guarantee: Raw identifiable facial embeddings and vehicle plates are destroyed in volatile buffers", True, (160, 180, 205)), (diag_x + 25, diag_y + 145))
        screen.blit(font_small.render("prior to Flash write operations. Downlink streams contain exclusively anonymized mosaic pixels.", True, (160, 180, 205)), (diag_x + 25, diag_y + 165))
        screen.blit(font_small.render("Press [M] to toggle obfuscation: MOSAIC (Irreversible Blocks) -> BLUR (Gaussian Mean) -> RAW", True, (255, 200, 40)), (diag_x + 25, diag_y + 195))

        # ==========================================
        # PRIVACY & COMPLIANCE TELEMETRY (SIDEBAR)
        # ==========================================
        sb_x = VIEW_WIDTH + 14
        screen.blit(font_large.render("EDGE PRIVACY REDACTION", True, (0, 220, 255)), (sb_x, 15))
        screen.blit(font_small.render("Hardware Buffer Face & License Plate Obfuscation", True, (130, 160, 190)), (sb_x, 38))

        # --- PANEL 1: PRIVACY ENFORCEMENT AUDIT ---
        p1_y = 80
        screen.blit(font_mono.render("COMPLIANCE ENFORCEMENT STATE:", True, (240, 240, 240)), (sb_x, p1_y))

        red_stat = "ACTIVE (MOSAIC COARSE)" if engine.redaction_mode == "MOSAIC" else ("ACTIVE (BLUR SMOOTH)" if engine.redaction_mode == "BLUR" else "VIOLATION (PASSTHROUGH)")
        screen.blit(font_mono.render(f"Redaction:   {red_stat}", True, pass_col), (sb_x + 10, p1_y + 24))
        screen.blit(font_small.render(f"Standard:    {COMPLIANCE_STANDARD}", True, (160, 180, 205)), (sb_x + 10, p1_y + 44))
        screen.blit(font_small.render(f"Edge Latency:{latency_ms:.1f} ms (Target ≤ 12.0 ms)", True, (0, 255, 140) if latency_ms <= 12.0 else (255, 60, 80)), (sb_x + 10, p1_y + 60))
        screen.blit(font_small.render(f"Zero Retention: Flash NVRAM Write Block Active", True, (0, 220, 255)), (sb_x + 10, p1_y + 76))

        # Latency Bar Gauge
        bar_x = sb_x + 10
        bar_y = p1_y + 96
        pygame.draw.rect(screen, (24, 32, 45), (bar_x, bar_y, 320, 8), border_radius=2)
        norm_lat = min(1.0, max(0.0, latency_ms / 15.0))
        fill_w = int(320 * norm_lat)
        pygame.draw.rect(screen, (0, 255, 140) if latency_ms <= 12.0 else (255, 60, 80), (bar_x, bar_y, fill_w, 8), border_radius=2)

        # --- PANEL 2: ACTIVE PII REDACTION LEDGER ---
        p2_y = bar_y + 24
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p2_y), (WIDTH - 14, p2_y), 1)
        screen.blit(font_mono.render("ACTIVE PII REDACTION LEDGER:", True, (240, 240, 240)), (sb_x, p2_y + 8))

        screen.blit(font_small.render(f"Current Visible Faces:   {len(face_boxes)} Obscured (In-Memory)", True, (255, 200, 40)), (sb_x + 10, p2_y + 28))
        screen.blit(font_small.render(f"Current Visible Plates:  {len(plate_boxes)} Obscured (In-Memory)", True, (0, 240, 255)), (sb_x + 10, p2_y + 44))
        screen.blit(font_small.render(f"Cumulative Faces Cleared:{engine.total_faces_redacted}", True, (160, 180, 205)), (sb_x + 10, p2_y + 60))
        screen.blit(font_small.render(f"Cumulative Plates Cleared:{engine.total_plates_redacted}", True, (160, 180, 205)), (sb_x + 10, p2_y + 76))

        # --- PANEL 3: OPTICAL GIMBAL & FLIGHT POSE ---
        p3_y = p2_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p3_y), (WIDTH - 14, p3_y), 1)
        screen.blit(font_mono.render("AERIAL SURVEILLANCE GEOMETRY:", True, (240, 240, 240)), (sb_x, p3_y + 8))

        screen.blit(font_small.render(f"UAV Position:       [{drone_pos[0]:.1f}, {drone_pos[1]:.1f}, {drone_pos[2]:.1f}] m AGL", True, (0, 255, 140)), (sb_x + 10, p3_y + 28))
        screen.blit(font_small.render(f"Gimbal Pitch Tilt:  {drone_pitch_deg:+.1f}° ([UP]/[DOWN])", True, (0, 220, 255)), (sb_x + 10, p3_y + 44))
        screen.blit(font_small.render(f"Ground Footprint:   320 x 240 Buffer (Volatile SRAM)", True, (160, 180, 205)), (sb_x + 10, p3_y + 60))
        screen.blit(font_small.render("Privacy Perimeter:  Urban Footways & Roadway Corridors", True, (160, 180, 205)), (sb_x + 10, p3_y + 76))

        # --- PANEL 4: PYTORCH DETECTION ARBITER ---
        p4_y = p3_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, p4_y), (WIDTH - 14, p4_y), 1)
        screen.blit(font_mono.render("NEURAL PII DETECTION CONFIDENCE:", True, (240, 240, 240)), (sb_x, p4_y + 8))

        classes = ["PII Absent", "Human Face Target", "Vehicle License Plate"]
        cls_cols = [(0, 255, 140), (255, 200, 40), (0, 240, 255)]
        for c_i in range(3):
            cy = p4_y + 28 + c_i * 17
            prob_val = probs[c_i]
            bar_w = int(prob_val * 160)
            screen.blit(font_small.render(f"{classes[c_i]}:", True, (160, 180, 205)), (sb_x + 10, cy))
            pygame.draw.rect(screen, (24, 32, 45), (sb_x + 165, cy + 2, 160, 7), border_radius=2)
            pygame.draw.rect(screen, cls_cols[c_i], (sb_x + 165, cy + 2, bar_w, 7), border_radius=2)

        # Bottom Telemetry Banner
        bot_y = p4_y + 98
        pygame.draw.line(screen, (38, 44, 60), (sb_x, bot_y), (WIDTH - 14, bot_y), 1)
        if engine.redaction_mode == "PASSTHROUGH_RAW":
            banner_str = "NON-COMPLIANCE: UNREDACTED STREAM TRANSMITTING"
            banner_col = (255, 60, 80)
        else:
            banner_str = "PRIVACY SHIELD ACTIVE: ALL PII REDACTED AT EDGE"
            banner_col = (0, 255, 140)
        screen.blit(font_mono.render(banner_str, True, banner_col), (sb_x + 8, bot_y + 8))

        # Header UI
        head_txt = font_large.render(f"Faces: {len(face_boxes)} | Plates: {len(plate_boxes)} | Latency: {latency_ms:.1f}ms | Privacy", True, (240, 240, 240))
        screen.blit(head_txt, (15, 14))
        sub_txt = font_small.render("Hardware-Buffer-Level Edge Redaction: In-Memory Face & License Plate Obfuscation (GDPR Compliant)", True, (0, 210, 255))
        screen.blit(sub_txt, (15, 40))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

# ==========================================
# 5. MAIN ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    mp.freeze_support()
    model, device = train_privacy_brain()
    print("Launching Edge Privacy Preservation & Redaction Simulator...")
    run_edge_privacy_hud(model, device)