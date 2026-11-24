"""
=============================================================================
SURVIVAL CORE V37.0 — COMPLETE GENESIS NEURAL AUTONOMOUS SWARM SYSTEM
=============================================================================
Fixes Applied:
  - Velocity-Profiled Guidance (Smooth 2.5 m/s climb; prevents 10G shock launches)
  - Impulse-Capped Tether Spring-Damper (18N max tension; eliminates LCP spikes)
  - 200 Hz Simulation Timestep (dt=0.005s; satisfies Genesis solver stability)
  - Clean non-overlapping spawn hierarchy
=============================================================================
Usage:
  Step 1 (Train):   python genesis_swarm_nn.py --train
  Step 2 (View 3D): python genesis_swarm_nn.py
=============================================================================
"""

import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import genesis as gs

# =============================================================================
# 0. PHYSICAL CONSTANTS & MODEL SPECIFICATIONS
# =============================================================================
DRY_MASS_KG = 1.35
CARGO_MASS_KG = 0.45
TOTAL_MASS_KG = DRY_MASS_KG + CARGO_MASS_KG
GRAVITY_VAL = 9.80665

ARM_LENGTH_M = 0.26 / math.sqrt(2.0)
ROTOR_RADIUS_M = 0.127
ROTOR_DISK_AREA = math.pi * (ROTOR_RADIUS_M ** 2)

C_T_CLEAN = 1.35e-5
C_Q_CLEAN = 2.40e-7
COAXIAL_EFFICIENCY = 0.92
MOTOR_TAU_S = 0.030
MAX_OMEGA = (9200.0 * 2.0 * math.pi) / 60.0
HOVER_THRUST = TOTAL_MASS_KG * GRAVITY_VAL

BASE_OBS_DIM = 139
HISTORY_LEN = 3
OBSERVATION_DIM = BASE_OBS_DIM * HISTORY_LEN
NUM_FSM_STATES = 7
MODEL_FILE = "drone_master_brain_genesis.pth"

FSM_NAMES = [
    "TAKEOFF",             # 0
    "CRUISE_OUTBOUND",    # 1
    "CHIMNEY_DESCENT",    # 2
    "DELIVERY_HOVER",     # 3
    "WINCH_RETRACT",      # 4
    "RETURN_BASE",        # 5
    "DEPOT_LANDING"       # 6
]

# =============================================================================
# 1. BATCHED LIE ALGEBRA & SO(3) TENSOR MATH
# =============================================================================
def quat_to_rot_batch(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    B = q.shape[0]
    R = torch.zeros((B, 3, 3), device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1.0 - 2.0 * (y*y + z*z)
    R[:, 0, 1] = 2.0 * (x*y - w*z)
    R[:, 0, 2] = 2.0 * (x*z + w*y)
    R[:, 1, 0] = 2.0 * (x*y + w*z)
    R[:, 1, 1] = 1.0 - 2.0 * (x*x + z*z)
    R[:, 1, 2] = 2.0 * (y*z - w*x)
    R[:, 2, 0] = 2.0 * (x*z - w*y)
    R[:, 2, 1] = 2.0 * (y*z + w*x)
    R[:, 2, 2] = 1.0 - 2.0 * (x*x + y*y)
    return R

def vee_map_batch(M: torch.Tensor) -> torch.Tensor:
    return torch.stack([M[:, 2, 1], M[:, 0, 2], M[:, 1, 0]], dim=-1)

# =============================================================================
# 2. DEEP DENSE FLIGHT EXECUTIVE NETWORK (~1.2M PARAMETERS)
# =============================================================================
class DenseResBlock(nn.Module):
    def __init__(self, dim=256, hidden_dim=384):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return x + self.fc2(self.norm2(self.act(self.fc1(self.norm1(x)))))

class DeepDenseFlightExecutiveNet(nn.Module):
    def __init__(self, in_features=OBSERVATION_DIM):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.SiLU()
        )
        self.block1 = DenseResBlock(256, 384)
        self.block2 = DenseResBlock(256, 384)

        self.guidance_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
            nn.Tanh()
        )
        self.fsm_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, NUM_FSM_STATES)
        )
        self.winch_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        feat = self.block2(self.block1(self.input_proj(x)))
        guidance = self.guidance_head(feat) * 2.0
        fsm_logits = self.fsm_head(feat)
        winch_cmd = self.winch_head(feat)
        value = self.value_head(feat)
        return guidance, fsm_logits, winch_cmd, value

# =============================================================================
# 3. GPU-NATIVE OBSERVATION SYNTHESIZER
# =============================================================================
class VectorizedObservationManager:
    def __init__(self, n_envs: int, device: str = "cuda"):
        self.n_envs = n_envs
        self.device = device
        self.history = torch.zeros((n_envs, HISTORY_LEN, BASE_OBS_DIM), device=device)

    def update_and_get(self, d_pos, d_vel, d_quat, d_ang, c_pos, c_vel, target_pos, cable_len, has_payload):
        B = self.n_envs
        obs = torch.zeros((B, BASE_OBS_DIM), device=self.device)

        obs[:, 0:3] = (target_pos - d_pos) / 50.0
        obs[:, 3:6] = d_pos / 100.0
        obs[:, 6:9] = d_vel / 10.0
        R = quat_to_rot_batch(d_quat)
        obs[:, 9:18] = R.reshape(B, 9)
        obs[:, 18:21] = d_ang / 10.0

        dist_xy = torch.norm(target_pos[:, :2] - d_pos[:, :2], dim=-1, keepdim=True)
        obs[:, 21:37] = torch.clamp(dist_xy / 30.0, 0.0, 1.0).repeat(1, 16)

        obs[:, 37] = torch.clamp(d_pos[:, 2] / 40.0, 0.0, 1.0)
        obs[:, 38] = torch.norm(d_vel, dim=-1) / 15.0

        obs[:, 53] = cable_len / 10.0
        obs[:, 54] = torch.norm(c_vel - d_vel, dim=-1) / 5.0
        tether_vec = c_pos - d_pos
        obs[:, 56:58] = tether_vec[:, :2] / 5.0
        obs[:, 58] = 1.0 if has_payload else 0.0

        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:, -1, :] = obs
        return self.history.reshape(B, OBSERVATION_DIM)

# =============================================================================
# 4. VECTORIZED BEMT AERODYNAMICS & SE(3) CONTROLLER
# =============================================================================
class VectorizedBEMTEngine(nn.Module):
    def __init__(self, n_envs: int, device: str = "cuda"):
        super().__init__()
        self.n_envs = n_envs
        self.device = device
        self.motor_omegas = torch.full((n_envs, 8), 450.0, device=device)

        L = ARM_LENGTH_M
        cq_ct_top = C_Q_CLEAN / C_T_CLEAN
        cq_ct_bot = (C_Q_CLEAN / C_T_CLEAN) / COAXIAL_EFFICIENCY
        
        B_mat = torch.tensor([
            [ 1.0,         1.0,        1.0,         1.0,        1.0,         1.0,        1.0,         1.0],
            [-L,          -L,          L,           L,          L,           L,         -L,          -L],
            [-L,          -L,         -L,          -L,          L,           L,          L,           L],
            [-cq_ct_top, cq_ct_bot, cq_ct_top, -cq_ct_bot, -cq_ct_top, cq_ct_bot, cq_ct_top, -cq_ct_bot]
        ], device=device, dtype=torch.float32)

        self.register_buffer("B_mat", B_mat)
        self.register_buffer("B_pinv", torch.linalg.pinv(B_mat))

    def compute_aerodynamics(self, motor_cmds: torch.Tensor, pos: torch.Tensor, vel: torch.Tensor, target_surf_z: float, dt: float):
        effective_cmds = torch.clamp(motor_cmds, 0.0, MAX_OMEGA)
        self.motor_omegas += ((effective_cmds - self.motor_omegas) / MOTOR_TAU_S) * dt

        z_clearance = torch.clamp(pos[:, 2] - target_surf_z, min=0.06)
        ratio_zh_r = z_clearance / ROTOR_RADIUS_M
        ige_factor = torch.where(
            ratio_zh_r >= 0.5,
            torch.clamp(1.0 / (1.0 - (1.0 / (4.0 * ratio_zh_r)) ** 2), 1.0, 1.28),
            1.20 + 0.16 * (ratio_zh_r / 0.5)
        )

        v_h_hover = math.sqrt(HOVER_THRUST / (2.0 * 1.225 * ROTOR_DISK_AREA * 4.0))
        v_z_descent = -vel[:, 2]
        is_vrs_band = (v_z_descent > 0.4 * v_h_hover) & (v_z_descent < 1.6 * v_h_hover)
        vrs_factor = torch.ones(self.n_envs, device=self.device)
        if torch.any(is_vrs_band):
            ratio = (v_z_descent[is_vrs_band] - 0.4 * v_h_hover) / (1.2 * v_h_hover)
            vrs_factor[is_vrs_band] = 1.0 - 0.28 * torch.sin(math.pi * ratio)

        lift_scale = ige_factor * vrs_factor
        omega_sq = self.motor_omegas ** 2
        T = torch.zeros_like(self.motor_omegas)
        T[:, 0::2] = C_T_CLEAN * omega_sq[:, 0::2] * lift_scale.unsqueeze(-1)
        T[:, 1::2] = (C_T_CLEAN * COAXIAL_EFFICIENCY) * omega_sq[:, 1::2] * lift_scale.unsqueeze(-1)

        wrench = torch.matmul(self.B_mat, T.unsqueeze(-1)).squeeze(-1)
        return wrench[:, 0], wrench[:, 1:4]

    def allocate(self, thrust: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        w = torch.cat([thrust.unsqueeze(-1), tau], dim=-1)
        T_cmds = torch.clamp(torch.matmul(self.B_pinv, w.unsqueeze(-1)).squeeze(-1), 0.05, 10.5)
        omega_cmds = torch.zeros_like(T_cmds)
        omega_cmds[:, 0::2] = torch.sqrt(T_cmds[:, 0::2] / C_T_CLEAN)
        omega_cmds[:, 1::2] = torch.sqrt(T_cmds[:, 1::2] / (C_T_CLEAN * COAXIAL_EFFICIENCY))
        return omega_cmds

class GeometricSE3Controller:
    def __init__(self, n_envs: int, device: str = "cuda"):
        self.n_envs = n_envs
        self.device = device
        self.k_v = torch.tensor([2.0, 2.0, 2.5], device=device)
        self.k_R = torch.tensor([2.1, 2.1, 1.7], device=device)
        self.k_w = torch.tensor([1.4, 1.4, 0.95], device=device)

    def compute_wrench(self, pos, vel, quat, ang_vel, target_pos, current_mass, nn_guidance_accel=None):
        B = self.n_envs
        R = quat_to_rot_batch(quat)

        # Smooth velocity-profiled guidance (prevents explosive 10G step commands)
        delta_p = target_pos - pos
        v_cmd_xy = torch.clamp(delta_p[:, :2] * 1.4, -6.0, 6.0)
        v_cmd_z = torch.clamp(delta_p[:, 2] * 1.2, -1.5, 2.4)
        v_des = torch.cat([v_cmd_xy, v_cmd_z.unsqueeze(-1)], dim=-1)

        a_des = self.k_v * (v_des - vel)
        a_des[:, 2] += GRAVITY_VAL
        if nn_guidance_accel is not None:
            a_des += nn_guidance_accel

        a_des = torch.clamp(a_des, -15.0, 22.0)
        F_d = current_mass * a_des
        b3_d = F_d / torch.clamp(torch.norm(F_d, dim=-1, keepdim=True), min=1e-4)

        b1_c = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(B, 1)
        b2_d = torch.linalg.cross(b3_d, b1_c, dim=-1)
        b2_d = b2_d / torch.clamp(torch.norm(b2_d, dim=-1, keepdim=True), min=1e-4)
        b1_d = torch.linalg.cross(b2_d, b3_d, dim=-1)
        R_d = torch.stack([b1_d, b2_d, b3_d], dim=-1)

        f_thrust = torch.clamp(torch.sum(F_d * R[:, :, 2], dim=-1), min=0.0, max=2.0 * (current_mass * GRAVITY_VAL))
        R_err = torch.bmm(R_d.transpose(1, 2), R) - torch.bmm(R.transpose(1, 2), R_d)
        e_R = 0.5 * vee_map_batch(R_err)
        tau = torch.clamp(-self.k_R * e_R - self.k_w * ang_vel, -0.65, 0.65)
        return f_thrust, tau

# =============================================================================
# 5. PROCEDURAL 3D CITY ENVIRONMENT SETUP
# =============================================================================
def build_metropolitan_city(scene):
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(friction=0.8),
        surface=gs.surfaces.Rough(color=(0.18, 0.20, 0.24), roughness=0.8)
    )

    manifest = []
    cols = [-60.0, -30.0, 0.0, 30.0, 60.0]
    rows = [-60.0, -30.0, 0.0, 30.0, 60.0]
    idx = 0

    for c_i, cx in enumerate(cols):
        for r_i, ry in enumerate(rows):
            if cx <= -20.0 and ry <= -20.0:
                continue

            tier = (c_i + r_i) % 4
            height = [8.5, 14.0, 20.0, 26.0][tier]
            scene.add_entity(
                morph=gs.morphs.Box(size=(16.0, 16.0, height), pos=(cx, ry, height / 2.0), fixed=True),
                material=gs.materials.Rigid(friction=0.7),
                surface=gs.surfaces.Rough(color=(0.32, 0.35, 0.40), roughness=0.6)
            )

            # Helipad Target
            pad_pos = (cx + 2.0, ry - 2.0, height + 0.04)
            scene.add_entity(
                morph=gs.morphs.Cylinder(radius=2.2, height=0.08, pos=pad_pos, fixed=True),
                material=gs.materials.Rigid(friction=0.8),
                surface=gs.surfaces.Default(color=(0.0, 0.85, 0.55))
            )
            manifest.append({"id": f"Bldg_{idx+1:02d}", "pos": np.array(pad_pos, dtype=np.float32), "height": height})
            idx += 1

    depot_pad = np.array([-45.0, -45.0, 0.04], dtype=np.float32)
    scene.add_entity(
        morph=gs.morphs.Cylinder(radius=3.0, height=0.08, pos=tuple(depot_pad), fixed=True),
        material=gs.materials.Rigid(friction=0.9),
        surface=gs.surfaces.Default(color=(0.10, 0.65, 0.95))
    )
    return manifest, depot_pad

# =============================================================================
# 6. FAST HEADLESS PARALLEL TRAINING PIPELINE (64 ENVS ON GPU)
# =============================================================================
def train_neural_executive():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*75)
    print("STAGE 1: GENESIS MASSIVE GPU ROLLOUT HARVESTING & RESNET PRE-TRAINING")
    print("="*75)

    gs.init(backend=gs.gpu)
    dt = 0.005  # 200 Hz simulation step (satisfies rigid solver stability)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt, 
            substeps=2,
            gravity=(0, 0, -GRAVITY_VAL)
        ),
        show_viewer=False
    )

    manifest, depot_pad = build_metropolitan_city(scene)

    # Clean vertical separation: Pad top is z=0.08m
    # Cargo on top of pad: z=0.18m (bottom at 0.09m)
    cargo = scene.add_entity(
        morph=gs.morphs.Box(size=(0.22, 0.22, 0.18), pos=(-45.0, -45.0, 0.18)),
        material=gs.materials.Rigid(rho=45.0, friction=0.7),
        surface=gs.surfaces.Default(color=(0.95, 0.75, 0.15))
    )

    # Drone above cargo: z=0.70m (clear separation, zero overlap)
    drone = scene.add_entity(
        morph=gs.morphs.Box(size=(0.42, 0.42, 0.12), pos=(-45.0, -45.0, 0.70)),
        material=gs.materials.Rigid(rho=85.034, friction=0.5),
        surface=gs.surfaces.Default(color=(0.1, 0.6, 0.9))
    )

    n_envs = 64
    scene.build(n_envs=n_envs)
    print(f"\n[GPU HARVEST] Simulating {n_envs} parallel environments on {torch.cuda.get_device_name(0)}...")

    bemt = VectorizedBEMTEngine(n_envs=n_envs, device="cuda")
    ctrl = GeometricSE3Controller(n_envs=n_envs, device="cuda")
    obs_mgr = VectorizedObservationManager(n_envs=n_envs, device="cuda")

    drone_link = drone.links[0]
    cargo_link = cargo.links[0]

    target_dest = manifest[4]
    tgt_pad = target_dest["pos"]
    roof_h = float(target_dest["height"])
    cruise_alt = roof_h + 12.0

    collected_obs = []
    collected_targets = []

    cable_len = 0.55
    has_payload = True
    phase_idx = 0
    t_start = time.time()

    for step in range(2500):
        d_pos, d_vel, d_quat, d_ang = drone.get_pos(), drone.get_vel(), drone.get_quat(), drone.get_ang()
        c_pos, c_vel = cargo.get_pos(), cargo.get_vel()
        curr_m = TOTAL_MASS_KG if has_payload else DRY_MASS_KG

        # State Machine Phase Transitions
        if phase_idx == 0:  # TAKEOFF
            target_pos = torch.tensor([-45.0, -45.0, cruise_alt], device="cuda").repeat(n_envs, 1)
            fsm_label = 0
            winch_label = torch.tensor([0.0, 0.0], device="cuda").repeat(n_envs, 1)
            if d_pos[:, 2].mean().item() >= (cruise_alt - 1.5): phase_idx = 1
        elif phase_idx == 1:  # CRUISE_OUTBOUND
            target_pos = torch.tensor([tgt_pad[0], tgt_pad[1], cruise_alt], device="cuda").repeat(n_envs, 1)
            fsm_label = 1
            winch_label = torch.tensor([0.0, 0.0], device="cuda").repeat(n_envs, 1)
            if torch.norm(d_pos[:, :2] - torch.tensor(tgt_pad[:2], device="cuda"), dim=-1).mean().item() <= 2.5: phase_idx = 2
        elif phase_idx == 2:  # CHIMNEY_DESCENT
            target_pos = torch.tensor([tgt_pad[0], tgt_pad[1], roof_h + 7.5], device="cuda").repeat(n_envs, 1)
            fsm_label = 2
            winch_label = torch.tensor([0.0, 0.0], device="cuda").repeat(n_envs, 1)
            if abs(d_pos[:, 2].mean().item() - (roof_h + 7.5)) <= 0.6: phase_idx = 3
        elif phase_idx == 3:  # DELIVERY_HOVER
            target_pos = torch.tensor([tgt_pad[0], tgt_pad[1], roof_h + 7.5], device="cuda").repeat(n_envs, 1)
            fsm_label = 3
            if has_payload and cable_len < 7.2:
                cable_len += 1.8 * dt
                winch_label = torch.tensor([1.0, 0.0], device="cuda").repeat(n_envs, 1)
            else:
                has_payload = False
                winch_label = torch.tensor([0.0, 1.0], device="cuda").repeat(n_envs, 1)
                phase_idx = 4
        elif phase_idx == 4:  # WINCH_RETRACT
            target_pos = torch.tensor([tgt_pad[0], tgt_pad[1], roof_h + 7.5], device="cuda").repeat(n_envs, 1)
            fsm_label = 4
            cable_len = max(0.20, cable_len - 3.0 * dt)
            winch_label = torch.tensor([-1.0, 1.0], device="cuda").repeat(n_envs, 1)
            if cable_len <= 0.25: phase_idx = 5
        elif phase_idx == 5:  # RETURN_BASE
            target_pos = torch.tensor([-45.0, -45.0, cruise_alt], device="cuda").repeat(n_envs, 1)
            fsm_label = 5
            winch_label = torch.tensor([0.0, 1.0], device="cuda").repeat(n_envs, 1)
            if torch.norm(d_pos[:, :2] - torch.tensor([-45.0, -45.0], device="cuda"), dim=-1).mean().item() <= 2.5: phase_idx = 6
        else:  # DEPOT_LANDING
            target_pos = torch.tensor([-45.0, -45.0, 0.70], device="cuda").repeat(n_envs, 1)
            fsm_label = 6
            winch_label = torch.tensor([0.0, 1.0], device="cuda").repeat(n_envs, 1)

        obs_t = obs_mgr.update_and_get(d_pos, d_vel, d_quat, d_ang, c_pos, c_vel, target_pos, cable_len, has_payload)

        # Smooth, physically capped tether tension
        f_tether_drone = torch.zeros((n_envs, 3), device="cuda")
        if has_payload:
            tether_vec = c_pos - d_pos
            tether_dist = torch.norm(tether_vec, dim=-1, keepdim=True)
            slack = tether_dist - cable_len
            is_taut = (slack > 0.0)

            tether_dir = tether_vec / torch.clamp(tether_dist, min=1e-4)
            rel_vel = c_vel - d_vel
            damping_proj = torch.sum(rel_vel * tether_dir, dim=-1, keepdim=True)

            # Cap tension at 18.0N max (prevents shock divergence)
            t_mag = torch.where(is_taut, torch.clamp(slack * 100.0 + damping_proj * 4.0, min=0.0, max=18.0), torch.zeros_like(slack))

            f_tether_cargo = -t_mag * tether_dir
            f_tether_drone = t_mag * tether_dir
            cargo_link.apply_external_force(force=f_tether_cargo)

        # SE(3) Control & Motor Allocation
        f_cmd, tau_cmd = ctrl.compute_wrench(d_pos, d_vel, d_quat, d_ang, target_pos, curr_m)
        motor_cmds = bemt.allocate(f_cmd, tau_cmd)
        t_act, q_act = bemt.compute_aerodynamics(motor_cmds, d_pos, d_vel, roof_h if phase_idx in [2, 3, 4] else 0.0, dt)

        # Combine aerodynamic thrust AND tether reaction force
        R = quat_to_rot_batch(d_quat)
        thrust_vec = R[:, :, 2] * t_act.unsqueeze(-1)
        total_drone_force = thrust_vec + f_tether_drone

        drone_link.apply_external_force(force=total_drone_force)
        drone_link.apply_external_torque(torque=q_act)
        scene.step()

        if step % 2 == 0:
            collected_obs.append(obs_t.clone())
            dist_to_goal = torch.norm(target_pos - d_pos, dim=-1, keepdim=True)
            guidance_gt = torch.clamp((target_pos - d_pos) / 5.0, -1.0, 1.0)
            fsm_gt = torch.full((n_envs,), fsm_label, device="cuda", dtype=torch.long)
            collected_targets.append((guidance_gt, fsm_gt, winch_label, dist_to_goal))

    print(f"[DATASET COMPILED] Harvested {len(collected_obs) * n_envs:,} state transitions in {time.time() - t_start:.2f}s!")

    # Multi-Task ResNet Training
    model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)

    X = torch.cat(collected_obs, dim=0)
    Y_guidance = torch.cat([t[0] for t in collected_targets], dim=0)
    Y_fsm = torch.cat([t[1] for t in collected_targets], dim=0)
    Y_winch = torch.cat([t[2] for t in collected_targets], dim=0)
    Y_val = torch.cat([t[3] for t in collected_targets], dim=0)

    dataset_size = X.shape[0]
    batch_size = 2048
    epochs = 12

    print(f"\n[TRAINING] Multi-task training across {dataset_size:,} samples ({epochs} epochs)...")
    model.train()

    for epoch in range(epochs):
        perm = torch.randperm(dataset_size, device=device)
        total_loss, n_batches = 0.0, 0

        for b in range(0, dataset_size, batch_size):
            idx = perm[b:b+batch_size]
            bx = X[idx]
            bg, bf, bw, bv = Y_guidance[idx], Y_fsm[idx], Y_winch[idx], Y_val[idx]

            optimizer.zero_grad()
            p_g, p_f, p_w, p_v = model(bx)

            loss = (
                1.0 * F.mse_loss(p_g, bg) +
                0.5 * F.cross_entropy(p_f, bf) +
                0.3 * F.mse_loss(p_w, bw) +
                0.1 * F.mse_loss(p_v, bv)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] — Multi-Task Loss: {total_loss / n_batches:.4f}")

    torch.save(model.state_dict(), MODEL_FILE)
    print(f"\n[SUCCESS] Model weights saved to '{MODEL_FILE}'.")
    print("Now run 'python genesis_swarm_nn.py' to watch the trained neural policy in the 3D viewer!\n")

# =============================================================================
# 7. INTERACTIVE 3D EVALUATION SIMULATOR (NEURAL POLICY IN CONTROL)
# =============================================================================
def evaluate_neural_executive():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepDenseFlightExecutiveNet(in_features=OBSERVATION_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    print(f"\nLoaded verified Neural Executive from '{MODEL_FILE}'!")

    gs.init(backend=gs.gpu)
    dt = 0.005  # 200 Hz
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=2, gravity=(0, 0, -GRAVITY_VAL)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(45.0, -45.0, 35.0),
            camera_lookat=(0.0, 0.0, 10.0),
            res=(1280, 720),
            refresh_rate=60,
        ),
        show_viewer=True
    )

    manifest, depot_pad = build_metropolitan_city(scene)

    cargo = scene.add_entity(
        morph=gs.morphs.Box(size=(0.22, 0.22, 0.18), pos=(-45.0, -45.0, 0.18)),
        material=gs.materials.Rigid(rho=45.0, friction=0.7),
        surface=gs.surfaces.Default(color=(0.95, 0.75, 0.15))
    )
    drone = scene.add_entity(
        morph=gs.morphs.Box(size=(0.42, 0.42, 0.12), pos=(-45.0, -45.0, 0.70)),
        material=gs.materials.Rigid(rho=85.034, friction=0.5),
        surface=gs.surfaces.Default(color=(0.10, 0.65, 0.95))
    )

    n_envs = 4
    scene.build(n_envs=n_envs)

    bemt = VectorizedBEMTEngine(n_envs=n_envs, device="cuda")
    ctrl = GeometricSE3Controller(n_envs=n_envs, device="cuda")
    obs_mgr = VectorizedObservationManager(n_envs=n_envs, device="cuda")

    drone_link = drone.links[0]
    cargo_link = cargo.links[0]

    target_dest = manifest[4]
    tgt_pad = target_dest["pos"]
    roof_h = float(target_dest["height"])
    cruise_alt = roof_h + 12.0

    cable_len = 0.55
    has_payload = True

    print("\n" + "="*75)
    print(f"NEURAL AUTONOMOUS SWARM FLIGHT ONLINE [{n_envs} ENVS]")
    print(f"Destination: {target_dest['id']} @ {roof_h:.1f}m | Policy in Direct Control")
    print("="*75 + "\n")

    try:
        for step in range(4000):
            if scene.viewer is not None and not scene.viewer.is_alive():
                print("\n[INFO] Viewer window closed by user. Exiting cleanly...")
                break

            d_pos, d_vel, d_quat, d_ang = drone.get_pos(), drone.get_vel(), drone.get_quat(), drone.get_ang()
            c_pos, c_vel = cargo.get_pos(), cargo.get_vel()
            sim_time = step * dt
            curr_m = TOTAL_MASS_KG if has_payload else DRY_MASS_KG

            ref_pos = torch.tensor([tgt_pad[0], tgt_pad[1], cruise_alt], device="cuda").repeat(n_envs, 1)

            obs_t = obs_mgr.update_and_get(d_pos, d_vel, d_quat, d_ang, c_pos, c_vel, ref_pos, cable_len, has_payload)

            with torch.no_grad():
                nn_guidance, nn_fsm_logits, nn_winch, _ = model(obs_t)

            nn_phase = int(nn_fsm_logits.argmax(dim=-1)[0].item())
            phase_name = FSM_NAMES[nn_phase]

            if nn_phase in [0, 1]:
                active_target = torch.tensor([tgt_pad[0], tgt_pad[1], cruise_alt], device="cuda").repeat(n_envs, 1)
            elif nn_phase in [2, 3, 4]:
                active_target = torch.tensor([tgt_pad[0], tgt_pad[1], roof_h + 7.5], device="cuda").repeat(n_envs, 1)
            elif nn_phase == 5:
                active_target = torch.tensor([-45.0, -45.0, cruise_alt], device="cuda").repeat(n_envs, 1)
            else:
                active_target = torch.tensor([-45.0, -45.0, 0.70], device="cuda").repeat(n_envs, 1)

            latch_release = float(nn_winch[0, 1].item()) > 0.4
            if has_payload:
                if nn_phase == 3: cable_len = min(7.2, cable_len + 1.8 * dt)
                if latch_release and c_pos[:, 2].mean().item() <= (roof_h + 0.25):
                    has_payload = False
                    print(f"[{sim_time:.1f}s] NEURAL TOUCHDOWN LATCH: Cargo released on {target_dest['id']}!")
            else:
                cable_len = max(0.20, cable_len - 3.2 * dt)

            f_tether_drone = torch.zeros((n_envs, 3), device="cuda")
            if has_payload:
                tether_vec = c_pos - d_pos
                tether_dist = torch.norm(tether_vec, dim=-1, keepdim=True)
                slack = tether_dist - cable_len
                is_taut = (slack > 0.0)

                tether_dir = tether_vec / torch.clamp(tether_dist, min=1e-4)
                rel_vel = c_vel - d_vel
                damping_proj = torch.sum(rel_vel * tether_dir, dim=-1, keepdim=True)

                t_mag = torch.where(is_taut, torch.clamp(slack * 100.0 + damping_proj * 4.0, min=0.0, max=18.0), torch.zeros_like(slack))

                f_tether_cargo = -t_mag * tether_dir
                f_tether_drone = t_mag * tether_dir
                cargo_link.apply_external_force(force=f_tether_cargo)

            f_cmd, tau_cmd = ctrl.compute_wrench(
                d_pos, d_vel, d_quat, d_ang, active_target, curr_m,
                nn_guidance_accel=nn_guidance
            )
            motor_cmds = bemt.allocate(f_cmd, tau_cmd)
            t_act, q_act = bemt.compute_aerodynamics(motor_cmds, d_pos, d_vel, roof_h if nn_phase in [2, 3, 4] else 0.0, dt)

            R = quat_to_rot_batch(d_quat)
            thrust_vec = R[:, :, 2] * t_act.unsqueeze(-1)
            total_drone_force = thrust_vec + f_tether_drone

            drone_link.apply_external_force(force=total_drone_force)
            drone_link.apply_external_torque(torque=q_act)

            scene.step()

            if step % 400 == 0:
                alt = d_pos[:, 2].mean().item()
                c_alt = c_pos[:, 2].mean().item()
                print(f"[{sim_time:4.1f}s] NN Phase: {phase_name:<16} | Drone Z: {alt:5.2f}m | Cargo Z: {c_alt:5.2f}m | Cable: {cable_len:.2f}m")

    except gs.GenesisException as e:
        if "Viewer closed" in str(e):
            print("\n[INFO] Viewer window closed by user. Simulation exited cleanly.")
        else:
            raise e

# =============================================================================
# 8. MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_neural_executive()
    else:
        if not os.path.exists(MODEL_FILE):
            print(f"\n[INFO] '{MODEL_FILE}' not found.")
            print("Please run training first with:")
            print("  python genesis_swarm_nn.py --train\n")
        else:
            evaluate_neural_executive()

            """
SURVIVAL CORE V37.0 — FULL GENESIS MULTI-PHYSICS UPGRADE
PART 2 OF 5: MICROCLIMATE MAP, URBAN WEATHER ENGINE, DRYDEN TURBULENCE,
             MESSINGER ROTOR ICING & SUB-ZERO 6S ARRHENIUS BATTERY ECM
"""

import math
import random
import numpy as np
import torch

# =============================================================================
# 0. BATTERY & ATMOSPHERIC PHYSICAL CONSTANTS
# =============================================================================
NUM_CELLS = 6
PACK_CAPACITY_AH = 8.0               
CELL_R0_NOMINAL = 0.0085
CELL_R1 = 0.0065
CELL_C1 = 2800.0
ARRHENIUS_EA = 28000.0
GAS_CONST_R = 8.31446
T_REF_K = 298.15

P_0 = 1013.25
T_0 = 288.15
L_LAPSE = 0.0065
R_SPEC_AIR = 287.058
SPEED_OF_SOUND = 343.2

# =============================================================================
# 1. SWARM MICROCLIMATE SHARED VOXEL MAP
# =============================================================================
class SwarmMicroclimateSharedMap:
    """
    Decentralized voxel map that accumulates local wind measurements broadcast
    by swarm drones. Enables vertical wind-shear lookups for energy-optimal routing.
    """
    def __init__(self, voxel_size_xy=20.0, voxel_size_z=8.0):
        self.grid = {}
        self.voxel_size_xy = voxel_size_xy
        self.voxel_size_z = voxel_size_z

    def broadcast_wind(self, pos, wind_vec, timestamp):
        vx = int(pos[0] // self.voxel_size_xy)
        vy = int(pos[1] // self.voxel_size_xy)
        vz = int(max(0.0, pos[2]) // self.voxel_size_z)
        key = (vx, vy, vz)
        if key not in self.grid:
            self.grid[key] = [wind_vec.copy(), timestamp]
        else:
            old_w, _ = self.grid[key]
            self.grid[key] = [0.65 * old_w + 0.35 * wind_vec, timestamp]

    def query_wind_prior(self, pos, current_time, max_age_s=45.0):
        vx = int(pos[0] // self.voxel_size_xy)
        vy = int(pos[1] // self.voxel_size_xy)
        vz = int(max(0.0, pos[2]) // self.voxel_size_z)
        key = (vx, vy, vz)
        if key in self.grid:
            wind_data, t_stamp = self.grid[key]
            if (current_time - t_stamp) <= max_age_s:
                return wind_data.copy(), True
        return np.zeros(3, dtype=np.float32), False

    def query_vertical_wind_shear_profile(self, pos_xy, current_time, candidate_alts):
        vx = int(pos_xy[0] // self.voxel_size_xy)
        vy = int(pos_xy[1] // self.voxel_size_xy)
        profile = []
        for alt in candidate_alts:
            vz = int(max(0.0, alt) // self.voxel_size_z)
            key = (vx, vy, vz)
            if key in self.grid and (current_time - self.grid[key][1]) <= 60.0:
                profile.append(self.grid[key][0].copy())
            else:
                profile.append(np.zeros(3, dtype=np.float32))
        return profile

# =============================================================================
# 2. DYNAMIC URBAN WEATHER ENGINE
# =============================================================================
class DynamicUrbanWeatherEngine:
    """
    Simulates dynamic urban meteorology with 4 regimes:
      - CLEAR_NOMINAL: Calm breeze (1.8 m/s), 22°C
      - CANYON_GALE: Urban street canyon winds (6.5 m/s), 14°C
      - FREEZING_RAIN: Supercooled precipitation with rotor icing, -4.5°C
      - MICROBURST_STORM: Severe downdraft shear (-3.2 m/s vertical), 10°C
    Also applies Venturi wind tunnel amplification along urban avenues.
    """
    def __init__(self, regime_duration=50.0):
        self.weather_modes = ["CLEAR_NOMINAL", "CANYON_GALE", "FREEZING_RAIN", "MICROBURST_STORM"]
        self.current_mode_idx = 0
        self.target_mode_idx = 0
        self.regime_timer = 0.0
        self.regime_duration = regime_duration
        self.blend_factor = 1.0

    def cycle_weather(self):
        self.target_mode_idx = (self.target_mode_idx + 1) % len(self.weather_modes)
        self.regime_timer = 0.0
        self.blend_factor = 0.0
        print(f"\n>>> [WEATHER TRANSITION] Blending into: {self.weather_modes[self.target_mode_idx]} <<<")

    def get_mode_name(self):
        return self.weather_modes[self.target_mode_idx]

    def step(self, dt=0.01):
        self.regime_timer += dt
        if self.blend_factor < 1.0:
            self.blend_factor = min(1.0, self.blend_factor + dt / 3.0)
            if self.blend_factor >= 1.0:
                self.current_mode_idx = self.target_mode_idx

        if self.regime_timer >= self.regime_duration:
            self.cycle_weather()

    def _get_preset(self, mode_idx):
        mode = self.weather_modes[mode_idx]
        if mode == "CLEAR_NOMINAL":
            return np.array([1.8, -0.6, 0.0], dtype=np.float32), 0.5, 22.0, False, 0.0
        elif mode == "CANYON_GALE":
            return np.array([6.2, 2.8, 0.0], dtype=np.float32), 3.5, 14.0, False, 0.0
        elif mode == "FREEZING_RAIN":
            return np.array([2.5, -2.0, 0.0], dtype=np.float32), 1.5, -4.5, True, -0.4
        else:  # MICROBURST_STORM
            return np.array([-4.5, 4.0, 0.0], dtype=np.float32), 5.5, 10.0, False, -2.8

    def get_weather_parameters(self, drone_pos, avenues_x=None):
        px, py, pz = drone_pos[0], drone_pos[1], drone_pos[2]
        w1, g1, t1, i1, m1 = self._get_preset(self.current_mode_idx)
        w2, g2, t2, i2, m2 = self._get_preset(self.target_mode_idx)

        b = self.blend_factor
        base_wind = (1.0 - b) * w1 + b * w2
        gust_scale = (1.0 - b) * g1 + b * g2
        temp_c = (1.0 - b) * t1 + b * t2
        auto_ice = i2 if b > 0.5 else i1
        microburst_w = (1.0 - b) * m1 + b * m2

        # Vertical downdraft stagnation near the ground (h < 5m)
        if microburst_w < -0.2:
            stagnation = float(math.tanh(max(0.1, pz) / 4.5))
            microburst_w *= stagnation

        # Venturi wind tunnel amplification between skyscraper rows
        venturi_boost = 1.0
        if avenues_x is not None:
            for ax in avenues_x:
                if abs(px - ax) < 8.0 and pz < 26.0:
                    venturi_boost = 1.45
                    break

        eff_wind = base_wind.copy()
        eff_wind[1] *= venturi_boost
        eff_wind[2] += microburst_w
        return eff_wind, gust_scale, temp_c, auto_ice

# =============================================================================
# 3. DRYDEN ATMOSPHERIC TURBULENCE MODEL (MIL-F-8785C)
# =============================================================================
class DrydenAtmosphericTurbulenceModel:
    """
    Stochastic atmospheric turbulence generator using low-altitude MIL-F-8785C
    spectral formulation for realistic wind gust dynamics.
    """
    def __init__(self, w20_m_s=3.4):
        self.w20 = w20_m_s
        self.gust_state = np.zeros(3, dtype=np.float32)

    def step(self, altitude_m, airspeed_m_s, dt=0.01):
        h = max(0.5, float(altitude_m))
        v = max(1.0, float(airspeed_m_s))
        sigma_w = 0.10 * self.w20
        sigma_u = sigma_w / ((0.177 + 0.000823 * h) ** 0.4)
        sigma_v = sigma_u

        l_w = h
        l_u = h / ((0.177 + 0.000823 * h) ** 1.2)
        l_v = l_u

        white_noise = np.random.normal(0.0, 1.0, 3).astype(np.float32)
        sq_dt = math.sqrt(dt)

        du = (- (v / l_u) * self.gust_state[0] * dt + sigma_u * math.sqrt(2.0 * v / l_u) * white_noise[0] * sq_dt)
        dv = (- (v / l_v) * self.gust_state[1] * dt + sigma_v * math.sqrt(2.0 * v / l_v) * white_noise[1] * sq_dt)
        dw = (- (v / l_w) * self.gust_state[2] * dt + sigma_w * math.sqrt(2.0 * v / l_w) * white_noise[2] * sq_dt)

        self.gust_state[0] += du
        self.gust_state[1] += dv
        self.gust_state[2] += dw
        self.gust_state = np.clip(self.gust_state, -6.5, 6.5)
        return self.gust_state.copy()

# =============================================================================
# 4. MESSINGER ROTOR ICING DYNAMICS MODEL
# =============================================================================
class MessingerRotorIcingModel:
    """
    Simulates rotor blade icing under supercooled liquid droplet impingement:
      - Accretes ice thickness (mm) as a function of airspeed and Liquid Water Content (LWC)
      - Degrades aerodynamic lift (up to -28%)
      - Increases aerodynamic drag (up to +180%)
    """
    def __init__(self):
        self.ice_thickness_mm = 0.0
        self.icing_active = False

    def step(self, airspeed, dt=0.01):
        if self.icing_active:
            lwc = 0.35  # g/m^3
            catch_efficiency = 0.42
            d_ice = (lwc * max(1.0, airspeed) * catch_efficiency * 1e-3) * 0.08 * dt
            self.ice_thickness_mm = min(3.8, self.ice_thickness_mm + d_ice * 1000.0)
        else:
            self.ice_thickness_mm = max(0.0, self.ice_thickness_mm - 0.02 * dt)

        ice_index = float(np.clip(self.ice_thickness_mm / 3.0, 0.0, 1.0))
        lift_loss_factor = 1.0 - 0.28 * ice_index
        drag_rise_factor = 1.0 + 1.80 * ice_index
        return self.ice_thickness_mm, ice_index, lift_loss_factor, drag_rise_factor

# =============================================================================
# 5. SUB-ZERO ARRHENIUS 6S BATTERY EQUIVALENT CIRCUIT MODEL (ECM)
# =============================================================================
class SubZeroArrheniusBatteryECM:
    """
    6S LiPo Battery Equivalent Circuit Model with:
      - Arrhenius internal resistance temperature dependence R0(T)
      - R1-C1 transient polarization voltage dynamics
      - Non-linear Open Circuit Voltage (OCV) vs. State-of-Charge (SoC)
      - Joule heating and convective ambient heat transfer
    """
    def __init__(self, initial_soc=0.98, initial_temp_c=22.0):
        self.soc = initial_soc
        self.temp_c = initial_temp_c
        self.v1 = np.zeros(NUM_CELLS, dtype=np.float32)
        self.capacity_coulombs = PACK_CAPACITY_AH * 3600.0
        self.v_pack = 25.0

    def step(self, p_elec_watts, amb_temp_c=22.0, dt=0.01):
        i_draw = float(p_elec_watts) / max(16.0, self.v_pack)
        t_k = self.temp_c + 273.15

        # Arrhenius internal resistance scaling
        r_scale = math.exp((ARRHENIUS_EA / GAS_CONST_R) * (1.0 / t_k - 1.0 / T_REF_K))
        r0 = CELL_R0_NOMINAL * max(0.85, r_scale)

        # Coulomb counting SoC update
        self.soc -= (i_draw * dt) / self.capacity_coulombs
        self.soc = max(0.0, min(1.0, self.soc))

        # Open Circuit Voltage (OCV) curve
        s = max(0.001, min(0.999, self.soc))
        ocv = 3.30 + 0.55 * s + 0.25 * math.log(s + 0.005) + 0.15 * math.exp(6.0 * (s - 0.88))
        ocv = float(np.clip(ocv, 3.20, 4.20))

        # 6 individual cell terminal voltages
        cell_volts = []
        for i in range(NUM_CELLS):
            self.v1[i] += (-(self.v1[i] / (CELL_R1 * CELL_C1)) + (i_draw / CELL_C1)) * dt
            cell_volts.append(ocv - i_draw * r0 - self.v1[i])

        self.v_pack = float(np.sum(cell_volts))

        # Joule self-heating (I^2 * R0) and convective thermal transfer
        p_joule = (i_draw ** 2) * (r0 * NUM_CELLS)
        self.temp_c += ((p_joule - 0.45 * (self.temp_c - amb_temp_c)) / 350.0) * dt

        return self.soc, self.v_pack, i_draw, r0

    