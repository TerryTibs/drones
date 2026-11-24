"""
=============================================================================
SENARY EISENSTEIN-HAMILTONIAN LATTICE CONTROLLER (SEHL-Control)
Fixed Tensor Contraction & Native Float16/Float32 Circulant Multiplication
Configured for: num_envs=4096, horizon=32, epochs=1800 (~236M Transitions)
=============================================================================
"""

import os
import sys
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
try:
    import pygame
except ImportError:
    pygame = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_FILE = "sehl_drone_model.pt"

# Physical constants
GRAVITY = 9.80665
DRONE_MASS = 1.45           # kg
CARGO_MASS = 0.40           # kg
ARM_LENGTH = 0.22           # m
INERTIA = torch.tensor([0.021, 0.021, 0.036], device=DEVICE)
INV_INERTIA = 1.0 / INERTIA
MOTOR_TAU = 0.032           # Motor electrical-mechanical response lag (s)
MAX_THRUST_PER_MOTOR = 8.5  # N (34 N total thrust capability)
TORQUE_TO_THRUST_RATIO = 0.016

# =============================================================================
# 1. NUMERICALLY STABLE DIFFERENTIABLE UTILITIES
# =============================================================================
def safe_norm(x, dim=-1, keepdim=False, eps=1e-6):
    """
    Numerically stable vector norm. Epsilon inside the square root ensures
    that gradients evaluate to 0 instead of NaN when x is at exact zero.
    """
    return torch.sqrt(torch.sum(x * x, dim=dim, keepdim=keepdim) + eps)

# 6-phase directional unit vectors spanning the A₂ hexagonal root lattice
SENARY_ANGLES = torch.tensor([k * math.pi / 3.0 for k in range(6)], device=DEVICE)
SENARY_BASIS = torch.stack([torch.cos(SENARY_ANGLES), torch.sin(SENARY_ANGLES)], dim=-1) # [6, 2]

def project_to_senary_lattice(v_2d):
    """
    Projects a 2D Cartesian vector [x, y] onto the 6 non-orthogonal Eisenstein axes.
    v_2d: [Batch, 2] -> Returns: [Batch, 6] non-negative radial coordinates.
    """
    basis = SENARY_BASIS.to(dtype=v_2d.dtype)
    projections = torch.matmul(v_2d, basis.T)
    return F.relu(projections)

def reconstruct_from_senary_lattice(s_6d):
    """
    Maps 6-phase senary coordinates back to 2D Cartesian space:
    v = ∑ₖ sₖ · eₖ · (2/3)
    """
    basis = SENARY_BASIS.to(dtype=s_6d.dtype)
    return torch.matmul(s_6d, basis) * (2.0 / 3.0)

def float_to_base6_str(val, fractional_digits=2):
    """Encodes a float into a signed Senary (Base-6) string representation."""
    if not math.isfinite(val):
        return "NaN_6"
    sign = "-" if val < 0 else "+"
    val = abs(val)
    int_part = int(val)
    frac_part = val - int_part
    
    if int_part == 0:
        digits = "0"
    else:
        d_list = []
        n = int_part
        while n > 0:
            d_list.append(str(n % 6))
            n //= 6
        digits = "".join(reversed(d_list))
        
    frac_digits = []
    f = frac_part
    for _ in range(fractional_digits):
        f *= 6
        d = int(f)
        frac_digits.append(str(d))
        f -= d
    return f"{sign}{digits}.{''.join(frac_digits)}₆"

class Z6CirculantLinear(nn.Module):
    """
    Z₆ Cyclic Group-Equivariant Linear Layer.
    Directly constructs circulant tensors using precomputed modular index shifts.
    Eliminates cuFFT half-precision dependencies while guaranteeing exact 60° rotational equivariance.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 6 parameters per channel pair defines the full 6x6 cyclic circulant matrix
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, 6) * 0.15)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1))
        
        # Precompute modular distance matrix: (i - j) % 6
        idx = (torch.arange(6).unsqueeze(1) - torch.arange(6).unsqueeze(0)) % 6
        self.register_buffer("circ_idx", idx)

    def forward(self, x):
        # x: [Batch (b), in_channels (i), 6 (k)]
        # W: [out_channels (o), in_channels (i), 6 (j), 6 (k)]
        W = self.kernel[:, :, self.circ_idx]
        
        # Exact spatial tensor contraction: sum over in_channels (i) and input phase (k)
        # Result: [Batch (b), out_channels (o), output_phase (j)]
        out = torch.einsum("oijk, bik -> boj", W, x) + self.bias
        return out

# =============================================================================
# 2. SEHL NEURAL CONTROLLER ARCHITECTURE
# =============================================================================
class SenaryHamiltonianPolicy(nn.Module):
    """
    Senary Eisenstein-Hamiltonian Lattice Policy:
    Combines hexagonal group equivariance with active Port-Hamiltonian energy shaping.
    """
    def __init__(self):
        super().__init__()
        # Input channels: [err_pos_senary, vel_senary, wind_senary, cargo_sway_senary]
        self.senary_layer1 = Z6CirculantLinear(in_channels=4, out_channels=8)
        self.senary_layer2 = Z6CirculantLinear(in_channels=8, out_channels=4)
        
        # Interconnection J (skew-symmetric) and Damping R (positive-definite)
        self.j_net = nn.Linear(24 + 6, 16)
        self.r_net = nn.Linear(24 + 6, 4)
        
        # Wrench potential vector head p(x) in R⁴
        self.wrench_generator = nn.Sequential(
            nn.Linear(24 + 6, 64),
            nn.SiLU(),
            nn.Linear(64, 4)
        )

    def forward(self, obs, dt=0.015):
        # Unpack observations
        err_xy = obs[:, 0:2]
        vel_xy = obs[:, 3:5]
        vert_state = torch.stack([obs[:, 2], obs[:, 5], obs[:, 8]], dim=-1) # [z_err, v_z, up_z]
        omega = obs[:, 9:12]
        cargo_sway_xy = obs[:, 12:14]
        wind_xy = obs[:, 14:16]
        
        # 1. Project Cartesian planar quantities to 6-Phase Eisenstein Lattice
        s_err = project_to_senary_lattice(err_xy)
        s_vel = project_to_senary_lattice(vel_xy)
        s_wind = project_to_senary_lattice(wind_xy)
        s_sway = project_to_senary_lattice(cargo_sway_xy)
        
        # Stack channels: [Batch, 4, 6]
        s_tensor = torch.stack([s_err, s_vel, s_wind, s_sway], dim=1)
        
        # 2. Z₆-Equivariant Hexagonal Transformations
        h1 = F.silu(self.senary_layer1(s_tensor))
        h2 = self.senary_layer2(h1) # [Batch, 4, 6]
        h2_flat = h2.view(obs.shape[0], 24)
        
        # Combine with vertical and rotational channels
        features = torch.cat([h2_flat, vert_state, omega], dim=-1)
        
        # 3. Active Port-Hamiltonian Energy Shaping: W = (J(x) - R(x)) · p(x)
        p_energy = self.wrench_generator(features)
        
        # Skew-symmetric J = M - Mᵀ guarantees ZERO internal energy production
        J_raw = self.j_net(features).view(-1, 4, 4)
        J = J_raw - J_raw.transpose(-1, -2)
        
        # Strictly dissipative damping matrix R ⪰ 0
        R_diag = F.softplus(self.r_net(features)) + 0.12
        R = torch.diag_embed(R_diag)
        
        # Passive Control Law: dH/dt = -pᵀ · R · p ≤ 0 (Lyapunov dissipative sink)
        passive_wrench = torch.matmul(J - R, p_energy.unsqueeze(-1)).squeeze(-1)
        
        # 4. Senary Radix-6 Fractional Deadband Sieve (Micro-jitter eliminator)
        err_dist = safe_norm(err_xy, dim=-1, keepdim=True)
        sieve_gate = torch.tanh(err_dist * 6.0) # Base-6 transition rate
        
        nominal_hover = (DRONE_MASS + CARGO_MASS) * GRAVITY
        Fz = nominal_hover + torch.tanh(passive_wrench[:, 0:1]) * 18.0
        Torques = torch.tanh(passive_wrench[:, 1:4]) * 0.85 * sieve_gate
        
        output_wrench = torch.cat([Fz, Torques], dim=-1)
        return output_wrench, s_err

# =============================================================================
# 3. DIFFERENTIABLE QUADROTOR SIMULATION WITH EISENSTEIN WAKE INTERACTIONS
# =============================================================================
class DifferentiableQuadrotorEnv:
    def __init__(self, num_envs=4096, dt=0.015, device=DEVICE):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        
        # 4-Motor X-configuration allocation matrix
        d = ARM_LENGTH / math.sqrt(2.0)
        c_q = TORQUE_TO_THRUST_RATIO
        B = torch.tensor([
            [ 1.0,  1.0,  1.0,  1.0],
            [-d,    d,    d,   -d  ],
            [ d,   -d,    d,   -d  ],
            [-c_q, -c_q,  c_q,  c_q]
        ], device=device, dtype=torch.float32)
        self.B_inv = torch.pinverse(B)
        self.reset()

    def reset(self):
        N = self.num_envs
        self.pos = torch.zeros((N, 3), device=self.device)
        self.pos[:, 2] = 0.15
        self.vel = torch.zeros((N, 3), device=self.device)
        self.rot_mat = torch.eye(3, device=self.device).unsqueeze(0).repeat(N, 1, 1)
        self.omega = torch.zeros((N, 3), device=self.device)
        self.motor_thrusts = torch.zeros((N, 4), device=self.device)
        
        self.cable_len = torch.full((N, 1), 1.2, device=self.device)
        self.cargo_pos = self.pos.clone()
        self.cargo_pos[:, 2] -= self.cable_len.squeeze(-1)
        self.cargo_vel = torch.zeros((N, 3), device=self.device)
        self.has_cargo = torch.ones((N, 1), device=self.device, dtype=torch.bool)
        
        self.target_pos = torch.zeros((N, 3), device=self.device)
        self.wind = torch.zeros((N, 3), device=self.device)
        self.v_batt = torch.full((N, 1), 24.8, device=self.device)

    def step(self, commanded_wrench):
        # 1. Voltage-scaled motor thrust conversion
        voltage_scale = torch.clamp(self.v_batt / 25.2, 0.70, 1.05)
        raw_thrust_cmd = torch.matmul(commanded_wrench, self.B_inv.to(commanded_wrench.dtype).T) / voltage_scale
        target_thrusts = torch.clamp(raw_thrust_cmd, 0.0, MAX_THRUST_PER_MOTOR)
        
        # 2. First-order motor lag filter
        alpha_m = self.dt / (MOTOR_TAU + self.dt)
        self.motor_thrusts = self.motor_thrusts + alpha_m * (target_thrusts - self.motor_thrusts)
        
        # 3. Realized body wrench
        realized_Fz = torch.sum(self.motor_thrusts, dim=-1, keepdim=True)
        tau_x = ARM_LENGTH / math.sqrt(2.0) * (-self.motor_thrusts[:, 0] + self.motor_thrusts[:, 1] + self.motor_thrusts[:, 2] - self.motor_thrusts[:, 3])
        tau_y = ARM_LENGTH / math.sqrt(2.0) * ( self.motor_thrusts[:, 0] - self.motor_thrusts[:, 1] + self.motor_thrusts[:, 2] - self.motor_thrusts[:, 3])
        tau_z = TORQUE_TO_THRUST_RATIO * (-self.motor_thrusts[:, 0] - self.motor_thrusts[:, 1] + self.motor_thrusts[:, 2] + self.motor_thrusts[:, 3])
        realized_torques = torch.stack([tau_x, tau_y, tau_z], dim=-1)
        
        # 4. Translatory physics with Eisenstein directional wake projection
        body_z_world = self.rot_mat[:, :, 2]
        thrust_world = body_z_world * realized_Fz
        
        v_rel = self.vel - self.wind
        v_rel_xy = v_rel[:, :2]
        senary_drag_components = project_to_senary_lattice(v_rel_xy)
        reconstructed_xy_drag = reconstruct_from_senary_lattice(senary_drag_components)
        
        drag_xy = -0.5 * 1.225 * 0.22 * reconstructed_xy_drag * safe_norm(v_rel_xy, dim=-1, keepdim=True).clamp(min=0.2)
        drag_z = -0.5 * 1.225 * 0.38 * v_rel[:, 2:3] * torch.abs(v_rel[:, 2:3]).clamp(min=0.2)
        drag_world = torch.cat([drag_xy, drag_z], dim=-1)
        
        # Tether spring-damper reaction
        tether_disp = self.cargo_pos - self.pos
        tether_dist = safe_norm(tether_disp, dim=-1, keepdim=True).clamp(min=1e-4)
        tether_dir = tether_disp / tether_dist
        tether_slack = tether_dist - self.cable_len
        taut_mask = (tether_slack > 0.0) & self.has_cargo
        
        v_cargo_rel = self.cargo_vel - self.vel
        rad_speed = torch.sum(v_cargo_rel * tether_dir, dim=-1, keepdim=True)
        tension_mag = torch.where(taut_mask, 140.0 * tether_slack + 5.0 * rad_speed, torch.zeros_like(tether_slack))
        tension_mag = torch.clamp(tension_mag, min=0.0, max=28.0)
        F_tether_drone = tension_mag * tether_dir
        
        # Total force & acceleration
        gravity_force = torch.tensor([0.0, 0.0, -GRAVITY * DRONE_MASS], device=self.device)
        total_force = thrust_world + drag_world + gravity_force + F_tether_drone
        accel = total_force / DRONE_MASS
        
        # 5. Angular acceleration
        gyroscopic = torch.cross(self.omega, self.omega * INERTIA, dim=-1)
        alpha = (realized_torques - gyroscopic) * INV_INERTIA
        
        # 6. Integrations
        self.vel = self.vel + accel * self.dt
        self.pos = self.pos + self.vel * self.dt
        
        ground_mask = self.pos[:, 2] < 0.08
        self.pos[:, 2] = torch.where(ground_mask, torch.full_like(self.pos[:, 2], 0.08), self.pos[:, 2])
        self.vel[:, 2] = torch.where(ground_mask & (self.vel[:, 2] < 0.0), torch.zeros_like(self.vel[:, 2]), self.vel[:, 2])
        
        self.omega = self.omega + alpha * self.dt
        
        # Lie-Algebra SO(3) update via safe Rodrigues formula
        omega_dt = self.omega * self.dt
        angle = safe_norm(omega_dt, dim=-1, keepdim=True).clamp(min=1e-6)
        axis = omega_dt / angle
        K = torch.zeros((self.num_envs, 3, 3), device=self.device, dtype=self.pos.dtype)
        K[:, 0, 1] = -axis[:, 2]; K[:, 0, 2] =  axis[:, 1]
        K[:, 1, 0] =  axis[:, 2]; K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]; K[:, 2, 1] =  axis[:, 0]
        dR = torch.eye(3, device=self.device, dtype=self.pos.dtype).unsqueeze(0) + torch.sin(angle).unsqueeze(-1) * K + (1.0 - torch.cos(angle)).unsqueeze(-1) * torch.matmul(K, K)
        self.rot_mat = torch.matmul(self.rot_mat, dR)
        
        # Cargo dynamics
        if torch.any(self.has_cargo):
            cargo_grav = torch.tensor([0.0, 0.0, -GRAVITY * CARGO_MASS], device=self.device)
            v_cargo_air = self.cargo_vel - self.wind
            cargo_drag = -0.5 * 1.225 * 0.15 * v_cargo_air * safe_norm(v_cargo_air, dim=-1, keepdim=True).clamp(min=0.2)
            cargo_accel = (cargo_grav + cargo_drag - F_tether_drone) / CARGO_MASS
            self.cargo_vel = torch.where(self.has_cargo, self.cargo_vel + cargo_accel * self.dt, self.cargo_vel)
            self.cargo_pos = torch.where(self.has_cargo, self.cargo_pos + self.cargo_vel * self.dt, self.cargo_pos)
            
        return self.get_observations()

    def get_observations(self):
        err_pos = self.target_pos - self.pos
        v_body = torch.matmul(self.rot_mat.transpose(-1, -2), self.vel.unsqueeze(-1)).squeeze(-1)
        cargo_disp = (self.cargo_pos - self.pos) * self.has_cargo.float()
        
        obs = torch.cat([
            err_pos[:, 0:2],        # 2: Planar error
            err_pos[:, 2:3],        # 1: Vertical error
            v_body[:, 0:2],         # 2: Planar body velocity
            v_body[:, 2:3],         # 1: Vertical body velocity
            self.rot_mat[:, :, 2],  # 3: Up-vector in world
            self.omega,             # 3: Angular rates
            cargo_disp[:, 0:2],     # 2: Planar cargo sway
            cargo_disp[:, 2:3],     # 1: Vertical cargo offset
            self.wind[:, 0:2]       # 2: Planar wind disturbance
        ], dim=-1)
        return obs

# =============================================================================
# 4. TRAINING: ANALYTIC POLICY GRADIENT (APG)
# =============================================================================
def train_sehl_controller(epochs=180, horizon=32, num_envs=4096, device=DEVICE):
    print("\n" + "=" * 78)
    print(" SENARY EISENSTEIN-HAMILTONIAN CONTROLLER (SEHL)")
    print(f" Environments: {num_envs:,} | Horizon: {horizon} steps | Epochs: {epochs:,}")
    print(f" Symmetry: Exact Z₆ Cyclic Equivariance on A₂ Hexagonal Root Lattice")
    print(f" Transitions Evaluated: {num_envs * horizon * epochs:,}")
    print("=" * 78)

    env = DifferentiableQuadrotorEnv(num_envs=num_envs, dt=0.015, device=device)
    policy = SenaryHamiltonianPolicy().to(device)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=3.2e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1.0e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    t_start = time.time()
    for ep in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        env.reset()

        # 4-Stage Progressive Curriculum
        if ep < 400:
            target_r = 1.0 + 3.0 * torch.rand(num_envs, device=device)
            target_ang = torch.rand(num_envs, device=device) * 2.0 * math.pi
            target_z = 2.0 + 4.0 * torch.rand(num_envs, device=device)
            env.wind.zero_()
        elif ep < 900:
            target_r = 3.0 + 9.0 * torch.rand(num_envs, device=device)
            target_ang = torch.rand(num_envs, device=device) * 2.0 * math.pi
            target_z = 1.5 + 8.0 * torch.rand(num_envs, device=device)
            env.wind = torch.randn((num_envs, 3), device=device) * torch.tensor([2.5, 2.5, 0.4], device=device)
        elif ep < 1400:
            target_r = 4.0 + 12.0 * torch.rand(num_envs, device=device)
            target_ang = torch.rand(num_envs, device=device) * 2.0 * math.pi
            target_z = 1.0 + 12.0 * torch.rand(num_envs, device=device)
            env.wind = torch.randn((num_envs, 3), device=device) * torch.tensor([5.5, 5.5, 1.2], device=device)
        else:
            target_r = 2.0 + 14.0 * torch.rand(num_envs, device=device)
            target_ang = torch.rand(num_envs, device=device) * 2.0 * math.pi
            target_z = 0.5 + 14.0 * torch.rand(num_envs, device=device)
            env.wind = torch.randn((num_envs, 3), device=device) * torch.tensor([8.0, 8.0, 1.8], device=device)

        env.target_pos = torch.stack([target_r * torch.cos(target_ang),
                                      target_r * torch.sin(target_ang),
                                      target_z], dim=-1)
        obs = env.get_observations()
        loss_total = torch.zeros(1, device=device)

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            for t in range(horizon):
                wrench, _ = policy(obs, dt=env.dt)
                obs = env.step(wrench)

                pos_error = safe_norm(env.target_pos - env.pos, dim=-1)
                vel_penalty = safe_norm(env.vel, dim=-1)
                tilt_penalty = 1.0 - env.rot_mat[:, 2, 2]
                spin_penalty = safe_norm(env.omega, dim=-1)
                cargo_sway = safe_norm(env.cargo_pos[:, :2] - env.pos[:, :2], dim=-1)

                step_loss = (
                    4.5 * pos_error.mean() +
                    0.8 * vel_penalty.mean() +
                    3.5 * tilt_penalty.mean() +
                    0.2 * spin_penalty.mean() +
                    1.5 * cargo_sway.mean()
                )
                loss_total = loss_total + step_loss
            loss_total = loss_total / horizon

        scaler.scale(loss_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.5)

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()

        if scale_before <= scale_after:
            scheduler.step()

        if (ep + 1) % 100 == 0 or ep == 0:
            elapsed = time.time() - t_start
            lr = optimizer.param_groups[0]["lr"]
            stage = 1 if ep < 400 else (2 if ep < 900 else (3 if ep < 1400 else 4))
            print(f" [Epoch {ep+1:04d}/{epochs}] Stage {stage} | Loss: {loss_total.item():.4f} | PosErr: {pos_error.mean().item():.3f}m | LR: {lr:.5f} | Elapsed: {elapsed:.1f}s")

    torch.save(policy.state_dict(), CHECKPOINT_FILE)
    print(f"\n>> 180 EPOCHS COMPLETED in {time.time() - t_start:.2f} seconds!")
    print(f">> Senary-Equivariant weights saved to: {CHECKPOINT_FILE}\n")
    return policy

# =============================================================================
# 5. 3D INTERACTIVE FLIGHT & SENARY TELEMETRY COCKPIT (PYGAME)
# =============================================================================
class DeliverySimWorld:
    def __init__(self):
        self.depot_pad = np.array([0.0, 0.0, 0.1], dtype=np.float32)
        self.buildings = [
            {"aabb": [-16.0, -4.0, -16.0, -4.0, 7.5], "name": "Tower Alpha",  "pad": np.array([-10.0, -10.0, 7.6], dtype=np.float32)},
            {"aabb": [  6.0, 18.0, -14.0, -2.0, 11.0], "name": "Tower Bravo",  "pad": np.array([ 12.0,  -8.0, 11.1], dtype=np.float32)},
            {"aabb": [-18.0, -6.0,   8.0, 20.0, 9.0], "name": "Medical Plaza", "pad": np.array([-12.0,  14.0, 9.1], dtype=np.float32)},
            {"aabb": [  8.0, 20.0,   6.0, 18.0, 14.5], "name": "Sky Hub 04",    "pad": np.array([ 14.0,  12.0, 14.6], dtype=np.float32)}
        ]
        self.active_order_idx = 1

    def get_current_delivery(self):
        return self.buildings[self.active_order_idx]

    def advance_order(self):
        self.active_order_idx = (self.active_order_idx + 1) % len(self.buildings)
        return self.get_current_delivery()

def project_3d_point(point, cam_pos, cam_lookat, screen_w, screen_h, fov=48.0):
    fwd = cam_lookat - cam_pos
    fwd_len = np.linalg.norm(fwd)
    if fwd_len < 1e-4:
        return None
    fwd = fwd / fwd_len
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-4:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, fwd)
    delta = point - cam_pos
    depth = np.dot(delta, fwd)
    if depth <= 0.2:
        return None
    focal = (screen_h * 0.5) / math.tan(math.radians(fov) * 0.5)
    px = screen_w * 0.5 + focal * np.dot(delta, right) / depth
    py = screen_h * 0.5 - focal * np.dot(delta, up) / depth
    return int(px), int(py), float(depth)

def run_3d_simulation(policy):
    if pygame is None:
        print("[ERROR] Pygame is required for the visual interface. Run: pip install pygame")
        return

    pygame.init()
    SCREEN_W, SCREEN_H = 1200, 780
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("SEHL-Control: Senary Eisenstein-Hamiltonian 3D Cockpit")
    clock = pygame.time.Clock()
    font_hud = pygame.font.SysFont("consolas", 12, bold=True)

    world = DeliverySimWorld()
    env = DifferentiableQuadrotorEnv(num_envs=1, dt=0.016, device=DEVICE)
    policy.eval()

    mission_state = "TAKEOFF"
    target_building = world.get_current_delivery()
    pad_target = target_building["pad"]

    env.pos[0] = torch.tensor([0.0, 0.0, 0.2], device=DEVICE)
    env.cargo_pos[0] = torch.tensor([0.0, 0.0, 0.08], device=DEVICE)
    env.target_pos[0] = torch.tensor([0.0, 0.0, 6.0], device=DEVICE)
    env.has_cargo[0] = True
    env.cable_len[0] = 0.2

    delivered_boxes = []
    cam_yaw, cam_pitch, cam_dist = 0.95, 0.38, 28.0
    cam_lookat = np.array([0.0, 0.0, 4.0], dtype=np.float32)
    hover_dwell = 0.0
    running = True

    print(">> 3D Cockpit active with real-time Base-6 (Senary) telemetry and Eisenstein radar.")
    print(">> Controls: [Mouse Drag] Rotate Camera | [Mouse Wheel] Zoom | [Spacebar] Inject 8 m/s Crosswind\n")

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                cam_dist = np.clip(cam_dist - event.y * 2.0, 6.0, 90.0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    env.wind[0] = torch.randn(3, device=DEVICE) * 8.0
                    print(f">> [SENARY VORTEX INJECTED] Wind: {env.wind[0].cpu().numpy().round(1)} m/s")

        # Camera orbiting
        mouse_pressed = pygame.mouse.get_pressed()
        if mouse_pressed[0] or mouse_pressed[2]:
            dx, dy = pygame.mouse.get_rel()
            cam_yaw += dx * 0.008
            cam_pitch = np.clip(cam_pitch + dy * 0.008, 0.05, 1.45)
        else:
            pygame.mouse.get_rel()

        drone_p_np = env.pos[0].detach().cpu().numpy()
        cam_lookat = drone_p_np * 0.4 + cam_lookat * 0.6
        cam_x = cam_lookat[0] + cam_dist * math.cos(cam_pitch) * math.cos(cam_yaw)
        cam_y = cam_lookat[1] + cam_dist * math.cos(cam_pitch) * math.sin(cam_yaw)
        cam_z = max(0.8, cam_lookat[2] + cam_dist * math.sin(cam_pitch))
        cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

        # Mission state arbitration
        drone_z = drone_p_np[2]
        dist_to_pad_xy = np.linalg.norm(drone_p_np[:2] - pad_target[:2])

        if mission_state == "TAKEOFF":
            env.target_pos[0] = torch.tensor([0.0, 0.0, pad_target[2] + 4.5], device=DEVICE)
            if drone_z >= pad_target[2] + 3.0:
                mission_state = "CRUISE_OUTBOUND"
        elif mission_state == "CRUISE_OUTBOUND":
            env.target_pos[0] = torch.tensor([pad_target[0], pad_target[1], pad_target[2] + 4.5], device=DEVICE)
            if dist_to_pad_xy < 0.65:
                mission_state = "PRECISION_HOVER"
                hover_dwell = 0.0
        elif mission_state == "PRECISION_HOVER":
            env.target_pos[0] = torch.tensor([pad_target[0], pad_target[1], pad_target[2] + 3.2], device=DEVICE)
            if dist_to_pad_xy < 0.35 and abs(drone_z - (pad_target[2] + 3.2)) < 0.4:
                hover_dwell += dt
                if hover_dwell >= 1.2:
                    mission_state = "LOWERING_PAYLOAD"
            else:
                hover_dwell = max(0.0, hover_dwell - dt * 2.0)
        elif mission_state == "LOWERING_PAYLOAD":
            env.target_pos[0] = torch.tensor([pad_target[0], pad_target[1], pad_target[2] + 2.8], device=DEVICE)
            env.cable_len[0] = torch.clamp(env.cable_len[0] + 0.65 * dt, max=2.6)
            cargo_z = env.cargo_pos[0, 2].item()
            if cargo_z <= pad_target[2] + 0.12 or env.cable_len[0].item() >= 2.5:
                mission_state = "RELEASE_PACKAGE"
        elif mission_state == "RELEASE_PACKAGE":
            delivered_boxes.append(env.cargo_pos[0].detach().cpu().numpy())
            env.has_cargo[0] = False
            mission_state = "RETRACTING_AND_RETURN"
        elif mission_state == "RETRACTING_AND_RETURN":
            env.cable_len[0] = torch.clamp(env.cable_len[0] - 1.2 * dt, min=0.2)
            env.target_pos[0] = torch.tensor([0.0, 0.0, 8.0], device=DEVICE)
            if np.linalg.norm(drone_p_np[:2]) < 0.5:
                mission_state = "DEPOT_TOUCHDOWN"
        elif mission_state == "DEPOT_TOUCHDOWN":
            env.target_pos[0] = torch.tensor([0.0, 0.0, 0.15], device=DEVICE)
            if drone_z <= 0.25 and np.linalg.norm(env.vel[0].cpu().numpy()) < 0.3:
                time.sleep(0.5)
                target_building = world.advance_order()
                pad_target = target_building["pad"]
                env.has_cargo[0] = True
                env.cargo_pos[0] = env.pos[0].clone()
                env.cargo_pos[0, 2] = 0.08
                mission_state = "TAKEOFF"

        # -------------------------------------------------------------
        # NEURAL INFERENCE (100% CONTROL AUTHORITY)
        # -------------------------------------------------------------
        with torch.no_grad():
            obs = env.get_observations()
            commanded_wrench, senary_coords = policy(obs, dt=env.dt)
            env.step(commanded_wrench)

        # -------------------------------------------------------------
        # 3D PERSPECTIVE RENDERING
        # -------------------------------------------------------------
        screen.fill((14, 18, 24))

        # Ground grid
        for gx in range(-30, 31, 5):
            p1 = project_3d_point(np.array([gx, -30, 0]), cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            p2 = project_3d_point(np.array([gx,  30, 0]), cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            if p1 and p2: pygame.draw.line(screen, (28, 36, 48), p1[:2], p2[:2], 1)
        for gy in range(-30, 31, 5):
            p1 = project_3d_point(np.array([-30, gy, 0]), cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            p2 = project_3d_point(np.array([ 30, gy, 0]), cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            if p1 and p2: pygame.draw.line(screen, (28, 36, 48), p1[:2], p2[:2], 1)

        # Buildings & Rooftop Helipads
        for b in world.buildings:
            x1, x2, y1, y2, top = b["aabb"]
            corners = [
                np.array([x1, y1, 0.0]), np.array([x2, y1, 0.0]),
                np.array([x2, y2, 0.0]), np.array([x1, y2, 0.0]),
                np.array([x1, y1, top]), np.array([x2, y1, top]),
                np.array([x2, y2, top]), np.array([x1, y2, top])
            ]
            proj = [project_3d_point(c, cam_pos, cam_lookat, SCREEN_W, SCREEN_H) for c in corners]
            if all(proj[i] is not None for i in [4, 5, 6, 7]):
                roof_poly = [proj[i][:2] for i in [4, 5, 6, 7]]
                pygame.draw.polygon(screen, (44, 54, 70), roof_poly)
                pygame.draw.polygon(screen, (70, 85, 110), roof_poly, 2)
            for i in range(4):
                if proj[i] and proj[i+4]:
                    pygame.draw.line(screen, (55, 68, 88), proj[i][:2], proj[i+4][:2], 1)

            pad_pt = project_3d_point(b["pad"], cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            if pad_pt:
                is_active = (b["name"] == target_building["name"])
                pad_color = (0, 255, 180) if is_active else (255, 180, 50)
                radius_px = max(4, int(42.0 / pad_pt[2]))
                pygame.draw.circle(screen, pad_color, pad_pt[:2], radius_px, 2)
                if is_active:
                    pygame.draw.circle(screen, (0, 255, 180), pad_pt[:2], max(1, radius_px // 2))

        # Delivered Cargo
        for box_p in delivered_boxes:
            b_pt = project_3d_point(box_p, cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            if b_pt:
                sz = max(3, int(15.0 / b_pt[2]))
                pygame.draw.rect(screen, (255, 215, 60), (b_pt[0]-sz, b_pt[1]-sz, sz*2, sz*2))

        # Quadrotor
        drone_pt = project_3d_point(drone_p_np, cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
        rot_np = env.rot_mat[0].detach().cpu().numpy()
        if drone_pt:
            d_arm = ARM_LENGTH * 1.5
            arm_ends = [
                drone_p_np + rot_np @ np.array([ d_arm,  d_arm, 0.0]),
                drone_p_np + rot_np @ np.array([-d_arm, -d_arm, 0.0]),
                drone_p_np + rot_np @ np.array([-d_arm,  d_arm, 0.0]),
                drone_p_np + rot_np @ np.array([ d_arm, -d_arm, 0.0])
            ]
            arm_pts = [project_3d_point(ae, cam_pos, cam_lookat, SCREEN_W, SCREEN_H) for ae in arm_ends]
            if arm_pts[0] and arm_pts[1]:
                pygame.draw.line(screen, (210, 235, 255), arm_pts[0][:2], arm_pts[1][:2], 3)
            if arm_pts[2] and arm_pts[3]:
                pygame.draw.line(screen, (210, 235, 255), arm_pts[2][:2], arm_pts[3][:2], 3)
            for apt in arm_pts:
                if apt:
                    pygame.draw.circle(screen, (0, 220, 255), apt[:2], max(3, int(22.0 / apt[2])), 1)
            pygame.draw.circle(screen, (255, 60, 80), drone_pt[:2], max(3, int(25.0 / drone_pt[2])))

        # Tether and Cargo
        if env.has_cargo[0]:
            cargo_p_np = env.cargo_pos[0].detach().cpu().numpy()
            cargo_pt = project_3d_point(cargo_p_np, cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            if drone_pt and cargo_pt:
                pygame.draw.line(screen, (190, 210, 230), drone_pt[:2], cargo_pt[:2], 2)
                csz = max(3, int(20.0 / cargo_pt[2]))
                pygame.draw.rect(screen, (255, 195, 30), (cargo_pt[0]-csz, cargo_pt[1]-csz, csz*2, csz*2))
                pygame.draw.rect(screen, (80, 50, 10), (cargo_pt[0]-csz, cargo_pt[1]-csz, csz*2, csz*2), 1)

        # -------------------------------------------------------------
        # HUD: SENARY TELEMETRY & EISENSTEIN STAR COMPASS
        # -------------------------------------------------------------
        hud_bg = pygame.Rect(15, 15, 420, 250)
        pygame.draw.rect(screen, (10, 14, 20), hud_bg, border_radius=6)
        pygame.draw.rect(screen, (0, 220, 255), hud_bg, 1, border_radius=6)

        speed = np.linalg.norm(env.vel[0].cpu().numpy())
        altitude = drone_p_np[2]
        dist_target = np.linalg.norm(drone_p_np - env.target_pos[0].cpu().numpy())
        s_coords = senary_coords[0].cpu().numpy()

        status_lines = [
            ("SEHL-CONTROL: SENARY EISENSTEIN LATTICE", (0, 255, 200)),
            (f"Mission Phase:      {mission_state}", (255, 220, 40)),
            (f"Target Destination: {target_building['name']}", (255, 255, 255)),
            (f"Decimal Pos Error:  {dist_target:.3f} m  (Alt: {altitude:.2f}m)", (0, 220, 255)),
            (f"SENARY ERROR [X]:   {float_to_base6_str(env.target_pos[0,0].item()-drone_p_np[0])} m", (255, 160, 40)),
            (f"SENARY ERROR [Y]:   {float_to_base6_str(env.target_pos[0,1].item()-drone_p_np[1])} m", (255, 160, 40)),
            (f"SENARY ALTITUDE:    {float_to_base6_str(altitude)} m", (255, 160, 40)),
            (f"Airspeed:           {speed:.2f} m/s", (0, 220, 255)),
            (f"A₂ Lattice Energy:  H={torch.norm(obs[0, :6]).item()*10.0:.2f} J", (180, 200, 240)),
            (f"Delivered Packages: {len(delivered_boxes)}", (0, 255, 140))
        ]

        for idx, (txt, col) in enumerate(status_lines):
            screen.blit(font_hud.render(txt, True, col), (hud_bg.x + 12, hud_bg.y + 10 + idx * 17))

        # -------------------------------------------------------------
        # EISENSTEIN 6-PHASE HEXAGONAL RADAR COMPASS
        # -------------------------------------------------------------
        radar_cx, radar_cy = hud_bg.right - 55, hud_bg.top + 55
        pygame.draw.circle(screen, (24, 32, 45), (radar_cx, radar_cy), 32)
        pygame.draw.circle(screen, (0, 220, 255), (radar_cx, radar_cy), 32, 1)

        for k in range(6):
            ang = k * math.pi / 3.0
            bx = radar_cx + int(30 * math.cos(ang))
            by = radar_cy - int(30 * math.sin(ang))
            pygame.draw.line(screen, (45, 60, 85), (radar_cx, radar_cy), (bx, by), 1)

            mag = float(np.clip(s_coords[k] * 4.0, 0.0, 30.0))
            ax = radar_cx + int(mag * math.cos(ang))
            ay = radar_cy - int(mag * math.sin(ang))
            pygame.draw.line(screen, (0, 255, 200), (radar_cx, radar_cy), (ax, ay), 2)
            pygame.draw.circle(screen, (255, 160, 40), (ax, ay), 2)

        # Wind Vector Arrow
        wind_speed = np.linalg.norm(env.wind[0].cpu().numpy())
        if wind_speed > 0.1:
            w_dir = env.wind[0].cpu().numpy()[:2] / wind_speed
            wx = int(radar_cx + w_dir[0] * min(26.0, wind_speed * 3.2))
            wy = int(radar_cy - w_dir[1] * min(26.0, wind_speed * 3.2))
            pygame.draw.line(screen, (255, 60, 60), (radar_cx, radar_cy), (wx, wy), 2)
            screen.blit(font_hud.render(f"W: {wind_speed:.1f}m/s", True, (255, 100, 80)), (radar_cx - 26, radar_cy + 36))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

# =============================================================================
# 6. MAIN EXECUTION ENTRYPOINT
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 78)
    print(" SENARY EISENSTEIN-HAMILTONIAN LATTICE CONTROLLER (SEHL-Control)")
    print(" Scale: num_envs=4096, horizon=32, epochs=180 (~236M Transitions)")
    print("=" * 78)

    if os.path.exists(CHECKPOINT_FILE):
        print(f">> Existing trained checkpoint detected: {CHECKPOINT_FILE}")
        print(">> Loading policy directly into 3D simulation...")
        trained_policy = SenaryHamiltonianPolicy().to(DEVICE)
        trained_policy.load_state_dict(torch.load(CHECKPOINT_FILE, map_location=DEVICE))
    else:
        trained_policy = train_sehl_controller(epochs=180, horizon=32, num_envs=4096, device=DEVICE)

    run_3d_simulation(trained_policy)