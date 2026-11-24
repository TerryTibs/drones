"""
=============================================================================
SENARY EISENSTEIN-HAMILTONIAN LATTICE CONTROLLER (SEHL-Control)
High-Throughput Configuration: num_envs=12288, horizon=128, epochs=512
Full Stack Implementation with Comprehensive Multi-Domain Fault Telemetry
Hardware Target: RTX 5060 Ti 16GB + Ryzen 5800X (Native Bfloat16 + TF32)
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
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
AMBIENT_TEMP_KELVIN = 293.15
MOTOR_COIL_RESISTANCE = 0.048 # Ohms

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
    """Projects 2D vectors onto the 6 non-orthogonal Eisenstein axes."""
    basis = SENARY_BASIS.to(dtype=v_2d.dtype)
    projections = torch.matmul(v_2d, basis.T)
    return F.relu(projections)

def reconstruct_from_senary_lattice(s_6d):
    """Maps 6-phase senary coordinates back to 2D Cartesian space."""
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
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, 6) * 0.15)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1))
        
        idx = (torch.arange(6).unsqueeze(1) - torch.arange(6).unsqueeze(0)) % 6
        self.register_buffer("circ_idx", idx)

    def forward(self, x):
        # x: [Batch (b), in_channels (i), 6 (k)]
        # W: [out_channels (o), in_channels (i), 6 (j), 6 (k)]
        W = self.kernel[:, :, self.circ_idx]
        out = torch.einsum("oijk, bik -> boj", W, x) + self.bias
        return out

# =============================================================================
# 2. PILLAR 2: WAVE SCATTERING PORT + ORBITAL MOMENTUM SIEVE
# =============================================================================
class WaveVariableTetherPort(nn.Module):
    """
    Scattering-Domain Wave-Variable Transmission Line with an Orbital Momentum Sieve.
    Drains rotational flail energy (r × v) while guaranteeing L₂-gain <= 1 across snap-loads.
    """
    def __init__(self, characteristic_impedance=4.5, gamma_whirl=3.2):
        super().__init__()
        self.b = characteristic_impedance
        self.gamma_whirl = gamma_whirl

    def forward(self, drone_velocity, cargo_velocity, cable_length, tether_displacement):
        center_distance = safe_norm(tether_displacement, dim=-1, keepdim=True)
        tether_dir = tether_displacement / center_distance.clamp(min=1e-4)
        slack = center_distance - cable_length
        is_taut = (slack > 0.0)

        # 1. Scattering Wave-Variable Line (Radial Passivity)
        spring_k = 120.0
        tension_nominal = torch.where(is_taut, spring_k * slack, torch.zeros_like(slack))

        v_rel_vec = cargo_velocity - drone_velocity
        v_rad = torch.sum(v_rel_vec * tether_dir, dim=-1, keepdim=True)

        sqrt_2b = math.sqrt(2.0 * self.b)
        u_wave = (tension_nominal + self.b * v_rad) / sqrt_2b
        v_wave = (tension_nominal - self.b * v_rad) / sqrt_2b

        # Passivity enforcement: wave power P = 0.5 * (|u|² - |v|²) >= 0
        v_wave_passive = torch.where(
            torch.abs(v_wave) > torch.abs(u_wave),
            torch.sign(v_wave) * torch.abs(u_wave),
            v_wave
        )

        passive_tension = (sqrt_2b * 0.5) * (u_wave + v_wave_passive)
        passive_tension = torch.clamp(passive_tension, min=0.0, max=25.0)

        # 2. Port-Hamiltonian Orbital Momentum Sieve (Anti-Whirl Port)
        # Selectively dissipates the tangential flailing component: (r × v) × r
        orbital_momentum = torch.cross(tether_displacement, v_rel_vec, dim=-1)
        r_sq = torch.sum(tether_displacement * tether_displacement, dim=-1, keepdim=True).clamp(min=1e-3)
        whirl_velocity = torch.cross(orbital_momentum, tether_displacement, dim=-1) / r_sq
        F_anti_whirl = -self.gamma_whirl * whirl_velocity

        wave_telemetry = torch.cat([u_wave, v_wave_passive], dim=-1)
        return passive_tension, F_anti_whirl, wave_telemetry, tether_dir, orbital_momentum, slack

# =============================================================================
# 3. PILLAR 3: THERMODYNAMIC ONSAGER LOSS WITH WEATHERVANE HEADING COUPLING
# =============================================================================
class ThermodynamicEntropyLoss(nn.Module):
    """
    Optimizes the Non-Equilibrium Thermodynamic Entropy Production Rate
    while enforcing weathervane heading convergence and cargo flail damping.
    """
    def __init__(self, t_ambient=AMBIENT_TEMP_KELVIN, motor_resistance=MOTOR_COIL_RESISTANCE):
        super().__init__()
        self.t_amb = t_ambient
        self.r_coil = motor_resistance

    def forward(self, relative_velocity, motor_thrusts, target_pos, current_pos, 
                rot_mat, omega, cargo_disp, cargo_vel, drone_vel, cos_psi_err):
        # 1. Aerodynamic Entropy Production: S_dot_aero = (1 / T_amb) * v · F_drag
        speed = safe_norm(relative_velocity, dim=-1, keepdim=True)
        drag_force = 0.5 * 1.225 * 0.25 * speed * relative_velocity
        s_dot_aero = torch.sum(relative_velocity * drag_force, dim=-1) / self.t_amb

        # 2. Motor Joulean Entropy Production: S_dot_joule = (1 / T_motor) * ∑ I² * R
        joulean_power = torch.sum(motor_thrusts * self.r_coil * 18.5, dim=-1)
        s_dot_joule = joulean_power / self.t_amb
        entropy_rate = (s_dot_aero + s_dot_joule).mean()

        # Kinematic tracking penalties
        pos_error = safe_norm(target_pos - current_pos, dim=-1).mean()
        vel_penalty = safe_norm(relative_velocity, dim=-1).mean()
        tilt_penalty = (1.0 - rot_mat[:, 2, 2]).mean()
        
        # Weathervane heading alignment penalty
        heading_penalty = (1.0 - cos_psi_err).mean()
        
        # Angular rate damping (heavily penalizing yaw spin)
        yaw_rate = torch.abs(omega[:, 2]).mean()
        total_rates = safe_norm(omega, dim=-1).mean()
        spin_penalty = total_rates + 3.0 * yaw_rate
        
        # Cargo sway offset and relative orbital flail velocity
        cargo_sway = safe_norm(cargo_disp, dim=-1).mean()
        cargo_whip = safe_norm(cargo_vel - drone_vel, dim=-1).mean()

        loss_pos_part = 4.50 * pos_error
        loss_vel_part = 0.85 * vel_penalty
        loss_tilt_part = 3.50 * tilt_penalty
        loss_head_part = 2.00 * heading_penalty
        loss_spin_part = 1.40 * spin_penalty
        loss_sway_part = 1.50 * cargo_sway + 1.20 * cargo_whip
        loss_sdot_part = 0.08 * entropy_rate

        total_loss = (
            loss_sdot_part +
            loss_pos_part +
            loss_vel_part +
            loss_tilt_part +
            loss_head_part +
            loss_spin_part +
            loss_sway_part
        )
        
        components = {
            "pos": loss_pos_part.detach(),
            "vel": loss_vel_part.detach(),
            "tilt": loss_tilt_part.detach(),
            "spin": loss_spin_part.detach(),
            "head": loss_head_part.detach(),
            "sway": loss_sway_part.detach(),
            "s_dot": loss_sdot_part.detach(),
            "s_dot_raw": entropy_rate.detach()
        }
        return total_loss, components

# =============================================================================
# 4. SEHL NEURAL CONTROLLER ARCHITECTURE (WEATHERVANE ATTRACTOR)
# =============================================================================
class SenaryHamiltonianPolicy(nn.Module):
    """
    Senary Eisenstein-Hamiltonian Lattice Policy:
    Actively shapes energy via Port-Hamiltonian routing and incorporates an
    Aerodynamic Weathervane Attractor to lock heading along the flight track.
    """
    def __init__(self):
        super().__init__()
        # Input channels: [err_pos_senary, vel_senary, wind_senary, cargo_sway_senary]
        self.senary_layer1 = Z6CirculantLinear(in_channels=4, out_channels=8)
        self.senary_layer2 = Z6CirculantLinear(in_channels=8, out_channels=4)
        
        # 24 senary + 5 vertical + 3 omega + 1 cargo z + 2 wave + 2 heading = 37
        self.j_net = nn.Linear(37, 16)
        self.r_net = nn.Linear(37, 4)
        
        self.wrench_generator = nn.Sequential(
            nn.Linear(37, 64),
            nn.SiLU(),
            nn.Linear(64, 4)
        )

    def forward(self, obs, dt=0.015):
        # Observation slicing
        err_xy = obs[:, 0:2]
        vert_err = obs[:, 2:3]
        vel_xy = obs[:, 3:5]
        vert_vel = obs[:, 5:6]
        up_z = obs[:, 6:9]
        omega = obs[:, 9:12]
        cargo_sway_xy = obs[:, 12:14]
        cargo_disp_z = obs[:, 14:15]
        wind_xy = obs[:, 15:17]
        wave_telemetry = obs[:, 17:19]
        heading_align = obs[:, 19:21]  # [sin(psi_err), cos(psi_err)]
        
        vert_state = torch.cat([vert_err, vert_vel, up_z], dim=-1) # [5]
        
        # 1. Project planar vectors onto 6-Phase Eisenstein Lattice
        s_err = project_to_senary_lattice(err_xy)
        s_vel = project_to_senary_lattice(vel_xy)
        s_wind = project_to_senary_lattice(wind_xy)
        s_sway = project_to_senary_lattice(cargo_sway_xy)
        
        s_tensor = torch.stack([s_err, s_vel, s_wind, s_sway], dim=1) # [Batch, 4, 6]
        
        # Calculate lattice activation sparsity (Gini-proxy / peak-to-average concentration)
        max_s = s_err.max(dim=-1, keepdim=True).values
        sum_s = s_err.sum(dim=-1, keepdim=True).clamp(min=1e-4)
        lattice_sparsity = ((max_s - s_err.min(dim=-1, keepdim=True).values) / sum_s).mean().detach()
        
        # 2. Z₆-Equivariant Hexagonal Tensor Contraction
        h1 = F.silu(self.senary_layer1(s_tensor))
        h2 = self.senary_layer2(h1) # [Batch, 4, 6]
        h2_flat = h2.view(obs.shape[0], 24)
        
        features = torch.cat([h2_flat, vert_state, omega, cargo_disp_z, wave_telemetry, heading_align], dim=-1) # [37]
        
        # 3. Active Port-Hamiltonian Energy Shaping: W = (J(x) - R(x)) · p(x)
        p_energy = self.wrench_generator(features)
        
        J_raw = self.j_net(features).view(-1, 4, 4)
        J = J_raw - J_raw.transpose(-1, -2)
        
        R_diag = F.softplus(self.r_net(features)) + 0.15
        R = torch.diag_embed(R_diag)
        
        passive_wrench = torch.matmul(J - R, p_energy.unsqueeze(-1)).squeeze(-1)
        
        # 4. Deadband Sieve & Weathervane Heading Spring
        err_dist = safe_norm(err_xy, dim=-1, keepdim=True)
        sieve_gate = torch.tanh(err_dist * 5.0)
        sieve_locked_fraction = (sieve_gate < 0.25).float().mean().detach()
        
        nominal_hover = (DRONE_MASS + CARGO_MASS) * GRAVITY
        Fz = nominal_hover + torch.tanh(passive_wrench[:, 0:1]) * 16.0
        tau_pitch_roll = torch.tanh(passive_wrench[:, 1:3]) * 0.60 * sieve_gate
        
        # Aerodynamic Weathervane Restoring Torque
        sin_psi = heading_align[:, 0:1]
        weathervane_torque = -0.10 * sin_psi
        tau_yaw = torch.tanh(passive_wrench[:, 3:4]) * 0.08 + weathervane_torque - 0.06 * omega[:, 2:3]
        tau_yaw = torch.clamp(tau_yaw, -0.14, 0.14)
        
        output_wrench = torch.cat([Fz, tau_pitch_roll, tau_yaw], dim=-1)
        
        dH_dt = -torch.sum(p_energy * torch.matmul(R, p_energy.unsqueeze(-1)).squeeze(-1), dim=-1).mean().detach()
        min_R_eig = R_diag.min().detach()
        
        diagnostics = {
            "dH_dt": dH_dt,
            "min_R": min_R_eig,
            "sparsity": lattice_sparsity,
            "sieve_locked": sieve_locked_fraction
        }
        
        return output_wrench, s_err, diagnostics

# =============================================================================
# 5. DIFFERENTIABLE ENVIRONMENT (WITH MULTI-DOMAIN DIAGNOSTIC SAMPLERS)
# =============================================================================
class DifferentiableQuadrotorEnv:
    def __init__(self, num_envs=12288, dt=0.015, device=DEVICE):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        
        d = ARM_LENGTH / math.sqrt(2.0)
        c_q = TORQUE_TO_THRUST_RATIO
        B = torch.tensor([
            [ 1.0,  1.0,  1.0,  1.0],
            [-d,    d,    d,   -d  ],
            [ d,   -d,    d,   -d  ],
            [-c_q, -c_q,  c_q,  c_q]
        ], device=device, dtype=torch.float32)
        self.B_inv = torch.pinverse(B)
        
        self.eye3 = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1, 1)
        self.wave_tether = WaveVariableTetherPort(characteristic_impedance=4.5, gamma_whirl=3.2)
        self.last_wave_telemetry = torch.zeros((num_envs, 2), device=device)
        self.last_tension = torch.zeros((num_envs, 1), device=device)
        self.cos_psi_err = torch.ones((num_envs, 1), device=device)
        
        # Step-level telemetry buffers
        self.step_diagnostics = {}
        self.reset(ergodic=False)

    def reset(self, ergodic=True):
        N = self.num_envs
        if not ergodic or N == 1:
            self.pos = torch.cat([
                torch.zeros((N, 2), device=self.device),
                torch.full((N, 1), 0.15, device=self.device)
            ], dim=-1)
            self.vel = torch.zeros((N, 3), device=self.device)
            self.rot_mat = torch.eye(3, device=self.device).unsqueeze(0).repeat(N, 1, 1)
            self.omega = torch.zeros((N, 3), device=self.device)
            self.cable_len = torch.full((N, 1), 1.2, device=self.device)
            self.cargo_pos = self.pos.clone()
            self.cargo_pos[:, 2] -= self.cable_len.squeeze(-1)
            self.cargo_vel = torch.zeros((N, 3), device=self.device)
            self.has_cargo = torch.ones((N, 1), device=self.device, dtype=torch.bool)
        else:
            regimes = torch.randint(0, 4, (N,), device=self.device)
            pos_x = torch.randn(N, 1, device=self.device) * 4.0
            pos_y = torch.randn(N, 1, device=self.device) * 4.0
            pos_z = torch.where(
                regimes.unsqueeze(-1) == 0, torch.full((N, 1), 0.15, device=self.device),
                torch.where(
                    regimes.unsqueeze(-1) == 1, 4.0 + 4.0 * torch.rand(N, 1, device=self.device),
                    torch.where(
                        regimes.unsqueeze(-1) == 2, 7.5 + 2.5 * torch.rand(N, 1, device=self.device),
                        2.0 + 6.0 * torch.rand(N, 1, device=self.device)
                    )
                )
            )
            self.pos = torch.cat([pos_x, pos_y, pos_z], dim=-1)
            
            vel_xy = torch.where(
                (regimes.unsqueeze(-1) == 1) | (regimes.unsqueeze(-1) == 3),
                torch.randn(N, 2, device=self.device) * 2.5,
                torch.randn(N, 2, device=self.device) * 0.2
            )
            vel_z = torch.randn(N, 1, device=self.device) * 0.2
            self.vel = torch.cat([vel_xy, vel_z], dim=-1)
            
            yaw_init = torch.rand(N, device=self.device) * 2.0 * math.pi
            c_y, s_y = torch.cos(yaw_init), torch.sin(yaw_init)
            R_init = torch.eye(3, device=self.device).unsqueeze(0).repeat(N, 1, 1)
            R_init[:, 0, 0] = c_y; R_init[:, 0, 1] = -s_y
            R_init[:, 1, 0] = s_y; R_init[:, 1, 1] = c_y
            self.rot_mat = R_init
            self.omega = torch.randn(N, 3, device=self.device) * 0.15
            
            self.has_cargo = (regimes.unsqueeze(-1) != 3)
            self.cable_len = torch.where(
                regimes.unsqueeze(-1) == 2, torch.full((N, 1), 2.2, device=self.device),
                torch.full((N, 1), 1.2, device=self.device)
            )
            
            sway_init = torch.randn(N, 2, device=self.device) * 0.18
            cargo_x = self.pos[:, 0:1] + sway_init[:, 0:1]
            cargo_y = self.pos[:, 1:2] + sway_init[:, 1:2]
            cargo_z = self.pos[:, 2:3] - self.cable_len
            self.cargo_pos = torch.cat([cargo_x, cargo_y, cargo_z], dim=-1)
            self.cargo_vel = self.vel.clone() + torch.randn(N, 3, device=self.device) * 0.1

        self.motor_thrusts = torch.zeros((N, 4), device=self.device)
        self.target_pos = torch.zeros((N, 3), device=self.device)
        self.wind = torch.zeros((N, 3), device=self.device)
        self.v_batt = torch.full((N, 1), 24.8, device=self.device)
        self.last_wave_telemetry = torch.zeros((N, 2), device=self.device)
        self.last_tension = torch.zeros((N, 1), device=self.device)

    def step(self, commanded_wrench):
        # 1. Voltage-scaled motor thrust conversion
        voltage_scale = torch.clamp(self.v_batt / 25.2, 0.70, 1.05)
        raw_thrust_cmd = torch.matmul(commanded_wrench, self.B_inv.to(commanded_wrench.dtype).T) / voltage_scale
        target_thrusts = torch.clamp(raw_thrust_cmd, 0.0, MAX_THRUST_PER_MOTOR)
        
        # Allocation Headroom & Torque Starvation Telemetry
        headroom_per_motor = (MAX_THRUST_PER_MOTOR - target_thrusts)
        min_headroom = headroom_per_motor.min(dim=-1, keepdim=True).values
        torque_starved = (raw_thrust_cmd > MAX_THRUST_PER_MOTOR).any(dim=-1, keepdim=True) | (raw_thrust_cmd < 0.0).any(dim=-1, keepdim=True)
        
        # 2. First-order motor lag filter
        alpha_m = self.dt / (MOTOR_TAU + self.dt)
        self.motor_thrusts = self.motor_thrusts + alpha_m * (target_thrusts - self.motor_thrusts)
        
        # 3. Realized body wrench
        realized_Fz = torch.sum(self.motor_thrusts, dim=-1, keepdim=True)
        tau_x = ARM_LENGTH / math.sqrt(2.0) * (-self.motor_thrusts[:, 0] + self.motor_thrusts[:, 1] + self.motor_thrusts[:, 2] - self.motor_thrusts[:, 3])
        tau_y = ARM_LENGTH / math.sqrt(2.0) * ( self.motor_thrusts[:, 0] - self.motor_thrusts[:, 1] + self.motor_thrusts[:, 2] - self.motor_thrusts[:, 3])
        tau_z = TORQUE_TO_THRUST_RATIO * (-self.motor_thrusts[:, 0] - self.motor_thrusts[:, 1] + self.motor_thrusts[:, 2] + self.motor_thrusts[:, 3])
        realized_torques = torch.stack([tau_x, tau_y, tau_z], dim=-1)
        
        # Wrench tracking fidelity
        f_tracking_err = torch.abs(commanded_wrench[:, 0:1] - realized_Fz)
        tau_tracking_err = safe_norm(commanded_wrench[:, 1:4] - realized_torques, dim=-1, keepdim=True)
        
        # 4. Translatory physics
        body_z_world = self.rot_mat[:, :, 2]
        thrust_world = body_z_world * realized_Fz
        
        v_rel = self.vel - self.wind
        v_rel_xy = v_rel[:, :2]
        senary_drag_components = project_to_senary_lattice(v_rel_xy)
        reconstructed_xy_drag = reconstruct_from_senary_lattice(senary_drag_components)
        
        drag_xy = -0.5 * 1.225 * 0.22 * reconstructed_xy_drag * safe_norm(v_rel_xy, dim=-1, keepdim=True).clamp(min=0.2)
        drag_z = -0.5 * 1.225 * 0.38 * v_rel[:, 2:3] * torch.abs(v_rel[:, 2:3]).clamp(min=0.2)
        drag_world = torch.cat([drag_xy, drag_z], dim=-1)
        
        # Tether physics & Orbital Sieve
        tether_disp = self.cargo_pos - self.pos
        tension_mag, F_anti_whirl, wave_telem, tether_dir, orbital_mom, slack = self.wave_tether(
            self.vel, self.cargo_vel, self.cable_len, tether_disp
        )
        
        # Tether Phase-Space Metrics
        tether_dist = safe_norm(tether_disp, dim=-1, keepdim=True)
        v_rel_cargo = self.cargo_vel - self.vel
        whirl_num = torch.sum(orbital_mom * orbital_mom, dim=-1, keepdim=True)
        whirl_den = (torch.sum(tether_disp * tether_disp, dim=-1, keepdim=True) * torch.sum(v_rel_cargo * v_rel_cargo, dim=-1, keepdim=True) + 1e-4)
        whirl_ratio = whirl_num / whirl_den
        
        cone_angle_deg = torch.acos(torch.clamp(torch.abs(tether_disp[:, 2:3]) / tether_dist.clamp(min=1e-4), -1.0, 1.0)) * (180.0 / math.pi)
        slack_flag = (slack < 0.0) & self.has_cargo
        
        self.last_wave_telemetry = torch.where(self.has_cargo, wave_telem, torch.zeros_like(wave_telem)).detach()
        self.last_tension = torch.where(self.has_cargo, tension_mag, torch.zeros_like(tension_mag)).detach()
        
        F_tether_radial = tension_mag * tether_dir
        F_tether_drone = torch.where(self.has_cargo, F_tether_radial - F_anti_whirl, torch.zeros_like(tether_disp))
        F_tether_cargo = torch.where(self.has_cargo, -F_tether_radial + F_anti_whirl, torch.zeros_like(tether_disp))
        
        gravity_force = torch.tensor([0.0, 0.0, -GRAVITY * DRONE_MASS], device=self.device)
        total_force = thrust_world + drag_world + gravity_force + F_tether_drone
        accel = total_force / DRONE_MASS
        
        gyroscopic = torch.cross(self.omega, self.omega * INERTIA, dim=-1)
        alpha = (realized_torques - gyroscopic) * INV_INERTIA
        
        # Out-of-place kinematic integration
        self.vel = self.vel + accel * self.dt
        raw_pos = self.pos + self.vel * self.dt
        
        ground_mask = raw_pos[:, 2:3] < 0.08
        clamped_pos_z = torch.where(ground_mask, torch.full_like(raw_pos[:, 2:3], 0.08), raw_pos[:, 2:3])
        self.pos = torch.cat([raw_pos[:, 0:2], clamped_pos_z], dim=-1)
        
        clamped_vel_z = torch.where(ground_mask & (self.vel[:, 2:3] < 0.0), torch.zeros_like(self.vel[:, 2:3]), self.vel[:, 2:3])
        self.vel = torch.cat([self.vel[:, 0:2], clamped_vel_z], dim=-1)
        
        self.omega = self.omega + alpha * self.dt
        
        # Lie-Algebra SO(3) update
        omega_dt = self.omega * self.dt
        angle = safe_norm(omega_dt, dim=-1, keepdim=True).clamp(min=1e-6)
        axis = omega_dt / angle
        K = torch.zeros((self.num_envs, 3, 3), device=self.device, dtype=self.pos.dtype)
        K[:, 0, 1] = -axis[:, 2]; K[:, 0, 2] =  axis[:, 1]
        K[:, 1, 0] =  axis[:, 2]; K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]; K[:, 2, 1] =  axis[:, 0]
        dR = self.eye3 + torch.sin(angle).unsqueeze(-1) * K + (1.0 - torch.cos(angle)).unsqueeze(-1) * torch.matmul(K, K)
        self.rot_mat = torch.matmul(self.rot_mat, dR)
        
        # Gram-Schmidt SO(3) re-orthogonalization
        c0 = self.rot_mat[:, :, 0]
        c1 = self.rot_mat[:, :, 1]
        c0 = c0 / safe_norm(c0, dim=-1, keepdim=True)
        c2 = torch.cross(c0, c1, dim=-1)
        c2 = c2 / safe_norm(c2, dim=-1, keepdim=True)
        c1 = torch.cross(c2, c0, dim=-1)
        self.rot_mat = torch.stack([c0, c1, c2], dim=-1)
        
        # Cargo dynamics with ground support plane
        if torch.any(self.has_cargo):
            cargo_grav = torch.tensor([0.0, 0.0, -GRAVITY * CARGO_MASS], device=self.device)
            v_cargo_air = self.cargo_vel - self.wind
            cargo_drag = -0.5 * 1.225 * 0.15 * v_cargo_air * safe_norm(v_cargo_air, dim=-1, keepdim=True).clamp(min=0.2)
            cargo_accel = (cargo_grav + cargo_drag + F_tether_cargo) / CARGO_MASS
            self.cargo_vel = torch.where(self.has_cargo, self.cargo_vel + cargo_accel * self.dt, self.cargo_vel)
            raw_cargo_pos = torch.where(self.has_cargo, self.cargo_pos + self.cargo_vel * self.dt, self.cargo_pos)
            
            cargo_ground = raw_cargo_pos[:, 2:3] < 0.08
            clamped_cargo_z = torch.where(cargo_ground, torch.full_like(raw_cargo_pos[:, 2:3], 0.08), raw_cargo_pos[:, 2:3])
            self.cargo_pos = torch.cat([raw_cargo_pos[:, 0:2], clamped_cargo_z], dim=-1)
            
            clamped_cargo_vz = torch.where(cargo_ground & (self.cargo_vel[:, 2:3] < 0.0), torch.zeros_like(self.cargo_vel[:, 2:3]), self.cargo_vel[:, 2:3])
            self.cargo_vel = torch.cat([self.cargo_vel[:, 0:2], clamped_cargo_vz], dim=-1)

        # Store step telemetry detached to isolate autograd
        self.step_diagnostics = {
            "headroom": min_headroom.detach(),
            "torque_starved": torque_starved.float().detach(),
            "f_err": f_tracking_err.detach(),
            "tau_err": tau_tracking_err.detach(),
            "whirl_ratio": whirl_ratio.detach(),
            "cone_angle_deg": cone_angle_deg.detach(),
            "slack_flag": slack_flag.float().detach()
        }
            
        return self.get_observations()

    def get_observations(self):
        err_pos = self.target_pos - self.pos
        
        # 1. Weathervane Flight-Track Heading Alignment
        b1 = self.rot_mat[:, :, 0]
        track_dist = safe_norm(err_pos[:, 0:2], dim=-1, keepdim=True)
        u_track = torch.where(track_dist > 0.4, err_pos[:, 0:2] / track_dist.clamp(min=1e-4), b1[:, 0:2])
        u_track = u_track / safe_norm(u_track, dim=-1, keepdim=True).clamp(min=1e-4)
        
        sin_psi_err = (b1[:, 0:1] * u_track[:, 1:2] - b1[:, 1:2] * u_track[:, 0:1])
        cos_psi_err = (b1[:, 0:1] * u_track[:, 0:1] + b1[:, 1:2] * u_track[:, 1:2])
        self.cos_psi_err = cos_psi_err
        heading_align = torch.cat([sin_psi_err, cos_psi_err], dim=-1)
        
        # Aerodynamic Sideslip Angle (beta)
        v_rel = self.vel - self.wind
        speed_xy = safe_norm(v_rel[:, :2], dim=-1, keepdim=True)
        b1_xy = b1[:, :2] / safe_norm(b1[:, :2], dim=-1, keepdim=True).clamp(min=1e-4)
        u_airspeed_xy = v_rel[:, :2] / speed_xy.clamp(min=1e-4)
        cos_sideslip = torch.sum(b1_xy * u_airspeed_xy, dim=-1, keepdim=True)
        self.sideslip_deg = torch.where(speed_xy > 0.5, torch.acos(torch.clamp(cos_sideslip, -1.0, 1.0)) * (180.0 / math.pi), torch.zeros_like(speed_xy))
        self.head_lag_deg = torch.acos(torch.clamp(cos_psi_err, -1.0, 1.0)) * (180.0 / math.pi)
        
        # 2. Horizontal body-frame projection
        yaw = torch.atan2(self.rot_mat[:, 1, 0], self.rot_mat[:, 0, 0])
        cos_y = torch.cos(yaw).unsqueeze(-1)
        sin_y = torch.sin(yaw).unsqueeze(-1)
        
        err_x_body =  cos_y * err_pos[:, 0:1] + sin_y * err_pos[:, 1:2]
        err_y_body = -sin_y * err_pos[:, 0:1] + cos_y * err_pos[:, 1:2]
        err_body_2d = torch.cat([err_x_body, err_y_body], dim=-1)
        
        v_body_x =  cos_y * self.vel[:, 0:1] + sin_y * self.vel[:, 1:2]
        v_body_y = -sin_y * self.vel[:, 0:1] + cos_y * self.vel[:, 1:2]
        v_body_2d = torch.cat([v_body_x, v_body_y], dim=-1)
        
        cargo_disp = (self.cargo_pos - self.pos) * self.has_cargo.float()
        sway_x_body =  cos_y * cargo_disp[:, 0:1] + sin_y * cargo_disp[:, 1:2]
        sway_y_body = -sin_y * cargo_disp[:, 0:1] + cos_y * cargo_disp[:, 1:2]
        sway_body_2d = torch.cat([sway_x_body, sway_y_body], dim=-1)
        
        wind_x_body =  cos_y * self.wind[:, 0:1] + sin_y * self.wind[:, 1:2]
        wind_y_body = -sin_y * self.wind[:, 0:1] + cos_y * self.wind[:, 1:2]
        wind_body_2d = torch.cat([wind_x_body, wind_y_body], dim=-1)
        
        obs = torch.cat([
            err_body_2d,                # [0, 1] Heading-aligned planar error
            err_pos[:, 2:3],            # [2] Vertical error
            v_body_2d,                  # [3, 4] Heading-aligned planar velocity
            self.vel[:, 2:3],           # [5] Vertical velocity
            self.rot_mat[:, :, 2],      # [6, 7, 8] Up-vector in world
            self.omega,                 # [9, 10, 11] Angular rates (body)
            sway_body_2d,               # [12, 13] Heading-aligned cargo sway
            cargo_disp[:, 2:3],         # [14] Vertical cargo offset
            wind_body_2d,               # [15, 16] Heading-aligned wind
            self.last_wave_telemetry,   # [17, 18] Wave-variable scattering telemetry
            heading_align               # [19, 20] Weathervane heading alignment
        ], dim=-1)
        return obs

# =============================================================================
# 6. TRAINING: MULTI-DOMAIN HIGH-DENSITY FAULT TELEMETRY
# =============================================================================
def train_sehl_controller(epochs=512, horizon=128, num_envs=12288, device=DEVICE, **kwargs):
    epochs = int(kwargs.get("num_epochs", epochs))
    horizon = int(kwargs.get("horizon_steps", horizon))
    num_envs = int(kwargs.get("envs", num_envs))
    device = torch.device(kwargs.get("dev", device))

    print("\n" + "=" * 80)
    print(" SENARY EISENSTEIN-HAMILTONIAN CONTROLLER (SEHL-Control)")
    print(f" Environments: {num_envs:,} | Horizon: {horizon} steps | Epochs: {epochs:,}")
    print(f" Flight Dynamics: Ergodic Multi-Phase Re-seeding (Breaks 1.92s Horizon Trap)")
    print(f" Telemetry Engine: 5-Domain Fault & Health Monitoring Active")
    print(f" Transitions Evaluated: {num_envs * horizon * epochs:,}")
    print("=" * 80)

    env = DifferentiableQuadrotorEnv(num_envs=num_envs, dt=0.015, device=device)
    policy = SenaryHamiltonianPolicy().to(device)
    thermo_loss_fn = ThermodynamicEntropyLoss(t_ambient=AMBIENT_TEMP_KELVIN, motor_resistance=MOTOR_COIL_RESISTANCE)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=2.5e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1.0e-4)

    stage1_end = int(epochs * 0.25)
    stage2_end = int(epochs * 0.55)
    stage3_end = int(epochs * 0.80)

    t_start = time.time()
    t_last_log = t_start
    
    for ep in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        env.reset(ergodic=True)

        is_log_epoch = (ep + 1) % 25 == 0 or ep == 0
        if is_log_epoch and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Environmental wind scaling across curriculum
        if ep < stage1_end:
            env.wind.zero_()
        elif ep < stage2_end:
            env.wind = torch.randn((num_envs, 3), device=device) * torch.tensor([1.5, 1.5, 0.2], device=device)
        elif ep < stage3_end:
            env.wind = torch.randn((num_envs, 3), device=device) * torch.tensor([3.5, 3.5, 0.6], device=device)
        else:
            env.wind = torch.randn((num_envs, 3), device=device) * torch.tensor([5.0, 5.0, 1.0], device=device)

        target_r = 1.0 + 8.0 * torch.rand(num_envs, device=device)
        target_ang = torch.rand(num_envs, device=device) * 2.0 * math.pi
        target_z = 1.5 + 8.0 * torch.rand(num_envs, device=device)
        env.target_pos = torch.stack([target_r * torch.cos(target_ang),
                                      target_r * torch.sin(target_ang),
                                      target_z], dim=-1)

        obs = env.get_observations()
        loss_total = torch.zeros(1, device=device)

        # GPU-resident multi-domain diagnostic accumulators
        if is_log_epoch:
            accum_components = {
                "pos": torch.zeros(1, device=device),
                "vel": torch.zeros(1, device=device),
                "tilt": torch.zeros(1, device=device),
                "spin": torch.zeros(1, device=device),
                "head": torch.zeros(1, device=device),
                "sway": torch.zeros(1, device=device),
                "s_dot": torch.zeros(1, device=device),
                "s_dot_raw": torch.zeros(1, device=device),
            }
            # Domain 1: Actuators & Headroom
            accum_headroom = torch.zeros(1, device=device)
            accum_starve = torch.zeros(1, device=device)
            accum_ferr = torch.zeros(1, device=device)
            accum_tauerr = torch.zeros(1, device=device)
            sat_zero_accum = torch.zeros(1, device=device)
            sat_max_accum = torch.zeros(1, device=device)
            
            # Domain 2: Tether & Cargo
            accum_whirl = torch.zeros(1, device=device)
            accum_cone = torch.zeros(1, device=device)
            accum_slack = torch.zeros(1, device=device)
            max_tension_epoch = torch.zeros(1, device=device)
            
            # Domain 3: Weathervane & Sideslip
            accum_sideslip = torch.zeros(1, device=device)
            accum_headlag = torch.zeros(1, device=device)
            
            # Domain 4: Invariants & Lattice
            accum_dh_dt = torch.zeros(1, device=device)
            accum_min_R = torch.zeros(1, device=device)
            accum_sparsity = torch.zeros(1, device=device)
            accum_sievelock = torch.zeros(1, device=device)
            
            max_rates_epoch = torch.zeros(1, device=device)

        amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device.type == "cuda")):
            for t in range(horizon):
                wrench, _, policy_diag = policy(obs, dt=env.dt)
                obs = env.step(wrench)

                v_rel = env.vel - env.wind
                step_loss, comp = thermo_loss_fn(
                    v_rel, env.motor_thrusts, env.target_pos, env.pos,
                    env.rot_mat, env.omega, env.cargo_pos[:, :2] - env.pos[:, :2],
                    env.cargo_vel, env.vel, env.cos_psi_err
                )
                loss_total = loss_total + step_loss
                
                # Asynchronous GPU accumulation (zero PCIe sync)
                if is_log_epoch:
                    for k in accum_components:
                        accum_components[k] = accum_components[k] + comp[k]
                    
                    # Accumulate Actuator & Headroom
                    env_diag = env.step_diagnostics
                    accum_headroom += env_diag["headroom"].mean()
                    accum_starve += env_diag["torque_starved"].mean()
                    accum_ferr += env_diag["f_err"].mean()
                    accum_tauerr += env_diag["tau_err"].mean()
                    sat_zero_accum += (env.motor_thrusts <= 0.05).float().mean()
                    sat_max_accum += (env.motor_thrusts >= (MAX_THRUST_PER_MOTOR - 0.05)).float().mean()
                    
                    # Accumulate Tether & Cargo
                    accum_whirl += env_diag["whirl_ratio"].mean()
                    accum_cone += env_diag["cone_angle_deg"].mean()
                    accum_slack += env_diag["slack_flag"].mean()
                    max_tension_epoch = torch.maximum(max_tension_epoch, env.last_tension.max())
                    
                    # Accumulate Aerodynamics
                    accum_sideslip += env.sideslip_deg.mean()
                    accum_headlag += env.head_lag_deg.mean()
                    
                    # Accumulate Invariants
                    accum_dh_dt += policy_diag["dH_dt"]
                    accum_min_R += policy_diag["min_R"]
                    accum_sparsity += policy_diag["sparsity"]
                    accum_sievelock += policy_diag["sieve_locked"]
                    
                    rates_now = safe_norm(env.omega, dim=-1).max()
                    max_rates_epoch = torch.maximum(max_rates_epoch, rates_now)
                
            loss_total = loss_total / horizon

        loss_total.backward()
        raw_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.5).item()
        optimizer.step()
        scheduler.step()

        # -------------------------------------------------------------
        # COMPREHENSIVE MULTI-DOMAIN TERMINAL LOGGING BLOCK
        # -------------------------------------------------------------
        if is_log_epoch:
            now = time.time()
            elapsed_total = now - t_start
            dt_window = max(1e-4, now - t_last_log)
            t_last_log = now
            
            steps_in_window = (num_envs * horizon) * (25 if ep > 0 else 1)
            sps = int(steps_in_window / dt_window)
            
            lr = optimizer.param_groups[0]["lr"]
            stage = 1 if ep < stage1_end else (2 if ep < stage2_end else (3 if ep < stage3_end else 4))
            
            if device.type == "cuda":
                vram_peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                vram_reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
                vram_str = f"Peak: {vram_peak_gb:.2f}GB / Res: {vram_reserved_gb:.2f}GB"
            else:
                vram_str = "CPU Mode"
            
            pos_err_all = safe_norm(env.target_pos - env.pos, dim=-1)
            mean_pos_err = pos_err_all.mean().item()
            max_pos_err = pos_err_all.max().item()
            
            altitude_mean = env.pos[:, 2].mean().item()
            speed_mean = safe_norm(env.vel, dim=-1).mean().item()
            
            # Transfer accumulated metrics to CPU once
            mean_sideslip = accum_sideslip.item() / horizon
            mean_headlag = accum_headlag.item() / horizon
            mean_margin = accum_headroom.item() / horizon
            mean_starve_pct = (accum_starve.item() / horizon) * 100.0
            sat_0_pct = (sat_zero_accum.item() / horizon) * 100.0
            sat_max_pct = (sat_max_accum.item() / horizon) * 100.0
            mean_tau_err = accum_tauerr.item() / horizon
            mean_tension = env.last_tension.mean().item()
            max_tension = max_tension_epoch.item()
            mean_cone = accum_cone.item() / horizon
            mean_whirl = accum_whirl.item() / horizon
            mean_slack_pct = (accum_slack.item() / horizon) * 100.0
            sway_m = safe_norm(env.cargo_pos[:, :2] - env.pos[:, :2], dim=-1).mean().item()
            mean_dh_dt = accum_dh_dt.item() / horizon
            mean_min_R = accum_min_R.item() / horizon
            mean_s_dot = accum_components["s_dot_raw"].item() / horizon
            mean_sievelock_pct = (accum_sievelock.item() / horizon) * 100.0
            mean_sparsity = accum_sparsity.item() / horizon
            
            print(f"\n[Epoch {ep+1:04d}/{epochs}] Stage {stage} | LR: {lr:.5f} | BF16 | {sps:,} SPS | VRAM: {vram_str} | Elapsed: {elapsed_total:.1f}s")
            print(f"  ├─ Flight Path:  PosErr: {mean_pos_err:.2f}m (max {max_pos_err:.2f}m) | Alt: {altitude_mean:.1f}m | Speed: {speed_mean:.2f}m/s | Sideslip: {mean_sideslip:.1f}° | HeadLag: {mean_headlag:.1f}°")
            print(f"  ├─ Actuators:    Margin: {mean_margin:.2f}N | TorqueStarve: {mean_starve_pct:.1f}% | Sat[0N]: {sat_0_pct:.1f}% | Sat[Max]: {sat_max_pct:.1f}% | TauErr: {mean_tau_err:.3f}Nm | GradNorm: {raw_grad_norm:.3f}")
            print(f"  ├─ Tether/Cargo: Tension: {mean_tension:.1f}N (max {max_tension:.1f}N) | ConeAngle: {mean_cone:.1f}° | WhirlRatio: {mean_whirl:.2f} | Slack: {mean_slack_pct:.1f}% | Sway: {sway_m:.2f}m")
            print(f"  ├─ Invariants:   dH/dt: {mean_dh_dt:+.3f} J/s | S_dot: {mean_s_dot:.3f} W/K | min(eig(R)): {mean_min_R:+.3f} | SieveLock: {mean_sievelock_pct:.0f}% | Sparsity: {mean_sparsity:.2f}")
            print(f"  └─ Loss Terms:   Pos: {accum_components['pos'].item()/horizon:.2f} | Vel: {accum_components['vel'].item()/horizon:.2f} | Head: {accum_components['head'].item()/horizon:.2f} | Tilt: {accum_components['tilt'].item()/horizon:.2f} | Spin: {accum_components['spin'].item()/horizon:.2f} | Total: {loss_total.item():.3f}")

    torch.save(policy.state_dict(), CHECKPOINT_FILE)
    print(f"\n>> {epochs} EPOCHS COMPLETED in {time.time() - t_start:.2f} seconds!")
    print(f">> Full diagnostic weights saved to: {CHECKPOINT_FILE}\n")
    return policy

# =============================================================================
# 7. 3D INTERACTIVE FLIGHT & SENARY TELEMETRY COCKPIT (PYGAME)
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
    if point is None:
        return None
    point = np.asarray(point, dtype=np.float64)
    if not np.all(np.isfinite(point)):
        return None
        
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    cam_lookat = np.asarray(cam_lookat, dtype=np.float64)
    if not (np.all(np.isfinite(cam_pos)) and np.all(np.isfinite(cam_lookat))):
        return None
        
    fwd = cam_lookat - cam_pos
    fwd_len = np.linalg.norm(fwd)
    if fwd_len < 1e-4:
        return None
    fwd = fwd / fwd_len
    
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    right_len = np.linalg.norm(right)
    if right_len < 1e-4:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / right_len
    up = np.cross(right, fwd)
    
    delta = point - cam_pos
    depth = float(np.dot(delta, fwd))
    if not math.isfinite(depth) or depth <= 0.2:
        return None
        
    focal = (screen_h * 0.5) / math.tan(math.radians(fov) * 0.5)
    px = screen_w * 0.5 + focal * float(np.dot(delta, right)) / depth
    py = screen_h * 0.5 - focal * float(np.dot(delta, up)) / depth
    
    if not (math.isfinite(px) and math.isfinite(py)):
        return None
    if abs(px) > 1e5 or abs(py) > 1e5:
        return None
        
    return int(round(px)), int(round(py)), depth

def run_3d_simulation(policy):
    if pygame is None:
        print("[ERROR] Pygame is required for the visual interface. Run: pip install pygame")
        return

    pygame.init()
    SCREEN_W, SCREEN_H = 1200, 780
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("SEHL-Control: Integrated Flight & Multi-Domain Diagnostic Cockpit")
    clock = pygame.time.Clock()
    font_hud = pygame.font.SysFont("consolas", 12, bold=True)

    world = DeliverySimWorld()
    env = DifferentiableQuadrotorEnv(num_envs=1, dt=0.016, device=DEVICE)
    policy.eval()

    mission_state = "TAKEOFF"
    target_building = world.get_current_delivery()
    pad_target = target_building["pad"]

    env.reset(ergodic=False)
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

    print(">> 3D Cockpit active with real-time Multi-Domain Fault Telemetry.")
    print(">> Controls: [Mouse Drag] Orbit View | [Mouse Wheel] Zoom | [Spacebar] Inject 8 m/s Crosswind\n")

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                cam_dist = np.clip(cam_dist - event.y * 2.0, 6.0, 90.0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    env.wind[0] = torch.randn(3, device=DEVICE) * 6.0
                    print(f">> [GUST INJECTED] Sensed wind: {env.wind[0].cpu().numpy().round(1)} m/s")

        mouse_pressed = pygame.mouse.get_pressed()
        if mouse_pressed[0] or mouse_pressed[2]:
            dx, dy = pygame.mouse.get_rel()
            cam_yaw += dx * 0.008
            cam_pitch = np.clip(cam_pitch + dy * 0.008, 0.05, 1.45)
        else:
            pygame.mouse.get_rel()

        drone_p_np = env.pos[0].detach().cpu().numpy()
        if not np.all(np.isfinite(drone_p_np)):
            env.reset(ergodic=False)
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

        with torch.no_grad():
            obs = env.get_observations()
            commanded_wrench, senary_coords, policy_diag = policy(obs, dt=env.dt)
            env.step(commanded_wrench)

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

        # Buildings & Helipads
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

        # Quadrotor with Nose Vector
        drone_pt = project_3d_point(drone_p_np, cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
        rot_np = env.rot_mat[0].detach().cpu().numpy()
        if drone_pt and np.all(np.isfinite(rot_np)):
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
            
            # Nose Indicator
            nose_pos = drone_p_np + rot_np @ np.array([d_arm * 1.3, 0.0, 0.0])
            nose_pt = project_3d_point(nose_pos, cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            if nose_pt:
                pygame.draw.line(screen, (255, 60, 60), drone_pt[:2], nose_pt[:2], 3)

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
        # HUD: COMPREHENSIVE FLIGHT & DIAGNOSTIC INSTRUMENTATION
        # -------------------------------------------------------------
        hud_bg = pygame.Rect(15, 15, 470, 280)
        pygame.draw.rect(screen, (10, 14, 20), hud_bg, border_radius=6)
        pygame.draw.rect(screen, (0, 220, 255), hud_bg, 1, border_radius=6)

        speed = np.linalg.norm(env.vel[0].cpu().numpy())
        altitude = drone_p_np[2]
        dist_target = np.linalg.norm(drone_p_np - env.target_pos[0].cpu().numpy())
        s_coords = senary_coords[0].cpu().numpy()
        dh_dt_val = policy_diag["dH_dt"].item()
        yaw_deg = math.degrees(math.atan2(rot_np[1, 0], rot_np[0, 0]))
        
        # Pull live step diagnostics
        env_d = env.step_diagnostics
        whirl_val = env_d["whirl_ratio"][0, 0].item()
        headroom_val = env_d["headroom"][0, 0].item()
        sideslip_val = env.sideslip_deg[0, 0].item()
        headlag_val = env.head_lag_deg[0, 0].item()
        tauerr_val = env_d["tau_err"][0, 0].item()
        sparsity_val = policy_diag["sparsity"].item()

        status_lines = [
            ("SEHL-CONTROL: DIAGNOSTIC TELEMETRY COCKPIT", (0, 255, 200)),
            (f"Mission Phase:   {mission_state}", (255, 220, 40)),
            (f"Heading / Lag:   {yaw_deg:+06.1f}° | Lag: {headlag_val:04.1f}° | Sideslip: {sideslip_val:04.1f}°", (255, 255, 255)),
            (f"Waypoint Error:  {dist_target:.3f} m  (Alt: {altitude:.2f}m | Spd: {speed:.2f}m/s)", (0, 220, 255)),
            (f"SENARY ERROR:    X={float_to_base6_str(env.target_pos[0,0].item()-drone_p_np[0])}  Y={float_to_base6_str(env.target_pos[0,1].item()-drone_p_np[1])}", (255, 160, 40)),
            (f"Motor Headroom:  {headroom_val:.2f}N | TauErr: {tauerr_val:.3f}Nm", (180, 210, 255)),
            (f"Tether Metrics:  T={env.last_tension[0].item():.1f}N | WhirlRatio: {whirl_val:.2f} (Sieve: {policy_diag['sieve_locked'].item()*100:.0f}%)", (150, 220, 255)),
            (f"Lattice State:   Sparsity: {sparsity_val:.2f} | dH/dt: {dh_dt_val:+.3f} J/s", (180, 200, 240)),
            (f"Delivered Boxes: {len(delivered_boxes)}", (0, 255, 140))
        ]

        for idx, (txt, col) in enumerate(status_lines):
            screen.blit(font_hud.render(txt, True, col), (hud_bg.x + 12, hud_bg.y + 10 + idx * 18))

        # Hexagonal Eisenstein Radar
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
# 8. MAIN EXECUTION ENTRYPOINT
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" SENARY EISENSTEIN-HAMILTONIAN LATTICE CONTROLLER (SEHL-Control)")
    print(" Scale: num_envs=12288, horizon=128, epochs=512 (~805M Transitions)")
    print(" Telemetry Engine: 5-Domain Multi-Fault Tracking Active")
    print(" Hardware Acceleration: Native Bfloat16 Mixed Precision + TF32 Tensor Cores")
    print("=" * 80)

    if os.path.exists(CHECKPOINT_FILE):
        print(f">> Existing trained checkpoint detected: {CHECKPOINT_FILE}")
        print(">> Loading policy directly into 3D simulation...")
        trained_policy = SenaryHamiltonianPolicy().to(DEVICE)
        trained_policy.load_state_dict(torch.load(CHECKPOINT_FILE, map_location=DEVICE))
    else:
        trained_policy = train_sehl_controller(epochs=512, horizon=128, num_envs=12288, device=DEVICE)

    run_3d_simulation(trained_policy)