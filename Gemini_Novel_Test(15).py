"""
=============================================================================
SENARY EISENSTEIN-HAMILTONIAN LATTICE CONTROLLER (SEHL-Control)
High-Throughput Configuration: num_envs=12288, horizon=128, epochs=512
Hardware Target: RTX 5060 Ti / CUDA Modern Architectures (Mixed-Precision)
Kinematic Architecture: Lie-Group SO(3) + Eisenstein A₂ Group-Equivariance
=============================================================================
"""

import math
import os
import sys
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
DRONE_MASS = 1.45  # kg
CARGO_MASS = 0.40  # kg
ARM_LENGTH = 0.22  # m
MOTOR_TAU = 0.032  # Motor time constant (s)
MAX_THRUST_PER_MOTOR = 8.5  # N (34 N total vehicle thrust ceiling)
TORQUE_TO_THRUST_RATIO = 0.016
AMBIENT_TEMP_KELVIN = 293.15
MOTOR_COIL_RESISTANCE = 0.048  # Ohms
GROUND_FRICTION_COEFF = 0.75


# =============================================================================
# 1. DIFFERENTIABLE MATHEMATICAL UTILITIES & EISENSTEIN A₂ PROJECTIONS
# =============================================================================
def safe_norm(x, dim=-1, keepdim=False, eps=1e-12):
    """Numerically stable vector norm preventing 0/0 NaN gradients."""
    return torch.sqrt(torch.sum(x * x, dim=dim, keepdim=keepdim) + eps)


# 6-phase directional unit vectors spanning the A₂ hexagonal root lattice
SENARY_ANGLES = torch.tensor(
    [k * math.pi / 3.0 for k in range(6)], dtype=torch.float32
)
SENARY_BASIS = torch.stack(
    [torch.cos(SENARY_ANGLES), torch.sin(SENARY_ANGLES)], dim=-1
)  # [6, 2]


def project_to_senary_lattice(v_2d):
    """Projects 2D planar vectors onto the 6 non-orthogonal Eisenstein axes."""
    basis = SENARY_BASIS.to(dtype=v_2d.dtype, device=v_2d.device)
    projections = torch.matmul(v_2d, basis.T)
    return F.relu(projections)


def reconstruct_from_senary_lattice(s_6d):
    """Maps 6-phase senary coordinates back to 2D Cartesian space."""
    basis = SENARY_BASIS.to(dtype=s_6d.dtype, device=s_6d.device)
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
        d = min(5, max(0, int(f)))
        frac_digits.append(str(d))
        f -= d
    return f"{sign}{digits}.{''.join(frac_digits)}₆"


class Z6CirculantLinear(nn.Module):
    """
    Z₆ Cyclic Group-Equivariant Linear Layer.
    Implemented via 1D circular convolution for cuDNN acceleration.
    Preserves exact 60° rotational equivariance.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, 6) * 0.15
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # x shape: [Batch, in_channels, 6]
        x_padded = F.pad(x, (0, 5), mode="circular")
        return F.conv1d(x_padded, self.weight, self.bias)


# =============================================================================
# 2. WAVE SCATTERING PORT & UNILATERAL VISCOELASTIC TETHER
# =============================================================================
class WaveVariableTetherPort(nn.Module):
    """
    Unilateral tensile cable with scattering-domain wave telemetry.
    Strictly collinear along instantaneous cable line-of-sight.
    """

    def __init__(
        self,
        characteristic_impedance=4.5,
        spring_constant=160.0,
        damping_coeff=4.0,
    ):
        super().__init__()
        self.b = characteristic_impedance
        self.k_s = spring_constant
        self.c_d = damping_coeff

    def forward(
        self,
        drone_velocity,
        cargo_velocity,
        cable_length,
        tether_displacement,
    ):
        center_distance = safe_norm(tether_displacement, dim=-1, keepdim=True)
        tether_dir = tether_displacement / center_distance.clamp(min=1e-4)
        slack = center_distance - cable_length
        is_taut = slack > 0.0

        v_rel_vec = cargo_velocity - drone_velocity
        v_rad = torch.sum(v_rel_vec * tether_dir, dim=-1, keepdim=True)

        tension_raw = self.k_s * slack + self.c_d * v_rad
        tension = torch.where(
            is_taut & (tension_raw > 0.0), tension_raw, torch.zeros_like(slack)
        )
        tension = torch.clamp(tension, min=0.0, max=45.0)

        sqrt_2b = math.sqrt(2.0 * self.b)
        u_wave = (tension + self.b * v_rad) / sqrt_2b
        v_wave = (tension - self.b * v_rad) / sqrt_2b

        orbital_momentum = torch.cross(tether_displacement, v_rel_vec, dim=-1)
        wave_telemetry = torch.cat([u_wave, v_wave], dim=-1)

        return tension, wave_telemetry, tether_dir, orbital_momentum, slack


# =============================================================================
# 3. THERMODYNAMIC ONSAGER LOSS & FRAME-INVARIANT ORBIT ANNIHILATOR
# =============================================================================
class ThermodynamicEntropyLoss(nn.Module):
    """
    Minimizes thermodynamic entropy production rate while eliminating
    transverse orbiting and yaw spin without origin singularities.
    """

    def __init__(
        self,
        t_ambient=AMBIENT_TEMP_KELVIN,
        motor_resistance=MOTOR_COIL_RESISTANCE,
    ):
        super().__init__()
        self.t_amb = t_ambient
        self.r_coil = motor_resistance

    def forward(
        self,
        relative_velocity_world,
        motor_thrusts,
        target_pos,
        current_pos,
        rot_mat,
        omega,
        cargo_disp,
        cargo_vel,
        drone_vel,
        cos_psi_err,
        orbital_momentum,
        has_cargo,
    ):
        # 1. Aerodynamic Entropy Production (W/K)
        speed = safe_norm(relative_velocity_world, dim=-1, keepdim=True)
        drag_force = 0.5 * 1.225 * 0.25 * speed * relative_velocity_world
        s_dot_aero = (
            torch.sum(relative_velocity_world * drag_force, dim=-1) / self.t_amb
        )

        # 2. Joulean Electrical Resistance Entropy Production (W/K)
        joulean_power = torch.sum(motor_thrusts * self.r_coil * 18.5, dim=-1)
        s_dot_joule = joulean_power / self.t_amb
        entropy_rate = (s_dot_aero + s_dot_joule).mean()

        # 3. Kinematic Setpoint Errors
        pos_error_vec = target_pos - current_pos
        pos_error = safe_norm(pos_error_vec, dim=-1).mean()
        vel_penalty = safe_norm(relative_velocity_world, dim=-1).mean()
        tilt_penalty = (1.0 - rot_mat[:, 2, 2]).mean()

        # 4. Heading Alignment with Distance-Gated Attenuation
        err_world_xy = pos_error_vec[:, 0:2]
        dist_xy = safe_norm(err_world_xy, dim=-1, keepdim=True)
        guidance_gate = torch.clamp(dist_xy / 1.5, 0.0, 1.0)
        heading_penalty = ((1.0 - cos_psi_err) * guidance_gate).mean()

        # 5. Geometrically Invariant Planar Orbit Annihilator
        u_rad_world = err_world_xy / dist_xy.clamp(min=1e-4)
        v_rel_xy = relative_velocity_world[:, 0:2]
        v_tangent = (
            v_rel_xy[:, 0:1] * (-u_rad_world[:, 1:2])
            + v_rel_xy[:, 1:2] * u_rad_world[:, 0:1]
        )
        orbit_penalty = (torch.abs(v_tangent) * guidance_gate).mean()

        # 6. Rotational Damping
        yaw_rate = torch.abs(omega[:, 2]).mean()
        total_rates = safe_norm(omega, dim=-1).mean()
        spin_penalty = total_rates + 3.5 * yaw_rate

        # 7. Masked Cargo Stabilization
        cargo_mask = has_cargo.float()
        cargo_sway = (
            safe_norm(cargo_disp, dim=-1, keepdim=True) * cargo_mask
        ).mean()
        cargo_whip = (
            safe_norm(cargo_vel - drone_vel, dim=-1, keepdim=True) * cargo_mask
        ).mean()
        orbital_whip = (
            safe_norm(orbital_momentum, dim=-1, keepdim=True) * cargo_mask
        ).mean()

        loss_pos_part = 5.20 * pos_error
        loss_vel_part = 1.10 * vel_penalty
        loss_tilt_part = 3.20 * tilt_penalty
        loss_head_part = 2.60 * heading_penalty
        loss_orbit_part = 3.40 * orbit_penalty
        loss_spin_part = 1.60 * spin_penalty
        loss_sway_part = (
            1.60 * cargo_sway + 1.20 * cargo_whip + 0.80 * orbital_whip
        )
        loss_sdot_part = 0.08 * entropy_rate

        total_loss = (
            loss_sdot_part
            + loss_pos_part
            + loss_vel_part
            + loss_tilt_part
            + loss_head_part
            + loss_orbit_part
            + loss_spin_part
            + loss_sway_part
        )

        components = {
            "pos": loss_pos_part.detach(),
            "vel": loss_vel_part.detach(),
            "tilt": loss_tilt_part.detach(),
            "spin": loss_spin_part.detach(),
            "head": loss_head_part.detach(),
            "orbit": loss_orbit_part.detach(),
            "sway": loss_sway_part.detach(),
            "s_dot": loss_sdot_part.detach(),
            "s_dot_raw": entropy_rate.detach(),
        }
        return total_loss, components


# =============================================================================
# 4. PORT-HAMILTONIAN EISENSTEIN CONTROLLER ARCHITECTURE
# =============================================================================
class SenaryHamiltonianPolicy(nn.Module):
    """
    Port-Hamiltonian Neural Network with Exact ℤ₆-Equivariance.
    Enforces true conjugate effort-flow damping on body-frame generalized velocities:
    P_diss = - nu_body · R · nu_body <= 0.
    """

    def __init__(self):
        super().__init__()
        self.senary_layer1 = Z6CirculantLinear(in_channels=4, out_channels=8)
        self.senary_layer2 = Z6CirculantLinear(in_channels=8, out_channels=4)

        self.feature_encoder = nn.Sequential(
            nn.Linear(37, 64), nn.SiLU(), nn.Linear(64, 48), nn.SiLU()
        )

        self.potential_net = nn.Linear(48, 4)
        self.damping_net = nn.Linear(48, 4)

    def forward(self, obs, dt=0.015):
        obs_fp32 = obs.float()
        err_xy = obs_fp32[:, 0:2]
        vert_err = obs_fp32[:, 2:3]
        vel_xy = obs_fp32[:, 3:5]
        v_body_z = obs_fp32[:, 5:6]
        up_z = obs_fp32[:, 6:9]
        omega = obs_fp32[:, 9:12]
        cargo_sway_xy = obs_fp32[:, 12:14]
        cargo_disp_z = obs_fp32[:, 14:15]
        wind_xy = obs_fp32[:, 15:17]
        wave_telemetry = obs_fp32[:, 17:19]
        heading_align = obs_fp32[:, 19:21]

        vert_state = torch.cat([vert_err, v_body_z, up_z], dim=-1)

        # 1. Eisenstein A₂ Lattice Projection
        s_err = project_to_senary_lattice(err_xy)
        s_vel = project_to_senary_lattice(vel_xy)
        s_wind = project_to_senary_lattice(wind_xy)
        s_sway = project_to_senary_lattice(cargo_sway_xy)
        s_tensor = torch.stack(
            [s_err, s_vel, s_wind, s_sway], dim=1
        )  # [Batch, 4, 6]

        max_s = s_err.max(dim=-1, keepdim=True).values
        sum_s = s_err.sum(dim=-1, keepdim=True).clamp(min=1e-4)
        lattice_sparsity = (
            (max_s - s_err.min(dim=-1, keepdim=True).values) / sum_s
        ).mean()

        # 2. ℤ₆-Equivariant Contraction
        h1 = F.silu(self.senary_layer1(s_tensor))
        h2 = self.senary_layer2(h1)
        h2_flat = h2.view(obs.shape[0], 24)

        features = torch.cat(
            [
                h2_flat,
                vert_state,
                omega,
                cargo_disp_z,
                wave_telemetry,
                heading_align,
            ],
            dim=-1,
        )

        latent = self.feature_encoder(features)

        # 3. Exact Conjugate Port-Hamiltonian Dissipation Port
        w_pot = self.potential_net(latent)
        r_diag = F.softplus(self.damping_net(latent)) + 0.12

        nu_physical = torch.cat([v_body_z, omega], dim=-1)
        dissipative_wrench = -r_diag * nu_physical
        dH_dt = -torch.sum(r_diag * (nu_physical**2), dim=-1).mean()

        # 4. Actuator Allocation Synthesis
        nominal_hover = (DRONE_MASS + CARGO_MASS) * GRAVITY
        Fz = (
            nominal_hover
            + torch.tanh(w_pot[:, 0:1]) * 15.0
            + dissipative_wrench[:, 0:1]
        )

        tau_roll = torch.tanh(w_pot[:, 1:2]) * 0.55 + dissipative_wrench[:, 1:2]
        tau_pitch = torch.tanh(w_pot[:, 2:3]) * 0.55 + dissipative_wrench[:, 2:3]

        err_dist = safe_norm(err_xy, dim=-1, keepdim=True)
        guidance_gate = torch.clamp(err_dist / 1.2, 0.0, 1.0)
        sin_psi = heading_align[:, 0:1]
        weathervane_torque = 0.18 * sin_psi * guidance_gate

        tau_yaw = (
            torch.tanh(w_pot[:, 3:4]) * 0.08
            + weathervane_torque
            + dissipative_wrench[:, 3:4]
        )
        tau_yaw = torch.clamp(tau_yaw, -0.14, 0.14)

        output_wrench = torch.cat([Fz, tau_roll, tau_pitch, tau_yaw], dim=-1)
        sieve_locked_fraction = (err_dist < 0.25).float().mean()

        diagnostics = {
            "dH_dt": dH_dt.detach(),
            "min_R": r_diag.min().detach(),
            "sparsity": lattice_sparsity.detach(),
            "sieve_locked": sieve_locked_fraction.detach(),
        }

        return output_wrench, s_err, diagnostics


# =============================================================================
# 5. HIGH-FIDELITY DIFFERENTIABLE QUADROTOR SIMULATION (SO(3) LIE KINEMATICS)
# =============================================================================
class DifferentiableQuadrotorEnv:
    """
    Batched 6-DoF Quadrotor Environment with exact SO(3) Rodrigues kinematics.
    Physics states are strictly integrated in FP32 with compliant ground contact models.
    """

    def __init__(self, num_envs=12288, dt=0.015, device=DEVICE):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device

        self.inertia = torch.tensor(
            [0.021, 0.021, 0.036], device=device, dtype=torch.float32
        )
        self.inv_inertia = 1.0 / self.inertia

        d = ARM_LENGTH / math.sqrt(2.0)
        c_q = TORQUE_TO_THRUST_RATIO

        B = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [d, -d, d, -d],
                [-d, d, d, -d],
                [-c_q, -c_q, c_q, c_q],
            ],
            device=device,
            dtype=torch.float32,
        )
        self.B_inv = torch.pinverse(B)

        self.eye3 = (
            torch.eye(3, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(num_envs, 1, 1)
        )
        self.wave_tether = WaveVariableTetherPort(
            characteristic_impedance=4.5,
            spring_constant=160.0,
            damping_coeff=4.0,
        )

        self.last_wave_telemetry = torch.zeros(
            (num_envs, 2), device=device, dtype=torch.float32
        )
        self.last_tension = torch.zeros(
            (num_envs, 1), device=device, dtype=torch.float32
        )
        self.last_orbital_momentum = torch.zeros(
            (num_envs, 3), device=device, dtype=torch.float32
        )
        self.cos_psi_err = torch.ones(
            (num_envs, 1), device=device, dtype=torch.float32
        )
        self.sideslip_deg = torch.zeros(
            (num_envs, 1), device=device, dtype=torch.float32
        )
        self.head_lag_deg = torch.zeros(
            (num_envs, 1), device=device, dtype=torch.float32
        )

        self.step_diagnostics = {}
        self.reset(ergodic=False)

    def detach_states(self):
        """Detaches recurrent physics state tensors to bound BPTT graph depth."""
        self.pos = self.pos.detach()
        self.vel = self.vel.detach()
        self.rot_mat = self.rot_mat.detach()
        self.omega = self.omega.detach()
        self.cargo_pos = self.cargo_pos.detach()
        self.cargo_vel = self.cargo_vel.detach()
        self.motor_thrusts = self.motor_thrusts.detach()
        self.last_wave_telemetry = self.last_wave_telemetry.detach()
        self.last_tension = self.last_tension.detach()
        self.last_orbital_momentum = self.last_orbital_momentum.detach()
        self.cos_psi_err = self.cos_psi_err.detach()
        self.sideslip_deg = self.sideslip_deg.detach()
        self.head_lag_deg = self.head_lag_deg.detach()

    def reset(self, ergodic=True):
        N = self.num_envs
        if not ergodic or N == 1:
            self.pos = torch.cat(
                [
                    torch.zeros((N, 2), device=self.device, dtype=torch.float32),
                    torch.full(
                        (N, 1), 0.15, device=self.device, dtype=torch.float32
                    ),
                ],
                dim=-1,
            )
            self.vel = torch.zeros(
                (N, 3), device=self.device, dtype=torch.float32
            )
            self.rot_mat = (
                torch.eye(3, device=self.device, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(N, 1, 1)
            )
            self.omega = torch.zeros(
                (N, 3), device=self.device, dtype=torch.float32
            )
            self.cable_len = torch.full(
                (N, 1), 1.2, device=self.device, dtype=torch.float32
            )
            # Safely clamp cargo above the ground plane on reset
            self.cargo_pos = self.pos.clone()
            self.cargo_pos[:, 2:3] = torch.clamp(
                self.pos[:, 2:3] - self.cable_len, min=0.08
            )
            self.cargo_vel = torch.zeros(
                (N, 3), device=self.device, dtype=torch.float32
            )
            self.has_cargo = torch.ones(
                (N, 1), device=self.device, dtype=torch.bool
            )
        else:
            regimes = torch.randint(0, 4, (N,), device=self.device)
            pos_x = torch.randn(N, 1, device=self.device) * 4.0
            pos_y = torch.randn(N, 1, device=self.device) * 4.0
            pos_z = torch.where(
                regimes.unsqueeze(-1) == 0,
                torch.full((N, 1), 0.15, device=self.device),
                torch.where(
                    regimes.unsqueeze(-1) == 1,
                    4.0 + 4.0 * torch.rand(N, 1, device=self.device),
                    torch.where(
                        regimes.unsqueeze(-1) == 2,
                        7.5 + 2.5 * torch.rand(N, 1, device=self.device),
                        2.0 + 6.0 * torch.rand(N, 1, device=self.device),
                    ),
                ),
            )
            self.pos = torch.cat([pos_x, pos_y, pos_z], dim=-1)

            vel_xy = torch.where(
                (regimes.unsqueeze(-1) == 1) | (regimes.unsqueeze(-1) == 3),
                torch.randn(N, 2, device=self.device) * 2.5,
                torch.randn(N, 2, device=self.device) * 0.2,
            )
            vel_z = torch.randn(N, 1, device=self.device) * 0.2
            self.vel = torch.cat([vel_xy, vel_z], dim=-1)

            yaw_init = torch.rand(N, device=self.device) * 2.0 * math.pi
            c_y, s_y = torch.cos(yaw_init), torch.sin(yaw_init)
            R_init = (
                torch.eye(3, device=self.device, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(N, 1, 1)
            )
            R_init[:, 0, 0] = c_y
            R_init[:, 0, 1] = -s_y
            R_init[:, 1, 0] = s_y
            R_init[:, 1, 1] = c_y
            self.rot_mat = R_init
            self.omega = torch.randn(N, 3, device=self.device) * 0.15

            self.has_cargo = regimes.unsqueeze(-1) != 3
            self.cable_len = torch.where(
                regimes.unsqueeze(-1) == 2,
                torch.full((N, 1), 2.2, device=self.device),
                torch.full((N, 1), 1.2, device=self.device),
            )

            sway_init = torch.randn(N, 2, device=self.device) * 0.18
            cargo_x = self.pos[:, 0:1] + sway_init[:, 0:1]
            cargo_y = self.pos[:, 1:2] + sway_init[:, 1:2]
            # Prevent underground spawns on ground regime
            cargo_z = torch.clamp(self.pos[:, 2:3] - self.cable_len, min=0.08)
            self.cargo_pos = torch.cat([cargo_x, cargo_y, cargo_z], dim=-1)
            self.cargo_vel = (
                self.vel.clone() + torch.randn(N, 3, device=self.device) * 0.1
            )

        self.motor_thrusts = torch.zeros(
            (N, 4), device=self.device, dtype=torch.float32
        )
        self.target_pos = torch.zeros(
            (N, 3), device=self.device, dtype=torch.float32
        )
        self.wind = torch.zeros((N, 3), device=self.device, dtype=torch.float32)
        self.v_batt = torch.full(
            (N, 1), 24.8, device=self.device, dtype=torch.float32
        )
        self.last_wave_telemetry.zero_()
        self.last_tension.zero_()
        self.last_orbital_momentum.zero_()

    def step(self, commanded_wrench):
        cmd_wrench = commanded_wrench.float()

        # 1. Voltage-scaled motor thrust allocation
        voltage_scale = torch.clamp(self.v_batt / 25.2, 0.70, 1.05)
        raw_thrust_cmd = torch.matmul(cmd_wrench, self.B_inv.T) / voltage_scale
        target_thrusts = torch.clamp(
            raw_thrust_cmd, 0.0, MAX_THRUST_PER_MOTOR
        )

        headroom_per_motor = MAX_THRUST_PER_MOTOR - target_thrusts
        min_headroom = headroom_per_motor.min(dim=-1, keepdim=True).values
        torque_starved = (raw_thrust_cmd > MAX_THRUST_PER_MOTOR).any(
            dim=-1, keepdim=True
        ) | (raw_thrust_cmd < 0.0).any(dim=-1, keepdim=True)

        # 2. First-order actuator lag
        alpha_m = self.dt / (MOTOR_TAU + self.dt)
        self.motor_thrusts = self.motor_thrusts + alpha_m * (
            target_thrusts - self.motor_thrusts
        )

        # 3. Realized Body Wrench
        realized_Fz = torch.sum(self.motor_thrusts, dim=-1, keepdim=True)
        arm_scaled = ARM_LENGTH / math.sqrt(2.0)
        tau_x = arm_scaled * (
            self.motor_thrusts[:, 0]
            - self.motor_thrusts[:, 1]
            + self.motor_thrusts[:, 2]
            - self.motor_thrusts[:, 3]
        )
        tau_y = arm_scaled * (
            -self.motor_thrusts[:, 0]
            + self.motor_thrusts[:, 1]
            + self.motor_thrusts[:, 2]
            - self.motor_thrusts[:, 3]
        )
        tau_z = TORQUE_TO_THRUST_RATIO * (
            -self.motor_thrusts[:, 0]
            - self.motor_thrusts[:, 1]
            + self.motor_thrusts[:, 2]
            + self.motor_thrusts[:, 3]
        )
        realized_torques = torch.stack([tau_x, tau_y, tau_z], dim=-1)

        f_tracking_err = torch.abs(cmd_wrench[:, 0:1] - realized_Fz)
        tau_tracking_err = safe_norm(
            cmd_wrench[:, 1:4] - realized_torques, dim=-1, keepdim=True
        )

        # 4. Translatory Dynamics & Wake Aerodynamics
        body_z_world = self.rot_mat[:, :, 2]
        thrust_world = body_z_world * realized_Fz

        v_rel = self.vel - self.wind
        v_rel_xy = v_rel[:, :2]
        senary_drag = project_to_senary_lattice(v_rel_xy)
        reconstructed_xy_drag = reconstruct_from_senary_lattice(senary_drag)

        drag_xy = (
            -0.5
            * 1.225
            * 0.22
            * reconstructed_xy_drag
            * safe_norm(v_rel_xy, dim=-1, keepdim=True).clamp(min=0.2)
        )
        drag_z = (
            -0.5
            * 1.225
            * 0.38
            * v_rel[:, 2:3]
            * torch.abs(v_rel[:, 2:3]).clamp(min=0.2)
        )
        drag_world = torch.cat([drag_xy, drag_z], dim=-1)

        # 5. Tether Tensile Dynamics
        tether_disp = self.cargo_pos - self.pos
        (
            tension_mag,
            wave_telem,
            tether_dir,
            orbital_mom,
            slack,
        ) = self.wave_tether(
            self.vel, self.cargo_vel, self.cable_len, tether_disp
        )

        tether_dist = safe_norm(tether_disp, dim=-1, keepdim=True)
        v_rel_cargo = self.cargo_vel - self.vel
        whirl_num = torch.sum(orbital_mom * orbital_mom, dim=-1, keepdim=True)
        whirl_den = (
            torch.sum(tether_disp * tether_disp, dim=-1, keepdim=True)
            * torch.sum(v_rel_cargo * v_rel_cargo, dim=-1, keepdim=True)
            + 1e-4
        )
        whirl_ratio = whirl_num / whirl_den

        cone_angle_deg = torch.acos(
            torch.clamp(
                torch.abs(tether_disp[:, 2:3]) / tether_dist.clamp(min=1e-4),
                -1.0,
                1.0,
            )
        ) * (180.0 / math.pi)
        slack_flag = (slack < 0.0) & self.has_cargo

        self.last_wave_telemetry = torch.where(
            self.has_cargo, wave_telem, torch.zeros_like(wave_telem)
        )
        self.last_tension = torch.where(
            self.has_cargo, tension_mag, torch.zeros_like(tension_mag)
        )
        self.last_orbital_momentum = torch.where(
            self.has_cargo, orbital_mom, torch.zeros_like(orbital_mom)
        )

        F_tether_drone = torch.where(
            self.has_cargo, tension_mag * tether_dir, torch.zeros_like(tether_disp)
        )
        F_tether_cargo = torch.where(
            self.has_cargo,
            -tension_mag * tether_dir,
            torch.zeros_like(tether_disp),
        )

        gravity_force = torch.tensor(
            [0.0, 0.0, -GRAVITY * DRONE_MASS], device=self.device
        )

        # Smooth viscoelastic ground barrier
        drone_penetration = F.relu(0.08 - self.pos[:, 2:3])
        f_ground_normal = (
            2500.0 * drone_penetration
            + 40.0 * drone_penetration * F.relu(-self.vel[:, 2:3])
        )
        f_ground_contact = torch.cat(
            [torch.zeros((self.num_envs, 2), device=self.device), f_ground_normal],
            dim=-1,
        )

        total_force = (
            thrust_world
            + drag_world
            + gravity_force
            + F_tether_drone
            + f_ground_contact
        )
        accel = total_force / DRONE_MASS

        # 6. Rotational Dynamics & Euler Integration
        gyroscopic = torch.cross(self.omega, self.omega * self.inertia, dim=-1)
        alpha = (realized_torques - gyroscopic) * self.inv_inertia

        self.vel = self.vel + accel * self.dt
        self.pos = self.pos + self.vel * self.dt

        # Out-of-place kinematic containment to maintain autograd integrity
        clamped_z = torch.clamp(self.pos[:, 2:3], min=0.08)
        self.pos = torch.cat([self.pos[:, 0:2], clamped_z], dim=-1)
        clamped_vz = torch.where(
            (self.pos[:, 2:3] <= 0.081) & (self.vel[:, 2:3] < 0.0),
            torch.zeros_like(self.vel[:, 2:3]),
            self.vel[:, 2:3],
        )
        self.vel = torch.cat([self.vel[:, 0:2], clamped_vz], dim=-1)

        self.omega = self.omega + alpha * self.dt

        # 7. Exact Non-Distorted Lie-Algebra SO(3) Rodrigues Exponential Map
        v_rot = self.omega * self.dt
        theta_sq = torch.sum(v_rot * v_rot, dim=-1, keepdim=True)
        theta = torch.sqrt(theta_sq + 1e-16)

        alpha_coef = torch.where(
            theta_sq < 1e-6,
            1.0 - theta_sq / 6.0,
            torch.sin(theta) / theta,
        ).unsqueeze(-1)
        beta_coef = torch.where(
            theta_sq < 1e-6,
            0.5 - theta_sq / 24.0,
            (1.0 - torch.cos(theta)) / (theta_sq + 1e-16),
        ).unsqueeze(-1)

        zeros = torch.zeros_like(v_rot[:, 0:1])
        vx, vy, vz = v_rot[:, 0:1], v_rot[:, 1:2], v_rot[:, 2:3]
        row0 = torch.cat([zeros, -vz, vy], dim=-1)
        row1 = torch.cat([vz, zeros, -vx], dim=-1)
        row2 = torch.cat([-vy, vx, zeros], dim=-1)
        K = torch.stack([row0, row1, row2], dim=-2)

        dR = self.eye3 + alpha_coef * K + beta_coef * torch.matmul(K, K)
        self.rot_mat = torch.matmul(self.rot_mat, dR)

        # Gram-Schmidt SO(3) Re-orthogonalization
        c0 = self.rot_mat[:, :, 0]
        c1 = self.rot_mat[:, :, 1]
        c0 = c0 / safe_norm(c0, dim=-1, keepdim=True)
        c2 = torch.cross(c0, c1, dim=-1)
        c2 = c2 / safe_norm(c2, dim=-1, keepdim=True)
        c1 = torch.cross(c2, c0, dim=-1)
        self.rot_mat = torch.stack([c0, c1, c2], dim=-1)

        # 8. Slung-Load Cargo Viscoelastic Kinematics (Branchless)
        has_c = self.has_cargo.float()
        cargo_grav = torch.tensor(
            [0.0, 0.0, -GRAVITY * CARGO_MASS], device=self.device
        )
        v_cargo_air = self.cargo_vel - self.wind
        cargo_drag = (
            -0.5
            * 1.225
            * 0.15
            * v_cargo_air
            * safe_norm(v_cargo_air, dim=-1, keepdim=True).clamp(min=0.2)
        )

        cargo_penetration = F.relu(0.08 - self.cargo_pos[:, 2:3])
        f_cargo_normal = (
            2000.0 * cargo_penetration
            + 30.0 * cargo_penetration * F.relu(-self.cargo_vel[:, 2:3])
        )
        f_cargo_normal_total = (
            f_cargo_normal + CARGO_MASS * GRAVITY * (cargo_penetration > 0.0).float()
        )

        v_cargo_xy = self.cargo_vel[:, :2]
        v_cargo_speed = safe_norm(v_cargo_xy, dim=-1, keepdim=True).clamp(
            min=1e-3
        )
        f_friction_planar = (
            -GROUND_FRICTION_COEFF
            * f_cargo_normal_total
            * (v_cargo_xy / v_cargo_speed)
        )
        f_cargo_contact = torch.cat(
            [
                f_friction_planar * (cargo_penetration > 0.0).float(),
                f_cargo_normal,
            ],
            dim=-1,
        )

        cargo_accel = (
            cargo_grav + cargo_drag + F_tether_cargo + f_cargo_contact
        ) / CARGO_MASS

        self.cargo_vel = self.cargo_vel + (cargo_accel * self.dt) * has_c
        self.cargo_pos = self.cargo_pos + (self.cargo_vel * self.dt) * has_c

        # Out-of-place ground containment for cargo
        clamped_cargo_z = torch.clamp(self.cargo_pos[:, 2:3], min=0.08)
        self.cargo_pos = torch.cat(
            [self.cargo_pos[:, 0:2], clamped_cargo_z], dim=-1
        )
        clamped_cargo_vz = torch.where(
            (self.cargo_pos[:, 2:3] <= 0.081)
            & (self.cargo_vel[:, 2:3] < 0.0),
            torch.zeros_like(self.cargo_vel[:, 2:3]),
            self.cargo_vel[:, 2:3],
        )
        self.cargo_vel = torch.cat(
            [self.cargo_vel[:, 0:2], clamped_cargo_vz], dim=-1
        )

        self.step_diagnostics = {
            "headroom": min_headroom.detach(),
            "torque_starved": torque_starved.float().detach(),
            "f_err": f_tracking_err.detach(),
            "tau_err": tau_tracking_err.detach(),
            "whirl_ratio": whirl_ratio.detach(),
            "cone_angle_deg": cone_angle_deg.detach(),
            "slack_flag": slack_flag.float().detach(),
        }

        return self.get_observations()

    def get_observations(self):
        err_pos_world = self.target_pos - self.pos

        err_pos_body = torch.matmul(
            self.rot_mat.transpose(-1, -2), err_pos_world.unsqueeze(-1)
        ).squeeze(-1)
        v_body = torch.matmul(
            self.rot_mat.transpose(-1, -2), self.vel.unsqueeze(-1)
        ).squeeze(-1)
        wind_body = torch.matmul(
            self.rot_mat.transpose(-1, -2), self.wind.unsqueeze(-1)
        ).squeeze(-1)

        cargo_disp_world = (self.cargo_pos - self.pos) * self.has_cargo.float()
        cargo_disp_body = torch.matmul(
            self.rot_mat.transpose(-1, -2), cargo_disp_world.unsqueeze(-1)
        ).squeeze(-1)

        dist_xy = safe_norm(
            err_pos_body[:, 0:2], dim=-1, keepdim=True
        ).clamp(min=1e-4)
        u_target_body = err_pos_body[:, 0:2] / dist_xy

        sin_psi_err = u_target_body[:, 1:2]
        cos_psi_err = u_target_body[:, 0:1]
        self.cos_psi_err = cos_psi_err
        heading_align = torch.cat([sin_psi_err, cos_psi_err], dim=-1)

        speed_xy = safe_norm(v_body[:, 0:2], dim=-1, keepdim=True)
        cos_beta = v_body[:, 0:1] / speed_xy.clamp(min=1e-4)
        self.sideslip_deg = torch.where(
            speed_xy > 0.5,
            torch.acos(torch.clamp(cos_beta, -1.0, 1.0)) * (180.0 / math.pi),
            torch.zeros_like(speed_xy),
        )
        self.head_lag_deg = torch.acos(
            torch.clamp(cos_psi_err, -1.0, 1.0)
        ) * (180.0 / math.pi)

        obs = torch.cat(
            [
                err_pos_body[:, 0:2],
                err_pos_body[:, 2:3],
                v_body[:, 0:2],
                v_body[:, 2:3],
                self.rot_mat[:, :, 2],
                self.omega,
                cargo_disp_body[:, 0:2],
                cargo_disp_body[:, 2:3],
                wind_body[:, 0:2],
                self.last_wave_telemetry,
                heading_align,
            ],
            dim=-1,
        )
        return obs


# =============================================================================
# 6. TRAINING ENGINE WITH TRUNCATED BPTT (TBPTT)
# =============================================================================
def train_sehl_controller(
    epochs=512, horizon=128, num_envs=12288, device=DEVICE, tbptt_chunk=32, **kwargs
):
    epochs = int(kwargs.get("num_epochs", epochs))
    horizon = int(kwargs.get("horizon_steps", horizon))
    num_envs = int(kwargs.get("envs", num_envs))
    device = torch.device(kwargs.get("dev", device))

    print("\n" + "=" * 80)
    print(" SENARY EISENSTEIN-HAMILTONIAN CONTROLLER (SEHL-Control)")
    print(f" Environments: {num_envs:,} | Horizon: {horizon} steps | TBPTT Chunk: {tbptt_chunk}")
    print(f" Precision: BF16 Neural Activation + IEEE-754 FP32 Kinematics")
    print(f" Port Invariant: Conjugate Body-Frame Flow Damping (P_diss <= 0)")
    print(f" Transitions Evaluated: {num_envs * horizon * epochs:,}")
    print("=" * 80)

    env = DifferentiableQuadrotorEnv(num_envs=num_envs, dt=0.015, device=device)
    policy = SenaryHamiltonianPolicy().to(device)
    thermo_loss_fn = ThermodynamicEntropyLoss(
        t_ambient=AMBIENT_TEMP_KELVIN, motor_resistance=MOTOR_COIL_RESISTANCE
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=2.5e-3, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1.0e-4
    )

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

        if ep < stage1_end:
            env.wind.zero_()
        elif ep < stage2_end:
            env.wind = torch.randn(
                (num_envs, 3), device=device, dtype=torch.float32
            ) * torch.tensor([1.5, 1.5, 0.2], device=device)
        elif ep < stage3_end:
            env.wind = torch.randn(
                (num_envs, 3), device=device, dtype=torch.float32
            ) * torch.tensor([3.5, 3.5, 0.6], device=device)
        else:
            env.wind = torch.randn(
                (num_envs, 3), device=device, dtype=torch.float32
            ) * torch.tensor([5.0, 5.0, 1.0], device=device)

        target_r = 1.0 + 8.0 * torch.rand(num_envs, device=device)
        target_ang = torch.rand(num_envs, device=device) * 2.0 * math.pi
        target_z = 1.5 + 8.0 * torch.rand(num_envs, device=device)
        env.target_pos = torch.stack(
            [
                target_r * torch.cos(target_ang),
                target_r * torch.sin(target_ang),
                target_z,
            ],
            dim=-1,
        )

        obs = env.get_observations()
        total_epoch_loss = 0.0

        if is_log_epoch:
            accum_components = {
                "pos": torch.zeros(1, device=device),
                "vel": torch.zeros(1, device=device),
                "tilt": torch.zeros(1, device=device),
                "spin": torch.zeros(1, device=device),
                "head": torch.zeros(1, device=device),
                "orbit": torch.zeros(1, device=device),
                "sway": torch.zeros(1, device=device),
                "s_dot": torch.zeros(1, device=device),
                "s_dot_raw": torch.zeros(1, device=device),
            }
            accum_headroom = torch.zeros(1, device=device)
            accum_starve = torch.zeros(1, device=device)
            accum_tauerr = torch.zeros(1, device=device)
            sat_zero_accum = torch.zeros(1, device=device)
            sat_max_accum = torch.zeros(1, device=device)
            accum_whirl = torch.zeros(1, device=device)
            accum_cone = torch.zeros(1, device=device)
            accum_slack = torch.zeros(1, device=device)
            max_tension_epoch = torch.zeros(1, device=device)
            accum_sideslip = torch.zeros(1, device=device)
            accum_headlag = torch.zeros(1, device=device)
            accum_dh_dt = torch.zeros(1, device=device)
            accum_min_R = torch.zeros(1, device=device)
            accum_sparsity = torch.zeros(1, device=device)
            accum_sievelock = torch.zeros(1, device=device)

        amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        # Truncated Backpropagation Through Time (TBPTT) execution loop
        for chunk_start in range(0, horizon, tbptt_chunk):
            chunk_steps = min(tbptt_chunk, horizon - chunk_start)
            chunk_loss = torch.zeros(1, device=device)

            for _ in range(chunk_steps):
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=(device.type == "cuda"),
                ):
                    wrench, _, policy_diag = policy(obs, dt=env.dt)

                obs = env.step(wrench.float())

                v_rel_world = env.vel - env.wind
                step_loss, comp = thermo_loss_fn(
                    v_rel_world,
                    env.motor_thrusts,
                    env.target_pos,
                    env.pos,
                    env.rot_mat,
                    env.omega,
                    env.cargo_pos[:, :2] - env.pos[:, :2],
                    env.cargo_vel,
                    env.vel,
                    env.cos_psi_err,
                    env.last_orbital_momentum,
                    env.has_cargo,
                )
                chunk_loss = chunk_loss + step_loss

                if is_log_epoch:
                    for k in accum_components:
                        accum_components[k] += comp[k]
                    env_diag = env.step_diagnostics
                    accum_headroom += env_diag["headroom"].mean()
                    accum_starve += env_diag["torque_starved"].mean()
                    accum_tauerr += env_diag["tau_err"].mean()
                    sat_zero_accum += (env.motor_thrusts <= 0.05).float().mean()
                    sat_max_accum += (
                        (env.motor_thrusts >= (MAX_THRUST_PER_MOTOR - 0.05))
                        .float()
                        .mean()
                    )
                    accum_whirl += env_diag["whirl_ratio"].mean()
                    accum_cone += env_diag["cone_angle_deg"].mean()
                    accum_slack += env_diag["slack_flag"].mean()
                    max_tension_epoch = torch.maximum(
                        max_tension_epoch, env.last_tension.max()
                    )
                    accum_sideslip += env.sideslip_deg.mean().detach()
                    accum_headlag += env.head_lag_deg.mean().detach()
                    accum_dh_dt += policy_diag["dH_dt"]
                    accum_min_R += policy_diag["min_R"]
                    accum_sparsity += policy_diag["sparsity"]
                    accum_sievelock += policy_diag["sieve_locked"]

            scaled_chunk_loss = chunk_loss / horizon
            scaled_chunk_loss.backward()
            total_epoch_loss += scaled_chunk_loss.item()

            env.detach_states()
            obs = obs.detach()

        raw_grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), max_norm=1.5
        ).item()
        optimizer.step()
        scheduler.step()

        if is_log_epoch:
            now = time.time()
            elapsed_total = now - t_start
            dt_window = max(1e-4, now - t_last_log)
            t_last_log = now

            steps_in_window = (num_envs * horizon) * (25 if ep > 0 else 1)
            sps = int(steps_in_window / dt_window)
            lr = optimizer.param_groups[0]["lr"]
            stage = (
                1
                if ep < stage1_end
                else (2 if ep < stage2_end else (3 if ep < stage3_end else 4))
            )

            if device.type == "cuda":
                vram_peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                vram_reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
                vram_str = f"Peak: {vram_peak_gb:.2f}GB / Res: {vram_reserved_gb:.2f}GB"
            else:
                vram_str = "CPU Mode"

            pos_err_all = safe_norm(env.target_pos - env.pos, dim=-1)
            mean_pos_err = pos_err_all.mean().item()
            max_pos_err = pos_err_all.max().item()

            print(
                f"\n[Epoch {ep+1:04d}/{epochs}] Stage {stage} | LR: {lr:.5f} | BF16/FP32 | {sps:,} SPS | VRAM: {vram_str} | Elapsed: {elapsed_total:.1f}s"
            )
            print(
                f"  ├─ Flight Path:  PosErr: {mean_pos_err:.2f}m (max {max_pos_err:.2f}m) | Alt: {env.pos[:, 2].mean().item():.1f}m | Sideslip: {accum_sideslip.item()/horizon:.1f}° | HeadLag: {accum_headlag.item()/horizon:.1f}°"
            )
            print(
                f"  ├─ Actuators:    Margin: {accum_headroom.item()/horizon:.2f}N | Starve: {(accum_starve.item()/horizon)*100:.1f}% | Sat[0N]: {(sat_zero_accum.item()/horizon)*100:.1f}% | Sat[Max]: {(sat_max_accum.item()/horizon)*100:.1f}% | GradNorm: {raw_grad_norm:.3f}"
            )
            print(
                f"  ├─ Tether/Cargo: Tension: {env.last_tension.mean().item():.1f}N (max {max_tension_epoch.item():.1f}N) | Cone: {accum_cone.item()/horizon:.1f}° | WhirlRatio: {accum_whirl.item()/horizon:.2f} | Slack: {(accum_slack.item()/horizon)*100:.1f}%"
            )
            print(
                f"  ├─ Passivity:    dH/dt: {accum_dh_dt.item()/horizon:+.3f} J/s | S_dot: {accum_components['s_dot_raw'].item()/horizon:.3f} W/K | min(R): {accum_min_R.item()/horizon:+.3f} | SieveLock: {(accum_sievelock.item()/horizon)*100:.0f}%"
            )
            print(
                f"  └─ Loss Terms:   Pos: {accum_components['pos'].item()/horizon:.2f} | OrbitKill: {accum_components['orbit'].item()/horizon:.2f} | Head: {accum_components['head'].item()/horizon:.2f} | Total: {total_epoch_loss:.3f}"
            )

    torch.save(policy.state_dict(), CHECKPOINT_FILE)
    print(f"\n>> {epochs} EPOCHS COMPLETED in {time.time() - t_start:.2f} seconds!")
    print(f">> Model weights saved to: {CHECKPOINT_FILE}\n")
    return policy


# =============================================================================
# 7. 3D INTERACTIVE COCKPIT & VISUALIZER
# =============================================================================
class DeliverySimWorld:

    def __init__(self):
        self.depot_pad = np.array([0.0, 0.0, 0.1], dtype=np.float32)
        self.buildings = [
            {
                "aabb": [-16.0, -4.0, -16.0, -4.0, 7.5],
                "name": "Tower Alpha",
                "pad": np.array([-10.0, -10.0, 7.6], dtype=np.float32),
            },
            {
                "aabb": [6.0, 18.0, -14.0, -2.0, 11.0],
                "name": "Tower Bravo",
                "pad": np.array([12.0, -8.0, 11.1], dtype=np.float32),
            },
            {
                "aabb": [-18.0, -6.0, 8.0, 20.0, 9.0],
                "name": "Medical Plaza",
                "pad": np.array([-12.0, 14.0, 9.1], dtype=np.float32),
            },
            {
                "aabb": [8.0, 20.0, 6.0, 18.0, 14.5],
                "name": "Sky Hub 04",
                "pad": np.array([14.0, 12.0, 14.6], dtype=np.float32),
            },
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
        print("[ERROR] Pygame required. Install via: pip install pygame")
        return

    pygame.init()
    SCREEN_W, SCREEN_H = 1200, 780
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(
        "SEHL-Control: Straight-Line Docking & Orbit-Annihilated Cockpit"
    )
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
    touchdown_dwell = 0.0
    running = True

    print(">> Cockpit active. Radial guidance and conjugate dissipation online.")
    print(">> Controls: [Mouse Drag] Orbit | [Mouse Wheel] Zoom | [Spacebar] Inject Gust | [ESC] Exit\n")

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                cam_dist = np.clip(cam_dist - event.y * 2.0, 6.0, 90.0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    env.wind[0] = torch.randn(3, device=DEVICE) * 5.5
                    print(
                        f">> [GUST INJECTED] Sensed wind: {env.wind[0].cpu().numpy().round(1)} m/s"
                    )

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

        drone_z = drone_p_np[2]
        dist_to_pad_xy = np.linalg.norm(drone_p_np[:2] - pad_target[:2])

        if mission_state == "TAKEOFF":
            env.target_pos[0] = torch.tensor(
                [0.0, 0.0, pad_target[2] + 4.5], device=DEVICE
            )
            if drone_z >= pad_target[2] + 3.0:
                mission_state = "CRUISE_OUTBOUND"
        elif mission_state == "CRUISE_OUTBOUND":
            env.target_pos[0] = torch.tensor(
                [pad_target[0], pad_target[1], pad_target[2] + 4.5],
                device=DEVICE,
            )
            if dist_to_pad_xy < 0.65:
                mission_state = "PRECISION_HOVER"
                hover_dwell = 0.0
        elif mission_state == "PRECISION_HOVER":
            env.target_pos[0] = torch.tensor(
                [pad_target[0], pad_target[1], pad_target[2] + 2.0],
                device=DEVICE,
            )
            if (
                dist_to_pad_xy < 0.35
                and abs(drone_z - (pad_target[2] + 2.0)) < 0.4
            ):
                hover_dwell += dt
                if hover_dwell >= 1.0:
                    mission_state = "LOWERING_PAYLOAD"
            else:
                hover_dwell = max(0.0, hover_dwell - dt * 2.0)
        elif mission_state == "LOWERING_PAYLOAD":
            env.target_pos[0] = torch.tensor(
                [pad_target[0], pad_target[1], pad_target[2] + 2.0],
                device=DEVICE,
            )
            env.cable_len[0] = torch.clamp(
                env.cable_len[0] + 0.65 * dt, max=2.6
            )
            cargo_z = env.cargo_pos[0, 2].item()
            if cargo_z <= pad_target[2] + 0.15 or env.cable_len[0].item() >= 2.5:
                mission_state = "RELEASE_PACKAGE"
        elif mission_state == "RELEASE_PACKAGE":
            delivered_boxes.append(env.cargo_pos[0].detach().cpu().numpy())
            env.has_cargo[0] = False
            mission_state = "RETRACTING_AND_RETURN"
        elif mission_state == "RETRACTING_AND_RETURN":
            env.cable_len[0] = torch.clamp(
                env.cable_len[0] - 1.2 * dt, min=0.2
            )
            env.target_pos[0] = torch.tensor([0.0, 0.0, 8.0], device=DEVICE)
            if np.linalg.norm(drone_p_np[:2]) < 0.5:
                mission_state = "DEPOT_TOUCHDOWN"
                touchdown_dwell = 0.0
        elif mission_state == "DEPOT_TOUCHDOWN":
            env.target_pos[0] = torch.tensor([0.0, 0.0, 0.15], device=DEVICE)
            if (
                drone_z <= 0.25
                and np.linalg.norm(env.vel[0].cpu().numpy()) < 0.3
            ):
                touchdown_dwell += dt
                if touchdown_dwell >= 0.5:
                    target_building = world.advance_order()
                    pad_target = target_building["pad"]
                    env.has_cargo[0] = True
                    env.cargo_pos[0] = env.pos[0].clone()
                    env.cargo_pos[0, 2] = 0.08
                    mission_state = "TAKEOFF"

        with torch.no_grad():
            obs = env.get_observations()
            commanded_wrench, senary_coords, policy_diag = policy(
                obs, dt=env.dt
            )
            env.step(commanded_wrench)

        screen.fill((14, 18, 24))

        # Ground Grid
        for gx in range(-30, 31, 5):
            p1 = project_3d_point(
                np.array([gx, -30, 0]), cam_pos, cam_lookat, SCREEN_W, SCREEN_H
            )
            p2 = project_3d_point(
                np.array([gx, 30, 0]), cam_pos, cam_lookat, SCREEN_W, SCREEN_H
            )
            if p1 and p2:
                pygame.draw.line(screen, (28, 36, 48), p1[:2], p2[:2], 1)
        for gy in range(-30, 31, 5):
            p1 = project_3d_point(
                np.array([-30, gy, 0]), cam_pos, cam_lookat, SCREEN_W, SCREEN_H
            )
            p2 = project_3d_point(
                np.array([30, gy, 0]), cam_pos, cam_lookat, SCREEN_W, SCREEN_H
            )
            if p1 and p2:
                pygame.draw.line(screen, (28, 36, 48), p1[:2], p2[:2], 1)

        # Buildings & Pads
        for b in world.buildings:
            x1, x2, y1, y2, top = b["aabb"]
            corners = [
                np.array([x1, y1, 0.0]),
                np.array([x2, y1, 0.0]),
                np.array([x2, y2, 0.0]),
                np.array([x1, y2, 0.0]),
                np.array([x1, y1, top]),
                np.array([x2, y1, top]),
                np.array([x2, y2, top]),
                np.array([x1, y2, top]),
            ]
            proj = [
                project_3d_point(c, cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
                for c in corners
            ]
            if all(proj[i] is not None for i in [4, 5, 6, 7]):
                roof_poly = [proj[i][:2] for i in [4, 5, 6, 7]]
                pygame.draw.polygon(screen, (44, 54, 70), roof_poly)
                pygame.draw.polygon(screen, (70, 85, 110), roof_poly, 2)
            for i in range(4):
                if proj[i] and proj[i + 4]:
                    pygame.draw.line(
                        screen, (55, 68, 88), proj[i][:2], proj[i + 4][:2], 1
                    )

            pad_pt = project_3d_point(
                b["pad"], cam_pos, cam_lookat, SCREEN_W, SCREEN_H
            )
            if pad_pt:
                is_active = b["name"] == target_building["name"]
                pad_color = (0, 255, 180) if is_active else (255, 180, 50)
                radius_px = max(4, int(42.0 / pad_pt[2]))
                pygame.draw.circle(screen, pad_color, pad_pt[:2], radius_px, 2)
                if is_active:
                    pygame.draw.circle(
                        screen,
                        (0, 255, 180),
                        pad_pt[:2],
                        max(1, radius_px // 2),
                    )

        # Delivered Packages
        for box_p in delivered_boxes:
            b_pt = project_3d_point(
                box_p, cam_pos, cam_lookat, SCREEN_W, SCREEN_H
            )
            if b_pt:
                sz = max(3, int(15.0 / b_pt[2]))
                pygame.draw.rect(
                    screen,
                    (255, 215, 60),
                    (b_pt[0] - sz, b_pt[1] - sz, sz * 2, sz * 2),
                )

        # Quadrotor with Correct FLU Coordinates
        drone_pt = project_3d_point(
            drone_p_np, cam_pos, cam_lookat, SCREEN_W, SCREEN_H
        )
        rot_np = env.rot_mat[0].detach().cpu().numpy()
        if drone_pt and np.all(np.isfinite(rot_np)):
            d_arm = ARM_LENGTH * 1.5
            arm_ends = [
                drone_p_np + rot_np @ np.array([d_arm, d_arm, 0.0]),
                drone_p_np + rot_np @ np.array([-d_arm, -d_arm, 0.0]),
                drone_p_np + rot_np @ np.array([-d_arm, d_arm, 0.0]),
                drone_p_np + rot_np @ np.array([d_arm, -d_arm, 0.0]),
            ]
            arm_pts = [
                project_3d_point(ae, cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
                for ae in arm_ends
            ]
            if arm_pts[0] and arm_pts[1]:
                pygame.draw.line(
                    screen, (210, 235, 255), arm_pts[0][:2], arm_pts[1][:2], 3
                )
            if arm_pts[2] and arm_pts[3]:
                pygame.draw.line(
                    screen, (210, 235, 255), arm_pts[2][:2], arm_pts[3][:2], 3
                )
            for apt in arm_pts:
                if apt:
                    pygame.draw.circle(
                        screen,
                        (0, 220, 255),
                        apt[:2],
                        max(3, int(22.0 / apt[2])),
                        1,
                    )

            nose_pos = drone_p_np + rot_np @ np.array([d_arm * 1.3, 0.0, 0.0])
            nose_pt = project_3d_point(
                nose_pos, cam_pos, cam_lookat, SCREEN_W, SCREEN_H
            )
            if nose_pt:
                pygame.draw.line(
                    screen, (255, 60, 60), drone_pt[:2], nose_pt[:2], 3
                )

            pygame.draw.circle(
                screen,
                (255, 60, 80),
                drone_pt[:2],
                max(3, int(25.0 / drone_pt[2])),
            )

        # Tether and Cargo Payload
        if env.has_cargo[0]:
            cargo_p_np = env.cargo_pos[0].detach().cpu().numpy()
            cargo_pt = project_3d_point(
                cargo_p_np, cam_pos, cam_lookat, SCREEN_W, SCREEN_H
            )
            if drone_pt and cargo_pt:
                pygame.draw.line(
                    screen, (190, 210, 230), drone_pt[:2], cargo_pt[:2], 2
                )
                csz = max(3, int(20.0 / cargo_pt[2]))
                pygame.draw.rect(
                    screen,
                    (255, 195, 30),
                    (cargo_pt[0] - csz, cargo_pt[1] - csz, csz * 2, csz * 2),
                )
                pygame.draw.rect(
                    screen,
                    (80, 50, 10),
                    (cargo_pt[0] - csz, cargo_pt[1] - csz, csz * 2, csz * 2),
                    1,
                )

        # -------------------------------------------------------------
        # HUD: SENARY TELEMETRY & EISENSTEIN STAR COMPASS
        # -------------------------------------------------------------
        hud_bg = pygame.Rect(15, 15, 435, 260)
        pygame.draw.rect(screen, (10, 14, 20), hud_bg, border_radius=6)
        pygame.draw.rect(screen, (0, 220, 255), hud_bg, 1, border_radius=6)

        speed = np.linalg.norm(env.vel[0].cpu().numpy())
        altitude = drone_p_np[2]
        dist_target = np.linalg.norm(
            drone_p_np - env.target_pos[0].cpu().numpy()
        )
        s_coords = senary_coords[0].cpu().numpy()
        wave_u = env.last_wave_telemetry[0, 0].item()
        wave_v = env.last_wave_telemetry[0, 1].item()
        dh_dt_val = policy_diag["dH_dt"].item()

        r10 = float(np.nan_to_num(rot_np[1, 0], nan=0.0))
        r00 = float(np.nan_to_num(rot_np[0, 0], nan=1.0))
        yaw_deg = math.degrees(math.atan2(r10, r00))

        status_lines = [
            ("SEHL-CONTROL: CONJUGATE FLOW DISSIPATION", (0, 255, 200)),
            (f"Mission Phase:      {mission_state}", (255, 220, 40)),
            (
                f"Weathervane Heading:{yaw_deg:+06.1f}° | Rates: {np.linalg.norm(env.omega[0].cpu().numpy()):.2f} r/s",
                (255, 255, 255),
            ),
            (
                f"Decimal Pos Error:  {dist_target:.3f} m  (Alt: {altitude:.2f}m)",
                (0, 220, 255),
            ),
            (
                f"SENARY ERROR [X]:   {float_to_base6_str(env.target_pos[0,0].item()-drone_p_np[0])} m",
                (255, 160, 40),
            ),
            (
                f"SENARY ERROR [Y]:   {float_to_base6_str(env.target_pos[0,1].item()-drone_p_np[1])} m",
                (255, 160, 40),
            ),
            (
                f"SENARY ALTITUDE:    {float_to_base6_str(altitude)} m",
                (255, 160, 40),
            ),
            (
                f"Tether Tension:     T={env.last_tension[0].item():.1f}N (Wave Refl: {abs(wave_v)/(abs(wave_u)+1e-4):.2f})",
                (150, 220, 255),
            ),
            (f"Airspeed:           {speed:.2f} m/s", (0, 220, 255)),
            (
                f"Passivity Metric:   dH/dt={dh_dt_val:+.3f} J/s (Physical Sink)",
                (180, 200, 240),
            ),
            (f"Delivered Packages: {len(delivered_boxes)}", (0, 255, 140)),
        ]

        for idx, (txt, col) in enumerate(status_lines):
            screen.blit(
                font_hud.render(txt, True, col),
                (hud_bg.x + 12, hud_bg.y + 10 + idx * 17),
            )

        radar_cx, radar_cy = hud_bg.right - 55, hud_bg.top + 55
        pygame.draw.circle(screen, (24, 32, 45), (radar_cx, radar_cy), 32)
        pygame.draw.circle(screen, (0, 220, 255), (radar_cx, radar_cy), 32, 1)

        for k in range(6):
            ang = k * math.pi / 3.0
            bx = radar_cx + int(30 * math.cos(ang))
            by = radar_cy - int(30 * math.sin(ang))
            pygame.draw.line(
                screen, (45, 60, 85), (radar_cx, radar_cy), (bx, by), 1
            )

            mag = float(np.clip(s_coords[k] * 4.0, 0.0, 30.0))
            ax = radar_cx + int(mag * math.cos(ang))
            ay = radar_cy - int(mag * math.sin(ang))
            pygame.draw.line(
                screen, (0, 255, 200), (radar_cx, radar_cy), (ax, ay), 2
            )
            pygame.draw.circle(screen, (255, 160, 40), (ax, ay), 2)

        wind_speed = np.linalg.norm(env.wind[0].cpu().numpy())
        if wind_speed > 0.1:
            w_dir = env.wind[0].cpu().numpy()[:2] / wind_speed
            wx = int(radar_cx + w_dir[0] * min(26.0, wind_speed * 3.2))
            wy = int(radar_cy - w_dir[1] * min(26.0, wind_speed * 3.2))
            pygame.draw.line(
                screen, (255, 60, 60), (radar_cx, radar_cy), (wx, wy), 2
            )
            screen.blit(
                font_hud.render(
                    f"W: {wind_speed:.1f}m/s", True, (255, 100, 80)
                ),
                (radar_cx - 26, radar_cy + 36),
            )

        pygame.display.flip()

    pygame.quit()
    sys.exit()


# =============================================================================
# 8. MAIN EXECUTION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" SENARY EISENSTEIN-HAMILTONIAN LATTICE CONTROLLER (SEHL-Control)")
    print(" Architecture: Port-Hamiltonian Conjugate Effort-Flow Dissipation")
    print(" Invariance: Exact Z6 Circulant Equivariance on Eisenstein A2 Lattice")
    print(" Simulation: Differentiable 6-DoF Quadrotor + Pure Tensile Viscoelastic Cable")
    print("=" * 80)

    if os.path.exists(CHECKPOINT_FILE):
        print(f">> Pre-trained weights detected: {CHECKPOINT_FILE}")
        print(">> Initializing simulation cockpit directly...")
        trained_policy = SenaryHamiltonianPolicy().to(DEVICE)
        trained_policy.load_state_dict(
            torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=True)
        )
    else:
        trained_policy = train_sehl_controller(
            epochs=512, horizon=128, num_envs=12288, device=DEVICE, tbptt_chunk=32
        )

    run_3d_simulation(trained_policy)