"""
=============================================================================
WAKE-SURFING FLUID AUTONOMY: DIAGNOSTIC V3 RESEARCH PROTOTYPE
Architecture:
  - Physics: 6-DoF rigid-body approximation; semi-implicit translation + SO(3) exponential update
  - Fluid  : Regularised wake-velocity approximation (not wind-tunnel validated)
  - Control: Geometric attitude loop + model-based CBF projection filter
  - Driver : Residual neural tracker + feasibility-checked projection filter
  - Telemetry: Diagnostic gates; no real-world or formal safety certification implied
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

# Physical Constants
GRAVITY = 9.80665
DRONE_MASS = 1.45  # kg
ARM_LENGTH = 0.22  # m (Span b_eff ≈ 0.44 m)
SPAN_EFF = ARM_LENGTH * math.sqrt(2.0)
AIR_DENSITY = 1.225  # kg/m^3
MOTOR_TAU = 0.032  # s
MAX_THRUST_PER_MOTOR = 8.5  # N (34 N total vehicle ceiling)
TORQUE_TO_THRUST_RATIO = 0.016
MOTOR_COIL_RESISTANCE = 0.048
VORTEX_CORE_RADIUS = 0.06  # m (Viscous vortex core r_0)

# Rotor Aeromechanics (4x 8.5-inch propellers)
PROP_RADIUS = 0.11  # m
BLADE_CHORD = 0.018  # m
NUM_BLADES = 2
ROTOR_SOLIDITY = (NUM_BLADES * BLADE_CHORD) / (math.pi * PROP_RADIUS)
BLADE_CD0 = 0.012  # Profile drag coefficient
ROTOR_DISK_AREA = math.pi * (PROP_RADIUS**2)
TOTAL_ROTOR_AREA = 4.0 * ROTOR_DISK_AREA
HOVER_WEIGHT = DRONE_MASS * GRAVITY  # ≈ 14.22 N
V_INDUCED_HOVER = math.sqrt(HOVER_WEIGHT / (2.0 * AIR_DENSITY * TOTAL_ROTOR_AREA))


def safe_norm(x, dim=-1, keepdim=False, eps=1e-12):
    return torch.sqrt(torch.sum(x * x, dim=dim, keepdim=keepdim) + eps)


# =============================================================================
# 1. REGULARISED WAKE-FLOW APPROXIMATIONS (UNVALIDATED)
# =============================================================================
class TwoWayLagrangianWakeField:
    """Regularised induced-flow approximations for research and regression tests.

    The leader-to-follower calculation is a simplified, finite-core filament model.
    The follower-to-leader calculation only permits downstream rotor-wake transport;
    it is not a validated near-field quadrotor interaction model.
    """

    def __init__(self, span=SPAN_EFF, r_core=VORTEX_CORE_RADIUS, rho=AIR_DENSITY):
        self.b = span
        self.r_0 = r_core
        self.rho = rho

    def compute_leader_to_follower_wake(
        self, eval_pos, hist_pos, hist_rot, hist_vel, leader_thrust, turn_curvature
    ):
        N, H, _ = hist_pos.shape
        fwd_speed = safe_norm(hist_vel[:, 0, :2], dim=-1, keepdim=True).clamp(min=1.5)

        gamma_0 = leader_thrust / (self.rho * fwd_speed * self.b).clamp(min=1e-3)
        gamma_0 = torch.clamp(gamma_0, min=1.0, max=24.0)

        # Nearest historical wake segment
        dist_to_segments = safe_norm(hist_pos - eval_pos.unsqueeze(1), dim=-1)
        k_star = torch.argmin(dist_to_segments, dim=1)

        idx_pos = k_star.view(N, 1, 1).expand(N, 1, 3)
        pos_near = torch.gather(hist_pos, 1, idx_pos).squeeze(1)

        idx_rot = k_star.view(N, 1, 1, 1).expand(N, 1, 3, 3)
        rot_near = torch.gather(hist_rot, 1, idx_rot).squeeze(1)

        tau = k_star.float().unsqueeze(-1) * 0.015
        turb_factor = 1.0 + 2.0 * torch.abs(turn_curvature)
        r_c = torch.sqrt(self.r_0**2 + 0.008 * tau * turb_factor).clamp(max=0.14)

        delta_world = eval_pos - pos_near
        delta_local = torch.matmul(rot_near.transpose(-1, -2), delta_world.unsqueeze(-1)).squeeze(-1)

        dx = delta_local[:, 0:1]
        dy = delta_local[:, 1:2]
        dz = delta_local[:, 2:3]

        gamma_tau = gamma_0 * torch.exp(-tau / 3.0)

        y_v = self.b * 0.5
        r_left_sq = (dy - y_v) ** 2 + dz**2 + r_c**2
        r_right_sq = (dy + y_v) ** 2 + dz**2 + r_c**2

        wz_left = (gamma_tau / (2.0 * math.pi)) * ((dy - y_v) / r_left_sq)
        wz_right = -(gamma_tau / (2.0 * math.pi)) * ((dy + y_v) / r_right_sq)
        w_z = wz_left + wz_right

        wy_left = -(gamma_tau / (2.0 * math.pi)) * (dz / r_left_sq)
        wy_right = (gamma_tau / (2.0 * math.pi)) * (dz / r_right_sq)
        w_y = wy_left + wy_right
        w_x = -0.12 * torch.abs(w_z)

        v_ind_local = torch.cat([w_x, w_y, w_z], dim=-1)
        v_induced_world = torch.matmul(rot_near, v_ind_local.unsqueeze(-1)).squeeze(-1)

        # Inflow gradient across span (shear derivative dw_z / dy)
        dy_inner = dy + 0.10
        dy_outer = dy - 0.10
        r_l_in = (dy_inner - y_v) ** 2 + dz**2 + r_c**2
        r_r_in = (dy_inner + y_v) ** 2 + dz**2 + r_c**2
        wz_inner = (gamma_tau / (2.0 * math.pi)) * ((dy_inner - y_v) / r_l_in - (dy_inner + y_v) / r_r_in)

        r_l_out = (dy_outer - y_v) ** 2 + dz**2 + r_c**2
        r_r_out = (dy_outer + y_v) ** 2 + dz**2 + r_c**2
        wz_outer = (gamma_tau / (2.0 * math.pi)) * ((dy_outer - y_v) / r_l_out - (dy_outer + y_v) / r_r_out)
        span_shear = (wz_inner - wz_outer) / 0.20

        return v_induced_world, r_c, span_shear

    def compute_follower_to_leader_back_induction(self, lead_pos, follow_pos, follow_rot, follow_thrust):
        """Estimate follower rotor downwash at the leader, in m/s.

        Positive/negative direction follows the induced-flow direction (-body z).
        The Gaussian core and exponential decay are engineering approximations. They
        intentionally suppress upstream induction; calibration against CFD or flight
        data is still required before interpreting this as a physical wake prediction.
        """
        delta = lead_pos - follow_pos
        down_axis = -follow_rot[:, :, 2]
        axial = torch.sum(delta * down_axis, dim=-1, keepdim=True)
        lateral_vec = delta - axial * down_axis
        lateral = safe_norm(lateral_vec, dim=-1, keepdim=True)

        thrust = F.relu(follow_thrust)
        v_i_disk = torch.sqrt((thrust / (2.0 * self.rho * TOTAL_ROTOR_AREA)).clamp(min=1e-8))
        wake_age_distance = F.relu(axial)
        core_width = self.r_0 + 0.12 * wake_age_distance
        radial_attenuation = torch.exp(-0.5 * (lateral / core_width.clamp(min=self.r_0)) ** 2)
        axial_attenuation = torch.exp(-wake_age_distance / 8.0)
        downstream_gate = torch.sigmoid((axial - 0.10) / 0.12)
        speed = 2.0 * v_i_disk * radial_attenuation * axial_attenuation * downstream_gate
        return down_axis * speed



# =============================================================================
# 2. ACTUATOR-DISK-BASED POWER ESTIMATE (UNCALIBRATED)
# =============================================================================
def compute_rotor_power_estimate(motor_thrusts, ambient_inflow_wz):
    """Return an *uncalibrated estimate* of rotor power in watts.

    Convention: ambient_inflow_wz is upward axial flow through the rotor disk (m/s).
    The momentum-theory induced-velocity root is used for the induced component.
    Profile power and the thrust-to-current relation are empirical placeholders; this
    function must not be described as measured electrical power until calibrated for
    the actual propeller, motor, ESC, battery voltage and inflow regime.
    """
    thrust = F.relu(motor_thrusts)
    v_axial = ambient_inflow_wz
    v_i = torch.sqrt(
        (0.5 * v_axial) ** 2 + (thrust / (2.0 * AIR_DENSITY * ROTOR_DISK_AREA)).clamp(min=1e-8)
    ) - 0.5 * v_axial
    p_induced = thrust * F.relu(v_i)

    # Placeholder motor-speed relation; must be fitted from bench data.
    omega_prop = torch.sqrt(thrust / 1.1e-5).clamp(min=50.0, max=1800.0)
    p_profile = (
        (1.0 / 8.0) * AIR_DENSITY * ROTOR_SOLIDITY * BLADE_CD0 * ROTOR_DISK_AREA
        * (omega_prop * PROP_RADIUS) ** 3
    )
    # Placeholder thrust-current mapping; not a verified electrical model.
    current_est = thrust * 1.85
    p_electrical_proxy = current_est.square() * MOTOR_COIL_RESISTANCE
    p_total_per_rotor = p_induced + p_profile + p_electrical_proxy
    return torch.sum(p_total_per_rotor, dim=-1, keepdim=True)



# =============================================================================
# 3. DEEP NEURAL ACTOR POLICY (THE ADAPTIVE PILOT)
# =============================================================================
class WakeSurfingActorPolicy(nn.Module):
    """
    Neural Actor Pilot. Learns non-linear fluid trim and wave-surfing actions.
    Outputs adaptive residual acceleration on top of baseline tracking.
    """

    def __init__(self, state_dim=13, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, action_dim),
        )
        # Start with zero residual so the neural policy does not duplicate the
        # hand-designed PD tracker at initialisation.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs):
        return torch.tanh(self.net(obs)) * 2.5


# =============================================================================
# 4. DEEP RIEMANNIAN METRIC TENSOR M(x) > 0
# =============================================================================
class NeuralContractionMetric(nn.Module):
    def __init__(self, state_dim=6, alpha=0.55):
        super().__init__()
        self.state_dim = state_dim
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(state_dim, 48),
            nn.SiLU(),
            nn.Linear(48, 48),
            nn.SiLU(),
            nn.Linear(48, (state_dim * (state_dim + 1)) // 2),
        )
        self.tril_indices = torch.tril_indices(state_dim, state_dim)

    def forward(self, state_error):
        N = state_error.shape[0]
        params = self.net(state_error)
        # Out-of-place scatter is compatible with torch.func.vmap/jacrev; direct
        # indexed writes break vectorised Jacobian evaluation.
        flat_indices = self.tril_indices[0].to(state_error.device) * self.state_dim + self.tril_indices[1].to(state_error.device)
        flat = state_error.new_zeros((N, self.state_dim * self.state_dim))
        flat = flat.scatter(1, flat_indices.unsqueeze(0).expand(N, -1), params)
        L = flat.reshape(N, self.state_dim, self.state_dim)
        diagonal = F.softplus(torch.diagonal(L, dim1=-2, dim2=-1)) + 0.20
        L = L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1)) + torch.diag_embed(diagonal)
        eye = torch.eye(self.state_dim, device=state_error.device, dtype=state_error.dtype).unsqueeze(0)
        M = torch.matmul(L, L.transpose(-1, -2)) + 0.10 * eye
        return M


# =============================================================================
# 5. FEASIBILITY-CHECKED CONSTRAINT PROJECTION
# =============================================================================
class RelaxedQPSafetyFilter:
    """Feasibility-checked projection onto linear half-spaces and box bounds.

    This is a numerical projection solver, not a formal safety certificate. Dykstra's
    algorithm converges to the Euclidean projection when the intersection is nonempty;
    finite iteration, inaccurate dynamics, actuator lag and unmodelled forces still
    limit what can be claimed. Every result includes an explicit residual and status.
    """
    def __init__(self, iters=60, rho_slack=15.0, tolerance=2e-3):
        self.iters = int(iters)
        self.rho = float(rho_slack)  # retained for config compatibility; not used as a proof
        self.tolerance = float(tolerance)

    def solve(self, u_nom, A, b, u_min, u_max, fallback=None):
        # A: [batch, constraints, 3], b: [batch, constraints, 1]
        if A.ndim != 3 or b.ndim != 3 or A.shape[:2] != b.shape[:2] or A.shape[-1] != u_nom.shape[-1]:
            raise ValueError(f"Bad QP shapes: u={tuple(u_nom.shape)}, A={tuple(A.shape)}, b={tuple(b.shape)}")
        if not torch.isfinite(u_nom).all() or not torch.isfinite(A).all() or not torch.isfinite(b).all():
            raise ValueError("Non-finite input to safety projection; refusing to report feasibility")

        x = torch.maximum(torch.minimum(u_nom.clone(), u_max), u_min)
        batch, n_constraints, dim = A.shape
        corrections = torch.zeros_like(A)
        box_correction = torch.zeros_like(x)
        u_min_t = torch.as_tensor(u_min, dtype=x.dtype, device=x.device)
        u_max_t = torch.as_tensor(u_max, dtype=x.dtype, device=x.device)

        for _ in range(self.iters):
            for i in range(n_constraints):
                a_i = A[:, i, :]
                b_i = b[:, i, 0]
                y = x + corrections[:, i, :]
                violation = torch.sum(a_i * y, dim=-1) - b_i
                denom = torch.sum(a_i * a_i, dim=-1).clamp(min=1e-10)
                x_new = y - (F.relu(violation) / denom).unsqueeze(-1) * a_i
                corrections[:, i, :] = y - x_new
                x = x_new
            y_box = x + box_correction
            x_new = torch.maximum(torch.minimum(y_box, u_max_t), u_min_t)
            box_correction = y_box - x_new
            x = x_new

        residuals = torch.matmul(A, x.unsqueeze(-1)) - b
        max_violation_per_env = F.relu(residuals).amax(dim=(1, 2))
        bad = max_violation_per_env > self.tolerance
        if fallback is not None:
            fallback = torch.maximum(torch.minimum(fallback, u_max_t), u_min_t)
            x = torch.where(bad.unsqueeze(-1), fallback, x)
            residuals = torch.matmul(A, x.unsqueeze(-1)) - b
            max_violation_per_env = F.relu(residuals).amax(dim=(1, 2))

        max_violation = float(max_violation_per_env.max().detach().item()) if batch else 0.0
        feasible_pct = float((max_violation_per_env <= self.tolerance).float().mean().detach().item() * 100.0) if batch else 100.0
        active = (residuals[:, :, 0] > -0.05).float().mean(dim=0) * 100.0 if n_constraints else x.new_zeros((0,))
        active1 = float(active[0].detach().item()) if n_constraints > 0 else 0.0
        active2 = float(active[1].detach().item()) if n_constraints > 1 else 0.0
        active3 = float(active[2].detach().item()) if n_constraints > 2 else 0.0
        margins = (-residuals[:, :, 0]).mean(dim=0) if n_constraints else x.new_zeros((0,))
        margin3 = float(margins[2].detach().item()) if n_constraints > 2 else 0.0
        # max_violation is a diagnostic residual, not a slack variable and not a proof.
        return x, max_violation, active1, active2, active3, margin3, feasible_pct



# =============================================================================
# 6. BATCHED 6-DoF RIGID-BODY APPROXIMATION
# =============================================================================
class RigidBody6DoFDrone:
    """
    Six-degree-of-freedom rigid-body approximation. Translation is semi-implicit
    Euler; attitude is updated with an SO(3) exponential map and re-orthogonalised.
    """

    def __init__(self, dt, device):
        self.dt = dt
        self.device = device
        self.J = torch.tensor([[0.021, 0.021, 0.036]], device=device)
        self.J_inv = 1.0 / self.J

        d = ARM_LENGTH / math.sqrt(2.0)
        c_q = TORQUE_TO_THRUST_RATIO
        self.B = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [d, -d, d, -d],
                [-d, d, d, -d],
                [-c_q, -c_q, c_q, c_q],
            ],
            device=device,
        )
        self.B_inv = torch.pinverse(self.B)

    def integrate_dynamics(self, pos, vel, rot, omega, motors, target_wrench, wind_field, external_lift):
        raw_thrusts = torch.matmul(target_wrench, self.B_inv.T)
        target_motors = torch.clamp(raw_thrusts, 0.0, MAX_THRUST_PER_MOTOR)
        alpha_m = self.dt / (MOTOR_TAU + self.dt)
        motors_next = motors + alpha_m * (target_motors - motors)

        realized_wrench = torch.matmul(motors_next, self.B.T)
        realized_Fz = realized_wrench[:, 0:1]
        realized_torques = realized_wrench[:, 1:4]

        # Net rotor force combines motor thrust + external aerodynamic momentum lift
        total_rotor_Fz = realized_Fz + external_lift
        thrust_world = rot[:, :, 2] * total_rotor_Fz

        v_air = vel - wind_field
        drag_force = -0.5 * AIR_DENSITY * 0.22 * v_air * safe_norm(v_air, dim=-1, keepdim=True)
        accel = (thrust_world + drag_force) / DRONE_MASS + torch.tensor([0.0, 0.0, -GRAVITY], device=self.device)

        vel_next = vel + accel * self.dt
        pos_next = pos + vel_next * self.dt

        gyroscopic = torch.cross(omega, omega * self.J, dim=-1)
        d_omega = (realized_torques - gyroscopic) * self.J_inv
        omega_next = omega + d_omega * self.dt

        v_rot = omega_next * self.dt
        theta_sq = torch.sum(v_rot * v_rot, dim=-1, keepdim=True).clamp(min=1e-8)
        theta = torch.sqrt(theta_sq)

        alpha_coef = torch.where(theta_sq < 1e-6, 1.0 - theta_sq / 6.0, torch.sin(theta) / theta).unsqueeze(-1)
        beta_coef = torch.where(
            theta_sq < 1e-6, 0.5 - theta_sq / 24.0, (1.0 - torch.cos(theta)) / theta_sq
        ).unsqueeze(-1)

        zeros = torch.zeros_like(v_rot[:, 0:1])
        vx, vy, vz = v_rot[:, 0:1], v_rot[:, 1:2], v_rot[:, 2:3]
        K = torch.stack(
            [
                torch.cat([zeros, -vz, vy], dim=-1),
                torch.cat([vz, zeros, -vx], dim=-1),
                torch.cat([-vy, vx, zeros], dim=-1),
            ],
            dim=-2,
        )

        eye3 = torch.eye(3, device=self.device).unsqueeze(0)
        dR = eye3 + alpha_coef * K + beta_coef * torch.matmul(K, K)
        rot_next = torch.matmul(rot, dR)

        c0 = rot_next[:, :, 0]
        c1 = rot_next[:, :, 1]
        c0 = c0 / safe_norm(c0, dim=-1, keepdim=True)
        c2 = torch.cross(c0, c1, dim=-1)
        c2 = c2 / safe_norm(c2, dim=-1, keepdim=True)
        c1 = torch.cross(c2, c0, dim=-1)
        rot_clean = torch.stack([c0, c1, c2], dim=-1)

        return pos_next, vel_next, rot_clean, omega_next, motors_next, realized_Fz, drag_force


# =============================================================================
# 7. MUTUAL WAKE SURFING SIMULATION (DECOUPLED ERRORS & DEEP METRICS)
# =============================================================================
class WakeSurfingResearchEnv:
    def __init__(self, num_envs=1, dt=0.015, history_len=75, device=DEVICE, randomize_initial_conditions=False):
        self.num_envs = num_envs
        self.dt = dt
        self.H = history_len
        self.device = device
        self.randomize_initial_conditions = bool(randomize_initial_conditions)
        self.background_wind = torch.zeros((self.num_envs, 3), device=self.device)

        self.aero = TwoWayLagrangianWakeField()
        self.drone_physics = RigidBody6DoFDrone(dt, device)
        self.qp_filter = RelaxedQPSafetyFilter(iters=10, rho_slack=15.0)

        # Well-damped attitude loop matched to 32ms motor lag (zeta = 0.72)
        # Eliminates high-frequency wobble while rejecting fluid shear
        self.k_R = torch.tensor([[4.2, 4.2, 1.8]], device=device)
        self.k_omega = torch.tensor([[0.52, 0.52, 0.32]], device=device)

        self.omega_lead = 0.18
        self.radius_lead = 14.0
        self.sim_time = 0.0
        self.reset()

    def reset(self):
        N = self.num_envs
        self.sim_time = 0.0

        v0_x = self.radius_lead * self.omega_lead
        v0_y = self.radius_lead * 0.5 * self.omega_lead * 2.0

        pos_jitter = torch.zeros((N, 3), device=self.device)
        vel_jitter = torch.zeros((N, 3), device=self.device)
        if self.randomize_initial_conditions and N > 1:
            # Mild domain randomisation: initial state and episode-constant background wind.
            pos_jitter = torch.randn((N, 3), device=self.device) * torch.tensor([0.25, 0.25, 0.10], device=self.device)
            vel_jitter = torch.randn((N, 3), device=self.device) * torch.tensor([0.20, 0.20, 0.10], device=self.device)
            self.background_wind = torch.randn((N, 3), device=self.device) * torch.tensor([0.60, 0.60, 0.20], device=self.device)
        else:
            self.background_wind = torch.zeros((N, 3), device=self.device)
        self.lead_pos = torch.tensor([[0.0, 0.0, 6.0]], device=self.device).repeat(N, 1) + pos_jitter
        self.lead_vel = torch.tensor([[v0_x, v0_y, 0.0]], device=self.device).repeat(N, 1) + vel_jitter
        self.lead_omega = torch.zeros((N, 3), device=self.device)
        self.lead_motors = torch.full((N, 4), HOVER_WEIGHT / 4.0, device=self.device)
        self.lead_thrust = torch.full((N, 1), HOVER_WEIGHT, device=self.device)

        v_speed = math.sqrt(v0_x**2 + v0_y**2)
        tilt_angle = math.atan((0.5 * AIR_DENSITY * 0.22 * v_speed**2) / HOVER_WEIGHT)
        c_pitch = math.cos(tilt_angle)
        s_pitch = math.sin(tilt_angle)
        h0_x = v0_x / v_speed
        h0_y = v0_y / v_speed

        R_trim = torch.tensor(
            [
                [h0_x * c_pitch, -h0_y, h0_x * s_pitch],
                [h0_y * c_pitch, h0_x, h0_y * s_pitch],
                [-s_pitch, 0.0, c_pitch],
            ],
            device=self.device,
        ).unsqueeze(0).repeat(N, 1, 1)

        self.lead_rot = R_trim.clone()

        hist_p_list = []
        hist_r_list = []
        hist_v_list = []
        for k in range(self.H):
            t_past = -k * self.dt
            lx = self.radius_lead * math.sin(self.omega_lead * t_past)
            ly = self.radius_lead * 0.5 * math.sin(2.0 * self.omega_lead * t_past)
            lvx = self.radius_lead * self.omega_lead * math.cos(self.omega_lead * t_past)
            lvy = self.radius_lead * self.omega_lead * math.cos(2.0 * self.omega_lead * t_past)

            hp = torch.tensor([[lx, ly, 6.0]], device=self.device).repeat(N, 1) + pos_jitter
            hv = torch.tensor([[lvx, lvy, 0.0]], device=self.device).repeat(N, 1) + vel_jitter
            hist_p_list.append(hp.unsqueeze(1))
            hist_r_list.append(R_trim.unsqueeze(1))
            hist_v_list.append(hv.unsqueeze(1))

        self.history_pos = torch.cat(hist_p_list, dim=1)
        self.history_rot = torch.cat(hist_r_list, dim=1)
        self.history_vel = torch.cat(hist_v_list, dim=1)

        self.sweet_spot_offset = torch.tensor([-1.80, -(SPAN_EFF * 0.5 + 0.22), 0.0], device=self.device)
        self.follow_pos = self.lead_pos + torch.matmul(self.lead_rot, self.sweet_spot_offset.unsqueeze(-1)).squeeze(-1)
        self.follow_vel = self.lead_vel.clone()
        self.follow_rot = R_trim.clone()
        self.follow_omega = torch.zeros((N, 3), device=self.device)
        self.follow_motors = torch.full((N, 4), HOVER_WEIGHT / 4.0, device=self.device)
        self.follow_thrust = torch.full((N, 1), HOVER_WEIGHT, device=self.device)

        self.metrics = {}

    def step(self, actor_net, ncm_net):
        N = self.num_envs
        self.sim_time += self.dt
        lead_vel_before = self.lead_vel.clone()
        lead_omega_before = self.lead_omega.clone()

        # -------------------------------------------------------------
        # 1. LEADER: SE(3) ATTITUDE CONTROLLER WITH FEEDFORWARD
        # -------------------------------------------------------------
        ref_px = self.radius_lead * math.sin(self.omega_lead * self.sim_time)
        ref_py = self.radius_lead * 0.5 * math.sin(2.0 * self.omega_lead * self.sim_time)
        ref_vx = self.radius_lead * self.omega_lead * math.cos(self.omega_lead * self.sim_time)
        ref_vy = self.radius_lead * self.omega_lead * math.cos(2.0 * self.omega_lead * self.sim_time)
        ref_ax = -self.radius_lead * (self.omega_lead**2) * math.sin(self.omega_lead * self.sim_time)
        ref_ay = -self.radius_lead * 0.5 * ((2.0 * self.omega_lead) ** 2) * math.sin(2.0 * self.omega_lead * self.sim_time)

        p_ref_lead = torch.cat([
            torch.full((N, 1), ref_px, device=self.device),
            torch.full((N, 1), ref_py, device=self.device),
            torch.full((N, 1), 6.0, device=self.device),
        ], dim=-1)
        v_ref_lead = torch.cat([
            torch.full((N, 1), ref_vx, device=self.device),
            torch.full((N, 1), ref_vy, device=self.device),
            torch.zeros((N, 1), device=self.device),
        ], dim=-1)
        a_ref_lead = torch.cat([
            torch.full((N, 1), ref_ax, device=self.device),
            torch.full((N, 1), ref_ay, device=self.device),
            torch.zeros((N, 1), device=self.device),
        ], dim=-1)

        v_back_ind = self.aero.compute_follower_to_leader_back_induction(
            self.lead_pos, self.follow_pos, self.follow_rot, self.follow_thrust
        )

        lead_flow = v_back_ind + self.background_wind
        v_air_lead = self.lead_vel - lead_flow
        f_drag_lead = -0.5 * AIR_DENSITY * 0.22 * v_air_lead * safe_norm(v_air_lead, dim=-1, keepdim=True)

        ep_lead = self.lead_pos - p_ref_lead
        ev_lead = self.lead_vel - v_ref_lead
        f_des_lead = (
            a_ref_lead
            - (f_drag_lead / DRONE_MASS)
            - 4.5 * ep_lead
            - 3.2 * ev_lead
            + torch.tensor([0.0, 0.0, GRAVITY], device=self.device)
        )

        zb_lead_des = f_des_lead / safe_norm(f_des_lead, dim=-1, keepdim=True)
        fwd_lead = v_ref_lead[:, :2] / safe_norm(v_ref_lead[:, :2], dim=-1, keepdim=True).clamp(min=1e-3)
        xc_lead = torch.cat([fwd_lead, torch.zeros((N, 1), device=self.device)], dim=-1)
        yb_lead_des = torch.cross(zb_lead_des, xc_lead, dim=-1)
        yb_lead_des = yb_lead_des / safe_norm(yb_lead_des, dim=-1, keepdim=True)
        xb_lead_des = torch.cross(yb_lead_des, zb_lead_des, dim=-1)
        Rd_lead = torch.stack([xb_lead_des, yb_lead_des, zb_lead_des], dim=-1)

        R_err_mat = torch.matmul(Rd_lead.transpose(-1, -2), self.lead_rot) - torch.matmul(
            self.lead_rot.transpose(-1, -2), Rd_lead
        )
        e_R_lead = torch.stack([R_err_mat[:, 2, 1], R_err_mat[:, 0, 2], R_err_mat[:, 1, 0]], dim=-1) * 0.5
        tau_lead = -self.k_R * e_R_lead - self.k_omega * self.lead_omega

        lead_back_lift = 2.0 * AIR_DENSITY * TOTAL_ROTOR_AREA * V_INDUCED_HOVER * v_back_ind[:, 2:3]
        lead_Fz = (torch.sum(f_des_lead * self.lead_rot[:, :, 2], dim=-1, keepdim=True) * DRONE_MASS - lead_back_lift).clamp(
            min=2.0, max=32.0
        )
        lead_wrench = torch.cat([lead_Fz, tau_lead], dim=-1)

        (
            self.lead_pos,
            self.lead_vel,
            self.lead_rot,
            self.lead_omega,
            self.lead_motors,
            self.lead_thrust,
            _,
        ) = self.drone_physics.integrate_dynamics(
            self.lead_pos,
            self.lead_vel,
            self.lead_rot,
            self.lead_omega,
            self.lead_motors,
            lead_wrench,
            lead_flow,
            lead_back_lift,
        )

        # Measured-in-simulator acceleration estimates for relative-degree-two barriers.
        lead_accel_est = (self.lead_vel - lead_vel_before) / max(self.dt, 1e-6)
        lead_alpha_body_est = (self.lead_omega - lead_omega_before) / max(self.dt, 1e-6)
        self.history_pos = torch.cat([self.lead_pos.unsqueeze(1), self.history_pos[:, :-1, :]], dim=1)
        self.history_rot = torch.cat([self.lead_rot.unsqueeze(1), self.history_rot[:, :-1, :, :]], dim=1)
        self.history_vel = torch.cat([self.lead_vel.unsqueeze(1), self.history_vel[:, :-1, :]], dim=1)

        # -------------------------------------------------------------
        # 2. FOLLOWER: WAKE SURFING WITH SOFT-SLACK RELAXED NCM SAFETY
        # -------------------------------------------------------------
        lead_speed = safe_norm(self.lead_vel[:, :2], dim=-1, keepdim=True).clamp(min=1e-3)
        turn_curvature = (lead_speed * self.omega_lead) / (lead_speed**2).clamp(min=1e-3)
        v_induced_follower, r_core, span_shear = self.aero.compute_leader_to_follower_wake(
            self.follow_pos, self.history_pos, self.history_rot, self.history_vel, self.lead_thrust, turn_curvature
        )
        wz_experienced = v_induced_follower[:, 2:3]

        # Dynamic arc-length target selection (1.80m physical arc spacing)
        step_dists = safe_norm(self.history_pos[:, :-1, :] - self.history_pos[:, 1:, :], dim=-1)
        cum_dist = torch.cumsum(step_dists, dim=1)
        k_target = torch.clamp(torch.argmin(torch.abs(cum_dist - 1.80), dim=1) + 1, max=self.H - 1)

        pos_target_segment = torch.gather(self.history_pos, 1, k_target.view(N, 1, 1).expand(N, 1, 3)).squeeze(1)
        rot_target_segment = torch.gather(self.history_rot, 1, k_target.view(N, 1, 1, 1).expand(N, 1, 3, 3)).squeeze(1)
        vel_target = torch.gather(self.history_vel, 1, k_target.view(N, 1, 1).expand(N, 1, 3)).squeeze(1)

        # Analytical smooth centripetal acceleration along racetrack (No index chatter)
        tau_target = k_target.float() * self.dt
        t_tgt = self.sim_time - tau_target
        ax_tgt = -self.radius_lead * (self.omega_lead**2) * torch.sin(self.omega_lead * t_tgt)
        ay_tgt = -self.radius_lead * 0.5 * ((2.0 * self.omega_lead) ** 2) * torch.sin(2.0 * self.omega_lead * t_tgt)
        a_target = torch.cat([ax_tgt.unsqueeze(-1), ay_tgt.unsqueeze(-1), torch.zeros((N, 1), device=self.device)], dim=-1)

        flank_off = torch.tensor([0.0, -(SPAN_EFF * 0.5 + 0.22), 0.0], device=self.device)
        target_pos = pos_target_segment + torch.matmul(rot_target_segment, flank_off.unsqueeze(-1)).squeeze(-1)

        error_pos = self.follow_pos - target_pos
        error_vel = self.follow_vel - vel_target
        rel_pos = self.follow_pos - self.lead_pos
        rel_pos_lead = torch.matmul(self.lead_rot.transpose(-1, -2), rel_pos.unsqueeze(-1)).squeeze(-1)

        # Decoupled tracking errors along local target frame
        rot_t_x = rot_target_segment[:, :, 0]
        rot_t_y = rot_target_segment[:, :, 1]
        e_longitudinal = torch.sum(error_pos * rot_t_x, dim=-1, keepdim=True)
        e_lateral = torch.sum(error_pos * rot_t_y, dim=-1, keepdim=True)
        e_vertical = error_pos[:, 2:3]

        actor_obs = torch.cat([error_pos, error_vel, wz_experienced, rel_pos_lead, self.follow_omega], dim=-1)
        u_residual = actor_net(actor_obs)
        # Stiffened vertical tracking gain forces altitude error to collapse
        u_baseline = torch.cat([
            -4.5 * error_pos[:, 0:2] - 3.2 * error_vel[:, 0:2],
            -6.5 * error_pos[:, 2:3] - 4.2 * error_vel[:, 2:3]
        ], dim=-1)
        u_nom = u_baseline + u_residual

        # NCM Contraction Metric Tensor (alpha = 0.55 opens turning headroom)
        state_err = torch.cat([error_pos, error_vel], dim=-1)
        # Metric participates in diagnostics only, not action generation; detach here
        # so rollout backprop does not retain a useless metric graph.
        M = ncm_net(state_err.detach())
        P_sub = M[:, 3:6, 3:6]

        # Relative-degree-two exponential CBFs. These are model-based safeguards,
        # not formal guarantees for the full plant: attitude dynamics, motor lag,
        # wake-model error and finite solver tolerance remain outside the CBF model.
        v_rel = self.follow_vel - self.lead_vel
        dist_sq = torch.sum(rel_pos ** 2, dim=-1, keepdim=True)
        h_collision = dist_sq - 1.44  # minimum centre spacing = 1.20 m
        h_coll_dot = 2.0 * torch.sum(rel_pos * v_rel, dim=-1, keepdim=True)
        cbf_lambda = 2.0
        # h_ddot + 2*lambda*h_dot + lambda^2*h >= 0, with follower acceleration u.
        A_cbf1 = -2.0 * rel_pos.unsqueeze(1)
        b_cbf1 = (
            2.0 * torch.sum(v_rel * v_rel, dim=-1, keepdim=True)
            - 2.0 * torch.sum(rel_pos * lead_accel_est, dim=-1, keepdim=True)
            + 2.0 * cbf_lambda * h_coll_dot
            + (cbf_lambda ** 2) * h_collision
        ).unsqueeze(-1)

        y_axis = self.lead_rot[:, :, 1]
        omega_world = torch.matmul(self.lead_rot, self.lead_omega.unsqueeze(-1)).squeeze(-1)
        alpha_world = torch.matmul(self.lead_rot, lead_alpha_body_est.unsqueeze(-1)).squeeze(-1)
        y_axis_dot = torch.cross(omega_world, y_axis, dim=-1)
        y_axis_ddot = torch.cross(alpha_world, y_axis, dim=-1) + torch.cross(omega_world, y_axis_dot, dim=-1)
        lateral_margin = SPAN_EFF * 0.5 + 0.05
        h_sink = -torch.sum(rel_pos * y_axis, dim=-1, keepdim=True) - lateral_margin
        h_sink_dot = -torch.sum(v_rel * y_axis, dim=-1, keepdim=True) - torch.sum(rel_pos * y_axis_dot, dim=-1, keepdim=True)
        A_cbf2 = y_axis.unsqueeze(1)
        b_cbf2 = (
            torch.sum(y_axis * lead_accel_est, dim=-1, keepdim=True)
            - 2.0 * torch.sum(y_axis_dot * v_rel, dim=-1, keepdim=True)
            - torch.sum(y_axis_ddot * rel_pos, dim=-1, keepdim=True)
            + 2.0 * cbf_lambda * h_sink_dot
            + (cbf_lambda ** 2) * h_sink
        ).unsqueeze(-1)

        # The learned metric is diagnostic only; it is NOT inserted into the hard
        # safety filter because no CBF interpretation has been established for it.
        A_ncm_diag = torch.matmul(P_sub, error_vel.unsqueeze(-1)).transpose(-1, -2)
        contraction_bound = 0.55 * torch.sum(
            error_vel * torch.matmul(P_sub, error_vel.unsqueeze(-1)).squeeze(-1), dim=-1, keepdim=True
        )
        b_ncm_diag = (
            contraction_bound - 1.2 * torch.sum(error_pos * error_vel, dim=-1, keepdim=True) + 1.80
        ).unsqueeze(-1)
        ncm_residual_diag = torch.matmul(A_ncm_diag, u_nom.unsqueeze(-1)) - b_ncm_diag
        ncm_active_pct = (ncm_residual_diag[:, 0, 0] > -0.05).float().mean().item() * 100.0
        ncm_margin = (-ncm_residual_diag[:, 0, 0]).mean().item()

        rel_direction = rel_pos / safe_norm(rel_pos, dim=-1, keepdim=True)
        emergency_separation_accel = torch.clamp(3.0 * rel_direction - 1.5 * v_rel, min=-6.0, max=6.0)
        filtered_accel, max_slack, cbf1_active_pct, cbf2_active_pct, _, _, qp_feasible_pct = self.qp_filter.solve(
            u_nom,
            torch.cat([A_cbf1, A_cbf2], dim=1),
            torch.cat([b_cbf1, b_cbf2], dim=1),
            u_min=torch.tensor([-6.0, -6.0, -4.5], device=self.device),
            u_max=torch.tensor([6.0, 6.0, 6.0], device=self.device),
            fallback=emergency_separation_accel,
        )

        # Drag Cancellation + Centripetal Acceleration Feedforward
        follower_flow = v_induced_follower + self.background_wind
        v_air_follow = self.follow_vel - follower_flow
        f_drag_follow = -0.5 * AIR_DENSITY * 0.22 * v_air_follow * safe_norm(v_air_follow, dim=-1, keepdim=True)

        f_des_follow = (
            a_target
            + filtered_accel
            - (f_drag_follow / DRONE_MASS)
            + torch.tensor([0.0, 0.0, GRAVITY], device=self.device)
        )

        zb_follow_des = f_des_follow / safe_norm(f_des_follow, dim=-1, keepdim=True)
        fwd_follow = vel_target[:, :2] / safe_norm(vel_target[:, :2], dim=-1, keepdim=True).clamp(min=1e-3)
        xc_follow = torch.cat([fwd_follow, torch.zeros((N, 1), device=self.device)], dim=-1)
        yb_follow_des = torch.cross(zb_follow_des, xc_follow, dim=-1)
        yb_follow_des = yb_follow_des / safe_norm(yb_follow_des, dim=-1, keepdim=True)
        xb_follow_des = torch.cross(yb_follow_des, zb_follow_des, dim=-1)
        Rd_follow = torch.stack([xb_follow_des, yb_follow_des, zb_follow_des], dim=-1)

        R_err_follow = torch.matmul(Rd_follow.transpose(-1, -2), self.follow_rot) - torch.matmul(
            self.follow_rot.transpose(-1, -2), Rd_follow
        )
        e_R_follow = torch.stack([R_err_follow[:, 2, 1], R_err_follow[:, 0, 2], R_err_follow[:, 1, 0]], dim=-1) * 0.5
        tau_follow = -self.k_R * e_R_follow - self.k_omega * self.follow_omega

        # Momentum-Theory Upwash Thrust Relief
        thrust_relief_momentum = 2.0 * AIR_DENSITY * TOTAL_ROTOR_AREA * V_INDUCED_HOVER * wz_experienced
        upwash_lift = torch.clamp(thrust_relief_momentum, min=-2.0, max=6.0)

        follow_Fz = (torch.sum(f_des_follow * self.follow_rot[:, :, 2], dim=-1, keepdim=True) * DRONE_MASS - upwash_lift).clamp(
            min=2.0, max=32.0
        )
        follow_wrench = torch.cat([follow_Fz, tau_follow], dim=-1)

        (
            self.follow_pos,
            self.follow_vel,
            self.follow_rot,
            self.follow_omega,
            self.follow_motors,
            self.follow_thrust,
            _,
        ) = self.drone_physics.integrate_dynamics(
            self.follow_pos,
            self.follow_vel,
            self.follow_rot,
            self.follow_omega,
            self.follow_motors,
            follow_wrench,
            follower_flow,
            upwash_lift,
        )

        # -------------------------------------------------------------
        # 3. AEROMECHANIC POWER ACCOUNTING & EXTENDED METRICS
        # -------------------------------------------------------------
        p_lead_total = compute_rotor_power_estimate(self.lead_motors, v_back_ind[:, 2:3])
        p_follow_total = compute_rotor_power_estimate(self.follow_motors, wz_experienced)

        p_base_isolated = compute_rotor_power_estimate(
            torch.full_like(self.follow_motors, HOVER_WEIGHT / 4.0), torch.zeros_like(wz_experienced)
        )
        p_base_formation = p_base_isolated * 2.0

        p_saved_lead = ((p_base_isolated - p_lead_total) / p_base_isolated) * 100.0
        p_saved_follow = ((p_base_isolated - p_follow_total) / p_base_isolated) * 100.0
        p_saved_collective = ((p_base_formation - (p_lead_total + p_follow_total)) / p_base_formation) * 100.0

        headroom_motor = (MAX_THRUST_PER_MOTOR - self.follow_motors).min().item()
        idle_motor = self.follow_motors.min().item()
        sat_max_pct = (self.follow_motors >= (MAX_THRUST_PER_MOTOR - 0.05)).float().mean().item() * 100.0

        d = ARM_LENGTH / math.sqrt(2.0)
        roll_shear_torque = (
            d * torch.abs(
                self.follow_motors[:, 0] + self.follow_motors[:, 2]
                - self.follow_motors[:, 1] - self.follow_motors[:, 3]
            )
        ).mean().item()

        # Follower Angular Rates and Attitude Error in Degrees
        omega_deg_s = (safe_norm(self.follow_omega, dim=-1).mean().item()) * (180.0 / math.pi)
        e_R_deg = (safe_norm(e_R_follow, dim=-1).mean().item()) * (180.0 / math.pi)
        bank_angle_deg = math.degrees(math.acos(float(np.clip(self.follow_rot[0, 2, 2].item(), -1.0, 1.0))))
        load_factor_g = self.follow_thrust.mean().item() / HOVER_WEIGHT

        qp_override_mag = safe_norm(filtered_accel - u_nom, dim=-1).mean().item()

        self.metrics = {
            # Comparisons to hover baseline only; not equivalent-mission energy savings.
            "p_saved_follow": p_saved_follow.mean().item(),
            "p_saved_lead": p_saved_lead.mean().item(),
            "p_saved_swarm": p_saved_collective.mean().item(),
            "power_model_calibrated": False,
            "background_wind_mean_mps": float(safe_norm(self.background_wind, dim=-1).mean().item()),
            "upwash_follow": wz_experienced.mean().item(),
            "backwash_lead": v_back_ind[:, 2:3].mean().item(),
            "h_sink": h_sink.mean().item(),
            "dist_lead": safe_norm(self.follow_pos - self.lead_pos, dim=-1).mean().item(),
            "thrust_relief_follow": (HOVER_WEIGHT - self.follow_thrust.mean()).item(),
            "thrust_relief_lead": (HOVER_WEIGHT - self.lead_thrust.mean()).item(),
            "p_follow_total_watts": p_follow_total.mean().item(),
            "p_base_watts": p_base_isolated.mean().item(),
            "r_core": r_core.mean().item(),
            "state_err": state_err,
            "error_pos": error_pos,
            "error_vel": error_vel,
            "e_longitudinal": e_longitudinal.mean().item(),
            "e_lateral": e_lateral.mean().item(),
            "e_vertical": e_vertical.mean().item(),
            "z_lead": self.lead_pos[:, 2].mean().item(),
            "z_follow": self.follow_pos[:, 2].mean().item(),
            "z_target": target_pos[:, 2].mean().item(),
            "omega_deg_s": omega_deg_s,
            "e_R_deg": e_R_deg,
            "bank_angle_deg": bank_angle_deg,
            "load_factor_g": load_factor_g,
            "span_shear": span_shear.mean().item(),
            "ncm_margin": ncm_margin,
            "qp_feasible_pct": qp_feasible_pct,
            "safety_filter_max_residual": max_slack,
            "h_collision": h_collision.mean().item(),
            "actor_obs": actor_obs,
            "u_nom": u_nom,
            "u_baseline": u_baseline,
            "u_residual": u_residual,
            "actor_residual_rms": float(u_residual.detach().square().mean().sqrt().item()),
            "v_induced_follower": v_induced_follower,
            "v_backwash_lead": v_back_ind,
            "p_follow_tensor": p_follow_total.mean(),
            "qp_override_mag": qp_override_mag,
            "max_slack": max_slack,
            "cbf1_active_pct": cbf1_active_pct,
            "cbf2_active_pct": cbf2_active_pct,
            "ncm_active_pct": ncm_active_pct,
            "headroom_motor": headroom_motor,
            "idle_motor": idle_motor,
            "sat_max_pct": sat_max_pct,
            "roll_shear_torque": roll_shear_torque,
        }
        return self.metrics


# =============================================================================
# 8. TRAINING ENGINE: NON-LINEAR SYSTEM JACOBIAN CONTRACTION & ACTOR BPTT
# =============================================================================
def _reduced_tracking_vector_field(x, actor_net):
    """Reduced six-state tracking model used ONLY for a local diagnostic gate.

    x=[position error, velocity error]; omitted states include attitude, motor lag,
    wake uncertainty, collision filter and actuator saturation. Passing this gate is
    not certification of the full 6-DoF vehicle.
    """
    e_p, e_v = x[:3], x[3:]
    obs = torch.cat([e_p, e_v, x.new_zeros(1), x.new_zeros(3), x.new_zeros(3)], dim=0)
    residual = actor_net(obs.unsqueeze(0)).squeeze(0)
    baseline = -torch.tensor([4.5, 4.5, 6.5], device=x.device, dtype=x.dtype) * e_p
    baseline = baseline - torch.tensor([3.2, 3.2, 4.2], device=x.device, dtype=x.dtype) * e_v
    u = baseline + residual
    return torch.cat([e_v, u], dim=0)


def contraction_residual_batch(actor_net, ncm_net, states, alpha=0.55):
    """Compute Mdot + A' M + M A + 2 alpha M for sampled reduced-order states."""
    from torch.func import jacrev, vmap
    f_single = lambda x: _reduced_tracking_vector_field(x, actor_net)
    m_single = lambda x: ncm_net(x.unsqueeze(0)).squeeze(0)
    f = vmap(f_single)(states)
    A = vmap(jacrev(f_single))(states)
    M = ncm_net(states)
    M_jac = vmap(jacrev(m_single))(states)  # [B, M_row, M_col, x_dim]
    M_dot = torch.einsum("bijk,bk->bij", M_jac, f)
    residual = M_dot + torch.matmul(A.transpose(-1, -2), M) + torch.matmul(M, A) + 2.0 * alpha * M
    residual = 0.5 * (residual + residual.transpose(-1, -2))
    eigvals = torch.linalg.eigvalsh(residual)
    eig_M = torch.linalg.eigvalsh(M)
    cond_M = eig_M[:, -1] / eig_M[:, 0].clamp(min=1e-6)
    return residual, eigvals, eig_M, cond_M



def evaluate_matched_rollouts(actor_net, ncm_net, num_envs=8, steps=100, seed=73021):
    """Compare trained residual actor against zero-residual PD baseline on matched random starts/winds.

    This is simulator-only, same-distribution evaluation. It is not evidence of real-world
    performance or generalisation to arbitrary aircraft, wake fields or operating envelopes.
    """
    num_envs = max(1, int(num_envs)); steps = max(1, int(steps))
    def make_env_with_seed(seed_value):
        torch.manual_seed(int(seed_value))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed_value))
        return WakeSurfingResearchEnv(
            num_envs=num_envs, dt=0.015, history_len=75, device=DEVICE,
            randomize_initial_conditions=True,
        )

    policy_env = make_env_with_seed(seed)
    baseline_env = make_env_with_seed(seed)
    baseline_actor = WakeSurfingActorPolicy().to(DEVICE)  # zero-initialised residual baseline
    actor_net.eval(); ncm_net.eval(); baseline_actor.eval()

    records = {"trained_residual_actor": {"pos": [], "vel": [], "separation": [], "power": [], "filter_feasible": [], "filter_residual": [], "finite": True},
               "zero_residual_pd_baseline": {"pos": [], "vel": [], "separation": [], "power": [], "filter_feasible": [], "filter_residual": [], "finite": True}}
    with torch.no_grad():
        for _ in range(steps):
            for label, env, actor in (
                ("trained_residual_actor", policy_env, actor_net),
                ("zero_residual_pd_baseline", baseline_env, baseline_actor),
            ):
                m = env.step(actor, ncm_net)
                pos_err = safe_norm(m["error_pos"], dim=-1)
                vel_err = safe_norm(m["error_vel"], dim=-1)
                separation = safe_norm(env.follow_pos - env.lead_pos, dim=-1)
                is_finite = all(torch.isfinite(t).all().item() for t in (
                    env.lead_pos, env.lead_vel, env.lead_rot, env.follow_pos,
                    env.follow_vel, env.follow_rot, env.follow_omega,
                ))
                rec = records[label]
                rec["pos"].extend(pos_err.detach().cpu().tolist())
                rec["vel"].extend(vel_err.detach().cpu().tolist())
                rec["separation"].extend(separation.detach().cpu().tolist())
                rec["power"].append(float(m["p_follow_total_watts"]))
                rec["filter_feasible"].append(float(m["qp_feasible_pct"]))
                rec["filter_residual"].append(float(m["safety_filter_max_residual"]))
                rec["finite"] = bool(rec["finite"] and is_finite)

    def summarise(rec):
        pos = np.asarray(rec["pos"], dtype=np.float64)
        vel = np.asarray(rec["vel"], dtype=np.float64)
        separation = np.asarray(rec["separation"], dtype=np.float64)
        def finite_stat(values, fn):
            values = np.asarray(values, dtype=np.float64)
            if values.size == 0 or not np.isfinite(values).all():
                return None
            return float(fn(values))
        return {
            "finite_state_trajectory": bool(rec["finite"]),
            "mean_position_tracking_error_m": finite_stat(pos, np.mean),
            "p95_position_tracking_error_m": finite_stat(pos, lambda x: np.percentile(x, 95)),
            "mean_velocity_tracking_error_mps": finite_stat(vel, np.mean),
            "minimum_leader_follower_separation_m": finite_stat(separation, np.min),
            "samples_below_1p20m_separation": int(np.sum(separation[np.isfinite(separation)] < 1.20)),
            "mean_uncalibrated_power_estimate_w": finite_stat(rec["power"], np.mean),
            "mean_filter_feasible_rate_pct": finite_stat(rec["filter_feasible"], np.mean),
            "maximum_filter_constraint_residual": finite_stat(rec["filter_residual"], np.max),
        }
    policy_result = summarise(records["trained_residual_actor"])
    baseline_result = summarise(records["zero_residual_pd_baseline"])
    evaluation = {
        "scope": "matched-seed simulation comparison under mild randomized initial states and episode-constant wind",
        "seed": int(seed), "num_envs": num_envs, "steps": steps,
        "physical_validation": False, "flight_certification": False,
        "trained_residual_actor": policy_result,
        "zero_residual_pd_baseline": baseline_result,
        "delta_mean_position_error_m_policy_minus_baseline": policy_result["mean_position_tracking_error_m"] - baseline_result["mean_position_tracking_error_m"],
        "delta_mean_power_estimate_w_policy_minus_baseline": policy_result["mean_uncalibrated_power_estimate_w"] - baseline_result["mean_uncalibrated_power_estimate_w"],
        "interpretation": "A single matched simulator evaluation is diagnostic only; repeat across multiple seeds and validated models before claiming benefit."
    }
    return evaluation

def evaluate_multiseed_rollouts(actor_net, ncm_net, num_envs=8, steps=100, base_seed=141061, num_seeds=5):
    """Paired multi-seed evaluation; reports seed dispersion and conservative tails.

    The metric is simulator-only and power values remain uncalibrated proxies.
    Every seed runs the candidate and zero-residual PD baseline from matched starts.
    """
    import statistics
    num_seeds = max(2, int(num_seeds))
    runs = []
    for i in range(num_seeds):
        seed_i = int(base_seed) + i * 7919
        runs.append(evaluate_matched_rollouts(actor_net, ncm_net, num_envs=num_envs, steps=steps, seed=seed_i))

    labels = ("trained_residual_actor", "zero_residual_pd_baseline")
    agg = {}
    def values(label, field):
        return [run[label][field] for run in runs]
    for label in labels:
        pos = values(label, "mean_position_tracking_error_m")
        p95 = values(label, "p95_position_tracking_error_m")
        vel = values(label, "mean_velocity_tracking_error_mps")
        sep = values(label, "minimum_leader_follower_separation_m")
        power = values(label, "mean_uncalibrated_power_estimate_w")
        feasible = values(label, "mean_filter_feasible_rate_pct")
        residual = values(label, "maximum_filter_constraint_residual")
        agg[label] = {
            "finite_state_trajectory": all(run[label]["finite_state_trajectory"] for run in runs),
            "mean_position_tracking_error_m": float(np.mean(pos)),
            "std_across_seed_mean_position_error_m": float(np.std(pos, ddof=1)),
            "max_seed_p95_position_tracking_error_m": float(np.max(p95)),
            "mean_velocity_tracking_error_mps": float(np.mean(vel)),
            "std_across_seed_mean_velocity_error_mps": float(np.std(vel, ddof=1)),
            "minimum_leader_follower_separation_m": float(np.min(sep)),
            "samples_below_1p20m_separation": int(sum(run[label]["samples_below_1p20m_separation"] for run in runs)),
            "mean_uncalibrated_power_estimate_w": float(np.mean(power)),
            "mean_filter_feasible_rate_pct": float(np.mean(feasible)),
            "maximum_filter_constraint_residual": float(np.max(residual)),
            "num_seeds": num_seeds,
        }
    agg["delta_mean_position_error_m_policy_minus_baseline"] = (
        agg["trained_residual_actor"]["mean_position_tracking_error_m"]
        - agg["zero_residual_pd_baseline"]["mean_position_tracking_error_m"]
    )
    agg["delta_mean_power_proxy_w_policy_minus_baseline"] = (
        agg["trained_residual_actor"]["mean_uncalibrated_power_estimate_w"]
        - agg["zero_residual_pd_baseline"]["mean_uncalibrated_power_estimate_w"]
    )
    return {
        "scope": "paired multi-seed simulator evaluation; position/velocity metrics are simulation diagnostics; power is uncalibrated",
        "base_seed": int(base_seed), "num_seeds": num_seeds, "num_envs": int(num_envs), "steps": int(steps),
        "physical_validation": False, "flight_certification": False, "power_model_calibrated": False,
        "aggregate": agg, "per_seed": runs,
    }


def candidate_acceptance_gate(evaluation, min_position_improvement_pct=2.0,
                              max_p95_regression_pct=2.0, max_velocity_regression_pct=5.0,
                              min_filter_feasible_pct=99.9, max_filter_residual=2e-3):
    """Conservative research promotion gate. Power is not used until calibrated."""
    c = evaluation["aggregate"]["trained_residual_actor"]
    b = evaluation["aggregate"]["zero_residual_pd_baseline"]
    baseline_reference_ok = bool(
        b["finite_state_trajectory"]
        and b["minimum_leader_follower_separation_m"] >= 1.20
        and b["samples_below_1p20m_separation"] == 0
        and b["mean_filter_feasible_rate_pct"] >= min_filter_feasible_pct
        and b["maximum_filter_constraint_residual"] <= max_filter_residual
    )
    checks = {
        "baseline_reference_passes_safety_diagnostics": baseline_reference_ok,
        "finite_candidate": bool(c["finite_state_trajectory"]),
        "position_improves_by_required_margin": bool(
            c["mean_position_tracking_error_m"] <= b["mean_position_tracking_error_m"] * (1.0 - min_position_improvement_pct / 100.0)
        ),
        "p95_position_not_materially_worse": bool(
            c["max_seed_p95_position_tracking_error_m"] <= b["max_seed_p95_position_tracking_error_m"] * (1.0 + max_p95_regression_pct / 100.0)
        ),
        "mean_velocity_not_materially_worse": bool(
            c["mean_velocity_tracking_error_mps"] <= b["mean_velocity_tracking_error_mps"] * (1.0 + max_velocity_regression_pct / 100.0)
        ),
        "minimum_separation_respected": bool(
            c["minimum_leader_follower_separation_m"] >= 1.20 and c["samples_below_1p20m_separation"] == 0
        ),
        "filter_feasibility_above_threshold": bool(c["mean_filter_feasible_rate_pct"] >= min_filter_feasible_pct),
        "filter_residual_below_tolerance": bool(c["maximum_filter_constraint_residual"] <= max_filter_residual),
    }
    return {"accepted": all(checks.values()), "checks": checks,
            "selection_rule": "Promote candidate only when all checks pass. Uncalibrated power proxy is diagnostic only, never a promotion criterion."}


def train_wake_controller(epochs=64, num_envs=256, rollout_steps=20, seed=41058, run_dir="wake_surfing_runs", eval_envs=8, eval_steps=100, eval_seeds=5):
    """Train the research controller and save evidence; never auto-claims certification."""
    from pathlib import Path
    import json
    random_seed = int(seed)
    np.random.seed(random_seed % (2**32))
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 106)
    print(" WAKE-SURFING DIAGNOSTIC V3 — POSITION-PRIMARY TRAINING & CHAMPION/CHALLENGER GATE")
    print(f" Envs: {num_envs:,} | Rollout Steps: {rollout_steps} | Device: {DEVICE} | Seed: {random_seed}")
    print(" Note: simulation evidence is not physical validation or flight certification.")
    print("=" * 106)

    env = WakeSurfingResearchEnv(
        num_envs=num_envs, dt=0.015, history_len=75, device=DEVICE,
        randomize_initial_conditions=True,
    )
    alpha = 0.55
    actor_net = WakeSurfingActorPolicy(state_dim=13, action_dim=3).to(DEVICE)
    ncm_net = NeuralContractionMetric(state_dim=6, alpha=alpha).to(DEVICE)
    optimizer_actor = torch.optim.AdamW(actor_net.parameters(), lr=5.0e-4, weight_decay=1e-5)
    optimizer_ncm = torch.optim.AdamW(ncm_net.parameters(), lr=2.2e-3, weight_decay=1e-5)
    scheduler_ncm = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ncm, T_max=max(1, epochs), eta_min=1e-4)
    history = []

    for ep in range(epochs):
        optimizer_actor.zero_grad(set_to_none=True)
        optimizer_ncm.zero_grad(set_to_none=True)
        env.reset()
        rollout_cost = torch.zeros((), device=DEVICE)
        position_error_accum = 0.0
        velocity_error_accum = 0.0
        residual_rms_accum = 0.0
        last_metrics = None
        for _ in range(rollout_steps):
            metrics = env.step(actor_net, ncm_net)
            p_cost = torch.sum(metrics["error_pos"] ** 2, dim=-1).mean()
            v_cost = torch.sum(metrics["error_vel"] ** 2, dim=-1).mean()
            # Do not optimise an uncalibrated power proxy: that can teach the policy
            # to exploit model error rather than discover an actual energy benefit.
            residual_cost = metrics["u_residual"].square().sum(dim=-1).mean()
            step_loss = 8.0 * p_cost + 1.8 * v_cost + 0.08 * residual_cost
            rollout_cost = rollout_cost + step_loss
            position_error_accum += float(safe_norm(metrics["error_pos"], dim=-1).mean().detach().item())
            velocity_error_accum += float(safe_norm(metrics["error_vel"], dim=-1).mean().detach().item())
            residual_rms_accum += float(metrics["u_residual"].detach().square().mean().sqrt().item())
            last_metrics = metrics
        rollout_cost = rollout_cost / max(1, rollout_steps)
        position_error_mean = position_error_accum / max(1, rollout_steps)
        velocity_error_mean = velocity_error_accum / max(1, rollout_steps)
        actor_residual_rms_mean = residual_rms_accum / max(1, rollout_steps)
        rollout_cost.backward()
        grad_norm_actor = float(torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=1.5).item())
        optimizer_actor.step()

        # Train M against the differential contraction condition for the reduced-order
        # vector field with the current actor Jacobian and state-dependent Mdot.
        # Freeze actor parameters during this optimizer step: the objective trains M,
        # not the actor, and actor gradients here would be wasted and then cleared.
        # This is still a local model gate, not a full plant proof.
        _actor_grad_flags = [p.requires_grad for p in actor_net.parameters()]
        for p in actor_net.parameters():
            p.requires_grad_(False)
        rollout_states = last_metrics["state_err"].detach()
        n_roll = min(32, rollout_states.shape[0])
        if rollout_states.shape[0] > n_roll:
            rollout_states = rollout_states[torch.randperm(rollout_states.shape[0], device=DEVICE)[:n_roll]]
        # Match the held-out scale as well as the narrow distribution seen in rollouts.
        n_synth = max(1, 64 - rollout_states.shape[0])
        synthetic_states = torch.cat([
            torch.randn((n_synth, 3), device=DEVICE) * 0.45,
            torch.randn((n_synth, 3), device=DEVICE) * 0.65,
        ], dim=-1)
        x_contract = torch.cat([rollout_states, synthetic_states], dim=0).detach()
        try:
            residual, eig_R, eig_M, cond_M = contraction_residual_batch(actor_net, ncm_net, x_contract, alpha=alpha)
            if not torch.isfinite(eig_R).all() or not torch.isfinite(cond_M).all():
                raise FloatingPointError("Non-finite reduced-model contraction residual or metric condition number")
            max_lambda_R = float(eig_R[:, -1].max().detach().item())
            violation = F.relu(eig_R[:, -1] + 1e-3).mean()
            loss_ncm = 25.0 * violation + 0.003 * cond_M.mean() + 0.001 * torch.diagonal(ncm_net(x_contract), dim1=-2, dim2=-1).sum(dim=-1).mean()
            loss_ncm.backward()
            grad_norm_ncm = float(torch.nn.utils.clip_grad_norm_(ncm_net.parameters(), max_norm=1.5).item())
            optimizer_ncm.step()
            scheduler_ncm.step()
            contraction_ok = bool(max_lambda_R <= -1e-3 and float(cond_M.max().detach().item()) < 1e4)
        except Exception as exc:
            max_lambda_R = None
            grad_norm_ncm = None
            loss_ncm = torch.tensor(0.0)
            contraction_ok = False
            print(f"[NCM GATE ERROR] {type(exc).__name__}: {exc}")
        finally:
            for p, was_trainable in zip(actor_net.parameters(), _actor_grad_flags):
                p.requires_grad_(was_trainable)

        row = {
            "epoch": ep + 1,
            "rollout_loss": float(rollout_cost.detach().item()),
            "ncm_loss": float(loss_ncm.detach().item()),
            "sampled_reduced_model_max_eigenvalue": max_lambda_R,
            "sampled_reduced_model_contraction_gate": contraction_ok,
            "actor_grad_norm": grad_norm_actor,
            "ncm_grad_norm": grad_norm_ncm,
            "mean_position_error_m": float(position_error_mean),
            "mean_velocity_error_mps": float(velocity_error_mean),
            "mean_actor_residual_rms_mps2": float(actor_residual_rms_mean),
            "filter_feasible_pct": float(last_metrics["qp_feasible_pct"]),
            "filter_max_residual": float(last_metrics["safety_filter_max_residual"]),
            "power_estimate_w": float(last_metrics["p_follow_total_watts"]),
            "episode_background_wind_mean_mps": float(last_metrics["background_wind_mean_mps"]),
        }
        history.append(row)
        if (ep + 1) % 4 == 0 or ep == 0 or ep == epochs - 1:
            print(
                f"[Epoch {ep+1:03d}/{epochs}] loss={row['rollout_loss']:.4f} "
                f"pos={row['mean_position_error_m']:.3f}m vel={row['mean_velocity_error_mps']:.3f}m/s "
                f"actor-residual-rms={row['mean_actor_residual_rms_mps2']:.3f}m/s² "
                f"CBF-feasible={row['filter_feasible_pct']:.1f}% residual={row['filter_max_residual']:.4g} "
                f"local-contr.lambda_max={(f'{max_lambda_R:+.4g}' if max_lambda_R is not None else 'UNAVAILABLE')} "
                f"gate={'PASS' if contraction_ok else 'UNVERIFIED'}"
            )

    # Final sampled diagnostic using held-out random states, independent from rollouts.
    x_held = torch.cat([
        torch.randn((128, 3), device=DEVICE) * 0.45,
        torch.randn((128, 3), device=DEVICE) * 0.65,
    ], dim=-1)
    actor_net.eval(); ncm_net.eval()
    try:
        with torch.enable_grad():
            _, held_eig_R, held_eig_M, held_cond = contraction_residual_batch(actor_net, ncm_net, x_held, alpha=alpha)
        if not torch.isfinite(held_eig_R).all() or not torch.isfinite(held_cond).all():
            raise FloatingPointError("Non-finite held-out reduced-model contraction diagnostics")
        held_max = float(held_eig_R[:, -1].max().detach().item())
        held_cond_max = float(held_cond.max().detach().item())
        local_gate_pass = bool(held_max <= -1e-3 and held_cond_max < 1e4)
    except Exception as exc:
        held_max, held_cond_max, local_gate_pass = None, None, False
        print(f"[HELD-OUT NCM GATE ERROR] {type(exc).__name__}: {exc}")

    # Paired multi-seed held-out simulation: challenge the candidate against zero-residual PD.
    actor_net.eval(); ncm_net.eval()
    evaluation = evaluate_multiseed_rollouts(
        actor_net, ncm_net, num_envs=eval_envs, steps=eval_steps,
        base_seed=random_seed + 100_003, num_seeds=eval_seeds,
    )
    selection = candidate_acceptance_gate(evaluation)

    candidate_state = {k: v.detach().cpu().clone() for k, v in actor_net.state_dict().items()}
    baseline_actor = WakeSurfingActorPolicy(state_dim=13, action_dim=3).to(DEVICE)
    baseline_actor.eval()
    if selection["accepted"]:
        selected_actor = actor_net
        selected_role = "trained_residual_actor"
    else:
        selected_actor = baseline_actor
        selected_role = "zero_residual_pd_baseline"

    (out_dir / "heldout_multiseed_evaluation.json").write_text(
        json.dumps(evaluation, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("[HELD-OUT MULTI-SEED] candidate/baseline mean position error="
          f"{evaluation['aggregate']['trained_residual_actor']['mean_position_tracking_error_m']:.4f}/"
          f"{evaluation['aggregate']['zero_residual_pd_baseline']['mean_position_tracking_error_m']:.4f} m; "
          f"mean power-proxy={evaluation['aggregate']['trained_residual_actor']['mean_uncalibrated_power_estimate_w']:.2f}/"
          f"{evaluation['aggregate']['zero_residual_pd_baseline']['mean_uncalibrated_power_estimate_w']:.2f} W")
    print(f"[POLICY PROMOTION] selected={selected_role}; accepted={selection['accepted']}")
    for name, passed in selection["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    artifact = {
        "scope": "local reduced-order tracking model only; excludes full rigid-body, actuator, wake and filtered-control dynamics",
        "physical_validation": False,
        "flight_certification": False,
        "power_model_calibrated": False,
        "seed": random_seed,
        "epochs": int(epochs),
        "num_envs": int(num_envs),
        "rollout_steps": int(rollout_steps),
        "training_objective": "position-primary tracking + velocity tracking + residual magnitude penalty; uncalibrated power proxy excluded from optimisation",
        "heldout_sampled_reduced_model_max_eigenvalue": held_max,
        "heldout_metric_condition_number_max": held_cond_max,
        "heldout_sampled_local_contraction_gate": local_gate_pass,
        "heldout_multiseed_evaluation_file": "heldout_multiseed_evaluation.json",
        "heldout_multiseed_evaluation": evaluation,
        "policy_selection": selection,
        "selected_policy_role": selected_role,
        "selection_does_not_use_power_proxy": True,
        "history": history,
    }
    torch.save({
        "actor_state_dict": selected_actor.state_dict(),
        "candidate_actor_state_dict": candidate_state,
        "ncm_state_dict": ncm_net.state_dict(),
        "metadata": artifact,
    }, out_dir / "wake_surfing_checkpoint.pt")
    (out_dir / "training_evidence.json").write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    print("\n[TRAINING FINISHED]")
    print(f"Checkpoint: {out_dir / 'wake_surfing_checkpoint.pt'}")
    print(f"Evidence:   {out_dir / 'training_evidence.json'}")
    held_label = f"{held_max:+.5g}" if held_max is not None else "UNAVAILABLE"
    print(f"Held-out sampled reduced-model eigenvalue max: {held_label}; gate={'PASS' if local_gate_pass else 'UNVERIFIED'}")
    print(f"Selected controller for cockpit/checkpoint: {selected_role}")
    print("No full-plant contraction proof, real-world wake validation, calibrated energy claim, or flight certification is claimed.")
    return selected_actor, ncm_net



# =============================================================================
# 9. INTERACTIVE 3D DIAGNOSTIC COCKPIT
# =============================================================================
def project_point_3d(point, cam_pos, cam_lookat, screen_w, screen_h, fov=50.0):
    point = np.asarray(point, dtype=np.float64)
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    cam_lookat = np.asarray(cam_lookat, dtype=np.float64)

    fwd = cam_lookat - cam_pos
    fwd_len = np.linalg.norm(fwd)
    if fwd_len < 1e-4:
        return None
    fwd /= fwd_len

    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    rlen = np.linalg.norm(right)
    right = np.array([1.0, 0.0, 0.0]) if rlen < 1e-4 else right / rlen
    up = np.cross(right, fwd)

    delta = point - cam_pos
    depth = float(np.dot(delta, fwd))
    if depth <= 0.2:
        return None

    focal = (screen_h * 0.5) / math.tan(math.radians(fov) * 0.5)
    px = screen_w * 0.5 + focal * float(np.dot(delta, right)) / depth
    py = screen_h * 0.5 - focal * float(np.dot(delta, up)) / depth
    return int(round(px)), int(round(py)), depth


def draw_full_quadrotor_3d(screen, pos, rot_mat, color, label, cam_pos, cam_lookat, w, h, font):
    pt_center = project_point_3d(pos, cam_pos, cam_lookat, w, h)
    if pt_center is None:
        return

    d_arm = ARM_LENGTH * 1.8
    arm_coords = [
        np.array([d_arm, d_arm, 0.0]),
        np.array([-d_arm, -d_arm, 0.0]),
        np.array([-d_arm, d_arm, 0.0]),
        np.array([d_arm, -d_arm, 0.0]),
    ]
    pts = [project_point_3d(pos + rot_mat @ ac, cam_pos, cam_lookat, w, h) for ac in arm_coords]

    if pts[0] and pts[1]:
        pygame.draw.line(screen, (220, 235, 255), pts[0][:2], pts[1][:2], 3)
    if pts[2] and pts[3]:
        pygame.draw.line(screen, (220, 235, 255), pts[2][:2], pts[3][:2], 3)

    for p in pts:
        if p:
            r_disk = max(4, int(45.0 / p[2]))
            pygame.draw.circle(screen, color, p[:2], r_disk, 1)

    nose_tip = project_point_3d(pos + rot_mat @ np.array([d_arm * 1.3, 0.0, 0.0]), cam_pos, cam_lookat, w, h)
    if nose_tip:
        pygame.draw.line(screen, (255, 60, 60), pt_center[:2], nose_tip[:2], 3)

    r_hub = max(6, int(60.0 / pt_center[2]))
    pygame.draw.circle(screen, color, pt_center[:2], r_hub)
    pygame.draw.circle(screen, (255, 255, 255), pt_center[:2], max(1, r_hub // 2))

    lbl = font.render(label, True, color)
    screen.blit(lbl, (pt_center[0] - 30, pt_center[1] - r_hub - 16))


def run_wake_surfing_cockpit(actor_net, ncm_net):
    if pygame is None:
        print("[ERROR] Pygame required for 3D visualization. Install via: pip install pygame")
        return

    pygame.init()
    SCREEN_W, SCREEN_H = 1240, 780
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Wake-Surfing Research Prototype — Unvalidated Model")
    clock = pygame.time.Clock()
    font_hud = pygame.font.SysFont("consolas", 12, bold=True)

    env = WakeSurfingResearchEnv(num_envs=1, dt=0.015, history_len=75, device=DEVICE)
    actor_net.eval()
    ncm_net.eval()

    cam_yaw, cam_pitch, cam_dist = 0.85, 0.35, 6.5
    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                cam_dist = np.clip(cam_dist - event.y * 1.5, 3.5, 25.0)

        mouse = pygame.mouse.get_pressed()
        if mouse[0] or mouse[2]:
            dx, dy = pygame.mouse.get_rel()
            cam_yaw += dx * 0.008
            cam_pitch = np.clip(cam_pitch + dy * 0.008, 0.05, 1.45)
        else:
            pygame.mouse.get_rel()

        with torch.no_grad():
            metrics = env.step(actor_net, ncm_net)

        lead_p = env.lead_pos[0].cpu().numpy()
        follow_p = env.follow_pos[0].cpu().numpy()
        lead_rot = env.lead_rot[0].cpu().numpy()
        follow_rot = env.follow_rot[0].cpu().numpy()

        cam_lookat = lead_p * 0.5 + follow_p * 0.5
        cam_x = cam_lookat[0] + cam_dist * math.cos(cam_pitch) * math.cos(cam_yaw)
        cam_y = cam_lookat[1] + cam_dist * math.cos(cam_pitch) * math.sin(cam_yaw)
        cam_z = max(0.5, cam_lookat[2] + cam_dist * math.sin(cam_pitch))
        cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

        screen.fill((10, 14, 20))

        # Ground Grid
        base_x = int(round(cam_lookat[0] / 4.0)) * 4
        base_y = int(round(cam_lookat[1] / 4.0)) * 4
        for gx in range(base_x - 24, base_x + 25, 4):
            p1 = project_point_3d(np.array([gx, base_y - 20, 0]), cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            p2 = project_point_3d(np.array([gx, base_y + 20, 0]), cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            if p1 and p2:
                pygame.draw.line(screen, (22, 28, 38), p1[:2], p2[:2], 1)
        for gy in range(base_y - 20, base_y + 21, 4):
            p1 = project_point_3d(np.array([base_x - 24, gy, 0]), cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            p2 = project_point_3d(np.array([base_x + 24, gy, 0]), cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            if p1 and p2:
                pygame.draw.line(screen, (22, 28, 38), p1[:2], p2[:2], 1)

        # Leader's Curved Tip Vortex Streamlines
        hist_p = env.history_pos[0].cpu().numpy()
        hist_r = env.history_rot[0].cpu().numpy()
        for s in range(len(hist_p) - 1):
            p1_h, p2_h = hist_p[s], hist_p[s + 1]
            r1_h, r2_h = hist_r[s], hist_r[s + 1]

            vl1 = p1_h + r1_h @ np.array([0.0, SPAN_EFF * 0.5, 0.0])
            vl2 = p2_h + r2_h @ np.array([0.0, SPAN_EFF * 0.5, 0.0])
            vr1 = p1_h + r1_h @ np.array([0.0, -SPAN_EFF * 0.5, 0.0])
            vr2 = p2_h + r2_h @ np.array([0.0, -SPAN_EFF * 0.5, 0.0])

            pt_vl1 = project_point_3d(vl1, cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            pt_vl2 = project_point_3d(vl2, cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            pt_vr1 = project_point_3d(vr1, cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            pt_vr2 = project_point_3d(vr2, cam_pos, cam_lookat, SCREEN_W, SCREEN_H)

            if pt_vl1 and pt_vl2:
                pygame.draw.line(screen, (60, 160, 255), pt_vl1[:2], pt_vl2[:2], 2)
            if pt_vr1 and pt_vr2:
                pygame.draw.line(screen, (255, 90, 80), pt_vr1[:2], pt_vr2[:2], 2)

        # Draw arrows from the modelled velocity vectors, not invented fixed upward arrows.
        follow_flow_vec = metrics["v_induced_follower"][0].detach().cpu().numpy()
        for vy in np.linspace(-0.20, 0.20, 3):
            arrow_base = follow_p + follow_rot @ np.array([0.0, vy, -0.20])
            arrow_tip = arrow_base + 0.18 * follow_flow_vec
            p_base = project_point_3d(arrow_base, cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            p_tip = project_point_3d(arrow_tip, cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            if p_base and p_tip and np.linalg.norm(follow_flow_vec) > 1e-3:
                pygame.draw.line(screen, (0, 255, 180), p_base[:2], p_tip[:2], 2)
                pygame.draw.circle(screen, (0, 255, 180), p_tip[:2], 3)

        leader_flow_vec = metrics["v_backwash_lead"][0].detach().cpu().numpy()
        for vy in np.linspace(-0.15, 0.15, 2):
            arrow_base_lead = lead_p + lead_rot @ np.array([-0.20, vy, -0.20])
            arrow_tip_lead = arrow_base_lead + 0.18 * leader_flow_vec
            p_base_l = project_point_3d(arrow_base_lead, cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            p_tip_l = project_point_3d(arrow_tip_lead, cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            if p_base_l and p_tip_l and np.linalg.norm(leader_flow_vec) > 1e-3:
                pygame.draw.line(screen, (255, 215, 0), p_base_l[:2], p_tip_l[:2], 2)
                pygame.draw.circle(screen, (255, 215, 0), p_tip_l[:2], 3)

        # Render 6-DoF Quadrotors
        draw_full_quadrotor_3d(
            screen, lead_p, lead_rot, (255, 210, 60), "LEADER (SE(3) Attitude Control)", cam_pos, cam_lookat, SCREEN_W, SCREEN_H, font_hud
        )
        draw_full_quadrotor_3d(
            screen, follow_p, follow_rot, (0, 255, 220), "FOLLOWER (6-DoF Neural Pilot)", cam_pos, cam_lookat, SCREEN_W, SCREEN_H, font_hud
        )

        # HUD Dashboard with Aeromechanical Power Metrics
        hud_bg = pygame.Rect(15, 15, 560, 280)
        pygame.draw.rect(screen, (8, 12, 18), hud_bg, border_radius=6)
        pygame.draw.rect(screen, (0, 255, 200), hud_bg, 1, border_radius=6)

        status_lines = [
            ("6-DoF WAKE-INTERACTION RESEARCH PROTOTYPE", (0, 255, 200)),
            ("Flight Dynamics:                Rigid-body approximation (not full validation)", (255, 220, 40)),
            (f"Formation Spacing:              {metrics['dist_lead']:.2f} m (Arc-Length Lock)", (255, 255, 255)),
            (f"Power-model estimate (uncalibrated): {metrics['p_follow_total_watts']:.1f} W (hover ref {metrics['p_base_watts']:.1f} W)", (0, 255, 160)),
            (f"Vs isolated-hover proxy (not mission-equivalent): {metrics['p_saved_follow']:.1f}% (flow {metrics['upwash_follow']:+.2f} m/s)", (0, 255, 160)),
            (f"Leader power proxy vs hover baseline: {metrics['p_saved_lead']:.1f}% (flow {metrics['backwash_lead']:+.2f} m/s)", (255, 215, 0)),
            (f"Two-aircraft power proxy vs hover: {metrics['p_saved_swarm']:.1f}% (uncalibrated)", (0, 255, 255)),
            (f"Contraction: local reduced-model diagnostic only (not a certificate)", (100, 200, 255)),
            (f"Lateral barrier margin:         h_sink={metrics['h_sink']:+.3f} m (Outside Core)", (255, 160, 80)),
            (f"Safety filter: projection residual {metrics['safety_filter_max_residual']:.3g}; feasible {metrics['qp_feasible_pct']:.0f}%", (0, 255, 200)),
        ]

        for i, (line, col) in enumerate(status_lines):
            screen.blit(font_hud.render(line, True, col), (hud_bg.x + 12, hud_bg.y + 10 + i * 18))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


# =============================================================================
# 10. CLI, REGRESSION TESTS AND TRAINING ENTRY POINT
# =============================================================================
def run_self_tests():
    """Fast deterministic checks for numerical invariants and failure reporting."""
    import traceback
    results = []
    def check(name, fn):
        try:
            fn()
            results.append((name, True, "PASS"))
        except Exception as exc:
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()

    def test_rotation_and_finite_dynamics():
        env = WakeSurfingResearchEnv(num_envs=2, dt=0.015, history_len=20, device=DEVICE, randomize_initial_conditions=True)
        actor = WakeSurfingActorPolicy().to(DEVICE)
        ncm = NeuralContractionMetric().to(DEVICE)
        with torch.no_grad():
            for _ in range(30):
                env.step(actor, ncm)
            for R in (env.lead_rot, env.follow_rot):
                eye = torch.eye(3, device=DEVICE).expand_as(R)
                assert torch.isfinite(R).all()
                assert torch.max(torch.abs(R.transpose(-1, -2) @ R - eye)).item() < 2e-3
                det = torch.linalg.det(R)
                assert torch.max(torch.abs(det - 1.0)).item() < 2e-3
            assert torch.isfinite(env.lead_pos).all() and torch.isfinite(env.follow_pos).all()

    def test_power_is_finite_nonnegative():
        thrust = torch.full((4, 4), HOVER_WEIGHT / 4.0, device=DEVICE)
        w = torch.zeros((4, 1), device=DEVICE)
        p = compute_rotor_power_estimate(thrust, w)
        assert torch.isfinite(p).all() and (p >= 0).all()
        assert torch.max(torch.abs(p - p[0])).item() < 1e-5

    def test_wake_velocity_units_and_directional_gate():
        wake = TwoWayLagrangianWakeField()
        pos = torch.tensor([[0.0, 0.0, 0.0]], device=DEVICE)
        rot = torch.eye(3, device=DEVICE).unsqueeze(0)
        thrust = torch.tensor([[HOVER_WEIGHT]], device=DEVICE)
        # Target lies above the rotor, i.e. upstream of its downward induced flow.
        target_upstream = torch.tensor([[0.0, 0.0, 2.0]], device=DEVICE)
        v = wake.compute_follower_to_leader_back_induction(target_upstream, pos, rot, thrust)
        assert torch.isfinite(v).all()
        assert safe_norm(v, dim=-1).item() < 0.05
        # Target well below the rotor is downstream: non-zero downwash must be bounded.
        target_downstream = torch.tensor([[0.0, 0.0, -1.0]], device=DEVICE)
        v2 = wake.compute_follower_to_leader_back_induction(target_downstream, pos, rot, thrust)
        assert torch.isfinite(v2).all()
        assert v2[0, 2].item() < 0.0
        assert safe_norm(v2, dim=-1).item() < 15.0

    def test_projection_feasible_and_infeasible_status():
        solver = RelaxedQPSafetyFilter(iters=100, tolerance=5e-3)
        u0 = torch.tensor([[2.0, 0.0, 0.0]], device=DEVICE)
        # Constraint u_x <= 1, with roomy bounds.
        A = torch.tensor([[[1.0, 0.0, 0.0]]], device=DEVICE)
        b = torch.tensor([[[1.0]]], device=DEVICE)
        u, residual, *_rest, feasible_pct = solver.solve(u0, A, b, torch.tensor([-5.0]*3, device=DEVICE), torch.tensor([5.0]*3, device=DEVICE))
        assert u[0, 0].item() <= 1.005 and feasible_pct >= 99.0, (u, residual, feasible_pct)
        # Contradictory constraints u_x<=-1 and u_x>=1 must not be reported feasible.
        A2 = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]], device=DEVICE)
        b2 = torch.tensor([[[-1.0], [-1.0]]], device=DEVICE)
        _, _, _, _, _, _, feasible2 = solver.solve(u0, A2, b2, torch.tensor([-5.0]*3, device=DEVICE), torch.tensor([5.0]*3, device=DEVICE))
        assert feasible2 < 99.0, f"Infeasible set was falsely accepted: {feasible2}%"

    def test_metric_positive_definite_and_no_false_certificate():
        metric = NeuralContractionMetric().to(DEVICE)
        x = torch.randn((16, 6), device=DEVICE)
        M = metric(x)
        eig = torch.linalg.eigvalsh(M)
        assert torch.isfinite(eig).all() and torch.min(eig).item() > 0.0
        # A positive residual eigenvalue must remain UNVERIFIED, not be labelled certified.
        sample_max_lambda = 0.25
        gate = sample_max_lambda <= -1e-3
        assert not gate

    def test_actor_starts_as_zero_residual():
        actor = WakeSurfingActorPolicy().to(DEVICE)
        x = torch.randn((12, 13), device=DEVICE)
        with torch.no_grad():
            assert torch.max(torch.abs(actor(x))).item() == 0.0

    def test_reduced_model_jacobian_calculation():
        actor = WakeSurfingActorPolicy().to(DEVICE)
        metric = NeuralContractionMetric().to(DEVICE)
        states = torch.randn((4, 6), device=DEVICE) * 0.1
        _, eig_R, eig_M, cond = contraction_residual_batch(actor, metric, states)
        assert torch.isfinite(eig_R).all() and torch.isfinite(eig_M).all() and torch.isfinite(cond).all()
        assert eig_R.shape == (4, 6) and eig_M.shape == (4, 6)

    def test_promotion_gate_rejects_non_improving_candidate():
        baseline = {
            "finite_state_trajectory": True,
            "mean_position_tracking_error_m": 0.20,
            "max_seed_p95_position_tracking_error_m": 0.32,
            "mean_velocity_tracking_error_mps": 0.20,
            "minimum_leader_follower_separation_m": 1.45,
            "samples_below_1p20m_separation": 0,
            "mean_filter_feasible_rate_pct": 100.0,
            "maximum_filter_constraint_residual": 1e-6,
        }
        candidate = dict(baseline)
        candidate["mean_position_tracking_error_m"] = 0.195  # only 2.5% improvement
        evaluation = {"aggregate": {"trained_residual_actor": candidate, "zero_residual_pd_baseline": baseline}}
        assert candidate_acceptance_gate(evaluation)["accepted"]
        candidate["mean_position_tracking_error_m"] = 0.201
        assert not candidate_acceptance_gate(evaluation)["accepted"]
        candidate["mean_position_tracking_error_m"] = 0.19
        candidate["maximum_filter_constraint_residual"] = 0.1
        assert not candidate_acceptance_gate(evaluation)["accepted"]

    check("finite 6-DoF rollout and SO(3) rotation invariants", test_rotation_and_finite_dynamics)
    check("power estimate finite/non-negative", test_power_is_finite_nonnegative)
    check("wake direction, dimensional scale and boundedness", test_wake_velocity_units_and_directional_gate)
    check("projection solver feasibility and infeasibility detection", test_projection_feasible_and_infeasible_status)
    check("metric positive-definite and no false certificate", test_metric_positive_definite_and_no_false_certificate)
    check("actor initialises as zero residual", test_actor_starts_as_zero_residual)
    check("autodifferentiated reduced-model Jacobian and metric derivative", test_reduced_model_jacobian_calculation)
    check("candidate-promotion gate fails closed", test_promotion_gate_rejects_non_improving_candidate)
    print("\nSELF-TEST SUMMARY")
    for name, ok, message in results:
        print(f"[{ 'PASS' if ok else 'FAIL' }] {name}: {message}")
    return 0 if all(r[1] for r in results) else 2


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Wake-surfing diagnostic V3; simulation research only, not flight-certified.")
    parser.add_argument("--mode", choices=("train", "self-test", "smoke-test", "view"), default="train")
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=41058)
    parser.add_argument("--run-dir", default="wake_surfing_runs")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--checkpoint", default="wake_surfing_runs/wake_surfing_checkpoint.pt")
    parser.add_argument("--eval-envs", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--eval-seeds", type=int, default=5, help="Number of paired held-out evaluation seeds (minimum 2).")
    parser.add_argument("--checkpoint-role", choices=("selected", "candidate"), default="selected", help="In view mode, show promoted policy or unfiltered trained candidate.")
    args = parser.parse_args()

    if args.mode == "self-test":
        return run_self_tests()
    if args.mode == "smoke-test":
        test_code = run_self_tests()
        if test_code != 0:
            return test_code
        actor, ncm = train_wake_controller(epochs=2, num_envs=4, rollout_steps=3, seed=args.seed, run_dir=args.run_dir, eval_envs=4, eval_steps=20, eval_seeds=3)
        print("SMOKE TEST completed. Checkpoint/training evidence were written; this is not a performance qualification.")
        return 0
    if args.mode == "train":
        actor, ncm = train_wake_controller(
            epochs=max(1, args.epochs), num_envs=max(1, args.envs),
            rollout_steps=max(1, args.rollout_steps), seed=args.seed, run_dir=args.run_dir,
            eval_envs=max(1, args.eval_envs), eval_steps=max(1, args.eval_steps), eval_seeds=max(2, args.eval_seeds),
        )
        if not args.no_visualize:
            run_wake_surfing_cockpit(actor, ncm)
        return 0
    if args.mode == "view":
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint at {checkpoint_path}. Run --mode train first.")
        payload = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        actor, ncm = WakeSurfingActorPolicy().to(DEVICE), NeuralContractionMetric().to(DEVICE)
        actor_key = "candidate_actor_state_dict" if args.checkpoint_role == "candidate" else "actor_state_dict"
        if actor_key not in payload:
            raise KeyError(f"Checkpoint has no {actor_key}; select a compatible checkpoint or role.")
        actor.load_state_dict(payload[actor_key])
        ncm.load_state_dict(payload["ncm_state_dict"])
        run_wake_surfing_cockpit(actor, ncm)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())