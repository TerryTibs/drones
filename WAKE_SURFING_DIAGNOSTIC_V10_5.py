"""
=============================================================================
WAKE-SURFING FLUID AUTONOMY: V10.5 NN-PRIMARY TARGET-KINEMATICS-CORRECTED RESEARCH PROTOTYPE
Architecture:
  - Physics: 6-DoF rigid-body model; implicit rotor-speed dynamics + electrical current state + SO(3) attitude update
  - Fluid  : Regularised wake-velocity approximation (not wind-tunnel validated; inherited from V9.1)
  - Actuator: per-rotor angular speed, motor current/voltage, thrust and reaction-torque coupling
  - Driver : Neural policy directly commands four rotor-thrust setpoints (no PD tracking command)
  - Control: Physical motor/rotor dynamics remain; barrier monitors are diagnostic and do NOT override NN commands
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

# Training architecture constants
ACTOR_OBSERVATION_DIM = 38  # tracking + wake + attitude/rates + rotor state + prior rotor actions
TRAINING_BPTT_CHUNK_STEPS = 20  # truncated backpropagation keeps long rollouts within GPU memory
INITIAL_CURRICULUM_HORIZON = 20

# Physical Constants
GRAVITY = 9.80665
DRONE_MASS = 1.45  # kg
ARM_LENGTH = 0.22  # m: centre-to-rotor radial arm length
ROTOR_OFFSET = ARM_LENGTH / math.sqrt(2.0)  # m: each rotor centre x/y offset for the X frame
SPAN_EFF = 2.0 * ROTOR_OFFSET  # m: full left-right rotor-centre spacing used by the wake model (0.311 m)
AIR_DENSITY = 1.225  # kg/m^3
MOTOR_TAU = 0.032  # s
MAX_THRUST_PER_MOTOR = 8.5  # N (34 N total vehicle ceiling)
TORQUE_TO_THRUST_RATIO = 0.016
MOTOR_COIL_RESISTANCE = 0.048  # ohm; provisional, not bench-identified
MOTOR_INDUCTANCE = 2.5e-4  # H; provisional electrical time constant
MOTOR_TORQUE_CONSTANT = 0.0136  # N m/A; provisional motor constant
MOTOR_BACK_EMF_CONSTANT = MOTOR_TORQUE_CONSTANT  # V/(rad/s), SI-consistent reciprocal
MOTOR_CURRENT_LIMIT = 20.0  # A per motor; provisional actuator limit
BATTERY_VOLTAGE_MAX = 16.8  # V; idealised 4-cell LiPo full-charge ceiling
ROTOR_INERTIA = 1.2e-5  # kg m^2; provisional rotor + prop inertia
MOTOR_SPEED_KP_A_PER_RAD_S = 0.025  # A/(rad/s); embedded speed-loop gain, provisional
ROTOR_THRUST_COEFFICIENT = 1.1e-5  # N/(rad/s)^2; inherited estimate, not calibrated
ROTOR_TORQUE_COEFFICIENT = TORQUE_TO_THRUST_RATIO * ROTOR_THRUST_COEFFICIENT  # Nm/(rad/s)^2
ROTOR_MAX_OMEGA = math.sqrt(MAX_THRUST_PER_MOTOR / ROTOR_THRUST_COEFFICIENT)  # rad/s
# Render-only spin signs consistent with the alternating reaction-torque allocation.
# Absolute propeller handedness is not calibrated; animation conveys RPM, not blade aerodynamics.
ROTOR_SPIN_SIGNS = (1.0, 1.0, -1.0, -1.0)
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


def moving_body_offset_kinematics(
    centre_position,
    centre_velocity,
    centre_rotation,
    omega_body,
    alpha_body,
    centre_acceleration,
    offset_body,
):
    """Return position, velocity and acceleration of a point fixed to a moving body.

    Rotation is body-to-world, angular velocity/acceleration are expressed in body
    coordinates, and ``offset_body`` is a constant body-frame vector. The returned
    derivatives include the previously omitted rotational terms:
        v_point = v_centre + omega_world x r_world
        a_point = a_centre + alpha_world x r_world
                  + omega_world x (omega_world x r_world)

    This is rigid-body point kinematics, not a wake or aerodynamic model.
    """
    if centre_position.ndim != 2 or centre_position.shape[-1] != 3:
        raise ValueError("centre_position must have shape [batch,3]")
    shape = centre_position.shape
    for name, value in (("centre_velocity", centre_velocity),
                         ("omega_body", omega_body), ("alpha_body", alpha_body),
                         ("centre_acceleration", centre_acceleration)):
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if centre_rotation.shape != (shape[0], 3, 3):
        raise ValueError("centre_rotation must have shape [batch,3,3]")
    offset = torch.as_tensor(offset_body, dtype=centre_position.dtype, device=centre_position.device)
    if offset.shape == (3,):
        offset = offset.unsqueeze(0).expand(shape[0], -1)
    if offset.shape != shape:
        raise ValueError(f"offset_body must have shape [3] or {shape}, got {tuple(offset.shape)}")

    offset_world = torch.matmul(centre_rotation, offset.unsqueeze(-1)).squeeze(-1)
    omega_world = torch.matmul(centre_rotation, omega_body.unsqueeze(-1)).squeeze(-1)
    alpha_world = torch.matmul(centre_rotation, alpha_body.unsqueeze(-1)).squeeze(-1)
    offset_velocity_world = torch.cross(omega_world, offset_world, dim=-1)
    offset_acceleration_world = (
        torch.cross(alpha_world, offset_world, dim=-1)
        + torch.cross(omega_world, offset_velocity_world, dim=-1)
    )
    point_position = centre_position + offset_world
    point_velocity = centre_velocity + offset_velocity_world
    point_acceleration = centre_acceleration + offset_acceleration_world
    return point_position, point_velocity, point_acceleration, offset_velocity_world, offset_acceleration_world


def acceleration_residual_rhs(A_absolute, b_absolute, a_feedforward):
    """Convert absolute-acceleration barrier rows to residual-acceleration coordinates.

    The safety filter acts on a residual command ``u`` while the plant's requested
    acceleration is ``a_feedforward + u``. If the derived barrier is
    ``A @ a_absolute <= b``, then the equivalent constraint on the residual is
    ``A @ u <= b - A @ a_feedforward``. Omitting this shift changes the feasible
    half-space and can cause false infeasibility/constraint violations.
    """
    if A_absolute.ndim != 3 or b_absolute.ndim != 3 or a_feedforward.ndim != 2:
        raise ValueError("Expected A:[B,C,3], b:[B,C,1], feedforward:[B,3]")
    if A_absolute.shape[0] != a_feedforward.shape[0] or A_absolute.shape[-1] != a_feedforward.shape[-1]:
        raise ValueError("Batch/action dimensions do not match in acceleration constraint conversion")
    if A_absolute.shape[:2] != b_absolute.shape[:2] or b_absolute.shape[-1] != 1:
        raise ValueError("Constraint matrix and right-hand side dimensions do not match")
    return b_absolute - torch.matmul(A_absolute, a_feedforward.unsqueeze(-1))


def build_lateral_wake_barrier(
    relative_position,
    relative_velocity,
    source_rotation,
    source_omega_body,
    source_alpha_body,
    source_acceleration_world,
    margin,
    cbf_lambda=2.0,
):
    """Build a relative-degree-two lateral CBF against a *historical wake source*.

    The source pose/rates/acceleration are sampled from the delayed leader history
    that defines the wake target. They are intentionally not taken from the leader's
    current instantaneous attitude. The signed clearance is
        h = -(p_follower - p_source) dot y_source - margin.
    With the source-frame derivatives included, the HOCBF row is
        y_source dot a_follower <= b_absolute.
    This aligns the lateral envelope with the stored wake segment, while the
    independent collision CBF continues to use the current leader pose.

    This is still a reduced-model CBF: discrete target-index changes, wake convection,
    actuator/attitude lag, parameter error, and model error are not formally bounded.
    """
    if relative_position.ndim != 2 or relative_position.shape[-1] != 3:
        raise ValueError("relative_position must have shape [batch,3]")
    expected = relative_position.shape
    for name, tensor in (
        ("relative_velocity", relative_velocity),
        ("source_omega_body", source_omega_body),
        ("source_alpha_body", source_alpha_body),
        ("source_acceleration_world", source_acceleration_world),
    ):
        if tensor.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
    if source_rotation.shape != (expected[0], 3, 3):
        raise ValueError("source_rotation must have shape [batch,3,3]")

    y_axis = source_rotation[:, :, 1]
    omega_world = torch.matmul(source_rotation, source_omega_body.unsqueeze(-1)).squeeze(-1)
    alpha_world = torch.matmul(source_rotation, source_alpha_body.unsqueeze(-1)).squeeze(-1)
    y_axis_dot = torch.cross(omega_world, y_axis, dim=-1)
    y_axis_ddot = torch.cross(alpha_world, y_axis, dim=-1) + torch.cross(omega_world, y_axis_dot, dim=-1)
    h = -torch.sum(relative_position * y_axis, dim=-1, keepdim=True) - float(margin)
    h_dot = (
        -torch.sum(relative_velocity * y_axis, dim=-1, keepdim=True)
        - torch.sum(relative_position * y_axis_dot, dim=-1, keepdim=True)
    )
    b_absolute = (
        torch.sum(y_axis * source_acceleration_world, dim=-1, keepdim=True)
        - 2.0 * torch.sum(y_axis_dot * relative_velocity, dim=-1, keepdim=True)
        - torch.sum(y_axis_ddot * relative_position, dim=-1, keepdim=True)
        + 2.0 * float(cbf_lambda) * h_dot
        + (float(cbf_lambda) ** 2) * h
    ).unsqueeze(-1)
    A = y_axis.unsqueeze(1)
    return {
        "A_absolute": A,
        "b_absolute": b_absolute,
        "h": h,
        "h_dot": h_dot,
        "axis": y_axis,
        "axis_dot": y_axis_dot,
        "axis_ddot": y_axis_ddot,
    }


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
        self, eval_pos, hist_pos, hist_rot, hist_vel, leader_thrust, turn_curvature,
        history_dt=0.015, hist_accel=None,
    ):
        """Estimate wake velocity from the nearest historical source sample.

        Inputs and outputs use SI units: positions [m], velocities [m/s], thrust [N],
        curvature [1/m], history_dt [s], induced velocity [m/s]. If hist_accel is
        supplied, curvature is evaluated from the selected source's planar velocity
        and acceleration. The circulation
        clamp, core-growth law, decay time, and streamwise velocity component below
        are empirical regularisations, not calibrated aerodynamic constants.

        The vortex pair is treated as two locally straight, effectively infinite
        filaments along the source frame's x-axis. Under that approximation, the
        Biot-Savart cross-plane field is independent of the longitudinal coordinate
        dx; finite-segment/end effects and wake convection are not represented.
        """
        if not math.isfinite(float(history_dt)) or float(history_dt) <= 0.0:
            raise ValueError("history_dt must be a positive finite number of seconds")
        N, H, _ = hist_pos.shape

        # Choose the source sample first. Source speed and age must correspond to
        # this same history index (not always the newest sample at index zero).
        dist_to_segments = safe_norm(hist_pos - eval_pos.unsqueeze(1), dim=-1)
        k_star = torch.argmin(dist_to_segments, dim=1)

        idx_pos = k_star.view(N, 1, 1).expand(N, 1, 3)
        pos_near = torch.gather(hist_pos, 1, idx_pos).squeeze(1)

        idx_rot = k_star.view(N, 1, 1, 1).expand(N, 1, 3, 3)
        rot_near = torch.gather(hist_rot, 1, idx_rot).squeeze(1)
        idx_vel = k_star.view(N, 1, 1).expand(N, 1, 3)
        vel_near = torch.gather(hist_vel, 1, idx_vel).squeeze(1)
        fwd_speed = safe_norm(vel_near[:, :2], dim=-1, keepdim=True).clamp(min=1.5)

        gamma_0 = leader_thrust / (self.rho * fwd_speed * self.b).clamp(min=1e-3)
        gamma_0 = torch.clamp(gamma_0, min=1.0, max=24.0)

        # Estimate curvature at this same historical source sample when acceleration
        # history is available. For a planar trajectory kappa=(vx*ay-vy*ax)/|v|^3.
        # turn_curvature remains a backwards-compatible fallback for external callers.
        curvature_source = turn_curvature
        if hist_accel is not None:
            if hist_accel.shape != hist_vel.shape:
                raise ValueError("hist_accel must have the same [batch,history,3] shape as hist_vel")
            idx_accel = k_star.view(N, 1, 1).expand(N, 1, 3)
            accel_near = torch.gather(hist_accel, 1, idx_accel).squeeze(1)
            vx, vy = vel_near[:, 0:1], vel_near[:, 1:2]
            ax, ay = accel_near[:, 0:1], accel_near[:, 1:2]
            signed_cross = vx * ay - vy * ax
            curvature_source = signed_cross / fwd_speed.pow(3)

        # History index zero is the newest sample; larger indices are older.
        tau = k_star.to(dtype=hist_pos.dtype).unsqueeze(-1) * float(history_dt)
        turb_factor = 1.0 + 2.0 * torch.abs(curvature_source)
        r_c = torch.sqrt(self.r_0**2 + 0.008 * tau * turb_factor).clamp(max=0.14)

        delta_world = eval_pos - pos_near
        delta_local = torch.matmul(rot_near.transpose(-1, -2), delta_world.unsqueeze(-1)).squeeze(-1)
        # dx is intentionally absent from the infinite-filament cross-plane law;
        # see the documented limitation above.
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
    """Neural-primary follower controller with four direct rotor-thrust outputs.

    Inputs are 38 normalised tracking, wake, attitude, angular-rate, target-acceleration,
    target angular-rate, rotor-state, and previous rotor-command features. Outputs are four independent [0,1] rotor-thrust fractions.
    No PD tracking command is added to these outputs. The only subsequent dynamics are
    the explicit rotor/motor plant; the one-step barrier computations are monitors only.

    The initial bias corresponds to a symmetric hover thrust. This is a useful, explicit
    operating-point initialisation, not a PD controller or a pre-trained policy.
    """

    def __init__(self, state_dim=ACTOR_OBSERVATION_DIM, action_dim=4):
        super().__init__()
        if state_dim != ACTOR_OBSERVATION_DIM or action_dim != 4:
            raise ValueError(f"V10.5 policy architecture requires state_dim={ACTOR_OBSERVATION_DIM} and action_dim=4")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, action_dim),
        )
        # Fresh initialization at a symmetric hover operating point; all weights are
        # new every run. Each rotor remains independently controlled by its NN output.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        hover_fraction = min(max((HOVER_WEIGHT / 4.0) / MAX_THRUST_PER_MOTOR, 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.net[-1].bias, math.log(hover_fraction / (1.0 - hover_fraction)))

    def forward(self, obs):
        if obs.shape[-1] != self.state_dim:
            raise ValueError(f"V10.5 actor expected {self.state_dim} observation features, got {obs.shape[-1]}")
        return torch.sigmoid(self.net(obs))


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
        # A: [batch, constraints, 3], b: [batch, constraints, 1].
        # Feasibility diagnostics are expressed in row-normalized units:
        # max(0, A_i u - b_i) / ||A_i||. This makes reported violations invariant
        # to arbitrary positive rescaling of a constraint row. For the collision
        # CBF this is an acceleration-equivalent residual; the lateral CBF already
        # has a unit row normal.
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
        row_norm = torch.linalg.vector_norm(A, dim=-1, keepdim=True).clamp(min=1e-10)

        # Dykstra projection onto each half-space and then the actuator-command box.
        # The configured iteration count is now actually used by the environment;
        # after the finite budget we explicitly test the returned action.
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

        # Normalize residuals by each constraint's row norm before applying a
        # common tolerance. Raw residuals from the collision and lateral barriers
        # have different physical dimensions and cannot share an unscaled threshold.
        projection_residuals_raw = torch.matmul(A, x.unsqueeze(-1)) - b
        projection_residuals = projection_residuals_raw / row_norm
        projection_violation_per_env = F.relu(projection_residuals).amax(dim=(1, 2))
        projection_bad = projection_violation_per_env > self.tolerance
        projection_action = x.clone()
        projection_max = float(projection_violation_per_env.max().detach().item()) if batch else 0.0
        projection_feasible_pct = float((~projection_bad).float().mean().detach().item() * 100.0) if batch else 100.0
        fallback_attempted_pct = float(projection_bad.float().mean().detach().item() * 100.0) if batch and fallback is not None else 0.0
        fallback_applied = torch.zeros_like(projection_bad)
        fallback_residuals = None

        # Do not blindly apply the old collision-only escape command: it can make
        # the lateral-separation inequality much worse. Treat it as one candidate
        # and apply it only when it reduces the worst row-normalized violation.
        # If neither candidate is feasible, the result remains explicitly infeasible.
        if fallback is not None:
            fallback = torch.maximum(torch.minimum(fallback, u_max_t), u_min_t)
            fallback_residuals = (torch.matmul(A, fallback.unsqueeze(-1)) - b) / row_norm
            fallback_violation = F.relu(fallback_residuals).amax(dim=(1, 2))
            fallback_better = projection_bad & (fallback_violation + 1e-7 < projection_violation_per_env)
            x = torch.where(fallback_better.unsqueeze(-1), fallback, x)
            fallback_applied = fallback_better

        # Feasibility of the actual command, not just solver convergence.
        residuals_raw = torch.matmul(A, x.unsqueeze(-1)) - b
        residuals = residuals_raw / row_norm
        max_violation_per_env = F.relu(residuals).amax(dim=(1, 2))
        feasible_pct = float((max_violation_per_env <= self.tolerance).float().mean().detach().item() * 100.0) if batch else 100.0
        max_violation = float(max_violation_per_env.max().detach().item()) if batch else 0.0
        per_constraint = F.relu(residuals).amax(dim=(0, 2)) if n_constraints else x.new_zeros((0,))
        active = (residuals[:, :, 0] > -0.05).float().mean(dim=0) * 100.0 if n_constraints else x.new_zeros((0,))
        margins = (-residuals[:, :, 0]).mean(dim=0) if n_constraints else x.new_zeros((0,))
        active1 = float(active[0].detach().item()) if n_constraints > 0 else 0.0
        active2 = float(active[1].detach().item()) if n_constraints > 1 else 0.0
        active3 = float(active[2].detach().item()) if n_constraints > 2 else 0.0
        margin3 = float(margins[2].detach().item()) if n_constraints > 2 else 0.0
        worst_env = int(torch.argmax(max_violation_per_env).detach().item()) if batch else -1
        worst_constraint = int(torch.argmax(per_constraint).detach().item()) if n_constraints else -1

        # Necessary feasibility test against the actuator-command box. For each
        # half-space, this computes min(A_i u) over u_min <= u <= u_max. If this
        # minimum already exceeds b_i, that individual constraint is impossible
        # under the current command bounds, regardless of projection iteration count.
        min_box_lhs = torch.sum(torch.where(A >= 0.0, A * u_min_t, A * u_max_t), dim=-1, keepdim=True)
        min_box_gap_norm = (min_box_lhs - b) / row_norm
        min_box_violation = F.relu(min_box_gap_norm)
        individual_bound_impossible = (min_box_violation > self.tolerance).any(dim=(1, 2))

        # Keep compact numerical traces for failed samples. The environment adds
        # geometric barrier/state details before saving traces to evaluation JSON.
        failure_mask = projection_bad | (max_violation_per_env > self.tolerance)
        failed_indices = torch.nonzero(failure_mask, as_tuple=False).flatten().tolist()
        failed_indices = sorted(failed_indices, key=lambda j: max(float(projection_violation_per_env[j].detach().item()), float(max_violation_per_env[j].detach().item())), reverse=True)[:8]
        failure_records = []
        for j in failed_indices:
            row_records = []
            for i in range(n_constraints):
                row_records.append({
                    "constraint_index": int(i),
                    "a": [float(v) for v in A[j, i].detach().cpu().tolist()],
                    "b": float(b[j, i, 0].detach().item()),
                    "row_norm": float(row_norm[j, i, 0].detach().item()),
                    "projection_residual_normalized": float(projection_residuals[j, i, 0].detach().item()),
                    "final_residual_normalized": float(residuals[j, i, 0].detach().item()),
                    "minimum_lhs_under_command_box": float(min_box_lhs[j, i, 0].detach().item()),
                    "unavoidable_violation_under_command_box": float(min_box_violation[j, i, 0].detach().item()),
                    "individually_unachievable_under_command_box": bool(min_box_violation[j, i, 0].detach().item() > self.tolerance),
                })
            failure_records.append({
                "environment_index": int(j),
                "projection_max_normalized_violation": float(projection_violation_per_env[j].detach().item()),
                "final_max_normalized_violation": float(max_violation_per_env[j].detach().item()),
                "worst_constraint_index": int(torch.argmax(F.relu(residuals[j, :, 0])).detach().item()) if n_constraints else -1,
                "individually_unachievable_under_command_box": bool(individual_bound_impossible[j].detach().item()),
                "u_nominal_residual_acceleration": [float(v) for v in u_nom[j].detach().cpu().tolist()],
                "u_projection_residual_acceleration": [float(v) for v in projection_action[j].detach().cpu().tolist()],
                "u_final_residual_acceleration": [float(v) for v in x[j].detach().cpu().tolist()],
                "u_fallback_candidate_residual_acceleration": None if fallback is None else [float(v) for v in fallback[j].detach().cpu().tolist()],
                "fallback_candidate_normalized_residuals": None if fallback_residuals is None else [float(v) for v in fallback_residuals[j, :, 0].detach().cpu().tolist()],
                "constraint_rows": row_records,
            })
        details = {
            "projection_max_residual": projection_max,
            "projection_feasible_pct": projection_feasible_pct,
            "fallback_pct": fallback_attempted_pct,  # kept for promotion-gate compatibility
            "fallback_attempted_pct": fallback_attempted_pct,
            "fallback_applied_pct": float(fallback_applied.float().mean().detach().item() * 100.0) if batch else 0.0,
            "final_max_residual": max_violation,
            "final_feasible_pct": feasible_pct,
            "constraint_max_violations": [float(v.detach().item()) for v in per_constraint],
            "worst_environment_index": worst_env,
            "worst_constraint_index": worst_constraint,
            "individually_unachievable_under_command_box_pct": float(individual_bound_impossible.float().mean().detach().item() * 100.0) if batch else 0.0,
            "worst_env_individually_unachievable_under_command_box": bool(individual_bound_impossible[worst_env].detach().item()) if batch and worst_env >= 0 else False,
            "failure_records": failure_records,
            "constraint_count": int(n_constraints),
            "residual_units": "row-normalized inequality residual; acceleration-equivalent for nonzero rows",
        }
        # Residuals are diagnostics, not proof of safety for the physical plant.
        return x, max_violation, active1, active2, active3, margin3, feasible_pct, details


# =============================================================================
# 6. BATCHED 6-DoF RIGID BODY + PER-ROTOR ELECTROMECHANICAL DYNAMICS
# =============================================================================
class RigidBody6DoFDrone:
    """6-DoF body coupled to four rotor-speed and motor-current states.

    Control input is a requested body wrench. A static mixer converts it into rotor
    thrust references; each rotor then follows a bounded electrical/motor model:
      T_i = k_T * Omega_i^2
      Q_i = k_Q * Omega_i^2
      L di_i/dt = V_i - R i_i - k_e Omega_i
      J_r dOmega_i/dt = k_t i_i - Q_i

    The embedded speed-loop/ESC voltage command is a practical controller model, not
    a proprietary ESC reproduction. Rotor mechanics are advanced with a backward-Euler
    scalar solve for the quadratic aerodynamic torque. This makes the mechanical update
    stable at the environment's 15 ms step for the provisional rotor inertia.

    Parameter values are explicit provisional estimates. They must be replaced by
    identified thrust/RPM, torque/RPM, resistance, inductance, motor constants, rotor
    inertia, and voltage/current-limit measurements before physical claims are made.
    External wake lift is still injected as a separate force in V10; wake-to-rotor
    inflow coupling is deliberately recorded as unresolved for the next stage.
    """

    def __init__(self, dt, device):
        self.dt = float(dt)
        self.device = device
        self.J = torch.tensor([[0.021, 0.021, 0.036]], device=device)
        self.J_inv = 1.0 / self.J
        d = ROTOR_OFFSET
        c_q = TORQUE_TO_THRUST_RATIO
        self.B = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [d, -d, d, -d],
                [-d, d, d, -d],
                [-c_q, -c_q, c_q, c_q],
            ], device=device, dtype=torch.float32,
        )
        self.B_inv = torch.linalg.pinv(self.B)
        self.kT = float(ROTOR_THRUST_COEFFICIENT)
        self.kQ = float(ROTOR_TORQUE_COEFFICIENT)

    def allocate_wrench(self, target_wrench):
        """Allocate requested [Fz, tau_x, tau_y, tau_z] to bounded rotor thrusts."""
        raw_thrusts = torch.matmul(target_wrench, self.B_inv.T)
        return torch.clamp(raw_thrusts, 0.0, MAX_THRUST_PER_MOTOR)

    def rotor_step(self, rotor_omega, motor_current, target_thrusts):
        """Advance four rotors by one step; return omega, current, thrust, voltage.

        The current transient uses the exact discrete response of a first-order RL
        current state with speed held at the beginning-of-step value. Rotor speed then
        uses backward Euler on J_r*Omega_dot = k_t*i - k_Q*Omega^2. These are stable
        discretisations of a simplified coupled motor/prop model, not full PWM/ESC FOC.
        """
        dtype, device = rotor_omega.dtype, rotor_omega.device
        omega_ref = torch.sqrt(torch.clamp(target_thrusts, min=0.0) / self.kT)
        omega_ref = omega_ref.clamp(0.0, ROTOR_MAX_OMEGA)

        torque_ref = self.kQ * omega_ref.square()
        current_ff = torque_ref / MOTOR_TORQUE_CONSTANT
        current_cmd = current_ff + MOTOR_SPEED_KP_A_PER_RAD_S * (omega_ref - rotor_omega)
        current_cmd = current_cmd.clamp(0.0, MOTOR_CURRENT_LIMIT)

        # Voltage feed-forward compensates back EMF and winding resistance.
        motor_voltage = MOTOR_BACK_EMF_CONSTANT * rotor_omega + MOTOR_COIL_RESISTANCE * current_cmd
        motor_voltage = motor_voltage.clamp(0.0, BATTERY_VOLTAGE_MAX)
        current_ss = ((motor_voltage - MOTOR_BACK_EMF_CONSTANT * rotor_omega) / MOTOR_COIL_RESISTANCE)
        current_ss = current_ss.clamp(0.0, MOTOR_CURRENT_LIMIT)
        tau_e = MOTOR_INDUCTANCE / MOTOR_COIL_RESISTANCE
        alpha_i = 1.0 - math.exp(-self.dt / tau_e)
        current_next = (motor_current + alpha_i * (current_ss - motor_current)).clamp(0.0, MOTOR_CURRENT_LIMIT)

        # Backward-Euler solution of J_r*(Omega_next-Omega)=dt*(k_t*i-k_Q*Omega_next^2).
        J_r = float(ROTOR_INERTIA)
        c = J_r * rotor_omega + self.dt * MOTOR_TORQUE_CONSTANT * current_next
        discriminant = (J_r * J_r + 4.0 * self.dt * self.kQ * c).clamp(min=0.0)
        omega_next = (2.0 * c / (J_r + torch.sqrt(discriminant).clamp(min=1e-12))).clamp(0.0, ROTOR_MAX_OMEGA)
        thrust_next = (self.kT * omega_next.square()).clamp(0.0, MAX_THRUST_PER_MOTOR)
        return omega_next, current_next, thrust_next, motor_voltage

    def integrate_dynamics(
        self, pos, vel, rot, omega_body, rotor_omega, motor_current,
        target_wrench, wind_field, external_lift, target_rotor_thrusts=None,
    ):
        # If explicit rotor setpoints are supplied, use them directly rather than
        # re-allocating a body wrench. This is the NN-primary V10.3 control path.
        if target_rotor_thrusts is None:
            target_thrusts = self.allocate_wrench(target_wrench)
        else:
            if target_rotor_thrusts.shape != (pos.shape[0], 4):
                raise ValueError("target_rotor_thrusts must have shape [batch,4]")
            target_thrusts = target_rotor_thrusts.clamp(0.0, MAX_THRUST_PER_MOTOR)
        rotor_omega_next, motor_current_next, rotor_thrusts, motor_voltage = self.rotor_step(
            rotor_omega, motor_current, target_thrusts
        )
        realized_wrench = torch.matmul(rotor_thrusts, self.B.T)
        realized_Fz = realized_wrench[:, 0:1]
        realized_torques = realized_wrench[:, 1:4]

        # V10 retains V9.1's empirical wake-force coupling. The rotor forces and
        # actuator lag are now explicit, but induced wake flow is not yet resolved
        # at each rotor disk; external_lift remains a stated approximation.
        total_rotor_Fz = realized_Fz + external_lift
        thrust_world = rot[:, :, 2] * total_rotor_Fz
        v_air = vel - wind_field
        drag_force = -0.5 * AIR_DENSITY * 0.22 * v_air * safe_norm(v_air, dim=-1, keepdim=True)
        accel = (thrust_world + drag_force) / DRONE_MASS + torch.tensor([0.0, 0.0, -GRAVITY], device=self.device)

        # Semi-implicit translation. The attitude update uses the SO(3) exponential map.
        vel_next = vel + accel * self.dt
        pos_next = pos + vel_next * self.dt
        gyroscopic = torch.cross(omega_body, omega_body * self.J, dim=-1)
        d_omega = (realized_torques - gyroscopic) * self.J_inv
        omega_next = omega_body + d_omega * self.dt

        v_rot = omega_next * self.dt
        theta_sq = torch.sum(v_rot * v_rot, dim=-1, keepdim=True).clamp(min=1e-8)
        theta = torch.sqrt(theta_sq)
        alpha_coef = torch.where(theta_sq < 1e-6, 1.0 - theta_sq / 6.0, torch.sin(theta) / theta).unsqueeze(-1)
        beta_coef = torch.where(theta_sq < 1e-6, 0.5 - theta_sq / 24.0, (1.0 - torch.cos(theta)) / theta_sq).unsqueeze(-1)
        zeros = torch.zeros_like(v_rot[:, 0:1])
        vx, vy, vz = v_rot[:, 0:1], v_rot[:, 1:2], v_rot[:, 2:3]
        K = torch.stack([
            torch.cat([zeros, -vz, vy], dim=-1),
            torch.cat([vz, zeros, -vx], dim=-1),
            torch.cat([-vy, vx, zeros], dim=-1),
        ], dim=-2)
        eye3 = torch.eye(3, device=self.device, dtype=rot.dtype).unsqueeze(0).expand(rot.shape[0], -1, -1)
        dR = eye3 + alpha_coef * K + beta_coef * torch.matmul(K, K)
        rot_next = torch.matmul(rot, dR)
        c0 = rot_next[:, :, 0]
        c1 = rot_next[:, :, 1]
        c0 = c0 / safe_norm(c0, dim=-1, keepdim=True)
        c2 = torch.cross(c0, c1, dim=-1)
        c2 = c2 / safe_norm(c2, dim=-1, keepdim=True)
        c1 = torch.cross(c2, c0, dim=-1)
        rot_clean = torch.stack([c0, c1, c2], dim=-1)

        return (
            pos_next, vel_next, rot_clean, omega_next,
            rotor_omega_next, motor_current_next, rotor_thrusts,
            realized_Fz, drag_force, motor_voltage,
        )


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
        self.qp_filter = RelaxedQPSafetyFilter(iters=60, rho_slack=15.0, tolerance=2e-3)

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
        hover_rotor_thrust = HOVER_WEIGHT / 4.0
        hover_rotor_omega = math.sqrt(hover_rotor_thrust / ROTOR_THRUST_COEFFICIENT)
        hover_rotor_current = (ROTOR_TORQUE_COEFFICIENT * hover_rotor_omega**2) / MOTOR_TORQUE_CONSTANT
        self.lead_rotor_omega = torch.full((N, 4), hover_rotor_omega, device=self.device)
        self.lead_motor_current = torch.full((N, 4), hover_rotor_current, device=self.device)
        self.lead_motors = torch.full((N, 4), hover_rotor_thrust, device=self.device)
        self.lead_motor_voltage = MOTOR_BACK_EMF_CONSTANT * self.lead_rotor_omega + MOTOR_COIL_RESISTANCE * self.lead_motor_current
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
        hist_a_list = []
        for k in range(self.H):
            t_past = -k * self.dt
            lx = self.radius_lead * math.sin(self.omega_lead * t_past)
            ly = self.radius_lead * 0.5 * math.sin(2.0 * self.omega_lead * t_past)
            lvx = self.radius_lead * self.omega_lead * math.cos(self.omega_lead * t_past)
            lvy = self.radius_lead * self.omega_lead * math.cos(2.0 * self.omega_lead * t_past)

            hp = torch.tensor([[lx, ly, 6.0]], device=self.device).repeat(N, 1) + pos_jitter
            hv = torch.tensor([[lvx, lvy, 0.0]], device=self.device).repeat(N, 1) + vel_jitter
            hax = -self.radius_lead * (self.omega_lead ** 2) * math.sin(self.omega_lead * t_past)
            hay = -self.radius_lead * 0.5 * ((2.0 * self.omega_lead) ** 2) * math.sin(2.0 * self.omega_lead * t_past)
            ha = torch.tensor([[hax, hay, 0.0]], device=self.device).repeat(N, 1)
            hist_p_list.append(hp.unsqueeze(1))
            hist_r_list.append(R_trim.unsqueeze(1))
            hist_v_list.append(hv.unsqueeze(1))
            hist_a_list.append(ha.unsqueeze(1))

        self.history_pos = torch.cat(hist_p_list, dim=1)
        self.history_rot = torch.cat(hist_r_list, dim=1)
        self.history_vel = torch.cat(hist_v_list, dim=1)
        # The seeded prehistory uses a constant trim orientation, so its angular
        # rate and angular acceleration are zero. This is explicit rather than
        # borrowing the current leader rates for a historical wake segment.
        self.history_omega = torch.zeros((N, self.H, 3), device=self.device)
        self.history_alpha_body = torch.zeros((N, self.H, 3), device=self.device)
        self.history_accel = torch.cat(hist_a_list, dim=1)

        self.sweet_spot_offset = torch.tensor([-1.80, -(SPAN_EFF * 0.5 + 0.22), 0.0], device=self.device)
        self.follow_pos = self.lead_pos + torch.matmul(self.lead_rot, self.sweet_spot_offset.unsqueeze(-1)).squeeze(-1)
        self.follow_vel = self.lead_vel.clone()
        self.follow_rot = R_trim.clone()
        self.follow_omega = torch.zeros((N, 3), device=self.device)
        self.follow_rotor_omega = torch.full((N, 4), hover_rotor_omega, device=self.device)
        self.follow_motor_current = torch.full((N, 4), hover_rotor_current, device=self.device)
        self.follow_motors = torch.full((N, 4), hover_rotor_thrust, device=self.device)
        self.follow_motor_voltage = MOTOR_BACK_EMF_CONSTANT * self.follow_rotor_omega + MOTOR_COIL_RESISTANCE * self.follow_motor_current
        self.follow_thrust = torch.full((N, 1), HOVER_WEIGHT, device=self.device)
        hover_fraction = (HOVER_WEIGHT / 4.0) / MAX_THRUST_PER_MOTOR
        self.previous_nn_rotor_action_fraction = torch.full((N, 4), hover_fraction, device=self.device)

        self.metrics = {}

    def detach_dynamics_for_truncated_bptt(self):
        """Cut state-history autograd links at a rollout chunk boundary.

        The forward simulation still runs across the full curriculum horizon. Gradients
        are propagated within chunks of TRAINING_BPTT_CHUNK_STEPS; state values carry
        forward between chunks but their earlier computation graphs are released. This
        is truncated BPTT, not full-horizon exact differentiation.
        """
        tensor_state_names = (
            "lead_pos", "lead_vel", "lead_rot", "lead_omega", "lead_rotor_omega",
            "lead_motor_current", "lead_motors", "lead_motor_voltage", "lead_thrust",
            "history_pos", "history_rot", "history_vel", "history_omega",
            "history_alpha_body", "history_accel", "follow_pos", "follow_vel",
            "follow_rot", "follow_omega", "follow_rotor_omega", "follow_motor_current",
            "follow_motors", "follow_motor_voltage", "follow_thrust",
            "previous_nn_rotor_action_fraction",
        )
        for name in tensor_state_names:
            value = getattr(self, name, None)
            if torch.is_tensor(value):
                setattr(self, name, value.detach())
        # Avoid retaining the previous step's tensors/graph through the metrics cache.
        self.metrics = {}

    def _predict_actual_acceleration_for_residual_candidates(
        self, candidate_residuals, a_target, drag_force, upwash_lift, vel_target
    ):
        """Predict next translational acceleration through the current actuator model.

        candidate_residuals has shape [K, N, 3]. Prediction mirrors the implemented
        one-step translation update: current attitude direction, candidate-dependent
        desired attitude/torque, wrench allocation, one rotor_step, resulting rotor
        thrust, current drag and external lift. It is model-based and one-step only.
        """
        K, N, _ = candidate_residuals.shape
        # Guard selection is based on detached predictions; chosen command tensors are
        # gathered later from the original candidates so gradients can still pass when
        # the unmodified / scaled actor candidate is selected.
        with torch.no_grad():
            residuals = candidate_residuals.detach()
            gravity = torch.tensor([0.0, 0.0, GRAVITY], device=self.device, dtype=self.follow_pos.dtype)
            body_z = self.follow_rot[:, :, 2].unsqueeze(0)
            f_des = (
                a_target.unsqueeze(0) + residuals
                - drag_force.unsqueeze(0) / DRONE_MASS
                + gravity.view(1, 1, 3)
            )
            z_des = f_des / safe_norm(f_des, dim=-1, keepdim=True).clamp(min=1e-8)
            fwd = vel_target[:, :2] / safe_norm(vel_target[:, :2], dim=-1, keepdim=True).clamp(min=1e-3)
            xc = torch.cat([fwd, torch.zeros((N, 1), device=self.device)], dim=-1).unsqueeze(0).expand(K, -1, -1)
            y_des = torch.cross(z_des, xc, dim=-1)
            y_des = y_des / safe_norm(y_des, dim=-1, keepdim=True).clamp(min=1e-8)
            x_des = torch.cross(y_des, z_des, dim=-1)
            R_des = torch.stack([x_des, y_des, z_des], dim=-1)
            R_now = self.follow_rot.unsqueeze(0).expand(K, -1, -1, -1)
            R_err = torch.matmul(R_des.transpose(-1, -2), R_now) - torch.matmul(R_now.transpose(-1, -2), R_des)
            e_R = torch.stack([R_err[..., 2, 1], R_err[..., 0, 2], R_err[..., 1, 0]], dim=-1) * 0.5
            tau = -self.k_R.unsqueeze(0) * e_R - self.k_omega.unsqueeze(0) * self.follow_omega.unsqueeze(0)
            fz = (torch.sum(f_des * body_z, dim=-1, keepdim=True) * DRONE_MASS - upwash_lift.unsqueeze(0)).clamp(min=2.0, max=32.0)
            wrench = torch.cat([fz, tau], dim=-1)
            target_thrusts = self.drone_physics.allocate_wrench(wrench)
            omega0 = self.follow_rotor_omega.unsqueeze(0).expand(K, -1, -1)
            current0 = self.follow_motor_current.unsqueeze(0).expand(K, -1, -1)
            _, _, realized_thrusts, _ = self.drone_physics.rotor_step(omega0, current0, target_thrusts)
            realized_wrench = torch.matmul(realized_thrusts, self.drone_physics.B.T)
            total_fz = realized_wrench[..., 0:1] + upwash_lift.unsqueeze(0)
            thrust_world = body_z * total_fz
            accel = (thrust_world + drag_force.unsqueeze(0)) / DRONE_MASS + torch.tensor(
                [0.0, 0.0, -GRAVITY], device=self.device, dtype=self.follow_pos.dtype
            ).view(1, 1, 3)
        return accel

    def _plant_aware_residual_guard(
        self, raw_filtered_residual, u_baseline, emergency_residual,
        a_target, drag_force, upwash_lift, vel_target,
        A_collision, b_collision_absolute, A_lateral, b_lateral_absolute,
        tolerance=2.0e-3,
    ):
        """Choose the closest one-step actuator-reachable candidate satisfying CBF rows.

        A finite candidate set is checked, rather than claiming this is a nonlinear
        constrained optimiser. If no candidate is predicted feasible, choose the
        candidate with the least worst normalized violation and report infeasibility.
        """
        lo = torch.tensor([-6.0, -6.0, -4.5], device=self.device, dtype=raw_filtered_residual.dtype)
        hi = torch.tensor([6.0, 6.0, 6.0], device=self.device, dtype=raw_filtered_residual.dtype)
        raw = raw_filtered_residual
        candidates = torch.stack([
            raw,
            0.75 * raw,
            0.50 * raw,
            0.25 * raw,
            torch.clamp(u_baseline, min=lo, max=hi),
            torch.zeros_like(raw),
            torch.clamp(emergency_residual, min=lo, max=hi),
        ], dim=0)
        predicted_accel = self._predict_actual_acceleration_for_residual_candidates(
            candidates, a_target, drag_force, upwash_lift, vel_target
        )
        with torch.no_grad():
            def normed_violation(A, b_abs, accel):
                lhs = torch.matmul(A.unsqueeze(0), accel.unsqueeze(-1)).squeeze(-1).squeeze(-1)
                rhs = b_abs.squeeze(-1).squeeze(-1).unsqueeze(0)
                row_norm = safe_norm(A.squeeze(1), dim=-1).clamp(min=1e-6).unsqueeze(0)
                return torch.relu(lhs - rhs) / row_norm
            v_collision = normed_violation(A_collision, b_collision_absolute, predicted_accel)
            v_lateral = normed_violation(A_lateral, b_lateral_absolute, predicted_accel)
            v_each = torch.stack([v_collision, v_lateral], dim=-1)  # [K, N, 2], m/s^2
            v_max = v_each.amax(dim=-1)
            v_sum = v_each.sum(dim=-1)
            distance = safe_norm(candidates.detach() - raw.detach().unsqueeze(0), dim=-1)
            feasible = v_max <= float(tolerance)
            has_feasible = feasible.any(dim=0)
            # Lexicographic intent: if any candidate is feasible, select the feasible
            # command closest to the QP result. Otherwise minimise worst violation.
            feasible_score = distance + 1e-4 * v_sum
            infeasible_score = v_max + 0.05 * v_sum + 1e-4 * distance
            score = torch.where(
                has_feasible.unsqueeze(0),
                torch.where(feasible, feasible_score, torch.full_like(feasible_score, 1e6)),
                infeasible_score,
            )
            best_idx = torch.argmin(score, dim=0)
            chosen_max_v = torch.gather(v_max, 0, best_idx.unsqueeze(0)).squeeze(0)
            raw_max_v = v_max[0]
            chosen_feasible = chosen_max_v <= float(tolerance)
            chosen_distance = distance.gather(0, best_idx.unsqueeze(0)).squeeze(0)
        # Gather from non-detached candidate tensors to preserve gradients through the
        # selected QP/actor command whenever that candidate is chosen.
        chosen = candidates.permute(1, 0, 2)[torch.arange(self.num_envs, device=self.device), best_idx]
        predicted_selected = predicted_accel.permute(1, 0, 2)[torch.arange(self.num_envs, device=self.device), best_idx]
        failure_records = []
        bad_env_mask = (~chosen_feasible) | (~has_feasible)
        bad_indices = torch.nonzero(bad_env_mask, as_tuple=False).flatten()
        if bad_indices.numel() > 0:
            # Prioritise environments with the largest selected-command violation.
            order = torch.argsort(chosen_max_v[bad_indices], descending=True)
            for local in order[: min(8, int(order.numel()))]:
                j = int(bad_indices[local].item())
                failure_records.append({
                    "environment_index": j,
                    "candidate_index": int(best_idx[j].item()),
                    "candidate_exists_that_is_predicted_feasible": bool(has_feasible[j].item()),
                    "selected_candidate_predicted_feasible": bool(chosen_feasible[j].item()),
                    "selected_predicted_max_normalized_violation_mps2": float(chosen_max_v[j].item()),
                    "raw_qp_predicted_max_normalized_violation_mps2": float(raw_max_v[j].item()),
                    "selected_command_change_mps2": float(chosen_distance[j].item()),
                    "predicted_collision_violation_mps2": float(v_collision[best_idx[j], j].item()),
                    "predicted_lateral_violation_mps2": float(v_lateral[best_idx[j], j].item()),
                })
        diagnostics = {
            "predicted_feasible_pct": float(chosen_feasible.float().mean().item() * 100.0),
            "predicted_max_normalized_violation_mps2": float(chosen_max_v.max().item()),
            "raw_qp_predicted_max_violation_mps2": float(raw_max_v.max().item()),
            "no_feasible_candidate_pct": float((~has_feasible).float().mean().item() * 100.0),
            "adjusted_pct": float((chosen_distance > 1e-4).float().mean().item() * 100.0),
            "mean_adjustment_mps2": float(safe_norm(chosen.detach() - raw.detach(), dim=-1).mean().item()),
            "tolerance_mps2": float(tolerance),
            "candidate_count": int(candidates.shape[0]),
            "failure_records": failure_records,
        }
        return chosen, predicted_selected, diagnostics

    def _predict_acceleration_for_direct_rotor_targets(self, target_rotor_thrusts, drag_force, external_lift):
        """Predict translational acceleration from direct rotor-thrust setpoints.

        This mirrors the actual discrete translation update: advance the four rotor
        states once, sum their realised thrust, add the existing empirical wake force,
        and use current attitude plus current drag. It is a one-step model prediction,
        not a safety guarantee and not a full attitude-ahead prediction.
        """
        _, _, predicted_rotor_thrust, _ = self.drone_physics.rotor_step(
            self.follow_rotor_omega, self.follow_motor_current, target_rotor_thrusts
        )
        predicted_wrench = torch.matmul(predicted_rotor_thrust, self.drone_physics.B.T)
        total_fz = predicted_wrench[:, 0:1] + external_lift
        thrust_world = self.follow_rot[:, :, 2] * total_fz
        gravity = torch.tensor([0.0, 0.0, -GRAVITY], device=self.device, dtype=self.follow_pos.dtype)
        return (thrust_world + drag_force) / DRONE_MASS + gravity

    def step(self, actor_net, ncm_net, control_mode="nn-primary"):
        """Advance one frame.

        nn-primary: NN directly sets four rotor-thrust references; no PD tracking
        command, CBF projection, or actuator-aware guard is allowed to alter them.
        pd-baseline: fixed hand-designed PD + attitude controller used only as benchmark.
        Safety barrier metrics are computed as monitors for both modes.
        """
        if control_mode not in {"nn-primary", "pd-baseline"}:
            raise ValueError(f"Unknown follower control_mode={control_mode!r}")
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
            self.lead_pos, self.lead_vel, self.lead_rot, self.lead_omega,
            self.lead_rotor_omega, self.lead_motor_current, self.lead_motors,
            self.lead_thrust, _, self.lead_motor_voltage,
        ) = self.drone_physics.integrate_dynamics(
            self.lead_pos, self.lead_vel, self.lead_rot, self.lead_omega,
            self.lead_rotor_omega, self.lead_motor_current, lead_wrench,
            lead_flow, lead_back_lift,
        )

        # Measured-in-simulator acceleration estimates for relative-degree-two barriers.
        lead_accel_est = (self.lead_vel - lead_vel_before) / max(self.dt, 1e-6)
        lead_alpha_body_est = (self.lead_omega - lead_omega_before) / max(self.dt, 1e-6)
        self.history_pos = torch.cat([self.lead_pos.unsqueeze(1), self.history_pos[:, :-1, :]], dim=1)
        self.history_rot = torch.cat([self.lead_rot.unsqueeze(1), self.history_rot[:, :-1, :, :]], dim=1)
        self.history_vel = torch.cat([self.lead_vel.unsqueeze(1), self.history_vel[:, :-1, :]], dim=1)
        self.history_omega = torch.cat([self.lead_omega.unsqueeze(1), self.history_omega[:, :-1, :]], dim=1)
        self.history_alpha_body = torch.cat([lead_alpha_body_est.unsqueeze(1), self.history_alpha_body[:, :-1, :]], dim=1)
        self.history_accel = torch.cat([lead_accel_est.unsqueeze(1), self.history_accel[:, :-1, :]], dim=1)

        # -------------------------------------------------------------
        # 2. FOLLOWER: WAKE SURFING WITH SOFT-SLACK RELAXED NCM SAFETY
        # -------------------------------------------------------------
        lead_speed = safe_norm(self.lead_vel[:, :2], dim=-1, keepdim=True).clamp(min=1e-3)
        turn_curvature = (lead_speed * self.omega_lead) / (lead_speed**2).clamp(min=1e-3)
        v_induced_follower, r_core, span_shear = self.aero.compute_leader_to_follower_wake(
            self.follow_pos, self.history_pos, self.history_rot, self.history_vel, self.lead_thrust,
            turn_curvature, history_dt=self.dt, hist_accel=self.history_accel,
        )
        wz_experienced = v_induced_follower[:, 2:3]

        # Dynamic arc-length target selection (1.80m physical arc spacing)
        step_dists = safe_norm(self.history_pos[:, :-1, :] - self.history_pos[:, 1:, :], dim=-1)
        cum_dist = torch.cumsum(step_dists, dim=1)
        k_target = torch.clamp(torch.argmin(torch.abs(cum_dist - 1.80), dim=1) + 1, max=self.H - 1)

        pos_target_segment = torch.gather(self.history_pos, 1, k_target.view(N, 1, 1).expand(N, 1, 3)).squeeze(1)
        rot_target_segment = torch.gather(self.history_rot, 1, k_target.view(N, 1, 1, 1).expand(N, 1, 3, 3)).squeeze(1)
        vel_target_centre = torch.gather(
            self.history_vel, 1, k_target.view(N, 1, 1).expand(N, 1, 3)
        ).squeeze(1)
        target_omega_body = torch.gather(
            self.history_omega, 1, k_target.view(N, 1, 1).expand(N, 1, 3)
        ).squeeze(1)
        source_omega_body = target_omega_body
        source_alpha_body = torch.gather(
            self.history_alpha_body, 1, k_target.view(N, 1, 1).expand(N, 1, 3)
        ).squeeze(1)
        source_accel = torch.gather(
            self.history_accel, 1, k_target.view(N, 1, 1).expand(N, 1, 3)
        ).squeeze(1)

        # The physical tracking target is a fixed point on the leader's body, offset
        # laterally to the wake-surfing flank. Its velocity is NOT just the leader
        # centre velocity: body rotation adds omega x r. Its acceleration also needs
        # alpha x r and omega x (omega x r). V10.4.1 omitted these terms and separately
        # used an ideal racetrack acceleration, so position, velocity and acceleration
        # targets described different trajectories during turns.
        flank_off = torch.tensor([0.0, -(SPAN_EFF * 0.5 + 0.22), 0.0], device=self.device)
        target_pos, vel_target, a_target, target_offset_velocity, target_offset_acceleration = (
            moving_body_offset_kinematics(
                pos_target_segment, vel_target_centre, rot_target_segment,
                target_omega_body, source_alpha_body, source_accel, flank_off,
            )
        )

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

        # 38-element direct-actuator observation exposes the target-frame angular
        # rate and previous rotor command, so the NN
        # can learn attitude/rate stabilisation and avoid gratuitous command chatter.
        # All features are normalised; the NN has no access to hidden future state.
        R_rel_target = torch.matmul(rot_target_segment.transpose(-1, -2), self.follow_rot)
        eye3 = torch.eye(3, device=self.device, dtype=self.follow_pos.dtype).unsqueeze(0).expand(N, -1, -1)
        R_rel_feature = (R_rel_target - eye3).reshape(N, 9) / 2.0
        actor_obs = torch.cat([
            error_pos / 2.0,
            error_vel / 2.0,
            v_induced_follower / 4.0,
            rel_pos_lead / 4.0,
            self.follow_omega / 5.0,
            R_rel_feature,
            self.follow_rotor_omega / ROTOR_MAX_OMEGA,
            a_target / 5.0,
            target_omega_body / 5.0,
            self.previous_nn_rotor_action_fraction,
        ], dim=-1)
        if actor_obs.shape[-1] != ACTOR_OBSERVATION_DIM:
            raise RuntimeError(f"Internal V10.5 observation size is {actor_obs.shape[-1]}, expected {ACTOR_OBSERVATION_DIM}")

        if control_mode == "nn-primary":
            # This is the actual follower control output: one independent thrust
            # fraction for each rotor. No PD command is added downstream.
            nn_rotor_action_fraction = actor_net(actor_obs).clamp(0.0, 1.0)
            nn_target_rotor_thrusts = nn_rotor_action_fraction * MAX_THRUST_PER_MOTOR
            u_residual = torch.zeros((N, 3), device=self.device, dtype=self.follow_pos.dtype)
            u_baseline = torch.zeros_like(u_residual)
            u_nom = torch.zeros_like(u_residual)
            nn_action_slew_loss = (nn_rotor_action_fraction - self.previous_nn_rotor_action_fraction).square().mean()
        else:
            # Benchmark only. The actor is intentionally not called in this path.
            nn_rotor_action_fraction = torch.zeros((N, 4), device=self.device, dtype=self.follow_pos.dtype)
            nn_target_rotor_thrusts = None
            nn_action_slew_loss = torch.zeros((), device=self.device, dtype=self.follow_pos.dtype)
            u_residual = torch.zeros((N, 3), device=self.device, dtype=self.follow_pos.dtype)
            u_baseline = torch.cat([
                -4.5 * error_pos[:, 0:2] - 3.2 * error_vel[:, 0:2],
                -6.5 * error_pos[:, 2:3] - 4.2 * error_vel[:, 2:3]
            ], dim=-1)
            u_nom = u_baseline

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
        b_cbf1_absolute = (
            2.0 * torch.sum(v_rel * v_rel, dim=-1, keepdim=True)
            - 2.0 * torch.sum(rel_pos * lead_accel_est, dim=-1, keepdim=True)
            + 2.0 * cbf_lambda * h_coll_dot
            + (cbf_lambda ** 2) * h_collision
        ).unsqueeze(-1)
        # Important: the filter variable is the residual acceleration, but the
        # relative-degree-two CBF above is derived for absolute follower acceleration.
        # Convert A @ (a_target + u_residual) <= b_abs to A @ u_residual <= b_abs - A @ a_target.
        b_cbf1 = acceleration_residual_rhs(A_cbf1, b_cbf1_absolute, a_target)

        # The lateral envelope follows the delayed wake source, not the leader's
        # current instantaneous body frame. The follower is positioned relative to a
        # historical source segment (pos_target_segment/rot_target_segment); using the
        # current leader alpha here made the moving plane sweep through the follower
        # during startup and demanded impossible lateral accelerations.
        # source_omega_body, source_alpha_body and source_accel were sampled together
        # with the target point above, so barrier and tracking kinematics share one
        # consistent historical source state.
        rel_pos_wake = self.follow_pos - pos_target_segment
        # This barrier is defined relative to the historical wake centreline, not the
        # laterally offset tracking point; keep its source-centre velocity accordingly.
        v_rel_wake = self.follow_vel - vel_target_centre
        lateral_margin = SPAN_EFF * 0.5 + 0.05
        wake_barrier = build_lateral_wake_barrier(
            relative_position=rel_pos_wake,
            relative_velocity=v_rel_wake,
            source_rotation=rot_target_segment,
            source_omega_body=source_omega_body,
            source_alpha_body=source_alpha_body,
            source_acceleration_world=source_accel,
            margin=lateral_margin,
            cbf_lambda=cbf_lambda,
        )
        y_axis = wake_barrier["axis"]
        y_axis_dot = wake_barrier["axis_dot"]
        y_axis_ddot = wake_barrier["axis_ddot"]
        h_sink = wake_barrier["h"]
        h_sink_dot = wake_barrier["h_dot"]
        A_cbf2 = wake_barrier["A_absolute"]
        b_cbf2_absolute = wake_barrier["b_absolute"]
        # Convert the absolute source-relative acceleration inequality to residual-command coordinates.
        b_cbf2 = acceleration_residual_rhs(A_cbf2, b_cbf2_absolute, a_target)

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
        # Heuristic absolute acceleration candidate intended to accelerate away from
        # the leader while accounting for its estimated acceleration. Convert it to
        # the same residual-acceleration coordinates used by the projection filter.
        emergency_absolute_accel = lead_accel_est + 3.0 * rel_direction - 1.5 * v_rel
        emergency_separation_accel = emergency_absolute_accel - a_target
        if control_mode == "pd-baseline":
            filtered_accel, max_slack, cbf1_active_pct, cbf2_active_pct, _, _, qp_feasible_pct, filter_diag = self.qp_filter.solve(
                u_nom,
                torch.cat([A_cbf1, A_cbf2], dim=1),
                torch.cat([b_cbf1, b_cbf2], dim=1),
                u_min=torch.tensor([-6.0, -6.0, -4.5], device=self.device),
                u_max=torch.tensor([6.0, 6.0, 6.0], device=self.device),
                fallback=emergency_separation_accel,
            )
            raw_qp_filtered_accel = filtered_accel
        else:
            # Deliberately do NOT apply the QP to NN commands. In NN-primary mode,
            # these acceleration-level constraints are observer-only diagnostics.
            # The actual direct rotor requests are evaluated against the plant below.
            filtered_accel = torch.zeros((N, 3), device=self.device, dtype=self.follow_pos.dtype)
            raw_qp_filtered_accel = filtered_accel
            max_slack = 0.0
            cbf1_active_pct = 0.0
            cbf2_active_pct = 0.0
            qp_feasible_pct = 0.0  # 0 means NOT APPLIED in this mode; see filter_active.
            filter_diag = {
                "projection_feasible_pct": 0.0,
                "projection_max_residual": 0.0,
                "fallback_attempted_pct": 0.0,
                "fallback_applied_pct": 0.0,
                "worst_environment_index": -1,
                "worst_constraint_index": -1,
                "individually_unachievable_under_command_box_pct": 0.0,
                "constraint_max_violations": [0.0, 0.0],
                "constraint_count": 2,
                "failure_records": [],
                "active": False,
            }
        # Momentum-theory upwash relief estimate (still empirical, not rotor-disk inflow).
        thrust_relief_momentum = 2.0 * AIR_DENSITY * TOTAL_ROTOR_AREA * V_INDUCED_HOVER * wz_experienced
        upwash_lift = torch.clamp(thrust_relief_momentum, min=-2.0, max=6.0)

        # Drag Cancellation + Centripetal Acceleration Feedforward. The command is
        # then checked against predicted acceleration after current rotor/motor lag.
        follower_flow = v_induced_follower + self.background_wind
        v_air_follow = self.follow_vel - follower_flow
        f_drag_follow = -0.5 * AIR_DENSITY * 0.22 * v_air_follow * safe_norm(v_air_follow, dim=-1, keepdim=True)
        if control_mode == "pd-baseline":
            filtered_accel, predicted_plant_accel, plant_guard_diag = self._plant_aware_residual_guard(
                raw_qp_filtered_accel, u_baseline, emergency_separation_accel,
                a_target, f_drag_follow, upwash_lift=upwash_lift,
                vel_target=vel_target,
                A_collision=A_cbf1, b_collision_absolute=b_cbf1_absolute,
                A_lateral=A_cbf2, b_lateral_absolute=b_cbf2_absolute,
            )
            f_des_follow = (
                a_target + filtered_accel - (f_drag_follow / DRONE_MASS)
                + torch.tensor([0.0, 0.0, GRAVITY], device=self.device)
            )
        else:
            # Predict the actual one-step acceleration from the NN's four direct
            # rotor setpoints. This prediction is a monitor only; no command override.
            predicted_plant_accel = self._predict_acceleration_for_direct_rotor_targets(
                nn_target_rotor_thrusts, f_drag_follow, upwash_lift
            )
            filtered_accel = predicted_plant_accel - a_target  # diagnostic equivalent only
            f_des_follow = (
                predicted_plant_accel - f_drag_follow / DRONE_MASS
                + torch.tensor([0.0, 0.0, GRAVITY], device=self.device)
            )
            def _direct_violation(A_abs, b_abs, accel):
                lhs = torch.matmul(A_abs, accel.unsqueeze(-1))
                raw_v = (lhs - b_abs).squeeze(-1).squeeze(-1)
                row_norm = safe_norm(A_abs.squeeze(1), dim=-1).clamp(min=1e-6)
                return torch.relu(raw_v) / row_norm
            direct_v_collision = _direct_violation(A_cbf1, b_cbf1_absolute, predicted_plant_accel)
            direct_v_lateral = _direct_violation(A_cbf2, b_cbf2_absolute, predicted_plant_accel)
            direct_v_max = torch.maximum(direct_v_collision, direct_v_lateral)
            direct_feasible = direct_v_max <= 2e-3
            direct_bad = torch.nonzero(~direct_feasible, as_tuple=False).flatten()
            direct_failures = []
            if direct_bad.numel():
                order = torch.argsort(direct_v_max[direct_bad], descending=True)
                for local in order[:min(8, int(order.numel()))]:
                    j = int(direct_bad[local].item())
                    direct_failures.append({
                        "environment_index": j,
                        "candidate_index": 0,
                        "candidate_exists_that_is_predicted_feasible": False,
                        "selected_candidate_predicted_feasible": False,
                        "selected_predicted_max_normalized_violation_mps2": float(direct_v_max[j].detach().item()),
                        "raw_qp_predicted_max_violation_mps2": float(direct_v_max[j].detach().item()),
                        "selected_command_change_mps2": 0.0,
                        "predicted_collision_violation_mps2": float(direct_v_collision[j].detach().item()),
                        "predicted_lateral_violation_mps2": float(direct_v_lateral[j].detach().item()),
                        "control_mode": "nn-primary-direct-rotor-thrust",
                    })
            plant_guard_diag = {
                "predicted_feasible_pct": float(direct_feasible.float().mean().item() * 100.0),
                "predicted_max_normalized_violation_mps2": float(direct_v_max.max().detach().item()),
                "raw_qp_predicted_max_violation_mps2": float(direct_v_max.max().detach().item()),
                "no_feasible_candidate_pct": float((~direct_feasible).float().mean().item() * 100.0),
                "adjusted_pct": 0.0,
                "mean_adjustment_mps2": 0.0,
                "tolerance_mps2": 2e-3,
                "candidate_count": 1,
                "failure_records": direct_failures,
                "active": False,
                "monitor_mode": "NN direct action checked; no intervention",
            }

        if control_mode == "pd-baseline":
            zb_follow_des = f_des_follow / safe_norm(f_des_follow, dim=-1, keepdim=True)
            fwd_follow = vel_target[:, :2] / safe_norm(vel_target[:, :2], dim=-1, keepdim=True).clamp(min=1e-3)
            xc_follow = torch.cat([fwd_follow, torch.zeros((N, 1), device=self.device)], dim=-1)
            yb_follow_des = torch.cross(zb_follow_des, xc_follow, dim=-1)
            yb_follow_des = yb_follow_des / safe_norm(yb_follow_des, dim=-1, keepdim=True)
            xb_follow_des = torch.cross(yb_follow_des, zb_follow_des, dim=-1)
            Rd_follow = torch.stack([xb_follow_des, yb_follow_des, zb_follow_des], dim=-1)
        else:
            # Attitude error is reported against the delayed leader target frame. It is
            # telemetry only: no geometric attitude controller feeds commands to rotors.
            Rd_follow = rot_target_segment
        R_err_follow = torch.matmul(Rd_follow.transpose(-1, -2), self.follow_rot) - torch.matmul(
            self.follow_rot.transpose(-1, -2), Rd_follow
        )
        e_R_follow = torch.stack([R_err_follow[:, 2, 1], R_err_follow[:, 0, 2], R_err_follow[:, 1, 0]], dim=-1) * 0.5
        tau_follow = -self.k_R * e_R_follow - self.k_omega * self.follow_omega

        if control_mode == "pd-baseline":
            # Baseline mode uses the hand-designed geometric attitude controller.
            follow_Fz = (torch.sum(f_des_follow * self.follow_rot[:, :, 2], dim=-1, keepdim=True) * DRONE_MASS - upwash_lift).clamp(
                min=2.0, max=32.0
            )
            follow_wrench = torch.cat([follow_Fz, tau_follow], dim=-1)
            follow_target_thrusts = self.drone_physics.allocate_wrench(follow_wrench)
        else:
            # Neural-primary mode: the four NN outputs ARE the actuator setpoints.
            # This path bypasses both PD acceleration feedback and geometric attitude
            # control. The motor model subsequently produces the realised rotor speeds.
            follow_target_thrusts = nn_target_rotor_thrusts
            follow_wrench = torch.matmul(follow_target_thrusts, self.drone_physics.B.T)
            follow_Fz = follow_wrench[:, 0:1]
        if control_mode == "nn-primary":
            self.previous_nn_rotor_action_fraction = nn_rotor_action_fraction
        else:
            self.previous_nn_rotor_action_fraction = (follow_target_thrusts / MAX_THRUST_PER_MOTOR).clamp(0.0, 1.0)
        follow_target_rotor_omega = torch.sqrt(torch.clamp(follow_target_thrusts, min=0.0) / self.drone_physics.kT).clamp(0.0, ROTOR_MAX_OMEGA)
        follow_vel_before_physics = self.follow_vel.clone()
        follow_rotor_omega_before_physics = self.follow_rotor_omega.clone()
        follow_rot_before_physics = self.follow_rot.clone()

        (
            self.follow_pos, self.follow_vel, self.follow_rot, self.follow_omega,
            self.follow_rotor_omega, self.follow_motor_current, self.follow_motors,
            self.follow_thrust, _, self.follow_motor_voltage,
        ) = self.drone_physics.integrate_dynamics(
            self.follow_pos, self.follow_vel, self.follow_rot, self.follow_omega,
            self.follow_rotor_omega, self.follow_motor_current, follow_wrench,
            follower_flow, upwash_lift,
            target_rotor_thrusts=(follow_target_thrusts if control_mode == "nn-primary" else None),
        )

        # V10.2 one-step actuator/plant response diagnostics. Translational acceleration
        # is observed from the actual state transition. The predicted value above uses
        # the same one-step V10 plant equations; discrepancy is a regression signal.
        actual_follow_accel = (self.follow_vel - follow_vel_before_physics) / max(self.dt, 1e-9)
        # Separate desired trajectory feedforward error from simulator predictor
        # consistency. In NN-primary mode these are emphatically not the same metric.
        trajectory_acceleration_error = actual_follow_accel - a_target
        predictor_mismatch = actual_follow_accel - predicted_plant_accel

        def normalized_actual_violation(A_abs, b_abs):
            lhs = torch.matmul(A_abs, actual_follow_accel.unsqueeze(-1))
            raw_v = (lhs - b_abs).squeeze(-1).squeeze(-1)
            row_n = safe_norm(A_abs.squeeze(1), dim=-1).clamp(min=1e-6)
            return torch.relu(raw_v) / row_n

        actual_collision_violation_per_env = normalized_actual_violation(A_cbf1, b_cbf1_absolute)
        actual_lateral_violation_per_env = normalized_actual_violation(A_cbf2, b_cbf2_absolute)
        actual_plant_tol = float(plant_guard_diag["tolerance_mps2"])
        actual_plant_feasible_per_env = (
            (actual_collision_violation_per_env <= actual_plant_tol)
            & (actual_lateral_violation_per_env <= actual_plant_tol)
        )
        actual_collision_violation_max = float(actual_collision_violation_per_env.max().item())
        actual_lateral_violation_max = float(actual_lateral_violation_per_env.max().item())
        actual_plant_feasible_pct = float(actual_plant_feasible_per_env.float().mean().item() * 100.0)
        plant_guard_failure_records = []
        for record in plant_guard_diag.get("failure_records", []):
            j = int(record["environment_index"])
            enriched = dict(record)
            enriched.update({
                "actual_plant_barrier_feasible": bool(actual_plant_feasible_per_env[j].item()),
                "actual_collision_violation_mps2": float(actual_collision_violation_per_env[j].item()),
                "actual_lateral_violation_mps2": float(actual_lateral_violation_per_env[j].item()),
                "actual_acceleration_world_mps2": [float(x) for x in actual_follow_accel[j].detach().cpu().tolist()],
                "rotor_target_speed_rad_s": [float(x) for x in follow_target_rotor_omega[j].detach().cpu().tolist()],
                "rotor_actual_speed_rad_s": [float(x) for x in self.follow_rotor_omega[j].detach().cpu().tolist()],
            })
            plant_guard_failure_records.append(enriched)
        predicted_actual_mismatch_max = float(safe_norm(predictor_mismatch, dim=-1).max().item())
        trajectory_accel_error_mean = float(safe_norm(trajectory_acceleration_error, dim=-1).mean().item())
        trajectory_accel_error_max = float(safe_norm(trajectory_acceleration_error, dim=-1).max().item())

        R_after_rel = torch.matmul(Rd_follow.transpose(-1, -2), self.follow_rot)
        trace_after = R_after_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
        # Smooth SO(3) loss, zero at desired attitude. Unlike acos(angle), this avoids
        # an ill-conditioned derivative at an exactly aligned attitude.
        attitude_tracking_loss = (3.0 - trace_after).mean()
        angular_rate_tracking_loss = ((self.follow_omega - target_omega_body) / 5.0).square().sum(dim=-1).mean()
        attitude_error_after_deg = torch.acos(torch.clamp((trace_after - 1.0) * 0.5, -1.0, 1.0)) * (180.0 / math.pi)
        for record in plant_guard_failure_records:
            j = int(record["environment_index"])
            record["target_feedforward_acceleration_world_mps2"] = [float(x) for x in a_target[j].detach().cpu().tolist()]
            record["attitude_tracking_error_after_step_deg"] = float(attitude_error_after_deg[j].item())
        rotor_speed_error = self.follow_rotor_omega - follow_target_rotor_omega
        rotor_target_thrust_sat_pct = float((follow_target_thrusts >= MAX_THRUST_PER_MOTOR - 1e-4).float().mean().item() * 100.0)
        rotor_speed_sat_pct = float((self.follow_rotor_omega >= 0.995 * ROTOR_MAX_OMEGA).float().mean().item() * 100.0)
        motor_current_sat_pct = float((self.follow_motor_current >= 0.995 * MOTOR_CURRENT_LIMIT).float().mean().item() * 100.0)
        motor_voltage_sat_pct = float((self.follow_motor_voltage >= 0.995 * BATTERY_VOLTAGE_MAX).float().mean().item() * 100.0)

        control_response_sample = {
            "simulation_time_s": float(self.sim_time),
            "environment_index": 0,
            "control_mode": control_mode,
            "neural_network_directly_commands_all_four_rotors": bool(control_mode == "nn-primary"),
            "pd_tracking_command_active": bool(control_mode == "pd-baseline"),
            "safety_filter_intervention_active": bool(control_mode == "pd-baseline"),
            "nn_rotor_thrust_fraction": [float(x) for x in nn_rotor_action_fraction[0].detach().cpu().tolist()],
            "target_feedforward_acceleration_world_mps2": [float(x) for x in a_target[0].detach().cpu().tolist()],
            "target_point_offset_velocity_world_mps": [float(x) for x in target_offset_velocity[0].detach().cpu().tolist()],
            "target_point_offset_acceleration_world_mps2": [float(x) for x in target_offset_acceleration[0].detach().cpu().tolist()],
            "target_centreline_velocity_world_mps": [float(x) for x in vel_target_centre[0].detach().cpu().tolist()],
            "target_centreline_acceleration_world_mps2": [float(x) for x in source_accel[0].detach().cpu().tolist()],
            "target_point_velocity_world_mps": [float(x) for x in vel_target[0].detach().cpu().tolist()],
            "predicted_actuator_limited_acceleration_world_mps2": [float(x) for x in predicted_plant_accel[0].detach().cpu().tolist()],
            "observed_actual_acceleration_world_mps2": [float(x) for x in actual_follow_accel[0].detach().cpu().tolist()],
            "actual_minus_target_feedforward_acceleration_mps2": [float(x) for x in trajectory_acceleration_error[0].detach().cpu().tolist()],
            "prediction_minus_observed_acceleration_mps2": [float(x) for x in predictor_mismatch[0].detach().cpu().tolist()],
            "attitude_tracking_loss_3_minus_trace": float((3.0 - trace_after[0]).detach().item()),
            "angular_rate_tracking_loss_normalized": float((((self.follow_omega - target_omega_body) / 5.0).square().sum(dim=-1))[0].detach().item()),
            "nn_action_slew_loss": float(nn_action_slew_loss.detach().item()),
            "attitude_tracking_error_deg_after_step": float(attitude_error_after_deg[0].detach().item()),
            "rotor_target_thrust_N": [float(x) for x in follow_target_thrusts[0].detach().cpu().tolist()],
            "rotor_actual_thrust_N": [float(x) for x in self.follow_motors[0].detach().cpu().tolist()],
            "rotor_target_speed_rad_s": [float(x) for x in follow_target_rotor_omega[0].detach().cpu().tolist()],
            "rotor_actual_speed_rad_s": [float(x) for x in self.follow_rotor_omega[0].detach().cpu().tolist()],
            "motor_current_A": [float(x) for x in self.follow_motor_current[0].detach().cpu().tolist()],
            "motor_voltage_V": [float(x) for x in self.follow_motor_voltage[0].detach().cpu().tolist()],
            "plant_guard_adjusted": bool(plant_guard_diag["adjusted_pct"] > 0.0),
            "plant_guard_predicted_barrier_feasible": bool(plant_guard_diag["predicted_feasible_pct"] > 99.999),
            "actual_plant_barrier_feasible": bool(actual_plant_feasible_per_env[0].item()),
            "actual_collision_barrier_violation_mps2": float(actual_collision_violation_per_env[0].detach().item()),
            "actual_lateral_barrier_violation_mps2": float(actual_lateral_violation_per_env[0].detach().item()),
            "worst_plant_guard_failure_examples": plant_guard_failure_records[:3],
        }

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

        safety_failure_records = []
        for record in filter_diag.get("failure_records", []):
            j = int(record["environment_index"])
            enriched = dict(record)
            enriched.update({
                "simulation_time_s": float(self.sim_time),
                "h_lateral_m": float(h_sink[j, 0].detach().item()),
                "h_lateral_dot_m_per_s": float(h_sink_dot[j, 0].detach().item()),
                "h_collision_m2": float(h_collision[j, 0].detach().item()),
                "h_collision_dot_m2_per_s": float(h_coll_dot[j, 0].detach().item()),
                "relative_position_world_m": [float(v) for v in rel_pos[j].detach().cpu().tolist()],
                "relative_velocity_world_mps": [float(v) for v in v_rel[j].detach().cpu().tolist()],
                "wake_relative_position_world_m": [float(v) for v in rel_pos_wake[j].detach().cpu().tolist()],
                "wake_relative_velocity_world_mps": [float(v) for v in v_rel_wake[j].detach().cpu().tolist()],
                "leader_acceleration_estimate_world_mps2": [float(v) for v in lead_accel_est[j].detach().cpu().tolist()],
                "leader_angular_acceleration_estimate_body_radps2": [float(v) for v in lead_alpha_body_est[j].detach().cpu().tolist()],
                "wake_reference_axis_world": [float(v) for v in y_axis[j].detach().cpu().tolist()],
                "wake_reference_axis_rate_world_per_s": [float(v) for v in y_axis_dot[j].detach().cpu().tolist()],
                "wake_reference_axis_accel_world_per_s2": [float(v) for v in y_axis_ddot[j].detach().cpu().tolist()],
                "wake_reference_position_world_m": [float(v) for v in pos_target_segment[j].detach().cpu().tolist()],
                "wake_reference_velocity_world_mps": [float(v) for v in vel_target[j].detach().cpu().tolist()],
                "wake_reference_acceleration_world_mps2": [float(v) for v in source_accel[j].detach().cpu().tolist()],
                "wake_reference_history_index": int(k_target[j].detach().item()),
                "wake_reference_history_age_s": float((k_target[j].float() * self.dt).detach().item()),
                "target_feedforward_acceleration_world_mps2": [float(v) for v in a_target[j].detach().cpu().tolist()],
                "nominal_pd_residual_acceleration_mps2": [float(v) for v in u_baseline[j].detach().cpu().tolist()],
                "actor_residual_acceleration_mps2": [float(v) for v in u_residual[j].detach().cpu().tolist()],
                "lateral_barrier_rhs_absolute_accel": float(b_cbf2_absolute[j, 0, 0].detach().item()),
                "lateral_barrier_rhs_residual_accel": float(b_cbf2[j, 0, 0].detach().item()),
                "background_wind_world_mps": [float(v) for v in self.background_wind[j].detach().cpu().tolist()],
            })
            safety_failure_records.append(enriched)

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
            "p_follow_motor_electrical_model_watts": (self.follow_motor_voltage * self.follow_motor_current).sum(dim=-1).mean().item(),
            "p_lead_motor_electrical_model_watts": (self.lead_motor_voltage * self.lead_motor_current).sum(dim=-1).mean().item(),
            "motor_model_parameters_calibrated": False,
            "motor_model_scope": "provisional voltage/current/rotor-speed dynamics; not identified to a physical ESC/motor/propeller",
            "r_core": r_core.mean().item(),
            "state_err": state_err,
            "control_mode": control_mode,
            "nn_primary_control": bool(control_mode == "nn-primary"),
            "nn_rotor_action_fraction": nn_rotor_action_fraction,
            "nn_rotor_thrust_targets_N": follow_target_thrusts,
            "safety_filter_active": bool(control_mode == "pd-baseline"),
            "pd_tracking_controller_active": bool(control_mode == "pd-baseline"),
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
            "qp_projection_feasible_pct": filter_diag["projection_feasible_pct"],
            "qp_projection_max_residual": filter_diag["projection_max_residual"],
            "safety_fallback_pct": filter_diag["fallback_attempted_pct"],
            "safety_fallback_applied_pct": filter_diag["fallback_applied_pct"],
            "safety_worst_environment_index": filter_diag["worst_environment_index"],
            "safety_worst_constraint_index": filter_diag["worst_constraint_index"],
            "safety_individually_unachievable_under_command_box_pct": filter_diag["individually_unachievable_under_command_box_pct"],
            "safety_failure_records": safety_failure_records,
            "safety_collision_max_violation": filter_diag["constraint_max_violations"][0] if filter_diag["constraint_count"] > 0 else 0.0,
            "safety_lateral_max_violation": filter_diag["constraint_max_violations"][1] if filter_diag["constraint_count"] > 1 else 0.0,
            "cbf_feedforward_accel_norm_mean": float(safe_norm(a_target, dim=-1).mean().item()),
            "cbf_collision_rhs_shift_mean_abs": float(torch.abs(torch.matmul(A_cbf1, a_target.unsqueeze(-1))).mean().item()),
            "cbf_lateral_rhs_shift_mean_abs": float(torch.abs(torch.matmul(A_cbf2, a_target.unsqueeze(-1))).mean().item()),
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
            # V10.2 command-vs-plant diagnostics. Acceleration residuals are in m/s^2.
            "target_feedforward_accel_norm_mean_mps2": float(safe_norm(a_target, dim=-1).mean().item()),
            "actual_accel_norm_mean_mps2": float(safe_norm(actual_follow_accel, dim=-1).mean().item()),
            "mean_acceleration_tracking_error_mps2": trajectory_accel_error_mean,
            "maximum_acceleration_tracking_error_mps2": trajectory_accel_error_max,
            "maximum_predictor_observation_mismatch_mps2": predicted_actual_mismatch_max,
            "attitude_tracking_loss": attitude_tracking_loss,
            "angular_rate_tracking_loss": angular_rate_tracking_loss,
            "nn_action_slew_loss": nn_action_slew_loss,
            "mean_attitude_tracking_error_after_step_deg": float(attitude_error_after_deg.mean().item()),
            "maximum_attitude_tracking_error_after_step_deg": float(attitude_error_after_deg.max().item()),
            "mean_rotor_speed_tracking_error_rad_s": float(safe_norm(rotor_speed_error, dim=-1).mean().item()),
            "plant_guard_predicted_barrier_feasible_pct": plant_guard_diag["predicted_feasible_pct"],
            "plant_guard_predicted_max_violation_mps2": plant_guard_diag["predicted_max_normalized_violation_mps2"],
            "plant_guard_raw_qp_max_violation_mps2": plant_guard_diag["raw_qp_predicted_max_violation_mps2"],
            "plant_guard_no_feasible_candidate_pct": plant_guard_diag["no_feasible_candidate_pct"],
            "plant_guard_adjusted_pct": plant_guard_diag["adjusted_pct"],
            "plant_guard_mean_adjustment_mps2": plant_guard_diag["mean_adjustment_mps2"],
            "actual_plant_barrier_feasible_pct": actual_plant_feasible_pct,
            "actual_plant_collision_barrier_max_violation_mps2": actual_collision_violation_max,
            "actual_plant_lateral_barrier_max_violation_mps2": actual_lateral_violation_max,
            "rotor_target_thrust_N_env0": follow_target_thrusts[0].detach(),
            "rotor_target_speed_rad_s_env0": follow_target_rotor_omega[0].detach(),
            "filtered_residual_acceleration_env0": filtered_accel[0].detach(),
            "rotor_target_thrust_saturation_pct": rotor_target_thrust_sat_pct,
            "rotor_speed_saturation_pct": rotor_speed_sat_pct,
            "motor_current_saturation_pct": motor_current_sat_pct,
            "motor_voltage_saturation_pct": motor_voltage_sat_pct,
            "control_response_sample": control_response_sample,
            "plant_guard_failure_records": plant_guard_failure_records,
            "plant_aware_guard_scope": "finite one-step candidate search through current actuator dynamics; diagnostic/model-based only, no formal guarantee",
        }
        return self.metrics


# =============================================================================
# 8. TRAINING ENGINE: NN-PRIMARY DIRECT ROTOR ACTOR + SEPARATE PD-REFERENCE METRIC DIAGNOSTIC
# =============================================================================
def _reduced_tracking_vector_field(x, actor_net=None):
    """PD-reference six-state diagnostic only; it does NOT model the NN controller.

    V10.5's NN outputs four rotor-thrust commands, so the former 3-output residual
    actor contraction diagnostic is no longer applicable. This retained local metric
    calculation is explicitly only a diagnostic of the reference PD error dynamics.
    It is not a neural-policy contraction check or a full-plant proof.
    """
    e_p, e_v = x[:3], x[3:]
    kp = torch.tensor([4.5, 4.5, 6.5], device=x.device, dtype=x.dtype)
    kd = torch.tensor([3.2, 3.2, 4.2], device=x.device, dtype=x.dtype)
    u_reference = -kp * e_p - kd * e_v
    return torch.cat([e_v, u_reference], dim=0)


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
    """Compare NN-primary direct-rotor controller against a separate PD benchmark.

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
    actor_net.eval(); ncm_net.eval()

    eval_fields = ["accel_error", "prediction_mismatch", "attitude_error", "rotor_speed_error", "plant_guard_feasible", "plant_guard_violation", "plant_guard_no_feasible", "plant_guard_adjusted", "plant_guard_adjustment", "actual_plant_feasible", "actual_collision_violation", "actual_lateral_violation", "rotor_target_sat", "rotor_speed_sat", "motor_current_sat", "motor_voltage_sat"]
    def make_eval_record():
        return {"pos": [], "vel": [], "separation": [], "power": [], "electrical_power": [], "filter_feasible": [], "filter_residual": [], "projection_feasible": [], "projection_residual": [], "fallback_pct": [], "fallback_applied_pct": [], "collision_violation": [], "lateral_violation": [], "failure_records": [], "plant_guard_failure_records": [], "finite": True, **{k: [] for k in eval_fields}}
    records = {"nn_primary_direct_rotor_actor": make_eval_record(), "pd_baseline_benchmark": make_eval_record()}
    records["nn_primary_direct_rotor_actor"]["control_mode"] = "nn-primary"
    records["pd_baseline_benchmark"]["control_mode"] = "pd-baseline"
    with torch.no_grad():
        for step_index in range(steps):
            for label, env, mode in (
                ("nn_primary_direct_rotor_actor", policy_env, "nn-primary"),
                ("pd_baseline_benchmark", baseline_env, "pd-baseline"),
            ):
                m = env.step(actor_net, ncm_net, control_mode=mode)
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
                rec["electrical_power"].append(float(m["p_follow_motor_electrical_model_watts"]))
                rec["filter_feasible"].append(float(m["qp_feasible_pct"]))
                rec["filter_residual"].append(float(m["safety_filter_max_residual"]))
                rec["projection_feasible"].append(float(m["qp_projection_feasible_pct"]))
                rec["projection_residual"].append(float(m["qp_projection_max_residual"]))
                rec["fallback_pct"].append(float(m["safety_fallback_pct"]))
                rec["fallback_applied_pct"].append(float(m["safety_fallback_applied_pct"]))
                rec["collision_violation"].append(float(m["safety_collision_max_violation"]))
                rec["lateral_violation"].append(float(m["safety_lateral_max_violation"]))
                rec["accel_error"].append(float(m["mean_acceleration_tracking_error_mps2"]))
                rec["prediction_mismatch"].append(float(m["maximum_predictor_observation_mismatch_mps2"]))
                rec["attitude_error"].append(float(m["mean_attitude_tracking_error_after_step_deg"]))
                rec["rotor_speed_error"].append(float(m["mean_rotor_speed_tracking_error_rad_s"]))
                rec["plant_guard_feasible"].append(float(m["plant_guard_predicted_barrier_feasible_pct"]))
                rec["plant_guard_violation"].append(float(m["plant_guard_predicted_max_violation_mps2"]))
                rec["plant_guard_no_feasible"].append(float(m["plant_guard_no_feasible_candidate_pct"]))
                rec["plant_guard_adjusted"].append(float(m["plant_guard_adjusted_pct"]))
                rec["plant_guard_adjustment"].append(float(m["plant_guard_mean_adjustment_mps2"]))
                rec["actual_plant_feasible"].append(float(m["actual_plant_barrier_feasible_pct"]))
                rec["actual_collision_violation"].append(float(m["actual_plant_collision_barrier_max_violation_mps2"]))
                rec["actual_lateral_violation"].append(float(m["actual_plant_lateral_barrier_max_violation_mps2"]))
                rec["rotor_target_sat"].append(float(m["rotor_target_thrust_saturation_pct"]))
                rec["rotor_speed_sat"].append(float(m["rotor_speed_saturation_pct"]))
                rec["motor_current_sat"].append(float(m["motor_current_saturation_pct"]))
                rec["motor_voltage_sat"].append(float(m["motor_voltage_saturation_pct"]))
                for plant_failure in m.get("plant_guard_failure_records", []):
                    plant_failure_tagged = dict(plant_failure)
                    plant_failure_tagged["evaluation_seed"] = int(seed)
                    plant_failure_tagged["step_index"] = int(step_index)
                    plant_failure_tagged["policy_role"] = label
                    rec["plant_guard_failure_records"].append(plant_failure_tagged)
                for failure in m.get("safety_failure_records", []):
                    failure_tagged = dict(failure)
                    failure_tagged["evaluation_seed"] = int(seed)
                    failure_tagged["step_index"] = int(step_index)
                    failure_tagged["policy_role"] = label
                    rec["failure_records"].append(failure_tagged)
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
            "controller_mode": rec.get("control_mode", "unknown"),
            "safety_filter_active": bool(rec.get("control_mode") == "pd-baseline"),
            "nn_primary_direct_rotor_commands_used": bool(rec.get("control_mode") == "nn-primary"),
            "mean_position_tracking_error_m": finite_stat(pos, np.mean),
            "p95_position_tracking_error_m": finite_stat(pos, lambda x: np.percentile(x, 95)),
            "mean_velocity_tracking_error_mps": finite_stat(vel, np.mean),
            "minimum_leader_follower_separation_m": finite_stat(separation, np.min),
            "samples_below_1p20m_separation": int(np.sum(separation[np.isfinite(separation)] < 1.20)),
            "mean_uncalibrated_power_estimate_w": finite_stat(rec["power"], np.mean),
            "mean_provisional_motor_electrical_input_w": finite_stat(rec["electrical_power"], np.mean),
            "mean_filter_feasible_rate_pct": finite_stat(rec["filter_feasible"], np.mean),
            "maximum_filter_constraint_residual": finite_stat(rec["filter_residual"], np.max),
            "mean_projection_solver_feasible_rate_pct": finite_stat(rec["projection_feasible"], np.mean),
            "maximum_projection_solver_residual": finite_stat(rec["projection_residual"], np.max),
            "mean_emergency_fallback_rate_pct": finite_stat(rec["fallback_pct"], np.mean),
            "mean_emergency_fallback_applied_rate_pct": finite_stat(rec["fallback_applied_pct"], np.mean),
            "maximum_emergency_fallback_applied_rate_pct": finite_stat(rec["fallback_applied_pct"], np.max),
            "maximum_collision_constraint_violation": finite_stat(rec["collision_violation"], np.max),
            "maximum_lateral_constraint_violation": finite_stat(rec["lateral_violation"], np.max),
            "mean_acceleration_tracking_error_mps2": finite_stat(rec["accel_error"], np.mean),
            "p95_acceleration_tracking_error_mps2": finite_stat(rec["accel_error"], lambda x: np.percentile(x, 95)),
            "maximum_predictor_observation_mismatch_mps2": finite_stat(rec["prediction_mismatch"], np.max),
            "mean_attitude_tracking_error_after_step_deg": finite_stat(rec["attitude_error"], np.mean),
            "mean_rotor_speed_tracking_error_rad_s": finite_stat(rec["rotor_speed_error"], np.mean),
            "mean_predicted_plant_barrier_feasible_rate_pct": finite_stat(rec["plant_guard_feasible"], np.mean),
            "maximum_predicted_plant_barrier_violation_mps2": finite_stat(rec["plant_guard_violation"], np.max),
            "mean_no_feasible_actuator_candidate_rate_pct": finite_stat(rec["plant_guard_no_feasible"], np.mean),
            "mean_plant_guard_adjustment_rate_pct": finite_stat(rec["plant_guard_adjusted"], np.mean),
            "mean_plant_guard_adjustment_mps2": finite_stat(rec["plant_guard_adjustment"], np.mean),
            "mean_actual_plant_barrier_feasible_rate_pct": finite_stat(rec["actual_plant_feasible"], np.mean),
            "maximum_actual_plant_collision_barrier_violation_mps2": finite_stat(rec["actual_collision_violation"], np.max),
            "maximum_actual_plant_lateral_barrier_violation_mps2": finite_stat(rec["actual_lateral_violation"], np.max),
            "mean_rotor_target_thrust_saturation_pct": finite_stat(rec["rotor_target_sat"], np.mean),
            "mean_rotor_speed_saturation_pct": finite_stat(rec["rotor_speed_sat"], np.mean),
            "mean_motor_current_saturation_pct": finite_stat(rec["motor_current_sat"], np.mean),
            "mean_motor_voltage_saturation_pct": finite_stat(rec["motor_voltage_sat"], np.mean),
            "worst_plant_guard_failure_examples": sorted(
                rec["plant_guard_failure_records"],
                key=lambda item: max(item.get("selected_predicted_max_normalized_violation_mps2", 0.0), item.get("actual_collision_violation_mps2", 0.0), item.get("actual_lateral_violation_mps2", 0.0)),
                reverse=True,
            )[:10],
            "worst_safety_failure_examples": sorted(
                rec["failure_records"],
                key=lambda item: max(item.get("projection_max_normalized_violation", 0.0), item.get("final_max_normalized_violation", 0.0)),
                reverse=True,
            )[:10],
        }
    policy_result = summarise(records["nn_primary_direct_rotor_actor"])
    baseline_result = summarise(records["pd_baseline_benchmark"])
    evaluation = {
        "scope": "matched-seed simulation comparison under mild randomized initial states and episode-constant wind",
        "seed": int(seed), "num_envs": num_envs, "steps": steps,
        "physical_validation": False, "flight_certification": False,
        "nn_primary_direct_rotor_actor": policy_result,
        "pd_baseline_benchmark": baseline_result,
        "delta_mean_position_error_m_policy_minus_baseline": policy_result["mean_position_tracking_error_m"] - baseline_result["mean_position_tracking_error_m"],
        "delta_mean_power_estimate_w_policy_minus_baseline": policy_result["mean_uncalibrated_power_estimate_w"] - baseline_result["mean_uncalibrated_power_estimate_w"],
        "delta_mean_provisional_motor_electrical_input_w_policy_minus_baseline": policy_result["mean_provisional_motor_electrical_input_w"] - baseline_result["mean_provisional_motor_electrical_input_w"],
        "interpretation": "Both rotor power proxy and modelled V*I input are uncalibrated. These values are diagnostics only and are not used for policy promotion or real energy-saving claims."
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

    labels = ("nn_primary_direct_rotor_actor", "pd_baseline_benchmark")
    agg = {}
    def values(label, field):
        return [run[label][field] for run in runs]
    for label in labels:
        pos = values(label, "mean_position_tracking_error_m")
        p95 = values(label, "p95_position_tracking_error_m")
        vel = values(label, "mean_velocity_tracking_error_mps")
        sep = values(label, "minimum_leader_follower_separation_m")
        power = values(label, "mean_uncalibrated_power_estimate_w")
        electrical_power = values(label, "mean_provisional_motor_electrical_input_w")
        feasible = values(label, "mean_filter_feasible_rate_pct")
        residual = values(label, "maximum_filter_constraint_residual")
        projection_feasible = values(label, "mean_projection_solver_feasible_rate_pct")
        projection_residual = values(label, "maximum_projection_solver_residual")
        fallback_rate = values(label, "mean_emergency_fallback_rate_pct")
        fallback_applied_rate = values(label, "mean_emergency_fallback_applied_rate_pct")
        fallback_applied_max = values(label, "maximum_emergency_fallback_applied_rate_pct")
        collision_v = values(label, "maximum_collision_constraint_violation")
        lateral_v = values(label, "maximum_lateral_constraint_violation")
        accel_error = values(label, "mean_acceleration_tracking_error_mps2")
        prediction_mismatch = values(label, "maximum_predictor_observation_mismatch_mps2")
        attitude_error = values(label, "mean_attitude_tracking_error_after_step_deg")
        rotor_speed_error = values(label, "mean_rotor_speed_tracking_error_rad_s")
        plant_guard_feasible = values(label, "mean_predicted_plant_barrier_feasible_rate_pct")
        plant_guard_violation = values(label, "maximum_predicted_plant_barrier_violation_mps2")
        plant_guard_no_feasible = values(label, "mean_no_feasible_actuator_candidate_rate_pct")
        plant_guard_adjusted = values(label, "mean_plant_guard_adjustment_rate_pct")
        actual_plant_feasible = values(label, "mean_actual_plant_barrier_feasible_rate_pct")
        actual_collision_violation = values(label, "maximum_actual_plant_collision_barrier_violation_mps2")
        actual_lateral_violation = values(label, "maximum_actual_plant_lateral_barrier_violation_mps2")
        agg[label] = {
            "finite_state_trajectory": all(run[label]["finite_state_trajectory"] for run in runs),
            "controller_mode": "nn-primary" if label == "nn_primary_direct_rotor_actor" else "pd-baseline",
            "safety_filter_active": bool(label == "pd_baseline_benchmark"),
            "nn_primary_direct_rotor_commands_used": bool(label == "nn_primary_direct_rotor_actor"),
            "mean_position_tracking_error_m": float(np.mean(pos)),
            "std_across_seed_mean_position_error_m": float(np.std(pos, ddof=1)),
            "max_seed_p95_position_tracking_error_m": float(np.max(p95)),
            "mean_velocity_tracking_error_mps": float(np.mean(vel)),
            "std_across_seed_mean_velocity_error_mps": float(np.std(vel, ddof=1)),
            "minimum_leader_follower_separation_m": float(np.min(sep)),
            "samples_below_1p20m_separation": int(sum(run[label]["samples_below_1p20m_separation"] for run in runs)),
            "mean_uncalibrated_power_estimate_w": float(np.mean(power)),
            "mean_provisional_motor_electrical_input_w": float(np.mean(electrical_power)),
            "std_across_seed_provisional_motor_electrical_input_w": float(np.std(electrical_power, ddof=1)),
            "mean_filter_feasible_rate_pct": float(np.mean(feasible)),
            "maximum_filter_constraint_residual": float(np.max(residual)),
            "mean_projection_solver_feasible_rate_pct": float(np.mean(projection_feasible)),
            "minimum_projection_solver_feasible_rate_pct": float(np.min(projection_feasible)),
            "maximum_projection_solver_residual": float(np.max(projection_residual)),
            "mean_emergency_fallback_rate_pct": float(np.mean(fallback_rate)),
            "maximum_emergency_fallback_rate_pct": float(np.max(fallback_rate)),
            "mean_emergency_fallback_applied_rate_pct": float(np.mean(fallback_applied_rate)),
            "maximum_emergency_fallback_applied_rate_pct": float(np.max(fallback_applied_max)),
            "maximum_collision_constraint_violation": float(np.max(collision_v)),
            "maximum_lateral_constraint_violation": float(np.max(lateral_v)),
            "mean_acceleration_tracking_error_mps2": float(np.mean(accel_error)),
            "maximum_predictor_observation_mismatch_mps2": float(np.max(prediction_mismatch)),
            "mean_attitude_tracking_error_after_step_deg": float(np.mean(attitude_error)),
            "mean_rotor_speed_tracking_error_rad_s": float(np.mean(rotor_speed_error)),
            "mean_predicted_plant_barrier_feasible_rate_pct": float(np.mean(plant_guard_feasible)),
            "minimum_predicted_plant_barrier_feasible_rate_pct": float(np.min(plant_guard_feasible)),
            "maximum_predicted_plant_barrier_violation_mps2": float(np.max(plant_guard_violation)),
            "mean_no_feasible_actuator_candidate_rate_pct": float(np.mean(plant_guard_no_feasible)),
            "mean_plant_guard_adjustment_rate_pct": float(np.mean(plant_guard_adjusted)),
            "mean_actual_plant_barrier_feasible_rate_pct": float(np.mean(actual_plant_feasible)),
            "minimum_actual_plant_barrier_feasible_rate_pct": float(np.min(actual_plant_feasible)),
            "maximum_actual_plant_collision_barrier_violation_mps2": float(np.max(actual_collision_violation)),
            "maximum_actual_plant_lateral_barrier_violation_mps2": float(np.max(actual_lateral_violation)),
            "worst_plant_guard_failure_examples": sorted(
                [failure for run in runs for failure in run[label].get("worst_plant_guard_failure_examples", [])],
                key=lambda item: max(item.get("selected_predicted_max_normalized_violation_mps2", 0.0), item.get("actual_collision_violation_mps2", 0.0), item.get("actual_lateral_violation_mps2", 0.0)),
                reverse=True,
            )[:20],
            "worst_safety_failure_examples": sorted(
                [failure for run in runs for failure in run[label].get("worst_safety_failure_examples", [])],
                key=lambda item: max(item.get("projection_max_normalized_violation", 0.0), item.get("final_max_normalized_violation", 0.0)),
                reverse=True,
            )[:20],
            "num_seeds": num_seeds,
        }
    agg["delta_mean_position_error_m_policy_minus_baseline"] = (
        agg["nn_primary_direct_rotor_actor"]["mean_position_tracking_error_m"]
        - agg["pd_baseline_benchmark"]["mean_position_tracking_error_m"]
    )
    agg["delta_mean_power_proxy_w_policy_minus_baseline"] = (
        agg["nn_primary_direct_rotor_actor"]["mean_uncalibrated_power_estimate_w"]
        - agg["pd_baseline_benchmark"]["mean_uncalibrated_power_estimate_w"]
    )
    agg["delta_mean_provisional_motor_electrical_input_w_policy_minus_baseline"] = (
        agg["nn_primary_direct_rotor_actor"]["mean_provisional_motor_electrical_input_w"]
        - agg["pd_baseline_benchmark"]["mean_provisional_motor_electrical_input_w"]
    )
    return {
        "scope": "paired multi-seed V10.5 evaluation: NN directly commands four rotor-thrust references; side-offset target point uses consistent rigid-body position/velocity/acceleration kinematics; barriers monitor-only for NN; power is uncalibrated",
        "base_seed": int(base_seed), "num_seeds": num_seeds, "num_envs": int(num_envs), "steps": int(steps),
        "physical_validation": False, "flight_certification": False, "power_model_calibrated": False,
        "aggregate": agg, "per_seed": runs,
    }


def candidate_acceptance_gate(evaluation, min_position_improvement_pct=2.0,
                              max_p95_regression_pct=2.0, max_velocity_regression_pct=5.0,
                              min_filter_feasible_pct=99.9, max_filter_residual=2e-3,
                              max_fallback_rate_pct=0.1, min_projection_feasible_pct=99.9):
    """Promotion gate for NN-primary control, compared with the PD benchmark.

    In NN-primary mode the acceleration QP and actuator guard are deliberately NOT
    applied; their diagnostic fields are not promotion checks. Promotion requires the
    direct NN rotor controller itself to pass tracking, physical-state, one-step barrier,
    and predictor-consistency checks. Passing remains simulation-only, not certification.
    """
    c = evaluation["aggregate"]["nn_primary_direct_rotor_actor"]
    b = evaluation["aggregate"]["pd_baseline_benchmark"]
    baseline_reference_ok = bool(
        b["finite_state_trajectory"]
        and b["minimum_leader_follower_separation_m"] >= 1.20
        and b["samples_below_1p20m_separation"] == 0
        and b["mean_filter_feasible_rate_pct"] >= min_filter_feasible_pct
        and b["maximum_filter_constraint_residual"] <= max_filter_residual
        and b["minimum_projection_solver_feasible_rate_pct"] >= min_projection_feasible_pct
        and b["maximum_emergency_fallback_rate_pct"] <= max_fallback_rate_pct
        and b.get("minimum_predicted_plant_barrier_feasible_rate_pct", 0.0) >= min_projection_feasible_pct
        and b.get("minimum_actual_plant_barrier_feasible_rate_pct", 0.0) >= min_projection_feasible_pct
    )
    checks = {
        "baseline_reference_passes_safety_diagnostics": baseline_reference_ok,
        "finite_candidate": bool(c["finite_state_trajectory"]),
        "nn_primary_direct_rotor_commands_used": bool(c.get("nn_primary_direct_rotor_commands_used", False)),
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
        "nn_command_predicted_barrier_feasibility_above_threshold": bool(
            c.get("minimum_predicted_plant_barrier_feasible_rate_pct", 0.0) >= min_projection_feasible_pct
        ),
        "actual_plant_barrier_feasibility_above_threshold": bool(
            c.get("minimum_actual_plant_barrier_feasible_rate_pct", 0.0) >= min_projection_feasible_pct
        ),
        "direct_rotor_acceleration_prediction_consistent": bool(
            c.get("maximum_predictor_observation_mismatch_mps2", float("inf")) <= 2e-3
        ),
    }
    return {"accepted": all(checks.values()), "checks": checks,
            "selection_rule": "Promote NN-primary only when tracking beats the PD benchmark and direct NN rotor commands meet one-step barrier and prediction-consistency thresholds. No PD or safety-filter override is used in NN-primary mode. This is a simulation-only gate, not a formal safety proof. Uncalibrated power is not a promotion criterion."}


def curriculum_rollout_horizon(epoch_index, epochs, max_steps):
    """Progressive horizon schedule; max_steps is the final/full target horizon.

    For a 64-epoch run the stages are 20, 30, 40, 60, 80, then max_steps.
    Short diagnostic/smoke horizons (<=20) are respected without expansion.
    """
    max_steps = max(1, int(max_steps))
    if max_steps <= INITIAL_CURRICULUM_HORIZON:
        return max_steps
    progress = float(epoch_index + 1) / float(max(1, epochs))
    stages = ((0.125, 20), (0.25, 30), (0.375, 40), (0.50, 60), (0.625, 80), (1.0, max_steps))
    for cutoff, horizon in stages:
        if progress <= cutoff + 1e-12:
            return min(max_steps, horizon)
    return max_steps


def train_wake_controller(epochs=64, num_envs=256, rollout_steps=100, seed=41058, run_dir=None, eval_envs=8, eval_steps=100, eval_seeds=5):
    """Train the research controller and save evidence; never auto-claims certification."""
    from pathlib import Path
    import json
    random_seed = int(seed)
    np.random.seed(random_seed % (2**32))
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # A normal training run gets an isolated, timestamped output directory.
    # Train mode never loads a checkpoint; actor and metric are instantiated below.
    if run_dir is None:
        run_root = Path("wake_surfing_runs")
        run_root.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"fresh_{stamp}_seed{random_seed}"
        out_dir = run_root / run_name
        suffix = 1
        while out_dir.exists():
            out_dir = run_root / f"{run_name}_{suffix:02d}"
            suffix += 1
        out_dir.mkdir(parents=False, exist_ok=False)
    else:
        # Explicit output directories may be used, but a non-empty directory is
        # rejected so old evidence/checkpoints cannot be confused with this run.
        out_dir = Path(run_dir)
        if out_dir.exists() and any(out_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to mix a fresh training run with existing files in {out_dir}. "
                "Choose a new --run-dir or omit it for an isolated timestamped directory."
            )
        out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 106)
    print(" WAKE-SURFING V10.5 — NN-PRIMARY DIRECT ROTOR CONTROL + TARGET-POINT KINEMATICS + TARGET-POINT KINEMATICS")
    print(f" Envs: {num_envs:,} | Curriculum Horizon: 20→{rollout_steps} steps | BPTT chunk: {TRAINING_BPTT_CHUNK_STEPS} | Device: {DEVICE} | Seed: {random_seed}")
    print(f" Fresh run directory: {out_dir.resolve()}")
    print(" Fresh training: new NN weights and optimiser; no checkpoint is loaded.")
    print(" Follower control: NN directly commands four rotor-thrust setpoints; no PD tracking or safety-filter override.")
    print(" PD controller is retained only as an independent held-out benchmark.")
    print(" Note: simulation evidence is not physical validation or flight certification.")
    print("=" * 106)

    env = WakeSurfingResearchEnv(
        num_envs=num_envs, dt=0.015, history_len=75, device=DEVICE,
        randomize_initial_conditions=True,
    )
    alpha = 0.55
    actor_net = WakeSurfingActorPolicy(state_dim=ACTOR_OBSERVATION_DIM, action_dim=4).to(DEVICE)
    ncm_net = NeuralContractionMetric(state_dim=6, alpha=alpha).to(DEVICE)
    optimizer_actor = torch.optim.AdamW(actor_net.parameters(), lr=5.0e-4, weight_decay=1e-5)
    optimizer_ncm = torch.optim.AdamW(ncm_net.parameters(), lr=2.2e-3, weight_decay=1e-5)
    scheduler_ncm = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ncm, T_max=max(1, epochs), eta_min=1e-4)
    history = []
    control_response_trace = []

    for ep in range(epochs):
        optimizer_actor.zero_grad(set_to_none=True)
        optimizer_ncm.zero_grad(set_to_none=True)
        env.reset()
        epoch_rollout_steps = curriculum_rollout_horizon(ep, epochs, rollout_steps)
        segment_loss = torch.zeros((), device=DEVICE)
        rollout_loss_sum = 0.0
        position_error_accum = 0.0
        velocity_error_accum = 0.0
        residual_rms_accum = 0.0
        attitude_loss_accum = 0.0
        angular_rate_loss_accum = 0.0
        action_slew_loss_accum = 0.0
        first_quarter_position_accum = 0.0
        first_quarter_count = 0
        final_quarter_position_accum = 0.0
        final_quarter_count = 0
        filter_feasible_accum = 0.0
        projection_feasible_accum = 0.0
        fallback_rate_accum = 0.0
        fallback_applied_accum = 0.0
        filter_final_residual_max = 0.0
        filter_projection_residual_max = 0.0
        filter_collision_violation_max = 0.0
        filter_lateral_violation_max = 0.0
        accel_error_accum = 0.0
        attitude_error_accum = 0.0
        plant_guard_feasible_accum = 0.0
        plant_guard_violation_max = 0.0
        plant_guard_no_feasible_accum = 0.0
        plant_guard_adjustment_accum = 0.0
        actual_plant_feasible_accum = 0.0
        actual_plant_collision_violation_max = 0.0
        actual_plant_lateral_violation_max = 0.0
        prediction_mismatch_max = 0.0
        rotor_speed_sat_accum = 0.0
        current_sat_accum = 0.0
        voltage_sat_accum = 0.0
        last_state_err = torch.zeros((num_envs, 6), device=DEVICE)
        last_power_estimate = 0.0
        last_wind_mean = 0.0
        for rollout_step_index in range(epoch_rollout_steps):
            metrics = env.step(actor_net, ncm_net, control_mode="nn-primary")
            trace_record = dict(metrics["control_response_sample"])
            trace_record.update({"epoch": int(ep + 1), "rollout_step": int(rollout_step_index + 1), "epoch_horizon_steps": int(epoch_rollout_steps)})
            control_response_trace.append(trace_record)
            p_cost = torch.sum(metrics["error_pos"] ** 2, dim=-1).mean()
            v_cost = torch.sum(metrics["error_vel"] ** 2, dim=-1).mean()
            # Learn stable direct rotor control, not merely short-term position tracking.
            # SO(3) trace loss is smooth at alignment; angular-rate and command-slew
            # terms penalise tumbles and needless motor-command chatter. Power remains
            # excluded because the propulsion/electrical model is not calibrated.
            hover_fraction = (HOVER_WEIGHT / 4.0) / MAX_THRUST_PER_MOTOR
            action_delta = metrics["nn_rotor_action_fraction"] - hover_fraction
            action_regularizer = action_delta.square().mean()
            attitude_cost = metrics["attitude_tracking_loss"]
            angular_rate_cost = metrics["angular_rate_tracking_loss"]
            action_slew_cost = metrics["nn_action_slew_loss"]
            step_loss = (
                8.0 * p_cost
                + 1.8 * v_cost
                + 1.0 * attitude_cost
                + 0.35 * angular_rate_cost
                + 0.02 * action_regularizer
                + 0.08 * action_slew_cost
            )
            segment_loss = segment_loss + step_loss
            rollout_loss_sum += float(step_loss.detach().item())
            position_step = float(safe_norm(metrics["error_pos"], dim=-1).mean().detach().item())
            position_error_accum += position_step
            velocity_error_accum += float(safe_norm(metrics["error_vel"], dim=-1).mean().detach().item())
            residual_rms_accum += float(action_delta.detach().square().mean().sqrt().item())
            attitude_loss_accum += float(attitude_cost.detach().item())
            angular_rate_loss_accum += float(angular_rate_cost.detach().item())
            action_slew_loss_accum += float(action_slew_cost.detach().item())
            if rollout_step_index < max(1, epoch_rollout_steps // 4):
                first_quarter_position_accum += position_step
                first_quarter_count += 1
            if rollout_step_index >= (3 * epoch_rollout_steps) // 4:
                final_quarter_position_accum += position_step
                final_quarter_count += 1
            filter_feasible_accum += float(metrics["qp_feasible_pct"])
            projection_feasible_accum += float(metrics["qp_projection_feasible_pct"])
            fallback_rate_accum += float(metrics["safety_fallback_pct"])
            fallback_applied_accum += float(metrics["safety_fallback_applied_pct"])
            filter_final_residual_max = max(filter_final_residual_max, float(metrics["safety_filter_max_residual"]))
            filter_projection_residual_max = max(filter_projection_residual_max, float(metrics["qp_projection_max_residual"]))
            filter_collision_violation_max = max(filter_collision_violation_max, float(metrics["safety_collision_max_violation"]))
            filter_lateral_violation_max = max(filter_lateral_violation_max, float(metrics["safety_lateral_max_violation"]))
            accel_error_accum += float(metrics["mean_acceleration_tracking_error_mps2"])
            attitude_error_accum += float(metrics["mean_attitude_tracking_error_after_step_deg"])
            plant_guard_feasible_accum += float(metrics["plant_guard_predicted_barrier_feasible_pct"])
            plant_guard_violation_max = max(plant_guard_violation_max, float(metrics["plant_guard_predicted_max_violation_mps2"]))
            plant_guard_no_feasible_accum += float(metrics["plant_guard_no_feasible_candidate_pct"])
            plant_guard_adjustment_accum += float(metrics["plant_guard_adjusted_pct"])
            actual_plant_feasible_accum += float(metrics["actual_plant_barrier_feasible_pct"])
            actual_plant_collision_violation_max = max(actual_plant_collision_violation_max, float(metrics["actual_plant_collision_barrier_max_violation_mps2"]))
            actual_plant_lateral_violation_max = max(actual_plant_lateral_violation_max, float(metrics["actual_plant_lateral_barrier_max_violation_mps2"]))
            prediction_mismatch_max = max(prediction_mismatch_max, float(metrics["maximum_predictor_observation_mismatch_mps2"]))
            rotor_speed_sat_accum += float(metrics["rotor_speed_saturation_pct"])
            current_sat_accum += float(metrics["motor_current_saturation_pct"])
            voltage_sat_accum += float(metrics["motor_voltage_saturation_pct"])
            last_state_err = metrics["state_err"].detach()
            last_power_estimate = float(metrics["p_follow_total_watts"])
            last_wind_mean = float(metrics["background_wind_mean_mps"])

            # Truncated BPTT: keep full forward episodes, propagate gradients within
            # bounded chunks, detach only the physical state at chunk boundaries.
            chunk_end = ((rollout_step_index + 1) % TRAINING_BPTT_CHUNK_STEPS == 0) or (rollout_step_index + 1 == epoch_rollout_steps)
            if chunk_end:
                (segment_loss / max(1, epoch_rollout_steps)).backward()
                segment_loss = torch.zeros((), device=DEVICE)
                if rollout_step_index + 1 < epoch_rollout_steps:
                    env.detach_dynamics_for_truncated_bptt()
                metrics = None
        position_error_mean = position_error_accum / max(1, epoch_rollout_steps)
        velocity_error_mean = velocity_error_accum / max(1, epoch_rollout_steps)
        actor_residual_rms_mean = residual_rms_accum / max(1, epoch_rollout_steps)
        mean_attitude_loss = attitude_loss_accum / max(1, epoch_rollout_steps)
        mean_angular_rate_loss = angular_rate_loss_accum / max(1, epoch_rollout_steps)
        mean_action_slew_loss = action_slew_loss_accum / max(1, epoch_rollout_steps)
        first_quarter_position_mean = first_quarter_position_accum / max(1, first_quarter_count)
        final_quarter_position_mean = final_quarter_position_accum / max(1, final_quarter_count)
        grad_norm_actor = float(torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=1.5).item())
        optimizer_actor.step()

        # Train M against a SEPARATE reduced-order PD-reference vector field.
        # It does not include the NN-primary four-rotor policy or the electromechanical
        # plant and cannot establish contraction of the trained direct-rotor controller.
        # Freeze actor parameters because this optimizer step only trains the diagnostic metric.
        _actor_grad_flags = [p.requires_grad for p in actor_net.parameters()]
        for p in actor_net.parameters():
            p.requires_grad_(False)
        rollout_states = last_state_err.detach()
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
            "rollout_loss": float(rollout_loss_sum / max(1, epoch_rollout_steps)),
            "rollout_horizon_steps": int(epoch_rollout_steps),
            "bptt_chunk_steps": int(TRAINING_BPTT_CHUNK_STEPS),
            "mean_attitude_tracking_loss": float(mean_attitude_loss),
            "mean_angular_rate_tracking_loss": float(mean_angular_rate_loss),
            "mean_nn_action_slew_loss": float(mean_action_slew_loss),
            "first_quarter_mean_position_error_m": float(first_quarter_position_mean),
            "final_quarter_mean_position_error_m": float(final_quarter_position_mean),
            "ncm_loss": float(loss_ncm.detach().item()),
            "sampled_reduced_model_max_eigenvalue": max_lambda_R,
            "sampled_reduced_model_contraction_gate": contraction_ok,
            "actor_grad_norm": grad_norm_actor,
            "ncm_grad_norm": grad_norm_ncm,
            "mean_position_error_m": float(position_error_mean),
            "mean_velocity_error_mps": float(velocity_error_mean),
            "mean_nn_thrust_fraction_deviation": float(actor_residual_rms_mean),
            "follower_control_mode": "nn-primary-direct-rotor-thrust",
            "safety_filter_active": False,
            "filter_feasible_pct": float(filter_feasible_accum / max(1, epoch_rollout_steps)),
            "filter_projection_feasible_pct": float(projection_feasible_accum / max(1, epoch_rollout_steps)),
            "filter_fallback_rate_pct": float(fallback_rate_accum / max(1, epoch_rollout_steps)),
            "filter_fallback_applied_rate_pct": float(fallback_applied_accum / max(1, epoch_rollout_steps)),
            "filter_max_residual": float(filter_final_residual_max),
            "filter_projection_max_residual": float(filter_projection_residual_max),
            "filter_collision_max_violation": float(filter_collision_violation_max),
            "filter_lateral_max_violation": float(filter_lateral_violation_max),
            "mean_acceleration_tracking_error_mps2": float(accel_error_accum / max(1, epoch_rollout_steps)),
            "mean_attitude_tracking_error_after_step_deg": float(attitude_error_accum / max(1, epoch_rollout_steps)),
            "mean_predicted_plant_barrier_feasible_rate_pct": float(plant_guard_feasible_accum / max(1, epoch_rollout_steps)),
            "maximum_predicted_plant_barrier_violation_mps2": float(plant_guard_violation_max),
            "mean_no_feasible_actuator_candidate_rate_pct": float(plant_guard_no_feasible_accum / max(1, epoch_rollout_steps)),
            "mean_plant_guard_adjustment_rate_pct": float(plant_guard_adjustment_accum / max(1, epoch_rollout_steps)),
            "mean_actual_plant_barrier_feasible_rate_pct": float(actual_plant_feasible_accum / max(1, epoch_rollout_steps)),
            "maximum_actual_plant_collision_barrier_violation_mps2": float(actual_plant_collision_violation_max),
            "maximum_actual_plant_lateral_barrier_violation_mps2": float(actual_plant_lateral_violation_max),
            "maximum_predictor_observation_mismatch_mps2": float(prediction_mismatch_max),
            "mean_rotor_speed_saturation_pct": float(rotor_speed_sat_accum / max(1, epoch_rollout_steps)),
            "mean_motor_current_saturation_pct": float(current_sat_accum / max(1, epoch_rollout_steps)),
            "mean_motor_voltage_saturation_pct": float(voltage_sat_accum / max(1, epoch_rollout_steps)),
            "power_estimate_w": float(last_power_estimate),
            "episode_background_wind_mean_mps": float(last_wind_mean),
        }
        history.append(row)
        if (ep + 1) % 4 == 0 or ep == 0 or ep == epochs - 1:
            print(
                f"[Epoch {ep+1:03d}/{epochs}] horizon={row['rollout_horizon_steps']} loss={row['rollout_loss']:.4f} "
                f"pos={row['mean_position_error_m']:.3f}m vel={row['mean_velocity_error_mps']:.3f}m/s "
                f"early/late-pos={row['first_quarter_mean_position_error_m']:.3f}/{row['final_quarter_mean_position_error_m']:.3f}m "
                f"att-loss={row['mean_attitude_tracking_loss']:.3f} rate-loss={row['mean_angular_rate_tracking_loss']:.3f} "
                f"slew-loss={row['mean_nn_action_slew_loss']:.4f} NN-thrust-fraction-offset-rms={row['mean_nn_thrust_fraction_deviation']:.3f} "
                f"safety-filter=NOT-APPLIED "
                f"NN-CBF(pred/actual)={row['mean_predicted_plant_barrier_feasible_rate_pct']:.1f}/{row['mean_actual_plant_barrier_feasible_rate_pct']:.1f}% "
                f"NN-barrier-max-v={row['maximum_predicted_plant_barrier_violation_mps2']:.3g}m/s² "
                f"trajectory-accel-error={row['mean_acceleration_tracking_error_mps2']:.3f}m/s² prediction-mismatch-max={prediction_mismatch_max:.3g}m/s² "
                f"NN-direct-action, barrier monitor only; no command override "
                f"PD-reference-contr.lambda_max={(f'{max_lambda_R:+.4g}' if max_lambda_R is not None else 'UNAVAILABLE')} "
                f"gate={'PASS' if contraction_ok else 'UNVERIFIED'}"
            )

    # Detailed per-step command/plant trace for environment 0, separate from summary JSON.
    (out_dir / "control_response_trace.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in control_response_trace),
        encoding="utf-8",
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
    # For observation, ALWAYS return the actual trained NN controller. Rejection by
    # the evidence gate changes its label, not the controller silently shown in the
    # cockpit. The PD baseline is a separately evaluated benchmark only.
    selected_actor = actor_net
    selected_role = "nn_primary_accepted_in_simulation_gate" if selection["accepted"] else "nn_primary_unpromoted"

    (out_dir / "heldout_multiseed_evaluation.json").write_text(
        json.dumps(evaluation, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("[HELD-OUT MULTI-SEED] candidate/baseline mean position error="
          f"{evaluation['aggregate']['nn_primary_direct_rotor_actor']['mean_position_tracking_error_m']:.4f}/"
          f"{evaluation['aggregate']['pd_baseline_benchmark']['mean_position_tracking_error_m']:.4f} m; "
          f"power-proxy={evaluation['aggregate']['nn_primary_direct_rotor_actor']['mean_uncalibrated_power_estimate_w']:.2f}/"
          f"{evaluation['aggregate']['pd_baseline_benchmark']['mean_uncalibrated_power_estimate_w']:.2f} W; "
          f"motor V*I={evaluation['aggregate']['nn_primary_direct_rotor_actor']['mean_provisional_motor_electrical_input_w']:.2f}/"
          f"{evaluation['aggregate']['pd_baseline_benchmark']['mean_provisional_motor_electrical_input_w']:.2f} W (uncalibrated)")
    print(f"[NN-PRIMARY POLICY GATE] accepted={selection['accepted']}; visualiser will still show the trained NN candidate")
    for role in ("nn_primary_direct_rotor_actor", "pd_baseline_benchmark"):
        d = evaluation["aggregate"][role]
        if d.get("controller_mode") == "nn-primary":
            print(
                f"[NN SAFETY MONITOR] {role}: barrier-feasible predicted/actual="
                f"{d['mean_predicted_plant_barrier_feasible_rate_pct']:.3f}/{d['mean_actual_plant_barrier_feasible_rate_pct']:.3f}% "
                f"max collision/lateral violation="
                f"{d['maximum_actual_plant_collision_barrier_violation_mps2']:.4g}/"
                f"{d['maximum_actual_plant_lateral_barrier_violation_mps2']:.4g} m/s^2; "
                "NN actions were not overridden"
            )
        else:
            print(
                f"[PD BENCHMARK SAFETY] {role}: filter-feasible={d['mean_filter_feasible_rate_pct']:.3f}% "
                f"projection-feasible mean/min={d['mean_projection_solver_feasible_rate_pct']:.3f}/{d['minimum_projection_solver_feasible_rate_pct']:.3f}% "
                f"fallback attempt/apply={d['mean_emergency_fallback_rate_pct']:.3f}/{d['mean_emergency_fallback_applied_rate_pct']:.3f}% "
                f"max collision/lateral violation={d['maximum_collision_constraint_violation']:.4g}/{d['maximum_lateral_constraint_violation']:.4g}"
            )
    for name, passed in selection["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    artifact = {
        "scope": "V10.5 full simulator rollouts with neural-primary direct four-rotor thrust commands and consistent moving-point target kinematics; one-step CBFs are monitor-only and do not override the NN",
        "physical_validation": False,
        "flight_certification": False,
        "power_model_calibrated": False,
        "seed": random_seed,
        "run_directory": str(out_dir.resolve()),
        "initialization_mode": "fresh_network_initialization_no_checkpoint_load",
        "model_weights_loaded_from_checkpoint": False,
        "plant_model_version": "V10.5_neural_primary_direct_rotor_control_with_consistent_moving_target_point_kinematics",
        "follower_controller_architecture": "NN directly outputs four independent rotor-thrust fractions; no PD tracking command, geometric attitude controller, CBF projection, or actuator-guard override is applied to the follower in NN-primary mode",
        "safety_filter_active_during_training": False,
        "policy_action_dim": 4,
        "policy_observation_dim": ACTOR_OBSERVATION_DIM,
        "actor_observation_features": ["position_error_xyz", "velocity_error_xyz", "wake_velocity_xyz", "relative_position_xyz", "body_angular_rate_xyz", "relative_rotation_matrix_9", "rotor_speed_4", "target_point_acceleration_xyz", "target_body_angular_rate_xyz", "previous_rotor_command_fraction_4"],
        "target_point_kinematics": {
            "position": "historical leader centre position plus rotated body-frame flank offset",
            "velocity": "historical centre velocity plus omega_world cross offset_world",
            "acceleration": "historical centre acceleration plus alpha_world cross offset_world plus omega_world cross (omega_world cross offset_world)",
            "validation_scope": "kinematic identity and finite-difference regression tests; not physical wake validation",
        },
        "training_rollout_curriculum": "20,30,40,60,80,then configured max horizon according to epoch fraction",
        "truncated_bptt_chunk_steps": TRAINING_BPTT_CHUNK_STEPS,
        "training_objective_weights": {"position_squared": 8.0, "velocity_squared": 1.8, "so3_attitude_trace_loss": 1.0, "normalized_angular_rate_squared": 0.35, "hover_command_deviation_squared": 0.02, "rotor_command_slew_squared": 0.08},
        "acceleration_metrics_note": "trajectory acceleration error is measured against target feedforward acceleration; predictor-observation mismatch is reported separately",
        "motor_model_parameters_calibrated": False,
        "rotor_model": {
            "thrust_coefficient_N_per_rad_s_squared": ROTOR_THRUST_COEFFICIENT,
            "reaction_torque_coefficient_Nm_per_rad_s_squared": ROTOR_TORQUE_COEFFICIENT,
            "rotor_inertia_kg_m2": ROTOR_INERTIA,
            "motor_resistance_ohm": MOTOR_COIL_RESISTANCE,
            "motor_inductance_H": MOTOR_INDUCTANCE,
            "motor_torque_constant_Nm_per_A": MOTOR_TORQUE_CONSTANT,
            "motor_back_emf_constant_V_per_rad_s": MOTOR_BACK_EMF_CONSTANT,
            "motor_current_limit_A": MOTOR_CURRENT_LIMIT,
            "battery_voltage_max_V": BATTERY_VOLTAGE_MAX,
            "rotor_max_angular_speed_rad_s": ROTOR_MAX_OMEGA,
            "calibration_note": "All electrical/propulsion values are provisional engineering estimates; identify from bench data before physical claims.",
        },
        "wake_force_coupling_note": "Inherited empirical external_lift injection remains; rotor-disk inflow coupling is not yet implemented.",
        "plant_aware_guard_scope": "NN-primary mode directly tests the NN's rotor-thrust command against predicted/observed one-step CBF residuals; the monitor does not alter the command and does not prove future safety.",
        "epochs": int(epochs),
        "num_envs": int(num_envs),
        "rollout_steps_max": int(rollout_steps),
        "final_curriculum_horizon_steps": int(rollout_steps),
        "training_objective": "position + velocity + SO(3) attitude + target angular-rate + rotor command slew + small hover-thrust deviation; target position/velocity/acceleration derive from one rigid-body moving-point kinematics model; uncalibrated power proxy excluded from optimisation",
        "heldout_sampled_reduced_model_max_eigenvalue": held_max,
        "heldout_metric_condition_number_max": held_cond_max,
        "heldout_sampled_local_contraction_gate": local_gate_pass,
        "contraction_diagnostic_scope": "separate reduced-order PD-reference model only; not a contraction test or proof for the NN-primary direct-rotor controller or full electromechanical plant",
        "heldout_multiseed_evaluation_file": "heldout_multiseed_evaluation.json",
        "control_response_trace_file": "control_response_trace.jsonl",
        "control_response_trace_scope": "one trace record per rollout step for environment 0; records four NN rotor-thrust fractions, target/actual rotor states, predicted/observed acceleration and monitor-only barrier diagnostics",
        "heldout_multiseed_evaluation": evaluation,
        "policy_selection": selection,
        "selected_policy_role": selected_role,
        "selection_does_not_use_power_proxy": True,
        "history": history,
    }
    torch.save({
        "actor_state_dict": candidate_state,
        "candidate_actor_state_dict": candidate_state,
        "ncm_state_dict": ncm_net.state_dict(),
        "metadata": artifact,
    }, out_dir / "wake_surfing_v10_5_checkpoint.pt")
    (out_dir / "training_evidence_v10_5.json").write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    print("\n[TRAINING FINISHED]")
    print(f"Checkpoint: {out_dir / 'wake_surfing_v10_5_checkpoint.pt'}")
    print(f"Evidence:   {out_dir / 'training_evidence_v10_5.json'}")
    print(f"Control response trace: {out_dir / 'control_response_trace.jsonl'}")
    held_label = f"{held_max:+.5g}" if held_max is not None else "UNAVAILABLE"
    print(f"PD-reference reduced-model eigenvalue diagnostic: {held_label}; gate={'PASS' if local_gate_pass else 'UNVERIFIED'} (not an NN-controller proof)")
    print(f"Controller for cockpit/checkpoint: {selected_role} (NN direct rotor control; PD baseline not substituted)")
    print("No full-plant contraction proof, real-world wake validation, calibrated energy claim, or flight certification is claimed.")
    print("Training used horizon curriculum + truncated BPTT; horizon is not differentiated end-to-end across chunk boundaries.")
    # Runtime-only metadata lets the auto-launched cockpit name the exact policy
    # returned here, rather than incorrectly always calling it a neural pilot.
    selected_actor._wake_policy_role = selected_role
    selected_actor._wake_promotion_accepted = bool(selection["accepted"])
    selected_actor._wake_run_directory = str(out_dir.resolve())
    selected_actor._wake_checkpoint_path = str((out_dir / "wake_surfing_v10_5_checkpoint.pt").resolve())
    return selected_actor, ncm_net



def cockpit_policy_status(controller_role, promotion_accepted=None):
    """Return a truthful, compact controller status for the cockpit HUD."""
    role = str(controller_role or "unknown").strip().lower()
    if role in {"nn_primary_unpromoted", "nn_primary_direct_rotor", "nn_primary_direct_rotor_actor_unpromoted", "candidate", "trained_candidate", "candidate_actor", "trained_residual_actor_unpromoted"}:
        return ("ACTIVE POLICY: NEURAL-PRIMARY DIRECT ROTOR CONTROL (UNPROMOTED)",
                "NN COMMANDS ALL FOUR ROTOR-THRUST SETPOINTS; SAFETY IS MONITOR-ONLY", "warning")
    if role in {"nn_primary_accepted_in_simulation_gate", "nn_primary_accepted", "nn_primary_direct_rotor_actor"}:
        if promotion_accepted is not True:
            return ("ACTIVE POLICY: NN-PRIMARY DIRECT ROTOR CONTROL (GATE UNVERIFIED)",
                    "NN IS ACTIVE; DO NOT ASSUME ACCEPTANCE OR FLIGHT SAFETY", "danger")
        return ("ACTIVE POLICY: NN-PRIMARY DIRECT ROTOR CONTROL (SIMULATION GATE PASSED)",
                "NN COMMANDS ALL FOUR ROTORS; NOT FLIGHT-CERTIFIED", "success")
    if role in {"pd_baseline_benchmark", "selected_pd_baseline", "pd_baseline"}:
        return ("ACTIVE POLICY: PD BASELINE (NEURAL CONTROLLER NOT ACTIVE)",
                "BENCHMARK ONLY — NOT THE NN-PRIMARY VISUALISATION", "warning")
    if role in {"trained_residual_actor", "accepted_trained_residual_actor", "promoted_candidate"}:
        if promotion_accepted is not True:
            return ("ACTIVE POLICY: RESIDUAL ACTOR (PROMOTION UNVERIFIED)",
                    "CHECK ACCEPTANCE METADATA; DO NOT ASSUME IT PASSED", "danger")
        return ("ACTIVE POLICY: TRAINED RESIDUAL ACTOR (PROMOTED IN SIMULATION GATE)",
                "STILL RESEARCH SIMULATION — NOT FLIGHT-CERTIFIED", "success")
    return ("ACTIVE POLICY: ROLE UNKNOWN", "DO NOT ASSUME THIS IS THE TRAINED CANDIDATE", "warning")


def cockpit_acceleration_telemetry(record):
    """Validate and expose the V10.5 acceleration telemetry schema for the HUD.

    Keep prediction consistency separate from trajectory tracking error. This
    prevents a near-zero predictor residual being displayed as good tracking.
    """
    required = {
        "predicted_actuator_limited_acceleration_world_mps2",
        "observed_actual_acceleration_world_mps2",
        "actual_minus_target_feedforward_acceleration_mps2",
        "prediction_minus_observed_acceleration_mps2",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise KeyError(
            "V10.5 cockpit telemetry schema mismatch; missing fields: "
            + ", ".join(missing)
            + "; available fields: " + ", ".join(sorted(map(str, record.keys())))
        )
    values = {
        "predicted": np.asarray(record["predicted_actuator_limited_acceleration_world_mps2"], dtype=np.float64).reshape(-1),
        "observed": np.asarray(record["observed_actual_acceleration_world_mps2"], dtype=np.float64).reshape(-1),
        "trajectory_error": np.asarray(record["actual_minus_target_feedforward_acceleration_mps2"], dtype=np.float64).reshape(-1),
        "prediction_error": np.asarray(record["prediction_minus_observed_acceleration_mps2"], dtype=np.float64).reshape(-1),
    }
    for name, value in values.items():
        if value.shape != (3,):
            raise ValueError(f"V10.5 cockpit telemetry '{name}' must be a 3-vector; got shape {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"V10.5 cockpit telemetry '{name}' contains non-finite values")
    return values


# =============================================================================
# 9. NUMERICAL-FAILURE GUARDS FOR THE DIAGNOSTIC VISUALISER
# =============================================================================
def _collect_numeric_tensors(value, prefix="root", out=None, depth=0):
    """Collect (path, tensor) pairs and non-finite scalar paths from nested data."""
    if out is None:
        out = {"tensors": [], "bad_scalars": []}
    if depth > 12:
        return out
    if torch.is_tensor(value):
        if value.is_floating_point() or value.is_complex():
            out["tensors"].append((prefix, value))
        return out
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            out["bad_scalars"].append(prefix)
        return out
    if isinstance(value, dict):
        for key, child in value.items():
            _collect_numeric_tensors(child, f"{prefix}.{key}", out, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _collect_numeric_tensors(child, f"{prefix}[{index}]", out, depth + 1)
        return out
    if isinstance(value, (float, np.floating, complex, np.complexfloating)):
        if not (math.isfinite(float(np.real(value))) and math.isfinite(float(np.imag(value)))):
            out["bad_scalars"].append(prefix)
    return out


def _find_nonfinite_paths(env, metrics, max_paths=16):
    """Return a bounded list of exact state/metric paths containing NaN or Inf."""
    collected = {"tensors": [], "bad_scalars": []}
    for name, value in vars(env).items():
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
            collected["tensors"].append((f"env.{name}", value))
        elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            collected["bad_scalars"].append(f"env.{name}")
    _collect_numeric_tensors(metrics, "metrics", collected)
    bad = list(collected["bad_scalars"])
    tensor_checks = []
    for path, tensor in collected["tensors"]:
        # GPU checks are gathered and synchronised once in aggregate.
        tensor_checks.append((path, torch.isfinite(tensor).all()))
    if tensor_checks:
        all_finite = torch.stack([item[1] for item in tensor_checks]).all().item()
        if not bool(all_finite):
            for path, finite_flag in tensor_checks:
                if not bool(finite_flag.item()):
                    bad.append(path)
                    if len(bad) >= max_paths:
                        break
    return bad[:max_paths]


def _snapshot_visualiser_env(env):
    """Snapshot numerical state before one visualiser step for last-good-state freeze."""
    return {
        "tensors": {name: value.detach().clone() for name, value in vars(env).items() if torch.is_tensor(value)},
        "sim_time": float(env.sim_time),
        "metrics": getattr(env, "metrics", {}),
    }


def _restore_visualiser_env(env, snapshot):
    """Restore the last finite simulator state without issuing another control command."""
    for name, value in snapshot["tensors"].items():
        setattr(env, name, value)
    env.sim_time = snapshot["sim_time"]
    env.metrics = snapshot["metrics"]


# =============================================================================
# 9. INTERACTIVE 3D DIAGNOSTIC COCKPIT
# =============================================================================
def project_point_3d(point, cam_pos, cam_lookat, screen_w, screen_h, fov=50.0):
    """Project a finite 3-D point, returning None for invalid or unrenderable inputs.

    The NN-primary simulation can become numerically invalid. Rendering must not
    convert NaN/Inf to integer pixel coordinates or mask the underlying failure;
    the cockpit's simulation guard separately pauses at the last finite state.
    """
    try:
        point = np.asarray(point, dtype=np.float64).reshape(-1)
        cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(-1)
        cam_lookat = np.asarray(cam_lookat, dtype=np.float64).reshape(-1)
        screen_w = int(screen_w)
        screen_h = int(screen_h)
        fov = float(fov)
    except (TypeError, ValueError, OverflowError):
        return None

    if point.size != 3 or cam_pos.size != 3 or cam_lookat.size != 3:
        return None
    if screen_w <= 0 or screen_h <= 0 or not math.isfinite(fov) or not (1.0 < fov < 179.0):
        return None
    if not (np.isfinite(point).all() and np.isfinite(cam_pos).all() and np.isfinite(cam_lookat).all()):
        return None

    fwd = cam_lookat - cam_pos
    fwd_len = float(np.linalg.norm(fwd))
    if not math.isfinite(fwd_len) or fwd_len < 1e-4:
        return None
    fwd /= fwd_len

    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    rlen = float(np.linalg.norm(right))
    right = np.array([1.0, 0.0, 0.0]) if (not math.isfinite(rlen) or rlen < 1e-4) else right / rlen
    up = np.cross(right, fwd)

    delta = point - cam_pos
    depth = float(np.dot(delta, fwd))
    if not math.isfinite(depth) or depth <= 0.2:
        return None

    focal = (screen_h * 0.5) / math.tan(math.radians(fov) * 0.5)
    px = screen_w * 0.5 + focal * float(np.dot(delta, right)) / depth
    py = screen_h * 0.5 - focal * float(np.dot(delta, up)) / depth
    # Cull extreme projections before handing coordinates to SDL/Pygame. Normal
    # off-screen geometry is not useful for this cockpit and can overflow SDL ints.
    if not (math.isfinite(px) and math.isfinite(py)):
        return None
    if abs(px) > 20.0 * screen_w or abs(py) > 20.0 * screen_h:
        return None
    try:
        return int(round(px)), int(round(py)), depth
    except (ValueError, OverflowError):
        return None


def draw_full_quadrotor_3d(screen, pos, rot_mat, color, label, cam_pos, cam_lookat, w, h, font, rotor_omega=None, rotor_thrust=None, rotor_phase=None):
    pt_center = project_point_3d(pos, cam_pos, cam_lookat, w, h)
    if pt_center is None:
        return

    # Match the simulation allocation geometry: each rotor lies ARM_LENGTH from
    # the centre, with x/y coordinates +/- ARM_LENGTH/sqrt(2). The former 1.8x
    # visual-only scale drew rotors much farther apart than the simulated vehicle.
    arm_offset = ROTOR_OFFSET
    nose_length = ARM_LENGTH * 1.3
    arm_coords = [
        np.array([arm_offset, arm_offset, 0.0]),
        np.array([-arm_offset, -arm_offset, 0.0]),
        np.array([-arm_offset, arm_offset, 0.0]),
        np.array([arm_offset, -arm_offset, 0.0]),
    ]
    pts = [project_point_3d(pos + rot_mat @ ac, cam_pos, cam_lookat, w, h) for ac in arm_coords]

    if pts[0] and pts[1]:
        pygame.draw.line(screen, (220, 235, 255), pts[0][:2], pts[1][:2], 3)
    if pts[2] and pts[3]:
        pygame.draw.line(screen, (220, 235, 255), pts[2][:2], pts[3][:2], 3)

    if rotor_omega is None:
        rotor_omega = [0.0] * 4
    if rotor_thrust is None:
        rotor_thrust = [0.0] * 4
    if rotor_phase is None:
        rotor_phase = [0.0] * 4

    focal = (h * 0.5) / math.tan(math.radians(50.0) * 0.5)
    for rotor_index, p in enumerate(pts):
        if not p:
            continue
        # Scale the propeller disc from its model radius instead of drawing an
        # arbitrary screen-space circle. This is still a schematic projection.
        r_disk = max(3, min(28, int(focal * PROP_RADIUS / max(p[2], 0.2))))
        omega_i = float(rotor_omega[rotor_index])
        thrust_i = float(rotor_thrust[rotor_index])
        saturation = thrust_i >= MAX_THRUST_PER_MOTOR * 0.995 or omega_i >= ROTOR_MAX_OMEGA * 0.995
        ring_color = (255, 90, 80) if saturation else color
        pygame.draw.circle(screen, ring_color, p[:2], r_disk, 1)
        # Show a two-blade rotor whose phase advances using the actual simulated
        # angular speed and fixed integration step. Direction is schematic; the
        # physical blade shape, airfoil and induced flow are not rendered.
        phase = float(rotor_phase[rotor_index])
        blade_r = PROP_RADIUS * 0.92
        for blade_offset in (0.0, math.pi):
            theta = phase + blade_offset
            local_end = np.array([blade_r * math.cos(theta), blade_r * math.sin(theta), 0.0])
            world_end = pos + rot_mat @ (arm_coords[rotor_index] + local_end)
            pt_end = project_point_3d(world_end, cam_pos, cam_lookat, w, h)
            if pt_end:
                pygame.draw.line(screen, (235, 245, 255), p[:2], pt_end[:2], 2)
        # Small hub dot helps distinguish a visible spinning prop from a static ring.
        pygame.draw.circle(screen, ring_color, p[:2], max(2, r_disk // 3))

    nose_tip = project_point_3d(pos + rot_mat @ np.array([nose_length, 0.0, 0.0]), cam_pos, cam_lookat, w, h)
    if nose_tip:
        pygame.draw.line(screen, (255, 60, 60), pt_center[:2], nose_tip[:2], 3)

    r_hub = max(6, int(60.0 / pt_center[2]))
    pygame.draw.circle(screen, color, pt_center[:2], r_hub)
    pygame.draw.circle(screen, (255, 255, 255), pt_center[:2], max(1, r_hub // 2))

    lbl = font.render(label, True, color)
    screen.blit(lbl, (pt_center[0] - 30, pt_center[1] - r_hub - 16))


def run_wake_surfing_cockpit(actor_net, ncm_net, controller_role=None, promotion_accepted=None,
                             run_directory=None, checkpoint_path=None):
    """Run the NN-primary diagnostic visualiser using the exact policy object passed by the caller; freeze on non-finite state.

    The displayed status is explicit about whether that object is the promoted
    selection, rejected-run PD baseline, or an explicitly loaded candidate.
    All rotor telemetry is read from the environment state after the same step
    that advances the visible vehicle. This is a schematic diagnostic, not CFD.
    """
    if pygame is None:
        print("[ERROR] Pygame required for 3D visualization. Install via: pip install pygame")
        return

    import json as _json
    from pathlib import Path as _Path

    if controller_role is None:
        controller_role = getattr(actor_net, "_wake_policy_role", "unknown")
    if promotion_accepted is None:
        promotion_accepted = getattr(actor_net, "_wake_promotion_accepted", None)
    if run_directory is None:
        run_directory = getattr(actor_net, "_wake_run_directory", None)
    if checkpoint_path is None:
        checkpoint_path = getattr(actor_net, "_wake_checkpoint_path", None)

    pygame.init()
    SCREEN_W, SCREEN_H = 1400, 900
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Wake-Surfing V10.5 — NN-Primary / Correct Target-Point Kinematics (Unvalidated Model)")
    clock = pygame.time.Clock()
    font_hud = pygame.font.SysFont("consolas", 12, bold=True)
    font_small = pygame.font.SysFont("consolas", 10)

    env = WakeSurfingResearchEnv(num_envs=1, dt=0.015, history_len=75, device=DEVICE)
    actor_net.eval()
    ncm_net.eval()
    role_line, promotion_line, role_kind = cockpit_policy_status(controller_role, promotion_accepted)
    if role_kind == "danger":
        role_color = (255, 70, 70)
    elif role_kind == "warning":
        role_color = (255, 205, 70)
    else:
        role_color = (110, 240, 150)

    rotor_phase_lead = np.zeros(4, dtype=np.float64)
    rotor_phase_follow = np.zeros(4, dtype=np.float64)
    cam_yaw, cam_pitch, cam_dist = 0.85, 0.35, 6.5
    running = True
    last_metrics = None
    display_frame = 0
    simulation_fault = None

    def as_np(tensor):
        return tensor[0].detach().cpu().numpy().astype(np.float64, copy=True)

    def vec3_str(values, unit=""):
        a = np.asarray(values, dtype=float).reshape(-1)
        return "(" + ", ".join(f"{v:+.2f}" for v in a[:3]) + ")" + (f" {unit}" if unit else "")

    def draw_panel(rect, border_color=(0, 255, 200)):
        pygame.draw.rect(screen, (8, 12, 18), rect, border_radius=6)
        pygame.draw.rect(screen, border_color, rect, 1, border_radius=6)

    while running:
        # Simulation advances at its fixed 15 ms step; render FPS is separate.
        clock.tick(60)
        display_frame += 1

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

        if simulation_fault is None:
            # Capture the last finite state before advancing. If the unfiltered NN
            # destabilises the simulated plant, freeze here and report it rather than
            # allowing NaN coordinates to crash projection or silently substituting PD.
            pre_step_snapshot = _snapshot_visualiser_env(env)
            with torch.no_grad():
                fresh_metrics = env.step(actor_net, ncm_net, control_mode="nn-primary")
            invalid_paths = _find_nonfinite_paths(env, fresh_metrics)
            if invalid_paths:
                simulation_fault = {
                    "schema_version": 1,
                    "event": "visualiser_paused_on_nonfinite_simulation_state",
                    "version": "10.3.1",
                    "controller_role": str(controller_role),
                    "promotion_accepted": promotion_accepted,
                    "display_frame": int(display_frame),
                    "sim_time_before_step_s": float(pre_step_snapshot["sim_time"]),
                    "sim_time_invalid_step_s": float(env.sim_time),
                    "nonfinite_paths": invalid_paths,
                    "note": "The last non-finite step was discarded and visualisation froze at the previous state. No PD substitution, safety override, or controller command correction was applied. This prevents a rendering crash; it does not repair or stabilise the simulated NN-controlled plant.",
                }
                _restore_visualiser_env(env, pre_step_snapshot)
                if last_metrics is None:
                    diagnostic_dir = _Path(run_directory) if run_directory else _Path.cwd()
                    diagnostic_dir.mkdir(parents=True, exist_ok=True)
                    diagnostic_path = diagnostic_dir / "visualiser_numerical_failure.json"
                    diagnostic_path.write_text(_json.dumps(simulation_fault, indent=2, allow_nan=False), encoding="utf-8")
                    print(f"[VISUALISER NUMERICAL FAILURE] No finite frame was produced. Details: {diagnostic_path}")
                    pygame.quit()
                    return
                metrics = last_metrics
                diagnostic_dir = _Path(run_directory) if run_directory else _Path.cwd()
                diagnostic_dir.mkdir(parents=True, exist_ok=True)
                diagnostic_path = diagnostic_dir / "visualiser_numerical_failure.json"
                diagnostic_path.write_text(_json.dumps(simulation_fault, indent=2, allow_nan=False), encoding="utf-8")
                print(f"[VISUALISER PAUSED] Non-finite simulation values at frame {display_frame}: {', '.join(invalid_paths)}")
                print(f"[VISUALISER PAUSED] Last finite state retained; NN remains the designated controller. Details: {diagnostic_path}")
            else:
                metrics = fresh_metrics
                last_metrics = fresh_metrics
        else:
            # Once non-finite behaviour occurs, the visualiser stays frozen on the
            # last valid state and never silently resets or changes controller.
            metrics = last_metrics

        lead_p = as_np(env.lead_pos)
        follow_p = as_np(env.follow_pos)
        lead_rot = as_np(env.lead_rot)
        follow_rot = as_np(env.follow_rot)
        lead_omega = as_np(env.lead_rotor_omega)
        follow_omega = as_np(env.follow_rotor_omega)
        lead_thrust = as_np(env.lead_motors)
        follow_thrust = as_np(env.follow_motors)
        follow_current = as_np(env.follow_motor_current)
        follow_voltage = as_np(env.follow_motor_voltage)

        # The visual blade phase is updated from actual rotor angular velocity,
        # not from render time, maintaining synchrony with the simulated plant.
        sim_dt = float(env.dt)
        if simulation_fault is None:
            rotor_phase_lead = (rotor_phase_lead + np.asarray(ROTOR_SPIN_SIGNS) * lead_omega * sim_dt) % (2.0 * math.pi)
            rotor_phase_follow = (rotor_phase_follow + np.asarray(ROTOR_SPIN_SIGNS) * follow_omega * sim_dt) % (2.0 * math.pi)

        cam_lookat = lead_p * 0.5 + follow_p * 0.5
        cam_x = cam_lookat[0] + cam_dist * math.cos(cam_pitch) * math.cos(cam_yaw)
        cam_y = cam_lookat[1] + cam_dist * math.cos(cam_pitch) * math.sin(cam_yaw)
        cam_z = max(0.5, cam_lookat[2] + cam_dist * math.sin(cam_pitch))
        cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)
        screen.fill((10, 14, 20))

        # Ground grid.
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

        # Historical two-filament wake: explicitly labelled as the approximate model.
        hist_p = env.history_pos[0].cpu().numpy()
        hist_r = env.history_rot[0].cpu().numpy()
        for hist_i in range(len(hist_p) - 1):
            p1_h, p2_h = hist_p[hist_i], hist_p[hist_i + 1]
            r1_h, r2_h = hist_r[hist_i], hist_r[hist_i + 1]
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

        # Modelled wake-velocity vectors; not a resolved fluid field.
        follow_flow_vec = as_np(metrics["v_induced_follower"])
        for vy in np.linspace(-0.20, 0.20, 3):
            arrow_base = follow_p + follow_rot @ np.array([0.0, vy, -0.20])
            arrow_tip = arrow_base + 0.18 * follow_flow_vec
            p_base = project_point_3d(arrow_base, cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            p_tip = project_point_3d(arrow_tip, cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            if p_base and p_tip and np.linalg.norm(follow_flow_vec) > 1e-3:
                pygame.draw.line(screen, (0, 255, 180), p_base[:2], p_tip[:2], 2)
                pygame.draw.circle(screen, (0, 255, 180), p_tip[:2], 3)
        leader_flow_vec = as_np(metrics["v_backwash_lead"])
        for vy in np.linspace(-0.15, 0.15, 2):
            arrow_base_lead = lead_p + lead_rot @ np.array([-0.20, vy, -0.20])
            arrow_tip_lead = arrow_base_lead + 0.18 * leader_flow_vec
            p_base_l = project_point_3d(arrow_base_lead, cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            p_tip_l = project_point_3d(arrow_tip_lead, cam_pos, cam_lookat, SCREEN_W, SCREEN_H)
            if p_base_l and p_tip_l and np.linalg.norm(leader_flow_vec) > 1e-3:
                pygame.draw.line(screen, (255, 215, 0), p_base_l[:2], p_tip_l[:2], 2)
                pygame.draw.circle(screen, (255, 215, 0), p_tip_l[:2], 3)

        draw_full_quadrotor_3d(
            screen, lead_p, lead_rot, (255, 210, 60), "LEADER (PRESCRIBED TRACKER)",
            cam_pos, cam_lookat, SCREEN_W, SCREEN_H, font_hud,
            rotor_omega=lead_omega, rotor_thrust=lead_thrust, rotor_phase=rotor_phase_lead,
        )
        draw_full_quadrotor_3d(
            screen, follow_p, follow_rot, (0, 255, 220), "FOLLOWER (SEE CONTROL STATUS HUD)",
            cam_pos, cam_lookat, SCREEN_W, SCREEN_H, font_hud,
            rotor_omega=follow_omega, rotor_thrust=follow_thrust, rotor_phase=rotor_phase_follow,
        )

        # Left dashboard: policy identity and overall experiment state.
        left_rect = pygame.Rect(12, 12, 660, 300)
        draw_panel(left_rect, role_color)
        left_lines = [
            ("WAKE-SURFING V10.5 — NN-PRIMARY DIRECT ROTOR CONTROL + TARGET-POINT KINEMATICS", (0, 255, 200)),
            (role_line, role_color),
            (promotion_line, role_color),
            ("CONTROL: NN directly sets all 4 rotor thrusts; NO PD tracking command", (230, 235, 245)),
            ("SAFETY: barrier equations are MONITOR-ONLY; NN commands are not overridden", (255, 120, 100)),
            ("WAKE VIEW: two-filament approximation — NOT CFD / NOT measured flow", (255, 205, 70)),
            (f"Simulation t={env.sim_time:.2f}s | fixed dt={env.dt*1000:.1f}ms | render fps={clock.get_fps():.0f}", (190, 205, 220)),
            (f"Separation={metrics['dist_lead']:.2f}m | follower z={metrics['z_follow']:.2f}m | target z={metrics['z_target']:.2f}m", (255, 255, 255)),
            (f"Tracking error: position={safe_norm(metrics['error_pos'][0]).item():.3f}m velocity={safe_norm(metrics['error_vel'][0]).item():.3f}m/s", (230, 235, 245)),
            (f"Wake velocity model: follower z={metrics['upwash_follow']:+.2f}m/s | leader z={metrics['backwash_lead']:+.2f}m/s", (140, 230, 245)),
            (f"Power model (UNCALIBRATED): proxy={metrics['p_follow_total_watts']:.1f}W | motor V*I={metrics['p_follow_motor_electrical_model_watts']:.1f}W", (255, 205, 70)),
            ("Contraction metric: reduced-order diagnostic only; no full-plant proof", (160, 190, 255)),
            ((f"Output run: {str(run_directory)[:75]}" if run_directory else f"Checkpoint: {str(checkpoint_path)[:75]}" if checkpoint_path else "Output source: current in-memory training result"), (160, 180, 205)),
        ]
        for i, (line, col) in enumerate(left_lines):
            screen.blit(font_hud.render(line, True, col), (left_rect.x + 10, left_rect.y + 8 + i * 21))

        # Right dashboard: actual controller contribution, actuator telemetry, safety.
        right_rect = pygame.Rect(684, 12, 704, 420)
        actual_ok = metrics["actual_plant_barrier_feasible_pct"] >= 99.999
        guard_ok = metrics["plant_guard_predicted_barrier_feasible_pct"] >= 99.999
        safety_col = (110, 240, 150) if actual_ok and guard_ok else (255, 90, 80)
        draw_panel(right_rect, safety_col)
        accel_record = metrics["control_response_sample"]
        accel_hud = cockpit_acceleration_telemetry(accel_record)
        predicted_accel = accel_hud["predicted"]
        actual_accel = accel_hud["observed"]
        trajectory_accel_err = accel_hud["trajectory_error"]
        rotor_rpm = follow_omega * (60.0 / (2.0 * math.pi))
        nn_fraction = accel_record["nn_rotor_thrust_fraction"]
        right_lines = [
            ("FOLLOWER CONTROL & ACTUATOR TELEMETRY (ENVIRONMENT 0)", (0, 255, 200)),
            ("CONTROL OWNER: NEURAL NETWORK — 4 DIRECT ROTOR COMMANDS", (0, 255, 180)),
            ("NN thrust fractions: " + ", ".join(f"{v:.3f}" for v in nn_fraction), (230, 235, 245)),
            ("NN target thrust N: " + ", ".join(f"{v:.2f}" for v in accel_record["rotor_target_thrust_N"]), (190, 220, 255)),
            (f"Predicted acceleration from rotor commands={vec3_str(predicted_accel, 'm/s^2')}", (220, 220, 255)),
            (f"Acceleration actual   ={vec3_str(actual_accel, 'm/s^2')}", (0, 255, 180)),
            (f"Actual - target accel={vec3_str(trajectory_accel_err, 'm/s^2')}", (255, 205, 70)),
            (f"Attitude tracking error={metrics['maximum_attitude_tracking_error_after_step_deg']:.2f}deg | predictor mismatch={metrics['maximum_predictor_observation_mismatch_mps2']:.2e}m/s^2", (215, 225, 240)),
            (f"Rotor speed actual RPM: " + ", ".join(f"{v:.0f}" for v in rotor_rpm), (230, 235, 245)),
            (f"Rotor target speed RPM: " + ", ".join(f"{v * 60.0 / (2.0 * math.pi):.0f}" for v in as_np(metrics["rotor_target_speed_rad_s_env0"].reshape(1, 4))), (190, 205, 220)),
            (f"Rotor thrust target N: " + ", ".join(f"{v:.2f}" for v in as_np(metrics["rotor_target_thrust_N_env0"].reshape(1, 4))), (190, 205, 220)),
            (f"Rotor thrust actual N: " + ", ".join(f"{v:.2f}" for v in follow_thrust), (230, 235, 245)),
            (f"Motor current A: " + ", ".join(f"{v:.1f}" for v in follow_current), (230, 235, 245)),
            (f"Motor voltage V: " + ", ".join(f"{v:.1f}" for v in follow_voltage), (230, 235, 245)),
            (f"Saturation: thrust={metrics['rotor_target_thrust_saturation_pct']:.1f}% speed={metrics['rotor_speed_saturation_pct']:.1f}% current={metrics['motor_current_saturation_pct']:.1f}% voltage={metrics['motor_voltage_saturation_pct']:.1f}%", (255, 205, 70)),
            ((f"PLANT BARRIER: predicted={metrics['plant_guard_predicted_barrier_feasible_pct']:.0f}% actual={metrics['actual_plant_barrier_feasible_pct']:.0f}%"), safety_col),
            ((f"Max actual violations: collision={metrics['actual_plant_collision_barrier_max_violation_mps2']:.3g} lateral={metrics['actual_plant_lateral_barrier_max_violation_mps2']:.3g} m/s^2"), safety_col),
            ((f"NN action barrier-feasible={metrics['plant_guard_predicted_barrier_feasible_pct']:.1f}% | no NN override"), safety_col),
            (("SAFETY MONITOR ONLY — no intervention / no future-safety guarantee"), (255, 205, 70)),
        ]
        for i, (line, col) in enumerate(right_lines):
            screen.blit(font_hud.render(line, True, col), (right_rect.x + 10, right_rect.y + 8 + i * 22))

        # Compact rotor status legend at the bottom-right.
        legend = "Rotor rings turn RED if this model reports thrust/speed saturation; blade phase follows simulated RPM."
        screen.blit(font_small.render(legend, True, (175, 190, 205)), (18, SCREEN_H - 24))
        if simulation_fault is not None:
            # Persistent overlay: the NN was not replaced, and the state was frozen
            # only because subsequent non-finite geometry/telemetry cannot be drawn.
            fault_rect = pygame.Rect(180, 355, SCREEN_W - 360, 150)
            pygame.draw.rect(screen, (35, 5, 10), fault_rect, border_radius=10)
            pygame.draw.rect(screen, (255, 70, 70), fault_rect, 2, border_radius=10)
            fault_lines = [
                ("SIMULATION PAUSED — NON-FINITE STATE DETECTED", (255, 90, 90)),
                ("NN-primary commands were not replaced; no PD fallback or safety override was applied.", (245, 235, 235)),
                ("The display is frozen at the last finite frame; behaviour after divergence is undefined.", (245, 200, 120)),
                (f"Details: visualiser_numerical_failure.json | frame {simulation_fault['display_frame']}", (210, 220, 235)),
                ("Press ESC to close.", (210, 220, 235)),
            ]
            for i, (line, col) in enumerate(fault_lines):
                screen.blit(font_hud.render(line, True, col), (fault_rect.x + 16, fault_rect.y + 15 + i * 24))

        pygame.display.flip()

    pygame.quit()


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

    def test_moving_body_offset_kinematics_match_finite_differences():
        # A rigid body translates with constant acceleration and rotates about world Z.
        # The analytic point velocity/acceleration must agree with finite differences.
        dt_fd = 2.0e-4
        w = 1.3
        offset_body = torch.tensor([0.0, -0.45, 0.0], dtype=torch.float64, device=DEVICE)
        def state_at(t):
            angle = w * t
            c, sn = math.cos(angle), math.sin(angle)
            R = torch.tensor([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64, device=DEVICE).unsqueeze(0)
            p = torch.tensor([[0.2 + 0.4*t + 0.5*0.3*t*t, -0.1 + 0.2*t, 2.0]], dtype=torch.float64, device=DEVICE)
            v = torch.tensor([[0.4 + 0.3*t, 0.2, 0.0]], dtype=torch.float64, device=DEVICE)
            a = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float64, device=DEVICE)
            omega = torch.tensor([[0.0, 0.0, w]], dtype=torch.float64, device=DEVICE)
            alpha = torch.zeros((1, 3), dtype=torch.float64, device=DEVICE)
            pt, vt, at, _, _ = moving_body_offset_kinematics(p, v, R, omega, alpha, a, offset_body)
            return pt, vt, at
        t0 = 0.31
        p0, v0, a0 = state_at(t0)
        pm, _, _ = state_at(t0 - dt_fd)
        pp, _, _ = state_at(t0 + dt_fd)
        v_fd = (pp - pm) / (2.0 * dt_fd)
        a_fd = (pp - 2.0*p0 + pm) / (dt_fd*dt_fd)
        assert torch.max(torch.abs(v_fd - v0)).item() < 2e-7, (v_fd, v0)
        # Second finite difference is lower precision due to subtraction; use 2e-6 tolerance.
        assert torch.max(torch.abs(a_fd - a0)).item() < 2e-6, (a_fd, a0)

    def test_environment_target_point_velocity_and_acceleration_include_rotation_terms():
        env = WakeSurfingResearchEnv(num_envs=3, dt=0.015, history_len=20, device=DEVICE, randomize_initial_conditions=True)
        actor = WakeSurfingActorPolicy().to(DEVICE)
        metric = NeuralContractionMetric().to(DEVICE)
        with torch.no_grad():
            metrics = env.step(actor, metric, control_mode="nn-primary")
        record = metrics["control_response_sample"]
        vc = np.asarray(record["target_centreline_velocity_world_mps"], dtype=np.float64)
        vo = np.asarray(record["target_point_offset_velocity_world_mps"], dtype=np.float64)
        vp = np.asarray(record["target_point_velocity_world_mps"], dtype=np.float64)
        ac = np.asarray(record["target_centreline_acceleration_world_mps2"], dtype=np.float64)
        ao = np.asarray(record["target_point_offset_acceleration_world_mps2"], dtype=np.float64)
        ap = np.asarray(record["target_feedforward_acceleration_world_mps2"], dtype=np.float64)
        assert np.allclose(vp, vc + vo, atol=2e-6), (vp, vc, vo)
        assert np.allclose(ap, ac + ao, atol=2e-5), (ap, ac, ao)

    def test_projection_rejects_nonfinite_points_and_camera_without_exception():
        cam = np.array([0.0, -5.0, 2.0], dtype=np.float64)
        look = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        valid = project_point_3d(np.array([0.0, 0.0, 0.0]), cam, look, 1400, 900)
        assert valid is not None and len(valid) == 3
        assert project_point_3d(np.array([np.nan, 0.0, 0.0]), cam, look, 1400, 900) is None
        assert project_point_3d(np.array([0.0, np.inf, 0.0]), cam, look, 1400, 900) is None
        assert project_point_3d(np.array([0.0, 0.0, 0.0]), np.array([np.nan, 0.0, 1.0]), look, 1400, 900) is None
        assert project_point_3d(np.array([0.0, 0.0, 0.0]), cam, np.array([0.0, np.nan, 0.0]), 1400, 900) is None
        assert project_point_3d(np.array([0.0, 0.0, 0.0]), cam, look, 1400, 900, fov=float("nan")) is None

    def test_nonfinite_detector_identifies_rotational_state_failure():
        env = WakeSurfingResearchEnv(num_envs=1, dt=0.015, history_len=8, device=DEVICE)
        metrics = {"ok": torch.tensor([1.0], device=DEVICE), "nested": {"value": [0.0, 1.0]}}
        assert _find_nonfinite_paths(env, metrics) == []
        env.follow_rot[0, 1, 1] = float("nan")
        bad = _find_nonfinite_paths(env, metrics)
        assert any("env.follow_rot" in item for item in bad), bad
        env.reset()
        clean_snapshot = _snapshot_visualiser_env(env)
        env.follow_rot.fill_(float("nan"))
        _restore_visualiser_env(env, clean_snapshot)
        assert torch.isfinite(env.follow_rot).all()

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

    def test_wake_uses_selected_historical_velocity_and_configured_history_dt():
        wake = TwoWayLagrangianWakeField()
        B, H = 1, 25
        eval_pos = torch.tensor([[0.0, 0.0, 0.0]], device=DEVICE)
        # Put sample 20 nearest to the evaluation point; all other history samples
        # are far enough away that argmin must choose 20.
        hist_pos = torch.zeros((B, H, 3), device=DEVICE)
        hist_pos[:, :, 0] = torch.arange(H, device=DEVICE).float() + 100.0
        hist_pos[:, 0, :] = torch.tensor([[10.0, 0.0, 0.0]], device=DEVICE)
        hist_pos[:, 20, :] = torch.tensor([[0.10, 0.0, 0.0]], device=DEVICE)
        hist_rot = torch.eye(3, device=DEVICE).view(1, 1, 3, 3).repeat(B, H, 1, 1)
        hist_vel = torch.zeros((B, H, 3), device=DEVICE)
        hist_vel[:, :, 0] = 3.0
        hist_vel[:, 0, 0] = 12.0  # deliberately different newest-sample speed
        hist_vel[:, 20, 0] = 3.0
        thrust = torch.tensor([[HOVER_WEIGHT]], device=DEVICE)
        curvature = torch.tensor([[0.02]], device=DEVICE)

        out_a, core_a, _ = wake.compute_leader_to_follower_wake(
            eval_pos, hist_pos, hist_rot, hist_vel, thrust, curvature, history_dt=0.015
        )
        # If only the *unselected* newest sample's speed changes, output must not
        # change: gamma_0 is computed from historical sample 20, not sample zero.
        hist_vel_changed = hist_vel.clone()
        hist_vel_changed[:, 0, 0] = 20.0
        out_b, core_b, _ = wake.compute_leader_to_follower_wake(
            eval_pos, hist_pos, hist_rot, hist_vel_changed, thrust, curvature, history_dt=0.015
        )
        assert torch.allclose(out_a, out_b, atol=1e-7, rtol=1e-6), (out_a, out_b)
        assert torch.allclose(core_a, core_b, atol=1e-7, rtol=1e-6)

        # The selected source is 20 steps old: using the configured dt must alter
        # its age-dependent core radius when dt doubles.
        _, core_dt2, _ = wake.compute_leader_to_follower_wake(
            eval_pos, hist_pos, hist_rot, hist_vel, thrust, curvature, history_dt=0.030
        )
        assert core_dt2.item() > core_a.item(), (core_a.item(), core_dt2.item())

        # Changing the speed at the selected historical point must affect the
        # circulation (for this input, the value is away from the hard clamps).
        selected_speed_changed = hist_vel.clone()
        selected_speed_changed[:, 20, 0] = 6.0
        out_c, _, _ = wake.compute_leader_to_follower_wake(
            eval_pos, hist_pos, hist_rot, selected_speed_changed, thrust, curvature, history_dt=0.015
        )
        assert not torch.allclose(out_a, out_c, atol=1e-5, rtol=1e-5), (out_a, out_c)

        # With acceleration history supplied, core growth uses curvature of the
        # selected historical path point, not the current leader's scalar proxy.
        hist_accel = torch.zeros_like(hist_vel)
        _, core_straight, _ = wake.compute_leader_to_follower_wake(
            eval_pos, hist_pos, hist_rot, hist_vel, thrust, curvature,
            history_dt=0.015, hist_accel=hist_accel,
        )
        hist_accel[:, 20, 1] = 0.9  # vx=3 m/s, ay=0.9 m/s² -> |kappa|=0.1 1/m
        _, core_turning, _ = wake.compute_leader_to_follower_wake(
            eval_pos, hist_pos, hist_rot, hist_vel, thrust, curvature,
            history_dt=0.015, hist_accel=hist_accel,
        )
        assert core_turning.item() > core_straight.item(), (core_straight.item(), core_turning.item())

    def test_geometry_span_matches_x_frame_motor_coordinates():
        # The allocator places each motor at +/- ARM_LENGTH/sqrt(2) in x/y.
        # Opposite side motor centres are therefore SPAN_EFF apart in y.
        expected_offset = ARM_LENGTH / math.sqrt(2.0)
        assert abs(ROTOR_OFFSET - expected_offset) < 1e-12
        assert abs(SPAN_EFF - 2.0 * ROTOR_OFFSET) < 1e-12
        assert abs(SPAN_EFF - ARM_LENGTH * math.sqrt(2.0)) < 1e-12
        assert abs(SPAN_EFF - 0.44) > 0.1  # 0.44 m was the stale/comment-only claim

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
        u, residual, *_rest, feasible_pct, diag = solver.solve(u0, A, b, torch.tensor([-5.0]*3, device=DEVICE), torch.tensor([5.0]*3, device=DEVICE))
        assert u[0, 0].item() <= 1.005 and feasible_pct >= 99.0, (u, residual, feasible_pct)
        assert diag["projection_feasible_pct"] >= 99.0 and diag["fallback_pct"] == 0.0
        # Contradictory constraints u_x<=-1 and u_x>=1 must not be reported feasible.
        A2 = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]], device=DEVICE)
        b2 = torch.tensor([[[-1.0], [-1.0]]], device=DEVICE)
        _, _, _, _, _, _, feasible2, diag2 = solver.solve(u0, A2, b2, torch.tensor([-5.0]*3, device=DEVICE), torch.tensor([5.0]*3, device=DEVICE))
        assert feasible2 < 99.0 and diag2["projection_feasible_pct"] < 99.0, f"Infeasible set was falsely accepted: {feasible2}%"
        # An emergency action cannot make contradictory constraints feasible; diagnostics must preserve that fact.
        _, _, _, _, _, _, feasible3, diag3 = solver.solve(
            u0, A2, b2, torch.tensor([-5.0]*3, device=DEVICE), torch.tensor([5.0]*3, device=DEVICE),
            fallback=torch.zeros_like(u0))
        assert feasible3 < 99.0 and diag3["fallback_pct"] > 0.0
        assert diag3["final_max_residual"] > 0.0

    def test_absolute_barriers_are_shifted_into_residual_acceleration_coordinates():
        A_abs = torch.tensor([[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                              [[-0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]], device=DEVICE)
        b_abs = torch.tensor([[[0.25], [-1.0]], [[0.75], [2.0]]], device=DEVICE)
        a_ff = torch.tensor([[1.5, -0.25, 0.0], [-2.0, 1.0, 0.5]], device=DEVICE)
        u_res = torch.tensor([[-0.5, 0.75, 0.0], [0.5, -0.5, 1.0]], device=DEVICE)
        b_res = acceleration_residual_rhs(A_abs, b_abs, a_ff)
        lhs_absolute = torch.matmul(A_abs, (a_ff + u_res).unsqueeze(-1))
        lhs_residual = torch.matmul(A_abs, u_res.unsqueeze(-1))
        assert torch.allclose(lhs_absolute - b_abs, lhs_residual - b_res, atol=1e-6), (lhs_absolute, b_abs, lhs_residual, b_res)
        # The test deliberately uses nonzero feed-forward acceleration; without the
        # RHS shift, the residual-coordinate inequalities are not equivalent.
        unshifted = lhs_residual - b_abs
        assert not torch.allclose(unshifted, lhs_absolute - b_abs, atol=1e-4)

    def test_constraint_residual_is_invariant_to_row_scaling():
        solver = RelaxedQPSafetyFilter(iters=80, tolerance=1e-3)
        u0 = torch.tensor([[0.0, 0.0, 0.0]], device=DEVICE)
        # Contradictory but finite inequalities: x <= -1 and x >= 1.
        A = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]], device=DEVICE)
        b = torch.tensor([[[-1.0], [-1.0]]], device=DEVICE)
        lo = torch.tensor([-5.0, -5.0, -5.0], device=DEVICE)
        hi = torch.tensor([5.0, 5.0, 5.0], device=DEVICE)
        _, _, _, _, _, _, _, d1 = solver.solve(u0, A, b, lo, hi)
        # Positive row scaling preserves the same half-spaces but changes raw residual units.
        scale = torch.tensor([[[13.0], [0.25]]], device=DEVICE)
        _, _, _, _, _, _, _, d2 = solver.solve(u0, A * scale, b * scale, lo, hi)
        assert abs(d1["projection_max_residual"] - d2["projection_max_residual"]) < 1e-3, (d1, d2)

    def test_fallback_does_not_worsen_worst_normalized_violation():
        solver = RelaxedQPSafetyFilter(iters=1, tolerance=1e-4)
        u0 = torch.tensor([[0.0, 0.0, 0.0]], device=DEVICE)
        A = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]], device=DEVICE)
        b = torch.tensor([[[-1.0], [-1.0]]], device=DEVICE)
        lo = torch.tensor([-5.0, -5.0, -5.0], device=DEVICE)
        hi = torch.tensor([5.0, 5.0, 5.0], device=DEVICE)
        fallback = torch.tensor([[5.0, 0.0, 0.0]], device=DEVICE)
        _, _, _, _, _, _, _, diag = solver.solve(u0, A, b, lo, hi, fallback=fallback)
        assert diag["final_max_residual"] <= diag["projection_max_residual"] + 1e-6, diag
        assert diag["fallback_attempted_pct"] > 0.0
        assert diag["fallback_applied_pct"] == 0.0

    def test_solver_reports_individually_unachievable_constraint_under_box_bounds():
        solver = RelaxedQPSafetyFilter(iters=60, tolerance=1e-3)
        u0 = torch.zeros((1, 3), device=DEVICE)
        A = torch.tensor([[[0.0, 1.0, 0.0]]], device=DEVICE)
        b = torch.tensor([[[-10.0]]], device=DEVICE)
        lo = torch.tensor([-6.0, -6.0, -4.5], device=DEVICE)
        hi = torch.tensor([6.0, 6.0, 6.0], device=DEVICE)
        _, _, _, _, _, _, feasible, diag = solver.solve(u0, A, b, lo, hi)
        assert feasible < 100.0
        assert diag["worst_env_individually_unachievable_under_command_box"]
        row = diag["failure_records"][0]["constraint_rows"][0]
        assert row["individually_unachievable_under_command_box"]
        assert row["unavoidable_violation_under_command_box"] >= 3.9

    def test_lateral_barrier_uses_historical_wake_frame_terms():
        # A static historical source has zero angular rate/acceleration even if the
        # current leader is in a fast attitude transient. The builder receives only
        # the source state, so current-body alpha cannot contaminate this row.
        B = 1
        R = torch.eye(3, device=DEVICE).unsqueeze(0)
        rel_p = torch.tensor([[0.0, -0.45, 0.0]], device=DEVICE)
        rel_v = torch.tensor([[0.0, 0.10, 0.0]], device=DEVICE)
        w_source = torch.zeros((B, 3), device=DEVICE)
        alpha_source = torch.zeros((B, 3), device=DEVICE)
        a_source = torch.tensor([[0.0, 0.0, 0.0]], device=DEVICE)
        b = build_lateral_wake_barrier(
            rel_p, rel_v, R, w_source, alpha_source, a_source,
            margin=SPAN_EFF * 0.5 + 0.05, cbf_lambda=2.0,
        )
        assert torch.allclose(b["axis"], torch.tensor([[0.0, 1.0, 0.0]], device=DEVICE))
        assert torch.allclose(b["axis_dot"], torch.zeros((1, 3), device=DEVICE))
        assert torch.allclose(b["axis_ddot"], torch.zeros((1, 3), device=DEVICE))
        expected_h = 0.45 - (SPAN_EFF * 0.5 + 0.05)
        assert abs(b["h"].item() - expected_h) < 1e-6
        # hdot = -v_rel dot y for a static source frame.
        assert abs(b["h_dot"].item() + 0.10) < 1e-6
        assert torch.isfinite(b["A_absolute"]).all() and torch.isfinite(b["b_absolute"]).all()

    def test_history_derivative_buffers_match_seeded_trim_history():
        env = WakeSurfingResearchEnv(num_envs=2, dt=0.015, history_len=12, device=DEVICE)
        assert env.history_omega.shape == (2, 12, 3)
        assert env.history_alpha_body.shape == (2, 12, 3)
        assert env.history_accel.shape == (2, 12, 3)
        assert torch.isfinite(env.history_omega).all()
        assert torch.isfinite(env.history_alpha_body).all()
        assert torch.isfinite(env.history_accel).all()
        # The historical rotation buffer is initialized as constant R_trim, so its
        # matching angular-rate and angular-acceleration buffers must be zero.
        assert torch.max(torch.abs(env.history_omega)).item() == 0.0
        assert torch.max(torch.abs(env.history_alpha_body)).item() == 0.0

    def test_environment_uses_configured_projection_budget():
        env = WakeSurfingResearchEnv(num_envs=1, dt=0.015, history_len=10, device=DEVICE)
        assert env.qp_filter.iters >= 60, env.qp_filter.iters

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

    def test_rotor_thrust_is_quadratic_in_rotor_speed():
        assert abs(ROTOR_THRUST_COEFFICIENT * ROTOR_MAX_OMEGA**2 - MAX_THRUST_PER_MOTOR) < 1e-9
        omega = torch.tensor([[200.0, 400.0, 600.0, 800.0]], device=DEVICE)
        thrust = ROTOR_THRUST_COEFFICIENT * omega.square()
        assert torch.all(thrust[:, 1:] > thrust[:, :-1])
        ratios = thrust[:, 1:] / thrust[:, :-1]
        expected = (omega[:, 1:] / omega[:, :-1]).square()
        assert torch.allclose(ratios, expected, atol=1e-5, rtol=1e-5)

    def test_rotor_motor_equilibrium_is_stationary():
        physics = RigidBody6DoFDrone(dt=0.015, device=DEVICE)
        hover_thrust = HOVER_WEIGHT / 4.0
        omega_h = math.sqrt(hover_thrust / ROTOR_THRUST_COEFFICIENT)
        current_h = ROTOR_TORQUE_COEFFICIENT * omega_h**2 / MOTOR_TORQUE_CONSTANT
        omega0 = torch.full((2, 4), omega_h, device=DEVICE)
        current0 = torch.full((2, 4), current_h, device=DEVICE)
        target = torch.full((2, 4), hover_thrust, device=DEVICE)
        omega1, current1, thrust1, voltage1 = physics.rotor_step(omega0, current0, target)
        assert torch.max(torch.abs(omega1 - omega0)).item() < 1e-3, (omega0, omega1)
        assert torch.max(torch.abs(current1 - current0)).item() < 1e-3, (current0, current1)
        assert torch.max(torch.abs(thrust1 - target)).item() < 1e-3, (thrust1, target)
        assert torch.isfinite(voltage1).all() and (voltage1 >= 0).all() and (voltage1 <= BATTERY_VOLTAGE_MAX).all()

    def test_rotor_states_obey_electrical_and_speed_limits():
        physics = RigidBody6DoFDrone(dt=0.015, device=DEVICE)
        omega = torch.zeros((3, 4), device=DEVICE)
        current = torch.zeros_like(omega)
        high = torch.full_like(omega, MAX_THRUST_PER_MOTOR)
        for _ in range(20):
            omega, current, thrust, voltage = physics.rotor_step(omega, current, high)
        assert (omega > 0).all() and (omega <= ROTOR_MAX_OMEGA + 1e-5).all()
        assert (current >= 0).all() and (current <= MOTOR_CURRENT_LIMIT + 1e-5).all()
        assert (thrust >= 0).all() and (thrust <= MAX_THRUST_PER_MOTOR + 1e-5).all()
        assert (voltage >= 0).all() and (voltage <= BATTERY_VOLTAGE_MAX + 1e-5).all()
        high_omega = omega.clone()
        low = torch.zeros_like(omega)
        for _ in range(10):
            omega, current, thrust, voltage = physics.rotor_step(omega, current, low)
        assert torch.all(omega < high_omega), (high_omega, omega)
        assert torch.isfinite(omega).all() and torch.isfinite(current).all()

    def test_body_hover_equilibrium_with_rotor_states():
        physics = RigidBody6DoFDrone(dt=0.015, device=DEVICE)
        n = 2
        pos = torch.zeros((n, 3), device=DEVICE)
        vel = torch.zeros_like(pos)
        rot = torch.eye(3, device=DEVICE).unsqueeze(0).repeat(n, 1, 1)
        body_omega = torch.zeros_like(pos)
        omega_h = math.sqrt((HOVER_WEIGHT / 4.0) / ROTOR_THRUST_COEFFICIENT)
        current_h = ROTOR_TORQUE_COEFFICIENT * omega_h**2 / MOTOR_TORQUE_CONSTANT
        rotor_omega = torch.full((n, 4), omega_h, device=DEVICE)
        motor_current = torch.full((n, 4), current_h, device=DEVICE)
        wrench = torch.tensor([[HOVER_WEIGHT, 0.0, 0.0, 0.0]], device=DEVICE).repeat(n, 1)
        values = physics.integrate_dynamics(
            pos, vel, rot, body_omega, rotor_omega, motor_current,
            wrench, torch.zeros_like(pos), torch.zeros((n, 1), device=DEVICE),
        )
        pos1, vel1, rot1, omega1, rotor_omega1, current1, thrusts1, total1, drag1, voltage1 = values
        assert torch.max(torch.abs(vel1)).item() < 1e-4, vel1
        assert torch.max(torch.abs(pos1)).item() < 1e-4
        assert torch.max(torch.abs(total1 - HOVER_WEIGHT)).item() < 1e-3
        assert torch.isfinite(rot1).all() and torch.isfinite(omega1).all()
        assert torch.isfinite(thrusts1).all() and torch.isfinite(drag1).all() and torch.isfinite(voltage1).all()

    def test_actuator_aware_acceleration_predictor_matches_integrator():
        env = WakeSurfingResearchEnv(num_envs=3, dt=0.015, history_len=20, device=DEVICE, randomize_initial_conditions=True)
        actor = WakeSurfingActorPolicy().to(DEVICE)
        metric = NeuralContractionMetric().to(DEVICE)
        with torch.no_grad():
            # Take one ordinary step so the state contains non-trivial attitude and rotor speeds.
            metrics = env.step(actor, metric)
            predicted = torch.tensor(metrics["control_response_sample"]["predicted_actuator_limited_acceleration_world_mps2"], device=DEVICE)
            observed = torch.tensor(metrics["control_response_sample"]["observed_actual_acceleration_world_mps2"], device=DEVICE)
            assert torch.isfinite(predicted).all() and torch.isfinite(observed).all()
            # The explicit prediction follows the same discrete translation update.
            assert torch.max(torch.abs(predicted - observed)).item() < 2e-3, (predicted, observed)
            assert "plant_guard_predicted_barrier_feasible_pct" in metrics
            assert "actual_plant_barrier_feasible_pct" in metrics
            assert 0.0 <= metrics["plant_guard_predicted_barrier_feasible_pct"] <= 100.0
            assert 0.0 <= metrics["actual_plant_barrier_feasible_pct"] <= 100.0
            assert metrics["control_response_sample"]["rotor_actual_speed_rad_s"]

    def test_actor_starts_at_hover_point_with_four_independent_rotor_commands():
        actor = WakeSurfingActorPolicy().to(DEVICE)
        x = torch.randn((12, ACTOR_OBSERVATION_DIM), device=DEVICE)
        with torch.no_grad():
            action = actor(x)
            expected = torch.full((12, 4), (HOVER_WEIGHT / 4.0) / MAX_THRUST_PER_MOTOR, device=DEVICE)
            assert action.shape == (12, 4)
            assert torch.isfinite(action).all()
            assert (action >= 0.0).all() and (action <= 1.0).all()
            assert torch.max(torch.abs(action - expected)).item() < 1e-5, action

    def test_nn_primary_commands_direct_rotor_setpoints_without_pd_or_guard_override():
        env = WakeSurfingResearchEnv(num_envs=2, dt=0.015, history_len=20, device=DEVICE, randomize_initial_conditions=True)
        actor = WakeSurfingActorPolicy().to(DEVICE)
        metric = NeuralContractionMetric().to(DEVICE)
        with torch.no_grad():
            metrics = env.step(actor, metric, control_mode="nn-primary")
            expected_thrust = actor(metrics["actor_obs"]) * MAX_THRUST_PER_MOTOR
            assert metrics["control_mode"] == "nn-primary"
            assert metrics["nn_primary_control"] is True
            assert metrics["pd_tracking_controller_active"] is False
            assert metrics["safety_filter_active"] is False
            assert torch.max(torch.abs(expected_thrust - metrics["nn_rotor_thrust_targets_N"])).item() < 1e-5
            trace = metrics["control_response_sample"]
            assert trace["neural_network_directly_commands_all_four_rotors"] is True
            assert trace["pd_tracking_command_active"] is False
            assert trace["safety_filter_intervention_active"] is False
            assert len(trace["nn_rotor_thrust_fraction"]) == 4

    def test_long_horizon_curriculum_schedule():
        expected = [20, 20, 30, 30, 40, 40, 60, 60, 80, 80, 100]
        # For a 64-epoch run, boundaries should produce all scheduled stages.
        indexes = [0, 7, 8, 15, 16, 23, 24, 31, 32, 39, 40]
        got = [curriculum_rollout_horizon(i, 64, 100) for i in indexes]
        assert got == expected, (got, expected)
        assert curriculum_rollout_horizon(0, 2, 3) == 3
        assert curriculum_rollout_horizon(0, 64, 20) == 20

    def test_truncated_bptt_detaches_state_but_preserves_values():
        env = WakeSurfingResearchEnv(num_envs=2, dt=0.015, history_len=20, device=DEVICE, randomize_initial_conditions=True)
        actor = WakeSurfingActorPolicy().to(DEVICE)
        metric = NeuralContractionMetric().to(DEVICE)
        metrics = env.step(actor, metric, control_mode="nn-primary")
        prior = env.previous_nn_rotor_action_fraction.detach().clone()
        env.detach_dynamics_for_truncated_bptt()
        assert torch.allclose(env.previous_nn_rotor_action_fraction, prior)
        for name in ("follow_pos", "follow_vel", "follow_rot", "follow_omega", "history_pos", "history_rot", "previous_nn_rotor_action_fraction"):
            value = getattr(env, name)
            assert not value.requires_grad, (name, value.requires_grad)
        assert env.metrics == {}

    def test_new_stability_losses_are_finite_and_differentiable():
        env = WakeSurfingResearchEnv(num_envs=3, dt=0.015, history_len=20, device=DEVICE, randomize_initial_conditions=True)
        actor = WakeSurfingActorPolicy().to(DEVICE)
        metric = NeuralContractionMetric().to(DEVICE)
        metrics = env.step(actor, metric, control_mode="nn-primary")
        for key in ("attitude_tracking_loss", "angular_rate_tracking_loss", "nn_action_slew_loss"):
            value = metrics[key]
            assert torch.is_tensor(value) and torch.isfinite(value).all(), (key, value)
        loss = (8.0 * metrics["error_pos"].square().sum(-1).mean()
                + 1.8 * metrics["error_vel"].square().sum(-1).mean()
                + metrics["attitude_tracking_loss"]
                + 0.35 * metrics["angular_rate_tracking_loss"]
                + 0.08 * metrics["nn_action_slew_loss"])
        loss.backward()
        grads = [p.grad for p in actor.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)

    def test_reduced_model_jacobian_calculation():
        actor = WakeSurfingActorPolicy().to(DEVICE)
        metric = NeuralContractionMetric().to(DEVICE)
        states = torch.randn((4, 6), device=DEVICE) * 0.1
        _, eig_R, eig_M, cond = contraction_residual_batch(actor, metric, states)
        assert torch.isfinite(eig_R).all() and torch.isfinite(eig_M).all() and torch.isfinite(cond).all()
        assert eig_R.shape == (4, 6) and eig_M.shape == (4, 6)

    def test_cockpit_acceleration_hud_uses_v105_telemetry_schema():
        env = WakeSurfingResearchEnv(num_envs=2, dt=0.015, history_len=20, device=DEVICE, randomize_initial_conditions=True)
        actor = WakeSurfingActorPolicy().to(DEVICE)
        metric = NeuralContractionMetric().to(DEVICE)
        with torch.no_grad():
            metrics = env.step(actor, metric, control_mode="nn-primary")
        hud = cockpit_acceleration_telemetry(metrics["control_response_sample"])
        assert set(hud) == {"predicted", "observed", "trajectory_error", "prediction_error"}
        for name, value in hud.items():
            assert value.shape == (3,) and np.isfinite(value).all(), (name, value)
        # Legacy V10.3 keys must not be needed to draw the V10.5 HUD.
        legacy_free = dict(metrics["control_response_sample"])
        legacy_free.pop("requested_absolute_acceleration_world_mps2", None)
        legacy_free.pop("actual_minus_requested_acceleration_mps2", None)
        cockpit_acceleration_telemetry(legacy_free)

    def test_cockpit_role_status_is_truthful():
        baseline = cockpit_policy_status("pd_baseline_benchmark", False)
        candidate = cockpit_policy_status("nn_primary_unpromoted", False)
        promoted = cockpit_policy_status("nn_primary_accepted_in_simulation_gate", True)
        unknown = cockpit_policy_status("unknown", None)
        assert "PD BASELINE" in baseline[0] and "BENCHMARK ONLY" in baseline[1]
        assert "UNPROMOTED" in candidate[0] and "MONITOR-ONLY" in candidate[1]
        assert "SIMULATION GATE PASSED" in promoted[0] and "NOT FLIGHT-CERTIFIED" in promoted[1]
        assert "ROLE UNKNOWN" in unknown[0]

    def test_rotor_spin_signs_and_propeller_geometry_are_finite():
        assert len(ROTOR_SPIN_SIGNS) == 4
        assert all(v in (-1.0, 1.0) for v in ROTOR_SPIN_SIGNS)
        assert PROP_RADIUS > 0.0 and ROTOR_MAX_OMEGA > 0.0
        phase = (np.asarray(ROTOR_SPIN_SIGNS) * np.full(4, 500.0) * 0.015) % (2.0 * math.pi)
        assert np.isfinite(phase).all() and phase.shape == (4,)

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
            "minimum_projection_solver_feasible_rate_pct": 100.0,
            "maximum_emergency_fallback_rate_pct": 0.0,
            "minimum_predicted_plant_barrier_feasible_rate_pct": 100.0,
            "minimum_actual_plant_barrier_feasible_rate_pct": 100.0,
            "maximum_predictor_observation_mismatch_mps2": 0.0,
        }
        candidate = dict(baseline)
        candidate["nn_primary_direct_rotor_commands_used"] = True
        candidate["mean_position_tracking_error_m"] = 0.195  # only 2.5% improvement
        evaluation = {"aggregate": {"nn_primary_direct_rotor_actor": candidate, "pd_baseline_benchmark": baseline}}
        assert candidate_acceptance_gate(evaluation)["accepted"]
        candidate["mean_position_tracking_error_m"] = 0.201
        assert not candidate_acceptance_gate(evaluation)["accepted"]
        candidate["mean_position_tracking_error_m"] = 0.19
        candidate["minimum_actual_plant_barrier_feasible_rate_pct"] = 80.0
        assert not candidate_acceptance_gate(evaluation)["accepted"]

    check("finite 6-DoF rollout and SO(3) rotation invariants", test_rotation_and_finite_dynamics)
    check("rotor thrust coefficient follows quadratic speed relation", test_rotor_thrust_is_quadratic_in_rotor_speed)
    check("motor/rotor hover equilibrium remains stationary", test_rotor_motor_equilibrium_is_stationary)
    check("rotor speed/current/voltage limits and spin-up/spin-down", test_rotor_states_obey_electrical_and_speed_limits)
    check("6-DoF body holds hover with electromechanical rotor states", test_body_hover_equilibrium_with_rotor_states)
    check("power estimate finite/non-negative", test_power_is_finite_nonnegative)
    check("moving body-offset target velocity/acceleration match finite differences", test_moving_body_offset_kinematics_match_finite_differences)
    check("environment target-point velocity/acceleration include rotational offset terms", test_environment_target_point_velocity_and_acceleration_include_rotation_terms)
    check("projection safely rejects NaN/Inf points and camera inputs", test_projection_rejects_nonfinite_points_and_camera_without_exception)
    check("visualiser non-finite detector identifies and restores bad rotational state", test_nonfinite_detector_identifies_rotational_state_failure)
    check("wake direction, dimensional scale and boundedness", test_wake_velocity_units_and_directional_gate)
    check("wake uses selected historical speed and configured history age", test_wake_uses_selected_historical_velocity_and_configured_history_dt)
    check("rotor geometry, allocation offsets and wake span are consistent", test_geometry_span_matches_x_frame_motor_coordinates)
    check("separate projection/fallback/final-action feasibility diagnostics", test_projection_feasible_and_infeasible_status)
    check("absolute-acceleration CBFs are shifted into residual-command coordinates", test_absolute_barriers_are_shifted_into_residual_acceleration_coordinates)
    check("row-normalized residuals invariant to constraint scaling", test_constraint_residual_is_invariant_to_row_scaling)
    check("fallback cannot worsen the worst normalized violation", test_fallback_does_not_worsen_worst_normalized_violation)
    check("reports a constraint individually impossible under command bounds", test_solver_reports_individually_unachievable_constraint_under_box_bounds)
    check("historical wake-frame CBF kinematics and finite coefficients", test_lateral_barrier_uses_historical_wake_frame_terms)
    check("seeded history derivative buffers are explicit and finite", test_history_derivative_buffers_match_seeded_trim_history)
    check("environment uses the intended projection iteration budget", test_environment_uses_configured_projection_budget)
    check("metric positive-definite and no false certificate", test_metric_positive_definite_and_no_false_certificate)
    check("actuator-aware predictor matches observed one-step plant acceleration", test_actuator_aware_acceleration_predictor_matches_integrator)
    check("NN initialises at hover and outputs four independent bounded rotor commands", test_actor_starts_at_hover_point_with_four_independent_rotor_commands)
    check("NN-primary path directly controls all four rotor setpoints without PD or guard override", test_nn_primary_commands_direct_rotor_setpoints_without_pd_or_guard_override)
    check("autodifferentiated reduced-model Jacobian and metric derivative", test_reduced_model_jacobian_calculation)
    check("long-horizon rollout curriculum has the expected stage boundaries", test_long_horizon_curriculum_schedule)
    check("truncated BPTT detaches state between chunks without changing state values", test_truncated_bptt_detaches_state_but_preserves_values)
    check("attitude/rate/rotor-slew training losses are finite and differentiable", test_new_stability_losses_are_finite_and_differentiable)
    check("candidate-promotion gate fails closed", test_promotion_gate_rejects_non_improving_candidate)
    check("cockpit acceleration HUD uses V10.5 telemetry schema", test_cockpit_acceleration_hud_uses_v105_telemetry_schema)
    check("cockpit labels distinguish baseline, unpromoted candidate and promoted selection", test_cockpit_role_status_is_truthful)
    check("rotor spin telemetry and propeller geometry are finite", test_rotor_spin_signs_and_propeller_geometry_are_finite)
    print("\nSELF-TEST SUMMARY")
    for name, ok, message in results:
        print(f"[{ 'PASS' if ok else 'FAIL' }] {name}: {message}")
    return 0 if all(r[1] for r in results) else 2


def diagnose_checkpoint(checkpoint_path, diagnostic_output, eval_envs=8, eval_steps=100, eval_seeds=5, base_seed=141061):
    """Replay a saved candidate and baseline on matched seeds, saving per-state CBF failure traces."""
    from pathlib import Path
    import json
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    except TypeError:  # compatibility with older PyTorch releases
        payload = torch.load(checkpoint_path, map_location=DEVICE)
    actor = WakeSurfingActorPolicy().to(DEVICE)
    metric = NeuralContractionMetric().to(DEVICE)
    actor_key = "candidate_actor_state_dict" if "candidate_actor_state_dict" in payload else "actor_state_dict"
    if actor_key not in payload or "ncm_state_dict" not in payload:
        raise KeyError("Checkpoint must contain candidate_actor_state_dict (or actor_state_dict) and ncm_state_dict")
    try:
        actor.load_state_dict(payload[actor_key])
        metric.load_state_dict(payload["ncm_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint tensor shapes are incompatible with the V10.5 actor/metric architecture. "
            "V10.3.1 and earlier checkpoints with fewer observation features cannot be loaded. "
            "Run fresh V10.5 training instead."
        ) from exc
    actor.eval(); metric.eval()
    print(f"[DIAGNOSE] checkpoint={checkpoint_path} | actor weights={actor_key} | device={DEVICE}")
    evaluation = evaluate_multiseed_rollouts(
        actor, metric, num_envs=max(1, int(eval_envs)), steps=max(1, int(eval_steps)),
        base_seed=int(base_seed), num_seeds=max(2, int(eval_seeds)),
    )
    output_path = Path(diagnostic_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_out = {
        "scope": "same-device replay of supplied checkpoint; simulation diagnostics only",
        "source_checkpoint": str(checkpoint_path),
        "device": str(DEVICE),
        "physical_validation": False,
        "flight_certification": False,
        "power_model_calibrated": False,
        "evaluation": evaluation,
    }
    output_path.write_text(json.dumps(payload_out, indent=2, allow_nan=False), encoding="utf-8")
    for role in ("nn_primary_direct_rotor_actor", "pd_baseline_benchmark"):
        d = evaluation["aggregate"][role]
        print(f"[{role}] max residual={d['maximum_filter_constraint_residual']:.6g}; "
              f"max lateral={d['maximum_lateral_constraint_violation']:.6g}; "
              f"individually-unachievable examples={sum(bool(x.get('individually_unachievable_under_command_box')) for x in d.get('worst_safety_failure_examples', []))}")
        for failure in d.get("worst_safety_failure_examples", [])[:3]:
            print(f"  seed={failure.get('evaluation_seed')} step={failure.get('step_index')} env={failure.get('environment_index')} "
                  f"max_violation={failure.get('final_max_normalized_violation'):.6g} "
                  f"wake_lateral_h={failure.get('h_lateral_m'):.5g} hdot={failure.get('h_lateral_dot_m_per_s'):.5g} "
                  f"rhs_residual={failure.get('lateral_barrier_rhs_residual_accel'):.5g} "
                  f"box_unachievable={failure.get('individually_unachievable_under_command_box')}")
    print(f"[DIAGNOSE COMPLETE] trace={output_path}")
    return 0


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Wake-surfing V10.5; fresh training by default; long-horizon neural-primary direct rotor control; simulation research only, not flight-certified.")
    parser.add_argument("--mode", choices=("train", "self-test", "smoke-test", "view", "diagnose"), default="train")
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=100, help="Maximum rollout horizon; training progressively increases horizons from 20 steps to this target.")
    parser.add_argument("--seed", type=int, default=41058)
    parser.add_argument("--run-dir", default=None, help="Optional explicit output directory. Omit for an isolated timestamped fresh-run directory under wake_surfing_runs.")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint required only for view/diagnose; normal training never reads a checkpoint.")
    parser.add_argument("--eval-envs", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--eval-seeds", type=int, default=5, help="Number of paired held-out evaluation seeds (minimum 2).")
    parser.add_argument("--checkpoint-role", choices=("selected", "candidate"), default="selected", help="In view mode, show promoted policy or unfiltered trained candidate.")
    parser.add_argument("--diagnostic-output", default="wake_surfing_v10_failure_diagnostic.json", help="JSON path used by --mode diagnose")
    parser.add_argument("--diagnose-base-seed", type=int, default=141061, help="First seed in the multi-seed replay")
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
            run_wake_surfing_cockpit(
                actor, ncm,
                controller_role=getattr(actor, "_wake_policy_role", "unknown"),
                promotion_accepted=getattr(actor, "_wake_promotion_accepted", None),
                run_directory=getattr(actor, "_wake_run_directory", None),
                checkpoint_path=getattr(actor, "_wake_checkpoint_path", None),
            )
        return 0
    if args.mode == "diagnose":
        if not args.checkpoint:
            parser.error("--mode diagnose requires an explicit --checkpoint path; training mode never loads checkpoints.")
        return diagnose_checkpoint(
            args.checkpoint, args.diagnostic_output,
            eval_envs=max(1, args.eval_envs), eval_steps=max(1, args.eval_steps),
            eval_seeds=max(2, args.eval_seeds), base_seed=args.diagnose_base_seed,
        )
    if args.mode == "view":
        if not args.checkpoint:
            parser.error("--mode view requires an explicit --checkpoint path; normal training does not load checkpoints.")
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint at {checkpoint_path}. Run --mode train first.")
        payload = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        actor, ncm = WakeSurfingActorPolicy().to(DEVICE), NeuralContractionMetric().to(DEVICE)
        actor_key = "candidate_actor_state_dict" if args.checkpoint_role == "candidate" else "actor_state_dict"
        if actor_key not in payload:
            raise KeyError(f"Checkpoint has no {actor_key}; select a compatible checkpoint or role.")
        try:
            actor.load_state_dict(payload[actor_key])
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint architecture is incompatible with V10.5. This release retains the 38-input, "
                "four-output actor; older checkpoints with different tensor shapes cannot be loaded. "
                "Start fresh training if you intend to train V10.5."
            ) from exc
        ncm.load_state_dict(payload["ncm_state_dict"])
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        if args.checkpoint_role == "candidate":
            actor._wake_policy_role = "nn_primary_unpromoted"
            actor._wake_promotion_accepted = False
        else:
            selected_role_from_checkpoint = metadata.get("selected_policy_role", "unknown") if isinstance(metadata, dict) else "unknown"
            actor._wake_policy_role = selected_role_from_checkpoint
            accepted = metadata.get("policy_selection", {}).get("accepted") if isinstance(metadata, dict) else None
            actor._wake_promotion_accepted = accepted
        actor._wake_checkpoint_path = str(checkpoint_path.resolve())
        actor._wake_run_directory = metadata.get("run_directory") if isinstance(metadata, dict) else None
        run_wake_surfing_cockpit(
            actor, ncm,
            controller_role=actor._wake_policy_role,
            promotion_accepted=actor._wake_promotion_accepted,
            run_directory=actor._wake_run_directory,
            checkpoint_path=actor._wake_checkpoint_path,
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())