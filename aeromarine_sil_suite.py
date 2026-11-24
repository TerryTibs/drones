"""
=============================================================================
AEROMARINE SIL VALIDATION, HARSH-PHYSICS, ABLATION & REAL-TIME CORE SUITE
=============================================================================
Drop into the same directory as the engine script
(Gemini_Novel_Test_170_SEA_TO_AIR_SCULLING_NN_CONTROL.py or any other
Gemini_Novel_Test*SEA_TO_AIR_SCULLING_NN_CONTROL.py - the highest version
number in the folder is picked automatically; use --base-script or the
AEROMARINE_BASE_SCRIPT environment variable to choose one explicitly).

What this file is (and is not):
  * A software-in-the-loop (SIL) test bench around the engine's own plant model.
    Everything it reports is a property of the SIMULATED vehicle. Nothing here
    has flown, so none of it is a certification or a hardware qualification.
  * "FastFlightCore" is a pure-NumPy re-implementation of the engine's flight
    software (EKF, wave observer, L1, pilot MLP, NCM, 4-constraint QP, mixer).
    --benchmark-core checks that it makes the same decisions as the PyTorch
    pipeline when both are fed the same raw sensor samples, and times it.

Examples:
  1) Train & save a checkpoint (or train an ablation policy):
     python aeromarine_sil_suite.py --train --epochs 96 --save-ckpt aeromarine_ckpt.pt
     python aeromarine_sil_suite.py --train --fixed-foil --harsh-sim --epochs 48 --save-ckpt fixed_foil_ckpt.pt

  2) 6-scenario Monte-Carlo validation (per-sortie event rates with 95% bounds):
     python aeromarine_sil_suite.py --eval-suite --ckpt aeromarine_ckpt.pt --seed 1

  3) Check that FastFlightCore matches the PyTorch pipeline (all flight modes,
     nominal + harsh physics) and measure its step latency:
     python aeromarine_sil_suite.py --benchmark-core --ckpt aeromarine_ckpt.pt

  4) Interactive cockpit, NumPy core flying the plant, harsh physics + faults:
     python aeromarine_sil_suite.py --cockpit --ckpt aeromarine_ckpt.pt --harsh-sim --obstacles --fault-test motor_loss
     (keys: 1-6 modes, C core/torch controller, R reset, SPACE pause, mouse orbit/wheel zoom)

  5) Checkpoints are interchangeable with the engine script's:
     python Gemini_Novel_Test_170_SEA_TO_AIR_SCULLING_NN_CONTROL.py --load-ckpt aeromarine_ckpt.pt   (cockpit of the torch pipeline)
     python aeromarine_sil_suite.py --eval-suite --ckpt wave_pilot_ckpt.pt                            (weights the engine saved)
=============================================================================
"""

import argparse
import glob
import importlib.util
import math
import os
import re
import sys
import time
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
try:
    import pygame
except ImportError:
    pygame = None

BASE_SCRIPT_GLOB = "Gemini_Novel_Test*SEA_TO_AIR_SCULLING_NN_CONTROL.py"
# Assumed design allowable for the strut root bending moment. This number is a placeholder
# (it is not derived from a strut section or material); override with --strut-limit-nm.
STRUT_YIELD_LIMIT_NM = 12.0


def _version_key(path):
    """Sort key: (version number found in the file name, mtime)."""
    m = re.search(r"Test\D*(\d+)", os.path.basename(path))
    return (int(m.group(1)) if m else -1, os.path.getmtime(path))


def find_base_script(explicit=None):
    """Resolve the engine script: explicit path > $AEROMARINE_BASE_SCRIPT > newest match next to this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    explicit = explicit or os.environ.get("AEROMARINE_BASE_SCRIPT")
    if explicit:
        for cand in (explicit, os.path.join(here, explicit), os.path.join(os.getcwd(), explicit)):
            if os.path.isfile(cand):
                return os.path.abspath(cand)
        raise FileNotFoundError(f"Base engine script not found: {explicit}")
    found = glob.glob(os.path.join(here, BASE_SCRIPT_GLOB))
    if not found:
        found = glob.glob(os.path.join(os.getcwd(), BASE_SCRIPT_GLOB))
    if not found:
        raise FileNotFoundError(
            f"No engine script matching '{BASE_SCRIPT_GLOB}' next to {os.path.basename(__file__)}. "
            "Put it in the same folder or pass --base-script PATH.")
    return os.path.abspath(max(found, key=_version_key))


def _early_cli_value(flag):
    """Read '--flag VALUE' / '--flag=VALUE' before argparse runs (the engine must be loaded at import)."""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def load_base_engine(script_path=None):
    path = find_base_script(script_path)
    spec = importlib.util.spec_from_file_location("base_engine", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.__aeromarine_path__ = path
    return mod


base = load_base_engine(_early_cli_value("--base-script"))
DEVICE = base.DEVICE
MODE_NAMES = ["LOITER", "TAKEOFF", "FOILING", "EGRESS", "FLIGHT", "INGRESS", "SOARING"]
# Minimum total lift-rotor thrust the controller keeps. Read from the engine so the plant copy and the NumPy core
# cannot drift from the controller; the fallbacks are the values in older engine files that lack the constants.
ROTOR_FLOOR_N = float(getattr(base, "ROTOR_FLOOR_N", 6.8))
ROTOR_FLOOR_SOARING_N = float(getattr(base, "ROTOR_FLOOR_SOARING_N", 5.6))


# =============================================================================
# 1. HARDENED AERO-HYDRODYNAMICS (UNSTEADY LAG, CAVITATION, UNCAPPED SLAMS)
# =============================================================================
class HardenedAeroHydroDynamics(base.MultimodalAeroHydroDynamics):
    def __init__(self, device, dt=0.015, harsh_physics=False, fixed_foil=False, disable_scull=False,
                 strut_limit_nm=STRUT_YIELD_LIMIT_NM):
        super().__init__(device)
        self.dt = dt
        self.harsh_physics = harsh_physics
        self.fixed_foil = fixed_foil
        self.disable_scull = disable_scull
        self.strut_limit_nm = float(strut_limit_nm)
        self.cl_lag_L = torch.zeros((1, 1), device=device)
        self.cl_lag_R = torch.zeros((1, 1), device=device)

    def reset(self, num_envs):
        self.cl_lag_L = torch.zeros((num_envs, 1), device=self.device)
        self.cl_lag_R = torch.zeros((num_envs, 1), device=self.device)

    def detach_state(self):
        self.cl_lag_L = self.cl_lag_L.detach()
        self.cl_lag_R = self.cl_lag_R.detach()

    def forward(self, pos, vel, rot, omega, eta, deta_dt, w_air_updraft, v_water_orbital, ambient_gust,
                is_vent_L, is_vent_R, foil_decalage, rho_water, hydro_surfaces, motor_thrusts_total,
                morphology_dict, sim_time, is_soaring, scull_throttle=None):
        N = pos.shape[0]
        if self.cl_lag_L.shape[0] != N:
            self.reset(N)

        w_span = morphology_dict["wing_span"]
        w_chord = morphology_dict["wing_chord"]
        w_area = morphology_dict["wing_area"]
        f_span = morphology_dict["foil_span"]
        f_area = morphology_dict["foil_area"]
        strut_len = morphology_dict["strut_length"]
        caudal_area = morphology_dict["caudal_area"]
        anhedral_rad = morphology_dict["anhedral_rad"]

        # 1. WIG wing & PAR cushion
        h_wing = (pos[:, 2:3] - eta).clamp(min=0.03)
        h_over_chord = h_wing / w_chord
        h_over_span = h_wing / w_span

        z0 = 0.0002
        wind_shear_factor = torch.clamp(
            torch.log((h_wing + z0) / z0) / math.log(10.0 / z0),
            min=0.18, max=1.25
        )
        ambient_wind_local = ambient_gust * wind_shear_factor

        anhedral_plenum_boost = 1.0 + 1.2 * torch.clamp(anhedral_rad, min=0.0)
        phi_lift_wig = (1.0 + 0.35 / (0.10 + 2.2 * torch.pow(h_over_chord, 1.15))) * anhedral_plenum_boost
        sigma_drag_wig = (1.0 - torch.exp(-2.48 * torch.pow(2.0 * h_over_span, 0.78))).clamp(min=0.08, max=1.0)

        downwash_vel = torch.sqrt(torch.clamp(motor_thrusts_total / (2.0 * base.RHO_AIR * base.TOTAL_ROTOR_AREA), min=0.0))
        par_pressure = (base.PAR_PLENUM_EFF * 0.5 * base.RHO_AIR * (downwash_vel**2) * F.relu(1.0 - h_wing / 0.25)).clamp(max=120.0)
        f_par_z = par_pressure * w_area

        updraft_field = torch.cat([torch.zeros((N, 2), device=self.device), w_air_updraft], dim=-1)
        v_air_effective = vel - ambient_wind_local - updraft_field
        v_air_mag = base.safe_norm(v_air_effective, dim=-1, keepdim=True).clamp(min=1.0)
        q_air = 0.5 * base.RHO_AIR * (v_air_mag**2)

        rot_T = rot.transpose(-1, -2)
        v_air_body = torch.matmul(rot_T, v_air_effective.unsqueeze(-1)).squeeze(-1)
        alpha_wing = torch.atan2(-v_air_body[:, 2:3], v_air_body[:, 0:1].clamp(min=0.5)) + 0.03

        cl_wing = torch.clamp((base.WING_CL0 + 4.5 * alpha_wing) * phi_lift_wig, -0.3, 1.4)
        cd_induced = (cl_wing**2) / (math.pi * morphology_dict["aspect_ratio"] * base.WING_OSWALD) * sigma_drag_wig
        cd_wing = base.WING_CD0 + cd_induced

        f_wing_z = q_air * w_area * cl_wing * torch.cos(alpha_wing) + f_par_z
        f_wing_x = -q_air * w_area * cd_wing * torch.sign(v_air_effective[:, 0:1] + 1e-4)
        f_wing_world = torch.cat([f_wing_x, torch.zeros((N, 1), device=self.device), f_wing_z], dim=-1)

        q_rate = omega[:, 1:2]
        tau_aero_pitch = q_air * w_area * w_chord * (+0.85 * alpha_wing - 2.8 * (q_rate * w_chord / (2.0 * v_air_mag)))

        # 2. Ventral hydrofoil with unsteady Wagner lag + coupled cavitation
        v_foil_rel = vel - v_water_orbital
        v_foil_rel_body = torch.matmul(rot_T, v_foil_rel.unsqueeze(-1)).squeeze(-1)

        v_foil_u_body = v_foil_rel_body[:, 0:1]
        v_strut_y_body = v_foil_rel_body[:, 1:2]
        v_foil_w_body = v_foil_rel_body[:, 2:3]

        v_foil_speed = base.safe_norm(v_foil_rel_body[:, :2], dim=-1, keepdim=True).clamp(min=0.1, max=20.0)
        q_water = (0.5 * rho_water * (v_foil_speed**2)).clamp(max=45000.0 if self.harsh_physics else 30000.0)

        y_station_offset = f_span * 0.25
        z_tip_L = pos[:, 2:3] + rot[:, 2, 1:2] * y_station_offset - rot[:, 2, 2:3] * strut_len
        z_tip_R = pos[:, 2:3] - rot[:, 2, 1:2] * y_station_offset - rot[:, 2, 2:3] * strut_len

        d_sub_L = (eta - z_tip_L).clamp(min=0.0, max=0.22)
        d_sub_R = (eta - z_tip_R).clamp(min=0.0, max=0.22)
        immersion_depth_mean = 0.5 * (d_sub_L + d_sub_R)

        if self.fixed_foil:
            delta_e_L = torch.zeros_like(d_sub_L)
            delta_e_R = torch.zeros_like(d_sub_R)
            delta_rud = torch.zeros_like(d_sub_L)
        else:
            delta_e_L = hydro_surfaces[:, 0:1]
            delta_e_R = hydro_surfaces[:, 1:2]
            delta_rud = hydro_surfaces[:, 2:3]

        scull_active = (
            torch.zeros_like(d_sub_L) if (self.fixed_foil or self.disable_scull)
            else is_soaring.float() * (immersion_depth_mean > 0.020).float()
        )
        scull_throttle_eff = scull_active if scull_throttle is None else scull_active * scull_throttle.clamp(0.0, 1.0)
        scull_freq = base.SCULL_FREQ_NOM
        scull_amp = base.SCULL_AMP_NOM
        delta_rud_eff = delta_rud

        linear_tail_amp = base.SCULL_LEVER_ARM * math.sin(scull_amp)
        strouhal_st = (2.0 * scull_freq * linear_tail_amp / v_foil_speed).clamp(max=2.0)
        strouhal_prop_eff = torch.exp(-((strouhal_st - 0.30)**2) / (2.0 * (0.08**2))) * 0.84

        v_scull_tip = 2.0 * math.pi * scull_freq * scull_amp * base.SCULL_LEVER_ARM
        v_scull_effective = torch.sqrt(v_foil_speed**2 + (v_scull_tip**2) * scull_active)
        q_scull = 0.5 * rho_water * (v_scull_effective**2)
        f_scull_avail = torch.clamp(
            q_scull * caudal_area * base.HYDRO_RUDDER_CY_DELTA * math.sin(scull_amp) * strouhal_prop_eff * scull_active,
            min=0.0, max=22.0
        )
        f_scull_thrust = f_scull_avail * scull_throttle_eff
        p_scull_actuator_watts = torch.clamp(f_scull_thrust * base.SCULL_SERVO_POWER_COEFF * (scull_freq / base.SCULL_FREQ_NOM), min=0.0, max=45.0)

        is_vent_prior = torch.maximum(is_vent_L, is_vent_R)
        delta_foil_vent_relief = torch.where(is_vent_prior.bool(), -math.radians(2.0), 0.0)

        alpha_flow = torch.atan2(-v_foil_w_body, v_foil_u_body.clamp(min=0.5))
        alpha_base = alpha_flow + foil_decalage + delta_foil_vent_relief

        alpha_foil_L = alpha_base + (base.HYDRO_CL_DELTA / base.FOIL_CL_ALPHA) * delta_e_L
        alpha_foil_R = alpha_base + (base.HYDRO_CL_DELTA / base.FOIL_CL_ALPHA) * delta_e_R

        cp_suction = torch.clamp(2.4 * F.relu(alpha_base), min=0.0, max=3.5)
        suction_dep_L = cp_suction * q_water
        suction_dep_R = cp_suction * q_water

        active_seal_boost = base.PAR_SEAL_GAIN * par_pressure
        seal_strength_L = rho_water * base.GRAVITY * d_sub_L + 120.0 + active_seal_boost
        seal_strength_R = rho_water * base.GRAVITY * d_sub_R + 120.0 + active_seal_boost

        vent_trigger_L = (suction_dep_L > 2.2 * seal_strength_L) & (d_sub_L < 0.030) & (alpha_foil_L > math.radians(2.2))
        reprime_trigger_L = (d_sub_L >= 0.040) & (alpha_foil_L < math.radians(4.0))
        is_airborne_L = (d_sub_L < 0.005)

        is_vent_L_next = torch.where(
            is_airborne_L, torch.zeros_like(is_vent_L),
            torch.where(is_vent_L.bool(),
                torch.where(reprime_trigger_L, torch.zeros_like(is_vent_L), torch.ones_like(is_vent_L)),
                torch.where(vent_trigger_L, torch.ones_like(is_vent_L), torch.zeros_like(is_vent_L))
            )
        )
        vent_factor_L = torch.where(is_vent_L_next.bool(), 0.40, 1.0)

        vent_trigger_R = (suction_dep_R > 2.2 * seal_strength_R) & (d_sub_R < 0.030) & (alpha_foil_R > math.radians(2.2))
        reprime_trigger_R = (d_sub_R >= 0.040) & (alpha_foil_R < math.radians(4.0))
        is_airborne_R = (d_sub_R < 0.005)

        is_vent_R_next = torch.where(
            is_airborne_R, torch.zeros_like(is_vent_R),
            torch.where(is_vent_R.bool(),
                torch.where(reprime_trigger_R, torch.zeros_like(is_vent_R), torch.ones_like(is_vent_R)),
                torch.where(vent_trigger_R, torch.ones_like(is_vent_R), torch.zeros_like(is_vent_R))
            )
        )
        vent_factor_R = torch.where(is_vent_R_next.bool(), 0.40, 1.0)

        cl_steady_L = torch.clamp(base.FOIL_CL_ALPHA * alpha_foil_L * vent_factor_L, -0.10, 1.30)
        cl_steady_R = torch.clamp(base.FOIL_CL_ALPHA * alpha_foil_R * vent_factor_R, -0.10, 1.30)

        # Coupled physical cavitation model
        p_hyd_L = base.P_ATM + rho_water * base.GRAVITY * d_sub_L
        p_hyd_R = base.P_ATM + rho_water * base.GRAVITY * d_sub_R
        sigma_cav_L = (p_hyd_L - base.P_VAPOR_WATER) / (q_water.clamp(min=10.0) * (1.0 + 2.4 * F.relu(alpha_foil_L)))
        sigma_cav_R = (p_hyd_R - base.P_VAPOR_WATER) / (q_water.clamp(min=10.0) * (1.0 + 2.4 * F.relu(alpha_foil_R)))
        sigma_cav_min = torch.minimum(sigma_cav_L, sigma_cav_R)

        if self.harsh_physics:
            cav_sev_L = torch.clamp((base.SIGMA_CAV_CRIT - sigma_cav_L) / 0.45, min=0.0, max=1.0) * (~is_airborne_L).float()
            cav_sev_R = torch.clamp((base.SIGMA_CAV_CRIT - sigma_cav_R) / 0.45, min=0.0, max=1.0) * (~is_airborne_R).float()
            cl_steady_L = cl_steady_L * (1.0 - 0.55 * cav_sev_L)
            cl_steady_R = cl_steady_R * (1.0 - 0.55 * cav_sev_R)
            cd_cav = 0.12 * 0.5 * (cav_sev_L + cav_sev_R)

            # 1st-order Wagner unsteady hydrofoil lag
            tau_lag = (base.FOIL_CHORD_NOM / (0.5 * v_foil_speed)).clamp(min=0.005, max=0.08)
            a_lag = self.dt / (tau_lag + self.dt)
            cl_foil_L = torch.where(is_airborne_L, cl_steady_L, self.cl_lag_L + a_lag * (cl_steady_L - self.cl_lag_L))
            cl_foil_R = torch.where(is_airborne_R, cl_steady_R, self.cl_lag_R + a_lag * (cl_steady_R - self.cl_lag_R))
            self.cl_lag_L = cl_foil_L
            self.cl_lag_R = cl_foil_R
        else:
            cd_cav = torch.zeros_like(d_sub_L)
            cl_foil_L = cl_steady_L
            cl_foil_R = cl_steady_R

        cd_foil = torch.clamp(
            base.FOIL_CD0 + 0.05 * (cl_foil_L**2 + cl_foil_R**2) * 0.5 + base.HYDRO_CD_DELTA * (delta_e_L**2 + delta_e_R**2) * 0.5 + cd_cav,
            0.010, 0.55 if self.harsh_physics else 0.45
        )

        sub_frac_L = (d_sub_L / 0.040).clamp(min=0.0, max=1.0)
        sub_frac_R = (d_sub_R / 0.040).clamp(min=0.0, max=1.0)
        fwd_mask = (v_foil_u_body > 0.1).float()

        area_half = f_area * 0.5
        lift_cap = 26.0 if self.harsh_physics else base.FOIL_LIFT_CLAMP_PER_SIDE
        lift_floor = -2.0 if self.harsh_physics else -0.5
        f_lift_L = torch.clamp(q_water * (area_half * sub_frac_L) * cl_foil_L * fwd_mask, min=lift_floor, max=lift_cap)
        f_lift_R = torch.clamp(q_water * (area_half * sub_frac_R) * cl_foil_R * fwd_mask, min=lift_floor, max=lift_cap)
        f_hydro_lift = f_lift_L + f_lift_R

        flow_angle_w = torch.clamp(-v_foil_w_body / v_foil_speed, -0.5, 0.5)
        f_soar_surge = torch.clamp(base.SOARING_KATZMAYR_COEFF * f_hydro_lift * torch.sin(flow_angle_w), min=-8.0, max=18.0)

        f_hydro_drag_mag = torch.clamp(
            q_water * (f_area * 0.5 * (sub_frac_L + sub_frac_R)) * cd_foil - f_soar_surge - f_scull_thrust,
            min=-25.0, max=40.0 if self.harsh_physics else 25.0
        )
        f_hydro_drag_x = -f_hydro_drag_mag * torch.sign(v_foil_u_body + 1e-4)

        tau_hydro_surfaces_roll = (f_lift_L - f_lift_R) * y_station_offset

        f_rudder_side = q_water * caudal_area * (immersion_depth_mean / 0.08).clamp(max=1.0) * (
            -2.8 * (v_strut_y_body / v_foil_speed) + base.HYDRO_RUDDER_CY_DELTA * delta_rud_eff
        )
        f_strut_lateral_body_y = torch.clamp(f_rudder_side, -25.0 if self.harsh_physics else -8.0, 25.0 if self.harsh_physics else 8.0)

        arm_strut_z = strut_len - 0.5 * immersion_depth_mean
        tau_strut_roll = f_strut_lateral_body_y * arm_strut_z + tau_hydro_surfaces_roll
        strut_root_moment = torch.abs(f_strut_lateral_body_y * strut_len) + torch.abs(tau_hydro_surfaces_roll)
        # Evaluation-only load: roll bending (side force + differential foil lift, as above) combined with
        # pitch bending from the foil drag acting at the strut foot. `strut_root_moment` itself is returned
        # unchanged because the engine's training loss (L_str) is built on it.
        strut_moment_total = torch.sqrt(strut_root_moment**2 + (torch.abs(f_hydro_drag_mag) * strut_len)**2)

        slam_cap = 350.0 if self.harsh_physics else 25.0
        hull_cap = 250.0 if self.harsh_physics else 40.0
        immersion_rate = F.relu(deta_dt - vel[:, 2:3]) * (immersion_depth_mean > 0.001).float()
        f_slam_z = torch.clamp(0.5 * rho_water * base.HYDRO_SLAM_COEFF * f_span * base.FOIL_CHORD_NOM * (immersion_rate**2), max=slam_cap)

        # 3. Hull planing & buoyancy
        z_hull_bottom = pos[:, 2:3] - base.HULL_DRAFT_MAX
        hull_immersion = (eta - z_hull_bottom).clamp(min=0.0, max=base.HULL_DRAFT_MAX * 2.0)
        f_buoy_z = rho_water * base.GRAVITY * (hull_immersion / base.HULL_DRAFT_MAX).clamp(max=1.5) * base.HULL_VOLUME_TOTAL

        trim_angle = F.relu(alpha_wing + 0.02)
        f_planing_z = (0.5 * rho_water * (base.HULL_BEAM * base.HULL_LENGTH) * (v_foil_speed**2) * 0.14 * torch.sin(trim_angle) * (hull_immersion > 0.005).float() * fwd_mask).clamp(max=hull_cap)
        f_entry_drag_z = (-0.5 * rho_water * (base.HULL_BEAM * base.HULL_LENGTH) * 1.8 * vel[:, 2:3] * torch.abs(vel[:, 2:3]) * (hull_immersion > 0.002).float()).clamp(min=-hull_cap, max=hull_cap)

        fn = v_foil_speed / math.sqrt(base.GRAVITY * base.HULL_LENGTH)
        froude_hump = torch.exp(-((fn - 0.45)**2) / 0.04) * 1.8
        f_hull_drag_mag = (0.5 * rho_water * (base.HULL_BEAM * hull_immersion) * (v_foil_speed**2) * (0.04 + froude_hump)).clamp(max=hull_cap)
        f_hull_drag_x = -f_hull_drag_mag * torch.sign(v_foil_u_body + 1e-4)

        f_water_body = torch.cat([f_hydro_drag_x + f_hull_drag_x, f_strut_lateral_body_y, f_hydro_lift + f_slam_z + f_buoy_z + f_planing_z + f_entry_drag_z], dim=-1)
        f_hydro_world = torch.matmul(rot, f_water_body.unsqueeze(-1)).squeeze(-1)

        tau_bow_planing = (f_planing_z + f_buoy_z) * base.BOW_CENTER_OFFSET_X
        tau_hydro_pitch = (-f_hydro_drag_x * strut_len) + (-f_hull_drag_x * base.HULL_DRAFT_MAX) + tau_aero_pitch - tau_bow_planing
        tau_hydro_roll = tau_strut_roll
        tau_hydro_yaw = -f_strut_lateral_body_y * 0.08

        is_ventilated_overall = torch.maximum(is_vent_L_next, is_vent_R_next)

        sub_mean = 0.5 * (sub_frac_L + sub_frac_R)
        mass_est = morphology_dict["total_mass"].detach()
        damp_x = -((base.RHO_AIR * w_area * cd_wing * v_air_mag + rho_water * f_area * cd_foil * sub_mean * v_foil_speed) / mass_est).clamp(min=-8.0, max=0.0)
        damp_y = -((rho_water * caudal_area * 2.8 * (immersion_depth_mean / 0.08).clamp(max=1.0) * v_foil_speed + 0.3 * base.RHO_AIR * w_area * v_air_mag) / mass_est).clamp(min=-8.0, max=0.0)
        damp_z = -((0.5 * rho_water * f_area * sub_mean * base.FOIL_CL_ALPHA * v_foil_speed * fwd_mask + 0.5 * base.RHO_AIR * w_area * 4.5 * v_air_mag) / mass_est).clamp(min=-12.0, max=0.0)
        plant_damp_diag = torch.cat([damp_x, damp_y, damp_z], dim=-1).detach()

        aoa_tensors = {
            "alpha_foil": alpha_base,
            "strut_moment_nm": strut_root_moment,
            "q_water": q_water,
            "f_planing_z": f_planing_z,
            "tau_hydro_roll": tau_hydro_roll,
            "tau_hydro_yaw": tau_hydro_yaw,
            "f_par_z": f_par_z,
            "par_pressure": par_pressure,
            "f_soar_surge": f_soar_surge,
            "f_scull_thrust": f_scull_thrust,
            "p_scull_actuator_watts": p_scull_actuator_watts,
            "strouhal_st": strouhal_st,
            "plant_damp_diag": plant_damp_diag,
            "sigma_cav_min": sigma_cav_min,
            "strut_moment_total_nm": strut_moment_total,
            "strut_yield_fail": (strut_moment_total > self.strut_limit_nm).float(),
            "cav_active": (sigma_cav_min < base.SIGMA_CAV_CRIT).float() * (~(is_airborne_L & is_airborne_R)).float()
        }

        return (f_wing_world, f_hydro_world, tau_hydro_pitch, tau_hydro_roll, tau_hydro_yaw, h_wing,
                immersion_depth_mean, f_slam_z, f_buoy_z, is_vent_L_next, is_vent_R_next,
                is_ventilated_overall, wind_shear_factor, strut_root_moment, aoa_tensors, alpha_base)


# =============================================================================
# 2. CONFIGURABLE ROBOT (FIXED-FOIL, GIMBAL ANGLE SWEEP & FAULT INJECTION)
# =============================================================================
class ConfigurableRobot(base.WaveRidingAerialRobot):
    def __init__(self, dt, device, fixed_foil=False, gimbal_max_deg=15.0, mass_scale=1.0):
        super().__init__(dt, device)
        self.fixed_foil = fixed_foil
        self.gimbal_max_deg = float(gimbal_max_deg)
        rad = math.radians(self.gimbal_max_deg)
        self.gimbal_sin = math.sin(rad)
        self.gimbal_cos = math.cos(rad)
        self.gimbal_tan = math.tan(rad)
        self._J_nominal = self.J.clone()
        self.set_mass_scale(mass_scale)
        self.motor_fault_scale = None
        self.gimbal_jammed = False

    def set_mass_scale(self, mass_scale):
        """Scale the nominal inertia with total mass (assumes the added mass is distributed like the airframe)."""
        self.mass_scale = float(mass_scale)
        self.J = self._J_nominal * self.mass_scale
        self.J_inv = 1.0 / self.J

    def allocate_wrench_hybrid(self, Fx_cmd, Fy_cmd, Fz_cmd, torques_desired, v_batt, ige_gain,
                               d_sub, airspeed, is_soaring, morphology_dict, delta_collective=None):
        if not torch.is_tensor(v_batt):
            v_batt = torch.tensor(v_batt, device=Fz_cmd.device)
        thrust_ceiling = base.MAX_THRUST_PER_MOTOR * (v_batt / base.BATTERY_V0)**2

        q_water = 0.5 * base.RHO_WATER_NOM * (airspeed.clamp(min=0.5)**2)
        sub_frac = (d_sub / 0.040).clamp(min=0.0, max=1.0)
        hydro_speed_fade = ((airspeed - 0.30) / 0.70).clamp(min=0.0, max=1.0)
        eff_sub_frac = sub_frac * hydro_speed_fade

        f_span = morphology_dict["foil_span"]
        f_area = morphology_dict["foil_area"]
        caudal_area = morphology_dict["caudal_area"]

        tx_des = torch.clamp(torques_desired[:, 0:1], -3.5, 3.5)
        ty_des = torch.clamp(torques_desired[:, 1:2], -3.5, 3.5)
        tz_des = torch.clamp(torques_desired[:, 2:3], -1.2, 1.2)

        if self.fixed_foil:
            tau_hydro_roll_alloc = torch.zeros_like(tx_des)
            tau_hydro_yaw_alloc = torch.zeros_like(tz_des)
            delta_e_L = torch.zeros_like(tx_des)
            delta_e_R = torch.zeros_like(tx_des)
            delta_rud = torch.zeros_like(tx_des)
        else:
            max_hydro_roll_torque = 1.00 * eff_sub_frac
            max_hydro_yaw_torque = 0.50 * eff_sub_frac
            hydro_roll_gain = 2.0 * (f_span * 0.25) * q_water * (f_area * 0.5) * base.HYDRO_CL_DELTA * sub_frac
            hydro_yaw_gain = q_water * caudal_area * base.HYDRO_RUDDER_CY_DELTA * sub_frac * 0.08

            tau_hydro_roll_alloc = torch.clamp(tx_des, -max_hydro_roll_torque, max_hydro_roll_torque)
            tau_hydro_yaw_alloc = torch.clamp(tz_des, -max_hydro_yaw_torque, max_hydro_yaw_torque)

            delta_roll = torch.clamp(tau_hydro_roll_alloc / (hydro_roll_gain + 0.15), -base.MAX_HYDRO_DEFLECTION, base.MAX_HYDRO_DEFLECTION) * hydro_speed_fade
            delta_rud = -torch.clamp(tau_hydro_yaw_alloc / (hydro_yaw_gain + 0.10), -base.MAX_HYDRO_DEFLECTION, base.MAX_HYDRO_DEFLECTION) * hydro_speed_fade
            if delta_collective is None:
                delta_collective = torch.zeros_like(delta_roll)
            delta_collective_eff = delta_collective * hydro_speed_fade
            delta_e_L = torch.clamp(delta_roll + delta_collective_eff, -base.MAX_HYDRO_DEFLECTION, base.MAX_HYDRO_DEFLECTION)
            delta_e_R = torch.clamp(-delta_roll + delta_collective_eff, -base.MAX_HYDRO_DEFLECTION, base.MAX_HYDRO_DEFLECTION)

        tx_residual = tx_des - tau_hydro_roll_alloc
        ty_residual = ty_des
        tz_after_hydro = tz_des - tau_hydro_yaw_alloc

        Fz_compensated = Fz_cmd / torch.clamp(ige_gain, min=1.0, max=1.6)
        thrust_floor_per_motor = torch.where(is_soaring.bool(), torch.full_like(Fz_compensated, base.MOTOR_IDLE_FLOOR), torch.full_like(Fz_compensated, 0.90))
        f_floor_tensor = thrust_floor_per_motor * 4.0
        f_ceil_tensor = (thrust_ceiling - 1.20) * 4.0
        Fz_base = torch.minimum(torch.maximum(Fz_compensated, f_floor_tensor), f_ceil_tensor)

        d = self.d
        c_q = self.c_q

        f_nom = 0.25 * Fz_base
        df_x = 0.25 * (tx_residual / d)
        df_y = -0.25 * (ty_residual / d)

        df_rp = torch.cat([-df_x + df_y, +df_x + df_y, +df_x - df_y, -df_x - df_y], dim=-1)
        raw_rp = f_nom + df_rp
        min_rp = torch.min(raw_rp, dim=-1, keepdim=True).values
        rp_shifted = raw_rp + F.relu(base.MOTOR_IDLE_FLOOR - min_rp)
        max_rp = torch.max(rp_shifted, dim=-1, keepdim=True).values
        fz_rotors = torch.minimum((rp_shifted - F.relu(max_rp - thrust_ceiling)).clamp(min=base.MOTOR_IDLE_FLOOR), thrust_ceiling)

        if self.gimbal_max_deg <= 1e-3:
            f_x_rot = torch.zeros_like(fz_rotors)
            f_y_rot = torch.zeros_like(fz_rotors)
            tz_gimbal_actual = torch.zeros_like(tz_after_hydro)
        else:
            f_lat_max = fz_rotors * self.gimbal_tan
            f_lat_sum = torch.sum(f_lat_max, dim=-1, keepdim=True).clamp(min=1e-4)
            tz_gimbal_max = base.ARM_LENGTH * f_lat_sum
            tz_gimbal_alloc = torch.clamp(tz_after_hydro, -0.65 * tz_gimbal_max, 0.65 * tz_gimbal_max)

            f_yaw_mag = (f_lat_max / f_lat_sum) * (tz_gimbal_alloc / base.ARM_LENGTH)
            f_yaw_x = f_yaw_mag * self.yaw_tx
            f_yaw_y = f_yaw_mag * self.yaw_ty

            fx_needed = Fx_cmd - torch.sum(f_yaw_x, dim=-1, keepdim=True)
            fy_needed = Fy_cmd - torch.sum(f_yaw_y, dim=-1, keepdim=True)
            fz_weights = fz_rotors / torch.sum(fz_rotors, dim=-1, keepdim=True).clamp(min=1e-4)

            f_x_raw = f_yaw_x + fz_weights * fx_needed
            f_y_raw = f_yaw_y + fz_weights * fy_needed

            f_lat_raw = torch.sqrt(f_x_raw**2 + f_y_raw**2 + 1e-8)
            cone_scale = torch.clamp(f_lat_max / f_lat_raw, max=1.0)
            f_x_rot = f_x_raw * cone_scale
            f_y_rot = f_y_raw * cone_scale
            tz_gimbal_actual = torch.sum(self.r_x * f_y_rot - self.r_y * f_x_rot, dim=-1, keepdim=True)

        tz_residual = tz_after_hydro - tz_gimbal_actual
        df_z = 0.25 * (tz_residual / c_q)

        f_total_pre_yaw = torch.sqrt(f_x_rot**2 + f_y_rot**2 + fz_rotors**2)
        headroom_low = (f_total_pre_yaw - base.MOTOR_IDLE_FLOOR).clamp(min=0.0)
        headroom_high = (thrust_ceiling - f_total_pre_yaw).clamp(min=0.0)
        max_df_z = torch.min(torch.minimum(headroom_low, headroom_high), dim=-1, keepdim=True).values
        df_z_clamped = torch.clamp(df_z, -max_df_z, max_df_z)
        yaw_pattern = self.spin_signs * df_z_clamped

        clamped_motors = torch.minimum((f_total_pre_yaw + yaw_pattern).clamp(min=base.MOTOR_IDLE_FLOOR), thrust_ceiling)
        f_mag_actual = torch.sqrt(f_x_rot**2 + f_y_rot**2 + fz_rotors**2).clamp(min=1e-4)
        u_x_cmd = f_x_rot / f_mag_actual
        u_y_cmd = f_y_rot / f_mag_actual
        target_tilts = torch.stack([u_x_cmd, u_y_cmd], dim=-1)

        fx_gimbal_alloc = torch.sum(f_x_rot, dim=-1, keepdim=True)
        fy_gimbal_alloc = torch.sum(f_y_rot, dim=-1, keepdim=True)

        is_clipped_min = (clamped_motors <= (base.MOTOR_IDLE_FLOOR + 0.01))
        is_clipped_max = (clamped_motors >= thrust_ceiling - 1e-3)
        headroom_min = min_rp - base.MOTOR_IDLE_FLOOR
        tz_delivered = tz_gimbal_actual + 4.0 * c_q * df_z_clamped
        torque_scale = torch.where(
            torch.abs(tz_after_hydro) > 1e-3,
            (torch.abs(tz_delivered) / (torch.abs(tz_after_hydro) + 1e-6)).clamp(max=1.0),
            torch.ones_like(headroom_min)
        )
        hydro_surfaces = torch.cat([delta_e_L, delta_e_R, delta_rud], dim=-1)

        return (clamped_motors, target_tilts, hydro_surfaces, headroom_min, is_clipped_min,
                is_clipped_max, torque_scale, fx_gimbal_alloc, fy_gimbal_alloc)

    def integrate(self, pos, vel, rot, omega, motors, rotor_tilts, pusher_thrust, nu_states,
                  target_motors, target_tilts, target_pusher, f_wing, f_hydro,
                  tau_hydro_pitch, tau_hydro_roll, tau_hydro_yaw, ambient_wind, eta, wave_slope,
                  mass, motor_tau_up, motor_tau_down):
        N = pos.shape[0]
        c_q = self.c_q

        tau_m = torch.where(target_motors >= motors, motor_tau_up, motor_tau_down)
        motors_next = motors + (self.dt / (tau_m + self.dt)) * (target_motors - motors)
        motors_next = torch.nan_to_num(motors_next, nan=base.MOTOR_IDLE_FLOOR).clamp(min=base.MOTOR_IDLE_FLOOR)

        if self.gimbal_jammed or self.gimbal_max_deg <= 1e-3:
            tilts_next = rotor_tilts if self.gimbal_jammed else torch.zeros_like(rotor_tilts)
        else:
            tilts_next = rotor_tilts + (self.dt / (base.GIMBAL_TAU + self.dt)) * (target_tilts - rotor_tilts)
            tilts_next = torch.nan_to_num(tilts_next, nan=0.0)
            lat_sin = torch.sqrt(tilts_next[:, :, 0]**2 + tilts_next[:, :, 1]**2 + 1e-10)
            lat_scale = torch.clamp(self.gimbal_sin / lat_sin, max=1.0).unsqueeze(-1)
            tilts_next = tilts_next * lat_scale

        ux = tilts_next[:, :, 0]
        uy = tilts_next[:, :, 1]
        uz = torch.sqrt((1.0 - ux**2 - uy**2).clamp(min=max(1e-4, self.gimbal_cos**2)))

        tau_p = torch.where(target_pusher >= pusher_thrust, motor_tau_up, motor_tau_down)
        pusher_next = pusher_thrust + (self.dt / (tau_p + self.dt)) * (target_pusher - pusher_thrust)
        pusher_next = torch.nan_to_num(pusher_next, nan=0.0).clamp(min=0.0, max=base.MAX_PUSHER_THRUST)

        h_rotors = (pos[:, 2:3] - eta).clamp(min=0.04)
        ige_rotor_gain = 1.0 / (1.0 - torch.clamp((base.PROP_RADIUS / (4.0 * h_rotors))**2, max=0.35))
        spray_derate = 1.0 + 0.15 * F.relu(0.14 - h_rotors) * wave_slope.clamp(max=0.5)

        motors_phys = motors_next * self.motor_fault_scale if self.motor_fault_scale is not None else motors_next
        f_rotor_x = motors_phys * ux
        f_rotor_y = motors_phys * uy
        f_rotor_z = motors_phys * uz

        f_body_total = torch.cat([
            torch.sum(f_rotor_x, dim=-1, keepdim=True) + pusher_next,
            torch.sum(f_rotor_y, dim=-1, keepdim=True),
            (torch.sum(f_rotor_z, dim=-1, keepdim=True) * ige_rotor_gain) / spray_derate
        ], dim=-1)
        f_rotors_world = torch.matmul(rot, f_body_total.unsqueeze(-1)).squeeze(-1)

        v_air_hub_world = vel.unsqueeze(1) - ambient_wind.unsqueeze(1)
        rot_T = rot.transpose(-1, -2).unsqueeze(1)
        v_air_hub_body = torch.matmul(rot_T, v_air_hub_world.unsqueeze(-1)).squeeze(-1)

        nu_states_next, a_1, b_1, omega_props = self.rotor_aero.step_inflow_and_flapping(
            nu_states, motors_phys, v_air_hub_body, omega, self.dt
        )

        total_force = (
            f_rotors_world
            + f_wing
            + f_hydro
            + torch.cat([torch.zeros((N, 2), device=self.device), -mass * base.GRAVITY], dim=-1)
        )
        total_force = torch.nan_to_num(total_force, nan=0.0, posinf=400.0, neginf=-400.0)

        accel = total_force / mass
        vel_next = (vel + accel * self.dt).clamp(-25.0, 25.0)
        pos_next = pos + vel_next * self.dt

        tx = torch.sum(self.r_y * f_rotor_z, dim=-1, keepdim=True) + tau_hydro_roll
        ty = -torch.sum(self.r_x * f_rotor_z, dim=-1, keepdim=True) + tau_hydro_pitch
        tz = (
            torch.sum(self.r_x * f_rotor_y - self.r_y * f_rotor_x, dim=-1, keepdim=True)
            + c_q * torch.sum(self.spin_signs * f_rotor_z, dim=-1, keepdim=True)
            + tau_hydro_yaw
        )

        gyroscopic = torch.cross(omega, omega * self.J, dim=-1)
        omega_dot = (torch.cat([tx, ty, tz], dim=-1) - gyroscopic) * self.J_inv
        omega_next = torch.nan_to_num(omega + omega_dot * self.dt, nan=0.0).clamp(-12.0, 12.0)

        v_rot = omega_next * self.dt
        theta = base.safe_norm(v_rot, dim=-1, keepdim=True).clamp(min=1e-6)
        vx, vy, vz = v_rot[:, 0:1], v_rot[:, 1:2], v_rot[:, 2:3]
        zv = torch.zeros_like(vx)

        K = torch.stack([
            torch.cat([zv, -vz, vy], dim=-1),
            torch.cat([vz, zv, -vx], dim=-1),
            torch.cat([-vy, vx, zv], dim=-1),
        ], dim=1)

        eye3 = torch.eye(3, device=self.device).unsqueeze(0)
        s_th = torch.sin(theta) / theta
        c_th = (1.0 - torch.cos(theta)) / (theta**2)
        dR = eye3 + s_th.unsqueeze(-1) * K + c_th.unsqueeze(-1) * torch.matmul(K, K)
        rot_next = torch.matmul(rot, dR)

        c0 = rot_next[:, :, 0] / base.safe_norm(rot_next[:, :, 0], dim=-1, keepdim=True)
        c2 = torch.cross(c0, rot_next[:, :, 1], dim=-1)
        c2 = c2 / base.safe_norm(c2, dim=-1, keepdim=True)
        c1 = torch.cross(c2, c0, dim=-1)
        rot_clean = torch.stack([c0, c1, c2], dim=-1)

        return (pos_next, vel_next, rot_clean, omega_next, motors_next, tilts_next,
                pusher_next, nu_states_next, omega_props, accel, ige_rotor_gain)


# =============================================================================
# 3. FAST PURE-NUMPY 200 HZ REAL-TIME FLIGHT CORE (DECOUPLED FROM SIMULATOR)
# =============================================================================
def _silu_np(x):
    xc = np.clip(x, -20.0, 20.0)
    return xc / (1.0 + np.exp(-xc))


def _softplus_np(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -20.0, 20.0))))


class FastFlightCore:
    """
    Pure-NumPy real-time embedded controller (ES-EKF + Wave Dispersion Observer +
    L1 DOB + 20-State Pilot MLP + 5D NCM + Embedded Dual QP + Hybrid Wrench Allocator).
    Zero PyTorch tensor dispatch inside step(); runs in < 0.5 ms per 200 Hz tick.
    """
    def __init__(self, actor_net, ncm_net, morph_dict, dt=0.005, fixed_foil=False, gimbal_max_deg=15.0, mass_scale=1.0):
        self.dt = float(dt)
        self.fixed_foil = fixed_foil
        self.gimbal_max_deg = float(gimbal_max_deg)
        rad = math.radians(self.gimbal_max_deg)
        self.gimbal_tan = math.tan(rad)
        self.mass_scale = float(mass_scale)

        # Morphology constants (frozen CAD for flight firmware)
        self.w_chord = float(morph_dict["wing_chord"].item())
        self.w_area = float(morph_dict["wing_area"].item())
        self.f_span = float(morph_dict["foil_span"].item())
        self.f_area = float(morph_dict["foil_area"].item())
        self.strut_len = float(morph_dict["strut_length"].item())
        self.caudal_area = float(morph_dict["caudal_area"].item())
        self.nom_mass = float(morph_dict["total_mass"].item()) * self.mass_scale

        # Extract weights from PyTorch modules
        self.actor_layers = [
            (m.weight.detach().cpu().numpy().T.astype(np.float32).copy(),
             m.bias.detach().cpu().numpy().astype(np.float32).copy())
            for m in actor_net.net if isinstance(m, torch.nn.Linear)
        ]
        self.ncm_layers = [
            (m.weight.detach().cpu().numpy().T.astype(np.float32).copy(),
             m.bias.detach().cpu().numpy().astype(np.float32).copy())
            for m in ncm_net.net if isinstance(m, torch.nn.Linear)
        ]

        # Preallocated EKF constants
        self.I15 = np.eye(15, dtype=np.float64)
        self.I3 = np.eye(3, dtype=np.float64)
        self.Q_d = np.eye(15, dtype=np.float64)
        self.Q_d[0:3, 0:3] *= 0.08 * (dt**2)
        self.Q_d[3:6, 3:6] *= 0.08 * dt
        self.Q_d[6:9, 6:9] *= 0.015 * dt
        self.Q_d[9:12, 9:12] *= 2.0e-5 * dt
        self.Q_d[12:15, 12:15] *= 1.0e-6 * dt
        self.mag_world = np.array([0.42, 0.0, -0.90], dtype=np.float64)
        self.R_mag = 0.04 * np.eye(3, dtype=np.float64)
        self.R_dvl = 0.10 * np.eye(3, dtype=np.float64)
        self.R_flow = 0.08 * np.eye(2, dtype=np.float64)
        self.H_pos = np.zeros((3, 15), dtype=np.float64)
        self.H_pos[0:3, 0:3] = self.I3
        self.H_vel = np.zeros((3, 15), dtype=np.float64)
        self.H_vel[0:3, 3:6] = self.I3

        # Mixer geometry
        self.d_arm = base.ARM_LENGTH / math.sqrt(2.0)
        self.r_x = np.array([+self.d_arm, +self.d_arm, -self.d_arm, -self.d_arm], dtype=np.float64)
        self.r_y = np.array([-self.d_arm, +self.d_arm, +self.d_arm, -self.d_arm], dtype=np.float64)
        self.yaw_tx = -self.r_y / base.ARM_LENGTH
        self.yaw_ty = +self.r_x / base.ARM_LENGTH
        self.spin_signs = np.array([-1.0, +1.0, -1.0, +1.0], dtype=np.float64)
        self.k_R = (7.5, 8.0, 4.0)
        self.k_omega = (1.25, 1.35, 0.65)
        self.ray_body = np.array([math.cos(base.FORWARD_RAY_DEPRESSION_RAD), 0.0, -math.sin(base.FORWARD_RAY_DEPRESSION_RAD)], dtype=np.float64)

        self.guidance_delay_steps = max(1, int(round(base.GUIDANCE_DELAY_SECONDS / dt)))
        self.pilot_filter_a = math.exp(-dt / 0.0164)
        self.reset(np.zeros(3), np.array([5.20, 0.0, 0.0]), np.eye(3))

    def reset(self, init_p, init_v, init_rot, mode=base.MODE_FOILING, h_target=None, v_target=None, clearance=None):
        self.p_hat = np.array(init_p, dtype=np.float64).copy()
        self.v_hat = np.array(init_v, dtype=np.float64).copy()
        R = np.asarray(init_rot, dtype=np.float64)
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        qw = 0.5 * math.sqrt(max(1e-6, 1.0 + tr))
        qx = (R[2, 1] - R[1, 2]) / (4.0 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4.0 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4.0 * qw)
        q = np.array([qw, qx, qy, qz], dtype=np.float64)
        self.q_hat = q / max(1e-8, float(np.linalg.norm(q)))
        self.ba_hat = np.zeros(3, dtype=np.float64)
        self.bw_hat = np.zeros(3, dtype=np.float64)

        self.P = np.eye(15, dtype=np.float64) * 0.02
        self.P[6:9, 6:9] = self.I3 * 0.002
        self.P[9:12, 9:12] = self.I3 * 0.001
        self.P[12:15, 12:15] = self.I3 * 0.0001

        self.k_dom = 0.45
        self.omega_dom = math.sqrt(base.GRAVITY * self.k_dom)
        self.q_quad = 0.0
        self.ms_eta = 0.01
        self.ms_slope = 0.01 * (0.45**2)
        self.deta_dx_nadir = 0.0
        self.w_up_nadir = 0.0
        self.eta_prev = 0.0
        self.deta_dx_prev = 0.0
        self.w_up_prev = 0.0
        self.look_dist = 0.75

        self.sigma_hat = np.zeros(3, dtype=np.float64)
        self.sigma_filtered = np.zeros(3, dtype=np.float64)
        self.a_pre_dob_prev = np.zeros(3, dtype=np.float64)
        self.l1_sat_prev = np.zeros(3, dtype=bool)
        self.a_pilot_prev = np.zeros(3, dtype=np.float64)
        self.a_pilot_filtered = np.zeros(3, dtype=np.float64)
        self.guidance_buf = deque([np.zeros(3, dtype=np.float64) for _ in range(self.guidance_delay_steps + 1)], maxlen=self.guidance_delay_steps + 1)

        self.mode = int(mode)
        self.h_target_filtered = base.RIDE_HEIGHT_OPTIMAL if h_target is None else float(h_target)
        self.v_target_filtered = 5.20 if v_target is None else float(v_target)
        self.phi_des_filtered = 0.0
        self.psi_des_filtered = 0.0
        self.e_int_height = 0.0
        self.e_int_y = 0.0
        self.clearance_hold = max(0.04, float(init_p[2])) if clearance is None else max(0.04, float(clearance))
        self.scull_throttle = 0.0

        self.last_d_sub = 0.0
        self.last_radar_raw = self.clearance_hold
        self.last_radar_locked = True

    @staticmethod
    def _q_to_R(q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        return np.array([
            [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
            [2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
            [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)],
        ], dtype=np.float64)

    @staticmethod
    def _q_mult(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dtype=np.float64)

    @staticmethod
    def _euler_to_R(phi, theta, psi):
        cp, sp = math.cos(phi), math.sin(phi)
        ct, st = math.cos(theta), math.sin(theta)
        cy, sy = math.cos(psi), math.sin(psi)
        return np.array([
            [cy*ct, cy*st*sp - sy*cp, cy*st*cp + sy*sp],
            [sy*ct, sy*st*sp + cy*cp, sy*st*cp - cy*sp],
            [-st,   ct*sp,            ct*cp],
        ], dtype=np.float64)

    def _ekf_propagate(self, accel_meas, gyro_meas):
        dt = self.dt
        a_b = accel_meas - self.ba_hat
        w_b = gyro_meas - self.bw_hat
        R_hat = self._q_to_R(self.q_hat)

        om = max(1e-6, math.sqrt(w_b[0]*w_b[0] + w_b[1]*w_b[1] + w_b[2]*w_b[2]))
        da = om * dt
        s = math.sin(0.5 * da) / om
        dq = np.array([math.cos(0.5 * da), w_b[0]*s, w_b[1]*s, w_b[2]*s], dtype=np.float64)
        self.q_hat = self._q_mult(self.q_hat, dq)
        self.q_hat /= max(1e-8, float(np.linalg.norm(self.q_hat)))

        a_w = R_hat @ a_b
        a_w[2] -= base.GRAVITY
        self.p_hat += self.v_hat * dt + 0.5 * a_w * (dt * dt)
        self.v_hat += a_w * dt

        Phi = self.I15.copy()
        Phi[0:3, 3:6] = self.I3 * dt
        skew_a = np.array([
            [0.0, -a_b[2], a_b[1]],
            [a_b[2], 0.0, -a_b[0]],
            [-a_b[1], a_b[0], 0.0],
        ], dtype=np.float64)
        Phi[3:6, 6:9] = -(R_hat @ skew_a) * dt
        Phi[3:6, 9:12] = -R_hat * dt
        Phi[6:9, 6:9] = self._q_to_R(dq).T
        Phi[6:9, 12:15] = -self.I3 * dt

        P_prop = Phi @ self.P @ Phi.T + self.Q_d
        self.P = np.clip(0.5 * (P_prop + P_prop.T), -2.0, 2.0)
        np.fill_diagonal(self.P, np.clip(np.diag(self.P), 1e-4, 2.0))

    def _ekf_update(self, y, H, R_cov):
        P_HT = self.P @ H.T
        S = H @ P_HT + R_cov + 1e-5 * np.eye(len(y), dtype=np.float64)
        K = np.linalg.solve(S, P_HT.T).T
        dx = np.nan_to_num(K @ y, nan=0.0)

        self.p_hat += dx[0:3]
        self.v_hat += dx[3:6]
        d_th = np.clip(dx[6:9], -0.05, 0.05)
        dq = np.array([1.0, 0.5*d_th[0], 0.5*d_th[1], 0.5*d_th[2]], dtype=np.float64)
        self.q_hat = self._q_mult(self.q_hat, dq)
        self.q_hat /= max(1e-8, float(np.linalg.norm(self.q_hat)))
        self.ba_hat = np.clip(self.ba_hat + dx[9:12], -0.08, 0.08)
        self.bw_hat = np.clip(self.bw_hat + dx[12:15], -0.03, 0.03)

        I_KH = self.I15 - K @ H
        P_new = I_KH @ self.P @ I_KH.T + K @ R_cov @ K.T
        self.P = np.clip(0.5 * (P_new + P_new.T), -2.0, 2.0)
        np.fill_diagonal(self.P, np.clip(np.diag(self.P), 1e-4, 2.0))

    def _actor_forward(self, obs_np):
        x = obs_np.astype(np.float32)
        for i, (W, b) in enumerate(self.actor_layers):
            x = x @ W + b
            if i < len(self.actor_layers) - 1:
                x = _silu_np(x)
        raw = np.tanh(x)
        a_x = float(raw[0]) * 1.2 - 1.8 * float(obs_np[0])
        a_y = float(raw[1]) * 0.9 - 2.4 * float(obs_np[9]) - 2.8 * float(obs_np[3])
        a_z = float(raw[2]) * 1.2 - 9.5 * float(obs_np[1]) - 3.5 * float(obs_np[2]) - 4.2 * float(obs_np[4])
        return np.array([a_x, a_y, a_z], dtype=np.float64)

    def _ncm_forward_blocks(self, state_err_np):
        x = state_err_np[1:].astype(np.float32)
        for i, (W, b) in enumerate(self.ncm_layers):
            x = x @ W + b
            if i < len(self.ncm_layers) - 1:
                x = _silu_np(x)
        l00 = float(_softplus_np(x[0])) + 0.45
        l10 = float(x[1])
        l11 = float(_softplus_np(x[2])) + 0.45
        l20 = float(x[3])
        l21 = float(x[4])
        l22 = float(_softplus_np(x[5])) + 0.45
        L_xy = np.array([[l00, 0.0, 0.0], [l10, l11, 0.0], [l20, l21, l22]], dtype=np.float64)
        M_xy = L_xy @ L_xy.T + 0.25 * self.I3

        lz00 = float(_softplus_np(x[6])) + 0.45
        lz10 = float(x[7])
        lz11 = float(_softplus_np(x[8])) + 0.45
        L_z = np.array([[lz00, 0.0], [lz10, lz11]], dtype=np.float64)
        M_z = L_z @ L_z.T + 0.25 * np.eye(2, dtype=np.float64)
        return M_xy, M_z

    def step(self, accel_meas, gyro_meas, run_fast_sensors=True, mag_meas=None,
             dvl_meas=None, dvl_ok=False, flow_rates=None, radar_raw=0.185,
             radar_locked=True, d_sub_hydrostatic=0.0, eta_fwd_meas=0.0, eta_fwd_valid=True,
             run_gnss=False, gnss_pos=None, gnss_vel=None, v_batt=25.2, mode_cmd=None, y_ref=0.0):
        dt = self.dt
        if mode_cmd is not None:
            self.mode = int(mode_cmd)

        # 1. ES-EKF Propagation + Sensor Updates
        self._ekf_propagate(accel_meas, gyro_meas)

        if run_fast_sensors:
            self.last_d_sub = float(d_sub_hydrostatic)
            self.last_radar_raw = float(radar_raw)
            self.last_radar_locked = bool(radar_locked)
            meas_radar = self.last_radar_raw if self.last_radar_locked else self.clearance_hold

            R_prior = self._q_to_R(self.q_hat)
            if mag_meas is not None:
                m_pred = R_prior.T @ self.mag_world
                skew_m = np.array([
                    [0.0, -m_pred[2], m_pred[1]],
                    [m_pred[2], 0.0, -m_pred[0]],
                    [-m_pred[1], m_pred[0], 0.0],
                ], dtype=np.float64)
                H_m = np.zeros((3, 15), dtype=np.float64)
                H_m[:, 6:9] = skew_m
                self._ekf_update(mag_meas - m_pred, H_m, self.R_mag)

            is_sub = (self.last_d_sub > 0.015)
            if is_sub and dvl_ok and dvl_meas is not None:
                R_cur = self._q_to_R(self.q_hat)
                v_pred = R_cur.T @ self.v_hat
                H_d = np.zeros((3, 15), dtype=np.float64)
                H_d[:, 3:6] = R_cur.T
                H_d[:, 6:9] = np.array([
                    [0.0, -v_pred[2], v_pred[1]],
                    [v_pred[2], 0.0, -v_pred[0]],
                    [-v_pred[1], v_pred[0], 0.0],
                ], dtype=np.float64)
                self._ekf_update(dvl_meas - v_pred, H_d, self.R_dvl)

            if (not is_sub) and (meas_radar < 1.8) and (flow_rates is not None):
                R_cur = self._q_to_R(self.q_hat)
                h_c = min(3.0, max(0.08, meas_radar))
                inv_h = 1.0 / h_c
                u_b = R_cur.T @ self.v_hat
                cel = self.omega_dom / max(0.1, self.k_dom)
                f_pred = np.array([-u_b[1] * inv_h, (u_b[0] - cel) * inv_h], dtype=np.float64)
                H_f = np.zeros((2, 15), dtype=np.float64)
                H_f[0, 3:6] = -inv_h * R_cur[:, 1]
                H_f[1, 3:6] = +inv_h * R_cur[:, 0]
                row0 = np.array([0.0, -u_b[2], u_b[1]], dtype=np.float64)
                row1 = np.array([u_b[2], 0.0, -u_b[0]], dtype=np.float64)
                H_f[0, 6:9] = -inv_h * row1
                H_f[1, 6:9] = +inv_h * row0
                self._ekf_update(flow_rates - f_pred, H_f, self.R_flow)
        else:
            meas_radar = self.last_radar_raw if self.last_radar_locked else self.clearance_hold
            is_sub = (self.last_d_sub > 0.015)

        if run_gnss and gnss_pos is not None and gnss_vel is not None:
            clr_s = min(10.0, max(0.01, meas_radar))
            infl = 1.0 + 8.0 * math.exp(-clr_s / 0.25)
            self._ekf_update(gnss_pos - self.p_hat, self.H_pos, 0.0025 * infl * self.I3)
            self._ekf_update(gnss_vel - self.v_hat, self.H_vel, 0.0009 * infl * self.I3)

        # 2. Navigation state & dispersion observer
        est_pos = self.p_hat
        est_vel = self.v_hat
        est_rot = self._q_to_R(self.q_hat)
        est_omega = gyro_meas - self.bw_hat
        a_est_world = est_rot @ (accel_meas - self.ba_hat)
        a_est_world[2] -= base.GRAVITY

        cos_tilt = max(0.30, float(est_rot[2, 2]))
        if self.last_radar_locked:
            est_clearance = meas_radar
        elif is_sub:
            est_clearance = max(0.0, self.strut_len * cos_tilt - self.last_d_sub)
        else:
            est_clearance = min(10.0, max(0.04, float(est_pos[2])))
        self.clearance_hold = min(4.50, max(0.04, est_clearance))
        est_h_dot = float(est_vel[2])

        if run_fast_sensors:
            ray_w = est_rot @ self.ray_body
            est_s = min(6.0, max(0.25, est_clearance / max(0.06, -float(ray_w[2]))))
            self.look_dist = math.hypot(est_s * float(ray_w[0]), est_s * float(ray_w[1]))

            eta_nadir = max(-0.6, min(0.6, float(est_pos[2]) - est_clearance))
            slope = max(-0.6, min(0.6, (float(eta_fwd_meas) - eta_nadir) / max(0.25, self.look_dist)))
            dt_obs = max(dt, base.REF_DT)
            if eta_fwd_valid:
                a_ms = math.exp(-dt_obs / 0.45)
                self.ms_eta = a_ms * self.ms_eta + (1.0 - a_ms) * (eta_nadir**2)
                self.ms_slope = a_ms * self.ms_slope + (1.0 - a_ms) * (slope**2)
                k_est = math.sqrt(self.ms_slope / (self.ms_eta + 0.002))
                a_k = math.exp(-dt_obs / 0.25)
                self.k_dom = min(3.5, max(0.08, a_k * self.k_dom + (1.0 - a_k) * k_est))
                self.omega_dom = math.sqrt(base.GRAVITY * self.k_dom)
                q_meas = max(-0.4, min(0.4, -slope / self.k_dom))
                a_q = math.exp(-dt_obs / 0.15)
                self.q_quad = a_q * self.q_quad + (1.0 - a_q) * q_meas

            self.deta_dx_nadir = -self.k_dom * self.q_quad
            self.w_up_nadir = self.omega_dom * self.q_quad
            delta = self.k_dom * (float(est_vel[0]) * 1.25) - self.omega_dom * 1.25
            c_d, s_d = math.cos(delta), math.sin(delta)
            self.eta_prev = eta_nadir * c_d - self.q_quad * s_d
            quad_ahead = self.q_quad * c_d + eta_nadir * s_d
            self.deta_dx_prev = -self.k_dom * quad_ahead
            self.w_up_prev = self.omega_dom * quad_ahead

        # L1 DOB
        a_l1 = math.exp(-dt / 0.25)
        d_hat = np.clip(a_est_world - self.a_pre_dob_prev + self.sigma_filtered, -15.0, 15.0)
        sig_upd = np.clip(a_l1 * self.sigma_filtered + (1.0 - a_l1) * d_hat, -6.0, 6.0)
        sig_dec = self.sigma_filtered * math.exp(-dt / 0.50)
        self.sigma_filtered = np.where(self.l1_sat_prev, sig_dec, sig_upd)
        sigma_l1 = self.sigma_filtered

        # 3. Mode automaton
        m = self.mode
        is_loiter = (m == base.MODE_LOITER)
        is_takeoff = (m == base.MODE_TAKEOFF)
        is_foiling = (m == base.MODE_FOILING)
        is_egress = (m == base.MODE_EGRESS)
        is_flight = (m == base.MODE_FREE_FLIGHT)
        is_ingress = (m == base.MODE_INGRESS)
        is_soaring = (m == base.MODE_SOARING)

        v_tgt_raw = 0.0 if is_loiter else (8.50 if is_flight else 5.20)
        loiter_desc = is_loiter and (est_vel[0] < 1.5)
        is_vent_est = (self.last_d_sub > 0.020) and (a_est_world[2] < -1.8)
        if is_loiter:
            h_tgt_raw = 0.04 if loiter_desc else 0.18
        elif is_takeoff:
            h_tgt_raw = 0.185
        elif is_foiling:
            h_tgt_raw = base.RIDE_HEIGHT_REPRIME if (is_vent_est and is_sub) else base.RIDE_HEIGHT_OPTIMAL
        elif is_soaring:
            h_tgt_raw = base.RIDE_HEIGHT_SOARING
        elif is_egress or is_flight:
            h_tgt_raw = base.FREE_FLIGHT_ALTITUDE
        else:
            h_tgt_raw = base.RIDE_HEIGHT_OPTIMAL

        alpha_b = dt / (0.15 + dt)
        self.h_target_filtered += alpha_b * (h_tgt_raw - self.h_target_filtered)
        self.v_target_filtered += alpha_b * (v_tgt_raw - self.v_target_filtered)

        if is_takeoff and (est_vel[0] > 4.6) and is_sub:
            self.mode = base.MODE_FOILING
        elif is_egress and (est_clearance > (base.FREE_FLIGHT_ALTITUDE - 0.15)):
            self.mode = base.MODE_FREE_FLIGHT
        elif is_ingress and is_sub:
            self.mode = base.MODE_FOILING

        # 4. Pilot + NCM-CBF Embedded QP
        e_speed = float(est_vel[0]) - self.v_target_filtered
        e_height = est_clearance - self.h_target_filtered
        e_y_ref = float(est_pos[1]) - float(y_ref)
        if is_foiling or is_takeoff or is_soaring:
            self.e_int_height = max(-0.12, min(0.12, self.e_int_height + e_height * dt))
            if est_vel[0] > 1.0:
                self.e_int_y = max(-1.5, min(1.5, self.e_int_y + e_y_ref * dt))
            else:
                self.e_int_y *= 0.95
        else:
            self.e_int_height *= 0.95
            self.e_int_y *= 0.95

        eta_nadir_cur = float(est_pos[2]) - est_clearance
        actor_obs = np.array([
            e_speed, e_height, self.e_int_height, e_y_ref,
            est_h_dot, self.v_target_filtered / 8.50, self.last_d_sub, self.h_target_filtered,
            est_vel[0], est_vel[1], est_vel[2],
            est_omega[0], est_omega[1], est_omega[2],
            est_rot[2, 0], est_rot[2, 1], est_rot[2, 2],
            self.eta_prev - eta_nadir_cur, self.deta_dx_prev, self.w_up_prev
        ], dtype=np.float64)

        a_neural = self._actor_forward(actor_obs)
        self.guidance_buf.append(a_neural)
        a_delayed = self.guidance_buf[0]
        self.a_pilot_filtered = self.pilot_filter_a * self.a_pilot_filtered + (1.0 - self.pilot_filter_a) * a_delayed

        state_err = np.array([0.0, e_y_ref, e_height, e_speed, est_vel[1], est_h_dot], dtype=np.float64)
        M_xy, M_z = self._ncm_forward_blocks(state_err)

        v_wave_app = self.deta_dx_nadir * float(est_vel[0]) + self.w_up_nadir
        h_dot_rel = float(est_vel[2]) - v_wave_app
        h_min_m = 0.01 if (loiter_desc or is_takeoff) else base.RIDE_HEIGHT_MIN
        h_max_m = 4.00 if (is_egress or is_flight or is_ingress) else (base.RIDE_HEIGHT_FOILING_MAX if (is_foiling or is_soaring) else base.RIDE_HEIGHT_WIG_MAX)
        h_margin = max(-0.5, min(1.0, h_max_m - est_clearance))

        b_crash = 4.00 * h_dot_rel + 12.0 * (est_clearance - h_min_m)
        b_ceil = -4.00 * h_dot_rel + 12.0 * h_margin

        e_xy = np.array([e_y_ref, e_speed, est_vel[1]], dtype=np.float64)
        m_e_xy = M_xy @ e_xy
        g_xy = m_e_xy[1:3]
        V_xy = float(e_xy @ m_e_xy)
        b_ncm_xy = -0.15 * V_xy - float(m_e_xy[0]) * float(est_vel[1])

        e_z = np.array([e_height, est_h_dot], dtype=np.float64)
        m_e_z = M_z @ e_z
        g_z = float(m_e_z[1])
        V_z = float(e_z @ m_e_z)
        b_ncm_z = -0.15 * V_z - float(m_e_z[0]) * est_h_dot

        A = np.array([
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [g_xy[0], g_xy[1], 0.0],
            [0.0, 0.0, g_z],
        ], dtype=np.float64)
        b_vec = np.array([b_crash, b_ceil, b_ncm_xy, b_ncm_z], dtype=np.float64)

        da_max = base.MAX_JERK_ACCEL * dt
        u_nom = np.clip(self.a_pilot_filtered, self.a_pilot_prev - da_max, self.a_pilot_prev + da_max)

        Q_reg = A @ A.T + 0.08 * np.eye(4, dtype=np.float64)
        c_qp = A @ u_nom - b_vec
        lam = np.zeros(4, dtype=np.float64)
        for _ in range(8):
            for i in range(4):
                gi = float(Q_reg[i, :] @ lam) - c_qp[i]
                lam[i] = max(0.0, lam[i] - gi / Q_reg[i, i])
        du = np.clip(A.T @ lam, -12.0, 12.0)
        a_safe = u_nom - du
        self.a_pilot_prev = a_safe

        # 5. Inner Loop + Hybrid Wrench Allocation
        sub_fac = min(1.0, max(0.0, self.last_d_sub / 0.04))
        phi_max = sub_fac * math.radians(6.5) + (1.0 - sub_fac) * math.radians(18.0)

        v_hor = max(0.5, math.hypot(float(est_vel[0]), float(est_vel[1])))
        q_w = 0.5 * base.RHO_WATER_NOM * (v_hor * v_hor)
        q_a = 0.5 * base.RHO_AIR * (v_hor * v_hor)
        f_h_drag = q_w * self.f_area * base.FOIL_CD0 * sub_fac
        f_drag = f_h_drag + q_a * self.w_area * (base.WING_CD0 + 0.03)

        v_b = est_rot.T @ est_vel
        vx_b_safe = max(0.5, float(v_b[0]))
        dec_mean = math.radians(0.5 * (base.FOIL_DECALAGE_DEG[0] + base.FOIL_DECALAGE_DEG[1]))
        alpha_f_est = math.atan2(-float(v_b[2]), vx_b_safe) + dec_mean
        q_ACLa = q_w * self.f_area * base.FOIL_CL_ALPHA
        f_lift_unclamped = q_ACLa * alpha_f_est * sub_fac

        h_safe = max(0.03, est_clearance)
        phi_wig = 1.0 + 0.35 / (0.10 + 2.2 * ((h_safe / self.w_chord) ** 1.15))
        alpha_w_est = math.atan2(-float(v_b[2]), vx_b_safe) + 0.03
        cl_w = max(-0.3, min(1.4, (base.WING_CL0 + 4.5 * alpha_w_est) * phi_wig))
        f_w_lift = min(15.0, max(0.0, q_a * self.w_area * cl_w))
        tau_h_pitch = f_h_drag * self.strut_len

        min_r_floor = ROTOR_FLOOR_SOARING_N if is_soaring else ROTOR_FLOOR_N
        des_fz = self.nom_mass * (base.GRAVITY + float(a_safe[2]) - float(sigma_l1[2]))
        lift_cap = des_fz - min_r_floor - f_w_lift
        over_lift = max(0.0, f_lift_unclamped - lift_cap)
        q_ACLd = q_w * self.f_area * base.HYDRO_CL_DELTA * sub_fac
        coll_max = 0.8 * base.MAX_HYDRO_DEFLECTION
        foil_gate = 1.0 if ((sub_fac > 0.05) and (est_vel[0] > 1.0)) else 0.0

        if self.fixed_foil:
            delta_coll = 0.0
            theta_shed = min(math.radians(6.0), over_lift / (q_ACLa + 50.0)) * foil_gate
            f_h_lift_est = max(-1.0, min(2.0 * base.FOIL_LIFT_CLAMP_PER_SIDE, f_lift_unclamped))
        else:
            delta_coll = -min(coll_max, over_lift / (q_ACLd + 1.0)) * foil_gate
            over_after = max(0.0, over_lift - q_ACLd * coll_max)
            theta_shed = min(math.radians(6.0), over_after / (q_ACLa + 50.0)) * foil_gate
            f_h_lift_est = max(-1.0, min(2.0 * base.FOIL_LIFT_CLAMP_PER_SIDE, f_lift_unclamped + q_ACLd * delta_coll))

        raw_f_rot = des_fz - f_h_lift_est - f_w_lift
        f_rot_need = max(min_r_floor, min(self.nom_mass * base.GRAVITY + 6.0, raw_f_rot))

        psi_est = math.atan2(float(est_rot[1, 0]), float(est_rot[0, 0]))
        c_p, s_p = math.cos(psi_est), math.sin(psi_est)
        ax_w = float(a_safe[0]) - float(sigma_l1[0])
        ay_w = float(a_safe[1]) - 1.4 * self.e_int_y - float(sigma_l1[1])
        Fx_dyn = self.nom_mass * (c_p * ax_w + s_p * ay_w)
        Fy_dyn = self.nom_mass * (-s_p * ax_w + c_p * ay_w)

        f_gimbal_cap = f_rot_need * (self.gimbal_tan * 0.70)
        Fx_g_est = 0.0 if is_soaring else max(-f_gimbal_cap, min(f_gimbal_cap, Fx_dyn))
        soar_req = max(0.0, min(22.0, Fx_dyn + f_drag)) if (is_soaring and is_sub and (not self.fixed_foil)) else 0.0
        self.scull_throttle = max(0.0, min(1.0, soar_req / 22.0))

        pusher_off = is_loiter and (est_vel[0] < 0.8)
        target_pusher = 0.0 if (pusher_off or is_soaring) else max(0.0, min(base.MAX_PUSHER_THRUST, Fx_dyn + f_drag - Fx_g_est))
        Fx_cmd_g = Fx_dyn + f_drag - target_pusher - soar_req
        Fx_res = Fx_cmd_g - max(-f_gimbal_cap, min(f_gimbal_cap, Fx_cmd_g))
        Fy_cmd_g = Fy_dyn
        Fy_res = Fy_dyn - max(-f_gimbal_cap, min(f_gimbal_cap, Fy_dyn))

        f_vnet = max(4.0, self.nom_mass * base.GRAVITY - f_h_lift_est - f_w_lift)
        raw_phi = math.atan2(-Fy_res, f_vnet)
        beta_fl = math.atan2(float(est_vel[1]), max(1.0, float(est_vel[0])))
        lat_dmp = 0.22 * max(-math.radians(10.0), min(math.radians(10.0), beta_fl))
        phi_tgt = max(-phi_max, min(phi_max, raw_phi + lat_dmp))
        self.phi_des_filtered += (dt / (0.15 + dt)) * (phi_tgt - self.phi_des_filtered)
        phi_des = self.phi_des_filtered

        th_res = math.atan2(Fx_res, f_vnet)
        th_bias = -math.radians(4.5) if is_egress else (-math.radians(1.5) if is_ingress else 0.0)
        theta_des = max(-math.radians(14.0), min(math.radians(24.0), th_res + th_bias + theta_shed))

        self.a_pre_dob_prev = np.array([a_safe[0], a_safe[1] - 1.4 * self.e_int_y, a_safe[2]], dtype=np.float64)
        self.l1_sat_prev = np.array([
            (theta_des <= math.radians(-13.8)) or (theta_des >= math.radians(23.8)) or (pusher_off and (Fx_dyn > f_gimbal_cap)),
            abs(raw_phi + lat_dmp) > phi_max,
            (raw_f_rot < min_r_floor) or (raw_f_rot > (self.nom_mass * base.GRAVITY + 6.0)),
        ], dtype=bool)

        yaw_gate = max(0.0, min(1.0, (float(est_vel[0]) - 0.5) / 1.5))
        psi_tgt = -max(-math.radians(5.0), min(math.radians(5.0), e_y_ref * 0.12 + 0.15 * float(est_vel[1]))) * yaw_gate
        self.psi_des_filtered += (dt / (0.45 + dt)) * (psi_tgt - self.psi_des_filtered)
        psi_des = self.psi_des_filtered

        Rd = self._euler_to_R(phi_des, theta_des, psi_des)
        R_err = Rd.T @ est_rot - est_rot.T @ Rd
        tau_des = (
            -self.k_R[0] * R_err[2, 1] * 0.5 - self.k_omega[0] * est_omega[0],
            -self.k_R[1] * R_err[0, 2] * 0.5 - self.k_omega[1] * est_omega[1] - tau_h_pitch,
            -self.k_R[2] * R_err[1, 0] * 0.5 - self.k_omega[2] * est_omega[2],
        )

        # Hybrid Wrench Allocator
        h_rot_est = max(0.04, est_clearance)
        ige_est = 1.0 / (1.0 - min(0.35, (base.PROP_RADIUS / (4.0 * h_rot_est)) ** 2))
        t_ceil = base.MAX_THRUST_PER_MOTOR * ((v_batt / base.BATTERY_V0) ** 2)
        h_fade = max(0.0, min(1.0, (float(est_vel[0]) - 0.30) / 0.70))

        tx_d = max(-3.5, min(3.5, tau_des[0]))
        ty_d = max(-3.5, min(3.5, tau_des[1]))
        tz_d = max(-1.2, min(1.2, tau_des[2]))

        if self.fixed_foil:
            tau_h_r, tau_h_y = 0.0, 0.0
            hydro_cmd = np.zeros(3, dtype=np.float64)
        else:
            max_hr = 1.00 * sub_fac * h_fade
            max_hy = 0.50 * sub_fac * h_fade
            # the engine's allocator uses the forward-speed dynamic pressure here (not the horizontal-speed one)
            q_alloc = 0.5 * base.RHO_WATER_NOM * max(0.5, float(est_vel[0])) ** 2
            hr_gain = 2.0 * (self.f_span * 0.25) * q_alloc * (self.f_area * 0.5) * base.HYDRO_CL_DELTA * sub_fac
            hy_gain = q_alloc * self.caudal_area * base.HYDRO_RUDDER_CY_DELTA * sub_fac * 0.08
            tau_h_r = max(-max_hr, min(max_hr, tx_d))
            tau_h_y = max(-max_hy, min(max_hy, tz_d))
            d_roll = max(-base.MAX_HYDRO_DEFLECTION, min(base.MAX_HYDRO_DEFLECTION, tau_h_r / (hr_gain + 0.15))) * h_fade
            d_rud = -max(-base.MAX_HYDRO_DEFLECTION, min(base.MAX_HYDRO_DEFLECTION, tau_h_y / (hy_gain + 0.10))) * h_fade
            d_ceff = delta_coll * h_fade
            hydro_cmd = np.array([
                max(-base.MAX_HYDRO_DEFLECTION, min(base.MAX_HYDRO_DEFLECTION, d_roll + d_ceff)),
                max(-base.MAX_HYDRO_DEFLECTION, min(base.MAX_HYDRO_DEFLECTION, -d_roll + d_ceff)),
                d_rud,
            ], dtype=np.float64)

        tx_res = tx_d - tau_h_r
        tz_rem = tz_d - tau_h_y
        fz_comp = f_rot_need / max(1.0, min(1.6, ige_est))
        f_fl = 4.0 * (base.MOTOR_IDLE_FLOOR if is_soaring else 0.90)
        f_cl = 4.0 * (t_ceil - 1.20)
        fz_base = min(f_cl, max(f_fl, fz_comp))

        df_x = 0.25 * (tx_res / self.d_arm)
        df_y = -0.25 * (ty_d / self.d_arm)
        raw_rp = 0.25 * fz_base + np.array([-df_x + df_y, +df_x + df_y, +df_x - df_y, -df_x - df_y], dtype=np.float64)
        min_rp = float(np.min(raw_rp))
        rp_sh = raw_rp + max(0.0, base.MOTOR_IDLE_FLOOR - min_rp)
        max_rp = float(np.max(rp_sh))
        fz_rot = np.minimum(np.maximum(rp_sh - max(0.0, max_rp - t_ceil), base.MOTOR_IDLE_FLOOR), t_ceil)

        if self.gimbal_max_deg <= 1e-3:
            fx_rot = np.zeros(4, dtype=np.float64)
            fy_rot = np.zeros(4, dtype=np.float64)
            tz_g_act = 0.0
        else:
            f_lmax = fz_rot * self.gimbal_tan
            f_lsum = max(1e-4, float(np.sum(f_lmax)))
            tz_gmax = base.ARM_LENGTH * f_lsum
            tz_ga = max(-0.65 * tz_gmax, min(0.65 * tz_gmax, tz_rem))
            fy_mag = (f_lmax / f_lsum) * (tz_ga / base.ARM_LENGTH)
            fy_x = fy_mag * self.yaw_tx
            fy_y = fy_mag * self.yaw_ty
            fx_need = Fx_cmd_g - float(np.sum(fy_x))
            fy_need = Fy_cmd_g - float(np.sum(fy_y))
            fw = fz_rot / max(1e-4, float(np.sum(fz_rot)))
            fx_raw = fy_x + fw * fx_need
            fy_raw = fy_y + fw * fy_need
            flat = np.sqrt(fx_raw*fx_raw + fy_raw*fy_raw + 1e-8)
            c_sc = np.minimum(1.0, f_lmax / flat)
            fx_rot = fx_raw * c_sc
            fy_rot = fy_raw * c_sc
            tz_g_act = float(np.sum(self.r_x * fy_rot - self.r_y * fx_rot))

        df_z = 0.25 * ((tz_rem - tz_g_act) / base.TORQUE_TO_THRUST_RATIO)
        ftot = np.sqrt(fx_rot*fx_rot + fy_rot*fy_rot + fz_rot*fz_rot)
        max_dfz = float(np.min(np.minimum(np.maximum(0.0, ftot - base.MOTOR_IDLE_FLOOR), np.maximum(0.0, t_ceil - ftot))))
        df_zc = max(-max_dfz, min(max_dfz, df_z))
        target_motors = np.minimum(np.maximum(ftot + self.spin_signs * df_zc, base.MOTOR_IDLE_FLOOR), t_ceil)
        fmag = np.maximum(1e-4, ftot)
        target_tilts = np.stack([fx_rot / fmag, fy_rot / fmag], axis=-1)

        return {
            "target_motors": target_motors,
            "target_tilts": target_tilts,
            "target_pusher": target_pusher,
            "hydro_surfaces_cmd": hydro_cmd,
            "scull_throttle": self.scull_throttle,
            "a_safe": a_safe,
            "est_clearance": est_clearance,
            "est_vel": est_vel,
            "est_pos": est_pos,
            "mode": self.mode,
        }


# =============================================================================
# 4. HARDENED SIM-TO-REAL ENVIRONMENT (ABLATIONS, BLACKOUTS, FAULTS, OBSTACLES)
# =============================================================================
class HardenedAutonomyEnv(base.WaveRidingAutonomyEnv):
    def __init__(self, num_envs=256, dt=0.015, device=DEVICE, use_embedded_qp=False,
                 fixed_foil=False, disable_scull=False, gimbal_max_deg=15.0,
                 mass_kg=None, harsh_physics=False, enable_obstacles=False,
                 gnss_denied_window=None, fault_scenario=None, fault_time=4.0,
                 strut_limit_nm=STRUT_YIELD_LIMIT_NM, realworld=None):
        if realworld:
            raise NotImplementedError(
                "The SIL suite's hardened plant / NumPy core are not ported to the engine's real-world conditions pack yet "
                "(wind, current, density, sensor suite). Use realworld=False (the engine's --legacy-physics).")
        self.fixed_foil = fixed_foil
        self.disable_scull = disable_scull
        self.gimbal_max_deg = gimbal_max_deg
        self.harsh_physics = harsh_physics
        self.enable_obstacles = enable_obstacles
        self.gnss_denied_window = gnss_denied_window
        self.fault_scenario = fault_scenario
        self.fault_time = fault_time
        self.strut_limit_nm = float(strut_limit_nm)
        # mass_kg is the TOTAL vehicle mass to simulate. The scale relative to the (evolving) CAD mass is
        # recomputed from the live morphology, so "3.2 kg" stays 3.2 kg whatever the genome has evolved to.
        self.mass_kg_target = None if mass_kg is None else float(mass_kg)
        self.mass_scale = 1.0

        super().__init__(num_envs=num_envs, dt=dt, device=device, use_embedded_qp=use_embedded_qp)

        # Replace plant & robot with configurable hardened implementations
        self.aero_hydro = HardenedAeroHydroDynamics(
            device, dt=dt, harsh_physics=harsh_physics, fixed_foil=fixed_foil, disable_scull=disable_scull,
            strut_limit_nm=strut_limit_nm
        )
        self.robot = ConfigurableRobot(
            dt, device, fixed_foil=fixed_foil, gimbal_max_deg=gimbal_max_deg, mass_scale=1.0
        )
        self.k_R = torch.tensor([[7.5, 8.0, 4.0]], device=device)
        self.k_omega = torch.tensor([[1.25, 1.35, 0.65]], device=device)
        self.reset()

    def _refresh_mass_scale(self, total_mass=None):
        """Recompute mass_scale (= target mass / current CAD mass) and push it into the robot inertia."""
        if self.mass_kg_target is None:
            scale = 1.0
        else:
            if total_mass is None:
                with torch.no_grad():
                    total_mass = self.morphology()["total_mass"]
            scale = self.mass_kg_target / max(1e-6, float(total_mass.detach().item() if torch.is_tensor(total_mass) else total_mass))
        self.mass_scale = scale
        if hasattr(self, "robot") and isinstance(self.robot, ConfigurableRobot) and abs(self.robot.mass_scale - scale) > 1e-9:
            self.robot.set_mass_scale(scale)
        return scale

    def detach_states(self):
        super().detach_states()
        self.aero_hydro.detach_state()
        self.y_ref = self.y_ref.detach()

    def reset(self, mode_mix=False):
        super().reset(mode_mix=mode_mix)
        N = self.num_envs
        if hasattr(self, "aero_hydro") and isinstance(self.aero_hydro, HardenedAeroHydroDynamics):
            self.aero_hydro.reset(N)
        if hasattr(self, "robot") and isinstance(self.robot, ConfigurableRobot):
            self.robot.motor_fault_scale = None
            self.robot.gimbal_jammed = False
        self._refresh_mass_scale()
        self.mass = self.mass * self.mass_scale

        # Markov burst blackout counters & breaking-wave aeration
        self.radar_blind_steps = torch.zeros((N, 1), dtype=torch.long, device=self.device)
        self.dvl_blind_steps = torch.zeros((N, 1), dtype=torch.long, device=self.device)
        self.aeration_steps = torch.zeros((N, 1), dtype=torch.long, device=self.device)

        # Synthetic obstacle corridor (6 obstacles ahead)
        self.num_obs = 6
        offsets = torch.tensor([[12.0, 24.0, 36.0, 48.0, 60.0, 72.0]], device=self.device).repeat(N, 1)
        self.obs_x = self.pos[:, 0:1] + offsets + torch.empty((N, self.num_obs), device=self.device).uniform_(-2.0, 2.0)
        self.obs_y = torch.empty((N, self.num_obs), device=self.device).uniform_(-0.65, 0.65)
        self.obs_r = torch.empty((N, self.num_obs), device=self.device).uniform_(0.35, 0.55)
        self.y_ref = torch.zeros((N, 1), device=self.device)
        self.fault_active = False

    def step(self, actor_net, ncm_net, export_diagnostics=False):
        N = self.num_envs
        self.sim_time = self.sim_time + self.dt
        self.step_count += 1
        ambient_gust = self.waves.step_gust()

        # Fault injection trigger
        t_cur = float(self.sim_time[0, 0].item())
        if self.fault_scenario is not None and t_cur >= self.fault_time:
            self.fault_active = True
            if self.fault_scenario == "motor_loss":
                mask = torch.ones((N, 4), device=self.device)
                mask[:, 2] = 0.65  # 35% thrust loss on M2
                self.robot.motor_fault_scale = mask
            elif self.fault_scenario == "gimbal_jam":
                self.robot.gimbal_jammed = True
            elif self.fault_scenario == "elevon_jam" and not self.fixed_foil:
                self.hydro_surfaces[:, 0] = math.radians(10.0)
            elif self.fault_scenario == "radar_blind":
                self.radar_blind_steps.fill_(9999)

        morph_dict = self.morphology()
        if self.mass_kg_target is not None:
            # fixed-mass experiment: constant (no gradient into the genome's mass), inertia follows the target
            self._refresh_mass_scale(morph_dict["total_mass"])
            self.mass = torch.full((N, 1), self.mass_kg_target, device=self.device) * self.mass_noise
        else:
            self.mass = morph_dict["total_mass"].repeat(N, 1) * self.mass_noise

        # 1. Ocean truth + breaking wave aeration when harsh_physics
        eta, deta_dx, deta_dy, deta_dt, w_updraft, v_water_orb = self.waves.evaluate_wave_kinematics(
            self.pos, self.sim_time, wave_scale=self.wave_curriculum_scale
        )
        if self.harsh_physics:
            new_aer = (torch.rand((N, 1), device=self.device) < 0.008) & (self.aeration_steps == 0)
            self.aeration_steps = torch.where(new_aer, torch.randint(10, 30, (N, 1), device=self.device), (self.aeration_steps - 1).clamp(min=0))
            rho_eff = torch.where(self.aeration_steps > 0, self.rho_water_local * 0.55, self.rho_water_local)
        else:
            rho_eff = self.rho_water_local

        is_soaring = (self.current_mode == base.MODE_SOARING)
        motor_thrusts_total = torch.sum(self.motors, dim=-1, keepdim=True)
        (
            f_wing, f_hydro, tau_hydro_pitch, tau_hydro_roll, tau_hydro_yaw, h_wing, d_sub_physical, f_slam, f_buoy,
            self.is_vent_L, self.is_vent_R, is_ventilated_overall, wind_shear_factor, strut_root_moment, aoa_tensors, alpha_foil
        ) = self.aero_hydro(
            self.pos, self.vel, self.rot, self.omega, eta, deta_dt, w_updraft, v_water_orb, ambient_gust,
            self.is_vent_L, self.is_vent_R, self.foil_decalage, rho_eff, self.hydro_surfaces,
            motor_thrusts_total, morph_dict, self.sim_time, is_soaring, scull_throttle=self.scull_throttle_cmd
        )

        # 3 & 4. Sensors + Markov burst blackouts + ES-EKF
        with torch.no_grad():
            rot_T_true = self.rot.transpose(-1, -2)
            v_water_rel_world = self.vel - v_water_orb
            v_water_rel_body = torch.matmul(rot_T_true, v_water_rel_world.unsqueeze(-1)).squeeze(-1)

            accel_body_true = torch.matmul(
                rot_T_true,
                (self.follower_accel_prev + torch.tensor([0.0, 0.0, base.GRAVITY], device=self.device)).unsqueeze(-1)
            ).squeeze(-1)
            meas_accel_body = accel_body_true + self.true_ba + torch.randn_like(accel_body_true) * 0.02
            meas_gyro_body = self.omega + self.true_bw + torch.randn_like(self.omega) * 0.008

            self.ekf.propagate(meas_accel_body, meas_gyro_body)

            if (self.step_count - 1) % self.sensor_period_steps == 0:
                R_hat_prior = base.quat_to_rot(self.ekf.q_hat)
                v_body_est = torch.matmul(R_hat_prior.transpose(-1, -2), self.ekf.v_hat.unsqueeze(-1)).squeeze(-1)

                d_sub_hydrostatic, p_static_decoupled, _, bernoulli_bias_cm = self.perception.sample_piezoresistive_tap(
                    d_sub_physical, v_water_rel_body, v_body_est, self.rho_water_local
                )
                radar_raw, spray_dropout, radar_locked, multipath_rejected = self.perception.process_fmcw_radar_dsp(
                    h_wing, deta_dx, deta_dy, v_water_rel_body[:, 0:1], v_water_rel_body[:, 2:3]
                )
                if self.harsh_physics:
                    trig_rad = ((f_slam > 15.0) | (torch.rand((N, 1), device=self.device) < 0.008)) & (self.radar_blind_steps == 0)
                    self.radar_blind_steps = torch.where(trig_rad, torch.randint(33, 120, (N, 1), device=self.device), (self.radar_blind_steps - 1).clamp(min=0))
                    radar_locked = radar_locked & (self.radar_blind_steps == 0)
                    spray_dropout = torch.where(self.radar_blind_steps > 0, torch.ones_like(spray_dropout), spray_dropout)

                    trig_dvl = ((is_ventilated_overall.bool()) | (f_buoy > 2.0) | (torch.rand((N, 1), device=self.device) < 0.005)) & (self.dvl_blind_steps == 0)
                    self.dvl_blind_steps = torch.where(trig_dvl, torch.randint(50, 150, (N, 1), device=self.device), (self.dvl_blind_steps - 1).clamp(min=0))
                elif self.fault_scenario == "radar_blind" and self.fault_active:
                    radar_locked = torch.zeros_like(radar_locked)
                    spray_dropout = torch.ones_like(spray_dropout)

                self.last_d_sub_hydrostatic = d_sub_hydrostatic
                self.last_p_static_decoupled = p_static_decoupled
                self.last_bernoulli_bias_cm = bernoulli_bias_cm
                self.last_radar_raw = radar_raw
                self.last_radar_locked = radar_locked
                self.last_spray_dropout = spray_dropout
                self.last_multipath_rejected = multipath_rejected
                meas_radar = torch.where(radar_locked, radar_raw, self.clearance_hold)

                mag_meas_body = torch.matmul(rot_T_true, base.MAG_FIELD_WORLD.to(self.device).repeat(N, 1).unsqueeze(-1)).squeeze(-1) + torch.randn_like(self.omega) * 0.02
                meas_dvl_body = v_water_rel_body + torch.randn_like(v_water_rel_body) * 0.03

                celerity_est = self.dispersion_observer.omega_dom / torch.clamp(self.dispersion_observer.k_dom, min=0.1)
                h_flow_true = torch.clamp(h_wing, min=0.08)
                flow_rates_body = torch.cat([
                    -(v_water_rel_body[:, 1:2]) / h_flow_true,
                    +(v_water_rel_body[:, 0:1] - celerity_est) / h_flow_true
                ], dim=-1) + torch.randn((N, 2), device=self.device) * 0.04

                self.ekf.update_magnetometer(mag_meas_body)
                is_submerged_sensor = (d_sub_hydrostatic > 0.015)
                dvl_acoustic_lock = (~is_ventilated_overall.bool()) & (torch.rand_like(is_ventilated_overall) > 0.01) & (self.dvl_blind_steps == 0)
                self.ekf.update_dvl(meas_dvl_body, is_submerged_sensor, (~dvl_acoustic_lock).float())
                self.ekf.update_optical_flow(flow_rates_body, meas_radar, celerity_est, (d_sub_hydrostatic <= 0.015))
                self.last_dvl_active = is_submerged_sensor & dvl_acoustic_lock
                self.last_flow_active = (~is_submerged_sensor) & (meas_radar < 1.8)
            else:
                d_sub_hydrostatic = self.last_d_sub_hydrostatic
                p_static_decoupled = self.last_p_static_decoupled
                bernoulli_bias_cm = self.last_bernoulli_bias_cm
                radar_raw = self.last_radar_raw
                radar_locked = self.last_radar_locked
                spray_dropout = self.last_spray_dropout
                multipath_rejected = self.last_multipath_rejected
                meas_radar = torch.where(radar_locked, radar_raw, self.clearance_hold)
                is_submerged_sensor = (d_sub_hydrostatic > 0.015)

            dvl_active = self.last_dvl_active
            flow_active = self.last_flow_active

            gnss_blocked = False
            if self.gnss_denied_window is not None:
                gnss_blocked = (self.gnss_denied_window[0] <= t_cur <= self.gnss_denied_window[1])

            if (self.step_count % self.gnss_period_steps == 0) and (not gnss_blocked):
                infl_true = 1.0 + 8.0 * torch.exp(-torch.clamp(h_wing, min=0.01, max=10.0) / 0.25)
                gnss_pos = self.pos + torch.randn_like(self.pos) * 0.05 * torch.sqrt(infl_true)
                gnss_vel = self.vel + torch.randn_like(self.vel) * 0.03 * torch.sqrt(infl_true)
                self.ekf.update_gnss(gnss_pos, gnss_vel, meas_radar)

            # Navigation state
            est_pos = self.ekf.p_hat
            est_vel = self.ekf.v_hat
            est_rot = base.quat_to_rot(self.ekf.q_hat)
            est_omega = meas_gyro_body - self.ekf.bw_hat

            grav_w = torch.tensor([0.0, 0.0, -base.GRAVITY], device=self.device).view(1, 3)
            a_est_world = torch.matmul(est_rot, (meas_accel_body - self.ekf.ba_hat).unsqueeze(-1)).squeeze(-1) + grav_w

            cos_tilt_est = est_rot[:, 2, 2:3].clamp(min=0.30)
            h_hydro_projected = F.relu(morph_dict["strut_length"].detach() * cos_tilt_est - d_sub_hydrostatic)
            est_clearance = torch.where(
                radar_locked,
                meas_radar,
                torch.where(is_submerged_sensor, h_hydro_projected, torch.clamp(est_pos[:, 2:3], min=0.04, max=10.0))
            )
            self.clearance_hold = est_clearance.detach().clamp(0.04, self.perception.radar_r_max)
            est_h_dot = est_vel[:, 2:3]

            ekf_err_h = est_clearance - h_wing
            ekf_err_vx = est_vel[:, 0:1] - self.vel[:, 0:1]
            R_ekf_err = torch.bmm(self.rot.transpose(-1, -2), est_rot)
            tr_ekf = R_ekf_err[:, 0, 0] + R_ekf_err[:, 1, 1] + R_ekf_err[:, 2, 2]
            ekf_err_att_deg = (torch.acos(((tr_ekf - 1.0) * 0.5).clamp(-0.99999, 0.99999)) * (180.0 / math.pi)).unsqueeze(-1)

            if (self.step_count - 1) % self.sensor_period_steps == 0:
                est_ray_w = torch.matmul(est_rot, self.waves.ray_body_vec).squeeze(-1)
                est_s = torch.clamp(est_clearance / (-est_ray_w[:, 2:3]).clamp(min=0.06), min=0.25, max=6.0)
                lookahead_dist = base.safe_norm(est_s * est_ray_w[:, :2], dim=-1, keepdim=True)
                eta_forward_meas, eta_forward_valid = self.waves.raycast_forward_surface_pose(
                    self.pos, self.rot, eta, self.sim_time, self.wave_curriculum_scale
                )
                eta_prev, deta_dx_prev, w_up_prev = self.dispersion_observer.update_and_predict(
                    est_clearance, est_pos[:, 2:3], eta_forward_meas, eta_forward_valid, lookahead_dist,
                    est_vel[:, 0:1], preview_time=1.25
                )
                self.last_eta_fwd_meas = eta_forward_meas
                self.last_eta_fwd_valid = eta_forward_valid
                self.last_eta_prev = eta_prev
                self.last_deta_dx_prev = deta_dx_prev
                self.last_w_up_prev = w_up_prev
                self.last_deta_dx_nadir = self.dispersion_observer.deta_dx_nadir
                self.last_w_up_nadir = self.dispersion_observer.w_up_nadir
                self.last_look_dist = lookahead_dist
            else:
                eta_forward_meas = self.last_eta_fwd_meas
                eta_prev = self.last_eta_prev
                deta_dx_prev = self.last_deta_dx_prev
                w_up_prev = self.last_w_up_prev
                lookahead_dist = self.last_look_dist

            sigma_l1 = self.l1_observer.step(a_est_world, self.a_pre_dob_prev, self.dt, sat_mask=self.l1_sat_prev)

            # Synthetic subsurface obstacle detection & lateral reference planner
            obstacle_hit = torch.zeros((N, 1), dtype=torch.bool, device=self.device)
            if self.enable_obstacles:
                dx_true = self.obs_x - self.pos[:, 0:1]
                dy_true = self.obs_y - self.pos[:, 1:2]
                dist_true = torch.sqrt(dx_true**2 + dy_true**2)
                obstacle_hit = ((dist_true < self.obs_r) & (d_sub_physical > 0.005)).any(dim=-1, keepdim=True)

                passed = self.obs_x < (self.pos[:, 0:1] - 3.0)
                if passed.any():
                    far_x = torch.max(self.obs_x, dim=-1, keepdim=True).values + 12.0 + torch.empty((N, self.num_obs), device=self.device).uniform_(0.0, 4.0)
                    self.obs_x = torch.where(passed, far_x, self.obs_x)
                    self.obs_y = torch.where(passed, torch.empty_like(self.obs_y).uniform_(-0.65, 0.65), self.obs_y)

                dx_est = self.obs_x - est_pos[:, 0:1]
                in_fov = (dx_est > 0.2) & (dx_est < 11.0) & (torch.rand_like(dx_est) < 0.90)
                dx_masked = torch.where(in_fov, dx_est, torch.full_like(dx_est, 1e6))
                idx_near = torch.argmin(dx_masked, dim=-1, keepdim=True)
                dx_n = torch.gather(dx_masked, -1, idx_near)
                oy_n = torch.gather(self.obs_y, -1, idx_near)
                or_n = torch.gather(self.obs_r, -1, idx_near)

                need_avoid = (dx_n < 10.5) & (torch.abs(est_pos[:, 1:2] - oy_n) < (or_n + 0.65))
                avoid_dir = torch.where(oy_n >= 0.0, -torch.ones_like(oy_n), torch.ones_like(oy_n))
                y_target_raw = torch.where(need_avoid, oy_n + avoid_dir * (or_n + 0.80), torch.zeros_like(oy_n))
                self.y_ref = self.y_ref + (self.dt / (0.25 + self.dt)) * (y_target_raw - self.y_ref)
            else:
                self.y_ref.zero_()

        # 7. Automaton
        is_loiter = (self.current_mode == base.MODE_LOITER)
        is_takeoff = (self.current_mode == base.MODE_TAKEOFF)
        is_foiling = (self.current_mode == base.MODE_FOILING)
        is_egress = (self.current_mode == base.MODE_EGRESS)
        is_flight = (self.current_mode == base.MODE_FREE_FLIGHT)
        is_ingress = (self.current_mode == base.MODE_INGRESS)
        is_soaring = (self.current_mode == base.MODE_SOARING)

        v_target_raw = torch.where(is_loiter, 0.0,
                       torch.where(is_takeoff, 5.20,
                       torch.where(is_flight, 8.50, self.target_speed)))

        loiter_descent_ready = is_loiter & (est_vel[:, 0:1] < 1.5)
        h_loiter_tgt = torch.where(loiter_descent_ready, 0.04, 0.18)
        is_ventilated_sensor_est = (d_sub_hydrostatic > 0.020) & (a_est_world[:, 2:3] < -1.8)
        h_foiling_cmd = torch.where(
            is_ventilated_sensor_est & is_submerged_sensor,
            torch.full_like(h_loiter_tgt, base.RIDE_HEIGHT_REPRIME),
            torch.full_like(h_loiter_tgt, base.RIDE_HEIGHT_OPTIMAL)
        )
        h_target_raw = torch.where(is_loiter, h_loiter_tgt,
                       torch.where(is_takeoff, 0.185,
                       torch.where(is_foiling, h_foiling_cmd,
                       torch.where(is_soaring, base.RIDE_HEIGHT_SOARING,
                       torch.where(is_egress, base.FREE_FLIGHT_ALTITUDE,
                       torch.where(is_flight, base.FREE_FLIGHT_ALTITUDE, base.RIDE_HEIGHT_OPTIMAL))))))

        alpha_blend = self.dt / (0.15 + self.dt)
        self.h_target_filtered = self.h_target_filtered + alpha_blend * (h_target_raw - self.h_target_filtered)
        self.v_target_filtered = self.v_target_filtered + alpha_blend * (v_target_raw - self.v_target_filtered)

        self.current_mode = torch.where(is_takeoff & (est_vel[:, 0:1] > 4.6) & is_submerged_sensor, torch.full_like(self.current_mode, base.MODE_FOILING), self.current_mode)
        self.current_mode = torch.where(is_egress & (est_clearance > (base.FREE_FLIGHT_ALTITUDE - 0.15)), torch.full_like(self.current_mode, base.MODE_FREE_FLIGHT), self.current_mode)
        self.current_mode = torch.where(is_ingress & is_submerged_sensor, torch.full_like(self.current_mode, base.MODE_FOILING), self.current_mode)

        # 8. Pilot & NCM-CBF QP (with lateral y_ref + cavitation/ventilation barrier)
        e_speed = est_vel[:, 0:1] - self.v_target_filtered
        e_height = est_clearance - self.h_target_filtered
        e_y_track = est_pos[:, 1:2] - self.y_ref
        self.e_int_height = torch.where(
            (is_foiling | is_takeoff | is_soaring),
            torch.clamp(self.e_int_height + e_height * self.dt, -0.12, 0.12),
            self.e_int_height * 0.95
        )
        y_int_active = (is_foiling | is_takeoff | is_soaring) & (est_vel[:, 0:1] > 1.0)
        self.e_int_y = torch.where(
            y_int_active,
            torch.clamp(self.e_int_y + e_y_track * self.dt, -1.5, 1.5),
            self.e_int_y * 0.95
        )

        actor_obs = torch.cat([
            e_speed, e_height, self.e_int_height, e_y_track,
            est_h_dot, self.v_target_filtered / 8.50, d_sub_hydrostatic, self.h_target_filtered,
            est_vel, est_omega, est_rot[:, 2, :],
            eta_prev - (est_pos[:, 2:3] - est_clearance), deta_dx_prev, w_up_prev
        ], dim=-1)

        a_neural = actor_net(actor_obs)
        self.guidance_cmd_buffer.append(a_neural)
        a_guidance_delayed = self.guidance_cmd_buffer[0]
        self.a_pilot_filtered = self.pilot_filter_a * self.a_pilot_filtered + (1.0 - self.pilot_filter_a) * a_guidance_delayed

        state_err = torch.cat([
            torch.zeros((N, 1), device=self.device), e_y_track, e_height,
            e_speed, est_vel[:, 1:2], est_h_dot
        ], dim=-1)
        M_metric = ncm_net(state_err)
        M_5 = M_metric[:, 1:, 1:]

        v_wave_apparent_est = self.last_deta_dx_nadir * est_vel[:, 0:1] + self.last_w_up_nadir
        h_dot_relative_est = est_vel[:, 2:3] - v_wave_apparent_est
        h_min_mode = torch.where(loiter_descent_ready | is_takeoff, 0.01, base.RIDE_HEIGHT_MIN)

        b_cbf_crash = (4.00 * h_dot_relative_est + 12.0 * (est_clearance - h_min_mode)).unsqueeze(-1)
        h_max_mode = torch.where(is_egress | is_flight | is_ingress, 4.00, torch.where(is_foiling | is_soaring, base.RIDE_HEIGHT_FOILING_MAX, base.RIDE_HEIGHT_WIG_MAX))
        h_margin = h_max_mode - est_clearance
        b_cbf_ceil = (-4.00 * h_dot_relative_est + 12.0 * torch.clamp(h_margin, min=-0.5, max=1.0)).unsqueeze(-1)

        e_xy = state_err[:, [1, 3, 4]]
        M_xy = M_5[:, [0, 2, 3], :][:, :, [0, 2, 3]]
        m_e_xy = torch.bmm(M_xy, e_xy.unsqueeze(-1)).squeeze(-1)
        g_xy = m_e_xy[:, 1:3]
        V_xy = torch.sum(e_xy * m_e_xy, dim=-1, keepdim=True)
        kin_xy = m_e_xy[:, 0:1] * e_xy[:, 2:3]
        A_ncm_xy = torch.cat([g_xy, torch.zeros((N, 1), device=self.device)], dim=-1).unsqueeze(1)
        b_ncm_xy = (-0.15 * V_xy - kin_xy).unsqueeze(-1)

        e_z = state_err[:, [2, 5]]
        M_z = M_5[:, [1, 4], :][:, :, [1, 4]]
        m_e_z = torch.bmm(M_z, e_z.unsqueeze(-1)).squeeze(-1)
        g_z = m_e_z[:, 1:2]
        V_z = torch.sum(e_z * m_e_z, dim=-1, keepdim=True)
        kin_z = m_e_z[:, 0:1] * e_z[:, 1:2]
        A_ncm_z = torch.cat([torch.zeros((N, 2), device=self.device), g_z], dim=-1).unsqueeze(1)
        b_ncm_z = (-0.15 * V_z - kin_z).unsqueeze(-1)

        A_all = torch.cat([self.A_cbf_crash, self.A_cbf_ceil, A_ncm_xy, A_ncm_z], dim=1)
        b_all = torch.cat([b_cbf_crash, b_cbf_ceil, b_ncm_xy, b_ncm_z], dim=1)

        if self.use_embedded_qp and not torch.is_grad_enabled():
            da_max = float(base.MAX_JERK_ACCEL * self.dt)
            a_nom_jl = torch.max(torch.min(self.a_pilot_filtered, self.a_pilot_prev + da_max), self.a_pilot_prev - da_max)
            a_safe, qp_override, active_qp_barrier_idx = self.embedded_qp.solve_tensor(a_nom_jl, A_all, b_all)
            slack = torch.zeros((N, A_all.shape[1]), device=self.device)
            active_qp_barrier_val = qp_override.clone()
            is_jerk_active = torch.zeros((N, 1), dtype=torch.bool, device=self.device)
        else:
            a_safe, qp_override, slack, active_qp_barrier_idx, active_qp_barrier_val, is_jerk_active = self.safety_filter(
                self.a_pilot_filtered, self.a_pilot_prev, A_all, b_all, self.dt
            )
        self.a_pilot_prev = a_safe

        # 9. Inner loop & Hybrid Wrench Allocation
        submerged_factor_est = torch.clamp(d_sub_hydrostatic / 0.04, min=0.0, max=1.0)
        phi_max_allowed = submerged_factor_est * math.radians(6.5) + (1.0 - submerged_factor_est) * math.radians(18.0)

        v_horiz_est = base.safe_norm(est_vel[:, :2], dim=-1, keepdim=True).clamp(min=0.5)
        q_water_est = 0.5 * base.RHO_WATER_NOM * (v_horiz_est**2)
        q_air_est = 0.5 * base.RHO_AIR * (v_horiz_est**2)

        f_hydro_drag_est = q_water_est * morph_dict["foil_area"] * base.FOIL_CD0 * submerged_factor_est
        f_wing_drag_est = q_air_est * morph_dict["wing_area"] * (base.WING_CD0 + 0.03)
        f_drag_est = f_hydro_drag_est + f_wing_drag_est

        nom_mass = float(morph_dict["total_mass"].detach().item()) * self.mass_scale

        dec_mean = math.radians(0.5 * (base.FOIL_DECALAGE_DEG[0] + base.FOIL_DECALAGE_DEG[1]))
        v_b_est = torch.matmul(est_rot.transpose(-1, -2), est_vel.unsqueeze(-1)).squeeze(-1)
        alpha_foil_est = torch.atan2(-v_b_est[:, 2:3], v_b_est[:, 0:1].clamp(min=0.5)) + dec_mean
        q_A_CLa = q_water_est * morph_dict["foil_area"] * base.FOIL_CL_ALPHA
        f_hydro_lift_unclamped = q_A_CLa * alpha_foil_est * submerged_factor_est

        h_est_safe = est_clearance.clamp(min=0.03)
        phi_wig_est = 1.0 + 0.35 / (0.10 + 2.2 * torch.pow(h_est_safe / morph_dict["wing_chord"], 1.15))
        alpha_wing_est = torch.atan2(-v_b_est[:, 2:3], v_b_est[:, 0:1].clamp(min=0.5)) + 0.03
        cl_wing_est = torch.clamp((base.WING_CL0 + 4.5 * alpha_wing_est) * phi_wig_est, -0.3, 1.4)
        f_wing_lift_est = (q_air_est * morph_dict["wing_area"] * cl_wing_est).clamp(min=0.0, max=15.0)

        tau_hydro_pitch_est = f_hydro_drag_est * morph_dict["strut_length"]

        min_rotor_floor = torch.where(is_soaring, torch.full((N, 1), ROTOR_FLOOR_SOARING_N, device=self.device), torch.full((N, 1), ROTOR_FLOOR_N, device=self.device))
        desired_fz_total = nom_mass * (base.GRAVITY + a_safe[:, 2:3] - sigma_l1[:, 2:3])
        lift_cap = desired_fz_total - min_rotor_floor - f_wing_lift_est
        over_lift_est = F.relu(f_hydro_lift_unclamped - lift_cap)
        q_A_CLd = q_water_est * morph_dict["foil_area"] * base.HYDRO_CL_DELTA * submerged_factor_est
        coll_max = 0.8 * base.MAX_HYDRO_DEFLECTION
        foil_active_gate = (submerged_factor_est > 0.05).float() * (est_vel[:, 0:1] > 1.0).float()

        if self.fixed_foil:
            delta_collective = torch.zeros_like(over_lift_est)
            theta_shed_lift = torch.clamp(over_lift_est / (q_A_CLa + 50.0), max=math.radians(6.0)) * foil_active_gate
            f_hydro_lift_est = torch.clamp(f_hydro_lift_unclamped, min=-1.0, max=2.0 * base.FOIL_LIFT_CLAMP_PER_SIDE)
        else:
            delta_collective = -torch.clamp(over_lift_est / (q_A_CLd + 1.0), max=coll_max) * foil_active_gate
            over_after_collective = F.relu(over_lift_est - q_A_CLd * coll_max)
            theta_shed_lift = torch.clamp(over_after_collective / (q_A_CLa + 50.0), max=math.radians(6.0)) * foil_active_gate
            f_hydro_lift_est = torch.clamp(f_hydro_lift_unclamped + q_A_CLd * delta_collective, min=-1.0, max=2.0 * base.FOIL_LIFT_CLAMP_PER_SIDE)

        raw_f_rotors = desired_fz_total - f_hydro_lift_est - f_wing_lift_est
        f_rotors_needed = torch.maximum(raw_f_rotors, min_rotor_floor).clamp(max=float(nom_mass * base.GRAVITY + 6.0))

        psi_est_rad = torch.atan2(est_rot[:, 1:2, 0], est_rot[:, 0:1, 0])
        c_psi, s_psi = torch.cos(psi_est_rad), torch.sin(psi_est_rad)
        ax_w = a_safe[:, 0:1] - sigma_l1[:, 0:1]
        ay_w = a_safe[:, 1:2] - 1.4 * self.e_int_y - sigma_l1[:, 1:2]
        Fx_dyn = nom_mass * (c_psi * ax_w + s_psi * ay_w)
        Fy_dyn = nom_mass * (-s_psi * ax_w + c_psi * ay_w)

        f_gimbal_horiz_cap = f_rotors_needed * (self.robot.gimbal_tan * 0.70)
        Fx_gimbal_est = torch.where(is_soaring, torch.zeros_like(Fx_dyn), torch.clamp(Fx_dyn, -f_gimbal_horiz_cap, f_gimbal_horiz_cap))

        soar_prop_req = torch.where(
            is_soaring & is_submerged_sensor & (~torch.tensor(self.fixed_foil or self.disable_scull, device=self.device)),
            torch.clamp(Fx_dyn + f_drag_est, min=0.0, max=22.0),
            torch.zeros_like(Fx_dyn)
        )
        self.scull_throttle_cmd = (soar_prop_req / 22.0).clamp(0.0, 1.0).detach()

        pusher_disabled = is_loiter & (est_vel[:, 0:1] < 0.8)
        target_pusher = torch.where(
            pusher_disabled | is_soaring,
            torch.zeros_like(Fx_dyn),
            torch.clamp(Fx_dyn + f_drag_est - Fx_gimbal_est, min=0.0, max=base.MAX_PUSHER_THRUST)
        )
        Fx_cmd_gimbal = Fx_dyn + f_drag_est - target_pusher - soar_prop_req
        Fx_residual = Fx_cmd_gimbal - torch.clamp(Fx_cmd_gimbal, -f_gimbal_horiz_cap, f_gimbal_horiz_cap)

        Fy_cmd_gimbal = Fy_dyn
        Fy_residual = Fy_dyn - torch.clamp(Fy_dyn, -f_gimbal_horiz_cap, f_gimbal_horiz_cap)

        f_vertical_net = (nom_mass * base.GRAVITY - f_hydro_lift_est - f_wing_lift_est).clamp(min=4.0)
        raw_phi_btt = torch.atan2(-Fy_residual, f_vertical_net)
        beta_fluid = torch.atan2(est_vel[:, 1:2], est_vel[:, 0:1].clamp(min=1.0))
        lateral_damping = 0.22 * torch.clamp(beta_fluid, -math.radians(10.0), math.radians(10.0))
        phi_target = torch.minimum(torch.maximum(raw_phi_btt + lateral_damping, -phi_max_allowed), phi_max_allowed)

        alpha_phi = self.dt / (0.15 + self.dt)
        self.phi_des_filtered = self.phi_des_filtered + alpha_phi * (phi_target - self.phi_des_filtered)
        phi_des = self.phi_des_filtered

        theta_tilt_residual = torch.atan2(Fx_residual, f_vertical_net)
        theta_bias = torch.where(is_egress, -math.radians(4.5),
                     torch.where(is_ingress, -math.radians(1.5),
                     torch.zeros_like(theta_tilt_residual)))
        theta_des = torch.clamp(theta_tilt_residual + theta_bias + theta_shed_lift, -math.radians(14.0), math.radians(24.0))

        self.a_pre_dob_prev = torch.cat([
            a_safe[:, 0:1],
            a_safe[:, 1:2] - 1.4 * self.e_int_y,
            a_safe[:, 2:3]
        ], dim=-1).detach()
        sat_x = (theta_des <= math.radians(-13.8)) | (theta_des >= math.radians(23.8)) | (pusher_disabled & (Fx_dyn > f_gimbal_horiz_cap))
        sat_y = torch.abs(raw_phi_btt + lateral_damping) > phi_max_allowed
        sat_z = (raw_f_rotors < min_rotor_floor) | (raw_f_rotors > float(nom_mass * base.GRAVITY + 6.0))
        self.l1_sat_prev = torch.cat([sat_x, sat_y, sat_z], dim=-1).detach()

        yaw_steer_gate = ((est_vel[:, 0:1] - 0.5) / 1.5).clamp(min=0.0, max=1.0)
        psi_target = -torch.clamp(
            e_y_track * 0.12 + 0.15 * est_vel[:, 1:2],
            -math.radians(5.0), math.radians(5.0)
        ) * yaw_steer_gate
        alpha_psi = self.dt / (0.45 + self.dt)
        self.psi_des_filtered = self.psi_des_filtered + alpha_psi * (psi_target - self.psi_des_filtered)
        psi_des = self.psi_des_filtered

        Rd = base.euler_to_rot(phi_des, theta_des, psi_des)
        R_err = torch.matmul(Rd.transpose(-1, -2), est_rot) - torch.matmul(est_rot.transpose(-1, -2), Rd)

        tau_des = torch.cat([
            -self.k_R[0, 0] * R_err[:, 2, 1:2] * 0.5 - self.k_omega[0, 0] * est_omega[:, 0:1],
            -self.k_R[0, 1] * R_err[:, 0, 2:3] * 0.5 - self.k_omega[0, 1] * est_omega[:, 1:2] - 1.00 * tau_hydro_pitch_est,
            -self.k_R[0, 2] * R_err[:, 1, 0:1] * 0.5 - self.k_omega[0, 2] * est_omega[:, 2:3]
        ], dim=-1)

        h_rotors_est = torch.clamp(est_clearance, min=0.04)
        ige_est = 1.0 / (1.0 - torch.clamp((base.PROP_RADIUS / (4.0 * h_rotors_est))**2, max=0.35))

        (
            target_motors, target_tilts, hydro_surfaces_cmd, headroom_min, is_clipped_min,
            is_clipped_max, torque_scale, fx_gimbal_alloc, fy_gimbal_alloc
        ) = self.robot.allocate_wrench_hybrid(
            Fx_cmd_gimbal, Fy_cmd_gimbal, f_rotors_needed, tau_des, self.v_batt, ige_est,
            d_sub_hydrostatic, est_vel[:, 0:1], is_soaring, morph_dict, delta_collective=delta_collective
        )

        if self.fault_scenario == "elevon_jam" and self.fault_active and not self.fixed_foil:
            hydro_surfaces_cmd[:, 0] = math.radians(10.0)

        max_d_delta = base.SERVO_RATE_LIMIT * self.dt
        self.hydro_surfaces_actual = self.hydro_surfaces_actual + torch.clamp(
            hydro_surfaces_cmd - self.hydro_surfaces_actual, -max_d_delta, max_d_delta
        )
        self.hydro_surfaces = self.hydro_surfaces_actual

        wave_slope_mag = base.safe_norm(torch.cat([deta_dx, deta_dy], dim=-1), dim=-1, keepdim=True)

        (
            self.pos, self.vel, self.rot, self.omega, self.motors, self.rotor_tilts, self.pusher_thrust,
            self.nu_states, omega_props, accel, ige_rotor_gain
        ) = self.robot.integrate(
            self.pos, self.vel, self.rot, self.omega, self.motors, self.rotor_tilts, self.pusher_thrust,
            self.nu_states, target_motors, target_tilts, target_pusher, f_wing, f_hydro,
            tau_hydro_pitch, tau_hydro_roll, tau_hydro_yaw,
            ambient_gust * wind_shear_factor, eta, wave_slope_mag,
            self.mass, self.motor_tau_up, self.motor_tau_down
        )
        self.follower_accel_prev = accel.detach()

        eta_post = eta + deta_dt * self.dt
        clearance_eval = (self.pos[:, 2:3] - eta_post).clamp(min=0.01)
        z_tip_L_post = self.pos[:, 2:3] + self.rot[:, 2, 1:2] * (morph_dict["foil_span"] * 0.25) - self.rot[:, 2, 2:3] * morph_dict["strut_length"]
        z_tip_R_post = self.pos[:, 2:3] - self.rot[:, 2, 1:2] * (morph_dict["foil_span"] * 0.25) - self.rot[:, 2, 2:3] * morph_dict["strut_length"]
        immersion_eval = 0.5 * (
            (eta_post - z_tip_L_post).clamp(min=0.0, max=0.22)
            + (eta_post - z_tip_R_post).clamp(min=0.0, max=0.22)
        )
        phi_true_deg = torch.atan2(self.rot[:, 2, 1:2], self.rot[:, 2, 2:3]) * (180.0 / math.pi)

        p_scull_load = aoa_tensors["p_scull_actuator_watts"]
        v_air_x_body = torch.matmul(self.rot.transpose(-1, -2), self.vel.unsqueeze(-1)).squeeze(-1)[:, 0:1]
        p_elec, self.v_batt, i_batt, self.u1, self.u2, self.soc, self.t_cell = self.powertrain.step(
            self.motors, w_updraft, self.nu_states[:, :, 0], omega_props,
            self.u1, self.u2, self.soc, self.t_cell, self.dt, p_scull_load,
            pusher_thrust=self.pusher_thrust, v_air_x=v_air_x_body
        )

        tilt_sin_mag = torch.sqrt(self.rotor_tilts[:, :, 0]**2 + self.rotor_tilts[:, :, 1]**2 + 1e-10).clamp(max=1.0)
        rotor_tilt_deg = torch.asin(tilt_sin_mag) * (180.0 / math.pi)

        self.metrics = {
            "mode": self.current_mode,
            "v_target_mode": self.v_target_filtered,
            "h_target_mode": self.h_target_filtered,
            "plant_damp_diag": aoa_tensors["plant_damp_diag"],
            "phi_true_deg": phi_true_deg,
            "ekf_err_h": ekf_err_h,
            "ekf_err_vx": ekf_err_vx,
            "ekf_err_att_deg": ekf_err_att_deg,
            "sigma_l1_vec": sigma_l1,
            "e_int_y": self.e_int_y,
            "fx_gimbal": fx_gimbal_alloc,
            "fy_gimbal": fy_gimbal_alloc,
            "clearance_eval": clearance_eval,
            "immersion_eval": immersion_eval,
            "f_hydro_lift_eval": f_hydro[:, 2:3],
            "vel_eval": self.vel,
            "pos_eval": self.pos,
            "omega_eval": self.omega,
            "h_wing": h_wing,
            "clearance_est": est_clearance,
            "immersion_depth": d_sub_hydrostatic,
            "f_slam": f_slam,
            "f_buoy": f_buoy,
            "f_hydro_lift": f_hydro[:, 2:3],
            "f_wing_lift": f_wing[:, 2:3],
            "f_rotors": torch.sum(self.motors, dim=-1, keepdim=True),
            "f_pusher": self.pusher_thrust,
            "motors": self.motors,
            "rotor_tilts": self.rotor_tilts,
            "rotor_tilt_deg": rotor_tilt_deg,
            "hydro_surfaces": self.hydro_surfaces,
            "hydro_deflections": (
                math.degrees(self.hydro_surfaces[0, 0].item()),
                math.degrees(self.hydro_surfaces[0, 1].item()),
                math.degrees(self.hydro_surfaces[0, 2].item())
            ),
            "cell_temp_c": self.t_cell[0, 0].item() - 273.15,
            "headroom_min": headroom_min,
            "is_clipped_min": is_clipped_min,
            "is_clipped_max": is_clipped_max,
            "power_saved_pct": torch.zeros((N, 1), device=self.device),
            "p_elec": p_elec,
            "v_batt": self.v_batt,
            "i_batt": i_batt,
            "soc": self.soc * 100.0,
            "qp_override": qp_override,
            "max_slack": slack,
            "actor_obs": actor_obs,
            "a_safe": a_safe,
            "a_neural": a_neural,
            "state_err": state_err,
            "pos": self.pos,
            "vel": self.vel,
            "eta": eta,
            "omega": self.omega,
            "e_R": torch.cat([R_err[:, 2, 1:2] * 0.5, R_err[:, 0, 2:3] * 0.5, R_err[:, 1, 0:1] * 0.5], dim=-1),
            "e_int_height": self.e_int_height,
            "is_vent_L": self.is_vent_L,
            "is_vent_R": self.is_vent_R,
            "is_ventilated": is_ventilated_overall[0, 0].item() > 0.5 if export_diagnostics else False,
            "h_dot_est": est_h_dot,
            "radar_dropout": spray_dropout[0, 0].item() > 0.5 if export_diagnostics else False,
            "eta_preview": eta_prev,
            "eta_forward_meas": eta_forward_meas,
            "eta_nadir_est": self.dispersion_observer.last_eta_nadir,
            "look_dist": lookahead_dist,
            "k_dom": self.dispersion_observer.k_dom,
            "omega_dom": self.dispersion_observer.omega_dom,
            "sigma_l1_norm": base.safe_norm(sigma_l1, dim=-1, keepdim=True),
            "strut_moment_nm": strut_root_moment,
            "strut_moment_total_nm": aoa_tensors["strut_moment_total_nm"],
            "strut_yield_fail": aoa_tensors["strut_yield_fail"],
            "sigma_cav_min": aoa_tensors["sigma_cav_min"],
            "cav_active": aoa_tensors["cav_active"],
            "obstacle_hit": obstacle_hit,
            "y_ref": self.y_ref,
            "radar_locked": bool(radar_locked[0, 0].item() > 0.5) if export_diagnostics else True,
            "multipath_rejected": bool(multipath_rejected[0, 0].item() > 0.5) if export_diagnostics else False,
            "delta_collective_deg": math.degrees(delta_collective[0, 0].item()),
            "foil_lift_est": f_hydro_lift_est,
            "status_banner": "[NOMINAL]",
        }
        if not torch.is_grad_enabled():
            self.detach_states()
        return self.metrics


# =============================================================================
# 4b. CORE HARNESS: FastFlightCore sees the SAME raw sensor samples as the torch pipeline
# =============================================================================
class CoreHarness:
    """
    Couples one FastFlightCore to a single-vehicle HardenedAutonomyEnv without touching the env's code.
    It wraps the env's sensor-update calls to record the raw samples the PyTorch pipeline consumed, runs
    the core on exactly those samples at the point where the PyTorch pipeline hands its commands to the
    actuators, and then either only compares them (shadow) or substitutes them (drive).

      drive=False  shadow: the PyTorch pipeline flies; the core only watches -> parity measurement.
      drive=True   drive : the NumPy core's motor / gimbal / pusher / elevon / scull commands are what the
                           plant receives. The PyTorch pipeline keeps running as the reference.
    After every step() `self.cmp` holds the core-vs-torch differences and `self.latency_us` the core's
    wall-clock time for that tick.
    """
    CMP_KEYS = ("p_err", "v_err", "att_err_deg", "clr_err", "a_safe_err", "motors_err",
                "tilt_err", "pusher_err", "hydro_err_deg", "mode_eq")

    def __init__(self, env, core, drive=False):
        if env.num_envs != 1:
            raise ValueError("CoreHarness drives a single vehicle (num_envs=1)")
        self.env, self.core, self.drive = env, core, bool(drive)
        self.rec = {}
        self.out = None
        self.cmp = {}
        self.latency_us = 0.0
        self._v_batt_pre = float(env.v_batt[0, 0])
        self._mode_pre = int(env.current_mode[0, 0])
        self._install()

    # -- plumbing ---------------------------------------------------------------------------------
    @staticmethod
    def _np(t):
        return t[0].detach().double().cpu().numpy().copy()

    def _tap(self, obj, name, fn):
        orig = getattr(obj, name)

        def wrapped(*a, **k):
            fn(*a, **k)
            return orig(*a, **k)
        setattr(obj, name, wrapped)

    def _install(self):
        env, rec, np_ = self.env, self.rec, self._np
        self._tap(env.ekf, "propagate", lambda accel, gyro, *r, **k: rec.update(accel=np_(accel), gyro=np_(gyro)))
        self._tap(env.ekf, "update_magnetometer", lambda mag, *r, **k: rec.update(mag=np_(mag)))
        # arg 2 of update_dvl is the 'DVL not locked' flag (1.0 = blind)
        self._tap(env.ekf, "update_dvl", lambda dvl, sub, blind, *r, **k: rec.update(dvl=np_(dvl), dvl_ok=bool(float(blind[0, 0]) < 0.5)))
        self._tap(env.ekf, "update_optical_flow", lambda flow, *r, **k: rec.update(flow=np_(flow)))
        self._tap(env.ekf, "update_gnss", lambda p, v, *r, **k: rec.update(gnss_pos=np_(p), gnss_vel=np_(v)))

        orig_alloc = env.robot.allocate_wrench_hybrid

        def alloc(*a, **k):
            out = orig_alloc(*a, **k)
            self.rec["torch_hydro"] = np_(out[2])
            self._run_core()
            if self.drive:
                out = list(out)
                out[2] = torch.tensor(self.out["hydro_surfaces_cmd"], dtype=out[2].dtype, device=out[2].device).view(1, 3)
                env.scull_throttle_cmd = torch.full_like(env.scull_throttle_cmd, float(self.out["scull_throttle"]))
                out = tuple(out)
            return out
        env.robot.allocate_wrench_hybrid = alloc

        orig_int = env.robot.integrate

        def integ(*a, **k):
            # positional layout of ConfigurableRobot.integrate: 8=target_motors 9=target_tilts 10=target_pusher
            self.rec["torch_motors"] = np_(a[8])
            self.rec["torch_tilts"] = np_(a[9])
            self.rec["torch_pusher"] = float(a[10][0, 0])
            if self.drive and self.out is not None:
                a = list(a)
                a[8] = torch.tensor(self.out["target_motors"], dtype=a[8].dtype, device=a[8].device).view(1, 4)
                a[9] = torch.tensor(self.out["target_tilts"], dtype=a[9].dtype, device=a[9].device).view(1, 4, 2)
                a[10] = torch.full_like(a[10], float(self.out["target_pusher"]))
                a = tuple(a)
            return orig_int(*a, **k)
        env.robot.integrate = integ

    def _run_core(self):
        env, rec = self.env, self.rec
        run_fast = ((env.step_count - 1) % env.sensor_period_steps == 0)
        t0 = time.perf_counter_ns()
        self.out = self.core.step(
            rec["accel"], rec["gyro"], run_fast_sensors=run_fast, mag_meas=rec.get("mag"),
            dvl_meas=rec.get("dvl"), dvl_ok=rec.get("dvl_ok", False), flow_rates=rec.get("flow"),
            radar_raw=float(env.last_radar_raw[0, 0]), radar_locked=bool(env.last_radar_locked[0, 0]),
            d_sub_hydrostatic=float(env.last_d_sub_hydrostatic[0, 0]),
            eta_fwd_meas=float(env.last_eta_fwd_meas[0, 0]), eta_fwd_valid=bool(env.last_eta_fwd_valid[0, 0]),
            run_gnss=("gnss_pos" in rec), gnss_pos=rec.get("gnss_pos"), gnss_vel=rec.get("gnss_vel"),
            v_batt=self._v_batt_pre, mode_cmd=self._mode_pre, y_ref=float(env.y_ref[0, 0]))
        self.latency_us = (time.perf_counter_ns() - t0) / 1000.0

    # -- public API -------------------------------------------------------------------------------
    def reset_core(self, mode=None):
        """Re-initialise the core from the env's current true state (call after env.reset())."""
        env = self.env
        self.core.reset(env.pos[0].numpy(), env.vel[0].numpy(), env.rot[0].numpy(),
                        mode=int(env.current_mode[0, 0]) if mode is None else int(mode),
                        h_target=float(env.h_target_filtered[0, 0]), v_target=float(env.v_target_filtered[0, 0]),
                        clearance=float(env.clearance_hold[0, 0]))
        self.out = None

    @staticmethod
    def _q_angle_deg(q1, q2):
        return math.degrees(2.0 * math.acos(min(1.0, abs(float(np.dot(q1, q2))))))

    def step(self, actor_net, ncm_net):
        env = self.env
        self.rec.clear()
        self.out = None
        self._v_batt_pre = float(env.v_batt[0, 0])
        self._mode_pre = int(env.current_mode[0, 0])
        with torch.no_grad():
            m = env.step(actor_net, ncm_net, export_diagnostics=False)
        o, core, rec = self.out, self.core, self.rec
        e = lambda t: t[0].detach().double().cpu().numpy()
        self.cmp = {
            "p_err": float(np.linalg.norm(core.p_hat - e(env.ekf.p_hat))),
            "v_err": float(np.linalg.norm(core.v_hat - e(env.ekf.v_hat))),
            "att_err_deg": self._q_angle_deg(core.q_hat, e(env.ekf.q_hat)),
            "clr_err": abs(float(o["est_clearance"]) - float(m["clearance_est"][0, 0])),
            "a_safe_err": float(np.linalg.norm(o["a_safe"] - e(m["a_safe"]))),
            "motors_err": float(np.abs(o["target_motors"] - rec["torch_motors"]).max()),
            "tilt_err": float(np.abs(o["target_tilts"] - rec["torch_tilts"]).max()),
            "pusher_err": abs(float(o["target_pusher"]) - rec["torch_pusher"]),
            "hydro_err_deg": float(np.degrees(np.abs(o["hydro_surfaces_cmd"] - rec["torch_hydro"]).max())),
            "mode_eq": 1.0 if int(o["mode"]) == int(env.current_mode[0, 0]) else 0.0,
        }
        return m


# =============================================================================
# 5. CHECKPOINT SAVE / LOAD + TRAINING WRAPPER
# =============================================================================
def train_and_save_checkpoint(ckpt_path, epochs=96, num_envs=2048, steps=200, tbptt=10,
                              fixed_foil=False, gimbal_deg=15.0, mass_kg=None, harsh_sim=False):
    orig_env_cls = base.WaveRidingAutonomyEnv

    def _patched_env(*args, **kwargs):
        return HardenedAutonomyEnv(
            *args,
            fixed_foil=fixed_foil,
            gimbal_max_deg=gimbal_deg,
            mass_kg=mass_kg,
            harsh_physics=harsh_sim,
            **kwargs
        )

    base.WaveRidingAutonomyEnv = _patched_env
    try:
        import inspect
        extra = {"realworld": False} if "realworld" in inspect.signature(base.train_wave_riding_pilot).parameters else {}
        actor_net, ncm_net, morph = base.train_wave_riding_pilot(
            epochs=epochs, num_envs=num_envs, rollout_steps=steps, tbptt_steps=tbptt, **extra
        )
    finally:
        base.WaveRidingAutonomyEnv = orig_env_cls

    torch.save({
        "actor_state_dict": actor_net.state_dict(),
        "ncm_state_dict": ncm_net.state_dict(),
        "morph_state_dict": morph.state_dict(),
        "config": {
            "fixed_foil": fixed_foil,
            "gimbal_deg": gimbal_deg,
            "mass_kg": mass_kg,
            "harsh_sim": harsh_sim,
        }
    }, ckpt_path)
    print(f">> [CHECKPOINT] Saved trained models & morphology to '{ckpt_path}'\n")
    return actor_net, ncm_net, morph


def load_or_train(ckpt_path, auto_train_epochs=24, auto_envs=1024):
    actor_net = base.FullNeuralWavePilot(state_dim=20, action_dim=3).to(DEVICE)
    ncm_net = base.NeuralContractionMetric(state_dim=6, alpha=0.10).to(DEVICE)
    morph = base.VehicleMorphologyGenome(DEVICE)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        n_in = int(ckpt["actor_state_dict"]["net.0.weight"].shape[1])
        if n_in != 20:
            raise SystemExit(
                f"'{ckpt_path}' is a {n_in}-input pilot (engine real-world conditions pack + environment sensors). "
                f"The SIL suite is not ported to that pack yet: use a 20-input checkpoint (train with the engine's "
                f"--legacy-physics, or let this suite train one).")
        actor_net.load_state_dict(ckpt["actor_state_dict"])
        ncm_net.load_state_dict(ckpt["ncm_state_dict"])
        morph.load_state_dict(ckpt["morph_state_dict"])
        print(f">> [CHECKPOINT] Loaded '{ckpt_path}' (cfg={ckpt.get('config', {})})")
        return actor_net, ncm_net, morph

    print(f">> [NOTICE] '{ckpt_path}' not found. Running {auto_train_epochs}-epoch training to generate it...")
    return train_and_save_checkpoint(ckpt_path, epochs=auto_train_epochs, num_envs=auto_envs)


# =============================================================================
# 6. FASTFLIGHTCORE vs PYTORCH PIPELINE: DECISION PARITY + 200 HZ LATENCY
# =============================================================================
# p95 tolerances for "the NumPy core makes the same decision as the PyTorch pipeline when both see the
# same raw sensor samples". The two EKFs run in different precision (float64 vs float32), so exact
# equality is not expected; these are ~2x above what the shipped engine shows.
PARITY_TOL_P95 = {
    "p_err": 0.02, "v_err": 0.05, "att_err_deg": 0.5, "clr_err": 0.005, "a_safe_err": 0.25,
    "motors_err": 0.6, "tilt_err": 0.08, "pusher_err": 0.5, "hydro_err_deg": 0.5,
}
PARITY_MODE_AGREE_MIN = 0.99
CORE_BUDGET_US = 5000.0  # one 200 Hz tick


def _cpu_copies(actor_net, ncm_net, morph_genome):
    cpu = torch.device("cpu")
    actor_cpu = base.FullNeuralWavePilot(20, 3).to(cpu).eval()
    ncm_cpu = base.NeuralContractionMetric(6, 0.10).to(cpu).eval()
    actor_cpu.load_state_dict({k: v.detach().cpu() for k, v in actor_net.state_dict().items()})
    ncm_cpu.load_state_dict({k: v.detach().cpu() for k, v in ncm_net.state_dict().items()})
    morph_cpu = base.VehicleMorphologyGenome(cpu).eval()
    morph_cpu.load_state_dict({k: v.detach().cpu() for k, v in morph_genome.state_dict().items()})
    return cpu, actor_cpu, ncm_cpu, morph_cpu


def reset_env_into_mode(env, mode, max_tries=400):
    """Reset a single-vehicle env until its randomly drawn start mode is `mode` (initial conditions are then
    the engine's own mode-consistent ones: e.g. TAKEOFF slow and low, FLIGHT fast and high). FOILING uses the
    nominal foiling-corridor start."""
    if mode == base.MODE_FOILING:
        env.reset(mode_mix=False)
        env.set_mode(mode)
        return
    for _ in range(max_tries):
        env.reset(mode_mix=True)
        if int(env.current_mode[0, 0]) == int(mode):
            return
    raise RuntimeError(f"could not draw a start state in mode {MODE_NAMES[mode]}")


DEFAULT_PARITY_CASES = [
    (base.MODE_FOILING, False), (base.MODE_TAKEOFF, False), (base.MODE_EGRESS, False),
    (base.MODE_FREE_FLIGHT, False), (base.MODE_INGRESS, False), (base.MODE_SOARING, False),
    (base.MODE_LOITER, False), (base.MODE_FOILING, True), (base.MODE_SOARING, True),
]


def run_fast_core_benchmark(actor_net, ncm_net, morph_genome, steps=1000, dt=None, cases=None, seed=7,
                            fixed_foil=False, gimbal_deg=15.0, verbose=True):
    """
    Shadow-mode parity: the PyTorch pipeline flies the simulated vehicle while FastFlightCore receives the
    identical raw sensor samples every tick; their estimates and actuator commands are compared. Cases start
    in each flight mode (nominal physics) plus FOILING/SOARING under harsh physics. Also times core.step().
    """
    dt = base.CONTROL_LOOP_DT if dt is None else float(dt)
    cases = DEFAULT_PARITY_CASES if cases is None else cases
    cpu, actor_cpu, ncm_cpu, morph_cpu = _cpu_copies(actor_net, ncm_net, morph_genome)
    with torch.no_grad():
        m_dict = morph_cpu()

    if verbose:
        print("\n" + "=" * 118)
        print(f" [FASTFLIGHTCORE vs PYTORCH PIPELINE] identical raw sensor stream | {steps} ticks/case @ {1.0/dt:.0f} Hz | {len(cases)} cases")
        print("=" * 118)
        print(f"  {'case':<16}| {'pos[m]':>7} {'vel[m/s]':>8} {'att[°]':>7} {'clr[m]':>7} {'a_safe':>7} {'motor[N]':>8} {'tilt':>6} {'push[N]':>7} {'elev[°]':>7} | {'mode==':>6} | verdict")
        print(f"  {'(p95 of |core - torch|)':<16}|")

    lat_all, worst, failures, rows = [], {k: 0.0 for k in PARITY_TOL_P95}, [], []
    for mode, harsh in cases:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = HardenedAutonomyEnv(num_envs=1, dt=dt, device=cpu, use_embedded_qp=True, harsh_physics=harsh,
                                  fixed_foil=fixed_foil, gimbal_max_deg=gimbal_deg)
        env.morphology.load_state_dict(morph_cpu.state_dict())
        env.set_wave_curriculum(1.0)
        reset_env_into_mode(env, mode)
        core = FastFlightCore(actor_cpu, ncm_cpu, m_dict, dt=dt, fixed_foil=fixed_foil, gimbal_max_deg=gimbal_deg,
                              mass_scale=env.mass_scale)
        harness = CoreHarness(env, core, drive=False)
        harness.reset_core(mode)

        hist = {k: [] for k in CoreHarness.CMP_KEYS}
        lat = []
        warm = min(100, steps // 4)          # skip start-up ticks (first-call overheads)
        for k in range(steps):
            harness.step(actor_cpu, ncm_cpu)
            for key in hist:
                hist[key].append(harness.cmp[key])
            if k >= warm:
                lat.append(harness.latency_us)
        lat_all.extend(lat)

        p95 = {key: float(np.percentile(hist[key], 95)) for key in PARITY_TOL_P95}
        mx = {key: float(np.max(hist[key])) for key in PARITY_TOL_P95}
        agree = float(np.mean(hist["mode_eq"]))
        bad = [key for key in PARITY_TOL_P95 if p95[key] > PARITY_TOL_P95[key]]
        if agree < PARITY_MODE_AGREE_MIN:
            bad.append("mode_eq")
        name = f"{MODE_NAMES[mode]}{'+harsh' if harsh else ''}"
        for key in worst:
            worst[key] = max(worst[key], mx[key])
        if bad:
            failures.append((name, bad))
        rows.append((name, p95, agree, bad))
        if verbose:
            print(f"  {name:<16}| {p95['p_err']:7.4f} {p95['v_err']:8.4f} {p95['att_err_deg']:7.3f} {p95['clr_err']:7.4f} "
                  f"{p95['a_safe_err']:7.4f} {p95['motors_err']:8.3f} {p95['tilt_err']:6.3f} {p95['pusher_err']:7.3f} "
                  f"{p95['hydro_err_deg']:7.3f} | {100*agree:5.1f}% | {'OK' if not bad else 'OUT OF TOL: ' + ','.join(bad)}")

    lat_arr = np.asarray(lat_all, dtype=np.float64)
    p50, p95l, p99, pmax = (float(np.percentile(lat_arr, q)) for q in (50, 95, 99, 100))
    over = int(np.sum(lat_arr > CORE_BUDGET_US))
    lat_ok = p99 < CORE_BUDGET_US
    parity_ok = not failures
    if verbose:
        print(f"  {'tolerance (p95)':<16}| {PARITY_TOL_P95['p_err']:7.4f} {PARITY_TOL_P95['v_err']:8.4f} {PARITY_TOL_P95['att_err_deg']:7.3f} "
              f"{PARITY_TOL_P95['clr_err']:7.4f} {PARITY_TOL_P95['a_safe_err']:7.4f} {PARITY_TOL_P95['motors_err']:8.3f} "
              f"{PARITY_TOL_P95['tilt_err']:6.3f} {PARITY_TOL_P95['pusher_err']:7.3f} {PARITY_TOL_P95['hydro_err_deg']:7.3f} | {100*PARITY_MODE_AGREE_MIN:5.1f}%")
        print(f"  {'worst single tick':<16}| {worst['p_err']:7.4f} {worst['v_err']:8.4f} {worst['att_err_deg']:7.3f} {worst['clr_err']:7.4f} "
              f"{worst['a_safe_err']:7.4f} {worst['motors_err']:8.3f} {worst['tilt_err']:6.3f} {worst['pusher_err']:7.3f} {worst['hydro_err_deg']:7.3f} |")
        print("  " + "-" * 114)
        print(f"  PARITY : {'PASS' if parity_ok else 'FAIL'}  ({len(cases) - len(failures)}/{len(cases)} cases within tolerance)")
        print(f"  LATENCY: core.step() over {len(lat_arr)} ticks | P50 {p50:6.0f} us | P95 {p95l:6.0f} us | P99 {p99:6.0f} us | max {pmax:7.0f} us "
              f"| {over} tick(s) > {CORE_BUDGET_US:.0f} us  [{'PASS' if lat_ok else 'FAIL'} on P99 < {CORE_BUDGET_US:.0f} us]")
        print("           (wall-clock on THIS machine while the PyTorch reference shares the CPU; it says nothing about a flight computer,")
        print("            and max/over-budget ticks are mostly OS scheduling jitter - check them on the target hardware.)")
        print("=" * 118 + "\n")
    return {"ok": parity_ok and lat_ok, "parity_ok": parity_ok, "latency_ok": lat_ok,
            "p50_us": p50, "p95_us": p95l, "p99_us": p99, "max_us": pmax, "rows": rows, "worst": worst}


# =============================================================================
# 7. 6-SCENARIO MONTE CARLO VALIDATION (SIMULATION ONLY - NOT A CERTIFICATION)
# =============================================================================
def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval for k events in n independent sorties."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2.0 * n)
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def _fmt_events(k, n, note=""):
    lo, hi = wilson_interval(k, n)
    if k == 0:
        return f"0/{n} (<{100*hi:.1f}%)"
    return f"{k}/{n}={100*k/n:.1f}% ({100*lo:.1f}-{100*hi:.1f}%){note}"


MC_SCENARIOS = [
    ("1. Nominal baseline (active foil + 15° gimbal)", dict(fixed_foil=False, gimbal_max_deg=15.0, mass_kg=None, harsh_physics=False)),
    ("2. Harsh physics (350 N slam cap, Wagner lag, aeration, sensor blackouts)", dict(fixed_foil=False, gimbal_max_deg=15.0, mass_kg=None, harsh_physics=True)),
    ("3. ABLATION: fixed passive foil (no underwater servos)", dict(fixed_foil=True, gimbal_max_deg=15.0, mass_kg=None, harsh_physics=True)),
    ("4. ABLATION: zero-gimbal rotors (0° tilt, hydro surfaces only)", dict(fixed_foil=False, gimbal_max_deg=0.0, mass_kg=None, harsh_physics=True)),
    ("5. STRESS: heavy airframe (3.2 kg total, inertia scaled) + harsh physics", dict(fixed_foil=False, gimbal_max_deg=15.0, mass_kg=3.2, harsh_physics=True)),
    ("6. FAULTS: motor M2 -35% at 2.5 s, GNSS off 1.5-4.5 s, subsurface obstacles", dict(fixed_foil=False, gimbal_max_deg=15.0, mass_kg=None, harsh_physics=True, enable_obstacles=True,
                                                                                     gnss_denied_window=(1.5, 4.5), fault_scenario="motor_loss", fault_time=2.5)),
]


def run_monte_carlo_suite(actor_net, ncm_net, morph_genome, eval_envs=256, steps=350, seed=1, settle_s=1.0,
                          strut_limit_nm=STRUT_YIELD_LIMIT_NM, scenarios=None, run_benchmark=True, dt=0.015,
                          bench_steps=1000):
    """
    Runs each scenario for `eval_envs` independent simulated sorties (same seed -> same waves / initial
    conditions in every scenario, so rows are directly comparable). Events are counted PER SORTIE: a sortie
    is flagged if the condition occurs at least once after `settle_s`. Occupancy (% of all env-steps spent in
    the condition) is printed beside it. 95% Wilson intervals are shown because 0 observed events in N
    sorties only bounds the true rate below ~3/N.
    """
    scenarios = MC_SCENARIOS if scenarios is None else scenarios
    duration = steps * dt
    print("\n" + "=" * 124)
    print(f" [MONTE CARLO VALIDATION - SIMULATION ONLY] {eval_envs} sorties x {steps} steps ({duration:.2f} s each) | seed {seed} | "
          f"first {settle_s:.1f} s excluded from event flags | strut allowable {strut_limit_nm:.1f} N·m (assumed)")
    print("=" * 124)

    actor_net.eval()
    ncm_net.eval()
    settle_steps = int(round(settle_s / dt))
    summary = []

    for title, cfg in scenarios:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = HardenedAutonomyEnv(num_envs=eval_envs, dt=dt, device=DEVICE, use_embedded_qp=True,
                                  strut_limit_nm=strut_limit_nm, **cfg)
        env.morphology.load_state_dict(morph_genome.state_dict())
        env.set_wave_curriculum(1.0)
        env.reset(mode_mix=False)  # FOILING corridor over the full 32-wave sea (same start in every scenario)

        N = eval_envs
        zeros_b = lambda: torch.zeros((N, 1), dtype=torch.bool, device=DEVICE)
        ev = {k: zeros_b() for k in ("slam", "breach", "capsize", "strut", "cav", "hit", "nan")}
        occ = {k: 0.0 for k in ("slam", "breach", "capsize", "strut", "cav", "hit")}
        peak_strut = torch.zeros((N, 1), device=DEVICE)
        min_sigma = float("inf")
        dh_sum = dy_sum = p_sum = 0.0
        ekf_h_sq = torch.zeros((N, 1), device=DEVICE)      # per-sortie sums of squares (RMSE per sortie)
        ekf_att_sq = torch.zeros((N, 1), device=DEVICE)
        n_eval = 0

        with torch.no_grad():
            for s in range(steps):
                if s == 120:
                    # mid-sortie wave slap / lateral gust upset
                    env.vel = (env.vel + torch.randn_like(env.vel) * torch.tensor([0.5, 0.7, 0.35], device=DEVICE)).clamp(-15.0, 15.0)
                    env.omega = (env.omega + torch.randn_like(env.omega) * 0.5).clamp(-6.0, 6.0)
                m = env.step(actor_net, ncm_net, export_diagnostics=False)
                bad_state = (torch.isnan(env.pos).any(-1, keepdim=True) | torch.isnan(env.vel).any(-1, keepdim=True)
                             | torch.isnan(env.rot).flatten(1).any(-1, keepdim=True))
                ev["nan"] |= bad_state
                if s < settle_steps:
                    continue
                n_eval += 1
                h = m["clearance_eval"]
                flags = {
                    "slam": h < 0.080,
                    "breach": h > 0.235,
                    "capsize": m["phi_true_deg"].abs() > 25.0,
                    "strut": m["strut_yield_fail"] > 0.5,
                    "cav": m["cav_active"] > 0.5,
                    "hit": m["obstacle_hit"].bool(),
                }
                for k_, f_ in flags.items():
                    ev[k_] |= f_
                    occ[k_] += f_.float().mean().item()
                peak_strut = torch.maximum(peak_strut, m["strut_moment_total_nm"])
                min_sigma = min(min_sigma, float(m["sigma_cav_min"].min().item()))
                dh_sum += (h - m["h_target_mode"]).abs().mean().item() * 100.0
                dy_sum += (m["pos_eval"][:, 1:2] - m["y_ref"]).abs().mean().item()
                ekf_h_sq += m["ekf_err_h"] ** 2
                ekf_att_sq += m["ekf_err_att_deg"] ** 2
                p_sum += m["p_elec"].mean().item()

        n_eval = max(1, n_eval)
        cnt = {k: int(v.sum().item()) for k, v in ev.items()}
        pk = peak_strut.flatten()
        rms_h = (torch.sqrt(ekf_h_sq / n_eval).flatten() * 100.0)      # cm, one value per sortie
        rms_att = torch.sqrt(ekf_att_sq / n_eval).flatten()            # deg, one value per sortie
        q3 = lambda t: tuple(float(torch.quantile(t, q).item()) for q in (0.5, 0.95, 1.0))
        (h50, h95, h100), (a50, a95, a100) = q3(rms_h), q3(rms_att)
        print(f"  {title}")
        print(f"     ├─ Tracking: |Δh| {dh_sum/n_eval:4.1f} cm | |Δy| {dy_sum/n_eval:4.2f} m | mean power {p_sum/n_eval:5.1f} W")
        print(f"     ├─ EKF RMSE per sortie (median / p95 / worst): height {h50:4.1f} / {h95:4.1f} / {h100:5.1f} cm | attitude {a50:4.2f} / {a95:4.2f} / {a100:5.2f}°")
        print(f"     ├─ Sorties with ≥1 event [95% CI]: slam(<8cm) {_fmt_events(cnt['slam'], N)} | breach(>23.5cm) {_fmt_events(cnt['breach'], N)} | "
              f"capsize(>25°) {_fmt_events(cnt['capsize'], N)}")
        cav_cap = int((ev["cav"] & ev["capsize"]).sum().item())
        strut_cap = int((ev["strut"] & ev["capsize"]).sum().item())
        print(f"     │                                  strut>{strut_limit_nm:.0f}N·m {_fmt_events(cnt['strut'], N, f' [{strut_cap} also capsized]' if cnt['strut'] else '')} | "
              f"cavitation {_fmt_events(cnt['cav'], N, f' [{cav_cap} also capsized]' if cnt['cav'] else '')} | "
              f"obstacle hit {_fmt_events(cnt['hit'], N)} | NaN {_fmt_events(cnt['nan'], N)}")
        print(f"     └─ Occupancy (% of env-steps): slam {100*occ['slam']/n_eval:5.2f} | breach {100*occ['breach']/n_eval:5.2f} | capsize {100*occ['capsize']/n_eval:5.2f} | "
              f"hit {100*occ['hit']/n_eval:5.2f}   Peak strut moment: median {float(pk.median()):.1f} / max {float(pk.max()):.1f} N·m | min cavitation σ {min_sigma:.2f} (limit {base.SIGMA_CAV_CRIT:.2f})")
        summary.append((title, cnt, N))

    print("=" * 124)
    print(" Reading guide: simulated vehicle in the engine's own plant model, short sorties. A 0 means 'not seen in N sorties' (95% bound")
    print(" shown), not 'cannot happen'. Cavitation sigma stays far above the limit while the foil is controlled; the flag only trips in")
    print(" sorties that are already out of control (huge foil angle of attack). The strut allowable is an assumed number - use your own.")
    print("=" * 124 + "\n")
    if run_benchmark:
        run_fast_core_benchmark(actor_net, ncm_net, morph_genome, steps=bench_steps, seed=seed)
    return summary


# =============================================================================
# 8. HARDENED INTERACTIVE COCKPIT (NUMPY FLIGHT CORE FLIES THE PLANT)
# =============================================================================
def run_hardened_cockpit(actor_net, ncm_net, morph_genome, fixed_foil=False, gimbal_deg=15.0,
                         mass_kg=None, harsh_sim=False, enable_obstacles=False, fault_scenario=None,
                         fault_time=4.0, core_drives=True, seed=None, strut_limit_nm=STRUT_YIELD_LIMIT_NM,
                         max_frames=None, frame_hook=None, screenshot_path=None):
    """
    Interactive 3-D cockpit on one simulated vehicle at 200 Hz (dt = 5 ms).
      core_drives=True : the NumPy FastFlightCore's commands are what the plant receives (key C toggles).
      core_drives=False: the PyTorch pipeline flies and the core shadows it.
    Either way the core and the PyTorch pipeline see the same raw sensor samples every tick, so the HUD can show
    how closely they agree. Keys: 1 LOITER 2 TAKEOFF 3 FOILING 4 EGRESS 5 INGRESS 6 SOARING 7 FLIGHT,
    C toggle controller, R reset, SPACE pause, mouse drag orbit, wheel zoom, ESC quit.
    `max_frames`, `frame_hook(frame_idx, state)` and `screenshot_path` exist for automated/headless tests.
    Returns a small dict of run statistics.
    """
    if pygame is None:
        print("[ERROR] pygame is required for the cockpit (pip install pygame).")
        return None
    torch.set_num_threads(1)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    pygame.init()
    SCREEN_W, SCREEN_H = 1260, 800
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Aeromarine SIL Cockpit - NumPy FastFlightCore @ 200 Hz")
    clock = pygame.time.Clock()
    font_hud = pygame.font.SysFont("consolas", 13, bold=True)
    font_chart = pygame.font.SysFont("consolas", 11)

    cpu, actor_cpu, ncm_cpu, morph_cpu = _cpu_copies(actor_net, ncm_net, morph_genome)
    with torch.no_grad():
        m_cad = morph_cpu()

    sim_dt = base.CONTROL_LOOP_DT
    env = HardenedAutonomyEnv(
        num_envs=1, dt=sim_dt, device=cpu, use_embedded_qp=True,
        fixed_foil=fixed_foil, gimbal_max_deg=gimbal_deg, mass_kg=mass_kg,
        harsh_physics=harsh_sim, enable_obstacles=enable_obstacles, fault_scenario=fault_scenario,
        fault_time=fault_time, strut_limit_nm=strut_limit_nm
    )
    env.morphology.load_state_dict(morph_cpu.state_dict())
    env.set_wave_curriculum(1.0)
    env.reset(mode_mix=False)

    core = FastFlightCore(actor_cpu, ncm_cpu, m_cad, dt=sim_dt, fixed_foil=fixed_foil,
                          gimbal_max_deg=gimbal_deg, mass_scale=env.mass_scale)
    harness = CoreHarness(env, core, drive=bool(core_drives))
    harness.reset_core(base.MODE_FOILING)

    cam_yaw, cam_pitch, cam_dist = 0.75, 0.32, 5.5
    focal_len = (SCREEN_H * 0.5) / math.tan(math.radians(52.0) * 0.5)
    gx = np.linspace(-10.0, 10.0, 15)
    gy = np.linspace(-8.0, 8.0, 13)
    dx_g, dy_g = float(gx[1] - gx[0]), float(gy[1] - gy[0])
    grid_offsets = torch.tensor([[x, y, 0.0] for x in gx for y in gy], device=cpu, dtype=torch.float32)
    pts_tensor = grid_offsets.clone()

    d_arm = base.ARM_LENGTH / math.sqrt(2.0)
    strut_len = m_cad["strut_length"].item()
    body_pts_local = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -base.HULL_DRAFT_MAX],
        [0.0, 0.0, -strut_len],
        [0.0, m_cad["wing_span"].item() * 0.5, 0.0],
        [0.0, -m_cad["wing_span"].item() * 0.5, 0.0],
        [0.0, -m_cad["foil_span"].item() * 0.5, -strut_len],
        [0.0, m_cad["foil_span"].item() * 0.5, -strut_len],
        # rotor hubs, same corner order as the mixer (r_x, r_y)
        [float(core.r_x[0]), float(core.r_y[0]), 0.0],
        [float(core.r_x[1]), float(core.r_y[1]), 0.0],
        [float(core.r_x[2]), float(core.r_y[2]), 0.0],
        [float(core.r_x[3]), float(core.r_y[3]), 0.0],
    ], device=cpu, dtype=torch.float32)

    keymap = {pygame.K_1: base.MODE_LOITER, pygame.K_2: base.MODE_TAKEOFF, pygame.K_3: base.MODE_FOILING,
              pygame.K_4: base.MODE_EGRESS, pygame.K_5: base.MODE_INGRESS, pygame.K_6: base.MODE_SOARING,
              pygame.K_7: base.MODE_FREE_FLIGHT}

    HIST = 300                      # strip chart: 300 samples @ 50 Hz = 6 s
    chart_every = max(1, int(round(0.02 / sim_dt)))
    h_true_hist, h_est_hist, h_tgt_hist = deque(maxlen=HIST), deque(maxlen=HIST), deque(maxlen=HIST)

    ema_lat_us, max_lat_us = 300.0, 0.0
    ema_cmp = {k: (1.0 if k == "mode_eq" else 0.0) for k in CoreHarness.CMP_KEYS}
    worst_cmp = {k: 0.0 for k in CoreHarness.CMP_KEYS if k != "mode_eq"}
    rt_factor = 1.0
    paused = False
    total_steps = 0
    frames = 0
    m = None
    sim_t_prev = 0.0

    class _State:  # handed to frame_hook
        pass
    state = _State()
    state.env, state.harness, state.screen = env, harness, screen

    running = True
    while running:
        frame_dt = min(0.05, clock.tick(60) / 1000.0)
        state.frames, state.paused, state.m = frames, paused, m
        if frame_hook is not None:
            frame_hook(frames, state)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                cam_dist = float(np.clip(cam_dist - event.y * 0.8, 2.5, 18.0))
            elif event.type == pygame.KEYDOWN:
                if event.key in keymap:
                    env.set_mode(keymap[event.key])
                elif event.key == pygame.K_c:
                    harness.drive = not harness.drive
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    env.reset(mode_mix=False)
                    harness.reset_core(base.MODE_FOILING)
                    h_true_hist.clear(); h_est_hist.clear(); h_tgt_hist.clear()
                    sim_t_prev = 0.0

        mouse = pygame.mouse.get_pressed()
        if mouse[0] or mouse[2]:
            dxm, dym = pygame.mouse.get_rel()
            cam_yaw += dxm * 0.008
            cam_pitch = float(np.clip(cam_pitch + dym * 0.008, 0.05, 1.45))
        else:
            pygame.mouse.get_rel()

        # --- simulate: aim for real time (frame_dt / dt ticks), never more than ~14 ms of CPU per frame ---
        n_target = max(1, min(12, int(round(frame_dt / sim_dt))))
        t_frame0 = time.perf_counter()
        n_done = 0
        if not paused:
            while n_done < n_target and (n_done == 0 or (time.perf_counter() - t_frame0) < 0.014):
                m = harness.step(actor_cpu, ncm_cpu)
                n_done += 1
                total_steps += 1
                ema_lat_us = 0.98 * ema_lat_us + 0.02 * harness.latency_us
                max_lat_us = max(max_lat_us, harness.latency_us) if total_steps > 30 else max_lat_us   # skip first-call overhead
                for k_ in ema_cmp:
                    ema_cmp[k_] = 0.98 * ema_cmp[k_] + 0.02 * harness.cmp[k_]
                    if k_ != "mode_eq" and total_steps > 30:
                        worst_cmp[k_] = max(worst_cmp[k_], harness.cmp[k_])
                if total_steps % chart_every == 0:
                    h_true_hist.append(float(m["clearance_eval"][0, 0]))
                    h_est_hist.append(float(m["clearance_est"][0, 0]))
                    h_tgt_hist.append(float(m["h_target_mode"][0, 0]))
            wall = max(1e-6, time.perf_counter() - t_frame0)
            rt_factor = 0.95 * rt_factor + 0.05 * ((n_done * sim_dt) / max(frame_dt, wall))
        if m is None:
            m = harness.step(actor_cpu, ncm_cpu)
            total_steps += 1

        robot_p_t = env.pos[0]
        cam_lookat = robot_p_t.clone()
        cam_pos = torch.tensor([
            float(cam_lookat[0] - cam_dist * math.cos(cam_pitch) * math.cos(cam_yaw)),
            float(cam_lookat[1] - cam_dist * math.cos(cam_pitch) * math.sin(cam_yaw)),
            max(0.5, float(cam_lookat[2].item() + cam_dist * math.sin(cam_pitch)))
        ], device=cpu, dtype=torch.float32)

        screen.fill((8, 14, 22))
        snap_x = math.floor(float(robot_p_t[0].item()) / dx_g) * dx_g
        snap_y = math.floor(float(robot_p_t[1].item()) / dy_g) * dy_g
        pts_tensor.copy_(grid_offsets)
        pts_tensor[:, 0] += snap_x
        pts_tensor[:, 1] += snap_y
        with torch.no_grad():
            eta_grid, _, _, _, _, _ = env.waves.evaluate_wave_kinematics(
                pts_tensor, env.sim_time.repeat(len(pts_tensor), 1), wave_scale=env.wave_curriculum_scale)
        ocean_3d = pts_tensor.clone()
        ocean_3d[:, 2:3] = eta_grid.view(-1, 1)
        grid_2d, grid_valid, _ = base.project_3d_batch(ocean_3d, cam_pos, cam_lookat, SCREEN_W, SCREEN_H, focal_len)

        nx, ny = len(gx), len(gy)
        for ix in range(nx - 1):
            for iy in range(ny - 1):
                idx = ix * ny + iy
                if grid_valid[idx] and grid_valid[(ix + 1) * ny + iy]:
                    p1, p2 = base.to_screen_pt(grid_2d[idx]), base.to_screen_pt(grid_2d[(ix + 1) * ny + iy])
                    if p1 and p2:
                        pygame.draw.line(screen, (22, 58, 92), p1, p2, 1)
                if grid_valid[idx] and grid_valid[ix * ny + iy + 1]:
                    p1, p3 = base.to_screen_pt(grid_2d[idx]), base.to_screen_pt(grid_2d[ix * ny + iy + 1])
                    if p1 and p3:
                        pygame.draw.line(screen, (22, 58, 92), p1, p3, 1)

        if enable_obstacles:
            obs_3d = torch.stack([env.obs_x[0], env.obs_y[0], torch.zeros_like(env.obs_x[0])], dim=-1)
            obs_2d, obs_val, obs_d = base.project_3d_batch(obs_3d, cam_pos, cam_lookat, SCREEN_W, SCREEN_H, focal_len)
            for oi in range(env.num_obs):
                if obs_val[oi]:
                    po = base.to_screen_pt(obs_2d[oi])
                    if po:
                        rad = max(5, int((env.obs_r[0, oi].item() * focal_len) / max(0.5, float(obs_d[oi]))))
                        hit = bool(m["obstacle_hit"][0, 0])
                        pygame.draw.circle(screen, (255, 40, 40) if hit else (255, 120, 60), po, rad, 3 if hit else 2)

        body_3d = robot_p_t.unsqueeze(0) + torch.matmul(body_pts_local, env.rot[0].T)
        c_2d, c_val, _ = base.project_3d_batch(body_3d, cam_pos, cam_lookat, SCREEN_W, SCREEN_H, focal_len)
        pts = [base.to_screen_pt(c_2d[i]) for i in range(body_pts_local.shape[0])]
        if c_val[3] and c_val[4] and pts[3] and pts[4]:
            pygame.draw.line(screen, (240, 245, 255), pts[3], pts[4], 4)
        if c_val[0] and c_val[1] and pts[0] and pts[1]:
            pygame.draw.line(screen, (180, 220, 255), pts[0], pts[1], 6)
        if c_val[1] and c_val[2] and pts[1] and pts[2]:
            pygame.draw.line(screen, (255, 60, 60), pts[1], pts[2], 3)
        if c_val[5] and c_val[6] and pts[5] and pts[6]:
            pygame.draw.line(screen, (0, 255, 200), pts[5], pts[6], 3)
        thr = env.motors[0].detach().numpy()
        for ri in range(4):
            if c_val[7 + ri] and pts[7 + ri]:
                frac = float(np.clip(thr[ri] / max(1e-3, base.MAX_THRUST_PER_MOTOR), 0.0, 1.0))
                col = (int(80 + 175 * frac), int(230 - 150 * frac), 90)
                pygame.draw.circle(screen, col, pts[7 + ri], 4 + int(6 * frac), 2)

        # ---------------- HUD ----------------
        mode_i = int(env.current_mode[0, 0].item())
        who = "NumPy FastFlightCore FLYING (torch = reference)" if harness.drive else "PyTorch pipeline flying (core shadowing)"
        fault_txt = ""
        if fault_scenario is not None:
            fault_txt = f" | FAULT {fault_scenario}: {'ACTIVE' if env.fault_active else f'armed @ {fault_time:.1f}s'}"
        lat_ok = max_lat_us < CORE_BUDGET_US
        c = harness.cmp
        lines = [
            (f"MODE {MODE_NAMES[mode_i]} | {who}", (0, 240, 255)),
            (f"FixedFoil={fixed_foil} Gimbal={gimbal_deg:.0f}° Mass={'CAD' if mass_kg is None else f'{mass_kg:.1f}kg'} Harsh={harsh_sim} Obstacles={enable_obstacles}{fault_txt}", (150, 200, 230)),
            (f"core.step {ema_lat_us:5.0f} us (max {max_lat_us:5.0f}; budget {CORE_BUDGET_US:.0f}) | sim {rt_factor:4.2f}x real-time | t={env.sim_time[0,0].item():6.2f}s{'  [PAUSED]' if paused else ''}",
             (0, 255, 120) if lat_ok else (255, 120, 80)),
            (f"core-vs-torch now: pos {c.get('p_err',0)*1000:5.1f}mm att {c.get('att_err_deg',0):5.2f}° motors {c.get('motors_err',0):4.2f}N a_safe {c.get('a_safe_err',0):5.3f} elev {c.get('hydro_err_deg',0):4.2f}° mode-agree {100*ema_cmp['mode_eq']:4.0f}%", (180, 200, 255)),
            (f"Clearance est {m['clearance_est'][0,0].item():.3f}m true {m['clearance_eval'][0,0].item():.3f}m tgt {m['h_target_mode'][0,0].item():.3f}m | Vx {m['vel'][0,0].item():.2f}m/s | Y {m['pos'][0,1].item():+.2f}m (ref {m['y_ref'][0,0].item():+.2f})", (120, 255, 140)),
            (f"Foil lift {m['f_hydro_lift'][0,0].item():5.1f}N Rotors {m['f_rotors'][0,0].item():5.1f}N Pusher {m['f_pusher'][0,0].item():4.1f}N | {m['p_elec'][0,0].item():5.0f}W {m['v_batt'][0,0].item():.1f}V SoC {m['soc'][0,0].item():.0f}%", (255, 210, 40)),
            (f"Strut moment {m['strut_moment_total_nm'][0,0].item():5.2f} N·m (assumed limit {strut_limit_nm:.0f}) | cavitation σ {m['sigma_cav_min'][0,0].item():5.2f} (limit {base.SIGMA_CAV_CRIT:.2f}){' | OBSTACLE HIT' if bool(m['obstacle_hit'][0,0]) else ''}", (255, 160, 120)),
            ("Keys: [1]LOITER [2]TAKEOFF [3]FOIL [4]EGRESS [5]INGRESS [6]SOAR [7]FLIGHT | [C] controller [R] reset [SPACE] pause | mouse orbit/zoom", (180, 200, 220)),
        ]
        hud = pygame.Rect(12, 12, 900, 20 + 21 * len(lines))
        pygame.draw.rect(screen, (10, 16, 26), hud, border_radius=6)
        pygame.draw.rect(screen, (0, 220, 255), hud, 1, border_radius=6)
        for idx, (txt, col) in enumerate(lines):
            screen.blit(font_hud.render(txt, True, col), (hud.x + 12, hud.y + 10 + idx * 21))

        # strip chart: clearance true / estimated / target with slam & breach limits
        ch = pygame.Rect(12, SCREEN_H - 170, 620, 158)
        pygame.draw.rect(screen, (10, 16, 26), ch, border_radius=6)
        pygame.draw.rect(screen, (60, 90, 120), ch, 1, border_radius=6)
        y_lo, y_hi = 0.0, 0.40
        def _cy(v):
            return int(ch.bottom - 8 - (min(max(v, y_lo), y_hi) - y_lo) / (y_hi - y_lo) * (ch.height - 20))
        for lim, col, lab in ((0.08, (255, 80, 80), "slam 8cm"), (0.235, (255, 170, 60), "breach 23.5cm")):
            pygame.draw.line(screen, col, (ch.x + 4, _cy(lim)), (ch.right - 4, _cy(lim)), 1)
            screen.blit(font_chart.render(lab, True, col), (ch.x + 8, _cy(lim) - 13))
        for hist_, col in ((h_tgt_hist, (120, 120, 255)), (h_est_hist, (255, 220, 80)), (h_true_hist, (0, 255, 160))):
            if len(hist_) > 1:
                xs = [ch.x + 6 + int(i * (ch.width - 12) / (HIST - 1)) for i in range(len(hist_))]
                pygame.draw.lines(screen, col, False, [(x_, _cy(v_)) for x_, v_ in zip(xs, hist_)], 2 if col == (0, 255, 160) else 1)
        screen.blit(font_chart.render("clearance: true (green) est (yellow) target (blue), last 6 s, 0-40 cm", True, (170, 190, 210)), (ch.x + 8, ch.y + 4))

        pygame.display.flip()
        frames += 1
        if max_frames is not None and frames >= max_frames:
            running = False

    if screenshot_path:
        try:
            pygame.image.save(screen, screenshot_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] could not save screenshot: {exc}")
    pygame.quit()
    stats = {
        "frames": frames, "sim_steps": total_steps, "sim_time_s": float(env.sim_time[0, 0].item()),
        "rt_factor": rt_factor, "core_lat_ema_us": ema_lat_us, "core_lat_max_us": max_lat_us,
        "parity_worst": worst_cmp, "mode_agree_ema": ema_cmp["mode_eq"], "final_mode": MODE_NAMES[int(env.current_mode[0, 0])],
        "core_drives": harness.drive,
    }
    return stats


# =============================================================================
# 9. CLI ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Aeromarine SIL validation, ablation & real-time core suite (simulation only)")
    p.add_argument("--base-script", type=str, default=None, help="engine script to import (default: newest Gemini_Novel_Test*SEA_TO_AIR_SCULLING_NN_CONTROL.py next to this file)")
    p.add_argument("--ckpt", type=str, default="aeromarine_ckpt.pt", help="checkpoint file to load (and to save to when --train has no --save-ckpt)")
    p.add_argument("--save-ckpt", type=str, default=None, help="output checkpoint path when --train is used")
    p.add_argument("--train", action="store_true", help="train a policy with the selected hardware/physics flags and save a checkpoint")
    p.add_argument("--eval-suite", action="store_true", help="6-scenario Monte Carlo (per-sortie event rates) + core parity/latency benchmark")
    p.add_argument("--benchmark-core", action="store_true", help="check FastFlightCore against the PyTorch pipeline in every flight mode and time it")
    p.add_argument("--cockpit", action="store_true", help="launch the interactive 3-D cockpit (NumPy core flies the plant)")
    p.add_argument("--epochs", type=int, default=48)
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--eval-envs", type=int, default=256)
    p.add_argument("--eval-steps", type=int, default=350)
    p.add_argument("--bench-steps", type=int, default=1000, help="ticks per parity case (200 Hz)")
    p.add_argument("--seed", type=int, default=1, help="RNG seed for training, eval, benchmark and cockpit (same seed -> same CPU results)")
    p.add_argument("--fixed-foil", action="store_true", help="lock underwater elevons/rudder/sculling to 0 (passive carbon T-foil)")
    p.add_argument("--gimbal-deg", type=float, default=15.0, help="rotor hub gimbal limit in degrees (0-15)")
    p.add_argument("--mass-kg", type=float, default=None, help="total vehicle mass in kg (inertia scales with it); default: CAD mass")
    p.add_argument("--harsh-sim", action="store_true", help="uncapped slams, unsteady lag, cavitation model, aeration & sensor blackouts")
    p.add_argument("--obstacles", action="store_true", help="cockpit: synthetic subsurface hazard corridor")
    p.add_argument("--fault-test", type=str, default=None, choices=["motor_loss", "elevon_jam", "gimbal_jam", "radar_blind"], help="cockpit: inject this fault")
    p.add_argument("--fault-time", type=float, default=4.0, help="cockpit: seconds before the fault is injected")
    p.add_argument("--strut-limit-nm", type=float, default=STRUT_YIELD_LIMIT_NM, help="assumed strut root bending allowable used by the strut metric")
    p.add_argument("--torch-drives", action="store_true", help="cockpit: let the PyTorch pipeline fly and the NumPy core only shadow it")
    p.add_argument("--cockpit-frames", type=int, default=None, help="cockpit: exit after N frames (smoke test)")
    p.add_argument("--cockpit-shot", type=str, default=None, help="cockpit: save a screenshot on exit")
    args = p.parse_args()

    if not (args.train or args.eval_suite or args.benchmark_core or args.cockpit):
        p.print_help()
        print("\nNothing to do: pass one of --train / --eval-suite / --benchmark-core / --cockpit.")
        sys.exit(0)

    torch.manual_seed(args.seed)      # the engine does not seed training; the eval/benchmark/cockpit paths re-seed themselves
    np.random.seed(args.seed)
    print(f">> [ENGINE] {os.path.basename(base.__aeromarine_path__)}")
    if args.eval_suite and (args.obstacles or args.fault_test or args.harsh_sim or args.fixed_foil or args.mass_kg is not None or args.gimbal_deg != 15.0):
        print(">> [NOTE] --eval-suite runs its own six fixed scenarios; --harsh-sim/--fixed-foil/--gimbal-deg/--mass-kg/--obstacles/--fault-test only apply to --train, --benchmark-core and --cockpit.")

    save_target = args.save_ckpt if args.save_ckpt else args.ckpt
    if args.train:
        actor, ncm, morph = train_and_save_checkpoint(
            save_target, epochs=args.epochs, num_envs=args.envs,
            fixed_foil=args.fixed_foil, gimbal_deg=args.gimbal_deg,
            mass_kg=args.mass_kg, harsh_sim=args.harsh_sim
        )
    else:
        actor, ncm, morph = load_or_train(args.ckpt, auto_train_epochs=args.epochs, auto_envs=args.envs)

    if args.benchmark_core:
        run_fast_core_benchmark(actor, ncm, morph, steps=args.bench_steps, seed=args.seed,
                                fixed_foil=args.fixed_foil, gimbal_deg=args.gimbal_deg)
    if args.eval_suite:
        run_monte_carlo_suite(actor, ncm, morph, eval_envs=args.eval_envs, steps=args.eval_steps, seed=args.seed,
                              strut_limit_nm=args.strut_limit_nm, bench_steps=args.bench_steps)
    if args.cockpit:
        run_hardened_cockpit(
            actor, ncm, morph, fixed_foil=args.fixed_foil, gimbal_deg=args.gimbal_deg, mass_kg=args.mass_kg,
            harsh_sim=args.harsh_sim, enable_obstacles=args.obstacles, fault_scenario=args.fault_test,
            fault_time=args.fault_time, core_drives=not args.torch_drives, seed=args.seed,
            strut_limit_nm=args.strut_limit_nm, max_frames=args.cockpit_frames, screenshot_path=args.cockpit_shot
        )
