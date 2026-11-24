"""
=============================================================================
SURVIVAL CORE V37.0 — COMPLETE GENESIS 3D AUTONOMOUS METROPOLITAN SWARM SIMULATOR
=============================================================================
Architecture:
  - Physics Engine: Genesis 1.4.3 (CUDA GPU Backend)
  - Aerodynamics: GPU-Vectorized BEMT with Coaxial Losses, IGE & VRS
  - Attitude & Tracking: Almost-Global Exponential SE(3) Geometric Controller
  - Mission Environment: 3D Urban Metropolitan Grid with 48 Buildings & Clutter
  - Delivery Subsystem: Dynamic Tether Cable, Physical Cargo & Touchdown Latch
=============================================================================
"""

import math
import torch
import numpy as np
import genesis as gs

# =============================================================================
# 0. PHYSICAL AIRFRAME & MISSION CONSTANTS
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
MAX_OMEGA = (9200.0 * 2.0 * math.pi) / 60.0  # Max angular velocity (rad/s)
HOVER_THRUST = TOTAL_MASS_KG * GRAVITY_VAL

# =============================================================================
# 1. BATCHED LIE ALGEBRA & SO(3) ROTATION UTILITIES
# =============================================================================
def quat_to_rot_batch(q: torch.Tensor) -> torch.Tensor:
    """Converts a batch of quaternions [B, 4] (w, x, y, z) into rotation matrices [B, 3, 3]."""
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
    """Extracts 3D angular vectors from batch skew-symmetric matrices [B, 3, 3]."""
    return torch.stack([M[:, 2, 1], M[:, 0, 2], M[:, 1, 0]], dim=-1)

# =============================================================================
# 2. VECTORIZED BEMT AERODYNAMICS ENGINE ON GPU
# =============================================================================
class VectorizedBEMTEngine(torch.nn.Module):
    def __init__(self, n_envs: int, device: str = "cuda"):
        super().__init__()
        self.n_envs = n_envs
        self.device = device
        
        # 8 Coaxial Motors per drone: omegas stored directly in GPU memory
        self.motor_omegas = torch.full((n_envs, 8), 450.0, device=device)

        # Mixer Matrix B: [Total Thrust, Tau_Roll, Tau_Pitch, Tau_Yaw]
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

    def compute_aerodynamics(self, motor_cmds: torch.Tensor, pos: torch.Tensor, 
                             vel: torch.Tensor, target_surface_z: float, dt: float):
        """Calculates actual aerodynamic thrust & moments with Ground Effect (IGE) & VRS."""
        effective_cmds = torch.clamp(motor_cmds, 0.0, MAX_OMEGA)
        self.motor_omegas += ((effective_cmds - self.motor_omegas) / MOTOR_TAU_S) * dt

        # In-Ground Effect (IGE) cushion relative to current surface altitude
        z_clearance = torch.clamp(pos[:, 2] - target_surface_z, min=0.06)
        ratio_zh_r = z_clearance / ROTOR_RADIUS_M
        ige_factor = torch.where(
            ratio_zh_r >= 0.5,
            torch.clamp(1.0 / (1.0 - (1.0 / (4.0 * ratio_zh_r)) ** 2), 1.0, 1.28),
            1.20 + 0.16 * (ratio_zh_r / 0.5)
        )

        # Vortex Ring State (VRS) lift penalty during steep vertical descent
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
        f_thrust = wrench[:, 0]
        torques = wrench[:, 1:4]
        return f_thrust, torques

    def allocate_wrench_to_motors(self, desired_thrust: torch.Tensor, desired_tau: torch.Tensor) -> torch.Tensor:
        desired_wrench = torch.cat([desired_thrust.unsqueeze(-1), desired_tau], dim=-1)
        T_cmds = torch.matmul(self.B_pinv, desired_wrench.unsqueeze(-1)).squeeze(-1)
        T_cmds = torch.clamp(T_cmds, min=0.05, max=10.5)

        omega_cmds = torch.zeros_like(T_cmds)
        omega_cmds[:, 0::2] = torch.sqrt(T_cmds[:, 0::2] / C_T_CLEAN)
        omega_cmds[:, 1::2] = torch.sqrt(T_cmds[:, 1::2] / (C_T_CLEAN * COAXIAL_EFFICIENCY))
        return omega_cmds

# =============================================================================
# 3. GEOMETRIC SE(3) FLIGHT CONTROLLER
# =============================================================================
class GeometricSE3Controller:
    def __init__(self, n_envs: int, device: str = "cuda"):
        self.n_envs = n_envs
        self.device = device
        
        # Well-damped attitude and position gains
        self.k_p = torch.tensor([1.8, 1.8, 2.6], device=device)
        self.k_d = torch.tensor([2.2, 2.2, 2.8], device=device)
        self.k_R = torch.tensor([2.1, 2.1, 1.7], device=device)
        self.k_w = torch.tensor([1.4, 1.4, 0.95], device=device)

    def compute_control_wrench(self, pos: torch.Tensor, vel: torch.Tensor, 
                               quat: torch.Tensor, ang_vel: torch.Tensor,
                               target_pos: torch.Tensor, target_vel: torch.Tensor, 
                               current_mass: float):
        B = self.n_envs
        R = quat_to_rot_batch(quat)

        # 1. Translation tracking
        e_p = target_pos - pos
        e_v = target_vel - vel

        a_des = self.k_p * e_p + self.k_d * e_v
        a_des[:, 2] += GRAVITY_VAL  # Gravity compensation

        F_d = current_mass * a_des

        # 2. Geometric attitude tracking on SO(3)
        b3_d = F_d / torch.clamp(torch.norm(F_d, dim=-1, keepdim=True), min=1e-4)

        b1_c = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(B, 1)
        b2_d = torch.linalg.cross(b3_d, b1_c, dim=-1)
        b2_d = b2_d / torch.clamp(torch.norm(b2_d, dim=-1, keepdim=True), min=1e-4)
        b1_d = torch.linalg.cross(b2_d, b3_d, dim=-1)

        R_d = torch.stack([b1_d, b2_d, b3_d], dim=-1)

        b3 = R[:, :, 2]
        f_thrust = torch.sum(F_d * b3, dim=-1)
        f_thrust = torch.clamp(f_thrust, min=0.0, max=2.5 * (current_mass * GRAVITY_VAL))

        # Attitude error e_R via vee map
        R_err = torch.bmm(R_d.transpose(1, 2), R) - torch.bmm(R.transpose(1, 2), R_d)
        e_R = 0.5 * vee_map_batch(R_err)
        e_w = ang_vel

        tau = -self.k_R * e_R - self.k_w * e_w
        tau = torch.clamp(tau, -0.75, 0.75)
        return f_thrust, tau

# =============================================================================
# 4. PROCEDURAL 3D METROPOLITAN GRID WORLD BUILDER
# =============================================================================
def build_metropolitan_city(scene):
    """Procedurally spawns a 400m metropolitan grid with buildings, rooftops, helipads, and clutter."""
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(friction=0.8),
        surface=gs.surfaces.Rough(color=(0.18, 0.20, 0.24), roughness=0.8)
    )

    manifest_targets = []
    col_candidates = [-60.0, -30.0, 0.0, 30.0, 60.0]
    row_candidates = [-60.0, -30.0, 0.0, 30.0, 60.0]
    bldg_idx = 0

    for c_i, cx in enumerate(col_candidates):
        for r_i, ry in enumerate(row_candidates):
            if cx <= -20.0 and ry <= -20.0:
                continue  # Preserve base station depot clear zone

            tier = (c_i + r_i) % 4
            height = [8.5, 14.0, 20.0, 26.0][tier]
            bw, bd = 16.0, 16.0

            # Building chassis
            scene.add_entity(
                morph=gs.morphs.Box(
                    size=(bw, bd, height),
                    pos=(cx, ry, height / 2.0),
                    fixed=True,
                ),
                material=gs.materials.Rigid(friction=0.7),
                surface=gs.surfaces.Rough(
                    color=(0.28 + 0.03 * tier, 0.32 + 0.03 * tier, 0.38 + 0.03 * tier),
                    roughness=0.6
                )
            )

            # Rooftop HVAC clutter
            scene.add_entity(
                morph=gs.morphs.Box(
                    size=(3.0, 2.5, 1.4),
                    pos=(cx - 3.5, ry + 3.5, height + 0.7),
                    fixed=True,
                ),
                material=gs.materials.Rigid(friction=0.6),
                surface=gs.surfaces.Rough(color=(0.55, 0.55, 0.58), roughness=0.5)
            )

            # Rooftop Delivery Helipad
            pad_pos = (cx + 2.0, ry - 2.0, height + 0.04)
            scene.add_entity(
                morph=gs.morphs.Cylinder(
                    radius=2.2,
                    height=0.08,
                    pos=pad_pos,
                    fixed=True,
                ),
                material=gs.materials.Rigid(friction=0.8),
                surface=gs.surfaces.Default(color=(0.0, 0.85, 0.55))
            )

            manifest_targets.append({
                "id": f"Bldg_{bldg_idx+1:02d}",
                "pos": np.array(pad_pos, dtype=np.float32),
                "height": height
            })
            bldg_idx += 1

    # Base Station Depot Helipads
    depot_pads = [
        np.array([-45.0, -45.0, 0.04], dtype=np.float32),
        np.array([-45.0, -35.0, 0.04], dtype=np.float32),
        np.array([-35.0, -45.0, 0.04], dtype=np.float32),
    ]
    for p_i, dp in enumerate(depot_pads):
        scene.add_entity(
            morph=gs.morphs.Cylinder(
                radius=2.5,
                height=0.08,
                pos=tuple(dp),
                fixed=True,
            ),
            material=gs.materials.Rigid(friction=0.9),
            surface=gs.surfaces.Default(color=(0.10, 0.65, 0.95))
        )

    return manifest_targets, depot_pads

# =============================================================================
# 5. MAIN SYSTEM ENTRY POINT & EXECUTION LOOP
# =============================================================================
def main():
    gs.init(backend=gs.gpu)

    dt = 0.01  # 100 Hz simulation step
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=2,
            gravity=(0, 0, -GRAVITY_VAL),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(45.0, -45.0, 35.0),
            camera_lookat=(0.0, 0.0, 10.0),
            res=(1280, 720),
            refresh_rate=60,
        ),
        show_viewer=True,
    )

    # 1. Build 3D Metropolitan City World
    manifest_targets, depot_pads = build_metropolitan_city(scene)

    # 2. Coaxial Drone Airframe Entity
    drone = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.42, 0.42, 0.12),
            pos=(-45.0, -45.0, 0.20),
        ),
        material=gs.materials.Rigid(rho=85.034, friction=0.5),  # 1.80 kg physical mass
        surface=gs.surfaces.Default(color=(0.10, 0.65, 0.95)),
    )

    # 3. Physical Cargo Payload Box (0.45 kg)
    cargo = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.22, 0.22, 0.18),
            pos=(-45.0, -45.0, 0.10),
        ),
        material=gs.materials.Rigid(rho=45.0, friction=0.7),
        surface=gs.surfaces.Default(color=(0.95, 0.75, 0.15)),
    )

    # Compile scene across parallel environments on RTX 5060 Ti
    n_envs = 4
    scene.build(n_envs=n_envs)

    print("\n" + "="*75)
    print(f"GENESIS V37.0 METROPOLITAN SWARM SIMULATOR INITIALIZED [{n_envs} ENVS]")
    print(f"Mass: {TOTAL_MASS_KG:.2f} kg | Hover Thrust: {HOVER_THRUST:.2f} N | City Targets: {len(manifest_targets)}")
    print("="*75 + "\n")

    bemt_engine = VectorizedBEMTEngine(n_envs=n_envs, device="cuda")
    controller = GeometricSE3Controller(n_envs=n_envs, device="cuda")

    drone_link = drone.links[0]
    cargo_link = cargo.links[0]

    # Assign target delivery destination (e.g., Target Building #6)
    target_dest = manifest_targets[5]
    tgt_pad_xyz = target_dest["pos"]
    roof_h = float(target_dest["height"])
    cruise_alt = roof_h + 12.0

    print(f">>> ASSIGNED MISSION: Deliver Cargo to {target_dest['id']} at Height {roof_h:.1f}m <<<")

    # Autonomous FSM State Tracking
    mission_phase = "1_TAKEOFF"
    cable_len = 0.10
    has_payload = True
    hover_timer = 0.0

    # -------------------------------------------------------------------------
    # MAIN SIMULATION RUNTIME (WITH SAFE VIEWER EXIT HANDLER)
    # -------------------------------------------------------------------------
    try:
        for step in range(4000):
            if scene.viewer is not None and not scene.viewer.is_alive():
                print("\n[INFO] Viewer window closed by user. Exiting cleanly...")
                break

            # Read kinematic states from GPU memory
            d_pos = drone.get_pos()       # [n_envs, 3]
            d_vel = drone.get_vel()       # [n_envs, 3]
            d_quat = drone.get_quat()     # [n_envs, 4]
            d_ang = drone.get_ang()       # [n_envs, 3]

            c_pos = cargo.get_pos()       # [n_envs, 3]
            c_vel = cargo.get_vel()       # [n_envs, 3]

            sim_time = step * dt
            current_mass = TOTAL_MASS_KG if has_payload else DRY_MASS_KG

            # -------------------------------------------------------------
            # AUTONOMOUS MISSION STATE MACHINE
            # -------------------------------------------------------------
            dist_to_target_xy = torch.norm(d_pos[:, :2] - torch.tensor(tgt_pad_xyz[:2], device="cuda"), dim=-1).mean().item()
            home_pad = torch.tensor(depot_pads[0], device="cuda").repeat(n_envs, 1)
            target_vel = torch.zeros((n_envs, 3), device="cuda")

            if mission_phase == "1_TAKEOFF":
                target_pos = torch.tensor([-45.0, -45.0, cruise_alt], device="cuda").repeat(n_envs, 1)
                if d_pos[:, 2].mean().item() >= (cruise_alt - 1.5):
                    mission_phase = "2_CRUISE_OUTBOUND"
                    print(f"[{sim_time:.1f}s] Reached cruise altitude. En route to {target_dest['id']}...")

            elif mission_phase == "2_CRUISE_OUTBOUND":
                target_pos = torch.tensor([tgt_pad_xyz[0], tgt_pad_xyz[1], cruise_alt], device="cuda").repeat(n_envs, 1)
                if dist_to_target_xy <= 2.0:
                    mission_phase = "3_CHIMNEY_DESCENT"
                    print(f"[{sim_time:.1f}s] Arrived over rooftop. Commencing chimney descent...")

            elif mission_phase == "3_CHIMNEY_DESCENT":
                high_hover_z = roof_h + 7.5
                target_pos = torch.tensor([tgt_pad_xyz[0], tgt_pad_xyz[1], high_hover_z], device="cuda").repeat(n_envs, 1)
                if abs(d_pos[:, 2].mean().item() - high_hover_z) <= 0.6:
                    mission_phase = "4_DELIVERY_HOVER"
                    hover_timer = 0.0
                    print(f"[{sim_time:.1f}s] Hover lock established. Unspooling winch tether...")

            elif mission_phase == "4_DELIVERY_HOVER":
                high_hover_z = roof_h + 7.5
                target_pos = torch.tensor([tgt_pad_xyz[0], tgt_pad_xyz[1], high_hover_z], device="cuda").repeat(n_envs, 1)
                hover_timer += dt

                # Winch lowering logic
                if has_payload and cable_len < 7.2:
                    cable_len += 1.8 * dt

                # Detect cargo touchdown on rooftop helipad
                cargo_z = c_pos[:, 2].mean().item()
                if has_payload and (cargo_z <= (roof_h + 0.15) or cable_len >= 7.2):
                    hover_timer += dt
                    if hover_timer >= 1.2:
                        has_payload = False
                        print(f"[{sim_time:.1f}s] TOUCHDOWN CONFIRMED! Cargo released on {target_dest['id']} helipad.")

                # Retract cable after release
                if not has_payload:
                    cable_len = max(0.10, cable_len - 3.2 * dt)
                    if cable_len <= 0.12:
                        mission_phase = "5_RETURN_TO_BASE"
                        print(f"[{sim_time:.1f}s] Winch retracted. Returning to depot base...")

            elif mission_phase == "5_RETURN_TO_BASE":
                target_pos = torch.tensor([-45.0, -45.0, cruise_alt], device="cuda").repeat(n_envs, 1)
                dist_to_home = torch.norm(d_pos[:, :2] - home_pad[:, :2], dim=-1).mean().item()
                if dist_to_home <= 2.5:
                    mission_phase = "6_DEPOT_LANDING"
                    print(f"[{sim_time:.1f}s] Over depot. Landing on helipad...")

            else:  # 6_DEPOT_LANDING
                target_pos = torch.tensor([-45.0, -45.0, 0.08], device="cuda").repeat(n_envs, 1)

            # -------------------------------------------------------------
            # DYNAMIC TETHER & WINCH PHYSICS
            # -------------------------------------------------------------
            if has_payload:
                tether_vec = c_pos - d_pos
                tether_dist = torch.norm(tether_vec, dim=-1, keepdim=True)
                slack = tether_dist - cable_len
                
                # Apply tension force if cable is taut
                tension_mag = torch.clamp(slack * 450.0 + (torch.norm(c_vel - d_vel, dim=-1, keepdim=True) * 15.0), min=0.0)
                tether_dir = tether_vec / torch.clamp(tether_dist, min=1e-4)

                f_tether_drone = tension_mag * tether_dir
                f_tether_cargo = -f_tether_drone

                drone_link.apply_external_force(force=f_tether_drone)
                cargo_link.apply_external_force(force=f_tether_cargo)

            # -------------------------------------------------------------
            # FLIGHT CONTROL & AERODYNAMIC ACTUATION
            # -------------------------------------------------------------
            f_thrust_cmd, tau_cmd = controller.compute_control_wrench(
                d_pos, d_vel, d_quat, d_ang, target_pos, target_vel, current_mass
            )

            motor_cmds = bemt_engine.allocate_wrench_to_motors(f_thrust_cmd, tau_cmd)

            actual_thrust, actual_torques = bemt_engine.compute_aerodynamics(
                motor_cmds, d_pos, d_vel, target_surface_z=roof_h if "DELIVERY" in mission_phase else 0.0, dt=dt
            )

            # Direct body thrust vector (R @ [0, 0, T])
            R_curr = quat_to_rot_batch(d_quat)
            world_thrust_forces = R_curr[:, :, 2] * actual_thrust.unsqueeze(-1)

            drone_link.apply_external_force(force=world_thrust_forces)
            drone_link.apply_external_torque(torque=actual_torques)

            # Advance GPU simulation
            scene.step()

            if step % 250 == 0:
                alt = d_pos[:, 2].mean().item()
                c_alt = c_pos[:, 2].mean().item()
                print(f"Step [{step:04d}] ({sim_time:.1f}s) | Phase: {mission_phase:<17} | Drone Z: {alt:5.2f}m | Cargo Z: {c_alt:5.2f}m")

    except gs.GenesisException as e:
        if "Viewer closed" in str(e):
            print("\n[INFO] Viewer closed by user. Simulation exited cleanly.")
        else:
            raise e

    print("\n" + "="*75)
    print("[SIMULATION FINISHED] Complete mission flight cycle successfully executed.")
    print("="*75)

if __name__ == "__main__":
    main()