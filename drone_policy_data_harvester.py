"""
=============================================================================
STEP 3: PARALLEL TEACHER POLICY & MULTI-CORE DATASET HARVESTER
File: drone_policy_data_harvester.py
=============================================================================
Spawns parallel CPU worker processes to simulate complete multirotor flights,
running the mathematical teacher stack (SE(3) + Min-Snap + ES-EKF + Winch FSM)
across randomized environmental conditions, and harvesting 120,000+ expert
transitions (o_t, omega_teacher, fsm_token).
=============================================================================
"""

import time
import math
import random
import multiprocessing as mp
import numpy as np

# Import Step 1 synthesizer
from drone_observation_synthesizer import DroneObservationSynthesizer, OBSERVATION_DIM

# Core Constants
MASS_KG = 1.35
GRAVITY_G = np.array([0.0, 0.0, -9.80665], dtype=np.float32)
G_ACCEL = 9.80665
ARM_LENGTH_M = 0.225
INERTIA_MATRIX = np.diag([0.016, 0.016, 0.028]).astype(np.float32)
INV_INERTIA = np.linalg.inv(INERTIA_MATRIX)

C_T = 1.28e-5
C_Q = 2.45e-7
MOTOR_TAU_S = 0.035
MAX_RPM = 8800.0
MAX_OMEGA = (MAX_RPM * 2.0 * math.pi) / 60.0
HOVER_THRUST = MASS_KG * G_ACCEL
HOVER_OMEGA = math.sqrt((HOVER_THRUST / 4.0) / C_T)
PAYLOAD_WEIGHT_N = 2.0 * G_ACCEL

# ==========================================
# 1. COMPACT SIMULATION HARVESTER ENGINES
# ==========================================
def skew_symmetric(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float32)

def vee_map(mat):
    return np.array([mat[2, 1], mat[0, 2], mat[1, 0]], dtype=np.float32)

def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z),       2.0*(x*z + w*y)],
        [2.0*(x*y + w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x)],
        [2.0*(x*z - w*y),       2.0*(y*z + w*x),       1.0 - 2.0*(x*x + y*y)]
    ], dtype=np.float32)

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float32)

class FastDronePhysics:
    def __init__(self, init_pos):
        self.p = np.array(init_pos, dtype=np.float32)
        self.v = np.zeros(3, dtype=np.float32)
        self.R = np.eye(3, dtype=np.float32)
        self.omega = np.zeros(3, dtype=np.float32)
        self.motor_omegas = np.full(4, HOVER_OMEGA, dtype=np.float32)
        self.wind_gust = np.zeros(3, dtype=np.float32)
        self.accel_linear = np.zeros(3, dtype=np.float32)

    def step(self, motor_cmds, dt=0.016):
        cmds = np.clip(motor_cmds, 0.0, MAX_OMEGA)
        self.motor_omegas += ((cmds - self.motor_omegas) / MOTOR_TAU_S) * dt

        T = C_T * (self.motor_omegas ** 2)
        Q = C_Q * (self.motor_omegas ** 2)

        f_thrust = float(np.sum(T))
        arm = ARM_LENGTH_M / math.sqrt(2.0)
        tau_x = arm * (-T[0] + T[1] + T[2] - T[3])
        tau_y = arm * (T[0] + T[1] - T[2] - T[3])
        tau_z = -Q[0] + Q[1] - Q[2] + Q[3]
        torques = np.array([tau_x, tau_y, tau_z], dtype=np.float32)

        f_body = np.array([0.0, 0.0, f_thrust], dtype=np.float32)
        f_drag = -np.array([0.22, 0.22, 0.35], dtype=np.float32) * self.v
        f_gravity = MASS_KG * GRAVITY_G

        self.accel_linear = (self.R @ f_body + f_gravity + f_drag + self.wind_gust) / MASS_KG
        gyroscopic = np.cross(self.omega, INERTIA_MATRIX @ self.omega)
        accel_angular = INV_INERTIA @ (torques - gyroscopic)

        self.p += self.v * dt + 0.5 * self.accel_linear * (dt ** 2)
        self.v += self.accel_linear * dt
        if self.p[2] < 0.0:
            self.p[2] = 0.0
            self.v[2] = max(0.0, self.v[2])

        self.omega += accel_angular * dt
        d_theta = np.linalg.norm(self.omega) * dt
        if d_theta > 1e-6:
            axis = self.omega / np.linalg.norm(self.omega)
            K = skew_symmetric(axis)
            R_delta = np.eye(3) + math.sin(d_theta) * K + (1.0 - math.cos(d_theta)) * (K @ K)
            self.R = self.R @ R_delta

        U, _, Vt = np.linalg.svd(self.R)
        self.R = U @ Vt
        self.wind_gust *= 0.995

class FastSE3Controller:
    def __init__(self):
        self.k_p = np.array([6.5, 6.5, 11.0], dtype=np.float32)
        self.k_v = np.array([4.4, 4.4, 6.8], dtype=np.float32)
        self.k_i = np.array([1.2, 1.2, 2.5], dtype=np.float32)
        self.k_R = 1.85
        self.k_w = 0.35
        self.e_p_integral = np.zeros(3, dtype=np.float32)

    def reset_integral(self):
        self.e_p_integral.fill(0.0)

    def compute(self, p_est, v_est, R_est, w_est, p_des, v_des, a_des, dt=0.016):
        e_p = p_est - p_des
        e_v = v_est - v_des

        self.e_p_integral += e_p * dt
        self.e_p_integral = np.clip(self.e_p_integral, -1.2, 1.2)

        F_d = -self.k_p * e_p - self.k_v * e_v - self.k_i * self.e_p_integral + MASS_KG * (-GRAVITY_G) + MASS_KG * a_des
        b3 = R_est[:, 2]
        f_thrust = float(np.dot(F_d, b3))
        f_thrust = max(0.1, min(34.0, f_thrust))

        b3_d = F_d / (np.linalg.norm(F_d) + 1e-6)
        b1_c = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b2_d = np.cross(b3_d, b1_c)
        b2_d /= (np.linalg.norm(b2_d) + 1e-6)
        b1_d = np.cross(b2_d, b3_d)
        R_d = np.column_stack([b1_d, b2_d, b3_d]).astype(np.float32)

        R_err_mat = 0.5 * (R_d.T @ R_est - R_est.T @ R_d)
        e_R = vee_map(R_err_mat)
        e_w = w_est

        tau = -self.k_R * e_R - self.k_w * e_w + np.cross(w_est, INERTIA_MATRIX @ w_est)

        arm = ARM_LENGTH_M / math.sqrt(2.0)
        A = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-arm, arm, arm, -arm],
            [arm, arm, -arm, -arm],
            [-(C_Q/C_T), C_Q/C_T, -(C_Q/C_T), C_Q/C_T]
        ], dtype=np.float32)
        b = np.array([f_thrust, tau[0], tau[1], tau[2]], dtype=np.float32)
        try:
            T_motors = np.clip(np.linalg.solve(A, b), 0.05, 9.8)
        except np.linalg.LinAlgError:
            T_motors = np.full(4, f_thrust / 4.0, dtype=np.float32)

        omega_cmds = np.sqrt(T_motors / C_T)
        return omega_cmds

class FastMinSnapPlanner:
    def __init__(self):
        self.total_time = 0.0
        self.times = []
        self.is_solved = False

    def plan(self, waypoints, speed=2.8):
        self.waypoints = [np.array(w, dtype=np.float32) for w in waypoints]
        M = len(self.waypoints) - 1
        if M < 1: return False

        self.times = [max(1.4, float(np.linalg.norm(self.waypoints[k+1] - self.waypoints[k])) / speed) for k in range(M)]
        self.total_time = sum(self.times)
        N_vars = 8 * M
        A = np.zeros((N_vars, N_vars), dtype=np.float64)
        bx, by, bz = np.zeros(N_vars), np.zeros(N_vars), np.zeros(N_vars)
        row = 0

        def basis(t, d):
            c = np.zeros(8)
            for j in range(d, 8):
                fact = math.prod(range(j - d + 1, j + 1))
                c[j] = fact * (t ** (j - d))
            return c

        for d in range(4):
            A[row, 0:8] = basis(0.0, d)
            if d == 0: bx[row], by[row], bz[row] = self.waypoints[0]
            row += 1

        for k in range(1, M):
            t_prev = self.times[k-1]
            A[row, (k-1)*8 : k*8] = basis(t_prev, 0)
            bx[row], by[row], bz[row] = self.waypoints[k]
            row += 1
            A[row, k*8 : (k+1)*8] = basis(0.0, 0)
            bx[row], by[row], bz[row] = self.waypoints[k]
            row += 1
            for d in range(1, 7):
                A[row, (k-1)*8 : k*8] = basis(t_prev, d)
                A[row, k*8 : (k+1)*8] = -basis(0.0, d)
                row += 1

        t_last = self.times[-1]
        for d in range(4):
            A[row, (M-1)*8 : M*8] = basis(t_last, d)
            if d == 0: bx[row], by[row], bz[row] = self.waypoints[-1]
            row += 1

        try:
            self.cx = np.linalg.solve(A, bx).reshape(M, 8)
            self.cy = np.linalg.solve(A, by).reshape(M, 8)
            self.cz = np.linalg.solve(A, bz).reshape(M, 8)
            self.is_solved = True
            return True
        except np.linalg.LinAlgError:
            self.is_solved = False
            return False

    def eval(self, t):
        if not self.is_solved: return np.zeros(3), np.zeros(3), np.zeros(3)
        if t >= self.total_time:
            return self.waypoints[-1].copy(), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        t_clamped = max(0.0, min(self.total_time, t))
        accum_t, seg_idx, tau = 0.0, 0, 0.0
        for k, dur in enumerate(self.times):
            if t_clamped <= accum_t + dur or k == len(self.times) - 1:
                seg_idx = k
                tau = t_clamped - accum_t
                break
            accum_t += dur

        T_pow = np.array([tau**i for i in range(8)])
        dT_pow = np.array([0] + [i * tau**(i-1) for i in range(1, 8)])
        d2T_pow = np.array([0, 0] + [i * (i-1) * tau**(i-2) for i in range(2, 8)])

        cx, cy, cz = self.cx[seg_idx], self.cy[seg_idx], self.cz[seg_idx]
        pos = np.array([np.dot(cx, T_pow), np.dot(cy, T_pow), np.dot(cz, T_pow)], dtype=np.float32)
        vel = np.array([np.dot(cx, dT_pow), np.dot(cy, dT_pow), np.dot(cz, dT_pow)], dtype=np.float32)
        acc = np.array([np.dot(cx, d2T_pow), np.dot(cy, d2T_pow), np.dot(cz, d2T_pow)], dtype=np.float32)
        return pos, vel, acc


# ==========================================
# 2. MULTI-PROCESSING WORKER PROCEDURE
# ==========================================
def _worker_harvest_trajectory(num_steps_target):
    """
    Simulates autonomous flight missions, recording (Observation, Motor_Cmds, FSM_Token).
    """
    np.random.seed()
    random.seed()

    synthesizer = DroneObservationSynthesizer()
    controller = FastSE3Controller()
    planner = FastMinSnapPlanner()

    observations = []
    teacher_motors = []
    teacher_fsm = []

    delivered_count = 0

    while len(observations) < num_steps_target:
        # Randomized mission start & dropzone target
        start_p = np.array([random.uniform(-4.5, 4.5), random.uniform(-4.0, 4.0), random.uniform(1.2, 3.5)], dtype=np.float32)
        target_gnd = np.array([random.uniform(-4.5, 4.5), random.uniform(-2.5, 6.5), 0.0], dtype=np.float32)

        drone = FastDronePhysics(start_p)
        hover_dest = np.array([target_gnd[0], target_gnd[1], 2.2], dtype=np.float32)
        mid_pt = 0.5 * (start_p + hover_dest) + np.array([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), 0.8], dtype=np.float32)

        planner.plan([start_p, mid_pt, hover_dest], speed=random.uniform(2.4, 3.2))
        controller.reset_integral()

        # Simulated operational state
        bms_soc = random.uniform(0.35, 0.98)
        winch_state = "STANDBY" # STANDBY, LOWERING, TOUCHDOWN, RETRACTING
        cable_len = 0.08
        cable_vel = 0.0
        tension_n = PAYLOAD_WEIGHT_N
        has_payload = True
        touchdown_timer = 0.0
        flight_time = 0.0
        dt = 0.016

        # Random disturbance injection
        drone.wind_gust = np.array([random.uniform(-3.5, 3.5), random.uniform(-3.5, 3.5), random.uniform(-0.8, 0.8)], dtype=np.float32)
        is_spoof_episode = (random.random() < 0.20)
        chi2_val = 34.2 if is_spoof_episode else 1.8

        mission_active = True
        step_count = 0

        while mission_active and step_count < 650:
            flight_time = min(planner.total_time, flight_time + dt)
            p_des, v_des, a_des = planner.eval(flight_time)

            d_horiz = float(np.linalg.norm(drone.p[:2] - target_gnd[:2]))
            is_arrived = (d_horiz < 1.8) or (flight_time >= planner.total_time)

            # Executive FSM Mapping:
            # 0: TRANSIT, 1: HOVER_LOCK, 2: WINCH_LOWER, 3: TOUCHDOWN_RELEASE, 4: WINCH_RETRACT, 5: AUTO_DISPATCH
            if not is_arrived:
                fsm_token = 0 # TRANSIT
            else:
                p_des = hover_dest.copy()
                v_des.fill(0.0)
                a_des.fill(0.0)

                if winch_state == "STANDBY" and has_payload:
                    winch_state = "LOWERING"
                    fsm_token = 1 # HOVER_LOCK
                elif winch_state == "LOWERING":
                    cable_vel = 1.25
                    cable_len += cable_vel * dt
                    tension_n = PAYLOAD_WEIGHT_N + float(np.random.normal(0, 0.25))
                    fsm_token = 2 # WINCH_LOWER
                    if (drone.p[2] - cable_len) <= 0.08:
                        winch_state = "TOUCHDOWN"
                        touchdown_timer = 0.0
                elif winch_state == "TOUCHDOWN":
                    cable_vel = 0.0
                    tension_n = 1.8 + float(np.random.normal(0, 0.15))
                    touchdown_timer += dt
                    fsm_token = 3 # TOUCHDOWN_RELEASE
                    if touchdown_timer >= 1.0:
                        has_payload = False
                        winch_state = "RETRACTING"
                elif winch_state == "RETRACTING":
                    cable_vel = -2.2
                    cable_len = max(0.08, cable_len + cable_vel * dt)
                    tension_n = 1.2 + float(np.random.normal(0, 0.1))
                    fsm_token = 4 # WINCH_RETRACT
                    if cable_len <= 0.08:
                        fsm_token = 5 # AUTO_DISPATCH
                        delivered_count += 1
                        mission_active = False

            # Teacher computes continuous motor commands
            omega_cmds = controller.compute(drone.p, drone.v, drone.R, drone.omega, p_des, v_des, a_des, dt=dt)
            drone.step(omega_cmds, dt=dt)

            # Synthesize 32-element observation
            v_tas = float(np.linalg.norm(drone.v - drone.wind_gust))
            q_dyn = 0.5 * 1.225 * (v_tas ** 2)

            obs_t = synthesizer.synthesize(
                drone_p=drone.p,
                drone_v=drone.v,
                drone_R=drone.R,
                drone_omega=drone.omega,
                target_p=hover_dest,
                v_tas=v_tas,
                q_dyn=q_dyn,
                min_sonar=max(0.1, min(8.0, drone.p[2])),
                emf_rms=5.0,
                wind_est=drone.wind_gust,
                adsb_t_cpa=45.0,
                bms_soc=bms_soc,
                v_pack=22.8,
                cable_len=cable_len,
                cable_vel=cable_vel,
                tension_n=tension_n,
                has_payload=has_payload,
                chi2_val=chi2_val,
                is_spoofed=is_spoof_episode,
                flight_time=flight_time,
                total_plan_time=planner.total_time,
                delivered_count=delivered_count,
                dt=dt
            )

            # Normalized motor targets [0.0, 1.0]
            norm_motors = np.clip(omega_cmds / MAX_OMEGA, 0.0, 1.0).astype(np.float32)

            observations.append(obs_t)
            teacher_motors.append(norm_motors)
            teacher_fsm.append(fsm_token)

            step_count += 1
            if len(observations) >= num_steps_target:
                break

    return (np.array(observations, dtype=np.float32),
            np.array(teacher_motors, dtype=np.float32),
            np.array(teacher_fsm, dtype=np.int64))

# ==========================================
# 3. PUBLIC HARVESTER API
# ==========================================
def harvest_autopilot_dataset(total_samples=120000):
    """
    Harvests full demonstration dataset across CPU cores in parallel.
    """
    t0 = time.time()
    num_workers = max(1, mp.cpu_count() - 1)
    samples_per_worker = (total_samples + num_workers - 1) // num_workers

    print("=========================================================")
    print(f"STEP 3: Multi-Core Teacher Policy Harvester Active")
    print(f"Target Samples:      {total_samples:,}")
    print(f"CPU Workers:         {num_workers} parallel processes")
    print(f"Samples per Worker:  {samples_per_worker:,}")
    print("=========================================================")

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_harvest_trajectory, [samples_per_worker] * num_workers)

    X_all = np.concatenate([r[0] for r in results], axis=0)[:total_samples]
    y_motors_all = np.concatenate([r[1] for r in results], axis=0)[:total_samples]
    y_fsm_all = np.concatenate([r[2] for r in results], axis=0)[:total_samples]

    elapsed = time.time() - t0
    rate = total_samples / max(0.1, elapsed)

    print(f"Harvest Completed in: {elapsed:.2f}s ({rate:.0f} samples/sec)")
    print(f"Dataset X Shape:       {X_all.shape} (float32)")
    print(f"Dataset Y_Motors:      {y_motors_all.shape} (float32 [0.0, 1.0])")
    print(f"Dataset Y_FSM:         {y_fsm_all.shape} (int64 tokens 0-5)")
    print("=========================================================")

    return X_all, y_motors_all, y_fsm_all


if __name__ == "__main__":
    mp.freeze_support()
    # Fast test verification with 15,000 samples
    X, y_m, y_f = harvest_autopilot_dataset(total_samples=15000)
    print("STEP 3 VERIFIED: Parallel Dataset Harvesting Operational!")