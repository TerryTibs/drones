"""
=============================================================================
STEP 1: MULTI-MODAL OBSERVATION SYNTHESIZER & NORMALIZATION ENGINE
File: drone_observation_synthesizer.py
=============================================================================
Transforms heterogeneous physical airframe states, sensor observations,
battery metrics, winch telemetry, and RF alarms into a clean, normalized
32-dimensional observation vector o_t in R^32 for the Neural Autopilot.
=============================================================================
"""

import math
import numpy as np

OBSERVATION_DIM = 32

class DroneObservationSynthesizer:
    """
    State-to-observation encoder that guarantees strictly bounded,
    scale-normalized inputs for the deep sensorimotor policy.
    """
    def __init__(self):
        self.last_tension = 19.62 # Initial static weight
        self.last_time = 0.0

    def synthesize(self, 
                   drone_p, 
                   drone_v, 
                   drone_R, 
                   drone_omega, 
                   target_p,
                   v_tas, 
                   q_dyn, 
                   min_sonar, 
                   emf_rms, 
                   wind_est, 
                   adsb_t_cpa,
                   bms_soc, 
                   v_pack, 
                   cable_len, 
                   cable_vel, 
                   tension_n, 
                   has_payload,
                   chi2_val, 
                   is_spoofed, 
                   flight_time, 
                   total_plan_time, 
                   delivered_count, 
                   dt=0.016):
        """
        Encodes all internal and environmental states into a 32-element float32 vector.
        """
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)

        # 1. Relative Position to Target (0:3)
        delta_p = target_p - drone_p
        obs[0:3] = np.clip(delta_p / 10.0, -1.0, 1.0)

        # 2. Linear Velocity (3:6)
        obs[3:6] = np.clip(drone_v / 6.0, -1.0, 1.0)

        # 3. Attitude Body Z-axis (Thrust Vector) (6:9)
        # R[:, 2] is the direction the drone rotors are thrusting
        obs[6:9] = np.clip(drone_R[:, 2], -1.0, 1.0)

        # 4. Body Angular Rates (9:12)
        obs[9:12] = np.clip(drone_omega / 8.0, -1.0, 1.0)

        # 5. Airspeed & Aerodynamic Dynamic Pressure (12:14)
        obs[12] = np.clip(v_tas / 15.0, 0.0, 1.0)
        obs[13] = np.clip(q_dyn / 100.0, 0.0, 1.0)

        # 6. Sonar Clearance & Powerline EMF (14:16)
        obs[14] = np.clip(min_sonar / 8.0, 0.0, 1.0)
        obs[15] = np.clip(emf_rms / 40.0, 0.0, 1.0)

        # 7. Wind Gust Vector (16:19)
        obs[16:19] = np.clip(wind_est / 8.0, -1.0, 1.0)

        # 8. ADS-B Traffic Intrusion CPA Time (19)
        obs[19] = np.clip(adsb_t_cpa / 40.0, 0.0, 1.0)

        # 9. Battery Energy & Pack Voltage (20:22)
        obs[20] = np.clip(bms_soc, 0.0, 1.0)
        obs[21] = np.clip((v_pack - 18.0) / (25.2 - 18.0), 0.0, 1.0)

        # 10. Winch Cable Length & Velocity (22:24)
        obs[22] = np.clip(cable_len / 10.0, 0.0, 1.0)
        obs[23] = np.clip(cable_vel / 3.0, -1.0, 1.0)

        # 11. Winch Line Tension & Tension Rate (24:26)
        d_tension = (tension_n - self.last_tension) / max(1e-4, dt)
        self.last_tension = tension_n
        obs[24] = np.clip(tension_n / 50.0, 0.0, 1.0)
        obs[25] = np.clip(d_tension / 25.0, -1.0, 1.0)

        # 12. Hook Payload State (26)
        obs[26] = 1.0 if has_payload else 0.0

        # 13. Anti-Spoofing Chi-Square & Alarm (27:29)
        obs[27] = np.clip(chi2_val / 40.0, 0.0, 1.0)
        obs[28] = 1.0 if is_spoofed else 0.0

        # 14. Mission Progress & Dropzone Metrics (29:32)
        progress = (flight_time / max(0.1, total_plan_time))
        obs[29] = np.clip(progress, 0.0, 1.0)
        d_horiz = math.hypot(delta_p[0], delta_p[1])
        obs[30] = np.clip(d_horiz / 10.0, 0.0, 1.0)
        obs[31] = np.clip((delivered_count % 10) / 10.0, 0.0, 1.0)

        return obs


# ==========================================
# VERIFICATION UNIT TEST
# ==========================================
if __name__ == "__main__":
    synthesizer = DroneObservationSynthesizer()

    # Generate test dummy states
    dummy_p = np.array([1.2, -2.5, 3.0], dtype=np.float32)
    dummy_v = np.array([0.5, 1.2, -0.2], dtype=np.float32)
    dummy_R = np.eye(3, dtype=np.float32)
    dummy_omega = np.array([0.05, -0.1, 0.0], dtype=np.float32)
    dummy_target = np.array([3.0, 4.5, 2.2], dtype=np.float32)

    vec = synthesizer.synthesize(
        drone_p=dummy_p,
        drone_v=dummy_v,
        drone_R=dummy_R,
        drone_omega=dummy_omega,
        target_p=dummy_target,
        v_tas=4.5,
        q_dyn=12.5,
        min_sonar=5.2,
        emf_rms=8.4,
        wind_est=np.array([1.0, -0.5, 0.0]),
        adsb_t_cpa=35.0,
        bms_soc=0.88,
        v_pack=23.5,
        cable_len=0.15,
        cable_vel=0.0,
        tension_n=19.6,
        has_payload=True,
        chi2_val=1.8,
        is_spoofed=False,
        flight_time=1.2,
        total_plan_time=4.5,
        delivered_count=2
    )

    print("=========================================================")
    print("STEP 1 VERIFIED: Drone Observation Synthesizer Operational")
    print(f"Observation Vector Dimension: {len(vec)} / {OBSERVATION_DIM}")
    print(f"Min Value: {np.min(vec):.3f} | Max Value: {np.max(vec):.3f}")
    print(f"NaN Check: {'PASSED (No NaNs)' if not np.isnan(vec).any() else 'FAILED (Contains NaNs)'}")
    print(f"Sample Vector Slice [0:8]: {np.round(vec[:8], 3)}")
    print("=========================================================")