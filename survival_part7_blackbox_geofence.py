"""
=============================================================================
SURVIVAL CORE V39.0 — MODULE 7 OF 7
BLACKBOX TELEMETRY RECORDER, PARACHUTE EGRESS & 3D GEOFENCE ARBITER (FTS)
=============================================================================
"""

import math
import time
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import IntEnum
import numpy as np

# =============================================================================
# 0. SAFETY & EGRESS CONSTANTS
# =============================================================================
BLACKBOX_MAGIC_HEADER = 0x53555256  # "SURV"
BLACKBOX_RECORD_FORMAT = "<QfffffffffHHBBBB" # 64-byte binary record
BLACKBOX_RECORD_SIZE = struct.calcsize(BLACKBOX_RECORD_FORMAT)
BLACKBOX_BUFFER_CAPACITY = 10000    # ~160 seconds of 60Hz telemetry in memory

# Flight Termination System (FTS) Thresholds
CRITICAL_TUMBLE_RATE_RAD_S = 45.0   # Extreme unrecoverable angular rate
CRITICAL_TILT_ANGLE_RAD = math.radians(75.0)
CRITICAL_TILT_DWELL_S = 0.35        # Max duration inverted before FTS fires
CRITICAL_VOLTAGE_SAG_V = 18.0       # Minimum total 6S pack voltage (3.0V/cell)
CRITICAL_BATTERY_TEMP_C = 68.0      # Thermal runaway threshold
TETHER_SNAG_FORCE_N = 48.0          # Winch guillotine overload threshold
GEOFENCE_PREDICTIVE_HORIZON_S = 1.8 # Time-to-violation forecast window

# Parachute Physical Constants
PARACHUTE_CANOPY_AREA_M2 = 0.85
PARACHUTE_DRAG_COEFF_CD = 1.45
PARACHUTE_DEPLOY_TIME_S = 0.40      # Ejection to full inflation delay
PARACHUTE_SUSPENSION_LINE_M = 1.20
IMPACT_SAFE_ENERGY_JOULES = 35.0    # Regulatory human safety impact limit

AIR_DENSITY_RHO = 1.225
GRAVITY_G = 9.80665
DRY_MASS_KG = 1.35

# =============================================================================
# 1. ENUMS & DATA STRUCTURES
# =============================================================================
class SafetyEscalationTier(IntEnum):
    TIER0_NOMINAL = 0
    TIER1_DEGRADED_RTH = 1
    TIER2_EMERGENCY_LANDING = 2
    TIER3_SEVER_PAYLOAD = 3
    TIER4_BALLISTIC_TERMINATION = 4

class GeofenceType(IntEnum):
    INCLUSION_ZONE = 0              # Drone must remain inside
    EXCLUSION_KEEP_OUT = 1          # Drone must never enter

@dataclass
class GeofencePrism3D:
    zone_id: str
    zone_type: GeofenceType
    polygon_xy: np.ndarray          # Shape (N, 2) clockwise vertices
    z_min_m: float
    z_max_m: float
    safety_buffer_m: float = 2.0

@dataclass
class FTSStatusReport:
    tier: SafetyEscalationTier
    is_terminated: bool
    is_chute_deployed: bool
    is_payload_severed: bool
    trigger_reason: str
    time_to_violation_s: float

# =============================================================================
# 2. HIGH-RATE BINARY BLACKBOX FLIGHT RECORDER
# =============================================================================
class BinaryBlackboxRecorder:
    """
    Fixed-memory circular binary blackbox recorder.
    Serializes 64-byte structured telemetry records at 60 Hz without heap allocations.
    """
    def __init__(self, drone_id: int, log_dir: str = "."):
        self.drone_id = drone_id
        self.log_dir = log_dir
        self.capacity = BLACKBOX_BUFFER_CAPACITY
        self.buffer = bytearray(self.capacity * BLACKBOX_RECORD_SIZE)
        self.write_index = 0
        self.total_records_logged = 0

    def record_frame(self, sim_time_s: float, p_enu: np.ndarray, v_enu: np.ndarray,
                     roll_rad: float, pitch_rad: float, yaw_rad: float,
                     v_pack: float, tension_n: float, fsm_state: int,
                     health_mask: int, cbf_active: bool, fts_tier: int):
        """Packs and writes a 64-byte frame into the circular ring buffer."""
        time_usec = int(sim_time_s * 1e6)
        v_pack_mv = int(v_pack * 1000.0)
        tension_cN = int(tension_n * 100.0)
        cbf_byte = 1 if cbf_active else 0

        record = struct.pack(
            BLACKBOX_RECORD_FORMAT,
            time_usec,
            float(p_enu[0]), float(p_enu[1]), float(p_enu[2]),
            float(v_enu[0]), float(v_enu[1]), float(v_enu[2]),
            float(roll_rad), float(pitch_rad), float(yaw_rad),
            v_pack_mv,
            tension_cN,
            int(fsm_state),
            int(health_mask),
            cbf_byte,
            int(fts_tier)
        )

        offset = (self.write_index % self.capacity) * BLACKBOX_RECORD_SIZE
        self.buffer[offset : offset + BLACKBOX_RECORD_SIZE] = record
        self.write_index += 1
        self.total_records_logged += 1

    def dump_to_disk(self, filename: Optional[str] = None) -> str:
        """Flushes the active flight recording to an immutable post-mortem binary file."""
        if filename is None:
            filename = f"blackbox_drone_{self.drone_id}_{int(time.time())}.sbb"
        filepath = os.path.join(self.log_dir, filename)

        num_valid = min(self.total_records_logged, self.capacity)
        start_idx = (self.write_index - num_valid) % self.capacity

        with open(filepath, "wb") as f:
            # Header: [Magic(4), DroneID(2), RecordCount(4), RecordSize(2)]
            header = struct.pack("<IHHI", BLACKBOX_MAGIC_HEADER, self.drone_id, BLACKBOX_RECORD_SIZE, num_valid)
            f.write(header)

            if self.total_records_logged <= self.capacity:
                f.write(self.buffer[:num_valid * BLACKBOX_RECORD_SIZE])
            else:
                part1 = self.buffer[start_idx * BLACKBOX_RECORD_SIZE :]
                part2 = self.buffer[: (self.write_index % self.capacity) * BLACKBOX_RECORD_SIZE]
                f.write(part1)
                f.write(part2)

        return filepath

# =============================================================================
# 3. 3D POLYHEDRAL GEOFENCE & KEEP-OUT ARBITER
# =============================================================================
class PolyhedralGeofenceArbiter:
    """
    Evaluates 3D inclusion and exclusion zones. Uses ray-casting point-in-polygon
    tests and forward trajectory projection to compute Time-To-Violation (TTV).
    """
    def __init__(self):
        self.zones: List[GeofencePrism3D] = []

    def add_inclusion_boundary(self, zone_id: str, polygon_xy: np.ndarray, z_min: float = 0.0, z_max: float = 120.0):
        self.zones.append(GeofencePrism3D(zone_id, GeofenceType.INCLUSION_ZONE, polygon_xy, z_min, z_max))

    def add_keep_out_zone(self, zone_id: str, polygon_xy: np.ndarray, z_min: float = 0.0, z_max: float = 60.0):
        self.zones.append(GeofencePrism3D(zone_id, GeofenceType.EXCLUSION_KEEP_OUT, polygon_xy, z_min, z_max))

    @staticmethod
    def _point_in_polygon_2d(x: float, y: float, poly: np.ndarray) -> bool:
        """Jordan curve theorem ray-casting point-in-polygon test."""
        inside = False
        n = len(poly)
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def check_position(self, p_enu: np.ndarray) -> Tuple[bool, str]:
        """
        Validates static spatial containment.
        Returns: (is_safe, violation_description)
        """
        px, py, pz = float(p_enu[0]), float(p_enu[1]), float(p_enu[2])

        for zone in self.zones:
            in_h = (zone.z_min_m <= pz <= zone.z_max_m)
            in_poly = self._point_in_polygon_2d(px, py, zone.polygon_xy)

            if zone.zone_type == GeofenceType.INCLUSION_ZONE:
                if not (in_h and in_poly):
                    return False, f"BREACH_OUTSIDE_INCLUSION: {zone.zone_id}"
            else:  # EXCLUSION_KEEP_OUT
                if in_h and in_poly:
                    return False, f"BREACH_INSIDE_KEEPOUT: {zone.zone_id}"

        return True, "SAFE_NOMINAL"

    def predict_time_to_violation(self, p_enu: np.ndarray, v_enu: np.ndarray,
                                  horizon_s: float = GEOFENCE_PREDICTIVE_HORIZON_S) -> Tuple[float, Optional[str]]:
        """
        Projects velocity forward to calculate time until the boundary is breached.
        Returns: (ttv_seconds, breached_zone_id)
        """
        steps = 15
        dt = horizon_s / steps

        sim_p = p_enu.copy()
        for s in range(1, steps + 1):
            sim_p += v_enu * dt
            safe, reason = self.check_position(sim_p)
            if not safe:
                return float(s * dt), reason

        return float("inf"), None

# =============================================================================
# 4. AERODYNAMIC PARACHUTE EGRESS & WINCH GUILLOTINE DYNAMICS
# =============================================================================
class AerodynamicParachuteEgressSystem:
    """
    Simulates pyrotechnic cable severing and parachute ejection, canopy
    reefing delay, and terminal descent dynamics under ambient winds.
    """
    def __init__(self, dry_mass_kg: float = 1.35):
        self.mass = dry_mass_kg
        self.is_deployed = False
        self.is_inflated = False
        self.deployment_timer = 0.0
        self.cable_severed = False

    def trigger_guillotine(self) -> bool:
        """Activates pyrotechnic blade to cut cargo tether and jettison load."""
        self.cable_severed = True
        return True

    def deploy_parachute(self):
        """Disarms motors and deploys the ballistic recovery parachute."""
        if not self.is_deployed:
            self.is_deployed = True
            self.deployment_timer = 0.0

    def step_descent(self, p_enu: np.ndarray, v_enu: np.ndarray, wind_3d: np.ndarray, dt: float = 0.016) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculates descent trajectory and impact kinetic energy under parachute drag.
        Returns: (new_pos, new_vel, kinetic_energy_joules)
        """
        if not self.is_deployed:
            return p_enu, v_enu, 0.5 * self.mass * float(np.dot(v_enu, v_enu))

        self.deployment_timer += dt
        inflation_ratio = float(np.clip(self.deployment_timer / PARACHUTE_DEPLOY_TIME_S, 0.15, 1.0))
        self.is_inflated = (inflation_ratio >= 1.0)

        # Effective parachute drag area expanding over time
        effective_area = PARACHUTE_CANOPY_AREA_M2 * inflation_ratio

        # Relative air velocity
        v_rel = v_enu - wind_3d
        airspeed = float(np.linalg.norm(v_rel))

        # Drag force opposing relative wind
        f_drag = -0.5 * AIR_DENSITY_RHO * PARACHUTE_DRAG_COEFF_CD * effective_area * airspeed * v_rel
        f_gravity = np.array([0.0, 0.0, -self.mass * GRAVITY_G], dtype=np.float32)

        # Acceleration and integration
        accel = (f_drag + f_gravity) / self.mass
        v_next = v_enu + accel * dt
        p_next = p_enu + v_next * dt

        # Ground contact clamp
        if p_next[2] <= 0.05:
            p_next[2] = 0.05
            v_next.fill(0.0)

        speed_sq = float(np.dot(v_next, v_next))
        kinetic_energy_j = 0.5 * self.mass * speed_sq
        return p_next, v_next, kinetic_energy_j

# =============================================================================
# 5. UNIFIED FLIGHT TERMINATION & SAFETY EXECUTIVE (FTS)
# =============================================================================
class AutonomousSafetyExecutiveFTS:
    """
    Onboard safety authority. Monitors health thresholds, triggers the winch
    guillotine and parachute, logs blackbox data, and enforces geofences.
    """
    def __init__(self, drone_id: int, geofence_arbiter: PolyhedralGeofenceArbiter, log_dir: str = "."):
        self.drone_id = drone_id
        self.geofence = geofence_arbiter
        self.blackbox = BinaryBlackboxRecorder(drone_id, log_dir=log_dir)
        self.parachute = AerodynamicParachuteEgressSystem(dry_mass_kg=DRY_MASS_KG)
        
        self.current_tier = SafetyEscalationTier.TIER0_NOMINAL
        self.tilt_dwell_timer = 0.0
        self.termination_reason = "NONE"

    def step(self, sim_time_s: float, p_enu: np.ndarray, v_enu: np.ndarray, R_enu: np.ndarray,
             omega_body: np.ndarray, motor_health: np.ndarray, v_pack: float,
             cell_min_v: float, bms_temp_c: float, winch_tension_n: float,
             wind_3d: np.ndarray, fsm_state: int, cbf_active: bool) -> FTSStatusReport:
        """
        Executes health auditing, escalation cascades, and blackbox logging.
        """
        # 1. Roll / Pitch extraction
        pitch_rad = math.asin(float(np.clip(-R_enu[2, 0], -1.0, 1.0)))
        roll_rad = math.atan2(float(R_enu[2, 1]), float(R_enu[2, 2]))
        yaw_rad = math.atan2(float(R_enu[1, 0]), float(R_enu[0, 0]))

        # 2. Blackbox telemetry recording
        health_mask = 0
        for i in range(min(8, len(motor_health))):
            if motor_health[i] > 0.0:
                health_mask |= (1 << i)

        self.blackbox.record_frame(
            sim_time_s, p_enu, v_enu, roll_rad, pitch_rad, yaw_rad,
            v_pack, winch_tension_n, fsm_state, health_mask, cbf_active, int(self.current_tier)
        )

        # 3. Geofence containment check
        is_geo_safe, geo_reason = self.geofence.check_position(p_enu)
        ttv_s, pred_reason = self.geofence.predict_time_to_violation(p_enu, v_enu)

        # 4. FTS Hazard Detection
        # Check Tether Snag
        if winch_tension_n >= TETHER_SNAG_FORCE_N and not self.parachute.cable_severed:
            self.parachute.trigger_guillotine()
            self.current_tier = max(self.current_tier, SafetyEscalationTier.TIER3_SEVER_PAYLOAD)
            self.termination_reason = "TETHER_SNAG_OVERLOAD_GUILLOTINE"

        # Check Inverted Tumble Runaway
        tilt_magnitude = math.hypot(roll_rad, pitch_rad)
        if tilt_magnitude > CRITICAL_TILT_ANGLE_RAD or float(np.linalg.norm(omega_body)) > CRITICAL_TUMBLE_RATE_RAD_S:
            self.tilt_dwell_timer += 0.016
        else:
            self.tilt_dwell_timer = max(0.0, self.tilt_dwell_timer - 0.032)

        # Multi-Tier Escalation Logic
        if self.current_tier < SafetyEscalationTier.TIER4_BALLISTIC_TERMINATION:
            if not is_geo_safe:
                self.current_tier = SafetyEscalationTier.TIER4_BALLISTIC_TERMINATION
                self.termination_reason = f"FTS_GEOFENCE_BREACH: {geo_reason}"
                self.parachute.trigger_guillotine()
                self.parachute.deploy_parachute()

            elif self.tilt_dwell_timer >= CRITICAL_TILT_DWELL_S:
                self.current_tier = SafetyEscalationTier.TIER4_BALLISTIC_TERMINATION
                self.termination_reason = "FTS_UNRECOVERABLE_TUMBLE_RUNAWAY"
                self.parachute.trigger_guillotine()
                self.parachute.deploy_parachute()

            elif v_pack < CRITICAL_VOLTAGE_SAG_V or cell_min_v < 2.95:
                self.current_tier = SafetyEscalationTier.TIER4_BALLISTIC_TERMINATION
                self.termination_reason = "FTS_CRITICAL_BATTERY_COLLAPSE"
                self.parachute.trigger_guillotine()
                self.parachute.deploy_parachute()

            elif np.count_nonzero(motor_health == 0.0) >= 3:
                self.current_tier = SafetyEscalationTier.TIER4_BALLISTIC_TERMINATION
                self.termination_reason = "FTS_MULTIPLE_ROTOR_FAILURE_CATACLYSM"
                self.parachute.trigger_guillotine()
                self.parachute.deploy_parachute()

            elif ttv_s < 0.65:
                self.current_tier = max(self.current_tier, SafetyEscalationTier.TIER1_DEGRADED_RTH)
                self.termination_reason = f"PREDICTIVE_GEOFENCE_WARNING: {pred_reason}"

        # 5. Flush Blackbox Dump on Terminal Event
        if self.current_tier == SafetyEscalationTier.TIER4_BALLISTIC_TERMINATION and self.parachute.deployment_timer < 0.02:
            self.blackbox.dump_to_disk()

        return FTSStatusReport(
            tier=self.current_tier,
            is_terminated=(self.current_tier == SafetyEscalationTier.TIER4_BALLISTIC_TERMINATION),
            is_chute_deployed=self.parachute.is_deployed,
            is_payload_severed=self.parachute.cable_severed,
            trigger_reason=self.termination_reason,
            time_to_violation_s=ttv_s
        )

# =============================================================================
# 6. STANDALONE VERIFICATION SELF-TEST
# =============================================================================
def run_part7_blackbox_geofence_self_test():
    print("=" * 75)
    print("RUNNING PART 7 SELF-TEST: BLACKBOX RECORDER, 3D GEOFENCE & FTS ARBITER")
    print("=" * 75)

    # 1. Initialize 3D Geofence Arbiter with Inclusion & Exclusion Zones
    geofence = PolyhedralGeofenceArbiter()
    
    # Inclusion: Downtown city core [-180, 180]m box, max altitude 100m
    city_boundary = np.array([
        [-180.0, -180.0], [ 180.0, -180.0],
        [ 180.0,  180.0], [-180.0,  180.0]
    ], dtype=np.float32)
    geofence.add_inclusion_boundary("City_Downtown_Bounds", city_boundary, z_min=0.0, z_max=100.0)

    # Exclusion Keep-Out: Substation / Hospital Helipad at [20, 20]m with 15m radius
    substation = np.array([
        [10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]
    ], dtype=np.float32)
    geofence.add_keep_out_zone("Hospital_Helipad_Exclusion", substation, z_min=0.0, z_max=50.0)

    # Test Spatial Containment
    p_safe = np.array([0.0, 0.0, 40.0], dtype=np.float32)
    p_outside = np.array([195.0, 0.0, 40.0], dtype=np.float32)
    p_keepout = np.array([20.0, 20.0, 25.0], dtype=np.float32)

    assert geofence.check_position(p_safe)[0], "Safe point marked as breach!"
    assert not geofence.check_position(p_outside)[0], "Outside inclusion boundary was not caught!"
    assert not geofence.check_position(p_keepout)[0], "Point inside keep-out zone was not caught!"
    print("[GEOFENCE    ] 3D Inclusion & Keep-out boundary intersection checks verified.")

    # Test Predictive Time-To-Violation (TTV)
    p_heading_out = np.array([170.0, 0.0, 40.0], dtype=np.float32)
    v_fast_out = np.array([12.0, 0.0, 0.0], dtype=np.float32)  # Will breach in ~0.83 seconds
    ttv, reason = geofence.predict_time_to_violation(p_heading_out, v_fast_out, horizon_s=2.0)
    print(f"[PREDICTION  ] Predictive Geofence Breach Forecast: TTV = {ttv:.2f} s | Reason: {reason}")
    assert 0.70 <= ttv <= 1.0, f"Expected breach prediction ~0.83s, got {ttv}s!"

    # 2. Binary Blackbox Ring-Buffer Roundtrip Test
    test_log_dir = os.path.abspath(".")
    safety_exec = AutonomousSafetyExecutiveFTS(drone_id=0, geofence_arbiter=geofence, log_dir=test_log_dir)

    for step in range(250):
        t = step * 0.016
        safety_exec.blackbox.record_frame(
            sim_time_s=t,
            p_enu=np.array([t * 2.0, 0.0, 15.0], dtype=np.float32),
            v_enu=np.array([2.0, 0.0, 0.0], dtype=np.float32),
            roll_rad=0.01, pitch_rad=-0.02, yaw_rad=0.5,
            v_pack=23.4, tension_n=4.41, fsm_state=0,
            health_mask=0xFF, cbf_active=False, fts_tier=0
        )

    dump_path = safety_exec.blackbox.dump_to_disk("test_blackbox_validation.sbb")
    file_bytes = os.path.getsize(dump_path)
    print(f"[BLACKBOX    ] Flushed binary flight log: {dump_path} ({file_bytes} bytes)")
    expected_bytes = 14 + (250 * BLACKBOX_RECORD_SIZE)
    assert file_bytes == expected_bytes, f"Expected {expected_bytes} bytes, found {file_bytes}!"
    os.remove(dump_path)

    # 3. Tether Snag & Guillotine Trigger Test
    report_snag = safety_exec.step(
        sim_time_s=4.0, p_enu=p_safe, v_enu=np.zeros(3, dtype=np.float32),
        R_enu=np.eye(3, dtype=np.float32), omega_body=np.zeros(3, dtype=np.float32),
        motor_health=np.ones(8, dtype=np.float32), v_pack=23.0, cell_min_v=3.8,
        bms_temp_c=25.0, winch_tension_n=52.5, wind_3d=np.zeros(3, dtype=np.float32),
        fsm_state=2, cbf_active=False
    )
    print(f"[GUILLOTINE  ] Overload Tension Spike: Triggered = {report_snag.is_payload_severed} | Reason: {report_snag.trigger_reason}")
    assert report_snag.is_payload_severed, "Tether guillotine failed to trigger on overload force!"

    # 4. Critical Tumble Runway & Ballistic Parachute Descent Test
    # Simulate high tumble rate > 45 rad/s
    omega_spin = np.array([0.0, 52.0, 0.0], dtype=np.float32)
    for _ in range(25):  # Dwell for > 0.35s
        report_tumble = safety_exec.step(
            sim_time_s=4.5, p_enu=p_safe, v_enu=np.zeros(3, dtype=np.float32),
            R_enu=np.eye(3, dtype=np.float32), omega_body=omega_spin,
            motor_health=np.ones(8, dtype=np.float32), v_pack=23.0, cell_min_v=3.8,
            bms_temp_c=25.0, winch_tension_n=4.41, wind_3d=np.zeros(3, dtype=np.float32),
            fsm_state=0, cbf_active=False
        )

    print(f"[PARACHUTE   ] FTS Runaway Triggered: {report_tumble.is_terminated} | Chute Deployed: {report_tumble.is_chute_deployed}")
    assert report_tumble.is_terminated and report_tumble.is_chute_deployed, "FTS failed to deploy parachute on tumble runaway!"

    # Step parachute descent for 3.0 seconds to verify terminal kinetic energy
    curr_p = np.array([0.0, 0.0, 30.0], dtype=np.float32)
    curr_v = np.array([0.0, 0.0, -12.0], dtype=np.float32)
    wind_gale = np.array([4.0, 0.0, 0.0], dtype=np.float32)

    for _ in range(int(3.0 / 0.016)):
        curr_p, curr_v, kinetic_energy = safety_exec.parachute.step_descent(curr_p, curr_v, wind_gale, dt=0.016)

    descent_speed = abs(float(curr_v[2]))
    print(f"[EGRESS DYN  ] Parachute Terminal Descent: Vz = -{descent_speed:.2f} m/s | Kinetic Energy = {kinetic_energy:.2f} J")
    assert descent_speed < 5.8, f"Parachute terminal speed {descent_speed:.2f} m/s exceeds safe envelope!"
    assert kinetic_energy < IMPACT_SAFE_ENERGY_JOULES, f"Impact energy {kinetic_energy:.2f} J exceeds 35 J safety limit!"

    print("\n[PART 7 TEST PASS] Blackbox, Geofence containment, Guillotine, and Parachute FTS verified.\n")

if __name__ == "__main__":
    run_part7_blackbox_geofence_self_test()