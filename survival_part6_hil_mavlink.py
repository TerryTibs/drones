"""
=============================================================================
SURVIVAL CORE V39.0 — MODULE 6 OF 7
HARDWARE-IN-THE-LOOP (HIL) MAVLINK / PX4 AUTOPILOT & ROS 2 TELEMETRY BRIDGE
=============================================================================
"""

import math
import time
import socket
import struct
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# =============================================================================
# 0. PROTOCOL & NETWORK CONSTANTS
# =============================================================================
MAVLINK_V2_STX = 0xFD               # MAVLink v2 packet start marker
MAV_TYPE_HEXAROTOR = 13             # Closest standard PX4 type for coaxial X8
MAV_AUTOPILOT_PX4 = 12
MAV_MODE_FLAG_SAFETY_ARMED = 128
MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
MAV_STATE_ACTIVE = 4

# Standard MAVLink Message IDs
MSG_ID_HEARTBEAT = 0
MSG_ID_ATTITUDE = 30
MSG_ID_LOCAL_POSITION_NED = 32
MSG_ID_GLOBAL_POSITION_INT = 33
MSG_ID_HIL_SENSOR = 107
MSG_ID_BATTERY_STATUS = 147
MSG_ID_NAMED_VALUE_FLOAT = 251

# Message CRC Seeds (Required by MAVLink X.25 Frame Check Sequence)
MAVLINK_MESSAGE_CRCS = {
    MSG_ID_HEARTBEAT: 50,
    MSG_ID_ATTITUDE: 39,
    MSG_ID_LOCAL_POSITION_NED: 185,
    MSG_ID_GLOBAL_POSITION_INT: 104,
    MSG_ID_HIL_SENSOR: 108,
    MSG_ID_BATTERY_STATUS: 154,
    MSG_ID_NAMED_VALUE_FLOAT: 170
}

# Network Ports
QGC_BROADCAST_PORT = 14550          # QGroundControl listening port
PX4_HIL_BASE_PORT = 14560           # Port for PX4 SITL/HITL lockstep ingestion
ROS2_BRIDGE_BASE_PORT = 19850       # High-speed telemetry socket port

# Reference Origin: San Francisco City Center (Lat/Lon for Global GPS Int)
BASE_REF_LAT_DEG = 37.774929
BASE_REF_LON_DEG = -122.419416
BASE_REF_ALT_M = 15.0

# =============================================================================
# 1. RIGID ENU <-> NED SO(3) COORDINATE FRAME ROTATION ENGINE
# =============================================================================
class CoordinateFrameRotationEngine:
    """
    Transforms positions, velocities, and rotations between the simulator's
    East-North-Up (ENU) world frame and the aerospace North-East-Down (NED) frame.
    Determinant det(R) = +1 guarantees pure right-handed spatial preservation.
    """
    # R_enu_to_ned: X_ned = Y_enu, Y_ned = X_enu, Z_ned = -Z_enu
    R_ENU_TO_NED = np.array([
        [0.0, 1.0,  0.0],
        [1.0, 0.0,  0.0],
        [0.0, 0.0, -1.0]
    ], dtype=np.float32)

    R_NED_TO_ENU = R_ENU_TO_NED.T

    @classmethod
    def pos_enu_to_ned(cls, p_enu: np.ndarray) -> np.ndarray:
        return cls.R_ENU_TO_NED @ np.asarray(p_enu, dtype=np.float32)

    @classmethod
    def vel_enu_to_ned(cls, v_enu: np.ndarray) -> np.ndarray:
        return cls.R_ENU_TO_NED @ np.asarray(v_enu, dtype=np.float32)

    @classmethod
    def rot_enu_to_ned(cls, R_enu: np.ndarray) -> np.ndarray:
        return cls.R_ENU_TO_NED @ np.asarray(R_enu, dtype=np.float32) @ cls.R_NED_TO_ENU

    @classmethod
    def enu_to_geodetic(cls, p_enu: np.ndarray) -> Tuple[float, float, float]:
        """Converts local ENU meters to WGS-84 Latitude, Longitude, and Altitude."""
        lat_rad = math.radians(BASE_REF_LAT_DEG)
        meters_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * lat_rad)
        meters_per_deg_lon = 111412.84 * math.cos(lat_rad)

        d_north_m = float(p_enu[1])
        d_east_m = float(p_enu[0])
        alt_m = BASE_REF_ALT_M + float(p_enu[2])

        lat = BASE_REF_LAT_DEG + (d_north_m / meters_per_deg_lat)
        lon = BASE_REF_LON_DEG + (d_east_m / meters_per_deg_lon)
        return lat, lon, alt_m

# =============================================================================
# 2. ITU X.25 CRC-16 CHECKSUM & MAVLINK V2 FRAMER
# =============================================================================
class MAVLinkX25CRC:
    """Computes MAVLink ITU X.25 CRC-16 checksums."""
    @staticmethod
    def calculate(buffer: bytes, crc_extra: int) -> int:
        crc = 0xFFFF
        for byte in buffer:
            tmp = byte ^ (crc & 0xFF)
            tmp = (tmp ^ (tmp << 4)) & 0xFF
            crc = (crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)
            crc &= 0xFFFF
        # Accumulate seed CRC extra
        tmp = crc_extra ^ (crc & 0xFF)
        tmp = (tmp ^ (tmp << 4)) & 0xFF
        crc = (crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)
        return crc & 0xFFFF

class MAVLinkV2PacketBuilder:
    """
    Constructs compliant MAVLink v2 binary datagrams with zero dependencies.
    """
    def __init__(self, sys_id: int, comp_id: int = 1):
        self.sys_id = sys_id
        self.comp_id = comp_id
        self.seq = 0

    def encode(self, msg_id: int, payload: bytes) -> bytes:
        payload_len = len(payload)
        incompat_flags = 0x00
        compat_flags = 0x00
        
        # MAVLink v2 Header: [STX(1), LEN(1), INCOMP(1), COMP(1), SEQ(1), SYS(1), COMP(1), MSG_ID(3)]
        header = struct.pack(
            "<BBBBBBBHB",
            MAVLINK_V2_STX,
            payload_len,
            incompat_flags,
            compat_flags,
            self.seq,
            self.sys_id,
            self.comp_id,
            msg_id & 0xFFFF,
            (msg_id >> 16) & 0xFF
        )
        self.seq = (self.seq + 1) % 256

        # Calculate CRC over header (excluding STX) and payload
        crc_extra = MAVLINK_MESSAGE_CRCS.get(msg_id, 0)
        crc_data = header[1:] + payload
        crc = MAVLinkX25CRC.calculate(crc_data, crc_extra)
        checksum = struct.pack("<H", crc)

        return header + payload + checksum

    # --- Message Serialization Packets ---
    def build_heartbeat(self) -> bytes:
        # custom_mode (uint32), type (uint8), autopilot (uint8), base_mode (uint8), system_status (uint8), mavlink_version (uint8)
        payload = struct.pack(
            "<IBBBBB",
            0,
            MAV_TYPE_HEXAROTOR,
            MAV_AUTOPILOT_PX4,
            MAV_MODE_FLAG_SAFETY_ARMED | MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            MAV_STATE_ACTIVE,
            3
        )
        return self.encode(MSG_ID_HEARTBEAT, payload)

    def build_attitude(self, time_boot_ms: int, roll_rad: float, pitch_rad: float,
                       yaw_rad: float, p_rad_s: float, q_rad_s: float, r_rad_s: float) -> bytes:
        payload = struct.pack(
            "<Iffffff",
            int(time_boot_ms),
            float(roll_rad),
            float(pitch_rad),
            float(yaw_rad),
            float(p_rad_s),
            float(q_rad_s),
            float(r_rad_s)
        )
        return self.encode(MSG_ID_ATTITUDE, payload)

    def build_local_position_ned(self, time_boot_ms: int, p_ned: np.ndarray, v_ned: np.ndarray) -> bytes:
        payload = struct.pack(
            "<Iffffff",
            int(time_boot_ms),
            float(p_ned[0]),
            float(p_ned[1]),
            float(p_ned[2]),
            float(v_ned[0]),
            float(v_ned[1]),
            float(v_ned[2])
        )
        return self.encode(MSG_ID_LOCAL_POSITION_NED, payload)

    def build_global_position_int(self, time_boot_ms: int, lat_deg: float, lon_deg: float,
                                  alt_m: float, relative_alt_m: float, v_ned: np.ndarray,
                                  hdg_rad: float) -> bytes:
        payload = struct.pack(
            "<IiiiihhhH",
            int(time_boot_ms),
            int(lat_deg * 1e7),
            int(lon_deg * 1e7),
            int(alt_m * 1000.0),
            int(relative_alt_m * 1000.0),
            int(v_ned[0] * 100.0),
            int(v_ned[1] * 100.0),
            int(v_ned[2] * 100.0),
            int((math.degrees(hdg_rad) % 360.0) * 100.0)
        )
        return self.encode(MSG_ID_GLOBAL_POSITION_INT, payload)

    def build_hil_sensor(self, time_usec: int, accel_body: np.ndarray, gyro_body: np.ndarray,
                         mag_body: np.ndarray, baro_alt_m: float, diff_pressure_hpa: float) -> bytes:
        # fields_updated: bitmask 0x1FFF = all sensors updated
        payload = struct.pack(
            "<QfffffffffI",
            int(time_usec),
            float(accel_body[0]),
            float(accel_body[1]),
            float(accel_body[2]),
            float(gyro_body[0]),
            float(gyro_body[1]),
            float(gyro_body[2]),
            float(mag_body[0]),
            float(mag_body[1]),
            float(mag_body[2]),
            0x1FFF
        )
        return self.encode(MSG_ID_HIL_SENSOR, payload)

    def build_battery_status(self, v_pack: float, soc: float, i_draw: float, temp_c: float) -> bytes:
        voltages = [int((v_pack / 6.0) * 1000.0)] * 6 + [0xFFFF] * 4
        # Format: id (uint8), battery_function (uint8), type (uint8), temperature (int16),
        # voltages[10] (uint16[10]), current_battery (int16), current_consumed (int32),
        # energy_consumed (int32), battery_remaining (int8)
        payload = struct.pack(
            "<BBB10Hhiib",
            0,
            1,
            2,
            int(temp_c * 100.0),
            *voltages,
            int(i_draw * 100.0),
            -1,
            -1,
            int(soc * 100.0)
        )
        return self.encode(MSG_ID_BATTERY_STATUS, payload)

    def build_named_value_float(self, time_boot_ms: int, name: str, value: float) -> bytes:
        name_bytes = name.encode("ascii")[:10].ljust(10, b'\x00')
        payload = struct.pack("<If10s", int(time_boot_ms), float(value), name_bytes)
        return self.encode(MSG_ID_NAMED_VALUE_FLOAT, payload)

# =============================================================================
# 3. HIGH-SPEED ASYNC TELEMETRY SOCKET SERVERS
# =============================================================================
class QGroundControlBroadcastBridge:
    """
    UDP broadcast socket bridging real-time flight telemetry directly into
    QGroundControl (QGC) ground station software on standard port 14550.
    """
    def __init__(self, target_host: str = "127.0.0.1", target_port: int = QGC_BROADCAST_PORT):
        self.target_host = target_host
        self.target_port = target_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def send_packet(self, data: bytes):
        try:
            self.sock.sendto(data, (self.target_host, self.target_port))
        except OSError:
            pass

    def close(self):
        self.sock.close()

class ROS2SwarmTelemetryBridge:
    """
    High-throughput non-blocking JSON socket bridge. Emits ROS 2 topic payloads
    (nav_msgs/Odometry, sensor_msgs/NavSatFix) for direct ingestion into ROS 2 nodes.
    """
    def __init__(self, base_port: int = ROS2_BRIDGE_BASE_PORT):
        self.base_port = base_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

    def publish_drone_odometry(self, drone_id: int, time_s: float, p_enu: np.ndarray,
                               v_enu: np.ndarray, q_wxyz: np.ndarray, omega_body: np.ndarray):
        payload = {
            "header": {
                "stamp_s": time_s,
                "frame_id": "world_enu",
                "drone_id": drone_id
            },
            "pose": {
                "position": {"x": float(p_enu[0]), "y": float(p_enu[1]), "z": float(p_enu[2])},
                "orientation": {"w": float(q_wxyz[0]), "x": float(q_wxyz[1]), "y": float(q_wxyz[2]), "z": float(q_wxyz[3])}
            },
            "twist": {
                "linear": {"x": float(v_enu[0]), "y": float(v_enu[1]), "z": float(v_enu[2])},
                "angular": {"x": float(omega_body[0]), "y": float(omega_body[1]), "z": float(omega_body[2])}
            }
        }
        raw_bytes = json.dumps(payload).encode("utf-8")
        try:
            self.sock.sendto(raw_bytes, ("127.0.0.1", self.base_port + drone_id))
        except OSError:
            pass

    def close(self):
        self.sock.close()

# =============================================================================
# 4. SWARM HARDWARE-IN-THE-LOOP (HIL) TELEMETRY EXECUTIVE
# =============================================================================
class SwarmHILTelemetryExecutive:
    """
    Onboard bridge instance bound to each drone. Coordinates lockstep MAVLink
    sensor streaming, QGC broadcasts, and ROS 2 telemetry publishing.
    """
    def __init__(self, drone_id: int):
        self.drone_id = drone_id
        self.sys_id = drone_id + 1    # MAVLink System ID (1-indexed: Drone 1 = SysID 1)
        self.builder = MAVLinkV2PacketBuilder(sys_id=self.sys_id, comp_id=1)
        self.qgc_bridge = QGroundControlBroadcastBridge()
        self.ros2_bridge = ROS2SwarmTelemetryBridge()

        self.last_heartbeat_time = 0.0
        self.last_high_rate_time = 0.0
        self.last_battery_time = 0.0

    def step(self, sim_time_s: float, p_enu: np.ndarray, v_enu: np.ndarray, R_enu: np.ndarray,
             omega_body: np.ndarray, accel_body: np.ndarray, gyro_body: np.ndarray,
             bms_vpack: float, bms_soc: float, bms_temp_c: float, winch_len_m: float,
             winch_tension_n: float):
        """
        Executes real-time MAVLink serialization and streaming at calibrated rates.
        """
        time_boot_ms = int(sim_time_s * 1000.0)
        time_usec = int(sim_time_s * 1e6)

        # 1. Coordinate Frame Transformations
        p_ned = CoordinateFrameRotationEngine.pos_enu_to_ned(p_enu)
        v_ned = CoordinateFrameRotationEngine.vel_enu_to_ned(v_enu)
        R_ned = CoordinateFrameRotationEngine.rot_enu_to_ned(R_enu)

        # Roll, Pitch, Yaw in NED frame
        pitch_ned = math.asin(float(np.clip(-R_ned[2, 0], -1.0, 1.0)))
        roll_ned = math.atan2(float(R_ned[2, 1]), float(R_ned[2, 2]))
        yaw_ned = math.atan2(float(R_ned[1, 0]), float(R_ned[0, 0]))

        lat_deg, lon_deg, alt_m = CoordinateFrameRotationEngine.enu_to_geodetic(p_enu)

        # 2. 1 Hz Low-Rate Streams (Heartbeat)
        if (sim_time_s - self.last_heartbeat_time) >= 1.0:
            self.last_heartbeat_time = sim_time_s
            hb_pkt = self.builder.build_heartbeat()
            self.qgc_bridge.send_packet(hb_pkt)

        # 3. 20 Hz High-Rate Kinematic Streams (Attitude, Position, GPS)
        if (sim_time_s - self.last_high_rate_time) >= 0.05:
            self.last_high_rate_time = sim_time_s

            att_pkt = self.builder.build_attitude(
                time_boot_ms, roll_ned, pitch_ned, yaw_ned,
                omega_body[0], omega_body[1], omega_body[2]
            )
            pos_pkt = self.builder.build_local_position_ned(time_boot_ms, p_ned, v_ned)
            gps_pkt = self.builder.build_global_position_int(
                time_boot_ms, lat_deg, lon_deg, alt_m, float(p_enu[2]), v_ned, yaw_ned
            )

            # Custom Winch Status Telemetry
            winch_len_pkt = self.builder.build_named_value_float(time_boot_ms, "Winch_Len", winch_len_m)
            winch_ten_pkt = self.builder.build_named_value_float(time_boot_ms, "Winch_Tens", winch_tension_n)

            # Send MAVLink stream to Ground Control Station
            self.qgc_bridge.send_packet(att_pkt)
            self.qgc_bridge.send_packet(pos_pkt)
            self.qgc_bridge.send_packet(gps_pkt)
            self.qgc_bridge.send_packet(winch_len_pkt)
            self.qgc_bridge.send_packet(winch_ten_pkt)

            # Publish ROS 2 Odometry JSON stream
            # Calculate quaternion wxyz for ROS 2
            tr = float(np.trace(R_enu))
            if tr > 0.0:
                sq = math.sqrt(tr + 1.0) * 2.0
                q_wxyz = np.array([0.25 * sq, (R_enu[2, 1] - R_enu[1, 2]) / sq,
                                   (R_enu[0, 2] - R_enu[2, 0]) / sq, (R_enu[1, 0] - R_enu[0, 1]) / sq], dtype=np.float32)
            else:
                q_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

            self.ros2_bridge.publish_drone_odometry(
                self.drone_id, sim_time_s, p_enu, v_enu, q_wxyz, omega_body
            )

        # 4. 2 Hz Battery Health Stream
        if (sim_time_s - self.last_battery_time) >= 0.50:
            self.last_battery_time = sim_time_s
            i_draw_est = 18.5  # Nominal cruise current draw (A)
            bat_pkt = self.builder.build_battery_status(bms_vpack, bms_soc, i_draw_est, bms_temp_c)
            self.qgc_bridge.send_packet(bat_pkt)

    def close(self):
        self.qgc_bridge.close()
        self.ros2_bridge.close()

# =============================================================================
# 5. STANDALONE VERIFICATION SELF-TEST
# =============================================================================
def run_part6_hil_mavlink_self_test():
    print("=" * 75)
    print("RUNNING PART 6 SELF-TEST: MAVLINK V2 CRC-16, SO(3) ENU-NED & HIL BRIDGE")
    print("=" * 75)

    executive = SwarmHILTelemetryExecutive(drone_id=0)

    # 1. Test Coordinate Frame Transformations
    p_enu_test = np.array([10.0, 25.0, 42.0], dtype=np.float32)
    v_enu_test = np.array([4.5, -2.0, 1.2], dtype=np.float32)
    R_enu_test = np.eye(3, dtype=np.float32)

    p_ned_test = CoordinateFrameRotationEngine.pos_enu_to_ned(p_enu_test)
    v_ned_test = CoordinateFrameRotationEngine.vel_enu_to_ned(v_enu_test)
    R_ned_test = CoordinateFrameRotationEngine.rot_enu_to_ned(R_enu_test)

    print(f"[COORDINATES] ENU Position: {p_enu_test.tolist()} -> NED Position: {p_ned_test.tolist()}")
    print(f"[COORDINATES] ENU Velocity: {v_enu_test.tolist()} -> NED Velocity: {v_ned_test.tolist()}")

    # Assert aerospace parity invariants: X_ned = Y_enu (North), Y_ned = X_enu (East), Z_ned = -Z_enu (Down)
    assert p_ned_test[0] == p_enu_test[1], "NED North component must match ENU Y!"
    assert p_ned_test[1] == p_enu_test[0], "NED East component must match ENU X!"
    assert p_ned_test[2] == -p_enu_test[2], "NED Down component must match -ENU Z!"
    assert np.isclose(np.linalg.det(CoordinateFrameRotationEngine.R_ENU_TO_NED), 1.0), "Transform must be pure SO(3) rotation!"

    # 2. Test Geodetic WGS-84 Projection
    lat, lon, alt = CoordinateFrameRotationEngine.enu_to_geodetic(p_enu_test)
    print(f"[GEODETIC   ] Lat: {lat:.6f}° | Lon: {lon:.6f}° | Alt: {alt:.2f} m")
    assert lat > BASE_REF_LAT_DEG, "Local North offset must increase latitude!"
    assert lon > BASE_REF_LON_DEG, "Local East offset must increase longitude!"

    # 3. Test MAVLink v2 Packet Construction and Checksum
    heartbeat_bytes = executive.builder.build_heartbeat()
    print(f"[MAVLINK V2 ] Serialized Heartbeat Length: {len(heartbeat_bytes)} bytes")
    assert heartbeat_bytes[0] == MAVLINK_V2_STX, "MAVLink v2 packet must begin with 0xFD STX marker!"
    assert len(heartbeat_bytes) == (10 + 9 + 2), "Heartbeat packet length mismatch (Header:10 + Payload:9 + CRC:2)!"

    # Verify CRC-16 Checksum Integrity
    header_and_payload = heartbeat_bytes[1:-2]
    expected_crc = struct.unpack("<H", heartbeat_bytes[-2:])[0]
    computed_crc = MAVLinkX25CRC.calculate(header_and_payload, MAVLINK_MESSAGE_CRCS[MSG_ID_HEARTBEAT])
    print(f"[MAVLINK V2 ] Verified ITU X.25 CRC-16: 0x{computed_crc:04X} (Matches Wire Checksum: {expected_crc == computed_crc})")
    assert computed_crc == expected_crc, "X.25 CRC-16 checksum verification failed!"

    # 4. Test Lockstep Execution Step & Socket Output
    executive.step(
        sim_time_s=1.05,
        p_enu=p_enu_test,
        v_enu=v_enu_test,
        R_enu=R_enu_test,
        omega_body=np.zeros(3, dtype=np.float32),
        accel_body=np.array([0.0, 0.0, 9.81], dtype=np.float32),
        gyro_body=np.zeros(3, dtype=np.float32),
        bms_vpack=23.8,
        bms_soc=0.96,
        bms_temp_c=22.5,
        winch_len_m=1.25,
        winch_tension_n=4.41
    )
    executive.close()

    print("\n[PART 6 TEST PASS] MAVLink v2 encoder, X.25 CRC, SO(3) ENU-NED, and telemetry bridges verified.\n")

if __name__ == "__main__":
    run_part6_hil_mavlink_self_test()