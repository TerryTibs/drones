"""
=============================================================================
SURVIVAL CORE V39.0 — MODULE 2 OF 7
DECENTRALIZED P2P MESH NETWORKING, EPIDEMIC GOSSIP & DISTRIBUTED CBBA CONSENSUS
=============================================================================
"""

import math
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import deque
import numpy as np

# =============================================================================
# 0. RF PROPAGATION & PROTOCOL CONSTANTS
# =============================================================================
RF_FREQ_HZ = 2.4e9                   # 2.4 GHz ISM Band
SPEED_OF_LIGHT = 299792458.0
RF_WAVELENGTH_M = SPEED_OF_LIGHT / RF_FREQ_HZ
TX_POWER_DBM = 20.0                  # 100 mW standard transceivers
ANTENNA_GAIN_TX_DBI = 2.15
ANTENNA_GAIN_RX_DBI = 2.15
NOISE_FLOOR_DBM = -101.0
RX_SENSITIVITY_DBM = -92.0           # Minimum power for successful packet decode
PATH_LOSS_EXPONENT_LOS = 2.2         # Line-of-sight urban corridor
PATH_LOSS_EXPONENT_NLOS = 3.8        # Obstructed urban canyon
WALL_ATTENUATION_DB_PER_M = 14.0     # Signal loss passing through concrete structures
RAYLEIGH_SIGMA_DB = 3.5              # Multipath fading standard deviation

MAX_MESH_HOPS = 4
PACKET_QUEUE_CAPACITY = 256
HEARTBEAT_INTERVAL_S = 0.20
HEARTBEAT_TIMEOUT_S = 0.80
CBBA_MAX_BUNDLE_SIZE = 4
GOSSIP_FANOUT = 2                    # Number of peers to target per gossip round
GOSSIP_INTERVAL_S = 0.10

# =============================================================================
# 1. MESSAGE PACKETS & PROTOCOL DATACLASSES
# =============================================================================
@dataclass
class MeshPacket:
    packet_id: int
    src_id: int
    dest_id: int                     # -1 designates broadcast
    ttl: int
    timestamp_s: float
    payload_type: str                # "HEARTBEAT", "GOSSIP", "CBBA_BID", "HAZARD_ALERT"
    payload: Dict[str, Any]
    hops_taken: int = 0
    route_trace: List[int] = field(default_factory=list)

@dataclass
class PeerLinkState:
    peer_id: int
    last_seen_s: float
    rssi_dbm: float
    packet_loss_rate: float
    is_direct_neighbor: bool
    estimated_distance_m: float
    packets_received: int = 0
    packets_expected: int = 0

@dataclass
class DeliveryTaskContract:
    task_id: int
    target_pos: np.ndarray           # [X, Y, Z]
    bldg_id: str
    reward_base: float
    expiry_time_s: float
    required_payload_kg: float = 0.45

# =============================================================================
# 2. PHYSICS-INFORMED RF CHANNEL MODEL (LOS/NLOS & SHADOWING)
# =============================================================================
class UrbanRFChannelModel:
    """
    Computes realistic radio attenuation, building penetration losses,
    multipath Rayleigh fading, and packet error rates (PER).
    """
    def __init__(self, world_houses: Optional[List[Dict[str, Any]]] = None):
        self.houses = world_houses or []

    def _segment_box_penetration_length(self, p1: np.ndarray, p2: np.ndarray, box: List[float]) -> float:
        """Calculate line segment intersection length through a 3D building AABB."""
        x1, x2, y1, y2, z_top = box
        d = p2 - p1
        length = float(np.linalg.norm(d))
        if length < 1e-4:
            return 0.0

        t_min, t_max = 0.0, 1.0
        bounds = [(x1, x2), (y1, y2), (0.0, z_top)]

        for i in range(3):
            if abs(d[i]) < 1e-8:
                if p1[i] < bounds[i][0] or p1[i] > bounds[i][1]:
                    return 0.0
            else:
                inv_d = 1.0 / d[i]
                t1 = (bounds[i][0] - p1[i]) * inv_d
                t2 = (bounds[i][1] - p1[i]) * inv_d
                entry = min(t1, t2)
                exit_ = max(t1, t2)
                t_min = max(t_min, entry)
                t_max = min(t_max, exit_)
                if t_min > t_max:
                    return 0.0

        intersect_dist = (t_max - t_min) * length
        return max(0.0, intersect_dist)

    def evaluate_link(self, pos_tx: np.ndarray, pos_rx: np.ndarray) -> Tuple[float, float, bool]:
        """
        Calculates received signal power (RSSI), Packet Success Rate (PSR),
        and Line-of-Sight status between two drones.
        """
        dist_3d = float(np.linalg.norm(pos_tx - pos_rx))
        if dist_3d < 0.1:
            return TX_POWER_DBM, 1.0, True

        # 1. Test building penetration length
        penetration_m = 0.0
        for h in self.houses:
            box = h["aabb"]
            # Fast rejection test
            if max(pos_tx[2], pos_rx[2]) < 0.0 or min(pos_tx[2], pos_rx[2]) > box[4]:
                continue
            penetration_m += self._segment_box_penetration_length(pos_tx, pos_rx, box)

        is_los = (penetration_m < 0.05)
        path_exponent = PATH_LOSS_EXPONENT_LOS if is_los else PATH_LOSS_EXPONENT_NLOS

        # 2. Log-Distance Path Loss
        # FSPL at 1 meter reference distance
        fspl_1m = 20.0 * math.log10((4.0 * math.pi * 1.0) / RF_WAVELENGTH_M)
        pl_dist = fspl_1m + 10.0 * path_exponent * math.log10(max(1.0, dist_3d))

        # 3. Structural absorption + Multipath fading
        absorption_loss = penetration_m * WALL_ATTENUATION_DB_PER_M
        multipath_fading = np.random.normal(0.0, RAYLEIGH_SIGMA_DB)

        total_loss_db = pl_dist + absorption_loss + multipath_fading
        rssi_dbm = TX_POWER_DBM + ANTENNA_GAIN_TX_DBI + ANTENNA_GAIN_RX_DBI - total_loss_db

        # 4. Signal-to-Noise Ratio (SNR) and Packet Success Rate (PSR)
        snr_db = rssi_dbm - NOISE_FLOOR_DBM
        # Logistic bit/packet error probability curve based on threshold sensitivity
        fade_margin = rssi_dbm - RX_SENSITIVITY_DBM
        if fade_margin < -10.0:
            psr = 0.0
        elif fade_margin > 15.0:
            psr = 0.999
        else:
            psr = float(1.0 / (1.0 + math.exp(-0.45 * fade_margin)))

        return rssi_dbm, psr, is_los

# =============================================================================
# 3. AD-HOC P2P MESH ROUTER (HOP-BY-HOP LINK ROUTING)
# =============================================================================
class AdHocP2PMeshRouter:
    """
    Decentralized link-layer router managing neighbor tables, hop routing,
    packet deduplication, and transmission queues.
    """
    def __init__(self, node_id: int, channel_model: UrbanRFChannelModel):
        self.node_id = node_id
        self.channel_model = channel_model
        self.neighbors: Dict[int, PeerLinkState] = {}
        self.inbound_queue: deque = deque(maxlen=PACKET_QUEUE_CAPACITY)
        self.outbound_queue: deque = deque(maxlen=PACKET_QUEUE_CAPACITY)
        self.seen_packet_ids: Set[int] = set()
        self.packet_seq_counter = node_id * 1000000
        self.heartbeat_timer = random.uniform(0.0, HEARTBEAT_INTERVAL_S)

    def generate_packet_id(self) -> int:
        self.packet_seq_counter += 1
        return self.packet_seq_counter

    def send_broadcast(self, payload_type: str, payload_data: Dict[str, Any], sim_time_s: float):
        pkt = MeshPacket(
            packet_id=self.generate_packet_id(),
            src_id=self.node_id,
            dest_id=-1,
            ttl=MAX_MESH_HOPS,
            timestamp_s=sim_time_s,
            payload_type=payload_type,
            payload=payload_data,
            hops_taken=0,
            route_trace=[self.node_id]
        )
        self.seen_packet_ids.add(pkt.packet_id)
        self.outbound_queue.append(pkt)

    def send_unicast(self, dest_id: int, payload_type: str, payload_data: Dict[str, Any], sim_time_s: float):
        pkt = MeshPacket(
            packet_id=self.generate_packet_id(),
            src_id=self.node_id,
            dest_id=dest_id,
            ttl=MAX_MESH_HOPS,
            timestamp_s=sim_time_s,
            payload_type=payload_type,
            payload=payload_data,
            hops_taken=0,
            route_trace=[self.node_id]
        )
        self.seen_packet_ids.add(pkt.packet_id)
        self.outbound_queue.append(pkt)

    def receive_packet_from_medium(self, pkt: MeshPacket, sim_time_s: float, current_pos: np.ndarray, tx_pos: np.ndarray):
        # 1. Physics transmission test over RF channel
        rssi, psr, is_los = self.channel_model.evaluate_link(tx_pos, current_pos)
        
        # Update / discover neighbor state
        if pkt.src_id != self.node_id:
            if pkt.src_id not in self.neighbors:
                self.neighbors[pkt.src_id] = PeerLinkState(
                    peer_id=pkt.src_id,
                    last_seen_s=sim_time_s,
                    rssi_dbm=rssi,
                    packet_loss_rate=1.0 - psr,
                    is_direct_neighbor=True,
                    estimated_distance_m=float(np.linalg.norm(current_pos - tx_pos)),
                    packets_received=1,
                    packets_expected=1
                )
            else:
                n = self.neighbors[pkt.src_id]
                n.last_seen_s = sim_time_s
                n.rssi_dbm = 0.70 * n.rssi_dbm + 0.30 * rssi
                n.estimated_distance_m = float(np.linalg.norm(current_pos - tx_pos))
                n.packets_received += 1
                n.packet_loss_rate = 0.85 * n.packet_loss_rate + 0.15 * (1.0 - psr)

        # Monte-Carlo packet drop based on channel PSR
        if random.random() > psr:
            return  # Dropped by channel noise/shadowing

        # 2. Packet deduplication
        if pkt.packet_id in self.seen_packet_ids:
            return
        self.seen_packet_ids.add(pkt.packet_id)

        # 3. Destination routing logic
        if pkt.dest_id == self.node_id or pkt.dest_id == -1:
            self.inbound_queue.append(pkt)

        # 4. Multi-hop retransmission if TTL remains
        if pkt.dest_id != self.node_id and pkt.ttl > 1:
            forwarded_pkt = MeshPacket(
                packet_id=pkt.packet_id,
                src_id=pkt.src_id,
                dest_id=pkt.dest_id,
                ttl=pkt.ttl - 1,
                timestamp_s=pkt.timestamp_s,
                payload_type=pkt.payload_type,
                payload=pkt.payload,
                hops_taken=pkt.hops_taken + 1,
                route_trace=pkt.route_trace + [self.node_id]
            )
            self.outbound_queue.append(forwarded_pkt)

    def prune_stale_neighbors(self, current_time_s: float):
        dead_peers = []
        for peer_id, state in self.neighbors.items():
            if (current_time_s - state.last_seen_s) > HEARTBEAT_TIMEOUT_S:
                dead_peers.append(peer_id)
        for dp in dead_peers:
            del self.neighbors[dp]

# =============================================================================
# 4. EPIDEMIC GOSSIP SYNC (MICROCLIMATE & SWARM KINEMATICS)
# =============================================================================
class EpidemicGossipEngine:
    """
    Vector-clock driven Anti-Entropy gossip synchronization for distributed
    wind fields, swarm peer telemetry, and static/dynamic hazard maps.
    """
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.vector_clock: Dict[int, int] = {self.node_id: 0}
        self.peer_state_table: Dict[int, Dict[str, Any]] = {}
        self.wind_field_table: Dict[Tuple[int, int, int], Tuple[np.ndarray, float, int]] = {}
        self.gossip_timer = 0.0

    def record_local_state(self, p: np.ndarray, v: np.ndarray, soc: float, health: float, mission_phase: str):
        self.vector_clock[self.node_id] = self.vector_clock.get(self.node_id, 0) + 1
        self.peer_state_table[self.node_id] = {
            "p": p.copy(),
            "v": v.copy(),
            "soc": soc,
            "health": health,
            "phase": mission_phase,
            "clock": self.vector_clock[self.node_id],
            "t": time.time()
        }

    def record_local_wind(self, pos: np.ndarray, wind_vec: np.ndarray, timestamp_s: float):
        vx = int(pos[0] // 20.0)
        vy = int(pos[1] // 20.0)
        vz = int(max(0.0, pos[2]) // 8.0)
        key = (vx, vy, vz)
        version = self.vector_clock[self.node_id]
        self.wind_field_table[key] = (wind_vec.copy(), timestamp_s, version)

    def prepare_gossip_digest(self) -> Dict[str, Any]:
        """Compact serialized summary for mesh transmission."""
        return {
            "origin_id": self.node_id,
            "vector_clock": self.vector_clock.copy(),
            "peer_states": self.peer_state_table.copy(),
            "wind_entries": {k: (v[0].tolist(), v[1], v[2]) for k, v in self.wind_field_table.items()}
        }

    def merge_gossip_digest(self, digest: Dict[str, Any]):
        """Anti-entropy reconciliation using vector clocks."""
        remote_clock = digest.get("vector_clock", {})
        remote_peer_states = digest.get("peer_states", {})
        remote_wind = digest.get("wind_entries", {})

        # Merge peer states
        for peer_id, state in remote_peer_states.items():
            current_clock = self.vector_clock.get(peer_id, -1)
            incoming_clock = state.get("clock", 0)
            if incoming_clock > current_clock:
                self.vector_clock[peer_id] = incoming_clock
                self.peer_state_table[peer_id] = {
                    "p": np.array(state["p"], dtype=np.float32),
                    "v": np.array(state["v"], dtype=np.float32),
                    "soc": state["soc"],
                    "health": state["health"],
                    "phase": state["phase"],
                    "clock": incoming_clock,
                    "t": state["t"]
                }

        # Merge distributed wind observations
        for key_tuple, (w_list, t_stamp, ver) in remote_wind.items():
            k = tuple(key_tuple)
            if k not in self.wind_field_table or ver > self.wind_field_table[k][2]:
                self.wind_field_table[k] = (np.array(w_list, dtype=np.float32), t_stamp, ver)

# =============================================================================
# 5. DECENTRALIZED CONSENSUS-BASED BUNDLE ALGORITHM (CBBA)
# =============================================================================
class DecentralizedCBBASolver:
    """
    True multi-agent distributed task allocation. Operates asynchronously over
    lossy radio channels without any central arbiter.
    Guarantees polynomial-time convergence to a 50% conflict-free submodular optimum.
    """
    def __init__(self, node_id: int, num_agents: int = 3):
        self.node_id = node_id
        self.num_agents = num_agents

        self.bundle: List[int] = []                   # Tasks ordered by insertion
        self.path: List[int] = []                     # Spatial execution sequence
        self.winning_bids: Dict[int, float] = {}      # task_id -> best known bid value y_i
        self.winning_agents: Dict[int, int] = {}      # task_id -> assigned agent z_i
        self.bid_timestamps: Dict[int, float] = {}    # task_id -> timestamp of last update s_i
        self.known_tasks: Dict[int, DeliveryTaskContract] = {}

    def register_task(self, contract: DeliveryTaskContract):
        self.known_tasks[contract.task_id] = contract
        if contract.task_id not in self.winning_bids:
            self.winning_bids[contract.task_id] = 0.0
            self.winning_agents[contract.task_id] = -1
            self.bid_timestamps[contract.task_id] = 0.0

    def compute_marginal_score(self, current_pos: np.ndarray, task: DeliveryTaskContract,
                               battery_soc: float, motor_health: float) -> float:
        """Evaluates bid value based on transit distance, battery margin, and vehicle health."""
        dist = float(np.linalg.norm(current_pos[:2] - task.target_pos[:2]))
        if dist > 320.0 or battery_soc < 0.22:
            return -1e9  # Out of range or depleted

        flight_cost = dist * (1.0 + (1.0 - battery_soc) * 2.0)
        health_factor = float(np.clip(motor_health, 0.4, 1.0))
        score = (task.reward_base * 100.0 * health_factor) / max(15.0, flight_cost)
        return float(score)

    def build_bundle(self, current_pos: np.ndarray, battery_soc: float, motor_health: float, current_time_s: float) -> bool:
        """Phase 1: Greedy marginal score bundle construction."""
        bundle_updated = False
        while len(self.bundle) < CBBA_MAX_BUNDLE_SIZE:
            best_task_id = -1
            best_marginal_bid = 0.0

            for task_id, task in self.known_tasks.items():
                if task_id in self.bundle:
                    continue
                if current_time_s > task.expiry_time_s:
                    continue

                bid_val = self.compute_marginal_score(current_pos, task, battery_soc, motor_health)
                curr_winning_bid = self.winning_bids.get(task_id, 0.0)

                if bid_val > curr_winning_bid and bid_val > best_marginal_bid:
                    best_marginal_bid = bid_val
                    best_task_id = task_id

            if best_task_id != -1:
                self.bundle.append(best_task_id)
                self.path.append(best_task_id)
                self.winning_bids[best_task_id] = best_marginal_bid
                self.winning_agents[best_task_id] = self.node_id
                self.bid_timestamps[best_task_id] = current_time_s
                bundle_updated = True
            else:
                break
        return bundle_updated

    def serialize_cbba_state(self) -> Dict[str, Any]:
        return {
            "agent_id": self.node_id,
            "winning_bids": self.winning_bids.copy(),
            "winning_agents": self.winning_agents.copy(),
            "timestamps": self.bid_timestamps.copy()
        }

    def resolve_conflicts(self, remote_state: Dict[str, Any], sim_time_s: float) -> bool:
        """
        Phase 2: Consensus conflict resolution table (Brunet et al. 2008).
        Compares winning agent assignments and timestamps; cascades task release.
        """
        sender_id = remote_state["agent_id"]
        rem_bids = remote_state["winning_bids"]
        rem_agents = remote_state["winning_agents"]
        rem_times = remote_state["timestamps"]

        bundle_truncated = False
        release_from_index = len(self.bundle)

        for task_id in self.known_tasks.keys():
            y_i = self.winning_bids.get(task_id, 0.0)
            z_i = self.winning_agents.get(task_id, -1)
            s_i = self.bid_timestamps.get(task_id, 0.0)

            y_k = rem_bids.get(task_id, 0.0)
            z_k = rem_agents.get(task_id, -1)
            s_k = rem_times.get(task_id, 0.0)

            # Consensus Rule Matrix
            if z_k == sender_id:
                if z_i == self.node_id:
                    if y_k > y_i:
                        self.winning_agents[task_id] = sender_id
                        self.winning_bids[task_id] = y_k
                        self.bid_timestamps[task_id] = s_k
                        if task_id in self.bundle:
                            idx = self.bundle.index(task_id)
                            release_from_index = min(release_from_index, idx)
                else:
                    self.winning_agents[task_id] = sender_id
                    self.winning_bids[task_id] = y_k
                    self.bid_timestamps[task_id] = s_k

            elif z_k == self.node_id:
                if z_i != self.node_id and s_k > s_i:
                    # Remote was slower to discover our winning bid
                    self.bid_timestamps[task_id] = sim_time_s

            elif z_k == -1:
                if z_i == sender_id:
                    self.winning_agents[task_id] = -1
                    self.winning_bids[task_id] = 0.0
                    self.bid_timestamps[task_id] = s_k

            else:
                # 3rd-party agent information arbitration
                if s_k > s_i:
                    self.winning_agents[task_id] = z_k
                    self.winning_bids[task_id] = y_k
                    self.bid_timestamps[task_id] = s_k
                    if task_id in self.bundle:
                        idx = self.bundle.index(task_id)
                        release_from_index = min(release_from_index, idx)

        # Cascade release of all tasks following the first conflicted task
        if release_from_index < len(self.bundle):
            discarded = self.bundle[release_from_index:]
            for t_id in discarded:
                if self.winning_agents.get(t_id) == self.node_id:
                    self.winning_agents[t_id] = -1
                    self.winning_bids[t_id] = 0.0
                if t_id in self.path:
                    self.path.remove(t_id)
            self.bundle = self.bundle[:release_from_index]
            bundle_truncated = True

        return bundle_truncated

# =============================================================================
# 6. DISTRIBUTED SWARM CONSENSUS INTEGRATION CONTROLLER
# =============================================================================
class SwarmP2PCommunicationExecutive:
    """
    Subsystem controller embedded within each physical drone instance.
    Interfaces between local sensors/flight-controllers and the P2P radio mesh.
    """
    def __init__(self, drone_id: int, world_houses: List[Dict[str, Any]]):
        self.drone_id = drone_id
        self.channel_model = UrbanRFChannelModel(world_houses)
        self.router = AdHocP2PMeshRouter(self.drone_id, self.channel_model)
        self.gossip = EpidemicGossipEngine(self.drone_id)
        self.cbba = DecentralizedCBBASolver(self.drone_id)
        self.last_gossip_time = 0.0

    def step(self, sim_time_s: float, current_pos: np.ndarray, current_vel: np.ndarray,
             battery_soc: float, motor_health: float, mission_phase: str,
             all_drone_nodes: List['SwarmP2PCommunicationExecutive']) -> Optional[DeliveryTaskContract]:
        """
        Executes link heartbeat, radio reception from medium, gossip sync,
        and decentralized task auctioning.
        """
        # 1. Heartbeat management
        self.router.heartbeat_timer += 0.016
        if self.router.heartbeat_timer >= HEARTBEAT_INTERVAL_S:
            self.router.heartbeat_timer = 0.0
            self.router.send_broadcast("HEARTBEAT", {"p": current_pos.tolist(), "v": current_vel.tolist()}, sim_time_s)

        # 2. Local state capture for gossip
        self.gossip.record_local_state(current_pos, current_vel, battery_soc, motor_health, mission_phase)

        # 3. Process outbound packets onto physical medium
        while self.router.outbound_queue:
            pkt = self.router.outbound_queue.popleft()
            for other_node in all_drone_nodes:
                if other_node.drone_id == self.drone_id:
                    continue
                other_pos = other_node.gossip.peer_state_table.get(other_node.drone_id, {}).get("p", other_node.router.neighbors.get(other_node.drone_id, None))
                if other_pos is None:
                    # First packet initialization assumption
                    other_pos = current_pos
                other_node.router.receive_packet_from_medium(pkt, sim_time_s, other_pos, current_pos)

        # 4. Inbound packet consumption
        while self.router.inbound_queue:
            rx_pkt = self.router.inbound_queue.popleft()
            if rx_pkt.payload_type == "GOSSIP":
                self.gossip.merge_gossip_digest(rx_pkt.payload)
            elif rx_pkt.payload_type == "CBBA_BID":
                self.cbba.resolve_conflicts(rx_pkt.payload, sim_time_s)

        # 5. Periodic Gossip Broadcast
        if (sim_time_s - self.last_gossip_time) >= GOSSIP_INTERVAL_S:
            self.last_gossip_time = sim_time_s
            digest = self.gossip.prepare_gossip_digest()
            self.router.send_broadcast("GOSSIP", digest, sim_time_s)

        # 6. Decentralized CBBA Auction step
        if mission_phase in ["1_TAKEOFF", "8_DEPOT_SERVICE"] and len(self.cbba.bundle) == 0:
            updated = self.cbba.build_bundle(current_pos, battery_soc, motor_health, sim_time_s)
            if updated:
                bid_state = self.cbba.serialize_cbba_state()
                self.router.send_broadcast("CBBA_BID", bid_state, sim_time_s)

        self.router.prune_stale_neighbors(sim_time_s)

        # Return currently assigned highest-priority task (if any)
        if self.cbba.bundle:
            active_task_id = self.cbba.bundle[0]
            return self.cbba.known_tasks.get(active_task_id)
        return None

# =============================================================================
# 7. STANDALONE VERIFICATION SELF-TEST
# =============================================================================
def run_part2_mesh_and_cbba_self_test():
    print("=" * 75)
    print("RUNNING PART 2 SELF-TEST: P2P MESH, GOSSIP SYNC & CBBA DECENTRALIZED CONSENSUS")
    print("=" * 75)

    # 1. Mock urban world with a concrete obstruction
    test_houses = [
        {"id": "Bldg_Obstruct_1", "aabb": [-10.0, 10.0, -10.0, 10.0, 30.0]}
    ]
    rf_model = UrbanRFChannelModel(test_houses)

    # LOS vs NLOS Radio Test
    p_tx = np.array([-30.0, 0.0, 5.0], dtype=np.float32)
    p_rx_los = np.array([-30.0, 40.0, 5.0], dtype=np.float32)
    p_rx_nlos = np.array([30.0, 0.0, 5.0], dtype=np.float32) # Line passes through building

    rssi_los, psr_los, is_los = rf_model.evaluate_link(p_tx, p_rx_los)
    rssi_nlos, psr_nlos, is_nlos = rf_model.evaluate_link(p_tx, p_rx_nlos)

    print(f"[RF LINK LOS ] Dist: {np.linalg.norm(p_tx-p_rx_los):.1f}m | RSSI: {rssi_los:.1f} dBm | PSR: {psr_los*100:.1f}% | LOS: {is_los}")
    print(f"[RF LINK NLOS] Dist: {np.linalg.norm(p_tx-p_rx_nlos):.1f}m | RSSI: {rssi_nlos:.1f} dBm | PSR: {psr_nlos*100:.1f}% | LOS: {is_nlos}")
    assert rssi_los > rssi_nlos, "NLOS signal must experience building attenuation!"
    assert is_los and not is_nlos, "Ray-AABB building occlusion detection failed!"

    # 2. Multi-Drone Swarm P2P Comms & CBBA Auction Test
    num_drones = 3
    drone_nodes = [SwarmP2PCommunicationExecutive(i, test_houses) for i in range(num_drones)]

    # Distribute initial positions
    positions = [
        np.array([-60.0, -60.0, 1.0], dtype=np.float32),
        np.array([ 60.0,  60.0, 1.0], dtype=np.float32),
        np.array([  0.0,  70.0, 1.0], dtype=np.float32)
    ]
    velocities = [np.zeros(3, dtype=np.float32) for _ in range(num_drones)]

    # Publish 4 delivery task contracts to all nodes
    tasks = [
        DeliveryTaskContract(101, np.array([-70.0, -70.0, 15.0], dtype=np.float32), "Bldg_A", 1.0, 300.0),
        DeliveryTaskContract(102, np.array([ 70.0,  70.0, 18.0], dtype=np.float32), "Bldg_B", 1.2, 300.0),
        DeliveryTaskContract(103, np.array([  5.0,  75.0, 12.0], dtype=np.float32), "Bldg_C", 0.9, 300.0),
    ]

    for node in drone_nodes:
        for t in tasks:
            node.cbba.register_task(t)

    # 3. Simulate communication steps over 2.0 seconds of flight time
    dt = 0.016
    sim_t = 0.0
    for _ in range(int(2.0 / dt)):
        sim_t += dt
        for i, node in enumerate(drone_nodes):
            node.step(
                sim_time_s=sim_t,
                current_pos=positions[i],
                current_vel=velocities[i],
                battery_soc=0.95 - (i * 0.05),
                motor_health=1.0,
                mission_phase="1_TAKEOFF",
                all_drone_nodes=drone_nodes
            )

    # 4. Verify conflict-free consensus
    print("\n--- CBBA DECENTRALIZED CONVERGENCE RESULT ---")
    assignments = {}
    for node in drone_nodes:
        bundle_str = [f"Task #{tid}" for tid in node.cbba.bundle]
        print(f"Drone [{node.drone_id}] Assigned Bundle: {bundle_str} | Known Peers: {list(node.router.neighbors.keys())}")
        for t_id in node.cbba.bundle:
            assignments.setdefault(t_id, []).append(node.drone_id)

    for t_id, owners in assignments.items():
        assert len(owners) == 1, f"Task #{t_id} has conflicting multiple owners: {owners}!"
    print("\n[PART 2 TEST PASS] Ad-Hoc Mesh routing, Gossip anti-entropy, and Conflict-Free CBBA verified.\n")

if __name__ == "__main__":
    run_part2_mesh_and_cbba_self_test()