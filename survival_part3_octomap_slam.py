"""
=============================================================================
SURVIVAL CORE V39.0 — MODULE 3 OF 7
DENSE 3D OCTOMAP VOLUMETRIC MAPPING, VIO SLAM & POINT-CLOUD FUSION
=============================================================================
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import deque
import numpy as np

# =============================================================================
# 0. VOLUMETRIC MAPPING & VIO CONSTANTS
# =============================================================================
DEFAULT_VOXEL_RES_M = 0.50          # Base voxel leaf resolution (meters)
OCTREE_MAX_DEPTH = 16               # Supports spatial spans up to 2^16 * resolution
LOG_ODDS_OCCUPIED = 0.85            # Probability P = 0.70 -> log(0.7/0.3)
LOG_ODDS_FREE = -0.40               # Probability P = 0.40 -> log(0.4/0.6)
LOG_ODDS_CLAMP_MIN = -3.5           # P ~ 0.029
LOG_ODDS_CLAMP_MAX = 3.5            # P ~ 0.971
OCCUPANCY_PROB_THRESH = 0.65        # Voxel considered solidly occupied above this

SENSOR_MAX_RANGE_M = 32.0
SENSOR_MIN_RANGE_M = 0.40
SENSOR_HORIZONTAL_FOY_DEG = 360.0
SENSOR_VERTICAL_FOV_DEG = 45.0
SENSOR_AZIMUTH_STEPS = 64
SENSOR_ELEVATION_STEPS = 16

ICP_MAX_ITERATIONS = 20
ICP_CONVERGENCE_EPSILON = 1e-4
ICP_OUTLIER_REJECTION_M = 1.8

ESDF_LOCAL_PATCH_RADIUS_M = 4.0     # Local radius around drone for exact distance transform

# =============================================================================
# 1. 3D GEOMETRY, POINT-CLOUDS & SPATIAL PRIMITIVES
# =============================================================================
@dataclass
class PointCloud3D:
    points: np.ndarray              # Shape (N, 3), float32
    intensities: Optional[np.ndarray] = None
    timestamp_s: float = 0.0
    sensor_origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

@dataclass
class VIOKeyframe:
    frame_id: int
    timestamp_s: float
    drone_pose_p: np.ndarray        # [X, Y, Z] world position
    drone_pose_R: np.ndarray        # 3x3 rotation matrix
    point_cloud: PointCloud3D
    is_loop_closure_candidate: bool = False

# =============================================================================
# 2. FAST 3D AMANATIDES-WOO RAYMARCHING (VOXEL DDA)
# =============================================================================
class AmanatidesWoo3D:
    """
    Exact, grid-traversal Digital Differential Analyzer (DDA) for 3D voxels.
    Traverses every intersected voxel between origin and endpoint without gaps.
    """
    @staticmethod
    def raymarch(origin: np.ndarray, endpoint: np.ndarray, voxel_res: float) -> Tuple[List[Tuple[int, int, int]], Tuple[int, int, int]]:
        direction = endpoint - origin
        dist = float(np.linalg.norm(direction))
        if dist < 1e-5:
            vx = int(math.floor(origin[0] / voxel_res))
            vy = int(math.floor(origin[1] / voxel_res))
            vz = int(math.floor(origin[2] / voxel_res))
            return [], (vx, vy, vz)

        dir_norm = direction / dist

        # Current voxel coordinates
        curr_x = int(math.floor(origin[0] / voxel_res))
        curr_y = int(math.floor(origin[1] / voxel_res))
        curr_z = int(math.floor(origin[2] / voxel_res))

        end_x = int(math.floor(endpoint[0] / voxel_res))
        end_y = int(math.floor(endpoint[1] / voxel_res))
        end_z = int(math.floor(endpoint[2] / voxel_res))

        step_x = 1 if dir_norm[0] > 0 else (-1 if dir_norm[0] < 0 else 0)
        step_y = 1 if dir_norm[1] > 0 else (-1 if dir_norm[1] < 0 else 0)
        step_z = 1 if dir_norm[2] > 0 else (-1 if dir_norm[2] < 0 else 0)

        # Distance to next voxel boundary
        t_max_x = (((curr_x + 1) * voxel_res - origin[0]) / dir_norm[0]) if step_x > 0 else \
                  (((curr_x) * voxel_res - origin[0]) / dir_norm[0]) if step_x < 0 else float('inf')
        t_max_y = (((curr_y + 1) * voxel_res - origin[1]) / dir_norm[1]) if step_y > 0 else \
                  (((curr_y) * voxel_res - origin[1]) / dir_norm[1]) if step_y < 0 else float('inf')
        t_max_z = (((curr_z + 1) * voxel_res - origin[2]) / dir_norm[2]) if step_z > 0 else \
                  (((curr_z) * voxel_res - origin[2]) / dir_norm[2]) if step_z < 0 else float('inf')

        # Voxel step length
        t_delta_x = (voxel_res / abs(dir_norm[0])) if step_x != 0 else float('inf')
        t_delta_y = (voxel_res / abs(dir_norm[1])) if step_y != 0 else float('inf')
        t_delta_z = (voxel_res / abs(dir_norm[2])) if step_z != 0 else float('inf')

        free_cells = []
        max_steps = int((dist / voxel_res) * 2.5) + 4
        steps_taken = 0

        while (curr_x != end_x or curr_y != end_y or curr_z != end_z) and steps_taken < max_steps:
            free_cells.append((curr_x, curr_y, curr_z))
            if t_max_x < t_max_y:
                if t_max_x < t_max_z:
                    curr_x += step_x
                    t_max_x += t_delta_x
                else:
                    curr_z += step_z
                    t_max_z += t_delta_z
            else:
                if t_max_y < t_max_z:
                    curr_y += step_y
                    t_max_y += t_delta_y
                else:
                    curr_z += step_z
                    t_max_z += t_delta_z
            steps_taken += 1

        end_cell = (end_x, end_y, end_z)
        return free_cells, end_cell

# =============================================================================
# 3. HIGH-PERFORMANCE HIERARCHICAL 3D OCTOMAP
# =============================================================================
class Dense3DOctoMap:
    """
    Log-odds spatial occupancy grid with octree-equivalent sub-chunk aggregation.
    Supports continuous free-space carving, surface integration, and fast pruning.
    """
    def __init__(self, resolution_m: float = DEFAULT_VOXEL_RES_M):
        self.res = resolution_m
        # Key: (vx, vy, vz) -> log-odds float32
        self.occupancy_log_odds: Dict[Tuple[int, int, int], float] = {}
        # Spatial chunk index: (cx, cy, cz) -> set of local voxel keys
        self.chunk_size = 16  # 16x16x16 voxels per chunk
        self.spatial_chunks: Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]] = {}
        self.cached_occupied_points: Optional[np.ndarray] = None
        self.cache_valid = False

    def _get_chunk_key(self, voxel_key: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return (
            voxel_key[0] // self.chunk_size,
            voxel_key[1] // self.chunk_size,
            voxel_key[2] // self.chunk_size
        )

    def update_ray(self, origin: np.ndarray, endpoint: np.ndarray, is_endpoint_hit: bool = True):
        free_cells, end_cell = AmanatidesWoo3D.raymarch(origin, endpoint, self.res)

        # Update carved free space
        for cell in free_cells:
            val = self.occupancy_log_odds.get(cell, 0.0) + LOG_ODDS_FREE
            clamped = max(LOG_ODDS_CLAMP_MIN, min(LOG_ODDS_CLAMP_MAX, val))
            self.occupancy_log_odds[cell] = clamped
            chunk = self._get_chunk_key(cell)
            if chunk not in self.spatial_chunks:
                self.spatial_chunks[chunk] = set()
            self.spatial_chunks[chunk].add(cell)

        # Update obstacle surface
        if is_endpoint_hit:
            val = self.occupancy_log_odds.get(end_cell, 0.0) + LOG_ODDS_OCCUPIED
            clamped = max(LOG_ODDS_CLAMP_MIN, min(LOG_ODDS_CLAMP_MAX, val))
            self.occupancy_log_odds[end_cell] = clamped
            chunk = self._get_chunk_key(end_cell)
            if chunk not in self.spatial_chunks:
                self.spatial_chunks[chunk] = set()
            self.spatial_chunks[chunk].add(end_cell)

        self.cache_valid = False

    def insert_point_cloud(self, cloud: PointCloud3D):
        origin = cloud.sensor_origin
        for pt in cloud.points:
            dist = float(np.linalg.norm(pt - origin))
            if dist > SENSOR_MAX_RANGE_M:
                # Carve free space along beam up to max range without registering hit
                unit_ray = (pt - origin) / dist
                end = origin + unit_ray * SENSOR_MAX_RANGE_M
                self.update_ray(origin, end, is_endpoint_hit=False)
            elif dist >= SENSOR_MIN_RANGE_M:
                self.update_ray(origin, pt, is_endpoint_hit=True)
        self.cache_valid = False

    def is_voxel_occupied(self, cell: Tuple[int, int, int]) -> bool:
        log_odds = self.occupancy_log_odds.get(cell, 0.0)
        prob = 1.0 - (1.0 / (1.0 + math.exp(log_odds)))
        return prob >= OCCUPANCY_PROB_THRESH

    def is_point_occupied(self, point: np.ndarray) -> bool:
        cell = (
            int(math.floor(point[0] / self.res)),
            int(math.floor(point[1] / self.res)),
            int(math.floor(point[2] / self.res))
        )
        return self.is_voxel_occupied(cell)

    def get_all_occupied_voxel_centers(self) -> np.ndarray:
        if self.cache_valid and self.cached_occupied_points is not None:
            return self.cached_occupied_points

        centers = []
        for cell, log_odds in self.occupancy_log_odds.items():
            prob = 1.0 - (1.0 / (1.0 + math.exp(log_odds)))
            if prob >= OCCUPANCY_PROB_THRESH:
                centers.append([
                    (cell[0] + 0.5) * self.res,
                    (cell[1] + 0.5) * self.res,
                    (cell[2] + 0.5) * self.res
                ])

        if len(centers) == 0:
            self.cached_occupied_points = np.empty((0, 3), dtype=np.float32)
        else:
            self.cached_occupied_points = np.array(centers, dtype=np.float32)

        self.cache_valid = True
        return self.cached_occupied_points

    def prune_homogeneous_chunks(self):
        """
        Octree pruning equivalent: clears uniform free/empty chunks to save memory.
        """
        pruned_chunks = []
        for chunk_key, cells in self.spatial_chunks.items():
            all_highly_free = True
            for c in cells:
                if self.occupancy_log_odds[c] > (LOG_ODDS_CLAMP_MIN + 0.5):
                    all_highly_free = False
                    break
            if all_highly_free:
                for c in cells:
                    del self.occupancy_log_odds[c]
                pruned_chunks.append(chunk_key)

        for pc in pruned_chunks:
            del self.spatial_chunks[pc]
        self.cache_valid = False

    def export_submap_delta(self, bounding_center: np.ndarray, radius_m: float) -> List[Tuple[int, int, int, float]]:
        """Compact delta serialization for mesh broadcasting to peer drones."""
        delta = []
        rad_sq = radius_m ** 2
        for cell, val in self.occupancy_log_odds.items():
            center_x = (cell[0] + 0.5) * self.res
            center_y = (cell[1] + 0.5) * self.res
            center_z = (cell[2] + 0.5) * self.res
            d2 = (center_x - bounding_center[0])**2 + (center_y - bounding_center[1])**2 + (center_z - bounding_center[2])**2
            if d2 <= rad_sq:
                delta.append((cell[0], cell[1], cell[2], float(val)))
        return delta

    def merge_submap_delta(self, delta: List[Tuple[int, int, int, float]]):
        """Reconciles peer voxel log-odds into local map."""
        for vx, vy, vz, val in delta:
            cell = (vx, vy, vz)
            curr = self.occupancy_log_odds.get(cell, 0.0)
            merged = max(LOG_ODDS_CLAMP_MIN, min(LOG_ODDS_CLAMP_MAX, curr + val * 0.5))
            self.occupancy_log_odds[cell] = merged
            chunk = self._get_chunk_key(cell)
            if chunk not in self.spatial_chunks:
                self.spatial_chunks[chunk] = set()
            self.spatial_chunks[chunk].add(cell)
        self.cache_valid = False

# =============================================================================
# 4. EUCLIDEAN SIGNED DISTANCE FIELD (ESDF) ENGINE
# =============================================================================
class ContinuousLocalESDFEngine:
    """
    Computes real-time Euclidean distance Phi(x) and analytical barrier gradients
    grad(Phi) from the discrete OctoMap for continuous trajectory and CBF filters.
    """
    def __init__(self, octomap: Dense3DOctoMap):
        self.octomap = octomap

    def query_distance_and_gradient(self, query_p: np.ndarray, search_radius_m: float = ESDF_LOCAL_PATCH_RADIUS_M) -> Tuple[float, np.ndarray]:
        """
        Returns:
            min_dist: Distance to nearest occupied surface point (meters).
            grad: Unit vector pointing away from nearest obstacle (world frame).
        """
        occupied_centers = self.octomap.get_all_occupied_voxel_centers()
        if len(occupied_centers) == 0:
            return search_radius_m, np.array([0.0, 0.0, 1.0], dtype=np.float32)

        p = np.asarray(query_p, dtype=np.float32)
        diffs = p - occupied_centers
        dist_sqs = np.sum(diffs ** 2, axis=1)

        min_idx = int(np.argmin(dist_sqs))
        min_dist = float(math.sqrt(dist_sqs[min_idx]))

        if min_dist < 1e-4:
            return 0.0, np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Gradient points away from obstacle
        grad = (p - occupied_centers[min_idx]) / min_dist
        return min_dist, grad.astype(np.float32)

# =============================================================================
# 5. SCAN-TO-SUBMAP ICP ESTIMATOR (VIO SCAN MATCHER)
# =============================================================================
class PointCloudScanMatcherICP:
    """
    Closed-form Kabsch SVD Iterative Closest Point algorithm.
    Refines visual/LiDAR frame-to-frame transform without GNSS.
    """
    def __init__(self, max_iterations: int = ICP_MAX_ITERATIONS, tolerance: float = ICP_CONVERGENCE_EPSILON):
        self.max_iter = max_iterations
        self.tolerance = tolerance

    def align(self, source_points: np.ndarray, target_points: np.ndarray,
              initial_R: Optional[np.ndarray] = None, initial_p: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Aligns source point cloud to target.
        Returns:
            R_opt: Refined 3x3 rotation matrix.
            p_opt: Refined 3D translation vector.
            residual_error: Mean squared correspondence distance.
            converged: Boolean success indicator.
        """
        if len(source_points) < 8 or len(target_points) < 8:
            return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 999.0, False

        R_curr = np.eye(3, dtype=np.float32) if initial_R is None else initial_R.copy()
        p_curr = np.zeros(3, dtype=np.float32) if initial_p is None else initial_p.copy()

        src = source_points.copy()
        prev_error = float('inf')

        for _ in range(self.max_iter):
            # Transform source by current estimate
            transformed_src = (R_curr @ src.T).T + p_curr

            # 1. Nearest neighbor correspondence (vectorized Euclidean)
            # Sample subset if dense to keep latency < 3ms
            sample_stride = max(1, len(transformed_src) // 96)
            sampled_src = transformed_src[::sample_stride]
            untransformed_sampled_src = src[::sample_stride]

            diffs = sampled_src[:, np.newaxis, :] - target_points[np.newaxis, :, :]
            dists = np.sum(diffs ** 2, axis=-1)
            nn_indices = np.argmin(dists, axis=-1)
            matched_targets = target_points[nn_indices]

            # 2. Outlier rejection based on distance threshold
            valid_mask = dists[np.arange(len(sampled_src)), nn_indices] < (ICP_OUTLIER_REJECTION_M ** 2)
            if np.count_nonzero(valid_mask) < 6:
                break

            P = untransformed_sampled_src[valid_mask]
            Q = matched_targets[valid_mask]

            # 3. SVD / Kabsch rigid alignment
            centroid_P = np.mean(P, axis=0)
            centroid_Q = np.mean(Q, axis=0)
            P_centered = P - centroid_P
            Q_centered = Q - centroid_Q

            H = P_centered.T @ Q_centered
            U, S, Vt = np.linalg.svd(H)
            R_step = Vt.T @ U.T

            # Reflection detection
            if np.linalg.det(R_step) < 0.0:
                Vt[-1, :] *= -1.0
                R_step = Vt.T @ U.T

            p_step = centroid_Q - R_step @ centroid_P

            R_curr = R_step.astype(np.float32)
            p_curr = p_step.astype(np.float32)

            # Error evaluation
            curr_error = float(np.mean(np.sum((R_curr @ P.T + p_curr[:, None] - Q.T) ** 2, axis=0)))
            if abs(prev_error - curr_error) < self.tolerance:
                return R_curr, p_curr, curr_error, True
            prev_error = curr_error

        return R_curr, p_curr, prev_error, (prev_error < 0.25)

# =============================================================================
# 6. ONBOARD SLAM & POINT-CLOUD SENSOR INTEGRATION PIPELINE
# =============================================================================
class OnboardVIOSlamExecutive:
    """
    Unified onboard SLAM engine for a single drone.
    Integrates point-cloud raycasting, ICP scan matching, OctoMap updating,
    and local ESDF gradient calculation.
    """
    def __init__(self, drone_id: int, resolution_m: float = DEFAULT_VOXEL_RES_M):
        self.drone_id = drone_id
        self.octomap = Dense3DOctoMap(resolution_m=resolution_m)
        self.esdf = ContinuousLocalESDFEngine(self.octomap)
        self.icp = PointCloudScanMatcherICP()
        
        self.keyframe_history: List[VIOKeyframe] = []
        self.last_keyframe_p = np.zeros(3, dtype=np.float32)
        self.last_keyframe_R = np.eye(3, dtype=np.float32)
        self.keyframe_counter = 0

    def synthesize_sensor_point_cloud(self, true_p: np.ndarray, true_R: np.ndarray, world_houses: List[Dict[str, Any]]) -> PointCloud3D:
        """
        Generates simulated physical 3D LiDAR/Depth sensor returns
        against world buildings with sensor noise and range dropout.
        """
        pts = []
        azimuths = np.linspace(-math.pi, math.pi, SENSOR_AZIMUTH_STEPS, endpoint=False)
        elevations = np.linspace(math.radians(-25.0), math.radians(20.0), SENSOR_ELEVATION_STEPS)

        px, py, pz = true_p
        for el in elevations:
            cos_el = math.cos(el)
            sin_el = math.sin(el)
            for az in azimuths:
                # Ray direction in body frame
                dir_b = np.array([cos_el * math.cos(az), cos_el * math.sin(az), sin_el], dtype=np.float32)
                dir_w = true_R @ dir_b

                # Test intersection with buildings
                closest_hit_dist = SENSOR_MAX_RANGE_M + 5.0
                for h in world_houses:
                    x1, x2, y1, y2, top = h["aabb"]
                    if (pz <= top and dir_w[2] < 0.0) or (pz > top and dir_w[2] < 0.0) or pz <= top:
                        # Simple ray-box intersection
                        dx = 1e-6 if abs(dir_w[0]) < 1e-6 else dir_w[0]
                        dy = 1e-6 if abs(dir_w[1]) < 1e-6 else dir_w[1]
                        tx1 = (x1 - px) / dx
                        tx2 = (x2 - px) / dx
                        ty1 = (y1 - py) / dy
                        ty2 = (y2 - py) / dy
                        t_min = max(min(tx1, tx2), min(ty1, ty2))
                        t_max = min(max(tx1, tx2), max(ty1, ty2))
                        if t_max >= max(0.0, t_min):
                            z_hit = pz + t_min * dir_w[2]
                            if 0.0 <= z_hit <= top:
                                closest_hit_dist = min(closest_hit_dist, float(t_min))

                # Ground plane intersection
                if dir_w[2] < -1e-4:
                    t_gnd = -pz / dir_w[2]
                    if 0.0 <= t_gnd < closest_hit_dist:
                        closest_hit_dist = float(t_gnd)

                if closest_hit_dist <= SENSOR_MAX_RANGE_M:
                    # Realistic range-dependent axial depth noise: sigma = 0.01 + 0.001 * d^2
                    axial_noise = np.random.normal(0.0, 0.015 + 0.0008 * (closest_hit_dist ** 2))
                    noisy_dist = max(SENSOR_MIN_RANGE_M, closest_hit_dist + axial_noise)
                    pts.append(true_p + dir_w * noisy_dist)

        cloud_arr = np.array(pts, dtype=np.float32) if len(pts) > 0 else np.empty((0, 3), dtype=np.float32)
        return PointCloud3D(points=cloud_arr, timestamp_s=time.time(), sensor_origin=true_p.copy())

    def step(self, current_p: np.ndarray, current_R: np.ndarray, world_houses: List[Dict[str, Any]]) -> Tuple[float, np.ndarray]:
        """
        Main execution step.
        Returns:
            min_obstacle_dist: Distance to nearest mapped voxel.
            obstacle_normal_grad: Repulsive unit vector for flight safety.
        """
        # 1. Capture sensor point cloud
        scan = self.synthesize_sensor_point_cloud(current_p, current_R, world_houses)

        # 2. Keyframe evaluation based on spatial translation and rotation
        delta_dist = float(np.linalg.norm(current_p - self.last_keyframe_p))
        delta_rot = float(np.linalg.norm(current_R - self.last_keyframe_R))

        if delta_dist > 1.2 or delta_rot > 0.35 or self.keyframe_counter == 0:
            self.last_keyframe_p = current_p.copy()
            self.last_keyframe_R = current_R.copy()
            kf = VIOKeyframe(
                frame_id=self.keyframe_counter,
                timestamp_s=scan.timestamp_s,
                drone_pose_p=current_p.copy(),
                drone_pose_R=current_R.copy(),
                point_cloud=scan
            )
            self.keyframe_history.append(kf)
            self.keyframe_counter += 1

            # 3. Integrate into hierarchical OctoMap
            self.octomap.insert_point_cloud(scan)

        # 4. Query ESDF obstacle gradient for active CBF-QP barrier guidance
        min_dist, grad = self.esdf.query_distance_and_gradient(current_p)
        return min_dist, grad

# =============================================================================
# 7. STANDALONE VERIFICATION SELF-TEST
# =============================================================================
def run_part3_octomap_and_slam_self_test():
    print("=" * 75)
    print("RUNNING PART 3 SELF-TEST: 3D OCTOMAP, VIO SCAN MATCHING & ESDF ENGINE")
    print("=" * 75)

    mock_world_houses = [
        {"id": "Bldg_1", "aabb": [-15.0, 15.0, 20.0, 40.0, 18.0]},
        {"id": "Bldg_2", "aabb": [-30.0, -10.0, -30.0, -10.0, 12.0]}
    ]

    drone_slam = OnboardVIOSlamExecutive(drone_id=0, resolution_m=0.50)
    drone_pos = np.array([0.0, 0.0, 6.0], dtype=np.float32)
    drone_R = np.eye(3, dtype=np.float32)

    t0 = time.time()
    # 1. Synthesize point cloud and update volumetric grid
    min_dist, grad = drone_slam.step(drone_pos, drone_R, mock_world_houses)
    elapsed_ms = (time.time() - t0) * 1000.0

    occupied_pts = drone_slam.octomap.get_all_occupied_voxel_centers()
    print(f"[OCTOMAP] Integrated initial scan in {elapsed_ms:.2f} ms | Occupied Voxels: {len(occupied_pts)}")
    print(f"[ESDF   ] Nearest Obstacle Distance: {min_dist:.2f} m | Repulsion Normal: {grad}")
    assert len(occupied_pts) > 0, "OctoMap failed to register physical building voxels!"
    assert min_dist > 0.0, "ESDF distance query returned non-positive value!"

    # 2. Test 3D DDA Raymarcher Invariant
    p_orig = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    p_target = np.array([5.0, 0.0, 2.0], dtype=np.float32)
    free_cells, end_cell = AmanatidesWoo3D.raymarch(p_orig, p_target, voxel_res=0.50)
    print(f"[3D DDA ] Carved {len(free_cells)} free voxels along 5.0m beam to endpoint {end_cell}")
    assert len(free_cells) == 10, f"Expected exactly 10 intermediate 0.5m voxels, got {len(free_cells)}"
    assert end_cell == (10, 0, 4), f"Unexpected endpoint voxel key: {end_cell}"

    # 3. Test Closed-form SVD ICP Alignment
    src_test = np.random.uniform(-4.0, 4.0, (48, 3)).astype(np.float32)
    true_rot_angle = math.radians(12.0)
    K = np.array([
        [math.cos(true_rot_angle), -math.sin(true_rot_angle), 0.0],
        [math.sin(true_rot_angle),  math.cos(true_rot_angle), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    true_trans = np.array([0.45, -0.25, 0.10], dtype=np.float32)
    tgt_test = (K @ src_test.T).T + true_trans

    R_est, p_est, err, converged = drone_slam.icp.align(src_test, tgt_test)
    print(f"[ICP VIO] Alignment Converged: {converged} | Residual MSE: {err:.6f}")
    assert converged, "ICP scan matcher failed to converge on known rigid transform!"
    assert np.allclose(p_est, true_trans, atol=0.08), f"ICP translation estimate off: {p_est} vs {true_trans}"

    # 4. Multi-Drone Submap Delta Fusion Test (Mesh integration)
    peer_slam = OnboardVIOSlamExecutive(drone_id=1, resolution_m=0.50)
    delta = drone_slam.octomap.export_submap_delta(bounding_center=drone_pos, radius_m=30.0)
    peer_slam.octomap.merge_submap_delta(delta)
    peer_occupied = peer_slam.octomap.get_all_occupied_voxel_centers()

    print(f"[SWARM FUSION] Exported {len(delta)} voxels | Peer Integrated Occupied Voxels: {len(peer_occupied)}")
    assert len(peer_occupied) == len(occupied_pts), "Submap delta merge resulted in mismatched voxel count!"

    print("\n[PART 3 TEST PASS] Hierarchical OctoMap, 3D DDA, ESDF, and SVD-ICP verified.\n")

if __name__ == "__main__":
    run_part3_octomap_and_slam_self_test()