"""Regression checks for the behaviour-preserving V99-to-Genesis renderer port.

Place this test file beside drone_master_autonomous_system_v99.py and
Survival_Core_V37_3D_Behavior_Preserved.py, then run:
    python -m unittest -v test_survival_core_v37_fidelity.py

These tests inspect source equivalence and isolated geometry helpers. They do not
replace an end-to-end Genesis/Pygame runtime test.
"""
from __future__ import annotations

import ast
import math
import pathlib
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
ORIGINAL = ROOT / "drone_master_autonomous_system_v99.py"
PORT = ROOT / "Survival_Core_V37_3D_Behavior_Preserved.py"


def parse_file(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def top_level_defs(tree: ast.Module):
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def get_top_level_assignment(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return node
    raise AssertionError(f"top-level assignment {name!r} not found")


def load_helper_functions():
    """Load pure math helpers from the port AST without importing Genesis/Pygame."""
    tree = parse_file(PORT)
    wanted = {"_ray_aabb_first_hit", "_rotation_matrix_to_quaternion_wxyz", "screen_to_world_raycast_genesis", "resolve_genesis_camera_occlusion"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    found = {n.name for n in nodes}
    if found != wanted:
        raise AssertionError(f"Missing pure helpers: {wanted - found}")
    namespace = {"np": np, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PORT), "exec"), namespace)
    return namespace


class SourceFidelityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ORIGINAL.exists() or not PORT.exists():
            raise RuntimeError("Keep the original V99 source beside this test and the behaviour-preserving port.")
        cls.original_tree = parse_file(ORIGINAL)
        cls.port_tree = parse_file(PORT)
        cls.original_defs = top_level_defs(cls.original_tree)
        cls.port_defs = top_level_defs(cls.port_tree)

    def test_all_original_top_level_definitions_are_present(self):
        self.assertEqual(set(self.original_defs) - set(self.port_defs), set())

    def test_only_main_loop_function_changed(self):
        changed = {
            name for name, node in self.original_defs.items()
            if ast.dump(node, include_attributes=False)
            != ast.dump(self.port_defs[name], include_attributes=False)
        }
        self.assertEqual(changed, {"run_all_physics_simulator"})

    def test_model_training_and_delivery_classes_are_ast_identical(self):
        required = {
            "DeepDenseFlightExecutiveNet",
            "DenseResBlock512",
            "_worker_harvest_metropolitan",
            "harvest_and_train_metropolitan",
            "DroneObservationSynthesizer139Dim",
            "SwarmMissionDispatcher",
            "SwarmDroneInstance",
            "RateDampedActuatorMixerWithCBF",
            "AutonomousWinchDeliverySystem",
            "BEMTCoaxialAirframe8DOF",
            "FullPhysicsSensorSuite",
            "SurfaceAndWireAssuranceSuite",
            "AdvancedMultiModalSensorSuite",
        }
        for name in sorted(required):
            self.assertIn(name, self.original_defs, name)
            self.assertIn(name, self.port_defs, name)
            self.assertEqual(
                ast.dump(self.original_defs[name], include_attributes=False),
                ast.dump(self.port_defs[name], include_attributes=False),
                f"{name} changed unexpectedly",
            )

    def test_original_model_contract_constants_preserved(self):
        for name in ("MODEL_WEIGHTS_FILE", "OBSERVATION_DIM", "BASE_OBS_DIM", "HISTORY_LEN", "NUM_SWARM_DRONES", "ACTION_NAMES"):
            self.assertEqual(
                ast.dump(get_top_level_assignment(self.original_tree, name), include_attributes=False),
                ast.dump(get_top_level_assignment(self.port_tree, name), include_attributes=False),
                name,
            )
        source = ORIGINAL.read_text(encoding="utf-8")
        port = PORT.read_text(encoding="utf-8")
        self.assertIn("drone_master_brain_v24_full_sensory.pth", source)
        self.assertIn("drone_master_brain_v24_full_sensory.pth", port)


class GeometryHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = load_helper_functions()

    def test_ray_aabb_near_face(self):
        fn = self.h["_ray_aabb_first_hit"]
        hit = fn(np.array([0., 0., 0.]), np.array([1., 0., 0.]), np.array([2., -1., -1.]), np.array([3., 1., 1.]))
        self.assertAlmostEqual(hit, 2.0)

    def test_ray_aabb_parallel_outside_misses(self):
        fn = self.h["_ray_aabb_first_hit"]
        hit = fn(np.array([0., 2., 0.]), np.array([1., 0., 0.]), np.array([2., -1., -1.]), np.array([3., 1., 1.]))
        self.assertIsNone(hit)

    def test_ray_aabb_origin_inside_returns_exit(self):
        fn = self.h["_ray_aabb_first_hit"]
        hit = fn(np.array([2.5, 0., 0.]), np.array([1., 0., 0.]), np.array([2., -1., -1.]), np.array([3., 1., 1.]))
        self.assertAlmostEqual(hit, 0.5)

    def test_rotation_to_quaternion_identity(self):
        fn = self.h["_rotation_matrix_to_quaternion_wxyz"]
        q = fn(np.eye(3, dtype=np.float32))
        self.assertTrue(np.all(np.isfinite(q)))
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=5)
        self.assertAlmostEqual(abs(float(q[0])), 1.0, places=5)


    def test_camera_is_pulled_in_front_of_building_occlusion(self):
        fn = self.h["resolve_genesis_camera_occlusion"]
        class BuildingWorld:
            houses = [{"aabb": [4.0, 6.0, -1.0, 1.0, 20.0], "clutter": []}]
            trees = []
            powerlines = []
        target = np.array([0.0, 0.0, 10.0])
        desired = np.array([10.0, 0.0, 10.0])
        corrected = fn(desired, target, BuildingWorld())
        self.assertLess(float(corrected[0]), 4.0)
        self.assertGreater(float(corrected[0]), 1.8)
        self.assertAlmostEqual(float(corrected[1]), 0.0, places=5)
        self.assertAlmostEqual(float(corrected[2]), 10.0, places=5)

    def test_screen_ray_returns_ground_hit(self):
        fn = self.h["screen_to_world_raycast_genesis"]
        class World:
            houses = []
            trees = []
            powerlines = []
            depot_helipads = [np.array([0., 0., 0.04], dtype=np.float32)]
        # Camera looks straight down; centre screen ray intersects ground z=0.
        point = fn((50, 50), np.array([0., 0., 10.]), np.array([0., 0., 0.]), 100, 100, World())
        self.assertTrue(np.all(np.isfinite(point)))
        self.assertAlmostEqual(float(point[2]), 0.0, places=5)
        self.assertAlmostEqual(float(point[0]), 0.0, places=5)
        self.assertAlmostEqual(float(point[1]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
