"""Targeted regression tests for the V38.59 audit remediation.

These tests exercise local numerical/control helpers, not the full Genesis GPU mission.
"""
from __future__ import annotations
import importlib.util
import math
import pathlib
import sys
import unittest
import numpy as np

SRC = pathlib.Path(__file__).with_name("Survival_Core_V38_59_Audit_Remediation_One_Click.py")
spec = importlib.util.spec_from_file_location("v3859_audit_module", SRC)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

try:
    import torch
except Exception:
    torch = None


class AuditRemediationTests(unittest.TestCase):
    def test_parameter_requires_grad_call_is_already_correct(self):
        import inspect
        source = inspect.getsource(mod._phase3_configure_trainable_parameters)
        self.assertIn("parameter.requires_grad_(name.startswith(trainable_prefixes))", source)
        self.assertNotIn("parameter.requires_grad(name.startswith(trainable_prefixes))", source)

    def test_hocbf_rhs_units_and_sign(self):
        # h=1 m, hdot=-2 m/s, curvature=0.5 m/s², alpha1=1.5/s, alpha2=2/s
        got = mod._hocbf_acceleration_rhs(1.0, -2.0, 0.5, 1.5, 2.0)
        self.assertAlmostEqual(got, 3.5, places=6)
        # Penetrating barrier should demand stronger positive acceleration if closing.
        self.assertGreater(mod._hocbf_acceleration_rhs(-0.2, -2.0, 0.5),
                           mod._hocbf_acceleration_rhs(1.0, -2.0, 0.5))

    def test_cbf_hildreth_feasible_sphere_constraint(self):
        f = mod.ReciprocalSwarmCBFQPFilter()
        nominal = np.array([-2.0, 0.0, 0.0], dtype=np.float32)
        p = np.array([2.0, 0.0, 5.0], dtype=np.float32)
        v = np.array([-3.0, 0.0, 0.0], dtype=np.float32)
        obs = [{"shape": "sphere", "pos": np.array([0.0, 0.0, 5.0]), "radius": 1.0, "height": 10.0}]
        result = f.enforce_safety(0, p, v, nominal, obs)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(f.last_solver_diagnostics["converged"], f.last_solver_diagnostics)
        self.assertLessEqual(f.last_solver_diagnostics["max_violation_mps2"], 1.1e-3)

    def test_all_roof_barrier_constraints_use_acceleration_dimension_rhs(self):
        import inspect
        body = inspect.getsource(mod.ReciprocalSwarmCBFQPFilter.enforce_safety)
        self.assertNotIn("b_rows.append(-drone_v[2] - 2.0 * h_roof)", body)
        self.assertGreaterEqual(body.count("_hocbf_acceleration_rhs(h_roof"), 2)

    def test_cbf_solver_reports_infeasible_opposed_constraints(self):
        f = mod.ReciprocalSwarmCBFQPFilter()
        # Contradictory half-space constraints: at the same point, opposing spheres
        # are placed so each creates an opposing near-boundary normal.
        p = np.zeros(3, dtype=np.float32)
        v = np.zeros(3, dtype=np.float32)
        obs = [
            {"shape": "sphere", "pos": np.array([-0.1, 0.0, 0.0]), "radius": 1.0, "height": 2.0},
            {"shape": "sphere", "pos": np.array([0.1, 0.0, 0.0]), "radius": 1.0, "height": 2.0},
        ]
        out = f.enforce_safety(0, p, v, np.zeros(3, dtype=np.float32), obs)
        self.assertTrue(np.isfinite(out).all())
        # This geometry is expected to be infeasible or iteration-limited; never
        # require a false success marker for conflicting constraints.
        self.assertFalse(f.last_solver_diagnostics["converged"], f.last_solver_diagnostics)
        self.assertIn(f.last_solver_diagnostics["reason"], {
            "infeasible_or_iteration_limit", "zero_normal_infeasible_constraint"})
        self.assertGreater(f.last_solver_diagnostics["max_violation_mps2"], 1e-3)

    def test_rotation_projection_is_proper_so3(self):
        R = np.array([[1.0, .02, 0.0], [-.01, 1.0, .03], [0.0, -.02, .99]])
        got = mod._project_rotation_to_so3(R)
        np.testing.assert_allclose(got.T @ got, np.eye(3), atol=2e-6)
        self.assertAlmostEqual(float(np.linalg.det(got)), 1.0, places=5)

    def test_dispatcher_removes_selected_non_head_deque_order(self):
        manifest = [
            {"pos": np.array([100., 0., 0.]), "height": 5., "bldg_id": 0, "addr": "a", "tag_id": 0},
            {"pos": np.array([1., 0., 0.]), "height": 5., "bldg_id": 1, "addr": "b", "tag_id": 1},
            {"pos": np.array([80., 0., 0.]), "height": 5., "bldg_id": 2, "addr": "c", "tag_id": 2},
        ]
        d = mod.SwarmMissionDispatcher(manifest, [np.array([0., 0., 0.])])
        m = d.dispatch_next_mission(0, drone_pos=np.array([0., 0., 0.]))
        self.assertEqual(m["order_idx"], 1)
        self.assertEqual(list(d.pending_orders), [0, 2])

    def test_ekf_gated_outlier_does_not_adapt_noise_or_update_position(self):
        ekf = mod.ErrorStateEKF15([1.0, 2.0, 3.0])
        before_p = ekf.p.copy()
        before_scale = ekf.residual_scale
        H = np.zeros((1, 15), dtype=np.float32)
        H[0, 0] = 1.0
        ekf.update_measurement(H, np.array([1e6], dtype=np.float32), np.array([[0.1]], dtype=np.float32))
        np.testing.assert_array_equal(ekf.p, before_p)
        self.assertEqual(ekf.residual_scale, before_scale)

    def test_normal_exit_codes_are_normalized(self):
        self.assertEqual(mod._v3859_normalize_system_exit_code(None), 0)
        self.assertEqual(mod._v3859_normalize_system_exit_code(0), 0)
        self.assertEqual(mod._v3859_normalize_system_exit_code(7), 7)
        self.assertEqual(mod._v3859_normalize_system_exit_code("bad"), 1)

    def test_slack_tether_projection_does_not_push_cargo(self):
        self.assertIsNotNone(torch)
        drone = torch.tensor([[[0., 0., 10.]]], dtype=torch.float32)
        vel = torch.zeros_like(drone)
        cargo = torch.tensor([[[0., 0., 8.]]], dtype=torch.float32)  # 2 m separation
        cable = torch.tensor([[6.0]], dtype=torch.float32)
        loaded = torch.tensor([[True]])
        winch_state = torch.tensor([[1]])
        phase = torch.tensor([[3]])
        roof = torch.tensor([[[100., 100., 0.]]], dtype=torch.float32)
        result, projected_contact = mod._phase3_project_tether_and_contact(
            drone, vel, cargo, cable, loaded, winch_state, phase, roof)
        torch.testing.assert_close(result, cargo)
        self.assertFalse(bool(projected_contact.any()))

    def test_taut_tether_projection_bounds_distance(self):
        self.assertIsNotNone(torch)
        drone = torch.tensor([[[0., 0., 10.]]], dtype=torch.float32)
        vel = torch.zeros_like(drone)
        cargo = torch.tensor([[[0., 0., 0.]]], dtype=torch.float32)
        cable = torch.tensor([[3.0]], dtype=torch.float32)
        loaded = torch.tensor([[True]])
        winch_state = torch.tensor([[1]])
        phase = torch.tensor([[3]])
        roof = torch.tensor([[[100., 100., 0.]]], dtype=torch.float32)
        result, _ = mod._phase3_project_tether_and_contact(
            drone, vel, cargo, cable, loaded, winch_state, phase, roof)
        dist = torch.linalg.vector_norm(result - drone, dim=-1)
        limit = cable + mod.PHASE3_TETHER_CENTER_OFFSET_M
        self.assertLessEqual(float(dist.max()), float(limit.max()) + 1e-5)

    def test_terminal_contact_velocity_is_applied_after_position_difference(self):
        velocity = np.array([[0., 0., 0.], [2., -4., -3.]], dtype=np.float32)
        got = mod._phase3_apply_terminal_contact_velocity(velocity.copy(), True, .85)
        self.assertEqual(float(got[-1, 2]), 0.0)
        np.testing.assert_allclose(got[-1, :2], [1.7, -3.4], atol=1e-6)

    def test_cargo_endpoint_origin_is_valid_only_with_valid_anchor_geometry(self):
        drone = np.array([0., 0., 10.], dtype=np.float32)
        anchor = drone + np.array([0., 0., -mod.PHASE3_DRONE_HALF_HEIGHT_M], dtype=np.float32)
        endpoint_origin = np.array([0., 0., 0.], dtype=np.float32)
        self.assertTrue(mod._phase3_winch_endpoint_valid(9.8, endpoint_origin, drone, anchor))
        self.assertFalse(mod._phase3_winch_endpoint_valid(9.8, np.zeros(3), drone, np.zeros(3)))

    @unittest.skipIf(torch is None, "PyTorch required")
    def test_zero_weighted_cross_entropy_is_finite_zero_with_finite_gradients(self):
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.full((4,), 2, dtype=torch.long)
        weights = torch.tensor([1.0, 1.0, 0.0])
        loss = mod._safe_weighted_cross_entropy(logits, targets, weights)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=8)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_rotational_integrator_remains_orthonormal_under_angular_motion(self):
        a = mod.BEMTCoaxialAirframe8DOF([0., 0., 10.])
        a.R = mod._project_rotation_to_so3(np.array([[1., 0., 0.], [0., .99, -.14], [0., .14, .99]]))
        a.omega = np.array([1.8, -0.9, 1.2], dtype=np.float32)
        cmds = np.full(8, mod.HOVER_OMEGA_NOMINAL * 1.02, dtype=np.float32)
        a.step(cmds, np.zeros(3, dtype=np.float32), 1.225, surface_z=0.0, has_payload=True, dt=.01)
        np.testing.assert_allclose(a.R.T @ a.R, np.eye(3), atol=3e-5)
        self.assertAlmostEqual(float(np.linalg.det(a.R)), 1.0, places=4)
        self.assertTrue(np.isfinite(a.p).all() and np.isfinite(a.v).all())

    def test_source_does_not_use_stale_wind_in_counterfactual_path(self):
        import inspect
        body = inspect.getsource(mod.run_all_physics_simulator)
        self.assertIn('"estimated_wind": np.asarray(estimated_wind, dtype=np.float32).copy()', body)
        self.assertNotIn('"estimated_wind": np.asarray(effective_wind_feed, dtype=np.float32).copy()', body)

    def test_lathed_motor_floor_is_not_applied_to_zero_thrust(self):
        import inspect
        body = inspect.getsource(mod.RateDampedActuatorMixerWithCBF.mix)
        self.assertIn("np.any(active_mask) and f_thrust > 1e-6", body)

    def test_replay_holdout_indices_are_shifted_to_fresh_slice(self):
        import inspect
        body = inspect.getsource(mod.train_phase3_genesis_parallel)
        self.assertIn("train_idx = np.concatenate((np.arange(old_n, dtype=np.int64), fresh_train_idx + old_n))", body)
        self.assertIn("val_idx = val_idx + old_n", body)
        # Current-run held-out rows are indexed into the fresh tail only; offsets are correct.


if __name__ == "__main__":
    unittest.main(verbosity=2)
