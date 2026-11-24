"""Focused tests for V38.55 sensor-fidelity model-owned control mode.

These tests validate action decoding, input contract, checkpoint markers and the
actual high-level mixer branch. They do not qualify the policy or run Genesis.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import Survival_Core_V38_55_Sensor_Fidelity_Parity as sim


class ModelOwnedControlTests(unittest.TestCase):
    def test_acceleration_decoder_bounds_si_command(self):
        np.testing.assert_allclose(
            sim._policy_owned_nominal_acceleration([0.5, -0.5, 0.25, 0.99]),
            np.array([6.0, -6.0, 3.0], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(
            sim._policy_owned_nominal_acceleration([1.0, -1.0, 1.0]),
            np.array([8.0, -8.0, 6.0], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(
            sim._policy_owned_nominal_acceleration([-1.0, 1.0, -1.0]),
            np.array([-8.0, 8.0, -5.0], dtype=np.float32), atol=1e-6)
        with self.assertRaises(ValueError):
            sim._policy_owned_nominal_acceleration([0.0, np.nan, 0.0])
        with self.assertRaises(ValueError):
            sim._policy_owned_nominal_acceleration([0.0, 0.0])

    def test_action_filter_blocks_unsupported_or_phase_illegal_actions(self):
        filt = sim._policy_owned_safe_action_token
        self.assertEqual(filt(6, "2_CRUISE_OUTBOUND", "STANDBY"), 1)
        self.assertEqual(filt(2, "2_CRUISE_OUTBOUND", "STANDBY"), 1)
        self.assertEqual(filt(4, "3_CHIMNEY_DESCENT", "STANDBY"), 1)
        self.assertEqual(filt(5, "4_DELIVERY_HOVER", "STANDBY"), 1)
        self.assertEqual(filt(3, "4_DELIVERY_HOVER", "STANDBY"), 1)
        self.assertEqual(filt(2, "4_DELIVERY_HOVER", "STANDBY"), 2)
        self.assertEqual(filt(3, "4_DELIVERY_HOVER", "LOWERING"), 3)
        self.assertEqual(filt(4, "4_DELIVERY_HOVER", "LOWERING"), 4)
        self.assertEqual(filt(5, "8_DEPOT_SERVICE", "STANDBY"), 5)
        self.assertEqual(filt(99, "2_CRUISE_OUTBOUND", "STANDBY"), 1)

    def test_feature_coverage_reports_all_420_channels_and_nonfinite_values(self):
        arr = np.zeros((4, sim.OBSERVATION_DIM), dtype=np.float32)
        arr[:, 0] = np.arange(4, dtype=np.float32)
        arr[1, 140 + 3] = np.nan
        report = sim._summarize_policy_observation_coverage(arr)
        self.assertEqual(report["observation_dim"], 420)
        self.assertEqual(len(report["features"]), 420)
        self.assertAlmostEqual(report["finite_value_fraction"], 1.0 - 1.0 / (4 * 420))
        self.assertEqual(report["channels_with_any_nonfinite"], 1)
        self.assertTrue(report["features"][0]["constant_or_frozen_in_sample"] is False)
        self.assertEqual(report["features"][143]["nonfinite_samples"], 1)
        with self.assertRaises(ValueError):
            sim._summarize_policy_observation_coverage(np.zeros((4, 140), dtype=np.float32))
        with self.assertRaises(ValueError):
            sim._summarize_policy_observation_coverage(np.empty((0, 417), dtype=np.float32))

    def test_training_and_live_sensor_reports_have_distinct_default_paths(self):
        obs = np.random.default_rng(7).normal(size=(3, 420)).astype(np.float32)
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                train_path, train_payload = sim._write_policy_observation_coverage_report(
                    obs, "phase3_genesis_policy_owned_rollouts")
                live_path, live_payload = sim._write_policy_observation_coverage_report(
                    obs, "interactive_live_multimodal_sensor_synthesizer")
                self.assertNotEqual(train_path, live_path)
                self.assertTrue(Path(train_path).is_file())
                self.assertTrue(Path(live_path).is_file())
                self.assertEqual(len(json.loads(Path(train_path).read_text())["coverage"]["features"]), 420)
                self.assertEqual(len(json.loads(Path(live_path).read_text())["coverage"]["features"]), 420)
                self.assertFalse(train_payload["live_control_authorized"])
                self.assertFalse(live_payload["live_control_authorized"])
            finally:
                os.chdir(old_cwd)

    def test_checkpoint_marker_requires_training_samples_and_current_schema(self):
        model = sim.DeepDenseFlightExecutiveNet(in_features=sim.OBSERVATION_DIM)
        self.assertFalse(sim._model_has_policy_owned_training_marker(model))
        model._policy_owned_control_trained_state.fill_(True)
        self.assertFalse(sim._model_has_policy_owned_training_marker(model),
                         "a boolean marker without sample provenance must not qualify")
        model._policy_owned_control_training_samples.fill_(1024)
        self.assertTrue(sim._model_has_policy_owned_training_marker(model))
        model._policy_owned_control_schema_version.fill_(sim.POLICY_OWNED_CONTROL_SCHEMA_VERSION + 1)
        self.assertFalse(sim._model_has_policy_owned_training_marker(model))

    def test_legacy_checkpoint_loads_but_does_not_gain_control_marker(self):
        model = sim.DeepDenseFlightExecutiveNet(in_features=sim.OBSERVATION_DIM)
        state = model.state_dict()
        state.pop("_policy_owned_control_trained_state")
        state.pop("_policy_owned_control_training_samples")
        state.pop("_policy_owned_control_schema_version")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy_state.pt"
            torch.save(state, path)
            target = sim.DeepDenseFlightExecutiveNet(in_features=sim.OBSERVATION_DIM)
            sim._load_model_checkpoint(target, str(path), torch.device("cpu"))
            self.assertFalse(sim._model_has_policy_owned_training_marker(target))
            self.assertEqual(int(target._policy_owned_control_training_samples.item()), 0)

    def test_real_mixer_branch_uses_model_acceleration_as_nominal_control(self):
        mixer = sim.RateDampedActuatorMixerWithCBF()
        motors, thrust, safe_accel = mixer.mix(
            drone_health=np.ones(8, dtype=np.float32),
            p_est=np.array([0.0, 0.0, 5.0], dtype=np.float32),
            v_est=np.zeros(3, dtype=np.float32),
            R_est=np.eye(3, dtype=np.float32),
            omega_est=np.zeros(3, dtype=np.float32),
            p_target=np.array([100.0, -80.0, 40.0], dtype=np.float32),
            neural_guidance=np.array([0.5, 0.0, 0.2, 0.0], dtype=np.float32),
            token_idx=0,
            estimated_wind=np.zeros(3, dtype=np.float32),
            obstacles=[],
            ref_vel=np.array([4.0, -2.0, 1.0], dtype=np.float32),
            ref_accel=np.array([2.0, 2.0, 2.0], dtype=np.float32),
            policy_owned_control=True,
        )
        self.assertEqual(motors.shape, (8,))
        self.assertTrue(np.isfinite(motors).all())
        self.assertTrue(np.isfinite(thrust))
        # Without obstacles the CBF layer should preserve the requested nominal
        # acceleration. Legacy position/velocity/wind controllers would add other terms.
        np.testing.assert_allclose(safe_accel, [6.0, 0.0, 2.4], atol=1e-5)

    def test_phase3_feature_generator_includes_siren_channel_in_420_contract(self):
        pos = torch.zeros((1, 3, 3), dtype=torch.float32)
        vel = torch.zeros_like(pos)
        waypoint = torch.ones_like(pos)
        roof_z = torch.zeros((1, 3), dtype=torch.float32)
        cable_len = torch.zeros_like(roof_z)
        cable_vel = torch.zeros_like(roof_z)
        has_payload = torch.zeros((1, 3), dtype=torch.bool)
        winch_state = torch.zeros((1, 3), dtype=torch.int64)
        phase = torch.zeros((1, 3), dtype=torch.int64)
        wind = torch.zeros((1, 3), dtype=torch.float32)
        accel = torch.zeros_like(pos)
        out = sim._phase3_torch_features(
            pos, vel, waypoint, roof_z, cable_len, cable_vel, has_payload,
            winch_state, phase, wind, accel, np.empty((0, 5), dtype=np.float32))
        self.assertEqual(tuple(out.shape), (1, 3, 140))
        torch.testing.assert_close(
            out[..., sim.SIREN_FREQUENCY_FEATURE_INDEX],
            torch.full((1, 3), sim.LEGACY_SIREN_FREQUENCY_FILL))

    def test_phase3_air_data_angles_and_gyro_use_simulated_state(self):
        pos = torch.zeros((1, 3, 3), dtype=torch.float32)
        vel = torch.tensor([[[12.0, 2.0, 1.5], [8.0, -1.0, 0.5], [5.0, 0.5, -0.5]]])
        waypoint = torch.ones_like(pos)
        roof_z = torch.zeros((1, 3), dtype=torch.float32)
        cable_len = torch.full((1, 3), 0.08, dtype=torch.float32)
        cable_vel = torch.zeros_like(roof_z)
        has_payload = torch.zeros((1, 3), dtype=torch.bool)
        winch_state = torch.zeros((1, 3), dtype=torch.int64)
        phase = torch.zeros((1, 3), dtype=torch.int64)
        wind = torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float32)
        accel = torch.zeros_like(pos)
        omega = torch.tensor([[[0.1, -0.2, 0.3], [0.2, 0.1, -0.1], [0.0, 0.3, 0.2]]])
        rot = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 3, 3, 3).clone()
        out = sim._phase3_torch_features(
            pos, vel, waypoint, roof_z, cable_len, cable_vel, has_payload,
            winch_state, phase, wind, accel, np.empty((0, 5), dtype=np.float32),
            rotation_matrix=rot, omega_body=omega)
        self.assertEqual(tuple(out.shape), (1, 3, 140))
        self.assertGreater(float(out[..., 94].abs().max()), 0.0)
        self.assertGreater(float(out[..., 95].abs().max()), 0.0)
        torch.testing.assert_close(out[..., 134:137], omega / 10.0)
        self.assertTrue(torch.isfinite(out).all())

    def test_legacy_dagger_replay_is_rejected_after_sensor_semantics_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy_v1_replay.npz"
            n = 2
            expected = {
                "X": np.zeros((n, 420), dtype=np.float32),
                "Yg": np.zeros((n, 4), dtype=np.float32),
                "Yf": np.ones((n,), dtype=np.int64),
                "Yw": np.zeros((n, 2), dtype=np.float32),
                "Yh": np.zeros((n, 8), dtype=np.float32),
                "Yt": np.zeros((n, 2), dtype=np.float32),
                "Yv": np.zeros((n, 1), dtype=np.float32),
                "Ykin": np.zeros((n, 7), dtype=np.float32),
                "Ycargo": np.zeros((n, 6), dtype=np.float32),
                "Yforce": np.zeros((n, 4), dtype=np.float32),
            }
            expected_shapes = sim._phase3_dagger_expected_shapes(expected)
            saved = {key: value.copy() for key, value in expected.items()}
            saved["X"] = np.zeros((n, 417), dtype=np.float32)
            saved["Yf"][:] = 1
            metadata = {"schema_version": 1, "observation_dim": 417, "sample_count": n}
            with open(path, "wb") as f:
                np.savez_compressed(
                    f, **saved, source_group_ids=np.asarray([10, 11], dtype=np.int64),
                    metadata_json=np.asarray(json.dumps(metadata)))
            loaded = sim._phase3_load_dagger_replay(str(path), expected)
            self.assertIsNone(loaded, "old DAgger data must be regenerated under V38.55 sensor semantics")

    def test_live_observation_includes_acoustic_siren_frequency(self):
        synth = sim.DroneObservationSynthesizer140Dim()
        peer = []
        common = dict(
            p_est=np.array([1.0, 2.0, 3.0], dtype=np.float32), v_est=np.zeros(3, dtype=np.float32),
            R_est=np.eye(3, dtype=np.float32), omega_est=np.zeros(3, dtype=np.float32),
            target_p=np.array([5.0, 2.0, 8.0], dtype=np.float32), lidar_16=np.ones(16, dtype=np.float32),
            conical_8=np.ones(8, dtype=np.float32), v_tas=1.0, q_dyn=2.0, min_sonar=3.0,
            wind_est=np.zeros(3, dtype=np.float32), ice_idx=0.0, cable_len=0.0, cable_vel=0.0,
            tension_n=0.0, sway_2d=np.zeros(2, dtype=np.float32), bms_soc=0.9, v_pack=24.0,
            bms_temp_c=22.0, r0_dyn=0.01, chi2_val=1.0, jerk_discrepancy=0.0,
            is_spoofed=False, emf_rms=1.0, wire_dist=20.0, agent_dist=25.0, ground_spl_dba=55.0,
            has_payload=False, hook_a=np.zeros(3, dtype=np.float32), hook_w=np.zeros(3, dtype=np.float32),
            load_cell_3d=np.zeros(3, dtype=np.float32), thermal_dt=0.0, dolp=0.2, ndvi=0.2,
            esc_i_rms=np.ones(8, dtype=np.float32), esc_t_mosfet=np.ones(8, dtype=np.float32),
            alpha_deg=0.0, beta_deg=0.0, rh_pct=50.0, dust_pm10=3.0, hdop=1.0, cn0_avg=40.0,
            rf_jam_pwr=-80.0, mag_grad=0.0, airframe_accel=np.zeros(3, dtype=np.float32),
            airframe_gyro=np.zeros(3, dtype=np.float32), peer_swarm_kinematics=peer,
            optical_flow_2d=np.zeros(2, dtype=np.float32), flow_quality=1.0,
            aruco_offset_3d=np.zeros(3, dtype=np.float32), aruco_locked=1.0,
            skid_force_left=0.0, skid_force_right=0.0, is_skid_grounded=False,
            siren_angle=0.0, siren_spl=55.0, adsb_rel_pos=np.zeros(3, dtype=np.float32),
            adsb_closing_spd=0.0)
        obs = synth.synthesize_instantaneous(**common, siren_freq=850.0)
        self.assertEqual(obs.shape, (140,))
        self.assertAlmostEqual(float(obs[sim.SIREN_FREQUENCY_FEATURE_INDEX]), 0.85, places=6)
        self.assertTrue(np.isfinite(obs).all())

    def test_legacy_417_input_upgrades_and_preserves_feature_order(self):
        old = np.arange(417, dtype=np.float32).reshape(1, 417) / 100.0
        new = sim._upgrade_legacy_observation_array(old)
        self.assertEqual(new.shape, (1, 420))
        old_frames = old.reshape(1, 3, 139)
        new_frames = new.reshape(1, 3, 140)
        np.testing.assert_allclose(new_frames[..., :114], old_frames[..., :114])
        np.testing.assert_allclose(new_frames[..., 114], sim.LEGACY_SIREN_FREQUENCY_FILL)
        np.testing.assert_allclose(new_frames[..., 115:], old_frames[..., 114:])
        tensor = sim._upgrade_legacy_observation_tensor(torch.from_numpy(old))
        np.testing.assert_allclose(tensor.numpy(), new)

    def test_legacy_checkpoint_projection_migration_preserves_outputs(self):
        torch.manual_seed(17)
        legacy_model = sim.DeepDenseFlightExecutiveNet(in_features=417).eval()
        state = legacy_model.state_dict()
        # Simulate an actual older checkpoint without the V38.54 ownership buffers.
        state.pop("_policy_owned_control_trained_state", None)
        state.pop("_policy_owned_control_training_samples", None)
        state.pop("_policy_owned_control_schema_version", None)
        migrated = sim.DeepDenseFlightExecutiveNet(in_features=sim.OBSERVATION_DIM).eval()
        incompatible = migrated.load_state_dict(state, strict=False)
        self.assertEqual(migrated.input_proj[0].weight.shape[1], 420)
        self.assertIn("_policy_owned_control_trained_state", incompatible.missing_keys)
        old_input = torch.randn(3, 417)
        new_input = sim._upgrade_legacy_observation_tensor(old_input)
        with torch.no_grad():
            old_out = legacy_model(old_input)[0]
            new_out = migrated(new_input)[0]
        torch.testing.assert_close(old_out, new_out, rtol=1e-5, atol=1e-6)

    def test_model_consumes_full_stacked_tensor_and_outputs_all_control_heads(self):
        model = sim.DeepDenseFlightExecutiveNet(in_features=sim.OBSERVATION_DIM).eval()
        obs = torch.linspace(-1.0, 1.0, steps=3 * sim.OBSERVATION_DIM).reshape(3, sim.OBSERVATION_DIM)
        captured = {}
        handle = model.input_proj.register_forward_pre_hook(
            lambda _module, inputs: captured.setdefault("x", inputs[0].detach().clone()))
        with torch.no_grad():
            guidance, fsm, winch, health, trust, value = model(obs)
        handle.remove()
        self.assertTrue(torch.equal(captured["x"], obs))
        self.assertEqual(tuple(guidance.shape), (3, 4))
        self.assertEqual(tuple(fsm.shape), (3, len(sim.ACTION_NAMES)))
        self.assertEqual(tuple(winch.shape), (3, 2))
        self.assertTrue(torch.isfinite(guidance).all())
        self.assertTrue(torch.isfinite(fsm).all())
        self.assertTrue(torch.isfinite(winch).all())

    def test_live_path_has_explicit_contract_and_no_legacy_phase_gate_in_model_owned_mode(self):
        source = Path(sim.__file__).read_text(encoding="utf-8")
        self.assertIn("live_observation_batch.shape != (len(drones), OBSERVATION_DIM)", source)
        self.assertIn("if model_owns_control_requested and not np.isfinite(live_observation_batch).all()", source)
        self.assertIn("b_guidance if model_owns_control_requested else (b_guidance * 0.15)", source)
        self.assertIn("policy_owned_control=model_owns_control_requested", source)
        self.assertIn("expert_delivery = torch.zeros_like(phase, dtype=torch.bool)", source)
        self.assertIn("teacher_winch_mask = torch.zeros((n_envs, NUM_SWARM_DRONES)", source)
        self.assertIn("if model_owns_control_requested:\n                nn_fsm_token = _policy_owned_safe_action_token", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
