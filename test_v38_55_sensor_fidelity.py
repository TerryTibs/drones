"""V38.55 sensor fidelity, live/training parity, and stale-data rejection tests.

These are synthetic CPU tests of the feature builder and report pipeline. They do
not replace an actual Genesis rollout or establish policy safety/performance.
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


class SensorFidelityTests(unittest.TestCase):
    def test_sensor_feature_contract_names_all_140_channels(self):
        self.assertEqual(len(sim.POLICY_SENSOR_FEATURE_NAMES), 140)
        self.assertEqual(len(set(sim.POLICY_SENSOR_FEATURE_NAMES)), 140)
        self.assertEqual(sim.POLICY_SENSOR_FEATURE_NAMES[114], "siren_frequency")
        self.assertEqual(sim.OBSERVATION_DIM, 420)
        self.assertEqual(sim.OBSERVATION_SCHEMA_VERSION, 3)
        self.assertEqual(sim.POLICY_OWNED_CONTROL_SCHEMA_VERSION, 3)

    def test_avionics_proxy_is_finite_stateful_and_bounded(self):
        torch.manual_seed(11055)
        state = sim._phase3_initialize_sensor_fidelity_state(2, 3, torch.device("cpu"))
        pos = torch.zeros((2, 3, 3), dtype=torch.float32)
        vel = torch.tensor([[[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [3.0, 1.0, -0.5]],
                            [[2.0, 0.0, 0.3], [5.0, 1.0, 0.0], [10.0, -2.0, 1.0]]])
        wind = torch.tensor([[1.0, 0.0, 0.0], [3.0, -1.0, 0.0]])
        accel = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 1.0], [-1.0, 0.0, -0.2]],
                              [[0.5, 0.2, 0.0], [0.0, -1.2, 0.4], [3.0, 0.0, -1.0]]])
        soc_before = state["battery_soc"].clone()
        esc_temp_before = state["esc_temp_c"].clone()
        state = sim._phase3_update_sensor_fidelity_state(state, pos, vel, wind, accel, 0.016)
        self.assertTrue(torch.isfinite(state["battery_v_pack"]).all())
        self.assertTrue(torch.isfinite(state["battery_temp_c"]).all())
        self.assertTrue(torch.isfinite(state["esc_current_a"]).all())
        self.assertTrue(torch.isfinite(state["esc_temp_c"]).all())
        self.assertTrue(torch.isfinite(state["ice_thickness_mm"]).all())
        self.assertTrue((state["battery_soc"] <= soc_before).all())
        self.assertTrue((state["battery_soc"] >= 0.0).all())
        self.assertTrue((state["battery_v_pack"] > 0.0).all())
        self.assertTrue((state["esc_current_a"] >= 3.5).all())
        self.assertTrue((state["esc_current_a"] <= 25.0).all())
        self.assertGreater(float(torch.abs(state["esc_temp_c"] - esc_temp_before).max()), 0.0)
        # The real Phase-3 worker passes independently sensed wind per drone [E,D,3].
        state_per_drone = sim._phase3_initialize_sensor_fidelity_state(2, 3, torch.device("cpu"))
        per_drone_wind = torch.randn((2, 3, 3), dtype=torch.float32) * 2.0
        state_per_drone = sim._phase3_update_sensor_fidelity_state(
            state_per_drone, pos, vel, per_drone_wind, accel, 0.016)
        self.assertEqual(tuple(state_per_drone["battery_soc"].shape), (2, 3))
        self.assertEqual(tuple(state_per_drone["esc_temp_c"].shape), (2, 3, 8))
        self.assertTrue(torch.isfinite(state_per_drone["battery_v_pack"]).all())
        with self.assertRaises(ValueError):
            sim._phase3_update_sensor_fidelity_state(
                state_per_drone, pos, vel, torch.zeros((3, 2)), accel, 0.016)

    def test_training_features_drive_traffic_and_adsb_channels_over_time(self):
        torch.manual_seed(41055)
        n_envs, n_drones = 1, 3
        pos = torch.tensor([[[0.0, -60.0, 10.0], [10.0, -60.0, 11.0], [20.0, -60.0, 12.0]]])
        vel = torch.tensor([[[3.0, 1.0, 0.0], [5.0, -1.0, 0.2], [8.0, 0.0, 0.5]]])
        target = torch.tensor([[[15.0, -65.0, 20.0]]]).expand(n_envs, n_drones, 3).clone()
        roof_z = torch.full((n_envs, n_drones), 15.0)
        cable_len = torch.full((n_envs, n_drones), 0.08)
        cable_vel = torch.zeros_like(cable_len)
        has_payload = torch.zeros((n_envs, n_drones), dtype=torch.bool)
        winch = torch.zeros((n_envs, n_drones), dtype=torch.int64)
        phase = torch.zeros_like(winch)
        wind = torch.tensor([[0.5, -1.0, 0.0]])
        accel = torch.zeros_like(pos)
        rot = torch.eye(3).reshape(1, 1, 3, 3).expand(n_envs, n_drones, 3, 3).clone()
        omega = torch.tensor([[[0.1, 0.0, 0.0], [0.0, -0.2, 0.1], [0.2, 0.0, -0.1]]])
        aabbs = np.asarray([[42.0, 50.0, -75.0, -45.0, 30.0]], dtype=np.float32)
        trees = torch.tensor([[12.0, -60.0], [-12.0, -12.0], [12.0, 12.0]])
        state = sim._phase3_initialize_sensor_fidelity_state(n_envs, n_drones, torch.device("cpu"))
        state["traffic_time_offset_s"].fill_(0.0)
        state["aircraft_time_offset_s"].fill_(0.0)
        common = dict(pos=pos, vel=vel, waypoint=target, roof_z=roof_z, cable_len=cable_len,
                      cable_vel=cable_vel, has_payload=has_payload, winch_state=winch, phase=phase,
                      wind=wind, accel=accel, building_aabbs=aabbs, rotation_matrix=rot,
                      omega_body=omega, sensor_state=state, tree_positions=trees,
                      avenue_x=[-120, -60, 0, 60, 120], street_y=[-120, -60, 0, 60, 120], target_pad=target)
        early = sim._phase3_torch_features(**common, sim_time_s=0.0)
        later = sim._phase3_torch_features(**common, sim_time_s=25.0)
        self.assertEqual(tuple(early.shape), (n_envs, n_drones, 140))
        self.assertTrue(torch.isfinite(early).all())
        self.assertTrue(torch.isfinite(later).all())
        # Ambulance is close at t=25 for the chosen geometry: acoustic signal changes.
        self.assertGreater(float(later[..., 114].max()), float(early[..., 114].max()))
        self.assertGreater(float((later[..., 112:114] - early[..., 112:114]).abs().max()), 0.0)
        # Civil aircraft transponder moves continuously; all three coordinates/closing cues change.
        self.assertGreater(float((later[..., 115:119] - early[..., 115:119]).abs().max()), 0.0)
        # Body-frame optical flow includes gyro compensation, not raw world XY velocity.
        self.assertGreater(float(early[..., 104:106].abs().max()), 0.0)
        # Spatial environmental channels differ at different XY locations.
        self.assertGreater(float(early[..., 99].max() - early[..., 99].min()), 0.0)

    def test_channel_metadata_flags_unmodelled_traffic_and_fixed_intruder(self):
        self.assertTrue(sim._policy_sensor_fidelity_metadata(41)["fidelity_class"].endswith("HIGH_RISK"))
        self.assertIn("not yet full rigid-body traffic colliders", sim._policy_sensor_fidelity_metadata(114)["known_limit"])
        self.assertIn("eight independent", sim._policy_sensor_fidelity_metadata(45)["known_limit"])
        self.assertIn("spoofing", sim._policy_sensor_fidelity_metadata(107)["known_limit"])

    def test_lane_report_paths_are_separate_and_reports_include_fidelity(self):
        with mock.patch.dict(os.environ, {
            "SURVIVAL_POLICY_SENSOR_COVERAGE_REPORT": "shared_old_path.json",
            "SURVIVAL_POLICY_SENSOR_TRAINING_REPORT": "train_only.json",
            "SURVIVAL_POLICY_SENSOR_LIVE_REPORT": "live_only.json",
        }):
            self.assertNotEqual(sim._policy_sensor_report_path("phase3_genesis_policy_owned_rollouts"),
                                sim._policy_sensor_report_path("interactive_live_rolling_policy_observations"))
            self.assertTrue(sim._policy_sensor_report_path("phase3_genesis_policy_owned_rollouts").endswith("train_only.json"))
            self.assertTrue(sim._policy_sensor_report_path("interactive_live_rolling_policy_observations").endswith("live_only.json"))
        with tempfile.TemporaryDirectory() as td:
            observations = np.random.default_rng(1).normal(size=(120, 420)).astype(np.float32)
            path, report = sim._write_policy_observation_coverage_report(
                observations, "phase3_genesis_policy_owned_rollouts", str(Path(td) / "training.json"),
                metadata={"checkpoint_sha256": "synthetic-test-only"})
            self.assertTrue(Path(path).is_file())
            self.assertEqual(report["coverage"]["sample_count"], 120)
            self.assertEqual(len(report["coverage"]["features"]), 420)
            self.assertEqual(report["coverage"]["features"][114]["feature_name"], "siren_frequency")
            self.assertIn("fidelity_class", report["coverage"]["features"][114])
            self.assertFalse(report["live_control_authorized"])

    def test_parity_report_detects_shift_insufficient_samples_and_schema_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            rng = np.random.default_rng(2026)
            train_x = rng.normal(0.0, 0.10, size=(180, 420)).astype(np.float32)
            live_x = train_x.copy()
            # Shift the feature in each of the three history slots.
            for slot in range(3):
                live_x[:, slot * 140 + 3] += 0.7
            t_path, _ = sim._write_policy_observation_coverage_report(train_x, "phase3_genesis_policy_owned_rollouts", str(td / "training.json"))
            l_path, _ = sim._write_policy_observation_coverage_report(live_x, "interactive_live_rolling_policy_observations", str(td / "live.json"))
            p_path, report = sim._write_policy_sensor_parity_report(t_path, l_path, str(td / "parity.json"), minimum_live_samples=96)
            self.assertEqual(report["status"], "COMPLETED_REVIEW_REQUIRED")
            self.assertGreater(report["feature_distribution_shift_count"], 0)
            self.assertFalse(report["live_control_authorized"])
            _, small = sim._write_policy_sensor_parity_report(t_path, l_path, str(td / "small.json"), minimum_live_samples=999)
            self.assertEqual(small["status"], "INSUFFICIENT_LIVE_SAMPLE_COVERAGE")
            live_json = json.loads(Path(l_path).read_text(encoding="utf-8"))
            live_json["observation_schema_version"] = 2
            Path(l_path).write_text(json.dumps(live_json), encoding="utf-8")
            _, mismatch = sim._write_policy_sensor_parity_report(t_path, l_path, str(td / "mismatch.json"), minimum_live_samples=96)
            self.assertEqual(mismatch["status"], "BLOCKED_SCHEMA_MISMATCH")

    def test_v2_dagger_corpus_is_not_accepted_under_v3_sensor_semantics(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "old_v2.npz"
            n = 2
            arrays = {
                "X": np.zeros((n, 420), np.float32), "Yg": np.zeros((n, 4), np.float32),
                "Yf": np.zeros((n,), np.int64), "Yw": np.zeros((n, 2), np.float32),
                "Yh": np.zeros((n, 8), np.float32), "Yt": np.zeros((n, 2), np.float32),
                "Yv": np.zeros((n, 1), np.float32), "Ykin": np.zeros((n, 7), np.float32),
                "Ycargo": np.zeros((n, 6), np.float32), "Yforce": np.zeros((n, 4), np.float32),
            }
            meta = {"schema_version": 2, "observation_dim": 420, "sample_count": n}
            with open(path, "wb") as f:
                np.savez_compressed(f, **arrays, source_group_ids=np.array([1, 2]),
                                    source_iteration_ids=np.array([1, 1]), metadata_json=np.asarray(json.dumps(meta)))
            loaded = sim._phase3_load_dagger_replay(str(path), arrays)
            self.assertIsNone(loaded)

    def test_live_acoustic_sensor_call_uses_real_traffic_list(self):
        source = Path(sim.__file__).read_text(encoding="utf-8")
        self.assertIn("poll_directional_acoustic_array(d.airframe.p, traffic_engine.vehicles, sim_time=sim_time)", source)
        self.assertNotIn("poll_directional_acoustic_array(d.airframe.p, [], sim_time=sim_time)", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
