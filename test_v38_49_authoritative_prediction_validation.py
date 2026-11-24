"""Synthetic V38.49 contract test. It validates evaluator plumbing, not model quality."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch

SOURCE = Path(__file__).with_name("Survival_Core_V38_49_Authoritative_Cargo_Winch_Prediction_Validation.py")
spec = importlib.util.spec_from_file_location("survival_v3849_test_module", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import source: {SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _make_data(path: Path, nseq: int = 2, steps: int = 20) -> None:
    m = module
    obs = np.zeros((nseq, steps, m.OBSERVATION_DIM), dtype=np.float32)
    act = np.zeros((nseq, steps, m.ACTION_TRANSITION_ACTION_DIM), dtype=np.float32)
    targets = np.zeros((nseq, steps, m.ACTION_TRANSITION_TARGET_DIM), dtype=np.float32)
    nodes = np.zeros((nseq, steps, 5, 3), dtype=np.float32)
    node_vel = np.zeros_like(nodes)
    scalars = np.zeros((nseq, steps, len(m.AUTH_CARGO_WINCH_SCALAR_NAMES)), dtype=np.float32)
    events = np.zeros((nseq, steps, len(m.AUTH_CARGO_WINCH_EVENT_NAMES)), dtype=np.float32)
    codes = np.zeros((nseq, steps, 4), dtype=np.int16)
    frames = np.zeros((nseq, steps), dtype=np.int64)
    reset = np.zeros((nseq, steps), dtype=np.int32)
    cargo_gen = np.zeros((nseq, steps), dtype=np.int32)
    si = {v: i for i, v in enumerate(m.AUTH_CARGO_WINCH_SCALAR_NAMES)}
    ei = {v: i for i, v in enumerate(m.AUTH_CARGO_WINCH_EVENT_NAMES)}
    for s in range(nseq):
        for t in range(steps):
            cable_before = 1.0 + 0.005 * t
            cable_target = cable_before + (0.002 if s == 0 else -0.001) * (t + 1)
            cable_rate = 0.125 if s == 0 else -0.0625
            tension_before = 5.0 + 0.1 * t
            tension_target = tension_before + (0.05 if s == 0 else -0.02)
            sway_before = np.array([0.12 + 0.002 * t, -0.08 + 0.001 * t], dtype=np.float32)
            sway_target = sway_before + np.array([0.001, -0.0005], dtype=np.float32)
            rate_before = np.array([0.03 * np.sin(t / 3.0), -0.02], dtype=np.float32)
            rate_target = rate_before + np.array([0.002, -0.001], dtype=np.float32)
            latest_fields = np.zeros(m.BASE_OBS_DIM, dtype=np.float32)
            latest_fields[9:18] = np.eye(3, dtype=np.float32).reshape(-1)
            latest_fields[53] = cable_before / 15.0
            latest_fields[54] = cable_rate / 5.0
            latest_fields[55] = tension_before / 50.0
            latest_fields[56:58] = sway_before / 2.0
            latest_fields[58] = 1.0
            latest_fields[89] = -rate_before[0] * 1.8 / 10.0
            latest_fields[88] = rate_before[1] * 1.8 / 10.0
            latest_fields[3:6] = [0.1 * s, 0.05 * t, 12.0 / 200.0]
            # same instantaneous frame replicated across history; valid rotation contract
            obs[s, t] = np.tile(latest_fields, m.HISTORY_LEN)
            act[s, t, 3 + 3] = 1.0  # one FSM token
            act[s, t, -1] = 0.3 if s == 0 else -0.2
            targets[s, t, -1] = 1.0
            drone_p = np.array([10.0 + s, -4.0 + 0.1 * t, 12.0], dtype=np.float32)
            drone_v = np.array([0.4, -0.1, 0.0], dtype=np.float32)
            scalar = scalars[s, t]
            scalar[si["cable_length_m"]] = cable_target
            scalar[si["cable_rate_mps"]] = cable_rate
            scalar[si["tension_n"]] = tension_target
            scalar[si["tension_derivative_nps"]] = 1.25
            scalar[si["contact_normal_force_n"]] = 0.0
            scalar[si["cargo_surface_clearance_m"]] = 1.8 - 0.01 * t
            scalar[si["target_surface_z_m"]] = 0.0
            scalar[si["sample_time_s"]] = t * 0.016
            scalar[si["release_count"]] = 0.0
            scalar[si["touchdown_timer_s"]] = 0.0
            scalar[si["drone_position_x_m"]:si["drone_position_z_m"] + 1] = drone_p
            scalar[si["drone_velocity_x_mps"]:si["drone_velocity_z_mps"] + 1] = drone_v
            scalar[si["cargo_yaw_rad"]] = 0.05
            scalar[si["cargo_yaw_rate_rads"]] = 0.01
            end = np.array([drone_p[0] + sway_target[0], drone_p[1] + sway_target[1], drone_p[2] - cable_target], dtype=np.float32)
            start = drone_p + np.array([0.0, 0.0, -0.05], dtype=np.float32)
            nodes[s, t] = np.linspace(start, end, 5, dtype=np.float32)
            rel_v = np.array([rate_target[0], rate_target[1], -cable_rate], dtype=np.float32)
            node_vel[s, t] = np.linspace(drone_v, drone_v + rel_v, 5, dtype=np.float32)
            ev = events[s, t]
            ev[ei["payload_present_before_step"]] = 1.0
            ev[ei["payload_present_after_step"]] = 1.0
            ev[ei["cargo_clearance_valid"]] = 1.0
            ev[ei["telemetry_valid"]] = 1.0
            codes[s, t, 0] = m.AUTH_WINCH_STATE_CODES["LOWERING"]
            frames[s, t] = 1000 * s + t
    metadata = {
        "schema_version": 2,
        "authoritative_cargo_winch_telemetry_schema_version": m.AUTH_CARGO_WINCH_TELEMETRY_SCHEMA_VERSION,
        "dt_s": 0.016,
        "scalar_fields": list(m.AUTH_CARGO_WINCH_SCALAR_NAMES),
        "event_fields": list(m.AUTH_CARGO_WINCH_EVENT_NAMES),
        "code_fields": ["winch_state_code", "payout_pause_reason_code", "abort_reason_code", "mission_phase_code"],
        "source_code_version": m.SURVIVAL_CORE_VERSION,
        "source_code_sha256": "synthetic-fixture-not-production",
        "split": "synthetic-test-only",
        "collection_run_id": "synthetic-test",
        "checkpoint_binding": "not applicable: synthetic fixture",
    }
    np.savez_compressed(
        path, observations=obs, actions=act, targets=targets,
        authoritative_node_positions_m=nodes,
        authoritative_node_velocities_mps=node_vel,
        authoritative_cargo_winch_scalars=scalars,
        authoritative_cargo_winch_events=events,
        authoritative_cargo_winch_codes=codes,
        authoritative_frame_ids=frames,
        authoritative_reset_generations=reset,
        authoritative_cargo_generations=cargo_gen,
        worker_ids=np.arange(nseq, dtype=np.int32),
        drone_ids=np.zeros(nseq, dtype=np.int32),
        start_frames=np.array([0, 0], dtype=np.int64)[:nseq],
        authoritative_valid_counts=np.full(nseq, steps, dtype=np.int32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def main() -> None:
    keys = (
        "SURVIVAL_AUTH_CARGO_WINCH_MODEL_DATASET", "SURVIVAL_AUTH_CARGO_WINCH_MODEL_CHECKPOINT",
        "SURVIVAL_AUTH_CARGO_WINCH_MODEL_REPORT", "SURVIVAL_AUTH_CARGO_WINCH_MODEL_DEVICE",
        "SURVIVAL_AUTH_CARGO_WINCH_MODEL_MAX_ROWS", "SURVIVAL_AUTH_CARGO_WINCH_MODEL_BATCH_SIZE",
    )
    old = {k: os.environ.get(k) for k in keys}
    try:
        torch.manual_seed(49)
        with tempfile.TemporaryDirectory(prefix="v38_49_authoritative_prediction_") as td:
            root = Path(td)
            dataset = root / "v38_48_schema_v2.npz"
            checkpoint = root / "synthetic_random_checkpoint.pth"
            report_path = root / "report.json"
            _make_data(dataset)
            model = module.DeepDenseFlightExecutiveNet(in_features=module.OBSERVATION_DIM)
            model._environment_observation_transition_trained_state.fill_(True)
            model._physics_uncertainty_base_trained_state.fill_(True)
            torch.save(model.state_dict(), checkpoint)
            os.environ.update({
                "SURVIVAL_AUTH_CARGO_WINCH_MODEL_DATASET": str(dataset),
                "SURVIVAL_AUTH_CARGO_WINCH_MODEL_CHECKPOINT": str(checkpoint),
                "SURVIVAL_AUTH_CARGO_WINCH_MODEL_REPORT": str(report_path),
                "SURVIVAL_AUTH_CARGO_WINCH_MODEL_DEVICE": "cpu",
                "SURVIVAL_AUTH_CARGO_WINCH_MODEL_MAX_ROWS": "200",
                "SURVIVAL_AUTH_CARGO_WINCH_MODEL_BATCH_SIZE": "16",
            })
            result = module.run_authoritative_cargo_winch_prediction_validation()
            assert report_path.is_file()
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            assert result["live_control_authorized"] is False
            assert result["actuation_connected"] is False
            assert result["deployment_or_safety_feasible"] is False
            assert saved["provenance"]["dataset_sha256"] and saved["provenance"]["checkpoint_sha256"]
            assert saved["sample_counts"]["valid_finite_authoritative_rows"] == 40
            assert saved["sample_counts"]["active_loaded_authoritative_cargo_rows"] == 40
            assert saved["metrics"]["cargo_dynamics_vs_authoritative_pbd"]["learned_head"]["count"] == 40
            assert saved["metrics"]["cable_transition_vs_authoritative_winch"]["learned_head_cable_length"]["count"] == 40
            assert saved["event_label_audit"]["learned_event_head_available"] is False
            assert saved["event_label_audit"]["learned_event_classification_status"] == "NOT_AVAILABLE_NO_LEARNED_EVENT_HEAD"
            assert saved["gate_diagnostics"]["readiness_claim"] is False

            # A schema-v1 dataset is explicitly blocked; no implicit label backfill.
            legacy_dataset = root / "schema_v1.npz"
            with np.load(dataset, allow_pickle=False) as data:
                arrays = {k: np.asarray(data[k]) for k in data.files}
            md = json.loads(str(arrays["metadata_json"].item()))
            md["schema_version"] = 1
            arrays["metadata_json"] = np.asarray(json.dumps(md))
            np.savez_compressed(legacy_dataset, **arrays)
            os.environ["SURVIVAL_AUTH_CARGO_WINCH_MODEL_DATASET"] = str(legacy_dataset)
            blocked = module.run_authoritative_cargo_winch_prediction_validation()
            assert blocked["status"] == "BLOCKED_SCHEMA_VERSION", blocked
            assert blocked["live_control_authorized"] is False

            # An inconsistent release label is not scored as a valid event outcome.
            bad_event_dataset = root / "event_semantic_violation.npz"
            arrays["metadata_json"] = np.asarray(json.dumps({**md, "schema_version": 2}))
            e_release = module.AUTH_CARGO_WINCH_EVENT_NAMES.index("package_release_event")
            arrays["authoritative_cargo_winch_events"] = arrays["authoritative_cargo_winch_events"].copy()
            arrays["authoritative_cargo_winch_events"][0, 5, e_release] = 1.0
            np.savez_compressed(bad_event_dataset, **arrays)
            os.environ["SURVIVAL_AUTH_CARGO_WINCH_MODEL_DATASET"] = str(bad_event_dataset)
            blocked_events = module.run_authoritative_cargo_winch_prediction_validation()
            assert blocked_events["status"] == "COMPLETED_CONTINUOUS_DIAGNOSTIC_EVENT_LABELS_BLOCKED", blocked_events
            assert blocked_events["event_label_audit"]["semantic_violation_counts"]["release_event_without_payload_latch_edge"] == 1
            assert blocked_events["live_control_authorized"] is False
            print("[V38.49 SYNTHETIC AUTHORITATIVE PREDICTION TEST PASS] schema-v2 scoring, hash provenance, PBD-derived targets, cable/tension baselines, uncertainty reporting, schema-v1 fail-closed, inconsistent-event blocking, and no-control boundary.")
            print("[LIMIT] Random weights and synthetic state validate code/data plumbing only; they prove no trained-model prediction quality, delivery outcome or physical safety.")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    main()
