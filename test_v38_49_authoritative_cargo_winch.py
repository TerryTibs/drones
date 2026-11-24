"""Focused V38.49 tests. Synthetic checkpoint/data verify scorer plumbing only."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import Survival_Core_V38_49_Authoritative_Cargo_Winch_Scoring as v49


def _make_artifact(path: Path, nseq: int = 3, steps: int = 64) -> None:
    obs = np.zeros((nseq, steps, v49.OBSERVATION_DIM), dtype=np.float32)
    actions = np.zeros((nseq, steps, v49.ACTION_TRANSITION_ACTION_DIM), dtype=np.float32)
    actions[:, :, 3] = 1.0
    targets = np.zeros((nseq, steps, v49.ACTION_TRANSITION_TARGET_DIM), dtype=np.float32)
    targets[:, :, -1] = 1.0
    wb = np.zeros((nseq, steps, len(v49.WINCH_STATE_FIELD_NAMES)), dtype=np.float32)
    wa = np.zeros_like(wb)
    pb = np.zeros((nseq, steps, 5, 3), dtype=np.float32)
    pa = np.zeros_like(pb)
    vb = np.zeros_like(pb)
    va = np.zeros_like(pb)
    events = np.zeros((nseq, steps, len(v49.WINCH_EVENT_FIELD_NAMES)), dtype=np.uint8)
    release_pos = np.zeros((nseq, steps, 3), dtype=np.float32)
    frames = np.zeros((nseq, steps), dtype=np.int64)
    valid = np.ones((nseq, steps), dtype=np.uint8)
    idx = v49.WINCH_STATE_INDEX

    for s in range(nseq):
        for t in range(steps):
            block = obs[s, t].reshape(v49.HISTORY_LEN, v49.BASE_OBS_DIM)
            for h in range(v49.HISTORY_LEN):
                block[h, 9:18] = np.eye(3, dtype=np.float32).reshape(-1)
                block[h, 3:6] = [0.0, 0.0, 10.0 / 200.0]
                block[h, 6:9] = [0.0, 0.0, 0.0]
                block[h, 53] = 2.0 / 15.0
                block[h, 54] = 0.01 / 5.0
                block[h, 55] = 3.0 / 50.0
                block[h, 56:58] = [0.1 + 0.001 * t, -0.02]  # proxy scale is 2 m
                block[h, 58] = 1.0
            base_state = np.zeros(len(v49.WINCH_STATE_FIELD_NAMES), dtype=np.float32)
            base_state[idx["cable_length_m"]] = 2.0 + 0.001 * t
            base_state[idx["cable_rate_mps"]] = 0.01
            base_state[idx["tension_n"]] = 3.0 + 0.01 * t
            base_state[idx["payload_latched"]] = 1.0
            base_state[idx["winch_state_code"]] = v49.WINCH_STATE_CODES["LOWERING"]
            base_state[idx["target_surface_z_m"]] = 0.0
            base_state[idx["cargo_target_clearance_m"]] = 7.0
            base_state[idx["nodes_valid"]] = 1.0
            base_state[idx["clearance_valid"]] = 1.0
            wb[s, t] = base_state
            after = base_state.copy()
            after[idx["cable_length_m"]] += 0.0005
            after[idx["tension_n"]] += 0.005
            wa[s, t] = after
            frames[s, t] = (1000 if s in (0, 1) else 3000) + t

            # 5 nodes from anchor at the airframe underside to cargo endpoint.
            sway_x = 0.10 + 0.002 * t
            end_before = np.asarray([sway_x, -0.05, 7.75], dtype=np.float32)
            end_after = np.asarray([sway_x + 0.001, -0.049, 7.75], dtype=np.float32)
            anchor = np.asarray([0.0, 0.0, 9.95], dtype=np.float32)
            pb[s, t] = np.linspace(anchor, end_before, 5, dtype=np.float32)
            pa[s, t] = np.linspace(anchor, end_after, 5, dtype=np.float32)
            # Relative endpoint velocity is approximately [0.0625,0.0625] m/s.
            for node in range(5):
                vb[s, t, node] = [0.0, 0.0, -0.01]
                va[s, t, node] = [0.0, 0.0, -0.01]
                va[s, t, node, 0] = 0.0625 * node / 4
                va[s, t, node, 1] = 0.0625 * node / 4
            events[s, t] = v49._authoritative_tether_event_labels(wb[s, t], wa[s, t])

    metadata = {
        "schema_version": 2,
        "authoritative_tether_schema_version": 1,
        "sequence_length": steps,
        "action_condition_dim": v49.ACTION_TRANSITION_ACTION_DIM,
        "observation_dim": v49.OBSERVATION_DIM,
        "transition_target_dim": v49.ACTION_TRANSITION_TARGET_DIM,
        "dt_s": 0.016,
        "split": "synthetic-test-heldout-workers",
        "collector_source_sha256": "a" * 64,
        "worker_ids": [1, 3],
        "source_worker_count": 2,
        "source_worker_chunks": {"1": {"sha256": "b" * 64, "bytes": 1234, "mtime_ns": 123456},
                                 "3": {"sha256": "c" * 64, "bytes": 1234, "mtime_ns": 123457}},
        "authoritative_tether": {
            "node_count": 5,
            "state_fields": v49.WINCH_STATE_FIELD_NAMES,
            "event_fields": v49.WINCH_EVENT_FIELD_NAMES,
        },
    }
    np.savez_compressed(
        path, observations=obs, actions=actions, targets=targets,
        worker_ids=np.asarray([1, 1, 3][:nseq], dtype=np.int16),
        drone_ids=np.asarray([0, 0, 2][:nseq], dtype=np.int8),
        start_frames=frames[:, 0].astype(np.int32), valid_counts=np.full(nseq, steps, dtype=np.int16),
        winch_state_before=wb, winch_state_after=wa,
        tether_nodes_pos_before=pb, tether_nodes_pos_after=pa,
        tether_nodes_vel_before=vb, tether_nodes_vel_after=va,
        winch_event_labels=events, release_event_positions_after=release_pos,
        simulation_frame_indices=frames, authoritative_tether_valid=valid,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def main() -> None:
    vars_used = (
        "SURVIVAL_AUTHORITATIVE_CARGO_WINCH_DATASET",
        "SURVIVAL_AUTHORITATIVE_CARGO_WINCH_CHECKPOINT",
        "SURVIVAL_AUTHORITATIVE_CARGO_WINCH_REPORT",
        "SURVIVAL_AUTHORITATIVE_CARGO_WINCH_SCHEMA_AUDIT_REPORT",
        "SURVIVAL_AUTHORITATIVE_CARGO_WINCH_DEVICE",
        "SURVIVAL_AUTHORITATIVE_CARGO_WINCH_MAX_ROWS",
        "SURVIVAL_AUTHORITATIVE_CARGO_WINCH_BATCH_SIZE",
        "SURVIVAL_AUTHORITATIVE_TETHER_DATASET",
        "SURVIVAL_AUTHORITATIVE_TETHER_REPORT",
    )
    old = {k: os.environ.get(k) for k in vars_used}
    try:
        # Small deterministic thread count avoids pathological CPU oversubscription in CI.
        torch.set_num_threads(1)
        torch.manual_seed(4901)
        np.random.seed(4901)
        with tempfile.TemporaryDirectory(prefix="v38_49_authoritative_cargo_") as td:
            root = Path(td)
            dataset = root / "synthetic_v38_48.npz"
            _make_artifact(dataset)
            model = v49.DeepDenseFlightExecutiveNet(in_features=v49.OBSERVATION_DIM)
            model._environment_observation_transition_trained_state.fill_(True)
            model._physics_uncertainty_base_trained_state.fill_(True)
            checkpoint = root / "synthetic_random_checkpoint.pth"
            torch.save(model.state_dict(), checkpoint)
            report_path = root / "scores.json"
            audit_path = root / "schema_audit.json"
            os.environ.update({
                "SURVIVAL_AUTHORITATIVE_CARGO_WINCH_DATASET": str(dataset),
                "SURVIVAL_AUTHORITATIVE_CARGO_WINCH_CHECKPOINT": str(checkpoint),
                "SURVIVAL_AUTHORITATIVE_CARGO_WINCH_REPORT": str(report_path),
                "SURVIVAL_AUTHORITATIVE_CARGO_WINCH_SCHEMA_AUDIT_REPORT": str(audit_path),
                "SURVIVAL_AUTHORITATIVE_CARGO_WINCH_DEVICE": "cpu",
                "SURVIVAL_AUTHORITATIVE_CARGO_WINCH_MAX_ROWS": "1000",
                "SURVIVAL_AUTHORITATIVE_CARGO_WINCH_BATCH_SIZE": "16",
            })
            result = v49.run_authoritative_cargo_winch_prediction_scoring()
            assert report_path.exists() and audit_path.exists()
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            assert result["status"] == "COMPLETED_DIAGNOSTIC_ONLY", result.get("reason")
            assert saved["schema_audit"]["schema_integrity"] == "PASS"
            assert saved["schema_audit"]["frame_alignment"] == "PASS"
            assert saved["schema_audit"]["event_label_integrity"] == "PASS"
            assert saved["time_alignment_contract"].startswith("observation[t]")
            assert saved["sample_counts"]["unique_worker_drone_frame_rows_before_cap"] == 128
            assert saved["schema_audit"]["status"] == "SCHEMA_VALID_COVERAGE_INSUFFICIENT"
            assert saved["sample_counts"]["loaded_cargo_transition_rows_all_domains"] > 0
            assert saved["sample_counts"]["loaded_cargo_transition_rows_in_training_domain"] > 0
            cargo = saved["metrics"]["authoritative_loaded_cargo_head"]
            assert cargo["training_domain_loaded_only"]["learned_head_vs_authoritative"]["count"] > 0
            assert "authoritative_persistence_baseline_vs_authoritative" in cargo["training_domain_loaded_only"]
            assert saved["metrics"]["authoritative_cable_winch_head"]["learned_environment_head"]["count"] > 0
            assert saved["metrics"]["authoritative_cargo_uncertainty_calibration"]["count"] > 0
            assert "by_recorded_winch_command" in saved["metrics"]
            assert "by_authoritative_pause_abort_severed_state" in saved["metrics"]
            assert "by_authoritative_cargo_clearance_context" in saved["metrics"]
            assert saved["live_control_authorized"] is False
            assert saved["actuation_connected"] is False
            assert saved["deployment_or_safety_feasible"] is False
            assert saved["gate_diagnostics"]["control_authority"].startswith("NO CONTROL AUTHORITY")
            print("[V38.49 SYNTHETIC SCORER PASS] same-row authoritative PBD/winch targets, audit prerequisite, de-duplication, matched baselines, uncertainty diagnostics and hard no-control invariants.")
            print("[LIMIT] Random checkpoint and synthetic simulator data validate plumbing only; no model quality or physical safety result is established.")

            # Fail-closed schema/provenance check: delete the authoritative node arrays.
            bad = root / "bad_schema.npz"
            with np.load(dataset, allow_pickle=False) as src:
                arr = {k: np.asarray(src[k]) for k in src.files if k != "tether_nodes_pos_after"}
            np.savez_compressed(bad, **arr)
            os.environ["SURVIVAL_AUTHORITATIVE_CARGO_WINCH_DATASET"] = str(bad)
            blocked = v49.run_authoritative_cargo_winch_prediction_scoring()
            assert blocked["status"].startswith("BLOCKED")
            print("[V38.49 FAIL-CLOSED PASS] malformed/missing authoritative schema is blocked before scoring.")

            # A small artifact can pass individual serialization checks but must still
            # be blocked by the authoritative audit's minimum unique-row gate.
            small = root / "too_few_rows.npz"
            with np.load(dataset, allow_pickle=False) as src:
                small_meta = json.loads(str(np.asarray(src["metadata_json"]).item()))
                small_meta["sequence_length"] = 16
                small_meta["worker_ids"] = [1]
                small_meta["source_worker_count"] = 1
                small_meta["source_worker_chunks"] = {"1": small_meta["source_worker_chunks"]["1"]}
                small_arrays = {}
                temporal_keys = {
                    "observations", "actions", "targets", "winch_state_before", "winch_state_after",
                    "tether_nodes_pos_before", "tether_nodes_pos_after", "tether_nodes_vel_before",
                    "tether_nodes_vel_after", "winch_event_labels", "release_event_positions_after",
                    "simulation_frame_indices", "authoritative_tether_valid",
                }
                for key in src.files:
                    if key == "metadata_json":
                        continue
                    arr = np.asarray(src[key])
                    if key in temporal_keys:
                        small_arrays[key] = arr[:1, :16]
                    elif key in ("worker_ids", "drone_ids", "start_frames", "valid_counts"):
                        small_arrays[key] = arr[:1].copy()
                small_arrays["valid_counts"][:] = 16
                small_arrays["metadata_json"] = np.asarray(json.dumps(small_meta, sort_keys=True))
            np.savez_compressed(small, **small_arrays)
            os.environ["SURVIVAL_AUTHORITATIVE_CARGO_WINCH_DATASET"] = str(small)
            blocked_small = v49.run_authoritative_cargo_winch_prediction_scoring()
            assert blocked_small["status"] == "BLOCKED_AUTHORITATIVE_ARTIFACT_INTEGRITY", blocked_small.get("status")
            assert blocked_small["schema_audit"]["status"] == "BLOCKED_INSUFFICIENT_AUTHORITATIVE_ROWS"
            print("[V38.49 MINIMUM-COVERAGE BLOCK PASS] well-formed but under-64-row artifact remains blocked despite schema fields being valid.")
    finally:
        for key, val in old.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


if __name__ == "__main__":
    main()
