"""Real simulator → schema-v2 sequences → V38.49 prediction-evaluation plumbing smoke.
The checkpoint is deliberately random; this tests integration, not learned quality.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch

SOURCE = Path(__file__).with_name("Survival_Core_V38_49_Authoritative_Cargo_Winch_Prediction_Validation.py")
spec = importlib.util.spec_from_file_location("survival_v3849_real_pred_smoke", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import source: {SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main() -> None:
    env_keys = (
        "SURVIVAL_AUTH_CARGO_WINCH_DATASET", "SURVIVAL_AUTH_CARGO_WINCH_AUDIT_REPORT",
        "SURVIVAL_AUTH_CARGO_WINCH_MODEL_DATASET", "SURVIVAL_AUTH_CARGO_WINCH_MODEL_CHECKPOINT",
        "SURVIVAL_AUTH_CARGO_WINCH_MODEL_REPORT", "SURVIVAL_AUTH_CARGO_WINCH_MODEL_DEVICE",
        "SURVIVAL_AUTH_CARGO_WINCH_MODEL_MAX_ROWS", "SURVIVAL_AUTH_CARGO_WINCH_MODEL_BATCH_SIZE",
    )
    old_env = {k: os.environ.get(k) for k in env_keys}
    old_tempdir = tempfile.tempdir
    try:
        with tempfile.TemporaryDirectory(prefix="v3849_real_harvest_prediction_") as tmp:
            root = Path(tmp)
            tempfile.tempdir = str(root)
            dataset = root / "heldout_v2.npz"
            checkpoint = root / "random_checkpoint.pth"
            audit_path = root / "audit.json"
            report_path = root / "prediction_report.json"
            os.environ.update({
                "SURVIVAL_AUTH_CARGO_WINCH_DATASET": str(dataset),
                "SURVIVAL_AUTH_CARGO_WINCH_AUDIT_REPORT": str(audit_path),
                "SURVIVAL_AUTH_CARGO_WINCH_MODEL_DATASET": str(dataset),
                "SURVIVAL_AUTH_CARGO_WINCH_MODEL_CHECKPOINT": str(checkpoint),
                "SURVIVAL_AUTH_CARGO_WINCH_MODEL_REPORT": str(report_path),
                "SURVIVAL_AUTH_CARGO_WINCH_MODEL_DEVICE": "cpu",
                "SURVIVAL_AUTH_CARGO_WINCH_MODEL_MAX_ROWS": "500",
                "SURVIVAL_AUTH_CARGO_WINCH_MODEL_BATCH_SIZE": "16",
            })
            chunk_path = Path(module._worker_harvest_metropolitan((0, 300)))
            assert chunk_path.is_file()
            with np.load(chunk_path, allow_pickle=False) as chunk:
                rows = int(chunk["obs"].shape[0])
                auth_valid = int(chunk["auth_events"][:, module.AUTH_CARGO_WINCH_EVENT_NAMES.index("telemetry_valid")].sum())
            assert rows > 0 and auth_valid > 0
            result = module._save_transition_rollout_sequence_artifact(
                [str(chunk_path)], heldout_worker_ids=[0], output_path=str(dataset),
                sequence_length=64, segments_per_drone=2)
            assert result.get("status") == "SAVED", result
            audit = module.run_authoritative_cargo_winch_dataset_audit()
            assert audit["status"] == "PASS_SCHEMA_ALIGNMENT_AND_EVENT_ENCODING", audit
            model = module.DeepDenseFlightExecutiveNet(in_features=module.OBSERVATION_DIM)
            model._environment_observation_transition_trained_state.fill_(True)
            model._physics_uncertainty_base_trained_state.fill_(True)
            torch.save(model.state_dict(), checkpoint)
            report = module.run_authoritative_cargo_winch_prediction_validation()
            assert report_path.is_file()
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            assert saved["status"].startswith("COMPLETED_"), saved
            assert saved["sample_counts"]["valid_finite_authoritative_rows"] > 0
            assert saved["provenance"]["dataset_sha256"] and saved["provenance"]["checkpoint_sha256"]
            assert saved["live_control_authorized"] is False
            assert saved["event_label_audit"]["learned_event_head_available"] is False
            print(f"[V38.49 REAL HARVEST/PREDICTION SMOKE PASS] rows={rows} schema2_sequences={result['sequence_count']} scored={saved['sample_counts']['prediction_rows_scored_after_cap']} status={saved['status']}")
            print("[LIMIT] Actual simulator PBD telemetry and real dataset writer are exercised, but checkpoint weights are random; no learned-model quality or safety claim.")
    finally:
        tempfile.tempdir = old_tempdir
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    main()
