"""Short real simulator-worker integration smoke; not a model-quality test."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import numpy as np

SOURCE = Path(__file__).with_name("Survival_Core_V38_48_Authoritative_Cargo_Winch_Telemetry.py")
spec = importlib.util.spec_from_file_location("survival_v3848_harvest_smoke", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import source: {SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main() -> None:
    env_keys = ("SURVIVAL_AUTH_CARGO_WINCH_DATASET", "SURVIVAL_AUTH_CARGO_WINCH_AUDIT_REPORT")
    old_env = {k: os.environ.get(k) for k in env_keys}
    old_tempdir = tempfile.tempdir
    try:
        with tempfile.TemporaryDirectory(prefix="v3848_real_harvest_") as tmp:
            root = Path(tmp)
            tempfile.tempdir = str(root)
            os.environ["SURVIVAL_AUTH_CARGO_WINCH_DATASET"] = str(root / "heldout_v2.npz")
            os.environ["SURVIVAL_AUTH_CARGO_WINCH_AUDIT_REPORT"] = str(root / "audit.json")
            chunk_path = Path(module._worker_harvest_metropolitan((0, 300)))
            assert chunk_path.is_file(), chunk_path
            with np.load(chunk_path, allow_pickle=False) as chunk:
                rows = int(chunk["obs"].shape[0])
                frames = int(np.unique(chunk["auth_frame_ids"]).size)
                auth_valid = int(chunk["auth_events"][:, module.AUTH_CARGO_WINCH_EVENT_NAMES.index("telemetry_valid")].sum())
                assert chunk["auth_codes"].shape == (rows, 4), chunk["auth_codes"].shape
                assert chunk["auth_events"].shape[1] == 17, chunk["auth_events"].shape
            print(f"WORKER_CHUNK {chunk_path.stat().st_size}")
            print(f"ROWS {rows} FRAMES {frames} AUTH_VALID {auth_valid}")
            result = module._save_transition_rollout_sequence_artifact(
                [str(chunk_path)], heldout_worker_ids=[0],
                output_path=str(root / "heldout_v2.npz"), sequence_length=64, segments_per_drone=2)
            print("SEQUENCE_RESULT " + json.dumps(result, sort_keys=True))
            assert result.get("status") == "SAVED", result
            with np.load(root / "heldout_v2.npz", allow_pickle=False) as ds:
                meta = json.loads(str(ds["metadata_json"].item()))
                print(f"ARTIFACT_SCHEMA {meta['schema_version']} SHAPES {ds['authoritative_node_positions_m'].shape} {ds['authoritative_cargo_winch_events'].shape} {ds['authoritative_cargo_winch_codes'].shape}")
                assert meta["schema_version"] == 2
                assert ds["authoritative_node_positions_m"].shape[:2] == ds["observations"].shape[:2]
                assert ds["authoritative_cargo_winch_codes"].shape[-1] == 4
            report = module.run_authoritative_cargo_winch_dataset_audit()
            print("AUDIT_STATUS " + report["status"])
            assert report["status"] == "PASS_SCHEMA_ALIGNMENT_AND_EVENT_ENCODING", report
            assert report["live_control_authorized"] is False
            print("[REAL HARVEST SEQUENCE SMOKE PASS] Existing worker → authoritative telemetry → sequence writer → dataset audit.")
            print("[LIMIT] Integration/alignment only, not physical fidelity or trained-model performance.")
    finally:
        tempfile.tempdir = old_tempdir
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    main()
