"""Short simulator worker -> full held-out artifact -> V38.50 event/trajectory report smoke."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import numpy as np

SOURCE = Path(__file__).with_name("Survival_Core_V38_50_Iterative_DAgger_Trajectory_Quality.py")
spec = importlib.util.spec_from_file_location("survival_v3850_real_episode_smoke", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import source: {SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main() -> None:
    env_keys = ("SURVIVAL_PHASE3_EPISODE_QUALITY_DATASET", "SURVIVAL_PHASE3_EPISODE_QUALITY_REPORT",
                "SURVIVAL_HARVEST_RUN_ID")
    previous = {k: os.environ.get(k) for k in env_keys}
    old_tempdir = tempfile.tempdir
    try:
        with tempfile.TemporaryDirectory(prefix="v3850_real_episode_quality_") as td:
            root = Path(td)
            tempfile.tempdir = td
            dataset = root / "heldout_full_worker.npz"
            report_path = root / "episode_quality.json"
            os.environ["SURVIVAL_PHASE3_EPISODE_QUALITY_DATASET"] = str(dataset)
            os.environ["SURVIVAL_PHASE3_EPISODE_QUALITY_REPORT"] = str(report_path)
            os.environ["SURVIVAL_HARVEST_RUN_ID"] = "v38.50-real-harvest-smoke"
            chunk_path = Path(module._worker_harvest_metropolitan((0, 300)))
            with np.load(chunk_path, allow_pickle=False) as chunk:
                rows = int(chunk["auth_frame_ids"].shape[0])
                assert rows > 0
            saved = module._phase3_save_authoritative_episode_quality_artifact(
                [str(chunk_path)], heldout_worker_ids=[0], output_path=str(dataset))
            assert saved["status"] == "SAVED", saved
            assert saved["row_count"] == rows
            report = module.run_authoritative_episode_quality_evaluation()
            assert report["status"] == "COMPLETED_DIAGNOSTIC_ONLY", report
            assert report["row_count"] == rows
            assert report["provenance"]["dataset_sha256"]
            assert report["control_authority"]["live_control_authorized"] is False
            assert report_path.is_file()
            stored = json.loads(report_path.read_text(encoding="utf-8"))
            assert stored["status"] == "COMPLETED_DIAGNOSTIC_ONLY"
            print(f"[V38.50 REAL HARVEST/EPISODE QUALITY SMOKE PASS] rows={rows} fragments={report['mission_fragment_count']} cycles={report['delivery_cycle_fragment_count']} valid={report['valid_authoritative_row_count']}")
            print("[LIMIT] This smoke uses a short teacher/harvest worker and whatever events occur in those 300 frames; it does not exercise Phase-3 Genesis training or establish delivery success rates.")
    finally:
        tempfile.tempdir = old_tempdir
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    main()
