#!/usr/bin/env python3
"""V38.52 inherited Phase-3 seed/provenance metadata and report-output regression test."""
import json
import os
import tempfile
from pathlib import Path

import Survival_Core_V38_52_Multiseed_Outcome_Aggregation as sc


def main():
    keys = ("SURVIVAL_PHASE3_SEED", "SURVIVAL_PHASE3_ROLLOUT_QUALITY_REPORT", "SURVIVAL_PHASE3_BASE_CHECKPOINT")
    old = {k: os.environ.get(k) for k in keys}
    try:
        with tempfile.TemporaryDirectory(prefix="v3851_seed_metadata_") as td:
            root = Path(td)
            checkpoint = root / "input_checkpoint.pth"
            checkpoint.write_bytes(b"dummy checkpoint identity for report metadata")
            report_path = root / "rollout_report.json"
            os.environ["SURVIVAL_PHASE3_SEED"] = "987654321"
            os.environ["SURVIVAL_PHASE3_ROLLOUT_QUALITY_REPORT"] = str(report_path)
            os.environ["SURVIVAL_PHASE3_BASE_CHECKPOINT"] = str(checkpoint)
            rows = [{
                "environment_id": 0, "drone_id": 0, "episode_index_within_environment_drone": 0,
                "start_step": 0, "end_step": 9, "observed_steps": 10,
                "started_from_takeoff_phase_zero": True, "ended_by_depot_service_reset": True,
                "right_censored_at_rollout_end": False,
                "winch_supervision_lane": "policy_predicted_winch",
                "package_release_count": 1, "fully_retracted_delivery_count": 1,
                "full_mission_cycle_completed_from_phase_zero": True,
                "delivery_attempts_armed": 1, "failure_trigger_counts": {},
                "peak_horizontal_cargo_sway_m": 0.3,
                "minimum_valid_cargo_surface_clearance_m": 0.1,
                "valid_clearance_sample_count": 1,
            }]
            output = sc._phase3_write_episode_quality_report(rows, 2, 3, 10, 0.01, 0.0,
                                                               simulator_backend="genesis_cpu")
            report = json.loads(Path(output).read_text(encoding="utf-8"))
            contract = report["comparison_contract"]
            assert report["version"] == sc.SURVIVAL_CORE_VERSION
            assert contract["seed"] == 987654321
            assert contract["seed_present"] is True
            assert contract["policy_only_winch_lane_requested"] is True
            assert len(contract["scenario_fingerprint_sha256"]) == 64
            assert report["provenance"]["checkpoint_sha256"] == sc._learned_mpc_sha256(str(checkpoint))
            assert report["control_authority"]["live_control_authorized"] is False

            os.environ.pop("SURVIVAL_PHASE3_SEED", None)
            unseeded = sc._phase3_rollout_comparison_contract(2, 3, 10, 0.01, 0.0, "genesis_cpu")
            assert unseeded["seed"] is None and unseeded["seed_present"] is False

            # Evaluation-only worker must return without saving an output checkpoint.
            # Stub the expensive Genesis rollout: this tests the worker write contract, not physics.
            base_state_path = root / "worker_base.pth"
            output_state_path = root / "must_not_be_written.pth"
            worker_model = sc.DeepDenseFlightExecutiveNet(in_features=sc.OBSERVATION_DIM)
            sc.torch.save(worker_model.state_dict(), base_state_path)
            original_train = sc.train_phase3_genesis_parallel
            try:
                sc.train_phase3_genesis_parallel = lambda model, device: model
                os.environ["SURVIVAL_PHASE3_BASE_CHECKPOINT"] = str(base_state_path)
                os.environ["SURVIVAL_PHASE3_OUTPUT_CHECKPOINT"] = str(output_state_path)
                os.environ["SURVIVAL_PHASE3_ROLLOUT_EVAL_ONLY"] = "1"
                sc._phase3_worker_main()
                assert not output_state_path.exists(), "evaluation-only worker wrote an output checkpoint"
            finally:
                sc.train_phase3_genesis_parallel = original_train
                os.environ.pop("SURVIVAL_PHASE3_ROLLOUT_EVAL_ONLY", None)
        print("[V38.52 inherited SEED METADATA TEST PASS] report seed/config fingerprint, checkpoint provenance, policy-only lane, unseeded marker, and evaluation-only no-write contract")
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    main()
