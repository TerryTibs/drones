"""End-to-end synthetic evaluator smoke test for V38.46.

Builds a temporary random checkpoint and tiny held-out sequence artifact. It checks
report creation, hashing/provenance, and hard no-control flags. It is NOT a trained
model test and must never be interpreted as flight-performance evidence.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch

SOURCE = Path(__file__).with_name("Survival_Core_V38_46_Learned_Transition_Offline_MPC.py")
spec = importlib.util.spec_from_file_location("survival_v3846_smoke_module", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import source for test: {SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main() -> None:
    torch.manual_seed(7)
    np.random.seed(7)
    names = (
        "SURVIVAL_ACTION_TRANSITION_REPORT", "SURVIVAL_TRANSITION_ROLLOUT_DATASET",
        "SURVIVAL_LEARNED_MPC_CHECKPOINT", "SURVIVAL_LEARNED_MPC_REPORT",
        "SURVIVAL_LEARNED_MPC_DEVICE", "SURVIVAL_LEARNED_MPC_SCENARIOS",
        "SURVIVAL_LEARNED_MPC_HORIZON_STEPS",
    )
    old_env = {name: os.environ.get(name) for name in names}
    try:
        with tempfile.TemporaryDirectory(prefix="v38_46_mpc_smoke_") as temp_dir:
            root = Path(temp_dir)
            model = module.DeepDenseFlightExecutiveNet(in_features=module.OBSERVATION_DIM)
            for name in (
                "_force_balance_head_trained_state", "_rotational_dynamics_head_trained_state",
                "_physics_uncertainty_base_trained_state", "_force_uncertainty_trained_state",
                "_trajectory_forecast_head_trained_state", "_action_conditioned_transition_head_trained_state",
                "_environment_observation_transition_trained_state",
            ):
                marker = getattr(model, name, None)
                if marker is not None:
                    marker.fill_(True)
            checkpoint = root / "synthetic_random_weights.pth"
            torch.save(model.state_dict(), checkpoint)

            action_report = {
                "version": "V38.45-uncertainty-aware-offline-safety-interface",
                "rollout_readiness_gate": {"status": "BLOCKED", "reasons": ["synthetic smoke-test readiness remains blocked"]},
                "rollout_sequence_artifact": {"path": "transition_rollout_sequences_v38_40.npz"},
                "action_input_distribution": {"training_teacher_actions": {
                    "fsm_action_counts": {"TRANSIT": 100},
                    "guidance_features": {"norm_max": 1.7, "norm_p95": 1.3},
                }},
            }
            report_path = root / "action_transition_validation_v38_42.json"
            report_path.write_text(json.dumps(action_report), encoding="utf-8")

            sequence_count, sequence_length = 1, 4
            observations = np.zeros((sequence_count, sequence_length, module.OBSERVATION_DIM), dtype=np.float32)
            for frame_idx in range(sequence_length):
                history = observations[0, frame_idx].reshape(module.HISTORY_LEN, module.BASE_OBS_DIM)
                for history_idx in range(module.HISTORY_LEN):
                    history[history_idx, 3:6] = [0.0, 0.0, 12.0 / 200.0]
                    history[history_idx, 9:18] = np.eye(3, dtype=np.float32).reshape(-1)
                history[-1, 58] = 0.0
            actions = np.zeros((sequence_count, sequence_length, module.ACTION_TRANSITION_ACTION_DIM), dtype=np.float32)
            actions[:, :, 3] = 1.0  # TRANSIT one-hot
            targets = np.zeros((sequence_count, sequence_length, module.ACTION_TRANSITION_TARGET_DIM), dtype=np.float32)
            targets[:, :, -1] = 1.0
            metadata = {"dt_s": 0.016, "sequence_length": sequence_length, "split": "synthetic-heldout"}
            dataset_path = root / "transition_rollout_sequences_v38_40.npz"
            np.savez_compressed(
                dataset_path,
                observations=observations,
                actions=actions,
                targets=targets,
                metadata_json=np.asarray(json.dumps(metadata)),
                worker_ids=np.asarray([777], dtype=np.int16),
                drone_ids=np.asarray([1], dtype=np.int8),
                start_frames=np.asarray([100], dtype=np.int32),
            )
            report_out = root / "learned_transition_mpc_v38_46.json"
            os.environ.update({
                "SURVIVAL_ACTION_TRANSITION_REPORT": str(report_path),
                "SURVIVAL_TRANSITION_ROLLOUT_DATASET": str(dataset_path),
                "SURVIVAL_LEARNED_MPC_CHECKPOINT": str(checkpoint),
                "SURVIVAL_LEARNED_MPC_REPORT": str(report_out),
                "SURVIVAL_LEARNED_MPC_DEVICE": "cpu",
                "SURVIVAL_LEARNED_MPC_SCENARIOS": "1",
                "SURVIVAL_LEARNED_MPC_HORIZON_STEPS": "2",
            })
            result = module.run_offline_learned_transition_mpc_evaluation()
            assert report_out.is_file()
            assert result["live_control_authorized"] is False
            assert result["selected_for_control"] is False
            assert result["all_candidates_rejected_for_safety_authority"] is True
            assert result["candidate_rollout_aggregate"]["safety_feasible_rollouts"] == 0
            assert result["provenance"]["checkpoint_sha256"]
            assert result["provenance"]["sequence_dataset_sha256"]
            assert result["model_quality_gate"]["checkpoint_binding_status"] == "UNVERIFIABLE_NO_HASH_IN_HISTORICAL_REPORT"
            assert result["model_quality_gate"]["current_checkpoint_one_step_quality"]["readiness_authorized"] is False

            # All-invalid target rows must yield a saved blocked report, not an exception
            # and not a relaxed mask that manufactures scenarios.
            targets[:, :, -1] = 0.0
            np.savez_compressed(
                dataset_path,
                observations=observations,
                actions=actions,
                targets=targets,
                metadata_json=np.asarray(json.dumps(metadata)),
                worker_ids=np.asarray([777], dtype=np.int16),
                drone_ids=np.asarray([1], dtype=np.int8),
                start_frames=np.asarray([100], dtype=np.int32),
            )
            blocked_out = root / "blocked_no_scenarios.json"
            os.environ["SURVIVAL_LEARNED_MPC_REPORT"] = str(blocked_out)
            blocked = module.run_offline_learned_transition_mpc_evaluation()
            assert blocked_out.is_file()
            assert blocked["status"] == "BLOCKED_NO_SCORABLE_SCENARIOS"
            assert blocked["live_control_authorized"] is False
            assert blocked["scenario_filter_counts"]["rejected_by_reason"]["nonfinite_or_invalid_target_window"] > 0
            print("[V38.46 SYNTHETIC EVALUATOR SMOKE PASS] completed and blocked paths write JSON; hashes/provenance recorded; all candidates remain blocked from safety/control authority.")
            print("[LIMIT] Random checkpoint and synthetic sequence test evaluator plumbing only; no trained model accuracy or physical safety established.")
    finally:
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    main()
