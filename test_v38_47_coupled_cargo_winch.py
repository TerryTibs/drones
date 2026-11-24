"""Focused V38.47 tests. Synthetic values validate contracts only, never physics quality."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch

SOURCE = Path(__file__).with_name("Survival_Core_V38_47_Coupled_Cargo_Winch_Validation.py")
spec = importlib.util.spec_from_file_location("survival_v3847_test_module", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import V38.47 source: {SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _obs(module, payload=True, cable=1.0, cable_rate=0.0, tension=6.0, sway=(0.1, -0.05), rate=(0.0, 0.0)):
    obs = np.zeros(module.OBSERVATION_DIM, dtype=np.float32)
    frames = obs.reshape(module.HISTORY_LEN, module.BASE_OBS_DIM)
    for frame in frames:
        frame[9:18] = np.eye(3, dtype=np.float32).reshape(-1)
        frame[37] = 1.0 / 15.0
        frame[53] = cable / 15.0
        frame[54] = cable_rate / 5.0
        frame[55] = tension / 50.0
        frame[56:58] = np.asarray(sway, dtype=np.float32) / 2.0
        # Inverse of existing gyro-proxy mapping: x_rate ~= -gx? Decode uses
        # [-feature[89]*10/1.8, feature[88]*10/1.8].
        frame[89] = -float(rate[0]) * 1.8 / 10.0
        frame[88] = float(rate[1]) * 1.8 / 10.0
        frame[58] = 1.0 if payload else 0.0
        frame[3:6] = [0.0, 0.0, 12.0 / 200.0]
    return obs


def main() -> None:
    decoded = module._cargo_winch_observation_decode(_obs(module, cable=2.0, cable_rate=-0.2,
                                                           tension=12.0, sway=(0.2, -0.1), rate=(0.3, -0.4)))
    assert decoded["payload_active"] is True
    assert abs(decoded["cable_length_m"] - 2.0) < 1e-6
    assert abs(decoded["cable_rate_mps"] + 0.2) < 1e-6
    assert abs(decoded["tension_observed_n"] - 12.0) < 1e-6
    assert np.allclose(decoded["sway_xy_observed_m"], [0.2, -0.1])
    assert np.allclose(decoded["sway_rate_proxy_mps"], [0.3, -0.4])

    pred = np.asarray([[1.0, 2.0], [2.0, 3.0]])
    truth = pred.copy()
    summary = module._cargo_winch_metric_summary(pred, truth, ["x", "y"])
    assert summary["count"] == 2
    assert summary["channels"]["x"]["rmse"] == 0.0

    names = (
        "SURVIVAL_CARGO_WINCH_DATASET", "SURVIVAL_CARGO_WINCH_CHECKPOINT",
        "SURVIVAL_CARGO_WINCH_REPORT", "SURVIVAL_CARGO_WINCH_DEVICE",
        "SURVIVAL_CARGO_WINCH_MAX_ROWS", "SURVIVAL_CARGO_WINCH_BATCH_SIZE",
    )
    old_env = {key: os.environ.get(key) for key in names}
    try:
        torch.manual_seed(47)
        np.random.seed(47)
        with tempfile.TemporaryDirectory(prefix="v38_47_cargo_winch_smoke_") as td:
            root = Path(td)
            model = module.DeepDenseFlightExecutiveNet(in_features=module.OBSERVATION_DIM)
            model._environment_observation_transition_trained_state.fill_(True)
            model._physics_uncertainty_base_trained_state.fill_(True)
            checkpoint = root / "synthetic_random_checkpoint.pth"
            torch.save(model.state_dict(), checkpoint)

            nseq, steps = 2, 16
            observations = np.zeros((nseq, steps, module.OBSERVATION_DIM), dtype=np.float32)
            for s in range(nseq):
                for t in range(steps):
                    cable = 1.0 + (0.001 * t if s == 0 else -0.001 * t)
                    cable_rate = 0.0625 if s == 0 else -0.0625
                    obs = _obs(module, payload=True, cable=cable, cable_rate=cable_rate,
                               tension=6.0 + 0.05 * t, sway=(0.10 + 0.001 * t, -0.05),
                               rate=(0.01 * t, -0.005 * t))
                    observations[s, t] = obs
            actions = np.zeros((nseq, steps, module.ACTION_TRANSITION_ACTION_DIM), dtype=np.float32)
            actions[:, :, 3] = 1.0  # documented TRANSIT one-hot; actions are frozen observations, not replayed
            actions[0, :, -1] = 0.3
            actions[1, :, -1] = -0.3
            targets = np.zeros((nseq, steps, module.ACTION_TRANSITION_TARGET_DIM), dtype=np.float32)
            targets[:, :, -1] = 1.0
            metadata = {"dt_s": 0.016, "sequence_length": steps, "split": "synthetic-test"}
            dataset = root / "sequences.npz"
            np.savez_compressed(dataset, observations=observations, actions=actions, targets=targets,
                                metadata_json=np.asarray(json.dumps(metadata)),
                                worker_ids=np.asarray([11, 12], dtype=np.int16),
                                drone_ids=np.asarray([0, 1], dtype=np.int8),
                                start_frames=np.asarray([1, 20], dtype=np.int32))
            out = root / "report.json"
            os.environ.update({
                "SURVIVAL_CARGO_WINCH_DATASET": str(dataset),
                "SURVIVAL_CARGO_WINCH_CHECKPOINT": str(checkpoint),
                "SURVIVAL_CARGO_WINCH_REPORT": str(out),
                "SURVIVAL_CARGO_WINCH_DEVICE": "cpu",
                "SURVIVAL_CARGO_WINCH_MAX_ROWS": "1000",
                "SURVIVAL_CARGO_WINCH_BATCH_SIZE": "16",
            })
            result = module.run_coupled_cargo_winch_validation()
            assert out.is_file()
            saved = json.loads(out.read_text(encoding="utf-8"))
            assert result["live_control_authorized"] is False and result["actuation_connected"] is False
            assert saved["event_proxies"]["actual_full_retraction_or_delivery_completion_events_available"] is False
            assert saved["provenance"]["dataset_sha256"] and saved["provenance"]["checkpoint_sha256"]
            assert saved["sample_counts"]["active_loaded_cargo_pairs"] > 0
            assert saved["metrics"]["loaded_cargo_dynamics"]["head_prediction"]["count"] > 0
            assert saved["metrics"]["cargo_uncertainty_vs_error"]["count"] > 0
            assert saved["metrics"]["cable_and_winch_observation_audit"]["model_head_metrics"] is not None
            cmd_metrics = saved["metrics"]["winch_command_vs_observed_cable_change"]
            assert "scope" in cmd_metrics
            assert all(v["interpretation"].startswith("descriptive only")
                       for k, v in cmd_metrics.items() if k != "scope")
            print("[V38.47 SYNTHETIC EVALUATOR PASS] schema/provenance, adjacent observation alignment, cargo/cable model scoring, baseline metrics, uncertainty diagnostic and hard no-control boundary.")
            print("[LIMIT] Random synthetic checkpoint and synthetic observations validate plumbing only; no trained-model accuracy, authoritative tether correctness or physical safety is established.")
    finally:
        for key, val in old_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


if __name__ == "__main__":
    main()
