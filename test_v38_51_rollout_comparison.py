#!/usr/bin/env python3
"""V38.51 seeded rollout comparison, provenance and fail-closed tests."""
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np

import Survival_Core_V38_51_Seeded_Rollout_Comparison as sc


def _episode(i, success, sway, clearance, trigger_count, lane="policy_predicted_winch"):
    return {
        "environment_id": i % 5,
        "drone_id": i % 3,
        "episode_index_within_environment_drone": i,
        "start_step": i * 100,
        "end_step": i * 100 + 99,
        "observed_steps": 100,
        "started_from_takeoff_phase_zero": True,
        "ended_by_depot_service_reset": True,
        "right_censored_at_rollout_end": False,
        "winch_supervision_lane": lane,
        "package_release_count": int(success),
        "fully_retracted_delivery_count": int(success),
        "full_mission_cycle_completed_from_phase_zero": bool(success),
        "peak_horizontal_cargo_sway_m": float(sway),
        "minimum_valid_cargo_surface_clearance_m": float(clearance),
        "valid_clearance_sample_count": 1,
        "failure_trigger_counts": {"alignment_or_cargo_zone_abort": int(trigger_count), "delivery_timeout": 0},
    }


def _make_report(checkpoint_path, rows, seed=12345, n_envs=16, steps=10000, teacher_ratio=0.0,
                 backend="genesis_cpu"):
    return sc._phase3_episode_quality_summary(
        rows, n_envs, 3, steps, 0.01, teacher_ratio,
        checkpoint_path=str(checkpoint_path), simulator_backend=backend)


def run():
    with tempfile.TemporaryDirectory(prefix="v38_51_test_") as td:
        td = Path(td)
        os.environ["SURVIVAL_PHASE3_ROLLOUT_EVAL_ONLY"] = "1"
        os.environ["SURVIVAL_PHASE3_SEED"] = "12345"
        # Seed Python/NumPy/Torch repeatedly and compare the streams.
        sc._phase3_seed_rngs(12345)
        a_py, a_np = random.random(), float(np.random.random())
        a_torch = float(sc.torch.rand(1).item()) if sc.torch is not None else None
        sc._phase3_seed_rngs(12345)
        assert random.random() == a_py
        assert float(np.random.random()) == a_np
        if sc.torch is not None:
            assert float(sc.torch.rand(1).item()) == a_torch

        contract_a = sc._phase3_rollout_comparison_contract(16, 3, 10000, 0.01, 0.0, "genesis_cpu")
        contract_b = sc._phase3_rollout_comparison_contract(16, 3, 10000, 0.01, 0.0, "genesis_cpu")
        assert contract_a["scenario_fingerprint_sha256"] == contract_b["scenario_fingerprint_sha256"]
        os.environ["SURVIVAL_PHASE3_SEED"] = "12346"
        contract_c = sc._phase3_rollout_comparison_contract(16, 3, 10000, 0.01, 0.0, "genesis_cpu")
        assert contract_a["scenario_fingerprint_sha256"] != contract_c["scenario_fingerprint_sha256"]
        os.environ["SURVIVAL_PHASE3_SEED"] = "12345"

        base_ckpt = td / "base.pth"
        cand_ckpt = td / "candidate.pth"
        base_ckpt.write_bytes(b"base checkpoint bytes")
        cand_ckpt.write_bytes(b"candidate checkpoint bytes")
        base_rows = [_episode(i, i < 15, 1.8 + i * 0.001, 0.04 + i * 0.0001, 3) for i in range(25)]
        cand_rows = [_episode(i, i < 20, 1.2 + i * 0.001, 0.08 + i * 0.0001, 1) for i in range(25)]
        base = _make_report(base_ckpt, base_rows)
        cand = _make_report(cand_ckpt, cand_rows)
        base_path, cand_path = td / "base.json", td / "candidate.json"
        base_path.write_text(json.dumps(base), encoding="utf-8")
        cand_path.write_text(json.dumps(cand), encoding="utf-8")
        result = sc._v3851_compare_report_payloads([
            {"path": str(base_path), "payload": base},
            {"path": str(cand_path), "payload": cand},
        ], minimum_complete_cycles=20)
        assert result["status"] == "COMPLETED_DIAGNOSTIC_ONLY", result.get("issues")
        comparison = result["comparisons"][0]
        assert comparison["comparison_status"] == "DIRECTIONAL_METRICS_FAVOUR_CANDIDATE", comparison
        assert comparison["candidate_metrics"]["complete_phase_zero_mission_cycles"] == 25
        assert comparison["candidate_metrics"]["successful_complete_mission_cycles"] == 20
        assert result["control_authority"]["live_control_authorized"] is False

        # Exercise the actual CLI-mode writer, which hashes the input reports and atomically writes JSON.
        old_env = {key: os.environ.get(key) for key in (
            "SURVIVAL_PHASE3_COMPARISON_REPORTS", "SURVIVAL_PHASE3_COMPARISON_OUTPUT",
            "SURVIVAL_PHASE3_COMPARISON_MIN_CYCLES")}
        try:
            cli_output = td / "comparison_output.json"
            os.environ["SURVIVAL_PHASE3_COMPARISON_REPORTS"] = f"{base_path};{cand_path}"
            os.environ["SURVIVAL_PHASE3_COMPARISON_OUTPUT"] = str(cli_output)
            os.environ["SURVIVAL_PHASE3_COMPARISON_MIN_CYCLES"] = "20"
            cli_result = sc.run_phase3_rollout_comparison()
            assert cli_result["status"] == "COMPLETED_DIAGNOSTIC_ONLY", cli_result.get("issues")
            on_disk = json.loads(cli_output.read_text(encoding="utf-8"))
            assert on_disk["status"] == "COMPLETED_DIAGNOSTIC_ONLY"
            assert on_disk["input_reports"][0]["report_sha256"]
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        # Small sample: metrics remain descriptive, no directional winner.
        small = _v = dict(base)
        small["episode_segments"] = base_rows[:4]
        small_result = sc._v3851_compare_report_payloads([
            {"path": str(base_path), "payload": small},
            {"path": str(cand_path), "payload": cand},
        ], minimum_complete_cycles=20)
        assert small_result["status"] == "COMPLETED_DIAGNOSTIC_ONLY"
        assert small_result["comparisons"][0]["comparison_status"] == "INCONCLUSIVE_BELOW_MIN_COMPLETE_CYCLES"

        # Nonzero teacher ratio and lane mix must be rejected.
        mixed_rows = [_episode(0, True, 1.0, 0.1, 0, lane="teacher_forced_winch")]
        mixed = _make_report(td / "mixed.pth", mixed_rows, teacher_ratio=0.5)
        mixed_path = td / "mixed.json"
        mixed_path.write_text(json.dumps(mixed), encoding="utf-8")
        blocked = sc._v3851_compare_report_payloads([
            {"path": str(base_path), "payload": base},
            {"path": str(mixed_path), "payload": mixed},
        ])
        assert blocked["status"] == "BLOCKED_INCOMPARABLE_REPORTS"
        assert any("teacher_winch_ratio" in issue or "non-policy winch lane" in issue for issue in blocked["issues"])

        # A mismatched seed/fingerprint cannot be treated as a same-condition comparison.
        os.environ["SURVIVAL_PHASE3_SEED"] = "999"
        mismatched = _make_report(cand_ckpt, cand_rows)
        mismatch = sc._v3851_compare_report_payloads([
            {"path": str(base_path), "payload": base},
            {"path": str(cand_path), "payload": mismatched},
        ])
        assert mismatch["status"] == "BLOCKED_INCOMPARABLE_REPORTS"
        assert any("seed differs" in issue or "scenario fingerprint differs" in issue for issue in mismatch["issues"])

        # Malformed/missing scenario config must block instead of throwing an uncaught exception.
        malformed = dict(cand)
        malformed["comparison_contract"] = dict(cand["comparison_contract"])
        malformed["comparison_contract"]["scenario_config"] = None
        malformed_result = sc._v3851_compare_report_payloads([
            {"path": str(base_path), "payload": base},
            {"path": str(cand_path), "payload": malformed},
        ])
        assert malformed_result["status"] == "BLOCKED_INCOMPARABLE_REPORTS"
        assert any("scenario_config" in issue for issue in malformed_result["issues"])
        print("[V38.51 TEST PASS] seed streams, scenario fingerprint, policy-only report validation, outcome deltas, sample-size caution and fail-closed provenance checks")


if __name__ == "__main__":
    run()
