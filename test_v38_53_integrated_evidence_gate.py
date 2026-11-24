#!/usr/bin/env python3
"""V38.53 integrated evidence bundle tests. Synthetic reports validate gate logic, not flight quality."""
import json
import os
import tempfile
from pathlib import Path

import Survival_Core_V38_53_Integrated_Validation_Evidence_Gate as sc


def _episode(i, success=True):
    return {
        "environment_id": i % 4, "drone_id": i % 3,
        "episode_index_within_environment_drone": i,
        "start_step": i * 100, "end_step": i * 100 + 99, "observed_steps": 100,
        "started_from_takeoff_phase_zero": True, "ended_by_depot_service_reset": True,
        "right_censored_at_rollout_end": False, "winch_supervision_lane": "policy_predicted_winch",
        "package_release_count": int(success), "fully_retracted_delivery_count": int(success),
        "full_mission_cycle_completed_from_phase_zero": bool(success),
        "delivery_attempts_armed": 1, "failure_trigger_counts": {},
        "peak_horizontal_cargo_sway_m": 0.7 + 0.001 * i,
        "minimum_valid_cargo_surface_clearance_m": 0.05,
        "valid_clearance_sample_count": 4,
    }


def _make_bundle(root: Path, mark_heads=True):
    checkpoint = root / "candidate.pth"
    model = sc.DeepDenseFlightExecutiveNet(in_features=sc.OBSERVATION_DIM)
    # Synthetic marker flags make the fixture compatible with the full architecture;
    # this is plumbing-only and does not represent trained quality.
    if mark_heads:
        for name in ("_force_balance_head_trained_state", "_rotational_dynamics_head_trained_state",
                     "_physics_uncertainty_base_trained_state", "_force_uncertainty_trained_state",
                     "_trajectory_forecast_head_trained_state", "_action_conditioned_transition_head_trained_state",
                     "_environment_observation_transition_trained_state"):
            getattr(model, name).fill_(1.0)
    sc.torch.save(model.state_dict(), checkpoint)
    checkpoint_sha = sc._learned_mpc_sha256(str(checkpoint))
    sc._load_model_checkpoint(model, str(checkpoint), sc.torch.device("cpu"))
    state_sha = sc._v3853_module_state_sha256(model)

    metrics = {}
    for name in ("complete_cycle_success_fraction", "aggregate_retracted_to_released_count_ratio",
                 "failure_trigger_rate_per_1000_observed_steps", "p95_episode_peak_horizontal_sway_m",
                 "minimum_valid_cargo_surface_clearance_m"):
        metrics[name] = {
            "sample_count": 6,
            "n_independent_seed_pairs_with_metric_and_cycle_coverage": 6,
            "directional_status": "NO_CLEAR_DIRECTIONAL_EVIDENCE",
            "ci95_percentile_for_mean": {"lower": -0.01, "upper": 0.01},
        }
    authoritative_dataset_sha = "a" * 64
    reports = {
        "multiseed_outcomes": {
            "status": "COMPLETED_DIAGNOSTIC_ONLY_NO_CLEAR_DIRECTIONAL_EVIDENCE",
            "candidate_checkpoint_hashes": [checkpoint_sha], "eligible_seed_pair_count": 6,
            "metric_aggregates": metrics,
            "control_authority": {"live_control_authorized": False},
        },
        "authoritative_shadow": {
            "status": "COMPLETED", "enabled": True, "requested": True,
            "total_valid_samples": 2000, "checkpoint_sha256": checkpoint_sha,
            "model_state_sha256": state_sha, "actuation_modified_by_evaluator": False,
            "windows": [{"valid_samples": 2000, "gain_vs_constant_velocity_baseline_pct": {
                "position_delta_rmse_m": 2.0, "velocity_delta_rmse_mps": 1.0,
                "angular_rate_delta_rmse_rads": 1.0}}],
        },
        "transition_mpc": {
            "status": "COMPLETED_DIAGNOSTIC_ONLY", "current_checkpoint_sha256": checkpoint_sha,
            "current_checkpoint_one_step_quality": {"status": "COMPLETED", "count": 2000,
                "gain_vs_observation_only_baseline_pct": {"position_delta_rmse_m": 2.0}},
            "live_control_authorized": False,
        },
        "force_balance": {
            "status": "COMPLETED_WITH_HOLDOUT_COMPARISON",
            "holdout_prediction_comparison": {"matched_finite_prediction_rows": 2000,
                "metrics": {"gain_vs_zero_correction_baseline_pct": 2.0}},
        },
        "uncertainty_calibration": {
            "status": "COMPLETED_DIAGNOSTIC_ONLY",
            "task_metrics": {"kinematics": {"count": 2000, "coverage_1sigma": 0.68}},
        },
        "cargo_winch_validation": {
            "status": "COMPLETED_DIAGNOSTIC_ONLY", "provenance": {
                "checkpoint_sha256": checkpoint_sha, "dataset_sha256": authoritative_dataset_sha},
            "sample_counts": {"valid_authoritative_rows": 2000},
            "live_control_authorized": False,
        },
        "phase3_rollout_quality": sc._phase3_episode_quality_summary(
            [_episode(i, True) for i in range(25)], 16, 3, 10000, 0.01, 0.0,
            checkpoint_path=str(checkpoint), simulator_backend="genesis_cpu"),
        "authoritative_episode_quality": {
            "status": "COMPLETED_DIAGNOSTIC_ONLY", "valid_authoritative_row_count": 3000,
            "delivery_outcomes": {"ordered_release_then_completion_then_full_retraction_fragments": 20},
            "cargo_and_clearance": {"valid_clearance_row_count": 1000},
        },
        "telemetry_dataset_audit": {
            "status": "PASS_SCHEMA_ALIGNMENT_AND_EVENT_ENCODING", "metadata": {"schema_version": 2},
            "provenance": {"dataset_sha256": authoritative_dataset_sha},
        },
    }
    evidence = {}
    for role, payload in reports.items():
        path = root / f"{role}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        evidence[role] = path.name
    manifest = {"schema_version": 1, "candidate_checkpoint": checkpoint.name, "evidence": evidence}
    manifest_path = root / "evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, manifest, reports, checkpoint_sha


def run():
    with tempfile.TemporaryDirectory(prefix="v38_53_evidence_gate_") as temp:
        root = Path(temp)
        manifest_path, manifest, reports, checkpoint_sha = _make_bundle(root)
        result = sc._v3853_evidence_bundle_payload(manifest, str(manifest_path),
            min_seed_pairs=5, min_complete_cycles=20, min_shadow_samples=1000,
            min_transition_samples=1000, min_cargo_rows=1000)
        assert result["status"] == "EVIDENCE_BUNDLE_COMPLETE_DIAGNOSTIC_ONLY", result
        assert result["evidence_complete"] is True
        assert result["candidate_checkpoint_sha256"] == checkpoint_sha
        assert result["cross_report_bindings"]["cargo_winch_prediction_to_telemetry_dataset_audit"]["status"] == "DATASET_SHA256_MATCH"
        assert result["decision"]["live_control_authorized"] is False
        assert result["decision"]["checkpoint_promotion_authorized"] is False
        assert all(result["candidate_checkpoint_head_status"].values())

        # A loadable but untrained/random architecture must not be accepted as the candidate.
        with tempfile.TemporaryDirectory(prefix="v38_53_untrained_checkpoint_") as nested:
            sub = Path(nested)
            mp_untrained, m_untrained, _, _ = _make_bundle(sub, mark_heads=False)
            untrained_result = sc._v3853_evidence_bundle_payload(m_untrained, str(mp_untrained), min_seed_pairs=5,
                min_complete_cycles=20, min_shadow_samples=1000, min_transition_samples=1000,
                min_cargo_rows=1000)
            assert untrained_result["status"] == "BLOCKED_INCOMPLETE_OR_INVALID_EVIDENCE"
            assert any("lacks trained readiness marker" in issue for issue in untrained_result["issues"])

        # Exact checkpoint provenance mismatch must block.
        bad = json.loads(json.dumps(manifest))
        shadow_path = root / bad["evidence"]["authoritative_shadow"]
        shadow = json.loads(shadow_path.read_text())
        shadow["checkpoint_sha256"] = "0" * 64
        shadow_path.write_text(json.dumps(shadow), encoding="utf-8")
        blocked = sc._v3853_evidence_bundle_payload(bad, str(manifest_path), min_seed_pairs=5,
            min_complete_cycles=20, min_shadow_samples=1000, min_transition_samples=1000,
            min_cargo_rows=1000)
        assert blocked["status"] == "BLOCKED_INCOMPLETE_OR_INVALID_EVIDENCE"
        assert any("CHECKPOINT_SHA256_MISMATCH" in str(x) or "not bound to candidate" in str(x)
                   for x in blocked["issues"])

        # Policy lane / complete-cycle and unsafe clearance issues should be visible,
        # not silently converted into a successful evidence bundle.
        # Rebuild in an isolated nested temp directory for lane and quality corruption.
        with tempfile.TemporaryDirectory(prefix="v38_53_bad_lane_") as nested:
            sub = Path(nested)
            mp, m, report_map, _ = _make_bundle(sub)
            rollout_path = sub / m["evidence"]["phase3_rollout_quality"]
            rollout = json.loads(rollout_path.read_text())
            rollout["rollout"]["requested_teacher_winch_ratio"] = 0.5
            rollout_path.write_text(json.dumps(rollout), encoding="utf-8")
            r = sc._v3853_evidence_bundle_payload(m, str(mp), min_seed_pairs=5,
                min_complete_cycles=20, min_shadow_samples=1000, min_transition_samples=1000,
                min_cargo_rows=1000)
            assert r["status"] == "BLOCKED_INCOMPLETE_OR_INVALID_EVIDENCE"
            assert any("requested_teacher_winch_ratio=0" in x for x in r["issues"])

        # Adverse model-vs-baseline evidence keeps bundle diagnostic-only with explicit red flag.
        with tempfile.TemporaryDirectory(prefix="v38_53_force_regression_") as nested:
            sub = Path(nested)
            mp, m, report_map, _ = _make_bundle(sub)
            force_path = sub / m["evidence"]["force_balance"]
            force = json.loads(force_path.read_text())
            force["holdout_prediction_comparison"]["metrics"]["gain_vs_zero_correction_baseline_pct"] = -1.1
            force_path.write_text(json.dumps(force), encoding="utf-8")
            r = sc._v3853_evidence_bundle_payload(m, str(mp), min_seed_pairs=5,
                min_complete_cycles=20, min_shadow_samples=1000, min_transition_samples=1000,
                min_cargo_rows=1000)
            assert r["status"] == "EVIDENCE_COMPLETE_WITH_RED_FLAGS_DIAGNOSTIC_ONLY"
            assert any("force-balance correction regresses" in x for x in r["red_flags"])
            assert r["decision"]["live_control_authorized"] is False

        # Even individually successful model and schema-audit reports must refer to identical dataset bytes.
        with tempfile.TemporaryDirectory(prefix="v38_53_dataset_sha_mismatch_") as nested:
            sub = Path(nested)
            mp, m, _, _ = _make_bundle(sub)
            audit_path = sub / m["evidence"]["telemetry_dataset_audit"]
            audit = json.loads(audit_path.read_text())
            audit["provenance"]["dataset_sha256"] = "b" * 64
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            r = sc._v3853_evidence_bundle_payload(m, str(mp), min_seed_pairs=5,
                min_complete_cycles=20, min_shadow_samples=1000, min_transition_samples=1000,
                min_cargo_rows=1000)
            assert r["status"] == "BLOCKED_INCOMPLETE_OR_INVALID_EVIDENCE", r
            assert r["cross_report_bindings"]["cargo_winch_prediction_to_telemetry_dataset_audit"]["status"] == "DATASET_SHA256_MISMATCH"
            assert any("different dataset SHA-256" in x for x in r["issues"])

        # Missing roles, duplicate paths and incompatible manifest all fail closed.
        invalid = {"schema_version": 1, "candidate_checkpoint": "candidate.pth", "evidence": {}}
        r = sc._v3853_evidence_bundle_payload(invalid, str(manifest_path))
        assert r["status"] == "BLOCKED_INCOMPLETE_OR_INVALID_EVIDENCE"
        r = sc._v3853_evidence_bundle_payload({"schema_version": 2}, str(manifest_path))
        assert r["status"] == "BLOCKED_INVALID_MANIFEST"

        # CLI runner writes a report for a malformed manifest and exits safely, no live authority.
        env_manifest = root / "bad_manifest.json"
        env_manifest.write_text('{"schema_version": 9}', encoding="utf-8")
        output = root / "gate_output.json"
        old_m, old_o = os.environ.get("SURVIVAL_PHASE3_EVIDENCE_MANIFEST"), os.environ.get("SURVIVAL_PHASE3_EVIDENCE_OUTPUT")
        try:
            os.environ["SURVIVAL_PHASE3_EVIDENCE_MANIFEST"] = str(env_manifest)
            os.environ["SURVIVAL_PHASE3_EVIDENCE_OUTPUT"] = str(output)
            cli = sc.run_v3853_integrated_validation_evidence_gate()
            assert cli["status"] == "BLOCKED_INVALID_MANIFEST"
            assert output.is_file()
            assert json.loads(output.read_text())["decision"]["live_control_authorized"] is False
        finally:
            if old_m is None: os.environ.pop("SURVIVAL_PHASE3_EVIDENCE_MANIFEST", None)
            else: os.environ["SURVIVAL_PHASE3_EVIDENCE_MANIFEST"] = old_m
            if old_o is None: os.environ.pop("SURVIVAL_PHASE3_EVIDENCE_OUTPUT", None)
            else: os.environ["SURVIVAL_PHASE3_EVIDENCE_OUTPUT"] = old_o
    print("[V38.53 EVIDENCE GATE PASS] provenance, schema, dataset hash binding, status, sample coverage, policy-only cycle contract, red flags, fail-closed CLI; live control never authorized")


if __name__ == "__main__":
    run()
