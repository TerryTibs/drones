#!/usr/bin/env python3
"""V38.52 multi-seed aggregation, provenance, confidence and fail-closed tests."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import Survival_Core_V38_52_Multiseed_Outcome_Aggregation as sc


def _episode(i, success, sway, clearance, triggers, lane="policy_predicted_winch"):
    return {
        "environment_id": i % 4,
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
        "minimum_valid_cargo_surface_clearance_m": (float(clearance) if clearance is not None else None),
        "valid_clearance_sample_count": 1,
        "failure_trigger_counts": {"delivery_timeout": int(triggers), "alignment_or_cargo_zone_abort": 0},
    }


def _write_report(path, checkpoint, seed, success_count, sway, clearance, triggers,
                  n_cycles=25, n_envs=16, steps=10000, teacher_ratio=0.0, lane="policy_predicted_winch"):
    os.environ["SURVIVAL_PHASE3_SEED"] = str(seed)
    os.environ["SURVIVAL_PHASE3_ROLLOUT_EVAL_ONLY"] = "1"
    rows = [_episode(i, i < success_count, sway + i * .001,
                     (None if clearance is None else clearance + i * .0001),
                     triggers, lane=lane) for i in range(n_cycles)]
    report = sc._phase3_episode_quality_summary(
        rows, n_envs, 3, steps, .01, teacher_ratio,
        checkpoint_path=str(checkpoint), simulator_backend="genesis_cpu")
    path.write_text(json.dumps(report), encoding="utf-8")
    return report


def _build_manifest(root, seeds=(41001, 41002, 41003, 41004, 41005, 41006), config_mismatch_seed=None,
                    low_cycles=False, teacher_forced=False):
    baseline_ckpt = root / "baseline.pth"
    candidate_ckpt = root / "candidate.pth"
    baseline_ckpt.write_bytes(b"same baseline checkpoint across all seeds")
    candidate_ckpt.write_bytes(b"same candidate checkpoint across all seeds")
    pairs = []
    baseline_successes = [12, 13, 11, 14, 12, 13, 12, 13]
    candidate_successes = [19, 20, 18, 21, 19, 20, 19, 20]
    for i, seed in enumerate(seeds):
        steps = 12000 if seed == config_mismatch_seed else 10000
        cycles = 8 if low_cycles else 25
        lane = "teacher_forced_winch" if teacher_forced else "policy_predicted_winch"
        bname, cname = f"base_{seed}.json", f"cand_{seed}.json"
        _write_report(root / bname, baseline_ckpt, seed, min(baseline_successes[i], cycles),
                      2.0, 0.02, 3, n_cycles=cycles, steps=steps)
        _write_report(root / cname, candidate_ckpt, seed, min(candidate_successes[i], cycles),
                      1.1, 0.08, 1, n_cycles=cycles, steps=steps,
                      teacher_ratio=0.5 if teacher_forced else 0.0, lane=lane)
        pairs.append({"seed": seed, "baseline_report": bname, "candidate_report": cname})
    manifest = {"schema_version": 1, "pairs": pairs}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, manifest


def run():
    with tempfile.TemporaryDirectory(prefix="v38_52_multiseed_") as temp:
        root = Path(temp)
        old = {k: os.environ.get(k) for k in (
            "SURVIVAL_PHASE3_SEED", "SURVIVAL_PHASE3_ROLLOUT_EVAL_ONLY",
            "SURVIVAL_PHASE3_MULTISEED_MANIFEST", "SURVIVAL_PHASE3_MULTISEED_OUTPUT",
            "SURVIVAL_PHASE3_MULTISEED_MIN_SEEDS", "SURVIVAL_PHASE3_MULTISEED_MIN_CYCLES",
            "SURVIVAL_PHASE3_MULTISEED_BOOTSTRAP_REPLICATES", "SURVIVAL_PHASE3_MULTISEED_BOOTSTRAP_SEED")}
        try:
            manifest_path, manifest = _build_manifest(root)
            result = sc._v3852_aggregate_payload(manifest, str(manifest_path), min_seeds=5,
                min_complete_cycles=20, bootstrap_replicates=1000, bootstrap_seed=1776)
            assert result["status"] == "COMPLETED_DIAGNOSTIC_ONLY_DIRECTIONAL_EVIDENCE_ON_SOME_METRICS", result
            assert result["valid_seed_pair_count"] == 6
            assert result["eligible_seed_pair_count"] == 6
            success = result["metric_aggregates"]["complete_cycle_success_fraction"]
            assert success["n_independent_seed_pairs_with_metric_and_cycle_coverage"] == 6
            assert success["ci95_percentile_for_mean"]["lower"] > 0
            assert success["resampling_unit"] == "independent seed-pair delta"
            assert result["control_authority"]["live_control_authorized"] is False

            # CLI wrapper must write the output atomically and preserve inconclusive/block status.
            out = root / "multi_seed_output.json"
            os.environ["SURVIVAL_PHASE3_MULTISEED_MANIFEST"] = str(manifest_path)
            os.environ["SURVIVAL_PHASE3_MULTISEED_OUTPUT"] = str(out)
            os.environ["SURVIVAL_PHASE3_MULTISEED_MIN_SEEDS"] = "5"
            os.environ["SURVIVAL_PHASE3_MULTISEED_MIN_CYCLES"] = "20"
            os.environ["SURVIVAL_PHASE3_MULTISEED_BOOTSTRAP_REPLICATES"] = "1000"
            os.environ["SURVIVAL_PHASE3_MULTISEED_BOOTSTRAP_SEED"] = "1776"
            cli_result = sc.run_phase3_multiseed_aggregation()
            saved = json.loads(out.read_text(encoding="utf-8"))
            assert saved["status"] == cli_result["status"]
            assert saved["manifest_sha256"]

            # The actual script entry point must dispatch to multi-seed mode, not the GUI/training path.
            subprocess_env = os.environ.copy()
            subprocess_env["SURVIVAL_PHASE3_MULTISEED_AGGREGATE"] = "1"
            subprocess_env["SURVIVAL_PHASE3_MULTISEED_MANIFEST"] = str(manifest_path)
            subprocess_env["SURVIVAL_PHASE3_MULTISEED_OUTPUT"] = str(root / "entrypoint_output.json")
            subprocess_env["SURVIVAL_PHASE3_MULTISEED_MIN_SEEDS"] = "5"
            subprocess_env["SURVIVAL_PHASE3_MULTISEED_MIN_CYCLES"] = "20"
            subprocess_env["SURVIVAL_PHASE3_MULTISEED_BOOTSTRAP_REPLICATES"] = "1000"
            subprocess_env["SURVIVAL_PHASE3_MULTISEED_BOOTSTRAP_SEED"] = "1776"
            for flag in ("SURVIVAL_PHASE3_COMPARE_REPORTS", "SURVIVAL_SHADOW_POLICY_EVAL",
                         "SURVIVAL_TRANSITION_RECURSIVE_EVAL", "SURVIVAL_TRANSITION_ROLLOUT_EVAL",
                         "SURVIVAL_PHASE3_EPISODE_QUALITY_EVAL", "SURVIVAL_AUTH_CARGO_WINCH_MODEL_EVAL",
                         "SURVIVAL_AUTH_CARGO_WINCH_AUDIT", "SURVIVAL_CARGO_WINCH_EVAL",
                         "SURVIVAL_LEARNED_MPC_EVAL", "SURVIVAL_OFFLINE_MPC_EVAL",
                         "SURVIVAL_ROUTE_SELF_TEST", "SURVIVAL_SELF_TEST"):
                subprocess_env[flag] = "0"
            subprocess_run = subprocess.run([sys.executable, str(Path(sc.__file__))],
                env=subprocess_env, capture_output=True, text=True, timeout=120)
            assert subprocess_run.returncode == 0, subprocess_run.stdout + "\n" + subprocess_run.stderr
            entrypoint_saved = json.loads((root / "entrypoint_output.json").read_text(encoding="utf-8"))
            assert entrypoint_saved["status"] == result["status"]

            # Too few independent seed pairs stays inconclusive even with large within-run episode counts.
            small_root = root / "small"
            small_root.mkdir()
            small_path, small_manifest = _build_manifest(small_root, seeds=(42001, 42002, 42003))
            small_result = sc._v3852_aggregate_payload(small_manifest, str(small_path), min_seeds=5,
                min_complete_cycles=20, bootstrap_replicates=1000, bootstrap_seed=9)
            assert small_result["status"] == "INCONCLUSIVE_INSUFFICIENT_INDEPENDENT_SEEDS_OR_CYCLE_COVERAGE"

            # Enough seeds but metric absence is a coverage limitation, not "no directional evidence".
            coverage_root = root / "metric_coverage"
            coverage_root.mkdir()
            coverage_ckpt_a = coverage_root / "base.pth"
            coverage_ckpt_b = coverage_root / "candidate.pth"
            coverage_ckpt_a.write_bytes(b"coverage base")
            coverage_ckpt_b.write_bytes(b"coverage candidate")
            coverage_pairs = []
            for seed in (42501, 42502, 42503, 42504, 42505):
                bpath, cpath = coverage_root / f"b{seed}.json", coverage_root / f"c{seed}.json"
                _write_report(bpath, coverage_ckpt_a, seed, 12, 2.0, None, 3, n_cycles=25)
                _write_report(cpath, coverage_ckpt_b, seed, 19, 1.1, None, 1, n_cycles=25)
                # Rewrite metric clearance to null in every row so there are no valid clearance values.
                for target in (bpath, cpath):
                    rpt = json.loads(target.read_text(encoding="utf-8"))
                    for ep in rpt["episode_segments"]:
                        ep["minimum_valid_cargo_surface_clearance_m"] = None
                        ep["valid_clearance_sample_count"] = 0
                    target.write_text(json.dumps(rpt), encoding="utf-8")
                coverage_pairs.append({"seed": seed, "baseline_report": bpath.name, "candidate_report": cpath.name})
            coverage_manifest = {"schema_version": 1, "pairs": coverage_pairs}
            coverage_manifest_path = coverage_root / "manifest.json"
            coverage_manifest_path.write_text(json.dumps(coverage_manifest), encoding="utf-8")
            coverage_result = sc._v3852_aggregate_payload(coverage_manifest, str(coverage_manifest_path),
                min_seeds=5, min_complete_cycles=20, bootstrap_replicates=1000, bootstrap_seed=8)
            clearance_summary = coverage_result["metric_aggregates"]["minimum_valid_cargo_surface_clearance_m"]
            assert clearance_summary["directional_status"] == "INCONCLUSIVE_INSUFFICIENT_METRIC_COVERAGE"
            assert clearance_summary["n_independent_seed_pairs_with_metric_and_cycle_coverage"] == 0

            # Too few complete cycles also excludes those seed-pairs from CI estimation.
            low_root = root / "low_cycles"
            low_root.mkdir()
            low_path, low_manifest = _build_manifest(low_root, seeds=(43001, 43002, 43003, 43004, 43005), low_cycles=True)
            low_result = sc._v3852_aggregate_payload(low_manifest, str(low_path), min_seeds=5,
                min_complete_cycles=20, bootstrap_replicates=1000, bootstrap_seed=10)
            assert low_result["eligible_seed_pair_count"] == 0
            assert low_result["status"].startswith("INCONCLUSIVE_")

            # Scenario settings cannot vary across seed pairs.
            mismatch_root = root / "mismatch"
            mismatch_root.mkdir()
            mismatch_path, mismatch_manifest = _build_manifest(mismatch_root, config_mismatch_seed=41002)
            mismatch_result = sc._v3852_aggregate_payload(mismatch_manifest, str(mismatch_path), min_seeds=5,
                min_complete_cycles=20, bootstrap_replicates=1000, bootstrap_seed=11)
            assert mismatch_result["status"] == "BLOCKED_INCOMPARABLE_OR_INVALID_REPORTS"
            assert any("scenario configuration differs" in x for x in mismatch_result["issues"])

            # Teacher-forced intervention cannot enter a pure policy-predicted-winch aggregation.
            lane_root = root / "teacher"
            lane_root.mkdir()
            lane_path, lane_manifest = _build_manifest(lane_root, seeds=(44001, 44002), teacher_forced=True)
            lane_result = sc._v3852_aggregate_payload(lane_manifest, str(lane_path), min_seeds=2,
                min_complete_cycles=20, bootstrap_replicates=1000, bootstrap_seed=12)
            assert lane_result["status"] == "BLOCKED_INCOMPARABLE_OR_INVALID_REPORTS"

            # A report reused as both baseline and candidate is rejected explicitly.
            reused_manifest = {"schema_version": 1, "pairs": [{
                "seed": manifest["pairs"][0]["seed"],
                "baseline_report": manifest["pairs"][0]["baseline_report"],
                "candidate_report": manifest["pairs"][0]["baseline_report"]}]}
            reused_result = sc._v3852_aggregate_payload(reused_manifest, str(manifest_path), min_seeds=2,
                min_complete_cycles=1, bootstrap_replicates=1000, bootstrap_seed=13)
            assert any("report path reused" in x for x in reused_result["issues"])

            missing_schema = sc._v3852_aggregate_payload({"pairs": []}, str(manifest_path), min_seeds=2,
                min_complete_cycles=1, bootstrap_replicates=1000, bootstrap_seed=13)
            assert missing_schema["status"] == "BLOCKED_INVALID_MANIFEST"
            assert "schema_version" in missing_schema["issues"][0]

            # Duplicate seed and missing report are explicit manifest errors, not silently dropped.
            dup_manifest = {"schema_version": 1, "pairs": [manifest["pairs"][0], manifest["pairs"][0]]}
            dup_result = sc._v3852_aggregate_payload(dup_manifest, str(manifest_path), min_seeds=2,
                min_complete_cycles=1, bootstrap_replicates=1000, bootstrap_seed=13)
            assert any("duplicate seed" in x for x in dup_result["issues"])
            missing_manifest = {"schema_version": 1, "pairs": [{"seed": 999, "baseline_report": "absent.json", "candidate_report": "also_absent.json"}]}
            missing_result = sc._v3852_aggregate_payload(missing_manifest, str(manifest_path), min_seeds=2,
                min_complete_cycles=1, bootstrap_replicates=1000, bootstrap_seed=14)
            assert missing_result["status"] == "BLOCKED_INCOMPARABLE_OR_INVALID_REPORTS"

            print("[V38.52 TEST PASS] multi-seed pairing, per-seed bootstrap CI, cycle/seed thresholds, provenance/config consistency, report writer and fail-closed manifest handling")
        finally:
            for key, value in old.items():
                if value is None: os.environ.pop(key, None)
                else: os.environ[key] = value


if __name__ == "__main__":
    run()
