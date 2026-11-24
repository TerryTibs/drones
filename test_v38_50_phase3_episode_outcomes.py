#!/usr/bin/env python3
"""Focused tests for V38.50 Phase-3 full-cycle/censoring and lane metrics."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile

SOURCE = Path(__file__).with_name("Survival_Core_V38_50_Iterative_DAgger_Trajectory_Quality.py")
spec = importlib.util.spec_from_file_location("survival_v3850_episode_outcomes", SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def row(start_phase, ended, release, retracted, lane, left=False):
    return {
        "environment_id": 0, "drone_id": 0,
        "episode_index_within_environment_drone": 0,
        "start_step": 0, "end_step": 9, "observed_steps": 10,
        "observed_duration_s": 0.9,
        "start_phase_code": start_phase,
        "started_from_takeoff_phase_zero": start_phase == 0,
        "left_censored_at_rollout_start": left,
        "ended_by_depot_service_reset": ended,
        "right_censored_at_rollout_end": not ended,
        "depot_service_state_reached": ended,
        "winch_supervision_lane": lane,
        "delivery_attempts_armed": 1,
        "package_release_count": release,
        "fully_retracted_delivery_count": retracted,
        "delivery_success_confirmed_in_segment": bool(release and retracted),
        "full_mission_cycle_completed_from_phase_zero": bool(start_phase == 0 and ended and release and retracted),
        "failure_trigger_counts": {"alignment_or_cargo_zone_abort": 3, "invalid_touchdown": 0, "delivery_timeout": 1},
        "failure_counter_semantics": "per-step triggers, not unique deliveries",
        "peak_horizontal_cargo_sway_m": 1.3,
        "maximum_cable_length_m": 7.0,
        "maximum_simulated_tension_n": 10.0,
        "maximum_horizontal_wind_mps": 4.0,
        "minimum_valid_cargo_surface_clearance_m": -0.02,
        "valid_clearance_sample_count": 4,
    }


def main():
    success = row(0, True, 1, 1, "teacher_forced_winch")
    censored = row(0, False, 1, 0, "policy_predicted_winch")
    left_censored = row(3, True, 1, 1, "policy_predicted_winch", left=True)
    report = mod._phase3_episode_quality_summary(
        [success, censored, left_censored], n_envs=1, n_drones=1, steps=10,
        dt_s=0.1, teacher_winch_ratio=0.5, checkpoint_path=None)
    counts = report["counts"]
    assert counts["episode_segments_recorded"] == 3
    assert counts["episodes_right_censored_at_rollout_end"] == 1
    # Only the phase-zero segment closed by depot reset enters full-cycle denominator.
    assert counts["full_mission_cycles_started_from_phase_zero_and_ended_at_depot_reset"] == 1
    assert counts["full_mission_cycles_with_confirmed_release_and_full_retraction"] == 1
    assert counts["full_mission_cycle_success_fraction"] == 1.0
    assert counts["confirmed_package_releases"] == 3
    assert counts["fully_retracted_deliveries"] == 2
    assert counts["failure_trigger_counts_by_cause"]["alignment_or_cargo_zone_abort"] == 9
    assert "not unique" in counts["failure_counter_semantics"]
    assert report["stratified_by_winch_supervision_lane"]["teacher_forced_winch"]["full_mission_cycles_successful"] == 1
    assert report["stratified_by_winch_supervision_lane"]["policy_predicted_winch"]["full_mission_cycles_successful"] == 0
    assert report["control_authority"]["live_control_authorized"] is False

    # The actual Phase-3 rollout report has its own path variable so it cannot
    # overwrite the separate full-worker artifact-evaluator report by accident.
    old_rollout = os.environ.get("SURVIVAL_PHASE3_ROLLOUT_QUALITY_REPORT")
    old_artifact = os.environ.get("SURVIVAL_PHASE3_EPISODE_QUALITY_REPORT")
    try:
        with tempfile.TemporaryDirectory(prefix="v3850_phase3_report_") as td:
            path = Path(td) / "rollout_quality.json"
            os.environ["SURVIVAL_PHASE3_ROLLOUT_QUALITY_REPORT"] = str(path)
            os.environ["SURVIVAL_PHASE3_EPISODE_QUALITY_REPORT"] = str(Path(td) / "artifact_eval_report.json")
            written = mod._phase3_write_episode_quality_report([success, censored, left_censored], 1, 1, 10, 0.1, 0.5)
            assert Path(written) == path and path.is_file()
            saved = json.loads(path.read_text(encoding="utf-8"))
            assert saved["counts"]["episode_segments_recorded"] == 3
    finally:
        if old_rollout is None:
            os.environ.pop("SURVIVAL_PHASE3_ROLLOUT_QUALITY_REPORT", None)
        else:
            os.environ["SURVIVAL_PHASE3_ROLLOUT_QUALITY_REPORT"] = old_rollout
        if old_artifact is None:
            os.environ.pop("SURVIVAL_PHASE3_EPISODE_QUALITY_REPORT", None)
        else:
            os.environ["SURVIVAL_PHASE3_EPISODE_QUALITY_REPORT"] = old_artifact
    print("[V38.50 PHASE-3 EPISODE OUTCOME TEST PASS] complete-cycle denominator, censored episodes, lane stratification, trigger-count semantics, and separate atomic report path")


if __name__ == "__main__":
    main()
