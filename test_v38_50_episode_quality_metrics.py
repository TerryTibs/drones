"""Synthetic authoritative event/trajectory metric tests for V38.50."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np

SOURCE = Path(__file__).with_name("Survival_Core_V38_50_Iterative_DAgger_Trajectory_Quality.py")
spec = importlib.util.spec_from_file_location("survival_v3850_episode_metrics_test", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import source: {SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_arrays(n=12):
    evn = len(module.AUTH_CARGO_WINCH_EVENT_NAMES)
    scn = len(module.AUTH_CARGO_WINCH_SCALAR_NAMES)
    node_pos = np.zeros((n, 5, 3), dtype=np.float32)
    node_vel = np.zeros_like(node_pos)
    scalar = np.zeros((n, scn), dtype=np.float32)
    events = np.zeros((n, evn), dtype=np.float32)
    codes = np.zeros((n, 4), dtype=np.int16)
    for name in ("telemetry_valid", "payload_present_before_step", "cargo_clearance_valid"):
        events[:, module.AUTH_CARGO_WINCH_EVENT_NAMES.index(name)] = 1.0
    # Endpoint offset ramps from 0.1 to 1.2 m; telemetry's drone anchor stays at origin.
    node_pos[:, -1, 0] = np.linspace(0.1, 1.2, n)
    scalar[:, module.AUTH_CARGO_WINCH_SCALAR_NAMES.index("cable_length_m")] = 2.0
    scalar[:, module.AUTH_CARGO_WINCH_SCALAR_NAMES.index("tension_n")] = 3.0
    clear_i = module.AUTH_CARGO_WINCH_SCALAR_NAMES.index("cargo_surface_clearance_m")
    scalar[:, clear_i] = np.array([0.7, 0.5, 0.2, 0.05, -0.02, -0.01, 0.1, 0.2, 0.3, 0.3, 0.4, 0.5])
    events[4, module.AUTH_CARGO_WINCH_EVENT_NAMES.index("package_release_event")] = 1.0
    events[5, module.AUTH_CARGO_WINCH_EVENT_NAMES.index("delivery_completed_event")] = 1.0
    events[7:, module.AUTH_CARGO_WINCH_EVENT_NAMES.index("fully_retracted_state")] = 1.0
    events[11, module.AUTH_CARGO_WINCH_EVENT_NAMES.index("skid_grounded")] = 1.0
    codes[:, 0] = module.AUTH_WINCH_STATE_CODES["LOWERING"]
    codes[:, 3] = 3
    codes[11, 3] = 8
    return {
        "authoritative_node_positions_m": node_pos,
        "authoritative_node_velocities_mps": node_vel,
        "authoritative_cargo_winch_scalars": scalar,
        "authoritative_cargo_winch_events": events,
        "authoritative_cargo_winch_codes": codes,
        "frame_ids": np.arange(n, dtype=np.int64),
        "worker_ids": np.zeros(n, dtype=np.int16),
        "drone_ids": np.zeros(n, dtype=np.int8),
        "reset_generations": np.zeros(n, dtype=np.int32),
        "cargo_generations": np.full(n, 7, dtype=np.int32),
    }


def main() -> None:
    metadata = {
        "dt_s": 0.016,
        "abort_reason_codes": {k: int(v) for k, v in module.AUTH_WINCH_ABORT_REASON_CODES.items()},
        "source_code_version": module.SURVIVAL_CORE_VERSION,
        "source_code_sha256": "synthetic",
        "collection_run_id": "synthetic-test",
        "selected_heldout_worker_ids": [0],
    }
    report = module._phase3_summarize_authoritative_episode_rows(make_arrays(), metadata)
    assert report["status"] == "COMPLETED_DIAGNOSTIC_ONLY"
    assert report["row_count"] == 12 and report["mission_fragment_count"] == 1
    assert report["delivery_cycle_fragment_count"] == 1
    assert report["delivery_outcomes"]["observed_package_release_events"] == 1
    assert report["delivery_outcomes"]["ordered_release_then_completion_then_full_retraction_fragments"] == 1
    assert abs(report["cargo_and_clearance"]["minimum_valid_cargo_surface_clearance_m"] - (-0.02)) < 1e-6
    assert abs(report["cargo_and_clearance"]["maximum_horizontal_endpoint_offset_m"] - 1.2) < 1e-6
    assert report["depot_landing_proxy"]["mission_phase_8_and_skid_grounded_row_count"] == 1
    assert report["control_authority"]["live_control_authorized"] is False

    # Removing full-retraction evidence must produce an unresolved/censored release, not success.
    missing_retract = make_arrays()
    missing_retract["authoritative_cargo_winch_events"][:, module.AUTH_CARGO_WINCH_EVENT_NAMES.index("fully_retracted_state")] = 0.0
    report2 = module._phase3_summarize_authoritative_episode_rows(missing_retract, metadata)
    assert report2["delivery_outcomes"]["ordered_release_then_completion_then_full_retraction_fragments"] == 0
    assert report2["delivery_outcomes"]["released_fragments_not_confirmed_fully_retracted_in_same_contiguous_generation"] == 1

    # Discontiguous frames must split fragments rather than hide missing transitions.
    broken = make_arrays()
    broken["frame_ids"][6:] += 1
    report3 = module._phase3_summarize_authoritative_episode_rows(broken, metadata)
    assert report3["mission_fragment_count"] == 2
    assert report3["delivery_outcomes"]["ordered_release_then_completion_then_full_retraction_fragments"] == 0
    print("[V38.50 EPISODE METRICS TEST PASS] ordered release/completion/retraction, valid-clearance minima, sway proxy, depot proxy, incomplete release handling, frame-gap segmentation")


if __name__ == "__main__":
    main()
