"""Focused V38.48 tests. Synthetic artifacts validate telemetry plumbing, not model physics."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np

SOURCE = Path(__file__).with_name("Survival_Core_V38_48_Authoritative_Cargo_Winch_Telemetry.py")
spec = importlib.util.spec_from_file_location("survival_v3848_test_module", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import V38.48 source: {SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fake_winch(*, released=False):
    node_positions = np.zeros((5, 3), dtype=np.float32)
    node_positions[:, 0] = -2.0
    node_positions[:, 1] = -3.0
    node_positions[:, 2] = np.linspace(9.0, 3.0, 5)
    node_velocities = np.zeros_like(node_positions)
    node_velocities[:, 2] = -0.1
    return SimpleNamespace(
        nodes_pos=node_positions, nodes_vel=node_velocities, num_nodes=5,
        cable_len=2.0, cable_vel=0.3, tension_n=7.5, tension_derivative=1.2,
        contact_normal_force=0.0, has_payload=not released, severed=False,
        state="RETRACTING" if released else "LOWERING", touchdown_timer=0.0,
        release_count=1 if released else 0,
        last_release_pos=np.array([-2.0, -3.0, 0.09], dtype=np.float32) if released else None,
        cargo_yaw=0.25, cargo_yaw_rate=0.1,
        payout_wind_paused=True, last_payout_pause_reason="wind pause",
        last_abort_reason=None, delivery_completed=False, is_skid_grounded=False,
    )


def make_chunk(path: Path) -> tuple[int, int]:
    drones = int(module.NUM_SWARM_DRONES)
    frames = 200
    n = frames * drones
    obs = np.zeros((n, module.OBSERVATION_DIM), dtype=np.float32)
    guidance = np.zeros((n, 4), dtype=np.float32)
    fsm = np.ones(n, dtype=np.int64)
    winch = np.zeros((n, 2), dtype=np.float32)
    targets = np.zeros((n, module.ACTION_TRANSITION_TARGET_DIM), dtype=np.float32)
    targets[:, -1] = 1.0
    nodes_pos = np.zeros((n, 5, 3), dtype=np.float32)
    nodes_vel = np.zeros_like(nodes_pos)
    nodes_pos[:, :, 0] = -2.0
    nodes_pos[:, :, 1] = -3.0
    nodes_pos[:, :, 2] = np.linspace(9.0, 3.0, 5)[None, :]
    scalars = np.zeros((n, len(module.AUTH_CARGO_WINCH_SCALAR_NAMES)), dtype=np.float32)
    scalars[:, module.AUTH_CARGO_WINCH_SCALAR_NAMES.index("cable_length_m")] = 2.0
    scalars[:, module.AUTH_CARGO_WINCH_SCALAR_NAMES.index("tension_n")] = 7.5
    scalars[:, module.AUTH_CARGO_WINCH_SCALAR_NAMES.index("cargo_surface_clearance_m")] = 2.91
    events = np.zeros((n, len(module.AUTH_CARGO_WINCH_EVENT_NAMES)), dtype=np.float32)
    eidx = {name: module.AUTH_CARGO_WINCH_EVENT_NAMES.index(name) for name in module.AUTH_CARGO_WINCH_EVENT_NAMES}
    events[:, eidx["payload_present_before_step"]] = 1.0
    events[:, eidx["payload_present_after_step"]] = 1.0
    events[:, eidx["cargo_clearance_valid"]] = 1.0
    events[:, eidx["telemetry_valid"]] = 1.0
    codes = np.zeros((n, 4), dtype=np.int16)
    codes[:, 0] = module.AUTH_WINCH_STATE_CODES["LOWERING"]
    frame_ids = np.repeat(np.arange(frames, dtype=np.int64), drones)
    worker_ids = np.zeros(n, dtype=np.int16)
    drone_ids = np.tile(np.arange(drones, dtype=np.int8), frames)
    reset_gen = np.repeat(np.where(np.arange(frames) < 90, 0, 1).astype(np.int32), drones)
    cargo_gen = np.repeat(np.where(np.arange(frames) < 150, 0, 1).astype(np.int32), drones)
    np.savez_compressed(
        path, obs=obs, yg=guidance, yf=fsm, yw=winch, yaction_transition=targets,
        auth_node_positions_m=nodes_pos, auth_node_velocities_mps=nodes_vel,
        auth_scalars=scalars, auth_events=events, auth_codes=codes,
        auth_frame_ids=frame_ids, auth_worker_ids=worker_ids, auth_drone_ids=drone_ids,
        auth_reset_generations=reset_gen, auth_cargo_generations=cargo_gen,
    )
    return frames, drones


def main() -> None:
    released_winch = fake_winch(released=True)
    sample = module._capture_authoritative_cargo_winch_state(
        released_winch, np.array([-2.0, -3.0, 10.0]), np.zeros(3), 0.0,
        0.016, 1, 7, 2, 0, 0, True, 0, False, 0.0,
    )
    assert sample["node_positions_m"].shape == (5, 3)
    assert sample["node_velocities_mps"].shape == (5, 3)
    assert sample["scalars"].shape == (len(module.AUTH_CARGO_WINCH_SCALAR_NAMES),)
    assert sample["events"].shape == (len(module.AUTH_CARGO_WINCH_EVENT_NAMES),)
    assert sample["events"][module.AUTH_CARGO_WINCH_EVENT_NAMES.index("package_release_event")] == 1.0
    assert sample["events"][module.AUTH_CARGO_WINCH_EVENT_NAMES.index("payload_latch_changed")] == 1.0
    assert sample["events"][module.AUTH_CARGO_WINCH_EVENT_NAMES.index("release_position_valid")] == 1.0
    assert sample["events"][module.AUTH_CARGO_WINCH_EVENT_NAMES.index("delivery_completed_event")] == 0.0
    assert sample["worker_id"] == 7 and sample["drone_id"] == 2 and sample["frame_id"] == 1

    # World-origin coordinates are physically valid and must not be rejected by an arbitrary nonzero-position heuristic.
    origin_winch = fake_winch()
    origin_winch.nodes_pos[:, 0:2] = 0.0
    origin = module._capture_authoritative_cargo_winch_state(
        origin_winch, np.array([0.0, 0.0, 10.0]), np.zeros(3), 0.0,
        0.016, 3, 7, 2, 0, 0, True, 0, False, 0.0,
    )
    assert origin["events"][module.AUTH_CARGO_WINCH_EVENT_NAMES.index("cargo_clearance_valid")] == 1.0

    # Non-finite tether state must be retained as invalid, never silently blessed.
    bad_winch = fake_winch()
    bad_winch.nodes_vel[2, 1] = np.nan
    bad = module._capture_authoritative_cargo_winch_state(
        bad_winch, np.array([-2.0, -3.0, 10.0]), np.zeros(3), 0.0,
        0.016, 2, 7, 2, 0, 0, True, 0, False, 0.0,
    )
    assert bad["events"][module.AUTH_CARGO_WINCH_EVENT_NAMES.index("telemetry_valid")] == 0.0
    assert not np.isfinite(bad["node_velocities_mps"]).all()

    names = ("SURVIVAL_AUTH_CARGO_WINCH_DATASET", "SURVIVAL_AUTH_CARGO_WINCH_AUDIT_REPORT")
    old_env = {key: os.environ.get(key) for key in names}
    try:
        with tempfile.TemporaryDirectory(prefix="v38_48_auth_cargo_winch_") as td:
            root = Path(td)
            chunk = root / "chunk0.npz"
            frames, drones = make_chunk(chunk)
            dataset = root / "sequences_v2.npz"
            saved = module._save_transition_rollout_sequence_artifact(
                [str(chunk)], heldout_worker_ids=[0], output_path=str(dataset),
                sequence_length=64, segments_per_drone=4,
            )
            assert saved["status"] == "SAVED", saved
            with np.load(dataset, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
                assert metadata["schema_version"] == 2
                assert metadata["authoritative_cargo_winch_telemetry_schema_version"] == 1
                assert data["authoritative_node_positions_m"].shape[:2] == data["observations"].shape[:2]
                assert data["authoritative_cargo_winch_events"].shape[-1] == len(module.AUTH_CARGO_WINCH_EVENT_NAMES)
                assert data["authoritative_cargo_winch_codes"].shape[-1] == 4
                assert metadata["source_code_sha256"]
                assert np.all(data["authoritative_frame_ids"] >= 0)
                reset = data["authoritative_reset_generations"]
                cargo = data["authoritative_cargo_generations"]
                frame = data["authoritative_frame_ids"]
                starts = data["start_frames"]
                assert len(starts) > 0
                for i in range(len(starts)):
                    assert np.unique(reset[i]).size == 1
                    assert np.unique(cargo[i]).size == 1
                    assert np.all(np.diff(frame[i]) == 1)
                    # The generated test has reset at frame 90; the segment may not cross it.
                    assert int(starts[i]) + 64 <= 90
            os.environ["SURVIVAL_AUTH_CARGO_WINCH_DATASET"] = str(dataset)
            audit_out = root / "audit.json"
            os.environ["SURVIVAL_AUTH_CARGO_WINCH_AUDIT_REPORT"] = str(audit_out)
            report = module.run_authoritative_cargo_winch_dataset_audit()
            assert audit_out.is_file()
            assert report["status"] == "PASS_SCHEMA_ALIGNMENT_AND_EVENT_ENCODING", report
            assert report["counts"]["valid_authoritative_telemetry_rows"] > 0
            assert report["live_control_authorized"] is False
            print("[V38.48 AUTHORITATIVE TELEMETRY TEST PASS] capture contract, release/latch edge, nonfinite masking, schema-v2 artifact, reset/cargo boundaries, aligned arrays and offline audit.")
            print("[LIMIT] All truth values are generated by a synthetic fixture; test proves data contract plumbing only, not physical-model fidelity or trained-model accuracy.")
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    main()
