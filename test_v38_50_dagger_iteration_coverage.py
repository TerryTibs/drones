"""V38.50 DAgger lineage, capping and legacy-migration regression tests."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np

SOURCE = Path(__file__).with_name("Survival_Core_V38_50_Iterative_DAgger_Trajectory_Quality.py")
spec = importlib.util.spec_from_file_location("survival_v3850_dagger_test", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import source: {SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample_arrays(n: int):
    return {
        "X": np.zeros((n, module.OBSERVATION_DIM), dtype=np.float32),
        "Yg": np.zeros((n, 4), dtype=np.float32),
        "Yf": (np.arange(n) % 6).astype(np.int64),
        "Yw": np.zeros((n, 2), dtype=np.float32),
        "Yh": np.zeros((n, 8), dtype=np.float32),
        "Yt": np.zeros((n, 2), dtype=np.float32),
        "Yv": np.zeros((n, 1), dtype=np.float32),
        "Ykin": np.zeros((n, 7), dtype=np.float32),
        "Ycargo": np.zeros((n, 6), dtype=np.float32),
        "Yforce": np.zeros((n, 4), dtype=np.float32),
    }


def main() -> None:
    labels = np.tile(np.arange(6, dtype=np.int64), 25)
    iterations = np.repeat(np.arange(10, dtype=np.int32), 15)
    keep = module._phase3_dagger_cap_indices(labels, 37, iterations)
    assert len(keep) == 37 and len(np.unique(keep)) == 37
    assert set(np.unique(iterations[keep])) == set(range(10)), "cap dropped an iteration despite enough capacity"
    assert set(np.unique(labels[keep])) == set(range(6)), "cap dropped a globally present FSM class"

    # Make the cap retain the five newest iteration IDs when the cap cannot retain all.
    keep_small = module._phase3_dagger_cap_indices(labels, 5, iterations)
    assert len(keep_small) == 5
    assert set(np.unique(iterations[keep_small])) == {5, 6, 7, 8, 9}

    arrays = sample_arrays(60)
    groups = np.repeat(np.arange(12, dtype=np.int64), 5)
    iteration_ids = np.repeat(np.arange(6, dtype=np.int32), 10)
    with tempfile.TemporaryDirectory(prefix="v3850_dagger_lineage_") as td:
        corpus = Path(td) / "replay_v2.npz"
        module._phase3_save_dagger_replay(str(corpus), arrays, groups, iteration_ids, max_samples=41)
        loaded = module._phase3_load_dagger_replay(str(corpus), arrays)
        assert loaded is not None
        assert len(loaded["X"]) == 41
        assert set(np.unique(loaded["source_iteration_ids"])) == set(range(6))
        with np.load(corpus, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            assert metadata["schema_version"] == 2
            assert sum(metadata["source_iteration_counts"].values()) == 41

        # Legacy v1 has no iteration metadata. Loader must preserve it as unknown (-1),
        # never guess iteration IDs based on group IDs.
        legacy = Path(td) / "legacy_v1.npz"
        legacy_meta = {"schema_version": 1, "observation_dim": module.OBSERVATION_DIM}
        with open(legacy, "wb") as handle:
            np.savez_compressed(handle, **arrays, source_group_ids=groups,
                                metadata_json=np.asarray(json.dumps(legacy_meta)))
        old = module._phase3_load_dagger_replay(str(legacy), arrays)
        assert old is not None
        assert set(np.unique(old["source_iteration_ids"])) == {-1}
        # Save the migrated corpus; its lineage is explicit and the schema is v2.
        module._phase3_save_dagger_replay(str(legacy), old, old["source_group_ids"],
                                          old["source_iteration_ids"], max_samples=60)
        with np.load(legacy, allow_pickle=False) as data:
            upgraded = json.loads(str(data["metadata_json"].item()))
            assert upgraded["schema_version"] == 2
            assert np.all(np.asarray(data["source_iteration_ids"]) == -1)

    train_idx, val_idx, val_groups = module._phase3_group_heldout_split(
        np.repeat(np.arange(10), 4), np.random.default_rng(3850), 0.2)
    assert not set(np.unique(np.repeat(np.arange(10), 4)[train_idx])).intersection(
        set(np.unique(np.repeat(np.arange(10), 4)[val_idx])))
    assert set(val_groups) == set(np.unique(np.repeat(np.arange(10), 4)[val_idx]))
    print("[V38.50 DAGGER ITERATION TEST PASS] iteration/class-preserving cap, recent-iteration priority when constrained, schema-v2 round-trip, schema-v1 unknown-lineage migration, and whole-group split")


if __name__ == "__main__":
    main()
