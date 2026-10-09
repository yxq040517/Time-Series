from concurrent.futures import ThreadPoolExecutor
import os
import threading
import uuid

import numpy as np
import pytest

from backend.storage import Storage


def test_repeated_array_reads_share_immutable_values_but_not_mapping(tmp_path):
    store = Storage(tmp_path)
    identifier = uuid.uuid4().hex
    store.save_arrays("dataset", identifier, {"values": np.arange(20.)})
    first = store.load_arrays("dataset", identifier)
    second = store.load_arrays("dataset", identifier)
    assert first["values"] is second["values"]
    with pytest.raises(ValueError):
        first["values"][0] = 50
    del first["values"]
    assert "values" in store.load_arrays("dataset", identifier)


def test_cache_detects_saved_and_external_array_replacements(tmp_path):
    store = Storage(tmp_path)
    identifier = uuid.uuid4().hex
    store.save_arrays("run", identifier, {"values": np.arange(20.)})
    first = store.load_arrays("run", identifier)["values"]
    store.save_arrays("run", identifier, {"values": np.arange(20.) + 5})
    second = store.load_arrays("run", identifier)["values"]
    assert second is not first
    assert second[0] == 5
    path = store.array_path("run", identifier)
    replacement = tmp_path / "replacement.npz"
    np.savez_compressed(replacement, values=np.arange(30.) + 9)
    os.replace(replacement, path)
    external = store.load_arrays("run", identifier)["values"]
    assert external is not second
    assert external[0] == 9
    assert len(external) == 30


def test_cache_evicts_oldest_arrays_by_uncompressed_memory_and_skips_oversized(tmp_path):
    store = Storage(tmp_path, array_cache_bytes=160)
    identifiers = [uuid.uuid4().hex for _ in range(3)]
    for identifier in identifiers[:2]:
        store.save_arrays("run", identifier, {"values": np.arange(20.)})
    first = store.load_arrays("run", identifiers[0])["values"]
    second = store.load_arrays("run", identifiers[1])["values"]
    assert store.load_arrays("run", identifiers[1])["values"] is second
    assert store.load_arrays("run", identifiers[0])["values"] is not first
    store.save_arrays("run", identifiers[2], {"values": np.arange(21.)})
    oversized = store.load_arrays("run", identifiers[2])["values"]
    assert store.load_arrays("run", identifiers[2])["values"] is not oversized
    assert not oversized.flags.writeable


def test_concurrent_cache_misses_decompress_once_and_concurrent_saves_remain_valid(tmp_path):
    store = Storage(tmp_path)
    identifier = uuid.uuid4().hex
    store.save_arrays("dataset", identifier, {"values": np.arange(1000.)})
    barrier = threading.Barrier(8)

    def read():
        barrier.wait()
        return store.load_arrays("dataset", identifier)["values"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        loaded = list(pool.map(lambda _: read(), range(8)))
    assert all(array is loaded[0] for array in loaded)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda number: store.save_arrays("dataset", identifier,
                                                       {"values": np.full(1000, number, dtype=float)}), range(8)))
    final = store.load_arrays("dataset", identifier)["values"]
    assert len(final) == 1000
    assert np.unique(final).size == 1
