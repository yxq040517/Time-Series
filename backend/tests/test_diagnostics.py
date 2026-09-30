import numpy as np
import pytest

from backend.diagnostics import dataset_profile, sampling_profile
from backend.schemas import Dataset, Quality


def metadata(rows, features):
    return Dataset(id="fixture", name="fixture", source="upload", created_at="2026-01-01T00:00:00Z",
                   rows=rows, features=features, start_time="0", end_time=str(rows - 1), has_labels=False,
                   description="", quality=Quality(missing_cells=0, missing_ratio=0., duplicate_timestamps=0,
                                                    constant_features=[], ignored_columns=[], warnings=[]))


def test_sampling_preserves_nanosecond_intervals_at_modern_dates():
    timestamps = np.array(["2026-01-01T00:00:00.000000001Z", "2026-01-01T00:00:00.000000002Z",
                           "2026-01-01T00:00:00.000000003Z"])
    profile = sampling_profile(timestamps)
    assert profile["median_interval_seconds"] == pytest.approx(1e-9)
    assert profile["duplicate_intervals"] == 0


def test_profile_unavailable_statistics_and_unpaired_correlation_return_null():
    source = {"values": np.array([[1., np.nan, np.nan], [np.nan, 2., np.nan], [np.nan, 2., np.nan]]),
              "timestamps": np.arange(3).astype(str)}
    profile = dataset_profile(metadata(3, ["a", "b", "empty"]), source)
    empty = profile["features"][2]
    assert empty["count"] == 0
    assert empty["missing"] == 3
    assert all(empty[key] is None for key in ("min", "max", "mean", "std", "p25", "median", "p75"))
    assert profile["correlation"]["values"] == [[None] * 3 for _ in range(3)]


def test_profile_very_large_finite_values_stay_finite_without_overflow():
    source = {"values": np.array([[-1e308, -5e307], [0., 0.], [1e308, 5e307]]),
              "timestamps": np.arange(3).astype(str)}
    with np.errstate(over="raise", invalid="raise"):
        profile = dataset_profile(metadata(3, ["a", "b"]), source)
    assert profile["features"][0]["mean"] == 0
    assert profile["features"][0]["std"] == pytest.approx(1e308 * np.sqrt(2 / 3))
    assert profile["correlation"]["values"][0][1] == pytest.approx(1)
