"""A detector must not accidentally retain the entire observation space."""
import numpy as np

from backend.engine import analyze
from backend.schemas import RunConfig


def test_independent_features_keep_a_nonempty_residual_space():
    # Equal independent feature variance makes a 90% target select all components.
    values = np.tile([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], (250, 1))
    values[800] = [50.0, 0.0]
    values[801] = [0.0, 50.0]
    labels = np.zeros(len(values), dtype=np.int8)
    labels[800:802] = 1
    data = {"values": values, "timestamps": np.arange(len(values)).astype(str), "labels": labels}
    output, _, summary, _ = analyze(data, ["sensor_a", "sensor_b"], RunConfig(
        dataset_id="regression", algorithm="pca", train_ratio=.35,
        threshold_quantile=.99, pca_variance=.9, min_event_length=1, merge_gap=0,
    ))
    # At least one orthogonal large excursion is outside any proper 1D subspace.
    assert output["flags"][800:802].any(), "full-rank reconstruction erased every anomaly"
    assert output["scores"][800:802].max() > 100 * max(summary["threshold"], 1e-12)
