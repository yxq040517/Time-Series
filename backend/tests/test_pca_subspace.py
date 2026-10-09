"""A detector must not accidentally retain the entire observation space."""
import numpy as np

from backend.engine import detect, train_model
from backend.schemas import DetectionConfig, TrainingConfig


def test_independent_features_keep_a_nonempty_residual_space():
    # Equal independent feature variance makes a 90% target select all components.
    values = np.tile([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], (250, 1))
    values[800] = [50.0, 0.0]
    values[801] = [0.0, 50.0]
    labels = np.zeros(len(values), dtype=np.int8)
    labels[800:802] = 1
    data = {"values": values, "timestamps": np.arange(len(values)).astype(str), "labels": labels}
    features = ["sensor_a", "sensor_b"]
    asset, _, _ = train_model(data, features, TrainingConfig(
        dataset_id="regression", name="regression", algorithm="pca", fit_end=280,
        calibration_end=350, threshold_quantile=.99, pca_variance=.9,
    ))
    output, _, summary, _, _ = detect(data, features, asset, DetectionConfig(
        dataset_id="regression", model_id="regression", min_event_length=1, merge_gap=0,
    ))
    # At least one orthogonal large excursion is outside any proper 1D subspace.
    assert output["flags"][800:802].any(), "full-rank reconstruction erased every anomaly"
    assert output["scores"][800:802].max() > 100 * max(summary["threshold"], 1e-12)
