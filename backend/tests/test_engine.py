import numpy as np
import pytest

from backend.engine import detect, segment_events, train_model
from backend.schemas import DetectionConfig, TrainingConfig


FEATURES = ["a", "b", "c"]


def sample_data():
    rng = np.random.default_rng(19)
    values = rng.normal(size=(360, 3))
    values[:, 1] = .8 * values[:, 0] + rng.normal(scale=.1, size=360)
    values[20, 2] = np.nan
    labels = np.zeros(360, dtype=np.int8)
    labels[240:252] = 1
    values[240:252, 2] += 7
    return {"values": values, "labels": labels, "timestamps": np.arange(360).astype(str)}


@pytest.mark.parametrize("algorithm", ["pca", "temporal", "isolation"])
def test_future_values_and_labels_cannot_change_historical_calibration(algorithm):
    data = sample_data()
    config = TrainingConfig(dataset_id="fixture", name="fixture", algorithm=algorithm,
                            fit_start=10, fit_end=120, calibration_end=180)
    asset, _, _ = train_model(data, FEATURES, config)
    detection = DetectionConfig(dataset_id="fixture", model_id="fixture")
    output, _, _, _, _ = detect(data, FEATURES, asset, detection)
    altered = {key: value.copy() for key, value in data.items()}
    altered["values"][:config.fit_start] += 100
    altered["values"][config.calibration_end:] += 100
    altered["labels"] = 1 - altered["labels"]
    other_asset, _, _ = train_model(altered, FEATURES, config)
    other_output, _, _, _, _ = detect(altered, FEATURES, other_asset, detection)
    assert asset["threshold"] == other_asset["threshold"]
    # Detection legitimately sees earlier inputs, so compare independent files
    # beginning at fit_start rather than a temporal stream with a changed prefix.
    historical = {key: value[config.fit_start:config.calibration_end] for key, value in data.items()}
    historical_other = {key: value[config.fit_start:config.calibration_end] for key, value in altered.items()}
    first, _, _, _, _ = detect(historical, FEATURES, asset, detection)
    second, _, _, _, _ = detect(historical_other, FEATURES, other_asset, detection)
    np.testing.assert_array_equal(first["scores"], second["scores"])
    np.testing.assert_array_equal(output["scores"][config.fit_start + config.window:config.calibration_end],
                                  other_output["scores"][config.fit_start + config.window:config.calibration_end])


def test_missing_imputation_never_uses_future_or_calibration_values():
    values = np.array([[1., 2.], [np.nan, 4.], [3., 6.], [2., 4.], [2., 4.],
                       [999., np.nan], [np.nan, 1000.], [5., 10.], [6., 12.], [7., 14.]])
    data = {"values": values, "timestamps": np.arange(len(values)).astype(str), "labels": np.array([])}
    config = TrainingConfig(dataset_id="fixture", name="fixture", algorithm="pca",
                            fit_end=5, calibration_end=10)
    asset, _, _ = train_model(data, ["a", "b"], config)
    output, _, _, _, _ = detect(data, ["a", "b"], asset,
                                DetectionConfig(dataset_id="fixture", model_id="fixture"))
    assert output["values"][1, 0] == 2
    assert output["values"][6, 0] == 2
    assert output["values"][5, 1] == 4
    values[:5, 0] = np.nan
    with pytest.raises(ValueError, match="完全缺失"):
        train_model(data, ["a", "b"], config)


def test_temporal_reference_does_not_see_current_observation():
    data = sample_data()
    asset, _, _ = train_model(data, FEATURES, TrainingConfig(
        dataset_id="fixture", name="fixture", algorithm="temporal", fit_end=120, calibration_end=180))
    config = DetectionConfig(dataset_id="fixture", model_id="fixture")
    output, _, _, _, _ = detect(data, FEATURES, asset, config)
    changed = {key: value.copy() for key, value in data.items()}
    changed["values"][200, 0] += 20
    changed_output, _, _, _, _ = detect(changed, FEATURES, asset, config)
    np.testing.assert_array_equal(output["reference"][:201], changed_output["reference"][:201])
    assert changed_output["scores"][200] != output["scores"][200]


def test_temporal_raw_history_survives_feature_permutation_and_short_batches():
    data = sample_data()
    window = 8
    asset, _, _ = train_model(data, FEATURES, TrainingConfig(
        dataset_id="fixture", name="fixture", algorithm="temporal", fit_end=120,
        calibration_end=180, window=window))
    config = DetectionConfig(dataset_id="fixture", model_id="fixture", min_event_length=1, merge_gap=0)
    incoming = {key: value[180:220].copy() for key, value in data.items()}
    # A required column can be entirely missing in a short incoming batch.
    incoming["values"][:3, 2] = np.nan
    whole, _, _, _, whole_history = detect(incoming, FEATURES, asset, config)
    first_data = {key: value[:3] for key, value in incoming.items()}
    first, events, summary, _, history = detect(first_data, FEATURES, asset, config)
    assert not first["scored"].any()
    assert events == []
    assert summary["scored_points"] == 0
    assert summary["warmup_points"] == 3
    assert summary["metrics"] is None and summary["score_stats"] is None
    assert summary["anomaly_ratio"] == 0
    np.testing.assert_array_equal(history, first["values"])
    permutation = [2, 0, 1]
    second_data = {key: value[3:].copy() for key, value in incoming.items()}
    second_data["values"] = second_data["values"][:, permutation]
    second, _, _, _, final_history = detect(second_data, [FEATURES[index] for index in permutation],
                                             asset, config, history=history)
    for key in ("reference", "contributions", "values"):
        np.testing.assert_allclose(second[key], whole[key][3:, permutation], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.r_[first["scores"], second["scores"]], whole["scores"],
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(np.r_[first["scored"], second["scored"]], whole["scored"])
    np.testing.assert_array_equal(np.r_[first["flags"], second["flags"]], whole["flags"])
    np.testing.assert_array_equal(final_history, whole_history)
    independent, _, _, _, _ = detect(second_data, [FEATURES[index] for index in permutation], asset, config)
    assert not independent["scored"][:window].any()
    assert second["scored"][window - 3]


def test_event_gap_merging_keeps_exclusive_end_and_true_point_count():
    flags = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0], dtype=np.int8)
    scores = flags.astype(float) * 3
    contribution = np.column_stack([scores, scores / 2])
    events = segment_events(flags, scores, contribution, np.arange(9).astype(str),
                            ["a", "b"], threshold=1, minimum=3, gap=1)
    assert len(events) == 1
    assert (events[0]["start"], events[0]["end"]) == (1, 5)
    assert events[0]["length"] == 4
    assert events[0]["anomaly_points"] == 3
    assert events[0]["end_time"] == "4"
