import numpy as np
import pytest

from backend.engine import analyze, prepare, segment_events
from backend.schemas import RunConfig


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
    config = RunConfig(dataset_id="fixture", algorithm=algorithm)
    output, _, summary, _ = analyze(data, ["a", "b", "c"], config)
    altered = {key: value.copy() for key, value in data.items()}
    altered["values"][summary["train_end"]:] += 100
    altered["labels"] = 1 - altered["labels"]
    other_output, _, other_summary, _ = analyze(altered, ["a", "b", "c"], config)
    assert summary["threshold"] == other_summary["threshold"]
    np.testing.assert_array_equal(output["scores"][:summary["train_end"]],
                                  other_output["scores"][:summary["train_end"]])
    assert not output["flags"][:summary["train_end"]].any()
    assert not other_output["flags"][:summary["train_end"]].any()


def test_missing_imputation_never_uses_future_or_calibration_values():
    values = np.array([[1., 2.], [np.nan, 4.], [3., 6.], [999., np.nan], [np.nan, 1000.]])
    filled, _, _, _ = prepare(values, fit_end=3)
    assert filled[1, 0] == 2
    assert filled[4, 0] == 2
    assert filled[3, 1] == 4
    values[:3, 0] = np.nan
    with pytest.raises(ValueError, match="完全缺失"):
        prepare(values, fit_end=3)


def test_temporal_reference_does_not_see_current_observation():
    data = sample_data()
    config = RunConfig(dataset_id="fixture", algorithm="temporal")
    output, _, _, _ = analyze(data, ["a", "b", "c"], config)
    changed = {key: value.copy() for key, value in data.items()}
    changed["values"][200, 0] += 20
    changed_output, _, _, _ = analyze(changed, ["a", "b", "c"], config)
    np.testing.assert_array_equal(output["reference"][:201], changed_output["reference"][:201])
    assert changed_output["scores"][200] != output["scores"][200]


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
