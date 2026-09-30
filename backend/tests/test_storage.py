import csv
import io

import numpy as np
import pytest

from backend.engine import analyze
from backend.schemas import RunConfig
from backend.storage import Storage, parse_csv


def csv_bytes(header, rows):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def test_label_free_missing_data_and_text_columns_are_not_fabricated():
    content = csv_bytes(["负荷", "压力", "说明"], [[i if i != 15 else "", 2 * i, "真实导入"] for i in range(120)])
    data, metadata = parse_csv(content)
    assert data["labels"].size == 0
    assert data["timestamps"][15] == "15"
    assert np.isnan(data["values"][15, 0])
    assert metadata["features"] == ["负荷", "压力"]
    assert metadata["quality"].missing_cells == 1
    assert metadata["quality"].ignored_columns == ["说明"]


@pytest.mark.parametrize("bad_value", ["sensor_failed", "inf", "1e999"])
def test_damaged_numeric_features_are_rejected_not_discarded(bad_value):
    rows = [[i, i + 1, i + 2] for i in range(120)]
    rows[25][0] = bad_value
    with pytest.raises(ValueError, match="无效数值"):
        parse_csv(csv_bytes(["a", "b", "c"], rows))


def test_duplicate_headers_and_out_of_order_time_are_rejected():
    with pytest.raises(ValueError, match="重复列名"):
        parse_csv(csv_bytes(["a", "a"], [[i, i] for i in range(120)]))
    rows = [[f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}Z", i, i] for i in range(120)]
    rows[10], rows[11] = rows[11], rows[10]
    with pytest.raises(ValueError, match="升序"):
        parse_csv(csv_bytes(["timestamp", "a", "b"], rows))


def test_annotations_survive_restart_without_changing_detection(tmp_path):
    store = Storage(tmp_path)
    rng = np.random.default_rng(23)
    values = rng.normal(size=(180, 2))
    values[110:125] += 15
    dataset = store.add_dataset(csv_bytes(["a", "b"], values.tolist()), "用户数据", "upload")
    config = RunConfig(dataset_id=dataset.id, algorithm="temporal", min_event_length=1)
    run = store.add_run(config, dataset)
    output, events, summary, notes = analyze(store.load_arrays("dataset", dataset.id), dataset.features, config)
    store.finish_run(run.id, output, events, summary, notes)
    target = next(event for event in events if event["start"] <= 110 < event["end"])
    store.annotate(run.id, target["id"], {"status": "confirmed", "note": "=HYPERLINK(\"unsafe\") <script>注释</script>"})
    interrupted = store.add_run(config, dataset)
    restarted = Storage(tmp_path)
    reviewed = next(event for event in restarted.events(run.id) if event["id"] == target["id"])
    assert reviewed["status"] == "confirmed"
    assert reviewed["note"].startswith("=HYPERLINK")
    assert reviewed["peak_score"] == target["peak_score"]
    np.testing.assert_array_equal(restarted.load_arrays("run", run.id)["scores"], output["scores"])
    assert restarted.get_run(interrupted.id).status == "failed"
    assert restarted.get_run(run.id).status == "completed"
