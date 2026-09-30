import csv
import io
import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import httpx
from fastapi.testclient import TestClient

from backend.app import app
from backend.schemas import RunConfig


def csv_content(headers, rows):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CHRONOLENS_DATA_DIR", str(tmp_path))
    with TestClient(app) as connected:
        yield connected


@pytest.fixture
def dataset(client):
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(120):
        seconds = i + (2 if i >= 60 else 0)
        if i == 40:
            seconds = 39
        rows.append([(origin + timedelta(seconds=seconds)).isoformat(),
                     "" if i == 3 else i, "" if i == 4 else 2 * i, 5])
    return app.state.storage.add_dataset(csv_content(["timestamp", "a", "b", "constant"], rows),
                                         "诊断样本", "upload")


@pytest.fixture
def completed_run(dataset):
    store = app.state.storage
    config = RunConfig(dataset_id=dataset.id, algorithm="pca")
    run = store.add_run(config, dataset)
    flags = np.zeros(dataset.rows, dtype=np.int8)
    flags[50:55] = 1
    flags[90:93] = 1
    values = np.nan_to_num(store.load_arrays("dataset", dataset.id)["values"])
    output = {"values": values, "reference": values.copy(), "contributions": np.ones_like(values),
              "scores": flags.astype(float) * 3, "flags": flags}
    events = []
    for identifier, start, end, severity, feature in [("e-0001", 50, 55, "high", "a"),
                                                       ("e-0002", 90, 93, "medium", "b")]:
        events.append({"id": identifier, "start": start, "end": end, "start_time": str(start),
                       "end_time": str(end - 1), "length": end - start, "anomaly_points": end - start,
                       "peak_score": 3., "mean_score": 3., "severity": severity, "top_feature": feature,
                       "status": "unreviewed", "note": ""})
    summary = {"points": 120, "n_features": 3, "train_end": 42, "fit_end": 34, "threshold": 1.,
               "anomaly_points": 8, "anomaly_ratio": 8 / 78, "event_count": 2, "duration_ms": 2.,
               "explanation_method": "test", "score_stats": {"min": 0., "max": 3., "median": 0., "p95": 3.},
               "metrics": None}
    store.finish_run(run.id, output, events, summary, [])
    store.annotate(run.id, "e-0001", {"note": "保留备注"})
    return run


def test_profile_reports_finite_statistics_pairwise_correlation_and_sampling(client, dataset):
    response = client.get(f"/api/datasets/{dataset.id}/profile")
    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == dataset.id
    assert body["rows"] == 120
    features = {feature["name"]: feature for feature in body["features"]}
    valid = np.delete(np.arange(120, dtype=float), 3)
    assert features["a"]["count"] == 119
    assert features["a"]["missing"] == 1
    assert features["a"]["missing_ratio"] == pytest.approx(1 / 120)
    for field, expected in {"min": valid.min(), "max": valid.max(), "mean": valid.mean(),
                            "std": valid.std(), "p25": np.quantile(valid, .25),
                            "median": np.median(valid), "p75": np.quantile(valid, .75)}.items():
        assert features["a"][field] == pytest.approx(expected)
    assert body["correlation"]["names"] == ["a", "b", "constant"]
    assert body["correlation"]["values"][0][1] == pytest.approx(1)
    assert body["correlation"]["values"][2] == [None, None, None]
    assert body["sampling"] == {"mode": "timestamp", "median_interval_seconds": 1.,
                                "irregular_intervals": 2, "duplicate_intervals": 1}
    assert body["warnings"]


def test_profile_without_timestamp_keeps_index_sampling(client):
    dataset = app.state.storage.add_dataset(csv_content(["a", "b"], [[i, i * 2] for i in range(120)]),
                                            "索引样本", "upload")
    response = client.get(f"/api/datasets/{dataset.id}/profile")
    assert response.status_code == 200
    assert response.json()["sampling"] == {"mode": "index", "median_interval_seconds": None,
                                           "irregular_intervals": 0, "duplicate_intervals": 0}


def test_insights_count_review_severity_and_real_flag_timeline(client, completed_run):
    store = app.state.storage
    store.annotate(completed_run.id, "e-0002", {"status": "false_positive"})
    response = client.get(f"/api/runs/{completed_run.id}/insights")
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == completed_run.id
    assert body["review"] == {"unreviewed": 1, "confirmed": 0, "false_positive": 1}
    assert body["severity"] == {"high": 1, "medium": 1, "low": 0}
    assert body["top_features"] == [{"name": "a", "event_count": 1}, {"name": "b", "event_count": 1}]
    buckets = body["timeline"]
    assert 1 <= len(buckets) <= 24
    assert buckets[0]["start"] == 42
    assert buckets[-1]["end"] == 120
    assert sum(bucket["total_points"] for bucket in buckets) == 78
    assert sum(bucket["anomaly_points"] for bucket in buckets) == 8
    assert all(left["end"] == right["start"] for left, right in zip(buckets[:-1], buckets[1:]))
    assert all(bucket["label"] for bucket in buckets)


def test_batch_review_updates_atomically_and_preserves_omitted_notes(client, completed_run):
    path = f"/api/runs/{completed_run.id}/events/review"
    response = client.post(path, json={"event_ids": ["e-0002", "e-0001"], "status": "confirmed"})
    assert response.status_code == 200
    reviewed = response.json()["items"]
    assert [event["id"] for event in reviewed] == ["e-0002", "e-0001"]
    assert all(event["status"] == "confirmed" for event in reviewed)
    assert reviewed[1]["note"] == "保留备注"
    missing = client.post(path, json={"event_ids": ["e-0001", "does-not-exist"],
                                       "status": "false_positive", "note": "不应写入"})
    assert missing.status_code == 404
    persisted = app.state.storage.events(completed_run.id)
    assert persisted[0]["status"] == "confirmed"
    assert persisted[0]["note"] == "保留备注"
    assert client.post(path, json={"event_ids": ["e-0001"], "status": "unreviewed", "note": ""}).status_code == 200
    assert app.state.storage.events(completed_run.id)[0]["note"] == ""


@pytest.mark.parametrize("body", [
    {"event_ids": [], "status": "confirmed"},
    {"event_ids": ["e-0001", "e-0001"], "status": "confirmed"},
    {"event_ids": ["e-0001"] * 501, "status": "confirmed"},
    {"event_ids": ["e-0001"], "status": "invalid"},
    {"event_ids": ["e-0001"], "status": "confirmed", "note": None},
    {"event_ids": ["e-0001"], "status": "confirmed", "note": "x" * 2001},
    {"event_ids": ["e-0001"], "status": "confirmed", "unexpected": True},
])
def test_batch_review_rejects_invalid_selection_or_payload(client, completed_run, body):
    response = client.post(f"/api/runs/{completed_run.id}/events/review", json=body)
    assert response.status_code == 422
    assert all(event["status"] == "unreviewed" for event in app.state.storage.events(completed_run.id))


def test_diagnostics_require_existing_dataset_and_completed_run(client, dataset):
    assert client.get("/api/datasets/missing/profile").status_code == 404
    assert client.get("/api/runs/missing/insights").status_code == 404
    run = app.state.storage.add_run(RunConfig(dataset_id=dataset.id, algorithm="pca"), dataset)
    assert client.get(f"/api/runs/{run.id}/insights").status_code == 409
    assert client.post(f"/api/runs/{run.id}/events/review", json={
        "event_ids": ["e-0001"], "status": "confirmed"}).status_code == 409


def test_upload_parsing_does_not_block_health_requests(client, monkeypatch):
    # A deliberately slow CSV parse represents a larger real-world import.
    import threading
    import time

    original = app.state.storage.add_dataset
    started = threading.Event()

    def slow_parse(*args, **kwargs):
        started.set()
        time.sleep(.25)
        return original(*args, **kwargs)

    monkeypatch.setattr(app.state.storage, "add_dataset", slow_parse)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as connected:
            data = csv_content(["a", "b"], [[i, i * 2] for i in range(120)])
            upload = asyncio.create_task(connected.post("/api/datasets/upload", files={"file": ("sample.csv", data)}))
            while not started.is_set():
                await asyncio.sleep(.001)
            assert not upload.done(), "同步 CSV 解析阻塞事件循环直到导入完成"
            response = await connected.get("/api/health")
            assert response.status_code == 200
            assert not upload.done(), "健康检查应能在解析尚未完成时返回"
            assert (await upload).status_code == 200

    asyncio.run(exercise())
