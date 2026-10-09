import csv
import html
import io
import logging
import os
import sqlite3
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from threadpoolctl import threadpool_limits

from .demo import make_demo
from .diagnostics import dataset_profile, run_insights, sampling_profile, scored_mask
from .engine import detect, peak_sample, summarize_detection, train_model
from .schemas import Annotation, BatchAnnotation, Dataset, DatasetProfile, DemoRequest, DetectionConfig, Event, ModelInfo, Run, RunInsights, TrainingConfig, TrainingJob
from .storage import MAX_UPLOAD, Storage, now

LOGGER = logging.getLogger("chronolens")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(application: FastAPI):
    root = Path(os.environ.get("CHRONOLENS_DATA_DIR", str(PROJECT_ROOT / "backend" / "data")))
    application.state.storage = Storage(root)
    application.state.slots = threading.BoundedSemaphore(6)
    application.state.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chronolens")
    application.state.futures = {}
    application.state.stream_futures = {}
    application.state.futures_lock = threading.Lock()
    # BLAS/OpenMP bounds apply process-wide; overlapping per-worker contexts are unsafe.
    limiter = threadpool_limits(limits=1)
    try:
        yield
    finally:
        application.state.executor.shutdown(wait=True, cancel_futures=True)
        with application.state.futures_lock:
            for identifier, future in application.state.futures.items():
                if future.cancelled():
                    update = application.state.storage.update_training if application.state.storage.get_training(identifier) else application.state.storage.update_run
                    update(identifier, status="failed", message="服务关闭中断任务",
                           error="任务在服务关闭时取消，请重新提交。", completed_at=now())
        limiter.restore_original_limits()


app = FastAPI(title="ChronoLens 时序异常检测与解释系统", version="2.0.0", lifespan=lifespan)


def storage() -> Storage:
    return app.state.storage


def dataset_or_404(identifier: str) -> Dataset:
    dataset = storage().get_dataset(identifier)
    if dataset is None:
        raise HTTPException(404, "数据集不存在。")
    return dataset


def model_or_404(identifier: str) -> ModelInfo:
    model = storage().get_model(identifier)
    if model is None:
        raise HTTPException(404, "模型不存在。")
    return model


def run_or_404(identifier: str) -> Run:
    run = storage().get_run(identifier)
    if run is None:
        raise HTTPException(404, "分析任务不存在。")
    return run


def completed_run(identifier: str) -> Run:
    run = run_or_404(identifier)
    if run.status != "completed":
        raise HTTPException(409, "分析尚未完成；请等待任务完成后查看结果。")
    return run


def result(identifier: str) -> tuple[Run, Dataset, dict, dict]:
    run = completed_run(identifier)
    dataset = dataset_or_404(run.dataset_id)
    return run, dataset, storage().load_arrays("dataset", dataset.id), storage().load_arrays("run", run.id)


def bounded_range(start: int, end: int | None, count: int) -> tuple[int, int]:
    left = min(count, max(0, start))
    right = count if end is None else min(count, max(0, end))
    if right <= left:
        raise HTTPException(422, "区间为空或结束位置不大于起始位置。索引为左闭右开。")
    return left, right


def selected_features(query: str | None, names: list[str]) -> list[str]:
    if query is None:
        return names
    if query in names:
        return [query]
    selected = list(dict.fromkeys(query.split(",")))
    if not selected or any(name not in names for name in selected):
        raise HTTPException(422, "请求的特征不存在。")
    return selected


def execute_training(identifier: str, config: TrainingConfig, dataset: Dataset):
    store = storage()
    try:
        store.update_training(identifier, status="running", progress=3, message="读取训练数据")
        asset, summary, notes = train_model(store.load_arrays("dataset", dataset.id), dataset.features, config,
                                            lambda value, message: store.update_training(identifier, progress=value, message=message))
        store.finish_training(identifier, asset, summary, notes)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        LOGGER.warning("Training %s rejected: %s", identifier, exc)
        store.update_training(identifier, status="failed", message="训练失败", error=str(exc), completed_at=now())
    except Exception:
        LOGGER.exception("Unexpected training failure: %s", identifier)
        store.update_training(identifier, status="failed", message="训练失败",
                              error="训练遇到内部错误；请检查服务终端日志。", completed_at=now())
    finally:
        app.state.slots.release()


def stream_context(store: Storage, model: ModelInfo, config: DetectionConfig, arrays: dict):
    state = store.get_stream(model.id, config.stream_id)
    sampling = sampling_profile(arrays["timestamps"])
    history, notes, cuts = None, [], []
    if sampling["mode"] == "timestamp":
        times = pd.to_datetime(arrays["timestamps"], utc=True, format="mixed")
        if times.duplicated().any() or not times.is_monotonic_increasing:
            raise ValueError("连续流时间戳必须严格递增，不允许重复、重叠或乱序。")
        interval = sampling["median_interval_seconds"]
        if state is not None and state["mode"] == "timestamp":
            last = pd.Timestamp(state["last_timestamp"])
            if times[0] <= last:
                raise ValueError("连续流批次与已提交历史重叠或乱序；流状态未改变。")
            gap = (int(times[0].value) - int(last.value)) / 1e9
            expected = state.get("interval_seconds") or interval or gap
            cadence_changed = interval is not None and expected is not None and not np.isclose(interval, expected, rtol=.01, atol=0)
            if cadence_changed or not np.isclose(gap, expected, rtol=.01, atol=0):
                notes.append("连续流出现采样间断，已重置历史窗口；本批次重新预热。")
            else:
                history = np.asarray(state["history"], dtype=float)
            if interval is None:
                interval = expected
        if interval is not None:
            deltas = np.asarray([(int(b) - int(a)) / 1e9 for a, b in zip(times.asi8[:-1], times.asi8[1:])])
            cuts = (np.flatnonzero(~np.isclose(deltas, interval, rtol=.01, atol=0)) + 1).tolist()
            if cuts:
                notes.append(f"本批次有 {len(cuts)} 个采样间断；每次间断后重置窗口并重新预热。")
        metadata = {"mode": "timestamp", "last_timestamp": str(arrays["timestamps"][-1]), "interval_seconds": interval}
    else:
        if state is not None and state["mode"] == "index":
            history = np.asarray(state["history"], dtype=float)
        metadata = {"mode": "index", "appended_points": (state.get("appended_points", 0) if state else 0) + len(arrays["values"])}
    if state is not None and state["mode"] != sampling["mode"]:
        notes.append("连续流采样模式改变，已重置历史窗口并重新预热。")
    return history, metadata, notes, cuts


def execute(identifier: str, config: DetectionConfig, dataset: Dataset, predecessor=None):
    store = storage()
    try:
        if predecessor is not None:
            try:
                predecessor.result()
            except CancelledError:
                pass
        store.update_run(identifier, status="running", progress=3, message="读取已发布模型与检测数据")
        model = store.get_model(config.model_id)
        if model is None:
            raise ValueError("检测模型不存在。")
        store.validate_detection(config, dataset, model)
        asset = store.load_model_asset(model.id)
        arrays = store.load_arrays("dataset", dataset.id)
        lock = store.stream_lock(model.id, config.stream_id) if config.stream_id is not None else nullcontext()
        with lock:
            history, metadata, stream_notes, cuts = stream_context(store, model, config, arrays) if config.stream_id is not None else (None, None, [], [])
            started = time.perf_counter()
            boundaries = [0, *cuts, len(arrays["values"])]
            outputs, notes, new_history = [], [], history
            for segment, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                batch = {key: value[left:right] if len(value) else value for key, value in arrays.items()}
                progress = lambda value, message: store.update_run(identifier,
                    progress=min(99, 3 + int(96 * (left + (right - left) * value / 100) / dataset.rows)), message=message)
                output, events, summary, segment_notes, new_history = detect(batch, dataset.features, asset, config,
                    progress, history=new_history if segment == 0 else None)
                outputs.append(output)
                notes.extend(segment_notes)
            if len(outputs) > 1:
                output = {key: np.concatenate([item[key] for item in outputs], axis=0) for key in outputs[0]}
                events, summary = summarize_detection(arrays, dataset.features, asset, config, output,
                                                       (time.perf_counter() - started) * 1000)
            notes = list(dict.fromkeys(notes))
            stream_update = None
            if metadata is not None:
                metadata["history"] = new_history.tolist()
                stream_update = (model.id, config.stream_id, metadata)
            store.finish_run(identifier, output, events, summary, stream_notes + notes, stream_update=stream_update)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        LOGGER.warning("Inference %s rejected: %s", identifier, exc)
        message = str(exc) if isinstance(exc, ValueError) else "数值计算不稳定，请检查输入数值量级。"
        store.update_run(identifier, status="failed", message="检测失败", error=message, completed_at=now())
    except Exception:
        LOGGER.exception("Unexpected inference failure: %s", identifier)
        store.update_run(identifier, status="failed", message="检测失败",
                         error="检测遇到内部错误；请检查服务终端日志。", completed_at=now())
    finally:
        app.state.slots.release()


def remember_future(identifier: str, future):
    with app.state.futures_lock:
        app.state.futures = {key: value for key, value in app.state.futures.items() if not value.done()}
        app.state.stream_futures = {key: value for key, value in app.state.stream_futures.items() if not value.done()}
        app.state.futures[identifier] = future


def mean_contributions(values: np.ndarray) -> np.ndarray:
    # Contributions are nonnegative; scaling avoids overflowing a long finite
    # bucket before division by its length.
    scale = np.maximum(values.max(axis=0), 1.)
    return (values / scale).mean(axis=0) * scale


@app.exception_handler(OSError)
@app.exception_handler(sqlite3.Error)
async def io_error(request: Request, exc: OSError | sqlite3.Error):
    LOGGER.error("Storage error on %s", request.url.path, exc_info=exc)
    return JSONResponse(status_code=503, content={"detail": "本地存储不可用，请检查数据目录权限、磁盘空间与服务日志。"})


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/api/datasets")
def datasets():
    return {"items": storage().list_datasets()}


@app.post("/api/datasets/demo", response_model=Dataset)
def demo(body: DemoRequest | None = Body(default=None)):
    frame, metadata = make_demo(42 if body is None else body.seed)
    try:
        return storage().add_dataset(frame.to_csv(index=False).encode("utf-8"), metadata["name"], "demo", metadata["description"])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/datasets/upload", response_model=Dataset)
async def upload(file: UploadFile = File(...), name: str | None = Form(None),
                 timestamp_column: str | None = Form(None), label_column: str | None = Form(None)):
    try:
        content = await file.read(MAX_UPLOAD + 1)
        if len(content) > MAX_UPLOAD:
            raise HTTPException(413, "CSV 不得超过 25 MiB。")
        return await run_in_threadpool(storage().add_dataset, content, name or file.filename or "导入数据", "upload",
                                       "用户导入的 CSV 数据；不预设真实异常。", timestamp_column, label_column)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        await file.close()


@app.get("/api/datasets/{identifier}", response_model=Dataset)
def dataset_detail(identifier: str):
    return dataset_or_404(identifier)


@app.get("/api/datasets/{identifier}/preview")
def preview(identifier: str, limit: int = Query(default=8, ge=1, le=100)):
    dataset = dataset_or_404(identifier)
    data = storage().load_arrays("dataset", dataset.id)
    columns = ["timestamp", *dataset.features] + (["is_anomaly"] if dataset.has_labels else [])
    rows = []
    for index in range(min(limit, dataset.rows)):
        row = {"timestamp": str(data["timestamps"][index])}
        row.update({feature: float(data["values"][index, j]) if np.isfinite(data["values"][index, j]) else None
                    for j, feature in enumerate(dataset.features)})
        if dataset.has_labels:
            row["is_anomaly"] = int(data["labels"][index])
        rows.append(row)
    return {"columns": columns, "rows": rows}


@app.get("/api/datasets/{identifier}/profile", response_model=DatasetProfile)
def profile(identifier: str):
    dataset = dataset_or_404(identifier)
    return dataset_profile(dataset, storage().load_arrays("dataset", dataset.id))


def safe_cell(value):
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r", "\n")):
        return "'" + value
    if isinstance(value, str) and value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


def csv_response(rows, filename: str) -> Response:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\r\n")
    for row in rows:
        writer.writerow([safe_cell(cell) for cell in row])
    return Response(buffer.getvalue().encode("utf-8-sig"), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename=\"chronolens.csv\"; filename*=UTF-8''{quote(filename)}"})


@app.get("/api/datasets/{identifier}/export.csv")
def source_csv(identifier: str):
    dataset = dataset_or_404(identifier)
    data = storage().load_arrays("dataset", dataset.id)
    def rows():
        yield ["timestamp", *dataset.features, *(["is_anomaly"] if dataset.has_labels else [])]
        for i in range(dataset.rows):
            yield [str(data["timestamps"][i]), *[float(v) if np.isfinite(v) else "" for v in data["values"][i]],
                   *([int(data["labels"][i])] if dataset.has_labels else [])]
    return csv_response(rows(), f"数据源-{dataset.id[:8]}.csv")


@app.post("/api/trainings", response_model=TrainingJob, status_code=202)
def create_training(config: TrainingConfig):
    dataset = dataset_or_404(config.dataset_id)
    if not app.state.slots.acquire(blocking=False):
        raise HTTPException(429, "任务队列已满（最多 2 个并行、4 个等待），请稍后重试。")
    job = None
    try:
        job = storage().add_training(config, dataset)
        future = app.state.executor.submit(execute_training, job.id, config, dataset)
    except Exception:
        app.state.slots.release()
        if job is not None:
            storage().update_training(job.id, status="failed", message="无法调度任务", error="训练调度失败。", completed_at=now())
        raise
    remember_future(job.id, future)
    return job


@app.get("/api/trainings")
def trainings():
    return {"items": storage().list_trainings()}


@app.get("/api/trainings/{identifier}", response_model=TrainingJob)
def training_detail(identifier: str):
    job = storage().get_training(identifier)
    if job is None:
        raise HTTPException(404, "训练任务不存在。")
    return job


@app.get("/api/models")
def models():
    return {"items": storage().list_models()}


@app.get("/api/models/{identifier}", response_model=ModelInfo)
def model_detail(identifier: str):
    return model_or_404(identifier)


@app.post("/api/models/{identifier}/publish", response_model=ModelInfo)
def publish_model(identifier: str):
    model_or_404(identifier)
    try:
        return storage().publish_model(identifier)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/models/{identifier}/disable", response_model=ModelInfo)
def disable_model(identifier: str):
    model_or_404(identifier)
    return storage().disable_model(identifier)


@app.post("/api/runs", response_model=Run, status_code=202)
def create_run(config: DetectionConfig):
    dataset = dataset_or_404(config.dataset_id)
    model = model_or_404(config.model_id)
    try:
        storage().validate_detection(config, dataset, model)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not app.state.slots.acquire(blocking=False):
        raise HTTPException(429, "任务队列已满（最多 2 个并行、4 个等待），请稍后重试。")
    run = None
    try:
        run = storage().add_run(config, dataset, model)
        with app.state.futures_lock:
            key = (model.id, config.stream_id) if config.stream_id is not None else None
            predecessor = app.state.stream_futures.get(key) if key is not None else None
            future = app.state.executor.submit(execute, run.id, config, dataset, predecessor)
            if key is not None:
                app.state.stream_futures[key] = future
    except Exception:
        app.state.slots.release()
        if run is not None:
            storage().update_run(run.id, status="failed", message="无法调度任务", error="检测调度失败。", completed_at=now())
        raise
    remember_future(run.id, future)
    return run


@app.get("/api/runs")
def runs(dataset_id: str | None = None):
    return {"items": storage().list_runs(dataset_id)}


@app.get("/api/runs/{identifier}", response_model=Run)
def run_detail(identifier: str):
    return run_or_404(identifier)


@app.get("/api/runs/{identifier}/series")
def series(identifier: str, start: int = 0, end: int | None = None,
           max_points: int = Query(default=1000, ge=100, le=2500), features: str | None = None):
    run, dataset, source, output = result(identifier)
    left, right = bounded_range(start, end, dataset.rows)
    names = selected_features(features, dataset.features)
    indices = peak_sample(output["scores"], left, right, max_points)
    mask = scored_mask(run, output)[indices]
    nullable = lambda array: [float(value) if scored and np.isfinite(value) else None for value, scored in zip(array, mask)]
    values, reference, contributions = {}, {}, {}
    for feature in names:
        column = dataset.features.index(feature)
        values[feature] = [float(value) if np.isfinite(value) else None for value in output["values"][indices, column]]
        reference[feature] = nullable(output["reference"][indices, column])
        contributions[feature] = nullable(output["contributions"][indices, column])
    return {"indices": indices.tolist(), "timestamps": source["timestamps"][indices].tolist(),
            "scores": nullable(output["scores"][indices]), "scored": mask.tolist(), "threshold": run.summary.threshold,
            "train_end": run.summary.train_end, "feature_names": names, "values": values,
            "reference": reference, "contributions": contributions, "flags": output["flags"][indices].tolist(),
            "labels": source["labels"][indices].tolist() if dataset.has_labels else None,
            "sampling": {"total_points": right - left, "returned_points": len(indices)}}


@app.get("/api/runs/{identifier}/heatmap")
def heatmap(identifier: str, start: int = 0, end: int | None = None,
            bins: int = Query(default=120, ge=20, le=250)):
    run, dataset, source, output = result(identifier)
    left, right = bounded_range(start, end, dataset.rows)
    edges = np.linspace(left, right, min(bins, right - left) + 1, dtype=int)
    indices = edges[:-1]
    mask = scored_mask(run, output)
    buckets = [mean_contributions(output["contributions"][a:b][mask[a:b]]).tolist() if mask[a:b].any()
               else [None] * len(dataset.features) for a, b in zip(edges[:-1], edges[1:])]
    values = [list(column) for column in zip(*buckets)]
    return {"features": dataset.features, "timestamps": source["timestamps"][indices].tolist(),
            "indices": indices.tolist(), "values": values, "method": run.summary.explanation_method,
            "range": {"start": left, "end": right}}


@app.get("/api/runs/{identifier}/events")
def events(identifier: str):
    completed_run(identifier)
    return {"items": storage().events(identifier)}


@app.get("/api/runs/{identifier}/insights", response_model=RunInsights)
def insights(identifier: str):
    run, dataset, source, output = result(identifier)
    return run_insights(run, storage().events(identifier), source, output)


@app.post("/api/runs/{identifier}/events/review")
def batch_review(identifier: str, annotation: BatchAnnotation):
    completed_run(identifier)
    changes = annotation.model_dump(exclude_unset=True, exclude={"event_ids"})
    updated = storage().annotate_batch(identifier, annotation.event_ids, changes)
    if updated is None:
        raise HTTPException(404, "选中的异常事件有不存在的条目；未更新任何事件。")
    return {"items": updated}


def event_or_404(identifier: str, event_id: str) -> dict:
    event = next((event for event in storage().events(identifier) if event["id"] == event_id), None)
    if event is None:
        raise HTTPException(404, "异常事件不存在。")
    return event


@app.get("/api/runs/{identifier}/events/{event_id}/explanation")
def explanation(identifier: str, event_id: str):
    run, dataset, source, output = result(identifier)
    event = event_or_404(identifier, event_id)
    interval = slice(event["start"], event["end"])
    scored = output["contributions"][interval][scored_mask(run, output)[interval]]
    contribution = mean_contributions(scored) if len(scored) else np.zeros(len(dataset.features))
    total = float(contribution.sum())
    order = np.argsort(-contribution, kind="stable")
    features = [{"name": dataset.features[j], "contribution": float(contribution[j]),
                 "share": float(contribution[j] / total) if total > 0 else 0.0, "rank": rank + 1}
                for rank, j in enumerate(order)]
    padding = max(30, event["length"])
    return {"event": event, "method": run.summary.explanation_method, "top_features": features,
            "context": {"start": max(0, event["start"] - padding), "end": min(dataset.rows, event["end"] + padding)},
            "notes": storage().result_notes(identifier)}


@app.patch("/api/runs/{identifier}/events/{event_id}", response_model=Event)
def annotate(identifier: str, event_id: str, annotation: Annotation):
    completed_run(identifier)
    changes = annotation.model_dump(exclude_unset=True)
    if not changes or any(value is None for value in changes.values()):
        raise HTTPException(422, "至少提交一个有效的审核状态或备注；字段不能为 null。")
    updated = storage().annotate(identifier, event_id, changes)
    if updated is None:
        raise HTTPException(404, "异常事件不存在。")
    return updated


@app.get("/api/runs/{identifier}/export.csv")
def analysis_csv(identifier: str):
    run, dataset, source, output = result(identifier)
    mask = scored_mask(run, output)
    event_ids = np.full(dataset.rows, "", dtype=object)
    statuses = np.full(dataset.rows, "", dtype=object)
    notes = np.full(dataset.rows, "", dtype=object)
    for event in storage().events(identifier):
        interval = slice(event["start"], event["end"])
        event_ids[interval], statuses[interval], notes[interval] = event["id"], event["status"], event["note"]
    def rows():
        yield ["timestamp", *dataset.features, "scorable", "score", "threshold", "is_anomaly", "model_id", "model_version", "event_id", "review_status", "review_note",
               *(["ground_truth"] if dataset.has_labels else [])]
        for i in range(dataset.rows):
            yield [str(source["timestamps"][i]), *[float(v) if np.isfinite(v) else "" for v in source["values"][i]],
                   int(mask[i]), float(output["scores"][i]) if mask[i] else "", run.summary.threshold,
                   int(output["flags"][i]) if mask[i] else "", run.model_id or "", run.model_version or "",
                   event_ids[i], statuses[i], notes[i], *([int(source["labels"][i])] if dataset.has_labels else [])]
    return csv_response(rows(), f"分析结果-{run.id[:8]}.csv")


def svg_chart(values: np.ndarray, title: str, threshold: float | None = None, train_end: int | None = None,
              second: np.ndarray | None = None) -> str:
    width, height, left, top, plot_width, plot_height = 960, 240, 70, 32, 865, 165
    combined = values if second is None else np.concatenate([values, second])
    finite = combined[np.isfinite(combined)]
    low, high = (float(finite.min()), float(finite.max())) if finite.size else (0., 1.)
    if threshold is not None:
        low, high = min(low, threshold), max(high, threshold)
    if high - low < 1e-12:
        low, high = low - .5, high + .5
    padding = (high - low) * .08
    low, high = low - padding, high + padding
    def x(index):
        return left + plot_width * index / max(1, len(values) - 1)
    def y(value):
        return top + plot_height * (high - float(value)) / (high - low)
    indices = peak_sample(np.nan_to_num(values, nan=0.), 0, len(values), 600)
    def polyline(array, color):
        paths, current, previous = [], [], None
        for i in indices:
            if not np.isfinite(array[i]) or (previous is not None and not np.isfinite(array[previous:i + 1]).all()):
                if current:
                    paths.append(current)
                current = []
            if np.isfinite(array[i]):
                current.append(f"{x(i):.2f},{y(array[i]):.2f}")
            previous = i
        if current:
            paths.append(current)
        return "".join(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.5"/>' for points in paths)
    train = ""
    if train_end is not None:
        train_width = x(train_end) - left
        train = f'<rect x="{left}" y="{top}" width="{train_width:.2f}" height="{plot_height}" fill="#eef2ff"/><text x="{left + 8}" y="{top + 16}" font-size="11" fill="#667085">历史训练段（不报警）</text>'
    threshold_line = ""
    if threshold is not None:
        threshold_line = f'<line x1="{left}" x2="{left + plot_width}" y1="{y(threshold):.2f}" y2="{y(threshold):.2f}" stroke="#e58428" stroke-dasharray="5 4"/><text x="{left + 8}" y="{y(threshold) - 5:.2f}" fill="#b85e0c" font-size="11">阈值 {threshold:.5g}</text>'
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title, quote=True)}" xmlns="http://www.w3.org/2000/svg"><title>{html.escape(title)}</title><rect width="{width}" height="{height}" fill="white"/>{train}<path d="M{left} {top}V{top + plot_height}H{left + plot_width}" fill="none" stroke="#d0d5dd"/>{threshold_line}{polyline(values, "#5267d9")}{polyline(second, "#14a594") if second is not None else ""}<g font-family="sans-serif" font-size="11" fill="#667085"><text x="8" y="{top + 8}">{high:.4g}</text><text x="8" y="{top + plot_height}">{low:.4g}</text><text x="{left}" y="222">0</text><text x="{left + plot_width - 35}" y="222">{len(values) - 1}</text><text x="{left}" y="18">{html.escape(title)}</text></g></svg>'''


@app.get("/api/runs/{identifier}/report.html", response_class=HTMLResponse)
def report(identifier: str):
    run, dataset, source, output = result(identifier)
    escape = lambda value: html.escape(str(value), quote=True)
    summary = run.summary
    metrics = summary.metrics
    config_rows = "".join(f"<tr><th>{escape(k)}</th><td>{escape(v)}</td></tr>" for k, v in run.config.items())
    mask = scored_mask(run, output)
    scores = np.where(mask, output["scores"], np.nan)
    provenance = f"模型 {escape(run.model_name)} · v{run.model_version} · {escape(run.model_id)}" if run.model_id else "历史训练检测记录（未关联已保存模型）"
    notes = "".join(f"<li>{escape(note)}</li>" for note in storage().result_notes(identifier))
    event_rows = "".join(f'''<tr><td>{escape(event['id'])}</td><td>[{event['start']}, {event['end']})<small>{escape(event['start_time'])}<br>至 {escape(event['end_time'])}（最后一个样本）</small></td><td>{event['length']} / {event['anomaly_points']}</td><td>{event['peak_score']:.5g}</td><td>{escape(event['top_feature'])}</td><td>{escape(event['severity'])}</td><td>{escape(event['status'])}</td><td class="note">{escape(event['note'])}</td></tr>''' for event in storage().events(identifier))
    metric_section = "<p>没有真实标签或没有可评分样本，不计算监督评估指标。</p>" if metrics is None else "<div class=metrics>" + "".join(f"<span>{escape(key)} <strong>{value:.4f}</strong></span>" for key, value in metrics.model_dump().items()) + "</div><p>指标仅基于可评分样本的原始逐点预测；预热点不可评分，未做 point adjustment。事件召回按真实连续标签事件与检测事件是否重叠计算。</p>"
    features = []
    for j, name in enumerate(dataset.features):
        features.append(f"<section><h3>{escape(name)}</h3>" + svg_chart(output["values"][:, j], f"{name} · 蓝色为填充后观测，绿色为模型参考；预热段参考不可用", second=np.where(mask, output["reference"][:, j], np.nan)) + "</section>")
    quality = "".join(f"<li>{escape(warning)}</li>" for warning in dataset.quality.warnings)
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'"><title>ChronoLens · {escape(dataset.name)}</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#f4f6fa;color:#243047;font:14px/1.65 system-ui,'Microsoft YaHei',sans-serif}}main{{max-width:1160px;margin:40px auto;padding:0 24px}}header,section{{background:#fff;border:1px solid #e4e8f0;border-radius:14px;padding:24px;margin-bottom:20px}}h1{{margin:6px 0;font-size:28px}}h2{{font-size:19px}}h3{{font-size:15px}}small{{display:block;color:#768197}}.tag{{color:#5267d9;font-weight:700;letter-spacing:2px}}.metrics{{display:flex;gap:24px;flex-wrap:wrap}}.metrics span{{display:grid;min-width:125px;color:#667085}}.metrics strong{{font-size:26px;color:#243047}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:12px 8px;text-align:left;border-bottom:1px solid #edf0f5;vertical-align:top;overflow-wrap:anywhere}}th{{background:#f7f9fc}}svg{{width:100%;height:auto}}.scroll{{overflow:auto}}.note{{white-space:pre-wrap;min-width:160px;max-width:280px}}li{{margin:8px 0}}footer{{color:#667085;text-align:center;padding:20px}}@media print{{body{{background:white}}main{{margin:0}}section{{break-inside:avoid}}}}</style></head><body><main><header><div class="tag">CHRONOLENS / 检测报告</div><h1>{escape(dataset.name)}</h1><p>{escape(dataset.description)}</p><p>{provenance}</p><small>来源：{escape(dataset.source)} · 生成于 {escape(now())} · 任务 {escape(run.id)}</small></header><section><h2>检测摘要</h2><div class="metrics"><span>可评分异常点<strong>{summary.anomaly_points}</strong></span><span>可评分异常比例<strong>{summary.anomaly_ratio:.2%}</strong></span><span>异常事件<strong>{summary.event_count}</strong></span><span>已保存校准阈值<strong>{summary.threshold:.5g}</strong></span><span>检测耗时<strong>{summary.duration_ms / 1000:.2f}s</strong></span></div><p>{dataset.rows} 行 · {summary.n_features} 变量 · 算法 {escape(run.algorithm)} · 可评分 {summary.scored_points} 点 · 预热/不可评分 {summary.warmup_points} 点</p><p>解释方法：{escape(summary.explanation_method)}</p><ul>{notes}</ul></section><section><h2>异常分数概览</h2><p>空白段不可评分/预热，不代表正常。新检测不拟合模型、不调整阈值。</p>{svg_chart(scores, '异常分数 · 保留分桶峰值的真实样本', summary.threshold, summary.train_end)}</section><section><h2>评估指标</h2>{metric_section}</section><section><h2>异常事件与人工审核</h2><p>索引区间左闭右开；合并间隙内可能包含正常点。人工审核不更改模型分数或真实标签。</p><div class="scroll"><table><thead><tr><th>事件</th><th>索引 / 时间</th><th>跨度 / 异常点</th><th>峰值</th><th>主要偏差变量</th><th>相对严重度</th><th>审核</th><th>备注</th></tr></thead><tbody>{event_rows or '<tr><td colspan="8">未发现达到最小事件跨度的异常事件。</td></tr>'}</tbody></table></div></section>{''.join(features)}<section><h2>检测配置</h2><table>{config_rows}</table><h3>数据质量</h3><ul>{quality or '<li>导入时未发现缺失、重复时间或常量特征。</li>'}</ul><p>原始缺失值在图中由已保存模型的拟合中位数填充；CSV 导出保留原始缺失单元格。</p></section><footer>本报告完全离线，不加载外部脚本、字体或网络资源。特征偏差不是因果根因结论。</footer></main></body></html>'''
    return HTMLResponse(document, headers={"Content-Disposition": f"attachment; filename=\"chronolens-report.html\"; filename*=UTF-8''{quote('分析报告-' + run.id[:8] + '.html')}"})


@app.api_route("/api", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"], include_in_schema=False)
@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"], include_in_schema=False)
def unknown_api(path: str = ""):
    raise HTTPException(404, "API 路由不存在。")


@app.get("/{path:path}", include_in_schema=False)
def frontend(path: str):
    dist = PROJECT_ROOT / "frontend" / "dist"
    candidate = (dist / path).resolve()
    if not candidate.is_relative_to(dist.resolve()):
        raise HTTPException(404, "资源不存在。")
    if path and candidate.is_file():
        return FileResponse(candidate)
    index = dist / "index.html"
    if not index.is_file():
        raise HTTPException(503, "前端尚未构建，请运行启动脚本或使用开发模式。")
    return FileResponse(index)
