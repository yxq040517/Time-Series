import csv
import io
import errno
import hashlib
import json
import os
import sqlite3
import threading
import uuid
import time
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .model_assets import load_model, save_model
from .schemas import Dataset, DetectionConfig, ModelInfo, Quality, Run, TrainingConfig, TrainingJob

MAX_UPLOAD = 25 * 1024 * 1024
TIME_NAMES = ("timestamp", "time", "datetime", "date", "时间")
LABEL_NAMES = ("is_anomaly", "label", "anomaly", "标签")
MISSING = {"", "na", "nan", "null", "none", "n/a"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv(content: bytes, timestamp_column: str | None = None,
              label_column: str | None = None) -> tuple[dict, dict]:
    """Validate raw headers before pandas can silently rename duplicates."""
    if len(content) > MAX_UPLOAD:
        raise ValueError("CSV 不得超过 25 MiB。")
    if b"\x00" in content:
        raise ValueError("CSV 含有不支持的空字节。")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = content.decode("gb18030")
        except UnicodeDecodeError as exc:
            raise ValueError("CSV 编码须为 UTF-8 或 GB18030。") from exc
    try:
        reader = csv.reader(io.StringIO(text), strict=True)
        headers = next(reader)
        if not headers or any(not h.strip() for h in headers):
            raise ValueError("CSV 列名不能为空。")
        if len(set(headers)) != len(headers) or len({h.strip() for h in headers}) != len(headers):
            raise ValueError("CSV 含有重复列名。")
        if len(headers) > 256 or any(len(h) > 200 or any(ord(c) < 32 for c in h) for h in headers):
            raise ValueError("CSV 列名过长、列数过多或含控制字符。")
        records = []
        for row in reader:
            if not row:
                continue
            if len(row) != len(headers):
                raise ValueError("CSV 行的列数与表头不一致。")
            records.append(row)
            if len(records) > 100000:
                raise ValueError("CSV 最多支持 100000 行。")
    except (csv.Error, StopIteration) as exc:
        raise ValueError("CSV 为空或格式无效。") from exc
    if not records:
        raise ValueError("数据至少需要 1 行。")
    frame = pd.DataFrame(records, columns=headers, dtype=str)
    normalized = {h.strip().lower(): h for h in headers}
    if timestamp_column and timestamp_column not in headers:
        raise ValueError("指定的时间列不存在。")
    if label_column and label_column not in headers:
        raise ValueError("指定的标签列不存在。")
    time_col = timestamp_column or next((normalized[n] for n in TIME_NAMES if n in normalized), None)
    label_col = label_column or next((normalized[n] for n in LABEL_NAMES if n in normalized), None)
    if time_col is not None and time_col == label_col:
        raise ValueError("时间列和标签列不能相同。")
    warnings = []
    duplicates = 0
    if time_col:
        try:
            times = pd.to_datetime(frame[time_col], errors="raise", utc=True, format="mixed")
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("时间列包含无效时间；请使用可解析的完整日期时间。") from exc
        if times.isna().any():
            raise ValueError("时间列不能缺失。")
        if not times.is_monotonic_increasing:
            raise ValueError("时间列必须按升序排列；系统不会自动重排时序。")
        duplicates = int(times.duplicated().sum())
        timestamps = np.asarray([t.isoformat() for t in times], dtype=str)
        if not frame[time_col].str.contains(r"(?:[zZ]|[+-]\d{2}:?\d{2})\s*$", regex=True).all():
            warnings.append("未显式带时区的时间戳按 UTC 解析；界面按浏览器本地时区显示。建议使用带 +08:00 或 Z 的 ISO 时间戳。")
        if duplicates:
            warnings.append(f"存在 {duplicates} 个重复时间戳，保留原始行顺序。")
    else:
        timestamps = np.arange(len(frame)).astype(str)
        warnings.append("未发现时间列；横轴使用从 0 开始的样本序号。")
    labels = np.empty(0, dtype=np.int8)
    if label_col:
        try:
            numeric_labels = pd.to_numeric(frame[label_col].str.strip(), errors="raise").to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("标签列只能包含非缺失的 0 或 1。") from exc
        if not np.isin(numeric_labels, [0, 1]).all():
            raise ValueError("标签列只能包含非缺失的 0 或 1。")
        labels = numeric_labels.astype(np.int8)
    features, columns, ignored, constant = [], [], [], []
    for name in headers:
        if name in (time_col, label_col):
            continue
        tokens = frame[name].str.strip()
        missing = tokens.str.lower().isin(MISSING)
        if missing.all():
            features.append(name)
            columns.append(np.full(len(frame), np.nan))
            continue
        parsed = pd.to_numeric(tokens.mask(missing), errors="coerce").to_numpy(dtype=float)
        valid = ~missing.to_numpy()
        numeric = np.isfinite(parsed[valid])
        # A partly numeric column is a damaged feature, not a silently discarded text column.
        special_numeric = tokens.str.lower().isin({"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}).any()
        numeric_syntax = tokens.str.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?").any()
        if not numeric.any() and not special_numeric and not numeric_syntax:
            ignored.append(name)
            continue
        if not numeric.all():
            raise ValueError(f"数值列「{name}」含无效数值或无限值。")
        features.append(name)
        columns.append(parsed)
        if np.unique(parsed[valid]).size == 1:
            constant.append(name)
    if len(features) < 2:
        raise ValueError("至少需要 2 个有效数值特征。")
    if len(features) > 64:
        raise ValueError("最多支持 64 个数值特征。")
    # Canonical exported metadata fields must never shadow a feature.
    reserved = {"timestamp", "is_anomaly", "score", "threshold", "scorable", "event_id", "review_status", "review_note", "ground_truth", "model_id", "model_version"}
    if any(f in reserved for f in features):
        raise ValueError("数值特征名与系统保留字段冲突，请重命名该列。")
    values = np.column_stack(columns)
    missing_count = int(np.isnan(values).sum())
    if missing_count:
        warnings.append(f"存在 {missing_count} 个缺失数值；推理使用已保存模型的拟合中位数填充，不做未来插值。")
    if ignored:
        warnings.append("已忽略非数值列：" + "、".join(ignored))
    if constant:
        warnings.append("常量特征不提供有效变化信息：" + "、".join(constant))
    quality = Quality(missing_cells=missing_count, missing_ratio=missing_count / values.size,
                      duplicate_timestamps=duplicates, constant_features=constant,
                      ignored_columns=ignored, warnings=warnings)
    return {"timestamps": timestamps, "values": values, "labels": labels}, {"features": features, "quality": quality}


class Storage:
    def __init__(self, root: Path, array_cache_bytes: int = 128 * 1024 * 1024, *, recover_interrupted: bool = True):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.database = root / "chronolens.sqlite3"
        self.lock = threading.RLock()
        self.array_lock = threading.RLock()
        self.array_cache_bytes = max(0, array_cache_bytes)
        self.array_cache_size = 0
        self.array_cache = OrderedDict()
        self.stream_locks: dict[tuple[str, str], threading.Lock] = {}
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, payload TEXT NOT NULL, result_meta TEXT);
                CREATE TABLE IF NOT EXISTS events (run_id TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'unreviewed', note TEXT NOT NULL DEFAULT '', PRIMARY KEY(run_id,id));
                CREATE INDEX IF NOT EXISTS runs_dataset ON runs(dataset_id);
                CREATE TABLE IF NOT EXISTS trainings (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS models (id TEXT PRIMARY KEY, name TEXT NOT NULL, version INTEGER NOT NULL,
                  payload TEXT NOT NULL, artifact_sha256 TEXT NOT NULL, UNIQUE(name,version));
                CREATE TABLE IF NOT EXISTS streams (model_id TEXT NOT NULL, stream_id TEXT NOT NULL,
                  payload TEXT NOT NULL, PRIMARY KEY(model_id,stream_id));
            """)
            if recover_interrupted:
                for row in db.execute("SELECT id,payload FROM runs").fetchall():
                    run = json.loads(row[1])
                    if run["status"] in ("queued", "running"):
                        run.update(status="failed", error="服务重启中断了分析，请重新运行。", message="分析已中断", completed_at=now())
                        db.execute("UPDATE runs SET payload=? WHERE id=?", (json.dumps(run, ensure_ascii=False), row[0]))
                for row in db.execute("SELECT id,payload FROM trainings").fetchall():
                    job = json.loads(row[1])
                    if job["status"] in ("queued", "running"):
                        job.update(status="failed", error="服务重启中断训练，请重新提交。", message="训练已中断", completed_at=now())
                        db.execute("UPDATE trainings SET payload=? WHERE id=?", (json.dumps(job, ensure_ascii=False), row[0]))

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def array_path(self, kind: str, identifier: str) -> Path:
        # Never accept user filenames, path components or arbitrary identifiers.
        validated = uuid.UUID(identifier).hex
        return self.root / f"{kind}-{validated}.npz"

    def save_arrays(self, kind: str, identifier: str, arrays: dict):
        path = self.array_path(kind, identifier)
        temporary = path.with_name(f"{path.stem}-{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            with self.array_lock:
                os.replace(temporary, path)
                cached = self.array_cache.pop(path, None)
                if cached is not None:
                    self.array_cache_size -= cached[2]
        finally:
            temporary.unlink(missing_ok=True)

    def load_arrays(self, kind: str, identifier: str) -> dict:
        path = self.array_path(kind, identifier)
        # Serialize cache misses and atomic replacement so concurrent chart requests
        # decompress the same archive once. Cached arrays are immutable to callers.
        with self.array_lock:
            stat = path.stat()
            fingerprint = (stat.st_mtime_ns, stat.st_size)
            cached = self.array_cache.get(path)
            if cached is not None and cached[0] == fingerprint:
                self.array_cache.move_to_end(path)
                return dict(cached[1])
            if cached is not None:
                del self.array_cache[path]
                self.array_cache_size -= cached[2]
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            for array in arrays.values():
                array.setflags(write=False)
            size = sum(array.nbytes for array in arrays.values())
            after = path.stat()
            # An external process may replace the file while we read it. Return
            # this consistent snapshot without caching under an old fingerprint.
            stable = fingerprint == (after.st_mtime_ns, after.st_size)
            if stable and size <= self.array_cache_bytes and self.array_cache_bytes > 0:
                while self.array_cache and (self.array_cache_size + size > self.array_cache_bytes
                                            or len(self.array_cache) >= 16):
                    _, evicted = self.array_cache.popitem(last=False)
                    self.array_cache_size -= evicted[2]
                self.array_cache[path] = (fingerprint, arrays, size)
                self.array_cache_size += size
            return dict(arrays)

    def add_dataset(self, content: bytes, name: str, source: str, description: str = "",
                    timestamp_column: str | None = None, label_column: str | None = None) -> Dataset:
        arrays, parsed = parse_csv(content, timestamp_column, label_column)
        name = name.strip() or "未命名数据集"
        if len(name) > 200:
            raise ValueError("数据集名称最多 200 字。")
        dataset = Dataset(id=uuid.uuid4().hex, name=name, source=source, created_at=now(),
                          rows=len(arrays["values"]), features=parsed["features"],
                          start_time=str(arrays["timestamps"][0]), end_time=str(arrays["timestamps"][-1]),
                          has_labels=bool(arrays["labels"].size), description=description, quality=parsed["quality"])
        self.save_arrays("dataset", dataset.id, arrays)
        with self.lock, self.connect() as db:
            db.execute("INSERT INTO datasets VALUES (?,?)", (dataset.id, dataset.model_dump_json()))
        return dataset

    def get_dataset(self, identifier: str) -> Dataset | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM datasets WHERE id=?", (identifier,)).fetchone()
        return Dataset.model_validate_json(row[0]) if row else None

    def list_datasets(self) -> list[Dataset]:
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM datasets ORDER BY rowid DESC").fetchall()
        return [Dataset.model_validate_json(row[0]) for row in rows]

    @staticmethod
    def decode_run(payload: str) -> Run:
        data = json.loads(payload)
        summary = data.get("summary")
        if summary is not None:
            summary.setdefault("scored_points", summary["points"] - (summary.get("train_end") or 0))
            summary.setdefault("warmup_points", 0)
        return Run.model_validate(data)

    @staticmethod
    def validate_detection(config: DetectionConfig, dataset: Dataset, model: ModelInfo):
        if config.dataset_id != dataset.id or config.model_id != model.id:
            raise ValueError("数据集或模型标识不匹配。")
        if model.status != "published":
            raise ValueError("仅已发布模型可用于检测；请先发布模型。")
        missing = [name for name in model.features if name not in dataset.features]
        unexpected = [name for name in dataset.features if name not in model.features]
        invalid = [name for name in model.features if name in dataset.quality.ignored_columns]
        if missing or unexpected or invalid:
            raise ValueError(f"特征契约不匹配：缺少 {missing}；多余 {unexpected}；非数值 {invalid}。")
        if config.stream_id is not None and model.algorithm != "temporal":
            raise ValueError("只有时序模型支持 stream_id。")

    def add_training(self, config: TrainingConfig, dataset: Dataset) -> TrainingJob:
        if config.dataset_id != dataset.id:
            raise ValueError("训练数据集标识不匹配。")
        job = TrainingJob(id=uuid.uuid4().hex, dataset_id=dataset.id, dataset_name=dataset.name,
                          config=config, status="queued", progress=0, message="等待训练资源", created_at=now())
        with self.lock, self.connect() as db:
            db.execute("INSERT INTO trainings VALUES (?,?)", (job.id, job.model_dump_json()))
        return job

    def get_training(self, identifier: str) -> TrainingJob | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM trainings WHERE id=?", (identifier,)).fetchone()
        return TrainingJob.model_validate_json(row[0]) if row else None

    def list_trainings(self) -> list[TrainingJob]:
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM trainings ORDER BY rowid DESC").fetchall()
        return [TrainingJob.model_validate_json(row[0]) for row in rows]

    def update_training(self, identifier: str, **changes) -> TrainingJob:
        with self.lock, self.connect() as db:
            row = db.execute("SELECT payload FROM trainings WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError(identifier)
            job = TrainingJob.model_validate_json(row[0])
            updated = TrainingJob.model_validate({**job.model_dump(), **changes})
            db.execute("UPDATE trainings SET payload=? WHERE id=?", (updated.model_dump_json(), identifier))
        return updated

    def model_path(self, identifier: str) -> Path:
        return self.root / f"model-{uuid.UUID(identifier).hex}.joblib"

    def finish_training(self, identifier: str, asset: dict, summary: dict, notes: list[str]) -> ModelInfo:
        model_id = uuid.uuid4().hex
        path = self.model_path(model_id)
        try:
            checksum = save_model(path, asset)
            with self.lock, self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT payload FROM trainings WHERE id=?", (identifier,)).fetchone()
                if row is None:
                    raise KeyError(identifier)
                job = TrainingJob.model_validate_json(row[0])
                if job.status not in ("queued", "running"):
                    raise ValueError("训练任务已结束，不能重复生成模型。")
                dataset = self.get_dataset(job.dataset_id)
                if dataset is None or asset["algorithm"] != job.config.algorithm or asset["features"] != dataset.features:
                    raise ValueError("训练产物与任务的算法或特征契约不匹配。")
                version = db.execute("SELECT COALESCE(MAX(version),0)+1 FROM models WHERE name=?", (job.config.name,)).fetchone()[0]
                model = ModelInfo(id=model_id, name=job.config.name, version=version, algorithm=asset["algorithm"],
                                  features=asset["features"], status="ready", created_at=now(), training_id=identifier,
                                  dataset_id=job.dataset_id, dataset_name=job.dataset_name,
                                  fit_start=asset["fit_start"], fit_end=asset["fit_end"], calibration_end=asset["calibration_end"],
                                  threshold=float(asset["threshold"]), window=int(asset["window"]), training_config=job.config,
                                  training_summary=summary, notes=notes, format_version=asset["format_version"],
                                  sklearn_version=asset["sklearn_version"])
                completed = TrainingJob.model_validate({**job.model_dump(), "status": "completed", "progress": 100,
                                                       "message": "训练完成，模型尚未发布", "completed_at": now(), "model_id": model_id})
                db.execute("INSERT INTO models VALUES (?,?,?,?,?)", (model.id, model.name, model.version, model.model_dump_json(), checksum))
                db.execute("UPDATE trainings SET payload=? WHERE id=?", (completed.model_dump_json(), identifier))
            return model
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def get_model(self, identifier: str) -> ModelInfo | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM models WHERE id=?", (identifier,)).fetchone()
        return ModelInfo.model_validate_json(row[0]) if row else None

    def list_models(self) -> list[ModelInfo]:
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM models ORDER BY rowid DESC").fetchall()
        return [ModelInfo.model_validate_json(row[0]) for row in rows]

    def load_model_asset(self, identifier: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT artifact_sha256 FROM models WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError(identifier)
        return load_model(self.model_path(identifier), row[0])

    def _model_status(self, identifier: str, status: str) -> ModelInfo:
        with self.lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM models WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError(identifier)
            model = ModelInfo.model_validate_json(row[0])
            updated = ModelInfo.model_validate({**model.model_dump(), "status": status})
            db.execute("UPDATE models SET payload=? WHERE id=?", (updated.model_dump_json(), identifier))
        return updated

    def publish_model(self, identifier: str) -> ModelInfo:
        self.load_model_asset(identifier)
        return self._model_status(identifier, "published")

    def disable_model(self, identifier: str) -> ModelInfo:
        return self._model_status(identifier, "disabled")

    @contextmanager
    def stream_lock(self, model_id: str, stream_id: str):
        key = (model_id, stream_id)
        with self.lock:
            lock = self.stream_locks.setdefault(key, threading.Lock())
        digest = hashlib.sha256(f"{model_id}\0{stream_id}".encode("utf-8")).hexdigest()
        # The advisory lock also protects another Storage instance/process. Keeping
        # lock files avoids unlink/recreate races; state itself remains in SQLite.
        with lock, (self.root / f"stream-{digest}.lock").open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as exc:
                        if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                            raise
                        time.sleep(.05)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def get_stream(self, model_id: str, stream_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM streams WHERE model_id=? AND stream_id=?", (model_id, stream_id)).fetchone()
        return json.loads(row[0]) if row else None

    def add_run(self, config: DetectionConfig, dataset: Dataset, model: ModelInfo) -> Run:
        current_model = self.get_model(model.id)
        if current_model is None:
            raise ValueError("模型不存在。")
        model = current_model
        self.validate_detection(config, dataset, model)
        run = Run(id=uuid.uuid4().hex, dataset_id=dataset.id, dataset_name=dataset.name,
                  algorithm=model.algorithm, config=config.model_dump(), status="queued", progress=0,
                  model_id=model.id, model_name=model.name, model_version=model.version,
                  message="等待推理资源", created_at=now())
        with self.lock, self.connect() as db:
            db.execute("INSERT INTO runs(id,dataset_id,payload) VALUES (?,?,?)", (run.id, dataset.id, run.model_dump_json()))
        return run

    def get_run(self, identifier: str) -> Run | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM runs WHERE id=?", (identifier,)).fetchone()
        return self.decode_run(row[0]) if row else None

    def list_runs(self, dataset_id: str | None = None) -> list[Run]:
        query = "SELECT payload FROM runs" + (" WHERE dataset_id=?" if dataset_id else "") + " ORDER BY rowid DESC"
        with self.connect() as db:
            rows = db.execute(query, (dataset_id,) if dataset_id else ()).fetchall()
        return [self.decode_run(row[0]) for row in rows]

    def update_run(self, identifier: str, **changes) -> Run:
        with self.lock, self.connect() as db:
            row = db.execute("SELECT payload FROM runs WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError(identifier)
            run = self.decode_run(row[0])
            updated = Run.model_validate({**run.model_dump(), **changes})
            db.execute("UPDATE runs SET payload=? WHERE id=?", (updated.model_dump_json(), identifier))
        return updated

    def finish_run(self, identifier: str, arrays: dict, events: list[dict], summary: dict, notes: list[str],
                   stream_update: tuple[str, str, dict] | None = None):
        with self.lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM runs WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError(identifier)
            run = self.decode_run(row[0])
            if run.status not in ("queued", "running"):
                raise ValueError("检测任务已结束，结果不可覆盖。")
            if run.model_id is not None and "scored" not in arrays:
                raise ValueError("模型检测结果必须包含可评分掩码。")
            self.save_arrays("run", identifier, arrays)
            completed = Run.model_validate({**run.model_dump(), "status": "completed", "progress": 100,
                                           "message": "分析完成；" + "；".join(notes), "summary": summary,
                                           "completed_at": now()})
            db.execute("UPDATE runs SET payload=?,result_meta=? WHERE id=?",
                       (completed.model_dump_json(), json.dumps({"notes": notes}, ensure_ascii=False), identifier))
            db.executemany("INSERT INTO events(run_id,id,payload) VALUES (?,?,?)",
                           [(identifier, event["id"], json.dumps(event, ensure_ascii=False)) for event in events])
            if stream_update is not None:
                model_id, stream_id, state = stream_update
                if run.model_id != model_id or run.config.get("stream_id") != stream_id:
                    raise ValueError("流状态与检测任务不匹配。")
                db.execute("INSERT INTO streams VALUES (?,?,?) ON CONFLICT(model_id,stream_id) DO UPDATE SET payload=excluded.payload",
                           (model_id, stream_id, json.dumps(state, ensure_ascii=False, allow_nan=False)))

    def result_notes(self, identifier: str) -> list[str]:
        with self.connect() as db:
            row = db.execute("SELECT result_meta FROM runs WHERE id=?", (identifier,)).fetchone()
        return json.loads(row[0])["notes"] if row and row[0] else []

    def events(self, identifier: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT payload,status,note FROM events WHERE run_id=? ORDER BY id", (identifier,)).fetchall()
        return [{**json.loads(row[0]), "status": row[1], "note": row[2]} for row in rows]

    def annotate(self, run_id: str, event_id: str, changes: dict) -> dict | None:
        with self.lock, self.connect() as db:
            row = db.execute("SELECT payload,status,note FROM events WHERE run_id=? AND id=?", (run_id, event_id)).fetchone()
            if row is None:
                return None
            status = changes.get("status", row[1])
            note = changes.get("note", row[2])
            db.execute("UPDATE events SET status=?,note=? WHERE run_id=? AND id=?", (status, note, run_id, event_id))
        return {**json.loads(row[0]), "status": status, "note": note}

    def annotate_batch(self, run_id: str, event_ids: list[str], changes: dict) -> list[dict] | None:
        with self.lock, self.connect() as db:
            # Validate the entire selection before writing any event, in one transaction.
            db.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _ in event_ids)
            rows = db.execute(f"SELECT id,payload,status,note FROM events WHERE run_id=? AND id IN ({placeholders})",
                              (run_id, *event_ids)).fetchall()
            if len(rows) != len(event_ids):
                return None
            selected = {row["id"]: row for row in rows}
            updated = []
            for event_id in event_ids:
                row = selected[event_id]
                status, note = changes["status"], changes.get("note", row["note"])
                db.execute("UPDATE events SET status=?,note=? WHERE run_id=? AND id=?",
                           (status, note, run_id, event_id))
                updated.append({**json.loads(row["payload"]), "status": status, "note": note})
        return updated
