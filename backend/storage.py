import csv
import io
import json
import os
import sqlite3
import threading
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import Dataset, Quality, Run, RunConfig

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
    if len(records) < 120:
        raise ValueError("数据至少需要 120 行。")
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
            raise ValueError(f"列「{name}」完全缺失，无法作为数值特征。")
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
    reserved = {"timestamp", "is_anomaly", "score", "threshold", "event_id", "review_status", "review_note", "ground_truth"}
    if any(f in reserved for f in features):
        raise ValueError("数值特征名与系统保留字段冲突，请重命名该列。")
    values = np.column_stack(columns)
    missing_count = int(np.isnan(values).sum())
    if missing_count:
        warnings.append(f"存在 {missing_count} 个缺失数值；分析仅使用历史拟合段中位数填充，不做未来插值。")
    if ignored:
        warnings.append("已忽略非数值列：" + "、".join(ignored))
    if constant:
        warnings.append("常量特征不提供有效变化信息：" + "、".join(constant))
    quality = Quality(missing_cells=missing_count, missing_ratio=missing_count / values.size,
                      duplicate_timestamps=duplicates, constant_features=constant,
                      ignored_columns=ignored, warnings=warnings)
    return {"timestamps": timestamps, "values": values, "labels": labels}, {"features": features, "quality": quality}


class Storage:
    def __init__(self, root: Path, array_cache_bytes: int = 128 * 1024 * 1024):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.database = root / "chronolens.sqlite3"
        self.lock = threading.RLock()
        self.array_lock = threading.RLock()
        self.array_cache_bytes = max(0, array_cache_bytes)
        self.array_cache_size = 0
        self.array_cache = OrderedDict()
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, payload TEXT NOT NULL, result_meta TEXT);
                CREATE TABLE IF NOT EXISTS events (run_id TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'unreviewed', note TEXT NOT NULL DEFAULT '', PRIMARY KEY(run_id,id));
                CREATE INDEX IF NOT EXISTS runs_dataset ON runs(dataset_id);
            """)
            for row in db.execute("SELECT id,payload FROM runs").fetchall():
                run = json.loads(row[1])
                if run["status"] in ("queued", "running"):
                    run.update(status="failed", error="服务重启中断了分析，请重新运行。", message="分析已中断", completed_at=now())
                    db.execute("UPDATE runs SET payload=? WHERE id=?", (json.dumps(run, ensure_ascii=False), row[0]))

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

    def add_run(self, config: RunConfig, dataset: Dataset) -> Run:
        run = Run(id=uuid.uuid4().hex, dataset_id=dataset.id, dataset_name=dataset.name,
                  algorithm=config.algorithm, config=config, status="queued", progress=0,
                  message="等待分析资源", created_at=now())
        with self.lock, self.connect() as db:
            db.execute("INSERT INTO runs(id,dataset_id,payload) VALUES (?,?,?)", (run.id, dataset.id, run.model_dump_json()))
        return run

    def get_run(self, identifier: str) -> Run | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM runs WHERE id=?", (identifier,)).fetchone()
        return Run.model_validate_json(row[0]) if row else None

    def list_runs(self, dataset_id: str | None = None) -> list[Run]:
        query = "SELECT payload FROM runs" + (" WHERE dataset_id=?" if dataset_id else "") + " ORDER BY rowid DESC"
        with self.connect() as db:
            rows = db.execute(query, (dataset_id,) if dataset_id else ()).fetchall()
        return [Run.model_validate_json(row[0]) for row in rows]

    def update_run(self, identifier: str, **changes) -> Run:
        with self.lock, self.connect() as db:
            row = db.execute("SELECT payload FROM runs WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError(identifier)
            run = Run.model_validate_json(row[0])
            updated = Run.model_validate({**run.model_dump(), **changes})
            db.execute("UPDATE runs SET payload=? WHERE id=?", (updated.model_dump_json(), identifier))
        return updated

    def finish_run(self, identifier: str, arrays: dict, events: list[dict], summary: dict, notes: list[str]):
        self.save_arrays("run", identifier, arrays)
        with self.lock, self.connect() as db:
            row = db.execute("SELECT payload FROM runs WHERE id=?", (identifier,)).fetchone()
            run = Run.model_validate_json(row[0])
            completed = Run.model_validate({**run.model_dump(), "status": "completed", "progress": 100,
                                           "message": "分析完成；" + "；".join(notes), "summary": summary,
                                           "completed_at": now()})
            db.execute("UPDATE runs SET payload=?,result_meta=? WHERE id=?",
                       (completed.model_dump_json(), json.dumps({"notes": notes}, ensure_ascii=False), identifier))
            db.executemany("INSERT INTO events(run_id,id,payload) VALUES (?,?,?)",
                           [(identifier, event["id"], json.dumps(event, ensure_ascii=False)) for event in events])

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
