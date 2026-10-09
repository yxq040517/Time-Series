"""Read-only diagnostics over original observations and completed detection outputs."""
from collections import Counter

import numpy as np
import pandas as pd

from .schemas import Dataset, Run


def sampling_profile(timestamps: np.ndarray) -> dict:
    if np.array_equal(timestamps, np.arange(len(timestamps)).astype(str)):
        return {"mode": "index", "median_interval_seconds": None,
                "irregular_intervals": 0, "duplicate_intervals": 0}
    times = pd.to_datetime(timestamps, utc=True, format="mixed")
    # Differencing unsigned nanoseconds first preserves small intervals and avoids
    # signed overflow for valid dates spanning both sides of the Unix epoch.
    intervals = np.diff(times.asi8.astype(np.uint64)).astype(np.float64) / 1e9
    positive = intervals[intervals > 0]
    median = float(np.median(positive)) if positive.size else None
    irregular = int((~np.isclose(positive, median, rtol=.01, atol=0)).sum()) if median is not None else 0
    return {"mode": "timestamp", "median_interval_seconds": median,
            "irregular_intervals": irregular, "duplicate_intervals": int((intervals == 0).sum())}


def dataset_profile(dataset: Dataset, source: dict) -> dict:
    values = source["values"]
    valid = np.isfinite(values)
    statistics = []
    for j, name in enumerate(dataset.features):
        column = values[valid[:, j], j]
        entry = {"name": name, "count": len(column), "missing": dataset.rows - len(column),
                 "missing_ratio": (dataset.rows - len(column)) / dataset.rows}
        if not column.size:
            entry.update({key: None for key in ("min", "max", "mean", "std", "p25", "median", "p75")})
        else:
            magnitude = float(np.max(np.abs(column)))
            scaled = column / magnitude if magnitude > 0 else column
            p25, median, p75 = np.quantile(scaled, [.25, .5, .75])
            entry.update(min=float(column.min()), max=float(column.max()),
                         mean=float(scaled.mean() * magnitude), std=float(scaled.std() * magnitude),
                         p25=float(p25 * magnitude), median=float(median * magnitude), p75=float(p75 * magnitude))
        statistics.append(entry)
    dimensions = len(dataset.features)
    correlation = [[None for _ in range(dimensions)] for _ in range(dimensions)]
    for a in range(dimensions):
        for b in range(a, dimensions):
            paired = valid[:, a] & valid[:, b]
            if int(paired.sum()) < 2:
                continue
            x, y = values[paired, a], values[paired, b]
            x_scale, y_scale = np.max(np.abs(x)), np.max(np.abs(y))
            if x_scale == 0 or y_scale == 0:
                continue
            x, y = x / x_scale, y / y_scale
            x, y = x - x.mean(), y - y.mean()
            denominator = np.linalg.norm(x) * np.linalg.norm(y)
            if denominator > 0:
                coefficient = float(np.clip(np.dot(x, y) / denominator, -1., 1.))
                correlation[a][b] = correlation[b][a] = coefficient
    sampling = sampling_profile(source["timestamps"])
    warnings = list(dataset.quality.warnings)
    if sampling["irregular_intervals"]:
        warnings.append(f"存在 {sampling['irregular_intervals']} 个不规则正采样间隔（相对中位间隔偏差超过 1%）；时序窗口仍按样本行计数。")
    return {"dataset_id": dataset.id, "rows": dataset.rows, "features": statistics,
            "correlation": {"names": dataset.features, "values": correlation},
            "sampling": sampling, "warnings": warnings}


def scored_mask(run: Run, output: dict) -> np.ndarray:
    if "scored" in output:
        return np.asarray(output["scored"], dtype=bool)
    mask = np.ones(len(output["flags"]), dtype=bool)
    mask[:run.summary.train_end or 0] = False
    return mask


def run_insights(run: Run, events: list[dict], source: dict, output: dict) -> dict:
    review = Counter(event["status"] for event in events)
    severity = Counter(event["severity"] for event in events)
    features = Counter(event["top_feature"] for event in events)
    mask = scored_mask(run, output)
    indices = np.flatnonzero(mask)
    start, end = (int(indices[0]), len(mask)) if indices.size else (0, 0)
    edges = np.linspace(start, end, min(24, end - start) + 1, dtype=int)
    timeline = [{"start": int(a), "end": int(b), "label": str(source["timestamps"][a]),
                 "anomaly_points": int(np.count_nonzero(output["flags"][a:b][mask[a:b]])),
                 "total_points": int(mask[a:b].sum())}
                for a, b in zip(edges[:-1], edges[1:]) if mask[a:b].any()]
    return {"run_id": run.id,
            "review": {key: review[key] for key in ("unreviewed", "confirmed", "false_positive")},
            "severity": {key: severity[key] for key in ("high", "medium", "low")},
            "top_features": [{"name": name, "event_count": count}
                             for name, count in sorted(features.items(), key=lambda item: (-item[1], item[0]))],
            "timeline": timeline}
