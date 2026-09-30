"""Deterministic illustrative data, deliberately not a real benchmark dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd


FEATURES = ["cpu_usage", "memory_usage", "disk_io", "network_rx", "latency_ms", "error_rate"]


def make_demo(seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Create 2,400 minute samples with a clean initial segment and five events.

    Labels describe injected intervals, not a claim about real operational faults.
    The default 35% training prefix contains no injected anomalies.
    """
    rng = np.random.default_rng(seed)
    n = 2400
    t = np.arange(n, dtype=float)
    load = np.sin(t * 2 * np.pi / 180) + 0.4 * np.sin(t * 2 * np.pi / 53)
    noise = rng.normal(size=(n, len(FEATURES)))
    values = np.column_stack((
        43 + 7 * load + 1.8 * noise[:, 0],
        57 + 3.0 * load + 1.3 * noise[:, 1],
        112 + 16 * load + 5.0 * noise[:, 2],
        75 + 10 * load + 3.0 * noise[:, 3],
        24 + 2.8 * load + 0.9 * noise[:, 4],
        0.18 + 0.025 * load + 0.013 * noise[:, 5],
    ))
    labels = np.zeros(n, dtype=np.int8)
    injections = [
        (1010, 1048, [0, 4], [24.0, 15.0]),
        (1280, 1328, [2, 3], [-48.0, -32.0]),
        (1550, 1610, [1, 4], [18.0, 11.0]),
        (1840, 1872, [4, 5], [20.0, 0.28]),
        (2150, 2205, [0, 2, 3], [22.0, 65.0, 38.0]),
    ]
    for start, end, dims, offsets in injections:
        values[start:end, dims] += np.asarray(offsets)
        labels[start:end] = 1
    values = np.maximum(values, 0)
    values[:, :2] = np.clip(values[:, :2], 0, 100)
    frame = pd.DataFrame(values.round(5), columns=FEATURES)
    frame.insert(0, "timestamp", pd.date_range("2026-01-12 08:00:00", periods=n, freq="min").strftime("%Y-%m-%dT%H:%M:%S+08:00"))
    frame["is_anomaly"] = labels
    metadata = {
        "name": "合成服务器监控 · 2400 点",
        "description": "可复现的合成教学数据，包含6个相关指标与5段人工注入异常；不是生产数据或真实基准。默认前35%训练段不含注入异常。",
        "truth_events": [
            {"start": start, "end": end, "features": [FEATURES[i] for i in dims]}
            for start, end, dims, _ in injections
        ],
        "seed": seed,
    }
    return frame, metadata
