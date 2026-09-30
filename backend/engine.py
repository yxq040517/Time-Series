import time
from collections.abc import Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from .schemas import RunConfig

METHODS = {
    "pca": "PCA 标准化重构残差平方（非因果解释）",
    "temporal": "逐变量因果滞后 Ridge 预测残差平方（不是因果归因）",
    "isolation": "历史稳健中心的绝对标准化偏差代理（不是 IsolationForest 内部归因）",
}


def intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def segment_events(flags: np.ndarray, scores: np.ndarray, contributions: np.ndarray,
                   timestamps: np.ndarray, features: list[str], threshold: float,
                   minimum: int, gap: int) -> list[dict]:
    merged = []
    for start, end in intervals(flags):
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    events = []
    for start, end in merged:
        if end - start < minimum:
            continue
        peak = float(scores[start:end].max())
        ratio = peak / max(threshold, np.finfo(float).eps)
        event = {"id": f"e-{len(events) + 1:04d}", "start": start, "end": end,
                 "start_time": str(timestamps[start]), "end_time": str(timestamps[end - 1]),
                 "length": end - start, "anomaly_points": int(flags[start:end].sum()),
                 "peak_score": peak, "mean_score": float(scores[start:end].mean()),
                 "severity": "high" if ratio >= 2 else "medium" if ratio >= 1.35 else "low",
                 "top_feature": features[int(contributions[start:end].sum(axis=0).argmax())],
                 "status": "unreviewed", "note": ""}
        events.append(event)
    return events


def prepare(values: np.ndarray, fit_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    historical = values[:fit_end]
    available = np.isfinite(historical).any(axis=0)
    if not available.all():
        raise ValueError("历史拟合段中有特征完全缺失，无法安全学习填充值；请调整训练比例或修复数据。")
    medians = np.nanmedian(historical, axis=0)
    imputed = np.where(np.isnan(values), medians, values)
    center = imputed[:fit_end].mean(axis=0)
    scale = imputed[:fit_end].std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        standardized = (imputed - center) / scale
    if not np.isfinite(standardized).all():
        raise ValueError("数值动态范围过大，无法稳定计算；请换用适当的数值单位。")
    return imputed, standardized, center, scale


def analyze(data: dict, features: list[str], config: RunConfig,
            progress: Callable[[int, str], None] = lambda value, message: None) -> tuple[dict, list[dict], dict, list[str]]:
    started = time.perf_counter()
    values = data["values"]
    n, dimensions = values.shape
    train_end = int(n * config.train_ratio)
    fit_end = train_end - max(5, int(train_end * .2))
    progress(10, "按时间切分历史拟合段、历史校准段与后续评估段")
    imputed, z, center, scale = prepare(values, fit_end)
    notes = ["假设历史训练段以正常行为为主；拟合与阈值校准只使用训练段，后续标签只用于评估。",
             f"拟合区间 [0, {fit_end})，校准区间 [{fit_end}, {train_end})，评估区间 [{train_end}, {n})；事件索引左闭右开。",
             "特征解释描述模型偏差，不证明根因或变量间的因果关系。"]
    labels = data["labels"]
    if labels.size and labels[:train_end].any():
        notes.append(f"训练污染警告：历史训练段含 {int(labels[:train_end].sum())} 个标记异常；未用标签筛选训练或调参，结果可能受污染影响。")
    if np.any(np.std(imputed[:fit_end], axis=0) <= 1e-12):
        notes.append("历史拟合段包含常量特征，缩放分母置为 1；常量变量本身不能提供正常波动信息。")
    progress(25, "仅从历史拟合段学习缺失值中位数与标准化参数")
    if config.algorithm == "pca":
        if not np.any(np.var(z[:fit_end], axis=0) > 1e-15):
            raise ValueError("历史拟合段的所有特征均为常量，无法学习 PCA 变化子空间；可改用时序预测或孤立森林。")
        model = PCA(n_components=config.pca_variance, svd_solver="full")
        model.fit(z[:fit_end])
        retained = min(int(model.n_components_), dimensions - 1)
        components = model.components_[:retained]
        retained_variance = float(model.explained_variance_ratio_[:retained].sum())
        progress(50, f"历史 PCA 已拟合，保留 {retained} 个主成分和至少一个残差方向")
        baseline = ((z - model.mean_) @ components.T) @ components + model.mean_
        notes.append(f"PCA 目标解释方差 {config.pca_variance:.1%}；实际保留 {retained}/{dimensions} 个主成分，解释方差 {retained_variance:.1%}。为避免完整重构抹去异常，最多保留变量数减一的主成分。")
        contributions = np.square(z - baseline)
        scores = contributions.mean(axis=1)
        reference = baseline * scale + center
        notes.append("PCA 重构可以使用当前样本各变量；这是横截面重构检测，不是未来预测。")
    elif config.algorithm == "temporal":
        window = config.window
        if fit_end < window + 5:
            raise ValueError("历史拟合段过短，至少需要窗口长度加 5 个样本；请增大训练比例或减小窗口。")
        baseline = np.zeros_like(z)
        for feature in range(dimensions):
            # Every row t uses only observations [t-window,t), never row t or future rows.
            history = np.lib.stride_tricks.sliding_window_view(z[:, feature], window)[:-1]
            model = Ridge(alpha=1.0)
            model.fit(history[:fit_end - window], z[window:fit_end, feature])
            for start in range(0, len(history), 4096):
                baseline[start + window:min(start + 4096, len(history)) + window, feature] = model.predict(history[start:start + 4096])
            progress(30 + int(25 * (feature + 1) / dimensions), f"历史滞后预测器已拟合 {feature + 1}/{dimensions} 个变量")
        contributions = np.square(z - baseline)
        scores = contributions.mean(axis=1)
        # There is no causal lag window for the initial rows; they are not scored.
        contributions[:window] = 0
        scores[:window] = 0
        reference = baseline * scale + center
        notes.append(f"每个变量独立使用前 {window} 个实际观测预测当前值；不使用当前或未来值作为输入，不建模跨变量因果关系。最初 {window} 行无完整历史，分数置 0。")
    else:
        model = IsolationForest(n_estimators=150, max_samples=min(256, fit_end),
                                contamination="auto", random_state=42, n_jobs=1)
        model.fit(z[:fit_end])
        progress(50, "历史孤立森林已拟合；计算样本隔离分数")
        scores = np.empty(n, dtype=float)
        for start in range(0, n, 4096):
            scores[start:start + 4096] = -model.score_samples(z[start:start + 4096])
        robust_center = np.median(imputed[:fit_end], axis=0)
        mad = np.median(np.abs(imputed[:fit_end] - robust_center), axis=0) * 1.4826
        robust_scale = np.where(mad > 1e-12, mad, scale)
        contributions = np.abs((imputed - robust_center) / robust_scale)
        reference = np.broadcast_to(robust_center, imputed.shape).copy()
        notes.append("孤立森林分数来自真实模型的隔离路径；特征热力图和贡献份额仅为历史稳健偏差代理，不分解森林分数。")
    if not all(np.isfinite(array).all() for array in (scores, reference, contributions)):
        raise ValueError("模型计算出现非有限结果；请检查数值量级或改用其他算法。")
    progress(70, "使用历史校准段分位数确定阈值，不读取后续标签")
    threshold = float(np.quantile(scores[fit_end:train_end], config.threshold_quantile))
    flags = (scores > threshold).astype(np.int8)
    flags[:train_end] = 0
    progress(82, "合并异常片段并汇总非因果特征偏差")
    events = segment_events(flags, scores, contributions, data["timestamps"], features, threshold,
                            config.min_event_length, config.merge_gap)
    metrics = None
    if labels.size:
        truth, predicted = labels[train_end:], flags[train_end:]
        precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="binary", zero_division=0)
        true_events = intervals(truth)
        hit = sum(any(event["start"] < train_end + end and event["end"] > train_end + start for event in events)
                  for start, end in true_events)
        metrics = {"precision": float(precision), "recall": float(recall), "f1": float(f1),
                   "auprc": float(average_precision_score(truth, scores[train_end:])) if truth.any() else 0.0,
                   "event_recall": hit / len(true_events) if true_events else 0.0}
        if not truth.any():
            notes.append("评估段标签无正例；AUPRC、召回率和事件召回率按 0 展示，不能用于判断正例检出能力。")
    post = scores[train_end:]
    summary = {"points": n, "n_features": dimensions, "train_end": train_end, "fit_end": fit_end,
               "threshold": threshold, "anomaly_points": int(flags.sum()), "anomaly_ratio": float(flags.sum() / len(post)),
               "event_count": len(events), "duration_ms": (time.perf_counter() - started) * 1000,
               "explanation_method": METHODS[config.algorithm], "metrics": metrics,
               "score_stats": {"min": float(post.min()), "max": float(post.max()),
                               "median": float(np.median(post)), "p95": float(np.quantile(post, .95))}}
    progress(94, "保存分析数组、事件和评估结果")
    return {"values": imputed, "reference": reference, "contributions": contributions,
            "scores": scores, "flags": flags}, events, summary, notes


def peak_sample(scores: np.ndarray, start: int, end: int, maximum: int) -> np.ndarray:
    """Retain both endpoints and each bucket's actual peak, without synthetic averages."""
    if end - start <= maximum:
        return np.arange(start, end, dtype=int)
    edges = np.linspace(start, end, maximum - 1, dtype=int)
    selected = [start, end - 1]
    for left, right in zip(edges[:-1], edges[1:]):
        if right > left:
            selected.append(left + int(np.argmax(scores[left:right])))
    return np.unique(selected)
