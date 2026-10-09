import time
from collections.abc import Callable

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from .schemas import DetectionConfig, TrainingConfig

METHODS = {
    "pca": "PCA 标准化重构残差平方（非因果解释）",
    "temporal": "逐变量因果滞后 Ridge 预测残差平方（不是因果归因）",
    "isolation": "历史稳健中心的绝对标准化偏差代理（不是 IsolationForest 内部归因）",
}


def intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _nonnegative_mean(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Average scores/deviations without overflowing their intermediate sum."""
    magnitude = values.max(axis=axis, keepdims=True)
    count = values.size if axis is None else values.shape[axis]
    if np.all(magnitude <= np.finfo(float).max / count):
        return values.mean(axis=axis)
    denominator = np.where(magnitude > 0, magnitude, 1.0)
    mean = (values / denominator).mean(axis=axis, keepdims=True) * magnitude
    return mean.squeeze(axis=axis)


def segment_events(flags: np.ndarray, scores: np.ndarray, contributions: np.ndarray,
                   timestamps: np.ndarray, features: list[str], threshold: float,
                   minimum: int, gap: int, scored: np.ndarray | None = None) -> list[dict]:
    merged = []
    for start, end in intervals(flags):
        if (merged and start - merged[-1][1] <= gap
                and (scored is None or scored[merged[-1][1]:start].all())):
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
                 "peak_score": peak, "mean_score": float(_nonnegative_mean(scores[start:end])),
                 "severity": "high" if ratio >= 2 else "medium" if ratio >= 1.35 else "low",
                 "top_feature": features[int(_nonnegative_mean(contributions[start:end], axis=0).argmax())],
                 "status": "unreviewed", "note": ""}
        events.append(event)
    return events


def _validate_data(data: dict, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 2 <= len(features) <= 64 or any(not isinstance(name, str) or not name.strip() for name in features):
        raise ValueError("特征必须包含 2 到 64 个非空名称。")
    if len(set(features)) != len(features):
        raise ValueError("特征名称不得重复。")
    try:
        values = np.asarray(data["values"], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("模型所需特征必须为数值列。") from error
    if values.ndim != 2 or values.shape[1] != len(features) or not 1 <= len(values) <= 100000:
        raise ValueError("数据形状与特征不一致，或样本数不在 1 到 100000 范围内。")
    if np.isinf(values).any():
        raise ValueError("输入包含无穷值；请检查数值单位。")
    timestamps = np.asarray(data["timestamps"])
    labels = np.asarray(data.get("labels", []))
    if timestamps.shape != (len(values),) or labels.ndim != 1 or labels.size not in (0, len(values)):
        raise ValueError("时间戳或标签长度与样本数不一致。")
    if labels.size and not np.isin(labels, [0, 1]).all():
        raise ValueError("标签只能是 0 或 1。")
    return values, timestamps, labels


def _fit_preprocessing(values: np.ndarray) -> dict:
    if not np.isfinite(values).any(axis=0).all():
        raise ValueError("历史拟合段中有特征完全缺失，无法安全学习填充值；请调整拟合区间或修复数据。")
    medians = np.nanmedian(values, axis=0)
    imputed = np.where(np.isnan(values), medians, values)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            center = imputed.mean(axis=0)
            scale = imputed.std(axis=0)
        except FloatingPointError as error:
            raise ValueError("数值动态范围过大，无法稳定拟合；请换用适当的数值单位。") from error
    scale = np.where(scale > 1e-12, scale, 1.0)
    if not all(np.isfinite(array).all() for array in (medians, center, scale)):
        raise ValueError("预处理参数出现非有限值；请检查数值量级。")
    return {"medians": medians, "center": center, "scale": scale}


def _standardize(values: np.ndarray, preprocessing: dict) -> np.ndarray:
    return (values - preprocessing["center"]) / preprocessing["scale"]


def _score(imputed: np.ndarray, asset: dict, history: np.ndarray | None = None
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preprocessing = asset["preprocessing"]
    n, dimensions = imputed.shape
    scored = np.ones(n, dtype=bool)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            if asset["algorithm"] == "temporal":
                window = asset["window"]
                prior = 0 if history is None else len(history)
                raw = imputed if not prior else np.concatenate((history, imputed), axis=0)
                z = _standardize(raw, preprocessing)
                baseline = np.zeros((n, dimensions), dtype=float)
                first = max(0, window - prior)
                scored[:first] = False
                if first < n:
                    for feature, predictor in enumerate(asset["detector"]):
                        lag = np.lib.stride_tricks.sliding_window_view(z[:, feature], window)[:-1]
                        lag = lag[prior + first - window:]
                        for start in range(0, len(lag), 4096):
                            stop = min(start + 4096, len(lag))
                            baseline[first + start:first + stop, feature] = predictor.predict(lag[start:stop])
                contributions = np.zeros_like(baseline)
                contributions[scored] = np.square(z[prior:][scored] - baseline[scored])
                scores = _nonnegative_mean(contributions, axis=1)
                reference = baseline * preprocessing["scale"] + preprocessing["center"]
                reference[~scored] = 0
            elif asset["algorithm"] == "pca":
                z = _standardize(imputed, preprocessing)
                components, mean = asset["pca_components"], asset["pca_mean"]
                baseline = ((z - mean) @ components.T) @ components + mean
                contributions = np.square(z - baseline)
                scores = _nonnegative_mean(contributions, axis=1)
                reference = baseline * preprocessing["scale"] + preprocessing["center"]
            elif asset["algorithm"] == "isolation":
                z = _standardize(imputed, preprocessing)
                scores = np.empty(n, dtype=float)
                for start in range(0, n, 4096):
                    scores[start:start + 4096] = -asset["detector"].score_samples(z[start:start + 4096])
                contributions = np.abs((imputed - asset["robust_center"]) / asset["robust_scale"])
                reference = np.broadcast_to(asset["robust_center"], imputed.shape).copy()
            else:
                raise ValueError("模型算法不受支持。")
        except FloatingPointError as error:
            raise ValueError("模型计算超出稳定数值范围；请检查数值单位。") from error
    if not all(np.isfinite(array).all() for array in (reference, contributions, scores)):
        raise ValueError("模型计算出现非有限结果；请检查数值量级。")
    return reference, contributions, scores, scored


def train_model(data: dict, features: list[str], config: TrainingConfig,
                progress: Callable[[int, str], None] = lambda value, message: None
                ) -> tuple[dict, dict, list[str]]:
    started = time.perf_counter()
    values, _, labels = _validate_data(data, features)
    n, dimensions = values.shape
    fit_start = config.fit_start
    fit_end = int(n * .8) if config.fit_end is None else config.fit_end
    calibration_end = n if config.calibration_end is None else config.calibration_end
    if not 0 <= fit_start < fit_end < calibration_end <= n:
        raise ValueError("区间必须满足 0 ≤ fit_start < fit_end < calibration_end ≤ 样本数。")
    fit_points = fit_end - fit_start
    minimum_fit = config.window + 5 if config.algorithm == "temporal" else 5
    if fit_points < minimum_fit or calibration_end - fit_end < 5:
        raise ValueError(f"拟合段至少需要 {minimum_fit} 个样本，校准段至少需要 5 个样本。")
    progress(10, "验证独立拟合与校准区间，不使用区间外数据或标签调参")
    preprocessing = _fit_preprocessing(values[fit_start:fit_end])
    imputed = np.where(np.isnan(values[fit_start:calibration_end]), preprocessing["medians"],
                       values[fit_start:calibration_end])
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            z_fit = _standardize(imputed[:fit_points], preprocessing)
    except FloatingPointError as error:
        raise ValueError("拟合数据超出稳定数值范围；请检查数值单位。") from error
    if not np.isfinite(z_fit).all():
        raise ValueError("拟合预处理产生非有限值；请检查数值量级。")
    notes = ["假设拟合与校准数据以正常行为为主；标签仅用于污染提示，不用于筛选训练或调参。",
             f"拟合区间 [{fit_start}, {fit_end})，校准区间 [{fit_end}, {calibration_end})；所有索引左闭右开。",
             "特征解释描述模型偏差，不证明根因或变量间的因果关系。"]
    if labels.size and labels[fit_start:calibration_end].any():
        notes.append(f"训练污染警告：拟合与校准段含 {int(labels[fit_start:calibration_end].sum())} 个标记异常；模型可能受污染影响。")
    if np.any(np.std(imputed[:fit_points], axis=0) <= 1e-12):
        notes.append("历史拟合段包含常量特征，缩放分母置为 1；常量变量不能提供正常波动信息。")
    asset = {"format_version": 1, "algorithm": config.algorithm, "features": list(features),
             "preprocessing": preprocessing, "window": config.window,
             "training_config": config.model_dump(), "fit_start": fit_start, "fit_end": fit_end,
             "calibration_end": calibration_end, "sklearn_version": sklearn.__version__}
    progress(25, "已固定拟合段学习的中位数与标准化参数")
    algorithm_summary = {}
    if config.algorithm == "pca":
        if not np.any(np.var(z_fit, axis=0) > 1e-15):
            raise ValueError("历史拟合段的所有特征均为常量，无法学习 PCA 变化子空间。")
        model = PCA(n_components=config.pca_variance, svd_solver="full").fit(z_fit)
        retained = min(int(model.n_components_), dimensions - 1)
        asset.update(detector=model, pca_components=model.components_[:retained].copy(),
                     pca_mean=model.mean_.copy())
        retained_variance = float(model.explained_variance_ratio_[:retained].sum())
        algorithm_summary = {"retained_components": retained, "retained_variance": retained_variance}
        notes.append(f"PCA 目标解释方差 {config.pca_variance:.1%}；实际保留 {retained}/{dimensions} 个主成分，解释方差 {retained_variance:.1%}。最多保留变量数减一的主成分，确保非空残差子空间。")
        notes.append("PCA 使用当前样本横截面变量重构，不是未来预测。")
    elif config.algorithm == "temporal":
        predictors = []
        for feature in range(dimensions):
            lag = np.lib.stride_tricks.sliding_window_view(z_fit[:, feature], config.window)[:-1]
            predictors.append(Ridge(alpha=1.0).fit(lag, z_fit[config.window:, feature]))
            progress(30 + int(25 * (feature + 1) / dimensions), f"已拟合滞后预测器 {feature + 1}/{dimensions}")
        asset["detector"] = predictors
        notes.append(f"每个变量独立使用前 {config.window} 个实际观测预测当前值，不使用当前或未来值作为输入。")
    elif config.algorithm == "isolation":
        asset["detector"] = IsolationForest(n_estimators=150, max_samples=min(256, fit_points),
                                            contamination="auto", random_state=42, n_jobs=1).fit(z_fit)
        robust_center = np.median(imputed[:fit_points], axis=0)
        with np.errstate(over="raise", invalid="raise"):
            try:
                mad = np.median(np.abs(imputed[:fit_points] - robust_center), axis=0) * 1.4826
            except FloatingPointError as error:
                raise ValueError("稳健偏差尺度超出数值范围；请检查数值单位。") from error
        asset.update(robust_center=robust_center,
                     robust_scale=np.where(mad > 1e-12, mad, preprocessing["scale"]))
        notes.append("孤立森林贡献仅为历史稳健偏差代理，不分解森林分数。")
    else:
        raise ValueError("模型算法不受支持。")
    progress(65, "仅使用独立校准段的固定模型分数确定阈值")
    _, _, scores, scored = _score(imputed, asset)
    calibration_scores = scores[fit_points:][scored[fit_points:]]
    if len(calibration_scores) < 5:
        raise ValueError("校准段可评分样本不足 5 个。")
    threshold = float(np.quantile(calibration_scores, config.threshold_quantile))
    asset["threshold"] = threshold
    summary = {"points": n, "n_features": dimensions, "fit_start": fit_start, "fit_end": fit_end,
               "calibration_end": calibration_end, "fit_points": fit_points,
               "calibration_points": len(calibration_scores), "threshold": threshold,
               "threshold_quantile": config.threshold_quantile,
               "duration_ms": (time.perf_counter() - started) * 1000,
               "calibration_score_stats": _score_stats(calibration_scores), **algorithm_summary}
    progress(94, "拟合与校准完成，准备保存不可变模型资产")
    return asset, summary, notes


def _score_stats(scores: np.ndarray) -> dict | None:
    if not len(scores):
        return None
    return {"min": float(scores.min()), "max": float(scores.max()),
            "median": float(np.quantile(scores, .5)), "p95": float(np.quantile(scores, .95))}


def detect(data: dict, features: list[str], asset: dict, config: DetectionConfig,
           progress: Callable[[int, str], None] = lambda value, message: None,
           history: np.ndarray | None = None
           ) -> tuple[dict, list[dict], dict, list[str], np.ndarray | None]:
    started = time.perf_counter()
    values, timestamps, labels = _validate_data(data, features)
    training_features = asset["features"]
    missing = [feature for feature in training_features if feature not in features]
    unexpected = [feature for feature in features if feature not in training_features]
    if missing or unexpected:
        raise ValueError(f"特征与模型不兼容；缺失特征：{missing}；多余特征：{unexpected}。")
    if asset.get("format_version") != 1 or asset.get("sklearn_version") != sklearn.__version__:
        raise ValueError("模型格式或 scikit-learn 版本不兼容；请使用兼容环境加载模型。")
    if asset["algorithm"] != "temporal" and (history is not None or config.stream_id is not None):
        raise ValueError("只有时序预测模型支持连续流历史。")
    dimensions = len(features)
    if history is not None:
        history = np.asarray(history, dtype=float)
        if (history.ndim != 2 or history.shape[1] != dimensions
                or len(history) > asset["window"] or not np.isfinite(history).all()):
            raise ValueError("连续流历史必须是训练特征顺序的有限原始观测，且不超过窗口长度。")
    progress(15, "按特征名称对齐，使用已保存的固定预处理与模型参数")
    incoming_to_training = [features.index(feature) for feature in training_features]
    training_to_incoming = [training_features.index(feature) for feature in features]
    ordered = values[:, incoming_to_training]
    imputed = np.where(np.isnan(ordered), asset["preprocessing"]["medians"], ordered)
    reference, contributions, scores, scored = _score(imputed, asset, history)
    flags = ((scores > asset["threshold"]) & scored).astype(np.int8)
    new_history = None
    if asset["algorithm"] == "temporal":
        raw = imputed if history is None or not len(history) else np.concatenate((history, imputed), axis=0)
        new_history = raw[-asset["window"]:].copy()
    arrays = {"values": imputed[:, training_to_incoming],
              "reference": reference[:, training_to_incoming],
              "contributions": contributions[:, training_to_incoming],
              "scores": scores, "flags": flags, "scored": scored}
    progress(75, "合并可评分异常片段，按输入特征顺序生成解释")
    events, summary = summarize_detection(data, features, asset, config, arrays,
                                          (time.perf_counter() - started) * 1000)
    scored_points = int(scored.sum())
    notes = ["本次检测仅使用保存的模型、预处理和阈值；未拟合模型或读取标签调参。",
             "特征解释描述模型偏差，不证明根因或变量间的因果关系。"]
    if asset["algorithm"] == "temporal":
        notes.append(f"前 {len(values) - scored_points} 行缺少完整因果窗口，标记为不可评分/预热；不计入指标和异常比例。")
    elif asset["algorithm"] == "isolation":
        notes.append("特征贡献是固定历史稳健偏差代理，不是孤立森林内部归因。")
    if labels.size and scored_points and not labels[scored].any():
        notes.append("可评分样本标签无正例；AUPRC、召回率和事件召回率按 0 展示，不能判断正例检出能力。")
    progress(94, "固定参数检测完成，准备保存数组、事件和指标")
    return arrays, events, summary, notes, new_history


def summarize_detection(data: dict, features: list[str], asset: dict, config: DetectionConfig,
                        arrays: dict, duration_ms: float = 0.0) -> tuple[list[dict], dict]:
    """Summarize fixed-model results, including batches with internal history resets."""
    scores, scored, flags = arrays["scores"], arrays["scored"], arrays["flags"]
    events = segment_events(flags, scores, arrays["contributions"], data["timestamps"], features,
                            asset["threshold"], config.min_event_length, config.merge_gap, scored)
    scored_points = int(scored.sum())
    labels = np.asarray(data.get("labels", []))
    metrics = None
    if labels.size and scored_points:
        truth, predicted = labels[scored], flags[scored]
        precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="binary", zero_division=0)
        true_events = intervals((labels == 1) & scored)
        hit = sum(any(event["start"] < end and event["end"] > start for event in events)
                  for start, end in true_events)
        metrics = {"precision": float(precision), "recall": float(recall), "f1": float(f1),
                   "auprc": float(average_precision_score(truth, scores[scored])) if truth.any() else 0.0,
                   "event_recall": hit / len(true_events) if true_events else 0.0}
    anomaly_points = int(flags[scored].sum())
    summary = {"points": len(scores), "n_features": len(features), "threshold": float(asset["threshold"]),
               "scored_points": scored_points, "warmup_points": len(scores) - scored_points,
               "anomaly_points": anomaly_points,
               "anomaly_ratio": anomaly_points / scored_points if scored_points else 0.0,
               "event_count": len(events), "duration_ms": duration_ms,
               "explanation_method": METHODS[asset["algorithm"]], "metrics": metrics,
               "score_stats": _score_stats(scores[scored])}
    return events, summary


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
