"""Persistence for internally generated, trusted model artifacts only.

Joblib contains executable pickle data. A checksum proves integrity, not trust;
callers must obtain both the path and expected digest from internal storage,
never from an uploaded artifact or user-supplied digest.
"""
import hashlib
import hmac
import io
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted


def _validate_asset(asset: dict) -> None:
    if not isinstance(asset, dict) or asset.get("format_version") != 1:
        raise ValueError("不支持的模型资产格式版本。")
    if asset.get("sklearn_version") != sklearn.__version__:
        raise ValueError("模型的 scikit-learn 版本与当前环境不兼容；必须使用训练时的相同版本。")
    features = asset.get("features")
    if (not isinstance(features, list) or not 2 <= len(features) <= 64
            or any(not isinstance(name, str) or not name.strip() for name in features)
            or len(set(features)) != len(features)):
        raise ValueError("模型资产的特征契约无效。")
    dimensions = len(features)

    def vector(value: object, shape: tuple[int, ...], positive: bool = False) -> np.ndarray:
        if (not isinstance(value, np.ndarray) or value.shape != shape
                or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all()
                or (positive and not (value > 0).all())):
            raise ValueError("模型资产的参数形状或数值无效。")
        return value

    preprocessing = asset.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("模型缺少固定预处理参数。")
    for name in ("medians", "center", "scale"):
        vector(preprocessing.get(name), (dimensions,), positive=name == "scale")
    threshold = asset.get("threshold")
    if (not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
            or not np.isfinite(threshold) or threshold < 0):
        raise ValueError("模型阈值无效。")
    window = asset.get("window")
    if not isinstance(window, int) or isinstance(window, bool) or not 2 <= window <= 32:
        raise ValueError("模型窗口长度无效。")
    ranges = [asset.get(name) for name in ("fit_start", "fit_end", "calibration_end")]
    if (any(not isinstance(value, int) or isinstance(value, bool) for value in ranges)
            or not 0 <= ranges[0] < ranges[1] < ranges[2]):
        raise ValueError("模型训练区间无效。")
    if not isinstance(asset.get("training_config"), dict):
        raise ValueError("模型缺少训练配置。")
    algorithm, detector = asset.get("algorithm"), asset.get("detector")
    try:
        if algorithm == "pca":
            if not isinstance(detector, PCA):
                raise ValueError("PCA 模型资产缺少真实拟合估计器。")
            check_is_fitted(detector)
            components = asset.get("pca_components")
            if not isinstance(components, np.ndarray) or components.ndim != 2:
                raise ValueError("PCA 有效残差子空间无效。")
            retained = components.shape[0]
            if not 1 <= retained < dimensions:
                raise ValueError("PCA 必须保留非空残差子空间。")
            vector(components, (retained, dimensions))
            vector(asset.get("pca_mean"), (dimensions,))
            if (detector.n_features_in_ != dimensions
                    or retained != min(int(detector.n_components_), dimensions - 1)
                    or not np.array_equal(components, detector.components_[:retained])
                    or not np.array_equal(asset["pca_mean"], detector.mean_)):
                raise ValueError("PCA 有效子空间与拟合估计器不一致。")
        elif algorithm == "temporal":
            if not isinstance(detector, list) or len(detector) != dimensions:
                raise ValueError("时序模型必须保存每个特征的预测器。")
            for predictor in detector:
                if not isinstance(predictor, Ridge):
                    raise ValueError("时序模型包含无效预测器。")
                check_is_fitted(predictor)
                vector(predictor.coef_, (window,))
                if predictor.n_features_in_ != window or not np.isfinite(predictor.intercept_):
                    raise ValueError("时序预测器窗口或截距无效。")
        elif algorithm == "isolation":
            if not isinstance(detector, IsolationForest):
                raise ValueError("孤立森林资产缺少真实拟合估计器。")
            check_is_fitted(detector)
            if detector.n_features_in_ != dimensions or not detector.estimators_:
                raise ValueError("孤立森林拟合状态与特征契约不一致。")
            vector(asset.get("robust_center"), (dimensions,))
            vector(asset.get("robust_scale"), (dimensions,), positive=True)
        else:
            raise ValueError("模型算法不受支持。")
    except (AttributeError, TypeError) as error:
        raise ValueError("模型拟合状态无效。") from error


def save_model(path: Path, asset: dict) -> str:
    """Atomically save a trusted fitted asset and return its SHA-256 digest."""
    _validate_asset(asset)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            joblib.dump(asset, file, compress=3)
            file.flush()
            os.fsync(file.fileno())
            file.seek(0)
            hasher = hashlib.sha256()
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                hasher.update(chunk)
            digest = hasher.hexdigest()
        os.replace(temporary, path)
        temporary = None
        return digest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_model(path: Path, expected_sha256: str) -> dict:
    """Verify internally recorded integrity before unpickling a trusted artifact."""
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in expected_sha256)):
        raise ValueError("模型资产缺少有效的内部 SHA-256 校验值。")
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(digest, expected_sha256.lower()):
        raise ValueError("模型资产校验失败，文件可能损坏或被修改。")
    try:
        asset = joblib.load(io.BytesIO(payload))
    except Exception as error:
        raise ValueError("模型资产无法反序列化，且不会自动重新训练。") from error
    _validate_asset(asset)
    return asset
