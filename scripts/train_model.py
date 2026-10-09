"""Train a reusable ChronoLens model using the same core and store as the platform."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.engine import train_model
from backend.schemas import TrainingConfig
from backend.storage import Storage, now


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and save a reusable ChronoLens model")
    parser.add_argument("csv", type=Path, help="Training CSV, not a serialized model")
    parser.add_argument("--name", help="Model name; defaults to the CSV stem")
    parser.add_argument("--algorithm", choices=("pca", "temporal", "isolation"), default="pca")
    parser.add_argument("--fit-start", type=int, default=0)
    parser.add_argument("--fit-end", type=int, help="Exclusive fitting end; defaults to 80%% of rows")
    parser.add_argument("--calibration-end", type=int, help="Exclusive calibration end; defaults to all rows")
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--pca-variance", type=float, default=0.9)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--timestamp-column")
    parser.add_argument("--label-column")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "backend" / "data")
    parser.add_argument("--publish", action="store_true", help="Publish the saved model for platform detection")
    args = parser.parse_args()
    training = None
    store = None
    try:
        content = args.csv.read_bytes()
        store = Storage(args.data_dir, recover_interrupted=False)
        dataset = store.add_dataset(content, args.csv.stem, "upload", "通过独立训练脚本导入",
                                    timestamp_column=args.timestamp_column, label_column=args.label_column)
        config = TrainingConfig(dataset_id=dataset.id, name=args.name or args.csv.stem,
                                algorithm=args.algorithm, fit_start=args.fit_start, fit_end=args.fit_end,
                                calibration_end=args.calibration_end, threshold_quantile=args.threshold_quantile,
                                pca_variance=args.pca_variance, window=args.window)
        training = store.add_training(config, dataset)
        store.update_training(training.id, status="running", progress=1, message="读取训练数据")

        def progress(value: int, message: str) -> None:
            store.update_training(training.id, progress=value, message=message)
            print(f"[{value:3d}%] {message}", file=sys.stderr, flush=True)

        asset, summary, notes = train_model(store.load_arrays("dataset", dataset.id), dataset.features, config, progress)
        model = store.finish_training(training.id, asset, summary, notes)
        if args.publish:
            model = store.publish_model(model.id)
        print(json.dumps(model.model_dump(), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        if store is not None and training is not None:
            store.update_training(training.id, status="failed", error=str(exc), message="本地训练失败", completed_at=now())
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
