"""Download one official SMD machine subset and add ChronoLens-compatible CSV headers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://api.github.com/repos/NetManAIOps/OmniAnomaly/contents/ServerMachineDataset"


def download(url: str, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = Request(url, headers={"User-Agent": "ChronoLens-SMD-Preparation", "Accept": "application/vnd.github.raw+json"})
        with urlopen(request, timeout=60) as response, tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as file:
            temporary = Path(file.name)
            shutil.copyfileobj(response, file)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if not 1 <= len(values) <= 100000 or not 2 <= values.shape[1] <= 64:
        raise ValueError(f"{path.name}: platform requires 1-100000 rows and 2-64 features")
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}: dataset contains non-finite numbers")
    return values


def convert(source: Path, target: Path, features: list[str], labels: np.ndarray | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            writer = csv.writer(output)
            writer.writerow(features + (["is_anomaly"] if labels is not None else []))
            with source.open(encoding="utf-8", newline="") as raw:
                for index, row in enumerate(csv.reader(raw)):
                    if labels is not None:
                        row.append(str(int(labels[index])))
                    writer.writerow(row)
        if temporary.stat().st_size > 25 * 1024 * 1024:
            raise ValueError("Converted CSV exceeds the platform 25 MiB file limit")
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare one official Server Machine Dataset subset for ChronoLens")
    parser.add_argument("--machine", default="machine-1-1", help="One machine only; do not concatenate independent machines")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "datasets" / "SMD" / "raw", help="Folder containing train/, test/, test_label/")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets" / "SMD" / "prepared")
    parser.add_argument("--download", action="store_true", help="Download missing official files; existing raw files are left untouched")
    args = parser.parse_args()
    if not re.fullmatch(r"machine-[1-3]-[1-9][0-9]*", args.machine):
        parser.error("--machine must have the form machine-1-1, machine-2-3, etc.")
    try:
        filename = args.machine + ".txt"
        paths = {kind: args.source_dir / kind / filename for kind in ("train", "test", "test_label")}
        if args.download:
            for kind, target in paths.items():
                print(f"Preparing official {kind}: {target}", file=sys.stderr, flush=True)
                download(f"{SOURCE}/{kind}/{filename}?ref=master", target)
            download(f"{SOURCE}/LICENSE?ref=master", args.source_dir / "LICENSE")
        absent = [str(path) for path in paths.values() if not path.is_file()]
        if absent:
            raise ValueError("Missing raw files: " + ", ".join(absent) + "; download them manually or use --download")
        training = matrix(paths["train"])
        testing = matrix(paths["test"])
        if training.shape[1] != testing.shape[1]:
            raise ValueError("Training/test feature counts differ")
        labels = np.loadtxt(paths["test_label"], delimiter=",", dtype=np.float64, ndmin=1)
        if labels.ndim != 1 or len(labels) != len(testing) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Test labels must be one 0/1 value per test row")
        features = [f"metric_{index + 1:02d}" for index in range(training.shape[1])]
        train_csv = args.output_dir / f"{args.machine}_train.csv"
        test_csv = args.output_dir / f"{args.machine}_test.csv"
        convert(paths["train"], train_csv, features)
        convert(paths["test"], test_csv, features, labels)
        if (args.source_dir / "LICENSE").exists():
            args.output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.source_dir / "LICENSE", args.output_dir / "LICENSE")
        manifest = {
            "dataset": "SMD / Server Machine Dataset", "machine": args.machine,
            "source": "https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset",
            "license": "https://github.com/NetManAIOps/OmniAnomaly/blob/master/ServerMachineDataset/LICENSE",
            "train_rows": len(training), "test_rows": len(testing), "features": features,
            "test_anomaly_points": int(labels.sum()), "train_csv": str(train_csv.resolve()),
            "test_csv": str(test_csv.resolve()), "fit_start": 0,
            "suggested_fit_end": int(len(training) * .8), "suggested_calibration_end": len(training),
            "timestamp_mode": "sample_index; original SMD files do not provide timestamps",
            "raw_sha256": {kind: checksum(path) for kind, path in paths.items()},
            "notes": ["Train/test belong to the same machine; train other machines separately.",
                      "Metric names are anonymous stable column identifiers, not inferred physical quantities.",
                      "Test labels are for evaluation only, never threshold tuning.",
                      "Original numeric tokens are retained; no additional normalization or artificial timestamps."]
        }
        (args.output_dir / f"{args.machine}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError) as error:
        print(f"SMD preparation failed: {error}\nOfficial source: {SOURCE}\nIf direct download is unavailable, use downloaded train/test/test_label folders with --source-dir.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
