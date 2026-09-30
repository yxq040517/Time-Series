"""Regenerate the bundled labelled example without launching the application."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.demo import make_demo


def main() -> None:
    frame, _ = make_demo()
    target = ROOT / "examples" / "server_metrics.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8-sig")
    print(f"Generated {len(frame)} labelled synthetic rows: {target}")


if __name__ == "__main__":
    main()
