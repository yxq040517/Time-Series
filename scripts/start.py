"""Windows-friendly launcher: isolated dependencies, prebuilt UI, local server."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen
import venv
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".venv"
ENV_PYTHON = ENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
DEPENDENCY_MARKER = ENV / ".chronolens-dependencies"


def command(args: list[str], cwd: Path = ROOT) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def prepare() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required.")
    if not ENV_PYTHON.exists():
        if sys.version_info >= (3, 13):
            raise RuntimeError("Pinned scientific packages require Python 3.10, 3.11 or 3.12. Create the environment with one of these versions.")
        print("Creating isolated Python environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(ENV)
    requirements = ROOT / "requirements.txt"
    signature = hashlib.sha256(requirements.read_bytes()).hexdigest()
    installed = DEPENDENCY_MARKER.read_text().strip() if DEPENDENCY_MARKER.exists() else ""
    probe = subprocess.run([str(ENV_PYTHON), "-c", "import fastapi, uvicorn, pandas, sklearn, multipart"], capture_output=True)
    if installed != signature or probe.returncode:
        print("Installing application dependencies (first run needs internet)...", flush=True)
        command([str(ENV_PYTHON), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)])
        DEPENDENCY_MARKER.write_text(signature, encoding="ascii")
    if not (ROOT / "frontend/dist/index.html").exists():
        npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
        if not npm:
            raise RuntimeError("Prebuilt UI is missing. Install Node.js 24 and run npm ci / npm run build in frontend.")
        print("Building frontend...", flush=True)
        command([npm, "ci", "--no-audit", "--no-fund"], ROOT / "frontend")
        command([npm, "run", "build"], ROOT / "frontend")


def is_our_server(url: str) -> bool:
    try:
        with urlopen(url + "/api/health", timeout=1) as response:
            body = json.load(response)
        return body.get("status") == "ok" and body.get("version") == "2.0.0"
    except (URLError, TimeoutError, OSError, ValueError):
        return False


def open_when_ready(url: str, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 30
    while process.poll() is None and time.monotonic() < deadline:
        if is_our_server(url):
            webbrowser.open(url)
            return
        time.sleep(0.3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Start ChronoLens locally")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    url = f"http://127.0.0.1:{args.port}"
    try:
        prepare()
        with socket.socket() as sock:
            occupied = sock.connect_ex(("127.0.0.1", args.port)) == 0
        if occupied:
            if is_our_server(url):
                print(f"ChronoLens is already running: {url}")
                if not args.no_browser:
                    webbrowser.open(url)
                return 0
            raise RuntimeError(f"Port {args.port} is occupied by an old ChronoLens version or another program. Stop the old service window or use start.bat --port 8768.")
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "2")
        env.setdefault("OPENBLAS_NUM_THREADS", "2")
        env.setdefault("MKL_NUM_THREADS", "2")
        process = subprocess.Popen([
            str(ENV_PYTHON), "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", str(args.port),
        ], cwd=ROOT, env=env)
        print(f"ChronoLens: {url}\nKeep this window open. Press Ctrl+C to stop.", flush=True)
        if not args.no_browser:
            threading.Thread(target=open_when_ready, args=(url, process), daemon=True).start()
        try:
            return process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            return 0
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
