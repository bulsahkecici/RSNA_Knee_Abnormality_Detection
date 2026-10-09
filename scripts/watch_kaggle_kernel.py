"""Poll one private Kaggle kernel, keep a heartbeat, download its output when it finishes.

Read-only toward the competition: never submits, never pushes. State lives in
state/campaigns/<campaign>/<slug>.json so a restarted watcher (or a new session)
sees the last observed remote status rather than assuming one. If this Mac sleeps
the watcher stops; the remote kernel keeps running and the next poll reconciles.

Usage: python scripts/watch_kaggle_kernel.py --campaign p0-20261009 --slug rsna-v4-dense-pilot-20261009
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = "bulsahkecici"


def kaggle(*args: str, timeout: int = 300) -> str:
    result = subprocess.run(["kaggle", *args], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"kaggle {args[0]} {args[1]} failed: {(result.stderr or result.stdout)[-300:]}")
    return result.stdout.strip()


def save(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def watch(campaign: str, slug: str, interval: int, max_hours: float, file_pattern: str | None = None) -> dict:
    folder = ROOT / "state/campaigns" / campaign
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{slug}.json"
    state = json.loads(path.read_text()) if path.exists() else {"kernel": f"{OWNER}/{slug}", "first_seen_ts": time.time()}
    started = time.time()
    while True:
        try:
            status = kaggle("kernels", "status", state["kernel"], timeout=120)
            state.update(observed=status, last_ok_ts=time.time(), poll_error=None)
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as error:
            state.update(poll_error=str(error)[:300])
            status = ""
        state["heartbeat_ts"] = time.time()
        if "COMPLETE" in status or "ERROR" in status or "CANCEL" in status:
            state["phase"] = "COMPLETE" if "COMPLETE" in status else "FAILED"
            out = ROOT / "artifacts/kaggle" / slug / "output"
            out.mkdir(parents=True, exist_ok=True)
            try:
                extra = ["--file-pattern", file_pattern] if file_pattern else []
                kaggle("kernels", "output", state["kernel"], "-p", str(out), "--page-size", "200", *extra, timeout=1800)
                state["output_dir"] = str(out.relative_to(ROOT))
                state["output_files"] = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())[:200]
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                state["output_error"] = str(error)[:300]
            save(path, state)
            return state
        state["phase"] = "RUNNING"
        save(path, state)
        if time.time() - started > max_hours * 3600:
            state["phase"] = "WATCH_TIMEOUT"
            save(path, state)
            return state
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--file-pattern", default=None,
                        help="regex of output files to download; use it for large caches that stay remote")
    args = parser.parse_args()
    state = watch(args.campaign, args.slug, args.interval, args.max_hours, args.file_pattern)
    print(json.dumps({k: state.get(k) for k in ("kernel", "phase", "observed", "output_dir", "output_error")}))


if __name__ == "__main__":
    main()
