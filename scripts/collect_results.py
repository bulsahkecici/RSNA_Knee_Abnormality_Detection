#!/usr/bin/env python3
"""Collect remote result.json files into the local registry. Does not treat stale heartbeats as success."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsna_knee.runtime.transport import FileTransport
from rsna_knee.workflow.registry import Registry


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", required=True)
    p.add_argument("--attempt-id", required=True)
    p.add_argument("--inbox", type=Path, default=None)
    args = p.parse_args()
    tr = FileTransport(args.inbox) if args.inbox else FileTransport()
    if tr.heartbeat_stale(args.job_id, args.attempt_id):
        print(json.dumps({"status": "heartbeat_stale", "complete": False}))
        return
    result = tr.read_result(args.job_id, args.attempt_id)
    if not result:
        print(json.dumps({"status": "missing_result"}))
        return
    Registry().put_job(args.job_id, result.get("run_id", ""), args.attempt_id, result.get("fencing_token", ""), "done", result)
    print(json.dumps({"status": "ingested", "metrics_kind": (result.get("metrics") or {}).get("kind")}))


if __name__ == "__main__":
    main()
