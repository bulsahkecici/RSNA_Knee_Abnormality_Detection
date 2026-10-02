#!/usr/bin/env python3
"""Collect remote result.json files into the local registry. Does not treat stale heartbeats as success."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsna_knee.runtime.ingest import ingest_result
from rsna_knee.runtime.transport import FileTransport
from rsna_knee.workflow.registry import Registry


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", required=True)
    p.add_argument("--attempt-id", required=True)
    p.add_argument("--expected-token", required=True)
    p.add_argument("--inbox", type=Path, default=None)
    args = p.parse_args()
    tr = FileTransport(args.inbox) if args.inbox else FileTransport()
    stale = tr.heartbeat_stale(args.job_id, args.attempt_id)
    result = tr.read_result(args.job_id, args.attempt_id)
    decision = ingest_result(expected_token=args.expected_token, result=result, heartbeat_stale=stale)
    if not decision["accepted"]:
        print(json.dumps(decision))
        return
    Registry().put_job(
        args.job_id,
        (result or {}).get("run_id", ""),
        args.attempt_id,
        args.expected_token,
        "done",
        result or {},
    )
    print(json.dumps({"status": "ingested", "metrics_kind": (result or {}).get("metrics", {}).get("kind")}))


if __name__ == "__main__":
    main()
