#!/usr/bin/env python3
"""Build an offline bundle. Wheels must match the scoring OS/Python/CUDA — measure on the worker."""

from __future__ import annotations

import argparse
from rsna_knee.submission.package import package_run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    args = p.parse_args()
    print(package_run(args.run_id))


if __name__ == "__main__":
    main()
