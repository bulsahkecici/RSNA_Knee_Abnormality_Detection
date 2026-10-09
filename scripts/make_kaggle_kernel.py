"""Generate a private Kaggle script kernel from a template plus one pinned source archive.

Replaces the earlier pattern of string-patching previous entry.py payloads.
The archive is a deterministic tar.gz of ``src/rsna_knee`` (sorted, mtime 0, no
caches), so the same commit always yields the same SOURCE_SHA256. Templates live
in ``scripts/kaggle_entries/`` and must define ``SOURCE_B64 = ""``,
``SOURCE_SHA256 = ""`` and ``BUILD_INFO = {}`` exactly once.

Usage: python scripts/make_kaggle_kernel.py dense_cache_pilot --slug rsna-v4-dense-pilot-20261009
Writes artifacts/kaggle/<slug>/{entry.py,kernel-metadata.json,build.json}. Does not push.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = "bulsahkecici"
MARKERS = ('SOURCE_B64 = ""', 'SOURCE_SHA256 = ""', "BUILD_INFO = {}")


def source_archive() -> tuple[bytes, list[str]]:
    files = sorted(p for p in (ROOT / "src/rsna_knee").rglob("*.py") if "__pycache__" not in p.parts)
    buffer = io.BytesIO()
    names = []
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for path in files:
            name = str(path.relative_to(ROOT))
            data = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            archive.addfile(info, io.BytesIO(data))
            names.append(name)
    # gzip header carries a timestamp; zero it for byte-identical rebuilds.
    raw = bytearray(buffer.getvalue())
    raw[4:8] = b"\0\0\0\0"
    return bytes(raw), names


def git_identity() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = run("status", "--porcelain", "--", "src/rsna_knee", "scripts/kaggle_entries")
    return {"git_head": run("rev-parse", "HEAD"), "src_dirty": bool(dirty)}


def generate(entry: str, slug: str, *, internet: bool, gpu: bool) -> Path:
    template = (ROOT / "scripts/kaggle_entries" / f"{entry}.py").read_text(encoding="utf-8")
    for marker in MARKERS:
        if template.count(marker) != 1:
            raise SystemExit(f"template must contain {marker!r} exactly once")
    archive, names = source_archive()
    sha = hashlib.sha256(archive).hexdigest()
    info = {"entry": entry, "slug": slug, "source_sha256": sha, "n_source_files": len(names),
            "template_sha256": hashlib.sha256(template.encode()).hexdigest(), **git_identity()}
    text = (template.replace(MARKERS[0], f"SOURCE_B64 = {base64.b64encode(archive).decode()!r}")
            .replace(MARKERS[1], f"SOURCE_SHA256 = {sha!r}")
            .replace(MARKERS[2], f"BUILD_INFO = {info!r}"))
    ast.parse(text)
    folder = ROOT / "artifacts/kaggle" / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "entry.py").write_text(text, encoding="utf-8")
    metadata = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": "entry.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_internet": internet,
                "competition_sources": ["rsna-knee-abnormality-detection"], "dataset_sources": [], "kernel_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (folder / "build.json").write_text(json.dumps({**info, "files": names}, indent=2) + "\n", encoding="utf-8")
    return folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("entry")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--no-internet", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    folder = generate(args.entry, args.slug, internet=not args.no_internet, gpu=args.gpu)
    print(json.dumps(json.loads((folder / "build.json").read_text()) | {"files": "...", "folder": str(folder)}))


if __name__ == "__main__":
    main()
