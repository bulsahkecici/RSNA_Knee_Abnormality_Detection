"""Generate a private Kaggle script kernel from a template plus one pinned source archive.

Replaces the earlier pattern of string-patching previous entry.py payloads.
The archive is a deterministic tar.gz of ``src/rsna_knee`` read from a git ref
(default HEAD; sorted, mtime 0), so the same commit always yields the same
SOURCE_SHA256 and uncommitted edits by concurrent work are never packaged. Templates live
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
import re
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = "bulsahkecici"
MARKERS = ('SOURCE_B64 = ""', 'SOURCE_SHA256 = ""', "BUILD_INFO = {}")


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def source_archive(ref: str, extra: tuple[str, ...] = ()) -> tuple[bytes, list[str]]:
    """Archive ``src/rsna_knee`` (plus ``extra`` committed paths) as of ``ref``; uncommitted edits never leak in.

    Code that locates the repository root from the working directory (configs, weight
    allowlist) needs ``extra=("configs", "pyproject.toml")`` and a chdir into the archive.
    """
    names = sorted(n for n in _git("ls-tree", "-r", "--name-only", ref, "--", "src/rsna_knee").decode().splitlines()
                   if n.endswith(".py"))
    if extra:
        names += sorted(_git("ls-tree", "-r", "--name-only", ref, "--", *extra).decode().splitlines())
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name in names:
            data = _git("show", f"{ref}:{name}")
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            archive.addfile(info, io.BytesIO(data))
    # gzip header carries a timestamp; zero it for byte-identical rebuilds.
    raw = bytearray(buffer.getvalue())
    raw[4:8] = b"\0\0\0\0"
    return bytes(raw), names


def git_identity(ref: str) -> dict:
    dirty = _git("status", "--porcelain", "--", "src/rsna_knee", "scripts/kaggle_entries").decode().strip()
    return {"git_ref": ref, "git_commit": _git("rev-parse", ref).decode().strip(),
            "worktree_dirty_ignored": bool(dirty)}


def apply_defines(template: str, defines: dict[str, object]) -> str:
    """Replace template lines ``NAME = ...  # define`` with generator-supplied Python literals."""
    for name, value in defines.items():
        pattern = re.compile(rf"^{re.escape(name)} = .*  # define$", re.M)
        if len(pattern.findall(template)) != 1:
            raise SystemExit(f"template has no unique definable constant {name!r}")
        line = f"{name} = {value!r}  # define"
        template = pattern.sub(lambda _, line=line: line, template)
    return template


def generate(entry: str, slug: str, *, internet: bool, gpu: bool, ref: str = "HEAD",
             defines: dict[str, object] | None = None) -> Path:
    template = _git("show", f"{ref}:scripts/kaggle_entries/{entry}.py").decode()
    for marker in MARKERS:
        if template.count(marker) != 1:
            raise SystemExit(f"template must contain {marker!r} exactly once")
    template = apply_defines(template, defines or {})
    archive, names = source_archive(ref)
    sha = hashlib.sha256(archive).hexdigest()
    info = {"entry": entry, "slug": slug, "source_sha256": sha, "n_source_files": len(names),
            "template_sha256": hashlib.sha256(template.encode()).hexdigest(), "defines": defines or {},
            **git_identity(ref)}
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
    parser.add_argument("--ref", default="HEAD", help="git ref whose committed src and template are packaged")
    parser.add_argument("--define", action="append", default=[], metavar="NAME=LITERAL",
                        help="override a template constant marked '# define' with a Python literal")
    args = parser.parse_args()
    defines = {}
    for item in args.define:
        name, _, literal = item.partition("=")
        defines[name] = ast.literal_eval(literal)
    folder = generate(args.entry, args.slug, internet=not args.no_internet, gpu=args.gpu, ref=args.ref, defines=defines)
    print(json.dumps(json.loads((folder / "build.json").read_text()) | {"files": "...", "folder": str(folder)}))


if __name__ == "__main__":
    main()
