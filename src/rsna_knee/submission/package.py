"""Package an offline inference bundle with hashes. No .env."""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

from rsna_knee.hashing import sha256_file
from rsna_knee.paths import ROOT, SUBMISSIONS_DIR, ensure_runtime_dirs

INCLUDE = [
    "src",
    "configs",
    "schemas",
    "prompts",
    "pyproject.toml",
    "notebooks/04_kaggle_inference.ipynb",
]

_BOOTSTRAP = """\
from pathlib import Path
import sys


def mount_asset(search=None):
    roots = []
    if search is not None:
        roots.append(Path(search))
    kaggle = Path("/kaggle/input")
    if kaggle.is_dir():
        roots.extend(sorted(path for path in kaggle.iterdir() if path.is_dir()))
    for root in roots:
        checkpoint = root / "assets" / "checkpoint.pt"
        source = root / "src"
        if checkpoint.is_file() and source.is_dir():
            if str(source) not in sys.path:
                sys.path.insert(0, str(source))
            return {"root": root, "checkpoint": checkpoint, "src": source}
    raise SystemExit("BLOCKED: versioned code/weights asset is not mounted")
"""


def _kaggle_username() -> str | None:
    import json as _json
    import os

    env = os.environ.get("KAGGLE_USERNAME")
    if env:
        return env
    path = Path.home() / ".kaggle" / "kaggle.json"
    if not path.is_file():
        return None
    try:
        return _json.loads(path.read_text(encoding="utf-8")).get("username")
    except (OSError, ValueError):
        return None


def _hash_tree(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json" and "__pycache__" not in path.parts:
            files[str(path.relative_to(root))] = sha256_file(path)
    return files


def offline_dependency_manifest() -> dict[str, Any]:
    import importlib.metadata as metadata

    packages = []
    for name in ("numpy", "torch", "pydicom", "pydantic"):
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = None
        packages.append({"name": name, "version": version})
    return {
        "packages": packages,
        "install_argv": [
            "python",
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            "asset/wheels",
            "-r",
            "asset/requirements.txt",
        ],
        "wheels_vendored": False,
        "note": "Pinned names are recorded. Missing or unhashed wheels stay blocked; this machine is not a clean offline venv proof.",
    }


def package_run(
    run_id: str,
    extra_files: list[Path] | None = None,
    *,
    dest_root: Path | None = None,
    synthetic: bool = False,
    checkpoint: Path | None = None,
    offline_wheels: Path | None = None,
    dataset_slug: str | None = None,
    kernel_slug: str | None = None,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    dest_root = dest_root or SUBMISSIONS_DIR
    dest_dir = dest_root / run_id
    frozen = dest_dir / "FROZEN"
    tar_path = dest_root / f"{run_id}.tar.gz"
    if frozen.is_file() and tar_path.is_file() and (dest_dir / "MANIFEST.json").is_file():
        manifest = json.loads((dest_dir / "MANIFEST.json").read_text(encoding="utf-8"))
        return {
            "dir": str(dest_dir),
            "tar": str(tar_path),
            "sha256": sha256_file(tar_path),
            "manifest": manifest,
            "identity_errors": verify_package(dest_dir),
            "frozen": True,
        }
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for rel in INCLUDE:
        src = ROOT / rel
        if not src.exists():
            continue
        target = dest_dir / rel
        if src.is_dir():
            shutil.copytree(src, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv"))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    copied_checkpoint = None
    asset_dir = dest_dir / "asset"
    asset_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint and checkpoint.is_file():
        copied_checkpoint = asset_dir / "assets" / "checkpoint.pt"
        copied_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, copied_checkpoint)
        src_tree = dest_dir / "src"
        if src_tree.is_dir():
            shutil.copytree(src_tree, asset_dir / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)
    user = _kaggle_username()
    dataset_slug = dataset_slug or (f"{user}/rsna-knee-runtime" if user else None)
    dataset_meta = {
        "title": dataset_slug.split("/")[-1] if dataset_slug else "rsna-knee-runtime",
        "id": dataset_slug,
        "licenses": [{"name": "CC-BY-NC-4.0"}],
    }
    (asset_dir / "dataset-metadata.json").write_text(json.dumps(dataset_meta, indent=2), encoding="utf-8")
    (asset_dir / "bootstrap.py").write_text(_BOOTSTRAP, encoding="utf-8")
    deps = offline_dependency_manifest()
    if offline_wheels is not None:
        import zipfile
        from email.parser import BytesParser

        wheel_files = sorted(offline_wheels.glob("*.whl"))
        if not wheel_files:
            raise ValueError("offline wheel directory is empty")
        wheel_dest = asset_dir / "wheels"
        wheel_dest.mkdir()
        packages = []
        for wheel in wheel_files:
            with zipfile.ZipFile(wheel) as archive:
                metadata_files = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
                if len(metadata_files) != 1:
                    raise ValueError(f"invalid wheel metadata: {wheel.name}")
                meta = BytesParser().parsebytes(archive.read(metadata_files[0]))
            packages.append({"name": meta["Name"], "version": meta["Version"]})
            shutil.copy2(wheel, wheel_dest / wheel.name)
        deps.update(
            packages=packages,
            wheels_vendored=True,
            wheel_hashes={wheel.name: sha256_file(wheel) for wheel in wheel_files},
            preinstalled_runtime=["numpy", "torch", "pillow", "pydantic", "pyyaml", "scikit-learn", "filelock"],
            note="Codec wheels are vendored. Preinstalled runtime must be verified by a real offline Kaggle rehearsal.",
        )
        deps["install_argv"].insert(5, "--no-deps")
    (asset_dir / "offline-deps.json").write_text(json.dumps(deps, indent=2), encoding="utf-8")
    (asset_dir / "requirements.txt").write_text(
        "\n".join(f"{item['name']}=={item['version']}" for item in deps["packages"] if item.get("version")) + "\n",
        encoding="utf-8",
    )
    (dest_dir / "kaggle-asset.json").write_text(
        json.dumps(
            {
                "dataset_dir": "asset",
                "dataset_sources": [dataset_slug] if dataset_slug else [],
                "create_argv": ["kaggle", "datasets", "create", "-p", str(asset_dir)],
                "executed": False,
                "note": "kernels push uploads the notebook only. Code and weights must be this dataset.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if extra_files:
        for path in extra_files:
            if path.exists() and path.is_file():
                shutil.copy2(path, dest_dir / path.name)
    user = _kaggle_username()
    kernel_meta = {
        "id": kernel_slug or (f"{user}/rsna-knee-infer" if user else None),
        "title": kernel_slug.split("/")[-1] if kernel_slug else "rsna-knee-infer",
        "code_file": "notebooks/04_kaggle_inference.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [dataset_slug] if dataset_slug else [],
        "kernel_sources": [],
        "model_sources": [],
        "push_ready": False,
        "push_note": "kernels push does not upload src or checkpoint. Create the dataset asset first; executed stays false here.",
    }
    (dest_dir / "kernel-metadata.json").write_text(json.dumps(kernel_meta, indent=2), encoding="utf-8")
    files = _hash_tree(dest_dir)
    manifest = {
        "run_id": run_id,
        "synthetic": synthetic,
        "encoder": "tiny_test_encoder" if synthetic else "dinov2_vits14",
        "checkpoint": None if copied_checkpoint is None else str(copied_checkpoint.relative_to(dest_dir)),
        "files": files,
        "internet_required": False,
        "lmstudio_required": False,
        "drive_required": False,
        "kernel_metadata": "kernel-metadata.json",
        "checkpoint_contract": "asset/assets/checkpoint.pt",
        "asset_dir": "asset",
        "note": "Scoring notebook must run offline. kernels push does not upload this folder.",
    }
    man_path = dest_dir / "MANIFEST.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tar_path = dest_root / f"{run_id}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(dest_dir, arcname=run_id)
    return {
        "dir": str(dest_dir),
        "tar": str(tar_path),
        "sha256": sha256_file(tar_path),
        "manifest": manifest,
        "identity_errors": verify_package(dest_dir),
    }


def verify_package(dest_dir: Path) -> list[str]:
    man_path = dest_dir / "MANIFEST.json"
    if not man_path.is_file():
        return ["manifest_missing"]
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("synthetic") is True:
        errors.append("synthetic_manifest")
    if not manifest.get("checkpoint"):
        errors.append("checkpoint_missing")
    else:
        ckpt = dest_dir / manifest["checkpoint"]
        if not ckpt.is_file():
            errors.append("checkpoint_file_missing")
        elif manifest.get("checkpoint") != "asset/assets/checkpoint.pt":
            errors.append("checkpoint_path_contract")
        else:
            errors.extend(_checkpoint_errors(ckpt))
    if not (dest_dir / "asset" / "bootstrap.py").is_file():
        errors.append("bootstrap_missing")
    if not (dest_dir / "asset" / "dataset-metadata.json").is_file():
        errors.append("dataset_metadata_missing")
    kernel = dest_dir / "kernel-metadata.json"
    if not kernel.is_file():
        errors.append("kernel_metadata_missing")
    else:
        meta = json.loads(kernel.read_text(encoding="utf-8"))
        if meta.get("enable_internet") is not False:
            errors.append("internet_not_disabled")
        if "rsna-knee-abnormality-detection" not in (meta.get("competition_sources") or []):
            errors.append("competition_source_missing")
        if manifest.get("synthetic") is not True and not meta.get("dataset_sources"):
            errors.append("dataset_source_missing")
        if meta.get("push_ready") is True:
            errors.append("push_marked_ready_without_execution")
    for rel, digest in (manifest.get("files") or {}).items():
        path = dest_dir / rel
        if not path.is_file():
            errors.append(f"missing_file:{rel}")
        elif sha256_file(path) != digest:
            errors.append(f"hash_mismatch:{rel}")
    if not (dest_dir / "asset" / "offline-deps.json").is_file():
        errors.append("offline_manifest_missing")
    else:
        deps = json.loads((dest_dir / "asset" / "offline-deps.json").read_text(encoding="utf-8"))
        if not deps.get("install_argv") or "--no-index" not in deps.get("install_argv", []):
            errors.append("offline_install_missing")
        if not (dest_dir / "asset" / "requirements.txt").is_file():
            errors.append("offline_requirements_missing")
    if manifest.get("synthetic") is not True and not (dest_dir / "asset" / "wheels").is_dir():
        errors.append("offline_wheels_not_vendored")
    return errors


def _checkpoint_errors(path: Path) -> list[str]:
    from rsna_knee.models.dinov2 import Dinov2ViTS14
    from rsna_knee.models.study import StudyModel
    from rsna_knee.training.checkpoint import torch_load

    try:
        blob = torch_load(path)
    except Exception:
        return ["checkpoint_unreadable"]
    errors: list[str] = []
    if blob.get("encoder_name") != "dinov2_vits14":
        errors.append("checkpoint_encoder")
    if int(blob.get("step") or 0) < 1:
        errors.append("checkpoint_no_optimizer_step")
    weights = blob.get("model")
    if not isinstance(weights, dict):
        errors.append("checkpoint_state_missing")
        return errors
    img = int(blob.get("img_size") or 0)
    if img <= 0:
        errors.append("checkpoint_img_size")
        return errors
    model = StudyModel(Dinov2ViTS14(img_size=img), freeze_encoder=True)
    try:
        model.load_state_dict(weights, strict=True)
    except Exception:
        errors.append("checkpoint_state_mismatch")
        return errors
    try:
        import torch

        images = torch.zeros(1, 1, 1, 3, img, img)
        with torch.no_grad():
            out = model(images, torch.ones(1, 1), torch.ones(1, 1, 1))
        if tuple(out.shape) != (1, 12):
            errors.append("checkpoint_forward")
    except Exception:
        errors.append("checkpoint_forward")
    return errors
