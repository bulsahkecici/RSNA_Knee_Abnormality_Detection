"""Read existing Drive archives/directories without modifying them.

Manifest identity and payload sizes are checked here; training still rehashes
and decodes every payload before any optimization step.
"""
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def pinned_cache_ready(folder, expected):
    folder = Path(folder)
    path = folder / 'cache/manifest.json'
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        return False
    m = json.loads(path.read_text())
    for meta in m['studies'].values():
        relative = PurePosixPath(meta['file'])
        if relative.is_absolute() or '..' in relative.parts:
            return False
        payload = path.parent / meta['file']
        if not payload.is_file() or payload.stat().st_size != meta['bytes']:
            return False
    return True


def restore_pinned_drive_cache(drive_cache, part, part_dir):
    """Return source evidence or None; never delete or update any Drive files."""
    drive_cache, part_dir = Path(drive_cache), Path(part_dir)
    slug = part['kernel'].split('/')[-1];expected = part['manifest_sha256']
    if pinned_cache_ready(part_dir, expected):
        return {'kind': 'existing_runtime_cache', 'manifest_sha256': expected}
    part_dir.parent.mkdir(parents=True, exist_ok=True)
    candidates = [('drive_directory', drive_cache / slug)]
    exact = drive_cache / f'{slug}-{expected[:12]}.tar'
    archives = [exact] + sorted(p for p in drive_cache.glob(f'{slug}*.tar') if p != exact)
    candidates += [('drive_archive', p) for p in archives]
    for kind, source in candidates:
        if not source.exists():
            continue
        try:
            with tempfile.TemporaryDirectory(prefix='.drive-restore-', dir=part_dir.parent) as scratch:
                staging = Path(scratch);restored = staging / slug
                if kind == 'drive_directory':
                    if not pinned_cache_ready(source, expected):
                        continue
                    shutil.copytree(source / 'cache', restored / 'cache')
                else:
                    with tarfile.open(source) as archive:
                        pinned_manifest = archive.extractfile(slug + '/cache/manifest.json')
                        if pinned_manifest is None or hashlib.sha256(pinned_manifest.read()).hexdigest() != expected:
                            print('Drive archive is a different cache version:', source.name, flush=True)
                            continue
                        for member in archive.getmembers():
                            relative = PurePosixPath(member.name)
                            if (relative.is_absolute() or '..' in relative.parts or not relative.parts
                                    or relative.parts[0] != slug or not (member.isfile() or member.isdir())):
                                raise ValueError('unsafe_or_unexpected_cache_archive_member')
                        archive.extractall(staging, filter='data')
                if not pinned_cache_ready(restored, expected):
                    print('Drive cache differs from pinned K1 volumes or is incomplete:', source, flush=True)
                    continue
                # Only ephemeral Colab files can be replaced; Drive source stays untouched.
                if part_dir.exists():
                    shutil.rmtree(part_dir)
                shutil.move(str(restored), part_dir)
                return {'kind': kind, 'source': str(source), 'manifest_sha256': expected}
        except (OSError, tarfile.TarError, ValueError, KeyError) as error:
            print('Drive cache cannot be reused:', source.name, str(error)[:180], flush=True)
    return None
