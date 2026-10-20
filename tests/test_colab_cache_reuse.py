"""Existing MRI archives remain intact; only exactly pinned complete caches are reused."""
import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('cache_reuse',Path(__file__).resolve().parents[1]/'scripts/colab_cache_reuse.py')
reuse=importlib.util.module_from_spec(spec);spec.loader.exec_module(reuse)


def make_archive(tmp_path, *, unsafe=False):
    slug='canonical-shard';drive=tmp_path/'drive';drive.mkdir()
    data=b'fixture-pixels';m={'studies':{'study':{'file':'study.npz','bytes':len(data)}}}
    raw=json.dumps(m).encode();digest=hashlib.sha256(raw).hexdigest()
    archive=drive/f'{slug}-{digest[:12]}.tar'
    with tarfile.open(archive,'w') as tar:
        for name,blob in [(f'{slug}/cache/manifest.json',raw),(f'{slug}/cache/study.npz',data)]:
            info=tarfile.TarInfo(name);info.size=len(blob);tar.addfile(info,io.BytesIO(blob))
        if unsafe:
            info=tarfile.TarInfo('../outside');info.size=1;tar.addfile(info,io.BytesIO(b'x'))
    return drive,archive,{'kernel':'owner/'+slug,'manifest_sha256':digest},data


def test_verified_drive_restore_skips_network_and_preserves_archive(tmp_path):
    drive,archive,part,data=make_archive(tmp_path);before=archive.read_bytes();dest=tmp_path/'runtime/canonical-shard'
    proof=reuse.restore_pinned_drive_cache(drive,part,dest)
    assert proof['kind']=='drive_archive' and (dest/'cache/study.npz').read_bytes()==data
    assert archive.read_bytes()==before
    assert reuse.restore_pinned_drive_cache(drive,part,dest)['kind']=='existing_runtime_cache'
    (dest/'cache/study.npz').unlink()
    assert not reuse.pinned_cache_ready(dest,part['manifest_sha256'])
    assert reuse.restore_pinned_drive_cache(drive,part,dest)['kind']=='drive_archive'


def test_incompatible_archive_is_preserved_for_its_original_experiment(tmp_path):
    drive,archive,part,_=make_archive(tmp_path);before=archive.read_bytes()
    part['manifest_sha256']='0'*64
    assert reuse.restore_pinned_drive_cache(drive,part,tmp_path/'runtime/canonical-shard') is None
    assert archive.read_bytes()==before


def test_unsafe_archive_cannot_escape_runtime_or_delete_drive_data(tmp_path):
    drive,archive,part,_=make_archive(tmp_path,unsafe=True);before=archive.read_bytes()
    assert reuse.restore_pinned_drive_cache(drive,part,tmp_path/'runtime/canonical-shard') is None
    assert not (tmp_path/'outside').exists() and archive.read_bytes()==before
