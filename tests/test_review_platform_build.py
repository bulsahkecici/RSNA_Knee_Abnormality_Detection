"""Selecting a platform build never waives any result identity checks."""
import importlib
import json
from pathlib import Path
import pytest


def test_explicit_colab_build_rejects_kaggle_platform_identity(tmp_path, monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    r=importlib.import_module('review_experiment')
    monkeypatch.setattr(r,'ROOT',tmp_path)
    kaggle=tmp_path/'notebooks/generated';colab=kaggle/'colab';colab.mkdir(parents=True)
    common={k:'fixed-'+k for k in r.PINNED}
    a={**common,'platform_recipe_sha256':'kaggle-recipe'}
    b={**common,'platform_recipe_sha256':'colab-recipe'}
    (kaggle/'CASE.build.json').write_text(json.dumps(a));(colab/'CASE.build.json').write_text(json.dumps(b))
    output=tmp_path/'output';output.mkdir();(output/'experiment.json').write_text(json.dumps(a))
    with pytest.raises(ValueError,match='platform_recipe_sha256'):
        r.verify('CASE',output,build_dir=colab)
    # The explicit directory does not waive data identity when platform identity matches.
    ran={**b,'train_uids_sha256':'wrong-cohort'};(output/'experiment.json').write_text(json.dumps(ran))
    with pytest.raises(ValueError,match='train_uids_sha256'):
        r.verify('CASE',output,build_dir=colab)
    (output/'experiment.json').write_text(json.dumps(b))
    with pytest.raises(ValueError,match='platform_recipe_sha256'):
        r.verify('CASE',output)  # default remains the original platform build
