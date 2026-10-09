"""Encoder registry: production guard, adaptation units, layer decay, weights sha256, cross-family refusal.

Tiny timm configs are random and local; nothing is downloaded and nothing here is a model result.
"""
import pytest
import torch

from rsna_knee.evaluation.heldout import load_trained_study
from rsna_knee.models import families
from rsna_knee.models.dinov2 import Dinov2ViTS14
from rsna_knee.models.encoder import TinyEncoder
from rsna_knee.models.study import StudyModel
from rsna_knee.training.loop import _optimizer, train_study_model

pytest.importorskip("timm")
TIMM = ["convnext_tiny", "coatnet_0_rw_224"]


def _tiny(name, size=64):
    torch.manual_seed(0)
    return families.build_encoder(name, size, test_config=True)


def _batches(size=64, n=2):
    out = []
    for i in range(n):
        y = torch.zeros(1, 12)
        y[0, 0] = float(i % 2)
        out.append({"images": torch.rand(1, 3, 2, 3, size, size, generator=torch.Generator().manual_seed(i)),
                    "slot_mask": torch.ones(1, 3), "center_mask": torch.ones(1, 3, 2),
                    "y": y, "y_mask": torch.ones(1, 12), "y_weight": torch.ones(1, 12)})
    return out


@pytest.mark.parametrize("name", TIMM)
def test_production_guard_refuses_tiny_and_unregistered_encoders(name):
    with pytest.raises(TypeError, match="tiny test encoder"):
        StudyModel(_tiny(name))
    tiny = _tiny(name)
    tiny.is_test_encoder = False  # a relabelled tiny config still fails the registered geometry
    with pytest.raises(TypeError, match="geometry"):
        StudyModel(tiny)
    with pytest.raises(TypeError, match="registered encoder family"):
        StudyModel(TinyEncoder())
    model = StudyModel(_tiny(name), allow_test_encoder=True)
    assert model.encoder_name == name and model.encoder_config["test_config"] is True


@pytest.mark.parametrize("name, units", [("convnext_tiny", 18), ("coatnet_0_rw_224", 4)])
def test_real_family_geometry_passes_guard_without_download(name, units):
    encoder = families.build_encoder(name, 224)
    model = StudyModel(encoder)
    spec = families.get_family(name)
    assert model.encoder_name == name and encoder.embed_dim == spec.embed_dim == 768
    assert len(families.encoder_layers(encoder)[1]) == units
    assert encoder.provenance["pretrained"] is False
    # Exposed normalization matches timm's pretrained config for the family.
    assert tuple(encoder.model.pretrained_cfg["mean"]) == spec.mean
    assert tuple(encoder.model.pretrained_cfg["std"]) == spec.std


@pytest.mark.parametrize("encoder", [Dinov2ViTS14(img_size=28), _tiny("convnext_tiny"), _tiny("coatnet_0_rw_224")])
def test_layers_cover_every_parameter_once(encoder):
    stem, units, top = families.encoder_layers(encoder)
    ids = [id(p) for p in stem] + [id(p) for unit in units for p in unit] + [id(p) for p in top]
    assert len(ids) == len(set(ids)) == len(list(encoder.parameters()))


def test_convnext_last_n_units_include_stage_downsample_and_all_trains_everything():
    model = StudyModel(_tiny("convnext_tiny"), allow_test_encoder=True)
    model.unfreeze_encoder(1)  # tiny config: one block per stage, so N=1 is the whole last stage
    last = model.encoder.model.stages[-1]
    assert all(p.requires_grad for p in last.parameters())
    assert not any(p.requires_grad for p in model.encoder.model.stem.parameters())
    assert not any(p.requires_grad for p in model.encoder.model.head.parameters())
    model.unfreeze_encoder("all")
    assert all(p.requires_grad for p in model.encoder.parameters())
    with pytest.raises(ValueError, match="invalid number"):
        model.unfreeze_encoder(5)
    with pytest.raises(ValueError, match="invalid number"):
        model.unfreeze_encoder(0)


def test_layer_decay_groups_scale_by_depth_and_cover_trainables():
    model = StudyModel(Dinov2ViTS14(img_size=28))
    model.unfreeze_encoder("all")
    groups = model.backbone_param_groups(1e-4, 0.5)
    lrs = [g["lr"] for g in groups]
    assert lrs[0] == pytest.approx(1e-4 * 0.5**12)  # stem
    assert lrs[-2] == lrs[-1] == pytest.approx(1e-4)  # last block and final norm
    assert lrs[1:-1] == pytest.approx([1e-4 * 0.5**k for k in range(11, -1, -1)])
    optimizer = _optimizer(model, 1e-3, 1e-4, True, 0.5)
    covered = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert sorted(covered) == sorted(id(p) for p in model.parameters() if p.requires_grad)
    assert len(covered) == len(set(covered))
    with pytest.raises(ValueError, match="layer_decay"):
        model.backbone_param_groups(1e-4, 1.5)


def test_full_adaptation_with_decay_resume_equivalence_and_refusals(tmp_path):
    def fresh():
        torch.manual_seed(0)
        return StudyModel(Dinov2ViTS14(img_size=28))

    args = dict(epochs=2, effective_batch=1, seed=0, unfreeze_after_steps=1, unfreeze_blocks="all", layer_decay=0.8)
    batches = _batches(28)
    full, part, resumed = fresh(), fresh(), fresh()
    stem_before = full.encoder.patch_embed.proj.weight.detach().clone()
    train_study_model(full, batches, ckpt_path=tmp_path / "full.pt", **args)
    train_study_model(part, batches, ckpt_path=tmp_path / "part.pt", interrupt_after_steps=2, **args)
    train_study_model(resumed, batches, ckpt_path=tmp_path / "part.pt", resume=True, **args)
    assert not torch.equal(stem_before, full.encoder.patch_embed.proj.weight)
    for key, value in full.state_dict().items():
        assert torch.allclose(value, resumed.state_dict()[key], atol=1e-6), key
    with pytest.raises(RuntimeError, match="block count changed"):
        train_study_model(fresh(), batches, ckpt_path=tmp_path / "part.pt", resume=True, **{**args, "unfreeze_blocks": 4})
    with pytest.raises(RuntimeError, match="layer decay changed"):
        train_study_model(fresh(), batches, ckpt_path=tmp_path / "part.pt", resume=True, **{**args, "layer_decay": 0.7})
    with pytest.raises(RuntimeError, match="layer decay changed"):
        train_study_model(fresh(), batches, ckpt_path=tmp_path / "part.pt", resume=True, **{**args, "layer_decay": None})


@pytest.mark.parametrize("name", TIMM)
def test_tiny_family_trains_and_refuses_cross_family_resume(tmp_path, name):
    model = StudyModel(_tiny(name), allow_test_encoder=True)
    before = [p.detach().clone() for p in families.encoder_layers(model.encoder)[1][-1]]
    out = train_study_model(model, _batches(), epochs=1, effective_batch=1, seed=0, unfreeze_after_steps=0,
                            unfreeze_blocks=1, ckpt_path=tmp_path / "tiny.pt")
    assert out["encoder_name"] == name and out["step"] == 2
    after = families.encoder_layers(model.encoder)[1][-1]
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))
    with pytest.raises(RuntimeError, match="cross-family resume"):
        train_study_model(StudyModel(Dinov2ViTS14(img_size=28)), _batches(28), epochs=1, effective_batch=1,
                          ckpt_path=tmp_path / "tiny.pt", resume=True)
    other = TIMM[1 - TIMM.index(name)]
    with pytest.raises(RuntimeError, match="cross-family resume"):
        train_study_model(StudyModel(_tiny(other), allow_test_encoder=True), _batches(), epochs=1,
                          effective_batch=1, ckpt_path=tmp_path / "tiny.pt", resume=True)
    # Evaluation refuses a tiny-encoder checkpoint unless explicitly in test mode.
    with pytest.raises(RuntimeError, match="tiny test encoder"):
        load_trained_study(tmp_path / "tiny.pt", 64, torch.device("cpu"))
    loaded, _ = load_trained_study(tmp_path / "tiny.pt", 64, torch.device("cpu"), allow_test_encoder=True)
    batch = _batches()[0]
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(batch["images"], batch["slot_mask"], batch["center_mask"]),
                              loaded(batch["images"], batch["slot_mask"], batch["center_mask"]), atol=1e-5)


def test_coatnet_batchnorm_statistics_stay_frozen_in_training():
    model = StudyModel(_tiny("coatnet_0_rw_224"), allow_test_encoder=True)
    model.unfreeze_encoder("all")
    norms = [m for m in model.encoder.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert norms and model.training and not any(m.training for m in norms)
    before = [m.running_mean.clone() for m in norms]
    batch = _batches()[0]
    model(batch["images"] * 3, batch["slot_mask"], batch["center_mask"]).sum().backward()
    assert all(torch.equal(a, m.running_mean) for a, m in zip(before, norms, strict=True))


@pytest.mark.parametrize("name", TIMM)
def test_local_weights_require_matching_sha256_and_keys(tmp_path, name):
    from rsna_knee.hashing import sha256_file

    source = _tiny(name)
    state = dict(source.model.state_dict())
    state["head.fc.weight"] = torch.zeros(10, 32)  # classifier keys from an ImageNet checkpoint are dropped
    path = tmp_path / "weights.pth"
    torch.save(state, path)
    digest = sha256_file(path)
    with pytest.raises(RuntimeError, match="sha256"):
        families.load_encoder(name, path, 64, test_config=True)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        families.load_encoder(name, path, 64, expected_sha256="0" * 64, test_config=True)
    loaded = families.load_encoder(name, path, 64, expected_sha256=digest, test_config=True)
    assert loaded.provenance["sha256"] == digest and loaded.provenance["pretrained"] is True
    for key, value in source.model.state_dict().items():
        assert torch.equal(value, loaded.model.state_dict()[key])
    state.pop(next(iter(source.model.state_dict())))
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        families.load_encoder(name, path, 64, expected_sha256=sha256_file(path), test_config=True)


def test_timm_weight_trust_needs_a_local_allowlist(tmp_path, monkeypatch):
    path = tmp_path / "w.pth"
    torch.save({}, path)
    assert families.weight_trust_for("convnext_tiny", path)["allowlisted"] is False
    allow = tmp_path / "allow.json"
    from rsna_knee.hashing import sha256_file
    allow.write_text(f'{{"sha256": ["{sha256_file(path)}"]}}')
    monkeypatch.setenv("RSNA_CONVNEXT_TINY_ALLOWLIST", str(allow))
    trust = families.weight_trust_for("convnext_tiny", path)
    assert trust["allowlisted"] is True and trust["official"] is False


def test_img_size_validation_per_family():
    assert families.validate_img_size("dinov2_vits14", 308) == 308
    assert families.validate_img_size("convnext_tiny", 320) == 320
    with pytest.raises(ValueError, match="multiple of 14"):
        families.validate_img_size("dinov2_vits14", 256)
    with pytest.raises(ValueError, match="multiple of 32"):
        families.validate_img_size("coatnet_0_rw_224", 308)
    with pytest.raises(ValueError, match="unregistered"):
        families.validate_img_size("vit_huge", 224)
