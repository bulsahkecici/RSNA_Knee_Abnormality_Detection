"""Prepared P1/P2 arms: valid, controlled against their control, and accepted by the job path."""
import copy

import pytest

from rsna_knee.training.experiments import (
    QUEUE_IDS,
    controlled_changes,
    job_train_spec,
    load_arms,
    validate_arm,
)

ARMS = load_arms()


def test_every_queue_item_has_exactly_one_prepared_arm():
    assert sorted(ARMS) == sorted(QUEUE_IDS)
    assert all(arm["status"] == "prepared" for arm in ARMS.values())


@pytest.mark.parametrize("arm_id", QUEUE_IDS)
def test_arm_is_valid_and_changes_only_what_it_declares(arm_id):
    arm = ARMS[arm_id]
    assert validate_arm(arm, ARMS) == []
    assert arm["folds"] == "state/folds-v2-allgold.csv"
    assert job_train_spec(arm)["init"] == "official_pretrained"
    if arm["control"]:
        assert controlled_changes(arm, ARMS[arm["control"]]) == set(arm["changes"])


def test_an_undeclared_change_is_rejected():
    arm = copy.deepcopy(ARMS["D3"])
    arm["train"]["seed"] = 7
    assert any(error.startswith("undeclared_or_missing_changes") for error in validate_arm(arm, ARMS))
    arm = copy.deepcopy(ARMS["C1"])
    arm["weights"]["encoder"] = "dinov2_vits14"
    assert "weights_encoder_mismatch" in validate_arm(arm, ARMS)
    arm = copy.deepcopy(ARMS["D2"])
    arm["train"]["pooling"] = "plane_concat"
    assert "plane_concat_requires_three_slots" in validate_arm(arm, ARMS)
    arm = copy.deepcopy(ARMS["D0"])
    arm["train"]["init"] = "supervised_checkpoint"
    assert "init_not_official_pretrained" in validate_arm(arm, ARMS)


def test_family_depths_match_the_dino_fraction_where_declared():
    assert ARMS["C1"]["train"]["unfreeze_blocks"] == 3  # 3/18 ConvNeXt blocks = 2/12 ViT blocks
    assert ARMS["C2"]["train"]["unfreeze_blocks"] == 1  # last CoAtNet stage
    assert ARMS["D4"]["train"]["unfreeze_blocks"] == "all" and ARMS["D4"]["train"]["layer_decay"] == 0.75
