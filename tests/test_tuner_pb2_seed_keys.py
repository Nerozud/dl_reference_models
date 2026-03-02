"""Tests for PB2 initial-seeding parameter handling in the tuner."""

import pytest
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2

from src.trainers import tuner


def _sample_param_space():
    return {
        "lr": 1e-4,
        "entropy_coeff": 0.01,
        "num_epochs": 12,
        "clip_param": 0.2,
        "gamma": 0.99,
        "lambda_": 0.95,
        "train_batch_size_per_learner": 8000,
        "minibatch_size": 8000,
        "env_config": {"training_execution_mode": "CTDE"},
        "unchanged_value": 123,
    }


def test_drop_pb2_seed_keys_if_present_removes_selected_keys():
    param_space = _sample_param_space()

    updated = tuner._drop_pb2_seed_keys_if_present(param_space, tuner.PB2_SEED_PARAM_KEYS)

    for key in tuner.PB2_SEED_PARAM_KEYS:
        assert key not in updated
    assert updated["minibatch_size"] == 8000
    assert updated["unchanged_value"] == 123
    # Original must stay untouched.
    assert "lr" in param_space
    assert "train_batch_size_per_learner" in param_space


def test_drop_pb2_seed_keys_if_present_ignores_missing_keys():
    param_space = {"gamma": 0.99, "unchanged_value": 7}

    updated = tuner._drop_pb2_seed_keys_if_present(param_space, ["lr", "entropy_coeff"])

    assert updated == param_space
    assert updated is not param_space


def test_apply_pb2_seed_key_drop_if_needed_gates_by_scheduler_type():
    param_space = _sample_param_space()

    unchanged = tuner._apply_pb2_seed_key_drop_if_needed(param_space, scheduler=None)
    pb2_updated = tuner._apply_pb2_seed_key_drop_if_needed(param_space, scheduler=tuner.pb2_scheduler)

    assert unchanged is param_space
    for key in tuner.PB2_SEED_PARAM_KEYS:
        assert key not in pb2_updated
    assert pb2_updated["minibatch_size"] == 8000


def test_validate_pb2_batch_minibatch_compatibility_passes_for_valid_bounds():
    param_space = _sample_param_space() | {"minibatch_size": 1024}

    tuner._validate_pb2_batch_minibatch_compatibility(param_space, tuner.pb2_scheduler)


def test_validate_pb2_batch_minibatch_compatibility_raises_for_invalid_bounds():
    scheduler = PB2(
        time_attr="training_iteration",
        metric="env_runners/episode_return_mean",
        mode="max",
        perturbation_interval=10,
        hyperparam_bounds={"train_batch_size_per_learner": [1024, 16384]},
    )
    param_space = _sample_param_space() | {"minibatch_size": 2048}

    with pytest.raises(ValueError, match="train_batch_size_per_learner"):
        tuner._validate_pb2_batch_minibatch_compatibility(param_space, scheduler)


def test_get_wandb_project_name_uses_pb2_suffix_for_pb2():
    assert tuner._get_wandb_project_name("ReferenceModel-2-1", tuner.pb2_scheduler) == "ReferenceModel-2-1-pb2"


def test_get_wandb_project_name_uses_test_suffix_for_none_scheduler():
    assert tuner._get_wandb_project_name("ReferenceModel-2-1", None) == "ReferenceModel-2-1-test"


def test_get_wandb_project_name_keeps_test_suffix_for_non_pb2_scheduler():
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="env_runners/episode_return_mean",
        mode="max",
        perturbation_interval=10,
        hyperparam_mutations={"lr": [1e-5, 1e-3]},
    )
    assert tuner._get_wandb_project_name("ReferenceModel-2-1", pbt) == "ReferenceModel-2-1-test"


def test_get_num_samples_uses_population_for_pb2():
    assert tuner._get_num_samples(tuner.pb2_scheduler) == 8


def test_get_num_samples_keeps_default_for_non_pb2():
    assert tuner._get_num_samples(None) == 1
