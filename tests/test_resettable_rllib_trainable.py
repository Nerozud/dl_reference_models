"""Tests for the resettable RLlib Tune trainable wrapper."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from src.trainers import resettable_rllib_trainable as rrlt


class _FakeAlgo:
    instances = []
    fail_on_config: Callable[[dict], bool] | None = None

    def __init__(self, config):
        if self.fail_on_config and self.fail_on_config(config):
            msg = "boom"
            raise RuntimeError(msg)
        self.config = config
        self.stopped = False
        self.saved_to = None
        self.restored_from = None
        self.restore_kwargs = None
        type(self).instances.append(self)

    @classmethod
    def reset_state(cls):
        cls.instances = []
        cls.fail_on_config = None

    @classmethod
    def default_resource_request(cls, config):
        return ("resource", config["x"])

    @classmethod
    def resource_help(cls, config):
        return f"help-{config['x']}"

    def train(self):
        return {"metric": 1, "config_x": self.config["x"], "training_iteration": 42}

    def save_to_path(self, checkpoint_dir):
        self.saved_to = checkpoint_dir
        return f"{checkpoint_dir}/checkpoint"

    def restore_from_path(self, checkpoint, *args, **kwargs):
        self.restored_from = checkpoint
        self.restore_kwargs = kwargs

    def stop(self):
        self.stopped = True


@pytest.fixture(autouse=True)
def _reset_fake_algo():
    _FakeAlgo.reset_state()
    yield
    _FakeAlgo.reset_state()


def _make_wrapper(monkeypatch):
    monkeypatch.setattr(rrlt, "_resolve_algo_cls", lambda _name: _FakeAlgo)
    return rrlt.make_resettable_rllib_trainable("PPO")


def test_factory_supports_repo_algorithms():
    for algo_name in ("PPO", "IMPALA", "DQN"):
        wrapper_cls = rrlt.make_resettable_rllib_trainable(algo_name)
        assert wrapper_cls.__name__ == f"Resettable{algo_name}Trainable"


def test_default_resource_request_delegates(monkeypatch):
    wrapper_cls = _make_wrapper(monkeypatch)
    assert wrapper_cls.default_resource_request({"x": 3}) == ("resource", 3)
    assert wrapper_cls.resource_help({"x": 3}) == "help-3"


def test_wrapper_lifecycle_and_reset_success(monkeypatch):
    wrapper_cls = _make_wrapper(monkeypatch)
    trainable = object.__new__(wrapper_cls)

    trainable.setup({"x": 1})
    first_algo = trainable._algo

    assert trainable._current_config == {"x": 1}
    assert trainable.step() == {"metric": 1, "config_x": 1}
    assert trainable.save_checkpoint("ckpt") == "ckpt/checkpoint"
    assert first_algo.saved_to == "ckpt"

    trainable.load_checkpoint("ckpt/checkpoint")
    assert first_algo.restored_from == "ckpt/checkpoint"
    assert first_algo.restore_kwargs == {}

    assert trainable.reset_config({"x": 2}) is True
    second_algo = trainable._algo
    assert second_algo is not first_algo
    assert first_algo.stopped is True
    assert trainable._current_config == {"x": 2}
    assert trainable.step() == {"metric": 1, "config_x": 2}

    trainable.cleanup()
    assert second_algo.stopped is True
    assert trainable._algo is None


def test_step_strips_rllib_training_iteration(monkeypatch):
    wrapper_cls = _make_wrapper(monkeypatch)
    trainable = object.__new__(wrapper_cls)

    trainable.setup({"x": 1})
    result = trainable.step()

    assert result == {"metric": 1, "config_x": 1}
    assert "training_iteration" not in result


def test_load_checkpoint_weights_only_restore_mode(monkeypatch):
    monkeypatch.setattr(rrlt, "_resolve_algo_cls", lambda _name: _FakeAlgo)
    wrapper_cls = rrlt.make_resettable_rllib_trainable("PPO", checkpoint_restore_mode="weights_only")
    trainable = object.__new__(wrapper_cls)

    trainable.setup({"x": 1})
    trainable.load_checkpoint("ckpt/checkpoint")

    algo = trainable._algo
    assert algo.restored_from == Path("ckpt/checkpoint") / "learner_group" / "learner" / "rl_module"
    assert algo.restore_kwargs == {"component": "learner_group/learner/rl_module"}


def test_invalid_checkpoint_restore_mode_raises(monkeypatch):
    monkeypatch.setattr(rrlt, "_resolve_algo_cls", lambda _name: _FakeAlgo)

    with pytest.raises(ValueError, match="checkpoint_restore_mode"):
        rrlt.make_resettable_rllib_trainable("PPO", checkpoint_restore_mode="bad-mode")


def test_reset_config_returns_false_on_rebuild_failure(monkeypatch):
    wrapper_cls = _make_wrapper(monkeypatch)
    trainable = object.__new__(wrapper_cls)
    trainable.setup({"x": 1})
    first_algo = trainable._algo

    _FakeAlgo.fail_on_config = lambda cfg: cfg["x"] == 99

    assert trainable.reset_config({"x": 99}) is False
    assert first_algo.stopped is True
    assert trainable._algo is None
