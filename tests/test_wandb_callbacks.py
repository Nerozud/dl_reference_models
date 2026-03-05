"""Tests for PB2/PBT-safe W&B callback behavior."""

from __future__ import annotations

from src.trainers.wandb_callbacks import (
    PBTSafeWandbLoggerCallback,
    _PBTSafeWandbLoggingActor,
    _QueueItem,
)


class _FakeConfig:
    def __init__(self):
        self.trial_log_path = None
        self.update_calls = []

    def update(self, payload, allow_val_change):
        self.update_calls.append((payload, allow_val_change))


class _FakeRun:
    def __init__(self):
        self.config = _FakeConfig()
        self.define_metric_calls = []
        self.log_calls = []

    def define_metric(self, *args, **kwargs):
        self.define_metric_calls.append((args, kwargs))

    def log(self, payload, *args, **kwargs):
        self.log_calls.append((payload, args, kwargs))


class _FakeWandb:
    def __init__(self, run):
        self.run = run
        self.init_calls = []
        self.finish_calls = 0

    def init(self, *args, **kwargs):
        self.init_calls.append((args, kwargs))
        return self.run

    def finish(self):
        self.finish_calls += 1


class _FakeQueue:
    def __init__(self, items):
        self._items = list(items)

    def get(self):
        return self._items.pop(0)


def _build_actor(run_items):
    run = _FakeRun()
    actor = object.__new__(_PBTSafeWandbLoggingActor)
    actor._wandb = _FakeWandb(run)
    actor.queue = _FakeQueue(run_items)
    actor._exclude = set()
    actor._to_config = set()
    actor.args = ()
    actor.kwargs = {"name": "trial-name"}
    actor._trial_name = "trial-name"
    actor._logdir = "/tmp/fake-trial-logdir"
    return actor, run


def test_pbtsafe_callback_uses_custom_actor_class():
    assert PBTSafeWandbLoggerCallback._logger_actor_cls is _PBTSafeWandbLoggingActor


def test_pbtsafe_actor_defines_training_iteration_metric(monkeypatch):
    monkeypatch.setattr("src.trainers.wandb_callbacks._run_wandb_process_run_info_hook", lambda _run: None)
    actor, run = _build_actor([(_QueueItem.END, None)])

    actor.run()

    assert (("training_iteration",), {}) in run.define_metric_calls
    assert (("*",), {"step_metric": "training_iteration"}) in run.define_metric_calls


def test_pbtsafe_actor_logs_without_explicit_step_argument(monkeypatch):
    monkeypatch.setattr("src.trainers.wandb_callbacks._run_wandb_process_run_info_hook", lambda _run: None)
    actor, run = _build_actor(
        [
            (_QueueItem.RESULT, {"training_iteration": 7, "metric": 1.23, "config": {"x": 5}}),
            (_QueueItem.END, None),
        ]
    )

    actor.run()

    assert run.log_calls
    payload, args, kwargs = run.log_calls[0]
    assert payload["training_iteration"] == 7
    assert payload["metric"] == 1.23
    assert args == ()
    assert kwargs == {}
