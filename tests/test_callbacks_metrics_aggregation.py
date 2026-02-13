"""Tests for callback metrics aggregation behavior."""

from __future__ import annotations

from types import SimpleNamespace

from src.trainers.callbacks import EpisodeMetricsCallback, SuccessRateCallback


class _DummyEpisode:
    def __init__(self, infos):
        self._infos = infos
        self.custom_data = None

    def get_last_infos(self):
        return self._infos


class _DummyDoneEpisode:
    def __init__(self, terminated, truncated):
        self._terminated = terminated
        self._truncated = truncated
        self.custom_data = {}

    def is_terminated(self):
        return self._terminated

    def is_truncated(self):
        return self._truncated


class _DummyMetricsLogger:
    def __init__(self):
        self.values = {}

    def log_value(self, key, value, **_kwargs):
        self.values[key] = value


def test_episode_step_prefers_all_metrics_without_double_counting():
    callback = EpisodeMetricsCallback()
    infos = {
        "__all__": {
            "goals_reached_step": 2.0,
            "blocking_count_step": 1.0,
            "deadlock_event_step": 1.0,
            "livelock_event_step": 0.0,
            "deadlock_step": 1.0,
            "livelock_step": 0.0,
        },
        "agent_0": {
            "goal_reached_step": 1.0,
            "blocking": 1.0,
            "deadlock_event_step": 5.0,
            "deadlock_step": 5.0,
        },
        "agent_1": {
            "goal_reached_step": 1.0,
            "blocking": 0.0,
            "livelock_event_step": 4.0,
            "livelock_step": 4.0,
        },
    }
    episode = _DummyEpisode(infos)
    callback.on_episode_start(episode=episode)
    callback.on_episode_step(episode=episode)

    assert episode.custom_data["goals_reached"] == 2.0
    assert episode.custom_data["blocking_count"] == 1.0
    assert episode.custom_data["deadlock_count"] == 1.0
    assert episode.custom_data["livelock_count"] == 0.0
    assert episode.custom_data["deadlock_steps"] == 1.0
    assert episode.custom_data["livelock_steps"] == 0.0


def test_episode_step_falls_back_to_agent_level_metrics():
    callback = EpisodeMetricsCallback()
    infos = {
        "agent_0": {
            "goal_reached_step": 1.0,
            "blocking": 1.0,
            "deadlock_event_step": 1.0,
            "livelock_event_step": 0.0,
            "deadlock_step": 1.0,
            "livelock_step": 0.0,
        },
        "agent_1": {
            "goal_reached_step": 0.0,
            "blocking": 0.0,
            "deadlock_event_step": 0.0,
            "livelock_event_step": 1.0,
            "deadlock_step": 0.0,
            "livelock_step": 1.0,
        },
    }
    episode = _DummyEpisode(infos)
    callback.on_episode_start(episode=episode)
    callback.on_episode_step(episode=episode)

    assert episode.custom_data["goals_reached"] == 1.0
    assert episode.custom_data["blocking_count"] == 1.0
    assert episode.custom_data["deadlock_count"] == 1.0
    assert episode.custom_data["livelock_count"] == 1.0
    assert episode.custom_data["deadlock_steps"] == 1.0
    assert episode.custom_data["livelock_steps"] == 1.0


def test_success_rate_finite_mode_uses_done_flags():
    callback = SuccessRateCallback()
    episode = _DummyDoneEpisode(terminated=True, truncated=False)
    logger = _DummyMetricsLogger()

    callback.on_episode_end(episode=episode, metrics_logger=logger, env=None, env_runner=None, env_index=0)

    assert logger.values["success_rate"] == 1.0


def test_success_rate_lifelong_mode_uses_completion_ratio():
    callback = SuccessRateCallback()
    episode = _DummyDoneEpisode(terminated=False, truncated=True)
    logger = _DummyMetricsLogger()
    env = SimpleNamespace(
        lifelong_mapf=True,
        _completed_once_arr=[True, False, True, False],
    )

    callback.on_episode_end(episode=episode, metrics_logger=logger, env=env, env_runner=None, env_index=0)

    assert logger.values["success_rate"] == 0.5


def test_episode_end_logs_lifelong_throughput_and_completion_ratio():
    callback = EpisodeMetricsCallback()
    episode = _DummyEpisode(infos={})
    episode.custom_data = {
        "goals_reached": 0.0,
        "blocking_count": 0.0,
        "deadlock_count": 0.0,
        "livelock_count": 0.0,
        "deadlock_steps": 0.0,
        "livelock_steps": 0.0,
    }
    logger = _DummyMetricsLogger()
    env = SimpleNamespace(
        lifelong_mapf=True,
        _episode_goals_reached_total=6.0,
        _completed_once_arr=[True, False, True, True],
        step_count=20,
        _episode_blocking_count=2.0,
        _episode_deadlock_events=1.0,
        _episode_livelock_events=0.0,
        _episode_deadlock_steps=3.0,
        _episode_livelock_steps=1.0,
    )

    callback.on_episode_end(
        episode=episode,
        env_runner=None,
        metrics_logger=logger,
        env=env,
        env_index=0,
    )

    assert logger.values["goals_reached"] == 6.0
    assert logger.values["throughput"] == 0.3
    assert logger.values["completion_ratio"] == 0.75
