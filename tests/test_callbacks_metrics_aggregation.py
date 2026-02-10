"""Tests for callback metrics aggregation behavior."""

from __future__ import annotations

from src.trainers.callbacks import EpisodeMetricsCallback


class _DummyEpisode:
    def __init__(self, infos):
        self._infos = infos
        self.custom_data = None

    def get_last_infos(self):
        return self._infos


def test_episode_step_prefers_all_metrics_without_double_counting():
    callback = EpisodeMetricsCallback()
    infos = {
        "__all__": {"goals_reached_step": 2.0, "blocking_count_step": 1.0},
        "agent_0": {"goal_reached_step": 1.0, "blocking": 1.0},
        "agent_1": {"goal_reached_step": 1.0, "blocking": 0.0},
    }
    episode = _DummyEpisode(infos)
    callback.on_episode_start(episode=episode)
    callback.on_episode_step(episode=episode)

    assert episode.custom_data["goals_reached"] == 2.0
    assert episode.custom_data["blocking_count"] == 1.0


def test_episode_step_falls_back_to_agent_level_metrics():
    callback = EpisodeMetricsCallback()
    infos = {
        "agent_0": {"goal_reached_step": 1.0, "blocking": 1.0},
        "agent_1": {"goal_reached_step": 0.0, "blocking": 0.0},
    }
    episode = _DummyEpisode(infos)
    callback.on_episode_start(episode=episode)
    callback.on_episode_step(episode=episode)

    assert episode.custom_data["goals_reached"] == 1.0
    assert episode.custom_data["blocking_count"] == 1.0
