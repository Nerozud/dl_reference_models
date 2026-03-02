"""Tests for PPO minibatch defaults."""

from src.agents.ppo import get_ppo_config


def _base_env_config(training_execution_mode: str) -> dict:
    return {
        "training_execution_mode": training_execution_mode,
        "render_env": False,
        "num_agents": 2,
    }


def test_get_ppo_config_sets_minibatch_1024_for_cte():
    config = get_ppo_config("ReferenceModel-2-1", env_config=_base_env_config("CTE"))

    assert config.to_dict()["minibatch_size"] == 1024


def test_get_ppo_config_sets_minibatch_1024_for_ctde():
    config = get_ppo_config("ReferenceModel-2-1", env_config=_base_env_config("CTDE"))

    assert config.to_dict()["minibatch_size"] == 1024
