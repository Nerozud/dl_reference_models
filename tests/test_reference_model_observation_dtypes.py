"""Smoke tests for observation dtype/bounds consistency."""

from __future__ import annotations

import numpy as np
import pytest

from src.environments.reference_model_multi_agent import ReferenceModel as MultiAgentReferenceModel
from src.environments.reference_model_single_agent import ReferenceModel as SingleAgentReferenceModel


def _single_env_config(**overrides) -> dict:
    config = {
        "env_name": "ReferenceModel-2-1",
        "seed": 123,
        "deterministic": True,
        "num_agents": 4,
        "steps_per_episode": 20,
        "sensor_range": 2,
        "render_env": False,
        "validate_observation_space": True,
    }
    config.update(overrides)
    return config


def _multi_env_config(**overrides) -> dict:
    config = {
        "env_name": "ReferenceModel-2-1",
        "seed": 123,
        "deterministic": True,
        "num_agents": 4,
        "steps_per_episode": 20,
        "sensor_range": 2,
        "info_mode": "lite",
        "training_execution_mode": "CTDE",
        "render_env": False,
        "validate_observation_space": True,
    }
    config.update(overrides)
    return config


def test_single_agent_reset_and_step_observation_is_float32_and_within_space():
    env = SingleAgentReferenceModel(_single_env_config())

    obs, _info = env.reset()
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)

    action_mask = obs[env._obs_slices["action_mask"]]
    assert np.all(np.isin(action_mask, np.array([0.0, 1.0], dtype=np.float32)))

    next_obs, _reward, _terminated, _truncated, _step_info = env.step([0] * env.num_agents)
    assert next_obs.dtype == np.float32
    assert env.observation_space.contains(next_obs)


@pytest.mark.parametrize("normalize_goal_delta", [True, False])
@pytest.mark.parametrize("include_goal_distance", [False, True])
def test_multi_agent_reset_and_step_observations_are_float32_and_within_space(
    normalize_goal_delta: bool,
    include_goal_distance: bool,
):
    env = MultiAgentReferenceModel(
        _multi_env_config(
            normalize_goal_delta=normalize_goal_delta,
            include_goal_distance=include_goal_distance,
        )
    )

    obs, _infos = env.reset()
    assert obs
    for agent_id, agent_obs in obs.items():
        assert agent_obs.dtype == np.float32
        assert env.observation_space.contains(agent_obs), agent_id
        action_mask = agent_obs[env._obs_slices["action_mask"]]
        assert np.all(np.isin(action_mask, np.array([0.0, 1.0], dtype=np.float32)))

    actions = dict.fromkeys(env.agents, 0)
    next_obs, _rewards, _terminated, _truncated, _step_infos = env.step(actions)
    for agent_id, agent_obs in next_obs.items():
        assert agent_obs.dtype == np.float32
        assert env.observation_space.contains(agent_obs), agent_id

