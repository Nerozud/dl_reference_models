"""Invariant tests for the multi-agent reference environment."""

from __future__ import annotations

import numpy as np
import pytest

from src.environments.reference_model_multi_agent import ReferenceModel


def _env_config(deterministic: bool = False, info_mode: str = "lite", **overrides) -> dict:
    config = {
        "env_name": "ReferenceModel-2-1",
        "seed": 123,
        "deterministic": deterministic,
        "num_agents": 4,
        "steps_per_episode": 100,
        "sensor_range": 2,
        "info_mode": info_mode,
        "training_execution_mode": "CTDE",
        "render_env": False,
    }
    config.update(overrides)
    return config


def test_unique_starts_goals_and_disjoint_sets():
    env = ReferenceModel(_env_config(deterministic=False))
    for _ in range(100):
        env.reset()
        starts = [tuple(map(int, env.starts[agent_id])) for agent_id in env.agents]
        goals = [tuple(map(int, env.goals[agent_id])) for agent_id in env.agents]
        assert len(starts) == len(set(starts))
        assert len(goals) == len(set(goals))
        assert set(starts).isdisjoint(set(goals))


def test_positions_stay_in_bounds_and_on_free_cells():
    env = ReferenceModel(_env_config(deterministic=False))
    rng = np.random.default_rng(77)
    obs, _ = env.reset()

    for _ in range(600):
        actions = {agent_id: int(rng.integers(0, 5)) for agent_id in env.agents}
        obs, _rewards, terminated, truncated, _infos = env.step(actions)
        positions = []
        for agent_id in env.agents:
            y, x = map(int, env.positions[agent_id])
            positions.append((y, x))
            assert 0 <= y < env.grid.shape[0]
            assert 0 <= x < env.grid.shape[1]
            assert env.grid[y, x] == env.EMPTY_CELL
        assert len(set(positions)) == len(positions)

        done = terminated.get("__all__", False) or truncated.get("__all__", False)
        if done:
            obs, _ = env.reset()

    assert obs


def test_action_mask_matches_local_observation():
    env = ReferenceModel(_env_config(deterministic=True))
    env.reset()
    traversable = set(env.TRAVERSABLE_LOCAL_VALUES)
    center = env.sensor_range

    for agent_id in env.agents:
        local_obs = env.get_obs(agent_id)
        action_mask = env.get_action_mask(local_obs)
        assert int(action_mask[0]) == 1

        can_up = center > 0 and int(local_obs[center - 1, center]) in traversable
        can_right = center < local_obs.shape[1] - 1 and int(local_obs[center, center + 1]) in traversable
        can_down = center < local_obs.shape[0] - 1 and int(local_obs[center + 1, center]) in traversable
        can_left = center > 0 and int(local_obs[center, center - 1]) in traversable

        assert int(action_mask[1]) == int(can_up)
        assert int(action_mask[2]) == int(can_right)
        assert int(action_mask[3]) == int(can_down)
        assert int(action_mask[4]) == int(can_left)


def test_action_mask_space_uses_python_int_dim():
    env = ReferenceModel(_env_config(deterministic=True))
    assert int(env.action_space.n) == 5
    assert env._action_mask_space.shape == (5,)


def test_info_mode_lite_and_full_payloads_have_same_metrics():
    lite_env = ReferenceModel(_env_config(deterministic=True, info_mode="lite"))
    full_env = ReferenceModel(_env_config(deterministic=True, info_mode="full"))
    rng = np.random.default_rng(2026)

    lite_obs, lite_info = lite_env.reset()
    full_obs, full_info = full_env.reset()
    assert set(lite_obs) == set(full_obs)
    for agent_id in lite_env.agents:
        np.testing.assert_array_equal(lite_obs[agent_id], full_obs[agent_id])
        assert lite_info[agent_id] == {}
        assert "local_obs" in full_info[agent_id]
        assert "action_mask" in full_info[agent_id]

    for _ in range(120):
        actions = {agent_id: int(rng.integers(0, 5)) for agent_id in lite_env.agents}
        lite_obs, lite_rewards, lite_term, lite_trunc, lite_info = lite_env.step(actions)
        full_obs, full_rewards, full_term, full_trunc, full_info = full_env.step(actions)

        assert lite_rewards == full_rewards
        assert lite_term == full_term
        assert lite_trunc == full_trunc
        assert lite_info["__all__"] == full_info["__all__"]

        for agent_id in lite_env.agents:
            np.testing.assert_array_equal(lite_obs[agent_id], full_obs[agent_id])
            for metric_key in ("blocking", "goal_reached_step", "goals_reached_total", "blocking_count_total"):
                assert lite_info[agent_id][metric_key] == full_info[agent_id][metric_key]
            assert "local_obs" not in lite_info[agent_id]
            assert "action_mask" not in lite_info[agent_id]
            assert "local_obs" in full_info[agent_id]
            assert "action_mask" in full_info[agent_id]

        done = lite_term.get("__all__", False) or lite_trunc.get("__all__", False)
        if done:
            lite_env.reset()
            full_env.reset()


def test_invalid_info_mode_raises():
    with pytest.raises(ValueError, match="Unsupported info_mode"):
        ReferenceModel(_env_config(info_mode="invalid"))
