"""Focused tests for deadlock/livelock metrics in the multi-agent environment."""

from __future__ import annotations

import numpy as np

from src.environments.actions import LEFT, NO_OP, RIGHT
from src.environments.reference_model_multi_agent import ReferenceModel


def _env_config(**overrides) -> dict:
    config = {
        "env_name": "ReferenceModel-1-3",
        "seed": 123,
        "deterministic": False,
        "num_agents": 2,
        "steps_per_episode": 50,
        "sensor_range": 1,
        "info_mode": "lite",
        "training_execution_mode": "CTDE",
        "render_env": False,
        "deadlock_window_steps": 2,
        "livelock_window_steps": 4,
        "lock_nearby_manhattan": 2,
        "lock_progress_epsilon": 1,
        "lock_min_neighbors": 1,
    }
    config.update(overrides)
    return config


def _set_state(env: ReferenceModel, positions: list[tuple[int, int]], goals: list[tuple[int, int]]) -> None:
    np.copyto(env._positions_arr, np.asarray(positions, dtype=env._coord_dtype))
    np.copyto(env._starts_arr, np.asarray(positions, dtype=env._coord_dtype))
    np.copyto(env._goals_arr, np.asarray(goals, dtype=env._coord_dtype))
    env._rebuild_goal_owner()
    env._rebuild_occupancy_owner()
    env._reached_arr[:] = False
    env.goal_reached_once = dict.fromkeys(env.agents, False)
    env._reset_lock_tracking()
    env.step_count = 0


def test_deadlock_detects_on_goal_blocker():
    env = ReferenceModel(_env_config())
    env.reset()
    _set_state(
        env,
        positions=[(2, 0), (2, 1)],
        goals=[(2, 2), (2, 1)],
    )

    actions = {"agent_0": RIGHT, "agent_1": NO_OP}
    _obs, _rewards, _term, _trunc, info_1 = env.step(actions)
    _obs, _rewards, _term, _trunc, info_2 = env.step(actions)
    _obs, _rewards, _term, _trunc, info_3 = env.step(actions)

    assert info_1["__all__"]["deadlock_event_step"] == 0.0
    assert info_2["__all__"]["deadlock_step"] == 1.0
    assert info_2["__all__"]["deadlock_event_step"] == 1.0
    assert info_2["__all__"]["livelock_step"] == 0.0
    assert info_2["__all__"]["deadlock_events_total"] == 1.0
    assert info_3["__all__"]["deadlock_event_step"] == 0.0


def test_deadlock_uses_current_state_not_sticky_reached_flags():
    env = ReferenceModel(_env_config())
    env.reset()
    _set_state(
        env,
        positions=[(2, 0), (2, 2)],
        goals=[(2, 1), (4, 2)],
    )

    env.step({"agent_0": RIGHT, "agent_1": NO_OP})
    assert env.goal_reached_once["agent_0"] is True

    env.step({"agent_0": LEFT, "agent_1": LEFT})
    assert tuple(map(int, env.positions["agent_0"])) == (2, 0)
    assert tuple(map(int, env.goals["agent_0"])) == (2, 1)

    env.step({"agent_0": RIGHT, "agent_1": NO_OP})
    _obs, _rewards, _term, _trunc, info_4 = env.step({"agent_0": RIGHT, "agent_1": NO_OP})

    assert info_4["__all__"]["deadlock_step"] == 1.0
    assert info_4["__all__"]["deadlock_event_step"] == 1.0
