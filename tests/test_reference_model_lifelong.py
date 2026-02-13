"""Tests for lifelong MAPF behavior in the multi-agent reference environment."""

from __future__ import annotations

import numpy as np
import pytest

from src.environments.actions import DOWN, LEFT, NO_OP, RIGHT, UP
from src.environments.reference_model_multi_agent import ReferenceModel


def _env_config(num_agents: int, **overrides) -> dict:
    config = {
        "env_name": "ReferenceModel-2-1",
        "seed": 123,
        "deterministic": False,
        "num_agents": num_agents,
        "steps_per_episode": 40,
        "sensor_range": 2,
        "info_mode": "lite",
        "lifelong_mapf": True,
        "training_execution_mode": "CTDE",
        "render_env": False,
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
    env._completed_once_arr[:] = False
    env.goal_reached_once = dict.fromkeys(env.agents, False)
    env._episode_goals_reached_total = 0.0
    env.step_count = 0


def _find_adjacent_pair(
    env: ReferenceModel,
    forbidden: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    free = {tuple(map(int, p)) for p in env._free_positions}
    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    for src in sorted(free):
        if src in forbidden:
            continue
        for dy, dx in deltas:
            dst = (src[0] + dy, src[1] + dx)
            if dst in free and dst not in forbidden:
                return src, dst
    raise RuntimeError("Could not find adjacent free-cell pair for test setup.")


def _action_towards(src: tuple[int, int], dst: tuple[int, int]) -> int:
    dy = dst[0] - src[0]
    dx = dst[1] - src[1]
    if (dy, dx) == (-1, 0):
        return UP
    if (dy, dx) == (0, 1):
        return RIGHT
    if (dy, dx) == (1, 0):
        return DOWN
    if (dy, dx) == (0, -1):
        return LEFT
    msg = f"Cells are not adjacent: {src} -> {dst}"
    raise ValueError(msg)


def test_reassigns_goal_immediately_after_reach():
    env = ReferenceModel(_env_config(num_agents=2))
    env.reset()

    pair_0 = _find_adjacent_pair(env, forbidden=set())
    pair_1 = _find_adjacent_pair(env, forbidden={pair_0[0], pair_0[1]})
    _set_state(env, positions=[pair_0[0], pair_1[0]], goals=[pair_0[1], pair_1[1]])
    old_goal_agent_0 = pair_0[1]

    actions = {
        "agent_0": _action_towards(pair_0[0], pair_0[1]),
        "agent_1": NO_OP,
    }
    _obs, _rewards, _term, _trunc, info = env.step(actions)

    new_goal_agent_0 = tuple(map(int, env.goals["agent_0"]))
    all_positions = {tuple(map(int, env.positions[aid])) for aid in env.agents}

    assert info["agent_0"]["goal_reached_step"] == 1.0
    assert new_goal_agent_0 != old_goal_agent_0
    assert new_goal_agent_0 not in all_positions
    assert new_goal_agent_0 != tuple(map(int, env.goals["agent_1"]))


def test_goals_remain_unique_and_unoccupied():
    env = ReferenceModel(_env_config(num_agents=4))
    rng = np.random.default_rng(2026)
    env.reset()

    for _ in range(240):
        actions = {agent_id: int(rng.integers(0, 5)) for agent_id in env.agents}
        _obs, _rewards, terminated, truncated, _info = env.step(actions)

        goals = [tuple(map(int, env.goals[agent_id])) for agent_id in env.agents]
        assert len(goals) == len(set(goals))
        for goal_y, goal_x in goals:
            assert env.grid[goal_y, goal_x] == env.EMPTY_CELL
        for agent_id in env.agents:
            assert tuple(map(int, env.goals[agent_id])) != tuple(map(int, env.positions[agent_id]))

        done = terminated.get("__all__", False) or truncated.get("__all__", False)
        if done:
            env.reset()


def test_no_early_termination_in_lifelong():
    env = ReferenceModel(_env_config(num_agents=1, steps_per_episode=10))
    env.reset()

    src, dst = _find_adjacent_pair(env, forbidden=set())
    _set_state(env, positions=[src], goals=[dst])

    action = {"agent_0": _action_towards(src, dst)}
    _obs, _rewards, terminated, truncated, _info = env.step(action)

    assert terminated["__all__"] is False
    assert truncated["__all__"] is False


def test_cumulative_goals_can_exceed_num_agents(monkeypatch: pytest.MonkeyPatch):
    env = ReferenceModel(_env_config(num_agents=1, steps_per_episode=12))
    env.reset()

    src, dst = _find_adjacent_pair(env, forbidden=set())
    _set_state(env, positions=[src], goals=[dst])

    def _assign_adjacent_goal(agent_idx: int) -> np.ndarray:
        old_goal = env._goals_arr[agent_idx]
        env._goal_owner[int(old_goal[0]), int(old_goal[1])] = env.UNASSIGNED_OWNER

        pos_y = int(env._positions_arr[agent_idx, 0])
        pos_x = int(env._positions_arr[agent_idx, 1])
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_y = pos_y + dy
            next_x = pos_x + dx
            if not (0 <= next_y < env.grid.shape[0] and 0 <= next_x < env.grid.shape[1]):
                continue
            if env.grid[next_y, next_x] != env.EMPTY_CELL:
                continue
            if env._occupancy_owner[next_y, next_x] != env.UNASSIGNED_OWNER:
                continue
            if env._goal_owner[next_y, next_x] != env.UNASSIGNED_OWNER:
                continue
            env._goals_arr[agent_idx, :] = [next_y, next_x]
            env._goal_owner[next_y, next_x] = agent_idx
            return env._goals_arr[agent_idx]

        raise RuntimeError("Could not assign adjacent goal in test setup.")

    monkeypatch.setattr(env, "_assign_new_goal", _assign_adjacent_goal)

    info = {}
    done = False
    while not done:
        pos = tuple(map(int, env.positions["agent_0"]))
        goal = tuple(map(int, env.goals["agent_0"]))
        action = {"agent_0": _action_towards(pos, goal)}
        _obs, _rewards, terminated, truncated, info = env.step(action)
        done = terminated.get("__all__", False) or truncated.get("__all__", False)

    assert info["__all__"]["goals_reached_total"] > float(env._num_agents)


def test_completion_ratio_and_throughput_reported():
    env = ReferenceModel(_env_config(num_agents=2, steps_per_episode=20))
    env.reset()

    pair_0 = _find_adjacent_pair(env, forbidden=set())
    pair_1 = _find_adjacent_pair(env, forbidden={pair_0[0], pair_0[1]})
    _set_state(env, positions=[pair_0[0], pair_1[0]], goals=[pair_0[1], pair_1[1]])

    actions = {
        "agent_0": _action_towards(pair_0[0], pair_0[1]),
        "agent_1": NO_OP,
    }
    _obs, _rewards, _term, _trunc, info = env.step(actions)
    info_all = info["__all__"]

    assert "completion_ratio" in info_all
    assert "throughput" in info_all
    assert info_all["completion_ratio"] == pytest.approx(0.5)
    assert info_all["throughput"] == pytest.approx(info_all["goals_reached_total"] / float(env.step_count))
