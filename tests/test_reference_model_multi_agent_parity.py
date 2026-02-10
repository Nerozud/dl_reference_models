"""Parity tests for the optimized multi-agent reference environment."""

from __future__ import annotations

import hashlib

import numpy as np

from src.environments.reference_model_multi_agent import ReferenceModel


EXPECTED_STOCHASTIC_DIGEST = "d58a9e9e0e383f29c5d7f96a1338dfd73c9035dc5335f66b2ce11a0d8e0452de"
EXPECTED_STOCHASTIC_SUMMARY = [
    {"episode": 0, "steps": 100, "reward_sum": -3.5},
    {"episode": 1, "steps": 100, "reward_sum": -4.0},
    {"episode": 2, "steps": 100, "reward_sum": -2.5},
]

EXPECTED_DETERMINISTIC_DIGEST = "2612dc3eeab5b4fd69d8cbe7fb4f01e35cf52a6f05b07bc955a306ae765c2595"
EXPECTED_DETERMINISTIC_SUMMARY = [
    {"episode": 0, "steps": 100, "reward_sum": -3.5},
    {"episode": 1, "steps": 100, "reward_sum": -3.5},
    {"episode": 2, "steps": 100, "reward_sum": -4.0},
]


def _update_with_array(hasher, arr) -> None:
    np_arr = np.asarray(arr)
    hasher.update(str(np_arr.dtype).encode())
    hasher.update(str(np_arr.shape).encode())
    hasher.update(np_arr.tobytes())


def _update_with_value(hasher, value) -> None:
    if isinstance(value, dict):
        for key in sorted(value.keys(), key=lambda item: str(item)):
            hasher.update(str(key).encode())
            _update_with_value(hasher, value[key])
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _update_with_value(hasher, item)
        return
    if isinstance(value, np.ndarray):
        _update_with_array(hasher, value)
        return
    if isinstance(value, (np.floating, float)):
        hasher.update(np.float32(value).tobytes())
        return
    if isinstance(value, (np.integer, int, np.bool_, bool)):
        hasher.update(str(int(value)).encode())
        return
    if value is None:
        hasher.update(b"None")
        return
    hasher.update(str(value).encode())


def _build_env_config(deterministic: bool) -> dict:
    return {
        "env_name": "ReferenceModel-2-1",
        "seed": 123,
        "deterministic": deterministic,
        "num_agents": 4,
        "steps_per_episode": 100,
        "sensor_range": 2,
        "info_mode": "full",
        "training_execution_mode": "CTDE",
        "render_env": False,
    }


def _compute_trace_digest(deterministic: bool) -> tuple[str, list[dict]]:
    env = ReferenceModel(_build_env_config(deterministic=deterministic))
    action_rng = np.random.default_rng(999)
    hasher = hashlib.sha256()
    summary = []

    for episode_idx in range(3):
        obs, infos = env.reset()
        hasher.update(f"episode_{episode_idx}_reset".encode())
        for agent_id in sorted(obs):
            hasher.update(agent_id.encode())
            _update_with_array(hasher, obs[agent_id])
        _update_with_value(hasher, infos)

        episode_reward_sum = 0.0
        for step_idx in range(140):
            actions = {f"agent_{i}": int(action_rng.integers(0, 5)) for i in range(4)}
            obs, rewards, terminated, truncated, infos = env.step(actions)
            episode_reward_sum += float(sum(rewards.values()))

            hasher.update(f"episode_{episode_idx}_step_{step_idx}".encode())
            _update_with_value(hasher, actions)
            for agent_id in sorted(obs):
                hasher.update(agent_id.encode())
                _update_with_array(hasher, obs[agent_id])
            _update_with_value(hasher, rewards)
            _update_with_value(hasher, terminated)
            _update_with_value(hasher, truncated)
            _update_with_value(hasher, infos)

            done = terminated.get("__all__", False) or truncated.get("__all__", False)
            if done:
                summary.append(
                    {"episode": episode_idx, "steps": step_idx + 1, "reward_sum": round(episode_reward_sum, 6)}
                )
                break

    return hasher.hexdigest(), summary


def test_trace_parity_stochastic():
    digest, summary = _compute_trace_digest(deterministic=False)
    assert digest == EXPECTED_STOCHASTIC_DIGEST
    assert summary == EXPECTED_STOCHASTIC_SUMMARY


def test_trace_parity_deterministic():
    digest, summary = _compute_trace_digest(deterministic=True)
    assert digest == EXPECTED_DETERMINISTIC_DIGEST
    assert summary == EXPECTED_DETERMINISTIC_SUMMARY
