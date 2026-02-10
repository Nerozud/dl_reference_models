"""Performance regression test for multi-agent environment throughput."""

from __future__ import annotations

import os
import time

import numpy as np

from src.environments.reference_model_multi_agent import ReferenceModel


def test_env_only_steps_per_second_target():
    env = ReferenceModel(
        {
            "env_name": "ReferenceModel-2-1",
            "seed": 123,
            "deterministic": False,
            "num_agents": 4,
            "steps_per_episode": 100,
            "sensor_range": 2,
            "info_mode": "lite",
            "training_execution_mode": "CTDE",
            "render_env": False,
        }
    )
    rng = np.random.default_rng(999)
    obs, _ = env.reset()

    warmup_steps = 2000
    measured_steps = 12000
    threshold = float(os.getenv("REFERENCE_MODEL_MIN_STEPS_PER_SEC", "5300"))

    for _ in range(warmup_steps):
        actions = {agent_id: int(rng.integers(0, 5)) for agent_id in env.agents}
        obs, _rewards, terminated, truncated, _info = env.step(actions)
        if terminated.get("__all__", False) or truncated.get("__all__", False):
            obs, _ = env.reset()

    t0 = time.perf_counter()
    for _ in range(measured_steps):
        actions = {agent_id: int(rng.integers(0, 5)) for agent_id in env.agents}
        obs, _rewards, terminated, truncated, _info = env.step(actions)
        if terminated.get("__all__", False) or truncated.get("__all__", False):
            obs, _ = env.reset()
    elapsed = time.perf_counter() - t0

    steps_per_second = measured_steps / elapsed
    assert steps_per_second >= threshold, (
        f"Measured {steps_per_second:.2f} steps/s, expected at least {threshold:.2f} steps/s "
        "for num_agents=4, sensor_range=2."
    )
