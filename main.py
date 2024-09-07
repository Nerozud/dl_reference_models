"""Main script to run the training of a chosen environment with chosen algorithm."""

import ray
from ray.tune.registry import register_env
from src.agents.ppo import get_ppo_config
from src.trainers.tuner import tune_with_callback
from src.environments.reference_model_1_1 import ReferenceModel

ENV_NAME = "ReferenceModel-1-1"
ALGO_NAME = "PPO"

env_setup = {
    "num_agents": 2,
    "deterministic": True,  # no randomness in the environment
    "training_execution_mode": "CTDE",  # CTDE or CTE or DTE
    "render_env": False,
}


def env_creator(env_config=None):
    """Create an environment instance."""
    return ReferenceModel(env_config)


if __name__ == "__main__":

    ray.init(
        _temp_dir="D:\\tmp"
    )  # make sure everything is on the same drive C: or D: etc.

    register_env(ENV_NAME, env_creator)

    if ALGO_NAME == "PPO":
        config = get_ppo_config(ENV_NAME, env_config=env_setup)

    tune_with_callback(config, ALGO_NAME, ENV_NAME)
