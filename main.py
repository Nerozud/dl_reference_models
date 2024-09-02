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
    "deterministic": False,
}

if __name__ == "__main__":

    def env_creator(env_config=None):
        """Create an environment instance."""
        return ReferenceModel(env_config)

    register_env(ENV_NAME, env_creator)
    ray.init()

    if ALGO_NAME == "PPO":
        config = get_ppo_config(ENV_NAME, render_env=True, env_config=env_setup)

    tune_with_callback(config, ALGO_NAME, ENV_NAME)
