import ray
from ray.tune.registry import register_env

from src.agents.ppo import get_ppo_config
from src.trainers.tuner import tune_with_callback
from src.environments.reference_model_1_1 import ReferenceModel

ENV_NAME = "ReferenceModel-1-1"
ALGO_NAME = "PPO"

if __name__ == "__main__":

    def env_creator(env_config):
        """Create an environment instance."""
        return ReferenceModel(env_config)

    register_env(ENV_NAME, env_creator)
    ray.init()

    if ALGO_NAME == "PPO":
        config = get_ppo_config()

    tune_with_callback(config, ALGO_NAME)
