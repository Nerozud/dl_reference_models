"""Main script to run the training of a chosen environment with chosen algorithm."""

import os
import ray
from ray.tune.registry import register_env
from src.agents.ppo import get_ppo_config
from src.agents.dqn import get_dqn_config
from src.agents.impala import get_impala_config
from src.trainers.tuner import tune_with_callback

ENV_NAME = "ReferenceModel-2-1"
ALGO_NAME = "PPO"  # PPO or IMPALA

env_setup = {
    "env_name": ENV_NAME,
    "num_agents": 4,
    "steps_per_episode": 100,
    "sensor_range": 2,  # 1: 3x3, 2: 5x5, 3: 7x7, not relevant for CTE
    "deterministic": False,  # False: random starts and goals
    "training_execution_mode": "CTDE",  # CTDE or CTE or DTE, if CTE use single agent env
    "render_env": False,
}

# Import the correct environment based on the training execution mode
if env_setup["training_execution_mode"] == "CTE":
    from src.environments.reference_model_single_agent import ReferenceModel
elif (
    env_setup["training_execution_mode"] == "CTDE"
    or env_setup["training_execution_mode"] == "DTE"
):
    from src.environments.reference_model_multi_agent import ReferenceModel
else:
    raise ValueError(
        f"Training execution mode {env_setup['training_execution_mode']} not supported."
    )


def env_creator(env_config=None):
    """Create an environment instance."""
    return ReferenceModel(env_config)


if __name__ == "__main__":

    drive = os.path.splitdrive(os.getcwd())[0]
    ray.init(
        _temp_dir=drive + "\\tmp"
    )  # make sure everything is on the same drive C: or D: etc.

    register_env(ENV_NAME, env_creator)

    if ALGO_NAME == "PPO":
        config = get_ppo_config(ENV_NAME, env_config=env_setup)
    elif ALGO_NAME == "DQN":  # not yet working
        config = get_dqn_config(ENV_NAME, env_config=env_setup)
    elif ALGO_NAME == "IMPALA":
        config = get_impala_config(ENV_NAME, env_config=env_setup)
    else:
        raise ValueError(f"Algorithm {ALGO_NAME} not supported.")

    tune_with_callback(
        config,
        ALGO_NAME,
        ENV_NAME,
    )
