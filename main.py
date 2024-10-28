"""Main script to run the training of a chosen environment with chosen algorithm."""

import os
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from src.agents.ppo import get_ppo_config
from src.agents.dqn import get_dqn_config
from src.agents.impala import get_impala_config
from src.trainers.tuner import tune_with_callback

ENV_NAME = "ReferenceModel-2-1"
ALGO_NAME = "PPO"  # PPO or IMPALA
MODE = "test"  # train or test an algorithm
CHECKPOINT_PATH = r"experiments\trained_models\PPO_2024-10-17_01-50-02\PPO-ReferenceModel-2-1-5c8c0_00000\checkpoint_000000"  # just for MODE = test

env_setup = {
    "env_name": ENV_NAME,
    "num_agents": 4,
    "steps_per_episode": 100,
    "sensor_range": 2,  # 1: 3x3, 2: 5x5, 3: 7x7, not relevant for CTE
    "deterministic": False,  # False: random starts and goals
    "training_execution_mode": "CTDE",  # CTDE or CTE or DTE, if CTE use single agent env
    "render_env": True,
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


def test_trained_model(cp_path, num_episodes=10):
    """
    Test a trained model with a given checkpoint path.
    Currently not working with trained RNNs.
    """

    # Initialize the RLlib Algorithm from a checkpoint.
    algo = Algorithm.from_checkpoint(cp_path)
    env = env_creator(env_config=env_setup)

    for episode in range(num_episodes):
        obs = env.reset()[0]
        done = {"__all__": False}
        episode_reward = 0
        steps = 0

        # state_list = {
        #     agent_id: [torch.zeros(128), torch.zeros(128)] for agent_id in obs
        # }
        # initial_state_list = state_list

        actions = {agent_id: 0 for agent_id in obs}
        rewards = {agent_id: 0.0 for agent_id in obs}

        # next_state_list = {}

        while not done["__all__"]:
            steps += 1
            for j in range(env_setup["num_agents"]):
                actions[f"agent_{j}"] = algo.compute_single_action(
                    obs[f"agent_{j}"],
                    policy_id="shared_policy",
                    explore=True,
                )
                # for future testing with RNNs
                # actions[f"agent_{j}"], next_state_list[f"agent_{j}"], _ = (
                # algo.compute_single_action(
                #     obs[f"agent_{j}"],
                #     # state=state_list[f"agent_{j}"],
                #     policy_id="shared_policy",
                #     # prev_action=actions[f"agent_{j}"],
                #     # prev_reward=rewards[f"agent_{j}"],
                #     explore=True,
                # )
                # )  # for PG algorithms true
            # Step the environment.
            obs, rewards, done, truncated, info = env.step(actions)
            episode_reward += sum(rewards.values())

            # Update states for all agents.
        #     state_list = next_state_list

        # state_list = initial_state_list
        print(f"Episode {episode + 1} reward: {episode_reward} timesteps: {steps}")


if __name__ == "__main__":

    drive = os.path.splitdrive(os.getcwd())[0]
    ray.init(
        _temp_dir=drive + "\\tmp"
    )  # make sure everything is on the same drive C: or D: etc.

    register_env(ENV_NAME, env_creator)

    if MODE == "train":
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
    elif MODE == "test":
        test_trained_model(CHECKPOINT_PATH)
    else:
        raise ValueError(f"Mode {MODE} not supported.")
