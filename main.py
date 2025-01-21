"""Main script to run the training of a chosen environment with chosen algorithm."""

import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from src.agents.ppo import get_ppo_config
from src.agents.dqn import get_dqn_config
from src.agents.impala import get_impala_config
from src.trainers.tuner import tune_with_callback

ENV_NAME = "ReferenceModel-3-1"
ALGO_NAME = "IMPALA"  # PPO, IMPALA, RANDOM
MODE = "train"  # train or test, test only works with CTDE for now
CHECKPOINT_PATH = r"experiments\trained_models\PPO_2024-11-21_11-17-59\PPO-ReferenceModel-3-1-e280c_00000\checkpoint_000000"  # just for MODE = test
# experiments\trained_models\IMPALA_2024-12-12_01-13-12\IMPALA-ReferenceModel-3-1-e01c6_00000\checkpoint_000000
# experiments\trained_models\IMPALA_2024-10-31_20-25-09\IMPALA-ReferenceModel-2-1-b-d7c2f_00000\checkpoint_000000
CHECKPOINT_RNN = True  # if the checkpoint model has RNN or LSTM layers

env_setup = {
    "env_name": ENV_NAME,
    "seed": None,  # int or None, same seed creates same sequence of starts and goals
    "deterministic": False,  # True: given difficult start and goals, False: random starts and goals, depending on seed
    "num_agents": 10,
    "steps_per_episode": 100,
    "sensor_range": 2,  # 1: 3x3, 2: 5x5, 3: 7x7, not relevant for CTE
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


def test_trained_model(cp_path, num_episodes=100):
    """
    Test a trained reinforcement learning model using a given checkpoint path and
    store the results in CSV format.

    Args:
        cp_path (str): Path to the checkpoint file of the trained model.
        num_episodes (int, optional): Number of episodes to run for testing. Defaults to 100.
    Returns:
        None
    The function performs the following steps:
    1. Initializes the RLlib Algorithm from the provided checkpoint.
    2. Creates the environment using the `env_creator` function.
    3. Runs the specified number of episodes, collecting rewards and other metrics.
    4. If using RNN checkpoints, manages the state for each agent.
    5. Records the total reward, timesteps, and CPU time for each episode.
    6. Collects per-agent rewards and positions.
    7. Stores the results in a pandas DataFrame and saves it as CSV in "experiments/results".
    8. Prints average reward and timesteps across all episodes.
    """
    # Initialize the RLlib Algorithm from a checkpoint.
    if ALGO_NAME != "RANDOM":
        algo = Algorithm.from_checkpoint(cp_path)
    env = env_creator(env_config=env_setup)

    total_reward = 0
    total_timesteps = 0

    # Initialize a list to store results for each episode
    results = []

    # Initialize occupancy grid
    grid_height, grid_width = env.grid.shape
    occupancy_grid = np.zeros((grid_height, grid_width), dtype=int)

    for episode in range(num_episodes):
        start_time = time.process_time()
        obs = env.reset()[0]
        done = {"__all__": False}
        episode_reward = 0
        steps = 0

        if CHECKPOINT_RNN:
            # Initialize the state for all agents; adjust the ANN size as needed.
            state_list = {
                agent_id: [torch.zeros(64), torch.zeros(64)] for agent_id in obs
            }
            initial_state_list = state_list.copy()
            next_state_list = {}

        actions = {agent_id: 0 for agent_id in obs}
        rewards = {agent_id: 0.0 for agent_id in obs}

        # Track per-agent rewards
        agent_episode_rewards = {agent_id: 0.0 for agent_id in obs}
        # Optionally, track trajectories if needed
        # agent_trajectories = {agent_id: [] for agent_id in obs}

        while not done["__all__"]:
            steps += 1
            for agent_id, observation in obs.items():
                if ALGO_NAME == "RANDOM":
                    actions[agent_id] = env.action_space.sample()
                elif CHECKPOINT_RNN:
                    actions[agent_id], next_state_list[agent_id], _ = (
                        algo.compute_single_action(
                            observation,
                            state=state_list[agent_id],
                            policy_id="shared_policy",
                            prev_action=actions[agent_id],
                            prev_reward=rewards[agent_id],
                            explore=True,
                        )
                    )
                else:
                    actions[agent_id] = algo.compute_single_action(
                        observation,
                        policy_id="shared_policy",
                        explore=True,
                    )

            # Step the environment.
            obs, rewards, done, _, _ = env.step(actions)
            episode_reward += sum(rewards.values())

            # Update per-agent rewards and record positions
            for agent_id, agent_obs in obs.items():
                agent_episode_rewards[agent_id] += rewards[agent_id]

                # agent_obs["position"] gives [y, x]
                pos_y, pos_x = agent_obs["position"]
                # Update the occupancy grid
                if 0 <= pos_y < grid_height and 0 <= pos_x < grid_width:
                    occupancy_grid[pos_y, pos_x] += 1

            # Update states for all agents.
            if CHECKPOINT_RNN:
                state_list = next_state_list.copy()

        if CHECKPOINT_RNN:
            state_list = initial_state_list.copy()

        print(f"Episode {episode + 1} reward: {episode_reward} timesteps: {steps}")
        total_reward += episode_reward
        total_timesteps += steps

        # Record the end time
        end_time = time.process_time()

        # Calculate the CPU time
        cpu_time = end_time - start_time

        # Collect episode data in a flat structure
        episode_data = {
            "episode": episode + 1,
            "cpu_time": cpu_time,
            "seed": env.seed,
            "total_reward": episode_reward,
            "timesteps": steps,
        }

        # Add per-agent rewards and positions
        for agent_id in env._agent_ids:
            agent_index = agent_id.split("_")[1]  # Extract agent index
            # Flatten the data by creating separate columns for each agent
            episode_data[f"agent_{agent_index}_reward"] = agent_episode_rewards[
                agent_id
            ]
            start_pos = env.starts[agent_id].tolist()
            goal_pos = env.goals[agent_id].tolist()
            episode_data[f"agent_{agent_index}_start_x"] = start_pos[0]
            episode_data[f"agent_{agent_index}_start_y"] = start_pos[1]
            episode_data[f"agent_{agent_index}_goal_x"] = goal_pos[0]
            episode_data[f"agent_{agent_index}_goal_y"] = goal_pos[1]
            # Optionally, include trajectories (but be cautious with CSV size)
            # trajectory = agent_trajectories[agent_id]
            # episode_data[f'agent_{agent_index}_trajectory'] = str(trajectory)

        results.append(episode_data)

    # Calculate and print average reward and timesteps
    print("Average reward:", total_reward / num_episodes)
    print("Average timesteps:", total_timesteps / num_episodes)

    # Calculate and print success rate
    successful_episodes = [
        result
        for result in results
        if result["timesteps"] <= env_setup["steps_per_episode"]
        and result["total_reward"] == 1.5 * env_setup["num_agents"]
    ]
    success_rate = len(successful_episodes) / num_episodes
    print("Success rate:", success_rate * 100, "%")

    df = pd.DataFrame(results)
    # Ensure the directory exists
    os.makedirs("experiments/results", exist_ok=True)

    # Generate a suitable filename with current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(
        "experiments/results",
        f"{ENV_NAME}_{ALGO_NAME}_{env_setup['num_agents']}_agents_{current_time}.csv",
    )

    # Save the DataFrame to CSV
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Create a heatmap of occupancy
    plt.figure()
    ax = plt.gca()
    plt.xlabel("X")
    plt.ylabel("Y")

    # Note: origin='upper' matches the indexing [y,x] with y=0 at top
    heatmap_plot = plt.imshow(occupancy_grid, origin="upper")

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(heatmap_plot, label="Number of visits", cax=cax)

    # Save the heatmap
    heatmap_file = os.path.join(
        "experiments/results",
        f"{ENV_NAME}_{ALGO_NAME}_{env_setup['num_agents']}_agents_{current_time}_heatmap.pdf",
    )
    plt.savefig(heatmap_file, bbox_inches="tight")
    print(f"Heatmap saved to {heatmap_file}")

    # Optionally show the plot
    # plt.show()


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
