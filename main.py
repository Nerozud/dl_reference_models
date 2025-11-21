"""Main script to run the training of a chosen environment with chosen algorithm."""

import time
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import ray
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from src.agents.dqn import get_dqn_config
from src.agents.impala import get_impala_config
from src.agents.ppo import get_ppo_config
from src.trainers.tuner import tune_with_callback

ENV_NAME = "ReferenceModel-3-1"
ALGO_NAME = "PPO"  # PPO, IMPALA, RANDOM
MODE = "test"  # train or test, test only works with CTDE for now

### Only relevant for MODE = test
TEST_NUM_EPISODES = 50  # number of episodes to run when testing
SAVE_RESULTS = False  # save results to CSV and heatmap
# CHECKPOINT_PATH = r"experiments\trained_models\PPO_2025-10-24_19-27-48\PPO-ReferenceModel-1-1-c2997_00000\checkpoint_000000"  # just for MODE = test
# experiments\trained_models\PPO_2024-11-21_11-17-59\PPO-ReferenceModel-3-1-e280c_00000\checkpoint_000000
CHECKPOINT_PATH = (
    r"experiments\trained_models\IMPALA_2025-05-08_16-18-00\IMPALA-ReferenceModel-3-1-3f831_00000\checkpoint_000124"
)
# experiments\trained_models\IMPALA_2024-12-12_01-13-12\IMPALA-ReferenceModel-3-1-e01c6_00000\checkpoint_000000
CHECKPOINT_RNN = True  # if the checkpoint model has RNN or LSTM layers
CP_TRAINED_ON_ENV_NAME = "ReferenceModel-3-1"  # the environment the model was trained on

SAVE_VIDEO = False  # record evaluation episodes as a GIF when testing
VIDEO_EPISODES_TO_SAVE = 5  # number of episodes to include in the exported video
VIDEO_FRAME_RATE = 5  # frames per second for the generated GIF
VIDEO_FINAL_FRAME_HOLD = 3  # duplicate last frame this many times between episodes
VIDEO_OUTPUT_DIR = Path("experiments/results/videos")

env_setup = {
    "env_name": ENV_NAME,
    "seed": 42,  # int or None, same seed creates same sequence of starts and goals
    "deterministic": False,  # True: given difficult start and goals, False: random starts and goals, depending on seed
    "num_agents": 6,
    "steps_per_episode": 100,
    "sensor_range": 2,  # 1: 3x3, 2: 5x5, 3: 7x7, not relevant for CTE
    "training_execution_mode": "CTDE",  # CTDE or CTE or DTE, if CTE use single agent env
    "render_env": False,
}

# Import the correct environment based on the training execution mode
if env_setup["training_execution_mode"] == "CTE":
    from src.environments.reference_model_single_agent import ReferenceModel
elif env_setup["training_execution_mode"] == "CTDE" or env_setup["training_execution_mode"] == "DTE":
    from src.environments.reference_model_multi_agent import ReferenceModel

else:
    msg = f"Training execution mode {env_setup['training_execution_mode']} not supported."
    raise ValueError(msg)


def env_creator(env_config=None):
    """Create an environment instance."""
    return ReferenceModel(env_config)


def load_checkpoint_local(cp_path: str) -> Algorithm:
    """Build a *fresh* tiny-footprint algo and then restore the weights."""
    # Select the appropriate config based on the algorithm name
    if ALGO_NAME == "PPO":
        from ray.rllib.algorithms.ppo import PPOConfig

        cfg = (
            PPOConfig()
            .environment(CP_TRAINED_ON_ENV_NAME, env_config=env_setup)
            .rollouts(
                num_rollout_workers=0,
                num_envs_per_worker=1,
                create_env_on_local_worker=True,
            )
            .evaluation(evaluation_num_workers=0)
            .resources(num_gpus=1)
            .framework("torch")
        )
    elif ALGO_NAME == "IMPALA":
        from ray.rllib.algorithms.impala import ImpalaConfig

        cfg = (
            ImpalaConfig()
            .environment(CP_TRAINED_ON_ENV_NAME, env_config=env_setup)
            .rollouts(
                num_rollout_workers=0,
                num_envs_per_worker=1,
                create_env_on_local_worker=True,
            )
            .evaluation(evaluation_num_workers=0)
            .resources(num_gpus=1)
            .framework("torch")
        )
    else:
        error_message = f"Algorithm {ALGO_NAME} not supported for checkpoint loading."
        raise ValueError(error_message)

    algo = cfg.build()  # allocates just the driver process
    algo.restore(cp_path)  # ← only weights / optimizer state are loaded
    return algo


def test_trained_model(num_episodes: int):
    """
    Test a trained reinforcement learning model using a given checkpoint path and
    store the results in CSV format.

    Args:
        cp_path (str): Path to the checkpoint file of the trained model.
        num_episodes (int, optional): Number of episodes to run for testing. Defaults to 100.

    Returns:
        None

    Description:
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
        register_env(CP_TRAINED_ON_ENV_NAME, env_creator)
        algo = load_checkpoint_local(CHECKPOINT_PATH)

    record_videos = SAVE_VIDEO and env_setup.get("render_env", False)
    if SAVE_VIDEO and not record_videos:
        msg = "Saving videos requires env_setup['render_env'] to be True."
        raise ValueError(msg)

    env = env_creator(env_config=env_setup)

    total_reward = 0
    total_timesteps = 0

    # Initialize a list to store results for each episode
    results = []

    # Initialize occupancy grid
    grid_height, grid_width = env.grid.shape
    occupancy_grid = np.zeros((grid_height, grid_width), dtype=int)

    recorded_episode_numbers = []
    video_frames = []
    final_frame_hold = max(VIDEO_FINAL_FRAME_HOLD, 0)

    for episode in range(num_episodes):
        start_time = time.process_time()
        obs = env.reset()[0]
        done = {"__all__": False}
        episode_reward = 0
        steps = 0

        should_record_episode = record_videos and len(recorded_episode_numbers) < VIDEO_EPISODES_TO_SAVE
        episode_frames = []
        if env_setup.get("render_env", False):
            if should_record_episode:
                episode_frames.append(env.render(mode="rgb_array"))
            else:
                env.render(mode="human")

        if CHECKPOINT_RNN:
            # Initialize the state for all agents; adjust the ANN size as needed.
            state_list = {agent_id: [torch.zeros(64), torch.zeros(64)] for agent_id in obs}
            initial_state_list = state_list.copy()
            next_state_list = {}

        actions = dict.fromkeys(obs, 0)
        rewards = dict.fromkeys(obs, 0.0)

        # Track per-agent rewards
        agent_episode_rewards = dict.fromkeys(obs, 0.0)
        # Optionally, track trajectories if needed
        # agent_trajectories = {agent_id: [] for agent_id in obs}

        while not done["__all__"]:
            steps += 1
            for agent_id, observation in obs.items():
                if ALGO_NAME == "RANDOM":
                    actions[agent_id] = env.action_space.sample()
                elif CHECKPOINT_RNN:
                    actions[agent_id], next_state_list[agent_id], _ = algo.compute_single_action(
                        observation,
                        state=state_list[agent_id],
                        policy_id="shared_policy",
                        prev_action=actions[agent_id],
                        prev_reward=rewards[agent_id],
                        explore=False,  # for deterministic actions
                    )
                else:
                    actions[agent_id] = algo.compute_single_action(
                        observation,
                        policy_id="shared_policy",
                        explore=False,  # for deterministic actions
                    )

            # Step the environment.
            obs, rewards, done, _, _ = env.step(actions)
            episode_reward += sum(rewards.values())

            if env_setup.get("render_env", False):
                if should_record_episode:
                    episode_frames.append(env.render(mode="rgb_array"))
                else:
                    env.render(mode="human")

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

        if should_record_episode and episode_frames:
            recorded_episode_numbers.append(episode + 1)
            video_frames.extend(episode_frames)
            if final_frame_hold:
                last_frame = episode_frames[-1]
                video_frames.extend([last_frame] * final_frame_hold)

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
        for agent_id in env.get_agent_ids():
            agent_index = agent_id.split("_")[1]  # Extract agent index
            # Flatten the data by creating separate columns for each agent
            episode_data[f"agent_{agent_index}_reward"] = agent_episode_rewards[agent_id]
            start_pos = np.asarray(env.starts[agent_id]).tolist()
            goal_pos = np.asarray(env.goals[agent_id]).tolist()
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

    results_df = pd.DataFrame(results)

    if record_videos and video_frames:
        VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(pytz.utc).strftime("%Y-%m-%d_%H-%M-%S")
        video_path = VIDEO_OUTPUT_DIR / f"{ENV_NAME}_{ALGO_NAME}_{timestamp}.gif"
        imageio.mimsave(video_path, video_frames, fps=VIDEO_FRAME_RATE)
        print(f"Saved evaluation video to {video_path}")

    if SAVE_RESULTS:
        # Ensure the directory exists
        Path("experiments/results").mkdir(parents=True, exist_ok=True)

        # Generate a suitable filename with current date and time

        current_time = datetime.now(pytz.utc).strftime("%Y-%m-%d_%H-%M-%S")

        output_file = (
            Path("experiments/results") / f"{ENV_NAME}_{ALGO_NAME}_{env_setup['num_agents']}_agents_{current_time}.csv"
        )

        # Save the DataFrame to CSV
        results_df.to_csv(output_file, index=False)
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
        heatmap_file = (
            Path("experiments/results")
            / f"{ENV_NAME}_{ALGO_NAME}_{env_setup['num_agents']}_agents_{current_time}_heatmap.pdf"
        )
        plt.savefig(heatmap_file, bbox_inches="tight")
        print(f"Heatmap saved to {heatmap_file}")

        # Optionally show the plot
        # plt.show()


if __name__ == "__main__":
    ray.init()

    register_env(ENV_NAME, env_creator)

    if MODE == "train":
        if ALGO_NAME == "PPO":
            config = get_ppo_config(ENV_NAME, env_config=env_setup)
        elif ALGO_NAME == "DQN":  # not yet working
            config = get_dqn_config(ENV_NAME, env_config=env_setup)
        elif ALGO_NAME == "IMPALA":
            config = get_impala_config(ENV_NAME, env_config=env_setup)
        else:
            msg = f"Algorithm {ALGO_NAME} not supported."
            raise ValueError(msg)

        tune_with_callback(
            config,
            ALGO_NAME,
            ENV_NAME,
        )
    elif MODE == "test":
        test_trained_model(num_episodes=TEST_NUM_EPISODES)
    else:
        msg = f"Mode {MODE} not supported."
        raise ValueError(msg)

    ray.shutdown()
