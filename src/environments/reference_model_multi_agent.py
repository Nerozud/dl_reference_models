"""
Module containing the ReferenceModel class, which is a multi-agent environment.
    Initialize the ReferenceModel environment.
        env_config (dict): Configuration options for the environment.
        None
    Reset the environment to its initial state.
        seed (int): The random seed for the environment.
        options (dict): Additional options for resetting the environment.
        obs (dict): The initial observations for each agent.
        info (dict): Additional information about the reset.
    Take a step in the environment.
        action_dict (dict): The actions to be taken by each agent.
        obs (dict): The new observations for each agent.
        rewards (dict): The rewards obtained by each agent.
        terminated (dict): Whether each agent has terminated.
        truncated (dict): Whether each agent's trajectory was truncated.
        info (dict): Additional information about the step.
        obs (numpy.ndarray): The observation array.
    Render the environment.
        None
"""

import logging

logger = logging.getLogger(__name__)

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from src.environments import get_grid
from src.environments.actions import DOWN, LEFT, NO_OP, RIGHT, UP


class ReferenceModel(MultiAgentEnv):
    """
    Reference Model Multi-Agent Environment.
    This is a simple environment with a grid where agents need to reach their respective goals.
    The environment has the following properties:
    - The grid is a 2D numpy array where each cell can be an empty cell (0) or an obstacle (1).
    - Agents can move in four directions: up, right, down, and left.
    - The observation space is a 3x3 grid centered around the agent,
      where each cell can have one of the following values:
        - 0: Empty cell
        - 1: Obstacle cell
        - 2: Cell occupied by another agent
        - 3: Cell occupied by current agent's goal but not occupied by another agent
        - 4: Cell occupied by another agent's goal but not occupied by another agent
    - The action space consists of the following actions:
        - 0: No-op
        - 1: Move up
        - 2: Move right
        - 3: Move down
        - 4: Move left
    - The environment is episodic and terminates after a fixed number of steps.
    - Each agent receives a reward of
        -1 for being on the same position as another agent
        -1 for not reaching its goal
        +0.5 for reaching its goal for the first time
        1 for reaching its goal
        0 otherwise.
    - The episode terminates after a fixed number of steps or when all agents reach their goals.
    """

    def __init__(self, env_config):
        super().__init__()

        self.step_count = 0
        self.steps_per_episode = env_config.get("steps_per_episode", 100)
        self._num_agents = int(env_config.get("num_agents", 2))
        self.sensor_range = env_config.get("sensor_range", 1)
        self.deterministic = env_config.get("deterministic", False)
        self.normalize_goal_delta = env_config.get("normalize_goal_delta", True)
        self.include_goal_distance = env_config.get("include_goal_distance", False)
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.agents = self.possible_agents.copy()
        self.render_env = env_config.get("render_env", False)
        self.goal_reached_once = dict.fromkeys(self.agents, False)
        self.blocking_penalty = env_config.get("blocking_penalty", -0.2)
        self.move_after_goal_penalty = env_config.get("move_after_goal_penalty", -0.05)
        self._episode_blocking_count = 0.0

        # Initialize a random number generator with the provided seed
        self.seed = env_config.get("seed", None)
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = np.random.default_rng()

        # TODO: Implement the environment initialization depending on the env_config
        # 0 - empty cell, 1 - obstacle,
        # coords are [y-1, x-1] from upper left, so [0, 4] is the aisle
        self.grid = get_grid.get_grid(env_config["env_name"])

        # TODO: Make env name fix and make env as list of envs to give as env_config
        # random_env_name = np.random.choice(
        #     ["ReferenceModel-1-2", "ReferenceModel-1-3", "ReferenceModel-1-4", "ReferenceModel-3-1"]
        # )
        # random_env_name = np.random.choice(
        #     ["ReferenceModel-2-1", "ReferenceModel-3-1"]
        # )
        # self.grid = get_grid.get_grid(random_env_name)

        if self.deterministic:
            self.starts = get_grid.get_start_positions(env_config["env_name"], self._num_agents)
            self.positions = self.starts.copy()
            self.goals = get_grid.get_goal_positions(env_config["env_name"], self._num_agents)
        else:
            self.generate_starts_goals()

        # POMPD, small grid around the agent
        view_side = self.sensor_range * 2 + 1
        self._local_obs_space = gym.spaces.Box(
            low=0,
            high=4,
            shape=(view_side, view_side),
            dtype=np.uint8,
        )
        goal_delta_low = np.array(
            [-(self.grid.shape[0] - 1), -(self.grid.shape[1] - 1)],
            dtype=np.float32,
        )
        goal_delta_high = np.array(
            [self.grid.shape[0] - 1, self.grid.shape[1] - 1],
            dtype=np.float32,
        )
        self._goal_delta_denominator = np.array(
            [max(self.grid.shape[0] - 1, 1), max(self.grid.shape[1] - 1, 1)],
            dtype=np.float32,
        )
        if self.normalize_goal_delta:
            goal_delta_low = goal_delta_low / self._goal_delta_denominator
            goal_delta_high = goal_delta_high / self._goal_delta_denominator
        self._goal_delta_space = gym.spaces.Box(
            low=goal_delta_low,
            high=goal_delta_high,
            shape=(2,),
            dtype=np.float32,
        )
        self._action_mask_space = gym.spaces.MultiBinary(5)

        # Assuming all agents have the same action space
        self._single_act_space = gym.spaces.Discrete(5)

        self._flat_local_obs_len = int(np.prod(self._local_obs_space.shape))
        self._flat_goal_delta_len = int(np.prod(self._goal_delta_space.shape))
        self._flat_goal_distance_len = 1 if self.include_goal_distance else 0
        self._flat_mask_len = int(np.prod(self._action_mask_space.shape))
        total_obs_len = (
            self._flat_local_obs_len + self._flat_goal_delta_len + self._flat_goal_distance_len + self._flat_mask_len
        )
        low = np.concatenate(
            [
                np.zeros(self._flat_local_obs_len, dtype=np.float32),
                self._goal_delta_space.low.astype(np.float32),
                np.zeros(self._flat_goal_distance_len, dtype=np.float32),
                np.zeros(self._flat_mask_len, dtype=np.float32),
            ]
        )
        max_manhattan = float(np.abs(self._goal_delta_space.high).sum())
        high = np.concatenate(
            [
                np.full(self._flat_local_obs_len, self._local_obs_space.high.max(), dtype=np.float32),
                self._goal_delta_space.high.astype(np.float32),
                np.full(self._flat_goal_distance_len, max_manhattan, dtype=np.float32),
                np.ones(self._flat_mask_len, dtype=np.float32),
            ]
        )
        self._single_obs_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._obs_slices = {
            "local_obs": slice(0, self._flat_local_obs_len),
            "goal_delta": slice(
                self._flat_local_obs_len,
                self._flat_local_obs_len + self._flat_goal_delta_len,
            ),
            "goal_distance": slice(
                self._flat_local_obs_len + self._flat_goal_delta_len,
                self._flat_local_obs_len + self._flat_goal_delta_len + self._flat_goal_distance_len,
            ),
            "action_mask": slice(total_obs_len - self._flat_mask_len, total_obs_len),
        }

        self.observation_spaces = dict.fromkeys(self.possible_agents, self._single_obs_space)
        self.action_spaces = dict.fromkeys(self.possible_agents, self._single_act_space)

        self.observation_space = self._single_obs_space
        self.action_space = self._single_act_space

        if self.render_env:
            # Initialize rendering attributes
            self.agent_patches = {}  # Initialize agent_patches as an empty dictionary
            self.goal_patches = {}  # Initialize goal_patches as an empty dictionary
            self.sensor_patches = {}
            self.fig = None
            self.ax = None

    def generate_starts_goals(self):
        """
        Generates random starting positions and goal positions for each agent.

        The function ensures that each agent has a unique starting position and goal position.

        Attributes:
            starts (dict): A dictionary where keys are agent identifiers (e.g., 'agent_0') and values are the starting positions.
            positions (dict): A copy of the starts dictionary representing the current positions of the agents.
            goals (dict): A dictionary where keys are agent identifiers (e.g., 'agent_0') and values are the goal positions.

        """
        self.starts = {}
        available_positions = np.argwhere(self.grid == 0)
        for agent_id in self.agents:
            while True:
                idx = self.rng.choice(len(available_positions))
                # print("idx", idx, "from", len(available_positions))
                start_pos = available_positions[idx]
                # Ensure that the starting position is unique
                if not any(np.array_equal(start_pos, pos) for pos in self.starts.values()):
                    self.starts[agent_id] = start_pos
                    break
        # print("starts", self.starts)
        self.positions = self.starts.copy()
        self.goals = {}
        for agent_id in self.agents:
            while True:
                idx = self.rng.choice(len(available_positions))
                goal_pos = available_positions[idx]
                # Ensure that the goal position is unique and not the same as any start position
                if not any(np.array_equal(goal_pos, pos) for pos in self.goals.values()) and not any(
                    np.array_equal(goal_pos, pos) for pos in self.starts.values()
                ):
                    self.goals[agent_id] = goal_pos
                    break

    def _flatten_observation(self, agent_id: str, local_obs=None, action_mask=None):
        """Pack the structured observation parts into a single flat Box."""
        if local_obs is None:
            local_obs = self.get_obs(agent_id)
        if action_mask is None:
            action_mask = self.get_action_mask(local_obs)

        goal_delta = self._get_goal_delta(agent_id)
        goal_distance = (
            np.asarray([np.abs(goal_delta).sum()], dtype=np.float32)
            if self.include_goal_distance
            else np.array([], dtype=np.float32)
        )

        return np.concatenate(
            [
                local_obs.astype(np.float32).flatten(),
                goal_delta.astype(np.float32),
                goal_distance,
                action_mask.astype(np.float32),
            ]
        )

    def split_flat_observation(self, flat_obs: np.ndarray):
        """
        Helper to recover the structured components from a flat observation.

        Useful for debugging or logging without changing the observation space contract.
        """
        local_flat = flat_obs[self._obs_slices["local_obs"]]
        goal_delta = flat_obs[self._obs_slices["goal_delta"]]
        goal_distance = flat_obs[self._obs_slices["goal_distance"]]
        action_mask = flat_obs[self._obs_slices["action_mask"]]
        return {
            "observations": local_flat.reshape(self._local_obs_space.shape),
            "goal_delta": goal_delta,
            "goal_distance": goal_distance,
            "action_mask": action_mask,
        }

    def _get_goal_delta(self, agent_id: str) -> np.ndarray:
        position = np.asarray(self.positions[agent_id], dtype=np.float32)
        goal = np.asarray(self.goals[agent_id], dtype=np.float32)
        goal_delta = goal - position
        if self.normalize_goal_delta:
            goal_delta = goal_delta / self._goal_delta_denominator
        return goal_delta

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        self._episode_blocking_count = 0.0
        infos = {aid: {} for aid in self.agents}
        obs = {}
        self.goal_reached_once = dict.fromkeys(self.agents, False)

        if self.deterministic:
            self.positions = self.starts.copy()
        else:
            self.generate_starts_goals()

        for agent_id in self.agents:
            local_obs = self.get_obs(agent_id)
            action_mask = self.get_action_mask(local_obs)
            obs[agent_id] = self._flatten_observation(agent_id, local_obs, action_mask)
            infos[agent_id] = {
                "position": np.asarray(self.positions[agent_id]),
                "goal": np.asarray(self.goals[agent_id]),
                "goal_delta": self._get_goal_delta(agent_id),
                "action_mask": action_mask,
                "local_obs": local_obs,
            }
        if self.render_env:
            self.render()

        # for agent_id in self.agents:
        #     print(f"Agent {agent_id}: {infos[agent_id]['goal_delta']}")
        #     print("obs goal delta:", obs[agent_id][self._obs_slices["goal_delta"]])
        return obs, infos

    def step(self, action_dict):
        self.step_count += 1
        rewards = dict.fromkeys(self.agents, 0.0)
        info = {aid: {} for aid in self.agents}
        obs = {}
        terminated = {}
        truncated = {}
        reached_goal = dict.fromkeys(self.agents, False)
        intended_next = {}
        prev_positions = {aid: np.array(self.positions[aid]) for aid in self.agents}
        goal_reached_step_flags = dict.fromkeys(self.agents, 0.0)

        if not action_dict or any(aid not in action_dict for aid in self.agents):
            print("action_dict:", action_dict)
            action_dict = dict.fromkeys(self.agents, 0)
            logger.warning("No actions provided or missing agent actions. Defaulting to no-op actions: %s", action_dict)

        for agent_id in self.agents:
            rewards[agent_id] = 0
            reached_goal[agent_id] = False
            action = action_dict[agent_id]

            pos = self.positions[agent_id]
            next_pos = self.get_next_position(action, pos)
            intended_next[agent_id] = np.array(next_pos)

            # Check if the next position is valid (action mask should prevent invalid moves)
            if (
                0 <= next_pos[0] < self.grid.shape[0]
                and 0 <= next_pos[1] < self.grid.shape[1]
                and self.grid[next_pos[0], next_pos[1]] == 0
                and not any(
                    np.array_equal(self.positions[agent], next_pos) for agent in self.positions if agent != agent_id
                )
            ):
                self.positions[agent_id] = next_pos
            # else:
            # rewards[agent_id] -= 0.1
            # print(
            #     f"Invalid move for agent {agent_id} with action {action} at position {pos}"
            # )

            local_obs = self.get_obs(agent_id)
            action_mask = self.get_action_mask(local_obs)
            obs[agent_id] = self._flatten_observation(agent_id, local_obs, action_mask)
            info[agent_id] = {
                "position": np.asarray(self.positions[agent_id]),
                "goal": np.asarray(self.goals[agent_id]),
                "goal_delta": self._get_goal_delta(agent_id),
                "action_mask": action_mask,
                "local_obs": local_obs,
            }

            if np.array_equal(self.positions[agent_id], self.goals[agent_id]):
                reached_goal[agent_id] = True
                if not self.goal_reached_once[agent_id]:
                    self.goal_reached_once[agent_id] = True
                    rewards[agent_id] += 0.5
                    goal_reached_step_flags[agent_id] = 1.0
                # print(
                #     f"Agent {agent_id} reached its goal, because {self.positions[f'agent_{agent_id}']} == {self.goals[f'agent_{agent_id}']}"
                # )
            # else:
            # print(
            #     f"Agent {agent_id} did not reach its goal, because {self.positions[f'agent_{agent_id}']} != {self.goals[f'agent_{agent_id}']}"
            # )

        blocking_flags = dict.fromkeys(self.agents, 0.0)
        # Intent-based local blocking penalty
        for blocker_id in self.agents:
            if not self.goal_reached_once[blocker_id]:
                continue
            if not np.array_equal(self.positions[blocker_id], prev_positions[blocker_id]):
                continue
            for other_id in self.agents:
                if other_id == blocker_id or self.goal_reached_once[other_id]:
                    continue
                if np.array_equal(intended_next.get(other_id), self.positions[blocker_id]):
                    rewards[blocker_id] += self.blocking_penalty
                    blocking_flags[blocker_id] = 1.0
                    break
        self._episode_blocking_count += float(sum(blocking_flags.values()))

        # Tiny penalty for moving after reaching the goal
        for agent_id in self.agents:
            if not self.goal_reached_once[agent_id]:
                continue
            if not np.array_equal(self.positions[agent_id], prev_positions[agent_id]):
                rewards[agent_id] += self.move_after_goal_penalty

        for agent_id in self.agents:
            info[agent_id]["blocking"] = blocking_flags[agent_id]
            info[agent_id]["goal_reached_step"] = goal_reached_step_flags[agent_id]
        goals_reached_total = float(sum(1 for reached in self.goal_reached_once.values() if reached))
        blocking_count_total = float(self._episode_blocking_count)
        for agent_id in self.agents:
            info[agent_id]["goals_reached_total"] = goals_reached_total
            info[agent_id]["blocking_count_total"] = blocking_count_total
        info["__all__"] = {
            "goals_reached_step": float(sum(goal_reached_step_flags.values())),
            "goals_reached_total": goals_reached_total,
            "blocking_count_step": float(sum(blocking_flags.values())),
            "blocking_count_total": blocking_count_total,
        }

        # Collision penalty (shouldn't happen, but keep it)
        for i, a in enumerate(self.agents):
            for b in self.agents[i + 1 :]:
                if np.array_equal(self.positions[a], self.positions[b]):
                    rewards[a] -= 1
                    rewards[b] -= 1
                    print(f"Agents {a} and {b} are on the same position {self.positions[a]}")

        terminated = dict.fromkeys(self.agents, False)
        truncated = dict.fromkeys(self.agents, False)
        # If all agents have reached their goals, end the episode (not truncated)
        if all(reached_goal.values()):
            for aid in self.agents:
                rewards[aid] += 1
                terminated[aid] = True
            terminated["__all__"] = True
            truncated["__all__"] = False
            # print(
            #     "All agents reached their goals in",
            #     self.step_count,
            #     "steps with a reward of",
            #     rewards,
            # )
            # print("Positions:", self.positions)
            # print("Goals:", self.goals)
        elif (
            self.step_count >= self.steps_per_episode
        ):  # If the step limit is reached, end the episode and mark it as truncated
            # minus reward at truncation for agents not on their goal
            for aid in self.agents:
                if not reached_goal[aid]:
                    rewards[aid] -= 1
                terminated[aid] = True
                truncated[aid] = True
            terminated["__all__"] = True
            truncated["__all__"] = True
        else:
            terminated["__all__"] = False
            truncated["__all__"] = False

        if self.render_env:
            self.render()

        # print("Stepping env with number of obs:", len(obs))
        return obs, rewards, terminated, truncated, info


    def get_next_position(self, action: int, pos):
        """
        Get the next position based on the given action and current position.

        Args:
            action (int): The action to be taken.
            pos (tuple): The current position.

        Returns:
            numpy.ndarray: The next position.

        Description:
            This function calculates the next position based on given action and current position.
            The possible actions are:
                - 0: No-op
                - 1: Move up
                - 2: Move right
                - 3: Move down
                - 4: Move left
            The next position is calculated by adding or subtracting 1 to the corresponding
            coordinate of the current position.

        """
        if action == NO_OP:  # no-op
            next_pos = np.array([pos[0], pos[1]])
        elif action == UP:  # up
            next_pos = np.array([pos[0] - 1, pos[1]])
        elif action == RIGHT:  # right
            next_pos = np.array([pos[0], pos[1] + 1])
        elif action == DOWN:  # down
            next_pos = np.array([pos[0] + 1, pos[1]])
        elif action == LEFT:  # left
            next_pos = np.array([pos[0], pos[1] - 1])
        else:
            msg = "Invalid action"
            raise ValueError(msg)

        return next_pos

    def get_obs(self, agent_id: str):
        """
        Get the observation for a given agent.

        Args:
            agent_id (str): The ID of the agent.

        Returns:
            numpy.ndarray: The observation array.

        Description:
            This function calculates the observation array for a given agent based on its position.
            The observation array is a 2D np array with the same shape as the env's obs space.
            Each element in the array represents the state of a cell in the grid.
            The possible values for each cell are:
                - 0: Empty cell
                - 1: Obstacle cell
                - 2: Cell occupied by another agent
                - 3: Cell occupied by current agent's goal but not occupied by another agent
                - 4: Cell occupied by another agent's goal but not occupied by another agent
            If a cell is outside the grid boundaries, it is considered an obstacle cell.

        """
        pos = self.positions[agent_id]
        obs = np.zeros(
            self._local_obs_space.shape,
            dtype=self._local_obs_space.dtype,
        )

        for i in range(self._local_obs_space.shape[0]):
            for j in range(self._local_obs_space.shape[1]):
                # Calculate the corresponding position on the grid
                x = pos[0] - self.sensor_range + i
                y = pos[1] - self.sensor_range + j

                # Check if the position is within the grid boundaries
                if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]:
                    # Default to empty cell
                    obs[i, j] = 0

                    # Check if the cell is an obstacle
                    if self.grid[x, y] == 1:
                        obs[i, j] = 1
                    elif any(
                        np.array_equal(self.positions[agent], (x, y)) for agent in self.positions if agent != agent_id
                    ):
                        obs[i, j] = 2
                    # Check if this is current agent's goal and not occupied by another agent
                    elif np.array_equal(self.goals[agent_id], (x, y)):
                        obs[i, j] = 3
                    # Check if this is another agent's goal and not occupied by another agent
                    elif any(
                        np.array_equal(self.goals[agent], (x, y)) for agent in self.positions if agent != agent_id
                    ):
                        obs[i, j] = 4
                else:
                    # If the cell is outside the grid, treat it as an obstacle
                    obs[i, j] = 1

        return obs

    def get_action_mask(self, obs):
        """
        Get the action mask for a given agent.

        Args:
            obs (numpy.ndarray): The observation array for the agent.

        Returns:
            numpy.ndarray: The action mask array.

        Description:
            This function calculates the action mask for a given agent based on its observation.
            The action mask is a binary array indicating which actions are valid for the agent.
            The possible actions are:
                - 0: No-op
                - 1: Move up
                - 2: Move right
                - 3: Move down
                - 4: Move left
            The action mask is calculated based on the current observation of the agent.
            Action 0 is always possible.
            Movement actions (1-4) are only possible if the corresponding cell in the observation is empty or a goal.

        """
        action_mask = np.zeros(
            self._action_mask_space.shape,
            dtype=self._action_mask_space.dtype,
        )

        action_mask[0] = 1  # No-op action is always possible

        pos = [
            self.sensor_range,
            self.sensor_range,
        ]
        x, y = pos

        if x > 0 and (obs[x - 1, y] == 0 or obs[x - 1, y] == 3 or obs[x - 1, y] == 4):
            action_mask[1] = 1  # Move up

        if y < obs.shape[1] - 1 and (obs[x, y + 1] == 0 or obs[x, y + 1] == 3 or obs[x, y + 1] == 4):
            action_mask[2] = 1  # Move right

        if x < obs.shape[0] - 1 and (obs[x + 1, y] == 0 or obs[x + 1, y] == 3 or obs[x + 1, y] == 4):
            action_mask[3] = 1  # Move down

        if y > 0 and (obs[x, y - 1] == 0 or obs[x, y - 1] == 3 or obs[x, y - 1] == 4):
            action_mask[4] = 1  # Move left

        return action_mask

    def render(self, mode="human"):
        """
        Render the environment.

        Args:
            mode (str): "human" updates the Matplotlib figure, "rgb_array" returns an RGB frame.

        """
        if mode not in {"human", "rgb_array"}:
            msg = f"Unsupported render mode {mode}. Expected 'human' or 'rgb_array'."
            raise ValueError(msg)

        if not hasattr(self, "fig") or self.fig is None:
            # Initialize the rendering environment if it hasn't been done yet
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(self.grid.shape[1] / 3, self.grid.shape[0] / 3))

            # Draw the grid
            for i in range(self.grid.shape[0]):
                for j in range(self.grid.shape[1]):
                    if self.grid[i, j] == 1:  # Obstacle cell
                        self.ax.add_patch(patches.Rectangle((j, i), 1, 1, color="black"))
                    else:  # Empty cell
                        self.ax.add_patch(patches.Rectangle((j, i), 1, 1, edgecolor="gray", fill=False))

            # Initialize goal patches
            self.goal_patches = {}
            colors = [
                "red",
                "blue",
                "green",
                "purple",
                "orange",
                "cyan",
                "magenta",
                "yellow",
                "brown",
                "pink",
                "olive",
                "teal",
                "navy",
                "gold",
                "lime",
                "gray",
            ]
            for i, agent_id in enumerate(self.agents):
                goal = self.goals[agent_id]
                goal_patch = patches.Polygon(
                    [
                        [goal[1] + 0.5, goal[0]],
                        [goal[1], goal[0] + 0.5],
                        [goal[1] + 0.5, goal[0] + 1],
                        [goal[1] + 1, goal[0] + 0.5],
                    ],
                    color=colors[i % len(colors)],
                    alpha=0.5,
                    label=f"Goal {i}",
                )
                self.ax.add_patch(goal_patch)
                self.goal_patches[agent_id] = goal_patch

            # Initialize agent patches
            self.agent_patches = {}
            for i, agent_id in enumerate(self.agents):
                pos = self.positions[agent_id]
                agent_patch = patches.Circle(
                    (pos[1] + 0.5, pos[0] + 0.5),
                    0.3,
                    color=colors[i % len(colors)],
                    label=f"Agent {i}",
                )
                sensor_range_size = 2 * self.sensor_range + 1
                sensor_patch = patches.Rectangle(
                    (
                        pos[1] - self.sensor_range,
                        pos[0] - self.sensor_range,
                    ),
                    sensor_range_size,
                    sensor_range_size,
                    edgecolor=colors[i % len(colors)],
                    facecolor=colors[i % len(colors)],
                    alpha=0.2,
                    fill=True,
                )
                self.ax.add_patch(agent_patch)
                self.ax.add_patch(sensor_patch)
                self.agent_patches[agent_id] = agent_patch
                self.sensor_patches[agent_id] = sensor_patch

            # Set the limits and aspect
            self.ax.set_xlim(0, self.grid.shape[1])
            self.ax.set_ylim(0, self.grid.shape[0])
            self.ax.set_aspect("equal")

            # Add grid lines for clarity
            self.ax.set_xticks(np.arange(0, self.grid.shape[1], 1))
            self.ax.set_yticks(np.arange(0, self.grid.shape[0], 1))
            self.ax.grid(visible=True, which="both", color="gray", linestyle="-", linewidth=0.5)

            # Reverse the y-axis to match typical grid layout (0,0 at top-left)
            self.ax.invert_yaxis()

        else:
            # Update agent positions
            for agent_id, agent_patch in self.agent_patches.items():
                pos = self.positions[agent_id]
                agent_patch.set_center((pos[1] + 0.5, pos[0] + 0.5))
                sensor_patch = self.sensor_patches[agent_id]
                sensor_patch.set_xy((pos[1] - self.sensor_range, pos[0] - self.sensor_range))

            # Update goal positions
            for agent_id, goal_patch in self.goal_patches.items():
                goal = self.goals[agent_id]
                goal_vertices = [
                    [goal[1] + 0.5, goal[0]],
                    [goal[1], goal[0] + 0.5],
                    [goal[1] + 0.5, goal[0] + 1],
                    [goal[1] + 1, goal[0] + 0.5],
                ]
                goal_patch.set_xy(goal_vertices)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if mode == "human":
            return True

        width, height = self.fig.canvas.get_width_height()

        # Prefer RGBA buffer if available (backend-dependent).
        if hasattr(self.fig.canvas, "buffer_rgba"):
            frame = np.asarray(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape((height, width, 4))
            return frame[:, :, :3].copy()

        # Fallback for backends that only expose ARGB.
        frame = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
        frame = frame.reshape((height, width, 4))
        return frame[:, :, 1:4].copy()

    def close(self):
        plt.close()
