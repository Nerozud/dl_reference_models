"""
This module contains the ReferenceModel class, which is a multi-agent environment.
The environment is interpreted as single RL agent environment, 
where all agents are controlled by the same policy.

Initialize the ReferenceModel environment.
    Parameters:
        env_config (dict): Configuration options for the environment.

Reset the environment to its initial state.
    Parameters:
        seed (int): The random seed for the environment.
        options (dict): Additional options for resetting the environment.
    Returns:
        obs (dict): The initial observations for each agent.
        info (dict): Additional information about the reset.

Take a step in the environment.
    Parameters:
        action_dict (dict): The actions to be taken by each agent.
    Returns:
        obs (dict): The new observations for each agent.
        rewards (dict): The rewards obtained by each agent.
        terminated (dict): Whether each agent has terminated.
        truncated (dict): Whether each agent's trajectory was truncated.
        info (dict): Additional information about the step.

Render the environment.
    None
"""

import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import gymnasium as gym

from src.environments import get_grid


class ReferenceModel(gym.Env):
    """
    Reference Model 1.1 for CTE
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
        self.num_agents = env_config.get("num_agents", 2)
        self.sensor_range = env_config.get("sensor_range", 1)
        self.deterministic = env_config.get("deterministic", False)
        self._agent_ids = {f"agent_{i}" for i in range(self.num_agents)}
        self.render_env = env_config.get("render_env", False)
        self.goal_reached_once = {f"agent_{i}": False for i in range(self.num_agents)}

        # TODO: Implement the environment initialization depending on the env_config´
        # 0 - empty cell, 1 - obstacle,
        # coords are [y-1, x-1] from upper left, so [0, 4] is the aisle
        self.grid = get_grid.get_grid(env_config["env_name"])

        if self.deterministic:
            self.starts = get_grid.get_start_positions(
                env_config["env_name"], self.num_agents
            )
            self.positions = self.starts.copy()
            self.goals = get_grid.get_goal_positions(
                env_config["env_name"], self.num_agents
            )
        else:
            self.generate_starts_goals()

        # Assuming all agents have the same observation space
        self.observation_space = gym.spaces.Dict(
            {
                "observations": gym.spaces.Box(
                    low=0,
                    high=2 * self.num_agents + 1,
                    shape=(self.grid.shape[0], self.grid.shape[1]),
                    dtype=np.uint8,
                ),
                "action_mask": gym.spaces.MultiBinary(5 * self.num_agents),
            }
        )

        # Assuming all agents have the same action space
        self.action_space = gym.spaces.MultiDiscrete([5] * self.num_agents)

        if self.render_env:
            # Initialize rendering attributes
            self.agent_patches = {}  # Initialize agent_patches as an empty dictionary
            self.goal_patches = {}  # Initialize goal_patches as an empty dictionary
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
        for i in range(self.num_agents):
            while True:
                start_pos = random.choice(np.argwhere(self.grid == 0))
                # Ensure that the starting position is unique
                if not any(
                    np.array_equal(start_pos, pos) for pos in self.starts.values()
                ):
                    self.starts[f"agent_{i}"] = start_pos
                    break
        self.positions = self.starts.copy()
        self.goals = {}
        for i in range(self.num_agents):
            while True:
                goal_pos = random.choice(np.argwhere(self.grid == 0))
                # Ensure that the goal position is unique
                if not any(
                    np.array_equal(goal_pos, pos) for pos in self.goals.values()
                ):
                    self.goals[f"agent_{i}"] = goal_pos
                    break

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        info = {}
        obs = {}
        self.goal_reached_once = {f"agent_{i}": False for i in range(self.num_agents)}

        if self.deterministic:
            self.positions = self.starts.copy()
        else:
            self.generate_starts_goals()

        obs = {}
        obs["observations"] = self.get_obs()
        obs["action_mask"] = self.get_action_mask(obs["observations"])
        if self.render_env:
            self.render()

        return obs, info

    def step(self, action):
        self.step_count += 1
        info = {}
        obs = {}
        terminated = {}
        truncated = {}
        reached_goal = {}
        reward = 0

        for i in range(self.num_agents):
            reached_goal[f"agent_{i}"] = False
            pos = self.positions[f"agent_{i}"]
            next_pos = self.get_next_position(action[i], pos)

            # Check if the next position is valid (action mask should prevent invalid moves)
            if (
                0 <= next_pos[0] < self.grid.shape[0]
                and 0 <= next_pos[1] < self.grid.shape[1]
                and self.grid[next_pos[0], next_pos[1]] == 0
                and not any(
                    np.array_equal(self.positions[agent], next_pos)
                    for agent in self.positions
                    if agent != f"agent_{i}"
                )
            ):
                self.positions[f"agent_{i}"] = next_pos
            # else:
            # rewards[f"agent_{i}"] -= 0.1
            # print(
            #     f"Invalid move for agent {i} with action {action} at position {pos}"
            # )

            if np.array_equal(self.positions[f"agent_{i}"], self.goals[f"agent_{i}"]):
                reached_goal[f"agent_{i}"] = True
                if not self.goal_reached_once[f"agent_{i}"]:
                    self.goal_reached_once[f"agent_{i}"] = True
                    reward += 0.5

        obs = {}
        obs["observations"] = self.get_obs()
        obs["action_mask"] = self.get_action_mask(obs["observations"])

        # minus reward for agents on the same position, shouldn't happen
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if np.array_equal(
                    self.positions[f"agent_{i}"], self.positions[f"agent_{j}"]
                ):
                    reward -= 1
                    print(
                        f"Agents {i} and {j} are on the same position {self.positions[f'agent_{i}']}"
                    )

        # If all agents have reached their goals, end the episode (not truncated)
        if all(reached_goal.values()):
            for i in range(self.num_agents):
                reward += 1
            terminated = True
            truncated = False
            # print(
            #     "All agents reached their goals in",
            #     self.step_count,
            #     "steps with a reward of",
            #     reward,
            # )
            # print("Positions:", self.positions)
            # print("Goals:", self.goals)
        elif (
            self.step_count >= self.steps_per_episode
        ):  # If the step limit is reached, end the episode and mark it as truncated
            # minus reward for not reaching the goal
            for i in range(self.num_agents):
                if not np.array_equal(
                    self.positions[f"agent_{i}"], self.goals[f"agent_{i}"]
                ):
                    reward -= 1
            terminated = True
            truncated = True
        else:
            terminated = False
            truncated = False

        if self.render_env:
            self.render()

        # print("Stepping env with number of obs:", len(obs))
        return obs, reward, terminated, truncated, info

    def get_next_position(self, action, pos):
        """
        Get the next position based on the given action and current position.
        Parameters:
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
        if action == 0:  # no-op
            next_pos = np.array([pos[0], pos[1]], dtype=np.uint8)
        elif action == 1:  # up
            next_pos = np.array([pos[0] - 1, pos[1]], dtype=np.uint8)
        elif action == 2:  # right
            next_pos = np.array([pos[0], pos[1] + 1], dtype=np.uint8)
        elif action == 3:  # down
            next_pos = np.array([pos[0] + 1, pos[1]], dtype=np.uint8)
        elif action == 4:  # left
            next_pos = np.array([pos[0], pos[1] - 1], dtype=np.uint8)
        else:
            raise ValueError("Invalid action")

        return next_pos

    def get_obs(self):
        """
        Get the observation for a given agent.
        Returns:
            numpy.ndarray: The observation array.
        Description:
            This function calculates the complete grid with agent positions and goals.
            The observation array is a 2D np array with the same shape as the env's obs space.
            Each element in the array represents the state of a cell in the grid.
            The possible values for each cell are:
                - 0: Empty cell
                - 1: Obstacle cell
                - 2: agent 0 position
                - 3: agent 0 goal
                - 4: agent 1 position
                - 5: agent 1 goal
                - ...
        """

        obs = self.grid.copy()

        # can be optimized by not creating the obs array from scratch
        # goals first, then agents overwriting goals
        for i in range(self.num_agents):
            agent_id = f"agent_{i}"
            goal = self.goals[agent_id]
            obs[goal[0], goal[1]] = i * 2 + 3

        for i in range(self.num_agents):
            agent_id = f"agent_{i}"
            pos = self.positions[agent_id]
            obs[pos[0], pos[1]] = i * 2 + 2

        return obs

    def get_action_mask(self, obs):
        """
        Get the action mask for a given agent.
        Parameters:
            agent_id (str): The ID of the agent.
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
            Movement actions (1-4) are only possible if the corresponding cell is empty or a goal.
        """

        action_mask = np.zeros(
            self.observation_space["action_mask"].shape,
            dtype=self.observation_space["action_mask"].dtype,
        )

        for i in range(self.num_agents):
            action_mask[i * 5] = 1  # No-op action is always possible

            pos = [
                self.positions[f"agent_{i}"][0],
                self.positions[f"agent_{i}"][1],
            ]
            x, y = pos

            if x > 0 and (obs[x - 1, y] == 0 or obs[x - 1, y] % 2 == 1):
                action_mask[i * 5 + 1] = 1  # Move up

            if y < obs.shape[1] - 1 and (obs[x, y + 1] == 0 or obs[x, y + 1] % 2 == 1):
                action_mask[i * 5 + 2] = 1  # Move right

            if x < obs.shape[0] - 1 and (obs[x + 1, y] == 0 or obs[x + 1, y] % 2 == 1):
                action_mask[i * 5 + 3] = 1  # Move down

            if y > 0 and (obs[x, y - 1] == 0 or obs[x, y - 1] % 2 == 1):
                action_mask[i * 5 + 4] = 1  # Move left

        return action_mask

    def render(self):
        """Render the environment."""
        if not hasattr(self, "fig") or self.fig is None:
            # Initialize the rendering environment if it hasn't been done yet
            plt.ion()
            self.fig, self.ax = plt.subplots(
                figsize=(self.grid.shape[1] / 3, self.grid.shape[0] / 3)
            )

            # Draw the grid
            for i in range(self.grid.shape[0]):
                for j in range(self.grid.shape[1]):
                    if self.grid[i, j] == 1:  # Obstacle cell
                        self.ax.add_patch(
                            patches.Rectangle((j, i), 1, 1, color="black")
                        )
                    else:  # Empty cell
                        self.ax.add_patch(
                            patches.Rectangle(
                                (j, i), 1, 1, edgecolor="gray", fill=False
                            )
                        )

            # Initialize goal patches
            self.goal_patches = {}
            colors = [
                "red",
                "blue",
                "green",
                "purple",
                "orange",
            ]  # Add more colors if needed
            for i in range(self.num_agents):
                agent_id = f"agent_{i}"
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
            for i in range(self.num_agents):
                agent_id = f"agent_{i}"
                pos = self.positions[agent_id]
                agent_patch = patches.Circle(
                    (pos[1] + 0.5, pos[0] + 0.5),
                    0.3,
                    color=colors[i % len(colors)],
                    label=f"Agent {i}",
                )

                self.ax.add_patch(agent_patch)
                self.agent_patches[agent_id] = agent_patch

            # Set the limits and aspect
            self.ax.set_xlim(0, self.grid.shape[1])
            self.ax.set_ylim(0, self.grid.shape[0])
            self.ax.set_aspect("equal")

            # Add grid lines for clarity
            self.ax.set_xticks(np.arange(0, self.grid.shape[1], 1))
            self.ax.set_yticks(np.arange(0, self.grid.shape[0], 1))
            self.ax.grid(True, which="both", color="gray", linestyle="-", linewidth=0.5)

            # Reverse the y-axis to match typical grid layout (0,0 at top-left)
            self.ax.invert_yaxis()

        else:
            # Update agent positions
            for agent_id, agent_patch in self.agent_patches.items():
                pos = self.positions[agent_id]
                agent_patch.set_center((pos[1] + 0.5, pos[0] + 0.5))

        # Redraw the updated plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        return True

    def close(self):
        plt.close()
