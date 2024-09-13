"""
This module contains the ReferenceModel class, which is a multi-agent environment.
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

import random
import logging

from ray.rllib.env.multi_agent_env import MultiAgentEnv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import gymnasium as gym


class ReferenceModel(MultiAgentEnv):
    """
    Reference Model 1.1
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
        -0.1 for each invalid move
        -1 for being on the same position as another agent
        -1 for not reaching its goal
        1 for reaching its goal
        0 otherwise.
    - The episode terminates after a fixed number of steps or when all agents reach their goals.
    """

    def __init__(self, env_config):
        super().__init__()

        self.step_count = 0
        self.steps_per_episode = 100
        self.num_agents = env_config.get("num_agents", 2)
        self.sensor_range = env_config.get("sensor_range", 1)
        self.deterministic = env_config.get("deterministic", False)
        self._agent_ids = {f"agent_{i}" for i in range(self.num_agents)}
        self.render_env = env_config.get("render_env", False)
        self.goal_reached_once = {f"agent_{i}": False for i in range(self.num_agents)}

        # TODO: Implement the environment initialization depending on the env_config´
        # 0 - empty cell, 1 - obstacle,
        # coords are [y-1, x-1] from upper left, so [0, 4] is the aisle
        self.grid = np.array(
            [
                [1, 1, 1, 1, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        if self.deterministic:
            self.starts = {"agent_0": (1, 1), "agent_1": (1, 7)}
            self.positions = self.starts.copy()
            self.goals = {"agent_0": (1, 8), "agent_1": (1, 0)}
        else:
            self.starts = {
                f"agent_{i}": (random.choice(np.argwhere(self.grid == 0)))
                for i in range(self.num_agents)
            }
            self.positions = self.starts.copy()
            self.goals = {
                f"agent_{i}": (random.choice(np.argwhere(self.grid == 0)))
                for i in range(self.num_agents)
            }

        # POMPD, small grid around the agent

        # Assuming all agents have the same observation space
        self.observation_space = gym.spaces.Dict(
            {
                "observations": gym.spaces.Box(
                    low=0,
                    high=4,
                    shape=(self.sensor_range * 2 + 1, self.sensor_range * 2 + 1),
                    dtype=np.uint8,
                ),
                "position": gym.spaces.Box(
                    low=0,
                    high=max(self.grid.shape),
                    shape=(2,),
                    dtype=np.uint8,
                ),
                "goal": gym.spaces.Box(
                    low=0,
                    high=max(self.grid.shape),
                    shape=(2,),
                    dtype=np.uint8,
                ),
                "action_mask": gym.spaces.MultiBinary(5),
            }
        )

        # Assuming all agents have the same action space
        self.action_space = gym.spaces.Discrete(5)

        if self.render_env:
            # Initialize rendering attributes
            self.agent_patches = {}  # Initialize agent_patches as an empty dictionary
            self.goal_patches = {}  # Initialize goal_patches as an empty dictionary
            self.sensor_patches = {}
            self.fig = None
            self.ax = None

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        info = {}
        obs = {}
        self.goal_reached_once = {f"agent_{i}": False for i in range(self.num_agents)}

        if self.deterministic:
            self.positions = self.starts.copy()
        else:
            self.starts = {
                f"agent_{i}": (random.choice(np.argwhere(self.grid == 0)))
                for i in range(self.num_agents)
            }
            self.positions = self.starts.copy()
            self.goals = {
                f"agent_{i}": (random.choice(np.argwhere(self.grid == 0)))
                for i in range(self.num_agents)
            }

        for i in range(self.num_agents):
            obs[f"agent_{i}"] = {}
            obs[f"agent_{i}"]["position"] = np.array(self.positions[f"agent_{i}"])
            obs[f"agent_{i}"]["goal"] = np.array(self.goals[f"agent_{i}"])
            obs[f"agent_{i}"]["observations"] = self.get_obs(f"agent_{i}")
            obs[f"agent_{i}"]["action_mask"] = self.get_action_mask(
                obs[f"agent_{i}"]["observations"]
            )
        if self.render_env:
            self.render()

        return obs, info

    def step(self, action_dict):
        self.step_count += 1
        rewards = {}
        info = {}
        obs = {}
        terminated = {}
        truncated = {}
        reached_goal = {}

        # Default to no-op actions if no actions are provided or missing agent actions
        if not action_dict or len(action_dict) != self.num_agents:
            print("action_dict:", action_dict)
            action_dict = {agent_id: 0 for agent_id in self._agent_ids}
            logging.warning(
                "No actions provided or missing agent actions. Defaulting to no-op actions: %s",
                action_dict,
            )

        for i in range(self.num_agents):
            rewards[f"agent_{i}"] = 0
            reached_goal[f"agent_{i}"] = False
            action = action_dict[f"agent_{i}"]

            pos = self.positions[f"agent_{i}"]
            next_pos = self.get_next_position(action, pos)

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
            else:
                # rewards[f"agent_{i}"] -= 0.1
                print(
                    f"Invalid move for agent {i} with action {action} at position {pos}"
                )

            obs[f"agent_{i}"] = {}
            obs[f"agent_{i}"]["position"] = np.array(self.positions[f"agent_{i}"])
            obs[f"agent_{i}"]["goal"] = np.array(self.goals[f"agent_{i}"])
            obs[f"agent_{i}"]["observations"] = self.get_obs(f"agent_{i}")
            obs[f"agent_{i}"]["action_mask"] = self.get_action_mask(
                obs[f"agent_{i}"]["observations"]
            )

            if np.array_equal(self.positions[f"agent_{i}"], self.goals[f"agent_{i}"]):
                reached_goal[f"agent_{i}"] = True
                if not self.goal_reached_once[f"agent_{i}"]:
                    self.goal_reached_once[f"agent_{i}"] = True
                    rewards[f"agent_{i}"] += 0.5
                # print(
                #     f"Agent {i} reached its goal, because {self.positions[f'agent_{i}']} == {self.goals[f'agent_{i}']}"
                # )
            # else:
            # print(
            #     f"Agent {i} did not reach its goal, because {self.positions[f'agent_{i}']} != {self.goals[f'agent_{i}']}"
            # )

        # minus reward for agents on the same position, shouldn't happen
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if np.array_equal(
                    self.positions[f"agent_{i}"], self.positions[f"agent_{j}"]
                ):
                    rewards[f"agent_{i}"] -= 1
                    rewards[f"agent_{j}"] -= 1
                    print(
                        f"Agents {i} and {j} are on the same position {self.positions[f'agent_{i}']}"
                    )

        # If all agents have reached their goals, end the episode (not truncated)
        if all(reached_goal.values()):
            for i in range(self.num_agents):
                rewards[f"agent_{i}"] += 1
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
            # minus reward for not reaching the goal
            for i in range(self.num_agents):
                if not np.array_equal(
                    self.positions[f"agent_{i}"], self.goals[f"agent_{i}"]
                ):
                    rewards[f"agent_{i}"] -= 1
            terminated["__all__"] = True
            truncated["__all__"] = True
        else:
            terminated["__all__"] = False
            truncated["__all__"] = False

        if self.render_env:
            self.render()

        # print("Stepping env with number of obs:", len(obs))
        return obs, rewards, terminated, truncated, info

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

    def get_obs(self, agent_id: str):
        """
        Get the observation for a given agent.
        Parameters:
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
            self.observation_space["observations"].shape,
            dtype=self.observation_space["observations"].dtype,
        )

        for i in range(self.observation_space["observations"].shape[0]):
            for j in range(self.observation_space["observations"].shape[1]):
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
                    else:
                        # Check if another agent occupies the cell
                        if any(
                            np.array_equal(self.positions[agent], (x, y))
                            for agent in self.positions
                            if agent != agent_id
                        ):
                            obs[i, j] = 2
                        # Check if this is current agent's goal and not occupied by another agent
                        elif np.array_equal(self.goals[agent_id], (x, y)):
                            obs[i, j] = 3
                        # Check if this is another agent's goal and not occupied by another agent
                        elif any(
                            np.array_equal(self.goals[agent], (x, y))
                            for agent in self.positions
                            if agent != agent_id
                        ):
                            obs[i, j] = 4
                else:
                    # If the cell is outside the grid, treat it as an obstacle
                    obs[i, j] = 1

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
            Movement actions (1-4) are only possible if the corresponding cell in the observation is empty or a goal.
        """

        action_mask = np.zeros(
            self.observation_space["action_mask"].shape,
            dtype=self.observation_space["action_mask"].dtype,
        )

        action_mask[0] = 1  # No-op action is always possible

        pos = [
            self.sensor_range,
            self.sensor_range,
        ]
        x, y = pos

        if x > 0 and (obs[x - 1, y] == 0 or obs[x - 1, y] == 3 or obs[x - 1, y] == 4):
            action_mask[1] = 1  # Move up

        if y < obs.shape[1] - 1 and (
            obs[x, y + 1] == 0 or obs[x, y + 1] == 3 or obs[x, y + 1] == 4
        ):
            action_mask[2] = 1  # Move right

        if x < obs.shape[0] - 1 and (
            obs[x + 1, y] == 0 or obs[x + 1, y] == 3 or obs[x + 1, y] == 4
        ):
            action_mask[3] = 1  # Move down

        if y > 0 and (obs[x, y - 1] == 0 or obs[x, y - 1] == 3 or obs[x, y - 1] == 4):
            action_mask[4] = 1  # Move left

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
            self.ax.grid(True, which="both", color="gray", linestyle="-", linewidth=0.5)

            # Reverse the y-axis to match typical grid layout (0,0 at top-left)
            self.ax.invert_yaxis()

        else:
            # Update agent positions
            for agent_id, agent_patch in self.agent_patches.items():
                pos = self.positions[agent_id]
                agent_patch.set_center((pos[1] + 0.5, pos[0] + 0.5))
                sensor_patch = self.sensor_patches[agent_id]
                sensor_patch.set_xy(
                    (pos[1] - self.sensor_range, pos[0] - self.sensor_range)
                )

        # Redraw the updated plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        return True

    def close(self):
        plt.close()
