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
import matplotlib.patches as patches
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
        -1 for each invalid move
        1 for reaching its goal
        0 otherwise.
    - The episode terminates after a fixed number of steps or when all agents reach their goals.
    """

    def __init__(self, env_config):
        super().__init__()

        self.step_count = 0
        self.steps_per_episode = 100
        self.num_agents = env_config.get("num_agents", 2)
        self._agent_ids = {f"agent_{i}" for i in range(self.num_agents)}

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
        # TODO: Implement the shape(vision range) depending on the env_config
        self.observation_space = gym.spaces.Box(
            low=0,
            high=4,
            shape=(3, 3),
            dtype=np.uint8,
        )

        # 0 - noop, 1 - up, 2 - right, 3 - down, 4 - left
        self.action_space = gym.spaces.Discrete(5)

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        info = {}
        obs = {}
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
            obs[f"agent_{i}"] = self.get_obs(f"agent_{i}")

        return obs, info

    def step(self, action_dict):
        self.step_count += 1
        rewards = {}
        info = {}
        obs = {}

        if not action_dict or len(action_dict) != self.num_agents:
            action_dict = {agent_id: 0 for agent_id in self._agent_ids}
            logging.warning(
                "No actions provided or missing agent actions. Defaulting to no-op actions: %s",
                action_dict,
            )

        for i in range(self.num_agents):
            rewards[f"agent_{i}"] = 0
            action = action_dict[f"agent_{i}"]

            if action == 0:  # noop
                continue

            pos = self.positions[f"agent_{i}"]
            if action == 1:  # up
                next_pos = (pos[0] - 1, pos[1])
            elif action == 2:  # right
                next_pos = (pos[0], pos[1] + 1)
            elif action == 3:  # down
                next_pos = (pos[0] + 1, pos[1])
            elif action == 4:  # left
                next_pos = (pos[0], pos[1] - 1)

            if (
                0 <= next_pos[0] < self.grid.shape[0]
                and 0 <= next_pos[1] < self.grid.shape[1]
                and self.grid[next_pos[0], next_pos[1]] == 0
            ):
                self.positions[f"agent_{i}"] = next_pos
            else:
                rewards[f"agent_{i}"] = (
                    rewards.get(f"agent_{i}", 0) - 1
                )  # TODO: Instead use an action mask
                # print(
                #     f"Invalid move for agent {i} with action {action} at position {pos}"
                # )

            obs[f"agent_{i}"] = self.get_obs(f"agent_{i}")

        terminated = {}
        truncated = {}

        # Track if all agents have reached their goals
        all_agents_reached_goals = True

        for i in range(len(self.positions)):
            agent_id = f"agent_{i}"

            if np.array_equal(self.positions[agent_id], self.goals[agent_id]):
                rewards[agent_id] = rewards.get(agent_id, 0) + 1
                terminated[agent_id] = True
            else:
                terminated[agent_id] = False
                all_agents_reached_goals = False

        # If all agents have reached their goals, end the episode (not truncated)
        if all_agents_reached_goals:
            terminated["__all__"] = True
            truncated["__all__"] = False
            print("All agents reached their goals in", self.step_count, "steps")
            print("Positions:", self.positions)
            print("Goals:", self.goals)

        # If the step limit is reached, end the episode and mark it as truncated
        elif self.step_count >= self.steps_per_episode:
            terminated["__all__"] = True
            truncated["__all__"] = True
        else:
            terminated["__all__"] = False
            truncated["__all__"] = False

        return obs, rewards, terminated, truncated, info

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
        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

        for i in range(self.observation_space.shape[0]):
            for j in range(self.observation_space.shape[1]):
                # Calculate the corresponding position on the grid
                x = pos[0] - 1 + i
                y = pos[1] - 1 + j

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

    def render(self):
        """
        Render the environment using Matplotlib.
        """
        # Create a plot
        _, ax = plt.subplots(figsize=(8, 4))

        # Draw the grid
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] == 1:  # Obstacle cell
                    ax.add_patch(patches.Rectangle((j, i), 1, 1, color="black"))
                else:  # Empty cell
                    ax.add_patch(
                        patches.Rectangle((j, i), 1, 1, edgecolor="gray", fill=False)
                    )

        # Draw agents and their goals
        colors = [
            "red",
            "blue",
            "green",
            "purple",
            "orange",
        ]  # Add more colors if needed
        for i in range(self.num_agents):
            agent_id = f"agent_{i}"
            pos = self.positions[agent_id]
            goal = self.goals[agent_id]

            # Draw the agent
            ax.add_patch(
                patches.Circle(
                    (pos[1] + 0.5, pos[0] + 0.5),
                    0.3,
                    color=colors[i % len(colors)],
                    label=f"Agent {i}",
                )
            )

            # Draw the goal
            ax.add_patch(
                patches.Circle(
                    (goal[1] + 0.5, goal[0] + 0.5),
                    0.15,
                    color=colors[i % len(colors)],
                    alpha=0.5,
                    label=f"Goal {i}",
                )
            )

        # Set the limits and aspect
        ax.set_xlim(0, self.grid.shape[1])
        ax.set_ylim(0, self.grid.shape[0])
        ax.set_aspect("equal")

        # Add grid lines for clarity
        ax.set_xticks(np.arange(0, self.grid.shape[1], 1))
        ax.set_yticks(np.arange(0, self.grid.shape[0], 1))
        ax.grid(True, which="both", color="gray", linestyle="-", linewidth=0.5)

        # Reverse the y-axis to match typical grid layout (0,0 at top-left)
        ax.invert_yaxis()

        plt.legend()
        plt.show()
