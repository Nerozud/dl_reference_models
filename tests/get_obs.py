"""
This module contains a TestEnvironment class that represents a test environment for agents. It provides a method to get observations for a given agent.
Example usage:
    # Get observations for Agent 0
    # Get observations for Agent 1
    # Print the observations
    print(obs_agent_0)
    print(obs_agent_1)
        Initializes the TestEnvironment class.
        The grid represents the environment grid with obstacles (0) and free spaces (1).
        The observation_space represents the observation space for each agent.
        The positions dictionary stores the positions of each agent.
        The goals dictionary stores the goals for each agent.
        Gets the observations for the specified agent.
        Args:
            agent_id (str): The ID of the agent.
        Returns:
            np.ndarray: The observations for the agent.
"""

import numpy as np


class TestEnvironment:
    """Test environment class for agents."""

    def __init__(self):
        self.grid = np.array([[1, 1, 0, 1], [0, 0, 0, 1], [1, 0, 1, 1], [0, 0, 0, 0]])
        self.observation_space = np.zeros((3, 3), dtype=int)
        self.positions = {"agent_0": (1, 1), "agent_1": (3, 3)}
        self.goals = {"agent_0": (1, 2), "agent_1": (0, 2)}

    def get_obs(self, agent_id: str):
        """Get the observations for the specified agent."""
        pos = self.positions[agent_id]
        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

        for i in range(self.observation_space.shape[0]):
            for j in range(self.observation_space.shape[1]):
                x = pos[0] - 1 + i
                y = pos[1] - 1 + j

                if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]:
                    obs[i, j] = 0

                    if self.grid[x, y] == 1:
                        obs[i, j] = 1
                    elif any(
                        self.positions[agent] == (x, y)
                        for agent in self.positions
                        if agent != agent_id
                    ):
                        obs[i, j] = 2
                    elif self.goals[agent_id] == (x, y):
                        obs[i, j] = 3
                    elif any(
                        self.goals[agent] == (x, y)
                        for agent in self.positions
                        if agent != agent_id
                    ):
                        obs[i, j] = 4
                else:
                    obs[i, j] = 1

        return obs


# Initialize the test environment
env = TestEnvironment()

# Test case 1: Agent 0
expected_obs_agent_0 = np.array([[1, 1, 4], [0, 0, 3], [1, 0, 1]])
obs_agent_0 = env.get_obs("agent_0")
assert np.array_equal(
    obs_agent_0, expected_obs_agent_0
), f"Test failed for agent_0, got {obs_agent_0}, expected {expected_obs_agent_0}"

# Test case 2: Agent 1
expected_obs_agent_1 = np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1]])
obs_agent_1 = env.get_obs("agent_1")
assert np.array_equal(
    obs_agent_1, expected_obs_agent_1
), f"Test failed for agent_1, got {obs_agent_1}, expected {expected_obs_agent_1}"

print("All tests passed!")
