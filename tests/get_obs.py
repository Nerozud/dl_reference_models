import numpy as np
import gymnasium as gym


class TestEnvironment:
    """Test environment class for agents."""

    def __init__(self):
        self.grid = np.array([[1, 1, 0, 1], [0, 0, 0, 1], [1, 0, 1, 1], [0, 0, 0, 0]])
        self.sensor_range = 2
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
        self.positions = {"agent_0": (1, 1), "agent_1": (3, 3)}
        self.goals = {"agent_0": (1, 2), "agent_1": (0, 2)}

    def get_obs(self, agent_id: str):
        """Get the observations for the specified agent."""
        pos = self.positions[agent_id]
        obs = np.zeros(
            self.observation_space["observations"].shape,
            dtype=self.observation_space["observations"].dtype,
        )

        for i in range(self.observation_space["observations"].shape[0]):
            for j in range(self.observation_space["observations"].shape[1]):
                x = pos[0] - self.sensor_range + i
                y = pos[1] - self.sensor_range + j

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

    def get_action_mask(self, obs):
        """
        Get the action mask for a given agent.
        Parameters:
            agent_id (str): The ID of the agent.
            obs (numpy.ndarray): The observation array for the agent.
        Returns:
            numpy.ndarray: The action mask array.
        Description:
            This function calculates the action mask array for a given agent based on its observation.
            The action mask array is a binary array indicating which actions are valid for the agent.
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


# Initialize the test environment
env = TestEnvironment()

# Test case 1: Agent 0
expected_obs_agent_0 = np.array([[1, 1, 4], [0, 0, 3], [1, 0, 1]])
obs_agent_0 = env.get_obs("agent_0")
assert np.array_equal(
    obs_agent_0, expected_obs_agent_0
), f"Test failed for agent_0, got {obs_agent_0}, expected {expected_obs_agent_0}"

expected_action_mask_agent_0 = np.array([1, 0, 1, 1, 1])
action_mask_agent_0 = env.get_action_mask(obs_agent_0)
assert np.array_equal(
    action_mask_agent_0, expected_action_mask_agent_0
), f"Test failed for agent_0, got {action_mask_agent_0}, expected {expected_action_mask_agent_0}, with obs {obs_agent_0}"

# Test case 2: Agent 1
expected_obs_agent_1 = np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1]])
obs_agent_1 = env.get_obs("agent_1")
assert np.array_equal(
    obs_agent_1, expected_obs_agent_1
), f"Test failed for agent_1, got {obs_agent_1}, expected {expected_obs_agent_1}"

expected_action_mask_agent_1 = np.array([1, 0, 0, 0, 1])
action_mask_agent_1 = env.get_action_mask(obs_agent_1)
assert np.array_equal(
    action_mask_agent_1, expected_action_mask_agent_1
), f"Test failed for agent_1, got {action_mask_agent_1}, expected {expected_action_mask_agent_1}, with obs {obs_agent_1}"

print("All tests passed!")
print("Obs agent 0:", obs_agent_0)
print("Action mask agent 0:", action_mask_agent_0)
print("Obs agent 1:", obs_agent_1)
print("Action mask agent 1:", action_mask_agent_1)
