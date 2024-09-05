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

    def get_action_mask(self, agent_id: str, obs):
        """Get the action mask for a given agent."""
        action_mask = np.zeros(5, dtype=int)

        action_mask[0] = 1  # No-op action is always possible

        pos = [1, 1]  # should be flexible depending on grid obs shape
        x, y = pos
        print("pos", pos, "x", x, "y", y)
        print("obs", obs)
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
action_mask_agent_0 = env.get_action_mask("agent_0", obs_agent_0)
assert np.array_equal(
    action_mask_agent_0, expected_action_mask_agent_0
), f"Test failed for agent_0, got {action_mask_agent_0}, expected {expected_action_mask_agent_0}"

# Test case 2: Agent 1
expected_obs_agent_1 = np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1]])
obs_agent_1 = env.get_obs("agent_1")
assert np.array_equal(
    obs_agent_1, expected_obs_agent_1
), f"Test failed for agent_1, got {obs_agent_1}, expected {expected_obs_agent_1}"

expected_action_mask_agent_1 = np.array([1, 0, 0, 0, 0])
action_mask_agent_1 = env.get_action_mask("agent_1", obs_agent_1)
assert np.array_equal(
    action_mask_agent_1, expected_action_mask_agent_1
), f"Test failed for agent_1, got {action_mask_agent_1}, expected {expected_action_mask_agent_1}"

print("All tests passed!")
