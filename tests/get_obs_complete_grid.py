import numpy as np

num_agents = 4

positions = {"agent_0": (1, 1), "agent_1": (1, 0), "agent_2": (1, 2), "agent_3": (1, 3)}
goals = {"agent_0": (1, 2), "agent_1": (1, 7), "agent_2": (1, 3), "agent_3": (1, 1)}

grid = np.array(
    [
        [1, 1, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)

obs = grid.copy()

for i in range(num_agents):
    agent_id = f"agent_{i}"
    goal = goals[agent_id]
    obs[goal[0], goal[1]] = i * 2 + 3

for i in range(num_agents):
    agent_id = f"agent_{i}"
    pos = positions[agent_id]
    obs[pos[0], pos[1]] = i * 2 + 2


print(obs)
