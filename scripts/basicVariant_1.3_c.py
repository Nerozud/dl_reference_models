# 3.11 a

from heapq import heappop, heappush
from itertools import product
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time  # Import the time module

# Record the start time
start_time = time.process_time()

# Define the grid
grid = [
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
]

# Define the start and goal positions for multiple agents
starts = [(2, 1), (2, 3)]
goals = [(4, 3), (4, 1)]

# Define directions for movement (left, right, stay, up, down)
directions = [(0, -1), (0, 1), (0, 0), (-1, 0), (1, 0)]  # left, right, stay, up, down


def heuristic(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def a_star_multiple_agents(starts, goals, grid):
    # Priority queue for A*
    open_set = []
    heappush(open_set, (0, starts, [starts]))

    # Visited set to avoid processing the same state
    visited = set([tuple(starts)])

    while open_set:
        _, current_positions, paths = heappop(open_set)

        # Check if all goals are reached
        if all(current_positions[i] == goals[i] for i in range(len(goals))):
            return paths

        for moves in product(directions, repeat=len(starts)):
            next_positions = [
                (
                    current_positions[i][0] + moves[i][0],
                    current_positions[i][1] + moves[i][1],
                )
                for i in range(len(starts))
            ]

            # Check for collisions and out-of-bound moves
            if all(
                0 <= next_positions[i][0] < len(grid)
                and 0 <= next_positions[i][1] < len(grid[0])
                and grid[next_positions[i][0]][next_positions[i][1]] != 1
                for i in range(len(next_positions))
            ):
                # Check for position collisions
                if len(set(next_positions)) == len(next_positions):
                    # Check for swap collisions
                    if all(
                        next_positions[i] != current_positions[j]
                        or next_positions[j] != current_positions[i]
                        for i in range(len(starts))
                        for j in range(len(starts))
                        if i != j
                    ):
                        new_state = tuple(next_positions)
                        if new_state not in visited:
                            visited.add(new_state)
                            g_cost = len(paths) + 1  # each move costs 1
                            f_cost = g_cost + sum(
                                heuristic(next_positions[i], goals[i])
                                for i in range(len(goals))
                            )
                            heappush(
                                open_set,
                                (f_cost, next_positions, paths + [next_positions]),
                            )

    return None


# Find the simultaneous paths
paths = a_star_multiple_agents(starts, goals, grid)

# Record the end time
end_time = time.process_time()

# Calculate the CPU time
cpu_time = end_time - start_time

if paths:
    for agent_index, agent_path in enumerate(zip(*paths)):
        steps_to_goal = 0
        total_distance = 0

        for step, position in enumerate(agent_path):
            steps_to_goal += 1

            # Compute the distance between consecutive positions
            if step > 0:
                previous_position = agent_path[step - 1]
                manhattan_distance = abs(position[0] - previous_position[0]) + abs(
                    position[1] - previous_position[1]
                )
                total_distance += manhattan_distance

            # Stop counting when the agent reaches the goal
            if position == goals[agent_index]:
                break

        print(f"Path for agent {agent_index + 1}: {agent_path}")
        print(f"Number of steps for agent {agent_index + 1}: {steps_to_goal - 1}")
        print(f"Total distance traveled by agent {agent_index + 1}: {total_distance}")
else:
    print("No path found.")

print(f"CPU time: {cpu_time:.4f} seconds")

number_of_agents = len(starts)

fig, ax = plt.subplots(figsize=(10, 8))

# Initialize the lines for each agent
lines = [
    ax.plot([], [], marker="o", label=f"Agent {i+1}")[0]
    for i in range(number_of_agents)
]

# Set up the plot limits
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 13)
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_title("Paths of Agents")
ax.legend()
ax.grid(True)


# Function to initialize the plot
def init():
    for line in lines:
        line.set_data([], [])
    return lines


# Function to update the plot for each frame
def update(num):
    for i, path in enumerate(zip(*paths)):
        if num < len(path):
            x, y = zip(*path[: num + 1])
            lines[i].set_data(x, y)
    return lines


# Create the animation
ani = animation.FuncAnimation(
    fig,
    update,
    frames=max(len(path) for path in zip(*paths)),
    init_func=init,
    blit=True,
    repeat=False,
)

plt.show()
