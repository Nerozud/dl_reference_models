""" A* algorithm for multiple agents in a grid world environment. """

import os
import time
from datetime import datetime
from heapq import heappop, heappush
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

SEED = 42  # int or None, same seed creates same sequence of starts and goals
NUM_AGENTS = 5
NUM_EPISODES = 100
MAX_CPU_TIME = 60  # seconds
DRAW_PLOT = False

results = []
rng = np.random.default_rng(SEED)  # set seed for reproducibility

# block layout 2.1
grid_list = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# layout with dead ends 2.1 b
# grid_list = [
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# ]

# fishbone layout 2.2
# grid_list = [
#     [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
#     [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
#     [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
#     [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
#     [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
#     [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
#     [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
#     [1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
#     [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1],
#     [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# ]


# TODO: use get_grid function to get the grid
grid = np.array(grid_list)

# Create a heatmap to track the number of visits to each cell
visit_counts = np.zeros_like(grid)

# make one additional episode, because RL looses one episode for some reason
for episode in range(NUM_EPISODES + 1):
    # Record the start time
    start_time = time.process_time()

    starts = {}
    goals = {}
    available_positions = np.argwhere(grid == 0)

    # Generate unique starting positions
    for i in range(NUM_AGENTS):
        while True:
            idx = rng.choice(len(available_positions))
            # print("idx", idx, "from", len(available_positions))
            start_pos = tuple(available_positions[idx])
            if start_pos not in starts.values():
                starts[f"agent_{i}"] = start_pos
                break
    # print("starts", starts)
    # Generate unique goal positions
    for i in range(NUM_AGENTS):
        while True:
            idx = rng.choice(len(available_positions))
            goal_pos = tuple(available_positions[idx])
            if goal_pos not in goals.values() and goal_pos not in starts.values():
                goals[f"agent_{i}"] = goal_pos
                break

    starts_list = [starts[f"agent_{i}"] for i in range(NUM_AGENTS)]
    goals_list = [goals[f"agent_{i}"] for i in range(NUM_AGENTS)]

    # Define the start and goal positions for multiple agents
    # blue green red pink
    # starts = [(5, 0), (3, 12), (6, 5), (6, 14)]
    # goals = [(6, 6), (9, 3), (6, 0), (3, 3)]

    # Define directions for movement (left, right, stay, up, down)
    directions = [
        (0, -1),
        (0, 1),
        (0, 0),
        (-1, 0),
        (1, 0),
    ]  # left, right, stay, up, down

    def heuristic(pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def a_star_multiple_agents(starts, goals, grid):
        # Priority queue for A*
        open_set = []
        heappush(open_set, (0, starts, [starts]))

        # Visited set to avoid processing the same state
        visited = set([tuple(starts)])

        while open_set:
            if time.process_time() - start_time > MAX_CPU_TIME:
                print(
                    f"A* search exceeded the time limit ({MAX_CPU_TIME}) with {time.process_time() - start_time} seconds."
                )
                return None

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
    paths = a_star_multiple_agents(starts_list, goals_list, grid_list)

    # Record the end time
    end_time = time.process_time()

    # Calculate the CPU time
    cpu_time = end_time - start_time

    episode_data = {
        "episode": episode,
        "cpu_time": cpu_time,
        "seed": SEED,
    }

    if paths:
        max_steps = 0  # To find the max number of steps among agents
        for agent_index, agent_path in enumerate(zip(*paths)):
            agent_path = [step[agent_index] for step in paths]
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
                if position == goals_list[agent_index]:
                    break

            # Update max_steps if this agent took more steps
            if steps_to_goal > max_steps:
                max_steps = steps_to_goal

            print(f"Path for agent {agent_index + 1}: {agent_path}")
            print(f"Number of steps for agent {agent_index + 1}: {len(agent_path)}")
            print(
                f"Total distance traveled by agent {agent_index + 1}: {total_distance}"
            )

            # Collect per-agent data
            episode_data["max_steps"] = max_steps
            agent_id = agent_index
            episode_data[f"agent_{agent_id}_steps"] = steps_to_goal
            episode_data[f"agent_{agent_id}_total_distance"] = total_distance
            start_pos = starts_list[agent_index]
            goal_pos = goals_list[agent_index]
            episode_data[f"agent_{agent_id}_start_x"] = start_pos[0]
            episode_data[f"agent_{agent_id}_start_y"] = start_pos[1]
            episode_data[f"agent_{agent_id}_goal_x"] = goal_pos[0]
            episode_data[f"agent_{agent_id}_goal_y"] = goal_pos[1]

        for path in zip(*paths):
            for position in path:
                visit_counts[position] += 1

    else:
        print("No path found.")
        # Handle the case when no path is found
        episode_data["max_steps"] = None
        for agent_index in range(NUM_AGENTS):
            agent_id = agent_index + 1
            episode_data[f"agent_{agent_id}_steps"] = None
            episode_data[f"agent_{agent_id}_total_distance"] = None
            start_pos = starts_list[agent_index]
            goal_pos = goals_list[agent_index]
            episode_data[f"agent_{agent_id}_start_x"] = start_pos[0]
            episode_data[f"agent_{agent_id}_start_y"] = start_pos[1]
            episode_data[f"agent_{agent_id}_goal_x"] = goal_pos[0]
            episode_data[f"agent_{agent_id}_goal_y"] = goal_pos[1]

    print(f"CPU time: {cpu_time:.4f} seconds for episode {episode}.")

    results.append(episode_data)

# After all episodes, create DataFrame and save to CSV
df = pd.DataFrame(results)

# Ensure the directory exists
os.makedirs("experiments/results", exist_ok=True)

# Generate a suitable filename with current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_file = os.path.join(
    "experiments/results",
    f"A_star_{NUM_AGENTS}agents_{current_time}.csv",
)

# Save the DataFrame to CSV
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")

# Plot the heatmap
plt.figure()
ax = plt.gca()
plt.xlabel("X")
plt.ylabel("Y")

# Note: origin='upper' matches the indexing [y,x] with y=0 at top
heatmap_plot = plt.imshow(visit_counts, origin="upper")

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(heatmap_plot, label="Number of visits", cax=cax)

# Save the heatmap
heatmap_file = os.path.join(
    "experiments/results", f"A_star_{NUM_AGENTS}agents_{current_time}_heatmap.pdf"
)
plt.savefig(heatmap_file, bbox_inches="tight")
print(f"Heatmap saved to {heatmap_file}")

# Length of the largest sublist
if grid_list:
    length_of_largest_sublist = max(len(sublist) for sublist in grid_list)
else:
    length_of_largest_sublist = 0  # Define as 0 if the outer list is empty


if DRAW_PLOT:
    fig, ax = plt.subplots(figsize=(10, 8))

    # Initialize the lines for each agent
    lines = [
        ax.plot([], [], marker="o", label=f"Agent {i+1}")[0] for i in range(NUM_AGENTS)
    ]

    # Set up the plot limits
    ax.set_xlim(-1, len(grid_list))
    ax.set_ylim(-1, length_of_largest_sublist)
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
