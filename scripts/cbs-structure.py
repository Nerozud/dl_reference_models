""" CBS algorithm for multiple agents in a grid world environment. """

import os
import heapq
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
NUM_AGENTS = 12
NUM_EPISODES = 100
MAX_CPU_TIME = 60  # seconds
DRAW_PLOT = False


###############################################################################
def a_star(grid, start, goal, constraints, agent_id):
    """
    Compute a path from start to goal using A*, considering vertex and edge constraints.

    Parameters
    ----------
    grid : list[list[int]]
        2D grid where 0 = free cell, 1 = blocked cell.
    start : tuple(int, int)
        Starting cell (row, col).
    goal : tuple(int, int)
        Goal cell (row, col).
    constraints : list[tuple]
        List of constraints in the form:
            - (agent_id, 'vertex', (row, col), time)
            - (agent_id, 'edge', ((rA, cA), (rB, cB)), time)
    agent_id : int
        The ID of the agent for whom we are planning the path.

    Returns
    -------
    list[tuple(int, int)] or None
        Path from start to goal if found, otherwise None.
    """

    def heuristic(cell, goal, constrained_positions):
        penalty = 50
        # Manhattan distance
        h = abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
        if (cell) in constrained_positions:
            h += penalty
        return h

    # Separate constraints for this agent into vertex and edge constraints
    vertex_constraints = set()
    edge_constraints = set()
    constrained_positions = set()
    for c_agent, ctype, data, ctime in constraints:
        if c_agent != agent_id:
            continue
        if ctype == "vertex":
            # Vertex constraint: cannot be at 'data' at 'ctime'
            vertex_constraints.add((data, ctime))
            constrained_positions.add(data)
        elif ctype == "edge":
            # Edge constraint: cannot move from 'data[0]' to 'data[1]' at 'ctime'
            edge_constraints.add((data[0], data[1], ctime))
            constrained_positions.add(data[0])
            constrained_positions.add(data[1])

    def get_neighbors(cell):
        # 4-directional moves plus waiting
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (0, 0),
        ]  # Added (0,0) for waiting
        neighbors = []
        for dx, dy in directions:
            nr, nc = cell[0] + dx, cell[1] + dy
            if (
                (0 <= nr < len(grid))
                and (0 <= nc < len(grid[0]))
                and (grid[nr][nc] == 0)
            ):
                neighbors.append((nr, nc))
        return neighbors

    # Initialize open list and scores
    open_list = []
    heapq.heappush(
        open_list,
        (heuristic(start, goal, constrained_positions), start, 0),
    )  # (f_score, cell, time)
    came_from = {}  # (cell, time) -> (prev_cell, prev_time)
    g_score = {(start, 0): 0}

    while open_list:
        current_f, current_cell, current_time = heapq.heappop(open_list)

        # Goal check: if current_cell is goal, reconstruct path
        if current_cell == goal:
            # To ensure agents can stay at the goal without conflicts, extend the path
            path = []
            back_state = (current_cell, current_time)
            while back_state in came_from:
                path.append(back_state[0])
                back_state = came_from[back_state]
            path.append(start)
            path.reverse()
            return path

        # Explore neighbors
        for nxt in get_neighbors(current_cell):
            next_time = current_time + 1

            # Check vertex constraint: cannot be in 'nxt' at 'next_time'
            if (nxt, next_time) in vertex_constraints:
                continue

            # Check edge constraint: cannot move from 'current_cell' to 'nxt' at 'current_time'
            if (current_cell, nxt, next_time) in edge_constraints:
                continue

            tentative_g = g_score[(current_cell, current_time)] + 1

            # Check if this state has been visited with a lower g_score
            if ((nxt, next_time) not in g_score) or (
                tentative_g < g_score[(nxt, next_time)]
            ):
                g_score[(nxt, next_time)] = tentative_g
                came_from[(nxt, next_time)] = (current_cell, current_time)
                f_score = tentative_g + heuristic(nxt, goal, constrained_positions)
                heapq.heappush(open_list, (f_score, nxt, next_time))

    # No path found
    return None


# --------------------------------------------------------
# 2. Data structure for CBS node
# --------------------------------------------------------
class CBSNode:
    """Stores constraints, paths, and total cost for a CBS node."""

    def __init__(self, constraints, paths, cost):
        self.constraints = constraints  # list of (agent_id, ctype, data, time)
        self.paths = paths  # list of paths for each agent
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


# --------------------------------------------------------
# 3. Utility: compute total cost and detect conflicts
# --------------------------------------------------------
def compute_cost(paths):
    """Compute total cost as the sum of path lengths."""
    return sum(len(path) for path in paths)


def detect_conflict(paths):
    """
    Detect the first conflict among a set of paths.
    Types of conflicts:
      1. Vertex Conflict: two agents occupy the same cell at the same time.
      2. Edge Conflict: two agents swap cells between consecutive time steps.

    Parameters
    ----------
    paths : list[list[tuple(int, int)]]
        List of paths for each agent.

    Returns
    -------
    tuple or None
        If a conflict is found, returns a tuple:
            (i, j, conflict_type, details, t)
        where:
            i, j : agent indices
            conflict_type : 'vertex' or 'edge'
            details :
                - For 'vertex': (row, col)
                - For 'edge': ((row1, col1), (row2, col2))
            t : time step of conflict
        If no conflict is found, returns None.
    """
    if not paths:
        return None

    num_agents = len(paths)
    max_time = max(len(p) for p in paths)

    for t in range(max_time):
        # Maps to track positions and movements
        position_map = {}
        movement_map = {}

        for i, path in enumerate(paths):
            # Current position
            pos = path[t] if t < len(path) else path[-1]

            # Previous position (for edge conflict)
            if t == 0:
                prev_pos = pos
            else:
                prev_pos = path[t - 1] if (t - 1) < len(path) else path[-1]

            # Check for vertex conflict
            if pos in position_map:
                j = position_map[pos]
                # print(
                #     f"Vertex conflict between Agent {i} and Agent {j} at {pos} at time {t}"
                # )
                return (i, j, "vertex", pos, t)
            position_map[pos] = i

            # Record movement for edge conflict
            movement_map[i] = (prev_pos, pos)

        # After recording all movements at time t, check for edge conflicts
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if i not in movement_map or j not in movement_map:
                    continue
                move_i = movement_map[i]
                move_j = movement_map[j]
                # Edge conflict: agent i moves from A to B and agent j moves from B to A
                if move_i[0] == move_j[1] and move_i[1] == move_j[0]:
                    # print(
                    #     f"Edge conflict between Agent {i} and Agent {j} swapping {move_i[0]} <-> {move_i[1]} at time {t}"
                    # )
                    return (i, j, "edge", (move_i[0], move_i[1]), t)

    return None


# --------------------------------------------------------
# 4. The high-level CBS
# --------------------------------------------------------
def cbs(grid, starts, goals):
    """
    Conflict-Based Search for multi-agent pathfinding.

    Parameters
    ----------
    grid : list[list[int]]
        2D grid with 0 = free, 1 = obstacle.
    starts : list[tuple(int, int)]
        Start positions for each agent.
    goals : list[tuple(int, int)]
        Goal positions for each agent.

    Returns
    -------
    list[list[tuple(int, int)]] or None
        List of paths, one per agent, if a solution is found. Otherwise None.
    """
    num_agents = len(starts)

    # 1. Create the root node with no constraints and an initial path for each agent
    root_constraints = []
    root_paths = []
    for i in range(num_agents):
        path = a_star(grid, starts[i], goals[i], root_constraints, i)
        if not path:
            # print(f"No initial path found for agent {i}.")
            return None  # If any agent can't reach its goal with no constraints, fail
        root_paths.append(path)
    root_cost = compute_cost(root_paths)
    root_node = CBSNode(root_constraints, root_paths, root_cost)

    # Initialize open list as a priority queue sorted by cost
    open_list = []
    heapq.heappush(open_list, (root_node.cost, root_node))

    while open_list:
        if time.process_time() - start_time > MAX_CPU_TIME:
            print(
                f"CBS exceeded the time limit ({MAX_CPU_TIME}) with {time.process_time() - start_time} seconds."
            )
            return None
        current_cost, node = heapq.heappop(open_list)

        # 2. Check for conflicts
        conflict = detect_conflict(node.paths)
        if not conflict:
            # No conflicts => valid solution
            return node.paths

        # 3. Resolve conflict by creating two child nodes with additional constraints
        i, j, conflict_type, details, t = conflict

        if conflict_type == "vertex":
            conflict_cell = details
            # print(
            #     f"Resolving vertex conflict between Agent {i} and Agent {j} at {conflict_cell} at time {t}"
            # )

            # Child 1: Constrain agent i from being at conflict_cell at time t
            child1_constraints = node.constraints.copy()
            child1_constraints.append((i, "vertex", conflict_cell, t))
            child1_paths = node.paths.copy()
            path1 = a_star(grid, starts[i], goals[i], child1_constraints, i)
            if path1:
                child1_paths[i] = path1
                child1_cost = compute_cost(child1_paths)
                child1_node = CBSNode(child1_constraints, child1_paths, child1_cost)
                heapq.heappush(open_list, (child1_node.cost, child1_node))
                # print(
                #     f"Child1 created by constraining Agent {i} from {conflict_cell} at time {t}"
                # )

            # Child 2: Constrain agent j from being at conflict_cell at time t
            child2_constraints = node.constraints.copy()
            child2_constraints.append((j, "vertex", conflict_cell, t))
            child2_paths = node.paths.copy()
            path2 = a_star(grid, starts[j], goals[j], child2_constraints, j)
            if path2:
                child2_paths[j] = path2
                child2_cost = compute_cost(child2_paths)
                child2_node = CBSNode(child2_constraints, child2_paths, child2_cost)
                heapq.heappush(open_list, (child2_node.cost, child2_node))
                # print(
                #     f"Child2 created by constraining Agent {j} from {conflict_cell} at time {t}"
                # )

        elif conflict_type == "edge":
            cellA, cellB = details
            # print(
            #     f"Resolving edge conflict between Agent {i} and Agent {j} swapping {cellA} <-> {cellB} at time {t}"
            # )

            # Child 1: Constrain agent i from moving from cellA to cellB at time t
            child1_constraints = node.constraints.copy()
            child1_constraints.append((i, "edge", (cellA, cellB), t))
            child1_paths = node.paths.copy()
            path1 = a_star(grid, starts[i], goals[i], child1_constraints, i)
            if path1:
                child1_paths[i] = path1
                child1_cost = compute_cost(child1_paths)
                child1_node = CBSNode(child1_constraints, child1_paths, child1_cost)
                heapq.heappush(open_list, (child1_node.cost, child1_node))
                # print(
                #     f"Child1 created by constraining Agent {i} from moving {cellA} -> {cellB} at time {t}"
                # )

            # Child 2: Constrain agent j from moving from cellB to cellA at time t
            child2_constraints = node.constraints.copy()
            child2_constraints.append((j, "edge", (cellB, cellA), t))
            child2_paths = node.paths.copy()
            path2 = a_star(grid, starts[j], goals[j], child2_constraints, j)
            if path2:
                child2_paths[j] = path2
                child2_cost = compute_cost(child2_paths)
                child2_node = CBSNode(child2_constraints, child2_paths, child2_cost)
                heapq.heappush(open_list, (child2_node.cost, child2_node))
                # print(
                #     f"Child2 created by constraining Agent {j} from moving {cellB} -> {cellA} at time {t}"
                # )

    # If the open list is exhausted without finding a solution
    return None


###############################################################################


results = []
rng = np.random.default_rng(SEED)  # set seed for reproducibility

# block layout 2.1
# grid_list = [
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# ]

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

grid_list = [
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
    ],
    [
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
    ],
    [
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
    ],
    [
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ],
    [
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ],
    [
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ],
    [
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
    ],
    [
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
    ],
    [
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
    ],
    [
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
    ],
    [
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
    ],
    [
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
    ],
    [
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
    ],
    [
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
]

# TODO: use get_grid function to get the grid
grid = np.array(grid_list)

# Create a heatmap to track the number of visits to each cell
visit_counts = np.zeros_like(grid)

# make one additional episode, because RL looses one episode for some reason
for episode in range(NUM_EPISODES + 1):
    # Record the start time
    start_time = time.process_time()

    ########################################################################

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

    starts_dict = {i: starts[f"agent_{i}"] for i in range(NUM_AGENTS)}
    goals_dict = {i: goals[f"agent_{i}"] for i in range(NUM_AGENTS)}

    starts = [starts_dict[i] for i in range(NUM_AGENTS)]
    goals = [goals_dict[i] for i in range(NUM_AGENTS)]

    # Define the start and goal positions for multiple agents
    # blue green red pink
    # starts = [(5, 0), (3, 12), (6, 5), (6, 14)]
    # goals = [(6, 6), (9, 3), (6, 0), (3, 3)]

    # Define directions for movement (left, right, stay, up, down)
    paths = cbs(grid, starts, goals)

    # Record the end time
    end_time = time.process_time()

    # Calculate the CPU time
    cpu_time = end_time - start_time

    episode_data = {
        "episode": episode,
        "cpu_time": cpu_time,
        "seed": SEED,
    }
    # Print results
    if paths is not None:
        max_steps = 0  # To find the max number of steps among agents
        for agent_index, path in enumerate(paths):
            steps_to_goal = len(path)
            if steps_to_goal > max_steps:
                max_steps = steps_to_goal
            total_distance = sum(
                abs(path[i][0] - path[i - 1][0]) + abs(path[i][1] - path[i - 1][1])
                for i in range(1, steps_to_goal)
            )
            # print(f"Path for agent {agent_index + 1}: {path}")
            # print(f"Number of steps for agent {agent_index + 1}: {steps_to_goal}")
            # print(
            #     f"Total distance traveled by agent {agent_index + 1}: {total_distance}"
            # )
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

        for path in paths:
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
    f"CBS_{NUM_AGENTS}agents_{current_time}.csv",
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
    "experiments/results", f"CBS_{NUM_AGENTS}agents_{current_time}_heatmap.pdf"
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
        for i, path in enumerate(paths):
            if num < len(path):
                x, y = zip(*path[: num + 1])
                lines[i].set_data(x, y)
        return lines

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=max(len(path) for path in paths),
        # frames=max(len(path) for path in zip(*paths)),
        init_func=init,
        blit=True,
        repeat=False,
    )

    plt.show()
