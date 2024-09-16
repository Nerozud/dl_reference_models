"""Get the grid of the environment and the start and goal positions of the agents."""

import numpy as np


def get_grid(env_name: str):
    """Get the grid of the environment.

    Args:
        env_name (str): The name of the environment.

    """

    grids = {
        "ReferenceModel-1-1": np.array(
            [
                [1, 1, 1, 1, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
        "ReferenceModel-1-2": np.array(
            [
                [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            ],
            dtype=np.uint8,
        ),
    }

    try:
        return grids[env_name]
    except KeyError as exc:
        raise ValueError(f"Unknown environment name: {env_name}") from exc


def get_start_positions(env_name: str, num_agents: int):
    """
    Get the start positions of the agents.

    Args:
        env_name (str): The name of the environment.
        num_agents (int): The number of agents.

    Returns:
        dict: Keys are agent names (e.g., 'agent_0') and values are the start positions (tuples).

    Raises:
        ValueError: If the environment name is unknown or if the number of agents exceeds available positions.
    """

    start_positions = {
        "ReferenceModel-1-1": {
            0: (1, 1),
            1: (1, 7),
        },
        "ReferenceModel-1-2": {
            0: (1, 1),
            1: (1, 8),
        },
    }

    # Check if environment name exists in start positions
    if env_name not in start_positions:
        raise ValueError(f"Unknown environment name: {env_name}")

    env_start_positions = start_positions[env_name]

    # Ensure the number of agents does not exceed available positions
    if num_agents > len(env_start_positions):
        raise ValueError(
            f"Requested number of agents ({num_agents}) exceeds available positions in {env_name}"
        )

    # Generate the dictionary with dynamic agent names
    return {f"agent_{i}": env_start_positions[i] for i in range(num_agents)}


def get_goal_positions(env_name: str, num_agents: int):
    """
    Get the goal positions of the agents.

    Args:
        env_name (str): The name of the environment.
        num_agents (int): The number of agents.

    Returns:
        dict: Keys are agent names (e.g., 'agent_0') and values are the goal positions (tuples).

    Raises:
        ValueError: If the environment name is unknown or if the number of agents exceeds available goal positions.
    """

    goal_positions = {
        "ReferenceModel-1-1": {
            0: (1, 8),
            1: (1, 0),
        },
        "ReferenceModel-1-2": {
            0: (1, 9),
            1: (1, 0),
        },
    }

    # Check if environment name exists in goal positions
    if env_name not in goal_positions:
        raise ValueError(f"Unknown environment name: {env_name}")

    env_goal_positions = goal_positions[env_name]

    # Ensure the number of agents does not exceed available goal positions
    if num_agents > len(env_goal_positions):
        raise ValueError(
            f"Requested number of agents ({num_agents}) exceeds available goal positions in {env_name}"
        )

    # Generate the dictionary with dynamic agent names
    return {f"agent_{i}": env_goal_positions[i] for i in range(num_agents)}
