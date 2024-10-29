""" Generate a number of given start and goal sets for the environment. """

import numpy as np

from src.environments.get_grid import get_grid


def generate_starts_goals(env_name: str, num_sets: int, num_agents: int):
    """Generate a number of given start and goal sets for the environment."""
    get_grid(env_name)
    start_goal_sets = []

    return start_goal_sets
