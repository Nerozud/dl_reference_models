from ray.rllib.env.multi_agent_env import MultiAgentEnv

import numpy as np

import gymnasium as gym


class ReferenceModel(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()

        self.step_count = 0
        self.steps_per_episode = 100
        self.num_agents = env_config.get("num_agents", 3)

        # TODO: Implement the environment initialization depending on the env_config
        self.grid = np.array(
            [
                [1, 1, 1, 1, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        # Define the start and goal positions for multiple agents
        # TODO: randomize start and goals and flexible agent numbers
        self.starts = [(1, 0), (2, 12), (0, 0)]
        self.goals = [(2, 12), (0, 0), (4, 12)]

        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(2, 9),
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, seed, options):
        self.step_count = 0

        return obs, info

    def step(self, action):
        return obs, rewards, done, truncated, info
