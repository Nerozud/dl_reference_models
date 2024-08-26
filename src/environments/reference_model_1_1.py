from ray.rllib.env.multi_agent_env import MultiAgentEnv


class ReferenceModel(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed, options):
        return obs, info

    def step(self, action):
        return obs, rewards, done, truncated, info
