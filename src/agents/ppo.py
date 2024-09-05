""" Proximal Policy Optimization (PPO) agent configuration. """

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec


def get_ppo_config(env_name, render_env: bool = False, env_config=None):
    """Get the PPO configuration."""
    config = (
        PPOConfig()
        .environment(env_name, render_env=render_env, env_config=env_config)
        .framework("torch")
        .training(
            num_sgd_iter=10,
            clip_param=0.2,
            lr=0.0003,
            gamma=0.99,
            lambda_=0.95,
            entropy_coeff=0.001,
            model={
                "fcnet_hiddens": [32, 32],
            },
        )
        .multi_agent(
            policies={"shared_policy": PolicySpec()},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
    )
    return config
