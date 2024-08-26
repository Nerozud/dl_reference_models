from ray.rllib.algorithms.ppo import PPOConfig


def get_ppo_config():
    """Get the PPO configuration."""
    config = PPOConfig().training(
        num_sgd_iter=10,
        clip_param=0.2,
        lr=0.0003,
        gamma=0.99,
        lambda_=0.95,
        entropy_coeff=0.001,
        model={
            "fcnet_hiddens": [64, 64],
        },
    )
    return config
