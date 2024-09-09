""" Proximal Policy Optimization (PPO) agent configuration. """

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models import ModelCatalog

from models.action_mask_model import TorchActionMaskModel

ModelCatalog.register_custom_model("my_action_mask_model", TorchActionMaskModel)


def get_ppo_config(env_name, env_config=None):
    """Get the PPO configuration."""
    config = (
        PPOConfig()
        .environment(
            env_name, render_env=env_config["render_env"], env_config=env_config
        )
        .framework("torch")
        .env_runners(num_env_runners=8, num_envs_per_env_runner=4)
        .training(
            train_batch_size=4000,
            sgd_minibatch_size=4000,
            num_sgd_iter=10,
            clip_param=0.2,
            lr=0.0003,
            gamma=0.99,
            lambda_=0.95,
            entropy_coeff=0.01,
            model={
                "custom_model": "my_action_mask_model",
                "custom_model_config": {
                    "no_masking": False,
                    # "use_attention": True,
                    # "lstm_use_prev_reward": True,
                    # "lstm_use_prev_action": True,
                    "fcnet_hiddens": [32, 32],
                },
                "fcnet_hiddens": [32, 32],
            },
        )
        .multi_agent(
            policies={"shared_policy": PolicySpec()},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
    )
    return config
