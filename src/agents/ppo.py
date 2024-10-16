""" Proximal Policy Optimization (PPO) agent configuration. """

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models import ModelCatalog
from ray import tune
from models.action_mask_model import TorchActionMaskModel
from models.action_mask_model_single import TorchActionMaskModelSingle

ModelCatalog.register_custom_model("action_mask_model", TorchActionMaskModel)
ModelCatalog.register_custom_model(
    "action_mask_model_single", TorchActionMaskModelSingle
)


def get_ppo_config(env_name, env_config=None):
    """Get the PPO configuration."""
    num_agents = env_config.get("num_agents", 2)

    if env_config.get("training_execution_mode") == "CTE":
        config = (
            PPOConfig()  # single agent config, CTE
            .environment(
                env_name, render_env=env_config["render_env"], env_config=env_config
            )
            .framework("torch")
            .resources(num_gpus=1)
            .env_runners(
                num_env_runners=10, num_envs_per_env_runner=2, sample_timeout_s=300
            )  # increase num_envs_per_env_runner if render is false
            .training(
                train_batch_size=4000,
                sgd_minibatch_size=4000,
                num_sgd_iter=10,
                # num_sgd_iter=tune.randint(5, 15),
                clip_param=0.3,
                # clip_param=tune.choice([0.05, 0.1, 0.2, 0.3]),
                lr=0.0005,
                # lr=tune.choice([0.0001, 0.0003, 0.0005, 0.001]),
                gamma=0.99,
                lambda_=0.95,
                entropy_coeff=0.001,
                # entropy_coeff=tune.choice([0, 0.001, 0.01]),
                model={
                    "custom_model": "action_mask_model_single",
                    "custom_model_config": {
                        "no_masking": False,
                        # "use_attention": True,
                        # "lstm_use_prev_reward": True,
                        # "lstm_use_prev_action": True,
                        "fcnet_hiddens": [256, 256],
                        # "fcnet_hiddens": tune.choice([[32, 32], [64, 64], [256, 256]]),
                    },
                },
            )
        )

    else:
        policies = {f"agent_{i}": PolicySpec() for i in range(num_agents)}
        policies["shared_policy"] = PolicySpec()

        config = (
            PPOConfig()  # multi agent config, CTDE or DTE
            .environment(
                env_name, render_env=env_config["render_env"], env_config=env_config
            )
            .framework("torch")
            .resources(num_gpus=1)
            .env_runners(
                num_env_runners=10, num_envs_per_env_runner=2, sample_timeout_s=300
            )  # increase num_envs_per_env_runner if render is false
            .training(
                train_batch_size=4000,
                sgd_minibatch_size=4000,
                num_sgd_iter=12,
                # num_sgd_iter=6,
                # num_sgd_iter=tune.randint(5, 15),
                clip_param=0.05,
                # clip_param=0.1,
                # clip_param=tune.choice([0.05, 0.1, 0.2, 0.3]),
                lr=0.001,
                # lr=0.0005,
                # lr=tune.choice([0.0001, 0.0003, 0.0005, 0.001]),
                gamma=0.99,
                lambda_=0.95,
                entropy_coeff=0.001,
                # entropy_coeff=0.01,
                # entropy_coeff=tune.choice([0, 0.001, 0.01]),
                model={
                    "custom_model": "action_mask_model",
                    "custom_model_config": {
                        "no_masking": False,
                        # "use_attention": True,
                        # "lstm_use_prev_reward": True,
                        # "lstm_use_prev_action": True,
                        "fcnet_hiddens": [32, 32],
                        # "fcnet_hiddens": [256, 256],
                        # "fcnet_hiddens": tune.choice([[32, 32], [64, 64], [256, 256]]),
                    },
                },
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=lambda agent_id, *args, **kwargs: (
                    agent_id
                    if env_config.get("training_execution_mode") == "DTE"
                    else "shared_policy"
                ),
            )
        )

    return config
