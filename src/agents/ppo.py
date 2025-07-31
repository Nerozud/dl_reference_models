"""Proximal Policy Optimization (PPO) agent configuration."""

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec

from models.action_mask_model import TorchActionMaskModel
from models.action_mask_model_single import TorchActionMaskModelSingle
from src.utils.wandb_video_callback import WandbVideoCallback

ModelCatalog.register_custom_model("action_mask_model", TorchActionMaskModel)
ModelCatalog.register_custom_model("action_mask_model_single", TorchActionMaskModelSingle)


def get_ppo_config(env_name, env_config=None):
    """Get the PPO configuration."""
    num_agents = env_config.get("num_agents", 2)
    
    # Video logging configuration
    video_config = env_config.get("video_logging", {})
    enable_video_logging = video_config.get("enabled", False)
    video_frequency = video_config.get("frequency", 50)
    
    # Create video callback if enabled
    callbacks = []
    if enable_video_logging:
        callbacks.append(WandbVideoCallback(
            video_frequency=video_frequency,
            max_episodes_per_iteration=video_config.get("max_episodes_per_iteration", 1),
            video_fps=video_config.get("fps", 5),
            max_frames_per_episode=video_config.get("max_frames_per_episode", 200)
        ))

    if env_config.get("training_execution_mode") == "CTE":
        config = (
            PPOConfig()  # single agent config, CTE
            .environment(env_name, render_env=env_config["render_env"], env_config=env_config)
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
                clip_param=0.2,
                # clip_param=tune.choice([0.05, 0.1, 0.2, 0.3]),
                lr=0.0001,
                # lr=tune.choice([0.0001, 0.0003, 0.0005, 0.001]),
                gamma=0.99,
                lambda_=0.95,
                entropy_coeff=0.01,
                # entropy_coeff=tune.choice([0, 0.001, 0.01]),
                model={
                    "custom_model": "action_mask_model_single",
                    "custom_model_config": {
                        "no_masking": False,
                    },
                    "fcnet_hiddens": [32, 32],
                },
            )
            .callbacks(callbacks_class=callbacks[0] if callbacks else None)
        )

    else:
        policies = {f"agent_{i}": PolicySpec() for i in range(num_agents)}
        policies["shared_policy"] = PolicySpec()

        config = (
            PPOConfig()  # multi agent config, CTDE or DTE
            .environment(env_name, render_env=env_config["render_env"], env_config=env_config)
            .framework("torch")
            .resources(num_gpus=1)
            .env_runners(
                num_env_runners=10, num_envs_per_env_runner=2, sample_timeout_s=300
            )  # increase num_envs_per_env_runner if render is false
            .training(
                train_batch_size=4000,
                sgd_minibatch_size=4000,
                # num_sgd_iter=7,
                num_sgd_iter=12,
                # num_sgd_iter=tune.randint(5, 15),
                # clip_param=0.1,
                clip_param=0.05,
                # clip_param=tune.choice([0.05, 0.1, 0.2, 0.3]),
                lr=0.001,
                # lr=0.0005,
                # lr=tune.choice([0.0001, 0.0003, 0.0005, 0.001]),
                gamma=0.99,
                lambda_=0.95,
                # entropy_coeff=0.01,
                entropy_coeff=0.001,
                # entropy_coeff=tune.choice([0, 0.001, 0.01]),
                model={
                    # "custom_model": "action_mask_model",
                    # "custom_model_config": {
                    #     "no_masking": False,
                    # },
                    "fcnet_hiddens": [64, 64],
                    "use_lstm": True,
                    "lstm_cell_size": 64,
                    "lstm_use_prev_action": True,
                    "lstm_use_prev_reward": True,
                },
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=lambda agent_id, *args, **kwargs: (
                    agent_id if env_config.get("training_execution_mode") == "DTE" else "shared_policy"
                ),
            )
            .callbacks(callbacks_class=callbacks[0] if callbacks else None)
        )

    return config
