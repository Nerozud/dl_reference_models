""" Deep Q-Networks (DQN) agent configuration. """

from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models import ModelCatalog

from models.action_mask_model import TorchActionMaskModel
from models.action_mask_model_single import TorchActionMaskModelSingle
from src.utils.wandb_video_callback import WandbVideoCallback

ModelCatalog.register_custom_model("action_mask_model", TorchActionMaskModel)
ModelCatalog.register_custom_model(
    "action_mask_model_single", TorchActionMaskModelSingle
)


def get_dqn_config(env_name, env_config=None):
    """Get the DQN configuration."""
    num_agents = env_config.get("num_agents", 2)
    
    # Video logging configuration
    video_config = env_config.get("video_logging", {})
    enable_video_logging = video_config.get("enabled", False)
    video_frequency = video_config.get("frequency", 50)
    
    # Create video callback if enabled
    callbacks_class = None
    if enable_video_logging:
        # Create a custom callback class with the configuration
        class ConfiguredWandbVideoCallback(WandbVideoCallback):
            def __init__(self):
                super().__init__(
                    video_frequency=video_frequency,
                    max_episodes_per_iteration=video_config.get("max_episodes_per_iteration", 1),
                    video_fps=video_config.get("fps", 5),
                    max_frames_per_episode=video_config.get("max_frames_per_episode", 200)
                )
        callbacks_class = ConfiguredWandbVideoCallback

    if env_config.get("training_execution_mode") == "CTE":
        config = (
            DQNConfig()  # single agent config, CTE
            .environment(
                env_name, render_env=env_config["render_env"], env_config=env_config
            )
            .framework("torch")
            .env_runners(
                num_env_runners=10, num_envs_per_env_runner=2, sample_timeout_s=300
            )  # increase num_envs_per_env_runner if render is false
            .training(
                train_batch_size=4000,
                lr=0.0003,
                gamma=0.99,
                model={
                    "custom_model": "action_mask_model_single",
                    "custom_model_config": {
                        "no_masking": False,
                        # "use_attention": True,
                        # "lstm_use_prev_reward": True,
                        # "lstm_use_prev_action": True,
                        "fcnet_hiddens": [64, 64],
                    },
                },
            )
            .callbacks(callbacks_class=callbacks_class)
        )

    else:
        policies = {f"agent_{i}": PolicySpec() for i in range(num_agents)}
        policies["shared_policy"] = PolicySpec()

        config = (
            DQNConfig()  # multi agent config, CTDE or DTE
            .environment(
                env_name, render_env=env_config["render_env"], env_config=env_config
            )
            .framework("torch")
            .env_runners(
                num_env_runners=10, num_envs_per_env_runner=2, sample_timeout_s=300
            )  # increase num_envs_per_env_runner if render is false
            .training(
                train_batch_size=4000,
                lr=0.0003,
                gamma=0.99,
                model={
                    "custom_model": "action_mask_model",
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
                policies=policies,
                policy_mapping_fn=lambda agent_id, *args, **kwargs: (
                    agent_id
                    if env_config.get("training_execution_mode") == "DTE"
                    else "shared_policy"
                ),
            )
            .callbacks(callbacks_class=callbacks_class)
        )

    return config
