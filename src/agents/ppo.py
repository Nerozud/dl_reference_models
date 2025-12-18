"""Proximal Policy Optimization (PPO) agent configuration."""

# from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

# from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec

# from models.action_mask_model import TorchActionMaskModel
# from models.action_mask_model_single import TorchActionMaskModelSingle

# ModelCatalog.register_custom_model("action_mask_model", TorchActionMaskModel)
# ModelCatalog.register_custom_model("action_mask_model_single", TorchActionMaskModelSingle)

# check https://docs.ray.io/en/latest/rllib/rl-modules.html#other-default-model-settings


def get_ppo_config(env_name, env_config=None):
    """Get the PPO configuration."""
    num_agents = env_config.get("num_agents", 2)

    if env_config.get("training_execution_mode") == "CTE":
        config = (
            PPOConfig()  # single agent config, CTE
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .environment(env_name, render_env=env_config["render_env"], env_config=env_config)
            .framework("torch")
            .resources(num_gpus=1)
            .env_runners(num_env_runners=8, sample_timeout_s=300)
            .rl_module(
                rl_module_spec=RLModuleSpec(
                    model_config={
                        "fcnet_hiddens": [64, 64],
                        "use_lstm": True,
                        "lstm_cell_size": 64,
                        "lstm_use_prev_action": True,
                        "lstm_use_prev_reward": True,
                        "max_seq_len": 32,
                        "vf_share_layers": True,
                    }
                )
            )
            .training(
                train_batch_size_per_learner=4000,
                minibatch_size=4000,
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
            )
        )

    else:
        model_config = {
            "fcnet_hiddens": [64, 64],
            "use_lstm": True,
            "lstm_cell_size": 64,
            "lstm_use_prev_action": True,
            "lstm_use_prev_reward": True,
            "max_seq_len": 32,
        }

        if env_config.get("training_execution_mode") == "DTE":
            policies = {f"agent_{i}": PolicySpec() for i in range(num_agents)}
            rl_module_specs = {pid: RLModuleSpec(model_config=model_config) for pid in policies}

            def policy_mapping_fn(agent_id, *_args, **_kwargs):
                return agent_id
        else:
            policies = {"shared_policy": PolicySpec()}
            rl_module_specs = {"shared_policy": RLModuleSpec(model_config=model_config)}

            def policy_mapping_fn(*_args, **_kwargs):
                return "shared_policy"

        config = (
            PPOConfig()  # multi agent config, CTDE or DTE
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .environment(env_name, render_env=env_config["render_env"], env_config=env_config)
            .framework("torch")
            .resources(num_gpus=1)
            .env_runners(num_env_runners=8, sample_timeout_s=300)
            .rl_module(
                rl_module_spec=MultiRLModuleSpec(rl_module_specs=rl_module_specs),
            )
            .training(
                train_batch_size_per_learner=4000,
                minibatch_size=4000,
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
            )
            .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        )

    return config
