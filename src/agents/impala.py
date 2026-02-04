"""Importance Weighted Actor-Learner Architecture (IMPALA) agent configuration."""

# from ray import tune
from ray.rllib.algorithms.impala import IMPALAConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.policy.policy import PolicySpec

from src.trainers.callbacks import SuccessRateCallbacks

def get_impala_config(env_name, env_config=None):
    """Get the IMPALA configuration."""
    num_agents = env_config.get("num_agents", 2)

    if env_config.get("training_execution_mode") == "CTE":
        model_config = {
            "fcnet_hiddens": [256, 256],
        }
        config = (
            IMPALAConfig()  # single agent config, CTE
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .environment(env_name, render_env=env_config["render_env"], env_config=env_config)
            .framework("torch")
            .resources(num_gpus=1)
            .learners(num_learners=0)
            .env_runners(
                num_env_runners=4, num_envs_per_env_runner=2, sample_timeout_s=300
            )  # increase num_envs_per_env_runner if render is false
            .callbacks(callbacks_class=SuccessRateCallbacks)
            .rl_module(
                rl_module_spec=RLModuleSpec(
                    model_config=model_config,
                )
            )
            .training(
                # train_batch_size=tune.choice([500, 1000, 4000]),
                train_batch_size_per_learner=4000,
                # lr=tune.choice([0.0001, 0.0003, 0.0005, 0.001]),
                lr=0.001,
                gamma=0.99,
                # vf_loss_coeff=tune.choice([0, 0.1, 0.3, 0.5, 0.9, 1]),
                vf_loss_coeff=0.3,
                # entropy_coeff=tune.choice([0, 0.001, 0.01]),
                entropy_coeff=0.001,
            )
        )
        config.model.update(model_config)

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
            IMPALAConfig()  # multi agent config, CTDE or DTE
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .environment(env_name, render_env=env_config["render_env"], env_config=env_config)
            .framework("torch")
            .resources(num_gpus=1)
            .learners(num_learners=0)
            .env_runners(
                num_env_runners=4, num_envs_per_env_runner=2, sample_timeout_s=300
            )  # increase num_envs_per_env_runner if render is false
            .callbacks(callbacks_class=SuccessRateCallbacks)
            .rl_module(
                rl_module_spec=MultiRLModuleSpec(rl_module_specs=rl_module_specs),
            )
            .training(
                # train_batch_size=tune.choice([500, 1000, 4000]),
                # train_batch_size=1000,
                train_batch_size_per_learner=8000,
                # lr=tune.choice([0.0001, 0.0003, 0.0005, 0.001]),
                lr=0.001,
                # lr=0.0001,
                gamma=0.99,
                # vf_loss_coeff=tune.choice([0, 0.1, 0.3, 0.5, 0.9, 1]),
                # vf_loss_coeff=0.1,
                vf_loss_coeff=0.5,
                # entropy_coeff=tune.choice([0, 0.001, 0.01]),
                # entropy_coeff=0.001,
                entropy_coeff=0,
            )
            .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        )
        config.model.update(model_config)

    return config
