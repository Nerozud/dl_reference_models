"""Deep Q-Networks (DQN) agent configuration."""

from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.policy.policy import PolicySpec

from src.trainers.callbacks import ReferenceModelCallbacks


def get_dqn_config(env_name, env_config=None):
    """Get the DQN configuration."""
    num_agents = env_config.get("num_agents", 2)

    if env_config.get("training_execution_mode") == "CTE":
        msg = "DQN does not support CTE in this project. Use CTDE/DTE or switch to PPO/IMPALA."
        raise ValueError(msg)

    model_config = {
        "fcnet_hiddens": [32, 32],
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
        DQNConfig()  # multi agent config, CTDE or DTE
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env_name, render_env=env_config["render_env"], env_config=env_config)
        .framework("torch")
        .resources(num_gpus=1)
        .env_runners(num_env_runners=8, sample_timeout_s=300)  # increase num_envs_per_env_runner if render is false
        .callbacks(callbacks_class=ReferenceModelCallbacks)
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(rl_module_specs=rl_module_specs),
        )
        .training(
            train_batch_size_per_learner=4000,
            lr=0.0003,
            gamma=0.99,
            replay_buffer_config={
                "type": "MultiAgentPrioritizedEpisodeReplayBuffer",
                "capacity": 50000,
                "alpha": 0.6,
                "beta": 0.4,
            },
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
    )
    config.model.update(model_config)

    return config
