"""Tuner module for training models with Ray Tune."""

from pathlib import Path

from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2

from src.trainers.resettable_rllib_trainable import make_resettable_rllib_trainable

pb2_scheduler = PB2(
    time_attr="training_iteration",
    metric="env_runners/episode_return_mean",
    mode="max",
    perturbation_interval=10,
    hyperparam_bounds={
        "lr": [1e-5, 1e-3],
        "entropy_coeff": [0.0, 0.015],
        "num_epochs": [1, 20],
        "clip_param": [0.05, 0.5],
        "gamma": [0.8, 0.999],
        "lambda_": [0.8, 0.99],
    },
)


def _is_pbt_like_scheduler(scheduler) -> bool:
    return isinstance(scheduler, (PopulationBasedTraining, PB2))


def _get_env_config_value(config, key):
    env_config = config["env_config"] if isinstance(config, dict) else config.env_config
    return env_config[key]


def tune_with_callback(config, algo_name, env_name):
    """
    Tune the model using a wandb callback.

    Args:
        config (dict): The configuration parameters for the tuner.
        algo_name (str): The name of the algorithm.
        env_name (str): The name of the environment.

    Returns:
        None

    """
    scheduler = None
    # scheduler = pb2_scheduler

    use_actor_reuse = _is_pbt_like_scheduler(scheduler)
    trainable = algo_name
    param_space = config

    if use_actor_reuse:
        trainable = make_resettable_rllib_trainable(
            algo_name,
            checkpoint_restore_mode="weights_only",
        )
        param_space = config.to_dict() if hasattr(config, "to_dict") else config

    tuner = tune.Tuner(
        trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            reuse_actors=use_actor_reuse,
            num_samples=1,
            trial_dirname_creator=lambda trial: f"{algo_name}-{env_name}-{trial.trial_id}",
            trial_name_creator=lambda trial: (
                f"{algo_name}-{_get_env_config_value(config, 'training_execution_mode')}-{trial.trial_id}"
            ),
            time_budget_s=3600 * 23.5,
        ),
        run_config=tune.RunConfig(
            storage_path=Path("./experiments/trained_models").resolve(),
            stop={
                # "env_runners/episode_return_mean": 1.5 * _get_env_config_value(config, "num_agents"),
                # "time_total_s": 3600 * 23.5,
                # "counters/num_env_steps_sampled": 14360000,
                "env_runners/success_rate": 1,
            },
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_score_attribute="env_runners/episode_return_mean",
                checkpoint_score_order="max",
                num_to_keep=5,
            ),
            callbacks=[
                WandbLoggerCallback(
                    project=f"{env_name}-test",
                    dir=Path("./experiments").resolve(),
                    group=f"{algo_name}-{config['env_config']['training_execution_mode']}",
                )
            ],
        ),
    )
    tuner.fit()
