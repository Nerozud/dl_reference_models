"""Tuner module for training models with Ray Tune."""

from pathlib import Path

from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers.pb2 import PB2
pb2_scheduler = PB2(
    time_attr="time_total_s",
    metric="env_runners/episode_return_mean",
    mode="max",
    perturbation_interval=1800,
    hyperparam_bounds={
        "lr": [1e-5, 1e-3],
        "entropy_coeff": [0.0, 0.01],
        "num_epochs": [1, 15],
        "clip_param": [0.05, 0.3],
        "gamma": [0.8, 0.999],
    },
)


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
    tuner = tune.Tuner(
        algo_name,
        param_space=config,
        tune_config=tune.TuneConfig(
            # scheduler=pb2_scheduler,
            num_samples=1,
            trial_dirname_creator=lambda trial: f"{algo_name}-{env_name}-{trial.trial_id}",
            trial_name_creator=lambda trial: f"{algo_name}-{config['env_config']['training_execution_mode']}-{trial.trial_id}",
            # time_budget_s=3600 * 4,
        ),
        run_config=air.RunConfig(
            storage_path=Path("./experiments/trained_models").resolve(),
            # stop=MaximumIterationStopper(max_iter=100),
            # stop={"timesteps_total": 1e6 * 2},
            # stop={"env_runners/episode_reward_mean": 3, "timesteps_total": 1e6 / 2},
            stop={
                "env_runners/episode_return_mean": 1.5 * config["env_config"]["num_agents"],
                # "time_total_s": 3600 * 18,
                # "counters/num_env_steps_sampled": 14360000,
            },
            callbacks=[
                WandbLoggerCallback(
                    # project=env_name,
                    project=f"{env_name}-test",
                    dir=Path("./experiments").resolve(),
                    group=f"{algo_name}-{config['env_config']['training_execution_mode']}",
                )
            ],
        ),
    )
    tuner.fit()
