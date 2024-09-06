"""Tuner module for training models with Ray Tune."""

import os
from ray import tune, air

from ray.air.integrations.wandb import WandbLoggerCallback


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
            trial_dirname_creator=lambda trial: f"{algo_name}-{env_name}-{trial.trial_id}",
            trial_name_creator=lambda trial: f"{algo_name}-{trial.trial_id}",
        ),
        run_config=air.RunConfig(
            storage_path=os.path.abspath("./experiments/trained_models"),
            # stop=MaximumIterationStopper(max_iter=100),
            callbacks=[
                WandbLoggerCallback(
                    project=env_name, dir=os.path.abspath("./experiments")
                )
            ],
        ),
    )
    tuner.fit()
