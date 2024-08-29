"""Tuner module for training models with Ray Tune."""

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
        run_config=air.RunConfig(
            # local_dir="./trained_models",
            # stop=MaximumIterationStopper(max_iter=100),
            callbacks=[WandbLoggerCallback(project=env_name)],
        ),
    )
    tuner.fit()
