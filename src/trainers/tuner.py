"""Tuner module for training models with Ray Tune."""

import os
from ray import tune, air

# from ray.tune.search.bayesopt import BayesOptSearch
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
            # search_alg=BayesOptSearch(
            #     utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0},
            #     metric="env_runners/episode_reward_mean",
            #     mode="max",
            # ),
            num_samples=-1,
            trial_dirname_creator=lambda trial: f"{algo_name}-{env_name}-{trial.trial_id}",
            trial_name_creator=lambda trial: f"{algo_name}-{trial.trial_id}",
            time_budget_s=3600,
        ),
        run_config=air.RunConfig(
            storage_path=os.path.abspath("./experiments/trained_models"),
            # stop=MaximumIterationStopper(max_iter=100),
            stop={"timesteps_total": 1e6},
            callbacks=[
                WandbLoggerCallback(
                    project=env_name, dir=os.path.abspath("./experiments")
                )
            ],
        ),
    )
    tuner.fit()
