from ray import tune, air
from ray.air.integrations.wandb import WandbLoggerCallback


def tune_with_callback(config, algo_name):
    tuner = tune.Tuner(
        algo_name,
        param_space=config,
        run_config=air.RunConfig(
            # local_dir="./trained_models",
            # stop=MaximumIterationStopper(max_iter=100),
            callbacks=[WandbLoggerCallback(project="branching-problem")],
        ),
    )
    tuner.fit()
