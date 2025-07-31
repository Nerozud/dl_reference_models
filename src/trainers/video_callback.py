import numpy as np
import wandb
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class VideoCallback(DefaultCallbacks):
    """Callback that logs a rendered episode as a video to wandb."""

    def __init__(self, record_interval: int = 100):
        super().__init__()
        self.record_interval = record_interval

    def on_train_result(self, *, algorithm, result, **kwargs):
        if result["training_iteration"] % self.record_interval != 0:
            return

        env = algorithm.workers.local_worker().env
        frames = []
        obs, _ = env.reset()
        done = {"__all__": False}
        while not done["__all__"]:
            actions = {
                a: algorithm.compute_single_action(o, explore=False)
                for a, o in obs.items()
            }
            obs, _, done, _, _ = env.step(actions)
            frames.append(env.render(return_array=True))

        if not frames:  # Ensure frames is not empty
            return
        video = np.stack(frames)
        wandb.log({"episode_video": wandb.Video(video, fps=4, format="mp4")})

