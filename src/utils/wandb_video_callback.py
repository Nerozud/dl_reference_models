"""Wandb video logging callback for RLLib training."""

import os
from typing import Dict, Optional

import wandb
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

from src.utils.video_logger import VideoLogger


class WandbVideoCallback(DefaultCallbacks):
    """Callback that records and logs episode videos to wandb every x iterations."""

    def __init__(self, 
                 video_frequency: int = 50,
                 max_episodes_per_iteration: int = 1,
                 video_fps: int = 5,
                 max_frames_per_episode: int = 200):
        """
        Initialize the wandb video callback.

        Args:
            video_frequency: Log videos every N iterations
            max_episodes_per_iteration: Maximum number of episodes to record per iteration
            video_fps: Frames per second for recorded videos
            max_frames_per_episode: Maximum frames to record per episode
        """
        super().__init__()
        self.video_frequency = video_frequency
        self.max_episodes_per_iteration = max_episodes_per_iteration
        self.video_fps = video_fps
        self.max_frames_per_episode = max_frames_per_episode
        self.video_logger = VideoLogger(fps=video_fps, max_frames=max_frames_per_episode)
        self.episodes_recorded_this_iteration = 0
        self.should_record_videos = False

    def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
        """Called when the algorithm is initialized."""
        super().on_algorithm_init(algorithm=algorithm, **kwargs)
        print(f"WandbVideoCallback initialized - will record videos every {self.video_frequency} iterations")

    def on_train_result(self, *, algorithm: Algorithm, result: dict, **kwargs) -> None:
        """Called after each training iteration."""
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)
        
        # Check if we should record videos this iteration
        iteration = result.get("training_iteration", 0)
        self.should_record_videos = (iteration % self.video_frequency == 0)
        self.episodes_recorded_this_iteration = 0
        
        if self.should_record_videos:
            print(f"Will record videos during iteration {iteration}")

    def on_episode_start(self, *, worker: RolloutWorker, base_env, policies: Dict[PolicyID, Policy], 
                        episode: Episode, env_index: Optional[int] = None, **kwargs) -> None:
        """Called at the beginning of each episode."""
        super().on_episode_start(worker=worker, base_env=base_env, policies=policies, 
                                episode=episode, env_index=env_index, **kwargs)
        
        # Start recording if we should record videos and haven't exceeded the limit
        if (self.should_record_videos and 
            self.episodes_recorded_this_iteration < self.max_episodes_per_iteration):
            
            # Get the environment from base_env
            if hasattr(base_env, 'get_sub_environments'):
                env = base_env.get_sub_environments()[env_index or 0]
            else:
                env = base_env
            
            # Set up video logging for this environment
            if hasattr(env, 'capture_video'):
                env.capture_video = True
                env.video_logger = self.video_logger
                self.video_logger.start_recording()
                
                print(f"Started recording episode {self.episodes_recorded_this_iteration + 1} for video logging")

    def on_episode_end(self, *, worker: RolloutWorker, base_env, policies: Dict[PolicyID, Policy], 
                      episode: Episode, env_index: Optional[int] = None, **kwargs) -> None:
        """Called at the end of each episode."""
        super().on_episode_end(worker=worker, base_env=base_env, policies=policies, 
                              episode=episode, env_index=env_index, **kwargs)
        
        # If we were recording, finalize the video and log it
        if (self.should_record_videos and 
            self.episodes_recorded_this_iteration < self.max_episodes_per_iteration and
            self.video_logger.recording):
            
            # Get the environment from base_env
            if hasattr(base_env, 'get_sub_environments'):
                env = base_env.get_sub_environments()[env_index or 0]
            else:
                env = base_env
            
            # Stop recording and log the video
            self.video_logger.stop_recording()
            
            # Create a meaningful step number for wandb logging
            training_iteration = episode.get_infos().get("training_iteration", 0)
            step = training_iteration * 1000 + self.episodes_recorded_this_iteration
            
            # Log the video to wandb
            success = self.video_logger.record_and_log_episode(
                step=step, 
                key=f"episode_video/iteration_{training_iteration}_episode_{self.episodes_recorded_this_iteration}"
            )
            
            if success:
                print(f"Successfully logged video for episode {self.episodes_recorded_this_iteration + 1}")
            else:
                print(f"Failed to log video for episode {self.episodes_recorded_this_iteration + 1}")
            
            # Reset environment video settings
            if hasattr(env, 'capture_video'):
                env.capture_video = False
                env.video_logger = None
            
            self.episodes_recorded_this_iteration += 1