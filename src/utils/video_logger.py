"""Video logging utility for creating wandb videos from environment episodes."""

import io
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import wandb
from PIL import Image


class VideoLogger:
    """Handles video recording and logging to wandb for RL training visualization."""

    def __init__(self, fps: int = 5, max_frames: int = 200):
        """
        Initialize the video logger.

        Args:
            fps: Frames per second for the video
            max_frames: Maximum number of frames to record per episode
        """
        self.fps = fps
        self.max_frames = max_frames
        self.frames: List[np.ndarray] = []
        self.recording = False

    def start_recording(self):
        """Start recording a new episode."""
        self.frames = []
        self.recording = True

    def stop_recording(self):
        """Stop recording the current episode."""
        self.recording = False

    def capture_frame_from_matplotlib_figure(self, fig) -> Optional[np.ndarray]:
        """
        Capture a frame from a matplotlib figure.

        Args:
            fig: Matplotlib figure object

        Returns:
            RGB frame as numpy array, or None if capture fails
        """
        if not self.recording or len(self.frames) >= self.max_frames:
            return None

        try:
            # Save the figure to a BytesIO buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)

            # Convert to PIL Image and then to numpy array
            img = Image.open(buf)
            frame = np.array(img.convert('RGB'))
            buf.close()

            self.frames.append(frame)
            return frame

        except Exception as e:
            print(f"Warning: Failed to capture frame: {e}")
            return None

    def create_video_array(self) -> Optional[np.ndarray]:
        """
        Create a video array from captured frames.

        Returns:
            Video array in format (time, height, width, channels) or None if no frames
        """
        if not self.frames:
            return None

        # Stack all frames into a video array
        video_array = np.stack(self.frames, axis=0)
        return video_array

    def log_to_wandb(self, video_array: np.ndarray, step: int, key: str = "episode_video"):
        """
        Log video to wandb.

        Args:
            video_array: Video array in format (time, height, width, channels)
            step: Training step/iteration
            key: wandb log key for the video
        """
        try:
            # Create wandb video object
            video = wandb.Video(video_array, fps=self.fps, format="mp4")
            
            # Log to wandb
            wandb.log({key: video}, step=step)
            
        except Exception as e:
            print(f"Warning: Failed to log video to wandb: {e}")

    def record_and_log_episode(self, step: int, key: str = "episode_video") -> bool:
        """
        Create video from recorded frames and log to wandb.

        Args:
            step: Training step/iteration
            key: wandb log key for the video

        Returns:
            True if successful, False otherwise
        """
        if not self.frames:
            return False

        video_array = self.create_video_array()
        if video_array is None:
            return False

        self.log_to_wandb(video_array, step, key)
        
        # Clear frames after logging
        self.frames = []
        
        return True

    def clear_frames(self):
        """Clear all recorded frames."""
        self.frames = []
        self.recording = False