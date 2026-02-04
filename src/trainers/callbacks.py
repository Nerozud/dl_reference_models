"""RLlib callbacks for custom metrics."""

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class SuccessRateCallbacks(DefaultCallbacks):
    """Log rolling success rate (window=100) to RLlib metrics."""

    def on_episode_end(self, *, episode, metrics_logger, **_kwargs):
        # Success: terminated (goal reached) and not truncated (time limit).
        is_terminated = getattr(episode, "is_terminated", None)
        is_truncated = getattr(episode, "is_truncated", None)

        terminated = is_terminated() if callable(is_terminated) else is_terminated
        truncated = is_truncated() if callable(is_truncated) else is_truncated

        if terminated is None or truncated is None:
            # Fallback to state dict if available.
            state = episode.get_state() if hasattr(episode, "get_state") else {}
            terminated = state.get("is_terminated", False) if terminated is None else terminated
            truncated = state.get("is_truncated", False) if truncated is None else truncated

        success = 1.0 if terminated and not truncated else 0.0
        metrics_logger.log_value("success_rate", success, window=100)
