"""RLlib callbacks for custom metrics."""

import contextlib
from typing import ClassVar

from ray.rllib.callbacks.callbacks import RLlibCallback


def _coerce_done(value):
    if isinstance(value, dict):
        if "__all__" in value:
            return bool(value.get("__all__", False))
        if len(value) == 1:
            return bool(next(iter(value.values())))
        return False
    if value is None:
        return None
    return bool(value)


def _unwrap_env(env, max_depth=5):
    candidate = env
    depth = 0
    while candidate is not None and depth < max_depth:
        unwrapped = getattr(candidate, "unwrapped", None)
        if unwrapped is None or unwrapped is candidate:
            break
        candidate = unwrapped
        depth += 1
    return candidate


def _select_sub_env(env, env_index):
    if env is None:
        return None
    if env_index is None:
        env_index = 0
    if hasattr(env, "get_sub_environments"):
        try:
            sub_envs = env.get_sub_environments()
        except TypeError:
            sub_envs = None
        if sub_envs:
            if 0 <= env_index < len(sub_envs):
                return sub_envs[env_index]
            return sub_envs[0]
    if hasattr(env, "envs"):
        sub_envs = env.envs
        if sub_envs:
            if 0 <= env_index < len(sub_envs):
                return sub_envs[env_index]
            return sub_envs[0]
    return env


def _resolve_env(env, env_runner, env_index):
    candidates = []
    if env is not None:
        candidates.append(env)
    if env_runner is not None:
        for name in ("env", "_env", "vector_env", "envs", "_envs"):
            candidate = getattr(env_runner, name, None)
            if candidate is not None:
                candidates.append(candidate)
    for candidate in candidates:
        sub_env = _select_sub_env(candidate, env_index)
        sub_env = _unwrap_env(sub_env)
        if sub_env is not None:
            return sub_env
    return None


def _iter_infos(episode):
    infos = None
    if hasattr(episode, "get_last_infos"):
        try:
            infos = episode.get_last_infos()
        except TypeError:
            infos = None
    if infos is None and hasattr(episode, "last_info_for"):
        try:
            infos = episode.last_info_for()
        except TypeError:
            infos = None
    if infos is None:
        return []
    if isinstance(infos, dict):
        if "__all__" in infos and isinstance(infos.get("__all__"), dict):
            ordered = [infos["__all__"]]
            ordered.extend([v for k, v in infos.items() if k != "__all__" and isinstance(v, dict)])
            return ordered
        if all(isinstance(v, dict) for v in infos.values()):
            return list(infos.values())
        return [infos]
    if isinstance(infos, (list, tuple)):
        return [info for info in infos if isinstance(info, dict)]
    return []


def _sum_metric(infos, keys):
    total = 0.0
    for info in infos:
        for key in keys:
            if key in info and info[key] is not None:
                with contextlib.suppress(TypeError, ValueError):
                    total += float(info[key])
                break
    return total


class SuccessRateCallback(RLlibCallback):
    """Log rolling success rate (window=100) to RLlib metrics."""

    def on_episode_end(self, *, episode, metrics_logger, **_kwargs):
        # Success: terminated (goal reached) and not truncated (time limit).
        is_terminated = getattr(episode, "is_terminated", None)
        is_truncated = getattr(episode, "is_truncated", None)

        terminated = is_terminated() if callable(is_terminated) else is_terminated
        truncated = is_truncated() if callable(is_truncated) else is_truncated

        terminated = _coerce_done(terminated)
        truncated = _coerce_done(truncated)

        if terminated is None or truncated is None:
            state = episode.get_state() if hasattr(episode, "get_state") else {}
            if terminated is None:
                terminated = _coerce_done(state.get("is_terminated", False))
            if truncated is None:
                truncated = _coerce_done(state.get("is_truncated", False))

        success = 1.0 if terminated and not truncated else 0.0
        metrics_logger.log_value("success_rate", success, reduce="mean", window=100)


class EpisodeMetricsCallback(RLlibCallback):
    """Accumulate and log goals/blocking metrics per episode."""

    def on_episode_start(
        self,
        *,
        episode,
        **_kwargs,
    ) -> None:
        if not hasattr(episode, "custom_data") or episode.custom_data is None:
            episode.custom_data = {}
        episode.custom_data["goals_reached"] = 0.0
        episode.custom_data["blocking_count"] = 0.0

    def on_episode_step(
        self,
        *,
        episode,
        **_kwargs,
    ) -> None:
        if not hasattr(episode, "custom_data") or episode.custom_data is None:
            episode.custom_data = {"goals_reached": 0.0, "blocking_count": 0.0}
        infos = _iter_infos(episode)
        global_info = next(
            (
                info
                for info in infos
                if "goals_reached_step" in info or "blocking_count_step" in info
            ),
            None,
        )
        if global_info is not None:
            with contextlib.suppress(TypeError, ValueError):
                episode.custom_data["goals_reached"] += float(global_info.get("goals_reached_step", 0.0))
            with contextlib.suppress(TypeError, ValueError):
                episode.custom_data["blocking_count"] += float(global_info.get("blocking_count_step", 0.0))
            return

        episode.custom_data["goals_reached"] += _sum_metric(infos, ["goal_reached_step", "goals_reached_step"])
        episode.custom_data["blocking_count"] += _sum_metric(infos, ["blocking", "blocking_count_step"])

    def on_episode_end(
        self,
        *,
        episode,
        env_runner,
        metrics_logger,
        env,
        env_index,
        **_kwargs,
    ) -> None:
        if not hasattr(episode, "custom_data") or episode.custom_data is None:
            episode.custom_data = {"goals_reached": 0.0, "blocking_count": 0.0}

        env_unwrapped = _resolve_env(env, env_runner, env_index)
        goals_total = None
        blocking_total = None
        if env_unwrapped is not None:
            if hasattr(env_unwrapped, "goal_reached_once"):
                values = env_unwrapped.goal_reached_once
                if isinstance(values, dict):
                    goals_total = float(sum(1 for v in values.values() if v))
                elif isinstance(values, (list, tuple, set)):
                    goals_total = float(sum(1 for v in values if v))
            if hasattr(env_unwrapped, "_episode_blocking_count"):
                try:
                    blocking_total = float(env_unwrapped._episode_blocking_count)
                except (TypeError, ValueError):
                    blocking_total = None

        goals_value = goals_total if goals_total is not None else float(episode.custom_data.get("goals_reached", 0.0))
        blocking_value = (
            blocking_total if blocking_total is not None else float(episode.custom_data.get("blocking_count", 0.0))
        )

        metrics_logger.log_value("goals_reached", goals_value, reduce="mean", window=100)
        metrics_logger.log_value("blocking_count", blocking_value, reduce="mean", window=100)


class ReferenceModelCallbacks(RLlibCallback):
    """Composite callbacks for ReferenceModel metrics."""

    callback_classes: ClassVar[list[type[RLlibCallback]]] = [SuccessRateCallback, EpisodeMetricsCallback]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._callbacks = [cb_class() for cb_class in self.callback_classes]

    def on_episode_start(self, *, episode, **kwargs):
        for callback in self._callbacks:
            if hasattr(callback, "on_episode_start"):
                callback.on_episode_start(episode=episode, **kwargs)

    def on_episode_step(self, *, episode, **kwargs):
        for callback in self._callbacks:
            if hasattr(callback, "on_episode_step"):
                callback.on_episode_step(episode=episode, **kwargs)

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        for callback in self._callbacks:
            if hasattr(callback, "on_episode_end"):
                callback.on_episode_end(episode=episode, metrics_logger=metrics_logger, **kwargs)


# Deprecated alias for backward compatibility.
SuccessRateCallbacks = ReferenceModelCallbacks
