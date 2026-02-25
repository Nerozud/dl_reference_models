"""Resettable Ray Tune trainable wrapper for RLlib algorithms."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

from ray.tune.trainable import Trainable

LOGGER = logging.getLogger(__name__)
_RL_MODULE_CHECKPOINT_COMPONENT = str(PurePosixPath("learner_group") / "learner" / "rl_module")
_SUPPORTED_CHECKPOINT_RESTORE_MODES = {"full", "weights_only"}


def _resolve_algo_cls(algo_name: str):
    """Resolve a Tune trainable string to the registered RLlib algorithm class."""
    from ray.tune.registry import get_trainable_cls

    algo_cls = get_trainable_cls(algo_name)
    if not issubclass(algo_cls, Trainable):
        msg = f"Resolved trainable {algo_name!r} is not a Tune Trainable subclass."
        raise TypeError(msg)
    return algo_cls


def make_resettable_rllib_trainable(
    algo_name: str,
    *,
    checkpoint_restore_mode: str = "full",
) -> type[Trainable]:
    """Create a Tune Trainable wrapper with a working reset_config() implementation."""
    if checkpoint_restore_mode not in _SUPPORTED_CHECKPOINT_RESTORE_MODES:
        msg = (
            "Unsupported checkpoint_restore_mode "
            f"{checkpoint_restore_mode!r}. Expected one of "
            f"{sorted(_SUPPORTED_CHECKPOINT_RESTORE_MODES)!r}."
        )
        raise ValueError(msg)

    algo_cls = _resolve_algo_cls(algo_name)
    wrapper_name = f"Resettable{algo_name}Trainable"

    class ResettableRLlibTrainable(Trainable):
        _algo_cls = algo_cls
        _checkpoint_restore_mode = checkpoint_restore_mode

        @classmethod
        def default_resource_request(cls, config: dict[str, Any]):
            return cls._algo_cls.default_resource_request(config)

        @classmethod
        def resource_help(cls, config: dict[str, Any]):
            if hasattr(cls._algo_cls, "resource_help"):
                return cls._algo_cls.resource_help(config)
            return super().resource_help(config)

        def setup(self, config: dict[str, Any]) -> None:
            self._algo = None
            self._current_config = None
            self._build_algo(config)

        def step(self) -> dict[str, Any]:
            result = self._algo.train().copy()
            # Let Tune inject a monotonic wrapper-level training_iteration.
            result.pop("training_iteration", None)
            return result

        def save_checkpoint(self, checkpoint_dir: str):
            return self._algo.save_to_path(checkpoint_dir)

        def load_checkpoint(self, checkpoint):
            if self._checkpoint_restore_mode == "weights_only":
                rl_module_checkpoint = Path(checkpoint) / _RL_MODULE_CHECKPOINT_COMPONENT
                LOGGER.info(
                    "Restoring %s from checkpoint with weights-only mode "
                    "(component=%s, path=%s) to avoid optimizer-state incompatibilities.",
                    self._algo_cls.__name__,
                    _RL_MODULE_CHECKPOINT_COMPONENT,
                    rl_module_checkpoint,
                )
                self._algo.restore_from_path(
                    rl_module_checkpoint,
                    component=_RL_MODULE_CHECKPOINT_COMPONENT,
                )
                return

            self._algo.restore_from_path(checkpoint)

        def cleanup(self) -> None:
            self._stop_algo()

        def reset_config(self, new_config: dict[str, Any]) -> bool:
            try:
                self._stop_algo()
                self._build_algo(new_config)
            except Exception:
                LOGGER.exception("Failed to reset %s with updated config.", self._algo_cls.__name__)
                return False
            return True

        def _build_algo(self, config: dict[str, Any]) -> None:
            self._current_config = deepcopy(config)
            self._algo = self._algo_cls(config=config)

        def _stop_algo(self) -> None:
            algo = getattr(self, "_algo", None)
            if algo is None:
                return

            try:
                algo.stop()
            except Exception:
                LOGGER.exception("Ignoring error while stopping %s during cleanup/reset.", self._algo_cls.__name__)
            finally:
                self._algo = None

    ResettableRLlibTrainable.__name__ = wrapper_name
    ResettableRLlibTrainable.__qualname__ = wrapper_name
    return ResettableRLlibTrainable
