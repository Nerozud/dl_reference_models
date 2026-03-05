"""W&B callback utilities for PB2/PBT-compatible step logging."""

from __future__ import annotations

import os
import urllib

from ray.air.integrations.wandb import (
    TRAINING_ITERATION,
    WandbLoggerCallback,
    _QueueItem,
    _WandbLoggingActor,
    _run_wandb_process_run_info_hook,
)
from ray import logger


class _PBTSafeWandbLoggingActor(_WandbLoggingActor):
    """W&B logging actor that allows out-of-order training_iteration values."""

    def run(self):
        # Since we're running in a separate process already, use threads.
        os.environ["WANDB_START_METHOD"] = "thread"
        run = self._wandb.init(*self.args, **self.kwargs)
        run.config.trial_log_path = self._logdir

        _run_wandb_process_run_info_hook(run)

        # Let W&B use `training_iteration` as x-axis while allowing out-of-order values.
        run.define_metric(TRAINING_ITERATION)
        run.define_metric("*", step_metric=TRAINING_ITERATION)

        while True:
            item_type, item_content = self.queue.get()
            if item_type == _QueueItem.END:
                break

            if item_type == _QueueItem.CHECKPOINT:
                self._handle_checkpoint(item_content)
                continue

            assert item_type == _QueueItem.RESULT
            log, config_update = self._handle_result(item_content)
            try:
                run.config.update(config_update, allow_val_change=True)
                run.log(log)
            except urllib.error.HTTPError as e:
                # Ignore HTTPError. Missing a few data points is not a
                # big issue, as long as things eventually recover.
                logger.warning("Failed to log result to w&b: {}".format(str(e)))
            except FileNotFoundError as e:
                logger.error(
                    "FileNotFoundError: Did not log result to Weights & Biases. "
                    "Possible cause: relative file path used instead of absolute path. "
                    "Error: %s",
                    e,
                )
        self._wandb.finish()


class PBTSafeWandbLoggerCallback(WandbLoggerCallback):
    """Wandb callback that uses an actor with PB2/PBT-safe step handling."""

    _logger_actor_cls = _PBTSafeWandbLoggingActor
