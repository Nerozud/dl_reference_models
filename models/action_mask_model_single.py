import numpy as np
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
from torch import nn


class TorchActionMaskModelSingle(TorchModelV2, nn.Module):
    """Action masking model for flat observations with an appended mask (single-agent view)."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        if len(orig_space.shape) != 1:
            msg = "Expected flat Box observation with an appended action mask."
            raise ValueError(msg)

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)
        nn.Module.__init__(self)

        total_obs_dim = int(orig_space.shape[0])
        mask_dim = int(np.sum(action_space.nvec)) if hasattr(action_space, "nvec") else int(action_space.n)
        self._feature_dim = total_obs_dim - mask_dim
        if self._feature_dim <= 0:
            msg = "Observation vector must contain features before the action mask section."
            raise ValueError(msg)
        self._mask_dim = mask_dim

        hidden_layers = model_config.get("fcnet_hiddens", [256, 256])
        layers = []
        last_size = self._feature_dim
        for size in hidden_layers:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU())
            last_size = size
        self._hidden_layers = nn.Sequential(*layers) if layers else nn.Identity()
        self._logits = nn.Linear(last_size, num_outputs)
        self._value_branch = nn.Linear(last_size, 1)
        self._last_features = None

        # Option to disable action masking
        self.no_masking = model_config.get("custom_model_config", {}).get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        flat_obs = input_dict["obs"].float()
        features = flat_obs[..., : self._feature_dim]
        action_mask = flat_obs[..., self._feature_dim : self._feature_dim + self._mask_dim]

        self._last_features = self._hidden_layers(features)
        logits = self._logits(self._last_features)

        if self.no_masking:
            return logits, state

        inf_mask = torch.clamp(torch.log(action_mask + 1e-6), min=FLOAT_MIN)
        masked_logits = logits + inf_mask
        return masked_logits, state

    def value_function(self):
        return self._value_branch(self._last_features).squeeze(1)
