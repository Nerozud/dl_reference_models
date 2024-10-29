import torch
from torch import nn

from gymnasium.spaces import Dict
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork
from ray.rllib.utils.torch_utils import FLOAT_MIN


class TorchActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

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

        assert (
            isinstance(orig_space, Dict)
            and "observations" in orig_space.spaces
            and "position" in orig_space.spaces
            and "goal" in orig_space.spaces
        )

        relevant_obs_space = Dict(
            {
                "observations": orig_space.spaces["observations"],
                "position": orig_space.spaces["position"],
                "goal": orig_space.spaces["goal"],
            }
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        self.internal_model = ComplexInputNetwork(
            relevant_obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # Option to disable action masking
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
        observations = {
            "observations": input_dict["obs"]["observations"],
            "position": input_dict["obs"]["position"],
            "goal": input_dict["obs"]["goal"],
        }

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": observations})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()
