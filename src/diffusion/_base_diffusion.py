from __future__ import annotations

from typing import Any

import torch

from src.models._base_model import BaseModel


class BaseDiffusion(BaseModel):
    def __init__(
        self, model: BaseModel, timesteps: int, sampling_timesteps: int = None, sampling_schedule=None, **kwargs
    ):
        if model is None:
            raise ValueError(
                "Arg ``model`` is missing. Please provide a backbone model for the diffusion model (e.g. a Unet)"
            )
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.spatial_shape = model.spatial_shape
        self.num_input_channels = model.num_input_channels
        self.num_output_channels = model.num_output_channels
        self.num_conditional_channels = model.num_conditional_channels

    @property
    def short_description(self) -> str:
        name = super().short_description
        name += f" (timesteps={self.num_timesteps})"
        return name

    def sample(self, condition=None, num_samples=1, **kwargs):
        raise NotImplementedError()

    def predict_forward(self, inputs, condition=None, metadata: Any = None, **kwargs):
        channel_dim = 1
        target_method = getattr(self, "sample_loop", self.sample)
        target_args = inspect.signature(target_method).parameters.keys()
        if inputs is not None and condition is not None:
            if "static_condition" in target_args:
                kwargs["static_condition"] = condition
                inital_condition = inputs
            else:
                try:
                    inital_condition = torch.cat([inputs, condition], dim=channel_dim)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Could not concatenate the inputs (shape={inputs.shape}) and the condition "
                        f"(shape={condition.shape}) along the channel dimension (dim={channel_dim})"
                        f" due to the following error:\n{e}"
                    )
        else:
            inital_condition = inputs

        return self.sample(inital_condition, **kwargs)
