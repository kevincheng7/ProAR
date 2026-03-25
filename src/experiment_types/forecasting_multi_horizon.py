from __future__ import annotations

import math
from abc import ABC
from collections import defaultdict
from typing import Any, List, Optional, Sequence

import numpy as np

from src.experiment_types._base_experiment import BaseExperiment


class AbstractMultiHorizonForecastingExperiment(BaseExperiment, ABC):
    """Base class for multi-horizon forecasting experiments.

    Provides autoregressive rollout infrastructure, prediction timestep management,
    and evaluation scaffolding. Concrete subclasses must override `_evaluation_step`
    and `get_preds_at_t_for_batch`.
    """

    def __init__(
        self, autoregressive_steps: int = 0, prediction_timesteps: Optional[Sequence[float]] = None, **kwargs
    ):
        assert autoregressive_steps >= 0, f"Autoregressive steps must be >= 0, but is {autoregressive_steps}"
        self.stack_window_to_channel_dim = True
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["model"])
        self.USE_TIME_AS_EXTRA_INPUT = False
        self._test_metrics_aggregate = defaultdict(list)
        self._prediction_timesteps = prediction_timesteps
        self.hparams.pop("prediction_timesteps", None)
        if prediction_timesteps is not None:
            self.log_text.info(f" Using prediction timesteps {prediction_timesteps}")

        if self.num_autoregressive_steps > 0:
            ar_steps = self.num_autoregressive_steps
            max_horizon = self.true_horizon * (ar_steps + 1)
            self.log_text.info(f" Inference with {ar_steps} autoregressive steps for max. horizon={max_horizon}.")

    @property
    def horizon_range(self) -> List[int]:
        return list(np.arange(1, self.horizon + 1))

    @property
    def true_horizon(self) -> int:
        return self.horizon

    @property
    def horizon_name(self) -> str:
        return f"{self.true_horizon}h"

    @property
    def prediction_timesteps(self) -> List[float]:
        """By default, predict at every integer step in the horizon range."""
        return self._prediction_timesteps or self.horizon_range

    @prediction_timesteps.setter
    def prediction_timesteps(self, value: List[float]):
        assert (
            max(value) <= self.horizon_range[-1]
        ), f"Prediction range {value} exceeds horizon range {self.horizon_range}"
        self._prediction_timesteps = value

    @property
    def num_autoregressive_steps(self) -> int:
        n_autoregressive_steps = self.hparams.autoregressive_steps
        if n_autoregressive_steps == 0 and self.prediction_horizon is not None:
            n_autoregressive_steps = max(1, math.ceil(self.prediction_horizon / self.true_horizon)) - 1
        return n_autoregressive_steps

    @property
    def short_description(self) -> str:
        name = super().short_description
        name += f" (h={self.horizon_name})"
        return name

    def actual_num_input_channels(self, num_input_channels: int) -> int:
        if self.is_diffusion_model:
            d_class = self.diffusion_config._target_.lower()
            if "dyffusion" not in d_class:
                return self.actual_num_output_channels(self.dims["output"])
            else:
                return num_input_channels
        if self.stack_window_to_channel_dim:
            return num_input_channels * self.window
        return num_input_channels

    @property
    def prediction_horizon(self) -> int:
        if hasattr(self.datamodule_config, "prediction_horizon") and self.datamodule_config.prediction_horizon:
            return self.datamodule_config.prediction_horizon
        return self.horizon * (self.hparams.autoregressive_steps + 1)

    @property
    def default_monitor_metric(self) -> str:
        return f"val/{self.horizon_name}_avg/mse"

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None, **kwargs):
        return super().test_step(batch, batch_idx, dataloader_idx, return_outputs=True)

    def on_test_epoch_end(self, **kwargs) -> None:
        return super().on_test_epoch_end(**kwargs)
